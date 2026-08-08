#!/usr/bin/env python3
"""Replay-Gap pilot: base rollouts + counterfactual branches on SWE-bench.

Usage (on the GPU VM, after `bash scripts/serve_models.sh`):

    python scripts/run_pilot.py --config configs/pilot.yaml --output runs/pilot

Resumable: rollouts whose trajectory file already exists are skipped.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import random
import sys
import threading
import traceback
from pathlib import Path

os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from datasets import load_dataset

from minisweagent.config import builtin_config_dir
from minisweagent.run.benchmarks.swebench import DATASET_MAPPING
from minisweagent.utils.serialize import recursive_merge

from replay_gap import branching
from replay_gap.pool import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pilot")
# litellm logs every single completion call at INFO; that's one line per agent
# step and it buries the orchestrator's progress lines.
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
_PREDS_LOCK = threading.Lock()

OK_EXIT_STATUSES = {"Submitted"}


def load_config(path: Path) -> dict:
    pilot = yaml.safe_load(path.read_text())
    base = yaml.safe_load((builtin_config_dir / "benchmarks" / "swebench.yaml").read_text())
    merged = recursive_merge(base, pilot.get("overrides", {}))
    if append := pilot.get("instance_template_append"):
        # Extend the builtin task prompt without duplicating it in configs.
        merged["agent"]["instance_template"] += "\n" + append
    merged["pilot"] = pilot
    return merged


def make_pool_model(alias: str, config: dict):
    """Model for a pool alias, with the benchmark's observation templates attached."""
    spec = dict(config["pilot"]["pool"][alias])
    for key in ("observation_template", "format_error_template"):
        if key in config.get("model", {}):
            spec.setdefault(key, config["model"][key])
    return build_model(spec)


def select_instances(config: dict) -> list[dict]:
    p = config["pilot"]
    dataset_name = DATASET_MAPPING.get(p.get("dataset", "verified"), p.get("dataset"))
    instances = list(load_dataset(dataset_name, split=p.get("split", "test")))
    if ids := p.get("instance_ids"):
        by_id = {i["instance_id"]: i for i in instances}
        return [by_id[i] for i in ids]
    if difficulty := p.get("difficulty"):
        # SWE-bench Verified annotates difficulty, e.g. "<15 min fix".
        instances = [i for i in instances if i.get("difficulty") == difficulty]
        if not instances:
            raise ValueError(f"No instances with difficulty == {difficulty!r}")
    rng = random.Random(p.get("seed", 0))
    rng.shuffle(instances)
    return instances[: p.get("n_instances", 20)]


def update_preds(preds_path: Path, instance_id: str, model_name: str, patch: str) -> None:
    with _PREDS_LOCK:
        preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}
        preds[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": patch,
        }
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        preds_path.write_text(json.dumps(preds, indent=2))


def load_traj(path: Path) -> dict | None:
    """Load a trajectory only if it represents a COMPLETED rollout.

    Files are written incrementally during a rollout, so a parseable file is
    not enough — an interrupted run leaves valid JSON with an empty
    exit_status, which must be re-run, not reused.
    """
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        if data.get("info", {}).get("exit_status"):
            return data
    return None


def process_instance(instance: dict, config: dict, output_dir: Path) -> None:
    p = config["pilot"]
    iid = instance["instance_id"]
    inst_dir = output_dir / iid
    inst_dir.mkdir(parents=True, exist_ok=True)
    base_alias = p["base_model"]

    # --- base rollout ---
    base_path = inst_dir / "base.traj.json"
    base_traj = load_traj(base_path)
    if base_traj is None:
        logger.info(f"[{iid}] running base rollout ({base_alias})")
        base_traj = branching.run_base(instance, make_pool_model(base_alias, config), config, base_path)
    else:
        logger.info(f"[{iid}] base rollout found, skipping")
    update_preds(
        output_dir / "preds" / "base" / "preds.json", iid, base_alias, base_traj["info"].get("submission", "")
    )

    if base_traj["info"].get("exit_status") not in OK_EXIT_STATUSES:
        logger.warning(f"[{iid}] base exit_status={base_traj['info'].get('exit_status')}; branching anyway")

    n_steps = branching.n_assistant_steps(base_traj["messages"])
    fork_steps = branching.resolve_fork_steps(p.get("fork_steps", [0.3, 0.7]), n_steps)
    if not fork_steps:
        logger.warning(f"[{iid}] base trajectory too short ({n_steps} steps); no branches")
        return

    # --- branches ---
    for k in fork_steps:
        for alias in p["branch_models"]:
            arm = f"{alias}@k{k}"
            arm_slug = arm.replace("@", "_")
            traj_path = inst_dir / f"branch_{arm_slug}.traj.json"
            branch_traj = load_traj(traj_path)
            if branch_traj is None:
                logger.info(f"[{iid}] branching at step {k}/{n_steps} -> {alias}")
                branch_traj = branching.run_branch(
                    instance, base_traj["messages"], k, make_pool_model(alias, config), config, traj_path, arm
                )
            update_preds(
                output_dir / "preds" / arm_slug / "preds.json",
                iid,
                alias,
                branch_traj["info"].get("submission", ""),
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    ap.add_argument("--output", type=Path, default=Path("runs/pilot"))
    ap.add_argument("--workers", type=int, default=None, help="parallel instances (default from config)")
    args = ap.parse_args()

    config = load_config(args.config)
    instances = select_instances(config)
    workers = args.workers or config["pilot"].get("workers", 4)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config_resolved.yaml").write_text(yaml.safe_dump(config))
    logger.info(f"{len(instances)} instances, {workers} workers -> {args.output}")

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_instance, inst, config, args.output): inst["instance_id"] for inst in instances}
        for fut in concurrent.futures.as_completed(futures):
            iid = futures[fut]
            try:
                fut.result()
                logger.info(f"[{iid}] done")
            except Exception as e:
                failures.append(iid)
                logger.error(f"[{iid}] FAILED: {e}\n{traceback.format_exc()}")

    logger.info(f"Finished. {len(failures)} failures: {failures}")
    logger.info(f"Next: python scripts/analyze.py {args.output}")


if __name__ == "__main__":
    main()
