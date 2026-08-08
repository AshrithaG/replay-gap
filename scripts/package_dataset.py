#!/usr/bin/env python3
"""Package the branched-trajectory dataset for public release.

Run on the VM (where runs/ lives):

    python scripts/package_dataset.py            # writes dataset/*.jsonl

Produces one JSONL row per rollout with the full message history, the
branching metadata (fork step, arm, replay fidelity), token counts, the
submitted patch, and the SWE-bench outcome where scored. Also writes a
compact rollouts_index.jsonl without message bodies for quick browsing.
"""

import json
import re
from pathlib import Path

BRANCH_RE = re.compile(r"branch_(?P<alias>.+)_k(?P<k>\d+)\.traj\.json")

# run -> (swap direction, harness run-id prefix used when scoring)
RUNS = {
    "pilot30": ("up", "replaygap"),
    "pilot30_rev": ("down", "rgrev"),
    "easy": ("up", "rgeasy"),
    "easy_rev": ("down", "rgeasyrev"),
    "nudge": ("up", "rgn"),
    "nudge_rev": ("down", "rgnrev"),
}
REPORT_RE = re.compile(r".+\.(?P<runid>.+)\.json")


def load_resolved(root: Path) -> dict:
    """{run_id: set(resolved instance ids)} from SWE-bench harness reports."""
    out = {}
    for path in root.glob("*.json"):
        m = REPORT_RE.fullmatch(path.name)
        if not m:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "resolved_ids" in data:
            out[m["runid"]] = set(data["resolved_ids"])
    return out


def tokens(messages):
    p = c = 0
    for m in messages:
        u = (m.get("extra", {}).get("response") or {}).get("usage") or {}
        p += u.get("prompt_tokens") or 0
        c += u.get("completion_tokens") or 0
    return p, c


def main() -> None:
    repo = Path(".")
    resolved = load_resolved(repo)
    out_dir = repo / "dataset"
    out_dir.mkdir(exist_ok=True)
    index_f = (out_dir / "rollouts_index.jsonl").open("w")
    n_total = 0

    for run, (direction, prefix) in RUNS.items():
        run_dir = repo / "runs" / run
        if not run_dir.is_dir():
            print(f"skip {run} (missing)")
            continue
        with (out_dir / f"{run}.jsonl").open("w") as f:
            for inst_dir in sorted(run_dir.iterdir()):
                if not inst_dir.is_dir():
                    continue
                for traj_path in sorted(inst_dir.glob("*.traj.json")):
                    data = json.loads(traj_path.read_text())
                    info = data.get("info", {})
                    rg = data.get("replay_gap", {})
                    messages = data.get("messages", [])
                    m = BRANCH_RE.fullmatch(traj_path.name)
                    arm_id = "base" if traj_path.name == "base.traj.json" else f"{m['alias']}_k{m['k']}"
                    p, c = tokens(messages)
                    row = {
                        "run": run,
                        "direction": direction,
                        "instance_id": inst_dir.name,
                        "arm": "base" if m is None else "branch",
                        "arm_id": arm_id,
                        "model_alias": None if m is None else m["alias"],
                        "fork_step": None if m is None else int(m["k"]),
                        "exit_status": info.get("exit_status", ""),
                        "patch": info.get("submission", ""),
                        "n_steps": sum(1 for x in messages if x.get("role") == "assistant"),
                        "prompt_tokens": p,
                        "completion_tokens": c,
                        "replay_mismatches": rg.get("replay_mismatches"),
                        "replay_fidelity": rg.get("replay_fidelity"),
                        "wall_time_s": rg.get("wall_time_s"),
                        "resolved": inst_dir.name in resolved.get(f"{prefix}_{arm_id}", set())
                        if f"{prefix}_{arm_id}" in resolved
                        else None,
                        "messages": messages,
                    }
                    f.write(json.dumps(row) + "\n")
                    index_row = {k: v for k, v in row.items() if k not in ("messages", "replay_fidelity", "patch")}
                    index_row["patch_len"] = len(row["patch"] or "")
                    index_f.write(json.dumps(index_row) + "\n")
                    n_total += 1
        print(f"{run}: written")

    index_f.close()
    print(f"\n{n_total} rollouts -> dataset/  ({sum(f.stat().st_size for f in out_dir.glob('*.jsonl')) / 1e6:.0f} MB)")
    print("Next: gzip dataset/*.jsonl and upload to HuggingFace.")


if __name__ == "__main__":
    main()
