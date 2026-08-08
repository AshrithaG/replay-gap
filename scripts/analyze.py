#!/usr/bin/env python3
"""Analyze a pilot run: divergence of branches from their base trajectories.

    python scripts/analyze.py runs/pilot

The number that matters: cross-model divergence vs same-model (control)
divergence. If swapping the model at step k diverges trajectories far more
than re-running the same model does, static replay evaluation of agentic
routers is measuring the wrong thing — and the paper is on.
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from replay_gap import metrics
from replay_gap.branching import n_assistant_steps

BRANCH_RE = re.compile(r"branch_(?P<alias>.+)_k(?P<k>\d+)\.traj\.json")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/pilot")
    rows = []
    for inst_dir in sorted(run_dir.iterdir()):
        base_path = inst_dir / "base.traj.json"
        if not inst_dir.is_dir() or not base_path.exists():
            continue
        base = json.loads(base_path.read_text())
        n_steps = n_assistant_steps(base["messages"])
        for bp in sorted(inst_dir.glob("branch_*.traj.json")):
            m = BRANCH_RE.fullmatch(bp.name)
            if not m:
                continue
            branch = json.loads(bp.read_text())
            fork_step = int(m["k"])
            row = metrics.compare(base, branch, fork_step)
            rg = branch.get("replay_gap", {})
            row.update(
                instance_id=inst_dir.name,
                alias=m["alias"],
                fork_step=fork_step,
                fork_bucket="early" if n_steps and fork_step / n_steps <= 0.5 else "late",
                replay_mismatches=rg.get("replay_mismatches"),
            )
            rows.append(row)

    if not rows:
        sys.exit(f"No branch trajectories found under {run_dir}")

    (run_dir / "divergence.json").write_text(json.dumps(rows, indent=2))

    by_alias = defaultdict(list)
    for r in rows:
        by_alias[f"{r['alias']}@{r['fork_bucket']}"].append(r)

    print(
        f"\n{'arm':<20}{'n':>4}{'edit-dist':>11}{'1st-div':>9}{'file-jac':>10}{'patch-sim':>11}{'identical':>11}"
        f"{'n-subm':>8}{'ident|subm':>12}"
    )
    print("-" * 96)
    for alias, rs in sorted(by_alias.items()):
        # Patch metrics are only meaningful when BOTH sides submitted a
        # non-empty patch; empty==empty inflates 'identical' trivially.
        rs_sub = [r for r in rs if r["both_submitted"]]
        print(
            f"{alias:<20}{len(rs):>4}"
            f"{mean([r['normalized_edit_distance'] for r in rs]):>11.3f}"
            f"{mean([r['first_divergent_action'] for r in rs]):>9.1f}"
            f"{mean([r['file_jaccard'] for r in rs]):>10.3f}"
            f"{mean([r['patch_similarity'] for r in rs]):>11.3f}"
            f"{mean([1.0 if r['patch_identical'] else 0.0 for r in rs]):>11.3f}"
            f"{len(rs_sub):>8}"
            f"{mean([1.0 if r['patch_identical'] else 0.0 for r in rs_sub]):>12.3f}"
        )
    print(
        "\nRead: the base-model arm is the control (sampling/replay noise floor)."
        "\nCross-model arms diverging well above the control = the replay gap is real."
        f"\nPer-pair details: {run_dir / 'divergence.json'}"
        "\nFor resolution rates, run the SWE-bench harness on each preds/<arm>/preds.json (see README)."
    )


if __name__ == "__main__":
    main()
