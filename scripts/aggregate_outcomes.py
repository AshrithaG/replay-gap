#!/usr/bin/env python3
"""Join SWE-bench harness reports with branch metadata into the outcome table.

    python scripts/aggregate_outcomes.py runs/pilot30 . replaygap
    python scripts/aggregate_outcomes.py runs/pilot30_rev . rgrev

Args: <run_dir> (trajectories), <report_dir> (where the harness wrote
<model>.<prefix>_<arm>.json files; '.' if run from the repo root), and the
run-id <prefix> used with run_evaluation (default: replaygap).

Headline metric: the OUTCOME FLIP RATE — how often forking a trajectory
changes whether the instance is resolved. Compare same-model control flips
(noise floor) against model-swap flips (the replay gap at outcome level).
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from replay_gap.branching import n_assistant_steps

BRANCH_FILE_RE = re.compile(r"branch_(?P<alias>.+)_k(?P<k>\d+)\.traj\.json")


def load_resolved(report_dir: Path, prefix: str) -> dict[str, dict]:
    """{arm_slug: {'resolved': set(ids), 'submitted': set(ids)}} from harness reports."""
    report_re = re.compile(rf".+\.{re.escape(prefix)}_(?P<arm>.+)\.json")
    out = {}
    for path in report_dir.glob(f"*.{prefix}_*.json"):
        m = report_re.fullmatch(path.name)
        if not m:
            continue
        data = json.loads(path.read_text())
        out[m["arm"]] = {
            "resolved": set(data.get("resolved_ids", [])),
            "submitted": set(data.get("submitted_ids", data.get("completed_ids", []))),
        }
    return out


def fork_fraction(k: int, n_steps: int) -> str:
    """Bucket an absolute fork step into 'early'/'late' by trajectory position."""
    return "early" if n_steps and k / n_steps <= 0.5 else "late"


def main() -> None:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/pilot30")
    report_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    prefix = sys.argv[3] if len(sys.argv) > 3 else "replaygap"
    reports = load_resolved(report_dir, prefix)
    if not reports:
        sys.exit(f"No *.{prefix}_*.json reports found in {report_dir}")

    base_resolved = reports.get("base", {}).get("resolved", set())
    base_n = 0
    groups = defaultdict(lambda: {"n": 0, "resolved": 0, "flips": 0, "upgrades": 0, "downgrades": 0})

    for inst_dir in sorted(run_dir.iterdir()):
        base_path = inst_dir / "base.traj.json"
        if not inst_dir.is_dir() or not base_path.exists():
            continue
        iid = inst_dir.name
        base_n += 1
        n_steps = n_assistant_steps(json.loads(base_path.read_text())["messages"])
        base_ok = iid in base_resolved

        for bp in inst_dir.glob("branch_*.traj.json"):
            m = BRANCH_FILE_RE.fullmatch(bp.name)
            if not m:
                continue
            arm_slug = f"{m['alias']}_k{m['k']}"
            if arm_slug not in reports:
                continue  # this arm was never scored
            branch_ok = iid in reports[arm_slug]["resolved"]
            g = groups[(m["alias"], fork_fraction(int(m["k"]), n_steps))]
            g["n"] += 1
            g["resolved"] += branch_ok
            if branch_ok != base_ok:
                g["flips"] += 1
                g["upgrades" if branch_ok else "downgrades"] += 1

    print(f"\nbase: {len(base_resolved)}/{base_n} resolved ({len(base_resolved) / max(base_n, 1):.1%})\n")
    print(f"{'arm':<16}{'n':>5}{'resolved':>10}{'flip-rate':>11}{'up':>5}{'down':>6}")
    print("-" * 53)
    for (alias, frac), g in sorted(groups.items()):
        print(
            f"{alias + '@' + frac:<16}{g['n']:>5}"
            f"{g['resolved'] / g['n']:>10.1%}"
            f"{g['flips'] / g['n']:>11.1%}"
            f"{g['upgrades']:>5}{g['downgrades']:>6}"
        )
    print(
        "\nRead: 'flip-rate' = branch outcome differs from base outcome."
        "\nThe arm whose alias matches the base model is the control (noise floor);"
        "\nthe other alias is the model swap. 'down' = resolved by base but not by branch."
        "\nSwap flip-rate >> control flip-rate = replay evaluation mispredicts outcomes."
    )


if __name__ == "__main__":
    main()
