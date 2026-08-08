#!/usr/bin/env python3
"""Quantify prefix-replay fidelity across all branch rollouts (reviewer ask #7).

Every branch logs, per replayed prefix action, whether the fresh container's
return code matched the one recorded in the base trajectory. This aggregates
those logs into the auditability table the reviewers asked for.

Run on the VM (where the trajectories live):

    python scripts/replay_fidelity.py            # writes runs/replay_fidelity.json
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

RUNS = ["pilot30", "pilot30_rev", "easy", "easy_rev", "nudge", "nudge_rev"]


def main() -> None:
    root = Path("runs")
    per_run = defaultdict(list)

    for run in RUNS:
        run_dir = root / run
        if not run_dir.is_dir():
            print(f"skip {run} (missing)")
            continue
        for traj_path in run_dir.glob("*/branch_*.traj.json"):
            rg = json.loads(traj_path.read_text()).get("replay_gap", {})
            fid = rg.get("replay_fidelity") or []
            comparable = [f for f in fid if f.get("match") is not None]
            if not comparable:
                continue
            matches = sum(1 for f in comparable if f["match"])
            per_run[run].append(
                {
                    "instance_id": traj_path.parent.name,
                    "fork_step": rg.get("fork_step"),
                    "n_actions": len(comparable),
                    "n_mismatch": len(comparable) - matches,
                    "action_match_rate": matches / len(comparable),
                    "perfect": matches == len(comparable),
                }
            )

    rows = [r for rs in per_run.values() for r in rs]
    if not rows:
        raise SystemExit("no replay-fidelity records found")

    out = {"per_run": dict(per_run)}
    (root / "replay_fidelity.json").write_text(json.dumps(out))

    print(f"\n{'run':<14}{'branches':>9}{'actions':>9}{'action match':>14}{'branches exact':>16}")
    print("-" * 62)
    for run in RUNS:
        rs = per_run.get(run)
        if not rs:
            continue
        n_act = sum(r["n_actions"] for r in rs)
        n_mis = sum(r["n_mismatch"] for r in rs)
        print(
            f"{run:<14}{len(rs):>9}{n_act:>9}"
            f"{(n_act - n_mis) / n_act:>13.2%}"
            f"{mean(1.0 * r['perfect'] for r in rs):>16.1%}"
        )
    n_act = sum(r["n_actions"] for r in rows)
    n_mis = sum(r["n_mismatch"] for r in rows)
    print("-" * 62)
    print(
        f"{'ALL':<14}{len(rows):>9}{n_act:>9}{(n_act - n_mis) / n_act:>13.2%}"
        f"{mean(1.0 * r['perfect'] for r in rows):>16.1%}"
    )
    print(f"\nMean per-branch action match rate: {mean(r['action_match_rate'] for r in rows):.2%}")
    print(f"Branches with >=1 mismatch: {sum(1 for r in rows if not r['perfect'])}/{len(rows)}")
    print("Wrote runs/replay_fidelity.json")


if __name__ == "__main__":
    main()
