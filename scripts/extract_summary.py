#!/usr/bin/env python3
"""Distill every trajectory across all runs into one compact JSON for the
replay-stitch analysis (runs on the VM; output is a few MB).

    python scripts/extract_summary.py            # writes runs/summary.json
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

RUNS = ["pilot30", "pilot30_rev", "easy", "easy_rev", "nudge", "nudge_rev"]
BRANCH_RE = re.compile(r"branch_(?P<alias>.+)_k(?P<k>\d+)\.traj\.json")


def tokens(messages):
    p = c = 0
    for m in messages:
        u = (m.get("extra", {}).get("response") or {}).get("usage") or {}
        p += u.get("prompt_tokens") or 0
        c += u.get("completion_tokens") or 0
    return p, c


def summarize(path, run, arm, alias, fork_step):
    data = json.loads(path.read_text())
    info = data.get("info", {})
    msgs = data.get("messages", [])
    p, c = tokens(msgs)
    return {
        "run": run,
        "instance_id": path.parent.name,
        "arm": arm,
        "alias": alias,
        "fork_step": fork_step,
        "exit_status": info.get("exit_status", ""),
        "patch": info.get("submission", ""),
        "n_steps": sum(1 for m in msgs if m.get("role") == "assistant"),
        "prompt_tokens": p,
        "completion_tokens": c,
    }


def main():
    root = Path("runs")
    out = []
    for run in RUNS:
        run_dir = root / run
        if not run_dir.is_dir():
            print(f"skip {run} (missing)")
            continue
        for inst_dir in sorted(run_dir.iterdir()):
            base = inst_dir / "base.traj.json"
            if not inst_dir.is_dir() or not base.exists():
                continue
            out.append(summarize(base, run, "base", None, None))
            for bp in inst_dir.glob("branch_*.traj.json"):
                m = BRANCH_RE.fullmatch(bp.name)
                if m:
                    out.append(summarize(bp, run, "branch", m["alias"], int(m["k"])))
    (root / "summary.json").write_text(json.dumps(out))
    print(f"wrote runs/summary.json: {len(out)} trajectories")


if __name__ == "__main__":
    main()
