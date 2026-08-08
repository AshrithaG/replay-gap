#!/usr/bin/env python3
"""Mini router-misprediction experiment, zero new rollouts.

A RouterBench-style replay evaluator predicts a switch policy's result using
the target model's LOGGED standalone run on the same instance (tiers are
seed-matched, so we have it). We compare those predictions against the
branched ground truth we already executed live.

    python scripts/replay_stitch.py
"""

import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text())
REPORTS = ROOT / "results" / "reports"

# tier -> (forward run, reverse run, forward report prefix, reverse report prefix)
TIERS = {
    "full": ("pilot30", "pilot30_rev", "replaygap", "rgrev"),
    "easy": ("easy", "easy_rev", "rgeasy", "rgeasyrev"),
    "nudge": ("nudge", "nudge_rev", "rgn", "rgnrev"),
}

REPORT_RE = re.compile(r".+\.(?P<runid>.+)\.json")


def load_resolved():
    """{run_id: set(resolved instance ids)} from all harness reports."""
    out = {}
    for path in REPORTS.glob("*.json"):
        m = REPORT_RE.fullmatch(path.name)
        if m:
            out[m["runid"]] = set(json.loads(path.read_text()).get("resolved_ids", []))
    return out


RESOLVED = load_resolved()


def resolved(prefix, arm, iid):
    return iid in RESOLVED.get(f"{prefix}_{arm}", set())


def patch_files(patch):
    return set(re.findall(r"^diff --git a/(\S+)", patch or "", re.M))


def patch_sim(a, b):
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return None  # trivially equal; exclude
    return difflib.SequenceMatcher(None, a, b).ratio()


def index_rows():
    idx = {}
    for r in SUMMARY:
        if r["arm"] == "base":
            idx[(r["run"], r["instance_id"], "base", None)] = r
        else:
            idx[(r["run"], r["instance_id"], r["alias"], r["fork_step"])] = r
    return idx


IDX = index_rows()


def branches(run, alias):
    rows = [r for r in SUMMARY if r["run"] == run and r["arm"] == "branch" and r["alias"] == alias]
    # bucket early/late by rank of fork step within (instance)
    by_inst = defaultdict(list)
    for r in rows:
        by_inst[r["instance_id"]].append(r)
    for rs in by_inst.values():
        rs.sort(key=lambda r: r["fork_step"])
        for i, r in enumerate(rs):
            r["bucket"] = "early" if (len(rs) == 1 or i == 0) else "late"
    return rows


def main():
    # Policies: swap-up (branch alias 'large' in forward runs, target logged run = reverse base)
    #           swap-down (branch alias 'small' in reverse runs, target logged run = forward base)
    results = defaultdict(list)
    ceiling = defaultdict(list)  # in-model reproducibility: target base vs target's own control branch

    for tier, (fwd, rev, pfx_f, pfx_r) in TIERS.items():
        specs = [
            ("swap-up", fwd, "large", rev, pfx_f, pfx_r),
            ("swap-down", rev, "small", fwd, pfx_r, pfx_f),
        ]
        for policy, run, alias, target_run, pfx_live, pfx_target in specs:
            for br in branches(run, alias):
                iid = br["instance_id"]
                target_base = IDX.get((target_run, iid, "base", None))
                if target_base is None:
                    continue  # instance not shared across the pair
                actual_res = resolved(pfx_live, f"{alias}_k{br['fork_step']}", iid)
                pred_res = resolved(pfx_target, "base", iid)
                sim = patch_sim(target_base["patch"], br["patch"])
                results[(policy, br["bucket"])].append(
                    {"iid": iid, "tier": tier, "pred_res": pred_res, "actual_res": actual_res, "patch_sim": sim}
                )
        # ceiling: how well does the target model's logged run predict ITSELF
        # (its own same-model control branch) — the best any log-based
        # predictor could do.
        for target_run, alias, pfx in [(rev, "large", pfx_r), (fwd, "small", pfx_f)]:
            for br in branches(target_run, alias):
                tb = IDX.get((target_run, br["instance_id"], "base", None))
                if tb is None:
                    continue
                s = patch_sim(tb["patch"], br["patch"])
                if s is not None:
                    ceiling[alias].append(s)

    print("Replay-stitch prediction vs branched ground truth")
    print(f"{'policy':<12}{'fork':<7}{'n':>4}{'outcome-agree':>14}{'miss-succ':>10}{'false-succ':>11}{'patch-sim':>10}{'n-sim':>6}")
    print("-" * 74)
    total_actual_succ = total_missed = 0
    for (policy, bucket), rs in sorted(results.items()):
        agree = mean(1.0 * (r["pred_res"] == r["actual_res"]) for r in rs)
        actual_succ = [r for r in rs if r["actual_res"]]
        missed = [r for r in actual_succ if not r["pred_res"]]
        false_succ = [r for r in rs if r["pred_res"] and not r["actual_res"]]
        sims = [r["patch_sim"] for r in rs if r["patch_sim"] is not None]
        total_actual_succ += len(actual_succ)
        total_missed += len(missed)
        print(
            f"{policy:<12}{bucket:<7}{len(rs):>4}{agree:>14.1%}"
            f"{f'{len(missed)}/{len(actual_succ)}':>10}{len(false_succ):>11}"
            f"{mean(sims) if sims else float('nan'):>10.3f}{len(sims):>6}"
        )
    print(f"\nActual switching successes missed by replay: {total_missed}/{total_actual_succ}")
    for alias, sims in ceiling.items():
        print(f"in-model prediction ceiling ({alias} base vs own control branch): {mean(sims):.3f} (n={len(sims)})")


if __name__ == "__main__":
    main()
