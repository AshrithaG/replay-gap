# The Replay Gap

**Static evaluation of model switching in LLM agents scores the wrong world.**

Accepted at the [Efficient Reasoning Workshop @ COLM 2026](https://wdlctc.github.io/efficient-reasoning-2026/) ·
[Paper / reviews on OpenReview](https://openreview.net/forum?id=8gqqiNrzyA)

Every LLM-routing benchmark scores routers by replaying pre-collected model
outputs. In a multi-step agent that is unsound: swap the model at step *k* and
the rest of the trajectory diverges, so replay grades decisions against states
that never occur. This repo contains the experiment that measures how wrong
that is — **branching rollouts**: run a base trajectory on SWE-bench Verified,
fork it at a chosen step into a fresh container (replaying the prefix actions),
continue with a different model, and compare against a **same-model control
fork** that isolates sampling and environment-replay noise.

## Headline results

From six seed-matched run pairs (~900 containerized rollouts, 717 scored branch
pairs) with Qwen3-4B and Qwen3-14B served by vLLM on a single 24GB GPU:

| | control@early | swap@early | control@late | swap@late |
|---|---|---|---|---|
| **up** (4B base → 14B) | 0.674 | **0.941** | 0.489 | 0.752 |
| **down** (14B base → 4B) | 0.232 | **0.895** | 0.158 | 0.611 |

*Normalized post-fork action edit distance. Paired swap−control deltas: +0.25 to
+0.66, bootstrap 95% CIs exclude zero under Bonferroni correction.*

- **74–77% of early swaps diverge at the very first post-fork action** (controls: 6–35%), leaving only ~3% of replayed post-fork states valid.
- **All 5 observed outcome flips occur in swap arms; zero across 359 same-model control forks.**
- A log-stitching replay evaluator **mispredicts every success-relevant outcome** and produces patches with 0.00–0.11 similarity to what the switch actually produced — worse than a constant-failure predictor.
- **Temperature-0 determinism is configuration-dependent**: FP8-served controls diverge on 90–96% of forks while AWQ-served controls stay near-identical.
- Under a tight step budget, the *stronger* model more often exhausts it without submitting (24/30 vs 17/30).

## Layout

```
configs/                    experiment configs (difficulty tier × swap direction)
src/replay_gap/branching.py BranchableAgent + prefix replay (core logic)
src/replay_gap/metrics.py   divergence metrics
src/replay_gap/pool.py      model factory for vLLM endpoints
scripts/run_pilot.py        orchestrator (resumable)
scripts/analyze.py          divergence tables
scripts/aggregate_outcomes.py  outcome-flip tables (needs SWE-bench harness reports)
scripts/replay_stitch.py    replay-prediction vs branched ground truth (§4.5)
scripts/replay_fidelity.py  prefix-replay fidelity audit
scripts/make_figures.py     paper figures
scripts/smoke_test.py       end-to-end test of the machinery (no GPU, ~5s)
paper/latex/                the paper source
```

## Reproducing

Prereqs: NVIDIA GPU (the default pool is sized for a single 24GB card), docker
(x86_64 — SWE-bench images are amd64), Python 3.10+, ~150GB free disk.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install vllm swebench

python scripts/smoke_test.py            # validates the branching machinery, no GPU needed
bash scripts/serve_models.sh            # vLLM servers for the model pool
python scripts/run_pilot.py --config configs/pilot.yaml --output runs/pilot   # resumable
python scripts/analyze.py runs/pilot    # divergence table
```

Then score each arm with the official SWE-bench harness and aggregate:

```bash
for arm in runs/pilot/preds/*/; do
  name=$(basename "$arm")
  python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-Bench_Verified \
    --predictions_path "$arm/preds.json" --run_id "rg_$name" --max_workers 4
done
python scripts/aggregate_outcomes.py runs/pilot . rg
```

## Dataset

The full branched-trajectory dataset (~900 rollouts: base and fork trajectories,
per-step actions and observations, token counts, exit statuses, patches, and
outcomes) is released separately — see the OpenReview page for the current link.

## Citation

```bibtex
@inproceedings{gonuguntla2026replaygap,
  title     = {The Replay Gap: Static Evaluation of Model Switching in {LLM} Agents Scores the Wrong World},
  author    = {Gonuguntla, Ashritha},
  booktitle = {Efficient Reasoning Workshop at COLM},
  year      = {2026},
  url       = {https://openreview.net/forum?id=8gqqiNrzyA}
}
```

## Limitations

A deliberately controlled pilot: one scaffold (mini-SWE-agent), one benchmark
(SWE-bench Verified), one model family in two quantizations, n=30 instances per
run pair, two fork positions, and a 24GB serving budget that keeps absolute
resolution rates low (0–3%). The action-level results do not depend on task
success, but outcome-level claims rest on five events and are reported as such.
See §6 of the paper.

## License

Code released under the MIT License (see `LICENSE`). The paper is CC BY 4.0.
