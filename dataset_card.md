---
license: cc-by-4.0
task_categories:
- text-generation
tags:
- llm-agents
- llm-routing
- swe-bench
- counterfactual-evaluation
- agent-trajectories
size_categories:
- 1K<n<10K
---

# The Replay Gap: Branched Agent Trajectories

Counterfactual ("branched") agent rollouts for studying **per-step model
switching** in LLM agents, from the paper *The Replay Gap: Static Evaluation of
Model Switching in LLM Agents Scores the Wrong World* (Efficient Reasoning
Workshop @ COLM 2026).

Routing benchmarks score routers by **replaying** logged model outputs. In a
multi-step agent that is unsound: swap the model at step *k* and the rest of
the trajectory diverges. This dataset contains the ground truth replay cannot
provide — trajectories actually forked mid-run and continued live with a
different model, each paired with a **same-model control fork** that isolates
sampling and environment-replay noise.

## What's in it

~900 containerized SWE-bench Verified rollouts across six seed-matched run
pairs (difficulty tier × swap direction):

| run | base model | branch arms | tier |
|---|---|---|---|
| `pilot30` / `pilot30_rev` | 4B / 14B | both models | full difficulty |
| `easy` / `easy_rev` | 4B / 14B | both models | "<15 min fix" |
| `nudge` / `nudge_rev` | 4B / 14B | both models | "<15 min fix", budget-nudged prompt |

Each instance contributes one **base** trajectory plus **branches** forked at
30% and 70% of its length, each continued by both the same model (control) and
the other model (swap).

## Fields

| field | description |
|---|---|
| `run`, `direction` | run pair and swap direction (`up` = 4B base, `down` = 14B base) |
| `instance_id` | SWE-bench Verified instance |
| `arm`, `arm_id`, `model_alias`, `fork_step` | base vs branch; which model continued; the assistant turn forked at |
| `messages` | full message history (system, user, assistant turns with parsed actions, observations) |
| `exit_status` | `Submitted`, `LimitsExceeded`, `ContextWindowExceededError`, `RepeatedFormatError` |
| `patch` | submitted git patch (may be empty) |
| `resolved` | official SWE-bench harness outcome, `null` if that arm was not scored |
| `replay_fidelity`, `replay_mismatches` | per-action return-code agreement when rebuilding the pre-fork environment |
| `n_steps`, `prompt_tokens`, `completion_tokens`, `wall_time_s` | cost and length |

`rollouts_index.jsonl` is the same data without message bodies, for quick
filtering.

## Generation setup

mini-SWE-agent (bash-only ReAct loop) on official SWE-bench Verified Docker
images; 50-step budget, 28k context. Models served by vLLM at temperature 0 on
a single 24GB GPU: Qwen3-4B-Instruct-2507-FP8 and Qwen3-14B-AWQ (thinking
disabled). Prefix replay fidelity: 99.99% return-code agreement across 11,702
replayed actions; 707/708 branches reconstructed exactly.

## Caveats

Absolute task-resolution rates are low (0–3%) because of the constrained
serving budget — the action-level signal is robust, but outcome-level analyses
rest on five flip events. The two models differ in both scale and quantization
stack (FP8 vs AWQ), so up/down comparisons are between deployment
configurations, not pure model scales. One scaffold, one benchmark family.

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

Code: https://github.com/AshrithaG/replay-gap
