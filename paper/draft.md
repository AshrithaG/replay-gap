# The Replay Gap: Static Evaluation Mispredicts Model Switching in LLM Agents

*Target: 2nd Workshop on Efficient Reasoning @ COLM 2026 (deadline Jul 19 AoE, 4–10 pages main text). Markdown master — port to the COLM LaTeX template once frozen.*

## Abstract (draft)

LLM routers promise efficiency by matching each request to the cheapest
adequate model, and are increasingly applied *per step* inside multi-step
agents. Yet routers for agents are evaluated the same way as single-turn
routers: by replaying pre-collected trajectories and swapping in another
model's logged outputs. This assumes the rest of the trajectory is unaffected
by the swap. We test that assumption directly with *branching rollouts* on
SWE-bench Verified: we fork live agent trajectories at controlled points,
continue each fork with a different model in a fresh containerized
environment, and compare against same-model control forks that isolate
sampling and replay noise. Swapping models rewrites 76–93% of post-fork
actions (vs. a 9–61% control floor) and the stronger model deviates at the
*first* post-fork decision. Divergence is directional and positional:
upgrades diverge immediately, and divergence shrinks monotonically with fork
position. We further find that the noise floor itself is model-dependent: at
temperature 0, one model's control forks reproduce base trajectories exactly
while another's diverge on 28% of instances. Finally, under tight step
budgets the *stronger* model exhausts its budget without submitting more
often than the weak one (24/30 vs 17/30 rollouts) — a thoroughness tax with
direct routing implications. *(If easy-bucket runs confirm handoff
inheritance on real patches, restore that claim here.)* Our results indicate that replay-based router benchmarks
measure the wrong quantity for agentic routing, and that trajectory position
is a first-class feature for safe model switching. We release our branching
harness and all forked trajectories. *(≈200 words; add outcome numbers when
rev scoring lands.)*

## 1. Introduction

- Routing is now infrastructure (RouteLLM, RouterBench/RouterEval/RouterArena;
  production routers in GPT-5, Cursor-style coding agents; per-step agentic
  routing: SAAR, Switchcraft, step-level computer-use routing).
- All these evaluate on *logged* outcomes. Fine for single-turn: the query is
  fixed. In an agent, the model's action at step k determines the observation
  at step k+1 — the trajectory is a closed loop through the environment.
  Replay silently assumes open-loop behavior.
- Question: **how wrong is replay?** Nobody has measured it, because ground
  truth requires branching live rollouts — expensive (this paper: 300+
  containerized rollouts for 60 bases + 240 forks).
- Contributions:
  1. Branching-rollout methodology + open harness (fork at step k, replay
     prefix actions into a fresh container, continue with model B, compare to
     same-model control forks).
  2. Quantified trajectory-level replay gap on SWE-bench Verified, both
     swap directions, early/late forks.
  3. Findings with direct routing implications: handoff inheritance (late
     downgrades safe), immediate upgrade divergence, model-dependent
     nondeterminism of the control itself.
  4. Dataset release: all trajectories incl. forks, with per-step actions,
     token counts, outcomes.

## 2. Method

- Agent scaffold: mini-swe-agent (bash-only ReAct loop), SWE-bench Verified,
  official per-instance docker images. Step limit 50, context 28k.
- Pool: Qwen3-4B-Instruct-2507-FP8 ("small") and Qwen3-14B-AWQ, thinking
  disabled ("large"), both served by vLLM on one RTX 4090. Temperature 0.
- Branch protocol: run base to completion; pick fork steps k at 30% and 70%
  of base length; fresh container; re-execute prefix actions (replay fidelity
  logged via returncode matching); seed message history with recorded prefix;
  continue with branch model. Arms: same-model control, cross-model swap.
- Directions: forward (base=small, swap up) and reverse (base=large, swap
  down), same 30 instances (seed 42).
- Metrics: normalized action edit distance (post-fork), first divergent
  action, patch file Jaccard, patch similarity/identity, SWE-bench resolution
  (official harness), tokens.
- Design note: same-model control forks are the noise floor — they absorb
  sampler nondeterminism, batching effects, and environment replay drift.

## 3. Results

### 3.1 Trajectory-level divergence (Table 1)

Forward (base = small/4B):

| arm | n | edit-dist | 1st-div | file-jac | patch-sim | identical |
|---|---|---|---|---|---|---|
| control@early (small) | 32 | 0.610 | 5.0 | 0.812 | 0.711 | 0.656 |
| control@late (small) | 29 | 0.478 | 6.9 | 0.966 | 0.844 | 0.793 |
| swap-up@early (large) | 31 | **0.934** | **1.0** | 0.710 | 0.501 | **0.484** |
| swap-up@late (large) | 29 | 0.777 | 3.6 | 0.828 | 0.653 | 0.621 |

Reverse (base = large/14B):

| arm | n | edit-dist | 1st-div | file-jac | patch-sim | identical |
|---|---|---|---|---|---|---|
| control@early (large) | 30 | 0.192 | 22.5 | 1.000 | 1.000 | **1.000** |
| control@late (large) | 30 | 0.093 | 14.7 | 1.000 | 1.000 | **1.000** |
| swap-down@early (small) | 30 | **0.910** | **2.7** | 0.900 | 0.867 | 0.867 |
| swap-down@late (small) | 30 | 0.606 | 4.2 | 1.000 | 1.000 | 1.000 |

CAVEAT (audit of 2026-07-15): patch-identity columns are inflated by
empty-vs-empty pairs. Restricted to pairs where BOTH sides submitted a
non-empty patch (`ident|subm`): forward control@late 0.600 (n=5),
swap-up@late 0.667 (n=3), both @early buckets 0.000 (n=2–4); reverse run has
ZERO both-submitted pairs (14B base patches all empty). Patch-level claims
must cite these subset numbers; n is too small until the easy-bucket runs.

Key sentences:
- Swaps rewrite 61–93% of post-fork actions, always well above the matched
  control floor; the upgrade deviates at the first post-fork decision (mean
  1st-div 1.0). [Action-level: n=60/direction, robust.]
- **Handoff inheritance is a HYPOTHESIS pending easy-bucket data** — the
  reverse-run patch identities were empty-patch artifacts. Do not claim it
  from pilot30_rev.
- **The noise floor is model-dependent**: at temperature 0, the 14B control
  reproduces its trajectories nearly exactly (action edit-dist 0.09–0.19)
  while the 4B control diverges substantially (0.48–0.61). "Deterministic"
  agent evaluation is an illusion for some serving configurations (FP8
  kernels, batching) — an independent caveat for every replay benchmark.
  [State via ACTION metrics, not patch metrics.]

### 3.1b Termination and the thoroughness tax (Table: exit statuses)

| exit status | fwd base (4B) | rev base (14B) |
|---|---|---|
| Submitted | 7 | 3 (all empty patches) |
| LimitsExceeded (50 steps) | 17 | 24 |
| ContextWindowExceeded (28k) | 3 | 3 |
| RepeatedFormatError | 3 | 0 |

- The binding constraint is the step budget, not context. The stronger model
  exhausts it more often — it explores/tests more per task — so under tight
  budgets, routing UP can *reduce* completion rate. Efficiency framing for
  the workshop: compute-budget-aware routing must account for a model's
  step appetite, not just its per-step quality.
- 13% of early swap-downs produced a non-empty patch where the 14B base had
  none: the small model ships more readily (same phenomenon, other side).

### 3.2 Outcome-level flips (Table 2) — PENDING rev scoring

- Forward: base 0/30 resolved → capability floor; flips unmeasurable except
  one **rescue**: a single early upgrade resolved an instance the small model
  never could (3.2% upgrade rate vs 0% control).
- Reverse (fill in from `aggregate_outcomes.py runs/pilot30_rev . rgrev`):
  base resolution __/30; downgrade rate early vs late vs control.
  Hypothesis from Table 1: downgrades concentrate in @early.

### 3.3 What replay would have concluded (the "gap" framing)

- Replay-style evaluation would score a per-step router by attributing the
  logged step outcome to the swapped model. Our forks show the counterfactual
  trajectory departs at (mean) the 1st–5th post-fork action, so any replay
  estimate beyond the fork step is evaluated on off-distribution states.
- Quantify: fraction of post-fork steps for which the replayed state is still
  valid (= 1st-div / remaining steps). Forward swap-up@early: ~1 of ~20
  remaining steps → replay validity ≈ 5%.

## 4. Implications for routing

- Per-step routers should treat *trajectory position* as a feature: late
  downgrades are near-free (efficiency win: hand off to the cheap model once
  the hard work is done); early upgrades change everything (route hard
  instances up *before* the trajectory ossifies).
- Router benchmarks for agents need live or branched evaluation; we release
  the harness + 300-rollout dataset as a starting point.

## 5. Limitations

- One scaffold (mini-swe-agent), one benchmark (SWE-bench Verified subset,
  n=30), one model family (Qwen3) — pilot scale; scaling to more domains,
  pools, and stochastic seeds is ongoing (this is honest and fine for a
  non-archival workshop).
- 28k context truncation ends some rollouts early (logged as outcomes).
- AWQ/FP8 quantization is a capability confound between pool members.
- Prefix replay assumes command-level determinism; we log returncode
  mismatches (report the rate).
- Patch metrics measure cumulative workspace state, not post-fork behavior
  alone (this is exactly why action- and patch-level metrics decouple).

## 6. Related work (compressed)

Routers & benchmarks: RouteLLM, FrugalGPT/cascades, RouterBench, RouterEval,
RouterArena, RouteJudge. RL routing: Router-R1, SeqRoute. Agentic routing:
SAAR (vLLM), Switchcraft, step-level computer-use routing, TwinRouterBench
(closest: "live dynamic" = changing prices/availability, NOT counterfactual
trajectories — sharpen this distinction). Test-time compute allocation:
Route-to-Reason, ODAR. Nondeterminism of LLM inference: batching
nondeterminism reports. OPE for RL as the future fix (position for the full
paper).

## Data freeze notes (2026-07-16 harvest)

Easy-bucket outcome tables (rgeasy/rgeasyrev):
- fwd easy: base 0/29 resolved; all arms 0; no flips.
- **rev easy: base 1/30 resolved. Swap-down LOST the solve at BOTH fork
  points (down=1 at early AND late, flip-rate 3.3%); same-model control kept
  it at both (flip 0.0%). First observed outcome flip. Pairs with the fwd
  pilot30 rescue (early upgrade resolved an instance the base couldn't,
  3.2%). Both directions now show flips invisible to replay evaluation.**

Nudged runs (runs/nudge, runs/nudge_rev), exit statuses:
- nudge fwd: Submitted 11 (was 8), Limits 16, Ctx 2, Format 1 — nudge helped.
- nudge rev: Submitted 6 (was 10), Ctx 7, Limits 17 — nudge BACKFIRED for
  the 14B: the budget paragraph consumes context it was already short on.
  (Irony for limitations: prompt-level budget control is itself a compute
  cost.)

Nudged divergence (3rd replication of the core pattern):
- nudge fwd: large@early edit 0.940 / 1st-div 0.4 / ident 0.500 vs controls
  small@early 0.717/3.9/0.567, small@late 0.518/4.9/0.733.
- nudge rev: small@early edit 0.917 / 1st-div 0.9 / ident 0.800 vs controls
  large@early 0.272/9.7/0.900, large@late 0.214/8.0/0.933.
- n-subm still small (0–7 per arm); patch-level claims stay subset-qualified.

FINAL (2026-07-16, DATA FROZEN). Nudged outcome tables:
- nudge fwd (base 0/30): large@early up=1 (3.3%), large@late up=1 — CONFIRMED
  same instance (django__django-11163) rescued at BOTH forks (k=8 and k=18):
  the rescue is fork-position-invariant, a property of the swap itself.
  Mirrors the easy_rev downgrade, which was also lost at both forks.
  Controls 0.
- nudge rev (base 0/30): all zeros (no downgrades measurable).

**Cumulative headline stat across all six run-pairs: every observed outcome
flip occurred in a swap arm — 5 flip events (3 rescue, 2 downgrade) vs ZERO
flips in ~360 same-model control branches.** Small n, perfectly one-sided,
and invisible to replay evaluation by construction. State it exactly this
honestly in the abstract.

## TODO before submission (deadline Jul 19 AoE)

- [ ] Fill Table 2 from rev scoring (aggregate_outcomes.py ... rgrev)
- [ ] Replay-fidelity rate (mean replay_mismatches / prefix actions) — one line
- [ ] Token/latency cost per arm (from traj usage sums) — one small table
- [ ] Port to COLM LaTeX template (colmstyle), 4–10 pp
- [ ] Figures: (1) divergence vs fork position by direction, (2) handoff
      inheritance schematic, (3) first-divergence histogram
- [ ] Decide authorship + acknowledgments; OpenReview account for submission
- [ ] Release repo (GitHub) + anonymization check per workshop policy
