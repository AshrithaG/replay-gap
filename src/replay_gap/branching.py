"""Branching (counterfactual) rollouts for agentic model routing.

Protocol per SWE-bench instance:
  1. BASE: run the base model to completion, saving the full trajectory.
  2. BRANCH: pick a fork step k (an assistant turn). Start a *fresh* container,
     re-execute the recorded actions of assistant turns 1..k-1 to rebuild
     environment state, seed the agent's message history with the recorded
     prefix (so the branch model sees exactly what the base model saw), then
     continue the rollout with the branch model from turn k onward.

The same-model branch (branch model == base model) is the control arm: any
divergence there is sampling/replay noise, and cross-model divergence must be
read relative to it.
"""

import logging
import re
import time
from pathlib import Path

from litellm.exceptions import ContextWindowExceededError
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import FormatError, InterruptAgentFlow, Submitted
from minisweagent.run.benchmarks.swebench import get_sb_environment

logger = logging.getLogger("replay_gap")

_RETURNCODE_RE = re.compile(r"<returncode>(-?\d+)</returncode>")


class BranchableAgent(DefaultAgent):
    """DefaultAgent that can resume from a recorded message prefix."""

    def run_loop(self) -> dict:
        # Same loop body as DefaultAgent.run() (mini-swe-agent 2.4.x), factored
        # out so it can start from an arbitrary message state.
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as e:
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def run(self, task: str = "", **kwargs) -> dict:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        return self.run_loop()

    def run_from_prefix(self, prefix_messages: list[dict], task: str = "", **kwargs) -> dict:
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(*prefix_messages)
        return self.run_loop()


def assistant_indices(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def n_assistant_steps(messages: list[dict]) -> int:
    return len(assistant_indices(messages))


def prefix_for_fork(messages: list[dict], fork_step: int) -> list[dict]:
    """Messages strictly before the fork_step-th (1-indexed) assistant turn.

    The branch model re-decides turn `fork_step` and everything after it.
    """
    idxs = assistant_indices(messages)
    if not 1 <= fork_step <= len(idxs):
        raise ValueError(f"fork_step {fork_step} out of range (1..{len(idxs)})")
    return messages[: idxs[fork_step - 1]]


def resolve_fork_steps(fork_spec: list, n_steps: int) -> list[int]:
    """Turn a config fork spec (ints, or floats as fractions of trajectory
    length) into valid, deduplicated 1-indexed assistant-step numbers.
    A fork at the very last step is allowed (the branch redoes the final turn).
    """
    steps = []
    for f in fork_spec:
        k = max(1, round(f * n_steps)) if isinstance(f, float) else int(f)
        if 1 <= k <= n_steps:
            steps.append(k)
    return sorted(set(steps))


def replay_prefix_actions(env, prefix_messages: list[dict]) -> list[dict]:
    """Re-execute recorded actions to rebuild container state.

    Returns a fidelity log comparing fresh return codes with the recorded ones
    (parsed from the recorded observation messages), so replay drift caused by
    environment nondeterminism is measurable instead of silent.
    """
    # Recorded returncodes, in action order, parsed from tool/user observations.
    recorded_rcs = []
    for m in prefix_messages:
        if m.get("role") in ("tool", "user"):
            content = m.get("content") or ""
            if isinstance(content, list):  # multimodal-style content blocks
                content = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
            recorded_rcs.extend(int(x) for x in _RETURNCODE_RE.findall(content))

    fidelity = []
    action_i = 0
    for m in prefix_messages:
        if m.get("role") != "assistant":
            continue
        for action in m.get("extra", {}).get("actions", []):
            try:
                out = env.execute(action)
                rc = out.get("returncode")
            except Submitted:  # defensive: a prefix should never contain the submission
                rc = 0
            except Exception as e:
                logger.warning(f"Replay action failed: {e}")
                rc = None
            recorded = recorded_rcs[action_i] if action_i < len(recorded_rcs) else None
            fidelity.append(
                {
                    "action_index": action_i,
                    "command": (action.get("command", "") or "")[:200],
                    "replayed_returncode": rc,
                    "recorded_returncode": recorded,
                    "match": rc == recorded if recorded is not None else None,
                }
            )
            action_i += 1
    return fidelity


def make_agent(model, env, agent_config: dict, output_path: Path) -> BranchableAgent:
    cfg = {k: v for k, v in agent_config.items() if k in AgentConfig.model_fields}
    return BranchableAgent(model, env, output_path=output_path, **cfg)


def run_base(instance: dict, model, config: dict, output_path: Path) -> dict:
    """Run the base trajectory for one instance. Returns the saved trajectory dict."""
    env = get_sb_environment(config, instance)
    try:
        agent = make_agent(model, env, config["agent"], output_path)
        t0 = time.time()
        try:
            agent.run(task=instance["problem_statement"])
        except ContextWindowExceededError:
            # A legitimate terminal outcome on small-context serving, not a
            # harness failure: the exit message is already in the trajectory.
            logger.warning(f"[{instance['instance_id']}] base hit the context limit")
        meta = {
            "replay_gap": {
                "instance_id": instance["instance_id"],
                "arm": "base",
                "fork_step": None,
                "wall_time_s": time.time() - t0,
            }
        }
        return agent.save(output_path, meta)
    finally:
        _cleanup(env)


def run_branch(
    instance: dict,
    base_messages: list[dict],
    fork_step: int,
    model,
    config: dict,
    output_path: Path,
    arm: str,
) -> dict:
    """Fork the base trajectory at `fork_step` and continue with `model`."""
    env = get_sb_environment(config, instance)
    try:
        prefix = prefix_for_fork(base_messages, fork_step)
        fidelity = replay_prefix_actions(env, prefix)
        agent = make_agent(model, env, config["agent"], output_path)
        # The continuation still gets a full step budget relative to the prefix.
        agent.n_calls = fork_step - 1
        t0 = time.time()
        try:
            agent.run_from_prefix(prefix, task=instance["problem_statement"])
        except ContextWindowExceededError:
            logger.warning(f"[{instance['instance_id']}] branch {arm} hit the context limit")
        meta = {
            "replay_gap": {
                "instance_id": instance["instance_id"],
                "arm": arm,
                "fork_step": fork_step,
                "replay_fidelity": fidelity,
                "replay_mismatches": sum(1 for f in fidelity if f["match"] is False),
                "wall_time_s": time.time() - t0,
            }
        }
        return agent.save(output_path, meta)
    finally:
        _cleanup(env)


def _cleanup(env) -> None:
    for attr in ("cleanup", "stop", "close"):
        if hasattr(env, attr):
            try:
                getattr(env, attr)()
            except Exception:
                pass
            return
