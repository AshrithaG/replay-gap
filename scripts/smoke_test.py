#!/usr/bin/env python3
"""End-to-end smoke test of the branching machinery — no GPU, no docker, no API.

Runs a scripted base trajectory in a temp dir, forks it at step 2 into
(a) a control branch (same scripted actions) and (b) a divergent branch,
then checks that prefix replay rebuilt state and metrics see the divergence.

    python scripts/smoke_test.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.test_models import DeterministicToolcallModel, make_toolcall_output

from replay_gap import metrics
from replay_gap.branching import BranchableAgent, n_assistant_steps, prefix_for_fork, replay_prefix_actions

AGENT_CFG = dict(
    system_template="You are a test agent.",
    instance_template="Task: {{task}}",
    step_limit=10,
    cost_limit=0,
)


def scripted(cmds: list[str]) -> DeterministicToolcallModel:
    outputs = []
    for i, cmd in enumerate(cmds):
        tc_id = f"call_{i}"
        outputs.append(
            make_toolcall_output(
                content=f"Step {i}",
                tool_calls=[
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"command": cmd})},
                    }
                ],
                actions=[{"command": cmd, "tool_call_id": tc_id}],
            )
        )
    return DeterministicToolcallModel(outputs=outputs)


SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
BASE_STEPS = [
    "mkdir -p work && echo base-1 > work/log.txt",
    "echo base-2 >> work/log.txt && printf 'diff --git a/foo.py b/foo.py\\n' > patch.txt",
    SUBMIT,
]
DIVERGENT_TAIL = [
    "echo branch-2 >> work/log.txt && printf 'diff --git a/bar.py b/bar.py\\n' > patch.txt",
    SUBMIT,
]
FORK = 2  # re-decide assistant turn 2


def run_base(tmp: Path) -> dict:
    env = LocalEnvironment(cwd=str(tmp))
    agent = BranchableAgent(scripted(BASE_STEPS), env, **AGENT_CFG)
    agent.run(task="smoke")
    return agent.serialize()


def run_branch(tmp: Path, base_traj: dict, tail_cmds: list[str]) -> tuple[dict, list[dict]]:
    env = LocalEnvironment(cwd=str(tmp))
    prefix = prefix_for_fork(base_traj["messages"], FORK)
    fidelity = replay_prefix_actions(env, prefix)
    agent = BranchableAgent(scripted(tail_cmds), env, **AGENT_CFG)
    agent.n_calls = FORK - 1
    agent.run_from_prefix(prefix, task="smoke")
    return agent.serialize(), fidelity


def main() -> None:
    checks = []

    def check(name: str, cond: bool):
        checks.append((name, cond))
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (base_dir := root / "base").mkdir()
        (ctrl_dir := root / "ctrl").mkdir()
        (div_dir := root / "div").mkdir()

        print("base rollout:")
        base = run_base(base_dir)
        check("base submitted", base["info"]["exit_status"] == "Submitted")
        check("base patch mentions foo.py", "foo.py" in base["info"]["submission"])
        check("base has 3 assistant steps", n_assistant_steps(base["messages"]) == 3)

        print("control branch (same model/actions from step 2):")
        ctrl, fid = run_branch(ctrl_dir, base, BASE_STEPS[FORK - 1 :])
        check("prefix replay ran 1 action, all returncodes match", len(fid) == 1 and all(f["match"] for f in fid))
        check("replay rebuilt state", (ctrl_dir / "work/log.txt").read_text().startswith("base-1"))
        m = metrics.compare(base, ctrl, FORK)
        check("control patch identical", m["patch_identical"])
        check("control zero divergence", m["normalized_edit_distance"] == 0.0)

        print("divergent branch (model swap at step 2):")
        div, _ = run_branch(div_dir, base, DIVERGENT_TAIL)
        check("divergent branch submitted", div["info"]["exit_status"] == "Submitted")
        m = metrics.compare(base, div, FORK)
        check("divergence detected at first post-fork action", m["first_divergent_action"] == 0)
        check("patch files disjoint (jaccard 0)", m["file_jaccard"] == 0.0)
        check(
            "branch env state = base prefix + branch tail",
            (div_dir / "work/log.txt").read_text() == "base-1\nbranch-2\n",
        )

    failed = [n for n, ok in checks if not ok]
    if failed:
        sys.exit(f"\n{len(failed)} check(s) FAILED: {failed}")
    print(f"\nAll {len(checks)} checks passed — branching machinery works.")


if __name__ == "__main__":
    main()
