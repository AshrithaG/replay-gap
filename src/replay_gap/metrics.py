"""Divergence metrics between a base trajectory and its branches. Stdlib only."""

import difflib
import re


def commands(messages: list[dict], from_step: int = 1) -> list[str]:
    """Flat list of executed commands from assistant turn `from_step` (1-indexed) onward."""
    out = []
    step = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        step += 1
        if step < from_step:
            continue
        for a in m.get("extra", {}).get("actions", []):
            out.append(a.get("command", "") or "")
    return out


def levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def action_divergence(base_cmds: list[str], branch_cmds: list[str]) -> dict:
    """Edit distance (normalized) + index of first differing action."""
    dist = levenshtein(base_cmds, branch_cmds)
    denom = max(len(base_cmds), len(branch_cmds), 1)
    first_diff = None
    for i, (x, y) in enumerate(zip(base_cmds, branch_cmds)):
        if x != y:
            first_diff = i
            break
    if first_diff is None and len(base_cmds) != len(branch_cmds):
        first_diff = min(len(base_cmds), len(branch_cmds))
    return {
        "edit_distance": dist,
        "normalized_edit_distance": dist / denom,
        "first_divergent_action": first_diff,
        "n_actions_base": len(base_cmds),
        "n_actions_branch": len(branch_cmds),
    }


_DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/\S+", re.MULTILINE)


def patch_files(patch: str) -> set[str]:
    return set(_DIFF_FILE_RE.findall(patch or ""))


def patch_metrics(base_patch: str, branch_patch: str) -> dict:
    base_patch, branch_patch = base_patch or "", branch_patch or ""
    fa, fb = patch_files(base_patch), patch_files(branch_patch)
    union = fa | fb
    return {
        "file_jaccard": (len(fa & fb) / len(union)) if union else 1.0,
        "patch_similarity": difflib.SequenceMatcher(None, base_patch, branch_patch).ratio()
        if (base_patch or branch_patch)
        else 1.0,
        "patch_identical": base_patch.strip() == branch_patch.strip(),
        "both_submitted": bool(base_patch.strip()) and bool(branch_patch.strip()),
    }


def token_usage(messages: list[dict]) -> dict:
    prompt = completion = 0
    for m in messages:
        usage = (m.get("extra", {}).get("response") or {}).get("usage") or {}
        prompt += usage.get("prompt_tokens") or 0
        completion += usage.get("completion_tokens") or 0
    return {"prompt_tokens": prompt, "completion_tokens": completion}


def compare(base_traj: dict, branch_traj: dict, fork_step: int) -> dict:
    """All divergence metrics for one (base, branch) pair, post-fork only."""
    base_msgs, branch_msgs = base_traj["messages"], branch_traj["messages"]
    base_cmds = commands(base_msgs, from_step=fork_step)
    branch_cmds = commands(branch_msgs, from_step=fork_step)
    base_info, branch_info = base_traj.get("info", {}), branch_traj.get("info", {})
    return {
        **action_divergence(base_cmds, branch_cmds),
        **patch_metrics(base_info.get("submission", ""), branch_info.get("submission", "")),
        "base_exit_status": base_info.get("exit_status", ""),
        "branch_exit_status": branch_info.get("exit_status", ""),
        "branch_tokens": token_usage(branch_msgs),
    }
