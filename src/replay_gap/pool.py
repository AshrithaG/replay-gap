"""Model pool: build mini-swe-agent Model objects for self-hosted vLLM endpoints.

Each pool entry in the config looks like:

    small:
      model_name: "hosted_vllm/Qwen/Qwen3-4B-Instruct-2507"
      api_base: "http://localhost:8001/v1"
      temperature: 0.0
      max_tokens: 4096

`model_name` uses litellm's provider syntax; "hosted_vllm/<served-model-name>"
targets any OpenAI-compatible vLLM server. Cost tracking is disabled (litellm
doesn't know prices for self-hosted models); we track tokens ourselves from the
usage field that vLLM returns.
"""

import os

# Must be set before minisweagent.models is imported anywhere.
os.environ.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from minisweagent.models.litellm_model import LitellmModel  # noqa: E402
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel  # noqa: E402

MODEL_CLASSES = {
    "toolcall": LitellmModel,
    "textbased": LitellmTextbasedModel,
}


def build_model(spec: dict, seed: int | None = None):
    """Build a Model from a pool-entry spec dict."""
    spec = dict(spec)
    model_class = MODEL_CLASSES[spec.pop("model_class", "toolcall")]
    model_kwargs = {
        "drop_params": True,
        "temperature": spec.pop("temperature", 0.0),
        "max_tokens": spec.pop("max_tokens", 4096),
        **spec.pop("model_kwargs", {}),
    }
    if api_base := spec.pop("api_base", None):
        model_kwargs["api_base"] = api_base
        # vLLM's OpenAI server requires some api_key string; value is ignored.
        model_kwargs.setdefault("api_key", os.getenv("VLLM_API_KEY", "EMPTY"))
    if seed is not None:
        model_kwargs["seed"] = seed
    return model_class(
        model_name=spec.pop("model_name"),
        model_kwargs=model_kwargs,
        cost_tracking="ignore_errors",
        **spec,
    )


def build_pool(pool_config: dict, seed: int | None = None) -> dict:
    """Build {alias: Model} for every entry in the config's `pool` section."""
    return {alias: build_model(spec, seed=seed) for alias, spec in pool_config.items()}
