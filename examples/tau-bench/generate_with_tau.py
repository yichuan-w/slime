"""
Tau-Bench Integration for slime Training

This module provides the main interface for training agents in tau-bench environments
using the slime framework. It handles agent-environment interactions and converts
results to the format expected by slime's training pipeline.
"""

import logging
import os
from typing import Any

from tau_bench.envs import get_env
from tau_bench.types import RunConfig
from trainable_agents import InteractionResult, Status, agent_factory

from slime.utils.types import Sample

# Set up logger for this module
logger = logging.getLogger(__name__)

# Tau-bench configuration.
#
# User simulator is routed through Meta's sanctioned MetaGen "Llama Public API"
# (OpenAI-compatible) instead of a direct Gemini key (direct GCP/3P access is
# not permitted for work; see the 3P Commercial Models policy). litellm talks to
# any OpenAI-compatible endpoint when custom_llm_provider="openai" and
# OPENAI_API_BASE / OPENAI_API_KEY are set.
#
# Provide via env (no secret committed):
#   LLAMA_API_KEY    : your MetaGen key            (required)
#   TAU_USER_MODEL   : MetaGen model id for user-sim (default below)
#   TAU_USER_BASE_URL: OpenAI-compatible base url   (default api.llama.com/compat/v1)
TAU_USER_MODEL = os.environ.get("TAU_USER_MODEL", "gemini-3-flash-preview-fair")
TAU_USER_BASE_URL = os.environ.get("TAU_USER_BASE_URL", "https://api.llama.com/compat/v1")
_LLAMA_KEY = os.environ.get("LLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY") or "NONE"

# Point litellm's OpenAI handler at MetaGen.
os.environ["OPENAI_API_KEY"] = _LLAMA_KEY
os.environ["OPENAI_API_BASE"] = TAU_USER_BASE_URL
os.environ["OPENAI_BASE_URL"] = TAU_USER_BASE_URL

# Call the user-sim through the OpenAI SDK directly instead of litellm.
# Why: litellm 1.87's response parser rejects Gemini's compat responses (the
# Google `extra_content`/`thought_signature` + `reasoning_tokens` fields) with
# "Invalid response object". The OpenAI SDK handles them fine. We patch
# tau-bench's `completion` (user.py does `from litellm import completion`) to a
# shim that (1) calls OpenAI SDK, (2) returns a litellm-shaped object exposing
# `_hidden_params`, (3) caps concurrency + retries 429 with backoff.
# NOTE: do NOT pass max_tokens — Gemini flash is a reasoning model and will spend
# the whole budget "thinking" and return content=None if the budget is small.
import random  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import tau_bench.envs.user as _tau_user  # noqa: E402
from openai import OpenAI  # noqa: E402

_oai_client = OpenAI(api_key=_LLAMA_KEY, base_url=TAU_USER_BASE_URL.rstrip("/") + "/")
_USER_SIM_SEM = threading.Semaphore(int(os.environ.get("TAU_USER_MAX_CONCURRENCY", "16")))
_MAX_RL_RETRIES = int(os.environ.get("TAU_USER_MAX_RETRIES", "20"))


def _is_rate_limit(e):
    name = type(e).__name__.lower()
    msg = str(e).lower()
    return "ratelimit" in name or "429" in msg or "rate limit" in msg or "maximum number" in msg


class _EmptyResponse(Exception):
    """Gemini flash (reasoning model) intermittently returns a choice with
    message=None or empty content (spent its budget thinking). If that reaches
    tau-bench it aborts the trajectory, and an aborted/empty sample then poisons
    the training batch (slime loss.py: `xs` becomes None -> crash). Treat it as
    a transient, retryable error instead."""


class _MsgShim:
    """Message whose model_dump() is ONLY {role, content}. Gemini flash returns
    Google `extra_content`/`thought_signature` fields; if those get appended to
    the conversation history and sent back, MetaGen rejects the request with
    `400 oneOf schema` (messages.N). Stripping them keeps multi-turn valid."""

    def __init__(self, m):
        object.__setattr__(self, "_m", m)

    @property
    def content(self):
        return object.__getattribute__(self, "_m").content or ""

    @property
    def role(self):
        return getattr(object.__getattribute__(self, "_m"), "role", "assistant") or "assistant"

    def model_dump(self, *a, **k):
        return {"role": self.role, "content": self.content}

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_m"), k)


class _ChoiceShim:
    def __init__(self, ch):
        object.__setattr__(self, "_ch", ch)
        object.__setattr__(self, "message", _MsgShim(ch.message))

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_ch"), k)


class _RespShim:
    """Wrap an OpenAI SDK response so tau-bench (expects litellm) is happy."""

    def __init__(self, r):
        object.__setattr__(self, "_r", r)
        object.__setattr__(self, "choices", [_ChoiceShim(c) for c in r.choices])
        object.__setattr__(self, "_hidden_params", {"response_cost": 0.0})

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_r"), k)


def _throttled_completion(*args, **kwargs):
    model = kwargs.get("model") or (args[0] if args else None)
    params = {"model": model, "messages": kwargs.get("messages")}
    # pass through sampling/format kwargs but NOT max_tokens (see note above)
    for k in ("temperature", "top_p", "response_format", "tools", "tool_choice", "stop"):
        if kwargs.get(k) is not None:
            params[k] = kwargs[k]
    delay = 2.0
    for attempt in range(_MAX_RL_RETRIES):
        with _USER_SIM_SEM:
            try:
                r = _oai_client.chat.completions.create(**params)
                # Reject empty/None responses so they get retried rather than
                # aborting the trajectory (see _EmptyResponse).
                msg = r.choices[0].message if getattr(r, "choices", None) else None
                if msg is None or not (msg.content or "").strip():
                    raise _EmptyResponse("user-sim returned empty/None content")
                return _RespShim(r)
            except Exception as e:  # noqa: BLE001
                retryable = _is_rate_limit(e) or isinstance(e, _EmptyResponse)
                if not retryable or attempt == _MAX_RL_RETRIES - 1:
                    raise
        time.sleep(delay + random.random())
        delay = min(delay * 1.7, 30.0)


_tau_user.completion = _throttled_completion

TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"]
    "user_model": TAU_USER_MODEL,  # MetaGen model id for the user simulator
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router",  # Unused, required
    "model": "qwen3-4b",  # Unused, required
    "user_model_provider": "openai",  # OpenAI-compatible -> MetaGen
}
tau_config = RunConfig(**TAU_CONFIGS)


def res_to_sample(res: InteractionResult, task_index: int) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.

    This function transforms the tau-bench interaction result into the format
    expected by slime's training pipeline, handling status mapping and response
    length calculation.

    Args:
        res: InteractionResult from tau-bench agent
        task_index: Index of the task being processed

    Returns:
        Sample object for slime training
    """
    # Map tau-bench status to slime status
    status_mapping = {
        Status.COMPLETED: Sample.Status.COMPLETED,
        Status.TRUNCATED: Sample.Status.TRUNCATED,
        Status.ABORTED: Sample.Status.ABORTED,
    }
    status = status_mapping.get(res.status)

    # Debug logging for response tracking
    logger.debug(
        f"res_to_sample: response_length="
        f"{res.response_length if hasattr(res, 'response_length') else 'None'}, "
        f"loss_mask_len={len(res.loss_mask) if res.loss_mask else 'None'}, "
        f"tokens_len={len(res.tokens) if res.tokens else 'None'}"
    )

    # Create sample with basic information
    sample = Sample(
        index=task_index,
        prompt=res.prompt,
        tokens=res.tokens,
        response=res.response,
        reward=res.reward,
        loss_mask=res.loss_mask,
        status=status,
        metadata=res.info,
    )

    # Ensure response_length is set correctly
    if hasattr(res, "response_length"):
        sample.response_length = res.response_length
    else:
        # Fallback: calculate from loss_mask if available
        if res.loss_mask:
            # loss_mask only contains response part, so length equals response_length
            sample.response_length = len(res.loss_mask)
        elif res.tokens:
            # If no loss_mask available, use total tokens as fallback
            sample.response_length = len(res.tokens)
        else:
            sample.response_length = 0
            logger.debug(f"res_to_sample: Set response_length={sample.response_length}")

    return sample


async def generate(args: dict[str, Any], sample: Sample, sampling_params: dict) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for tau-bench.

    This is the main entry point for slime training. It creates a tau-bench
    environment, initializes a trainable agent, and executes a full interaction
    trajectory. The result is converted to slime's Sample format for training.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task index in prompt field
        sampling_params: LLM sampling parameters

    Returns:
        Sample object containing the complete interaction trajectory

    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."

    # Extract task index from sample prompt
    task_index = int(sample.prompt)
    logger.info(f"Starting agent-environment interaction for task {task_index}")

    # Whole-trajectory retry guard. If asolve aborts (e.g. a user-sim hiccup
    # the per-call retry couldn't save) or returns an empty trajectory
    # (response_length == 0), the resulting sample has no response tokens / no
    # log_probs and POISONS the training batch -> slime crashes with
    # `'NoneType' object is not iterable` in compute_advantages_and_returns,
    # killing the entire run. Re-run the trajectory from a fresh env instead.
    max_traj_retries = max(1, int(os.environ.get("TAU_TRAJ_RETRIES", "4")))
    interaction_result = None
    for attempt in range(max_traj_retries):
        env = get_env(
            env_name=tau_config.env,
            user_strategy=tau_config.user_strategy,
            user_model=tau_config.user_model,
            user_provider=tau_config.user_model_provider,
            task_split=tau_config.task_split,
            task_index=task_index,
        )
        agent = agent_factory(
            tools_info=env.tools_info,
            wiki=env.wiki,
            config=tau_config,
            rollout_args=args,
            sampling_params=sampling_params,
        )
        # The sample.prompt field contains the task index for repeatability
        interaction_result = await agent.asolve(env, agent.rollout_args, agent.sampling_params, task_index)
        if interaction_result.status != Status.ABORTED and (interaction_result.response_length or 0) > 0:
            break
        logger.warning(
            f"Task {task_index}: aborted/empty trajectory "
            f"(status={interaction_result.status}, len={interaction_result.response_length}), "
            f"retry {attempt + 1}/{max_traj_retries}"
        )

    # Convert to slime Sample format
    result_sample = res_to_sample(interaction_result, task_index)

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return result_sample
