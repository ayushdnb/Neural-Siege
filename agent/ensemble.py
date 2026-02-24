from __future__ import annotations
# ^ Postpones evaluation of type hints (PEP 563 / modern Python behavior via __future__).
#   This helps avoid issues when using forward references (types not yet defined at parse time).

from typing import List, Tuple, Optional, Any
# ^ Standard typing imports:
#   - List[T]: list of T
#   - Tuple[A, B]: a 2-item tuple
#   - Optional[T]: either T or None
#   - Any: intentionally unconstrained type

import torch
import torch.nn as nn
import config
# ^ Local project configuration module. This code reads feature flags from config, such as:
#   - USE_VMAP: enable/disable vmap path
#   - VMAP_MIN_BUCKET: minimum bucket size for vmap
#   - VMAP_DEBUG: print debug messages

# --------------------------------------------------------------------------------------
# Optional imports from torch.func (PyTorch functional transforms).
#
# These APIs may not exist in older or minimal PyTorch builds, so we import them defensively.
# If the import fails, we set them to None and fall back to a safe Python loop.
# --------------------------------------------------------------------------------------
try:
    from torch.func import functional_call, vmap, stack_module_state
except Exception:
    functional_call = None
    vmap = None
    stack_module_state = None


class _DistWrap:
    """
    Lightweight container to mimic a torch.distributions object with logits.

    Why this exists:
    - Some RL policy code expects a "distribution-like" object that exposes `.logits`.
    - This wrapper provides a minimal, compatible interface: only `.logits` is stored.
    - It avoids depending on a specific distribution class (e.g., Categorical).
    """
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


# Global state used to ensure we only emit the "warn once" message a single time.
_VMAP_WARNED: bool = False

def _is_torchscript_module(m: nn.Module) -> bool:
    """
    Returns True if the given module is a TorchScript-compiled module.

    TorchScript modules (torch.jit.ScriptModule / RecursiveScriptModule) can have constraints
    that make certain torch.func operations unsafe or unsupported.
    """
    return isinstance(m, torch.jit.ScriptModule) or isinstance(m, torch.jit.RecursiveScriptModule)

def _maybe_debug(msg: str) -> None:
    """
    Debug-print helper gated by config.VMAP_DEBUG.

    This keeps the main control flow clean while still allowing opt-in diagnostics.
    """
    if bool(getattr(config, "VMAP_DEBUG", False)):
        print(msg)

def _maybe_warn_once(msg: str) -> None:
    """
    "Warn once" helper.

    This is used for vmap fallback messaging so logs do not get spammed every step.
    """
    global _VMAP_WARNED
    if _VMAP_WARNED:
        return
    _VMAP_WARNED = True
    _maybe_debug(msg)


@torch.no_grad()
def _ensemble_forward_loop(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    Original safe loop implementation (kept as canonical fallback).

    Purpose:
    - We have K independent models (one per agent).
    - Each model receives its aligned observation obs[i].
    - We run forward passes one-by-one (Python loop), then concatenate outputs.

    Inputs:
    - models: list of nn.Module, length K
    - obs:    tensor shaped (K, F), where:
              K = number of agents in the bucket
              F = feature dimension of a single observation

    Outputs:
    - (_DistWrap with logits): logits shaped (K, A)
    - values tensor:           values shaped (K,)

    Notes:
    - This path is robust and does not require torch.func.
    - It is often slower than vectorized approaches, but it is a reliable baseline.
    """
    device = obs.device

    # Determine bucket size K in a safe way:
    # - If obs has at least one dimension, K = obs.size(0)
    # - If obs is unexpected (dim == 0), treat as empty.
    K = int(obs.size(0)) if obs.dim() > 0 else 0

    # Handle empty bucket:
    # Return empty logits and empty values on the same device.
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=device)), torch.empty((0,), device=device)

    logits_out: List[torch.Tensor] = []
    values_out: List[torch.Tensor] = []

    # Iterate over each model/agent in the bucket.
    for i, model in enumerate(models):
        # obs[i] has shape (F,). The model expects (batch, F), so we add a batch dimension.
        o = obs[i].unsqueeze(0)  # (1,F)

        # Forward pass. Contract: model(o) -> (head, val)
        out = model(o)

        # Validate contract: must be a 2-tuple.
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("Brain.forward must return (logits_or_dist, value)")

        head, val = out

        # Some policies return a distribution object with `.logits`.
        # Others return logits tensor directly. Support both.
        logits = head.logits if hasattr(head, "logits") else head  # (1,A)

        # Value head may have shape (1,1), (1,), etc.
        # We standardize it to a 1D tensor of length 1 via view(-1).
        v = val.view(-1)  # (1,)

        logits_out.append(logits)
        values_out.append(v)

    # Concatenate across agents to produce batched outputs:
    # - logits: (K, A)
    # - values: (K,)
    logits_cat = torch.cat(logits_out, dim=0)  # (K,A)
    values_cat = torch.cat(values_out, dim=0)  # (K,)

    # Ensure values never become 0-dim (scalar tensor).
    # This is defensive: some edge cases can cause shape collapsing.
    if values_cat.dim() == 0:
        values_cat = values_cat.unsqueeze(0)

    # Invariants (hard checks):
    # 1) logits must be rank-2 with leading dimension K.
    # 2) values must be rank-1 with length K.
    if logits_cat.dim() != 2 or logits_cat.size(0) != K:
        raise RuntimeError(f"ensemble_forward loop: logits shape invalid: got {tuple(logits_cat.shape)}, K={K}")
    if values_cat.dim() != 1 or values_cat.size(0) != K:
        raise RuntimeError(f"ensemble_forward loop: values shape invalid: got {tuple(values_cat.shape)}, K={K}")

    return _DistWrap(logits=logits_cat), values_cat


@torch.no_grad()
def _ensemble_forward_vmap(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    vmap-based inference across *independent* parameter sets.

    Key idea:
    - We want to evaluate K models "in parallel" without writing a Python loop.
    - PyTorch's torch.func can treat a module as a pure function:
        f(params, buffers, x) -> y
    - If we "stack" K sets of params/buffers, we can vmap over them:
        vmap(f)(params_batched, buffers_batched, x_batched)

    Important constraints explicitly stated in the docstring:
    - NO parameter sharing: each agent has its own weights.
    - NO optimizer sharing: this path is inference-only (and function is @no_grad).

    Inputs:
    - models: list of nn.Module, length K, same architecture
    - obs:    tensor (K, F)

    Outputs:
    - logits (K, A)
    - values (K,)
    """
    # If torch.func features are not available, we cannot run this path.
    if functional_call is None or vmap is None or stack_module_state is None:
        raise RuntimeError("torch.func is not available in this PyTorch build")

    K = int(obs.size(0))

    # Empty bucket handling.
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=obs.device)), torch.empty((0,), device=obs.device)

    # Safety: TorchScript modules may be incompatible with torch.func transforms.
    if any(_is_torchscript_module(m) for m in models):
        raise RuntimeError("TorchScript module in bucket (vmap disabled)")

    # stack_module_state requires identical module structure across the list.
    # We use the first model as the "base" definition for functional_call.
    base = models[0]

    # Stack parameters and buffers from each model:
    # - params_batched: pytree-like structure where each tensor has leading dim K
    # - buffers_batched: same idea for buffers (e.g., running stats in BatchNorm)
    params_batched, buffers_batched = stack_module_state(models)

    # Ensure obs matches (K, F) exactly.
    x = obs
    if x.dim() != 2 or x.size(0) != K:
        raise RuntimeError(f"vmap: obs must be (K,F). got {tuple(x.shape)} expected K={K}")

    # Define a per-model forward function for vmap.
    #
    # Inputs (per single model i):
    # - params_i: parameters of model i
    # - buffers_i: buffers of model i
    # - x_i: observation for model i (shape (F,))
    #
    # Output:
    # - logits for model i, shape (A,)
    # - value for model i, scalar
    def _f(params_i: Any, buffers_i: Any, x_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # The base model expects input shape (batch, F), so add batch dimension: (1, F)
        out = functional_call(base, (params_i, buffers_i), (x_i.unsqueeze(0),))

        # Validate contract: must return (head, val)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise RuntimeError("Brain.forward must return (logits_or_dist, value)")

        head, val = out

        # Support either a distribution-like head with `.logits` or direct logits tensor.
        logits = head.logits if hasattr(head, "logits") else head  # (1,A)

        # Remove batch dimension so per-item logits are (A,)
        logits = logits.squeeze(0)  # (A,)

        # Return scalar value so that vmap stacks values into shape (K,)
        v = val.view(-1)[0]
        return logits, v

    # Vectorize _f over leading dimension of (params, buffers, x).
    # in_dims=(0,0,0) means:
    # - take params_batched[i]
    # - take buffers_batched[i]
    # - take x[i]
    # and apply _f for each i in 0..K-1, stacking results.
    logits_KA, values_K = vmap(_f, in_dims=(0, 0, 0))(params_batched, buffers_batched, x)

    # Invariants: enforce output shapes.
    if logits_KA.dim() != 2 or logits_KA.size(0) != K:
        raise RuntimeError(f"vmap: logits shape invalid: got {tuple(logits_KA.shape)}, K={K}")
    if values_K.dim() != 1 or values_K.size(0) != K:
        raise RuntimeError(f"vmap: values shape invalid: got {tuple(values_K.shape)}, K={K}")

    return _DistWrap(logits=logits_KA), values_K


@torch.no_grad()
def ensemble_forward(models: List[nn.Module], obs: torch.Tensor) -> Tuple[_DistWrap, torch.Tensor]:
    """
    Fuses per-agent models for a bucket into one batched tensor of outputs.

    This function is the public entry point that chooses between:
    1) A safe Python loop (_ensemble_forward_loop), and
    2) A potentially faster torch.func + vmap path (_ensemble_forward_vmap).

    Args:
      models: list of nn.Module, length K
      obs:    (K, F) observation batch aligned with models ordering

    Returns:
      - dist-like object with .logits -> (K, A)
      - values tensor -> (K,)    (NEVER 0-dim)

    Contract (hard requirement for model.forward):
      Each model.forward(x: (1,F)) -> (logits: (1,A)) or (dist_with_logits, value)

    Design intent:
    - "Bucket" means a group of agents/models assumed to share the same architecture.
    - If bucket size is large enough, we attempt vmap for performance.
    - If anything is incompatible or fails, we fall back safely to the loop.
    """
    # Determine K robustly.
    K = int(obs.size(0)) if obs.dim() > 0 else 0

    # Empty bucket returns empty outputs.
    if K == 0:
        return _DistWrap(logits=torch.empty((0, 0), device=obs.device)), torch.empty((0,), device=obs.device)

    # Feature flags / thresholds from config:
    # - USE_VMAP: master switch
    # - VMAP_MIN_BUCKET: minimum K to justify vmap overhead
    use_vmap = bool(getattr(config, "USE_VMAP", False))
    min_bucket = int(getattr(config, "VMAP_MIN_BUCKET", 8))

    # vmap path is inference-only; this function is already @no_grad
    if use_vmap and K >= min_bucket:
        # Quick structural checks before attempting vmap:
        # - requires torch.func
        # - avoid TorchScript modules
        # - bucket should be homogeneous architecture (already intended by build_buckets)
        if functional_call is None or vmap is None or stack_module_state is None:
            _maybe_warn_once("[vmap] torch.func not available; falling back to loop")
        elif any(_is_torchscript_module(m) for m in models):
            _maybe_debug("[vmap] TorchScript module detected; falling back to loop")
        else:
            # Attempt vmap path; if it fails for any reason, fall back to loop.
            # This is a reliability-first design: performance is optional, correctness is mandatory.
            try:
                return _ensemble_forward_vmap(models, obs)
            except Exception as e:
                _maybe_debug(f"[vmap] vmap path failed ({type(e).__name__}: {e}); falling back to loop")

    # Default and fallback: the canonical safe implementation.
    return _ensemble_forward_loop(models, obs)