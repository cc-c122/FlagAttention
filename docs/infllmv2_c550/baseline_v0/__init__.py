# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

from typing import Literal

# Older C550 checkouts predate the common operator registry.
from importlib import import_module
from flag_attn.runtime import backend as _backend


def resolve_operator(name, module):
    resolver = getattr(_backend, "resolve_operator", None)
    if resolver is not None:
        return resolver(name, module)
    return getattr(import_module(module), name)

from .operator import InfLLMV2Config

infllmv2_decode = resolve_operator("infllmv2_decode", "flag_attn.infllmv2.operator")


def infllmv2_attention(*args, stage: Literal["prefill", "decode"] = "prefill", **kwargs):
    """Run InfLLMv2 attention for the explicitly selected stage.

    Arguments and return values follow operator.infllmv2_attention for
    prefill (the default) and operator.infllmv2_decode for decode.
    The original config, cache layout and autograd behavior are preserved.
    """
    if stage == "prefill":
        return resolve_operator("infllmv2_attention", "flag_attn.infllmv2.operator")(*args, **kwargs)
    if stage == "decode":
        return infllmv2_decode(*args, **kwargs)
    raise ValueError(f"Unsupported InfLLMv2 stage: {stage!r}; expected 'prefill' or 'decode'")


__all__ = [
    "InfLLMV2Config",
    "infllmv2_attention",
    "infllmv2_decode",
]
