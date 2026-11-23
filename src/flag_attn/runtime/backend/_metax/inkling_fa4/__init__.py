# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from typing import Any

import torch

from .forward import inkling_fa4_rel_attention_triton

_TLE_UNAVAILABLE = "TLE is unavailable on MetaX"


def inkling_fa4_rel_attention_tle(*args: Any, **kwargs: Any) -> torch.Tensor:
    raise RuntimeError(_TLE_UNAVAILABLE)


def tle_available() -> tuple[bool, str]:
    return False, _TLE_UNAVAILABLE


def _require_triton(name: str | None, tle_error: type[Exception]) -> None:
    selected = (name or os.getenv("INKLING_FA4_BACKEND", "auto")).strip().lower()
    if selected in ("auto", "triton", "base"):
        return
    if selected in ("tle", "triton_tle", "triton-tle"):
        raise tle_error(_TLE_UNAVAILABLE)
    raise ValueError(f"unknown Inkling FA4 backend={selected!r}; choose from auto, triton, base")


def get_backend(name: str | None = None):
    _require_triton(name, ImportError)
    return inkling_fa4_rel_attention_triton


def inkling_fa4_rel_attention(
    *args: Any,
    backend: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    _require_triton(backend, RuntimeError)
    return inkling_fa4_rel_attention_triton(*args, **kwargs)


__all__ = [
    "get_backend",
    "inkling_fa4_rel_attention",
    "inkling_fa4_rel_attention_tle",
    "inkling_fa4_rel_attention_triton",
    "tle_available",
]
