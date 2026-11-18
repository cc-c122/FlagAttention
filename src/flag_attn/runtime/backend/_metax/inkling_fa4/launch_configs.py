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

from functools import lru_cache

import triton

DOT_TILE = 16
FP32_BYTES = 4


@lru_cache
def select_num_splits(
    *,
    kv_tiles: int,
    query_groups: int,
    rows_per_group: int,
    output_rows: int,
    head_dim_q: int,
    head_dim_v: int,
    element_bytes: int,
    warp_size: int,
    num_sms: int,
) -> int:
    query_bytes = rows_per_group * head_dim_q * element_bytes
    kv_tile_bytes = DOT_TILE * (head_dim_q + head_dim_v + rows_per_group) * element_bytes
    output_bytes = rows_per_group * head_dim_v * element_bytes
    partial_bytes = rows_per_group * (head_dim_v + 2) * FP32_BYTES

    best_splits = 1
    main_waves = triton.cdiv(query_groups, num_sms)
    best_wave_bytes = main_waves * (query_bytes + kv_tiles * kv_tile_bytes + output_bytes)

    combine_width = min(head_dim_v, warp_size)
    combine_groups = output_rows * triton.cdiv(head_dim_v, combine_width)
    combine_waves = triton.cdiv(combine_groups, num_sms)
    combine_output_bytes = combine_width * element_bytes
    combine_partial_bytes = (combine_width + 2) * FP32_BYTES
    for splits in range(2, min(kv_tiles, num_sms) + 1):
        main_waves = triton.cdiv(query_groups * splits, num_sms)
        tiles_per_split = triton.cdiv(kv_tiles, splits)
        main_bytes = main_waves * (query_bytes + tiles_per_split * kv_tile_bytes + partial_bytes)
        combine_bytes = combine_waves * (FP32_BYTES + splits * combine_partial_bytes + combine_output_bytes)
        wave_bytes = main_bytes + combine_bytes
        if wave_bytes < best_wave_bytes:
            best_splits = splits
            best_wave_bytes = wave_bytes
    return best_splits
