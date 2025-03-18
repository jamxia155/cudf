# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from cudf_polars.experimental.base import PartitionInfo


def test_partition_info_repr():
    partition_info = PartitionInfo(count=10, partitioned_on=("a", "b"))
    assert repr(partition_info) == "PartitionInfo(count=10, partitioned_on=('a', 'b'))"
