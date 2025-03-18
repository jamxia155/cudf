# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl


@pytest.fixture(scope="module")
def engine():
    """
    Fixture for a `GPUEngine` with the `dask-experimental` executor.

    Sets the maximum number of rows per partition to 4.
    """
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        executor_options={"max_rows_per_partition": 4},
    )
