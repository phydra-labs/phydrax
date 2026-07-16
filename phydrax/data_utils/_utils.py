#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

import polars as pl


def _is_numeric_dtype(data: pl.DataFrame | pl.Series) -> bool:
    if isinstance(data, pl.DataFrame):
        return all(dtype.is_numeric() for dtype in data.dtypes)
    return data.dtype.is_numeric()
