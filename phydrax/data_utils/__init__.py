#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

"""
# Data utilities

Data utilities provide lightweight preprocessing and tabular ingestion helpers
that return JAX-compatible arrays where possible.
"""

from . import scalers
from ._csv_reader import CSVReader
from .scalers import (
    AffineScaler,
    MaxAbsScaler,
    MinMaxScaler,
    NormScaler,
    scaler_transform_fn,
    StdScaler,
)


__all__ = [
    "AffineScaler",
    "CSVReader",
    "MaxAbsScaler",
    "MinMaxScaler",
    "NormScaler",
    "StdScaler",
    "scalers",
    "scaler_transform_fn",
]
