#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generic external-representation interchange contracts."""

from ._report import (
    AdapterError,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    require_lossless,
)


__all__ = [
    "AdapterError",
    "AdapterLoss",
    "AdapterReport",
    "AdapterStatus",
    "require_lossless",
]
