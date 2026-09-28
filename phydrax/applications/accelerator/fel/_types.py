#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared nominal extents of the averaged FEL contracts."""

from __future__ import annotations

from ....typing import Dim


class FELSliceDim(Dim, minimum=1):
    """Independent time-independent beam slices."""


class FELRecordDim(Dim, minimum=2):
    """Recorded longitudinal positions (entrance plus every step end)."""


class FELHarmonicDim(Dim, minimum=1):
    """Modeled radiation harmonics."""


__all__ = ["FELHarmonicDim", "FELRecordDim", "FELSliceDim"]
