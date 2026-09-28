#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Shared nominal extents of the averaged FEL contracts."""

from __future__ import annotations

from ....typing import Dim


class FELSliceDim(Dim, minimum=1):
    """Electron beam slices."""


class FELRecordDim(Dim, minimum=2):
    """Recorded longitudinal positions (entrance plus every step end)."""


class FELHarmonicDim(Dim, minimum=1):
    """Modeled radiation harmonics."""


class FELWindowDim(Dim, minimum=2):
    """Time-dependent radiation window slots (head padding plus beam slices)."""


class FELFrequencyDim(Dim, minimum=2):
    """Discrete angular-frequency samples of a time-dependent window spectrum."""


__all__ = [
    "FELFrequencyDim",
    "FELHarmonicDim",
    "FELRecordDim",
    "FELSliceDim",
    "FELWindowDim",
]
