#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nominal atomistic extents; scientific axis identity remains port-owned."""

from ..typing import Dim


class AtomisticCaseDim(Dim):
    pass


class AtomisticAtomDim(Dim):
    pass


class AtomisticNodeDim(Dim):
    pass


__all__ = ["AtomisticAtomDim", "AtomisticCaseDim", "AtomisticNodeDim"]
