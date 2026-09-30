#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array

from ..exterior._complex import AbstractDeRhamComplex, ComplexBoundary
from ._topology import CellComplexTopology


class AbstractCellDeRhamComplex(AbstractDeRhamComplex):
    """A de Rham realization carrying canonical cell topology and boundaries."""

    topology: eqx.AbstractVar[CellComplexTopology]
    boundary_masks: eqx.AbstractVar[tuple[Array, ...]]

    @abstractmethod
    def active_indices(
        self, degree: int, /, *, boundary: ComplexBoundary = "absolute"
    ) -> Array:
        """Return realization DOF indices, not topology entity indices."""
        raise NotImplementedError


__all__ = ["AbstractCellDeRhamComplex"]
