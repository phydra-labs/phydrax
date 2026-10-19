#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nominal unstructured FV geometry consumed by reconstruction and side traces."""

from abc import abstractmethod

import equinox as eqx
from jax import Array

from .._cell_complex import (
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from .._hexahedral import HexahedralConnectivity
from .._lifecycle import AbstractPreparedDiscretization
from .._spaces import DiscreteFieldSpace
from .._topology import CellComplexTopology


class AbstractUnstructuredFiniteVolumeDiscretization(AbstractPreparedDiscretization):
    """Expose prepared cell-average geometry without depending on its preparation owner.

    Abstract declarations add no stored fields. Implementations bind the arrays,
    spaces, and counts to one prepared discretization identity.
    """

    vertices: eqx.AbstractVar[Array]
    topology: eqx.AbstractVar[CellComplexTopology]
    topology_id: eqx.AbstractVar[str]
    geometry_id: eqx.AbstractVar[str]
    connectivity: eqx.AbstractVar[
        PolygonalConnectivity
        | TetrahedralConnectivity
        | HexahedralConnectivity
        | PolyhedralConnectivity
    ]
    cell_space: eqx.AbstractVar[DiscreteFieldSpace]
    component_names: eqx.AbstractVar[tuple[str, ...]]
    cell_dimension: eqx.AbstractVar[int]
    cell_volumes: eqx.AbstractVar[Array]
    cell_centers: eqx.AbstractVar[Array]
    cell_quadrature_points: eqx.AbstractVar[Array]
    cell_quadrature_weights: eqx.AbstractVar[Array]
    cell_quadrature_degree: eqx.AbstractVar[int]
    owner_cells: eqx.AbstractVar[Array]
    neighbor_cells: eqx.AbstractVar[Array]
    face_centers: eqx.AbstractVar[Array]

    @property
    @abstractmethod
    def cell_count(self) -> int:
        """Return the extent of the prepared cell-average axis."""
        raise NotImplementedError

    @property
    @abstractmethod
    def component_count(self) -> int:
        """Return the number of named physical components per cell."""
        raise NotImplementedError

    @property
    @abstractmethod
    def state_shape(self) -> tuple[int, ...]:
        """Return the cell-average coefficient shape."""
        raise NotImplementedError
