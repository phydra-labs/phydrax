#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import equinox as eqx
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PreparedTensorGrid


PulseTimeTopology = Literal["finite-window", "periodic-cell"]


class PulseTimeSpace(StrictModule, NonTrainableState):
    """One sampled pulse-time axis with an explicit physical topology.

    A periodic cell is an endpoint-excluded Fourier grid and is the only topology
    on which exact carrier-bin shifts are defined. A finite window is a
    nonperiodic, endpoint-including uniform point grid. The prepared tensor grid
    remains the sole owner of coordinates and quadrature weights.
    """

    temporal_grid: PreparedTensorGrid
    topology: PulseTimeTopology = eqx.field(static=True)
    space_id: str = eqx.field(static=True)

    def __init__(
        self,
        temporal_grid: PreparedTensorGrid,
        /,
        *,
        topology: PulseTimeTopology,
    ):
        if not isinstance(temporal_grid, PreparedTensorGrid):
            raise TypeError("temporal_grid must be a PreparedTensorGrid.")
        if len(temporal_grid.shape) != 1:
            raise ValueError("PulseTimeSpace requires an exactly one-dimensional grid.")
        if topology not in ("finite-window", "periodic-cell"):
            raise ValueError("topology must be 'finite-window' or 'periodic-cell'.")
        axis = temporal_grid.axes[0]
        if axis.primary_entity != "point":
            raise ValueError("Pulse time requires a point-primary temporal grid.")
        if topology == "periodic-cell":
            if axis.basis != "fourier" or not axis.periodic:
                raise ValueError(
                    "periodic-cell pulse time requires a periodic Fourier axis."
                )
        elif axis.periodic or axis.basis != "uniform":
            raise ValueError(
                "finite-window pulse time requires a nonperiodic uniform axis."
            )
        elif not axis.lower_endpoint_included or not axis.upper_endpoint_included:
            raise ValueError("finite-window pulse time requires both physical endpoints.")
        nodes = np.asarray(axis.nodes, dtype=np.float64)
        if nodes.size < 2:
            raise ValueError("Pulse time requires at least two samples.")
        differences = np.diff(nodes)
        spacing = float(differences[0])
        tolerance = 64.0 * np.finfo(nodes.dtype).eps * max(1.0, abs(spacing))
        if (
            not np.isfinite(spacing)
            or spacing <= 0.0
            or not np.all(np.isfinite(differences))
            or not np.allclose(differences, spacing, rtol=1.0e-10, atol=tolerance)
        ):
            raise ValueError("Pulse-time coordinates must be finite and uniform.")
        self.temporal_grid = temporal_grid
        self.topology = topology
        self.space_id = canonical_fingerprint(
            {
                "kind": "pulse-time-space",
                "temporal_grid": temporal_grid.prepared_id,
                "topology": topology,
            }
        )

    @property
    def shape(self) -> tuple[int]:
        return self.temporal_grid.shape  # type: ignore[return-value]

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def coordinates(self) -> Array:
        return self.temporal_grid.axes[0].nodes

    @property
    def weights(self) -> Array:
        return self.temporal_grid.quadrature_weights

    @property
    def sample_spacing(self) -> float:
        nodes = np.asarray(self.coordinates, dtype=np.float64)
        return float(nodes[1] - nodes[0])

    @property
    def period(self) -> float:
        if self.topology != "periodic-cell":
            raise ValueError("Only periodic-cell pulse time has a period.")
        bounds = self.temporal_grid.axes[0].bounds
        if bounds is None:
            raise RuntimeError("A periodic pulse-time axis must have finite bounds.")
        values = np.asarray(bounds, dtype=np.float64)
        return float(values[1] - values[0])


__all__ = ["PulseTimeSpace", "PulseTimeTopology"]
