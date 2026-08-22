#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._tensor_support import GridLocation, PreparedTensorGrid


StencilBias: TypeAlias = Literal["centered", "forward", "backward"]
BoundaryClosureKind: TypeAlias = Literal["periodic", "one_sided"]
GridRegionKind: TypeAlias = Literal[
    "interior",
    "physical_boundary",
    "owned",
    "halo",
    "coarse_fine",
    "full",
]


class GridRegion(StrictModule, NonTrainableState):
    """Semantic execution region distinct from its physical boundary condition."""

    kind: GridRegionKind = eqx.field(static=True)
    region_id: str = eqx.field(static=True)

    def __init__(self, kind: GridRegionKind, /, *, region_id: str | None = None):
        if kind not in (
            "interior",
            "physical_boundary",
            "owned",
            "halo",
            "coarse_fine",
            "full",
        ):
            raise ValueError("Unknown grid region kind.")
        identifier = (
            canonical_fingerprint({"kind": "grid-region", "region": kind})
            if region_id is None
            else str(region_id)
        )
        if not identifier:
            raise ValueError("region_id must be non-empty.")
        self.kind = kind
        self.region_id = identifier


class DerivativeRequest(StrictModule, NonTrainableState):
    """Numerical derivative request with exact source and target locations."""

    name: str = eqx.field(static=True)
    axis: str = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)
    accuracy_order: int = eqx.field(static=True)
    bias: StencilBias = eqx.field(static=True)
    boundary: BoundaryClosureKind = eqx.field(static=True)
    source_location: GridLocation
    target_location: GridLocation
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        grid: PreparedTensorGrid,
        axis: str,
        /,
        *,
        derivative_order: int = 1,
        accuracy_order: int = 2,
        bias: StencilBias = "centered",
        boundary: BoundaryClosureKind | None = None,
        source_location: GridLocation | None = None,
        target_location: GridLocation | None = None,
        request_id: str | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        name_ = str(name)
        axis_ = str(axis)
        if not name_ or axis_ not in grid.axis_names:
            raise ValueError("Derivative name must be non-empty and axis must exist.")
        derivative = int(derivative_order)
        accuracy = int(accuracy_order)
        if derivative <= 0 or accuracy <= 0:
            raise ValueError("Derivative and accuracy orders must be positive.")
        if bias not in ("centered", "forward", "backward"):
            raise ValueError("Unknown stencil bias.")
        axis_index = grid.axis_names.index(axis_)
        boundary_ = (
            ("periodic" if grid.axes[axis_index].periodic else "one_sided")
            if boundary is None
            else boundary
        )
        if boundary_ not in ("periodic", "one_sided"):
            raise ValueError("Unknown boundary closure kind.")
        if boundary_ == "periodic" and not grid.axes[axis_index].periodic:
            raise ValueError("Periodic closure requires a periodic axis.")
        source = grid.centered_location if source_location is None else source_location
        target = source if target_location is None else target_location
        if (
            not isinstance(source, GridLocation)
            or not isinstance(target, GridLocation)
            or source.axis_names != grid.axis_names
            or target.axis_names != grid.axis_names
        ):
            raise ValueError("Derivative locations must belong to the prepared grid.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "derivative-request",
                    "name": name_,
                    "grid": grid.prepared_id,
                    "axis": axis_,
                    "derivative_order": derivative,
                    "accuracy_order": accuracy,
                    "bias": bias,
                    "boundary": boundary_,
                    "source_location": source.location_id,
                    "target_location": target.location_id,
                }
            )
            if request_id is None
            else str(request_id)
        )
        if not identifier:
            raise ValueError("request_id must be non-empty.")
        self.name = name_
        self.axis = axis_
        self.derivative_order = derivative
        self.accuracy_order = accuracy
        self.bias = bias
        self.boundary = boundary_
        self.source_location = source
        self.target_location = target
        self.request_id = identifier


__all__ = [
    "BoundaryClosureKind",
    "DerivativeRequest",
    "GridRegion",
    "GridRegionKind",
    "StencilBias",
]
