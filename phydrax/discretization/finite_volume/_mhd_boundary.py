#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


MHDBoundarySide: TypeAlias = Literal["lower", "upper"]


class MHDBoundaryTrace(StrictModule):
    exterior_state: Array
    boundary_electromotive: Array
    material_normal_flux: Array
    poynting_normal_flux: Array


class AbstractConstrainedMHDBoundary(StrictModule, NonTrainableState):
    boundary_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def trace(
        self,
        system: Any,
        interior_state: Array,
        normal_field: Array,
        axis: int,
        side: MHDBoundarySide,
        time: Array,
        args: Any = None,
        /,
    ) -> MHDBoundaryTrace:
        raise NotImplementedError


class PerfectlyConductingWallBoundary(AbstractConstrainedMHDBoundary):
    """Stationary slip wall with frozen normal B and vanishing tangential E."""

    def __init__(self):
        self.boundary_id = canonical_fingerprint({"kind": "conducting-mhd-wall"})

    def trace(
        self,
        system: Any,
        interior_state: Array,
        normal_field: Array,
        axis: int,
        side: MHDBoundarySide,
        time: Array,
        args: Any = None,
        /,
    ) -> MHDBoundaryTrace:
        del side, time, args
        primitive = system.conserved_to_primitive(interior_state)
        exterior = primitive.at[..., 1 + int(axis)].multiply(-1.0)
        exterior = exterior.at[..., 5 + int(axis)].set(normal_field)
        state = system.primitive_to_conserved(exterior)
        shape = interior_state.shape[:-1]
        return MHDBoundaryTrace(
            exterior_state=state,
            boundary_electromotive=jnp.zeros(shape + (3,), dtype=state.dtype),
            material_normal_flux=jnp.zeros(shape, dtype=state.dtype),
            poynting_normal_flux=jnp.zeros(shape, dtype=state.dtype),
        )


class MHDOutflowBoundary(AbstractConstrainedMHDBoundary):
    """Zero-gradient outflow with incoming normal velocity suppression."""

    allow_inflow: bool = eqx.field(static=True)

    def __init__(self, *, allow_inflow: bool = False):
        self.allow_inflow = bool(allow_inflow)
        self.boundary_id = canonical_fingerprint(
            {"kind": "mhd-outflow", "allow_inflow": self.allow_inflow}
        )

    def trace(
        self,
        system: Any,
        interior_state: Array,
        normal_field: Array,
        axis: int,
        side: MHDBoundarySide,
        time: Array,
        args: Any = None,
        /,
    ) -> MHDBoundaryTrace:
        del time, args
        primitive = system.conserved_to_primitive(interior_state)
        normal_velocity = primitive[..., 1 + int(axis)]
        outward = normal_velocity if side == "upper" else -normal_velocity
        corrected = (
            normal_velocity
            if self.allow_inflow
            else jnp.where(outward >= 0.0, normal_velocity, 0.0)
        )
        primitive = primitive.at[..., 1 + int(axis)].set(corrected)
        primitive = primitive.at[..., 5 + int(axis)].set(normal_field)
        state = system.primitive_to_conserved(primitive)
        velocity = primitive[..., 1:4]
        magnetic = primitive[..., 5:8]
        electric = -jnp.cross(velocity, magnetic)
        flux = system.physical_flux(state, int(axis))
        return MHDBoundaryTrace(
            exterior_state=state,
            boundary_electromotive=electric,
            material_normal_flux=flux[..., 0],
            poynting_normal_flux=flux[..., 4]
            - corrected * (state[..., 4] + system.pressure(state)),
        )


class PrescribedMHDInflowBoundary(AbstractConstrainedMHDBoundary):
    primitive_state: Array

    def __init__(self, primitive_state: ArrayLike, /):
        primitive = np.asarray(primitive_state, dtype=float)
        if primitive.shape[-1:] != (8,) or np.any(~np.isfinite(primitive)):
            raise ValueError("Prescribed MHD inflow primitive state is invalid.")
        self.primitive_state = jnp.asarray(primitive)
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "prescribed-mhd-inflow",
                "primitive": array_tree_fingerprint(primitive),
            }
        )

    def trace(
        self,
        system: Any,
        interior_state: Array,
        normal_field: Array,
        axis: int,
        side: MHDBoundarySide,
        time: Array,
        args: Any = None,
        /,
    ) -> MHDBoundaryTrace:
        del side, time, args
        primitive = jnp.broadcast_to(self.primitive_state, interior_state.shape)
        primitive = primitive.at[..., 5 + int(axis)].set(normal_field)
        state = system.primitive_to_conserved(primitive)
        velocity = primitive[..., 1:4]
        magnetic = primitive[..., 5:8]
        electric = -jnp.cross(velocity, magnetic)
        flux = system.physical_flux(state, int(axis))
        return MHDBoundaryTrace(
            exterior_state=state,
            boundary_electromotive=electric,
            material_normal_flux=flux[..., 0],
            poynting_normal_flux=flux[..., 4]
            - velocity[..., int(axis)] * (state[..., 4] + system.pressure(state)),
        )


class ConstrainedMHDBoundarySet(StrictModule, NonTrainableState):
    boundaries: tuple[
        tuple[AbstractConstrainedMHDBoundary, AbstractConstrainedMHDBoundary], ...
    ]
    axis_names: tuple[str, ...] = eqx.field(static=True)
    boundary_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: tuple[str, ...],
        boundaries: Mapping[
            str,
            tuple[AbstractConstrainedMHDBoundary, AbstractConstrainedMHDBoundary],
        ],
        /,
    ):
        names = tuple(str(name) for name in axis_names)
        supplied = dict(boundaries)
        if set(supplied) != set(names):
            raise ValueError("MHD boundary set must define both sides of every axis.")
        values = tuple(supplied[name] for name in names)
        if any(
            len(pair) != 2
            or any(
                not isinstance(boundary, AbstractConstrainedMHDBoundary)
                for boundary in pair
            )
            for pair in values
        ):
            raise TypeError("MHD boundary pairs contain invalid policies.")
        self.boundaries = values
        self.axis_names = names
        self.boundary_set_id = canonical_fingerprint(
            {
                "kind": "constrained-mhd-boundary-set",
                "axis_names": list(names),
                "boundaries": [
                    [lower.boundary_id, upper.boundary_id] for lower, upper in values
                ],
            }
        )

    def boundary(
        self,
        axis: int,
        side: MHDBoundarySide,
        /,
    ) -> AbstractConstrainedMHDBoundary:
        pair = self.boundaries[int(axis)]
        return pair[0] if side == "lower" else pair[1]


__all__ = [
    "AbstractConstrainedMHDBoundary",
    "ConstrainedMHDBoundarySet",
    "MHDBoundarySide",
    "MHDBoundaryTrace",
    "MHDOutflowBoundary",
    "PerfectlyConductingWallBoundary",
    "PrescribedMHDInflowBoundary",
]
