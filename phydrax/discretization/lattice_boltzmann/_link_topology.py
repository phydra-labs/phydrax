#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


BoundarySide: TypeAlias = Literal["lower", "upper"]
BoundaryFace: TypeAlias = tuple[str, BoundarySide]
FlowDirection: TypeAlias = Literal["any", "inlet", "outlet"]


class LatticeBoltzmannLinkOwner(IntEnum):
    """Exclusive owner of one destination-cell population."""

    LOCAL = 0
    PERIODIC = 1
    HALO = 2
    HALFWAY = 3
    BOUZIDI = 4
    VELOCITY = 5
    PRESSURE = 6
    CONVECTIVE = 7


class LatticeBoltzmannBoundaryStage(IntEnum):
    """Ordered stage at which a compiled owner may write its population."""

    STREAM = 0
    WALL = 1
    OPEN = 2


_OWNER_STAGE = {
    LatticeBoltzmannLinkOwner.LOCAL: LatticeBoltzmannBoundaryStage.STREAM,
    LatticeBoltzmannLinkOwner.PERIODIC: LatticeBoltzmannBoundaryStage.STREAM,
    LatticeBoltzmannLinkOwner.HALO: LatticeBoltzmannBoundaryStage.STREAM,
    LatticeBoltzmannLinkOwner.HALFWAY: LatticeBoltzmannBoundaryStage.WALL,
    LatticeBoltzmannLinkOwner.BOUZIDI: LatticeBoltzmannBoundaryStage.WALL,
    LatticeBoltzmannLinkOwner.VELOCITY: LatticeBoltzmannBoundaryStage.OPEN,
    LatticeBoltzmannLinkOwner.PRESSURE: LatticeBoltzmannBoundaryStage.OPEN,
    LatticeBoltzmannLinkOwner.CONVECTIVE: LatticeBoltzmannBoundaryStage.OPEN,
}


def owner_stage(owner: LatticeBoltzmannLinkOwner, /) -> LatticeBoltzmannBoundaryStage:
    if not isinstance(owner, LatticeBoltzmannLinkOwner):
        raise TypeError("owner must be a LatticeBoltzmannLinkOwner.")
    return _OWNER_STAGE[owner]


def _face(axis: str, side: BoundarySide, /) -> BoundaryFace:
    axis_ = str(axis)
    if not axis_ or side not in ("lower", "upper"):
        raise ValueError(
            "A boundary face requires a non-empty axis and lower/upper side."
        )
    return axis_, side


class LatticeBoltzmannFaceBoundary(StrictModule, NonTrainableState):
    """Typed ownership declaration for one exterior grid face."""

    axis: str = eqx.field(static=True)
    side: BoundarySide = eqx.field(static=True)
    owner: LatticeBoltzmannLinkOwner = eqx.field(static=True)
    parameter_id: str | None = eqx.field(static=True)
    body_id: str | None = eqx.field(static=True)
    link_fraction: float | None = eqx.field(static=True)
    flow_direction: FlowDirection = eqx.field(static=True)
    declaration_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: str,
        side: BoundarySide,
        owner: LatticeBoltzmannLinkOwner,
        /,
        *,
        parameter_id: str | None = None,
        body_id: str | None = None,
        link_fraction: float | None = None,
        flow_direction: FlowDirection = "any",
    ):
        axis_, side_ = _face(axis, side)
        if not isinstance(owner, LatticeBoltzmannLinkOwner):
            raise TypeError("owner must be a LatticeBoltzmannLinkOwner.")
        if owner is LatticeBoltzmannLinkOwner.LOCAL:
            raise ValueError("Exterior faces cannot declare LOCAL ownership.")
        parameter_owners = {
            LatticeBoltzmannLinkOwner.HALO,
            LatticeBoltzmannLinkOwner.VELOCITY,
            LatticeBoltzmannLinkOwner.PRESSURE,
            LatticeBoltzmannLinkOwner.CONVECTIVE,
        }
        parameter = None if parameter_id is None else str(parameter_id)
        if owner in parameter_owners:
            parameter = (
                f"{axis_}:{side_}:{owner.name.lower()}"
                if parameter is None
                else parameter
            )
            if not parameter:
                raise ValueError(
                    "Parameterized boundaries require a non-empty parameter_id."
                )
        elif parameter is not None:
            raise ValueError(f"{owner.name} faces do not accept parameter_id.")
        wall = owner in {
            LatticeBoltzmannLinkOwner.HALFWAY,
            LatticeBoltzmannLinkOwner.BOUZIDI,
        }
        body = None if body_id is None else str(body_id)
        if wall:
            body = f"wall:{axis_}:{side_}" if body is None else body
            if not body:
                raise ValueError("Wall faces require a non-empty body_id.")
        elif body is not None:
            raise ValueError("Only wall faces accept body_id.")
        fraction = None if link_fraction is None else float(link_fraction)
        if owner is LatticeBoltzmannLinkOwner.BOUZIDI:
            fraction = 0.5 if fraction is None else fraction
            if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                raise ValueError("Bouzidi face link_fraction must lie in (0, 1].")
        elif fraction is not None:
            raise ValueError("Only Bouzidi faces accept link_fraction.")
        if flow_direction not in ("any", "inlet", "outlet"):
            raise ValueError("flow_direction must be 'any', 'inlet', or 'outlet'.")
        if owner is not LatticeBoltzmannLinkOwner.VELOCITY and flow_direction != "any":
            raise ValueError("Only velocity faces accept a directional flow constraint.")
        self.axis = axis_
        self.side = side_
        self.owner = owner
        self.parameter_id = parameter
        self.body_id = body
        self.link_fraction = fraction
        self.flow_direction = flow_direction
        self.declaration_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-face-boundary",
                "axis": axis_,
                "side": side_,
                "owner": owner.name,
                "parameter": parameter,
                "body": body,
                "link_fraction": fraction,
                "flow_direction": flow_direction,
            }
        )

    @property
    def face(self) -> BoundaryFace:
        return self.axis, self.side


class LatticeBoltzmannBodyBoundary(StrictModule, NonTrainableState):
    """Typed ownership declaration for links terminating in one solid body."""

    body_id: str = eqx.field(static=True)
    owner: LatticeBoltzmannLinkOwner = eqx.field(static=True)
    declaration_id: str = eqx.field(static=True)

    def __init__(self, body_id: str, owner: LatticeBoltzmannLinkOwner, /):
        body = str(body_id)
        if not body:
            raise ValueError("body_id must be non-empty.")
        if owner not in (
            LatticeBoltzmannLinkOwner.HALFWAY,
            LatticeBoltzmannLinkOwner.BOUZIDI,
        ):
            raise ValueError("Solid bodies support HALFWAY or BOUZIDI ownership.")
        self.body_id = body
        self.owner = owner
        self.declaration_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-body-boundary",
                "body": body,
                "owner": owner.name,
            }
        )


class LatticeBoltzmannCornerRule(StrictModule, NonTrainableState):
    """Explicitly choose one face declaration at an intersecting-face cell."""

    faces: tuple[BoundaryFace, ...] = eqx.field(static=True)
    source_face: BoundaryFace = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        faces: Sequence[BoundaryFace],
        source_face: BoundaryFace,
        /,
    ):
        normalized = tuple(_face(axis, side) for axis, side in faces)
        if len(normalized) < 2 or len(set(normalized)) != len(normalized):
            raise ValueError("A corner rule requires at least two unique faces.")
        source = _face(*source_face)
        if source not in normalized:
            raise ValueError("Corner source_face must be one of the intersecting faces.")
        self.faces = normalized
        self.source_face = source
        self.rule_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-corner-rule",
                "faces": [list(face) for face in normalized],
                "source_face": list(source),
            }
        )


class LatticeBoltzmannBoundaryStageState(StrictModule):
    """Functional write-once population state for compiled boundary stages."""

    populations: Array
    written: Array

    def __init__(self, populations: ArrayLike, written: ArrayLike, /):
        values = jnp.asarray(populations)
        marks = jnp.asarray(written, dtype=bool)
        if values.shape != marks.shape:
            raise ValueError("Boundary stage populations and written mask must match.")
        self.populations = values
        self.written = marks


class CompiledLatticeBoltzmannLinkTopology(StrictModule, NonTrainableState):
    """Frozen owner, stage, parameter, normal, body, and fraction per population."""

    owner: Array
    stage: Array
    parameter_index: Array
    normal_axis: Array
    normal_sign: Array
    body_index: Array
    link_fraction: Array
    fluid_mask: Array
    topology_id: str = eqx.field(static=True)
    population_shape: tuple[int, ...] = eqx.field(static=True)
    owner_counts: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        owner: ArrayLike,
        parameter_index: ArrayLike,
        normal_axis: ArrayLike,
        normal_sign: ArrayLike,
        body_index: ArrayLike,
        link_fraction: ArrayLike,
        fluid_mask: ArrayLike,
        /,
        *,
        topology_id: str,
    ):
        owners = np.asarray(owner, dtype=np.int8)
        if owners.ndim < 2:
            raise ValueError(
                "Link ownership must have spatial axes and a trailing Q axis."
            )
        if np.any(owners < 0) or np.any(owners >= len(LatticeBoltzmannLinkOwner)):
            raise ValueError("Every population must have exactly one recognized owner.")
        shape = owners.shape
        parameter = np.asarray(parameter_index, dtype=np.int32)
        axes = np.asarray(normal_axis, dtype=np.int8)
        signs = np.asarray(normal_sign, dtype=np.int8)
        bodies = np.asarray(body_index, dtype=np.int32)
        fractions = np.asarray(link_fraction, dtype=np.float64)
        if any(
            value.shape != shape for value in (parameter, axes, signs, bodies, fractions)
        ):
            raise ValueError("All compiled link fields must have the population shape.")
        fluid = np.asarray(fluid_mask, dtype=bool)
        if fluid.shape != shape[:-1]:
            raise ValueError(
                "Topology fluid_mask must match the spatial population shape."
            )
        stages = np.asarray(
            [
                int(owner_stage(LatticeBoltzmannLinkOwner(int(value))))
                for value in owners.flat
            ],
            dtype=np.int8,
        ).reshape(shape)
        identifier = str(topology_id)
        if not identifier:
            raise ValueError("topology_id must be non-empty.")
        self.owner = jnp.asarray(owners, dtype=jnp.int8)
        self.stage = jnp.asarray(stages, dtype=jnp.int8)
        self.parameter_index = jnp.asarray(parameter, dtype=jnp.int32)
        self.normal_axis = jnp.asarray(axes, dtype=jnp.int8)
        self.normal_sign = jnp.asarray(signs, dtype=jnp.int8)
        self.body_index = jnp.asarray(bodies, dtype=jnp.int32)
        self.link_fraction = jnp.asarray(fractions)
        self.fluid_mask = jnp.asarray(fluid, dtype=bool)
        self.topology_id = identifier
        self.population_shape = shape
        self.owner_counts = tuple(
            int(np.count_nonzero(owners == int(candidate)))
            for candidate in LatticeBoltzmannLinkOwner
        )

    def begin(self, populations: ArrayLike, /) -> LatticeBoltzmannBoundaryStageState:
        values = jnp.asarray(populations)
        if values.shape != self.population_shape:
            raise ValueError("Boundary populations do not match the compiled topology.")
        return LatticeBoltzmannBoundaryStageState(
            values, jnp.zeros(values.shape, dtype=bool)
        )

    def commit(
        self,
        state: LatticeBoltzmannBoundaryStageState,
        candidate: ArrayLike,
        stage: LatticeBoltzmannBoundaryStage,
        owners: Sequence[LatticeBoltzmannLinkOwner],
        /,
    ) -> LatticeBoltzmannBoundaryStageState:
        if not isinstance(state, LatticeBoltzmannBoundaryStageState):
            raise TypeError("state must be a LatticeBoltzmannBoundaryStageState.")
        if not isinstance(stage, LatticeBoltzmannBoundaryStage):
            raise TypeError("stage must be a LatticeBoltzmannBoundaryStage.")
        owner_tuple = tuple(owners)
        if not owner_tuple or len(set(owner_tuple)) != len(owner_tuple):
            raise ValueError("A stage commit requires unique owners.")
        if any(not isinstance(owner, LatticeBoltzmannLinkOwner) for owner in owner_tuple):
            raise TypeError("Stage owners must be LatticeBoltzmannLinkOwner values.")
        if any(owner_stage(owner) is not stage for owner in owner_tuple):
            raise ValueError("A stage cannot commit an owner assigned to another stage.")
        values = jnp.asarray(candidate, dtype=state.populations.dtype)
        if values.shape != self.population_shape:
            raise ValueError("Boundary candidate does not match the compiled topology.")
        selected = jnp.zeros(self.population_shape, dtype=bool)
        for owner in owner_tuple:
            selected = selected | (self.owner == int(owner))
        checked = eqx.error_if(
            state.populations,
            jnp.any(selected & state.written),
            "A compiled population owner may write exactly once.",
        )
        populations = jnp.where(selected, values, checked)
        written = state.written | selected
        return LatticeBoltzmannBoundaryStageState(populations, written)

    def finish(self, state: LatticeBoltzmannBoundaryStageState, /) -> Array:
        if not isinstance(state, LatticeBoltzmannBoundaryStageState):
            raise TypeError("state must be a LatticeBoltzmannBoundaryStageState.")
        return eqx.error_if(
            state.populations,
            jnp.any(~state.written),
            "Every population must be written by exactly one compiled owner.",
        )


__all__ = [
    "BoundaryFace",
    "BoundarySide",
    "CompiledLatticeBoltzmannLinkTopology",
    "FlowDirection",
    "LatticeBoltzmannBodyBoundary",
    "LatticeBoltzmannBoundaryStage",
    "LatticeBoltzmannBoundaryStageState",
    "LatticeBoltzmannCornerRule",
    "LatticeBoltzmannFaceBoundary",
    "LatticeBoltzmannLinkOwner",
    "owner_stage",
]
