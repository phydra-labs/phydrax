#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


BoundaryTarget = Callable[[Array, Array, Array, Array, Any], ArrayLike]


def _canonical_ale_identity(value: object, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical stripped string.")
    return value


def _dynamic_geometry_version(value: ArrayLike, /) -> Array:
    version = jnp.asarray(value)
    if version.shape != () or version.dtype.kind not in "iu":
        raise ValueError("geometry_version must be a scalar integer.")
    version = eqx.error_if(
        version,
        (version < 0) | (version > jnp.iinfo(jnp.int32).max),
        "geometry_version must be nonnegative and representable as int32.",
    )
    return version.astype(jnp.int32)


class ALEBoundaryContext(StrictModule, NonTrainableState):
    """Immutable ALE face geometry, stage route, and wall kinematic evidence."""

    face_point: Array
    outward_normal: Array
    quadrature_grid_velocity: Array
    wall_velocity: Array
    time: Array
    args: Any
    grid_normal_velocity: Array
    wall_normal_velocity: Array
    kinematic_defect: Array
    kinematic_tolerance: Array
    kinematics_consistent: Array
    geometry_version: Array
    topology_epoch_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    face_block_id: str = eqx.field(static=True)
    motion_plan_id: str = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        face_point: ArrayLike,
        outward_normal: ArrayLike,
        quadrature_grid_velocity: ArrayLike,
        wall_velocity: ArrayLike,
        time: ArrayLike,
        args: Any,
        topology_epoch_id: str,
        geometry_layout_id: str,
        geometry_version: ArrayLike,
        face_block_id: str,
        motion_plan_id: str,
        absolute_tolerance: float,
        relative_tolerance: float,
    ):
        topology_id = _canonical_ale_identity(topology_epoch_id, "topology_epoch_id")
        layout_id = _canonical_ale_identity(geometry_layout_id, "geometry_layout_id")
        version = _dynamic_geometry_version(geometry_version)
        block_id = _canonical_ale_identity(face_block_id, "face_block_id")
        motion_id = _canonical_ale_identity(motion_plan_id, "motion_plan_id")
        point = jnp.asarray(face_point)
        normal = jnp.asarray(outward_normal)
        grid_velocity = jnp.asarray(quadrature_grid_velocity)
        wall_velocity_ = jnp.asarray(wall_velocity)
        if (
            point.ndim == 0
            or point.shape[-1] == 0
            or normal.shape != point.shape
            or grid_velocity.shape != point.shape
            or wall_velocity_.shape != point.shape
        ):
            raise ValueError(
                "ALE boundary point, normal, grid velocity, and wall velocity must "
                "have the same non-scalar shape."
            )
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not math.isfinite(absolute)
            or not math.isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
        ):
            raise ValueError("ALE boundary tolerances must be finite and nonnegative.")
        time_ = jnp.asarray(time)
        if time_.shape != ():
            raise ValueError("ALE boundary time must be a scalar.")
        point = eqx.error_if(
            point,
            jnp.any(~jnp.isfinite(point))
            | jnp.any(~jnp.isfinite(normal))
            | jnp.any(~jnp.isfinite(grid_velocity))
            | jnp.any(~jnp.isfinite(wall_velocity_))
            | ~jnp.isfinite(time_),
            "ALE boundary context values must be finite.",
        )
        point = eqx.error_if(
            point,
            jnp.any(oe.contract("...i,...i->...", normal, normal) <= 0.0),
            "ALE boundary normals must be nonzero.",
        )
        grid_normal = oe.contract("...i,...i->...", grid_velocity, normal)
        wall_normal = oe.contract("...i,...i->...", wall_velocity_, normal)
        defect = jnp.abs(wall_normal - grid_normal)
        tolerance = absolute + relative * jnp.maximum(
            jnp.abs(wall_normal), jnp.abs(grid_normal)
        )

        self.face_point = point
        self.outward_normal = normal
        self.quadrature_grid_velocity = grid_velocity
        self.wall_velocity = wall_velocity_
        self.time = time_
        self.args = args
        self.grid_normal_velocity = grid_normal
        self.wall_normal_velocity = wall_normal
        self.kinematic_defect = defect
        self.kinematic_tolerance = tolerance
        self.kinematics_consistent = defect <= tolerance
        self.geometry_version = version
        self.topology_epoch_id = topology_id
        self.geometry_layout_id = layout_id
        self.face_block_id = block_id
        self.motion_plan_id = motion_id
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative

    def validate_consumer_identity(
        self,
        value: ArrayLike,
        /,
        *,
        topology_epoch_id: str,
        geometry_layout_id: str,
        geometry_version: ArrayLike,
        face_block_id: str,
        motion_plan_id: str,
    ) -> Array:
        """Bind a consumed value to this context's exact stage route."""
        expected_static = (
            (
                self.topology_epoch_id,
                _canonical_ale_identity(topology_epoch_id, "topology_epoch_id"),
                "topology_epoch_id",
            ),
            (
                self.geometry_layout_id,
                _canonical_ale_identity(geometry_layout_id, "geometry_layout_id"),
                "geometry_layout_id",
            ),
            (
                self.face_block_id,
                _canonical_ale_identity(face_block_id, "face_block_id"),
                "face_block_id",
            ),
            (
                self.motion_plan_id,
                _canonical_ale_identity(motion_plan_id, "motion_plan_id"),
                "motion_plan_id",
            ),
        )
        for actual, expected, name in expected_static:
            if actual != expected:
                raise ValueError(
                    f"ALE boundary context {name} does not match its consumer."
                )
        expected_version = _dynamic_geometry_version(geometry_version)
        return eqx.error_if(
            jnp.asarray(value),
            self.geometry_version != expected_version,
            "ALE boundary context geometry_version does not match its consumer.",
        )


def _require_axis_aligned_ale_normal(
    context: ALEBoundaryContext,
    axis: int,
    value: ArrayLike,
    /,
) -> Array:
    """Bind a consumed value to a supported coordinate-axis face normal."""
    if not isinstance(context, ALEBoundaryContext):
        raise TypeError("ALE exterior states require an ALEBoundaryContext.")
    if not 0 <= int(axis) < context.outward_normal.shape[-1]:
        raise ValueError("ALE boundary axis is out of range.")
    normal = context.outward_normal
    transverse = normal.at[..., axis].set(0.0)
    scale = jnp.max(jnp.abs(normal), axis=-1)
    tolerance = context.absolute_tolerance + context.relative_tolerance * scale
    return eqx.error_if(
        jnp.asarray(value),
        jnp.any(jnp.max(jnp.abs(transverse), axis=-1) > tolerance),
        "Axis-based ALE boundary policy does not support oblique face normals.",
    )


def _validate_ale_boundary_context(
    system: Any,
    interior: Array,
    context: ALEBoundaryContext,
    axis: int,
    /,
) -> None:
    if not isinstance(context, ALEBoundaryContext):
        raise TypeError("ALE exterior states require an ALEBoundaryContext.")
    state = jnp.asarray(interior)
    if state.ndim == 0:
        raise ValueError("ALE boundary interior state must have a component axis.")
    dimension = int(system.dimension)
    if context.face_point.shape[-1] != dimension:
        raise ValueError("ALE boundary context must match the system dimension.")
    if context.face_point.shape[:-1] != state.shape[:-1]:
        raise ValueError("ALE boundary context and interior state batches must match.")
    if not 0 <= int(axis) < dimension:
        raise ValueError("ALE boundary axis is out of range.")


def _static_ale_exterior_state(
    boundary: AbstractFiniteVolumeBoundary,
    system: Any,
    interior: Array,
    context: ALEBoundaryContext,
    axis: int,
    /,
) -> Array:
    _validate_ale_boundary_context(system, interior, context, axis)
    state = eqx.error_if(
        jnp.asarray(interior),
        jnp.any(jnp.abs(context.grid_normal_velocity) > context.kinematic_tolerance)
        | jnp.any(jnp.abs(context.wall_normal_velocity) > context.kinematic_tolerance),
        "Static ALE boundaries require zero grid-normal and wall-normal velocity; "
        "use MovingSlipWallBoundary for nonzero conforming motion.",
    )
    return boundary.exterior_state(
        system,
        context.time,
        state,
        context.face_point,
        context.outward_normal,
        axis,
        context.args,
    )


def _boundary_value(value: ArrayLike, shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(value)
    if array.shape == () or array.shape == (shape[-1],):
        return jnp.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError(
            f"Boundary state must have shape {shape}, scalar, or components."
        )
    return array


class AbstractFiniteVolumeBoundary(StrictModule, NonTrainableState):
    """Physical boundary policy that constructs an exterior face state."""

    boundary_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        """Construct an exterior state from explicit ALE face kinematics."""
        raise NotImplementedError


class ExtrapolationBoundary(AbstractFiniteVolumeBoundary):
    """Zero-normal-gradient exterior state."""

    def __init__(self):
        self.boundary_id = canonical_fingerprint({"kind": "fv-extrapolation"})

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del system, time, coordinates, outward_normal, axis, args
        return jnp.asarray(interior)

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        return _static_ale_exterior_state(self, system, interior, context, axis)


class ConstantStateBoundary(AbstractFiniteVolumeBoundary):
    """Constant exterior conservative state."""

    value: Array

    def __init__(self, value: ArrayLike, /):
        value_ = jnp.asarray(value)
        if value_.ndim > 1:
            raise ValueError(
                "Constant boundary state must be scalar or component vector."
            )
        self.value = value_
        self.boundary_id = canonical_fingerprint(
            {"kind": "fv-constant-state", "value": array_tree_fingerprint(value_)}
        )

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del system, time, coordinates, outward_normal, axis, args
        return _boundary_value(self.value, interior.shape)

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        return _static_ale_exterior_state(self, system, interior, context, axis)


class PrescribedStateBoundary(AbstractFiniteVolumeBoundary):
    """Time-, state-, coordinate-, and parameter-dependent exterior state."""

    target: BoundaryTarget = eqx.field(static=True)

    def __init__(self, target: BoundaryTarget, /, *, boundary_id: str):
        if not callable(target):
            raise TypeError("target must be callable.")
        identifier = str(boundary_id)
        if not identifier:
            raise ValueError("boundary_id must be non-empty.")
        self.target = target
        self.boundary_id = identifier

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del system, axis
        value = self.target(time, interior, coordinates, outward_normal, args)
        return _boundary_value(value, interior.shape)

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        return _static_ale_exterior_state(self, system, interior, context, axis)


class ReflectiveBoundary(AbstractFiniteVolumeBoundary):
    """Equation-owned reflective state transformation."""

    def __init__(self):
        self.boundary_id = canonical_fingerprint({"kind": "fv-reflective"})

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del time, coordinates, outward_normal, args
        return system.reflect_state(interior, axis)

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        aligned_interior = _require_axis_aligned_ale_normal(context, axis, interior)
        return _static_ale_exterior_state(self, system, aligned_interior, context, axis)


class PrescribedNormalFluxBoundary(AbstractFiniteVolumeBoundary):
    """Direct outward integrated-flux-density policy."""

    target: BoundaryTarget = eqx.field(static=True)

    def __init__(self, target: BoundaryTarget, /, *, boundary_id: str):
        if not callable(target):
            raise TypeError("target must be callable.")
        identifier = str(boundary_id)
        if not identifier:
            raise ValueError("boundary_id must be non-empty.")
        self.target = target
        self.boundary_id = identifier

    def exterior_state(
        self,
        system: Any,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        del system, time, interior, coordinates, outward_normal, axis, args
        raise ValueError(
            "PrescribedNormalFluxBoundary supplies flux, not exterior state."
        )

    def ale_exterior_state(
        self,
        system: Any,
        interior: Array,
        context: ALEBoundaryContext,
        axis: int,
        /,
    ) -> Array:
        del system, interior, context, axis
        raise ValueError(
            "PrescribedNormalFluxBoundary supplies flux, not an ALE exterior state."
        )

    def normal_flux(
        self,
        time: Array,
        interior: Array,
        coordinates: Array,
        outward_normal: Array,
        args: Any,
        /,
    ) -> Array:
        value = self.target(time, interior, coordinates, outward_normal, args)
        return _boundary_value(value, interior.shape)


class FiniteVolumeBoundaryPair(StrictModule, NonTrainableState):
    """Lower and upper physical boundaries for one bounded axis."""

    lower: AbstractFiniteVolumeBoundary
    upper: AbstractFiniteVolumeBoundary
    pair_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: AbstractFiniteVolumeBoundary,
        upper: AbstractFiniteVolumeBoundary,
        /,
    ):
        if not isinstance(lower, AbstractFiniteVolumeBoundary) or not isinstance(
            upper, AbstractFiniteVolumeBoundary
        ):
            raise TypeError("Boundary pairs require finite-volume boundary policies.")
        self.lower = lower
        self.upper = upper
        self.pair_id = canonical_fingerprint(
            {
                "kind": "fv-boundary-pair",
                "lower": lower.boundary_id,
                "upper": upper.boundary_id,
            }
        )


class FiniteVolumeBoundarySet(StrictModule, NonTrainableState):
    """Axis-ordered bounded policies; periodic axes use ``None``."""

    axis_names: tuple[str, ...] = eqx.field(static=True)
    pairs: tuple[FiniteVolumeBoundaryPair | None, ...]
    boundary_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis_names: Sequence[str],
        pairs: Sequence[FiniteVolumeBoundaryPair | None],
        /,
    ):
        names = tuple(str(name) for name in axis_names)
        pairs_ = tuple(pairs)
        if (
            not names
            or len(names) != len(pairs_)
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Boundary axes and pairs must align with unique names.")
        if any(
            pair is not None and not isinstance(pair, FiniteVolumeBoundaryPair)
            for pair in pairs_
        ):
            raise TypeError("Boundary entries must be FiniteVolumeBoundaryPair or None.")
        self.axis_names = names
        self.pairs = pairs_
        self.boundary_set_id = canonical_fingerprint(
            {
                "kind": "fv-boundary-set",
                "axes": list(names),
                "pairs": [None if pair is None else pair.pair_id for pair in pairs_],
            }
        )

    @classmethod
    def periodic(cls, axis_names: Sequence[str], /) -> "FiniteVolumeBoundarySet":
        names = tuple(axis_names)
        return cls(names, (None,) * len(names))


__all__ = [
    "ALEBoundaryContext",
    "AbstractFiniteVolumeBoundary",
    "ConstantStateBoundary",
    "ExtrapolationBoundary",
    "FiniteVolumeBoundaryPair",
    "FiniteVolumeBoundarySet",
    "PrescribedNormalFluxBoundary",
    "PrescribedStateBoundary",
    "ReflectiveBoundary",
]
