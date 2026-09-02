#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._precision import FiniteVolumePrecisionPolicy
from ._reconstruction import PiecewiseConstantReconstruction, reconstruct_ghosted_axis


if TYPE_CHECKING:
    from ...equations._hyperbolic_systems import ShallowWaterSystem


class ShallowWaterWetDryPolicy(StrictModule, NonTrainableState):
    """Physical wet/dry thresholds without a cell-average thin film."""

    wet_depth: float = eqx.field(static=True)
    velocity_depth: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        wet_depth: float = 1e-10,
        velocity_depth: float = 1e-10,
    ):
        wet = float(wet_depth)
        velocity = float(velocity_depth)
        if (
            not np.isfinite(wet)
            or not np.isfinite(velocity)
            or wet < 0.0
            or velocity < 0.0
        ):
            raise ValueError(
                "Shallow-water wet/dry thresholds must be finite and nonnegative."
            )
        self.wet_depth = wet
        self.velocity_depth = velocity
        self.policy_id = canonical_fingerprint(
            {
                "kind": "shallow-water-wet-dry",
                "wet_depth": wet,
                "velocity_depth": velocity,
                "dry_momentum": "zero",
            }
        )

    def wet(self, depth: ArrayLike, /) -> Array:
        return jnp.asarray(depth) > self.wet_depth

    def velocity(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        depth = value[..., :1]
        threshold = max(self.wet_depth, self.velocity_depth)
        wet = depth > threshold
        safe_depth = jnp.where(wet, depth, 1.0)
        velocity = value[..., 1:] / safe_depth
        return jnp.where(wet, velocity, 0.0)

    def enforce_dry_momentum(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        depth = value[..., :1]
        momentum = jnp.where(self.wet(depth), value[..., 1:], 0.0)
        return jnp.concatenate((depth, momentum), axis=-1)


class ShallowWaterBathymetryPlan(StrictModule, NonTrainableState):
    """Geometry-bound cell values or a static physical-bed evaluator."""

    cell_values: Array | None
    evaluator: Callable[[Array], ArrayLike] | None = eqx.field(static=True)
    field_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cell_values: ArrayLike | None = None,
        evaluator: Callable[[Array], ArrayLike] | None = None,
        field_id: str,
    ):
        if (cell_values is None) == (evaluator is None):
            raise ValueError(
                "Bathymetry requires exactly one of cell_values or evaluator."
            )
        if evaluator is not None and not callable(evaluator):
            raise TypeError("Bathymetry evaluator must be callable.")
        identifier = str(field_id)
        if not identifier:
            raise ValueError("Bathymetry field_id must be non-empty.")
        values = None if cell_values is None else jnp.asarray(cell_values)
        self.cell_values = values
        self.evaluator = evaluator
        self.field_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shallow-water-bathymetry-plan",
                "field": identifier,
                "representation": "cell-values" if values is not None else "evaluator",
                "values": (
                    None if values is None else array_tree_fingerprint(np.asarray(values))
                ),
            }
        )

    def prepare(
        self,
        discretization: Any,
        /,
        *,
        precision: FiniteVolumePrecisionPolicy | None = None,
    ) -> "PreparedShallowWaterBathymetry":
        policy = FiniteVolumePrecisionPolicy() if precision is None else precision
        if not isinstance(policy, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be FiniteVolumePrecisionPolicy or None.")
        values = (
            self.cell_values
            if self.cell_values is not None
            else self.evaluator(discretization.cell_centers)
        )
        return PreparedShallowWaterBathymetry(
            values,
            discretization.cell_shape,
            geometry_id=discretization.prepared_id,
            precision_id=policy.policy_id,
            dtype=policy.storage_dtype,
            evaluator=self.evaluator,
            field_id=self.field_id,
        )


class PreparedShallowWaterBathymetry(StrictModule, NonTrainableState):
    """Static upward-positive bathymetry bound to prepared FV geometry."""

    values: Array
    geometry_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    bed_id: str = eqx.field(static=True)
    evaluator: Callable[[Array], ArrayLike] | None = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        cell_shape: tuple[int, ...],
        /,
        *,
        geometry_id: str,
        precision_id: str,
        dtype: Any,
        evaluator: Callable[[Array], ArrayLike] | None = None,
        field_id: str = "bathymetry",
    ):
        host = np.asarray(values)
        if host.shape != cell_shape:
            raise ValueError("Bathymetry must match the finite-volume cell shape.")
        if not np.issubdtype(host.dtype, np.floating):
            host = host.astype(dtype)
        if np.any(~np.isfinite(host)):
            raise ValueError("Bathymetry must contain only finite values.")
        geometry = str(geometry_id)
        precision = str(precision_id)
        if not geometry or not precision:
            raise ValueError(
                "Prepared bathymetry requires geometry and precision identities."
            )
        values_ = jnp.asarray(host, dtype=dtype)
        identifier = str(field_id)
        if not identifier:
            raise ValueError("Prepared bathymetry field_id must be non-empty.")
        self.values = values_
        self.geometry_id = geometry
        self.precision_id = precision
        self.evaluator = evaluator
        self.field_id = identifier
        self.bed_id = canonical_fingerprint(
            {
                "kind": "prepared-shallow-water-bathymetry",
                "geometry": geometry,
                "precision": precision,
                "field": identifier,
                "values": array_tree_fingerprint(host),
                "sign": "upward-positive",
            }
        )

    def stage_values(self, coordinates: ArrayLike, /) -> Array:
        """Refresh a static physical bed on moving fixed-topology geometry."""
        points = jnp.asarray(coordinates)
        if self.evaluator is None:
            if points.shape[:-1] != self.values.shape:
                raise ValueError("Stage coordinates do not match bathymetry support.")
            return self.values
        values = jnp.asarray(self.evaluator(points), dtype=self.values.dtype)
        if values.shape != points.shape[:-1]:
            raise ValueError("Bathymetry evaluator must return coordinate leading shape.")
        return eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Stage bathymetry contains non-finite values.",
        )

    def ghost_axis(self, axis: int, depth: int, /, *, periodic: bool) -> Array:
        axis_ = int(axis)
        depth_ = int(depth)
        if depth_ <= 0:
            raise ValueError("Bathymetry ghost depth must be positive.")
        moved = jnp.moveaxis(self.values, axis_, 0)
        if periodic:
            lower = moved[-depth_:]
            upper = moved[:depth_]
        else:
            lower = jnp.broadcast_to(moved[:1], (depth_, *moved.shape[1:]))
            upper = jnp.broadcast_to(moved[-1:], (depth_, *moved.shape[1:]))
        return jnp.moveaxis(jnp.concatenate((lower, moved, upper), axis=0), 0, axis_)


class ShallowWaterBalancedFaceResult(StrictModule):
    """Shared transport flux plus one-sided hydrostatic corrections."""

    normal_flux: Array
    left_correction: Array
    right_correction: Array
    max_speed: Array
    reconstructed_left: Array
    reconstructed_right: Array
    dry_face: Array

    def __init__(
        self,
        normal_flux: ArrayLike,
        left_correction: ArrayLike,
        right_correction: ArrayLike,
        max_speed: ArrayLike,
        reconstructed_left: ArrayLike,
        reconstructed_right: ArrayLike,
        dry_face: ArrayLike,
        /,
    ):
        flux = jnp.asarray(normal_flux)
        left = jnp.asarray(left_correction)
        right = jnp.asarray(right_correction)
        speed = jnp.asarray(max_speed)
        state_left = jnp.asarray(reconstructed_left)
        state_right = jnp.asarray(reconstructed_right)
        dry = jnp.asarray(dry_face, dtype=bool)
        if left.shape != flux.shape or right.shape != flux.shape:
            raise ValueError("Hydrostatic corrections must match the shared flux shape.")
        if state_left.shape != flux.shape or state_right.shape != flux.shape:
            raise ValueError(
                "Hydrostatic reconstructed states must match the flux shape."
            )
        if speed.shape != flux.shape[:-1] or dry.shape != speed.shape:
            raise ValueError("Balanced-face speed and dry mask must match face batches.")
        self.normal_flux = flux
        self.left_correction = left
        self.right_correction = right
        self.max_speed = speed
        self.reconstructed_left = state_left
        self.reconstructed_right = state_right
        self.dry_face = dry

    @property
    def left_flux(self) -> Array:
        return self.normal_flux + self.left_correction

    @property
    def right_flux(self) -> Array:
        return self.normal_flux + self.right_correction


class ShallowWaterHydrostaticHLLPlan(StrictModule, NonTrainableState):
    """Chen--Noelle hydrostatic reconstruction with a dry-safe HLL flux."""

    wet_dry: ShallowWaterWetDryPolicy
    differentiability: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, wet_dry: ShallowWaterWetDryPolicy | None = None, /):
        policy = ShallowWaterWetDryPolicy() if wet_dry is None else wet_dry
        if not isinstance(policy, ShallowWaterWetDryPolicy):
            raise TypeError("wet_dry must be a ShallowWaterWetDryPolicy.")
        self.wet_dry = policy
        self.differentiability = "branchwise"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "shallow-water-hydrostatic-hll",
                "wet_dry": policy.policy_id,
                "bed_model": "chen-noelle-subcell",
            }
        )

    def face_contribution(
        self,
        system: ShallowWaterSystem,
        left: Array,
        right: Array,
        bathymetry_left: Array,
        bathymetry_right: Array,
        axis: int,
        args: Any = None,
        /,
    ) -> ShallowWaterBalancedFaceResult:
        del args
        from ...equations._hyperbolic_systems import ShallowWaterSystem

        if not isinstance(system, ShallowWaterSystem):
            raise TypeError("Hydrostatic HLL requires ShallowWaterSystem.")
        axis_ = int(axis)
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        bed_left = jnp.asarray(bathymetry_left, dtype=left_.dtype)

        bed_right = jnp.asarray(bathymetry_right, dtype=right_.dtype)
        depth_left = jnp.maximum(left_[..., 0], 0.0)
        depth_right = jnp.maximum(right_[..., 0], 0.0)
        surface_left = depth_left + bed_left
        surface_right = depth_right + bed_right
        bed_star = jnp.minimum(
            jnp.maximum(bed_left, bed_right),
            jnp.minimum(surface_left, surface_right),
        )
        reconstructed_depth_left = jnp.maximum(
            jnp.minimum(surface_left - bed_star, depth_left), 0.0
        )
        reconstructed_depth_right = jnp.maximum(
            jnp.minimum(surface_right - bed_star, depth_right), 0.0
        )
        left_wet = self.wet_dry.wet(reconstructed_depth_left)
        right_wet = self.wet_dry.wet(reconstructed_depth_right)
        reconstructed_depth_left = jnp.where(left_wet, reconstructed_depth_left, 0.0)
        reconstructed_depth_right = jnp.where(right_wet, reconstructed_depth_right, 0.0)
        velocity_left = self.wet_dry.velocity(left_)
        velocity_right = self.wet_dry.velocity(right_)
        momentum_left = reconstructed_depth_left[..., None] * velocity_left
        momentum_right = reconstructed_depth_right[..., None] * velocity_right
        state_left = jnp.concatenate(
            (reconstructed_depth_left[..., None], momentum_left), axis=-1
        )
        state_right = jnp.concatenate(
            (reconstructed_depth_right[..., None], momentum_right), axis=-1
        )
        lower, upper = system.signal_bounds(state_left, state_right, axis_)
        lower = jnp.minimum(lower, 0.0)
        upper = jnp.maximum(upper, 0.0)
        left_flux = system.physical_flux(state_left, axis_)
        right_flux = system.physical_flux(state_right, axis_)
        denominator = upper - lower
        middle = (
            upper[..., None] * left_flux
            - lower[..., None] * right_flux
            + (lower * upper)[..., None] * (state_right - state_left)
        ) / jnp.where(denominator == 0.0, 1.0, denominator)[..., None]
        normal_flux = jnp.where(
            (lower >= 0.0)[..., None],
            left_flux,
            jnp.where((upper <= 0.0)[..., None], right_flux, middle),
        )
        dry_face = (~left_wet) & (~right_wet)
        normal_flux = jnp.where(dry_face[..., None], 0.0, normal_flux)
        left_correction = (
            jnp.zeros_like(normal_flux)
            .at[..., 1 + axis_]
            .set(0.5 * system.gravity * (depth_left**2 - reconstructed_depth_left**2))
        )
        right_correction = (
            jnp.zeros_like(normal_flux)
            .at[..., 1 + axis_]
            .set(0.5 * system.gravity * (depth_right**2 - reconstructed_depth_right**2))
        )
        speed = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
        return ShallowWaterBalancedFaceResult(
            normal_flux,
            left_correction,
            right_correction,
            speed,
            state_left,
            state_right,
            dry_face,
        )

    def normal_face_contribution(
        self,
        system: ShallowWaterSystem,
        left: Array,
        right: Array,
        bathymetry_left: Array,
        bathymetry_right: Array,
        normal: Array,
        grid_normal_velocity: Array | None = None,
        args: Any = None,
        /,
    ) -> ShallowWaterBalancedFaceResult:
        """Hydrostatic HLL contribution for an arbitrary unit normal and ALE speed."""
        del args
        from ...equations._hyperbolic_systems import ShallowWaterSystem

        if not isinstance(system, ShallowWaterSystem):
            raise TypeError("Hydrostatic HLL requires ShallowWaterSystem.")
        left_, right_, normal_ = (
            jnp.asarray(left),
            jnp.asarray(right),
            jnp.asarray(normal),
        )
        if normal_.shape[-1:] != (system.dimension,):
            raise ValueError("Shallow-water face normal has the wrong dimension.")
        norm = jnp.sqrt(jnp.sum(normal_ * normal_, axis=-1))
        normal_ = eqx.error_if(
            normal_,
            jnp.any(~jnp.isfinite(normal_))
            | jnp.any(jnp.abs(norm - 1.0) > 64.0 * jnp.finfo(normal_.dtype).eps),
            "Shallow-water face normals must be finite and unit length.",
        )
        bed_left = jnp.asarray(bathymetry_left, dtype=left_.dtype)
        bed_right = jnp.asarray(bathymetry_right, dtype=right_.dtype)
        depth_left, depth_right = (
            jnp.maximum(left_[..., 0], 0),
            jnp.maximum(right_[..., 0], 0),
        )
        surface_left, surface_right = depth_left + bed_left, depth_right + bed_right
        bed_star = jnp.minimum(
            jnp.maximum(bed_left, bed_right), jnp.minimum(surface_left, surface_right)
        )
        reconstructed_left = jnp.maximum(
            jnp.minimum(surface_left - bed_star, depth_left), 0
        )
        reconstructed_right = jnp.maximum(
            jnp.minimum(surface_right - bed_star, depth_right), 0
        )
        left_wet, right_wet = (
            self.wet_dry.wet(reconstructed_left),
            self.wet_dry.wet(reconstructed_right),
        )
        reconstructed_left = jnp.where(left_wet, reconstructed_left, 0)
        reconstructed_right = jnp.where(right_wet, reconstructed_right, 0)
        state_left = jnp.concatenate(
            (
                reconstructed_left[..., None],
                reconstructed_left[..., None] * self.wet_dry.velocity(left_),
            ),
            -1,
        )
        state_right = jnp.concatenate(
            (
                reconstructed_right[..., None],
                reconstructed_right[..., None] * self.wet_dry.velocity(right_),
            ),
            -1,
        )
        grid_speed = (
            jnp.zeros(state_left.shape[:-1], dtype=state_left.dtype)
            if grid_normal_velocity is None
            else jnp.asarray(grid_normal_velocity, dtype=state_left.dtype)
        )
        if grid_speed.shape != state_left.shape[:-1]:
            raise ValueError("grid_normal_velocity must match the face batch.")
        lower, upper = system.normal_signal_bounds(state_left, state_right, normal_)
        lower, upper = (
            jnp.minimum(lower - grid_speed, 0),
            jnp.maximum(upper - grid_speed, 0),
        )
        flux_left = jnp.stack(
            tuple(
                system.physical_flux(state_left, axis) for axis in range(system.dimension)
            ),
            -1,
        )
        flux_right = jnp.stack(
            tuple(
                system.physical_flux(state_right, axis)
                for axis in range(system.dimension)
            ),
            -1,
        )
        flux_left = (
            oe.contract("...id,...d->...i", flux_left, normal_, backend="jax")
            - grid_speed[..., None] * state_left
        )
        flux_right = (
            oe.contract("...id,...d->...i", flux_right, normal_, backend="jax")
            - grid_speed[..., None] * state_right
        )
        denominator = upper - lower
        middle = (
            upper[..., None] * flux_left
            - lower[..., None] * flux_right
            + (lower * upper)[..., None] * (state_right - state_left)
        ) / jnp.where(denominator == 0, 1, denominator)[..., None]
        flux = jnp.where(
            (lower >= 0)[..., None],
            flux_left,
            jnp.where((upper <= 0)[..., None], flux_right, middle),
        )
        dry = ~left_wet & ~right_wet
        flux = jnp.where(dry[..., None], 0, flux)
        left_pressure = 0.5 * system.gravity * (depth_left**2 - reconstructed_left**2)
        right_pressure = 0.5 * system.gravity * (depth_right**2 - reconstructed_right**2)
        left_correction = jnp.concatenate(
            (
                jnp.zeros_like(left_pressure)[..., None],
                left_pressure[..., None] * normal_,
            ),
            -1,
        )
        right_correction = jnp.concatenate(
            (
                jnp.zeros_like(right_pressure)[..., None],
                right_pressure[..., None] * normal_,
            ),
            -1,
        )
        return ShallowWaterBalancedFaceResult(
            flux,
            left_correction,
            right_correction,
            jnp.maximum(jnp.abs(lower), jnp.abs(upper)),
            state_left,
            state_right,
            dry,
        )


class ShallowWaterAcceptedFaceIntegrals(StrictModule):
    """Accepted SSPRK transport and one-sided bed-correction integrals."""

    normal_flux_integrals: tuple[Array, ...]
    left_correction_integrals: tuple[Array, ...]
    right_correction_integrals: tuple[Array, ...]
    accepted_step_size: Array
    axis_names: tuple[str, ...] = eqx.field(static=True)
    bed_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        contributions: tuple[ShallowWaterBalancedFaceResult, ...],
        face_measures: tuple[Array, ...],
        accepted_step_size: ArrayLike,
        /,
        *,
        axis_names: tuple[str, ...],
        bed_id: str,
        plan_id: str,
    ):
        if len(contributions) != len(face_measures) or len(contributions) != len(
            axis_names
        ):
            raise ValueError(
                "Accepted shallow-water face blocks must align with geometry axes."
            )
        dt = jnp.asarray(accepted_step_size).reshape(())
        normal = []
        left = []
        right = []
        for contribution, measure in zip(contributions, face_measures, strict=True):
            scale = (
                jnp.asarray(measure, dtype=contribution.normal_flux.dtype)[..., None] * dt
            )
            normal.append(contribution.normal_flux * scale)
            left.append(contribution.left_correction * scale)
            right.append(contribution.right_correction * scale)
        self.normal_flux_integrals = tuple(normal)
        self.left_correction_integrals = tuple(left)
        self.right_correction_integrals = tuple(right)
        self.accepted_step_size = dt
        self.axis_names = tuple(str(name) for name in axis_names)
        self.bed_id = str(bed_id)
        self.plan_id = str(plan_id)


def reconstruct_shallow_water_faces(
    reconstruction: Any,
    system: ShallowWaterSystem,
    wet_dry: ShallowWaterWetDryPolicy,
    ghosted_state: ArrayLike,
    ghosted_bathymetry: ArrayLike,
    axis: int,
    /,
    *,
    interior_cell_count: int,
    ghost_depth: int,
    periodic: bool,
    axis_coordinates: ArrayLike,
) -> tuple[Array, Array, Array, Array]:
    """Reconstruct free surface, discharge, and bed with one stencil."""
    state = jnp.asarray(ghosted_state)
    bed = jnp.asarray(ghosted_bathymetry, dtype=state.dtype)
    if bed.shape != state.shape[:-1]:
        raise ValueError("Ghosted bathymetry must match ghosted state cells.")
    augmented = jnp.concatenate(
        (state[..., :1] + bed[..., None], state[..., 1:], bed[..., None]), axis=-1
    )
    left, right = reconstruct_ghosted_axis(
        reconstruction,
        augmented,
        int(axis),
        interior_cell_count=interior_cell_count,
        ghost_depth=ghost_depth,
        periodic=periodic,
        axis_coordinates=axis_coordinates,
    )
    piecewise_left, piecewise_right = reconstruct_ghosted_axis(
        PiecewiseConstantReconstruction(),
        augmented,
        int(axis),
        interior_cell_count=interior_cell_count,
        ghost_depth=ghost_depth,
        periodic=periodic,
        axis_coordinates=axis_coordinates,
    )
    piecewise_depth_left = jnp.maximum(
        piecewise_left[..., 0] - piecewise_left[..., -1], 0.0
    )
    left = left.at[..., -1].set(piecewise_left[..., -1])
    right = right.at[..., -1].set(piecewise_right[..., -1])
    piecewise_depth_right = jnp.maximum(
        piecewise_right[..., 0] - piecewise_right[..., -1], 0.0
    )
    partially_dry = (~wet_dry.wet(piecewise_depth_left)) | (
        ~wet_dry.wet(piecewise_depth_right)
    )
    left = jnp.where(partially_dry[..., None], piecewise_left, left)
    right = jnp.where(partially_dry[..., None], piecewise_right, right)

    def unpack(trace: Array) -> tuple[Array, Array]:
        surface = trace[..., 0]
        discharge = trace[..., 1:-1]
        bathymetry = trace[..., -1]
        depth = jnp.maximum(surface - bathymetry, 0.0)
        discharge = jnp.where(wet_dry.wet(depth)[..., None], discharge, 0.0)
        value = jnp.concatenate((depth[..., None], discharge), axis=-1)
        return wet_dry.enforce_dry_momentum(value), bathymetry

    state_left, bed_left = unpack(left)
    state_right, bed_right = unpack(right)
    if state_left.shape[-1] != system.component_count:
        raise ValueError(
            "Reconstructed shallow-water state has the wrong component count."
        )
    return state_left, state_right, bed_left, bed_right


class ShallowWaterObservables(StrictModule):
    """Renderer-neutral shallow-water fields derived from authoritative state."""

    depth: Array
    bathymetry: Array
    surface: Array
    momentum: Array
    velocity: Array
    wet_mask: Array
    energy_density: Array
    bed_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)


def shallow_water_observables(
    system: ShallowWaterSystem,
    state: ArrayLike,
    bathymetry: PreparedShallowWaterBathymetry,
    wet_dry: ShallowWaterWetDryPolicy,
    /,
) -> ShallowWaterObservables:
    value = jnp.asarray(state)
    depth = value[..., 0]
    momentum = value[..., 1:]
    bed = jnp.asarray(bathymetry.values, dtype=value.dtype)
    velocity = wet_dry.velocity(value)
    kinetic = 0.5 * jnp.sum(momentum * velocity, axis=-1)
    energy = kinetic + 0.5 * system.gravity * depth**2 + system.gravity * depth * bed
    return ShallowWaterObservables(
        depth=depth,
        bathymetry=bed,
        surface=depth + bed,
        momentum=momentum,
        velocity=velocity,
        wet_mask=wet_dry.wet(depth),
        energy_density=energy,
        bed_id=bathymetry.bed_id,
        precision_id=bathymetry.precision_id,
    )


__all__ = [
    "PreparedShallowWaterBathymetry",
    "ShallowWaterBathymetryPlan",
    "ShallowWaterAcceptedFaceIntegrals",
    "ShallowWaterBalancedFaceResult",
    "ShallowWaterHydrostaticHLLPlan",
    "ShallowWaterObservables",
    "ShallowWaterWetDryPolicy",
    "reconstruct_shallow_water_faces",
    "shallow_water_observables",
]
