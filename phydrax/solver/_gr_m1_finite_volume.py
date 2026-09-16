#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._relativistic_hydrodynamics import (
    valencia_geometric_source_from_projection,
)
from ..equations._relativistic_radiation import GRGreyM1RadiationSystem
from ..metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ._relativistic_finite_volume import ValenciaFiniteVolumeStageGeometry


GRM1BoundaryKind: TypeAlias = Literal["outflow", "vacuum", "reflective", "prescribed"]
GRM1ReconstructionKind: TypeAlias = Literal["piecewise_constant", "plm"]


class GRM1FiniteVolumeRunStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    GEOMETRY_INVALID = 2
    STABILITY_LIMIT_EXCEEDED = 3
    REALIZABILITY_FAILED = 4
    BOUNDARY_FAILED = 5
    CONSERVATION_DEFECT = 6
    NONFINITE_STATE = 7


class GRM1BoundaryCondition(StrictModule, NonTrainableState):
    """One metric-aware grey-M1 exterior moment policy."""

    kind: GRM1BoundaryKind = eqx.field(static=True)
    prescribed_moments: Array
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: GRM1BoundaryKind,
        /,
        *,
        prescribed_moments: ArrayLike | None = None,
    ) -> None:
        if kind not in ("outflow", "vacuum", "reflective", "prescribed"):
            raise ValueError("Unknown GR M1 boundary kind.")
        if kind == "prescribed":
            if prescribed_moments is None:
                raise ValueError("A prescribed M1 boundary requires moments.")
            moments = np.asarray(prescribed_moments, dtype=float)
            if moments.shape != (4,) or np.any(~np.isfinite(moments)):
                raise ValueError("Prescribed M1 moments must be one finite four-vector.")
        else:
            if prescribed_moments is not None:
                raise ValueError("Only a prescribed M1 boundary accepts moments.")
            moments = np.zeros((4,), dtype=float)
        self.kind = kind
        self.prescribed_moments = jnp.asarray(moments)
        self.boundary_id = canonical_fingerprint(
            {
                "kind": f"gr-m1-boundary:{kind}",
                "moments": (
                    None if kind != "prescribed" else array_tree_fingerprint(moments)
                ),
            }
        )

    def exterior(
        self,
        system: GRGreyM1RadiationSystem,
        interior: Array,
        face_geometry: ADMGridGeometry,
        axis: int,
        side: Literal["lower", "upper"],
        /,
    ) -> tuple[Array, Array]:
        if self.kind == "prescribed":
            result = jnp.broadcast_to(
                self.prescribed_moments.astype(interior.dtype), interior.shape
            )
            closure = system.closure(result[..., 0], result[..., 1:], face_geometry)
            return result, closure.qualified
        if self.kind == "outflow":
            closure = system.closure(interior[..., 0], interior[..., 1:], face_geometry)
            return interior, closure.qualified
        covector = jnp.zeros(interior.shape[:-1] + (3,), dtype=interior.dtype)
        orientation = -1.0 if side == "lower" else 1.0
        covector = covector.at[..., int(axis)].set(orientation)
        inverse = face_geometry.inverse_spatial_metric.astype(interior.dtype)
        norm = jnp.sqrt(
            jnp.maximum(
                ein.contract("...i,...ij,...j->...", covector, inverse, covector),
                jnp.finfo(interior.dtype).tiny,
            )
        )
        normal_covector = covector / norm[..., None]
        flux_vector = ein.contract("...ij,...j->...i", inverse, interior[..., 1:])
        outward_flux = ein.contract("...i,...i->...", flux_vector, normal_covector)
        multiplier = (
            2.0 * outward_flux
            if self.kind == "reflective"
            else jnp.minimum(outward_flux, 0.0)
        )
        exterior_flux = interior[..., 1:] - multiplier[..., None] * normal_covector
        result = interior.at[..., 1:].set(exterior_flux)
        closure = system.closure(result[..., 0], result[..., 1:], face_geometry)
        return result, closure.qualified


class GRM1BoundaryPair(StrictModule, NonTrainableState):
    lower: GRM1BoundaryCondition
    upper: GRM1BoundaryCondition
    pair_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: GRM1BoundaryCondition,
        upper: GRM1BoundaryCondition,
        /,
    ) -> None:
        if not isinstance(lower, GRM1BoundaryCondition) or not isinstance(
            upper, GRM1BoundaryCondition
        ):
            raise TypeError("M1 boundary pair values must be GRM1BoundaryCondition.")
        self.lower = lower
        self.upper = upper
        self.pair_id = canonical_fingerprint(
            {
                "kind": "gr-m1-boundary-pair",
                "lower": lower.boundary_id,
                "upper": upper.boundary_id,
            }
        )


class GRM1FiniteVolumeState(StrictModule):
    radiation_state: Array
    time: Array
    step_size: Array
    accepted_step: Array
    status: Array


class GRM1SpatialRate(StrictModule):
    radiation_rate: Array
    face_fluxes: tuple[Array, ...]
    integrated_face_fluxes: tuple[Array, ...]
    geometric_source: Array
    stress_energy: StressEnergyProjection
    stable_step: Array
    maximum_reduced_flux: Array
    maximum_realizability_correction: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class GRM1ConservationLedger(StrictModule):
    face_flux_integrals: tuple[Array, ...]
    geometric_source_integral: Array
    state_change: Array
    volume_integrated_change: Array
    boundary_flux: Array
    volume_integrated_source: Array
    balance_defect: Array
    accepted: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class GRM1StepResult(StrictModule):
    candidate: GRM1FiniteVolumeState
    state: GRM1FiniteVolumeState
    accepted: Array
    status: Array
    stages: tuple[GRM1SpatialRate, GRM1SpatialRate, GRM1SpatialRate]
    attempted_ledger: GRM1ConservationLedger
    accepted_ledger: GRM1ConservationLedger
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


def _minmod(first: Array, second: Array, third: Array, /) -> Array:
    values = jnp.stack((first, second, third), axis=0)
    positive = jnp.all(values > 0.0, axis=0)
    negative = jnp.all(values < 0.0, axis=0)
    magnitude = jnp.min(jnp.abs(values), axis=0)
    return jnp.where(positive, magnitude, jnp.where(negative, -magnitude, 0.0))


def _take_adm_geometry(
    geometry: ADMGridGeometry, index: int, axis: int, /
) -> ADMGridGeometry:
    return ADMGridGeometry(
        jnp.take(geometry.alpha, index, axis=axis),
        jnp.take(geometry.beta_contravariant, index, axis=axis),
        jnp.take(geometry.spatial_metric, index, axis=axis),
        jnp.take(geometry.inverse_spatial_metric, index, axis=axis),
        jnp.take(geometry.sqrt_det_spatial_metric, index, axis=axis),
        jnp.take(geometry.extrinsic_curvature, index, axis=axis),
        jnp.take(geometry.active, index, axis=axis),
        jnp.take(geometry.valid, index, axis=axis),
        snapshot_token=geometry.snapshot_token,
        chart_id=geometry.chart_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        geometry_lineage_id=geometry.geometry_lineage_id,
    )


class FixedGridGRM1SSPRK3Plan(StrictModule, NonTrainableState):
    """Atomic metric-aware grey-M1 finite volume with SSPRK(3,3)."""

    system: GRGreyM1RadiationSystem
    discretization: object
    boundaries: tuple[GRM1BoundaryPair | None, ...]
    reconstruction: GRM1ReconstructionKind = eqx.field(static=True)
    plm_theta: float = eqx.field(static=True)
    cfl: float = eqx.field(static=True)
    balance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: GRGreyM1RadiationSystem,
        discretization,
        /,
        *,
        boundaries: tuple[GRM1BoundaryPair | None, ...] | None = None,
        reconstruction: GRM1ReconstructionKind = "piecewise_constant",
        plm_theta: float = 1.5,
        cfl: float = 0.35,
        balance_tolerance: float = 1.0e-9,
    ) -> None:
        from ..discretization.finite_volume import FiniteVolumeDiscretization

        if not isinstance(system, GRGreyM1RadiationSystem):
            raise TypeError("system must be GRGreyM1RadiationSystem.")
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be FiniteVolumeDiscretization.")
        if reconstruction not in ("piecewise_constant", "plm"):
            raise ValueError("GR M1 reconstruction must be piecewise_constant or plm.")
        theta = float(plm_theta)
        cfl_ = float(cfl)
        tolerance = float(balance_tolerance)
        if (
            not np.isfinite(theta)
            or not 1.0 <= theta <= 2.0
            or not np.isfinite(cfl_)
            or not 0.0 < cfl_ <= 1.0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("GR M1 finite-volume controls are invalid.")
        dimension = len(discretization.cell_shape)
        boundary_values = (
            tuple(None for _ in range(dimension))
            if boundaries is None
            else tuple(boundaries)
        )
        if len(boundary_values) != dimension or any(
            value is not None and not isinstance(value, GRM1BoundaryPair)
            for value in boundary_values
        ):
            raise TypeError("One optional GR M1 boundary pair is required per axis.")
        for axis, pair in zip(
            discretization.grid.structured_axes, boundary_values, strict=True
        ):
            if axis.periodic != (pair is None):
                raise ValueError(
                    "Periodic M1 axes require no boundary pair and bounded axes require one."
                )
        self.system = system
        self.discretization = discretization
        self.boundaries = boundary_values
        self.reconstruction = reconstruction
        self.plm_theta = theta
        self.cfl = cfl_
        self.balance_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-gr-m1-ssprk3",
                "system": system.system_id,
                "discretization": discretization.prepared_id,
                "boundaries": [
                    None if value is None else value.pair_id for value in boundary_values
                ],
                "reconstruction": reconstruction,
                "plm_theta": theta,
                "cfl": cfl_,
                "balance_tolerance": tolerance,
            }
        )

    @property
    def cell_shape(self) -> tuple[int, ...]:
        return tuple(self.discretization.cell_shape)

    def _check_stage(self, stage: ValenciaFiniteVolumeStageGeometry, /) -> None:
        if not isinstance(stage, ValenciaFiniteVolumeStageGeometry):
            raise TypeError("stage must be ValenciaFiniteVolumeStageGeometry.")
        if stage.cell.leading_shape != self.cell_shape:
            raise ValueError("M1 cell geometry does not match its finite-volume grid.")
        if len(stage.faces) != len(self.cell_shape):
            raise ValueError("M1 stage requires one face geometry per grid axis.")
        if (
            stage.cell.scale_id != self.system.scale.scale_id
            or stage.cell.convention_id != self.system.convention.convention_id
        ):
            raise ValueError("M1 stage geometry and radiation contracts differ.")

    def _state(self, value: ArrayLike, /) -> Array:
        state = jnp.asarray(value)
        expected = self.cell_shape + (4,)
        if state.shape != expected:
            raise ValueError(f"Densitized M1 state must have shape {expected}.")
        return state

    def initialize(
        self,
        moments: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        /,
        *,
        time: ArrayLike = 0.0,
        step_size: ArrayLike | None = None,
    ) -> GRM1FiniteVolumeState:
        self._check_stage(geometry)
        value = jnp.asarray(moments)
        if value.shape != self.cell_shape + (4,):
            raise ValueError("Initial M1 moments do not match the grid.")
        closure = self.system.closure(value[..., 0], value[..., 1:], geometry.cell)
        valid = jnp.all(closure.qualified | ~geometry.cell.active)
        value = eqx.error_if(value, ~valid, "Initial M1 moments are not realizable.")
        densitized = geometry.cell.sqrt_det_spatial_metric[..., None] * value
        time_ = jnp.asarray(time, dtype=densitized.dtype).reshape(())
        step_ = jnp.asarray(
            jnp.nan if step_size is None else step_size, dtype=densitized.dtype
        ).reshape(())
        return GRM1FiniteVolumeState(
            densitized,
            time_,
            step_,
            jnp.zeros((), dtype=jnp.int32),
            jnp.asarray(int(GRM1FiniteVolumeRunStatus.SUCCESS), dtype=jnp.int32),
        )

    def moments(self, state: ArrayLike, geometry: ADMGridGeometry, /) -> Array:
        value = self._state(state)
        return value / geometry.sqrt_det_spatial_metric[..., None]

    def _limit_realizability(
        self, moments: Array, geometry: ADMGridGeometry, /
    ) -> tuple[Array, Array]:
        energy = jnp.maximum(
            moments[..., 0],
            jnp.asarray(self.system.energy_floor, dtype=moments.dtype),
        )
        flux = moments[..., 1:]
        inverse = geometry.inverse_spatial_metric.astype(moments.dtype)
        norm = jnp.sqrt(
            jnp.maximum(ein.contract("...i,...ij,...j->...", flux, inverse, flux), 0.0)
        )
        maximum = (
            jnp.asarray(self.system.physical_light_speed, dtype=moments.dtype)
            * energy
            * (1.0 - 64.0 * jnp.finfo(moments.dtype).eps)
        )
        factor = jnp.minimum(1.0, maximum / jnp.maximum(norm, maximum))
        limited = jnp.concatenate((energy[..., None], factor[..., None] * flux), -1)
        correction = jnp.max(jnp.abs(limited - moments), axis=-1)
        return limited, correction

    def _boundary_exterior(
        self,
        moments: Array,
        face_geometry: ADMGridGeometry,
        axis: int,
        side: Literal["lower", "upper"],
        /,
    ) -> tuple[Array, Array]:
        pair = self.boundaries[axis]
        if pair is None:
            raise RuntimeError("A bounded M1 axis has no boundary pair.")
        boundary = pair.lower if side == "lower" else pair.upper
        return boundary.exterior(self.system, moments, face_geometry, axis, side)

    def _reconstruct(
        self,
        moments: Array,
        stage: ValenciaFiniteVolumeStageGeometry,
        axis: int,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        periodic = self.discretization.grid.structured_axes[axis].periodic
        values = jnp.moveaxis(moments, axis, 0)
        if periodic:
            if self.reconstruction == "piecewise_constant":
                left = values
                right = jnp.roll(values, -1, axis=0)
            else:
                backward = values - jnp.roll(values, 1, axis=0)
                forward = jnp.roll(values, -1, axis=0) - values
                centered = 0.5 * (
                    jnp.roll(values, -1, axis=0) - jnp.roll(values, 1, axis=0)
                )
                slope = _minmod(
                    self.plm_theta * backward,
                    centered,
                    self.plm_theta * forward,
                )
                left = values + 0.5 * slope
                right = jnp.roll(values - 0.5 * slope, -1, axis=0)
            boundary_valid = jnp.asarray(True)
        else:
            lower_interior = jnp.take(moments, 0, axis=axis)
            upper_interior = jnp.take(moments, moments.shape[axis] - 1, axis=axis)
            lower_geometry = _take_adm_geometry(stage.faces[axis], 0, axis)
            upper_geometry = _take_adm_geometry(
                stage.faces[axis], stage.faces[axis].leading_shape[axis] - 1, axis
            )
            lower, lower_valid = self._boundary_exterior(
                lower_interior, lower_geometry, axis, "lower"
            )
            upper, upper_valid = self._boundary_exterior(
                upper_interior, upper_geometry, axis, "upper"
            )
            extended = jnp.concatenate(
                (lower[None, ...], values, upper[None, ...]), axis=0
            )
            if self.reconstruction == "piecewise_constant":
                left = extended[:-1]
                right = extended[1:]
            else:
                backward = extended - jnp.roll(extended, 1, axis=0)
                forward = jnp.roll(extended, -1, axis=0) - extended
                centered = 0.5 * (
                    jnp.roll(extended, -1, axis=0) - jnp.roll(extended, 1, axis=0)
                )
                slope = _minmod(
                    self.plm_theta * backward,
                    centered,
                    self.plm_theta * forward,
                )
                slope = slope.at[0].set(0.0).at[-1].set(0.0)
                left = extended[:-1] + 0.5 * slope[:-1]
                right = extended[1:] - 0.5 * slope[1:]
            boundary_valid = jnp.all(lower_valid) & jnp.all(upper_valid)
        left = jnp.moveaxis(left, 0, axis)
        right = jnp.moveaxis(right, 0, axis)
        left, left_correction = self._limit_realizability(left, stage.faces[axis])
        right, right_correction = self._limit_realizability(right, stage.faces[axis])
        return left, right, jnp.maximum(left_correction, right_correction), boundary_valid

    def _face_extinction(
        self, extinction: Array, axis: int, periodic: bool, /
    ) -> tuple[Array, Array]:
        if periodic:
            face = 0.5 * (extinction + jnp.roll(extinction, -1, axis=axis))
            width = self.discretization.grid.structured_axes[axis].interval_widths
            face_width = 0.5 * (width + jnp.roll(width, -1))
        else:
            lower = jnp.take(extinction, jnp.asarray([0]), axis=axis)
            upper = jnp.take(
                extinction, jnp.asarray([extinction.shape[axis] - 1]), axis=axis
            )
            interior = 0.5 * (
                jnp.take(extinction, jnp.arange(extinction.shape[axis] - 1), axis=axis)
                + jnp.take(extinction, jnp.arange(1, extinction.shape[axis]), axis=axis)
            )
            face = jnp.concatenate((lower, interior, upper), axis=axis)
            width = self.discretization.grid.structured_axes[axis].interval_widths
            face_width = jnp.concatenate(
                (width[:1], 0.5 * (width[:-1] + width[1:]), width[-1:])
            )
        shape = [1] * face.ndim
        shape[axis] = face_width.size
        return face, face_width.reshape(shape)

    def rate(
        self,
        time: ArrayLike,
        radiation_state: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        /,
        *,
        transport_extinction: ArrayLike = 0.0,
    ) -> GRM1SpatialRate:
        del time
        self._check_stage(geometry)
        state = self._state(radiation_state)
        moments = self.moments(state, geometry.cell)
        extinction = jnp.broadcast_to(
            jnp.asarray(transport_extinction, dtype=state.dtype), self.cell_shape
        )
        cell_projection = self.system.stress_energy_projection(
            moments[..., 0], moments[..., 1:], geometry.cell
        )
        momentum_source, energy_source = valencia_geometric_source_from_projection(
            cell_projection, geometry.source, self.system.convention
        )
        source = jnp.concatenate((energy_source[..., None], momentum_source), axis=-1)
        residual = jnp.zeros_like(state)
        inverse_step = jnp.zeros(self.cell_shape, dtype=state.dtype)
        face_fluxes = []
        integrated_fluxes = []
        corrections = []
        boundary_valid = []
        volumes = self.discretization.cell_volumes.astype(state.dtype)
        for axis in range(len(self.cell_shape)):
            periodic = self.discretization.grid.structured_axes[axis].periodic
            face_geometry = geometry.faces[axis]
            left, right, correction, valid = self._reconstruct(moments, geometry, axis)
            left_flux = self.system.coordinate_flux(
                left[..., 0], left[..., 1:], axis, face_geometry
            )
            right_flux = self.system.coordinate_flux(
                right[..., 0], right[..., 1:], axis, face_geometry
            )
            face_volume = face_geometry.sqrt_det_spatial_metric[..., None]
            left_state = face_volume * left
            right_state = face_volume * right
            physical_average = 0.5 * face_volume * (left_flux + right_flux)
            covector = jnp.zeros(face_geometry.leading_shape + (3,), dtype=state.dtype)
            covector = covector.at[..., axis].set(1.0)
            lower, upper = self.system.coordinate_characteristic_bounds(
                covector, face_geometry
            )
            maximum_speed = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
            face_extinction, normal_width = self._face_extinction(
                extinction, axis, periodic
            )
            optical_depth = face_extinction * normal_width
            diffusion_speed = 2.0 / jnp.maximum(3.0 * optical_depth, 1.0)
            dissipation_speed = jnp.minimum(maximum_speed, diffusion_speed)
            numerical_flux = physical_average - 0.5 * dissipation_speed[..., None] * (
                right_state - left_state
            )
            measure = self.discretization.face_measures[axis].astype(state.dtype)
            integrated = numerical_flux * measure[..., None]
            face_fluxes.append(numerical_flux)
            integrated_fluxes.append(integrated)
            corrections.append(jnp.max(correction, initial=0.0))
            boundary_valid.append(valid)
            if periodic:
                residual = (
                    residual
                    - (integrated - jnp.roll(integrated, 1, axis=axis))
                    / volumes[..., None]
                )
                inverse_step = inverse_step + maximum_speed * measure / volumes
            else:
                lower_indices = jnp.arange(integrated.shape[axis] - 1)
                upper_indices = jnp.arange(1, integrated.shape[axis])
                lower_flux = jnp.take(integrated, lower_indices, axis=axis)
                upper_flux = jnp.take(integrated, upper_indices, axis=axis)
                residual = residual - (upper_flux - lower_flux) / volumes[..., None]
                face_rate = maximum_speed * measure
                inverse_step = (
                    inverse_step
                    + jnp.maximum(
                        jnp.take(face_rate, lower_indices, axis=axis),
                        jnp.take(face_rate, upper_indices, axis=axis),
                    )
                    / volumes
                )
        rate = residual + source
        maximum_inverse = jnp.max(inverse_step, initial=0.0)
        stable = jnp.where(
            maximum_inverse > 0.0,
            jnp.asarray(self.cfl, dtype=state.dtype) / maximum_inverse,
            jnp.asarray(jnp.inf, dtype=state.dtype),
        )
        closure = self.system.closure(moments[..., 0], moments[..., 1:], geometry.cell)
        correction = jnp.max(jnp.stack(tuple(corrections)), initial=0.0)
        finite = (
            geometry.finite
            & jnp.all(jnp.isfinite(rate))
            & jnp.all(closure.finite | ~geometry.cell.active)
        )
        physically_valid = (
            geometry.physically_valid
            & jnp.all(closure.physically_valid | ~geometry.cell.active)
            & jnp.all(jnp.stack(tuple(boundary_valid)))
        )
        qualified = physically_valid & jnp.all(closure.qualified | ~geometry.cell.active)
        derivative_valid = (
            qualified
            & jnp.all(closure.derivative_valid | ~geometry.cell.active)
            & (correction == 0.0)
        )
        return GRM1SpatialRate(
            rate,
            tuple(face_fluxes),
            tuple(integrated_fluxes),
            source,
            cell_projection,
            stable,
            jnp.max(closure.reduced_flux, initial=0.0),
            correction,
            finite,
            physically_valid,
            qualified,
            derivative_valid,
        )

    @staticmethod
    def _euler(base: Array, increment: Array, rate: GRM1SpatialRate, /) -> Array:
        return base + increment * rate.radiation_rate

    def _ledger(
        self,
        state: GRM1FiniteVolumeState,
        candidate: GRM1FiniteVolumeState,
        step: Array,
        rates: tuple[GRM1SpatialRate, GRM1SpatialRate, GRM1SpatialRate],
        accepted: Array,
        /,
    ) -> GRM1ConservationLedger:
        first, second, third = rates
        face_integrals = tuple(
            step * (one / 6.0 + two / 6.0 + 2.0 * three / 3.0)
            for one, two, three in zip(
                first.integrated_face_fluxes,
                second.integrated_face_fluxes,
                third.integrated_face_fluxes,
                strict=True,
            )
        )
        source_integral = step * (
            first.geometric_source / 6.0
            + second.geometric_source / 6.0
            + 2.0 * third.geometric_source / 3.0
        )
        change = candidate.radiation_state - state.radiation_state
        volumes = self.discretization.cell_volumes.astype(change.dtype)
        axes = tuple(range(len(self.cell_shape)))
        volume_change = jnp.sum(volumes[..., None] * change, axis=axes)
        volume_source = jnp.sum(volumes[..., None] * source_integral, axis=axes)
        boundary_flux = jnp.zeros_like(volume_change)
        for axis, integrated in enumerate(face_integrals):
            if self.discretization.grid.structured_axes[axis].periodic:
                continue
            lower = jnp.take(integrated, 0, axis=axis)
            upper = jnp.take(integrated, integrated.shape[axis] - 1, axis=axis)
            transverse = tuple(range(lower.ndim - 1))
            outward = upper - lower
            if transverse:
                outward = jnp.sum(outward, axis=transverse)
            boundary_flux = boundary_flux + outward
        defect = volume_change + boundary_flux - volume_source
        scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(volume_change), initial=0.0),
                jnp.maximum(
                    jnp.max(jnp.abs(boundary_flux), initial=0.0),
                    jnp.max(jnp.abs(volume_source), initial=0.0),
                ),
            ),
            1.0,
        )
        tolerance = jnp.maximum(
            jnp.asarray(self.balance_tolerance, dtype=change.dtype),
            256.0 * jnp.finfo(change.dtype).eps * scale,
        )
        finite = jnp.all(jnp.isfinite(defect))
        qualified = finite & (jnp.max(jnp.abs(defect), initial=0.0) <= tolerance)
        return GRM1ConservationLedger(
            face_integrals,
            source_integral,
            change,
            volume_change,
            boundary_flux,
            volume_source,
            defect,
            jnp.asarray(accepted, dtype=bool),
            finite,
            qualified,
            self.plan_id,
        )

    def _accepted_ledger(
        self, ledger: GRM1ConservationLedger, accepted: Array, /
    ) -> GRM1ConservationLedger:
        selected = lambda value: jnp.where(accepted, value, jnp.zeros_like(value))
        return GRM1ConservationLedger(
            tuple(selected(value) for value in ledger.face_flux_integrals),
            selected(ledger.geometric_source_integral),
            selected(ledger.state_change),
            selected(ledger.volume_integrated_change),
            selected(ledger.boundary_flux),
            selected(ledger.volume_integrated_source),
            selected(ledger.balance_defect),
            accepted,
            ledger.finite,
            ledger.qualified,
            ledger.plan_id,
        )

    def advance(
        self,
        state: GRM1FiniteVolumeState,
        start_time: ArrayLike,
        end_time: ArrayLike,
        stage_geometries: tuple[
            ValenciaFiniteVolumeStageGeometry,
            ValenciaFiniteVolumeStageGeometry,
            ValenciaFiniteVolumeStageGeometry,
        ],
        /,
        *,
        transport_extinction: ArrayLike = 0.0,
    ) -> GRM1StepResult:
        if not isinstance(state, GRM1FiniteVolumeState):
            raise TypeError("state must be GRM1FiniteVolumeState.")
        stages_ = tuple(stage_geometries)
        if len(stages_) != 3:
            raise ValueError("GR M1 SSPRK3 requires three stage geometries.")
        for stage in stages_:
            self._check_stage(stage)
        start = jnp.asarray(start_time, dtype=state.time.dtype).reshape(())
        end = jnp.asarray(end_time, dtype=state.time.dtype).reshape(())
        step = end - start
        time_tolerance = (
            32.0 * jnp.finfo(start.dtype).eps * jnp.maximum(jnp.abs(start), 1.0)
        )
        start = eqx.error_if(
            start,
            ~jnp.isfinite(start)
            | ~jnp.isfinite(end)
            | (step <= 0.0)
            | (jnp.abs(state.time - start) > time_tolerance),
            "GR M1 interval is invalid or state time is stale.",
        )
        base = self._state(state.radiation_state)
        rate_1 = self.rate(
            start,
            base,
            stages_[0],
            transport_extinction=transport_extinction,
        )
        value_1 = self._euler(base, step, rate_1)
        rate_2 = self.rate(
            end,
            value_1,
            stages_[1],
            transport_extinction=transport_extinction,
        )
        value_2 = 0.75 * base + 0.25 * self._euler(value_1, step, rate_2)
        rate_3 = self.rate(
            start + 0.5 * step,
            value_2,
            stages_[2],
            transport_extinction=transport_extinction,
        )
        value_3 = base / 3.0 + 2.0 / 3.0 * self._euler(value_2, step, rate_3)
        candidate = GRM1FiniteVolumeState(
            value_3,
            end,
            step,
            state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(int(GRM1FiniteVolumeRunStatus.SUCCESS), dtype=jnp.int32),
        )
        attempted = self._ledger(
            state, candidate, step, (rate_1, rate_2, rate_3), jnp.asarray(False)
        )
        final_moments = self.moments(value_3, stages_[2].cell)
        final_closure = self.system.closure(
            final_moments[..., 0], final_moments[..., 1:], stages_[2].cell
        )
        stage_values = (rate_1, rate_2, rate_3)
        finite = jnp.all(
            jnp.stack(tuple(value.finite for value in stage_values))
        ) & jnp.all(final_closure.finite | ~stages_[2].cell.active)
        physical = jnp.all(
            jnp.stack(tuple(value.physically_valid for value in stage_values))
        ) & jnp.all(final_closure.physically_valid | ~stages_[2].cell.active)
        qualified = (
            jnp.all(jnp.stack(tuple(value.qualified for value in stage_values)))
            & jnp.all(final_closure.qualified | ~stages_[2].cell.active)
            & attempted.qualified
        )
        derivative = jnp.all(
            jnp.stack(tuple(value.derivative_valid for value in stage_values))
        ) & jnp.all(final_closure.derivative_valid | ~stages_[2].cell.active)
        stable = (
            step
            <= jnp.min(jnp.stack(tuple(value.stable_step for value in stage_values)))
            + time_tolerance
        )
        successful = finite & physical & qualified & stable
        status = jnp.where(
            successful,
            int(GRM1FiniteVolumeRunStatus.SUCCESS),
            jnp.where(
                ~finite,
                int(GRM1FiniteVolumeRunStatus.NONFINITE_STATE),
                jnp.where(
                    ~physical,
                    int(GRM1FiniteVolumeRunStatus.REALIZABILITY_FAILED),
                    jnp.where(
                        ~stable,
                        int(GRM1FiniteVolumeRunStatus.STABILITY_LIMIT_EXCEEDED),
                        int(GRM1FiniteVolumeRunStatus.CONSERVATION_DEFECT),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        candidate = eqx.tree_at(lambda value: value.status, candidate, status)
        rejected = GRM1FiniteVolumeState(
            state.radiation_state,
            state.time,
            state.step_size,
            state.accepted_step,
            status,
        )
        accepted_state = jax.lax.cond(
            successful, lambda _: candidate, lambda _: rejected, operand=None
        )
        attempted = eqx.tree_at(lambda value: value.accepted, attempted, successful)
        accepted_ledger = self._accepted_ledger(attempted, successful)
        return GRM1StepResult(
            candidate,
            accepted_state,
            successful,
            status,
            stage_values,
            attempted,
            accepted_ledger,
            finite,
            physical,
            qualified,
            derivative,
        )


__all__ = [
    "FixedGridGRM1SSPRK3Plan",
    "GRM1BoundaryCondition",
    "GRM1BoundaryKind",
    "GRM1BoundaryPair",
    "GRM1ConservationLedger",
    "GRM1FiniteVolumeRunStatus",
    "GRM1FiniteVolumeState",
    "GRM1ReconstructionKind",
    "GRM1SpatialRate",
    "GRM1StepResult",
]
