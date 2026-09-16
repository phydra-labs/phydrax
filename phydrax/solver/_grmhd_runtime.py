#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._relativistic_hydrodynamics import ValenciaGeometrySource
from ..equations._relativistic_mhd import IdealValenciaGRMHDSystem
from ..metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ..metrix._metric import _metric_inverse
from ._grmhd_boundary import GRMHDBoundaryPair
from ._grmhd_ct import (
    GRMHDConstrainedTransportPlan,
    GRMHDCTRate,
    GRMHDCTState,
)


class GRMHDRunStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    GEOMETRY_INVALID = 2
    STABILITY_LIMIT_EXCEEDED = 3
    PRIMITIVE_RECOVERY_FAILED = 4
    UNQUALIFIED_HIGH_MAGNETIZATION = 5
    MAGNETIC_CONSTRAINT_FAILED = 6
    VECTOR_POTENTIAL_CONSTRAINT_FAILED = 7
    CONSERVATION_DEFECT = 8
    NONFINITE_STATE = 9


class GRMHDState(StrictModule):
    """Atomic material-cell and magnetic-cochain state at one accepted time."""

    material_state: Array
    constrained_transport: GRMHDCTState
    time: Array
    step_size: Array
    accepted_step: Array
    status: Array


class GRMHDSpatialRate(StrictModule):
    material_rate: Array
    transport_rate: GRMHDCTRate
    face_fluxes: tuple[Array, ...]
    integrated_face_fluxes: tuple[Array, ...]
    geometric_source: Array
    stress_energy: StressEnergyProjection
    stable_step: Array
    maximum_recovery_residual: Array
    maximum_magnetization: Array
    reconstruction_fallback: Array
    boundary_valid: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class GRMHDStageProposal(StrictModule):
    material_state: Array
    constrained_transport: GRMHDCTState
    rate: GRMHDSpatialRate
    stress_energy: StressEnergyProjection
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class GRMHDStageEvidence(StrictModule):
    stable_steps: Array
    maximum_recovery_residuals: Array
    maximum_magnetizations: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class GRMHDDefectLedger(StrictModule):
    face_flux_integrals: tuple[Array, ...]
    edge_electromotive_integral: Array
    geometric_source_integral: Array
    material_state_change: Array
    volume_integrated_material_change: Array
    boundary_material_flux: Array
    volume_integrated_source: Array
    magnetic_flux_change: Array
    material_balance_defect: Array
    faraday_balance_defect: Array
    magnetic_divergence_before: Array
    magnetic_divergence_after: Array
    vector_potential_defect: Array
    gauge_constraint: Array
    rest_mass_floor_added: Array
    energy_floor_added: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)


class GRMHDStepResult(StrictModule):
    candidate: GRMHDState
    state: GRMHDState
    accepted: Array
    status: Array
    stages: GRMHDStageEvidence
    attempted_ledger: GRMHDDefectLedger
    accepted_ledger: GRMHDDefectLedger
    finite: Array
    converged: Array
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


class GRMHDSSPRK3Plan(StrictModule, NonTrainableState):
    """Finite-volume Valencia GRMHD with compatible constrained transport.

    Primitive reconstruction, metric-aware bounded exterior states, material
    conservation, and magnetic Faraday evolution are accepted atomically.
    """

    system: IdealValenciaGRMHDSystem
    constrained_transport: GRMHDConstrainedTransportPlan
    boundaries: tuple[GRMHDBoundaryPair | None, ...]
    reconstruction: str = eqx.field(static=True)
    plm_theta: float = eqx.field(static=True)
    cfl: float = eqx.field(static=True)
    divergence_tolerance: float = eqx.field(static=True)
    balance_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: IdealValenciaGRMHDSystem,
        constrained_transport: GRMHDConstrainedTransportPlan,
        /,
        *,
        boundaries: tuple[GRMHDBoundaryPair | None, ...] | None = None,
        reconstruction: str = "piecewise_constant",
        plm_theta: float = 1.5,
        cfl: float = 0.35,
        divergence_tolerance: float = 1.0e-10,
        balance_tolerance: float = 1.0e-9,
    ):
        if not isinstance(system, IdealValenciaGRMHDSystem):
            raise TypeError("system must be IdealValenciaGRMHDSystem.")
        if not isinstance(constrained_transport, GRMHDConstrainedTransportPlan):
            raise TypeError(
                "constrained_transport must be GRMHDConstrainedTransportPlan."
            )
        dimension = constrained_transport.layout.dimension
        boundary_values = (
            tuple(None for _ in range(dimension))
            if boundaries is None
            else tuple(boundaries)
        )
        if len(boundary_values) != dimension or any(
            value is not None and not isinstance(value, GRMHDBoundaryPair)
            for value in boundary_values
        ):
            raise TypeError("One optional GRMHD boundary pair is required per axis.")
        for axis, pair in zip(
            constrained_transport.bridge.grid.structured_axes,
            boundary_values,
            strict=True,
        ):
            if axis.periodic != (pair is None):
                raise ValueError(
                    "Periodic GRMHD axes require no boundary pair and bounded axes require one."
                )
        if reconstruction not in ("piecewise_constant", "plm"):
            raise ValueError("GRMHD reconstruction must be piecewise_constant or plm.")
        theta = float(plm_theta)
        cfl_ = float(cfl)
        divergence = float(divergence_tolerance)
        balance = float(balance_tolerance)
        if (
            not np.isfinite(cfl_)
            or not 0.0 < cfl_ <= 1.0
            or not np.isfinite(divergence)
            or divergence < 0.0
            or not np.isfinite(balance)
            or balance < 0.0
            or not np.isfinite(theta)
            or not 1.0 <= theta <= 2.0
        ):
            raise ValueError("GRMHD SSPRK controls are invalid.")
        self.system = system
        self.constrained_transport = constrained_transport
        self.boundaries = boundary_values
        self.reconstruction = reconstruction
        self.plm_theta = theta
        self.cfl = cfl_
        self.divergence_tolerance = divergence
        self.balance_tolerance = balance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomic-valencia-grmhd-ssprk3",
                "system": system.system_id,
                "constrained_transport": constrained_transport.plan_id,
                "boundaries": [
                    None if value is None else value.pair_id for value in boundary_values
                ],
                "reconstruction": reconstruction,
                "plm_theta": theta,
                "cfl": cfl_,
                "divergence_tolerance": divergence,
                "balance_tolerance": balance,
            }
        )

    @property
    def cell_shape(self) -> tuple[int, ...]:
        return self.constrained_transport.cell_shape

    def _geometry(self, geometry: ADMGridGeometry, /) -> ADMGridGeometry:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if geometry.leading_shape != self.cell_shape:
            raise ValueError("ADM geometry shape does not match the GRMHD grid.")
        if (
            geometry.topology_id
            != self.constrained_transport.bridge.grid.topology.topology_id
        ):
            raise ValueError("ADM geometry and GRMHD grid topology identities differ.")
        if geometry.scale_id != self.system.scale.scale_id:
            raise ValueError("ADM geometry and GRMHD scale identities differ.")
        if geometry.convention_id != self.system.convention.convention_id:
            raise ValueError("ADM geometry and GRMHD convention identities differ.")
        alpha = eqx.error_if(
            geometry.alpha,
            ~jnp.all(geometry.active),
            "GRMHD runtime requires every cell active until an explicit "
            "excision-boundary flux is configured.",
        )
        return eqx.tree_at(lambda value: value.alpha, geometry, alpha)

    def _material_state(self, value: ArrayLike, /) -> Array:
        state = jnp.asarray(value)
        expected = self.cell_shape + (
            self.constrained_transport.layout.reduced_component_count,
        )
        if state.shape != expected:
            raise ValueError(f"GRMHD material state must have shape {expected}.")
        return state

    def initialize(
        self,
        conserved: ArrayLike,
        geometry: ADMGridGeometry,
        /,
        *,
        magnetic_flux: ArrayLike | None = None,
        vector_potential: ArrayLike | None = None,
        gauge_scalar: ArrayLike | None = None,
        composition: ArrayLike | None = None,
        time: ArrayLike = 0.0,
        step_size: ArrayLike | None = None,
    ) -> GRMHDState:
        geometry_ = self._geometry(geometry)
        full = self.system._state(conserved, "Initial Valencia conserved state")
        if full.shape[:-1] != self.cell_shape:
            raise ValueError(
                "Initial Valencia state shape does not match the GRMHD grid."
            )
        transport = self.constrained_transport.initialize(
            magnetic_flux,
            vector_potential=vector_potential,
            gauge_scalar=gauge_scalar,
        )
        material = self.constrained_transport.layout.reduce_full_state(full)
        synchronized = self.constrained_transport.full_state(
            material, transport.magnetic_flux
        )
        mismatch = jnp.max(jnp.abs(synchronized[..., 5:8] - full[..., 5:8]), initial=0.0)
        recovery = self.system.recover(synchronized, geometry_, composition)
        divergence = self.constrained_transport.magnetic_divergence(
            transport.magnetic_flux
        )
        valid = (
            geometry_.all_active_valid
            & jnp.all(recovery.qualified | ~geometry_.active)
            & (mismatch <= self.constrained_transport.compatibility_tolerance)
            & (jnp.max(jnp.abs(divergence), initial=0.0) <= self.divergence_tolerance)
        )
        material = eqx.error_if(
            material,
            ~valid,
            "Initial GRMHD state is inadmissible, unqualified, or CT-inconsistent.",
        )
        time_ = jnp.asarray(time, dtype=material.dtype).reshape(())
        step_ = jnp.asarray(
            jnp.nan if step_size is None else step_size, dtype=material.dtype
        ).reshape(())
        return GRMHDState(
            material,
            transport,
            time_,
            step_,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(GRMHDRunStatus.SUCCESS), dtype=jnp.int32),
        )

    @staticmethod
    def _face_average(value: Array, axis: int, periodic: bool, /) -> Array:
        if periodic:
            return 0.5 * (value + jnp.roll(value, -1, axis=axis))
        lower = jnp.take(value, jnp.asarray([0]), axis=axis)
        upper = jnp.take(value, jnp.asarray([value.shape[axis] - 1]), axis=axis)
        interior = 0.5 * (
            jnp.take(value, jnp.arange(value.shape[axis] - 1), axis=axis)
            + jnp.take(value, jnp.arange(1, value.shape[axis]), axis=axis)
        )
        return jnp.concatenate((lower, interior, upper), axis=axis)

    @staticmethod
    def _face_pair(value: Array, axis: int, periodic: bool, /) -> tuple[Array, Array]:
        if periodic:
            return value, jnp.roll(value, -1, axis=axis)
        lower_boundary = jnp.take(value, jnp.asarray([0]), axis=axis)
        upper_boundary = jnp.take(value, jnp.asarray([value.shape[axis] - 1]), axis=axis)
        left = jnp.concatenate((lower_boundary, value), axis=axis)
        right = jnp.concatenate((value, upper_boundary), axis=axis)
        return left, right

    def _primitive_face_pair(
        self,
        primitive: Array,
        normal_magnetic: Array,
        face_geometry: ADMGridGeometry,
        axis: int,
        periodic: bool,
        /,
    ) -> tuple[Array, Array, Array, Array]:
        values = jnp.moveaxis(primitive, axis, 0)
        boundary_valid = jnp.asarray(True)
        if periodic:
            piece_left = values
            piece_right = jnp.roll(values, -1, axis=0)
            if self.reconstruction == "piecewise_constant":
                left, right = piece_left, piece_right
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
        else:
            pair = self.boundaries[axis]
            if pair is None:
                raise RuntimeError("Bounded GRMHD axis has no boundary pair.")
            lower_geometry = _take_adm_geometry(face_geometry, 0, axis)
            upper_geometry = _take_adm_geometry(
                face_geometry, face_geometry.leading_shape[axis] - 1, axis
            )
            lower_normal = jnp.take(normal_magnetic, 0, axis=axis)
            upper_normal = jnp.take(
                normal_magnetic, normal_magnetic.shape[axis] - 1, axis=axis
            )
            lower = pair.lower.trace(
                jnp.take(primitive, 0, axis=axis),
                lower_normal / lower_geometry.sqrt_det_spatial_metric,
                lower_geometry,
                axis,
                "lower",
            )
            upper = pair.upper.trace(
                jnp.take(primitive, primitive.shape[axis] - 1, axis=axis),
                upper_normal / upper_geometry.sqrt_det_spatial_metric,
                upper_geometry,
                axis,
                "upper",
            )
            boundary_valid = jnp.all(lower.physically_valid) & jnp.all(
                upper.physically_valid
            )
            extended = jnp.concatenate(
                (
                    jnp.moveaxis(lower.exterior_primitive, axis, 0)[None, ...]
                    if lower.exterior_primitive.ndim == primitive.ndim
                    else lower.exterior_primitive[None, ...],
                    values,
                    jnp.moveaxis(upper.exterior_primitive, axis, 0)[None, ...]
                    if upper.exterior_primitive.ndim == primitive.ndim
                    else upper.exterior_primitive[None, ...],
                ),
                axis=0,
            )
            piece_left, piece_right = extended[:-1], extended[1:]
            if self.reconstruction == "piecewise_constant":
                left, right = piece_left, piece_right
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
        left = jnp.moveaxis(left, 0, axis)
        right = jnp.moveaxis(right, 0, axis)
        piece_left = jnp.moveaxis(piece_left, 0, axis)
        piece_right = jnp.moveaxis(piece_right, 0, axis)
        normal = normal_magnetic / face_geometry.sqrt_det_spatial_metric
        left = left.at[..., 5 + axis].set(normal)
        right = right.at[..., 5 + axis].set(normal)
        piece_left = piece_left.at[..., 5 + axis].set(normal)
        piece_right = piece_right.at[..., 5 + axis].set(normal)

        def primitive_valid(value: Array) -> Array:
            velocity_covector = ein.contract(
                "...ij,...j->...i",
                face_geometry.spatial_metric,
                value[..., 1:4],
            )
            speed_squared = ein.contract(
                "...i,...i->...", velocity_covector, value[..., 1:4]
            )
            return (
                jnp.all(jnp.isfinite(value), axis=-1)
                & (value[..., 0] >= self.system.density_floor)
                & (value[..., 4] >= self.system.pressure_floor)
                & (speed_squared < 1.0)
            )

        left_valid = primitive_valid(left)
        right_valid = primitive_valid(right)
        fallback = ~(left_valid & right_valid)
        left = jnp.where(left_valid[..., None], left, piece_left)
        right = jnp.where(right_valid[..., None], right, piece_right)
        return left, right, jnp.any(fallback), boundary_valid

    def _face_measure(self, axis: int, face_shape: tuple[int, ...], dtype, /) -> Array:
        measure = jnp.ones(face_shape, dtype=dtype)
        for transverse in range(self.constrained_transport.layout.dimension):
            if transverse == axis:
                continue
            widths = self.constrained_transport.bridge.grid.structured_axes[
                transverse
            ].interval_widths.astype(dtype)
            reshape = [1] * len(face_shape)
            reshape[transverse] = widths.size
            measure = measure * widths.reshape(reshape)
        return measure

    def _face_geometry(
        self,
        geometry: ADMGridGeometry,
        axis: int,
        /,
    ) -> ADMGridGeometry:
        periodic = self.constrained_transport.bridge.grid.structured_axes[axis].periodic
        spatial = self._face_average(geometry.spatial_metric, axis, periodic)
        spatial = 0.5 * (spatial + jnp.swapaxes(spatial, -1, -2))
        inverse = _metric_inverse(spatial, positive_definite=True)
        determinant = jnp.sqrt(jnp.linalg.det(spatial))
        active_numeric = self._face_average(
            geometry.active.astype(geometry.alpha.dtype), axis, periodic
        )
        valid_numeric = self._face_average(
            geometry.valid.astype(geometry.alpha.dtype), axis, periodic
        )
        return ADMGridGeometry(
            self._face_average(geometry.alpha, axis, periodic),
            self._face_average(geometry.beta_contravariant, axis, periodic),
            spatial,
            inverse,
            determinant,
            self._face_average(geometry.extrinsic_curvature, axis, periodic),
            active_numeric == 1.0,
            valid_numeric == 1.0,
            chart_id=geometry.chart_id,
            convention_id=geometry.convention_id,
            scale_id=geometry.scale_id,
            topology_id=geometry.topology_id,
            snapshot_token=geometry.snapshot_token,
            geometry_lineage_id=canonical_fingerprint(
                {
                    "kind": "grmhd-face-adm-geometry",
                    "cell_geometry_lineage": geometry.geometry_lineage_id,
                    "axis": axis,
                }
            ),
        )

    def _axis_derivative(self, value: Array, axis: int, /) -> Array:
        structured = self.constrained_transport.bridge.grid.structured_axes[axis]
        coordinates = structured.interval_centers.astype(value.dtype)
        reshape = [1] * value.ndim
        reshape[axis] = coordinates.size
        if structured.periodic:
            period = (structured.bounds[1] - structured.bounds[0]).astype(value.dtype)
            forward = jnp.roll(coordinates, -1) - coordinates
            backward = coordinates - jnp.roll(coordinates, 1)
            forward = jnp.where(forward <= 0.0, forward + period, forward)
            backward = jnp.where(backward <= 0.0, backward + period, backward)
            denominator = (forward + backward).reshape(reshape)
            return (
                jnp.roll(value, -1, axis=axis) - jnp.roll(value, 1, axis=axis)
            ) / denominator
        count = value.shape[axis]
        if count < 2:
            return jnp.zeros_like(value)
        forward_spacing = coordinates[1:] - coordinates[:-1]
        forward_shape = [1] * value.ndim
        forward_shape[axis] = count - 1
        differences = (
            jnp.take(value, jnp.arange(1, count), axis=axis)
            - jnp.take(value, jnp.arange(count - 1), axis=axis)
        ) / forward_spacing.reshape(forward_shape)
        lower = jnp.take(differences, jnp.asarray([0]), axis=axis)
        upper = jnp.take(differences, jnp.asarray([count - 2]), axis=axis)
        if count == 2:
            return jnp.concatenate((lower, upper), axis=axis)
        interior_left = jnp.take(differences, jnp.arange(count - 2), axis=axis)
        interior_right = jnp.take(differences, jnp.arange(1, count - 1), axis=axis)
        left_width = forward_spacing[:-1]
        right_width = forward_spacing[1:]
        width_shape = [1] * value.ndim
        width_shape[axis] = count - 2
        interior = (
            right_width.reshape(width_shape) * interior_left
            + left_width.reshape(width_shape) * interior_right
        ) / (left_width + right_width).reshape(width_shape)
        return jnp.concatenate((lower, interior, upper), axis=axis)

    def metric_derivatives(
        self,
        geometry: ADMGridGeometry,
        /,
    ) -> ValenciaGeometrySource:
        geometry_ = self._geometry(geometry)
        lapse = jnp.zeros(self.cell_shape + (3,), dtype=geometry_.alpha.dtype)
        shift = jnp.zeros(self.cell_shape + (3, 3), dtype=geometry_.alpha.dtype)
        spatial = jnp.zeros(self.cell_shape + (3, 3, 3), dtype=geometry_.alpha.dtype)
        for axis in range(self.constrained_transport.layout.dimension):
            lapse = lapse.at[..., axis].set(self._axis_derivative(geometry_.alpha, axis))
            shift = shift.at[..., axis, :].set(
                self._axis_derivative(geometry_.beta_contravariant, axis)
            )
            spatial = spatial.at[..., axis, :, :].set(
                self._axis_derivative(geometry_.spatial_metric, axis)
            )
        return ValenciaGeometrySource(geometry_, lapse, shift, spatial)

    def rate(
        self,
        time: ArrayLike,
        material_state: ArrayLike,
        transport_state: GRMHDCTState,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None = None,
        /,
        *,
        source_geometry: ValenciaGeometrySource | None = None,
    ) -> GRMHDSpatialRate:
        del time
        geometry_ = self._geometry(geometry)
        material = self._material_state(material_state)
        full = self.constrained_transport.full_state(
            material, transport_state.magnetic_flux
        )
        cell_recovery = self.system.recover(full, geometry_, composition)
        projection = self.system.stress_energy(
            cell_recovery.primitive,
            geometry_,
            composition,
            conserved=full,
        )
        normal_fields = self.constrained_transport.bridge.unpack_normal_flux(
            transport_state.magnetic_flux
        )
        fluxes = []
        speeds = []
        finite_faces = []
        physical_faces = []
        converged_faces = []
        qualified_faces = []
        derivative_faces = []
        residual_faces = []
        magnetization_faces = []
        reconstruction_fallbacks = []
        boundary_valids = []
        for axis in range(self.constrained_transport.layout.dimension):
            periodic = self.constrained_transport.bridge.grid.structured_axes[
                axis
            ].periodic
            face_geometry = self._face_geometry(geometry_, axis)
            normal_densitized = normal_fields[axis]
            (
                left_primitive,
                right_primitive,
                reconstruction_fallback,
                boundary_valid,
            ) = self._primitive_face_pair(
                cell_recovery.primitive,
                normal_densitized,
                face_geometry,
                axis,
                periodic,
            )
            if composition is None:
                left_composition = None
                right_composition = None
                face_composition = None
            else:
                composition_ = jnp.asarray(composition)
                if composition_.shape == ():
                    left_composition = composition_
                    right_composition = composition_
                    face_composition = composition_
                else:
                    left_composition, right_composition = self._face_pair(
                        composition_, axis, periodic
                    )
                    face_composition = 0.5 * (left_composition + right_composition)
            left = self.system.primitive_to_conserved(
                left_primitive, face_geometry, left_composition
            )
            right = self.system.primitive_to_conserved(
                right_primitive, face_geometry, right_composition
            )
            result = self.system.hlle_flux(
                left,
                right,
                face_geometry,
                axis,
                face_composition,
            )
            fluxes.append(result.normal_flux)
            speeds.append(result.maximum_speed)
            reconstruction_fallbacks.append(reconstruction_fallback)
            boundary_valids.append(boundary_valid)
            finite_faces.append(jnp.all(result.finite | ~face_geometry.active))
            physical_faces.append(
                jnp.all(result.physically_valid | ~face_geometry.active)
            )
            qualified_faces.append(jnp.all(result.qualified | ~face_geometry.active))
            converged_faces.append(
                jnp.all(
                    (result.left_recovery.converged & result.right_recovery.converged)
                    | ~face_geometry.active
                )
            )
            derivative_faces.append(
                jnp.all(
                    (
                        result.left_recovery.derivative_valid
                        & result.right_recovery.derivative_valid
                    )
                    | ~face_geometry.active
                )
            )
            residual_faces.append(
                jnp.maximum(
                    jnp.max(jnp.abs(result.left_recovery.residual), initial=0.0),
                    jnp.max(jnp.abs(result.right_recovery.residual), initial=0.0),
                )
            )
            magnetization_faces.append(
                jnp.maximum(
                    jnp.max(result.left_recovery.magnetization, initial=0.0),
                    jnp.max(result.right_recovery.magnetization, initial=0.0),
                )
            )
        flux_tuple = tuple(fluxes)
        speed_tuple = tuple(speeds)
        residual = jnp.zeros_like(full)
        inverse_dt = jnp.zeros(self.cell_shape, dtype=full.dtype)
        cell_volumes = self.constrained_transport.bridge.grid.measure.weights.reshape(
            self.cell_shape
        ).astype(full.dtype)
        integrated_fluxes = []
        for axis, (flux, speed) in enumerate(zip(flux_tuple, speed_tuple, strict=True)):
            periodic = self.constrained_transport.bridge.grid.structured_axes[
                axis
            ].periodic
            face_measures = self._face_measure(
                axis,
                normal_fields[axis].shape,
                full.dtype,
            )
            integrated = flux * face_measures[..., None]
            integrated_fluxes.append(integrated)
            if periodic:
                residual = (
                    residual
                    - (integrated - jnp.roll(integrated, 1, axis=axis))
                    / cell_volumes[..., None]
                )
                inverse_dt = inverse_dt + speed * face_measures / cell_volumes
            else:
                lower_indices = jnp.arange(integrated.shape[axis] - 1)
                upper_indices = jnp.arange(1, integrated.shape[axis])
                lower_flux = jnp.take(integrated, lower_indices, axis=axis)
                upper_flux = jnp.take(integrated, upper_indices, axis=axis)
                residual = residual - (upper_flux - lower_flux) / cell_volumes[..., None]
                face_rate = speed * face_measures
                inverse_dt = (
                    inverse_dt
                    + jnp.maximum(
                        jnp.take(face_rate, lower_indices, axis=axis),
                        jnp.take(face_rate, upper_indices, axis=axis),
                    )
                    / cell_volumes
                )
        residual = residual.at[
            ..., jnp.asarray(self.constrained_transport.layout.face_magnetic_indices)
        ].set(0.0)
        source_geometry_ = (
            self.metric_derivatives(geometry_)
            if source_geometry is None
            else source_geometry
        )
        source = self.system.geometric_source_from_projection(
            projection, source_geometry_
        )
        cell_rate = residual + source
        edge, uct_defect, uct_dissipation = self.constrained_transport.edge_electromotive(
            full,
            flux_tuple,
            speed_tuple,
        )
        transport_rate = self.constrained_transport.rate(
            transport_state,
            edge,
            uct_consistency_defect=uct_defect,
            uct_maximum_dissipation=uct_dissipation,
        )
        maximum_inverse_step = jnp.max(inverse_dt, initial=0.0)
        stable = jnp.where(
            maximum_inverse_step > 0.0,
            jnp.asarray(self.cfl, dtype=full.dtype) / maximum_inverse_step,
            jnp.asarray(jnp.inf, dtype=full.dtype),
        )
        cell_active = geometry_.active
        boundary_valid = jnp.all(jnp.stack(tuple(boundary_valids)))
        reconstruction_fallback = jnp.any(jnp.stack(tuple(reconstruction_fallbacks)))
        finite = (
            jnp.all(cell_recovery.finite | ~cell_active)
            & jnp.all(jnp.stack(tuple(finite_faces)))
            & jnp.all(jnp.isfinite(cell_rate))
            & jnp.all(jnp.isfinite(transport_rate.magnetic_rate))
        )
        converged = jnp.all(cell_recovery.converged | ~cell_active) & jnp.all(
            jnp.stack(tuple(converged_faces))
        )
        physically_valid = (
            jnp.all(cell_recovery.physically_valid | ~cell_active)
            & jnp.all(jnp.stack(tuple(physical_faces)))
            & jnp.all(source_geometry_.physically_valid | ~cell_active)
            & boundary_valid
        )
        qualified = (
            jnp.all(cell_recovery.qualified | ~cell_active)
            & jnp.all(jnp.stack(tuple(qualified_faces)))
            & physically_valid
        )
        derivative_valid = (
            jnp.all(cell_recovery.derivative_valid | ~cell_active)
            & jnp.all(jnp.stack(tuple(derivative_faces)))
            & ~reconstruction_fallback
        )
        maximum_residual = jnp.maximum(
            jnp.max(jnp.abs(cell_recovery.residual), initial=0.0),
            jnp.max(jnp.stack(tuple(residual_faces)), initial=0.0),
        )
        maximum_magnetization = jnp.maximum(
            jnp.max(cell_recovery.magnetization, initial=0.0),
            jnp.max(jnp.stack(tuple(magnetization_faces)), initial=0.0),
        )
        return GRMHDSpatialRate(
            material_rate=cell_rate[
                ...,
                jnp.asarray(self.constrained_transport.layout.reduced_component_indices),
            ],
            transport_rate=transport_rate,
            face_fluxes=flux_tuple,
            integrated_face_fluxes=tuple(integrated_fluxes),
            geometric_source=source[
                ...,
                jnp.asarray(self.constrained_transport.layout.reduced_component_indices),
            ],
            stress_energy=projection,
            stable_step=stable,
            maximum_recovery_residual=maximum_residual,
            maximum_magnetization=maximum_magnetization,
            reconstruction_fallback=reconstruction_fallback,
            boundary_valid=boundary_valid,
            finite=finite,
            converged=converged,
            physically_valid=physically_valid,
            qualified=qualified,
            derivative_valid=derivative_valid,
        )

    @staticmethod
    def _combine_transport(
        first_weight: Array,
        first: GRMHDCTState,
        second_weight: Array,
        second: GRMHDCTState,
        /,
    ) -> GRMHDCTState:
        return GRMHDCTState(
            first_weight * first.magnetic_flux + second_weight * second.magnetic_flux,
            first_weight * first.vector_potential
            + second_weight * second.vector_potential,
            first_weight * first.gauge_scalar + second_weight * second.gauge_scalar,
        )

    @staticmethod
    def _euler_transport(
        base: GRMHDCTState,
        increment: Array,
        rate: GRMHDCTRate,
        /,
    ) -> GRMHDCTState:
        return GRMHDCTState(
            base.magnetic_flux + increment * rate.magnetic_rate,
            base.vector_potential + increment * rate.vector_potential_rate,
            base.gauge_scalar + increment * rate.gauge_scalar_rate,
        )

    def _stage(
        self,
        evaluation_time: Array,
        evaluation_material: Array,
        evaluation_transport: GRMHDCTState,
        base_material: Array,
        base_transport: GRMHDCTState,
        increment: Array,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None,
        source_geometry: ValenciaGeometrySource | None,
        /,
    ) -> tuple[Array, GRMHDCTState, GRMHDSpatialRate]:
        rate = self.rate(
            evaluation_time,
            evaluation_material,
            evaluation_transport,
            geometry,
            composition,
            source_geometry=source_geometry,
        )
        return (
            base_material + increment * rate.material_rate,
            self._euler_transport(base_transport, increment, rate.transport_rate),
            rate,
        )

    def propose_stage(
        self,
        evaluation_time: ArrayLike,
        working_material: ArrayLike,
        working_transport: GRMHDCTState,
        base_material: ArrayLike,
        base_transport: GRMHDCTState,
        increment: ArrayLike,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None = None,
        /,
        *,
        source_geometry: ValenciaGeometrySource | None = None,
    ) -> GRMHDStageProposal:
        """Propose one pure SSPRK recurrence stage for a coupled coordinator."""
        geometry_ = self._geometry(geometry)
        working = self._material_state(working_material)
        base = self._material_state(base_material)
        increment_ = jnp.asarray(increment, dtype=working.dtype).reshape(())
        material, transport, rate = self._stage(
            jnp.asarray(evaluation_time, dtype=working.dtype).reshape(()),
            working,
            working_transport,
            base,
            base_transport,
            increment_,
            geometry_,
            composition,
            source_geometry,
        )
        return GRMHDStageProposal(
            material,
            transport,
            rate,
            rate.stress_energy,
            rate.finite,
            rate.converged,
            rate.physically_valid,
            rate.qualified,
            rate.derivative_valid,
        )

    def _ledger(
        self,
        state: GRMHDState,
        candidate: GRMHDState,
        step: Array,
        rates: tuple[GRMHDSpatialRate, GRMHDSpatialRate, GRMHDSpatialRate],
        accepted: Array,
        /,
    ) -> GRMHDDefectLedger:
        first, second, third = rates
        face_integrals = tuple(
            step * (a / 6.0 + b / 6.0 + 2.0 * c / 3.0)
            for a, b, c in zip(
                first.integrated_face_fluxes,
                second.integrated_face_fluxes,
                third.integrated_face_fluxes,
                strict=True,
            )
        )
        edge_integral = step * (
            first.transport_rate.edge_electromotive_circulation / 6.0
            + second.transport_rate.edge_electromotive_circulation / 6.0
            + 2.0 * third.transport_rate.edge_electromotive_circulation / 3.0
        )
        source_integral = step * (
            first.geometric_source / 6.0
            + second.geometric_source / 6.0
            + 2.0 * third.geometric_source / 3.0
        )
        material_change = candidate.material_state - state.material_state
        cell_volumes = self.constrained_transport.bridge.grid.measure.weights.reshape(
            self.cell_shape
        ).astype(material_change.dtype)
        spatial_axes = tuple(range(len(self.cell_shape)))
        volume_material_change = jnp.sum(
            cell_volumes[..., None] * material_change,
            axis=spatial_axes,
        )
        volume_source = jnp.sum(
            cell_volumes[..., None] * source_integral,
            axis=spatial_axes,
        )
        reduced_indices = jnp.asarray(
            self.constrained_transport.layout.reduced_component_indices
        )
        boundary_flux = jnp.zeros_like(volume_material_change)
        for axis, integrated in enumerate(face_integrals):
            if self.constrained_transport.bridge.grid.structured_axes[axis].periodic:
                continue
            material_flux = integrated[..., reduced_indices]
            lower = jnp.take(material_flux, 0, axis=axis)
            upper = jnp.take(material_flux, material_flux.shape[axis] - 1, axis=axis)
            transverse_axes = tuple(range(lower.ndim - 1))
            outward = upper - lower
            if transverse_axes:
                outward = jnp.sum(outward, axis=transverse_axes)
            boundary_flux = boundary_flux + outward
        material_defect = volume_material_change + boundary_flux - volume_source
        ct = self.constrained_transport.defects(
            state.constrained_transport,
            candidate.constrained_transport,
            edge_integral,
        )
        finite = (
            jnp.all(jnp.isfinite(material_change))
            & jnp.all(jnp.isfinite(volume_material_change))
            & jnp.all(jnp.isfinite(boundary_flux))
            & jnp.all(jnp.isfinite(volume_source))
            & jnp.all(jnp.isfinite(material_defect))
            & ct.finite
        )
        maximum_material_defect = jnp.max(jnp.abs(material_defect), initial=0.0)
        material_scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(volume_material_change), initial=0.0),
                jnp.max(jnp.abs(volume_source), initial=0.0),
            ),
            1.0,
        )
        material_tolerance = jnp.maximum(
            jnp.asarray(self.balance_tolerance, dtype=material_change.dtype),
            256.0 * jnp.finfo(material_change.dtype).eps * material_scale,
        )
        physically_valid = finite & ct.physically_valid
        qualified = (
            physically_valid
            & ct.qualified
            & (maximum_material_defect <= material_tolerance)
        )
        return GRMHDDefectLedger(
            face_flux_integrals=face_integrals,
            edge_electromotive_integral=edge_integral,
            geometric_source_integral=source_integral,
            material_state_change=material_change,
            volume_integrated_material_change=volume_material_change,
            boundary_material_flux=boundary_flux,
            volume_integrated_source=volume_source,
            magnetic_flux_change=ct.magnetic_flux_change,
            material_balance_defect=material_defect,
            faraday_balance_defect=ct.faraday_balance_defect,
            magnetic_divergence_before=ct.divergence_before,
            magnetic_divergence_after=ct.divergence_after,
            vector_potential_defect=ct.vector_potential_defect,
            gauge_constraint=ct.gauge_constraint,
            rest_mass_floor_added=jnp.asarray(0.0, dtype=material_change.dtype),
            energy_floor_added=jnp.asarray(0.0, dtype=material_change.dtype),
            finite=finite,
            physically_valid=physically_valid,
            qualified=qualified,
            accepted=jnp.asarray(accepted, dtype=bool),
            plan_id=self.plan_id,
        )

    def _accepted_ledger(
        self,
        attempted: GRMHDDefectLedger,
        accepted: Array,
        /,
    ) -> GRMHDDefectLedger:
        def selected(value):
            return jnp.where(accepted, value, jnp.zeros_like(value))

        return GRMHDDefectLedger(
            face_flux_integrals=tuple(
                selected(value) for value in attempted.face_flux_integrals
            ),
            edge_electromotive_integral=selected(attempted.edge_electromotive_integral),
            geometric_source_integral=selected(attempted.geometric_source_integral),
            material_state_change=selected(attempted.material_state_change),
            volume_integrated_material_change=selected(
                attempted.volume_integrated_material_change
            ),
            boundary_material_flux=selected(attempted.boundary_material_flux),
            volume_integrated_source=selected(attempted.volume_integrated_source),
            magnetic_flux_change=selected(attempted.magnetic_flux_change),
            material_balance_defect=selected(attempted.material_balance_defect),
            faraday_balance_defect=selected(attempted.faraday_balance_defect),
            magnetic_divergence_before=selected(attempted.magnetic_divergence_before),
            magnetic_divergence_after=selected(attempted.magnetic_divergence_after),
            vector_potential_defect=selected(attempted.vector_potential_defect),
            gauge_constraint=selected(attempted.gauge_constraint),
            rest_mass_floor_added=selected(attempted.rest_mass_floor_added),
            energy_floor_added=selected(attempted.energy_floor_added),
            finite=attempted.finite,
            physically_valid=attempted.physically_valid,
            qualified=attempted.qualified,
            accepted=accepted,
            plan_id=attempted.plan_id,
        )

    def advance(
        self,
        state: GRMHDState,
        start_time: ArrayLike,
        end_time: ArrayLike,
        geometry: ADMGridGeometry,
        composition: ArrayLike | None = None,
        /,
        *,
        source_geometry: ValenciaGeometrySource | None = None,
    ) -> GRMHDStepResult:
        if not isinstance(state, GRMHDState):
            raise TypeError("state must be GRMHDState.")
        geometry_ = self._geometry(geometry)
        start = jnp.asarray(start_time, dtype=state.time.dtype).reshape(())
        end = jnp.asarray(end_time, dtype=state.time.dtype).reshape(())
        step = end - start
        tolerance = 32.0 * jnp.finfo(start.dtype).eps * jnp.maximum(jnp.abs(start), 1.0)
        start = eqx.error_if(
            start,
            ~jnp.isfinite(start)
            | ~jnp.isfinite(end)
            | (step <= 0.0)
            | (jnp.abs(state.time - start) > tolerance),
            "GRMHD interval is invalid or the state time is stale.",
        )
        del start
        material_0 = self._material_state(state.material_state)
        transport_0 = state.constrained_transport
        material_1, transport_1, rate_1 = self._stage(
            state.time,
            material_0,
            transport_0,
            material_0,
            transport_0,
            step,
            geometry_,
            composition,
            source_geometry,
        )
        base_material_2 = 0.75 * material_0 + 0.25 * material_1
        base_transport_2 = self._combine_transport(
            jnp.asarray(0.75, dtype=step.dtype),
            transport_0,
            jnp.asarray(0.25, dtype=step.dtype),
            transport_1,
        )
        material_2, transport_2, rate_2 = self._stage(
            state.time + step,
            material_1,
            transport_1,
            base_material_2,
            base_transport_2,
            0.25 * step,
            geometry_,
            composition,
            source_geometry,
        )
        base_material_3 = material_0 / 3.0 + 2.0 * material_2 / 3.0
        base_transport_3 = self._combine_transport(
            jnp.asarray(1.0 / 3.0, dtype=step.dtype),
            transport_0,
            jnp.asarray(2.0 / 3.0, dtype=step.dtype),
            transport_2,
        )
        material_3, transport_3, rate_3 = self._stage(
            state.time + 0.5 * step,
            material_2,
            transport_2,
            base_material_3,
            base_transport_3,
            2.0 * step / 3.0,
            geometry_,
            composition,
            source_geometry,
        )
        stable_steps = jnp.stack(
            (rate_1.stable_step, rate_2.stable_step, rate_3.stable_step)
        )
        stage_finite = jnp.stack((rate_1.finite, rate_2.finite, rate_3.finite))
        stage_converged = jnp.stack(
            (rate_1.converged, rate_2.converged, rate_3.converged)
        )
        stage_physical = jnp.stack(
            (
                rate_1.physically_valid,
                rate_2.physically_valid,
                rate_3.physically_valid,
            )
        )
        stage_qualified = jnp.stack(
            (rate_1.qualified, rate_2.qualified, rate_3.qualified)
        )
        stage_derivative = jnp.stack(
            (
                rate_1.derivative_valid,
                rate_2.derivative_valid,
                rate_3.derivative_valid,
            )
        )
        candidate = GRMHDState(
            material_3,
            transport_3,
            end,
            step,
            state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(int(GRMHDRunStatus.SUCCESS), dtype=jnp.int32),
        )
        attempted = self._ledger(
            state,
            candidate,
            step,
            (rate_1, rate_2, rate_3),
            jnp.asarray(False),
        )
        candidate_full = self.constrained_transport.full_state(
            candidate.material_state,
            candidate.constrained_transport.magnetic_flux,
        )
        candidate_recovery = self.system.recover(candidate_full, geometry_, composition)
        candidate_active = geometry_.active
        finite = (
            jnp.all(stage_finite)
            & attempted.finite
            & jnp.all(candidate_recovery.finite | ~candidate_active)
        )
        converged = jnp.all(stage_converged) & jnp.all(
            candidate_recovery.converged | ~candidate_active
        )
        physically_valid = (
            jnp.all(stage_physical)
            & attempted.physically_valid
            & jnp.all(candidate_recovery.physically_valid | ~candidate_active)
        )
        qualified = (
            jnp.all(stage_qualified)
            & attempted.qualified
            & jnp.all(candidate_recovery.qualified | ~candidate_active)
        )
        derivative_valid = jnp.all(stage_derivative) & jnp.all(
            candidate_recovery.derivative_valid | ~candidate_active
        )
        stable = step <= jnp.min(stable_steps) + tolerance
        magnetic_scale = jnp.maximum(
            jnp.max(
                jnp.abs(candidate.constrained_transport.magnetic_flux),
                initial=0.0,
            ),
            1.0,
        )
        magnetic_roundoff = (
            256.0 * jnp.finfo(candidate.material_state.dtype).eps * magnetic_scale
        )
        divergence_valid = jnp.max(
            jnp.abs(attempted.magnetic_divergence_after), initial=0.0
        ) <= jnp.maximum(self.divergence_tolerance, magnetic_roundoff)
        vector_valid = jnp.max(
            jnp.abs(attempted.vector_potential_defect), initial=0.0
        ) <= jnp.maximum(
            self.constrained_transport.compatibility_tolerance,
            magnetic_roundoff,
        )
        material_scale = jnp.maximum(
            jnp.maximum(
                jnp.max(
                    jnp.abs(attempted.volume_integrated_material_change),
                    initial=0.0,
                ),
                jnp.maximum(
                    jnp.max(
                        jnp.abs(attempted.volume_integrated_source),
                        initial=0.0,
                    ),
                    jnp.max(
                        jnp.abs(attempted.boundary_material_flux),
                        initial=0.0,
                    ),
                ),
            ),
            1.0,
        )
        material_roundoff = (
            256.0 * jnp.finfo(candidate.material_state.dtype).eps * material_scale
        )
        balance_valid = (
            jnp.max(jnp.abs(attempted.material_balance_defect), initial=0.0)
            <= jnp.maximum(self.balance_tolerance, material_roundoff)
        ) & (
            jnp.max(jnp.abs(attempted.faraday_balance_defect), initial=0.0)
            <= jnp.maximum(
                self.constrained_transport.compatibility_tolerance,
                magnetic_roundoff,
            )
        )
        step_qualified = (
            qualified
            & geometry_.all_active_valid
            & stable
            & divergence_valid
            & vector_valid
            & balance_valid
        )
        successful = (
            geometry_.all_active_valid
            & finite
            & converged
            & physically_valid
            & qualified
            & stable
            & divergence_valid
            & vector_valid
            & balance_valid
        )
        maximum_magnetization = jnp.maximum(
            jnp.max(
                jnp.stack(
                    (
                        rate_1.maximum_magnetization,
                        rate_2.maximum_magnetization,
                        rate_3.maximum_magnetization,
                    )
                )
            ),
            jnp.max(candidate_recovery.magnetization, initial=0.0),
        )
        status = jnp.where(
            successful,
            int(GRMHDRunStatus.SUCCESS),
            jnp.where(
                ~geometry_.all_active_valid,
                int(GRMHDRunStatus.GEOMETRY_INVALID),
                jnp.where(
                    ~finite,
                    int(GRMHDRunStatus.NONFINITE_STATE),
                    jnp.where(
                        ~converged | ~physically_valid,
                        int(GRMHDRunStatus.PRIMITIVE_RECOVERY_FAILED),
                        jnp.where(
                            maximum_magnetization > self.system.maximum_magnetization,
                            int(GRMHDRunStatus.UNQUALIFIED_HIGH_MAGNETIZATION),
                            jnp.where(
                                ~stable,
                                int(GRMHDRunStatus.STABILITY_LIMIT_EXCEEDED),
                                jnp.where(
                                    ~divergence_valid,
                                    int(GRMHDRunStatus.MAGNETIC_CONSTRAINT_FAILED),
                                    jnp.where(
                                        ~vector_valid,
                                        int(
                                            GRMHDRunStatus.VECTOR_POTENTIAL_CONSTRAINT_FAILED
                                        ),
                                        int(GRMHDRunStatus.CONSERVATION_DEFECT),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        candidate = eqx.tree_at(lambda value: value.status, candidate, status)
        rejected = GRMHDState(
            state.material_state,
            state.constrained_transport,
            state.time,
            state.step_size,
            state.accepted_step,
            status,
        )
        accepted_state = jax.lax.cond(
            successful,
            lambda _: candidate,
            lambda _: rejected,
            operand=None,
        )
        attempted = eqx.tree_at(
            lambda value: value.accepted,
            attempted,
            successful,
        )
        accepted_ledger = self._accepted_ledger(attempted, successful)
        stages = GRMHDStageEvidence(
            stable_steps=stable_steps,
            maximum_recovery_residuals=jnp.stack(
                (
                    rate_1.maximum_recovery_residual,
                    rate_2.maximum_recovery_residual,
                    rate_3.maximum_recovery_residual,
                )
            ),
            maximum_magnetizations=jnp.stack(
                (
                    rate_1.maximum_magnetization,
                    rate_2.maximum_magnetization,
                    rate_3.maximum_magnetization,
                )
            ),
            finite=stage_finite,
            converged=stage_converged,
            physically_valid=stage_physical,
            qualified=stage_qualified,
            derivative_valid=stage_derivative,
        )
        return GRMHDStepResult(
            candidate,
            accepted_state,
            successful,
            status,
            stages,
            attempted,
            accepted_ledger,
            finite,
            converged,
            physically_valid,
            step_qualified,
            derivative_valid,
        )


__all__ = [
    "GRMHDDefectLedger",
    "GRMHDRunStatus",
    "GRMHDSSPRK3Plan",
    "GRMHDSpatialRate",
    "GRMHDStageEvidence",
    "GRMHDStageProposal",
    "GRMHDState",
    "GRMHDStepResult",
]
