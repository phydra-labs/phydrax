#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Metric-aware fixed-grid finite-volume evolution for Valencia GRHD."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._fv_precision import FiniteVolumePrecisionPolicy
from ..discretization.finite_volume._reconstruction import (
    AbstractFaceReconstructionPlan,
    MCLimiter,
    MUSCLReconstruction,
    PiecewiseConstantReconstruction,
)
from ..discretization.finite_volume._structured import FiniteVolumeDiscretization
from ..equations._relativistic_hydrodynamics import (
    ValenciaGeometrySource,
    ValenciaGRHDSystem,
)
from ..linalg import inverse_small_linear, SmallLinearSolvePlan
from ..metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ._finite_volume_content import FiniteVolumeConservativeContentState
from ._fixed_step import AbstractFixedStepMethod, FixedStepResult
from ._relativistic_primitive import (
    AtmosphereFloorStatus,
    GRHDC2PPolicy,
    GRHDC2PResult,
)


GRHDBoundaryKind: TypeAlias = Literal["outflow", "reflective", "atmosphere"]
GRHDFaceFluxKind: TypeAlias = Literal["hlle", "rusanov"]


class GRHDFiniteVolumeRunStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    INVALID_GEOMETRY = 2
    C2P_FAILED = 3
    STABILITY_LIMIT_EXCEEDED = 4
    ATMOSPHERE_BUDGET_EXCEEDED = 5
    NONFINITE_STATE = 6
    CONSERVATION_DEFECT = 7


class GRHDBoundaryCondition(StrictModule, NonTrainableState):
    """One metric-aware exterior primitive-state policy."""

    kind: GRHDBoundaryKind = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(self, kind: GRHDBoundaryKind = "outflow", /):
        if kind not in ("outflow", "reflective", "atmosphere"):
            raise ValueError("GRHD boundary kind is unsupported.")
        self.kind = kind
        self.boundary_id = canonical_fingerprint(
            {"kind": "grhd-boundary-condition", "policy": kind}
        )


class GRHDBoundaryPair(StrictModule, NonTrainableState):
    lower: GRHDBoundaryCondition
    upper: GRHDBoundaryCondition
    pair_id: str = eqx.field(static=True)

    def __init__(self, lower: GRHDBoundaryCondition, upper: GRHDBoundaryCondition, /):
        if not isinstance(lower, GRHDBoundaryCondition) or not isinstance(
            upper, GRHDBoundaryCondition
        ):
            raise TypeError("GRHD boundaries require GRHDBoundaryCondition values.")
        self.lower = lower
        self.upper = upper
        self.pair_id = canonical_fingerprint(
            {
                "kind": "grhd-boundary-pair",
                "lower": lower.boundary_id,
                "upper": upper.boundary_id,
            }
        )


class GRHDFaceFluxResult(StrictModule):
    normal_flux: Array
    maximum_speed: Array
    finite: Array
    physically_valid: Array
    fallback_activated: Array
    flux_plan_id: str = eqx.field(static=True)


class GRHDFaceFluxPlan(StrictModule, NonTrainableState):
    """Eulerian-normal Valencia HLLE/Rusanov flux on coordinate faces."""

    kind: GRHDFaceFluxKind = eqx.field(static=True)
    flux_id: str = eqx.field(static=True)

    def __init__(self, kind: GRHDFaceFluxKind = "hlle", /):
        if kind not in ("hlle", "rusanov"):
            raise ValueError("GRHD face flux must be 'hlle' or 'rusanov'.")
        self.kind = kind
        self.flux_id = canonical_fingerprint(
            {
                "kind": "grhd-face-flux",
                "solver": kind,
                "frame": "eulerian-unit-normal",
                "representation": "coordinate-valencia-flux",
            }
        )

    def evaluate(
        self,
        system: ValenciaGRHDSystem,
        left_primitive: ArrayLike,
        right_primitive: ArrayLike,
        geometry: ADMGridGeometry,
        axis: int,
        /,
    ) -> GRHDFaceFluxResult:
        if not isinstance(system, ValenciaGRHDSystem):
            raise TypeError("system must be a ValenciaGRHDSystem.")
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be an ADMGridGeometry.")
        axis_ = int(axis)
        if not 0 <= axis_ < 3:
            raise ValueError("GRHD face axis is out of range.")
        left = jnp.asarray(left_primitive, dtype=geometry.alpha.dtype)
        right = jnp.asarray(right_primitive, dtype=geometry.alpha.dtype)
        expected = geometry.leading_shape + (system.component_count,)
        if left.shape != expected or right.shape != expected:
            raise ValueError("GRHD face primitive states must match face geometry.")
        covector = jnp.broadcast_to(
            jax.nn.one_hot(axis_, 3, dtype=left.dtype),
            geometry.leading_shape + (3,),
        )
        lower, upper = system.characteristic_bounds_from_primitive(
            left, right, geometry, covector
        )
        left_flux = system.physical_flux_from_primitive(left, geometry, axis_)
        right_flux = system.physical_flux_from_primitive(right, geometry, axis_)
        left_conserved = system.primitive_to_conserved(left, geometry)
        right_conserved = system.primitive_to_conserved(right, geometry)
        if self.kind == "rusanov":
            speed = jnp.maximum(jnp.abs(lower), jnp.abs(upper))
            flux = 0.5 * (left_flux + right_flux) - 0.5 * speed[..., None] * (
                right_conserved - left_conserved
            )
        else:
            negative = jnp.minimum(lower, 0.0)
            positive = jnp.maximum(upper, 0.0)
            denominator = positive - negative
            middle = (
                positive[..., None] * left_flux
                - negative[..., None] * right_flux
                + (negative * positive)[..., None] * (right_conserved - left_conserved)
            ) / jnp.where(denominator == 0.0, 1.0, denominator)[..., None]
            flux = jnp.where(
                (negative >= 0.0)[..., None],
                left_flux,
                jnp.where((positive <= 0.0)[..., None], right_flux, middle),
            )
            speed = jnp.maximum(jnp.abs(negative), jnp.abs(positive))
        equal_state = jnp.all(left == right, axis=-1)
        flux = jnp.where(equal_state[..., None], left_flux, flux)
        left_evaluation = system.primitive_evaluation(left, geometry)
        right_evaluation = system.primitive_evaluation(right, geometry)
        finite = (
            jnp.all(jnp.isfinite(flux), axis=-1)
            & jnp.isfinite(speed)
            & left_evaluation.finite
            & right_evaluation.finite
            & geometry.finite
        )
        physical = (
            finite
            & left_evaluation.physically_valid
            & right_evaluation.physically_valid
            & geometry.physically_valid
        )
        return GRHDFaceFluxResult(
            flux,
            speed,
            finite,
            physical,
            jnp.zeros(geometry.leading_shape, dtype=jnp.bool_),
            self.flux_id,
        )


class GRHDBoundaryTrace(StrictModule):
    interior_primitive: Array
    exterior_primitive: Array
    interior_conserved: Array
    exterior_conserved: Array
    outward_unit_covector: Array
    outward_flux: Array
    maximum_speed: Array
    incoming_characteristic_count: Array
    finite: Array
    physically_valid: Array
    snapshot_token: Array
    axis: int = eqx.field(static=True)
    side: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)
    geometry_lineage_id: str = eqx.field(static=True)

    def compatible_with(self, geometry: ADMGridGeometry, /) -> Array:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be an ADMGridGeometry.")
        static_compatible = (
            self.geometry_lineage_id == geometry.geometry_lineage_id
            and self.interior_primitive.shape[:-1] == geometry.leading_shape
        )
        return jnp.asarray(static_compatible) & (
            self.snapshot_token == geometry.snapshot_token
        )


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


def _boundary_exterior_primitive(
    system: ValenciaGRHDSystem,
    interior: Array,
    geometry: ADMGridGeometry,
    axis: int,
    side: str,
    boundary: GRHDBoundaryCondition,
    c2p: GRHDC2PPolicy,
    /,
) -> tuple[Array, Array]:
    if boundary.kind == "atmosphere":
        exterior = c2p.atmosphere.primitive(geometry.leading_shape, interior.dtype)
    elif boundary.kind == "outflow":
        exterior = interior
    else:
        sign = -1.0 if side == "lower" else 1.0
        coordinate_covector = jnp.broadcast_to(
            sign * jax.nn.one_hot(axis, 3, dtype=interior.dtype),
            geometry.leading_shape + (3,),
        )
        inverse_norm = jnp.sqrt(
            ein.contract(
                "...i,...ij,...j->...",
                coordinate_covector,
                geometry.inverse_spatial_metric,
                coordinate_covector,
            )
        )
        unit_covector = coordinate_covector / inverse_norm[..., None]
        unit_vector = ein.contract(
            "...ij,...j->...i", geometry.inverse_spatial_metric, unit_covector
        )
        velocity = interior[..., 2:5]
        normal_velocity = ein.contract("...i,...i->...", velocity, unit_covector)
        reflected = velocity - 2.0 * normal_velocity[..., None] * unit_vector
        exterior = interior.at[..., 2:5].set(reflected)
    evaluation = system.primitive_evaluation(exterior, geometry)
    return exterior, evaluation.physically_valid


def metric_aware_grhd_boundary_trace(
    system: ValenciaGRHDSystem,
    c2p: GRHDC2PPolicy,
    flux_plan: GRHDFaceFluxPlan,
    interior_primitive: ArrayLike,
    face_geometry: ADMGridGeometry,
    axis: int,
    side: str,
    boundary: GRHDBoundaryCondition,
    /,
) -> GRHDBoundaryTrace:
    """Evaluate one boundary in the face ADM metric and outward normal frame."""

    if side not in ("lower", "upper"):
        raise ValueError("GRHD boundary side must be 'lower' or 'upper'.")
    interior = jnp.asarray(interior_primitive, dtype=face_geometry.alpha.dtype)
    exterior, exterior_valid = _boundary_exterior_primitive(
        system, interior, face_geometry, int(axis), side, boundary, c2p
    )
    result = (
        flux_plan.evaluate(system, exterior, interior, face_geometry, int(axis))
        if side == "lower"
        else flux_plan.evaluate(system, interior, exterior, face_geometry, int(axis))
    )
    sign = -1.0 if side == "lower" else 1.0
    coordinate_covector = jnp.broadcast_to(
        sign * jax.nn.one_hot(int(axis), 3, dtype=interior.dtype),
        face_geometry.leading_shape + (3,),
    )
    inverse_norm = jnp.sqrt(
        ein.contract(
            "...i,...ij,...j->...",
            coordinate_covector,
            face_geometry.inverse_spatial_metric,
            coordinate_covector,
        )
    )
    unit_covector = coordinate_covector / inverse_norm[..., None]
    lower, upper = system.characteristic_bounds_from_primitive(
        interior, exterior, face_geometry, unit_covector
    )
    evaluation = system.primitive_evaluation(interior, face_geometry)
    contact = face_geometry.alpha * ein.contract(
        "...i,...i->...", evaluation.velocity, unit_covector
    ) - ein.contract("...i,...i->...", face_geometry.beta_contravariant, unit_covector)
    incoming = (
        (lower < 0.0).astype(jnp.int32)
        + 3 * (contact < 0.0).astype(jnp.int32)
        + (upper < 0.0).astype(jnp.int32)
    )
    interior_conserved = system.primitive_to_conserved(interior, face_geometry)
    exterior_conserved = system.primitive_to_conserved(exterior, face_geometry)
    finite = result.finite & evaluation.finite
    return GRHDBoundaryTrace(
        interior,
        exterior,
        interior_conserved,
        exterior_conserved,
        unit_covector,
        sign * result.normal_flux,
        result.maximum_speed,
        incoming,
        finite,
        result.physically_valid & exterior_valid,
        face_geometry.snapshot_token,
        int(axis),
        side,
        boundary.boundary_id,
        face_geometry.geometry_lineage_id,
    )


def _face_average(value: Array, axis: int, periodic: bool, /) -> Array:
    if periodic:
        return 0.5 * (jnp.roll(value, 1, axis=axis) + value)
    lower = jnp.take(value, 0, axis=axis)
    upper = jnp.take(value, value.shape[axis] - 1, axis=axis)
    first = jnp.expand_dims(lower, axis=axis)
    last = jnp.expand_dims(upper, axis=axis)
    left = [slice(None)] * value.ndim
    right = [slice(None)] * value.ndim
    left[axis] = slice(0, value.shape[axis] - 1)
    right[axis] = slice(1, value.shape[axis])
    middle = 0.5 * (value[tuple(left)] + value[tuple(right)])
    return jnp.concatenate((first, middle, last), axis=axis)


def _face_boolean(
    active: Array, valid: Array, axis: int, periodic: bool, /
) -> tuple[Array, Array]:
    if periodic:
        left_active = jnp.roll(active, 1, axis=axis)
        left_valid = jnp.roll(valid, 1, axis=axis)
        face_active = left_active | active
        face_valid = face_active & (~left_active | left_valid) & (~active | valid)
        return face_active, face_valid
    lower_active = jnp.take(active, 0, axis=axis)
    upper_active = jnp.take(active, active.shape[axis] - 1, axis=axis)
    lower_valid = jnp.take(valid, 0, axis=axis)
    upper_valid = jnp.take(valid, valid.shape[axis] - 1, axis=axis)
    left_active = [slice(None)] * active.ndim
    right_active = [slice(None)] * active.ndim
    left_active[axis] = slice(0, active.shape[axis] - 1)
    right_active[axis] = slice(1, active.shape[axis])
    a0, a1 = active[tuple(left_active)], active[tuple(right_active)]
    v0, v1 = valid[tuple(left_active)], valid[tuple(right_active)]
    internal_active = a0 | a1
    internal_valid = internal_active & (~a0 | v0) & (~a1 | v1)
    return (
        jnp.concatenate(
            (
                jnp.expand_dims(lower_active, axis),
                internal_active,
                jnp.expand_dims(upper_active, axis),
            ),
            axis=axis,
        ),
        jnp.concatenate(
            (
                jnp.expand_dims(lower_active & lower_valid, axis),
                internal_valid,
                jnp.expand_dims(upper_active & upper_valid, axis),
            ),
            axis=axis,
        ),
    )


def _cell_derivative(
    value: Array, centers: Array, widths: Array, axis: int, periodic: bool, /
) -> Array:
    moved = jnp.moveaxis(value, axis, 0)
    count = moved.shape[0]
    if periodic:
        period = jnp.sum(widths)
        previous = jnp.roll(moved, 1, axis=0)
        following = jnp.roll(moved, -1, axis=0)
        previous_x = jnp.roll(centers, 1).at[0].add(-period)
        following_x = jnp.roll(centers, -1).at[-1].add(period)
        scale = following_x - previous_x
        derivative = (following - previous) / scale.reshape(
            (count,) + (1,) * (moved.ndim - 1)
        )
    else:
        previous = jnp.concatenate((moved[:1], moved[:-1]), axis=0)
        following = jnp.concatenate((moved[1:], moved[-1:]), axis=0)
        previous_x = jnp.concatenate((centers[:1], centers[:-1]))
        following_x = jnp.concatenate((centers[1:], centers[-1:]))
        scale = following_x - previous_x
        scale = scale.at[0].set(centers[1] - centers[0])
        scale = scale.at[-1].set(centers[-1] - centers[-2])
        derivative = (following - previous) / scale.reshape(
            (count,) + (1,) * (moved.ndim - 1)
        )
    return jnp.moveaxis(derivative, 0, axis)


class ValenciaFiniteVolumeStageGeometry(StrictModule, NonTrainableState):
    """Cell/source and coordinate-face ADM snapshots at one SSPRK stage."""

    source: ValenciaGeometrySource
    faces: tuple[ADMGridGeometry, ...]
    time: Array
    stage_geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: ValenciaGeometrySource,
        faces: Sequence[ADMGridGeometry],
        time: ArrayLike,
        /,
    ):
        if not isinstance(source, ValenciaGeometrySource):
            raise TypeError("source must be a ValenciaGeometrySource.")
        faces_ = tuple(faces)
        if not faces_ or any(not isinstance(value, ADMGridGeometry) for value in faces_):
            raise TypeError("faces must contain ADMGridGeometry values.")
        geometry = source.geometry
        identities = (
            geometry.chart_id,
            geometry.convention_id,
            geometry.scale_id,
            geometry.topology_id,
        )
        if any(
            (
                value.chart_id,
                value.convention_id,
                value.scale_id,
                value.topology_id,
            )
            != identities
            for value in faces_
        ):
            raise ValueError("GRHD cell and face ADM identities must agree.")
        time_ = jnp.asarray(time, dtype=geometry.alpha.dtype)
        if time_.shape != ():
            raise ValueError("GRHD stage geometry time must be scalar.")
        time_ = eqx.error_if(
            time_,
            jnp.any(
                jnp.stack(
                    tuple(
                        face.snapshot_token != geometry.snapshot_token for face in faces_
                    )
                )
            ),
            "GRHD cell and face ADM snapshot tokens must agree.",
        )
        self.source = source
        self.faces = faces_
        self.time = time_
        self.stage_geometry_id = canonical_fingerprint(
            {
                "kind": "grhd-stage-geometry",
                "source": source.source_geometry_id,
                "faces": [face.geometry_lineage_id for face in faces_],
                "topology": geometry.topology_id,
            }
        )

    @property
    def cell(self) -> ADMGridGeometry:
        return self.source.geometry

    @property
    def finite(self) -> Array:
        values = (jnp.all(self.source.finite),) + tuple(
            jnp.all(face.finite) for face in self.faces
        )
        return jnp.all(jnp.stack(values)) & jnp.isfinite(self.time)

    @property
    def physically_valid(self) -> Array:
        def valid_geometry(geometry: ADMGridGeometry, /) -> Array:
            scale = jnp.maximum(
                jnp.max(jnp.abs(geometry.spatial_metric), axis=(-2, -1)),
                1.0,
            )
            tolerance = 256.0 * jnp.finfo(geometry.alpha.dtype).eps * scale
            algebraically_valid = (
                (geometry.inverse_defect <= tolerance)
                & (geometry.determinant_defect <= tolerance)
                & (geometry.spatial_symmetry_defect <= tolerance)
            )
            return jnp.all(
                ~geometry.active | (geometry.physically_valid & algebraically_valid)
            )

        values = (valid_geometry(self.cell),) + tuple(
            valid_geometry(face) for face in self.faces
        )
        return jnp.all(jnp.stack(values))


def lower_valencia_stage_geometry(
    discretization: FiniteVolumeDiscretization,
    cell_geometry: ADMGridGeometry,
    time: ArrayLike,
    /,
) -> ValenciaFiniteVolumeStageGeometry:
    """Lower cell ADM data to source derivatives and metric-consistent faces."""

    if not isinstance(discretization, FiniteVolumeDiscretization):
        raise TypeError("discretization must be a FiniteVolumeDiscretization.")
    if not isinstance(cell_geometry, ADMGridGeometry):
        raise TypeError("cell_geometry must be an ADMGridGeometry.")
    if cell_geometry.leading_shape != discretization.cell_shape:
        raise ValueError("Cell ADM geometry must match the finite-volume cell shape.")
    grid_dimension = len(discretization.cell_shape)
    dtype = cell_geometry.alpha.dtype
    alpha_gradient = jnp.zeros(discretization.cell_shape + (3,), dtype=dtype)
    beta_gradient = jnp.zeros(discretization.cell_shape + (3, 3), dtype=dtype)
    metric_gradient = jnp.zeros(discretization.cell_shape + (3, 3, 3), dtype=dtype)
    faces = []
    inverse_plan = SmallLinearSolvePlan(3)
    for axis in range(grid_dimension):
        structured_axis = discretization.grid.structured_axes[axis]
        centers = structured_axis.interval_centers.astype(dtype)
        widths = structured_axis.interval_widths.astype(dtype)
        periodic = structured_axis.periodic
        alpha_gradient = alpha_gradient.at[..., axis].set(
            _cell_derivative(cell_geometry.alpha, centers, widths, axis, periodic)
        )
        beta_gradient = beta_gradient.at[..., axis, :].set(
            _cell_derivative(
                cell_geometry.beta_contravariant, centers, widths, axis, periodic
            )
        )
        metric_gradient = metric_gradient.at[..., axis, :, :].set(
            _cell_derivative(
                cell_geometry.spatial_metric, centers, widths, axis, periodic
            )
        )
        face_metric = _face_average(cell_geometry.spatial_metric, axis, periodic)
        inverse = inverse_small_linear(inverse_plan, face_metric)
        face_active, face_valid = _face_boolean(
            cell_geometry.active, cell_geometry.valid, axis, periodic
        )
        valid = face_valid & inverse.successful & (inverse.determinant > 0.0)
        faces.append(
            ADMGridGeometry(
                _face_average(cell_geometry.alpha, axis, periodic),
                _face_average(cell_geometry.beta_contravariant, axis, periodic),
                face_metric,
                inverse.value,
                jnp.sqrt(jnp.maximum(inverse.determinant, 0.0)),
                _face_average(cell_geometry.extrinsic_curvature, axis, periodic),
                face_active,
                valid,
                snapshot_token=cell_geometry.snapshot_token,
                chart_id=cell_geometry.chart_id,
                convention_id=cell_geometry.convention_id,
                scale_id=cell_geometry.scale_id,
                topology_id=cell_geometry.topology_id,
                geometry_lineage_id=canonical_fingerprint(
                    {
                        "kind": "grhd-coordinate-face-geometry-lineage",
                        "cell_geometry_lineage": cell_geometry.geometry_lineage_id,
                        "discretization": discretization.prepared_id,
                        "axis": axis,
                    }
                ),
            )
        )
    source = ValenciaGeometrySource(
        cell_geometry, alpha_gradient, beta_gradient, metric_gradient
    )
    return ValenciaFiniteVolumeStageGeometry(source, tuple(faces), time)


class GRHDFiniteVolumeEvaluation(StrictModule):
    """One pure spatial stage evaluation reusable by coupled runtimes."""

    residual: Array
    primitive_recovery: GRHDC2PResult
    stress_energy: StressEnergyProjection
    face_fluxes: tuple[Array, ...]
    face_speeds: tuple[Array, ...]
    fallback_masks: tuple[Array, ...]
    boundary_outward_rate: Array
    excision_outward_rate: Array
    source_integral_rate: Array
    maximum_relative_rate: Array
    stable_step: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    successful: Array
    stage_geometry_id: str = eqx.field(static=True)


class GRHDConservationLedger(StrictModule):
    """Accepted-step Valencia balance with repairs kept outside physical sources."""

    initial_content_integral: Array
    final_content_integral: Array
    content_change: Array
    outer_boundary_flux_integral: Array
    excision_flux_integral: Array
    geometric_source_integral: Array
    atmosphere_increment: Array
    c2p_replacement_increment: Array
    content_defect: Array
    baryon_defect: Array
    coordinate_momentum_defect: Array
    coordinate_energy_defect: Array
    geometry_gcl_defect: Array
    stationary_geometry: Array
    eligible_for_conservation_claim: Array
    finite: Array
    successful: Array
    accepted: Array
    initial_snapshot_token: Array
    final_snapshot_token: Array
    geometry_lineage_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)


class GRHDFiniteVolumeState(StrictModule):
    content: FiniteVolumeConservativeContentState
    pressure_seed: Array
    initial_integral: Array
    initial_atmosphere_increment: Array
    cumulative_outer_boundary_flux: Array
    cumulative_excision_flux: Array
    cumulative_geometric_source: Array
    cumulative_atmosphere_increment: Array
    cumulative_c2p_replacement_increment: Array
    accepted_step: Array
    status: Array
    cell_shape: tuple[int, ...] = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    @property
    def time(self) -> Array:
        return self.content.time

    @property
    def conserved(self) -> Array:
        return self.content.cell_average().reshape(
            self.cell_shape + self.content.component_shape
        )


class GRHDFiniteVolumeStepResult(StrictModule):
    candidate: GRHDFiniteVolumeState
    accepted: GRHDFiniteVolumeState
    stage_evaluations: tuple[
        GRHDFiniteVolumeEvaluation,
        GRHDFiniteVolumeEvaluation,
        GRHDFiniteVolumeEvaluation,
    ]
    stage_recoveries: tuple[GRHDC2PResult, GRHDC2PResult, GRHDC2PResult]
    ledger: GRHDConservationLedger
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    successful: Array
    runtime_id: str = eqx.field(static=True)


class FixedGridGRHDSSPRK3Plan(AbstractFixedStepMethod):
    """Atomic fixed-grid SSPRK(3,3) with explicit C2P and repair ledgers."""

    system: ValenciaGRHDSystem
    c2p: GRHDC2PPolicy
    discretization: FiniteVolumeDiscretization
    reconstruction: AbstractFaceReconstructionPlan
    fallback_reconstruction: PiecewiseConstantReconstruction
    face_flux: GRHDFaceFluxPlan
    boundaries: tuple[GRHDBoundaryPair | None, ...]
    precision: FiniteVolumePrecisionPolicy
    cfl: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    maximum_step_atmosphere_mass: float = eqx.field(static=True)
    maximum_step_atmosphere_energy: float = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: ValenciaGRHDSystem,
        c2p: GRHDC2PPolicy,
        discretization: FiniteVolumeDiscretization,
        /,
        *,
        reconstruction: AbstractFaceReconstructionPlan | None = None,
        face_flux: GRHDFaceFluxPlan | None = None,
        boundaries: Sequence[GRHDBoundaryPair | None] | None = None,
        cfl: float = 0.4,
        conservation_tolerance: float = 1.0e-9,
        maximum_step_atmosphere_mass: float = 1.0e30,
        maximum_step_atmosphere_energy: float = 1.0e30,
    ):
        if not isinstance(system, ValenciaGRHDSystem):
            raise TypeError("system must be a ValenciaGRHDSystem.")
        if not isinstance(c2p, GRHDC2PPolicy) or c2p.system.system_id != system.system_id:
            raise ValueError("c2p must target the supplied ValenciaGRHDSystem.")
        if not isinstance(discretization, FiniteVolumeDiscretization):
            raise TypeError("discretization must be a FiniteVolumeDiscretization.")
        if discretization.component_count != system.component_count:
            raise ValueError("GRHD system and finite-volume component counts must agree.")
        if len(discretization.cell_shape) not in (1, 2, 3):
            raise ValueError("Fixed-grid GRHD requires one, two, or three grid axes.")
        reconstruction_ = (
            MUSCLReconstruction(MCLimiter()) if reconstruction is None else reconstruction
        )
        if not isinstance(reconstruction_, AbstractFaceReconstructionPlan):
            raise TypeError("reconstruction must be a finite-volume reconstruction plan.")
        flux_ = GRHDFaceFluxPlan() if face_flux is None else face_flux
        if not isinstance(flux_, GRHDFaceFluxPlan):
            raise TypeError("face_flux must be a GRHDFaceFluxPlan or None.")
        if boundaries is None:
            boundary_values = tuple(
                None
                if axis.periodic
                else GRHDBoundaryPair(
                    GRHDBoundaryCondition("outflow"),
                    GRHDBoundaryCondition("outflow"),
                )
                for axis in discretization.grid.structured_axes
            )
        else:
            boundary_values = tuple(boundaries)
        if len(boundary_values) != len(discretization.cell_shape) or any(
            value is not None and not isinstance(value, GRHDBoundaryPair)
            for value in boundary_values
        ):
            raise ValueError("GRHD boundary pairs must align with grid axes.")
        for axis, boundary in zip(
            discretization.grid.structured_axes, boundary_values, strict=True
        ):
            if axis.periodic != (boundary is None):
                raise ValueError(
                    "Periodic GRHD axes require no boundary pair and bounded axes require one."
                )
        controls = tuple(
            float(value)
            for value in (
                cfl,
                conservation_tolerance,
                maximum_step_atmosphere_mass,
                maximum_step_atmosphere_energy,
            )
        )
        cfl_, tolerance, mass_budget, energy_budget = controls
        if (
            not all(np.isfinite(value) for value in controls)
            or not 0.0 < cfl_ <= 1.0
            or tolerance <= 0.0
            or mass_budget < 0.0
            or energy_budget < 0.0
        ):
            raise ValueError("GRHD finite-volume controls are invalid.")
        self.system = system
        self.c2p = c2p
        self.discretization = discretization
        self.reconstruction = reconstruction_
        self.fallback_reconstruction = PiecewiseConstantReconstruction()
        self.face_flux = flux_
        self.boundaries = boundary_values
        self.precision = FiniteVolumePrecisionPolicy(
            jnp.dtype(discretization.cell_volumes.dtype).name
        )
        self.cfl = cfl_
        self.conservation_tolerance = tolerance
        self.maximum_step_atmosphere_mass = mass_budget
        self.maximum_step_atmosphere_energy = energy_budget
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "fixed-grid-grhd-ssprk33",
                "system": system.system_id,
                "c2p": c2p.policy_id,
                "discretization": discretization.prepared_id,
                "reconstruction": reconstruction_.plan_id,
                "fallback_reconstruction": self.fallback_reconstruction.plan_id,
                "face_flux": flux_.flux_id,
                "boundaries": [
                    None if value is None else value.pair_id for value in boundary_values
                ],
                "cfl": cfl_,
                "conservation_tolerance": tolerance,
                "maximum_step_atmosphere_mass": mass_budget,
                "maximum_step_atmosphere_energy": energy_budget,
                "precision": self.precision.policy_id,
            }
        )
        self.method_id = self.runtime_id

    def _check_geometry(self, stage: ValenciaFiniteVolumeStageGeometry, /) -> None:
        if not isinstance(stage, ValenciaFiniteVolumeStageGeometry):
            raise TypeError("stage geometry must be a ValenciaFiniteVolumeStageGeometry.")
        if stage.cell.leading_shape != self.discretization.cell_shape:
            raise ValueError("GRHD stage cell geometry does not match the grid.")
        if len(stage.faces) != len(self.discretization.cell_shape):
            raise ValueError("GRHD stage requires one face geometry per grid axis.")
        for axis, face in enumerate(stage.faces):
            expected = self.discretization.face_layouts[axis].shape
            if face.leading_shape != expected:
                raise ValueError("GRHD face geometry does not match its FV face layout.")

    def _integral(self, state: Array) -> Array:
        return jnp.sum(
            (state * self.discretization.cell_volumes[..., None]).reshape(
                (-1, self.system.component_count)
            ),
            axis=0,
        )

    def initialize(
        self,
        conserved: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        time: ArrayLike = 0.0,
        /,
    ) -> GRHDFiniteVolumeState:
        self._check_geometry(geometry)
        value = jnp.asarray(conserved, dtype=geometry.cell.alpha.dtype)
        expected = self.discretization.cell_shape + (self.system.component_count,)
        if value.shape != expected:
            raise ValueError(f"Initial GRHD conserved state must have shape {expected}.")
        recovery = self.c2p.recover(value, geometry.cell)
        valid = jnp.all(recovery.successful) & geometry.physically_valid
        corrected = eqx.error_if(
            recovery.conservative_state,
            ~valid,
            "Initial GRHD state, C2P recovery, or ADM geometry is invalid.",
        )
        volumes = jnp.where(
            geometry.cell.active,
            self.discretization.cell_volumes,
            jnp.zeros_like(self.discretization.cell_volumes),
        ).reshape((-1,))
        active = geometry.cell.active.reshape((-1,))
        content = FiniteVolumeConservativeContentState.from_cell_average(
            corrected.reshape((-1, self.system.component_count)),
            volumes,
            active,
            jnp.asarray(time, dtype=value.dtype),
            topology_epoch_id=geometry.cell.topology_id,
            geometry_family_id=geometry.cell.geometry_lineage_id,
            geometry_layout_id=self.discretization.cell_layout.layout_id,
            geometry_version=geometry.cell.snapshot_token,
            evidence_policy_id=self.runtime_id,
            evidence_version=jnp.asarray(0, dtype=jnp.int32),
            precision=self.precision,
        )
        initial_atmosphere = self._integral(recovery.atmosphere.conservative_increment)
        initial = self._integral(corrected)
        zeros = jnp.zeros_like(initial)
        return GRHDFiniteVolumeState(
            content,
            recovery.pressure,
            initial,
            initial_atmosphere,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(GRHDFiniteVolumeRunStatus.SUCCESS), dtype=jnp.int32),
            self.discretization.cell_shape,
            self.runtime_id,
        )

    def _state_check(self, state: GRHDFiniteVolumeState, /) -> None:
        if (
            not isinstance(state, GRHDFiniteVolumeState)
            or state.runtime_id != self.runtime_id
        ):
            raise ValueError("GRHD state belongs to another fixed-grid runtime.")

    def _boundary_exteriors(
        self,
        primitive: Array,
        stage: ValenciaFiniteVolumeStageGeometry,
        axis: int,
        /,
    ) -> tuple[Array | None, Array | None]:
        pair = self.boundaries[axis]
        if pair is None:
            return None, None
        lower_interior = jnp.take(primitive, 0, axis=axis)
        upper_interior = jnp.take(primitive, primitive.shape[axis] - 1, axis=axis)
        lower_geometry = _take_adm_geometry(stage.faces[axis], 0, axis)
        upper_geometry = _take_adm_geometry(
            stage.faces[axis], stage.faces[axis].leading_shape[axis] - 1, axis
        )
        lower, _ = _boundary_exterior_primitive(
            self.system,
            lower_interior,
            lower_geometry,
            axis,
            "lower",
            pair.lower,
            self.c2p,
        )
        upper, _ = _boundary_exterior_primitive(
            self.system,
            upper_interior,
            upper_geometry,
            axis,
            "upper",
            pair.upper,
            self.c2p,
        )
        return lower, upper

    def _active_face_sides(self, active: Array, axis: int, periodic: bool, /):
        if periodic:
            return jnp.roll(active, 1, axis=axis), active
        lower = jnp.zeros_like(jnp.take(active, 0, axis=axis))
        left = jnp.concatenate(
            (
                jnp.expand_dims(lower, axis),
                active,
            ),
            axis=axis,
        )
        right = jnp.concatenate(
            (
                active,
                jnp.expand_dims(lower, axis),
            ),
            axis=axis,
        )
        return left, right

    def evaluate_stage(
        self,
        conserved: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        /,
        *,
        warm_pressure: ArrayLike | None = None,
    ) -> GRHDFiniteVolumeEvaluation:
        """Pure same-stage RHS/projection kernel for standalone or coupled SSPRK."""

        self._check_geometry(geometry)
        value = jnp.asarray(conserved, dtype=geometry.cell.alpha.dtype)
        expected = self.discretization.cell_shape + (self.system.component_count,)
        if value.shape != expected:
            raise ValueError("GRHD stage state shape does not match the runtime grid.")
        recovery = self.c2p.recover(value, geometry.cell, warm_pressure=warm_pressure)
        primitive = recovery.primitive
        atmosphere = self.c2p.atmosphere.primitive(
            self.discretization.cell_shape, value.dtype
        )
        primitive = jnp.where(geometry.cell.active[..., None], primitive, atmosphere)
        fluxes = []
        speeds = []
        fallback_masks = []
        flux_valid = []
        boundary_rate = jnp.zeros((self.system.component_count,), dtype=value.dtype)
        excision_rate = jnp.zeros_like(boundary_rate)
        relative_rate = jnp.zeros(self.discretization.cell_shape, dtype=value.dtype)
        for axis in range(len(self.discretization.cell_shape)):
            structured_axis = self.discretization.grid.structured_axes[axis]
            lower_exterior, upper_exterior = self._boundary_exteriors(
                primitive, geometry, axis
            )
            high_left, high_right = self.reconstruction.reconstruct_axis(
                primitive,
                axis,
                periodic=structured_axis.periodic,
                lower_exterior=lower_exterior,
                upper_exterior=upper_exterior,
                cell_widths=structured_axis.interval_widths,
            )
            low_left, low_right = self.fallback_reconstruction.reconstruct_axis(
                primitive,
                axis,
                periodic=structured_axis.periodic,
                lower_exterior=lower_exterior,
                upper_exterior=upper_exterior,
                cell_widths=structured_axis.interval_widths,
            )
            left_active, right_active = self._active_face_sides(
                geometry.cell.active, axis, structured_axis.periodic
            )
            excision = left_active ^ right_active
            active_low = jnp.where(left_active[..., None], low_left, low_right)
            high_left = jnp.where(excision[..., None], active_low, high_left)
            high_right = jnp.where(excision[..., None], active_low, high_right)
            low_left = jnp.where(excision[..., None], active_low, low_left)
            low_right = jnp.where(excision[..., None], active_low, low_right)
            face_geometry = geometry.faces[axis]
            high_left_evaluation = self.system.primitive_evaluation(
                high_left, face_geometry
            )
            high_right_evaluation = self.system.primitive_evaluation(
                high_right, face_geometry
            )
            low_left_evaluation = self.system.primitive_evaluation(
                low_left, face_geometry
            )
            low_right_evaluation = self.system.primitive_evaluation(
                low_right, face_geometry
            )
            high_valid = (~face_geometry.active) | (
                high_left_evaluation.physically_valid
                & high_right_evaluation.physically_valid
                & face_geometry.physically_valid
            )
            low_valid = (~face_geometry.active) | (
                low_left_evaluation.physically_valid
                & low_right_evaluation.physically_valid
                & face_geometry.physically_valid
            )
            use_fallback = ~high_valid & low_valid
            left = jnp.where(use_fallback[..., None], low_left, high_left)
            right = jnp.where(use_fallback[..., None], low_right, high_right)
            face_result = self.face_flux.evaluate(
                self.system, left, right, face_geometry, axis
            )
            flux = face_result.normal_flux
            speed = face_result.maximum_speed
            fluxes.append(flux)
            speeds.append(speed)
            fallback_masks.append(use_fallback)
            flux_valid.append(
                jnp.all(
                    (~face_geometry.active)
                    | (face_result.finite & (high_valid | low_valid))
                )
            )
            measure = self.discretization.face_measures[axis]
            integrated = flux * measure[..., None]
            if structured_axis.periodic:
                difference = jnp.roll(integrated, -1, axis=axis) - integrated
                cell_speed = jnp.maximum(speed, jnp.roll(speed, -1, axis=axis))
            else:
                lower_slice = [slice(None)] * integrated.ndim
                upper_slice = [slice(None)] * integrated.ndim
                lower_slice[axis] = slice(0, integrated.shape[axis] - 1)
                upper_slice[axis] = slice(1, integrated.shape[axis])
                difference = (
                    integrated[tuple(upper_slice)] - integrated[tuple(lower_slice)]
                )
                speed_lower = [slice(None)] * speed.ndim
                speed_upper = [slice(None)] * speed.ndim
                speed_lower[axis] = slice(0, speed.shape[axis] - 1)
                speed_upper[axis] = slice(1, speed.shape[axis])
                cell_speed = jnp.maximum(
                    speed[tuple(speed_lower)], speed[tuple(speed_upper)]
                )
                lower_flux = jnp.take(integrated, 0, axis=axis)
                upper_flux = jnp.take(integrated, integrated.shape[axis] - 1, axis=axis)
                sum_axes = tuple(range(lower_flux.ndim - 1))
                boundary_rate = boundary_rate + jnp.sum(
                    -lower_flux + upper_flux, axis=sum_axes
                )
            excision_sign = jnp.where(left_active & ~right_active, 1.0, -1.0)
            excision_integrated = jnp.where(
                excision[..., None],
                excision_sign[..., None] * integrated,
                0.0,
            )
            excision_rate = excision_rate + jnp.sum(
                excision_integrated.reshape((-1, self.system.component_count)), axis=0
            )
            widths = structured_axis.interval_widths.astype(value.dtype)
            width_shape = [1] * len(self.discretization.cell_shape)
            width_shape[axis] = widths.size
            relative_rate = relative_rate + cell_speed / widths.reshape(width_shape)
            if axis == 0:
                flux_residual = -difference / self.discretization.cell_volumes[..., None]
            else:
                flux_residual = (
                    flux_residual
                    - difference / self.discretization.cell_volumes[..., None]
                )
        source = self.system.source_from_primitive(primitive, geometry.source)
        source = jnp.where(geometry.cell.active[..., None], source, 0.0)
        residual = jnp.where(geometry.cell.active[..., None], flux_residual + source, 0.0)
        source_integral = jnp.sum(
            (source * self.discretization.cell_volumes[..., None]).reshape(
                (-1, self.system.component_count)
            ),
            axis=0,
        )
        maximum = jnp.max(
            jnp.where(geometry.cell.active, relative_rate, 0.0), initial=0.0
        )
        stable_step = self.cfl / jnp.maximum(maximum, jnp.finfo(value.dtype).tiny)
        stress = self.system.stress_energy_projection(primitive, geometry.cell)
        finite = (
            jnp.all(jnp.isfinite(residual))
            & jnp.all(jnp.isfinite(source_integral))
            & jnp.all(recovery.finite)
            & geometry.finite
        )
        converged = jnp.all(recovery.converged)
        physical = (
            jnp.all(recovery.physically_valid)
            & jnp.all(jnp.stack(tuple(flux_valid)))
            & geometry.physically_valid
            & stress.all_active_valid
        )
        qualified = jnp.all(recovery.qualified) & physical
        derivative_valid = jnp.all(recovery.derivative_valid) & ~jnp.any(
            jnp.stack(tuple(jnp.any(mask) for mask in fallback_masks))
        )
        successful = finite & converged & physical
        return GRHDFiniteVolumeEvaluation(
            residual,
            recovery,
            stress,
            tuple(fluxes),
            tuple(speeds),
            tuple(fallback_masks),
            boundary_rate,
            excision_rate,
            source_integral,
            maximum,
            stable_step,
            finite,
            converged,
            physical,
            qualified,
            derivative_valid,
            successful,
            geometry.stage_geometry_id,
        )

    def recover_stage(
        self,
        candidate: ArrayLike,
        geometry: ValenciaFiniteVolumeStageGeometry,
        /,
        *,
        warm_pressure: ArrayLike | None = None,
    ) -> GRHDC2PResult:
        """Pure stage-candidate recovery/repair kernel for coupled SSPRK owners."""

        self._check_geometry(geometry)
        return self.c2p.recover(
            jnp.asarray(candidate, dtype=geometry.cell.alpha.dtype),
            geometry.cell,
            warm_pressure=warm_pressure,
        )

    def _repair_integral(self, repair: Array, /) -> Array:
        return jnp.sum(
            (repair * self.discretization.cell_volumes[..., None]).reshape(
                (-1, self.system.component_count)
            ),
            axis=0,
        )

    def _new_state(
        self,
        previous: GRHDFiniteVolumeState,
        conserved: Array,
        pressure: Array,
        time: Array,
        status: Array,
        boundary: Array,
        excision: Array,
        source: Array,
        atmosphere: Array,
        replacement: Array,
        geometry_version: Array,
        accepted_step: Array,
        /,
    ) -> GRHDFiniteVolumeState:
        content_values = (
            conserved.reshape((-1, self.system.component_count))
            * previous.content.effective_cell_volumes[:, None]
        )
        content = previous.content.with_content(
            content_values,
            time=time,
            evidence_version=previous.content.evidence_version + accepted_step,
        )
        content = content.with_topology_epoch(
            content.topology_epoch_id,
            geometry_version=geometry_version,
            evidence_version=content.evidence_version,
        )
        return GRHDFiniteVolumeState(
            content,
            pressure,
            previous.initial_integral,
            previous.initial_atmosphere_increment,
            previous.cumulative_outer_boundary_flux + boundary,
            previous.cumulative_excision_flux + excision,
            previous.cumulative_geometric_source + source,
            previous.cumulative_atmosphere_increment + atmosphere,
            previous.cumulative_c2p_replacement_increment + replacement,
            previous.accepted_step + accepted_step,
            status,
            previous.cell_shape,
            previous.runtime_id,
        )

    def advance(
        self,
        state: GRHDFiniteVolumeState,
        start_time: ArrayLike,
        end_time: ArrayLike,
        stage_geometries: tuple[
            ValenciaFiniteVolumeStageGeometry,
            ValenciaFiniteVolumeStageGeometry,
            ValenciaFiniteVolumeStageGeometry,
        ],
        /,
    ) -> GRHDFiniteVolumeStepResult:
        self._state_check(state)
        if not isinstance(stage_geometries, tuple) or len(stage_geometries) != 3:
            raise TypeError("GRHD SSPRK requires exactly three stage geometries.")
        for geometry in stage_geometries:
            self._check_geometry(geometry)
        start = jnp.asarray(start_time, dtype=state.time.dtype)
        end = jnp.asarray(end_time, dtype=state.time.dtype)
        step = end - start
        if start.shape != () or end.shape != ():
            raise ValueError("GRHD SSPRK interval endpoints must be scalar.")
        tolerance = (
            32.0
            * jnp.finfo(start.dtype).eps
            * jnp.maximum(jnp.maximum(jnp.abs(start), jnp.abs(end)), 1.0)
        )
        start = eqx.error_if(
            start,
            ~jnp.isfinite(start)
            | ~jnp.isfinite(end)
            | (step <= 0.0)
            | (jnp.abs(state.time - start) > tolerance),
            "GRHD SSPRK interval is invalid or state time is stale.",
        )
        stage_0, stage_1, stage_2 = stage_geometries
        expected_times = (start, end, start + 0.5 * step)
        time_consistent = jnp.all(
            jnp.stack(
                tuple(
                    jnp.abs(geometry.time - expected) <= tolerance
                    for geometry, expected in zip(
                        stage_geometries, expected_times, strict=True
                    )
                )
            )
        ) & (state.content.geometry_version == stage_0.cell.snapshot_token)
        static_consistent = all(
            geometry.cell.topology_id == stage_0.cell.topology_id
            and geometry.cell.chart_id == stage_0.cell.chart_id
            and geometry.cell.convention_id == stage_0.cell.convention_id
            and geometry.cell.scale_id == stage_0.cell.scale_id
            and geometry.cell.geometry_lineage_id == stage_0.cell.geometry_lineage_id
            and all(
                face.geometry_lineage_id == stage_0.faces[axis].geometry_lineage_id
                for axis, face in enumerate(geometry.faces)
            )
            for geometry in stage_geometries
        )
        if not static_consistent:
            raise ValueError("GRHD stage geometry static identities must agree.")
        if (
            state.content.geometry_family_id != stage_0.cell.geometry_lineage_id
            or state.content.topology_epoch_id != stage_0.cell.topology_id
        ):
            raise ValueError(
                "GRHD state geometry lineage/topology does not match the first stage."
            )
        u0 = state.conserved
        evaluation_0 = self.evaluate_stage(u0, stage_0, warm_pressure=state.pressure_seed)
        raw_1 = u0 + step * evaluation_0.residual
        recovery_1 = self.recover_stage(
            raw_1, stage_1, warm_pressure=evaluation_0.primitive_recovery.pressure
        )
        u1 = recovery_1.conservative_state
        evaluation_1 = self.evaluate_stage(u1, stage_1, warm_pressure=recovery_1.pressure)
        raw_2 = 0.75 * u0 + 0.25 * (u1 + step * evaluation_1.residual)
        recovery_2 = self.recover_stage(raw_2, stage_2, warm_pressure=recovery_1.pressure)
        u2 = recovery_2.conservative_state
        evaluation_2 = self.evaluate_stage(u2, stage_2, warm_pressure=recovery_2.pressure)
        raw_3 = (1.0 / 3.0) * u0 + (2.0 / 3.0) * (u2 + step * evaluation_2.residual)
        recovery_3 = self.recover_stage(raw_3, stage_1, warm_pressure=recovery_2.pressure)
        u3 = recovery_3.conservative_state
        evaluations = (evaluation_0, evaluation_1, evaluation_2)
        recoveries = (recovery_1, recovery_2, recovery_3)
        weights = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)
        boundary = step * sum(
            weight * evaluation.boundary_outward_rate
            for weight, evaluation in zip(weights, evaluations, strict=True)
        )
        excision = step * sum(
            weight * evaluation.excision_outward_rate
            for weight, evaluation in zip(weights, evaluations, strict=True)
        )
        source = step * sum(
            weight * evaluation.source_integral_rate
            for weight, evaluation in zip(weights, evaluations, strict=True)
        )
        repair = (
            (1.0 / 6.0) * recovery_1.atmosphere.conservative_increment
            + (2.0 / 3.0) * recovery_2.atmosphere.conservative_increment
            + recovery_3.atmosphere.conservative_increment
        )
        near_vacuum = (
            (recovery_1.atmosphere.status == int(AtmosphereFloorStatus.NEAR_VACUUM))[
                ..., None
            ]
            * ((1.0 / 6.0) * recovery_1.atmosphere.conservative_increment)
            + (recovery_2.atmosphere.status == int(AtmosphereFloorStatus.NEAR_VACUUM))[
                ..., None
            ]
            * ((2.0 / 3.0) * recovery_2.atmosphere.conservative_increment)
            + (recovery_3.atmosphere.status == int(AtmosphereFloorStatus.NEAR_VACUUM))[
                ..., None
            ]
            * recovery_3.atmosphere.conservative_increment
        )
        atmosphere_increment = self._repair_integral(near_vacuum)
        replacement_increment = self._repair_integral(repair - near_vacuum)
        repair_integral = atmosphere_increment + replacement_increment
        initial_integral = self._integral(u0)
        final_integral = self._integral(u3)
        content_change = final_integral - initial_integral
        content_defect = content_change + boundary + excision - source - repair_integral
        all_geometries = tuple(
            geometry
            for stage in stage_geometries
            for geometry in (stage.cell, *stage.faces)
        )
        geometry_gcl_defect = jnp.max(
            jnp.stack(
                tuple(
                    jnp.maximum(
                        jnp.max(geometry.inverse_defect),
                        jnp.maximum(
                            jnp.max(geometry.determinant_defect),
                            jnp.max(geometry.spatial_symmetry_defect),
                        ),
                    )
                    for geometry in all_geometries
                )
            )
        )

        def same_snapshot(
            first: ValenciaFiniteVolumeStageGeometry,
            second: ValenciaFiniteVolumeStageGeometry,
            /,
        ):
            first_cell, second_cell = first.cell, second.cell
            return (
                jnp.all(first_cell.alpha == second_cell.alpha)
                & jnp.all(first_cell.beta_contravariant == second_cell.beta_contravariant)
                & jnp.all(first_cell.spatial_metric == second_cell.spatial_metric)
                & jnp.all(
                    first_cell.extrinsic_curvature == second_cell.extrinsic_curvature
                )
                & jnp.all(first.source.alpha_gradient == second.source.alpha_gradient)
                & jnp.all(first.source.beta_gradient == second.source.beta_gradient)
                & jnp.all(
                    first.source.spatial_metric_gradient
                    == second.source.spatial_metric_gradient
                )
            )

        stationary_geometry = same_snapshot(stage_0, stage_1) & same_snapshot(
            stage_0, stage_2
        )
        atmosphere_mass = jnp.maximum(atmosphere_increment[0], 0.0)
        atmosphere_energy = jnp.maximum(atmosphere_increment[-1], 0.0)
        atmosphere_budget = (atmosphere_mass <= self.maximum_step_atmosphere_mass) & (
            atmosphere_energy <= self.maximum_step_atmosphere_energy
        )
        stable_step = jnp.min(
            jnp.stack(tuple(evaluation.stable_step for evaluation in evaluations))
        )
        finite = (
            jnp.all(jnp.stack(tuple(evaluation.finite for evaluation in evaluations)))
            & jnp.all(jnp.stack(tuple(recovery.finite for recovery in recoveries)))
            & jnp.all(jnp.isfinite(u3))
            & jnp.all(jnp.isfinite(content_defect))
        )
        converged = jnp.all(
            jnp.stack(tuple(evaluation.converged for evaluation in evaluations))
        ) & jnp.all(
            jnp.stack(tuple(jnp.all(recovery.converged) for recovery in recoveries))
        )
        physical = jnp.all(
            jnp.stack(tuple(evaluation.physically_valid for evaluation in evaluations))
        ) & jnp.all(
            jnp.stack(
                tuple(jnp.all(recovery.physically_valid) for recovery in recoveries)
            )
        )
        geometry_valid = time_consistent & jnp.all(
            jnp.stack(tuple(geometry.physically_valid for geometry in stage_geometries))
        )
        stable = step <= stable_step + tolerance
        defect_scale = jnp.maximum(
            jnp.maximum(jnp.abs(content_change), jnp.abs(source)), 1.0
        )
        conservation_valid = jnp.all(
            jnp.abs(content_defect) <= self.conservation_tolerance * defect_scale
        )
        successful = (
            finite
            & converged
            & physical
            & geometry_valid
            & stable
            & atmosphere_budget
            & conservation_valid
        )
        status = jnp.where(
            successful,
            int(GRHDFiniteVolumeRunStatus.SUCCESS),
            jnp.where(
                ~finite,
                int(GRHDFiniteVolumeRunStatus.NONFINITE_STATE),
                jnp.where(
                    ~geometry_valid,
                    int(GRHDFiniteVolumeRunStatus.INVALID_GEOMETRY),
                    jnp.where(
                        ~converged | ~physical,
                        int(GRHDFiniteVolumeRunStatus.C2P_FAILED),
                        jnp.where(
                            ~stable,
                            int(GRHDFiniteVolumeRunStatus.STABILITY_LIMIT_EXCEEDED),
                            jnp.where(
                                ~atmosphere_budget,
                                int(GRHDFiniteVolumeRunStatus.ATMOSPHERE_BUDGET_EXCEEDED),
                                int(GRHDFiniteVolumeRunStatus.CONSERVATION_DEFECT),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        no_repairs = jnp.all(repair_integral == 0.0)
        eligible = successful & stationary_geometry & no_repairs
        ledger = GRHDConservationLedger(
            initial_integral,
            final_integral,
            content_change,
            boundary,
            excision,
            source,
            atmosphere_increment,
            replacement_increment,
            content_defect,
            content_defect[0],
            content_defect[1:4],
            content_defect[-1],
            geometry_gcl_defect,
            stationary_geometry,
            eligible,
            finite,
            conservation_valid,
            successful,
            stage_0.cell.snapshot_token,
            stage_1.cell.snapshot_token,
            stage_0.cell.geometry_lineage_id,
            self.system.system_id,
            self.runtime_id,
        )
        candidate_values = u3
        candidate = self._new_state(
            state,
            candidate_values,
            recovery_3.pressure,
            end,
            status,
            boundary,
            excision,
            source,
            atmosphere_increment,
            replacement_increment,
            stage_1.cell.snapshot_token,
            jnp.asarray(1, dtype=jnp.int32),
        )
        accepted = self._new_state(
            state,
            jnp.where(successful, candidate_values, u0),
            jnp.where(successful, recovery_3.pressure, state.pressure_seed),
            jnp.where(successful, end, state.time),
            status,
            jnp.where(successful, boundary, 0.0),
            jnp.where(successful, excision, 0.0),
            jnp.where(successful, source, 0.0),
            jnp.where(successful, atmosphere_increment, 0.0),
            jnp.where(successful, replacement_increment, 0.0),
            jnp.where(
                successful,
                stage_1.cell.snapshot_token,
                state.content.geometry_version,
            ),
            successful.astype(jnp.int32),
        )
        qualified = (
            successful
            & conservation_valid
            & jnp.all(
                jnp.stack(tuple(evaluation.qualified for evaluation in evaluations))
            )
            & jnp.all(
                jnp.stack(tuple(jnp.all(recovery.qualified) for recovery in recoveries))
            )
        )
        derivative_valid = (
            qualified
            & no_repairs
            & jnp.all(
                jnp.stack(
                    tuple(evaluation.derivative_valid for evaluation in evaluations)
                )
            )
            & jnp.all(
                jnp.stack(
                    tuple(jnp.all(recovery.derivative_valid) for recovery in recoveries)
                )
            )
        )
        return GRHDFiniteVolumeStepResult(
            candidate,
            accepted,
            evaluations,
            recoveries,
            ledger,
            status,
            finite,
            converged,
            physical,
            qualified,
            derivative_valid,
            successful,
            self.runtime_id,
        )

    def step(self, step_index, time, state, step_size, args, /) -> FixedStepResult:
        if (
            not isinstance(args, tuple)
            or len(args) != 3
            or any(
                not isinstance(value, ValenciaFiniteVolumeStageGeometry) for value in args
            )
        ):
            raise TypeError(
                "Fixed-step GRHD args must be the three synchronized stage geometries."
            )
        self._state_check(state)
        index = eqx.error_if(
            jnp.asarray(step_index),
            jnp.asarray(step_index) != state.accepted_step,
            "GRHD fixed-step index does not match the accepted state.",
        )
        del index
        result = self.advance(
            state,
            time,
            jnp.asarray(time) + jnp.asarray(step_size),
            args,
        )
        return FixedStepResult(
            result.candidate,
            result.accepted,
            result.successful,
            jnp.max(jnp.abs(result.ledger.content_defect)),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(3, dtype=jnp.int32),
            jnp.any(
                jnp.stack(
                    tuple(
                        jnp.any(mask)
                        for evaluation in result.stage_evaluations
                        for mask in evaluation.fallback_masks
                    )
                )
            ),
            jnp.max(
                jnp.abs(
                    result.ledger.atmosphere_increment
                    + result.ledger.c2p_replacement_increment
                )
            ),
        )


__all__ = [
    "FixedGridGRHDSSPRK3Plan",
    "GRHDBoundaryCondition",
    "GRHDBoundaryPair",
    "GRHDBoundaryTrace",
    "GRHDConservationLedger",
    "GRHDFaceFluxPlan",
    "GRHDFaceFluxResult",
    "GRHDFiniteVolumeEvaluation",
    "GRHDFiniteVolumeRunStatus",
    "GRHDFiniteVolumeState",
    "GRHDFiniteVolumeStepResult",
    "ValenciaFiniteVolumeStageGeometry",
    "lower_valencia_stage_geometry",
    "metric_aware_grhd_boundary_trace",
]
