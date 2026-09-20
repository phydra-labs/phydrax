#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, pi

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._physical import RelativityScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    determinant_small_linear,
    inverse_small_linear,
    SmallLinearSolvePlan,
)
from ...metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ...metrix._spacetime_conventions import RelativityConvention
from ._derivatives import FourthOrderDerivatives
from ._gauge import AbstractZ4cGauge
from ._grid import FixedGridGeometry
from ._state import make_z4c_state, Z4cState


_SMALL_3X3 = SmallLinearSolvePlan(3)
_Z4C_SNAPSHOT_SLOTS_PER_STEP = 8
_MAX_Z4C_SNAPSHOT_STEP = (
    jnp.iinfo(jnp.int32).max - (_Z4C_SNAPSHOT_SLOTS_PER_STEP - 1)
) // _Z4C_SNAPSHOT_SLOTS_PER_STEP


def z4c_snapshot_token(step_index: ArrayLike, stage_slot: ArrayLike, /) -> Array:
    """Encode one bounded step/stage address as an exact dynamic int32 token."""

    step = jnp.asarray(step_index)
    stage = jnp.asarray(stage_slot)
    if (
        step.shape != ()
        or stage.shape != ()
        or not jnp.issubdtype(step.dtype, jnp.integer)
        or not jnp.issubdtype(stage.dtype, jnp.integer)
    ):
        raise TypeError("step_index and stage_slot must be scalar integers.")
    invalid = (
        (step < 0)
        | (step > _MAX_Z4C_SNAPSHOT_STEP)
        | (stage < 0)
        | (stage >= _Z4C_SNAPSHOT_SLOTS_PER_STEP)
    )
    step32 = step.astype(jnp.int32)
    stage32 = stage.astype(jnp.int32)
    token = step32 * _Z4C_SNAPSHOT_SLOTS_PER_STEP + stage32
    return eqx.error_if(
        token,
        invalid,
        "Z4c snapshot address is outside the collision-free int32 range.",
    )


def _trailing_matrix(value: Array, /) -> Array:
    return jnp.moveaxis(value, (0, 1), (-2, -1))


def _component_matrix(value: Array, /) -> Array:
    return jnp.moveaxis(value, (-2, -1), (0, 1))


def _component_vector(value: Array, /) -> Array:
    return jnp.moveaxis(value, -1, 0)


def _trailing_vector(value: Array, /) -> Array:
    return jnp.moveaxis(value, 0, -1)


def _connection(inverse_metric: Array, metric_gradient: Array, /) -> Array:
    shape = metric_gradient.shape[-3:]
    connection = jnp.zeros((3, 3, 3) + shape, dtype=metric_gradient.dtype)
    for upper in range(3):
        for first in range(3):
            for second in range(3):
                value = jnp.zeros(shape, dtype=metric_gradient.dtype)
                for contracted in range(3):
                    value = value + 0.5 * inverse_metric[upper, contracted] * (
                        metric_gradient[first, contracted, second]
                        + metric_gradient[second, contracted, first]
                        - metric_gradient[contracted, first, second]
                    )
                connection = connection.at[upper, first, second].set(value)
    return connection


def _ricci_tensor(
    connection: Array,
    derivatives: FourthOrderDerivatives,
    /,
    *,
    riemann_sign: int,
) -> Array:
    derivative = derivatives.gradient(connection)
    shape = connection.shape[-3:]
    ricci = jnp.zeros((3, 3) + shape, dtype=connection.dtype)
    trace = jnp.stack(
        tuple(
            sum(connection[upper, first, upper] for upper in range(3))
            for first in range(3)
        ),
        axis=0,
    )
    for first in range(3):
        for second in range(3):
            differential = sum(
                derivative[upper, upper, first, second]
                - derivative[second, upper, first, upper]
                for upper in range(3)
            )
            quadratic = sum(
                connection[upper, first, second] * trace[upper] for upper in range(3)
            ) - sum(
                connection[upper, first, contracted]
                * connection[contracted, second, upper]
                for upper in range(3)
                for contracted in range(3)
            )
            ricci = ricci.at[first, second].set(riemann_sign * (differential + quadratic))
    return 0.5 * (ricci + jnp.swapaxes(ricci, 0, 1))


def _covariant_scalar_hessian(
    scalar: Array,
    connection: Array,
    derivatives: FourthOrderDerivatives,
    /,
) -> Array:
    gradient = derivatives.gradient(scalar)
    hessian = derivatives.hessian(scalar)
    return hessian - ein.contract(
        "kij...,k...->ij...", connection, gradient, backend="jax"
    )


def _divergence_trace_reversed_extrinsic(
    metric_inverse: Array,
    connection: Array,
    extrinsic: Array,
    trace: Array,
    derivatives: FourthOrderDerivatives,
    /,
) -> Array:
    raised = ein.contract(
        "ik...,jl...,kl...->ij...",
        metric_inverse,
        metric_inverse,
        extrinsic,
        backend="jax",
    )
    trace_reversed = raised - metric_inverse * trace[None, None, ...]
    partial = derivatives.gradient(trace_reversed)
    shape = trace.shape
    divergence = jnp.zeros((3,) + shape, dtype=trace.dtype)
    connection_trace = jnp.stack(
        tuple(
            sum(connection[upper, first, upper] for upper in range(3))
            for first in range(3)
        ),
        axis=0,
    )
    for upper in range(3):
        value = sum(partial[second, upper, second] for second in range(3))
        value = value + sum(
            connection[upper, second, contracted] * trace_reversed[contracted, second]
            for second in range(3)
            for contracted in range(3)
        )
        value = value + sum(
            connection_trace[contracted] * trace_reversed[upper, contracted]
            for contracted in range(3)
        )
        divergence = divergence.at[upper].set(value)
    return divergence


class Z4cConstraintEvidence(StrictModule):
    hamiltonian: Array
    momentum: Array
    conformal_connection: Array
    determinant: Array
    trace_free: Array
    theta: Array
    l2_norm: Array
    maximum_norm: Array
    finite: Array
    within_tolerance: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array


class Z4cRHSEvaluation(StrictModule):
    rates: Z4cState
    geometry: ADMGridGeometry
    constraints: Z4cConstraintEvidence
    finite: Array
    physically_valid: Array
    source_valid: Array
    derivative_valid: Array


class Z4cSystem(StrictModule, NonTrainableState):
    """Canonical mostly-plus conformal Z4c equations with explicit damping."""

    scale: RelativityScaleContract = eqx.field(static=True)
    convention: RelativityConvention = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    constraint_damping: float = eqx.field(static=True)
    damping_coupling: float = eqx.field(static=True)
    einstein_coupling: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        /,
        *,
        chart_id: str,
        constraint_damping: float = 0.02,
        damping_coupling: float = 0.0,
        einstein_coupling: float = 8.0 * pi,
        constraint_tolerance: float = 1.0e-6,
    ):
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be a RelativityScaleContract.")
        if scale.gravitational_constant != 1 or scale.speed_of_light != 1:
            raise ValueError(
                "Z4cSystem requires explicitly declared geometric G=c=1 units."
            )
        if not isinstance(convention, RelativityConvention):
            raise TypeError("convention must be a RelativityConvention.")
        if (
            convention.metric_signature != "mostly_plus"
            or convention.riemann_sign != 1
            or convention.extrinsic_curvature_sign != -1
        ):
            raise ValueError(
                "This Z4c system requires the canonical mostly-plus curvature signs."
            )
        chart = str(chart_id)
        values = (
            float(constraint_damping),
            float(damping_coupling),
            float(einstein_coupling),
            float(constraint_tolerance),
        )
        if not chart:
            raise ValueError("chart_id must be non-empty.")
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Z4c coefficients must be finite and non-negative.")
        self.scale = scale
        self.convention = convention
        self.chart_id = chart
        self.constraint_damping = values[0]
        self.damping_coupling = values[1]
        self.einstein_coupling = values[2]
        self.constraint_tolerance = values[3]
        self.system_id = canonical_fingerprint(
            {
                "kind": "canonical-vacuum-matter-z4c",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "chart": chart,
                "constraint_damping": values[0],
                "damping_coupling": values[1],
                "einstein_coupling": values[2],
                "constraint_tolerance": values[3],
            }
        )

    def geometry_lineage_id(self, grid: FixedGridGeometry, /) -> str:
        if not isinstance(grid, FixedGridGeometry):
            raise TypeError("grid must be a FixedGridGeometry.")
        return canonical_fingerprint(
            {
                "kind": "z4c-adm-grid-geometry-lineage",
                "system": self.system_id,
                "topology": grid.grid_id,
            }
        )


def z4c_adm_geometry(
    system: Z4cSystem,
    grid: FixedGridGeometry,
    state: Z4cState,
    /,
    *,
    snapshot_token: ArrayLike,
) -> ADMGridGeometry:
    """Expose one Z4c state through the shared immutable ADM exchange contract."""

    if not isinstance(system, Z4cSystem):
        raise TypeError("system must be a Z4cSystem.")
    if not isinstance(grid, FixedGridGeometry):
        raise TypeError("grid must be a FixedGridGeometry.")
    if not isinstance(state, Z4cState) or state.grid_id != grid.grid_id:
        raise ValueError("state must belong to the supplied fixed grid.")
    physical_metric = state.physical_metric
    inverse_result = inverse_small_linear(_SMALL_3X3, _trailing_matrix(physical_metric))
    determinant = determinant_small_linear(_SMALL_3X3, _trailing_matrix(physical_metric))
    inverse = inverse_result.value
    spatial_metric = _trailing_matrix(physical_metric)
    eigenvalues = jnp.linalg.eigvalsh(spatial_metric)
    finite = jnp.all(jnp.isfinite(state.values), axis=0)
    valid = (
        finite
        & (state.chi > 0.0)
        & (state.lapse > 0.0)
        & inverse_result.successful
        & (determinant > 0.0)
        & (jnp.min(eigenvalues, axis=-1) > 0.0)
    )
    return ADMGridGeometry(
        state.lapse,
        _trailing_vector(state.shift),
        spatial_metric,
        inverse,
        jnp.sqrt(determinant),
        _trailing_matrix(state.physical_extrinsic_curvature),
        jnp.ones(grid.shape, dtype=jnp.bool_),
        valid,
        snapshot_token=snapshot_token,
        chart_id=system.chart_id,
        convention_id=system.convention.convention_id,
        scale_id=system.scale.scale_id,
        topology_id=grid.grid_id,
        geometry_lineage_id=system.geometry_lineage_id(grid),
    )


def _matter_fields(
    geometry: ADMGridGeometry,
    stress_energy: StressEnergyProjection | None,
    /,
) -> tuple[Array, Array, Array, Array]:
    shape = geometry.leading_shape
    dtype = geometry.alpha.dtype
    if stress_energy is None:
        return (
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros((3,) + shape, dtype=dtype),
            jnp.zeros((3, 3) + shape, dtype=dtype),
            jnp.asarray(True),
        )
    if not isinstance(stress_energy, StressEnergyProjection):
        raise TypeError("stress_energy must be a StressEnergyProjection or None.")
    active = stress_energy.active
    energy = jnp.where(
        active, jnp.asarray(stress_energy.energy_density, dtype=dtype), 0.0
    )
    momentum = jnp.where(
        active[None, ...],
        _component_vector(jnp.asarray(stress_energy.momentum_covector, dtype=dtype)),
        0.0,
    )
    stress = jnp.where(
        active[None, None, ...],
        _component_matrix(jnp.asarray(stress_energy.stress_covariant, dtype=dtype)),
        0.0,
    )
    source_valid = stress_energy.compatible_with(geometry) & jnp.all(
        ~active | stress_energy.physically_valid
    )
    return energy, momentum, stress, source_valid


def _constraint_evidence(
    system: Z4cSystem,
    state: Z4cState,
    geometry: ADMGridGeometry,
    derivatives: FourthOrderDerivatives,
    physical_connection: Array,
    physical_ricci: Array,
    conformal_inverse: Array,
    conformal_connection_constraint: Array,
    energy: Array,
    momentum_covector: Array,
    source_valid: Array,
    derivative_valid: Array,
    /,
) -> Z4cConstraintEvidence:
    physical_inverse = _component_matrix(geometry.inverse_spatial_metric)
    physical_extrinsic = state.physical_extrinsic_curvature
    trace = state.trace_extrinsic_curvature
    raised_extrinsic = ein.contract(
        "ik...,jl...,kl...->ij...",
        physical_inverse,
        physical_inverse,
        physical_extrinsic,
        backend="jax",
    )
    extrinsic_square = ein.contract(
        "ij...,ij...->...", physical_extrinsic, raised_extrinsic, backend="jax"
    )
    scalar_curvature = ein.contract(
        "ij...,ij...->...", physical_inverse, physical_ricci, backend="jax"
    )
    hamiltonian = (
        scalar_curvature
        + trace**2
        - extrinsic_square
        - 2.0 * system.einstein_coupling * energy
    )
    momentum = _divergence_trace_reversed_extrinsic(
        physical_inverse,
        physical_connection,
        physical_extrinsic,
        trace,
        derivatives,
    ) - system.einstein_coupling * ein.contract(
        "ij...,j...->i...", physical_inverse, momentum_covector, backend="jax"
    )
    conformal_determinant = determinant_small_linear(
        _SMALL_3X3, _trailing_matrix(state.conformal_metric)
    )
    determinant_constraint = conformal_determinant - 1.0
    trace_free_constraint = ein.contract(
        "ij...,ij...->...",
        conformal_inverse,
        state.conformal_extrinsic_curvature,
        backend="jax",
    )
    count = (
        hamiltonian.size
        + momentum.size
        + conformal_connection_constraint.size
        + determinant_constraint.size
        + trace_free_constraint.size
        + state.theta.size
    )
    squared = (
        jnp.sum(hamiltonian**2)
        + jnp.sum(momentum**2)
        + jnp.sum(conformal_connection_constraint**2)
        + jnp.sum(determinant_constraint**2)
        + jnp.sum(trace_free_constraint**2)
        + jnp.sum(state.theta**2)
    )
    maximum = jnp.max(
        jnp.stack(
            (
                jnp.max(jnp.abs(hamiltonian)),
                jnp.max(jnp.abs(momentum)),
                jnp.max(jnp.abs(conformal_connection_constraint)),
                jnp.max(jnp.abs(determinant_constraint)),
                jnp.max(jnp.abs(trace_free_constraint)),
                jnp.max(jnp.abs(state.theta)),
            )
        )
    )
    l2 = jnp.sqrt(squared / count)
    finite = (
        jnp.isfinite(l2)
        & jnp.isfinite(maximum)
        & jnp.all(jnp.isfinite(hamiltonian))
        & jnp.all(jnp.isfinite(momentum))
    )
    physically_valid = geometry.all_active_valid & source_valid
    within_tolerance = finite & (maximum <= system.constraint_tolerance)
    qualified = physically_valid & within_tolerance & derivative_valid
    return Z4cConstraintEvidence(
        hamiltonian,
        momentum,
        conformal_connection_constraint,
        determinant_constraint,
        trace_free_constraint,
        state.theta,
        l2,
        maximum,
        finite,
        within_tolerance,
        physically_valid,
        qualified,
        derivative_valid,
    )


def evaluate_z4c_rhs(
    system: Z4cSystem,
    grid: FixedGridGeometry,
    derivatives: FourthOrderDerivatives,
    gauge: AbstractZ4cGauge,
    state: Z4cState,
    /,
    *,
    snapshot_token: ArrayLike,
    stress_energy: StressEnergyProjection | None = None,
) -> Z4cRHSEvaluation:
    """Evaluate the conformal Z4c method-of-lines RHS and constraint evidence."""

    if not isinstance(system, Z4cSystem):
        raise TypeError("system must be a Z4cSystem.")
    if not isinstance(grid, FixedGridGeometry):
        raise TypeError("grid must be a FixedGridGeometry.")
    if not isinstance(derivatives, FourthOrderDerivatives):
        raise TypeError("derivatives must be FourthOrderDerivatives.")
    if not isinstance(gauge, AbstractZ4cGauge):
        raise TypeError("gauge must implement AbstractZ4cGauge.")
    if not isinstance(state, Z4cState) or state.grid_id != grid.grid_id:
        raise ValueError("state must belong to the supplied fixed grid.")
    if derivatives.grid_shape != grid.shape:
        raise ValueError("derivative and grid shapes do not match.")

    geometry = z4c_adm_geometry(system, grid, state, snapshot_token=snapshot_token)
    energy, momentum_covector, stress_covariant, source_valid = _matter_fields(
        geometry, stress_energy
    )
    conformal_metric = state.conformal_metric
    conformal_inverse_result = inverse_small_linear(
        _SMALL_3X3, _trailing_matrix(conformal_metric)
    )
    conformal_inverse = _component_matrix(conformal_inverse_result.value)
    physical_metric = state.physical_metric
    physical_inverse = _component_matrix(geometry.inverse_spatial_metric)
    physical_metric_gradient = derivatives.gradient(physical_metric)
    physical_connection = _connection(physical_inverse, physical_metric_gradient)
    physical_ricci = _ricci_tensor(
        physical_connection,
        derivatives,
        riemann_sign=system.convention.riemann_sign,
    )
    conformal_metric_gradient = derivatives.gradient(conformal_metric)
    conformal_christoffel = _connection(conformal_inverse, conformal_metric_gradient)
    contracted_christoffel = ein.contract(
        "jk...,ijk...->i...",
        conformal_inverse,
        conformal_christoffel,
        backend="jax",
    )
    connection_constraint = state.conformal_connection - contracted_christoffel
    z_covector = 0.5 * ein.contract(
        "ij...,j...->i...",
        conformal_metric,
        connection_constraint,
        backend="jax",
    )
    z_gradient = derivatives.gradient(z_covector)
    covariant_z = z_gradient - ein.contract(
        "kij...,k...->ij...", physical_connection, z_covector, backend="jax"
    )
    z4c_ricci = physical_ricci + covariant_z + jnp.swapaxes(covariant_z, 0, 1)

    conformal_extrinsic = state.conformal_extrinsic_curvature
    raised_conformal_extrinsic = ein.contract(
        "ik...,jl...,kl...->ij...",
        conformal_inverse,
        conformal_inverse,
        conformal_extrinsic,
        backend="jax",
    )
    conformal_extrinsic_square = ein.contract(
        "ij...,ij...->...",
        conformal_extrinsic,
        raised_conformal_extrinsic,
        backend="jax",
    )
    mixed_extrinsic_square = ein.contract(
        "ik...,kl...,lj...->ij...",
        conformal_extrinsic,
        conformal_inverse,
        conformal_extrinsic,
        backend="jax",
    )
    trace = state.trace_extrinsic_curvature
    lapse_gradient = derivatives.gradient(state.lapse)
    lapse_hessian = _covariant_scalar_hessian(
        state.lapse, physical_connection, derivatives
    )
    lapse_laplacian = ein.contract(
        "ij...,ij...->...", physical_inverse, lapse_hessian, backend="jax"
    )
    shift_gradient = derivatives.gradient(state.shift)
    shift_divergence = sum(shift_gradient[index, index] for index in range(3))
    shift_hessian = derivatives.hessian(state.shift)
    stress_trace = ein.contract(
        "ij...,ij...->...", physical_inverse, stress_covariant, backend="jax"
    )

    chi_rate = derivatives.advect(state.chi, state.shift) + (
        (2.0 / 3.0) * state.chi * (state.lapse * trace - shift_divergence)
    )
    metric_shift = (
        ein.contract(
            "ik...,jk...->ij...",
            conformal_metric,
            shift_gradient,
            backend="jax",
        )
        + ein.contract(
            "jk...,ik...->ij...",
            conformal_metric,
            shift_gradient,
            backend="jax",
        )
        - (2.0 / 3.0) * conformal_metric * shift_divergence[None, None, ...]
    )
    conformal_metric_rate = (
        derivatives.advect(conformal_metric, state.shift)
        - 2.0 * state.lapse[None, None, ...] * conformal_extrinsic
        + metric_shift
    )
    k_hat_rate = (
        derivatives.advect(state.k_hat, state.shift)
        - lapse_laplacian
        + state.lapse
        * (
            conformal_extrinsic_square
            + trace**2 / 3.0
            + system.constraint_damping * (1.0 - system.damping_coupling) * state.theta
        )
        + 0.5 * system.einstein_coupling * state.lapse * (energy + stress_trace)
    )
    curvature_driver = -lapse_hessian + state.lapse[None, None, ...] * (
        z4c_ricci - system.einstein_coupling * stress_covariant
    )
    curvature_trace = ein.contract(
        "ij...,ij...->...", physical_inverse, curvature_driver, backend="jax"
    )
    curvature_driver_tf = state.chi[None, None, ...] * (
        curvature_driver - physical_metric * curvature_trace[None, None, ...] / 3.0
    )
    extrinsic_shift = (
        ein.contract(
            "ik...,jk...->ij...",
            conformal_extrinsic,
            shift_gradient,
            backend="jax",
        )
        + ein.contract(
            "jk...,ik...->ij...",
            conformal_extrinsic,
            shift_gradient,
            backend="jax",
        )
        - (2.0 / 3.0) * conformal_extrinsic * shift_divergence[None, None, ...]
    )
    conformal_extrinsic_rate = (
        derivatives.advect(conformal_extrinsic, state.shift)
        + curvature_driver_tf
        + state.lapse[None, None, ...]
        * (trace[None, None, ...] * conformal_extrinsic - 2.0 * mixed_extrinsic_square)
        + extrinsic_shift
    )
    z_contravariant = ein.contract(
        "ij...,j...->i...", physical_inverse, z_covector, backend="jax"
    )
    covariant_z_divergence = ein.contract(
        "ij...,ij...->...", physical_inverse, covariant_z, backend="jax"
    )
    scalar_curvature = ein.contract(
        "ij...,ij...->...", physical_inverse, physical_ricci, backend="jax"
    )
    theta_rate = (
        derivatives.advect(state.theta, state.shift)
        + 0.5
        * state.lapse
        * (
            scalar_curvature
            + 2.0 * covariant_z_divergence
            - conformal_extrinsic_square
            + (2.0 / 3.0) * trace**2
            - 2.0 * system.einstein_coupling * energy
        )
        - state.lapse
        * system.constraint_damping
        * (2.0 + system.damping_coupling)
        * state.theta
        - ein.contract("i...,i...->...", z_contravariant, lapse_gradient, backend="jax")
    )

    chi_gradient = derivatives.gradient(state.chi)
    modified_trace_gradient = derivatives.gradient(2.0 * state.k_hat + state.theta)
    connection_geometric = -2.0 * ein.contract(
        "ij...,j...->i...",
        raised_conformal_extrinsic,
        lapse_gradient,
        backend="jax",
    ) + 2.0 * state.lapse[None, ...] * (
        ein.contract(
            "ijk...,jk...->i...",
            conformal_christoffel,
            raised_conformal_extrinsic,
            backend="jax",
        )
        - (1.5 / state.chi)[None, ...]
        * ein.contract(
            "ij...,j...->i...",
            raised_conformal_extrinsic,
            chi_gradient,
            backend="jax",
        )
        - (1.0 / 3.0)
        * ein.contract(
            "ij...,j...->i...",
            conformal_inverse,
            modified_trace_gradient,
            backend="jax",
        )
        - system.einstein_coupling
        * ein.contract(
            "ij...,j...->i...",
            conformal_inverse,
            momentum_covector,
            backend="jax",
        )
    )
    beta_laplacian = ein.contract(
        "jk...,jki...->i...",
        conformal_inverse,
        shift_hessian,
        backend="jax",
    )
    divergence_gradient = derivatives.gradient(shift_divergence)
    connection_shift = (
        beta_laplacian
        + (1.0 / 3.0)
        * ein.contract(
            "ij...,j...->i...",
            conformal_inverse,
            divergence_gradient,
            backend="jax",
        )
        + (2.0 / 3.0) * state.conformal_connection * shift_divergence[None, ...]
        - ein.contract(
            "j...,ji...->i...",
            state.conformal_connection,
            shift_gradient,
            backend="jax",
        )
        + derivatives.advect(state.conformal_connection, state.shift)
    )
    conformal_connection_rate = (
        connection_geometric
        + connection_shift
        - system.constraint_damping * state.lapse[None, ...] * connection_constraint
    )
    gauge_rates = gauge.rates(state, derivatives, conformal_connection_rate)
    rates = make_z4c_state(
        chi_rate,
        conformal_metric_rate,
        k_hat_rate,
        conformal_extrinsic_rate,
        theta_rate,
        conformal_connection_rate,
        gauge_rates.lapse,
        gauge_rates.shift,
        gauge_rates.shift_driver,
        grid_id=grid.grid_id,
    )
    rates = rates.with_values(rates.values + derivatives.dissipation(state.values))
    derivative_valid = (
        jnp.all(conformal_inverse_result.successful)
        & jnp.all(jnp.isfinite(physical_ricci))
        & jnp.all(jnp.isfinite(lapse_hessian))
        & jnp.all(jnp.isfinite(conformal_christoffel))
    )
    constraints = _constraint_evidence(
        system,
        state,
        geometry,
        derivatives,
        physical_connection,
        physical_ricci,
        conformal_inverse,
        connection_constraint,
        energy,
        momentum_covector,
        source_valid,
        derivative_valid,
    )
    finite = (
        jnp.all(jnp.isfinite(rates.values))
        & constraints.finite
        & jnp.all(geometry.finite)
    )
    physically_valid = geometry.all_active_valid & source_valid
    return Z4cRHSEvaluation(
        rates,
        geometry,
        constraints,
        finite,
        physically_valid,
        source_valid,
        derivative_valid,
    )


def vacuum_z4c_rhs(
    system: Z4cSystem,
    grid: FixedGridGeometry,
    derivatives: FourthOrderDerivatives,
    gauge: AbstractZ4cGauge,
    state: Z4cState,
    /,
    *,
    snapshot_token: ArrayLike,
) -> Z4cRHSEvaluation:
    """Evaluate the source-free Z4c RHS without manufacturing a matter object."""

    return evaluate_z4c_rhs(
        system,
        grid,
        derivatives,
        gauge,
        state,
        snapshot_token=snapshot_token,
    )


__all__ = [
    "Z4cConstraintEvidence",
    "Z4cRHSEvaluation",
    "Z4cSystem",
    "evaluate_z4c_rhs",
    "vacuum_z4c_rhs",
    "z4c_adm_geometry",
    "z4c_snapshot_token",
]
