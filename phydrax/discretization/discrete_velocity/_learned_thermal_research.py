#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Local ingredients for a matched learned-thermal collision experiment.

This module adapts the two-population thermal construction described by NeurDE
canonical to PHYDRAX's weight-absorbed, native-energy convention: ``sum(g) = E``.
It contains no spatial transport, shock ownership, TVD regularization, or
full-discretization theorem. A later experimental collide-stream owner must
consume the results and statuses explicitly.
"""

from __future__ import annotations

from enum import IntEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein
from phydrax.linalg import SmallLinearSolvePlan, solve_small_linear

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._quadrature import CertifiedDiscreteVelocityQuadrature


class LearnedThermalResearchStatus(IntEnum):
    """Outcome of one local learned-thermal research operation."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    OUTSIDE_DECLARED_SUPPORT = 2
    NONPOSITIVE_DENSITY = 3
    NONPOSITIVE_PRESSURE = 4
    NONPOSITIVE_TOTAL_ENERGY = 5
    MOMENT_SOLVE_FAILED = 6
    NONPOSITIVE_POPULATIONS = 7
    INVALID_RELAXATION_PARAMETERS = 8
    INCONSISTENT_ENERGY_MOMENT = 9
    NONFINITE_OUTPUT = 10
    INADMISSIBLE_FRAME_SHIFT = 11


def _require_standard_d2q9(
    quadrature: CertifiedDiscreteVelocityQuadrature, /, *, owner: str
) -> None:
    expected = {
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    }
    velocities = np.asarray(quadrature.velocities)
    rounded = np.rint(velocities)
    actual = {tuple(row) for row in rounded}
    tolerance = quadrature.certification.tolerance
    if (
        quadrature.dimension != 2
        or quadrature.population_count != 9
        or quadrature.transport_kind != "integer_lattice"
        or float(np.max(np.abs(velocities - rounded))) > tolerance
        or actual != expected
    ):
        raise ValueError(
            f"{owner} requires the tensor-product D2Q9 integer velocity support."
        )


class ExtendedParticleEquilibriumEvidence(StrictModule):
    """Exact mass, momentum, and pressure-tensor evidence for ``f_eq``."""

    target_density: Array
    recovered_density: Array
    density_residual: Array
    target_momentum: Array
    recovered_momentum: Array
    momentum_residual: Array
    target_particle_stress: Array
    recovered_particle_stress: Array
    particle_stress_residual: Array
    raw_moment_residual: Array
    invariant_correction: Array
    stress_correction: Array
    maximum_absolute_moment_residual: Array
    extended_support_margin: Array
    minimum_population: Array
    finite: Array
    positive: Array
    status: Array
    quadrature_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(LearnedThermalResearchStatus.SUCCESS)


class ExtendedParticleEquilibriumResult(StrictModule):
    """Pressure-extended particle equilibrium and its exact-moment evidence."""

    populations: Array
    evidence: ExtendedParticleEquilibriumEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class PressureExtendedParticleEquilibriumPlan(StrictModule, NonTrainableState):
    """Prepare the pressure-dependent extended D2Q9 particle equilibrium.

    This is the weight-absorbed factorized equilibrium from NeurDE canonical Eq. (97):
    in each direction ``Psi_0 = 1 - (T + u²)`` and
    ``Psi_± = (T + u² ± u) / 2``, with ``T = p/rho``. Two native
    three-by-three solves remove floating-point residuals while imposing
    ``sum(f)=rho``, ``sum(c f)=rho*u``, and
    ``sum(c c f)=rho*u*u + p I``. The second correction is constructed in the
    nullspace of the first three moments, so all six identities hold together.
    The plan refuses non-D2Q9 support and states outside the strictly positive
    directional-factor envelope.
    """

    quadrature: CertifiedDiscreteVelocityQuadrature
    invariant_moment_matrix: Array
    invariant_gram: Array
    stress_moment_matrix: Array
    invariant_free_stress_matrix: Array
    stress_gram: Array
    linear_solve: SmallLinearSolvePlan
    plan_id: str = eqx.field(static=True)

    def __init__(self, quadrature: CertifiedDiscreteVelocityQuadrature, /):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        _require_standard_d2q9(quadrature, owner="The pressure-extended research plan")
        if quadrature.certification.maximum_degree < 4:
            raise ValueError(
                "The pressure-extended research plan requires fourth-degree certification."
            )

        velocities = quadrature.velocities
        cx = velocities[:, 0]
        cy = velocities[:, 1]
        invariant = jnp.stack((jnp.ones_like(cx), cx, cy), axis=0)
        stress = jnp.stack((cx * cx, cx * cy, cy * cy), axis=0)
        invariant_gram = ein.contract("mq,nq->mn", invariant, invariant)
        linear_solve = SmallLinearSolvePlan(
            3,
            singular_tolerance=1.0e-14,
            maximum_condition=1.0e14,
            refinement_iterations=2,
        )
        stress_projection = solve_small_linear(
            linear_solve,
            invariant_gram,
            ein.contract("mq,nq->mn", invariant, stress),
        )
        if not bool(np.asarray(stress_projection.successful)):
            raise ValueError("Particle invariant moments are singular on this support.")
        invariant_free_stress = stress - ein.contract(
            "mn,mq->nq", stress_projection.value, invariant
        )
        stress_gram = ein.contract(
            "mq,nq->mn", invariant_free_stress, invariant_free_stress
        )
        stress_check = solve_small_linear(
            linear_solve, stress_gram, jnp.ones((3,), dtype=velocities.dtype)
        )
        if not bool(np.asarray(stress_check.successful)):
            raise ValueError("Particle stress moments are singular on this support.")

        self.quadrature = quadrature
        self.invariant_moment_matrix = invariant
        self.invariant_gram = invariant_gram
        self.stress_moment_matrix = stress
        self.invariant_free_stress_matrix = invariant_free_stress
        self.stress_gram = stress_gram
        self.linear_solve = linear_solve
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pressure-extended-particle-equilibrium-research",
                "quadrature": quadrature.quadrature_id,
                "invariant_moments": array_tree_fingerprint(np.asarray(invariant)),
                "stress_moments": array_tree_fingerprint(np.asarray(stress)),
                "linear_solve": linear_solve.plan_id,
            }
        )

    def evaluate(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> ExtendedParticleEquilibriumResult:
        """Evaluate the pressure extension and certify its six raw moments."""

        rho = jnp.asarray(density)
        flow = jnp.asarray(velocity)
        p = jnp.asarray(pressure)
        for name, value in (("density", rho), ("velocity", flow), ("pressure", p)):
            if not jnp.issubdtype(value.dtype, jnp.number) or jnp.issubdtype(
                value.dtype, jnp.complexfloating
            ):
                raise TypeError(f"{name} must be real numeric data.")
        if flow.ndim == 0 or flow.shape[-1] != 2:
            raise ValueError("velocity must have trailing shape (2,).")
        if flow.shape[:-1] != rho.shape or p.shape != rho.shape:
            raise ValueError("density, velocity, and pressure batch axes must agree.")

        dtype = self.quadrature.velocities.dtype
        rho = jnp.asarray(rho, dtype=dtype)
        flow = jnp.asarray(flow, dtype=dtype)
        p = jnp.asarray(p, dtype=dtype)
        finite = (
            jnp.isfinite(rho) & jnp.isfinite(p) & jnp.all(jnp.isfinite(flow), axis=-1)
        )
        positive_density = rho > 0.0
        positive_pressure = p > 0.0
        physical = finite & positive_density & positive_pressure
        safe_rho = jnp.where(physical, rho, 1.0)
        safe_p = jnp.where(physical, p, self.quadrature.reference_temperature)
        safe_flow = jnp.where(finite[..., None], flow, 0.0)

        theta = safe_p / safe_rho
        directional_second_moment = theta[..., None] + safe_flow * safe_flow
        zero_factor = 1.0 - directional_second_moment
        positive_factor = 0.5 * (directional_second_moment + safe_flow)
        negative_factor = 0.5 * (directional_second_moment - safe_flow)
        lattice_velocities = self.quadrature.velocities
        directional_factors = jnp.where(
            lattice_velocities == 0.0,
            zero_factor[..., None, :],
            jnp.where(
                lattice_velocities > 0.0,
                positive_factor[..., None, :],
                negative_factor[..., None, :],
            ),
        )
        extended_support_margin = jnp.min(directional_factors, axis=(-2, -1))
        declared_support = extended_support_margin > 0.0
        eligible = physical & declared_support
        raw = safe_rho[..., None] * jnp.prod(directional_factors, axis=-1)

        target_momentum = safe_rho[..., None] * safe_flow
        target_invariants = jnp.concatenate(
            (safe_rho[..., None], target_momentum), axis=-1
        )
        target_stress = ein.contract(
            "...,...d,...e->...de", safe_rho, safe_flow, safe_flow
        ) + safe_p[..., None, None] * jnp.eye(2, dtype=dtype)
        target_stress_vector = jnp.stack(
            (
                target_stress[..., 0, 0],
                target_stress[..., 0, 1],
                target_stress[..., 1, 1],
            ),
            axis=-1,
        )
        raw_invariants = ein.contract("mq,...q->...m", self.invariant_moment_matrix, raw)
        raw_stress = ein.contract("mq,...q->...m", self.stress_moment_matrix, raw)
        raw_moment_residual = jnp.concatenate(
            (raw_invariants - target_invariants, raw_stress - target_stress_vector),
            axis=-1,
        )

        batch_shape = rho.shape
        invariant_gram = jnp.broadcast_to(self.invariant_gram, (*batch_shape, 3, 3))
        invariant_solve = solve_small_linear(
            self.linear_solve,
            invariant_gram,
            target_invariants - raw_invariants,
        )
        invariant_correction = ein.contract(
            "qm,...m->...q",
            jnp.swapaxes(self.invariant_moment_matrix, 0, 1),
            invariant_solve.value,
        )
        invariant_corrected = raw + invariant_correction
        recovered_intermediate_stress = ein.contract(
            "mq,...q->...m", self.stress_moment_matrix, invariant_corrected
        )
        stress_gram = jnp.broadcast_to(self.stress_gram, (*batch_shape, 3, 3))
        stress_solve = solve_small_linear(
            self.linear_solve,
            stress_gram,
            target_stress_vector - recovered_intermediate_stress,
        )
        stress_correction = ein.contract(
            "qm,...m->...q",
            jnp.swapaxes(self.invariant_free_stress_matrix, 0, 1),
            stress_solve.value,
        )
        corrected = invariant_corrected + stress_correction
        solves_succeeded = invariant_solve.successful & stress_solve.successful
        populations = jnp.where((eligible & solves_succeeded)[..., None], corrected, 0.0)

        recovered_invariants = ein.contract(
            "mq,...q->...m", self.invariant_moment_matrix, populations
        )
        recovered_stress_vector = ein.contract(
            "mq,...q->...m", self.stress_moment_matrix, populations
        )
        recovered_stress = jnp.stack(
            (
                recovered_stress_vector[..., 0],
                recovered_stress_vector[..., 1],
                recovered_stress_vector[..., 1],
                recovered_stress_vector[..., 2],
            ),
            axis=-1,
        ).reshape((*batch_shape, 2, 2))
        recovered_density = recovered_invariants[..., 0]
        recovered_momentum = recovered_invariants[..., 1:]
        density_residual = recovered_density - rho
        momentum_residual = recovered_momentum - rho[..., None] * flow
        stress_residual = recovered_stress - (
            ein.contract("...,...d,...e->...de", rho, flow, flow)
            + p[..., None, None] * jnp.eye(2, dtype=dtype)
        )
        all_residuals = jnp.concatenate(
            (
                density_residual[..., None],
                momentum_residual,
                stress_residual[..., 0, 0, None],
                stress_residual[..., 0, 1, None],
                stress_residual[..., 1, 1, None],
            ),
            axis=-1,
        )
        output_finite = jnp.all(jnp.isfinite(populations), axis=-1)
        positive = jnp.all(populations > 0.0, axis=-1)

        status = jnp.full(rho.shape, int(LearnedThermalResearchStatus.SUCCESS), jnp.int32)
        status = jnp.where(
            ~finite, int(LearnedThermalResearchStatus.NONFINITE_INPUT), status
        )
        status = jnp.where(
            finite & ~positive_density,
            int(LearnedThermalResearchStatus.NONPOSITIVE_DENSITY),
            status,
        )
        status = jnp.where(
            finite & positive_density & ~positive_pressure,
            int(LearnedThermalResearchStatus.NONPOSITIVE_PRESSURE),
            status,
        )
        status = jnp.where(
            physical & ~declared_support,
            int(LearnedThermalResearchStatus.OUTSIDE_DECLARED_SUPPORT),
            status,
        )
        status = jnp.where(
            eligible & ~solves_succeeded,
            int(LearnedThermalResearchStatus.MOMENT_SOLVE_FAILED),
            status,
        )
        status = jnp.where(
            eligible & solves_succeeded & ~output_finite,
            int(LearnedThermalResearchStatus.NONFINITE_OUTPUT),
            status,
        )
        status = jnp.where(
            eligible & solves_succeeded & output_finite & ~positive,
            int(LearnedThermalResearchStatus.NONPOSITIVE_POPULATIONS),
            status,
        ).astype(jnp.int32)

        evidence = ExtendedParticleEquilibriumEvidence(
            target_density=rho,
            recovered_density=recovered_density,
            density_residual=density_residual,
            target_momentum=rho[..., None] * flow,
            recovered_momentum=recovered_momentum,
            momentum_residual=momentum_residual,
            target_particle_stress=(
                ein.contract("...,...d,...e->...de", rho, flow, flow)
                + p[..., None, None] * jnp.eye(2, dtype=dtype)
            ),
            recovered_particle_stress=recovered_stress,
            particle_stress_residual=stress_residual,
            raw_moment_residual=raw_moment_residual,
            invariant_correction=invariant_correction,
            stress_correction=stress_correction,
            maximum_absolute_moment_residual=jnp.max(jnp.abs(all_residuals), axis=-1),
            extended_support_margin=extended_support_margin,
            minimum_population=jnp.min(populations, axis=-1),
            finite=finite & output_finite,
            positive=positive,
            status=status,
            quadrature_id=self.quadrature.quadrature_id,
            plan_id=self.plan_id,
        )
        return ExtendedParticleEquilibriumResult(populations, evidence, self.plan_id)


class LearnedThermalEnergyEvidence(StrictModule):
    """Normalization and learned sufficient-moment evidence for ``g_eq``."""

    target_total_energy: Array
    recovered_total_energy: Array
    total_energy_residual: Array
    recovered_sufficient_moments: Array
    log_partition: Array
    constant_statistic_residual: Array
    logit_span: Array
    minimum_population: Array
    finite: Array
    positive: Array
    within_declared_support: Array
    status: Array
    quadrature_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(LearnedThermalResearchStatus.SUCCESS)


class LearnedThermalEnergyResult(StrictModule):
    """Positive native-energy populations and the supplied learned coordinates."""

    populations: Array
    sufficient_statistics: Array
    natural_parameters: Array
    evidence: LearnedThermalEnergyEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class PositiveLearnedThermalEnergyPlan(StrictModule, NonTrainableState):
    """Evaluate a supplied exponential family with exact native energy.

    The caller supplies both velocity-wise sufficient-statistic values and
    natural parameters. Consequently the values may be fixed analytic features
    or differentiable outputs of a learned trunk. The first statistic must be
    the fixed constant one. For an eligible state,

    ``g_i = E w_i exp(lambda · phi_i) / sum_j w_j exp(lambda · phi_j)``.

    Thus ``sum_i g_i = E`` in PHYDRAX's native convention. This evaluates a
    learned family; it does not claim that the supplied parameters solve a
    maximum-entropy moment inversion.
    """

    quadrature: CertifiedDiscreteVelocityQuadrature
    statistic_count: int = eqx.field(static=True)
    minimum_total_energy: float = eqx.field(static=True)
    maximum_total_energy: float = eqx.field(static=True)
    maximum_absolute_statistic: float = eqx.field(static=True)
    maximum_absolute_natural_parameter: float = eqx.field(static=True)
    maximum_logit_span: float = eqx.field(static=True)
    constant_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        statistic_count: int,
        /,
        *,
        minimum_total_energy: float = 1.0e-12,
        maximum_total_energy: float = 1.0e12,
        maximum_absolute_statistic: float = 1.0e3,
        maximum_absolute_natural_parameter: float = 1.0e3,
        maximum_logit_span: float = 80.0,
        constant_tolerance: float = 1.0e-12,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        count = int(statistic_count)
        if count <= 0 or count > quadrature.population_count:
            raise ValueError("statistic_count must lie in [1, Q].")
        bounds = (
            float(minimum_total_energy),
            float(maximum_total_energy),
            float(maximum_absolute_statistic),
            float(maximum_absolute_natural_parameter),
            float(maximum_logit_span),
            float(constant_tolerance),
        )
        if any(not isfinite(value) or value <= 0.0 for value in bounds):
            raise ValueError(
                "Learned thermal support bounds must be finite and positive."
            )
        if bounds[1] <= bounds[0]:
            raise ValueError("maximum_total_energy must exceed minimum_total_energy.")

        self.quadrature = quadrature
        self.statistic_count = count
        self.minimum_total_energy = bounds[0]
        self.maximum_total_energy = bounds[1]
        self.maximum_absolute_statistic = bounds[2]
        self.maximum_absolute_natural_parameter = bounds[3]
        self.maximum_logit_span = bounds[4]
        self.constant_tolerance = bounds[5]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "positive-learned-thermal-energy-research",
                "quadrature": quadrature.quadrature_id,
                "statistic_count": count,
                "minimum_total_energy": bounds[0],
                "maximum_total_energy": bounds[1],
                "maximum_absolute_statistic": bounds[2],
                "maximum_absolute_natural_parameter": bounds[3],
                "maximum_logit_span": bounds[4],
                "constant_tolerance": bounds[5],
                "energy_convention": "weight-absorbed-native-total-energy-sum",
            }
        )

    def evaluate(
        self,
        total_energy: ArrayLike,
        sufficient_statistics: ArrayLike,
        natural_parameters: ArrayLike,
        /,
    ) -> LearnedThermalEnergyResult:
        """Evaluate fixed or learned supplied statistics and natural parameters."""

        energy = jnp.asarray(total_energy)
        statistics = jnp.asarray(sufficient_statistics)
        parameters = jnp.asarray(natural_parameters)
        for name, value in (
            ("total_energy", energy),
            ("sufficient_statistics", statistics),
            ("natural_parameters", parameters),
        ):
            if not jnp.issubdtype(value.dtype, jnp.number) or jnp.issubdtype(
                value.dtype, jnp.complexfloating
            ):
                raise TypeError(f"{name} must be real numeric data.")

        batch_shape = energy.shape
        fixed_statistics_shape = (
            self.quadrature.population_count,
            self.statistic_count,
        )
        batched_statistics_shape = (*batch_shape, *fixed_statistics_shape)
        if statistics.shape == fixed_statistics_shape:
            statistics = jnp.broadcast_to(statistics, batched_statistics_shape)
        elif statistics.shape != batched_statistics_shape:
            raise ValueError(
                "sufficient_statistics must have shape (Q, K) or batch + (Q, K)."
            )
        fixed_parameter_shape = (self.statistic_count,)
        batched_parameter_shape = (*batch_shape, self.statistic_count)
        if parameters.shape == fixed_parameter_shape:
            parameters = jnp.broadcast_to(parameters, batched_parameter_shape)
        elif parameters.shape != batched_parameter_shape:
            raise ValueError("natural_parameters must have shape (K,) or batch + (K,).")

        dtype = self.quadrature.velocities.dtype
        energy = jnp.asarray(energy, dtype=dtype)
        statistics = jnp.asarray(statistics, dtype=dtype)
        parameters = jnp.asarray(parameters, dtype=dtype)
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(statistics), axis=(-2, -1))
            & jnp.all(jnp.isfinite(parameters), axis=-1)
        )
        safe_statistics = jnp.where(jnp.isfinite(statistics), statistics, 0.0)
        safe_parameters = jnp.where(jnp.isfinite(parameters), parameters, 0.0)
        logits = ein.contract("...qk,...k->...q", safe_statistics, safe_parameters)
        maximum_logit = jnp.max(logits, axis=-1)
        minimum_logit = jnp.min(logits, axis=-1)
        logit_span = maximum_logit - minimum_logit
        constant_residual = jnp.max(jnp.abs(safe_statistics[..., 0] - 1.0), axis=-1)
        within_support = (
            (energy >= self.minimum_total_energy)
            & (energy <= self.maximum_total_energy)
            & (
                jnp.max(jnp.abs(safe_statistics), axis=(-2, -1))
                <= self.maximum_absolute_statistic
            )
            & (
                jnp.max(jnp.abs(safe_parameters), axis=-1)
                <= self.maximum_absolute_natural_parameter
            )
            & (logit_span <= self.maximum_logit_span)
            & (constant_residual <= self.constant_tolerance)
        )
        positive_energy = energy > 0.0
        eligible = finite & positive_energy & within_support
        shifted_logits = logits - maximum_logit[..., None]
        weighted = self.quadrature.weights * jnp.exp(shifted_logits)
        partition = jnp.sum(weighted, axis=-1)
        distribution = weighted / partition[..., None]
        raw_populations = energy[..., None] * distribution
        output_finite = (
            jnp.all(jnp.isfinite(raw_populations), axis=-1)
            & jnp.isfinite(partition)
            & (partition > 0.0)
        )
        positive = jnp.all(raw_populations > 0.0, axis=-1)
        accepted = eligible & output_finite & positive
        populations = jnp.where(accepted[..., None], raw_populations, 0.0)
        recovered_energy = jnp.sum(populations, axis=-1)
        recovered_statistics = ein.contract("...q,...qk->...k", populations, statistics)
        log_partition = maximum_logit + jnp.log(partition)

        status = jnp.full(
            batch_shape, int(LearnedThermalResearchStatus.SUCCESS), jnp.int32
        )
        status = jnp.where(
            ~finite, int(LearnedThermalResearchStatus.NONFINITE_INPUT), status
        )
        status = jnp.where(
            finite & ~positive_energy,
            int(LearnedThermalResearchStatus.NONPOSITIVE_TOTAL_ENERGY),
            status,
        )
        status = jnp.where(
            finite & positive_energy & ~within_support,
            int(LearnedThermalResearchStatus.OUTSIDE_DECLARED_SUPPORT),
            status,
        )
        status = jnp.where(
            eligible & ~output_finite,
            int(LearnedThermalResearchStatus.NONFINITE_OUTPUT),
            status,
        )
        status = jnp.where(
            eligible & output_finite & ~positive,
            int(LearnedThermalResearchStatus.NONPOSITIVE_POPULATIONS),
            status,
        ).astype(jnp.int32)

        evidence = LearnedThermalEnergyEvidence(
            target_total_energy=energy,
            recovered_total_energy=recovered_energy,
            total_energy_residual=recovered_energy - energy,
            recovered_sufficient_moments=recovered_statistics,
            log_partition=log_partition,
            constant_statistic_residual=constant_residual,
            logit_span=logit_span,
            minimum_population=jnp.min(populations, axis=-1),
            finite=finite & output_finite,
            positive=positive,
            within_declared_support=within_support,
            status=status,
            quadrature_id=self.quadrature.quadrature_id,
            plan_id=self.plan_id,
        )
        return LearnedThermalEnergyResult(
            populations, statistics, parameters, evidence, self.plan_id
        )


class ThermalQuasiEquilibriumEvidence(StrictModule):
    """Stress coupling and native-energy identity for the thermal ``g_star``."""

    equilibrium_total_energy: Array
    quasi_equilibrium_total_energy: Array
    total_energy_residual: Array
    particle_stress: Array
    equilibrium_particle_stress: Array
    stress_defect: Array
    contracted_stress_defect: Array
    weighted_velocity_sum: Array
    raw_correction_energy_residual: Array
    correction_energy_residual: Array
    minimum_population: Array
    finite: Array
    positive: Array
    status: Array
    quadrature_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(LearnedThermalResearchStatus.SUCCESS)


class ThermalQuasiEquilibriumResult(StrictModule):
    """Native-energy quasi-equilibrium and its explicit stress correction."""

    populations: Array
    raw_correction: Array
    correction: Array
    evidence: ThermalQuasiEquilibriumEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class ThermalCrossRelaxationEvidence(StrictModule):
    """Rate and exact energy evidence for the matched two-rate update."""

    particle_relaxation_time: Array
    energy_relaxation_time: Array
    particle_relaxation_rate: Array
    energy_relaxation_rate: Array
    prandtl_number: Array
    effective_prandtl_number: Array
    pre_collision_total_energy: Array
    equilibrium_total_energy: Array
    quasi_equilibrium_total_energy: Array
    post_collision_total_energy: Array
    conservation_residual: Array
    maximum_input_energy_mismatch: Array
    minimum_population: Array
    finite: Array
    positive: Array
    status: Array
    quadrature_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(LearnedThermalResearchStatus.SUCCESS)


class ThermalCrossRelaxationResult(StrictModule):
    """Candidate energy populations from the matched Prandtl collision term."""

    populations: Array
    equilibrium_increment: Array
    cross_relaxation_increment: Array
    evidence: ThermalCrossRelaxationEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class MatchedThermalCrossRelaxationPlan(StrictModule, NonTrainableState):
    """Prepare native ``g_star`` and the matched independently-Pr collision.

    NeurDE canonical writes ``sum(g_paper)=2 E`` and uses ``2/T`` in its stress
    correction. Here ``g = g_paper/2``, so the native coefficient is ``1/T``.
    The D2Q9 temperature weights are ``W_0=1-T`` and ``W_±=T/2`` in each
    direction. A residual sum is removed
    explicitly along those same weights; this exposes rather than assumes the
    finite-precision symmetry identity.

    Given ``tau_1 > 1/2`` and a prescribed ``Pr > 0``, this plan sets
    ``tau_2 = 1/2 + (tau_1 - 1/2)/Pr`` and applies

    ``g' = g + omega_2 (g_eq-g) + (omega_2-omega_1) (g_star-g)``.
    """

    quadrature: CertifiedDiscreteVelocityQuadrature
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        /,
        *,
        conservation_tolerance: float = 1.0e-11,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        _require_standard_d2q9(quadrature, owner="The matched thermal plan")
        tolerance = float(conservation_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be finite and positive.")
        self.quadrature = quadrature
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "matched-thermal-cross-relaxation-research",
                "quadrature": quadrature.quadrature_id,
                "thermal_weights": "tensor-product-W0=1-T-Wpm=T/2",
                "conservation_tolerance": tolerance,
                "energy_convention": "weight-absorbed-native-total-energy-sum",
            }
        )

    def quasi_equilibrium(
        self,
        energy_equilibrium: ArrayLike,
        particle_populations: ArrayLike,
        particle_equilibrium: ArrayLike,
        velocity: ArrayLike,
        temperature: ArrayLike,
        /,
    ) -> ThermalQuasiEquilibriumResult:
        """Construct ``g_star`` from the particle stress defect."""

        energy = self.quadrature.validate_populations(energy_equilibrium)
        particles = self.quadrature.validate_populations(particle_populations)
        particles_eq = self.quadrature.validate_populations(particle_equilibrium)
        flow = jnp.asarray(velocity)
        thermal = jnp.asarray(temperature)
        if particles.shape != energy.shape or particles_eq.shape != energy.shape:
            raise ValueError("All quasi-equilibrium populations must have equal shapes.")
        batch_shape = energy.shape[:-1]
        if flow.shape != (*batch_shape, 2):
            raise ValueError("velocity must have population batch axes plus (2,).")
        if thermal.shape != batch_shape:
            raise ValueError("temperature must have the population batch axes.")
        for name, value in (
            ("energy_equilibrium", energy),
            ("particle_populations", particles),
            ("particle_equilibrium", particles_eq),
            ("velocity", flow),
            ("temperature", thermal),
        ):
            if not jnp.issubdtype(value.dtype, jnp.number) or jnp.issubdtype(
                value.dtype, jnp.complexfloating
            ):
                raise TypeError(f"{name} must be real numeric data.")

        dtype = self.quadrature.velocities.dtype
        energy = jnp.asarray(energy, dtype=dtype)
        particles = jnp.asarray(particles, dtype=dtype)
        particles_eq = jnp.asarray(particles_eq, dtype=dtype)
        flow = jnp.asarray(flow, dtype=dtype)
        thermal = jnp.asarray(thermal, dtype=dtype)
        finite = (
            jnp.all(jnp.isfinite(energy), axis=-1)
            & jnp.all(jnp.isfinite(particles), axis=-1)
            & jnp.all(jnp.isfinite(particles_eq), axis=-1)
            & jnp.all(jnp.isfinite(flow), axis=-1)
            & jnp.isfinite(thermal)
        )
        positive_inputs = (
            jnp.all(energy > 0.0, axis=-1)
            & jnp.all(particles > 0.0, axis=-1)
            & jnp.all(particles_eq > 0.0, axis=-1)
        )
        thermal_support = (thermal > 0.0) & (thermal < 1.0)
        equilibrium_total_energy = jnp.sum(energy, axis=-1)
        positive_energy = equilibrium_total_energy > 0.0
        eligible = finite & positive_inputs & thermal_support & positive_energy
        safe_energy = jnp.where(jnp.isfinite(energy), energy, 0.0)
        safe_particles = jnp.where(jnp.isfinite(particles), particles, 0.0)
        safe_particles_eq = jnp.where(jnp.isfinite(particles_eq), particles_eq, 0.0)
        safe_flow = jnp.where(jnp.isfinite(flow), flow, 0.0)
        safe_temperature = jnp.where(thermal_support & finite, thermal, 0.5)
        directional_weights = jnp.where(
            self.quadrature.velocities == 0.0,
            1.0 - safe_temperature[..., None, None],
            0.5 * safe_temperature[..., None, None],
        )
        thermal_weights = jnp.prod(directional_weights, axis=-1)
        weighted_velocity_sum = ein.contract(
            "...q,qd->...d", thermal_weights, self.quadrature.velocities
        )
        energy_lift = thermal_weights / jnp.sum(thermal_weights, axis=-1, keepdims=True)

        stress = ein.contract(
            "...q,qa,qb->...ab",
            safe_particles,
            self.quadrature.velocities,
            self.quadrature.velocities,
        )
        equilibrium_stress = ein.contract(
            "...q,qa,qb->...ab",
            safe_particles_eq,
            self.quadrature.velocities,
            self.quadrature.velocities,
        )
        stress_defect = stress - equilibrium_stress
        contracted_defect = ein.contract("...ab,...b->...a", stress_defect, safe_flow)
        projected_defect = ein.contract(
            "qa,...a->...q", self.quadrature.velocities, contracted_defect
        )
        raw_correction = thermal_weights * projected_defect / safe_temperature[..., None]
        raw_energy_residual = jnp.sum(raw_correction, axis=-1)
        correction = raw_correction - energy_lift * raw_energy_residual[..., None]
        corrected_energy_residual = jnp.sum(correction, axis=-1)
        raw_quasi = safe_energy + correction
        output_finite = jnp.all(jnp.isfinite(raw_quasi), axis=-1)
        positive = jnp.all(raw_quasi > 0.0, axis=-1)
        populations = jnp.where(eligible[..., None], raw_quasi, 0.0)
        quasi_total_energy = jnp.sum(populations, axis=-1)
        energy_residual = quasi_total_energy - equilibrium_total_energy
        scale = jnp.maximum(jnp.abs(equilibrium_total_energy), 1.0)
        conserved = jnp.abs(energy_residual) <= self.conservation_tolerance * scale

        status = jnp.full(
            batch_shape, int(LearnedThermalResearchStatus.SUCCESS), jnp.int32
        )
        status = jnp.where(
            ~finite, int(LearnedThermalResearchStatus.NONFINITE_INPUT), status
        )
        status = jnp.where(
            finite & ~thermal_support,
            int(LearnedThermalResearchStatus.OUTSIDE_DECLARED_SUPPORT),
            status,
        )
        status = jnp.where(
            finite & thermal_support & ~positive_energy,
            int(LearnedThermalResearchStatus.NONPOSITIVE_TOTAL_ENERGY),
            status,
        )
        status = jnp.where(
            finite & thermal_support & positive_energy & ~positive_inputs,
            int(LearnedThermalResearchStatus.NONPOSITIVE_POPULATIONS),
            status,
        )
        status = jnp.where(
            eligible & ~output_finite,
            int(LearnedThermalResearchStatus.NONFINITE_OUTPUT),
            status,
        )
        status = jnp.where(
            eligible & output_finite & ~positive,
            int(LearnedThermalResearchStatus.NONPOSITIVE_POPULATIONS),
            status,
        )
        status = jnp.where(
            eligible & output_finite & positive & ~conserved,
            int(LearnedThermalResearchStatus.INCONSISTENT_ENERGY_MOMENT),
            status,
        ).astype(jnp.int32)

        evidence = ThermalQuasiEquilibriumEvidence(
            equilibrium_total_energy=equilibrium_total_energy,
            quasi_equilibrium_total_energy=quasi_total_energy,
            total_energy_residual=energy_residual,
            particle_stress=stress,
            equilibrium_particle_stress=equilibrium_stress,
            stress_defect=stress_defect,
            contracted_stress_defect=contracted_defect,
            weighted_velocity_sum=weighted_velocity_sum,
            raw_correction_energy_residual=raw_energy_residual,
            correction_energy_residual=corrected_energy_residual,
            minimum_population=jnp.min(populations, axis=-1),
            finite=finite & output_finite,
            positive=positive,
            status=status,
            quadrature_id=self.quadrature.quadrature_id,
            plan_id=self.plan_id,
        )
        return ThermalQuasiEquilibriumResult(
            populations, raw_correction, correction, evidence, self.plan_id
        )

    def cross_relax(
        self,
        energy_populations: ArrayLike,
        energy_equilibrium: ArrayLike,
        quasi_equilibrium: ArrayLike,
        particle_relaxation_time: ArrayLike,
        prandtl_number: ArrayLike,
        /,
    ) -> ThermalCrossRelaxationResult:
        """Apply the two-rate energy update for an independently supplied Prandtl number."""

        energy = self.quadrature.validate_populations(energy_populations)
        equilibrium = self.quadrature.validate_populations(energy_equilibrium)
        quasi = self.quadrature.validate_populations(quasi_equilibrium)
        if equilibrium.shape != energy.shape or quasi.shape != energy.shape:
            raise ValueError("Cross-relaxation populations must have equal shapes.")
        batch_shape = energy.shape[:-1]
        particle_time = jnp.asarray(particle_relaxation_time)
        prandtl = jnp.asarray(prandtl_number)
        if particle_time.shape == ():
            particle_time = jnp.broadcast_to(particle_time, batch_shape)
        if prandtl.shape == ():
            prandtl = jnp.broadcast_to(prandtl, batch_shape)
        if particle_time.shape != batch_shape or prandtl.shape != batch_shape:
            raise ValueError("Relaxation parameters must be scalar or match batch axes.")
        for name, value in (
            ("particle_relaxation_time", particle_time),
            ("prandtl_number", prandtl),
        ):
            if not jnp.issubdtype(value.dtype, jnp.number) or jnp.issubdtype(
                value.dtype, jnp.complexfloating
            ):
                raise TypeError(f"{name} must be real numeric data.")

        dtype = self.quadrature.velocities.dtype
        energy = jnp.asarray(energy, dtype=dtype)
        equilibrium = jnp.asarray(equilibrium, dtype=dtype)
        quasi = jnp.asarray(quasi, dtype=dtype)
        particle_time = jnp.asarray(particle_time, dtype=dtype)
        prandtl = jnp.asarray(prandtl, dtype=dtype)
        finite = (
            jnp.all(jnp.isfinite(energy), axis=-1)
            & jnp.all(jnp.isfinite(equilibrium), axis=-1)
            & jnp.all(jnp.isfinite(quasi), axis=-1)
            & jnp.isfinite(particle_time)
            & jnp.isfinite(prandtl)
        )
        valid_parameters = (particle_time > 0.5) & (prandtl > 0.0)
        safe_particle_time = jnp.where(valid_parameters & finite, particle_time, 1.0)
        safe_prandtl = jnp.where(valid_parameters & finite, prandtl, 1.0)
        energy_time = 0.5 + (safe_particle_time - 0.5) / safe_prandtl
        particle_rate = 1.0 / safe_particle_time
        energy_rate = 1.0 / energy_time
        effective_prandtl = (safe_particle_time - 0.5) / (energy_time - 0.5)
        positive_inputs = (
            jnp.all(energy > 0.0, axis=-1)
            & jnp.all(equilibrium > 0.0, axis=-1)
            & jnp.all(quasi > 0.0, axis=-1)
        )
        pre_energy = jnp.sum(energy, axis=-1)
        equilibrium_energy = jnp.sum(equilibrium, axis=-1)
        quasi_energy = jnp.sum(quasi, axis=-1)
        mismatch = jnp.maximum(
            jnp.abs(equilibrium_energy - pre_energy),
            jnp.abs(quasi_energy - pre_energy),
        )
        scale = jnp.maximum(jnp.abs(pre_energy), 1.0)
        consistent = mismatch <= self.conservation_tolerance * scale
        equilibrium_increment = energy_rate[..., None] * (equilibrium - energy)
        cross_increment = (energy_rate - particle_rate)[..., None] * (quasi - energy)
        candidate = energy + equilibrium_increment + cross_increment
        eligible = finite & valid_parameters
        populations = jnp.where(eligible[..., None], candidate, 0.0)
        post_energy = jnp.sum(populations, axis=-1)
        residual = post_energy - pre_energy
        output_finite = jnp.all(jnp.isfinite(candidate), axis=-1)
        positive = jnp.all(candidate > 0.0, axis=-1)
        conserved = jnp.abs(residual) <= self.conservation_tolerance * scale

        status = jnp.full(
            batch_shape, int(LearnedThermalResearchStatus.SUCCESS), jnp.int32
        )
        status = jnp.where(
            ~finite, int(LearnedThermalResearchStatus.NONFINITE_INPUT), status
        )
        status = jnp.where(
            finite & ~valid_parameters,
            int(LearnedThermalResearchStatus.INVALID_RELAXATION_PARAMETERS),
            status,
        )
        status = jnp.where(
            eligible & ~positive_inputs,
            int(LearnedThermalResearchStatus.NONPOSITIVE_POPULATIONS),
            status,
        )
        status = jnp.where(
            eligible & positive_inputs & ~consistent,
            int(LearnedThermalResearchStatus.INCONSISTENT_ENERGY_MOMENT),
            status,
        )
        status = jnp.where(
            eligible & positive_inputs & consistent & ~output_finite,
            int(LearnedThermalResearchStatus.NONFINITE_OUTPUT),
            status,
        )
        status = jnp.where(
            eligible & positive_inputs & consistent & output_finite & ~positive,
            int(LearnedThermalResearchStatus.NONPOSITIVE_POPULATIONS),
            status,
        )
        status = jnp.where(
            eligible
            & positive_inputs
            & consistent
            & output_finite
            & positive
            & ~conserved,
            int(LearnedThermalResearchStatus.INCONSISTENT_ENERGY_MOMENT),
            status,
        ).astype(jnp.int32)

        evidence = ThermalCrossRelaxationEvidence(
            particle_relaxation_time=particle_time,
            energy_relaxation_time=energy_time,
            particle_relaxation_rate=particle_rate,
            energy_relaxation_rate=energy_rate,
            prandtl_number=prandtl,
            effective_prandtl_number=effective_prandtl,
            pre_collision_total_energy=pre_energy,
            equilibrium_total_energy=equilibrium_energy,
            quasi_equilibrium_total_energy=quasi_energy,
            post_collision_total_energy=post_energy,
            conservation_residual=residual,
            maximum_input_energy_mismatch=mismatch,
            minimum_population=jnp.min(populations, axis=-1),
            finite=finite & output_finite,
            positive=positive,
            status=status,
            quadrature_id=self.quadrature.quadrature_id,
            plan_id=self.plan_id,
        )
        return ThermalCrossRelaxationResult(
            populations,
            equilibrium_increment,
            cross_increment,
            evidence,
            self.plan_id,
        )


def _cross_2d(origin: np.ndarray, first: np.ndarray, second: np.ndarray, /) -> float:
    left = first - origin
    right = second - origin
    return float(left[0] * right[1] - left[1] * right[0])


def _velocity_halfspaces(velocities: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(velocities, axis=0)
    if unique.shape[0] < 3:
        raise ValueError("Frame-shift velocity support must span two dimensions.")
    scale = max(float(np.max(np.abs(unique))), 1.0)
    tolerance = 128.0 * np.finfo(unique.dtype).eps * scale * scale
    lower: list[np.ndarray] = []
    for point in unique:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], point) <= tolerance:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in unique[::-1]:
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], point) <= tolerance:
            upper.pop()
        upper.append(point)
    hull = np.stack(lower[:-1] + upper[:-1], axis=0)
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.sqrt(np.sum(edges * edges, axis=-1))
    if hull.shape[0] < 3 or np.any(lengths <= tolerance):
        raise ValueError("Frame-shift velocity support must span two dimensions.")
    normals = np.stack((edges[:, 1], -edges[:, 0]), axis=-1) / lengths[:, None]
    offsets = np.sum(normals * hull, axis=-1)
    return normals, offsets


class IntegerVelocityFrameAdmissibilityEvidence(StrictModule):
    """Convex-support evidence for one intended laboratory velocity."""

    laboratory_velocity: Array
    relative_velocity: Array
    halfspace_slack: Array
    interior_margin: Array
    finite: Array
    admissible: Array
    status: Array
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(LearnedThermalResearchStatus.SUCCESS)


class ThermalFrameMoments(StrictModule):
    """Raw moments of unchanged populations with respect to one velocity frame."""

    density: Array
    particle_momentum: Array
    particle_stress: Array
    total_energy: Array
    total_energy_flux: Array


class IntegerVelocityFrameShiftEvidence(StrictModule):
    """Admissibility and exact moment-translation evidence for one frame shift."""

    admissibility: IntegerVelocityFrameAdmissibilityEvidence
    source_moments: ThermalFrameMoments
    transformed_moments: ThermalFrameMoments
    predicted_moments: ThermalFrameMoments
    maximum_identity_residual: Array
    minimum_particle_population: Array
    minimum_total_energy_population: Array
    finite: Array
    positive: Array
    status: Array
    quadrature_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(LearnedThermalResearchStatus.SUCCESS)


class IntegerVelocityFrameShiftResult(StrictModule):
    """Moments evaluated at the integer-translated physical abscissae."""

    moments: ThermalFrameMoments
    evidence: IntegerVelocityFrameShiftEvidence
    plan_id: str = eqx.field(static=True)

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class IntegerVelocityFrameShiftPlan(StrictModule, NonTrainableState):
    """Interpret an integer lattice rule in a translated velocity frame.

    The stored populations are unchanged and physical abscissae become
    ``c_lab = c + shift``. Therefore, exactly,

    ``m_lab = m + rho shift``,
    ``P_lab = P + shift⊗m + m⊗shift + rho shift⊗shift``, and
    ``q_lab = q + E shift`` while ``rho`` and native ``E=sum(g)`` are unchanged.

    This is a local frame/moment plan, not a population remap or spatial
    streaming implementation. Only integer-lattice quadratures and integral
    shifts are accepted. The relative mean velocity must remain strictly inside
    the convex hull of the unshifted rule.
    """

    quadrature: CertifiedDiscreteVelocityQuadrature
    shift: Array
    halfspace_normals: Array
    halfspace_offsets: Array
    minimum_interior_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        shift: ArrayLike,
        /,
        *,
        minimum_interior_margin: float = 1.0e-10,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        if quadrature.dimension != 2:
            raise ValueError("The integer frame-shift plan requires dimension two.")
        if quadrature.transport_kind != "integer_lattice":
            raise ValueError("Velocity-frame shifting requires an integer-lattice rule.")
        shift_host = np.asarray(shift)
        if shift_host.shape != (2,) or not np.issubdtype(shift_host.dtype, np.number):
            raise ValueError("shift must be one real vector with shape (2,).")
        if np.iscomplexobj(shift_host) or np.any(~np.isfinite(shift_host)):
            raise ValueError("shift must be finite and real.")
        rounded = np.rint(shift_host)
        if np.max(np.abs(shift_host - rounded)) > quadrature.certification.tolerance:
            raise ValueError("Velocity-frame shift components must be integers.")
        margin = float(minimum_interior_margin)
        if not isfinite(margin) or margin < 0.0:
            raise ValueError("minimum_interior_margin must be finite and non-negative.")
        normals, offsets = _velocity_halfspaces(np.asarray(quadrature.velocities))
        dtype = quadrature.velocities.dtype
        self.quadrature = quadrature
        self.shift = jnp.asarray(rounded, dtype=dtype)
        self.halfspace_normals = jnp.asarray(normals, dtype=dtype)
        self.halfspace_offsets = jnp.asarray(offsets, dtype=dtype)
        self.minimum_interior_margin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "integer-velocity-frame-shift-research",
                "quadrature": quadrature.quadrature_id,
                "shift": array_tree_fingerprint(rounded),
                "halfspace_normals": array_tree_fingerprint(normals),
                "halfspace_offsets": array_tree_fingerprint(offsets),
                "minimum_interior_margin": margin,
            }
        )

    def assess(
        self, laboratory_velocity: ArrayLike, /
    ) -> IntegerVelocityFrameAdmissibilityEvidence:
        """Report whether ``u_lab - shift`` lies inside the base velocity hull."""

        velocity = jnp.asarray(laboratory_velocity)
        if not jnp.issubdtype(velocity.dtype, jnp.number) or jnp.issubdtype(
            velocity.dtype, jnp.complexfloating
        ):
            raise TypeError("laboratory_velocity must be real numeric data.")
        if velocity.ndim == 0 or velocity.shape[-1] != 2:
            raise ValueError("laboratory_velocity must have trailing shape (2,).")
        velocity = jnp.asarray(velocity, dtype=self.quadrature.velocities.dtype)
        finite = jnp.all(jnp.isfinite(velocity), axis=-1)
        safe_velocity = jnp.where(finite[..., None], velocity, self.shift)
        relative = safe_velocity - self.shift
        slack = self.halfspace_offsets - ein.contract(
            "hd,...d->...h", self.halfspace_normals, relative
        )
        margin = jnp.min(slack, axis=-1)
        admissible = finite & (margin > self.minimum_interior_margin)
        status = jnp.where(
            ~finite,
            int(LearnedThermalResearchStatus.NONFINITE_INPUT),
            jnp.where(
                admissible,
                int(LearnedThermalResearchStatus.SUCCESS),
                int(LearnedThermalResearchStatus.INADMISSIBLE_FRAME_SHIFT),
            ),
        ).astype(jnp.int32)
        return IntegerVelocityFrameAdmissibilityEvidence(
            laboratory_velocity=velocity,
            relative_velocity=relative,
            halfspace_slack=slack,
            interior_margin=margin,
            finite=finite,
            admissible=admissible,
            status=status,
            plan_id=self.plan_id,
        )

    def _moments(
        self, particle_populations: Array, energy_populations: Array, velocities: Array, /
    ) -> ThermalFrameMoments:
        return ThermalFrameMoments(
            density=jnp.sum(particle_populations, axis=-1),
            particle_momentum=ein.contract(
                "...q,qd->...d", particle_populations, velocities
            ),
            particle_stress=ein.contract(
                "...q,qd,qe->...de", particle_populations, velocities, velocities
            ),
            total_energy=jnp.sum(energy_populations, axis=-1),
            total_energy_flux=ein.contract(
                "...q,qd->...d", energy_populations, velocities
            ),
        )

    @staticmethod
    def _translate(moments: ThermalFrameMoments, delta: Array, /) -> ThermalFrameMoments:
        momentum = moments.particle_momentum + moments.density[..., None] * delta
        stress = (
            moments.particle_stress
            + ein.contract("d,...e->...de", delta, moments.particle_momentum)
            + ein.contract("...d,e->...de", moments.particle_momentum, delta)
            + moments.density[..., None, None] * ein.contract("d,e->de", delta, delta)
        )
        return ThermalFrameMoments(
            density=moments.density,
            particle_momentum=momentum,
            particle_stress=stress,
            total_energy=moments.total_energy,
            total_energy_flux=(
                moments.total_energy_flux + moments.total_energy[..., None] * delta
            ),
        )

    def inverse(self, laboratory_moments: ThermalFrameMoments, /) -> ThermalFrameMoments:
        """Apply the exact inverse moment translation without remapping populations."""

        if not isinstance(laboratory_moments, ThermalFrameMoments):
            raise TypeError("laboratory_moments must be ThermalFrameMoments.")
        return self._translate(laboratory_moments, -self.shift)

    def forward(
        self,
        particle_populations: ArrayLike,
        total_energy_populations: ArrayLike,
        /,
    ) -> IntegerVelocityFrameShiftResult:
        """Evaluate source and translated moments of unchanged populations."""

        particles = self.quadrature.validate_populations(particle_populations)
        energy = self.quadrature.validate_populations(total_energy_populations)
        if particles.shape != energy.shape:
            raise ValueError("Particle and energy population shapes must agree.")
        for name, value in (
            ("particle_populations", particles),
            ("total_energy_populations", energy),
        ):
            if not jnp.issubdtype(value.dtype, jnp.number) or jnp.issubdtype(
                value.dtype, jnp.complexfloating
            ):
                raise TypeError(f"{name} must be real numeric data.")
        dtype = self.quadrature.velocities.dtype
        particles = jnp.asarray(particles, dtype=dtype)
        energy = jnp.asarray(energy, dtype=dtype)
        source = self._moments(particles, energy, self.quadrature.velocities)
        shifted_velocities = self.quadrature.velocities + self.shift
        transformed = self._moments(particles, energy, shifted_velocities)
        predicted = self._translate(source, self.shift)
        safe_density = jnp.where(source.density > 0.0, source.density, 1.0)
        laboratory_velocity = transformed.particle_momentum / safe_density[..., None]
        admissibility = self.assess(laboratory_velocity)

        density_residual = transformed.density - predicted.density
        momentum_residual = transformed.particle_momentum - predicted.particle_momentum
        stress_residual = transformed.particle_stress - predicted.particle_stress
        energy_residual = transformed.total_energy - predicted.total_energy
        flux_residual = transformed.total_energy_flux - predicted.total_energy_flux
        maximum_residual = jnp.maximum(
            jnp.maximum(
                jnp.abs(density_residual),
                jnp.max(jnp.abs(momentum_residual), axis=-1),
            ),
            jnp.maximum(
                jnp.max(jnp.abs(stress_residual), axis=(-2, -1)),
                jnp.maximum(
                    jnp.abs(energy_residual),
                    jnp.max(jnp.abs(flux_residual), axis=-1),
                ),
            ),
        )
        finite = (
            jnp.all(jnp.isfinite(particles), axis=-1)
            & jnp.all(jnp.isfinite(energy), axis=-1)
            & jnp.isfinite(maximum_residual)
            & (source.density > 0.0)
            & (source.total_energy > 0.0)
        )
        positive = jnp.all(particles > 0.0, axis=-1) & jnp.all(energy > 0.0, axis=-1)
        identity_scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(transformed.particle_stress), axis=(-2, -1)),
                jnp.max(jnp.abs(transformed.total_energy_flux), axis=-1),
            ),
            1.0,
        )
        identity_ok = (
            maximum_residual <= self.quadrature.certification.tolerance * identity_scale
        )
        status = jnp.full(
            particles.shape[:-1],
            int(LearnedThermalResearchStatus.SUCCESS),
            jnp.int32,
        )
        status = jnp.where(
            ~finite, int(LearnedThermalResearchStatus.NONFINITE_INPUT), status
        )
        status = jnp.where(
            finite & ~positive,
            int(LearnedThermalResearchStatus.NONPOSITIVE_POPULATIONS),
            status,
        )
        status = jnp.where(
            finite & positive & ~admissibility.admissible,
            int(LearnedThermalResearchStatus.INADMISSIBLE_FRAME_SHIFT),
            status,
        )
        status = jnp.where(
            finite & positive & admissibility.admissible & ~identity_ok,
            int(LearnedThermalResearchStatus.NONFINITE_OUTPUT),
            status,
        ).astype(jnp.int32)
        evidence = IntegerVelocityFrameShiftEvidence(
            admissibility=admissibility,
            source_moments=source,
            transformed_moments=transformed,
            predicted_moments=predicted,
            maximum_identity_residual=maximum_residual,
            minimum_particle_population=jnp.min(particles, axis=-1),
            minimum_total_energy_population=jnp.min(energy, axis=-1),
            finite=finite,
            positive=positive,
            status=status,
            quadrature_id=self.quadrature.quadrature_id,
            plan_id=self.plan_id,
        )
        return IntegerVelocityFrameShiftResult(transformed, evidence, self.plan_id)


__all__ = [
    "ExtendedParticleEquilibriumEvidence",
    "ExtendedParticleEquilibriumResult",
    "IntegerVelocityFrameAdmissibilityEvidence",
    "IntegerVelocityFrameShiftEvidence",
    "IntegerVelocityFrameShiftPlan",
    "IntegerVelocityFrameShiftResult",
    "LearnedThermalEnergyEvidence",
    "LearnedThermalEnergyResult",
    "LearnedThermalResearchStatus",
    "MatchedThermalCrossRelaxationPlan",
    "PositiveLearnedThermalEnergyPlan",
    "PressureExtendedParticleEquilibriumPlan",
    "ThermalCrossRelaxationEvidence",
    "ThermalCrossRelaxationResult",
    "ThermalFrameMoments",
    "ThermalQuasiEquilibriumEvidence",
    "ThermalQuasiEquilibriumResult",
]
