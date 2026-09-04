#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....nonlinear import NewtonKrylov, NonlinearResult, NonlinearTermination
from ._gasam import (
    PreparedEngelhardtGasam2025Material,
    QualifiedExactMixedGasamProblem,
)


class GasamQualificationEvidence(StrictModule, NonTrainableState):
    """Pointwise objectivity, derivatives, power, limits, and local stability."""

    objectivity_energy_error: Array
    objectivity_stress_error: Array
    stress_gradient_error: Array
    tangent_jvp_error: Array
    power_error: Array
    passive_reference_energy: Array
    passive_reference_stress_norm: Array
    active_fiber_stress_increment_pa: Array
    minimum_acoustic_value_pa: Array
    finite: Array
    within_tolerance: Array
    valid: Array
    passive_polyconvexity_source: str = eqx.field(static=True)
    active_global_stability_claimed: bool = eqx.field(static=True)
    branch_scope: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)


class GasamQualificationPlan(StrictModule, NonTrainableState):
    """Fixed-capacity qualification plan for one smooth constitutive branch.

    The acoustic scan is evidence over the declared six direction pairs; it is
    deliberately not advertised as a proof of global rank-one convexity.
    """

    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance_pa: float = eqx.field(static=True)
    minimum_acoustic_value_pa: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        relative_tolerance: float = 2.0e-5,
        absolute_tolerance_pa: float = 1.0e-4,
        minimum_acoustic_value_pa: float = 0.0,
    ):
        relative = float(relative_tolerance)
        absolute = float(absolute_tolerance_pa)
        minimum = float(minimum_acoustic_value_pa)
        if not isfinite(relative) or relative <= 0.0:
            raise ValueError("relative_tolerance must be positive and finite.")
        if not isfinite(absolute) or absolute <= 0.0:
            raise ValueError("absolute_tolerance_pa must be positive and finite.")
        if not isfinite(minimum):
            raise ValueError("minimum_acoustic_value_pa must be finite.")
        self.relative_tolerance = relative
        self.absolute_tolerance_pa = absolute
        self.minimum_acoustic_value_pa = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "engelhardt-gasam-2025-qualification-plan",
                "relative_tolerance": relative.hex(),
                "absolute_tolerance_pa": absolute.hex(),
                "minimum_acoustic_value_pa": minimum.hex(),
                "acoustic_pair_capacity": 6,
            }
        )

    def evaluate(
        self,
        material: PreparedEngelhardtGasam2025Material,
        deformation_gradient: ArrayLike,
        deformation_rate_per_s: ArrayLike,
        /,
        *,
        pressure_pa: ArrayLike = 0.0,
    ) -> GasamQualificationEvidence:
        if not isinstance(material, PreparedEngelhardtGasam2025Material):
            raise TypeError("material must be PreparedEngelhardtGasam2025Material.")
        deformation = jnp.asarray(deformation_gradient)
        rate = jnp.asarray(deformation_rate_per_s)
        if deformation.shape != (3, 3) or rate.shape != (3, 3):
            raise ValueError("deformation_gradient and deformation_rate_per_s must be 3x3.")
        pressure = jnp.asarray(pressure_pa, dtype=deformation.dtype)
        if pressure.shape != ():
            raise ValueError("pressure_pa must be one scalar.")

        response = material.evaluate(deformation, pressure)
        law = material.mixed_law()
        tangent = material.block_tangent(deformation, pressure).deformation_deformation
        energy_gradient = jax.grad(law.isochoric_value)(deformation)
        source_stress = law.first_piola(deformation, pressure)
        stress_gradient_error = jnp.linalg.norm(
            response.mixed.isochoric_first_piola - energy_gradient
        )

        stress_jvp = jax.jvp(
            lambda value: law.first_piola(value, pressure),
            (deformation,),
            (rate,),
        )[1]
        tangent_rate = ein.contract("iJkL,kL->iJ", tangent, rate)
        tangent_jvp_error = jnp.linalg.norm(stress_jvp - tangent_rate)
        energy_rate = jax.jvp(
            lambda value: law.potential(value, pressure),
            (deformation,),
            (rate,),
        )[1]
        stress_power = ein.contract("iJ,iJ->", source_stress, rate)
        power_error = jnp.abs(energy_rate - stress_power)

        axis = jnp.asarray([0.0, 0.0, 1.0], dtype=deformation.dtype)
        angle = jnp.asarray(0.37, dtype=deformation.dtype)
        cross = jnp.asarray(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=deformation.dtype,
        )
        rotation = jnp.eye(3, dtype=deformation.dtype) + jnp.sin(angle) * cross + (
            1.0 - jnp.cos(angle)
        ) * (cross @ cross)
        rotated = material.evaluate(rotation @ deformation, pressure)
        objectivity_energy_error = jnp.abs(
            rotated.reference_energy_density - response.reference_energy_density
        )
        objectivity_stress_error = jnp.linalg.norm(
            rotated.first_piola - rotation @ response.first_piola
        )

        passive_candidate = material.propose_activation(0.0)
        passive_commit = passive_candidate.commit()
        passive = material.with_commit(passive_commit)
        identity = jnp.eye(3, dtype=deformation.dtype)
        passive_reference = passive.evaluate(identity, 0.0)
        active_reference = material.evaluate(identity, 0.0)
        fiber = material.architecture.reference_direction
        active_increment = ein.contract(
            "i,iJ,J->",
            fiber,
            active_reference.first_piola - passive_reference.first_piola,
            fiber,
        )

        spatial = jnp.stack(
            (
                fiber,
                jnp.asarray([1.0, 0.0, 0.0], dtype=deformation.dtype),
                jnp.asarray([0.0, 1.0, 0.0], dtype=deformation.dtype),
                jnp.asarray([0.0, 0.0, 1.0], dtype=deformation.dtype),
                jnp.asarray([1.0, 1.0, 0.0], dtype=deformation.dtype)
                / jnp.sqrt(2.0),
                jnp.asarray([0.0, 1.0, 1.0], dtype=deformation.dtype)
                / jnp.sqrt(2.0),
            )
        )
        normals = jnp.stack(
            (
                fiber,
                jnp.asarray([0.0, 1.0, 0.0], dtype=deformation.dtype),
                jnp.asarray([0.0, 0.0, 1.0], dtype=deformation.dtype),
                jnp.asarray([1.0, 0.0, 0.0], dtype=deformation.dtype),
                jnp.asarray([1.0, -1.0, 0.0], dtype=deformation.dtype)
                / jnp.sqrt(2.0),
                jnp.asarray([0.0, 1.0, -1.0], dtype=deformation.dtype)
                / jnp.sqrt(2.0),
            )
        )
        acoustic = jax.vmap(
            lambda a, n: ein.contract("i,J,iJkL,k,L->", a, n, tangent, a, n)
        )(spatial, normals)
        minimum_acoustic = jnp.min(acoustic)

        scale = jnp.maximum(
            jnp.asarray(1.0, dtype=deformation.dtype),
            jnp.linalg.norm(source_stress),
        )
        relative_bound = (
            self.absolute_tolerance_pa + self.relative_tolerance * scale
        )
        passive_reference_stress_norm = jnp.linalg.norm(
            passive_reference.first_piola
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        objectivity_energy_error,
                        objectivity_stress_error,
                        stress_gradient_error,
                        tangent_jvp_error,
                        power_error,
                        passive_reference.reference_energy_density,
                        passive_reference_stress_norm,
                        active_increment,
                        minimum_acoustic,
                    )
                )
            )
        )
        within = (
            (objectivity_energy_error <= relative_bound)
            & (objectivity_stress_error <= relative_bound)
            & (stress_gradient_error <= relative_bound)
            & (tangent_jvp_error <= relative_bound)
            & (power_error <= relative_bound)
            & (
                jnp.abs(passive_reference.reference_energy_density)
                <= self.absolute_tolerance_pa
            )
            & (passive_reference_stress_norm <= self.absolute_tolerance_pa)
            & (active_increment > 0.0)
            & (minimum_acoustic > self.minimum_acoustic_value_pa)
        )
        valid = finite & within & response.evidence.valid
        return GasamQualificationEvidence(
            objectivity_energy_error,
            objectivity_stress_error,
            stress_gradient_error,
            tangent_jvp_error,
            power_error,
            passive_reference.reference_energy_density,
            passive_reference_stress_norm,
            active_increment,
            minimum_acoustic,
            finite,
            within,
            valid,
            "Ehret, Bol & Itskov (2011), DOI 10.1016/j.jmps.2010.12.008: passive law polyconvex and coercive",
            False,
            "local AD and acoustic evidence excludes the hard lambda=lambda_min branch",
            canonical_fingerprint(
                {
                    "kind": "engelhardt-gasam-2025-qualification",
                    "plan": self.plan_id,
                    "material": material.prepared_id,
                }
            ),
        )


class AffineMeshPowerEvidence(StrictModule, NonTrainableState):
    """Fixed-capacity mesh sequence evidence for affine energy and power."""

    active_cell_counts: Array
    reference_volumes_m3: Array
    integrated_energies_j: Array
    integrated_powers_w: Array
    energy_errors_j: Array
    power_errors_w: Array
    nonincreasing_error: Array
    valid: Array


def affine_mesh_power_evidence(
    material: PreparedEngelhardtGasam2025Material,
    cell_reference_volumes_m3: ArrayLike,
    active_cell_mask: ArrayLike,
    deformation_gradient: ArrayLike,
    deformation_rate_per_s: ArrayLike,
    /,
    *,
    pressure_pa: ArrayLike = 0.0,
) -> AffineMeshPowerEvidence:
    """Verify affine patch energy/power across fixed-capacity mesh levels.

    Each row is one mesh level and inactive capacity slots have exactly zero
    measure.  This evaluates the continuum power identity without introducing a
    second assembly or quadrature owner.
    """
    if not isinstance(material, PreparedEngelhardtGasam2025Material):
        raise TypeError("material must be PreparedEngelhardtGasam2025Material.")
    volumes = jnp.asarray(cell_reference_volumes_m3)
    mask = jnp.asarray(active_cell_mask, dtype=bool)
    deformation = jnp.asarray(deformation_gradient)
    rate = jnp.asarray(deformation_rate_per_s)
    if volumes.ndim != 2 or volumes.shape != mask.shape:
        raise ValueError("cell volumes and masks must share shape (level, capacity).")
    if deformation.shape != (3, 3) or rate.shape != (3, 3):
        raise ValueError("deformation and rate must have shape (3, 3).")
    if volumes.shape[0] < 2:
        raise ValueError("Mesh power evidence requires at least two mesh levels.")
    active_volumes = jnp.where(mask, volumes, 0.0)
    response = material.evaluate(deformation, pressure_pa)
    density = response.reference_energy_density
    power_density = ein.contract("iJ,iJ->", response.first_piola, rate)
    totals = jnp.sum(active_volumes, axis=1)
    energies = jnp.sum(active_volumes * density, axis=1)
    powers = jnp.sum(active_volumes * power_density, axis=1)
    exact_energy = totals[0] * density
    exact_power = totals[0] * power_density
    energy_errors = jnp.abs(energies - exact_energy)
    power_errors = jnp.abs(powers - exact_power)
    machine_scale = 32.0 * jnp.finfo(volumes.dtype).eps
    energy_tolerance = machine_scale * jnp.maximum(
        jnp.abs(exact_energy), jnp.finfo(volumes.dtype).tiny
    )
    power_tolerance = machine_scale * jnp.maximum(
        jnp.abs(exact_power), jnp.finfo(volumes.dtype).tiny
    )
    volume_tolerance = machine_scale * totals[0]
    nonincreasing = jnp.all(
        energy_errors[1:] <= energy_errors[:-1] + energy_tolerance
    ) & jnp.all(power_errors[1:] <= power_errors[:-1] + power_tolerance)
    finite = (
        jnp.all(jnp.isfinite(volumes))
        & jnp.all(volumes >= 0.0)
        & jnp.all(totals > 0.0)
        & jnp.all(jnp.abs(totals - totals[0]) <= volume_tolerance)
        & jnp.all(~mask | (volumes > 0.0))
        & jnp.all(mask | (volumes == 0.0))
        & jnp.all(jnp.isfinite(energies))
        & jnp.all(jnp.isfinite(powers))
    )
    return AffineMeshPowerEvidence(
        jnp.sum(mask, axis=1),
        totals,
        energies,
        powers,
        energy_errors,
        power_errors,
        nonincreasing,
        finite & nonincreasing & response.evidence.valid,
    )


class ManufacturedRestEvidence(StrictModule, NonTrainableState):
    """End-to-end exact mixed solve and rollback evidence at manufactured rest."""

    initial_residual_norm: Array
    final_residual_norm: Array
    nonlinear_successful: Array
    mixed_qualification_valid: Array
    valid: Array
    rollback_applied: bool = eqx.field(static=True)


class ManufacturedRestCommit(StrictModule, NonTrainableState):
    state: tuple[Array, Array]
    committed: bool = eqx.field(static=True)
    evidence: ManufacturedRestEvidence


class ManufacturedRestCandidate(StrictModule, NonTrainableState):
    """Whole-state transaction for the manufactured zero-load mixed solve."""

    previous_state: tuple[Array, Array]
    proposed_state: tuple[Array, Array]
    nonlinear_result: NonlinearResult
    evidence: ManufacturedRestEvidence

    def commit(self, /) -> ManufacturedRestCommit:
        accepted = bool(self.evidence.valid)
        return ManufacturedRestCommit(
            self.proposed_state if accepted else self.previous_state,
            accepted,
            self.evidence,
        )


def solve_manufactured_rest(
    problem: QualifiedExactMixedGasamProblem,
    /,
    *,
    method: NewtonKrylov | None = None,
    termination: NonlinearTermination | None = None,
) -> ManufacturedRestCandidate:
    """Solve the exact zero-load rest patch through Phydrax's nonlinear owner.

    The manufactured solution is ``u=0, p=0``.  It exercises the compiled
    Q2/Q1 or P2/P1 residual, pressure gauge, nonlinear solver, and atomic commit
    without manufacturing an unphysical load or a second FEM implementation.
    """
    if not isinstance(problem, QualifiedExactMixedGasamProblem):
        raise TypeError("problem must be QualifiedExactMixedGasamProblem.")
    initial = problem.prepared.problem.state_space.zeros()
    initial_evaluation = problem.prepared.evaluate(initial)
    method_ = NewtonKrylov() if method is None else method
    termination_ = NonlinearTermination() if termination is None else termination
    if not isinstance(method_, NewtonKrylov):
        raise TypeError("method must be NewtonKrylov or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    result = method_.solve(
        problem.prepared.problem.as_nonlinear_problem(),
        initial,
        termination=termination_,
    )
    final_evaluation = problem.prepared.evaluate(result.state)
    initial_norm = jnp.sqrt(
        sum(jnp.vdot(value, value).real for value in initial_evaluation.residual)
    )
    final_norm = jnp.sqrt(
        sum(jnp.vdot(value, value).real for value in final_evaluation.residual)
    )
    nonlinear_successful = jnp.asarray(result.successful)
    mixed_valid = jnp.asarray(problem.qualification.valid)
    valid = nonlinear_successful & mixed_valid & final_evaluation.valid
    evidence = ManufacturedRestEvidence(
        initial_norm,
        final_norm,
        nonlinear_successful,
        mixed_valid,
        valid,
        not bool(valid),
    )
    return ManufacturedRestCandidate(initial, result.state, result, evidence)


__all__ = [
    "GasamQualificationEvidence",
    "AffineMeshPowerEvidence",
    "affine_mesh_power_evidence",
    "GasamQualificationPlan",
    "ManufacturedRestCandidate",
    "ManufacturedRestCommit",
    "ManufacturedRestEvidence",
    "solve_manufactured_rest",
]
