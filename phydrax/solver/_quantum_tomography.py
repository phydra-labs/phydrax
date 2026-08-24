#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import HermitianPrecisionPolicy
from ..metrix import BuresDensityManifold, FaithfulDensityReport
from ..uq import (
    QuantumPOVM,
    QuantumTomographyData,
    tomography_log_likelihood,
)


class QuantumTomographyProblem(StrictModule):
    povm: QuantumPOVM
    data: QuantumTomographyData
    initial_density: Array
    manifold: BuresDensityManifold
    precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        povm: QuantumPOVM,
        data: QuantumTomographyData,
        initial_density: ArrayLike,
        /,
        *,
        problem_id: str = "quantum-tomography",
        precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
    ):
        if not isinstance(povm, QuantumPOVM) or not isinstance(
            data, QuantumTomographyData
        ):
            raise TypeError("povm and data must use quantum tomography contracts.")
        precision_ = povm.precision if precision is None else precision
        hermitian_ = (
            povm.hermitian_precision
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be a HermitianPrecisionPolicy or None."
            )
        if precision_.policy_id != povm.precision.policy_id:
            raise ValueError("Tomography problem precision must match the POVM.")
        if hermitian_.policy_id != povm.hermitian_precision.policy_id:
            raise ValueError("Tomography Hermitian precision must match the POVM.")
        manifold = BuresDensityManifold(
            povm.dimension,
            precision=precision_,
            hermitian_precision=hermitian_,
        )
        density = jnp.asarray(initial_density)
        precision_.validate_coordinates(density)
        if not bool(jax.device_get(manifold.contains(density))):
            raise ValueError("initial_density must be faithful and trace one.")
        if data.counts.shape != (povm.outcome_count,):
            raise ValueError("Data outcomes do not match the POVM.")
        self.povm = povm
        self.data = data
        self.initial_density = density
        self.manifold = manifold
        self.precision = precision_
        self.hermitian_precision = hermitian_
        self.problem_id = str(problem_id)


class QuantumTomographyPolicy(StrictModule):
    iterations: int
    learning_rate: float
    maximum_backtracks: int
    contraction: float
    likelihood_tolerance: float

    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.1,
        maximum_backtracks: int = 8,
        contraction: float = 0.5,
        likelihood_tolerance: float = 1e-9,
    ):
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.maximum_backtracks = int(maximum_backtracks)
        self.contraction = float(contraction)
        self.likelihood_tolerance = float(likelihood_tolerance)


class QuantumTomographyResult(StrictModule):
    density: Array
    log_likelihood_history: Array
    minimum_eigenvalue_history: Array
    accepted_history: Array
    fidelity_to_initial: Array
    identifiable_rank: Array
    valid: Array
    converged: Array
    precision_evidence: PrecisionEvidenceEnvelope
    hermitian_precision_evidence: PrecisionEvidenceEnvelope
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        density: ArrayLike,
        log_likelihood_history: ArrayLike,
        minimum_eigenvalue_history: ArrayLike,
        accepted_history: ArrayLike,
        fidelity_to_initial: ArrayLike,
        identifiable_rank: ArrayLike,
        /,
        *,
        converged: ArrayLike,
        problem_id: str,
        precision_evidence: PrecisionEvidenceEnvelope,
        hermitian_precision_evidence: PrecisionEvidenceEnvelope,
    ):
        self.density = jnp.asarray(density)
        self.log_likelihood_history = jnp.asarray(log_likelihood_history)
        self.minimum_eigenvalue_history = jnp.asarray(minimum_eigenvalue_history)
        self.accepted_history = jnp.asarray(accepted_history, dtype=bool)
        self.fidelity_to_initial = jnp.asarray(fidelity_to_initial)
        self.identifiable_rank = jnp.asarray(identifiable_rank)
        self.valid = (
            jnp.all(jnp.isfinite(self.density))
            & jnp.all(jnp.isfinite(self.log_likelihood_history))
            & jnp.all(self.minimum_eigenvalue_history > 0.0)
        )
        self.converged = jnp.asarray(converged, dtype=bool)
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be PrecisionEvidenceEnvelope.")
        if not isinstance(hermitian_precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError(
                "hermitian_precision_evidence must be PrecisionEvidenceEnvelope."
            )
        self.precision_evidence = precision_evidence
        self.hermitian_precision_evidence = hermitian_precision_evidence
        self.problem_id = str(problem_id)


class QuantumTomographyArtifact(StrictModule):
    density: Array
    povm_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    schema_version: int = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope
    hermitian_precision_evidence: PrecisionEvidenceEnvelope
    precision_policy_id: str = eqx.field(static=True)
    hermitian_precision_policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        density: ArrayLike,
        /,
        *,
        povm_id: str,
        data_id: str,
        problem_id: str,
        precision_evidence: PrecisionEvidenceEnvelope,
        hermitian_precision_evidence: PrecisionEvidenceEnvelope,
        precision_policy_id: str,
        hermitian_precision_policy_id: str,
        schema_version: int = 2,
    ):
        self.density = jnp.asarray(density)
        self.povm_id = str(povm_id)
        self.data_id = str(data_id)
        self.problem_id = str(problem_id)
        self.schema_version = int(schema_version)
        self.precision_evidence = precision_evidence
        self.hermitian_precision_evidence = hermitian_precision_evidence
        self.precision_policy_id = str(precision_policy_id)
        self.hermitian_precision_policy_id = str(hermitian_precision_policy_id)


def _negative_log_likelihood_cotangent(
    problem: QuantumTomographyProblem, density: Array, /
) -> Array:
    probabilities = problem.precision.accumulation(problem.povm.probabilities(density))
    safe = jnp.maximum(probabilities, jnp.finfo(probabilities.dtype).tiny)
    return -oe.contract(
        "k,kij->ij",
        problem.precision.accumulation(problem.data.counts / safe),
        problem.precision.accumulation(problem.povm.effects),
    )


def solve_quantum_tomography(
    problem: QuantumTomographyProblem,
    /,
    *,
    policy: QuantumTomographyPolicy | None = None,
) -> QuantumTomographyResult:
    if not isinstance(problem, QuantumTomographyProblem):
        raise TypeError("problem must be a QuantumTomographyProblem.")
    policy_ = QuantumTomographyPolicy() if policy is None else policy
    density = problem.precision.output(problem.initial_density)
    likelihoods = []
    eigenvalues = []
    accepted = []
    converged = False
    previous = tomography_log_likelihood(
        problem.povm,
        problem.data,
        density,
        precision=problem.precision,
    ).log_likelihood
    for _ in range(policy_.iterations):
        cotangent = _negative_log_likelihood_cotangent(problem, density)
        direction = -problem.manifold.egrad_to_rgrad(density, cotangent)
        step = policy_.learning_rate
        candidate = density
        candidate_likelihood = previous
        did_accept = False
        for _ in range(policy_.maximum_backtracks + 1):
            candidate = problem.manifold.retract(density, step * direction)
            candidate_result = tomography_log_likelihood(
                problem.povm,
                problem.data,
                candidate,
                precision=problem.precision,
            )
            if bool(
                jax.device_get(
                    candidate_result.valid
                    & (candidate_result.log_likelihood >= previous)
                    & problem.manifold.contains(candidate)
                )
            ):
                candidate_likelihood = candidate_result.log_likelihood
                did_accept = True
                break
            step *= policy_.contraction
        if did_accept:
            density = candidate
        likelihoods.append(candidate_likelihood if did_accept else previous)
        report = FaithfulDensityReport(
            density,
            tolerance=problem.manifold.tolerance,
            precision=problem.hermitian_precision,
        )
        eigenvalues.append(report.minimum_eigenvalue)
        accepted.append(did_accept)
        improvement = (
            problem.precision.decision(candidate_likelihood - previous)
            if did_accept
            else problem.precision.decision(0.0)
        )
        previous = candidate_likelihood if did_accept else previous
        if bool(
            jax.device_get(
                jnp.abs(improvement)
                <= problem.precision.decision(policy_.likelihood_tolerance)
            )
        ):
            converged = True
            break
    from ..metrix import density_fidelity

    return QuantumTomographyResult(
        problem.precision.output(density),
        jnp.stack(likelihoods) if likelihoods else jnp.zeros((0,)),
        jnp.stack(eigenvalues) if eigenvalues else jnp.zeros((0,)),
        jnp.asarray(accepted),
        density_fidelity(problem.initial_density, density),
        problem.povm.identifiability_rank(),
        converged=converged,
        problem_id=problem.problem_id,
        precision_evidence=problem.precision.evidence_for(problem.initial_density),
        hermitian_precision_evidence=problem.hermitian_precision.evidence_for(
            problem.initial_density
        ),
    )


def freeze_quantum_tomography(
    result: QuantumTomographyResult,
    problem: QuantumTomographyProblem,
    /,
) -> QuantumTomographyArtifact:
    return QuantumTomographyArtifact(
        result.density,
        povm_id=problem.povm.povm_id,
        data_id=problem.data.data_id,
        problem_id=problem.problem_id,
        precision_evidence=result.precision_evidence,
        hermitian_precision_evidence=result.hermitian_precision_evidence,
        precision_policy_id=problem.precision.policy_id,
        hermitian_precision_policy_id=problem.hermitian_precision.policy_id,
        schema_version=2,
    )


__all__ = [
    "QuantumTomographyArtifact",
    "QuantumTomographyPolicy",
    "QuantumTomographyProblem",
    "QuantumTomographyResult",
    "freeze_quantum_tomography",
    "solve_quantum_tomography",
]
