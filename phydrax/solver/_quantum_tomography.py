#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
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
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        povm: QuantumPOVM,
        data: QuantumTomographyData,
        initial_density: ArrayLike,
        /,
        *,
        problem_id: str = "quantum-tomography",
    ):
        if not isinstance(povm, QuantumPOVM) or not isinstance(
            data, QuantumTomographyData
        ):
            raise TypeError("povm and data must use quantum tomography contracts.")
        manifold = BuresDensityManifold(povm.dimension)
        density = jnp.asarray(initial_density)
        if not bool(jax.device_get(manifold.contains(density))):
            raise ValueError("initial_density must be faithful and trace one.")
        if data.counts.shape != (povm.outcome_count,):
            raise ValueError("Data outcomes do not match the POVM.")
        self.povm = povm
        self.data = data
        self.initial_density = density
        self.manifold = manifold
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
        self.problem_id = str(problem_id)


class QuantumTomographyArtifact(StrictModule):
    density: Array
    povm_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    schema_version: int = eqx.field(static=True)

    def __init__(
        self,
        density: ArrayLike,
        /,
        *,
        povm_id: str,
        data_id: str,
        problem_id: str,
        schema_version: int = 1,
    ):
        self.density = jnp.asarray(density)
        self.povm_id = str(povm_id)
        self.data_id = str(data_id)
        self.problem_id = str(problem_id)
        self.schema_version = int(schema_version)


def _negative_log_likelihood_cotangent(
    problem: QuantumTomographyProblem, density: Array, /
) -> Array:
    probabilities = problem.povm.probabilities(density)
    safe = jnp.maximum(probabilities, jnp.finfo(probabilities.dtype).tiny)
    return -jnp.einsum("k,kij->ij", problem.data.counts / safe, problem.povm.effects)


def solve_quantum_tomography(
    problem: QuantumTomographyProblem,
    /,
    *,
    policy: QuantumTomographyPolicy | None = None,
) -> QuantumTomographyResult:
    if not isinstance(problem, QuantumTomographyProblem):
        raise TypeError("problem must be a QuantumTomographyProblem.")
    policy_ = QuantumTomographyPolicy() if policy is None else policy
    density = problem.initial_density
    likelihoods = []
    eigenvalues = []
    accepted = []
    converged = False
    previous = tomography_log_likelihood(
        problem.povm, problem.data, density
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
                problem.povm, problem.data, candidate
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
        report = FaithfulDensityReport(density, tolerance=problem.manifold.tolerance)
        eigenvalues.append(report.minimum_eigenvalue)
        accepted.append(did_accept)
        improvement = candidate_likelihood - previous if did_accept else 0.0
        previous = candidate_likelihood if did_accept else previous
        if bool(jax.device_get(jnp.abs(improvement) <= policy_.likelihood_tolerance)):
            converged = True
            break
    from ..metrix import density_fidelity

    return QuantumTomographyResult(
        density,
        jnp.stack(likelihoods) if likelihoods else jnp.zeros((0,)),
        jnp.stack(eigenvalues) if eigenvalues else jnp.zeros((0,)),
        jnp.asarray(accepted),
        density_fidelity(problem.initial_density, density),
        problem.povm.identifiability_rank(),
        converged=converged,
        problem_id=problem.problem_id,
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
    )


__all__ = [
    "QuantumTomographyArtifact",
    "QuantumTomographyPolicy",
    "QuantumTomographyProblem",
    "QuantumTomographyResult",
    "freeze_quantum_tomography",
    "solve_quantum_tomography",
]
