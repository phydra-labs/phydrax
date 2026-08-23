#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..operators.quantum import Pseudomode
from ..operators.quantum._fock import jaynes_cummings_hamiltonian
from ._lindblad import LindbladProblem, LindbladSolution, solve_lindblad


class PseudomodeEmbeddingProblem(StrictModule):
    mode: Pseudomode
    lindblad_problem: LindbladProblem
    system_dimension: int = eqx.field(static=True)
    embedding_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: Pseudomode,
        lindblad_problem: LindbladProblem,
        /,
        *,
        system_dimension: int,
        embedding_id: str,
    ):
        self.mode = mode
        self.lindblad_problem = lindblad_problem
        self.system_dimension = int(system_dimension)
        self.embedding_id = str(embedding_id)

    def reduced_system_density(self, enlarged_density: ArrayLike, /) -> Array:
        density = jnp.asarray(enlarged_density)
        cutoff = self.mode.cutoff
        system = self.system_dimension
        expected = cutoff * system
        if density.shape != (expected, expected):
            raise ValueError("Enlarged density shape does not match the embedding.")
        tensor = density.reshape((cutoff, system, cutoff, system))
        return jnp.trace(tensor, axis1=0, axis2=2)


class PseudomodeSolution(StrictModule):
    enlarged: LindbladSolution
    reduced_states: Array
    valid: Array
    embedding_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: PseudomodeEmbeddingProblem,
        enlarged: LindbladSolution,
        /,
    ):
        self.enlarged = enlarged
        self.reduced_states = jnp.stack(
            [problem.reduced_system_density(state) for state in enlarged.states]
        )
        traces = jnp.trace(self.reduced_states, axis1=-2, axis2=-1)
        minimum = jnp.min(jnp.linalg.eigvalsh(self.reduced_states), axis=-1)
        self.valid = (
            enlarged.valid
            & jnp.all(jnp.abs(traces - 1.0) <= 1e-8)
            & jnp.all(minimum >= -1e-8)
        )
        self.embedding_id = problem.embedding_id


def jaynes_cummings_pseudomode_problem(
    mode: Pseudomode,
    initial_system_density: ArrayLike,
    /,
) -> PseudomodeEmbeddingProblem:
    cavity, hamiltonian = jaynes_cummings_hamiltonian(
        mode.cutoff,
        mode.frequency,
        mode.frequency,
        abs(mode.coupling),
    )
    initial_mode = jnp.zeros((mode.cutoff, mode.cutoff), dtype=complex).at[0, 0].set(1.0)
    system_density = jnp.asarray(initial_system_density)
    initial = jnp.kron(initial_mode, system_density)
    jump = jnp.sqrt(mode.damping) * jnp.kron(cavity.annihilation_matrix(0), jnp.eye(2))
    lindblad = LindbladProblem(
        hamiltonian,
        jump[None, ...],
        initial,
        problem_id="jaynes-cummings-pseudomode",
    )
    return PseudomodeEmbeddingProblem(
        mode,
        lindblad,
        system_dimension=2,
        embedding_id="jaynes-cummings-pseudomode",
    )


def solve_pseudomode(
    problem: PseudomodeEmbeddingProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> PseudomodeSolution:
    enlarged = solve_lindblad(problem.lindblad_problem, step_size=step_size, steps=steps)
    return PseudomodeSolution(problem, enlarged)


__all__ = [
    "PseudomodeEmbeddingProblem",
    "PseudomodeSolution",
    "jaynes_cummings_pseudomode_problem",
    "solve_pseudomode",
]
