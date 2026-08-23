#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import bures_squared_distance
from ..operators.quantum import lorentzian_pseudomode
from ._heom import HEOMHierarchy, HEOMProblem, solve_heom
from ._memory_kernel import (
    exponential_memory_qubit_problem,
    MemoryKernelMasterEquation,
    QuantumMemoryKernel,
    solve_memory_kernel,
)
from ._pseudomode import jaynes_cummings_pseudomode_problem, solve_pseudomode


class NonMarkovianComparisonResult(StrictModule):
    pseudomode_states: Array
    heom_states: Array
    memory_states: Array
    pseudomode_heom_distance: Array
    pseudomode_memory_distance: Array
    valid: Array

    def __init__(
        self,
        pseudomode_states: ArrayLike,
        heom_states: ArrayLike,
        memory_states: ArrayLike,
        /,
    ):
        pseudomode = jnp.asarray(pseudomode_states)
        heom = jnp.asarray(heom_states)
        memory = jnp.asarray(memory_states)
        if pseudomode.shape != heom.shape or pseudomode.shape != memory.shape:
            raise ValueError("Non-Markovian solution trajectories must share shape.")
        self.pseudomode_states = pseudomode
        self.heom_states = heom
        self.memory_states = memory
        self.pseudomode_heom_distance = jnp.stack(
            [
                bures_squared_distance(left, right)
                for left, right in zip(pseudomode, heom, strict=True)
            ]
        )
        self.pseudomode_memory_distance = jnp.stack(
            [
                bures_squared_distance(left, right)
                for left, right in zip(pseudomode, memory, strict=True)
            ]
        )
        self.valid = (
            jnp.all(jnp.isfinite(pseudomode))
            & jnp.all(jnp.isfinite(heom))
            & jnp.all(jnp.isfinite(memory))
        )


class SpinBosonComparisonResult(StrictModule):
    heom_states: Array
    memory_states: Array
    bures_distance: Array
    valid: Array

    def __init__(self, heom_states: ArrayLike, memory_states: ArrayLike, /):
        heom = jnp.asarray(heom_states)
        memory = jnp.asarray(memory_states)
        if heom.shape != memory.shape:
            raise ValueError("Spin-boson trajectories must share shape.")
        self.heom_states = heom
        self.memory_states = memory
        self.bures_distance = jnp.stack(
            [
                bures_squared_distance(left, right)
                for left, right in zip(heom, memory, strict=True)
            ]
        )
        self.valid = jnp.all(jnp.isfinite(heom)) & jnp.all(jnp.isfinite(memory))


def lorentzian_qubit_comparison(
    initial_density: ArrayLike,
    /,
    *,
    center_frequency: float = 1.0,
    linewidth: float = 0.5,
    coupling: float = 0.1,
    cutoff: int = 3,
    heom_depth: int = 1,
    step_size: float = 0.01,
    steps: int = 4,
) -> NonMarkovianComparisonResult:
    expansion, mode, _ = lorentzian_pseudomode(
        center_frequency, linewidth, coupling, cutoff=cutoff
    )
    pseudomode_problem = jaynes_cummings_pseudomode_problem(mode, initial_density)
    pseudomode = solve_pseudomode(pseudomode_problem, step_size=step_size, steps=steps)
    sigma_minus = jnp.asarray([[0, 0], [1, 0]], dtype=complex)
    heom_problem = HEOMProblem(
        jnp.zeros((2, 2), dtype=complex),
        sigma_minus + jnp.conj(sigma_minus.T),
        expansion,
        HEOMHierarchy(expansion.rank, heom_depth),
        initial_density,
        problem_id="lorentzian-comparison-heom",
    )
    heom = solve_heom(heom_problem, step_size=step_size, steps=steps)
    memory = solve_memory_kernel(
        exponential_memory_qubit_problem(coupling**2, linewidth, initial_density),
        step_size=step_size,
        steps=steps,
    )
    return NonMarkovianComparisonResult(
        pseudomode.reduced_states, heom.root_states, memory.states
    )


def spin_boson_dephasing_comparison(
    initial_density: ArrayLike,
    /,
    *,
    bias: float = 0.0,
    tunneling: float = 1.0,
    coupling: float = 0.05,
    decay: float = 1.0,
    heom_depth: int = 2,
    step_size: float = 0.01,
    steps: int = 4,
) -> SpinBosonComparisonResult:
    sigma_x = jnp.asarray([[0, 1], [1, 0]], dtype=complex)
    sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=complex)
    hamiltonian = 0.5 * (float(bias) * sigma_z + float(tunneling) * sigma_x)
    expansion = lorentzian_pseudomode(0.0, 2.0 * float(decay), coupling, cutoff=2)[0]
    heom_problem = HEOMProblem(
        hamiltonian,
        sigma_z,
        expansion,
        HEOMHierarchy(expansion.rank, heom_depth),
        initial_density,
        problem_id="spin-boson-dephasing-heom",
    )
    heom = solve_heom(heom_problem, step_size=step_size, steps=steps)

    def kernel(lag, density):
        correlation = expansion(lag)
        return -jnp.real(correlation) * (
            sigma_z @ (sigma_z @ density - density @ sigma_z)
            - (sigma_z @ density - density @ sigma_z) @ sigma_z
        )

    memory_problem = MemoryKernelMasterEquation(
        lambda time, density: -1j * (hamiltonian @ density - density @ hamiltonian),
        QuantumMemoryKernel(
            kernel,
            2,
            memory_horizon=8.0 / float(decay),
            kernel_id="spin-boson-dephasing-kernel",
        ),
        initial_density,
        problem_id="spin-boson-dephasing-memory",
    )
    memory = solve_memory_kernel(memory_problem, step_size=step_size, steps=steps)
    return SpinBosonComparisonResult(heom.root_states, memory.states)


__all__ = [
    "NonMarkovianComparisonResult",
    "SpinBosonComparisonResult",
    "lorentzian_qubit_comparison",
    "spin_boson_dephasing_comparison",
]
