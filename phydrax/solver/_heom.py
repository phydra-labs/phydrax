#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from itertools import product

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..operators.quantum import BathCorrelationExpansion


class HEOMHierarchy(StrictModule):
    multi_indices: Array
    levels: Array
    upward: Array
    downward: Array
    depth: int = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    auxiliary_count: int = eqx.field(static=True)

    def __init__(self, term_count: int, depth: int, /):
        terms = int(term_count)
        depth_ = int(depth)
        if terms < 1 or depth_ < 0:
            raise ValueError("HEOM term count/depth are invalid.")
        indices = tuple(
            values
            for values in product(range(depth_ + 1), repeat=terms)
            if sum(values) <= depth_
        )
        lookup = {values: index for index, values in enumerate(indices)}
        upward = []
        downward = []
        for values in indices:
            up_row = []
            down_row = []
            for term in range(terms):
                up = list(values)
                up[term] += 1
                down = list(values)
                down[term] -= 1
                up_row.append(lookup.get(tuple(up), -1))
                down_row.append(lookup.get(tuple(down), -1) if down[term] >= 0 else -1)
            upward.append(up_row)
            downward.append(down_row)
        self.multi_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.levels = jnp.sum(self.multi_indices, axis=-1)
        self.upward = jnp.asarray(upward, dtype=jnp.int32)
        self.downward = jnp.asarray(downward, dtype=jnp.int32)
        self.depth = depth_
        self.term_count = terms
        self.auxiliary_count = len(indices)


class HEOMProblem(StrictModule):
    hamiltonian: Array
    coupling_operator: Array
    expansion: BathCorrelationExpansion
    hierarchy: HEOMHierarchy
    initial_state: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: ArrayLike,
        coupling_operator: ArrayLike,
        expansion: BathCorrelationExpansion,
        hierarchy: HEOMHierarchy,
        initial_density: ArrayLike,
        /,
        *,
        problem_id: str = "heom",
    ):
        hamiltonian_ = jnp.asarray(hamiltonian)
        coupling = jnp.asarray(coupling_operator, dtype=hamiltonian_.dtype)
        density = jnp.asarray(initial_density, dtype=hamiltonian_.dtype)
        if hamiltonian_.ndim != 2 or hamiltonian_.shape[0] != hamiltonian_.shape[1]:
            raise ValueError("HEOM Hamiltonian must be square.")
        if coupling.shape != hamiltonian_.shape or density.shape != hamiltonian_.shape:
            raise ValueError("HEOM operator/state shapes are incompatible.")
        if hierarchy.term_count != expansion.rank:
            raise ValueError("HEOM hierarchy and bath expansion ranks differ.")
        auxiliaries = (
            jnp.zeros((hierarchy.auxiliary_count,) + density.shape, dtype=density.dtype)
            .at[0]
            .set(density)
        )
        self.hamiltonian = hamiltonian_
        self.coupling_operator = coupling
        self.expansion = expansion
        self.hierarchy = hierarchy
        self.initial_state = auxiliaries
        self.problem_id = str(problem_id)

    def rhs(self, auxiliaries: ArrayLike, /) -> Array:
        values = jnp.asarray(auxiliaries)
        result = jnp.zeros_like(values)
        coupling = self.coupling_operator
        for auxiliary in range(self.hierarchy.auxiliary_count):
            rho = values[auxiliary]
            derivative = -1j * (self.hamiltonian @ rho - rho @ self.hamiltonian)
            occupation = self.hierarchy.multi_indices[auxiliary]
            derivative = derivative - jnp.sum(occupation * self.expansion.exponents) * rho
            for term in range(self.hierarchy.term_count):
                up = self.hierarchy.upward[auxiliary, term]
                down = self.hierarchy.downward[auxiliary, term]
                upper = values[jnp.maximum(up, 0)]
                upper_term = -1j * (coupling @ upper - upper @ coupling)
                derivative = derivative + jnp.where(up >= 0, upper_term, 0.0)
                lower = values[jnp.maximum(down, 0)]
                coefficient = self.expansion.coefficients[term]
                lower_term = (
                    -1j
                    * occupation[term]
                    * (
                        coefficient * coupling @ lower
                        - jnp.conj(coefficient) * lower @ coupling
                    )
                )
                derivative = derivative + jnp.where(down >= 0, lower_term, 0.0)
            result = result.at[auxiliary].set(derivative)
        return result


class HEOMSolution(StrictModule):
    root_states: Array
    final_auxiliaries: Array
    times: Array
    maximum_auxiliary_norm_by_level: Array
    valid: Array
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: HEOMProblem,
        root_states: ArrayLike,
        final_auxiliaries: ArrayLike,
        times: ArrayLike,
        /,
    ):
        roots = jnp.asarray(root_states)
        auxiliaries = jnp.asarray(final_auxiliaries)
        self.root_states = roots
        self.final_auxiliaries = auxiliaries
        self.times = jnp.asarray(times)
        norms = jnp.linalg.norm(auxiliaries, axis=(-2, -1))
        self.maximum_auxiliary_norm_by_level = jnp.stack(
            [
                jnp.max(jnp.where(problem.hierarchy.levels == level, norms, 0.0))
                for level in range(problem.hierarchy.depth + 1)
            ]
        )
        trace = jnp.trace(roots, axis1=-2, axis2=-1)
        hermiticity = jnp.max(
            jnp.abs(roots - jnp.swapaxes(jnp.conj(roots), -1, -2)), axis=(-2, -1)
        )
        self.valid = (
            jnp.all(jnp.isfinite(roots))
            & jnp.all(jnp.abs(trace - 1.0) <= 1e-6)
            & jnp.all(hermiticity <= 1e-6)
        )
        self.problem_id = problem.problem_id


def solve_heom(
    problem: HEOMProblem,
    /,
    *,
    step_size: ArrayLike,
    steps: int,
) -> HEOMSolution:
    step = jnp.asarray(step_size, dtype=float).reshape(())
    count = int(steps)
    if count < 0 or float(step) <= 0.0:
        raise ValueError("HEOM steps and step_size must be positive.")

    def advance(state, _):
        k1 = problem.rhs(state)
        k2 = problem.rhs(state + 0.5 * step * k1)
        k3 = problem.rhs(state + 0.5 * step * k2)
        k4 = problem.rhs(state + step * k3)
        next_state = state + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return next_state, next_state[0]

    final, roots = jax.lax.scan(advance, problem.initial_state, xs=None, length=count)
    roots = jnp.concatenate((problem.initial_state[0][None, ...], roots), axis=0)
    return HEOMSolution(problem, roots, final, step * jnp.arange(count + 1))


def drude_lorentz_qubit_heom(
    coupling_strength: float,
    decay_rate: float,
    initial_density: ArrayLike,
    /,
    *,
    depth: int = 2,
) -> HEOMProblem:
    expansion = BathCorrelationExpansion(
        jnp.asarray([float(coupling_strength)], dtype=complex),
        jnp.asarray([float(decay_rate)], dtype=complex),
        expansion_id="drude-lorentz-one-term",
    )
    hierarchy = HEOMHierarchy(1, depth)
    sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=complex)
    return HEOMProblem(
        jnp.zeros((2, 2), dtype=complex),
        sigma_z,
        expansion,
        hierarchy,
        initial_density,
        problem_id="drude-lorentz-qubit-heom",
    )


def thermal_drude_lorentz_qubit_heom(
    coupling_strength: float,
    decay_rate: float,
    temperature: float,
    initial_density: ArrayLike,
    /,
    *,
    depth: int = 2,
) -> HEOMProblem:
    """One-term finite-temperature Drude approximation with explicit truncation."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    thermal_factor = 1.0 / jnp.tanh(float(decay_rate) / (2.0 * float(temperature)))
    return drude_lorentz_qubit_heom(
        float(coupling_strength) * float(thermal_factor),
        decay_rate,
        initial_density,
        depth=depth,
    )


__all__ = [
    "HEOMHierarchy",
    "HEOMProblem",
    "HEOMSolution",
    "drude_lorentz_qubit_heom",
    "solve_heom",
    "thermal_drude_lorentz_qubit_heom",
]
