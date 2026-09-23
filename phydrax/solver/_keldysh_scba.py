#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Local matrix optical-phonon Fock SCBA on a commensurate energy grid."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ..nonlinear import FixedPointIteration, FixedPointProblem, NonlinearTermination
from ..operators.quantum import FermionicKeldyshTransportState


class KeldyshSCBAProblem(StrictModule, NonTrainableState):
    energies: Array
    bare_inverse_retarded: Array
    contact_lesser: Array
    contact_greater: Array
    local_coupling: Array
    phonon_bins: int = eqx.field(static=True)
    phonon_occupation: float = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies: ArrayLike,
        bare_inverse_retarded: ArrayLike,
        contact_lesser: ArrayLike,
        contact_greater: ArrayLike,
        local_coupling: ArrayLike,
        /,
        *,
        phonon_bins: int,
        phonon_occupation: float,
    ):
        energy = np.asarray(energies, dtype=np.float64)
        bare = np.asarray(bare_inverse_retarded, dtype=np.complex128)
        lesser = np.asarray(contact_lesser, dtype=np.complex128)
        greater = np.asarray(contact_greater, dtype=np.complex128)
        coupling = np.asarray(local_coupling, dtype=np.float64)
        bins = int(phonon_bins)
        occupation = float(phonon_occupation)
        if (
            energy.ndim != 1
            or energy.size < 8
            or np.any(np.diff(energy) <= 0.0)
            or not np.allclose(np.diff(energy), np.diff(energy)[0])
            or bare.shape != (energy.size, coupling.size, coupling.size)
            or lesser.shape != bare.shape
            or greater.shape != bare.shape
            or np.any(~np.isfinite(bare))
            or np.any(~np.isfinite(lesser))
            or np.any(~np.isfinite(greater))
            or coupling.ndim != 1
            or np.any(~np.isfinite(coupling))
            or bins < 1
            or bins >= energy.size
            or not isfinite(occupation)
            or occupation < 0.0
        ):
            raise ValueError("SCBA energy grid, matrices, coupling, or bath is invalid.")
        self.energies = jnp.asarray(energy)
        self.bare_inverse_retarded = jnp.asarray(bare)
        self.contact_lesser = jnp.asarray(lesser)
        self.contact_greater = jnp.asarray(greater)
        self.local_coupling = jnp.asarray(coupling)
        self.phonon_bins = bins
        self.phonon_occupation = occupation
        self.spacing = float(np.diff(energy)[0])
        self.problem_id = canonical_fingerprint(
            {
                "kind": "keldysh-scba-problem",
                "arrays": array_tree_fingerprint(
                    {
                        "energies": energy,
                        "bare": bare,
                        "lesser": lesser,
                        "greater": greater,
                        "coupling": coupling,
                    }
                ),
                "phonon_bins": bins,
                "phonon_occupation": occupation,
            }
        )


class KeldyshSCBAPolicy(StrictModule, NonTrainableState):
    tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self, *, tolerance: float = 1.0e-8, maximum_steps: int = 100, damping: float = 0.5
    ):
        tolerance_ = float(tolerance)
        steps = int(maximum_steps)
        damping_ = float(damping)
        if (
            not isfinite(tolerance_)
            or tolerance_ <= 0.0
            or steps < 1
            or not 0.0 < damping_ <= 1.0
        ):
            raise ValueError("SCBA tolerance, steps, or damping is invalid.")
        self.tolerance = tolerance_
        self.maximum_steps = steps
        self.damping = damping_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "keldysh-scba-policy",
                "tolerance": tolerance_,
                "steps": steps,
                "damping": damping_,
            }
        )


class KeldyshSCBAResult(StrictModule, NonTrainableState):
    state: FermionicKeldyshTransportState
    retarded_self_energy: Array
    lesser_self_energy: Array
    greater_self_energy: Array
    fixed_point_residual: Array
    collision_balance: Array
    linear_solve_successful: Array
    successful: Array
    problem_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _shift(values: Array, bins: int, /) -> Array:
    if bins > 0:
        return jnp.concatenate((jnp.zeros_like(values[:bins]), values[:-bins]), axis=0)
    amount = -bins
    return jnp.concatenate((values[amount:], jnp.zeros_like(values[:amount])), axis=0)


def _causal_real(gamma: Array, spacing: float, /) -> Array:
    count = gamma.shape[0]
    edges = (jnp.arange(count + 1) - 0.5) * spacing

    def row(index):
        coordinate = index * spacing
        weights = jnp.log(
            jnp.abs((coordinate - edges[:-1]) / (coordinate - edges[1:]))
        ) / (2.0 * jnp.pi)
        return contract("e,ed->d", weights, gamma, backend="jax")

    return jax.lax.map(row, jnp.arange(count))


def _solve_matrix(matrix: Array, /) -> tuple[Array, Array]:
    identity = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    result = solve(
        LinearSystem(DenseLinearOperator(matrix)),
        identity,
        policy=LinearSolvePolicy(DenseLU()),
    )
    return result.value, result.successful


def solve_keldysh_scba(
    problem: KeldyshSCBAProblem,
    policy: KeldyshSCBAPolicy,
    /,
) -> KeldyshSCBAResult:
    if not isinstance(problem, KeldyshSCBAProblem) or not isinstance(
        policy, KeldyshSCBAPolicy
    ):
        raise TypeError("problem and policy have invalid types.")
    coupling_squared = problem.local_coupling**2
    occupation = problem.phonon_occupation
    bins = problem.phonon_bins

    def evaluate(state):
        lesser_diagonal, greater_diagonal = state
        broadening = lesser_diagonal + greater_diagonal
        retarded_diagonal = _causal_real(broadening, problem.spacing) - 0.5j * broadening
        inverse = problem.bare_inverse_retarded - jax.vmap(jnp.diag)(retarded_diagonal)
        green, linear_success = jax.vmap(_solve_matrix)(inverse)
        lesser_source = problem.contact_lesser + jax.vmap(jnp.diag)(lesser_diagonal)
        greater_source = problem.contact_greater + jax.vmap(jnp.diag)(greater_diagonal)
        advanced = jnp.swapaxes(jnp.conj(green), -1, -2)
        lesser_green = green @ lesser_source @ advanced
        greater_green = green @ greater_source @ advanced
        return retarded_diagonal, green, lesser_green, greater_green, linear_success

    def mapping(state, args):
        del args
        _, _, lesser_green, greater_green, _ = evaluate(state)
        lesser_density = jnp.real(jnp.diagonal(lesser_green, axis1=-2, axis2=-1))
        greater_density = jnp.real(jnp.diagonal(greater_green, axis1=-2, axis2=-1))
        lesser = coupling_squared * (
            occupation * _shift(lesser_density, bins)
            + (occupation + 1.0) * _shift(lesser_density, -bins)
        )
        greater = coupling_squared * (
            (occupation + 1.0) * _shift(greater_density, bins)
            + occupation * _shift(greater_density, -bins)
        )
        return jnp.stack((lesser, greater))

    initial = jnp.zeros((2, problem.energies.size, problem.local_coupling.size))
    nonlinear = FixedPointIteration(damping=policy.damping).solve(
        FixedPointProblem(mapping, problem_id=problem.problem_id),
        initial,
        termination=NonlinearTermination(
            absolute_residual=policy.tolerance,
            relative_residual=policy.tolerance,
            maximum_steps=policy.maximum_steps,
        ),
    )
    state_value = nonlinear.state
    retarded_diagonal, green, lesser_green, greater_green, linear_success = evaluate(
        state_value
    )
    fixed_residual = jnp.max(jnp.abs(mapping(state_value, None) - state_value))
    collision = (
        jnp.sum(
            state_value[0] * jnp.real(jnp.diagonal(greater_green, axis1=-2, axis2=-1))
            - state_value[1] * jnp.real(jnp.diagonal(lesser_green, axis1=-2, axis2=-1))
        )
        * problem.spacing
    )
    advanced = jnp.swapaxes(jnp.conj(green), -1, -2)
    spectral = 1.0j * (green - advanced)
    keldysh_state = FermionicKeldyshTransportState(
        lesser_green,
        spectral,
        green,
        advanced,
        problem.energies,
        mode_order_id="matrix-device-order",
        source_id=problem.problem_id,
        particle_continuity_residual=jnp.abs(collision),
    )
    successful = (
        nonlinear.successful
        & jnp.all(linear_success)
        & keldysh_state.valid
        & (fixed_residual <= policy.tolerance)
    )
    return KeldyshSCBAResult(
        keldysh_state,
        jax.vmap(jnp.diag)(retarded_diagonal),
        jax.vmap(jnp.diag)(state_value[0]),
        jax.vmap(jnp.diag)(state_value[1]),
        fixed_residual,
        jnp.abs(collision),
        linear_success,
        successful,
        problem.problem_id,
        canonical_fingerprint(
            {
                "kind": "keldysh-scba-result",
                "problem": problem.problem_id,
                "policy": policy.policy_id,
            }
        ),
    )


__all__ = [
    "KeldyshSCBAProblem",
    "KeldyshSCBAPolicy",
    "KeldyshSCBAResult",
    "solve_keldysh_scba",
]
