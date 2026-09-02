#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Target-generic fixed-capacity Hamiltonian Monte Carlo kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from blackjax.mcmc import integrators as blackjax_integrators, nuts as blackjax_nuts
from jaxtyping import Array, ArrayLike, Key

from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    HermitianSpectrum,
    OperatorProperties,
    PreparedFactorization,
)
from ._addressing import derive_key, SampleAddress


_MOMENTUM_ADDRESS = SampleAddress(
    "markov", "hamiltonian", target="momentum", role="transition"
)
_NUTS_ADDRESS = SampleAddress(
    "markov",
    "hamiltonian",
    algorithm_version=2,
    target="nuts-tree",
    role="transition",
)
_ACCEPT_ADDRESS = SampleAddress(
    "markov", "hamiltonian", target="acceptance", role="transition"
)


class HamiltonianAdaptationPlan(StrictModule):
    """Explicit finite warmup epoch; production always freezes its result."""

    warmup_steps: int = eqx.field(static=True)
    target_acceptance: float = eqx.field(static=True)
    adaptation_rate: float = eqx.field(static=True)
    minimum_step_size: float = eqx.field(static=True)
    maximum_step_size: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        warmup_steps: int,
        target_acceptance: float = 0.8,
        adaptation_rate: float = 0.05,
        minimum_step_size: float = 1e-5,
        maximum_step_size: float = 1.0,
    ):
        warmup = int(warmup_steps)
        target, rate = float(target_acceptance), float(adaptation_rate)
        lower, upper = float(minimum_step_size), float(maximum_step_size)
        if warmup < 0 or not 0.0 < target < 1.0 or rate <= 0.0:
            raise ValueError("Invalid Hamiltonian adaptation warmup/target/rate.")
        if not 0.0 < lower < upper or not all(
            np.isfinite(v) for v in (rate, lower, upper)
        ):
            raise ValueError("Step-size adaptation bounds must be finite and increasing.")
        self.warmup_steps = warmup
        self.target_acceptance = target
        self.adaptation_rate = rate
        self.minimum_step_size = lower
        self.maximum_step_size = upper


class PreparedHamiltonianKernel(StrictModule):
    log_target: Callable[[Array], Array] = eqx.field(static=True)
    mass_matrix: Array
    mass_spectrum: HermitianSpectrum
    mass_factorization: PreparedFactorization
    step_size: Array
    maximum_leapfrog_steps: int = eqx.field(static=True)
    maximum_tree_depth: int = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    method: Literal["hmc", "nuts"] = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    valid: Array


class HamiltonianChainState(StrictModule):
    position: Array
    log_target: Array
    gradient: Array
    step_index: Array
    valid: Array


class HamiltonianSampleResult(StrictModule):
    samples: Array
    log_target: Array
    accepted: Array
    acceptance_probability: Array
    divergent: Array
    maximum_depth_reached: Array
    nonfinite_gradient: Array
    leapfrog_steps: Array
    final_state: HamiltonianChainState
    root_key: Array
    frozen_step_size: Array
    target_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class HamiltonianAdaptationResult(StrictModule):
    kernel: PreparedHamiltonianKernel
    final_state: HamiltonianChainState
    step_size_history: Array
    acceptance_history: Array
    valid: Array
    frozen: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def prepare_hamiltonian_kernel(
    log_target: Callable[[Array], Array],
    mass_matrix: ArrayLike,
    /,
    *,
    step_size: float,
    method: Literal["hmc", "nuts"] = "hmc",
    leapfrog_steps: int = 8,
    maximum_tree_depth: int = 6,
    divergence_threshold: float = 1000.0,
    target_id: str,
) -> PreparedHamiltonianKernel:
    if not callable(log_target):
        raise TypeError("log_target must be callable.")
    mass = jnp.asarray(mass_matrix)
    if mass.ndim != 2 or mass.shape[0] != mass.shape[1] or mass.shape[0] < 1:
        raise ValueError("mass_matrix must be a nonempty square matrix.")
    if jnp.iscomplexobj(mass):
        raise TypeError("Hamiltonian mass matrices must be real.")
    size, steps, depth = float(step_size), int(leapfrog_steps), int(maximum_tree_depth)
    threshold = float(divergence_threshold)
    if method not in ("hmc", "nuts") or steps <= 0 or depth <= 0:
        raise ValueError("method/capacities are invalid.")
    if (
        not np.isfinite(size)
        or size <= 0.0
        or not np.isfinite(threshold)
        or threshold <= 0.0
    ):
        raise ValueError(
            "step_size and divergence_threshold must be finite and positive."
        )
    if not isinstance(target_id, str) or not target_id:
        raise ValueError("target_id must be nonempty.")
    spectrum = HermitianSpectrum(mass)
    valid = (
        spectrum.valid & (spectrum.minimum_eigenvalue > 0.0) & jnp.all(jnp.isfinite(mass))
    )
    symmetric_mass = 0.5 * (mass + mass.T)
    safe_mass = jnp.where(valid, symmetric_mass, jnp.eye(mass.shape[0], dtype=mass.dtype))
    factorization = factorize(
        DenseLinearOperator(
            safe_mass,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={"positive_definite": "verified"},
            ),
        ),
        FactorizationPolicy("cholesky"),
    )
    return PreparedHamiltonianKernel(
        log_target=log_target,
        mass_matrix=mass,
        mass_spectrum=spectrum,
        mass_factorization=factorization,
        step_size=jnp.asarray(size, dtype=mass.dtype),
        maximum_leapfrog_steps=steps,
        maximum_tree_depth=depth,
        divergence_threshold=threshold,
        method=method,
        target_id=target_id,
        valid=valid,
    )


def initialize_hamiltonian_state(
    kernel: PreparedHamiltonianKernel, initial_positions: ArrayLike, /
) -> HamiltonianChainState:
    if not isinstance(kernel, PreparedHamiltonianKernel):
        raise TypeError("kernel must be PreparedHamiltonianKernel.")
    positions = jnp.asarray(initial_positions, dtype=kernel.mass_matrix.dtype)
    dimension = int(kernel.mass_matrix.shape[0])
    if positions.ndim != 2 or positions.shape[1] != dimension or positions.shape[0] < 1:
        raise ValueError("initial_positions must have shape (chains, mass_dimension).")
    values, gradients = jax.vmap(jax.value_and_grad(kernel.log_target))(positions)
    if values.shape != (positions.shape[0],) or jnp.iscomplexobj(values):
        raise ValueError("log_target must return one real scalar per position.")
    valid = (
        kernel.valid
        & jnp.all(jnp.isfinite(positions), axis=-1)
        & jnp.isfinite(values)
        & jnp.all(jnp.isfinite(gradients), axis=-1)
    )
    return HamiltonianChainState(
        position=positions,
        log_target=values,
        gradient=gradients,
        step_index=jnp.asarray(0, dtype=jnp.uint32),
        valid=valid,
    )


def _mass_solve(kernel: PreparedHamiltonianKernel, momentum: Array, /) -> Array:
    return jnp.asarray(kernel.mass_factorization.solve(momentum).value)


def _kinetic(kernel: PreparedHamiltonianKernel, momentum: Array, /) -> Array:
    return 0.5 * jnp.vdot(momentum, _mass_solve(kernel, momentum)).real


def _sample_momentum(kernel: PreparedHamiltonianKernel, key: Key[Array, ""], /) -> Array:
    normal = jr.normal(
        key, (kernel.mass_matrix.shape[0],), dtype=kernel.mass_matrix.dtype
    )
    return kernel.mass_spectrum.eigenvectors @ (
        jnp.sqrt(kernel.mass_spectrum.eigenvalues) * normal
    )


def _trajectory(
    kernel: PreparedHamiltonianKernel,
    position: Array,
    gradient: Array,
    momentum: Array,
    step_count: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    value_and_gradient = jax.value_and_grad(kernel.log_target)
    maximum = kernel.maximum_leapfrog_steps

    def body(carry, index):
        q, p, g, active, used, nonfinite = carry
        do_step = active & (index < step_count)
        half = p + 0.5 * kernel.step_size * g
        proposed_q = q + kernel.step_size * _mass_solve(kernel, half)
        proposed_value, proposed_g = value_and_gradient(proposed_q)
        proposed_p = half + 0.5 * kernel.step_size * proposed_g
        finite = (
            jnp.all(jnp.isfinite(proposed_q))
            & jnp.isfinite(proposed_value)
            & jnp.all(jnp.isfinite(proposed_g))
            & jnp.all(jnp.isfinite(proposed_p))
        )
        commit = do_step & finite
        return (
            jnp.where(commit, proposed_q, q),
            jnp.where(commit, proposed_p, p),
            jnp.where(commit, proposed_g, g),
            commit,
            used + do_step.astype(jnp.int32),
            nonfinite | (do_step & ~finite),
        ), proposed_value

    (q, p, g, _, used, nonfinite), _ = jax.lax.scan(
        body,
        (
            position,
            momentum,
            gradient,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        ),
        jnp.arange(maximum, dtype=jnp.int32),
    )
    return q, -p, g, kernel.log_target(q), jnp.stack((used, nonfinite.astype(jnp.int32)))


def _one_hmc_transition(kernel, position, log_target, gradient, key, chain, step_index):
    momentum_key = derive_key(key, _MOMENTUM_ADDRESS, chain, step_index)
    accept_key = derive_key(key, _ACCEPT_ADDRESS, chain, step_index)
    momentum = _sample_momentum(kernel, momentum_key)
    step_count = jnp.asarray(kernel.maximum_leapfrog_steps, dtype=jnp.int32)
    proposed_q, proposed_p, proposed_g, proposed_value, evidence = _trajectory(
        kernel, position, gradient, momentum, step_count
    )
    used, nonfinite_int = evidence
    energy_error = (-proposed_value + _kinetic(kernel, proposed_p)) - (
        -log_target + _kinetic(kernel, momentum)
    )
    divergent = (
        nonfinite_int.astype(bool)
        | (~jnp.isfinite(energy_error))
        | (jnp.abs(energy_error) > kernel.divergence_threshold)
    )
    log_accept = jnp.minimum(-energy_error, 0.0)
    accepted = ~divergent & (jnp.log(jr.uniform(accept_key)) < log_accept)
    return (
        jnp.where(accepted, proposed_q, position),
        jnp.where(accepted, proposed_value, log_target),
        jnp.where(accepted, proposed_g, gradient),
        accepted,
        jnp.where(~divergent & jnp.isfinite(log_accept), jnp.exp(log_accept), 0.0),
        divergent,
        jnp.asarray(False),
        nonfinite_int.astype(bool),
        used,
    )


def _one_nuts_transition(kernel, position, log_target, gradient, key, chain, step_index):
    momentum_key = derive_key(key, _MOMENTUM_ADDRESS, chain, step_index)
    transition_key = derive_key(key, _NUTS_ADDRESS, chain, step_index)

    def finite_log_target(candidate):
        value = kernel.log_target(candidate)
        return jnp.where(jnp.isfinite(value), value, -jnp.inf)

    def kinetic_energy(momentum, position=None):
        del position
        return _kinetic(kernel, momentum)

    def check_turning(
        momentum_left,
        momentum_right,
        momentum_sum,
        position_left=None,
        position_right=None,
    ):
        del position_left, position_right
        velocity_left = _mass_solve(kernel, momentum_left)
        velocity_right = _mass_solve(kernel, momentum_right)
        rho = momentum_sum - 0.5 * (momentum_left + momentum_right)
        return (jnp.vdot(velocity_left, rho).real <= 0.0) | (
            jnp.vdot(velocity_right, rho).real <= 0.0
        )

    integrator = blackjax_integrators.velocity_verlet(finite_log_target, kinetic_energy)
    transition = blackjax_nuts.iterative_nuts_proposal(
        integrator,
        kinetic_energy,
        check_turning,
        kernel.maximum_tree_depth,
        kernel.divergence_threshold,
    )
    momentum = _sample_momentum(kernel, momentum_key)
    initial = blackjax_integrators.IntegratorState(
        position, momentum, log_target, gradient
    )
    proposal, info = transition(transition_key, initial, kernel.step_size)
    proposal_finite = (
        jnp.all(jnp.isfinite(proposal.position))
        & jnp.isfinite(proposal.logdensity)
        & jnp.all(jnp.isfinite(proposal.logdensity_grad))
    )
    endpoint_gradients_finite = jnp.all(
        jnp.isfinite(info.trajectory_leftmost_state.logdensity_grad)
    ) & jnp.all(jnp.isfinite(info.trajectory_rightmost_state.logdensity_grad))
    nonfinite_gradient = ~endpoint_gradients_finite
    divergent = info.is_divergent | ~proposal_finite | nonfinite_gradient
    proposal_usable = proposal_finite & ~divergent
    selected_position = jnp.where(proposal_usable, proposal.position, position)
    selected_value = jnp.where(proposal_usable, proposal.logdensity, log_target)
    selected_gradient = jnp.where(proposal_usable, proposal.logdensity_grad, gradient)
    accepted = proposal_usable & jnp.any(selected_position != position)
    acceptance_probability = jnp.where(
        ~divergent & jnp.isfinite(info.acceptance_rate),
        info.acceptance_rate,
        0.0,
    )
    maximum_depth_reached = info.num_integration_steps >= (
        2**kernel.maximum_tree_depth - 1
    )
    return (
        selected_position,
        selected_value,
        selected_gradient,
        accepted,
        acceptance_probability,
        divergent,
        maximum_depth_reached,
        nonfinite_gradient,
        info.num_integration_steps.astype(jnp.int32),
    )


def _one_transition(
    kernel, position, log_target, gradient, state_valid, key, chain, step_index
):
    result = (
        _one_nuts_transition(
            kernel, position, log_target, gradient, key, chain, step_index
        )
        if kernel.method == "nuts"
        else _one_hmc_transition(
            kernel, position, log_target, gradient, key, chain, step_index
        )
    )
    initial_finite = (
        state_valid
        & jnp.all(jnp.isfinite(position))
        & jnp.isfinite(log_target)
        & jnp.all(jnp.isfinite(gradient))
    )
    return (
        jnp.where(initial_finite, result[0], position),
        jnp.where(initial_finite, result[1], log_target),
        jnp.where(initial_finite, result[2], gradient),
        initial_finite & result[3],
        jnp.where(initial_finite, result[4], 0.0),
        ~initial_finite | result[5],
        initial_finite & result[6],
        ~jnp.all(jnp.isfinite(gradient)) | result[7],
        jnp.where(initial_finite, result[8], 0),
    )


def sample_hamiltonian(
    kernel: PreparedHamiltonianKernel,
    state: HamiltonianChainState,
    /,
    *,
    key: Key[Array, ""],
    num_draws: int,
) -> HamiltonianSampleResult:
    if not isinstance(kernel, PreparedHamiltonianKernel) or not isinstance(
        state, HamiltonianChainState
    ):
        raise TypeError("kernel/state types are invalid.")
    draws = int(num_draws)
    if draws <= 0:
        raise ValueError("num_draws must be positive.")
    chain_indices = jnp.arange(state.position.shape[0], dtype=jnp.uint32)

    def draw(carry, _):
        positions, values, gradients, valid, index = carry
        result = jax.vmap(
            lambda q, value, gradient, state_valid, chain: _one_transition(
                kernel, q, value, gradient, state_valid, key, chain, index
            )
        )(positions, values, gradients, valid, chain_indices)
        next_positions, next_values, next_gradients = result[:3]
        return (next_positions, next_values, next_gradients, valid, index + 1), (
            next_positions,
            next_values,
        ) + result[3:]

    (positions, values, gradients, valid, index), outputs = jax.lax.scan(
        draw,
        (
            state.position,
            state.log_target,
            state.gradient,
            state.valid,
            state.step_index,
        ),
        None,
        length=draws,
    )
    samples, log_values, accepted, probabilities, divergent, depth, nonfinite, steps = (
        jnp.swapaxes(value, 0, 1) for value in outputs
    )
    final = HamiltonianChainState(
        position=positions,
        log_target=values,
        gradient=gradients,
        step_index=index,
        valid=valid
        & kernel.valid
        & jnp.all(jnp.isfinite(positions), axis=-1)
        & jnp.isfinite(values)
        & jnp.all(jnp.isfinite(gradients), axis=-1),
    )
    return HamiltonianSampleResult(
        samples=samples,
        log_target=log_values,
        accepted=accepted,
        acceptance_probability=probabilities,
        divergent=divergent,
        maximum_depth_reached=depth,
        nonfinite_gradient=nonfinite,
        leapfrog_steps=steps,
        final_state=final,
        root_key=jnp.asarray(key),
        frozen_step_size=kernel.step_size,
        target_id=kernel.target_id,
        method=kernel.method,
        claim="finite-capacity-frozen-production-hamiltonian-chain",
    )


def adapt_hamiltonian_kernel(
    kernel: PreparedHamiltonianKernel,
    state: HamiltonianChainState,
    plan: HamiltonianAdaptationPlan,
    /,
    *,
    key: Key[Array, ""],
) -> HamiltonianAdaptationResult:
    """Run an explicit finite warmup epoch and return one frozen production kernel."""
    if (
        not isinstance(kernel, PreparedHamiltonianKernel)
        or not isinstance(state, HamiltonianChainState)
        or not isinstance(plan, HamiltonianAdaptationPlan)
    ):
        raise TypeError("kernel/state/plan types are invalid.")
    fraction = (kernel.step_size - plan.minimum_step_size) / (
        plan.maximum_step_size - plan.minimum_step_size
    )
    raw = jnp.log(fraction) - jnp.log1p(-fraction)
    current = state
    sizes = []
    acceptances = []
    adapted = kernel
    for step in range(plan.warmup_steps):
        scale = plan.minimum_step_size + (
            plan.maximum_step_size - plan.minimum_step_size
        ) * jax.nn.sigmoid(raw)
        adapted = eqx.tree_at(
            lambda value: value.step_size,
            adapted,
            scale,
        )
        draw = sample_hamiltonian(
            adapted,
            current,
            key=key,
            num_draws=1,
        )
        acceptance = jnp.mean(draw.acceptance_probability)
        raw = raw + (
            plan.adaptation_rate / jnp.sqrt(jnp.asarray(step + 1, dtype=raw.dtype))
        ) * (acceptance - plan.target_acceptance)
        current = draw.final_state
        sizes.append(scale)
        acceptances.append(acceptance)
    final_scale = plan.minimum_step_size + (
        plan.maximum_step_size - plan.minimum_step_size
    ) * jax.nn.sigmoid(raw)
    adapted = eqx.tree_at(
        lambda value: value.step_size,
        adapted,
        final_scale,
    )
    size_history = (
        jnp.stack(sizes) if sizes else jnp.empty((0,), dtype=kernel.step_size.dtype)
    )
    acceptance_history = (
        jnp.stack(acceptances)
        if acceptances
        else jnp.empty((0,), dtype=kernel.step_size.dtype)
    )
    return HamiltonianAdaptationResult(
        kernel=adapted,
        final_state=current,
        step_size_history=size_history,
        acceptance_history=acceptance_history,
        valid=adapted.valid
        & current.valid
        & jnp.all(jnp.isfinite(size_history))
        & jnp.all(jnp.isfinite(acceptance_history)),
        frozen=True,
        claim="finite-warmup-only-production-frozen",
    )


__all__ = [
    "HamiltonianAdaptationResult",
    "HamiltonianAdaptationPlan",
    "HamiltonianChainState",
    "HamiltonianSampleResult",
    "PreparedHamiltonianKernel",
    "adapt_hamiltonian_kernel",
    "initialize_hamiltonian_state",
    "prepare_hamiltonian_kernel",
    "sample_hamiltonian",
]
