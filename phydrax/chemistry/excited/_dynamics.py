#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Energy-ledger fewest-switches surface hopping with unitary electronic propagation."""

from __future__ import annotations

import abc
from collections.abc import Callable
from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...solver import solve_unitary_propagator, UnitaryPropagatorProblem


class DecoherenceKind(StrEnum):
    NONE = "none"
    ENERGY_BASED = "energy-based"


class FrustratedHopPolicy(StrEnum):
    REJECT = "reject"
    REVERSE = "reverse"


class NonadiabaticSurfaceEvaluation(StrictModule, NonTrainableState):
    energies: Array
    gradients: Array
    derivative_couplings: Array
    spin_orbit_couplings: Array
    successful: Array
    state_ids: tuple[str, ...] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies: ArrayLike,
        gradients: ArrayLike,
        derivative_couplings: ArrayLike,
        state_ids: tuple[str, ...],
        provider_id: str,
        /,
        *,
        spin_orbit_couplings: ArrayLike | None = None,
        successful: ArrayLike = True,
    ):
        energy = jnp.asarray(energies)
        gradient = jnp.asarray(gradients, dtype=energy.real.dtype)
        coupling = jnp.asarray(derivative_couplings, dtype=energy.real.dtype)
        states = tuple(str(value).strip() for value in state_ids)
        provider = str(provider_id).strip()
        count = energy.size
        spin_orbit = (
            jnp.zeros((count, count), dtype=jnp.result_type(energy.dtype, jnp.complex64))
            if spin_orbit_couplings is None
            else jnp.asarray(spin_orbit_couplings)
        )
        if energy.shape != (count,) or gradient.ndim != 3 or gradient.shape[0] != count:
            raise ValueError("Nonadiabatic energies and gradients do not align by state.")
        if coupling.shape != (count, count) + gradient.shape[1:]:
            raise ValueError(
                "Derivative couplings must have shape (state,state,atom,xyz)."
            )
        if (
            spin_orbit.shape != (count, count)
            or len(states) != count
            or len(set(states)) != count
            or any(not value for value in states)
            or not provider
        ):
            raise ValueError(
                "Nonadiabatic couplings or state/provider identities are invalid."
            )
        antisymmetry = jnp.max(
            jnp.abs(coupling + jnp.swapaxes(coupling, 0, 1)), initial=0.0
        )
        hermiticity = jnp.max(jnp.abs(spin_orbit - jnp.conj(spin_orbit.T)), initial=0.0)
        valid = (
            jnp.asarray(successful, dtype=jnp.bool_)
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(coupling))
            & (antisymmetry <= 1.0e-8)
            & (hermiticity <= 1.0e-8)
        )
        self.energies = energy
        self.gradients = gradient
        self.derivative_couplings = coupling
        self.spin_orbit_couplings = spin_orbit
        self.successful = valid
        self.state_ids = states
        self.provider_id = provider
        self.result_id = canonical_fingerprint(
            {
                "kind": "nonadiabatic-surface-evaluation",
                "states": list(states),
                "provider": provider,
                "successful": bool(valid),
                "arrays": array_tree_fingerprint(
                    {
                        "energies": np.asarray(energy),
                        "gradients": np.asarray(gradient),
                        "derivative_couplings": np.asarray(coupling),
                        "spin_orbit_couplings": np.asarray(spin_orbit),
                    }
                ),
            }
        )


class AbstractNonadiabaticSurfaceProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]

    @abc.abstractmethod
    def evaluate(self, positions: ArrayLike, /) -> NonadiabaticSurfaceEvaluation:
        raise NotImplementedError


NonadiabaticEvaluator = Callable[[ArrayLike], NonadiabaticSurfaceEvaluation]


class CallableNonadiabaticSurfaceProvider(AbstractNonadiabaticSurfaceProvider):
    evaluator: NonadiabaticEvaluator = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self, evaluator: NonadiabaticEvaluator, provider_id: str, /):
        if not callable(evaluator):
            raise TypeError("evaluator must be callable.")
        provider = str(provider_id).strip()
        if not provider:
            raise ValueError("provider_id must be non-empty.")
        self.evaluator = evaluator
        self.provider_id = provider

    def evaluate(self, positions: ArrayLike, /) -> NonadiabaticSurfaceEvaluation:
        result = self.evaluator(positions)
        if (
            not isinstance(result, NonadiabaticSurfaceEvaluation)
            or result.provider_id != self.provider_id
        ):
            raise ValueError("Nonadiabatic provider changed result type or identity.")
        return result


class SurfaceHopEvent(StrictModule):
    source_state: Array
    proposed_state: Array
    accepted_state: Array
    probabilities: Array
    random_draw: Array
    energy_gap: Array
    discriminant: Array
    attempted: Array
    frustrated: Array
    accepted: Array


class SurfaceHoppingState(StrictModule):
    positions: Array
    momenta: Array
    electronic_coefficients: Array
    active_state: Array
    time: Array
    step_index: Array
    random_key: Array
    total_energy: Array
    cumulative_energy_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class SurfaceHoppingStep(StrictModule):
    state: SurfaceHoppingState
    event: SurfaceHopEvent
    electronic_norm_residual: Array
    energy_residual: Array
    surface_result_id: str = eqx.field(static=True)
    successful: Array


class FewestSwitchesSurfaceHoppingPlan(StrictModule, NonTrainableState):
    provider: AbstractNonadiabaticSurfaceProvider
    masses: Array
    time_step: float = eqx.field(static=True)
    electronic_substeps: int = eqx.field(static=True)
    hbar: float = eqx.field(static=True)
    decoherence: DecoherenceKind = eqx.field(static=True)
    decoherence_parameter: float = eqx.field(static=True)
    frustrated_hop: FrustratedHopPolicy = eqx.field(static=True)
    rescaling_direction: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider: AbstractNonadiabaticSurfaceProvider,
        masses: ArrayLike,
        time_step: float,
        /,
        *,
        electronic_substeps: int = 4,
        hbar: float = 1.0,
        decoherence: DecoherenceKind = DecoherenceKind.NONE,
        decoherence_parameter: float = 0.1,
        frustrated_hop: FrustratedHopPolicy = FrustratedHopPolicy.REJECT,
        rescaling_direction: str = "derivative-coupling",
    ):
        if not isinstance(provider, AbstractNonadiabaticSurfaceProvider):
            raise TypeError(
                "provider must implement AbstractNonadiabaticSurfaceProvider."
            )
        mass = jnp.asarray(masses, dtype=jnp.float64).reshape((-1,))
        dt = float(time_step)
        substeps = int(electronic_substeps)
        hbar_ = float(hbar)
        parameter = float(decoherence_parameter)
        direction = str(rescaling_direction).strip()
        if (
            np.any(~np.isfinite(np.asarray(mass)))
            or np.any(np.asarray(mass) <= 0.0)
            or not isfinite(dt)
            or dt <= 0.0
            or substeps <= 0
            or not isfinite(hbar_)
            or hbar_ <= 0.0
            or not isfinite(parameter)
            or parameter <= 0.0
            or not isinstance(decoherence, DecoherenceKind)
            or not isinstance(frustrated_hop, FrustratedHopPolicy)
            or direction not in ("derivative-coupling", "gradient-difference")
        ):
            raise ValueError(
                "Surface-hopping masses, steps, policies, or units are invalid."
            )
        self.provider = provider
        self.masses = mass
        self.time_step = dt
        self.electronic_substeps = substeps
        self.hbar = hbar_
        self.decoherence = decoherence
        self.decoherence_parameter = parameter
        self.frustrated_hop = frustrated_hop
        self.rescaling_direction = direction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fewest-switches-surface-hopping-plan",
                "provider": provider.provider_id,
                "time_step": dt,
                "electronic_substeps": substeps,
                "hbar": hbar_,
                "decoherence": decoherence.value,
                "decoherence_parameter": parameter,
                "frustrated_hop": frustrated_hop.value,
                "rescaling_direction": direction,
                "masses": array_tree_fingerprint(np.asarray(mass)),
            }
        )

    def initialize(
        self,
        positions: ArrayLike,
        momenta: ArrayLike,
        electronic_coefficients: ArrayLike,
        active_state: int,
        random_key: ArrayLike,
        /,
    ) -> SurfaceHoppingState:
        coordinate = jnp.asarray(positions)
        momentum = jnp.asarray(momenta, dtype=coordinate.dtype)
        coefficients = jnp.asarray(electronic_coefficients)
        active = int(active_state)
        key = jnp.asarray(random_key)
        evaluation = self.provider.evaluate(coordinate)
        if (
            coordinate.shape != (self.masses.size, 3)
            or momentum.shape != coordinate.shape
        ):
            raise ValueError("Surface-hopping nuclear arrays do not align with masses.")
        if (
            coefficients.shape != evaluation.energies.shape
            or active < 0
            or active >= coefficients.size
        ):
            raise ValueError(
                "Electronic coefficients or active state do not align with surfaces."
            )
        norm = jnp.sqrt(jnp.real(jnp.vdot(coefficients, coefficients)))
        if not bool(jnp.isfinite(norm) & (norm > 0.0)):
            raise ValueError("Electronic coefficients must have a finite nonzero norm.")
        coefficients = coefficients / norm
        kinetic = 0.5 * jnp.sum(momentum**2 / self.masses[:, None])
        total = kinetic + evaluation.energies[active]
        successful = evaluation.successful
        return SurfaceHoppingState(
            coordinate,
            momentum,
            coefficients,
            jnp.asarray(active, dtype=jnp.int32),
            jnp.asarray(0.0, dtype=coordinate.dtype),
            jnp.asarray(0, dtype=jnp.int64),
            key,
            total,
            jnp.asarray(0.0, dtype=coordinate.dtype),
            successful,
            self.plan_id,
        )

    def _electronic_step(self, coefficients, evaluation, velocity, /):
        coupling_rate = contract("nx,ijnx->ij", velocity, evaluation.derivative_couplings)
        hamiltonian = (
            jnp.diag(evaluation.energies.astype(evaluation.spin_orbit_couplings.dtype))
            + evaluation.spin_orbit_couplings
            - 1.0j * self.hbar * coupling_rate
        )
        problem = UnitaryPropagatorProblem(
            lambda time, args: args,
            coefficients.size,
            t0=0.0,
            t1=self.time_step,
            hbar=self.hbar,
            args=hamiltonian,
        )
        propagation = solve_unitary_propagator(
            problem,
            save_times=jnp.asarray([self.time_step]),
            dt0=self.time_step / self.electronic_substeps,
            max_steps=max(16, 4 * self.electronic_substeps),
        )
        updated = propagation.propagators[-1] @ coefficients
        norm_residual = jnp.abs(jnp.real(jnp.vdot(updated, updated)) - 1.0)
        return updated, coupling_rate, norm_residual, propagation.valid

    def step(self, state: SurfaceHoppingState, /) -> SurfaceHoppingStep:
        if state.plan_id != self.plan_id:
            raise ValueError("Surface-hopping state belongs to another plan.")
        initial = self.provider.evaluate(state.positions)
        active = state.active_state
        half_momentum = state.momenta - 0.5 * self.time_step * initial.gradients[active]
        positions = (
            state.positions + self.time_step * half_momentum / self.masses[:, None]
        )
        final = self.provider.evaluate(positions)
        velocity = half_momentum / self.masses[:, None]
        coefficients, coupling_rate, norm_residual, electronic_valid = (
            self._electronic_step(state.electronic_coefficients, final, velocity)
        )
        denominator = jnp.maximum(
            jnp.abs(coefficients[active]) ** 2,
            jnp.finfo(coefficients.real.dtype).tiny,
        )
        coherence = 0.5 * (
            jnp.conj(state.electronic_coefficients[active])
            * state.electronic_coefficients
            + jnp.conj(coefficients[active]) * coefficients
        )
        population_outflow = (
            jnp.real(coherence * coupling_rate[active])
            - jnp.imag(coherence * final.spin_orbit_couplings[active]) / self.hbar
        )
        raw_probabilities = (
            2.0 * self.time_step * jnp.maximum(population_outflow, 0.0) / denominator
        )
        raw_probabilities = raw_probabilities.at[active].set(0.0)
        total_probability = jnp.sum(raw_probabilities)
        probabilities = raw_probabilities / jnp.maximum(total_probability, 1.0)
        key, event_key = jax.random.split(state.random_key)
        draw = jax.random.uniform(event_key, dtype=positions.dtype)
        cumulative = jnp.cumsum(probabilities)
        proposed = jnp.searchsorted(cumulative, draw, side="right")
        attempted = draw < jnp.minimum(total_probability, 1.0)
        proposed = jnp.where(
            attempted, jnp.minimum(proposed, coefficients.size - 1), active
        )
        gap = final.energies[proposed] - final.energies[active]
        direction = jnp.where(
            self.rescaling_direction == "derivative-coupling",
            final.derivative_couplings[active, proposed],
            final.gradients[proposed] - final.gradients[active],
        )
        quadratic = 0.5 * jnp.sum(direction**2 / self.masses[:, None])
        linear = jnp.sum(half_momentum * direction / self.masses[:, None])
        discriminant = linear**2 - 4.0 * quadratic * gap
        can_rescale = attempted & (quadratic > 0.0) & (discriminant >= 0.0)
        square_root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
        first = (-linear + square_root) / jnp.where(
            2.0 * quadratic > 0.0, 2.0 * quadratic, 1.0
        )
        second = (-linear - square_root) / jnp.where(
            2.0 * quadratic > 0.0, 2.0 * quadratic, 1.0
        )
        scale = jnp.where(jnp.abs(first) < jnp.abs(second), first, second)
        rescaled = half_momentum + scale * direction
        frustrated = attempted & ~can_rescale
        if self.frustrated_hop is FrustratedHopPolicy.REVERSE:
            projection = linear / jnp.where(2.0 * quadratic > 0.0, 2.0 * quadratic, 1.0)
            frustrated_momentum = half_momentum - 2.0 * projection * direction
        else:
            frustrated_momentum = half_momentum
        selected_momentum = jnp.where(
            can_rescale,
            rescaled,
            jnp.where(frustrated, frustrated_momentum, half_momentum),
        )
        accepted_state = jnp.where(can_rescale, proposed, active)
        momentum = (
            selected_momentum - 0.5 * self.time_step * final.gradients[accepted_state]
        )
        if self.decoherence is DecoherenceKind.ENERGY_BASED:
            gaps = jnp.abs(final.energies - final.energies[accepted_state])
            damping = jnp.exp(-self.time_step * gaps / self.decoherence_parameter)
            damping = damping.at[accepted_state].set(1.0)
            coefficients = coefficients * damping
            coefficients = coefficients / jnp.sqrt(
                jnp.real(jnp.vdot(coefficients, coefficients))
            )
        kinetic = 0.5 * jnp.sum(momentum**2 / self.masses[:, None])
        total = kinetic + final.energies[accepted_state]
        energy_residual = total - state.total_energy
        successful = (
            state.successful
            & initial.successful
            & final.successful
            & electronic_valid
            & jnp.isfinite(total)
            & jnp.all(jnp.isfinite(momentum))
            & (norm_residual <= 1.0e-8)
        )
        next_state = SurfaceHoppingState(
            positions,
            momentum,
            coefficients,
            accepted_state,
            state.time + self.time_step,
            state.step_index + 1,
            key,
            total,
            state.cumulative_energy_residual + energy_residual,
            successful,
            self.plan_id,
        )
        event = SurfaceHopEvent(
            active,
            proposed,
            accepted_state,
            probabilities,
            draw,
            gap,
            discriminant,
            attempted,
            frustrated,
            can_rescale,
        )
        return SurfaceHoppingStep(
            next_state,
            event,
            norm_residual,
            energy_residual,
            final.result_id,
            successful,
        )

    def run(self, initial: SurfaceHoppingState, step_count: int, /):
        count = int(step_count)
        if count < 0:
            raise ValueError("step_count must be non-negative.")
        states = [initial]
        events = []
        current = initial
        for _ in range(count):
            result = self.step(current)
            states.append(result.state)
            events.append(result.event)
            current = result.state
        return tuple(states), tuple(events)


__all__ = [
    "AbstractNonadiabaticSurfaceProvider",
    "CallableNonadiabaticSurfaceProvider",
    "DecoherenceKind",
    "FewestSwitchesSurfaceHoppingPlan",
    "FrustratedHopPolicy",
    "NonadiabaticSurfaceEvaluation",
    "SurfaceHopEvent",
    "SurfaceHoppingState",
    "SurfaceHoppingStep",
]
