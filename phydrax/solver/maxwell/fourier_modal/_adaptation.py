#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral import (
    LatticeHarmonicDiscretization,
    prepare_spectral_modal_transfer,
)
from ._contracts import FourierModalMaxwellProblem, HomogeneousMaxwellPort
from ._runtime import (
    FourierModalSolvePolicy,
    FourierModalSolveResult,
    prepare_fourier_modal_maxwell,
    PreparedFourierModalMaxwell,
    solve_fourier_modal_maxwell,
)
from ._sources import FourierModalExcitation


class FourierModalHarmonicAdaptationPolicy(StrictModule, NonTrainableState):
    """Finite nested harmonic candidates and declared observable tolerances."""

    candidate_plans: tuple[LatticeHarmonicDiscretization, ...]
    observable_tolerances: Array
    maximum_epochs: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate_plans: Sequence[LatticeHarmonicDiscretization],
        observable_tolerances: ArrayLike,
        maximum_epochs: int,
        /,
    ):
        candidates = tuple(candidate_plans)
        tolerances = np.asarray(observable_tolerances, dtype=float)
        epochs = int(maximum_epochs)
        if (
            not candidates
            or any(
                not isinstance(value, LatticeHarmonicDiscretization)
                for value in candidates
            )
            or tolerances.ndim != 1
            or tolerances.size < 1
            or np.any(~np.isfinite(tolerances))
            or np.any(tolerances <= 0.0)
            or epochs < 1
            or epochs > len(candidates)
        ):
            raise ValueError("Fourier harmonic adaptation policy is invalid.")
        previous: set[str] = set()
        primitive_id = candidates[0].preparation_id
        for index, candidate in enumerate(candidates):
            current = set(candidate.plan.layout.mode_ids)
            if index and not previous.issubset(current):
                raise ValueError("Harmonic adaptation candidates must be nested.")
            if candidate.plan.layout.conjugate_indices.shape != (
                candidate.harmonic_count,
            ):
                raise ValueError("Harmonic candidates must be conjugate closed.")
            if not np.allclose(
                np.asarray(candidate.primitive_vectors),
                np.asarray(candidates[0].primitive_vectors),
            ):
                raise ValueError(
                    "Harmonic candidates must share one PeriodicCell lattice."
                )
            previous = current
        self.candidate_plans = candidates
        self.observable_tolerances = jnp.asarray(tolerances)
        self.maximum_epochs = epochs
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fourier-modal-harmonic-adaptation",
                "candidates": tuple(value.preparation_id for value in candidates),
                "observable_tolerances": tolerances.tolist(),
                "maximum_epochs": epochs,
                "primitive": primitive_id,
            }
        )


class AdaptiveFourierModalCase(StrictModule, NonTrainableState):
    prepared: PreparedFourierModalMaxwell
    result: FourierModalSolveResult
    observables: Array
    observable_history: Array
    removed_energy_history: Array
    active_epochs: Array
    epoch_count: Array
    converged: Array
    exhausted: Array
    policy_id: str = eqx.field(static=True)


def _transfer_excitation(
    excitation: FourierModalExcitation,
    source: LatticeHarmonicDiscretization,
    target: LatticeHarmonicDiscretization,
    /,
) -> tuple[FourierModalExcitation, Array]:
    transfer = prepare_spectral_modal_transfer(source, target)
    source_count = source.harmonic_count
    target_count = target.harmonic_count
    rhs_count = excitation.rhs_count

    def port(values: Array) -> tuple[Array, Array]:
        shaped = values.reshape((source_count, 2, rhs_count))
        result = transfer.apply_with_evidence(shaped)
        return result.coefficients.reshape(
            (2 * target_count, rhs_count)
        ), result.removed_coefficient_energy

    left, left_removed = port(excitation.left_incident)
    right, right_removed = port(excitation.right_incident)

    def current(values: Array) -> tuple[Array, Array]:
        result = transfer.apply_with_evidence(jnp.moveaxis(values, 1, 0))
        return jnp.moveaxis(result.coefficients, 0, 1), result.removed_coefficient_energy

    electric = []
    magnetic = []
    removed = left_removed + right_removed
    for value in excitation.electric_currents:
        transferred, loss = current(value)
        electric.append(transferred)
        removed = removed + loss
    for value in excitation.magnetic_currents:
        transferred, loss = current(value)
        magnetic.append(transferred)
        removed = removed + loss
    return (
        FourierModalExcitation(
            left,
            right,
            source_ids=excitation.source_ids,
            electric_currents=tuple(electric),
            magnetic_currents=tuple(magnetic),
            channel_weights=excitation.channel_weights,
        ),
        removed,
    )


def solve_adaptive_fourier_modal_case(
    problems: Sequence[FourierModalMaxwellProblem],
    excitation: FourierModalExcitation,
    policy: FourierModalHarmonicAdaptationPolicy,
    observable: Callable[[PreparedFourierModalMaxwell, FourierModalSolveResult], Array],
    /,
    *,
    solve_policy: FourierModalSolvePolicy | None = None,
) -> AdaptiveFourierModalCase:
    """Run finite host epochs; every in-epoch solve remains pure and fixed-layout."""

    cases = tuple(problems)
    if len(cases) < policy.maximum_epochs:
        raise ValueError("One Fourier-modal problem is required per permitted epoch.")
    for problem, candidate in zip(cases, policy.candidate_plans, strict=False):
        if problem.harmonics.preparation_id != candidate.preparation_id:
            raise ValueError("Adaptation problem harmonic layout does not match policy.")
        if not isinstance(problem.superstrate, HomogeneousMaxwellPort) or not isinstance(
            problem.substrate, HomogeneousMaxwellPort
        ):
            raise ValueError(
                "Harmonic amplitude transfer currently requires homogeneous ports."
            )
    tolerance_count = int(policy.observable_tolerances.size)
    history = jnp.zeros((policy.maximum_epochs, tolerance_count))
    removed_history = jnp.zeros((policy.maximum_epochs,))
    active = jnp.zeros((policy.maximum_epochs,), dtype=bool)
    current_excitation = excitation
    previous_observable = None
    converged = False
    prepared = None
    result = None
    current_observable = None
    epoch_count = 0
    for epoch in range(policy.maximum_epochs):
        prepared = prepare_fourier_modal_maxwell(cases[epoch], solve_policy)
        result = solve_fourier_modal_maxwell(prepared, current_excitation)
        current_observable = jnp.asarray(observable(prepared, result)).reshape((-1,))
        if current_observable.shape != (tolerance_count,):
            raise ValueError("Adaptive observable does not match declared tolerances.")
        history = history.at[epoch].set(current_observable)
        active = active.at[epoch].set(True)
        epoch_count = epoch + 1
        if previous_observable is not None:
            scale = jnp.maximum(jnp.abs(current_observable), 1.0)
            converged = bool(
                np.asarray(
                    jnp.all(
                        jnp.abs(current_observable - previous_observable)
                        <= policy.observable_tolerances * scale
                    )
                )
            )
            if converged:
                break
        if epoch + 1 < policy.maximum_epochs:
            current_excitation, removed = _transfer_excitation(
                current_excitation,
                policy.candidate_plans[epoch],
                policy.candidate_plans[epoch + 1],
            )
            removed_history = removed_history.at[epoch + 1].set(removed)
        previous_observable = current_observable
    if prepared is None or result is None or current_observable is None:
        raise RuntimeError("Adaptive Fourier-modal case executed no epoch.")
    return AdaptiveFourierModalCase(
        prepared,
        result,
        current_observable,
        history,
        removed_history,
        active,
        jnp.asarray(epoch_count, dtype=jnp.int32),
        jnp.asarray(converged),
        jnp.asarray(not converged and epoch_count == policy.maximum_epochs),
        policy.policy_id,
    )


__all__ = [
    "AdaptiveFourierModalCase",
    "FourierModalHarmonicAdaptationPolicy",
    "solve_adaptive_fourier_modal_case",
]
