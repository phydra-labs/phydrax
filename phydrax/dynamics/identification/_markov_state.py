#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import DenseLinearOperator
from ...linalg.eigen import general_eigensolve, GeneralEigenproblem
from .._trajectory import TrajectoryData
from ._status import (
    IDENTIFICATION_INFEASIBLE,
    IDENTIFICATION_INSUFFICIENT_SAMPLES,
    IDENTIFICATION_NONFINITE,
    IDENTIFICATION_SUCCESS,
)
from ._variational_kinetics import (
    _lagged_pair_data,
    LaggedKineticEvidence,
    LaggedPairWeighting,
)


class MarkovStateDiagnostics(StrictModule):
    counts: Array
    row_mass: Array
    active_states: Array
    empty_states: Array
    communicating_labels: Array
    communicating_class_count: Array
    irreducible: Array
    row_stochasticity_residual: Array
    detailed_balance_residual: Array
    stationarity_residual: Array
    effective_samples: Array
    lag: LaggedKineticEvidence


class MarkovStateModel(StrictModule):
    transition_matrix: Array
    stationary_probabilities: Array
    eigenvalues: Array
    diagnostics: MarkovStateDiagnostics
    valid: Array
    status: Array
    reversible: bool = eqx.field(static=True)
    state_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def propagate(self, probabilities: ArrayLike, steps: int = 1, /) -> Array:
        value = jnp.asarray(probabilities, dtype=self.transition_matrix.dtype)
        if value.shape[-1:] != (self.state_count,):
            raise ValueError(
                f"probabilities must end in state dimension {self.state_count}."
            )
        count = int(steps)
        if count < 0:
            raise ValueError("steps must be nonnegative.")
        power = jnp.eye(self.state_count, dtype=self.transition_matrix.dtype)
        for _ in range(count):
            power = power @ self.transition_matrix
        return contract("...i,ij->...j", value, power)

    def implied_timescales(self, /) -> tuple[Array, Array]:
        values = jnp.abs(self.eigenvalues)
        admissible = (
            self.valid
            & self.diagnostics.lag.uniform_physical_lag
            & jnp.isfinite(values)
            & (values > 0.0)
            & (values < 1.0)
        )
        safe = jnp.where(admissible, values, 0.5)
        times = -self.diagnostics.lag.physical_lag_mean / jnp.log(safe)
        return jnp.where(admissible, times, jnp.nan), admissible


def _assignments(
    values: ArrayLike,
    sample_shape: tuple[int, ...],
    state_count: int | None,
    /,
) -> tuple[Array, Array, int, str]:
    assignment = jnp.asarray(values)
    if tuple(assignment.shape) == sample_shape:
        if not jnp.issubdtype(assignment.dtype, jnp.integer):
            raise TypeError("Hard state assignments must use an integer dtype.")
        if state_count is None:
            host = np.asarray(assignment)
            if host.size == 0 or np.any(host < 0):
                raise ValueError(
                    "Hard assignments require nonnegative states and inferable support."
                )
            states = int(np.max(host)) + 1
        else:
            states = int(state_count)
        if states <= 0:
            raise ValueError("state_count must be positive.")
        valid = (assignment >= 0) & (assignment < states)
        safe = jnp.where(valid, assignment, 0)
        probabilities = jax.nn.one_hot(safe, states, dtype=float)
        return probabilities, valid, states, "hard"
    if (
        assignment.ndim == len(sample_shape) + 1
        and tuple(assignment.shape[:-1]) == sample_shape
    ):
        states = int(assignment.shape[-1])
        if state_count is not None and int(state_count) != states:
            raise ValueError("state_count does not match soft assignment width.")
        if states <= 0:
            raise ValueError("Soft assignments require at least one state.")
        if not jnp.issubdtype(assignment.dtype, jnp.inexact):
            assignment = assignment.astype(float)
        tolerance = 64.0 * jnp.finfo(assignment.dtype).eps
        finite = jnp.all(jnp.isfinite(assignment), axis=-1)
        nonnegative = jnp.all(assignment >= -tolerance, axis=-1)
        normalized = jnp.abs(jnp.sum(assignment, axis=-1) - 1.0) <= tolerance * states
        valid = finite & nonnegative & normalized
        safe = jnp.where(valid[..., None], jnp.maximum(assignment, 0.0), 0.0)
        mass = jnp.sum(safe, axis=-1, keepdims=True)
        safe = safe / jnp.where(mass > 0.0, mass, 1.0)
        return safe, valid, states, "soft"
    raise ValueError(
        "assignments must be hard sample indices or soft sample-by-state probabilities."
    )


def _communicating_classes(matrix: Array, active: Array, /) -> tuple[Array, Array, Array]:
    count = int(matrix.shape[0])
    reach = (matrix > 0.0) | jnp.eye(count, dtype=bool)
    for _ in range(count):
        reach = reach | ((reach.astype(jnp.int32) @ reach.astype(jnp.int32)) > 0)
    mutual = reach & reach.T & active[:, None] & active[None, :]
    indices = jnp.arange(count, dtype=jnp.int32)
    labels = jnp.min(jnp.where(mutual, indices[None, :], count), axis=1)
    labels = jnp.where(active, labels, -1)
    leaders = active & (labels == indices)
    class_count = jnp.sum(leaders).astype(jnp.int32)
    irreducible = class_count == 1
    return labels, class_count, irreducible


def _stationary_distribution(matrix: Array, active: Array, /) -> Array:
    initial = active.astype(matrix.dtype)
    initial = initial / jnp.maximum(jnp.sum(initial), 1.0)

    def body(_, carry):
        current, average = carry
        following = current @ matrix
        return following, average + following

    _, accumulated = jax.lax.fori_loop(0, 1024, body, (initial, jnp.zeros_like(initial)))
    distribution = accumulated / 1024.0
    return distribution / jnp.maximum(jnp.sum(distribution), 1.0)


def fit_markov_state_model(
    data: TrajectoryData,
    assignments: ArrayLike,
    /,
    *,
    state_count: int | None = None,
    lag: int = 1,
    reversible: bool = False,
    weighting: LaggedPairWeighting = LaggedPairWeighting.GEOMETRIC,
    pseudocount: float = 0.0,
    lag_tolerance: float = 1.0e-8,
) -> MarkovStateModel:
    """Fit a hard- or soft-assignment transition model without crossing resets."""

    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    pseudo = float(pseudocount)
    if not isfinite(pseudo) or pseudo < 0.0:
        raise ValueError("pseudocount must be finite and nonnegative.")
    probabilities, assignment_valid, states, assignment_kind = _assignments(
        assignments, data.case_shape + (data.capacity,), state_count
    )
    transitions, weights, lag_evidence = _lagged_pair_data(
        data, lag, weighting, lag_tolerance
    )
    pair_count = data.capacity - transitions.lag
    source = probabilities[..., :pair_count, :]
    target = probabilities[..., transitions.lag :, :]
    valid = (
        transitions.valid
        & assignment_valid[..., :pair_count]
        & assignment_valid[..., transitions.lag :]
    )
    flat_weights = jnp.where(valid, weights, 0.0).reshape((-1,))
    source = source.reshape((-1, states))
    target = target.reshape((-1, states))
    raw_counts = contract("n,ni,nj->ij", flat_weights, source, target)
    counts = 0.5 * (raw_counts + raw_counts.T) if reversible else raw_counts
    counts = counts + jnp.asarray(pseudo, dtype=counts.dtype)
    raw_row_mass = jnp.sum(raw_counts, axis=1)
    active = raw_row_mass > 0.0
    row_mass = jnp.sum(counts, axis=1)
    safe_rows = jnp.where(row_mass[:, None] > 0.0, counts / row_mass[:, None], 0.0)
    transition = jnp.where(
        (row_mass > 0.0)[:, None], safe_rows, jnp.eye(states, dtype=counts.dtype)
    )
    if reversible:
        stationary = row_mass / jnp.maximum(jnp.sum(row_mass), 1.0)
    else:
        stationary = _stationary_distribution(transition, active)
    labels, component_count, irreducible = _communicating_classes(transition, active)
    row_residual = jnp.max(jnp.abs(jnp.sum(transition, axis=1) - 1.0))
    flux = stationary[:, None] * transition
    detailed_balance = jnp.max(jnp.abs(flux - flux.T))
    stationarity = jnp.max(jnp.abs(stationary @ transition - stationary))
    spectrum = general_eigensolve(
        GeneralEigenproblem(
            DenseLinearOperator(transition),
            problem_id=f"markov-state:{data.dataset_id}:{int(lag)}",
        )
    )
    order = jnp.argsort(jnp.abs(spectrum.eigenvalues - 1.0))
    eigenvalues = spectrum.eigenvalues[order]
    finite = (
        jnp.all(jnp.isfinite(transition))
        & jnp.all(jnp.isfinite(stationary))
        & jnp.all(jnp.isfinite(eigenvalues))
    )
    valid_pairs = jnp.sum(valid)
    enough = valid_pairs > 0
    valid_model = (
        finite
        & enough
        & spectrum.successful
        & lag_evidence.uniform_physical_lag
        & (row_residual <= 128.0 * jnp.finfo(transition.dtype).eps * states)
    )
    status = jnp.where(
        ~finite,
        IDENTIFICATION_NONFINITE,
        jnp.where(
            ~enough,
            IDENTIFICATION_INSUFFICIENT_SAMPLES,
            jnp.where(
                spectrum.successful & lag_evidence.uniform_physical_lag,
                IDENTIFICATION_SUCCESS,
                IDENTIFICATION_INFEASIBLE,
            ),
        ),
    ).astype(jnp.int32)
    diagnostics = MarkovStateDiagnostics(
        counts=counts,
        row_mass=row_mass,
        active_states=active,
        empty_states=~active,
        communicating_labels=labels,
        communicating_class_count=component_count,
        irreducible=irreducible,
        row_stochasticity_residual=row_residual,
        detailed_balance_residual=detailed_balance,
        stationarity_residual=stationarity,
        effective_samples=lag_evidence.effective_samples,
        lag=lag_evidence,
    )
    model_id = canonical_fingerprint(
        {
            "kind": "markov-state-model",
            "dataset": data.dataset_id,
            "assignments": assignment_kind,
            "states": states,
            "lag": int(lag),
            "reversible": bool(reversible),
            "weighting": weighting.value,
            "pseudocount": pseudo.hex(),
        }
    )
    return MarkovStateModel(
        transition_matrix=transition,
        stationary_probabilities=stationary,
        eigenvalues=eigenvalues,
        diagnostics=diagnostics,
        valid=valid_model,
        status=status,
        reversible=bool(reversible),
        state_count=states,
        source_id=data.source_id,
        model_id=model_id,
    )


__all__ = [
    "MarkovStateDiagnostics",
    "MarkovStateModel",
    "fit_markov_state_model",
]
