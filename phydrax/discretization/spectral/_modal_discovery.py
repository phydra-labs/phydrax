# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ModalSupportDiscoveryPlan(StrictModule, NonTrainableState):
    candidate_layout: Any
    conjugate_indices: Array | None
    conjugate_signs: Array | None
    capacity: int = eqx.field(static=True)
    method: Literal["top_energy", "omp"] = eqx.field(static=True)
    omp_iterations: int = eqx.field(static=True)
    omp_step_size: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        candidate_layout: Any,
        capacity: int,
        /,
        *,
        method: Literal["top_energy", "omp"] = "top_energy",
        conjugate_indices: ArrayLike | None = None,
        conjugate_signs: ArrayLike | None = None,
        omp_iterations: int = 64,
        omp_step_size: float = 1.0e-2,
    ):
        size = 1
        for value in candidate_layout.coefficient_shape:
            size *= int(value)
        count = int(capacity)
        if count < 0 or count > size:
            raise ValueError("Modal support capacity lies outside the candidate layout.")
        if method not in ("top_energy", "omp"):
            raise ValueError("Unknown modal discovery method.")
        conjugates = (
            None
            if conjugate_indices is None
            else jnp.asarray(conjugate_indices, dtype=jnp.int32).reshape((-1,))
        )
        signs = (
            None
            if conjugate_signs is None
            else jnp.asarray(conjugate_signs).reshape((-1,))
        )
        if conjugates is not None:
            if conjugates.shape != (size,) or bool(
                jnp.any((conjugates < 0) | (conjugates >= size))
            ):
                raise ValueError("conjugate_indices must map every finite candidate.")
            if bool(jnp.any(conjugates[conjugates] != jnp.arange(size))):
                raise ValueError("conjugate_indices must be an involution.")
            if signs is None:
                signs = jnp.ones((size,), dtype=float)
            if signs.shape != (size,):
                raise ValueError("conjugate_signs must match candidate capacity.")
        elif signs is not None:
            raise ValueError("conjugate_signs require conjugate_indices.")
        if int(omp_iterations) <= 0 or float(omp_step_size) <= 0.0:
            raise ValueError("OMP fixed-work policy values must be positive.")
        self.candidate_layout = candidate_layout
        self.conjugate_indices = conjugates
        self.conjugate_signs = signs
        self.capacity = count
        self.method = method
        self.omp_iterations = int(omp_iterations)
        self.omp_step_size = float(omp_step_size)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "modal-support-discovery",
                "layout": candidate_layout.layout_id,
                "capacity": count,
                "method": method,
                "conjugate_indices": (
                    None if conjugates is None else array_tree_fingerprint(conjugates)
                ),
                "conjugate_signs": (
                    None if signs is None else array_tree_fingerprint(signs)
                ),
                "omp_iterations": int(omp_iterations),
                "omp_step_size": float(omp_step_size),
            }
        )


class PreparedModalSupport(StrictModule):
    multi_indices: Array
    coefficients: Array
    active: Array
    energies: Array
    residual_curve: Array
    support_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def discover_modal_support(
    plan: ModalSupportDiscoveryPlan,
    coefficients: ArrayLike,
    /,
    *,
    measurement: ArrayLike | None = None,
    observations: ArrayLike | None = None,
) -> PreparedModalSupport:
    if not isinstance(plan, ModalSupportDiscoveryPlan):
        raise TypeError("plan must be ModalSupportDiscoveryPlan.")
    values = jnp.asarray(coefficients)
    shape = tuple(int(value) for value in plan.candidate_layout.coefficient_shape)
    channel_last = (
        values.shape[-len(shape) - 1 : -1] == shape if values.ndim > len(shape) else False
    )
    payload = values if channel_last else values[..., None]
    if payload.shape[-len(shape) - 1 : -1] != shape:
        raise ValueError("Coefficient tensor does not match candidate layout.")
    leading = payload.shape[: -len(shape) - 1]
    flat = payload.reshape(leading + (-1, payload.shape[-1]))
    if plan.method == "omp":
        if measurement is None or observations is None:
            raise ValueError("OMP discovery requires measurement and observations.")
        matrix = jnp.asarray(measurement)
        target = jnp.asarray(observations)
        if matrix.ndim != 2 or matrix.shape[1] != flat.shape[-2]:
            raise ValueError("OMP measurement width must match candidate modes.")
        if target.shape[-2] != matrix.shape[0]:
            raise ValueError("OMP observations must carry measurement and channel axes.")
        estimate = jnp.zeros(
            target.shape[:-2] + (matrix.shape[1], target.shape[-1]),
            dtype=jnp.result_type(matrix.dtype, target.dtype),
        )
        selected_mask = jnp.zeros((matrix.shape[1],), dtype=bool)
        selected_count = jnp.asarray(0, dtype=jnp.int32)
        selection_steps = plan.capacity
        if plan.conjugate_indices is not None:
            mode_indices = jnp.arange(matrix.shape[1], dtype=jnp.int32)
            self_conjugate = plan.conjugate_indices == mode_indices
            orbit_sizes = jnp.where(self_conjugate, 1, 2)
        for _ in range(selection_steps):
            residual = target - contract("mn,...nc->...mc", matrix, estimate)
            correlation = jnp.sum(
                jnp.abs(contract("mn,...mc->...nc", jnp.conj(matrix), residual)) ** 2,
                axis=tuple(range(residual.ndim - 2)) + (-1,),
            )
            if plan.conjugate_indices is None:
                eligible = ~selected_mask
            else:
                correlation = jnp.where(
                    self_conjugate,
                    correlation,
                    correlation + correlation[plan.conjugate_indices],
                )
                eligible = (~selected_mask) & (
                    selected_count + orbit_sizes <= plan.capacity
                )
            has_candidate = jnp.any(eligible)
            chosen = jnp.argmax(jnp.where(eligible, correlation, -jnp.inf))
            updated_mask = selected_mask.at[chosen].set(True)
            chosen_size = jnp.asarray(1, dtype=jnp.int32)
            if plan.conjugate_indices is not None:
                updated_mask = updated_mask.at[plan.conjugate_indices[chosen]].set(True)
                chosen_size = orbit_sizes[chosen]
            selected_mask = jnp.where(has_candidate, updated_mask, selected_mask)
            selected_count = selected_count + jnp.where(has_candidate, chosen_size, 0)
            for _ in range(plan.omp_iterations):
                residual = contract("mn,...nc->...mc", matrix, estimate) - target
                gradient = contract(
                    "mn,...mc->...nc",
                    jnp.conj(matrix),
                    residual,
                )
                estimate = estimate - plan.omp_step_size * jnp.where(
                    selected_mask.reshape(
                        (1,) * (estimate.ndim - 2) + (estimate.shape[-2], 1)
                    ),
                    gradient,
                    0.0,
                )
        if plan.conjugate_indices is not None:
            assert plan.conjugate_signs is not None
            mirrored = plan.conjugate_signs.reshape(
                (1,) * (estimate.ndim - 2) + (estimate.shape[-2], 1)
            ) * jnp.conj(estimate[..., plan.conjugate_indices, :])
            estimate = 0.5 * (estimate + mirrored)
        flat = estimate
    energy = jnp.sum(
        jnp.abs(flat) ** 2,
        axis=tuple(range(len(flat.shape[:-2]))) + (-1,),
    )
    if plan.conjugate_indices is None:
        selected = jnp.argsort(-energy, stable=True)[: plan.capacity]
    else:
        group_energy = jnp.where(
            plan.conjugate_indices == jnp.arange(energy.size),
            energy,
            energy + energy[plan.conjugate_indices],
        )
        order = tuple(
            int(value)
            for value in jax.device_get(jnp.argsort(-group_energy, stable=True))
        )
        selected_host: list[int] = []
        active_host: list[bool] = []
        for candidate in order:
            partner = int(jax.device_get(plan.conjugate_indices[candidate]))
            group = (candidate,) if partner == candidate else (candidate, partner)
            if any(value in selected_host for value in group):
                continue
            if len(selected_host) + len(group) > plan.capacity:
                continue
            selected_host.extend(group)
            group_active = bool(jax.device_get(group_energy[candidate] > 0.0))
            active_host.extend((group_active,) * len(group))
            if len(selected_host) == plan.capacity:
                break
        padding = plan.capacity - len(selected_host)
        selected_host.extend((0,) * padding)
        active_host.extend((False,) * padding)
        selected = jnp.asarray(selected_host, dtype=jnp.int32)
        active = jnp.asarray(active_host, dtype=bool)
    if plan.conjugate_indices is None:
        active = energy[selected] > 0.0
    selected_coefficients = jnp.take(flat, selected, axis=-2)
    coefficient_active = active.reshape(
        (1,) * (selected_coefficients.ndim - 2) + (active.size, 1)
    )
    selected_coefficients = jnp.where(coefficient_active, selected_coefficients, 0.0)
    unraveled = jnp.unravel_index(selected, shape)
    multi_indices = jnp.stack(unraveled, axis=-1).astype(jnp.int32)
    sorted_energy = jnp.where(active, energy[selected], 0.0)
    total = jnp.sum(energy)
    residual_curve = total - jnp.cumsum(sorted_energy)
    support_id = canonical_fingerprint(
        {
            "kind": "prepared-modal-support",
            "plan": plan.plan_id,
            "indices": tuple(int(value) for value in jax.device_get(selected)),
            "active": tuple(bool(value) for value in jax.device_get(active)),
        }
    )
    return PreparedModalSupport(
        multi_indices,
        jax.lax.stop_gradient(selected_coefficients),
        jax.lax.stop_gradient(active),
        sorted_energy,
        residual_curve,
        support_id,
        plan.plan_id,
    )


class SpectralRegularityEstimate(StrictModule):
    sobolev_slope: Array
    analytic_slope: Array
    fit_residual: Array
    r_squared: Array
    shell_counts: Array
    valid: Array
    estimate_kind: str = eqx.field(static=True, default="finite-spectrum-diagnostic")


def estimate_spectral_regularity(
    wavenumbers: ArrayLike,
    energies: ArrayLike,
    shell_edges: ArrayLike,
    /,
    *,
    noise_floor: float = 0.0,
) -> SpectralRegularityEstimate:
    wave = jnp.asarray(wavenumbers, dtype=float).reshape((-1,))
    energy = jnp.asarray(energies, dtype=float).reshape((-1,))
    edges = jnp.asarray(shell_edges, dtype=float)
    if wave.shape != energy.shape or edges.ndim != 1 or edges.shape[0] < 3:
        raise ValueError("Regularity inputs have incompatible finite shapes.")
    shell = jnp.clip(
        jnp.searchsorted(edges, wave, side="right") - 1, 0, edges.shape[0] - 2
    )
    count = jnp.zeros((edges.shape[0] - 1,), dtype=jnp.int32).at[shell].add(1)
    sums = (
        jnp.zeros(count.shape, dtype=energy.dtype)
        .at[shell]
        .add(jnp.where(energy > noise_floor, energy, 0.0))
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    usable = (count > 0) & (sums > noise_floor) & (centers > 0.0)
    x_log = jnp.log(jnp.where(usable, centers, 1.0))
    y = jnp.log(jnp.where(usable, sums / jnp.maximum(count, 1), 1.0))
    weights = usable.astype(float)

    def slope(x):
        mean_x = jnp.sum(weights * x) / jnp.maximum(jnp.sum(weights), 1.0)
        mean_y = jnp.sum(weights * y) / jnp.maximum(jnp.sum(weights), 1.0)
        centered_x = x - mean_x
        value = jnp.sum(weights * centered_x * (y - mean_y)) / jnp.maximum(
            jnp.sum(weights * centered_x**2), 1.0e-30
        )
        prediction = mean_y + value * centered_x
        residual = jnp.sum(weights * (y - prediction) ** 2)
        total = jnp.sum(weights * (y - mean_y) ** 2)
        return value, residual, jnp.where(total > 0.0, 1.0 - residual / total, 0.0)

    sobolev, residual, r_squared = slope(x_log)
    analytic, _, _ = slope(centers)
    valid = jnp.sum(usable) >= 2
    return SpectralRegularityEstimate(
        sobolev, analytic, residual, r_squared, count, valid
    )


class MissingModeRecoveryPolicy(StrictModule, NonTrainableState):
    regularization: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        regularization: float = 1.0e-6,
        iterations: int = 256,
        step_size: float = 1.0e-2,
    ):
        if regularization < 0.0 or iterations <= 0 or step_size <= 0.0:
            raise ValueError("Missing-mode recovery policy values are invalid.")
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.step_size = float(step_size)


class MissingModeRecoveryProblem(StrictModule):
    measurement: Array
    observations: Array
    candidate_layout: Any
    held_out_measurement: Array | None
    held_out_observations: Array | None

    def __init__(
        self,
        measurement: ArrayLike,
        observations: ArrayLike,
        candidate_layout: Any,
        /,
        *,
        held_out_measurement: ArrayLike | None = None,
        held_out_observations: ArrayLike | None = None,
    ):
        matrix = jnp.asarray(measurement)
        values = jnp.asarray(observations)
        if matrix.ndim != 2 or values.shape[-1] != matrix.shape[0]:
            raise ValueError("Measurement operator and observations are incompatible.")
        if matrix.shape[1] <= matrix.shape[0] and bool(
            jnp.all(matrix == jnp.eye(*matrix.shape, dtype=matrix.dtype))
        ):
            raise ValueError("A diagonal coefficient mask cannot identify missing modes.")
        self.measurement = matrix
        self.observations = values
        self.candidate_layout = candidate_layout
        self.held_out_measurement = (
            None if held_out_measurement is None else jnp.asarray(held_out_measurement)
        )
        self.held_out_observations = (
            None if held_out_observations is None else jnp.asarray(held_out_observations)
        )


class MissingModeRecoveryResult(StrictModule):
    coefficients: Array
    data_residual: Array
    regularizer: Array
    kkt_residual: Array
    held_out_residual: Array
    iterations: Array
    valid: Array


def recover_missing_modes(
    problem: MissingModeRecoveryProblem,
    policy: MissingModeRecoveryPolicy | None = None,
    /,
) -> MissingModeRecoveryResult:
    if not isinstance(problem, MissingModeRecoveryProblem):
        raise TypeError("problem must be MissingModeRecoveryProblem.")
    selected = MissingModeRecoveryPolicy() if policy is None else policy
    matrix = problem.measurement
    target = problem.observations
    initial = jnp.zeros(
        target.shape[:-1] + (matrix.shape[1],),
        dtype=jnp.result_type(matrix.dtype, target.dtype),
    )

    def step(_, coefficients):
        residual = contract("mn,...n->...m", matrix, coefficients) - target
        gradient = (
            contract("mn,...m->...n", jnp.conj(matrix), residual)
            + selected.regularization * coefficients
        )
        return coefficients - selected.step_size * gradient

    coefficients = jax.lax.fori_loop(0, selected.iterations, step, initial)
    residual = contract("mn,...n->...m", matrix, coefficients) - target
    gradient = (
        contract("mn,...m->...n", jnp.conj(matrix), residual)
        + selected.regularization * coefficients
    )
    data_residual = jnp.sqrt(jnp.sum(jnp.abs(residual) ** 2, axis=-1))
    regularizer = selected.regularization * jnp.sum(jnp.abs(coefficients) ** 2, axis=-1)
    kkt = jnp.sqrt(jnp.sum(jnp.abs(gradient) ** 2, axis=-1))
    if problem.held_out_measurement is None:
        held_out = jnp.full(data_residual.shape, jnp.nan)
    else:
        assert problem.held_out_observations is not None
        held = (
            contract("mn,...n->...m", problem.held_out_measurement, coefficients)
            - problem.held_out_observations
        )
        held_out = jnp.sqrt(jnp.sum(jnp.abs(held) ** 2, axis=-1))
    valid = jnp.isfinite(data_residual) & jnp.isfinite(kkt)
    return MissingModeRecoveryResult(
        coefficients,
        data_residual,
        regularizer,
        kkt,
        held_out,
        jnp.asarray(selected.iterations, dtype=jnp.int32),
        valid,
    )


__all__ = [
    "MissingModeRecoveryPolicy",
    "MissingModeRecoveryProblem",
    "MissingModeRecoveryResult",
    "ModalSupportDiscoveryPlan",
    "PreparedModalSupport",
    "SpectralRegularityEstimate",
    "discover_modal_support",
    "estimate_spectral_regularity",
    "recover_missing_modes",
]
