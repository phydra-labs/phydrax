#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sign-free CT-INT control and low-order fermionic diagram Monte Carlo."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import determinant_small_linear, SmallLinearSolvePlan
from ...uq import correlated_observable_diagnostics, CorrelatedObservablePolicy
from ._core import DiagramGraph


class FermionMonteCarloChain(StrictModule, NonTrainableState):
    orders: Array
    phases: Array
    proposal_ratios: Array
    weight_ratios: Array
    acceptance_probabilities: Array
    accepted: Array
    move_types: Array
    configuration_indices: Array


class FermionMonteCarloEvidence(StrictModule, NonTrainableState):
    average_phase: Array
    average_sign: Array
    signed_effective_sample_size: Array
    phase_covariance: Array
    integrated_order_autocorrelation_time: Array
    order_effective_sample_size: Array
    acceptance_rate: Array
    detailed_balance_residual: Array
    sign_free_symmetry_residual: Array
    minimum_real_phase: Array
    overflow_proposals: Array
    finite: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class SignFreeCTINTResult(StrictModule, NonTrainableState):
    chain: FermionMonteCarloChain
    order_histogram: Array
    final_sites: Array
    final_times: Array
    final_order: Array
    evidence: FermionMonteCarloEvidence
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class LowOrderDiagramMonteCarloResult(StrictModule, NonTrainableState):
    chain: FermionMonteCarloChain
    order_histogram: Array
    order_probabilities: Array
    evidence: FermionMonteCarloEvidence
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class SignFreeCTINTPlan(StrictModule, NonTrainableState):
    """Repulsive half-filled bipartite CT-INT determinant control, order at most three."""

    beta: float = eqx.field(static=True)
    interaction: float = eqx.field(static=True)
    sublattice_signs: Array
    steps: int = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    symmetry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        beta: float,
        interaction: float,
        sublattice_signs: ArrayLike,
        /,
        *,
        steps: int,
        maximum_order: int = 3,
        symmetry_tolerance: float = 1.0e-10,
    ):
        beta_ = float(beta)
        coupling = float(interaction)
        signs = np.asarray(sublattice_signs, dtype=int)
        draws = int(steps)
        order = int(maximum_order)
        tolerance = float(symmetry_tolerance)
        if (
            not np.isfinite(beta_)
            or beta_ <= 0.0
            or not np.isfinite(coupling)
            or coupling <= 0.0
            or signs.ndim != 1
            or signs.size == 0
            or np.any(np.abs(signs) != 1)
            or draws < 8
            or order not in (1, 2, 3)
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Sign-free CT-INT control parameters are invalid.")
        self.beta = beta_
        self.interaction = coupling
        self.sublattice_signs = jnp.asarray(signs)
        self.steps = draws
        self.maximum_order = order
        self.symmetry_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "half-filled-bipartite-sign-free-ct-int-control",
                "beta": beta_,
                "interaction": coupling,
                "sublattice_signs": array_tree_fingerprint(signs),
                "steps": draws,
                "maximum_order": order,
                "symmetry_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        up_kernel: Callable[[Array, Array, Array], Array],
        down_kernel: Callable[[Array, Array, Array], Array],
        /,
        *,
        kernel_id: str,
    ) -> "PreparedSignFreeCTINT":
        if not callable(up_kernel) or not callable(down_kernel):
            raise TypeError("CT-INT kernels must be callable.")
        if not str(kernel_id):
            raise ValueError("kernel_id must be non-empty.")
        return PreparedSignFreeCTINT(self, up_kernel, down_kernel, str(kernel_id))


class _CTINTState(StrictModule, NonTrainableState):
    sites: Array
    times: Array
    order: Array
    weight: Array
    overflow: Array


class PreparedSignFreeCTINT(StrictModule, NonTrainableState):
    __hash__ = object.__hash__

    plan: SignFreeCTINTPlan
    up_kernel: Callable = eqx.field(static=True)
    down_kernel: Callable = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    determinant_plan: SmallLinearSolvePlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, up_kernel, down_kernel, kernel_id, /):
        self.plan = plan
        self.up_kernel = up_kernel
        self.down_kernel = down_kernel
        self.kernel_id = kernel_id
        self.determinant_plan = SmallLinearSolvePlan(plan.maximum_order)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-sign-free-ct-int-control",
                "plan": plan.plan_id,
                "kernel_id": kernel_id,
            }
        )

    def _weight(
        self, sites: Array, times: Array, order: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        active = jnp.arange(self.plan.maximum_order) < order
        pair_active = active[:, None] & active[None, :]
        delta = times[:, None] - times[None, :]
        up_raw = jnp.asarray(self.up_kernel(sites[:, None], sites[None, :], delta))
        down_raw = jnp.asarray(self.down_kernel(sites[:, None], sites[None, :], delta))
        if (
            up_raw.shape != (self.plan.maximum_order,) * 2
            or down_raw.shape != up_raw.shape
        ):
            raise ValueError(
                "CT-INT kernels must return one maximum_order square matrix."
            )
        identity = jnp.eye(
            self.plan.maximum_order, dtype=jnp.result_type(up_raw, down_raw)
        )
        up = jnp.where(pair_active, up_raw, identity)
        down = jnp.where(pair_active, down_raw, identity)
        determinant_up = determinant_small_linear(self.determinant_plan, up)
        determinant_down = determinant_small_linear(self.determinant_plan, down)
        signs = self.plan.sublattice_signs[sites]
        symmetry = jnp.max(
            jnp.abs(
                jnp.where(
                    pair_active,
                    down_raw + signs[:, None] * signs[None, :] * jnp.conj(up_raw),
                    0.0,
                )
            )
        )
        weight = (-self.plan.interaction) ** order * determinant_up * determinant_down
        return weight, determinant_up, determinant_down, symmetry

    def run(self, key: Key[Array, ""], /) -> SignFreeCTINTResult:
        sites = jnp.zeros((self.plan.maximum_order,), dtype=jnp.int32)
        times = jnp.zeros((self.plan.maximum_order,))
        weight, _, _, _ = self._weight(sites, times, jnp.asarray(0, dtype=jnp.int32))
        initial = _CTINTState(
            sites,
            times,
            jnp.asarray(0, dtype=jnp.int32),
            weight,
            jnp.asarray(0, dtype=jnp.int32),
        )
        keys = jr.split(key, self.plan.steps)

        def advance(state, step_key):
            move_key, site_key, time_key, slot_key, accept_key = jr.split(step_key, 5)
            insertion = jr.bernoulli(move_key)
            can_insert = state.order < self.plan.maximum_order
            can_remove = state.order > 0
            available = jnp.where(insertion, can_insert, can_remove)
            proposed_site = jr.randint(
                site_key,
                (),
                0,
                self.plan.sublattice_signs.size,
                dtype=jnp.int32,
            )
            proposed_time = jr.uniform(time_key, (), minval=0.0, maxval=self.plan.beta)
            insert_sites = state.sites.at[state.order].set(proposed_site)
            insert_times = state.times.at[state.order].set(proposed_time)
            removal_slot = jnp.floor(
                jr.uniform(slot_key) * jnp.maximum(state.order, 1)
            ).astype(jnp.int32)
            last_slot = jnp.maximum(state.order - 1, 0)
            remove_sites = state.sites.at[removal_slot].set(state.sites[last_slot])
            remove_times = state.times.at[removal_slot].set(state.times[last_slot])
            candidate_sites = jnp.where(insertion, insert_sites, remove_sites)
            candidate_times = jnp.where(insertion, insert_times, remove_times)
            candidate_order = jnp.where(insertion, state.order + 1, state.order - 1)
            safe_order = jnp.clip(candidate_order, 0, self.plan.maximum_order)
            candidate_weight, _, _, symmetry = self._weight(
                candidate_sites, candidate_times, safe_order
            )
            weight_ratio = jnp.abs(candidate_weight) / jnp.maximum(
                jnp.abs(state.weight),
                jnp.finfo(jnp.asarray(state.weight).real.dtype).tiny,
            )
            proposal_ratio = jnp.where(
                available,
                jnp.where(
                    insertion,
                    self.plan.sublattice_signs.size * self.plan.beta / (state.order + 1),
                    state.order / (self.plan.sublattice_signs.size * self.plan.beta),
                ),
                0.0,
            )
            acceptance_probability = jnp.where(
                available, jnp.minimum(1.0, weight_ratio * proposal_ratio), 0.0
            )
            accepted = jr.uniform(accept_key) < acceptance_probability
            next_state = _CTINTState(
                jnp.where(accepted, candidate_sites, state.sites),
                jnp.where(accepted, candidate_times, state.times),
                jnp.where(accepted, safe_order, state.order),
                jnp.where(accepted, candidate_weight, state.weight),
                state.overflow + (insertion & ~can_insert).astype(jnp.int32),
            )
            phase = next_state.weight / jnp.where(
                jnp.abs(next_state.weight) > 0.0, jnp.abs(next_state.weight), 1.0
            )
            record = (
                next_state.order,
                phase,
                proposal_ratio,
                weight_ratio,
                acceptance_probability,
                accepted,
                insertion.astype(jnp.int32),
                symmetry,
            )
            return next_state, record

        final, records = jax.lax.scan(advance, initial, keys)
        (
            orders,
            phases,
            proposal_ratios,
            weight_ratios,
            probabilities,
            accepted,
            moves,
            symmetries,
        ) = records
        chain = FermionMonteCarloChain(
            orders,
            phases,
            proposal_ratios,
            weight_ratios,
            probabilities,
            accepted,
            moves,
            jnp.full_like(orders, -1),
        )
        evidence = _chain_evidence(
            chain,
            jnp.asarray(0.0),
            jnp.max(symmetries),
            jnp.min(jnp.real(phases)),
            final.overflow,
            self.prepared_id,
        )
        histogram = jnp.bincount(orders, length=self.plan.maximum_order + 1)
        return SignFreeCTINTResult(
            chain,
            histogram,
            final.sites,
            final.times,
            final.order,
            evidence,
            self.prepared_id,
            "candidate half-filled bipartite sign-free CT-INT control only; not a generic sign cure",
        )


class LowOrderFermionDiagramMonteCarloPlan(StrictModule, NonTrainableState):
    steps: int = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    maximum_diagrams: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, /, *, steps: int, maximum_order: int = 6, maximum_diagrams: int = 256
    ):
        draws, order, count = int(steps), int(maximum_order), int(maximum_diagrams)
        if draws < 8 or order < 1 or count < 2:
            raise ValueError("Low-order diagram Monte Carlo bounds are invalid.")
        self.steps = draws
        self.maximum_order = order
        self.maximum_diagrams = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "low-order-fermion-diagram-monte-carlo",
                "steps": draws,
                "maximum_order": order,
                "maximum_diagrams": count,
            }
        )

    def prepare(
        self,
        diagrams: tuple[DiagramGraph, ...],
        complex_weights: ArrayLike,
        proposal_matrix: ArrayLike,
        /,
    ) -> "PreparedLowOrderFermionDiagramMonteCarlo":
        items = tuple(diagrams)
        weights = np.asarray(complex_weights, dtype=complex)
        proposal = np.asarray(proposal_matrix, dtype=float)
        count = len(items)
        if (
            count < 2
            or count > self.maximum_diagrams
            or any(not isinstance(item, DiagramGraph) for item in items)
        ):
            raise ValueError("Diagram catalogue violates fixed bounds.")
        if any(
            not any(
                field.statistics == "fermion"
                for vertex in item.vertices
                for field in vertex.rule.fields
            )
            for item in items
        ):
            raise ValueError("Every low-order catalogue diagram must contain fermions.")
        orders = np.asarray([item.order for item in items], dtype=np.int32)
        if (
            np.max(orders) > self.maximum_order
            or weights.shape != (count,)
            or np.any(~np.isfinite(weights))
            or np.any(np.abs(weights) <= 0.0)
        ):
            raise ValueError("Diagram orders and weights are invalid.")
        if (
            proposal.shape != (count, count)
            or np.any(~np.isfinite(proposal))
            or np.any(proposal < 0.0)
            or not np.allclose(np.sum(proposal, axis=1), 1.0)
            or np.any((proposal > 0.0) != (proposal.T > 0.0))
        ):
            raise ValueError(
                "proposal_matrix must be finite, row-stochastic, and have symmetric support."
            )
        magnitudes = np.abs(weights)
        maximum_balance = 0.0
        for source in range(count):
            for target in range(count):
                if proposal[source, target] == 0.0:
                    continue
                ratio = (
                    magnitudes[target]
                    * proposal[target, source]
                    / (magnitudes[source] * proposal[source, target])
                )
                reverse = 1.0 / ratio
                maximum_balance = max(
                    maximum_balance,
                    abs(
                        magnitudes[source] * proposal[source, target] * min(1.0, ratio)
                        - magnitudes[target]
                        * proposal[target, source]
                        * min(1.0, reverse)
                    ),
                )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-low-order-fermion-diagram-monte-carlo",
                "plan": self.plan_id,
                "graphs": [item.graph_id for item in items],
                "weights": array_tree_fingerprint(weights),
                "proposal": array_tree_fingerprint(proposal),
            }
        )
        return PreparedLowOrderFermionDiagramMonteCarlo(
            self,
            jnp.asarray(orders),
            jnp.asarray(weights),
            jnp.asarray(proposal),
            jnp.asarray(maximum_balance),
            prepared_id,
        )


class PreparedLowOrderFermionDiagramMonteCarlo(StrictModule, NonTrainableState):
    plan: LowOrderFermionDiagramMonteCarloPlan
    orders: Array
    weights: Array
    proposal_matrix: Array
    detailed_balance_residual: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, orders, weights, proposal_matrix, residual, prepared_id, /):
        self.plan, self.orders, self.weights, self.proposal_matrix = (
            plan,
            orders,
            weights,
            proposal_matrix,
        )
        self.detailed_balance_residual, self.prepared_id = residual, str(prepared_id)

    def run(
        self, key: Key[Array, ""], /, *, initial_index: int = 0
    ) -> LowOrderDiagramMonteCarloResult:
        if initial_index < 0 or initial_index >= self.orders.size:
            raise ValueError("initial_index is outside the diagram catalogue.")
        keys = jr.split(key, self.plan.steps)

        def advance(index, step_key):
            proposal_key, acceptance_key = jr.split(step_key)
            candidate = jr.categorical(
                proposal_key, jnp.log(self.proposal_matrix[index])
            ).astype(jnp.int32)
            proposal_ratio = (
                self.proposal_matrix[candidate, index]
                / self.proposal_matrix[index, candidate]
            )
            weight_ratio = jnp.abs(self.weights[candidate]) / jnp.abs(self.weights[index])
            probability = jnp.minimum(1.0, proposal_ratio * weight_ratio)
            accepted = jr.uniform(acceptance_key) < probability
            next_index = jnp.where(accepted, candidate, index)
            phase = self.weights[next_index] / jnp.abs(self.weights[next_index])
            return next_index, (
                self.orders[next_index],
                phase,
                proposal_ratio,
                weight_ratio,
                probability,
                accepted,
                next_index,
            )

        _, records = jax.lax.scan(
            advance, jnp.asarray(initial_index, dtype=jnp.int32), keys
        )
        (
            orders,
            phases,
            proposal_ratios,
            weight_ratios,
            probabilities,
            accepted,
            indices,
        ) = records
        chain = FermionMonteCarloChain(
            orders,
            phases,
            proposal_ratios,
            weight_ratios,
            probabilities,
            accepted,
            jnp.zeros_like(orders),
            indices,
        )
        evidence = _chain_evidence(
            chain,
            self.detailed_balance_residual,
            jnp.asarray(jnp.nan),
            jnp.asarray(jnp.nan),
            jnp.asarray(0),
            self.prepared_id,
        )
        histogram = jnp.bincount(orders, length=self.plan.maximum_order + 1)
        return LowOrderDiagramMonteCarloResult(
            chain,
            histogram,
            histogram / jnp.sum(histogram),
            evidence,
            self.prepared_id,
            "candidate low-order fermionic diagram Monte Carlo; no phase-diagram claim",
        )


def _chain_evidence(
    chain, detailed_balance, symmetry, minimum_weight, overflow, prepared_id
):
    phase_components = jnp.stack(
        (jnp.real(chain.phases), jnp.imag(chain.phases)), axis=-1
    )
    centered = phase_components - jnp.mean(phase_components, axis=0)
    covariance = centered.T @ centered / jnp.maximum(phase_components.shape[0] - 1, 1)
    order_diagnostics = correlated_observable_diagnostics(
        chain.orders[None, :],
        policy=CorrelatedObservablePolicy(
            max_lag=min(64, chain.orders.size - 1), minimum_draws=8
        ),
    )
    average_phase = jnp.mean(chain.phases)
    average_sign = jnp.abs(average_phase)
    sign_ess = order_diagnostics.effective_sample_size * average_sign**2
    finite = jnp.all(
        jnp.isfinite(
            jnp.stack(
                (
                    jnp.real(chain.phases),
                    jnp.imag(chain.phases),
                    chain.proposal_ratios,
                    chain.weight_ratios,
                    chain.acceptance_probabilities,
                )
            )
        )
    ) & jnp.isfinite(detailed_balance)
    symmetry_ok = jnp.isnan(symmetry) | (symmetry <= 1.0e-10)
    successful = finite & symmetry_ok & (detailed_balance <= 1.0e-10)
    return FermionMonteCarloEvidence(
        average_phase,
        average_sign,
        sign_ess,
        covariance,
        order_diagnostics.integrated_autocorrelation_time,
        order_diagnostics.effective_sample_size,
        jnp.mean(chain.accepted),
        detailed_balance,
        symmetry,
        minimum_weight,
        overflow,
        finite,
        successful,
        prepared_id,
    )


__all__ = [
    "FermionMonteCarloChain",
    "FermionMonteCarloEvidence",
    "LowOrderDiagramMonteCarloResult",
    "LowOrderFermionDiagramMonteCarloPlan",
    "PreparedLowOrderFermionDiagramMonteCarlo",
    "PreparedSignFreeCTINT",
    "SignFreeCTINTPlan",
    "SignFreeCTINTResult",
]
