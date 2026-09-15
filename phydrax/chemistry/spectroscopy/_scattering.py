#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Kinematic elastic scattering and exact finite-state dynamic structure."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...units import UnitDefinition
from ._response import (
    SpectralResponseEvidence,
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)


class XRayFormFactorRequest(StrictModule, NonTrainableState):
    cartesian_q: Array
    atom_ids: tuple[str, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self, cartesian_q: ArrayLike, atom_ids: tuple[str, ...], structure_id: str, /
    ):
        q = jnp.asarray(cartesian_q, dtype=float)
        atoms = tuple(str(atom).strip() for atom in atom_ids)
        structure = str(structure_id).strip()
        if (
            q.ndim != 2
            or q.shape[1] != 3
            or q.shape[0] == 0
            or bool(jnp.any(~jnp.isfinite(q)))
            or not atoms
            or len(set(atoms)) != len(atoms)
            or any(not atom for atom in atoms)
            or not structure
        ):
            raise ValueError("X-ray form-factor request is invalid.")
        self.cartesian_q = q
        self.atom_ids = atoms
        self.structure_id = structure
        self.request_id = canonical_fingerprint(
            {
                "kind": "xray-form-factor-request",
                "structure": structure,
                "atoms": list(atoms),
                "q": array_tree_fingerprint(np.asarray(q)),
            }
        )


class XRayFormFactorResult(StrictModule, NonTrainableState):
    form_factors: Array
    request: XRayFormFactorRequest
    provider_id: str = eqx.field(static=True)
    source_hashes: tuple[str, ...] = eqx.field(static=True)
    converged: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        form_factors: ArrayLike,
        request: XRayFormFactorRequest,
        provider_id: str,
        source_hashes: tuple[str, ...],
        converged: ArrayLike,
        /,
    ):
        factors = jnp.asarray(form_factors, dtype=float)
        provider = str(provider_id).strip()
        hashes = tuple(str(value).strip() for value in source_hashes)
        if (
            factors.shape != (request.cartesian_q.shape[0], len(request.atom_ids))
            or bool(jnp.any(~jnp.isfinite(factors)))
            or bool(jnp.any(factors < 0.0))
            or not provider
            or not hashes
            or any(not value for value in hashes)
        ):
            raise ValueError("Nonresonant X-ray factors or provenance are invalid.")
        self.form_factors = factors
        self.request = request
        self.provider_id = provider
        self.source_hashes = hashes
        self.converged = jnp.asarray(converged, dtype=bool).reshape(())
        self.result_id = canonical_fingerprint(
            {
                "kind": "nonresonant-xray-form-factors",
                "request": request.request_id,
                "provider": provider,
                "hashes": list(hashes),
                "converged": bool(self.converged),
                "factors": array_tree_fingerprint(np.asarray(factors)),
            }
        )


class ElasticScatteringEvidence(StrictModule, NonTrainableState):
    friedel_residual: Array
    passivity_residual: Array
    successful: Array


class ElasticScatteringResult(StrictModule, NonTrainableState):
    amplitudes: Array
    intensities: Array
    cartesian_q: Array
    evidence: ElasticScatteringEvidence
    probe: str = eqx.field(static=True)
    factor_source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        amplitudes: ArrayLike,
        intensities: ArrayLike,
        cartesian_q: ArrayLike,
        evidence: ElasticScatteringEvidence,
        probe: str,
        factor_source_id: str,
        plan_id: str,
        /,
    ):
        self.amplitudes = jnp.asarray(amplitudes)
        self.intensities = jnp.asarray(intensities)
        self.cartesian_q = jnp.asarray(cartesian_q)
        self.evidence = evidence
        self.probe = str(probe)
        self.factor_source_id = str(factor_source_id)
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "coherent-elastic-scattering-result",
                "probe": self.probe,
                "factor_source": self.factor_source_id,
                "plan": self.plan_id,
                "successful": bool(evidence.successful),
                "intensity": array_tree_fingerprint(np.asarray(self.intensities)),
            }
        )


class _AbstractElasticScatteringPlan(StrictModule, NonTrainableState):
    positions: Array
    debye_waller_tensors: Array
    reverse_q: Array
    q_capacity: int = eqx.field(static=True)
    atom_capacity: int = eqx.field(static=True)
    symmetry_tolerance: float = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions: ArrayLike,
        debye_waller_tensors: ArrayLike,
        reverse_q: ArrayLike,
        structure_id: str,
        /,
        *,
        q_capacity: int,
        atom_capacity: int,
        symmetry_tolerance: float = 1.0e-10,
    ):
        coordinate = jnp.asarray(positions, dtype=float)
        displacement = jnp.asarray(debye_waller_tensors, dtype=float)
        reverse = jnp.asarray(reverse_q, dtype=int)
        structure = str(structure_id).strip()
        tolerance = float(symmetry_tolerance)
        atoms = coordinate.shape[0] if coordinate.ndim == 2 else 0
        if (
            coordinate.shape != (atoms, 3)
            or atoms == 0
            or displacement.shape != (atoms, 3, 3)
            or reverse.ndim != 1
            or bool(jnp.any(~jnp.isfinite(coordinate)))
            or bool(jnp.any(~jnp.isfinite(displacement)))
            or bool(
                jnp.any(
                    jnp.abs(displacement - jnp.swapaxes(displacement, 1, 2)) > tolerance
                )
            )
            or int(q_capacity) <= 0
            or int(atom_capacity) < atoms
            or reverse.size > int(q_capacity)
            or bool(jnp.any(reverse < 0))
            or bool(jnp.any(reverse >= reverse.size))
            or bool(jnp.any(reverse[reverse] != jnp.arange(reverse.size)))
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or not structure
        ):
            raise ValueError("Elastic scattering geometry or capacity is invalid.")
        self.positions = coordinate
        self.debye_waller_tensors = displacement
        self.reverse_q = reverse
        self.q_capacity = int(q_capacity)
        self.atom_capacity = int(atom_capacity)
        self.symmetry_tolerance = tolerance
        self.structure_id = structure
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coherent-elastic-scattering-plan",
                "structure": structure,
                "q_capacity": self.q_capacity,
                "atom_capacity": self.atom_capacity,
                "symmetry_tolerance": tolerance,
                "arrays": array_tree_fingerprint(
                    {
                        "positions": np.asarray(coordinate),
                        "debye_waller": np.asarray(displacement),
                        "reverse_q": np.asarray(reverse),
                    }
                ),
            }
        )

    def _evaluate(
        self, cartesian_q: Array, factors: Array, probe: str, factor_source_id: str
    ) -> ElasticScatteringResult:
        if cartesian_q.shape != (self.reverse_q.size, 3) or factors.shape != (
            self.reverse_q.size,
            self.positions.shape[0],
        ):
            raise ValueError(
                "Scattering q points and atom factors must match the planned geometry."
            )
        phases = jnp.exp(1j * (cartesian_q @ self.positions.T))
        attenuation = jnp.exp(
            -0.5
            * contract(
                "qi,aij,qj->qa",
                cartesian_q,
                self.debye_waller_tensors,
                cartesian_q,
            )
        )
        amplitudes = jnp.sum(factors * attenuation * phases, axis=1)
        intensities = jnp.real(amplitudes * jnp.conj(amplitudes))
        friedel = jnp.max(jnp.abs(amplitudes[self.reverse_q] - jnp.conj(amplitudes)))
        scale = jnp.maximum(
            jnp.max(jnp.abs(amplitudes)), jnp.finfo(intensities.dtype).tiny
        )
        friedel_residual = friedel / scale
        passivity = jnp.maximum(-jnp.min(intensities), 0.0)
        successful = bool(friedel_residual <= self.symmetry_tolerance) and bool(
            passivity <= self.symmetry_tolerance
        )
        evidence = ElasticScatteringEvidence(
            friedel_residual, passivity, jnp.asarray(successful)
        )
        return ElasticScatteringResult(
            amplitudes,
            intensities,
            cartesian_q,
            evidence,
            probe,
            factor_source_id,
            self.plan_id,
        )


class ElasticXRayScatteringPlan(_AbstractElasticScatteringPlan):
    def evaluate(self, factors: XRayFormFactorResult, /) -> ElasticScatteringResult:
        if not bool(factors.converged):
            raise ValueError("Unconverged X-ray form factors cannot enter scattering.")
        if factors.request.structure_id != self.structure_id:
            raise ValueError("X-ray factors do not belong to the planned structure.")
        return self._evaluate(
            factors.request.cartesian_q,
            factors.form_factors,
            "xray-nonresonant-kinematic",
            factors.result_id,
        )


class ElasticNeutronScatteringPlan(_AbstractElasticScatteringPlan):
    def evaluate(
        self,
        cartesian_q: ArrayLike,
        coherent_lengths: ArrayLike,
        source_id: str,
        source_hashes: tuple[str, ...],
        /,
    ) -> ElasticScatteringResult:
        q = jnp.asarray(cartesian_q, dtype=float)
        lengths = jnp.asarray(coherent_lengths, dtype=float)
        source = str(source_id).strip()
        hashes = tuple(str(value).strip() for value in source_hashes)
        if (
            q.shape != (self.reverse_q.size, 3)
            or bool(jnp.any(~jnp.isfinite(q)))
            or lengths.shape != (self.positions.shape[0],)
            or bool(jnp.any(~jnp.isfinite(lengths)))
            or not source
            or not hashes
            or any(not value for value in hashes)
        ):
            raise ValueError(
                "Coherent neutron lengths require explicit finite values and provenance."
            )
        factors = jnp.broadcast_to(lengths[None, :], (q.shape[0], lengths.size))
        factor_source = canonical_fingerprint(
            {
                "kind": "coherent-neutron-lengths",
                "source": source,
                "hashes": list(hashes),
                "lengths": array_tree_fingerprint(np.asarray(lengths)),
            }
        )
        return self._evaluate(
            q, factors, "neutron-coherent-nuclear-kinematic", factor_source
        )


class DynamicStructureFactorEvidence(StrictModule, NonTrainableState):
    probability_residual: Array
    operator_adjoint_residual: Array
    equal_time_residual: Array
    detailed_balance_residual: Array
    successful: Array


class DynamicStructureFactorResult(StrictModule, NonTrainableState):
    raw_response: SpectralResponseProduct
    transition_energies: Array
    transition_weights: Array
    evidence: DynamicStructureFactorEvidence
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        raw_response: SpectralResponseProduct,
        transition_energies: ArrayLike,
        transition_weights: ArrayLike,
        evidence: DynamicStructureFactorEvidence,
        plan_id: str,
        /,
    ):
        self.raw_response = raw_response
        self.transition_energies = jnp.asarray(transition_energies)
        self.transition_weights = jnp.asarray(transition_weights)
        self.evidence = evidence
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "exact-finite-dynamic-structure-factor",
                "raw": raw_response.product_id,
                "plan": self.plan_id,
                "successful": bool(evidence.successful),
            }
        )


class DynamicStructureFactorPlan(StrictModule, NonTrainableState):
    beta: float = eqx.field(static=True)
    state_capacity: int = eqx.field(static=True)
    q_capacity: int = eqx.field(static=True)
    transition_capacity: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        beta: float,
        state_capacity: int,
        q_capacity: int,
        transition_capacity: int,
        residual_tolerance: float = 1.0e-10,
    ):
        beta_ = float(beta)
        tolerance = float(residual_tolerance)
        if (
            not isfinite(beta_)
            or beta_ < 0.0
            or int(state_capacity) <= 0
            or int(q_capacity) <= 0
            or int(transition_capacity) <= 0
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Dynamic structure factor plan is invalid.")
        self.beta = beta_
        self.state_capacity = int(state_capacity)
        self.q_capacity = int(q_capacity)
        self.transition_capacity = int(transition_capacity)
        self.residual_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "exact-finite-dynamic-structure-factor-plan",
                "beta": beta_,
                "state_capacity": self.state_capacity,
                "q_capacity": self.q_capacity,
                "transition_capacity": self.transition_capacity,
                "residual_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        energies: ArrayLike,
        probabilities: ArrayLike,
        operators_q: ArrayLike,
        reverse_q: ArrayLike,
        energy_unit: UnitDefinition,
        response_unit: UnitDefinition,
        source_id: str,
        /,
    ) -> DynamicStructureFactorResult:
        energy = jnp.asarray(energies, dtype=float)
        probability = jnp.asarray(probabilities, dtype=float)
        operators = jnp.asarray(operators_q)
        reverse = jnp.asarray(reverse_q, dtype=int)
        states = energy.size
        q_count = operators.shape[0] if operators.ndim == 3 else 0
        if (
            states > self.state_capacity
            or q_count > self.q_capacity
            or q_count * states * states > self.transition_capacity
        ):
            raise ValueError("Exact dynamic structure input exceeds a planned capacity.")
        if (
            energy.shape != (states,)
            or probability.shape != (states,)
            or operators.shape != (q_count, states, states)
            or reverse.shape != (q_count,)
        ):
            raise ValueError(
                "Exact finite-state energies, probabilities, operators, and q map do not align."
            )
        if (
            bool(jnp.any(~jnp.isfinite(energy)))
            or bool(jnp.any(~jnp.isfinite(probability)))
            or bool(jnp.any(probability < 0.0))
            or bool(jnp.any(~jnp.isfinite(operators)))
        ):
            raise ValueError(
                "Exact dynamic structure inputs must be finite and probabilities non-negative."
            )
        if (
            bool(jnp.any(reverse < 0))
            or bool(jnp.any(reverse >= q_count))
            or bool(jnp.any(reverse[reverse] != jnp.arange(q_count)))
        ):
            raise ValueError("q/−q map must be an in-range involution.")
        probability_residual = jnp.abs(jnp.sum(probability) - 1.0)
        adjoint_residual = jnp.max(
            jnp.abs(operators[reverse] - jnp.swapaxes(jnp.conj(operators), 1, 2))
        )
        delta = energy[:, None] - energy[None, :]
        weights = probability[None, None, :] * jnp.real(operators * jnp.conj(operators))
        reverse_weights = jnp.swapaxes(weights[reverse], 1, 2)
        balance_expected = jnp.exp(-self.beta * delta)[None, :, :] * weights
        balance_scale = jnp.maximum(jnp.max(weights), jnp.finfo(weights.dtype).tiny)
        detailed_balance = (
            jnp.max(jnp.abs(reverse_weights - balance_expected)) / balance_scale
        )
        equal_time = jnp.sum(weights, axis=(1, 2))

        delta_np = np.asarray(delta).reshape((-1,))
        unique_energy, inverse = np.unique(delta_np, return_inverse=True)
        coalesced = np.zeros(
            (q_count, unique_energy.size), dtype=np.asarray(weights).dtype
        )
        flat_weights = np.asarray(weights).reshape((q_count, -1))
        for q_index in range(q_count):
            np.add.at(coalesced[q_index], inverse, 2.0 * np.pi * flat_weights[q_index])
        coalesced_equal_time = np.sum(coalesced, axis=1) / (2.0 * np.pi)
        denominator = np.maximum(np.abs(np.asarray(equal_time)), np.finfo(float).tiny)
        equal_time_residual = float(
            np.max(np.abs(coalesced_equal_time - np.asarray(equal_time)) / denominator)
        )
        successful = (
            bool(probability_residual <= self.residual_tolerance)
            and bool(adjoint_residual <= self.residual_tolerance)
            and bool(equal_time_residual <= self.residual_tolerance)
            and bool(detailed_balance <= self.residual_tolerance)
        )
        evidence = DynamicStructureFactorEvidence(
            probability_residual,
            adjoint_residual,
            jnp.asarray(equal_time_residual),
            detailed_balance,
            jnp.asarray(successful),
        )
        raw_evidence = SpectralResponseEvidence(
            0.0, equal_time_residual, detailed_balance, 0.0, successful
        )
        source = str(source_id).strip()
        if not source:
            raise ValueError("Dynamic structure source ID must be non-empty.")
        raw = SpectralResponseProduct(
            unique_energy,
            coalesced,
            np.ones(unique_energy.shape, dtype=bool),
            energy_unit,
            response_unit,
            tuple(f"q[{index}]" for index in range(q_count)),
            SpectralResponseRepresentation.LINES,
            "exact-finite-lehmann-dynamic-structure",
            source,
            raw_evidence,
        )
        return DynamicStructureFactorResult(
            raw,
            jnp.broadcast_to(jnp.asarray(delta)[None, :, :], (q_count, states, states)),
            weights,
            evidence,
            self.plan_id,
        )


__all__ = [
    "DynamicStructureFactorEvidence",
    "DynamicStructureFactorPlan",
    "DynamicStructureFactorResult",
    "ElasticNeutronScatteringPlan",
    "ElasticScatteringEvidence",
    "ElasticScatteringResult",
    "ElasticXRayScatteringPlan",
    "XRayFormFactorRequest",
    "XRayFormFactorResult",
]
