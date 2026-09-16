#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, LeastSquaresProblem, solve


class SpectralSubmanifoldEvidence(StrictModule, NonTrainableState):
    selected_eigenvalues: Array
    spectral_gap: Array
    spectral_quotient: int = eqx.field(static=True)
    minimum_resonance_detuning: float = eqx.field(static=True)
    hyperbolic: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        selected_eigenvalues: ArrayLike,
        spectral_gap: ArrayLike,
        /,
        *,
        spectral_quotient: int,
        minimum_resonance_detuning: float,
    ):
        eigenvalues = jnp.asarray(selected_eigenvalues)
        gap = jnp.asarray(spectral_gap)
        quotient = int(spectral_quotient)
        detuning = float(minimum_resonance_detuning)
        if eigenvalues.ndim != 1 or eigenvalues.size == 0 or gap.shape != ():
            raise ValueError("SSM spectral evidence has invalid shape.")
        if quotient < 1 or not np.isfinite(detuning) or detuning <= 0.0:
            raise ValueError("SSM spectral quotient and detuning must be positive.")
        hyperbolic = bool(np.all(np.real(np.asarray(eigenvalues)) < 0.0))
        self.selected_eigenvalues = eigenvalues
        self.spectral_gap = gap
        self.spectral_quotient = quotient
        self.minimum_resonance_detuning = detuning
        self.hyperbolic = hyperbolic
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "spectral-submanifold-evidence",
                "spectral_quotient": quotient,
                "minimum_resonance_detuning": detuning,
                "hyperbolic": hyperbolic,
                "content": array_tree_fingerprint(
                    {"eigenvalues": eigenvalues, "gap": gap}
                )["sha256"],
            }
        )


class SpectralSubmanifoldModel(StrictModule, NonTrainableState):
    chart_coefficients: Array
    flow_coefficients: Array
    exponents: Array
    evidence: SpectralSubmanifoldEvidence
    validity_radius: float = eqx.field(static=True)
    observation_contract_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        chart_coefficients: ArrayLike,
        flow_coefficients: ArrayLike,
        exponents: ArrayLike,
        evidence: SpectralSubmanifoldEvidence,
        /,
        *,
        validity_radius: float,
        observation_contract_id: str,
        partition_id: str,
    ):
        chart = jnp.asarray(chart_coefficients)
        flow = jnp.asarray(flow_coefficients)
        powers = jnp.asarray(exponents, dtype=jnp.int32)
        if (
            not isinstance(evidence, SpectralSubmanifoldEvidence)
            or not evidence.hyperbolic
        ):
            raise ValueError("SSM construction requires hyperbolic spectral evidence.")
        if (
            powers.ndim != 2
            or chart.ndim != 2
            or flow.ndim != 2
            or chart.shape[1] != powers.shape[0]
            or flow.shape[1] != powers.shape[0]
            or flow.shape[0] != powers.shape[1]
        ):
            raise ValueError(
                "SSM chart, flow, and monomial exponent shapes are inconsistent."
            )
        if np.any(np.asarray(powers) < 0):
            raise ValueError("SSM monomial exponents must be nonnegative.")
        radius = float(validity_radius)
        observation = str(observation_contract_id)
        partition = str(partition_id)
        if not np.isfinite(radius) or radius <= 0.0 or not observation or not partition:
            raise ValueError("SSM validity and identities must be valid.")
        self.chart_coefficients = chart
        self.flow_coefficients = flow
        self.exponents = powers
        self.evidence = evidence
        self.validity_radius = radius
        self.observation_contract_id = observation
        self.partition_id = partition
        self.model_id = canonical_fingerprint(
            {
                "kind": "spectral-submanifold-model",
                "spectral_evidence": evidence.evidence_id,
                "validity_radius": radius,
                "observation": observation,
                "partition": partition,
                "content": array_tree_fingerprint(
                    {"chart": chart, "flow": flow, "exponents": powers}
                )["sha256"],
            }
        )

    def features(self, reduced: ArrayLike, /) -> Array:
        value = jnp.asarray(reduced)
        if value.shape != (self.exponents.shape[1],):
            raise ValueError("SSM reduced coordinate shape is invalid.")
        return jnp.prod(value[None, :] ** self.exponents, axis=-1)

    def decode(self, reduced: ArrayLike, /) -> Array:
        value = jnp.asarray(reduced)
        value = eqx.error_if(
            value,
            jnp.linalg.norm(value) > self.validity_radius,
            "SSM query lies outside the local validity radius.",
        )
        return self.chart_coefficients @ self.features(value)

    def flow(self, reduced: ArrayLike, /) -> Array:
        value = jnp.asarray(reduced)
        return self.flow_coefficients @ self.features(value)

    def invariance_residual(self, reduced: ArrayLike, full_vector_field, /) -> Array:
        value = jnp.asarray(reduced)
        tangent = jax.jacfwd(self.decode)(value)
        return tangent @ self.flow(value) - jnp.asarray(
            full_vector_field(self.decode(value))
        )


def fit_spectral_submanifold(
    embedded_states: ArrayLike,
    reduced_coordinates: ArrayLike,
    reduced_rates: ArrayLike,
    exponents: ArrayLike,
    evidence: SpectralSubmanifoldEvidence,
    /,
    *,
    validity_radius: float,
    observation_contract_id: str,
    partition_id: str,
) -> SpectralSubmanifoldModel:
    states = jnp.asarray(embedded_states)
    reduced = jnp.asarray(reduced_coordinates)
    rates = jnp.asarray(reduced_rates)
    powers = jnp.asarray(exponents, dtype=jnp.int32)
    if (
        states.ndim != 2
        or reduced.ndim != 2
        or rates.shape != reduced.shape
        or powers.ndim != 2
        or powers.shape[1] != reduced.shape[1]
    ):
        raise ValueError("SSM training arrays have incompatible shape.")
    design = jax.vmap(lambda value: jnp.prod(value[None, :] ** powers, axis=-1))(reduced)
    chart_result = solve(LeastSquaresProblem(DenseLinearOperator(design)), states)
    flow_result = solve(LeastSquaresProblem(DenseLinearOperator(design)), rates)
    if not bool(
        np.asarray(jnp.all(chart_result.successful) & jnp.all(flow_result.successful))
    ):
        raise ValueError("SSM polynomial least-squares fit failed.")
    return SpectralSubmanifoldModel(
        jnp.swapaxes(chart_result.value, 0, 1),
        jnp.swapaxes(flow_result.value, 0, 1),
        powers,
        evidence,
        validity_radius=validity_radius,
        observation_contract_id=observation_contract_id,
        partition_id=partition_id,
    )


__all__ = [
    "SpectralSubmanifoldEvidence",
    "SpectralSubmanifoldModel",
    "fit_spectral_submanifold",
]
