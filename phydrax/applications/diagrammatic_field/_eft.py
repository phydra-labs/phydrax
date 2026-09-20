#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    prepare,
    PreparedLinearSolve,
    solve,
)
from ._core import FieldSpec


class EFTOperator(StrictModule, NonTrainableState):
    """One local EFT operator with field content and canonical dimension."""

    name: str = eqx.field(static=True)
    fields: tuple[FieldSpec, ...]
    canonical_dimension: float = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)
    quantum_numbers: tuple[tuple[str, int], ...] = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        fields: tuple[FieldSpec, ...],
        canonical_dimension: float,
        /,
        *,
        derivative_order: int = 0,
        quantum_numbers: tuple[tuple[str, int], ...] = (),
    ):
        name_ = str(name).strip()
        fields_ = tuple(fields)
        dimension = float(canonical_dimension)
        derivative = int(derivative_order)
        numbers = tuple(sorted((str(key), int(value)) for key, value in quantum_numbers))
        if not name_:
            raise ValueError("EFT operator name must be non-empty.")
        if not fields_ or any(not isinstance(field, FieldSpec) for field in fields_):
            raise ValueError("EFT operators need a non-empty tuple of FieldSpec values.")
        if not np.isfinite(dimension) or dimension < 0.0 or derivative < 0:
            raise ValueError("EFT dimensions and derivative orders must be non-negative.")
        if any(not key for key, _ in numbers) or len({key for key, _ in numbers}) != len(
            numbers
        ):
            raise ValueError("EFT quantum-number keys must be unique and non-empty.")
        self.name = name_
        self.fields = fields_
        self.canonical_dimension = dimension
        self.derivative_order = derivative
        self.quantum_numbers = numbers
        self.operator_id = canonical_fingerprint(
            {
                "kind": "eft-operator",
                "name": name_,
                "fields": [field.field_id for field in fields_],
                "canonical_dimension": dimension,
                "derivative_order": derivative,
                "quantum_numbers": numbers,
            }
        )


class EFTOperatorBasis(StrictModule, NonTrainableState):
    """Ordered, duplicate-free finite EFT operator basis."""

    operators: tuple[EFTOperator, ...]
    spacetime_dimension: int = eqx.field(static=True)
    maximum_operators: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: tuple[EFTOperator, ...],
        /,
        *,
        spacetime_dimension: int = 4,
        maximum_operators: int = 4_096,
    ):
        operators_ = tuple(operators)
        dimension, maximum = int(spacetime_dimension), int(maximum_operators)
        if not operators_ or any(
            not isinstance(item, EFTOperator) for item in operators_
        ):
            raise ValueError("operators must contain EFTOperator values.")
        if len(operators_) > maximum or maximum <= 0:
            raise ValueError("EFT basis exceeds maximum_operators before allocation.")
        if len({item.operator_id for item in operators_}) != len(operators_):
            raise ValueError("EFT operator identities must be unique.")
        if dimension <= 0:
            raise ValueError("spacetime_dimension must be positive.")
        canonical = tuple(
            sorted(
                operators_,
                key=lambda item: (
                    item.canonical_dimension,
                    item.derivative_order,
                    item.name,
                    item.operator_id,
                ),
            )
        )
        self.operators = canonical
        self.spacetime_dimension = dimension
        self.maximum_operators = maximum
        self.basis_id = canonical_fingerprint(
            {
                "kind": "eft-operator-basis",
                "operators": [item.operator_id for item in canonical],
                "spacetime_dimension": dimension,
            }
        )

    @property
    def size(self) -> int:
        return len(self.operators)


class PowerCountingRule(StrictModule, NonTrainableState):
    """Expansion scale and loop suppression kept separate from coefficients."""

    expansion_parameter: Array
    reference_scale: Array
    loop_factor: Array
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        expansion_parameter: float,
        reference_scale: float,
        /,
        *,
        loop_factor: float = 1.0 / (16.0 * math.pi * math.pi),
    ):
        expansion, scale, loop = (
            float(expansion_parameter),
            float(reference_scale),
            float(loop_factor),
        )
        if (
            not all(np.isfinite(value) for value in (expansion, scale, loop))
            or not 0.0 < expansion < 1.0
            or scale <= 0.0
            or loop <= 0.0
        ):
            raise ValueError(
                "Power counting needs 0 < expansion_parameter < 1 and positive scales."
            )
        self.expansion_parameter = jnp.asarray(expansion)
        self.reference_scale = jnp.asarray(scale)
        self.loop_factor = jnp.asarray(loop)
        self.rule_id = canonical_fingerprint(
            {
                "kind": "eft-power-counting",
                "expansion_parameter": expansion,
                "reference_scale": scale,
                "loop_factor": loop,
            }
        )

    def suppression(self, power: int, /, *, loops: int = 0) -> Array:
        power_, loops_ = int(power), int(loops)
        if power_ < 0 or loops_ < 0:
            raise ValueError(
                "Power-counting powers and loop counts must be non-negative."
            )
        return self.expansion_parameter**power_ * self.loop_factor**loops_


class WilsonCoefficientState(StrictModule, NonTrainableState):
    coefficients: Array
    statistical_covariance: Array
    scale: Array
    statistical_uncertainty_available: Array
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        statistical_covariance: ArrayLike,
        scale: ArrayLike,
        statistical_uncertainty_available: ArrayLike,
        basis_id: str,
        /,
    ):
        values = jnp.asarray(coefficients)
        covariance = jnp.asarray(statistical_covariance)
        if values.ndim != 1 or covariance.shape != (values.size, values.size):
            raise ValueError(
                "Wilson coefficients and covariance have incompatible shapes."
            )
        self.coefficients = values
        self.statistical_covariance = covariance
        self.scale = jnp.asarray(scale).reshape(())
        self.statistical_uncertainty_available = jnp.asarray(
            statistical_uncertainty_available, dtype=jnp.bool_
        ).reshape(())
        self.basis_id = str(basis_id)


class EFTMatchingEvidence(StrictModule, NonTrainableState):
    residual_norm: Array
    coefficient_phase: Array
    coefficient_real_sign: Array
    finite: Array
    successful: Array
    status: Array


class EFTMatchingResult(StrictModule, NonTrainableState):
    state: WilsonCoefficientState
    evidence: EFTMatchingEvidence
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class EFTMatchingPlan(StrictModule, NonTrainableState):
    """Immutable full-rank finite matching system and allocation guard."""

    basis: EFTOperatorBasis
    matching_matrix: Array
    maximum_matrix_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: EFTOperatorBasis,
        matching_matrix: ArrayLike,
        /,
        *,
        maximum_matrix_elements: int = 1_000_000,
    ):
        if not isinstance(basis, EFTOperatorBasis):
            raise TypeError("basis must be an EFTOperatorBasis.")
        matrix = np.asarray(matching_matrix, dtype=np.complex128)
        maximum = int(maximum_matrix_elements)
        if matrix.shape != (basis.size, basis.size):
            raise ValueError("matching_matrix must be square with the EFT basis size.")
        if maximum <= 0 or matrix.size > maximum:
            raise ValueError("Matching system exceeds maximum_matrix_elements.")
        if not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) != basis.size:
            raise ValueError("matching_matrix must be finite and full rank.")
        self.basis = basis
        self.matching_matrix = jnp.asarray(matrix)
        self.maximum_matrix_elements = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "eft-matching-plan",
                "basis": basis.basis_id,
                "matrix": array_tree_fingerprint(matrix),
                "maximum_matrix_elements": maximum,
            }
        )

    def prepare(self, /) -> "PreparedEFTMatching":
        operator = DenseLinearOperator(self.matching_matrix)
        prepared = prepare(
            LinearSystem(operator),
            LinearSolvePolicy(DenseLU()),
        )
        inverse_result = solve(
            prepared,
            jnp.eye(self.basis.size, dtype=self.matching_matrix.dtype),
        )
        inverse = inverse_result.value
        prepared_id = canonical_fingerprint(
            {"kind": "prepared-eft-matching", "plan": self.plan_id}
        )
        return PreparedEFTMatching(
            self.basis,
            self.matching_matrix,
            prepared,
            inverse,
            self.plan_id,
            prepared_id,
        )


class PreparedEFTMatching(StrictModule, NonTrainableState):
    basis: EFTOperatorBasis
    matching_matrix: Array
    prepared_solve: PreparedLinearSolve
    inverse_matching: Array
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: EFTOperatorBasis,
        matching_matrix: Array,
        prepared_solve: PreparedLinearSolve,
        inverse_matching: Array,
        plan_id: str,
        prepared_id: str,
        /,
    ):
        self.basis = basis
        self.matching_matrix = matching_matrix
        self.prepared_solve = prepared_solve
        self.inverse_matching = inverse_matching
        self.plan_id = str(plan_id)
        self.prepared_id = str(prepared_id)

    def match(
        self,
        amplitudes: ArrayLike,
        /,
        *,
        scale: ArrayLike,
        statistical_covariance: ArrayLike | None = None,
    ) -> EFTMatchingResult:
        amplitudes_ = jnp.asarray(amplitudes, dtype=self.matching_matrix.dtype)
        if amplitudes_.shape != (self.basis.size,):
            raise ValueError("amplitudes must have one value per matching condition.")
        coefficients_result = solve(self.prepared_solve, amplitudes_)
        coefficients = coefficients_result.value
        available = statistical_covariance is not None
        source_covariance = (
            jnp.zeros((self.basis.size, self.basis.size), dtype=coefficients.dtype)
            if statistical_covariance is None
            else jnp.asarray(statistical_covariance, dtype=coefficients.dtype)
        )
        if source_covariance.shape != (self.basis.size, self.basis.size):
            raise ValueError("statistical_covariance must match the matching conditions.")
        covariance = (
            self.inverse_matching @ source_covariance @ jnp.conj(self.inverse_matching.T)
        )
        residual = self.matching_matrix @ coefficients - amplitudes_
        residual_norm = jnp.linalg.norm(residual)
        absolute = jnp.abs(coefficients)
        phase = jnp.where(absolute > 0.0, coefficients / absolute, 1.0 + 0.0j)
        real_sign = jnp.where(
            jnp.abs(coefficients.imag) <= 64.0 * jnp.finfo(coefficients.real.dtype).eps,
            jnp.sign(coefficients.real),
            0.0,
        )
        finite = (
            jnp.all(jnp.isfinite(coefficients))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.isfinite(residual_norm)
        )
        tolerance = (
            256.0
            * jnp.finfo(coefficients.real.dtype).eps
            * jnp.maximum(1.0, jnp.linalg.norm(amplitudes_))
        )
        successful = (
            finite & coefficients_result.successful & (residual_norm <= tolerance)
        )
        state = WilsonCoefficientState(
            coefficients,
            covariance,
            scale,
            jnp.asarray(available),
            self.basis.basis_id,
        )
        evidence = EFTMatchingEvidence(
            residual_norm,
            phase,
            real_sign,
            finite,
            successful,
            jnp.where(successful, 0, 1).astype(jnp.int32),
        )
        return EFTMatchingResult(state, evidence, self.plan_id, self.prepared_id)


class RGRunningEvidence(StrictModule, NonTrainableState):
    integration_error_estimate: Array
    finite: Array
    successful: Array
    status: Array


class RGRunningResult(StrictModule, NonTrainableState):
    state: WilsonCoefficientState
    evolution_operator: Array
    evidence: RGRunningEvidence
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class RGFlowPlan(StrictModule, NonTrainableState):
    """Immutable anomalous-dimension matrix and bounded RK4 policy."""

    basis: EFTOperatorBasis
    anomalous_dimension: Array
    maximum_steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: EFTOperatorBasis,
        anomalous_dimension: ArrayLike,
        /,
        *,
        maximum_steps: int = 65_536,
    ):
        if not isinstance(basis, EFTOperatorBasis):
            raise TypeError("basis must be an EFTOperatorBasis.")
        gamma = np.asarray(anomalous_dimension, dtype=np.complex128)
        maximum = int(maximum_steps)
        if gamma.shape != (basis.size, basis.size) or not np.all(np.isfinite(gamma)):
            raise ValueError("anomalous_dimension must be a finite basis-square matrix.")
        if maximum <= 0:
            raise ValueError("maximum_steps must be positive.")
        self.basis = basis
        self.anomalous_dimension = jnp.asarray(gamma)
        self.maximum_steps = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "eft-rg-flow-plan",
                "basis": basis.basis_id,
                "anomalous_dimension": array_tree_fingerprint(gamma),
                "maximum_steps": maximum,
            }
        )

    def prepare(
        self,
        initial_scale: float,
        final_scale: float,
        /,
        *,
        steps: int,
    ) -> "PreparedRGFlow":
        initial, final, count = float(initial_scale), float(final_scale), int(steps)
        if (
            not np.isfinite(initial)
            or not np.isfinite(final)
            or initial <= 0.0
            or final <= 0.0
        ):
            raise ValueError("RG scales must be finite and positive.")
        if count <= 0 or count > self.maximum_steps:
            raise ValueError("RG steps violate the prepared maximum_steps capacity.")
        generator = DenseLinearOperator(jnp.swapaxes(self.anomalous_dimension, -1, -2))
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-eft-rg-flow",
                "plan": self.plan_id,
                "initial_scale": initial,
                "final_scale": final,
                "steps": count,
            }
        )
        return PreparedRGFlow(
            self.basis,
            generator,
            jnp.asarray(initial),
            jnp.asarray(final),
            count,
            self.plan_id,
            prepared_id,
        )


class PreparedRGFlow(StrictModule, NonTrainableState):
    basis: EFTOperatorBasis
    generator: DenseLinearOperator
    initial_scale: Array
    final_scale: Array
    steps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: EFTOperatorBasis,
        generator: DenseLinearOperator,
        initial_scale: Array,
        final_scale: Array,
        steps: int,
        plan_id: str,
        prepared_id: str,
        /,
    ):
        self.basis = basis
        self.generator = generator
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.steps = int(steps)
        self.plan_id = str(plan_id)
        self.prepared_id = str(prepared_id)

    def run(self, state: WilsonCoefficientState, /) -> RGRunningResult:
        if state.basis_id != self.basis.basis_id:
            raise ValueError("Wilson state belongs to a different EFT basis.")
        coefficients = eqx.error_if(
            state.coefficients,
            ~jnp.isclose(state.scale, self.initial_scale),
            "Wilson state scale does not match the prepared initial scale.",
        ).astype(self.generator.matrix.dtype)
        logarithmic_interval = jnp.log(self.final_scale / self.initial_scale)
        step_size = logarithmic_interval / self.steps
        identity = jnp.eye(self.basis.size, dtype=coefficients.dtype)

        def rk4_step(_, evolution):
            k1 = self.generator.mv_block(evolution)
            k2 = self.generator.mv_block(evolution + 0.5 * step_size * k1)
            k3 = self.generator.mv_block(evolution + 0.5 * step_size * k2)
            k4 = self.generator.mv_block(evolution + step_size * k3)
            return evolution + step_size / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        evolution = jax.lax.fori_loop(0, self.steps, rk4_step, identity)
        evolved = contract("ij,j->i", evolution, coefficients, backend="jax")
        covariance = contract(
            "ij,jk,lk->il",
            evolution,
            state.statistical_covariance.astype(self.generator.matrix.dtype),
            jnp.conj(evolution),
            backend="jax",
        )
        last_derivative = self.generator.mv_block(evolution)
        error_estimate = (
            jnp.linalg.norm(last_derivative)
            * jnp.abs(step_size) ** 5
            / jnp.maximum(1.0, jnp.linalg.norm(evolution))
        )
        finite = (
            jnp.all(jnp.isfinite(evolved))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.isfinite(error_estimate)
        )
        successful = finite
        next_state = WilsonCoefficientState(
            evolved,
            covariance,
            self.final_scale,
            state.statistical_uncertainty_available,
            self.basis.basis_id,
        )
        evidence = RGRunningEvidence(
            error_estimate,
            finite,
            successful,
            jnp.where(successful, 0, 1).astype(jnp.int32),
        )
        return RGRunningResult(
            next_state,
            evolution,
            evidence,
            self.plan_id,
            self.prepared_id,
        )


class EFTPredictionEvidence(StrictModule, NonTrainableState):
    phase: Array
    real_sign: Array
    statistical_uncertainty_available: Array
    finite: Array
    successful: Array
    status: Array


class EFTPrediction(StrictModule, NonTrainableState):
    value: Array
    statistical_uncertainty: Array
    truncation_uncertainty: Array
    combined_uncertainty: Array
    evidence: EFTPredictionEvidence
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class EFTObservablePlan(StrictModule, NonTrainableState):
    """Observable contraction with independent statistical and truncation errors."""

    basis: EFTOperatorBasis
    matrix_elements: Array
    power_counting: PowerCountingRule
    first_omitted_power: int = eqx.field(static=True)
    omitted_coefficient_scale: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: EFTOperatorBasis,
        matrix_elements: ArrayLike,
        power_counting: PowerCountingRule,
        /,
        *,
        first_omitted_power: int,
        omitted_coefficient_scale: float = 1.0,
    ):
        if not isinstance(basis, EFTOperatorBasis) or not isinstance(
            power_counting, PowerCountingRule
        ):
            raise TypeError(
                "EFT observable plans require a basis and power-counting rule."
            )
        elements = np.asarray(matrix_elements, dtype=np.complex128)
        power = int(first_omitted_power)
        coefficient_scale = float(omitted_coefficient_scale)
        if elements.shape != (basis.size,) or not np.all(np.isfinite(elements)):
            raise ValueError("matrix_elements must be one finite value per EFT operator.")
        if power <= 0 or not np.isfinite(coefficient_scale) or coefficient_scale < 0.0:
            raise ValueError(
                "Omitted power must be positive and coefficient scale non-negative."
            )
        self.basis = basis
        self.matrix_elements = jnp.asarray(elements)
        self.power_counting = power_counting
        self.first_omitted_power = power
        self.omitted_coefficient_scale = coefficient_scale
        self.plan_id = canonical_fingerprint(
            {
                "kind": "eft-observable-plan",
                "basis": basis.basis_id,
                "matrix_elements": array_tree_fingerprint(elements),
                "power_counting": power_counting.rule_id,
                "first_omitted_power": power,
                "omitted_coefficient_scale": coefficient_scale,
            }
        )

    def prepare(self, /) -> "PreparedEFTObservable":
        return PreparedEFTObservable(
            self,
            canonical_fingerprint(
                {"kind": "prepared-eft-observable", "plan": self.plan_id}
            ),
        )


class PreparedEFTObservable(StrictModule, NonTrainableState):
    """Prepared observable contraction with a JAX-compatible runtime."""

    plan: EFTObservablePlan
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: EFTObservablePlan, prepared_id: str, /):
        self.plan = plan
        self.prepared_id = str(prepared_id)

    def evaluate(self, state: WilsonCoefficientState, /) -> EFTPrediction:
        if state.basis_id != self.plan.basis.basis_id:
            raise ValueError("Wilson state belongs to a different EFT basis.")
        value = contract(
            "i,i->",
            self.plan.matrix_elements,
            state.coefficients,
            backend="jax",
        )
        variance = contract(
            "i,ij,j->",
            self.plan.matrix_elements,
            state.statistical_covariance,
            jnp.conj(self.plan.matrix_elements),
            backend="jax",
        ).real
        statistical = jnp.sqrt(jnp.maximum(0.0, variance))
        truncation = (
            self.plan.omitted_coefficient_scale
            * jnp.sum(jnp.abs(self.plan.matrix_elements))
            * self.plan.power_counting.suppression(self.plan.first_omitted_power)
        )
        combined = jnp.sqrt(statistical * statistical + truncation * truncation)
        absolute = jnp.abs(value)
        phase = jnp.where(absolute > 0.0, value / absolute, 1.0 + 0.0j)
        real_sign = jnp.where(
            jnp.abs(value.imag) <= 64.0 * jnp.finfo(value.real.dtype).eps,
            jnp.sign(value.real),
            0.0,
        )
        finite = (
            jnp.isfinite(value.real)
            & jnp.isfinite(value.imag)
            & jnp.isfinite(statistical)
            & jnp.isfinite(truncation)
        )
        successful = finite
        evidence = EFTPredictionEvidence(
            phase,
            real_sign,
            state.statistical_uncertainty_available,
            finite,
            successful,
            jnp.where(successful, 0, 1).astype(jnp.int32),
        )
        return EFTPrediction(
            value,
            statistical,
            truncation,
            combined,
            evidence,
            self.plan.plan_id,
            self.prepared_id,
        )


__all__ = [
    "EFTMatchingEvidence",
    "EFTMatchingPlan",
    "EFTMatchingResult",
    "EFTObservablePlan",
    "EFTOperator",
    "EFTOperatorBasis",
    "EFTPrediction",
    "EFTPredictionEvidence",
    "PowerCountingRule",
    "PreparedEFTMatching",
    "PreparedEFTObservable",
    "PreparedRGFlow",
    "RGFlowPlan",
    "RGRunningEvidence",
    "RGRunningResult",
    "WilsonCoefficientState",
]
