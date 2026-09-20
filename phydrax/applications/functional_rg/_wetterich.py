#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._regulators import FunctionalRGStatus, Regulator, ThresholdQuadraturePlan


DerivativeExpansion = Literal["lpa", "lpa-prime"]


def _volume_factor(dimension: float) -> float:
    return 1.0 / (
        2.0 ** (dimension + 1.0)
        * math.pi ** (0.5 * dimension)
        * math.gamma(0.5 * dimension)
    )


def _three_point_derivative_matrices(
    nodes: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    count = nodes.size
    first = np.zeros((count, count), dtype=np.float64)
    second = np.zeros((count, count), dtype=np.float64)
    for row in range(count):
        start = min(max(row - 1, 0), count - 3)
        indices = np.arange(start, start + 3)
        points = nodes[indices]
        x = nodes[row]
        for local, index in enumerate(indices):
            others = np.delete(points, local)
            denominator = (points[local] - others[0]) * (points[local] - others[1])
            first[row, index] = (2.0 * x - others[0] - others[1]) / denominator
            second[row, index] = 2.0 / denominator
    return first, second


class ONPotentialState(StrictModule):
    """Runtime O(N)-invariant potential samples and field renormalization."""

    potential: Array
    wavefunction_renormalization: Array

    def __init__(
        self,
        potential: ArrayLike,
        wavefunction_renormalization: ArrayLike = 1.0,
        /,
    ):
        self.potential = jnp.asarray(potential)
        self.wavefunction_renormalization = jnp.asarray(
            wavefunction_renormalization
        ).reshape(())


class ONPotentialFlowEvaluation(StrictModule):
    beta_potential: Array
    first_derivative: Array
    second_derivative: Array
    goldstone_mass_squared: Array
    radial_mass_squared: Array
    anomalous_dimension: Array
    quadrature_error: Array
    tail_indicator: Array
    finite: Array
    admissible: Array
    status: Array
    prepared_id: str = eqx.field(static=True)


class ONTruncationIdentityEvidence(StrictModule):
    origin_mass_splitting: Array
    component_trace_residual: Array
    finite: Array
    satisfied: Array
    prepared_id: str = eqx.field(static=True)


class ONLocalPotentialPlan(StrictModule, NonTrainableState):
    """Immutable physics and resource policy for O(N) LPA/LPA-prime flow."""

    regulator: Regulator
    threshold: ThresholdQuadraturePlan
    component_count: int = eqx.field(static=True)
    dimension: float = eqx.field(static=True)
    approximation: DerivativeExpansion = eqx.field(static=True)
    maximum_field_nodes: int = eqx.field(static=True)
    volume_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_count: int,
        dimension: float,
        regulator: Regulator,
        threshold: ThresholdQuadraturePlan,
        /,
        *,
        approximation: DerivativeExpansion = "lpa",
        maximum_field_nodes: int = 4096,
    ):
        components = int(component_count)
        dimension_ = float(dimension)
        capacity = int(maximum_field_nodes)
        if not isinstance(regulator, Regulator):
            raise TypeError("regulator must be a Regulator.")
        if not isinstance(threshold, ThresholdQuadraturePlan):
            raise TypeError("threshold must be a ThresholdQuadraturePlan.")
        if (
            components <= 0
            or not 1.0 < dimension_ < 6.0
            or threshold.dimension != dimension_
            or approximation not in ("lpa", "lpa-prime")
            or capacity < 3
        ):
            raise ValueError("O(N) flow physics or field-node capacity is invalid.")
        self.regulator = regulator
        self.threshold = threshold
        self.component_count = components
        self.dimension = dimension_
        self.approximation = approximation
        self.maximum_field_nodes = capacity
        self.volume_factor = _volume_factor(dimension_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "on-local-potential-wetterich-flow",
                "components": components,
                "dimension": dimension_,
                "regulator": regulator.regulator_id,
                "threshold": threshold.plan_id,
                "approximation": approximation,
                "maximum_field_nodes": capacity,
            }
        )

    def prepare(self, field_nodes: ArrayLike, /) -> "PreparedONLocalPotentialFlow":
        return PreparedONLocalPotentialFlow(self, field_nodes)


class PreparedONLocalPotentialFlow(StrictModule, NonTrainableState):
    """Fixed field grid and derivative actions, separate from runtime potentials."""

    __hash__ = object.__hash__

    plan: ONLocalPotentialPlan
    field_nodes: Array
    first_derivative_matrix: Array
    second_derivative_matrix: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ONLocalPotentialPlan, field_nodes: ArrayLike, /):
        if not isinstance(plan, ONLocalPotentialPlan):
            raise TypeError("plan must be ONLocalPotentialPlan.")
        nodes = np.asarray(field_nodes, dtype=np.float64)
        if (
            nodes.ndim != 1
            or nodes.size < 3
            or nodes.size > plan.maximum_field_nodes
            or np.any(~np.isfinite(nodes))
            or nodes[0] < 0.0
            or np.any(np.diff(nodes) <= 0.0)
        ):
            raise ValueError(
                "Field nodes must be finite, increasing, non-negative, and bounded."
            )
        first, second = _three_point_derivative_matrices(nodes)
        self.plan = plan
        self.field_nodes = jnp.asarray(nodes)
        self.first_derivative_matrix = jnp.asarray(first)
        self.second_derivative_matrix = jnp.asarray(second)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-on-local-potential-flow",
                "plan": plan.plan_id,
                "field_nodes": array_tree_fingerprint(nodes),
                "first_derivative": array_tree_fingerprint(first),
                "second_derivative": array_tree_fingerprint(second),
            }
        )

    def anomalous_dimension(
        self, state: ONPotentialState, first: Array, second: Array, /
    ) -> Array:
        if self.plan.approximation == "lpa":
            return jnp.asarray(0.0, dtype=state.potential.dtype)
        minimum_index = jnp.argmin(state.potential)
        kappa = self.field_nodes[minimum_index]
        curvature = second[minimum_index]
        denominator = 1.0 + 2.0 * kappa * curvature
        return (
            16.0
            * self.plan.volume_factor
            / self.plan.dimension
            * kappa
            * curvature**2
            / denominator**2
        )

    def evaluate(self, state: ONPotentialState, /) -> ONPotentialFlowEvaluation:
        if not isinstance(state, ONPotentialState):
            raise TypeError("state must be ONPotentialState.")
        if state.potential.shape != self.field_nodes.shape:
            raise ValueError("Potential samples must match the prepared field grid.")
        first = contract("ij,j->i", self.first_derivative_matrix, state.potential)
        second = contract("ij,j->i", self.second_derivative_matrix, state.potential)
        eta = self.anomalous_dimension(state, first, second)
        goldstone_mass = first
        radial_mass = first + 2.0 * self.field_nodes * second
        goldstone = self.plan.threshold.evaluate(self.plan.regulator, goldstone_mass, eta)
        radial = self.plan.threshold.evaluate(self.plan.regulator, radial_mass, eta)
        canonical = (
            -self.plan.dimension * state.potential
            + (self.plan.dimension - 2.0 + eta) * self.field_nodes * first
        )
        loop = (
            2.0
            * self.plan.volume_factor
            * ((self.plan.component_count - 1) * goldstone.value + radial.value)
        )
        beta = canonical + loop
        finite = (
            jnp.all(jnp.isfinite(state.potential))
            & jnp.isfinite(state.wavefunction_renormalization)
            & (state.wavefunction_renormalization > 0.0)
            & jnp.isfinite(eta)
            & jnp.all(goldstone.finite)
            & jnp.all(radial.finite)
            & jnp.all(jnp.isfinite(beta))
        )
        admissible = finite & jnp.all(goldstone.admissible) & jnp.all(radial.admissible)
        status = jnp.where(
            admissible,
            int(FunctionalRGStatus.SUCCESS),
            jnp.where(
                finite,
                int(FunctionalRGStatus.POLE_ENCOUNTERED),
                int(FunctionalRGStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        return ONPotentialFlowEvaluation(
            beta,
            first,
            second,
            goldstone_mass,
            radial_mass,
            eta,
            jnp.maximum(goldstone.quadrature_error, radial.quadrature_error),
            jnp.maximum(goldstone.tail_indicator, radial.tail_indicator),
            finite,
            admissible,
            status,
            self.prepared_id,
        )

    def truncation_identity(
        self, state: ONPotentialState, /
    ) -> ONTruncationIdentityEvidence:
        evaluation = self.evaluate(state)
        splitting = (
            evaluation.radial_mass_squared[0] - evaluation.goldstone_mass_squared[0]
        )
        expected_trace = (
            self.plan.component_count - 1
        ) * evaluation.goldstone_mass_squared[0] + evaluation.radial_mass_squared[0]
        trace_from_derivatives = (
            self.plan.component_count * evaluation.first_derivative[0]
        )
        trace_residual = expected_trace - trace_from_derivatives
        finite = jnp.isfinite(splitting) & jnp.isfinite(trace_residual)
        scale = 1.0 + jnp.abs(evaluation.first_derivative[0])
        satisfied = finite & (
            jnp.abs(splitting) <= 64.0 * jnp.finfo(scale.dtype).eps * scale
        )
        return ONTruncationIdentityEvidence(
            splitting,
            trace_residual,
            finite,
            satisfied,
            self.prepared_id,
        )


class SchemeRefinementEvidence(StrictModule):
    coarse_beta: Array
    refined_beta: Array
    absolute_difference: Array
    relative_difference: Array
    coarse_indicator: Array
    refined_indicator: Array
    finite: Array
    accepted: Array
    coarse_id: str = eqx.field(static=True)
    refined_id: str = eqx.field(static=True)


def evaluate_scheme_refinement(
    coarse: PreparedONLocalPotentialFlow,
    refined: PreparedONLocalPotentialFlow,
    state: ONPotentialState,
    /,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> SchemeRefinementEvidence:
    if not isinstance(coarse, PreparedONLocalPotentialFlow) or not isinstance(
        refined, PreparedONLocalPotentialFlow
    ):
        raise TypeError("Scheme comparison requires prepared O(N) flows.")
    if (
        coarse.plan.component_count != refined.plan.component_count
        or coarse.plan.dimension != refined.plan.dimension
        or coarse.plan.approximation != refined.plan.approximation
        or coarse.field_nodes.shape != refined.field_nodes.shape
        or not np.array_equal(
            np.asarray(coarse.field_nodes), np.asarray(refined.field_nodes)
        )
    ):
        raise ValueError(
            "Scheme refinement must compare the same physical truncation and grid."
        )
    absolute = float(absolute_tolerance)
    relative = float(relative_tolerance)
    if (
        not np.isfinite(absolute)
        or not np.isfinite(relative)
        or absolute <= 0.0
        or relative <= 0.0
    ):
        raise ValueError("Scheme refinement tolerances must be finite and positive.")
    coarse_evaluation = coarse.evaluate(state)
    refined_evaluation = refined.evaluate(state)
    difference = jnp.max(
        jnp.abs(refined_evaluation.beta_potential - coarse_evaluation.beta_potential)
    )
    scale = jnp.maximum(jnp.max(jnp.abs(refined_evaluation.beta_potential)), absolute)
    relative_difference = difference / scale
    coarse_indicator = jnp.max(
        coarse_evaluation.quadrature_error + coarse_evaluation.tail_indicator
    )
    refined_indicator = jnp.max(
        refined_evaluation.quadrature_error + refined_evaluation.tail_indicator
    )
    finite = (
        coarse_evaluation.finite & refined_evaluation.finite & jnp.isfinite(difference)
    )
    accepted = finite & (
        (difference <= absolute + relative * scale)
        | (refined_indicator <= coarse_indicator)
    )
    return SchemeRefinementEvidence(
        coarse_evaluation.beta_potential,
        refined_evaluation.beta_potential,
        difference,
        relative_difference,
        coarse_indicator,
        refined_indicator,
        finite,
        accepted,
        coarse.prepared_id,
        refined.prepared_id,
    )


__all__ = [
    "DerivativeExpansion",
    "ONLocalPotentialPlan",
    "ONPotentialFlowEvaluation",
    "ONPotentialState",
    "ONTruncationIdentityEvidence",
    "PreparedONLocalPotentialFlow",
    "SchemeRefinementEvidence",
    "evaluate_scheme_refinement",
]
