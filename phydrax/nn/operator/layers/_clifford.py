#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike

from phydrax._doc import DOC_KEY0
from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.metrix.clifford import (
    CliffordProductPlan,
    extract_layout,
    MetricIsometryAuditSet,
    prepare_product,
)
from phydrax.nn.operator.representations import (
    CliffordGradeFeatures,
    CliffordGradeRepresentation,
)


class CliffordEquivarianceCertificate(StrictModule, NonTrainableState):
    """By-construction equivariance claim for one Clifford neural primitive."""

    algebra_id: str = eqx.field(static=True)
    input_representation_id: str = eqx.field(static=True)
    output_representation_id: str = eqx.field(static=True)
    construction: str = eqx.field(static=True)
    group_scope: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        algebra_id: str,
        input_representation_id: str,
        output_representation_id: str,
        construction: str,
        group_scope: str = "orthogonal-euclidean",
    ):
        values = tuple(
            str(value)
            for value in (
                algebra_id,
                input_representation_id,
                output_representation_id,
                construction,
                group_scope,
            )
        )
        if any(not value for value in values):
            raise ValueError(
                "Clifford equivariance certificate fields must be non-empty."
            )
        self.algebra_id = values[0]
        self.input_representation_id = values[1]
        self.output_representation_id = values[2]
        self.construction = values[3]
        self.group_scope = values[4]
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "clifford-equivariance-certificate-v1",
                "algebra": values[0],
                "input": values[1],
                "output": values[2],
                "construction": values[3],
                "group_scope": values[4],
            }
        )


class CliffordGradeLinear(eqx.Module):
    """Equivariant scalar channel mixing independently within each complete grade."""

    __hash__ = object.__hash__

    input_representation: CliffordGradeRepresentation
    output_representation: CliffordGradeRepresentation
    weights: tuple[Array | None, ...]
    scalar_bias: Array | None
    certificate: CliffordEquivarianceCertificate

    def __init__(
        self,
        input_representation: CliffordGradeRepresentation,
        output_representation: CliffordGradeRepresentation,
        /,
        *,
        use_scalar_bias: bool = True,
        key: Array = DOC_KEY0,
    ):
        if not isinstance(
            input_representation, CliffordGradeRepresentation
        ) or not isinstance(output_representation, CliffordGradeRepresentation):
            raise TypeError("Clifford grade linear layers require grade representations.")
        input_representation.algebra.require_compatible(output_representation.algebra)
        keys = iter(jr.split(key, input_representation.algebra.dimension + 2))
        weights = []
        for input_count, output_count in zip(
            input_representation.multiplicities,
            output_representation.multiplicities,
        ):
            weight_key = next(keys)
            if input_count == 0 or output_count == 0:
                weights.append(None)
            else:
                weights.append(
                    jr.normal(weight_key, (output_count, input_count))
                    / math.sqrt(float(input_count))
                )
        bias_key = next(keys)
        scalar_bias = (
            0.01
            * jr.normal(
                bias_key,
                (output_representation.multiplicities[0],),
            )
            if use_scalar_bias and output_representation.multiplicities[0] > 0
            else None
        )
        self.input_representation = input_representation
        self.output_representation = output_representation
        self.weights = tuple(weights)
        self.scalar_bias = scalar_bias
        self.certificate = CliffordEquivarianceCertificate(
            algebra_id=input_representation.algebra.algebra_id,
            input_representation_id=input_representation.representation_id,
            output_representation_id=output_representation.representation_id,
            construction="grade-wise-scalar-channel-linear",
        )

    def __call__(self, values: Array, /) -> Array:
        features = self.input_representation.split(values)
        leading = jnp.asarray(values).shape[:-1]
        output = []
        for grade, (grade_values, weight, output_count, layout) in enumerate(
            zip(
                features.grades,
                self.weights,
                self.output_representation.multiplicities,
                self.output_representation.grade_layouts,
            )
        ):
            if weight is None:
                mixed = jnp.zeros(
                    leading + (output_count, layout.blade_count),
                    dtype=jnp.asarray(values).dtype,
                )
            else:
                mixed = jnp.einsum("oi,...ib->...ob", weight, grade_values)
            if grade == 0 and self.scalar_bias is not None:
                mixed = mixed + self.scalar_bias.reshape(
                    (1,) * len(leading) + self.scalar_bias.shape + (1,)
                )
            output.append(mixed)
        return self.output_representation.join(CliffordGradeFeatures(tuple(output)))


class CliffordGeometricProductLayer(eqx.Module):
    """Elementwise channel interactions parameterized by product grade triples."""

    __hash__ = object.__hash__

    representation: CliffordGradeRepresentation
    context: CliffordGradeLinear
    output: CliffordGradeLinear
    pair_grades: tuple[tuple[int, int], ...]
    pair_plans: tuple[CliffordProductPlan, ...]
    routes: tuple[tuple[int, int], ...]
    route_weights: Array
    certificate: CliffordEquivarianceCertificate

    def __init__(
        self,
        representation: CliffordGradeRepresentation,
        /,
        *,
        key: Array = DOC_KEY0,
    ):
        if not isinstance(representation, CliffordGradeRepresentation):
            raise TypeError("representation must be CliffordGradeRepresentation.")
        channels = representation.uniform_multiplicity
        if channels is None or channels <= 0:
            raise ValueError(
                "Clifford product layer requires one common positive latent multiplicity."
            )
        context_key, output_key, weight_key = jr.split(key, 3)
        context = CliffordGradeLinear(
            representation,
            representation,
            use_scalar_bias=False,
            key=context_key,
        )
        output = CliffordGradeLinear(
            representation,
            representation,
            use_scalar_bias=True,
            key=output_key,
        )
        pair_grades = []
        pair_plans = []
        routes = []
        active_grades = tuple(
            grade
            for grade, multiplicity in enumerate(representation.multiplicities)
            if multiplicity > 0
        )
        for left_grade in active_grades:
            for right_grade in active_grades:
                plan = prepare_product(
                    representation.algebra,
                    representation.grade_layouts[left_grade],
                    representation.grade_layouts[right_grade],
                    backend="sparse",
                )
                pair_index = len(pair_plans)
                pair_grades.append((left_grade, right_grade))
                pair_plans.append(plan)
                for output_grade in plan.output_layout.grade_set:
                    if representation.multiplicities[output_grade] > 0:
                        routes.append((pair_index, output_grade))
        if not routes:
            raise ValueError(
                "Clifford product layer has no supported grade interactions."
            )
        route_weights = jr.normal(weight_key, (len(routes), channels)) / math.sqrt(
            float(len(routes))
        )
        self.representation = representation
        self.context = context
        self.output = output
        self.pair_grades = tuple(pair_grades)
        self.pair_plans = tuple(pair_plans)
        self.routes = tuple(routes)
        self.route_weights = route_weights
        self.certificate = CliffordEquivarianceCertificate(
            algebra_id=representation.algebra.algebra_id,
            input_representation_id=representation.representation_id,
            output_representation_id=representation.representation_id,
            construction="grade-projected-geometric-product-polynomial",
        )

    def __call__(self, values: Array, /) -> Array:
        source = self.representation.split(values)
        context = self.representation.split(self.context(values))
        leading = jnp.asarray(values).shape[:-1]
        output = [
            jnp.zeros(
                leading + (multiplicity, layout.blade_count),
                dtype=jnp.asarray(values).dtype,
            )
            for multiplicity, layout in zip(
                self.representation.multiplicities,
                self.representation.grade_layouts,
            )
        ]
        pair_values = []
        for (left_grade, right_grade), plan in zip(
            self.pair_grades,
            self.pair_plans,
        ):
            pair_values.append(
                plan(source.grades[left_grade], context.grades[right_grade])
            )
        for route, (pair_index, output_grade) in enumerate(self.routes):
            plan = self.pair_plans[pair_index]
            target_layout = self.representation.grade_layouts[output_grade]
            selected = extract_layout(
                pair_values[pair_index],
                plan.output_layout,
                target_layout,
            )
            weight = self.route_weights[route].reshape((1,) * len(leading) + (-1, 1))
            output[output_grade] = output[output_grade] + weight * selected
        packed = self.representation.join(CliffordGradeFeatures(tuple(output)))
        return self.output(packed)


def clifford_gated_activation(
    values: Array,
    representation: CliffordGradeRepresentation,
    /,
) -> Array:
    """Apply scalar activation and invariant Euclidean grade gates."""
    if not representation.algebra.positive_definite:
        raise ValueError(
            "Initial Clifford gated activation requires a positive-definite algebra."
        )
    features = representation.split(values)
    output = []
    for grade, grade_values in enumerate(features.grades):
        if grade == 0:
            output.append(jnn.gelu(grade_values))
            continue
        norm = jnp.sqrt(jnp.sum(grade_values**2, axis=-1, keepdims=True) + 1e-12)
        output.append(grade_values * jnn.sigmoid(norm))
    return representation.join(CliffordGradeFeatures(tuple(output)))


class CliffordEquivarianceAuditReport(StrictModule, NonTrainableState):
    """Sampled action-commutation evidence for one Clifford neural map."""

    finite: Array
    valid: Array
    maximum_residual: Array
    root_mean_square_residual: Array
    reference_scale: Array
    tolerance: Array
    audit_set_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        finite: ArrayLike,
        maximum_residual: ArrayLike,
        root_mean_square_residual: ArrayLike,
        reference_scale: ArrayLike,
        tolerance: ArrayLike,
        audit_set_id: str,
    ):
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.maximum_residual = jnp.asarray(maximum_residual).reshape(())
        self.root_mean_square_residual = jnp.asarray(root_mean_square_residual).reshape(
            ()
        )
        self.reference_scale = jnp.asarray(reference_scale).reshape(())
        self.tolerance = jnp.asarray(tolerance).reshape(())
        self.valid = self.finite & (self.maximum_residual <= self.tolerance)
        self.audit_set_id = str(audit_set_id)
        self.report_id = canonical_fingerprint(
            {
                "kind": "clifford-equivariance-audit-v1",
                "audit_set": self.audit_set_id,
                "maximum_residual": float(self.maximum_residual),
                "rms_residual": float(self.root_mean_square_residual),
                "reference_scale": float(self.reference_scale),
                "tolerance": float(self.tolerance),
            }
        )


def audit_clifford_equivariance(
    function: Callable[[Array], Array],
    values: Array,
    input_representation: CliffordGradeRepresentation,
    output_representation: CliffordGradeRepresentation,
    actions: MetricIsometryAuditSet,
    /,
    *,
    tolerance: float = 1e-9,
) -> CliffordEquivarianceAuditReport:
    """Audit one neural map against independent metric-isometry actions."""
    if not callable(function):
        raise TypeError("function must be callable.")
    input_representation.algebra.require_compatible(output_representation.algebra)
    input_representation.algebra.require_compatible(actions.algebra)
    residuals = []
    reference = []
    for action in actions.actions:
        transformed_input = input_representation.transform(values, action)
        left = jnp.asarray(function(transformed_input))
        right = output_representation.transform(
            jnp.asarray(function(values)),
            action,
        )
        residuals.append(jnp.ravel(left - right))
        reference.append(jnp.ravel(right))
    residual = jnp.concatenate(residuals)
    reference_values = jnp.concatenate(reference)
    scale = jnp.maximum(jnp.max(jnp.abs(reference_values)), 1.0)
    normalized = jnp.abs(residual) / scale
    finite = jnp.all(jnp.isfinite(residual)) & jnp.all(jnp.isfinite(reference_values))
    return CliffordEquivarianceAuditReport(
        finite=finite,
        maximum_residual=jnp.max(normalized),
        root_mean_square_residual=jnp.sqrt(jnp.mean(normalized**2)),
        reference_scale=scale,
        tolerance=tolerance,
        audit_set_id=actions.audit_set_id,
    )


__all__ = [
    "audit_clifford_equivariance",
    "clifford_gated_activation",
    "CliffordEquivarianceAuditReport",
    "CliffordEquivarianceCertificate",
    "CliffordGeometricProductLayer",
    "CliffordGradeLinear",
]
