#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._strict import StrictModule
from ._boundary_cascade import (
    BoundaryCascadePolicy,
    BoundaryRelation,
    compose_boundary_relations,
    identity_boundary_relation,
    prepare_layer_boundary,
)
from ._contracts import ContinuousFourierModalLayer, FrequencyMaxwellMaterial
from ._factorization import prepare_fourier_material
from ._layer import prepare_layer_operator, PreparedLayerOperator


class ContinuousFourierModalStatus(IntEnum):
    SUCCESS = 0
    REFINEMENT_REQUIRED = 1
    NONFINITE_PROFILE = 2


class PreparedContinuousFourierModalLayer(StrictModule):
    layer: ContinuousFourierModalLayer
    segment_edges: Array
    segment_active: Array
    segment_defects: Array
    operator: PreparedLayerOperator
    boundary: BoundaryRelation
    maximum_defect: Array
    status: Array
    successful: Array


def _boundary_norm(boundary: BoundaryRelation, /) -> Array:
    return jnp.sqrt(
        sum(
            jnp.sum(jnp.abs(value) ** 2)
            for value in (boundary.a, boundary.b, boundary.c, boundary.d)
        )
    )


def _boundary_difference(left: BoundaryRelation, right: BoundaryRelation, /) -> Array:
    return jnp.sqrt(
        sum(
            jnp.sum(jnp.abs(x - y) ** 2)
            for x, y in zip(
                (left.a, left.b, left.c, left.d),
                (right.a, right.b, right.c, right.d),
                strict=True,
            )
        )
    )


def _operator_at(
    problem,
    layer: ContinuousFourierModalLayer,
    coordinate: float,
    /,
) -> PreparedLayerOperator:
    material = layer.material_profile(
        jnp.asarray(coordinate, dtype=layer.thickness.dtype)
    )
    if not isinstance(material, FrequencyMaxwellMaterial):
        raise TypeError(
            "Continuous material_profile must return FrequencyMaxwellMaterial."
        )
    prepared = prepare_fourier_material(material, problem.harmonics, layer.factorization)
    return prepare_layer_operator(
        prepared,
        problem.harmonics,
        problem.angular_frequency,
        problem.bloch_wavevector,
    )


def _with_matrix(
    operator: PreparedLayerOperator, matrix: Array, /
) -> PreparedLayerOperator:
    return eqx.tree_at(lambda value: value.matrix, operator, matrix)


def _segment_boundary(
    problem,
    layer: ContinuousFourierModalLayer,
    cascade: BoundaryCascadePolicy,
    left: float,
    right: float,
    /,
) -> tuple[BoundaryRelation, Array, Array, Array, PreparedLayerOperator]:
    width = right - left
    center = 0.5 * (left + right)
    offset = 0.5 * width / np.sqrt(3.0)
    first = _operator_at(problem, layer, center - offset)
    second = _operator_at(problem, layer, center + offset)
    midpoint = _operator_at(problem, layer, center)
    second_order = prepare_layer_boundary(midpoint, width, cascade)
    coefficient_a = (3.0 - 2.0 * np.sqrt(3.0)) / 12.0
    coefficient_b = (3.0 + 2.0 * np.sqrt(3.0)) / 12.0
    first_matrix = coefficient_a * first.matrix + coefficient_b * second.matrix
    second_matrix = coefficient_b * first.matrix + coefficient_a * second.matrix
    first_boundary = prepare_layer_boundary(
        _with_matrix(midpoint, first_matrix), width, cascade
    )
    second_boundary = prepare_layer_boundary(
        _with_matrix(midpoint, second_matrix), width, cascade
    )
    fourth_order = compose_boundary_relations(first_boundary, second_boundary)
    defect = _boundary_difference(fourth_order, second_order)
    scale = jnp.maximum(_boundary_norm(fourth_order), 1.0)
    normalized = defect / scale
    return fourth_order, defect, normalized, scale, midpoint


def prepare_continuous_fourier_modal_layer(
    problem,
    layer: ContinuousFourierModalLayer,
    cascade: BoundaryCascadePolicy,
    /,
) -> PreparedContinuousFourierModalLayer:
    """Host-adapt one profile; returned execution topology is immutable."""

    thickness = float(np.asarray(layer.thickness))
    if not np.isfinite(thickness) or thickness < 0.0:
        raise ValueError("Continuous Fourier layer thickness must be finite/nonnegative.")
    policy = layer.integration_policy
    segments: list[tuple[float, float]] = [(0.0, thickness)]
    relations: list[BoundaryRelation] = []
    absolute_defects: list[float] = []
    defect_scales: list[float] = []
    defects: list[float] = []
    operators: list[PreparedLayerOperator] = []
    status = ContinuousFourierModalStatus.SUCCESS
    while True:
        relations.clear()
        absolute_defects.clear()
        defect_scales.clear()
        defects.clear()
        operators.clear()
        for left, right in segments:
            relation, absolute, normalized, scale, operator = _segment_boundary(
                problem, layer, cascade, left, right
            )
            relations.append(relation)
            absolute_defects.append(float(np.asarray(absolute)))
            defect_scales.append(float(np.asarray(scale)))
            defects.append(float(np.asarray(normalized)))
            operators.append(operator)
        tolerances = policy.absolute_tolerance + policy.relative_tolerance * np.asarray(
            defect_scales
        )
        ratios = np.asarray(absolute_defects) / np.maximum(tolerances, 1.0e-300)
        worst = int(np.argmax(ratios))
        if absolute_defects[worst] <= tolerances[worst]:
            break
        left, right = segments[worst]
        if (
            len(segments) >= policy.maximum_segments
            or right - left <= policy.minimum_segment_fraction * max(thickness, 1.0)
        ):
            status = ContinuousFourierModalStatus.REFINEMENT_REQUIRED
            break
        midpoint = 0.5 * (left + right)
        segments[worst : worst + 1] = [(left, midpoint), (midpoint, right)]
    size = 2 * problem.harmonics.harmonic_count
    dtype = jnp.dtype(problem.harmonics.plan.precision.coefficient_dtype)
    boundary = identity_boundary_relation(size, dtype)
    for relation in relations:
        boundary = compose_boundary_relations(boundary, relation)
    edges = np.ones((policy.maximum_segments + 1,), dtype=float) * thickness
    active = np.zeros((policy.maximum_segments,), dtype=bool)
    defect_values = np.zeros((policy.maximum_segments,), dtype=float)
    edges[: len(segments)] = np.asarray([value[0] for value in segments])
    edges[len(segments)] = thickness
    active[: len(segments)] = True
    defect_values[: len(defects)] = np.asarray(defects)
    finite = bool(
        np.all(np.isfinite(defect_values))
        and all(bool(np.asarray(value.diagnostics.finite)) for value in operators)
    )
    if not finite:
        status = ContinuousFourierModalStatus.NONFINITE_PROFILE
    return PreparedContinuousFourierModalLayer(
        layer,
        jnp.asarray(edges, dtype=layer.thickness.dtype),
        jnp.asarray(active),
        jnp.asarray(defect_values, dtype=layer.thickness.dtype),
        operators[len(operators) // 2],
        boundary,
        jnp.asarray(max(defects, default=0.0), dtype=layer.thickness.dtype),
        jnp.asarray(int(status), dtype=jnp.int32),
        jnp.asarray(status == ContinuousFourierModalStatus.SUCCESS),
    )


__all__ = [
    "ContinuousFourierModalStatus",
    "PreparedContinuousFourierModalLayer",
    "prepare_continuous_fourier_modal_layer",
]
