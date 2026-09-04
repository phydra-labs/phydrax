#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
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
    segment_prefix_boundaries: BoundaryRelation
    boundary: BoundaryRelation
    maximum_defect: Array
    maximum_constitutive_residual: Array
    profile_finite: Array
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


def _fourth_order_boundary(
    problem,
    layer: ContinuousFourierModalLayer,
    cascade: BoundaryCascadePolicy,
    left: Array | float,
    right: Array | float,
    /,
) -> tuple[BoundaryRelation, PreparedLayerOperator, PreparedLayerOperator]:
    width = right - left
    center = 0.5 * (left + right)
    offset = 0.5 * width / np.sqrt(3.0)
    first = _operator_at(problem, layer, center - offset)
    second = _operator_at(problem, layer, center + offset)
    coefficient_a = (3.0 - 2.0 * np.sqrt(3.0)) / 12.0
    coefficient_b = (3.0 + 2.0 * np.sqrt(3.0)) / 12.0
    first_matrix = coefficient_a * first.matrix + coefficient_b * second.matrix
    second_matrix = coefficient_b * first.matrix + coefficient_a * second.matrix
    first_boundary = prepare_layer_boundary(
        _with_matrix(first, first_matrix), width, cascade
    )
    second_boundary = prepare_layer_boundary(
        _with_matrix(second, second_matrix), width, cascade
    )
    return (
        compose_boundary_relations(first_boundary, second_boundary),
        first,
        second,
    )


def _segment_boundary(
    problem,
    layer: ContinuousFourierModalLayer,
    cascade: BoundaryCascadePolicy,
    left: float,
    right: float,
    /,
) -> tuple[BoundaryRelation, Array, Array, Array, Array, Array]:
    fourth_order, first, second = _fourth_order_boundary(
        problem, layer, cascade, left, right
    )
    midpoint = _operator_at(problem, layer, 0.5 * (left + right))
    second_order = prepare_layer_boundary(midpoint, right - left, cascade)
    defect = _boundary_difference(fourth_order, second_order)
    scale = jnp.maximum(_boundary_norm(fourth_order), 1.0)
    normalized = defect / scale
    operators = (first, second, midpoint)
    maximum_constitutive_residual = jnp.max(
        jnp.stack(tuple(value.diagnostics.constitutive_residual for value in operators))
    )
    finite = jnp.all(jnp.stack(tuple(value.diagnostics.finite for value in operators)))
    return (
        fourth_order,
        defect,
        normalized,
        scale,
        maximum_constitutive_residual,
        finite,
    )


def _stack_boundary_relations(
    relations: tuple[BoundaryRelation, ...], /
) -> BoundaryRelation:
    return jax.tree.map(lambda *values: jnp.stack(values, axis=0), *relations)


def continuous_boundary_at(
    prepared: PreparedContinuousFourierModalLayer,
    problem,
    longitudinal_offset: Array,
    cascade: BoundaryCascadePolicy,
    /,
) -> tuple[BoundaryRelation, PreparedLayerOperator, Array, Array]:
    """Reconstruct one dense-output boundary from accepted fixed-capacity data."""
    completed = jnp.sum(
        prepared.segment_active & (longitudinal_offset >= prepared.segment_edges[1:]),
        dtype=jnp.int32,
    )
    final_index = jnp.maximum(
        jnp.sum(prepared.segment_active, dtype=jnp.int32) - 1,
        0,
    )
    segment_index = jnp.minimum(completed, final_index)
    segment_left = prepared.segment_edges[segment_index]
    prefix = jax.tree.map(
        lambda value: value[segment_index],
        prepared.segment_prefix_boundaries,
    )
    partial, _, _ = _fourth_order_boundary(
        problem,
        prepared.layer,
        cascade,
        segment_left,
        longitudinal_offset,
    )
    local_operator = _operator_at(problem, prepared.layer, longitudinal_offset)
    return (
        compose_boundary_relations(prefix, partial),
        local_operator,
        prepared.segment_defects[segment_index],
        segment_index,
    )


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
    constitutive_residuals: list[float] = []
    finite_samples: list[bool] = []
    status = ContinuousFourierModalStatus.SUCCESS
    while True:
        relations.clear()
        absolute_defects.clear()
        defect_scales.clear()
        defects.clear()
        constitutive_residuals.clear()
        finite_samples.clear()
        for left, right in segments:
            relation, absolute, normalized, scale, residual, finite = _segment_boundary(
                problem, layer, cascade, left, right
            )
            relations.append(relation)
            absolute_defects.append(float(np.asarray(absolute)))
            defect_scales.append(float(np.asarray(scale)))
            defects.append(float(np.asarray(normalized)))
            constitutive_residuals.append(float(np.asarray(residual)))
            finite_samples.append(bool(np.asarray(finite)))
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
    real_dtype = jnp.empty((), dtype=dtype).real.dtype
    prefix_relations = [identity_boundary_relation(size, dtype)]
    for relation in relations:
        prefix_relations.append(
            compose_boundary_relations(prefix_relations[-1], relation)
        )
    boundary = prefix_relations[-1]
    prefix_relations.extend(
        boundary for _ in range(policy.maximum_segments + 1 - len(prefix_relations))
    )
    edges = np.ones((policy.maximum_segments + 1,), dtype=float) * thickness
    active = np.zeros((policy.maximum_segments,), dtype=bool)
    defect_values = np.zeros((policy.maximum_segments,), dtype=float)
    edges[: len(segments)] = np.asarray([value[0] for value in segments])
    edges[len(segments)] = thickness
    active[: len(segments)] = True
    defect_values[: len(defects)] = np.asarray(defects)
    finite = bool(
        np.all(np.isfinite(defect_values))
        and np.all(np.isfinite(constitutive_residuals))
        and all(finite_samples)
        and all(bool(np.asarray(value.diagnostics.finite)) for value in relations)
    )
    if not finite:
        status = ContinuousFourierModalStatus.NONFINITE_PROFILE
    return PreparedContinuousFourierModalLayer(
        layer,
        jnp.asarray(edges, dtype=real_dtype),
        jnp.asarray(active),
        jnp.asarray(defect_values, dtype=real_dtype),
        _stack_boundary_relations(tuple(prefix_relations)),
        boundary,
        jnp.asarray(max(defects, default=0.0), dtype=real_dtype),
        jnp.asarray(
            max(constitutive_residuals, default=0.0),
            dtype=real_dtype,
        ),
        jnp.asarray(finite),
        jnp.asarray(int(status), dtype=jnp.int32),
        jnp.asarray(status == ContinuousFourierModalStatus.SUCCESS),
    )


__all__ = [
    "ContinuousFourierModalStatus",
    "PreparedContinuousFourierModalLayer",
    "prepare_continuous_fourier_modal_layer",
]
