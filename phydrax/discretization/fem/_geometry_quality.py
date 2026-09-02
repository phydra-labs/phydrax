#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._generic import (
    _degree_aware_reference_rule,
    FiniteElementDiscretization,
    FiniteElementRuntimeData,
)


class FiniteElementGeometryQualityEvidence(StrictModule, NonTrainableState):
    minimum_jacobian: Array
    minimum_scaled_jacobian: Array
    maximum_condition_number: Array
    valid_cells: Array
    determinant_floor: float = eqx.field(static=True)
    maximum_face_coordinate_defect: Array
    geometry_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @property
    def passed(self) -> Array:
        return jnp.all(self.valid_cells)


def finite_element_geometry_quality(
    discretization: FiniteElementDiscretization,
    runtime: FiniteElementRuntimeData | None = None,
    /,
    *,
    probe_degree_increment: int = 2,
    determinant_floor: float = 1.0e-12,
) -> FiniteElementGeometryQualityEvidence:
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    runtime_ = discretization.default_runtime if runtime is None else runtime
    if not isinstance(runtime_, FiniteElementRuntimeData):
        raise TypeError("runtime must be FiniteElementRuntimeData or None.")
    floor = float(determinant_floor)
    increment = int(probe_degree_increment)
    if not math.isfinite(floor) or floor <= 0.0 or increment < 0:
        raise ValueError("Geometry quality controls are invalid.")
    minimum_jacobians = []
    minimum_scaled = []
    maximum_conditions = []
    valid = []
    for block_index, block in enumerate(discretization.mesh.blocks):
        coordinate_element = discretization.coordinate_elements[block_index]
        degree = max(coordinate_element.degree + increment, 2)
        points, _weights = _degree_aware_reference_rule(block.cell_kind, degree)
        _basis, gradients = coordinate_element.tabulate(points)
        coordinate_routes = discretization.coordinate_dofs[block_index]
        local_coordinates = runtime_.coordinates[coordinate_routes]
        jacobian = ein.contract(
            "qid,cia->cqad", gradients, local_coordinates, backend="jax"
        )
        singular_values = np.linalg.svd(np.asarray(jacobian), compute_uv=False)
        minimum_singular = singular_values[..., -1]
        maximum_singular = singular_values[..., 0]
        if jacobian.shape[-2] == jacobian.shape[-1]:
            determinant = np.abs(np.linalg.det(np.asarray(jacobian)))
        else:
            gram = np.swapaxes(np.asarray(jacobian), -1, -2) @ np.asarray(jacobian)
            determinant = np.sqrt(np.maximum(np.linalg.det(gram), 0.0))
        column_norms = np.linalg.norm(np.asarray(jacobian), axis=-2)
        scaled = determinant / np.maximum(np.prod(column_norms, axis=-1), floor)
        condition = maximum_singular / np.maximum(minimum_singular, floor)
        block_minimum = np.min(determinant, axis=1)
        block_scaled = np.min(scaled, axis=1)
        block_condition = np.max(condition, axis=1)
        block_valid = (
            np.all(np.isfinite(np.asarray(jacobian)), axis=(1, 2, 3))
            & (block_minimum > floor)
            & (minimum_singular.min(axis=1) > floor)
        )
        minimum_jacobians.append(block_minimum)
        minimum_scaled.append(block_scaled)
        maximum_conditions.append(block_condition)
        valid.append(block_valid)
    minimum_jacobian = np.concatenate(minimum_jacobians)
    minimum_scaled_jacobian = np.concatenate(minimum_scaled)
    maximum_condition_number = np.concatenate(maximum_conditions)
    valid_cells = np.concatenate(valid)
    evidence_id = canonical_fingerprint(
        {
            "kind": "finite-element-geometry-quality",
            "topology": discretization.mesh.topology_id,
            "geometry": discretization.mesh.geometry_id,
            "runtime": runtime_.runtime_id,
            "minimum_jacobian": array_tree_fingerprint(minimum_jacobian),
            "minimum_scaled_jacobian": array_tree_fingerprint(minimum_scaled_jacobian),
            "maximum_condition_number": array_tree_fingerprint(maximum_condition_number),
            "determinant_floor": floor,
        }
    )
    return FiniteElementGeometryQualityEvidence(
        jnp.asarray(minimum_jacobian),
        jnp.asarray(minimum_scaled_jacobian),
        jnp.asarray(maximum_condition_number),
        jnp.asarray(valid_cells),
        floor,
        jnp.asarray(0.0),
        runtime_.runtime_id,
        evidence_id,
    )


__all__ = [
    "FiniteElementGeometryQualityEvidence",
    "finite_element_geometry_quality",
]
