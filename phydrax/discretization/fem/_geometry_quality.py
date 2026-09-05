#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
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
        singular_values = jnp.linalg.svd(jacobian, compute_uv=False)
        minimum_singular = singular_values[..., -1]
        maximum_singular = singular_values[..., 0]
        if jacobian.shape[-2] == jacobian.shape[-1]:
            determinant = jnp.abs(jnp.linalg.det(jacobian))
        else:
            gram = jnp.swapaxes(jacobian, -1, -2) @ jacobian
            determinant = jnp.sqrt(jnp.maximum(jnp.linalg.det(gram), 0.0))
        column_norms = jnp.linalg.norm(jacobian, axis=-2)
        scaled = determinant / jnp.maximum(jnp.prod(column_norms, axis=-1), floor)
        condition = maximum_singular / jnp.maximum(minimum_singular, floor)
        block_minimum = jnp.min(determinant, axis=1)
        block_scaled = jnp.min(scaled, axis=1)
        block_condition = jnp.max(condition, axis=1)
        block_valid = (
            jnp.all(jnp.isfinite(jacobian), axis=(1, 2, 3))
            & (block_minimum > floor)
            & (jnp.min(minimum_singular, axis=1) > floor)
        )
        minimum_jacobians.append(block_minimum)
        minimum_scaled.append(block_scaled)
        maximum_conditions.append(block_condition)
        valid.append(block_valid)
    minimum_jacobian = jnp.concatenate(minimum_jacobians)
    minimum_scaled_jacobian = jnp.concatenate(minimum_scaled)
    maximum_condition_number = jnp.concatenate(maximum_conditions)
    valid_cells = jnp.concatenate(valid)
    evidence_id = canonical_fingerprint(
        {
            "kind": "finite-element-geometry-quality",
            "topology": discretization.mesh.topology_id,
            "geometry": discretization.mesh.geometry_id,
            "runtime": runtime_.runtime_id,
            "probe_degree_increment": increment,
            "determinant_floor": floor,
        }
    )
    return FiniteElementGeometryQualityEvidence(
        minimum_jacobian,
        minimum_scaled_jacobian,
        maximum_condition_number,
        valid_cells,
        floor,
        jnp.asarray(0.0),
        runtime_.runtime_id,
        evidence_id,
    )


__all__ = [
    "FiniteElementGeometryQualityEvidence",
    "finite_element_geometry_quality",
]
