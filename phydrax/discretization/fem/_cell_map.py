#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._generic import _evaluate_coordinate_map, FiniteElementDiscretization
from ._precision import FiniteElementPrecisionPolicy
from ._reference import FiniteElementSpec


class FiniteElementCellMapEvaluation(StrictModule):
    """Paired physical/reference evaluation of one prepared FE cell map."""

    physical_points: Array
    jacobian: Array
    inverse_jacobian: Array
    determinant: Array
    measure: Array
    minimum_metric_eigenvalue: Array
    validity_margin: Array
    valid: Array


class PreparedFiniteElementCellMap(StrictModule, NonTrainableState):
    """Fixed-topology coordinate map for one finite-element cell block."""

    coordinate_element: FiniteElementSpec
    coordinate_dofs: Array
    precision_policy: FiniteElementPrecisionPolicy
    block_name: str = eqx.field(static=True)
    block_index: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    reference_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    cell_map_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        block_index: int,
        /,
    ):
        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        index = int(block_index)
        if index < 0 or index >= len(discretization.mesh.blocks):
            raise IndexError("block_index is outside the finite-element mesh.")
        block = discretization.mesh.blocks[index]
        coordinate_element = discretization.coordinate_elements[index]
        coordinate_dofs = discretization.coordinate_dofs[index]
        if coordinate_element.cell_kind != block.cell_kind:
            raise ValueError("Coordinate element and cell block kinds differ.")
        self.coordinate_element = coordinate_element
        self.coordinate_dofs = jnp.asarray(coordinate_dofs)
        self.precision_policy = discretization.precision_policy
        self.block_name = block.name
        self.block_index = index
        self.cell_count = block.cell_count
        self.coordinate_count = int(discretization.default_runtime.coordinates.shape[0])
        self.reference_dimension = block.topological_dimension
        self.ambient_dimension = discretization.mesh.ambient_dimension
        self.topology_id = discretization.mesh.topology_id
        self.geometry_layout_id = discretization.default_runtime.geometry_layout_id
        self.cell_map_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element-cell-map",
                "topology": self.topology_id,
                "geometry_layout": self.geometry_layout_id,
                "block": block.block_id,
                "coordinate_element": coordinate_element.element_id,
                "coordinate_dofs": array_tree_fingerprint(coordinate_dofs),
            }
        )

    def evaluate(
        self,
        coordinates: ArrayLike,
        cell_indices: ArrayLike,
        reference_points: ArrayLike,
        /,
    ) -> FiniteElementCellMapEvaluation:
        """Evaluate paired cell/reference rows with explicit validity margins."""

        coordinate_values = jnp.asarray(coordinates)
        if coordinate_values.shape != (
            self.coordinate_count,
            self.ambient_dimension,
        ):
            raise ValueError(
                "coordinates must preserve the prepared geometry coordinate shape."
            )
        indices = jnp.asarray(cell_indices)
        points = jnp.asarray(reference_points)
        if indices.ndim != 1 or not jnp.issubdtype(indices.dtype, jnp.integer):
            raise ValueError("cell_indices must be a rank-1 integer array.")
        if points.shape != (indices.shape[0], self.reference_dimension):
            raise ValueError(
                "reference_points must have shape (cell_indices.size, reference_dimension)."
            )
        index_valid = (indices >= 0) & (indices < self.cell_count)
        safe_indices = jnp.clip(indices, 0, self.cell_count - 1)
        routes = self.coordinate_dofs[safe_indices]
        (
            physical_points,
            jacobian,
            metric,
            _,
            inverse_jacobian,
            _,
            measure,
            determinant,
        ) = _evaluate_coordinate_map(
            self.coordinate_element,
            routes,
            coordinate_values,
            points,
            precision_policy=self.precision_policy,
            paired=True,
        )
        minimum_metric_eigenvalue = jnp.linalg.eigvalsh(metric)[..., 0]
        if self.reference_dimension == self.ambient_dimension:
            validity_margin = jnp.minimum(minimum_metric_eigenvalue, determinant)
        else:
            validity_margin = jnp.minimum(minimum_metric_eigenvalue, measure)
        finite = (
            jnp.all(jnp.isfinite(physical_points), axis=-1)
            & jnp.all(jnp.isfinite(jacobian), axis=(-2, -1))
            & jnp.all(jnp.isfinite(inverse_jacobian), axis=(-2, -1))
            & jnp.isfinite(determinant)
            & jnp.isfinite(measure)
            & jnp.isfinite(validity_margin)
        )
        valid = index_valid & finite & (validity_margin > 0.0)
        return FiniteElementCellMapEvaluation(
            physical_points=physical_points,
            jacobian=jacobian,
            inverse_jacobian=inverse_jacobian,
            determinant=determinant,
            measure=measure,
            minimum_metric_eigenvalue=minimum_metric_eigenvalue,
            validity_margin=validity_margin,
            valid=valid,
        )


def prepare_finite_element_cell_map(
    discretization: FiniteElementDiscretization,
    block_index: int,
    /,
) -> PreparedFiniteElementCellMap:
    """Prepare one reusable fixed-topology finite-element coordinate map."""

    return PreparedFiniteElementCellMap(discretization, block_index)


__all__ = [
    "FiniteElementCellMapEvaluation",
    "PreparedFiniteElementCellMap",
    "prepare_finite_element_cell_map",
]
