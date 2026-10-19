#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from phydrax.discretization.fem._reference_types import FiniteElementSpec

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DensePropertyVerificationPolicy, verify_dense_properties
from ...typing import checked
from .._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometrySpec,
    LayerColumnCellGeometryElement,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
    RestrictedCellGeometryElement,
    ScalarCoordinateElement,
    SplineCellGeometryElement,
)
from .._cell_mesh import CellMesh
from .._coordinate_budget import CoordinateEnclosureBudget
from .._coordinate_enclosure import CoordinateSourceBank
from .._reference_cell_types import CellGeometryElement
from . import _generic as generic_owner
from ._assembly import evaluate_coordinate_map
from ._precision import FiniteElementPrecisionPolicy


# Strict positivity matches the cell-map margin contract; native property
# evidence also controls finite/Hermitian failure admission.
_CELL_METRIC_PROPERTIES = DensePropertyVerificationPolicy(
    require_positive_definite=True, relative_tolerance=0.0, absolute_tolerance=0.0
)


class FiniteElementCellMapEvaluation(StrictModule):
    """Coordinate and metric evidence for paired physical/reference rows.

    ``valid`` includes native finite/Hermitian/positive-definite admission and
    the coordinate-map orientation margin.
    """

    physical_points: Array
    jacobian: Array
    inverse_jacobian: Array
    determinant: Array
    measure: Array
    minimum_metric_eigenvalue: Array
    validity_margin: Array
    valid: Array


def _tabulated_coordinate_element(
    element: CellGeometryElement,
) -> ScalarCoordinateElement:
    match element:
        case (
            FiniteElementSpec()
            | BarycentricCellGeometryElement()
            | RestrictedCellGeometryElement()
            | PolynomialComposedCellGeometryElement()
            | RationalComposedCellGeometryElement()
            | SplineCellGeometryElement()
            | LayerColumnCellGeometryElement()
        ):
            return element
        case _:
            raise TypeError(
                "cell_map.coordinate_element must support tabulated coordinates; got an unsupported element"
            )


def _whole_support_routes(
    discretization: generic_owner.FiniteElementDiscretization,
    coordinate_element: ScalarCoordinateElement,
    /,
) -> tuple[Array, int, tuple[str, ...]]:
    for element, mesh_block in zip(
        discretization.coordinate_elements, discretization.mesh.blocks, strict=True
    ):
        element = _tabulated_coordinate_element(element)
        if element.cell_kind != mesh_block.cell_kind:
            raise ValueError(
                "cell_map.cell_kind must match the block; got different element and block kinds"
            )
        if element.element_id != coordinate_element.element_id:
            raise ValueError(
                "cell_map.coordinate_elements must share one element identity; got different identities"
            )
    coordinate_dofs = jnp.concatenate(discretization.coordinate_dofs, axis=0)
    cell_count = sum(mesh_block.cell_count for mesh_block in discretization.mesh.blocks)
    block_ids = tuple(mesh_block.block_id for mesh_block in discretization.mesh.blocks)
    return coordinate_dofs, cell_count, block_ids


class PreparedFiniteElementCellMap(StrictModule, NonTrainableState):
    """Fixed-topology coordinate map for one block or one homogeneous support."""

    mesh: CellMesh
    coordinate_element: ScalarCoordinateElement
    coordinate_dofs: Array
    precision_policy: FiniteElementPrecisionPolicy
    block_name: str = eqx.field(static=True)
    block_index: int | None = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    coordinate_count: int = eqx.field(static=True)
    reference_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    cell_map_id: str = eqx.field(static=True)
    coordinate_spec: CellGeometrySpec

    @checked
    def __init__(
        self,
        discretization: generic_owner.FiniteElementDiscretization,
        block_index: int | None,
        /,
    ) -> None:
        if block_index is None:
            index = None
            block = discretization.mesh.blocks[0]
            coordinate_element = _tabulated_coordinate_element(
                discretization.coordinate_elements[0]
            )
            coordinate_dofs, cell_count, block_ids = _whole_support_routes(
                discretization, coordinate_element
            )
            block_name = "whole-mesh"
            block_identity = {"blocks": block_ids}
        else:
            index = int(block_index)
            if index < 0 or index >= len(discretization.mesh.blocks):
                raise IndexError(
                    "cell_map.block_index must index a mesh block; got an out-of-range index"
                )
            block = discretization.mesh.blocks[index]
            coordinate_element = _tabulated_coordinate_element(
                discretization.coordinate_elements[index]
            )
            coordinate_dofs = discretization.coordinate_dofs[index]
            if coordinate_element.cell_kind != block.cell_kind:
                raise ValueError(
                    "cell_map.cell_kind must match the block; got different element and block kinds"
                )
            block_name = block.name
            cell_count = block.cell_count
            block_identity = {"block": block.block_id}
        self.mesh = discretization.mesh
        self.coordinate_element = coordinate_element
        self.coordinate_spec = discretization.coordinate_spec
        self.coordinate_dofs = jnp.asarray(coordinate_dofs)
        self.precision_policy = discretization.precision_policy
        self.block_name = block_name
        self.block_index = index
        self.cell_count = cell_count
        self.coordinate_count = discretization.default_runtime.coordinates.shape[0]
        self.reference_dimension = block.topological_dimension
        self.ambient_dimension = discretization.mesh.ambient_dimension
        self.topology_id = discretization.mesh.topology_id
        self.geometry_layout_id = discretization.default_runtime.geometry_layout_id
        self.cell_map_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-element-cell-map",
                "topology": self.topology_id,
                "geometry_layout": self.geometry_layout_id,
                **block_identity,
                "coordinate_element": coordinate_element.element_id,
                "coordinate_dofs": array_tree_fingerprint(coordinate_dofs),
            }
        )

    @checked
    def source_coordinates(
        self, coordinates: ArrayLike, /, *, budget: CoordinateEnclosureBudget
    ) -> CoordinateSourceBank:
        """Prepare the supplied source bank outside compiled execution."""
        values = jnp.asarray(coordinates)
        if values.shape != (self.coordinate_count, self.ambient_dimension):
            raise ValueError(
                "cell_map.coordinates must preserve the prepared coordinate shape; got a different shape"
            )
        source = self.coordinate_spec.with_coordinates(values, budget=budget)
        return source.source_coordinates(budget=budget)

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
                "cell_map.coordinates must preserve the prepared coordinate shape; got a different shape"
            )
        indices = jnp.asarray(cell_indices)
        points = jnp.asarray(reference_points)
        if indices.ndim != 1 or not jnp.issubdtype(indices.dtype, jnp.integer):
            raise ValueError(
                "cell_map.cell_indices must be a rank-1 integer array; got an incompatible rank or dtype"
            )
        if points.shape != (indices.shape[0], self.reference_dimension):
            raise ValueError(
                "cell_map.reference_points must have shape (cell_indices.size, reference_dimension); got a different shape"
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
        ) = evaluate_coordinate_map(
            self.coordinate_element,
            routes,
            coordinate_values,
            points,
            precision_policy=self.precision_policy,
            paired=True,
        )
        metric_evidence = verify_dense_properties(metric, policy=_CELL_METRIC_PROPERTIES)
        minimum_metric_eigenvalue = metric_evidence.eigenvalues[..., 0]
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
        valid = (
            index_valid & finite & metric_evidence.successful & (validity_margin > 0.0)
        )
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


@checked
def prepare_finite_element_cell_map(
    discretization: generic_owner.FiniteElementDiscretization,
    block_index: int | None,
    /,
) -> PreparedFiniteElementCellMap:
    """Prepare a reusable block or homogeneous whole-support coordinate map."""

    return PreparedFiniteElementCellMap(discretization, block_index)


__all__ = [
    "FiniteElementCellMapEvaluation",
    "PreparedFiniteElementCellMap",
    "prepare_finite_element_cell_map",
]
