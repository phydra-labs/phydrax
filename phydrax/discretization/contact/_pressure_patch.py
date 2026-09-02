#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import intersect_tetrahedra
from .._cell_mesh import CellMesh


class HydroelasticPressureFieldPlan(StrictModule, NonTrainableState):
    """One affine tetrahedral ``CellMesh`` supporting a dynamic P1 pressure."""

    mesh: CellMesh
    tetrahedra: Array
    cell_global_ids: Array
    field_id: str = eqx.field(static=True)

    def __init__(self, mesh: CellMesh, /):
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be a CellMesh.")
        if (
            mesh.topological_dimension != 3
            or mesh.ambient_dimension != 3
            or len(mesh.blocks) != 1
            or mesh.blocks[0].cell_kind != "tetrahedron"
        ):
            raise ValueError(
                "Hydroelastic pressure requires one affine tetrahedral CellMesh block."
            )
        self.mesh = mesh
        self.tetrahedra = mesh.blocks[0].vertices
        self.cell_global_ids = mesh.blocks[0].global_ids
        self.field_id = canonical_fingerprint(
            {
                "kind": "hydroelastic-pressure-field-plan",
                "mesh": mesh.mesh_id,
            }
        )

    @property
    def vertex_count(self) -> int:
        return int(self.mesh.coordinates.shape[0])

    @property
    def cell_count(self) -> int:
        return int(self.tetrahedra.shape[0])


class HydroelasticPressureFieldState(StrictModule):
    pressure: Array

    def __init__(self, pressure: ArrayLike, /):
        value = jnp.asarray(pressure)
        if value.ndim != 1 or not jnp.issubdtype(value.dtype, jnp.floating):
            raise ValueError("Hydroelastic pressure state must be a real vector.")
        self.pressure = value


class HydroelasticPressurePatch(StrictModule, NonTrainableState):
    quadrature_point: Array
    normal: Array
    pressure: Array
    quadrature_weight: Array
    plus_source_cell: Array
    minus_source_cell: Array
    plus_interpolation: Array
    minus_interpolation: Array
    valid: Array
    patch_id: str = eqx.field(static=True)

    @property
    def capacity(self) -> int:
        return int(self.valid.size)

    @property
    def source_tetrahedron(self) -> Array:
        """The plus source cell, retained as the one canonical source spelling."""
        return self.plus_source_cell


class HydroelasticPatchEvidence(StrictModule):
    intersected_tetrahedra: Array
    triangle_count: Array
    overflow_count: Array
    total_area: Array
    minimum_pressure: Array
    pressure_balance_residual: Array
    predicate_margin: Array
    tie_margin: Array
    finite: Array
    derivative_valid: Array
    complete: Array
    successful: Array
    plus_field_id: str = eqx.field(static=True)
    minus_field_id: str = eqx.field(static=True)


class HydroelasticPatchExtraction(StrictModule):
    patch: HydroelasticPressurePatch
    evidence: HydroelasticPatchEvidence


class HydroelasticPatchExtractionPlan(StrictModule, NonTrainableState):
    maximum_overlap_pairs: int = eqx.field(static=True)
    maximum_polytope_edges: int = eqx.field(static=True)
    patch_capacity: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_overlap_pairs: int,
        maximum_polytope_edges: int,
        patch_capacity: int,
        tolerance: float = 1.0e-12,
    ):
        pairs = int(maximum_overlap_pairs)
        edges = int(maximum_polytope_edges)
        capacity = int(patch_capacity)
        tolerance_ = float(tolerance)
        if pairs <= 0 or edges < 3 or capacity <= 0:
            raise ValueError("Hydroelastic extraction capacities are invalid.")
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Hydroelastic extraction tolerance must be positive.")
        self.maximum_overlap_pairs = pairs
        self.maximum_polytope_edges = edges
        self.patch_capacity = capacity
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hydroelastic-patch-extraction-plan",
                "maximum_overlap_pairs": pairs,
                "maximum_polytope_edges": edges,
                "patch_capacity": capacity,
                "tolerance": tolerance_.hex(),
            }
        )

    def prepare(
        self,
        plus: HydroelasticPressureFieldPlan,
        minus: HydroelasticPressureFieldPlan,
        /,
    ) -> PreparedHydroelasticPatchExtraction:
        if not isinstance(plus, HydroelasticPressureFieldPlan) or not isinstance(
            minus, HydroelasticPressureFieldPlan
        ):
            raise TypeError("plus/minus must be HydroelasticPressureFieldPlan values.")
        plus_coordinates = np.asarray(plus.mesh.coordinates)
        minus_coordinates = np.asarray(minus.mesh.coordinates)
        plus_tetrahedra = np.asarray(plus.tetrahedra, dtype=np.int32)
        minus_tetrahedra = np.asarray(minus.tetrahedra, dtype=np.int32)
        plus_ids = np.asarray(plus.cell_global_ids, dtype=np.int64)
        minus_ids = np.asarray(minus.cell_global_ids, dtype=np.int64)
        records = []
        predicate_margin = np.inf
        preparation_complete = True
        for plus_index, plus_vertices in enumerate(plus_tetrahedra):
            plus_cell = plus_coordinates[plus_vertices]
            plus_minimum = np.min(plus_cell, axis=0)
            plus_maximum = np.max(plus_cell, axis=0)
            for minus_index, minus_vertices in enumerate(minus_tetrahedra):
                minus_cell = minus_coordinates[minus_vertices]
                if np.any(plus_maximum < np.min(minus_cell, axis=0)) or np.any(
                    np.max(minus_cell, axis=0) < plus_minimum
                ):
                    continue
                intersection = intersect_tetrahedra(
                    plus_cell,
                    minus_cell,
                    source_id=int(plus_ids[plus_index]),
                    target_id=int(minus_ids[minus_index]),
                )
                if intersection.evidence.predicate_uncertain:
                    preparation_complete = False
                if not intersection.successful:
                    continue
                edges = sorted(
                    {
                        tuple(sorted((face[index], face[(index + 1) % len(face)])))
                        for face in intersection.faces
                        for index in range(len(face))
                    }
                )
                if len(edges) > self.maximum_polytope_edges:
                    preparation_complete = False
                    continue
                plus_basis = _tetrahedron_basis(plus_cell)
                minus_basis = _tetrahedron_basis(minus_cell)
                edge_points = np.asarray(
                    [
                        (intersection.vertices[first], intersection.vertices[second])
                        for first, second in edges
                    ],
                    dtype=float,
                )
                plus_weights = _barycentric_weights(edge_points, plus_cell, plus_basis)
                minus_weights = _barycentric_weights(edge_points, minus_cell, minus_basis)
                records.append(
                    (
                        int(plus_vertices[0]),
                        plus_vertices,
                        minus_vertices,
                        int(plus_ids[plus_index]),
                        int(minus_ids[minus_index]),
                        edge_points,
                        plus_weights,
                        minus_weights,
                        plus_basis,
                        minus_basis,
                    )
                )
                predicate_margin = min(
                    predicate_margin,
                    max(
                        intersection.volume - intersection.evidence.volume_error,
                        0.0,
                    ),
                )
        records.sort(key=lambda value: (value[3], value[4]))
        if len(records) > self.maximum_overlap_pairs:
            preparation_complete = False
        records = records[: self.maximum_overlap_pairs]
        pair_capacity = self.maximum_overlap_pairs
        edge_capacity = self.maximum_polytope_edges
        edge_points = np.zeros((pair_capacity, edge_capacity, 2, 3), dtype=float)
        plus_weights = np.zeros((pair_capacity, edge_capacity, 2, 4), dtype=float)
        minus_weights = np.zeros((pair_capacity, edge_capacity, 2, 4), dtype=float)
        edge_valid = np.zeros((pair_capacity, edge_capacity), dtype=bool)
        plus_cells = np.zeros((pair_capacity, 4), dtype=np.int32)
        minus_cells = np.zeros((pair_capacity, 4), dtype=np.int32)
        plus_cell_ids = np.zeros((pair_capacity,), dtype=np.int64)
        minus_cell_ids = np.zeros((pair_capacity,), dtype=np.int64)
        plus_gradients = np.zeros((pair_capacity, 4, 3), dtype=float)
        minus_gradients = np.zeros((pair_capacity, 4, 3), dtype=float)
        pair_valid = np.zeros((pair_capacity,), dtype=bool)
        for pair_index, record in enumerate(records):
            (
                _,
                plus_cell,
                minus_cell,
                plus_id,
                minus_id,
                points,
                plus_route,
                minus_route,
                plus_basis,
                minus_basis,
            ) = record
            count = points.shape[0]
            edge_points[pair_index, :count] = points
            plus_weights[pair_index, :count] = plus_route
            minus_weights[pair_index, :count] = minus_route
            edge_valid[pair_index, :count] = True
            plus_cells[pair_index] = plus_cell
            minus_cells[pair_index] = minus_cell
            plus_cell_ids[pair_index] = plus_id
            minus_cell_ids[pair_index] = minus_id
            plus_gradients[pair_index] = plus_basis
            minus_gradients[pair_index] = minus_basis
            pair_valid[pair_index] = True
        return PreparedHydroelasticPatchExtraction(
            plus,
            minus,
            jnp.asarray(edge_points),
            jnp.asarray(plus_weights),
            jnp.asarray(minus_weights),
            jnp.asarray(edge_valid),
            jnp.asarray(plus_cells),
            jnp.asarray(minus_cells),
            jnp.asarray(plus_cell_ids),
            jnp.asarray(minus_cell_ids),
            jnp.asarray(plus_gradients),
            jnp.asarray(minus_gradients),
            jnp.asarray(pair_valid),
            jnp.asarray(predicate_margin),
            jnp.asarray(preparation_complete),
            self.patch_capacity,
            self.tolerance,
            canonical_fingerprint(
                {
                    "kind": "prepared-hydroelastic-patch-extraction",
                    "plan": self.plan_id,
                    "plus": plus.field_id,
                    "minus": minus.field_id,
                    "overlap_pairs": len(records),
                }
            ),
        )


class PreparedHydroelasticPatchExtraction(StrictModule, NonTrainableState):
    plus: HydroelasticPressureFieldPlan
    minus: HydroelasticPressureFieldPlan
    edge_points: Array
    plus_edge_weights: Array
    minus_edge_weights: Array
    edge_valid: Array
    plus_cells: Array
    minus_cells: Array
    plus_cell_ids: Array
    minus_cell_ids: Array
    plus_basis_gradients: Array
    minus_basis_gradients: Array
    pair_valid: Array
    predicate_margin: Array
    preparation_complete: Array
    patch_capacity: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def evaluate(
        self,
        plus_state: HydroelasticPressureFieldState,
        minus_state: HydroelasticPressureFieldState,
        /,
    ) -> HydroelasticPatchExtraction:
        if not isinstance(plus_state, HydroelasticPressureFieldState) or not isinstance(
            minus_state, HydroelasticPressureFieldState
        ):
            raise TypeError("plus_state/minus_state must be pressure field states.")
        if plus_state.pressure.shape != (
            self.plus.vertex_count,
        ) or minus_state.pressure.shape != (self.minus.vertex_count,):
            raise ValueError("Hydroelastic pressure state has the wrong vertex capacity.")
        plus_local = plus_state.pressure[self.plus_cells]
        minus_local = minus_state.pressure[self.minus_cells]
        plus_endpoint = oe.contract(
            "pevk,pk->pev", self.plus_edge_weights, plus_local, backend="jax"
        )
        minus_endpoint = oe.contract(
            "pevk,pk->pev", self.minus_edge_weights, minus_local, backend="jax"
        )
        difference = plus_endpoint - minus_endpoint
        denominator = difference[..., 0] - difference[..., 1]
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(difference[..., 0]), jnp.abs(difference[..., 1])),
            1.0,
        )
        edge_tie_margin = jnp.abs(denominator) / scale
        crossing = (
            self.edge_valid
            & self.pair_valid[:, None]
            & (difference[..., 0] * difference[..., 1] <= 0.0)
            & (jnp.abs(denominator) > self.tolerance * scale)
        )
        parameter = jnp.where(crossing, difference[..., 0] / denominator, 0.0)
        crossing = crossing & (parameter >= 0.0) & (parameter <= 1.0)
        point = (1.0 - parameter)[..., None] * self.edge_points[..., 0, :] + parameter[
            ..., None
        ] * self.edge_points[..., 1, :]
        plus_route = (1.0 - parameter)[..., None] * self.plus_edge_weights[
            ..., 0, :
        ] + parameter[..., None] * self.plus_edge_weights[..., 1, :]
        minus_route = (1.0 - parameter)[..., None] * self.minus_edge_weights[
            ..., 0, :
        ] + parameter[..., None] * self.minus_edge_weights[..., 1, :]
        pressure = 0.5 * (
            (1.0 - parameter) * (plus_endpoint[..., 0] + minus_endpoint[..., 0])
            + parameter * (plus_endpoint[..., 1] + minus_endpoint[..., 1])
        )
        count = jnp.sum(crossing, axis=-1, dtype=jnp.int32)
        safe_count = jnp.maximum(count, 1).astype(point.dtype)
        centroid = (
            jnp.sum(jnp.where(crossing[..., None], point, 0.0), axis=1)
            / safe_count[:, None]
        )
        plus_gradient = oe.contract(
            "pkd,pk->pd", self.plus_basis_gradients, plus_local, backend="jax"
        )
        minus_gradient = oe.contract(
            "pkd,pk->pd", self.minus_basis_gradients, minus_local, backend="jax"
        )
        gradient = plus_gradient - minus_gradient
        gradient_norm = jnp.sqrt(jnp.sum(gradient * gradient, axis=-1))
        normal = gradient / jnp.where(gradient_norm > 0.0, gradient_norm, 1.0)[:, None]
        axis_index = jnp.argmin(jnp.abs(normal), axis=-1)
        reference_axis = jnp.eye(3, dtype=point.dtype)[axis_index]
        tangent_first = jnp.cross(normal, reference_axis)
        tangent_first = (
            tangent_first
            / jnp.where(
                jnp.linalg.norm(tangent_first, axis=-1) > 0.0,
                jnp.linalg.norm(tangent_first, axis=-1),
                1.0,
            )[:, None]
        )
        tangent_second = jnp.cross(normal, tangent_first)
        relative = point - centroid[:, None, :]
        angle = jnp.arctan2(
            oe.contract("ped,pd->pe", relative, tangent_second, backend="jax"),
            oe.contract("ped,pd->pe", relative, tangent_first, backend="jax"),
        )
        order = jnp.argsort(jnp.where(crossing, angle, jnp.inf), axis=-1)
        ordered_point = jnp.take_along_axis(point, order[..., None], axis=1)
        ordered_pressure = jnp.take_along_axis(pressure, order, axis=1)
        ordered_plus = jnp.take_along_axis(plus_route, order[..., None], axis=1)
        ordered_minus = jnp.take_along_axis(minus_route, order[..., None], axis=1)
        triangle_index = jnp.arange(1, self.edge_points.shape[1] - 1)
        first_point = ordered_point[:, :1, :]
        second_point = ordered_point[:, triangle_index, :]
        third_point = ordered_point[:, triangle_index + 1, :]
        area_vector = 0.5 * jnp.cross(
            second_point - first_point, third_point - first_point
        )
        area = jnp.sqrt(jnp.sum(area_vector * area_vector, axis=-1))
        triangle_valid = (
            self.pair_valid[:, None]
            & (triangle_index[None, :] + 1 < count[:, None])
            & (area > self.tolerance)
            & (gradient_norm[:, None] > self.tolerance)
        )
        triangle_point = (first_point + second_point + third_point) / 3.0
        triangle_pressure = (
            ordered_pressure[:, :1]
            + ordered_pressure[:, triangle_index]
            + ordered_pressure[:, triangle_index + 1]
        ) / 3.0
        triangle_plus = (
            ordered_plus[:, :1, :]
            + ordered_plus[:, triangle_index, :]
            + ordered_plus[:, triangle_index + 1, :]
        ) / 3.0
        triangle_minus = (
            ordered_minus[:, :1, :]
            + ordered_minus[:, triangle_index, :]
            + ordered_minus[:, triangle_index + 1, :]
        ) / 3.0
        flat_valid = triangle_valid.reshape((-1,))
        actual = jnp.sum(flat_valid, dtype=jnp.int32)
        selected = jnp.nonzero(flat_valid, size=self.patch_capacity, fill_value=0)[0]
        overflow = jnp.maximum(actual - self.patch_capacity, 0)
        valid = (jnp.arange(self.patch_capacity) < actual) & (overflow == 0)
        pair_for_triangle = jnp.repeat(
            jnp.arange(self.pair_valid.size, dtype=jnp.int32),
            triangle_valid.shape[1],
        )[selected]
        points = triangle_point.reshape((-1, 3))[selected]
        pressures = triangle_pressure.reshape((-1,))[selected]
        weights = area.reshape((-1,))[selected]
        plus_interpolation = triangle_plus.reshape((-1, 4))[selected]
        minus_interpolation = triangle_minus.reshape((-1, 4))[selected]
        normals = normal[pair_for_triangle]
        plus_source = self.plus_cell_ids[pair_for_triangle]
        minus_source = self.minus_cell_ids[pair_for_triangle]
        points = jnp.where(valid[:, None], points, 0.0)
        normals = jnp.where(valid[:, None], normals, 0.0)
        pressures = jnp.where(valid, pressures, 0.0)
        weights = jnp.where(valid, weights, 0.0)
        plus_interpolation = jnp.where(valid[:, None], plus_interpolation, 0.0)
        minus_interpolation = jnp.where(valid[:, None], minus_interpolation, 0.0)
        plus_source = jnp.where(valid, plus_source, -1)
        minus_source = jnp.where(valid, minus_source, -1)
        patch = HydroelasticPressurePatch(
            points,
            normals,
            pressures,
            weights,
            plus_source,
            minus_source,
            plus_interpolation,
            minus_interpolation,
            valid,
            canonical_fingerprint(
                {
                    "kind": "hydroelastic-pressure-patch",
                    "prepared": self.prepared_id,
                    "capacity": self.patch_capacity,
                }
            ),
        )
        plus_quad = oe.contract(
            "qk,qk->q",
            plus_interpolation,
            plus_local[pair_for_triangle],
            backend="jax",
        )
        minus_quad = oe.contract(
            "qk,qk->q",
            minus_interpolation,
            minus_local[pair_for_triangle],
            backend="jax",
        )
        balance_residual = jnp.max(
            jnp.where(valid, jnp.abs(plus_quad - minus_quad), 0.0), initial=0.0
        )
        finite_state = (
            jnp.all(jnp.isfinite(plus_state.pressure))
            & jnp.all(jnp.isfinite(minus_state.pressure))
            & jnp.all(plus_state.pressure >= 0.0)
            & jnp.all(minus_state.pressure >= 0.0)
        )
        tie_margin = jnp.min(
            jnp.where(crossing, edge_tie_margin, jnp.inf), initial=jnp.inf
        )
        relevant_pair = self.pair_valid & (count >= 3)
        ambiguous_zero = self.pair_valid & jnp.all(
            jnp.where(
                self.edge_valid[..., None],
                jnp.abs(difference) <= self.tolerance * scale[..., None],
                True,
            ),
            axis=(1, 2),
        )
        derivative_valid = (
            finite_state
            & (tie_margin > self.tolerance)
            & ~jnp.any(ambiguous_zero)
            & jnp.all(jnp.where(relevant_pair, gradient_norm > self.tolerance, True))
        )
        finite = (
            finite_state
            & jnp.all(jnp.isfinite(points))
            & jnp.all(jnp.isfinite(pressures))
            & jnp.all(jnp.isfinite(weights))
        )
        complete = self.preparation_complete & finite & (overflow == 0)
        successful = (
            complete & derivative_valid & (balance_residual <= 64.0 * self.tolerance)
        )
        evidence = HydroelasticPatchEvidence(
            jnp.sum(self.pair_valid, dtype=jnp.int32),
            actual,
            overflow,
            jnp.sum(weights),
            jnp.min(jnp.where(valid, pressures, jnp.inf), initial=jnp.inf),
            balance_residual,
            self.predicate_margin,
            tie_margin,
            finite,
            derivative_valid,
            complete,
            successful,
            self.plus.field_id,
            self.minus.field_id,
        )
        return HydroelasticPatchExtraction(patch, evidence)


def _tetrahedron_basis(vertices: np.ndarray, /) -> np.ndarray:
    jacobian = (vertices[1:] - vertices[0]).T
    inverse = np.linalg.inv(jacobian)
    gradients = np.concatenate((-np.sum(inverse, axis=0)[None, :], inverse), axis=0)
    return gradients


def _barycentric_weights(
    points: np.ndarray,
    vertices: np.ndarray,
    gradients: np.ndarray,
    /,
) -> np.ndarray:
    del gradients
    jacobian = (vertices[1:] - vertices[0]).T
    inverse = np.linalg.inv(jacobian)
    reduced = oe.contract("ij,...j->...i", inverse, points - vertices[0])
    return np.concatenate(
        (1.0 - np.sum(reduced, axis=-1, keepdims=True), reduced), axis=-1
    )


__all__ = [
    "HydroelasticPatchEvidence",
    "HydroelasticPatchExtraction",
    "HydroelasticPatchExtractionPlan",
    "HydroelasticPressureFieldPlan",
    "HydroelasticPressureFieldState",
    "HydroelasticPressurePatch",
    "PreparedHydroelasticPatchExtraction",
]
