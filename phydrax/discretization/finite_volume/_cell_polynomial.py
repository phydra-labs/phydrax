#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._total_degree import TotalDegreePolynomialFeatures
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._geometry_protocol import FiniteVolumeStageMetrics
from ._unstructured import UnstructuredFiniteVolumeDiscretization


def _positive_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _monomials(values: np.ndarray, exponents: np.ndarray, /) -> np.ndarray:
    return np.prod(values[..., None, :] ** exponents, axis=-1)


def _jax_monomials(values: Array, exponents: Array, /) -> Array:
    return jnp.prod(values[..., None, :] ** exponents, axis=-1)


def _adjacency(discretization: UnstructuredFiniteVolumeDiscretization, /):
    adjacency = [set() for _ in range(discretization.cell_count)]
    owner = np.asarray(discretization.owner_cells, dtype=np.int32)
    neighbour = np.asarray(discretization.neighbour_cells, dtype=np.int32)
    for left, right in zip(owner, neighbour, strict=True):
        if right >= 0:
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))
    return adjacency


def _bfs_order(adjacency, cell: int, /):
    visited = {cell}
    frontier = sorted(adjacency[cell])
    ordered = []
    depth = 1
    while frontier:
        next_frontier = []
        for candidate in frontier:
            if candidate in visited:
                continue
            visited.add(candidate)
            ordered.append((candidate, depth))
            next_frontier.extend(adjacency[candidate])
        frontier = sorted(set(next_frontier) - visited)
        depth += 1
    return ordered


def _cell_moments(
    discretization: UnstructuredFiniteVolumeDiscretization,
    basis: "CellPolynomialBasis",
    /,
):
    centers = np.asarray(discretization.cell_centers)
    volumes = np.asarray(discretization.cell_volumes)
    lengths = volumes ** (1.0 / discretization.cell_dimension)
    quadrature_points = np.asarray(discretization.cell_quadrature_points)
    quadrature_weights = np.asarray(discretization.cell_quadrature_weights)
    normalized = (quadrature_points - centers[:, None, :]) / lengths[:, None, None]
    monomials = _monomials(normalized, np.asarray(basis.exponents))
    moments = oe.contract("cq,cqf->cf", quadrature_weights / volumes[:, None], monomials)
    return moments, lengths


def _design_rows(
    discretization: UnstructuredFiniteVolumeDiscretization,
    basis: "CellPolynomialBasis",
    moments: np.ndarray,
    lengths: np.ndarray,
    cell: int,
    stencil: list[int],
    /,
):
    centers = np.asarray(discretization.cell_centers)
    volumes = np.asarray(discretization.cell_volumes)
    points = np.asarray(discretization.cell_quadrature_points)[stencil]
    weights = np.asarray(discretization.cell_quadrature_weights)[stencil]
    normalized = (points - centers[cell]) / lengths[cell]
    monomials = _monomials(normalized, np.asarray(basis.exponents))
    averages = oe.contract("sq,sqf->sf", weights / volumes[stencil, None], monomials)
    return averages - moments[cell]


def _selected_stencils(
    discretization: UnstructuredFiniteVolumeDiscretization,
    basis: "CellPolynomialBasis",
    moments: np.ndarray,
    lengths: np.ndarray,
    oversampling: int,
    direction: np.ndarray | None,
    /,
):
    adjacency = _adjacency(discretization)
    centers = np.asarray(discretization.cell_centers)
    stencils: list[tuple[int, ...]] = []
    depths: list[int] = []
    required = basis.feature_count + oversampling
    for cell in range(discretization.cell_count):
        candidates = _bfs_order(adjacency, cell)
        if direction is not None:
            candidates = sorted(
                candidates,
                key=lambda item: (
                    item[1],
                    -float(
                        np.dot(
                            (centers[item[0]] - centers[cell]) / lengths[cell],
                            direction,
                        )
                    ),
                    item[0],
                ),
            )
        if len(candidates) < basis.feature_count:
            raise ValueError(
                f"Cell polynomial stencil for cell {cell} has fewer cells than features."
            )
        selected: list[int] = []
        selected_depth = 0
        rank = 0
        target = min(len(candidates), required)
        for candidate, depth in candidates:
            selected.append(candidate)
            selected_depth = max(selected_depth, depth)
            if len(selected) >= target:
                design = _design_rows(
                    discretization,
                    basis,
                    moments,
                    lengths,
                    cell,
                    selected,
                )
                rank = np.linalg.matrix_rank(design)
                if rank == basis.feature_count:
                    break
        if rank != basis.feature_count:
            raise ValueError(
                f"Cell polynomial stencil for cell {cell} is rank deficient."
            )
        stencils.append(tuple(selected))
        depths.append(selected_depth)
    capacity = max(len(stencil) for stencil in stencils)
    indices = np.zeros((discretization.cell_count, capacity), dtype=np.int32)
    valid = np.zeros((discretization.cell_count, capacity), dtype=bool)
    for cell, stencil in enumerate(stencils):
        indices[cell, : len(stencil)] = stencil
        valid[cell, : len(stencil)] = True
    return indices, valid, np.asarray(depths, dtype=np.int32)


def _smoothness_gram(
    discretization: UnstructuredFiniteVolumeDiscretization,
    basis: "CellPolynomialBasis",
    lengths: np.ndarray,
    /,
):
    centers = np.asarray(discretization.cell_centers)
    volumes = np.asarray(discretization.cell_volumes)
    points = np.asarray(discretization.cell_quadrature_points)
    weights = np.asarray(discretization.cell_quadrature_weights) / volumes[:, None]
    normalized = (points - centers[:, None, :]) / lengths[:, None, None]
    exponents = np.asarray(basis.exponents, dtype=np.int32)
    derivatives = np.asarray(
        TotalDegreePolynomialFeatures(basis.dimension, basis.degree).exponents,
        dtype=np.int32,
    )
    gram = np.zeros((discretization.cell_count, basis.feature_count, basis.feature_count))
    for derivative in derivatives:
        active = np.all(exponents >= derivative[None, :], axis=1)
        reduced = np.maximum(exponents - derivative[None, :], 0)
        coefficient = np.ones((basis.feature_count,))
        for axis in range(basis.dimension):
            for count in range(int(derivative[axis])):
                coefficient *= np.maximum(exponents[:, axis] - count, 0)
        values = coefficient[None, None, :] * _monomials(normalized, reduced)
        values[..., ~active] = 0.0
        gram += oe.contract("cq,cqi,cqj->cij", weights, values, values)
    return gram


class CellPolynomialBasis(StrictModule, NonTrainableState):
    """Nonconstant total-degree basis for conservative cell-average polynomials."""

    exponents: Array
    dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(self, dimension: int, degree: int, /):
        features = TotalDegreePolynomialFeatures(dimension, degree)
        if features.feature_count == 0:
            raise ValueError("Cell polynomial degree must be positive.")
        self.exponents = features.exponents
        self.dimension = features.dimension
        self.degree = features.degree
        self.feature_count = features.feature_count
        self.basis_id = canonical_fingerprint(
            {
                "kind": "cell-average-total-degree-basis",
                "dimension": features.dimension,
                "degree": features.degree,
                "exponents": array_tree_fingerprint(features.exponents),
            }
        )


class CellPolynomialReconstructionReport(StrictModule):
    maximum_condition_number: Array
    minimum_singular_value: Array
    minimum_rank: Array
    maximum_stencil_depth: Array
    worst_cell: Array
    stencil_capacity: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)


class CellPolynomialReconstructionPlan(StrictModule, NonTrainableState):
    """Preparation policy for one normalized k-exact cell polynomial."""

    degree: int = eqx.field(static=True)
    weight_power: float = eqx.field(static=True)
    oversampling: int = eqx.field(static=True)
    rcond: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        weight_power: float = 2.0,
        oversampling: int = 2,
        rcond: float = 1e-12,
        condition_limit: float = 1e8,
    ):
        degree_ = _positive_integer(degree, "degree")
        oversampling_ = int(oversampling)
        if oversampling_ < 0:
            raise ValueError("oversampling must be nonnegative.")
        if not np.isfinite(weight_power) or weight_power < 0.0:
            raise ValueError("weight_power must be finite and nonnegative.")
        if not np.isfinite(rcond) or rcond <= 0.0:
            raise ValueError("rcond must be positive and finite.")
        if not np.isfinite(condition_limit) or condition_limit <= 1.0:
            raise ValueError("condition_limit must be finite and greater than one.")
        self.degree = degree_
        self.weight_power = float(weight_power)
        self.oversampling = oversampling_
        self.rcond = float(rcond)
        self.condition_limit = float(condition_limit)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cell-polynomial-reconstruction-plan",
                "degree": degree_,
                "weight_power": float(weight_power),
                "oversampling": oversampling_,
                "rcond": float(rcond),
                "condition_limit": float(condition_limit),
            }
        )

    def prepare(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        stencil_direction: ArrayLike | None = None,
    ) -> "PreparedCellPolynomialReconstruction":
        return PreparedCellPolynomialReconstruction(
            self,
            discretization,
            stencil_direction=stencil_direction,
        )


class PreparedCellPolynomialReconstruction(StrictModule, NonTrainableState):
    """Fixed-capacity cell-average polynomial reconstruction."""

    discretization: UnstructuredFiniteVolumeDiscretization
    basis: CellPolynomialBasis
    moments: Array
    characteristic_lengths: Array
    stencil_cells: Array
    stencil_valid: Array
    factors: Array
    smoothness_gram: Array
    report: CellPolynomialReconstructionReport
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: CellPolynomialReconstructionPlan,
        discretization: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        stencil_direction: ArrayLike | None = None,
    ):
        if not isinstance(plan, CellPolynomialReconstructionPlan):
            raise TypeError("plan must be CellPolynomialReconstructionPlan.")
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Cell polynomials require unstructured FV geometry.")
        required_quadrature_degree = max(plan.degree, 2 * (plan.degree - 1))
        if required_quadrature_degree > discretization.cell_quadrature_degree:
            raise ValueError(
                "Cell quadrature degree is insufficient for polynomial smoothness data."
            )
        basis = CellPolynomialBasis(discretization.cell_dimension, plan.degree)
        moments, lengths = _cell_moments(discretization, basis)
        direction = None
        if stencil_direction is not None:
            direction = np.asarray(stencil_direction, dtype=float)
            if direction.shape != (discretization.cell_dimension,):
                raise ValueError(
                    "stencil_direction must have one entry per spatial dimension."
                )
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm == 0.0:
                raise ValueError("stencil_direction must be finite and nonzero.")
            direction = direction / norm
        indices, valid, depths = _selected_stencils(
            discretization,
            basis,
            moments,
            lengths,
            plan.oversampling,
            direction,
        )
        centers = np.asarray(discretization.cell_centers)
        capacity = indices.shape[1]
        factors = np.zeros((discretization.cell_count, basis.feature_count, capacity))
        conditions = np.empty((discretization.cell_count,))
        minimum_singular = np.empty((discretization.cell_count,))
        ranks = np.empty((discretization.cell_count,), dtype=np.int32)
        for cell in range(discretization.cell_count):
            stencil = indices[cell, valid[cell]].tolist()
            stencil_count = len(stencil)
            design = _design_rows(
                discretization,
                basis,
                moments,
                lengths,
                cell,
                stencil,
            )
            distance = np.linalg.norm(
                (centers[stencil] - centers[cell]) / lengths[cell], axis=-1
            )
            weights = 1.0 / np.maximum(distance, 1e-14) ** plan.weight_power
            root_weight = np.sqrt(weights)
            weighted_design = root_weight[:, None] * design
            column_scale = np.linalg.norm(weighted_design, axis=0)
            if np.any(~np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
                raise ValueError(
                    f"Cell polynomial design for cell {cell} has a zero column."
                )
            normalized_design = weighted_design / column_scale[None, :]
            left, singular, right_t = np.linalg.svd(
                normalized_design, full_matrices=False
            )
            rank = int(np.sum(singular > plan.rcond * singular[0]))
            condition = singular[0] / singular[-1]
            if (
                rank != basis.feature_count
                or not np.isfinite(condition)
                or condition > plan.condition_limit
            ):
                raise ValueError(
                    f"Cell polynomial design for cell {cell} violates rank/condition policy."
                )
            pseudoinverse = oe.contract("ij,nj->in", right_t.T / singular[None, :], left)
            factors[cell, :, :stencil_count] = (
                pseudoinverse * root_weight[None, :]
            ) / column_scale[:, None]
            conditions[cell] = condition
            minimum_singular[cell] = singular[-1]
            ranks[cell] = rank
        gram = _smoothness_gram(discretization, basis, lengths)
        worst = int(np.argmax(conditions))
        self.discretization = discretization
        self.basis = basis
        self.moments = jnp.asarray(moments)
        self.characteristic_lengths = jnp.asarray(lengths)
        self.stencil_cells = jnp.asarray(indices)
        self.stencil_valid = jnp.asarray(valid)
        self.factors = jnp.asarray(factors)
        self.smoothness_gram = jnp.asarray(gram)
        self.report = CellPolynomialReconstructionReport(
            maximum_condition_number=jnp.asarray(conditions[worst]),
            minimum_singular_value=jnp.asarray(np.min(minimum_singular)),
            minimum_rank=jnp.asarray(np.min(ranks), dtype=jnp.int32),
            maximum_stencil_depth=jnp.asarray(np.max(depths), dtype=jnp.int32),
            worst_cell=jnp.asarray(worst, dtype=jnp.int32),
            stencil_capacity=capacity,
            feature_count=basis.feature_count,
            degree=basis.degree,
        )
        self.plan_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-cell-polynomial-reconstruction",
                "plan": plan.plan_id,
                "geometry": discretization.prepared_id,
                "direction": None
                if direction is None
                else array_tree_fingerprint(direction),
                "stencils": array_tree_fingerprint(indices),
                "valid": array_tree_fingerprint(valid),
                "moments": array_tree_fingerprint(moments),
            }
        )

    def coefficients(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[0] != self.discretization.cell_count:
            raise ValueError("Cell polynomial state must begin with cell count.")
        difference = value[self.stencil_cells] - value[:, None, ...]
        mask = self.stencil_valid.reshape(
            self.stencil_valid.shape + (1,) * (difference.ndim - 2)
        )
        return oe.contract(
            "cfs,cs...->c...f",
            self.factors.astype(value.dtype),
            jnp.where(mask, difference, 0.0),
        )

    def stage_coefficients(
        self,
        state: Array,
        metrics: FiniteVolumeStageMetrics,
        /,
    ) -> tuple[Array, Array]:
        """Refresh degree-one WLSQ factors from fixed-topology stage centers."""
        if not isinstance(metrics, FiniteVolumeStageMetrics):
            raise TypeError("metrics must be FiniteVolumeStageMetrics.")
        if self.basis.degree not in (1, 2):
            raise ValueError(
                "Moving reconstruction certifies degree one generally and "
                "degree two for rigid translations."
            )
        value = jnp.asarray(state)
        centers = jnp.asarray(metrics.cell_centers, dtype=value.dtype)
        if centers.shape != self.discretization.cell_centers.shape:
            raise ValueError("Stage centers do not match reconstruction topology.")
        lengths = jnp.asarray(metrics.effective_cell_volumes, dtype=value.dtype) ** (
            1.0 / self.basis.dimension
        )
        if self.basis.degree == 2:
            reference_centers = self.discretization.cell_centers.astype(value.dtype)
            translation = centers - reference_centers
            tolerance = (
                128.0
                * jnp.finfo(value.dtype).eps
                * jnp.maximum(jnp.max(jnp.abs(reference_centers)), 1.0)
            )
            rigid = jnp.all(
                jnp.abs(translation - translation[:1]) <= tolerance
            ) & jnp.all(
                jnp.abs(metrics.effective_cell_volumes - self.discretization.cell_volumes)
                <= tolerance
            )
            value = eqx.error_if(
                value,
                ~rigid,
                "Moving degree-two reconstruction requires rigid translation.",
            )
            return self.coefficients(value), self.characteristic_lengths
        neighbour_centers = centers[self.stencil_cells]
        design = _jax_monomials(
            (neighbour_centers - centers[:, None, :]) / lengths[:, None, None],
            self.basis.exponents,
        )
        mask = self.stencil_valid
        distance = jnp.sqrt(jnp.sum(design * design, axis=-1))
        weight = jnp.where(
            mask,
            1.0 / jnp.maximum(distance, 1.0e-14) ** 2,
            0.0,
        )
        gram = oe.contract("cs,csi,csj->cij", weight, design, design, backend="jax")
        right = oe.contract(
            "cs,csi,cs...->ci...",
            weight,
            design,
            jnp.where(
                mask.reshape(mask.shape + (1,) * (value.ndim - 1)),
                value[self.stencil_cells] - value[:, None, ...],
                0.0,
            ),
            backend="jax",
        )
        coefficients = jnp.linalg.solve(
            gram,
            right.reshape((right.shape[0], right.shape[1], -1)),
        )
        coefficients = jnp.moveaxis(coefficients, 1, -1).reshape(
            value.shape + (self.basis.feature_count,)
        )
        return coefficients, lengths

    def evaluate_stage_coefficients(
        self,
        state: Array,
        coefficients: Array,
        lengths: Array,
        metrics: FiniteVolumeStageMetrics,
        cell_routes: Array,
        points: Array,
        /,
    ) -> Array:
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        evaluation_points = jnp.asarray(points)
        normalized = (
            evaluation_points - metrics.cell_centers[routes, None, :]
        ) / lengths[routes, None, None]
        basis = (
            _jax_monomials(normalized, self.basis.exponents)
            - self.moments[routes, None, :]
        )
        delta = oe.contract(
            "r...f,rqf->rq...",
            coefficients[routes],
            basis,
            backend="jax",
        )
        return jnp.asarray(state)[routes, None, ...] + delta

    def basis_values(self, cell_routes: Array, points: Array, /) -> Array:
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        evaluation_points = jnp.asarray(points)
        if routes.ndim != 1:
            raise ValueError("cell_routes must be one-dimensional.")
        if evaluation_points.ndim != 3 or evaluation_points.shape[0] != routes.size:
            raise ValueError("points must have shape (routes, points, dimension).")
        if evaluation_points.shape[-1] != self.basis.dimension:
            raise ValueError("Point dimension does not match the cell polynomial basis.")
        dtype = evaluation_points.dtype
        centers = self.discretization.cell_centers.astype(dtype)[routes]
        lengths = self.characteristic_lengths.astype(dtype)[routes]
        normalized = (evaluation_points - centers[:, None, :]) / lengths[:, None, None]
        monomials = _jax_monomials(normalized, self.basis.exponents)
        return monomials - self.moments.astype(dtype)[routes, None, :]

    def evaluate_coefficients(
        self,
        state: Array,
        coefficients: Array,
        cell_routes: Array,
        points: Array,
        /,
    ) -> Array:
        value = jnp.asarray(state)
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        basis = self.basis_values(routes, jnp.asarray(points, dtype=value.dtype))
        delta = oe.contract("r...f,rqf->rq...", coefficients[routes], basis)
        return value[routes, None, ...] + delta

    def evaluate(self, state: Array, cell_routes: Array, points: Array, /) -> Array:
        return self.evaluate_coefficients(
            state,
            self.coefficients(state),
            cell_routes,
            points,
        )

    def reconstruct_at(self, state: Array, points: Array, /) -> tuple[Array, Array]:
        owner = self.discretization.owner_cells
        neighbour = self.discretization.neighbour_cells
        safe_neighbour = jnp.maximum(neighbour, 0)
        coefficients = self.coefficients(state)
        return (
            self.evaluate_coefficients(state, coefficients, owner, points),
            self.evaluate_coefficients(state, coefficients, safe_neighbour, points),
        )

    def reconstruct(self, state: Array, /) -> tuple[Array, Array]:
        left, right = self.reconstruct_at(
            state, self.discretization.face_centers[:, None, :]
        )
        return left[:, 0], right[:, 0]

    def smoothness(self, coefficients: Array, /) -> Array:
        value = jnp.asarray(coefficients)
        return oe.contract(
            "c...i,cij,c...j->c...",
            value,
            self.smoothness_gram.astype(value.dtype),
            value,
        )


__all__ = [
    "CellPolynomialBasis",
    "CellPolynomialReconstructionPlan",
    "CellPolynomialReconstructionReport",
    "PreparedCellPolynomialReconstruction",
]
