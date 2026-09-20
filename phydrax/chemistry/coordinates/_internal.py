#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Regularized redundant molecular internals and trust-bounded Cartesian retraction."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    solve,
)


class InternalCoordinateState(StrictModule, NonTrainableState):
    values: Array
    jacobian: Array
    singular_values: Array
    rank: Array
    condition_estimate: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        values,
        jacobian,
        singular_values,
        rank,
        condition_estimate,
        successful,
        plan_id,
        /,
    ):
        values_ = jnp.asarray(values)
        jacobian_ = jnp.asarray(jacobian, dtype=values_.dtype)
        singular = jnp.asarray(singular_values, dtype=values_.dtype)
        if values_.ndim != 1 or jacobian_.shape[0] != values_.size or singular.ndim != 1:
            raise ValueError(
                "Internal values, Jacobian, and singular spectrum do not align."
            )
        self.values = values_
        self.jacobian = jacobian_
        self.singular_values = singular
        self.rank = jnp.asarray(rank, dtype=jnp.int32).reshape(())
        self.condition_estimate = jnp.asarray(
            condition_estimate, dtype=values_.dtype
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.plan_id = str(plan_id)
        self.state_id = canonical_fingerprint(
            {
                "kind": "internal-coordinate-state",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "values": np.asarray(values_),
                        "jacobian": np.asarray(jacobian_),
                        "singular_values": np.asarray(singular),
                    }
                ),
            }
        )


class InternalCoordinateRetractionResult(StrictModule, NonTrainableState):
    positions: Array
    achieved_values: Array
    target_values: Array
    residual: Array
    iterations: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        positions,
        achieved_values,
        target_values,
        residual,
        iterations,
        successful,
        plan_id,
        /,
    ):
        positions_ = jnp.asarray(positions)
        achieved = jnp.asarray(achieved_values, dtype=positions_.dtype)
        target = jnp.asarray(target_values, dtype=positions_.dtype)
        residual_ = jnp.asarray(residual, dtype=positions_.dtype)
        if (
            positions_.ndim != 2
            or positions_.shape[1] != 3
            or achieved.shape != target.shape
            or residual_.shape != target.shape
        ):
            raise ValueError("Internal-coordinate retraction arrays do not align.")
        self.positions = positions_
        self.achieved_values = achieved
        self.target_values = target
        self.residual = residual_
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "internal-coordinate-retraction-result",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "positions": np.asarray(positions_),
                        "target": np.asarray(target),
                        "residual": np.asarray(residual_),
                    }
                ),
            }
        )


class MolecularCoordinateSystemPlan(StrictModule, NonTrainableState):
    atom_count: int = eqx.field(static=True)
    bonds: tuple[tuple[int, int], ...] = eqx.field(static=True)
    angles: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    dihedrals: tuple[tuple[int, int, int, int], ...] = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        atom_count: int,
        /,
        *,
        bonds=(),
        angles=(),
        dihedrals=(),
        regularization: float = 1.0e-12,
        rank_tolerance: float = 1.0e-9,
    ):
        count = int(atom_count)
        bonds_ = tuple(tuple(item) for item in bonds)
        angles_ = tuple(tuple(item) for item in angles)
        dihedrals_ = tuple(tuple(item) for item in dihedrals)
        regularization_ = float(regularization)
        rank_tolerance_ = float(rank_tolerance)
        groups = ((bonds_, 2), (angles_, 3), (dihedrals_, 4))
        if (
            count < 2
            or not any(group for group, _ in groups)
            or any(
                len(item) != width
                or len(set(item)) != width
                or any(index < 0 or index >= count for index in item)
                for group, width in groups
                for item in group
            )
            or not isfinite(regularization_)
            or regularization_ <= 0.0
            or not isfinite(rank_tolerance_)
            or rank_tolerance_ <= 0.0
        ):
            raise ValueError(
                "Molecular internal-coordinate topology or tolerances are invalid."
            )
        if (
            len(set(bonds_)) != len(bonds_)
            or len(set(angles_)) != len(angles_)
            or len(set(dihedrals_)) != len(dihedrals_)
        ):
            raise ValueError("Internal-coordinate tuples must be unique.")
        bonds_typed = tuple((item[0], item[1]) for item in bonds_)
        angles_typed = tuple((item[0], item[1], item[2]) for item in angles_)
        dihedrals_typed = tuple(
            (item[0], item[1], item[2], item[3]) for item in dihedrals_
        )
        self.atom_count = count
        self.bonds = bonds_typed
        self.angles = angles_typed
        self.dihedrals = dihedrals_typed
        self.regularization = regularization_
        self.rank_tolerance = rank_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "molecular-coordinate-system-plan",
                "atom_count": count,
                "bonds": [list(value) for value in bonds_],
                "angles": [list(value) for value in angles_],
                "dihedrals": [list(value) for value in dihedrals_],
                "regularization": regularization_,
                "rank_tolerance": rank_tolerance_,
            }
        )

    def values(self, positions: ArrayLike, /) -> Array:
        coordinate = jnp.asarray(positions)
        if coordinate.shape != (self.atom_count, 3):
            raise ValueError("Molecular coordinates must have shape (atom_count, 3).")
        values = []
        for first, second in self.bonds:
            delta = coordinate[first] - coordinate[second]
            values.append(jnp.sqrt(jnp.sum(delta**2) + self.regularization**2))
        for first, center, third in self.angles:
            left = coordinate[first] - coordinate[center]
            right = coordinate[third] - coordinate[center]
            cross_norm = jnp.sqrt(
                jnp.sum(jnp.cross(left, right) ** 2) + self.regularization**2
            )
            values.append(jnp.arctan2(cross_norm, jnp.dot(left, right)))
        for first, second, third, fourth in self.dihedrals:
            first_bond = coordinate[second] - coordinate[first]
            center_bond = coordinate[third] - coordinate[second]
            last_bond = coordinate[fourth] - coordinate[third]
            center_unit = center_bond / jnp.sqrt(
                jnp.sum(center_bond**2) + self.regularization**2
            )
            first_plane = first_bond - jnp.dot(first_bond, center_unit) * center_unit
            last_plane = last_bond - jnp.dot(last_bond, center_unit) * center_unit
            numerator = jnp.dot(jnp.cross(first_plane, last_plane), center_unit)
            denominator = jnp.dot(first_plane, last_plane)
            values.append(jnp.arctan2(numerator, denominator))
        return jnp.stack(tuple(values))

    def evaluate(self, positions: ArrayLike, /) -> InternalCoordinateState:
        coordinate = jnp.asarray(positions)
        values = self.values(coordinate)
        jacobian = jax.jacfwd(self.values)(coordinate).reshape((values.size, -1))
        singular = np.linalg.svd(np.asarray(jacobian), compute_uv=False)
        scale = float(singular[0]) if singular.size else 1.0
        retained = singular > self.rank_tolerance * scale
        rank = int(np.count_nonzero(retained))
        condition = float(singular[0] / singular[retained][-1]) if rank else np.inf
        successful = (
            np.all(np.isfinite(np.asarray(values)))
            and np.all(np.isfinite(np.asarray(jacobian)))
            and rank > 0
        )
        return InternalCoordinateState(
            values,
            jacobian,
            singular,
            rank,
            condition,
            successful,
            self.plan_id,
        )

    def _residual(self, target, current):
        residual = target - current
        start = len(self.bonds) + len(self.angles)
        if self.dihedrals:
            periodic = residual[start:]
            periodic = jnp.arctan2(jnp.sin(periodic), jnp.cos(periodic))
            residual = residual.at[start:].set(periodic)
        return residual

    def retract(
        self,
        positions: ArrayLike,
        internal_step: ArrayLike,
        /,
        *,
        trust_radius: float = 0.2,
        tolerance: float = 1.0e-10,
        maximum_iterations: int = 32,
    ) -> InternalCoordinateRetractionResult:
        coordinate = jnp.asarray(positions)
        step = jnp.asarray(internal_step, dtype=coordinate.dtype)
        initial = self.values(coordinate)
        if step.shape != initial.shape:
            raise ValueError("Internal step must align with coordinate values.")
        trust = float(trust_radius)
        tolerance_ = float(tolerance)
        maximum = int(maximum_iterations)
        if (
            not isfinite(trust)
            or trust <= 0.0
            or not isfinite(tolerance_)
            or tolerance_ <= 0.0
            or maximum <= 0
        ):
            raise ValueError(
                "Retraction trust radius, tolerance, or iteration limit is invalid."
            )
        target = initial + step
        completed = 0
        successful = False
        residual = self._residual(target, initial)
        for iteration in range(maximum):
            state = self.evaluate(coordinate)
            residual = self._residual(target, state.values)
            if bool(jnp.max(jnp.abs(residual), initial=0.0) <= tolerance_):
                successful = True
                completed = iteration
                break
            linear = solve(
                LeastSquaresProblem(DenseLinearOperator(state.jacobian)),
                residual,
                policy=LinearSolvePolicy(DenseSVD()),
            )
            displacement = linear.value.reshape(coordinate.shape)
            norm = jnp.sqrt(jnp.sum(displacement**2))
            scale = jnp.minimum(
                1.0, trust / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny)
            )
            coordinate = coordinate + scale * displacement
            completed = iteration + 1
            if not bool(linear.successful):
                break
        achieved = self.values(coordinate)
        residual = self._residual(target, achieved)
        successful = bool(jnp.max(jnp.abs(residual), initial=0.0) <= tolerance_) and bool(
            jnp.all(jnp.isfinite(coordinate))
        )
        return InternalCoordinateRetractionResult(
            coordinate,
            achieved,
            target,
            residual,
            completed,
            successful,
            self.plan_id,
        )


__all__ = [
    "InternalCoordinateRetractionResult",
    "InternalCoordinateState",
    "MolecularCoordinateSystemPlan",
]
