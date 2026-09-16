#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Polynomially reproducing RBF-FD and GMLS point-cloud operators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import factorial
from typing import Literal, TypeAlias

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint


MeshfreeOperatorKind: TypeAlias = Literal["value", "gradient", "laplacian"]
MeshfreeMethod: TypeAlias = Literal["rbf-fd", "gmls"]


def _multiindices(dimension: int, degree: int, /) -> tuple[tuple[int, ...], ...]:
    return tuple(
        exponent
        for exponent in product(range(degree + 1), repeat=dimension)
        if sum(exponent) <= degree
    )


def _monomial_matrix(offsets: np.ndarray, exponents, /) -> np.ndarray:
    return np.stack(
        [np.prod(offsets ** np.asarray(exponent)[None, :], axis=1) for exponent in exponents],
        axis=1,
    )


def _operator_moments(exponents, kind, axis, /) -> np.ndarray:
    values = []
    for exponent in exponents:
        if kind == "value":
            values.append(float(all(power == 0 for power in exponent)))
        elif kind == "gradient":
            values.append(
                float(
                    exponent[axis] == 1
                    and all(
                        power == 0 for index, power in enumerate(exponent) if index != axis
                    )
                )
            )
        else:
            values.append(
                float(
                    sum(
                        factorial(power)
                        if power == 2
                        and all(
                            other == 0
                            for index, other in enumerate(exponent)
                            if index != axis_
                        )
                        else 0
                        for axis_, power in enumerate(exponent)
                    )
                )
            )
    return np.asarray(values, dtype=float)


def _rbf_rhs(offsets: np.ndarray, power: int, kind, axis, /) -> np.ndarray:
    radius = np.linalg.norm(offsets, axis=1)
    if kind == "value":
        return radius**power
    safe = np.where(radius > 0.0, radius, 1.0)
    if kind == "gradient":
        values = -power * offsets[:, axis] * safe ** (power - 2)
        return np.where(radius > 0.0, values, 0.0)
    dimension = offsets.shape[1]
    values = power * (power + dimension - 2) * safe ** (power - 2)
    return np.where(radius > 0.0, values, 0.0)


@dataclass(frozen=True, slots=True)
class MeshfreeStencilPlan:
    """Fixed point cloud and nearest-neighbour topology."""

    coordinates: Array
    neighbours: Array
    polynomial_degree: int
    plan_id: str

    def __init__(
        self,
        coordinates: ArrayLike,
        /,
        *,
        stencil_size: int,
        polynomial_degree: int = 2,
    ):
        points = np.asarray(coordinates, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] == 0:
            raise ValueError("coordinates must have shape (points, dimensions).")
        if not np.all(np.isfinite(points)):
            raise ValueError("coordinates must be finite.")
        if np.unique(points, axis=0).shape[0] != points.shape[0]:
            raise ValueError("meshfree coordinates must be unique.")
        size = int(stencil_size)
        degree = int(polynomial_degree)
        exponent_count = len(_multiindices(points.shape[1], degree))
        if size < exponent_count or size > points.shape[0]:
            raise ValueError(
                "stencil_size must cover the polynomial basis and not exceed point count."
            )
        differences = points[:, None, :] - points[None, :, :]
        distances = np.sum(differences * differences, axis=-1)
        neighbours = np.argsort(distances, axis=1, kind="stable")[:, :size]
        payload = {
            "kind": "meshfree-stencil-plan",
            "coordinates": points.tolist(),
            "neighbours": neighbours.tolist(),
            "polynomial_degree": degree,
        }
        object.__setattr__(self, "coordinates", jnp.asarray(points))
        object.__setattr__(self, "neighbours", jnp.asarray(neighbours, dtype=jnp.int32))
        object.__setattr__(self, "polynomial_degree", degree)
        object.__setattr__(self, "plan_id", canonical_fingerprint(payload))

    def prepare(
        self,
        kind: MeshfreeOperatorKind,
        /,
        *,
        method: MeshfreeMethod = "rbf-fd",
        axis: int | None = None,
        rbf_power: int = 3,
        gmls_scale: float = 1.0,
    ) -> PreparedMeshfreeOperator:
        return prepare_meshfree_operator(
            self,
            kind,
            method=method,
            axis=axis,
            rbf_power=rbf_power,
            gmls_scale=gmls_scale,
        )


@dataclass(frozen=True, slots=True)
class MeshfreeReproductionEvidence:
    maximum_polynomial_error: float
    maximum_condition_number: float
    passed: bool
    evidence_id: str


@dataclass(frozen=True, slots=True)
class PreparedMeshfreeOperator:
    plan: MeshfreeStencilPlan
    kind: MeshfreeOperatorKind
    method: MeshfreeMethod
    axis: int | None
    weights: Array
    evidence: MeshfreeReproductionEvidence
    operator_id: str

    def apply(self, values: ArrayLike, /) -> Array:
        field = jnp.asarray(values)
        if field.shape[0] != self.plan.coordinates.shape[0]:
            raise ValueError("meshfree values must begin with the point axis.")
        gathered = field[self.plan.neighbours]
        weights = self.weights[(...,) + (None,) * (field.ndim - 1)]
        return jnp.sum(weights * gathered, axis=1)

    __call__ = apply


def prepare_meshfree_operator(
    plan: MeshfreeStencilPlan,
    kind: MeshfreeOperatorKind,
    /,
    *,
    method: MeshfreeMethod = "rbf-fd",
    axis: int | None = None,
    rbf_power: int = 3,
    gmls_scale: float = 1.0,
) -> PreparedMeshfreeOperator:
    if not isinstance(plan, MeshfreeStencilPlan):
        raise TypeError("plan must be MeshfreeStencilPlan.")
    if kind not in ("value", "gradient", "laplacian"):
        raise ValueError("Unknown meshfree operator kind.")
    if method not in ("rbf-fd", "gmls"):
        raise ValueError("Unknown meshfree method.")
    dimension = int(plan.coordinates.shape[1])
    axis_ = 0 if axis is None else int(axis)
    if kind == "gradient" and not 0 <= axis_ < dimension:
        raise ValueError("gradient axis is outside the coordinate dimension.")
    if kind != "gradient" and axis is not None:
        raise ValueError("axis is only valid for gradient operators.")
    power = int(rbf_power)
    if power < 3 or power % 2 == 0:
        raise ValueError("Polyharmonic RBF power must be odd and at least three.")
    scale = float(gmls_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("gmls_scale must be finite and positive.")

    points = np.asarray(plan.coordinates)
    neighbours = np.asarray(plan.neighbours)
    exponents = _multiindices(dimension, plan.polynomial_degree)
    moments = _operator_moments(exponents, kind, axis_)
    weights = []
    conditions = []
    reproduction_errors = []
    for center_index, stencil in enumerate(neighbours):
        offsets = points[stencil] - points[center_index]
        polynomial = _monomial_matrix(offsets, exponents)
        if method == "rbf-fd":
            difference = offsets[:, None, :] - offsets[None, :, :]
            radial = np.linalg.norm(difference, axis=-1) ** power
            matrix = np.block(
                [
                    [radial, polynomial],
                    [polynomial.T, np.zeros((len(exponents), len(exponents)))],
                ]
            )
            right = np.concatenate(
                (_rbf_rhs(offsets, power, kind, axis_), moments)
            )
            solution = np.linalg.solve(matrix, right)
            stencil_weights = solution[: stencil.size]
            condition = np.linalg.cond(matrix)
        else:
            radius = np.linalg.norm(offsets, axis=1)
            maximum = max(float(np.max(radius)), np.finfo(float).eps)
            diagonal = np.exp(-((radius / (scale * maximum)) ** 2))
            normal = polynomial.T @ (diagonal[:, None] * polynomial)
            stencil_weights = diagonal * (polynomial @ np.linalg.solve(normal, moments))
            condition = np.linalg.cond(normal)
        reproduction = polynomial.T @ stencil_weights - moments
        weights.append(stencil_weights)
        conditions.append(condition)
        reproduction_errors.append(np.max(np.abs(reproduction)))
    maximum_error = float(np.max(reproduction_errors))
    maximum_condition = float(np.max(conditions))
    evidence_payload = {
        "kind": "meshfree-reproduction-evidence",
        "plan_id": plan.plan_id,
        "operator_kind": kind,
        "method": method,
        "maximum_polynomial_error": maximum_error,
        "maximum_condition_number": maximum_condition,
    }
    evidence = MeshfreeReproductionEvidence(
        maximum_error,
        maximum_condition,
        bool(np.isfinite(maximum_condition) and maximum_error <= 1.0e-9),
        canonical_fingerprint(evidence_payload),
    )
    operator_payload = {
        **evidence_payload,
        "axis": None if kind != "gradient" else axis_,
        "rbf_power": power,
        "gmls_scale": scale,
    }
    return PreparedMeshfreeOperator(
        plan,
        kind,
        method,
        None if kind != "gradient" else axis_,
        jnp.asarray(np.stack(weights)),
        evidence,
        canonical_fingerprint(operator_payload),
    )


__all__ = [
    "MeshfreeMethod",
    "MeshfreeOperatorKind",
    "MeshfreeReproductionEvidence",
    "MeshfreeStencilPlan",
    "PreparedMeshfreeOperator",
    "prepare_meshfree_operator",
]
