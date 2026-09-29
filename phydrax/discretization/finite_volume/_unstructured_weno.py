#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse
from ._cell_polynomial import (
    CellPolynomialReconstructionPlan,
    PreparedCellPolynomialReconstruction,
)
from ._unstructured import UnstructuredFiniteVolumeDiscretization


UnstructuredWENOLimiter: TypeAlias = Literal["none", "cell_extrema"]


class UnstructuredWENOZReconstructionPlan(StrictModule, NonTrainableState):
    """CWENO decomposition with componentwise WENO-Z nonlinear weights."""

    degree: int = eqx.field(static=True)
    weight_power: float = eqx.field(static=True)
    oversampling: int = eqx.field(static=True)
    linear_weights: tuple[float, ...] | None = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    power: int = eqx.field(static=True)
    limiter: UnstructuredWENOLimiter = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int = 2,
        /,
        *,
        weight_power: float = 2.0,
        oversampling: int = 2,
        linear_weights: Sequence[float] | None = None,
        epsilon: float = 1e-12,
        power: int = 2,
        limiter: UnstructuredWENOLimiter = "cell_extrema",
    ) -> None:
        if isinstance(degree, bool) or not isinstance(degree, Integral):
            raise TypeError("degree must be an integer.")
        degree_ = int(degree)
        if degree_ < 2:
            raise ValueError("Unstructured WENO-Z requires degree at least two.")
        oversampling_ = int(oversampling)
        if oversampling_ < 0:
            raise ValueError("oversampling must be nonnegative.")
        if not np.isfinite(weight_power) or weight_power < 0.0:
            raise ValueError("weight_power must be finite and nonnegative.")
        weights = (
            None
            if linear_weights is None
            else tuple(float(value) for value in linear_weights)
        )
        if weights is not None and (
            not weights
            or any(not np.isfinite(value) or value <= 0.0 for value in weights)
            or not np.isclose(sum(weights), 1.0)
        ):
            raise ValueError("linear_weights must be positive, finite, and sum to one.")
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be positive and finite.")
        if isinstance(power, bool) or not isinstance(power, Integral) or int(power) < 1:
            raise ValueError("power must be a positive integer.")
        limiter = parse(limiter, UnstructuredWENOLimiter, "limiter")
        self.degree = degree_
        self.weight_power = float(weight_power)
        self.oversampling = oversampling_
        self.linear_weights = weights
        self.epsilon = float(epsilon)
        self.power = int(power)
        self.limiter = limiter
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-cweno-z-plan",
                "degree": degree_,
                "weight_power": float(weight_power),
                "oversampling": oversampling_,
                "sector_oversampling": 1,
                "linear_weights": weights,
                "epsilon": float(epsilon),
                "power": int(power),
                "limiter": limiter,
            }
        )

    def prepare(
        self, discretization: UnstructuredFiniteVolumeDiscretization, /
    ) -> "PreparedUnstructuredWENOZReconstruction":
        return PreparedUnstructuredWENOZReconstruction(self, discretization)


class PreparedUnstructuredWENOZReconstruction(StrictModule, NonTrainableState):
    """Prepared optimal and sector cell polynomials with WENO-Z blending."""

    optimal: PreparedCellPolynomialReconstruction
    sectors: tuple[PreparedCellPolynomialReconstruction, ...]
    linear_weights: Array
    epsilon: float = eqx.field(static=True)
    power: int = eqx.field(static=True)
    limiter: UnstructuredWENOLimiter = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: UnstructuredWENOZReconstructionPlan,
        discretization: UnstructuredFiniteVolumeDiscretization,
        /,
    ) -> None:
        if not isinstance(plan, UnstructuredWENOZReconstructionPlan):
            raise TypeError("plan must be UnstructuredWENOZReconstructionPlan.")
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Unstructured WENO-Z requires unstructured FV geometry.")
        optimal = CellPolynomialReconstructionPlan(
            plan.degree,
            weight_power=plan.weight_power,
            oversampling=plan.oversampling,
        ).prepare(discretization)
        sector_plan = CellPolynomialReconstructionPlan(
            plan.degree - 1,
            weight_power=plan.weight_power,
            oversampling=1,
        )
        directions = tuple(
            direction
            for axis in range(discretization.cell_dimension)
            for direction in (
                np.eye(discretization.cell_dimension)[axis],
                -np.eye(discretization.cell_dimension)[axis],
            )
        )
        sectors = tuple(
            sector_plan.prepare(discretization, stencil_direction=direction)
            for direction in directions
        )
        candidate_count = 1 + len(sectors)
        if plan.linear_weights is None:
            weights = np.full((candidate_count,), 0.5 / len(sectors))
            weights[0] = 0.5
        else:
            if len(plan.linear_weights) != candidate_count:
                raise ValueError(
                    f"linear_weights must contain {candidate_count} entries."
                )
            weights = np.asarray(plan.linear_weights)
        self.optimal = optimal
        self.sectors = sectors
        self.linear_weights = jnp.asarray(weights)
        self.epsilon = plan.epsilon
        self.power = plan.power
        self.limiter = plan.limiter
        self.plan_id = plan.plan_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-cweno-z",
                "plan": plan.plan_id,
                "geometry": discretization.prepared_id,
                "optimal": optimal.prepared_id,
                "sectors": [sector.prepared_id for sector in sectors],
                "linear_weights": weights.tolist(),
            }
        )

    @property
    def discretization(self) -> UnstructuredFiniteVolumeDiscretization:
        return self.optimal.discretization

    def candidate_coefficients(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return self._candidates(
            value,
            self.optimal.coefficients(value),
            tuple(sector.coefficients(value) for sector in self.sectors),
        )

    def _candidates(
        self, value: Array, optimal: Array, sectors: tuple[Array, ...], /
    ) -> Array:
        feature_count = self.optimal.basis.feature_count
        padded_sectors = []
        for coefficients in sectors:
            padding = feature_count - coefficients.shape[-1]
            padded_sectors.append(
                jnp.pad(
                    coefficients, ((0, 0),) * (coefficients.ndim - 1) + ((0, padding),)
                )
            )
        sector_stack = jnp.stack(tuple(padded_sectors), axis=0)
        weights = self.linear_weights.astype(value.dtype)
        sector_weight_shape = (len(self.sectors),) + (1,) * (sector_stack.ndim - 1)
        sector_sum = jnp.sum(
            weights[1:].reshape(sector_weight_shape) * sector_stack,
            axis=0,
        )
        central = (optimal - sector_sum) / weights[0]
        return jnp.concatenate((central[None, ...], sector_stack), axis=0)

    def coefficients(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return self._blend(
            value, self.candidate_coefficients(value), self.optimal.smoothness_gram
        )

    def _blend(self, value: Array, candidates: Array, gram: Array, /) -> Array:
        smoothness = ein.contract(
            "kc...i,cij,kc...j->kc...",
            candidates,
            gram.astype(value.dtype),
            candidates,
        )
        tau = jnp.max(smoothness, axis=0) - jnp.min(smoothness, axis=0)
        weights = self.linear_weights.astype(value.dtype)
        weight_shape = (weights.size,) + (1,) * (smoothness.ndim - 1)
        alpha = weights.reshape(weight_shape) * (
            1.0
            + (
                tau[None, ...]
                / (smoothness + jnp.asarray(self.epsilon, dtype=value.dtype))
            )
            ** self.power
        )
        nonlinear_weights = alpha / jnp.sum(alpha, axis=0, keepdims=True)
        return jnp.sum(nonlinear_weights[..., None] * candidates, axis=0)

    def evaluate_cells(self, state: Array, cell_routes: Array, points: Array, /) -> Array:
        """WENO-Z traces of the selected cells at their own points.

        Only the routed cells' candidate polynomials are formed. `points` has
        shape `(routes, points, dimension)`; the `"cell_extrema"` limiter scales
        each route by its own point set, exactly as `reconstruct_at` does for
        face quadrature points.
        """
        value = jnp.asarray(state)
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        candidates = self._candidates(
            value,
            self.optimal.cell_coefficients(value, routes),
            tuple(sector.cell_coefficients(value, routes) for sector in self.sectors),
        )
        coefficients = self._blend(
            value, candidates, self.optimal.smoothness_gram[routes]
        )
        basis = self.optimal.basis_values(routes, jnp.asarray(points, dtype=value.dtype))
        traces = value[routes, None, ...] + ein.contract(
            "r...f,rqf->rq...", coefficients, basis
        )
        return self._limit(value, traces, routes)

    def _limit(self, state: Array, traces: Array, cell_routes: Array, /) -> Array:
        if self.limiter == "none":
            return traces
        value = jnp.asarray(state)
        routes = jnp.asarray(cell_routes, dtype=jnp.int32)
        stencils = self.optimal.stencil_cells[routes]
        valid = self.optimal.stencil_valid[routes]
        gathered = value[stencils]
        mask = valid.reshape(valid.shape + (1,) * (gathered.ndim - 2))
        base = value[routes]
        minimum = jnp.minimum(
            base, jnp.min(jnp.where(mask, gathered, base[:, None, ...]), axis=1)
        )
        maximum = jnp.maximum(
            base, jnp.max(jnp.where(mask, gathered, base[:, None, ...]), axis=1)
        )
        delta = traces - base[:, None, ...]
        upper = (maximum - base)[:, None, ...]
        lower = (minimum - base)[:, None, ...]
        allowed = jnp.where(delta >= 0.0, upper, lower)
        active = jnp.abs(delta) > 0.0
        safe_delta = jnp.where(active, delta, 1.0)
        ratio = jnp.where(active, allowed / safe_delta, 1.0)
        theta = jnp.min(jnp.clip(ratio, 0.0, 1.0), axis=1)
        return base[:, None, ...] + theta[:, None, ...] * delta

    def reconstruct_at(self, state: Array, points: Array, /) -> tuple[Array, Array]:
        value = jnp.asarray(state)
        coefficients = self.coefficients(value)
        owner = self.discretization.owner_cells
        neighbor = self.discretization.neighbor_cells
        safe_neighbor = jnp.maximum(neighbor, 0)
        left = self.optimal.evaluate_coefficients(value, coefficients, owner, points)
        right = self.optimal.evaluate_coefficients(
            value, coefficients, safe_neighbor, points
        )
        return (
            self._limit(value, left, owner),
            self._limit(value, right, safe_neighbor),
        )

    def reconstruct(self, state: Array, /) -> tuple[Array, Array]:
        left, right = self.reconstruct_at(
            state, self.discretization.face_centers[:, None, :]
        )
        return left[:, 0], right[:, 0]


__all__ = [
    "PreparedUnstructuredWENOZReconstruction",
    "UnstructuredWENOLimiter",
    "UnstructuredWENOZReconstructionPlan",
]
