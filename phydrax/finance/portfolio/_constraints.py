#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


def _finite_array(value: ArrayLike, name: str, /, *, ndim: int) -> Array:
    result = jnp.asarray(value)
    if result.ndim != ndim or 0 in result.shape:
        kind = "vector" if ndim == 1 else "matrix"
        raise ValueError(f"{name} must be a non-empty {kind}.")
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    result = result.astype(jnp.result_type(result.dtype, jnp.float32))
    if not bool(np.all(np.isfinite(np.asarray(result)))):
        raise ValueError(f"{name} must be finite.")
    return result


def _weight_bound(value: ArrayLike, name: str, /, *, lower: bool) -> Array:
    result = jnp.asarray(value)
    if result.ndim != 1 or result.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty vector.")
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    result = result.astype(jnp.result_type(result.dtype, jnp.float32))
    host = np.asarray(result)
    invalid_infinity = np.isposinf(host) if lower else np.isneginf(host)
    if np.any(np.isnan(host)) or np.any(invalid_infinity):
        direction = "positive" if lower else "negative"
        raise ValueError(f"{name} cannot contain NaN or {direction} infinity.")
    return result


class RobustSOCConstraint(StrictModule):
    """Ellipsoidal robust bound ``nominal·w + radius ||Lᵀw||₂ <= bound``."""

    nominal: Array
    factor_loading: Array
    radius: float = eqx.field(static=True)
    bound: float = eqx.field(static=True)
    constraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        nominal: ArrayLike,
        factor_loading: ArrayLike,
        /,
        *,
        radius: float,
        bound: float,
        constraint_id: str = "robust-soc",
    ):
        nominal_ = _finite_array(nominal, "nominal", ndim=1)
        loading = _finite_array(factor_loading, "factor_loading", ndim=2).astype(
            nominal_.dtype
        )
        if loading.shape[0] != nominal_.shape[0]:
            raise ValueError(
                "factor_loading must have one row per nominal exposure coordinate."
            )
        radius_ = float(radius)
        bound_ = float(bound)
        identifier = str(constraint_id)
        if not isfinite(radius_) or radius_ < 0.0:
            raise ValueError("radius must be finite and non-negative.")
        if not isfinite(bound_):
            raise ValueError("bound must be finite.")
        if not identifier:
            raise ValueError("constraint_id must be non-empty.")
        self.nominal = nominal_
        self.factor_loading = loading
        self.radius = radius_
        self.bound = bound_
        self.constraint_id = identifier


class ScenarioTree(StrictModule):
    """Finite scenario histories used to compile deterministic nonanticipativity."""

    history_labels: Array
    scenario_count: int = eqx.field(static=True)
    stage_count: int = eqx.field(static=True)
    tree_id: str = eqx.field(static=True)

    def __init__(self, history_labels: ArrayLike, /, *, tree_id: str = "scenario-tree"):
        labels = np.asarray(history_labels)
        if labels.ndim != 2 or 0 in labels.shape:
            raise ValueError(
                "history_labels must be a non-empty (scenario, stage) matrix."
            )
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("history_labels must contain integers.")
        if np.any(labels < 0):
            raise ValueError("history_labels must be non-negative.")
        scenarios, stages = map(int, labels.shape)
        if np.unique(labels[:, 0]).size != 1:
            raise ValueError("Every scenario must share the root history at stage zero.")
        for stage in range(1, stages):
            previous = labels[:, stage - 1]
            current = labels[:, stage]
            for history in np.unique(current):
                members = previous[current == history]
                if np.unique(members).size != 1:
                    raise ValueError(
                        "Scenario histories cannot reconverge after becoming distinct."
                    )
        identifier = str(tree_id)
        if not identifier:
            raise ValueError("tree_id must be non-empty.")
        self.history_labels = jnp.asarray(labels, dtype=jnp.int32)
        self.scenario_count = scenarios
        self.stage_count = stages
        self.tree_id = identifier

    def nonanticipativity_matrix(self, asset_count: int, /) -> Array:
        """Return deterministic equality rows for equal observed histories."""

        if isinstance(asset_count, bool) or int(asset_count) <= 0:
            raise ValueError("asset_count must be a positive integer.")
        assets = int(asset_count)
        labels = np.asarray(self.history_labels)
        variables = self.scenario_count * self.stage_count * assets
        rows: list[np.ndarray] = []
        for stage in range(self.stage_count):
            for history in np.unique(labels[:, stage]):
                members = np.flatnonzero(labels[:, stage] == history)
                representative = int(members[0])
                for scenario in members[1:]:
                    for asset in range(assets):
                        row = np.zeros((variables,), dtype=np.float64)
                        left = (int(scenario) * self.stage_count + stage) * assets + asset
                        right = (
                            representative * self.stage_count + stage
                        ) * assets + asset
                        row[left], row[right] = 1.0, -1.0
                        rows.append(row)
        return jnp.asarray(
            np.stack(rows) if rows else np.empty((0, variables), dtype=np.float64)
        )


class PortfolioConstraints(StrictModule):
    """Validated portfolio feasibility contract, independent of an objective law."""

    lower_weights: Array | None
    upper_weights: Array | None
    budget: float = eqx.field(static=True)
    linear_matrix: Array | None
    linear_lower: Array | None
    linear_upper: Array | None
    turnover_limit: float | None = eqx.field(static=True)
    gross_limit: float | None = eqx.field(static=True)
    lot_sizes: Array | None
    maximum_cardinality: int | None = eqx.field(static=True)
    fixed_fees: Array | None
    robust: tuple[RobustSOCConstraint, ...]
    scenario_tree: ScenarioTree | None

    def __init__(
        self,
        *,
        lower_weights: ArrayLike | None = None,
        upper_weights: ArrayLike | None = None,
        budget: float = 1.0,
        linear_matrix: ArrayLike | None = None,
        linear_lower: ArrayLike | None = None,
        linear_upper: ArrayLike | None = None,
        turnover_limit: float | None = None,
        gross_limit: float | None = None,
        lot_sizes: ArrayLike | None = None,
        maximum_cardinality: int | None = None,
        fixed_fees: ArrayLike | None = None,
        robust: tuple[RobustSOCConstraint, ...] = (),
        scenario_tree: ScenarioTree | None = None,
    ):
        lower = (
            None
            if lower_weights is None
            else _weight_bound(lower_weights, "lower_weights", lower=True)
        )
        upper = (
            None
            if upper_weights is None
            else _weight_bound(upper_weights, "upper_weights", lower=False)
        )
        if lower is not None and upper is not None:
            if lower.shape != upper.shape:
                raise ValueError("Weight bounds must have identical shapes.")
            if bool(np.any(np.asarray(lower) > np.asarray(upper))):
                raise ValueError("lower_weights must not exceed upper_weights.")
        budget_ = float(budget)
        if not isfinite(budget_):
            raise ValueError("budget must be finite.")
        supplied_linear = (linear_matrix, linear_lower, linear_upper)
        if any(value is not None for value in supplied_linear) and not all(
            value is not None for value in supplied_linear
        ):
            raise ValueError(
                "linear_matrix, linear_lower, and linear_upper must be supplied together."
            )
        matrix = (
            None
            if linear_matrix is None
            else _finite_array(linear_matrix, "linear_matrix", ndim=2)
        )
        linear_lo = (
            None
            if linear_lower is None
            else jnp.asarray(linear_lower, dtype=matrix.dtype)
        )
        linear_hi = (
            None
            if linear_upper is None
            else jnp.asarray(linear_upper, dtype=matrix.dtype)
        )
        if matrix is not None:
            rows = int(matrix.shape[0])
            if linear_lo.shape != (rows,) or linear_hi.shape != (rows,):
                raise ValueError(f"Linear bounds must both have shape ({rows},).")
            lo = np.asarray(linear_lo)
            hi = np.asarray(linear_hi)
            if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi):
                raise ValueError("Linear bounds must be ordered and cannot contain NaN.")
        turnover = None if turnover_limit is None else float(turnover_limit)
        gross = None if gross_limit is None else float(gross_limit)
        if turnover is not None and (not isfinite(turnover) or turnover < 0.0):
            raise ValueError("turnover_limit must be finite and non-negative.")
        if gross is not None and (not isfinite(gross) or gross <= 0.0):
            raise ValueError("gross_limit must be finite and positive.")
        lots = (
            None if lot_sizes is None else _finite_array(lot_sizes, "lot_sizes", ndim=1)
        )
        if lots is not None and bool(np.any(np.asarray(lots) <= 0.0)):
            raise ValueError("lot_sizes must be strictly positive.")
        cardinality = None if maximum_cardinality is None else int(maximum_cardinality)
        if isinstance(maximum_cardinality, bool) or (
            cardinality is not None and cardinality <= 0
        ):
            raise ValueError("maximum_cardinality must be a positive integer or None.")
        fees = (
            None
            if fixed_fees is None
            else _finite_array(fixed_fees, "fixed_fees", ndim=1)
        )
        if fees is not None and bool(np.any(np.asarray(fees) < 0.0)):
            raise ValueError("fixed_fees must be non-negative.")
        robust_ = tuple(robust)
        if any(not isinstance(item, RobustSOCConstraint) for item in robust_):
            raise TypeError("robust entries must be RobustSOCConstraint values.")
        if scenario_tree is not None and not isinstance(scenario_tree, ScenarioTree):
            raise TypeError("scenario_tree must be a ScenarioTree or None.")
        self.lower_weights, self.upper_weights = lower, upper
        self.budget = budget_
        self.linear_matrix, self.linear_lower, self.linear_upper = (
            matrix,
            linear_lo,
            linear_hi,
        )
        self.turnover_limit, self.gross_limit = turnover, gross
        self.lot_sizes = lots
        self.maximum_cardinality = cardinality
        self.fixed_fees = fees
        self.robust = robust_
        self.scenario_tree = scenario_tree


__all__ = ["PortfolioConstraints", "RobustSOCConstraint", "ScenarioTree"]
