#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Provider-neutral ordered scientific limit and finite-size studies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from ._fingerprint import canonical_fingerprint


LimitStudyStatus: TypeAlias = Literal["complete", "abstained"]
AxisTransform: TypeAlias = Literal["identity", "inverse", "square", "log"]


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result or result != value:
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return result


def _finite(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _canonical_optional_float(value: float, /) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


@dataclass(frozen=True, slots=True)
class ScientificLimitAxis:
    """One declared coordinate approaching a finite or asymptotic target."""

    name: str
    target: float
    transform: AxisTransform
    minimum_span: float
    axis_id: str

    def __init__(
        self,
        name: str,
        target: float,
        /,
        *,
        transform: AxisTransform = "identity",
        minimum_span: float = 0.0,
    ):
        name_ = _identifier(name, "axis name")
        target_ = _finite(target, "axis target")
        if transform not in ("identity", "inverse", "square", "log"):
            raise ValueError("Unknown limit-axis transform.")
        span = _finite(minimum_span, "minimum_span")
        if span < 0.0:
            raise ValueError("minimum_span must be non-negative.")
        content = {
            "kind": "scientific-limit-axis",
            "name": name_,
            "target": target_,
            "transform": transform,
            "minimum_span": span,
        }
        object.__setattr__(self, "name", name_)
        object.__setattr__(self, "target", target_)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "minimum_span", span)
        object.__setattr__(self, "axis_id", canonical_fingerprint(content))

    def coordinate(self, value: float, /) -> float:
        raw = _finite(value, f"{self.name} coordinate")
        if self.transform == "identity":
            transformed = raw
            target = self.target
        elif self.transform == "inverse":
            if raw == 0.0 or self.target == 0.0:
                raise ValueError("Inverse limit coordinates require nonzero values.")
            transformed = 1.0 / raw
            target = 1.0 / self.target
        elif self.transform == "square":
            transformed = raw * raw
            target = self.target * self.target
        else:
            if raw <= 0.0 or self.target <= 0.0:
                raise ValueError("Log limit coordinates require positive values.")
            transformed = math.log(raw)
            target = math.log(self.target)
        return transformed - target


@dataclass(frozen=True, slots=True)
class ScientificLimitDatum:
    """One scalar observation with all declared limit coordinates."""

    datum_id: str
    coordinates: tuple[tuple[str, float], ...]
    value: float
    standard_error: float
    ancestry_ids: tuple[str, ...]

    def __init__(
        self,
        datum_id: str,
        coordinates: Mapping[str, float],
        value: float,
        standard_error: float,
        /,
        *,
        ancestry_ids: Sequence[str] = (),
    ):
        datum = _identifier(datum_id, "datum_id")
        if not isinstance(coordinates, Mapping) or not coordinates:
            raise TypeError("coordinates must be a non-empty mapping.")
        resolved = tuple(
            sorted(
                (
                    _identifier(name, "coordinate name"),
                    _finite(axis_value, f"coordinate {name}"),
                )
                for name, axis_value in coordinates.items()
            )
        )
        if len({name for name, _ in resolved}) != len(resolved):
            raise ValueError("Coordinate names must be unique.")
        value_ = _finite(value, "datum value")
        error = _finite(standard_error, "standard_error")
        if error <= 0.0:
            raise ValueError("standard_error must be positive.")
        ancestry = tuple(
            sorted(_identifier(item, "ancestry ID") for item in ancestry_ids)
        )
        if len(set(ancestry)) != len(ancestry):
            raise ValueError("Ancestry IDs must be unique.")
        object.__setattr__(self, "datum_id", datum)
        object.__setattr__(self, "coordinates", resolved)
        object.__setattr__(self, "value", value_)
        object.__setattr__(self, "standard_error", error)
        object.__setattr__(self, "ancestry_ids", ancestry)

    def coordinate_map(self) -> dict[str, float]:
        return dict(self.coordinates)


@dataclass(frozen=True, slots=True)
class ScientificLimitVariation:
    """One prespecified additive polynomial limit ansatz."""

    variation_id: str
    axis_orders: tuple[tuple[str, int], ...]
    included_datum_ids: tuple[str, ...]
    minimum_points: int

    def __init__(
        self,
        variation_id: str,
        axis_orders: Mapping[str, int],
        /,
        *,
        included_datum_ids: Sequence[str] = (),
        minimum_points: int = 0,
    ):
        identifier = _identifier(variation_id, "variation_id")
        if not isinstance(axis_orders, Mapping) or not axis_orders:
            raise TypeError("axis_orders must be a non-empty mapping.")
        orders = tuple(
            sorted(
                (_identifier(name, "axis name"), int(order))
                for name, order in axis_orders.items()
            )
        )
        if any(order < 0 for _, order in orders):
            raise ValueError("Limit-study polynomial orders must be non-negative.")
        included = tuple(
            sorted(_identifier(item, "included datum ID") for item in included_datum_ids)
        )
        if len(set(included)) != len(included):
            raise ValueError("Included datum IDs must be unique.")
        minimum = int(minimum_points)
        if minimum < 0:
            raise ValueError("minimum_points must be non-negative.")
        object.__setattr__(self, "variation_id", identifier)
        object.__setattr__(self, "axis_orders", orders)
        object.__setattr__(self, "included_datum_ids", included)
        object.__setattr__(self, "minimum_points", minimum)


@dataclass(frozen=True, slots=True)
class ScientificLimitStudyPlan:
    """Ordered limit axes and prespecified systematic variations."""

    axes: tuple[ScientificLimitAxis, ...]
    variations: tuple[ScientificLimitVariation, ...]
    maximum_condition_number: float
    study_id: str

    def __init__(
        self,
        axes: Sequence[ScientificLimitAxis],
        variations: Sequence[ScientificLimitVariation],
        /,
        *,
        maximum_condition_number: float = 1e12,
    ):
        axes_ = tuple(axes)
        variations_ = tuple(variations)
        if not axes_ or any(not isinstance(item, ScientificLimitAxis) for item in axes_):
            raise TypeError("axes must contain ScientificLimitAxis values.")
        if not variations_ or any(
            not isinstance(item, ScientificLimitVariation) for item in variations_
        ):
            raise TypeError("variations must contain ScientificLimitVariation values.")
        if len({item.name for item in axes_}) != len(axes_):
            raise ValueError("Limit-study axis names must be unique.")
        if len({item.variation_id for item in variations_}) != len(variations_):
            raise ValueError("Limit-study variation IDs must be unique.")
        names = {item.name for item in axes_}
        if any(set(dict(item.axis_orders)) != names for item in variations_):
            raise ValueError("Every variation must declare every study axis.")
        condition = _finite(maximum_condition_number, "maximum_condition_number")
        if condition <= 1.0:
            raise ValueError("maximum_condition_number must exceed one.")
        content = {
            "kind": "scientific-limit-study-plan",
            "axes": [item.axis_id for item in axes_],
            "variations": [
                {
                    "id": item.variation_id,
                    "orders": dict(item.axis_orders),
                    "included": list(item.included_datum_ids),
                    "minimum_points": item.minimum_points,
                }
                for item in variations_
            ],
            "maximum_condition_number": condition,
        }
        object.__setattr__(self, "axes", axes_)
        object.__setattr__(self, "variations", variations_)
        object.__setattr__(self, "maximum_condition_number", condition)
        object.__setattr__(self, "study_id", canonical_fingerprint(content))


@dataclass(frozen=True, slots=True)
class ScientificLimitFit:
    variation_id: str
    status: LimitStudyStatus
    estimate: float
    standard_error: float
    chi_square: float
    degrees_of_freedom: int
    condition_number: float
    datum_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class ScientificLimitStudyResult:
    study_id: str
    status: LimitStudyStatus
    estimate: float
    statistical_error: float
    systematic_error: float
    fits: tuple[ScientificLimitFit, ...]
    result_id: str


def _design(
    plan: ScientificLimitStudyPlan,
    variation: ScientificLimitVariation,
    data: Sequence[ScientificLimitDatum],
) -> np.ndarray:
    orders = dict(variation.axis_orders)
    columns = [np.ones((len(data),), dtype=np.float64)]
    for axis in plan.axes:
        coordinate = np.asarray(
            [axis.coordinate(item.coordinate_map()[axis.name]) for item in data],
            dtype=np.float64,
        )
        for power in range(1, orders[axis.name] + 1):
            columns.append(coordinate**power)
    return np.stack(columns, axis=1)


def run_scientific_limit_study(
    plan: ScientificLimitStudyPlan,
    data: Sequence[ScientificLimitDatum],
    /,
) -> ScientificLimitStudyResult:
    """Run every prespecified weighted fit and abstain on inadequate support."""

    if not isinstance(plan, ScientificLimitStudyPlan):
        raise TypeError("plan must be ScientificLimitStudyPlan.")
    values = tuple(data)
    if not values or any(not isinstance(item, ScientificLimitDatum) for item in values):
        raise TypeError("data must contain ScientificLimitDatum values.")
    if len({item.datum_id for item in values}) != len(values):
        raise ValueError("Scientific limit datum IDs must be unique.")
    axis_names = {item.name for item in plan.axes}
    if any(set(item.coordinate_map()) != axis_names for item in values):
        raise ValueError("Every datum must supply every declared study axis.")
    by_id = {item.datum_id: item for item in values}
    fits: list[ScientificLimitFit] = []
    for variation in plan.variations:
        unknown = tuple(
            datum_id for datum_id in variation.included_datum_ids if datum_id not in by_id
        )
        if unknown:
            raise ValueError(
                f"Variation {variation.variation_id!r} references unknown datum IDs "
                f"{unknown!r}."
            )
    for variation in plan.variations:
        selected = (
            values
            if not variation.included_datum_ids
            else tuple(by_id[item] for item in variation.included_datum_ids)
        )
        design = _design(plan, variation, selected) if selected else np.empty((0, 0))
        required = max(variation.minimum_points, design.shape[1] + 1)
        reason = ""
        status: LimitStudyStatus = "complete"
        estimate = math.nan
        standard_error = math.nan
        chi_square = math.nan
        degrees = 0
        condition = math.inf
        if len(selected) < required:
            status = "abstained"
            reason = "insufficient-points"
        else:
            spans = {
                axis.name: max(
                    axis.coordinate(item.coordinate_map()[axis.name]) for item in selected
                )
                - min(
                    axis.coordinate(item.coordinate_map()[axis.name]) for item in selected
                )
                for axis in plan.axes
            }
            if any(spans[axis.name] < axis.minimum_span for axis in plan.axes):
                status = "abstained"
                reason = "insufficient-axis-span"
            else:
                observations = np.asarray([item.value for item in selected])
                errors = np.asarray([item.standard_error for item in selected])
                weighted = design / errors[:, None]
                target = observations / errors
                try:
                    condition = float(np.linalg.cond(weighted))
                except np.linalg.LinAlgError:
                    condition = math.inf
                if (
                    not math.isfinite(condition)
                    or condition > plan.maximum_condition_number
                ):
                    status = "abstained"
                    reason = "ill-conditioned-design"
                else:
                    try:
                        coefficients, _, rank, _ = np.linalg.lstsq(
                            weighted, target, rcond=None
                        )
                    except np.linalg.LinAlgError:
                        status = "abstained"
                        reason = "linear-solve-failed"
                    else:
                        if rank != design.shape[1]:
                            status = "abstained"
                            reason = "rank-deficient-design"
                        else:
                            intercept = np.zeros((design.shape[1],), dtype=np.float64)
                            intercept[0] = 1.0
                            try:
                                influence, _, covariance_rank, _ = np.linalg.lstsq(
                                    weighted.T,
                                    intercept,
                                    rcond=None,
                                )
                            except np.linalg.LinAlgError:
                                status = "abstained"
                                reason = "covariance-solve-failed"
                            else:
                                variance = float(influence @ influence)
                                if (
                                    covariance_rank != design.shape[1]
                                    or not math.isfinite(variance)
                                    or variance < 0.0
                                ):
                                    status = "abstained"
                                    reason = "covariance-solve-failed"
                                else:
                                    residual = (
                                        design @ coefficients - observations
                                    ) / errors
                                    estimate = float(coefficients[0])
                                    standard_error = float(math.sqrt(variance))
                                    chi_square = float(residual @ residual)
                                    degrees = len(selected) - design.shape[1]
        fits.append(
            ScientificLimitFit(
                variation.variation_id,
                status,
                estimate,
                standard_error,
                chi_square,
                degrees,
                condition,
                tuple(item.datum_id for item in selected),
                reason,
            )
        )
    complete = tuple(item for item in fits if item.status == "complete")
    if not complete:
        status = "abstained"
        estimate = math.nan
        statistical = math.nan
        systematic = math.nan
    else:
        status = "complete"
        estimates = np.asarray([item.estimate for item in complete])
        estimate = float(np.mean(estimates))
        statistical = float(max(item.standard_error for item in complete))
        systematic = float(np.max(np.abs(estimates - estimate)))
    content = {
        "kind": "scientific-limit-study-result",
        "study_id": plan.study_id,
        "status": status,
        "estimate": _canonical_optional_float(estimate),
        "statistical_error": _canonical_optional_float(statistical),
        "systematic_error": _canonical_optional_float(systematic),
        "fits": [
            {
                "variation_id": item.variation_id,
                "status": item.status,
                "estimate": _canonical_optional_float(item.estimate),
                "standard_error": _canonical_optional_float(item.standard_error),
                "chi_square": _canonical_optional_float(item.chi_square),
                "degrees_of_freedom": item.degrees_of_freedom,
                "condition_number": _canonical_optional_float(item.condition_number),
                "datum_ids": list(item.datum_ids),
                "reason": item.reason,
            }
            for item in fits
        ],
    }
    return ScientificLimitStudyResult(
        plan.study_id,
        status,
        estimate,
        statistical,
        systematic,
        tuple(fits),
        canonical_fingerprint(content),
    )


__all__ = [
    "AxisTransform",
    "LimitStudyStatus",
    "ScientificLimitAxis",
    "ScientificLimitDatum",
    "ScientificLimitFit",
    "ScientificLimitStudyPlan",
    "ScientificLimitStudyResult",
    "ScientificLimitVariation",
    "run_scientific_limit_study",
]
