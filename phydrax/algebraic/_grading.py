#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal, Sequence, TYPE_CHECKING, TypeAlias

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


if TYPE_CHECKING:
    from ._system import SparsePolynomialSupport


PolynomialVariableGeometry: TypeAlias = Literal["affine", "projective"]
BezoutForecastKind: TypeAlias = Literal["total_degree", "multihomogeneous"]


class PolynomialVariableGroup(StrictModule, NonTrainableState):
    """A named variable block with explicit affine or projective geometry."""

    label: str = eqx.field(static=True)
    variable_indices: tuple[int, ...] = eqx.field(static=True)
    geometry: PolynomialVariableGeometry = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    def __init__(
        self,
        label: str,
        variable_indices: Sequence[int],
        /,
        *,
        geometry: PolynomialVariableGeometry = "affine",
    ):
        label_ = str(label).strip()
        if not label_:
            raise ValueError("Polynomial variable-group labels must be non-empty.")
        indices = tuple(int(index) for index in variable_indices)
        if not indices:
            raise ValueError("A polynomial variable group must contain variables.")
        if any(index < 0 for index in indices):
            raise ValueError("Polynomial variable indices must be non-negative.")
        if len(set(indices)) != len(indices):
            raise ValueError("A polynomial variable group cannot repeat a variable.")
        if tuple(sorted(indices)) != indices:
            raise ValueError("Polynomial variable indices must be strictly increasing.")
        if geometry not in ("affine", "projective"):
            raise ValueError("Variable-group geometry must be 'affine' or 'projective'.")
        if geometry == "projective" and len(indices) < 2:
            raise ValueError(
                "A projective variable group requires at least two homogeneous "
                "coordinates."
            )
        dimension = len(indices) if geometry == "affine" else len(indices) - 1
        self.label = label_
        self.variable_indices = indices
        self.geometry = geometry
        self.dimension = dimension
        self.group_id = canonical_fingerprint(
            {
                "kind": "polynomial-variable-group-v1",
                "label": label_,
                "variable_indices": list(indices),
                "geometry": geometry,
            }
        )


class PolynomialDegreeProfile(StrictModule, NonTrainableState):
    """Exact support degrees, independent of the current coefficient values."""

    support_id: str = eqx.field(static=True)
    total_degrees: tuple[int, ...] = eqx.field(static=True)
    multidegrees: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    group_labels: tuple[str, ...] = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        support_id: str,
        total_degrees: Sequence[int],
        multidegrees: Sequence[Sequence[int]],
        group_labels: Sequence[str],
    ):
        support_id_ = str(support_id)
        totals = tuple(int(value) for value in total_degrees)
        multidegrees_ = tuple(tuple(int(value) for value in row) for row in multidegrees)
        labels = tuple(str(value) for value in group_labels)
        if not support_id_:
            raise ValueError("Degree evidence requires a support identity.")
        if any(value < 0 for value in totals):
            raise ValueError("Polynomial degrees must be non-negative.")
        if len(multidegrees_) != len(totals) or any(
            len(row) != len(labels) or any(value < 0 for value in row)
            for row in multidegrees_
        ):
            raise ValueError("Multidegrees must have one non-negative row per equation.")
        self.support_id = support_id_
        self.total_degrees = totals
        self.multidegrees = multidegrees_
        self.group_labels = labels
        self.profile_id = canonical_fingerprint(
            {
                "kind": "polynomial-degree-profile-v1",
                "support": support_id_,
                "total_degrees": list(totals),
                "group_labels": list(labels),
                "multidegrees": [list(row) for row in multidegrees_],
            }
        )


class PolynomialBezoutForecast(StrictModule, NonTrainableState):
    """An exact host-integer path-count forecast under stated degree geometry."""

    kind: BezoutForecastKind = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    degree_profile_id: str = eqx.field(static=True)
    equation_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    group_labels: tuple[str, ...] = eqx.field(static=True)
    group_dimensions: tuple[int, ...] = eqx.field(static=True)
    path_count: int = eqx.field(static=True)
    status: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: BezoutForecastKind,
        /,
        *,
        support_id: str,
        degree_profile_id: str,
        equation_count: int,
        ambient_dimension: int,
        group_labels: Sequence[str],
        group_dimensions: Sequence[int],
        path_count: int,
    ):
        if kind not in ("total_degree", "multihomogeneous"):
            raise ValueError("Unknown polynomial Bézout forecast kind.")
        labels = tuple(str(value) for value in group_labels)
        dimensions = tuple(int(value) for value in group_dimensions)
        count = int(path_count)
        if count < 0 or any(value < 0 for value in dimensions):
            raise ValueError(
                "Bézout forecast counts and dimensions must be non-negative."
            )
        if len(labels) != len(dimensions):
            raise ValueError("Bézout group labels and dimensions must align.")
        self.kind = kind
        self.support_id = str(support_id)
        self.degree_profile_id = str(degree_profile_id)
        self.equation_count = int(equation_count)
        self.ambient_dimension = int(ambient_dimension)
        self.group_labels = labels
        self.group_dimensions = dimensions
        self.path_count = count
        self.status = "applicable"
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "polynomial-bezout-forecast-v1",
                "forecast_kind": kind,
                "support": self.support_id,
                "degree_profile": self.degree_profile_id,
                "equation_count": self.equation_count,
                "ambient_dimension": self.ambient_dimension,
                "group_labels": list(labels),
                "group_dimensions": list(dimensions),
                "path_count": count,
                "status": self.status,
            }
        )

    @property
    def bound(self) -> int:
        """Return the forecast path count as an arbitrary-precision host integer."""
        return self.path_count


def polynomial_degree_profile(
    support: SparsePolynomialSupport,
    /,
) -> PolynomialDegreeProfile:
    """Compute exact declared-support total degrees and block multidegrees."""
    equations = np.asarray(support.equation_indices)
    exponents = np.asarray(support.exponents)
    totals: list[int] = []
    multidegrees: list[tuple[int, ...]] = []
    for equation in range(support.equation_count):
        rows = exponents[equations == equation]
        if rows.shape[0] == 0:
            raise ValueError(
                "Every equation must have at least one declared support term."
            )
        totals.append(max(sum(int(value) for value in row) for row in rows))
        multidegrees.append(
            tuple(
                max(
                    sum(int(row[index]) for index in group.variable_indices)
                    for row in rows
                )
                for group in support.groups
            )
        )
    return PolynomialDegreeProfile(
        support_id=support.support_id,
        total_degrees=totals,
        multidegrees=multidegrees,
        group_labels=tuple(group.label for group in support.groups),
    )


def total_degrees(support: SparsePolynomialSupport, /) -> tuple[int, ...]:
    """Return each equation's exact total degree in the declared fixed support."""
    return polynomial_degree_profile(support).total_degrees


def multihomogeneous_degrees(
    support: SparsePolynomialSupport,
    /,
) -> tuple[tuple[int, ...], ...]:
    """Return equation-by-group maximum degrees for the declared fixed support."""
    return polynomial_degree_profile(support).multidegrees


def total_degree_bezout_forecast(
    support: SparsePolynomialSupport,
    /,
) -> PolynomialBezoutForecast:
    """Forecast total-degree paths for a square affine polynomial system."""
    if support.equation_count != support.variable_count:
        raise ValueError("A total-degree Bézout forecast requires a square system.")
    if any(group.geometry != "affine" for group in support.groups):
        raise ValueError(
            "Total-degree forecasting uses affine variables; use the grouped forecast "
            "for projective geometry."
        )
    profile = polynomial_degree_profile(support)
    return PolynomialBezoutForecast(
        "total_degree",
        support_id=support.support_id,
        degree_profile_id=profile.profile_id,
        equation_count=support.equation_count,
        ambient_dimension=support.variable_count,
        group_labels=(),
        group_dimensions=(),
        path_count=prod(profile.total_degrees),
    )


def multihomogeneous_bezout_forecast(
    support: SparsePolynomialSupport,
    /,
) -> PolynomialBezoutForecast:
    """Forecast grouped paths via an exact coefficient extraction over host integers."""
    if not support.groups:
        raise ValueError("A multihomogeneous forecast requires variable groups.")
    covered = tuple(
        sorted(index for group in support.groups for index in group.variable_indices)
    )
    if covered != tuple(range(support.variable_count)):
        raise ValueError(
            "A multihomogeneous forecast requires groups that cover every variable."
        )
    dimensions = tuple(group.dimension for group in support.groups)
    ambient_dimension = sum(dimensions)
    if support.equation_count != ambient_dimension:
        raise ValueError(
            "A multihomogeneous Bézout forecast requires one equation per ambient "
            "dimension."
        )
    profile = polynomial_degree_profile(support)
    target = dimensions
    coefficients: dict[tuple[int, ...], int] = {(0,) * len(target): 1}
    for degrees in profile.multidegrees:
        next_coefficients: dict[tuple[int, ...], int] = {}
        for exponent, coefficient in coefficients.items():
            for group_index, degree in enumerate(degrees):
                advanced = list(exponent)
                advanced[group_index] += 1
                advanced_tuple = tuple(advanced)
                if advanced_tuple[group_index] > target[group_index]:
                    continue
                next_coefficients[advanced_tuple] = (
                    next_coefficients.get(advanced_tuple, 0) + coefficient * degree
                )
        coefficients = next_coefficients
    return PolynomialBezoutForecast(
        "multihomogeneous",
        support_id=support.support_id,
        degree_profile_id=profile.profile_id,
        equation_count=support.equation_count,
        ambient_dimension=ambient_dimension,
        group_labels=profile.group_labels,
        group_dimensions=dimensions,
        path_count=coefficients.get(target, 0),
    )


__all__ = [
    "BezoutForecastKind",
    "multihomogeneous_bezout_forecast",
    "multihomogeneous_degrees",
    "polynomial_degree_profile",
    "PolynomialBezoutForecast",
    "PolynomialDegreeProfile",
    "PolynomialVariableGeometry",
    "PolynomialVariableGroup",
    "total_degree_bezout_forecast",
    "total_degrees",
]
