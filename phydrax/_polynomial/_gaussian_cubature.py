#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
import math
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
from scipy.linalg import hadamard

from ._cubature import CubatureRuleData
from ._orthogonal import standard_normal_hermite_rule_data


GaussianCubatureFamily: TypeAlias = Literal[
    "auto",
    "stroud-secrest-3-1",
    "hadamard-3",
    "stroud-secrest-5-2",
    "stroud-secrest-5-3",
    "tensor-hermite",
]
_DEFAULT_MAXIMUM_POINTS = 65_536
_DEFAULT_MAXIMUM_BYTES = 64 * 1024**2


def _dimension(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Gaussian cubature dimension must be an integer.")
    dimension = int(value)
    if dimension < 1:
        raise ValueError("Gaussian cubature dimension must be positive.")
    return dimension


def _degree(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("Gaussian cubature degree must be an integer.")
    degree = int(value)
    if degree < 0:
        raise ValueError("Gaussian cubature degree must be nonnegative.")
    if degree > 5:
        raise ValueError(
            "Positive built-in Gaussian cubature supports degree at most five."
        )
    return degree


def _maximum_points(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("maximum_points must be an integer.")
    maximum = int(value)
    if maximum < 1:
        raise ValueError("maximum_points must be positive.")
    return maximum


def _guard_count(count: int, maximum: int, /) -> None:
    if count > maximum:
        raise ValueError(
            f"Gaussian cubature requires {count} points, exceeding maximum_points={maximum}."
        )


def _axis_points(dimension: int, radius: float, /) -> np.ndarray:
    axes = radius * np.eye(dimension)
    return np.concatenate((axes, -axes), axis=0)


def _two_axis_points(dimension: int, radius: float, /) -> np.ndarray:
    points = []
    for first, second in itertools.combinations(range(dimension), 2):
        for first_sign, second_sign in itertools.product((-1.0, 1.0), repeat=2):
            point = np.zeros((dimension,), dtype=float)
            point[first] = first_sign * radius
            point[second] = second_sign * radius
            points.append(point)
    return np.asarray(points, dtype=float).reshape((-1, dimension))


def _reflected_diagonal(dimension: int, radius: float, /) -> np.ndarray:
    signs = np.asarray(tuple(itertools.product((-1.0, 1.0), repeat=dimension)))
    return radius * signs


def _stroud_secrest_3_1(
    dimension: int, maximum_points: int, maximum_rule_bytes: int
) -> CubatureRuleData:
    count = 2 * dimension
    _guard_count(count, maximum_points)
    points = _axis_points(dimension, math.sqrt(float(dimension)))
    weights = np.full((count,), 1.0 / count)
    return CubatureRuleData(
        points,
        weights,
        exact_degree=3,
        family="stroud-secrest-3-1",
        reference_domain="standard-normal",
        backend="analytic",
        source_id=f"stroud-secrest-1966:3-1:{dimension}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def _hadamard_3(
    dimension: int, maximum_points: int, maximum_rule_bytes: int
) -> CubatureRuleData:
    half_count = 2 ** math.ceil(math.log2(dimension))
    count = 2 * half_count
    _guard_count(count, maximum_points)
    directions = np.asarray(hadamard(half_count)[:, :dimension], dtype=float)
    points = np.concatenate((directions, -directions), axis=0)
    weights = np.full((count,), 1.0 / count)
    return CubatureRuleData(
        points,
        weights,
        exact_degree=3,
        family="hadamard-3",
        reference_domain="standard-normal",
        backend="scipy-analytic",
        source_id=f"victoir-2004:hadamard:{dimension}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def _tensor_hermite(
    dimension: int, maximum_points: int, maximum_rule_bytes: int
) -> CubatureRuleData:
    if dimension != 1:
        raise ValueError(
            "tensor-hermite is built in only for one-dimensional degree five."
        )
    _guard_count(3, maximum_points)
    rule = standard_normal_hermite_rule_data(3)
    return CubatureRuleData(
        np.asarray(rule.nodes)[:, None],
        np.asarray(rule.weights),
        exact_degree=5,
        family="tensor-hermite",
        reference_domain="standard-normal",
        backend="numpy",
        source_id=f"{rule.rule_id}:tensor-dimension-1",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def _stroud_secrest_5_2(
    dimension: int, maximum_points: int, maximum_rule_bytes: int
) -> CubatureRuleData:
    if dimension != 2:
        raise ValueError("stroud-secrest-5-2 is retained only in dimension two.")
    count = 2 * dimension**2 + 1
    _guard_count(count, maximum_points)
    radius_axis = math.sqrt(dimension + 2.0)
    radius_pair = math.sqrt((dimension + 2.0) / 2.0)
    origin = np.zeros((1, dimension))
    axes = _axis_points(dimension, radius_axis)
    pairs = _two_axis_points(dimension, radius_pair)
    points = np.concatenate((origin, axes, pairs), axis=0)
    weight_origin = 2.0 / (dimension + 2.0)
    weight_axis = (4.0 - dimension) / (2.0 * (dimension + 2.0) ** 2)
    weight_pair = 1.0 / (dimension + 2.0) ** 2
    weights = np.concatenate(
        (
            np.asarray([weight_origin]),
            np.full((axes.shape[0],), weight_axis),
            np.full((pairs.shape[0],), weight_pair),
        )
    )
    return CubatureRuleData(
        points,
        weights,
        exact_degree=5,
        family="stroud-secrest-5-2",
        reference_domain="standard-normal",
        backend="analytic",
        source_id=f"stroud-secrest-1966:5-2:{dimension}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def _stroud_secrest_5_3(
    dimension: int, maximum_points: int, maximum_rule_bytes: int
) -> CubatureRuleData:
    if dimension <= 2:
        raise ValueError("stroud-secrest-5-3 requires dimension greater than two.")
    reflected_count = 2**dimension
    count = 2 * dimension + reflected_count
    _guard_count(count, maximum_points)
    radius_axis = math.sqrt((dimension + 2.0) / 2.0)
    radius_diagonal = math.sqrt((dimension + 2.0) / (dimension - 2.0))
    axes = _axis_points(dimension, radius_axis)
    diagonal = _reflected_diagonal(dimension, radius_diagonal)
    points = np.concatenate((axes, diagonal), axis=0)
    weight_axis = 4.0 / (dimension + 2.0) ** 2
    weight_diagonal = ((dimension - 2.0) / (dimension + 2.0)) ** 2 / reflected_count
    weights = np.concatenate(
        (
            np.full((axes.shape[0],), weight_axis),
            np.full((diagonal.shape[0],), weight_diagonal),
        )
    )
    return CubatureRuleData(
        points,
        weights,
        exact_degree=5,
        family="stroud-secrest-5-3",
        reference_domain="standard-normal",
        backend="analytic",
        source_id=f"stroud-secrest-1966:5-3:{dimension}",
        maximum_rule_bytes=maximum_rule_bytes,
    )


def gaussian_cubature_rule_data(
    dimension: int,
    degree: int,
    /,
    *,
    family: GaussianCubatureFamily = "auto",
    maximum_points: int = _DEFAULT_MAXIMUM_POINTS,
    maximum_rule_bytes: int = _DEFAULT_MAXIMUM_BYTES,
) -> CubatureRuleData:
    """Return one positive cubature rule for a standard-normal measure."""
    dimension_ = _dimension(dimension)
    degree_ = _degree(degree)
    maximum = _maximum_points(maximum_points)
    if family not in (
        "auto",
        "stroud-secrest-3-1",
        "hadamard-3",
        "stroud-secrest-5-2",
        "stroud-secrest-5-3",
        "tensor-hermite",
    ):
        raise ValueError(f"Unsupported Gaussian cubature family: {family!r}.")
    selected = family
    if selected == "auto":
        if degree_ <= 3:
            selected = "stroud-secrest-3-1"
        elif dimension_ == 1:
            selected = "tensor-hermite"
        elif dimension_ == 2:
            selected = "stroud-secrest-5-2"
        else:
            selected = "stroud-secrest-5-3"
    if selected in ("stroud-secrest-3-1", "hadamard-3") and degree_ > 3:
        raise ValueError(f"{selected} is exact only through degree three.")
    if selected == "stroud-secrest-3-1":
        return _stroud_secrest_3_1(dimension_, maximum, maximum_rule_bytes)
    if selected == "hadamard-3":
        return _hadamard_3(dimension_, maximum, maximum_rule_bytes)
    if selected == "tensor-hermite":
        return _tensor_hermite(dimension_, maximum, maximum_rule_bytes)
    if selected == "stroud-secrest-5-2":
        return _stroud_secrest_5_2(dimension_, maximum, maximum_rule_bytes)
    return _stroud_secrest_5_3(dimension_, maximum, maximum_rule_bytes)


__all__ = ["GaussianCubatureFamily", "gaussian_cubature_rule_data"]
