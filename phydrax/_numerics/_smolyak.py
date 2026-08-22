#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from functools import lru_cache
from typing import cast, Literal, NamedTuple, Sequence, TypeAlias

import numpy as np

from .._polynomial._orthogonal import standard_normal_hermite_rule_data
from ._quadrature_rules import clenshaw_curtis_data


SmolyakAxisRule: TypeAlias = Literal[
    "clenshaw-curtis",
    "leja",
    "gauss-hermite",
]
SparseIndex: TypeAlias = tuple[tuple[int, int], ...]
NodeIdentifier: TypeAlias = tuple[str, int, int]


class SmolyakTerm(NamedTuple):
    """One nonzero tensor-product term in a Smolyak combination."""

    index: SparseIndex
    coefficient: int


class SmolyakAxisData(NamedTuple):
    """Canonical nodes and operator data for one Smolyak axis level."""

    nodes: np.ndarray
    quadrature_weights: np.ndarray | None
    barycentric_weights: np.ndarray
    node_ids: tuple[NodeIdentifier, ...]
    nested: bool
    reference_measure: Literal["uniform", "standard-normal"]


def normalize_anisotropy(
    dimension: int,
    anisotropy: Sequence[float] | None,
    /,
) -> tuple[float, ...]:
    """Validate and normalize one positive finite weight per dimension."""
    dimension_ = int(dimension)
    if dimension_ < 1:
        raise ValueError("Smolyak dimension must be positive.")
    if anisotropy is None:
        return (1.0,) * dimension_
    weights = tuple(float(value) for value in anisotropy)
    if len(weights) != dimension_:
        raise ValueError("anisotropy must contain one value per dimension.")
    if any(not math.isfinite(value) or value <= 0.0 for value in weights):
        raise ValueError("anisotropy values must be finite and positive.")
    return weights


def normalize_axis_rules(
    dimension: int,
    axis_rules: SmolyakAxisRule | Sequence[SmolyakAxisRule] | None,
    /,
    *,
    default: SmolyakAxisRule,
    allowed: tuple[SmolyakAxisRule, ...],
) -> tuple[SmolyakAxisRule, ...]:
    """Broadcast and validate a closed set of per-axis Smolyak rules."""
    dimension_ = int(dimension)
    if axis_rules is None:
        rules: tuple[str, ...] = (default,) * dimension_
    elif isinstance(axis_rules, str):
        rules = (axis_rules,) * dimension_
    else:
        rules = tuple(str(rule) for rule in axis_rules)
    if len(rules) != dimension_:
        raise ValueError("axis_rules must contain one rule per dimension.")
    invalid = tuple(rule for rule in rules if rule not in allowed)
    if invalid:
        choices = ", ".join(repr(rule) for rule in allowed)
        raise ValueError(
            f"Unsupported Smolyak axis rule {invalid[0]!r}; expected {choices}."
        )
    return cast(tuple[SmolyakAxisRule, ...], rules)


def _increment(index: SparseIndex, axis: int, /) -> SparseIndex:
    out = list(index)
    for position, (current_axis, level) in enumerate(out):
        if current_axis == axis:
            out[position] = (axis, level + 1)
            return tuple(out)
        if current_axis > axis:
            out.insert(position, (axis, 1))
            return tuple(out)
    out.append((axis, 1))
    return tuple(out)


def _decrement(index: SparseIndex, axis: int, /) -> SparseIndex:
    out = list(index)
    for position, (current_axis, level) in enumerate(out):
        if current_axis != axis:
            continue
        if level == 1:
            del out[position]
        else:
            out[position] = (axis, level - 1)
        return tuple(out)
    raise ValueError(f"Sparse index does not contain axis {axis}.")


def dense_index(index: SparseIndex, dimension: int, /) -> tuple[int, ...]:
    """Expand a sparse multi-index into a dense tuple."""
    dense = [0] * int(dimension)
    for axis, level in index:
        dense[axis] = level
    return tuple(dense)


def _maximum_level(
    products: tuple[float, ...],
    budget: float,
    weight: float,
    /,
) -> int:
    upper = math.nextafter(budget, math.inf)
    remaining = max(0.0, math.fsum((upper, -math.fsum(products))))
    level = max(0, int(math.floor(remaining / weight)) + 2)
    while level > 0 and math.fsum((*products, level * weight)) > upper:
        level -= 1
    while math.fsum((*products, (level + 1) * weight)) <= upper:
        level += 1
    return level


@lru_cache(maxsize=256)
def _weighted_total_degree_indices_cached(
    dimension: int,
    level: int,
    anisotropy: tuple[float, ...],
) -> tuple[SparseIndex, ...]:
    budget = float(level - 1)
    indices: list[SparseIndex] = []
    stack: list[tuple[int, SparseIndex, tuple[float, ...]]] = [(0, (), ())]
    while stack:
        axis, prefix, products = stack.pop()
        if axis == dimension:
            indices.append(prefix)
            continue
        weight = anisotropy[axis]
        maximum = _maximum_level(products, budget, weight)
        for axis_level in range(maximum, -1, -1):
            if axis_level == 0:
                next_prefix = prefix
                next_products = products
            else:
                next_prefix = prefix + ((axis, axis_level),)
                next_products = products + (axis_level * weight,)
            stack.append((axis + 1, next_prefix, next_products))
    indices.sort(key=lambda index: dense_index(index, dimension))
    return tuple(indices)


def weighted_total_degree_indices(
    dimension: int,
    level: int,
    anisotropy: Sequence[float] | None = None,
    /,
) -> tuple[SparseIndex, ...]:
    """Enumerate only members of a weighted total-degree lower set."""
    dimension_ = int(dimension)
    level_ = int(level)
    if level_ < 1:
        raise ValueError("Smolyak level must be positive.")
    weights = normalize_anisotropy(dimension_, anisotropy)
    return _weighted_total_degree_indices_cached(dimension_, level_, weights)


@lru_cache(maxsize=256)
def _smolyak_terms_cached(
    dimension: int,
    level: int,
    anisotropy: tuple[float, ...],
) -> tuple[SmolyakTerm, ...]:
    indices = _weighted_total_degree_indices_cached(dimension, level, anisotropy)
    coefficients = {index: 1 for index in indices}
    by_axis: list[list[SparseIndex]] = [[] for _ in range(dimension)]
    for index in indices:
        for axis, _ in index:
            by_axis[axis].append(index)
    for axis, members in enumerate(by_axis):
        updates = tuple(
            (_decrement(index, axis), -coefficients[index]) for index in members
        )
        for lower, delta in updates:
            coefficients[lower] += delta
    return tuple(
        SmolyakTerm(index, coefficients[index])
        for index in indices
        if coefficients[index] != 0
    )


def smolyak_terms(
    dimension: int,
    level: int,
    anisotropy: Sequence[float] | None = None,
    /,
) -> tuple[SmolyakTerm, ...]:
    """Return the nonzero combination terms for a weighted lower set."""
    dimension_ = int(dimension)
    level_ = int(level)
    if level_ < 1:
        raise ValueError("Smolyak level must be positive.")
    weights = normalize_anisotropy(dimension_, anisotropy)
    return _smolyak_terms_cached(dimension_, level_, weights)


def _normalized_barycentric_weights(nodes: np.ndarray, /) -> np.ndarray:
    count = int(nodes.shape[0])
    if count == 1:
        return np.ones((1,), dtype=float)
    differences = nodes[:, None] - nodes[None, :]
    differences[np.diag_indices(count)] = 1.0
    weights = 1.0 / np.prod(differences, axis=1)
    scale = np.max(np.abs(weights))
    if not np.isfinite(scale) or scale == 0.0:
        raise FloatingPointError("Barycentric weights are not finite and nonzero.")
    return weights / scale


def _clenshaw_curtis_ids(level: int, count: int, /) -> tuple[NodeIdentifier, ...]:
    if level == 0:
        return (("clenshaw-curtis", 1, 1),)
    identifiers: list[NodeIdentifier] = []
    for index in range(count):
        numerator = index
        exponent = level
        while exponent > 0 and numerator % 2 == 0:
            numerator //= 2
            exponent -= 1
        identifiers.append(("clenshaw-curtis", numerator, exponent))
    return tuple(identifiers)


@lru_cache(maxsize=512)
def _leja_nodes(count: int) -> tuple[float, ...]:
    count_ = int(count)
    if count_ < 1:
        raise ValueError("Leja node count must be positive.")
    nodes = [0.0, 1.0, -1.0, 1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)]
    if count_ > len(nodes):
        for index in range(len(nodes), count_):
            if index % 2 == 0:
                nodes.append(-nodes[index - 1])
            else:
                parent = nodes[(index + 1) // 2]
                nodes.append(math.sqrt(0.5 * (parent + 1.0)))
    return tuple(nodes[:count_])


@lru_cache(maxsize=512)
def smolyak_axis_data(rule: SmolyakAxisRule, level: int, /) -> SmolyakAxisData:
    """Materialize one canonical interpolation/quadrature operator level."""
    level_ = int(level)
    if level_ < 0:
        raise ValueError("Smolyak axis level must be non-negative.")
    if rule == "clenshaw-curtis":
        if level_ == 0:
            nodes = np.asarray([0.0], dtype=float)
            quadrature_weights = np.asarray([2.0], dtype=float)
        else:
            data = clenshaw_curtis_data(2**level_ + 1)
            nodes = np.asarray(data.nodes, dtype=float)
            quadrature_weights = np.asarray(data.weights, dtype=float)
        count = int(nodes.shape[0])
        signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
        if count > 1:
            signs[0] *= 0.5
            signs[-1] *= 0.5
        return SmolyakAxisData(
            nodes,
            quadrature_weights,
            signs,
            _clenshaw_curtis_ids(level_, count),
            True,
            "uniform",
        )
    if rule == "leja":
        nodes = np.asarray(_leja_nodes(level_ + 1), dtype=float)
        return SmolyakAxisData(
            nodes,
            None,
            _normalized_barycentric_weights(nodes),
            tuple(("leja", index, 0) for index in range(level_ + 1)),
            True,
            "uniform",
        )
    if rule == "gauss-hermite":
        rule_data = standard_normal_hermite_rule_data(level_ + 1)
        nodes = np.asarray(rule_data.nodes, dtype=float)
        quadrature_weights = np.asarray(rule_data.weights, dtype=float)
        return SmolyakAxisData(
            nodes,
            quadrature_weights,
            _normalized_barycentric_weights(nodes),
            tuple(("gauss-hermite", level_, index) for index in range(level_ + 1)),
            False,
            "standard-normal",
        )
    raise ValueError(f"Unsupported Smolyak axis rule {rule!r}.")


def axis_level(index: SparseIndex, axis: int, /) -> int:
    """Return one axis level from a sparse multi-index."""
    for current_axis, level in index:
        if current_axis == axis:
            return level
        if current_axis > axis:
            break
    return 0


__all__ = [
    "NodeIdentifier",
    "SmolyakAxisData",
    "SmolyakAxisRule",
    "SmolyakTerm",
    "SparseIndex",
    "axis_level",
    "dense_index",
    "normalize_anisotropy",
    "normalize_axis_rules",
    "smolyak_axis_data",
    "smolyak_terms",
    "weighted_total_degree_indices",
]
