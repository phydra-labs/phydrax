#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Callable, Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .. import ein
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer, positive_integer
from ..typing import (
    AnyShape,
    as_array,
    as_host_array,
    Dim,
    HostInexact,
    Inexact,
    Int32,
    parse,
    Scope,
    Size,
    VariadicDim,
)


class _VariableDim(Dim, minimum=1):
    """Polynomial input variables of one product graph."""


class _StepDim(Dim):
    """Non-constant product-graph nodes, one multiplication each."""


class _TermDim(Dim, minimum=1):
    """Canonical monomial terms of one product graph."""


class _LaneDims(VariadicDim):
    """Independent evaluation lanes sharing one product graph."""


class _PayloadDims(VariadicDim):
    """Output payload carried by each canonical term coefficient."""


def _term_order(term: tuple[int, ...], /) -> tuple[int, tuple[int, ...]]:
    return len(term), term


def _canonical_monomial(
    monomial: Sequence[int], variable_count: int, name: str, /
) -> tuple[int, ...]:
    if isinstance(monomial, (str, bytes)) or not isinstance(monomial, Sequence):
        raise TypeError(f"{name} must be a sequence of variable indices.")
    indices: list[int] = []
    for index in monomial:
        index_ = nonnegative_integer(index, f"{name} variable index")
        if index_ >= variable_count:
            raise ValueError(
                f"{name} variable index {index_} is outside [0, {variable_count})."
            )
        indices.append(index_)
    return tuple(sorted(indices))


def _prefix_levels(
    terms: tuple[tuple[int, ...], ...], depth: int, /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Every distinct nondecreasing prefix, grouped by length, lexicographic per level."""
    return tuple(
        tuple(sorted({term[:level] for term in terms if len(term) >= level}))
        for level in range(depth + 1)
    )


@final
class ProductGraphPlan(StrictModule, NonTrainableState):
    """Canonical prefix-sharing product graph for commutative monomials.

    Every distinct sorted prefix of every canonical term is one node; the node of
    prefix ``(i_1, ..., i_k)`` is its parent prefix ``(i_1, ..., i_{k-1})`` times
    ``x[i_k]``. Level ``0`` is the single constant node with value one. Products are
    formed by multiplication only, so derivatives are exact at zero inputs.

    ``parent`` and ``factor`` list the non-constant nodes level by level (level
    sizes in ``level_sizes``); ``parent`` indexes the previous level's nodes.
    ``term_node`` gathers each canonical term from the concatenated node values.
    ``node_count`` includes the constant node.
    """

    __strict_contract__ = True

    parent: Int32[_StepDim]
    factor: Int32[_StepDim]
    term_node: Int32[_TermDim]
    variable_count: int = eqx.field(static=True)
    maximum_degree: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    level_sizes: tuple[int, ...] = eqx.field(static=True)
    terms: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    request_terms: tuple[int, ...] = eqx.field(static=True)
    term_requests: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        variable_count: int,
        monomials: Sequence[Sequence[int]],
        /,
        *,
        maximum_nodes: int,
        maximum_degree: int | None = None,
    ) -> None:
        variable_count_ = positive_integer(variable_count, "variable_count")
        maximum_nodes_ = positive_integer(maximum_nodes, "maximum_nodes")
        degree_bound = (
            None
            if maximum_degree is None
            else nonnegative_integer(maximum_degree, "maximum_degree")
        )
        if isinstance(monomials, (str, bytes)) or not isinstance(monomials, Sequence):
            raise TypeError("monomials must be a sequence of monomials.")
        if len(monomials) == 0:
            raise ValueError("monomials must contain at least one monomial.")
        requests = tuple(
            _canonical_monomial(monomial, variable_count_, f"monomials[{position}]")
            for position, monomial in enumerate(monomials)
        )
        if degree_bound is not None:
            for position, request in enumerate(requests):
                if len(request) > degree_bound:
                    raise ValueError(
                        f"monomials[{position}] has degree {len(request)}, "
                        f"exceeding maximum_degree={degree_bound}."
                    )
        terms = tuple(sorted(set(requests), key=_term_order))
        term_position = {term: position for position, term in enumerate(terms)}
        request_terms = tuple(term_position[request] for request in requests)
        term_requests = tuple(
            np.bincount(
                np.asarray(request_terms, dtype=np.int64), minlength=len(terms)
            ).tolist()
        )
        depth = len(terms[-1])
        levels = _prefix_levels(terms, depth)
        level_sizes = tuple(len(level) for level in levels)
        node_count = sum(level_sizes)
        if node_count > maximum_nodes_:
            raise ValueError(
                f"Product graph requires {node_count} nodes, exceeding "
                f"maximum_nodes={maximum_nodes_}."
            )
        level_position = [
            {prefix: position for position, prefix in enumerate(level)}
            for level in levels
        ]
        level_offsets = tuple(itertools.accumulate(level_sizes, initial=0))
        parent = np.asarray(
            [
                level_position[level - 1][prefix[:-1]]
                for level in range(1, depth + 1)
                for prefix in levels[level]
            ],
            dtype=np.int32,
        ).reshape((node_count - 1,))
        factor = np.asarray(
            [prefix[-1] for level in levels[1:] for prefix in level], dtype=np.int32
        ).reshape((node_count - 1,))
        term_node = np.asarray(
            [
                level_offsets[len(term)] + level_position[len(term)][term]
                for term in terms
            ],
            dtype=np.int32,
        )
        self.parent = jnp.asarray(parent)
        self.factor = jnp.asarray(factor)
        self.term_node = jnp.asarray(term_node)
        self.variable_count = variable_count_
        self.maximum_degree = depth
        self.node_count = node_count
        self.level_sizes = level_sizes
        self.terms = terms
        self.request_terms = request_terms
        self.term_requests = term_requests
        self.plan_id = canonical_fingerprint(
            {
                "kind": "product-graph-plan",
                "variable_count": variable_count_,
                "terms": [list(term) for term in terms],
                "level_sizes": list(level_sizes),
                "parent": parent.tolist(),
                "factor": factor.tolist(),
                "term_node": term_node.tolist(),
            }
        )

    @classmethod
    def total_degree(
        cls,
        variable_count: int,
        maximum_degree: int,
        /,
        *,
        include_constant: bool = False,
        minimum_degree: int = 1,
        maximum_nodes: int,
    ) -> ProductGraphPlan:
        """Plan every monomial of total degree in ``[minimum_degree, maximum_degree]``.

        With ``include_constant`` the range is ``[0, maximum_degree]`` and
        ``minimum_degree`` is ignored. Every multiset of degree below the maximum is
        a prefix of a maximal term, so the graph has ``comb(n + D, D)`` nodes; that
        count is refused before any monomial is enumerated.
        """
        variable_count_ = positive_integer(variable_count, "variable_count")
        degree = nonnegative_integer(maximum_degree, "maximum_degree")
        maximum_nodes_ = positive_integer(maximum_nodes, "maximum_nodes")
        if not isinstance(include_constant, bool):
            raise TypeError("include_constant must be a bool.")
        lowest = (
            0 if include_constant else positive_integer(minimum_degree, "minimum_degree")
        )
        if lowest > degree:
            raise ValueError(f"minimum_degree={lowest} exceeds maximum_degree={degree}.")
        node_count = math.comb(variable_count_ + degree, degree)
        if node_count > maximum_nodes_:
            raise ValueError(
                f"Total-degree product graph requires {node_count} nodes, exceeding "
                f"maximum_nodes={maximum_nodes_}."
            )
        monomials = [
            term
            for level in range(lowest, degree + 1)
            for term in itertools.combinations_with_replacement(
                range(variable_count_), level
            )
        ]
        return cls(
            variable_count_,
            monomials,
            maximum_nodes=maximum_nodes_,
            maximum_degree=degree,
        )

    def term_index(self, monomial: Sequence[int], /) -> int:
        """Return the canonical term index of ``monomial`` in any factor order."""
        term = _canonical_monomial(monomial, self.variable_count, "monomial")
        position = bisect.bisect_left(self.terms, _term_order(term), key=_term_order)
        if position == len(self.terms) or self.terms[position] != term:
            raise ValueError(f"Monomial {term} is not a term of this product graph.")
        return position

    def live_bytes(self, lane_count: int, itemsize: int, /) -> int:
        """Bytes of node values live for ``lane_count`` lanes (all levels retained)."""
        lanes = nonnegative_integer(lane_count, "lane_count")
        item = positive_integer(itemsize, "itemsize")
        return lanes * self.node_count * item

    def evaluate(self, variables: ArrayLike, /, *, lane_tile: int | None = None) -> Array:
        """Evaluate ``(*lanes, variable_count)`` inputs to ``(*lanes, term_count)``."""
        values, _ = self._variables(variables)
        lane_shape = values.shape[:-1]
        flat = values.reshape((-1, self.variable_count))
        terms = _map_lanes(self._monomials, flat, lane_tile)
        return terms.reshape((*lane_shape, len(self.terms)))

    def contract(
        self,
        variables: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        lane_tile: int | None = None,
    ) -> Array:
        """Contract term values with dynamic ``(term_count, *out)`` coefficients."""
        values, scope = self._variables(variables)
        weights = as_array(
            coefficients, Inexact[_TermDim, _PayloadDims], "coefficients", scope=scope
        )
        if weights.dtype != values.dtype:
            raise TypeError(
                f"coefficients dtype {weights.dtype} must equal variables dtype "
                f"{values.dtype}."
            )
        lane_shape = values.shape[:-1]
        payload_shape = weights.shape[1:]
        flat = values.reshape((-1, self.variable_count))
        flat_weights = weights.reshape((len(self.terms), -1))

        def project(lanes: Array, /) -> Array:
            return ein.contract("lt,tp->lp", self._monomials(lanes), flat_weights)

        projected = _map_lanes(project, flat, lane_tile)
        return projected.reshape((*lane_shape, *payload_shape))

    def _variables(self, variables: ArrayLike, /) -> tuple[Array, Scope]:
        scope = Scope()
        parse(self.variable_count, Size[_VariableDim], "variable_count", scope=scope)
        parse(len(self.terms), Size[_TermDim], "term_count", scope=scope)
        values = as_array(
            variables, Inexact[_LaneDims, _VariableDim], "variables", scope=scope
        )
        return values, scope

    def _monomials(self, variables: Array, /) -> Array:
        """Term values of flat ``(lanes, variable_count)`` inputs."""
        level = jnp.ones((variables.shape[0], 1), dtype=variables.dtype)
        nodes = [level]
        start = 0
        # Graph depth is static architecture: one multiplication pass per level.
        for size in self.level_sizes[1:]:
            stop = start + size
            level = (
                level[:, self.parent[start:stop]] * variables[:, self.factor[start:stop]]
            )
            nodes.append(level)
            start = stop
        return jnp.concatenate(nodes, axis=-1)[:, self.term_node]


def _map_lanes(
    kernel: Callable[[Array], Array], lanes: Array, lane_tile: int | None, /
) -> Array:
    """Apply ``kernel`` to flat lanes, optionally in bounded ``lane_tile`` tiles."""
    if lane_tile is None:
        return kernel(lanes)
    tile = positive_integer(lane_tile, "lane_tile")
    lane_count = lanes.shape[0]
    tile_count = -(-lane_count // tile)
    # Zero padding is domain-safe: the graph only multiplies, and padded lanes are
    # removed before return.
    padded = jnp.pad(lanes, ((0, tile_count * tile - lane_count), (0, 0)))
    tiles = jax.lax.map(kernel, padded.reshape((tile_count, tile, lanes.shape[1])))
    return tiles.reshape((tile_count * tile, *tiles.shape[2:]))[:lane_count]


def project_tuple_coefficients(
    tensor: ArrayLike,
    variable_count: int,
    degree: int,
    /,
    *,
    maximum_entries: int,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Fold an index-tuple coefficient tensor onto canonical commutative monomials.

    ``tensor`` has shape ``(variable_count,) * degree + payload``; entry
    ``T[i_1, ..., i_d]`` multiplies ``x[i_1] * ... * x[i_d]``. Returns the canonical
    degree-``degree`` terms touched by at least one nonzero entry, in lexicographic
    order, and coefficients of shape ``(terms, *payload)`` where each row sums every
    entry whose sorted index tuple is that term. A term is dropped only when every
    contributing entry is exactly zero.
    """
    variable_count_ = positive_integer(variable_count, "variable_count")
    degree_ = positive_integer(degree, "degree")
    maximum = positive_integer(maximum_entries, "maximum_entries")
    shape = np.shape(tensor)
    index_shape = (variable_count_,) * degree_
    if tuple(shape[:degree_]) != index_shape:
        raise ValueError(
            f"tensor must have leading shape {index_shape}; got {tuple(shape)}."
        )
    payload_shape = tuple(shape[degree_:])
    payload_size = math.prod(payload_shape)
    tuple_count = variable_count_**degree_
    entries = tuple_count * payload_size
    if entries > maximum:
        raise ValueError(
            f"tensor has {entries} entries, exceeding maximum_entries={maximum}."
        )
    host = as_host_array(tensor, HostInexact[AnyShape], "tensor")
    values = host.reshape((tuple_count, payload_size))
    index_rows = np.indices(index_shape, dtype=np.int64).reshape((degree_, tuple_count))
    canonical_rows = np.sort(index_rows, axis=0)
    # Row-major keys of sorted tuples order canonical terms lexicographically.
    keys = np.ravel_multi_index(tuple(canonical_rows), index_shape)
    term_keys, term_of_tuple = np.unique(keys, return_inverse=True)
    sums = np.zeros((term_keys.shape[0], payload_size), dtype=host.dtype)
    np.add.at(sums, term_of_tuple, values)
    touched = np.zeros((term_keys.shape[0],), dtype=np.bool_)
    np.logical_or.at(touched, term_of_tuple, np.any(values != 0, axis=1))
    term_rows = np.stack(np.unravel_index(term_keys[touched], index_shape), axis=1)
    terms = tuple(tuple(row) for row in term_rows.tolist())
    return terms, sums[touched].reshape((len(terms), *payload_shape))


__all__ = ["ProductGraphPlan", "project_tuple_coefficients"]
