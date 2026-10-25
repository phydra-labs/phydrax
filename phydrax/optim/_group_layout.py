#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Group and displacement-site reductions shared by stepwise optimizers."""

from __future__ import annotations

from typing import final, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, core
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._dtype_names import inexact_result_type
from .._strict import StrictModule
from .._tree_math import (
    tree_allfinite as _tree_allfinite,
    tree_inner as _tree_inner,
)
from ..sparse._ops import route_dot, route_multiply, route_reduce
from ..sparse._relation import EdgeRelation
from ..typing import Dim


class OptimizerGroupDim(Dim, minimum=1):
    """Number of independently controlled optimizer groups."""


_INDEX_RANGE_MESSAGE = "group_index values must lie in [-1, group_count)."


def _site_relation(index: ArrayLike, group_count: int, /) -> EdgeRelation:
    """Prepare the site-to-group routes of one ``group_index`` leaf.

    Route ``r`` is the ``r``-th site in C order; it targets its group and is
    valid only for a non-negative index, so inactive sites are inert in every
    reduction and receive zero from every group-to-site map.
    """

    if isinstance(index, core.Tracer):
        if not jnp.issubdtype(index.dtype, jnp.integer):
            raise TypeError("group_index leaves must be integer arrays.")
        flat = index.reshape(-1).astype(jnp.int32)
        flat = eqx.error_if(flat, jnp.any(flat < -1), _INDEX_RANGE_MESSAGE)
        valid = flat >= 0
        return EdgeRelation(
            jnp.arange(flat.shape[0], dtype=jnp.int32),
            jnp.where(valid, flat, 0),
            source_size=flat.shape[0],
            target_size=group_count,
            valid=valid,
        )
    host = np.asarray(index)
    if not np.issubdtype(host.dtype, np.integer):
        raise TypeError("group_index leaves must be integer arrays.")
    if host.size and (host.min() < -1 or host.max() >= group_count):
        raise ValueError(_INDEX_RANGE_MESSAGE)
    flat = host.reshape(-1).astype(np.int32)
    valid = flat >= 0
    return EdgeRelation(
        np.arange(flat.size, dtype=np.int32),
        np.where(valid, flat, 0),
        source_size=flat.size,
        target_size=group_count,
        valid=valid,
    )


_LeafRoutes: TypeAlias = tuple[EdgeRelation, tuple[int, ...], Array]


def _leaf_routes(
    relations: tuple[EdgeRelation, ...],
    site_shapes: tuple[tuple[int, ...], ...],
    tree: PyTree[Array],
    /,
) -> tuple[_LeafRoutes, ...]:
    """Pair every leaf, viewed as ``(sites, *site_vector)``, with its routes."""

    return tuple(
        (
            relation,
            site_shape,
            jnp.reshape(leaf, (relation.capacity, *jnp.shape(leaf)[len(site_shape) :])),
        )
        for relation, site_shape, leaf in zip(
            relations, site_shapes, jax.tree.leaves(tree), strict=True
        )
    )


def _site_total(routes: Array, /) -> Array:
    """Sum each site vector to one value per route."""

    return jnp.sum(routes, axis=tuple(range(1, routes.ndim)), dtype=routes.dtype)


def _to_leaf(
    site_values: Array, site_shape: tuple[int, ...], leaf_shape: tuple[int, ...], /
) -> Array:
    """Broadcast one value per route over its site vector."""

    expanded = jnp.reshape(
        site_values, site_shape + (1,) * (len(leaf_shape) - len(site_shape))
    )
    return jnp.broadcast_to(expanded, leaf_shape)


@final
class OptimizerGroupLayout(StrictModule):
    """Independent optimizer groups and displacement sites over one parameter PyTree.

    Every ``group_index`` leaf is an integer array whose shape is a leading prefix
    of the matching parameter leaf shape. Its axes enumerate displacement sites and
    the remaining trailing parameter axes form one site vector: a length-``N``
    index over ``(N, 3)`` coordinates makes one Cartesian site per atom, a
    zero-rank index makes the whole leaf one site. Values in
    ``[0, group_count)`` assign a site to a group. ``-1`` marks an inactive site
    (capacity padding or fixed degrees of freedom); it never contributes to a
    reduction and is never moved by a method using this layout.

    Construction prepares one sparse site-to-group `EdgeRelation` per leaf
    (``relations``, with the leaf ``site_shapes`` and parameter ``structure``);
    every group reduction and group-to-site map executes these prepared routes,
    and an iteration never re-derives them. Without ``group_index`` the whole
    PyTree is one group and one site.

    Group reductions are sums of site contributions and maxima of site norms.
    When sites are sharded over a mapped axis, ``collective_axis`` names that axis
    and completes each partial reduction with ``psum``/``pmax``, so every shard
    applies identical group controls.
    """

    relations: tuple[EdgeRelation, ...] | None
    site_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    structure: PyTreeDef | None = eqx.field(static=True)
    group_count: int = eqx.field(static=True)
    collective_axis: str | None = eqx.field(static=True)

    def __init__(
        self,
        group_index: PyTree[ArrayLike] | None = None,
        /,
        *,
        group_count: int | None = None,
        collective_axis: str | None = None,
    ) -> None:
        if collective_axis is not None and (
            not isinstance(collective_axis, str) or not collective_axis
        ):
            raise ValueError("collective_axis must be a non-empty string or None.")
        if group_index is None:
            if group_count not in (None, 1):
                raise ValueError("A layout without group_index has exactly one group.")
            self.relations = None
            self.site_shapes = ()
            self.structure = None
            self.group_count = 1
            self.collective_axis = collective_axis
            return
        if isinstance(group_count, bool) or not isinstance(group_count, int):
            raise TypeError("group_count must be an int when group_index is given.")
        if group_count < 1:
            raise ValueError("group_count must be positive.")
        leaves, structure = jax.tree.flatten(group_index)
        if not leaves:
            raise ValueError("group_index must contain at least one leaf.")
        relations = tuple(_site_relation(leaf, group_count) for leaf in leaves)
        self.relations = relations
        self.site_shapes = tuple(
            tuple(int(size) for size in np.shape(leaf)) for leaf in leaves
        )
        self.structure = structure
        self.group_count = group_count
        self.collective_axis = collective_axis

    @property
    def grouped(self) -> bool:
        """Whether explicit site-to-group routes partition the parameters."""

        return self.relations is not None

    # -- structural admission -------------------------------------------------

    def validate(self, parameters: PyTree[Array], /) -> None:
        """Refuse a parameter PyTree this layout does not partition."""

        if self.structure is None:
            return
        if self.structure != jax.tree.structure(parameters):
            raise ValueError("group_index must have the parameter PyTree structure.")
        for site_shape, parameter in zip(
            self.site_shapes,
            jax.tree.leaves(parameters),
            strict=True,
        ):
            shape = tuple(jnp.shape(parameter))
            if len(site_shape) > len(shape) or site_shape != shape[: len(site_shape)]:
                raise ValueError(
                    "Each group_index leaf shape must be a leading prefix of its "
                    "parameter leaf shape."
                )

    def _dtype(self, *trees: PyTree[Array]) -> np.dtype:
        return inexact_result_type(
            *(leaf.dtype for tree in trees for leaf in jax.tree.leaves(tree))
        )

    def _complete_sum(self, values: Array, /) -> Array:
        if self.collective_axis is None:
            return values
        return jax.lax.psum(values, self.collective_axis)

    def _complete_max(self, values: Array, /) -> Array:
        if self.collective_axis is None:
            return values
        return jax.lax.pmax(values, self.collective_axis)

    # -- reductions -----------------------------------------------------------

    def inner(self, left: PyTree[Array], right: PyTree[Array], /) -> Array:
        """Group inner products ``sum_g <left, right>`` of shape ``(group_count,)``."""

        relations = self.relations
        if relations is None:
            return self._complete_sum(jnp.reshape(_tree_inner(left, right), (1,)))
        dtype = self._dtype(left, right)
        total = jnp.zeros((self.group_count,), dtype=dtype)
        for (relation, _, left_routes), (_, _, right_routes) in zip(
            _leaf_routes(relations, self.site_shapes, left),
            _leaf_routes(relations, self.site_shapes, right),
            strict=True,
        ):
            products, _ = route_dot(
                relation, left_routes.astype(dtype), right_routes.astype(dtype)
            )
            total = total + products
        return self._complete_sum(total)

    def squared_norm(self, tree: PyTree[Array], /) -> Array:
        return self.inner(tree, tree)

    def norm(self, tree: PyTree[Array], /) -> Array:
        return jnp.sqrt(self.squared_norm(tree))

    def site_norm_max(self, tree: PyTree[Array], /) -> Array:
        """Largest site Euclidean norm in every group; zero for an empty group."""

        relations = self.relations
        if relations is None:
            return jnp.sqrt(self.squared_norm(tree))
        dtype = self._dtype(tree)
        maximum = jnp.zeros((self.group_count,), dtype=dtype)
        for relation, _, routes in _leaf_routes(relations, self.site_shapes, tree):
            squared = _site_total(jnp.square(routes.astype(dtype)))
            maximum = jnp.maximum(
                maximum, route_reduce(relation, squared, reduction="max")
            )
        return jnp.sqrt(self._complete_max(maximum))

    def count_true(self, mask: PyTree[Array], /) -> Array:
        """Number of true active elements of a boolean PyTree in each group."""

        relations = self.relations
        if relations is None:
            count = jnp.zeros((), dtype=jnp.int32)
            for leaf in jax.tree.leaves(mask):
                count = count + jnp.sum(leaf, dtype=jnp.int32)
            return self._complete_sum(jnp.reshape(count, (1,)))
        count = jnp.zeros((self.group_count,), dtype=jnp.int32)
        for relation, _, routes in _leaf_routes(relations, self.site_shapes, mask):
            count = count + route_reduce(relation, _site_total(routes.astype(jnp.int32)))
        return self._complete_sum(count)

    def group_all_finite(self, tree: PyTree[Array], /) -> Array:
        """Whether every active element of each group is finite."""

        if self.relations is None:
            finite = _tree_allfinite(tree)
            if self.collective_axis is not None:
                finite = (
                    jax.lax.psum((~finite).astype(jnp.int32), self.collective_axis) == 0
                )
            return jnp.reshape(finite, (1,))
        return self.count_true(jax.tree.map(lambda leaf: ~jnp.isfinite(leaf), tree)) == 0

    # -- group-to-element maps ------------------------------------------------

    def _group_values(self, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values)
        if values_.shape != (self.group_count,):
            raise ValueError("Group values must have shape (group_count,).")
        return values_

    def expand(self, values: ArrayLike, like: PyTree[Array], /) -> PyTree[Array]:
        """Broadcast ``(group_count,)`` values onto every element of ``like``.

        Inactive sites receive zero (or ``False``). Leaves are broadcast to the
        full parameter leaf shape.
        """

        values_ = self._group_values(values)
        relations = self.relations
        if relations is None:
            return jax.tree.map(
                lambda leaf: jnp.broadcast_to(values_[0], jnp.shape(leaf)), like
            )
        leaves = [
            _to_leaf(
                route_multiply(
                    relation,
                    jnp.ones((relation.capacity,), dtype=values_.dtype),
                    values_,
                ),
                site_shape,
                tuple(jnp.shape(leaf)),
            )
            for relation, site_shape, leaf in zip(
                relations, self.site_shapes, jax.tree.leaves(like), strict=True
            )
        ]
        return jax.tree.unflatten(jax.tree.structure(like), leaves)

    def scale(self, values: ArrayLike, tree: PyTree[Array], /) -> PyTree[Array]:
        """Multiply every element by its group value; inactive sites become zero."""

        values_ = self._group_values(values)
        relations = self.relations
        if relations is None:
            return jax.tree.map(lambda leaf: leaf * values_[0].astype(leaf.dtype), tree)
        leaves = [
            jnp.reshape(
                route_multiply(relation, routes, values_.astype(routes.dtype)),
                jnp.shape(leaf),
            )
            for (relation, _, routes), leaf in zip(
                _leaf_routes(relations, self.site_shapes, tree),
                jax.tree.leaves(tree),
                strict=True,
            )
        ]
        return jax.tree.unflatten(jax.tree.structure(tree), leaves)

    def select(
        self,
        mask: ArrayLike,
        on_true: PyTree[Array],
        on_false: PyTree[Array],
        /,
    ) -> PyTree[Array]:
        """Choose ``on_true`` for active sites of selected groups, else ``on_false``."""

        mask_ = jnp.asarray(mask, dtype=jnp.bool_)
        return jax.tree.map(
            jnp.where,
            self.expand(mask_, on_true),
            on_true,
            on_false,
        )

    def clamp_sites(
        self,
        displacement: PyTree[Array],
        maximum: float | Array,
        /,
    ) -> tuple[PyTree[Array], Array]:
        """Shorten every site displacement longer than ``maximum`` to that length.

        Returns the clamped displacement and, per group, whether any site was
        shortened.
        """

        maximum_ = jnp.asarray(maximum)
        relations = self.relations
        if relations is None:
            norm = self.norm(displacement)
            limited = norm > maximum_
            factor = jnp.where(limited, maximum_ / jnp.where(limited, norm, 1.0), 1.0)[0]
            return jax.tree.map(
                lambda leaf: leaf * factor.astype(leaf.dtype), displacement
            ), limited

        clamped = []
        limited_count = jnp.zeros((self.group_count,), dtype=jnp.int32)
        for (relation, site_shape, routes), leaf in zip(
            _leaf_routes(relations, self.site_shapes, displacement),
            jax.tree.leaves(displacement),
            strict=True,
        ):
            norm = jnp.sqrt(_site_total(jnp.square(routes)))
            limited = (norm > maximum_) & relation.valid
            factor = jnp.where(limited, maximum_ / jnp.where(limited, norm, 1.0), 1.0)
            leaf_shape = tuple(jnp.shape(leaf))
            clamped.append(
                leaf * _to_leaf(factor.astype(leaf.dtype), site_shape, leaf_shape)
            )
            limited_count = limited_count + route_reduce(
                relation, limited.astype(jnp.int32)
            )
        return (
            jax.tree.unflatten(jax.tree.structure(displacement), clamped),
            self._complete_sum(limited_count) > 0,
        )


__all__ = ["OptimizerGroupDim", "OptimizerGroupLayout"]
