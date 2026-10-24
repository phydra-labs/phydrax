#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coarse articulated tree sweeps with a native device-resident CUDA schedule.

Three primitives own the tree recurrences of the articulated-body algorithm:

* ``factor`` — the reverse articulated-inertia recurrence producing final
  articulated inertias, projected columns ``U_e = I^A S_e``, pivots
  ``D_e = S_eᵀ U_e``, absolute pivot scales ``|S_e|ᵀ|I^A||S_e|`` and reciprocal
  pivots (zero for fixed, nonfinite or exactly zero pivots);
* ``solve`` — one reverse plus one forward substitution of ``K`` independent
  right-hand-side columns through a prepared factor, with optional body bias
  forces and edge convective accelerations;
* ``reduce`` — one reverse traversal accumulating ``K`` independent body force
  columns into scalar edge efforts ``S_eᵀ f_child``.

Every primitive carries a leading world axis ``W``. Each numerical operand has
leading extent ``W`` or one; an extent of one is shared by every world and is
read in place, never copied per world. Public wrappers bind one world;
``jax.vmap`` folds mapped operands into the world axis so batched worlds bind
one native operation while unmapped factors stay shared. Topology indices are
shared ``(E,)`` or per-world ``(1 or W, E)``. Edges are visited in reverse
storage order by reverse sweeps and in storage order by forward sweeps; the
caller supplies a topological edge order.

``_*_world`` functions are the canonical JAX meanings and preserve the original
recurrence arithmetic and grouping. Index semantics are those of ``jnp``
scalar indexing: gathers normalize a negative index once and then clamp to the
body range; scatters normalize a negative index once and drop indices that
remain out of range. Nonfinite values propagate exactly as in the reference,
including through zero products, and columns never mix. Floating operands may
differ in dtype: each carry keeps its original dtype, every step computes in
the promoted type of its operands and rounds into its carry, and result types
are those of the reference.

On an admitted CUDA target (Ampere or newer, see
``backends._mosaic.lower_mosaic_or_reference``) each primitive lowers to one
Mosaic GPU kernel whose threads each own one complete sequential tree
traversal: one thread per world for the factor and one thread per
``(world, column)`` lane for substitutions and reductions. The tree iteration
is an on-device loop; no kernel or host round trip is issued per edge.
Admission is static: float32/float64 floating operands all of one dtype (mixed
precision always lowers the reference), one signed 32/64-bit index dtype,
at least one world, body, edge and column, at most
``_MAXIMUM_EDGES`` edges, ``_MAXIMUM_BODIES`` bodies and ``_MAXIMUM_COLUMNS``
columns per world, and every buffer within signed 32-bit element range. Any
other target, dtype or shape lowers the reference. Derivatives always
differentiate the reference; direct transposes of the linear substitution and
reduction transpose the reference.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax._src import dispatch
from jax._src.core import ShapedArray
from jax.extend import core
from jax.interpreters import ad, batching, mlir
from jax.typing import ArrayLike

from ...backends._mosaic import lower_mosaic_or_reference


if TYPE_CHECKING:
    from jax.extend.mlir import ir


type Tangent = Array | ad.Zero
type Primal = Array | ad.UndefinedPrimal

# One warpgroup per CTA; one thread per independent sequential traversal.
_TILE = 128
# Static per-thread work and workspace bounds of the qualified native profile.
_MAXIMUM_EDGES = 1024
_MAXIMUM_BODIES = 4096
_MAXIMUM_COLUMNS = 256
_INT32_LIMIT = (1 << 31) - 1
_NATIVE_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
_NATIVE_INDEX_DTYPES = (np.dtype(np.int32), np.dtype(np.int64))


class _Role(IntEnum):
    """Operand roles that declare world batching of the sweep primitives."""

    WORLD = 0
    INDEX = 1
    PLACEHOLDER = 2


# ---------------------------------------------------------------------------
# Canonical single-world references.
# ---------------------------------------------------------------------------


def _add_to_carry(carry: Array, index: Array, update: Array, /) -> Array:
    """Add in the promoted arithmetic dtype, then round into the carry dtype."""
    arithmetic_dtype = jnp.result_type(carry, update)
    return (
        carry.astype(arithmetic_dtype)
        .at[index]
        .add(update.astype(arithmetic_dtype))
        .astype(carry.dtype)
    )


def _factor_world(
    spatial_inertia: Array,
    transforms: Array,
    subspaces: Array,
    moving: Array,
    parents: Array,
    children: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    """Reverse articulated-inertia recurrence of one world.

    The inertia carry keeps the spatial-inertia dtype and the edge results
    the transform (configuration) dtype; each step computes in the promoted
    type of its operands and rounds into its carry, as the original loop did.
    """

    edge_count = transforms.shape[0]
    dtype = transforms.dtype
    initial = (
        spatial_inertia,
        jnp.zeros((edge_count, 6), dtype=dtype),
        jnp.zeros((edge_count,), dtype=dtype),
        jnp.zeros((edge_count,), dtype=dtype),
        jnp.zeros((edge_count,), dtype=dtype),
    )
    if edge_count == 0:
        return initial

    type _FactorCarry = tuple[Array, Array, Array, Array, Array]

    def reverse_edge(index: Array, carry: _FactorCarry) -> _FactorCarry:
        inertia, projected, pivots, scales, reciprocal = carry
        edge = edge_count - 1 - index
        child_inertia = inertia[children[edge]]
        subspace = subspaces[edge]
        column = child_inertia @ subspace
        pivot = subspace @ column
        scale = jnp.abs(subspace) @ jnp.abs(child_inertia) @ jnp.abs(subspace)
        invertible = moving[edge] & jnp.isfinite(pivot) & (pivot != 0.0)
        inverse = jnp.where(invertible, 1.0 / jnp.where(invertible, pivot, 1.0), 0.0)
        reduced = child_inertia - inverse * jnp.outer(column, column)
        inertia = _add_to_carry(
            inertia, parents[edge], transforms[edge].T @ reduced @ transforms[edge]
        )
        return (
            inertia,
            projected.at[edge].set(column.astype(projected.dtype)),
            pivots.at[edge].set(pivot.astype(pivots.dtype)),
            scales.at[edge].set(scale.astype(scales.dtype)),
            reciprocal.at[edge].set(inverse.astype(reciprocal.dtype)),
        )

    return jax.lax.fori_loop(0, edge_count, reverse_edge, initial)


def _solve_world(
    articulated: Array,
    transforms: Array,
    subspaces: Array,
    projected: Array,
    reciprocal: Array,
    parents: Array,
    children: Array,
    edge_effort: Array,
    body_bias: Array,
    edge_convective: Array,
    /,
    *,
    bodies: int,
    has_bias: bool,
    has_convective: bool,
) -> tuple[Array]:
    """Reverse then forward substitution of ``(E, K)`` columns of one world.

    ``articulated`` is read only with ``has_convective``; ``body_bias`` and
    ``edge_convective`` only with their flags. Scalar efforts, accelerations
    and results carry the dtype of ``articulated`` (or its placeholder); a
    supplied ``body_bias`` keeps its own carry dtype.
    """

    edge_count = transforms.shape[0]
    columns = edge_effort.shape[-1]
    dtype = articulated.dtype
    if edge_count == 0 or columns == 0:
        return (jnp.zeros((edge_count, columns), dtype=dtype),)
    bias = body_bias if has_bias else jnp.zeros((bodies, 6, columns), dtype=dtype)

    type _ReverseCarry = tuple[Array, Array]

    def reverse_edge(index: Array, carry: _ReverseCarry) -> _ReverseCarry:
        bias_, scalar_effort_ = carry
        edge = edge_count - 1 - index
        child_bias = bias_[children[edge]]
        effort = edge_effort[edge] - subspaces[edge] @ child_bias
        reduced_bias = (
            child_bias + projected[edge][:, None] * (reciprocal[edge] * effort)[None, :]
        )
        if has_convective:
            reduced_inertia = articulated[children[edge]] - reciprocal[edge] * jnp.outer(
                projected[edge], projected[edge]
            )
            reduced_bias = reduced_bias + reduced_inertia @ edge_convective[edge]
        bias_ = _add_to_carry(bias_, parents[edge], transforms[edge].T @ reduced_bias)
        return bias_, scalar_effort_.at[edge].set(effort.astype(scalar_effort_.dtype))

    _, scalar_effort = jax.lax.fori_loop(
        0,
        edge_count,
        reverse_edge,
        (bias, jnp.zeros((edge_count, columns), dtype=dtype)),
    )

    type _ForwardCarry = tuple[Array, Array]

    def forward_edge(edge: Array, carry: _ForwardCarry) -> _ForwardCarry:
        acceleration_, edge_acceleration_ = carry
        base = transforms[edge] @ acceleration_[parents[edge]]
        if has_convective:
            base = base + edge_convective[edge]
        rate = reciprocal[edge] * (scalar_effort[edge] - projected[edge] @ base)
        acceleration_ = acceleration_.at[children[edge]].set(
            (base + subspaces[edge][:, None] * rate[None, :]).astype(acceleration_.dtype)
        )
        return acceleration_, edge_acceleration_.at[edge].set(
            rate.astype(edge_acceleration_.dtype)
        )

    _, edge_acceleration = jax.lax.fori_loop(
        0,
        edge_count,
        forward_edge,
        (
            jnp.zeros((bodies, 6, columns), dtype=dtype),
            jnp.zeros((edge_count, columns), dtype=dtype),
        ),
    )
    return (edge_acceleration,)


def _reduce_world(
    transforms: Array,
    subspaces: Array,
    parents: Array,
    children: Array,
    body_force: Array,
    /,
) -> tuple[Array]:
    """Reverse force accumulation of ``(C, 6, K)`` columns of one world.

    The force carry keeps the ``body_force`` dtype. Each effort is computed in
    the promoted type of subspace and force and stored once, so a caller's
    final cast equals storing into its own carry dtype.
    """

    edge_count = transforms.shape[0]
    columns = body_force.shape[-1]
    effort_dtype = jnp.result_type(subspaces, body_force)
    if edge_count == 0 or columns == 0:
        return (jnp.zeros((edge_count, columns), dtype=effort_dtype),)

    type _ReduceCarry = tuple[Array, Array]

    def reverse_edge(index: Array, carry: _ReduceCarry) -> _ReduceCarry:
        force, effort = carry
        edge = edge_count - 1 - index
        child_force = force[children[edge]]
        effort = effort.at[edge].set(subspaces[edge] @ child_force)
        force = _add_to_carry(force, parents[edge], transforms[edge].T @ child_force)
        return force, effort

    _, edge_effort = jax.lax.fori_loop(
        0,
        edge_count,
        reverse_edge,
        (body_force, jnp.zeros((edge_count, columns), dtype=effort_dtype)),
    )
    return (edge_effort,)


# ---------------------------------------------------------------------------
# World-axis references bound by the primitives.
# ---------------------------------------------------------------------------


def _solve_roles(*, has_bias: bool, has_convective: bool) -> tuple[_Role, ...]:
    world, index, placeholder = _Role.WORLD, _Role.INDEX, _Role.PLACEHOLDER
    return (
        world if has_convective else placeholder,
        world,
        world,
        world,
        world,
        index,
        index,
        world,
        world if has_bias else placeholder,
        world if has_convective else placeholder,
    )


_FACTOR_ROLES = (
    _Role.WORLD,
    _Role.WORLD,
    _Role.WORLD,
    _Role.WORLD,
    _Role.INDEX,
    _Role.INDEX,
)
_REDUCE_ROLES = (_Role.WORLD, _Role.WORLD, _Role.INDEX, _Role.INDEX, _Role.WORLD)


def _has_world_axis(shape: Sequence[int], role: _Role, /) -> bool:
    """Numerical world operands and per-world ``(W, E)`` topology carry ``W``."""
    return role is _Role.WORLD or (role is _Role.INDEX and len(shape) == 2)


def _world_count(shapes: Sequence[Sequence[int]], roles: Sequence[_Role], /) -> int:
    """Common world count; a leading extent of one is shared by every world."""

    extents = set()
    for shape, role in zip(shapes, roles, strict=True):
        if not _has_world_axis(shape, role):
            continue
        if len(shape) == 0:
            raise ValueError(
                "Articulated sweep world operands need a leading world axis."
            )
        extents.add(shape[0])
    wide = extents - {1}
    if len(wide) > 1:
        raise ValueError(
            "Articulated sweep world extents must be one or a single common count."
        )
    return next(iter(wide)) if wide else 1


def _map_worlds(
    world_function: Callable[..., tuple[Array, ...]],
    roles: Sequence[_Role],
    operands: Sequence[Array],
    /,
) -> tuple[Array, ...]:
    """Apply one single-world reference over the leading world axis.

    Operands with a leading extent of one are shared logically by every world
    and are never copied per world.
    """

    worlds = _world_count([operand.shape for operand in operands], roles)
    world_axis = tuple(
        _has_world_axis(operand.shape, role)
        for operand, role in zip(operands, roles, strict=True)
    )
    if worlds == 1:
        # A single world evaluates the unmapped recurrence exactly.
        single = tuple(
            operand[0] if axis else operand
            for operand, axis in zip(operands, world_axis, strict=True)
        )
        return tuple(value[None] for value in world_function(*single))
    mapped = tuple(
        axis and operand.shape[0] == worlds
        for operand, axis in zip(operands, world_axis, strict=True)
    )
    arguments = tuple(
        operand[0] if axis and not mapping else operand
        for operand, axis, mapping in zip(operands, world_axis, mapped, strict=True)
    )
    in_axes = tuple(0 if mapping else None for mapping in mapped)
    return tuple(jax.vmap(world_function, in_axes=in_axes)(*arguments))


def _factor_reference(
    spatial_inertia: Array,
    transforms: Array,
    subspaces: Array,
    moving: Array,
    parents: Array,
    children: Array,
) -> tuple[Array, ...]:
    """Canonical meaning of the factor primitive over its world axis."""

    return _map_worlds(
        _factor_world,
        _FACTOR_ROLES,
        (spatial_inertia, transforms, subspaces, moving, parents, children),
    )


def _solve_reference(
    articulated: Array,
    transforms: Array,
    subspaces: Array,
    projected: Array,
    reciprocal: Array,
    parents: Array,
    children: Array,
    edge_effort: Array,
    body_bias: Array,
    edge_convective: Array,
    *,
    bodies: int,
    has_bias: bool,
    has_convective: bool,
) -> tuple[Array, ...]:
    """Canonical meaning of the substitution primitive over its world axis."""

    return _map_worlds(
        partial(
            _solve_world, bodies=bodies, has_bias=has_bias, has_convective=has_convective
        ),
        _solve_roles(has_bias=has_bias, has_convective=has_convective),
        (
            articulated,
            transforms,
            subspaces,
            projected,
            reciprocal,
            parents,
            children,
            edge_effort,
            body_bias,
            edge_convective,
        ),
    )


def _forces_reference(
    transforms: Array,
    subspaces: Array,
    parents: Array,
    children: Array,
    body_force: Array,
) -> tuple[Array, ...]:
    """Canonical meaning of the force-reduction primitive over its world axis."""

    return _map_worlds(
        _reduce_world,
        _REDUCE_ROLES,
        (transforms, subspaces, parents, children, body_force),
    )


# ---------------------------------------------------------------------------
# Abstract evaluation.
# ---------------------------------------------------------------------------


def _require_real(value: ShapedArray, name: str, /) -> None:
    if not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(f"Articulated sweep {name} must be a real floating array.")


def _require(
    value: ShapedArray, worlds: int, tail: tuple[int, ...], name: str, /
) -> None:
    """Real floating operand of shape ``(1 or W,) + tail``; dtypes may differ."""
    _require_real(value, name)
    if (
        value.ndim != 1 + len(tail)
        or value.shape[0] not in (1, worlds)
        or value.shape[1:] != tail
    ):
        raise ValueError(
            f"Articulated sweep {name} must have shape (1 or {worlds},) + {tail}; "
            f"got {value.shape}."
        )


def _require_placeholder(value: ShapedArray, name: str, /) -> None:
    _require_real(value, name)
    if value.shape != ():
        raise ValueError(f"Absent articulated sweep {name} must be a scalar placeholder.")


def _require_indices(
    parents: ShapedArray, children: ShapedArray, worlds: int, edges: int, /
) -> None:
    for value, name in ((parents, "parents"), (children, "children")):
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise TypeError(f"Articulated sweep {name} must be an integer array.")
        if value.shape not in ((edges,), (1, edges), (worlds, edges)):
            raise ValueError(
                f"Articulated sweep {name} must have shape {(edges,)} or "
                f"(1 or {worlds}, {edges}); got {value.shape}."
            )


def _abstract_worlds(operands: Sequence[ShapedArray], roles: Sequence[_Role], /) -> int:
    return _world_count([value.shape for value in operands], roles)


def _reference_avals(
    reference: Callable[..., tuple[Array, ...]],
    operands: Sequence[ShapedArray],
    /,
    **params: int | bool,
) -> tuple[ShapedArray, ...]:
    """Result types are those of the canonical reference, including promotion."""

    specs = tuple(
        jax.ShapeDtypeStruct(value.shape, value.dtype, weak_type=value.weak_type)
        for value in operands
    )
    result = jax.eval_shape(partial(reference, **params), *specs)
    return tuple(
        ShapedArray(value.shape, value.dtype, weak_type=value.weak_type)
        for value in result
    )


def _factor_abstract(
    spatial_inertia: ShapedArray,
    transforms: ShapedArray,
    subspaces: ShapedArray,
    moving: ShapedArray,
    parents: ShapedArray,
    children: ShapedArray,
) -> tuple[ShapedArray, ...]:
    operands = (spatial_inertia, transforms, subspaces, moving, parents, children)
    _require_real(spatial_inertia, "spatial inertia")
    if spatial_inertia.ndim != 4 or spatial_inertia.shape[2:] != (6, 6):
        raise ValueError("Articulated sweep spatial inertia must be (world, body, 6, 6).")
    if transforms.ndim != 4:
        raise ValueError("Articulated sweep transforms must be (world, edge, 6, 6).")
    worlds = _abstract_worlds(operands, _FACTOR_ROLES)
    bodies = spatial_inertia.shape[1]
    edges = transforms.shape[1]
    _require(spatial_inertia, worlds, (bodies, 6, 6), "spatial inertia")
    _require(transforms, worlds, (edges, 6, 6), "transforms")
    _require(subspaces, worlds, (edges, 6), "subspaces")
    if (
        moving.ndim != 2
        or moving.shape[0] not in (1, worlds)
        or moving.shape[1] != edges
        or moving.dtype != jnp.bool_
    ):
        raise TypeError(
            "Articulated sweep moving mask must be a (1 or world, edge) bool array."
        )
    _require_indices(parents, children, worlds, edges)
    return _reference_avals(_factor_reference, operands)


def _solve_abstract(
    articulated: ShapedArray,
    transforms: ShapedArray,
    subspaces: ShapedArray,
    projected: ShapedArray,
    reciprocal: ShapedArray,
    parents: ShapedArray,
    children: ShapedArray,
    edge_effort: ShapedArray,
    body_bias: ShapedArray,
    edge_convective: ShapedArray,
    *,
    bodies: int,
    has_bias: bool,
    has_convective: bool,
) -> tuple[ShapedArray, ...]:
    operands = (
        articulated,
        transforms,
        subspaces,
        projected,
        reciprocal,
        parents,
        children,
        edge_effort,
        body_bias,
        edge_convective,
    )
    _require_real(transforms, "transforms")
    if transforms.ndim != 4 or transforms.shape[2:] != (6, 6):
        raise ValueError("Articulated sweep transforms must be (world, edge, 6, 6).")
    if type(bodies) is not int or bodies < 0:
        raise ValueError(
            "Articulated sweep body capacity must be a non-negative integer."
        )
    if edge_effort.ndim != 3:
        raise ValueError("Articulated sweep edge effort must be (world, edge, column).")
    worlds = _abstract_worlds(
        operands, _solve_roles(has_bias=has_bias, has_convective=has_convective)
    )
    edges = transforms.shape[1]
    columns = edge_effort.shape[2]
    _require(transforms, worlds, (edges, 6, 6), "transforms")
    _require(subspaces, worlds, (edges, 6), "subspaces")
    _require(projected, worlds, (edges, 6), "projected columns")
    _require(reciprocal, worlds, (edges,), "reciprocal pivots")
    _require(edge_effort, worlds, (edges, columns), "edge effort")
    _require_indices(parents, children, worlds, edges)
    if has_convective:
        _require(articulated, worlds, (bodies, 6, 6), "articulated inertia")
        _require(edge_convective, worlds, (edges, 6, columns), "edge convective")
    else:
        _require_placeholder(articulated, "articulated inertia")
        _require_placeholder(edge_convective, "edge convective")
    if has_bias:
        _require(body_bias, worlds, (bodies, 6, columns), "body bias")
    else:
        _require_placeholder(body_bias, "body bias")
    return _reference_avals(
        _solve_reference,
        operands,
        bodies=bodies,
        has_bias=has_bias,
        has_convective=has_convective,
    )


def _reduce_abstract(
    transforms: ShapedArray,
    subspaces: ShapedArray,
    parents: ShapedArray,
    children: ShapedArray,
    body_force: ShapedArray,
) -> tuple[ShapedArray, ...]:
    operands = (transforms, subspaces, parents, children, body_force)
    _require_real(body_force, "body force")
    if body_force.ndim != 4 or body_force.shape[2] != 6:
        raise ValueError("Articulated sweep body force must be (world, body, 6, column).")
    if transforms.ndim != 4:
        raise ValueError("Articulated sweep transforms must be (world, edge, 6, 6).")
    worlds = _abstract_worlds(operands, _REDUCE_ROLES)
    edges = transforms.shape[1]
    _require(body_force, worlds, body_force.shape[1:], "body force")
    _require(transforms, worlds, (edges, 6, 6), "transforms")
    _require(subspaces, worlds, (edges, 6), "subspaces")
    _require_indices(parents, children, worlds, edges)
    return _reference_avals(_forces_reference, operands)


# ---------------------------------------------------------------------------
# Native Mosaic GPU kernels.
# ---------------------------------------------------------------------------


def _admitted(
    *,
    floating: Sequence[Array],
    indices: Sequence[Array],
    worlds: int,
    bodies: int,
    edges: int,
    lanes: int,
    columns: int,
    buffers: Sequence[int],
) -> bool:
    """Static native profile; anything else lowers the canonical reference.

    Every floating operand must share one float32/float64 dtype: mixed
    precision keeps the reference's per-carry promotion and rounding.
    """

    floating_dtypes = {np.dtype(value.dtype) for value in floating}
    index_dtypes = {np.dtype(value.dtype) for value in indices}
    return (
        len(floating_dtypes) == 1
        and next(iter(floating_dtypes)) in _NATIVE_DTYPES
        and len(index_dtypes) == 1
        and next(iter(index_dtypes)) in _NATIVE_INDEX_DTYPES
        and worlds >= 1
        and 1 <= bodies <= _MAXIMUM_BODIES
        and 1 <= edges <= _MAXIMUM_EDGES
        and 1 <= columns <= _MAXIMUM_COLUMNS
        # Lane ids are 32-bit; buffers stay within signed 32-bit element range.
        and lanes <= _INT32_LIMIT - _TILE
        and all(size <= _INT32_LIMIT for size in buffers)
    )


class _Emitter:
    """Scalar SSA helpers for one lane of a sequential tree traversal."""

    def __init__(self, dtype: np.dtype, index_dtype: np.dtype, /) -> None:
        from jax.experimental.mosaic import gpu as mgpu
        from jax.extend.mlir import ir
        from jax.extend.mlir.dialects import arith, math as mlir_math, memref

        self._mgpu = mgpu
        self._arith = arith
        self._memref = memref
        self._math = mlir_math
        self.index = ir.IndexType.get()
        self.integer = ir.IntegerType.get_signless(8 * index_dtype.itemsize)
        self.real = (
            ir.F32Type.get() if dtype == np.dtype(np.float32) else ir.F64Type.get()
        )
        self.zero = mgpu.c(0.0, self.real)
        self.one = mgpu.c(1.0, self.real)
        self.infinity = mgpu.c(math.inf, self.real)

    def constant(self, value: int, /) -> ir.Value:
        return self._mgpu.c(value, self.index)

    def _indices(self, indices: Sequence[ir.Value | int], /) -> list[ir.Value]:
        return [
            self.constant(value) if isinstance(value, int) else value for value in indices
        ]

    def load(self, ref: ir.Value, *indices: ir.Value | int) -> ir.Value:
        return self._memref.load(ref, self._indices(indices))

    def store(self, value: ir.Value, ref: ir.Value, *indices: ir.Value | int) -> None:
        self._memref.store(value, ref, self._indices(indices))

    def add(self, left: ir.Value, right: ir.Value, /) -> ir.Value:
        return self._arith.addf(left, right)

    def sub(self, left: ir.Value, right: ir.Value, /) -> ir.Value:
        return self._arith.subf(left, right)

    def mul(self, left: ir.Value, right: ir.Value, /) -> ir.Value:
        return self._arith.mulf(left, right)

    def div(self, left: ir.Value, right: ir.Value, /) -> ir.Value:
        return self._arith.divf(left, right)

    def abs(self, value: ir.Value, /) -> ir.Value:
        return self._math.absf(value)

    def dot(self, left: Sequence[ir.Value], right: Sequence[ir.Value], /) -> ir.Value:
        """Sequential contraction in ascending component order."""
        total = self.mul(left[0], right[0])
        for first, second in zip(left[1:], right[1:], strict=True):
            total = self.add(total, self.mul(first, second))
        return total

    def select(self, condition: ir.Value, left: ir.Value, right: ir.Value, /) -> ir.Value:
        return self._arith.select(condition, left, right)

    def both(self, left: ir.Value, right: ir.Value, /) -> ir.Value:
        return self._arith.andi(left, right)

    def finite(self, value: ir.Value, /) -> ir.Value:
        # Ordered comparison: NaN and infinities are both rejected.
        return self._arith.cmpf(
            self._arith.CmpFPredicate.OLT, self.abs(value), self.infinity
        )

    def nonzero(self, value: ir.Value, /) -> ir.Value:
        return self._arith.cmpf(self._arith.CmpFPredicate.UNE, value, self.zero)

    def reversed_edge(self, step: ir.Value, edges: int, /) -> ir.Value:
        return self._arith.subi(self.constant(edges - 1), step)

    def split(self, lane: ir.Value, columns: int, /) -> tuple[ir.Value, ir.Value]:
        width = self.constant(columns)
        return self._arith.divui(lane, width), self._arith.remui(lane, width)

    def loop(self, count: int, /) -> Callable[[Callable[[ir.Value], None]], None]:
        """Emit one device-side ``scf.for`` over ``range(count)``."""

        def decorate(body: Callable[[ir.Value], None]) -> None:
            @self._mgpu.fori(self.constant(count), [])
            def _(position: ir.Value, carry: list[ir.Value]) -> list[ir.Value]:
                del carry
                body(position)
                return []

        return decorate

    def when(self, condition: ir.Value, /) -> AbstractContextManager[None]:
        return self._mgpu.when(condition)

    def _normalized(self, raw: ir.Value, size: int, /) -> ir.Value:
        """``jnp`` scalar-index normalization: one wrap of a negative index."""
        zero = self._mgpu.c(0, self.integer)
        negative = self._arith.cmpi(self._arith.CmpIPredicate.slt, raw, zero)
        return self.select(
            negative, self._arith.addi(raw, self._mgpu.c(size, self.integer)), raw
        )

    def gather_index(self, raw: ir.Value, size: int, /) -> ir.Value:
        """Normalize once, then clamp into ``[0, size)`` as ``dynamic_slice``."""
        normalized = self._normalized(raw, size)
        bounded = self._arith.minsi(
            self._arith.maxsi(normalized, self._mgpu.c(0, self.integer)),
            self._mgpu.c(size - 1, self.integer),
        )
        return self._arith.index_cast(self.index, bounded)

    def scatter_index(self, raw: ir.Value, size: int, /) -> tuple[ir.Value, ir.Value]:
        """Normalize once; out-of-range updates are dropped as ``.at[]`` does."""
        normalized = self._normalized(raw, size)
        zero = self._mgpu.c(0, self.integer)
        predicate = self._arith.CmpIPredicate
        inside = self.both(
            self._arith.cmpi(predicate.sge, normalized, zero),
            self._arith.cmpi(predicate.slt, normalized, self._mgpu.c(size, self.integer)),
        )
        safe = self.select(inside, normalized, zero)
        return self._arith.index_cast(self.index, safe), inside


def _launch(
    body: Callable[[_Emitter, ir.Value, tuple[ir.Value, ...]], None],
    operands: Sequence[Array],
    out_type: tuple[jax.ShapeDtypeStruct, ...],
    *,
    dtype: np.dtype,
    index_dtype: np.dtype,
    lanes: int,
    flops: int,
    bytes_accessed: int,
) -> tuple[Array, ...]:
    """One Mosaic GPU kernel; each thread runs ``body`` for its own lane."""

    from jax.experimental import pallas as pl
    from jax.experimental.mosaic import gpu as mgpu
    from jax.experimental.pallas import mosaic_gpu as plgpu
    from jax.extend.mlir import ir
    from jax.extend.mlir.dialects import arith, vector

    tiles = -(-lanes // _TILE)
    layout = plgpu.Layout.WG_STRIDED((_TILE,), vec_size=1)

    @plgpu.inline_mgpu(
        arg_types=(layout,) + (plgpu.RefType(),) * (len(operands) + len(out_type))
    )
    def traverse(_: object, lane_array: mgpu.FragmentedArray, *refs: ir.Value) -> None:
        i32 = ir.IntegerType.get_signless(32)
        # WG_STRIDED with vec_size=1 gives each of the 128 threads one lane.
        (register,) = lane_array.registers.flat
        lane = vector.extract(register, [], [0])
        # Tail lanes issue no loads and no stores.
        with mgpu.when(arith.cmpi(arith.CmpIPredicate.slt, lane, mgpu.c(lanes, i32))):
            emitter = _Emitter(dtype, index_dtype)
            body(emitter, arith.index_cast(emitter.index, lane), tuple(refs))

    def kernel(*refs: jax.Ref) -> None:
        start = jax.lax.axis_index("lanes") * _TILE
        ids = start + plgpu.broadcasted_iota(jnp.int32, (_TILE,), 0, layout=layout)
        traverse(plgpu.layout_cast(ids, layout), *refs)

    result = plgpu.kernel(
        kernel,
        out_type=out_type,
        grid=(tiles,),
        grid_names=("lanes",),
        compiler_params=plgpu.CompilerParams(
            lowering_semantics=plgpu.LoweringSemantics.Lane
        ),
        cost_estimate=pl.CostEstimate(
            flops=flops, transcendentals=0, bytes_accessed=bytes_accessed
        ),
    )(*operands)
    return tuple(result)


def _matrix(
    emit: _Emitter, ref: ir.Value, *prefix: ir.Value | int
) -> list[list[ir.Value]]:
    return [
        [emit.load(ref, *prefix, row, column) for column in range(6)] for row in range(6)
    ]


def _vector(emit: _Emitter, ref: ir.Value, *prefix: ir.Value | int) -> list[ir.Value]:
    return [emit.load(ref, *prefix, component) for component in range(6)]


def _column(
    emit: _Emitter, ref: ir.Value, *prefix: ir.Value | int, column: ir.Value
) -> list[ir.Value]:
    return [emit.load(ref, *prefix, component, column) for component in range(6)]


def _transposed_action(
    emit: _Emitter, transform: list[list[ir.Value]], value: Sequence[ir.Value], /
) -> list[ir.Value]:
    """``Xᵀ v`` with ascending contraction order."""
    return [
        emit.dot([transform[row][column] for row in range(6)], value)
        for column in range(6)
    ]


def _index(
    emit: _Emitter, ref: ir.Value, world: ir.Value, edge: ir.Value, shared: bool, /
) -> ir.Value:
    return emit.load(ref, edge) if shared else emit.load(ref, world, edge)


def _spans_worlds(
    operands: Sequence[Array], roles: Sequence[_Role], worlds: int, /
) -> tuple[bool, ...]:
    """Whether each operand has a full world axis rather than a shared one.

    Operands without a world axis, and world-axis operands of leading extent
    one, are read at world zero by every lane.
    """
    return tuple(
        _has_world_axis(operand.shape, role) and operand.shape[0] == worlds
        for operand, role in zip(operands, roles, strict=True)
    )


def _world_reader(
    emit: _Emitter, world: ir.Value, spans: Sequence[bool], /
) -> Callable[[int], ir.Value]:
    """World coordinate of operand ``position`` for this lane."""
    origin = emit.constant(0)

    def at(position: int) -> ir.Value:
        return world if spans[position] else origin

    return at


def _factor_native(
    spatial_inertia: Array,
    transforms: Array,
    subspaces: Array,
    moving: Array,
    parents: Array,
    children: Array,
) -> tuple[Array, ...]:
    operands = (spatial_inertia, transforms, subspaces, moving, parents, children)
    worlds = _world_count([operand.shape for operand in operands], _FACTOR_ROLES)
    bodies = spatial_inertia.shape[1]
    edges = transforms.shape[1]
    dtype = np.dtype(spatial_inertia.dtype)
    if not _admitted(
        floating=(spatial_inertia, transforms, subspaces),
        indices=(parents, children),
        worlds=worlds,
        bodies=bodies,
        edges=edges,
        lanes=worlds,
        columns=1,
        buffers=(worlds * bodies * 36, worlds * edges * 36),
    ):
        return _factor_reference(*operands)
    spans = _spans_worlds(operands, _FACTOR_ROLES, worlds)
    parents_shared = parents.ndim == 1
    children_shared = children.ndim == 1

    def body(emit: _Emitter, world: ir.Value, refs: tuple[ir.Value, ...]) -> None:
        at = _world_reader(emit, world, spans)
        (
            spatial_ref,
            transform_ref,
            subspace_ref,
            moving_ref,
            parent_ref,
            child_ref,
            articulated_ref,
            projected_ref,
            pivot_ref,
            scale_ref,
            reciprocal_ref,
        ) = refs

        @emit.loop(bodies)
        def _(body_index: ir.Value) -> None:
            for row in range(6):
                for column in range(6):
                    emit.store(
                        emit.load(spatial_ref, at(0), body_index, row, column),
                        articulated_ref,
                        world,
                        body_index,
                        row,
                        column,
                    )

        @emit.loop(edges)
        def _(step: ir.Value) -> None:
            edge = emit.reversed_edge(step, edges)
            child = emit.gather_index(
                _index(emit, child_ref, at(5), edge, children_shared), bodies
            )
            inertia = _matrix(emit, articulated_ref, world, child)
            subspace = _vector(emit, subspace_ref, at(2), edge)
            column = [emit.dot(inertia[row], subspace) for row in range(6)]
            pivot = emit.dot(subspace, column)
            absolute = [emit.abs(value) for value in subspace]
            weighted = [
                emit.dot(absolute, [emit.abs(inertia[row][index]) for row in range(6)])
                for index in range(6)
            ]
            scale = emit.dot(weighted, absolute)
            invertible = emit.both(
                emit.both(emit.load(moving_ref, at(3), edge), emit.finite(pivot)),
                emit.nonzero(pivot),
            )
            inverse = emit.select(
                invertible,
                emit.div(emit.one, emit.select(invertible, pivot, emit.one)),
                emit.zero,
            )
            reduced = [
                [
                    emit.sub(
                        inertia[row][index],
                        emit.mul(inverse, emit.mul(column[row], column[index])),
                    )
                    for index in range(6)
                ]
                for row in range(6)
            ]
            transform = _matrix(emit, transform_ref, at(1), edge)
            # (Xᵀ R) X in the grouping of the reference expression.
            left = [
                [
                    emit.dot(
                        [transform[inner][row] for inner in range(6)],
                        [reduced[inner][index] for inner in range(6)],
                    )
                    for index in range(6)
                ]
                for row in range(6)
            ]
            update = [
                [
                    emit.dot(left[row], [transform[inner][index] for inner in range(6)])
                    for index in range(6)
                ]
                for row in range(6)
            ]
            for row in range(6):
                emit.store(column[row], projected_ref, world, edge, row)
            emit.store(pivot, pivot_ref, world, edge)
            emit.store(scale, scale_ref, world, edge)
            emit.store(inverse, reciprocal_ref, world, edge)
            parent, inside = emit.scatter_index(
                _index(emit, parent_ref, at(4), edge, parents_shared), bodies
            )
            with emit.when(inside):
                for row in range(6):
                    for index in range(6):
                        emit.store(
                            emit.add(
                                emit.load(articulated_ref, world, parent, row, index),
                                update[row][index],
                            ),
                            articulated_ref,
                            world,
                            parent,
                            row,
                            index,
                        )

    itemsize = dtype.itemsize
    out_type = (
        jax.ShapeDtypeStruct((worlds, bodies, 6, 6), dtype),
        jax.ShapeDtypeStruct((worlds, edges, 6), dtype),
        jax.ShapeDtypeStruct((worlds, edges), dtype),
        jax.ShapeDtypeStruct((worlds, edges), dtype),
        jax.ShapeDtypeStruct((worlds, edges), dtype),
    )
    return _launch(
        body,
        operands,
        out_type,
        dtype=dtype,
        index_dtype=np.dtype(parents.dtype),
        lanes=worlds,
        flops=worlds * edges * 1100,
        bytes_accessed=worlds
        * (itemsize * (2 * 36 * bodies + edges * (36 + 6 + 36 * 3 + 9)) + edges * 17),
    )


def _solve_native(
    articulated: Array,
    transforms: Array,
    subspaces: Array,
    projected: Array,
    reciprocal: Array,
    parents: Array,
    children: Array,
    edge_effort: Array,
    body_bias: Array,
    edge_convective: Array,
    *,
    bodies: int,
    has_bias: bool,
    has_convective: bool,
) -> tuple[Array, ...]:
    operands = (
        articulated,
        transforms,
        subspaces,
        projected,
        reciprocal,
        parents,
        children,
        edge_effort,
        body_bias,
        edge_convective,
    )
    roles = _solve_roles(has_bias=has_bias, has_convective=has_convective)
    worlds = _world_count([operand.shape for operand in operands], roles)
    edges = transforms.shape[1]
    columns = edge_effort.shape[2]
    dtype = np.dtype(edge_effort.dtype)
    if not _admitted(
        floating=(
            articulated,
            transforms,
            subspaces,
            projected,
            reciprocal,
            edge_effort,
            body_bias,
            edge_convective,
        ),
        indices=(parents, children),
        worlds=worlds,
        bodies=bodies,
        edges=edges,
        lanes=worlds * columns,
        columns=columns,
        buffers=(
            worlds * bodies * 36,
            worlds * edges * 36,
            worlds * bodies * 6 * columns,
            worlds * edges * 6 * columns,
        ),
    ):
        return _solve_reference(
            *operands, bodies=bodies, has_bias=has_bias, has_convective=has_convective
        )
    spans = _spans_worlds(operands, roles, worlds)
    parents_shared = parents.ndim == 1
    children_shared = children.ndim == 1
    kernel_operands = (
        ((articulated,) if has_convective else ())
        + (transforms, subspaces, projected, reciprocal, parents, children, edge_effort)
        + ((body_bias,) if has_bias else ())
        + ((edge_convective,) if has_convective else ())
    )

    def body(emit: _Emitter, lane: ir.Value, refs: tuple[ir.Value, ...]) -> None:
        iterator = iter(refs)
        articulated_ref = next(iterator) if has_convective else None
        transform_ref = next(iterator)
        subspace_ref = next(iterator)
        projected_ref = next(iterator)
        reciprocal_ref = next(iterator)
        parent_ref = next(iterator)
        child_ref = next(iterator)
        effort_ref = next(iterator)
        bias_ref = next(iterator) if has_bias else None
        convective_ref = next(iterator) if has_convective else None
        output_ref = next(iterator)
        workspace_ref = next(iterator)
        world, column = emit.split(lane, columns)
        at = _world_reader(emit, world, spans)

        @emit.loop(bodies)
        def _(body_index: ir.Value) -> None:
            for component in range(6):
                emit.store(
                    emit.zero
                    if bias_ref is None
                    else emit.load(bias_ref, at(8), body_index, component, column),
                    workspace_ref,
                    world,
                    body_index,
                    component,
                    column,
                )

        @emit.loop(edges)
        def _(step: ir.Value) -> None:
            edge = emit.reversed_edge(step, edges)
            child = emit.gather_index(
                _index(emit, child_ref, at(6), edge, children_shared), bodies
            )
            child_bias = _column(emit, workspace_ref, world, child, column=column)
            subspace = _vector(emit, subspace_ref, at(2), edge)
            projection = _vector(emit, projected_ref, at(3), edge)
            inverse = emit.load(reciprocal_ref, at(4), edge)
            effort = emit.sub(
                emit.load(effort_ref, at(7), edge, column),
                emit.dot(subspace, child_bias),
            )
            scaled = emit.mul(inverse, effort)
            reduced = [
                emit.add(child_bias[row], emit.mul(projection[row], scaled))
                for row in range(6)
            ]
            if articulated_ref is not None and convective_ref is not None:
                inertia = _matrix(emit, articulated_ref, at(0), child)
                convective = _column(emit, convective_ref, at(9), edge, column=column)
                reduced = [
                    emit.add(
                        reduced[row],
                        emit.dot(
                            [
                                emit.sub(
                                    inertia[row][index],
                                    emit.mul(
                                        inverse,
                                        emit.mul(projection[row], projection[index]),
                                    ),
                                )
                                for index in range(6)
                            ],
                            convective,
                        ),
                    )
                    for row in range(6)
                ]
            update = _transposed_action(
                emit, _matrix(emit, transform_ref, at(1), edge), reduced
            )
            emit.store(effort, output_ref, world, edge, column)
            parent, inside = emit.scatter_index(
                _index(emit, parent_ref, at(5), edge, parents_shared), bodies
            )
            with emit.when(inside):
                for component in range(6):
                    emit.store(
                        emit.add(
                            emit.load(workspace_ref, world, parent, component, column),
                            update[component],
                        ),
                        workspace_ref,
                        world,
                        parent,
                        component,
                        column,
                    )

        # The reverse bias is dead; the workspace now carries accelerations.
        @emit.loop(bodies)
        def _(body_index: ir.Value) -> None:
            for component in range(6):
                emit.store(emit.zero, workspace_ref, world, body_index, component, column)

        @emit.loop(edges)
        def _(edge: ir.Value) -> None:
            parent = emit.gather_index(
                _index(emit, parent_ref, at(5), edge, parents_shared), bodies
            )
            acceleration = _column(emit, workspace_ref, world, parent, column=column)
            transform = _matrix(emit, transform_ref, at(1), edge)
            base = [emit.dot(transform[row], acceleration) for row in range(6)]
            if convective_ref is not None:
                base = [
                    emit.add(
                        base[row], emit.load(convective_ref, at(9), edge, row, column)
                    )
                    for row in range(6)
                ]
            rate = emit.mul(
                emit.load(reciprocal_ref, at(4), edge),
                emit.sub(
                    emit.load(output_ref, world, edge, column),
                    emit.dot(_vector(emit, projected_ref, at(3), edge), base),
                ),
            )
            subspace = _vector(emit, subspace_ref, at(2), edge)
            child, inside = emit.scatter_index(
                _index(emit, child_ref, at(6), edge, children_shared), bodies
            )
            with emit.when(inside):
                for component in range(6):
                    emit.store(
                        emit.add(base[component], emit.mul(subspace[component], rate)),
                        workspace_ref,
                        world,
                        child,
                        component,
                        column,
                    )
            emit.store(rate, output_ref, world, edge, column)

    itemsize = dtype.itemsize
    lanes = worlds * columns
    out_type = (
        jax.ShapeDtypeStruct((worlds, edges, columns), dtype),
        jax.ShapeDtypeStruct((worlds, bodies, 6, columns), dtype),
    )
    convective_terms = int(has_convective)
    edge_bytes = (
        itemsize * (2 * 36 + 4 * 6 + 2 + 3 + (36 + 12) * convective_terms) + 2 * 8
    )
    (acceleration, _) = _launch(
        body,
        kernel_operands,
        out_type,
        dtype=dtype,
        index_dtype=np.dtype(parents.dtype),
        lanes=lanes,
        flops=lanes * edges * (150 + 170 * convective_terms),
        bytes_accessed=lanes
        * (edges * edge_bytes + bodies * itemsize * 6 * (3 + int(has_bias))),
    )
    return (acceleration,)


def _reduce_native(
    transforms: Array,
    subspaces: Array,
    parents: Array,
    children: Array,
    body_force: Array,
) -> tuple[Array, ...]:
    operands = (transforms, subspaces, parents, children, body_force)
    worlds = _world_count([operand.shape for operand in operands], _REDUCE_ROLES)
    _, bodies, _, columns = body_force.shape
    edges = transforms.shape[1]
    dtype = np.dtype(body_force.dtype)
    if not _admitted(
        floating=(transforms, subspaces, body_force),
        indices=(parents, children),
        worlds=worlds,
        bodies=bodies,
        edges=edges,
        lanes=worlds * columns,
        columns=columns,
        buffers=(worlds * edges * 36, worlds * bodies * 6 * columns),
    ):
        return _forces_reference(*operands)
    spans = _spans_worlds(operands, _REDUCE_ROLES, worlds)
    parents_shared = parents.ndim == 1
    children_shared = children.ndim == 1

    def body(emit: _Emitter, lane: ir.Value, refs: tuple[ir.Value, ...]) -> None:
        (
            transform_ref,
            subspace_ref,
            parent_ref,
            child_ref,
            force_ref,
            output_ref,
            workspace_ref,
        ) = refs
        world, column = emit.split(lane, columns)
        at = _world_reader(emit, world, spans)

        @emit.loop(bodies)
        def _(body_index: ir.Value) -> None:
            for component in range(6):
                emit.store(
                    emit.load(force_ref, at(4), body_index, component, column),
                    workspace_ref,
                    world,
                    body_index,
                    component,
                    column,
                )

        @emit.loop(edges)
        def _(step: ir.Value) -> None:
            edge = emit.reversed_edge(step, edges)
            child = emit.gather_index(
                _index(emit, child_ref, at(3), edge, children_shared), bodies
            )
            child_force = _column(emit, workspace_ref, world, child, column=column)
            emit.store(
                emit.dot(_vector(emit, subspace_ref, at(1), edge), child_force),
                output_ref,
                world,
                edge,
                column,
            )
            update = _transposed_action(
                emit, _matrix(emit, transform_ref, at(0), edge), child_force
            )
            parent, inside = emit.scatter_index(
                _index(emit, parent_ref, at(2), edge, parents_shared), bodies
            )
            with emit.when(inside):
                for component in range(6):
                    emit.store(
                        emit.add(
                            emit.load(workspace_ref, world, parent, component, column),
                            update[component],
                        ),
                        workspace_ref,
                        world,
                        parent,
                        component,
                        column,
                    )

    itemsize = dtype.itemsize
    lanes = worlds * columns
    out_type = (
        jax.ShapeDtypeStruct((worlds, edges, columns), dtype),
        jax.ShapeDtypeStruct((worlds, bodies, 6, columns), dtype),
    )
    (effort, _) = _launch(
        body,
        operands,
        out_type,
        dtype=dtype,
        index_dtype=np.dtype(parents.dtype),
        lanes=lanes,
        flops=lanes * edges * 96,
        bytes_accessed=lanes
        * (edges * (itemsize * (36 + 6 + 19) + 16) + bodies * itemsize * 6 * 2),
    )
    return (effort,)


# ---------------------------------------------------------------------------
# Primitives, derivatives, transposes and batching.
# ---------------------------------------------------------------------------


def _primitive(
    name: str,
    abstract: Callable[..., tuple[ShapedArray, ...]],
    reference: Callable[..., tuple[Array, ...]],
    native: Callable[..., tuple[Array, ...]],
    /,
) -> core.Primitive:
    primitive = core.Primitive(name)
    primitive.multiple_results = True
    primitive.def_impl(partial(dispatch.apply_primitive, primitive))
    primitive.def_abstract_eval(abstract)
    mlir.register_lowering(primitive, mlir.lower_fun(reference, multiple_results=True))
    mlir.register_lowering(
        primitive, partial(lower_mosaic_or_reference, native, reference), platform="cuda"
    )
    return primitive


_factor_p = _primitive(
    "phydrax_articulated_inertia_factor",
    _factor_abstract,
    _factor_reference,
    _factor_native,
)
_solve_p = _primitive(
    "phydrax_articulated_factor_solve",
    _solve_abstract,
    _solve_reference,
    _solve_native,
)
_forces_p = _primitive(
    "phydrax_articulated_force_reduction",
    _reduce_abstract,
    _forces_reference,
    _reduce_native,
)


def _reference_jvp(
    primitive: core.Primitive,
    reference: Callable[..., tuple[Array, ...]],
    roles: Sequence[_Role],
    primals: Sequence[Array],
    tangents: Sequence[Tangent],
    /,
    **params: object,
) -> tuple[tuple[Array, ...], tuple[Tangent, ...]]:
    """Native primal; tangents always belong to the owning JAX reference."""

    result = tuple(primitive.bind(*primals, **params))
    active = tuple(
        position
        for position, (primal, tangent, role) in enumerate(
            zip(primals, tangents, roles, strict=True)
        )
        if role is _Role.WORLD
        and jnp.issubdtype(primal.dtype, jnp.floating)
        and not isinstance(tangent, ad.Zero)
    )
    if not active:
        return result, tuple(ad.Zero(jax.typeof(value)) for value in result)

    def evaluate(*values: Array) -> tuple[Array, ...]:
        arguments = list(primals)
        for position, value in zip(active, values, strict=True):
            arguments[position] = value
        return reference(*arguments, **params)

    _, tangent = jax.jvp(
        evaluate,
        tuple(primals[position] for position in active),
        tuple(tangents[position] for position in active),
    )
    return result, tuple(tangent)


def _factor_jvp(
    primals: tuple[Array, ...], tangents: tuple[Tangent, ...]
) -> tuple[tuple[Array, ...], tuple[Tangent, ...]]:
    return _reference_jvp(_factor_p, _factor_reference, _FACTOR_ROLES, primals, tangents)


def _solve_jvp(
    primals: tuple[Array, ...],
    tangents: tuple[Tangent, ...],
    *,
    bodies: int,
    has_bias: bool,
    has_convective: bool,
) -> tuple[tuple[Array, ...], tuple[Tangent, ...]]:
    return _reference_jvp(
        _solve_p,
        _solve_reference,
        _solve_roles(has_bias=has_bias, has_convective=has_convective),
        primals,
        tangents,
        bodies=bodies,
        has_bias=has_bias,
        has_convective=has_convective,
    )


def _reduce_jvp(
    primals: tuple[Array, ...], tangents: tuple[Tangent, ...]
) -> tuple[tuple[Array, ...], tuple[Tangent, ...]]:
    return _reference_jvp(_forces_p, _forces_reference, _REDUCE_ROLES, primals, tangents)


def _reference_transpose(
    reference: Callable[..., tuple[Array, ...]],
    linear: frozenset[int],
    cotangents: Sequence[Tangent],
    operands: Sequence[Primal],
    /,
    **params: int | bool,
) -> tuple[Tangent | None, ...]:
    """Transpose the reference in its linear operands, coefficients held fixed.

    Defined linear operands contribute an affine offset, not part of the
    linear map, and are replaced by zeros before transposition.
    """

    nonlinear = [
        position
        for position, value in enumerate(operands)
        if isinstance(value, ad.UndefinedPrimal) and position not in linear
    ]
    if nonlinear:
        raise ValueError(
            "Articulated sweeps are linear only in their right-hand-side operands; "
            "factor coefficients and topology cannot be transposed."
        )
    (cotangent,) = cotangents
    if isinstance(cotangent, ad.Zero):
        return tuple(
            ad.Zero(value.aval) if isinstance(value, ad.UndefinedPrimal) else None
            for value in operands
        )

    def evaluate(*values: Array) -> Array:
        supplied = iter(values)
        arguments: list[Array] = []
        for position, value in enumerate(operands):
            if isinstance(value, ad.UndefinedPrimal):
                arguments.append(next(supplied))
            elif position in linear:
                arguments.append(jnp.zeros(value.shape, value.dtype))
            else:
                arguments.append(value)
        return reference(*arguments, **params)[0]

    templates = tuple(
        jax.ShapeDtypeStruct(value.aval.shape, value.aval.dtype)
        for value in operands
        if isinstance(value, ad.UndefinedPrimal)
    )
    pullback = iter(jax.linear_transpose(evaluate, *templates)(cotangent))
    return tuple(
        next(pullback) if isinstance(value, ad.UndefinedPrimal) else None
        for value in operands
    )


def _solve_transpose(
    cotangents: Sequence[Tangent],
    *operands: Primal,
    bodies: int,
    has_bias: bool,
    has_convective: bool,
) -> tuple[Tangent | None, ...]:
    linear = frozenset(
        (7,) + ((8,) if has_bias else ()) + ((9,) if has_convective else ())
    )
    return _reference_transpose(
        _solve_reference,
        linear,
        cotangents,
        operands,
        bodies=bodies,
        has_bias=has_bias,
        has_convective=has_convective,
    )


def _reduce_transpose(
    cotangents: Sequence[Tangent], *operands: Primal
) -> tuple[Tangent | None, ...]:
    return _reference_transpose(_forces_reference, frozenset((4,)), cotangents, operands)


def _world_batch(
    primitive: core.Primitive,
    reference: Callable[..., tuple[Array, ...]],
    roles: Sequence[_Role],
    operands: Sequence[Array],
    axes: Sequence[int | None],
    /,
    **params: object,
) -> tuple[tuple[Array, ...], tuple[int | None, ...]]:
    """Fold a mapped axis into the world axis and bind the primitive once.

    The folded world count is ``size * W``. Unmapped numerical operands of
    world extent one stay shared (never copied per world), mapped operands of
    full world extent are reshaped in place, and topology indices are
    broadcast per world when any topology is mapped or per-world. Partially
    shared nesting that a single shared/full extent cannot express — an
    unmapped operand already spanning ``W > 1`` worlds, a mapped operand of
    extent one beside ``W > 1`` worlds, or a mapped placeholder — maps the
    canonical reference instead of copying numerical data.
    """

    if all(axis is None for axis in axes):
        result = tuple(primitive.bind(*operands, **params))
        return result, (None,) * len(result)

    def mapped_reference() -> tuple[tuple[Array, ...], tuple[int | None, ...]]:
        result = jax.vmap(partial(reference, **params), in_axes=tuple(axes))(*operands)
        return tuple(result), (0,) * len(result)

    size = next(
        operand.shape[axis]
        for operand, axis in zip(operands, axes, strict=True)
        if axis is not None
    )
    logical = [
        operand.shape
        if axis is None
        else operand.shape[:axis] + operand.shape[axis + 1 :]
        for operand, axis in zip(operands, axes, strict=True)
    ]
    worlds = _world_count(logical, roles)
    folded: list[Array] = []
    for operand, axis, role, shape in zip(operands, axes, roles, logical, strict=True):
        if role is _Role.PLACEHOLDER:
            if axis is not None:
                return mapped_reference()
            folded.append(operand)
        elif role is _Role.INDEX:
            if axis is None and (len(shape) == 1 or shape[0] == 1):
                # Shared topology remains valid for every folded world.
                folded.append(operand)
                continue
            topology = (
                jnp.broadcast_to(operand, (size,) + operand.shape)
                if axis is None
                else jnp.moveaxis(operand, axis, 0)
            )
            if topology.ndim == 2:
                topology = topology[:, None, :]
            # Integer topology is small; numerical operands are never tiled.
            topology = jnp.broadcast_to(topology, (size, worlds, topology.shape[-1]))
            folded.append(topology.reshape((size * worlds, topology.shape[-1])))
        elif axis is None:
            if shape[0] != 1:
                return mapped_reference()
            folded.append(operand)
        else:
            if shape[0] != worlds:
                return mapped_reference()
            numerical = jnp.moveaxis(operand, axis, 0)
            folded.append(numerical.reshape((size * worlds,) + numerical.shape[2:]))
    result = tuple(primitive.bind(*folded, **params))
    return (
        tuple(value.reshape((size, worlds) + value.shape[1:]) for value in result),
        (0,) * len(result),
    )


def _factor_batch(
    operands: tuple[Array, ...], axes: tuple[int | None, ...]
) -> tuple[tuple[Array, ...], tuple[int | None, ...]]:
    return _world_batch(_factor_p, _factor_reference, _FACTOR_ROLES, operands, axes)


def _solve_batch(
    operands: tuple[Array, ...],
    axes: tuple[int | None, ...],
    *,
    bodies: int,
    has_bias: bool,
    has_convective: bool,
) -> tuple[tuple[Array, ...], tuple[int | None, ...]]:
    return _world_batch(
        _solve_p,
        _solve_reference,
        _solve_roles(has_bias=has_bias, has_convective=has_convective),
        operands,
        axes,
        bodies=bodies,
        has_bias=has_bias,
        has_convective=has_convective,
    )


def _reduce_batch(
    operands: tuple[Array, ...], axes: tuple[int | None, ...]
) -> tuple[tuple[Array, ...], tuple[int | None, ...]]:
    return _world_batch(_forces_p, _forces_reference, _REDUCE_ROLES, operands, axes)


ad.primitive_jvps[_factor_p] = _factor_jvp
batching.primitive_batchers[_factor_p] = _factor_batch
ad.primitive_jvps[_solve_p] = _solve_jvp
ad.primitive_transposes[_solve_p] = _solve_transpose
batching.primitive_batchers[_solve_p] = _solve_batch
ad.primitive_jvps[_forces_p] = _reduce_jvp
ad.primitive_transposes[_forces_p] = _reduce_transpose
batching.primitive_batchers[_forces_p] = _reduce_batch


# ---------------------------------------------------------------------------
# Single-world interfaces.
# ---------------------------------------------------------------------------


def _array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be a real floating array.")
    return array


def _topology(parents: ArrayLike, children: ArrayLike, /) -> tuple[Array, Array]:
    parent = jnp.asarray(parents)
    child = jnp.asarray(children)
    if not jnp.issubdtype(parent.dtype, jnp.integer) or not jnp.issubdtype(
        child.dtype, jnp.integer
    ):
        raise TypeError("parents and children must be integer arrays.")
    if parent.ndim != 1 or parent.shape != child.shape:
        raise ValueError("parents and children must be one-dimensional edge arrays.")
    return parent, child


def factor_articulated_inertia(
    spatial_inertia: ArrayLike,
    transforms: ArrayLike,
    subspaces: ArrayLike,
    moving: ArrayLike,
    parents: ArrayLike,
    children: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    """Factor one tree's spatial inertias by the reverse articulated recurrence.

    Shapes: ``spatial_inertia`` ``(C, 6, 6)``, ``transforms`` ``(E, 6, 6)``,
    ``subspaces`` ``(E, 6)``, boolean ``moving`` ``(E,)`` and integer
    ``parents``/``children`` ``(E,)``. Returns final articulated inertias
    ``(C, 6, 6)``, projected columns ``(E, 6)``, pivots, pivot scales and
    reciprocal pivots ``(E,)``. A pivot is inverted only on a moving edge when
    finite and nonzero; otherwise its reciprocal is zero and the complete child
    inertia is transmitted. Floating operands may differ in dtype: the inertia
    carries the spatial-inertia dtype, edge results the transform dtype, and
    every step computes in the promoted type of its operands.
    """

    spatial = _array(spatial_inertia, "spatial_inertia")
    transform = _array(transforms, "transforms")
    subspace = _array(subspaces, "subspaces")
    mask = jnp.asarray(moving)
    if mask.dtype != jnp.bool_:
        raise TypeError("moving must be a boolean edge mask.")
    parent, child = _topology(parents, children)
    articulated, projected, pivots, scales, reciprocal = _factor_p.bind(
        spatial[None], transform[None], subspace[None], mask[None], parent, child
    )
    return articulated[0], projected[0], pivots[0], scales[0], reciprocal[0]


def solve_articulated_factor(
    articulated: ArrayLike,
    transforms: ArrayLike,
    subspaces: ArrayLike,
    projected: ArrayLike,
    reciprocal: ArrayLike,
    parents: ArrayLike,
    children: ArrayLike,
    edge_effort: ArrayLike,
    body_bias: ArrayLike | None,
    edge_convective: ArrayLike | None,
    /,
) -> Array:
    """Substitute ``(E, K)`` edge-effort columns through one articulated factor.

    One operation owns the reverse and forward substitutions. ``body_bias``
    ``(C, 6, K)`` seeds the reverse body forces and ``edge_convective``
    ``(E, 6, K)`` adds edge convective accelerations; ``None`` omits either
    term. Returns edge accelerations ``(E, K)``; fixed edges return zero rates
    unless a nonfinite value reaches them. The caller owns DOF gather/scatter.
    Scalar efforts, accelerations and the result carry the ``articulated``
    dtype; a supplied ``body_bias`` keeps its own carry dtype.
    """

    inertia = _array(articulated, "articulated")
    dtype = np.dtype(inertia.dtype)
    if inertia.ndim != 3:
        raise ValueError("articulated must have (body, 6, 6) shape.")
    transform = _array(transforms, "transforms")
    subspace = _array(subspaces, "subspaces")
    projection = _array(projected, "projected")
    inverse = _array(reciprocal, "reciprocal")
    effort = _array(edge_effort, "edge_effort")
    parent, child = _topology(parents, children)
    if effort.ndim != 2:
        raise ValueError("edge_effort must have (edge, column) shape.")
    placeholder = jnp.zeros((), dtype=dtype)
    has_bias = body_bias is not None
    has_convective = edge_convective is not None
    bias = placeholder
    if body_bias is not None:
        bias = _array(body_bias, "body_bias")[None]
    convective = placeholder
    if edge_convective is not None:
        convective = _array(edge_convective, "edge_convective")[None]
    (acceleration,) = _solve_p.bind(
        inertia[None] if has_convective else placeholder,
        transform[None],
        subspace[None],
        projection[None],
        inverse[None],
        parent,
        child,
        effort[None],
        bias,
        convective,
        bodies=inertia.shape[0],
        has_bias=has_bias,
        has_convective=has_convective,
    )
    return acceleration[0]


def reduce_articulated_forces(
    transforms: ArrayLike,
    subspaces: ArrayLike,
    parents: ArrayLike,
    children: ArrayLike,
    body_force: ArrayLike,
    /,
) -> Array:
    """Accumulate ``(C, 6, K)`` body force columns into ``(E, K)`` edge efforts.

    One reverse traversal computes ``S_eᵀ f_child`` before transmitting
    ``X_eᵀ f_child`` to the parent. Columns are independent. The force carry
    keeps the ``body_force`` dtype; efforts are returned in the promoted type
    of subspace and force.
    """

    force = _array(body_force, "body_force")
    transform = _array(transforms, "transforms")
    subspace = _array(subspaces, "subspaces")
    if force.ndim != 3:
        raise ValueError("body_force must have (body, 6, column) shape.")
    parent, child = _topology(parents, children)
    (effort,) = _forces_p.bind(
        transform[None], subspace[None], parent, child, force[None]
    )
    return effort[0]


__all__ = [
    "factor_articulated_inertia",
    "reduce_articulated_forces",
    "solve_articulated_factor",
]
