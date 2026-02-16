#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import operator
from collections.abc import Callable, Mapping, Sequence
from math import comb
from typing import Any, Literal

import coordax as cx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key, PyTree

from .._callable import _ensure_special_kwonly_args
from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from ._domain import _AbstractDomain
from ._structure import CoordSeparableBatch, Points, PointsBatch


def _first_field_leaf(tree: PyTree[Any]) -> cx.Field:
    leaves = jax.tree_util.tree_leaves(tree, is_leaf=lambda x: isinstance(x, cx.Field))
    for leaf in leaves:
        if isinstance(leaf, cx.Field):
            return leaf
    raise ValueError("Expected at least one coordax.Field leaf.")


def _unwrap_fields_to_data(tree: PyTree[Any]) -> PyTree[Any]:
    return jax.tree_util.tree_map(
        lambda x: x.data if isinstance(x, cx.Field) else x,
        tree,
        is_leaf=lambda x: isinstance(x, cx.Field),
    )


def _axis_size(points: Mapping[str, PyTree[Any]], axis: str, /) -> int:
    leaves = jax.tree_util.tree_leaves(points, is_leaf=lambda x: isinstance(x, cx.Field))
    for leaf in leaves:
        if not isinstance(leaf, cx.Field):
            continue
        if axis in leaf.named_shape:
            return int(leaf.named_shape[axis])
    raise ValueError(f"Cannot infer size for axis {axis!r} from points.")


def _reorder_named_axes(field: cx.Field, axis_order: tuple[str, ...]) -> cx.Field:
    dims = field.dims
    if not dims:
        return field
    named_dims = [d for d in dims if d is not None]
    if not named_dims:
        return field
    desired = [d for d in axis_order if d in named_dims]
    remaining = [d for d in named_dims if d not in axis_order]
    target_named = desired + remaining

    index_by_dim = {d: i for i, d in enumerate(dims) if d is not None}
    perm = [index_by_dim[d] for d in target_named]
    perm.extend([i for i, d in enumerate(dims) if d is None])
    if perm == list(range(len(dims))):
        return field
    data = jnp.transpose(jnp.asarray(field.data), perm)
    new_dims = tuple(dims[i] for i in perm)
    return cx.Field(data, dims=new_dims)


def _singleton_axis_for_label(structure: Any, label: str, /) -> str | None:
    axis_names = structure.axis_names
    if axis_names is None:
        return None
    for block, axis in zip(structure.blocks, axis_names, strict=True):
        if label in block:
            if len(block) != 1:
                return None
            return axis
    return None


def _dedupe_axes(axis_names: Sequence[str], /) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for axis in axis_names:
        if axis in seen:
            continue
        seen.add(axis)
        out.append(axis)
    return tuple(out)


def _as_blockwise_arg(value: Any, /) -> Array | tuple[Array, ...] | None:
    if isinstance(value, cx.Field):
        return jnp.asarray(value.data)
    if isinstance(value, tuple):
        out: list[Array] = []
        for item in value:
            if not isinstance(item, cx.Field):
                return None
            out.append(jnp.asarray(item.data))
        return tuple(out)
    return None


def _rank1_leading_broadcast_op(
    op: Callable[[Any, Any], Any],
    left: Any,
    right: Any,
    /,
) -> Any:
    left_arr = jnp.asarray(left)
    right_arr = jnp.asarray(right)

    if left_arr.ndim == 1 and right_arr.ndim == 1:
        if int(left_arr.shape[0]) != int(right_arr.shape[0]):
            return op(left_arr[:, None], right_arr[None, :])

    if left_arr.ndim == 1 and right_arr.ndim >= 2:
        if int(left_arr.shape[0]) == int(right_arr.shape[0]):
            left_b = left_arr.reshape(
                (int(left_arr.shape[0]),) + (1,) * (right_arr.ndim - 1)
            )
            return op(left_b, right_arr)

    if right_arr.ndim == 1 and left_arr.ndim >= 2:
        if int(right_arr.shape[0]) == int(left_arr.shape[0]):
            right_b = right_arr.reshape(
                (int(right_arr.shape[0]),) + (1,) * (left_arr.ndim - 1)
            )
            return op(left_arr, right_b)

    return op(left, right)


class _ConstCallable(StrictModule):
    value: jax.Array

    def __init__(self, value: ArrayLike | None):
        if value is None:
            raise TypeError("DomainFunction constants must be array-like, not None.")
        self.value = jnp.asarray(value)

    def __call__(self, *args, key=None, **kwargs):
        del args, key, kwargs
        return self.value


class _UnaryCallable(StrictModule):
    func: Callable
    op: Callable[[Any], Any]

    def __init__(self, func: Callable, op: Callable[[Any], Any]):
        self.func = func
        self.op = op

    def __call__(self, *args, key=None, **kwargs):
        return self.op(self.func(*args, key=key, **kwargs))


class _SwapAxesCallable(StrictModule):
    func: Callable
    axis1: int
    axis2: int

    def __init__(self, func: Callable, axis1: int, axis2: int):
        self.func = func
        self.axis1 = int(axis1)
        self.axis2 = int(axis2)

    def __call__(self, *args, key=None, **kwargs):
        return jnp.swapaxes(self.func(*args, key=key, **kwargs), self.axis1, self.axis2)


class _BinaryCallable(StrictModule):
    a: "DomainFunction"
    b: "DomainFunction"
    op: Callable[[Any, Any], Any]
    a_pos: tuple[int, ...]
    b_pos: tuple[int, ...]
    reverse: bool

    def __init__(
        self,
        *,
        a: "DomainFunction",
        b: "DomainFunction",
        op: Callable[[Any, Any], Any],
        a_pos: tuple[int, ...],
        b_pos: tuple[int, ...],
        reverse: bool,
    ):
        self.a = a
        self.b = b
        self.op = op
        self.a_pos = tuple(int(i) for i in a_pos)
        self.b_pos = tuple(int(i) for i in b_pos)
        self.reverse = bool(reverse)

    def __call__(self, *args, key=None, **kwargs):
        a_args = [args[i] for i in self.a_pos]
        b_args = [args[i] for i in self.b_pos]
        if self.reverse:
            left = self.b.func(*b_args, key=key, **kwargs)
            right = self.a.func(*a_args, key=key, **kwargs)
        else:
            left = self.a.func(*a_args, key=key, **kwargs)
            right = self.b.func(*b_args, key=key, **kwargs)

        try:
            return self.op(left, right)
        except (TypeError, ValueError):
            if self.op in (
                operator.add,
                operator.sub,
                operator.mul,
                operator.truediv,
            ):
                return _rank1_leading_broadcast_op(self.op, left, right)
            raise


_DERIVATIVE_HOOK_KEY = "_optimized_derivative_hook"


def _drop_derivative_hook_metadata(
    metadata: frozendict[str, Any],
    /,
) -> frozendict[str, Any]:
    if _DERIVATIVE_HOOK_KEY not in metadata:
        return metadata
    return frozendict({k: v for k, v in metadata.items() if k != _DERIVATIVE_HOOK_KEY})


def _transpose_lifted_hook(
    hook: Callable[..., "DomainFunction | None"],
    /,
) -> Callable[..., "DomainFunction | None"]:
    def _lifted(
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: Literal["reverse", "forward"],
        backend: Literal["ad", "jet", "fd", "basis"],
        basis: Literal["poly", "fourier", "sine", "cosine"],
        periodic: bool,
    ) -> "DomainFunction | None":
        out = hook(
            var=var,
            axis=axis,
            order=order,
            mode=mode,
            backend=backend,
            basis=basis,
            periodic=periodic,
        )
        if out is None:
            return None
        return out.T

    return _lifted


def _transpose_hook_metadata(
    metadata: frozendict[str, Any],
    /,
) -> frozendict[str, Any]:
    hook = metadata.get(_DERIVATIVE_HOOK_KEY)
    clean = _drop_derivative_hook_metadata(metadata)
    if not callable(hook):
        return clean
    merged = dict(clean)
    merged[_DERIVATIVE_HOOK_KEY] = _transpose_lifted_hook(hook)
    return frozendict(merged)


def _compose_binary_derivative_hook(
    op: Callable[[Any, Any], Any],
    /,
    *,
    left: "DomainFunction",
    right: "DomainFunction",
) -> Callable[..., "DomainFunction | None"] | None:
    if op not in (
        operator.add,
        operator.sub,
        operator.mul,
        operator.matmul,
        operator.truediv,
    ):
        return None

    def _hook(
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: Literal["reverse", "forward"],
        backend: Literal["ad", "jet", "fd", "basis"],
        basis: Literal["poly", "fourier", "sine", "cosine"],
        periodic: bool,
    ) -> "DomainFunction | None":
        if backend not in ("ad", "jet"):
            return None

        from ..operators.differential._domain_ops import partial_n
        from ..operators.differential._hooks import nth_product_rule, nth_quotient_rule

        n = int(order)
        if n < 0:
            return None

        def _derive(fn: DomainFunction, k: int, /) -> DomainFunction:
            return partial_n(
                fn,
                var=var,
                axis=axis,
                order=int(k),
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            )

        if op is operator.add:
            return _derive(left, n) + _derive(right, n)

        if op is operator.sub:
            return _derive(left, n) - _derive(right, n)

        if op is operator.mul:
            return nth_product_rule(left, right, var=var, order=n, derive=_derive)

        if op is operator.truediv:
            return nth_quotient_rule(left, right, var=var, order=n, derive=_derive)

        if op is operator.matmul:
            if n == 0:
                return left @ right
            out = DomainFunction(domain=left.domain, deps=(), func=0.0, metadata={})
            for k in range(n + 1):
                out = out + float(comb(n, k)) * (_derive(left, k) @ _derive(right, n - k))
            return out

        return None

    return _hook


class DomainFunction(StrictModule):
    r"""A callable with explicit domain and coordinate semantics.

    A `DomainFunction` represents a mathematical map

    $$
    u:\Omega \to \mathbb{R}^m,
    $$

    where the domain $\Omega$ is a `phydrax.domain` object carrying *labeled* factors
    (e.g. a space-time domain $\Omega = \Omega_x \times \Omega_t$ with labels `"x"` and
    `"t"`). The tuple `deps` specifies which labels the function actually depends on.

    Phydrax operators (gradients, divergences, integrals, etc.) act on `DomainFunction`
    objects, and constraints evaluate them on sampled batches.

    **Algebra**

    `DomainFunction` implements pointwise arithmetic. For example, for compatible
    domains,

    $$
    (u+v)(z)=u(z)+v(z),\qquad (uv)(z)=u(z)v(z),
    $$

    and when combining functions with different domains, the domains are joined and
    both functions are promoted to the joined domain.

    **Notes**

    - If `func` is array-like, it is treated as a constant function on $\Omega$.
    - If `func` is callable, Phydrax passes randomness through a keyword-only `key`
      argument (when provided by downstream sampling/solvers).
    - Evaluation returns a `coordax.Field` whose named axes are inferred from the
      sampling structure (paired blocks and/or coord-separable axes).
    """

    domain: _AbstractDomain
    deps: tuple[str, ...]
    func: Callable
    metadata: frozendict[str, Any]

    def __init__(
        self,
        *,
        domain: _AbstractDomain,
        deps: Sequence[str],
        func: Callable | ArrayLike | None,
        metadata: Mapping[str, Any] | None = None,
    ):
        self.domain = domain
        self.deps = tuple(deps)

        if callable(func):
            self.func = _ensure_special_kwonly_args(func)
        else:
            self.func = _ConstCallable(func)

        self.metadata = frozendict({} if metadata is None else metadata)

    def depends_on(self, var: str, /) -> bool:
        """Return whether this function depends on the labeled variable `var`."""
        return var in self.deps

    def promote(self, new_domain: _AbstractDomain, /) -> "DomainFunction":
        r"""View this function as defined on a larger domain.

        If $\Omega\subseteq\Omega'$, then promotion constructs $u':\Omega'\to\mathbb{R}^m$
        by ignoring the extra coordinates:

        $$
        u'(z) = u(z|_{\Omega}).
        $$

        In practice this means that all labels of the current domain must appear in
        `new_domain.labels`, and the underlying callable is reused unchanged.
        """
        for lbl in self.domain.labels:
            if lbl not in new_domain.labels:
                raise ValueError(
                    f"Cannot promote from domain {self.domain.labels} to {new_domain.labels}."
                )
        return DomainFunction(
            domain=new_domain, deps=self.deps, func=self.func, metadata=self.metadata
        )

    def with_metadata(self, **metadata: Any) -> "DomainFunction":
        """Return a copy with `metadata` merged into the existing metadata."""
        merged = dict(self.metadata)
        merged.update(metadata)
        return DomainFunction(
            domain=self.domain, deps=self.deps, func=self.func, metadata=merged
        )

    def _binary_op(
        self,
        other: "DomainFunction | ArrayLike | None",
        op: Callable[[Any, Any], Any],
        /,
        *,
        reverse: bool = False,
    ) -> "DomainFunction":
        if isinstance(other, DomainFunction):
            other_fn = other
        else:
            other_fn = DomainFunction(
                domain=self.domain, deps=(), func=other, metadata={}
            )

        if self.domain.labels == other_fn.domain.labels:
            # Assume different domains carry different labels to avoid label collisions.
            joined = self.domain
        else:
            joined = self.domain.join(other_fn.domain)
        a = self.promote(joined)
        b = other_fn.promote(joined)

        deps = tuple(lbl for lbl in joined.labels if (lbl in a.deps) or (lbl in b.deps))
        idx = {lbl: i for i, lbl in enumerate(deps)}
        a_pos = tuple(idx[lbl] for lbl in a.deps)
        b_pos = tuple(idx[lbl] for lbl in b.deps)

        a_user_meta = _drop_derivative_hook_metadata(a.metadata)
        b_user_meta = _drop_derivative_hook_metadata(b.metadata)

        meta: frozendict[str, Any]
        can_attach_hook = True
        if not b_user_meta:
            meta = a_user_meta
        elif not a_user_meta:
            meta = b_user_meta
        elif a_user_meta == b_user_meta:
            meta = a_user_meta
        else:
            meta = frozendict({})
            can_attach_hook = False

        left = b if reverse else a
        right = a if reverse else b
        hook = _compose_binary_derivative_hook(op, left=left, right=right)
        if hook is not None and can_attach_hook and len(meta) == 0:
            merged_meta = dict(meta)
            merged_meta[_DERIVATIVE_HOOK_KEY] = hook
            meta = frozendict(merged_meta)

        return DomainFunction(
            domain=joined,
            deps=deps,
            func=_BinaryCallable(
                a=a, b=b, op=op, a_pos=a_pos, b_pos=b_pos, reverse=reverse
            ),
            metadata=meta,
        )

    def __add__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.add)

    def __radd__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.add, reverse=True)

    def __sub__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.sub, reverse=True)

    def __mul__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.mul, reverse=True)

    def __matmul__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.matmul)

    def __rmatmul__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.matmul, reverse=True)

    def __truediv__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(
        self, other: "DomainFunction | ArrayLike | None"
    ) -> "DomainFunction":
        return self._binary_op(other, operator.truediv, reverse=True)

    def __pow__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.pow)

    def __rpow__(self, other: "DomainFunction | ArrayLike | None") -> "DomainFunction":
        return self._binary_op(other, operator.pow, reverse=True)

    def __neg__(self) -> "DomainFunction":
        return (-1.0) * self

    def __abs__(self) -> "DomainFunction":
        return DomainFunction(
            domain=self.domain,
            deps=self.deps,
            func=_UnaryCallable(self.func, operator.abs),
            metadata=_drop_derivative_hook_metadata(self.metadata),
        )

    @property
    def T(self) -> "DomainFunction":
        r"""Transpose the last two array axes of the output.

        If $u(z)\in\mathbb{R}^{m\times n}$ then $(u^T)(z)=u(z)^T$.
        """
        return DomainFunction(
            domain=self.domain,
            deps=self.deps,
            func=_SwapAxesCallable(self.func, -2, -1),
            metadata=_transpose_hook_metadata(self.metadata),
        )

    def __call__(
        self,
        points: PointsBatch | CoordSeparableBatch | Points,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        from ._model_function import _ConcatenatedModelCallable

        blockwise_eval = kwargs.pop("_blockwise_eval", False)
        blockwise_mode = "vmap"
        if isinstance(blockwise_eval, str):
            mode = blockwise_eval.strip().lower()
            if mode not in ("vmap", "direct"):
                raise ValueError(
                    "_blockwise_eval string mode must be 'vmap' or 'direct'."
                )
            force_blockwise = True
            blockwise_mode = mode
        else:
            force_blockwise = bool(blockwise_eval)

        def _emit_blockwise_fallback(reason: str, /) -> None:
            if isinstance(self.func, _ConcatenatedModelCallable):
                self.func.emit_auto_fallback_warning(
                    "Falling back to pointwise/axis-vmapped evaluation for DomainFunction model call: "
                    + reason
                )

        def _try_blockwise_eval(
            *,
            points_map: Mapping[str, PyTree[Any]],
            structure: Any | None,
            dense_structure: Any | None,
            coord_axes_by_label: Mapping[str, tuple[str, ...]] | None,
        ) -> tuple[cx.Field | None, str | None]:
            if not isinstance(self.func, _ConcatenatedModelCallable):
                return None, None
            if not self.func.supports_blockwise_input:
                return None, None
            if not self.deps:
                return None, "blockwise model execution requires non-empty dependencies."

            args: list[Any] = []
            axis_order: list[str] = []

            if dense_structure is None:
                if structure is None:
                    return None, "blockwise model execution requires a structured batch."
                if structure.axis_names is None:
                    return (
                        None,
                        "ProductStructure must be canonicalized for blockwise execution.",
                    )
                for dep in self.deps:
                    axis = _singleton_axis_for_label(structure, dep)
                    if axis is None:
                        return (
                            None,
                            f"dependency {dep!r} is not sampled in a singleton block.",
                        )
                    arg = _as_blockwise_arg(points_map[dep])
                    if arg is None:
                        return (
                            None,
                            f"dependency {dep!r} has unsupported sample type for blockwise evaluation.",
                        )
                    if isinstance(arg, tuple):
                        return (
                            None,
                            f"dependency {dep!r} is coord-separable but batch is not CoordSeparableBatch.",
                        )
                    args.append(arg)
                    axis_order.append(axis)
            else:
                if dense_structure.axis_names is None:
                    return (
                        None,
                        "CoordSeparableBatch dense_structure must be canonicalized.",
                    )
                for dep in self.deps:
                    dep_axes = (
                        coord_axes_by_label.get(dep)
                        if coord_axes_by_label is not None
                        else None
                    )
                    arg = _as_blockwise_arg(points_map[dep])
                    if arg is None:
                        return (
                            None,
                            f"dependency {dep!r} has unsupported sample type for blockwise evaluation.",
                        )
                    if dep_axes is not None:
                        if not isinstance(arg, tuple):
                            return (
                                None,
                                f"dependency {dep!r} expected coord-separable tuple inputs.",
                            )
                        args.append(arg)
                        axis_order.extend(dep_axes)
                        continue

                    axis = _singleton_axis_for_label(dense_structure, dep)
                    if axis is None:
                        return (
                            None,
                            f"dependency {dep!r} is not sampled in a singleton dense block.",
                        )
                    if isinstance(arg, tuple):
                        return (
                            None,
                            f"dependency {dep!r} unexpectedly produced tuple inputs in dense structure.",
                        )
                    args.append(arg)
                    axis_order.append(axis)

            axis_order = list(_dedupe_axes(axis_order))
            y = jnp.asarray(self.func(*args, key=key, **kwargs))
            if y.ndim < len(axis_order):
                return (
                    None,
                    "model output rank is smaller than inferred blockwise axis rank.",
                )
            out_dims = tuple(axis_order) + (None,) * (y.ndim - len(axis_order))
            return cx.Field(y, dims=out_dims), None

        if isinstance(points, PointsBatch):
            points_map = points.points
            structure = points.structure
            dense_structure = None
            coord_axes_by_label = None
        else:
            structure = None
            if isinstance(points, CoordSeparableBatch):
                points_map = points.points
                dense_structure = points.dense_structure
                coord_axes_by_label = points.coord_axes_by_label
            else:
                points_map = points
                dense_structure = None
                coord_axes_by_label = None

        for lbl in self.domain.labels:
            if lbl not in points_map:
                raise KeyError(
                    f"Missing label {lbl!r} in points; expected at least {self.domain.labels}."
                )

        if dense_structure is not None:
            if dense_structure.axis_names is None:
                raise ValueError(
                    "CoordSeparableBatch.dense_structure must be canonicalized (axis_names set)."
                )

            out, fallback_reason = _try_blockwise_eval(
                points_map=points_map,
                structure=structure,
                dense_structure=dense_structure,
                coord_axes_by_label=coord_axes_by_label,
            )
            if out is None:
                if fallback_reason is not None:
                    _emit_blockwise_fallback(fallback_reason)
                mapped_axes: list[str] = []
                mapped_blocks: list[tuple[str, ...]] = []
                for block, axis in zip(
                    dense_structure.blocks, dense_structure.axis_names, strict=True
                ):
                    if any(lbl in self.deps for lbl in block):
                        mapped_blocks.append(block)
                        mapped_axes.append(axis)

                if not self.deps:
                    val = self.func(key=key, **kwargs)
                    y = jnp.asarray(val)
                else:
                    dep_values = tuple(
                        _unwrap_fields_to_data(points_map[d]) for d in self.deps
                    )

                    def _call(*args):
                        return self.func(*args, key=key, **kwargs)

                    mapped = _call
                    for block in reversed(mapped_blocks):
                        in_axes = tuple(0 if dep in block else None for dep in self.deps)
                        mapped = jax.vmap(mapped, in_axes=in_axes, out_axes=0)

                    y = jnp.asarray(mapped(*dep_values))

                if not mapped_axes and not (
                    coord_axes_by_label
                    and any(dep in coord_axes_by_label for dep in self.deps)
                ):
                    out = cx.Field(y, dims=(None,) * y.ndim)
                else:
                    axis_order: list[str] = []
                    axis_order.extend(mapped_axes)
                    if coord_axes_by_label is not None:
                        for dep in self.deps:
                            axes = coord_axes_by_label.get(dep)
                            if axes is not None:
                                axis_order.extend(axes)

                    used_axes: list[str] = []
                    shape_i = 0
                    started = False
                    for axis in axis_order:
                        if shape_i >= y.ndim:
                            break
                        n = _axis_size(points_map, axis)
                        if y.shape[shape_i] == n:
                            used_axes.append(axis)
                            shape_i += 1
                            started = True
                            continue
                        if started:
                            break
                    out_dims = tuple(used_axes) + (None,) * (y.ndim - shape_i)
                    out = cx.Field(y, dims=out_dims)

            if not isinstance(out, cx.Field):
                raise TypeError("DomainFunction must return a coordax.Field.")

            for lbl in self.domain.labels:
                axes: tuple[str, ...] | None = None
                if coord_axes_by_label is not None:
                    axes = coord_axes_by_label.get(lbl)
                if axes is not None:
                    for axis in axes:
                        if axis in out.named_dims:
                            continue
                        n = _axis_size(points_map, axis)
                        one = cx.Field(jnp.ones((n,), dtype=float), dims=(axis,))
                        out = out * one
                    continue

                axis = dense_structure.axis_for(lbl)
                if axis is None or axis in out.named_dims:
                    continue
                n = _axis_size(points_map, axis)
                one = cx.Field(jnp.ones((n,), dtype=float), dims=(axis,))
                out = out * one

            out = _reorder_named_axes(out, dense_structure.axis_names)
            return out

        out, fallback_reason = _try_blockwise_eval(
            points_map=points_map,
            structure=structure,
            dense_structure=dense_structure,
            coord_axes_by_label=coord_axes_by_label,
        )
        if out is None:
            if fallback_reason is not None:
                _emit_blockwise_fallback(fallback_reason)
            if force_blockwise:
                if structure is None:
                    raise ValueError(
                        "_blockwise_eval requires a structured PointsBatch input."
                    )
                if structure.axis_names is None:
                    raise ValueError(
                        "PointsBatch.structure must be canonicalized (axis_names set)."
                    )
                mapped_axes: list[str] = []
                mapped_blocks: list[tuple[str, ...]] = []
                for block, axis in zip(
                    structure.blocks, structure.axis_names, strict=True
                ):
                    if any(lbl in self.deps for lbl in block):
                        mapped_blocks.append(block)
                        mapped_axes.append(axis)

                if not self.deps:
                    val = self.func(key=key, **kwargs)
                    y = jnp.asarray(val)
                else:
                    dep_values = tuple(
                        _unwrap_fields_to_data(points_map[d]) for d in self.deps
                    )
                    if blockwise_mode == "direct":
                        y = jnp.asarray(self.func(*dep_values, key=key, **kwargs))
                    elif mapped_blocks:

                        def _call(*args):
                            return self.func(*args, key=key, **kwargs)

                        mapped = _call
                        for block in reversed(mapped_blocks):
                            in_axes = tuple(
                                0 if dep in block else None for dep in self.deps
                            )
                            mapped = jax.vmap(mapped, in_axes=in_axes, out_axes=0)
                        y = jnp.asarray(mapped(*dep_values))
                    else:
                        y = jnp.asarray(self.func(*dep_values, key=key, **kwargs))

                if not mapped_axes:
                    out = cx.Field(y, dims=(None,) * y.ndim)
                else:
                    used_axes: list[str] = []
                    shape_i = 0
                    started = False
                    for axis in mapped_axes:
                        if shape_i >= y.ndim:
                            break
                        n = _axis_size(points_map, axis)
                        if y.shape[shape_i] == n:
                            used_axes.append(axis)
                            shape_i += 1
                            started = True
                            continue
                        if started:
                            break
                    out_dims = tuple(used_axes) + (None,) * (y.ndim - shape_i)
                    out = cx.Field(y, dims=out_dims)
            else:
                if not self.deps:
                    val = self.func(key=key, **kwargs)
                    out = cx.Field(jnp.asarray(val), dims=(None,) * jnp.asarray(val).ndim)
                else:
                    dep_values = tuple(points_map[d] for d in self.deps)
                    out = cx.cmap(self.func, out_axes="leading")(
                        *dep_values, key=key, **kwargs
                    )
        if not isinstance(out, cx.Field):
            raise TypeError("DomainFunction must return a coordax.Field.")

        if structure is None:
            return out

        if structure.axis_names is None:
            raise ValueError(
                "PointsBatch.structure must be canonicalized (axis_names set)."
            )

        for lbl in self.domain.labels:
            axis = structure.axis_for(lbl)
            if axis is None or axis in out.named_dims:
                continue

            field = _first_field_leaf(points_map[lbl])
            if axis not in field.named_shape:
                raise ValueError(
                    f"Cannot infer size for sampling axis {axis!r} from points[{lbl!r}]."
                )
            n = int(field.named_shape[axis])
            one = cx.Field(jnp.ones((n,), dtype=float), dims=(axis,))
            out = out * one

        out = _reorder_named_axes(out, structure.axis_names)
        return out
