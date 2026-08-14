#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

import coordax as cx
import jax
import jax.core as jax_core
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from .._trainable import is_non_trainable_leaf, is_trainable_leaf, NonTrainableState
from ._derivative import (
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
)
from ._domain import Domain
from ._evaluation import BatchEvaluator, evaluate_domain_function


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


class _ConstCallable(StrictModule, NonTrainableState):
    value: jax.Array

    def __init__(self, value: ArrayLike | None):
        if value is None:
            raise TypeError("DomainFunction constants must be array-like, not None.")
        self.value = jnp.asarray(value)

    def __call__(self, *args, key=None, **kwargs):
        del args, key, kwargs
        return self.value


class _TrainableConstCallable(StrictModule):
    value: jax.Array

    def __init__(self, value: ArrayLike | None):
        if value is None:
            raise TypeError("Domain.Parameter constants must be array-like, not None.")
        self.value = jnp.asarray(value)

    def __call__(self, *args, key=None, **kwargs):
        del args, key, kwargs
        return self.value


class UnaryFieldEvaluator(StrictModule):
    func: Callable
    op: Callable[[Any], Any]

    def __init__(self, func: Callable, op: Callable[[Any], Any]):
        self.func = func
        self.op = op

    def __call__(self, *args, key=None, **kwargs):
        return self.op(self.func(*args, key=key, **kwargs))


class SwapAxesFieldEvaluator(StrictModule):
    func: Callable
    axis1: int
    axis2: int

    def __init__(self, func: Callable, axis1: int, axis2: int):
        self.func = func
        self.axis1 = int(axis1)
        self.axis2 = int(axis2)

    def __call__(self, *args, key=None, **kwargs):
        return jnp.swapaxes(self.func(*args, key=key, **kwargs), self.axis1, self.axis2)


class BinaryFieldEvaluator(StrictModule, BatchEvaluator):
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

    def __call_batch__(
        self,
        batch: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        left_fn = self.b if self.reverse else self.a
        right_fn = self.a if self.reverse else self.b
        left = left_fn(batch, key=key, **kwargs)
        right = right_fn(batch, key=key, **kwargs)
        out = self.op(left, right)
        if not isinstance(out, cx.Field):
            raise TypeError("Batch-aware binary DomainFunction must return a Field.")
        return out

    def __call__(self, *args, key=None, **kwargs):
        a_args = [args[i] for i in self.a_pos]
        b_args = [args[i] for i in self.b_pos]
        if self.reverse:
            left = self.b.func(*b_args, key=key, **kwargs)
            right = self.a.func(*a_args, key=key, **kwargs)
        else:
            left = self.a.func(*a_args, key=key, **kwargs)
            right = self.b.func(*b_args, key=key, **kwargs)

        if self.op in (
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
        ):
            return _rank1_leading_broadcast_op(self.op, left, right)
        return self.op(left, right)


@dataclass(frozen=True, slots=True, eq=False)
class _TransposeDerivativeRule(DerivativeRule):
    source: DerivativeRule

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> "DomainFunction | None":
        out = self.source.derive(
            var=var,
            axis=axis,
            order=order,
            mode=mode,
            backend=backend,
            basis=basis,
            periodic=periodic,
        )
        return None if out is None else out.T


def _has_trainable_arrays(function: "DomainFunction", /) -> bool:
    leaves = jax.tree_util.tree_leaves(
        function.func,
        is_leaf=is_non_trainable_leaf,
    )
    return any(is_trainable_leaf(leaf) for leaf in leaves)


def _domain_has_tracer(domain: Domain, /) -> bool:
    return any(
        isinstance(leaf, jax_core.Tracer) for leaf in jax.tree_util.tree_leaves(domain)
    )


def _staged_subdomain(source: Domain, target: Domain, /) -> bool:
    for factor in source.joint_factors:
        matches = tuple(
            candidate
            for candidate in target.joint_factors
            if candidate.labels == factor.labels
        )
        if len(matches) != 1 or not factor.schema_compatible(matches[0]):
            return False
    return True


def _join_field_domains(left: Domain, right: Domain, /) -> Domain:
    if left is right:
        return left
    if not (_domain_has_tracer(left) or _domain_has_tracer(right)):
        return left.join(right)

    from ._product_domain import ProductDomain

    factors = list(left.joint_factors)
    for candidate in right.joint_factors:
        overlaps = tuple(
            factor
            for factor in factors
            if set(candidate.labels).intersection(factor.labels)
        )
        if not overlaps:
            factors.append(candidate)
            continue
        if (
            len(overlaps) != 1
            or overlaps[0].labels != candidate.labels
            or not overlaps[0].schema_compatible(candidate)
        ):
            raise ValueError(
                "Label collision between traced joint domain factors "
                f"{overlaps[0].labels if overlaps else ()} and {candidate.labels}."
            )

    if len(factors) == 1:
        return factors[0]
    return ProductDomain(*factors)


@dataclass(frozen=True, slots=True, eq=False)
class _BinaryDerivativeRule(DerivativeRule):
    op: Callable[[Any, Any], Any]
    left: "DomainFunction"
    right: "DomainFunction"
    operands_are_trainable: bool

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> "DomainFunction | None":
        if self.operands_are_trainable:
            return None
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

        if self.op is operator.add:
            return _derive(self.left, n) + _derive(self.right, n)
        if self.op is operator.sub:
            return _derive(self.left, n) - _derive(self.right, n)
        if self.op is operator.mul:
            return nth_product_rule(
                self.left,
                self.right,
                var=var,
                order=n,
                derive=_derive,
            )
        if self.op is operator.truediv:
            return nth_quotient_rule(
                self.left,
                self.right,
                var=var,
                order=n,
                derive=_derive,
            )
        if self.op is operator.matmul:
            if n == 0:
                return self.left @ self.right
            out = DomainFunction(
                domain=self.left.domain,
                deps=(),
                func=0.0,
                metadata={},
            )
            for k in range(n + 1):
                out = out + float(comb(n, k)) * (
                    _derive(self.left, k) @ _derive(self.right, n - k)
                )
            return out
        return None


def _compose_binary_derivative_rule(
    op: Callable[[Any, Any], Any],
    /,
    *,
    left: "DomainFunction",
    right: "DomainFunction",
) -> DerivativeRule | None:
    if op not in (
        operator.add,
        operator.sub,
        operator.mul,
        operator.matmul,
        operator.truediv,
    ):
        return None
    return _BinaryDerivativeRule(
        op=op,
        left=left,
        right=right,
        operands_are_trainable=(
            _has_trainable_arrays(left) or _has_trainable_arrays(right)
        ),
    )


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

    domain: Domain
    deps: tuple[str, ...]
    func: Callable
    metadata: frozendict[str, Any]
    derivative_rule: DerivativeRule | None

    def __init__(
        self,
        *,
        domain: Domain,
        deps: Sequence[str],
        func: Callable | ArrayLike,
        metadata: Mapping[str, Any] | None = None,
        derivative_rule: DerivativeRule | None = None,
    ):
        if not isinstance(domain, Domain):
            raise TypeError("DomainFunction.domain must be a Domain.")
        deps_ = tuple(deps)
        if len(set(deps_)) != len(deps_):
            raise ValueError(
                f"DomainFunction dependencies must be unique, got {deps_!r}."
            )
        unknown = tuple(label for label in deps_ if label not in domain.labels)
        if unknown:
            raise ValueError(
                f"Unknown dependencies {unknown!r}; expected a subset of {domain.labels!r}."
            )
        self.domain = domain
        if derivative_rule is not None and not isinstance(
            derivative_rule, DerivativeRule
        ):
            raise TypeError(
                "DomainFunction.derivative_rule must be a DerivativeRule or None."
            )
        self.deps = deps_

        self.func = func if callable(func) else _ConstCallable(func)

        self.metadata = frozendict({} if metadata is None else metadata)
        self.derivative_rule = derivative_rule

    def depends_on(self, var: str, /) -> bool:
        """Return whether this function depends on the labeled variable `var`."""
        return var in self.deps

    def promote(self, new_domain: Domain, /) -> "DomainFunction":
        r"""View this function as defined on a larger domain.

        If $\Omega\subseteq\Omega'$, then promotion constructs $u':\Omega'\to\mathbb{R}^m$
        by ignoring the extra coordinates:

        $$
        u'(z) = u(z|_{\Omega}).
        $$

        Promotion is valid only when every complete joint factor of the current
        domain occurs with the same support in ``new_domain``. Matching labels
        alone is insufficient.
        """
        if self.domain is new_domain:
            return self
        if _domain_has_tracer(self.domain) or _domain_has_tracer(new_domain):
            compatible = _staged_subdomain(self.domain, new_domain)
        else:
            compatible = self.domain.is_subdomain_of(new_domain)
        if not compatible:
            raise ValueError(
                f"Cannot promote domain {self.domain.labels} into "
                f"{new_domain.labels}: factor supports are incompatible."
            )
        return DomainFunction(
            domain=new_domain,
            deps=self.deps,
            func=self.func,
            metadata=self.metadata,
            derivative_rule=self.derivative_rule,
        )

    def with_metadata(self, **metadata: Any) -> "DomainFunction":
        """Return a copy with `metadata` merged into the existing metadata."""
        merged = dict(self.metadata)
        merged.update(metadata)
        return DomainFunction(
            domain=self.domain,
            deps=self.deps,
            func=self.func,
            metadata=merged,
            derivative_rule=self.derivative_rule,
        )

    def with_derivative_rule(
        self,
        rule: DerivativeRule | None,
        /,
    ) -> "DomainFunction":
        """Return a copy using an explicit derivative strategy."""
        return DomainFunction(
            domain=self.domain,
            deps=self.deps,
            func=self.func,
            metadata=self.metadata,
            derivative_rule=rule,
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

        joined = _join_field_domains(self.domain, other_fn.domain)
        a = self.promote(joined)
        b = other_fn.promote(joined)

        deps = tuple(lbl for lbl in joined.labels if (lbl in a.deps) or (lbl in b.deps))
        idx = {lbl: i for i, lbl in enumerate(deps)}
        a_pos = tuple(idx[lbl] for lbl in a.deps)
        b_pos = tuple(idx[lbl] for lbl in b.deps)

        if not b.metadata:
            meta = a.metadata
        elif not a.metadata:
            meta = b.metadata
        elif a.metadata == b.metadata:
            meta = a.metadata
        else:
            meta = frozendict({})

        left = b if reverse else a
        right = a if reverse else b
        derivative_rule = _compose_binary_derivative_rule(
            op,
            left=left,
            right=right,
        )

        return DomainFunction(
            domain=joined,
            deps=deps,
            func=BinaryFieldEvaluator(
                a=a, b=b, op=op, a_pos=a_pos, b_pos=b_pos, reverse=reverse
            ),
            metadata=meta,
            derivative_rule=derivative_rule,
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
            func=UnaryFieldEvaluator(self.func, operator.abs),
            metadata=self.metadata,
        )

    @property
    def T(self) -> "DomainFunction":
        r"""Transpose the last two array axes of the output.

        If $u(z)\in\mathbb{R}^{m\times n}$ then $(u^T)(z)=u(z)^T$.
        """
        return DomainFunction(
            domain=self.domain,
            deps=self.deps,
            func=SwapAxesFieldEvaluator(self.func, -2, -1),
            metadata=self.metadata,
            derivative_rule=(
                None
                if self.derivative_rule is None
                else _TransposeDerivativeRule(self.derivative_rule)
            ),
        )

    def __call__(
        self,
        points: Any,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> cx.Field:
        return evaluate_domain_function(
            self.func,
            deps=self.deps,
            domain_labels=self.domain.labels,
            points=points,
            key=key,
            kwargs=kwargs,
        )
