#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import factorial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._bvh import beam_select_leaf_items, build_point_bvh
from .._callable import _ensure_special_kwonly_args
from .._doc import DOC_KEY0
from .._strict import StrictModule
from ..domain._base import _AbstractGeometry
from ..domain._components import (
    Boundary,
    DomainComponent,
    DomainComponentUnion,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
)
from ..domain._domain import RelabeledDomain
from ..domain._function import DomainFunction
from ..domain._scalar import _AbstractScalarDomain
from ..operators.differential._domain_ops import (
    cauchy_stress,
    directional_derivative,
    dt,
)
from ..operators.linalg import einsum


class _IdentityCallable(StrictModule):
    def __call__(self, x, /, *, key=None, **kwargs):
        del key, kwargs
        return x


def _constant_weight(value: float, /) -> Callable[[Array], Array]:
    def _w(x):
        del x
        return jnp.asarray(value, dtype=float)

    return _w


def _coerce_value(value: Any, u: DomainFunction, /) -> DomainFunction | ArrayLike:
    if isinstance(value, DomainFunction):
        return value
    if callable(value):
        return DomainFunction(domain=u.domain, deps=u.deps, func=value, metadata={})
    return value


def _guard_no_coord_separable(
    u: DomainFunction,
    /,
    *,
    var: str,
    op_name: str,
) -> DomainFunction:
    if var not in u.deps:
        return u

    var_pos = u.deps.index(var)

    def _guarded(*args, key=None, **kwargs):
        x = args[var_pos]
        if isinstance(x, tuple):
            raise ValueError(
                f"{op_name} does not support coord-separable (tuple-of-axes) evaluation "
                "because it relies on geometry boundary normals. Use dense/point sampling "
                "or a compatible constraint/ansatz."
            )
        return u.func(*args, key=key, **kwargs)

    return DomainFunction(
        domain=u.domain, deps=u.deps, func=_guarded, metadata=u.metadata
    )


def _safe_norms(values: Array, /, *, keepdims: bool = False) -> Array:
    if values.ndim == 1:
        values = jnp.expand_dims(values, axis=-1)
    value_norms_squared = jnp.sum(jnp.square(values), axis=-1, keepdims=keepdims)
    value_norms = jnp.sqrt(value_norms_squared + jnp.finfo(float).eps)
    return value_norms


def _enforced_constraint_weight_fn(
    geom: _AbstractGeometry,
    where: Callable | None,
    /,
    *,
    num_reference: int = 3_000_000,
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
    on_empty: Literal["error", "zero"] = "error",
) -> Callable[[Array], Array]:
    """Compute a boundary-subset weight function using a point BVH.

    Computes an oriented MLS distance-to-subset field, then returns an inverse-power
    weight function suitable for blending multiple enforced ansatz pieces via weighted
    averaging.
    """
    if where is None:
        return _constant_weight(1.0)

    where_wrapped = _ensure_special_kwonly_args(where)

    bounds = jnp.asarray(geom.mesh_bounds, dtype=float)
    diameter = float(jnp.linalg.norm(bounds[1] - bounds[0]) + 1e-12)

    n_ref = int(num_reference)
    ref_points = jnp.asarray(
        geom.sample_boundary(n_ref, sampler=sampler, key=key),
        dtype=float,
    )
    mask = jax.vmap(where_wrapped)(ref_points)
    if int(jnp.sum(mask)) == 0:
        if on_empty == "error":
            raise ValueError(
                "Enforced-constraint subset predicate selects no boundary points."
            )
        return _constant_weight(0.0)

    P = jnp.asarray(ref_points[mask], dtype=float)
    ref_normals = jnp.asarray(geom._boundary_normals(P), dtype=float)
    ref_normals = ref_normals / (_safe_norms(ref_normals, keepdims=True) + 1e-12)
    N = ref_normals

    h = 0.02 * diameter
    sigma = h
    kappa = 0.25
    normal_floor = 0.5
    leaf_size = 32
    beam_width = 16  # beam_width * leaf_size == 512 candidates
    eps = 1e-12

    bvh = build_point_bvh(np.asarray(P), leaf_size=leaf_size, dtype=jnp.float64)
    steps = int(bvh.max_depth + 2)

    def _select_candidates(q: Array, /) -> tuple[Array, Array]:
        return beam_select_leaf_items(
            q,
            bvh=bvh,
            beam_width=beam_width,
            steps=steps,
        )

    def _mls_distance(x: Array, /) -> Array:
        x = jnp.asarray(x, dtype=float).reshape((-1,))

        q_sel = jax.lax.stop_gradient(x)
        idx, valid = _select_candidates(q_sel)

        p = P[idx]
        n = N[idx]

        r = x[None, :] - p
        dist2 = jnp.sum(r * r, axis=1)

        has_valid = jnp.any(valid)

        def _do(_: Any) -> Array:
            inv_h2 = 1.0 / (h * h + eps)
            s_dist = -dist2 * inv_h2
            s_dist = jnp.where(valid, s_dist, jnp.asarray(-jnp.inf))
            s_dist = s_dist - jax.lax.stop_gradient(jnp.max(s_dist))
            w_dist = jnp.exp(s_dist)

            rnorm = jnp.sqrt(dist2 + eps)
            rhat = r / rnorm[:, None]
            cos = jnp.sum(rhat * n, axis=1)
            cos = jnp.clip(cos, -1.0, 1.0)
            penalty = (1.0 - cos) ** 2
            w_norm = 1.0 / (1.0 + kappa * penalty)
            w_norm = normal_floor + (1.0 - normal_floor) * w_norm
            w_norm = jnp.where(valid, w_norm, 0.0)

            w = w_dist * w_norm
            wsum = jnp.sum(w) + eps
            w = w / wsum

            proj = jnp.sum(n * r, axis=1)
            return jnp.sum(w * proj)

        return jax.lax.cond(has_valid, _do, lambda _: jnp.asarray(1.0), operand=None)

    def _weight_point(x: Array, /) -> Array:
        f = _mls_distance(x)

        alpha = sigma + eps
        rho = 2.0 * alpha * (jax.nn.softplus(f / alpha) - jnp.log(2.0))
        rho = jnp.abs(rho)
        return (rho + 1e-16) ** -2

    return _weight_point


def enforce_dirichlet(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    target: DomainFunction | ArrayLike | None = None,
) -> DomainFunction:
    r"""Enforced Dirichlet ansatz enforcing $u=g$ on a component.

    Constructs an ansatz $u^*$ that satisfies the Dirichlet condition exactly on the
    selected component (boundary or fixed slice). For a geometry boundary with signed
    distance field $\phi$ (so $\phi=0$ on $\partial\Omega$), this uses

    $$
    u^*(x) = g(x) + \phi(x)\,(u(x) - g(x)),
    $$

    which guarantees $u^*(x)=g(x)$ on $\partial\Omega$.

    For scalar domains (e.g. time), an appropriate vanishing factor $\phi(t)$ is
    constructed from $(t-t_0)$, $(t-t_1)$, etc.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "enforce_dirichlet requires a DomainComponent, not a DomainComponentUnion."
        )

    if var not in component.domain.labels:
        raise KeyError(f"Label {var!r} not in domain {component.domain.labels}.")

    factor = component.domain.factor(var)
    if isinstance(factor, RelabeledDomain):
        factor = factor.base

    comp = component.spec.component_for(var)
    if isinstance(comp, Interior):
        raise ValueError("enforce_dirichlet requires a non-interior component for var.")

    value = 0.0 if target is None else _coerce_value(target, u)

    if isinstance(factor, _AbstractGeometry):
        if not isinstance(comp, Boundary):
            raise ValueError(
                "enforce_dirichlet for geometry vars requires component Boundary()."
            )
        phi = component.sdf(var=var)
        return value + phi * (u - value)

    if isinstance(factor, _AbstractScalarDomain):
        t = DomainFunction(domain=component.domain, deps=(var,), func=_IdentityCallable())
        if isinstance(comp, FixedStart):
            phi = t - factor.fixed("start")
        elif isinstance(comp, FixedEnd):
            phi = t - factor.fixed("end")
        elif isinstance(comp, Fixed):
            phi = t - jnp.asarray(comp.value, dtype=float).reshape(())
        elif isinstance(comp, Boundary):
            phi = (t - factor.fixed("start")) * (t - factor.fixed("end"))
        else:
            raise TypeError(f"Unsupported scalar component {type(comp).__name__}.")

        return value + phi * (u - value)

    raise TypeError(f"Unsupported unary domain type {type(factor).__name__}.")


def enforce_neumann(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    target: DomainFunction | ArrayLike | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Enforced Neumann ansatz enforcing $\partial u/\partial n = g$ on a boundary.

    For a geometry boundary with signed distance field $\phi$ and outward normal $n$,
    this constructs

    $$
    u^* = u + \frac{\phi}{\partial\phi/\partial n}\,\bigl(g - \partial u/\partial n\bigr),
    $$

    which yields $\partial u^*/\partial n = g$ on $\partial\Omega$ under mild regularity
    assumptions.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "enforce_neumann requires a DomainComponent, not a DomainComponentUnion."
        )

    if var not in component.domain.labels:
        raise KeyError(f"Label {var!r} not in domain {component.domain.labels}.")

    factor = component.domain.factor(var)
    if isinstance(factor, RelabeledDomain):
        factor = factor.base
    if not isinstance(factor, _AbstractGeometry):
        raise TypeError("enforce_neumann is only defined for geometry variables.")

    comp = component.spec.component_for(var)
    if not isinstance(comp, Boundary):
        raise ValueError("enforce_neumann requires component Boundary() for var.")

    phi = component.sdf(var=var)
    n = component.normal(var=var)

    du_dn = directional_derivative(u, n, var=var, mode=mode)
    dphi_dn = directional_derivative(phi, n, var=var, mode=mode)

    value = 0.0 if target is None else _coerce_value(target, u)
    out = u + (phi / dphi_dn) * (value - du_dn)
    return _guard_no_coord_separable(out, var=var, op_name="enforce_neumann")


def enforce_traction(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    target: DomainFunction | ArrayLike | None = None,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Enforced traction ansatz for linear elasticity on a boundary.

    For isotropic linear elasticity, the traction is $t=\sigma(u)\,n$.
    This constructs a corrected displacement $u^*$ that aims to satisfy

    $$
    \sigma(u^*)\,n = t_{\text{target}}
    $$

    on the boundary by adding a displacement correction proportional to the signed
    distance $\phi$.

    Let

    $$
    r \;=\; t_{\text{target}} - \sigma(u)\,n
    $$

    be the traction residual, and decompose it into normal/tangential parts

    $$
    r_n = (r\cdot n)\,n,\qquad r_t = r - r_n.
    $$

    Using Lamé parameters $\lambda,\mu$, define the correction field

    $$
    v \;=\; \frac{r_t}{\mu} \;+\; \frac{r_n}{\lambda + 2\mu},
    $$

    and return the hard-constraint ansatz

    $$
    u^*(x) \;=\; u(x) + \phi(x)\,v(x).
    $$

    In the idealized setting where $\phi$ is a signed distance field near the boundary
    (so $\partial\phi/\partial n \approx 1$) and $v$ varies slowly in the normal
    direction, the induced traction correction is approximately
    $\sigma(\phi v)\,n \approx \mu\,v_t + (\lambda+2\mu)\,v_n$ (with
    $v_n=(v\cdot n)\,n$ and $v_t=v-v_n$), making $u^*$ enforce the target traction to
    first order.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "enforce_traction requires a DomainComponent, not a DomainComponentUnion."
        )

    if var not in component.domain.labels:
        raise KeyError(f"Label {var!r} not in domain {component.domain.labels}.")

    factor = component.domain.factor(var)
    if isinstance(factor, RelabeledDomain):
        factor = factor.base
    if not isinstance(factor, _AbstractGeometry):
        raise TypeError("enforce_traction is only defined for geometry variables.")

    comp = component.spec.component_for(var)
    if not isinstance(comp, Boundary):
        raise ValueError("enforce_traction requires component Boundary() for var.")

    phi = component.sdf(var=var)
    n = component.normal(var=var)

    if target is None:
        target_val: DomainFunction | ArrayLike = 0.0
    else:
        target_val = _coerce_value(target, u)
    target_fn = (
        target_val
        if isinstance(target_val, DomainFunction)
        else DomainFunction(domain=u.domain, deps=(), func=target_val, metadata={})
    )

    lambda_val = _coerce_value(lambda_, u)
    lambda_fn = (
        lambda_val
        if isinstance(lambda_val, DomainFunction)
        else DomainFunction(domain=u.domain, deps=(), func=lambda_val, metadata={})
    )

    mu_val = _coerce_value(mu, u)
    mu_fn = (
        mu_val
        if isinstance(mu_val, DomainFunction)
        else DomainFunction(domain=u.domain, deps=(), func=mu_val, metadata={})
    )

    sigma = cauchy_stress(u, lambda_=lambda_fn, mu=mu_fn, var=var, mode=mode)
    traction = einsum("...ij,...j->...i", sigma, n)

    r = target_fn - traction
    r_dot_n = einsum("...i,...i->...", r, n)
    r_n = einsum("...,...i->...i", r_dot_n, n)
    r_t = r - r_n

    denom_n = lambda_fn + 2.0 * mu_fn
    v = (r_t / mu_fn) + (r_n / denom_n)

    out = u + phi * v
    return _guard_no_coord_separable(out, var=var, op_name="enforce_traction")


def enforce_robin(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    dirichlet_coeff: DomainFunction | ArrayLike | None = None,
    neumann_coeff: DomainFunction | ArrayLike | None = None,
    target: DomainFunction | ArrayLike | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Enforced Robin ansatz enforcing $a\,u + b\,\partial u/\partial n = g$.

    On a geometry boundary, this constructs a corrected field that satisfies the Robin
    condition exactly by using a signed distance factor $\phi$ and normal derivative
    of $\phi$.

    Let $r = g - a\,u - b\,\partial u/\partial n$ be the residual. With signed distance
    field $\phi$ and outward normal $n$, define $d\phi/dn = \partial\phi/\partial n$.
    The returned ansatz is

    $$
    u^* \;=\; u + \frac{\phi}{b\,(\partial\phi/\partial n)}\,r,
    $$

    which yields $a\,u^* + b\,\partial u^*/\partial n = g$ on $\partial\Omega$ under
    mild regularity assumptions.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "enforce_robin requires a DomainComponent, not a DomainComponentUnion."
        )

    if var not in component.domain.labels:
        raise KeyError(f"Label {var!r} not in domain {component.domain.labels}.")

    factor = component.domain.factor(var)
    if isinstance(factor, RelabeledDomain):
        factor = factor.base
    if not isinstance(factor, _AbstractGeometry):
        raise TypeError("enforce_robin is only defined for geometry variables.")

    comp = component.spec.component_for(var)
    if not isinstance(comp, Boundary):
        raise ValueError("enforce_robin requires component Boundary() for var.")

    g = 0.0 if target is None else _coerce_value(target, u)
    a = 1.0 if dirichlet_coeff is None else _coerce_value(dirichlet_coeff, u)

    if neumann_coeff is None:
        return enforce_dirichlet(u, component, var=var, target=g / a)
    if not isinstance(neumann_coeff, DomainFunction) and not callable(neumann_coeff):
        b_val = jnp.asarray(neumann_coeff, dtype=float)
        if b_val.shape == () and float(b_val) == 0.0:
            return enforce_dirichlet(u, component, var=var, target=g / a)

    phi = component.sdf(var=var)
    n = component.normal(var=var)
    du_dn = directional_derivative(u, n, var=var, mode=mode)
    dphi_dn = directional_derivative(phi, n, var=var, mode=mode)

    if isinstance(a, DomainFunction):
        a_fn = a
    else:
        a_fn = DomainFunction(domain=u.domain, deps=(), func=a, metadata={})
    b_val = _coerce_value(neumann_coeff, u)
    if isinstance(b_val, DomainFunction):
        b_fn = b_val
    else:
        b_fn = DomainFunction(domain=u.domain, deps=(), func=b_val, metadata={})
    if isinstance(g, DomainFunction):
        g_fn = g
    else:
        g_fn = DomainFunction(domain=u.domain, deps=(), func=g, metadata={})

    r = g_fn - a_fn * u - b_fn * du_dn
    out = u + (phi / (b_fn * dphi_dn)) * r
    return _guard_no_coord_separable(out, var=var, op_name="enforce_robin")


def enforce_sommerfeld(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str = "x",
    time_var: str = "t",
    wavespeed: DomainFunction | ArrayLike | None = None,
    target: DomainFunction | ArrayLike | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Enforced Sommerfeld/absorbing boundary ansatz.

    Enforces the first-order radiation condition

    $$
    \frac{\partial u}{\partial n} + \frac{1}{c}\frac{\partial u}{\partial t} = g
    $$

    on the selected boundary component by correcting $u$ using a signed-distance factor.

    With signed distance field $\phi$ and $d\phi/dn = \partial\phi/\partial n$, let

    $$
    r = g - \frac{\partial u}{\partial n} - \frac{1}{c}\frac{\partial u}{\partial t}.
    $$

    The returned ansatz is

    $$
    u^* \;=\; u + \frac{\phi}{\partial\phi/\partial n}\,r,
    $$

    which yields the Sommerfeld condition on $\partial\Omega$ under the same
    assumptions as `enforce_neumann`.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "enforce_sommerfeld requires a DomainComponent, not a DomainComponentUnion."
        )

    if var not in component.domain.labels:
        raise KeyError(f"Label {var!r} not in domain {component.domain.labels}.")
    if time_var not in component.domain.labels:
        raise KeyError(f"Label {time_var!r} not in domain {component.domain.labels}.")

    factor = component.domain.factor(var)
    if isinstance(factor, RelabeledDomain):
        factor = factor.base
    if not isinstance(factor, _AbstractGeometry):
        raise TypeError("enforce_sommerfeld is only defined for geometry variables.")

    comp = component.spec.component_for(var)
    if not isinstance(comp, Boundary):
        raise ValueError("enforce_sommerfeld requires component Boundary() for var.")

    phi = component.sdf(var=var)
    n = component.normal(var=var)
    du_dn = directional_derivative(u, n, var=var, mode=mode)
    du_dt = dt(u, var=time_var, mode=mode)
    dphi_dn = directional_derivative(phi, n, var=var, mode=mode)

    c = 1.0 if wavespeed is None else _coerce_value(wavespeed, u)
    g = 0.0 if target is None else _coerce_value(target, u)

    if isinstance(c, DomainFunction):
        c_fn = c
    else:
        c_fn = DomainFunction(domain=u.domain, deps=(), func=c, metadata={})
    if isinstance(g, DomainFunction):
        g_fn = g
    else:
        g_fn = DomainFunction(domain=u.domain, deps=(), func=g, metadata={})

    r = g_fn - du_dn - (1.0 / c_fn) * du_dt
    out = u + (phi / dphi_dn) * r
    return _guard_no_coord_separable(out, var=var, op_name="enforce_sommerfeld")


def enforce_initial(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str = "t",
    targets: Mapping[int, DomainFunction | ArrayLike],
) -> DomainFunction:
    r"""Enforced initial-condition ansatz matching time derivatives at a fixed time.

    Let $t_0$ be the fixed time selected by `component` (e.g. `FixedStart()`).
    Given targets for derivatives $u^{(k)}(t_0)$ for $k=0,\dots,m$, this constructs a
    Taylor-like polynomial

    $$
    p(t) = \sum_{k=0}^{m} \frac{u^{(k)}(t_0)}{k!}\,(t-t_0)^k
    $$

    and returns

    $$
    u^*(t) = p(t) + (t-t_0)^{m+1}\,(u(t)-p(t)),
    $$

    which matches all specified derivatives exactly at $t=t_0$.
    """
    if isinstance(component, DomainComponentUnion):
        raise TypeError(
            "enforce_initial requires a DomainComponent, not a DomainComponentUnion."
        )
    if var not in component.domain.labels:
        raise KeyError(f"Label {var!r} not in domain {component.domain.labels}.")
    if not targets:
        raise ValueError("enforce_initial requires at least one derivative target.")

    factor = component.domain.factor(var)
    if isinstance(factor, RelabeledDomain):
        factor = factor.base
    if not isinstance(factor, _AbstractScalarDomain):
        raise TypeError("enforce_initial requires a scalar evolution variable.")

    comp = component.spec.component_for(var)
    if isinstance(comp, FixedStart):
        t0 = factor.fixed("start")
    elif isinstance(comp, FixedEnd):
        t0 = factor.fixed("end")
    elif isinstance(comp, Fixed):
        t0 = jnp.asarray(comp.value, dtype=float).reshape(())
    else:
        raise ValueError(
            "enforce_initial requires FixedStart/FixedEnd/Fixed for the evolution var."
        )

    targets_by_order: dict[int, DomainFunction | ArrayLike] = {}
    for order, target in targets.items():
        if target is None:
            raise ValueError("enforce_initial targets may not be None.")
        order_i = int(order)
        if order_i < 0:
            raise ValueError(
                "enforce_initial targets must use non-negative derivative orders."
            )
        if order_i in targets_by_order:
            raise ValueError(
                f"enforce_initial received duplicate targets for order {order_i}."
            )
        if callable(target) and not isinstance(target, DomainFunction):
            target = DomainFunction(
                domain=u.domain, deps=u.deps, func=target, metadata={}
            )
        targets_by_order[order_i] = target

    max_order = max(targets_by_order)
    for order in range(max_order + 1):
        if order not in targets_by_order:
            raise ValueError(
                "enforce_initial requires targets for all derivative orders from 0..max_order."
            )

    t = DomainFunction(domain=component.domain, deps=(var,), func=_IdentityCallable())
    dt = t - t0

    poly = DomainFunction(domain=u.domain, deps=(), func=0.0, metadata=u.metadata)
    for order in range(max_order + 1):
        target = targets_by_order[order]
        coeff = 1.0 / float(factorial(order))
        poly = poly + target * (dt**order) * coeff

    return poly + (dt ** (max_order + 1)) * (u - poly)


def _complement_where(
    wheres: Sequence[Callable | None],
    /,
) -> Callable[[Array], Array] | None:
    if any(w is None for w in wheres):
        return None
    if not wheres:
        return _constant_weight(1.0)

    wrapped = tuple(_ensure_special_kwonly_args(w) for w in wheres if w is not None)

    def _union(x: Array, /) -> Array:
        out = wrapped[0](x)
        for fn in wrapped[1:]:
            out = jnp.logical_or(out, fn(x))
        return out

    def _comp(x: Array, /) -> Array:
        return jnp.logical_not(_union(x))

    return _comp


def enforce_blend(
    u: DomainFunction,
    pieces: Sequence[tuple[DomainComponent, DomainFunction]],
    /,
    *,
    var: str = "x",
    include_identity_remainder: bool = True,
    num_reference: int = 3_000_000,
    sampler: str = "latin_hypercube",
    key: Key[Array, ""] = DOC_KEY0,
) -> DomainFunction:
    r"""Blend multiple enforced ansatz pieces via ported MLS/BVH weights.

    Each piece contributes $w_i\,u_i^*$ and the final output is:

    $$
    u_{\text{enforced}} = \frac{\sum_i w_i\,u_i^*}{\sum_i w_i}.
    $$

    If `include_identity_remainder=True`, this also adds an identity term for the
    complement of the union of piece predicates (using `component.where[var]`), which
    prevents subset enforced constraints from leaking onto other boundary segments.
    """
    if not pieces:
        raise ValueError("enforce_blend requires at least one piece.")

    if var not in u.domain.labels:
        raise KeyError(f"Label {var!r} not in domain {u.domain.labels}.")

    base_factor = u.domain.factor(var)
    if isinstance(base_factor, RelabeledDomain):
        base_factor = base_factor.base
    if not isinstance(base_factor, _AbstractGeometry):
        raise TypeError("enforce_blend currently supports only geometry variables.")
    geom = base_factor

    resolved_pieces: list[tuple[DomainComponent, DomainFunction]] = []
    for component, u_piece in pieces:
        if isinstance(component, DomainComponentUnion):
            raise TypeError("enforce_blend pieces must be DomainComponent (not a union).")
        if not isinstance(u_piece, DomainFunction):
            if callable(u_piece):
                u_piece = DomainFunction(
                    domain=u.domain, deps=u.deps, func=u_piece, metadata={}
                )
            else:
                u_piece = DomainFunction(
                    domain=u.domain, deps=(), func=u_piece, metadata={}
                )

        if var not in component.domain.labels:
            raise KeyError(
                f"Label {var!r} not in piece domain {component.domain.labels}."
            )
        factor = component.domain.factor(var)
        if isinstance(factor, RelabeledDomain):
            factor = factor.base
        if not isinstance(factor, _AbstractGeometry):
            raise TypeError("enforce_blend pieces must use a geometry label for var.")
        if not geom.equivalent(factor):
            raise ValueError(
                "enforce_blend requires all pieces to share an equivalent geometry."
            )

        comp = component.spec.component_for(var)
        if not isinstance(comp, Boundary):
            raise ValueError("enforce_blend pieces require component Boundary() for var.")
        resolved_pieces.append((component, u_piece))

    num_terms = len(resolved_pieces) + (1 if include_identity_remainder else 0)
    keys = jr.split(key, num_terms)
    key_iter = iter(keys)

    numerator = DomainFunction(domain=u.domain, deps=(), func=0.0, metadata=u.metadata)
    denominator = DomainFunction(domain=u.domain, deps=(), func=0.0, metadata={})

    wheres: list[Callable | None] = []
    for component, u_piece in resolved_pieces:
        where_fn = component.where.get(var)
        wheres.append(where_fn)
        w_fn = _enforced_constraint_weight_fn(
            geom,
            where_fn,
            num_reference=num_reference,
            sampler=sampler,
            key=next(key_iter),
            on_empty="error",
        )
        w = DomainFunction(domain=u.domain, deps=(var,), func=w_fn, metadata={})
        numerator = numerator + w * u_piece
        denominator = denominator + w

    if include_identity_remainder:
        rem_where = _complement_where(wheres)
        if rem_where is not None:
            w_rem_fn = _enforced_constraint_weight_fn(
                geom,
                rem_where,
                num_reference=num_reference,
                sampler=sampler,
                key=next(key_iter),
                on_empty="zero",
            )
            w_rem = DomainFunction(
                domain=u.domain, deps=(var,), func=w_rem_fn, metadata={}
            )
            numerator = numerator + w_rem * u
            denominator = denominator + w_rem

    return numerator / denominator
