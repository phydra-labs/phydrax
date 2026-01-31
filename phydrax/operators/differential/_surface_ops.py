#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import opt_einsum as oe

from ...domain._components import DomainComponent
from ...domain._function import DomainFunction
from ...domain._scalar import _AbstractScalarDomain
from ._domain_ops import _factor_and_dim, _resolve_var, curl, grad


def _proj_from_normals(n):
    eps = jnp.finfo(float).eps
    n = jnp.asarray(n, dtype=float)
    nrm = jnp.linalg.norm(n, axis=-1, keepdims=True) + eps
    n = n / nrm
    d = int(n.shape[-1])
    I = jnp.eye(d, dtype=n.dtype)
    I = jnp.broadcast_to(I, n.shape[:-1] + (d, d))
    nnT = oe.contract("...i,...j->...ij", n, n)
    return I - nnT


def tangential_component(
    w: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    r"""Project a vector field onto the local tangent space.

    Given a unit normal field $n(x)$ on a boundary component, the tangential projection
    of a vector field $w$ is

    $$
    w_{\tau} = w - (w\cdot n)\,n.
    $$

    **Arguments:**

    - `w`: Vector field to project (trailing size = ambient dimension).
    - `component`: Boundary `DomainComponent` used to supply the unit normal field.
    - `var`: Geometry label for the boundary variable (defaults to inferred geometry label).

    **Returns:**

    - A `DomainFunction` representing the tangential projection $w_\tau$.
    """
    var = _resolve_var(w, var)
    factor, _ = _factor_and_dim(w, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "tangential_component(var=...) requires a geometry variable, not a scalar variable."
        )

    n = component.normal(var=var)
    joined = w.domain.join(n.domain)
    w2 = w.promote(joined)
    n2 = n.promote(joined)

    deps = tuple(lbl for lbl in joined.labels if (lbl in w2.deps) or (lbl in n2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    w_pos = tuple(idx[lbl] for lbl in w2.deps)
    n_pos = tuple(idx[lbl] for lbl in n2.deps)

    def _op(*args, key=None, **kwargs):
        wv = jnp.asarray(w2.func(*[args[i] for i in w_pos], key=key, **kwargs))
        nv = jnp.asarray(n2.func(*[args[i] for i in n_pos], key=key, **kwargs))
        nv = jax.lax.stop_gradient(nv)
        dot = jnp.sum(wv * nv, axis=-1, keepdims=True)
        return wv - dot * nv

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=w.metadata)


def surface_grad(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Surface (tangential) gradient on a boundary component.

    Let $n$ be the outward unit normal and $P = I - n\otimes n$ the tangential
    projector. For a scalar field $u$, the surface gradient is

    $$
    \nabla_{\Gamma} u = P\,\nabla u.
    $$

    **Arguments:**

    - `u`: Field to differentiate (typically scalar-valued).
    - `component`: Boundary `DomainComponent` used to supply the unit normal field.
    - `var`: Geometry label for the boundary variable.
    - `mode`: Autodiff mode passed to `grad`.

    **Returns:**

    - A `DomainFunction` representing $\nabla_\Gamma u$ (tangential vector field).
    """
    var = _resolve_var(u, var)
    factor, _ = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "surface_grad(var=...) requires a geometry variable, not a scalar variable."
        )

    n = component.normal(var=var)
    joined = u.domain.join(n.domain)
    u2 = u.promote(joined)
    n2 = n.promote(joined)
    g = grad(u2, var=var, mode=mode)

    deps = tuple(lbl for lbl in joined.labels if (lbl in g.deps) or (lbl in n2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    g_pos = tuple(idx[lbl] for lbl in g.deps)
    n_pos = tuple(idx[lbl] for lbl in n2.deps)

    def _op(*args, key=None, **kwargs):
        gv = jnp.asarray(g.func(*[args[i] for i in g_pos], key=key, **kwargs))
        nv = jnp.asarray(n2.func(*[args[i] for i in n_pos], key=key, **kwargs))
        nv = jax.lax.stop_gradient(nv)
        P = _proj_from_normals(nv)

        if gv.ndim == nv.ndim:
            return oe.contract("...ij,...j->...i", P, gv)
        if gv.ndim == nv.ndim + 1:
            return oe.contract("...md,...dj->...mj", gv, P)
        raise ValueError(
            f"surface_grad got incompatible ranks: grad(u).ndim={gv.ndim}, normal.ndim={nv.ndim}."
        )

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def surface_div(
    v: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Surface (tangential) divergence on a boundary component.

    With tangential projector $P = I - n\otimes n$, this implements

    $$
    \nabla_{\Gamma}\cdot v = \text{tr}(P\,\nabla v),
    $$

    where $\nabla v$ is the Jacobian of $v$ with respect to the ambient coordinates.

    **Arguments:**

    - `v`: Tangential or ambient vector field.
    - `component`: Boundary `DomainComponent` used to supply the unit normal field.
    - `var`: Geometry label for the boundary variable.
    - `mode`: Autodiff mode passed to `grad`.

    **Returns:**

    - A `DomainFunction` representing $\nabla_\Gamma\cdot v$ (scalar field).
    """
    var = _resolve_var(v, var)
    factor, _ = _factor_and_dim(v, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "surface_div(var=...) requires a geometry variable, not a scalar variable."
        )

    n = component.normal(var=var)
    joined = v.domain.join(n.domain)
    v2 = v.promote(joined)
    n2 = n.promote(joined)
    J = grad(v2, var=var, mode=mode)

    deps = tuple(lbl for lbl in joined.labels if (lbl in J.deps) or (lbl in n2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    j_pos = tuple(idx[lbl] for lbl in J.deps)
    n_pos = tuple(idx[lbl] for lbl in n2.deps)

    def _op(*args, key=None, **kwargs):
        Jv = jnp.asarray(J.func(*[args[i] for i in j_pos], key=key, **kwargs))
        nv = jnp.asarray(n2.func(*[args[i] for i in n_pos], key=key, **kwargs))
        nv = jax.lax.stop_gradient(nv)
        P = _proj_from_normals(nv)
        if Jv.ndim < 2 or P.ndim < 2:
            raise ValueError(
                "surface_div expects a Jacobian and projector with at least 2 dims."
            )
        return oe.contract("...ij,...ji->...", P, Jv)

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=v.metadata)


def surface_curl_scalar(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Surface curl of a scalar field on a 3D surface.

    For a scalar field $u$ on a surface in $\mathbb{R}^3$, returns the tangential
    vector field

    $$
    \text{curl}_{\Gamma} u = n \times \nabla_{\Gamma} u.
    $$

    **Arguments:**

    - `u`: Scalar field on the surface.
    - `component`: Boundary `DomainComponent` used to supply the unit normal field.
    - `var`: Geometry label (must be 3D).
    - `mode`: Autodiff mode used by `surface_grad`.

    **Returns:**

    - A `DomainFunction` representing the tangential vector field $\text{curl}_\Gamma u$.
    """
    var = _resolve_var(u, var)
    _, var_dim = _factor_and_dim(u, var)
    if var_dim != 3:
        raise ValueError("surface_curl_scalar requires a 3D geometry variable.")

    n = component.normal(var=var)
    sg = surface_grad(u, component, var=var, mode=mode)
    joined = sg.domain.join(n.domain)
    sg2 = sg.promote(joined)
    n2 = n.promote(joined)

    deps = tuple(lbl for lbl in joined.labels if (lbl in sg2.deps) or (lbl in n2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    sg_pos = tuple(idx[lbl] for lbl in sg2.deps)
    n_pos = tuple(idx[lbl] for lbl in n2.deps)

    def _op(*args, key=None, **kwargs):
        nv = jnp.asarray(n2.func(*[args[i] for i in n_pos], key=key, **kwargs))
        nv = jax.lax.stop_gradient(nv)
        gv = jnp.asarray(sg2.func(*[args[i] for i in sg_pos], key=key, **kwargs))
        return jnp.cross(nv, gv)

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def surface_curl_vector(
    v: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Surface curl of a vector field on a 3D surface.

    For a vector field $v$ on a surface in $\mathbb{R}^3$, returns the scalar

    $$
    \text{curl}_{\Gamma} v = n \cdot (\nabla \times v).
    $$

    **Arguments:**

    - `v`: Vector field on the surface.
    - `component`: Boundary `DomainComponent` used to supply the unit normal field.
    - `var`: Geometry label (must be 3D).
    - `mode`: Autodiff mode used by `curl`.

    **Returns:**

    - A `DomainFunction` representing the scalar surface curl $\text{curl}_\Gamma v$.
    """
    var = _resolve_var(v, var)
    _, var_dim = _factor_and_dim(v, var)
    if var_dim != 3:
        raise ValueError("surface_curl_vector requires a 3D geometry variable.")

    n = component.normal(var=var)
    c = curl(v, var=var, mode=mode)
    joined = c.domain.join(n.domain)
    c2 = c.promote(joined)
    n2 = n.promote(joined)

    deps = tuple(lbl for lbl in joined.labels if (lbl in c2.deps) or (lbl in n2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    c_pos = tuple(idx[lbl] for lbl in c2.deps)
    n_pos = tuple(idx[lbl] for lbl in n2.deps)

    def _op(*args, key=None, **kwargs):
        nv = jnp.asarray(n2.func(*[args[i] for i in n_pos], key=key, **kwargs))
        nv = jax.lax.stop_gradient(nv)
        cv = jnp.asarray(c2.func(*[args[i] for i in c_pos], key=key, **kwargs))
        return jnp.sum(nv * cv, axis=-1)

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=v.metadata)


def laplace_beltrami(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str | None = None,
    curvature_aware: bool = False,
    variant: Literal["contraction", "divgrad"] | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Laplace–Beltrami operator on a boundary component.

    The Laplace–Beltrami operator $\Delta_{\Gamma}$ is the surface analogue of the
    Laplacian. Two common realizations are supported:

    - projection–contraction form (default): $\Delta_{\Gamma} u \approx \text{tr}(P\,\nabla^2 u\,P)$;
    - divergence-of-surface-gradient (via `variant="divgrad"`): $\Delta_{\Gamma} u =
      \nabla_{\Gamma}\cdot\nabla_{\Gamma}u$.

    The `curvature_aware=True` option selects the `divgrad` variant by default.

    **Arguments:**

    - `u`: Field to differentiate.
    - `component`: Boundary `DomainComponent` used to supply the unit normal field.
    - `var`: Geometry label.
    - `curvature_aware`: If `True`, defaults to `variant="divgrad"`.
    - `variant`: `"contraction"` (projection–contraction) or `"divgrad"`.
    - `mode`: Autodiff mode used by the underlying `grad`/`surface_grad`.

    **Returns:**

    - A `DomainFunction` representing $\Delta_\Gamma u$.
    """
    if variant == "divgrad" or (curvature_aware and variant is None):
        return laplace_beltrami_divgrad(u, component, var=var, mode=mode)

    var = _resolve_var(u, var)
    _, var_dim = _factor_and_dim(u, var)

    n = component.normal(var=var)
    joined = u.domain.join(n.domain)
    u2 = u.promote(joined)
    n2 = n.promote(joined)

    H = grad(grad(u2, var=var, mode=mode), var=var, mode=mode)

    deps = tuple(lbl for lbl in joined.labels if (lbl in H.deps) or (lbl in n2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    h_pos = tuple(idx[lbl] for lbl in H.deps)
    n_pos = tuple(idx[lbl] for lbl in n2.deps)

    def _op(*args, key=None, **kwargs):
        Hv = jnp.asarray(H.func(*[args[i] for i in h_pos], key=key, **kwargs))
        nv = jnp.asarray(n2.func(*[args[i] for i in n_pos], key=key, **kwargs))
        nv = jax.lax.stop_gradient(nv)
        if nv.shape[-1] != var_dim:
            raise ValueError(
                f"laplace_beltrami expected normal last axis {var_dim}, got {nv.shape[-1]}."
            )
        P = _proj_from_normals(nv)
        if Hv.ndim == P.ndim:
            return oe.contract("...ij,...jk,...ki->...", P, Hv, P)
        if Hv.ndim == P.ndim + 1:
            return oe.contract("...ij,...mjk,...ki->...m", P, Hv, P)
        raise ValueError(
            f"laplace_beltrami got incompatible ranks: H.ndim={Hv.ndim}, P.ndim={P.ndim}."
        )

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def laplace_beltrami_divgrad(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Laplace–Beltrami operator via surface divergence of the surface gradient.

    Implements

    $$
    \Delta_{\Gamma} u = \nabla_{\Gamma}\cdot(\nabla_{\Gamma}u).
    $$

    **Arguments:**

    - `u`: Field to differentiate.
    - `component`: Boundary `DomainComponent`.
    - `var`: Geometry label.
    - `mode`: Autodiff mode used by `surface_grad`/`surface_div`.

    **Returns:**

    - A `DomainFunction` representing $\Delta_\Gamma u$.
    """
    g = surface_grad(u, component, var=var, mode=mode)
    return surface_div(g, component, var=var, mode=mode)
