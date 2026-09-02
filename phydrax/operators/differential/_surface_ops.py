#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

import phydrax.ein as ein
from phydrax.domain import AbstractScalarDomain, DomainComponent, DomainFunction

from ...metrix import RiemannianMetric, tangent_projector_from_normal
from ._domain_ops import _factor_and_dim, _resolve_var, curl, grad


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
    if isinstance(factor, AbstractScalarDomain):
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
    if isinstance(factor, AbstractScalarDomain):
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
        P = tangent_projector_from_normal(nv)

        if gv.ndim == nv.ndim:
            return ein.contract("...ij,...j->...i", P, gv)
        if gv.ndim == nv.ndim + 1:
            return ein.contract("...md,...dj->...mj", gv, P)
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
    if isinstance(factor, AbstractScalarDomain):
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
        P = tangent_projector_from_normal(nv)
        if Jv.ndim < 2 or P.ndim < 2:
            raise ValueError(
                "surface_div expects a Jacobian and projector with at least 2 dims."
            )
        return ein.contract("...ij,...ji->...", P, Jv)

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
        cv = jnp.asarray(c2.func(*[args[i] for i in c_pos], key=key, **kwargs))
        return jnp.sum(nv * cv, axis=-1)

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=v.metadata)


def ambient_surface_hessian_trace(
    u: DomainFunction,
    component: DomainComponent,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Trace an ambient Hessian over a boundary tangent space.

    With outward unit normal $n$ and tangent projector $P=I-n\otimes n$, this
    operator computes

    $$
    \operatorname{tr}\left(P\,\nabla^2u\,P\right).
    $$

    This is an ambient, extension-dependent contraction. It agrees with the
    intrinsic Laplace--Beltrami operator only when the ambient extension of
    $u$ is compatible with the surface, including flat surfaces and
    closest-point extensions. Use `laplace_beltrami` with a
    `RiemannianMetric` for intrinsic curved-manifold calculus.

    **Arguments:**

    - `u`: Ambient field to differentiate.
    - `component`: Boundary component supplying the unit normal field.
    - `var`: Geometry label.
    - `mode`: Autodiff mode used to construct the ambient Hessian.

    **Returns:**

    - A `DomainFunction` representing the tangent-space Hessian trace.
    """
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
        if nv.shape[-1] != var_dim:
            raise ValueError(
                "ambient_surface_hessian_trace expected normal last axis "
                f"{var_dim}, got {nv.shape[-1]}."
            )
        P = tangent_projector_from_normal(nv)
        if Hv.ndim == P.ndim:
            return ein.contract("...ij,...jk,...ki->...", P, Hv, P)
        if Hv.ndim == P.ndim + 1:
            return ein.contract("...ij,...mjk,...ki->...m", P, Hv, P)
        raise ValueError(
            "ambient_surface_hessian_trace got incompatible ranks: "
            f"H.ndim={Hv.ndim}, P.ndim={P.ndim}."
        )

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def laplace_beltrami(
    u: DomainFunction,
    metric: RiemannianMetric,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Apply the intrinsic Laplace--Beltrami operator.

    For a scalar field $u$ and Riemannian metric $g$, this computes

    $$
    \Delta_g u
    = \frac{1}{\sqrt{\lvert g\rvert}}
      \partial_i\left(\sqrt{\lvert g\rvert}\,g^{ij}\partial_j u\right).
    $$

    Boundary-component normal projections are intentionally not accepted here;
    use `ambient_surface_hessian_trace` for the extension-dependent ambient
    contraction.
    """
    if not isinstance(metric, RiemannianMetric):
        raise TypeError(
            "laplace_beltrami requires a RiemannianMetric; "
            "use ambient_surface_hessian_trace for a boundary component."
        )
    from ._riemannian_ops import intrinsic_laplace_beltrami

    return intrinsic_laplace_beltrami(u, metric, var=var, mode=mode)
