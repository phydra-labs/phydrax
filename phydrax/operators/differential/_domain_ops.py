#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import ArrayLike

from ...domain._base import _AbstractGeometry
from ...domain._domain import RelabeledDomain
from ...domain._function import DomainFunction
from ...domain._scalar import _AbstractScalarDomain
from ._jet import jet_d1_d2, jet_dn, jet_dn_multi


def _unwrap_factor(factor: object, /) -> object:
    return factor.base if isinstance(factor, RelabeledDomain) else factor


def _resolve_var(u: DomainFunction, var: str | None, /) -> str:
    if var is not None:
        if var not in u.domain.labels:
            raise ValueError(f"Unknown var {var!r}; expected one of {u.domain.labels}.")
        return var

    differentiable: list[str] = []
    for lbl in u.domain.labels:
        factor = _unwrap_factor(u.domain.factor(lbl))
        if isinstance(factor, (_AbstractGeometry, _AbstractScalarDomain)):
            differentiable.append(lbl)

    if len(differentiable) != 1:
        raise ValueError(
            "var=None is only valid when there is exactly one differentiable "
            f"variable in the domain; found {tuple(differentiable)!r}."
        )
    return differentiable[0]


def _factor_and_dim(
    u: DomainFunction, var: str, /
) -> tuple[_AbstractGeometry | _AbstractScalarDomain, int]:
    factor = _unwrap_factor(u.domain.factor(var))
    if isinstance(factor, _AbstractGeometry):
        return factor, int(factor.var_dim)
    if isinstance(factor, _AbstractScalarDomain):
        return factor, 1
    raise TypeError(
        f"Differential operators are not defined for var={var!r} with domain factor {type(factor).__name__}."
    )


def _coord_axis_position(
    args: Sequence[object], /, *, arg_index: int, coord_index: int
) -> int:
    """Axis position induced by coord-separable broadcasting in Domain.Function wrappers."""
    axis_pos: dict[tuple[int, int], int] = {}
    total_axes = 0
    for i, arg in enumerate(args):
        if not isinstance(arg, tuple):
            continue
        coords = arg
        for j in range(len(coords)):
            axis_pos[(i, j)] = total_axes
            total_axes += 1
    pos = axis_pos.get((int(arg_index), int(coord_index)))
    if pos is None:
        raise ValueError("Could not resolve coord-separable axis position.")
    return int(pos)


def _fd_first_derivative(
    y: jax.Array, /, *, dx: jax.Array, axis: int, periodic: bool
) -> jax.Array:
    dx_ = jnp.asarray(dx, dtype=float).reshape(())
    if periodic:
        return (jnp.roll(y, -1, axis=axis) - jnp.roll(y, 1, axis=axis)) / (2.0 * dx_)

    y0 = jnp.moveaxis(y, axis, 0)
    n = y0.shape[0]
    if n < 2:
        return jnp.zeros_like(y)
    out0 = jnp.zeros_like(y0)
    out0 = out0.at[1:-1].set((y0[2:] - y0[:-2]) / (2.0 * dx_))
    out0 = out0.at[0].set((y0[1] - y0[0]) / dx_)
    out0 = out0.at[-1].set((y0[-1] - y0[-2]) / dx_)
    return jnp.moveaxis(out0, 0, axis)


def _fd_nth_derivative(
    y: jax.Array, /, *, dx: jax.Array, axis: int, order: int, periodic: bool
) -> jax.Array:
    out = y
    for _ in range(int(order)):
        out = _fd_first_derivative(out, dx=dx, axis=axis, periodic=periodic)
    return out


def _barycentric_diff_matrix(x: jax.Array, /) -> jax.Array:
    x1 = jnp.asarray(x, dtype=float).reshape((-1,))
    n = int(x1.shape[0])
    if n < 2:
        return jnp.zeros((n, n), dtype=float)
    diff = x1[:, None] - x1[None, :]
    diff_safe = diff + jnp.eye(n, dtype=float)
    w = 1.0 / jnp.prod(diff_safe, axis=1)
    D = (w[None, :] / w[:, None]) / diff_safe
    D = D - jnp.diag(jnp.diag(D))
    D = D.at[jnp.arange(n), jnp.arange(n)].set(-jnp.sum(D, axis=1))
    return D


def _poly_nth_derivative(
    y: jax.Array, x: jax.Array, /, *, axis: int, order: int
) -> jax.Array:
    out = y
    for _ in range(int(order)):
        D = _barycentric_diff_matrix(x)
        out0 = jnp.moveaxis(out, axis, 0)
        n = int(out0.shape[0])
        flat = out0.reshape((n, -1))
        dflat = D @ flat
        out = jnp.moveaxis(dflat.reshape(out0.shape), 0, axis)
    return out


def _fourier_nth_derivative(
    y: jax.Array, x: jax.Array, /, *, axis: int, order: int
) -> jax.Array:
    x1 = jnp.asarray(x, dtype=float).reshape((-1,))
    n = int(x1.shape[0])
    if n < 2:
        return jnp.zeros_like(y)

    dx = x1[1] - x1[0]
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)
    shape = [1] * y.ndim
    shape[int(axis)] = n
    k = k.reshape(tuple(shape))

    yhat = jnp.fft.fft(y, axis=axis)
    mult = (1j * k) ** int(order)
    dy = jnp.fft.ifft(mult * yhat, axis=axis)
    if not jnp.iscomplexobj(y):
        dy = jnp.real(dy)
    return dy


def _cosine_nth_derivative(
    y: jax.Array, x: jax.Array, /, *, axis: int, order: int
) -> jax.Array:
    x1 = jnp.asarray(x, dtype=float).reshape((-1,))
    n = int(x1.shape[0])
    if n < 2:
        return jnp.zeros_like(y)

    dx = x1[1] - x1[0]
    m = 2 * (n - 1)

    y0 = jnp.moveaxis(y, axis, 0)
    y_ext = jnp.concatenate([y0, y0[-2:0:-1]], axis=0)

    k = 2.0 * jnp.pi * jnp.fft.fftfreq(m, d=dx)
    shape = [m] + [1] * (y_ext.ndim - 1)
    k = k.reshape(tuple(shape))

    yhat = jnp.fft.fft(y_ext, axis=0)
    mult = (1j * k) ** int(order)
    dy_ext = jnp.fft.ifft(mult * yhat, axis=0)[:n]

    if not jnp.iscomplexobj(y):
        dy_ext = jnp.real(dy_ext)
    return jnp.moveaxis(dy_ext, 0, axis)


def _sine_nth_derivative(
    y: jax.Array, x: jax.Array, /, *, axis: int, order: int
) -> jax.Array:
    x1 = jnp.asarray(x, dtype=float).reshape((-1,))
    n = int(x1.shape[0])
    if n < 2:
        return jnp.zeros_like(y)

    dx = x1[1] - x1[0]
    m = 2 * n

    y0 = jnp.moveaxis(y, axis, 0)
    y_ext = jnp.concatenate([y0, -y0[::-1]], axis=0)

    k = 2.0 * jnp.pi * jnp.fft.fftfreq(m, d=dx)
    shape = [m] + [1] * (y_ext.ndim - 1)
    k = k.reshape(tuple(shape))

    yhat = jnp.fft.fft(y_ext, axis=0)
    mult = (1j * k) ** int(order)
    dy_ext = jnp.fft.ifft(mult * yhat, axis=0)[:n]

    if not jnp.iscomplexobj(y):
        dy_ext = jnp.real(dy_ext)
    return jnp.moveaxis(dy_ext, 0, axis)


def _basis_nth_derivative(
    y: jax.Array,
    x: jax.Array,
    /,
    *,
    axis: int,
    order: int,
    basis: Literal["poly", "fourier", "sine", "cosine"],
) -> jax.Array:
    if int(order) == 0:
        return y
    if basis == "fourier":
        return _fourier_nth_derivative(y, x, axis=axis, order=order)
    if basis == "cosine":
        return _cosine_nth_derivative(y, x, axis=axis, order=order)
    if basis == "sine":
        return _sine_nth_derivative(y, x, axis=axis, order=order)
    return _poly_nth_derivative(y, x, axis=axis, order=order)


def grad(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Gradient/Jacobian of `u` with respect to a labeled variable.

    For a geometry variable $x\in\mathbb{R}^d$ this constructs $\nabla_x u$.
    Concretely:

    If $u:\Omega\to\mathbb{R}$ is scalar-valued, then `grad(u)` returns a vector field
    in $\mathbb{R}^d$:

    $$
    \nabla_x u = \left(\frac{\partial u}{\partial x_1},\dots,\frac{\partial u}{\partial x_d}\right).
    $$

    If $u:\Omega\to\mathbb{R}^m$ is vector-valued, then `grad(u)` returns the Jacobian
    $J\in\mathbb{R}^{m\times d}$ with entries $J_{ij}=\partial u_i/\partial x_j$.

    For a scalar domain variable (e.g. time $t$), this reduces to the partial derivative
    $\partial u/\partial t$.

    **Arguments:**

    - `u`: Input function.
    - `var`: Variable label to differentiate with respect to. If `None`, then the
      domain must have exactly one differentiable variable.
    - `mode`: Autodiff mode: `"reverse"` uses `jax.jacrev`, `"forward"` uses `jax.jacfwd`.
    - `backend`: Differentiation backend.
      - `"ad"`: autodiff (works for point batches and coord-separable batches).
      - `"fd"`: finite differences on coord-separable grids (falls back to `"ad"`).
      - `"basis"`: spectral/barycentric methods on coord-separable grids (falls back to `"ad"`).
    - `basis`: Basis method used when `backend="basis"`.
    - `periodic`: Whether to treat the differentiated axis as periodic (used by `backend="fd"`).

    **Notes:**

    - Coord-separable inputs (tuples of 1D coordinate arrays) are handled by taking JVPs
      along each coordinate axis and stacking.

    **Returns:**

    - A `DomainFunction` representing the gradient/Jacobian. For geometry variables,
      the derivative axis is appended on the right:
      - scalar $u$: `grad(u)` has trailing shape `(..., d)`;
      - vector-valued $u$ with trailing size $m$: trailing shape `(..., m, d)`.

    **Example:**

    ```python
    import phydrax as phx

    geom = phx.domain.Interval1d(0.0, 1.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    du = phx.operators.grad(u, var="x")
    ```
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)

    jac = jax.jacrev if mode == "reverse" else jax.jacfwd

    if var not in u.deps:

        def _zero(*args, key=None, **kwargs):
            y = jnp.asarray(u.func(*args, key=key, **kwargs))
            if isinstance(factor, _AbstractScalarDomain):
                return jnp.zeros_like(y)
            return jnp.zeros(y.shape + (var_dim,), dtype=y.dtype)

        return DomainFunction(
            domain=u.domain, deps=u.deps, func=_zero, metadata=u.metadata
        )

    idx = u.deps.index(var)

    def _grad(*args, key=None, **kwargs):
        x0 = args[idx]

        def f(xi):
            call_args = tuple(xi if i == idx else args[i] for i in range(len(args)))
            return u.func(*call_args, key=key, **kwargs)

        if isinstance(x0, tuple) and backend != "ad":
            coords = tuple(jnp.asarray(xi) for xi in x0)
            if len(coords) != var_dim:
                raise ValueError(
                    f"coord-separable grad expects {var_dim} coordinate arrays for var={var!r}."
                )
            if not all(xi.ndim == 1 for xi in coords):
                raise ValueError(
                    "coord-separable grad requires a tuple of 1D coordinate arrays."
                )

            y = jnp.asarray(u.func(*args, key=key, **kwargs))
            terms = []
            for i in range(var_dim):
                axis = _coord_axis_position(args, arg_index=idx, coord_index=i)
                if backend == "fd":
                    dx = (
                        coords[i][1] - coords[i][0]
                        if coords[i].shape[0] > 1
                        else jnp.asarray(1.0, dtype=coords[i].dtype)
                    )
                    di = _fd_nth_derivative(
                        y, dx=dx, axis=axis, order=1, periodic=bool(periodic)
                    )
                else:
                    di = _basis_nth_derivative(
                        y, coords[i], axis=axis, order=1, basis=basis
                    )
                terms.append(di)
            if var_dim == 1:
                if isinstance(factor, _AbstractScalarDomain):
                    return terms[0]
                return terms[0][..., None]
            return jnp.stack(terms, axis=-1)

        if isinstance(x0, tuple):
            coords = tuple(jnp.asarray(xi) for xi in x0)
            if len(coords) != var_dim:
                raise ValueError(
                    f"coord-separable grad expects {var_dim} coordinate arrays for var={var!r}."
                )
            if not all(xi.ndim == 1 for xi in coords):
                raise ValueError(
                    "coord-separable grad requires a tuple of 1D coordinate arrays."
                )

            jvps = []
            for i in range(var_dim):

                def f_i(coord, *, i=i):
                    call_args = list(args)
                    call_args[idx] = coords[:i] + (coord,) + coords[i + 1 :]
                    return u.func(*call_args, key=key, **kwargs)

                jvp_i = jax.jvp(f_i, (coords[i],), (jnp.ones_like(coords[i]),))[1]
                jvps.append(jvp_i)
            if var_dim == 1:
                if isinstance(factor, _AbstractScalarDomain):
                    return jvps[0]
                return jvps[0][..., None]
            return jnp.stack(jvps, axis=-1)

        y0 = f(x0)
        if jnp.iscomplexobj(y0):
            jac_r = jac(lambda xi: jnp.real(f(xi)))(x0)
            jac_i = jac(lambda xi: jnp.imag(f(xi)))(x0)
            return jac_r + 1j * jac_i
        return jac(f)(x0)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_grad, metadata=u.metadata)


def hessian(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    r"""Hessian of `u` with respect to a labeled variable.

    For a geometry variable $x\in\mathbb{R}^d$ and scalar-valued $u$, this returns the
    matrix of second derivatives

    $$
    H_{ij}(x) = \frac{\partial^2 u}{\partial x_i\,\partial x_j}.
    $$

    For vector-valued $u:\Omega\to\mathbb{R}^m$, the Hessian is taken componentwise,
    producing a trailing shape `(..., m, d, d)`.

    For a scalar variable (e.g. time $t$) this reduces to $\partial^2 u/\partial t^2$.

    **Arguments:**

    - `u`: Input function.
    - `var`: Variable label to differentiate with respect to. If `None`, the domain
      must have exactly one differentiable variable.

    **Returns:**

    - A `DomainFunction` representing the Hessian/second derivative.

    **Example:**

    ```python
    import phydrax as phx

    geom = phx.domain.Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2 + x[1] ** 2

    H = phx.operators.hessian(u, var="x")
    ```
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)

    if var not in u.deps:

        def _zero(*args, key=None, **kwargs):
            y = jnp.asarray(u.func(*args, key=key, **kwargs))
            if isinstance(factor, _AbstractScalarDomain):
                return jnp.zeros_like(y)
            return jnp.zeros(y.shape + (var_dim, var_dim), dtype=y.dtype)

        return DomainFunction(
            domain=u.domain, deps=u.deps, func=_zero, metadata=u.metadata
        )

    idx = u.deps.index(var)

    def _hess(*args, key=None, **kwargs):
        x0 = args[idx]

        def f(xi):
            call_args = tuple(xi if i == idx else args[i] for i in range(len(args)))
            return u.func(*call_args, key=key, **kwargs)

        if isinstance(x0, tuple):
            coords = tuple(jnp.asarray(xi) for xi in x0)
            if len(coords) != var_dim:
                raise ValueError(
                    f"coord-separable hessian expects {var_dim} coordinate arrays for var={var!r}."
                )
            if not all(xi.ndim == 1 for xi in coords):
                raise ValueError(
                    "coord-separable hessian requires a tuple of 1D coordinate arrays."
                )

            ones = tuple(jnp.ones_like(xi) for xi in coords)

            if var_dim == 1:

                def f_0(coord):
                    call_args = list(args)
                    call_args[idx] = (coord,)
                    return u.func(*call_args, key=key, **kwargs)

                v0 = ones[0]
                return jax.jvp(
                    lambda c: jax.jvp(f_0, (c,), (v0,))[1], (coords[0],), (v0,)
                )[1]

            rows = []
            for i in range(var_dim):
                row = []
                v_i = ones[i]

                def f_i(coord, *, i=i):
                    call_args = list(args)
                    call_args[idx] = coords[:i] + (coord,) + coords[i + 1 :]
                    return u.func(*call_args, key=key, **kwargs)

                for j in range(var_dim):
                    if i == j:
                        d2 = jax.jvp(
                            lambda c: jax.jvp(f_i, (c,), (v_i,))[1],
                            (coords[i],),
                            (v_i,),
                        )[1]
                        row.append(d2)
                        continue

                    def df_dxi(coord_j, *, i=i, j=j):
                        def f_ij(coord_i):
                            call_args = list(args)
                            coords_mod = list(coords)
                            coords_mod[j] = coord_j
                            coords_mod[i] = coord_i
                            call_args[idx] = tuple(coords_mod)
                            return u.func(*call_args, key=key, **kwargs)

                        return jax.jvp(f_ij, (coords[i],), (v_i,))[1]

                    d2 = jax.jvp(df_dxi, (coords[j],), (ones[j],))[1]
                    row.append(d2)

                rows.append(jnp.stack(row, axis=-1))
            return jnp.stack(rows, axis=-2)

        y0 = f(x0)
        if jnp.iscomplexobj(y0):
            hess_r = jax.hessian(lambda xi: jnp.real(f(xi)))(x0)
            hess_i = jax.hessian(lambda xi: jnp.imag(f(xi)))(x0)
            return hess_r + 1j * hess_i
        return jax.hessian(f)(x0)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_hess, metadata=u.metadata)


def directional_derivative(
    u: DomainFunction,
    v: DomainFunction,
    /,
    *,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Directional derivative of `u` along a direction field `v`.

    For a geometry variable $x\in\mathbb{R}^d$, this computes

    $$
    D_v u \;=\; v\cdot\nabla_x u,
    $$

    where $v=v(x)$ is a vector field and $\nabla_x u$ is computed by `grad`.

    If $u$ is vector- or tensor-valued, the directional derivative is applied
    componentwise, i.e. $D_v u$ has the same value shape as $u$.

    **Arguments:**

    - `u`: Input function.
    - `v`: Direction field (must be vector-valued with trailing size $d$ for `var`).
    - `var`: Geometry label to differentiate with respect to (must be a geometry variable).
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `grad(u, ...)`.

    **Returns:**

    - A `DomainFunction` representing $D_v u$.
    """
    factor, _ = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "directional_derivative(var=...) requires a geometry variable, not a scalar variable."
        )

    joined = u.domain.join(v.domain)
    u2 = u.promote(joined)
    v2 = v.promote(joined)

    g = grad(u2, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)

    deps = tuple(lbl for lbl in joined.labels if (lbl in g.deps) or (lbl in v2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    g_pos = tuple(idx[lbl] for lbl in g.deps)
    v_pos = tuple(idx[lbl] for lbl in v2.deps)

    def _dd(*args, key=None, **kwargs):
        g_args = [args[i] for i in g_pos]
        v_args = [args[i] for i in v_pos]
        gu = jnp.asarray(g.func(*g_args, key=key, **kwargs))
        vv = jnp.asarray(v2.func(*v_args, key=key, **kwargs))
        return jnp.sum(gu * vv, axis=-1)

    return DomainFunction(domain=joined, deps=deps, func=_dd, metadata=g.metadata)


def div(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Divergence of a vector field.

    For a vector field $u:\Omega\to\mathbb{R}^d$, returns

    $$
    \nabla\cdot u = \sum_{i=1}^{d}\frac{\partial u_i}{\partial x_i}.
    $$

    The input is expected to be vector-valued with last axis size $d$ (the geometry
    dimension for `var`). If the value has additional leading axes (e.g. multiple
    vector fields stacked), divergence is applied componentwise over those axes.

    **Arguments:**

    - `u`: Vector field to differentiate.
    - `var`: Geometry variable label. If `None`, the domain must have exactly one
      geometry variable.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `grad(u, ...)`.

    **Returns:**

    - A `DomainFunction` representing $\nabla\cdot u$.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "div(var=...) requires a geometry variable, not a scalar variable."
        )

    g = grad(u, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)

    def _div(*args, key=None, **kwargs):
        jac = jnp.asarray(g.func(*args, key=key, **kwargs))
        if jac.ndim < 2:
            raise ValueError(
                "div expects a field with at least one trailing vector axis."
            )
        if jac.shape[-2] != var_dim:
            raise ValueError(
                f"div expects the field to have last axis size {var_dim} (vector-valued)."
            )
        return jnp.trace(jac, axis1=-2, axis2=-1)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_div, metadata=u.metadata)


def curl(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Curl of a 3D vector field.
    
    For $u:\Omega\to\mathbb{R}^3$, returns $\nabla\times u$:
    
    $$
    \nabla\times u =
    \begin{pmatrix}
      \partial_y u_z - \partial_z u_y \\
      \partial_z u_x - \partial_x u_z \\
      \partial_x u_y - \partial_y u_x
    \end{pmatrix}.
    $$
    
    **Arguments:**
    
    - `u`: Vector field (must be 3D-valued).
    - `var`: Geometry variable label. If `None`, the domain must have exactly one
      geometry variable.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `grad(u, ...)`.
    
    **Returns:**
    
    - A `DomainFunction` representing $\nabla\times u$ (vector-valued with trailing size 3).
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "curl(var=...) requires a geometry variable, not a scalar variable."
        )
    if var_dim != 3:
        raise ValueError(
            f"curl(var=...) requires a 3D geometry variable, got var_dim={var_dim}."
        )

    g = grad(u, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)

    def _curl(*args, key=None, **kwargs):
        jac = jnp.asarray(g.func(*args, key=key, **kwargs))
        if jac.ndim < 2 or jac.shape[-2:] != (3, 3):
            raise ValueError(
                f"curl expects grad(u) to have trailing shape (3, 3), got {jac.shape[-2:]}."
            )
        curl_x = jac[..., 2, 1] - jac[..., 1, 2]
        curl_y = jac[..., 0, 2] - jac[..., 2, 0]
        curl_z = jac[..., 1, 0] - jac[..., 0, 1]
        return jnp.stack((curl_x, curl_y, curl_z), axis=-1)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_curl, metadata=u.metadata)


def div_tensor(
    T: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Divergence of a second-order tensor field.

    For a tensor field $T:\Omega\to\mathbb{R}^{d\times d}$, returns the vector field

    $$
    (\nabla\cdot T)_j = \sum_{i=1}^{d}\frac{\partial T_{ij}}{\partial x_i}.
    $$

    **Arguments:**

    - `T`: Tensor field with trailing shape `(d, d)` (optionally with extra leading value axes).
    - `var`: Geometry variable label. If `None`, the domain must have exactly one
      geometry variable.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `grad(T, ...)`.

    **Returns:**

    - A `DomainFunction` representing $\nabla\cdot T$ (vector-valued with trailing size `d`).
    """
    var = _resolve_var(T, var)
    factor, var_dim = _factor_and_dim(T, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "div_tensor(var=...) requires a geometry variable, not a scalar variable."
        )

    gT = grad(T, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)

    def _divT(*args, key=None, **kwargs):
        g = jnp.asarray(gT.func(*args, key=key, **kwargs))
        if g.ndim < 3 or g.shape[-3:] != (var_dim, var_dim, var_dim):
            raise ValueError(
                "div_tensor expects grad(T) to have trailing shape "
                f"({var_dim}, {var_dim}, {var_dim}), got {g.shape[-3:]}."
            )
        return oe.contract("...iji->...j", g)

    return DomainFunction(domain=T.domain, deps=T.deps, func=_divT, metadata=T.metadata)


def cauchy_strain(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Cauchy (small) strain tensor for a displacement/velocity field.

    For a vector field $u:\Omega\to\mathbb{R}^d$, the small-strain tensor is

    $$
    \varepsilon(u) = \tfrac12\left(\nabla u + (\nabla u)^\top\right).
    $$

    **Arguments:**

    - `u`: Displacement/velocity field (vector-valued with trailing size $d$).
    - `var`: Geometry label to differentiate with respect to.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `grad(u, ...)`.

    **Returns:**

    - A `DomainFunction` representing $\varepsilon(u)$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    factor, _ = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "cauchy_strain(var=...) requires a geometry variable, not a scalar variable."
        )
    G = grad(u, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)
    return 0.5 * (G + G.T)


def strain_rate(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Alias for `cauchy_strain` (common notation $D(u)$ for velocity fields)."""
    return cauchy_strain(
        u,
        var=var,
        mode=mode,
        backend=backend,
        basis=basis,
        periodic=periodic,
    )


def strain_rate_magnitude(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Magnitude of the strain-rate tensor.

    With $D(u)=\tfrac12(\nabla u+(\nabla u)^\top)$, this returns the scalar field

    $$
    \|D\| = \sqrt{2\,D{:}D} = \sqrt{2\sum_{i,j} D_{ij}^2}.
    $$

    **Arguments:**

    - `u`: Velocity field.
    - `var`: Geometry label to differentiate with respect to.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `strain_rate(u, ...)`.

    **Returns:**

    - A `DomainFunction` representing $\|D(u)\|$.
    """
    D = strain_rate(
        u, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic
    )

    def _mag(*args, key=None, **kwargs):
        d = jnp.asarray(D.func(*args, key=key, **kwargs))
        return jnp.sqrt(2.0 * jnp.sum(d * d, axis=(-2, -1)))

    return DomainFunction(domain=D.domain, deps=D.deps, func=_mag, metadata=D.metadata)


def _trace_last2(T: DomainFunction, /, *, keepdims: bool = False) -> DomainFunction:
    keep = bool(keepdims)

    def _tr(*args, key=None, **kwargs):
        x = jnp.asarray(T.func(*args, key=key, **kwargs))
        tr = jnp.trace(x, axis1=-2, axis2=-1)
        if keep:
            return tr[..., None, None]
        return tr

    return DomainFunction(domain=T.domain, deps=T.deps, func=_tr, metadata=T.metadata)


def cauchy_stress(
    u: DomainFunction,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Linear-elastic Cauchy stress (Hooke's law).

    $$
    \sigma(u) = 2\mu\,\varepsilon(u) + \lambda\,\operatorname{tr}(\varepsilon(u))\,I.
    $$

    Here $\varepsilon(u)=\tfrac12(\nabla u + (\nabla u)^\top)$ is the small-strain tensor,
    and $\lambda,\mu$ are Lamé parameters (which may be constants or `DomainFunction`s).

    **Arguments:**

    - `u`: Displacement field (vector-valued).
    - `lambda_`: First Lamé parameter $\lambda$.
    - `mu`: Shear modulus $\mu$.
    - `var`: Geometry label to differentiate with respect to.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `cauchy_strain(u, ...)`.

    **Returns:**

    - A `DomainFunction` representing $\sigma(u)$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "cauchy_stress(var=...) requires a geometry variable, not a scalar variable."
        )

    mu_fn = _as_domain_function(u, mu)
    lambda_fn = _as_domain_function(u, lambda_)
    strain = cauchy_strain(
        u,
        var=var,
        mode=mode,
        backend=backend,
        basis=basis,
        periodic=periodic,
    )
    tr = _trace_last2(strain, keepdims=True)
    I = jnp.eye(var_dim)
    return 2.0 * mu_fn * strain + lambda_fn * tr * I


def div_cauchy_stress(
    u: DomainFunction,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Divergence of the linear-elastic Cauchy stress.

    Computes the vector field

    $$
    \nabla\cdot\sigma(u),
    $$

    where $\sigma(u)$ is given by `cauchy_stress`.

    **Arguments:**

    - `u`, `lambda_`, `mu`: As in `cauchy_stress`.
    - `var`: Geometry label to differentiate with respect to.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `cauchy_stress` and `div_tensor`.

    **Returns:**

    - A `DomainFunction` representing $\nabla\cdot\sigma(u)$ (vector-valued).
    """
    var = _resolve_var(u, var)
    sigma = cauchy_stress(
        u,
        lambda_=lambda_,
        mu=mu,
        var=var,
        mode=mode,
        backend=backend,
        basis=basis,
        periodic=periodic,
    )
    return div_tensor(
        sigma,
        var=var,
        mode=mode,
        backend=backend,
        basis=basis,
        periodic=periodic,
    )


def viscous_stress(
    u: DomainFunction,
    /,
    *,
    mu: DomainFunction | ArrayLike,
    lambda_: DomainFunction | ArrayLike | None = None,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Newtonian viscous stress tensor for a velocity field.

    Uses the symmetric strain-rate tensor

    $$
    D(u) = \tfrac12(\nabla u + \nabla u^\top),
    $$

    and returns

    $$
    \tau(u) = 2\mu\,D(u) + \lambda\,\text{tr}(D(u))\,I.
    $$

    If `lambda_` is not provided, uses Stokes' hypothesis $\lambda=-\tfrac{2}{3}\mu$.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "viscous_stress(var=...) requires a geometry variable, not a scalar variable."
        )

    mu_fn = _as_domain_function(u, mu)
    if lambda_ is None:
        lambda_fn: DomainFunction | None = None
    else:
        lambda_fn = _as_domain_function(u, lambda_)

    G = grad(u, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)
    D = 0.5 * (G + G.T)
    trD = _trace_last2(D, keepdims=True)
    I = jnp.eye(var_dim)

    if lambda_fn is None:
        lam = (-2.0 / 3.0) * mu_fn
    else:
        lam = lambda_fn
    return 2.0 * mu_fn * D + lam * trD * I


def navier_stokes_stress(
    u: DomainFunction,
    p: DomainFunction | ArrayLike,
    /,
    *,
    mu: DomainFunction | ArrayLike,
    lambda_: DomainFunction | ArrayLike | None = None,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Cauchy stress for incompressible/compressible Navier–Stokes.

    Given a velocity field $u$ and pressure $p$, this returns

    $$
    \sigma(u,p) = \tau(u) - p\,I,
    $$

    where $\tau(u)$ is the (Newtonian) viscous stress returned by `viscous_stress`.

    **Arguments:**

    - `u`: Velocity field.
    - `p`: Pressure (scalar field or constant).
    - `mu`, `lambda_`: Viscosity parameters for `viscous_stress` (with Stokes'
      hypothesis used when `lambda_` is omitted).
    - `var`: Geometry label to differentiate with respect to.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `viscous_stress`.

    **Returns:**

    - A `DomainFunction` representing $\sigma(u,p)$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    _, var_dim = _factor_and_dim(u, var)

    tau = viscous_stress(
        u,
        mu=mu,
        lambda_=lambda_,
        var=var,
        mode=mode,
        backend=backend,
        basis=basis,
        periodic=periodic,
    )
    p_fn = _as_domain_function(u, p)
    I = jnp.eye(var_dim)
    return tau - p_fn * I


def navier_stokes_divergence(
    u: DomainFunction,
    p: DomainFunction | ArrayLike,
    /,
    *,
    mu: DomainFunction | ArrayLike,
    lambda_: DomainFunction | ArrayLike | None = None,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Divergence of the Navier–Stokes stress.

    Computes

    $$
    \nabla\cdot(\tau(u) - p I),
    $$

    where `navier_stokes_stress(u, p, ...)` returns the stress tensor.

    **Arguments:**

    - `u`, `p`, `mu`, `lambda_`: As in `navier_stokes_stress`.
    - `var`: Geometry label to differentiate with respect to.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `navier_stokes_stress` and `div_tensor`.

    **Returns:**

    - A `DomainFunction` representing $\nabla\cdot(\tau(u) - p I)$ (vector-valued).
    """
    var = _resolve_var(u, var)
    sig = navier_stokes_stress(
        u,
        p,
        mu=mu,
        lambda_=lambda_,
        var=var,
        mode=mode,
        backend=backend,
        basis=basis,
        periodic=periodic,
    )
    return div_tensor(
        sig, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic
    )


def laplacian(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "jet", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Laplacian of a scalar field with respect to a geometry variable.

    For $u:\Omega\to\mathbb{R}$ and $x\in\mathbb{R}^d$, this returns

    $$
    \Delta u \;=\; \nabla\cdot\nabla u \;=\; \sum_{i=1}^{d}\frac{\partial^2 u}{\partial x_i^2}.
    $$

    **Arguments:**

    - `u`: Input function (typically scalar-valued).
    - `var`: Geometry label to differentiate with respect to. If `None`, the domain
      must have exactly one geometry variable.
    - `mode`: Autodiff mode for Jacobian/Hessian construction when applicable.
    - `backend`: Differentiation backend:
      - `"ad"`: uses nested autodiff (`grad(grad(u))`).
      - `"jet"`: uses JAX Jet expansions for second directional derivatives.
      - `"fd"`: uses finite differences on coord-separable grids (falls back to `"ad"`).
      - `"basis"`: uses spectral/barycentric methods on coord-separable grids (falls back to `"ad"`).
    - `basis`: Basis method used when `backend="basis"`.
    - `periodic`: Whether to treat differentiated axes as periodic (used by `backend="fd"`).

    **Returns:**

    - A `DomainFunction` representing $\Delta u$.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "laplacian(var=...) requires a geometry variable, not a scalar variable."
        )

    if var not in u.deps:

        def _zero(*args, key=None, **kwargs):
            y = jnp.asarray(u.func(*args, key=key, **kwargs))
            return jnp.zeros_like(y)

        return DomainFunction(
            domain=u.domain, deps=u.deps, func=_zero, metadata=u.metadata
        )

    if backend == "fd" or backend == "basis":
        total = None
        for i in range(var_dim):
            d2 = partial_n(
                u,
                var=var,
                axis=i,
                order=2,
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            )
            total = d2 if total is None else (total + d2)
        assert total is not None
        return total

    if backend == "ad":
        hess = grad(grad(u, var=var, mode=mode), var=var, mode=mode)

        def _lap(*args, key=None, **kwargs):
            x0 = args[u.deps.index(var)]
            if isinstance(x0, tuple):
                coords = tuple(jnp.asarray(xi) for xi in x0)
                if len(coords) != var_dim:
                    raise ValueError(
                        f"coord-separable laplacian expects {var_dim} coordinate arrays for var={var!r}."
                    )
                if not all(xi.ndim == 1 for xi in coords):
                    raise ValueError(
                        "coord-separable laplacian requires a tuple of 1D coordinate arrays."
                    )

                terms = []
                for i in range(var_dim):

                    def f_i(coord, *, i=i):
                        call_args = list(args)
                        call_args[u.deps.index(var)] = (
                            coords[:i] + (coord,) + coords[i + 1 :]
                        )
                        return u.func(*call_args, key=key, **kwargs)

                    v = jnp.ones_like(coords[i])
                    d2 = jax.jvp(
                        lambda c: jax.jvp(f_i, (c,), (v,))[1], (coords[i],), (v,)
                    )[1]
                    terms.append(d2)
                out = terms[0]
                for t in terms[1:]:
                    out = out + t
                return out

            h = jnp.asarray(hess.func(*args, key=key, **kwargs))
            if h.ndim < 2:
                raise ValueError("laplacian expects a Hessian with two trailing axes.")
            if h.shape[-2] != var_dim or h.shape[-1] != var_dim:
                raise ValueError(
                    f"laplacian expects Hessian trailing axes ({var_dim}, {var_dim})."
                )
            return jnp.trace(h, axis1=-2, axis2=-1)

        return DomainFunction(
            domain=u.domain, deps=u.deps, func=_lap, metadata=u.metadata
        )

    if backend != "jet":
        raise ValueError("backend must be 'ad', 'jet', 'fd', or 'basis'.")

    idx = u.deps.index(var)

    def _lap(*args, key=None, **kwargs):
        x0 = args[idx]

        def f(xi):
            call_args = tuple(xi if i == idx else args[i] for i in range(len(args)))
            return u.func(*call_args, key=key, **kwargs)

        if isinstance(x0, tuple):
            coords = tuple(jnp.asarray(xi) for xi in x0)
            if len(coords) != var_dim:
                raise ValueError(
                    f"coord-separable laplacian expects {var_dim} coordinate arrays for var={var!r}."
                )
            if not all(xi.ndim == 1 for xi in coords):
                raise ValueError(
                    "coord-separable laplacian requires a tuple of 1D coordinate arrays."
                )

            total = None
            for i in range(var_dim):

                def f_i(coord, *, i=i):
                    call_args = list(args)
                    call_args[idx] = coords[:i] + (coord,) + coords[i + 1 :]
                    return u.func(*call_args, key=key, **kwargs)

                v = jnp.ones_like(coords[i])
                d2 = jet_d1_d2(f_i, coords[i], v)[1]
                total = d2 if total is None else (total + d2)
            assert total is not None
            return total

        x = jnp.asarray(x0)
        total = None
        for i in range(var_dim):
            if var_dim == 1:
                v = jnp.ones_like(x)
            else:
                v = jnp.zeros_like(x).at[..., i].set(1.0)
            d2 = jet_d1_d2(f, x, v)[1]
            total = d2 if total is None else (total + d2)
        assert total is not None
        return total

    return DomainFunction(domain=u.domain, deps=u.deps, func=_lap, metadata=u.metadata)


def bilaplacian(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "jet", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Bi-Laplacian of a scalar field with respect to a geometry variable.

    Returns the fourth-order operator

    $$
    \Delta^2 u = \Delta(\Delta u).
    $$

    The `backend` options match `laplacian`.

    **Arguments:**

    - `u`: Input function (typically scalar-valued).
    - `var`: Geometry label to differentiate with respect to. If `None`, the domain
      must have exactly one geometry variable.
    - `mode`, `backend`, `basis`, `periodic`: As in `laplacian`.

    **Returns:**

    - A `DomainFunction` representing $\Delta^2 u$.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "bilaplacian(var=...) requires a geometry variable, not a scalar variable."
        )

    if var not in u.deps:

        def _zero(*args, key=None, **kwargs):
            y = jnp.asarray(u.func(*args, key=key, **kwargs))
            return jnp.zeros_like(y)

        return DomainFunction(
            domain=u.domain, deps=u.deps, func=_zero, metadata=u.metadata
        )

    if backend == "fd" or backend == "basis":
        return laplacian(
            laplacian(
                u,
                var=var,
                mode=mode,
                backend=backend,
                basis=basis,
                periodic=periodic,
            ),
            var=var,
            mode=mode,
            backend=backend,
            basis=basis,
            periodic=periodic,
        )

    if backend == "ad":
        return laplacian(
            laplacian(u, var=var, mode=mode, backend="ad"),
            var=var,
            mode=mode,
            backend="ad",
        )

    if backend != "jet":
        raise ValueError("backend must be 'ad', 'jet', 'fd', or 'basis'.")

    idx = u.deps.index(var)

    def _bilap(*args, key=None, **kwargs):
        x0 = args[idx]

        def f(xi):
            call_args = tuple(xi if i == idx else args[i] for i in range(len(args)))
            return u.func(*call_args, key=key, **kwargs)

        if isinstance(x0, tuple):
            coords = tuple(jnp.asarray(xi) for xi in x0)
            if len(coords) != var_dim:
                raise ValueError(
                    f"coord-separable bilaplacian expects {var_dim} coordinate arrays for var={var!r}."
                )
            if not all(xi.ndim == 1 for xi in coords):
                raise ValueError(
                    "coord-separable bilaplacian requires a tuple of 1D coordinate arrays."
                )

            pure_terms: list[jax.Array] = []
            for i in range(var_dim):

                def f_i(coord, *, i=i):
                    call_args = list(args)
                    call_args[idx] = coords[:i] + (coord,) + coords[i + 1 :]
                    return u.func(*call_args, key=key, **kwargs)

                pure_terms.append(jet_dn(f_i, coords[i], jnp.ones_like(coords[i]), n=4))

            total = pure_terms[0]
            for t in pure_terms[1:]:
                total = total + t

            for i in range(var_dim):
                for j in range(i + 1, var_dim):

                    def f_ij(ci, cj, *, i=i, j=j):
                        call_args = list(args)
                        call_args[idx] = (
                            coords[:i]
                            + (ci,)
                            + coords[i + 1 : j]
                            + (cj,)
                            + coords[j + 1 :]
                        )
                        return u.func(*call_args, key=key, **kwargs)

                    v_i = jnp.ones_like(coords[i])
                    v_j = jnp.ones_like(coords[j])
                    d4_plus = jet_dn_multi(f_ij, (coords[i], coords[j]), (v_i, v_j), n=4)
                    d4_minus = jet_dn_multi(
                        f_ij, (coords[i], coords[j]), (v_i, -v_j), n=4
                    )
                    mixed = (
                        d4_plus + d4_minus - 2.0 * pure_terms[i] - 2.0 * pure_terms[j]
                    ) / 12.0
                    total = total + 2.0 * mixed

            return total

        x = jnp.asarray(x0)
        if var_dim == 1:
            return jet_dn(f, x, jnp.ones_like(x), n=4)

        pure_terms: list[jax.Array] = []
        for i in range(var_dim):
            v = jnp.zeros_like(x).at[..., i].set(1.0)
            pure_terms.append(jet_dn(f, x, v, n=4))

        total = pure_terms[0]
        for t in pure_terms[1:]:
            total = total + t

        for i in range(var_dim):
            for j in range(i + 1, var_dim):
                v_plus = jnp.zeros_like(x).at[..., i].set(1.0).at[..., j].set(1.0)
                v_minus = jnp.zeros_like(x).at[..., i].set(1.0).at[..., j].set(-1.0)
                d4_plus = jet_dn(f, x, v_plus, n=4)
                d4_minus = jet_dn(f, x, v_minus, n=4)
                mixed = (
                    d4_plus + d4_minus - 2.0 * pure_terms[i] - 2.0 * pure_terms[j]
                ) / 12.0
                total = total + 2.0 * mixed

        return total

    return DomainFunction(domain=u.domain, deps=u.deps, func=_bilap, metadata=u.metadata)


def partial(
    u: DomainFunction,
    /,
    *,
    var: str,
    axis: int | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    _, var_dim = _factor_and_dim(u, var)
    g = grad(u, var=var, mode=mode)

    if var_dim == 1:

        def _partial(*args, key=None, **kwargs):
            jac = jnp.asarray(g.func(*args, key=key, **kwargs))
            if jac.ndim > 0 and jac.shape[-1] == 1:
                return jnp.squeeze(jac, axis=-1)
            return jac

        return DomainFunction(
            domain=u.domain, deps=u.deps, func=_partial, metadata=u.metadata
        )

    if axis is None:
        raise ValueError("partial(var=...) for a vector variable requires axis=...")
    axis_i = int(axis)
    if not (0 <= axis_i < var_dim):
        raise ValueError(f"axis must be in [0, {var_dim}), got {axis_i}.")

    def _partial(*args, key=None, **kwargs):
        jac = jnp.asarray(g.func(*args, key=key, **kwargs))
        return jnp.take(jac, axis_i, axis=-1)

    return DomainFunction(
        domain=u.domain, deps=u.deps, func=_partial, metadata=u.metadata
    )


def partial_n(
    u: DomainFunction,
    /,
    *,
    var: str,
    axis: int | None = None,
    order: int,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "jet", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Nth partial derivative with respect to a labeled variable.

    Computes $\partial^n u / \partial x_i^n$ (for geometry variables with `axis=i`) or
    $\partial^n u / \partial t^n$ (for scalar variables).

    **Arguments:**

    - `u`: Input function.
    - `var`: Variable label to differentiate with respect to.
    - `axis`: For geometry variables, which coordinate axis $i$ to differentiate along.
      For scalar variables, must be `None`.
    - `order`: Derivative order $n\ge 0$.
    - `mode`: Autodiff mode used when `backend="ad"`.
    - `backend`:
      - `"ad"`: repeated application of `partial`.
      - `"jet"`: uses Jet expansions for $n\ge 2$ (point inputs and coord-separable inputs).
      - `"fd"`: finite differences on coord-separable grids (falls back to `"ad"` for point inputs).
      - `"basis"`: spectral/barycentric methods on coord-separable grids (falls back to `"ad"` for point inputs).
    - `basis`: Basis method used when `backend="basis"`.
    - `periodic`: Whether to treat the differentiated axis as periodic (used by `backend="fd"`).

    **Returns:**

    - A `DomainFunction` representing the requested $n$th partial derivative.
    """
    order_i = int(order)
    if order_i < 0:
        raise ValueError("order must be non-negative.")
    if order_i == 0:
        return u
    if backend == "ad":
        out = u
        for _ in range(order_i):
            out = partial(out, var=var, axis=axis, mode=mode)
        return out
    if order_i == 1 and backend == "jet":
        return partial(u, var=var, axis=axis, mode=mode)
    if backend == "fd" or backend == "basis":
        # Discrete backends require coord-separable inputs; fall back to AD otherwise.
        fallback = partial_n(
            u,
            var=var,
            axis=axis,
            order=order_i,
            mode=mode,
            backend="ad",
        )

    _, var_dim = _factor_and_dim(u, var)

    if var not in u.deps:

        def _zero(*args, key=None, **kwargs):
            y = jnp.asarray(u.func(*args, key=key, **kwargs))
            return jnp.zeros_like(y)

        return DomainFunction(
            domain=u.domain, deps=u.deps, func=_zero, metadata=u.metadata
        )

    if var_dim == 1:
        if axis is None:
            axis_i = 0
        else:
            axis_i = int(axis)
            if axis_i != 0:
                raise ValueError("axis must be 0 for a 1D variable.")
    else:
        if axis is None:
            raise ValueError("partial_n(var=...) for a vector variable requires axis=...")
        axis_i = int(axis)
        if not (0 <= axis_i < int(var_dim)):
            raise ValueError(f"axis must be in [0,{int(var_dim)}), got {axis_i}.")

    idx = u.deps.index(var)

    def _nth(*args, key=None, **kwargs):
        x0 = args[idx]

        if backend == "fd" or backend == "basis":
            if not isinstance(x0, tuple):
                return fallback.func(*args, key=key, **kwargs)
            coords = tuple(jnp.asarray(xi) for xi in x0)
            if len(coords) != var_dim:
                raise ValueError(
                    f"coord-separable partial_n expects {var_dim} coordinate arrays for var={var!r}."
                )
            if not all(xi.ndim == 1 for xi in coords):
                raise ValueError(
                    "coord-separable partial_n requires a tuple of 1D coordinate arrays."
                )

            y = jnp.asarray(u.func(*args, key=key, **kwargs))
            axis_pos = _coord_axis_position(args, arg_index=idx, coord_index=axis_i)
            if backend == "fd":
                dx = (
                    coords[axis_i][1] - coords[axis_i][0]
                    if coords[axis_i].shape[0] > 1
                    else jnp.asarray(1.0, dtype=coords[axis_i].dtype)
                )
                return _fd_nth_derivative(
                    y, dx=dx, axis=axis_pos, order=order_i, periodic=bool(periodic)
                )
            return _basis_nth_derivative(
                y, coords[axis_i], axis=axis_pos, order=order_i, basis=basis
            )

        def f(xi):
            call_args = tuple(xi if i == idx else args[i] for i in range(len(args)))
            return u.func(*call_args, key=key, **kwargs)

        if isinstance(x0, tuple):
            coords = tuple(jnp.asarray(xi) for xi in x0)
            if len(coords) != var_dim:
                raise ValueError(
                    f"coord-separable partial_n expects {var_dim} coordinate arrays for var={var!r}."
                )
            if not all(xi.ndim == 1 for xi in coords):
                raise ValueError(
                    "coord-separable partial_n requires a tuple of 1D coordinate arrays."
                )

            def f_i(coord):
                call_args = list(args)
                call_args[idx] = coords[:axis_i] + (coord,) + coords[axis_i + 1 :]
                return u.func(*call_args, key=key, **kwargs)

            v = jnp.ones_like(coords[axis_i])
            return jet_dn(f_i, coords[axis_i], v, n=order_i)

        x = jnp.asarray(x0)
        if var_dim == 1:
            v = jnp.ones_like(x)
        else:
            v = jnp.zeros_like(x).at[..., axis_i].set(1.0)
        return jet_dn(f, x, v, n=order_i)

    return DomainFunction(domain=u.domain, deps=u.deps, func=_nth, metadata=u.metadata)


def partial_x(
    u: DomainFunction,
    /,
    *,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Convenience wrapper for $\partial u/\partial x$ (axis 0 of geometry variable `var`).

    Equivalent to `partial(u, var=var, axis=0, mode=mode)`.
    """
    return partial(u, var=var, axis=0, mode=mode)


def partial_y(
    u: DomainFunction,
    /,
    *,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Convenience wrapper for $\partial u/\partial y$ (axis 1 of geometry variable `var`).

    Equivalent to `partial(u, var=var, axis=1, mode=mode)`.
    """
    return partial(u, var=var, axis=1, mode=mode)


def partial_z(
    u: DomainFunction,
    /,
    *,
    var: str = "x",
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Convenience wrapper for $\partial u/\partial z$ (axis 2 of geometry variable `var`).

    Equivalent to `partial(u, var=var, axis=2, mode=mode)`.
    """
    return partial(u, var=var, axis=2, mode=mode)


def partial_t(
    u: DomainFunction,
    /,
    *,
    var: str = "t",
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Convenience wrapper for $\partial u/\partial t$ (scalar variable `var`).

    Equivalent to `partial(u, var=var, axis=None, mode=mode)`.
    """
    return partial(u, var=var, axis=None, mode=mode)


def dt(
    u: DomainFunction,
    /,
    *,
    var: str = "t",
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Alias for $\partial u/\partial t$ (`partial_t`)."""
    return partial_t(u, var=var, mode=mode)


def dt_n(
    u: DomainFunction,
    /,
    *,
    var: str = "t",
    order: int,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "jet"] = "ad",
) -> DomainFunction:
    r"""Nth time derivative $\partial^n u/\partial t^n$.

    This is a thin wrapper around `partial_n(u, var=time_var, axis=None, order=n, ...)`.

    **Arguments:**

    - `u`: Input function.
    - `var`: Time label.
    - `order`: Derivative order $n\ge 0$.
    - `mode`: Autodiff mode used when `backend="ad"`.
    - `backend`: `"ad"` (repeated application) or `"jet"` (Jet expansions for $n\ge 2$).
    """
    return partial_n(u, var=var, axis=None, order=int(order), mode=mode, backend=backend)


def material_derivative(
    u: DomainFunction,
    v: DomainFunction,
    /,
    *,
    spatial_var: str = "x",
    time_var: str = "t",
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Material derivative (advective derivative) of a field.

    Given a velocity field $v(x,t)$, the material derivative is

    $$
    \frac{D u}{D t} = \frac{\partial u}{\partial t} + v\cdot\nabla_x u.
    $$

    **Arguments:**

    - `u`: Field to differentiate.
    - `v`: Velocity field (vector-valued over `spatial_var`).
    - `spatial_var`: Geometry label (default `"x"`).
    - `time_var`: Time label (default `"t"`).
    - `mode`: Autodiff mode used by `dt` and `directional_derivative`.
    """
    return dt(u, var=time_var, mode=mode) + directional_derivative(
        u, v, var=spatial_var, mode=mode
    )


def _as_domain_function(
    u: DomainFunction, other: DomainFunction | ArrayLike, /
) -> DomainFunction:
    """Coerce `other` to a `DomainFunction` on `u.domain` (treating constants as constant fields)."""
    if isinstance(other, DomainFunction):
        return other
    return DomainFunction(domain=u.domain, deps=(), func=other, metadata={})


def div_k_grad(
    u: DomainFunction,
    k: DomainFunction | ArrayLike = 1.0,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Compute $\nabla\cdot(k\,\nabla u)$ for scalar diffusivity/permeability $k$.

    For scalar $k$ and (typically scalar) $u$, this implements the product-rule form

    $$
    \nabla\cdot(k\nabla u) = k\,\Delta u + \nabla k \cdot \nabla u.
    $$

    If `u` is vector-valued, this is applied componentwise.

    **Arguments:**

    - `u`: Field $u$.
    - `k`: Scalar field $k$ (constant or `DomainFunction`).
    - `var`: Geometry variable label.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `grad`/`laplacian`.

    **Returns:**

    - A `DomainFunction` representing $\nabla\cdot(k\nabla u)$.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "div_k_grad(var=...) requires a geometry variable, not a scalar variable."
        )

    k_fn = _as_domain_function(u, k)
    joined = u.domain.join(k_fn.domain)
    u2 = u.promote(joined)
    k2 = k_fn.promote(joined)

    gu = grad(u2, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)
    gk = grad(k2, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)
    lap_u = laplacian(
        u2,
        var=var,
        mode=mode,
        backend=backend,
        basis=basis,
        periodic=periodic,
    )

    deps = tuple(lbl for lbl in joined.labels if (lbl in u2.deps) or (lbl in k2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    k_pos = tuple(idx[lbl] for lbl in k2.deps)

    def _op(*args, key=None, **kwargs):
        u_args = [args[i] for i in u_pos]
        k_args = [args[i] for i in k_pos]

        gu_x = jnp.asarray(gu.func(*u_args, key=key, **kwargs))
        gk_x = jnp.asarray(gk.func(*k_args, key=key, **kwargs))
        kv = jnp.asarray(k2.func(*k_args, key=key, **kwargs))
        lu = jnp.asarray(lap_u.func(*u_args, key=key, **kwargs))

        if gu_x.shape[-1] != var_dim or gk_x.shape[-1] != var_dim:
            raise ValueError(
                f"div_k_grad expected grad(...).shape[-1]=={var_dim} for var={var!r}, "
                f"got grad(u).shape[-1]=={gu_x.shape[-1]} and grad(k).shape[-1]=={gk_x.shape[-1]}."
            )

        if gu_x.ndim == gk_x.ndim:
            dot_term = jnp.sum(gu_x * gk_x, axis=-1)
        elif gu_x.ndim == gk_x.ndim + 1:
            dot_term = jnp.sum(gu_x * gk_x[..., None, :], axis=-1)
        else:
            raise ValueError(
                f"Incompatible gradient ranks: grad(u).ndim={gu_x.ndim}, grad(k).ndim={gk_x.ndim}."
            )

        if lu.ndim == kv.ndim:
            k_lap = kv * lu
        elif lu.ndim == kv.ndim + 1:
            k_lap = kv[..., None] * lu
        else:
            raise ValueError(
                f"Incompatible ranks for k*laplacian(u): k.ndim={kv.ndim}, laplacian(u).ndim={lu.ndim}."
            )

        return dot_term + k_lap

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def div_diag_k_grad(
    u: DomainFunction,
    k_vec: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "jet"] = "ad",
) -> DomainFunction:
    r"""Compute $\nabla\cdot(\text{diag}(k)\,\nabla u)$ for diagonal anisotropy.

    Given a vector field $k=(k_1,\dots,k_d)$, this computes

    $$
    \nabla\cdot(\text{diag}(k)\nabla u)=\sum_{i=1}^d \frac{\partial}{\partial x_i}
    \left(k_i\,\frac{\partial u}{\partial x_i}\right).
    $$

    `k_vec` is expected to have a trailing axis of size $d$.

    **Arguments:**

    - `u`: Field $u$ (typically scalar-valued).
    - `k_vec`: Vector field of diagonal coefficients (trailing size $d$).
    - `var`: Geometry label to differentiate with respect to.
    - `mode`: Autodiff mode used when `backend="ad"`.
    - `backend`: `"ad"` (autodiff) or `"jet"` (Jet expansions).

    **Returns:**

    - A `DomainFunction` representing $\nabla\cdot(\text{diag}(k)\nabla u)$.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "div_diag_k_grad(var=...) requires a geometry variable, not a scalar variable."
        )

    joined = u.domain.join(k_vec.domain)
    u2 = u.promote(joined)
    k2 = k_vec.promote(joined)

    deps = tuple(lbl for lbl in joined.labels if (lbl in u2.deps) or (lbl in k2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    k_pos = tuple(idx[lbl] for lbl in k2.deps)

    if backend == "ad":
        gu = grad(u2, var=var, mode=mode)
        gk = grad(k2, var=var, mode=mode)
        hess_u = grad(grad(u2, var=var, mode=mode), var=var, mode=mode)

        def _op(*args, key=None, **kwargs):
            u_args = [args[i] for i in u_pos]
            k_args = [args[i] for i in k_pos]

            if var in u2.deps:
                x0 = u_args[u2.deps.index(var)]
            else:
                x0 = None

            if isinstance(x0, tuple):
                coords = tuple(jnp.asarray(xi) for xi in x0)
                if len(coords) != var_dim:
                    raise ValueError(
                        f"coord-separable div_diag_k_grad expects {var_dim} coordinate arrays for var={var!r}."
                    )
                if not all(xi.ndim == 1 for xi in coords):
                    raise ValueError(
                        "coord-separable div_diag_k_grad requires a tuple of 1D coordinate arrays."
                    )

                total = None
                for i in range(var_dim):

                    def u_i(coord, *, i=i):
                        call_args = list(u_args)
                        call_args[u2.deps.index(var)] = (
                            coords[:i] + (coord,) + coords[i + 1 :]
                        )
                        return u2.func(*call_args, key=key, **kwargs)

                    def k_i(coord, *, i=i):
                        call_args = list(k_args)
                        if var in k2.deps:
                            call_args[k2.deps.index(var)] = (
                                coords[:i] + (coord,) + coords[i + 1 :]
                            )
                        kval = jnp.asarray(k2.func(*call_args, key=key, **kwargs))
                        if kval.ndim == 0 or kval.shape[-1] != var_dim:
                            got = "()" if kval.ndim == 0 else str(kval.shape[-1])
                            raise ValueError(
                                f"div_diag_k_grad expects k_vec(...).shape[-1]=={var_dim}, got {got}."
                            )
                        return kval[..., i]

                    ones = jnp.ones_like(coords[i])
                    du_dxi = jax.jvp(u_i, (coords[i],), (ones,))[1]
                    dk_i_dxi = jax.jvp(k_i, (coords[i],), (ones,))[1]
                    d2u_dxi2 = jax.jvp(
                        lambda c: jax.jvp(u_i, (c,), (ones,))[1], (coords[i],), (ones,)
                    )[1]

                    kv_full = jnp.asarray(k2.func(*k_args, key=key, **kwargs))
                    if kv_full.ndim == 0 or kv_full.shape[-1] != var_dim:
                        got = "()" if kv_full.ndim == 0 else str(kv_full.shape[-1])
                        raise ValueError(
                            f"div_diag_k_grad expects k_vec(...).shape[-1]=={var_dim}, got {got}."
                        )
                    ki_val = kv_full[..., i]
                    term = dk_i_dxi * du_dxi + ki_val * d2u_dxi2
                    total = term if total is None else (total + term)
                assert total is not None
                return total

            gu_x = jnp.asarray(gu.func(*u_args, key=key, **kwargs))
            gk_x = jnp.asarray(gk.func(*k_args, key=key, **kwargs))
            kv = jnp.asarray(k2.func(*k_args, key=key, **kwargs))
            hu = jnp.asarray(hess_u.func(*u_args, key=key, **kwargs))

            if kv.ndim == 0 or kv.shape[-1] != var_dim:
                raise ValueError(
                    f"div_diag_k_grad expects k_vec(...).shape[-1]=={var_dim}, got {kv.shape[-1] if kv.ndim else '()'}."
                )
            if gk_x.ndim < 2 or gk_x.shape[-2:] != (var_dim, var_dim):
                raise ValueError(
                    f"div_diag_k_grad expected grad(k_vec).shape[-2:]==({var_dim}, {var_dim}), got {gk_x.shape[-2:]}."
                )

            dk_diag = jnp.diagonal(gk_x, axis1=-2, axis2=-1)
            d2u_diag = jnp.diagonal(hu, axis1=-2, axis2=-1)

            if gu_x.ndim == dk_diag.ndim:
                term1 = jnp.sum(dk_diag * gu_x, axis=-1)
                term2 = jnp.sum(kv * d2u_diag, axis=-1)
            elif gu_x.ndim == dk_diag.ndim + 1:
                term1 = jnp.sum(dk_diag[..., None, :] * gu_x, axis=-1)
                term2 = jnp.sum(kv[..., None, :] * d2u_diag, axis=-1)
            else:
                raise ValueError(
                    f"Incompatible ranks: grad(u).ndim={gu_x.ndim}, grad(k_vec).ndim={gk_x.ndim}."
                )
            return term1 + term2

        return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)

    if backend != "jet":
        raise ValueError("backend must be 'ad' or 'jet'.")

    u_var_idx = u2.deps.index(var) if var in u2.deps else None
    k_var_idx = k2.deps.index(var) if var in k2.deps else None

    def _op(*args, key=None, **kwargs):
        u_args = [args[i] for i in u_pos]
        k_args = [args[i] for i in k_pos]

        if u_var_idx is None:
            y = jnp.asarray(u2.func(*u_args, key=key, **kwargs))
            return jnp.zeros_like(y)

        x0 = u_args[u_var_idx]

        kv_full = jnp.asarray(k2.func(*k_args, key=key, **kwargs))
        if kv_full.ndim == 0 or kv_full.shape[-1] != var_dim:
            got = "()" if kv_full.ndim == 0 else str(kv_full.shape[-1])
            raise ValueError(
                f"div_diag_k_grad expects k_vec(...).shape[-1]=={var_dim}, got {got}."
            )

        if isinstance(x0, tuple):
            coords = tuple(jnp.asarray(xi) for xi in x0)
            if len(coords) != var_dim:
                raise ValueError(
                    f"coord-separable div_diag_k_grad expects {var_dim} coordinate arrays for var={var!r}."
                )
            if not all(xi.ndim == 1 for xi in coords):
                raise ValueError(
                    "coord-separable div_diag_k_grad requires a tuple of 1D coordinate arrays."
                )

            total = None
            for i in range(var_dim):

                def u_i(coord, *, i=i):
                    call_args = list(u_args)
                    call_args[u_var_idx] = coords[:i] + (coord,) + coords[i + 1 :]
                    return u2.func(*call_args, key=key, **kwargs)

                def k_i(coord, *, i=i):
                    call_args = list(k_args)
                    if k_var_idx is not None:
                        call_args[k_var_idx] = coords[:i] + (coord,) + coords[i + 1 :]
                    kval = jnp.asarray(k2.func(*call_args, key=key, **kwargs))
                    if kval.ndim == 0 or kval.shape[-1] != var_dim:
                        got = "()" if kval.ndim == 0 else str(kval.shape[-1])
                        raise ValueError(
                            f"div_diag_k_grad expects k_vec(...).shape[-1]=={var_dim}, got {got}."
                        )
                    return kval[..., i]

                ones = jnp.ones_like(coords[i])
                du_dxi, d2u_dxi2 = jet_d1_d2(u_i, coords[i], ones)
                if k_var_idx is None:
                    dk_i_dxi = jnp.zeros_like(du_dxi)
                else:
                    dk_i_dxi = jax.jvp(k_i, (coords[i],), (ones,))[1]

                ki_val = kv_full[..., i]
                term = dk_i_dxi * du_dxi + ki_val * d2u_dxi2
                total = term if total is None else (total + term)
            assert total is not None
            return total

        x = jnp.asarray(x0)

        def u_f(xi):
            call_args = list(u_args)
            call_args[u_var_idx] = xi
            return u2.func(*call_args, key=key, **kwargs)

        total = None
        for i in range(var_dim):
            if var_dim == 1:
                v = jnp.ones_like(x)
            else:
                v = jnp.zeros_like(x).at[..., i].set(1.0)

            def k_f(xi, *, i=i):
                call_args = list(k_args)
                if k_var_idx is not None:
                    call_args[k_var_idx] = xi
                kval = jnp.asarray(k2.func(*call_args, key=key, **kwargs))
                if kval.ndim == 0 or kval.shape[-1] != var_dim:
                    got = "()" if kval.ndim == 0 else str(kval.shape[-1])
                    raise ValueError(
                        f"div_diag_k_grad expects k_vec(...).shape[-1]=={var_dim}, got {got}."
                    )
                return kval[..., i]

            du_dxi, d2u_dxi2 = jet_d1_d2(u_f, x, v)
            if k_var_idx is None:
                dk_i_dxi = jnp.zeros_like(du_dxi)
            else:
                dk_i_dxi = jax.jvp(k_f, (x,), (v,))[1]

            ki_val = kv_full[..., i]
            term = dk_i_dxi * du_dxi + ki_val * d2u_dxi2
            total = term if total is None else (total + term)
        assert total is not None
        return total

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def div_K_grad(
    u: DomainFunction,
    K: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    backend: Literal["ad", "fd", "basis"] = "ad",
    basis: Literal["poly", "fourier", "sine", "cosine"] = "poly",
    periodic: bool = False,
) -> DomainFunction:
    r"""Compute $\nabla\cdot(K\,\nabla u)$ for a full anisotropic tensor $K$.

    For a matrix-valued field $K:\Omega\to\mathbb{R}^{d\times d}$ and scalar $u$, this
    computes the second-order elliptic operator

    $$
    \nabla\cdot(K\nabla u).
    $$

    When expanded, this contains both first-derivative and second-derivative terms:

    $$
    \nabla\cdot(K\nabla u) = (\nabla\cdot K)\cdot\nabla u + K : \nabla^2 u.
    $$

    If `u` is vector-valued, the operator is applied componentwise.

    **Arguments:**

    - `u`: Field $u$.
    - `K`: Matrix-valued coefficient field with trailing shape `(d, d)`.
    - `var`: Geometry label to differentiate with respect to.
    - `mode`, `backend`, `basis`, `periodic`: Passed through to `grad` and `div`.

    **Returns:**

    - A `DomainFunction` representing $\nabla\cdot(K\nabla u)$.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "div_K_grad(var=...) requires a geometry variable, not a scalar variable."
        )

    joined = u.domain.join(K.domain)
    u2 = u.promote(joined)
    K2 = K.promote(joined)

    gu = grad(u2, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)

    deps = tuple(lbl for lbl in joined.labels if (lbl in u2.deps) or (lbl in K2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    k_pos = tuple(idx[lbl] for lbl in K2.deps)

    def _flux(*args, key=None, **kwargs):
        u_args = [args[i] for i in u_pos]
        k_args = [args[i] for i in k_pos]

        gu_x = jnp.asarray(gu.func(*u_args, key=key, **kwargs))
        Kx = jnp.asarray(K2.func(*k_args, key=key, **kwargs))

        if Kx.shape[-2:] != (var_dim, var_dim):
            raise ValueError(
                f"div_K_grad expected K(...).shape[-2:]==({var_dim}, {var_dim}), got {Kx.shape[-2:]}."
            )
        if gu_x.shape[-1] != var_dim:
            raise ValueError(
                f"div_K_grad expected grad(u).shape[-1]=={var_dim}, got {gu_x.shape[-1]}."
            )

        batch_rank = Kx.ndim - 2
        out_rank = gu_x.ndim - batch_rank - 1
        if out_rank < 0:
            raise ValueError("div_K_grad received incompatible K and grad(u) ranks.")
        Kx_exp = Kx.reshape(Kx.shape[:-2] + (1,) * int(out_rank) + Kx.shape[-2:])
        return oe.contract("...ij,...j->...i", Kx_exp, gu_x)

    flux = DomainFunction(domain=joined, deps=deps, func=_flux, metadata=u.metadata)
    return div(flux, var=var, mode=mode, backend=backend, basis=basis, periodic=periodic)


def deformation_gradient(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Deformation gradient for a displacement field.

    For a displacement field $u:\Omega\to\mathbb{R}^d$, the deformation gradient is

    $$
    F = I + \nabla u.
    $$

    **Arguments:**

    - `u`: Displacement field (vector-valued).
    - `var`: Geometry label to differentiate with respect to.
    - `mode`: Autodiff mode passed to `grad`.

    **Returns:**

    - A `DomainFunction` representing $F$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "deformation_gradient(var=...) requires a geometry variable, not a scalar variable."
        )

    G = grad(u, var=var, mode=mode)
    I = jnp.eye(var_dim)
    return G + I


def green_lagrange_strain(
    u: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Green–Lagrange strain tensor.

    With deformation gradient $F=I+\nabla u$, the Green–Lagrange strain is

    $$
    E = \tfrac12\left(F^\top F - I\right).
    $$

    **Arguments:**

    - `u`: Displacement field.
    - `var`: Geometry label to differentiate with respect to.
    - `mode`: Autodiff mode used by `deformation_gradient`.

    **Returns:**

    - A `DomainFunction` representing $E$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "green_lagrange_strain(var=...) requires a geometry variable, not a scalar variable."
        )

    F = deformation_gradient(u, var=var, mode=mode)
    I = jnp.eye(var_dim)

    def _E(*args, key=None, **kwargs):
        Fx = jnp.asarray(F.func(*args, key=key, **kwargs))
        C = jnp.swapaxes(Fx, -1, -2) @ Fx
        return 0.5 * (C - I)

    return DomainFunction(domain=F.domain, deps=F.deps, func=_E, metadata=F.metadata)


def svk_pk2_stress(
    u: DomainFunction,
    /,
    *,
    lambda_: DomainFunction | ArrayLike,
    mu: DomainFunction | ArrayLike,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""St. Venant–Kirchhoff 2nd Piola–Kirchhoff stress.

    With Green–Lagrange strain $E$, the StVK constitutive law is

    $$
    S(E) = 2\mu\,E + \lambda\,\text{tr}(E)\,I.
    $$

    **Arguments:**

    - `u`: Displacement field.
    - `lambda_`, `mu`: Lamé parameters (constants or `DomainFunction`s).
    - `var`: Geometry label.
    - `mode`: Autodiff mode used by `green_lagrange_strain`.

    **Returns:**

    - A `DomainFunction` representing the PK2 stress $S$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "svk_pk2_stress(var=...) requires a geometry variable, not a scalar variable."
        )

    mu_fn = _as_domain_function(u, mu)
    lambda_fn = _as_domain_function(u, lambda_)
    joined = u.domain.join(mu_fn.domain).join(lambda_fn.domain)
    u2 = u.promote(joined)
    mu2 = mu_fn.promote(joined)
    lam2 = lambda_fn.promote(joined)

    E = green_lagrange_strain(u2, var=var, mode=mode)
    trE = _trace_last2(E, keepdims=True)
    I = jnp.eye(var_dim)
    return 2.0 * mu2 * E + lam2 * trE * I


def pk1_from_pk2(
    u: DomainFunction,
    S: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Convert 2nd PK stress to 1st PK stress.

    With deformation gradient $F$, the first Piola–Kirchhoff stress is

    $$
    P = F S.
    $$

    **Arguments:**

    - `u`: Displacement field (used to compute $F$).
    - `S`: Second PK stress field.
    - `var`: Geometry label.
    - `mode`: Autodiff mode used by `deformation_gradient`.

    **Returns:**

    - A `DomainFunction` representing $P$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    factor, _ = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "pk1_from_pk2(var=...) requires a geometry variable, not a scalar variable."
        )

    joined = u.domain.join(S.domain)
    u2 = u.promote(joined)
    S2 = S.promote(joined)

    F = deformation_gradient(u2, var=var, mode=mode)

    deps = tuple(lbl for lbl in joined.labels if (lbl in u2.deps) or (lbl in S2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    s_pos = tuple(idx[lbl] for lbl in S2.deps)

    def _op(*args, key=None, **kwargs):
        u_args = [args[i] for i in u_pos]
        s_args = [args[i] for i in s_pos]
        Fx = jnp.asarray(F.func(*u_args, key=key, **kwargs))
        Sx = jnp.asarray(S2.func(*s_args, key=key, **kwargs))
        return Fx @ Sx

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def cauchy_from_pk2(
    u: DomainFunction,
    S: DomainFunction,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Convert 2nd PK stress to Cauchy stress.

    With deformation gradient $F$ and $J=\det F$, the Cauchy stress is

    $$
    \sigma = \frac{1}{J} F S F^\top.
    $$

    **Arguments:**

    - `u`: Displacement field (used to compute $F$).
    - `S`: Second PK stress field.
    - `var`: Geometry label.
    - `mode`: Autodiff mode used by `deformation_gradient`.

    **Returns:**

    - A `DomainFunction` representing $\sigma$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    factor, _ = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "cauchy_from_pk2(var=...) requires a geometry variable, not a scalar variable."
        )

    joined = u.domain.join(S.domain)
    u2 = u.promote(joined)
    S2 = S.promote(joined)

    F = deformation_gradient(u2, var=var, mode=mode)

    deps = tuple(lbl for lbl in joined.labels if (lbl in u2.deps) or (lbl in S2.deps))
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    s_pos = tuple(idx[lbl] for lbl in S2.deps)

    def _op(*args, key=None, **kwargs):
        u_args = [args[i] for i in u_pos]
        s_args = [args[i] for i in s_pos]
        Fx = jnp.asarray(F.func(*u_args, key=key, **kwargs))
        Sx = jnp.asarray(S2.func(*s_args, key=key, **kwargs))
        J = jnp.linalg.det(Fx)
        return (Fx @ Sx @ jnp.swapaxes(Fx, -1, -2)) / J[..., None, None]

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def neo_hookean_pk1(
    u: DomainFunction,
    /,
    *,
    mu: DomainFunction | ArrayLike,
    kappa: DomainFunction | ArrayLike,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Compressible neo-Hookean first Piola–Kirchhoff stress.

    With deformation gradient $F = I + \nabla u$ and $J=\det F$, this returns

    $$
    P = \mu\,(F - F^{-T}) + \kappa\,\ln(J)\,F^{-T},
    $$

    where $\mu$ is the shear modulus and $\kappa$ is the bulk modulus.

    **Arguments:**

    - `u`: Displacement field.
    - `mu`: Shear modulus $\mu$ (constant or `DomainFunction`).
    - `kappa`: Bulk modulus $\kappa$ (constant or `DomainFunction`).
    - `var`: Geometry label.
    - `mode`: Autodiff mode used by `deformation_gradient`.

    **Returns:**

    - A `DomainFunction` representing the PK1 stress $P$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    factor, _ = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "neo_hookean_pk1(var=...) requires a geometry variable, not a scalar variable."
        )

    mu_fn = _as_domain_function(u, mu)
    k_fn = _as_domain_function(u, kappa)
    joined = u.domain.join(mu_fn.domain).join(k_fn.domain)
    u2 = u.promote(joined)
    mu2 = mu_fn.promote(joined)
    k2 = k_fn.promote(joined)

    F = deformation_gradient(u2, var=var, mode=mode)

    deps = tuple(
        lbl
        for lbl in joined.labels
        if (lbl in u2.deps) or (lbl in mu2.deps) or (lbl in k2.deps)
    )
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    mu_pos = tuple(idx[lbl] for lbl in mu2.deps)
    k_pos = tuple(idx[lbl] for lbl in k2.deps)

    def _op(*args, key=None, **kwargs):
        u_args = [args[i] for i in u_pos]
        mu_args = [args[i] for i in mu_pos]
        k_args = [args[i] for i in k_pos]
        Fx = jnp.asarray(F.func(*u_args, key=key, **kwargs))
        FinvT = jnp.linalg.inv(Fx).swapaxes(-1, -2)
        J = jnp.linalg.det(Fx)
        mu_v = jnp.asarray(mu2.func(*mu_args, key=key, **kwargs))
        k_v = jnp.asarray(k2.func(*k_args, key=key, **kwargs))
        return mu_v * (Fx - FinvT) + k_v * jnp.log(J)[..., None, None] * FinvT

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def neo_hookean_cauchy(
    u: DomainFunction,
    /,
    *,
    mu: DomainFunction | ArrayLike,
    kappa: DomainFunction | ArrayLike,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""Compressible neo-Hookean Cauchy stress.

    With $F = I + \nabla u$, $J=\det F$, and left Cauchy–Green tensor $B=FF^\top$, this
    returns

    $$
    \sigma = \frac{\mu}{J}(B - I) + \frac{\kappa\ln(J)}{J}\,I.
    $$

    **Arguments:**

    - `u`: Displacement field.
    - `mu`: Shear modulus $\mu$.
    - `kappa`: Bulk modulus $\kappa$.
    - `var`: Geometry label.
    - `mode`: Autodiff mode used by `deformation_gradient`.

    **Returns:**

    - A `DomainFunction` representing the Cauchy stress $\sigma$ with trailing shape `(..., d, d)`.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "neo_hookean_cauchy(var=...) requires a geometry variable, not a scalar variable."
        )

    mu_fn = _as_domain_function(u, mu)
    k_fn = _as_domain_function(u, kappa)
    joined = u.domain.join(mu_fn.domain).join(k_fn.domain)
    u2 = u.promote(joined)
    mu2 = mu_fn.promote(joined)
    k2 = k_fn.promote(joined)

    F = deformation_gradient(u2, var=var, mode=mode)
    I = jnp.eye(var_dim)

    deps = tuple(
        lbl
        for lbl in joined.labels
        if (lbl in u2.deps) or (lbl in mu2.deps) or (lbl in k2.deps)
    )
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    mu_pos = tuple(idx[lbl] for lbl in mu2.deps)
    k_pos = tuple(idx[lbl] for lbl in k2.deps)

    def _op(*args, key=None, **kwargs):
        u_args = [args[i] for i in u_pos]
        mu_args = [args[i] for i in mu_pos]
        k_args = [args[i] for i in k_pos]
        Fx = jnp.asarray(F.func(*u_args, key=key, **kwargs))
        B = Fx @ jnp.swapaxes(Fx, -1, -2)
        J = jnp.linalg.det(Fx)
        mu_v = jnp.asarray(mu2.func(*mu_args, key=key, **kwargs))
        k_v = jnp.asarray(k2.func(*k_args, key=key, **kwargs))
        term = (mu_v / J)[..., None, None] * (B - I)
        vol = (k_v * jnp.log(J) / J)[..., None, None] * I
        return term + vol

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def deviatoric_stress(
    sigma: DomainFunction,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    r"""Deviatoric part of a stress tensor.

    For a stress tensor $\sigma$, the deviatoric part is

    $$
    \sigma_{\text{dev}} = \sigma - \frac{\text{tr}(\sigma)}{d}\,I.
    $$

    **Arguments:**

    - `sigma`: Stress tensor field (trailing shape `(d, d)`).
    - `var`: Geometry label used to infer $d$.

    **Returns:**

    - A `DomainFunction` representing $\sigma_{\text{dev}}$.
    """
    var = _resolve_var(sigma, var)
    factor, var_dim = _factor_and_dim(sigma, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "deviatoric_stress(var=...) requires a geometry variable, not a scalar variable."
        )

    tr = _trace_last2(sigma, keepdims=True)
    I = jnp.eye(var_dim)
    return sigma - (tr / float(var_dim)) * I


def hydrostatic_pressure(
    sigma: DomainFunction,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    r"""Hydrostatic pressure from a stress tensor.

    Uses the sign convention

    $$
    p = -\frac{1}{d}\text{tr}(\sigma).
    $$

    **Arguments:**

    - `sigma`: Stress tensor field.
    - `var`: Geometry label used to infer $d$.

    **Returns:**

    - A `DomainFunction` representing the scalar pressure field $p$.
    """
    var = _resolve_var(sigma, var)
    factor, var_dim = _factor_and_dim(sigma, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "hydrostatic_pressure(var=...) requires a geometry variable, not a scalar variable."
        )

    tr = _trace_last2(sigma, keepdims=False)
    return (-1.0 / float(var_dim)) * tr


def hydrostatic_stress(
    sigma: DomainFunction,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    r"""Hydrostatic (spherical) part of a stress tensor.

    Returns

    $$
    \sigma_{\text{hyd}} = \frac{\text{tr}(\sigma)}{d}\,I.
    $$

    **Arguments:**

    - `sigma`: Stress tensor field.
    - `var`: Geometry label used to infer $d$.

    **Returns:**

    - A `DomainFunction` representing the hydrostatic stress tensor.
    """
    var = _resolve_var(sigma, var)
    factor, var_dim = _factor_and_dim(sigma, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "hydrostatic_stress(var=...) requires a geometry variable, not a scalar variable."
        )

    tr = _trace_last2(sigma, keepdims=True)
    I = jnp.eye(var_dim)
    return (tr / float(var_dim)) * I


def von_mises_stress(
    sigma: DomainFunction,
    /,
    *,
    var: str | None = None,
) -> DomainFunction:
    r"""Von Mises equivalent stress.

    For deviatoric stress $s = \sigma - \tfrac{1}{d}\text{tr}(\sigma)\,I$, returns

    $$
    \sigma_{\text{vm}} = \sqrt{\frac{3}{2}\,s:s},
    $$

    with a specialized closed form used in 2D.

    **Arguments:**

    - `sigma`: Stress tensor field.
    - `var`: Geometry label used to infer dimension and select the 2D/3D formula.

    **Returns:**

    - A `DomainFunction` representing the scalar von Mises equivalent stress.
    """
    var = _resolve_var(sigma, var)
    factor, var_dim = _factor_and_dim(sigma, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "von_mises_stress(var=...) requires a geometry variable, not a scalar variable."
        )

    def _op(*args, key=None, **kwargs):
        sig = jnp.asarray(sigma.func(*args, key=key, **kwargs))
        if var_dim == 2:
            sx = sig[..., 0, 0]
            sy = sig[..., 1, 1]
            txy = sig[..., 0, 1]
            vm2 = sx * sx - sx * sy + sy * sy + 3.0 * (txy * txy)
            return jnp.sqrt(jnp.maximum(vm2, 0.0))

        tr = jnp.trace(sig, axis1=-2, axis2=-1)[..., None, None]
        I = jnp.eye(var_dim)
        I = jnp.broadcast_to(I, sig.shape)
        s = sig - (tr / float(var_dim)) * I
        j2 = 0.5 * jnp.sum(s * s, axis=(-1, -2))
        return jnp.sqrt(jnp.maximum(3.0 * j2, 0.0))

    return DomainFunction(
        domain=sigma.domain, deps=sigma.deps, func=_op, metadata=sigma.metadata
    )


def maxwell_stress(
    *,
    E: DomainFunction | None = None,
    H: DomainFunction | None = None,
    epsilon: DomainFunction | ArrayLike | None = None,
    mu: DomainFunction | ArrayLike | None = None,
) -> DomainFunction:
    r"""Maxwell stress tensor from electric and/or magnetic fields.

    For electric field $E$ and magnetic field $H$, the (vacuum-style) Maxwell stress
    tensor is

    $$
    T = \epsilon\left(E\otimes E - \tfrac12\|E\|_2^2 I\right)
      + \mu\left(H\otimes H - \tfrac12\|H\|_2^2 I\right),
    $$

    where $\epsilon$ and $\mu$ can be scalars or fields.

    At least one of `E` or `H` must be provided.

    **Arguments:**

    - `E`: Electric field (vector-valued) or `None`.
    - `H`: Magnetic field (vector-valued) or `None`.
    - `epsilon`: Permittivity $\epsilon$ (scalar/field); defaults to 1.
    - `mu`: Permeability $\mu$ (scalar/field); defaults to 1.

    **Returns:**

    - A `DomainFunction` representing the Maxwell stress tensor $T$ with trailing shape `(..., d, d)`.
    """
    if E is None and H is None:
        raise ValueError("maxwell_stress requires at least one of E or H.")

    base = E or H
    assert base is not None

    eps_fn = _as_domain_function(base, 1.0 if epsilon is None else epsilon)
    mu_fn = _as_domain_function(base, 1.0 if mu is None else mu)

    funcs: list[DomainFunction] = [base]
    if E is not None:
        funcs.append(E)
    if H is not None:
        funcs.append(H)
    if isinstance(eps_fn, DomainFunction):
        funcs.append(eps_fn)
    if isinstance(mu_fn, DomainFunction):
        funcs.append(mu_fn)

    joined = funcs[0].domain
    for fn in funcs[1:]:
        joined = joined.join(fn.domain)

    E2 = E.promote(joined) if E is not None else None
    H2 = H.promote(joined) if H is not None else None
    eps2 = eps_fn.promote(joined)
    mu2 = mu_fn.promote(joined)

    deps_set = set()
    for fn in (E2, H2, eps2, mu2):
        if fn is None:
            continue
        deps_set.update(fn.deps)
    deps = tuple(lbl for lbl in joined.labels if lbl in deps_set)
    idx = {lbl: i for i, lbl in enumerate(deps)}

    def _pos(fn: DomainFunction | None) -> tuple[int, ...]:
        if fn is None:
            return ()
        return tuple(idx[lbl] for lbl in fn.deps)

    E_pos = _pos(E2)
    H_pos = _pos(H2)
    eps_pos = _pos(eps2)
    mu_pos = _pos(mu2)

    def _op(*args, key=None, **kwargs):
        T = 0.0
        n = None

        if E2 is not None:
            Ex = jnp.asarray(E2.func(*[args[i] for i in E_pos], key=key, **kwargs))
            n = int(Ex.shape[-1])
            I = jnp.eye(n)
            I = jnp.broadcast_to(I, Ex.shape[:-1] + (n, n))
            eps_v = jnp.asarray(eps2.func(*[args[i] for i in eps_pos], key=key, **kwargs))
            ee = oe.contract("...i,...j->...ij", Ex, Ex)
            e2 = jnp.sum(Ex * Ex, axis=-1)[..., None, None]
            T = T + eps_v * (ee - 0.5 * e2 * I)

        if H2 is not None:
            Hx = jnp.asarray(H2.func(*[args[i] for i in H_pos], key=key, **kwargs))
            if n is None:
                n = int(Hx.shape[-1])
            I = jnp.eye(int(n))
            I = jnp.broadcast_to(I, Hx.shape[:-1] + (int(n), int(n)))
            mu_v = jnp.asarray(mu2.func(*[args[i] for i in mu_pos], key=key, **kwargs))
            hh = oe.contract("...i,...j->...ij", Hx, Hx)
            h2 = jnp.sum(Hx * Hx, axis=-1)[..., None, None]
            T = T + mu_v * (hh - 0.5 * h2 * I)

        return T

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata={})


def linear_elastic_cauchy_stress_2d(
    u: DomainFunction,
    /,
    *,
    E: DomainFunction | ArrayLike,
    nu: DomainFunction | ArrayLike,
    mode2d: Literal["plane_stress", "plane_strain"] = "plane_stress",
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""2D isotropic linear-elastic Cauchy stress (plane stress/strain).

    Given Young's modulus $E$ and Poisson ratio $\nu$, returns the stress tensor
    $\sigma$ computed from the small strain tensor $\varepsilon(u)$ using either:

    - plane stress, or
    - plane strain,

    controlled by `mode2d`.

    **Arguments:**

    - `u`: Displacement field.
    - `E`: Young's modulus $E$ (constant or field).
    - `nu`: Poisson ratio $\nu$ (constant or field).
    - `mode2d`: `"plane_stress"` or `"plane_strain"`.
    - `var`: Geometry label (must be 2D).
    - `mode`: Autodiff mode used by `cauchy_strain`.

    **Returns:**

    - A `DomainFunction` representing the 2D Cauchy stress tensor with trailing shape `(2, 2)`.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "linear_elastic_cauchy_stress_2d(var=...) requires a geometry variable, not a scalar variable."
        )
    if var_dim != 2:
        raise ValueError(
            f"linear_elastic_cauchy_stress_2d requires var_dim=2, got var_dim={var_dim}."
        )

    E_fn = _as_domain_function(u, E)
    nu_fn = _as_domain_function(u, nu)
    joined = u.domain.join(E_fn.domain).join(nu_fn.domain)
    u2 = u.promote(joined)
    E2 = E_fn.promote(joined)
    nu2 = nu_fn.promote(joined)

    eps = cauchy_strain(u2, var=var, mode=mode)

    deps = tuple(
        lbl
        for lbl in joined.labels
        if (lbl in u2.deps) or (lbl in E2.deps) or (lbl in nu2.deps)
    )
    idx = {lbl: i for i, lbl in enumerate(deps)}
    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    e_pos = tuple(idx[lbl] for lbl in E2.deps)
    nu_pos = tuple(idx[lbl] for lbl in nu2.deps)

    def _op(*args, key=None, **kwargs):
        u_args = [args[i] for i in u_pos]
        e_args = [args[i] for i in e_pos]
        nu_args = [args[i] for i in nu_pos]

        eps_x = jnp.asarray(eps.func(*u_args, key=key, **kwargs))
        exx = eps_x[..., 0, 0]
        eyy = eps_x[..., 1, 1]
        gxy = 2.0 * eps_x[..., 0, 1]

        E_v = jnp.asarray(E2.func(*e_args, key=key, **kwargs))
        nu_v = jnp.asarray(nu2.func(*nu_args, key=key, **kwargs))

        if mode2d == "plane_stress":
            coef = E_v / (1.0 - nu_v * nu_v)
            C11 = coef
            C12 = coef * nu_v
            C33 = coef * (1.0 - nu_v) / 2.0
            s_xx = C11 * exx + C12 * eyy
            s_yy = C12 * exx + C11 * eyy
            t_xy = C33 * gxy
        elif mode2d == "plane_strain":
            coef = E_v / ((1.0 + nu_v) * (1.0 - 2.0 * nu_v))
            C11 = coef * (1.0 - nu_v)
            C12 = coef * nu_v
            C33 = coef * (1.0 - 2.0 * nu_v) / 2.0
            s_xx = C11 * exx + C12 * eyy
            s_yy = C12 * exx + C11 * eyy
            t_xy = C33 * gxy
        else:
            raise ValueError("mode2d must be 'plane_stress' or 'plane_strain'.")

        return jnp.stack(
            [jnp.stack([s_xx, t_xy], axis=-1), jnp.stack([t_xy, s_yy], axis=-1)],
            axis=-1,
        )

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)


def linear_elastic_orthotropic_stress_2d(
    u: DomainFunction,
    /,
    *,
    E1: DomainFunction | ArrayLike,
    E2: DomainFunction | ArrayLike,
    nu12: DomainFunction | ArrayLike,
    G12: DomainFunction | ArrayLike,
    mode2d: Literal["plane_stress", "plane_strain"] = "plane_stress",
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
) -> DomainFunction:
    r"""2D orthotropic linear-elastic Cauchy stress (plane stress/strain).

    Uses orthotropic material parameters $(E_1, E_2, \nu_{12}, G_{12})$ and computes
    the 2D stress response under either plane stress or plane strain assumptions.

    **Arguments:**

    - `u`: Displacement field.
    - `E1`, `E2`: Orthotropic Young's moduli.
    - `nu12`: Major Poisson ratio.
    - `G12`: In-plane shear modulus.
    - `mode2d`: `"plane_stress"` or `"plane_strain"`.
    - `var`: Geometry label (must be 2D).
    - `mode`: Autodiff mode used by `cauchy_strain`.

    **Returns:**

    - A `DomainFunction` representing the 2D orthotropic stress tensor with trailing shape `(2, 2)`.
    """
    var = _resolve_var(u, var)
    factor, var_dim = _factor_and_dim(u, var)
    if isinstance(factor, _AbstractScalarDomain):
        raise ValueError(
            "linear_elastic_orthotropic_stress_2d(var=...) requires a geometry variable, not a scalar variable."
        )
    if var_dim != 2:
        raise ValueError(
            f"linear_elastic_orthotropic_stress_2d requires var_dim=2, got var_dim={var_dim}."
        )

    E1_fn = _as_domain_function(u, E1)
    E2_fn = _as_domain_function(u, E2)
    nu12_fn = _as_domain_function(u, nu12)
    G12_fn = _as_domain_function(u, G12)
    joined = (
        u.domain.join(E1_fn.domain)
        .join(E2_fn.domain)
        .join(nu12_fn.domain)
        .join(G12_fn.domain)
    )
    u2 = u.promote(joined)
    E1_2 = E1_fn.promote(joined)
    E2_2 = E2_fn.promote(joined)
    nu12_2 = nu12_fn.promote(joined)
    G12_2 = G12_fn.promote(joined)

    eps = cauchy_strain(u2, var=var, mode=mode)

    deps_set = (
        set(u2.deps)
        | set(E1_2.deps)
        | set(E2_2.deps)
        | set(nu12_2.deps)
        | set(G12_2.deps)
    )
    deps = tuple(lbl for lbl in joined.labels if lbl in deps_set)
    idx = {lbl: i for i, lbl in enumerate(deps)}

    u_pos = tuple(idx[lbl] for lbl in u2.deps)
    e1_pos = tuple(idx[lbl] for lbl in E1_2.deps)
    e2_pos = tuple(idx[lbl] for lbl in E2_2.deps)
    nu_pos = tuple(idx[lbl] for lbl in nu12_2.deps)
    g_pos = tuple(idx[lbl] for lbl in G12_2.deps)

    def _op(*args, key=None, **kwargs):
        u_args = [args[i] for i in u_pos]
        e1_args = [args[i] for i in e1_pos]
        e2_args = [args[i] for i in e2_pos]
        nu_args = [args[i] for i in nu_pos]
        g_args = [args[i] for i in g_pos]

        eps_x = jnp.asarray(eps.func(*u_args, key=key, **kwargs))
        exx = eps_x[..., 0, 0]
        eyy = eps_x[..., 1, 1]
        gxy = 2.0 * eps_x[..., 0, 1]

        E1v = jnp.asarray(E1_2.func(*e1_args, key=key, **kwargs))
        E2v = jnp.asarray(E2_2.func(*e2_args, key=key, **kwargs))
        nu12v = jnp.asarray(nu12_2.func(*nu_args, key=key, **kwargs))
        G12v = jnp.asarray(G12_2.func(*g_args, key=key, **kwargs))
        nu21v = nu12v * E2v / E1v

        if mode2d == "plane_stress":
            S11 = 1.0 / E1v
            S22 = 1.0 / E2v
            S12 = -nu12v / E1v
            S21 = -nu21v / E2v
            S33 = 1.0 / G12v
            detS = S11 * S22 - S12 * S21
            C11 = S22 / detS
            C22 = S11 / detS
            C12 = -S12 / detS
            C33 = 1.0 / S33
            s_xx = C11 * exx + C12 * eyy
            s_yy = C12 * exx + C22 * eyy
            t_xy = C33 * gxy
        elif mode2d == "plane_strain":
            S11 = 1.0 / E1v
            S22 = 1.0 / E2v
            S12 = -nu12v / E1v
            S21 = -nu21v / E2v
            detS = S11 * S22 - S12 * S21
            C11 = S22 / detS
            C22 = S11 / detS
            C12 = -S12 / detS
            C33 = G12v
            s_xx = C11 * exx + C12 * eyy
            s_yy = C12 * exx + C22 * eyy
            t_xy = C33 * gxy
        else:
            raise ValueError("mode2d must be 'plane_stress' or 'plane_strain'.")

        return jnp.stack(
            [jnp.stack([s_xx, t_xy], axis=-1), jnp.stack([t_xy, s_yy], axis=-1)],
            axis=-1,
        )

    return DomainFunction(domain=joined, deps=deps, func=_op, metadata=u.metadata)
