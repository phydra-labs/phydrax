#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp

from ...domain._base import _AbstractGeometry
from ...domain._product_domain import ProductDomain


def _spatial_geometry(domain: object, /) -> _AbstractGeometry:
    if isinstance(domain, _AbstractGeometry):
        return domain
    if isinstance(domain, ProductDomain):
        for factor in domain.factors:
            if isinstance(factor, _AbstractGeometry):
                return factor
    raise TypeError(
        "Expected a spatial geometry or a ProductDomain containing a spatial geometry."
    )


def _domain_total_measure(domain, *, over: Literal["interior", "boundary"]):
    """Return the total measure value of the domain region to integrate over.

    Assumes a Phydrax domain with:
    - `domain.volume` for interior measure (length/area/volume)
    - `domain.boundary_measure_value` for boundary measure (boundary length / surface area)
    """
    geom = _spatial_geometry(domain)
    if over == "interior":
        return jnp.asarray(geom.volume, dtype=float)
    return jnp.asarray(geom.boundary_measure_value, dtype=float)


_QUAD_CACHE: dict[tuple, dict] = {}
_SUBSET_MEASURE_CACHE: dict[tuple, float] = {}


def build_quadrature(
    domain,
    *,
    over: Literal["interior", "boundary"] = "interior",
    num_points: int = 2048,
    where: Callable | None = None,
    seed: int = 0,
    use_subset_measure: bool | None = None,
    subset_measure: float | None = None,
    num_subset_samples: int = 4096,
    cache_subset_measure: bool = True,
    sampler: str | None = None,
    strata: dict | None = None,
) -> dict:
    key = (
        id(domain),
        over,
        int(num_points),
        id(where),
        int(seed),
        bool(use_subset_measure) if use_subset_measure is not None else None,
        float(subset_measure) if subset_measure is not None else None,
        int(num_subset_samples),
        bool(cache_subset_measure),
        sampler,
        None
        if strata is None
        else tuple(sorted((k, float(v)) for k, v in strata.items())),
    )
    if key in _QUAD_CACHE:
        return _QUAD_CACHE[key]

    rng = jr.key(seed)
    geom = _spatial_geometry(domain)
    if over == "interior":
        if sampler is None:
            pts = geom.sample_interior(num_points, where=where, key=rng)
        else:
            pts = geom.sample_interior(num_points, where=where, sampler=sampler, key=rng)
        measure_val = _domain_total_measure(geom, over=over)
        if where is not None and (use_subset_measure or subset_measure is not None):
            if subset_measure is not None:
                subset_val = float(subset_measure)
            else:
                sm_key = (id(domain), over, id(where))
                if sm_key in _SUBSET_MEASURE_CACHE:
                    subset_val = _SUBSET_MEASURE_CACHE[sm_key]
                else:
                    pts_est = geom.sample_interior(num_subset_samples, key=rng)
                    pts_val = jnp.asarray(pts_est, dtype=float)
                    mask = jax.vmap(where)(pts_val).astype(float)
                    frac = float(jnp.mean(mask))
                    subset_val = frac * float(jnp.asarray(measure_val))
                    if cache_subset_measure:
                        _SUBSET_MEASURE_CACHE[sm_key] = subset_val
            w = jnp.full((num_points,), subset_val / float(num_points))
        else:
            w = jnp.full(
                (num_points,), float(jnp.asarray(measure_val)) / float(num_points)
            )
    else:
        if sampler is None:
            pts = geom.sample_boundary(num_points, where=where, key=rng)
        else:
            pts = geom.sample_boundary(num_points, where=where, sampler=sampler, key=rng)
        measure_val = _domain_total_measure(geom, over=over)
        if where is not None and (use_subset_measure or subset_measure is not None):
            if subset_measure is not None:
                subset_val = float(subset_measure)
            else:
                sm_key = (id(domain), over, id(where))
                if sm_key in _SUBSET_MEASURE_CACHE:
                    subset_val = _SUBSET_MEASURE_CACHE[sm_key]
                else:
                    pts_est = geom.sample_boundary(num_subset_samples, key=rng)
                    pts_val = jnp.asarray(pts_est, dtype=float)
                    mask = jax.vmap(where)(pts_val).astype(float)
                    frac = float(jnp.mean(mask))
                    subset_val = frac * float(jnp.asarray(measure_val))
                    if cache_subset_measure:
                        _SUBSET_MEASURE_CACHE[sm_key] = subset_val
            w = jnp.full((num_points,), subset_val / float(num_points))
        else:
            w = jnp.full(
                (num_points,), float(jnp.asarray(measure_val)) / float(num_points)
            )

    quad = {"points": pts, "weights": w}
    _QUAD_CACHE[key] = quad
    return quad


def build_ball_quadrature(
    *,
    radius: float,
    dim: int,
    num_points: int = 2048,
    seed: int = 0,
    method: Literal["fibonacci", "grid"] = "fibonacci",
    angular_design: str | None = None,
    radial_strata: int | list[float] | None = None,
) -> dict:
    r"""Precompute isotropic quadrature offsets for a ball.

    Constructs a rule on the Euclidean ball

    $$
    B_r(0) = \{\xi\in\mathbb{R}^d : \|\xi\|_2 \le r\},
    $$

    returning a dict with:
    - `"offsets"`: array of shape `(N, dim)` containing offsets $\xi_i$;
    - `"weights"`: array of shape `(N,)` with equal weights
      $w_i = |B_r(0)|/N$.

    This is intended for local/nonlocal operators of the form:

    $$
    \int_{B_r(0)} \mathcal{I}(x, x+\xi, \ldots)\,\mathrm{d}\xi
    \approx \sum_{i=1}^{N} w_i\,\mathcal{I}(x, x+\xi_i, \ldots).
    $$
    """
    cache_key = (
        float(radius),
        int(dim),
        int(num_points),
        int(seed),
        method,
        angular_design,
        None
        if radial_strata is None
        else (
            tuple(radial_strata)
            if isinstance(radial_strata, (list, tuple))
            else int(radial_strata)
        ),
    )
    if cache_key in _QUAD_CACHE:
        quad = _QUAD_CACHE[cache_key]
        if "offsets" in quad and "weights" in quad:
            return quad

    rng = jr.key(seed)

    def _octa_dirs():
        return jnp.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    def _ico_dirs():
        phi = (1.0 + jnp.sqrt(5.0)) / 2.0
        v = []
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                v.append([0.0, s1, s2 * phi])
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                v.append([s1, s2 * phi, 0.0])
        for s1 in (-1.0, 1.0):
            for s2 in (-1.0, 1.0):
                v.append([s1 * phi, 0.0, s2])
        dirs = jnp.asarray(v, dtype=float)
        dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs

    def _axis4_dirs_2d():
        return jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=float)

    def _random_qmc_dirs(dim: int, n: int, key):
        from ...domain._sampling import get_sampler as _get

        samp = _get("sobol_scrambled")
        U = samp(n, 2 if dim == 3 else 1, key)
        if dim == 3:
            theta = 2.0 * jnp.pi * U[:, 0]
            z = 2.0 * U[:, 1] - 1.0
            r = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
            dirs = jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta), z], axis=1)
        else:
            theta = 2.0 * jnp.pi * U.squeeze(-1)
            dirs = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
        return dirs

    if angular_design:
        ad = angular_design.lower()
        if dim == 3 and ad in ("octa6", "ico12", "random_qmc"):
            if ad == "octa6":
                dirs = _octa_dirs()
            elif ad == "ico12":
                dirs = _ico_dirs()
            else:
                dirs = _random_qmc_dirs(3, max(1, num_points // 2), rng)
            n_ang = dirs.shape[0]
            if radial_strata is None:
                k = max(1, num_points // n_ang)
                rr = jnp.linspace(1.0 / (k + 1), k / (k + 1), k)
                rr = rr ** (1.0 / dim)
            elif isinstance(radial_strata, (list, tuple)):
                edges = jnp.asarray(radial_strata, dtype=float)
                rr = jnp.power(edges, 1.0 / dim)
            else:
                k = int(radial_strata)
                rr = jnp.linspace(1.0 / (k + 1), k / (k + 1), k)
                rr = rr ** (1.0 / dim)
            dirs_rep = jnp.repeat(dirs, rr.shape[0], axis=0)
            rr_rep = jnp.tile(rr, n_ang)
        elif dim == 2 and ad in ("axis4", "random_qmc"):
            if ad == "axis4":
                dirs = _axis4_dirs_2d()
            else:
                dirs = _random_qmc_dirs(2, max(1, num_points // 2), rng)
            n_ang = dirs.shape[0]
            if radial_strata is None:
                k = max(1, num_points // n_ang)
                rr = jnp.linspace(1.0 / (k + 1), k / (k + 1), k)
                rr = rr ** (1.0 / dim)
            elif isinstance(radial_strata, (list, tuple)):
                edges = jnp.asarray(radial_strata, dtype=float)
                rr = jnp.power(edges, 1.0 / dim)
            else:
                k = int(radial_strata)
                rr = jnp.linspace(1.0 / (k + 1), k / (k + 1), k)
                rr = rr ** (1.0 / dim)
            dirs_rep = jnp.repeat(dirs, rr.shape[0], axis=0)
            rr_rep = jnp.tile(rr, n_ang)
        else:
            dirs_rep = None
    else:
        dirs_rep = None

    if dirs_rep is None:
        k = jnp.arange(num_points, dtype=float) + 0.5
        if dim == 1:
            theta = 2.39996322972865332 * k
            dirs0 = jnp.sign(jnp.sin(theta))[:, None]
        elif dim == 2:
            theta = 2.39996322972865332 * k
            dirs0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=1)
        else:
            cz = (2 * k / num_points) - 1.0
            sz = jnp.sqrt(jnp.maximum(0.0, 1.0 - cz**2))
            theta = 2.39996322972865332 * k
            dirs0 = jnp.stack([jnp.cos(theta) * sz, jnp.sin(theta) * sz, cz], axis=1)
        rr = jnp.power(jnp.linspace(1e-6, 1.0, num_points), 1.0 / dim)
        dirs_rep = dirs0[:, :dim]
        rr_rep = rr

    offsets = (radius * rr_rep)[:, None] * dirs_rep

    vol_ball = jnp.pi ** (dim / 2.0) / jsp.gamma(dim / 2.0 + 1.0) * (radius**dim)
    n_eff = offsets.shape[0]
    w = jnp.full((n_eff,), float(vol_ball) / float(n_eff))
    quad = {"offsets": offsets, "weights": w}
    _QUAD_CACHE[cache_key] = quad
    return quad
