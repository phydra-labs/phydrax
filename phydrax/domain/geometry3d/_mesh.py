#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""CAD-backed 3D geometry implementation."""

import functools as ft
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import trimesh
from jaxtyping import Array, ArrayLike, Bool, Float, Key

from ..._doc import DOC_KEY0
from ._base import _AbstractGeometry3D
from ._utils import (
    _boolean_mesh,
    _canonicalize_mesh_arrays,
    _MESH_EQ_DECIMALS,
    _sanitize_mesh,
)


class Geometry3DFromCAD(_AbstractGeometry3D):
    r"""A 3D geometry represented by a watertight surface mesh.

    This class treats a closed triangulated surface mesh as defining a solid region
    $\Omega\subset\mathbb{R}^3$. Sampling routines draw:

    - boundary points $x\in\partial\Omega$ by sampling triangles proportional to their area;
    - interior points $x\in\Omega$ via mesh-based strategies (see `sample_interior`).

    A smooth signed distance-like function $\phi(x)$ is provided via `adf`, and is
    used for containment tests and for estimating boundary normals.
    """

    mesh: trimesh.Trimesh
    diameter: float
    mesh_vertices: Array
    mesh_faces: Array
    mesh_bounds: Array
    triangle_probs: Array
    surface_area_value: Array
    volume_proportion: Array
    immersed: bool
    adf: Callable[[Array], Array]
    _boundary_normals_field: Callable[[Array], Array]

    def __init__(
        self,
        mesh: trimesh.Trimesh | meshio.Mesh | Path | str,
        *,
        recenter: bool = False,
        immersed: bool = False,
    ):
        cleanup_path: Path | None = None
        if isinstance(mesh, (Path, str)):
            mesh_path = Path(mesh).resolve(strict=True)
            from ._cad_io import _maybe_convert_cad_to_stl

            mesh_path, cleanup_path = _maybe_convert_cad_to_stl(mesh_path)
            mesh = mesh_path

        self.mesh = _sanitize_mesh(mesh, recenter=recenter)
        if cleanup_path is not None:
            cleanup_path.unlink()

        self.mesh_vertices = jnp.array(self.mesh.vertices, dtype=float)
        self.mesh_faces = jnp.array(self.mesh.faces, dtype=int)
        self.mesh_bounds = jnp.array(self.mesh.bounds, dtype=float)
        self.diameter = self._max_pairwise_distance

        triangle_areas = jnp.array(self.mesh.area_faces, dtype=float)
        self.triangle_probs = triangle_areas / jnp.sum(triangle_areas)
        self.surface_area_value = jnp.array(self.mesh.area, dtype=float)

        min_bounds, max_bounds = self.mesh.bounds
        bounds = max_bounds - min_bounds
        self.volume_proportion = jnp.array(
            self.mesh.volume / np.prod(bounds), dtype=float
        )
        self.immersed = immersed

        self.adf = self.adf_blur(self.adf_orig, radius_fn=self.adf_orig)
        from ._normals import make_smooth_mesh_normal_field

        self._boundary_normals_field = make_smooth_mesh_normal_field(self)

    @ft.cached_property
    def _mesh_signature(self) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.asarray(self.mesh.vertices, dtype=float)
        faces = np.asarray(self.mesh.faces, dtype=np.int64)
        return _canonicalize_mesh_arrays(
            vertices,
            faces,
            decimals=_MESH_EQ_DECIMALS,
        )

    def equivalent(self, other: object, /) -> bool:
        if not isinstance(other, Geometry3DFromCAD):
            return False
        va, fa = self._mesh_signature
        vb, fb = other._mesh_signature
        if va.shape != vb.shape or fa.shape != fb.shape:
            return False
        return np.array_equal(va, vb) and np.array_equal(fa, fb)

    @ft.cached_property
    def adf_alt(self) -> Callable[[Array], Array]:
        """Alternative ADF using an exact BVH traversal for the `d_exact` term."""
        from ._sdf import make_smooth_mesh_sdf_alt

        return make_smooth_mesh_sdf_alt(self)

    @ft.cached_property
    def adf_orig(self) -> Callable[[Array], Array]:
        """Unsquashed smooth distance (returns `d` before tanh saturation)."""
        from ._sdf import make_smooth_mesh_sdf

        return make_smooth_mesh_sdf(self, squash=False)

    def adf_blur(
        self,
        fn: Callable[[Array], Array],
        /,
        *,
        sigma_scale: float = 0.69420,
        num_samples: int = 12,
        smooth_abs_eps: float = 1e-6,
        min_radius: float = 1e-3,
        eps: float = 1e-12,
        radius_fn: Callable[[Array], Array] | None = None,
    ) -> Callable[[Array], Array]:
        r"""Gaussian-blur a pointwise function using an ADF-adaptive radius.

        Given a function $f:\mathbb{R}^3\to\mathbb{R}^k$, this returns a new function
        $\tilde f$ defined by sampling $f$ in a local neighborhood of each query point:

        $$
        \tilde f(x) = \sum_{i=1}^{m} w_i\, f\!\left(x + r(x)\,u_i\right),
        \qquad r(x) = |\phi(x)|,
        $$

        where $\phi$ is this geometry's `adf` (or `radius_fn` if provided), $(u_i)$ are
        deterministic offsets in the unit ball (a Fibonacci sphere + radial stratification),
        and $(w_i)$ are Gaussian weights with $\sigma = \text{sigma_scale}\cdot r(x)$. When
        $\sigma\propto r$, the weights depend only on $\|u_i\|$ and the effective smoothing
        radius grows with $|\phi(x)|$.
        """
        if float(sigma_scale) <= 0.0:
            raise ValueError(f"sigma_scale must be positive, got {sigma_scale}.")
        if int(num_samples) <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if float(smooth_abs_eps) <= 0.0:
            raise ValueError(f"smooth_abs_eps must be positive, got {smooth_abs_eps}.")
        if float(min_radius) < 0.0:
            raise ValueError(f"min_radius must be non-negative, got {min_radius}.")

        m = int(num_samples)
        i = jnp.arange(m, dtype=float)
        golden_angle = jnp.pi * (3.0 - jnp.sqrt(5.0))
        theta = golden_angle * i
        z = 1.0 - 2.0 * (i + 0.5) / float(m)
        r_xy = jnp.sqrt(jnp.maximum(1.0 - z**2, 0.0))
        dirs = jnp.stack([r_xy * jnp.cos(theta), r_xy * jnp.sin(theta), z], axis=1)
        radius_unit = ((i + 0.5) / float(m)) ** (1.0 / 3.0)
        offsets = radius_unit[:, None] * dirs
        offsets_norm2 = jnp.sum(offsets**2, axis=1)
        adf_fn = self.adf if radius_fn is None else radius_fn
        smooth_abs_eps_const = jnp.asarray(float(smooth_abs_eps), dtype=float)
        min_radius_const = jnp.asarray(float(min_radius), dtype=float)

        def blurred_single(p: Array) -> Array:
            p = jnp.asarray(p, dtype=float)
            if p.ndim == 0:
                p = jnp.array([p, 0.0, 0.0], dtype=p.dtype)
            elif p.ndim > 1:
                p = p.reshape(-1)[-3:]
            elif p.shape[0] != 3:
                n = int(p.shape[0])
                if n < 3:
                    buf = jnp.zeros((3,), dtype=p.dtype)
                    buf = buf.at[:n].set(p)
                    p = buf
                else:
                    p = p[-3:]

            adf_p = adf_fn(p)
            r = jnp.sqrt(adf_p**2 + smooth_abs_eps_const**2) - smooth_abs_eps_const
            r = jnp.maximum(r, min_radius_const)
            sigma2 = (sigma_scale * r) ** 2
            w_offsets = jnp.exp(-(r**2 * offsets_norm2) / (2.0 * sigma2 + eps))

            center_val = jnp.asarray(adf_p) if fn is adf_fn else fn(p)
            pts = p[None, :] + r * offsets
            offset_vals = jax.vmap(fn)(pts)
            num = center_val + jnp.tensordot(w_offsets, offset_vals, axes=1)
            den = 1.0 + jnp.sum(w_offsets)
            return num / (den + eps)

        def blurred(points: Array) -> Array:
            pts = jnp.asarray(points, dtype=float)
            if pts.ndim <= 1:
                return blurred_single(pts)
            if pts.shape[-1] != 3:
                return blurred_single(pts)
            out_shape = pts.shape[:-1]
            flat = pts.reshape((-1, 3))
            vals = jax.vmap(blurred_single)(flat)
            return vals.reshape(out_shape)

        return blurred

    # --- CSG Operators ---------------------------------------------------
    def _csg(
        self,
        other: "Geometry3DFromCAD",
        op: Literal["union", "difference", "intersection"],
    ) -> "Geometry3DFromCAD":
        if not isinstance(other, Geometry3DFromCAD):
            raise TypeError(
                f"Boolean ops require Geometry3DFromCAD, got {type(other).__name__}"
            )
        # Force the Manifold backend for 3D boolean ops
        result = _boolean_mesh(
            [self.mesh, other.mesh],
            operation=op,
            engine="manifold",
        )
        return Geometry3DFromCAD(result, recenter=True)

    def __add__(self, other: "Geometry3DFromCAD") -> "Geometry3DFromCAD":
        r"""Boolean union of two solids: $\Omega = \Omega_1 \cup \Omega_2$."""
        return self._csg(other, "union")

    def __sub__(self, other: "Geometry3DFromCAD") -> "Geometry3DFromCAD":
        r"""Boolean difference of two solids: $\Omega = \Omega_1 \setminus \Omega_2$."""
        return self._csg(other, "difference")

    def __and__(self, other: "Geometry3DFromCAD") -> "Geometry3DFromCAD":
        r"""Boolean intersection of two solids: $\Omega = \Omega_1 \cap \Omega_2$."""
        return self._csg(other, "intersection")

    # --- Rigid transforms ------------------------------------------------
    def translate(self, offset: ArrayLike) -> "Geometry3DFromCAD":
        r"""Return a translated copy of the geometry.

        Applies the rigid transform $x \mapsto x + b$ with translation vector
        $b\in\mathbb{R}^3$.
        """
        vec = jnp.asarray(offset, dtype=float)
        if vec.shape != (3,):
            raise ValueError(f"translate offset must have shape (3,), got {vec.shape!r}")
        mesh = self.mesh.copy()
        mesh.apply_translation(np.asarray(vec, dtype=float))
        return Geometry3DFromCAD(
            mesh,
            recenter=False,
            immersed=self.immersed,
        )

    def scale(self, factor: ArrayLike) -> "Geometry3DFromCAD":
        r"""Return a uniformly scaled copy of the geometry.

        Applies the dilation $x \mapsto s x$ with scalar $s>0$.
        """
        factor_arr = jnp.asarray(factor, dtype=float)
        if factor_arr.ndim != 0:
            raise ValueError(
                f"scale factor must be a scalar (shape ()), got shape {factor_arr.shape!r}"
            )
        scale_val = float(factor_arr)
        mesh = self.mesh.copy()
        mesh.apply_scale(scale_val)
        return Geometry3DFromCAD(
            mesh,
            recenter=False,
            immersed=self.immersed,
        )

    def _make_mesh_sdf(
        self, *, eps: float | None = None
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        from ._sdf import make_mesh_sdf

        return make_mesh_sdf(self, eps=eps)

    def _make_smooth_mesh_sdf(
        self,
        *,
        eps: float | None = None,
        beta: float | None = None,
        beam_width: int = 32,
        beam_steps: int | None = None,
        r0: float | None = None,
        rho_delta: float | None = None,
        squash: bool = True,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        from ._sdf import make_smooth_mesh_sdf

        return make_smooth_mesh_sdf(
            self,
            eps=eps,
            beta=beta,
            beam_width=beam_width,
            beam_steps=beam_steps,
            r0=r0,
            rho_delta=rho_delta,
            squash=squash,
        )

    @ft.cached_property
    def volume(self) -> Float[Array, ""]:
        return jnp.array(self.mesh.volume, dtype=float)

    @property
    def boundary_measure_value(self) -> Float[Array, ""]:
        """Alias for the total boundary measure value (surface area)."""
        return self.surface_area_value

    def estimate_boundary_subset_measure(
        self,
        where,
        *,
        num_samples: int = 4096,
        key=DOC_KEY0,
    ) -> Array:
        """Estimate boundary subset area via Monte Carlo proportion."""
        pts = self.sample_boundary(num_samples, key=key)
        mask = jax.vmap(where)(pts).astype(float)
        frac = jnp.mean(mask)
        return frac * self.surface_area_value

    @property
    def bounds(self) -> Float[Array, "2 3"]:
        return jnp.asarray(self.mesh_bounds, dtype=float)

    @ft.cached_property
    def _max_pairwise_distance(self) -> float:
        verts = self.mesh_vertices
        # Pairwise squared distances via broadcasting
        x_diff = verts[:, None, 0] - verts[None, :, 0]
        y_diff = verts[:, None, 1] - verts[None, :, 1]
        z_diff = verts[:, None, 2] - verts[None, :, 2]
        squared_distances = x_diff**2 + y_diff**2 + z_diff**2
        max_squared_distance = jnp.max(squared_distances)
        return jnp.sqrt(max_squared_distance + jnp.finfo(float).eps).item()

    def _boundary_normals_orig_nograd(
        self, points: Array
    ) -> Float[Array, "num_points 3"]:
        return jnp.asarray(self._boundary_normals_field(points), dtype=float)

    def _boundary_normals(self, points: Array) -> Float[Array, "num_points 3"]:
        return jnp.asarray(self._boundary_normals_field(points), dtype=float)

    def _sample_barycentric_coords(
        self, num_points: int, sampler: str, key: Key[Array, ""]
    ) -> Array:
        from ._sampling import _sample_barycentric_coords as _impl

        return _impl(self, num_points, sampler, key)

    def sample_boundary(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Sample points from the boundary $\partial\Omega$.

        Returns an array of shape `(num_points, 3)` containing boundary points,
        optionally filtered by `where(x)`.
        """
        from ._sampling import sample_boundary as _impl

        return _impl(
            self,
            num_points,
            where=where,
            sampler=sampler,
            key=key,
        )

    def sample_interior(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Sample points from the interior of $\Omega$.

        Returns an array of shape `(num_points, 3)` containing points in
        $\Omega\subset\mathbb{R}^3$, optionally filtered by `where(x)`.
        """
        from ._sampling import sample_interior as _impl

        return _impl(
            self,
            num_points,
            where=where,
            sampler=sampler,
            key=key,
        )

    def _sample_interior_separable(
        self,
        num_points: int | Sequence[int],
        *,
        sampler: str = "latin_hypercube",
        where: Callable | None = None,
        add_delta: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> tuple[
        tuple[Array, ...],
        Bool[Array, "..."],
    ]:
        from ._sampling import _sample_interior_separable as _impl

        num_points_xyz: Sequence[int]
        if isinstance(num_points, int):
            num_points_xyz = (num_points, num_points, num_points)
        else:
            num_points_xyz = num_points

        return _impl(
            self,
            num_points_xyz,
            sampler=sampler,
            where=where,
            add_delta=add_delta,
            key=key,
        )

    def _contains(self, points: Array) -> Bool[Array, " num_points"]:
        pts = jnp.asarray(points, dtype=float)
        if pts.ndim <= 1:
            return self.adf_orig(pts) <= 1e-8
        return self.adf_orig(pts) <= 1e-8

    def _on_boundary(self, points: Array) -> Bool[Array, " num_points"]:
        pts = jnp.asarray(points, dtype=float)
        if pts.ndim <= 1:
            return jnp.isclose(self.adf_orig(pts), 0.0, atol=1e-8)
        return jnp.isclose(self.adf_orig(pts), 0.0, atol=1e-8)
