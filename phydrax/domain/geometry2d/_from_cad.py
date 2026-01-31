#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable, Sequence
from functools import cached_property
from pathlib import Path
from typing import Literal
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import meshio
import numpy as np
import opt_einsum as oe
import pyvista as pv
import trimesh
from jaxtyping import Array, ArrayLike, Bool, Float, Key

from ..._doc import DOC_KEY0
from .._sampling import get_sampler_host, seed_from_key
from ..geometry3d._mesh import Geometry3DFromCAD
from ..geometry3d._utils import (
    _boolean_mesh,
    _canonicalize_mesh_arrays,
    _MESH_EQ_DECIMALS,
    _sanitize_meshio_mesh,
    _z0_faces_to_meshio,
)
from ._base import _AbstractGeometry2D


class Geometry2DFromCAD(_AbstractGeometry2D):
    r"""A 2D geometry represented by a triangulated surface mesh.

    This class treats a watertight triangulated mesh as defining a planar region
    $\Omega\subset\mathbb{R}^2$. Sampling routines draw:

    - interior points $x\in\Omega$ by sampling triangles proportional to their area;
    - boundary points $x\in\partial\Omega$ by sampling edges proportional to length.

    The geometry also provides a smooth signed distance-like function $\phi(x)$ (via
    `adf`) that is used for containment tests and for estimating normals when needed.
    """

    mesh: meshio.Mesh
    geom_3d_extruded: Geometry3DFromCAD
    diameter: float
    mesh_vertices: Array
    mesh_faces: Array
    boundary_edges: Array
    interior_edges: Array
    _boundary_length_value: Array
    _boundary_measure_value: Array
    triangle_probs: Array
    area_faces: Array
    area_proportion: Array
    boundary_probs: Array
    interior_edge_probs: Array
    adf: Callable[[Array], Array]

    def __init__(
        self,
        mesh: meshio.Mesh | Path | str,
        *,
        recenter: bool = True,
    ):
        if isinstance(mesh, Path | str):
            mesh: meshio.Mesh = meshio.read(mesh)
        self.mesh: meshio.Mesh = _sanitize_meshio_mesh(
            mesh,
            output_type="meshio",
            recenter=recenter,
        )

        self.mesh_vertices = jnp.array(self.mesh.points, dtype=float)
        self.mesh_faces = jnp.array(self.mesh.cells[0].data, dtype=int)
        self.boundary_edges = self._boundary_edges
        self.interior_edges = self._interior_edges
        self.geom_3d_extruded = self._geom_3d_extruded
        self.diameter = self._max_pairwise_distance

        trimesh_mesh = trimesh.Trimesh(
            vertices=self.mesh.points, faces=self.mesh.cells[0].data
        )
        self.area_faces = jnp.array(trimesh_mesh.area_faces, dtype=float)

        bounds = jnp.asarray(self.bounds, dtype=float)
        xmin, ymin = bounds[0]
        xmax, ymax = bounds[1]
        bbox_area = (xmax - xmin) * (ymax - ymin)
        self.area_proportion = jnp.sum(self.area_faces) / bbox_area

        edge_vertices = self.mesh_vertices[self.boundary_edges]
        edge_lengths = jnp.linalg.norm(edge_vertices[:, 1] - edge_vertices[:, 0], axis=1)
        self._boundary_length_value = jnp.sum(edge_lengths)
        self._boundary_measure_value = self._boundary_length_value
        self.boundary_probs = edge_lengths / self._boundary_length_value

        if self._interior_edges.shape[0] > 0:
            i_edge_vertices = self.mesh_vertices[self._interior_edges]
            i_edge_lengths = jnp.linalg.norm(
                i_edge_vertices[:, 1] - i_edge_vertices[:, 0], axis=1
            )
            self.interior_edge_probs = i_edge_lengths / jnp.sum(i_edge_lengths)
        else:
            # Empty interior (e.g., single triangle). Keep a well-typed empty array.
            self.interior_edge_probs = jnp.array([], dtype=float)

        self.triangle_probs = self.area_faces / jnp.sum(self.area_faces)

        self.adf = self.adf_blur(self.adf_orig, radius_fn=self.adf_orig)

    @cached_property
    def _mesh_signature(self) -> tuple[np.ndarray, np.ndarray]:
        vertices = np.asarray(self.mesh.points, dtype=float)
        faces = np.asarray(self.mesh.cells[0].data, dtype=np.int64)
        return _canonicalize_mesh_arrays(
            vertices,
            faces,
            decimals=_MESH_EQ_DECIMALS,
        )

    def equivalent(self, other: object, /) -> bool:
        if not isinstance(other, Geometry2DFromCAD):
            return False
        va, fa = self._mesh_signature
        vb, fb = other._mesh_signature
        if va.shape != vb.shape or fa.shape != fb.shape:
            return False
        return np.array_equal(va, vb) and np.array_equal(fa, fb)

    @property
    def boundary_length_value(self) -> Float[Array, ""]:
        """Total boundary length (dimensionless value)."""
        return self._boundary_length_value

    @property
    def boundary_measure_value(self) -> Float[Array, ""]:
        """Alias for the total boundary measure value (boundary length)."""
        return self._boundary_measure_value

    def _make_smooth_mesh_sdf(
        self,
        *,
        eps: float | None = None,
        beta: float | None = None,
        beam_width: int = 64,
        beam_steps: int | None = None,
        r0: float | None = None,
        rho_delta: float | None = None,
        squash: bool = True,
    ) -> Callable[[Array], Array]:
        sdf_3d = self.geom_3d_extruded._make_smooth_mesh_sdf(
            eps=eps,
            beta=beta,
            beam_width=beam_width,
            beam_steps=beam_steps,
            r0=r0,
            rho_delta=rho_delta,
            squash=squash,
        )
        z0 = jnp.asarray(self.diameter * 2.0, dtype=float)

        def _coerce_points2(
            points: Array,
        ) -> tuple[Array, tuple[int, ...], bool]:
            pts = jnp.asarray(points, dtype=float)
            if pts.ndim == 0:
                pts = jnp.array([pts, 0.0], dtype=pts.dtype)
                return pts.reshape((1, 2)), (), True
            if pts.ndim == 1:
                if pts.shape[0] == 2:
                    return pts.reshape((1, 2)), (), True
                n = int(pts.shape[0])
                if n < 2:
                    buf = jnp.zeros((2,), dtype=pts.dtype)
                    buf = buf.at[:n].set(pts)
                    pts = buf
                else:
                    pts = pts[-2:]
                return pts.reshape((1, 2)), (), True
            if pts.shape[-1] != 2:
                pts = pts.reshape(-1)[-2:]
                return pts.reshape((1, 2)), (), True
            out_shape = pts.shape[:-1]
            return pts.reshape((-1, 2)), out_shape, False

        def sdf_2d(points: Array) -> Array:
            pts, out_shape, is_single = _coerce_points2(points)
            z = jnp.full((pts.shape[0], 1), z0, dtype=pts.dtype)
            p3 = jnp.concatenate([pts, z], axis=1)
            sdf = sdf_3d(p3)
            if is_single:
                return sdf.reshape(())
            return sdf.reshape(out_shape)

        return jax.jit(sdf_2d)

    @cached_property
    def adf_orig(self) -> Callable[[Array], Array]:
        """Unsquashed smooth distance (returns `d` before tanh saturation)."""
        return self._make_smooth_mesh_sdf(squash=False)

    def adf_blur(
        self,
        fn: Callable[[Array], Array],
        /,
        *,
        sigma_scale: float = 0.69420,
        num_samples: int = 6,
        smooth_abs_eps: float = 1e-6,
        min_radius: float = 1e-3,
        eps: float = 1e-12,
        radius_fn: Callable[[Array], Array] | None = None,
    ) -> Callable[[Array], Array]:
        r"""Gaussian-blur a pointwise function using an ADF-adaptive radius.

        Given a function $f:\mathbb{R}^2\to\mathbb{R}^k$, this returns a new function
        $\tilde f$ defined by sampling $f$ in a local neighborhood of each query point:

        $$
        \tilde f(x) = \sum_{i=1}^{m} w_i\, f\!\left(x + r(x)\,u_i\right),
        \qquad r(x) = |\phi(x)|,
        $$

        where $\phi$ is this geometry's `adf` (or `radius_fn` if provided), $(u_i)$ are
        deterministic offsets on the unit disk (a Fibonacci disk pattern), and $(w_i)$
        are Gaussian weights with $\sigma = \text{sigma_scale}\cdot r(x)$. When
        $\sigma\propto r$, the weights depend only on $\|u_i\|$ and the effective
        smoothing radius grows with $|\phi(x)|$.
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
        radius_unit = jnp.sqrt((i + 0.5) / float(m))
        offsets = radius_unit[:, None] * jnp.stack(
            [jnp.cos(theta), jnp.sin(theta)], axis=1
        )
        offsets_norm2 = jnp.sum(offsets**2, axis=1)
        adf_fn = self.adf if radius_fn is None else radius_fn
        smooth_abs_eps_const = jnp.asarray(float(smooth_abs_eps), dtype=float)
        min_radius_const = jnp.asarray(float(min_radius), dtype=float)

        def blurred_single(p: Array) -> Array:
            p = jnp.asarray(p, dtype=float)
            if p.ndim == 0:
                p = jnp.array([p, 0.0], dtype=p.dtype)
            elif p.ndim > 1:
                p = p.reshape(-1)[-2:]
            elif p.shape[0] != 2:
                n = int(p.shape[0])
                if n < 2:
                    buf = jnp.zeros((2,), dtype=p.dtype)
                    buf = buf.at[:n].set(p)
                    p = buf
                else:
                    p = p[-2:]

            adf_p = adf_fn(p)
            r = jnp.sqrt(adf_p**2 + smooth_abs_eps_const**2) - smooth_abs_eps_const
            r = jnp.maximum(r, min_radius_const)
            sigma2 = (sigma_scale * r) ** 2
            w_offsets = jnp.exp(-(r**2 * offsets_norm2) / (2.0 * sigma2 + eps))

            center_val = jnp.asarray(adf_p) if fn is adf_fn else fn(p)
            pts = p[None, :] + r * offsets
            offset_vals = jax.vmap(fn)(pts)
            num = center_val + oe.contract("n,n...->...", w_offsets, offset_vals)
            den = 1.0 + jnp.sum(w_offsets)
            return num / (den + eps)

        def blurred(points: Array) -> Array:
            pts = jnp.asarray(points, dtype=float)
            if pts.ndim <= 1:
                return blurred_single(pts)
            if pts.shape[-1] != 2:
                return blurred_single(pts)
            out_shape = pts.shape[:-1]
            flat = pts.reshape((-1, 2))
            vals = jax.vmap(blurred_single)(flat)
            return vals.reshape(out_shape)

        return blurred

    @cached_property
    def adf_alt(self) -> Callable[[Array], Array]:
        """Alternative ADF using an exact BVH traversal for the `d_exact` term."""
        return self._make_smooth_mesh_sdf_alt()

    def _make_smooth_mesh_sdf_alt(self) -> Callable[[Array], Array]:
        sdf_3d = self.geom_3d_extruded.adf_alt
        # Evaluate the 3D (extruded) ADF away from the end-caps so that the
        # nearest surface is the vertical sidewall and the induced distance
        # coincides with the 2D boundary distance.
        z0 = jnp.asarray(self.diameter * 2.0, dtype=float)

        def _coerce_points2(
            points: Array,
        ) -> tuple[Array, tuple[int, ...], bool]:
            pts = jnp.asarray(points, dtype=float)
            if pts.ndim == 0:
                p2 = jnp.repeat(pts, 2)
                return p2.reshape((1, 2)), (), True
            if pts.ndim == 1:
                if pts.shape[0] == 2:
                    return pts.reshape((1, 2)), (), True
                n = int(pts.shape[0])
                if n < 2:
                    buf = jnp.zeros((2,), dtype=pts.dtype)
                    buf = buf.at[:n].set(pts)
                    pts = buf
                else:
                    pts = pts[-2:]
                return pts.reshape((1, 2)), (), True
            if pts.shape[-1] != 2:
                pts = pts.reshape(-1)[-2:]
                return pts.reshape((1, 2)), (), True
            out_shape = pts.shape[:-1]
            return pts.reshape((-1, 2)), out_shape, False

        def sdf_2d(points: Array) -> Array:
            pts, out_shape, is_single = _coerce_points2(points)
            z = jnp.full((pts.shape[0], 1), z0, dtype=pts.dtype)
            p3 = jnp.concatenate([pts, z], axis=1)
            sdf = sdf_3d(p3)
            if is_single:
                return sdf.reshape(())
            return sdf.reshape(out_shape)

        return jax.jit(sdf_2d)

    # --- CSG Operators via 3D extrusion ---------------------------------
    def _csg(
        self,
        other: "Geometry2DFromCAD",
        op: Literal["union", "difference", "intersection"],
    ) -> "Geometry2DFromCAD":
        if not isinstance(other, Geometry2DFromCAD):
            raise TypeError(
                f"Boolean ops require Geometry2DFromCAD, got {type(other).__name__}"
            )
        # Perform 3D boolean on extrusions and slice z=0 faces back to 2D
        a3 = self._geom_3d_extruded.mesh
        b3 = other._geom_3d_extruded.mesh
        # Force the Manifold backend for boolean ops
        res3 = _boolean_mesh(
            [a3, b3],
            operation=op,
            engine="manifold",
            check_volume=True,
        )
        tol = 1e-6 * float(max(self.diameter, other.diameter) or 1.0)
        mesh2d = _z0_faces_to_meshio(res3, tol=tol)
        # Construct new 2D geometry; preserve original coordinates (no recenter)
        return Geometry2DFromCAD(mesh2d, recenter=False)

    def __add__(self, other: "Geometry2DFromCAD") -> "Geometry2DFromCAD":
        """Union of two 2D geometries via extruded 3D CSG."""
        return self._csg(other, "union")

    def __sub__(self, other: "Geometry2DFromCAD") -> "Geometry2DFromCAD":
        """Difference (self minus other) of two 2D geometries via extruded 3D CSG."""
        return self._csg(other, "difference")

    def __and__(self, other: "Geometry2DFromCAD") -> "Geometry2DFromCAD":
        """Intersection of two 2D geometries via extruded 3D CSG."""
        return self._csg(other, "intersection")

    @property
    def volume_proportion(self) -> Float[Array, ""]:
        return self.area_proportion

    def estimate_boundary_subset_measure(
        self,
        where: Callable[[Array], Bool[Array, ""]],
        *,
        num_samples: int = 4096,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        """Estimate boundary subset length via Monte Carlo proportion."""
        pts = self.sample_boundary(num_samples, key=key)
        mask = jax.vmap(where)(pts).astype(float)
        frac = jnp.mean(mask)
        return frac * self.boundary_length_value

    @cached_property
    def _boundary_edges(self) -> Float[Array, "num_boundary_edges 2"]:
        edges = jnp.vstack(
            (
                self.mesh_faces[:, [0, 1]],
                self.mesh_faces[:, [1, 2]],
                self.mesh_faces[:, [2, 0]],
            )
        )
        edges = jnp.sort(edges, axis=1)
        unique_edges, counts = jnp.unique(edges, axis=0, return_counts=True)
        return unique_edges[counts == 1]

    @cached_property
    def _interior_edges(self) -> Float[Array, "num_interior_edges 2"]:
        edges = jnp.vstack(
            (
                self.mesh_faces[:, [0, 1]],
                self.mesh_faces[:, [1, 2]],
                self.mesh_faces[:, [2, 0]],
            )
        )
        edges = jnp.sort(edges, axis=1)
        unique_edges, counts = jnp.unique(edges, axis=0, return_counts=True)
        return unique_edges[counts > 1]

    @cached_property
    def _max_pairwise_distance(self) -> float:
        mesh_vertices = self.mesh_vertices[:, :2]
        x_diff = mesh_vertices[:, None, 0] - mesh_vertices[None, :, 0]  # Shape: (N, N)
        y_diff = mesh_vertices[:, None, 1] - mesh_vertices[None, :, 1]  # Shape: (N, N)
        squared_distances = x_diff**2 + y_diff**2

        max_squared_distance = np.max(squared_distances)
        max_distance = np.sqrt(max_squared_distance + np.finfo(float).eps)
        return max_distance

    @cached_property
    def _geom_3d_extruded(self):
        extruded_mesh = trimesh.creation.extrude_triangulation(
            np.array(self.mesh_vertices[:, :2]),
            np.array(self.mesh_faces),
            height=self._max_pairwise_distance * 4,
        )
        return Geometry3DFromCAD(
            extruded_mesh,
            recenter=False,
        )

    @cached_property
    def area(self) -> Float[Array, ""]:
        return jnp.sum(self.area_faces)

    @cached_property
    def bounds(self) -> Float[Array, "2 2"]:
        bounds3 = jnp.asarray(self.geom_3d_extruded.bounds, dtype=float)
        return bounds3[:, :2]

    def _sample_barycentric_coords(
        self, num_points: int, sampler: str, num_vertices: int, *, key: Key[Array, ""]
    ) -> Array:
        rng = np.random.default_rng(seed_from_key(key))

        if num_vertices == 3:  # Sampling points in triangles
            sampler_fn = get_sampler_host(sampler, dim=2, seed=rng)
            uv = jnp.asarray(sampler_fn(int(num_points)), dtype=float)
            u = uv[:, :1]
            v = uv[:, 1:2]

            def _uvw(u, v):
                mask = u + v > 1.0
                u = jnp.where(mask, 1.0 - u, u)
                v = jnp.where(mask, 1.0 - v, v)
                w = 1.0 - (u + v)
                return jnp.hstack([u, v, w])

            return _uvw(u, v)
        else:  # Sampling points on edges (num_vertices == 2)
            sampler_fn = get_sampler_host(sampler, dim=1, seed=rng)
            u = jnp.asarray(sampler_fn(int(num_points)), dtype=float)
            return jnp.hstack([u, 1.0 - u])

    def sample_interior(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        interior_edge_fraction: float = 0.0,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Sample points from the interior of $\Omega$.

        Returns an array of shape `(num_points, 2)` containing points in
        $\Omega\subset\mathbb{R}^2$, optionally filtered by `where(x)`.

        The optional `interior_edge_fraction` can be used to bias samples towards
        interior edges (useful for capturing sharp features in PDE solutions).
        """

        def _sample_interior_host(
            num_points, sampler, interior_edge_fraction, where, key
        ):
            sampled_points = np.empty((0, 2), dtype=float)

            # Determine target counts for edge vs. face samples
            frac = (
                float(interior_edge_fraction)
                if interior_edge_fraction is not None
                else 0.0
            )
            frac = max(0.0, min(1.0, frac))
            n_edge_target = int(round(num_points * frac))
            n_face_target = num_points - n_edge_target

            # 1) Sample from interior edges (non-boundary edges)
            if n_edge_target > 0 and self.interior_edges.shape[0] > 0:
                edge_points = np.empty((0, 2), dtype=float)
                while edge_points.shape[0] < n_edge_target:
                    remaining = n_edge_target - edge_points.shape[0]
                    # Split key for edges and barycentric coords, advance key
                    ekey, bkey, key = jr.split(key, 3)
                    edge_indices = jr.choice(
                        ekey,
                        self.interior_edges.shape[0],
                        shape=(remaining,),
                        p=self.interior_edge_probs
                        if self.interior_edge_probs.size
                        else None,
                    )
                    edges = self.interior_edges[edge_indices]
                    # Sample along each edge using barycentric coords for 2 vertices
                    bary = self._sample_barycentric_coords(
                        remaining, sampler, num_vertices=2, key=bkey
                    )
                    verts = self.mesh_vertices[edges]
                    pts_batch = jnp.sum(verts * bary[..., jnp.newaxis], axis=1)[:, :2]
                    # Apply optional user filter
                    if where is not None:
                        where_mask = jax.vmap(where)(pts_batch)
                    else:
                        where_mask = jnp.ones((pts_batch.shape[0],), dtype=bool)

                    # Enforce interior check using original no-grad SDF
                    sdf_mask = jax.vmap(self.adf_orig)(pts_batch) < -1e-6

                    mask = where_mask & sdf_mask
                    pts_batch = pts_batch[mask]
                    edge_points = np.vstack((edge_points, np.asarray(pts_batch)))
                sampled_points = np.vstack((sampled_points, edge_points[:n_edge_target]))

            # 2) Sample from triangle interiors
            if n_face_target > 0:
                face_points = np.empty((0, 2), dtype=float)
                while face_points.shape[0] < n_face_target:
                    remaining = n_face_target - face_points.shape[0]
                    # Split random key for different sampling operations
                    tkey, bkey, key = jr.split(key, 3)

                    # Sample triangles with probability proportional to their area
                    tri_indices = jr.choice(
                        tkey,
                        self.mesh_faces.shape[0],
                        shape=(remaining,),
                        p=self.triangle_probs,
                    )
                    tris = self.mesh_faces[tri_indices]

                    # Generate random points within each triangle using barycentric coordinates
                    bary = self._sample_barycentric_coords(
                        remaining, sampler, num_vertices=3, key=bkey
                    )

                    # Convert barycentric coordinates to 2D points
                    verts = self.mesh_vertices[tris]
                    pts_batch = jnp.sum(verts * bary[..., jnp.newaxis], axis=1)
                    # Expect Z=0
                    assert jnp.allclose(pts_batch[:, 2], 0.0)

                    # Build combined mask: user filter (if any) AND interior SDF check
                    if where is not None:
                        where_mask = jax.vmap(where)(pts_batch)
                    else:
                        where_mask = jnp.ones((pts_batch.shape[0],), dtype=bool)

                    # Check interior using 2D points with original no-grad SDF
                    sdf_mask = jax.vmap(self.adf_orig)(pts_batch[:, :2]) < -1e-6

                    mask = where_mask & sdf_mask
                    pts_batch = pts_batch[mask]

                    # Accumulate valid points (drop Z)
                    face_points = np.vstack((face_points, np.asarray(pts_batch[:, :2])))

                sampled_points = np.vstack((sampled_points, face_points[:n_face_target]))

            # Safety: truncate to exact number of points requested
            return sampled_points[:num_points]

        zeros = jnp.zeros((num_points, 2), dtype=float)
        shape_dtype = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)

        sampled_points = eqx.filter_pure_callback(
            _sample_interior_host,
            num_points,
            sampler,
            interior_edge_fraction,
            where,
            key,
            result_shape_dtypes=shape_dtype,
        )

        return sampled_points

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
        if isinstance(num_points, int):
            num_points_x, num_points_y = num_points, num_points
        else:
            num_points_x, num_points_y = num_points

        def _sample_interior_separable_host(
            num_points_x, num_points_y, sampler, where, key
        ):
            bounds = np.asarray(self.bounds, dtype=float)
            xmin, ymin = bounds[0]
            xmax, ymax = bounds[1]

            if add_delta:
                delta = 1e-2
                xmin = xmin - delta
                xmax = xmax + delta
                ymin = ymin - delta
                ymax = ymax + delta

            xkey, ykey = jr.split(key, 2)  # Split the random key for each dimension

            x_sampler = get_sampler_host(
                sampler, dim=1, seed=np.random.default_rng(seed_from_key(xkey))
            )
            y_sampler = get_sampler_host(
                sampler, dim=1, seed=np.random.default_rng(seed_from_key(ykey))
            )

            sampled_x = (
                jnp.asarray(x_sampler(int(num_points_x)), dtype=float).squeeze()
                * (xmax - xmin)
                + xmin
            )
            sampled_y = (
                jnp.asarray(y_sampler(int(num_points_y)), dtype=float).squeeze()
                * (ymax - ymin)
                + ymin
            )

            grid_x, grid_y = jnp.meshgrid(sampled_x, sampled_y, indexing="ij")
            grid = jnp.column_stack((grid_x.ravel(), grid_y.ravel()))

            mask = jax.vmap(self.adf_orig)(grid) < -1e-6
            if where is not None:
                mask = mask & jax.vmap(where)(grid)

            mask = mask.reshape((sampled_x.size, sampled_y.size))
            return (sampled_x, sampled_y), mask

        zeros_x = jnp.zeros(num_points_x, dtype=float)
        zeros_y = jnp.zeros(num_points_y, dtype=float)
        zeros_mask = jnp.zeros((num_points_x, num_points_y), dtype=bool)

        shape_dtype_x = jax.ShapeDtypeStruct(zeros_x.shape, zeros_x.dtype)
        shape_dtype_y = jax.ShapeDtypeStruct(zeros_y.shape, zeros_y.dtype)
        shape_dtype_mask = jax.ShapeDtypeStruct(zeros_mask.shape, zeros_mask.dtype)

        (sampled_x, sampled_y), mask = eqx.filter_pure_callback(
            _sample_interior_separable_host,
            num_points_x,
            num_points_y,
            sampler,
            where,
            key,
            result_shape_dtypes=(
                (shape_dtype_x, shape_dtype_y),
                shape_dtype_mask,
            ),
        )
        return (sampled_x, sampled_y), mask

    def _boundary_normals(self, points: Array) -> Float[Array, "num_points 2"]:
        pts = jnp.asarray(points, dtype=float)
        squeeze = False
        if pts.ndim == 1:
            pts = pts.reshape((1, 2))
            squeeze = True
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected points with shape (N, 2), got {pts.shape}.")

        zeros = jnp.zeros((pts.shape[0], 1), dtype=pts.dtype)
        points_3d = jnp.concatenate([pts, zeros], axis=1)
        normals_3d = self.geom_3d_extruded._boundary_normals(points_3d)
        normals_2d = normals_3d[:, :2]
        nrm = jnp.linalg.norm(normals_2d, axis=1, keepdims=True) + jnp.finfo(float).eps
        normals_2d = normals_2d / nrm

        if squeeze:
            return normals_2d.reshape((2,))
        return normals_2d

    def sample_boundary(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Sample points from the boundary $\partial\Omega$.

        Returns an array of shape `(num_points, 2)` containing boundary points in
        $\partial\Omega$, optionally filtered by `where(x)`.
        """

        def _sample_boundary_host(num_points, sampler, where, key):
            sampled_points = np.empty((0, 2), dtype=float)

            while sampled_points.shape[0] < num_points:
                remaining_points = num_points - sampled_points.shape[0]
                ekey, bkey, key = jr.split(key, 3)

                sampled_edge_indices = jr.choice(
                    ekey,
                    self.boundary_edges.shape[0],
                    shape=(remaining_points,),
                    p=self.boundary_probs,
                )
                sampled_edges = self.boundary_edges[sampled_edge_indices]

                barycentric_coords = self._sample_barycentric_coords(
                    remaining_points, sampler, num_vertices=2, key=bkey
                )

                vertices = self.mesh_vertices[sampled_edges]
                sampled_points_batch = jnp.sum(
                    vertices * barycentric_coords[..., jnp.newaxis],
                    axis=1,
                )[:, :2]

                if where is not None:
                    inside = jax.vmap(where)(sampled_points_batch)
                    inside = inside.astype(bool)
                    sampled_points_batch = sampled_points_batch[inside]

                sampled_points = np.vstack((sampled_points, sampled_points_batch))

            sampled_points = sampled_points[:num_points]

            return sampled_points

        zeros = jnp.zeros((num_points, 2), dtype=float)
        shape_dtype = jax.ShapeDtypeStruct(zeros.shape, zeros.dtype)

        sampled_points = eqx.filter_pure_callback(
            _sample_boundary_host,
            num_points,
            sampler,
            where,
            key,
            result_shape_dtypes=shape_dtype,
        )
        return sampled_points

    def _contains(self, points: Array) -> Bool[Array, " num_points"]:
        pts = jnp.asarray(points, dtype=float)
        if pts.ndim <= 1:
            return self.adf_orig(pts) <= 1e-6
        return self.adf_orig(pts) <= 1e-6

    def _on_boundary(self, points: Array) -> Bool[Array, " num_points"]:
        pts = jnp.asarray(points, dtype=float)
        if pts.ndim <= 1:
            return jnp.isclose(self.adf_orig(pts), 0.0, atol=1e-6)
        return jnp.isclose(self.adf_orig(pts), 0.0, atol=1e-6)


def Geometry2DFromPointCloud(
    points: ArrayLike,
    *,
    recenter: bool = True,
    alpha: float | None = None,
    tol: float | None = None,
    offset: float | None = None,
    bound: float | str | None = None,
    progress_bar: bool = False,
) -> Geometry2DFromCAD:
    r"""Reconstruct a 2D mesh geometry from a boundary point cloud.

    Interprets `points` as samples from (or near) the boundary $\partial\Omega$ of an
    unknown planar region $\Omega\subset\mathbb{R}^2$. A triangulation is constructed
    in the plane and used to build a `Geometry2DFromCAD` instance.

    **Arguments:**

    - `points`: Array-like of shape `(N, 2)` (or `(N, >=2)`, where only the first two
      coordinates are used).
    - `recenter`: Whether to recenter the reconstructed mesh coordinates.
    - `alpha`: Optional PyVista `delaunay_2d` control (passed through when supported).
    - `tol`: Optional PyVista `delaunay_2d` control (passed through when supported).
    - `offset`: Optional PyVista `delaunay_2d` control (passed through when supported).
    - `bound`: Optional PyVista `delaunay_2d` control (passed through when supported).
    - `progress_bar`: Optional PyVista `delaunay_2d` control (passed through when supported).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must have shape (N, 2)")
    if pts.shape[1] > 2:
        pts = pts[:, :2]

    # Embed in Z=0 plane for surface export
    pts3 = np.column_stack([pts, np.zeros((pts.shape[0],), dtype=float)])
    cloud = pv.PolyData(pts3)
    # Build kwargs from provided options, filtering to those accepted by this PyVista version
    tol_value = 1e-5 if tol is None else float(tol)
    alpha_value = 0.0 if alpha is None else float(alpha)
    offset_value = 1.0 if offset is None else float(offset)
    bound_value = False if bound is None else bool(bound)

    surface = cloud.delaunay_2d(
        tol=tol_value,
        alpha=alpha_value,
        offset=offset_value,
        bound=bound_value,
        progress_bar=bool(progress_bar),
    )
    if hasattr(surface, "triangulate"):
        surface = surface.triangulate()

    tmp_path = Path(f"/tmp/pv_pointcloud2d_{uuid4().hex}.stl").resolve()
    surface.save(tmp_path)

    geom = Geometry2DFromCAD(tmp_path, recenter=recenter)
    tmp_path.unlink(missing_ok=True)
    return geom
