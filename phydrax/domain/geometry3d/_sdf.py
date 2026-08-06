#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from typing import no_type_check

import jax
import jax.numpy as jnp
import numpy as np

from ..._bvh import beam_select_leaf_items, build_packed_bvh


class _MeshSDFImpl:
    @no_type_check
    def _make_mesh_sdf(
        self,
        *,
        eps: float | None = None,
        bvh=None,
        beam_width: int = 16,
        leaf_size: int = 16,
        dtype: jnp.dtype | None = None,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Build a fast, differentiable signed-distance function (SDF) for a watertight triangle mesh.

        Inputs
        ------
        V : (nV, 3) float array
            Vertex positions.
        F : (nF, 3) int array
            Triangle indices into V.

        Returns
        -------
        sdf(p) -> scalar
            Signed distance at query point p (shape (3,)).
            Convention: negative inside, positive outside.

        Notes
        -----
        - Distance is exact point-to-triangle distance (Ericson/RTCD region tests), reduced via min.
        - Queries are accelerated with a static AABB BVH built once at SDF construction time.
        - Sign is determined via a local pseudonormal at the closest point (face/edge/vertex),
          assuming consistently oriented normals; a global flip is corrected via signed mesh volume.
        - A custom JVP is used so `jax.grad` returns a stable geometric normal almost everywhere.
        """
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}.")
        if leaf_size <= 0:
            raise ValueError(f"leaf_size must be positive, got {leaf_size}.")

        V_in = jnp.asarray(self.mesh_vertices)
        if dtype is None:
            dtype = V_in.dtype
        if bvh is None:
            bounds_min = np.min(np.asarray(self.mesh_vertices, dtype=float), axis=0)
            bounds_max = np.max(np.asarray(self.mesh_vertices, dtype=float), axis=0)
            origin_value = 0.5 * (bounds_min + bounds_max)
            distance_scale_value = float(np.linalg.norm(bounds_max - bounds_min))
            if not np.isfinite(distance_scale_value) or distance_scale_value <= 0.0:
                raise ValueError("Mesh diameter must be finite and positive.")
        else:
            # A caller-supplied BVH uses the caller's coordinate system.
            origin_value = np.zeros((3,), dtype=float)
            distance_scale_value = 1.0
        world_origin = jnp.asarray(origin_value, dtype=dtype)
        distance_scale = jnp.asarray(distance_scale_value, dtype=dtype)
        V = (jnp.asarray(V_in, dtype=dtype) - world_origin) / distance_scale
        F = jnp.asarray(self.mesh_faces)

        if V.ndim != 2 or V.shape[-1] != 3:
            raise ValueError(f"V must have shape (nV, 3). Got {V.shape}.")
        if F.ndim != 2 or F.shape[-1] != 3:
            raise ValueError(f"F must have shape (nF, 3). Got {F.shape}.")
        if int(F.shape[0]) == 0:
            raise ValueError("Mesh has no faces; cannot build an SDF.")

        # Pick a small epsilon that prevents division-by-zero without biasing distances.
        if eps is None:
            if dtype == jnp.float16:
                eps = 1e-4
            elif dtype == jnp.float32:
                eps = 1e-9
            else:
                eps = 1e-15

        eps = jnp.array(eps, dtype=dtype)
        zero = jnp.array(0.0, dtype=dtype)
        one = jnp.array(1.0, dtype=dtype)
        inf = jnp.array(jnp.inf, dtype=dtype)

        # Precompute per-triangle vertex positions and edges (constant wrt query point).
        a = V[F[:, 0]]
        ab = V[F[:, 1]] - a
        ac = V[F[:, 2]] - a

        def _dot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x * y, axis=-1)

        # ---------------------------------------------------------------------
        # Pseudonormal sign precomputation (host-side, static)
        # ---------------------------------------------------------------------
        V_np = np.asarray(V, dtype=float)
        F_np = np.asarray(self.mesh_faces, dtype=np.int64)
        a_np = V_np[F_np[:, 0]]
        b_np = V_np[F_np[:, 1]]
        c_np = V_np[F_np[:, 2]]

        face_n_unnorm_np = np.cross(b_np - a_np, c_np - a_np)
        face_n_norm_np = np.linalg.norm(face_n_unnorm_np, axis=1, keepdims=True)
        face_n_unit_np = face_n_unnorm_np / np.maximum(face_n_norm_np, 1e-30)

        # Ensure outward orientation: if the signed volume is negative, flip all normals.
        signed_vol6 = np.sum(np.einsum("ij,ij->i", a_np, np.cross(b_np, c_np)))
        if signed_vol6 < 0.0:
            face_n_unit_np = -face_n_unit_np

        def _corner_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
            denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            cos = np.sum(v1 * v2, axis=1) / np.maximum(denom, 1e-30)
            cos = np.clip(cos, -1.0, 1.0)
            return np.arccos(cos)

        ab_np = b_np - a_np
        ac_np = c_np - a_np
        ba_np = a_np - b_np
        bc_np = c_np - b_np
        ca_np = a_np - c_np
        cb_np = b_np - c_np

        ang_a = _corner_angle(ab_np, ac_np)
        ang_b = _corner_angle(ba_np, bc_np)
        ang_c = _corner_angle(ca_np, cb_np)

        n_vert_np = np.zeros_like(V_np, dtype=float)
        np.add.at(n_vert_np, F_np[:, 0], face_n_unit_np * ang_a[:, None])
        np.add.at(n_vert_np, F_np[:, 1], face_n_unit_np * ang_b[:, None])
        np.add.at(n_vert_np, F_np[:, 2], face_n_unit_np * ang_c[:, None])
        n_vert_np = n_vert_np / np.maximum(
            np.linalg.norm(n_vert_np, axis=1, keepdims=True), 1e-30
        )

        nV = int(V_np.shape[0])
        nF = int(F_np.shape[0])
        edges = np.vstack([F_np[:, [0, 1]], F_np[:, [1, 2]], F_np[:, [2, 0]]])
        edges_sorted = np.sort(edges, axis=1)
        edge_key = edges_sorted[:, 0].astype(np.int64) * np.int64(nV) + edges_sorted[
            :, 1
        ].astype(np.int64)

        order = np.argsort(edge_key)
        edge_key_sorted = edge_key[order]
        occ = np.arange(3 * nF, dtype=np.int64)
        face_occ_sorted = (occ % np.int64(nF))[order]

        normals_sorted = face_n_unit_np[face_occ_sorted]
        unique_keys, start = np.unique(edge_key_sorted, return_index=True)
        edge_sum = np.add.reduceat(normals_sorted, start, axis=0)
        edge_sum = edge_sum / np.maximum(
            np.linalg.norm(edge_sum, axis=1, keepdims=True), 1e-30
        )

        group_idx_sorted = np.searchsorted(unique_keys, edge_key_sorted)
        edge_norm_occ_sorted = edge_sum[group_idx_sorted]
        edge_norm_occ = np.empty_like(edge_norm_occ_sorted)
        edge_norm_occ[order] = edge_norm_occ_sorted

        n_edge_np = np.stack(
            [
                edge_norm_occ[0:nF],  # AB
                edge_norm_occ[nF : (2 * nF)],  # BC
                edge_norm_occ[(2 * nF) : (3 * nF)],  # CA
            ],
            axis=1,
        )  # (nF, 3, 3)

        n_face = jnp.asarray(face_n_unit_np, dtype=dtype)
        n_vert = jnp.asarray(n_vert_np, dtype=dtype)
        n_edge = jnp.asarray(n_edge_np, dtype=dtype)

        # Per-triangle vertex/edge normals (constant wrt query point)
        n_va = n_vert[F[:, 0]]
        n_vb = n_vert[F[:, 1]]
        n_vc = n_vert[F[:, 2]]
        n_eab = n_edge[:, 0]
        n_ebc = n_edge[:, 1]
        n_eca = n_edge[:, 2]

        # ---------------------------------------------------------------------
        # Static BVH (AABB tree) build (host-side, static)
        # ---------------------------------------------------------------------
        if bvh is None:
            tri_bbox_min = np.minimum(np.minimum(a_np, b_np), c_np)
            tri_bbox_max = np.maximum(np.maximum(a_np, b_np), c_np)
            tri_centers = 0.5 * (tri_bbox_min + tri_bbox_max)
            bvh = build_packed_bvh(
                tri_bbox_min,
                tri_bbox_max,
                tri_centers,
                leaf_size=leaf_size,
                dtype=dtype,
            )
        beam_steps = int(bvh.max_depth + 1)

        def _point_triangle_dist2_a_ab_ac(
            p: jnp.ndarray,
            a_t: jnp.ndarray,
            ab_t: jnp.ndarray,
            ac_t: jnp.ndarray,
        ) -> jnp.ndarray:
            u = a_t - p
            v = u + ab_t
            w = u + ac_t
            bc_t = ac_t - ab_t

            d1 = -_dot(ab_t, u)
            d2 = -_dot(ac_t, u)
            d3 = -_dot(ab_t, v)
            d4 = -_dot(ac_t, v)
            d5 = -_dot(ab_t, w)
            d6 = -_dot(ac_t, w)

            cond_a = (d1 <= zero) & (d2 <= zero)
            cond_b = (d3 >= zero) & (d4 <= d3)
            cond_c = (d6 >= zero) & (d5 <= d6)

            dist2_a = _dot(u, u)
            dist2_b = _dot(v, v)
            dist2_c = _dot(w, w)

            vc = d1 * d4 - d3 * d2
            cond_ab = (vc <= zero) & (d1 >= zero) & (d3 <= zero)
            t_ab = d1 / jnp.maximum(d1 - d3, eps)
            diff_ab = u + t_ab[..., None] * ab_t
            dist2_ab = _dot(diff_ab, diff_ab)

            vb = d5 * d2 - d1 * d6
            cond_ac = (vb <= zero) & (d2 >= zero) & (d6 <= zero)
            t_ac = d2 / jnp.maximum(d2 - d6, eps)
            diff_ac = u + t_ac[..., None] * ac_t
            dist2_ac = _dot(diff_ac, diff_ac)

            va = d3 * d6 - d5 * d4
            cond_bc = (va <= zero) & ((d4 - d3) >= zero) & ((d5 - d6) >= zero)
            t_bc = (d4 - d3) / jnp.maximum((d4 - d3) + (d5 - d6), eps)
            diff_bc = v + t_bc[..., None] * bc_t
            dist2_bc = _dot(diff_bc, diff_bc)

            denom = va + vb + vc
            v_face = vb / jnp.maximum(denom, eps)
            w_face = vc / jnp.maximum(denom, eps)
            diff_face = u + v_face[..., None] * ab_t + w_face[..., None] * ac_t
            dist2_face = _dot(diff_face, diff_face)

            dist2 = dist2_face
            dist2 = jnp.where(cond_bc, dist2_bc, dist2)
            dist2 = jnp.where(cond_ac, dist2_ac, dist2)
            dist2 = jnp.where(cond_c, dist2_c, dist2)
            dist2 = jnp.where(cond_ab, dist2_ab, dist2)
            dist2 = jnp.where(cond_b, dist2_b, dist2)
            dist2 = jnp.where(cond_a, dist2_a, dist2)
            return dist2

        def _point_triangle_diff_side_and_pseudonormal_a_ab_ac(
            p: jnp.ndarray,
            a_t: jnp.ndarray,
            ab_t: jnp.ndarray,
            ac_t: jnp.ndarray,
            n_face_t: jnp.ndarray,
            n_va_t: jnp.ndarray,
            n_vb_t: jnp.ndarray,
            n_vc_t: jnp.ndarray,
            n_eab_t: jnp.ndarray,
            n_ebc_t: jnp.ndarray,
            n_eca_t: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            u = a_t - p
            v = u + ab_t
            w = u + ac_t
            bc_t = ac_t - ab_t

            d1 = -_dot(ab_t, u)
            d2 = -_dot(ac_t, u)
            d3 = -_dot(ab_t, v)
            d4 = -_dot(ac_t, v)
            d5 = -_dot(ab_t, w)
            d6 = -_dot(ac_t, w)

            cond_a = (d1 <= zero) & (d2 <= zero)
            cond_b = (d3 >= zero) & (d4 <= d3)
            cond_c = (d6 >= zero) & (d5 <= d6)

            vc = d1 * d4 - d3 * d2
            cond_ab = (vc <= zero) & (d1 >= zero) & (d3 <= zero)
            t_ab = d1 / jnp.maximum(d1 - d3, eps)
            diff_ab = u + t_ab[..., None] * ab_t

            vb = d5 * d2 - d1 * d6
            cond_ac = (vb <= zero) & (d2 >= zero) & (d6 <= zero)
            t_ac = d2 / jnp.maximum(d2 - d6, eps)
            diff_ac = u + t_ac[..., None] * ac_t

            va = d3 * d6 - d5 * d4
            cond_bc = (va <= zero) & ((d4 - d3) >= zero) & ((d5 - d6) >= zero)
            t_bc = (d4 - d3) / jnp.maximum((d4 - d3) + (d5 - d6), eps)
            diff_bc = v + t_bc[..., None] * bc_t

            denom = va + vb + vc
            v_face = vb / jnp.maximum(denom, eps)
            w_face = vc / jnp.maximum(denom, eps)
            diff_face = u + v_face[..., None] * ab_t + w_face[..., None] * ac_t

            diff = diff_face
            diff = jnp.where(cond_bc[..., None], diff_bc, diff)
            diff = jnp.where(cond_ac[..., None], diff_ac, diff)
            diff = jnp.where(cond_c[..., None], w, diff)
            diff = jnp.where(cond_ab[..., None], diff_ab, diff)
            diff = jnp.where(cond_b[..., None], v, diff)
            diff = jnp.where(cond_a[..., None], u, diff)

            n_pseudo = n_face_t
            n_pseudo = jnp.where(cond_bc[..., None], n_ebc_t, n_pseudo)
            n_pseudo = jnp.where(cond_ac[..., None], n_eca_t, n_pseudo)
            n_pseudo = jnp.where(cond_c[..., None], n_vc_t, n_pseudo)
            n_pseudo = jnp.where(cond_ab[..., None], n_eab_t, n_pseudo)
            n_pseudo = jnp.where(cond_b[..., None], n_vb_t, n_pseudo)
            n_pseudo = jnp.where(cond_a[..., None], n_va_t, n_pseudo)

            side = -_dot(diff, n_pseudo)
            return diff, side, n_pseudo

        def _coerce_points3(
            points: jnp.ndarray,
        ) -> tuple[jnp.ndarray, tuple[int, ...], bool]:
            pts = jnp.asarray(points, dtype=dtype)
            if pts.ndim == 0:
                raise ValueError("Expected points with shape (..., 3) or (3,).")
            if pts.ndim == 1:
                if pts.shape[0] != 3:
                    raise ValueError(f"Expected points with shape (3,), got {pts.shape}.")
                return pts.reshape((1, 3)), (), True
            if pts.shape[-1] != 3:
                raise ValueError(f"Expected points with shape (..., 3), got {pts.shape}.")
            out_shape = pts.shape[:-1]
            return pts.reshape((-1, 3)), out_shape, False

        def _sdf_primal_and_geom(
            points: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """Return (signed, diff, side, unsigned, n_pseudo) for points."""
            pts_world, out_shape, is_single = _coerce_points3(points)
            pts = (pts_world - world_origin) / distance_scale
            tri, valid = beam_select_leaf_items(
                pts, bvh=bvh, beam_width=beam_width, steps=beam_steps
            )
            tri = jnp.asarray(tri, dtype=jnp.int32)
            safe_tri = jnp.where(valid, tri, jnp.int32(0))

            a_t = a[safe_tri]
            ab_t = ab[safe_tri]
            ac_t = ac[safe_tri]

            dist2_all = _point_triangle_dist2_a_ab_ac(pts[:, None, :], a_t, ab_t, ac_t)
            dist2_all = jnp.where(valid, dist2_all, inf)
            best_idx = jnp.argmin(dist2_all, axis=1)
            dist2_min = jnp.take_along_axis(dist2_all, best_idx[:, None], axis=1).reshape(
                (-1,)
            )

            best_tri = jnp.take_along_axis(tri, best_idx[:, None], axis=1).reshape((-1,))
            best_tri = jnp.where(best_tri >= 0, best_tri, jnp.int32(0))

            diff, side, n_pseudo = _point_triangle_diff_side_and_pseudonormal_a_ab_ac(
                pts,
                a[best_tri],
                ab[best_tri],
                ac[best_tri],
                n_face[best_tri],
                n_va[best_tri],
                n_vb[best_tri],
                n_vc[best_tri],
                n_eab[best_tri],
                n_ebc[best_tri],
                n_eca[best_tri],
            )

            unsigned = jnp.sqrt(jnp.maximum(dist2_min, zero))
            sgn = jnp.where(side < zero, -one, one)
            signed = sgn * unsigned * distance_scale

            if is_single:
                return (
                    signed.reshape(()),
                    diff.reshape((3,)),
                    side.reshape(()),
                    unsigned.reshape(()),
                    n_pseudo.reshape((3,)),
                )
            return (
                signed.reshape(out_shape),
                diff.reshape(out_shape + (3,)),
                side.reshape(out_shape),
                unsigned.reshape(out_shape),
                n_pseudo.reshape(out_shape + (3,)),
            )

        @jax.custom_jvp
        def sdf(points: jnp.ndarray) -> jnp.ndarray:
            val, _, _, _, _ = _sdf_primal_and_geom(points)
            return val

        @sdf.defjvp
        def sdf_jvp(primals, tangents):
            (points,) = primals
            (t_points,) = tangents

            # Nearest-feature routing is discrete, and a polyhedral distance has no
            # classical higher derivative at edge/vertex seams. Freeze that geometry
            # under nested AD; the smooth envelope supplies derivatives away from
            # the exact collar, while this exact field retains its unit first jet.
            points_stopped = jax.lax.stop_gradient(points)
            val, diff, side, unsigned, n_pseudo = _sdf_primal_and_geom(points_stopped)
            boundary_collar = jnp.sqrt(eps)
            safe_unsigned = jnp.maximum(unsigned, boundary_collar)
            sgn = jnp.where(side < zero, -one, one)
            raw = sgn[..., None] * (-diff) / safe_unsigned[..., None]
            raw_norm_sq = _dot(raw, raw)
            safe_raw_norm = jnp.sqrt(
                jnp.maximum(raw_norm_sq, jnp.asarray(0.25, dtype=dtype))
            )
            grad_cp = raw / safe_raw_norm[..., None]
            use_cp = unsigned > boundary_collar
            grad = jnp.where(use_cp[..., None], grad_cp, n_pseudo)

            points_array = jnp.asarray(points, dtype=dtype)
            linear_primal = _dot(points_array, grad)
            primal_val = (
                jax.lax.stop_gradient(val)
                + linear_primal
                - jax.lax.stop_gradient(linear_primal)
            )
            if t_points is None:
                return primal_val, jnp.zeros_like(val)
            return primal_val, _dot(grad, t_points)

        return jax.jit(sdf)

    @no_type_check
    def _make_mesh_sdf_traversal(
        self,
        *,
        eps: float | None = None,
        bvh=None,
        leaf_size: int = 16,
        dtype: jnp.dtype | None = None,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Build an exact (global-nearest) mesh SDF via BVH stack traversal.

        The smooth boundary factor uses this traversal for its exact boundary collar,
        avoiding candidate-routing seams on dense or highly nonuniform meshes. The
        beam-selected implementation remains available for sampling and containment
        paths that prioritize throughput.
        """
        if leaf_size <= 0:
            raise ValueError(f"leaf_size must be positive, got {leaf_size}.")

        V_in = jnp.asarray(self.mesh_vertices)
        if dtype is None:
            dtype = V_in.dtype
        if bvh is None:
            bounds_min = np.min(np.asarray(self.mesh_vertices, dtype=float), axis=0)
            bounds_max = np.max(np.asarray(self.mesh_vertices, dtype=float), axis=0)
            origin_value = 0.5 * (bounds_min + bounds_max)
            distance_scale_value = float(np.linalg.norm(bounds_max - bounds_min))
            if not np.isfinite(distance_scale_value) or distance_scale_value <= 0.0:
                raise ValueError("Mesh diameter must be finite and positive.")
        else:
            # A caller-supplied BVH uses the caller's coordinate system.
            origin_value = np.zeros((3,), dtype=float)
            distance_scale_value = 1.0
        world_origin = jnp.asarray(origin_value, dtype=dtype)
        distance_scale = jnp.asarray(distance_scale_value, dtype=dtype)
        V = (jnp.asarray(V_in, dtype=dtype) - world_origin) / distance_scale
        F = jnp.asarray(self.mesh_faces)

        if V.ndim != 2 or V.shape[-1] != 3:
            raise ValueError(f"V must have shape (nV, 3). Got {V.shape}.")
        if F.ndim != 2 or F.shape[-1] != 3:
            raise ValueError(f"F must have shape (nF, 3). Got {F.shape}.")
        if int(F.shape[0]) == 0:
            raise ValueError("Mesh has no faces; cannot build an SDF.")

        if eps is None:
            if dtype == jnp.float16:
                eps = 1e-4
            elif dtype == jnp.float32:
                eps = 1e-9
            else:
                eps = 1e-15

        eps = jnp.asarray(eps, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        inf = jnp.asarray(jnp.inf, dtype=dtype)

        a = V[F[:, 0]]
        ab = V[F[:, 1]] - a
        ac = V[F[:, 2]] - a

        def _dot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x * y, axis=-1)

        # ---------------------------------------------------------------------
        # Pseudonormal sign precomputation (host-side, static)
        # ---------------------------------------------------------------------
        V_np = np.asarray(V, dtype=float)
        F_np = np.asarray(self.mesh_faces, dtype=np.int64)
        a_np = V_np[F_np[:, 0]]
        b_np = V_np[F_np[:, 1]]
        c_np = V_np[F_np[:, 2]]

        face_n_unnorm_np = np.cross(b_np - a_np, c_np - a_np)
        face_n_norm_np = np.linalg.norm(face_n_unnorm_np, axis=1, keepdims=True)
        face_n_unit_np = face_n_unnorm_np / np.maximum(face_n_norm_np, 1e-30)

        signed_vol6 = np.sum(np.einsum("ij,ij->i", a_np, np.cross(b_np, c_np)))
        if signed_vol6 < 0.0:
            face_n_unit_np = -face_n_unit_np

        def _corner_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
            denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            cos = np.sum(v1 * v2, axis=1) / np.maximum(denom, 1e-30)
            cos = np.clip(cos, -1.0, 1.0)
            return np.arccos(cos)

        ab_np = b_np - a_np
        ac_np = c_np - a_np
        ba_np = a_np - b_np
        bc_np = c_np - b_np
        ca_np = a_np - c_np
        cb_np = b_np - c_np

        ang_a = _corner_angle(ab_np, ac_np)
        ang_b = _corner_angle(ba_np, bc_np)
        ang_c = _corner_angle(ca_np, cb_np)

        n_vert_np = np.zeros_like(V_np, dtype=float)
        np.add.at(n_vert_np, F_np[:, 0], face_n_unit_np * ang_a[:, None])
        np.add.at(n_vert_np, F_np[:, 1], face_n_unit_np * ang_b[:, None])
        np.add.at(n_vert_np, F_np[:, 2], face_n_unit_np * ang_c[:, None])
        n_vert_np = n_vert_np / np.maximum(
            np.linalg.norm(n_vert_np, axis=1, keepdims=True), 1e-30
        )

        nV = int(V_np.shape[0])
        nF = int(F_np.shape[0])
        edges = np.vstack([F_np[:, [0, 1]], F_np[:, [1, 2]], F_np[:, [2, 0]]])
        edges_sorted = np.sort(edges, axis=1)
        edge_key = edges_sorted[:, 0].astype(np.int64) * np.int64(nV) + edges_sorted[
            :, 1
        ].astype(np.int64)

        order = np.argsort(edge_key)
        edge_key_sorted = edge_key[order]
        occ = np.arange(3 * nF, dtype=np.int64)
        face_occ_sorted = (occ % np.int64(nF))[order]

        normals_sorted = face_n_unit_np[face_occ_sorted]
        unique_keys, start = np.unique(edge_key_sorted, return_index=True)
        edge_sum = np.add.reduceat(normals_sorted, start, axis=0)
        edge_sum = edge_sum / np.maximum(
            np.linalg.norm(edge_sum, axis=1, keepdims=True), 1e-30
        )

        group_idx_sorted = np.searchsorted(unique_keys, edge_key_sorted)
        edge_norm_occ_sorted = edge_sum[group_idx_sorted]
        edge_norm_occ = np.empty_like(edge_norm_occ_sorted)
        edge_norm_occ[order] = edge_norm_occ_sorted

        n_edge_np = np.stack(
            [
                edge_norm_occ[0:nF],  # AB
                edge_norm_occ[nF : (2 * nF)],  # BC
                edge_norm_occ[(2 * nF) : (3 * nF)],  # CA
            ],
            axis=1,
        )  # (nF, 3, 3)

        n_face = jnp.asarray(face_n_unit_np, dtype=dtype)
        n_vert = jnp.asarray(n_vert_np, dtype=dtype)
        n_edge = jnp.asarray(n_edge_np, dtype=dtype)

        n_va = n_vert[F[:, 0]]
        n_vb = n_vert[F[:, 1]]
        n_vc = n_vert[F[:, 2]]
        n_eab = n_edge[:, 0]
        n_ebc = n_edge[:, 1]
        n_eca = n_edge[:, 2]

        # ---------------------------------------------------------------------
        # Static BVH (AABB tree) build (host-side, static)
        # ---------------------------------------------------------------------
        if bvh is None:
            tri_bbox_min = np.minimum(np.minimum(a_np, b_np), c_np)
            tri_bbox_max = np.maximum(np.maximum(a_np, b_np), c_np)
            tri_centers = 0.5 * (tri_bbox_min + tri_bbox_max)
            bvh = build_packed_bvh(
                tri_bbox_min,
                tri_bbox_max,
                tri_centers,
                leaf_size=leaf_size,
                dtype=dtype,
            )

        bbox_min = bvh.bbox_min
        bbox_max = bvh.bbox_max
        left = bvh.left
        right = bvh.right
        leaf_id = bvh.leaf_id
        leaf_items = bvh.leaf_items
        max_stack = int(bvh.max_depth + 4)

        def _aabb_dist2(
            p: jnp.ndarray, bmin: jnp.ndarray, bmax: jnp.ndarray
        ) -> jnp.ndarray:
            d = jnp.maximum(zero, jnp.maximum(bmin - p, p - bmax))
            return _dot(d, d)

        def _point_triangle_dist2_a_ab_ac(
            p: jnp.ndarray,
            a_t: jnp.ndarray,
            ab_t: jnp.ndarray,
            ac_t: jnp.ndarray,
        ) -> jnp.ndarray:
            u = a_t - p
            v = u + ab_t
            w = u + ac_t
            bc_t = ac_t - ab_t

            d1 = -_dot(ab_t, u)
            d2 = -_dot(ac_t, u)
            d3 = -_dot(ab_t, v)
            d4 = -_dot(ac_t, v)
            d5 = -_dot(ab_t, w)
            d6 = -_dot(ac_t, w)

            cond_a = (d1 <= zero) & (d2 <= zero)
            cond_b = (d3 >= zero) & (d4 <= d3)
            cond_c = (d6 >= zero) & (d5 <= d6)

            dist2_a = _dot(u, u)
            dist2_b = _dot(v, v)
            dist2_c = _dot(w, w)

            vc = d1 * d4 - d3 * d2
            cond_ab = (vc <= zero) & (d1 >= zero) & (d3 <= zero)
            t_ab = d1 / jnp.maximum(d1 - d3, eps)
            diff_ab = u + t_ab[..., None] * ab_t
            dist2_ab = _dot(diff_ab, diff_ab)

            vb = d5 * d2 - d1 * d6
            cond_ac = (vb <= zero) & (d2 >= zero) & (d6 <= zero)
            t_ac = d2 / jnp.maximum(d2 - d6, eps)
            diff_ac = u + t_ac[..., None] * ac_t
            dist2_ac = _dot(diff_ac, diff_ac)

            va = d3 * d6 - d5 * d4
            cond_bc = (va <= zero) & ((d4 - d3) >= zero) & ((d5 - d6) >= zero)
            t_bc = (d4 - d3) / jnp.maximum((d4 - d3) + (d5 - d6), eps)
            diff_bc = v + t_bc[..., None] * bc_t
            dist2_bc = _dot(diff_bc, diff_bc)

            denom = va + vb + vc
            v_face = vb / jnp.maximum(denom, eps)
            w_face = vc / jnp.maximum(denom, eps)
            diff_face = u + v_face[..., None] * ab_t + w_face[..., None] * ac_t
            dist2_face = _dot(diff_face, diff_face)

            dist2 = dist2_face
            dist2 = jnp.where(cond_bc, dist2_bc, dist2)
            dist2 = jnp.where(cond_ac, dist2_ac, dist2)
            dist2 = jnp.where(cond_c, dist2_c, dist2)
            dist2 = jnp.where(cond_ab, dist2_ab, dist2)
            dist2 = jnp.where(cond_b, dist2_b, dist2)
            dist2 = jnp.where(cond_a, dist2_a, dist2)
            return dist2

        def _point_triangle_diff_side_and_pseudonormal_a_ab_ac(
            p: jnp.ndarray,
            a_t: jnp.ndarray,
            ab_t: jnp.ndarray,
            ac_t: jnp.ndarray,
            n_face_t: jnp.ndarray,
            n_va_t: jnp.ndarray,
            n_vb_t: jnp.ndarray,
            n_vc_t: jnp.ndarray,
            n_eab_t: jnp.ndarray,
            n_ebc_t: jnp.ndarray,
            n_eca_t: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            u = a_t - p
            v = u + ab_t
            w = u + ac_t
            bc_t = ac_t - ab_t

            d1 = -_dot(ab_t, u)
            d2 = -_dot(ac_t, u)
            d3 = -_dot(ab_t, v)
            d4 = -_dot(ac_t, v)
            d5 = -_dot(ab_t, w)
            d6 = -_dot(ac_t, w)

            cond_a = (d1 <= zero) & (d2 <= zero)
            cond_b = (d3 >= zero) & (d4 <= d3)
            cond_c = (d6 >= zero) & (d5 <= d6)

            vc = d1 * d4 - d3 * d2
            cond_ab = (vc <= zero) & (d1 >= zero) & (d3 <= zero)
            t_ab = d1 / jnp.maximum(d1 - d3, eps)
            diff_ab = u + t_ab * ab_t

            vb = d5 * d2 - d1 * d6
            cond_ac = (vb <= zero) & (d2 >= zero) & (d6 <= zero)
            t_ac = d2 / jnp.maximum(d2 - d6, eps)
            diff_ac = u + t_ac * ac_t

            va = d3 * d6 - d5 * d4
            cond_bc = (va <= zero) & ((d4 - d3) >= zero) & ((d5 - d6) >= zero)
            t_bc = (d4 - d3) / jnp.maximum((d4 - d3) + (d5 - d6), eps)
            diff_bc = v + t_bc * bc_t

            denom = va + vb + vc
            v_face = vb / jnp.maximum(denom, eps)
            w_face = vc / jnp.maximum(denom, eps)
            diff_face = u + v_face * ab_t + w_face * ac_t

            diff = diff_face
            diff = jnp.where(cond_bc, diff_bc, diff)
            diff = jnp.where(cond_ac, diff_ac, diff)
            diff = jnp.where(cond_c, w, diff)
            diff = jnp.where(cond_ab, diff_ab, diff)
            diff = jnp.where(cond_b, v, diff)
            diff = jnp.where(cond_a, u, diff)

            n_pseudo = n_face_t
            n_pseudo = jnp.where(cond_bc, n_ebc_t, n_pseudo)
            n_pseudo = jnp.where(cond_ac, n_eca_t, n_pseudo)
            n_pseudo = jnp.where(cond_c, n_vc_t, n_pseudo)
            n_pseudo = jnp.where(cond_ab, n_eab_t, n_pseudo)
            n_pseudo = jnp.where(cond_b, n_vb_t, n_pseudo)
            n_pseudo = jnp.where(cond_a, n_va_t, n_pseudo)

            side = -_dot(diff, n_pseudo)
            return diff, side, n_pseudo

        def _coerce_points3(
            points: jnp.ndarray,
        ) -> tuple[jnp.ndarray, tuple[int, ...], bool]:
            pts = jnp.asarray(points, dtype=dtype)
            if pts.ndim == 0:
                raise ValueError("Expected points with shape (..., 3) or (3,).")
            if pts.ndim == 1:
                if pts.shape[0] != 3:
                    raise ValueError(f"Expected points with shape (3,), got {pts.shape}.")
                return pts.reshape((1, 3)), (), True
            if pts.shape[-1] != 3:
                raise ValueError(f"Expected points with shape (..., 3), got {pts.shape}.")
            out_shape = pts.shape[:-1]
            return pts.reshape((-1, 3)), out_shape, False

        def _traverse_best_triangle(
            p: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            stack = jnp.full((max_stack,), jnp.int32(-1))
            stack = stack.at[0].set(jnp.int32(0))
            sp0 = jnp.int32(1)
            best_d20 = inf
            best_tri0 = jnp.int32(0)

            def _push(stack, sp, node):
                stack = stack.at[sp].set(node)
                return stack, sp + jnp.int32(1)

            def _body(_, state):
                stack, sp, best_d2, best_tri = state
                active = sp > 0
                sp = jnp.maximum(sp - jnp.int32(1), jnp.int32(0))
                node = jnp.where(active, stack[sp], jnp.int32(-1))
                state = (stack, sp, best_d2, best_tri)

                def _process_node(state):
                    stack, sp, best_d2, best_tri = state
                    d2_node = _aabb_dist2(p, bbox_min[node], bbox_max[node])

                    def _visit(state):
                        stack, sp, best_d2, best_tri = state
                        lid = leaf_id[node]

                        def _leaf(state):
                            stack, sp, best_d2, best_tri = state
                            tris = leaf_items[lid]
                            valid = tris >= 0
                            has_valid = jnp.any(valid)
                            safe_tris = jnp.where(valid, tris, jnp.int32(0))

                            a_t = a[safe_tris]
                            ab_t = ab[safe_tris]
                            ac_t = ac[safe_tris]

                            dist2 = _point_triangle_dist2_a_ab_ac(p, a_t, ab_t, ac_t)
                            dist2 = jnp.where(valid, dist2, inf)
                            k = jnp.argmin(dist2)
                            d2_min = dist2[k]
                            tri_min = safe_tris[k]

                            better = has_valid & (d2_min < best_d2)
                            best_d2 = jnp.where(better, d2_min, best_d2)
                            best_tri = jnp.where(better, tri_min, best_tri)
                            return stack, sp, best_d2, best_tri

                        def _internal(state):
                            stack, sp, best_d2, best_tri = state
                            l = left[node]
                            r = right[node]
                            l_valid = l >= 0
                            r_valid = r >= 0
                            d2_l = jnp.where(
                                l_valid, _aabb_dist2(p, bbox_min[l], bbox_max[l]), inf
                            )
                            d2_r = jnp.where(
                                r_valid, _aabb_dist2(p, bbox_min[r], bbox_max[r]), inf
                            )

                            swap = d2_l < d2_r
                            near = jnp.where(swap, l, r)
                            far = jnp.where(swap, r, l)
                            d2_near = jnp.where(swap, d2_l, d2_r)
                            d2_far = jnp.where(swap, d2_r, d2_l)

                            def _push_node(state, node_to_push):
                                stack, sp, best_d2, best_tri = state
                                stack, sp = _push(stack, sp, node_to_push)
                                return stack, sp, best_d2, best_tri

                            push_far = (
                                (d2_far < best_d2)
                                & (far >= 0)
                                & (sp < jnp.int32(max_stack))
                            )
                            state = jax.lax.cond(
                                push_far,
                                lambda s: _push_node(s, far),
                                lambda s: s,
                                (stack, sp, best_d2, best_tri),
                            )

                            stack, sp, best_d2, best_tri = state
                            push_near = (
                                (d2_near < best_d2)
                                & (near >= 0)
                                & (sp < jnp.int32(max_stack))
                            )
                            state = jax.lax.cond(
                                push_near,
                                lambda s: _push_node(s, near),
                                lambda s: s,
                                (stack, sp, best_d2, best_tri),
                            )
                            return state

                        return jax.lax.cond(lid >= 0, _leaf, _internal, state)

                    return jax.lax.cond(d2_node < best_d2, _visit, lambda s: s, state)

                return jax.lax.cond(node >= 0, _process_node, lambda s: s, state)

            stack, sp, best_d2, best_tri = jax.lax.fori_loop(
                0,
                int(bbox_min.shape[0]),
                _body,
                (stack, sp0, best_d20, best_tri0),
            )
            return best_d2, best_tri

        def _sdf_primal_and_geom(
            points: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            pts_world, out_shape, is_single = _coerce_points3(points)
            pts = (pts_world - world_origin) / distance_scale
            dist2_min, best_tri = jax.vmap(_traverse_best_triangle)(pts)
            unsigned = jnp.sqrt(jnp.maximum(dist2_min, zero))

            def _geom_for_best(p, tri):
                tri = jnp.asarray(tri, dtype=jnp.int32)
                tri = jnp.where(tri >= 0, tri, jnp.int32(0))
                return _point_triangle_diff_side_and_pseudonormal_a_ab_ac(
                    p,
                    a[tri],
                    ab[tri],
                    ac[tri],
                    n_face[tri],
                    n_va[tri],
                    n_vb[tri],
                    n_vc[tri],
                    n_eab[tri],
                    n_ebc[tri],
                    n_eca[tri],
                )

            diff, side, n_pseudo = jax.vmap(_geom_for_best)(pts, best_tri)
            sgn = jnp.where(side < zero, -one, one)
            signed = sgn * unsigned * distance_scale

            if is_single:
                return (
                    signed.reshape(()),
                    diff.reshape((3,)),
                    side.reshape(()),
                    unsigned.reshape(()),
                    n_pseudo.reshape((3,)),
                )
            return (
                signed.reshape(out_shape),
                diff.reshape(out_shape + (3,)),
                side.reshape(out_shape),
                unsigned.reshape(out_shape),
                n_pseudo.reshape(out_shape + (3,)),
            )

        @jax.custom_jvp
        def sdf(points: jnp.ndarray) -> jnp.ndarray:
            val, _, _, _, _ = _sdf_primal_and_geom(points)
            return val

        @sdf.defjvp
        def sdf_jvp(primals, tangents):
            (points,) = primals
            (t_points,) = tangents

            # See the beam-search implementation above: freeze the selected mesh
            # feature under higher AD and expose its stable unit first derivative.
            points_stopped = jax.lax.stop_gradient(points)
            val, diff, side, unsigned, n_pseudo = _sdf_primal_and_geom(points_stopped)
            boundary_collar = jnp.sqrt(eps)
            safe_unsigned = jnp.maximum(unsigned, boundary_collar)
            sgn = jnp.where(side < zero, -one, one)
            raw = sgn[..., None] * (-diff) / safe_unsigned[..., None]
            raw_norm_sq = _dot(raw, raw)
            safe_raw_norm = jnp.sqrt(
                jnp.maximum(raw_norm_sq, jnp.asarray(0.25, dtype=dtype))
            )
            grad_cp = raw / safe_raw_norm[..., None]
            use_cp = unsigned > boundary_collar
            grad = jnp.where(use_cp[..., None], grad_cp, n_pseudo)

            points_array = jnp.asarray(points, dtype=dtype)
            linear_primal = _dot(points_array, grad)
            primal_val = (
                jax.lax.stop_gradient(val)
                + linear_primal
                - jax.lax.stop_gradient(linear_primal)
            )
            if t_points is None:
                return primal_val, jnp.zeros_like(val)
            return primal_val, _dot(grad, t_points)

        return jax.jit(sdf)

    @no_type_check
    def _make_smooth_mesh_sdf(
        self,
        *,
        eps: float | None = None,
        beta: float | None = None,
        beam_width: int = 32,
        beam_steps: int | None = None,
        r0: float | None = None,
        rho_delta: float | None = None,
        equivalence_power: float | None = None,
        squash: bool = True,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Build a pointwise smoothed mesh SDF with exact boundary behaviour.

        This returns a signed distance-like function `phi(p)` that:
        - Equals the exact mesh SDF throughout an open boundary collar, with a stable
          pseudonormal custom derivative at the zero set.
        - Replaces the non-smooth min over triangles by a smooth aggregate over a
          BVH-selected leaf beam away from that collar.

        Notes
        -----
        - Candidate triangle selection is treated as non-differentiable routing via
          `stop_gradient`, so JAX higher derivatives reflect the smooth distance
          aggregate within the selected candidate set.
        - ``equivalence_power`` selects an inverse-power R-equivalence aggregate;
          ``None`` retains the Gibbs-weighted RMS aggregate.
        """
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}.")
        if equivalence_power is not None:
            equivalence_power = float(equivalence_power)
            if not np.isfinite(equivalence_power) or equivalence_power <= 0.0:
                raise ValueError(
                    "equivalence_power must be finite and positive, "
                    f"got {equivalence_power}."
                )

        V_world = jnp.asarray(self.mesh_vertices)
        dtype = V_world.dtype
        bounds_min = np.min(np.asarray(self.mesh_vertices, dtype=float), axis=0)
        bounds_max = np.max(np.asarray(self.mesh_vertices, dtype=float), axis=0)
        origin_value = 0.5 * (bounds_min + bounds_max)
        distance_scale_value = float(np.linalg.norm(bounds_max - bounds_min))
        if not np.isfinite(distance_scale_value) or distance_scale_value <= 0.0:
            raise ValueError("Mesh diameter must be finite and positive.")
        world_origin = jnp.asarray(origin_value, dtype=dtype)
        distance_scale = jnp.asarray(distance_scale_value, dtype=dtype)
        V = (V_world - world_origin) / distance_scale
        F = jnp.asarray(self.mesh_faces)

        if V.ndim != 2 or V.shape[-1] != 3:
            raise ValueError(f"V must have shape (nV, 3). Got {V.shape}.")
        if F.ndim != 2 or F.shape[-1] != 3:
            raise ValueError(f"F must have shape (nF, 3). Got {F.shape}.")
        if int(F.shape[0]) == 0:
            raise ValueError("Mesh has no faces; cannot build an SDF.")

        if eps is None:
            if dtype == jnp.float16:
                eps = 1e-4
            elif dtype == jnp.float32:
                eps = 1e-9
            else:
                eps = 1e-15
        eps = jnp.asarray(eps, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        inf = jnp.asarray(jnp.inf, dtype=dtype)

        # Work in normalized coordinates so all regularization is scale-covariant.
        if beta is None:
            beta = 1.0 / (0.075**2 + 1e-30)
        else:
            beta = float(beta) * distance_scale_value**2
        beta = jnp.asarray(beta, dtype=dtype)
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}.")
        if r0 is None:
            r0 = 0.05
        else:
            r0 = float(r0) / distance_scale_value
        r0 = jnp.asarray(r0, dtype=dtype)

        # Boundary-linear, interior-saturating transform:
        # rho(s) = delta * tanh(s / delta), with rho'(0) = 1 and rho''(0) = 0.
        # This helps suppress large interior curvature (e.g. medial-axis artifacts)
        # from entering higher-order PDE residuals when `sdf` is used as an envelope.
        squash_flag = bool(squash)
        if squash_flag:
            if rho_delta is None:
                rho_delta = 0.0111
            else:
                rho_delta = float(rho_delta) / distance_scale_value
            rho_delta = jnp.asarray(rho_delta, dtype=dtype)
            rho_delta = jnp.maximum(rho_delta, jnp.asarray(1e-12, dtype=dtype))

        # Precompute per-triangle vertex positions and edges.
        a = V[F[:, 0]]
        ab = V[F[:, 1]] - a
        ac = V[F[:, 2]] - a

        def _dot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x * y, axis=-1)

        # ---------------------------------------------------------------------
        # Static BVH (AABB tree) build (host-side)
        # ---------------------------------------------------------------------
        V_np = np.asarray(V, dtype=float)
        F_np = np.asarray(self.mesh.faces, dtype=np.int64)
        a_np = V_np[F_np[:, 0]]
        b_np = V_np[F_np[:, 1]]
        c_np = V_np[F_np[:, 2]]

        leaf_size = 16
        tri_bbox_min = np.minimum(np.minimum(a_np, b_np), c_np)
        tri_bbox_max = np.maximum(np.maximum(a_np, b_np), c_np)
        tri_centers = 0.5 * (tri_bbox_min + tri_bbox_max)
        bvh = build_packed_bvh(
            tri_bbox_min,
            tri_bbox_max,
            tri_centers,
            leaf_size=leaf_size,
            dtype=dtype,
        )

        if beam_steps is None:
            beam_steps = int(bvh.max_depth + 2)
        if beam_steps <= 0:
            raise ValueError(f"beam_steps must be positive, got {beam_steps}.")

        sdf_exact = _MeshSDFImpl._make_mesh_sdf_traversal(self, eps=eps)

        def _point_triangle_dist2_a_ab_ac(
            p: jnp.ndarray,
            a_t: jnp.ndarray,
            ab_t: jnp.ndarray,
            ac_t: jnp.ndarray,
        ) -> jnp.ndarray:
            u = a_t - p
            v = u + ab_t
            w = u + ac_t
            bc_t = ac_t - ab_t

            d1 = -_dot(ab_t, u)
            d2 = -_dot(ac_t, u)
            d3 = -_dot(ab_t, v)
            d4 = -_dot(ac_t, v)
            d5 = -_dot(ab_t, w)
            d6 = -_dot(ac_t, w)

            cond_a = (d1 <= zero) & (d2 <= zero)
            cond_b = (d3 >= zero) & (d4 <= d3)
            cond_c = (d6 >= zero) & (d5 <= d6)

            dist2_a = _dot(u, u)
            dist2_b = _dot(v, v)
            dist2_c = _dot(w, w)

            vc = d1 * d4 - d3 * d2
            cond_ab = (vc <= zero) & (d1 >= zero) & (d3 <= zero)
            t_ab = d1 / jnp.maximum(d1 - d3, eps)
            diff_ab = u + t_ab[..., None] * ab_t
            dist2_ab = _dot(diff_ab, diff_ab)

            vb = d5 * d2 - d1 * d6
            cond_ac = (vb <= zero) & (d2 >= zero) & (d6 <= zero)
            t_ac = d2 / jnp.maximum(d2 - d6, eps)
            diff_ac = u + t_ac[..., None] * ac_t
            dist2_ac = _dot(diff_ac, diff_ac)

            va = d3 * d6 - d5 * d4
            cond_bc = (va <= zero) & ((d4 - d3) >= zero) & ((d5 - d6) >= zero)
            t_bc = (d4 - d3) / jnp.maximum((d4 - d3) + (d5 - d6), eps)
            diff_bc = v + t_bc[..., None] * bc_t
            dist2_bc = _dot(diff_bc, diff_bc)

            denom = va + vb + vc
            v_face = vb / jnp.maximum(denom, eps)
            w_face = vc / jnp.maximum(denom, eps)
            diff_face = u + v_face[..., None] * ab_t + w_face[..., None] * ac_t
            dist2_face = _dot(diff_face, diff_face)

            dist2 = dist2_face
            dist2 = jnp.where(cond_bc, dist2_bc, dist2)
            dist2 = jnp.where(cond_ac, dist2_ac, dist2)
            dist2 = jnp.where(cond_c, dist2_c, dist2)
            dist2 = jnp.where(cond_ab, dist2_ab, dist2)
            dist2 = jnp.where(cond_b, dist2_b, dist2)
            dist2 = jnp.where(cond_a, dist2_a, dist2)
            return dist2

        def _coerce_points3(
            points: jnp.ndarray,
        ) -> tuple[jnp.ndarray, tuple[int, ...], bool]:
            pts = jnp.asarray(points, dtype=dtype)
            if pts.ndim == 0:
                p3 = jnp.repeat(pts, 3)
                return p3.reshape((1, 3)), (), True
            if pts.ndim == 1:
                if pts.shape[0] == 3:
                    return pts.reshape((1, 3)), (), True
                n = int(pts.shape[0])
                if n < 3:
                    buf = jnp.zeros((3,), dtype=pts.dtype)
                    buf = buf.at[:n].set(pts)
                    pts = buf
                else:
                    pts = pts[-3:]
                return pts.reshape((1, 3)), (), True
            if pts.shape[-1] != 3:
                pts = pts.reshape(-1)[-3:]
                return pts.reshape((1, 3)), (), True
            out_shape = pts.shape[:-1]
            return pts.reshape((-1, 3)), out_shape, False

        def sdf_smooth(points: jnp.ndarray) -> jnp.ndarray:
            pts_world, out_shape, is_single = _coerce_points3(points)
            pts = (pts_world - world_origin) / distance_scale

            d_exact = sdf_exact(pts_world) / distance_scale
            d_for_blend = jax.lax.stop_gradient(d_exact)

            pts_select = jax.lax.stop_gradient(pts)
            tri, valid = beam_select_leaf_items(
                pts_select, bvh=bvh, beam_width=beam_width, steps=beam_steps
            )
            tri = jnp.asarray(tri, dtype=jnp.int32)
            safe_tri = jnp.where(valid, tri, jnp.int32(0))

            a_t = a[safe_tri]
            ab_t = ab[safe_tri]
            ac_t = ac[safe_tri]

            dist2 = _point_triangle_dist2_a_ab_ac(pts[:, None, :], a_t, ab_t, ac_t)
            dist2_use = jnp.where(valid, dist2, zero)
            has_valid = jnp.any(valid, axis=1)

            if equivalence_power is None:
                scaled = jnp.where(valid, -beta * dist2, -inf)
                logZ = jax.scipy.special.logsumexp(scaled, axis=1, keepdims=True)
                weights = jnp.exp(scaled - logZ)
                weights = jnp.where(valid, weights, zero)
                m = jnp.sum(weights * dist2_use, axis=1)
                unsigned_smooth = jnp.sqrt(jnp.maximum(m, eps))
            else:
                power = jnp.asarray(equivalence_power, dtype=dtype)
                log_inverse_distance = jnp.where(
                    valid,
                    -0.5 * power * jnp.log(jnp.maximum(dist2, eps)),
                    -inf,
                )
                log_inverse_sum = jax.scipy.special.logsumexp(
                    log_inverse_distance, axis=1
                )
                unsigned_smooth = jnp.exp(-log_inverse_sum / power)
            unsigned = jnp.where(has_valid, unsigned_smooth, jnp.abs(d_exact))

            sgn = jnp.where(d_exact < zero, -one, one)
            sgn = jax.lax.stop_gradient(sgn)
            d_smooth = sgn * unsigned

            blend_start = 0.5 * r0
            blend_width = r0 - blend_start
            blend_coordinate_raw = jnp.clip(
                (jnp.abs(d_for_blend) - blend_start) / (blend_width + eps),
                zero,
                one,
            )
            blend_active = jnp.abs(d_for_blend) > blend_start
            blend_coordinate = jnp.where(blend_active, blend_coordinate_raw, one)
            S = (
                462.0 * blend_coordinate**6
                - 1980.0 * blend_coordinate**7
                + 3465.0 * blend_coordinate**8
                - 3080.0 * blend_coordinate**9
                + 1386.0 * blend_coordinate**10
                - 252.0 * blend_coordinate**11
            )
            S = jnp.where(blend_active, S, zero)

            d = d_exact + S * (d_smooth - d_exact)
            if squash_flag:
                out = rho_delta * jnp.tanh(d / rho_delta)
            else:
                out = d
            out = out * distance_scale
            if is_single:
                return out.reshape(())
            return out.reshape(out_shape)

        return jax.jit(sdf_smooth)

    @no_type_check
    def _make_smooth_mesh_sdf_alt(
        self,
        *,
        eps: float | None = None,
        beta: float | None = None,
        beam_width: int = 32,
        beam_steps: int | None = None,
        r0: float | None = None,
        rho_delta: float | None = None,
    ) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Build the legacy original-coordinate smooth SDF for diagnostics."""
        if beam_width <= 0:
            raise ValueError(f"beam_width must be positive, got {beam_width}.")

        V = jnp.asarray(self.mesh_vertices)
        F = jnp.asarray(self.mesh_faces)

        if V.ndim != 2 or V.shape[-1] != 3:
            raise ValueError(f"V must have shape (nV, 3). Got {V.shape}.")
        if F.ndim != 2 or F.shape[-1] != 3:
            raise ValueError(f"F must have shape (nF, 3). Got {F.shape}.")
        if int(F.shape[0]) == 0:
            raise ValueError("Mesh has no faces; cannot build an SDF.")

        dtype = V.dtype

        if eps is None:
            if dtype == jnp.float16:
                eps = 1e-4
            elif dtype == jnp.float32:
                eps = 1e-9
            else:
                eps = 1e-15
        eps = jnp.asarray(eps, dtype=dtype)
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        inf = jnp.asarray(jnp.inf, dtype=dtype)

        if beta is None:
            smooth_radius = float(0.05 * (self.diameter or 1.0))
            beta = 1.0 / (smooth_radius * smooth_radius + 1e-30)
        beta = jnp.asarray(beta, dtype=dtype)
        if float(beta) <= 0.0:
            raise ValueError(f"beta must be positive, got {beta}.")
        if r0 is None:
            r0 = float(0.05 * (self.diameter or 1.0))
        r0 = jnp.asarray(r0, dtype=dtype)

        if rho_delta is None:
            rho_delta = float(0.0111 * (self.diameter or 1.0))
        rho_delta = jnp.asarray(rho_delta, dtype=dtype)
        rho_delta = jnp.maximum(rho_delta, jnp.asarray(1e-12, dtype=dtype))

        a = V[F[:, 0]]
        ab = V[F[:, 1]] - a
        ac = V[F[:, 2]] - a

        def _dot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x * y, axis=-1)

        V_np = np.asarray(self.mesh.vertices, dtype=float)
        F_np = np.asarray(self.mesh.faces, dtype=np.int64)
        a_np = V_np[F_np[:, 0]]
        b_np = V_np[F_np[:, 1]]
        c_np = V_np[F_np[:, 2]]

        leaf_size = 16
        tri_bbox_min = np.minimum(np.minimum(a_np, b_np), c_np)
        tri_bbox_max = np.maximum(np.maximum(a_np, b_np), c_np)
        tri_centers = 0.5 * (tri_bbox_min + tri_bbox_max)
        bvh = build_packed_bvh(
            tri_bbox_min,
            tri_bbox_max,
            tri_centers,
            leaf_size=leaf_size,
            dtype=dtype,
        )

        if beam_steps is None:
            beam_steps = int(bvh.max_depth + 2)
        if beam_steps <= 0:
            raise ValueError(f"beam_steps must be positive, got {beam_steps}.")

        sdf_exact = _MeshSDFImpl._make_mesh_sdf_traversal(self, eps=eps, bvh=bvh)

        def _point_triangle_dist2_a_ab_ac(
            p: jnp.ndarray,
            a_t: jnp.ndarray,
            ab_t: jnp.ndarray,
            ac_t: jnp.ndarray,
        ) -> jnp.ndarray:
            u = a_t - p
            v = u + ab_t
            w = u + ac_t
            bc_t = ac_t - ab_t

            d1 = -_dot(ab_t, u)
            d2 = -_dot(ac_t, u)
            d3 = -_dot(ab_t, v)
            d4 = -_dot(ac_t, v)
            d5 = -_dot(ab_t, w)
            d6 = -_dot(ac_t, w)

            cond_a = (d1 <= zero) & (d2 <= zero)
            cond_b = (d3 >= zero) & (d4 <= d3)
            cond_c = (d6 >= zero) & (d5 <= d6)

            dist2_a = _dot(u, u)
            dist2_b = _dot(v, v)
            dist2_c = _dot(w, w)

            vc = d1 * d4 - d3 * d2
            cond_ab = (vc <= zero) & (d1 >= zero) & (d3 <= zero)
            t_ab = d1 / jnp.maximum(d1 - d3, eps)
            diff_ab = u + t_ab[..., None] * ab_t
            dist2_ab = _dot(diff_ab, diff_ab)

            vb = d5 * d2 - d1 * d6
            cond_ac = (vb <= zero) & (d2 >= zero) & (d6 <= zero)
            t_ac = d2 / jnp.maximum(d2 - d6, eps)
            diff_ac = u + t_ac[..., None] * ac_t
            dist2_ac = _dot(diff_ac, diff_ac)

            va = d3 * d6 - d5 * d4
            cond_bc = (va <= zero) & ((d4 - d3) >= zero) & ((d5 - d6) >= zero)
            t_bc = (d4 - d3) / jnp.maximum((d4 - d3) + (d5 - d6), eps)
            diff_bc = v + t_bc[..., None] * bc_t
            dist2_bc = _dot(diff_bc, diff_bc)

            denom = va + vb + vc
            v_face = vb / jnp.maximum(denom, eps)
            w_face = vc / jnp.maximum(denom, eps)
            diff_face = u + v_face[..., None] * ab_t + w_face[..., None] * ac_t
            dist2_face = _dot(diff_face, diff_face)

            dist2 = dist2_face
            dist2 = jnp.where(cond_bc, dist2_bc, dist2)
            dist2 = jnp.where(cond_ac, dist2_ac, dist2)
            dist2 = jnp.where(cond_c, dist2_c, dist2)
            dist2 = jnp.where(cond_ab, dist2_ab, dist2)
            dist2 = jnp.where(cond_b, dist2_b, dist2)
            dist2 = jnp.where(cond_a, dist2_a, dist2)
            return dist2

        def _coerce_points3(
            points: jnp.ndarray,
        ) -> tuple[jnp.ndarray, tuple[int, ...], bool]:
            pts = jnp.asarray(points, dtype=dtype)
            if pts.ndim == 0:
                p3 = jnp.repeat(pts, 3)
                return p3.reshape((1, 3)), (), True
            if pts.ndim == 1:
                if pts.shape[0] == 3:
                    return pts.reshape((1, 3)), (), True
                n = int(pts.shape[0])
                if n < 3:
                    buf = jnp.zeros((3,), dtype=pts.dtype)
                    buf = buf.at[:n].set(pts)
                    pts = buf
                else:
                    pts = pts[-3:]
                return pts.reshape((1, 3)), (), True
            if pts.shape[-1] != 3:
                pts = pts.reshape(-1)[-3:]
                return pts.reshape((1, 3)), (), True
            out_shape = pts.shape[:-1]
            return pts.reshape((-1, 3)), out_shape, False

        def sdf_smooth(points: jnp.ndarray) -> jnp.ndarray:
            pts, out_shape, is_single = _coerce_points3(points)

            d_exact = sdf_exact(pts)
            d_for_blend = jax.lax.stop_gradient(d_exact)

            pts_select = jax.lax.stop_gradient(pts)
            tri, valid = beam_select_leaf_items(
                pts_select, bvh=bvh, beam_width=beam_width, steps=beam_steps
            )
            tri = jnp.asarray(tri, dtype=jnp.int32)
            safe_tri = jnp.where(valid, tri, jnp.int32(0))

            a_t = a[safe_tri]
            ab_t = ab[safe_tri]
            ac_t = ac[safe_tri]

            dist2 = _point_triangle_dist2_a_ab_ac(pts[:, None, :], a_t, ab_t, ac_t)
            dist2_use = jnp.where(valid, dist2, zero)
            has_valid = jnp.any(valid, axis=1)

            scaled = jnp.where(valid, -beta * dist2, -inf)
            logZ = jax.scipy.special.logsumexp(scaled, axis=1, keepdims=True)
            weights = jnp.exp(scaled - logZ)
            weights = jnp.where(valid, weights, zero)
            m = jnp.sum(weights * dist2_use, axis=1)
            unsigned_smooth = jnp.sqrt(jnp.maximum(m, zero))
            unsigned = jnp.where(has_valid, unsigned_smooth, jnp.abs(d_exact))

            sgn = jnp.where(d_exact < zero, -one, one)
            sgn = jax.lax.stop_gradient(sgn)
            d_smooth = sgn * unsigned

            inv_r0 = one / (r0 + eps)
            t = (d_for_blend * inv_r0) ** 2
            t = jnp.minimum(t, one)
            S = t * t * (3.0 - 2.0 * t)

            d = d_exact + S * (d_smooth - d_exact)
            out = rho_delta * jnp.tanh(d / rho_delta)
            if is_single:
                return out.reshape(())
            return out.reshape(out_shape)

        return jax.jit(sdf_smooth)


def make_mesh_sdf(
    geom,
    *,
    eps: float | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return _MeshSDFImpl._make_mesh_sdf(geom, eps=eps)


def make_mesh_sdf_fast(
    geom,
    *,
    eps: float | None = None,
    beam_width: int = 8,
    leaf_size: int = 8,
    dtype: jnp.dtype = jnp.float32,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a faster (approximate) mesh SDF for inside/outside predicates.

    Uses a smaller BVH leaf size and beam width, typically in float32, to reduce
    per-query work. This is intended for sampling/containment, not for PDE
    optimization (use `make_smooth_mesh_sdf` / `geom.adf` for smooth envelopes).
    """
    return _MeshSDFImpl._make_mesh_sdf(
        geom,
        eps=eps,
        beam_width=beam_width,
        leaf_size=leaf_size,
        dtype=dtype,
    )


def make_mesh_inside_predicate_fast(
    geom,
    *,
    inside_tol: float = -1e-8,
    eps: float | None = None,
    beam_width: int = 8,
    leaf_size: int = 8,
    dtype: jnp.dtype = jnp.float32,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a fast inside/outside predicate using a beam-accelerated mesh SDF."""
    sdf = make_mesh_sdf_fast(
        geom,
        eps=eps,
        beam_width=beam_width,
        leaf_size=leaf_size,
        dtype=dtype,
    )
    tol = jnp.asarray(inside_tol, dtype=dtype)

    def inside(points: jnp.ndarray) -> jnp.ndarray:
        return sdf(points) < tol

    return jax.jit(inside)


def make_smooth_mesh_sdf(
    geom,
    *,
    eps: float | None = None,
    beta: float | None = None,
    beam_width: int = 32,
    beam_steps: int | None = None,
    r0: float | None = None,
    rho_delta: float | None = None,
    equivalence_power: float | None = None,
    squash: bool = True,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return _MeshSDFImpl._make_smooth_mesh_sdf(
        geom,
        eps=eps,
        beta=beta,
        beam_width=beam_width,
        beam_steps=beam_steps,
        r0=r0,
        rho_delta=rho_delta,
        equivalence_power=equivalence_power,
        squash=squash,
    )


def make_smooth_mesh_sdf_alt(
    geom,
    *,
    eps: float | None = None,
    beta: float | None = None,
    beam_width: int = 32,
    beam_steps: int | None = None,
    r0: float | None = None,
    rho_delta: float | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the legacy original-coordinate smooth SDF for diagnostics.

    Unlike the scale-normalized default, this variant shares one physical-coordinate
    BVH between its global-nearest and soft-min terms.
    """
    return _MeshSDFImpl._make_smooth_mesh_sdf_alt(
        geom,
        eps=eps,
        beta=beta,
        beam_width=beam_width,
        beam_steps=beam_steps,
        r0=r0,
        rho_delta=rho_delta,
    )


def make_compact_boundary_factor(
    distance: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    scale: float,
    saturation_fraction: float = 0.05,
    linear_fraction: float = 0.5,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    r"""Compactly saturate a signed-distance-like field after an exact linear collar.

    The scalar transform is odd and exactly the identity while
    ``|distance| <= linear_fraction * saturation_fraction * scale``. A C6 transition
    then reaches an exact plateau at ``saturation_fraction * scale``. Consequently,
    the transform preserves every boundary derivative while preventing distant
    medial-axis nonsmoothness from entering derivatives of the boundary factor.
    """
    scale_value = float(scale)
    saturation_value = float(saturation_fraction)
    linear_value = float(linear_fraction)
    if not np.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {scale!r}.")
    if not np.isfinite(saturation_value) or not 0.0 < saturation_value < 1.0:
        raise ValueError(
            "saturation_fraction must be finite and strictly between zero and one."
        )
    if not np.isfinite(linear_value) or not 0.0 < linear_value < 1.0:
        raise ValueError(
            "linear_fraction must be finite and strictly between zero and one."
        )
    delta_value = scale_value * saturation_value
    transition_width = 1.0 - linear_value
    plateau = linear_value + 0.5 * transition_width

    def boundary_factor(points: jnp.ndarray) -> jnp.ndarray:
        signed = jnp.asarray(distance(points))
        delta = jnp.asarray(delta_value, dtype=signed.dtype)
        z = jnp.abs(signed) / delta
        u_raw = jnp.clip(
            (z - linear_value) / transition_width,
            jnp.asarray(0.0, dtype=signed.dtype),
            jnp.asarray(1.0, dtype=signed.dtype),
        )
        # The transition polynomial is inactive in the linear collar. Evaluating
        # its zero powers there can still produce 0*NaN tangents under nested AD
        # when ``distance`` has a custom JVP, so keep the inactive input at one.
        u = jnp.where(
            z > linear_value,
            u_raw,
            jnp.asarray(1.0, dtype=signed.dtype),
        )

        # Integral of one minus the order-five generalized smoothstep. It agrees
        # with ``u`` through sixth order at zero and is flat through sixth order at one.
        transition = (
            u
            - 66.0 * u**7
            + 247.5 * u**8
            - 385.0 * u**9
            + 308.0 * u**10
            - 126.0 * u**11
            + 21.0 * u**12
        )
        magnitude = linear_value + transition_width * transition
        transition_denominator = jnp.where(
            z > linear_value, z, jnp.asarray(1.0, dtype=signed.dtype)
        )
        plateau_denominator = jnp.where(z >= 1.0, z, jnp.asarray(1.0, dtype=signed.dtype))
        ratio_transition = magnitude / transition_denominator
        ratio_plateau = plateau / plateau_denominator
        ratio = jnp.where(
            z <= linear_value,
            jnp.asarray(1.0, dtype=signed.dtype),
            jnp.where(z < 1.0, ratio_transition, ratio_plateau),
        )
        return signed * ratio

    return jax.jit(boundary_factor)


__all__ = [
    "make_mesh_inside_predicate_fast",
    "make_compact_boundary_factor",
    "make_mesh_sdf",
    "make_mesh_sdf_fast",
    "make_smooth_mesh_sdf",
    "make_smooth_mesh_sdf_alt",
]
