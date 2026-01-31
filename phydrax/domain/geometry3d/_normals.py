#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import no_type_check

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from ..._bvh import beam_select_leaf_items, build_packed_bvh


def make_smooth_mesh_normal_field(
    geom,
    *,
    eps: float | None = None,
    beta: float | None = None,
    beam_width: int = 16,
    beam_steps: int | None = None,
    r0: float | None = None,
) -> Callable[[Array], Float[Array, "... 3"]]:
    """Build a fast mesh normal field, exact on the boundary.

    Returns a callable `n(points)` that maps points with shape `(..., 3)` to
    outward unit normals with the same shape.

    Construction mirrors `make_smooth_mesh_sdf`:
    - `n_exact` is a pseudonormal at the closest point (face/edge/vertex).
    - `n_soft` is a distance-weighted average of per-candidate distance directions.
    - `n = normalize((1-S)*n_exact + S*n_soft)`, with `S(|phi|)` chosen so `S(0)=0`
      and `dS/d|phi|(0)=0`, making the field exact on the boundary.
    """
    if beam_width <= 0:
        raise ValueError(f"beam_width must be positive, got {beam_width}.")

    V = jnp.asarray(geom.mesh_vertices)
    F = jnp.asarray(geom.mesh_faces)

    if V.ndim != 2 or V.shape[-1] != 3:
        raise ValueError(f"V must have shape (nV, 3). Got {V.shape}.")
    if F.ndim != 2 or F.shape[-1] != 3:
        raise ValueError(f"F must have shape (nF, 3). Got {F.shape}.")
    if int(F.shape[0]) == 0:
        raise ValueError("Mesh has no faces; cannot build a normal field.")

    dtype = V.dtype

    eps_value = eps
    if eps_value is None:
        if dtype == jnp.float16:
            eps_value = 1e-4
        elif dtype == jnp.float32:
            eps_value = 1e-9
        else:
            eps_value = 1e-12
    eps_arr = jnp.asarray(eps_value, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    one = jnp.asarray(1.0, dtype=dtype)
    inf = jnp.asarray(jnp.inf, dtype=dtype)

    beta_value = beta
    if beta_value is None:
        smooth_radius = float(0.05 * (geom.diameter or 1.0))
        beta_value = 1.0 / (smooth_radius * smooth_radius + 1e-30)
    if float(beta_value) <= 0.0:
        raise ValueError(f"beta must be positive, got {beta_value}.")
    beta_arr = jnp.asarray(beta_value, dtype=dtype)

    r0_value = r0
    if r0_value is None:
        r0_value = float(0.05 * (geom.diameter or 1.0))
    r0_arr = jnp.asarray(r0_value, dtype=dtype)

    # Precompute per-triangle vertex positions and edges.
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    ab = b - a
    ac = c - a
    bc = c - b

    def _dot(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(x * y, axis=-1)

    # ---------------------------------------------------------------------
    # Pseudonormal sign precomputation (host-side, static)
    # ---------------------------------------------------------------------
    V_np = np.asarray(geom.mesh.vertices, dtype=float)
    F_np = np.asarray(geom.mesh.faces, dtype=np.int64)
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
    # Static BVH (host-side)
    # ---------------------------------------------------------------------
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

    def _point_triangle_dist2_and_diff(
        p: jnp.ndarray,
        a_t: jnp.ndarray,
        b_t: jnp.ndarray,
        c_t: jnp.ndarray,
        ab_t: jnp.ndarray,
        ac_t: jnp.ndarray,
        bc_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        p = jnp.asarray(p, dtype=dtype)
        u = a_t - p
        v = b_t - p
        w = c_t - p

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
        t_ab = d1 / (d1 - d3 + eps_arr)
        diff_ab = u + t_ab[..., None] * ab_t
        dist2_ab = _dot(diff_ab, diff_ab)

        vb = d5 * d2 - d1 * d6
        cond_ac = (vb <= zero) & (d2 >= zero) & (d6 <= zero)
        t_ac = d2 / (d2 - d6 + eps_arr)
        diff_ac = u + t_ac[..., None] * ac_t
        dist2_ac = _dot(diff_ac, diff_ac)

        va = d3 * d6 - d5 * d4
        cond_bc = (va <= zero) & ((d4 - d3) >= zero) & ((d5 - d6) >= zero)
        t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6) + eps_arr)
        diff_bc = v + t_bc[..., None] * bc_t
        dist2_bc = _dot(diff_bc, diff_bc)

        denom = va + vb + vc
        v_face = vb / (denom + eps_arr)
        w_face = vc / (denom + eps_arr)
        diff_face = u + v_face[..., None] * ab_t + w_face[..., None] * ac_t
        dist2_face = _dot(diff_face, diff_face)

        dist2 = dist2_face
        dist2 = jnp.where(cond_bc, dist2_bc, dist2)
        dist2 = jnp.where(cond_ac, dist2_ac, dist2)
        dist2 = jnp.where(cond_c, _dot(w, w), dist2)
        dist2 = jnp.where(cond_ab, dist2_ab, dist2)
        dist2 = jnp.where(cond_b, _dot(v, v), dist2)
        dist2 = jnp.where(cond_a, _dot(u, u), dist2)

        diff = diff_face
        diff = jnp.where(cond_bc[..., None], diff_bc, diff)
        diff = jnp.where(cond_ac[..., None], diff_ac, diff)
        diff = jnp.where(cond_c[..., None], w, diff)
        diff = jnp.where(cond_ab[..., None], diff_ab, diff)
        diff = jnp.where(cond_b[..., None], v, diff)
        diff = jnp.where(cond_a[..., None], u, diff)

        return dist2, diff

    def _point_triangle_dist2_side_diff_n(
        p: jnp.ndarray,
        a_t: jnp.ndarray,
        b_t: jnp.ndarray,
        c_t: jnp.ndarray,
        ab_t: jnp.ndarray,
        ac_t: jnp.ndarray,
        bc_t: jnp.ndarray,
        n_face_t: jnp.ndarray,
        n_va_t: jnp.ndarray,
        n_vb_t: jnp.ndarray,
        n_vc_t: jnp.ndarray,
        n_eab_t: jnp.ndarray,
        n_ebc_t: jnp.ndarray,
        n_eca_t: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dist2, diff = _point_triangle_dist2_and_diff(p, a_t, b_t, c_t, ab_t, ac_t, bc_t)

        p = jnp.asarray(p, dtype=dtype)
        u = a_t - p
        v = b_t - p
        w = c_t - p

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
        vb = d5 * d2 - d1 * d6
        cond_ac = (vb <= zero) & (d2 >= zero) & (d6 <= zero)
        va = d3 * d6 - d5 * d4
        cond_bc = (va <= zero) & ((d4 - d3) >= zero) & ((d5 - d6) >= zero)

        n_pseudo = n_face_t
        n_pseudo = jnp.where(cond_bc[..., None], n_ebc_t, n_pseudo)
        n_pseudo = jnp.where(cond_ac[..., None], n_eca_t, n_pseudo)
        n_pseudo = jnp.where(cond_c[..., None], n_vc_t, n_pseudo)
        n_pseudo = jnp.where(cond_ab[..., None], n_eab_t, n_pseudo)
        n_pseudo = jnp.where(cond_b[..., None], n_vb_t, n_pseudo)
        n_pseudo = jnp.where(cond_a[..., None], n_va_t, n_pseudo)

        side = -_dot(diff, n_pseudo)
        return dist2, side, diff, n_pseudo

    def _coerce_points(points: Array) -> tuple[jnp.ndarray, tuple[int, ...]]:
        pts = jnp.asarray(points, dtype=dtype)
        if pts.ndim == 1:
            if pts.shape[0] != 3:
                raise ValueError(
                    f"Expected points with shape (..., 3), got {pts.shape!r}"
                )
            return pts.reshape((1, 3)), (3,)
        if pts.ndim == 0:
            raise ValueError(f"Expected points with shape (..., 3), got {pts.shape!r}")
        if pts.shape[-1] != 3:
            raise ValueError(f"Expected points with shape (..., 3), got {pts.shape!r}")
        flat = pts.reshape((-1, 3))
        return flat, pts.shape

    def _normalize(v: jnp.ndarray) -> jnp.ndarray:
        nrm = jnp.sqrt(_dot(v, v) + jnp.asarray(1e-30, dtype=dtype))
        return v / nrm[..., None]

    def normal_single(p: jnp.ndarray) -> jnp.ndarray:
        p = jnp.asarray(p, dtype=dtype).reshape((3,))

        p_select = jax.lax.stop_gradient(p)
        tri, valid = beam_select_leaf_items(
            p_select, bvh=bvh, beam_width=beam_width, steps=beam_steps
        )
        safe = jnp.asarray(tri, dtype=jnp.int32)

        a_t = a[safe]
        b_t = b[safe]
        c_t = c[safe]
        ab_t = ab[safe]
        ac_t = ac[safe]
        bc_t = bc[safe]

        dist2_cand, diff_cand = _point_triangle_dist2_and_diff(
            p, a_t, b_t, c_t, ab_t, ac_t, bc_t
        )
        dist2_masked = jnp.where(valid, dist2_cand, inf)
        best_k = jnp.argmin(dist2_masked)
        best_tri = safe[best_k]

        dist2_b, side_b, _diff_b, n_pseudo_b = _point_triangle_dist2_side_diff_n(
            p,
            a[best_tri][None, :],
            b[best_tri][None, :],
            c[best_tri][None, :],
            ab[best_tri][None, :],
            ac[best_tri][None, :],
            bc[best_tri][None, :],
            n_face[best_tri][None, :],
            n_va[best_tri][None, :],
            n_vb[best_tri][None, :],
            n_vc[best_tri][None, :],
            n_eab[best_tri][None, :],
            n_ebc[best_tri][None, :],
            n_eca[best_tri][None, :],
        )

        dist2_b = dist2_b.reshape(())
        side_b = side_b.reshape(())
        n_exact = _normalize(n_pseudo_b.reshape((3,)))

        # Snap-to-vertex safeguard: for points numerically on a mesh vertex, the
        # strict region predicates can misclassify the closest feature (face vs
        # vertex) due to tiny coordinate discrepancies. In that case, prefer the
        # corresponding vertex pseudonormal.
        snap_tol2 = (jnp.asarray(10.0, dtype=dtype) * eps_arr) ** 2
        d2_a = _dot(p - a[best_tri], p - a[best_tri])
        d2_bv = _dot(p - b[best_tri], p - b[best_tri])
        d2_c = _dot(p - c[best_tri], p - c[best_tri])
        d2_verts = jnp.stack([d2_a, d2_bv, d2_c], axis=0)
        k_vert = jnp.argmin(d2_verts)
        n_verts = jnp.stack([n_va[best_tri], n_vb[best_tri], n_vc[best_tri]], axis=0)
        n_vert = n_verts[k_vert]
        use_vert = jnp.min(d2_verts) < snap_tol2
        n_exact = jnp.where(use_vert, _normalize(n_vert), n_exact)

        unsigned = jnp.sqrt(jnp.maximum(dist2_b, zero))
        signed = jnp.where(side_b < zero, -unsigned, unsigned)

        # Smooth direction: average candidate distance directions with softmin weights.
        has_valid = jnp.any(valid)
        sgn = jnp.where(signed < zero, -one, one)

        def _smooth_dir(_: None) -> jnp.ndarray:
            scaled = jnp.where(valid, -beta_arr * dist2_cand, -inf)
            logZ = jax.scipy.special.logsumexp(scaled)
            weights = jnp.exp(scaled - logZ)
            inv_r = one / jnp.sqrt(dist2_cand + eps_arr)
            dirs = (-sgn) * diff_cand * inv_r[..., None]
            vec = jnp.sum(weights[..., None] * dirs, axis=0)
            vec_norm = jnp.sqrt(_dot(vec, vec))
            use_vec = vec_norm > jnp.asarray(1e-12, dtype=dtype)
            vec_norm_safe = vec_norm + jnp.asarray(1e-30, dtype=dtype)
            n_soft = jnp.where(use_vec, vec / vec_norm_safe, n_exact)
            return n_soft

        n_soft = jax.lax.cond(has_valid, _smooth_dir, lambda _: n_exact, operand=None)

        d_for_blend = jax.lax.stop_gradient(jnp.abs(signed))
        inv_r0 = one / (r0_arr + eps_arr)
        t = (d_for_blend * inv_r0) ** 2
        t = jnp.minimum(t, one)
        S = t * t * (3.0 - 2.0 * t)

        n = (one - S) * n_exact + S * n_soft
        n = _normalize(n)
        return jax.lax.stop_gradient(n)

    normal_batched = jax.jit(jax.vmap(normal_single))

    def normal(points: Array) -> Float[Array, "... 3"]:
        pts_flat, out_shape = _coerce_points(points)
        normals_flat = normal_batched(pts_flat)
        if out_shape == (3,):
            return normals_flat.reshape((3,))
        return normals_flat.reshape(out_shape)

    return normal


@no_type_check
def _boundary_normals_orig_nograd_impl(
    geom, points: Array
) -> Float[Array, "num_points 3"]:
    return geom._boundary_normals_field(points)


def _boundary_normals_orig_nograd(geom, points: Array) -> Float[Array, "num_points 3"]:
    return _boundary_normals_orig_nograd_impl(geom, points)


def _boundary_normals(geom, points: Array) -> Float[Array, "num_points 3"]:
    return _boundary_normals_orig_nograd_impl(geom, points)


__all__ = [
    "_boundary_normals",
    "_boundary_normals_orig_nograd",
    "make_smooth_mesh_normal_field",
]
