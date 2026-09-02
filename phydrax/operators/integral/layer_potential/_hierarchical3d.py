#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import ArraySpace, FunctionLinearOperator, OperatorProperties
from ._fast_provider import (
    BEMExecutionEnvelope,
    BEMLocalBlock3D,
    LaplaceDP0ExactNearProvider3D,
)
from ._galerkin3d import LaplaceSingleLayerDP0Galerkin3D


ScalarFastAlgorithm3D = Literal["fmm", "h-matrix", "h2-matrix"]


class ScalarFastPolicy3D(StrictModule, NonTrainableState):
    """Static accuracy and resource envelope for a 3-D scalar fast provider."""

    tolerance: float = eqx.field(static=True)
    leaf_size: int = eqx.field(static=True)
    admissibility: float = eqx.field(static=True)
    interpolation_order: int = eqx.field(static=True)
    maximum_rank: int = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    maximum_blocks: int = eqx.field(static=True)
    maximum_resident_bytes: int = eqx.field(static=True)
    maximum_block_entries: int = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        tolerance: float = 1e-5,
        leaf_size: int = 32,
        admissibility: float = 1.0,
        interpolation_order: int = 3,
        maximum_rank: int = 32,
        maximum_depth: int = 24,
        maximum_blocks: int = 1_000_000,
        maximum_resident_bytes: int = 2_000_000_000,
        maximum_block_entries: int = 1_000_000,
        formulation: str = "strong",
    ):
        tolerance_ = float(tolerance)
        eta = float(admissibility)
        integers = (
            int(leaf_size),
            int(interpolation_order),
            int(maximum_rank),
            int(maximum_depth),
            int(maximum_blocks),
            int(maximum_resident_bytes),
            int(maximum_block_entries),
        )
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Fast-provider tolerance must be finite and positive.")
        if not np.isfinite(eta) or eta <= 0.0:
            raise ValueError("Fast-provider admissibility must be finite and positive.")
        if any(value <= 0 for value in integers):
            raise ValueError("Fast-provider resource bounds must be positive.")
        formulation_ = str(formulation)
        if formulation_ not in ("weak", "strong"):
            raise ValueError("formulation must be 'weak' or 'strong'.")
        self.tolerance = tolerance_
        self.leaf_size = integers[0]
        self.admissibility = eta
        self.interpolation_order = integers[1]
        self.maximum_rank = integers[2]
        self.maximum_depth = integers[3]
        self.maximum_blocks = integers[4]
        self.maximum_resident_bytes = integers[5]
        self.maximum_block_entries = integers[6]
        self.formulation = formulation_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "scalar-fast-policy-3d",
                "tolerance": tolerance_,
                "leaf_size": integers[0],
                "admissibility": eta,
                "interpolation_order": integers[1],
                "maximum_rank": integers[2],
                "maximum_depth": integers[3],
                "maximum_blocks": integers[4],
                "maximum_resident_bytes": integers[5],
                "maximum_block_entries": integers[6],
                "formulation": formulation_,
            }
        )


class ScalarFastEvidence3D(StrictModule, NonTrainableState):
    algorithm: ScalarFastAlgorithm3D = eqx.field(static=True)
    cluster_count: int = eqx.field(static=True)
    far_block_count: int = eqx.field(static=True)
    exact_near_block_count: int = eqx.field(static=True)
    maximum_depth: int = eqx.field(static=True)
    maximum_active_rank: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    maximum_block_relative_error: float = eqx.field(static=True)
    maximum_nested_transfer_defect: float = eqx.field(static=True)
    exact_near: bool = eqx.field(static=True)
    global_dense_materialized: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class ScalarFastCluster3D(StrictModule, NonTrainableState):
    indices: Array
    basis: Array
    center: Array
    children: tuple[int, ...] = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    diameter: float = eqx.field(static=True)


class ScalarHBlock3D(StrictModule, NonTrainableState):
    target_indices: Array
    source_indices: Array
    left_factor: Array
    right_factor: Array
    relative_error: float = eqx.field(static=True)


class ScalarH2Coupling3D(StrictModule, NonTrainableState):
    matrix: Array
    target_cluster: int = eqx.field(static=True)
    source_cluster: int = eqx.field(static=True)
    relative_error: float = eqx.field(static=True)


class PreparedScalarFastProvider3D(StrictModule, NonTrainableState):
    """No-global-dense DP0 Galerkin fast action with exact prepared near blocks."""

    clusters: tuple[ScalarFastCluster3D, ...]
    h_blocks: tuple[ScalarHBlock3D, ...]
    h2_couplings: tuple[ScalarH2Coupling3D, ...]
    near_blocks: tuple[BEMLocalBlock3D, ...]
    evidence: ScalarFastEvidence3D
    envelope: BEMExecutionEnvelope
    algorithm: ScalarFastAlgorithm3D = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def _matrix_input(self, values: ArrayLike, /) -> tuple[Array, bool]:
        value = jnp.asarray(values)
        if value.ndim == 1:
            if value.shape != (self.face_count,):
                raise ValueError("Fast BEM input has incompatible face dimension.")
            return value[:, None], True
        if value.ndim == 2 and value.shape[0] == self.face_count:
            return value, False
        raise ValueError(
            "Fast BEM input must have shape (face_count,) or (face_count, rhs)."
        )

    def apply(self, values: ArrayLike, /, *, transpose: bool = False) -> Array:
        state, squeeze = self._matrix_input(values)
        result = jnp.zeros_like(state)
        if self.algorithm == "h-matrix":
            for block in self.h_blocks:
                if transpose:
                    local = block.left_factor.T @ state[block.target_indices]
                    contribution = block.right_factor @ local
                    result = result.at[block.source_indices].add(contribution)
                else:
                    local = block.right_factor.T @ state[block.source_indices]
                    contribution = block.left_factor @ local
                    result = result.at[block.target_indices].add(contribution)
        else:
            coefficients = tuple(
                cluster.basis.T @ state[cluster.indices] for cluster in self.clusters
            )
            for coupling in self.h2_couplings:
                if transpose:
                    target = self.clusters[coupling.target_cluster]
                    source = self.clusters[coupling.source_cluster]
                    contribution = source.basis @ (
                        coupling.matrix.T @ coefficients[coupling.target_cluster]
                    )
                    result = result.at[source.indices].add(contribution)
                else:
                    target = self.clusters[coupling.target_cluster]
                    contribution = target.basis @ (
                        coupling.matrix @ coefficients[coupling.source_cluster]
                    )
                    result = result.at[target.indices].add(contribution)
        for block in self.near_blocks:
            if transpose:
                contribution = block.values.T @ state[block.target_indices]
                result = result.at[block.source_indices].add(contribution)
            else:
                contribution = block.values @ state[block.source_indices]
                result = result.at[block.target_indices].add(contribution)
        return result[:, 0] if squeeze else result

    def mv(self, values: ArrayLike, /) -> Array:
        return self.apply(values)

    def transpose_mv(self, values: ArrayLike, /) -> Array:
        return self.apply(values, transpose=True)

    def adjoint_mv(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        return jnp.conj(self.transpose_mv(jnp.conj(value)))

    def as_linear_operator(self, /) -> FunctionLinearOperator:
        dtype = self.clusters[0].basis.dtype
        space = ArraySpace((self.face_count,), dtype=dtype)
        return FunctionLinearOperator(
            self.mv,
            source=space,
            target=space,
            transpose_action=self.transpose_mv,
            adjoint_action=self.adjoint_mv,
            properties=OperatorProperties(evidence={}),
            operator_id=self.provider_id,
            closure_convert=False,
        )


@dataclass(slots=True)
class _HostCluster:
    indices: np.ndarray
    center: np.ndarray
    diameter: float
    depth: int
    children: tuple[int, ...]


def _cluster_tree(
    points: np.ndarray, policy: ScalarFastPolicy3D, /
) -> tuple[list[_HostCluster], int]:
    clusters: list[_HostCluster] = []

    def build(indices: np.ndarray, depth: int) -> int:
        selected = points[indices]
        lower = np.min(selected, axis=0)
        upper = np.max(selected, axis=0)
        center = 0.5 * (lower + upper)
        diameter = float(np.linalg.norm(upper - lower))
        cluster_index = len(clusters)
        clusters.append(_HostCluster(indices, center, diameter, depth, ()))
        if indices.size <= policy.leaf_size:
            return cluster_index
        if depth >= policy.maximum_depth:
            raise ValueError("Fast BEM cluster tree exceeded maximum_depth.")
        axis = int(np.argmax(upper - lower))
        order = np.lexsort((indices, selected[:, axis]))
        ordered = indices[order]
        midpoint = ordered.size // 2
        left = build(ordered[:midpoint], depth + 1)
        right = build(ordered[midpoint:], depth + 1)
        clusters[cluster_index].children = (left, right)
        return cluster_index

    root = build(np.arange(points.shape[0], dtype=np.int32), 0)
    return clusters, root


def _admissible(
    target: _HostCluster,
    source: _HostCluster,
    eta: float,
    /,
) -> bool:
    separation = float(np.linalg.norm(target.center - source.center))
    distance = separation - 0.5 * (target.diameter + source.diameter)
    return distance > 0.0 and max(target.diameter, source.diameter) <= eta * distance


def _block_partition(
    clusters: list[_HostCluster],
    root: int,
    policy: ScalarFastPolicy3D,
    /,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    far: list[tuple[int, int]] = []
    near: list[tuple[int, int]] = []

    def visit(target_index: int, source_index: int) -> None:
        if len(far) + len(near) >= policy.maximum_blocks:
            raise ValueError("Fast BEM block tree exceeded maximum_blocks.")
        target = clusters[target_index]
        source = clusters[source_index]
        if _admissible(target, source, policy.admissibility):
            far.append((target_index, source_index))
            return
        if not target.children and not source.children:
            near.append((target_index, source_index))
            return
        split_target = bool(target.children) and (
            not source.children or target.diameter >= source.diameter
        )
        if split_target:
            for child in target.children:
                visit(child, source_index)
        else:
            for child in source.children:
                visit(target_index, child)

    visit(root, root)
    return far, near


def _monomial_exponents(
    order: int, maximum_rank: int, /
) -> tuple[tuple[int, int, int], ...]:
    values = tuple(
        exponent
        for exponent in product(range(order + 1), repeat=3)
        if sum(exponent) <= order
    )
    return values[:maximum_rank]


def _cluster_basis(
    points: np.ndarray,
    cluster: _HostCluster,
    exponents: tuple[tuple[int, int, int], ...],
    /,
) -> np.ndarray:
    scale = max(0.5 * cluster.diameter, np.finfo(float).eps)
    local = (points[cluster.indices] - cluster.center[None, :]) / scale
    vandermonde = np.stack(
        tuple(
            local[:, 0] ** exponent[0]
            * local[:, 1] ** exponent[1]
            * local[:, 2] ** exponent[2]
            for exponent in exponents
        ),
        axis=1,
    )
    basis, _ = np.linalg.qr(vandermonde, mode="reduced")
    return basis


def _laplace_block(
    centroids: np.ndarray,
    areas: np.ndarray,
    target_indices: np.ndarray,
    source_indices: np.ndarray,
    formulation: str,
    /,
) -> np.ndarray:
    differences = centroids[target_indices, None, :] - centroids[None, source_indices, :]
    radii = np.linalg.norm(differences, axis=-1)
    if np.any(radii <= 0.0):
        raise ValueError("Admissible scalar blocks cannot contain coincident centroids.")
    matrix = (
        areas[target_indices, None] * areas[None, source_indices] / (4.0 * pi * radii)
    )
    if formulation == "strong":
        matrix = matrix / areas[target_indices, None]
    return matrix


def _prepare_scalar_fast_provider_3d(
    prepared: LaplaceSingleLayerDP0Galerkin3D,
    algorithm: ScalarFastAlgorithm3D,
    policy: ScalarFastPolicy3D | None,
    /,
) -> PreparedScalarFastProvider3D:
    if not isinstance(prepared, LaplaceSingleLayerDP0Galerkin3D):
        raise TypeError("3-D scalar fast providers require Laplace DP0 Galerkin data.")
    selected = ScalarFastPolicy3D() if policy is None else policy
    if not isinstance(selected, ScalarFastPolicy3D):
        raise TypeError("policy must be ScalarFastPolicy3D or None.")
    triangle_mesh = prepared._binding.region.triangle_mesh
    vertices = np.asarray(triangle_mesh.vertices, dtype=float)
    faces = np.asarray(triangle_mesh.faces, dtype=np.int32)
    centroids = np.mean(vertices[faces], axis=1)
    areas = np.asarray(prepared.face_areas, dtype=float)
    clusters_host, root = _cluster_tree(centroids, selected)
    far_pairs, near_pairs = _block_partition(clusters_host, root, selected)
    exact = LaplaceDP0ExactNearProvider3D(
        prepared,
        formulation=selected.formulation,
        max_block_entries=selected.maximum_block_entries,
        max_block_workspace_bytes=selected.maximum_resident_bytes,
    )
    near_blocks = tuple(
        exact.local_block(
            clusters_host[target].indices,
            clusters_host[source].indices,
        )
        for target, source in near_pairs
    )

    exponents = _monomial_exponents(selected.interpolation_order, selected.maximum_rank)
    bases_host = tuple(
        _cluster_basis(centroids, cluster, exponents) for cluster in clusters_host
    )
    clusters = tuple(
        ScalarFastCluster3D(
            indices=jnp.asarray(cluster.indices),
            basis=jnp.asarray(basis),
            center=jnp.asarray(cluster.center),
            children=cluster.children,
            depth=cluster.depth,
            diameter=cluster.diameter,
        )
        for cluster, basis in zip(clusters_host, bases_host, strict=True)
    )
    nested_defect = 0.0
    for parent_index, parent in enumerate(clusters_host):
        for child_index in parent.children:
            parent_positions = {
                int(index): position for position, index in enumerate(parent.indices)
            }
            rows = np.asarray(
                [
                    parent_positions[int(index)]
                    for index in clusters_host[child_index].indices
                ]
            )
            restricted = bases_host[parent_index][rows]
            child_basis = bases_host[child_index]
            transfer = child_basis.T @ restricted
            nested_defect = max(
                nested_defect,
                float(np.linalg.norm(restricted - child_basis @ transfer, ord=np.inf)),
            )

    h_blocks: list[ScalarHBlock3D] = []
    h2_couplings: list[ScalarH2Coupling3D] = []
    maximum_error = 0.0
    maximum_rank = 0
    resident_bytes = sum(
        int(block.values.size * block.values.dtype.itemsize) for block in near_blocks
    )
    for target_index, source_index in far_pairs:
        target = clusters_host[target_index]
        source = clusters_host[source_index]
        matrix = _laplace_block(
            centroids,
            areas,
            target.indices,
            source.indices,
            selected.formulation,
        )
        norm = max(float(np.linalg.norm(matrix)), np.finfo(float).tiny)
        if algorithm == "h-matrix":
            left, singular_values, right_transpose = np.linalg.svd(
                matrix, full_matrices=False
            )
            threshold = selected.tolerance * norm
            rank = int(np.sum(singular_values > threshold))
            rank = max(rank, 1)
            if rank > selected.maximum_rank:
                raise ValueError(
                    "H-matrix far block exhausted maximum_rank without meeting tolerance."
                )
            left_factor = left[:, :rank] * singular_values[:rank][None, :]
            right_factor = right_transpose[:rank].T
            approximation = left_factor @ right_factor.T
            relative_error = float(np.linalg.norm(matrix - approximation) / norm)
            if relative_error > selected.tolerance:
                raise ValueError("H-matrix far block did not meet tolerance.")
            h_blocks.append(
                ScalarHBlock3D(
                    target_indices=jnp.asarray(target.indices),
                    source_indices=jnp.asarray(source.indices),
                    left_factor=jnp.asarray(left_factor),
                    right_factor=jnp.asarray(right_factor),
                    relative_error=relative_error,
                )
            )
            maximum_rank = max(maximum_rank, rank)
            resident_bytes += left_factor.nbytes + right_factor.nbytes
        else:
            target_basis = bases_host[target_index]
            source_basis = bases_host[source_index]
            coupling = target_basis.T @ matrix @ source_basis
            approximation = target_basis @ coupling @ source_basis.T
            relative_error = float(np.linalg.norm(matrix - approximation) / norm)
            if relative_error > selected.tolerance:
                raise ValueError(
                    f"{algorithm} interpolation order did not meet the block tolerance."
                )
            h2_couplings.append(
                ScalarH2Coupling3D(
                    matrix=jnp.asarray(coupling),
                    target_cluster=target_index,
                    source_cluster=source_index,
                    relative_error=relative_error,
                )
            )
            maximum_rank = max(maximum_rank, target_basis.shape[1], source_basis.shape[1])
            resident_bytes += coupling.nbytes
        maximum_error = max(maximum_error, relative_error)
    resident_bytes += sum(basis.nbytes for basis in bases_host)
    if resident_bytes > selected.maximum_resident_bytes:
        raise ValueError("Fast BEM prepared storage exceeds maximum_resident_bytes.")

    provider_name = {
        "fmm": "laplace-fmm-3d",
        "h-matrix": "scalar-h-matrix-3d",
        "h2-matrix": "scalar-h2-matrix-3d",
    }[algorithm]
    envelope = BEMExecutionEnvelope(
        ambient_dimension=3,
        pde="laplace",
        geometry="closed-oriented-triangular-surface",
        formulation=f"dp0-galerkin-single-layer-{selected.formulation}",
        provider=provider_name,
        precision=np.dtype(areas.dtype).name,
        resource_evidence=(
            f"clusters={len(clusters)}",
            f"far-blocks={len(far_pairs)}",
            f"exact-near-blocks={len(near_blocks)}",
            f"resident-bytes={resident_bytes}",
        ),
        error_evidence=(
            f"maximum-independent-block-relative-error={maximum_error}",
            "exact-near-prepared-quadrature-parity",
        ),
        non_goals=(
            "continuum-error-certification",
            "unbounded-rank-or-order",
            "geometry-derivatives-across-tree-or-admissibility-changes",
        ),
        accelerated=True,
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "scalar-fast-evidence-3d",
            "algorithm": algorithm,
            "policy": selected.policy_id,
            "envelope": envelope.envelope_id,
            "clusters": len(clusters),
            "far": len(far_pairs),
            "near": len(near_blocks),
            "rank": maximum_rank,
            "resident_bytes": resident_bytes,
            "maximum_error": maximum_error,
            "nested_defect": nested_defect,
        }
    )
    evidence = ScalarFastEvidence3D(
        algorithm=algorithm,
        cluster_count=len(clusters),
        far_block_count=len(far_pairs),
        exact_near_block_count=len(near_blocks),
        maximum_depth=max(cluster.depth for cluster in clusters_host),
        maximum_active_rank=maximum_rank,
        resident_bytes=resident_bytes,
        maximum_block_relative_error=maximum_error,
        maximum_nested_transfer_defect=nested_defect,
        exact_near=True,
        global_dense_materialized=False,
        evidence_id=evidence_id,
    )
    provider_id = canonical_fingerprint(
        {
            "kind": "prepared-scalar-fast-provider-3d",
            "algorithm": algorithm,
            "operator": prepared.weak_operator.operator_id,
            "formulation": selected.formulation,
            "policy": selected.policy_id,
            "evidence": evidence_id,
        }
    )
    return PreparedScalarFastProvider3D(
        clusters=clusters,
        h_blocks=tuple(h_blocks),
        h2_couplings=tuple(h2_couplings),
        near_blocks=near_blocks,
        evidence=evidence,
        envelope=envelope,
        algorithm=algorithm,
        formulation=selected.formulation,
        face_count=prepared.face_count,
        provider_id=provider_id,
    )


def prepare_scalar_fmm_provider_3d(
    prepared: LaplaceSingleLayerDP0Galerkin3D,
    policy: ScalarFastPolicy3D | None = None,
    /,
) -> PreparedScalarFastProvider3D:
    """Prepare a bounded kernel-independent multilevel Laplace FMM action."""

    return _prepare_scalar_fast_provider_3d(prepared, "fmm", policy)


def prepare_scalar_h_matrix_provider_3d(
    prepared: LaplaceSingleLayerDP0Galerkin3D,
    policy: ScalarFastPolicy3D | None = None,
    /,
) -> PreparedScalarFastProvider3D:
    """Prepare a bounded adaptive low-rank H-matrix action."""

    return _prepare_scalar_fast_provider_3d(prepared, "h-matrix", policy)


def prepare_scalar_h2_matrix_provider_3d(
    prepared: LaplaceSingleLayerDP0Galerkin3D,
    policy: ScalarFastPolicy3D | None = None,
    /,
) -> PreparedScalarFastProvider3D:
    """Prepare a bounded nested-basis H² action."""

    return _prepare_scalar_fast_provider_3d(prepared, "h2-matrix", policy)


__all__ = [
    "PreparedScalarFastProvider3D",
    "ScalarFastAlgorithm3D",
    "ScalarFastEvidence3D",
    "ScalarFastPolicy3D",
    "prepare_scalar_fmm_provider_3d",
    "prepare_scalar_h_matrix_provider_3d",
    "prepare_scalar_h2_matrix_provider_3d",
]
