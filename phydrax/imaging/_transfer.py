#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Quantity-aware sparse image/mesh transfers and conservation evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from ..discretization import CellMesh, TetrahedralConnectivity
from ..ein import contract
from ..linalg import (
    LinearSystem,
    OperatorProperties,
    prepare,
    PreparedLinearSolve,
    solve,
)
from ..sparse import EdgeRelation, SparseLinearMap
from ..spatial_sampling import PreparedObservationOperator, VoxelObservationPlan
from ._core import DiffusionTensorImage, ImageValueKind, LabelVolume, MedicalImageAsset


class CoveragePolicy(StrEnum):
    REJECT = "reject"
    MASK = "mask"
    RENORMALIZE = "renormalize"


class TensorInterpolationPolicy(StrEnum):
    EUCLIDEAN = "euclidean"
    LOG_EUCLIDEAN = "log_euclidean"


class TransferEvidence(StrictModule):
    covered: Array
    coverage_fraction: Array
    partition_residual: Array
    constant_residual: Array
    conservation_residual: Array
    adjoint_residual: Array
    finite: Array
    successful: Array


class ImageTransferResult(StrictModule):
    values: Array
    evidence: TransferEvidence
    transfer_id: str = eqx.field(static=True)


class ConservativeVoxelCellTransfer(StrictModule):
    """Sparse exact-overlap map from voxel averages to cell averages.

    Overlap measures are prepared by geometry-specific intersection code. This
    runtime owns the conservative algebra and refuses incomplete measure covers.
    """

    voxel_indices: Array
    cell_indices: Array
    overlap_measures: Array
    voxel_measures: Array
    cell_measures: Array
    voxel_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        voxel_indices: ArrayLike,
        cell_indices: ArrayLike,
        overlap_measures: ArrayLike,
        voxel_measures: ArrayLike,
        cell_measures: ArrayLike,
        /,
        *,
        source_id: str,
        target_id: str,
        measure_tolerance: float = 1.0e-10,
    ):
        voxel = np.asarray(voxel_indices)
        cell = np.asarray(cell_indices)
        overlap = np.asarray(overlap_measures, dtype=float)
        voxel_measure = np.asarray(voxel_measures, dtype=float)
        cell_measure = np.asarray(cell_measures, dtype=float)
        if not np.issubdtype(voxel.dtype, np.integer) or not np.issubdtype(
            cell.dtype, np.integer
        ):
            raise TypeError("Overlap indices must have integer dtype.")
        if voxel.ndim != 1 or cell.shape != voxel.shape or overlap.shape != voxel.shape:
            raise ValueError(
                "Overlap indices and measures must share one rank-one shape."
            )
        if voxel_measure.ndim != 1 or cell_measure.ndim != 1:
            raise ValueError("Entity measures must be rank-one arrays.")
        if np.any(voxel < 0) or np.any(voxel >= len(voxel_measure)):
            raise ValueError("A voxel overlap index is out of bounds.")
        if np.any(cell < 0) or np.any(cell >= len(cell_measure)):
            raise ValueError("A cell overlap index is out of bounds.")
        if np.any(~np.isfinite(overlap)) or np.any(overlap <= 0.0):
            raise ValueError("Overlap measures must be finite and positive.")
        if np.any(~np.isfinite(voxel_measure)) or np.any(voxel_measure <= 0.0):
            raise ValueError("Voxel measures must be finite and positive.")
        if np.any(~np.isfinite(cell_measure)) or np.any(cell_measure <= 0.0):
            raise ValueError("Cell measures must be finite and positive.")
        tolerance = float(measure_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("measure_tolerance must be finite and non-negative.")
        cell_cover = np.zeros_like(cell_measure)
        np.add.at(cell_cover, cell, overlap)
        scale = np.maximum(1.0, np.abs(cell_measure))
        if np.any(np.abs(cell_cover - cell_measure) > tolerance * scale):
            raise ValueError("Overlap measures do not cover every target cell exactly.")
        voxel_cover = np.zeros_like(voxel_measure)
        np.add.at(voxel_cover, voxel, overlap)
        if np.any(
            voxel_cover
            > voxel_measure + tolerance * np.maximum(1.0, np.abs(voxel_measure))
        ):
            raise ValueError("Overlap measures exceed a source voxel measure.")
        self.voxel_indices = jnp.asarray(voxel, dtype=jnp.int32)
        self.cell_indices = jnp.asarray(cell, dtype=jnp.int32)
        self.overlap_measures = jnp.asarray(overlap)
        self.voxel_measures = jnp.asarray(voxel_measure)
        self.cell_measures = jnp.asarray(cell_measure)
        self.voxel_count = len(voxel_measure)
        self.cell_count = len(cell_measure)
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "conservative-voxel-cell-transfer",
                "source": str(source_id),
                "target": str(target_id),
                "overlap": array_tree_fingerprint(
                    {"voxel": voxel, "cell": cell, "measure": overlap}
                ),
                "voxel_measures": array_tree_fingerprint(voxel_measure),
                "cell_measures": array_tree_fingerprint(cell_measure),
            }
        )

    def apply(self, voxel_values: ArrayLike, /) -> ImageTransferResult:
        values = jnp.asarray(voxel_values)
        if values.shape[:1] != (self.voxel_count,):
            raise ValueError("voxel_values must begin with the planned voxel count.")
        payload = values.shape[1:]
        weighted = values[self.voxel_indices] * self.overlap_measures.reshape(
            self.overlap_measures.shape + (1,) * len(payload)
        )
        total = (
            jnp.zeros((self.cell_count,) + payload, dtype=weighted.dtype)
            .at[self.cell_indices]
            .add(weighted)
        )
        cell = total / self.cell_measures.reshape(
            self.cell_measures.shape + (1,) * len(payload)
        )
        evidence = self.evidence(values, cell)
        return ImageTransferResult(cell, evidence, self.transfer_id)

    def dual_pullback(self, cell_dual: ArrayLike, /) -> Array:
        dual = jnp.asarray(cell_dual)
        if dual.shape[:1] != (self.cell_count,):
            raise ValueError("cell_dual must begin with the planned cell count.")
        payload = dual.shape[1:]
        messages = dual[self.cell_indices] * (
            self.overlap_measures / self.cell_measures[self.cell_indices]
        ).reshape(self.overlap_measures.shape + (1,) * len(payload))
        return (
            jnp.zeros((self.voxel_count,) + payload, dtype=messages.dtype)
            .at[self.voxel_indices]
            .add(messages)
        )

    def hilbert_adjoint(self, cell_values: ArrayLike, /) -> Array:
        values = jnp.asarray(cell_values)
        pulled = self.dual_pullback(
            values
            * self.cell_measures.reshape(
                self.cell_measures.shape + (1,) * (values.ndim - 1)
            )
        )
        return pulled / self.voxel_measures.reshape(
            self.voxel_measures.shape + (1,) * (values.ndim - 1)
        )

    def evidence(
        self, voxel_values: ArrayLike, cell_values: ArrayLike, /
    ) -> TransferEvidence:
        voxel = jnp.asarray(voxel_values)
        cell = jnp.asarray(cell_values)
        one = self.apply_constant()
        constant_residual = jnp.max(jnp.abs(one - 1.0))
        overlap_scale = self.overlap_measures.reshape(
            self.overlap_measures.shape + (1,) * (voxel.ndim - 1)
        )
        source_mass = jnp.sum(voxel[self.voxel_indices] * overlap_scale)
        target_mass = jnp.sum(
            cell
            * self.cell_measures.reshape(
                self.cell_measures.shape + (1,) * (cell.ndim - 1)
            )
        )
        scale = jnp.maximum(1.0, jnp.maximum(jnp.abs(source_mass), jnp.abs(target_mass)))
        conservation = jnp.abs(source_mass - target_mass) / scale
        probe = jnp.arange(self.cell_count, dtype=cell.real.dtype) + 1.0
        forward = self._apply_scalar(
            jnp.arange(self.voxel_count, dtype=cell.real.dtype) + 1.0
        )
        left = jnp.vdot(forward, probe)
        right = jnp.vdot(
            jnp.arange(self.voxel_count, dtype=cell.real.dtype) + 1.0,
            self.dual_pullback(probe),
        )
        adjoint = jnp.abs(left - right) / jnp.maximum(
            1.0, jnp.maximum(jnp.abs(left), jnp.abs(right))
        )
        finite = jnp.all(jnp.isfinite(cell))
        successful = (
            finite
            & (constant_residual <= 1.0e-10)
            & (conservation <= 1.0e-10)
            & (adjoint <= 1.0e-10)
        )
        return TransferEvidence(
            jnp.ones((self.cell_count,), dtype=bool),
            jnp.asarray(1.0),
            constant_residual,
            constant_residual,
            conservation,
            adjoint,
            finite,
            successful,
        )

    def _apply_scalar(self, values: Array) -> Array:
        weighted = values[self.voxel_indices] * self.overlap_measures
        return (
            jnp.zeros((self.cell_count,), dtype=weighted.dtype)
            .at[self.cell_indices]
            .add(weighted)
            / self.cell_measures
        )

    def apply_constant(self) -> Array:
        return self._apply_scalar(
            jnp.ones((self.voxel_count,), dtype=self.cell_measures.dtype)
        )


class PreparedImageToP1Projection(StrictModule):
    sampling: PreparedObservationOperator
    cell_vertices: Array
    basis: Array
    quadrature_measures: Array
    mass_solver: PreparedLinearSolve
    vertex_count: int = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def apply(
        self, image_values: ArrayLike, /, *, valid_mask: ArrayLike | None = None
    ) -> ImageTransferResult:
        sampled = self.sampling.apply(
            image_values, source_mask=valid_mask, mask_mode="strict"
        )
        values = sampled.values.reshape(self.cell_vertices.shape[0], self.basis.shape[0])
        local = contract("cq,qi,cq->ci", self.quadrature_measures, self.basis, values)
        right = (
            jnp.zeros((self.vertex_count,), dtype=local.dtype)
            .at[self.cell_vertices.reshape((-1,))]
            .add(local.reshape((-1,)))
        )
        solved = solve(self.mass_solver, right)
        result = solved.value
        finite = jnp.all(jnp.isfinite(result))
        successful = sampled.evidence.successful & jnp.all(solved.successful) & finite
        zero = jnp.asarray(0.0, dtype=result.real.dtype)
        evidence = TransferEvidence(
            sampled.evidence.support,
            sampled.evidence.coverage_fraction,
            zero,
            zero,
            zero,
            zero,
            finite,
            successful,
        )
        return ImageTransferResult(result, evidence, self.projection_id)


@dataclass(frozen=True, slots=True)
class ImageToP1ProjectionPlan:
    asset: MedicalImageAsset
    mesh: CellMesh
    coordinate_contract: SpatialCoordinateContract
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MedicalImageAsset):
            raise TypeError("asset must be MedicalImageAsset.")
        if (
            self.asset.layout.kind is not ImageValueKind.SCALAR
            or self.asset.time_axis is not None
        ):
            raise ValueError("P1 projection requires one static scalar image.")
        if not isinstance(self.mesh, CellMesh) or not isinstance(
            self.mesh.connectivity, TetrahedralConnectivity
        ):
            raise TypeError("P1 image projection requires a tetrahedral CellMesh.")
        if (
            self.asset.spatial_affine.coordinate_contract.spatial_id
            != self.coordinate_contract.spatial_id
        ):
            raise ValueError("Image and mesh coordinate contracts differ.")
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "image-to-p1-projection",
                    "image": self.asset.content_id,
                    "mesh": self.mesh.mesh_id,
                }
            ),
        )

    def prepare(self) -> PreparedImageToP1Projection:
        cells = np.concatenate([np.asarray(block.vertices) for block in self.mesh.blocks])
        coordinates = np.asarray(self.mesh.coordinates)
        if cells.shape[1] != 4:
            raise ValueError("P1 image projection supports tetrahedral blocks only.")
        if np.unique(cells).size != len(coordinates):
            raise ValueError(
                "P1 image projection requires every mesh vertex to belong to a cell."
            )
        a, b = 0.5854101966249685, 0.1381966011250105
        basis = np.asarray(((a, b, b, b), (b, a, b, b), (b, b, a, b), (b, b, b, a)))
        points = contract("qi,cid->cqd", basis, coordinates[cells])
        edges = coordinates[cells[:, 1:]] - coordinates[cells[:, :1]]
        volumes = np.abs(np.linalg.det(edges)) / 6.0
        if np.any(volumes <= 0.0):
            raise ValueError("P1 projection requires positive-volume tetrahedra.")
        quadrature = np.broadcast_to((volumes / 4.0)[:, None], (len(cells), 4)).copy()
        sampling = VoxelObservationPlan(
            self.asset.values.shape[:3],
            self.asset.spatial_affine,
            np.asarray(points),
            require_complete_coverage=True,
        ).prepare()
        local_mass = (np.ones((4, 4)) + np.eye(4)) / 20.0
        source_indices = np.tile(cells, (1, 4)).reshape((-1,))
        target_indices = np.repeat(cells, 4, axis=1).reshape((-1,))
        coefficients = (volumes[:, None, None] * local_mass[None]).reshape((-1,))
        relation = EdgeRelation(
            source_indices,
            target_indices,
            source_size=len(coordinates),
            target_size=len(coordinates),
        )
        operator = SparseLinearMap(
            relation,
            jnp.asarray(coefficients),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
            operator_id=f"{self.plan_id}:consistent-mass",
        )
        solver = prepare(LinearSystem(operator))
        return PreparedImageToP1Projection(
            sampling,
            jnp.asarray(cells, dtype=jnp.int32),
            jnp.asarray(basis),
            jnp.asarray(quadrature),
            solver,
            len(coordinates),
            self.plan_id,
        )


@dataclass(frozen=True, slots=True)
class TensorImageTransferPlan:
    image: DiffusionTensorImage
    query_points: np.ndarray
    target_frame_id: str
    reorientation: np.ndarray
    interpolation: TensorInterpolationPolicy = TensorInterpolationPolicy.LOG_EUCLIDEAN
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.image, DiffusionTensorImage):
            raise TypeError("image must be DiffusionTensorImage.")
        if self.image.asset.time_axis is not None:
            raise ValueError(
                "Tensor transfer currently requires one static tensor image."
            )
        if not isinstance(self.interpolation, TensorInterpolationPolicy):
            raise TypeError("interpolation must be TensorInterpolationPolicy.")
        target_frame = str(self.target_frame_id).strip()
        if not target_frame or target_frame != self.target_frame_id:
            raise ValueError("target_frame_id must be a canonical non-empty identifier.")
        points = np.asarray(self.query_points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("query_points must end in finite 3D coordinates.")
        rotations = np.asarray(self.reorientation, dtype=float)
        expected = points.shape[:-1] + (3, 3)
        if rotations.shape == (3, 3):
            rotations = np.broadcast_to(rotations, expected).copy()
        if rotations.shape != expected or not np.all(np.isfinite(rotations)):
            raise ValueError(f"reorientation must have shape {expected} or (3, 3).")
        u, _, vh = np.linalg.svd(rotations)
        orthogonal = u @ vh
        if np.any(np.linalg.det(orthogonal) <= 0.0):
            raise ValueError("Tensor reorientation must preserve orientation.")
        object.__setattr__(self, "query_points", points)
        object.__setattr__(self, "reorientation", orthogonal)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "tensor-image-transfer",
                    "image": self.image.tensor_image_id,
                    "points": array_tree_fingerprint(points),
                    "rotation": array_tree_fingerprint(orthogonal),
                    "target_frame": target_frame,
                    "interpolation": self.interpolation.value,
                }
            ),
        )

    def execute(self) -> ImageTransferResult:
        tensors = np.asarray(self.image.asset.values)
        valid = np.asarray(self.image.asset.valid_mask)
        safe_tensors = np.where(
            valid[..., None, None], tensors, np.eye(3, dtype=tensors.dtype)
        )
        if self.interpolation is TensorInterpolationPolicy.LOG_EUCLIDEAN:
            eigenvalues, eigenvectors = np.linalg.eigh(safe_tensors)
            floor = max(self.image.minimum_eigenvalue, np.finfo(tensors.dtype).tiny)
            if np.any(eigenvalues[valid] <= 0.0):
                raise ValueError(
                    "Log-Euclidean interpolation requires positive-definite tensors."
                )
            logs = contract(
                "...ik,...k,...jk->...ij",
                eigenvectors,
                np.log(np.maximum(eigenvalues, floor)),
                eigenvectors,
            )
            source = logs
        else:
            source = safe_tensors
        sampling = VoxelObservationPlan(
            source.shape[:3],
            self.image.asset.spatial_affine,
            self.query_points,
            require_complete_coverage=True,
        ).prepare()
        sampled = sampling.apply(source, source_mask=valid, mask_mode="strict")
        values = np.asarray(sampled.values)
        if self.interpolation is TensorInterpolationPolicy.LOG_EUCLIDEAN:
            eigenvalues, eigenvectors = np.linalg.eigh(values)
            values = contract(
                "...ik,...k,...jk->...ij", eigenvectors, np.exp(eigenvalues), eigenvectors
            )
        rotation = np.asarray(self.reorientation)
        rotated = contract("...ik,...kl,...jl->...ij", rotation, values, rotation)
        symmetric = 0.5 * (rotated + np.swapaxes(rotated, -1, -2))
        minimum = float(np.min(np.linalg.eigvalsh(symmetric)))
        finite = bool(np.all(np.isfinite(symmetric)))
        zero = jnp.asarray(0.0)
        evidence = TransferEvidence(
            sampled.evidence.support,
            sampled.evidence.coverage_fraction,
            zero,
            zero,
            zero,
            zero,
            jnp.asarray(finite),
            sampled.evidence.successful
            & finite
            & (minimum >= self.image.minimum_eigenvalue),
        )
        return ImageTransferResult(jnp.asarray(symmetric), evidence, self.plan_id)


@dataclass(frozen=True, slots=True)
class ProbabilityImageTransferPlan:
    asset: MedicalImageAsset
    query_points: np.ndarray
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MedicalImageAsset):
            raise TypeError("asset must be MedicalImageAsset.")
        if self.asset.layout.kind is not ImageValueKind.PROBABILITY:
            raise ValueError("Probability transfer requires a probability image asset.")
        points = np.asarray(self.query_points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("query_points must end in finite 3D coordinates.")
        object.__setattr__(self, "query_points", points)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "probability-image-transfer",
                    "asset": self.asset.content_id,
                    "points": array_tree_fingerprint(points),
                }
            ),
        )

    def execute(self) -> ImageTransferResult:
        sampling = VoxelObservationPlan(
            self.asset.values.shape[:3],
            self.asset.spatial_affine,
            self.query_points,
            require_complete_coverage=True,
        ).prepare()
        sampled = sampling.apply(
            self.asset.values,
            source_mask=self.asset.valid_mask,
            mask_mode="strict",
        )
        total = jnp.sum(sampled.values, axis=-1, keepdims=True)
        normalized = sampled.values / jnp.where(total > 0.0, total, 1.0)
        partition_residual = jnp.max(jnp.abs(jnp.sum(normalized, axis=-1) - 1.0))
        finite = jnp.all(jnp.isfinite(normalized))
        zero = jnp.asarray(0.0, dtype=normalized.dtype)
        evidence = TransferEvidence(
            sampled.evidence.support,
            sampled.evidence.coverage_fraction,
            partition_residual,
            zero,
            zero,
            zero,
            finite,
            sampled.evidence.successful
            & finite
            & (partition_residual <= 1.0e-10)
            & jnp.all(normalized >= 0.0),
        )
        return ImageTransferResult(normalized, evidence, self.plan_id)


@dataclass(frozen=True, slots=True)
class LabelImageTransferPlan:
    labels: LabelVolume
    query_points: np.ndarray
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.labels, LabelVolume):
            raise TypeError("labels must be LabelVolume.")
        points = np.asarray(self.query_points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("query_points must end in finite 3D coordinates.")
        object.__setattr__(self, "query_points", points)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "label-image-transfer",
                    "labels": self.labels.label_volume_id,
                    "points": array_tree_fingerprint(points),
                }
            ),
        )

    def execute(self) -> ImageTransferResult:
        coordinates = self.labels.asset.spatial_affine.world_to_index(self.query_points)
        rounded = np.rint(coordinates).astype(np.int64)
        shape = np.asarray(self.labels.asset.values.shape[:3])
        support = np.all((rounded >= 0) & (rounded < shape), axis=-1)
        clipped = np.clip(rounded, 0, shape - 1)
        values = np.asarray(self.labels.asset.values)[tuple(np.moveaxis(clipped, -1, 0))]
        valid = np.asarray(self.labels.asset.valid_mask)[
            tuple(np.moveaxis(clipped, -1, 0))
        ]
        covered = support & valid
        finite = np.all(np.isfinite(values[covered]))
        fraction = np.mean(covered) if covered.size else 0.0
        zero = jnp.asarray(0.0)
        evidence = TransferEvidence(
            jnp.asarray(covered),
            jnp.asarray(fraction),
            zero,
            zero,
            zero,
            zero,
            jnp.asarray(finite),
            jnp.asarray(finite and bool(np.all(covered))),
        )
        return ImageTransferResult(jnp.asarray(values), evidence, self.plan_id)


__all__ = [
    "ConservativeVoxelCellTransfer",
    "CoveragePolicy",
    "ImageToP1ProjectionPlan",
    "ImageTransferResult",
    "ProbabilityImageTransferPlan",
    "LabelImageTransferPlan",
    "PreparedImageToP1Projection",
    "TensorImageTransferPlan",
    "TensorInterpolationPolicy",
    "TransferEvidence",
]
