#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared conservative averaging between bulk P1 fields and embedded sites."""

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
from ..spatial_sampling import (
    DG0ObservationPlan,
    P1ObservationPlan,
    PreparedObservationOperator,
)
from ._cell_complex import TetrahedralConnectivity
from ._cell_mesh import CellMesh


class EmbeddedKernelKind(StrEnum):
    CIRCLE = "circle"
    BALL = "ball"


class EmbeddedSourceAssociation(StrEnum):
    VERTEX = "vertex"
    CELL = "cell"


@dataclass(frozen=True, slots=True)
class CircleAverageKernel:
    radius: float
    angular_points: int = 12

    def __post_init__(self) -> None:
        radius = float(self.radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("Circle radius must be finite and positive.")
        if self.angular_points < 4:
            raise ValueError("Circle averages require at least four angular points.")
        object.__setattr__(self, "radius", radius)


@dataclass(frozen=True, slots=True)
class BallAverageKernel:
    radius: float

    def __post_init__(self) -> None:
        radius = float(self.radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("Ball radius must be finite and positive.")
        object.__setattr__(self, "radius", radius)


EmbeddedKernel = CircleAverageKernel | BallAverageKernel


class EmbeddedTransferEvidence(StrictModule):
    support: Array
    coverage_fraction: Array
    constant_residual: Array
    dual_pairing_residual: Array
    weighted_adjoint_residual: Array
    finite: Array
    successful: Array


class EmbeddedTransferResult(StrictModule):
    values: Array
    evidence: EmbeddedTransferEvidence
    transfer_id: str = eqx.field(static=True)


class EmbeddedTransferPartition(StrictModule):
    target_owner: Array
    source_route_owner: Array
    partition_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_owner: ArrayLike,
        source_route_owner: ArrayLike,
        partition_count: int,
        /,
    ):
        target = np.asarray(target_owner)
        routes = np.asarray(source_route_owner)
        if not np.issubdtype(target.dtype, np.integer) or not np.issubdtype(
            routes.dtype, np.integer
        ):
            raise TypeError("Embedded transfer ownership arrays must be integers.")
        if target.ndim != 1 or routes.ndim != 2 or routes.shape[0] != target.shape[0]:
            raise ValueError("Route ownership must begin with the target-site count.")
        count = int(partition_count)
        if count < 1 or np.any(target < 0) or np.any(target >= count):
            raise ValueError("Target owners must lie in the partition range.")
        if np.any(routes < 0) or np.any(routes >= count):
            raise ValueError("Route owners must lie in the partition range.")
        self.target_owner = jnp.asarray(target, dtype=jnp.int32)
        self.source_route_owner = jnp.asarray(routes, dtype=jnp.int32)
        self.partition_count = count
        self.partition_id = canonical_fingerprint(
            {
                "kind": "embedded-transfer-partition",
                "target_owner": array_tree_fingerprint(target),
                "route_owner": array_tree_fingerprint(routes),
                "partition_count": count,
            }
        )


class PreparedEmbeddedMeasureTransfer(StrictModule):
    sampling: PreparedObservationOperator
    quadrature_weights: Array
    source_measures: Array
    target_measures: Array
    partition: EmbeddedTransferPartition | None
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    @property
    def target_count(self) -> int:
        return int(self.target_measures.shape[0])

    @property
    def quadrature_count(self) -> int:
        return int(self.quadrature_weights.shape[1])

    def average(self, source_values: ArrayLike, /) -> EmbeddedTransferResult:
        sampled = self.sampling.apply(source_values)
        values = sampled.values.reshape(
            (self.target_count, self.quadrature_count) + sampled.values.shape[2:]
        )
        weights = self.quadrature_weights.reshape(
            self.quadrature_weights.shape + (1,) * (values.ndim - 2)
        )
        averaged = jnp.sum(weights * values, axis=1)
        evidence = self.evidence(source_values, averaged, sampled.evidence.support)
        return EmbeddedTransferResult(averaged, evidence, self.transfer_id)

    def dual_pullback(self, target_dual: ArrayLike, /) -> Array:
        dual = jnp.asarray(target_dual)
        if dual.shape[:1] != (self.target_count,):
            raise ValueError("target_dual must begin with the target-site count.")
        weights = self.quadrature_weights.reshape(
            self.quadrature_weights.shape + (1,) * (dual.ndim - 1)
        )
        messages = weights * dual[:, None]
        return self.sampling.transpose(messages)

    def hilbert_adjoint(self, target_values: ArrayLike, /) -> Array:
        values = jnp.asarray(target_values)
        weighted = values * self.target_measures.reshape(
            self.target_measures.shape + (1,) * (values.ndim - 1)
        )
        pulled = self.dual_pullback(weighted)
        return pulled / self.source_measures.reshape(
            self.source_measures.shape + (1,) * (pulled.ndim - 1)
        )

    def lower_distributed(
        self,
        source_owner: ArrayLike,
        target_owner: ArrayLike,
        partition_count: int,
        /,
    ) -> PreparedEmbeddedMeasureTransfer:
        owners = np.asarray(source_owner)
        if owners.shape != self.source_measures.shape or not np.issubdtype(
            owners.dtype, np.integer
        ):
            raise ValueError(
                "source_owner must contain one integer owner per source row."
            )
        indices = np.asarray(self.sampling.stencil.relation.source_indices)
        route_owner = owners[indices.reshape((self.target_count, -1))]
        partition = EmbeddedTransferPartition(target_owner, route_owner, partition_count)
        return PreparedEmbeddedMeasureTransfer(
            self.sampling,
            self.quadrature_weights,
            self.source_measures,
            self.target_measures,
            partition,
            self.source_geometry_id,
            self.target_geometry_id,
            canonical_fingerprint(
                {
                    "kind": "distributed-embedded-transfer",
                    "source": self.transfer_id,
                    "partition": partition.partition_id,
                }
            ),
        )

    def evidence(
        self,
        source_values: ArrayLike,
        target_values: ArrayLike,
        support: ArrayLike | None = None,
        /,
    ) -> EmbeddedTransferEvidence:
        source = jnp.asarray(source_values)
        target = jnp.asarray(target_values)
        if source.shape[:1] != self.source_measures.shape:
            raise ValueError("source_values must begin with the source row count.")
        support_ = (
            jnp.ones((self.target_count, self.quadrature_count), dtype=bool)
            if support is None
            else jnp.asarray(support, dtype=bool).reshape(
                (self.target_count, self.quadrature_count)
            )
        )
        coverage = jnp.mean(support_, axis=1)
        constant = self.average_constant()
        constant_residual = jnp.max(jnp.abs(constant - 1.0))
        source_probe = (
            jnp.arange(self.source_measures.shape[0], dtype=target.real.dtype) + 1.0
        )
        target_probe = jnp.arange(self.target_count, dtype=target.real.dtype) + 1.0
        primal = self._average_scalar(source_probe)
        left = jnp.vdot(primal, target_probe)
        right = jnp.vdot(source_probe, self.dual_pullback(target_probe))
        dual_residual = jnp.abs(left - right) / jnp.maximum(
            1.0, jnp.maximum(jnp.abs(left), jnp.abs(right))
        )
        weighted_left = jnp.vdot(primal * self.target_measures, target_probe)
        weighted_right = jnp.vdot(
            source_probe * self.source_measures,
            self.hilbert_adjoint(target_probe),
        )
        adjoint_residual = jnp.abs(weighted_left - weighted_right) / jnp.maximum(
            1.0, jnp.maximum(jnp.abs(weighted_left), jnp.abs(weighted_right))
        )
        finite = jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(target))
        successful = (
            finite
            & jnp.all(coverage == 1.0)
            & (constant_residual <= 1.0e-10)
            & (dual_residual <= 1.0e-10)
            & (adjoint_residual <= 1.0e-10)
        )
        return EmbeddedTransferEvidence(
            support_,
            coverage,
            constant_residual,
            dual_residual,
            adjoint_residual,
            finite,
            successful,
        )

    def _average_scalar(self, source: Array) -> Array:
        sampled = self.sampling.apply(source).values.reshape(
            (self.target_count, self.quadrature_count)
        )
        return jnp.sum(self.quadrature_weights * sampled, axis=1)

    def average_constant(self) -> Array:
        return self._average_scalar(jnp.ones_like(self.source_measures))


@dataclass(frozen=True, slots=True)
class EmbeddedMeasureTransferPlan:
    source_mesh: CellMesh
    coordinate_contract: SpatialCoordinateContract
    target_points: np.ndarray
    target_measures: np.ndarray
    source_lumped_measures: np.ndarray
    target_geometry_id: str
    kernel: EmbeddedKernel
    source_association: EmbeddedSourceAssociation
    target_tangents: np.ndarray | None = None
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_mesh, CellMesh) or not isinstance(
            self.source_mesh.connectivity, TetrahedralConnectivity
        ):
            raise TypeError("Embedded transfer requires a tetrahedral source mesh.")
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        points = np.asarray(self.target_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("target_points must have finite shape (N, 3).")
        target_measures = np.asarray(self.target_measures, dtype=float)
        source_measures = np.asarray(self.source_lumped_measures, dtype=float)
        if target_measures.shape != (len(points),) or np.any(target_measures <= 0.0):
            raise ValueError(
                "target_measures must be positive with one value per target."
            )
        if not isinstance(self.source_association, EmbeddedSourceAssociation):
            raise TypeError("source_association must be EmbeddedSourceAssociation.")
        expected_source_count = (
            len(self.source_mesh.coordinates)
            if self.source_association is EmbeddedSourceAssociation.VERTEX
            else sum(block.cell_count for block in self.source_mesh.blocks)
        )
        if (
            source_measures.ndim != 1
            or len(source_measures) != expected_source_count
            or np.any(source_measures <= 0.0)
        ):
            raise ValueError(
                "source_lumped_measures must match the declared source association."
            )
        if not isinstance(self.kernel, (CircleAverageKernel, BallAverageKernel)):
            raise TypeError("kernel must be CircleAverageKernel or BallAverageKernel.")
        tangents = None
        if isinstance(self.kernel, CircleAverageKernel):
            if self.target_tangents is None:
                raise ValueError("Circle averages require target_tangents.")
            tangents = np.asarray(self.target_tangents, dtype=float)
            if tangents.shape != points.shape or not np.all(np.isfinite(tangents)):
                raise ValueError("target_tangents must match target_points.")
            norms = np.linalg.norm(tangents, axis=1)
            if np.any(norms <= 0.0):
                raise ValueError("Circle tangents must be nonzero.")
            tangents = tangents / norms[:, None]
        for name, value in (
            ("target_points", points),
            ("target_measures", target_measures),
            ("source_lumped_measures", source_measures),
            ("target_tangents", tangents),
        ):
            if value is not None:
                value = np.array(value, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "embedded-measure-transfer-plan",
                    "source": self.source_mesh.mesh_id,
                    "spatial": self.coordinate_contract.spatial_id,
                    "target": str(self.target_geometry_id),
                    "source_association": self.source_association.value,
                    "points": array_tree_fingerprint(points),
                    "kernel": (
                        {
                            "kind": "circle",
                            "radius": self.kernel.radius,
                            "points": self.kernel.angular_points,
                        }
                        if isinstance(self.kernel, CircleAverageKernel)
                        else {"kind": "ball", "radius": self.kernel.radius}
                    ),
                }
            ),
        )

    def prepare(self) -> PreparedEmbeddedMeasureTransfer:
        points, weights = _kernel_points(
            np.asarray(self.target_points), self.target_tangents, self.kernel
        )
        cells = np.concatenate(
            [np.asarray(block.vertices) for block in self.source_mesh.blocks]
        )
        if self.source_association is EmbeddedSourceAssociation.VERTEX:
            sampling = P1ObservationPlan(
                np.asarray(self.source_mesh.coordinates),
                cells,
                points,
                self.source_mesh.mesh_id,
                require_complete_coverage=True,
            ).prepare()
        else:
            sampling = DG0ObservationPlan(
                np.asarray(self.source_mesh.coordinates),
                cells,
                points,
                self.source_mesh.mesh_id,
            ).prepare()
        return PreparedEmbeddedMeasureTransfer(
            sampling,
            jnp.asarray(weights),
            jnp.asarray(self.source_lumped_measures),
            jnp.asarray(self.target_measures),
            None,
            self.source_mesh.geometry_id,
            str(self.target_geometry_id),
            self.plan_id,
        )


def _kernel_points(
    centers: np.ndarray,
    tangents: np.ndarray | None,
    kernel: EmbeddedKernel,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(kernel, BallAverageKernel):
        directions = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        )
        radius = kernel.radius * np.sqrt(3.0 / 5.0)
        points = centers[:, None, :] + radius * directions[None, :, :]
        weights = np.full((len(centers), len(directions)), 1.0 / len(directions))
        return points, weights
    if tangents is None:
        raise ValueError("Circle kernel requires tangents.")
    reference = np.where(
        (np.abs(tangents[:, 2]) < 0.9)[:, None],
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((0.0, 1.0, 0.0)),
    )
    first = np.cross(tangents, reference)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(tangents, first)
    angles = 2.0 * np.pi * np.arange(kernel.angular_points) / kernel.angular_points
    offsets = kernel.radius * (
        first[:, None, :] * np.cos(angles)[None, :, None]
        + second[:, None, :] * np.sin(angles)[None, :, None]
    )
    points = centers[:, None, :] + offsets
    weights = np.full((len(centers), kernel.angular_points), 1.0 / kernel.angular_points)
    return points, weights


__all__ = [
    "BallAverageKernel",
    "CircleAverageKernel",
    "EmbeddedKernelKind",
    "EmbeddedSourceAssociation",
    "EmbeddedMeasureTransferPlan",
    "EmbeddedTransferEvidence",
    "EmbeddedTransferPartition",
    "EmbeddedTransferResult",
    "PreparedEmbeddedMeasureTransfer",
]
