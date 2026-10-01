# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Intrinsic fixed-support surface operators on declared smooth closed sheets."""

from __future__ import annotations

from itertools import product
from typing import final, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._bvh import BVHBuildPolicy, PackedBVH, point_select_leaf_items, prepare_bvh
from ..._differentiation import DerivativeRegularity
from ..._fingerprint import canonical_fingerprint
from ..._identity import execution_metadata_payload, NumericRevision, SemanticProvenance
from ..._interpolation import GatherStencil
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ...ein import contract
from ...geometry._contracts import CompiledGeometry
from ...linalg import ArraySpace
from ...metrix._ambient import RegularLevelSetManifold
from ...sparse import (
    EdgeRelation,
    linear_apply,
    linear_transpose_apply,
    RowRelation,
    SparseCoordinateOperator,
)
from ...typing import Bool, Dim, Float, Int32, Scalar
from .._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldTracePolicy,
    FieldTraceSide,
    PreparedFieldReconstruction,
)
from ._exterior_metric import MeshfreeMetricPolicy
from ._neighbors import (
    _integer,
    MeshfreeEdgeRelationPlan,
    MeshfreeNeighborhoodPlan,
    PreparedMeshfreeNeighborhood,
)
from ._stencils import (
    chart_stencil_kernel,
    LocalStencilEvidence,
    LocalStencilPolicy,
    weighted_svd_factors,
)
from ._surface_geometry import (
    ImplicitSurfaceGeometry,
    SampledSurfaceGeometry,
    SurfaceGeometryEvaluation,
    SurfaceGeometryEvidence,
    SurfaceGeometryStatus,
    SurfaceNodeDim,
)
from ._surface_quadrature import SurfaceQuadratureEvidence, SurfaceQuadraturePolicy


if TYPE_CHECKING:
    from ._exterior import PreparedMeshfreeExteriorCalculus


class _SurfaceFeatureDim(Dim):
    pass


_DERIVATIVES = ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))


def _intrinsic_coefficients(geometry: SurfaceGeometryEvaluation) -> Array:
    """Graph-chart gradient and Laplace--Beltrami, including metric drift."""
    slope = geometry.chart_slope
    denominator = 1 + jnp.sum(slope * slope, axis=-1)
    inverse = (
        jnp.eye(2, dtype=slope.dtype)
        - slope[..., :, None] * slope[..., None, :] / denominator[:, None, None]
    )
    chart = (
        geometry.chart_frames + geometry.chart_normal[..., :, None] * slope[:, None, :]
    )
    gradient = contract("nai,nij->naj", chart, inverse)
    mean = contract("nij,nji->n", inverse, geometry.chart_hessian)
    drift = -contract("nij,nj->ni", inverse, slope) * mean[:, None] / denominator[:, None]
    coefficients = jnp.zeros((slope.shape[0], 4, 5), dtype=slope.dtype)
    coefficients = coefficients.at[:, :3, :2].set(gradient)
    return coefficients.at[:, 3, :].set(
        jnp.stack(
            (
                drift[:, 0],
                drift[:, 1],
                inverse[:, 0, 0],
                2 * inverse[:, 0, 1],
                inverse[:, 1, 1],
            ),
            axis=-1,
        )
    )


def _operators(
    geometry: SurfaceGeometryEvaluation,
    relation: RowRelation,
    measures: Array,
    policy: LocalStencilPolicy,
    identifier: str,
) -> tuple[
    SparseCoordinateOperator,
    SparseCoordinateOperator,
    SparseCoordinateOperator,
    SparseCoordinateOperator,
    LocalStencilEvidence,
]:
    offsets = geometry.points[relation.source_indices] - geometry.points[:, None, :]
    charts = contract("nka,nai->nki", offsets, geometry.chart_frames)
    count = charts.shape[0]
    chunk = min(policy.chunk_rows, count)
    padding = (-count) % chunk
    coefficients = _intrinsic_coefficients(geometry)

    def batches(value: Array) -> Array:
        padded = jnp.pad(value, ((0, padding),) + ((0, 0),) * (value.ndim - 1))
        return padded.reshape((-1, chunk) + value.shape[1:])

    def fit(batch: tuple[Array, Array, Array]) -> tuple[Array, LocalStencilEvidence]:
        coordinates, valid, coefficients_ = batch
        return chart_stencil_kernel(
            coordinates, valid, _DERIVATIVES, coefficients_, policy
        )

    chunk_weights, chunk_evidence = jax.lax.map(
        fit, (batches(charts), batches(relation.valid), batches(coefficients))
    )
    weights = chunk_weights.reshape((-1,) + chunk_weights.shape[2:])[:count]
    evidence = jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[2:])[:count], chunk_evidence
    )
    scalar = ArraySpace(
        (geometry.points.shape[0],),
        dtype=geometry.points.dtype,
        space_id=f"{identifier}:scalar",
    )
    vector = ArraySpace(
        tuple(geometry.points.shape),
        dtype=geometry.points.dtype,
        space_id=f"{identifier}:ambient-vector",
    )
    gradient_coefficients = jnp.moveaxis(weights[:, :3, :], 1, 2)[..., None]
    gradient = SparseCoordinateOperator(
        relation,
        gradient_coefficients,
        source=scalar,
        target=vector,
        block_shape=(3, 1),
        operator_id=f"{identifier}:surface-gradient",
    )
    laplace = SparseCoordinateOperator(
        relation,
        weights[:, 3, :],
        source=scalar,
        target=scalar,
        operator_id=f"{identifier}:laplace-beltrami",
    )
    strong_divergence = SparseCoordinateOperator(
        relation,
        jnp.swapaxes(gradient_coefficients, -1, -2),
        source=vector,
        target=scalar,
        block_shape=(1, 3),
        operator_id=f"{identifier}:strong-divergence",
    )
    edges = relation.as_edge_relation()
    transpose_relation = EdgeRelation(
        edges.target_indices,
        edges.source_indices,
        source_size=edges.target_size,
        target_size=edges.source_size,
        valid=edges.valid,
    )
    paired = (
        -jnp.swapaxes(gradient_coefficients.reshape((-1, 3, 1)), -1, -2)
        * measures[edges.target_indices, None, None]
        / measures[edges.source_indices, None, None]
    )
    divergence = SparseCoordinateOperator(
        transpose_relation,
        paired,
        source=vector,
        target=scalar,
        block_shape=(1, 3),
        operator_id=f"{identifier}:paired-divergence",
    )
    return gradient, divergence, laplace, strong_divergence, evidence


@final
class SurfacePointCloudPlan(StrictModule):
    """Host admission of smooth closed surface points and immutable support.

    Sharp/open sources, duplicate points, projection failures, orientation
    inconsistencies and deficient tangent fits are refused. Rebuilding support
    is a separate host operation; refresh never differentiates topology.
    """

    __strict_contract__ = True
    points: Float[SurfaceNodeDim, Literal[3]]
    geometry: ImplicitSurfaceGeometry | SampledSurfaceGeometry
    quadrature: SurfaceQuadraturePolicy
    stencil_policy: LocalStencilPolicy
    neighbors: int = eqx.field(static=True)
    require_tube: bool = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        points: Array,
        geometry: ImplicitSurfaceGeometry | SampledSurfaceGeometry,
        neighbors: int = 24,
        *,
        quadrature: SurfaceQuadraturePolicy,
        stencil_policy: LocalStencilPolicy | None = None,
        require_tube: bool = False,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
    ) -> None:
        cloud = np.asarray(points)
        if (
            cloud.ndim != 2
            or cloud.shape[1] != 3
            or cloud.shape[0] < 6
            or not np.issubdtype(cloud.dtype, np.floating)
            or not np.all(np.isfinite(cloud))
        ):
            raise ValueError("Surface points must be finite floating (N,3), N>=6.")
        if np.unique(cloud, axis=0).shape[0] != cloud.shape[0]:
            raise ValueError("Degenerate duplicate surface points are refused.")
        if not isinstance(
            geometry, (ImplicitSurfaceGeometry, SampledSurfaceGeometry)
        ) or not isinstance(quadrature, SurfaceQuadraturePolicy):
            raise TypeError("A surface geometry and quadrature policy are required.")
        policy = LocalStencilPolicy() if stencil_policy is None else stencil_policy
        if not isinstance(policy, LocalStencilPolicy) or policy.polynomial_degree < 2:
            raise ValueError(
                "Surface operators require a degree >=2 local stencil policy."
            )
        neighbors_ = _integer(neighbors, "neighbors", 6)
        if neighbors_ > cloud.shape[0]:
            raise ValueError("neighbors exceeds surface point capacity.")
        candidates_ = (
            None
            if maximum_candidates is None
            else _integer(maximum_candidates, "maximum_candidates")
        )
        chunk_ = (
            None
            if target_chunk_size is None
            else _integer(target_chunk_size, "target_chunk_size")
        )
        if (
            candidates_ is not None
            and not min(neighbors_ + 1, cloud.shape[0]) <= candidates_ <= cloud.shape[0]
        ):
            raise ValueError(
                "maximum_candidates must contain the gap witness and fit point capacity."
            )
        if not isinstance(require_tube, bool):
            raise TypeError("require_tube must be bool.")
        self.points, self.geometry, self.quadrature, self.stencil_policy = (
            jnp.asarray(cloud),
            geometry,
            quadrature,
            policy,
        )
        self.neighbors = neighbors_
        self.require_tube, self.maximum_candidates, self.target_chunk_size = (
            require_tube,
            candidates_,
            chunk_,
        )

    def prepare(self) -> PreparedSurfacePointCloud:
        initial = MeshfreeNeighborhoodPlan(
            self.points,
            self.neighbors,
            maximum_candidates=self.maximum_candidates,
            target_chunk_size=self.target_chunk_size,
        ).prepare()
        if isinstance(self.geometry, SampledSurfaceGeometry):
            normals = self.geometry.prepare_reference(self.points, initial.relation)
            geometry = self.geometry.evaluate(
                self.points,
                initial.relation,
                reference_normals=normals,
                require_tube=self.require_tube,
            )
        else:
            geometry = self.geometry.evaluate(
                self.points, initial.relation, require_tube=self.require_tube
            )
        if not np.all(np.asarray(geometry.evidence.valid)):
            raise ValueError(
                "Surface geometry admission failed; inspect projection, fit, orientation or tube declaration."
            )
        # Projected coordinates own the actual relation, not pre-projection points.
        neighborhood = MeshfreeNeighborhoodPlan(
            geometry.points,
            self.neighbors,
            maximum_candidates=self.maximum_candidates,
            target_chunk_size=self.target_chunk_size,
        ).prepare()
        if isinstance(self.geometry, SampledSurfaceGeometry):
            geometry = self.geometry.evaluate(
                geometry.points,
                neighborhood.relation,
                reference=geometry,
                require_tube=self.require_tube,
            )
        else:
            geometry = self.geometry.evaluate(
                geometry.points,
                neighborhood.relation,
                reference=geometry,
                require_tube=self.require_tube,
            )
        measures, quadrature_evidence = self.quadrature.prepare(
            geometry, neighborhood.relation
        )
        identifier = f"{self.geometry.geometry_id}:{neighborhood.neighborhood_id}"
        gradient, divergence, laplace, strong, evidence = _operators(
            geometry, neighborhood.relation, measures, self.stencil_policy, identifier
        )
        if np.any(np.asarray(evidence.status)) or not np.all(
            np.asarray(geometry.evidence.valid)
        ):
            raise ValueError(
                "Intrinsic surface stencils or geometry refused at admission."
            )
        return PreparedSurfacePointCloud(
            plan=self,
            neighborhood=neighborhood,
            geometry=geometry,
            reference_geometry=geometry,
            measures=measures,
            reference_measures=measures,
            quadrature_evidence=quadrature_evidence,
            surface_gradient=gradient,
            surface_divergence=divergence,
            laplace_beltrami=laplace,
            strong_surface_divergence=strong,
            stencil_evidence=evidence,
        )


@final
class PreparedSurfacePointCloud(StrictModule):
    __strict_contract__ = True
    plan: SurfacePointCloudPlan
    neighborhood: PreparedMeshfreeNeighborhood
    geometry: SurfaceGeometryEvaluation
    reference_geometry: SurfaceGeometryEvaluation
    measures: Float[SurfaceNodeDim]
    reference_measures: Float[SurfaceNodeDim]
    quadrature_evidence: SurfaceQuadratureEvidence
    surface_gradient: SparseCoordinateOperator
    surface_divergence: SparseCoordinateOperator
    laplace_beltrami: SparseCoordinateOperator
    strong_surface_divergence: SparseCoordinateOperator
    stencil_evidence: LocalStencilEvidence

    @property
    def points(self) -> Array:
        return self.geometry.points

    @property
    def normals(self) -> Array:
        return self.geometry.normals

    @property
    def relation(self) -> RowRelation:
        return self.neighborhood.relation

    @property
    def geometry_evidence(self) -> SurfaceGeometryEvidence:
        return self.geometry.evidence

    @property
    def prepared_id(self) -> str:
        """Host-only binding identity of this exact scientific numeric revision.

        Reconstruction/component preparation uses this boundary; device refresh
        never hashes or synchronizes. Diagnostic NaNs are not scientific state.
        """
        operators = (
            self.surface_gradient,
            self.surface_divergence,
            self.laplace_beltrami,
            self.strong_surface_divergence,
        )
        source_provenance: dict[str, object] = {}
        if isinstance(self.plan.geometry, ImplicitSurfaceGeometry):
            source = self.plan.geometry.source
            point_template = jax.ShapeDtypeStruct((3,), self.points.dtype)
            if isinstance(source, CompiledGeometry):
                compiled_source = source

                def source_field(point: Array) -> Array:
                    return compiled_source.boundary_field(point[None, :])

                projection_template = jax.eval_shape(
                    compiled_source.closest_point,
                    jax.ShapeDtypeStruct((1, 3), self.points.dtype),
                )
                source_provenance["represented_geometry_id"] = (
                    projection_template.represented_geometry_id
                )
                source_provenance["physical_geometry_id"] = (
                    projection_template.physical_geometry_id
                )
                source_provenance["exact_to_physical"] = (
                    projection_template.exact_to_physical
                )
            else:
                regular_source: RegularLevelSetManifold = source

                def source_field(point: Array) -> Array:
                    return regular_source.constraint(point)

                def ambient_metric(point: Array) -> Array:
                    return regular_source.ambient_metric(point)

                source_provenance["manifold_id"] = regular_source.manifold_id
                source_provenance["orientation"] = regular_source.orientation_sign
                source_provenance["codimension"] = regular_source.codimension
                source_provenance["metric_program"] = execution_metadata_payload(
                    jax.make_jaxpr(ambient_metric)(point_template), dynamic_captures=False
                )
            source_provenance["field_program"] = execution_metadata_payload(
                jax.make_jaxpr(source_field)(point_template), dynamic_captures=False
            )
        semantic = SemanticProvenance(
            {
                "kind": "prepared-intrinsic-surface",
                "geometry_id": self.plan.geometry.geometry_id,
                "geometry_representation": type(self.plan.geometry).__qualname__,
                "source_provenance": source_provenance,
                "neighborhood_id": self.neighborhood.neighborhood_id,
                "operator_ids": tuple(operator.operator_id for operator in operators),
                "source_space_ids": tuple(
                    operator.source.space_id for operator in operators
                ),
                "target_space_ids": tuple(
                    operator.target.space_id for operator in operators
                ),
                "approximation": self.plan.stencil_policy.approximation,
                "polynomial_degree": self.plan.stencil_policy.polynomial_degree,
                "phs_power": self.plan.stencil_policy.phs_power,
                "weight_kernel": self.plan.stencil_policy.weight_kernel,
            },
            resource_ids={
                "geometry": self.plan.geometry.geometry_id,
                "support": self.neighborhood.neighborhood_id,
            },
        )
        return NumericRevision(
            semantic,
            {
                "points": self.points,
                "measures": self.measures,
                "normals": self.normals,
                "curvature_tensor": self.geometry.curvature_tensor,
                "chart_frames": self.geometry.chart_frames,
                "chart_normal": self.geometry.chart_normal,
                "chart_slope": self.geometry.chart_slope,
                "chart_hessian": self.geometry.chart_hessian,
                "relation_indices": self.relation.source_indices,
                "relation_valid": self.relation.valid,
                "operator_coefficients": tuple(
                    operator.coefficients for operator in operators
                ),
            },
        ).revision_id

    def refresh(
        self,
        points: Array,
        *,
        geometry: ImplicitSurfaceGeometry | SampledSurfaceGeometry | None = None,
    ) -> SurfaceRefreshResult:
        """Differentiate fixed-support geometry and metric, never neighbor discovery.

        Surface measures follow the fitted local material area Jacobian relative
        to the admitted reference. This is an estimate for sample motion, not a
        replacement for exact source-area quadrature. Failure is returned, and
        callers must inspect ``accepted`` before consuming the candidate.
        ``geometry`` refreshes source parameters under the same scientific
        identity; omitting it retains the current source. Reference quadrature
        and the material area denominator stay anchored to the support epoch.
        """
        points = jnp.asarray(points, dtype=self.points.dtype)
        if points.shape != self.points.shape:
            raise ValueError("Surface refresh preserves fixed point capacity and shape.")
        source = self.plan.geometry if geometry is None else geometry
        if (
            not isinstance(source, type(self.plan.geometry))
            or source.geometry_id != self.plan.geometry.geometry_id
        ):
            raise ValueError(
                "Source refresh must preserve surface representation and scientific identity."
            )
        plan = (
            self.plan
            if geometry is None
            else eqx.tree_at(lambda item: item.geometry, self.plan, source)
        )
        evaluated = source.evaluate(
            points,
            self.relation,
            reference=self.reference_geometry,
            require_tube=self.plan.require_tube,
        )
        displacement = jnp.max(
            jnp.linalg.norm(evaluated.points - self.reference_geometry.points, axis=-1)
        )
        support_valid = displacement <= self.neighborhood.trust_margin
        base_offsets = (
            self.reference_geometry.points[self.relation.source_indices]
            - self.reference_geometry.points[:, None, :]
        )
        base_charts = contract(
            "nka,nai->nki", base_offsets, self.reference_geometry.tangent_frames
        )
        current_offsets = (
            evaluated.points[self.relation.source_indices] - evaluated.points[:, None, :]
        )
        factors, rank, condition, _ = weighted_svd_factors(
            base_charts,
            jnp.ones(self.relation.valid.shape, dtype=points.dtype),
            self.relation.valid,
        )
        jacobian = contract("nik,nka->nia", factors, current_offsets)
        gram = contract("nia,nja->nij", jacobian, jacobian)
        reference_jacobian = contract("nik,nka->nia", factors, base_offsets)
        reference_gram = contract("nia,nja->nij", reference_jacobian, reference_jacobian)
        determinant = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] * gram[:, 1, 0]
        reference_determinant = (
            reference_gram[:, 0, 0] * reference_gram[:, 1, 1]
            - reference_gram[:, 0, 1] * reference_gram[:, 1, 0]
        )
        area_ratio = jnp.sqrt(
            jnp.maximum(determinant, 0)
            / jnp.maximum(reference_determinant, jnp.finfo(points.dtype).tiny)
        )
        measures = self.reference_measures * area_ratio
        measure_valid = (
            (rank == 2)
            & jnp.isfinite(condition)
            & jnp.isfinite(measures)
            & (measures > 0)
            & (reference_determinant > 0)
        )
        # Invalid candidates retain finite safe measures for status construction;
        # status, not a substituted value, determines admission.
        safe_measures = jnp.where(measure_valid, measures, self.reference_measures)
        identifier = (
            f"{self.plan.geometry.geometry_id}:{self.neighborhood.neighborhood_id}"
        )
        gradient, divergence, laplace, strong, evidence = _operators(
            evaluated, self.relation, safe_measures, self.plan.stencil_policy, identifier
        )
        valid = (
            evaluated.evidence.valid
            & support_valid
            & measure_valid
            & (evidence.status == 0)
        )
        status = jnp.where(
            ~support_valid,
            int(SurfaceGeometryStatus.SUPPORT_INVALID),
            jnp.where(
                ~measure_valid,
                int(SurfaceGeometryStatus.DEGENERATE),
                jnp.where(
                    evidence.status != 0,
                    int(SurfaceGeometryStatus.DEGENERATE),
                    evaluated.evidence.status,
                ),
            ),
        ).astype(jnp.int32)
        geometry_evidence = eqx.tree_at(
            lambda x: (x.valid, x.status, x.trust_margin),
            evaluated.evidence,
            (
                valid,
                status,
                jnp.minimum(
                    evaluated.evidence.trust_margin,
                    self.neighborhood.trust_margin - displacement,
                ),
            ),
        )
        evaluated = eqx.tree_at(lambda x: x.evidence, evaluated, geometry_evidence)
        quadrature_evidence = SurfaceQuadratureEvidence(
            positive=jnp.all(measure_valid),
            total_area=jnp.sum(measures),
            minimum_measure=jnp.min(measures),
            maximum_measure=jnp.max(measures),
            normalized_to_declared_area=False,
            geometric_estimate=True,
            frozen_support=True,
        )
        prepared = PreparedSurfacePointCloud(
            plan=plan,
            neighborhood=self.neighborhood,
            geometry=evaluated,
            reference_geometry=self.reference_geometry,
            measures=safe_measures,
            reference_measures=self.reference_measures,
            quadrature_evidence=quadrature_evidence,
            surface_gradient=gradient,
            surface_divergence=divergence,
            laplace_beltrami=laplace,
            strong_surface_divergence=strong,
            stencil_evidence=evidence,
        )
        return SurfaceRefreshResult(
            prepared=prepared, accepted=jnp.all(valid), status=status
        )

    def prepare_field_reconstruction(
        self,
        *,
        support_geometry: CompiledGeometry,
        radius: float | None = None,
        maximum_candidates: int | None = None,
        tolerance: float = 1e-7,
    ) -> PreparedFieldReconstruction:
        """Native intrinsic values; region is only an admission envelope.

        This is not a volumetric extension and admits no ambient derivatives.
        Implicit geometry admits on-source queries; sampled geometry admits
        coincident sample sites only (partial coverage, no invented projection).
        """
        radius_ = (
            float(np.max(np.asarray(self.neighborhood.row_scale))) * 1.25
            if radius is None
            else radius
        )
        capacity = (
            min(self.points.shape[0], 2 * self.plan.neighbors + 16)
            if maximum_candidates is None
            else maximum_candidates
        )
        identity = f"{self.prepared_id}:surface-field"
        kernel = SurfaceFieldReconstructionKernel(
            self, radius_, capacity, tolerance, identity
        )
        return PreparedFieldReconstruction(
            kernel,
            support_geometry=support_geometry,
            value_port=ValuePort(
                "surface-field",
                event_shape=(),
                component_ids=("surface-field",),
                representation="surface-point-field",
                space_id=identity,
            ),
            regularity=DerivativeRegularity.piecewise_smooth(continuity=0),
            trace_policy=FieldTracePolicy("single-valued"),
            coefficient_shape=(self.points.shape[0],),
            physical_dimension=3,
            maximum_derivative_order=0,
            field_space_id=identity,
            support_id=f"{identity}:on-surface-partial-support",
            coefficient_dtype=self.points.dtype,
        )

    def conservative_exterior(
        self,
        radius: float,
        maximum_pairs: int,
        *,
        metric_policy: MeshfreeMetricPolicy | None = None,
    ) -> PreparedMeshfreeExteriorCalculus:
        """Delegate intrinsic chart moment equations to the native exterior owner."""
        from ._exterior import MeshfreeExteriorCalculusPlan

        edge = MeshfreeEdgeRelationPlan(self.points, radius, maximum_pairs).prepare()
        relation = edge.relation
        first, second = relation.source_indices, relation.target_indices
        offset = self.points[second] - self.points[first]
        charts = jnp.stack(
            (
                contract("ea,eai->ei", offset, self.geometry.tangent_frames[first]),
                contract("ea,eai->ei", -offset, self.geometry.tangent_frames[second]),
            ),
            axis=1,
        )
        return MeshfreeExteriorCalculusPlan(
            self.points,
            radius,
            maximum_pairs,
            node_volumes=self.measures,
            metric_policy=metric_policy,
            intrinsic_displacements=charts,
        ).prepare(edge_relation=edge)


@final
class SurfaceRefreshResult(StrictModule):
    __strict_contract__ = True
    prepared: PreparedSurfacePointCloud
    accepted: Bool[Scalar]
    status: Int32[SurfaceNodeDim]

    @property
    def points(self) -> Array:
        return self.prepared.points

    @property
    def measures(self) -> Array:
        return self.prepared.measures

    @property
    def normals(self) -> Array:
        return self.prepared.normals

    @property
    def geometry_evidence(self) -> SurfaceGeometryEvidence:
        return self.prepared.geometry_evidence


@final
class SurfaceFieldReconstructionKernel(AbstractFieldReconstructionKernel):
    """Bounded intrinsic GMLS values with explicit on-surface admission."""

    __strict_contract__ = True
    surface: PreparedSurfacePointCloud
    bvh: PackedBVH
    exponents: Int32[_SurfaceFeatureDim, Literal[2]]
    radius: float = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    source_owner_id: str = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface: PreparedSurfacePointCloud,
        radius: float,
        capacity: int,
        tolerance: float,
        identity: str,
    ) -> None:
        capacity_ = _integer(capacity, "maximum_candidates", 6)
        if (
            not np.isfinite(radius)
            or radius <= 0
            or not np.isfinite(tolerance)
            or tolerance <= 0
            or capacity_ > surface.points.shape[0]
            or not identity
        ):
            raise ValueError(
                "Invalid surface reconstruction radius/capacity/tolerance/identity."
            )
        exponents = tuple(
            index
            for index in product(
                range(surface.plan.stencil_policy.polynomial_degree + 1), repeat=2
            )
            if sum(index) <= surface.plan.stencil_policy.polynomial_degree
        )
        if capacity_ < len(exponents):
            raise ValueError(
                "Surface reconstruction capacity is below intrinsic feature count."
            )
        host = np.asarray(surface.points)
        bvh = prepare_bvh(
            host,
            host,
            policy=BVHBuildPolicy(leaf_size=min(8, host.shape[0])),
            dtype=surface.points.dtype,
        )
        source_owner_id = surface.prepared_id
        kernel_id = canonical_fingerprint(
            {
                "kind": "intrinsic-surface-field",
                "surface_id": identity,
                "source_owner_id": source_owner_id,
                "radius": radius,
                "capacity": capacity_,
                "tolerance": tolerance,
            }
        )
        self.surface, self.bvh, self.exponents = (
            surface,
            bvh,
            jnp.asarray(exponents, dtype=jnp.int32),
        )
        self.radius, self.capacity, self.tolerance, self._kernel_id = (
            radius,
            capacity_,
            tolerance,
            kernel_id,
        )
        self.source_owner_id = source_owner_id

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def cell_count(self) -> int:
        return 0

    @property
    def support_coverage(self) -> Literal["partial"]:
        return "partial"

    def locate(
        self, points: Array, derivative: tuple[int, ...], side: FieldSideBinding | None, /
    ) -> tuple[GatherStencil, FieldQueryEvidence]:
        del side
        if any(derivative):
            raise ValueError(
                "Intrinsic surface reconstruction provides values, not ambient derivatives."
            )
        finite = jnp.all(jnp.isfinite(points), axis=-1)
        query = jnp.where(finite[:, None], points, self.surface.points[0])
        indices, candidate_valid, complete = point_select_leaf_items(
            query, bvh=self.bvh, maximum_candidates=self.capacity, tolerance=self.radius
        )
        offsets = self.surface.points[indices] - query[:, None, :]
        lengths = jnp.linalg.norm(offsets, axis=-1)
        inside = candidate_valid & (lengths < self.radius)
        nearest_slot = jnp.argmin(jnp.where(candidate_valid, lengths, jnp.inf), axis=-1)
        nearest = jnp.take_along_axis(indices, nearest_slot[:, None], axis=1)[:, 0]
        geometry = self.surface.plan.geometry
        if isinstance(geometry, ImplicitSurfaceGeometry):
            source = geometry.source
            if isinstance(source, CompiledGeometry):
                projection = source.closest_point(query)
                on_surface = (
                    projection.unique
                    & projection.regular
                    & (
                        jnp.linalg.norm(projection.closest_point - query, axis=-1)
                        <= self.tolerance
                    )
                )
                normals = projection.oriented_normal
            else:
                local = source.local_geometry(query)
                normals = local.constraint_jacobian[:, 0, :] * source.orientation_sign

                def constraint(point: Array) -> Array:
                    return source.constraint(point)[0]

                residual = jnp.abs(jax.vmap(constraint)(query))
                on_surface = local.valid & (
                    residual <= self.tolerance * jnp.linalg.norm(normals, axis=-1)
                )
        else:
            normals = self.surface.normals[nearest]
            on_surface = (
                jnp.min(jnp.where(candidate_valid, lengths, jnp.inf), axis=-1)
                <= self.tolerance
            )
        normals /= jnp.maximum(
            jnp.linalg.norm(normals, axis=-1, keepdims=True), jnp.finfo(query.dtype).tiny
        )
        from ._surface_geometry import tangent_frames

        charts = contract("nka,nai->nki", offsets, tangent_frames(normals)) / self.radius
        design = jnp.prod(
            charts[:, :, None, :] ** self.exponents[None, None, :, :], axis=-1
        )
        ratio = jnp.minimum(lengths / self.radius, 1)
        weights = jnp.where(inside, (1 - ratio) ** 4 * (4 * ratio + 1), 0)
        factors, rank, condition, _ = weighted_svd_factors(design, weights, inside)
        count = jnp.sum(inside, axis=-1, dtype=jnp.int32)
        full_rank = rank == self.exponents.shape[0]
        admitted = (
            full_rank
            & jnp.isfinite(condition)
            & (condition <= self.surface.plan.stencil_policy.condition_limit)
        )
        status = jnp.where(
            ~finite,
            int(FieldQueryStatus.NONFINITE),
            jnp.where(
                ~complete,
                int(FieldQueryStatus.LOCATION_FAILED),
                jnp.where(
                    ~on_surface | (count < self.exponents.shape[0]),
                    int(FieldQueryStatus.OUTSIDE_SUPPORT),
                    jnp.where(
                        admitted,
                        int(FieldQueryStatus.VALID),
                        int(FieldQueryStatus.ILL_CONDITIONED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        valid = status == int(FieldQueryStatus.VALID)
        route = GatherStencil(
            indices=indices,
            weights=jnp.where(valid[:, None] & inside, factors[:, 0, :], 0),
            source_size=self.surface.points.shape[0],
            valid=inside,
            support=valid,
        )
        return route, FieldQueryEvidence(
            status, condition, count, kernel_id=self.kernel_id
        )

    def apply(self, route: GatherStencil, coefficients: Array, /) -> Array:
        return linear_apply(route.relation, route.weights, coefficients)

    def transpose(self, route: GatherStencil, cotangent: Array, /) -> Array:
        return linear_transpose_apply(route.relation, route.weights, cotangent)

    def bind_side(
        self, sites: np.ndarray, side: FieldTraceSide, cell_ids: np.ndarray | None, /
    ) -> tuple[np.ndarray | None, np.ndarray]:
        del sites, side, cell_ids
        raise ValueError(
            "Intrinsic single-valued surface reconstruction has no trace cells."
        )


__all__ = [
    "SurfacePointCloudPlan",
    "PreparedSurfacePointCloud",
    "SurfaceRefreshResult",
    "SurfaceFieldReconstructionKernel",
]
