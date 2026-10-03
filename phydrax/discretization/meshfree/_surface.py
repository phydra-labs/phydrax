# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Intrinsic fixed-support operators on declared smooth curves and sheets.

Closed sources need no boundary. Open sources use one-sided physical supports
and a declared ``SurfaceBoundary`` quadrature; sharp seams are separate smooth
patches joined by oriented ``SurfacePatchInterface`` data, never by averaging
normals across a crease.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from itertools import product
from typing import assert_never, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._bvh import BVHBuildPolicy, PackedBVH, point_select_leaf_items, prepare_bvh
from ..._differentiation import DerivativeRegularity
from ..._fingerprint import canonical_fingerprint
from ..._identity import execution_metadata_payload, NumericRevision, SemanticProvenance
from ..._interpolation import GatherStencil
from ..._model._ports import ValuePort
from ..._strict import StrictModule
from ...ein import contract
from ...geometry._contracts import CompiledGeometry
from ...linalg import (
    ArraySpace,
    inverse_small_linear,
    LinearSystem,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ...metrix._ambient import RegularLevelSetManifold
from ...sparse import (
    EdgeRelation,
    linear_apply,
    linear_transpose_apply,
    RowRelation,
    SparseCoordinateOperator,
)
from ...typing import Bool, Dim, Float, Int32, parse, Scalar
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
    fit_chart_stencils,
    LocalStencilEvidence,
    LocalStencilPolicy,
    weighted_svd_factors,
)
from ._surface_geometry import (
    _oriented_frames,
    ChartSurfaceGeometry,
    ImplicitSurfaceGeometry,
    SampledSurfaceGeometry,
    SurfaceAmbientDim,
    SurfaceBoundary,
    SurfaceGeometry,
    SurfaceGeometryEvaluation,
    SurfaceGeometryEvidence,
    SurfaceGeometryStatus,
    SurfaceIntrinsicDim,
    SurfaceNodeDim,
)
from ._surface_quadrature import SurfaceQuadratureEvidence, SurfaceQuadraturePolicy


if TYPE_CHECKING:
    from ._exterior import PreparedMeshfreeExteriorCalculus


class _SurfaceFeatureDim(Dim):
    pass


class SurfaceQueryDim(Dim):
    pass


class SurfaceFunctionalDim(Dim):
    pass


class SurfaceNeighborDim(Dim):
    pass


class SurfaceInterfaceDim(Dim):
    pass


class SurfaceSystemNodeDim(Dim):
    pass


SurfaceBoundaryKind: TypeAlias = Literal["dirichlet", "neumann"]


class SurfaceRowKind(IntEnum):
    INTERIOR = 0
    DIRICHLET = 1
    NEUMANN = 2
    INTERFACE_CONTINUITY = 3
    INTERFACE_FLUX = 4


def derivative_terms(intrinsic: int) -> tuple[tuple[int, ...], ...]:
    """Chart multi-indices: first derivatives, then ordered second derivatives."""
    first = tuple(tuple(int(i == k) for k in range(intrinsic)) for i in range(intrinsic))
    second = tuple(
        tuple(int(i == k) + int(j == k) for k in range(intrinsic))
        for i in range(intrinsic)
        for j in range(i, intrinsic)
    )
    return first + second


def _intrinsic_coefficients(geometry: SurfaceGeometryEvaluation) -> Array:
    """Graph-chart gradient, Laplace--Beltrami and covariant Hessian functionals.

    With ``x(u) = p + T u + N h(u)``, ``g = I + S^T S`` and Christoffel symbols
    ``G^k_ij = g^kl S_cl H_cij``. Rows: ambient gradient (A), Laplace--Beltrami
    (1), covariant Hessian in ambient tangential components (A*A).
    """
    slope, hessian = geometry.chart_slope, geometry.chart_hessian
    intrinsic, ambient = geometry.intrinsic_dimension, geometry.ambient_dimension
    dtype = slope.dtype
    basis = geometry.chart_frames + contract(
        "nac,ncd->nad", geometry.chart_normal_frames, slope
    )
    metric = jnp.eye(intrinsic, dtype=dtype) + contract("ncd,nce->nde", slope, slope)
    inverse = inverse_small_linear(SmallLinearSolvePlan(intrinsic), metric).value
    dual = contract("nai,nij->naj", basis, inverse)
    christoffel = contract("nkl,ncl,ncij->nkij", inverse, slope, hessian)
    drift = -contract("nij,nkij->nk", inverse, christoffel)
    terms = derivative_terms(intrinsic)
    pairs = [(i, j) for i in range(intrinsic) for j in range(i, intrinsic)]
    pair_i = np.asarray([pair[0] for pair in pairs])
    pair_j = np.asarray([pair[1] for pair in pairs])
    symmetry = jnp.asarray(np.where(pair_i == pair_j, 1.0, 2.0), dtype=dtype)
    laplace = jnp.concatenate(
        (drift, inverse[:, pair_i, pair_j] * symmetry[None, :]), axis=-1
    )
    hessian_first = -contract("nai,nbj,nkij->nabk", dual, dual, christoffel)
    hessian_second = dual[:, :, None, pair_i] * dual[:, None, :, pair_j] + jnp.where(
        jnp.asarray(pair_i != pair_j)[None, None, None, :],
        dual[:, :, None, pair_j] * dual[:, None, :, pair_i],
        0,
    )
    hessian_rows = jnp.concatenate((hessian_first, hessian_second), axis=-1).reshape(
        (-1, ambient * ambient, len(terms))
    )
    gradient = jnp.concatenate(
        (dual, jnp.zeros((dual.shape[0], ambient, len(pairs)), dtype=dtype)), axis=-1
    )
    return jnp.concatenate((gradient, laplace[:, None, :], hessian_rows), axis=1)


@final
class _SurfaceOperators(StrictModule):
    gradient: SparseCoordinateOperator
    divergence: SparseCoordinateOperator
    laplace: SparseCoordinateOperator
    strong_divergence: SparseCoordinateOperator
    hessian: SparseCoordinateOperator
    evidence: LocalStencilEvidence


def _self_slots(relation: RowRelation) -> Array:
    rows = jnp.arange(relation.source_indices.shape[0])[:, None]
    return relation.valid & (relation.source_indices == rows)


def _operators(
    geometry: SurfaceGeometryEvaluation,
    relation: RowRelation,
    measures: Array,
    policy: LocalStencilPolicy,
    identifier: str,
    boundary: SurfaceBoundary | None,
) -> _SurfaceOperators:
    offsets = geometry.points[relation.source_indices] - geometry.points[:, None, :]
    charts = contract("nka,nai->nki", offsets, geometry.chart_frames)
    count, ambient = geometry.points.shape
    # The ambient candidate envelope is selected by ambient distance; the
    # compact weight must use the same distance (a chart distance never exceeds
    # it and could weight a source outside the envelope).
    weights, evidence = fit_chart_stencils(
        charts,
        relation.valid,
        derivative_terms(geometry.intrinsic_dimension),
        _intrinsic_coefficients(geometry),
        policy,
        support_offsets=None if policy.support is None else offsets,
    )
    dtype = geometry.points.dtype
    scalar = ArraySpace((count,), dtype=dtype, space_id=f"{identifier}:scalar")
    vector = ArraySpace(
        (count, ambient), dtype=dtype, space_id=f"{identifier}:ambient-vector"
    )
    tensor = ArraySpace(
        (count, ambient, ambient), dtype=dtype, space_id=f"{identifier}:ambient-tensor"
    )
    gradient_coefficients = jnp.moveaxis(weights[:, :ambient, :], 1, 2)[..., None]
    gradient = SparseCoordinateOperator(
        relation,
        gradient_coefficients,
        source=scalar,
        target=vector,
        block_shape=(ambient, 1),
        operator_id=f"{identifier}:surface-gradient",
    )
    laplace = SparseCoordinateOperator(
        relation,
        weights[:, ambient, :],
        source=scalar,
        target=scalar,
        operator_id=f"{identifier}:laplace-beltrami",
    )
    hessian = SparseCoordinateOperator(
        relation,
        jnp.moveaxis(weights[:, ambient + 1 :, :], 1, 2)[..., None],
        source=scalar,
        target=tensor,
        block_shape=(ambient * ambient, 1),
        operator_id=f"{identifier}:covariant-hessian",
    )
    strong_divergence = SparseCoordinateOperator(
        relation,
        jnp.swapaxes(gradient_coefficients, -1, -2),
        source=vector,
        target=scalar,
        block_shape=(1, ambient),
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
        -jnp.swapaxes(gradient_coefficients.reshape((-1, ambient, 1)), -1, -2)
        * measures[edges.target_indices, None, None]
        / measures[edges.source_indices, None, None]
    )
    if boundary is not None:
        # Declared boundary quadrature closes the discrete Green identity:
        # sum m phi div v = -sum m grad(phi).v + sum_b phi_b (m nu)_b . v_b.
        flux = boundary.dense_conormals(count).astype(dtype)
        self_edge = edges.valid & (edges.source_indices == edges.target_indices)
        paired = paired + jnp.where(
            self_edge[:, None, None],
            flux[edges.target_indices, None, :]
            / measures[edges.target_indices, None, None],
            0,
        )
    divergence = SparseCoordinateOperator(
        transpose_relation,
        paired,
        source=vector,
        target=scalar,
        block_shape=(1, ambient),
        operator_id=f"{identifier}:paired-divergence",
    )
    return _SurfaceOperators(
        gradient=gradient,
        divergence=divergence,
        laplace=laplace,
        strong_divergence=strong_divergence,
        hessian=hessian,
        evidence=evidence,
    )


def _evaluate(
    geometry: SurfaceGeometry,
    points: Array,
    relation: RowRelation,
    *,
    reference: SurfaceGeometryEvaluation | None,
    reference_orientation: Array | None,
    require_tube: bool,
    boundary: SurfaceBoundary | None,
) -> SurfaceGeometryEvaluation:
    if isinstance(geometry, SampledSurfaceGeometry):
        return geometry.evaluate(
            points,
            relation,
            reference=reference,
            reference_orientation=reference_orientation,
            require_tube=require_tube,
            boundary=boundary,
        )
    return geometry.evaluate(
        points,
        relation,
        reference=reference,
        require_tube=require_tube,
        boundary=boundary,
    )


@final
class SurfacePointCloudPlan(StrictModule):
    """Host admission of a declared smooth curve/sheet and immutable support.

    Closed sources refuse a boundary; open sources require a declared
    ``SurfaceBoundary`` whose rows admit one-sided supports. Duplicate points,
    projection failures, orientation inconsistencies and deficient tangent fits
    are refused. Rebuilding support is a separate host operation.
    """

    __strict_contract__ = True
    points: Float[SurfaceNodeDim, SurfaceAmbientDim]
    geometry: SurfaceGeometry
    quadrature: SurfaceQuadraturePolicy
    stencil_policy: LocalStencilPolicy
    boundary: SurfaceBoundary | None
    neighbors: int = eqx.field(static=True)
    require_tube: bool = eqx.field(static=True)
    maximum_candidates: int | None = eqx.field(static=True)
    target_chunk_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        points: Array,
        geometry: SurfaceGeometry,
        neighbors: int = 24,
        *,
        quadrature: SurfaceQuadraturePolicy,
        stencil_policy: LocalStencilPolicy | None = None,
        boundary: SurfaceBoundary | None = None,
        require_tube: bool = False,
        maximum_candidates: int | None = None,
        target_chunk_size: int | None = None,
    ) -> None:
        if not isinstance(
            geometry,
            (ImplicitSurfaceGeometry, ChartSurfaceGeometry, SampledSurfaceGeometry),
        ) or not isinstance(quadrature, SurfaceQuadraturePolicy):
            raise TypeError("A surface geometry and quadrature policy are required.")
        cloud = np.asarray(points)
        if (
            cloud.ndim != 2
            or cloud.shape[1] != geometry.ambient_dimension
            or cloud.shape[0] < 6
            or not np.issubdtype(cloud.dtype, np.floating)
            or not np.all(np.isfinite(cloud))
        ):
            raise ValueError(
                "Surface points must be finite floating (N, declared ambient dimension), N>=6."
            )
        if np.unique(cloud, axis=0).shape[0] != cloud.shape[0]:
            raise ValueError("Degenerate duplicate surface points are refused.")
        if boundary is not None and not isinstance(boundary, SurfaceBoundary):
            raise TypeError("boundary must be a SurfaceBoundary.")
        if geometry.closed and boundary is not None:
            raise ValueError("A closed surface source refuses a declared boundary.")
        if not geometry.closed and boundary is None:
            raise ValueError(
                "Open surface sources require a declared SurfaceBoundary quadrature."
            )
        if boundary is not None and (
            boundary.weighted_conormals.shape[1] != cloud.shape[1]
            or int(np.max(np.asarray(boundary.nodes))) >= cloud.shape[0]
        ):
            raise ValueError("Boundary nodes/conormals must match the surface points.")
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
        self.boundary = boundary
        self.neighbors = neighbors_
        self.require_tube, self.maximum_candidates, self.target_chunk_size = (
            require_tube,
            candidates_,
            chunk_,
        )

    def _neighborhood(self, points: Array) -> PreparedMeshfreeNeighborhood:
        # A smooth fixed-radius stencil support owns the support epoch: the
        # relation holds every candidate of its envelope and the refresh trust
        # margin is the declared displacement envelope, not a selection gap.
        return MeshfreeNeighborhoodPlan(
            points,
            self.neighbors,
            maximum_candidates=self.maximum_candidates,
            target_chunk_size=self.target_chunk_size,
            envelope=self.stencil_policy.support,
        ).prepare()

    def prepare(self) -> PreparedSurfacePointCloud:
        initial = self._neighborhood(self.points)
        orientation = (
            self.geometry.prepare_reference(self.points, initial.relation)
            if isinstance(self.geometry, SampledSurfaceGeometry)
            else None
        )
        geometry = _evaluate(
            self.geometry,
            self.points,
            initial.relation,
            reference=None,
            reference_orientation=orientation,
            require_tube=self.require_tube,
            boundary=self.boundary,
        )
        if not np.all(np.asarray(geometry.evidence.valid)):
            raise ValueError(
                "Surface geometry admission failed; inspect projection, fit, orientation or tube declaration."
            )
        # Projected coordinates own the actual relation, not pre-projection points.
        neighborhood = self._neighborhood(geometry.points)
        geometry = _evaluate(
            self.geometry,
            geometry.points,
            neighborhood.relation,
            reference=geometry,
            reference_orientation=None,
            require_tube=self.require_tube,
            boundary=self.boundary,
        )
        if self.boundary is not None and not np.all(
            np.asarray(jnp.any(_self_slots(neighborhood.relation), axis=-1))[
                np.asarray(self.boundary.nodes)
            ]
        ):
            raise ValueError("Boundary rows must contain their own node in the support.")
        measures, quadrature_evidence = self.quadrature.prepare(
            geometry, neighborhood.relation
        )
        identifier = f"{self.geometry.geometry_id}:{neighborhood.neighborhood_id}"
        operators = _operators(
            geometry,
            neighborhood.relation,
            measures,
            self.stencil_policy,
            identifier,
            self.boundary,
        )
        if np.any(np.asarray(operators.evidence.status)) or not np.all(
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
            surface_gradient=operators.gradient,
            surface_divergence=operators.divergence,
            laplace_beltrami=operators.laplace,
            strong_surface_divergence=operators.strong_divergence,
            surface_hessian=operators.hessian,
            stencil_evidence=operators.evidence,
        )


def _source_provenance(geometry: SurfaceGeometry, dtype: jnp.dtype) -> dict[str, object]:
    provenance: dict[str, object] = {}
    if isinstance(geometry, ChartSurfaceGeometry):
        template = jax.ShapeDtypeStruct((geometry.intrinsic_dimension,), dtype)
        provenance["charts"] = tuple(
            {
                "chart": chart.chart.name,
                "coordinates": chart.chart.coordinates,
                "embedding_program": execution_metadata_payload(
                    jax.make_jaxpr(chart.embedding)(template), dynamic_captures=False
                ),
            }
            for chart in geometry.charts
        )
        provenance["orientation"] = geometry.orientation
        return provenance
    if not isinstance(geometry, ImplicitSurfaceGeometry):
        return provenance
    source = geometry.source
    point_template = jax.ShapeDtypeStruct((geometry.ambient_dimension,), dtype)
    if isinstance(source, CompiledGeometry):
        compiled_source = source

        def source_field(point: Array) -> Array:
            return compiled_source.boundary_field(point[None, :])

        projection_template = jax.eval_shape(
            compiled_source.closest_point,
            jax.ShapeDtypeStruct((1, geometry.ambient_dimension), dtype),
        )
        provenance["represented_geometry_id"] = (
            projection_template.represented_geometry_id
        )
        provenance["physical_geometry_id"] = projection_template.physical_geometry_id
        provenance["exact_to_physical"] = projection_template.exact_to_physical
    else:
        regular_source: RegularLevelSetManifold = source

        def source_field(point: Array) -> Array:
            return regular_source.constraint(point)

        def ambient_metric(point: Array) -> Array:
            return regular_source.ambient_metric(point)

        provenance["manifold_id"] = regular_source.manifold_id
        provenance["orientation"] = regular_source.orientation_sign
        provenance["codimension"] = regular_source.codimension
        provenance["metric_program"] = execution_metadata_payload(
            jax.make_jaxpr(ambient_metric)(point_template), dynamic_captures=False
        )
    provenance["field_program"] = execution_metadata_payload(
        jax.make_jaxpr(source_field)(point_template), dynamic_captures=False
    )
    return provenance


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
    surface_hessian: SparseCoordinateOperator
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
    def boundary(self) -> SurfaceBoundary | None:
        return self.plan.boundary

    @property
    def geometry_evidence(self) -> SurfaceGeometryEvidence:
        return self.geometry.evidence

    @property
    def conormal_derivative(self) -> SparseCoordinateOperator:
        """Outward conormal derivative at declared boundary nodes (scalar -> B)."""
        boundary = self.plan.boundary
        if boundary is None:
            raise ValueError("A closed surface has no conormal boundary derivative.")
        rows = boundary.nodes
        count = self.points.shape[0]
        dtype = self.points.dtype
        gradient = self.surface_gradient.coefficients[rows, :, :, 0]
        return SparseCoordinateOperator(
            RowRelation(
                self.relation.source_indices[rows],
                source_size=count,
                valid=self.relation.valid[rows],
            ),
            contract("ba,bka->bk", boundary.conormals.astype(dtype), gradient),
            source=self.surface_gradient.source,
            target=ArraySpace(
                (rows.shape[0],),
                dtype=dtype,
                space_id=f"{self.surface_gradient.operator_id}:{boundary.boundary_id}",
            ),
            operator_id=f"{self.surface_gradient.operator_id}:conormal-derivative",
        )

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
            self.surface_hessian,
        )
        geometry = self.plan.geometry
        boundary = self.plan.boundary
        semantic = SemanticProvenance(
            {
                "kind": "prepared-intrinsic-surface",
                "geometry_id": geometry.geometry_id,
                "geometry_representation": type(geometry).__qualname__,
                "intrinsic_dimension": geometry.intrinsic_dimension,
                "ambient_dimension": geometry.ambient_dimension,
                "closed": geometry.closed,
                "boundary_id": None if boundary is None else boundary.boundary_id,
                "source_provenance": _source_provenance(geometry, self.points.dtype),
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
                "geometry": geometry.geometry_id,
                "support": self.neighborhood.neighborhood_id,
            },
        )
        numeric: dict[str, object] = {
            "points": self.points,
            "measures": self.measures,
            "tangent_frames": self.geometry.tangent_frames,
            "normal_frames": self.geometry.normal_frames,
            "second_fundamental_form": self.geometry.second_fundamental_form,
            "chart_frames": self.geometry.chart_frames,
            "chart_normal_frames": self.geometry.chart_normal_frames,
            "chart_slope": self.geometry.chart_slope,
            "chart_hessian": self.geometry.chart_hessian,
            "relation_indices": self.relation.source_indices,
            "relation_valid": self.relation.valid,
            "operator_coefficients": tuple(
                operator.coefficients for operator in operators
            ),
        }
        if isinstance(geometry, ChartSurfaceGeometry):
            numeric["chart_coordinates"] = geometry.chart_coordinates
            numeric["chart_indices"] = geometry.chart_indices
        if boundary is not None:
            numeric["boundary"] = (
                boundary.nodes,
                boundary.measures,
                boundary.weighted_conormals,
            )
        return NumericRevision(semantic, numeric).revision_id

    def refresh(
        self,
        points: Array,
        *,
        geometry: SurfaceGeometry | None = None,
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
        evaluated = _evaluate(
            source,
            points,
            self.relation,
            reference=self.reference_geometry,
            reference_orientation=None,
            require_tube=self.plan.require_tube,
            boundary=self.plan.boundary,
        )
        displacement = jnp.max(
            jnp.linalg.norm(evaluated.points - self.reference_geometry.points, axis=-1)
        )
        # Support is certified only strictly inside the neighbor-gap trust margin.
        support_valid = (displacement < self.neighborhood.trust_margin) | (
            displacement == 0
        )
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
        intrinsic = evaluated.intrinsic_dimension
        jacobian = contract("nik,nka->nia", factors, current_offsets)
        reference_jacobian = contract("nik,nka->nia", factors, base_offsets)
        plan_small = SmallLinearSolvePlan(intrinsic)
        identity = jnp.broadcast_to(
            jnp.eye(intrinsic, dtype=points.dtype),
            (points.shape[0], intrinsic, intrinsic),
        )
        determinant = solve_small_linear(
            plan_small, contract("nia,nja->nij", jacobian, jacobian), identity
        ).determinant
        reference_determinant = solve_small_linear(
            plan_small,
            contract("nia,nja->nij", reference_jacobian, reference_jacobian),
            identity,
        ).determinant
        area_ratio = jnp.sqrt(
            jnp.maximum(determinant, 0)
            / jnp.maximum(reference_determinant, jnp.finfo(points.dtype).tiny)
        )
        measures = self.reference_measures * area_ratio
        measure_valid = (
            (rank == intrinsic)
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
        operators = _operators(
            evaluated,
            self.relation,
            safe_measures,
            self.plan.stencil_policy,
            identifier,
            self.plan.boundary,
        )
        valid = (
            evaluated.evidence.valid
            & support_valid
            & measure_valid
            & (operators.evidence.status == 0)
        )
        status = jnp.where(
            ~support_valid,
            int(SurfaceGeometryStatus.SUPPORT_INVALID),
            jnp.where(
                ~measure_valid,
                int(SurfaceGeometryStatus.DEGENERATE),
                jnp.where(
                    operators.evidence.status != 0,
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
        nan = jnp.asarray(jnp.nan, dtype=points.dtype)
        quadrature_evidence = SurfaceQuadratureEvidence(
            positive=jnp.all(measure_valid),
            total_area=jnp.sum(measures),
            minimum_measure=jnp.min(measures),
            maximum_measure=jnp.max(measures),
            reference_area=nan,
            transfer_residual=nan,
            normalized_to_declared_area=False,
            geometric_estimate=True,
            authoritative=False,
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
            surface_gradient=operators.gradient,
            surface_divergence=operators.divergence,
            laplace_beltrami=operators.laplace,
            strong_surface_divergence=operators.strong_divergence,
            surface_hessian=operators.hessian,
            stencil_evidence=operators.evidence,
        )
        return SurfaceRefreshResult(
            prepared=prepared, accepted=jnp.all(valid), status=status
        )

    def chart_derivatives(
        self,
        chart_coordinates: ArrayLike,
        multi_indices: tuple[tuple[int, ...], ...],
        *,
        chart: int = 0,
        neighbors: int | None = None,
    ) -> SurfaceChartDerivatives:
        """Intrinsic derivatives in a declared chart's coordinates at chart queries.

        Only an authoritative ``ChartSurfaceGeometry`` declares chart coordinates;
        the stencils fit ``f(x(u))`` in that chart's own coordinates using samples
        located in the same chart. Ambient volumetric derivatives stay refused.
        """
        geometry = self.plan.geometry
        if not isinstance(geometry, ChartSurfaceGeometry):
            raise ValueError(
                "Intrinsic chart derivatives require an authoritative declared chart source."
            )
        if not 0 <= chart < len(geometry.charts):
            raise ValueError("chart must select a declared chart.")
        policy = self.plan.stencil_policy
        dimension = geometry.intrinsic_dimension
        terms = tuple(tuple(int(value) for value in index) for index in multi_indices)
        if not terms or any(
            len(index) != dimension
            or any(value < 0 for value in index)
            or sum(index) > policy.polynomial_degree
            for index in terms
        ):
            raise ValueError(
                "Chart multi-indices must match the chart dimension and stencil degree."
            )
        queries = np.asarray(chart_coordinates)
        if (
            queries.ndim != 2
            or queries.shape[1] != dimension
            or not np.issubdtype(queries.dtype, np.floating)
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError("chart_coordinates must be finite floating (Q, intrinsic).")
        members = np.flatnonzero(np.asarray(geometry.chart_indices) == chart)
        count = (
            self.plan.neighbors if neighbors is None else _integer(neighbors, "neighbors")
        )
        if count > members.size:
            raise ValueError(
                "Chart support capacity exceeds samples located in the chart."
            )
        local = MeshfreeNeighborhoodPlan(
            np.asarray(geometry.chart_coordinates)[members], count, targets=queries
        ).prepare()
        dtype = self.points.dtype
        indices = jnp.asarray(members, dtype=jnp.int32)[local.relation.source_indices]
        offsets = (
            geometry.chart_coordinates.astype(dtype)[indices]
            - jnp.asarray(queries, dtype=dtype)[:, None, :]
        )
        coefficients = jnp.broadcast_to(
            jnp.eye(len(terms), dtype=dtype), (queries.shape[0], len(terms), len(terms))
        )
        weights, evidence = fit_chart_stencils(
            offsets, local.relation.valid, terms, coefficients, policy
        )
        return SurfaceChartDerivatives(
            relation=RowRelation(
                indices, source_size=self.points.shape[0], valid=local.relation.valid
            ),
            weights=weights,
            status=evidence.status.astype(jnp.int32),
            condition=evidence.condition,
            multi_indices=terms,
            chart=chart,
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
        Implicit geometry admits on-source queries; sampled and chart geometry
        admit coincident sample sites only (partial coverage, no invented
        projection). Chart derivatives use ``chart_derivatives`` instead.
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
            physical_dimension=self.points.shape[1],
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
        """Delegate intrinsic chart moment equations to the native exterior owner.

        Moment displacements are tangent-frame chart coordinates, not ambient
        coordinate displacements, so the default exact signed metric is the
        ``solver="lsmr"`` route; the multilevel Craig solver over coordinate
        vector-field jets does not apply here. Explicit signed-exact policies
        must declare ``solver="lsmr"``.
        """
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
            metric_policy=(
                MeshfreeMetricPolicy(solver="lsmr")
                if metric_policy is None
                else metric_policy
            ),
            intrinsic_displacements=charts,
        ).prepare(edge_relation=edge)


@final
class SurfaceChartDerivatives(StrictModule):
    """Gather stencils of declared-chart derivatives ``d^a (f o x)(u_q)``."""

    __strict_contract__ = True
    relation: RowRelation
    weights: Float[SurfaceQueryDim, SurfaceFunctionalDim, SurfaceNeighborDim]
    status: Int32[SurfaceQueryDim]
    condition: Float[SurfaceQueryDim]
    multi_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    chart: int = eqx.field(static=True)

    @property
    def valid(self) -> Array:
        return self.status == 0

    def apply(self, values: Array) -> Array:
        """Derivative values (queries, multi-indices); NaN where refused."""
        gathered = jnp.where(self.relation.valid, values[self.relation.source_indices], 0)
        result = contract("qfk,qk->qf", self.weights, gathered)
        return jnp.where(self.valid[:, None], result, jnp.nan)


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
class SurfacePatchInterface(StrictModule):
    """Oriented seam between two smooth patches with coincident node pairs.

    Each side keeps the outward conormal of its own declared boundary; the
    collocated transmission rows impose value continuity on the first side and
    conormal-flux balance on the second, without any normal averaging.
    """

    __strict_contract__ = True
    first_nodes: Int32[SurfaceInterfaceDim]
    second_nodes: Int32[SurfaceInterfaceDim]
    first_patch: int = eqx.field(static=True)
    second_patch: int = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        first_patch: int,
        first_nodes: ArrayLike,
        second_patch: int,
        second_nodes: ArrayLike,
        *,
        interface_id: str = "surface-interface",
    ) -> None:
        first = np.asarray(first_nodes)
        second = np.asarray(second_nodes)
        if (
            first.ndim != 1
            or first.shape != second.shape
            or first.size == 0
            or not np.issubdtype(first.dtype, np.integer)
            or not np.issubdtype(second.dtype, np.integer)
            or np.unique(first).size != first.size
            or np.unique(second).size != second.size
        ):
            raise ValueError(
                "Interface node pairs must be matching unique integer vectors."
            )
        first_patch_ = _integer(first_patch, "first_patch", 0)
        second_patch_ = _integer(second_patch, "second_patch", 0)
        if first_patch_ == second_patch_:
            raise ValueError("An interface joins two distinct patches.")
        if not isinstance(interface_id, str) or not interface_id:
            raise ValueError("interface_id must be nonempty.")
        self.first_nodes = jnp.asarray(first, dtype=jnp.int32)
        self.second_nodes = jnp.asarray(second, dtype=jnp.int32)
        self.first_patch, self.second_patch = first_patch_, second_patch_
        self.interface_id = interface_id


def _boundary_rows(patch: PreparedSurfacePointCloud) -> dict[int, int]:
    boundary = patch.plan.boundary
    if boundary is None:
        return {}
    return {int(node): row for row, node in enumerate(np.asarray(boundary.nodes))}


@final
class SurfaceEllipticSystem(StrictModule):
    """Collocated ``-k Lap_G u + r u = f`` on one or more oriented smooth patches.

    Open-boundary rows impose Dirichlet values or the conormal flux ``k du/dnu``;
    interface rows impose continuity and flux balance. The global operator is a
    native sparse operator for native Krylov solves.
    """

    __strict_contract__ = True
    patches: tuple[PreparedSurfacePointCloud, ...]
    interfaces: tuple[SurfacePatchInterface, ...]
    operator: SparseCoordinateOperator
    row_kind: Int32[SurfaceSystemNodeDim]
    diffusivity: float = eqx.field(static=True)
    reaction: float = eqx.field(static=True)
    boundary_kind: SurfaceBoundaryKind = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        patches: Sequence[PreparedSurfacePointCloud],
        interfaces: Sequence[SurfacePatchInterface] = (),
        *,
        diffusivity: float = 1.0,
        reaction: float = 0.0,
        boundary_kind: SurfaceBoundaryKind = "dirichlet",
        interface_tolerance: float = 1e-10,
        system_id: str = "surface-elliptic",
    ) -> None:
        patches_ = tuple(patches)
        interfaces_ = tuple(interfaces)
        if not patches_ or any(
            not isinstance(patch, PreparedSurfacePointCloud) for patch in patches_
        ):
            raise TypeError("patches must be PreparedSurfacePointCloud values.")
        if any(not isinstance(item, SurfacePatchInterface) for item in interfaces_):
            raise TypeError("interfaces must be SurfacePatchInterface values.")
        kind = parse(boundary_kind, SurfaceBoundaryKind, "boundary_kind")
        if not np.isfinite(diffusivity) or diffusivity <= 0:
            raise ValueError("diffusivity must be finite and positive.")
        if not np.isfinite(reaction) or reaction < 0:
            raise ValueError("reaction must be finite and nonnegative.")
        ambient = patches_[0].points.shape[1]
        dtype = patches_[0].points.dtype
        if any(
            patch.points.shape[1] != ambient or patch.points.dtype != dtype
            for patch in patches_
        ):
            raise ValueError("Patches must share one ambient space and dtype.")
        sizes = [patch.points.shape[0] for patch in patches_]
        offsets = tuple(
            int(value) for value in np.concatenate(([0], np.cumsum(sizes)[:-1]))
        )
        total = int(sum(sizes))
        capacity = max(patch.relation.source_indices.shape[1] for patch in patches_)
        boundary_rows = [_boundary_rows(patch) for patch in patches_]
        match kind:
            case "dirichlet":
                boundary_row = SurfaceRowKind.DIRICHLET
            case "neumann":
                boundary_row = SurfaceRowKind.NEUMANN
            case _:
                assert_never(kind)
        kinds = np.zeros((total,), dtype=np.int32)
        for patch_index, rows in enumerate(boundary_rows):
            for node in rows:
                kinds[offsets[patch_index] + node] = boundary_row
        partner_patch = np.full((total,), -1, dtype=np.int32)
        partner_node = np.full((total,), -1, dtype=np.int32)
        for interface in interfaces_:
            first_patch, second_patch = interface.first_patch, interface.second_patch
            if max(first_patch, second_patch) >= len(patches_):
                raise ValueError("Interface patches must exist.")
            pairs = zip(
                np.asarray(interface.first_nodes).tolist(),
                np.asarray(interface.second_nodes).tolist(),
                strict=True,
            )
            for first, second in pairs:
                if (
                    first not in boundary_rows[first_patch]
                    or second not in boundary_rows[second_patch]
                ):
                    raise ValueError(
                        "Interface nodes must be declared boundary nodes of their patch."
                    )
                first_row = offsets[first_patch] + first
                second_row = offsets[second_patch] + second
                if partner_patch[first_row] >= 0 or partner_patch[second_row] >= 0:
                    raise ValueError("An interface node may join only one seam pair.")
                gap = np.linalg.norm(
                    np.asarray(patches_[first_patch].points[first])
                    - np.asarray(patches_[second_patch].points[second])
                )
                if gap > interface_tolerance:
                    raise ValueError("Interface node pairs must be coincident.")
                kinds[first_row] = SurfaceRowKind.INTERFACE_CONTINUITY
                kinds[second_row] = SurfaceRowKind.INTERFACE_FLUX
                partner_patch[first_row], partner_node[first_row] = second_patch, second
                partner_patch[second_row], partner_node[second_row] = first_patch, first
        own_indices, own_valid, own_coefficients = [], [], []
        flux_rows = []
        for patch_index, patch in enumerate(patches_):
            relation = patch.relation
            pad = capacity - relation.source_indices.shape[1]
            indices = jnp.pad(relation.source_indices, ((0, 0), (0, pad)))
            valid = jnp.pad(relation.valid, ((0, 0), (0, pad)))
            laplace = jnp.pad(patch.laplace_beltrami.coefficients, ((0, 0), (0, pad)))
            gradient = jnp.pad(
                patch.surface_gradient.coefficients[..., 0], ((0, 0), (0, pad), (0, 0))
            )
            self_slot = (
                valid & (indices == jnp.arange(indices.shape[0])[:, None])
            ).astype(dtype)
            conormals = jnp.zeros((indices.shape[0], ambient), dtype=dtype)
            boundary = patch.plan.boundary
            if boundary is not None:
                conormals = conormals.at[boundary.nodes].set(
                    boundary.conormals.astype(dtype)
                )
            flux = diffusivity * contract("na,nka->nk", conormals, gradient)
            local_kind = jnp.asarray(
                kinds[offsets[patch_index] : offsets[patch_index] + indices.shape[0]]
            )
            interior = -diffusivity * laplace + reaction * self_slot
            coefficients = jnp.where(
                (local_kind == SurfaceRowKind.INTERIOR)[:, None],
                interior,
                jnp.where(
                    (
                        (local_kind == SurfaceRowKind.NEUMANN)
                        | (local_kind == SurfaceRowKind.INTERFACE_FLUX)
                    )[:, None],
                    flux,
                    self_slot,
                ),
            )
            own_indices.append(indices + offsets[patch_index])
            own_valid.append(valid)
            own_coefficients.append(coefficients)
            flux_rows.append((indices + offsets[patch_index], valid, flux))
        own_indices_ = jnp.concatenate(own_indices)
        own_valid_ = jnp.concatenate(own_valid)
        own_coefficients_ = jnp.concatenate(own_coefficients)
        partner_indices = jnp.zeros((total, capacity), dtype=own_indices_.dtype)
        partner_valid = jnp.zeros((total, capacity), dtype=jnp.bool_)
        partner_coefficients = jnp.zeros((total, capacity), dtype=dtype)
        continuity = np.flatnonzero(kinds == SurfaceRowKind.INTERFACE_CONTINUITY)
        if continuity.size:
            targets = np.asarray(
                [offsets[partner_patch[row]] + partner_node[row] for row in continuity]
            )
            partner_indices = partner_indices.at[continuity, 0].set(jnp.asarray(targets))
            partner_valid = partner_valid.at[continuity, 0].set(True)
            partner_coefficients = partner_coefficients.at[continuity, 0].set(-1.0)
        balance = np.flatnonzero(kinds == SurfaceRowKind.INTERFACE_FLUX)
        for row in balance:
            other_indices, other_valid, other_flux = flux_rows[partner_patch[row]]
            node = int(partner_node[row])
            partner_indices = partner_indices.at[row].set(other_indices[node])
            partner_valid = partner_valid.at[row].set(other_valid[node])
            partner_coefficients = partner_coefficients.at[row].set(other_flux[node])
        space = ArraySpace((total,), dtype=dtype, space_id=f"{system_id}:nodes")
        self.patches, self.interfaces = patches_, interfaces_
        self.operator = SparseCoordinateOperator(
            RowRelation(
                jnp.concatenate((own_indices_, partner_indices), axis=1),
                source_size=total,
                valid=jnp.concatenate((own_valid_, partner_valid), axis=1),
            ),
            jnp.concatenate((own_coefficients_, partner_coefficients), axis=1),
            source=space,
            target=space,
            operator_id=f"{system_id}:collocated-elliptic",
        )
        self.row_kind = jnp.asarray(kinds)
        self.diffusivity, self.reaction = float(diffusivity), float(reaction)
        self.boundary_kind, self.offsets = kind, offsets

    @property
    def linear_system(self) -> LinearSystem:
        return LinearSystem(self.operator)

    def split(self, values: Array) -> tuple[Array, ...]:
        sizes = [patch.points.shape[0] for patch in self.patches]
        return tuple(
            values[offset : offset + size]
            for offset, size in zip(self.offsets, sizes, strict=True)
        )

    def rhs(
        self,
        sources: Sequence[ArrayLike],
        boundary_values: Sequence[ArrayLike | None],
    ) -> Array:
        """Interior sources and per-node boundary data; seam rows are homogeneous."""
        if len(sources) != len(self.patches) or len(boundary_values) != len(self.patches):
            raise ValueError("Provide one source and boundary array per patch.")
        dtype = self.patches[0].points.dtype
        interior = jnp.concatenate([jnp.asarray(value, dtype=dtype) for value in sources])
        boundary = jnp.concatenate(
            [
                jnp.zeros((patch.points.shape[0],), dtype=dtype)
                if value is None
                else jnp.asarray(value, dtype=dtype)
                for patch, value in zip(self.patches, boundary_values, strict=True)
            ]
        )
        if interior.shape != self.row_kind.shape or boundary.shape != self.row_kind.shape:
            raise ValueError("Sources and boundary values must be per patch node.")
        return jnp.where(
            self.row_kind == SurfaceRowKind.INTERIOR,
            interior,
            jnp.where(
                (self.row_kind == SurfaceRowKind.DIRICHLET)
                | (self.row_kind == SurfaceRowKind.NEUMANN),
                boundary,
                0,
            ),
        )


@final
class SurfaceFieldReconstructionKernel(AbstractFieldReconstructionKernel):
    """Bounded intrinsic GMLS values with explicit on-surface admission."""

    __strict_contract__ = True
    surface: PreparedSurfacePointCloud
    bvh: PackedBVH
    exponents: Int32[_SurfaceFeatureDim, SurfaceIntrinsicDim]
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
        degree = surface.plan.stencil_policy.polynomial_degree
        exponents = tuple(
            index
            for index in product(
                range(degree + 1), repeat=surface.geometry.intrinsic_dimension
            )
            if sum(index) <= degree
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

    def _query_frames(
        self, query: Array, nearest: Array, lengths: Array, candidate_valid: Array
    ) -> tuple[Array, Array]:
        """Tangent frames at queries and on-source admission."""
        geometry = self.surface.plan.geometry
        intrinsic = self.surface.geometry.intrinsic_dimension
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
                normals = projection.oriented_normal.astype(query.dtype)
                normals /= jnp.maximum(
                    jnp.linalg.norm(normals, axis=-1, keepdims=True),
                    jnp.finfo(query.dtype).tiny,
                )
                projector = jnp.eye(query.shape[1], dtype=query.dtype) - (
                    normals[:, :, None] * normals[:, None, :]
                )
            else:
                local = source.local_geometry(query)
                residual = jnp.max(jnp.abs(jax.vmap(source.constraint)(query)), axis=-1)
                on_surface = local.valid & (residual <= self.tolerance)
                projector = local.tangent_projector.astype(query.dtype)
            frames, _, _, frame_ok = _oriented_frames(
                projector, intrinsic, projector[:, :, 0]
            )
            return frames, on_surface & frame_ok
        on_surface = (
            jnp.min(jnp.where(candidate_valid, lengths, jnp.inf), axis=-1)
            <= self.tolerance
        )
        return self.surface.geometry.tangent_frames[nearest], on_surface

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
        frames, on_surface = self._query_frames(query, nearest, lengths, candidate_valid)
        charts = contract("nka,nai->nki", offsets, frames) / self.radius
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
    "SurfaceBoundaryKind",
    "SurfaceChartDerivatives",
    "SurfaceEllipticSystem",
    "SurfacePatchInterface",
    "SurfaceRowKind",
    "SurfacePointCloudPlan",
    "PreparedSurfacePointCloud",
    "SurfaceRefreshResult",
    "SurfaceFieldReconstructionKernel",
]
