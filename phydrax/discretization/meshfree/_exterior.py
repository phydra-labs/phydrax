#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Meshfree moment preparation and binding to the native graph one-complex.

Signed edge coefficients are conservative constitutive stiffness data, never a
signed Hilbert pairing. Only admitted positive metrics enter native cochains.
"""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import assert_never, cast, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...exterior import ComplexBoundary
from ...graph import CochainComplexIR
from ...linalg import (
    ArraySpace,
    HilbertComplex,
    OperatorProperties,
    prepare_sparse_factorization,
    prepare_sparse_ordering,
    refresh_sparse_factorization_values,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationStatus,
    SparseOrderingPolicy,
)
from ...sparse import EdgeRelation, linear_apply, SparseCoordinateOperator
from ...sparse._linear import _SparseStoragePlan
from ...typing import Bool, Dim, Float64, Int32, parse, Scalar, Size
from .._cochain import CochainDiscretization
from .._cochain_hodge import DiagonalHodge
from .._topology import CellComplexTopology, EntitySet, OrientedIncidence
from ..spatial import MortonAddressPlan, MortonRadiusShellWitnessPlan
from ..spatial._neighbor_query import _minimum_image
from ._exterior_metric import (
    _boundary_moment_indices,
    _laplacian_moment_rhs,
    _moment_exponents,
    MeshfreeMetricGeometry,
    MeshfreeMetricPolicy,
    MeshfreeMetricResult,
    PreparedMeshfreeMetric,
)
from ._neighbors import MeshfreeEdgeRelationPlan, PreparedMeshfreeEdgeRelation


class _ExteriorNodeDim(Dim):
    """Compact active graph vertices."""


class _ExteriorEdgeDim(Dim):
    """Compact canonical graph edges."""


class _ExteriorCoordinateDim(Dim):
    """Ambient coordinates of meshfree vertices."""


class _ExteriorIntrinsicDim(Dim):
    """Intrinsic chart coordinates, distinct from ambient coordinates."""


class _ExteriorEdgeCapacityDim(Dim):
    """Prepared spatial pair capacity before edge compaction."""


class _ExteriorEquationDim(Dim):
    """Non-Dirichlet compact equation vertices."""


class _ExteriorMomentRouteDim(Dim):
    """Uncoalesced local moment coefficient routes."""


class _ExteriorReducedRouteDim(Dim):
    """Constitutive stiffness routes retained in the Dirichlet reduction."""


class _ExteriorMomentNodeDim(Dim):
    """Compact nodes carrying moment rows: interior, then boundary closure."""


class _ExteriorMomentRowDim(Dim):
    """Every moment row of the metric, grouped by moment node."""


class _ExteriorMomentDim(Dim):
    """Declared polynomial moments of one equation node, degree-graded."""


class _ExteriorMomentFactorDim(Dim):
    """Axis factors of one monomial, padded with the unit chart coordinate."""


EdgeCoefficientAverage: TypeAlias = Literal["harmonic", "arithmetic", "supplied"]
CoercivityAssessment: TypeAlias = Literal["sparse-cholesky", "unassessed"]


@final
class MeshfreeCoercivityPolicy(StrictModule):
    """Selected property assessment of Dirichlet-reduced edge stiffnesses.

    ``"sparse-cholesky"`` factors the reduced stiffness without shift under
    ``factorization``. Its fill-reducing ordering and symbolic-work, fill and
    byte limits are the one resource charge shared by every meshfree consumer
    assessing coercivity with this policy; an unaffordable factor is refused at
    preparation. ``"unassessed"`` factors nothing: SPD and inertia evidence are
    reported unavailable, while separately admitted solves may still converge.
    """

    assessment: CoercivityAssessment = eqx.field(static=True)
    factorization: SparseFactorizationPolicy

    def __init__(
        self,
        assessment: CoercivityAssessment = "sparse-cholesky",
        /,
        *,
        factorization: SparseFactorizationPolicy | None = None,
    ) -> None:
        assessment_ = parse(assessment, CoercivityAssessment, "assessment")
        factorization_ = (
            SparseFactorizationPolicy(
                "cholesky",
                ordering="approximate-minimum-degree",
                max_symbolic_work=2_000_000,
            )
            if factorization is None
            else factorization
        )
        if not isinstance(factorization_, SparseFactorizationPolicy):
            raise TypeError("factorization must be a SparseFactorizationPolicy.")
        if (
            factorization_.kind != "cholesky"
            or factorization_.fill_level is not None
            or factorization_.drop_tolerance != 0
            or factorization_.diagonal_shift != 0
            or factorization_.allow_pivot_replacement
        ):
            raise ValueError(
                "Coercivity evidence needs a complete no-shift Cholesky factor without pivot replacement."
            )
        self.assessment = assessment_
        self.factorization = factorization_

    def prepare(
        self, operator: SparseCoordinateOperator, coordinates: np.ndarray, /
    ) -> SparseFactorizationPlan | None:
        """Symbolic factor of one reduced pattern, or ``None`` when unassessed."""
        match self.assessment:
            case "unassessed":
                return None
            case "sparse-cholesky":
                pass
            case unknown:
                assert_never(unknown)
        method = self.factorization.ordering
        match method:
            case "nested-dissection":
                geometric: np.ndarray | None = coordinates
            case "natural" | "reverse-cuthill-mckee" | "approximate-minimum-degree":
                geometric = None
            case unknown:
                assert_never(unknown)
        ordering = prepare_sparse_ordering(
            operator, SparseOrderingPolicy(method), coordinates=geometric
        )
        return prepare_sparse_factorization(
            operator, self.factorization, ordering=ordering
        )

    @property
    def source(self) -> str:
        match self.assessment:
            case "unassessed":
                return "unassessed"
            case "sparse-cholesky":
                return f"native-sparse-cholesky:no-shift:{self.factorization.ordering}"
            case unknown:
                assert_never(unknown)


@final
class MeshfreeStiffnessEvidence(StrictModule):
    """Native no-shift sparse Cholesky evidence for the Dirichlet reduction.

    Successful Cholesky establishes inertia (n,0,0). Failure does not establish
    negative or zero counts: unavailable counts are -1, never fabricated. An
    unassessed policy reports no SPD, inertia or maximum-principle claim.
    """

    __strict_contract__ = True
    spd: Bool[Scalar]
    factorization_status: Int32[Scalar]
    minimum_pivot: Float64[Scalar]
    positive_inertia: Int32[Scalar]
    negative_inertia: Int32[Scalar]
    zero_inertia: Int32[Scalar]
    inertia_available: Bool[Scalar]
    anchored_components: Bool[Scalar]
    maximum_principle: Bool[Scalar]
    dimension: int = eqx.field(static=True)
    assessed: bool = eqx.field(static=True)
    source: str = eqx.field(static=True)


@final
class MeshfreeBoundaryQuadrature(StrictModule):
    """Declared nodal boundary quadrature; natural flux is never guessed."""

    __strict_contract__ = True
    weights: Float64[_ExteriorNodeDim]

    def __init__(self, weights: ArrayLike, /) -> None:
        values = jnp.asarray(weights, dtype=jnp.float64)
        if values.ndim != 1:
            raise ValueError("Boundary quadrature must be one nodal vector.")
        self.weights = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values) | (values < 0)),
            "Boundary quadrature must be finite and nonnegative.",
        )

    def load(self, outward_flux: ArrayLike, /) -> Array:
        flux = jnp.asarray(outward_flux, dtype=jnp.float64)
        if flux.shape not in ((), self.weights.shape):
            raise ValueError("Natural flux must be scalar or match boundary quadrature.")
        return -self.weights * flux


@final
class MeshfreeDiffusionOperator(StrictModule):
    """Conservative weak stiffness with native positive cochain actions."""

    __strict_contract__ = True
    operator: SparseCoordinateOperator
    conductances: Float64[_ExteriorEdgeDim]
    node_volumes: Float64[_ExteriorNodeDim]
    native: CochainDiscretization | None
    native_active: Bool[Scalar]
    admitted: Bool[Scalar]
    evidence: MeshfreeStiffnessEvidence

    def mv(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values, dtype=jnp.float64)
        if value.shape != self.node_volumes.shape:
            raise ValueError("Diffusion values must match compact active vertices.")
        value = eqx.error_if(
            value,
            ~self.admitted,
            "Meshfree metric provider did not admit this diffusion operator.",
        )
        native = self.native
        if native is not None:
            return jax.lax.cond(
                self.native_active,
                lambda u: -native.hodge_laplacian(0, u),
                lambda u: -self.operator.mv(u) / self.node_volumes,
                value,
            )
        return -self.operator.mv(value) / self.node_volumes

    def conservation_residual(self, values: ArrayLike, /) -> Array:
        return jnp.sum(self.node_volumes * self.mv(values))


@final
class MeshfreeExteriorCalculusPlan(StrictModule):
    """Bounded host graph/moment preparation; no inactive zero-mass spaces.

    Node volumes must be supplied, or a declared domain volume must normalize
    a positive density/kernel vector. A kernel density does not infer a domain.
    Intrinsic endpoint displacements, when supplied, follow edge-plan capacity
    order and have shape (maximum_pairs, 2, intrinsic_dimension).

    ``boundary_area_vectors`` (capacity, intrinsic_dimension) declares, per node,
    ``s_i = sum over boundary facets of its control volume of area * outward
    normal`` (zero inside). Every node with ``s_i != 0`` carries boundary-closure
    moment rows of degree one and two even when prescribed: Gauss on its control
    volume gives ``sum_e w_e d_e = -s_i`` and ``sum_e w_e d_e d_e^T = 2 V_i I``
    (nodal quadrature of the boundary face). Edges between two such nodes are
    therefore kept, so boundary nodes keep their tangential fluxes; edge fluxes
    stay antisymmetric and conservation exact. Dirichlet semantics of
    ``equation_mask`` are unchanged. The vectors are fixed by preparation.
    """

    __strict_contract__ = True
    points: Float64[_ExteriorNodeDim, _ExteriorCoordinateDim]
    radius: float = eqx.field(static=True)
    maximum_pairs: Size[_ExteriorEdgeCapacityDim] = eqx.field(static=True)
    node_volumes: Float64[_ExteriorNodeDim] | None
    domain_volume: float | None = eqx.field(static=True)
    volume_kernel: Float64[_ExteriorNodeDim] | None
    active: Bool[_ExteriorNodeDim]
    dirichlet: Bool[_ExteriorNodeDim]
    metric_policy: MeshfreeMetricPolicy
    intrinsic_displacements: (
        Float64[_ExteriorEdgeCapacityDim, Literal[2], _ExteriorIntrinsicDim] | None
    )
    edge_prior: Float64[_ExteriorEdgeCapacityDim] | None
    boundary_quadrature: MeshfreeBoundaryQuadrature | None
    coercivity_policy: MeshfreeCoercivityPolicy
    boundary_area_vectors: Float64[_ExteriorNodeDim, _ExteriorIntrinsicDim] | None

    def __init__(
        self,
        points: ArrayLike,
        radius: float,
        maximum_pairs: int,
        *,
        node_volumes: ArrayLike | None = None,
        domain_volume: float | None = None,
        volume_kernel: ArrayLike | None = None,
        active: ArrayLike | None = None,
        dirichlet: ArrayLike | None = None,
        metric_policy: MeshfreeMetricPolicy | None = None,
        intrinsic_displacements: ArrayLike | None = None,
        edge_prior: ArrayLike | None = None,
        boundary_quadrature: MeshfreeBoundaryQuadrature | None = None,
        coercivity_policy: MeshfreeCoercivityPolicy | None = None,
        boundary_area_vectors: ArrayLike | None = None,
    ) -> None:
        coordinates = jnp.asarray(points, dtype=jnp.float64)
        if (
            coordinates.ndim != 2
            or coordinates.shape[0] < 1
            or coordinates.shape[1] not in (1, 2, 3)
        ):
            raise ValueError("Exterior points must be a nonempty (capacity,1|2|3) array.")
        if not isfinite(radius) or radius <= 0 or maximum_pairs < 1:
            raise ValueError("Exterior radius and pair capacity must be positive.")
        if node_volumes is None and domain_volume is None:
            raise ValueError("Supply positive node volumes or an explicit domain_volume.")
        if node_volumes is not None and (
            domain_volume is not None or volume_kernel is not None
        ):
            raise ValueError(
                "Supplied volumes and domain-normalized kernel volumes are distinct policies."
            )
        if domain_volume is not None and (
            not isfinite(domain_volume) or domain_volume <= 0
        ):
            raise ValueError("Declared domain volume must be finite and positive.")
        count = coordinates.shape[0]
        volumes_ = (
            None if node_volumes is None else jnp.asarray(node_volumes, dtype=jnp.float64)
        )
        kernel_ = (
            None
            if volume_kernel is None
            else jnp.asarray(volume_kernel, dtype=jnp.float64)
        )
        active_ = (
            jnp.ones((count,), dtype=jnp.bool_)
            if active is None
            else jnp.asarray(active, dtype=jnp.bool_)
        )
        dirichlet_ = (
            jnp.zeros((count,), dtype=jnp.bool_)
            if dirichlet is None
            else jnp.asarray(dirichlet, dtype=jnp.bool_)
        )
        policy_ = MeshfreeMetricPolicy() if metric_policy is None else metric_policy
        intrinsic_ = (
            None
            if intrinsic_displacements is None
            else jnp.asarray(intrinsic_displacements, dtype=jnp.float64)
        )
        prior_ = (
            None if edge_prior is None else jnp.asarray(edge_prior, dtype=jnp.float64)
        )
        if active_.shape != (count,) or dirichlet_.shape != (count,):
            raise ValueError("Active and Dirichlet masks must match point capacity.")
        if volumes_ is not None and volumes_.shape != (count,):
            raise ValueError("Supplied volumes must match point capacity.")
        if kernel_ is not None and kernel_.shape != (count,):
            raise ValueError("Volume kernel must match point capacity.")
        if intrinsic_ is not None and (
            intrinsic_.ndim != 3
            or intrinsic_.shape[:2] != (maximum_pairs, 2)
            or intrinsic_.shape[2] not in (1, 2, 3)
        ):
            raise ValueError(
                "Intrinsic endpoint charts must have shape (maximum_pairs,2,1|2|3)."
            )
        if prior_ is not None and prior_.shape != (maximum_pairs,):
            raise ValueError("Metric prior must preserve declared pair capacity.")
        if not isinstance(policy_, MeshfreeMetricPolicy):
            raise TypeError("metric_policy must be a MeshfreeMetricPolicy.")
        if boundary_quadrature is not None and boundary_quadrature.weights.shape != (
            count,
        ):
            raise ValueError("Boundary quadrature must match point capacity.")
        coercivity_ = (
            MeshfreeCoercivityPolicy() if coercivity_policy is None else coercivity_policy
        )
        if not isinstance(coercivity_, MeshfreeCoercivityPolicy):
            raise TypeError("coercivity_policy must be a MeshfreeCoercivityPolicy.")
        areas_ = (
            None
            if boundary_area_vectors is None
            else jnp.asarray(boundary_area_vectors, dtype=jnp.float64)
        )
        if areas_ is not None and (
            areas_.ndim != 2
            or areas_.shape[0] != count
            or areas_.shape[1] not in (1, 2, 3)
            or not np.all(np.isfinite(np.asarray(areas_)))
        ):
            raise ValueError(
                "Boundary area vectors must be finite with shape (capacity, 1|2|3)."
            )
        self.points = coordinates
        self.radius = float(radius)
        self.maximum_pairs = int(maximum_pairs)
        self.node_volumes = volumes_
        self.domain_volume = None if domain_volume is None else float(domain_volume)
        self.volume_kernel = kernel_
        self.active = active_
        self.dirichlet = dirichlet_
        self.metric_policy = policy_
        self.intrinsic_displacements = intrinsic_
        self.edge_prior = prior_
        self.boundary_quadrature = boundary_quadrature
        self.coercivity_policy = coercivity_
        self.boundary_area_vectors = areas_

    def prepare(
        self,
        *,
        edge_relation: PreparedMeshfreeEdgeRelation | None = None,
    ) -> PreparedMeshfreeExteriorCalculus:
        """Prepare one canonical incidence and moment ordering.

        The assembly remains together to preserve sparse row ordering, refusal
        precedence, and the shared topology witness across the returned owners.
        Numerical solves and rank decisions belong to their native substrates.
        """
        edges = (
            MeshfreeEdgeRelationPlan(
                self.points,
                self.radius,
                self.maximum_pairs,
                active=self.active,
            ).prepare()
            if edge_relation is None
            else edge_relation
        )
        if not isinstance(edges, PreparedMeshfreeEdgeRelation):
            raise TypeError(
                "edge_relation must be a prepared native meshfree edge relation."
            )
        if edges.source_id != canonical_fingerprint(
            array_tree_fingerprint(np.asarray(self.points))
        ):
            raise ValueError(
                "Supplied edge relation belongs to a different point source."
            )
        if (
            edges.active_id
            != canonical_fingerprint(array_tree_fingerprint(np.asarray(self.active)))
            or edges.radius != self.radius
        ):
            raise ValueError(
                "Supplied edge relation has a different active domain or radius."
            )
        if not bool(np.asarray(edges.evidence.successful)):
            raise ValueError(
                "Supplied relation lacks successful bounded spatial evidence."
            )
        relation = edges.relation
        if (
            relation.source_size != self.points.shape[0]
            or relation.target_size != self.points.shape[0]
        ):
            raise ValueError("Supplied edge plan must refer to this point capacity.")
        if relation.capacity != self.maximum_pairs:
            raise ValueError(
                "Supplied edge relation must preserve declared pair capacity."
            )
        active = np.asarray(self.active, dtype=np.bool_)
        indices = np.flatnonzero(active).astype(np.int32)
        if indices.size == 0:
            raise ValueError("Exterior preparation requires at least one active vertex.")
        points = np.asarray(self.points)[indices]
        if not np.all(np.isfinite(points)):
            raise ValueError("Active exterior coordinates must be finite.")
        inverse = np.full(active.size, -1, dtype=np.int32)
        inverse[indices] = np.arange(indices.size, dtype=np.int32)
        valid = np.asarray(relation.valid, dtype=np.bool_)
        sources = np.asarray(relation.source_indices)
        targets = np.asarray(relation.target_indices)
        valid = valid & active[sources] & active[targets]
        # Nodes with a declared boundary area vector carry closure moment rows
        # even when prescribed. An edge joining two nodes without moment rows
        # appears in no equation; its exact minimum-norm coefficient is zero, so
        # it is omitted rather than put a zero mass into a Hodge.
        moment_capacity = ~np.asarray(self.dirichlet, dtype=np.bool_)
        if self.boundary_area_vectors is not None:
            moment_capacity = moment_capacity | np.any(
                np.asarray(self.boundary_area_vectors) != 0, axis=1
            )
        valid = valid & (moment_capacity[sources] | moment_capacity[targets])
        slots = np.flatnonzero(valid).astype(np.int32)
        pairs = np.stack((inverse[sources[slots]], inverse[targets[slots]]), axis=1)
        if pairs.size == 0 or np.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError(
                "Exterior edges must be nonempty canonical source<target pairs."
            )
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError(
                "Exterior relation must contain each canonical pair exactly once."
            )
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[order]
        slots = slots[order]
        # Periodic relations pair minimum images: lengths, priors and default
        # charts use the seam-crossing displacement, never the raw coordinate gap.
        displacements = np.asarray(
            _minimum_image(
                jnp.asarray(points[pairs[:, 1]] - points[pairs[:, 0]]), edges.address
            )
        )
        lengths = np.linalg.norm(displacements, axis=1)
        if np.any(lengths <= 0):
            raise ValueError(
                "Distinct exterior graph vertices must have positive edge lengths."
            )
        if self.node_volumes is None:
            density = (
                np.ones(indices.size)
                if self.volume_kernel is None
                else np.asarray(self.volume_kernel)[indices]
            )
            if not np.all(np.isfinite(density) & (density > 0)):
                raise ValueError(
                    "Domain-normalized volume kernel must be finite and positive on active nodes."
                )
            domain_volume = self.domain_volume
            if domain_volume is None:
                raise ValueError(
                    "Kernel normalization requires a declared domain volume."
                )
            volumes = domain_volume * density / density.sum()
        else:
            volumes = np.asarray(self.node_volumes)[indices]
        if not np.all(np.isfinite(volumes) & (volumes > 0)):
            raise ValueError("Active node volumes must be finite and strictly positive.")
        dirichlet = np.asarray(self.dirichlet)[indices]
        equations = np.flatnonzero(~dirichlet).astype(np.int32)
        if equations.size == 0:
            raise ValueError("At least one non-Dirichlet equation node is required.")
        if self.intrinsic_displacements is None:
            charts = np.stack((displacements, -displacements), axis=1)
        else:
            supplied = np.asarray(self.intrinsic_displacements)
            if (
                supplied.ndim != 3
                or supplied.shape[:2] != (relation.capacity, 2)
                or supplied.shape[2] not in (1, 2, 3)
            ):
                raise ValueError(
                    "Intrinsic endpoint displacements must follow supplied edge capacity, shape (capacity,2,1|2|3)."
                )
            charts = supplied[slots]
        if not np.all(np.isfinite(charts)):
            raise ValueError("Intrinsic endpoint displacements must be finite.")
        dimension = charts.shape[2]
        exponents = _moment_exponents(dimension, self.metric_policy.moment_degree)
        moment_count = exponents.shape[0]
        boundary_moments = _boundary_moment_indices(
            dimension, self.metric_policy.boundary_closure
        )
        if self.boundary_area_vectors is None:
            areas = np.zeros((indices.size, dimension))
        else:
            areas = np.asarray(self.boundary_area_vectors)[indices]
            if areas.shape[1] != dimension:
                raise ValueError(
                    "Boundary area vectors must use the intrinsic chart dimension."
                )
        closed = np.any(areas != 0, axis=1)
        interior = np.flatnonzero(~dirichlet & ~closed).astype(np.int32)
        boundary_nodes = np.flatnonzero(closed).astype(np.int32)
        # Row blocks: interior nodes carry every declared moment; boundary
        # closure nodes carry the moments selected by the metric policy.
        selected = np.zeros((indices.size, moment_count), dtype=np.bool_)
        selected[interior] = True
        selected[np.ix_(boundary_nodes, boundary_moments)] = True
        row_width = selected.sum(axis=1)
        moment_nodes = np.concatenate((interior, boundary_nodes))
        row_offset = np.full(indices.size, -1, dtype=np.int64)
        row_offset[moment_nodes] = np.concatenate(
            ((0,), np.cumsum(row_width[moment_nodes])[:-1])
        )
        position = np.cumsum(selected, axis=1) - 1
        scales = np.zeros(indices.size)
        np.maximum.at(
            scales, pairs.reshape(-1), np.linalg.norm(charts, axis=2).reshape(-1)
        )
        if np.any(scales[moment_nodes] <= 0):
            raise ValueError(
                "Every moment node requires a nonzero intrinsic neighborhood."
            )
        # Routes follow (edge, endpoint, multi-index) order; only endpoints with
        # moment rows carry them. Every requested moment is retained.
        monomials = np.prod(charts[:, :, None, :] ** exponents[None, None, :, :], axis=3)
        kept = selected[pairs]
        if np.count_nonzero(kept) > self.metric_policy.maximum_symbolic_entries:
            raise ValueError("Moment route count exceeds declared symbolic capacity.")
        route_edges = np.broadcast_to(
            np.arange(pairs.shape[0], dtype=np.int32)[:, None, None], monomials.shape
        )[kept]
        route_rows = (row_offset[pairs][:, :, None] + position[pairs])[kept].astype(
            np.int32
        )
        route_values = monomials[kept]
        row_nodes = np.repeat(moment_nodes, row_width[moment_nodes])
        row_moments = np.concatenate(
            [np.flatnonzero(selected[node]) for node in moment_nodes]
        ).astype(np.int32)
        # Gauss on the node's control volume: V Delta p(0) minus the outward
        # boundary flux grad p(0) . s, nonzero only for degree-one monomials.
        first = row_moments < dimension
        rhs = _laplacian_moment_rhs(exponents)[row_moments] * volumes[row_nodes]
        rhs[first] -= areas[row_nodes[first], row_moments[first]]
        row_scaling = scales[row_nodes] ** exponents.sum(axis=1)[row_moments]
        owner_id = canonical_fingerprint(
            {
                "kind": "meshfree-exterior-binding",
                "source": array_tree_fingerprint(np.asarray(self.points)),
                "active": array_tree_fingerprint(indices),
                "edges": array_tree_fingerprint(pairs),
                "neighborhood": edges.neighborhood_id,
            }
        )
        node_space = ArraySpace(
            (indices.size,), dtype=jnp.float64, space_id=f"{owner_id}:nodes"
        )
        edge_space = ArraySpace(
            (pairs.shape[0],), dtype=jnp.float64, space_id=f"{owner_id}:edges"
        )
        moment_space = ArraySpace(
            (row_nodes.size,),
            dtype=jnp.float64,
            space_id=f"{owner_id}:moments",
        )
        moment_relation = EdgeRelation(
            np.asarray(route_edges, dtype=np.int32),
            np.asarray(route_rows, dtype=np.int32),
            source_size=edge_space.size,
            target_size=moment_space.size,
        )
        constraint = SparseCoordinateOperator(
            moment_relation,
            jnp.asarray(route_values, dtype=jnp.float64),
            source=edge_space,
            target=moment_space,
            storage_plan=_SparseStoragePlan(moment_relation),
        )
        if self.edge_prior is None:
            prior = np.exp(-((lengths / self.radius) ** 2))
        else:
            supplied_prior = np.asarray(self.edge_prior)
            if supplied_prior.shape != (relation.capacity,):
                raise ValueError("Supplied metric prior must follow edge-plan capacity.")
            prior = supplied_prior[slots]
        if not np.all(np.isfinite(prior) & (prior > 0)):
            raise ValueError("Edge prior must be finite and strictly positive.")
        system = PreparedMeshfreeMetric(
            constraint,
            rhs,
            row_scaling,
            prior,
            self.metric_policy,
            equation_count=interior.size,
            intrinsic_dimension=dimension,
            boundary_count=boundary_nodes.size,
            geometry=_metric_geometry(
                points, moment_nodes, pairs, displacements, charts, edges.address
            ),
        )
        result = system.solve()
        vertices = EntitySet("meshfree-vertices", 0, indices)
        edge_entities = EntitySet("meshfree-edges", 1, slots)
        incidence_relation = EdgeRelation(
            pairs.reshape(-1),
            np.repeat(np.arange(pairs.shape[0]), 2),
            source_size=indices.size,
            target_size=pairs.shape[0],
        )
        signs = np.tile(np.asarray([-1.0, 1.0]), pairs.shape[0])
        incidence = OrientedIncidence(
            1, vertices, edge_entities, incidence_relation, signs
        )
        topology = CellComplexTopology((vertices, edge_entities), (incidence,))
        incidence_operator = SparseCoordinateOperator(
            incidence.relation,
            incidence.signs,
            source=node_space,
            target=edge_space,
            storage_plan=_SparseStoragePlan(incidence.relation),
        )
        native: CochainDiscretization | None = None
        if bool(np.asarray(result.hilbert_admitted)):
            native = CochainDiscretization(
                topology,
                (DiagonalHodge(volumes), DiagonalHodge(result.weights)),
                boundary_masks=(dirichlet, np.zeros(pairs.shape[0], dtype=np.bool_)),
                coordinates=(points, 0.5 * (points[pairs[:, 0]] + points[pairs[:, 1]])),
                primal_measures=(np.ones(indices.size), lengths),
                dual_measures=(volumes, lengths * np.asarray(result.weights)),
                numeric_revision="meshfree-metric",
            )
        quadrature = (
            None
            if self.boundary_quadrature is None
            else MeshfreeBoundaryQuadrature(self.boundary_quadrature.weights[indices])
        )
        # Bounded radius-shell witness: every incidence within 1.5 radius is
        # enumerated on a fixed candidate buffer sized from the admitted edge
        # query, so its work is charged to the spatial query rather than the
        # metric symbolic budget. The certified gap is capped at radius/2 and is
        # zero when the witness overflows or misses an owner, in which case no
        # coordinate motion is trusted.
        edge_capacity, edge_required = jax.device_get(
            (
                edges.evidence.candidate_capacity,
                edges.evidence.required_candidates,
            )
        )
        capacity = self.points.shape[0]
        shell = MortonRadiusShellWitnessPlan(
            edges.address,
            capacity,
            capacity,
            maximum_candidates=min(
                capacity,
                max(int(edge_capacity), 2 ** points.shape[1] * int(edge_required)),
            ),
        ).certify(
            self.points,
            self.points,
            self.radius,
            0.5 * self.radius,
            source_mask=self.active,
            target_mask=self.active,
            source_stable_ids=edges.point_ids,
            target_stable_ids=edges.point_ids,
            exclude_self=True,
        )
        radius_gap = float(jax.device_get(shell.evidence.certified_gap))
        return PreparedMeshfreeExteriorCalculus(
            points=jnp.asarray(points),
            pairs=jnp.asarray(pairs, dtype=jnp.int32),
            lengths=jnp.asarray(lengths),
            node_volumes=jnp.asarray(volumes),
            capacity_indices=jnp.asarray(indices),
            edge_capacity_indices=jnp.asarray(slots),
            equation_mask=jnp.asarray(~dirichlet),
            equation_indices=jnp.asarray(equations),
            incidence=incidence_operator,
            topology=topology,
            metric_system=system,
            metric_result=result,
            native=native,
            edge_relation=edges,
            boundary_quadrature=quadrature,
            endpoint_displacements=jnp.asarray(charts),
            intrinsic=self.intrinsic_displacements is not None,
            topology_trust_margin=0.5 * radius_gap,
            coercivity_policy=self.coercivity_policy,
            moment_node_indices=jnp.asarray(moment_nodes),
            boundary_area_vectors=jnp.asarray(areas),
        )


def _metric_geometry(
    points: np.ndarray,
    moment_nodes: np.ndarray,
    pairs: np.ndarray,
    displacements: np.ndarray,
    charts: np.ndarray,
    address: MortonAddressPlan,
    /,
) -> MeshfreeMetricGeometry | None:
    """Coordinate frame of the moment displacements; ``None`` for surface charts.

    The jet hierarchy of the multilevel metric solve needs the moment
    displacements of both endpoints to be ``+-`` the minimum-image coordinate
    difference (default or minimum-image charts). Non-periodic axes bound the
    moment nodes tightly; periodic axes keep the address period.
    """
    coordinate_frame = charts.shape[2] == points.shape[1] and np.allclose(
        charts,
        np.stack((displacements, -displacements), axis=1),
        rtol=1e-12,
        atol=1e-12 * float(np.abs(displacements).max()),
    )
    if not coordinate_frame:
        return None
    lengths = np.linalg.norm(displacements, axis=1)
    nearest = np.full(points.shape[0], np.inf)
    np.minimum.at(nearest, pairs.reshape(-1), np.repeat(lengths, 2))
    spacing = float(np.median(nearest[moment_nodes]))
    periodic = tuple(bool(axis) for axis in address.periodic_axes)
    moment_points = points[moment_nodes]
    lower = np.where(periodic, np.asarray(address.lower), moment_points.min(axis=0))
    upper = np.where(periodic, np.asarray(address.upper), moment_points.max(axis=0))
    # A flat cloud (zero extent along an axis) still needs a positive cell.
    upper = np.maximum(upper, lower + spacing)
    return MeshfreeMetricGeometry(
        moment_points, lower=lower, upper=upper, periodic=periodic, spacing=spacing
    )


@final
class PreparedMeshfreeExteriorCalculus(StrictModule):
    """Meshfree metric/boundary owner; native exterior algebra is canonical."""

    __strict_contract__ = True
    points: Float64[_ExteriorNodeDim, _ExteriorCoordinateDim]
    pairs: Int32[_ExteriorEdgeDim, Literal[2]]
    lengths: Float64[_ExteriorEdgeDim]
    node_volumes: Float64[_ExteriorNodeDim]
    capacity_indices: Int32[_ExteriorNodeDim]
    edge_capacity_indices: Int32[_ExteriorEdgeDim]
    equation_mask: Bool[_ExteriorNodeDim]
    equation_indices: Int32[_ExteriorEquationDim]
    moment_node_indices: Int32[_ExteriorMomentNodeDim]
    boundary_area_vectors: Float64[_ExteriorNodeDim, _ExteriorIntrinsicDim]
    incidence: SparseCoordinateOperator
    topology: CellComplexTopology
    metric_system: PreparedMeshfreeMetric
    metric_result: MeshfreeMetricResult
    native: CochainDiscretization | None
    edge_relation: PreparedMeshfreeEdgeRelation
    boundary_quadrature: MeshfreeBoundaryQuadrature | None
    _stiffness_relation: EdgeRelation
    _stiffness_storage: _SparseStoragePlan
    _reduced_relation: EdgeRelation
    _reduced_storage: _SparseStoragePlan
    _reduced_routes: Int32[_ExteriorReducedRouteDim]
    _reduced_space: ArraySpace
    _factor_plan: SparseFactorizationPlan | None
    coercivity_policy: MeshfreeCoercivityPolicy
    endpoint_displacements: Float64[_ExteriorEdgeDim, Literal[2], _ExteriorIntrinsicDim]
    intrinsic: bool = eqx.field(static=True)
    topology_trust_margin: float = eqx.field(static=True)
    _initial_points: Float64[_ExteriorNodeDim, _ExteriorCoordinateDim]
    _moment_endpoints: Int32[_ExteriorMomentRouteDim]
    _moment_indices: Int32[_ExteriorMomentRouteDim]
    _moment_axes: Int32[_ExteriorMomentDim, _ExteriorMomentFactorDim]
    _row_nodes: Int32[_ExteriorMomentRowDim]
    _row_rhs_unit: Float64[_ExteriorMomentRowDim]
    _row_boundary_flux: Float64[_ExteriorMomentRowDim]

    def __init__(
        self,
        *,
        points: Array,
        pairs: Array,
        lengths: Array,
        node_volumes: Array,
        capacity_indices: Array,
        edge_capacity_indices: Array,
        equation_mask: Array,
        equation_indices: Array,
        incidence: SparseCoordinateOperator,
        topology: CellComplexTopology,
        metric_system: PreparedMeshfreeMetric,
        metric_result: MeshfreeMetricResult,
        native: CochainDiscretization | None,
        edge_relation: PreparedMeshfreeEdgeRelation,
        boundary_quadrature: MeshfreeBoundaryQuadrature | None,
        endpoint_displacements: Array,
        intrinsic: bool,
        topology_trust_margin: float,
        coercivity_policy: MeshfreeCoercivityPolicy,
        moment_node_indices: Array,
        boundary_area_vectors: Array,
    ) -> None:
        endpoints = np.asarray(pairs)
        i, j = endpoints[:, 0], endpoints[:, 1]
        rows = np.stack((i, j, i, j), axis=1).reshape(-1)
        columns = np.stack((i, j, j, i), axis=1).reshape(-1)
        stiffness_relation = EdgeRelation(
            columns, rows, source_size=points.shape[0], target_size=points.shape[0]
        )
        stiffness_storage = _SparseStoragePlan(stiffness_relation)
        reduced = np.asarray(equation_indices)
        inverse = np.full(points.shape[0], -1, dtype=np.int32)
        inverse[reduced] = np.arange(reduced.size, dtype=np.int32)
        routes = np.flatnonzero((inverse[rows] >= 0) & (inverse[columns] >= 0))
        reduced_relation = EdgeRelation(
            inverse[columns[routes]],
            inverse[rows[routes]],
            source_size=reduced.size,
            target_size=reduced.size,
        )
        reduced_storage = _SparseStoragePlan(reduced_relation)
        reduced_space = ArraySpace(
            (reduced.size,),
            dtype=jnp.float64,
            space_id=f"{incidence.source.space_id}:equations",
        )
        seed = SparseCoordinateOperator(
            reduced_relation,
            jnp.ones(routes.shape, dtype=jnp.float64),
            source=reduced_space,
            target=reduced_space,
            properties=OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
            storage_plan=reduced_storage,
        )
        factor_plan = coercivity_policy.prepare(seed, np.asarray(points)[reduced])
        moment_relation = cast(EdgeRelation, metric_system.constraint.relation)
        moment_rows = np.asarray(moment_relation.target_indices)
        moment_edges = np.asarray(moment_relation.source_indices)
        # Metric rows are grouped by moment node (interior, then boundary
        # closure); map them back to compact nodes and multi-indices.
        row_nodes = np.asarray(moment_node_indices)[np.asarray(metric_system.row_nodes)]
        row_moments = np.asarray(metric_system.row_moments)
        moment_nodes = row_nodes[moment_rows]
        moment_endpoints = np.where(
            endpoints[moment_edges, 0] == moment_nodes, 0, 1
        ).astype(np.int32)
        dimension = metric_system.intrinsic_dimension
        exponents = _moment_exponents(dimension, metric_system.policy.moment_degree)
        # Monomial axis factors, padded with the unit coordinate at index d, so
        # refreshed moments are products of chart entries without powers.
        degree = metric_system.policy.moment_degree
        moment_axes = np.full((exponents.shape[0], degree), dimension, dtype=np.int32)
        for moment, exponent in enumerate(exponents):
            axes = np.repeat(np.arange(dimension, dtype=np.int32), exponent)
            moment_axes[moment, : axes.size] = axes
        areas = np.asarray(boundary_area_vectors)
        first = row_moments < dimension
        row_boundary_flux = np.zeros(row_nodes.shape)
        row_boundary_flux[first] = -areas[row_nodes[first], row_moments[first]]
        self.points = points
        self.pairs = pairs
        self.lengths = lengths
        self.node_volumes = node_volumes
        self.capacity_indices = capacity_indices
        self.edge_capacity_indices = edge_capacity_indices
        self.equation_mask = equation_mask
        self.equation_indices = equation_indices
        self.moment_node_indices = moment_node_indices
        self.boundary_area_vectors = boundary_area_vectors
        self.incidence = incidence
        self.topology = topology
        self.metric_system = metric_system
        self.metric_result = metric_result
        self.native = native
        self.edge_relation = edge_relation
        self.boundary_quadrature = boundary_quadrature
        self._stiffness_relation = stiffness_relation
        self._stiffness_storage = stiffness_storage
        self._reduced_routes = jnp.asarray(routes, dtype=jnp.int32)
        self._reduced_relation = reduced_relation
        self._reduced_storage = reduced_storage
        self._reduced_space = reduced_space
        self._factor_plan = factor_plan
        self.coercivity_policy = coercivity_policy
        self.endpoint_displacements = endpoint_displacements
        self.intrinsic = intrinsic
        self.topology_trust_margin = topology_trust_margin
        self._initial_points = points
        self._moment_endpoints = jnp.asarray(moment_endpoints)
        self._moment_indices = jnp.asarray(row_moments[moment_rows], dtype=jnp.int32)
        self._moment_axes = jnp.asarray(moment_axes)
        self._row_nodes = jnp.asarray(row_nodes, dtype=jnp.int32)
        self._row_rhs_unit = jnp.asarray(_laplacian_moment_rhs(exponents)[row_moments])
        self._row_boundary_flux = jnp.asarray(row_boundary_flux)

    def displacement(self, points: ArrayLike, /) -> Array:
        """Largest nodal motion of compact ``points`` from the prepared anchor.

        Coordinates are raw (unwrapped): periodic consumers pass continuous
        trajectories, so seam crossings are motion, not teleportation.
        """
        coordinates = jnp.asarray(points, dtype=jnp.float64)
        if coordinates.shape != self.points.shape:
            raise ValueError("Exterior motion is measured on compact node coordinates.")
        return jnp.max(jnp.linalg.norm(coordinates - self._initial_points, axis=1))

    def within_topology_trust(self, points: ArrayLike, /) -> Array:
        """Whether ``points`` keep the certified fixed-radius edge topology.

        Admitted motion is zero or strictly below ``topology_trust_margin``; the
        margin is zero when the prepared shell witness certified no gap.
        """
        displacement = self.displacement(points)
        return (displacement == 0) | (displacement < self.topology_trust_margin)

    def _refresh_inputs(
        self,
        points: ArrayLike | None,
        node_volumes: ArrayLike | None,
        intrinsic_displacements: ArrayLike | None,
        /,
    ) -> tuple[Array, Array, Array | None]:
        coordinates = (
            self.points if points is None else jnp.asarray(points, dtype=jnp.float64)
        )
        volumes = (
            self.node_volumes
            if node_volumes is None
            else jnp.asarray(node_volumes, dtype=jnp.float64)
        )
        if (
            coordinates.shape != self.points.shape
            or volumes.shape != self.node_volumes.shape
        ):
            raise ValueError("Exterior refresh must preserve compact node dimensions.")
        if self.intrinsic and points is not None and intrinsic_displacements is None:
            raise ValueError(
                "Intrinsic coordinate refresh requires refreshed endpoint charts."
            )
        charts = (
            None
            if intrinsic_displacements is None
            else jnp.asarray(intrinsic_displacements, dtype=jnp.float64)
        )
        if charts is not None and charts.shape != self.endpoint_displacements.shape:
            raise ValueError(
                "Refreshed intrinsic charts must preserve compact edge/endpoint dimensions."
            )
        return coordinates, volumes, charts

    def refresh(
        self,
        *,
        points: ArrayLike | None = None,
        node_volumes: ArrayLike | None = None,
        edge_prior: ArrayLike | None = None,
        intrinsic_displacements: ArrayLike | None = None,
    ) -> PreparedMeshfreeExteriorCalculus:
        """Refresh numerical data within the prepared radius-topology witness.

        All inputs use compact coordinates. The moment pattern and row scaling
        stay fixed; every original moment residual, the all-equation solver
        evidence and the policy-selected stiffness assessment audit the refreshed
        solve. Derivatives are fixed-rank derivatives under the metric contract;
        nonnegative results need an eager ``metric_system.bind_active_set``.
        Leaving the topology trust or nonpositive volumes raise; use
        :meth:`try_refresh` for a status-returning transaction.
        """
        coordinates, volumes, charts = self._refresh_inputs(
            points, node_volumes, intrinsic_displacements
        )
        coordinates = eqx.error_if(
            coordinates,
            ~jnp.all(jnp.isfinite(coordinates))
            | ~self.within_topology_trust(coordinates),
            "Exterior coordinates leave the certified fixed-radius topology; prepare a new epoch.",
        )
        volumes = eqx.error_if(
            volumes,
            jnp.any(~jnp.isfinite(volumes) | (volumes <= 0)),
            "Refreshed node volumes must remain finite and positive.",
        )
        return self._refreshed(coordinates, volumes, edge_prior, charts)

    def try_refresh(
        self,
        *,
        points: ArrayLike | None = None,
        node_volumes: ArrayLike | None = None,
        edge_prior: ArrayLike | None = None,
        intrinsic_displacements: ArrayLike | None = None,
    ) -> MeshfreeExteriorRefreshResult:
        """Status-returning refresh: the owner is replaced only when accepted.

        Coordinates, volumes and topology trust are admitted before any metric
        solve; a geometric refusal returns the unchanged owner without solving.
        Otherwise the raw candidate metric evidence is retained, and an
        unaccepted metric also leaves the owner unchanged (rollback). Results
        bound by ``bind_active_set`` must be rebound after a refresh, so a bound
        owner is refused here.
        """
        if self.metric_result.active_set is not None:
            raise ValueError(
                "Refresh the unbound exterior; active-set evidence is rebound after refresh."
            )
        coordinates, volumes, charts = self._refresh_inputs(
            points, node_volumes, intrinsic_displacements
        )
        finite = jnp.all(jnp.isfinite(coordinates))
        displacement = self.displacement(coordinates)
        trusted = self.within_topology_trust(coordinates)
        positive = jnp.all(jnp.isfinite(volumes) & (volumes > 0))
        geometric = finite & trusted & positive
        candidate = jax.lax.cond(
            geometric,
            lambda _: self._refreshed(coordinates, volumes, edge_prior, charts),
            lambda _: self,
            None,
        )
        metric = candidate.metric_result.accepted
        accepted = geometric & metric
        status = jnp.where(
            ~finite,
            int(MeshfreeExteriorRefreshStatus.INVALID_COORDINATES),
            jnp.where(
                ~trusted,
                int(MeshfreeExteriorRefreshStatus.TOPOLOGY_TRUST_EXCEEDED),
                jnp.where(
                    ~positive,
                    int(MeshfreeExteriorRefreshStatus.INVALID_VOLUMES),
                    jnp.where(
                        ~metric,
                        int(MeshfreeExteriorRefreshStatus.METRIC_REFUSED),
                        int(MeshfreeExteriorRefreshStatus.ACCEPTED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        owner = jax.tree.map(
            lambda new, old: jnp.where(accepted, new, old), candidate, self
        )
        return MeshfreeExteriorRefreshResult(
            exterior=owner,
            candidate_metric=candidate.metric_result,
            displacement=displacement,
            trust_margin=jnp.asarray(self.topology_trust_margin, dtype=jnp.float64),
            status=status,
            accepted=accepted,
        )

    def _refreshed(
        self,
        coordinates: Array,
        volumes: Array,
        edge_prior: ArrayLike | None,
        intrinsic_displacements: Array | None,
        /,
    ) -> PreparedMeshfreeExteriorCalculus:
        offset = _minimum_image(
            coordinates[self.pairs[:, 1]] - coordinates[self.pairs[:, 0]],
            self.edge_relation.address,
        )
        if intrinsic_displacements is not None:
            charts = intrinsic_displacements
        else:
            charts = (
                self.endpoint_displacements
                if self.intrinsic
                else jnp.stack((offset, -offset), axis=1)
            )
        unit = jnp.ones(charts.shape[:2] + (1,), dtype=charts.dtype)
        local_moments = jnp.prod(
            jnp.concatenate((charts, unit), axis=2)[..., self._moment_axes], axis=-1
        )
        routes = self.metric_system.constraint.relation.source_indices
        raw_values = local_moments[routes, self._moment_endpoints, self._moment_indices]
        # Boundary area vectors are fixed by preparation: motion within the
        # topology trust keeps them; boundary geometry changes are a new epoch.
        rhs = self._row_rhs_unit * volumes[self._row_nodes] + self._row_boundary_flux
        result = self.metric_system.solve(
            prior=edge_prior, rhs=rhs, coefficients=raw_values
        )
        prior = (
            self.metric_system.prior
            if edge_prior is None
            else jnp.asarray(edge_prior, dtype=jnp.float64)
        )
        system = eqx.tree_at(
            lambda owner: (owner.constraint.coefficients, owner.rhs, owner.prior),
            self.metric_system,
            (raw_values, rhs, prior),
        )
        native = self.native
        if native is not None:
            rollback = native

            def bind_positive(weights: Array) -> CochainDiscretization:
                bound = rollback.with_metric(
                    (DiagonalHodge(volumes), DiagonalHodge(weights)),
                    numeric_revision=rollback.numeric_revision,
                )
                lengths = jnp.linalg.norm(offset, axis=1)
                return eqx.tree_at(
                    lambda owner: (
                        owner.coordinates,
                        owner.primal_measures,
                        owner.dual_measures,
                    ),
                    bound,
                    (
                        (
                            coordinates,
                            0.5
                            * (
                                coordinates[self.pairs[:, 0]]
                                + coordinates[self.pairs[:, 1]]
                            ),
                        ),
                        (jnp.ones_like(volumes), lengths),
                        (volumes, lengths * weights),
                    ),
                )

            # The raw candidate and its refusal status remain metric_result.
            # A refused candidate retains native only as explicit rollback state;
            # it never replaces candidate weights or passes native admission.
            native = jax.lax.cond(
                result.hilbert_admitted, bind_positive, lambda _: rollback, result.weights
            )
        return eqx.tree_at(
            lambda owner: (
                owner.points,
                owner.node_volumes,
                owner.lengths,
                owner.endpoint_displacements,
                owner.metric_system,
                owner.metric_result,
                owner.native,
            ),
            self,
            (
                coordinates,
                volumes,
                jnp.linalg.norm(offset, axis=1),
                charts,
                system,
                result,
                native,
            ),
            is_leaf=lambda value: value is None,
        )

    @property
    def boundary_normal_measure(self) -> Array:
        """Discrete outward normal measure ``S_i = -sum_e w_e d_e^(i)`` per node.

        Under the ``"second-moment"`` boundary closure this deficit is the
        boundary flux measure of conservative edge transport; it satisfies
        ``sum_i S_i = 0`` and, with exact second moments at every moment node,
        ``sum_i x_i S_i^T = sum_i V_i I``. Compare it with the declared
        ``boundary_area_vectors`` for geometric fidelity evidence.
        """
        weights = self.metric_result.weights[:, None]
        charts = self.endpoint_displacements
        count = self.points.shape[0]
        return -(
            jax.ops.segment_sum(weights * charts[:, 0], self.pairs[:, 0], count)
            + jax.ops.segment_sum(weights * charts[:, 1], self.pairs[:, 1], count)
        )

    def gradient(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values, dtype=jnp.float64)
        native = self.native
        if native is not None:
            return jax.lax.cond(
                self.metric_result.hilbert_admitted,
                lambda u: native.exterior_derivative(0, u),
                lambda u: self.incidence.mv(u),
                value,
            )
        return self.incidence.mv(value)

    def divergence(self, oriented_volume_flux: ArrayLike, /) -> Array:
        flux = jnp.asarray(oriented_volume_flux, dtype=jnp.float64)
        if flux.shape != self.lengths.shape:
            raise ValueError("Oriented volume flux must match compact canonical edges.")
        native = self.native
        if native is not None:
            return jax.lax.cond(
                self.metric_result.hilbert_admitted,
                lambda f: -native.codifferential(1, f / native.hodge_diagonal(1)),
                lambda f: -self.incidence.transpose_mv(f) / self.node_volumes,
                flux,
            )
        return -self.incidence.transpose_mv(flux) / self.node_volumes

    def stiffness(self, edge_coefficients: ArrayLike, /) -> SparseCoordinateOperator:
        coefficients = jnp.asarray(edge_coefficients, dtype=jnp.float64)
        if coefficients.shape != self.lengths.shape:
            raise ValueError("Stiffness conductances must match compact canonical edges.")
        values = (coefficients[:, None] * jnp.asarray((1.0, 1.0, -1.0, -1.0))).reshape(-1)
        return SparseCoordinateOperator(
            self._stiffness_relation,
            values,
            source=self.incidence.source,
            target=self.incidence.source,
            storage_plan=self._stiffness_storage,
            properties=OperatorProperties(
                self_adjoint=True, evidence={"self_adjoint": "construction"}
            ),
        )

    def stiffness_evidence(
        self, edge_coefficients: ArrayLike, /
    ) -> MeshfreeStiffnessEvidence:
        coefficients = jnp.asarray(edge_coefficients, dtype=jnp.float64)
        operator = self.stiffness(coefficients)
        reduced = SparseCoordinateOperator(
            self._reduced_relation,
            operator.coefficients[self._reduced_routes],
            source=self._reduced_space,
            target=self._reduced_space,
            storage_plan=self._reduced_storage,
            properties=operator.properties,
        )
        plan = self._factor_plan
        if plan is None:
            status = jnp.asarray(-1, dtype=jnp.int32)
            minimum_pivot = jnp.asarray(jnp.nan, dtype=jnp.float64)
            spd = jnp.asarray(False)
        else:
            factor = refresh_sparse_factorization_values(
                plan, reduced.sparse_storage().values
            )
            status = factor.status.astype(jnp.int32)
            minimum_pivot = factor.diagnostics.minimum_pivot
            spd = (
                factor.status == int(SparseFactorizationStatus.SUCCESS)
            ) & factor.diagnostics.finite
        # Positive-edge reachability proves maximum-principle anchoring.
        # Nonzero-edge reachability is a necessary coercivity condition even
        # for signed coefficients: every unanchored component has a constant
        # null mode, regardless of a floating-point Cholesky pivot.
        count = self.points.shape[0]
        anchored = jnp.broadcast_to((~self.equation_mask)[:, None], (count, 2))
        positive_routes = jnp.repeat((coefficients > 0).astype(jnp.float64), 4)
        nonzero_routes = jnp.repeat((coefficients != 0).astype(jnp.float64), 4)

        def needs_expansion(state: tuple[Array, Array, Array]) -> Array:
            iteration, _, changed = state
            return (iteration < count) & changed

        def expand(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
            iteration, reachable, _ = state
            additions = jnp.stack(
                (
                    linear_apply(
                        self._stiffness_relation,
                        positive_routes,
                        reachable[:, 0].astype(jnp.float64),
                    ),
                    linear_apply(
                        self._stiffness_relation,
                        nonzero_routes,
                        reachable[:, 1].astype(jnp.float64),
                    ),
                ),
                axis=1,
            )
            updated = reachable | (additions > 0)
            return iteration + 1, updated, jnp.any(updated != reachable)

        _, reached, _ = jax.lax.while_loop(
            needs_expansion,
            expand,
            (jnp.asarray(0), anchored, jnp.asarray(True)),
        )
        anchored_components = jnp.all(reached[:, 0])
        spd = spd & jnp.all(reached[:, 1])
        nonnegative = jnp.all(jnp.isfinite(coefficients) & (coefficients >= 0))
        return MeshfreeStiffnessEvidence(
            spd=spd,
            factorization_status=status,
            minimum_pivot=minimum_pivot,
            positive_inertia=jnp.where(spd, self._reduced_space.size, -1).astype(
                jnp.int32
            ),
            negative_inertia=jnp.where(spd, 0, -1).astype(jnp.int32),
            zero_inertia=jnp.where(spd, 0, -1).astype(jnp.int32),
            inertia_available=spd,
            anchored_components=anchored_components,
            maximum_principle=nonnegative
            & anchored_components
            & spd
            & self.metric_result.accepted,
            dimension=self._reduced_space.size,
            assessed=plan is not None,
            source=self.coercivity_policy.source,
        )

    def diffusion(
        self,
        coefficient: ArrayLike = 1.0,
        *,
        averaging: EdgeCoefficientAverage = "harmonic",
    ) -> MeshfreeDiffusionOperator:
        averaging = parse(averaging, EdgeCoefficientAverage, "averaging")
        values = jnp.asarray(coefficient, dtype=jnp.float64)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values) | (values < 0)),
            "Diffusion coefficients must be finite and nonnegative.",
        )
        if averaging == "supplied":
            if values.shape != self.lengths.shape:
                raise ValueError(
                    "Supplied diffusion coefficient must match compact edges."
                )
            edge_values = values
        elif values.shape == ():
            edge_values = jnp.broadcast_to(values, self.lengths.shape)
        else:
            if values.shape != self.node_volumes.shape:
                raise ValueError(
                    "Nodal diffusion coefficients must match compact active vertices."
                )
            left, right = values[self.pairs[:, 0]], values[self.pairs[:, 1]]
            denominator = left + right
            edge_values = (
                2 * left * right / jnp.where(denominator > 0, denominator, 1.0)
                if averaging == "harmonic"
                else 0.5 * denominator
            )
        conductances = self.metric_result.weights * edge_values
        native_active = (
            jnp.asarray(self.native is not None)
            & self.metric_result.hilbert_admitted
            & jnp.all(conductances > 0)
        )
        native = self.native
        if native is not None:
            rollback = native
            native = jax.lax.cond(
                native_active,
                lambda a: rollback.with_metric(
                    (rollback.hodges[0], DiagonalHodge(a)),
                    numeric_revision=rollback.numeric_revision,
                ),
                lambda _: rollback,
                conductances,
            )
        return MeshfreeDiffusionOperator(
            operator=self.stiffness(conductances),
            conductances=conductances,
            node_volumes=self.node_volumes,
            native=native,
            admitted=self.metric_result.accepted,
            native_active=native_active,
            evidence=self.stiffness_evidence(conductances),
        )

    def natural_boundary_load(self, outward_flux: ArrayLike, /) -> Array:
        if self.boundary_quadrature is None:
            raise ValueError("Natural loads require declared boundary quadrature.")
        return self.boundary_quadrature.load(outward_flux)

    def to_cochain(self) -> CochainDiscretization:
        if self.native is None:
            raise ValueError(
                "Signed, zero, or unaccepted edge metrics cannot become native positive cochains."
            )
        return eqx.error_if(
            self.native,
            ~self.metric_result.hilbert_admitted,
            "Candidate metric has failed native admission; retained native state is rollback only.",
        )

    def hilbert_complex(
        self, *, boundary: ComplexBoundary = "absolute"
    ) -> HilbertComplex:
        return self.to_cochain().hilbert_complex(boundary=boundary)

    def to_graph_ir(self, *, boundary: ComplexBoundary = "absolute") -> CochainComplexIR:
        return CochainComplexIR(self.to_cochain(), boundary=boundary)


class MeshfreeExteriorRefreshStatus(IntEnum):
    ACCEPTED = 0
    INVALID_COORDINATES = 1
    TOPOLOGY_TRUST_EXCEEDED = 2
    INVALID_VOLUMES = 3
    METRIC_REFUSED = 4


@final
class MeshfreeExteriorRefreshResult(StrictModule):
    """Transaction outcome of :meth:`PreparedMeshfreeExteriorCalculus.try_refresh`.

    ``exterior`` is the refreshed owner when ``accepted`` and the unchanged
    owner otherwise. ``candidate_metric`` is the raw candidate evidence (the
    unchanged metric when geometry refused before any solve).
    """

    __strict_contract__ = True
    exterior: PreparedMeshfreeExteriorCalculus
    candidate_metric: MeshfreeMetricResult
    displacement: Float64[Scalar]
    trust_margin: Float64[Scalar]
    status: Int32[Scalar]
    accepted: Bool[Scalar]
