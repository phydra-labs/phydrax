#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Moving least-squares point-cloud values as an evidenced field view.

At a query point `x` the reconstruction fits the point cloud's total-degree
polynomial basis in standardized offsets `(x_j - x) / h` by weighted least
squares over the cloud points within the neighborhood radius `h` (Wendland
`C^2` weights `(1 - r)^4 (4 r + 1)`, `r = |x_j - x| / h`) and returns the
constant coefficient. The local system is solved by the native dense SVD
least-squares route, whose rank and condition estimate are the pointwise
conditioning evidence; no inverse is formed. Neighborhoods come from a
prepared bounded `PackedBVH` query with an explicit candidate capacity, never
an all-pairs search.

Coverage is `partial`: points with too few neighbors are `OUTSIDE_SUPPORT`,
rank-deficient or ill-conditioned local fits are `ILL_CONDITIONED`, and an
exceeded candidate capacity is `LOCATION_FAILED`. Views therefore answer
`.query(...)` with evidence and refuse `as_domain_function()`.
"""

from __future__ import annotations

from math import comb, isfinite
from numbers import Integral
from typing import Any, final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._bvh import BVHBuildPolicy, PackedBVH, point_select_leaf_items, prepare_bvh
from .._differentiation import DerivativeRegularity
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._interpolation import GatherStencil
from .._model._ports import ValuePort
from .._polynomial._total_degree import TotalDegreePolynomialFeatures
from .._trainable import NonTrainableState
from ..sparse import linear_apply, linear_transpose_apply
from ..typing import parse
from ._point_cloud import PreparedPointCloudDiscretization
from ._views import (
    AbstractFieldReconstructionKernel,
    FieldQueryEvidence,
    FieldQueryStatus,
    FieldSideBinding,
    FieldTracePolicy,
    FieldTraceSide,
    PreparedFieldReconstruction,
)
from .meshfree._stencils import weighted_svd_factors


# Leaf granularity of the prepared point hierarchy; the candidate capacity
# bounds every leaf item within the radius box of one query.
_LEAF_SIZE = 8
_RANK_CUTOFF = 1.0e-12

PointCloudReconstruction: TypeAlias = Literal["polynomial", "shepard"]


def _wendland_c2(ratio: Array, /) -> Array:
    return (1.0 - ratio) ** 4 * (4.0 * ratio + 1.0)


@final
class PointCloudFieldReconstructionKernel(
    AbstractFieldReconstructionKernel, NonTrainableState
):
    """Moving least-squares evaluation over bounded point-cloud neighborhoods."""

    points: Array
    bvh: PackedBVH
    exponents: Array
    radius: float = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    minimum_neighbors: int = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    reconstruction: PointCloudReconstruction = eqx.field(static=True)
    source_owner_id: str = eqx.field(static=True)
    _kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        points: np.ndarray,
        degree: int,
        /,
        *,
        radius: float,
        capacity: int,
        minimum_neighbors: int,
        condition_limit: float,
        field_space_id: str,
        source_owner_id: str,
        reconstruction: PointCloudReconstruction = "polynomial",
    ) -> None:
        cloud = np.asarray(points, dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[0] == 0 or not np.all(np.isfinite(cloud)):
            raise ValueError(
                "Point-cloud coordinates must be a finite (points, dimension) array."
            )
        features = TotalDegreePolynomialFeatures(cloud.shape[1], degree)
        if not isinstance(source_owner_id, str) or not source_owner_id:
            raise ValueError(
                "source_owner_id must identify the prepared point-cloud owner."
            )
        reconstruction_ = parse(
            reconstruction, PointCloudReconstruction, "reconstruction"
        )
        if reconstruction_ == "shepard" and degree != 0:
            raise ValueError("Shepard reconstruction requires explicit degree zero.")
        if (
            not isfinite(radius)
            or radius <= 0
            or not isfinite(condition_limit)
            or condition_limit <= 1
        ):
            raise ValueError(
                "Reconstruction radius and condition limit must be admissible."
            )
        if (
            not features.feature_count + 1
            <= minimum_neighbors
            <= capacity
            <= cloud.shape[0]
        ):
            raise ValueError("Reconstruction capacities must cover the feature count.")
        bvh = prepare_bvh(
            cloud,
            cloud,
            policy=BVHBuildPolicy(leaf_size=min(_LEAF_SIZE, cloud.shape[0])),
            dtype=jnp.float64,
        )
        kernel_id = canonical_fingerprint(
            {
                "kind": "point-cloud-moving-least-squares-kernel",
                "points": array_tree_fingerprint(cloud),
                "features": features.feature_id,
                "weight": "wendland-c2",
                "reconstruction": reconstruction_,
                "radius": radius,
                "capacity": capacity,
                "minimum_neighbors": minimum_neighbors,
                "condition_limit": condition_limit,
                "rank_cutoff": _RANK_CUTOFF,
                "field_space": field_space_id,
                "source_owner": source_owner_id,
            }
        )
        self.points = jnp.asarray(cloud)
        self.bvh = bvh
        self.exponents = features.exponents
        self.radius = radius
        self.capacity = capacity
        self.minimum_neighbors = minimum_neighbors
        self.condition_limit = condition_limit
        self.reconstruction = reconstruction_
        self.source_owner_id = source_owner_id
        self._kernel_id = kernel_id

    @property
    def kernel_id(self) -> str:
        return self._kernel_id

    @property
    def cell_count(self) -> int:
        return 0

    @property
    def support_coverage(self) -> str:
        return "partial"

    @property
    def feature_count(self) -> int:
        """Polynomial features including the constant."""
        return self.exponents.shape[0] + 1

    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[GatherStencil, FieldQueryEvidence]:
        del side
        if any(derivative):
            raise ValueError("Moving least-squares views evaluate values only.")
        finite = jnp.all(jnp.isfinite(points), axis=1)
        query = jnp.where(finite[:, None], points, self.points[0].astype(points.dtype))
        candidates, candidate_valid, complete = point_select_leaf_items(
            query, bvh=self.bvh, maximum_candidates=self.capacity, tolerance=self.radius
        )
        offsets = (self.points[candidates] - query[:, None, :]) / self.radius
        ratio = jnp.sqrt(jnp.sum(offsets * offsets, axis=-1))
        inside = candidate_valid & (ratio < 1.0)
        count = jnp.sum(inside, axis=1, dtype=jnp.int32)
        root = jnp.where(inside, jnp.sqrt(_wendland_c2(jnp.minimum(ratio, 1.0))), 0.0)
        if self.reconstruction == "shepard":
            positive = root * root
            total = jnp.sum(positive, axis=1)
            weights = positive / jnp.where(total > 0, total, 1.0)[:, None]
            full_rank = total > 0
            conditioning = jnp.where(full_rank, 1.0, jnp.inf).astype(points.dtype)
        else:
            design = jnp.concatenate(
                (
                    jnp.ones(offsets.shape[:2] + (1,), dtype=offsets.dtype),
                    jnp.prod(offsets[..., None, :] ** self.exponents, axis=-1),
                ),
                axis=-1,
            )
            factors, rank, conditioning, _ = weighted_svd_factors(
                design, root * root, inside
            )
            full_rank = rank == self.feature_count
            conditioning = jnp.where(full_rank, conditioning, jnp.inf).astype(
                points.dtype
            )
            weights = factors[:, 0, :]
        well_conditioned = full_rank & (conditioning <= self.condition_limit)
        status = jnp.where(
            ~finite,
            int(FieldQueryStatus.NONFINITE),
            jnp.where(
                ~complete,
                int(FieldQueryStatus.LOCATION_FAILED),
                jnp.where(
                    count < self.minimum_neighbors,
                    int(FieldQueryStatus.OUTSIDE_SUPPORT),
                    jnp.where(
                        well_conditioned,
                        int(FieldQueryStatus.VALID),
                        int(FieldQueryStatus.ILL_CONDITIONED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        valid = status == int(FieldQueryStatus.VALID)
        route = GatherStencil(
            indices=candidates,
            weights=jnp.where(valid[:, None] & inside, weights, 0.0),
            source_size=self.points.shape[0],
            valid=inside,
            support=valid,
        )
        return route, FieldQueryEvidence(
            status, conditioning, count, kernel_id=self.kernel_id
        )

    def apply(self, route: GatherStencil, coefficients: Array, /) -> Array:
        return linear_apply(route.relation, route.weights, coefficients)

    def transpose(self, route: GatherStencil, cotangent: Array, /) -> Array:
        return linear_transpose_apply(route.relation, route.weights, cotangent)

    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        del sites, side, cell_ids
        raise ValueError("Single-valued point-cloud reconstructions have no trace cells.")


def prepare_point_cloud_field_reconstruction(
    discretization: PreparedPointCloudDiscretization,
    /,
    *,
    support_geometry: Any,
    radius: float,
    capacity: int,
    minimum_neighbors: int | None = None,
    value_port: ValuePort | None = None,
    support_tolerance: float = 1.0e-9,
    reconstruction: PointCloudReconstruction = "polynomial",
) -> PreparedFieldReconstruction:
    """Prepare an evidenced moving least-squares reconstruction of point values.

    The fit uses the point cloud's declared polynomial degree and condition
    limit. `radius` is the neighborhood radius and `capacity` bounds the
    candidate cloud points gathered per query from the prepared hierarchy
    (queries whose radius box holds more candidates are `LOCATION_FAILED`).
    `minimum_neighbors` (at least the feature count, which is the default)
    gates `OUTSIDE_SUPPORT`. `support_geometry` is the explicit region the
    cloud samples; every cloud point must lie in it. The reconstruction is
    `C^2` with smooth pieces, values only, and support coverage is `partial`.
    Coefficients are point values with shape `(points, *value_port.event_shape)`.
    Explicit `reconstruction="shepard"` instead uses degree-zero positive
    normalized Wendland weights. It reproduces constants, not the cloud's
    polynomial degree, and permits minimum_neighbors=1 without downgrading a
    polynomial fit after failure. The selector is part of kernel provenance.
    """
    from ..geometry import CompiledGeometry, GeometryKind

    if not isinstance(discretization, PreparedPointCloudDiscretization):
        raise TypeError("discretization must be a PreparedPointCloudDiscretization.")
    if not isinstance(support_geometry, CompiledGeometry):
        raise TypeError("support_geometry must be a CompiledGeometry.")
    cloud = np.asarray(discretization.points, dtype=np.float64)
    count, dimension = cloud.shape
    if support_geometry.kind is not GeometryKind.REGION:
        raise ValueError("support_geometry must be a region geometry.")
    if support_geometry.ambient_dimension != dimension:
        raise ValueError("support_geometry dimension must equal the cloud dimension.")
    tolerance = float(support_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("support_tolerance must be finite and non-negative.")
    if np.any(
        np.asarray(support_geometry.boundary_field(jnp.asarray(cloud))) > tolerance
    ):
        raise ValueError("Point-cloud points lie outside the support geometry.")
    radius_ = float(radius)
    if not isfinite(radius_) or radius_ <= 0.0:
        raise ValueError("radius must be finite and positive.")
    if isinstance(capacity, bool) or not isinstance(capacity, Integral):
        raise TypeError("capacity must be an int.")
    reconstruction_ = parse(reconstruction, PointCloudReconstruction, "reconstruction")
    degree = (
        0
        if reconstruction_ == "shepard"
        else discretization.plan.stencil.polynomial_degree
    )
    features = comb(dimension + degree, degree)
    minimum = features if minimum_neighbors is None else minimum_neighbors
    if isinstance(minimum, bool) or not isinstance(minimum, Integral):
        raise TypeError("minimum_neighbors must be an int or None.")
    if minimum < features:
        raise ValueError(
            f"minimum_neighbors must be at least the {features} polynomial features."
        )
    if not features <= minimum <= capacity <= count:
        raise ValueError(
            "capacity must lie between minimum_neighbors and the cloud size."
        )
    if value_port is None:
        components: tuple[int, ...] = ()
    elif isinstance(value_port, ValuePort):
        components = value_port.event_shape
    else:
        raise TypeError("value_port must be a ValuePort or None.")
    (space,) = discretization.field_spaces
    field_space_id = canonical_fingerprint(
        {
            "kind": "point-cloud-field-space",
            "discretization": discretization.prepared_id,
            "space": space.field_space_id,
            "components": list(components),
        }
    )
    kernel = PointCloudFieldReconstructionKernel(
        cloud,
        degree,
        radius=radius_,
        capacity=int(capacity),
        minimum_neighbors=int(minimum),
        condition_limit=discretization.plan.stencil.condition_limit,
        field_space_id=field_space_id,
        source_owner_id=discretization.prepared_id,
        reconstruction=reconstruction_,
    )
    port = (
        ValuePort(
            space.name,
            event_shape=(),
            component_ids=(space.name,),
            representation="point-cloud-field",
            space_id=field_space_id,
        )
        if value_port is None
        else value_port
    )
    return PreparedFieldReconstruction(
        kernel,
        support_geometry=support_geometry,
        value_port=port,
        regularity=DerivativeRegularity.piecewise_smooth(continuity=2),
        trace_policy=FieldTracePolicy("single-valued"),
        coefficient_shape=(count, *components),
        physical_dimension=dimension,
        maximum_derivative_order=0,
        field_space_id=field_space_id,
        support_id=canonical_fingerprint(
            {
                "kind": "point-cloud-neighborhood-support",
                "support": discretization.support.support_id,
                "radius": radius_,
                "minimum_neighbors": int(minimum),
            }
        ),
    )


__all__ = [
    "PointCloudFieldReconstructionKernel",
    "PointCloudReconstruction",
    "prepare_point_cloud_field_reconstruction",
]
