#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import BoundaryAtlas, BoundaryFrame


def _positive_finite(name: str, value: float, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _metric_inverse(metric: Array, dimension: int, /) -> tuple[Array, Array]:
    if dimension == 1:
        determinant = metric[..., 0, 0]
        return (1.0 / determinant)[..., None, None], determinant
    if dimension == 2:
        first = metric[..., 0, 0]
        off_diagonal = metric[..., 0, 1]
        second = metric[..., 1, 1]
        determinant = first * second - off_diagonal * off_diagonal
        inverse = jnp.stack(
            (second, -off_diagonal, -off_diagonal, first), axis=-1
        ).reshape(metric.shape)
        return inverse / determinant[..., None, None], determinant
    if dimension == 3:
        first = metric[..., 0, :]
        second = metric[..., 1, :]
        third = metric[..., 2, :]
        cofactors = jnp.stack(
            (jnp.cross(second, third), jnp.cross(third, first), jnp.cross(first, second)),
            axis=-2,
        )
        determinant = jnp.sum(first * cofactors[..., 0, :], axis=-1)
        return jnp.swapaxes(cofactors, -1, -2) / determinant[..., None, None], determinant
    raise ValueError("Manifold metrics support parameter dimensions one through three.")


def _mapping_hessian(
    atlas: BoundaryAtlas, chart_indices: Array, reference: Array, /
) -> Array:
    leading = chart_indices.shape
    flat_indices = chart_indices.reshape((-1,))
    flat_reference = reference.reshape((-1, atlas.reference_dimension))
    values = jax.vmap(
        lambda index, coordinate: jax.jacfwd(
            jax.jacfwd(lambda point: atlas.mapping.map(index, point))
        )(coordinate)
    )(flat_indices, flat_reference)
    return values.reshape(
        (
            *leading,
            atlas.ambient_dimension,
            atlas.reference_dimension,
            atlas.reference_dimension,
        )
    )


class ParametricBasisPayload(StrictModule):
    """Values and first/second parameter derivatives from one basis provider."""

    values: Array
    gradients: Array
    hessians: Array
    provider_revision: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        gradients: ArrayLike,
        hessians: ArrayLike,
        /,
        *,
        provider_revision: str,
    ):
        values_ = jnp.asarray(values)
        gradients_ = jnp.asarray(gradients)
        hessians_ = jnp.asarray(hessians)
        revision = str(provider_revision)
        if values_.ndim < 1:
            raise ValueError("Basis values require a trailing basis-function axis.")
        if gradients_.shape[:-1] != values_.shape:
            raise ValueError("Basis gradients must append one parameter axis to values.")
        dimension = int(gradients_.shape[-1])
        if dimension <= 0 or hessians_.shape != (*values_.shape, dimension, dimension):
            raise ValueError("Basis Hessians must append two square parameter axes.")
        if not revision:
            raise ValueError("provider_revision must be non-empty.")
        if any(
            jnp.issubdtype(value.dtype, jnp.complexfloating)
            for value in (values_, gradients_, hessians_)
        ):
            raise TypeError("Manifold basis payloads must be real.")
        dtype = jnp.result_type(values_, gradients_, hessians_, float)
        self.values = values_.astype(dtype)
        self.gradients = gradients_.astype(dtype)
        self.hessians = hessians_.astype(dtype)
        self.provider_revision = revision
        self.payload_id = canonical_fingerprint(
            {
                "kind": "iga-parametric-basis-payload",
                "values": array_tree_fingerprint(np.asarray(values_)),
                "gradients": array_tree_fingerprint(np.asarray(gradients_)),
                "hessians": array_tree_fingerprint(np.asarray(hessians_)),
                "provider_revision": revision,
            }
        )

    @property
    def parametric_dimension(self) -> int:
        return int(self.gradients.shape[-1])


@runtime_checkable
class ManifoldBasisPayloadProvider(Protocol):
    """Exact provider contract consumed by ``ManifoldBasisProviderAdapter``."""

    def basis_payload(
        self, chart_indices: Array, reference: Array, /, *, maximum_derivative: int
    ) -> ParametricBasisPayload:
        """Return derivatives in atlas parameter coordinates."""
        ...


class ManifoldPointGeometry(StrictModule):
    """Full-rank rectangular differential geometry at atlas query points."""

    frame: BoundaryFrame
    differential: Array
    mapping_hessian: Array
    metric: Array
    inverse_metric: Array
    tangent_projector: Array
    christoffel: Array
    metric_rank_ratio: Array
    measure: Array
    parametric_dimension: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def tangent_gradient(self, parametric_gradient: ArrayLike, /) -> Array:
        gradient = jnp.asarray(parametric_gradient)
        if gradient.shape != self.metric.shape[:-1]:
            raise ValueError(
                "A parametric gradient must match the geometry query shape and dimension."
            )
        raised = contract("...ij,...j->...i", self.inverse_metric, gradient)
        return contract("...ai,...i->...a", self.differential, raised)

    def conormal(self, parametric_covector: ArrayLike, /) -> Array:
        covector = jnp.asarray(parametric_covector)
        if covector.shape != self.metric.shape[:-1]:
            raise ValueError(
                "A conormal covector must match the geometry query shape and dimension."
            )
        raised = contract("...ij,...j->...i", self.inverse_metric, covector)
        physical = contract("...ai,...i->...a", self.differential, raised)
        norm_squared = contract("...i,...i->...", covector, raised)
        physical = eqx.error_if(
            physical,
            jnp.any(~jnp.isfinite(norm_squared)) | jnp.any(norm_squared <= 0.0),
            "A manifold conormal covector must be finite and nonzero.",
        )
        return physical / jnp.sqrt(norm_squared)[..., None]

    def covariant_hessian(
        self, parametric_gradient: ArrayLike, parametric_hessian: ArrayLike, /
    ) -> Array:
        gradient = jnp.asarray(parametric_gradient)
        hessian = jnp.asarray(parametric_hessian)
        if gradient.shape != self.metric.shape[:-1] or hessian.shape != self.metric.shape:
            raise ValueError(
                "Scalar gradient/Hessian shapes must match the manifold query geometry."
            )
        correction = contract("...kij,...k->...ij", self.christoffel, gradient)
        return hessian - correction

    def ambient_covariant_hessian(
        self, parametric_gradient: ArrayLike, parametric_hessian: ArrayLike, /
    ) -> Array:
        covariant = self.covariant_hessian(parametric_gradient, parametric_hessian)
        raised = contract(
            "...ik,...kl,...lj->...ij",
            self.inverse_metric,
            covariant,
            self.inverse_metric,
        )
        return contract(
            "...ai,...ij,...bj->...ab", self.differential, raised, self.differential
        )


class AtlasManifold(StrictModule, NonTrainableState):
    """Differential-geometry adapter over the shared ``BoundaryAtlas`` contract."""

    atlas: BoundaryAtlas
    rank_tolerance: float = eqx.field(static=True)
    atlas_id: str = eqx.field(static=True)

    def __init__(self, atlas: BoundaryAtlas, /, *, rank_tolerance: float = 1.0e-10):
        if not isinstance(atlas, BoundaryAtlas):
            raise TypeError("atlas must be a BoundaryAtlas.")
        if atlas.reference_dimension not in (1, 2, 3):
            raise ValueError(
                "Atlas manifold parameter dimension must be one through three."
            )
        if atlas.ambient_dimension <= atlas.reference_dimension:
            raise ValueError(
                "Atlas manifolds require a rectangular embedded differential."
            )
        tolerance = _positive_finite("rank_tolerance", rank_tolerance)
        self.atlas = atlas
        self.rank_tolerance = tolerance
        self.atlas_id = canonical_fingerprint(
            {
                "kind": "iga-atlas-manifold",
                "source": atlas.source_id,
                "charts": atlas.num_charts,
                "reference_dimension": atlas.reference_dimension,
                "ambient_dimension": atlas.ambient_dimension,
                "rank_tolerance": tolerance,
            }
        )

    def evaluate(
        self, chart_indices: ArrayLike, reference: ArrayLike, /
    ) -> ManifoldPointGeometry:
        indices, reference_ = self.atlas._validate_inputs(chart_indices, reference)
        frame = self.atlas.frame(indices, reference_)
        differential = self.atlas.differential(indices, reference_)
        mapping_hessian = _mapping_hessian(self.atlas, indices, reference_)
        metric = contract("...ai,...aj->...ij", differential, differential)
        eigenvalues = jnp.linalg.eigvalsh(metric)
        maximum = eigenvalues[..., -1]
        minimum = eigenvalues[..., 0]
        tiny = jnp.asarray(jnp.finfo(metric.dtype).tiny, dtype=metric.dtype)
        rank_ratio = minimum / jnp.maximum(maximum, tiny)
        differential = eqx.error_if(
            differential,
            jnp.any(~jnp.isfinite(rank_ratio))
            | jnp.any(maximum <= 0.0)
            | jnp.any(rank_ratio <= self.rank_tolerance),
            "Atlas differential is not full column rank at every query point.",
        )
        inverse_metric, determinant = _metric_inverse(
            metric, self.atlas.reference_dimension
        )
        measure = jnp.sqrt(determinant)
        reported_measure = jnp.asarray(frame.jacobian, dtype=measure.dtype)
        scale = jnp.maximum(measure, jnp.asarray(1.0, dtype=measure.dtype))
        consistency_tolerance = 64.0 * jnp.sqrt(jnp.finfo(measure.dtype).eps)
        measure = eqx.error_if(
            measure,
            jnp.any(~jnp.isfinite(measure))
            | jnp.any(measure <= 0.0)
            | jnp.any(
                jnp.abs(reported_measure - measure) > consistency_tolerance * scale
            ),
            "BoundaryAtlas Jacobian is inconsistent with its rectangular metric.",
        )
        tangent_projector = contract(
            "...ai,...ij,...bj->...ab", differential, inverse_metric, differential
        )
        christoffel = contract(
            "...kl,...al,...aij->...kij",
            inverse_metric,
            differential,
            mapping_hessian,
        )
        return ManifoldPointGeometry(
            frame,
            differential,
            mapping_hessian,
            metric,
            inverse_metric,
            tangent_projector,
            christoffel,
            rank_ratio,
            measure,
            self.atlas.reference_dimension,
            self.atlas.ambient_dimension,
            self.atlas.source_id,
        )


class ManifoldBasisRealization(StrictModule):
    """A basis payload transformed into tangent and covariant derivatives."""

    values: Array
    tangent_gradients: Array
    covariant_hessians: Array
    ambient_covariant_hessians: Array
    geometry: ManifoldPointGeometry
    payload_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)


def realize_manifold_basis(
    payload: ParametricBasisPayload, geometry: ManifoldPointGeometry, /
) -> ManifoldBasisRealization:
    if not isinstance(payload, ParametricBasisPayload):
        raise TypeError("payload must be a ParametricBasisPayload.")
    if not isinstance(geometry, ManifoldPointGeometry):
        raise TypeError("geometry must be a ManifoldPointGeometry.")
    dimension = geometry.parametric_dimension
    query_shape = geometry.metric.shape[:-2]
    if payload.parametric_dimension != dimension:
        raise ValueError("Basis and manifold parameter dimensions differ.")
    if payload.values.shape[:-1] != query_shape:
        raise ValueError("Basis and manifold payload query shapes differ.")
    raised_gradient = contract(
        "...ij,...nj->...ni", geometry.inverse_metric, payload.gradients
    )
    tangent_gradients = contract(
        "...ai,...ni->...na", geometry.differential, raised_gradient
    )
    correction = contract("...kij,...nk->...nij", geometry.christoffel, payload.gradients)
    covariant = payload.hessians - correction
    raised_hessian = contract(
        "...ik,...nkl,...lj->...nij",
        geometry.inverse_metric,
        covariant,
        geometry.inverse_metric,
    )
    ambient = contract(
        "...ai,...nij,...bj->...nab",
        geometry.differential,
        raised_hessian,
        geometry.differential,
    )
    return ManifoldBasisRealization(
        payload.values,
        tangent_gradients,
        covariant,
        ambient,
        geometry,
        payload.payload_id,
        canonical_fingerprint(
            {
                "kind": "iga-manifold-basis-realization",
                "payload": payload.payload_id,
                "atlas": geometry.source_id,
                "parameter_dimension": dimension,
                "ambient_dimension": geometry.ambient_dimension,
            }
        ),
    )


class ManifoldBasisProviderAdapter(StrictModule, NonTrainableState):
    """Bind an exact parametric payload provider to an atlas manifold."""

    provider: ManifoldBasisPayloadProvider
    manifold: AtlasManifold
    provider_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider: ManifoldBasisPayloadProvider,
        manifold: AtlasManifold,
        /,
        *,
        provider_id: str,
    ):
        if not isinstance(provider, ManifoldBasisPayloadProvider):
            raise TypeError("provider must implement ManifoldBasisPayloadProvider.")
        if not isinstance(manifold, AtlasManifold):
            raise TypeError("manifold must be an AtlasManifold.")
        identifier = str(provider_id)
        if not identifier:
            raise ValueError("provider_id must be non-empty.")
        self.provider = provider
        self.manifold = manifold
        self.provider_id = identifier
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "iga-manifold-basis-provider-adapter",
                "provider": identifier,
                "manifold": manifold.atlas_id,
            }
        )

    def realize(
        self, chart_indices: ArrayLike, reference: ArrayLike, /
    ) -> ManifoldBasisRealization:
        indices, reference_ = self.manifold.atlas._validate_inputs(
            chart_indices, reference
        )
        payload = self.provider.basis_payload(indices, reference_, maximum_derivative=2)
        if not isinstance(payload, ParametricBasisPayload):
            raise TypeError("Basis provider returned an invalid manifold payload.")
        return realize_manifold_basis(
            payload, self.manifold.evaluate(indices, reference_)
        )


class SurfaceChartTransition(StrictModule, NonTrainableState):
    """Matched overlap samples used to prove one oriented atlas transition."""

    left_chart: int = eqx.field(static=True)
    right_chart: int = eqx.field(static=True)
    left_reference: Array
    right_reference: Array
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_chart: int,
        right_chart: int,
        left_reference: ArrayLike,
        right_reference: ArrayLike,
        /,
    ):
        left = int(left_chart)
        right = int(right_chart)
        left_points = jnp.asarray(left_reference, dtype=float)
        right_points = jnp.asarray(right_reference, dtype=float)
        if left < 0 or right < 0 or left == right:
            raise ValueError("A chart transition requires two distinct chart indices.")
        if (
            left_points.ndim != 2
            or left_points.shape[1] != 2
            or right_points.shape != left_points.shape
            or left_points.shape[0] == 0
        ):
            raise ValueError(
                "Surface transition references must have matching shape (n, 2)."
            )
        if np.any(~np.isfinite(np.asarray(left_points))) or np.any(
            ~np.isfinite(np.asarray(right_points))
        ):
            raise ValueError("Surface transition references must be finite.")
        self.left_chart = left
        self.right_chart = right
        self.left_reference = left_points
        self.right_reference = right_points
        self.transition_id = canonical_fingerprint(
            {
                "kind": "iga-surface-chart-transition",
                "left_chart": left,
                "right_chart": right,
                "left_reference": array_tree_fingerprint(np.asarray(left_points)),
                "right_reference": array_tree_fingerprint(np.asarray(right_points)),
            }
        )


class ManifoldQualificationEvidence(StrictModule, NonTrainableState):
    """Dimensionless sampled evidence supporting a surface embedding claim."""

    minimum_metric_rank_ratio: Array
    minimum_measure_ratio: Array
    maximum_transition_residual: Array
    minimum_oriented_normal_dot: Array
    sample_count: int = eqx.field(static=True)
    transition_sample_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class SurfaceEmbeddingCertificate(StrictModule, NonTrainableState):
    """Checked full-rank and orientable surface-atlas certificate."""

    evidence: ManifoldQualificationEvidence
    source_id: str = eqx.field(static=True)
    chart_count: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    trace_tolerance: float = eqx.field(static=True)
    orientation_tolerance: float = eqx.field(static=True)
    transition_ids: tuple[str, ...] = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def assert_matches(self, atlas: BoundaryAtlas, /) -> None:
        if not isinstance(atlas, BoundaryAtlas):
            raise TypeError("atlas must be a BoundaryAtlas.")
        if atlas.source_id != self.source_id or atlas.num_charts != self.chart_count:
            raise ValueError("Surface embedding certificate does not match this atlas.")
        if atlas.reference_dimension != 2 or atlas.ambient_dimension != 3:
            raise ValueError("Surface embedding certificate requires a surface in 3D.")


def _connected_chart_graph(
    chart_count: int, transitions: Sequence[SurfaceChartTransition]
) -> bool:
    if chart_count == 1:
        return True
    adjacency = [set() for _ in range(chart_count)]
    for transition in transitions:
        if transition.left_chart >= chart_count or transition.right_chart >= chart_count:
            raise ValueError("Surface transition chart index is outside the atlas.")
        adjacency[transition.left_chart].add(transition.right_chart)
        adjacency[transition.right_chart].add(transition.left_chart)
    visited = {0}
    frontier = [0]
    while frontier:
        chart = frontier.pop()
        for neighbour in adjacency[chart] - visited:
            visited.add(neighbour)
            frontier.append(neighbour)
    return len(visited) == chart_count


def certify_surface_embedding(
    atlas: BoundaryAtlas,
    chart_indices: ArrayLike,
    reference: ArrayLike,
    /,
    *,
    transitions: Sequence[SurfaceChartTransition] = (),
    rank_tolerance: float = 1.0e-10,
    trace_tolerance: float = 1.0e-9,
    orientation_tolerance: float = 1.0e-8,
) -> SurfaceEmbeddingCertificate:
    """Certify a sampled surface atlas; unsupported or incomplete evidence fails closed."""
    if not isinstance(atlas, BoundaryAtlas):
        raise TypeError("atlas must be a BoundaryAtlas.")
    if atlas.reference_dimension != 2 or atlas.ambient_dimension != 3:
        raise ValueError("Surface embedding certification requires 2D charts in 3D.")
    rank_tol = _positive_finite("rank_tolerance", rank_tolerance)
    trace_tol = _positive_finite("trace_tolerance", trace_tolerance)
    orientation_tol = _positive_finite("orientation_tolerance", orientation_tolerance)
    transition_values = tuple(transitions)
    if any(not isinstance(value, SurfaceChartTransition) for value in transition_values):
        raise TypeError("transitions must contain SurfaceChartTransition values.")
    if not _connected_chart_graph(atlas.num_charts, transition_values):
        raise ValueError("Orientability evidence must connect every surface chart.")
    indices, reference_ = atlas._validate_inputs(chart_indices, reference)
    if indices.ndim != 1 or indices.shape[0] == 0:
        raise ValueError("Surface qualification samples must form one non-empty batch.")
    index_host = np.asarray(indices)
    if set(index_host.tolist()) != set(range(atlas.num_charts)):
        raise ValueError("Surface qualification samples must cover every atlas chart.")
    manifold = AtlasManifold(atlas, rank_tolerance=rank_tol)
    geometry = manifold.evaluate(indices, reference_)
    rank_ratio = np.asarray(geometry.metric_rank_ratio)
    measure = np.asarray(geometry.measure)
    coordinate_scale = max(
        float(np.max(np.linalg.norm(np.asarray(geometry.frame.origin), axis=-1))), 1.0
    )
    minimum_rank = float(np.min(rank_ratio))
    minimum_measure_ratio = float(np.min(measure) / coordinate_scale**2)
    if not np.isfinite(minimum_rank) or minimum_rank <= rank_tol:
        raise ValueError("Surface embedding rank qualification failed.")
    if not np.isfinite(minimum_measure_ratio) or minimum_measure_ratio <= rank_tol:
        raise ValueError("Surface embedding measure qualification failed.")
    maximum_transition_residual = 0.0
    minimum_normal_dot = 1.0
    transition_sample_count = 0
    for transition in transition_values:
        count = int(transition.left_reference.shape[0])
        left_indices = jnp.full((count,), transition.left_chart, dtype=jnp.int32)
        right_indices = jnp.full((count,), transition.right_chart, dtype=jnp.int32)
        left_frame = atlas.frame(left_indices, transition.left_reference)
        right_frame = atlas.frame(right_indices, transition.right_reference)
        left_origin = np.asarray(left_frame.origin)
        right_origin = np.asarray(right_frame.origin)
        transition_scale = max(
            float(np.max(np.linalg.norm(left_origin, axis=-1))),
            float(np.max(np.linalg.norm(right_origin, axis=-1))),
            1.0,
        )
        residual = float(
            np.max(np.linalg.norm(left_origin - right_origin, axis=-1)) / transition_scale
        )
        normal_dot = np.sum(
            np.asarray(left_frame.normal) * np.asarray(right_frame.normal), axis=-1
        )
        maximum_transition_residual = max(maximum_transition_residual, residual)
        minimum_normal_dot = min(minimum_normal_dot, float(np.min(normal_dot)))
        transition_sample_count += count
    if maximum_transition_residual > trace_tol:
        raise ValueError("Surface chart transition trace equality qualification failed.")
    if minimum_normal_dot < 1.0 - orientation_tol:
        raise ValueError("Surface chart orientations are inconsistent on an overlap.")
    evidence_id = canonical_fingerprint(
        {
            "kind": "iga-manifold-qualification-evidence",
            "source": atlas.source_id,
            "samples": array_tree_fingerprint(
                {"chart_indices": index_host, "reference": np.asarray(reference_)}
            ),
            "transitions": [value.transition_id for value in transition_values],
            "minimum_metric_rank_ratio": minimum_rank,
            "minimum_measure_ratio": minimum_measure_ratio,
            "maximum_transition_residual": maximum_transition_residual,
            "minimum_oriented_normal_dot": minimum_normal_dot,
        }
    )
    evidence = ManifoldQualificationEvidence(
        jnp.asarray(minimum_rank),
        jnp.asarray(minimum_measure_ratio),
        jnp.asarray(maximum_transition_residual),
        jnp.asarray(minimum_normal_dot),
        int(indices.shape[0]),
        transition_sample_count,
        evidence_id,
    )
    return SurfaceEmbeddingCertificate(
        evidence,
        atlas.source_id,
        atlas.num_charts,
        rank_tol,
        trace_tol,
        orientation_tol,
        tuple(value.transition_id for value in transition_values),
        canonical_fingerprint(
            {
                "kind": "iga-surface-embedding-certificate",
                "source": atlas.source_id,
                "charts": atlas.num_charts,
                "evidence": evidence_id,
                "rank_tolerance": rank_tol,
                "trace_tolerance": trace_tol,
                "orientation_tolerance": orientation_tol,
            }
        ),
    )


__all__ = [
    "AtlasManifold",
    "ManifoldBasisPayloadProvider",
    "ManifoldBasisProviderAdapter",
    "ManifoldBasisRealization",
    "ManifoldPointGeometry",
    "ManifoldQualificationEvidence",
    "ParametricBasisPayload",
    "SurfaceChartTransition",
    "SurfaceEmbeddingCertificate",
    "certify_surface_embedding",
    "realize_manifold_basis",
]
