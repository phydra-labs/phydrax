#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent collective acceptance of authored-patch native initial surfaces.

The premise is the actual source domain and generated owner rows, never an
accepted serial mesh or a fictitious subdivision predecessor. Local chart and
trim coverage are established by the geometry owner; cross-owner contacts use
bounded candidate packets and exact native predicates. Canonical content stays
on the existing sharded logical-array substrate.

This route does not implement within-patch distributed CDT/cavity ownership or
distributed PLC segment/facet recovery. Those are separate missing primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import combinations
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from numpy.typing import NDArray

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from .._physical import SpatialCoordinateContract
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry import CellGeometryStorageProjection
from ..discretization._cell_geometry_validity import certify_cell_geometry_validity
from ..geometry._mesh_certificates import (
    certify_global_embedding,
    certify_source_fidelity,
    MeshCertificateLimits,
)
from ..geometry._meshing_domain import MeshingDomainBoundarySource, PatchCurveUse
from ._audit_topology import _triangle_pairs_intersect
from ._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    SurfaceMeshingSpec,
)
from ._device_adaptation import _canonical_rows, _key_positions
from ._device_generation import DeviceGenerationLayout
from ._distributed_generation import (
    _consensus,
    DistributedSurfaceConstruction,
    PreparedDistributedSurfaceGeneration,
)
from ._domain import CompiledSurfaceDomain
from ._metric import interpolate_mesh_metric, metric_edge_lengths
from ._organization import MeshLabel, MeshPatch, RegionBoundaryEvidence
from ._result import AbstractCollectiveMeshTheorem
from ._surface_generation import SurfaceConstruction
from .providers._native_options import NativeSurfaceSchedule
from .providers._native_sources import NativeSurfaceSource


if TYPE_CHECKING:
    from ._result import CellMeshingResult

_CHECKS = (
    "authored_source_binding",
    "chart_trim_coverage",
    "continuous_source_fidelity",
    "local_embedding",
    "physical_controls",
    "unique_cell_ownership",
    "shared_vertex_geometry",
    "reciprocal_source_facets",
    "cross_owner_embedding",
)


@dataclass(frozen=True, slots=True)
class _LocalProof:
    certificate_ids: tuple[str, ...]
    patch_checks: NDArray[np.bool_]


def _failure(message: str, /) -> MeshingFailure:
    return MeshingFailure(
        MeshingFailureCategory.COMPLIANCE_FAILED,
        message,
        stage="distributed-initial-certification",
    )


def _source_binding(
    prepared: PreparedDistributedSurfaceGeneration | InitialCollectiveMeshEvidence,
    generated: DistributedSurfaceConstruction,
    /,
    *,
    partition_index: int | None = None,
) -> None:
    domain = prepared.compiled.domain
    target = prepared.specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 2
        or target.ambient_dimension != 3
        or target.geometry_order != 1
        or set((*families.required, *families.preferred)) != {"triangle"}
        or families.allowed_transitions
        or families.allow_mixed
    ):
        raise _failure(
            "Initial parametric publication requires its declared affine triangle source contract."
        )
    if (
        generated.source_id != domain.source_id
        or generated.source_revision != domain.source_revision
        or generated.specification_id != prepared.specification.specification_id
        or prepared.compiled.specification_id != generated.specification_id
    ):
        raise _failure(
            "Generated initial rows are not bound to the original source and specification."
        )
    construction = generated.construction
    part = jax.process_index() if partition_index is None else partition_index
    local_patches = prepared.compiled.patches[prepared.patch_owners == part]
    if construction is None:
        if local_patches.size or generated.vertex_ids.size or generated.cell_ids.size:
            raise _failure("An authored patch owner has no actual generated rows.")
        return
    nv, nc = construction.vertices.shape[0], construction.triangles.shape[0]
    if (
        generated.vertex_ids.shape != (nv,)
        or generated.vertex_owners.shape != (nv,)
        or generated.cell_ids.shape != (nc,)
        or np.unique(generated.vertex_ids).size != nv
        or np.unique(generated.cell_ids).size != nc
        or np.any(generated.vertex_ids < 0)
        or np.any(generated.vertex_ids >= generated.global_vertex_count)
        or np.any(generated.cell_ids < 0)
        or np.any(generated.cell_ids >= generated.global_cell_count)
        or np.any(generated.vertex_owners < 0)
        or np.any(generated.vertex_owners >= prepared.partition_count)
        or construction.triangle_patches.shape != (nc,)
        or construction.triangle_parameters.shape != (nc, 3, 2)
        or not np.array_equal(np.unique(construction.triangle_patches), local_patches)
    ):
        raise _failure(
            "Initial generated IDs, chart rows, or authored patch ownership are incomplete."
        )
    for (
        patch,
        charts,
        points,
        cells,
        boundary,
        provenance,
        restriction_required,
        restriction_vertices,
        restriction_edges,
        restriction_parameters,
    ) in construction.chart_triangulations:
        if patch not in local_patches:
            raise _failure("A generated chart belongs to another authored patch owner.")
        if (
            charts.ndim != 2
            or charts.shape[1] != 2
            or points.shape != (charts.shape[0], 3)
            or cells.ndim != 2
            or cells.shape[1] != 3
            or np.any(cells < 0)
            or np.any(cells >= charts.shape[0])
        ):
            raise _failure(
                "A generated source chart has invalid physical/chart incidence."
            )
        if (
            restriction_required.shape != (charts.shape[0],)
            or restriction_required.dtype != np.bool_
            or restriction_vertices.ndim != 1
            or restriction_edges.shape != (restriction_vertices.size, 2)
            or restriction_parameters.shape != (restriction_vertices.size, 2)
        ):
            raise _failure(
                "A generated source chart lost rational restriction authority."
            )
    dimensions = construction.vertex_source_dimensions
    indices = construction.vertex_source_indices
    parameters = construction.vertex_parameters
    if (
        dimensions.shape != (nv,)
        or indices.shape != (nv,)
        or parameters.shape != (nv, 2)
        or np.any(dimensions < 0)
        or np.any(dimensions > 2)
    ):
        raise _failure("Initial vertex source-stratum facts are incomplete.")
    for degree in range(3):
        rows = np.flatnonzero(dimensions == degree)
        upper = (domain.corner_points.shape[0], len(domain.curves), len(domain.patches))[
            degree
        ]
        if np.any(indices[rows] < 0) or np.any(indices[rows] >= upper):
            raise _failure(
                "Initial vertex source-stratum identities are outside their authored domain."
            )
        if degree == 0:
            exact = domain.corner_points[indices[rows]]
        elif degree == 1:
            normalized = np.empty((rows.size, 1), dtype=np.float64)
            for curve in np.unique(indices[rows]).tolist():
                selected = np.flatnonzero(indices[rows] == curve)
                owner_patch, owner_loop, owner_position = domain.curve_owners[curve]
                use = domain.patches[owner_patch].loops[owner_loop][owner_position]
                if not isinstance(use, PatchCurveUse):
                    raise _failure(
                        "Initial curve facts require their actual authored curve owner."
                    )
                normalized[selected, 0] = (parameters[rows[selected], 0] - use.first) / (
                    use.last - use.first
                )
            if np.any(~np.isfinite(normalized)) or np.any(
                (normalized <= 0.0) | (normalized >= 1.0)
            ):
                raise _failure(
                    "Initial curve interiors have invalid authored parameters."
                )
            exact = np.asarray(
                domain.curve_atlas.map(
                    jnp.asarray(indices[rows], dtype=jnp.int32),
                    jnp.asarray(normalized, dtype=jnp.float64),
                ),
                dtype=np.float64,
            ).reshape((-1, 3))
        else:
            if np.any(~np.isfinite(parameters[rows])) or not np.all(
                np.isin(indices[rows], local_patches)
            ):
                raise _failure(
                    "Initial patch vertices do not belong to their actual owner charts."
                )
            exact = domain.evaluate(indices[rows], parameters[rows])
        if not np.allclose(
            construction.vertices[rows],
            exact,
            rtol=0.0,
            atol=128 * np.finfo(np.float64).eps * domain.scale,
        ):
            raise _failure(
                "Initial vertex coordinates do not reproduce their authored source-stratum facts."
            )


def _physical_controls(
    prepared: PreparedDistributedSurfaceGeneration | InitialCollectiveMeshEvidence,
    construction: SurfaceConstruction,
    minimum_angle: float,
    maximum_normal_angle: float,
    /,
) -> None:
    """Recompute physical quantities from source charts and actual coordinates."""
    compiled, domain = prepared.compiled, prepared.compiled.domain
    points = construction.vertices[construction.triangles]
    charts = construction.triangle_parameters
    first = np.roll(points, -1, axis=1) - points
    second = np.roll(points, -2, axis=1) - points
    squared_first = np.sum(first * first, axis=2)
    squared_second = np.sum(second * second, axis=2)
    product = np.sum(first * second, axis=2)
    metric = compiled.source_sizing.metric
    if metric is not None:
        values, _ = metric.sample(
            construction.vertices, prepared.specification.limits.maximum_work_units
        )
        edges = construction.triangles[:, np.asarray(((0, 1), (1, 2), (2, 0)))].reshape(
            (-1, 2)
        )
        bound = metric.control.maximum_metric_edge_length or 1.0
        if np.any(
            np.asarray(metric_edge_lengths(values, construction.vertices, edges)) > bound
        ):
            raise _failure("Initial physical metric edge bound is not satisfied.")
        mean = np.asarray(
            interpolate_mesh_metric(
                values[construction.triangles],
                np.full(construction.triangles.shape, 1.0 / 3.0, dtype=np.float64),
            )
        )
        squared_first = np.sum(first * (mean[:, None] @ first[..., None])[..., 0], axis=2)
        squared_second = np.sum(
            second * (mean[:, None] @ second[..., None])[..., 0], axis=2
        )
        product = np.sum(first * (mean[:, None] @ second[..., None])[..., 0], axis=2)
    cosine = product / np.sqrt(
        np.maximum(squared_first * squared_second, np.finfo(np.float64).tiny)
    )
    if np.any(
        np.arccos(np.clip(cosine, -1.0, 1.0))
        < minimum_angle - 256 * np.finfo(np.float64).eps
    ):
        raise _failure("Initial physical minimum-angle control is not satisfied.")
    for patch in np.unique(construction.triangle_patches).tolist():
        rows = np.flatnonzero(construction.triangle_patches == patch)
        uv = charts[rows]
        center = np.mean(uv, axis=1)
        source_center = domain.evaluate(np.full(rows.size, patch, dtype=np.int64), center)
        sizes = compiled.source_sizing.evaluate(domain, patch, center, source_center)
        midpoint_uv = 0.5 * (np.roll(uv, -1, axis=1) + np.roll(uv, -2, axis=1))
        midpoint = domain.evaluate(
            np.full(3 * rows.size, patch, dtype=np.int64), midpoint_uv.reshape((-1, 2))
        ).reshape((-1, 3, 3))
        ends_a, ends_b = (
            np.roll(points[rows], -1, axis=1),
            np.roll(points[rows], -2, axis=1),
        )
        lengths = np.linalg.norm(midpoint - ends_a, axis=2) + np.linalg.norm(
            midpoint - ends_b, axis=2
        )
        if np.any(lengths > sizes[:, None] * (1.0 + 1.0e-9)):
            raise _failure("Initial physical source sizing control is not satisfied.")
        normals, regular = domain.oriented_normals(
            np.full(rows.size, patch, dtype=np.int64), center
        )
        face = np.cross(
            points[rows, 1] - points[rows, 0], points[rows, 2] - points[rows, 0]
        )
        if np.any(~regular) or np.any(np.sum(face * normals, axis=1) <= 0.0):
            raise _failure(
                "Initial physical cells do not preserve their authored source orientation."
            )
        requested = min(
            maximum_normal_angle,
            float(compiled.patch_normal_angles[np.searchsorted(compiled.patches, patch)]),
        )
        if np.isfinite(requested) and np.any(
            domain.normal_turn_bounds(patch, uv) > requested
        ):
            raise _failure("Initial continuous normal-turn control is not satisfied.")


def _source_feature_controls(
    prepared: PreparedDistributedSurfaceGeneration | InitialCollectiveMeshEvidence,
    construction: SurfaceConstruction,
    /,
) -> None:
    """Independently certify the actual native parameter chains of source curves."""
    compiled, domain = prepared.compiled, prepared.compiled.domain
    for curve, parameters in construction.curve_parameters:
        if (
            parameters.ndim != 1
            or parameters.size < 2
            or parameters[0] != 0.0
            or parameters[-1] != 1.0
            or np.any(~np.isfinite(parameters))
            or np.any(np.diff(parameters) <= 0.0)
        ):
            raise _failure(
                "Initial source feature chains alter their authored parameter coverage."
            )
        rows = np.flatnonzero(construction.curve_edge_curves == curve)
        if rows.size != parameters.size - 1:
            raise _failure(
                "Initial source feature chains omit or duplicate actual edge segments."
            )
        points = np.array(
            domain.curve_atlas.map(
                jnp.full((parameters.size,), curve, dtype=jnp.int32),
                jnp.asarray(parameters[:, None], dtype=jnp.float64),
            ),
            dtype=np.float64,
            copy=True,
        ).reshape((-1, 3))
        source = domain.curves[curve]
        points[0], points[-1] = (
            domain.corner_points[source.start],
            domain.corner_points[source.end],
        )
        expected = np.stack((points[:-1], points[1:]), axis=1)
        actual = construction.vertices[construction.curve_edges[rows]]
        if not np.allclose(
            actual, expected, rtol=0.0, atol=128 * np.finfo(np.float64).eps * domain.scale
        ):
            raise _failure(
                "Initial feature edge incidence differs from its actual authored curve chain."
            )
        row = np.searchsorted(compiled.curves, curve)
        if np.any(
            np.linalg.norm(actual[:, 1] - actual[:, 0], axis=1)
            > compiled.curve_sizes[row] * (1.0 + 1.0e-9)
        ):
            raise _failure(
                "Initial feature edges violate their original physical source sizing control."
            )
        if np.any(
            domain.curve_interpolation_bounds(curve, parameters)
            > compiled.curve_deviations[row]
        ):
            raise _failure(
                "Initial feature edges lack their requested continuous source fidelity bound."
            )


def _local_proof(
    prepared: PreparedDistributedSurfaceGeneration | InitialCollectiveMeshEvidence,
    generated: DistributedSurfaceConstruction,
    limits: MeshCertificateLimits,
    minimum_angle: float,
    maximum_normal_angle: float,
    /,
    *,
    partition_index: int | None = None,
) -> _LocalProof:
    _source_binding(prepared, generated, partition_index=partition_index)
    construction = generated.construction
    patch_checks = np.zeros(prepared.compiled.patches.shape, dtype=np.bool_)
    if construction is None:
        return _LocalProof((), patch_checks)
    mesh = CellMesh(
        construction.vertices,
        (
            CellBlock(
                "surface",
                "triangle",
                construction.triangles,
                global_ids=generated.cell_ids,
            ),
        ),
        vertex_global_ids=generated.vertex_ids,
    )
    geometry = CellGeometrySpec.affine(mesh)
    validity = certify_cell_geometry_validity(geometry, mesh=mesh)
    embedding = certify_global_embedding(mesh, geometry, validity, limits=limits)
    if embedding.status != "certified":
        raise _failure(
            "Initial owner physical embedding has violated or unresolved geometry predicates."
        )
    certificates = [validity.certificate_id, embedding.certificate_id]
    for patch in np.unique(construction.triangle_patches).tolist():
        rows = np.flatnonzero(construction.triangle_patches == patch)
        patch_mesh = CellMesh(
            construction.vertices,
            (
                CellBlock(
                    "surface",
                    "triangle",
                    construction.triangles[rows],
                    global_ids=generated.cell_ids[rows],
                ),
            ),
            vertex_global_ids=generated.vertex_ids,
        )
        patch_geometry = CellGeometrySpec.affine(patch_mesh)
        source = MeshingDomainBoundarySource(
            prepared.compiled.domain,
            (patch,),
            chart_triangulations=tuple(
                value for value in construction.chart_triangulations if value[0] == patch
            ),
        )
        cover = source.boundary_chart_cover(limits.maximum_source_samples)
        if not cover.complete or cover.semantics != "certified":
            raise _failure(
                "Initial chart chain does not independently cover the authored source trims."
            )
        tolerance = float(
            prepared.compiled.patch_deviations[
                np.searchsorted(prepared.compiled.patches, patch)
            ]
        )
        # An unbounded request still requires a finite independently established
        # continuous enclosure, not an infinite or sampled success criterion.
        if not np.isfinite(tolerance):
            tolerance = (
                float(np.max(cover.deviation_bounds, initial=0.0))
                + 1024 * np.finfo(np.float64).eps * prepared.compiled.domain.scale
            )
        fidelity = certify_source_fidelity(
            patch_mesh, patch_geometry, source, tolerance=tolerance, limits=limits
        )
        if fidelity.status != "certified":
            raise _failure(
                "Initial cells lack continuous two-sided fidelity to their actual authored source."
            )
        certificates.extend((cover.cover_id, fidelity.certificate_id))
        patch_checks[np.searchsorted(prepared.compiled.patches, patch)] = True
    _source_feature_controls(prepared, construction)
    _physical_controls(prepared, construction, minimum_angle, maximum_normal_angle)
    return _LocalProof(tuple(certificates), patch_checks)


def _logical_initial_content(
    prepared: PreparedDistributedSurfaceGeneration,
    generated: DistributedSurfaceConstruction,
    local: _LocalProof,
    sharding: NamedSharding,
    /,
) -> tuple[
    dict[str, Array],
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[Array, ...],
    tuple[int, ...],
    str,
    str,
    Array,
]:
    """Establish canonical source facts from bounded actual owner rows."""
    construction = generated.construction
    vertex_capacity, cell_capacity = (
        prepared.layout.vertex_capacity,
        prepared.layout.cell_capacity,
    )
    coordinates = (
        np.empty((0, 3), dtype=np.float64)
        if construction is None
        else construction.vertices
    )
    corners = (
        np.empty((0, 3), dtype=np.int64)
        if construction is None
        else generated.vertex_ids[construction.triangles]
    )
    vertex_ids = _global_rows(generated.vertex_ids, vertex_capacity, sharding, -1)
    vertex_owners = _global_rows(generated.vertex_owners, vertex_capacity, sharding, -1)
    raw_coordinates = _global_rows(coordinates, vertex_capacity, sharding)
    cell_ids = _global_rows(generated.cell_ids, cell_capacity, sharding, -1)
    raw_corners = _global_rows(corners, cell_capacity, sharding, -1)
    ranks = jnp.repeat(jnp.arange(jax.process_count(), dtype=jnp.int32), cell_capacity)
    cell_keys, cell_order, nc = _canonical_rows(cell_ids[:, None], cell_ids >= 0)
    vertex_keys, vertex_order, nv = _canonical_rows(vertex_ids[:, None], vertex_ids >= 0)
    ordered = jnp.argsort(
        jnp.where(vertex_ids >= 0, vertex_ids, jnp.iinfo(jnp.int64).max), stable=True
    )
    duplicate = (vertex_ids[ordered[1:]] >= 0) & (
        vertex_ids[ordered[1:]] == vertex_ids[ordered[:-1]]
    )
    shared = jnp.all(
        ~duplicate[:, None]
        | (raw_coordinates[ordered[1:]] == raw_coordinates[ordered[:-1]])
    )
    shared &= jnp.all(
        ~duplicate | (vertex_owners[ordered[1:]] == vertex_owners[ordered[:-1]])
    )
    owned_vertex = vertex_ids >= 0
    owned_vertex &= vertex_owners == jnp.repeat(
        jnp.arange(jax.process_count(), dtype=jnp.int32), vertex_capacity
    )
    unique_ownership = jnp.sum(owned_vertex) == nv
    unique_ownership &= (
        (jnp.sum(cell_ids >= 0) == nc)
        & (nc == generated.global_cell_count)
        & (nv == generated.global_vertex_count)
    )
    canonical_corners = jnp.where(
        (jnp.arange(cell_keys.shape[0]) < nc)[:, None], raw_corners[cell_order], -1
    )
    canonical_coordinates = jnp.where(
        (jnp.arange(vertex_keys.shape[0]) < nv)[:, None],
        raw_coordinates[vertex_order],
        0.0,
    )
    arrays = {
        "cell_global_ids": cell_keys[:, 0],
        "cell_vertices": canonical_corners,
        "vertex_global_ids": vertex_keys[:, 0],
        "coordinates": canonical_coordinates,
        "cell_owners": ranks[cell_order],
    }
    fields = (
        ("source_dimensions", "vertex_source_dimensions", np.int64, -1),
        ("source_indices", "vertex_source_indices", np.int64, -1),
        ("source_parameters", "vertex_parameters", np.float64, 0),
    )
    for name, attribute, dtype, fill in fields:
        shape = (0, 2) if name == "source_parameters" else (0,)
        value = (
            np.empty(shape, dtype=dtype)
            if construction is None
            else np.asarray(getattr(construction, attribute), dtype=dtype)
        )
        bank = _global_rows(value, vertex_capacity, sharding, fill)
        arrays[f"initial/{name}"] = bank[vertex_order]
        equal = bank[ordered[1:]] == bank[ordered[:-1]]
        if name == "source_parameters":
            equal |= jnp.isnan(bank[ordered[1:]]) & jnp.isnan(bank[ordered[:-1]])
        if equal.ndim == 2:
            equal = jnp.all(equal, axis=1)
        shared &= jnp.all(~duplicate | equal)
    for name in (
        "patches",
        "charts",
        "parameters",
        "deviations",
        "deviation_bounds",
        "normal_bounds",
    ):
        shape = (
            (0, 3, 2) if name == "parameters" else (0, 2) if name == "charts" else (0,)
        )
        dtype = np.int64 if name == "patches" else np.float64
        value = (
            np.empty(shape, dtype=dtype)
            if construction is None
            else np.asarray(getattr(construction, f"triangle_{name}"), dtype=dtype)
        )
        arrays[f"initial/cell_{name}"] = _global_rows(
            value, cell_capacity, sharding, -1 if name == "patches" else 0
        )[cell_order]
    edge_columns = np.asarray(((0, 1), (0, 2), (1, 2)), dtype=np.int32)
    edge_rows = jnp.sort(canonical_corners[:, edge_columns], axis=2).reshape((-1, 2))
    edge_valid = jnp.repeat(jnp.arange(cell_keys.shape[0]) < nc, 3)
    edge_keys, edge_order, ne = _canonical_rows(edge_rows, edge_valid)
    edge_ids = jnp.where(
        jnp.arange(edge_keys.shape[0]) < ne,
        jnp.arange(edge_keys.shape[0], dtype=jnp.int64),
        -1,
    )
    edge_owners = jnp.repeat(ranks[cell_order], 3)[edge_order]
    positions = _key_positions(
        jnp.where(edge_keys >= 0, edge_keys, jnp.iinfo(jnp.int64).max), edge_rows
    )
    incidence = (
        jnp.zeros(edge_ids.shape, dtype=jnp.int32)
        .at[jnp.maximum(positions, 0)]
        .add(edge_valid.astype(jnp.int32))
    )
    directed = canonical_corners[:, edge_columns].reshape((-1, 2))
    sign = jnp.where(directed[:, 0] < directed[:, 1], 1, -1) * jnp.tile(
        jnp.asarray((1, -1, 1), dtype=jnp.int32), cell_keys.shape[0]
    )
    orientation = (
        jnp.zeros(edge_ids.shape, dtype=jnp.int32)
        .at[jnp.maximum(positions, 0)]
        .add(jnp.where(edge_valid, sign, 0))
    )
    curves = (
        np.empty((0,), dtype=np.int64)
        if construction is None
        else construction.curve_edge_curves
    )
    curve_edges = (
        np.empty((0, 2), dtype=np.int64)
        if construction is None
        else np.sort(generated.vertex_ids[construction.curve_edges], axis=1)
    )
    curve_keys = _global_rows(curve_edges, 3 * cell_capacity, sharding, -1)
    curve_values = _global_rows(curves, 3 * cell_capacity, sharding, -1)
    curve_table, curve_order, curve_count = _canonical_rows(curve_keys, curve_values >= 0)
    curve_positions = _key_positions(
        jnp.where(curve_table >= 0, curve_table, jnp.iinfo(jnp.int64).max), edge_keys
    )
    edge_curves = jnp.where(
        curve_positions >= 0,
        curve_values[curve_order[jnp.maximum(curve_positions, 0)]],
        -1,
    )
    uses = np.asarray(
        [
            sum(
                patch in prepared.compiled.patches
                for patch, _ in prepared.compiled.domain.curve_uses(curve)
            )
            for curve in range(len(prepared.compiled.domain.curves))
        ],
        dtype=np.int32,
    )
    expected = jnp.where(
        edge_curves >= 0, jnp.asarray(uses)[jnp.maximum(edge_curves, 0)], 2
    )
    reciprocal = jnp.all((jnp.arange(edge_ids.shape[0]) >= ne) | (incidence == expected))
    curve_rows = _key_positions(
        jnp.where(edge_keys >= 0, edge_keys, jnp.iinfo(jnp.int64).max), curve_table
    )
    reciprocal &= jnp.all(
        (jnp.arange(curve_table.shape[0]) >= curve_count) | (curve_rows >= 0)
    )
    reciprocal &= jnp.all((incidence != 2) | (orientation == 0))
    arrays.update(
        {
            "entity_global_ids_1": edge_ids,
            "entity_vertices_1": edge_keys,
            "initial/edge_curves": edge_curves,
            "initial/edge_physical": incidence == 1,
        }
    )
    for name in ("deviations", "deviation_bounds"):
        value = (
            np.empty((0,), dtype=np.float64)
            if construction is None
            else getattr(construction, f"curve_edge_{name}")
        )
        bank = _global_rows(value, 3 * cell_capacity, sharding)
        arrays[f"initial/edge_{name}"] = bank[
            curve_order[jnp.maximum(curve_positions, 0)]
        ]
    along = (
        np.empty((0, 2), dtype=np.int64)
        if construction is None
        else generated.vertex_ids[construction.curve_edges]
    )
    bank = _global_rows(along, 3 * cell_capacity, sharding, -1)
    arrays["initial/edge_along"] = bank[curve_order[jnp.maximum(curve_positions, 0)]]
    keys, identifiers, owners = (
        (vertex_keys, edge_keys, cell_keys),
        (vertex_keys[:, 0], edge_ids, cell_keys[:, 0]),
        (vertex_owners[vertex_order], edge_owners, ranks[cell_order]),
    )
    counts = (nv, ne, nc)
    for degree, count in enumerate(counts):
        arrays[f"entity_active_{degree}"] = jnp.arange(keys[degree].shape[0]) < count
    shapes = {
        "cell_global_ids": (nc,),
        "cell_vertices": (nc, 3),
        "vertex_global_ids": (nv,),
        "entity_global_ids_1": (ne,),
        "entity_vertices_1": (ne, 2),
    }
    topology_id = canonical_fingerprint(
        {
            "kind": "cell-mesh-topology",
            "dimension": 2,
            "blocks": [("surface", "triangle")],
            "arrays": logical_array_value_collection_digest(
                {name: arrays[name] for name in shapes}, logical_shapes=shapes
            ),
        }
    )
    geometry_id = canonical_fingerprint(
        {
            "kind": "cell-mesh-geometry",
            "topology": topology_id,
            "ambient_dimension": 3,
            "coordinates": logical_array_value_collection_digest(
                {"coordinates": canonical_coordinates},
                logical_shapes={"coordinates": (nv, 3)},
            ),
        }
    )
    arrays.update(
        {
            "geometry/coordinate_ids": vertex_keys[:, 0],
            "geometry/coordinates": canonical_coordinates,
            "geometry/coordinate_owners": owners[0],
            "geometry/cell_ids/surface": cell_keys[:, 0],
            "geometry/routes/surface": canonical_corners,
        }
    )
    from ..discretization._cell_geometry import coordinate_lagrange_element
    from ..discretization._coordinate_enclosure import coordinate_source_signature

    element = coordinate_lagrange_element("triangle", 1)
    basis_id = canonical_fingerprint(
        {
            "signature": coordinate_source_signature(element),
            "arrays": array_tree_fingerprint(element),
        }
    )
    basis = jnp.asarray(np.frombuffer(bytes.fromhex(basis_id), dtype=np.uint8))
    arrays["geometry/source_basis/surface"] = jnp.where(
        (jnp.arange(cell_keys.shape[0]) < nc)[:, None],
        jnp.broadcast_to(basis, (cell_keys.shape[0], 32)),
        0,
    )
    patch_checks = jax.make_array_from_process_local_data(
        sharding,
        local.patch_checks[None],
        (jax.process_count(), prepared.compiled.patches.size),
    )
    covered = jnp.all(jnp.sum(patch_checks.astype(jnp.int32), axis=0) == 1)
    local_checks = np.asarray(
        (True, True, True, True, True, True, True, True, True), dtype=np.bool_
    )
    checks = jax.make_array_from_process_local_data(
        sharding, local_checks[None], (jax.process_count(), len(_CHECKS))
    )
    checks = (
        checks.at[:, 1]
        .set(covered)
        .at[:, 5]
        .set(unique_ownership)
        .at[:, 6]
        .set(shared)
        .at[:, 7]
        .set(reciprocal)
    )
    return arrays, keys, identifiers, owners, counts, topology_id, geometry_id, checks


def _initial_coordinate_geometry_id(
    arrays: dict[str, Array], counts: tuple[int, ...], topology_id: str, /
) -> str:
    """Bind the actual affine coordinate basis, coefficient IDs, values and routes."""
    shapes = {
        "geometry/coordinate_ids": (counts[0],),
        "geometry/coordinates": (counts[0], 3),
        "geometry/cell_ids/surface": (counts[2],),
        "geometry/routes/surface": (counts[2], 3),
        "geometry/source_basis/surface": (counts[2], 32),
    }
    digest = logical_array_value_collection_digest(
        {name: arrays[name] for name in shapes}, logical_shapes=shapes
    )
    return canonical_fingerprint(
        {
            "kind": "collective-initial-coordinate-geometry",
            "topology": topology_id,
            "arrays": digest,
        }
    )


@partial(jax.jit, static_argnames=("vertex_count", "cell_count", "capacity", "parts"))
def _initial_closures(
    arrays: dict[str, Array],
    *,
    vertex_count: int,
    cell_count: int,
    capacity: int,
    parts: int,
) -> tuple[dict[str, Array], Array]:
    """Materialize every owner closure in one common-shape device operation."""
    ids, corners = arrays["cell_global_ids"], arrays["cell_vertices"]
    table = jnp.where(
        jnp.arange(arrays["vertex_global_ids"].shape[0]) < vertex_count,
        arrays["vertex_global_ids"],
        jnp.iinfo(jnp.int64).max,
    )
    rows = jnp.searchsorted(table, corners)
    active = jnp.arange(ids.shape[0]) < cell_count

    def one(part: Array) -> tuple[Array, Array, Array]:
        own_cells = active & (arrays["cell_owners"] == part)
        own_vertices = (
            jnp.zeros(table.shape, dtype=jnp.bool_)
            .at[jnp.minimum(rows, table.shape[0] - 1)]
            .max(own_cells[:, None])
        )
        selected = active & jnp.any(
            own_vertices[jnp.minimum(rows, table.shape[0] - 1)], axis=1
        )
        count = jnp.sum(selected, dtype=jnp.int32)
        order = jnp.nonzero(selected, size=capacity, fill_value=0)[0]
        return order, jnp.arange(capacity) < count, count <= capacity

    order, valid, fits = jax.vmap(one)(jnp.arange(parts, dtype=jnp.int32))
    packets = {
        "cell_ids": jnp.where(valid, ids[order], -1),
        "cell_vertices": jnp.where(valid[..., None], corners[order], -1),
        "cell_valid": valid,
        "cell_owner": arrays["cell_owners"][order],
        "cell_coordinates": arrays["coordinates"][
            jnp.minimum(rows[order], table.shape[0] - 1)
        ],
    }
    return {f"closure/{name}": value for name, value in packets.items()}, jnp.all(fits)


_INITIAL_INPUT_FIELDS = (
    ("vertices", np.float64, (3,)),
    ("vertex_source_dimensions", np.int64, ()),
    ("vertex_source_indices", np.int64, ()),
    ("vertex_parameters", np.float64, (2,)),
    ("chart_restriction_vertices", np.int64, ()),
    ("chart_restriction_edges", np.int64, (2,)),
    ("chart_restriction_endpoint_parameters", np.float64, (2, 2)),
    ("chart_restriction_parameters", np.int64, (2,)),
    ("triangles", np.int64, (3,)),
    ("triangle_patches", np.int64, ()),
    ("triangle_charts", np.float64, (2,)),
    ("triangle_parameters", np.float64, (3, 2)),
    ("triangle_deviations", np.float64, ()),
    ("triangle_deviation_bounds", np.float64, ()),
    ("triangle_normal_bounds", np.float64, ()),
    ("curve_edges", np.int64, (2,)),
    ("curve_edge_curves", np.int64, ()),
    ("curve_edge_deviations", np.float64, ()),
    ("curve_edge_deviation_bounds", np.float64, ()),
    ("unresolved", np.bool_, ()),
)


def _retain_initial_inputs(
    prepared: PreparedDistributedSurfaceGeneration,
    generated: DistributedSurfaceConstruction,
    sharding: NamedSharding,
    /,
) -> dict[str, Array]:
    """Retain actual native chart/trim premises for independent cold replay."""
    construction = generated.construction
    parts = prepared.partition_count
    result: dict[str, Array] = {}

    def retain(name: str, value: np.ndarray) -> None:
        counts = np.asarray(
            multihost_utils.process_allgather(np.int64(value.shape[0]), tiled=False)
        )
        capacity = max(1, int(np.max(counts)))
        result[f"initial/input/{name}"] = _global_rows(value, capacity, sharding).reshape(
            (parts, capacity, *value.shape[1:])
        )
        result[f"initial/input_count/{name}"] = jax.make_array_from_process_local_data(
            sharding, np.asarray((value.shape[0],), dtype=np.int64), (parts,)
        )

    for name, dtype, tail in _INITIAL_INPUT_FIELDS:
        value = (
            np.empty((0, *tail), dtype=dtype)
            if construction is None
            else np.asarray(getattr(construction, name), dtype=dtype)
        )
        retain(name, value)
    retain("vertex_ids", generated.vertex_ids)
    retain("vertex_owners", generated.vertex_owners)
    retain("cell_ids", generated.cell_ids)
    retain("shared_curve_ids", generated.shared_curve_ids)
    charts = (
        {}
        if construction is None
        else {row[0]: row[1:] for row in construction.chart_triangulations}
    )
    for patch in prepared.compiled.patches.tolist():
        for index, (name, dtype, tail) in enumerate(
            (
                ("charts", np.float64, (2,)),
                ("points", np.float64, (3,)),
                ("cells", np.int64, (3,)),
                ("boundary", np.int64, (2,)),
                ("provenance", np.float64, (4,)),
                ("restriction_required", np.bool_, ()),
                ("restriction_vertices", np.int64, ()),
                ("restriction_edges", np.int64, (2,)),
                ("restriction_parameters", np.int64, (2,)),
            )
        ):
            value = (
                np.empty((0, *tail), dtype=dtype)
                if patch not in charts
                else np.asarray(charts[patch][index], dtype=dtype)
            )
            retain(f"chart/{patch}/{name}", value)
    curves = {} if construction is None else dict(construction.curve_parameters)
    for curve in prepared.compiled.curves.tolist():
        retain(f"curve/{curve}", curves.get(curve, np.empty((0,), dtype=np.float64)))
    integers = np.asarray(
        (
            0 if construction is None else construction.rounds,
            0 if construction is None else construction.inserted,
            0 if construction is None else construction.refused,
            0 if construction is None else construction.flips,
            0 if construction is None else construction.geometry_queries,
            0 if construction is None else construction.work_units,
            0 if construction is None else construction.incomplete_refinement,
            generated.neighbor_packet_count,
            generated.neighbor_transferred_bytes,
            generated.metadata_bytes,
        ),
        dtype=np.int64,
    )
    result["initial/input_metadata"] = jax.make_array_from_process_local_data(
        sharding, integers[None], (parts, integers.size)
    )
    result["initial/input_minimum_angle"] = jax.make_array_from_process_local_data(
        sharding,
        np.asarray(
            (np.inf if construction is None else construction.minimum_angle,),
            dtype=np.float64,
        ),
        (parts,),
    )
    result["initial/input_present"] = jax.make_array_from_process_local_data(
        sharding, np.asarray((construction is not None,), dtype=np.bool_), (parts,)
    )
    return result


def _placement() -> NamedSharding:
    devices = [
        next(device for device in jax.devices() if device.process_index == rank)
        for rank in range(jax.process_count())
    ]
    return NamedSharding(
        Mesh(np.asarray(devices), ("initial_owner",)), PartitionSpec("initial_owner")
    )


def _global_rows(
    local: np.ndarray,
    capacity: int,
    sharding: NamedSharding,
    fill: int | float | bool = 0,
    /,
) -> Array:
    if local.shape[0] > capacity:
        raise _failure("Initial certificate rows exceed their native reserved capacity.")
    padded = np.full((capacity, *local.shape[1:]), fill, dtype=local.dtype)
    padded[: local.shape[0]] = local
    return jax.make_array_from_process_local_data(
        sharding, padded, (jax.process_count() * capacity, *local.shape[1:])
    )


def _exchange(
    local: np.ndarray, first: int, second: int, sharding: NamedSharding, /
) -> np.ndarray:
    """One bounded candidate packet, with no global target host materialization."""
    data = jax.make_array_from_process_local_data(
        sharding, local[None], (jax.process_count(), *local.shape)
    )
    permutation = tuple(
        (rank, second if rank == first else first if rank == second else rank)
        for rank in range(jax.process_count())
    )

    @partial(
        jax.shard_map,
        mesh=sharding.mesh,
        in_specs=PartitionSpec("initial_owner"),
        out_specs=PartitionSpec("initial_owner"),
        check_vma=False,
    )
    def send(value: Array) -> Array:
        return jax.lax.ppermute(value, "initial_owner", permutation)

    received = jax.jit(send)(data)
    return np.asarray(jax.device_get(received.addressable_shards[0].data))[0]


def _cross_owner_contacts(
    generated: DistributedSurfaceConstruction,
    limits: MeshCertificateLimits,
    maximum_scratch_bytes: int,
    sharding: NamedSharding,
    /,
) -> tuple[NDArray[np.int64], int]:
    construction = generated.construction
    points = (
        np.empty((0, 3, 3), dtype=np.float64)
        if construction is None
        else construction.vertices[construction.triangles]
    )
    ids = (
        np.empty((0, 3), dtype=np.int64)
        if construction is None
        else generated.vertex_ids[construction.triangles]
    )
    lower = np.min(points, axis=1) if points.size else np.empty((0, 3), dtype=np.float64)
    upper = np.max(points, axis=1) if points.size else np.empty((0, 3), dtype=np.float64)
    box = (
        np.stack((np.min(lower, axis=0), np.max(upper, axis=0)))
        if points.size
        else np.asarray(((np.inf,) * 3, (-np.inf,) * 3))
    )
    boxes = np.asarray(multihost_utils.process_allgather(box, tiled=False))
    # Packets have distinct int64 identity and binary64 coordinate channels.
    # No ID is encoded as a float (large scientific IDs retain exact identity).
    capacity = min(256, maximum_scratch_bytes // (4 * (9 * 8 + 4 * 8)))
    if capacity < 1:
        raise _failure(
            "Initial intersection candidate packet exceeds the scratch budget."
        )
    findings: list[tuple[int, int, int]] = []
    tested = 0
    for first, second in combinations(range(jax.process_count()), 2):
        if np.any(boxes[first, 1] < boxes[second, 0]) or np.any(
            boxes[second, 1] < boxes[first, 0]
        ):
            continue
        rank = jax.process_index()
        other = second if rank == first else first
        selected = (
            np.flatnonzero(
                np.all(lower <= boxes[other, 1], axis=1)
                & np.all(upper >= boxes[other, 0], axis=1)
            )
            if rank in (first, second)
            else np.empty((0,), dtype=np.int64)
        )
        counts = np.asarray(
            multihost_utils.process_allgather(np.int64(selected.size), tiled=False)
        )
        rounds = (max(int(counts[first]), int(counts[second])) + capacity - 1) // capacity
        for step in range(rounds):
            rows = selected[step * capacity : (step + 1) * capacity]
            geometry_packet = np.zeros((capacity, 3, 3), dtype=np.float64)
            identity_packet = np.full((capacity, 4), -1, dtype=np.int64)
            geometry_packet[: rows.size] = points[rows]
            identity_packet[: rows.size, :3] = ids[rows]
            identity_packet[: rows.size, 3] = generated.cell_ids[rows]
            remote_points = _exchange(geometry_packet, first, second, sharding)
            remote_ids = _exchange(identity_packet, first, second, sharding)
            if rank != first:
                continue
            for triangle, identities in zip(remote_points, remote_ids, strict=True):
                if identities[3] < 0:
                    continue
                candidates = np.flatnonzero(
                    np.all(lower <= np.max(triangle, axis=0), axis=1)
                    & np.all(upper >= np.min(triangle, axis=0), axis=1)
                )
                tested += candidates.size
                if tested > limits.maximum_candidate_pairs:
                    findings.append((-1, -1, 2))
                    continue
                for start in range(0, candidates.size, capacity):
                    batch = candidates[start : start + capacity]
                    joined_ids = np.concatenate((ids[batch].reshape(-1), identities[:3]))
                    unique, inverse = np.unique(joined_ids, return_inverse=True)
                    coordinates = np.empty((unique.size, 3), dtype=np.float64)
                    coordinates[inverse] = np.concatenate(
                        (points[batch].reshape((-1, 3)), triangle)
                    )
                    a = inverse[: 3 * batch.size].reshape((-1, 3))
                    b = np.broadcast_to(inverse[-3:], a.shape)
                    hit, certain = _triangle_pairs_intersect(coordinates, a, b)
                    for cell, intersects, decided in zip(
                        generated.cell_ids[batch], hit, certain, strict=True
                    ):
                        if intersects or not decided:
                            findings.append(
                                (int(cell), int(identities[3]), 1 if decided else 2)
                            )
    return np.asarray(findings, dtype=np.int64).reshape((-1, 3)), tested


class InitialCollectiveMeshEvidence(AbstractCollectiveMeshTheorem, NonTrainableState):
    """Consumed independent source theorem and complete canonical initial content."""

    compiled: CompiledSurfaceDomain
    specification: SurfaceMeshingSpec
    authored_source: NativeSurfaceSource
    schedule: NativeSurfaceSchedule
    layout: DeviceGenerationLayout
    patch_owners: NDArray[np.int32] = eqx.field(static=True)
    cell_kind: Literal["triangle"] = eqx.field(static=True)
    partition_checks: Array
    global_checks: Array
    logical_arrays: tuple[tuple[str, Array], ...]
    entity_keys: tuple[Array, ...]
    entity_ids: tuple[Array, ...]
    entity_owners: tuple[Array, ...]
    global_entity_counts: tuple[int, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    coordinate_geometry_id: str = eqx.field(static=True)
    global_organization_id: str = eqx.field(static=True)
    source_evidence_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    partition_count: int = eqx.field(static=True)
    checks: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    minimum_angle: float = eqx.field(static=True)
    maximum_normal_angle: float = eqx.field(static=True)
    limits: MeshCertificateLimits
    content_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedDistributedSurfaceGeneration,
        generated: DistributedSurfaceConstruction,
        /,
        *,
        limits: MeshCertificateLimits | None = None,
        minimum_angle: float,
        maximum_normal_angle: float = np.inf,
    ) -> None:
        limits_ = MeshCertificateLimits() if limits is None else limits
        if not isinstance(
            prepared, PreparedDistributedSurfaceGeneration
        ) or not isinstance(generated, DistributedSurfaceConstruction):
            raise TypeError(
                "Initial certification consumes actual native prepared source and generated rows."
            )
        if not isinstance(limits_, MeshCertificateLimits):
            raise TypeError("limits must be MeshCertificateLimits.")
        if (
            not np.isfinite(minimum_angle)
            or minimum_angle < 0.0
            or minimum_angle >= np.pi / 3.0
        ):
            raise ValueError("minimum_angle must lie in [0, pi/3).")
        if np.isnan(maximum_normal_angle) or maximum_normal_angle <= 0.0:
            raise ValueError("maximum_normal_angle must be positive.")
        binding = canonical_fingerprint(
            (
                prepared.compiled.compiled_id,
                prepared.specification.specification_id,
                prepared.schedule.schedule_id,
                limits_.limits_id,
                (
                    prepared.layout.vertex_capacity,
                    prepared.layout.cell_capacity,
                    prepared.layout.candidate_capacity,
                    prepared.layout.maximum_work_units,
                ),
                array_tree_fingerprint(prepared.patch_owners),
                array_tree_fingerprint(
                    np.asarray((minimum_angle, maximum_normal_angle), dtype=np.float64)
                ),
            )
        )
        bindings = np.asarray(
            multihost_utils.process_allgather(
                np.frombuffer(bytes.fromhex(binding), dtype=np.uint8), tiled=False
            )
        )
        if np.any(bindings != bindings[0]):
            raise _failure(
                "Initial theorem owners disagree on their source, controls, limits, or common publication capacities."
            )
        local = _LocalProof((), np.zeros(prepared.compiled.patches.shape, dtype=np.bool_))
        failure = None
        try:
            local = _local_proof(
                prepared, generated, limits_, minimum_angle, maximum_normal_angle
            )
        except (ValueError, MeshingFailure) as error:
            failure = error if isinstance(error, MeshingFailure) else _failure(str(error))
        sharding = _placement()
        categories = tuple(MeshingFailureCategory)
        code = 0 if failure is None else categories.index(failure.category) + 1
        rejected = np.asarray(
            multihost_utils.process_allgather(np.int32(code), tiled=False)
        )
        if np.any(rejected):
            rejected_codes = jax.make_array_from_process_local_data(
                sharding, np.asarray((code,), dtype=np.int32), (jax.process_count(),)
            )
            raw_cells = _global_rows(
                generated.cell_ids, prepared.layout.cell_capacity, sharding, -1
            )
            raise MeshingFailure(
                categories[int(np.min(rejected[rejected > 0])) - 1],
                "An owner rejected the independent initial source theorem; no part was published.",
                stage="distributed-initial-certification",
                logical_findings=(
                    ("initial/failed_owner_category", rejected_codes),
                    ("initial/rejected_cell_ids", raw_cells),
                ),
            )
        arrays, keys, identifiers, owners, counts, topology_id, geometry_id, checks = (
            _logical_initial_content(prepared, generated, local, sharding)
        )
        request_limits = prepared.specification.limits
        quotas = (
            ("vertices", counts[0], request_limits.maximum_vertices),
            ("edges", counts[1], request_limits.maximum_edges),
            ("faces", counts[2], request_limits.maximum_faces),
            ("cells", counts[2], request_limits.maximum_cells),
            (
                "connectivity_entries",
                3 * counts[2],
                request_limits.maximum_connectivity_entries,
            ),
            (
                "data_bytes",
                24 * counts[0] + 12 * counts[2],
                request_limits.maximum_data_bytes,
            ),
        )
        exceeded = tuple(name for name, measured, maximum in quotas if measured > maximum)
        if exceeded:
            raise MeshingFailure(
                MeshingFailureCategory.RESOURCE_EXHAUSTED,
                "Initial canonical construction exceeds global request quotas: "
                + ", ".join(exceeded),
                stage="distributed-initial-resources",
                requested=tuple((name, float(maximum)) for name, _, maximum in quotas),
                achieved=tuple((name, float(measured)) for name, measured, _ in quotas),
            )
        coordinate_geometry_id = _initial_coordinate_geometry_id(
            arrays, counts, topology_id
        )
        findings, pairs = _cross_owner_contacts(
            generated,
            limits_,
            prepared.specification.limits.maximum_scratch_bytes,
            sharding,
        )
        cross = np.asarray(
            multihost_utils.process_allgather(
                np.bool_(findings.shape[0] == 0), tiled=False
            )
        )
        pair_counts = np.asarray(
            multihost_utils.process_allgather(np.int64(pairs), tiled=False)
        )
        cross = cross & (np.sum(pair_counts) <= limits_.maximum_candidate_pairs)
        checks = checks.at[:, -1].set(jnp.asarray(cross))
        verdict = jnp.all(checks, axis=0)
        summary = np.asarray(jax.device_get(verdict), dtype=np.bool_)
        finding_capacity = max(
            1,
            int(
                np.max(
                    np.asarray(
                        multihost_utils.process_allgather(
                            np.int64(findings.shape[0]), tiled=False
                        )
                    )
                )
            ),
        )
        raw_findings = _global_rows(findings, finding_capacity, sharding, -1)
        if not np.all(summary):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "Initial collective theorem failed: "
                + ", ".join(
                    name
                    for name, passed in zip(_CHECKS, summary, strict=True)
                    if not passed
                ),
                stage="distributed-initial-certification",
                logical_findings=(
                    ("initial/cross_owner_findings", raw_findings),
                    ("initial/partition_checks", checks),
                ),
            )
        arrays["initial/cross_owner_findings"] = raw_findings
        arrays["initial/partition_checks"] = checks
        arrays["initial/intersection_pairs"] = jax.make_array_from_process_local_data(
            sharding, np.asarray((pairs,), dtype=np.int64), (jax.process_count(),)
        )
        certificate_rows = np.asarray(
            [
                np.frombuffer(bytes.fromhex(value), dtype=np.uint8)
                for value in local.certificate_ids
            ],
            dtype=np.uint8,
        ).reshape((-1, 32))
        arrays["initial/certificate_ids"] = _global_rows(
            certificate_rows, 2 + 2 * prepared.compiled.patches.size, sharding
        )
        arrays["initial/certificate_counts"] = jax.make_array_from_process_local_data(
            sharding,
            np.asarray((certificate_rows.shape[0],), dtype=np.int64),
            (jax.process_count(),),
        )
        arrays.update(_retain_initial_inputs(prepared, generated, sharding))
        closures, fits = _initial_closures(
            arrays,
            vertex_count=counts[0],
            cell_count=counts[2],
            capacity=prepared.layout.cell_capacity,
            parts=jax.process_count(),
        )
        if not bool(jax.device_get(fits)):
            raise _failure(
                "Initial publication closure exceeds its declared fixed capacity."
            )
        arrays.update(
            {name: jax.device_put(value, sharding) for name, value in closures.items()}
        )
        from ._collective_organization import initial_organization_membership

        arrays.update(
            initial_organization_membership(
                prepared.compiled, prepared.specification, arrays, counts
            )
        )
        organization_names = tuple(
            name
            for name in arrays
            if name.startswith("initial/source_")
            or name.startswith("initial/cell_patches")
            or name.startswith("initial/edge_curves")
            or name.startswith("organization/")
        )
        organization_id = canonical_fingerprint(
            {
                "kind": "collective-initial-source-organization",
                "source": prepared.compiled.domain.domain_id,
                "specification": prepared.specification.specification_id,
                "content": logical_array_value_collection_digest(
                    {name: arrays[name] for name in organization_names}
                ),
            }
        )
        content_id = logical_array_value_collection_digest(arrays)
        source_evidence_id = canonical_fingerprint(
            {
                "kind": "collective-initial-source-theorem",
                "source": prepared.compiled.domain.source_id,
                "revision": prepared.compiled.domain.source_revision,
                "compiled": prepared.compiled.compiled_id,
                "specification": prepared.specification.specification_id,
                "limits": limits_.limits_id,
                "physical_angles": array_tree_fingerprint(
                    np.asarray((minimum_angle, maximum_normal_angle), dtype=np.float64)
                ),
                "content": content_id,
                "organization": organization_id,
            }
        )
        mesh_id = canonical_fingerprint(
            {"kind": "cell-mesh", "topology": topology_id, "geometry": geometry_id}
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "collective-native-initial-evidence",
                "mesh": mesh_id,
                "source_theorem": source_evidence_id,
                "content": content_id,
                "checks": list(zip(_CHECKS, summary.tolist(), strict=True)),
            }
        )
        self.compiled = prepared.compiled
        self.specification = prepared.specification
        self.partition_checks = checks
        self.global_checks = verdict
        self.logical_arrays = tuple(sorted(arrays.items()))
        self.entity_keys = keys
        self.entity_ids = identifiers
        self.entity_owners = owners
        self.global_entity_counts = counts
        self.topology_id = topology_id
        self.geometry_id = geometry_id
        self.coordinate_geometry_id = coordinate_geometry_id
        self.global_organization_id = organization_id
        self.source_evidence_id = source_evidence_id
        self.mesh_id = mesh_id
        self.partition_count = jax.process_count()
        self.checks = _CHECKS
        self.evidence_id = evidence_id
        self.minimum_angle = minimum_angle
        self.maximum_normal_angle = maximum_normal_angle
        self.cell_kind = "triangle"
        self.limits = limits_
        self.content_id = content_id
        self.authored_source = NativeSurfaceSource(prepared.compiled.domain)
        self.schedule = prepared.schedule
        self.layout = prepared.layout
        self.patch_owners = prepared.patch_owners

    def require_passed(self) -> None:
        if not bool(jax.device_get(jnp.all(self.global_checks))):
            raise _failure(
                "Initial collective evidence contains a rejected global verdict."
            )

    def require_current(self) -> None:
        """Collective immutable-content replay before any asymmetric constructor."""
        self.require_passed()
        if (
            logical_array_value_collection_digest(dict(self.logical_arrays))
            != self.content_id
        ):
            raise _failure(
                "Initial collective source theorem numerical content was changed."
            )
        if (
            _initial_coordinate_geometry_id(
                dict(self.logical_arrays), self.global_entity_counts, self.topology_id
            )
            != self.coordinate_geometry_id
        ):
            raise _failure(
                "Initial coordinate-map authority differs from its actual basis and coefficient banks."
            )
        expected = canonical_fingerprint(
            {
                "kind": "collective-initial-source-theorem",
                "source": self.compiled.domain.source_id,
                "revision": self.compiled.domain.source_revision,
                "compiled": self.compiled.compiled_id,
                "specification": self.specification.specification_id,
                "limits": self.limits.limits_id,
                "physical_angles": array_tree_fingerprint(
                    np.asarray(
                        (self.minimum_angle, self.maximum_normal_angle), dtype=np.float64
                    )
                ),
                "content": self.content_id,
                "organization": self.global_organization_id,
            }
        )
        if expected != self.source_evidence_id:
            raise _failure(
                "Initial theorem differs from its original authored domain and physical controls."
            )


def restore_initial_construction(
    evidence: InitialCollectiveMeshEvidence,
    partition_index: int,
    /,
) -> DistributedSurfaceConstruction:
    """Restore the original numerical premises, never a target or accepted flag."""
    from ._publication_lowering import _addressable

    if not isinstance(evidence, InitialCollectiveMeshEvidence):
        raise TypeError(
            "Initial input restoration requires its actual independent source theorem."
        )
    if not 0 <= partition_index < evidence.partition_count:
        raise ValueError(
            "Initial input partition is outside the retained source placement."
        )
    banks = dict(evidence.logical_arrays)

    def local(name: str) -> np.ndarray:
        return np.asarray(jax.device_get(_addressable(banks[name], partition_index)))

    def rows(name: str) -> np.ndarray:
        count = int(local(f"initial/input_count/{name}"))
        values = local(f"initial/input/{name}")
        if not 0 <= count <= values.shape[0]:
            raise _failure(
                "Retained initial input counts exceed their actual numerical packet."
            )
        return values[:count]

    metadata = local("initial/input_metadata")
    construction = None
    if bool(local("initial/input_present")):
        arrays = {name: rows(name) for name, _, _ in _INITIAL_INPUT_FIELDS}
        charts = []
        for patch in evidence.compiled.patches.tolist():
            values = rows(f"chart/{patch}/charts")
            if values.shape[0]:
                charts.append(
                    (
                        patch,
                        values,
                        rows(f"chart/{patch}/points"),
                        rows(f"chart/{patch}/cells"),
                        rows(f"chart/{patch}/boundary"),
                        rows(f"chart/{patch}/provenance"),
                        rows(f"chart/{patch}/restriction_required"),
                        rows(f"chart/{patch}/restriction_vertices"),
                        rows(f"chart/{patch}/restriction_edges"),
                        rows(f"chart/{patch}/restriction_parameters"),
                    )
                )
        curves = []
        for curve in evidence.compiled.curves.tolist():
            values = rows(f"curve/{curve}")
            if values.shape[0]:
                curves.append((curve, values))
        construction = SurfaceConstruction(
            **arrays,
            chart_triangulations=tuple(charts),
            curve_parameters=tuple(curves),
            minimum_angle=float(local("initial/input_minimum_angle")),
            rounds=int(metadata[0]),
            inserted=int(metadata[1]),
            refused=int(metadata[2]),
            flips=int(metadata[3]),
            geometry_queries=int(metadata[4]),
            work_units=int(metadata[5]),
            incomplete_refinement=bool(metadata[6]),
        )
    return DistributedSurfaceConstruction(
        construction,
        rows("vertex_ids"),
        rows("vertex_owners"),
        rows("cell_ids"),
        evidence.global_entity_counts[0],
        evidence.global_entity_counts[-1],
        rows("shared_curve_ids"),
        int(metadata[7]),
        int(metadata[8]),
        int(metadata[9]),
        None,
        evidence.compiled.domain.source_id,
        evidence.compiled.domain.source_revision,
        evidence.specification.specification_id,
    )


def _lower_initial_mesh(
    evidence: InitialCollectiveMeshEvidence,
    receipts: dict[str, np.ndarray],
    geometry_projection: CellGeometryStorageProjection,
    /,
) -> CellMesh:
    from ..discretization._cell_mesh import CellMeshStorage
    from ._topology_edit import entity_keys, key_rows

    valid = receipts["entity/0/valid"]
    vertex_ids = receipts["entity/0/ids"][valid]
    coordinates = receipts["coordinates"][valid]
    active = receipts["closure/cell_valid"]
    cell_ids = receipts["closure/cell_ids"][active]
    corners = receipts["closure/cell_vertices"][active]
    cells = np.searchsorted(vertex_ids, corners).astype(np.int32)
    blocks = (
        ()
        if cell_ids.size == 0
        else (CellBlock("surface", evidence.cell_kind, cells, global_ids=cell_ids),)
    )
    probe = (
        None
        if not blocks
        else CellMesh(coordinates, blocks, vertex_global_ids=vertex_ids)
    )
    identifiers, owners = [], []
    for degree in range(3):
        mask = receipts[f"entity/{degree}/valid"]
        positions = (
            np.empty((0,), dtype=np.int64)
            if probe is None
            else key_rows(
                receipts[f"entity/{degree}/keys"][mask], entity_keys(probe, degree)
            )
        )
        identifiers.append(receipts[f"entity/{degree}/ids"][mask][positions])
        owners.append(receipts[f"entity/{degree}/owners"][mask][positions])
    edge_mask = receipts["entity/1/valid"]
    receipt_edge_keys = receipts["entity/1/keys"][edge_mask]
    edge_keys = receipt_edge_keys if probe is None else entity_keys(probe, 1)
    positions = (
        np.empty((0,), dtype=np.int64)
        if probe is None
        else key_rows(receipt_edge_keys, edge_keys)
    )
    physical = receipts["initial/edge_physical"][edge_mask][positions]
    facets = np.sort(
        corners[:, np.asarray(((0, 1), (0, 2), (1, 2)), dtype=np.int32)], axis=2
    )
    facet_rows = (
        np.empty((0, 3), dtype=np.int64)
        if facets.shape[0] == 0
        else key_rows(edge_keys, facets.reshape((-1, 2))).reshape((-1, 3))
    )
    edge_incidence = np.bincount(facet_rows.reshape(-1), minlength=physical.size)
    complete = np.all((edge_incidence[facet_rows] == 2) | physical[facet_rows], axis=1)
    storage = CellMeshStorage(
        evidence.global_entity_counts,
        identifiers,
        owners,
        partition_index=jax.process_index(),
        partition_count=evidence.partition_count,
        local_coordinates=coordinates,
        local_blocks=blocks,
        global_coordinate_count=evidence.global_entity_counts[0],
        coordinate_global_ids=vertex_ids,
        coordinate_owner=owners[0],
        geometry_projection=geometry_projection,
        local_physical_boundary_facets=physical,
        local_neighborhood_complete=complete,
        neighborhood_depth=1,
        logical_topology_id=evidence.topology_id,
        logical_geometry_id=evidence.geometry_id,
        logical_coordinate_geometry_id=evidence.coordinate_geometry_id,
        evidence_id=evidence.evidence_id,
        logical_arrays=evidence.logical_arrays,
    )
    return CellMesh(
        coordinates,
        blocks,
        storage=storage,
        numeric_version=evidence.compiled.domain.source_revision,
    )


def _lower_initial_construction(
    mesh: CellMesh,
    receipts: dict[str, np.ndarray],
    generated: DistributedSurfaceConstruction,
    minimum_angle: float,
    /,
) -> SurfaceConstruction:
    from ._topology_edit import key_rows

    vertex_mask, cell_mask = receipts["entity/0/valid"], receipts["entity/2/valid"]
    vertices = np.asarray(mesh.vertex_global_ids)
    order = key_rows(receipts["entity/0/ids"][vertex_mask, None], vertices[:, None])
    cell_order = key_rows(
        receipts["entity/2/ids"][cell_mask, None],
        np.asarray(mesh.blocks[0].global_ids)[:, None],
    )
    edge_mask = receipts["entity/1/valid"] & (receipts["initial/edge_curves"] >= 0)
    along = receipts["initial/edge_along"][edge_mask]
    original = generated.construction
    if original is None:
        restriction_vertices = np.empty((0,), dtype=np.int64)
        restriction_edges = np.empty((0, 2), dtype=np.int64)
        restriction_endpoint_parameters = np.empty((0, 2, 2), dtype=np.float64)
        restriction_parameters = np.empty((0, 2), dtype=np.int64)
    else:
        restriction_vertices = np.searchsorted(
            vertices,
            generated.vertex_ids[original.chart_restriction_vertices],
        ).astype(np.int64)
        restriction_edges = np.searchsorted(
            vertices,
            generated.vertex_ids[original.chart_restriction_edges],
        ).astype(np.int64)
        restriction_endpoint_parameters = original.chart_restriction_endpoint_parameters
        restriction_parameters = original.chart_restriction_parameters
    return SurfaceConstruction(
        vertices=np.asarray(mesh.coordinates),
        vertex_source_dimensions=receipts["initial/source_dimensions"][vertex_mask][
            order
        ],
        vertex_source_indices=receipts["initial/source_indices"][vertex_mask][order],
        vertex_parameters=receipts["initial/source_parameters"][vertex_mask][order],
        chart_restriction_vertices=restriction_vertices,
        chart_restriction_edges=restriction_edges,
        chart_restriction_endpoint_parameters=restriction_endpoint_parameters,
        chart_restriction_parameters=restriction_parameters,
        triangles=np.asarray(mesh.blocks[0].vertices),
        triangle_patches=receipts["initial/cell_patches"][cell_mask][cell_order],
        triangle_charts=receipts["initial/cell_charts"][cell_mask][cell_order],
        triangle_parameters=receipts["initial/cell_parameters"][cell_mask][cell_order],
        triangle_deviations=receipts["initial/cell_deviations"][cell_mask][cell_order],
        triangle_deviation_bounds=receipts["initial/cell_deviation_bounds"][cell_mask][
            cell_order
        ],
        triangle_normal_bounds=receipts["initial/cell_normal_bounds"][cell_mask][
            cell_order
        ],
        curve_edges=np.searchsorted(vertices, along).astype(np.int32),
        curve_edge_curves=receipts["initial/edge_curves"][edge_mask],
        curve_edge_deviations=receipts["initial/edge_deviations"][edge_mask],
        curve_edge_deviation_bounds=receipts["initial/edge_deviation_bounds"][edge_mask],
        unresolved=np.zeros((mesh.entity_set(2).count,), dtype=np.bool_),
        chart_triangulations=() if original is None else original.chart_triangulations,
        curve_parameters=() if original is None else original.curve_parameters,
        minimum_angle=minimum_angle,
        rounds=0 if original is None else original.rounds,
        inserted=0 if original is None else original.inserted,
        refused=0 if original is None else original.refused,
        flips=0 if original is None else original.flips,
        geometry_queries=0 if original is None else original.geometry_queries,
        work_units=0 if original is None else original.work_units,
        incomplete_refinement=False
        if original is None
        else original.incomplete_refinement,
    )


def _initial_region_boundaries(
    evidence: InitialCollectiveMeshEvidence,
    mesh: CellMesh,
    patches: tuple[MeshPatch, ...],
    labels: tuple[MeshLabel, ...],
    /,
) -> tuple[RegionBoundaryEvidence, ...]:
    """Bind authored region sides to the final globally scoped patch identities."""
    from ._scope import MeshingEntityKind, MeshingScope

    domain, specification = evidence.compiled.domain, evidence.specification
    result = []
    for region in range(len(domain.regions)):
        identifier = domain.entity_id(3, region)
        sides = tuple(
            (
                patch.patch_id,
                1 if patch.source_adjacent_region_ids[0] == identifier else -1,
            )
            for patch in patches
            if patch.source_adjacent_region_ids is not None
            and identifier in patch.source_adjacent_region_ids
        )
        if not sides:
            continue
        controls = tuple(
            control
            for control in specification.region_controls
            if control.scope.entity_dimension == 3
            and region
            in domain.resolve_indices(
                3, np.asarray(control.scope.global_entity_ids, dtype=np.int64)
            )
        )
        control = controls[0] if controls else None
        name = (
            f"region:{identifier}"
            if control is None
            else control.region_name
            if control.scope.global_entity_ids.shape[0] == 1
            else f"{control.region_name}:{identifier}"
        )
        selected = tuple(label for label in labels if label.name == name)
        if len(selected) != 1:
            raise ValueError(
                "Initial source region lacks its unique complete logical boundary label."
            )
        scope = MeshingScope(
            domain.source_id,
            domain.source_revision,
            MeshingEntityKind.GEOMETRY,
            3,
            domain.entity_set_id(3),
            np.asarray((domain.scope_indices(3)[region],), dtype=np.int64),
        )
        result.append(
            RegionBoundaryEvidence(
                mesh,
                scope,
                identifier,
                domain.domain_id if control is None else control.control_id,
                None if control is None else control.material_id,
                None if control is None else control.role,
                selected[0],
                sides,
            )
        )
    return tuple(result)


def publish_distributed_surface_initial(
    prepared: PreparedDistributedSurfaceGeneration,
    generated: DistributedSurfaceConstruction,
    source: NativeSurfaceSource,
    coordinate_contract: SpatialCoordinateContract,
    provider: MeshingProviderInfo,
    /,
    *,
    minimum_angle: float,
    maximum_normal_angle: float = np.inf,
    limits: MeshCertificateLimits | None = None,
    plan_id: str | None = None,
) -> CellMeshingResult:
    """Accept the authored source theorem before lowering any result constructor."""
    from .._identity import SemanticProvenance
    from ._audit import audit_cell_mesh, CellMeshAuditPolicy
    from ._contracts import MeshingDerivativeMode, MeshingExecutionMode
    from ._organization import lower_mesh_organization, prepare_mesh_organization_scopes
    from ._publication_lowering import PreparedPublicationLowering
    from ._result import (
        CellMeshingResult,
        CollectiveMeshStorageBinding,
        MeshingComplianceReport,
        MeshingRuntimeInfo,
    )
    from ._trace import (
        MeshingStageKind,
        MeshingStageReport,
        MeshingStageStatus,
        MeshingTrace,
    )
    from .providers._native_surface import _associations

    if source.domain.domain_id != prepared.compiled.domain.domain_id:
        raise ValueError(
            "Initial publication source differs from the actual compiled geometry."
        )
    evidence = InitialCollectiveMeshEvidence(
        prepared,
        generated,
        minimum_angle=minimum_angle,
        maximum_normal_angle=maximum_normal_angle,
        limits=limits,
    )
    projection = PreparedPublicationLowering(
        None, evidence, None, vertex_capacity=prepared.layout.vertex_capacity
    ).execute()
    projection.require_source(evidence.logical_arrays)
    storage_binding = CollectiveMeshStorageBinding(
        evidence, evidence.logical_arrays, evidence.global_entity_counts
    )
    scopes = prepare_mesh_organization_scopes(
        evidence, evidence, projection, evidence.compiled.domain.source_revision
    )
    receipts = {
        name: np.asarray(jax.device_get(value))
        for name, value in projection.addressable_arrays(jax.process_index())
    }
    local_angle = (
        np.inf if generated.construction is None else generated.construction.minimum_angle
    )
    achieved_angle = float(
        np.min(
            np.asarray(
                multihost_utils.process_allgather(np.float64(local_angle), tiled=False)
            )
        )
    )
    quality_target = prepared.specification.quality_target
    compliance = MeshingComplianceReport(
        prepared.specification.specification_id,
        requested=()
        if quality_target is None
        else (("minimum_angle", quality_target.minimum_angle),),
        achieved=(("minimum_angle", achieved_angle),),
    )
    result = None
    failure = None
    try:
        mesh = _lower_initial_mesh(evidence, receipts, projection.geometry)
        if mesh.storage is None:
            raise ValueError("Initial publication lost its actual logical storage.")
        geometry = mesh.storage.restore_geometry()
        associations = ()
        if np.any(receipts["closure/cell_valid"]):
            construction = _lower_initial_construction(
                mesh, receipts, generated, minimum_angle
            )
            associations, _, _ = _associations(source, mesh, construction)
        patches, zones, labels, attributes = lower_mesh_organization(
            evidence, mesh, dict(evidence.logical_arrays), scope_projections=scopes
        )
        region_boundaries = _initial_region_boundaries(evidence, mesh, patches, labels)
        audit = audit_cell_mesh(
            mesh,
            geometry,
            policy=CellMeshAuditPolicy(require_complete_association=True),
            patches=patches,
            labels=labels,
            zones=zones,
            attributes=attributes,
            associations=associations,
        )
        audit.require_passed()
        audit.require_decided()
        trace = MeshingTrace(
            (
                MeshingStageReport(
                    MeshingStageKind.SURFACE_MESHING,
                    MeshingStageStatus.PASSED,
                    input_ids=(
                        prepared.compiled.compiled_id,
                        prepared.specification.specification_id,
                    ),
                    output_ids=(evidence.evidence_id, audit.report_id),
                ),
            )
        )
        result = CellMeshingResult(
            mesh,
            geometry,
            coordinate_contract,
            audit,
            audit.quality,
            compliance,
            trace,
            provider,
            MeshingRuntimeInfo(
                provider.provider_id,
                provider.version,
                MeshingExecutionMode.IN_PROCESS,
                deterministic=True,
                enforced_limits=(
                    "source_theorem",
                    "collective_acceptance",
                    "fixed_capacity_closure",
                ),
            ),
            MeshingDerivativeMode.NONDIFFERENTIABLE,
            SemanticProvenance(
                {
                    "kind": "native-distributed-initial-publication",
                    "source": source.binding_id,
                    "theorem": evidence.evidence_id,
                    "plan": plan_id,
                }
            ),
            patches=patches,
            zones=zones,
            labels=labels,
            attributes=attributes,
            associations=associations,
            region_boundary_evidence=region_boundaries,
            collective_evidence=evidence,
            scope_projections=scopes,
            storage_binding=storage_binding,
        )
    except (ValueError, MeshingFailure) as error:
        failure = error if isinstance(error, MeshingFailure) else _failure(str(error))
    _consensus(failure)
    if result is None:
        raise ValueError(
            "Successful initial publication must produce an actual result on every owner."
        )
    return result


__all__ = ["InitialCollectiveMeshEvidence", "publish_distributed_surface_initial"]
