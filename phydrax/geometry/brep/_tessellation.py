#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native source-chart tessellation through the canonical surface generator."""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from typing import NamedTuple, TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from ...geometry._meshing_domain import MeshingDomain
from ...meshing._contracts import CellFamilyPolicy, CellMeshingTarget, SurfaceMeshingSpec
from ...meshing._scope import MeshingScope
from ...meshing._sizing import UniformSizeControl
from ...meshing.providers._native_options import NativeMeshingOptions
from ...meshing.providers._native_schedules import (
    NativeCurveSchedule,
    NativeSurfaceSchedule,
)
from ...meshing.providers._native_surface import prepare_native_surface
from ...typing import checked
from ._assembly import BRepAffinePlacement, BRepAssemblyOccurrence
from ._constructors import BRepTessellationPolicy
from ._model import BRepModel, BRepTessellation


if TYPE_CHECKING:
    from ...discretization._surface_chart_records import SurfaceChartWitness
    from ...meshing._native_construction import NativeMeshConstruction
    from ._patches import AbstractSurfacePatch


class _TessellationUnresolved(NotImplementedError):
    """A native chart could not be enclosed or resolved at its numeric precision."""


class _TessellationNonconvergent(_TessellationUnresolved):
    """A violated continuous bound did not contract under one refinement.

    ``bound`` names the violated bound (``"chord"`` or ``"normal"``); ``history``
    holds ``(linear_target, maximum_chord_bound, maximum_normal_bound)`` for every
    refinement attempt, the last two of which fail to contract.
    """

    def __init__(
        self,
        bound: str,
        history: tuple[tuple[float, float, float], ...],
        /,
    ) -> None:
        self.bound = bound
        self.history = history
        super().__init__(
            f"the maximum continuous {bound} bound did not contract under refinement: "
            + "; ".join(
                f"target {target:.6g}: chord {chord:.6g}, normal {normal:.6g}"
                for target, chord, normal in history
            )
        )


class _TessellationRows(NamedTuple):
    vertices: np.ndarray
    triangles: np.ndarray
    face_ids: np.ndarray
    parameters: np.ndarray
    vertex_source_dimensions: np.ndarray
    vertex_source_indices: np.ndarray
    vertex_parameters: np.ndarray
    triangle_occurrence_ids: np.ndarray
    vertex_occurrence_ids: np.ndarray
    vertex_global_ids: np.ndarray
    triangle_global_ids: np.ndarray
    source_chart_witness: SurfaceChartWitness


def _world_patch_points(
    patch: AbstractSurfacePatch,
    parameters: np.ndarray,
    placement: BRepAffinePlacement | None,
    /,
) -> np.ndarray:
    values = np.asarray(
        patch.evaluate(jnp.asarray(parameters, dtype=jnp.float64)), dtype=np.float64
    ).reshape((-1, 3))
    if placement is None:
        return values
    matrix = np.asarray(
        [
            [float(placement.matrix[row][column]) for column in range(3)]
            for row in range(3)
        ],
        dtype=np.float64,
    )
    offset = np.asarray([float(placement.matrix[row][3]) for row in range(3)])
    return values @ matrix.T + offset


def _match_triangle_parameters(
    patch: AbstractSurfacePatch,
    chart: np.ndarray,
    exact_chart: tuple[tuple[Fraction, Fraction], ...],
    target_vertices: np.ndarray,
    placement: BRepAffinePlacement | None,
    /,
) -> tuple[np.ndarray, tuple[tuple[Fraction, Fraction], ...]]:
    world = _world_patch_points(patch, chart, placement)
    costs = np.linalg.norm(world[:, None, :] - target_vertices[None, :, :], axis=2)
    choices = sorted(
        permutations(range(3)),
        key=lambda order: max(
            costs[source, target] for source, target in enumerate(order)
        ),
    )
    best = choices[0]
    scale = max(1.0, float(np.max(np.abs(target_vertices), initial=0.0)))
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    if max(costs[source, target] for source, target in enumerate(best)) > tolerance:
        raise _TessellationUnresolved(
            "native surface chart ancestry does not reproduce the generated triangle vertices"
        )
    parameters = np.empty((3, 2), dtype=np.float64)
    for source_corner, target_corner in enumerate(best):
        parameters[target_corner] = chart[source_corner]
    exact_parameters = tuple(
        exact_chart[best.index(target_corner)] for target_corner in range(3)
    )
    return parameters, exact_parameters


def _source_occurrence_data(
    model: BRepModel, /
) -> tuple[tuple[BRepAssemblyOccurrence, ...], dict[tuple[str, ...], int]]:
    occurrences = tuple(model.geometry.assembly_graph.iter_occurrences())
    return occurrences, {value.path: index for index, value in enumerate(occurrences)}


def _construction_arrays(
    model: BRepModel,
    domain: MeshingDomain,
    construction: NativeMeshConstruction,
    /,
) -> _TessellationRows:
    from ...discretization._cell_geometry_validity import cell_geometry_id
    from ...discretization._surface_chart_records import (
        SurfaceRootMeshCellBinding,
    )
    from ...meshing._association import GeometryAssociationKind

    mesh = construction.mesh
    if mesh.topological_dimension != 2 or mesh.ambient_dimension != 3:
        raise _TessellationUnresolved(
            "native surface generation did not return a 2D-in-3D mesh"
        )
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind != "triangle":
        raise _TessellationUnresolved(
            "native B-Rep tessellation requires one conforming triangle block"
        )
    organization = construction.organization
    source = None if organization is None else organization.surface_source
    if source is None:
        raise _TessellationUnresolved(
            "native generation did not retain source chart ancestry"
        )
    source.require_current()
    witness = source.chart_witness

    root_cell_ids = tuple(
        int(value)
        for block in source.mesh.blocks
        for value in np.asarray(block.global_ids, dtype=np.int64)
    )
    witness_cell_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    if set(root_cell_ids) != set(witness_cell_ids.tolist()):
        raise _TessellationUnresolved(
            "native surface chart witness does not bind every retained root mesh cell"
        )
    if source.mesh.mesh_id != mesh.mesh_id or cell_geometry_id(
        source.geometry
    ) != cell_geometry_id(construction.geometry):
        raise _TessellationUnresolved(
            "native surface source chart root differs from its generated mesh"
        )
    root_geometry_id = cell_geometry_id(source.geometry)
    root_topology_id = source.mesh.topology_id
    for row, cell_id in enumerate(witness_cell_ids.tolist()):
        expected_binding = SurfaceRootMeshCellBinding(
            root_geometry_id,
            root_topology_id,
            int(cell_id),
        )
        if witness.root_mesh_cell_bindings[row] != expected_binding:
            raise _TessellationUnresolved(
                "native surface chart witness does not retain "
                "its actual root mesh cell binding"
            )
    block = mesh.blocks[0]
    vertices = np.asarray(mesh.coordinates, dtype=np.float64)
    triangles = np.asarray(block.vertices, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise _TessellationUnresolved("native triangle connectivity is malformed")
    witness_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(witness.cell_global_ids, dtype=np.int64)
        )
    }
    occurrences, occurrence_rows = _source_occurrence_data(model)
    face_ids = np.empty((triangles.shape[0],), dtype=np.int32)
    triangle_occurrences = np.full((triangles.shape[0],), -1, dtype=np.int32)
    parameters = np.empty((triangles.shape[0], 3, 2), dtype=np.float64)
    exact_charts: list[tuple[tuple[Fraction, Fraction], ...]] = []
    for row, cell_id in enumerate(np.asarray(block.global_ids, dtype=np.int64).tolist()):
        try:
            witness_row = witness_rows[int(cell_id)]
        except KeyError as error:
            raise _TessellationUnresolved(
                "native surface chart witness omits a mesh triangle"
            ) from error
        domain_patch = int(np.asarray(witness.patches, dtype=np.int64)[witness_row])
        if domain_patch < 0 or domain_patch >= len(domain.source_indices[2]):
            raise _TessellationUnresolved(
                "native surface witness references an absent source patch"
            )
        face = int(domain.source_indices[2][domain_patch])
        path = domain.source_occurrences[2][domain_patch]
        occurrence_id = -1 if not path else occurrence_rows.get(path, -2)
        if occurrence_id == -2:
            raise _TessellationUnresolved(
                "native surface witness names an unknown source occurrence"
            )
        chart = np.asarray(witness.charts, dtype=np.float64)[witness_row]
        source_box = np.stack((np.min(chart, axis=0), np.max(chart, axis=0)))
        try:
            source_c1 = model.patches[face].is_c1_on(source_box)
        except (NotImplementedError, TypeError, ValueError) as error:
            raise _TessellationUnresolved(
                f"source-face C1 continuity is unresolved: {error}"
            ) from error
        if not source_c1:
            raise _TessellationUnresolved(
                "source-face C1 continuity is not established over a tessellation cell"
            )
        exact_chart = witness.exact_chart(witness_row)
        placement = None if occurrence_id < 0 else occurrences[occurrence_id].placement
        parameters[row], exact_parameters = _match_triangle_parameters(
            model.patches[face],
            chart,
            exact_chart,
            vertices[triangles[row]],
            placement,
        )
        exact_charts.append(exact_parameters)
        face_ids[row] = face
        triangle_occurrences[row] = occurrence_id

    from ...discretization._surface_chart_records import (
        rational_chart_tokens,
        SurfaceChartWitness,
    )

    triangle_global_ids = np.asarray(block.global_ids, dtype=np.int64)
    aligned_witness_rows = tuple(
        witness_rows[int(identifier)] for identifier in triangle_global_ids.tolist()
    )
    aligned_witness = SurfaceChartWitness(
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        domain_id=witness.domain_id,
        cell_global_ids=triangle_global_ids,
        patches=np.asarray(witness.patches, dtype=np.int64)[
            np.asarray(aligned_witness_rows, dtype=np.int64)
        ],
        charts=parameters,
        chart_rational_tokens=rational_chart_tokens(tuple(exact_charts)),
        source_root_cell_ids=np.asarray(witness.source_root_cell_ids, dtype=np.int64)[
            np.asarray(aligned_witness_rows, dtype=np.int64)
        ],
        source_piece_ids=tuple(
            witness.source_piece_ids[row] for row in aligned_witness_rows
        ),
        geometry_entity_ids=tuple(
            witness.geometry_entity_ids[row] for row in aligned_witness_rows
        ),
        occurrence_paths=tuple(
            witness.occurrence_paths[row] for row in aligned_witness_rows
        ),
        root_mesh_cell_bindings=tuple(
            witness.root_mesh_cell_bindings[row] for row in aligned_witness_rows
        ),
    )
    vertex_set = mesh.entity_set(0)
    vertex_set_id = vertex_set.entity_set_id
    associations = tuple(
        value
        for value in organization.associations
        if value.target_entity_set_id == vertex_set_id
        and value.association_kind is GeometryAssociationKind.BREP
    )
    if len(associations) != 1:
        raise _TessellationUnresolved(
            "native generation did not provide one B-Rep vertex association"
        )
    association = associations[0]
    if (
        association.source_id != model.source_id
        or association.source_revision != model.source_revision
        or not np.all(np.asarray(association.resolved, dtype=np.bool_))
        or np.any(np.asarray(association.ambiguous, dtype=np.bool_))
    ):
        raise _TessellationUnresolved(
            "native B-Rep vertex ancestry is stale, ambiguous, or unresolved"
        )
    association_rows = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(association.target_global_ids, dtype=np.int64)
        )
    }
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    dimensions = np.empty((vertices.shape[0],), dtype=np.int32)
    source_indices = np.empty((vertices.shape[0],), dtype=np.int64)
    vertex_parameters = np.empty((vertices.shape[0], 2), dtype=np.float64)
    vertex_occurrences = np.full((vertices.shape[0],), -1, dtype=np.int32)
    assoc_dimensions = np.asarray(association.source_dimensions, dtype=np.int64)
    assoc_indices = np.asarray(association.source_indices, dtype=np.int64)
    assoc_parameters = np.asarray(association.parameters, dtype=np.float64)
    for row, identifier in enumerate(vertex_ids.tolist()):
        try:
            source_row = association_rows[int(identifier)]
        except KeyError as error:
            raise _TessellationUnresolved(
                "native B-Rep vertex association omits a mesh vertex"
            ) from error
        dimension = int(assoc_dimensions[source_row])
        index = int(assoc_indices[source_row])
        path = association.source_occurrence_paths[source_row]
        occurrence_id = -1 if not path else occurrence_rows.get(path, -2)
        if occurrence_id == -2:
            raise _TessellationUnresolved(
                "native vertex association names an unknown source occurrence"
            )
        parameters_row = np.nan_to_num(assoc_parameters[source_row], nan=0.0)
        if (
            dimension not in (0, 1, 2)
            or index < 0
            or not np.all(np.isfinite(parameters_row))
        ):
            raise _TessellationUnresolved(
                "native B-Rep vertex association has invalid source parameters"
            )
        if dimension == 1:
            edge_range = np.asarray(model.geometry.edge_ranges, dtype=np.float64)[index]
            parameters_row[0] = edge_range[0] + parameters_row[0] * (
                edge_range[1] - edge_range[0]
            )
        dimensions[row] = dimension
        source_indices[row] = index
        vertex_parameters[row] = parameters_row
        vertex_occurrences[row] = occurrence_id
    return _TessellationRows(
        vertices,
        triangles,
        face_ids,
        parameters,
        dimensions,
        source_indices,
        vertex_parameters,
        triangle_occurrences,
        vertex_occurrences,
        vertex_ids,
        triangle_global_ids,
        aligned_witness,
    )


def _certify_construction(
    construction: NativeMeshConstruction,
    policy: BRepTessellationPolicy,
    linear_target: float,
    witness: SurfaceChartWitness,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    from ...geometry._source_fidelity import certify_source_fidelity

    organization = construction.organization
    surface_source = None if organization is None else organization.surface_source
    if surface_source is None:
        raise _TessellationUnresolved(
            "native generation did not retain its authoritative surface source and chart witness"
        )
    try:
        certificate = certify_source_fidelity(
            construction.mesh,
            construction.geometry,
            surface_source.boundary_source,
            tolerance=linear_target,
            maximum_normal_deviation=policy.angular_deflection,
            witness=witness,
        )
    except (ArithmeticError, FloatingPointError, TypeError, ValueError) as error:
        raise _TessellationUnresolved(
            f"canonical continuous source-fidelity proof failed: {error}"
        ) from error
    # The proof was derived from exactly these inputs; a full replay would
    # recompute the same certificate. Only its identity binding is checked.
    try:
        certificate.binding.require(
            construction.mesh,
            construction.geometry,
            source=surface_source.boundary_source,
        )
    except (ArithmeticError, FloatingPointError, TypeError, ValueError) as error:
        raise _TessellationUnresolved(
            f"canonical source-fidelity proof is not bound to its construction: {error}"
        ) from error
    block = construction.mesh.blocks[0]
    cell_ids = tuple(int(value) for value in np.asarray(block.global_ids, dtype=np.int64))
    cell_rows = {cell_id: row for row, cell_id in enumerate(cell_ids)}
    certificate_rows = {
        cell_id: row for row, cell_id in enumerate(certificate.source_cell_global_ids)
    }
    if set(certificate_rows) != set(cell_ids):
        raise _TessellationUnresolved(
            "continuous source-fidelity evidence does not cover each generated triangle exactly once"
        )
    deviation_rows = np.asarray(certificate.deviation_bounds, dtype=np.float64)
    normal_rows = certificate.normal_deviation_bounds
    if deviation_rows.shape != (len(cell_ids), 2) or normal_rows is None:
        raise _TessellationUnresolved(
            "canonical source-fidelity evidence omitted per-triangle chord or normal bounds"
        )
    normal_rows = np.asarray(normal_rows, dtype=np.float64)
    if normal_rows.shape != (len(cell_ids),):
        raise _TessellationUnresolved(
            "canonical source-fidelity normal bounds do not align with generated triangles"
        )
    deviations = np.empty((len(cell_ids),), dtype=np.float64)
    normals = np.empty_like(deviations)
    for row, cell_id in enumerate(cell_ids):
        proof_row = certificate_rows[cell_id]
        deviations[row] = deviation_rows[proof_row, 1]
        normals[row] = normal_rows[proof_row]
    for finding in certificate.findings:
        if finding.check != "normal_deviation_range":
            continue
        if not finding.entity_ids:
            normals.fill(np.inf)
            continue
        for cell_id in finding.entity_ids:
            row = cell_rows.get(int(cell_id))
            if row is None:
                normals.fill(np.inf)
            else:
                normals[row] = np.inf
    if not np.all(np.isfinite(deviations)) or np.any(np.isnan(normals)):
        raise _TessellationUnresolved(
            "canonical continuous source-fidelity chord bounds remain unresolved"
        )
    retryable_findings = (
        "source_fidelity_tolerance",
        "normal_deviation_limit",
        "normal_deviation_range",
    )
    unsupported = tuple(
        finding
        for finding in certificate.findings
        if finding.check not in retryable_findings
    )
    if unsupported:
        raise _TessellationUnresolved(
            "canonical source-fidelity proof has unresolved premises: "
            + "; ".join(finding.detail for finding in unsupported)
        )
    if not np.all(np.isfinite(normals)) and not any(
        finding.check == "normal_deviation_range" for finding in certificate.findings
    ):
        raise _TessellationUnresolved(
            "canonical continuous normal bounds remain unresolved without a refinable chart finding"
        )
    return deviations, normals


def _build_request(
    domain: MeshingDomain,
    policy: BRepTessellationPolicy,
    linear_target: float,
    /,
) -> tuple[SurfaceMeshingSpec, NativeMeshingOptions]:
    scope = MeshingScope.whole(domain, 2)
    size = max(float(domain.scale), linear_target)
    target = CellMeshingTarget(
        2, 3, CellFamilyPolicy(required=("triangle",)), require_conforming=True
    )
    size_control = UniformSizeControl(scope, size)
    specification = SurfaceMeshingSpec(
        target,
        scope,
        size_controls=(size_control,),
        maximum_deviation=linear_target,
        maximum_normal_deviation=None,
    )
    schedule = NativeSurfaceSchedule(
        quality_angle_degrees=20.0,
        spacing_fraction=0.95,
        minimum_edge_samples=policy.trim_samples_per_edge,
        curve_schedule=NativeCurveSchedule(fidelity_samples=policy.trim_samples_per_edge),
    )
    return specification, NativeMeshingOptions(
        "parametric_surface", surface_schedule=schedule
    )


@checked
def tessellate_brep_model(
    model: BRepModel,
    policy: BRepTessellationPolicy,
    /,
) -> BRepTessellation:
    """Realize native source charts as conforming linear triangles with enclosures."""
    from ._model import BRepModel, BRepTessellation

    if not isinstance(model, BRepModel):
        raise TypeError("model must be a BRepModel.")
    from ._constructors import BRepTessellationPolicy

    if not isinstance(policy, BRepTessellationPolicy):
        raise TypeError("policy must be a BRepTessellationPolicy.")
    if not model.patches:
        raise ValueError("A faceless B-Rep has no surface tessellation.")

    domain = MeshingDomain.from_brep(model)
    linear_target = policy.linear_deflection
    history: list[tuple[float, float, float]] = []
    while True:
        specification, options = _build_request(domain, policy, linear_target)
        construction = prepare_native_surface(
            domain, specification, options, model.coordinate_contract
        ).execute()
        arrays = _construction_arrays(model, domain, construction)
        (
            vertices,
            triangles,
            face_ids,
            parameters,
            vertex_dimensions,
            vertex_indices,
            vertex_parameters,
            triangle_occurrences,
            vertex_occurrences,
            vertex_global_ids,
            triangle_global_ids,
            source_chart_witness,
        ) = arrays
        deviation_bounds, normal_bounds = _certify_construction(
            construction, policy, linear_target, source_chart_witness
        )
        if np.all(deviation_bounds <= policy.linear_deflection) and np.all(
            normal_bounds <= policy.angular_deflection
        ):
            return BRepTessellation(
                linear_deflection=policy.linear_deflection,
                angular_deflection=policy.angular_deflection,
                source_revision=model.source_revision,
                policy_id=policy.policy_id,
                vertices=vertices,
                triangles=triangles,
                face_ids=face_ids,
                parameters=parameters,
                deviation_bounds=deviation_bounds,
                normal_bounds=normal_bounds,
                vertex_source_dimensions=vertex_dimensions,
                vertex_source_indices=vertex_indices,
                vertex_parameters=vertex_parameters,
                triangle_occurrence_ids=triangle_occurrences,
                vertex_occurrence_ids=vertex_occurrences,
                vertex_global_ids=vertex_global_ids,
                triangle_global_ids=triangle_global_ids,
                source_chart_witness=source_chart_witness,
            )
        # Halving the linear target must contract every violated bound: a
        # maximum chord or normal bound that is violated in two consecutive
        # attempts and does not strictly decrease between them has no
        # refinement evidence of ever meeting the policy.
        attempt = (
            linear_target,
            float(np.max(deviation_bounds)),
            float(np.max(normal_bounds)),
        )
        if history:
            previous = history[-1]
            for bound, axis, limit in (
                ("chord", 1, policy.linear_deflection),
                ("normal", 2, policy.angular_deflection),
            ):
                if (
                    previous[axis] > limit
                    and attempt[axis] > limit
                    and not attempt[axis] < previous[axis]
                ):
                    raise _TessellationNonconvergent(bound, (*history, attempt))
        history.append(attempt)
        tighter = linear_target * 0.5
        if not np.isfinite(tighter) or tighter <= 0.0 or tighter >= linear_target:
            raise _TessellationUnresolved(
                "continuous source-to-triangle bounds remain above policy at "
                "the positive float64 resolution limit"
            )
        linear_target = tighter


__all__ = ["tessellate_brep_model"]
