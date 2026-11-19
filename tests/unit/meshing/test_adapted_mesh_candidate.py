"""Pending adaptations (`prove=False`) and their proof."""

from itertools import combinations
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import CellMesh
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.geometry.implicit import (
    AdaptiveImplicitSurfacePolicy,
    AnalyticImplicitProfile,
)
from phydrax.meshing import (
    AdaptedMeshCandidate,
    execute_mesh_adaptation,
    ImplicitAssociationTransfer,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationResult,
    MeshAdaptationRoute,
    MeshMetricField,
    MetricMeshAdaptation,
    prepare_mesh_adaptation,
    PreparedMeshAdaptation,
)
from phydrax.meshing._adaptation import _PendingTetraAdaptation
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._tetra_metric import TetraMetricEvidence
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers._native import NativeMeshingProvider
from phydrax.meshing.providers._native_implicit import (
    normalize_analytic_implicit_source,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from tests.unit.meshing.test_bisection import _cell_ids, _certified, _triangle_grid
from tests.unit.meshing.test_native_implicit_routes import _geometry_and_grid


def _implicit_volume(maximum_deviation: float) -> tuple[Any, Any]:
    """An accepted adaptive implicit sphere volume and its authoring profile.

    The profile's tube radius is 0.5. A metric adaptation moves boundary facets
    inside the source tube only when the accepted fidelity tolerance (here the
    maximum deviation) is below it; otherwise the boundary facets are those of
    the accepted mesh and keep their exact membership tests.
    """
    geometry, _ = _geometry_and_grid()
    contract = phx.SpatialCoordinateContract.si()
    profile = AnalyticImplicitProfile(
        geometry,
        contract,
        tube_radius=0.5,
        source_id="candidate-implicit-profile",
        source_revision="candidate-implicit-r1",
    )
    policy = AdaptiveImplicitSurfacePolicy(
        relative_cell_diameter=0.08, root_tolerance=1.0e-8
    )
    source = normalize_analytic_implicit_source(profile, policy, contract)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(),
        maximum_deviation=maximum_deviation,
    )
    options = NativeMeshingOptions(
        "implicit_adaptive_tetrahedral",
        implicit_policy=policy,
        volume_schedule=NativeVolumeSchedule(),
    )
    accepted = (
        NativeMeshingProvider(options)
        .plan(source, specification, coordinate_contract=contract)
        .execute(prove=True)
    )
    return profile, accepted


@pytest.fixture(scope="module")
def _fixed_boundary_volume() -> tuple[Any, Any]:
    return _implicit_volume(0.8)


@pytest.fixture(scope="module")
def _tube_volume() -> tuple[Any, Any]:
    return _implicit_volume(0.2)


def _prepared_bisection() -> PreparedMeshAdaptation:
    source = _certified(_triangle_grid(1, 1))
    request = MarkedMeshAdaptation(_cell_ids(source.mesh), np.empty(0, dtype=np.int64))
    return prepare_mesh_adaptation(
        source, request, policy=MeshAdaptationPolicy(MeshAdaptationRoute.NATIVE_BISECTION)
    )


def _prepared_coarsening(profile: Any, source: Any) -> PreparedMeshAdaptation:
    """A uniform metric 1.5 times the median edge of the accepted volume."""
    coordinates = np.asarray(source.mesh.coordinates, dtype=np.float64)
    connectivity = source.mesh.connectivity
    assert isinstance(connectivity, SimplicialConnectivity)
    edges = np.asarray(connectivity.entities[1], dtype=np.int64)
    lengths = np.linalg.norm(coordinates[edges[:, 1]] - coordinates[edges[:, 0]], axis=1)
    size = 1.5 * float(np.median(lengths))
    vertices = source.mesh.entity_set(0)
    scope = MeshingScope(
        source.mesh.mesh_id,
        source.mesh.numeric_version,
        MeshingEntityKind.MESH,
        0,
        vertices.entity_set_id,
        vertices.entity_ids,
    )
    metric = MeshMetricField(
        scope,
        np.broadcast_to(np.eye(3) / size**2, (len(coordinates), 3, 3)).copy(),
        minimum_size=size,
        maximum_size=size,
        maximum_anisotropy=1.0,
    )
    return prepare_mesh_adaptation(
        source,
        MetricMeshAdaptation(metric),
        policy=MeshAdaptationPolicy(
            MeshAdaptationRoute.NATIVE_METRIC_3D,
            association_transfer=ImplicitAssociationTransfer(profile),
        ),
    )


def _assert_closed_positive_volume(mesh: CellMesh) -> None:
    """Positive tetrahedra, faces shared by at most two, and a closed boundary."""
    points = np.asarray(mesh.coordinates)
    cells = np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    corners = points[cells]
    assert np.all(np.linalg.det(corners[:, 1:] - corners[:, :1]) > 0.0)
    faces = np.sort(
        np.concatenate([cells[:, list(rows)] for rows in combinations(range(4), 3)]),
        axis=1,
    )
    unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
    assert np.all((counts == 1) | (counts == 2))
    boundary = unique_faces[counts == 1]
    edges = np.sort(
        np.concatenate([boundary[:, list(rows)] for rows in combinations(range(3), 2)]),
        axis=1,
    )
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)


def test_prove_requires_bool() -> None:
    with pytest.raises(TypeError, match="prove must be bool\\."):
        execute_mesh_adaptation(_prepared_bisection(), prove=1)  # ty: ignore[no-matching-overload]


def test_route_without_a_screened_form_says_so_and_proves_to_the_same_result() -> None:
    prepared = _prepared_bisection()
    candidate = execute_mesh_adaptation(prepared)
    assert isinstance(candidate, AdaptedMeshCandidate)
    assert candidate.certified_steps == ("entire_certified_route",)
    assert not hasattr(candidate, "certification")
    assert not hasattr(candidate, "audit")
    assert not hasattr(candidate, "compliance")
    accepted = execute_mesh_adaptation(prepared, prove=True)
    proved = candidate.prove()
    assert proved.result_id == accepted.result_id
    assert proved.target.mesh.mesh_id == accepted.target.mesh.mesh_id
    assert candidate.mesh.mesh_id == accepted.target.mesh.mesh_id
    assert candidate.prove().result_id == proved.result_id


def test_screened_tetra_candidate_names_exact_source_faces_and_proves(
    _fixed_boundary_volume: tuple[Any, Any],
) -> None:
    profile, source = _fixed_boundary_volume
    candidate = execute_mesh_adaptation(_prepared_coarsening(profile, source))
    assert isinstance(candidate, AdaptedMeshCandidate)
    # The tolerance of this volume is not below the tube radius, so its
    # boundary facets keep their exact membership tests, and the candidate
    # says so; no acceptance ran.
    assert candidate.certified_steps == ("source_face_membership",)
    assert not hasattr(candidate, "certification")
    assert isinstance(candidate.evidence, TetraMetricEvidence)
    assert candidate.evidence.collapses > 0
    _assert_closed_positive_volume(candidate.mesh)
    result = candidate.prove()
    assert isinstance(result, MeshAdaptationResult)
    assert result.target.mesh.mesh_id == candidate.mesh.mesh_id
    assert result.target.certification is not None
    assert result.target.certification.passed


def test_screened_tetra_candidate_in_the_source_tube_names_its_admission_and_proves(
    _tube_volume: tuple[Any, Any],
) -> None:
    profile, source = _tube_volume
    candidate = execute_mesh_adaptation(_prepared_coarsening(profile, source))
    assert isinstance(candidate, AdaptedMeshCandidate)
    # The boundary facets are implicit here. Vertices they carry are not
    # relocated under screened steering, and a facet created by a split,
    # collapse or reconnection is still admitted by the interval proof, which
    # the candidate names; no acceptance ran.
    assert candidate.certified_steps == ("implicit_facet_admission",)
    assert not hasattr(candidate, "certification")
    assert isinstance(candidate.evidence, TetraMetricEvidence)
    assert candidate.evidence.collapses > 0
    _assert_closed_positive_volume(candidate.mesh)
    result = candidate.prove()
    assert isinstance(result, MeshAdaptationResult)
    assert result.target.mesh.mesh_id == candidate.mesh.mesh_id
    assert result.target.certification is not None
    assert result.target.certification.passed


def test_pending_adaptation_with_a_tampered_edit_is_refused_by_its_proof(
    _fixed_boundary_volume: tuple[Any, Any],
) -> None:
    profile, source = _fixed_boundary_volume
    prepared = _prepared_coarsening(profile, source)
    candidate = execute_mesh_adaptation(prepared)
    pending = candidate._outcome
    assert isinstance(pending, _PendingTetraAdaptation)
    # The proof works on the retained edit. With two corners of its first cell
    # exchanged (the same vertices and coordinates, one tetrahedron of negative
    # orientation) the edit no longer assembles to the candidate's mesh, and
    # the proof refuses it instead of certifying anything.
    block = pending.outcome.edit.blocks[0]
    inverted = np.array(block.cells, copy=True)
    inverted[0, [0, 1]] = inverted[0, [1, 0]]
    edit = pending.outcome.edit._replace(
        blocks=(block._replace(cells=inverted), *pending.outcome.edit.blocks[1:])
    )
    tampered = AdaptedMeshCandidate(
        prepared,
        pending._replace(outcome=pending.outcome._replace(edit=edit)),
        candidate.certified_steps,
        candidate.construction_seconds,
    )
    with pytest.raises(ValueError):
        tampered.prove()
