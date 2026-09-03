#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cardiovascular.anatomy._coordinates import (
    atrial_coordinate_specs,
    biventricular_coordinate_specs,
    HarmonicCoordinateFields,
    HarmonicCoordinatePlan,
    HarmonicCoordinateSpec,
    left_ventricular_coordinate_specs,
)
from phydrax.applications.cardiovascular.anatomy._microstructure import (
    VentricularMicrostructurePlan,
)
from phydrax.applications.cardiovascular.anatomy._roles import (
    atrial_boundary_profile,
    biventricular_boundary_profile,
    CardiacBoundaryProfile,
    CardiacBoundaryRoles,
    left_ventricular_boundary_profile,
    whole_heart_boundary_profile,
)
from phydrax.applications.cardiovascular.anatomy._surfaces import ChamberSurfacePlan
from phydrax.discretization import CellMesh


def _vertex_index(i: int, j: int, k: int, /, *, size: int) -> int:
    return i + size * (j + size * k)


def _affine_lv_slab(subdivisions: int = 2):
    size = subdivisions + 1
    axis = np.linspace(0.0, 1.0, size)
    coordinates = np.asarray(
        [
            (axis[i], axis[j], axis[k])
            for k in range(size)
            for j in range(size)
            for i in range(size)
        ]
    )
    tetrahedra = []
    for k in range(subdivisions):
        for j in range(subdivisions):
            for i in range(subdivisions):
                v000 = _vertex_index(i, j, k, size=size)
                v100 = _vertex_index(i + 1, j, k, size=size)
                v010 = _vertex_index(i, j + 1, k, size=size)
                v110 = _vertex_index(i + 1, j + 1, k, size=size)
                v001 = _vertex_index(i, j, k + 1, size=size)
                v101 = _vertex_index(i + 1, j, k + 1, size=size)
                v011 = _vertex_index(i, j + 1, k + 1, size=size)
                v111 = _vertex_index(i + 1, j + 1, k + 1, size=size)
                tetrahedra.extend(
                    (
                        (v000, v100, v110, v111),
                        (v000, v110, v010, v111),
                        (v000, v010, v011, v111),
                        (v000, v011, v001, v111),
                        (v000, v001, v101, v111),
                        (v000, v101, v100, v111),
                    )
                )
    mesh = CellMesh.from_tetrahedra(
        coordinates,
        np.asarray(tetrahedra, dtype=np.int32),
        vertex_global_ids=np.arange(coordinates.shape[0], dtype=np.int64) + 1000,
    )
    faces = np.asarray(mesh.connectivity.faces)
    face_points = coordinates[faces]
    role_faces = {
        "endocardium": np.flatnonzero(np.all(face_points[..., 0] == 0.0, axis=1)),
        "epicardium": np.flatnonzero(np.all(face_points[..., 0] == 1.0, axis=1)),
        "apex": np.flatnonzero(np.all(face_points[..., 2] == 0.0, axis=1)),
        "base": np.flatnonzero(np.all(face_points[..., 2] == 1.0, axis=1)),
        "posterior": np.flatnonzero(np.all(face_points[..., 1] == 0.0, axis=1)),
        "anterior": np.flatnonzero(np.all(face_points[..., 1] == 1.0, axis=1)),
    }
    orthogonal_pairs = tuple(
        (first, second)
        for first in ("endocardium", "epicardium")
        for second in ("apex", "base", "posterior", "anterior")
    ) + tuple(
        (first, second)
        for first in ("apex", "base")
        for second in ("posterior", "anterior")
    )
    profile = CardiacBoundaryProfile(
        "manufactured-lv-slab",
        required_roles=tuple(role_faces),
        connected_roles=tuple(role_faces),
        disjoint_closure_pairs=(
            ("endocardium", "epicardium"),
            ("apex", "base"),
            ("posterior", "anterior"),
        ),
        shared_closure_pairs=orthogonal_pairs,
        exhaustive=True,
    )
    return mesh, CardiacBoundaryRoles(mesh, role_faces, profile=profile)


def _harmonic_fields():
    mesh, roles = _affine_lv_slab()
    plan = HarmonicCoordinatePlan(
        mesh,
        roles,
        (
            HarmonicCoordinateSpec("transmural", "endocardium", "epicardium"),
            HarmonicCoordinateSpec("longitudinal", "apex", "base"),
        ),
    )
    candidate = plan.prepare(numeric_version="manufactured").solve()
    return mesh, roles, plan, candidate.commit()


def test_boundary_profiles_are_extensible_and_validate_closure_contracts():
    mesh, roles = _affine_lv_slab()
    assert bool(roles.evidence.successful)
    assert int(roles.evidence.unassigned_face_count) == 0
    assert np.all(np.asarray(roles.evidence.role_component_counts) == 1)
    assert set(roles.role_names) == {
        "endocardium",
        "epicardium",
        "apex",
        "base",
        "posterior",
        "anterior",
    }
    assert roles.roles_id == _affine_lv_slab()[1].roles_id
    assert (
        np.intersect1d(
            np.asarray(roles.vertex_indices("endocardium")),
            np.asarray(roles.vertex_indices("epicardium")),
        ).size
        == 0
    )

    overlapping = {
        assignment.name: np.asarray(assignment.face_indices)
        for assignment in roles.assignments
    }
    overlapping["epicardium"] = np.concatenate(
        (overlapping["epicardium"], overlapping["endocardium"][:1])
    )
    with pytest.raises(ValueError, match="disjoint face ownership"):
        CardiacBoundaryRoles(mesh, overlapping, profile=roles.profile)


def test_chamber_profiles_and_coordinate_recipes_keep_explicit_semantics():
    lv_profile = left_ventricular_boundary_profile(
        endocardium="endo",
        epicardium="epi",
        apex="apical-cut",
        base="basal-cut",
    )
    assert set(lv_profile.required_roles) == {
        "endo",
        "epi",
        "apical-cut",
        "basal-cut",
    }
    assert tuple(
        spec.name
        for spec in left_ventricular_coordinate_specs(
            endocardium="endo",
            epicardium="epi",
            apex="apical-cut",
            base="basal-cut",
        )
    ) == ("lv-transmural", "lv-apicobasal")

    biv_profile = biventricular_boundary_profile()
    biv_specs = biventricular_coordinate_specs()
    assert set(biv_profile.required_roles) == {
        "lv-endocardium",
        "rv-endocardium",
        "epicardium",
        "apex",
        "base",
    }
    assert tuple((spec.lower_role, spec.upper_role) for spec in biv_specs) == (
        ("lv-endocardium", "epicardium"),
        ("rv-endocardium", "epicardium"),
        ("apex", "base"),
        ("lv-endocardium", "rv-endocardium"),
    )

    atrial_profile = atrial_boundary_profile(
        left_openings=("left-superior-pv", "left-inferior-pv"),
        right_openings=("svc", "ivc"),
    )
    assert {"left-superior-pv", "left-inferior-pv", "svc", "ivc"}.issubset(
        atrial_profile.required_roles
    )
    assert tuple(spec.name for spec in atrial_coordinate_specs()) == (
        "la-transmural",
        "ra-transmural",
    )

    whole_profile = whole_heart_boundary_profile(
        pulmonary_vein_openings=("lspv", "lipv", "rspv", "ripv"),
        vena_cava_openings=("svc", "ivc"),
    )
    assert {
        "lv-endocardium",
        "rv-endocardium",
        "la-endocardium",
        "ra-endocardium",
        "mitral-plane",
        "tricuspid-plane",
        "aortic-plane",
        "pulmonary-plane",
        "lspv",
        "svc",
    }.issubset(whole_profile.required_roles)

    _, roles = _affine_lv_slab()
    validated = left_ventricular_coordinate_specs(
        endocardium="endocardium",
        epicardium="epicardium",
        apex="apex",
        base="base",
        roles=roles,
    )
    assert len(validated) == 2
    with pytest.raises(ValueError, match="absent from the profile"):
        biventricular_coordinate_specs(roles=roles)
    with pytest.raises(ValueError, match="distinct names"):
        atrial_coordinate_specs(right_endocardium="la-endocardium")


def test_affine_p1_harmonic_coordinates_reproduce_linear_fields_and_gradients():
    mesh, _, plan, fields = _harmonic_fields()
    coordinates = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.blocks[0].vertices)
    np.testing.assert_allclose(fields.nodal("transmural"), coordinates[:, 0], atol=2.0e-6)
    np.testing.assert_allclose(
        fields.nodal("longitudinal"), coordinates[:, 2], atol=2.0e-6
    )
    np.testing.assert_allclose(
        fields.cell("transmural"), np.mean(coordinates[cells, 0], axis=1), atol=2.0e-6
    )
    np.testing.assert_allclose(
        fields.gradient("transmural"),
        np.broadcast_to((1.0, 0.0, 0.0), (cells.shape[0], 3)),
        atol=3.0e-6,
    )
    np.testing.assert_allclose(
        fields.gradient("longitudinal"),
        np.broadcast_to((0.0, 0.0, 1.0), (cells.shape[0], 3)),
        atol=3.0e-6,
    )
    assert bool(fields.evidence.all_successful)
    assert np.max(np.asarray(fields.evidence.maximum_boundary_error)) == 0.0
    assert plan.plan_id == HarmonicCoordinatePlan(mesh, plan.roles, plan.specs).plan_id


def test_exact_helix_rule_material_frame_and_line_tensor_sign_invariance():
    _, _, _, fields = _harmonic_fields()
    plan = VentricularMicrostructurePlan(
        "transmural",
        "longitudinal",
        helix_endocardium_degrees=60.0,
        helix_epicardium_degrees=-60.0,
    )
    microstructure = plan.prepare(fields).build().commit()
    fraction = np.asarray(microstructure.transmural_fraction)
    expected_angle = np.deg2rad(60.0 - 120.0 * fraction)
    np.testing.assert_allclose(
        microstructure.helix_angle_radians, expected_angle, atol=2.0e-6
    )
    expected_fiber = np.stack(
        (
            np.zeros_like(expected_angle),
            np.cos(expected_angle),
            np.sin(expected_angle),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(microstructure.fiber, expected_fiber, atol=3.0e-6)
    frame = np.asarray(microstructure.material_frame.matrix)
    np.testing.assert_allclose(
        np.swapaxes(frame, -1, -2) @ frame,
        np.broadcast_to(np.eye(3), frame.shape),
        atol=4.0e-6,
    )
    np.testing.assert_allclose(np.linalg.det(frame), 1.0, atol=4.0e-6)
    assert bool(microstructure.evidence.all_successful)

    reversed_gradients = np.asarray(fields.cell_gradients).copy()
    reversed_gradients[fields.coordinate_index("longitudinal")] *= -1.0
    reversed_fields = HarmonicCoordinateFields(
        fields.names,
        fields.nodal_values,
        fields.cell_values,
        reversed_gradients,
        fields.dirichlet_masks,
        fields.evidence,
        fields_id="longitudinal-gauge-reversed",
    )
    reversed_microstructure = plan.prepare(reversed_fields).build().commit()
    np.testing.assert_allclose(
        reversed_microstructure.fiber_structure_tensor,
        microstructure.fiber_structure_tensor,
        atol=4.0e-6,
    )
    np.testing.assert_allclose(
        reversed_microstructure.fiber,
        -np.asarray(microstructure.fiber),
        atol=4.0e-6,
    )


def test_microstructure_degeneracy_is_fail_closed_without_epsilon_repair():
    _, _, _, fields = _harmonic_fields()
    degenerate_gradients = np.asarray(fields.cell_gradients).copy()
    degenerate_gradients[fields.coordinate_index("longitudinal")] = degenerate_gradients[
        fields.coordinate_index("transmural")
    ]
    degenerate = HarmonicCoordinateFields(
        fields.names,
        fields.nodal_values,
        fields.cell_values,
        degenerate_gradients,
        fields.dirichlet_masks,
        fields.evidence,
        fields_id="parallel-coordinate-gradients",
    )
    candidate = (
        VentricularMicrostructurePlan(
            "transmural",
            "longitudinal",
            gradient_tolerance=1.0e-12,
        )
        .prepare(degenerate)
        .build()
    )
    assert not bool(candidate.evidence.all_successful)
    assert np.all(~np.asarray(candidate.evidence.nondegenerate))
    assert np.all(np.isnan(np.asarray(candidate.material_frame.fiber)))


def _tetrahedral_cavity():
    vertices = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    scrambled_faces = jnp.asarray(((3, 2, 1), (2, 0, 3), (1, 3, 0), (0, 2, 1)))
    return vertices, scrambled_faces


def test_closed_chamber_orientation_volume_derivative_and_translation_evidence():
    vertices, triangles = _tetrahedral_cavity()
    surface = ChamberSurfacePlan("manufactured-lv-cavity", vertices, triangles).prepare()
    result = surface.evaluate().commit()
    np.testing.assert_allclose(result.volume, 1.0 / 6.0, atol=2.0e-7)
    assert bool(surface.topology_evidence.successful)
    assert np.all(np.asarray(surface.topology_evidence.edge_incidence_counts) == 2)
    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.evidence.translation_volume_error, 0.0, atol=2.0e-7)
    np.testing.assert_allclose(
        result.evidence.translation_derivative_norm, 0.0, atol=2.0e-7
    )

    differentiated = jax.grad(lambda points: surface.evaluate(points).volume)(vertices)
    np.testing.assert_allclose(result.coordinate_derivative, differentiated, atol=2.0e-7)
    translated = surface.evaluate(vertices + jnp.asarray((11.0, -7.0, 3.5))).commit()
    np.testing.assert_allclose(translated.volume, result.volume, atol=2.0e-7)

    reflected = vertices.at[:, 0].multiply(-1.0)
    reflected_candidate = surface.evaluate(reflected)
    assert not bool(reflected_candidate.evidence.successful)
    assert not bool(reflected_candidate.evidence.positive_orientation)


def test_chamber_surface_ids_are_canonical_and_open_surfaces_are_rejected():
    vertices, triangles = _tetrahedral_cavity()
    first = ChamberSurfacePlan("lv", vertices, triangles)
    second = ChamberSurfacePlan("lv", vertices, triangles[::-1, ::-1])
    assert first.plan_id == second.plan_id
    assert first.prepare().surface_id == second.prepare().surface_id
    open_vertices = jnp.concatenate((vertices, jnp.asarray(((1.0, 1.0, 1.0),))), axis=0)
    open_faces = jnp.asarray(((0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 4)))
    with pytest.raises(ValueError, match="exactly two faces per edge"):
        ChamberSurfacePlan("open-lv", open_vertices, open_faces).prepare()
