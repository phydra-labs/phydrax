import jax.numpy as jnp
import pytest

import phydrax as phx


_TETRA_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)


def _tetra_vertices(shift=(0.0, 0.0, 0.0), scale=1.0):
    base = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return scale * base + jnp.asarray(shift)


def _two_component_region(*, second_faces=_TETRA_FACES, shift=(3.0, 0.0, 0.0), scale=1.0):
    vertices = jnp.concatenate((_tetra_vertices(), _tetra_vertices(shift, scale)))
    faces = jnp.concatenate((_TETRA_FACES, second_faces + 4))
    return phx.geometry.MeshRegion(vertices, faces)


def _fast_policy():
    return phx.operators.LaplaceSingleLayerDP0GalerkinPolicy3D(
        singular_order=3,
        near_ratio=1.0,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
    )


def test_triangle_topology_retains_face_components_and_region_mesh_order():
    region = _two_component_region()
    mesh = region.triangle_mesh

    assert mesh.topology.num_face_components == 2
    assert jnp.array_equal(
        mesh.topology.face_component_ids,
        jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32),
    )
    assert jnp.array_equal(mesh.faces, region.faces)


def test_surface_binding_uses_one_dp0_dof_and_panel_per_face():
    prepared = phx.operators.prepare_laplace_single_layer_dp0_3d(
        _two_component_region(), policy=_fast_policy(), numeric_version="fixture"
    )

    assert prepared.face_count == 8
    assert prepared.component_count == 2
    assert prepared.surface_entities.count == 8
    assert prepared.panelization.panel_count == 8
    assert prepared.assembly_report.numeric_version == "fixture"
    assert sum(prepared.assembly_report.pair_counts) == 64
    assert jnp.all(prepared.face_areas > 0.0)


def test_surface_binding_rejects_inward_or_unseparated_components():
    inward = _TETRA_FACES[:, [0, 2, 1]]
    inward_region = _two_component_region(
        second_faces=inward,
        shift=(3.0, 0.0, 0.0),
        scale=0.5,
    )
    with pytest.raises(ValueError, match="positive outward signed volume"):
        phx.operators.prepare_laplace_single_layer_dp0_3d(
            inward_region, policy=_fast_policy()
        )

    overlapping_bounds = _two_component_region(shift=(0.5, 0.0, 0.0))
    with pytest.raises(ValueError, match="strictly separated component bounding boxes"):
        phx.operators.prepare_laplace_single_layer_dp0_3d(
            overlapping_bounds, policy=_fast_policy()
        )
