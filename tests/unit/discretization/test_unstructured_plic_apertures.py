#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume._embedded_dynamics import (
    lower_embedded_stage_metrics,
    UnstructuredEmbeddedBoundarySet,
)
from phydrax.discretization.finite_volume._geometry_protocol import (
    lower_static_unstructured_stage_metrics,
)
from phydrax.discretization.finite_volume._unstructured_vof import (
    JAXPLICStageReconstruction,
    PLICInterfaceStatus,
    PLICReconstruction,
)


def _grid(
    nx=2,
    ny=2,
    *,
    shift=(0.0, 0.0),
    angle=0.0,
    reverse=False,
):
    vertices = np.asarray(
        [(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle)),
            (np.sin(angle), np.cos(angle)),
        )
    )
    vertices = vertices @ rotation.T + np.asarray(shift)
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            cell = [lower_left, lower_left + 1, lower_left + nx + 2, lower_left + nx + 1]
            cells.append(cell[::-1] if reverse else cell)
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, quadrilaterals=np.asarray(cells)
    ).prepare()


def _triangular_grid(nx=2, ny=2):
    vertices = np.asarray(
        [(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    cells = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, triangles=np.asarray(cells)
    ).prepare()


def _vof(discretization):
    gradient = phx.discretization.CellPolynomialReconstructionPlan(
        1, oversampling=4
    ).prepare(discretization)
    return phx.discretization.UnstructuredVOFPlan(discretization, gradient)


def test_planar_plic_apertures_are_complementary_and_exact_on_static_routes():
    discretization = _grid()
    vof = _vof(discretization)
    alpha = jnp.full((discretization.cell_count,), 0.5)
    plic = vof.reconstruct(alpha)
    metrics = lower_static_unstructured_stage_metrics(discretization)
    apertures = vof.face_phase_apertures(alpha, plic, metrics)

    assert isinstance(plic, PLICReconstruction)
    assert jnp.all(plic.interface_evidence)
    assert not jnp.any(plic.interface_status == int(PLICInterfaceStatus.AMBIGUOUS))
    np.testing.assert_allclose(
        apertures.owner_phase_apertures[:, 0] + apertures.owner_phase_apertures[:, 1],
        1.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        apertures.receptor_phase_apertures[:, 0]
        + apertures.receptor_phase_apertures[:, 1],
        1.0,
        atol=2e-12,
    )
    assert jnp.all(jnp.isfinite(apertures.owner_phase_centroids))


def test_translation_and_reversed_cell_orientation_preserve_phase_complements():
    first = _grid()
    translated = _grid(shift=(3.0, -2.0), reverse=True)
    first_vof = _vof(first)
    translated_vof = _vof(translated)
    first_alpha = jnp.full((first.cell_count,), 0.5)
    translated_alpha = jnp.full((translated.cell_count,), 0.5)
    first_apertures = first_vof.face_phase_apertures(
        first_alpha,
        first_vof.reconstruct(first_alpha),
        lower_static_unstructured_stage_metrics(first),
    )
    translated_apertures = translated_vof.face_phase_apertures(
        translated_alpha,
        translated_vof.reconstruct(translated_alpha),
        lower_static_unstructured_stage_metrics(translated),
    )

    np.testing.assert_allclose(
        first_apertures.owner_phase_apertures,
        translated_apertures.owner_phase_apertures,
        atol=2e-12,
    )
    assert first.geometry_id != translated.geometry_id


def test_phase_swept_flux_has_exact_sum_and_jit_stage_evaluation():
    discretization = _grid()
    vof = _vof(discretization)
    alpha = jnp.asarray((0.25, 0.5, 0.75, 0.5))
    plic = vof.reconstruct(alpha)
    metrics = lower_static_unstructured_stage_metrics(discretization)
    apertures = vof.face_phase_apertures(alpha, plic, metrics)
    total = jnp.linspace(-0.2, 0.2, apertures.face_ids.size)

    phase0, phase1, alpha_flux = eqx.filter_jit(vof.phase_swept_flux)(total, apertures)
    np.testing.assert_allclose(phase0 + phase1, total, atol=2e-12)
    np.testing.assert_allclose(alpha_flux, phase0, atol=2e-12)


def test_full_empty_limits_and_invalid_route_or_identity_are_rejected():
    discretization = _grid()
    vof = _vof(discretization)
    metrics = lower_static_unstructured_stage_metrics(discretization)
    for alpha_value in (0.0, 1.0):
        alpha = jnp.full((discretization.cell_count,), alpha_value)
        plic = vof.reconstruct(alpha)
        apertures = vof.face_phase_apertures(alpha, plic, metrics)
        expected = 0.0 if alpha_value == 0.0 else 1.0
        np.testing.assert_allclose(
            apertures.owner_phase_apertures[:, 0], expected, atol=2e-12
        )

    stale = jnp.full((discretization.cell_count,), 0.5).at[0].set(0.25)
    with pytest.raises(ValueError, match="stale"):
        vof.face_phase_apertures(
            stale,
            vof.reconstruct(jnp.full((discretization.cell_count,), 0.5)),
            metrics,
        )
    inactive = eqx.tree_at(
        lambda value: value.face_blocks[0].layout.active_mask,
        metrics,
        jnp.zeros_like(metrics.face_blocks[0].layout.active_mask),
    )
    with pytest.raises(ValueError, match="inactive"):
        vof.face_phase_apertures(
            jnp.full((discretization.cell_count,), 0.5),
            vof.reconstruct(jnp.full((discretization.cell_count,), 0.5)),
            inactive,
        )


def test_embedded_open_and_cut_routes_use_exact_segment_geometry():
    discretization = _grid()
    embedded = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.45,
        field_id="plic-open-cut",
    ).prepare()
    vof = _vof(discretization)
    alpha = embedded.volume_fraction
    plic = vof.reconstruct(alpha)
    boundary_set = UnstructuredEmbeddedBoundarySet(
        {0: phx.discretization.SlipWallBoundary()}
    )
    metrics = lower_embedded_stage_metrics(
        discretization,
        embedded,
        boundary_set,
        discretization.topology_id,
        0,
        0,
    )
    apertures = vof.face_phase_apertures(alpha, plic, metrics)

    assert jnp.any(
        jnp.asarray([block.layout.block_kind == "cut" for block in metrics.face_blocks])
    )
    np.testing.assert_allclose(
        apertures.owner_phase_apertures[:, 0] + apertures.owner_phase_apertures[:, 1],
        1.0,
        atol=2e-12,
    )


def test_stage_reconstruction_is_filter_jittable_jacfwd_safe_and_alpha_bound():
    discretization = _grid()
    vof = _vof(discretization)
    first_alpha = jnp.full((discretization.cell_count,), 0.25)
    second_alpha = jnp.full((discretization.cell_count,), 0.75)

    first = eqx.filter_jit(vof.reconstruct_stage)(first_alpha)
    second = eqx.filter_jit(vof.reconstruct_stage)(second_alpha)
    offset_jacobian = jax.jacfwd(lambda alpha: vof.reconstruct_stage(alpha).offsets)(
        jnp.full((discretization.cell_count,), 0.5)
    )

    assert isinstance(first, JAXPLICStageReconstruction)
    assert jnp.all(jnp.isfinite(offset_jacobian))
    assert not jnp.array_equal(first.apertures_id, second.apertures_id)
    assert not jnp.allclose(
        first.owner_phase_apertures,
        second.owner_phase_apertures,
    )
    np.testing.assert_array_equal(first.volume_fraction_id, first_alpha)
    np.testing.assert_array_equal(second.volume_fraction_id, second_alpha)


def test_stage_planar_reconstruction_matches_cell_volume_and_exact_half_planes():
    discretization = _grid()
    vof = _vof(discretization)
    alpha = jnp.full((discretization.cell_count,), 0.5)

    stage = vof.reconstruct_stage(alpha)

    np.testing.assert_allclose(
        stage.reconstructed_volume_fraction,
        alpha,
        rtol=0.0,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        stage.offsets,
        discretization.cell_centers[:, 0],
        rtol=0.0,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        stage.owner_phase_apertures[:, 0] + stage.owner_phase_apertures[:, 1],
        1.0,
        rtol=0.0,
        atol=2e-7,
    )
    assert jnp.all(stage.interface_active)
    assert jnp.all(stage.interface_evidence)
    assert jnp.all(stage.interface_status == int(PLICInterfaceStatus.INTERFACE))


@pytest.mark.parametrize("factory", (_grid, _triangular_grid))
def test_stage_reconstruction_supports_fixed_capacity_triangles_and_quads(factory):
    discretization = factory()
    vof = _vof(discretization)
    alpha = jnp.linspace(0.15, 0.85, discretization.cell_count)

    stage = eqx.filter_jit(vof.reconstruct_stage)(alpha)

    np.testing.assert_allclose(
        stage.reconstructed_volume_fraction,
        alpha,
        rtol=0.0,
        atol=3e-5,
    )
    assert stage.interface_endpoints.shape == (
        discretization.cell_count,
        2,
        2,
    )
    assert jnp.all(stage.interface_evidence)


def test_stage_reconstruction_is_covariant_under_translation_and_rotation():
    base = _grid()
    transformed = _grid(shift=(2.5, -1.75), angle=0.63)
    alpha = 0.2 + 0.6 * base.cell_centers[:, 0]

    base_stage = _vof(base).reconstruct_stage(alpha)
    transformed_stage = _vof(transformed).reconstruct_stage(alpha)

    np.testing.assert_allclose(
        base_stage.owner_phase_apertures,
        transformed_stage.owner_phase_apertures,
        rtol=0.0,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        base_stage.receptor_phase_apertures,
        transformed_stage.receptor_phase_apertures,
        rtol=0.0,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        transformed_stage.reconstructed_volume_fraction,
        alpha,
        rtol=0.0,
        atol=3e-5,
    )


def test_stage_full_empty_limits_are_exact_for_both_face_sides():
    discretization = _grid()
    vof = _vof(discretization)

    empty = vof.reconstruct_stage(jnp.zeros((discretization.cell_count,)))
    full = vof.reconstruct_stage(jnp.ones((discretization.cell_count,)))

    np.testing.assert_array_equal(
        empty.owner_phase_apertures[:, 0],
        jnp.zeros((discretization.face_measures.size,)),
    )
    np.testing.assert_array_equal(
        empty.receptor_phase_apertures[:, 0],
        jnp.zeros((discretization.face_measures.size,)),
    )
    np.testing.assert_array_equal(
        full.owner_phase_apertures[:, 0],
        jnp.ones((discretization.face_measures.size,)),
    )
    np.testing.assert_array_equal(
        full.receptor_phase_apertures[:, 0],
        jnp.ones((discretization.face_measures.size,)),
    )
    np.testing.assert_array_equal(
        empty.reconstructed_volume_fraction,
        jnp.zeros((discretization.cell_count,)),
    )
    np.testing.assert_array_equal(
        full.reconstructed_volume_fraction,
        jnp.ones((discretization.cell_count,)),
    )
    assert not jnp.any(empty.interface_active)
    assert not jnp.any(full.interface_active)


def test_stage_donor_selection_reverses_on_internal_flux_and_conserves_total():
    discretization = _grid()
    vof = _vof(discretization)
    alpha = jnp.asarray((0.2, 0.8, 0.35, 0.65))
    stage = vof.reconstruct_stage(alpha)
    positive = jnp.full((discretization.face_measures.size,), 0.125)
    negative = -positive

    positive_donor = eqx.filter_jit(vof.donor_phase_apertures)(
        positive,
        stage,
    )
    negative_donor = eqx.filter_jit(vof.donor_phase_apertures)(
        negative,
        stage,
    )
    internal = stage.receptor_cells >= 0

    np.testing.assert_allclose(
        positive_donor,
        stage.owner_phase_apertures,
        rtol=0.0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        negative_donor[internal],
        stage.receptor_phase_apertures[internal],
        rtol=0.0,
        atol=2e-7,
    )
    phase0, phase1, alpha_flux = eqx.filter_jit(vof.phase_swept_flux)(
        negative,
        stage,
    )
    np.testing.assert_allclose(phase0 + phase1, negative, atol=2e-7)
    np.testing.assert_allclose(alpha_flux, phase0, atol=2e-7)


def test_stage_reconstruction_rejects_invalid_geometry_rank_alpha_and_stale_routes():
    discretization = _grid()
    vof = _vof(discretization)
    alpha = jnp.full((discretization.cell_count,), 0.5)
    stage = vof.reconstruct_stage(alpha)

    invalid_vertices = vof.discretization.vertices.at[0, 0].set(jnp.nan)
    invalid_geometry = eqx.tree_at(
        lambda plan: plan.discretization.vertices,
        vof,
        invalid_vertices,
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="invalid or uncertain",
    ):
        result = eqx.filter_jit(invalid_geometry.reconstruct_stage)(alpha)
        jax.block_until_ready(result.offsets)

    uncertain_rank = eqx.tree_at(
        lambda plan: plan.gradient.report.minimum_rank,
        vof,
        jnp.asarray(1, dtype=jnp.int32),
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="rank evidence",
    ):
        result = eqx.filter_jit(uncertain_rank.reconstruct_stage)(alpha)
        jax.block_until_ready(result.offsets)

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match=r"\[0, 1\]",
    ):
        result = eqx.filter_jit(vof.reconstruct_stage)(alpha.at[0].set(jnp.nan))
        jax.block_until_ready(result.offsets)

    stale_routes = eqx.tree_at(
        lambda reconstruction: reconstruction.owner_cells,
        stage,
        stage.owner_cells.at[0].set(
            (stage.owner_cells[0] + 1) % stage.volume_fraction.size
        ),
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="topology, layout",
    ):
        donor = eqx.filter_jit(vof.donor_phase_apertures)(
            jnp.ones((discretization.face_measures.size,)),
            stale_routes,
        )
        jax.block_until_ready(donor)

    translated_vof = _vof(_grid(shift=(1.0, 0.0)))
    translated_stage = translated_vof.reconstruct_stage(alpha)
    with pytest.raises(ValueError, match="stale"):
        vof.donor_phase_apertures(
            jnp.ones((discretization.face_measures.size,)),
            translated_stage,
        )


def test_stage_embedded_plic_uses_oblique_fluid_polygons_and_open_segments():
    discretization = _grid()
    vof = _vof(discretization)
    embedded = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] + points[:, 1] - 1.25,
        field_id="oblique-effective-plic",
    ).prepare()
    alpha = jnp.full((discretization.cell_count,), 0.5)

    def reconstruct(value):
        return vof.reconstruct_stage(
            value,
            effective_geometry=embedded,
            geometry_layout_id="embedded-stage-layout",
            geometry_version=7,
        )

    stage = reconstruct(alpha)
    compiled = eqx.filter_jit(reconstruct)(alpha)
    offset_jacobian = jax.jacfwd(lambda value: reconstruct(value).offsets)(alpha)

    np.testing.assert_allclose(embedded.fluid_cell_volumes[3], 7.0 / 32.0)
    np.testing.assert_allclose(
        stage.reconstructed_volume_fraction[3] * embedded.fluid_cell_volumes[3],
        7.0 / 64.0,
        rtol=0.0,
        atol=2e-7,
    )
    open_faces = np.asarray(embedded.open_face_measures) > 0.0
    closed_faces = ~open_faces
    np.testing.assert_allclose(
        np.asarray(stage.owner_phase_apertures)[open_faces].sum(axis=-1),
        1.0,
        rtol=0.0,
        atol=2e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(stage.owner_phase_apertures)[closed_faces],
        np.zeros_like(np.asarray(stage.owner_phase_apertures)[closed_faces]),
    )
    np.testing.assert_array_equal(
        np.asarray(stage.receptor_phase_apertures)[closed_faces],
        np.zeros_like(np.asarray(stage.receptor_phase_apertures)[closed_faces]),
    )
    np.testing.assert_allclose(
        np.linalg.norm(
            np.asarray(embedded.open_face_segment_endpoints)[:, 1]
            - np.asarray(embedded.open_face_segment_endpoints)[:, 0],
            axis=-1,
        ),
        embedded.open_face_measures,
        rtol=0.0,
        atol=2e-7,
    )
    inactive = ~np.asarray(embedded.active_fluid_cells)
    assert np.any(inactive)
    np.testing.assert_array_equal(
        np.asarray(stage.offsets)[inactive],
        np.zeros_like(np.asarray(stage.offsets)[inactive]),
    )
    np.testing.assert_array_equal(
        np.asarray(stage.reconstructed_volume_fraction)[inactive],
        np.zeros_like(np.asarray(stage.reconstructed_volume_fraction)[inactive]),
    )
    assert not np.any(np.asarray(stage.interface_active)[inactive])
    assert np.all(np.asarray(stage.interface_evidence)[inactive])
    assert np.all(np.isfinite(np.asarray(offset_jacobian)))
    np.testing.assert_allclose(compiled.offsets, stage.offsets)
    assert stage.effective_geometry_id == embedded.metrics_id
    assert stage.geometry_layout_id == "embedded-stage-layout"
    np.testing.assert_array_equal(stage.geometry_version, 7)


def test_stage_full_fluid_effective_geometry_matches_background_geometry():
    discretization = _grid()
    vof = _vof(discretization)
    full_fluid = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: jnp.ones((points.shape[0],)),
        field_id="full-fluid-effective-plic",
    ).prepare()
    alpha = jnp.full((discretization.cell_count,), 0.5)

    background = vof.reconstruct_stage(alpha)
    effective = vof.reconstruct_stage(alpha, effective_geometry=full_fluid)

    np.testing.assert_allclose(effective.offsets, background.offsets)
    np.testing.assert_allclose(
        effective.reconstructed_volume_fraction,
        background.reconstructed_volume_fraction,
    )
    np.testing.assert_allclose(
        effective.owner_phase_apertures,
        background.owner_phase_apertures,
    )
    np.testing.assert_allclose(
        effective.receptor_phase_apertures,
        background.receptor_phase_apertures,
    )
    np.testing.assert_array_equal(
        effective.open_face_active,
        np.ones_like(np.asarray(effective.open_face_active), dtype=bool),
    )
    assert background.effective_geometry_id == discretization.prepared_id
    assert effective.effective_geometry_id == full_fluid.metrics_id


def test_stage_reconstruction_rejects_stale_effective_embedded_geometry():
    discretization = _grid()
    vof = _vof(discretization)
    translated = _grid(shift=(1.0, 0.0))
    stale = phx.discretization.EmbeddedBoundaryPlan(
        translated,
        lambda points, args: points[:, 0] + points[:, 1] - 1.25,
        field_id="stale-effective-plic",
    ).prepare()

    with pytest.raises(ValueError, match="stale"):
        vof.reconstruct_stage(
            jnp.full((discretization.cell_count,), 0.5),
            effective_geometry=stale,
        )


def test_vof_plan_rejects_tetrahedral_geometry_before_gradient_use():
    tetrahedral = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        tetrahedra=np.asarray(((0, 1, 2, 3),)),
    ).prepare()

    with pytest.raises(ValueError, match="2-D polygons"):
        phx.discretization.UnstructuredVOFPlan(tetrahedral, None)
