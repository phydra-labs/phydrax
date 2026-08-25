#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _overset_meshes():
    donor_vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    donor = phx.discretization.UnstructuredFiniteVolumePlan(
        donor_vertices,
        quadrilaterals=np.asarray(((0, 1, 4, 3), (1, 2, 5, 4))),
        cell_global_ids=np.asarray((101, 103)),
    ).prepare()
    receptor_vertices = np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)))
    receptor = phx.discretization.UnstructuredFiniteVolumePlan(
        receptor_vertices,
        quadrilaterals=np.asarray(((0, 1, 2, 3),)),
        cell_global_ids=np.asarray((201,)),
    ).prepare()
    return donor, receptor


def test_overset_overlap_is_constant_exact_and_conservative():
    donor, receptor = _overset_meshes()
    overset = phx.discretization.UnstructuredOversetPlan(
        donor,
        receptor,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((1.0, 1.0)),
    )
    donor_values = jnp.asarray(((1.0, 2.0), (3.0, 4.0)))
    interpolated = eqx.filter_jit(overset.interpolate)(donor_values)

    np.testing.assert_allclose(interpolated, ((2.0, 3.0),))
    np.testing.assert_allclose(overset.interpolate(jnp.ones((2,))), 1.0)
    np.testing.assert_allclose(
        overset.conservation_defect(donor_values, interpolated), 0.0, atol=1e-14
    )
    receptor_values = jnp.zeros((1, 2))
    np.testing.assert_allclose(overset.apply(receptor_values, donor_values), interpolated)
    assert overset.report.maximum_receptor_coverage_defect < 1e-12
    assert overset.report.maximum_donor_covered_fraction <= 1.0 + 1e-12

    with pytest.raises(ValueError, match="cover"):
        phx.discretization.UnstructuredOversetPlan(
            donor,
            receptor,
            np.asarray((0,), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((0.5, 0.5)),
        )


def test_periodic_sliding_overlap_rebuilds_and_closes_flux_budget():
    plan = phx.discretization.PeriodicSlidingInterfacePlan(
        np.asarray((0.0, 0.4, 1.0)),
        np.asarray((0.0, 0.25, 0.75, 1.0)),
        1.0,
        interface_id="rotor-stator-seam",
    )
    coupling = plan.coupling(0.2)
    equivalent = plan.coupling(1.2)
    changed = plan.coupling(0.3)

    assert coupling.coupling_id == equivalent.coupling_id
    assert coupling.coupling_id != changed.coupling_id
    constant = eqx.filter_jit(coupling.interpolate_left_to_right)(jnp.asarray((2.0, 2.0)))
    np.testing.assert_allclose(constant, 2.0)
    left_flux_density = jnp.asarray(((1.0, -0.5), (3.0, 2.0)))
    right_integrated = coupling.right_integrated_flux(left_flux_density)
    np.testing.assert_allclose(
        coupling.flux_conservation_defect(left_flux_density, right_integrated),
        0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(jnp.sum(coupling.overlap_measures), 1.0, atol=2e-14)


def test_sliding_interface_rejects_invalid_partitions():
    with pytest.raises(ValueError, match="partition"):
        phx.discretization.PeriodicSlidingInterfacePlan(
            np.asarray((0.0, 0.7, 0.6, 1.0)),
            np.asarray((0.0, 1.0)),
            1.0,
            interface_id="invalid",
        )


def test_overset_masks_exclude_holes_and_require_active_fringe_endpoints():
    donor, receptor = _overset_meshes()
    common = dict(
        donor_active_mask=np.asarray((True, True)),
        donor_hole_mask=np.asarray((False, False)),
        receptor_active_mask=np.asarray((True,)),
        receptor_fringe_mask=np.asarray((True,)),
    )
    with pytest.raises(ValueError, match="hole"):
        phx.discretization.UnstructuredOversetPlan(
            donor,
            receptor,
            np.asarray((0,), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1.0, 1.0)),
            donor_hole_mask=np.asarray((True, False)),
            **{key: value for key, value in common.items() if key != "donor_hole_mask"},
        )
    with pytest.raises(ValueError, match="hole"):
        phx.discretization.UnstructuredOversetPlan(
            donor,
            receptor,
            np.asarray((0,), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1.0, 1.0)),
            receptor_hole_mask=np.asarray((True,)),
            **{
                key: value
                for key, value in common.items()
                if key != "receptor_active_mask"
            },
        )
    with pytest.raises(ValueError, match="inactive"):
        phx.discretization.UnstructuredOversetPlan(
            donor,
            receptor,
            np.asarray((0,), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1.0, 1.0)),
            receptor_active_mask=np.asarray((False,)),
        )
    with pytest.raises(ValueError, match="fringe"):
        phx.discretization.UnstructuredOversetPlan(
            donor,
            receptor,
            np.asarray((0,), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1.0, 1.0)),
            receptor_fringe_mask=np.asarray((False,)),
        )


def test_overset_donor_global_ids_permit_permuted_route_indices():
    donor, receptor = _overset_meshes()
    overset = phx.discretization.UnstructuredOversetPlan(
        donor,
        receptor,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((103, 101), dtype=np.int64),
        np.asarray((1.0, 1.0)),
        donor_indices_are_global_ids=True,
        epoch_id="mesh-7",
        tolerance_id="tol-a",
    )
    assert tuple(np.asarray(overset.donor_global_ids)) == (101, 103)
    assert overset.coverage_status == "complete"
    assert overset.epoch_id == "mesh-7"
    assert overset.tolerance_id == "tol-a"
    np.testing.assert_allclose(overset.interpolate(jnp.asarray((1.0, 3.0))), 2.0)


def test_overset_rejects_donor_overcoverage_and_bad_union_certificate():
    donor, receptor = _overset_meshes()
    args = (
        donor,
        receptor,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((0, 0), dtype=np.int32),
        np.asarray((1.0, 1.0)),
    )
    with pytest.raises(ValueError, match="double"):
        phx.discretization.UnstructuredOversetPlan(*args)
    with pytest.raises(ValueError, match="union-volume"):
        phx.discretization.UnstructuredOversetPlan(
            donor,
            receptor,
            np.asarray((0,), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1.0, 1.0)),
            union_volume_certificate=np.asarray((0.5, 1.0)),
        )


def test_overset_union_certificate_identity_and_incomplete_map():
    donor, receptor = _overset_meshes()
    kwargs = dict(
        union_volume_certificate=np.asarray((1.0, 1.0)),
        epoch_id="epoch-a",
        tolerance_id="tol-a",
    )
    overset = phx.discretization.UnstructuredOversetPlan(
        donor,
        receptor,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((1.0, 1.0)),
        **kwargs,
    )
    changed_tolerance = phx.discretization.UnstructuredOversetPlan(
        donor,
        receptor,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((1.0, 1.0)),
        tolerance=1e-9,
        epoch_id="epoch-a",
        tolerance_id="tol-b",
    )
    changed_epoch = phx.discretization.UnstructuredOversetPlan(
        donor,
        receptor,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((1.0, 1.0)),
        **{**kwargs, "epoch_id": "epoch-b"},
    )
    np.testing.assert_allclose(overset.union_volume_certificate, (1.0, 1.0))
    assert overset.report.union_volume_measure == pytest.approx(2.0)
    assert overset.identity != changed_tolerance.identity
    assert overset.identity != changed_epoch.identity
    with pytest.raises(ValueError, match="coverage"):
        phx.discretization.UnstructuredOversetPlan(
            donor,
            receptor,
            np.asarray((0,), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((0.5, 0.5)),
        )


def test_overset_nonconservative_policy_rejected_and_bounded_is_explicit():
    donor, receptor = _overset_meshes()
    common = (
        donor,
        receptor,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((1.0, 1.0)),
    )
    with pytest.raises(ValueError, match="nonconservative"):
        phx.discretization.UnstructuredOversetPlan(*common, policy="nearest")
    bounded = phx.discretization.UnstructuredOversetPlan(
        *common, policy="conservative_bounded"
    )
    assert bounded.bounded_interpolation


def test_overset_rejects_stale_geometry_epoch():
    donor, receptor = _overset_meshes()
    overset = phx.discretization.UnstructuredOversetPlan(
        donor,
        receptor,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((1.0, 1.0)),
        epoch_id="epoch-a",
    )
    stale = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(
            (
                (0.125, 0.0),
                (1.125, 0.0),
                (2.125, 0.0),
                (0.125, 1.0),
                (1.125, 1.0),
                (2.125, 1.0),
            )
        ),
        quadrilaterals=np.asarray(((0, 1, 4, 3), (1, 2, 5, 4))),
        cell_global_ids=np.asarray((101, 103)),
    ).prepare()
    with pytest.raises(ValueError, match="stale"):
        overset.validate_geometry(stale, receptor)


def test_overset_face_artifact_is_complete_and_identity_bound():
    donor, receptor = _overset_meshes()
    face_ids = np.asarray((0, 1), dtype=np.int32)
    points = np.asarray(receptor.face_quadrature_points)[face_ids]
    unit_normals = (
        np.asarray(receptor.area_vectors)[face_ids]
        / np.asarray(receptor.face_measures)[face_ids, None]
    )
    normals = np.broadcast_to(unit_normals[:, None, :], points.shape)
    measures = np.asarray(receptor.face_quadrature_weights)[face_ids]
    cells = np.asarray((0, 0), dtype=np.int32)
    overset = phx.discretization.UnstructuredOversetPlan(
        donor,
        receptor,
        np.asarray((0,), dtype=np.int32),
        np.asarray((0, 2), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((1.0, 1.0)),
        receptor_face_ids=face_ids,
        receptor_face_points=points,
        receptor_face_normals=normals,
        receptor_face_measures=measures,
        receptor_face_cells=cells,
    )
    assert overset.face_artifact_id
    np.testing.assert_allclose(overset.receptor_face_points, points)
    assert overset.receptor_face_cells.shape == (2,)
    with pytest.raises(ValueError, match="require"):
        phx.discretization.UnstructuredOversetPlan(
            donor,
            receptor,
            np.asarray((0,), dtype=np.int32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1.0, 1.0)),
            receptor_face_ids=face_ids,
            receptor_face_points=points,
        )
