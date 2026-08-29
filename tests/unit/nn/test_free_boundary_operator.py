#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _reference_samples():
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, 5),
        quadrature_weights=jnp.ones((5,)) / 5.0,
    )
    return phx.nn.operator.FunctionSamples(values=None, axes=(axis,))


def _simple_batch(values):
    coordinates = jnp.linspace(0.0, 1.0, values.size)[:, None]
    samples = phx.nn.operator.FunctionSamples(values=values, coordinates=coordinates)
    query = phx.nn.operator.FunctionSamples(values=None, coordinates=coordinates)
    return phx.nn.operator.OperatorBatch(inputs={"state": samples}, queries={"q": query})


def test_reference_map_jacobian_gcl_and_gradient_pullback():
    reference = _reference_samples()
    nodes = reference.coordinates_array()
    current = nodes
    following = 1.1 * nodes

    evidence = phx.nn.operator.reference_map_evidence(current, reference)
    constraint = phx.nn.operator.reference_map_constraint_loss(
        current,
        following,
        reference,
        1.0,
    )
    pulled = phx.nn.operator.pullback_scalar_gradient(
        jnp.full((5, 1), 2.0),
        jnp.full((5, 1, 1), 2.0),
    )

    np.testing.assert_allclose(evidence.determinant, 1.0, atol=1.0e-14)
    assert bool(evidence.orientation_preserving)
    assert bool(evidence.nonsingular)
    np.testing.assert_allclose(constraint.total, 0.0, atol=1.0e-12)
    assert bool(constraint.successful)
    np.testing.assert_allclose(pulled, 1.0, atol=1.0e-14)


def test_free_boundary_operator_spec_rejects_topology_changing_reference_map():
    with np.testing.assert_raises(ValueError):
        phx.nn.operator.FreeBoundaryOperatorSpec(
            "reference_map",
            "map",
            ("temperature",),
            "q",
            topology_changes=True,
        )


def test_solver_corrected_rollout_uses_only_accepted_improving_states():
    initial = _simple_batch(jnp.asarray((0.0, 1.0, 2.0)))

    def model(batch, *, key):
        del key
        return batch.input("state").values + 1.0

    def correct(prediction, batch, index):
        del batch, index
        return phx.nn.operator.CorrectedOperatorStep(
            values=prediction - 0.5,
            residual_before=jnp.asarray(1.0),
            residual_after=jnp.asarray(0.25),
            conservation_error=jnp.asarray(0.0),
            accepted=jnp.asarray(True),
        )

    def advance(batch, values, index):
        del batch, index
        return _simple_batch(values)

    result = phx.nn.operator.solver_corrected_operator_rollout(
        model,
        initial,
        3,
        correct,
        advance,
        key=jr.key(0),
    )

    np.testing.assert_allclose(result.corrected[-1], jnp.asarray((1.5, 2.5, 3.5)))
    np.testing.assert_allclose(result.residual_after, 0.25)
    assert bool(jnp.all(result.accepted))


def test_sph_free_surface_adapter_preserves_surface_measure_and_mask():
    geometry = phx.discretization.FreeSurfaceGeometryState(
        surface_point=jnp.asarray(((0.0, 0.0), (1.0, 0.0))),
        normal=jnp.asarray(((0.0, 1.0), (0.0, 1.0))),
        curvature=jnp.asarray((0.0, 0.1)),
        signed_distance=jnp.asarray((0.0, 0.0)),
        kernel_volume_fraction=jnp.asarray((0.5, 0.25)),
        fit_residual=jnp.asarray((0.0, 0.2)),
        confidence=jnp.asarray((1.0, 0.5)),
        successful=jnp.asarray((True, False)),
    )
    batch = phx.nn.operator.operator_batch_from_sph_free_surface(
        geometry,
        jnp.asarray(((0.0, -0.1), (1.0, -0.1))),
        jnp.asarray((2.0, 4.0)),
        {"density": jnp.asarray((1.0, 1.2))},
    )

    surface = batch.input("free_surface")
    np.testing.assert_array_equal(surface.mask, jnp.asarray((True, False)))
    np.testing.assert_allclose(surface.quadrature_weights, jnp.asarray((1.0, 1.0)))
    assert surface.values.shape == (2, 5)


def test_vof_adapter_preserves_plic_interface_branch():
    plic = phx.discretization.JAXPLICStageReconstruction(
        volume_fraction=jnp.asarray((0.25, 0.75)),
        normals=jnp.asarray(((1.0, 0.0), (1.0, 0.0))),
        offsets=jnp.zeros((2,)),
        reconstructed_volume_fraction=jnp.asarray((0.25, 0.75)),
        volume_residual=jnp.zeros((2,)),
        interface_endpoints=jnp.zeros((2, 2, 2)),
        interface_centers=jnp.asarray(((0.25, 0.5), (0.75, 0.5))),
        interface_measures=jnp.asarray((0.5, 0.5)),
        interface_active=jnp.asarray((True, False)),
        interface_status=jnp.zeros((2,), dtype=jnp.int32),
        interface_evidence=jnp.asarray((1.0, 0.0)),
        face_ids=jnp.asarray((0,), dtype=jnp.int32),
        owner_cells=jnp.asarray((0,), dtype=jnp.int32),
        receptor_cells=jnp.asarray((1,), dtype=jnp.int32),
        open_face_active=jnp.asarray((True,)),
        owner_phase_apertures=jnp.ones((1, 2)),
        receptor_phase_apertures=jnp.ones((1, 2)),
        owner_phase_centroids=jnp.zeros((1, 2, 2)),
        receptor_phase_centroids=jnp.zeros((1, 2, 2)),
        aperture_ids=jnp.asarray((1,), dtype=jnp.int32),
        geometry_version=jnp.asarray(0, dtype=jnp.int32),
        plan_id="vof",
        geometry_id="mesh",
        prepared_id="prepared",
        topology_id="topology",
        physical_layout_id="faces",
        effective_geometry_id="effective",
        geometry_layout_id="layout",
    )
    batch = phx.nn.operator.operator_batch_from_vof(
        plic,
        jnp.asarray(((0.25, 0.5), (0.75, 0.5))),
        jnp.asarray((0.5, 0.5)),
        {"pressure": jnp.asarray((1.0, 2.0))},
    )

    np.testing.assert_allclose(batch.input("volume_fraction").values, (0.25, 0.75))
    np.testing.assert_array_equal(batch.input("interface").mask, (True, False))
    assert batch.input("interface").values.shape == (2, 4)
