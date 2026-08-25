#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _mesh(system):
    vertices = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0))
    )
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=np.int32),
        vertex_global_ids=np.arange(10, 16),
        cell_global_ids=np.asarray((101, 103)),
        component_names=system.component_names,
    )


def _compiled(
    *,
    hole_mask=None,
    epoch_id="overset-epoch",
    donor_cell=0,
    receptor_cell=1,
):
    system = phx.equations.EulerSystem(2)
    plan = _mesh(system)
    discretization = plan.prepare()
    internal_faces = np.where(np.asarray(discretization.neighbour_cells) >= 0)[0]
    face_id = int(internal_faces[0])
    face_points = np.asarray(discretization.face_quadrature_points)[face_id : face_id + 1]
    unit_normal = np.asarray(discretization.area_vectors)[face_id] / float(
        np.asarray(discretization.face_measures)[face_id]
    )
    if int(np.asarray(discretization.owner_cells)[face_id]) != receptor_cell:
        unit_normal = -unit_normal
    face_normals = np.broadcast_to(unit_normal, face_points.shape)
    face_measures = np.asarray(discretization.face_quadrature_weights)[
        face_id : face_id + 1
    ]
    overset = phx.discretization.UnstructuredOversetPlan(
        discretization,
        discretization,
        np.asarray((receptor_cell,), dtype=np.int32),
        np.asarray((0, 1), dtype=np.int32),
        np.asarray((donor_cell,), dtype=np.int32),
        np.asarray((1.0,)),
        hole_mask=hole_mask,
        epoch_id=epoch_id,
        interpolation_policy="conservative",
        receptor_face_ids=np.asarray((face_id,), dtype=np.int32),
        receptor_face_points=face_points,
        receptor_face_normals=face_normals,
        receptor_face_measures=face_measures,
        receptor_face_cells=np.asarray((receptor_cell,), dtype=np.int32),
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(overset=overset)
    problem = phx.equations.ConservationProblemIR(
        "overset-runtime", "state", system, boundaries
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method, coupling=coupling
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(maximum_retries=2),
    )
    return system, discretization, overset, dynamics, runtime


def _nonuniform_state(system, discretization, *, donor_cell=0, receptor_cell=1):
    assert donor_cell != receptor_cell
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 1.0)),
        discretization.state_shape,
    )
    primitive = primitive.at[receptor_cell, 0].set(2.0)
    return system.primitive_to_conserved(primitive)


def test_donor_traces_and_accepted_correction_are_jit_safe_and_conservative():
    system, discretization, overset, dynamics, runtime = _compiled()
    constant = jnp.ones((discretization.cell_count, system.component_count))
    np.testing.assert_allclose(eqx.filter_jit(overset.interpolate)(constant), 1.0)
    linear = jnp.asarray(
        ((1.0,) * system.component_count, (3.0,) * system.component_count)
    )
    np.testing.assert_allclose(overset.interpolate(linear), 1.0)

    initial = runtime.initialize_state(
        _nonuniform_state(system, discretization),
        0.0,
        1e-3,
    )
    result = runtime.advance(initial)
    assert bool(np.asarray(result.accepted))
    block = result.accepted_flux_integrals.blocks[-1]
    assert block.block_kind == "overset-correction"
    np.testing.assert_array_equal(np.asarray(block.owner_cells), np.asarray((0,)))
    np.testing.assert_array_equal(np.asarray(block.neighbour_cells), np.asarray((1,)))
    scattered = np.zeros(
        (discretization.cell_count, system.component_count),
        dtype=np.asarray(block.flux_integral).dtype,
    )
    np.add.at(scattered, np.asarray(block.owner_cells), -np.asarray(block.flux_integral))
    np.add.at(
        scattered, np.asarray(block.neighbour_cells), np.asarray(block.flux_integral)
    )
    assert scattered[0, 0] > 0.0
    assert scattered[1, 0] < 0.0
    np.testing.assert_allclose(scattered.sum(axis=0), 0.0, atol=1e-12)


def test_canonical_route_sign_reverses_receptor_normal_and_preserves_cfl():
    route_rates = []
    route_speeds = []
    route_measures = []
    for donor_cell, receptor_cell in ((0, 1), (1, 0)):
        system, discretization, _, dynamics, _ = _compiled(
            donor_cell=donor_cell,
            receptor_cell=receptor_cell,
        )
        metrics = phx.discretization.lower_static_unstructured_stage_metrics(
            discretization
        )
        state = _nonuniform_state(
            system,
            discretization,
            donor_cell=donor_cell,
            receptor_cell=receptor_cell,
        )

        def correction_flux(value):
            block, _, _ = dynamics._overset_correction(value, metrics, None)
            assert block is not None
            return block.flux_rate

        flux_rate = eqx.filter_jit(correction_flux)(state)
        gradient = jax.grad(lambda value: jnp.sum(correction_flux(value) ** 2))(state)
        block, speed, measures = dynamics._overset_correction(state, metrics, None)
        assert block is not None
        np.testing.assert_allclose(flux_rate, block.flux_rate)
        assert bool(np.asarray(jnp.all(jnp.isfinite(gradient))))
        assert bool(np.asarray(jnp.any(jnp.abs(gradient) > 0.0)))

        scattered = np.zeros(
            discretization.state_shape,
            dtype=np.asarray(flux_rate).dtype,
        )
        np.add.at(scattered, np.asarray(block.owner_cells), -np.asarray(flux_rate))
        np.add.at(
            scattered,
            np.asarray(block.neighbour_cells),
            np.asarray(flux_rate),
        )
        assert scattered[donor_cell, 0] > 0.0
        assert scattered[receptor_cell, 0] < 0.0
        np.testing.assert_allclose(scattered.sum(axis=0), 0.0, atol=1e-12)
        route_rates.append(np.asarray(flux_rate))
        route_speeds.append(np.asarray(speed))
        route_measures.append(np.asarray(measures))

    np.testing.assert_allclose(
        route_rates[1],
        route_rates[0] * np.asarray((1.0, -1.0, 1.0, 1.0)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(route_speeds[1], route_speeds[0], rtol=1e-12)
    np.testing.assert_allclose(route_measures[1], route_measures[0], rtol=1e-12)


def test_holes_cannot_be_donors_and_fail_closed():
    with pytest.raises(ValueError, match="ineligible|hole"):
        _compiled(hole_mask=np.asarray((True, False), dtype=bool))


def test_overset_map_epoch_and_geometry_are_compiler_identities():
    system, discretization, overset, dynamics, runtime = _compiled(epoch_id="epoch-a")
    assert dynamics.coupling.overset_epoch_id == "epoch-a"
    assert dynamics.overset_mapping_id == overset.identity
    assert dynamics.overset_policy_id
    assert runtime.runtime_id
    _, _, stale_map, _, _ = _compiled(epoch_id="stale-epoch")
    prepared = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        overset=stale_map
    ).prepare(discretization)
    assert prepared.overset_epoch_id == "stale-epoch"
