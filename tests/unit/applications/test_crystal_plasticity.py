#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cpfem = phx.applications.crystal_plasticity


def _slip_xy():
    return cpfem.CrystalSlipSystem(
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.asarray((0.0, 1.0, 0.0)),
    )


def _slip_yz():
    return cpfem.CrystalSlipSystem(
        jnp.asarray((0.0, 1.0, 0.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
    )


def _model(*systems, maximum_slip_increment=0.2):
    return cpfem.CrystalPlasticityModel(
        systems or (_slip_xy(),),
        cpfem.CrystalPlasticityParameters(
            8.0,
            20.0,
            0.1,
            1.0,
            1.5,
            1.0,
            maximum_slip_increment=maximum_slip_increment,
        ),
    )


def _rotation_z():
    return jnp.asarray(
        (
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )


def _simple_shear(amount=0.4):
    return jnp.eye(3).at[0, 1].set(amount)


def _two_block_discretization():
    points = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 0.0, 1.0),
        )
    )
    blocks = (
        phx.discretization.CellBlock(
            "phase-a",
            "tetrahedron",
            jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
            global_ids=jnp.asarray((10,)),
        ),
        phx.discretization.CellBlock(
            "phase-b",
            "tetrahedron",
            jnp.asarray(((4, 5, 6, 7),), dtype=jnp.int32),
            global_ids=jnp.asarray((20,)),
        ),
    )
    mesh = phx.discretization.CellMesh(points, blocks)
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("tetrahedron", 1),
            component_shape=(3,),
        ),
    ).prepare()


def _one_block_two_cell_discretization():
    points = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 0.0, 1.0),
        )
    )
    block = phx.discretization.CellBlock(
        "phase",
        "tetrahedron",
        jnp.asarray(((0, 1, 2, 3), (4, 5, 6, 7)), dtype=jnp.int32),
        global_ids=jnp.asarray((10, 20)),
    )
    mesh = phx.discretization.CellMesh(points, (block,))
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element("tetrahedron", 1),
            component_shape=(3,),
        ),
    ).prepare()


def _two_phase_route(discretization, *, bound=0.2):
    one_slip = _model(_slip_xy(), maximum_slip_increment=bound)
    two_slip = _model(_slip_xy(), _slip_yz(), maximum_slip_increment=bound)
    return cpfem.CrystalPlasticityRoute(
        discretization,
        "u",
        (
            ("phase-a", one_slip, jnp.eye(3)),
            ("phase-b", two_slip, _rotation_z()),
        ),
    )


def test_state_initialization_and_packing_are_exact_and_slip_local():
    model = _model(_slip_xy(), _slip_yz())
    state = model.initial_state()
    packed = state.pack()
    unpacked = cpfem.CrystalPlasticityState.unpack(packed, model.slip_count)

    assert packed.shape == (12,)
    np.testing.assert_allclose(unpacked.plastic_deformation, jnp.eye(3))
    np.testing.assert_allclose(unpacked.strengths, jnp.ones((2,)))
    assert unpacked.accumulated_slip == pytest.approx(0.0)
    assert model.model_id == _model(_slip_xy(), _slip_yz()).model_id
    assert model.model_id != _model(_slip_xy()).model_id
    with pytest.raises(ValueError, match="invalid shape"):
        cpfem.CrystalPlasticityState.unpack(packed[:-1], model.slip_count)


def test_slip_systems_and_orientations_require_finite_proper_geometry():
    with pytest.raises(ValueError, match="nonzero"):
        cpfem.CrystalSlipSystem(jnp.zeros(3), jnp.asarray((0.0, 1.0, 0.0)))
    with pytest.raises(ValueError, match="orthogonal"):
        cpfem.CrystalSlipSystem(
            jnp.asarray((1.0, 0.0, 0.0)), jnp.asarray((1.0, 1.0, 0.0))
        )
    slip = _slip_xy()
    with pytest.raises(ValueError, match="distinct"):
        _model(
            slip,
            cpfem.CrystalSlipSystem(-slip.direction, slip.normal),
        )

    reflection = jnp.diag(jnp.asarray((1.0, 1.0, -1.0)))
    model = _model(slip)
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match=r"SO\(3\)"):
        invalid = model.update(jnp.eye(3), model.initial_state(), reflection, 0.1)
        jax.block_until_ready(invalid.first_piola)

    rotated = model.update(jnp.eye(3), model.initial_state(), _rotation_z(), 0.1)
    assert bool(rotated.accepted)


def test_elastic_dilatation_preserves_plastic_volume_and_energy_stress_relation():
    model = _model()
    state = model.initial_state()
    deformation = 1.04 * jnp.eye(3)
    update = model.update(deformation, state, jnp.eye(3), 0.1)

    assert bool(update.accepted)
    np.testing.assert_allclose(update.slip_increment, 0.0, atol=1.0e-7)
    assert update.plastic_determinant == pytest.approx(1.0, abs=2.0e-6)
    assert update.elastic_determinant == pytest.approx(1.04**3, rel=2.0e-5)
    assert update.elastic_energy > 0.0

    energy_gradient = jax.grad(lambda value: model.free_energy(value, state))(deformation)
    np.testing.assert_allclose(
        energy_gradient,
        model.first_piola(deformation, state),
        rtol=2.0e-4,
        atol=2.0e-5,
    )


def test_active_slip_has_hardening_storage_and_nonnegative_incremental_dissipation():
    model = _model()
    state = model.initial_state()
    update = model.update(_simple_shear(), state, jnp.eye(3), 0.1)

    assert bool(update.converged)
    assert bool(update.admissible)
    assert bool(update.thermodynamic_admissible)
    assert update.slip_increment[0] > 0.0
    assert update.state.accumulated_slip > state.accumulated_slip
    assert jnp.all(update.state.strengths > state.strengths)
    assert update.hardening_energy > 0.0
    assert update.plastic_work >= update.hardening_energy
    tolerance = 512.0 * jnp.finfo(update.incremental_dissipation.dtype).eps
    assert update.incremental_dissipation >= -tolerance
    assert update.plastic_determinant == pytest.approx(1.0, abs=2.0e-5)
    assert update.elastic_determinant > 0.0


def test_implicit_root_has_one_finite_jvp_consistent_with_directional_difference():
    model = _model()
    state = model.initial_state()
    deformation = _simple_shear(0.3)
    orientation = jnp.eye(3)
    direction = jnp.asarray(((0.02, -0.03, 0.01), (0.01, 0.0, -0.02), (0.0, 0.01, 0.02)))

    stress, tangent_action = jax.jvp(
        lambda value: model.update(value, state, orientation, 0.1).first_piola,
        (deformation,),
        (direction,),
    )
    epsilon = 1.0e-3
    finite_difference = (
        model.update(
            deformation + epsilon * direction, state, orientation, 0.1
        ).first_piola
        - model.update(
            deformation - epsilon * direction, state, orientation, 0.1
        ).first_piola
    ) / (2.0 * epsilon)

    assert jnp.all(jnp.isfinite(stress))
    assert jnp.all(jnp.isfinite(tangent_action))
    np.testing.assert_allclose(
        tangent_action, finite_difference, rtol=8.0e-3, atol=3.0e-3
    )


def test_update_vmaps_and_jits_dynamic_rotation_and_step_inputs():
    model = _model()
    state = model.initial_state()
    rotation = _rotation_z()
    deformation = _simple_shear(0.3)
    deformations = jnp.stack((deformation, rotation @ deformation @ rotation.T))
    orientations = jnp.stack((jnp.eye(3), rotation))
    step_sizes = jnp.asarray((0.1, 0.1))

    @jax.jit
    def batched_update(values, frames, steps):
        return jax.vmap(
            lambda value, frame, step: model.update(value, state, frame, step).first_piola
        )(values, frames, steps)

    stresses = batched_update(deformations, orientations, step_sizes)
    np.testing.assert_allclose(
        stresses[1],
        rotation @ stresses[0] @ rotation.T,
        rtol=3.0e-4,
        atol=3.0e-5,
    )


def test_spatial_objectivity_and_crystal_frame_covariance_hold_for_two_orientations():
    model = _model()
    state = model.initial_state()
    deformation = _simple_shear(0.35)
    rotation = _rotation_z()
    base = model.update(deformation, state, jnp.eye(3), 0.1)

    superposed = model.update(rotation @ deformation, state, jnp.eye(3), 0.1)
    np.testing.assert_allclose(
        superposed.first_piola, rotation @ base.first_piola, rtol=3.0e-4, atol=3.0e-5
    )
    np.testing.assert_allclose(
        superposed.state.plastic_deformation,
        base.state.plastic_deformation,
        rtol=3.0e-4,
        atol=3.0e-5,
    )

    transformed = model.update(
        rotation @ deformation @ rotation.T,
        state,
        rotation,
        0.1,
    )
    np.testing.assert_allclose(
        transformed.slip_increment, base.slip_increment, rtol=3.0e-4, atol=3.0e-5
    )
    np.testing.assert_allclose(
        transformed.state.plastic_deformation,
        rotation @ base.state.plastic_deformation @ rotation.T,
        rtol=3.0e-4,
        atol=3.0e-5,
    )
    np.testing.assert_allclose(
        transformed.first_piola,
        rotation @ base.first_piola @ rotation.T,
        rtol=3.0e-4,
        atol=3.0e-5,
    )


def test_multiblock_route_keeps_ragged_states_and_shared_routing():
    discretization = _two_block_discretization()
    route = _two_phase_route(discretization)
    transaction = route.initialize()

    assert route.state_shapes[0][-1] == 11
    assert route.state_shapes[1][-1] == 12
    assert transaction.state(route.site_ids[0]).committed.shape == route.state_shapes[0]
    assert transaction.state(route.site_ids[1]).committed.shape == route.state_shapes[1]
    assert set(np.asarray(route.domains[0].entity_indices)) == {0}
    assert route.crystal_to_sample[0].shape == route.state_shapes[0][:2] + (
        3,
        3,
    )
    assert set(np.asarray(route.domains[1].entity_indices)) == {1}

    form = cpfem.cpfem_equilibrium_form(discretization, "u", route, transaction, 0.1)
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    displacement = jnp.zeros((8, 3))
    residual, auxiliary = compiled.residual_with_auxiliary(displacement)

    np.testing.assert_allclose(residual, 0.0, atol=2.0e-6)
    assert bool(auxiliary.valid)
    assert isinstance(auxiliary.trial_state, phx.equations.MaterialTransaction)
    assert auxiliary.trial_state.layout_id == transaction.layout_id
    route.validate(auxiliary.trial_state)


def test_form_residual_and_auxiliary_use_same_nonsymmetric_gradient():
    discretization = _two_block_discretization()
    route = _two_phase_route(discretization)
    transaction = route.initialize()
    form = cpfem.cpfem_equilibrium_form(discretization, "u", route, transaction, 0.1)
    deformation = jnp.asarray(
        ((1.02, 0.27, 0.03), (-0.08, 0.96, 0.05), (0.01, -0.02, 1.01))
    )
    quadrature_count = route.state_shapes[0][1]
    gradients = (jnp.broadcast_to(deformation - jnp.eye(3), (1, quadrature_count, 3, 3)),)
    weights = jnp.full((1, quadrature_count), 1.0 / quadrature_count)
    test_gradients = jnp.zeros((1, quadrature_count, 1, 3))
    test_gradients = test_gradients.at[..., 1].set(1.0)
    local_residual = form.actions[0].kernel(
        (jnp.zeros((1, quadrature_count, 3)),),
        gradients,
        jnp.zeros((1, quadrature_count, 3)),
        weights,
        jnp.zeros((quadrature_count, 1)),
        test_gradients,
        None,
    )
    committed = cpfem.CrystalPlasticityState.unpack(
        transaction.state(route.site_ids[0]).committed[0, 0],
        route.models[0].slip_count,
    )
    expected = route.models[0].update(
        deformation,
        committed,
        route.crystal_to_sample[0][0, 0],
        0.1,
    )
    np.testing.assert_allclose(
        local_residual[0, 0], expected.first_piola[:, 1], rtol=2.0e-5, atol=2.0e-6
    )

    displacement = (
        discretization.dof_maps[0].dof_coordinates @ (deformation - jnp.eye(3)).T
    )
    auxiliary = form.auxiliary_evaluator(displacement, None)
    trial = auxiliary.trial_state.state(route.site_ids[0]).trial
    np.testing.assert_allclose(
        trial,
        jnp.broadcast_to(expected.state.pack(), trial.shape),
        rtol=2.0e-5,
        atol=2.0e-6,
    )


def test_constrained_cpfem_auxiliary_uses_expanded_equilibrium_field():
    discretization = _two_block_discretization()
    route = _two_phase_route(discretization)
    transaction = route.initialize()
    form = cpfem.cpfem_equilibrium_form(discretization, "u", route, transaction, 0.1)
    boundary_mask = jnp.asarray((True, False, False, False, True, False, False, False))
    constraint = phx.discretization.dirichlet_constraint(
        discretization,
        "u",
        boundary_mask=boundary_mask,
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=0.0,
    )
    reduced = compiled.state_space.zeros()
    residual, auxiliary = compiled.residual_with_auxiliary(reduced)

    assert residual.shape == reduced.shape
    assert bool(auxiliary.valid)
    route.validate(auxiliary.trial_state)
    for site, shape in zip(route.site_ids, route.state_shapes, strict=True):
        assert auxiliary.trial_state.state(site).trial.shape == shape


def test_route_supports_exact_texture_fields_with_two_orientations():
    discretization = _one_block_two_cell_discretization()
    model = _model()
    quadrature_count = int(
        discretization.block_geometries[0][0].physical_weights.shape[1]
    )
    orientations = (
        jnp.broadcast_to(jnp.eye(3), (2, quadrature_count, 3, 3)).at[1].set(_rotation_z())
    )
    route = cpfem.CrystalPlasticityRoute(
        discretization,
        "u",
        (("phase", model, orientations),),
    )
    assert route.crystal_to_sample[0].shape == orientations.shape
    assert jnp.allclose(route.crystal_to_sample[0], orientations)

    transaction = route.initialize()
    form = cpfem.cpfem_equilibrium_form(discretization, "u", route, transaction, 0.1)
    displacement_gradient = _simple_shear(0.35) - jnp.eye(3)
    displacement = discretization.dof_maps[0].dof_coordinates @ displacement_gradient.T
    auxiliary = form.auxiliary_evaluator(displacement, None)
    trial = auxiliary.trial_state.state(route.site_ids[0]).trial
    assert bool(auxiliary.valid)
    assert not bool(jnp.allclose(trial[0, 0, :9], trial[1, 0, :9]))

    with pytest.raises(ValueError, match="exact site shape"):
        cpfem.CrystalPlasticityRoute(
            discretization,
            "u",
            (("phase", model, orientations[:, :-1]),),
        )
    reflection = jnp.diag(jnp.asarray((1.0, 1.0, -1.0)))
    invalid = orientations.at[1, 0].set(reflection)
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        cpfem.CrystalPlasticityRoute(
            discretization,
            "u",
            (("phase", model, invalid),),
        )


def test_global_rejection_requests_cutback_and_rolls_back_every_route():
    discretization = _two_block_discretization()
    route = _two_phase_route(discretization, bound=1.0e-6)
    transaction = route.initialize()
    form = cpfem.cpfem_equilibrium_form(discretization, "u", route, transaction, 0.1)
    deformation = _simple_shear(0.4)
    displacement_gradient = deformation - jnp.eye(3)
    displacement = discretization.dof_maps[0].dof_coordinates @ displacement_gradient.T
    auxiliary = form.auxiliary_evaluator(displacement, None)

    assert bool(auxiliary.successful)
    assert not bool(auxiliary.admissible)
    assert bool(auxiliary.retry_requested)
    assert 0.0 < auxiliary.suggested_step < 0.1
    candidate = auxiliary.trial_state
    assert any(
        not bool(jnp.allclose(state.trial, state.committed)) for state in candidate.states
    )
    rolled_back = route.rollback(candidate)
    for before, after in zip(transaction.states, rolled_back.states, strict=True):
        np.testing.assert_allclose(after.committed, before.committed)
        np.testing.assert_allclose(after.trial, before.committed)
        assert after.state_version == before.state_version


def test_route_rejects_overlap_gap_and_foreign_checkpoint_or_layout():
    discretization = _two_block_discretization()
    model = _model()
    with pytest.raises(ValueError, match="overlap"):
        cpfem.CrystalPlasticityRoute(
            discretization,
            "u",
            (
                ("phase-a", model, jnp.eye(3)),
                ("phase-a", model, _rotation_z()),
            ),
        )
    with pytest.raises(ValueError, match="gaps"):
        cpfem.CrystalPlasticityRoute(
            discretization,
            "u",
            (("phase-a", model, jnp.eye(3)),),
        )

    route = _two_phase_route(discretization)
    transaction = route.initialize()
    checkpoint = route.checkpoint(transaction)
    restored = route.restore(checkpoint)
    assert route.support_id == discretization.support.support_id
    assert checkpoint.plan_id == route.route_id
    assert checkpoint.payload_id
    assert restored.transaction_id == transaction.transaction_id

    foreign = cpfem.CrystalPlasticityRoute(
        discretization,
        "u",
        (
            ("phase-a", route.models[0], _rotation_z()),
            ("phase-b", route.models[1], _rotation_z()),
        ),
    )
    assert foreign.route_id != route.route_id
    assert foreign.orientation_ids[0] != route.orientation_ids[0]
    with pytest.raises(ValueError, match="another material route"):
        foreign.restore(checkpoint)

    first = transaction.state(route.site_ids[0])
    malformed = phx.equations.MaterialState(
        first.site_id,
        first.model_id,
        first.committed[..., :-1],
    )
    bad_transaction = phx.equations.MaterialTransaction(
        (
            malformed,
            transaction.state(route.site_ids[1]),
        )
    )
    bad_checkpoint = bad_transaction.checkpoint_payload(plan_id=route.route_id)
    with pytest.raises(ValueError, match="layout"):
        route.restore(bad_checkpoint)


def test_convergence_and_admissibility_are_distinct_cutback_decisions():
    model = _model(maximum_slip_increment=1.0e-6)
    update = model.update(_simple_shear(0.4), model.initial_state(), jnp.eye(3), 0.1)

    assert bool(update.converged)
    assert not bool(update.admissible)
    assert not bool(update.accepted)
    assert 0.0 < update.suggested_step_factor < 1.0
