from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


mn = phx.applications.solid_mechanics.member_network


def _beam_section():
    base = mn.BeamSection(1.0, 2.0, 1.0, 0.5, 0.8, 0.8)
    return mn.WarpingBeamSection(base, 0.1, 0.0, 3.0)


def test_warping_beam_and_bracing_energy():
    section = _beam_section()
    state = mn.evaluate_warping_beam(
        2.0,
        0.01,
        jnp.asarray(((0.0, 0.0, 0.0), (0.1, 0.02, -0.03))),
        jnp.asarray((0.0, 0.04)),
        200.0,
        80.0,
        section,
        load_height_force=2.0,
    )
    assert state.valid
    assert state.warping_energy > 0.0
    assert state.bimoment != 0.0
    brace = mn.evaluate_bracing(
        jnp.asarray((0.1,)),
        jnp.asarray((0.02,)),
        jnp.asarray((0.03,)),
        jnp.asarray((10.0,)),
        jnp.asarray((20.0,)),
        jnp.asarray((30.0,)),
    )
    assert brace.total_reaction > 0.0


def test_fiber_section_elastic_plastic_transaction():
    geometry = mn.FiberSectionGeometry(
        jnp.asarray(((-0.5, 0.0), (0.5, 0.0))),
        jnp.asarray((0.5, 0.5)),
        jnp.asarray((0, 0)),
    )
    material = mn.BilinearFiberMaterial(
        200.0, 2.0, isotropic_hardening=5.0, kinematic_hardening=5.0
    )
    history = mn.FiberMaterialHistory.zeros(2, jnp.float64)
    transaction = mn.FiberSectionTransaction(history, history)
    elastic, elastic_transaction = mn.evaluate_fiber_section(
        geometry, (material,), jnp.asarray((0.001, 0.0, 0.0)), transaction
    )
    assert not jnp.any(elastic.yielded)
    plastic, plastic_transaction = mn.evaluate_fiber_section(
        geometry, (material,), jnp.asarray((0.02, 0.0, 0.0)), elastic_transaction
    )
    assert jnp.all(plastic.yielded)
    assert plastic.plastic_dissipation > 0.0
    committed = plastic_transaction.commit()
    assert jnp.allclose(
        committed.committed.plastic_strain, plastic.trial_history.plastic_strain
    )


def test_gbt_finite_strip_shell_hierarchy():
    section = mn.ThinWalledSection(
        jnp.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 0.2))),
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        jnp.asarray((0.01, 0.01)),
        free_edge_nodes=jnp.asarray((True, False, True)),
    )
    basis = mn.compute_gbt_modes(section)
    assert basis.modes.shape == (3, 3)
    problem = mn.FiniteStripBucklingProblem(
        section,
        200_000.0,
        0.3,
        jnp.asarray((-100.0, -100.0)),
        jnp.geomspace(0.1, 10.0, 20),
    )
    strip = mn.solve_finite_strip_buckling(problem)
    assert strip.successful
    assert strip.critical_stress > 0.0
    shell = mn.compare_shell_submodel(
        strip.critical_stress * 1.05,
        strip.critical_stress,
        strip.critical_stress * 0.98,
        1.0e-8,
    )
    assert shell.successful
    assert shell.governing_factor > 0.0


def test_collapse_and_dynamic_evidence():
    collapse = mn.classify_collapse(
        1.5,
        jnp.asarray((-0.1, 2.0)),
        yielded=jnp.asarray((True, True)),
        state_norm=2.0,
    )
    assert collapse.event == int(mn.CollapseEventType.TANGENT_INSTABILITY)

    state = mn.MemberDynamicState(
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
        jnp.asarray((0.0,)),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(0.0),
    )
    advanced = mn.newmark_step(
        jnp.asarray(((1.0,),)),
        jnp.asarray(((0.1,),)),
        jnp.asarray(((10.0,),)),
        jnp.asarray((1.0,)),
        state,
        0.01,
    )
    assert advanced.displacement[0] > 0.0
    assert advanced.kinetic_energy >= 0.0
