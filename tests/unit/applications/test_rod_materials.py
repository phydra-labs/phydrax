from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax.ein as ein
from phydrax.applications.solid_mechanics._rod_dynamics import (
    evaluate_rod,
    prepare_rod,
    rod_potential_energy,
    RodPlan,
    RodState,
)
from phydrax.applications.solid_mechanics._rod_materials import (
    KelvinVoigtRodMaterialPlan,
    RodConstitutiveTrial,
)


def _planar_rod():
    return prepare_rod(
        RodPlan(
            jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
            jnp.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
            jnp.asarray((1.0, 1.5, 1.0)),
            jnp.asarray((0.2, 0.3)),
            jnp.asarray(
                (
                    ((100.0, 4.0), (4.0, 30.0)),
                    ((80.0, -3.0), (-3.0, 25.0)),
                )
            ),
            jnp.asarray((((5.0,),),)),
        )
    )


def test_prepared_rod_owns_content_derived_native_sites_measures_and_inertia():
    first = _planar_rod()
    second = _planar_rod()

    assert tuple(site.owner_kind for site in first.stretch_shear_sites) == (
        "segment",
        "segment",
    )
    assert tuple(site.owner_index for site in first.stretch_shear_sites) == (0, 1)
    assert tuple(site.owner_kind for site in first.bend_twist_sites) == ("junction",)
    assert jnp.array_equal(first.stretch_shear_measures, first.plan.rest_lengths)
    assert jnp.array_equal(first.bend_twist_measures, first.dual_lengths)
    assert jnp.all(first.stretch_shear_measures > 0.0)
    assert jnp.all(first.bend_twist_measures > 0.0)
    assert jnp.array_equal(
        first.stretch_shear_reference_strains, first.rest_stretch_shear
    )
    assert jnp.array_equal(first.bend_twist_reference_strains, jnp.zeros((1, 1)))
    assert jnp.array_equal(first.node_masses, first.plan.node_masses)
    assert jnp.array_equal(first.segment_inertias, first.plan.segment_inertias)
    assert first.material_workset_id == second.material_workset_id
    assert (
        first.stretch_shear_workset.workset_id == second.stretch_shear_workset.workset_id
    )
    assert first.bend_twist_workset.workset_id == second.bend_twist_workset.workset_id


def test_lowered_linear_material_reproduces_current_resultants_and_energy():
    rod = _planar_rod()
    rest = rod.initialize_state()
    state = RodState(
        rest.positions.at[1, 1].set(0.16).at[2, 0].add(0.08),
        rest.velocities,
        rest.orientations + jnp.asarray((0.07, 0.21)),
        rest.angular_velocities,
    )
    evaluation = evaluate_rod(rod, state)
    stretch_candidate = (
        rod.stretch_shear_reference_strains + evaluation.constitutive_stretch_shear_strain
    )
    bend_candidate = rod.bend_twist_reference_strains + evaluation.bend_twist_strain
    zero = jnp.asarray(0.0, dtype=state.positions.dtype)
    step = jnp.asarray(1.0, dtype=state.positions.dtype)
    stretch_trial = rod.stretch_shear_material(
        stretch_candidate,
        stretch_candidate,
        jnp.zeros_like(stretch_candidate),
        rod.stretch_shear_material.initialize_history(),
        None,
        zero,
        step,
    )
    bend_trial = rod.bend_twist_material(
        bend_candidate,
        bend_candidate,
        jnp.zeros_like(bend_candidate),
        rod.bend_twist_material.initialize_history(),
        None,
        zero,
        step,
    )
    expected_stretch_resultants = ein.contract(
        "sij,sj->si",
        rod.plan.stretch_shear_stiffness,
        evaluation.constitutive_stretch_shear_strain,
    )
    expected_bend_resultants = ein.contract(
        "sij,sj->si",
        rod.plan.bend_twist_stiffness,
        evaluation.bend_twist_strain,
    )

    assert isinstance(rod.stretch_shear_material, RodConstitutiveTrial)
    assert jnp.allclose(stretch_trial.resultants, expected_stretch_resultants)
    assert jnp.allclose(bend_trial.resultants, expected_bend_resultants)
    assert stretch_trial.evidence.valid
    assert bend_trial.evidence.valid
    assert stretch_trial.stored_energy + bend_trial.stored_energy == pytest.approx(
        rod_potential_energy(rod, state.positions, state.orientations),
        rel=2.0e-6,
        abs=2.0e-6,
    )
    assert evaluation.potential_energy == pytest.approx(
        stretch_trial.stored_energy + bend_trial.stored_energy,
        rel=2.0e-6,
        abs=2.0e-6,
    )


def test_kelvin_voigt_trial_is_zero_history_nonmutating_and_dissipative():
    rod = _planar_rod()
    stiffness = rod.plan.stretch_shear_stiffness
    viscosity = jnp.asarray(
        (
            ((2.0, 0.2), (0.2, 1.0)),
            ((1.5, -0.1), (-0.1, 0.8)),
        )
    )
    material = KelvinVoigtRodMaterialPlan(stiffness, viscosity).prepare(
        rod.stretch_shear_workset
    )
    source = rod.stretch_shear_reference_strains
    rate = jnp.asarray(((0.3, -0.2), (-0.1, 0.4)))
    step = jnp.asarray(0.125, dtype=source.dtype)
    candidate = source + step * rate
    history = material.initialize_history()

    first = material(
        source,
        candidate,
        rate,
        history,
        material.initialize_control(),
        jnp.asarray(0.5, dtype=source.dtype),
        step,
    )
    second = material(
        source,
        candidate,
        rate,
        history,
        None,
        jnp.asarray(0.5, dtype=source.dtype),
        step,
    )
    elastic = candidate - rod.stretch_shear_reference_strains
    expected_resultants = ein.contract("sij,sj->si", stiffness, elastic) + ein.contract(
        "sij,sj->si", viscosity, rate
    )
    expected_tangent = stiffness + viscosity / step
    expected_dissipation = step * jnp.sum(
        rod.stretch_shear_measures * ein.contract("si,sij,sj->s", rate, viscosity, rate)
    )

    assert material.history_size == 0
    assert history.shape == (rod.plan.segment_count, 0)
    assert first.evidence.valid
    assert first.evidence.dissipation_nonnegative
    assert first.viscous_dissipation >= 0.0
    assert first.viscous_dissipation == pytest.approx(expected_dissipation)
    assert jnp.allclose(first.resultants, expected_resultants)
    assert jnp.allclose(first.consistent_tangent, expected_tangent)
    assert jnp.array_equal(history, material.initialize_history())
    assert jnp.array_equal(first.candidate_history, history)
    assert jnp.array_equal(second.candidate_history, history)
    assert jnp.array_equal(first.resultants, second.resultants)
