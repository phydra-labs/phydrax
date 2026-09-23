#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


PF = phx.applications.phase_field


def _thermal_plan():
    solid = PF.NonisothermalGrandPotentialPhase(
        "solid",
        reference_temperature=1.0,
        reference_grand_potential=0.0,
        reference_entropy=0.0,
        reference_composition=jnp.asarray((0.2,)),
        susceptibility=jnp.asarray(((1.0,),)),
        heat_capacity=1.0,
        thermal_conductivity=1.0,
    )
    liquid = PF.NonisothermalGrandPotentialPhase(
        "liquid",
        reference_temperature=1.0,
        reference_grand_potential=0.5,
        reference_entropy=1.0,
        reference_composition=jnp.asarray((0.8,)),
        susceptibility=jnp.asarray(((1.0,),)),
        heat_capacity=1.0,
        thermal_conductivity=1.0,
    )
    model = PF.NonisothermalSolidificationModel(
        PF.NonisothermalMaterialCatalog((solid, liquid)), barrier_scale=0.1
    )
    return PF.NonisothermalSolidificationPlan(model)


def _mechanics():
    identity = jnp.eye(2)
    stiffness = 2.0 * jnp.einsum("ij,kl->ijkl", identity, identity) + (
        jnp.einsum("ik,jl->ijkl", identity, identity)
        + jnp.einsum("il,jk->ijkl", identity, identity)
    )
    phase_a = PF.LinearElasticPhaseMaterial(
        stiffness, jnp.zeros((2, 2)), material_id="mechanical-a"
    )
    phase_b = PF.LinearElasticPhaseMaterial(
        stiffness, 0.01 * identity, material_id="mechanical-b"
    )
    return PF.PhaseMechanicalModel((phase_a, phase_b))


def _coupling_graph():
    return PF.PhaseFieldCouplingGraph(
        (
            PF.PhaseFieldCouplingTerm(
                "thermal",
                output_fields=("phase", "temperature", "composition", "chemical"),
                storage_channels=("thermochemical",),
                dissipation_channels=("thermal",),
                work_channels=("heat",),
                exchange_inputs=("capillary", "joule-heating", "nucleation-energy"),
                conservation_channels=("material-component",),
            ),
            PF.PhaseFieldCouplingTerm(
                "nucleation",
                input_fields=("temperature",),
                output_fields=("events",),
                storage_channels=("nucleation-interface",),
                exchange_outputs=("nucleation-energy",),
            ),
            PF.PhaseFieldCouplingTerm(
                "mechanics",
                input_fields=("phase",),
                output_fields=("displacement",),
                storage_channels=("elastic",),
                dissipation_channels=("mechanical",),
                work_channels=("mechanical-work",),
            ),
            PF.PhaseFieldCouplingTerm(
                "flow",
                input_fields=("phase", "chemical"),
                output_fields=("velocity",),
                storage_channels=("kinetic",),
                dissipation_channels=("viscous",),
                work_channels=("flow-work",),
                exchange_outputs=("capillary",),
            ),
            PF.PhaseFieldCouplingTerm(
                "electrostatic",
                input_fields=("phase", "composition"),
                output_fields=("potential",),
                storage_channels=("electrostatic",),
                dissipation_channels=("electrical",),
                work_channels=("electrical-work",),
                exchange_outputs=("joule-heating",),
                conservation_channels=("electric-charge",),
            ),
        )
    )


def _coupled_plan():
    clock = phx.stochastic.PoissonClockRealization(
        jax.random.key(4),
        1,
        support=(0.0, 10.0),
        max_events_per_channel=4,
        process_id="coupled-nucleation",
    )
    nucleation = PF.NucleationEventPlan(
        clock,
        PF.ClassicalNucleationRateLaw(
            surface_energy=0.1,
            kinetic_prefactor=1.0,
            boltzmann_constant=1.0,
            spatial_dimension=2,
            law_id="coupled-cnt",
        ),
        jnp.asarray((1.0,)),
        component_cost_density=0.1,
        energy_cost_density=0.2,
    )
    flow = PF.ModelHCouplingPlan(
        PF.PhaseFluidMaterial(
            jnp.asarray((1.0, 1.0)),
            jnp.asarray((1.0, 1.0)),
            material_id="coupled-flow",
        )
    )
    electrostatic = PF.PhaseElectrostaticCouplingPlan(
        PF.PhasePermittivityLaw(jnp.asarray((1.0, 2.0)), law_id="coupled-dielectric"),
        ensemble="fixed-charge",
    )
    electrochemical = PF.ElectrochemicalCouplingPlan(
        jnp.asarray((0.0,)),
        jnp.asarray((1.0,)),
        faraday_constant=1.0,
    )
    return PF.CoupledMultiphysicsPlan(
        _coupling_graph(),
        _thermal_plan(),
        PF.AntiTrappingCurrentPlan(0.3, 0.1, 0.2, calibration_id="coupled-anti-trapping"),
        nucleation,
        _mechanics(),
        flow,
        electrostatic,
        electrochemical,
    )


def _coupled_inputs():
    return PF.CoupledMultiphysicsStepInputs(
        phase_logits=jnp.asarray((0.0, 0.0)),
        chemical_potential=jnp.asarray((0.0,)),
        phase_rate=jnp.asarray(0.0),
        phase_value=jnp.asarray(0.0),
        phase_gradient=jnp.asarray((1.0, 0.0)),
        scalar_chemical_potential=jnp.asarray(0.0),
        scalar_chemical_gradient=jnp.asarray((0.0, 0.0)),
        displacement_gradient=jnp.zeros((2, 2)),
        velocity=jnp.zeros((2,)),
        velocity_gradient=jnp.zeros((2, 2)),
        electric_potential=jnp.asarray(0.0),
        potential_gradient=jnp.asarray((-1.0, 0.0)),
        displacement_divergence=jnp.asarray(0.0),
        free_charge=jnp.asarray(0.0),
        concentrations=jnp.asarray((0.5,)),
        component_chemical_potentials=jnp.asarray((0.0,)),
        component_chemical_gradients=jnp.zeros((1, 2)),
        nucleation_driving_force=jnp.asarray((0.0,)),
        nucleation_temperature=jnp.asarray((1.0,)),
        heat_input=jnp.asarray(0.0),
        entropy_flux=jnp.asarray(0.0),
        entropy_production=jnp.asarray(0.0),
        mechanical_work=jnp.asarray(0.0),
        flow_work=jnp.asarray(0.0),
        electrical_work=jnp.asarray(0.0),
    )


def test_coupling_graph_rejects_duplicate_storage_owner():
    first = PF.PhaseFieldCouplingTerm(
        "first", output_fields=("x",), storage_channels=("energy",)
    )
    second = PF.PhaseFieldCouplingTerm(
        "second", output_fields=("y",), storage_channels=("energy",)
    )
    with pytest.raises(ValueError, match="one owner"):
        PF.PhaseFieldCouplingGraph((first, second))


def test_power_adjoint_transfer_closes_discrete_power():
    source = phx.linalg.ArraySpace((2,), dtype=np.float64)
    target = phx.linalg.ArraySpace((2,), dtype=np.float64)
    matrix = jnp.asarray(((0.75, 0.25), (0.25, 0.75)), dtype=jnp.float64)
    primal = phx.linalg.DenseLinearOperator(matrix, source=source, target=target)
    dual = phx.linalg.DenseLinearOperator(matrix.T, source=target, target=source)
    transfer = PF.PowerAdjointTransferPair(primal, dual, transfer_id="symmetric-transfer")

    evidence = transfer.evaluate(jnp.asarray((1.0, 2.0)), jnp.asarray((3.0, -1.0)))

    assert bool(evidence.successful)
    np.testing.assert_allclose(evidence.power_defect, 0.0)
    np.testing.assert_allclose(evidence.constant_defect, 0.0)


def test_nonisothermal_phase_change_closes_enthalpy_and_entropy():
    plan = _thermal_plan()
    chemical = jnp.asarray((0.0,))
    initial_logits = jnp.asarray((10.0, -10.0))
    final_logits = jnp.asarray((-10.0, 10.0))
    initial = plan.initialize(initial_logits, chemical, jnp.asarray(1.0))
    final_reference = plan.model.evaluate(final_logits, chemical, jnp.asarray(1.0))
    heat = final_reference.internal_energy - initial.internal_energy
    entropy_flux = jnp.sum(final_reference.entropy - initial.entropy)

    final, evidence = plan.step(
        initial,
        final_logits,
        chemical,
        heat_input=heat,
        entropy_flux=entropy_flux,
        entropy_production=0.0,
    )

    assert bool(evidence.successful)
    np.testing.assert_allclose(final.temperature, 1.0)
    np.testing.assert_allclose(evidence.energy_defect, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(evidence.entropy.residual, 0.0, atol=1.0e-12)


def test_anti_trapping_current_is_directional_and_zero_off_interface():
    plan = PF.AntiTrappingCurrentPlan(
        1.0 / (2.0 * jnp.sqrt(2.0)),
        0.1,
        0.2,
        calibration_id="quantitative-dilute-alloy",
    )
    evaluation = plan.evaluate(
        jnp.asarray((1.0, 0.0)),
        jnp.asarray(((1.0, 0.0), (0.0, 0.0))),
        jnp.asarray((0.1, 0.1)),
    )

    assert bool(evaluation.successful)
    assert evaluation.current[0, 0] < 0.0
    np.testing.assert_array_equal(evaluation.current[1], jnp.zeros((2,)))


def test_nucleation_clock_is_prefix_stable_and_transactional():
    clock = phx.stochastic.PoissonClockRealization(
        jax.random.key(3),
        2,
        support=(0.0, 10.0),
        max_events_per_channel=8,
        process_id="nucleation-regression",
    )
    plan = PF.NucleationEventPlan(
        clock,
        PF.ClassicalNucleationRateLaw(
            surface_energy=0.1,
            kinetic_prefactor=2.0,
            boltzmann_constant=1.0,
            spatial_dimension=2,
            law_id="cnt-2d",
        ),
        jnp.asarray((1.0, 1.0)),
        component_cost_density=0.1,
        energy_cost_density=0.2,
    )
    state = plan.initialize()
    candidate, proposal = plan.propose(
        state,
        0.0,
        1.0,
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 1.0)),
    )
    replay, repeated = plan.propose(
        state,
        0.0,
        1.0,
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 1.0)),
    )
    transaction = plan.transact(
        state,
        candidate,
        proposal,
        available_component=10.0,
        available_energy=10.0,
    )

    assert bool(transaction.successful)
    np.testing.assert_array_equal(proposal.valid, repeated.valid)
    np.testing.assert_array_equal(proposal.event_times, repeated.event_times)
    np.testing.assert_array_equal(candidate.integrated_hazard, replay.integrated_hazard)
    assert transaction.candidate.accepted_events > 0
    other_candidate, other_proposal = plan.propose(
        state,
        0.0,
        0.5,
        jnp.asarray((0.5, 0.5)),
        jnp.asarray((1.0, 1.0)),
    )
    del other_candidate
    with pytest.raises(eqx.EquinoxRuntimeError, match="source/candidate"):
        plan.transact(
            state,
            candidate,
            other_proposal,
            available_component=10.0,
            available_energy=10.0,
        )


def test_mechanical_flow_and_electrostatic_exchange_contracts():
    mechanics = _mechanics().evaluate(jnp.asarray((0.0, 0.0)), jnp.zeros((2, 2)))
    flow = PF.ModelHCouplingPlan(
        PF.PhaseFluidMaterial(
            jnp.asarray((1.0, 1.0)),
            jnp.asarray((1.0, 2.0)),
            material_id="two-fluid-regression",
        )
    ).evaluate(
        jnp.asarray(((0.5, 0.5),)),
        jnp.asarray((0.0,)),
        jnp.asarray(((1.0, 0.0),)),
        jnp.asarray((2.0,)),
        jnp.asarray(((0.0, 0.0),)),
        jnp.asarray(((3.0, 0.0),)),
        jnp.zeros((1, 2, 2)),
    )
    electrostatic = PF.PhaseElectrostaticCouplingPlan(
        PF.PhasePermittivityLaw(jnp.asarray((1.0, 2.0)), law_id="dielectric-regression"),
        ensemble="fixed-charge",
    ).evaluate(
        jnp.asarray(((0.0, 0.0),)),
        jnp.asarray(((-1.0, 0.0),)),
        free_charge=jnp.asarray((1.0,)),
        displacement_divergence=jnp.asarray((1.0,)),
    )

    assert bool(mechanics.successful)
    assert bool(flow.successful)
    assert bool(electrostatic.successful)
    np.testing.assert_allclose(flow.exchange_defect, 0.0)
    assert mechanics.energy > 0.0
    assert electrostatic.field_energy[0] > 0.0


def test_complete_multiphysics_step_closes_all_ledgers():
    plan = _coupled_plan()
    thermal = plan.thermal.initialize(
        jnp.asarray((0.0, 0.0)), jnp.asarray((0.0,)), jnp.asarray(1.0)
    )
    mechanical = plan.mechanics.evaluate(jnp.asarray((0.0, 0.0)), jnp.zeros((2, 2)))
    electric = plan.electrostatic.evaluate(
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((-1.0, 0.0)),
        free_charge=jnp.asarray(0.0),
        displacement_divergence=jnp.asarray(0.0),
    )
    state = plan.initialize(
        thermal,
        mechanical_energy=mechanical.energy,
        kinetic_energy=0.0,
        electrostatic_energy=electric.field_energy,
        component_inventory=10.0,
        thermal_reservoir=10.0,
    )

    result = plan.step(state, _coupled_inputs(), 0.0, 0.01)

    assert bool(result.successful)
    np.testing.assert_allclose(result.evidence.ledger.energy_residual, 0.0)
    np.testing.assert_allclose(result.evidence.ledger.exchange_defect, 0.0)
    np.testing.assert_allclose(result.evidence.ledger.maximum_conservation_defect, 0.0)
    np.testing.assert_allclose(result.evidence.ledger.entropy.residual, 0.0)
    fixed_method = PF.CoupledMultiphysicsFixedStepMethod(plan)
    fixed = fixed_method.step(
        jnp.asarray(0),
        jnp.asarray(0.0),
        state,
        jnp.asarray(0.01),
        _coupled_inputs(),
    )
    assert bool(fixed.successful)
    case = fixed_method.production_case(
        "multiphysics-regression",
        state,
        precision_id="float64",
        topology_id="point-reference",
        geometry_layout_id="point-reference",
        dtype="float64",
    )
    assert case.manifest.method_id == fixed_method.method_id


def test_coupled_profiles_and_epoch_identity_are_exact():
    candidates = PF.coupled_phase_field_candidate_profiles()
    released = PF.coupled_phase_field_released_profiles(
        "coupled-qualification-artifact",
        reviewer_id="coupled-reviewer",
        issued_at=1767225600,
        expires_at=1798761600,
    )
    epoch = PF.CoupledMultiphysicsEpochIdentity(
        phase_epoch_id="phase",
        thermal_epoch_id="thermal",
        mechanics_topology_id="mechanics",
        flow_topology_id="flow",
        electrostatic_topology_id="electrostatic",
        partition_id="partition",
        event_realization_id="events",
    )

    assert len(candidates) == 8
    assert not any(profile.released for profile in candidates)
    assert all(profile.released for profile in released)
    assert released[-1].name == "phase-field-electro-elasto-hydrodynamic-flagship"
    assert len(released[-1].dependencies) == 7
    assert epoch.epoch_id
