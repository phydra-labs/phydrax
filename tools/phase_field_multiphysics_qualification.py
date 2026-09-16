#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint


PF = phx.applications.phase_field


def _mechanics():
    identity = jnp.eye(2)
    stiffness = 2.0 * jnp.einsum("ij,kl->ijkl", identity, identity) + (
        jnp.einsum("ik,jl->ijkl", identity, identity)
        + jnp.einsum("il,jk->ijkl", identity, identity)
    )
    return PF.PhaseMechanicalModel(
        (
            PF.LinearElasticPhaseMaterial(
                stiffness, jnp.zeros((2, 2)), material_id="qualification-mechanics-a"
            ),
            PF.LinearElasticPhaseMaterial(
                stiffness, 0.01 * identity, material_id="qualification-mechanics-b"
            ),
        )
    )


def _thermal():
    phases = (
        PF.NonisothermalGrandPotentialPhase(
            "qualification-solid",
            reference_temperature=1.0,
            reference_grand_potential=0.0,
            reference_entropy=0.0,
            reference_composition=jnp.asarray((0.2,)),
            susceptibility=jnp.asarray(((1.0,),)),
            heat_capacity=1.0,
            thermal_conductivity=1.0,
        ),
        PF.NonisothermalGrandPotentialPhase(
            "qualification-liquid",
            reference_temperature=1.0,
            reference_grand_potential=0.5,
            reference_entropy=1.0,
            reference_composition=jnp.asarray((0.8,)),
            susceptibility=jnp.asarray(((1.0,),)),
            heat_capacity=1.0,
            thermal_conductivity=1.0,
        ),
    )
    return PF.NonisothermalSolidificationPlan(
        PF.NonisothermalSolidificationModel(
            PF.NonisothermalMaterialCatalog(phases), barrier_scale=0.1
        )
    )


def _graph():
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


def build_coupled_case():
    thermal = _thermal()
    nucleation = PF.NucleationEventPlan(
        phx.stochastic.PoissonClockRealization(
            jax.random.key(21),
            1,
            support=(0.0, 10.0),
            max_events_per_channel=8,
            process_id="multiphysics-qualification-nucleation",
        ),
        PF.ClassicalNucleationRateLaw(
            surface_energy=0.1,
            kinetic_prefactor=1.0,
            boltzmann_constant=1.0,
            spatial_dimension=2,
            law_id="multiphysics-qualification-cnt",
        ),
        jnp.asarray((1.0,)),
        component_cost_density=0.1,
        energy_cost_density=0.2,
    )
    mechanics = _mechanics()
    flow = PF.ModelHCouplingPlan(
        PF.PhaseFluidMaterial(
            jnp.asarray((1.0, 1.0)),
            jnp.asarray((1.0, 1.0)),
            material_id="multiphysics-qualification-fluid",
        )
    )
    electrostatic = PF.PhaseElectrostaticCouplingPlan(
        PF.PhasePermittivityLaw(
            jnp.asarray((1.0, 2.0)), law_id="multiphysics-qualification-dielectric"
        ),
        ensemble="fixed-charge",
    )
    electrochemical = PF.ElectrochemicalCouplingPlan(
        jnp.asarray((0.0,)), jnp.asarray((1.0,)), faraday_constant=1.0
    )
    plan = PF.CoupledMultiphysicsPlan(
        _graph(),
        thermal,
        PF.AntiTrappingCurrentPlan(
            0.3, 0.1, 0.2, calibration_id="multiphysics-qualification-antitrapping"
        ),
        nucleation,
        mechanics,
        flow,
        electrostatic,
        electrochemical,
    )
    thermal_state = thermal.initialize(
        jnp.asarray((0.0, 0.0)), jnp.asarray((0.0,)), jnp.asarray(1.0)
    )
    mechanical = mechanics.evaluate(jnp.asarray((0.0, 0.0)), jnp.zeros((2, 2)))
    electric = electrostatic.evaluate(
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((-1.0, 0.0)),
        free_charge=jnp.asarray(0.0),
        displacement_divergence=jnp.asarray(0.0),
    )
    state = plan.initialize(
        thermal_state,
        mechanical_energy=mechanical.energy,
        kinetic_energy=0.0,
        electrostatic_energy=electric.field_energy,
        component_inventory=10.0,
        thermal_reservoir=10.0,
    )
    inputs = PF.CoupledMultiphysicsStepInputs(
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
    return plan, state, inputs


def _thermal_solidification() -> dict[str, object]:
    plan = _thermal()
    chemical = jnp.asarray((0.0,))
    initial_logits = jnp.asarray((10.0, -10.0))
    final_logits = jnp.asarray((-10.0, 10.0))
    initial = plan.initialize(initial_logits, chemical, jnp.asarray(1.0))
    reference = plan.model.evaluate(final_logits, chemical, jnp.asarray(1.0))
    heat = reference.internal_energy - initial.internal_energy
    final, evidence = plan.step(
        initial,
        final_logits,
        chemical,
        heat_input=heat,
        entropy_flux=jnp.sum(reference.entropy - initial.entropy),
        entropy_production=0.0,
    )
    passed = bool(evidence.successful)
    return {
        "status": "pass" if passed else "fail",
        "temperature": float(np.asarray(final.temperature)),
        "energy_defect": float(np.asarray(evidence.energy_defect)),
        "entropy_defect": float(np.asarray(evidence.entropy.residual)),
        "constitutive_residual": float(np.asarray(evidence.constitutive_residual)),
    }


def _anti_trapping_and_nucleation() -> dict[str, object]:
    anti = PF.AntiTrappingCurrentPlan(
        1.0 / (2.0 * jnp.sqrt(2.0)),
        0.1,
        0.2,
        calibration_id="qualification-anti-trapping",
    ).evaluate(
        jnp.asarray((1.0, 0.0)),
        jnp.asarray(((1.0, 0.0), (0.0, 0.0))),
        jnp.asarray((0.1, 0.1)),
    )
    clock = phx.stochastic.PoissonClockRealization(
        jax.random.key(7),
        2,
        support=(0.0, 10.0),
        max_events_per_channel=8,
        process_id="qualification-nucleation",
    )
    plan = PF.NucleationEventPlan(
        clock,
        PF.ClassicalNucleationRateLaw(
            surface_energy=0.1,
            kinetic_prefactor=2.0,
            boltzmann_constant=1.0,
            spatial_dimension=2,
            law_id="qualification-cnt",
        ),
        jnp.asarray((1.0, 1.0)),
        component_cost_density=0.1,
        energy_cost_density=0.2,
    )
    initial = plan.initialize()
    candidate, proposal = plan.propose(
        initial,
        0.0,
        1.0,
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 1.0)),
    )
    replay, repeated = plan.propose(
        initial,
        0.0,
        1.0,
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 1.0)),
    )
    transaction = plan.transact(
        initial,
        candidate,
        proposal,
        available_component=10.0,
        available_energy=10.0,
    )
    replay_defect = float(
        np.max(np.abs(np.asarray(proposal.event_times - repeated.event_times)))
    )
    hazard_defect = float(
        np.max(np.abs(np.asarray(candidate.integrated_hazard - replay.integrated_hazard)))
    )
    passed = (
        bool(anti.successful)
        and bool(transaction.successful)
        and replay_defect == 0.0
        and hazard_defect == 0.0
    )
    return {
        "status": "pass" if passed else "fail",
        "anti_trapping_current": np.asarray(anti.current).tolist(),
        "off_interface_current": np.asarray(anti.current[1]).tolist(),
        "event_count": int(np.asarray(transaction.candidate.accepted_events)),
        "event_replay_defect": replay_defect,
        "hazard_replay_defect": hazard_defect,
        "component_used": float(np.asarray(transaction.component_used)),
        "energy_used": float(np.asarray(transaction.energy_used)),
    }


def _mechanics_flow_electrostatic() -> dict[str, object]:
    mechanics = _mechanics().evaluate(jnp.asarray((0.0, 0.0)), jnp.zeros((2, 2)))
    flow = PF.ModelHCouplingPlan(
        PF.PhaseFluidMaterial(
            jnp.asarray((1.0, 1.0)),
            jnp.asarray((1.0, 2.0)),
            material_id="qualification-two-fluid",
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
        PF.PhasePermittivityLaw(
            jnp.asarray((1.0, 2.0)), law_id="qualification-dielectric"
        ),
        ensemble="fixed-charge",
    ).evaluate(
        jnp.asarray(((0.0, 0.0),)),
        jnp.asarray(((-1.0, 0.0),)),
        free_charge=jnp.asarray((1.0,)),
        displacement_divergence=jnp.asarray((1.0,)),
    )
    passed = bool(mechanics.successful & flow.successful & electrostatic.successful)
    return {
        "status": "pass" if passed else "fail",
        "elastic_energy": float(np.asarray(mechanics.energy)),
        "mechanical_phase_force": np.asarray(mechanics.phase_force).tolist(),
        "capillary_exchange_defect": float(
            np.max(np.abs(np.asarray(flow.exchange_defect)))
        ),
        "electrostatic_energy": float(np.asarray(electrostatic.field_energy[0])),
        "gauss_defect": float(np.max(np.abs(np.asarray(electrostatic.gauss_residual)))),
        "maxwell_stress": np.asarray(electrostatic.maxwell_stress[0]).tolist(),
    }


def _power_transfer_and_distributed() -> dict[str, object]:
    source = phx.linalg.ArraySpace((2,), dtype=np.float64)
    target = phx.linalg.ArraySpace((2,), dtype=np.float64)
    matrix = jnp.asarray(((0.75, 0.25), (0.25, 0.75)), dtype=jnp.float64)
    transfer = PF.PowerAdjointTransferPair(
        phx.linalg.DenseLinearOperator(matrix, source=source, target=target),
        phx.linalg.DenseLinearOperator(matrix.T, source=target, target=source),
        transfer_id="qualification-power-transfer",
    )
    power = transfer.evaluate(jnp.asarray((1.0, 2.0)), jnp.asarray((3.0, -1.0)))
    mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            dtype=jnp.float64,
        ),
        jnp.asarray(((0, 1, 3), (1, 2, 3)), dtype=jnp.int32),
    )
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("eta", element)
    ).prepare()
    distributed = PF.DistributedPhaseFieldPlan(discretization, 2)
    values = jnp.asarray((1.0, 2.0))
    reference = distributed.reference_owned_sum(values)
    if jax.device_count() >= 2:
        device_mesh = Mesh(
            np.asarray(jax.devices()[:2], dtype=object), (distributed.axis_name,)
        )

        def execute(cell_values):
            return distributed.collective_owned_sum(cell_values).global_value

        collective = jax.shard_map(
            execute,
            mesh=device_mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )(values)
        collective_defect = float(np.max(np.abs(np.asarray(collective - reference))))
    else:
        collective_defect = float("inf")
    passed = bool(power.successful) and collective_defect == 0.0
    return {
        "status": "pass" if passed else "fail",
        "power_defect": float(np.asarray(power.power_defect)),
        "constant_defect": float(np.asarray(power.constant_defect)),
        "distributed_collective_defect": collective_defect,
        "distributed_device_count": jax.device_count(),
    }


def _flagship() -> dict[str, object]:
    plan, state, inputs = build_coupled_case()
    result = plan.step(state, inputs, 0.0, 0.01)
    method = PF.CoupledMultiphysicsFixedStepMethod(plan)
    case = method.production_case(
        "multiphysics-qualification",
        state,
        precision_id="float64",
        topology_id="point-reference",
        geometry_layout_id="point-reference",
        dtype="float64",
    )
    run_plan = method.production_run_plan(
        step_size=0.01,
        end_time=0.01,
        maximum_steps=1,
        checkpoint_interval=1,
        segment_steps=1,
        retry_policy=phx.solver.RobustRetryPolicy(maximum_retries=0),
    )
    args_id = canonical_fingerprint(
        {
            "kind": "coupled-multiphysics-qualification-arguments",
            "arrays": array_tree_fingerprint(inputs),
        }
    )
    with tempfile.TemporaryDirectory(prefix="phydrax-multiphysics-") as directory:
        policy = phx.solver.CheckpointGenerationPolicy(2)
        store = phx.solver.DurableCheckpointStore(directory, case.manifest, policy)
        runtime = phx.solver.PreparedProductionRun(
            case.manifest,
            run_plan,
            store,
            args=inputs,
            args_id=args_id,
        )
        run = runtime.run(runtime.initial_state(case.initial_state))
        resumed_store = phx.solver.DurableCheckpointStore(
            directory, case.manifest, policy
        )
        resumed_runtime = phx.solver.PreparedProductionRun(
            case.manifest,
            run_plan,
            resumed_store,
            args=inputs,
            args_id=args_id,
        )
        resumed = resumed_runtime.resume(
            resumed_runtime.initial_state(case.initial_state)
        )
    restart_defect = max(
        float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
        for left, right in zip(
            jax.tree.leaves(run.state.accepted_state),
            jax.tree.leaves(resumed.accepted_state),
            strict=True,
        )
    )
    passed = bool(result.successful) and bool(run.successful) and restart_defect == 0.0
    return {
        "status": "pass" if passed else "fail",
        "energy_residual": float(np.asarray(result.evidence.ledger.energy_residual)),
        "exchange_defect": float(np.asarray(result.evidence.ledger.exchange_defect)),
        "conservation_defect": float(
            np.asarray(result.evidence.ledger.maximum_conservation_defect)
        ),
        "entropy_defect": float(np.asarray(result.evidence.ledger.entropy.residual)),
        "anti_trapping_finite": bool(result.evidence.anti_trapping.finite),
        "nucleation_events": int(
            np.asarray(result.evidence.nucleation_transaction.candidate.accepted_events)
        ),
        "production_step": int(np.asarray(run.state.step_index)),
        "restart_defect": restart_defect,
    }


def qualify() -> dict[str, object]:
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError("Multiphysics phase-field qualification requires float64.")
    sections = {
        "thermal_solidification_reference": _thermal_solidification(),
        "anti_trapping_nucleation_reference": _anti_trapping_and_nucleation(),
        "mechanics_flow_electrostatic_reference": _mechanics_flow_electrostatic(),
        "power_transfer_distributed_reference": _power_transfer_and_distributed(),
        "integrated_flagship_reference": _flagship(),
    }
    passed = all(section["status"] == "pass" for section in sections.values())
    artifact_id = canonical_fingerprint(
        {"kind": "phase-field-multiphysics-qualification", "sections": sections}
    )
    released = (
        PF.coupled_phase_field_released_profiles(
            artifact_id,
            reviewer_id="phydrax-multiphysics-qualification",
            issued_at=1767225600,
            expires_at=1798761600,
        )
        if passed
        else ()
    )
    return {
        "status": "pass" if passed else "fail",
        "capability": "coupled-phase-field-multiphysics",
        "artifact_id": artifact_id,
        "released_profiles": [profile.to_record() for profile in released],
        "sections": sections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify coupled nonisothermal phase-field multiphysics."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/phase_field_multiphysics_qualification.json"),
    )
    arguments = parser.parse_args()
    report = qualify()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(arguments.output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
