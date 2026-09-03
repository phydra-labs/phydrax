#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.cardiovascular import anatomy, electrophysiology, mechanics


_SOURCE_EPOCH = (11, 7)
_TARGET_EPOCH = (12, 8)


def _cardiac_mesh(cell_kind: str):
    if cell_kind == "tetrahedron":
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    elif cell_kind == "hexahedron":
        vertices = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (1.0, 1.0, 1.0),
                (0.0, 1.0, 1.0),
            )
        )
        cells = np.asarray(((0, 1, 2, 3, 4, 5, 6, 7),), dtype=np.int32)
    else:
        raise ValueError(
            "Cardiac high-order qualification admits tetrahedra or hexahedra."
        )
    block = phx.discretization.CellBlock("myocardium", cell_kind, cells)
    return phx.discretization.CellMesh(vertices, (block,))


def _curved_coordinates(element, amplitude: float):
    reference = jnp.asarray(element.reference_nodes)
    x = reference[:, 0]
    y = reference[:, 1]
    lift = float(amplitude) * x * (1.0 - x) * (0.75 + 0.25 * y)
    return reference.at[:, 2].add(lift)


def _prepare_geometry(cell_kind: str, amplitude: float, epoch):
    mesh = _cardiac_mesh(cell_kind)
    element = phx.discretization.lagrange_element(cell_kind, 2)
    coordinate_spec = phx.discretization.FiniteElementCoordinateSpec(
        {"myocardium": element},
        {"myocardium": jnp.arange(element.local_dof_count)[None, :]},
        _curved_coordinates(element, amplitude),
    )
    profile = anatomy.CardiacBoundaryProfile(
        f"qualified-{cell_kind}-cardiac-volume",
        required_roles=("epicardium",),
    )
    prepared_epoch = anatomy.HighOrderGeometryEpoch(*epoch)
    plan = anatomy.HighOrderCardiacGeometryPlan(
        mesh,
        coordinate_spec,
        boundary_role_id=f"qualified-{cell_kind}-boundary-roles",
        boundary_profile=profile,
        prepared_epoch=prepared_epoch,
    )
    return plan.prepare()


def _geometry_transition(cell_kind: str):
    source = _prepare_geometry(cell_kind, 0.025, _SOURCE_EPOCH)
    source_candidate = source.evaluate(
        source.plan.coordinate_spec.coordinates,
        source.plan.prepared_epoch,
        boundary_role_id=source.plan.boundary_role_id,
        boundary_profile_id=source.plan.boundary_profile.profile_id,
    )
    target_coordinates = _curved_coordinates(source.coordinate_elements[0], 0.075)
    candidate = source.evaluate(
        target_coordinates,
        source.plan.prepared_epoch,
        boundary_role_id=source.plan.boundary_role_id,
        boundary_profile_id=source.plan.boundary_profile.profile_id,
    )
    target_epoch = anatomy.HighOrderGeometryEpoch(*_TARGET_EPOCH)
    target = source.commit_epoch(candidate, target_epoch)
    target_candidate = target.evaluate(
        target_coordinates,
        target_epoch,
        boundary_role_id=target.plan.boundary_role_id,
        boundary_profile_id=target.plan.boundary_profile.profile_id,
    )
    stale_candidate = target.evaluate(
        target_coordinates,
        source.plan.prepared_epoch,
        boundary_role_id=target.plan.boundary_role_id,
        boundary_profile_id=target.plan.boundary_profile.profile_id,
    )
    return source, source_candidate, target, target_candidate, stale_candidate


def _ep_discretization(prepared_geometry, label: str):
    element_by_block = dict(
        zip(
            prepared_geometry.plan.coordinate_spec.block_names,
            prepared_geometry.coordinate_elements,
            strict=True,
        )
    )
    field = phx.discretization.FiniteElementFieldSpec("voltage_mV", element_by_block)
    epoch = prepared_geometry.plan.prepared_epoch
    numeric_version = (
        f"{label}-geometry-{int(np.asarray(epoch.geometry))}"
        f"-reference-{int(np.asarray(epoch.reference))}"
    )
    return phx.discretization.FiniteElementPlan(
        prepared_geometry.plan.mesh,
        field,
        coordinate_spec=prepared_geometry.plan.coordinate_spec,
    ).prepare(numeric_version=numeric_version)


def _tensor_diffusion(discretization):
    dtype = discretization.field_spaces[0].vector_space.dtype
    conductivity = jnp.asarray(
        ((0.0016, 0.0002, 0.0), (0.0002, 0.0007, 0.0), (0.0, 0.0, 0.0003)),
        dtype=dtype,
    )
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_semidefinite": "construction",
        },
    )
    action = phx.equations.TensorDiffusionAction(
        "voltage_mV",
        conductivity,
        properties=properties,
        action_id="qualified-cardiac-conductivity-tensor",
    )
    form = phx.equations.FiniteElementForm(
        "qualified-high-order-cardiac-monodomain",
        "voltage_mV",
        (action,),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(realization="sparse"),
    )
    return action, compiled.affine_operator(), conductivity


def _regional_assignment(node_count: int):
    plan = electrophysiology.RegionalElectrophysiologyPlan(
        node_count,
        (electrophysiology.RegionalPhenotype("ventricular-myocardium", 0),),
    )
    return plan.prepare(
        np.arange(10_000, 10_000 + node_count, dtype=np.int64),
        np.zeros((node_count,), dtype=np.int32),
    )


def _node_volumes(candidate, node_count: int, dtype):
    total_measure = jnp.sum(jnp.concatenate(tuple(candidate.block_cell_measures_mm3)))
    return jnp.full((node_count,), total_measure / node_count, dtype=dtype)


def _physical_runtime(action, operator, assignment, volumes, integration_plan, label):
    diffusion = electrophysiology.TensorDiffusionOperatorInput(
        action,
        operator,
        assignment.runtime_id,
        tensor_action_id=action.action_id,
        input_id=f"{label}-tensor-diffusion-input",
    )
    spatial = electrophysiology.PhysicalMonodomainSpatialBinding(
        volumes,
        diffusion,
        binding_id=f"{label}-physical-spatial-binding",
    )
    dtype = np.dtype(volumes.dtype)
    model = electrophysiology.TenTusscherPanfilov2006Model()
    reaction = electrophysiology.prepare_reaction(
        electrophysiology.plan_reaction(model, assignment.node_count, dtype=dtype)
    )
    return integration_plan.prepare(spatial, assignment, (reaction,)), model


def _cardiac_transfer(
    source_discretization,
    target_discretization,
    source_geometry,
    target_geometry,
    *,
    quantity_id: str,
    value_unit: str,
):
    source_space = source_discretization.field_spaces[0]
    target_space = target_discretization.field_spaces[0]
    route_id = (
        f"{source_discretization.prepared_id}-to-"
        f"{target_discretization.prepared_id}-{quantity_id}"
    )
    primal = phx.linalg.IdentityLinearOperator(
        source_space.vector_space,
        operator_id=f"high-order-primal-{route_id}",
    )
    adjoint = phx.linalg.IdentityLinearOperator(
        target_space.vector_space,
        operator_id=f"high-order-adjoint-{route_id}",
    )
    transfer = phx.discretization.FieldTransfer(
        source_space,
        target_space,
        primal,
        adjoint_operator=adjoint,
        properties=phx.discretization.TransferProperties(
            constant_preserving=True,
            conservative=True,
            positivity_preserving=True,
            nested=True,
            adjoint_paired=True,
            exact_on=("point_value",),
        ),
    )
    source_epoch = source_geometry.plan.prepared_epoch
    target_epoch = target_geometry.plan.prepared_epoch
    epoch = anatomy.CardiacTransferEpoch(
        source_epoch.geometry,
        target_epoch.geometry,
        source_epoch.reference,
        target_epoch.reference,
    )
    configuration = anatomy.CardiacTransferConfiguration.for_transfer(
        transfer,
        quantity_id,
        value_unit,
        source_geometry.prepared_id,
        target_geometry.prepared_id,
    )
    cardiac = anatomy.CardiacFieldTransfer(
        transfer,
        configuration,
        epoch,
        constant_tolerance=2.0e-6,
        adjoint_tolerance=2.0e-6,
    )
    return cardiac, epoch


def _apply_transfer(transfer, epoch, values):
    return transfer.apply(
        values,
        epoch,
        configuration_id=transfer.configuration.configuration_id,
    )


def _transfer_monodomain_state(
    source_state,
    target_runtime,
    voltage_transfer,
    voltage_epoch,
    lane_transfers,
    current_transfer,
    current_epoch,
):
    voltage_result = _apply_transfer(
        voltage_transfer, voltage_epoch, source_state.voltage_mV
    )
    lane_results = tuple(
        _apply_transfer(transfer, epoch, source_state.local_states[0][:, index])
        for index, (transfer, epoch) in enumerate(lane_transfers)
    )
    transferred_locals = (
        jnp.stack(tuple(result.value for result in lane_results), axis=1),
    )
    current_result = _apply_transfer(
        current_transfer,
        current_epoch,
        source_state.last_applied_inward_current_uA_per_mm3,
    )
    source_buffer = source_state.checkpoints
    checkpoint_voltage = jnp.stack(
        tuple(
            _apply_transfer(voltage_transfer, voltage_epoch, values).value
            for values in source_buffer.voltage_mV
        )
    )
    checkpoint_lanes = []
    for stored in source_buffer.local_states:
        slots = []
        for slot in range(stored.shape[0]):
            slots.append(
                jnp.stack(
                    tuple(
                        _apply_transfer(transfer, epoch, stored[slot, :, lane]).value
                        for lane, (transfer, epoch) in enumerate(lane_transfers)
                    ),
                    axis=1,
                )
            )
        checkpoint_lanes.append(jnp.stack(tuple(slots)))
    checkpoint_current = jnp.stack(
        tuple(
            _apply_transfer(current_transfer, current_epoch, values).value
            for values in source_buffer.last_applied_inward_current_uA_per_mm3
        )
    )
    target_buffer = electrophysiology.MonodomainCheckpointBuffer(
        checkpoint_voltage,
        tuple(checkpoint_lanes),
        source_buffer.tick,
        source_buffer.macro_step_index,
        checkpoint_current,
        source_buffer.has_previous_stimulus,
        source_buffer.valid,
        source_buffer.write_cursor,
        target_runtime.runtime_id,
    )
    target_state = electrophysiology.PhysicalMonodomainState(
        voltage_result.value,
        transferred_locals,
        source_state.tick,
        source_state.macro_step_index,
        current_result.value,
        source_state.has_previous_stimulus,
        target_buffer,
        target_runtime.runtime_id,
    )
    return target_state, voltage_result, lane_results, current_result


def _mechanics_qualification(target_geometry, target_candidate):
    frame = anatomy.CardiacMaterialFrame(
        jnp.asarray(((1.0, 0.0, 0.0),)),
        jnp.asarray(((0.0, 1.0, 0.0),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((True,)),
        frame_id="qualified-high-order-mechanics-frame",
    )
    energy = mechanics.Guccione1991Energy(
        mechanics.Guccione1991Parameters(0.9, 8.0, 2.0, 4.0),
        frame,
        cell_index=0,
    )
    material = energy.exact_incompressible()
    plan = phx.discretization.MixedFiniteElementConstraintPlan(
        target_geometry.plan.mesh,
        phx.discretization.PressureGaugePolicy("mean-zero"),
        coordinate_spec=target_geometry.plan.coordinate_spec,
        displacement_field="u",
        pressure_field="p",
    )
    qualified = material.prepare_qualified(
        plan,
        form_id="qualified-high-order-cardiac-exact-mixed",
    )
    prepared = qualified.prepared
    evaluation = prepared.evaluate(prepared.problem.state_space.zeros())
    deformation = jnp.asarray(
        ((1.04, 0.02, 0.0), (0.0, 1.0 / 1.04, 0.01), (0.0, 0.0, 1.0))
    )
    material_response = material.evaluate(deformation, jnp.asarray(0.5))
    coordinate_match = bool(
        jnp.allclose(
            prepared.discretization.default_runtime.coordinates,
            target_candidate.coordinates_mm,
        )
    )
    coordinate_identity_match = (
        prepared.discretization.default_runtime.geometry_layout_id
        == target_geometry.plan.coordinate_spec.coordinate_spec_id
    )
    qualification = qualified.qualification
    passed = bool(
        coordinate_match
        and coordinate_identity_match
        and evaluation.valid
        and material_response.evidence.valid
        and qualification.gauge_valid
        and qualification.residual_finite
        and qualification.stable_pair
        and qualification.assembled_inf_sup_stable
        and qualification.locking_safe
        and qualification.valid
    )
    return {
        "route": "exact-q2-q1",
        "coordinate_values_match": coordinate_match,
        "coordinate_identity_matches": coordinate_identity_match,
        "displacement_degree": prepared.spaces.displacement_degree,
        "pressure_degree": prepared.spaces.pressure_degree,
        "pair_names": list(qualification.pair_names),
        "gauge_mode": qualification.gauge_mode,
        "gauge_valid": bool(qualification.gauge_valid),
        "residual_finite": bool(qualification.residual_finite),
        "assembled_inf_sup_constant": float(qualification.inf_sup_constant),
        "assembled_inf_sup_stable": bool(qualification.assembled_inf_sup_stable),
        "locking_safe": bool(qualification.locking_safe),
        "evaluation_valid": bool(evaluation.valid),
        "material_evaluation_valid": bool(material_response.evidence.valid),
        "prepared_id": prepared.prepared_id,
        "passed": passed,
    }


def qualify_case(cell_kind: str) -> dict[str, object]:
    (
        source_geometry,
        source_candidate,
        target_geometry,
        target_candidate,
        stale_candidate,
    ) = _geometry_transition(cell_kind)
    source_discretization = _ep_discretization(source_geometry, f"source-{cell_kind}")
    target_discretization = _ep_discretization(target_geometry, f"target-{cell_kind}")
    source_action, source_operator, conductivity = _tensor_diffusion(
        source_discretization
    )
    target_action, target_operator, _ = _tensor_diffusion(target_discretization)
    node_count = target_discretization.dof_maps[0].global_dof_count
    assignment = _regional_assignment(node_count)
    dtype = target_discretization.field_spaces[0].vector_space.dtype
    integration_plan = electrophysiology.PhysicalMonodomainPlan(
        node_count,
        electrophysiology.EventAlignedMultirateSchedule(
            1.0e-4, 2, 3, event_ticks=(0,), checkpoint_stride=1
        ),
        electrophysiology.LieSplit(),
        electrophysiology.ExplicitReferenceDiffusion(1.0e-3),
        residual_tolerance=2.0e-4,
        checkpoint_capacity=2,
    )
    source_runtime, model = _physical_runtime(
        source_action,
        source_operator,
        assignment,
        _node_volumes(source_candidate, node_count, dtype),
        integration_plan,
        f"source-{cell_kind}",
    )
    target_runtime, _ = _physical_runtime(
        target_action,
        target_operator,
        assignment,
        _node_volumes(target_candidate, node_count, dtype),
        integration_plan,
        f"target-{cell_kind}",
    )
    source_state = electrophysiology.initialize_physical_monodomain(source_runtime)
    source_current = jnp.zeros((2, node_count), dtype=dtype).at[:, 0].set(5.0)
    source_step = electrophysiology.step_physical_monodomain(
        source_runtime,
        source_state,
        electrophysiology.MonodomainMacroInputs(source_current),
    )
    baseline_voltage_transfer, baseline_epoch = _cardiac_transfer(
        source_discretization,
        source_discretization,
        source_geometry,
        source_geometry,
        quantity_id="voltage_mV",
        value_unit="mV",
    )
    baseline_result = _apply_transfer(
        baseline_voltage_transfer, baseline_epoch, source_step.state.voltage_mV
    )
    voltage_transfer, voltage_epoch = _cardiac_transfer(
        source_discretization,
        target_discretization,
        source_geometry,
        target_geometry,
        quantity_id="voltage_mV",
        value_unit="mV",
    )
    lane_names = model.state_layout.state_names[1:]
    lane_units = model.state_layout.state_units[1:]
    lane_transfers = tuple(
        _cardiac_transfer(
            source_discretization,
            target_discretization,
            source_geometry,
            target_geometry,
            quantity_id=f"reaction:{name}",
            value_unit=unit,
        )
        for name, unit in zip(lane_names, lane_units, strict=True)
    )
    current_transfer, current_epoch = _cardiac_transfer(
        source_discretization,
        target_discretization,
        source_geometry,
        target_geometry,
        quantity_id="inward_current",
        value_unit="uA/mm3",
    )
    transferred_state, voltage_result, lane_results, current_result = (
        _transfer_monodomain_state(
            source_step.state,
            target_runtime,
            voltage_transfer,
            voltage_epoch,
            lane_transfers,
            current_transfer,
            current_epoch,
        )
    )
    target_step = electrophysiology.step_physical_monodomain(
        target_runtime,
        transferred_state,
        electrophysiology.MonodomainMacroInputs(source_current),
    )
    voltage_error = float(
        jnp.max(jnp.abs(voltage_result.value - source_step.state.voltage_mV))
    )
    lane_errors = tuple(
        float(jnp.max(jnp.abs(result.value - source_step.state.local_states[0][:, lane])))
        for lane, result in enumerate(lane_results)
    )
    transfer_evidence = (
        voltage_result.evidence,
        current_result.evidence,
        *(result.evidence for result in lane_results),
    )
    transfer_passed = bool(
        baseline_result.evidence.accepted
        and all(bool(evidence.accepted) for evidence in transfer_evidence)
        and all(bool(evidence.coverage_complete) for evidence in transfer_evidence)
        and all(bool(evidence.constant_preserved) for evidence in transfer_evidence)
        and all(bool(evidence.adjoint_consistent) for evidence in transfer_evidence)
        and voltage_error <= 2.0e-6
        and max(lane_errors, default=0.0) <= 2.0e-6
        and len(lane_results) == model.state_layout.state_count - 1
    )
    operator_probe = jnp.linspace(-1.0, 1.0, node_count, dtype=dtype)
    source_action_value = source_operator.mv(operator_probe)
    target_action_value = target_operator.mv(operator_probe)
    operator_change = float(jnp.max(jnp.abs(target_action_value - source_action_value)))
    conductivity_eigenvalues = np.linalg.eigvalsh(np.asarray(conductivity))
    mechanics_result = (
        _mechanics_qualification(target_geometry, target_candidate)
        if cell_kind == "hexahedron"
        else None
    )
    geometry_passed = bool(
        source_candidate.evidence.accepted
        and target_candidate.evidence.accepted
        and not stale_candidate.evidence.accepted
        and stale_candidate.evidence.transfer_required
        and stale_candidate.evidence.rebuild_required
        and np.all(
            np.asarray(target_candidate.evidence.minimum_jacobian_determinants) > 0.0
        )
        and np.all(np.asarray(target_candidate.evidence.minimum_cell_measures_mm3) > 0.0)
        and source_geometry.prepared_id != target_geometry.prepared_id
        and source_geometry.plan.plan_id != target_geometry.plan.plan_id
    )
    ep_passed = bool(
        source_step.evidence.successful
        and target_step.evidence.successful
        and target_step.evidence.reaction_admissible
        and jnp.any(target_step.evidence.diffusion_stage_active)
        and jnp.all(target_step.evidence.reaction_rate_call_count > 0)
        and jnp.all(target_step.evidence.exact_gate_call_count > 0)
        and jnp.max(jnp.abs(target_step.state.voltage_mV - transferred_state.voltage_mV))
        > 0.0
    )
    operator_passed = bool(
        source_operator.operator_id != target_operator.operator_id
        and source_discretization.prepared_id != target_discretization.prepared_id
        and operator_change > 0.0
        and np.min(conductivity_eigenvalues) > 0.0
        and target_operator.properties.self_adjoint is True
        and target_operator.properties.positive_semidefinite is True
    )
    passed = bool(
        geometry_passed
        and transfer_passed
        and ep_passed
        and operator_passed
        and (mechanics_result is None or mechanics_result["passed"])
    )
    return {
        "cell_kind": cell_kind,
        "geometry_family": "P2" if cell_kind == "tetrahedron" else "Q2",
        "node_count": node_count,
        "geometry": {
            "source_prepared_id": source_geometry.prepared_id,
            "target_prepared_id": target_geometry.prepared_id,
            "source_plan_id": source_geometry.plan.plan_id,
            "target_plan_id": target_geometry.plan.plan_id,
            "minimum_jacobian_determinants": np.asarray(
                target_candidate.evidence.minimum_jacobian_determinants
            ).tolist(),
            "minimum_cell_measures_mm3": np.asarray(
                target_candidate.evidence.minimum_cell_measures_mm3
            ).tolist(),
            "stale_epoch_transfer_required": bool(
                stale_candidate.evidence.transfer_required
            ),
            "stale_epoch_rebuild_required": bool(
                stale_candidate.evidence.rebuild_required
            ),
            "stale_epoch_rejected": not bool(stale_candidate.evidence.accepted),
            "passed": geometry_passed,
        },
        "operator": {
            "source_discretization_id": source_discretization.prepared_id,
            "target_discretization_id": target_discretization.prepared_id,
            "source_operator_id": source_operator.operator_id,
            "target_operator_id": target_operator.operator_id,
            "maximum_rebuild_action_change": operator_change,
            "minimum_conductivity_eigenvalue": float(np.min(conductivity_eigenvalues)),
            "passed": operator_passed,
        },
        "transfer": {
            "baseline_transfer_id": baseline_voltage_transfer.cardiac_transfer_id,
            "rebuilt_transfer_id": voltage_transfer.cardiac_transfer_id,
            "transfer_identity_rebuilt": baseline_voltage_transfer.cardiac_transfer_id
            != voltage_transfer.cardiac_transfer_id,
            "voltage_error": voltage_error,
            "reaction_lane_count": len(lane_results),
            "expected_reaction_lane_count": model.state_layout.state_count - 1,
            "maximum_reaction_lane_error": max(lane_errors, default=0.0),
            "source_coverage_fraction": float(
                voltage_result.evidence.source_coverage_fraction
            ),
            "target_coverage_fraction": float(
                voltage_result.evidence.target_coverage_fraction
            ),
            "constant_error": float(voltage_result.evidence.constant_error),
            "adjoint_error": float(voltage_result.evidence.adjoint_error),
            "coverage_complete": bool(voltage_result.evidence.coverage_complete),
            "constant_preserved": bool(voltage_result.evidence.constant_preserved),
            "adjoint_consistent": bool(voltage_result.evidence.adjoint_consistent),
            "all_lane_evidence_accepted": all(
                bool(result.evidence.accepted) for result in lane_results
            ),
            "passed": transfer_passed,
        },
        "electrophysiology": {
            "tensor_action_id": target_action.action_id,
            "tensor_input_id": target_runtime.spatial.diffusion.input_id,
            "source_step_accepted": bool(source_step.evidence.successful),
            "target_step_accepted": bool(target_step.evidence.successful),
            "reaction_admissible": bool(target_step.evidence.reaction_admissible),
            "diffusion_stage_count": int(target_step.evidence.diffusion_stage_count),
            "reaction_tick_count": int(target_step.evidence.reaction_tick_count),
            "maximum_voltage_change_mV": float(
                jnp.max(
                    jnp.abs(target_step.state.voltage_mV - transferred_state.voltage_mV)
                )
            ),
            "passed": ep_passed,
        },
        "mechanics": mechanics_result,
        "passed": passed,
    }


def qualification() -> dict[str, object]:
    cases = tuple(qualify_case(kind) for kind in ("tetrahedron", "hexahedron"))
    mechanics_cases = tuple(case for case in cases if case["mechanics"] is not None)
    return {
        "cases": cases,
        "qualified_cell_kinds": [case["cell_kind"] for case in cases],
        "mechanics_case_count": len(mechanics_cases),
        "passed": all(bool(case["passed"]) for case in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify P2/Q2 cardiovascular geometry, transfer, EP, and mechanics."
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = qualification()
    if not bool(payload["passed"]):
        raise RuntimeError("Cardiovascular high-order qualification failed.")
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
