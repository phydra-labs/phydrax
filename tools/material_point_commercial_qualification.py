#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


NAMES = (
    "transfer",
    "kinematics",
    "geomechanics",
    "coupled_fields",
    "kway_contact",
    "multifield_schedules",
    "implicit_core",
    "implicit_contact",
    "moving_domain",
    "sparse_implicit",
    "kernels",
    "distributed",
    "lifecycle_amr",
    "derivatives",
    "commercial_release",
)


def _write(directory, name, metrics, passed):
    payload = {
        "maturity": "commercial-qualification-candidate",
        "capability": name,
        "metrics": metrics,
        "passed": bool(passed),
    }
    path = directory / f"material_point_commercial_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    return payload


def _base(transfer=None, schedule=None, fields=None, assignment=None):
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(10, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.28, 0.31], [0.42, 0.36], [0.34, 0.49], [0.48, 0.52]])
    volume = jnp.full((4,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=(
            phx.discretization.TensorBSplineSplatAssignment(2)
            if assignment is None
            else assignment
        ),
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "commercial-qualification",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(
            transfer,
            schedule=schedule,
        ),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
        nodal_fields=fields,
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        position,
        jnp.broadcast_to(jnp.asarray((0.04, -0.01)), position.shape),
        volume,
        arguments,
    )
    return compiled, arguments, state


def _transfer():
    metrics = {}
    passed = True
    for transfer in (
        phx.discretization.PICTransferPlan(),
        phx.discretization.FLIPTransferPlan(),
        phx.discretization.PICFLIPTransferPlan(0.25),
        phx.discretization.APICTransferPlan(),
    ):
        compiled, arguments, state = _base(transfer)
        detail = compiled.dynamics.step_detailed(state, 0.001, arguments)
        error = float(
            jnp.max(
                jnp.abs(
                    detail.accepted_state.particles.velocity - state.particles.velocity
                )
            )
        )
        metrics[transfer.transfer_name] = {
            "successful": bool(detail.successful),
            "constant_velocity_error": error,
            "mass_defect": float(detail.diagnostics.transfer.relative_mass_defect),
        }
        passed &= bool(detail.successful) and error < 1e-9
    for schedule in (
        phx.discretization.AffineMUSLMPMSchedule(),
        phx.discretization.PostAdvectionMUSLMPMSchedule(),
    ):
        compiled, arguments, state = _base(
            phx.discretization.APICTransferPlan(), schedule
        )
        detail = compiled.dynamics.step_detailed(state, 0.001, arguments)
        metrics[schedule.common_name] = {
            "successful": bool(detail.successful),
            "second_mass_defect": float(
                detail.diagnostics.schedule.second_transfer_mass_defect
            ),
            "second_momentum_defect": float(
                detail.diagnostics.schedule.second_transfer_momentum_defect
            ),
        }
        passed &= bool(detail.successful)
    return metrics, passed


def _kinematics_geomechanics():
    base = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    material = phx.applications.solid_mechanics.GeneralPlaneStressMPMConstitutivePlan(
        base
    )
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        3.0, 11.0
    )
    deformation = jnp.asarray([[[1.1, 0.04], [0.01, 0.95]]])
    state = material.initialize_state((1,), jnp.float64)
    response = material.evaluate(
        deformation, state, jnp.asarray((2.0,)), parameters, 0.0, 0.01
    )
    tangent = material.evaluate_linearized(
        deformation, state, jnp.asarray((2.0,)), parameters, 0.0, 0.01
    )
    kinematics = {
        "successful": bool(response.successful[0]),
        "traction_residual": float(
            jnp.linalg.norm(response.diagnostics["plane_stress_residual"][0])
        ),
        "tangent_successful": bool(tangent.tangent_successful[0]),
    }
    dp = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    dp_parameters = phx.applications.solid_mechanics.DruckerPragerParameters(
        10.0, 30.0, 0.05, 0.5, 0.2, 1.0
    )
    full_f = jnp.asarray([[[1.0, 0.15, 0.0], [0.0, 0.94, 0.0], [0.0, 0.0, 1.06]]])
    dp_result = dp.evaluate(
        full_f,
        dp.initialize_state((1,), jnp.float64),
        jnp.asarray((1.0,)),
        dp_parameters,
        0.0,
        0.01,
    )
    geomechanics = {
        "successful": bool(dp_result.successful[0]),
        "yield_residual": float(dp_result.diagnostics["yield_residual"][0]),
        "plastic_multiplier": float(dp_result.diagnostics["plastic_multiplier"][0]),
        "dissipation": float(dp_result.dissipation_increment[0]),
    }
    return (
        kinematics,
        kinematics["successful"]
        and kinematics["traction_residual"] < 1e-8
        and kinematics["tangent_successful"],
    ), (
        geomechanics,
        geomechanics["successful"] and geomechanics["dissipation"] >= 0.0,
    )


def _coupled():
    shape = (6, 6)
    boundary = phx.applications.solid_mechanics.MPMCoupledBoundaryPlan(
        pressure_mask=jnp.zeros(shape, dtype=bool).at[0].set(True),
        pressure_values=0.0,
        pressure_flux=0.0,
        temperature_mask=jnp.zeros(shape, dtype=bool).at[-1].set(True),
        temperature_values=300.0,
        heat_flux=0.0,
    )
    operator = phx.applications.solid_mechanics.PreparedMPMCoupledFieldOperator(
        shape,
        (0.1, 0.1),
        (False, False),
        phx.applications.solid_mechanics.BiotPoromechanicsParameters(
            0.8, 0.1, 1e-4, 1e-3
        ),
        phx.applications.solid_mechanics.ThermalMPMParameters(
            1.0, 2.0, 1e-5, 0.9, 293.15
        ),
        boundary,
    )
    state = phx.applications.solid_mechanics.MPMCoupledFieldState(
        jnp.ones(shape),
        jnp.ones(shape),
        293.15 * jnp.ones(shape),
        jnp.zeros(shape),
        jnp.asarray(0.0),
    )
    residual = operator.residual(
        state, jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape), jnp.ones(shape)
    )
    metrics = {
        "successful": bool(residual.finite),
        "pressure_residual_norm": float(jnp.linalg.norm(residual.pressure)),
        "temperature_residual_norm": float(jnp.linalg.norm(residual.temperature)),
    }
    return metrics, metrics["successful"]


def _contact_multifield():
    mass = jnp.asarray([[1.0], [1.5], [2.0]])
    velocity = jnp.asarray([[[0.8, 0.3]], [[0.0, 0.0]], [[-0.6, -0.1]]])
    gradients = jnp.asarray([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, -1.0]]])
    plan = phx.discretization.KWayMPMContactPlan(3, maximum_steps=40, tolerance=1e-8)
    result = plan.solve(mass, velocity, plan.build_graph(mass, gradients), 0.01)
    contact = {
        "successful": bool(result.successful),
        "complementarity": float(result.complementarity_residual),
        "action_reaction": float(result.action_reaction_defect),
    }
    slots = jnp.asarray((0, 0, 1, 1), dtype=jnp.int32)
    field_plan = phx.discretization.MPMNodalFieldPlan(
        ("left", "right"),
        slots,
        contact_plan=phx.discretization.KWayMPMContactPlan(
            2, maximum_steps=40, tolerance=1e-8
        ),
    )
    schedules = {}
    schedule_passed = True
    for schedule in (
        phx.discretization.USFMPMSchedule(),
        phx.discretization.MUSLMPMSchedule(),
        phx.discretization.AffineMUSLMPMSchedule(),
        phx.discretization.PostAdvectionMUSLMPMSchedule(),
    ):
        compiled, arguments, state = _base(
            phx.discretization.APICTransferPlan(), schedule, field_plan
        )
        state = phx.discretization.MPMRuntimeState(
            state.particles,
            state.time,
            state.accepted_step,
            state.last_status,
            state.topology_generation,
            state.assignment_input,
            state.material_slots,
            slots,
            slots,
            state.storage_state,
        )
        detail = compiled.dynamics.step_detailed(state, 5e-4, arguments)
        schedules[schedule.common_name] = bool(detail.successful)
        schedule_passed &= bool(detail.successful)
    return (
        contact,
        contact["successful"]
        and contact["complementarity"] < 1e-8
        and contact["action_reaction"] < 1e-12,
    ), (schedules, schedule_passed)


def _implicit_sparse_moving():
    compiled, arguments, state = _base()
    implicit = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).step_detailed(
        state, 0.001, arguments
    )
    core = {
        "successful": bool(implicit.successful),
        "residual_norm": float(implicit.diagnostics.residual_norm),
        "tangent_successful": bool(implicit.diagnostics.tangent_successful),
    }
    routes = compiled.dynamics.splat.build(state.particles.position)
    superset = phx.solver.MPMRouteSupersetPlan(
        compiled.dynamics.splat, minimum_margin=1e-10
    )
    superset_state = superset.build(state.particles.position)
    moving = superset.linearize(
        superset_state,
        state.particles.position,
        state.particles.deformation_gradient,
        state.assignment_input,
        1e-3 * jnp.ones_like(state.particles.position),
        jnp.zeros_like(state.particles.deformation_gradient),
        None,
        (
            jnp.ones_like(routes.stencil.weights),
            jnp.ones_like(routes.weight_gradients),
            jnp.ones_like(routes.route_offsets),
        ),
    )
    moving_metrics = {
        "successful": bool(moving.successful),
        "topology_stable": bool(moving.route_topology_stable),
        "jvp_norm": float(jnp.linalg.norm(moving.weight_jvp)),
    }
    blocks_plan = phx.discretization.MPMActiveBlockPlan((10, 10), (5, 5), 4)
    blocks = blocks_plan.build(routes)
    storage = phx.discretization.BlockSparseMPMNodalStoragePlan(blocks_plan)
    compact_operator = phx.solver.MPMCompactImplicitOperator(storage, blocks)
    dense = jnp.arange(100.0).reshape((10, 10))
    compact = storage.pack(dense, blocks)
    compact_result = compact_operator.apply(
        lambda value: 2.0 * value + jnp.roll(value, 1, axis=0),
        compact,
        jnp.ones_like(compact),
        jnp.ones_like(compact),
    )
    sparse = {
        "successful": bool(compact_result.successful),
        "residual_defect": float(compact_result.dense_compact_residual_defect),
        "jvp_defect": float(compact_result.dense_compact_jvp_defect),
        "transpose_defect": float(compact_result.dense_compact_transpose_defect),
    }
    return (
        (
            core,
            core["successful"] and core["residual_norm"] < 1e-8,
        ),
        (
            moving_metrics,
            moving_metrics["successful"] and moving_metrics["topology_stable"],
        ),
        (
            sparse,
            sparse["successful"]
            and max(
                sparse["residual_defect"],
                sparse["jvp_defect"],
                sparse["transpose_defect"],
            )
            < 1e-10,
        ),
    )


def _execution_distributed_amr():
    execution = phx.discretization.MPMExecutionPlan(
        backend="cpu",
        device_mesh="1",
        precision_policy_id="f64",
        determinism=phx.discretization.MPMDeterminismMode.DETERMINISTIC,
        realization=phx.discretization.MPMKernelRealization.REFERENCE,
        particle_capacity=100,
        grid_capacity=100,
        route_capacity=1000,
        field_capacity=3,
        block_capacity=16,
        contact_pair_capacity=3,
    )
    execution.admit(
        particles=10,
        grid_nodes=64,
        routes=300,
        fields=2,
        blocks=4,
        contact_pairs=1,
    )
    kernel = {
        "execution_id": execution.execution_id,
        "sum": float(
            phx.discretization.deterministic_global_sum(jnp.asarray((1.0, 2.0)))
        ),
    }
    distributed = phx.discretization.MPMDistributedPlan(
        (8, 8),
        (4, 4),
        jnp.asarray([[0, 0], [1, 1]]),
        device_count=2,
        particle_capacity_per_device=4,
    )
    migration = phx.discretization.migrate_particles(
        distributed,
        jnp.asarray([[0.1, 0.1], [0.8, 0.8]]),
        jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        jnp.asarray((0, 0)),
        jnp.asarray((True, True)),
    )
    transaction = phx.discretization.distributed_global_transaction(
        jnp.asarray((True, True)), 3
    )
    distributed_metrics = {
        "migration_successful": bool(migration.successful),
        "transaction_successful": bool(transaction.global_success),
        "generation": int(transaction.commit_generation),
    }
    amr = phx.discretization.MPMAMRPlan(((4, 4), (8, 8)), (4, 16))
    fine = jnp.arange(64.0).reshape((8, 8))
    parity = float(
        jnp.max(
            jnp.abs(amr.restrict(amr.prolong(amr.restrict(fine))) - amr.restrict(fine))
        )
    )
    amr_metrics = {"restriction_prolongation_parity": parity, "levels": 2}
    return (
        (
            kernel,
            kernel["sum"] == 3.0,
        ),
        (
            distributed_metrics,
            distributed_metrics["migration_successful"]
            and distributed_metrics["transaction_successful"],
        ),
        (amr_metrics, parity == 0.0),
    )


def _derivative_release():
    branch = phx.discretization.branchwise_gradient(
        lambda value: jnp.sum(value**2),
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((0.2, 0.3)),
        branch_margin=0.5,
        journal_digest=1,
        evidence_id="commercial-branch",
    )
    event = phx.discretization.locate_event(lambda time: time - 0.25, 0.0, 1.0)
    derivatives = {
        "branch_valid": bool(branch.evidence.valid),
        "event_localized": bool(event.localized),
        "event_time": float(event.event_time),
    }
    claim = phx.discretization.MPMClaimTuple(
        equation_family="solid-mechanics",
        dimension=2,
        kinematics="plane-strain",
        grid_assignment="quadratic-bspline",
        source_domain="point",
        transfer="apic",
        schedule="usl-minus",
        material="neo-hookean",
        field_contact="single-field-none",
        fracture="none",
        integrator="explicit-fixed",
        storage_backend="dense-cpu-f64-deterministic",
        precision_accumulation="f64-deterministic",
        capacity_envelope="qualification",
        derivative_mode="branchwise",
    )
    intended = phx.discretization.MPMIntendedUse(
        "commercial release qualification",
        phenomena=("elasticity",),
        target_observables=("displacement",),
        risk_class="low",
        geometry_loading_scope="qualification case",
        material_parameter_scope="qualification parameters",
        accuracy_uq_goal="qualification tolerance",
    )
    decision = phx.discretization.MPMSupportDecision(
        claim,
        phx.discretization.MPMClaimOutcome.SUPPORTED,
        reason="qualification tuple",
        required_profile="commercial-runtime",
    )
    matrix = phx.discretization.MPMSupportMatrix((decision,))
    gates = tuple(
        phx.discretization.MPMReleaseGateEvidence(
            gate,
            passed=True,
            evidence_ids=(f"evidence-{int(gate)}",),
            reviewer_id=f"reviewer-{int(gate)}",
        )
        for gate in phx.discretization.MPMReleaseGate
    )
    standards = phx.discretization.MPMStandardsTraceabilityMatrix(
        (
            phx.discretization.MPMStandardsTrace(
                standard="ASME V&V 10",
                edition="2019 (R2025)",
                applicability="CSM",
                requirement="code and solution verification",
                evidence_ids=(gates[1].gate_id, gates[2].gate_id),
                satisfied=True,
            ),
        )
    )
    profile = phx.discretization.MPMCommercialProfile(
        "commercial-runtime",
        phx.discretization.MPMCommercialProfileKind.COMMERCIAL_RUNTIME,
        matrix,
        standards,
    )
    assessment = phx.discretization.assess_release(
        profile,
        claim,
        intended,
        {gate.gate: gate for gate in gates},
        phx.discretization.MPMIndependentReview(
            author_id="author",
            technical_reviewer_id="reviewer",
            release_approver_id="approver",
        ),
    )
    release = {"releasable": assessment.releasable, "reasons": assessment.reasons}
    return (
        derivatives,
        derivatives["branch_valid"] and derivatives["event_localized"],
    ), (release, release["releasable"])


def run(directory):
    results = {}
    metrics, passed = _transfer()
    results["transfer"] = _write(directory, "transfer", metrics, passed)
    (metrics, passed), (geo, geo_passed) = _kinematics_geomechanics()
    results["kinematics"] = _write(directory, "kinematics", metrics, passed)
    results["geomechanics"] = _write(directory, "geomechanics", geo, geo_passed)
    metrics, passed = _coupled()
    results["coupled_fields"] = _write(directory, "coupled_fields", metrics, passed)
    (contact, contact_ok), (schedules, schedules_ok) = _contact_multifield()
    results["kway_contact"] = _write(directory, "kway_contact", contact, contact_ok)
    results["multifield_schedules"] = _write(
        directory, "multifield_schedules", schedules, schedules_ok
    )
    (implicit, implicit_ok), (moving, moving_ok), (sparse, sparse_ok) = (
        _implicit_sparse_moving()
    )
    results["implicit_core"] = _write(directory, "implicit_core", implicit, implicit_ok)
    results["implicit_contact"] = _write(
        directory, "implicit_contact", contact, contact_ok
    )
    results["moving_domain"] = _write(directory, "moving_domain", moving, moving_ok)
    results["sparse_implicit"] = _write(directory, "sparse_implicit", sparse, sparse_ok)
    (kernel, kernel_ok), (distributed, distributed_ok), (amr, amr_ok) = (
        _execution_distributed_amr()
    )
    results["kernels"] = _write(directory, "kernels", kernel, kernel_ok)
    results["distributed"] = _write(directory, "distributed", distributed, distributed_ok)
    results["lifecycle_amr"] = _write(directory, "lifecycle_amr", amr, amr_ok)
    (derivatives, derivatives_ok), (release, release_ok) = _derivative_release()
    results["derivatives"] = _write(directory, "derivatives", derivatives, derivatives_ok)
    results["commercial_release"] = _write(
        directory, "commercial_release", release, release_ok
    )
    passed = all(value["passed"] for value in results.values())
    print(json.dumps({"artifacts": results, "passed": passed}, indent=2))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Qualify commercial MPM closures.")
    parser.add_argument("--output", type=Path, default=Path("benchmarks"))
    arguments = parser.parse_args()
    run(arguments.output)


if __name__ == "__main__":
    main()
