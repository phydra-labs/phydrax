#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


ARTIFACTS = (
    "schedules",
    "adaptive",
    "plane_stress",
    "plasticity",
    "gimp",
    "cpdi",
    "contact",
    "multifield",
    "active_blocks",
    "implicit",
    "fracture",
    "sharp_fracture",
    "sparse",
)


def _write(directory: Path, name: str, metrics: dict, passed: bool):
    payload = {
        "maturity": "experimental",
        "capability": name,
        "metrics": metrics,
        "passed": bool(passed),
    }
    path = directory / f"material_point_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)
    return payload


def _compile(
    material,
    parameters,
    *,
    dimension=2,
    assignment=None,
    schedule=None,
    nodal_fields=None,
    active_blocks=None,
):
    grid_points = 10 if dimension == 2 else 6
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(grid_points, periodic=True, endpoint=False)
            for _ in range(dimension)
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))))
    base = jnp.asarray(
        [[0.28, 0.31, 0.35], [0.42, 0.36, 0.44], [0.34, 0.49, 0.53], [0.48, 0.52, 0.61]]
    )[:, :dimension]
    volume = jnp.full((4,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), volume, ambient_dimension=dimension
    ).prepare()
    assignment_ = (
        phx.discretization.TensorBSplineSplatAssignment(2)
        if assignment is None
        else assignment
    )
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=assignment_
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR("advanced-qualification", material),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(schedule=schedule),
        phx.discretization.MPMParticleDomainPlan(
            jnp.stack((jnp.zeros((dimension,)), jnp.ones((dimension,)))),
            periodic=(True,) * dimension,
            support_margin=0.0,
        ),
        nodal_fields=nodal_fields,
        active_blocks=active_blocks,
    )
    arguments = phx.equations.MaterialPointArguments(parameters)
    velocity = jnp.broadcast_to(jnp.asarray((0.04, -0.015, 0.02))[:dimension], base.shape)
    state = compiled.initialize_state(base, velocity, volume, arguments)
    return compiled, arguments, state


def _schedule_metrics():
    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        2.0, 8.0
    )
    values = {}
    passed = True
    for schedule in (
        phx.discretization.USLMPMSchedule(),
        phx.discretization.USFMPMSchedule(),
        phx.discretization.MUSLMPMSchedule(),
    ):
        compiled, arguments, state = _compile(material, parameters, schedule=schedule)
        detail = compiled.dynamics.step_detailed(state, 0.001, arguments)
        name = schedule.common_name
        values[name] = {
            "successful": bool(detail.successful),
            "mass_defect": float(detail.diagnostics.transfer.relative_mass_defect),
            "momentum_defect": float(
                detail.diagnostics.transfer.relative_momentum_defect
            ),
            "energy_defect": float(detail.diagnostics.energy.balance_defect),
            "second_mass_defect": float(
                detail.diagnostics.schedule.second_transfer_mass_defect
            ),
            "second_momentum_defect": float(
                detail.diagnostics.schedule.second_transfer_momentum_defect
            ),
        }
        passed &= bool(detail.successful)
        passed &= values[name]["mass_defect"] < 1e-10
        passed &= values[name]["momentum_defect"] < 1e-9
    return values, passed


def _adaptive_metrics():
    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        2.0, 8.0
    )
    compiled, arguments, state = _compile(material, parameters)
    plan = phx.solver.AdaptiveMPMRolloutPlan(
        compiled.dynamics,
        phx.solver.MPMAdaptivePolicy(16, maximum_retries=8),
        final_time=0.01,
        initial_step_size=1.0,
    )
    result = plan.rollout(state, arguments)
    replay = phx.solver.ScheduledMPMRolloutPlan.from_realized(
        compiled.dynamics, result.realized_mesh
    ).rollout(state, arguments)
    parity = float(
        jnp.max(
            jnp.abs(
                replay.final_state.particles.position
                - result.final_state.particles.position
            )
        )
    )
    metrics = {
        "completed": bool(result.completed),
        "attempts": int(result.journal.attempt_count),
        "accepted_steps": int(result.journal.accepted_count),
        "replay_parity": parity,
    }
    return metrics, metrics["completed"] and parity < 1e-10


def _material_metrics():
    neo = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    plane = phx.applications.solid_mechanics.IsotropicPlaneStressMPMConstitutivePlan(neo)
    neo_parameters = (
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(3.0, 11.0)
    )
    deformation = jnp.asarray([[[1.12, 0.06], [0.02, 0.93]]])
    history = plane.initialize_state((1,), jnp.float64)
    response = plane.evaluate(
        deformation, history, jnp.asarray((2.0,)), neo_parameters, 0.0, 0.01
    )
    linearized = plane.evaluate_linearized(
        deformation, history, jnp.asarray((2.0,)), neo_parameters, 0.0, 0.01
    )
    plane_metrics = {
        "successful": bool(response.successful[0]),
        "p33_residual": abs(float(response.diagnostics["plane_stress_residual"][0])),
        "out_of_plane_stretch": float(response.diagnostics["out_of_plane_stretch"][0]),
        "tangent_finite": bool(linearized.tangent_successful[0]),
    }
    plastic = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    plastic_parameters = phx.applications.solid_mechanics.FiniteStrainJ2Parameters(
        10.0, 30.0, 0.15, 2.0
    )
    plastic_history = plastic.initialize_state((1,), jnp.float64)
    plastic_response = plastic.evaluate(
        jnp.asarray([[[1.0, 0.18, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
        plastic_history,
        jnp.asarray((1.0,)),
        plastic_parameters,
        0.0,
        0.01,
    )
    plastic_metrics = {
        "successful": bool(plastic_response.successful[0]),
        "branch": int(plastic_response.branch_code[0]),
        "plastic_multiplier": float(
            plastic_response.diagnostics["plastic_multiplier"][0]
        ),
        "plastic_determinant": float(
            plastic_response.diagnostics["plastic_determinant"][0]
        ),
        "dissipation": float(plastic_response.dissipation_increment[0]),
    }
    plane_passed = (
        plane_metrics["successful"]
        and plane_metrics["p33_residual"] < 1e-8
        and plane_metrics["tangent_finite"]
    )
    plastic_passed = (
        plastic_metrics["successful"]
        and plastic_metrics["branch"] == 1
        and plastic_metrics["plastic_multiplier"] > 0.0
        and abs(plastic_metrics["plastic_determinant"] - 1.0) < 1e-8
        and plastic_metrics["dissipation"] >= 0.0
    )
    return (plane_metrics, plane_passed), (plastic_metrics, plastic_passed)


def _domain_metrics():
    widths = jnp.full((4, 2), 0.025)
    gimp = phx.discretization.UniformGIMPSplatAssignment(
        widths, maximum_half_width_cells=0.75
    )
    neo = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        2.0, 8.0
    )
    compiled, arguments, state = _compile(neo, parameters, assignment=gimp)
    route = compiled.dynamics.splat.build(state.particles.position)
    gimp_metrics = {
        "partition_defect": float(jnp.max(jnp.abs(route.partition_sums - 1.0))),
        "gradient_defect": float(jnp.max(jnp.abs(route.gradient_sums))),
        "first_moment_defect": float(jnp.max(jnp.abs(route.first_moments))),
        "successful": bool(route.successful),
    }
    edges = jnp.broadcast_to(0.025 * jnp.eye(2), (4, 2, 2))
    cpdi = phx.discretization.AffineCPDISplatAssignment(edges)
    compiled_cpdi, _, state_cpdi = _compile(neo, parameters, assignment=cpdi)
    cpdi_route = compiled_cpdi.dynamics.splat.build(
        state_cpdi.particles.position,
        assignment_input=state_cpdi.assignment_input,
    )
    cpdi_metrics = {
        "partition_defect": float(jnp.max(jnp.abs(cpdi_route.partition_sums - 1.0))),
        "gradient_defect": float(jnp.max(jnp.abs(cpdi_route.gradient_sums))),
        "first_moment_defect": float(jnp.max(jnp.abs(cpdi_route.first_moments))),
        "successful": bool(cpdi_route.successful),
    }
    return (
        gimp_metrics,
        gimp_metrics["successful"]
        and max(
            gimp_metrics["partition_defect"],
            gimp_metrics["gradient_defect"],
            gimp_metrics["first_moment_defect"],
        )
        < 1e-9,
    ), (
        cpdi_metrics,
        cpdi_metrics["successful"]
        and max(
            cpdi_metrics["partition_defect"],
            cpdi_metrics["gradient_defect"],
            cpdi_metrics["first_moment_defect"],
        )
        < 1e-8,
    )


def _contact_field_metrics():
    geometry = phx.geometry.Circle((0.0, 0.0), 0.5).compile()
    contact = phx.discretization.RigidMPMContactPlan(
        geometry,
        phx.discretization.SharpCoulombMPMFrictionPlan(0.25),
        contact_band=0.02,
    )
    result = contact.apply(
        jnp.asarray([[0.49, 0.0]]),
        jnp.asarray([[-1.0, 0.6]]),
        jnp.asarray((2.0,)),
        0.0,
        0.01,
    )
    contact_metrics = {
        "successful": bool(result.successful),
        "post_normal_velocity": float(result.velocity[0, 0]),
        "dissipation": float(result.dissipation),
        "work": float(result.work),
    }
    field = phx.discretization.project_two_field_contact(
        jnp.asarray([[1.0], [2.0]]),
        jnp.asarray([[[1.0, 0.4]], [[-0.5, 0.0]]]),
        jnp.asarray([[[1.0, 0.0]], [[-1.0, 0.0]]]),
        friction=phx.discretization.SharpCoulombMPMFrictionPlan(0.3),
    )
    field_metrics = {
        "successful": bool(field.successful),
        "contact": bool(field.contact_mask[0]),
        "action_reaction_defect": float(field.action_reaction_defect),
        "dissipation": float(field.dissipation),
    }
    return (
        contact_metrics,
        contact_metrics["successful"]
        and contact_metrics["post_normal_velocity"] >= -1e-10
        and contact_metrics["dissipation"] >= 0.0,
    ), (
        field_metrics,
        field_metrics["successful"]
        and field_metrics["contact"]
        and field_metrics["action_reaction_defect"] < 1e-12,
    )


def _storage_metrics():
    neo = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        2.0, 8.0
    )
    compiled, _, state = _compile(neo, parameters)
    routes = compiled.dynamics.splat.build(state.particles.position)
    blocks = phx.discretization.MPMActiveBlockPlan((10, 10), (5, 5), 4)
    active = blocks.build(routes)
    storage = phx.discretization.BlockSparseMPMNodalStoragePlan(blocks)
    dense = jnp.arange(10 * 10 * 2, dtype=jnp.float64).reshape((10, 10, 2))
    compact = storage.pack(dense, active)
    restored = storage.unpack(compact, active)
    parity = float(
        jnp.max(
            jnp.abs(jnp.where(active.active_node_mask[..., None], restored - dense, 0.0))
        )
    )
    active_metrics = {
        "active_blocks": int(active.active_block_count),
        "overflow": bool(active.overflow),
        "active_nodes": int(jnp.sum(active.active_node_mask)),
    }
    sparse_metrics = {
        "dense_sparse_parity": parity,
        "compact_values": int(compact.size),
        "dense_values": int(dense.size),
    }
    return (
        active_metrics,
        not active_metrics["overflow"] and active_metrics["active_blocks"] > 0,
    ), (sparse_metrics, parity == 0.0)


def _implicit_metrics():
    material = phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2)
    parameters = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(
        2.0, 8.0
    )
    compiled, arguments, state = _compile(material, parameters)
    result = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).step_detailed(
        state, 0.001, arguments
    )
    metrics = {
        "successful": bool(result.successful),
        "residual_norm": float(result.diagnostics.residual_norm),
        "nonlinear_steps": int(result.diagnostics.nonlinear_steps),
        "linear_iterations": int(result.diagnostics.linear_iterations),
        "tangent_successful": bool(result.diagnostics.tangent_successful),
    }
    return metrics, metrics["successful"] and metrics["residual_norm"] < 1e-8


def _fracture_metrics():
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(2)
    parameters = phx.applications.solid_mechanics.MPMPhaseFieldParameters(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0),
        1.0,
        0.1,
    )
    compiled, arguments, mechanics = _compile(material, parameters)
    prepared = phx.solver.PreparedMPMPhaseFieldDynamics(
        compiled.dynamics,
        phx.solver.MPMPhaseFieldFracturePlan(
            maximum_damage_iterations=200, tolerance=1e-6
        ),
    )
    result = prepared.step_detailed(
        prepared.initialize_state(mechanics), 0.001, arguments
    )
    metrics = {
        "successful": bool(result.successful),
        "damage_residual": float(result.evidence.damage_residual_norm),
        "minimum_damage_increment": float(result.evidence.minimum_damage_increment),
        "fracture_energy": float(result.evidence.fracture_energy),
    }
    partition = phx.discretization.MPMFieldPartitionFracturePlan(2).update(
        jnp.asarray((0.2, 0.99)),
        jnp.asarray((-1.0, 1.0)),
        jnp.zeros((2,), dtype=jnp.int32),
        0,
    )
    sharp_metrics = {
        "successful": bool(partition.successful),
        "topology_generation": int(partition.topology_generation),
        "field_count": int(jnp.max(partition.velocity_field_slots)) + 1,
    }
    return (
        metrics,
        metrics["successful"]
        and metrics["minimum_damage_increment"] >= -1e-12
        and metrics["fracture_energy"] >= 0.0,
    ), (
        sharp_metrics,
        sharp_metrics["successful"] and sharp_metrics["topology_generation"] == 1,
    )


def run(output: Path):
    results = {}
    schedule, ok = _schedule_metrics()
    results["schedules"] = _write(output, "schedules", schedule, ok)
    adaptive, ok = _adaptive_metrics()
    results["adaptive"] = _write(output, "adaptive", adaptive, ok)
    (plane, plane_ok), (plastic, plastic_ok) = _material_metrics()
    results["plane_stress"] = _write(output, "plane_stress", plane, plane_ok)
    results["plasticity"] = _write(output, "plasticity", plastic, plastic_ok)
    (gimp, gimp_ok), (cpdi, cpdi_ok) = _domain_metrics()
    results["gimp"] = _write(output, "gimp", gimp, gimp_ok)
    results["cpdi"] = _write(output, "cpdi", cpdi, cpdi_ok)
    (contact, contact_ok), (fields, fields_ok) = _contact_field_metrics()
    results["contact"] = _write(output, "contact", contact, contact_ok)
    results["multifield"] = _write(output, "multifield", fields, fields_ok)
    (blocks, blocks_ok), (sparse, sparse_ok) = _storage_metrics()
    results["active_blocks"] = _write(output, "active_blocks", blocks, blocks_ok)
    results["sparse"] = _write(output, "sparse", sparse, sparse_ok)
    implicit, implicit_ok = _implicit_metrics()
    results["implicit"] = _write(output, "implicit", implicit, implicit_ok)
    (fracture, fracture_ok), (sharp, sharp_ok) = _fracture_metrics()
    results["fracture"] = _write(output, "fracture", fracture, fracture_ok)
    results["sharp_fracture"] = _write(output, "sharp_fracture", sharp, sharp_ok)
    passed = all(payload["passed"] for payload in results.values())
    print(json.dumps({"artifacts": results, "passed": passed}, indent=2))
    if not passed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Qualify advanced MPM capabilities.")
    parser.add_argument("--output", type=Path, default=Path("benchmarks"))
    arguments = parser.parse_args()
    run(arguments.output)


if __name__ == "__main__":
    main()
