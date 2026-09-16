#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint


def _square_mesh():
    return phx.discretization.CellMesh.from_triangles(
        jnp.asarray(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            dtype=jnp.float64,
        ),
        jnp.asarray(((0, 1, 3), (1, 2, 3)), dtype=jnp.int32),
    )


def _multiblock_mesh():
    coordinates = jnp.asarray(
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0),
        ),
        dtype=jnp.float64,
    )
    return phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "left",
                "triangle",
                jnp.asarray(((0, 1, 3), (1, 4, 3)), dtype=jnp.int32),
                global_ids=jnp.asarray((0, 1)),
            ),
            phx.discretization.CellBlock(
                "right",
                "triangle",
                jnp.asarray(((1, 2, 4), (2, 5, 4)), dtype=jnp.int32),
                global_ids=jnp.asarray((2, 3)),
            ),
        ),
    )


def _binary_model(*, polynomial=False):
    parameters = phx.equations.BinaryThermodynamicParameters(1.0, 1.0)
    if not polynomial:
        return phx.applications.phase_field.BinaryPhaseFieldModel(parameters)
    potential = phx.equations.PolynomialBulkFreeEnergy((0.25, 0.0, -0.5, 0.0, 0.25))
    return phx.applications.phase_field.BinaryPhaseFieldModel(
        parameters,
        closure=phx.equations.BinaryPhaseThermodynamicClosure(potential),
    )


def _binary_and_boundary() -> dict[str, object]:
    element = phx.discretization.lagrange_element("triangle", 1)
    mesh = _multiblock_mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "eta", {"left": element, "right": element}
        ),
    ).prepare()
    method = phx.applications.phase_field.AllenCahnFEMPlan(
        _binary_model(polynomial=True), 1.0
    ).prepare(discretization, "eta")
    initial = method.initialize(
        jnp.asarray((-0.2, 0.1, 0.3, -0.1, 0.2, -0.3), dtype=jnp.float64)
    )
    result = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.01)
    )

    square = _square_mesh()
    boundary_discretization = phx.discretization.FiniteElementPlan(
        square,
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    wetting = phx.applications.phase_field.YoungAngleSurfaceEnergy(1.0, jnp.pi / 3.0)
    loading = phx.applications.phase_field.PrescribedMicrotractionEnergy(
        lambda points, time, args: jnp.full(points.shape[:-1], 0.2 * time),
        traction_id="qualification-linear-load",
    )
    boundary = phx.applications.phase_field.PhaseFieldBoundaryPlan(
        boundary_discretization,
        {
            "wetting": ((0,), wetting, None),
            "loaded": ((4,), loading, None),
        },
    )
    driven = phx.applications.phase_field.AllenCahnFEMPlan(_binary_model(), 1.0).prepare(
        boundary_discretization, "eta", boundary=boundary
    )
    driven_initial = driven.initialize(
        jnp.asarray((-0.2, 0.1, 0.3, -0.1), dtype=jnp.float64)
    )
    driven_result = driven.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), driven_initial, jnp.asarray(0.01)
    )
    passed = bool(result.successful & driven_result.successful)
    return {
        "status": "pass" if passed else "fail",
        "multiblock_count": len(mesh.blocks),
        "discrete_gradient_ledger_residual": float(
            np.asarray(result.evidence.ledger.total_residual)
        ),
        "surface_energy_before": float(
            np.asarray(driven_result.evidence.ledger.surface_before)
        ),
        "boundary_work": float(np.asarray(driven_result.evidence.boundary_work)),
        "driven_ledger_residual": float(
            np.asarray(driven_result.evidence.ledger.total_residual)
        ),
    }


def _transport_and_periodic() -> dict[str, object]:
    element = phx.discretization.lagrange_element("triangle", 1)
    mesh = _square_mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec("c", element),
            phx.discretization.FiniteElementFieldSpec("mu", element),
        ),
    ).prepare()
    mobility = phx.applications.phase_field.TensorPhaseFieldMobility(
        jnp.asarray(((1.0, 0.0), (0.0, 0.5)), dtype=jnp.float64)
    )
    flux = phx.applications.phase_field.PrescribedPhaseFieldFlux(
        lambda points, time, args: jnp.full(points.shape[:-1], 1.0e-3),
        flux_id="qualification-inflow",
    )
    boundary = phx.applications.phase_field.PhaseFieldBoundaryPlan(
        discretization,
        {"inlet": ((0,), None, flux)},
    )
    method = phx.applications.phase_field.CahnHilliardFEMPlan(
        _binary_model(), mobility
    ).prepare(discretization, "c", "mu", boundary=boundary)
    initial = method.initialize(jnp.asarray((-0.2, 0.1, 0.3, -0.1), dtype=jnp.float64))
    result = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.005)
    )

    periodic_discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("u", element),
    ).prepare()
    orientation = phx.discretization.facet_orientation_actions("edge")[0]
    periodic = phx.discretization.FiniteElementBoundarySet(
        periodic_discretization,
        {},
        periodic_pairs=(
            phx.discretization.FiniteElementPeriodicFacetPair(
                0,
                4,
                transform=phx.discretization.FiniteElementPeriodicTransform(
                    jnp.eye(2), jnp.asarray((0.0, 1.0)), orientation
                ),
            ),
            phx.discretization.FiniteElementPeriodicFacetPair(
                2,
                3,
                transform=phx.discretization.FiniteElementPeriodicTransform(
                    jnp.eye(2), jnp.asarray((1.0, 0.0)), orientation
                ),
            ),
        ),
    )
    constraint = phx.discretization.periodic_constraint(
        periodic_discretization, "u", periodic
    )
    periodic_values = constraint.constraint_map.prolongation.mv(jnp.asarray((1.0,)))
    periodic_defect = float(np.max(np.abs(np.asarray(periodic_values) - 1.0)))
    passed = bool(result.successful) and periodic_defect == 0.0
    return {
        "status": "pass" if passed else "fail",
        "mass_source": float(np.asarray(result.evidence.mass_source)),
        "mass_defect": float(np.asarray(result.evidence.mass_defect)),
        "tensor_dissipation": float(np.asarray(result.evidence.dissipation)),
        "periodic_reduced_dofs": constraint.constraint_map.reduced_space.size,
        "periodic_constant_defect": periodic_defect,
    }


def _grand_and_active() -> dict[str, object]:
    phase_a = phx.applications.phase_field.QuadraticGrandPotentialPhase(
        "qualification-a", 0.0, jnp.asarray((0.2,)), jnp.asarray(((1.0,),))
    )
    phase_b = phx.applications.phase_field.QuadraticGrandPotentialPhase(
        "qualification-b", 0.0, jnp.asarray((0.8,)), jnp.asarray(((1.0,),))
    )
    catalog = phx.applications.phase_field.GrandPotentialMaterialCatalog(
        (phase_a, phase_b)
    )
    model = phx.applications.phase_field.GrandPotentialMixtureModel(
        catalog,
        barrier_scale=0.1,
        gradient_coefficient=0.2,
        kinetic_coefficient=1.0,
        mobility=1.0,
    )
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        (
            phx.discretization.FiniteElementFieldSpec(
                "eta", element, component_shape=(2,)
            ),
            phx.discretization.FiniteElementFieldSpec(
                "mu", element, component_shape=(1,)
            ),
        ),
    ).prepare()
    method = phx.applications.phase_field.GrandPotentialFEMPlan(
        model,
        absolute_energy_tolerance=1.0,
        relative_energy_tolerance=1.0,
        component_tolerance=1.0e-7,
    ).prepare(discretization, "eta", "mu")
    initial = method.initialize(
        jnp.asarray(
            ((1.0, -1.0), (0.5, -0.5), (-0.5, 0.5), (-1.0, 1.0)),
            dtype=jnp.float64,
        ),
        jnp.zeros((4, 1), dtype=jnp.float64),
    )
    result = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(1.0e-3)
    )

    active_plan = phx.applications.phase_field.ActivePhaseStoragePlan(
        discretization.dof_maps[0].cell_dofs[0],
        catalog.phase_count,
        catalog.phase_count,
        cell_phase_capacity=catalog.phase_count,
    )
    dense = jax.nn.softmax(result.accepted_state.phase_logits, axis=-1)
    active = active_plan.from_dense(dense)
    dense_defect = float(np.max(np.abs(np.asarray(active_plan.dense(active) - dense))))
    passed = bool(result.successful & active.evidence.successful) and dense_defect == 0.0
    return {
        "status": "pass" if passed else "fail",
        "component_defect": float(np.asarray(result.evidence.component_defect)),
        "energy_defect": float(np.asarray(result.evidence.energy_defect)),
        "active_dense_defect": dense_defect,
        "maximum_local_phase_count": int(
            np.max(np.asarray(active.evidence.required_per_dof))
        ),
        "maximum_cell_phase_count": int(
            np.max(np.asarray(active.evidence.required_per_cell))
        ),
    }


def _adaptive_stochastic_distributed() -> dict[str, object]:
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    realization = phx.stochastic.WienerRealization(
        jax.random.key(19),
        (4,),
        support=(0.0, 1.0),
        tolerance=1.0e-5,
        noise_id="phase-field-extended-qualification",
    )
    noise = phx.applications.phase_field.PhaseFieldNoisePlan(
        "allen-cahn", realization, 1.0e-3 * jnp.eye(4)
    )
    method = phx.applications.phase_field.AllenCahnFEMPlan(_binary_model(), 1.0).prepare(
        discretization, "eta", noise=noise
    )
    initial = method.initialize(jnp.asarray((-0.2, 0.1, 0.3, -0.1), dtype=jnp.float64))
    first = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.01)
    )
    replay = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.01)
    )
    replay_defect = float(
        np.max(
            np.abs(np.asarray(first.candidate_state.phase - replay.candidate_state.phase))
        )
    )
    epoch = phx.applications.phase_field.PhaseFieldAdaptiveEpoch(method, initial)
    adaptation = phx.applications.phase_field.PhaseFieldAdaptivityPlan(
        gradient_threshold=0.0,
        energy_tolerance=1.0,
    ).refine(epoch)
    distributed = phx.applications.phase_field.DistributedPhaseFieldPlan(
        discretization, 2
    )
    cell_values = jnp.asarray((1.0, 2.0))
    reduction = float(np.asarray(distributed.reference_owned_sum(cell_values)))
    if jax.device_count() >= 2:
        device_mesh = Mesh(
            np.asarray(jax.devices()[:2], dtype=object),
            (distributed.axis_name,),
        )

        def execute_collective(values):
            return distributed.collective_owned_sum(values).global_value

        collective = jax.shard_map(
            execute_collective,
            mesh=device_mesh,
            in_specs=P(),
            out_specs=P(),
            check_vma=False,
        )(cell_values)
        collective_defect = float(np.max(np.abs(np.asarray(collective) - reduction)))
        distributed_passed = collective_defect == 0.0
    else:
        collective_defect = float("inf")
        distributed_passed = False
    profiles = phx.applications.phase_field.phase_field_candidate_profiles()
    passed = (
        bool(first.successful)
        and replay_defect == 0.0
        and bool(adaptation.committed)
        and reduction == 3.0
        and distributed_passed
        and len(profiles) == 4
    )
    return {
        "status": "pass" if passed else "fail",
        "stochastic_replay_defect": replay_defect,
        "stochastic_work": float(np.asarray(first.evidence.stochastic_work)),
        "adaptation_committed": bool(adaptation.committed),
        "adapted_cells": adaptation.candidate.method.discretization.mesh.blocks[
            0
        ].cell_count,
        "distributed_reference_sum": reduction,
        "distributed_collective_defect": collective_defect,
        "distributed_device_count": jax.device_count(),
        "candidate_profile_names": [profile.name for profile in profiles],
    }


def qualify() -> dict[str, object]:
    if not bool(jax.config.read("jax_enable_x64")):
        raise ValueError("Extended phase-field qualification requires float64.")
    sections = {
        "binary_boundary": _binary_and_boundary(),
        "transport_periodic": _transport_and_periodic(),
        "grand_active": _grand_and_active(),
        "adaptive_stochastic_distributed": _adaptive_stochastic_distributed(),
    }
    passed = all(section["status"] == "pass" for section in sections.values())
    artifact_id = canonical_fingerprint(
        {"kind": "phase-field-extended-qualification", "sections": sections}
    )
    released = (
        phx.applications.phase_field.phase_field_released_profiles(
            artifact_id,
            reviewer_id="phydrax-phase-field-qualification",
            issued_at=1767225600,
            expires_at=1798761600,
        )
        if passed
        else ()
    )
    return {
        "status": "pass" if passed else "fail",
        "capability": "phase-field-complete-closure",
        "artifact_id": artifact_id,
        "released_profiles": [profile.to_record() for profile in released],
        "sections": sections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qualify the integrated phase-field closure."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/phase_field_extended_qualification.json"),
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
