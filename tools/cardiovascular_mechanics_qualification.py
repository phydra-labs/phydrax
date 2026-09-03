#!/usr/bin/env python3
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

from phydrax.applications.cardiovascular.anatomy._microstructure import (
    CardiacMaterialFrame,
)
from phydrax.applications.cardiovascular.anatomy._surfaces import ChamberSurfacePlan
from phydrax.applications.cardiovascular.mechanics._chambers import (
    ChamberVolumePlan,
    FollowerPressurePlan,
)
from phydrax.applications.cardiovascular.mechanics._guccione import (
    Guccione1991Energy,
    Guccione1991Parameters,
)
from phydrax.applications.cardiovascular.mechanics._holzapfel_ogden import (
    HolzapfelOgden2009Parameters,
    HolzapfelOgden2009TensionOnlyEnergy,
)
from phydrax.applications.cardiovascular.mechanics._supports import PericardialSupport
from phydrax.applications.cardiovascular.mechanics._unloading import (
    ForwardContinuationResult,
    read_unloaded_reference_checkpoint,
    recover_unloaded_reference,
    UnloadedReferenceRecoveryPlan,
    write_unloaded_reference_checkpoint,
)
from phydrax.discretization import (
    CellBlock,
    CellMesh,
    MixedFiniteElementConstraintPlan,
    PressureGaugePolicy,
)


def _maximum_absolute(value):
    return jnp.max(jnp.abs(value))


def _material_frame():
    return CardiacMaterialFrame(
        jnp.asarray(((1.0, 0.0, 0.0),)),
        jnp.asarray(((0.0, 1.0, 0.0),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((True,)),
        frame_id="qualification-material-frame",
    )


def _mixed_hexahedral_mesh():
    coordinates = jnp.asarray(
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
    block = CellBlock(
        "myocardium",
        "hexahedron",
        jnp.asarray(((0, 1, 2, 3, 4, 5, 6, 7),), dtype=jnp.int32),
    )
    return CellMesh(coordinates, (block,))


def _material_qualification():
    frame = _material_frame()
    energies = (
        Guccione1991Energy(
            Guccione1991Parameters(0.9, 8.0, 2.0, 4.0),
            frame,
            cell_index=0,
        ),
        HolzapfelOgden2009TensionOnlyEnergy(
            HolzapfelOgden2009Parameters(0.12, 5.0, 1.8, 8.0, 0.7, 6.0, 0.3, 4.0),
            frame,
            cell_index=0,
        ),
    )
    deformation = jnp.asarray(((1.08, 0.06, 0.01), (0.02, 0.96, 0.04), (0.0, 0.01, 1.01)))
    rotation = jnp.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    cases = []
    all_passed = True
    for energy in energies:
        material = energy.finite_bulk(80.0)
        response = material.evaluate(deformation)
        rotated = material.evaluate(rotation @ deformation)
        stress = jax.grad(material.reference_energy_density)(deformation)
        tangent = jax.jacfwd(
            lambda value: jax.grad(material.reference_energy_density)(value)
        )(deformation)
        objectivity_error = jnp.maximum(
            jnp.abs(rotated.reference_energy_density - response.reference_energy_density),
            _maximum_absolute(rotated.first_piola - rotation @ response.first_piola),
        )
        stress_error = _maximum_absolute(stress - response.first_piola)
        tangent_error = _maximum_absolute(tangent - response.tangent)
        passed = (
            response.admissible
            & (objectivity_error < 3.0e-5)
            & (stress_error < 3.0e-5)
            & (tangent_error < 3.0e-5)
        )
        all_passed = all_passed and bool(passed)
        cases.append(
            {
                "energy_id": energy.energy_id,
                "objectivity_error": float(objectivity_error),
                "energy_stress_error": float(stress_error),
                "stress_tangent_error": float(tangent_error),
                "passed": bool(passed),
            }
        )
    return {"passed": all_passed, "cases": cases}


def _mixed_qualification():
    energy = Guccione1991Energy(
        Guccione1991Parameters(0.9, 8.0, 2.0, 4.0),
        _material_frame(),
        cell_index=0,
    )
    exact = energy.exact_incompressible()
    deformation = jnp.asarray(((1.1, 0.03, 0.0), (0.0, 1.0 / 1.1, 0.02), (0.0, 0.0, 1.0)))
    response = exact.evaluate(deformation, jnp.asarray(1.7))
    blocks = exact.block_tangent(deformation, jnp.asarray(1.7))
    adjoint_error = _maximum_absolute(
        blocks.deformation_pressure - blocks.pressure_deformation
    )
    gauge = PressureGaugePolicy("mean-zero")
    plan = MixedFiniteElementConstraintPlan(
        _mixed_hexahedral_mesh(),
        gauge,
        displacement_field="u",
        pressure_field="p",
        plan_id="qualification-exact-mixed-q2-q1",
    )
    qualified = exact.prepare_qualified(
        plan,
        form_id="qualification-cardiac-exact-mixed",
    )
    qualification = qualified.qualification
    prepared = qualified.prepared
    passed = (
        response.evidence.valid
        & (jnp.abs(response.constraint_residual) < 2.0e-6)
        & (adjoint_error < 2.0e-6)
        & qualification.gauge_valid
        & qualification.residual_finite
        & qualification.stable_pair
        & qualification.assembled_inf_sup_stable
        & qualification.locking_safe
        & qualification.valid
    )
    return {
        "passed": bool(passed),
        "route": "exact-mixed-u-p",
        "assembled_form_id": prepared.problem.form.form_id,
        "mesh_id": plan.mesh.mesh_id,
        "pair_names": list(qualification.pair_names),
        "displacement_degree": prepared.spaces.displacement_degree,
        "pressure_degree": prepared.spaces.pressure_degree,
        "gauge_mode": qualification.gauge_mode,
        "gauge_valid": bool(qualification.gauge_valid),
        "residual_finite": bool(qualification.residual_finite),
        "constraint_residual": float(jnp.abs(response.constraint_residual)),
        "block_adjoint_error": float(adjoint_error),
        "assembled_adjoint_defect": prepared.inf_sup.adjoint_defect,
        "inf_sup_constant": float(qualification.inf_sup_constant),
        "assembled_inf_sup_stable": bool(qualification.assembled_inf_sup_stable),
        "locking_safe": bool(qualification.locking_safe),
    }


def _chamber_qualification():
    coordinates = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    triangles = jnp.asarray(
        ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
        dtype=jnp.int32,
    )
    surface = ChamberSurfacePlan("qualification-lv", coordinates, triangles).prepare()
    volume_plan = ChamberVolumePlan(surface)
    pressure_plan = FollowerPressurePlan(volume_plan, load_id="qualification-pressure")
    pressure = jnp.asarray(2.5)
    response = volume_plan.evaluate(coordinates)
    derivative = jax.grad(volume_plan.volume)(coordinates)
    derivative_error = _maximum_absolute(derivative - response.volume_gradient)
    pressure_response = pressure_plan.evaluate(coordinates, pressure)
    potential_gradient = jax.grad(
        lambda value: pressure_plan.evaluate(value, pressure).pressure_potential
    )(coordinates)
    force_error = _maximum_absolute(pressure_response.nodal_force + potential_gradient)
    expanded = 1.02 * coordinates
    work = pressure_plan.work_between(coordinates, expanded, pressure)
    potential_change = (
        pressure_plan.evaluate(expanded, pressure).pressure_potential
        - pressure_response.pressure_potential
    )
    work_error = jnp.abs(work + potential_change)
    passed = (
        response.valid
        & pressure_response.valid
        & (derivative_error < 2.0e-6)
        & (force_error < 2.0e-6)
        & (work_error < 2.0e-6)
    )
    return {
        "passed": bool(passed),
        "surface_id": surface.surface_id,
        "volume": float(response.volume),
        "volume_derivative_error": float(derivative_error),
        "pressure_force_error": float(force_error),
        "constant_pressure_work_error": float(work_error),
    }


def _support_qualification():
    displacement = jnp.asarray((0.11, -0.04, 0.08))
    support = PericardialSupport((0.0, 0.0, 1.0), 3.0, 1.5)
    response = support.evaluate(displacement)
    gradient = jax.grad(support.energy_density)(displacement)
    traction_error = _maximum_absolute(response.restoring_traction + gradient)
    free = PericardialSupport((0.0, 0.0, 1.0), 0.0, 0.0).evaluate(displacement)
    free_error = jnp.maximum(
        jnp.abs(free.energy_density),
        _maximum_absolute(free.restoring_traction),
    )
    passed = response.valid & (traction_error < 2.0e-7) & (free_error == 0.0)
    return {
        "passed": bool(passed),
        "traction_energy_error": float(traction_error),
        "traction_free_limit_error": float(free_error),
        "pericardial_model_claim": "Robin foundation, not contact",
    }


def _unloading_qualification():
    unloaded = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    displacement = jnp.asarray(
        ((0.02, 0.0, 0.01), (0.04, -0.01, 0.0), (0.0, 0.03, -0.01), (-0.01, 0.0, 0.04))
    )
    loaded = unloaded + displacement

    def forward_path(reference, factors, args):
        del args
        coordinates = reference[None, ...] + factors[:, None, None] * displacement
        return ForwardContinuationResult(
            coordinates,
            jnp.zeros_like(factors),
            jnp.ones_like(factors, dtype=bool),
        )

    plan = UnloadedReferenceRecoveryPlan(
        jnp.linspace(0.0, 1.0, 6),
        residual_tolerance=2.0e-6,
        equilibrium_tolerance=1.0e-10,
        maximum_steps=20,
        plan_id="qualification-unloading",
    )
    result = recover_unloaded_reference(plan.prepare(loaded, forward_path), loaded)
    recovery_error = _maximum_absolute(result.reference_coordinates - unloaded)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "unloaded-reference.phx"
        write_unloaded_reference_checkpoint(checkpoint_path, result.state)
        restored = read_unloaded_reference_checkpoint(
            checkpoint_path,
            plan.prepare(loaded, forward_path),
        )
    checkpoint_exact = jnp.array_equal(
        restored.reference_coordinates,
        result.reference_coordinates,
    ) & jnp.array_equal(
        restored.continuation_coordinates,
        result.state.continuation_coordinates,
    )
    passed = result.successful & (recovery_error < 2.0e-6) & checkpoint_exact
    return {
        "passed": bool(passed),
        "load_stations": int(plan.load_factors.size),
        "recovery_error": float(recovery_error),
        "relative_loaded_residual": float(result.evidence.relative_residual),
        "zero_load_residual": float(result.evidence.zero_load_residual),
        "maximum_equilibrium_residual": float(
            result.evidence.maximum_equilibrium_residual
        ),
        "checkpoint_exact": bool(checkpoint_exact),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    sections = {
        "materials": _material_qualification(),
        "mixed_incompressibility": _mixed_qualification(),
        "chamber": _chamber_qualification(),
        "supports": _support_qualification(),
        "unloading": _unloading_qualification(),
    }
    payload = {
        "qualification": "cardiovascular-passive-mechanics",
        "passed": all(section["passed"] for section in sections.values()),
        "sections": sections,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
