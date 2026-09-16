#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


def _manifest(name: str) -> phx.qualification.ReferenceArtifactManifest:
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=("a" if name == "model" else "b") * 64,
        size_bytes=1,
        license_id="synthetic-qualification",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"identity": 1.0},
        uncertainty={"synthetic": 0.0},
        lineage_ids=(f"synthetic:{name}",),
    )


def _zero_uncertainty(features):
    return jnp.zeros(features.shape[:-1])


def _extent_model(features):
    return 1.0e-4 * jnp.ones(features.shape[:-1] + (1,))


def build_low_mach_case():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.01, 0.01)),
        ("E",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
        provenance="synthetic-reacting-closure",
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((20.0, 20.0)),
        jnp.asarray((0.0, -5.0e3)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=3000.0,
    )
    thermodynamics = phx.equations.HomogeneousHelmholtzPlan(
        phx.equations.IdealGasReferenceHelmholtzTerm(schema, species),
        phx.equations.ZeroResidualHelmholtzTerm(schema),
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "A-to-B",
        schema,
        species,
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(0.1),
            ),
        ),
    ).prepare()
    properties = phx.equations.ReferencePowerLawGasTransportPlan(
        jnp.asarray(((0.0, 1.0e-5), (1.0e-5, 0.0))),
        jnp.asarray((1.0e-5, 1.0e-5)),
        jnp.asarray((0.02, 0.02)),
    )
    mixture = phx.equations.MixtureAveragedTransportPlan(thermodynamics, properties)
    formulation = phx.applications.reacting_flow.LowMachReactingFormulation(
        thermodynamics, 1, mechanism=mechanism, constraint_tolerance=1.0e-8
    )
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    scalar = phx.discretization.MACScalarProblem(
        tuple(
            phx.discretization.MACScalarTransport(name, 0.0)
            for name in ("rho:A", "rho:B", "rhoh")
        )
    ).prepare(operators)
    projection = phx.solver.MACVariableDensityProjectionPlan(
        operators, tolerance=1.0e-8, maximum_iterations=200
    )
    plan = phx.applications.reacting_flow.LowMachReactingFlowPlan(
        formulation,
        mixture,
        scalar,
        projection,
        phx.applications.reacting_flow.LowMachReactingSDCPlan(3, 1.0e-6),
        pressure_mode="closed",
        conservation_tolerance=1.0e-7,
        eos_tolerance=1.0e-6,
    )
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    state = plan.initialize(
        velocity,
        jnp.full(finite_volume.cell_shape, 800.0),
        jnp.broadcast_to(jnp.asarray((0.8, 0.2)), finite_volume.cell_shape + (2,)),
        1.0e5,
    )
    return thermodynamics, mechanism, plan, state


def qualify() -> dict[str, object]:
    thermodynamics, mechanism, low_mach, state = build_low_mach_case()
    low_mach_result = low_mach.advance(state, 0.01)
    equilibrium = phx.solver.ChemicalEquilibriumPlan(
        thermodynamics,
        phx.solver.ChemicalEquilibriumEnsemble.TP,
        tolerance=1.0e-6,
        maximum_steps=400,
    ).solve(700.0, 1.0e5, jnp.asarray((0.9, 0.1)))
    cema = phx.applications.reacting_flow.ChemicalExplosiveModePlan(mechanism).evaluate(
        jnp.asarray((80.0, 20.0)), 800.0, 1.0e5
    )
    source = phx.applications.reacting_flow.EnergyDepositionSourcePlan(
        jnp.ones((8,)),
        jnp.ones((8,)),
        total_energy=10.0,
        start_time=0.0,
        end_time=1.0,
        source_id="synthetic-ignition",
    ).evaluate(0.5)
    features = phx.applications.reacting_flow.LearnedChemicalFeatureSchema(
        ("log_A", "log_B", "temperature", "pressure", "log_step"),
        ("log(mol/m3)", "log(mol/m3)", "K", "Pa", "log(s)"),
        jnp.asarray((-10.0, -10.0, 300.0, 5.0e4, -10.0)),
        jnp.asarray((10.0, 10.0, 2000.0, 2.0e5, 0.0)),
    )
    learned = phx.applications.reacting_flow.LearnedChemicalTransitionPlan(
        mechanism,
        features,
        _extent_model,
        _zero_uncertainty,
        _manifest("model"),
        (_manifest("training"),),
        model_id="synthetic-extent",
        maximum_uncertainty=0.1,
    ).advance(jnp.asarray((0.8, 0.2)), 800.0, 1.0e5, 0.01)
    checks = {
        "low_mach": bool(low_mach_result.successful),
        "equilibrium": bool(equilibrium.evidence.successful),
        "cema": bool(cema.evidence.successful),
        "source_work": bool(source.successful),
        "learned_transition": bool(learned.successful),
    }
    return {
        "qualification": "reacting-flow-native-closure",
        "evidence_scope": "synthetic-numerical-validity-only",
        "checks": checks,
        "metrics": {
            "low_mach_sdc_residual": float(low_mach_result.diagnostics.sdc_residual),
            "low_mach_eos_defect": float(
                jnp.max(jnp.abs(low_mach_result.diagnostics.eos_pressure_defect))
            ),
            "equilibrium_balance": float(
                jnp.max(jnp.abs(equilibrium.evidence.element_residual))
            ),
            "cema_eigen_residual": float(cema.evidence.eigen_residual),
            "source_cumulative_work": float(source.cumulative_work),
        },
        "successful": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if not report["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
