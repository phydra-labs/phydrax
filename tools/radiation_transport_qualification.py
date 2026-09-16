#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx


def _manifest(name: str, character: str):
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=character * 64,
        size_bytes=1,
        license_id="synthetic-qualification",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"identity": 1.0},
        uncertainty={"synthetic": 0.0},
        lineage_ids=(f"synthetic:{name}",),
    )


def _photon_table(name, character, energy_grid, values):
    provenance = phx.nuclear.NuclearDataProvenance(
        _manifest(name, character),
        f"https://example.invalid/{name}",
        "synthetic-photon-qualification",
        "qualification",
        name,
    )
    unit = phx.units.derived_unit(
        "m2/kg", ((phx.units.METER, 2), (phx.units.KILOGRAM, -1))
    )
    return phx.equations.DiagnosticPhotonCoefficientTable(
        phx.equations.DiagnosticPhotonCoefficientRole.MASS_ATTENUATION,
        energy_grid,
        ("material",),
        jnp.asarray((values,)),
        unit,
        provenance,
        phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
    )


def build_photon_case(history_count=256):
    energy_grid = phx.equations.PhotonEnergyGrid(
        jnp.asarray((500.0, 1500.0))
        * float(phx.units.conversion_factor(phx.units.ELECTRONVOLT, phx.units.JOULE))
    )
    photoelectric = _photon_table(
        "photoelectric", "a", energy_grid, (np.log(2.0), np.log(2.0))
    )
    compton = _photon_table("compton", "c", energy_grid, (0.0, 0.0))
    rayleigh = _photon_table("rayleigh", "d", energy_grid, (0.0, 0.0))
    library = phx.equations.RadiationCrossSectionLibrary(
        photoelectric, compton, rayleigh, jnp.asarray((1.0,))
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.zeros((3,)),
        jnp.ones((3,)),
        jnp.zeros((1, 1, 1), dtype=jnp.int32),
        material_count=1,
    )
    plan = phx.solver.PhotonTransportPlan(
        geometry, library, maximum_events=32, cutoff_energy=500.0
    )
    origins = jnp.broadcast_to(jnp.asarray((0.5, 0.5, 0.0)), (history_count, 3))
    directions = jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (history_count, 3))
    energies = jnp.full((history_count,), 1000.0)
    return plan, origins, directions, energies


def _sn_case():
    quadrature = phx.discretization.CertifiedSlabAngularQuadrature.gauss_legendre(4)
    incident = (
        jnp.zeros((1, quadrature.angle_count)).at[0, quadrature.ordinates > 0.0].set(1.0)
    )
    boundaries = phx.equations.SlabTransportBoundaryPlan(
        1, quadrature, left_kind="incident", left_incident=incident
    )
    problem = phx.equations.MultigroupSlabTransportProblem(
        jnp.linspace(0.0, 1.0, 33),
        jnp.ones((32, 1)),
        jnp.zeros((32, 1, 1)),
        jnp.zeros((32, 1)),
        quadrature,
        boundaries,
    )
    return phx.solver.DiscreteOrdinatesTransportPlan(
        problem, maximum_iterations=2, tolerance=1.0e-12
    )


def _charged_case():
    material = phx.equations.ChargedRadiationMaterialLibrary(
        jnp.asarray((10.0, 2000.0)),
        jnp.full((1, 2), 1000.0),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        ("material",),
        _manifest("charged-material", "b"),
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.zeros((3,)),
        jnp.asarray((1.0, 1.0, 2.0)),
        jnp.zeros((1, 1, 2), dtype=jnp.int32),
        material_count=1,
    )
    return phx.solver.ChargedParticleTransportPlan(
        geometry,
        material,
        maximum_steps=256,
        maximum_step_length=0.05,
        cutoff_energy_ev=10.0,
    )


def qualify() -> dict[str, object]:
    photon_plan, origins, directions, energies = build_photon_case()
    photon = photon_plan.simulate(origins, directions, energies, jr.key(1))
    sn = _sn_case().solve()
    charged = _charged_case().simulate(
        jnp.asarray(((0.5, 0.5, 0.1),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((1000.0,)),
        jnp.asarray((int(phx.equations.ChargedRadiationParticleKind.ELECTRON),)),
        jr.key(2),
    )
    imc_plan = phx.solver.HybridIMCDDMCPlan(
        jnp.asarray((1.0,)),
        jnp.asarray(((1.0e-3,),)),
        jnp.zeros((1, 1)),
        jnp.asarray((1000.0,)),
        maximum_packet_energy=1000.0,
    )
    imc_state = imc_plan.initialize(
        jnp.asarray((0,)),
        jnp.asarray((0,)),
        jnp.asarray((1,)),
        jnp.asarray((100.0,)),
        jnp.asarray((True,)),
        jnp.asarray((300.0,)),
    )
    imc = imc_plan.advance(imc_state, 1.0e-3, jr.key(3))
    transfer = phx.applications.astrophysics.RayTransferPlan(
        jnp.asarray(((1.0, 1.0),)), ray_id="qualification-ray"
    )
    spectral = phx.applications.radiation_transport.CorrelatedKDistributionPlan(
        jnp.asarray(((0.5, 0.5),)), ("band",)
    )
    sensor = phx.applications.radiation_transport.RadiativeSensorPlan(
        jnp.asarray((1.0,)), sensor_id="qualification-sensor"
    )
    experiment = phx.applications.radiation_transport.ScalarRadiativeExperimentPlan(
        transfer, spectral, sensor
    ).evaluate(
        jnp.ones((1, 2, 1, 2)),
        jnp.zeros((1, 2, 1, 2)),
    )
    checks = {
        "photon": bool(photon.all_successful),
        "discrete_ordinates": bool(sn.evidence.successful),
        "charged": bool(charged.all_successful),
        "imc_ddmc": bool(imc.successful),
        "spectral_experiment": bool(experiment.successful),
    }
    return {
        "qualification": "radiation-transport-native-closure",
        "evidence_scope": "synthetic-numerical-validity-only",
        "checks": checks,
        "metrics": {
            "photon_ledger": float(photon.maximum_ledger_residual),
            "sn_residual": float(sn.evidence.source_iteration_residual),
            "sn_balance": float(sn.evidence.global_balance_residual),
            "charged_ledger": float(charged.maximum_kinetic_ledger_residual),
            "imc_ledger": float(imc.evidence.energy_residual),
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
