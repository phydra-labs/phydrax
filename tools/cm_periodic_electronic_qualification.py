#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment


periodic = phx.chemistry.periodic


def _cell():
    return phx.discretization.PeriodicCell(
        np.asarray([[5.0, 0.0, 0.0], [0.6, 4.8, 0.0], [0.3, 0.2, 5.2]]),
        periodic_axes=(True, True, True),
    )


def _units():
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _manifest(source_id: str):
    return periodic.PeriodicProvenanceManifest.for_bytes(
        f"{source_id}:qualification".encode(),
        source_id,
        "qualification-generated",
    )


def _spin_case(reference_kind: str, smearing: float, magnetization: float):
    cell = _cell()
    basis = periodic.PeriodicOrbitalBasisPlan(
        cell,
        ("lower", "upper"),
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        phx.units.ANGSTROM,
        periodic.PeriodicBlochGauge("lattice"),
    )
    blocks = np.asarray([np.diag([-1.0, 0.8])]).reshape((1, 2, 1, 2, 1))
    family = phx.operators.periodic.periodic_translation_family_from_dense_blocks(
        [[0, 0, 0]], blocks
    )
    pencil = periodic.PeriodicOrbitalPencilPlan.orthonormal(
        basis,
        family.plan,
        family.state,
        phx.units.ELECTRONVOLT,
    ).prepare()
    electron_count = 2.0 if reference_kind == "restricted" else 1.0
    mean_field = periodic.PeriodicHubbardMeanFieldPlan(
        basis,
        [0.0, 0.0],
        [electron_count, 0.0],
        0.0,
        phx.units.ELECTRONVOLT,
    )
    mesh = phx.discretization.ReciprocalMeshPlan.monkhorst_pack(cell, (2, 1, 1))
    keywords = {}
    if reference_kind == "collinear":
        keywords["reference_spin_populations"] = (
            [0.5 * (electron_count + magnetization), 0.0],
            [0.5 * (electron_count - magnetization), 0.0],
        )
    result = periodic.SpinPeriodicSCFPlan(
        mesh,
        phx.chemistry.PeriodicElectronicSectorPlan(
            electron_count, spin_magnetization=magnetization
        ),
        pencil,
        mean_field,
        reference_kind=reference_kind,
        smearing_energy=smearing,
        **keywords,
    ).evaluate()
    return {
        "reference_kind": result.reference_kind,
        "smearing_energy": smearing,
        "electron_count_residual": float(result.evidence.electron_count_residual),
        "spin_count_residual": float(result.evidence.spin_count_residual),
        "commutator_residual": float(result.evidence.commutator_residual),
        "eigenpair_residual": float(result.evidence.eigenpair_residual),
        "energy_closure_residual": float(result.energy_ledger.closure_residual),
        "free_energy_residual": float(result.evidence.free_energy_residual),
        "passed": bool(result.successful),
    }


def _derivative_case():
    def orbital(positions, vectors):
        return 0.5 * jnp.sum(positions**2) + 0.02 * jnp.linalg.det(vectors)

    def pulay(positions, vectors):
        return 0.1 * jnp.sum(positions) + 0.0 * jnp.sum(vectors)

    def zero(positions, vectors):
        return 0.0 * (jnp.sum(positions) + jnp.sum(vectors))

    components = (
        periodic.PeriodicStationaryEnergyComponent(
            "orbital", "hellmann-feynman", orbital, "analytic-orbital"
        ),
        periodic.PeriodicStationaryEnergyComponent(
            "metric", "pulay", pulay, "analytic-pulay"
        ),
        periodic.PeriodicStationaryEnergyComponent(
            "entropy", "entropy", zero, "zero-entropy"
        ),
        periodic.PeriodicStationaryEnergyComponent(
            "projector", "nonlocal", zero, "zero-nonlocal"
        ),
        periodic.PeriodicStationaryEnergyComponent("ionic", "ionic", zero, "zero-ionic"),
    )
    result = periodic.PeriodicStationaryDerivativePlan(
        _cell(),
        components,
        phx.units.ELECTRONVOLT,
        phx.units.ANGSTROM,
        energy_kind="free-energy",
        directional_tolerance=2.0e-8,
    ).evaluate(
        [[0.2, 0.3, 0.4], [0.8, 0.1, 0.5]],
        0.0,
        [[1.0, 0.0, 0.0], [-0.5, 0.2, 0.0]],
        [[0.2, 0.1, 0.0], [0.1, -0.1, 0.0], [0.0, 0.0, 0.05]],
    )
    return {
        "stationarity_residual": float(result.evidence.stationarity_residual),
        "force_directional_residual": float(result.evidence.force_directional_residual),
        "stress_directional_residual": float(result.evidence.stress_directional_residual),
        "energy_closure_residual": float(result.ledger.energy_closure_residual),
        "force_closure_residual": float(result.ledger.force_closure_residual),
        "stress_closure_residual": float(result.ledger.stress_closure_residual),
        "roles": list(result.ledger.component_roles),
        "passed": bool(result.successful),
    }


def _provider_case():
    units = _units()
    cell = _cell()
    system = phx.atomistic.AtomisticSystemPlan(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([1.008, 1.008]),
        units,
        molecule_ids=np.zeros(2, dtype=np.int32),
        cell=cell,
    )
    model = phx.chemistry.ElectronicModelChemistryPlan(
        phx.chemistry.ExternalElectronicMethodPlan(
            "provider-scalar-relativistic",
            phx.chemistry.ElectronicReferenceKind.RESTRICTED,
            definition_ids=("scalar-relativistic-v1", "gth-fixture-v1"),
        ),
        basis=phx.chemistry.BasisSetReference("periodic-fixture", "qualification"),
    )
    task = phx.chemistry.BandStructureTaskPlan([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    calculation = phx.chemistry.ElectronicCalculationPlan(
        system,
        phx.chemistry.PeriodicElectronicSectorPlan(2.0),
        model,
        task,
    )
    provenance = _manifest("qualification-periodic-provider")
    capabilities = phx.chemistry.ElectronicProviderCapabilities(
        phx.chemistry.ElectronicTheoryCapabilities(
            (phx.chemistry.ElectronicMethodFamily.CUSTOM_EXTERNAL,),
            (phx.chemistry.ElectronicReferenceKind.RESTRICTED,),
        ),
        phx.chemistry.ElectronicGeometryCapabilities(
            finite=False, periodic_ranks=(3,), variable_cell=True
        ),
        phx.chemistry.ElectronicObservableCapabilities(
            (task.task_kind,),
            (phx.chemistry.ElectronicProperty.BAND_ENERGIES,),
            derivative_orders=(0,),
        ),
    )

    def evaluate(plan, positions, cell_vectors):
        return phx.chemistry.make_electronic_evaluation(
            plan,
            "qualification-periodic-provider",
            positions,
            -1.0,
            band_energies=[[-1.0, 0.5], [-0.8, 0.7]],
            cell_vectors=cell_vectors,
            artifact_ids=(provenance.manifest_id,),
        )

    result = (
        phx.chemistry.CallableElectronicProvider(
            evaluate, "qualification-periodic-provider", capabilities
        )
        .prepare(calculation)
        .evaluate(np.zeros((2, 3)), np.asarray(cell.vectors))
    )
    passed = (
        bool(result.successful)
        and result.header.provider_id == "qualification-periodic-provider"
        and result.header.task_id == task.task_id
        and result.header.artifact_ids == (provenance.manifest_id,)
    )
    return {
        "provider_id": result.header.provider_id,
        "task_id": result.header.task_id,
        "geometry_id": result.header.geometry_id,
        "artifact_ids": list(result.header.artifact_ids),
        "passed": passed,
    }


def qualification():
    positions = np.asarray([[0.2, 0.3, 0.4], [1.2, 0.3, 0.4]])
    neutral = (
        periodic.PeriodicEwaldPlan(
            _cell(), _units(), 0.8, real_shell=2, reciprocal_shell=2
        )
        .prepare()
        .evaluate(positions, [1.0, -1.0])
    )
    background = (
        periodic.PeriodicEwaldPlan(
            _cell(),
            _units(),
            0.8,
            real_shell=2,
            reciprocal_shell=2,
            neutrality="uniform-background",
        )
        .prepare()
        .evaluate(positions, [1.0, 0.0])
    )

    gth_manifest = _manifest("qualification-gth")
    gth = periodic.GTHPseudopotentialPlan(
        2.0,
        0.3,
        [0.1, -0.02, 0.0, 0.0],
        phx.units.ELECTRONVOLT,
        phx.units.ANGSTROM,
        gth_manifest,
    )
    local = gth.local_reciprocal([0.0, 1.0, 4.0])

    gdf_manifest = _manifest("qualification-gdf")
    gdf = periodic.GammaGDFPlan(
        [[-1.0]],
        [[1.0]],
        phx.operators.quantum.gaussian.FactorizedERITensor(
            [[[0.5]]], 0.0, gdf_manifest.source_id, "gdf"
        ),
        2.0,
        0.0,
        phx.units.ELECTRONVOLT,
        gdf_manifest,
    ).evaluate()
    fftdf = periodic.GammaFFTDFPlan(
        _cell(),
        (2, 2, 2),
        [[0.0, 0.0, 0.0]],
        (gth,),
        2.0,
        0.0,
        phx.units.ELECTRONVOLT,
        phx.units.ANGSTROM,
        smearing_energy=0.1,
        convergence_tolerance=1.0e-7,
        maximum_iterations=300,
        damping=0.5,
    ).evaluate()

    gw_manifest = _manifest("qualification-gw")
    gw = periodic.DiagonalGWPlan(
        [-0.5, 0.3],
        [0.0, 0.0],
        [[-0.6, -0.2], [0.2, 0.6]],
        lambda index, energy: 0.1 + 0.0 * index + 0.0 * energy,
        "constant-self-energy",
        "qualification-gw-provider",
        gw_manifest,
        phx.units.HARTREE,
    ).evaluate()
    dipole_unit = phx.units.derived_unit(
        "e*bohr-periodic-bse-qualification",
        ((phx.units.ELEMENTARY_CHARGE, 1), (phx.units.BOHR, 1)),
    )
    bse_manifest = _manifest("qualification-bse")
    bse = periodic.BetheSalpeterPlan(
        ("v0-c0",),
        [0.5],
        [[0.1]],
        [[0.0]],
        [[1.0, 0.0, 0.0]],
        -1.0,
        phx.units.HARTREE,
        dipole_unit,
        "qualification-bse-provider",
        bse_manifest,
    )
    tda = bse.tda(1)
    full = bse.full(1)

    cases = {
        "ewald-neutral": {
            "charge_residual": float(neutral.evidence.charge_residual),
            "force_balance_residual": float(neutral.evidence.force_balance_residual),
            "stress_symmetry_residual": float(neutral.evidence.stress_symmetry_residual),
            "passed": bool(neutral.successful)
            and not bool(neutral.evidence.background_applied),
        },
        "ewald-background": {
            "background_energy": float(background.energy_ledger.components[-1]),
            "charge_residual": float(background.evidence.charge_residual),
            "passed": bool(background.successful)
            and bool(background.evidence.background_applied),
        },
        "gth-components": {
            "source_manifest_id": local.evidence.source_manifest_id,
            "component_norms": np.asarray(local.evidence.component_norms).tolist(),
            "passed": bool(local.successful)
            and local.evidence.source_manifest_id == gth_manifest.manifest_id,
        },
        "spin-restricted-insulator": _spin_case("restricted", 0.0, 0.0),
        "spin-collinear-metal": _spin_case("collinear", 0.15, 0.4),
        "gamma-gdf-rhf": {
            "classification": gdf.classification,
            "electron_count_residual": float(gdf.evidence.electron_count_residual),
            "factorization_residual": float(gdf.evidence.factorization_residual),
            "passed": bool(gdf.successful)
            and gdf.classification == "production-supplied-gdf-rhf",
        },
        "gamma-local-gth-fftdf": {
            "classification": fftdf.classification,
            "electron_count_residual": float(fftdf.evidence.electron_count_residual),
            "passed": fftdf.classification == "candidate-local-gth-lda-x",
        },
        "stationary-force-stress": _derivative_case(),
        "supplied-diagonal-gw": {
            "provider_id": gw.provider_id,
            "maximum_root_residual": float(np.max(gw.evidence.root_residuals)),
            "source_manifest_id": gw.source_manifest_id,
            "passed": bool(gw.successful)
            and gw.source_manifest_id == gw_manifest.manifest_id,
        },
        "supplied-bse": {
            "tda_eigenpair_residual": float(tda.evidence.maximum_eigenpair_residual),
            "full_eigenpair_residual": float(full.evidence.maximum_eigenpair_residual),
            "transition_order_id": bse.transition_order_id,
            "source_manifest_id": tda.evidence.source_manifest_id,
            "passed": bool(tda.successful)
            and bool(full.successful)
            and tda.evidence.source_manifest_id == bse_manifest.manifest_id,
        },
        "generic-periodic-provider": _provider_case(),
    }
    return {
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "passed": all(bool(value["passed"]) for value in cases.values()),
        "nonclaims": [
            "qualification controls do not establish accuracy for an external material",
            "finite Ewald shells require separate convergence studies",
            "the FFTDF route is candidate-only local-GTH Dirac exchange",
            "GW and BSE are supplied-data postprocessors, not kernel generators",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/cm_periodic_electronic_qualification.json"),
    )
    arguments = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError("Periodic electronic qualification requires JAX_ENABLE_X64=1.")
    payload = qualification()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
