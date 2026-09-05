# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Actual 1L2Y/ff14SB→native handoff and short NVE benchmark.

Run from the worktree: .venv/bin/python benchmarks/protein_binding.py
Raw 38-model CC0 PDB must already exist; no download or parameter generation is
hidden in runtime APIs. Model 1 is a structural fixture, never a population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

from phydrax import atomistic, discretization
from phydrax.applications.protein_folding import (
    PreparedProteinQualification,
    ProteinConstruct,
    ResidueKey,
    ResolvedProteinChemistry,
)
from phydrax.applications.protein_folding.interchange import (
    bind_protein_openmm,
    protein_hypothesis_from_pdb_records,
)
from phydrax.applications.protein_folding.workflows import run_protein_dynamics
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.atomistic.interchange._structure_records import (
    read_pdb_atom_records,
    select_pdb_model,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import KILOJOULE_PER_MOLE


_PDB_SHA256 = "5d1bbb545a312dfff1ae1e64b6d8addecb2f561ddc4011aeb5bee9d1dfcd4438"


def _rights(name, data, license_id, lineage, *, uncertainty):
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        license_id=license_id,
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"source_length_angstrom": 1.0},
        uncertainty=uncertainty,
        lineage_ids=lineage,
    )


def prepare_fixture(path):
    import openmm
    from openmm import app, unit

    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _PDB_SHA256:
        raise ValueError(
            "This source-pinned benchmark requires the admitted 1L2Y PDB artifact."
        )
    source = ScientificArtifactEnvelope(
        artifact_kind="pdb-source",
        content_digest=_PDB_SHA256,
        producer="wwPDB",
        producer_version="PDB-source-file",
        build_id="1L2Y",
        license_id="CC0-1.0",
        resource_id="https://files.rcsb.org/download/1L2Y.pdb",
        status="complete",
    )
    source_rights = _rights(
        "wwPDB-1L2Y",
        raw,
        "CC0-1.0",
        ("https://www.wwpdb.org/about/usage", source.artifact_id),
        uncertainty=None,
    )
    records = read_pdb_atom_records(raw.decode("ascii"), source_id=source.artifact_id)
    selected = select_pdb_model(records, "1", alternate_locations={})
    pdb = app.PDBFile(str(path))
    atoms = tuple(pdb.topology.atoms())
    by_serial = {atom.id: atom for atom in atoms}
    if len(by_serial) != len(atoms) or set(by_serial) != {
        row.atom_serial for row in selected
    }:
        raise ValueError(
            "OpenMM did not preserve a bijective source-serial identity map."
        )
    construct = ProteinConstruct(("A",), ("NLYIQWLKDGGPSSGRPPPS",))
    residue_map = {
        ("A", str(index + 1), ""): ResidueKey("A", index)
        for index in range(construct.residue_count)
    }
    hypothesis = protein_hypothesis_from_pdb_records(
        selected,
        construct,
        residue_map,
        source=source,
        rights=(source_rights,),
        canonical_atom_names={
            row.record_id: by_serial[row.atom_serial].name for row in selected
        },
    )
    states = tuple(
        "protonated"
        if letter in "KR"
        else "deprotonated"
        if letter in "DE"
        else "standard"
        for letter in construct.sequences[0]
    )
    hydrogens = tuple(
        sum(
            row.element == 1 and row.atom_key.residue == key
            for row in hypothesis.source_atoms
        )
        for key in construct.residue_keys
    )
    chemistry = ResolvedProteinChemistry(
        construct,
        tuple(row.atom_key for row in hypothesis.source_atoms),
        tuple(row.element for row in hypothesis.source_atoms),
        states,
        hydrogens,
        "NH3+",
        "COO-",
        source.artifact_id,
    )
    parameter_path = Path(app.__file__).parent / "data" / "amber14" / "protein.ff14SB.xml"
    parameter_bytes = parameter_path.read_bytes()
    parameter_rights = _rights(
        "Amber14-protein-ff14SB",
        parameter_bytes,
        "public-domain-force-field-data",
        ("https://ambermd.org/AmberModels.php", "https://ambermd.org/index.php"),
        uncertainty=None,
    )
    forcefield = app.ForceField("amber14/protein.ff14SB.xml")
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False
    )
    units = atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    row_by_serial = {row.atom_serial: row for row in selected}
    converted = bind_protein_openmm(
        hypothesis,
        chemistry,
        system,
        units,
        source_record_ids_by_particle=tuple(
            row_by_serial[atom.id].record_id for atom in atoms
        ),
        parameter_rights=(parameter_rights,),
        source_id=parameter_rights.manifest_id,
        cutoff=100.0,
        accept_bounded_no_cutoff=True,
    )
    binding = converted.binding
    n = binding.force_field.system.capacity
    neighborhood = discretization.DenseParticleNeighborhoodPlan(n * (n - 1) // 2).prepare(
        binding.force_field.system.particles
    )
    qualification = PreparedProteinQualification(
        binding,
        bond_bounds=np.tile(
            [0.7, 2.2], (binding.force_field.system.topology.bond_indices.shape[0], 1)
        ),
        clash_distance=0.5,
        peptide_tolerance=0.45,
    )
    # Compare the same finite configuration; the bounded-LJ caveat remains in
    # the output. Its 100 Å cutoff exceeds this isolated 20-residue fixture.
    reference_integrator = openmm.VerletIntegrator(0.001 * unit.femtosecond)
    context = openmm.Context(
        system, reference_integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy_factor = atomistic.molar_energy_to_single_system_factor(
        KILOJOULE_PER_MOLE, units.scale.energy_unit, constant_set_id=units.constant_set_id
    )
    reference_energy = (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * energy_factor
    )
    reference_forces = (
        np.asarray(
            state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.angstrom
            )
        )
        * energy_factor
    )
    return (
        converted,
        neighborhood,
        qualification,
        reference_energy,
        reference_forces,
        len({row.model_id for row in records}),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "tests/fixtures/protein_folding/1L2Y.pdb",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    prepared, preparation_seconds = measure_synchronized(
        lambda: prepare_fixture(args.pdb)
    )
    (
        converted,
        neighborhood,
        qualification,
        reference_energy,
        reference_forces,
        retained_models,
    ) = prepared
    binding = converted.binding
    positions = binding.realized_positions
    neighbors = neighborhood.build(positions)
    execute = jax.jit(lambda x: binding.evaluate(neighbors, x))
    compiled, compilation = measure_lower_and_compile(
        lambda: execute.lower(positions), lambda lowered: lowered.compile()
    )
    evaluated, steady = measure_repeated(
        lambda: compiled(positions), warmup=1, repeats=args.repeats
    )
    workflow, dynamics_seconds = measure_synchronized(
        lambda: run_protein_dynamics(
            binding,
            neighborhood,
            atomistic.VelocityVerletPlan(0.05),
            qualification,
            velocity=jnp.zeros_like(positions),
            velocity_unit=binding.force_field.system.plan.units.velocity_unit,
            key=jax.random.key(710),
            step_count=args.steps,
        )
    )
    energies = np.asarray(workflow.rollout.trajectory.energies)[:, 2]
    report = {
        "source": "wwPDB-1L2Y-model-1",
        "parameter_artifact": "Amber14-protein-ff14SB",
        "claim": "caller-parameterized-native-handoff-not-folding-accuracy",
        "retained_raw_models": retained_models,
        "capacity": binding.force_field.system.capacity,
        "active_atoms": binding.coverage.required_count,
        "preparation_seconds": preparation_seconds,
        "compilation": asdict(compilation),
        "execution_seconds": steady.to_dict(),
        "memory": logical_array_bytes(binding.force_field),
        "compiler": asdict(
            compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="jax-compiled",
            )
        ),
        "reference_energy_abs_error_eV": abs(float(evaluated.energy) - reference_energy),
        "reference_force_max_abs_error_eV_per_angstrom": float(
            np.max(np.abs(np.asarray(evaluated.forces) - reference_forces))
        ),
        "native_energy_eV": float(evaluated.energy),
        "native_force_successful": bool(evaluated.successful),
        "short_nve_successful": bool(workflow.rollout.successful),
        "geometry_successful": bool(workflow.final_geometry.successful),
        "nve_total_energy_range_eV": float(np.ptp(energies)),
        "dynamics_seconds": dynamics_seconds,
        "interchange_warnings": converted.interchange_report.warnings,
        "raw_artifact": binding.hypothesis.source.artifact_id,
        "parameterized_artifact": binding.artifact.artifact_id,
        "trajectory_artifact": workflow.artifact.artifact_id,
        "environment": asdict(capture_environment()),
    }
    print(json.dumps(report, indent=2))
    if (
        not bool(evaluated.successful)
        or not bool(workflow.rollout.successful)
        or not bool(workflow.final_geometry.successful)
    ):
        raise RuntimeError(
            "Real parameterized protein workflow failed; inspect emitted evidence."
        )


if __name__ == "__main__":
    main()
