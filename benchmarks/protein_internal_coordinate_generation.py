# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fixed-construct protein coordinate generation comparison.

This executable uses an original synthetic cis-proline topology to compare the
preserved Cartesian flow, exact no-learning reconstruction, and the learned
periodic internal-coordinate flow. It qualifies proposal mechanics only: the
corpus is non-equilibrium, is not sequence-general, and generated frequencies
are not physical populations.

Run: python -m benchmarks.protein_internal_coordinate_generation --steps 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax import atomistic
from phydrax.applications.protein_folding import (
    bind_protein,
    PreparedProteinQualification,
    ProteinAtomKey,
    ProteinConstruct,
    ProteinSourceAtom,
    ProteinStructureHypothesis,
    ResolvedProteinChemistry,
)
from phydrax.applications.protein_folding.generation import (
    fit_coordinate_model,
    prepare_bound_protein_coordinate_generation,
    prepare_coordinate_sampler,
    prepare_coordinate_training_data,
    sample_coordinate_proposals,
    sample_protein_coordinate_proposals,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM


def _fixed_construct():
    construct = ProteinConstruct(("A",), ("AP",))
    alanine, proline = construct.residue_keys
    alanine_names = ("N", "CA", "C", "O", "CB", "H")
    proline_names = ("N", "CA", "C", "O", "OXT", "CB", "CG", "CD", "H")
    keys = tuple(ProteinAtomKey(alanine, name) for name in alanine_names) + tuple(
        ProteinAtomKey(proline, name) for name in proline_names
    )
    positions = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.6, 1.3, 0.0],
            [0.1, 2.4, 0.0],
            [0.2, -0.7, -1.0],
            [-1.7, 0.2, 0.4],
            [1.7, 1.3, 0.0],
            [2.3, 2.6, 0.0],
            [1.5, 3.7, 0.0],
            [0.3, 3.6, 0.0],
            [2.1, 4.8, 0.0],
            [3.5, 2.5, -0.7],
            [3.7, 1.3, -1.0],
            [2.7, 0.7, -0.5],
            [1.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    numbers = tuple(
        1 if key.atom_name == "H" else {"C": 6, "N": 7, "O": 8}[key.atom_name[0]]
        for key in keys
    )
    index = {key: row for row, key in enumerate(keys)}

    def atom(residue, name):
        return index[ProteinAtomKey(residue, name)]

    routes = np.asarray(
        [
            (atom(alanine, "N"), atom(alanine, "CA")),
            (atom(alanine, "CA"), atom(alanine, "C")),
            (atom(alanine, "C"), atom(alanine, "O")),
            (atom(alanine, "CA"), atom(alanine, "CB")),
            (atom(alanine, "N"), atom(alanine, "H")),
            (atom(alanine, "C"), atom(proline, "N")),
            (atom(proline, "N"), atom(proline, "CA")),
            (atom(proline, "CA"), atom(proline, "C")),
            (atom(proline, "C"), atom(proline, "O")),
            (atom(proline, "C"), atom(proline, "OXT")),
            (atom(proline, "CA"), atom(proline, "CB")),
            (atom(proline, "CB"), atom(proline, "CG")),
            (atom(proline, "CG"), atom(proline, "CD")),
            (atom(proline, "CD"), atom(proline, "N")),
            (atom(proline, "N"), atom(proline, "H")),
        ],
        dtype=np.int32,
    )
    payload = positions.tobytes() + routes.tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    rights = ReferenceArtifactManifest(
        "original-synthetic-cis-proline-generation-benchmark",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=len(payload),
        license_id="LicenseRef-Phydrax-OriginalSyntheticFixture",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted-original-synthetic-data",
        nondimensionalization={"coordinate_angstrom": 1.0},
        uncertainty=None,
        lineage_ids=("hand-authored-fixed-topology",),
    )
    source = ScientificArtifactEnvelope(
        artifact_kind="synthetic-protein-topology",
        content_digest=digest,
        producer="phydrax-benchmark",
        producer_version="native",
        build_id="cis-proline-v1",
        license_id=rights.license_id,
        resource_id="cis-proline-v1",
        status="complete",
    )
    chemistry = ResolvedProteinChemistry(
        construct,
        keys,
        numbers,
        ("standard", "standard"),
        (1, 1),
        "NH2",
        "COO-",
        source.artifact_id,
    )
    source_atoms = tuple(
        ProteinSourceAtom(
            str(row),
            key,
            "1",
            "A",
            str(key.residue.position + 1),
            "",
            "",
            1.0,
            number,
        )
        for row, (key, number) in enumerate(zip(keys, numbers, strict=True))
    )
    hypothesis = ProteinStructureHypothesis(
        construct, source_atoms, positions, ANGSTROM, source, (rights,)
    )
    ids = np.arange(len(keys), dtype=np.int64) * 17 + 101
    topology = atomistic.MolecularTopologyPlan(
        bonds=ids[routes], bond_type_ids=np.arange(len(routes))
    )
    units = atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    masses = np.asarray(
        [1.0 if z == 1 else 14.0 if z == 7 else 16.0 if z == 8 else 12.0 for z in numbers]
    )
    system = atomistic.AtomisticSystemPlan(
        ids,
        numbers,
        masses,
        units,
        topology=topology,
        molecule_ids=np.zeros(len(keys), dtype=int),
    )
    lengths = np.linalg.norm(positions[routes[:, 0]] - positions[routes[:, 1]], axis=-1)
    potential = atomistic.AtomisticPotentialProgram(
        [atomistic.HarmonicBondPotential(np.ones(len(routes)), lengths)]
    )
    force_field = atomistic.AtomisticForceFieldPlan(
        system,
        potential,
        atomistic.AtomisticNonbondedPolicy(8.0, electrostatics="direct"),
        atomistic.AtomisticForceFieldProvenance(
            "synthetic", (digest,), "analytic-harmonic", "explicit-unit-fixture"
        ),
    ).prepare()
    binding = bind_protein(
        hypothesis,
        chemistry,
        force_field,
        dict(zip(keys, map(int, ids), strict=True)),
        parameter_energy_unit=units.scale.energy_unit,
        parameter_rights=(rights,),
    )
    qualification = PreparedProteinQualification(
        binding,
        bond_bounds=np.stack((0.7 * lengths, 1.3 * lengths), axis=1),
        clash_distance=0.1,
        minimum_chiral_volume=0.01,
    )
    return binding, qualification, rights


def _prepare_corpora():
    binding, qualification, rights = _fixed_construct()
    support, decoder = prepare_bound_protein_coordinate_generation(
        binding, qualification, cis_peptide_indices=(0,)
    )
    reference = decoder.reference_periodic_coordinates().reshape((-1, 2))
    base_angles = jnp.arctan2(reference[:, 0], reference[:, 1])
    offsets = np.linspace(-0.45, 0.45, 16)
    periodic = jnp.stack(
        tuple(
            jnp.stack(
                (jnp.sin(base_angles + offset), jnp.cos(base_angles + offset)),
                axis=1,
            ).reshape((-1,))
            for offset in offsets
        )
    )
    positions = decoder.decode(periodic).positions
    groups = tuple("validation" if row % 4 == 0 else "training" for row in range(16))
    arguments = dict(
        condition_names=("declared_torsion_offset_radian",),
        record_ids=tuple(f"synthetic-{row}" for row in range(16)),
        source_manifest_ids=(rights.manifest_id,) * 16,
        split_group_ids=groups,
        validation_groups=("validation",),
        rights=(rights,),
    )
    internal = prepare_coordinate_training_data(
        support,
        positions,
        offsets[:, None],
        corpus_description=(
            "Original fixed-construct kinematic perturbations; non-equilibrium, "
            "not independent proteins, and not sequence-general."
        ),
        decoder=decoder,
        **arguments,
    )
    cartesian = prepare_coordinate_training_data(
        support,
        positions,
        offsets[:, None],
        corpus_description=(
            "Cartesian baseline over the same original non-equilibrium fixed construct."
        ),
        **arguments,
    )
    return binding, qualification, support, decoder, internal, cartesian


def _accepted_diverse_count(positions, accepted, threshold=0.05):
    rows = np.asarray(positions)[np.asarray(accepted, dtype=bool)]
    representatives = []
    for row in rows:
        if all(
            np.sqrt(np.mean((row - prior) ** 2)) >= threshold for prior in representatives
        ):
            representatives.append(row)
    return len(representatives)


def _full_evidence(qualification, positions):
    return jax.vmap(qualification.evaluate)(positions)


def _failure_counts(batch):
    failures = batch.failures
    rows = (
        ("raw_nonfinite", failures.raw_nonfinite),
        ("solver_failed", failures.solver_failed),
        (
            "periodic_representation_failed",
            failures.periodic_representation_failed,
        ),
        ("closure_failed", failures.closure_failed),
        ("decoded_nonfinite", failures.decoded_nonfinite),
        ("sparse_gauge_failed", failures.sparse_gauge_failed),
        ("sparse_bond_failed", failures.sparse_bond_failed),
        ("sparse_chirality_failed", failures.sparse_chirality_failed),
        ("full_bond_failed", failures.full_bond_failed),
        ("full_chirality_failed", failures.full_chirality_failed),
        ("full_clash_failed", failures.full_clash_failed),
        ("full_peptide_failed", failures.full_peptide_failed),
        ("full_torsion_failed", failures.full_torsion_failed),
    )
    return {name: int(jnp.sum(values)) for name, values in rows}


def run(*, steps=50, samples=16, repeats=3):
    prepared, preparation_seconds = measure_synchronized(_prepare_corpora)
    binding, qualification, support, decoder, internal_data, cartesian_data = prepared
    cartesian_fit, cartesian_fit_seconds = measure_synchronized(
        lambda: fit_coordinate_model(
            cartesian_data,
            key=jr.key(101),
            steps=steps,
            pairs_per_step=16,
            width=24,
            depth=2,
        )
    )
    internal_fit, internal_fit_seconds = measure_synchronized(
        lambda: fit_coordinate_model(
            internal_data,
            key=jr.key(102),
            steps=steps,
            pairs_per_step=16,
            width=24,
            depth=2,
        )
    )
    conditions = jnp.linspace(-0.4, 0.4, samples)[:, None]
    cartesian_sampler = prepare_coordinate_sampler(cartesian_fit)
    internal_sampler = prepare_coordinate_sampler(internal_fit)
    compiled_cartesian, cartesian_compilation = measure_lower_and_compile(
        lambda: eqx.filter_jit(cartesian_sampler).lower(jr.key(201), conditions),
        lambda lowered: lowered.compile(),
    )
    compiled_internal, internal_compilation = measure_lower_and_compile(
        lambda: eqx.filter_jit(internal_sampler).lower(jr.key(202), conditions),
        lambda lowered: lowered.compile(),
    )
    _, cartesian_solver_timing = measure_repeated(
        lambda: compiled_cartesian(jr.key(201), conditions), warmup=1, repeats=repeats
    )
    _, internal_solver_timing = measure_repeated(
        lambda: compiled_internal(jr.key(202), conditions), warmup=1, repeats=repeats
    )
    cartesian_batch, cartesian_end_to_end = measure_repeated(
        lambda: sample_coordinate_proposals(cartesian_fit, jr.key(201), conditions),
        warmup=1,
        repeats=repeats,
    )
    internal_batch, internal_end_to_end = measure_repeated(
        lambda: sample_protein_coordinate_proposals(
            internal_fit, jr.key(202), conditions, qualification
        ),
        warmup=1,
        repeats=repeats,
    )
    reconstruction, reconstruction_seconds = measure_synchronized(
        lambda: decoder.reconstruct(binding.realized_positions)
    )
    _, reconstructed = reconstruction
    reconstruction_full = qualification.evaluate(reconstructed.positions)
    cartesian_full = _full_evidence(qualification, cartesian_batch.canonical_positions)
    reference_pairs = decoder.reference_periodic_coordinates().reshape((-1, 2))
    reference_angles = jnp.arctan2(reference_pairs[:, 0], reference_pairs[:, 1])
    target_periodic = jnp.stack(
        (
            jnp.sin(reference_angles[None, :] + conditions),
            jnp.cos(reference_angles[None, :] + conditions),
        ),
        axis=-1,
    ).reshape((samples, decoder.coordinate_size))
    target_positions = decoder.decode(target_periodic).positions
    target_positions, _ = support.canonicalize(target_positions)
    cartesian_rmse = jnp.sqrt(
        jnp.mean(
            (cartesian_batch.canonical_positions - target_positions) ** 2, axis=(1, 2)
        )
    )
    internal_rmse = jnp.sqrt(
        jnp.mean(
            (internal_batch.decoded_coordinates - target_positions) ** 2, axis=(1, 2)
        )
    )
    cartesian_success = cartesian_batch.qualification.accepted & cartesian_full.successful
    internal_success = internal_batch.failures.successful
    cartesian_diverse = _accepted_diverse_count(
        cartesian_batch.canonical_positions, cartesian_success
    )
    internal_diverse = _accepted_diverse_count(
        internal_batch.decoded_coordinates, internal_success
    )
    cartesian_median = cartesian_end_to_end.median_seconds
    internal_median = internal_end_to_end.median_seconds
    return {
        "claim": (
            "fixed-construct proposal mechanics only; non-equilibrium, not "
            "sequence-general, and no physical-population interpretation"
        ),
        "environment": capture_environment().to_dict(),
        "source_manifest_id": internal_data.rights[0].manifest_id,
        "support_id": support.support_id,
        "internal_coordinate_plan_id": decoder.plan.plan_id,
        "qualification_id": qualification.qualification_id,
        "preparation_seconds": preparation_seconds,
        "prepared_logical_array_bytes": logical_array_bytes(prepared),
        "retained_requested_samples": samples,
        "no_learning_reconstruction": {
            "seconds": reconstruction_seconds,
            "decoder_valid": bool(reconstructed.valid),
            "closure_valid": bool(jnp.all(reconstructed.closure.valid)),
            "full_geometry_valid": bool(reconstruction_full.successful),
        },
        "cartesian_baseline": {
            "dataset_id": cartesian_data.dataset_id,
            "fit_id": cartesian_fit.fit_id,
            "fit_seconds_including_compilation": cartesian_fit_seconds,
            "model_logical_array_bytes": logical_array_bytes(cartesian_fit.model),
            "sampling_lowering_seconds": cartesian_compilation.lowering_seconds,
            "sampling_compilation_seconds": cartesian_compilation.compilation_seconds,
            "solver_steady_seconds": cartesian_solver_timing.to_seconds_dict(),
            "end_to_end_steady_seconds": cartesian_end_to_end.to_seconds_dict(),
            "full_geometry_pass_count": int(jnp.sum(cartesian_success)),
            "accepted_diverse_count": cartesian_diverse,
            "mean_proper_gauge_rmse_angstrom": float(jnp.mean(cartesian_rmse)),
            "cost_per_accepted_diverse_seconds": (
                None if not cartesian_diverse else cartesian_median / cartesian_diverse
            ),
        },
        "periodic_internal_model": {
            "dataset_id": internal_data.dataset_id,
            "fit_id": internal_fit.fit_id,
            "fit_seconds_including_compilation": internal_fit_seconds,
            "model_logical_array_bytes": logical_array_bytes(internal_fit.model),
            "sampling_lowering_seconds": internal_compilation.lowering_seconds,
            "sampling_compilation_seconds": internal_compilation.compilation_seconds,
            "solver_steady_seconds": internal_solver_timing.to_seconds_dict(),
            "end_to_end_steady_seconds": internal_end_to_end.to_seconds_dict(),
            "construction_valid_count": int(
                jnp.sum(internal_batch.decoder_evidence.valid)
            ),
            "closure_valid_count": int(
                jnp.sum(jnp.all(internal_batch.decoder_evidence.closure.valid, axis=1))
            ),
            "full_geometry_pass_count": int(jnp.sum(internal_success)),
            "accepted_diverse_count": internal_diverse,
            "mean_proper_gauge_rmse_angstrom": float(jnp.mean(internal_rmse)),
            "failure_counts": _failure_counts(internal_batch),
            "cost_per_accepted_diverse_seconds": (
                None if not internal_diverse else internal_median / internal_diverse
            ),
        },
        "physical_energy_evidence": None,
        "hidden_minimization_or_resampling": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    evidence = run(
        steps=arguments.steps, samples=arguments.samples, repeats=arguments.repeats
    )
    rendered = json.dumps(evidence, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
