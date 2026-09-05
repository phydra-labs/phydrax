# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Actual native coordinate training on the admitted 1L2Y NMR-model snapshot.

This is SAME-CONSTRUCT RECONSTRUCTION, not independent predictor qualification.
The 38 deposited models share one NMR experiment/refinement and are correlated;
model-index holdout below is only a numerical reconstruction diagnostic. Models
are not thermodynamic populations. Hydrogen omission is an explicit heavy-atom
model ABI, not chemical completion or a parameterized force-field realization.
No download occurs. Supply the source-pinned local PDB and its acquisition JSON.

python -m benchmarks.biophysical_coordinate_generation_pdb \
    --pdb .tmp/biophysical-inputs/1L2Y.pdb \
    --source .tmp/biophysical-inputs/1L2Y.source.json --steps 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import equinox as eqx
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
from phydrax.applications.protein_folding._construct import (
    ProteinAtomKey,
    ProteinConstruct,
)
from phydrax.applications.protein_folding._hypotheses import ProteinSourceAtom
from phydrax.applications.protein_folding.generation import (
    CoordinateGeometryPolicy,
    CoordinateProviderProvenance,
    fit_coordinate_model,
    import_protein_hypotheses,
    map_protein_hypothesis,
    prepare_coordinate_sampler,
    prepare_coordinate_training_data,
    prepare_protein_coordinate_support,
    qualify_coordinate_proposals,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.atomistic import AtomisticBatch, AtomisticScaleContract
from phydrax.atomistic.interchange import read_pdb_atom_records, select_pdb_model
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM, ELECTRONVOLT


_PINNED_SHA256 = "5d1bbb545a312dfff1ae1e64b6d8addecb2f561ddc4011aeb5bee9d1dfcd4438"
_SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"
_RESIDUES = (
    "ASN",
    "LEU",
    "TYR",
    "ILE",
    "GLN",
    "TRP",
    "LEU",
    "LYS",
    "ASP",
    "GLY",
    "GLY",
    "PRO",
    "SER",
    "SER",
    "GLY",
    "ARG",
    "PRO",
    "PRO",
    "PRO",
    "SER",
)


def prepare_pdb_reconstruction_data(pdb_path, source_path):
    """Consume neutral records with an exact, source-pinned 1L2Y identity profile."""
    payload = Path(pdb_path).read_bytes()
    source = json.loads(Path(source_path).read_text())
    digest = hashlib.sha256(payload).hexdigest()
    if (
        digest != _PINNED_SHA256
        or source["sha256"] != digest
        or source["size_bytes"] != len(payload)
    ):
        raise ValueError(
            "This named reconstruction profile requires the exact pinned 1L2Y snapshot."
        )
    if source["archive_id"] != "1L2Y" or source["license_id"] != "CC0-1.0":
        raise ValueError(
            "The reconstruction profile requires admitted wwPDB core coordinate provenance."
        )
    rights = ReferenceArtifactManifest(
        "wwPDB-1L2Y-NMR-models",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=len(payload),
        license_id=source["license_id"],
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="CC0-core-coordinate-data",
        nondimensionalization={"coordinate_angstrom": 1.0, "condition_kelvin": 282.0},
        uncertainty=None,
        lineage_ids=(source["source_url"], source["license_url"], source["citation"]),
    )
    records = read_pdb_atom_records(
        payload.decode("ascii"), source_id="wwPDB:1L2Y:" + digest
    )
    model_ids = tuple(dict.fromkeys(record.model_id for record in records))
    if len(model_ids) != 38:
        raise ValueError("The admitted snapshot must retain all 38 raw deposited models.")
    construct = ProteinConstruct(("A",), (_SEQUENCE,))
    model_rows = []
    for model_id in model_ids:
        rows = select_pdb_model(records, model_id, alternate_locations={})
        selected = tuple(row for row in rows if row.element != "H")
        for row in selected:
            position = int(row.author_residue_number) - 1
            if (
                row.record_kind != "ATOM"
                or row.chain_id != "A"
                or row.insertion_code
                or not 0 <= position < 20
            ):
                raise ValueError(
                    "Source atoms lie outside the explicitly supported TC5B construct profile."
                )
            if row.residue_name != _RESIDUES[position] or row.element not in (
                "C",
                "N",
                "O",
            ):
                raise ValueError(
                    "Source residue/element differs from the admitted heavy-atom profile."
                )
        model_rows.append(selected)

    def atom_key(row):
        return ProteinAtomKey(
            construct.residue_keys[int(row.author_residue_number) - 1], row.atom_name
        )

    first = model_rows[0]
    # Stable IDs come from an explicit atom-key binding, never sequence indices.
    atom_ids = {atom_key(row): 10000 + int(row.atom_serial) for row in first}
    positions = np.asarray([row.position for row in first])
    numbers = {"C": 6, "N": 7, "O": 8}
    # Nominal element mass numbers are DECLARED centering weights only. There is
    # no physical MD/force-field mass calibration in this reconstruction model.
    weights = {"C": 12.0, "N": 14.0, "O": 16.0}
    template = AtomisticBatch(
        np.array([[numbers[row.element] for row in first]]),
        positions[None],
        np.array([[weights[row.element] for row in first]]),
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        particle_ids=np.array([[atom_ids[atom_key(row)] for row in first]]),
    )

    def atom(residue, name):
        return atom_ids[ProteinAtomKey(residue, name)]

    bonds, chiral = [], []
    for residue, amino in zip(construct.residue_keys, _SEQUENCE, strict=True):
        bonds.extend(
            (
                (atom(residue, "N"), atom(residue, "CA")),
                (atom(residue, "CA"), atom(residue, "C")),
                (atom(residue, "C"), atom(residue, "O")),
            )
        )
        if amino != "G":
            bonds.append((atom(residue, "CA"), atom(residue, "CB")))
            chiral.append(tuple(atom(residue, name) for name in ("CA", "N", "C", "CB")))
    for left, right in zip(
        construct.residue_keys[:-1], construct.residue_keys[1:], strict=True
    ):
        bonds.append((atom(left, "C"), atom(right, "N")))
    geometry = CoordinateGeometryPolicy(
        tuple(bonds),
        ((0.9, 1.9),) * len(bonds),
        tuple(chiral),
        (1,) * len(chiral),
        0.1,
        "broad-backbone-and-alpha-chirality-screen-not-full-physical-qualification",
    )
    first_residue = construct.residue_keys[0]
    support = prepare_protein_coordinate_support(
        construct,
        template,
        atom_ids,
        gauge_atom_ids=tuple(atom(first_residue, name) for name in ("CA", "N", "C")),
        geometry=geometry,
    )
    provenance = CoordinateProviderProvenance(
        "wwPDB-experimental-coordinate-models", (rights,)
    )
    mapped = []
    for model_id, rows in zip(model_ids, model_rows, strict=True):
        source_rows = tuple(
            ProteinSourceAtom(
                row.record_id,
                atom_key(row),
                row.model_id,
                row.chain_id,
                row.author_residue_number,
                row.insertion_code,
                row.alternate_location,
                row.occupancy,
                numbers[row.element],
            )
            for row in rows
        )
        envelope = ScientificArtifactEnvelope(
            artifact_kind="raw-wwPDB-NMR-coordinate-model",
            content_digest=digest,
            producer="wwPDB",
            producer_version="pinned-1L2Y-snapshot",
            build_id=digest,
            license_id=rights.license_id,
            resource_id="1L2Y:model:" + model_id,
            status="complete",
        )
        imported = import_protein_hypotheses(
            construct,
            source_rows,
            np.asarray([row.position for row in rows])[None],
            ANGSTROM,
            (envelope,),
            provenance=provenance,
        )
        mapped.append(
            map_protein_hypothesis(
                imported.hypotheses[0], support, atom_ids, training_use=True
            )
        )
    # Disjoint MODEL IDs, NOT independent experiments; this diagnostic grouping
    # cannot qualify generalization and is explicitly recorded as correlated.
    groups = tuple(
        "diagnostic-heldout-models" if i % 5 == 0 else "fit-models"
        for i in range(len(model_ids))
    )
    return prepare_coordinate_training_data(
        support,
        np.stack(mapped),
        np.ones((len(model_ids), 1)),
        condition_names=("NMR_temperature_kelvin_over_282",),
        record_ids=tuple("1L2Y:model:" + i for i in model_ids),
        source_manifest_ids=(rights.manifest_id,) * len(model_ids),
        split_group_ids=groups,
        validation_groups=("diagnostic-heldout-models",),
        rights=(rights,),
        corpus_description=(
            "Pinned CC0 wwPDB 1L2Y heavy-atom NMR coordinates at 282 K; "
            "correlated model-index reconstruction holdout, NOT independent "
            "predictor validation. No thermodynamic populations or physical force field."
        ),
    )


def run(pdb_path, source_path, *, steps=200, repeats=3):
    data = prepare_pdb_reconstruction_data(pdb_path, source_path)
    fit, fit_seconds = measure_synchronized(
        lambda: fit_coordinate_model(
            data,
            key=jr.key(31),
            steps=steps,
            pairs_per_step=16,
            width=64,
            depth=2,
            learning_rate=2e-3,
        )
    )
    sampler = prepare_coordinate_sampler(fit)
    conditions = jnp.ones((8, 1))
    compiled, compilation = measure_lower_and_compile(
        lambda: eqx.filter_jit(sampler).lower(jr.key(32), conditions),
        lambda lowered: lowered.compile(),
    )
    result, timing = measure_repeated(
        lambda: compiled(jr.key(32), conditions), warmup=1, repeats=repeats
    )
    raw, valid, status = result
    canonical, gauge_valid = data.support.canonicalize(raw)
    qualification = qualify_coordinate_proposals(
        data.support, canonical, solver_valid=valid & gauge_valid
    )
    reference = data.canonical_positions[jnp.array(data.validation_indices)]
    pair_mse = jnp.mean((canonical[:, None] - reference[None]) ** 2, axis=(-1, -2))
    return {
        "claim": "same-construct reconstruction only; correlated NMR-model holdout, not corpus-qualified prediction",
        "source_sha256": _PINNED_SHA256,
        "source_rights": "CC0-1.0",
        "temperature_kelvin": 282.0,
        "all_raw_models_retained": len(data.record_ids),
        "fit_models": len(data.train_indices),
        "diagnostic_heldout_models": len(data.validation_indices),
        "active_atoms": data.support.template.atom_capacity,
        "atom_profile": "explicit heavy-atom model support; hydrogens omitted, not imputed",
        "centering_weights": "nominal C/N/O mass numbers; no physical-force-field calibration",
        "training_steps": steps,
        "fit_seconds_including_training_compilation": fit_seconds,
        "sampling_lowering_seconds": compilation.lowering_seconds,
        "sampling_compilation_seconds": compilation.compilation_seconds,
        "sampling_steady_seconds": timing.to_dict(),
        "model_logical_array_bytes": logical_array_bytes(fit.model),
        "initial_training_loss": fit.initial_training_loss,
        "final_training_loss": fit.final_training_loss,
        "correlated_holdout_flow_loss": fit.validation_loss,
        "nearest_correlated_holdout_coordinate_rmse_angstrom": np.sqrt(
            np.asarray(jnp.min(pair_mse, axis=1))
        ).tolist(),
        "solver_status": np.asarray(status).tolist(),
        "all_samples_retained": len(raw),
        "geometry_accepted_fraction": float(jnp.mean(qualification.accepted)),
        "confidence": "uncalibrated",
        "Boltzmann_weights": "not available",
        "likelihood": "not available",
        "open_gates": [
            "independent multi-construct corpus",
            "qualified chemistry and physical geometry coverage",
            "compute campaign",
            "independent predictive/confidence calibration",
        ],
        "environment": capture_environment().to_dict(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.pdb, args.source, steps=args.steps, repeats=args.repeats), indent=2
        )
    )
