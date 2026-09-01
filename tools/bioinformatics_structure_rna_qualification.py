#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Macromolecular lowering and exact pseudoknot-free RNA qualification."""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics import rna, structure
from tools.bioinformatics_common_qualification import (
    emit_report,
    external_dataset_campaign,
    fingerprint,
    method_contract_evidence,
    qualification_report,
)


def _water_record() -> structure.MacromolecularRecord:
    return structure.MacromolecularRecord(
        "water",
        entities=(structure.EntityRecord("water", structure.EntityKind.WATER),),
        chains=(structure.ChainRecord("A", "A", "water"),),
        residues=(structure.ResidueRecord(0, "HOH", "HOH", 1, 1),),
        atoms=(
            structure.AtomRecord(
                "1",
                0,
                1,
                "O",
                "O",
                "O",
                8,
                (0.0, 0.0, 0.0),
            ),
            structure.AtomRecord(
                "2",
                0,
                1,
                "H1",
                "H1",
                "H",
                1,
                (0.96, 0.0, 0.0),
            ),
        ),
        experimental_method="unit fixture",
    )


def _structure_lowering_case() -> dict[str, object]:
    record = _water_record()
    plan = structure.StructureLoweringPlan.for_record(record)
    result = structure.lower_macromolecular_record(record, plan)
    expected_positions = np.asarray(((0.96, 0.0, 0.0), (0.0, 0.0, 0.0)))
    expected_atomic_numbers = np.asarray((1, 8))
    if result.structure is None:
        coordinate_error = math.inf
        atomic_number_match = False
    else:
        coordinate_error = float(
            np.max(
                np.abs(np.asarray(result.structure.positions[0, :2]) - expected_positions)
            )
        )
        atomic_number_match = np.array_equal(
            np.asarray(result.structure.atomic_numbers[:2]),
            expected_atomic_numbers,
        )

    insufficient_plan = structure.StructureLoweringPlan(
        atom_capacity=1,
        residue_capacity=1,
        chain_capacity=1,
        model_capacity=1,
        bond_capacity=0,
    )
    insufficient = structure.lower_macromolecular_record(record, insufficient_plan)
    capacity_rejected = (
        not bool(np.asarray(insufficient.valid))
        and insufficient.structure is None
        and int(np.asarray(insufficient.status)) != 0
        and int(np.asarray(insufficient.evidence[0])) == 2
    )
    contract = method_contract_evidence(result.method_contract)
    inputs = {
        "record_id": record.record_id,
        "plan_id": plan.plan_id,
        "positions": [atom.position for atom in record.atoms],
        "atomic_numbers": [atom.atomic_number for atom in record.atoms],
    }
    return {
        "scope": "unit_qualification",
        "oracle": "lossless identity-sorted host-to-numeric coordinate lowering",
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "record_id": record.record_id,
        "plan_id": plan.plan_id,
        "maximum_coordinate_error_angstrom": coordinate_error,
        "atomic_numbers_match": atomic_number_match,
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "capacity_check": {
            "configured_atom_capacity": 1,
            "required_atom_capacity": int(np.asarray(insufficient.evidence[0])),
            "status": int(np.asarray(insufficient.status)),
            "rejected": capacity_rejected,
        },
        "passed": bool(
            np.asarray(result.valid)
            and result.structure is not None
            and coordinate_error <= 2.0e-7
            and atomic_number_match
            and capacity_rejected
        ),
    }


def _brute_rna_partition(
    sequence_codes: tuple[int, ...],
    pair_energies: np.ndarray,
    allowed_pairs: np.ndarray,
    unpaired_energies: np.ndarray,
    thermal_energy: float,
    minimum_hairpin_length: int,
) -> float:
    """Recursively sum every noncrossing partial matching."""

    @lru_cache(maxsize=None)
    def interval(start: int, stop: int) -> float:
        if start == stop:
            return 1.0
        unpaired = math.exp(
            -float(unpaired_energies[sequence_codes[start]]) / thermal_energy
        ) * interval(start + 1, stop)
        total = unpaired
        for partner in range(start + 1, stop):
            if partner - start - 1 < minimum_hairpin_length:
                continue
            left_code = sequence_codes[start]
            right_code = sequence_codes[partner]
            if not bool(allowed_pairs[left_code, right_code]):
                continue
            weight = math.exp(
                -float(pair_energies[left_code, right_code]) / thermal_energy
            )
            total += weight * interval(start + 1, partner) * interval(partner + 1, stop)
        return total

    return interval(0, len(sequence_codes))


def _rna_partition_case() -> dict[str, object]:
    model = rna.nussinov_energy_model(
        pair_energy=-0.8,
        wobble_energy=-0.3,
        unpaired_energy=0.1,
        temperature=300.0,
        minimum_hairpin_length=0,
    )
    sequence_values = (0, 3, 1, 2)
    sequence_codes = jnp.asarray(sequence_values, dtype=jnp.int32)
    result = rna.partition_function(sequence_codes, model)
    oracle_partition = _brute_rna_partition(
        sequence_values,
        np.asarray(model.pair_energies),
        np.asarray(model.allowed_pairs),
        np.asarray(model.unpaired_energies),
        float(np.asarray(model.thermal_energy)),
        model.minimum_hairpin_length,
    )
    oracle_log_partition = math.log(oracle_partition)
    observed_log_partition = float(np.asarray(result.log_partition))
    partition_error = abs(observed_log_partition - oracle_log_partition)

    def objective(pair_energies):
        varied = eqx.tree_at(
            lambda candidate: candidate.pair_energies,
            model,
            pair_energies,
        )
        return rna.rna_log_partition(sequence_codes, varied)

    automatic_gradient = jax.grad(objective)(model.pair_energies)
    expected_gradient = jnp.zeros_like(model.pair_energies)
    for left in range(len(sequence_values)):
        for right in range(left + 1, len(sequence_values)):
            expected_gradient = expected_gradient.at[
                sequence_values[left], sequence_values[right]
            ].add(-result.pair_marginals[left, right] / model.thermal_energy)
    gradient_error = float(
        np.max(np.abs(np.asarray(automatic_gradient - expected_gradient)))
    )

    invalid = rna.partition_function(
        jnp.asarray((0, model.alphabet_size), dtype=jnp.int32), model
    )
    invalid_rejected = (
        not bool(np.asarray(invalid.valid)) and int(np.asarray(invalid.status)) != 0
    )
    contract = method_contract_evidence(result.method_contract)
    inputs = {
        "sequence_codes": sequence_codes,
        "pair_energies": model.pair_energies,
        "allowed_pairs": model.allowed_pairs,
        "unpaired_energies": model.unpaired_energies,
        "temperature": model.temperature,
        "gas_constant": model.gas_constant,
        "minimum_hairpin_length": model.minimum_hairpin_length,
    }
    return {
        "scope": "unit_qualification",
        "oracle": "recursive enumeration of every noncrossing partial matching",
        "gradient_oracle": (
            "d(log Z)/d(pair energy) equals negative expected pair count "
            "divided by thermal energy"
        ),
        "input_fingerprint": fingerprint(inputs),
        "method_fingerprint": contract["fingerprint"],
        "method": contract,
        "observed_log_partition": observed_log_partition,
        "oracle_log_partition": oracle_log_partition,
        "absolute_log_partition_error": partition_error,
        "maximum_gradient_identity_error": gradient_error,
        "maximum_base_normalization_error": float(np.asarray(result.evidence[3])),
        "status": int(np.asarray(result.status)),
        "valid": bool(np.asarray(result.valid)),
        "invalid_sequence_status_check": {
            "status": int(np.asarray(invalid.status)),
            "rejected": invalid_rejected,
        },
        "passed": bool(
            np.asarray(result.valid)
            and partition_error <= 3.0e-6
            and gradient_error <= 3.0e-5
            and float(np.asarray(result.evidence[3])) <= 3.0e-5
            and invalid_rejected
        ),
    }


def qualification(
    *,
    mmcif_root: Path | None = None,
    mmcif_sha256: str | None = None,
) -> dict[str, object]:
    campaigns = {"mmcif": external_dataset_campaign("mmCIF", mmcif_root, mmcif_sha256)}
    return qualification_report(
        "structure_rna",
        {
            "macromolecular_lowering": _structure_lowering_case(),
            "rna_partition": _rna_partition_case(),
        },
        external_campaigns=campaigns,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify public structure and RNA APIs; external mmCIF roots are "
            "opt-in and never downloaded."
        )
    )
    parser.add_argument("--mmcif-root", type=Path)
    parser.add_argument("--mmcif-sha256")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualification(
        mmcif_root=arguments.mmcif_root,
        mmcif_sha256=arguments.mmcif_sha256,
    )
    return emit_report(report, arguments.output)


if __name__ == "__main__":
    raise SystemExit(main())
