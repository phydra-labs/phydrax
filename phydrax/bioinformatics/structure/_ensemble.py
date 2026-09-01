#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._alignment import align_coordinates
from ._topology import MacromolecularStructure
from ._types import StructureStatus


class StructureEnsembleResult(StrictModule):
    """Occupancy-aware moments across fixed-topology coordinate models."""

    mean_positions: Array
    rmsf: Array
    mean_occupancy: Array
    aligned_positions: Array
    aligned_mask: Array
    model_rmsd: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        mean_positions: Array,
        rmsf: Array,
        mean_occupancy: Array,
        aligned_positions: Array,
        aligned_mask: Array,
        model_rmsd: Array,
        valid: Array,
        status: Array,
        evidence: Array,
        method_contract: BioinformaticsMethodContract,
    ):
        self.mean_positions = mean_positions
        self.rmsf = rmsf
        self.mean_occupancy = mean_occupancy
        self.aligned_positions = aligned_positions
        self.aligned_mask = aligned_mask
        self.model_rmsd = model_rmsd
        self.valid = valid
        self.status = status
        self.evidence = evidence
        self.method_contract = method_contract
        self.evidence_labels = (
            "model_count",
            "atoms_observed_once",
            "atoms_observed_all_models",
        )


def analyze_structure_ensemble(
    structure: MacromolecularStructure,
    /,
    *,
    align_models: bool = True,
    reference_model_index: int = 0,
) -> StructureEnsembleResult:
    """Compute conformer-coupled ensemble means and RMS fluctuations."""

    if not isinstance(structure, MacromolecularStructure):
        raise TypeError("structure must be a MacromolecularStructure.")
    if not 0 <= reference_model_index < structure.model_capacity:
        raise IndexError("reference_model_index is outside the model capacity.")
    masks = jnp.stack(
        [structure.altloc_mask(index) for index in range(structure.model_capacity)]
    )
    reference = structure.positions[reference_model_index]
    reference_mask = masks[reference_model_index]
    aligned_values: list[Array] = []
    model_rmsd: list[Array] = []
    model_valid: list[Array] = []
    for index in range(structure.model_capacity):
        if align_models and index != reference_model_index:
            common = masks[index] & reference_mask
            weights = jnp.minimum(
                structure.occupancies[index], structure.occupancies[reference_model_index]
            )
            alignment = align_coordinates(
                structure.positions[index], reference, weights=weights, mask=common
            )
            aligned_values.append(alignment.aligned)
            model_rmsd.append(alignment.rmsd)
            model_valid.append(alignment.valid)
        else:
            aligned_values.append(structure.positions[index])
            model_rmsd.append(jnp.asarray(0.0, dtype=structure.positions.dtype))
            model_valid.append(jnp.asarray(True))
    aligned = jnp.stack(aligned_values)
    valid_models = jnp.stack(model_valid)
    effective_mask = masks & valid_models[:, None]
    weights = jnp.where(effective_mask, structure.occupancies, 0.0)
    total = jnp.sum(weights, axis=0)
    safe_total = jnp.maximum(total, jnp.asarray(1.0, dtype=weights.dtype))
    mean = jnp.sum(weights[..., None] * aligned, axis=0) / safe_total[:, None]
    squared = jnp.sum((aligned - mean[None, ...]) ** 2, axis=-1)
    rmsf = jnp.sqrt(jnp.sum(weights * squared, axis=0) / safe_total)
    atom_valid = total > 0.0
    mean = jnp.where(atom_valid[:, None], mean, 0.0)
    rmsf = jnp.where(atom_valid, rmsf, jnp.nan)
    mean_occupancy = jnp.sum(
        jnp.where(effective_mask, structure.occupancies, 0.0), axis=0
    ) / jnp.maximum(jnp.sum(effective_mask, axis=0), 1)
    valid = jnp.any(atom_valid) & jnp.all(valid_models)
    status = jnp.where(
        valid, int(StructureStatus.SUCCESS), int(StructureStatus.DEGENERATE_GEOMETRY)
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        [
            structure.model_capacity,
            jnp.sum(atom_valid, dtype=jnp.int32),
            jnp.sum(jnp.all(effective_mask, axis=0), dtype=jnp.int32),
        ],
        dtype=jnp.int32,
    )
    dtype = np.dtype(structure.positions.dtype).name
    method = BioinformaticsMethodContract(
        "occupancy-weighted-structure-ensemble-moments",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Optional Kabsch registration is piecewise differentiable away from "
            "singular-value degeneracy."
        ),
        truncation_statement="No models or atoms are truncated.",
        capacity_semantics="All compiled models and atom sites are evaluated.",
        assumptions=("Model atom identities share the compiled topology.",),
        nondifferentiable_outputs=("masks", "status", "evidence"),
        input_dtype=dtype,
        compute_dtype=dtype,
        output_dtype=dtype,
    )
    return StructureEnsembleResult(
        mean,
        rmsf,
        mean_occupancy,
        aligned,
        effective_mask,
        jnp.stack(model_rmsd),
        valid,
        status,
        evidence,
        method,
    )


__all__ = ["StructureEnsembleResult", "analyze_structure_ensemble"]
