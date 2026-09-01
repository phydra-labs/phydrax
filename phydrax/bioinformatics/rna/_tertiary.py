#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic._topology import MolecularTopologyPlan
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..structure._topology import MacromolecularStructure
from ..structure._types import StructureStatus


class RNATertiaryRestraints(StrictModule, NonTrainableState):
    """Fixed residue-anchor distance constraints in the structure length unit."""

    residue_pairs: Array
    target_distances: Array
    active_mask: Array
    capacity: int = eqx.field(static=True)
    restraint_id: str = eqx.field(static=True)

    def __init__(
        self,
        residue_pairs: ArrayLike,
        target_distances: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ):
        pairs = np.asarray(residue_pairs)
        distances = np.asarray(target_distances)
        if (
            pairs.ndim != 2
            or pairs.shape[1] != 2
            or not np.issubdtype(pairs.dtype, np.integer)
        ):
            raise TypeError("residue_pairs must be an integer (capacity, 2) array.")
        pairs = pairs.astype(np.int32, copy=False)
        if np.any(pairs < 0) or np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError(
                "Residue restraint endpoints must be distinct and non-negative."
            )
        if distances.shape != (pairs.shape[0],) or not np.issubdtype(
            distances.dtype, np.inexact
        ):
            raise TypeError(
                "target_distances must be an inexact vector aligned with residue_pairs."
            )
        if np.any(~np.isfinite(distances)) or np.any(distances <= 0.0):
            raise ValueError("target_distances must be finite and positive.")
        active = (
            np.ones((pairs.shape[0],), dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if active.shape != (pairs.shape[0],):
            raise ValueError("active_mask must align with residue_pairs.")
        canonical = np.sort(pairs[active], axis=1)
        if canonical.shape[0] != np.unique(canonical, axis=0).shape[0]:
            raise ValueError(
                "Active residue restraints must not repeat an unordered pair."
            )
        self.residue_pairs = jnp.asarray(pairs)
        self.target_distances = jnp.asarray(distances)
        self.active_mask = jnp.asarray(active)
        self.capacity = int(pairs.shape[0])
        self.restraint_id = canonical_fingerprint(
            {
                "kind": "rna-tertiary-restraints",
                "arrays": array_tree_fingerprint(
                    {
                        "residue_pairs": pairs,
                        "target_distances": distances,
                        "active_mask": active,
                    }
                ),
            }
        )


class TertiaryRestraintLoweringResult(StrictModule):
    topology: MolecularTopologyPlan | None
    atom_pairs: Array
    target_distances: Array
    active_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    restraint_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        topology: MolecularTopologyPlan | None,
        atom_pairs: Array,
        target_distances: Array,
        active_mask: Array,
        valid: bool,
        status: StructureStatus,
        evidence: np.ndarray,
        method_contract: BioinformaticsMethodContract,
        restraint_id: str,
        structure_id: str,
    ):
        self.topology = topology
        self.atom_pairs = jnp.asarray(atom_pairs, dtype=jnp.int32)
        self.target_distances = jnp.asarray(target_distances)
        self.active_mask = jnp.asarray(active_mask, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(int(status), dtype=jnp.int32)
        self.evidence = jnp.asarray(evidence, dtype=jnp.int32)
        self.method_contract = method_contract
        self.restraint_id = restraint_id
        self.structure_id = structure_id
        self.evidence_labels = (
            "restraint_capacity",
            "active_restraints",
            "unresolved_anchors",
            "absent_anchors",
        )


def lower_tertiary_restraints(
    restraints: RNATertiaryRestraints,
    structure: MacromolecularStructure,
    /,
    *,
    model_index: int = 0,
    constraint_capacity: int | None = None,
) -> TertiaryRestraintLoweringResult:
    """Resolve residue anchors to exact atomistic distance constraints."""

    if not isinstance(restraints, RNATertiaryRestraints) or not isinstance(
        structure, MacromolecularStructure
    ):
        raise TypeError("restraints and structure have incompatible types.")
    if not 0 <= model_index < structure.model_capacity:
        raise IndexError("model_index is outside the model capacity.")
    capacity = (
        restraints.capacity if constraint_capacity is None else int(constraint_capacity)
    )
    if capacity < 0:
        raise ValueError("constraint_capacity must be non-negative.")
    pairs = np.asarray(restraints.residue_pairs, dtype=np.int32)
    active = np.asarray(restraints.active_mask, dtype=bool)
    distances = np.asarray(restraints.target_distances)
    evidence = np.asarray(
        [restraints.capacity, np.count_nonzero(active), 0, 0], dtype=np.int32
    )
    method = BioinformaticsMethodContract(
        "rna-residue-anchor-restraint-lowering",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Declared residue anchors are resolved to stable atom particle IDs "
            "without geometric approximation."
        ),
        truncation_statement="No active restraint is truncated.",
        capacity_semantics="Active restraint count is preflighted against constraint_capacity.",
        assumptions=(
            "Target distances are expressed in the compiled structure length unit.",
        ),
        nondifferentiable_outputs=("all outputs",),
        input_dtype=np.dtype(distances.dtype).name,
        compute_dtype=np.dtype(distances.dtype).name,
        output_dtype=np.dtype(distances.dtype).name,
    )

    def result(
        topology: MolecularTopologyPlan | None,
        atom_pairs: np.ndarray,
        valid: bool,
        status: StructureStatus,
    ) -> TertiaryRestraintLoweringResult:
        return TertiaryRestraintLoweringResult(
            topology,
            atom_pairs,
            distances,
            active,
            valid,
            status,
            evidence,
            method,
            restraints.restraint_id,
            structure.structure_id,
        )

    if np.count_nonzero(active) > capacity:
        return result(
            None, np.zeros_like(pairs), False, StructureStatus.CAPACITY_EXCEEDED
        )
    if np.any(pairs[active] >= structure.residue_capacity):
        evidence[2] = int(np.count_nonzero(pairs[active] >= structure.residue_capacity))
        return result(
            None, np.zeros_like(pairs), False, StructureStatus.UNRESOLVED_REFERENCE
        )
    anchors = np.asarray(structure.residue_anchor_atoms, dtype=np.int32)
    atom_pairs = np.where(active[:, None], anchors[pairs], 0)
    unresolved = active[:, None] & (atom_pairs < 0)
    evidence[2] = int(np.count_nonzero(unresolved))
    present = np.asarray(structure.altloc_mask(model_index), dtype=bool)
    absent = active[:, None] & ~present[np.maximum(atom_pairs, 0)]
    evidence[3] = int(np.count_nonzero(absent))
    if evidence[2] or evidence[3]:
        return result(None, atom_pairs, False, StructureStatus.UNRESOLVED_REFERENCE)
    active_atom_pairs = atom_pairs[active]
    active_distances = distances[active]
    topology = MolecularTopologyPlan(
        constraints=active_atom_pairs.astype(np.int64),
        constraint_distances=active_distances,
    )
    return result(topology, atom_pairs, True, StructureStatus.SUCCESS)


__all__ = [
    "RNATertiaryRestraints",
    "TertiaryRestraintLoweringResult",
    "lower_tertiary_restraints",
]
