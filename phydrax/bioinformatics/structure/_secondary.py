#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._topology import MacromolecularStructure
from ._types import SecondaryStructureKind, StructureStatus


class ContactAnalysisPlan(StrictModule, NonTrainableState):
    """Dense exact residue-contact definition over resolved atoms."""

    cutoff: float = eqx.field(static=True)
    exclude_bonded: bool = eqx.field(static=True)
    minimum_chain_separation: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cutoff: float = 4.5,
        *,
        exclude_bonded: bool = False,
        minimum_chain_separation: int = 0,
    ):
        cutoff_ = float(cutoff)
        separation = int(minimum_chain_separation)
        if not np.isfinite(cutoff_) or cutoff_ <= 0.0 or separation < 0:
            raise ValueError(
                "cutoff must be positive finite and minimum_chain_separation non-negative."
            )
        self.cutoff = cutoff_
        self.exclude_bonded = bool(exclude_bonded)
        self.minimum_chain_separation = separation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "residue-contact-analysis",
                "cutoff": cutoff_,
                "exclude_bonded": self.exclude_bonded,
                "minimum_chain_separation": separation,
            }
        )


class ResidueContactResult(StrictModule):
    adjacency: Array
    minimum_distances: Array
    atom_pair_counts: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        adjacency: Array,
        minimum_distances: Array,
        atom_pair_counts: Array,
        valid: Array,
        status: Array,
        evidence: Array,
        method_contract: BioinformaticsMethodContract,
    ):
        self.adjacency = adjacency
        self.minimum_distances = minimum_distances
        self.atom_pair_counts = atom_pair_counts
        self.valid = valid
        self.status = status
        self.evidence = evidence
        self.method_contract = method_contract
        self.evidence_labels = ("active_atom_count", "residue_count", "contact_count")


class SecondaryStructureResult(StrictModule):
    assignments: Array
    confidence: Array
    anchor_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        assignments: Array,
        confidence: Array,
        anchor_mask: Array,
        valid: Array,
        status: Array,
        evidence: Array,
        method_contract: BioinformaticsMethodContract,
    ):
        self.assignments = assignments
        self.confidence = confidence
        self.anchor_mask = anchor_mask
        self.valid = valid
        self.status = status
        self.evidence = evidence
        self.method_contract = method_contract
        self.evidence_labels = (
            "assigned_residue_count",
            "helix_count",
            "strand_count",
            "turn_count",
        )


def residue_contacts(
    structure: MacromolecularStructure,
    plan: ContactAnalysisPlan | None = None,
    /,
    *,
    model_index: int = 0,
) -> ResidueContactResult:
    """Compute exact minimum atom distances for every residue pair."""

    if not isinstance(structure, MacromolecularStructure):
        raise TypeError("structure must be a MacromolecularStructure.")
    resolved_plan = ContactAnalysisPlan() if plan is None else plan
    if not isinstance(resolved_plan, ContactAnalysisPlan):
        raise TypeError("plan must be a ContactAnalysisPlan.")
    if not 0 <= model_index < structure.model_capacity:
        raise IndexError("model_index is outside the model capacity.")
    positions = structure.positions[model_index]
    atom_mask = structure.altloc_mask(model_index)
    displacement = positions[:, None, :] - positions[None, :, :]
    distances = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
    pair_mask = atom_mask[:, None] & atom_mask[None, :]
    if resolved_plan.exclude_bonded and structure.bond_indices.shape[0]:
        bonded = jnp.zeros(pair_mask.shape, dtype=bool)
        bonded = bonded.at[
            structure.bond_indices[:, 0], structure.bond_indices[:, 1]
        ].set(True)
        bonded = bonded | bonded.T
        pair_mask = pair_mask & ~bonded
    residues = structure.atom_to_residue
    residue_count = structure.residue_capacity
    segment = (residues[:, None] * residue_count + residues[None, :]).reshape((-1,))
    flat_distance = jnp.where(pair_mask, distances, jnp.inf).reshape((-1,))
    minimum = jax.ops.segment_min(
        flat_distance, segment, num_segments=residue_count * residue_count
    ).reshape((residue_count, residue_count))
    within = pair_mask & (distances <= resolved_plan.cutoff)
    counts = jax.ops.segment_sum(
        within.astype(jnp.int32).reshape((-1,)),
        segment,
        num_segments=residue_count * residue_count,
    ).reshape((residue_count, residue_count))
    residue_index = jnp.arange(residue_count, dtype=jnp.int32)
    same_chain = (
        structure.residue_to_chain[:, None] == structure.residue_to_chain[None, :]
    )
    separation = jnp.abs(residue_index[:, None] - residue_index[None, :])
    allowed = ~jnp.eye(residue_count, dtype=bool)
    allowed = allowed & (
        ~same_chain | (separation > resolved_plan.minimum_chain_separation)
    )
    adjacency = (minimum <= resolved_plan.cutoff) & allowed
    minimum = jnp.where(allowed, minimum, jnp.inf)
    counts = jnp.where(allowed, counts, 0)
    valid = jnp.any(atom_mask)
    status = jnp.where(
        valid, int(StructureStatus.SUCCESS), int(StructureStatus.NO_VALID_MODEL)
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        [
            jnp.sum(atom_mask, dtype=jnp.int32),
            residue_count,
            jnp.sum(jnp.triu(adjacency, 1), dtype=jnp.int32),
        ],
        dtype=jnp.int32,
    )
    dtype = np.dtype(positions.dtype).name
    method = BioinformaticsMethodContract(
        "exact-residue-contact-map",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.GRAPH,
        conditioning_statement=(
            "Euclidean distances are stable away from coincident atom pairs; "
            "threshold output is nondifferentiable."
        ),
        truncation_statement="All compiled atom pairs are evaluated.",
        capacity_semantics="Dense atom and residue pair capacities are determined by the compiled structure.",
        assumptions=("Coordinates share one Cartesian frame and length unit.",),
        nondifferentiable_outputs=("adjacency", "counts", "status", "evidence"),
        input_dtype=dtype,
        compute_dtype=dtype,
        output_dtype=dtype,
    )
    return ResidueContactResult(
        adjacency, minimum, counts, valid, status, evidence, method
    )


def assign_geometric_secondary_structure(
    structure: MacromolecularStructure,
    /,
    *,
    model_index: int = 0,
    helix_i3_cutoff: float = 6.2,
    strand_contact_cutoff: float = 5.5,
    turn_angle_threshold: float = 1.15,
) -> SecondaryStructureResult:
    """Assign coarse backbone geometry without claiming DSSP thermodynamics."""

    if not 0 <= model_index < structure.model_capacity:
        raise IndexError("model_index is outside the model capacity.")
    if (
        helix_i3_cutoff <= 0.0
        or strand_contact_cutoff <= 0.0
        or not 0.0 < turn_angle_threshold < np.pi
    ):
        raise ValueError("Secondary-structure thresholds are outside their valid ranges.")
    anchors = structure.residue_anchor_atoms
    safe_anchor = jnp.maximum(anchors, 0)
    positions = structure.positions[model_index, safe_anchor]
    atom_mask = structure.altloc_mask(model_index)
    anchor_mask = (anchors >= 0) & atom_mask[safe_anchor]
    residue_count = structure.residue_capacity
    index = jnp.arange(residue_count, dtype=jnp.int32)
    same_chain = (
        structure.residue_to_chain[:, None] == structure.residue_to_chain[None, :]
    )
    delta = positions[:, None, :] - positions[None, :, :]
    distances = jnp.sqrt(jnp.sum(delta**2, axis=-1))
    pair_valid = anchor_mask[:, None] & anchor_mask[None, :] & same_chain
    helix_pair = (
        pair_valid
        & (jnp.abs(index[:, None] - index[None, :]) == 3)
        & (distances <= helix_i3_cutoff)
    )
    helix = jnp.any(helix_pair, axis=0) | jnp.any(helix_pair, axis=1)
    nonlocal_contact = (
        pair_valid
        & (jnp.abs(index[:, None] - index[None, :]) >= 4)
        & (distances <= strand_contact_cutoff)
    )
    strand = ~helix & (jnp.sum(nonlocal_contact, axis=1) >= 2)
    previous = positions - jnp.roll(positions, 1, axis=0)
    following = jnp.roll(positions, -1, axis=0) - positions
    previous_norm = jnp.sqrt(jnp.sum(previous**2, axis=-1))
    following_norm = jnp.sqrt(jnp.sum(following**2, axis=-1))
    cosine = jnp.sum(previous * following, axis=-1) / jnp.maximum(
        previous_norm * following_norm, jnp.finfo(positions.dtype).tiny
    )
    angle = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))
    neighboring = (
        (index > 0)
        & (index + 1 < residue_count)
        & (jnp.roll(structure.residue_to_chain, 1) == structure.residue_to_chain)
        & (jnp.roll(structure.residue_to_chain, -1) == structure.residue_to_chain)
    )
    turn = ~helix & ~strand & neighboring & anchor_mask & (angle >= turn_angle_threshold)
    assignments = jnp.full(
        (residue_count,), int(SecondaryStructureKind.UNKNOWN), dtype=jnp.int32
    )
    assignments = jnp.where(anchor_mask, int(SecondaryStructureKind.COIL), assignments)
    assignments = jnp.where(turn, int(SecondaryStructureKind.TURN), assignments)
    assignments = jnp.where(strand, int(SecondaryStructureKind.STRAND), assignments)
    assignments = jnp.where(helix, int(SecondaryStructureKind.HELIX), assignments)
    confidence = jnp.where(
        helix,
        jnp.clip(
            1.0
            - jnp.min(
                jnp.where(helix_pair, distances / helix_i3_cutoff, jnp.inf), axis=1
            ),
            0.0,
            1.0,
        ),
        jnp.where(
            strand,
            jnp.clip(jnp.sum(nonlocal_contact, axis=1) / 4.0, 0.0, 1.0),
            jnp.where(turn, jnp.clip(angle / jnp.pi, 0.0, 1.0), 0.5),
        ),
    )
    confidence = jnp.where(anchor_mask, confidence, 0.0)
    valid = jnp.any(anchor_mask)
    status = jnp.where(
        valid, int(StructureStatus.SUCCESS), int(StructureStatus.DEGENERATE_GEOMETRY)
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        [jnp.sum(anchor_mask), jnp.sum(helix), jnp.sum(strand), jnp.sum(turn)],
        dtype=jnp.int32,
    )
    dtype = np.dtype(positions.dtype).name
    method = BioinformaticsMethodContract(
        "coarse-geometric-secondary-structure",
        MethodKind.HEURISTIC,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.NONE,
        OutputKind.DISCRETE,
        conditioning_statement="Assignments use declared anchor-distance and turning-angle thresholds.",
        truncation_statement="All compiled residue anchors are evaluated.",
        capacity_semantics="One assignment is returned per compiled residue.",
        assumptions=("Anchor geometry is not a thermodynamic or DSSP assignment.",),
        nondifferentiable_outputs=("all outputs",),
        input_dtype=dtype,
        compute_dtype=dtype,
        output_dtype="int32",
    )
    return SecondaryStructureResult(
        assignments, confidence, anchor_mask, valid, status, evidence, method
    )


__all__ = [
    "ContactAnalysisPlan",
    "ResidueContactResult",
    "SecondaryStructureResult",
    "assign_geometric_secondary_structure",
    "residue_contacts",
]
