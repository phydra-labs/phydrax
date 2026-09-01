#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
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
from ._topology import MacromolecularStructure
from ._types import StructureStatus


class ChainInterfaceResult(StrictModule):
    """Exact atom-contact and centroid geometry for every ordered chain pair."""

    contact_counts: Array
    minimum_distances: Array
    centroid_displacements: Array
    chain_centroids: Array
    chain_present: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    evidence_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        contact_counts: Array,
        minimum_distances: Array,
        centroid_displacements: Array,
        chain_centroids: Array,
        chain_present: Array,
        valid: Array,
        status: Array,
        evidence: Array,
        method_contract: BioinformaticsMethodContract,
    ):
        self.contact_counts = contact_counts
        self.minimum_distances = minimum_distances
        self.centroid_displacements = centroid_displacements
        self.chain_centroids = chain_centroids
        self.chain_present = chain_present
        self.valid = valid
        self.status = status
        self.evidence = evidence
        self.method_contract = method_contract
        self.evidence_labels = (
            "present_chain_count",
            "interface_count",
            "contact_atom_pair_count",
        )


def analyze_chain_interfaces(
    structure: MacromolecularStructure,
    /,
    *,
    model_index: int = 0,
    cutoff: float = 5.0,
) -> ChainInterfaceResult:
    """Analyze chain interfaces with rigid-invariant scalars and equivariant vectors."""

    if not isinstance(structure, MacromolecularStructure):
        raise TypeError("structure must be a MacromolecularStructure.")
    if not 0 <= model_index < structure.model_capacity:
        raise IndexError("model_index is outside the model capacity.")
    cutoff_ = float(cutoff)
    if not np.isfinite(cutoff_) or cutoff_ <= 0.0:
        raise ValueError("cutoff must be finite and positive.")
    positions = structure.positions[model_index]
    atom_mask = structure.altloc_mask(model_index)
    atom_chain = structure.residue_to_chain[structure.atom_to_residue]
    chain_count = structure.chain_capacity
    weights = atom_mask.astype(positions.dtype)
    counts = jax.ops.segment_sum(weights, atom_chain, num_segments=chain_count)
    sums = jax.ops.segment_sum(
        weights[:, None] * positions, atom_chain, num_segments=chain_count
    )
    centroids = sums / jnp.maximum(counts[:, None], 1.0)
    chain_present = counts > 0.0
    displacement = positions[:, None, :] - positions[None, :, :]
    distance = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
    pair_mask = atom_mask[:, None] & atom_mask[None, :]
    segment = (atom_chain[:, None] * chain_count + atom_chain[None, :]).reshape((-1,))
    minimum = jax.ops.segment_min(
        jnp.where(pair_mask, distance, jnp.inf).reshape((-1,)),
        segment,
        num_segments=chain_count * chain_count,
    ).reshape((chain_count, chain_count))
    atom_contacts = pair_mask & (distance <= cutoff_)
    contact_counts = jax.ops.segment_sum(
        atom_contacts.astype(jnp.int32).reshape((-1,)),
        segment,
        num_segments=chain_count * chain_count,
    ).reshape((chain_count, chain_count))
    off_diagonal = ~jnp.eye(chain_count, dtype=bool)
    pair_present = chain_present[:, None] & chain_present[None, :] & off_diagonal
    minimum = jnp.where(pair_present, minimum, jnp.inf)
    contact_counts = jnp.where(pair_present, contact_counts, 0)
    centroid_displacements = centroids[None, :, :] - centroids[:, None, :]
    centroid_displacements = jnp.where(
        pair_present[..., None], centroid_displacements, 0.0
    )
    interface = (contact_counts > 0) & jnp.triu(
        jnp.ones((chain_count, chain_count), dtype=bool), 1
    )
    valid = jnp.sum(chain_present) >= 2
    status = jnp.where(
        valid, int(StructureStatus.SUCCESS), int(StructureStatus.DEGENERATE_GEOMETRY)
    ).astype(jnp.int32)
    evidence = jnp.asarray(
        [
            jnp.sum(chain_present),
            jnp.sum(interface),
            jnp.sum(jnp.triu(contact_counts, 1)),
        ],
        dtype=jnp.int32,
    )
    dtype = np.dtype(positions.dtype).name
    method = BioinformaticsMethodContract(
        "exact-chain-interface-contact-analysis",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.GRAPH,
        conditioning_statement=(
            "Distances and centroids are smooth away from coincident atoms; contact "
            "thresholds are discrete."
        ),
        truncation_statement="All compiled inter-chain atom pairs are evaluated.",
        capacity_semantics="Dense chain-pair outputs use the compiled chain capacity.",
        assumptions=("Coordinates share a Cartesian frame and length unit.",),
        nondifferentiable_outputs=("contact_counts", "status", "evidence"),
        input_dtype=dtype,
        compute_dtype=dtype,
        output_dtype=dtype,
    )
    return ChainInterfaceResult(
        contact_counts,
        minimum,
        centroid_displacements,
        centroids,
        chain_present,
        valid,
        status,
        evidence,
        method,
    )


__all__ = ["ChainInterfaceResult", "analyze_chain_interfaces"]
