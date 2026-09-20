#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._snapshot import PrimitivePathSnapshot


class Z1PlusExportPlan(StrictModule, NonTrainableState):
    maximum_bytes: int = eqx.field(static=True)
    tool_version: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, maximum_bytes: int, tool_version: str):
        maximum = int(maximum_bytes)
        version = str(tool_version).strip()
        if maximum <= 0 or not version:
            raise ValueError("Z1+ export capacity and tool version are invalid.")
        self.maximum_bytes = maximum
        self.tool_version = version
        self.plan_id = canonical_fingerprint(
            {
                "kind": "z1plus-export-plan",
                "maximum_bytes": maximum,
                "tool_version": version,
                "format": "lammps-custom-unwrapped",
            }
        )


class Z1PlusInputArtifact(StrictModule, NonTrainableState):
    payload: bytes = eqx.field(static=True)
    source_snapshot_id: str = eqx.field(static=True)
    payload_sha256: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class Z1PlusOracleResult(StrictModule):
    primitive_positions: Array
    primitive_mask: Array
    contour_lengths: Array
    kink_counts: Array
    successful: Array
    source_snapshot_id: str = eqx.field(static=True)
    tool_version: str = eqx.field(static=True)
    tool_digest: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def export_z1plus_lammps_dump(
    plan: Z1PlusExportPlan,
    snapshot: PrimitivePathSnapshot,
    /,
) -> Z1PlusInputArtifact:
    if not isinstance(plan, Z1PlusExportPlan):
        raise TypeError("plan must be Z1PlusExportPlan.")
    if not isinstance(snapshot, PrimitivePathSnapshot):
        raise TypeError("snapshot must be PrimitivePathSnapshot.")
    positions = np.asarray(snapshot.unwrapped_positions, dtype=np.float64)
    active = np.asarray(snapshot.chain_mask, dtype=np.bool_)
    indices = np.asarray(snapshot.chain_indices, dtype=np.int32)
    particle_ids = np.asarray(snapshot.stable_particle_ids, dtype=np.int64)
    selected = indices[active]
    if snapshot.cell_vectors.shape != (3, 3):
        raise ValueError("Z1+ LAMMPS export requires one full periodic 3-D cell.")
    vectors = np.asarray(snapshot.cell_vectors, dtype=np.float64)
    off_diagonal = vectors - np.diag(np.diag(vectors))
    if not np.allclose(off_diagonal, 0.0):
        raise ValueError("Initial Z1+ interchange supports orthorhombic cells only.")
    lengths = np.diag(vectors)
    if np.any(lengths <= 0.0):
        raise ValueError("Z1+ export cell lengths must be positive.")
    molecule = np.full((positions.shape[0],), -1, dtype=np.int32)
    for chain_index, (row, mask) in enumerate(zip(indices, active, strict=True)):
        molecule[row[mask]] = chain_index + 1
    lines = [
        "ITEM: TIMESTEP",
        str(int(snapshot.step_index)),
        "ITEM: NUMBER OF ATOMS",
        str(selected.size),
        "ITEM: BOX BOUNDS pp pp pp",
        f"0 {lengths[0]:.17g}",
        f"0 {lengths[1]:.17g}",
        f"0 {lengths[2]:.17g}",
        "ITEM: ATOMS id mol xu yu zu",
    ]
    for slot in selected:
        x, y, z = positions[slot]
        lines.append(
            f"{int(particle_ids[slot])} {int(molecule[slot])} {x:.17g} {y:.17g} {z:.17g}"
        )
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    if len(payload) > plan.maximum_bytes:
        raise ValueError("Z1+ export exceeds maximum_bytes.")
    digest = canonical_fingerprint(
        {"kind": "z1plus-input-payload", "bytes": payload.hex()}
    )
    return Z1PlusInputArtifact(
        payload,
        snapshot.snapshot_id,
        digest,
        plan.plan_id,
    )


def import_z1plus_result(
    source: PrimitivePathSnapshot,
    record: Mapping,
    /,
    *,
    tool_version: str,
    tool_digest: str,
) -> Z1PlusOracleResult:
    if not isinstance(source, PrimitivePathSnapshot):
        raise TypeError("source must be PrimitivePathSnapshot.")
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping.")
    version = str(tool_version).strip()
    digest = str(tool_digest).strip()
    if not version or not digest:
        raise ValueError("Z1+ tool version and digest are required.")
    required = {
        "source_snapshot_id",
        "primitive_positions",
        "primitive_mask",
        "contour_lengths",
        "kink_counts",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError("Missing Z1+ result fields: " + ", ".join(sorted(missing)))
    if str(record["source_snapshot_id"]) != source.snapshot_id:
        raise ValueError("Z1+ result belongs to another source snapshot.")
    positions = np.asarray(record["primitive_positions"], dtype=np.float64)
    mask = np.asarray(record["primitive_mask"], dtype=np.bool_)
    contour = np.asarray(record["contour_lengths"], dtype=np.float64)
    kinks = np.asarray(record["kink_counts"], dtype=np.int32)
    chain_count = source.chain_indices.shape[0]
    if (
        positions.ndim != 3
        or positions.shape[:2] != mask.shape
        or positions.shape[-1] != 3
        or positions.shape[0] != chain_count
        or contour.shape != (chain_count,)
        or kinks.shape != (chain_count,)
        or np.any(~np.isfinite(positions[mask]))
        or np.any(~np.isfinite(contour))
        or np.any(contour <= 0.0)
        or np.any(kinks < 0)
        or np.any(np.sum(mask, axis=1) < 2)
    ):
        raise ValueError("Z1+ result arrays are invalid.")
    result_id = canonical_fingerprint(
        {
            "kind": "z1plus-oracle-result",
            "source": source.snapshot_id,
            "tool_version": version,
            "tool_digest": digest,
            "positions": array_tree_fingerprint(positions),
            "mask": array_tree_fingerprint(mask),
            "contour": array_tree_fingerprint(contour),
            "kinks": array_tree_fingerprint(kinks),
        }
    )
    return Z1PlusOracleResult(
        jnp.asarray(positions),
        jnp.asarray(mask),
        jnp.asarray(contour),
        jnp.asarray(kinks),
        jnp.asarray(True),
        source.snapshot_id,
        version,
        digest,
        result_id,
    )


__all__ = [
    "Z1PlusExportPlan",
    "Z1PlusInputArtifact",
    "Z1PlusOracleResult",
    "export_z1plus_lammps_dump",
    "import_z1plus_result",
]
