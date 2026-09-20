#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ..entanglement import (
    EntanglementEstimatorResult,
    ForcePrimitivePathResult,
    PrimitivePathSnapshot,
)
from ..reptation import (
    DoiEdwardsTubePlan,
    LikhtmanMcLeishPlan,
    SlipSpringPlan,
)


class SlipSpringSeed(StrictModule):
    particle_pairs: Array
    active_mask: Array
    count: Array
    source_snapshot_id: str = eqx.field(static=True)
    source_primitive_path_id: str = eqx.field(static=True)
    seed_id: str = eqx.field(static=True)


def primitive_path_contacts_to_slip_spring_seed(
    snapshot: PrimitivePathSnapshot,
    primitive_path: ForcePrimitivePathResult,
    /,
) -> SlipSpringSeed:
    if not isinstance(snapshot, PrimitivePathSnapshot) or not isinstance(
        primitive_path, ForcePrimitivePathResult
    ):
        raise TypeError("snapshot and primitive_path must use entanglement contracts.")
    if primitive_path.source_snapshot_id != snapshot.snapshot_id:
        raise ValueError("Primitive-path result belongs to another source snapshot.")
    if not bool(np.asarray(primitive_path.successful)):
        raise ValueError("Only a successful primitive path can seed slip springs.")
    contacts = primitive_path.contact_state
    active = np.asarray(contacts.active, dtype=np.bool_)
    left_chain = np.asarray(contacts.left_chain, dtype=np.int32)
    left_contour = np.asarray(contacts.left_contour, dtype=np.int32)
    right_chain = np.asarray(contacts.right_chain, dtype=np.int32)
    right_contour = np.asarray(contacts.right_contour, dtype=np.int32)
    chain_indices = np.asarray(snapshot.chain_indices, dtype=np.int32)
    capacity = active.size
    pairs = np.full((capacity, 2), -1, dtype=np.int32)
    active_rows = np.flatnonzero(active)
    if active_rows.size:
        left = chain_indices[left_chain[active_rows], left_contour[active_rows]]
        right = chain_indices[right_chain[active_rows], right_contour[active_rows]]
        selected = np.sort(np.stack((left, right), axis=1), axis=1)
        unique = np.unique(selected, axis=0)
        pairs[: unique.shape[0]] = unique
        mask = np.arange(capacity) < unique.shape[0]
    else:
        mask = np.zeros((capacity,), dtype=np.bool_)
    count = int(np.sum(mask))
    seed_id = canonical_fingerprint(
        {
            "kind": "primitive-path-slip-spring-seed",
            "snapshot": snapshot.snapshot_id,
            "primitive_path": primitive_path.prepared_id,
            "pairs": pairs[mask],
        }
    )
    return SlipSpringSeed(
        jnp.asarray(pairs),
        jnp.asarray(mask),
        jnp.asarray(count, dtype=jnp.int32),
        snapshot.snapshot_id,
        primitive_path.prepared_id,
        seed_id,
    )


def slip_spring_plan_from_primitive_path(
    seed: SlipSpringSeed,
    /,
    *,
    maximum_particles: int,
    inverse_temperature: float,
    stiffness: float,
    chemical_potential: float,
    maximum_extension: float,
) -> tuple[SlipSpringPlan, Array]:
    if not isinstance(seed, SlipSpringSeed):
        raise TypeError("seed must be SlipSpringSeed.")
    count = int(np.asarray(seed.count))
    if count <= 0:
        raise ValueError(
            "A slip-spring plan requires at least one primitive-path contact."
        )
    allowed = seed.particle_pairs[seed.active_mask]
    plan = SlipSpringPlan(
        maximum_particles,
        count,
        inverse_temperature,
        stiffness,
        chemical_potential,
        maximum_extension=maximum_extension,
    )
    return plan, allowed


def likhtman_mcleish_from_entanglement(
    result: EntanglementEstimatorResult,
    monomer_count: int,
    entanglement_time: float,
    /,
    *,
    constraint_release_time: float = float("inf"),
    odd_mode_count: int = 64,
) -> LikhtmanMcLeishPlan:
    if not isinstance(result, EntanglementEstimatorResult):
        raise TypeError("result must be EntanglementEstimatorResult.")
    if not bool(np.asarray(result.successful)):
        raise ValueError(
            "Only a successful entanglement estimate can parameterize LM rheology."
        )
    entanglement_length = float(np.asarray(result.coil_entanglement_length))
    count = int(monomer_count)
    if count <= 0 or entanglement_length <= 0.0:
        raise ValueError("monomer_count and entanglement length must be positive.")
    entanglement_count = max(2, int(round(count / entanglement_length)))
    return LikhtmanMcLeishPlan(
        entanglement_count,
        entanglement_time,
        float(np.asarray(result.plateau_modulus)),
        constraint_release_time=constraint_release_time,
        odd_mode_count=odd_mode_count,
    )


def doi_edwards_from_entanglement(
    result: EntanglementEstimatorResult,
    disengagement_time: float,
    /,
    *,
    odd_mode_count: int = 64,
) -> DoiEdwardsTubePlan:
    if not isinstance(result, EntanglementEstimatorResult):
        raise TypeError("result must be EntanglementEstimatorResult.")
    if not bool(np.asarray(result.successful)):
        raise ValueError(
            "Only a successful entanglement estimate can parameterize tube rheology."
        )
    return DoiEdwardsTubePlan(
        disengagement_time,
        float(np.asarray(result.plateau_modulus)),
        odd_mode_count=odd_mode_count,
    )


__all__ = [
    "SlipSpringSeed",
    "doi_edwards_from_entanglement",
    "likhtman_mcleish_from_entanglement",
    "primitive_path_contacts_to_slip_spring_seed",
    "slip_spring_plan_from_primitive_path",
]
