#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generation-stream fixtures shared by unit tests and the multi-process smoke.

`HeterogeneousCountdown` is a stateful sampler whose ordinals need different
numbers of advance steps (so lanes complete out of ordinal order) and whose
ordinals ``6 mod 7`` report a sampler failure; a minimum-distance filter adds
geometric refusals. `expected_positions` independently recomputes a proposal
from its semantic address. `ThrottledSink` refuses a declared number of
publications (sink backpressure) before delegating. The structure-store schema
and record builder make proposals durable typed records.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax._identity import NumericRevision
from phydrax.atomistic._conditioning import AtomisticConditioning
from phydrax.atomistic._generation import (
    AbstractAtomisticGenerationSink,
    AbstractPreparedAtomisticSampler,
    AtomisticChemistryAdmission,
    AtomisticGeneratedProposal,
    AtomisticGenerationReceipt,
    AtomisticGenerationStreamPlan,
    AtomisticGenerationSupport,
    AtomisticMinimumDistanceFilter,
    AtomisticPayloadSpec,
    AtomisticProposalBlock,
    AtomisticProposalStatus,
    proposal_address,
    proposal_keys,
    role_keys,
)
from phydrax.atomistic._structure_store import (
    AtomisticStoreField,
    AtomisticStoreLayout,
    AtomisticStoreLevel,
    AtomisticStoreLimits,
    AtomisticStructureRecord,
    AtomisticStructureStoreSchema,
    AtomisticStructureStoreWriter,
)
from phydrax.axes import Axis, AxisKey
from phydrax.units import ANGSTROM


CAMPAIGN = "distributed-generation"
PAYLOAD = AtomisticPayloadSpec((("draw", (), "float64"), ("remaining", (), "int32")))
MINIMUM_DISTANCE = 2.0
RATE = 0.01
_CARTESIAN = Axis(AxisKey("atomistic", "cartesian"), 3, ("x", "y", "z"))


def generation_support() -> AtomisticGenerationSupport:
    """Fixed two-atom chemistry in a three-slot capacity."""
    return AtomisticGenerationSupport(
        atom_capacity=3,
        species_vocabulary=(1, 8),
        vocabulary_masses=(1.008, 15.999),
        mass_provenance="test masses",
        chemistry=AtomisticChemistryAdmission.FIXED,
        length_unit=ANGSTROM,
        fixed_species=(1, 8, 0),
        fixed_mask=(True, True, False),
    )


def steps_needed(ordinals: jax.Array) -> jax.Array:
    """Advance steps ordinal ``o`` needs: ``1 + 3 o mod 5`` (heterogeneous)."""
    return (1 + (3 * ordinals) % 5).astype(jnp.int32)


def _draw(keys: jax.Array) -> jax.Array:
    return jax.vmap(lambda key: 4.0 * jr.uniform(key, (3, 3), dtype=jnp.float64))(
        role_keys(keys, "positions")
    )


class HeterogeneousCountdown(AbstractPreparedAtomisticSampler):
    """Stateful sampler with ordinal-dependent completion and sampler failures."""

    support: AtomisticGenerationSupport
    payload_spec: AtomisticPayloadSpec
    sampler_id: str = eqx.field(static=True)
    stateful: bool = eqx.field(static=True)
    idempotent: bool = eqx.field(static=True)

    def __init__(self) -> None:
        self.support = generation_support()
        self.payload_spec = PAYLOAD
        self.sampler_id = "heterogeneous-countdown"
        self.stateful = True
        self.idempotent = False

    def vacant_method(self, lanes: int) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.zeros((lanes, 3, 3), dtype=jnp.float64),
            jnp.zeros((lanes,), dtype=jnp.int32),
        )

    def initialize_method(
        self,
        keys: jax.Array,
        ordinals: jax.Array,
        conditioning: AtomisticConditioning | None,
        selected: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _draw(keys), steps_needed(ordinals)

    def advance_method(
        self,
        method: tuple[jax.Array, jax.Array],
        running: jax.Array,
        steps: int,
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        positions, remaining = method
        taken = jnp.minimum(remaining, steps)
        moved = positions + RATE * taken[:, None, None].astype(positions.dtype)
        remaining = remaining - taken
        return (moved, remaining), remaining == 0

    def emit_method(
        self,
        method: tuple[jax.Array, jax.Array],
        ordinals: jax.Array,
    ) -> AtomisticProposalBlock:
        positions, remaining = method
        mask = jnp.asarray(self.support.fixed_mask)
        status = jnp.where(
            ordinals % 7 == 6, int(AtomisticProposalStatus.SAMPLER_FAILURE), 0
        )
        return AtomisticProposalBlock(
            ordinals,
            jnp.where(mask[None, :, None], positions, 0.0),
            jnp.broadcast_to(
                jnp.asarray(self.support.fixed_species), positions.shape[:2]
            ),
            jnp.broadcast_to(mask, positions.shape[:2]),
            None,
            status,
            (positions[:, 0, 0], remaining),
            sampler_id=self.sampler_id,
            payload_names=self.payload_spec.names,
        )

    def signature_facts(self) -> dict[str, str | int]:
        return {"kind": "test-heterogeneous-countdown"}

    def numeric_revision(self) -> NumericRevision:
        return NumericRevision(self.sampler_id, {"rate": np.asarray(RATE)})


def stream_plan(
    *, lanes_per_member: int, pending_capacity: int = 6
) -> AtomisticGenerationStreamPlan:
    """The smoke stream: heterogeneous countdown plus a minimum-distance filter."""
    return AtomisticGenerationStreamPlan(
        HeterogeneousCountdown(),
        campaign_id=CAMPAIGN,
        lanes_per_member=lanes_per_member,
        pending_capacity=pending_capacity,
        steps_per_transition=1,
        filters=(AtomisticMinimumDistanceFilter(MINIMUM_DISTANCE),),
    )


def expected_positions(root: int, ordinals: Sequence[int]) -> np.ndarray:
    """Independent recomputation of the active-atom positions of ``ordinals``.

    The draw is re-derived from the semantic address ``(campaign, support,
    sampler, ordinal, role)``; one step of ``RATE`` per needed advance follows
    (steps of one per transition).
    """
    sampler = HeterogeneousCountdown()
    address = proposal_address(
        CAMPAIGN, sampler.support.support_id, sampler.sampler_id, role="proposal"
    )
    values = jnp.asarray(ordinals, dtype=jnp.int64)
    draws = np.asarray(_draw(proposal_keys(jr.key(root), address, values)))
    steps = np.asarray(steps_needed(values))
    moved = draws + RATE * steps[:, None, None].astype(np.float64)
    return moved[:, :2]


class ThrottledSink(AbstractAtomisticGenerationSink):
    """Refuses the first ``refusals`` publications (backpressure), then delegates."""

    def __init__(self, inner: AbstractAtomisticGenerationSink, refusals: int, /) -> None:
        self._inner = inner
        self._remaining = refusals
        self.refused = 0

    @property
    def sink_id(self) -> str:
        return self._inner.sink_id

    def publish(
        self, proposals: Sequence[AtomisticGeneratedProposal], /
    ) -> AtomisticGenerationReceipt | None:
        if self._remaining > 0:
            self._remaining -= 1
            self.refused += 1
            return None
        return self._inner.publish(proposals)

    def published(self, proposal_ids: Sequence[str], /) -> tuple[bool, ...]:
        return self._inner.published(proposal_ids)


def store_limits() -> AtomisticStoreLimits:
    return AtomisticStoreLimits(
        max_commit_bytes=1 << 21,
        max_member_bytes=1 << 22,
        max_shard_bytes=1 << 24,
        max_shard_members=8192,
        max_chunk_bytes=1 << 20,
        max_index_bytes=1 << 21,
        max_decoded_bytes=1 << 24,
        max_read_bytes=1 << 24,
        max_records=1000,
    )


def generation_store_schema() -> AtomisticStructureStoreSchema:
    """Typed store schema of published generation proposals."""
    return AtomisticStructureStoreSchema(
        (
            AtomisticStoreLevel("structure", "structure"),
            AtomisticStoreLevel("atom", "atom", parent="structure"),
        ),
        (
            AtomisticStoreField(
                "status",
                "structure",
                "int32",
                kind="categorical",
                role="label",
                convention="atomistic-proposal-status-bits",
            ),
            AtomisticStoreField(
                "ordinal",
                "structure",
                "int64",
                kind="integer",
                role="identity",
                convention="generation-attempt-ordinal",
            ),
            AtomisticStoreField(
                "positions",
                "atom",
                "float64",
                kind="continuous",
                role="geometry",
                convention="cartesian",
                axes=(_CARTESIAN,),
                unit=ANGSTROM,
                extensivity="intensive",
            ),
            AtomisticStoreField(
                "species",
                "atom",
                "int32",
                kind="categorical",
                role="identity",
                convention="atomic-number",
            ),
        ),
    )


def generation_record(
    schema: AtomisticStructureStoreSchema, proposal: AtomisticGeneratedProposal, /
) -> AtomisticStructureRecord:
    """One typed store record per published proposal (record id = proposal id)."""
    count = proposal.positions.shape[0]
    return AtomisticStructureRecord(
        schema,
        record_id=proposal.proposal_id,
        lineage_id=proposal.campaign_id,
        counts={"atom": [count]},
        stable_ids=np.asarray(proposal.slots, dtype=np.int64),
        values={
            "status": np.asarray([proposal.status], dtype=np.int32),
            "ordinal": np.asarray([proposal.ordinal], dtype=np.int64),
            "positions": np.asarray(proposal.positions, dtype=np.float64),
            "species": np.asarray(proposal.species, dtype=np.int32),
        },
    )


def generation_store_writer(root: Path, /) -> AtomisticStructureStoreWriter:
    return AtomisticStructureStoreWriter(
        root,
        layout=AtomisticStoreLayout(
            records_per_shard=8, target_chunk_bytes=4096, compression="none"
        ),
        limits=store_limits(),
    )
