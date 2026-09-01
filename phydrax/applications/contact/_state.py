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
from ._geometry import ContactPatchSet, ContactQueryResult


CONTACT_OPEN = 0
CONTACT_STICK = 1
CONTACT_SLIP = 2


class AcceptedContactState(StrictModule, NonTrainableState):
    """Accepted multiplier and transported friction history keyed by stable pair IDs."""

    normal_pressure: Array
    contact_normals: Array
    tangential_traction: Array
    accumulated_slip: Array
    mode: Array
    pair_ids: tuple[str, ...] = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    state_version: int = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    patch_set_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        pair_ids: tuple[str, ...],
        normal_pressure: ArrayLike,
        contact_normals: ArrayLike,
        tangential_traction: ArrayLike,
        accumulated_slip: ArrayLike,
        mode: ArrayLike,
        /,
        *,
        epoch: int,
        law_id: str,
        patch_set_id: str,
        state_version: int = 0,
    ):
        pairs = tuple(str(value) for value in pair_ids)
        pressure = np.asarray(normal_pressure)
        normals = np.asarray(contact_normals)
        traction = np.asarray(tangential_traction)
        slip = np.asarray(accumulated_slip)
        mode_ = np.asarray(mode, dtype=np.int32)
        epoch_ = int(epoch)
        version = int(state_version)
        law_identity = str(law_id)
        patch_identity = str(patch_set_id)
        count = len(pairs)
        if (
            len(set(pairs)) != count
            or pressure.shape != (count,)
            or normals.ndim != 2
            or normals.shape[0] != count
            or normals.shape[1] not in (2, 3)
            or not np.issubdtype(pressure.dtype, np.inexact)
            or not np.issubdtype(normals.dtype, np.inexact)
            or not np.issubdtype(traction.dtype, np.inexact)
            or not np.issubdtype(slip.dtype, np.inexact)
            or traction.shape != normals.shape
            or slip.shape != normals.shape
            or mode_.shape != (count,)
            or epoch_ < 0
            or version < 0
            or not law_identity
            or not patch_identity
        ):
            raise ValueError(
                "Accepted contact-state arrays, epoch, or version are invalid."
            )
        normal_tolerance = 128.0 * np.finfo(normals.dtype).eps
        normal_norm = np.linalg.norm(normals, axis=-1)
        if (
            np.any(~np.isfinite(pressure))
            or np.any(pressure < 0.0)
            or np.any(~np.isfinite(normal_norm))
            or np.any(np.abs(normal_norm - 1.0) > normal_tolerance)
            or np.any(~np.isfinite(traction))
            or np.any(~np.isfinite(slip))
            or np.any(~np.isin(mode_, (CONTACT_OPEN, CONTACT_STICK, CONTACT_SLIP)))
        ):
            raise ValueError(
                "Accepted contact history must be finite and physically admissible."
            )
        self.normal_pressure = jnp.asarray(pressure)
        self.contact_normals = jnp.asarray(normals)
        self.tangential_traction = jnp.asarray(traction)
        self.accumulated_slip = jnp.asarray(slip)
        self.mode = jnp.asarray(mode_)
        self.pair_ids = pairs
        self.epoch = epoch_
        self.state_version = version
        self.law_id = law_identity
        self.patch_set_id = patch_identity
        self.state_id = canonical_fingerprint(
            {
                "kind": "accepted-contact-state",
                "pairs": list(pairs),
                "pressure": array_tree_fingerprint(pressure),
                "normals": array_tree_fingerprint(normals),
                "traction": array_tree_fingerprint(traction),
                "slip": array_tree_fingerprint(slip),
                "mode": array_tree_fingerprint(mode_),
                "epoch": epoch_,
                "law": law_identity,
                "patches": patch_identity,
                "version": version,
            }
        )

    @classmethod
    def zeros(
        cls,
        patches: ContactPatchSet,
        /,
        *,
        law_id: str,
        state_version: int = 0,
        dtype=None,
    ) -> AcceptedContactState:
        if not isinstance(patches, ContactPatchSet):
            raise TypeError("AcceptedContactState.zeros requires ContactPatchSet.")
        dtype_ = patches.gaps.dtype if dtype is None else dtype
        count = len(patches)
        return cls(
            patches.pair_ids,
            jnp.zeros((count,), dtype=dtype_),
            patches.normals.astype(dtype_),
            jnp.zeros((count, patches.dimension), dtype=dtype_),
            jnp.zeros((count, patches.dimension), dtype=dtype_),
            jnp.zeros((count,), dtype=jnp.int32),
            epoch=patches.epoch,
            law_id=law_id,
            patch_set_id=patches.patch_set_id,
            state_version=state_version,
        )

    def for_patches(self, patches: ContactPatchSet, /) -> AcceptedContactState:
        """Transfer accepted history by pair identity into a later search epoch."""
        if not isinstance(patches, ContactPatchSet):
            raise TypeError("Contact history transfer requires ContactPatchSet.")
        if self.tangential_traction.shape[1] != patches.dimension:
            raise ValueError("Contact history and patch dimensions disagree.")
        if patches.epoch < self.epoch:
            raise ValueError("Accepted contact history cannot move to an earlier epoch.")
        if patches.epoch == self.epoch:
            if patches.patch_set_id != self.patch_set_id:
                raise ValueError(
                    "Contact patches cannot change within an accepted epoch."
                )
            return self
        old_index = {pair_id: index for index, pair_id in enumerate(self.pair_ids)}
        count = len(patches)
        pressure = np.zeros((count,), dtype=np.asarray(self.normal_pressure).dtype)
        normals = np.asarray(patches.normals).copy()
        traction = np.zeros(
            (count, patches.dimension), dtype=np.asarray(self.tangential_traction).dtype
        )
        slip = np.zeros_like(traction)
        mode = np.zeros((count,), dtype=np.int32)
        for new_index, pair_id in enumerate(patches.pair_ids):
            previous = old_index.get(pair_id)
            if previous is None:
                continue
            pressure[new_index] = np.asarray(self.normal_pressure)[previous]
            normals[new_index] = np.asarray(self.contact_normals)[previous]
            traction[new_index] = np.asarray(self.tangential_traction)[previous]
            slip[new_index] = np.asarray(self.accumulated_slip)[previous]
            mode[new_index] = np.asarray(self.mode)[previous]
        return AcceptedContactState(
            patches.pair_ids,
            pressure,
            normals,
            traction,
            slip,
            mode,
            epoch=patches.epoch,
            law_id=self.law_id,
            patch_set_id=patches.patch_set_id,
            state_version=self.state_version,
        )

    def promote(self, /) -> AcceptedContactState:
        return AcceptedContactState(
            self.pair_ids,
            self.normal_pressure,
            self.contact_normals,
            self.tangential_traction,
            self.accumulated_slip,
            self.mode,
            epoch=self.epoch,
            law_id=self.law_id,
            patch_set_id=self.patch_set_id,
            state_version=self.state_version + 1,
        )


class ContactEvaluation(StrictModule):
    """Contact forces, complementarity, conservation, and dissipation evidence."""

    query: ContactQueryResult
    gap: Array
    normals: Array
    closest_points: Array
    normal_pressure: Array
    trial_accumulated_slip: Array
    tangential_traction: Array
    traction: Array
    normal_tangent: Array
    active: Array
    mode: Array
    relative_displacement_increment: Array
    transport_ambiguous: Array
    transport_defect: Array
    complementarity_residual: Array
    primal_violation: Array
    dual_violation: Array
    friction_cone_violation: Array
    active_set_ambiguous: Array
    plus_patch_forces: Array
    minus_patch_forces: Array
    plus_nodal_forces: Array
    minus_nodal_forces: Array
    pair_dissipation: Array
    patch_action_reaction_defect: Array
    action_reaction_defect: Array
    dissipation: Array
    maximum_penetration: Array
    total_reaction: Array
    finite: Array
    epoch: int = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)


class ContactStateTransaction(StrictModule, NonTrainableState):
    """Trial contact history with explicit accepted commit and exact rollback."""

    base: AcceptedContactState
    trial: AcceptedContactState
    evaluation: ContactEvaluation
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: AcceptedContactState,
        trial: AcceptedContactState,
        evaluation: ContactEvaluation,
        /,
    ):
        if not isinstance(base, AcceptedContactState) or not isinstance(
            trial, AcceptedContactState
        ):
            raise TypeError(
                "Contact transaction requires accepted base and trial states."
            )
        if not isinstance(evaluation, ContactEvaluation):
            raise TypeError("Contact transaction requires ContactEvaluation evidence.")
        if (
            base.pair_ids != trial.pair_ids
            or base.epoch != trial.epoch
            or base.state_version != trial.state_version
            or base.law_id != trial.law_id
            or base.patch_set_id != trial.patch_set_id
            or trial.patch_set_id != evaluation.query.patches.patch_set_id
            or trial.law_id != evaluation.law_id
            or trial.epoch != evaluation.epoch
        ):
            raise ValueError("Contact transaction base, trial, and evaluation disagree.")
        self.base = base
        self.trial = trial
        self.evaluation = evaluation
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "contact-state-transaction",
                "base": base.state_id,
                "trial": trial.state_id,
                "query": evaluation.query.query_id,
            }
        )

    def commit(self) -> AcceptedContactState:
        return self.trial.promote()

    def rollback(self) -> AcceptedContactState:
        return self.base

    def resolve(self, accepted: bool, /) -> AcceptedContactState:
        return self.commit() if bool(accepted) else self.rollback()


class ContactEpochTransaction(StrictModule, NonTrainableState):
    """Atomic search-epoch transition retaining the exact prior accepted state."""

    source: AcceptedContactState
    candidate: ContactStateTransaction
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: AcceptedContactState,
        candidate: ContactStateTransaction,
        /,
    ):
        if not isinstance(source, AcceptedContactState) or not isinstance(
            candidate, ContactStateTransaction
        ):
            raise TypeError(
                "Contact epoch transaction requires accepted source and candidate."
            )
        if (
            candidate.base.epoch < source.epoch
            or candidate.base.state_version != source.state_version
            or candidate.base.law_id != source.law_id
        ):
            raise ValueError(
                "Contact epoch candidate does not descend from its source state."
            )
        self.source = source
        self.candidate = candidate
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "contact-epoch-transaction",
                "source": source.state_id,
                "candidate": candidate.transaction_id,
            }
        )

    @property
    def evaluation(self) -> ContactEvaluation:
        return self.candidate.evaluation

    def commit(self) -> AcceptedContactState:
        return self.candidate.commit()

    def rollback(self) -> AcceptedContactState:
        return self.source

    def resolve(self, accepted: bool, /) -> AcceptedContactState:
        return self.commit() if bool(accepted) else self.rollback()


__all__ = [
    "AcceptedContactState",
    "CONTACT_OPEN",
    "CONTACT_SLIP",
    "CONTACT_STICK",
    "ContactEpochTransaction",
    "ContactEvaluation",
    "ContactStateTransaction",
]
