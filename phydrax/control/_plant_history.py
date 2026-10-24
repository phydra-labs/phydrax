"""Complete plant-state retention for control trajectories."""

from __future__ import annotations

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, core as jax_core
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics._plant import PlantRuntimeState


@final
class PlantRuntimeHistory(StrictModule, NonTrainableState):
    """Time-leading complete states; indexing restores an ordinary runtime state."""

    payload: Any
    time: Array
    step_index: Array
    key: Array
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)

    def __init__(self, states: PlantRuntimeState, /) -> None:
        if states.time.ndim < 1 or states.time.shape[0] == 0:
            raise ValueError(
                "Plant runtime history requires a nonempty leading time axis."
            )
        if (
            states.step_index.shape != states.time.shape
            or states.key.shape != states.time.shape
        ):
            raise ValueError("History clock, step and typed-key shapes must agree.")
        if not jax.dtypes.issubdtype(states.key.dtype, jax.dtypes.prng_key):
            raise TypeError("Plant runtime history requires typed PRNG keys.")
        for leaf in jax.tree.leaves(states.payload):
            if (
                not isinstance(leaf, (Array, jax_core.Tracer))
                or leaf.shape[: states.time.ndim] != states.time.shape
            ):
                raise ValueError(
                    "Every history payload leaf must carry the declared time/case axes."
                )
        self.payload = states.payload
        self.time = states.time
        self.step_index = states.step_index
        self.key = states.key
        self.semantic_provenance_id = states.semantic_provenance_id
        self.numeric_revision_id = states.numeric_revision_id
        self.state_schema_id = states.state_schema_id
        self.execution_signature_id = states.execution_signature_id

    @property
    def node_count(self) -> int:
        return self.time.shape[0]

    def at(self, index: ArrayLike, /) -> PlantRuntimeState:
        selected = jnp.asarray(index)
        if selected.shape != () or not jnp.issubdtype(selected.dtype, jnp.integer):
            raise TypeError("History index must be a scalar integer.")
        selected = eqx.error_if(
            selected,
            (selected < 0) | (selected >= self.node_count),
            "Plant history index is outside retained nodes.",
        )
        return PlantRuntimeState(
            jax.tree.map(lambda leaf: leaf[selected], self.payload),
            self.time[selected],
            self.step_index[selected],
            self.key[selected],
            self.semantic_provenance_id,
            self.numeric_revision_id,
            self.state_schema_id,
            self.execution_signature_id,
        )


@final
class PlantControlHistory(StrictModule, NonTrainableState):
    """Complete states, optionally with an explicitly declared realization axis.

    A realization ensemble retains time-leading histories. Select one model
    before restoring its ordinary plant state; realizations are not plant cases.
    """

    initial_state: PlantRuntimeState
    final_state: PlantRuntimeState
    final_candidate_state: PlantRuntimeState
    accepted: PlantRuntimeHistory | None
    candidates: PlantRuntimeHistory | None
    retained_bytes: int = eqx.field(static=True)
    realization_count: int | None = eqx.field(static=True)

    def __init__(
        self,
        initial_state: PlantRuntimeState,
        final_state: PlantRuntimeState,
        final_candidate_state: PlantRuntimeState,
        accepted: PlantRuntimeHistory | None,
        candidates: PlantRuntimeHistory | None,
        retained_bytes: int,
        /,
        *,
        realization_count: int | None = None,
    ) -> None:
        if (accepted is None) != (candidates is None):
            raise ValueError(
                "Accepted and candidate histories must be retained together."
            )
        if retained_bytes < 0:
            raise ValueError("Retained storage must be nonnegative.")
        identity = (
            initial_state.semantic_provenance_id,
            initial_state.numeric_revision_id,
            initial_state.state_schema_id,
            initial_state.execution_signature_id,
        )
        for state in (final_state, final_candidate_state):
            if (
                state.semantic_provenance_id,
                state.numeric_revision_id,
                state.state_schema_id,
                state.execution_signature_id,
            ) != identity:
                raise ValueError(
                    "Plant control history cannot combine foreign runtime identities."
                )
        if any(
            state.time.shape != initial_state.time.shape
            for state in (final_state, final_candidate_state)
        ):
            raise ValueError("Plant history endpoint case shapes must agree.")
        if realization_count is not None:
            if (
                isinstance(realization_count, bool)
                or not isinstance(realization_count, int)
                or realization_count < 1
            ):
                raise ValueError("realization_count must be a positive integer.")
            if (
                not initial_state.time.shape
                or initial_state.time.shape[0] != realization_count
                or retained_bytes % realization_count != 0
            ):
                raise ValueError("Plant history realization axis or storage is invalid.")
        for history in (accepted, candidates):
            if history is not None and (
                history.time.shape[1:] != initial_state.time.shape
                or (
                    history.semantic_provenance_id,
                    history.numeric_revision_id,
                    history.state_schema_id,
                    history.execution_signature_id,
                )
                != identity
            ):
                raise ValueError("Retained plant history axes or identities disagree.")
        if (
            accepted is not None
            and candidates is not None
            and accepted.node_count != candidates.node_count + 1
        ):
            raise ValueError(
                "Accepted history must contain the initial node and every transition."
            )
        self.initial_state = initial_state
        self.final_state = final_state
        self.final_candidate_state = final_candidate_state
        self.accepted = accepted
        self.candidates = candidates
        self.retained_bytes = retained_bytes
        self.realization_count = realization_count

    def state_at(self, index: ArrayLike, /) -> PlantRuntimeState:
        if self.realization_count is not None:
            raise ValueError("Select a realization before restoring a plant state.")
        if self.accepted is None:
            raise ValueError(
                "Endpoint-only retention does not provide intermediate complete states."
            )
        return self.accepted.at(index)

    def select_realization(self, index: ArrayLike, /) -> PlantControlHistory:
        """Remove the explicit model axis without changing the plant's case axes."""
        count = self.realization_count
        if count is None:
            raise ValueError("Plant history has no explicit realization axis.")
        selected = jnp.asarray(index)
        if selected.shape != () or not jnp.issubdtype(selected.dtype, jnp.integer):
            raise TypeError("Realization index must be a scalar integer.")
        selected = eqx.error_if(
            selected,
            (selected < 0) | (selected >= count),
            "Realization index is outside retained models.",
        )
        return PlantControlHistory(
            jax.tree.map(lambda leaf: leaf[selected], self.initial_state),
            jax.tree.map(lambda leaf: leaf[selected], self.final_state),
            jax.tree.map(lambda leaf: leaf[selected], self.final_candidate_state),
            (
                None
                if self.accepted is None
                else jax.tree.map(lambda leaf: leaf[:, selected], self.accepted)
            ),
            (
                None
                if self.candidates is None
                else jax.tree.map(lambda leaf: leaf[:, selected], self.candidates)
            ),
            self.retained_bytes // count,
        )


__all__ = ["PlantControlHistory", "PlantRuntimeHistory"]
