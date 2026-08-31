#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, Key, PyTree

from ..._fingerprint import canonical_fingerprint
from ..._sampling import AbstractChainSampleResult, derive_key, SampleAddress
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


_CONDITIONAL_ADDRESS = SampleAddress(
    "conditional-program",
    "stage-update",
    target="group",
    role="sample",
)


class ConditionalVariableGroup(StrictModule, NonTrainableState):
    """Named homogeneous node group with an arbitrary PyTree state specification."""

    state_spec: PyTree[jax.ShapeDtypeStruct]
    name: str = eqx.field(static=True)
    count: int = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    def __init__(
        self, name: str, count: int, state_spec: PyTree[jax.ShapeDtypeStruct], /
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("Conditional variable-group name must be non-empty.")
        size = int(count)
        if size < 1:
            raise ValueError("Conditional variable-group count must be positive.")
        leaves = jax.tree_util.tree_leaves(state_spec)
        if not leaves or any(
            not isinstance(leaf, jax.ShapeDtypeStruct) for leaf in leaves
        ):
            raise TypeError(
                "state_spec must be a nonempty PyTree of ShapeDtypeStruct leaves."
            )
        self.name = name
        self.count = size
        self.state_spec = state_spec
        self.group_id = canonical_fingerprint(
            {
                "kind": "conditional-variable-group",
                "name": name,
                "tree": str(jax.tree_util.tree_structure(state_spec)),
                "count": size,
                "state": [
                    {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
                    for leaf in leaves
                ],
            }
        )


class ConditionalInteractionGroup(StrictModule):
    """Directed batched interaction from parallel tail nodes to head nodes."""

    head_indices: Array
    tail_indices: tuple[Array, ...]
    parameters: Any
    head_group: str = eqx.field(static=True)
    tail_groups: tuple[str, ...] = eqx.field(static=True)
    interaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        head_group: str,
        head_indices: Array,
        tail_groups: Sequence[str],
        tail_indices: Sequence[Array],
        parameters: Any,
        /,
        *,
        interaction_id: str,
    ):
        heads = jnp.asarray(head_indices, dtype=jnp.int32).reshape((-1,))
        groups = tuple(str(value) for value in tail_groups)
        tails = tuple(jnp.asarray(value, dtype=jnp.int32) for value in tail_indices)
        if heads.size == 0:
            raise ValueError("Conditional interactions require at least one head node.")
        if not head_group or any(not group for group in groups):
            raise ValueError("Interaction group names must be non-empty.")
        if len(groups) != len(tails):
            raise ValueError("tail_groups and tail_indices must have equal lengths.")
        if any(tail.shape != heads.shape for tail in tails):
            raise ValueError("Every tail index vector must match head_indices.")
        if not isinstance(interaction_id, str) or not interaction_id:
            raise ValueError("interaction_id must be non-empty.")
        self.head_group = str(head_group)
        self.head_indices = heads
        self.tail_groups = groups
        self.tail_indices = tails
        self.parameters = parameters
        parameter_signature = [
            {
                "shape": list(jnp.asarray(leaf).shape),
                "dtype": str(jnp.asarray(leaf).dtype),
            }
            for leaf in jax.tree_util.tree_leaves(parameters)
        ]
        self.interaction_id = canonical_fingerprint(
            {
                "kind": "conditional-interaction",
                "declared_id": interaction_id,
                "head_group": self.head_group,
                "head_indices": heads.tolist(),
                "tail_groups": list(groups),
                "tail_indices": [tail.tolist() for tail in tails],
                "parameter_signature": parameter_signature,
            }
        )


class AbstractConditionalKernel(StrictModule):
    """Stateful exact or approximate conditional update over one interaction group."""

    kernel_id: AbstractAttribute[str]

    @abstractmethod
    def initialize(self, state_spec: PyTree[jax.ShapeDtypeStruct], /) -> Any:
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        key: Key[Array, ""],
        parameters: Any,
        tails: tuple[Any, ...],
        current: Any,
        kernel_state: Any,
        /,
    ) -> tuple[Any, Any]:
        raise NotImplementedError


class CallableConditionalKernel(AbstractConditionalKernel):
    """Pure callable conditional sampler with explicit state initialization."""

    sample_function: Callable = eqx.field(static=True)
    initialize_function: Callable = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample: Callable,
        /,
        *,
        kernel_id: str,
        initialize: Callable | None = None,
    ):
        if not callable(sample):
            raise TypeError("sample must be callable.")
        if not isinstance(kernel_id, str) or not kernel_id:
            raise ValueError("kernel_id must be non-empty.")
        self.sample_function = sample
        self.initialize_function = (
            (lambda _spec: None) if initialize is None else initialize
        )
        self.kernel_id = kernel_id

    def initialize(self, state_spec, /):
        return self.initialize_function(state_spec)

    def sample(self, key, parameters, tails, current, kernel_state, /):
        return self.sample_function(key, parameters, tails, current, kernel_state)


class MetropolisWithinConditionalKernel(AbstractConditionalKernel):
    """Metropolis-within-Gibbs kernel from explicit proposal and local log-target callables."""

    proposal: Callable = eqx.field(static=True)
    proposal_log_prob: Callable = eqx.field(static=True)
    log_target: Callable = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(self, proposal, proposal_log_prob, log_target, /, *, kernel_id: str):
        if not all(
            callable(value) for value in (proposal, proposal_log_prob, log_target)
        ):
            raise TypeError("Metropolis conditional callables must be callable.")
        if not isinstance(kernel_id, str) or not kernel_id:
            raise ValueError("kernel_id must be non-empty.")
        self.proposal = proposal
        self.proposal_log_prob = proposal_log_prob
        self.log_target = log_target
        self.kernel_id = kernel_id

    def initialize(self, state_spec, /):
        del state_spec
        return jnp.asarray(0, dtype=jnp.uint32)

    def sample(self, key, parameters, tails, current, kernel_state, /):
        proposal_key = derive_key(key, _CONDITIONAL_ADDRESS, kernel_state, 0)
        acceptance_key = derive_key(key, _CONDITIONAL_ADDRESS, kernel_state, 1)
        proposed = self.proposal(proposal_key, current, parameters)
        current_target = self.log_target(current, tails, parameters)
        proposed_target = self.log_target(proposed, tails, parameters)
        forward = self.proposal_log_prob(proposed, current, parameters)
        reverse = self.proposal_log_prob(current, proposed, parameters)
        ratio = proposed_target - current_target + reverse - forward
        accepted = jnp.log(jr.uniform(acceptance_key, jnp.shape(ratio))) < jnp.minimum(
            ratio, 0.0
        )

        def select(candidate, previous):
            mask = accepted.reshape(
                accepted.shape + (1,) * (candidate.ndim - accepted.ndim)
            )
            return jnp.where(mask, candidate, previous)

        output = jax.tree_util.tree_map(select, proposed, current)
        return output, kernel_state + jnp.asarray(1, dtype=jnp.uint32)


class ConditionalUpdate(StrictModule):
    interaction: ConditionalInteractionGroup
    kernel: AbstractConditionalKernel


class ConditionalUpdateStage(StrictModule):
    """Updates whose heads are committed together from one immutable stage snapshot."""

    update_indices: tuple[int, ...] = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(self, update_indices: Sequence[int], /, *, stage_id: str):
        indices = tuple(int(value) for value in update_indices)
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError("Stage update indices must be unique and non-negative.")
        if not isinstance(stage_id, str) or not stage_id:
            raise ValueError("stage_id must be non-empty.")
        self.update_indices = indices
        self.stage_id = stage_id


class PreparedConditionalUpdateProgram(StrictModule):
    groups: tuple[ConditionalVariableGroup, ...]
    updates: tuple[ConditionalUpdate, ...]
    stages: tuple[ConditionalUpdateStage, ...]
    group_names: tuple[str, ...] = eqx.field(static=True)
    program_id: str = eqx.field(static=True)


class ConditionalProgramState(StrictModule):
    values: tuple[Any, ...]
    kernel_states: tuple[Any, ...]
    step_index: Array


class ConditionalSampleResult(AbstractChainSampleResult):
    samples: tuple[Any, ...]
    final_state: ConditionalProgramState
    root_key: Array
    program_id: str = eqx.field(static=True)
    draws: int = eqx.field(static=True)
    chains: int = eqx.field(static=True)

    @property
    def num_chains(self) -> int:
        return self.chains

    @property
    def num_draws(self) -> int:
        return self.draws

    @property
    def chain_provenance(self) -> str:
        return f"markov:conditional-program:{self.program_id}"


def prepare_conditional_program(
    groups: Sequence[ConditionalVariableGroup],
    updates: Sequence[ConditionalUpdate],
    stages: Sequence[ConditionalUpdateStage],
    /,
) -> PreparedConditionalUpdateProgram:
    """Validate indices, group semantics, and same-stage dependency independence."""
    group_values = tuple(groups)
    update_values = tuple(updates)
    stage_values = tuple(stages)
    if not group_values or any(
        not isinstance(group, ConditionalVariableGroup) for group in group_values
    ):
        raise ValueError("groups must contain ConditionalVariableGroup values.")
    if not update_values or any(
        not isinstance(update, ConditionalUpdate) for update in update_values
    ):
        raise ValueError("updates must contain ConditionalUpdate values.")
    if not stage_values or any(
        not isinstance(stage, ConditionalUpdateStage) for stage in stage_values
    ):
        raise ValueError("stages must contain ConditionalUpdateStage values.")
    names = tuple(group.name for group in group_values)
    if len(set(names)) != len(names):
        raise ValueError("Conditional variable-group names must be unique.")
    lookup = {group.name: group for group in group_values}
    for update in update_values:
        interaction = update.interaction
        if interaction.head_group not in lookup or any(
            group not in lookup for group in interaction.tail_groups
        ):
            raise ValueError("Conditional interaction references an unknown group.")
        head_count = lookup[interaction.head_group].count
        heads = np.asarray(interaction.head_indices)
        if heads.size and (heads.min() < 0 or heads.max() >= head_count):
            raise ValueError("Conditional head index is out of bounds.")
        for group_name, indices in zip(interaction.tail_groups, interaction.tail_indices):
            values = np.asarray(indices)
            if values.size and (
                values.min() < 0 or values.max() >= lookup[group_name].count
            ):
                raise ValueError("Conditional tail index is out of bounds.")
    covered = [index for stage in stage_values for index in stage.update_indices]
    if sorted(covered) != list(range(len(update_values))):
        raise ValueError("Stages must cover every update exactly once.")
    for stage in stage_values:
        heads = set()
        for index in stage.update_indices:
            update = update_values[index]
            tagged = {
                (update.interaction.head_group, int(value))
                for value in np.asarray(update.interaction.head_indices)
            }
            if heads & tagged:
                raise ValueError("Same-stage updates cannot write the same node.")
            heads |= tagged
    program_id = canonical_fingerprint(
        {
            "kind": "conditional-update-program",
            "groups": [group.group_id for group in group_values],
            "updates": [
                [update.interaction.interaction_id, update.kernel.kernel_id]
                for update in update_values
            ],
            "stages": [
                [stage.stage_id, list(stage.update_indices)] for stage in stage_values
            ],
        }
    )
    return PreparedConditionalUpdateProgram(
        groups=group_values,
        updates=update_values,
        stages=stage_values,
        group_names=names,
        program_id=program_id,
    )


def initialize_conditional_program(
    program: PreparedConditionalUpdateProgram,
    values: Mapping[str, Any],
    /,
) -> ConditionalProgramState:
    """Validate chain-leading PyTree states and initialize every stateful kernel."""
    if set(values) != set(program.group_names):
        raise ValueError("Conditional initial-state keys must match group names.")
    state_values = []
    chain_count = None
    for group in program.groups:
        value = jax.tree_util.tree_map(jnp.asarray, values[group.name])
        if jax.tree_util.tree_structure(value) != jax.tree_util.tree_structure(
            group.state_spec
        ):
            raise TypeError("Conditional state PyTree must match its group state_spec.")
        leaves = jax.tree_util.tree_leaves(value)
        specifications = jax.tree_util.tree_leaves(group.state_spec)
        if any(leaf.ndim < 2 for leaf in leaves):
            raise ValueError("Conditional states require leading chain and node axes.")
        if any(int(leaf.shape[1]) != group.count for leaf in leaves):
            raise ValueError("Conditional state node axis does not match group count.")
        if any(
            leaf.shape[2:] != specification.shape or leaf.dtype != specification.dtype
            for leaf, specification in zip(leaves, specifications)
        ):
            raise ValueError(
                "Conditional state event shapes and dtypes must match state_spec."
            )
        current = int(leaves[0].shape[0])
        if any(int(leaf.shape[0]) != current for leaf in leaves):
            raise ValueError("Every conditional state leaf must share one chain axis.")
        if chain_count is None:
            chain_count = current
        elif current != chain_count:
            raise ValueError("Every conditional group must share one chain axis.")
        state_values.append(value)
    kernel_states = tuple(
        update.kernel.initialize(
            next(
                group.state_spec
                for group in program.groups
                if group.name == update.interaction.head_group
            )
        )
        for update in program.updates
    )
    return ConditionalProgramState(
        values=tuple(state_values),
        kernel_states=kernel_states,
        step_index=jnp.asarray(0, dtype=jnp.uint32),
    )


def conditional_program_step(
    program: PreparedConditionalUpdateProgram,
    state: ConditionalProgramState,
    key: Key[Array, ""],
    /,
) -> ConditionalProgramState:
    """Advance every validated stage while preserving immutable snapshot semantics."""
    values = list(state.values)
    kernel_states = list(state.kernel_states)
    lookup = {name: index for index, name in enumerate(program.group_names)}
    for stage_index, stage in enumerate(program.stages):
        snapshot = tuple(values)
        pending = []
        for update_index in stage.update_indices:
            update = program.updates[update_index]
            interaction = update.interaction
            head_group = lookup[interaction.head_group]
            current = jax.tree_util.tree_map(
                lambda leaf, indices=interaction.head_indices: leaf[:, indices],
                snapshot[head_group],
            )
            tails = tuple(
                jax.tree_util.tree_map(
                    lambda leaf, indices=indices: leaf[:, indices],
                    snapshot[lookup[group_name]],
                )
                for group_name, indices in zip(
                    interaction.tail_groups,
                    interaction.tail_indices,
                )
            )
            subkey = derive_key(
                key,
                _CONDITIONAL_ADDRESS,
                state.step_index,
                stage_index,
                update_index,
            )
            output, kernel_state = update.kernel.sample(
                subkey,
                interaction.parameters,
                tails,
                current,
                kernel_states[update_index],
            )
            output = jax.tree_util.tree_map(jnp.asarray, output)
            if jax.tree_util.tree_structure(output) != jax.tree_util.tree_structure(
                current
            ):
                raise TypeError("Conditional kernel output must match the head PyTree.")
            if any(
                candidate.shape != previous.shape or candidate.dtype != previous.dtype
                for candidate, previous in zip(
                    jax.tree_util.tree_leaves(output),
                    jax.tree_util.tree_leaves(current),
                )
            ):
                raise ValueError(
                    "Conditional kernel output shapes and dtypes must match the heads."
                )
            pending.append((head_group, interaction.head_indices, output))
            kernel_states[update_index] = kernel_state
        for group_index, indices, output in pending:
            values[group_index] = jax.tree_util.tree_map(
                lambda base, update, indices=indices: base.at[:, indices].set(update),
                values[group_index],
                output,
            )
    return ConditionalProgramState(
        values=tuple(values),
        kernel_states=tuple(kernel_states),
        step_index=state.step_index + 1,
    )


def sample_conditional_program(
    program: PreparedConditionalUpdateProgram,
    state: ConditionalProgramState,
    /,
    *,
    key: Key[Array, ""],
    warmup_steps: int,
    num_draws: int,
    steps_per_draw: int = 1,
) -> ConditionalSampleResult:
    """Warm and retain arbitrary-PyTree conditional-program chain states."""
    warmup = int(warmup_steps)
    draws = int(num_draws)
    transitions = int(steps_per_draw)
    if warmup < 0 or draws < 1 or transitions < 1:
        raise ValueError("Conditional sampling counts are invalid.")

    def advance(carry, _):
        return conditional_program_step(program, carry, key), None

    warmed, _ = jax.lax.scan(advance, state, xs=None, length=warmup)

    def collect(carry, _):
        updated, _ = jax.lax.scan(advance, carry, xs=None, length=transitions)
        return updated, updated.values

    final, samples = jax.lax.scan(collect, warmed, xs=None, length=draws)
    samples = tuple(
        jax.tree_util.tree_map(lambda leaf: jnp.swapaxes(leaf, 0, 1), group)
        for group in samples
    )
    first_leaf = jax.tree_util.tree_leaves(samples[0])[0]
    return ConditionalSampleResult(
        samples=samples,
        final_state=final,
        root_key=key,
        program_id=program.program_id,
        draws=draws,
        chains=int(first_leaf.shape[0]),
    )


__all__ = [
    "AbstractConditionalKernel",
    "CallableConditionalKernel",
    "ConditionalInteractionGroup",
    "ConditionalProgramState",
    "ConditionalSampleResult",
    "ConditionalUpdate",
    "ConditionalUpdateStage",
    "ConditionalVariableGroup",
    "MetropolisWithinConditionalKernel",
    "PreparedConditionalUpdateProgram",
    "conditional_program_step",
    "initialize_conditional_program",
    "prepare_conditional_program",
    "sample_conditional_program",
]
