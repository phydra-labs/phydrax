#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array

from ...._strict import StrictModule
from ._operator import OperatorBatch, stack_operator_batches


class OperatorSupervisedExample(StrictModule):
    """One case-aligned operator input/query batch and its observed target field."""

    batch: OperatorBatch
    targets: Array

    def __init__(self, batch: OperatorBatch, targets: Array, /):
        target = jnp.asarray(targets)
        prefix = batch.case_shape + batch.require_single_query().sample_shape
        scalar = tuple(int(size) for size in target.shape) == prefix
        channel_last = (
            target.ndim == len(prefix) + 1
            and tuple(int(size) for size in target.shape[:-1]) == prefix
            and int(target.shape[-1]) > 0
        )
        if not scalar and not channel_last:
            raise ValueError(
                "Operator demonstration targets must have case/query shape, "
                "optionally followed by one channel axis."
            )
        self.batch = batch
        self.targets = target

    @property
    def target_channels(self) -> int:
        prefix_rank = len(self.batch.case_shape) + len(
            self.batch.require_single_query().sample_shape
        )
        return 1 if self.targets.ndim == prefix_rank else int(self.targets.shape[-1])


class OperatorPrompt(StrictModule):
    """Fixed-capacity, case-aligned demonstration set with explicit padding mask."""

    examples: tuple[OperatorSupervisedExample, ...]
    mask: Array
    case_shape: tuple[int, ...]

    def __init__(
        self,
        examples: Sequence[OperatorSupervisedExample],
        /,
        *,
        mask: Array | None = None,
    ):
        resolved = tuple(examples)
        if not resolved:
            raise ValueError("OperatorPrompt requires at least one capacity slot.")
        first = resolved[0]
        if any(example.batch.case_shape != first.batch.case_shape for example in resolved):
            raise ValueError("Every prompt example must share one case shape.")
        if any(example.target_channels != first.target_channels for example in resolved):
            raise ValueError("Every prompt target must use the same channel count.")
        cases = first.batch.case_shape
        expected = cases + (len(resolved),)
        mask_ = (
            jnp.ones(expected, dtype=bool)
            if mask is None
            else jnp.asarray(mask, dtype=bool)
        )
        if tuple(int(size) for size in mask_.shape) != expected:
            raise ValueError(f"Operator prompt mask must have shape {expected}.")
        self.examples = resolved
        self.mask = mask_
        self.case_shape = cases

    @property
    def capacity(self) -> int:
        return len(self.examples)

    @property
    def target_channels(self) -> int:
        return self.examples[0].target_channels

    def permute(self, permutation: Sequence[int], /) -> "OperatorPrompt":
        indices = tuple(int(index) for index in permutation)
        if sorted(indices) != list(range(self.capacity)):
            raise ValueError("Prompt permutation must contain every capacity index once.")
        return OperatorPrompt(
            tuple(self.examples[index] for index in indices),
            mask=jnp.take(self.mask, jnp.asarray(indices), axis=-1),
        )


class PromptedOperatorBatch(StrictModule):
    """A query operator batch paired with case-aligned in-context demonstrations."""

    batch: OperatorBatch
    prompt: OperatorPrompt

    def __init__(self, batch: OperatorBatch, prompt: OperatorPrompt, /):
        if batch.case_shape != prompt.case_shape:
            raise ValueError("Prompt and query batch case shapes must match.")
        self.batch = batch
        self.prompt = prompt

    @property
    def case_shape(self) -> tuple[int, ...]:
        return self.batch.case_shape


def pad_operator_prompt(prompt: OperatorPrompt, capacity: int, /) -> OperatorPrompt:
    """Pad a prompt by masked repetition without inventing dummy geometries."""
    target = int(capacity)
    if target < prompt.capacity:
        raise ValueError("Prompt capacity cannot shrink during padding.")
    if target == prompt.capacity:
        return prompt
    added = target - prompt.capacity
    examples = prompt.examples + (prompt.examples[0],) * added
    padding = jnp.zeros(prompt.case_shape + (added,), dtype=bool)
    return OperatorPrompt(examples, mask=jnp.concatenate((prompt.mask, padding), axis=-1))


def stack_operator_prompts(
    prompts: Sequence[OperatorPrompt],
    /,
    *,
    axis_name: str = "case",
) -> OperatorPrompt:
    """Stack prompts along a new leading case axis, slot by slot."""
    resolved = tuple(prompts)
    if not resolved:
        raise ValueError("Cannot stack an empty prompt sequence.")
    capacity = resolved[0].capacity
    if any(prompt.capacity != capacity for prompt in resolved):
        raise ValueError("Stacked prompts must have equal capacity.")
    examples = tuple(
        OperatorSupervisedExample(
            stack_operator_batches(
                tuple(prompt.examples[index].batch for prompt in resolved),
                case_axis=axis_name,
            ),
            jnp.stack(
                tuple(prompt.examples[index].targets for prompt in resolved), axis=0
            ),
        )
        for index in range(capacity)
    )
    return OperatorPrompt(examples, mask=jnp.stack(tuple(prompt.mask for prompt in resolved)))


__all__ = [
    "OperatorPrompt",
    "OperatorSupervisedExample",
    "PromptedOperatorBatch",
    "pad_operator_prompt",
    "stack_operator_prompts",
]
