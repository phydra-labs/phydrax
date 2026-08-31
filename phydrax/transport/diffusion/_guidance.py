#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._frozendict import frozendict
from ..._score_field import StateTimeScoreField
from ..._strict import AbstractAttribute, StrictModule
from ...domain import DomainFunction


GuidanceExactness = Literal["exact", "approximate", "heuristic"]


class ScoreContext(StrictModule):
    """One immutable named context shared by all score evaluations in a realization."""

    values: frozendict[str, Array]
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: Mapping[str, ArrayLike],
        /,
        *,
        context_id: str | None = None,
    ):
        converted = tuple((str(name), jnp.asarray(value)) for name, value in values.items())
        names = tuple(name for name, _ in converted)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Score context names must be unique and non-empty.")
        resolved = frozendict(sorted(converted, key=lambda item: item[0]))
        identifier = context_id or canonical_fingerprint(
            {
                "kind": "score-context",
                "names": tuple(resolved),
                "shapes": {name: list(value.shape) for name, value in resolved.items()},
            }
        )
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("context_id must be non-empty or None.")
        self.values = resolved
        self.context_id = identifier


class GuidanceEvaluation(StrictModule):
    correction: Array
    valid: Array
    exactness: GuidanceExactness = eqx.field(static=True)
    guidance_id: str = eqx.field(static=True)


class AbstractScoreGuidance(StrictModule):
    exactness: AbstractAttribute[GuidanceExactness]
    guidance_id: AbstractAttribute[str]

    @abstractmethod
    def evaluate(
        self,
        state: ArrayLike,
        time: ArrayLike,
        context: ScoreContext,
        /,
        *,
        key: Key[Array, ""] | None = None,
    ) -> GuidanceEvaluation:
        raise NotImplementedError


class _AbstractScalarFieldGradientGuidance(AbstractScoreGuidance):
    field: DomainFunction
    state_label: str = eqx.field(static=True)
    time_label: str = eqx.field(static=True)
    context_labels: tuple[str, ...] = eqx.field(static=True)
    scale: float = eqx.field(static=True)
    exactness: GuidanceExactness = eqx.field(static=True)
    guidance_id: str = eqx.field(static=True)

    def __init__(
        self,
        field: DomainFunction,
        /,
        *,
        state_label: str,
        time_label: str,
        context_labels: Sequence[str],
        scale: float,
        exactness: GuidanceExactness,
        guidance_id: str,
    ):
        if not isinstance(field, DomainFunction):
            raise TypeError("Guidance field must be a DomainFunction.")
        contexts = tuple(str(name) for name in context_labels)
        if (
            not state_label
            or not time_label
            or state_label == time_label
            or any(not name for name in contexts)
            or len(set(contexts)) != len(contexts)
            or state_label in contexts
            or time_label in contexts
        ):
            raise ValueError("Guidance state, time, and context labels must be distinct.")
        allowed = {state_label, time_label, *contexts}
        if state_label not in field.deps or any(dep not in allowed for dep in field.deps):
            raise ValueError("Guidance field dependencies exceed the declared labels.")
        value = float(scale)
        if not jnp.isfinite(value):
            raise ValueError("Guidance scale must be finite.")
        if exactness not in ("exact", "approximate", "heuristic"):
            raise ValueError("Unknown guidance exactness.")
        if not guidance_id:
            raise ValueError("guidance_id must be non-empty.")
        self.field = field
        self.state_label = state_label
        self.time_label = time_label
        self.context_labels = contexts
        self.scale = value
        self.exactness = exactness
        self.guidance_id = guidance_id

    def evaluate(self, state, time, context, /, *, key=None) -> GuidanceEvaluation:
        if not isinstance(context, ScoreContext):
            raise TypeError("context must be a ScoreContext.")
        required = tuple(
            dep
            for dep in self.field.deps
            if dep not in (self.state_label, self.time_label)
        )
        if any(name not in context.values for name in required):
            raise ValueError("Score context is missing a guidance dependency.")
        if jnp.asarray(time).shape != ():
            raise ValueError("One guidance evaluation requires scalar time.")
        time_array = jnp.asarray(time)

        def scalar(current):
            arguments = tuple(
                current
                if dep == self.state_label
                else (
                    time_array
                    if dep == self.time_label
                    else context.values[dep]
                )
                for dep in self.field.deps
            )
            value = jnp.asarray(self.field.func(*arguments, key=key))
            if value.shape != ():
                raise ValueError("Guidance potential/likelihood must return one scalar.")
            return value

        correction = self.scale * jax.grad(scalar)(jnp.asarray(state))
        valid = jnp.all(jnp.isfinite(correction))
        return GuidanceEvaluation(correction, valid, self.exactness, self.guidance_id)


class TimeConditionedLikelihoodGuidance(_AbstractScalarFieldGradientGuidance):
    """Exact noised-state likelihood score correction."""

    def __init__(
        self,
        log_likelihood: DomainFunction,
        /,
        *,
        state_label: str = "x",
        time_label: str = "t",
        context_labels: Sequence[str] = (),
        guidance_id: str = "time-conditioned-likelihood",
    ):
        super().__init__(
            log_likelihood,
            state_label=state_label,
            time_label=time_label,
            context_labels=context_labels,
            scale=1.0,
            exactness="exact",
            guidance_id=guidance_id,
        )


class PotentialGuidance(_AbstractScalarFieldGradientGuidance):
    """Score correction from one explicitly scaled log potential."""

    def __init__(
        self,
        potential: DomainFunction,
        /,
        *,
        scale: float = 1.0,
        exactness: GuidanceExactness = "approximate",
        state_label: str = "x",
        time_label: str = "t",
        context_labels: Sequence[str] = (),
        guidance_id: str = "potential-guidance",
    ):
        super().__init__(
            potential,
            state_label=state_label,
            time_label=time_label,
            context_labels=context_labels,
            scale=scale,
            exactness=exactness,
            guidance_id=guidance_id,
        )


class ClassifierFreeGuidance(AbstractScoreGuidance):
    unconditional: StateTimeScoreField
    conditional: StateTimeScoreField
    weight: float = eqx.field(static=True)
    exactness: GuidanceExactness = eqx.field(static=True)
    guidance_id: str = eqx.field(static=True)

    def __init__(
        self,
        unconditional: StateTimeScoreField,
        conditional: StateTimeScoreField,
        /,
        *,
        weight: float,
        guidance_id: str = "classifier-free-guidance",
    ):
        if not isinstance(unconditional, StateTimeScoreField) or not isinstance(
            conditional, StateTimeScoreField
        ):
            raise TypeError("Classifier-free guidance requires two score fields.")
        value = float(weight)
        if not jnp.isfinite(value):
            raise ValueError("Classifier-free guidance weight must be finite.")
        if not guidance_id:
            raise ValueError("guidance_id must be non-empty.")
        self.unconditional = unconditional
        self.conditional = conditional
        self.weight = value
        self.exactness = "exact" if value == 1.0 else "heuristic"
        self.guidance_id = guidance_id

    def evaluate(self, state, time, context, /, *, key=None) -> GuidanceEvaluation:
        if not isinstance(context, ScoreContext):
            raise TypeError("context must be a ScoreContext.")
        conditional_names = tuple(
            dependency
            for dependency in self.conditional.function.deps
            if dependency
            not in (self.conditional.state_label, self.conditional.time_label)
        )
        unconditional_names = tuple(
            dependency
            for dependency in self.unconditional.function.deps
            if dependency
            not in (self.unconditional.state_label, self.unconditional.time_label)
        )
        required = conditional_names + unconditional_names
        if any(name not in context.values for name in required):
            raise ValueError("Score context is missing a classifier-free dependency.")
        conditional = self.conditional(
            state,
            time,
            key=key,
            context={name: context.values[name] for name in conditional_names},
        )
        unconditional = self.unconditional(
            state,
            time,
            key=key,
            context={name: context.values[name] for name in unconditional_names},
        )
        correction = self.weight * (conditional - unconditional)
        return GuidanceEvaluation(
            correction,
            jnp.all(jnp.isfinite(correction)),
            self.exactness,
            self.guidance_id,
        )


class GuidedScoreField(StrictModule):
    """Ordered composition of a base score and explicit guidance corrections."""

    base: StateTimeScoreField
    guidance: tuple[AbstractScoreGuidance, ...]
    guided_score_id: str = eqx.field(static=True)

    def __init__(
        self,
        base: StateTimeScoreField,
        guidance: Sequence[AbstractScoreGuidance],
        /,
        *,
        guided_score_id: str | None = None,
    ):
        if not isinstance(base, StateTimeScoreField):
            raise TypeError("base must be a StateTimeScoreField.")
        values = tuple(guidance)
        if any(not isinstance(item, AbstractScoreGuidance) for item in values):
            raise TypeError("guidance must contain AbstractScoreGuidance objects.")
        if any(
            isinstance(item, ClassifierFreeGuidance) and item.unconditional is not base
            for item in values
        ):
            raise ValueError(
                "Classifier-free correction requires its exact unconditional field as base."
            )
        identifier = guided_score_id or canonical_fingerprint(
            {
                "kind": "guided-score-field",
                "base_dependencies": base.function.deps,
                "guidance_ids": tuple(item.guidance_id for item in values),
            }
        )
        if not identifier:
            raise ValueError("guided_score_id must be non-empty or None.")
        self.base = base
        self.guidance = values
        self.guided_score_id = identifier

    def evaluate(self, state, time, context, /, *, key=None):
        if not isinstance(context, ScoreContext):
            raise TypeError("context must be a ScoreContext.")
        required = tuple(
            dependency
            for dependency in self.base.function.deps
            if dependency not in (self.base.state_label, self.base.time_label)
        )
        if any(name not in context.values for name in required):
            raise ValueError("Score context is missing a base-score dependency.")
        base_context = {name: context.values[name] for name in required}
        score = self.base(state, time, key=key, context=base_context)
        evaluations = tuple(
            item.evaluate(state, time, context, key=key) for item in self.guidance
        )
        for evaluation in evaluations:
            score = score + evaluation.correction
        valid = jnp.all(jnp.isfinite(score)) & jnp.all(
            jnp.asarray([evaluation.valid for evaluation in evaluations], dtype=bool)
        )
        return score, evaluations, valid

    def __call__(self, state, time, /, *, context, key=None):
        return self.evaluate(state, time, context, key=key)[0]


__all__ = [
    "AbstractScoreGuidance",
    "ClassifierFreeGuidance",
    "GuidanceEvaluation",
    "GuidanceExactness",
    "GuidedScoreField",
    "PotentialGuidance",
    "ScoreContext",
    "TimeConditionedLikelihoodGuidance",
]
