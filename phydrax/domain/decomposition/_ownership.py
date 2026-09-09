#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._components import DomainComponent
from .._function import DomainFunction
from ._cover import SubdomainCover


def _dependency_union(fields: tuple[DomainFunction, ...], /) -> tuple[str, ...]:
    labels: list[str] = []
    for field in fields:
        for label in field.deps:
            if label not in labels:
                labels.append(label)
    return tuple(labels)


class _NormalizedWeight(StrictModule, NonTrainableState):
    fields: tuple[DomainFunction, ...]
    positions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    index: int = eqx.field(static=True)

    def __init__(
        self,
        fields: tuple[DomainFunction, ...],
        deps: tuple[str, ...],
        index: int,
    ):
        by_label = {label: position for position, label in enumerate(deps)}
        self.fields = fields
        self.positions = tuple(
            tuple(by_label[label] for label in field.deps) for field in fields
        )
        self.index = int(index)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        values = jnp.stack(
            tuple(
                jnp.asarray(
                    field.func(
                        *(args[position] for position in positions),
                        key=key,
                        **kwargs,
                    )
                )
                for field, positions in zip(self.fields, self.positions, strict=True)
            )
        )
        denominator = jnp.sum(values)
        return jnp.where(denominator > 0.0, values[self.index] / denominator, jnp.nan)


class IntegrationOwnershipEvidence(StrictModule, NonTrainableState):
    cover_id: str = eqx.field(static=True)
    maximum_sum_defect: float = eqx.field(static=True)
    minimum_weight: float = eqx.field(static=True)
    verified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        cover_id: str,
        maximum_sum_defect: float,
        minimum_weight: float,
        verified: bool,
    ):
        self.cover_id = str(cover_id)
        self.maximum_sum_defect = float(maximum_sum_defect)
        self.minimum_weight = float(minimum_weight)
        self.verified = bool(verified)


class IntegrationOwnership(StrictModule, NonTrainableState):
    """A non-negative partition of physical integration measure over a cover."""

    cover: SubdomainCover
    weights: tuple[DomainFunction, ...]
    kind: Literal["window", "support"] = eqx.field(static=True)

    def __init__(
        self,
        cover: SubdomainCover,
        weights: tuple[DomainFunction, ...],
        /,
        *,
        kind: Literal["window", "support"],
    ):
        if not isinstance(cover, SubdomainCover):
            raise TypeError("cover must be a SubdomainCover.")
        weights_ = tuple(weights)
        if len(weights_) != len(cover.patches):
            raise ValueError("Ownership requires one weight per cover patch.")
        for weight in weights_:
            if not isinstance(weight, DomainFunction):
                raise TypeError("Ownership weights must be DomainFunction objects.")
            if not weight.domain.same_support(cover.ambient):
                raise ValueError("Ownership weights must live on the ambient domain.")
        if kind not in ("window", "support"):
            raise ValueError("Unknown integration ownership kind.")
        self.cover = cover
        self.weights = weights_
        self.kind = kind

    def weight(self, patch_id: str, /) -> DomainFunction:
        patch = self.cover.patch(patch_id)
        return self.weights[self.cover.patches.index(patch)]

    def local_weight(self, patch_id: str, /) -> DomainFunction:
        patch = self.cover.patch(patch_id)
        return patch.restrict(self.weight(patch_id))

    def component(self, patch_id: str, /) -> DomainComponent:
        patch = self.cover.patch(patch_id)
        return patch.interior.with_density(self.local_weight(patch_id))

    def audit(
        self,
        points: Any,
        /,
        *,
        tolerance: float = 1.0e-8,
    ) -> IntegrationOwnershipEvidence:
        values = jnp.stack(
            tuple(jnp.asarray(weight(points).data) for weight in self.weights),
            axis=0,
        )
        defect = float(jnp.max(jnp.abs(jnp.sum(values, axis=0) - 1.0)))
        minimum = float(jnp.min(values))
        return IntegrationOwnershipEvidence(
            cover_id=self.cover.cover_id,
            maximum_sum_defect=defect,
            minimum_weight=minimum,
            verified=defect <= float(tolerance) and minimum >= -float(tolerance),
        )


def cover_integration_ownership(
    cover: SubdomainCover,
    /,
    *,
    kind: Literal["window", "support"] = "window",
) -> IntegrationOwnership:
    """Build normalized fixed ownership weights from windows or supports."""
    if not isinstance(cover, SubdomainCover):
        raise TypeError("cover must be a SubdomainCover.")
    if kind == "window":
        fields = tuple(patch.window for patch in cover.patches)
        if any(field is None for field in fields):
            raise ValueError("Window ownership requires a window on every patch.")
        raw = tuple(field for field in fields if field is not None)
    elif kind == "support":
        raw = tuple(patch.support for patch in cover.patches)
    else:
        raise ValueError("kind must be 'window' or 'support'.")
    deps = _dependency_union(raw)
    weights = tuple(
        DomainFunction(
            domain=cover.ambient,
            deps=deps,
            func=_NormalizedWeight(raw, deps, index),
            metadata={
                "cover_id": cover.cover_id,
                "ownership_kind": kind,
                "patch_id": patch.patch_id,
            },
        )
        for index, patch in enumerate(cover.patches)
    )
    return IntegrationOwnership(cover, weights, kind=kind)


__all__ = [
    "IntegrationOwnership",
    "IntegrationOwnershipEvidence",
    "cover_integration_ownership",
]
