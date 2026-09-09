#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..._strict import StrictModule
from .._function import DomainFunction
from ._cover import SubdomainCover


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _dependency_union(fields: tuple[DomainFunction, ...], /) -> tuple[str, ...]:
    labels: list[str] = []
    for field in fields:
        for label in field.deps:
            if label not in labels:
                labels.append(label)
    return tuple(labels)


def _positions(
    field: DomainFunction,
    deps: tuple[str, ...],
    /,
) -> tuple[int, ...]:
    by_label = {label: index for index, label in enumerate(deps)}
    return tuple(by_label[label] for label in field.deps)


class LocalFieldRef(StrictModule):
    """Typed reference to one logical field on one cover patch."""

    field_id: str = eqx.field(static=True)
    patch_id: str = eqx.field(static=True)

    def __init__(self, field_id: str, patch_id: str, /):
        self.field_id = _identifier(field_id, "field_id")
        self.patch_id = _identifier(patch_id, "patch_id")

    @property
    def solver_name(self) -> str:
        return f"{self.field_id}::{self.patch_id}"


class LocalFieldFamily(StrictModule):
    """One logical field represented by one local function per cover patch."""

    cover: SubdomainCover
    fields: tuple[DomainFunction, ...]
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_id: str,
        cover: SubdomainCover,
        fields: Mapping[str, DomainFunction],
        /,
    ):
        if not isinstance(cover, SubdomainCover):
            raise TypeError("cover must be a SubdomainCover.")
        values = dict(fields)
        if set(values) != set(cover.patch_ids):
            raise ValueError(
                "fields must define exactly one local function for every cover patch."
            )
        ordered: list[DomainFunction] = []
        for patch in cover.patches:
            field = values[patch.patch_id]
            if not isinstance(field, DomainFunction):
                raise TypeError("Local fields must be DomainFunction objects.")
            if not field.domain.same_support(patch.domain):
                raise ValueError(
                    f"Local field for {patch.patch_id!r} must live on its patch domain."
                )
            ordered.append(field)
        self.field_id = _identifier(field_id, "field_id")
        self.cover = cover
        self.fields = tuple(ordered)

    def field(self, patch_id: str, /) -> DomainFunction:
        patch = self.cover.patch(patch_id)
        return self.fields[self.cover.patches.index(patch)]

    def ref(self, patch_id: str, /) -> LocalFieldRef:
        patch = self.cover.patch(patch_id)
        return LocalFieldRef(self.field_id, patch.patch_id)

    def field_name(self, patch_id: str, /) -> str:
        return self.ref(patch_id).solver_name

    def solver_functions(self) -> dict[str, DomainFunction]:
        return {
            self.field_name(patch.patch_id): field
            for patch, field in zip(self.cover.patches, self.fields, strict=True)
        }

    def lifted_fields(self) -> tuple[DomainFunction, ...]:
        return tuple(
            patch.lift(field)
            for patch, field in zip(self.cover.patches, self.fields, strict=True)
        )


class _PartitionOfUnityEvaluator(StrictModule):
    fields: tuple[DomainFunction, ...]
    supports: tuple[DomainFunction, ...]
    windows: tuple[DomainFunction, ...]
    cover: SubdomainCover
    field_positions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    support_positions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    window_positions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    maximum_overlap: int = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: LocalFieldFamily,
        deps: tuple[str, ...],
    ):
        lifted = family.lifted_fields()
        supports = tuple(patch.support for patch in family.cover.patches)
        windows = tuple(patch.window for patch in family.cover.patches)
        if any(window is None for window in windows):
            raise ValueError(
                "Partition-of-unity assembly requires a window on every patch."
            )
        overlap = family.cover.maximum_overlap
        if overlap is None:
            raise ValueError(
                "Partition-of-unity assembly requires a declared maximum_overlap."
            )
        self.fields = lifted
        self.supports = supports
        self.windows = tuple(window for window in windows if window is not None)
        self.cover = family.cover
        self.field_positions = tuple(_positions(field, deps) for field in lifted)
        self.support_positions = tuple(_positions(field, deps) for field in supports)
        self.window_positions = tuple(_positions(window, deps) for window in self.windows)
        self.maximum_overlap = int(overlap)
        self.field_id = family.field_id

    @staticmethod
    def _call(
        field: DomainFunction,
        positions: tuple[int, ...],
        args: tuple[Any, ...],
        *,
        key,
        kwargs: dict[str, Any],
    ):
        selected = tuple(args[position] for position in positions)
        return field.func(*selected, key=key, **kwargs)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        arguments = tuple(args)
        effective_key = jr.key(0) if key is None else key
        call_kwargs = dict(kwargs)
        supports = jnp.stack(
            tuple(
                jnp.asarray(
                    self._call(
                        field,
                        positions,
                        arguments,
                        key=effective_key,
                        kwargs=call_kwargs,
                    )
                )
                > 0
                for field, positions in zip(
                    self.supports,
                    self.support_positions,
                    strict=True,
                )
            )
        )
        weights = jnp.stack(
            tuple(
                jnp.asarray(
                    self._call(
                        field,
                        positions,
                        arguments,
                        key=key,
                        kwargs=call_kwargs,
                    )
                )
                for field, positions in zip(
                    self.windows,
                    self.window_positions,
                    strict=True,
                )
            )
        )
        active_count = jnp.sum(supports.astype(jnp.int32))
        active_indices = jnp.nonzero(
            supports,
            size=self.maximum_overlap,
            fill_value=0,
        )[0]

        branches = tuple(
            lambda operands, field=field, positions=positions, index=index: self._call(
                field,
                positions,
                operands[0],
                key=jr.fold_in(operands[1], index),
                kwargs=call_kwargs,
            )
            for index, (field, positions) in enumerate(
                zip(self.fields, self.field_positions, strict=True)
            )
        )
        first_index = active_indices[0]
        first_value = jax.lax.switch(
            first_index,
            branches,
            (arguments, effective_key),
        )
        first_weight = weights[first_index]
        numerator = first_weight * first_value
        denominator = first_weight

        for slot in range(1, self.maximum_overlap):
            index = active_indices[slot]
            valid = slot < active_count

            def contribution(_, index=index):
                value = jax.lax.switch(
                    index,
                    branches,
                    (arguments, effective_key),
                )
                return weights[index] * value, weights[index]

            def empty(_):
                return jnp.zeros_like(first_value), jnp.zeros_like(first_weight)

            value_part, weight_part = jax.lax.cond(
                valid,
                contribution,
                empty,
                operand=None,
            )
            numerator = numerator + value_part
            denominator = denominator + weight_part

        valid_point = (
            (active_count > 0)
            & (active_count <= self.maximum_overlap)
            & (denominator > 0)
        )
        return jnp.where(valid_point, numerator / denominator, jnp.nan)


class _BrokenFieldEvaluator(StrictModule):
    fields: tuple[DomainFunction, ...]
    supports: tuple[DomainFunction, ...]
    field_positions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    support_positions: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(self, family: LocalFieldFamily, deps: tuple[str, ...]):
        lifted = family.lifted_fields()
        supports = tuple(patch.support for patch in family.cover.patches)
        self.fields = lifted
        self.supports = supports
        self.field_positions = tuple(_positions(field, deps) for field in lifted)
        self.support_positions = tuple(_positions(field, deps) for field in supports)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        arguments = tuple(args)
        effective_key = jr.key(0) if key is None else key
        supports = jnp.stack(
            tuple(
                jnp.asarray(
                    field.func(
                        *(arguments[position] for position in positions),
                        key=effective_key,
                        **kwargs,
                    )
                )
                > 0
                for field, positions in zip(
                    self.supports,
                    self.support_positions,
                    strict=True,
                )
            )
        )
        index = jnp.argmax(supports.astype(jnp.int32))
        branches = tuple(
            lambda operands, field=field, positions=positions, branch=index_: field.func(
                *(operands[0][position] for position in positions),
                key=jr.fold_in(operands[1], branch),
                **kwargs,
            )
            for index_, (field, positions) in enumerate(
                zip(self.fields, self.field_positions, strict=True)
            )
        )
        value = jax.lax.switch(index, branches, (arguments, effective_key))
        return jnp.where(jnp.any(supports), value, jnp.nan)


class BrokenField(StrictModule):
    """Side-aware piecewise field over a non-overlapping cover."""

    family: LocalFieldFamily

    def __init__(self, family: LocalFieldFamily, /):
        if not isinstance(family, LocalFieldFamily):
            raise TypeError("family must be a LocalFieldFamily.")
        self.family = family

    def local(self, patch_id: str, /) -> DomainFunction:
        return self.family.field(patch_id)

    def trace(
        self,
        pairing_id: str,
        /,
        *,
        side: Literal["left", "right"],
    ) -> DomainFunction:
        pairing = self.family.cover.pairing(pairing_id)
        patch_id = pairing.left_patch_id if side == "left" else pairing.right_patch_id
        return pairing.trace(self.family.field(patch_id), side=side)

    def as_domain_function(
        self,
        *,
        ownership: Literal["first"],
    ) -> DomainFunction:
        if ownership != "first":
            raise ValueError("ownership must be the explicit policy 'first'.")
        lifted = self.family.lifted_fields()
        supports = tuple(patch.support for patch in self.family.cover.patches)
        deps = _dependency_union((*lifted, *supports))
        return DomainFunction(
            domain=self.family.cover.ambient,
            deps=deps,
            func=_BrokenFieldEvaluator(self.family, deps),
            metadata={
                "field_id": self.family.field_id,
                "cover_id": self.family.cover.cover_id,
                "assembly": "broken-first-owner",
            },
        )


def partition_of_unity_field(family: LocalFieldFamily, /) -> DomainFunction:
    """Assemble one smooth ambient field from fixed local windows."""
    if not isinstance(family, LocalFieldFamily):
        raise TypeError("family must be a LocalFieldFamily.")
    lifted = family.lifted_fields()
    supports = tuple(patch.support for patch in family.cover.patches)
    windows = tuple(
        patch.window for patch in family.cover.patches if patch.window is not None
    )
    deps = _dependency_union((*lifted, *supports, *windows))
    regularities = tuple(
        int(patch.window.metadata.get("regularity_order", 0))
        for patch in family.cover.patches
        if patch.window is not None
    )
    return DomainFunction(
        domain=family.cover.ambient,
        deps=deps,
        func=_PartitionOfUnityEvaluator(family, deps),
        metadata={
            "field_id": family.field_id,
            "cover_id": family.cover.cover_id,
            "assembly": "partition-of-unity",
            "window_regularity_order": min(regularities) if regularities else 0,
        },
    )


def partition_of_unity_family(field: DomainFunction, /) -> LocalFieldFamily:
    """Return the canonical trained local family owned by an assembled field."""
    if not isinstance(field, DomainFunction) or not isinstance(
        field.func, _PartitionOfUnityEvaluator
    ):
        raise TypeError("field is not a native partition-of-unity field.")
    from ...operators._composition import _PullbackCallable

    local_fields: dict[str, DomainFunction] = {}
    for patch, lifted in zip(
        field.func.cover.patches,
        field.func.fields,
        strict=True,
    ):
        if not isinstance(lifted.func, _PullbackCallable):
            raise TypeError("Native assembled field contains an invalid local lift.")
        local_fields[patch.patch_id] = lifted.func.source
    return LocalFieldFamily(field.func.field_id, field.func.cover, local_fields)


def broken_field(family: LocalFieldFamily, /) -> BrokenField:
    """Return an explicitly side-aware broken representation."""
    return BrokenField(family)


__all__ = [
    "BrokenField",
    "LocalFieldFamily",
    "LocalFieldRef",
    "broken_field",
    "partition_of_unity_field",
    "partition_of_unity_family",
]
