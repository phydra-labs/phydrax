#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._frozendict import frozendict
from .._strict import StrictModule
from ..discretization._state_transfer import AbstractRefinementTransfer
from ..stochastic._hierarchy import StochasticCouplingPlan, StochasticLevelSpec
from ..stochastic._realization import is_stochastic_realization, StochasticRealization


CoupledLevelSolver: TypeAlias = Callable[
    [
        StochasticLevelSpec,
        StochasticRealization | None,
        Any | None,
        AbstractRefinementTransfer | None,
    ],
    Any,
]
CoupledObservable: TypeAlias = Callable[[Any, StochasticLevelSpec], ArrayLike]
CoupledValidity: TypeAlias = Callable[[Any, StochasticLevelSpec], ArrayLike]
CoupledCost: TypeAlias = Callable[[Any, StochasticLevelSpec], ArrayLike]


def _nonnegative_scalar(value: ArrayLike, name: str, /) -> Array:
    resolved = jnp.asarray(value, dtype=float)
    if resolved.shape != () or bool(~jnp.isfinite(resolved)) or bool(resolved < 0.0):
        raise ValueError(f"{name} must be a finite non-negative scalar.")
    return resolved


def _default_validity(observable: Array, sample_shape: tuple[int, ...], /) -> Array:
    if observable.shape[: len(sample_shape)] != sample_shape:
        raise ValueError(
            "Every level observable must begin with the realization sample_shape."
        )
    finite = jnp.isfinite(observable)
    trailing_axes = tuple(range(len(sample_shape), observable.ndim))
    return jnp.all(finite, axis=trailing_axes) if trailing_axes else finite


def _validate_validity(value: ArrayLike, sample_shape: tuple[int, ...], /) -> Array:
    valid = jnp.asarray(value, dtype=bool)
    if valid.shape != sample_shape:
        raise ValueError(
            f"Level validity must have realization sample shape {sample_shape}; "
            f"got {valid.shape}."
        )
    return valid


def _block_arrays(value: Any, /) -> None:
    for leaf in jax.tree.leaves(value):
        if eqx.is_array(leaf):
            jax.block_until_ready(leaf)


class CoupledLevelResult(StrictModule):
    """One hierarchy solve with observable, validity, cost, and provenance."""

    result: Any
    observable: Array
    valid: Array
    cost_seconds: Array
    level: StochasticLevelSpec = eqx.field(static=True)
    realization_id: str | None = eqx.field(static=True)
    coupling_id: str | None = eqx.field(static=True)
    state_transfer_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        result: Any,
        observable: ArrayLike,
        valid: ArrayLike,
        cost_seconds: ArrayLike,
        /,
        *,
        level: StochasticLevelSpec,
        sample_shape: Sequence[int],
        realization_id: str | None,
        coupling_id: str | None,
        state_transfer_id: str | None,
    ):
        samples = tuple(int(size) for size in sample_shape)
        values = jnp.asarray(observable)
        if values.shape[: len(samples)] != samples:
            raise ValueError("observable does not begin with sample_shape.")
        self.result = result
        self.observable = values
        self.valid = _validate_validity(valid, samples)
        self.cost_seconds = _nonnegative_scalar(cost_seconds, "cost_seconds")
        self.level = level
        self.realization_id = realization_id
        self.coupling_id = coupling_id
        self.state_transfer_id = state_transfer_id

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid)


class CoupledHierarchyResult(StrictModule):
    """Pathwise level evaluations and telescoping corrections for one hierarchy."""

    levels: tuple[CoupledLevelResult, ...]
    corrections: tuple[Array, ...]
    correction_valid: tuple[Array, ...]
    hierarchy: StochasticCouplingPlan = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    realization_id: str | None = eqx.field(static=True)
    coupling_id: str | None = eqx.field(static=True)
    metadata: frozendict[str, Any] = eqx.field(static=True)

    def __init__(
        self,
        levels: Sequence[CoupledLevelResult],
        corrections: Sequence[ArrayLike],
        correction_valid: Sequence[ArrayLike],
        /,
        *,
        hierarchy: StochasticCouplingPlan,
        sample_shape: Sequence[int],
        realization_id: str | None,
        coupling_id: str | None,
        metadata: Mapping[str, Any] | None = None,
    ):
        records = tuple(levels)
        values = tuple(jnp.asarray(correction) for correction in corrections)
        valid = tuple(jnp.asarray(mask, dtype=bool) for mask in correction_valid)
        samples = tuple(int(size) for size in sample_shape)
        if len(records) != hierarchy.num_levels:
            raise ValueError(
                "Coupled records must align one-to-one with hierarchy levels."
            )
        if len(values) != len(records) or len(valid) != len(records):
            raise ValueError("Corrections and validity masks must align with levels.")
        expected_shape = records[0].observable.shape
        if any(correction.shape != expected_shape for correction in values):
            raise ValueError("Every pathwise correction must share one observable shape.")
        if any(mask.shape != samples for mask in valid):
            raise ValueError("Correction validity masks must equal sample_shape.")
        self.levels = records
        self.corrections = values
        self.correction_valid = valid
        self.hierarchy = hierarchy
        self.sample_shape = samples
        self.realization_id = realization_id
        self.coupling_id = coupling_id
        self.metadata = frozendict({} if metadata is None else metadata)

    @property
    def finest_observable(self) -> Array:
        return self.levels[-1].observable

    @property
    def total_cost_seconds(self) -> Array:
        return sum((level.cost_seconds for level in self.levels), start=jnp.asarray(0.0))

    @property
    def successful(self) -> Array:
        return jnp.all(jnp.stack(self.correction_valid, axis=0), axis=0)

    def correction_means(self) -> tuple[Array, ...]:
        """Masked sample means for each pathwise telescoping correction."""
        axes = tuple(range(len(self.sample_shape)))
        if not axes:
            return self.corrections
        means: list[Array] = []
        trailing = self.corrections[0].ndim - len(self.sample_shape)
        for correction, valid in zip(
            self.corrections, self.correction_valid, strict=True
        ):
            expanded = valid.reshape(valid.shape + (1,) * trailing)
            count = jnp.sum(valid)
            numerator = jnp.sum(jnp.where(expanded, correction, 0.0), axis=axes)
            means.append(jnp.where(count > 0, numerator / count, jnp.nan))
        return tuple(means)

    def telescoping_mean(self) -> Array:
        """Sum the independently masked means of all level corrections."""
        return sum(self.correction_means()[1:], start=self.correction_means()[0])


def solve_coupled_hierarchy(
    hierarchy: StochasticCouplingPlan,
    realization: StochasticRealization | None,
    solve_level: CoupledLevelSolver,
    observable: CoupledObservable,
    /,
    *,
    validity: CoupledValidity | None = None,
    cost: CoupledCost | None = None,
    state_transfers: Mapping[str, AbstractRefinementTransfer] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CoupledHierarchyResult:
    """Solve every hierarchy level against one explicitly shared realization.

    ``solve_level`` receives ``(level, realization, parent_result, transfer)``. The
    transfer maps the current fine layout to/from its immediate parent layout. All
    levels use the same realization object, so corrections cannot accidentally pair
    unrelated paths. Independent level noise is rejected because it does not define a
    strong pathwise correction.
    """
    if not isinstance(hierarchy, StochasticCouplingPlan):
        raise TypeError("hierarchy must be a StochasticCouplingPlan.")
    if realization is not None and not is_stochastic_realization(realization):
        raise TypeError("realization must be a supported stochastic realization or None.")
    if not callable(solve_level) or not callable(observable):
        raise TypeError("solve_level and observable must be callable.")
    if validity is not None and not callable(validity):
        raise TypeError("validity must be callable or None.")
    if cost is not None and not callable(cost):
        raise TypeError("cost must be callable or None.")
    if any(level.noise_coupling == "independent" for level in hierarchy.levels[1:]):
        raise ValueError(
            "Pathwise coupled execution does not permit independent fine-level noise."
        )
    transfers = {} if state_transfers is None else dict(state_transfers)
    if any(
        not isinstance(value, AbstractRefinementTransfer) for value in transfers.values()
    ):
        raise TypeError(
            "state_transfers values must implement AbstractRefinementTransfer."
        )
    samples = () if realization is None else realization.sample_shape
    realization_id = None if realization is None else realization.realization_id
    coupling_id = None if realization is None else realization.coupling_id

    records: list[CoupledLevelResult] = []
    corrections: list[Array] = []
    correction_masks: list[Array] = []
    parent_result: Any | None = None
    for position, level in enumerate(hierarchy.levels):
        transfer: AbstractRefinementTransfer | None = None
        if level.state_transfer_id is not None:
            if level.state_transfer_id not in transfers:
                raise ValueError(
                    f"Hierarchy level {level.level_id!r} requires state transfer "
                    f"{level.state_transfer_id!r}."
                )
            transfer = transfers[level.state_transfer_id]
            parent = hierarchy.levels[position - 1]
            if transfer.coarse_shape != parent.state_shape:
                raise ValueError(
                    "State transfer coarse_shape does not match parent level."
                )
            if transfer.fine_shape != level.state_shape:
                raise ValueError(
                    "State transfer fine_shape does not match current level."
                )
            if transfer.transfer_id != level.state_transfer_id:
                raise ValueError(
                    "State transfer object ID does not match level provenance."
                )
        started = perf_counter()
        output = solve_level(level, realization, parent_result, transfer)
        _block_arrays(output)
        measured_cost = perf_counter() - started
        level_observable = jnp.asarray(observable(output, level))
        level_valid = (
            _default_validity(level_observable, samples)
            if validity is None
            else _validate_validity(validity(output, level), samples)
        )
        level_cost = measured_cost if cost is None else cost(output, level)
        record = CoupledLevelResult(
            output,
            level_observable,
            level_valid,
            level_cost,
            level=level,
            sample_shape=samples,
            realization_id=realization_id,
            coupling_id=coupling_id,
            state_transfer_id=None if transfer is None else transfer.transfer_id,
        )
        records.append(record)
        if position == 0:
            corrections.append(level_observable)
            correction_masks.append(level_valid)
        else:
            previous = records[position - 1]
            if level_observable.shape != previous.observable.shape:
                raise ValueError(
                    "Coupled observables must have identical shapes after state transfer."
                )
            corrections.append(level_observable - previous.observable)
            correction_masks.append(level_valid & previous.valid)
        parent_result = output
    return CoupledHierarchyResult(
        records,
        corrections,
        correction_masks,
        hierarchy=hierarchy,
        sample_shape=samples,
        realization_id=realization_id,
        coupling_id=coupling_id,
        metadata=metadata,
    )


__all__ = [
    "CoupledCost",
    "CoupledHierarchyResult",
    "CoupledLevelResult",
    "CoupledLevelSolver",
    "CoupledObservable",
    "CoupledValidity",
    "solve_coupled_hierarchy",
]
