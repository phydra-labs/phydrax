#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Literal, TYPE_CHECKING

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, Key

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._strict import StrictModule
from ..domain._structure import PointsBatch


if TYPE_CHECKING:
    from ..domain._function import DomainFunction
    from ._functional import FunctionalConstraint


CollocationAlgorithm = Literal["periodic", "r3", "rar_d"]

class AbstractCollocationPolicy(StrictModule):
    """Lifecycle shared by solver-managed adaptive collocation policies."""

    refresh_every: int

    @abstractmethod
    def initialize(
        self,
        constraint: FunctionalConstraint,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def should_refresh(self, population: Any, iter_: int | Array) -> Array:
        """Return whether a host-managed refresh is due."""
        raise NotImplementedError

    @abstractmethod
    def data_metrics(self, population: Any, /) -> dict[str, Array]:
        """Return scalar diagnostics for solver logging."""
        raise NotImplementedError


    @abstractmethod
    def refresh(
        self,
        constraint: FunctionalConstraint,
        functions: Mapping[str, DomainFunction],
        population: Any,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def loss_batch_and_weight(
        self,
        population: Any,
        /,
    ) -> tuple[Any, cx.Field | None]:
        """Return the explicit loss batch and any point-local multiplier."""
        raise NotImplementedError

    @abstractmethod
    def refresh_residual_evaluations(self, population: Any, /) -> int:
        """Return the logical residual evaluations performed by one refresh."""
        raise NotImplementedError







class CollocationPopulation(StrictModule):
    """Persistent fixed-shape state for one sampled functional constraint."""

    batch: PointsBatch
    active: cx.Field | None
    age: cx.Field
    refresh_count: Array
    last_refresh: Array

    def __init__(
        self,
        batch: PointsBatch,
        *,
        active: cx.Field | None = None,
        age: cx.Field | None = None,
        refresh_count: int | Array = 0,
        last_refresh: int | Array = 0,
    ):
        axis, n = _single_axis_and_size(batch)
        if active is not None:
            _validate_axis_field(active, axis=axis, size=n, name="active")
        if age is None:
            age = cx.Field(jnp.zeros((n,), dtype=jnp.int32), dims=(axis,))
        else:
            _validate_axis_field(age, axis=axis, size=n, name="age")
        self.batch = batch
        self.active = active
        self.age = age
        self.refresh_count = jnp.asarray(refresh_count, dtype=jnp.int32)
        self.last_refresh = jnp.asarray(last_refresh, dtype=jnp.int32)

    def loss_weight(self) -> cx.Field | None:
        return self.active


def _collocation_population_metrics(
    population: CollocationPopulation,
    /,
) -> dict[str, Array]:
    _, size = _single_axis_and_size(population.batch)
    active_count = (
        jnp.asarray(size, dtype=float)
        if population.active is None
        else jnp.sum(jnp.asarray(population.active.data, dtype=float))
    )
    return {
        "refresh_count": jnp.asarray(population.refresh_count, dtype=float),
        "last_refresh": jnp.asarray(population.last_refresh, dtype=float),
        "point_count": jnp.asarray(size, dtype=float),
        "active_point_count": active_count,
        "effective_sample_size": active_count,
        "mean_age": jnp.mean(jnp.asarray(population.age.data, dtype=float)),
    }


class CollocationPolicy(AbstractCollocationPolicy):
    """Configuration for the retained paired-point collocation policies."""

    algorithm: CollocationAlgorithm
    refresh_every: int
    sampler: str
    candidate_multiplier: int
    exponent: Array
    uniform_floor: Array
    min_replace_fraction: Array
    max_retain_fraction: Array
    initial_active_fraction: Array
    refinement_fraction: Array
    epsilon: Array

    def __init__(
        self,
        algorithm: CollocationAlgorithm = "periodic",
        *,
        refresh_every: int = 100,
        sampler: str = "sobol_scrambled",
        candidate_multiplier: int = 10,
        exponent: float = 1.0,
        uniform_floor: float = 1.0,
        min_replace_fraction: float = 0.0,
        max_retain_fraction: float = 1.0,
        initial_active_fraction: float = 0.5,
        refinement_fraction: float = 0.05,
        epsilon: float = 1e-12,
    ):
        if algorithm not in ("periodic", "r3", "rar_d"):
            raise ValueError(f"Unsupported collocation algorithm {algorithm!r}.")
        if int(refresh_every) <= 0:
            raise ValueError("refresh_every must be positive.")
        if int(candidate_multiplier) <= 0:
            raise ValueError("candidate_multiplier must be positive.")
        if float(exponent) < 0.0 or float(uniform_floor) < 0.0:
            raise ValueError("exponent and uniform_floor must be non-negative.")
        if not 0.0 <= float(min_replace_fraction) <= 1.0:
            raise ValueError("min_replace_fraction must lie in [0, 1].")
        if not 0.0 <= float(max_retain_fraction) <= 1.0:
            raise ValueError("max_retain_fraction must lie in [0, 1].")
        if not 0.0 < float(initial_active_fraction) <= 1.0:
            raise ValueError("initial_active_fraction must lie in (0, 1].")
        if not 0.0 < float(refinement_fraction) <= 1.0:
            raise ValueError("refinement_fraction must lie in (0, 1].")
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive.")
        self.algorithm = algorithm
        self.refresh_every = int(refresh_every)
        self.sampler = str(sampler)
        self.candidate_multiplier = int(candidate_multiplier)
        self.exponent = jnp.asarray(exponent, dtype=float)
        self.uniform_floor = jnp.asarray(uniform_floor, dtype=float)
        self.min_replace_fraction = jnp.asarray(min_replace_fraction, dtype=float)
        self.max_retain_fraction = jnp.asarray(max_retain_fraction, dtype=float)
        self.initial_active_fraction = jnp.asarray(initial_active_fraction, dtype=float)
        self.refinement_fraction = jnp.asarray(refinement_fraction, dtype=float)
        self.epsilon = jnp.asarray(epsilon, dtype=float)

    def should_refresh(
        self,
        population: CollocationPopulation,
        iter_: int | Array,
    ) -> Array:
        step = jnp.asarray(iter_, dtype=jnp.int32)
        return (step - population.last_refresh) >= self.refresh_every

    def initialize(
        self,
        constraint: FunctionalConstraint,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> CollocationPopulation:
        batch = constraint._sample_once(key=key)
        if not isinstance(batch, PointsBatch):
            raise TypeError("Adaptive collocation currently requires a PointsBatch.")
        axis, n = _single_axis_and_size(batch)
        active = None
        if self.algorithm == "rar_d":
            n_active = max(1, int(round(n * float(self.initial_active_fraction))))
            active = cx.Field(
                (jnp.arange(n) < n_active).astype(float),
                dims=(axis,),
            )
        return CollocationPopulation(batch, active=active)

    def loss_batch_and_weight(
        self,
        population: CollocationPopulation,
        /,
    ) -> tuple[PointsBatch, cx.Field | None]:
        return population.batch, population.loss_weight()

    def data_metrics(
        self,
        population: CollocationPopulation,
        /,
    ) -> dict[str, Array]:
        return _collocation_population_metrics(population)

    def refresh_residual_evaluations(
        self,
        population: CollocationPopulation,
        /,
    ) -> int:
        _, size = _single_axis_and_size(population.batch)
        if self.algorithm == "periodic":
            return 0
        if self.algorithm == "r3":
            return size
        if population.active is None:
            return size * self.candidate_multiplier
        active_count = int(jnp.sum(jnp.asarray(population.active.data) > 0))
        return 0 if active_count == size else size * self.candidate_multiplier

    def refresh(
        self,
        constraint: FunctionalConstraint,
        functions: Mapping[str, DomainFunction],
        population: CollocationPopulation,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_: int | Array,
    ) -> CollocationPopulation:
        step = jnp.asarray(iter_, dtype=jnp.int32)
        if self.algorithm == "periodic":
            batch = constraint._sample_once(key=key)
            if not isinstance(batch, PointsBatch):
                raise TypeError("Adaptive collocation currently requires a PointsBatch.")
            out = _replace_all(population, batch)
        elif self.algorithm == "r3":
            out = self._refresh_r3(constraint, functions, population, key=key)
        else:
            out = self._refresh_rar_d(constraint, functions, population, key=key)
        return CollocationPopulation(
            out.batch,
            active=out.active,
            age=out.age,
            refresh_count=population.refresh_count + 1,
            last_refresh=step,
        )

    def _scores(
        self,
        constraint: FunctionalConstraint,
        functions: Mapping[str, DomainFunction],
        batch: PointsBatch,
        /,
        *,
        key: Key[Array, ""],
    ) -> cx.Field:
        score = constraint.pointwise_loss(functions, batch=batch, key=key)
        axis, n = _single_axis_and_size(batch)
        _validate_axis_field(score, axis=axis, size=n, name="pointwise loss")
        data = jax.lax.stop_gradient(jnp.asarray(score.data, dtype=float))
        data = jnp.nan_to_num(
            data,
            nan=0.0,
            posinf=jnp.finfo(data.dtype).max,
            neginf=0.0,
        )
        return cx.Field(jnp.maximum(data, 0.0), dims=score.dims)

    def _refresh_r3(self, constraint, functions, population, *, key):
        axis, n = _single_axis_and_size(population.batch)
        scores = self._scores(
            constraint,
            functions,
            population.batch,
            key=jr.fold_in(key, 1),
        )
        score_data = jnp.asarray(scores.data, dtype=float)
        retain = score_data > jnp.mean(score_data)
        max_retain = int(round(n * float(self.max_retain_fraction)))
        min_replace = int(round(n * float(self.min_replace_fraction)))
        max_retain = min(max_retain, n - min_replace)
        order = jnp.argsort(score_data)[::-1]
        eligible = order[retain[order]]
        keep_n = min(int(eligible.shape[0]), max_retain)
        keep_idx = eligible[:keep_n]
        new_n = n - keep_n
        replacement = constraint.component.sample(
            new_n,
            structure=constraint.structure,
            sampler=self.sampler,
            key=jr.fold_in(key, 2),
        )
        if not isinstance(replacement, PointsBatch):
            raise TypeError("R3 requires a PointsBatch replacement population.")
        batch = _concat_batches(
            _take_batch(population.batch, keep_idx),
            replacement,
        )
        age = cx.Field(
            jnp.concatenate(
                [
                    population.age.data[keep_idx] + 1,
                    jnp.zeros((new_n,), dtype=jnp.int32),
                ]
            ),
            dims=(axis,),
        )
        return CollocationPopulation(batch, age=age)

    def _refresh_rar_d(self, constraint, functions, population, *, key):
        if population.active is None:
            raise ValueError("RAR-D population requires an active mask.")
        axis, n = _single_axis_and_size(population.batch)
        active_n = int(jnp.sum(population.active.data > 0))
        add_n = min(
            max(1, int(round(n * float(self.refinement_fraction)))),
            n - active_n,
        )
        if add_n == 0:
            return population
        candidate_n = n * self.candidate_multiplier
        candidate = constraint.component.sample(
            candidate_n,
            structure=constraint.structure,
            sampler=self.sampler,
            key=jr.fold_in(key, 1),
        )
        if not isinstance(candidate, PointsBatch):
            raise TypeError("RAR-D requires PointsBatch candidates.")
        scores = self._scores(
            constraint,
            functions,
            candidate,
            key=jr.fold_in(key, 2),
        )
        score_data = jnp.asarray(scores.data, dtype=float)
        powered = jnp.power(score_data + self.epsilon, self.exponent)
        density = (
            powered / jnp.maximum(jnp.mean(powered), self.epsilon)
            + self.uniform_floor
        )
        probabilities = density / jnp.sum(density)
        idx = jr.choice(
            jr.fold_in(key, 3),
            candidate_n,
            shape=(add_n,),
            replace=True,
            p=probabilities,
        )
        additions = _take_batch(candidate, idx)
        target_idx = jnp.arange(active_n, active_n + add_n)
        batch = _set_batch_rows(population.batch, target_idx, additions)
        active = population.active.data.at[target_idx].set(1.0)
        age = cx.Field(
            population.age.data.at[target_idx].set(
                jnp.asarray(0, dtype=jnp.int32)
            ),
            dims=(axis,),
        )
        return CollocationPopulation(
            batch,
            active=cx.Field(active, dims=(axis,)),
            age=age,
        )


def PeriodicCollocation(**kwargs: Any) -> CollocationPolicy:
    return CollocationPolicy("periodic", **kwargs)




def R3(**kwargs: Any) -> CollocationPolicy:
    return CollocationPolicy("r3", **kwargs)






def RARD(**kwargs: Any) -> CollocationPolicy:
    return CollocationPolicy("rar_d", **kwargs)




def with_collocation_policy(
    constraint: FunctionalConstraint,
    policy: AbstractCollocationPolicy,
    /,
) -> FunctionalConstraint:
    """Attach a collocation policy to any existing functional constraint."""
    return eqx.tree_at(lambda c: c.collocation_policy, constraint, policy)


def _single_axis_and_size(batch: PointsBatch) -> tuple[str, int]:
    axes = batch.structure.axis_names
    if axes is None or len(axes) != 1:
        raise ValueError("Adaptive collocation currently requires exactly one sampling axis.")
    axis = axes[0]
    size: int | None = None
    for leaf in jtu.tree_leaves(batch.points, is_leaf=lambda x: isinstance(x, cx.Field)):
        if isinstance(leaf, cx.Field) and axis in leaf.named_shape:
            candidate = int(leaf.named_shape[axis])
            if size is None:
                size = candidate
            elif size != candidate:
                raise ValueError("Batch fields disagree on adaptive sampling-axis size.")
    if size is None:
        raise ValueError("Could not infer adaptive sampling-axis size.")
    return axis, size


def _validate_axis_field(field: cx.Field, *, axis: str, size: int, name: str) -> None:
    if field.dims != (axis,):
        raise ValueError(f"{name} must have dims ({axis!r},), got {field.dims}.")
    if field.data.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {field.data.shape}.")


def _map_batch_fields(batch: PointsBatch, fn) -> PointsBatch:
    points = jtu.tree_map(fn, batch.points, is_leaf=lambda x: isinstance(x, cx.Field))
    metadata = jtu.tree_map(
        fn,
        batch.metadata,
        is_leaf=lambda x: isinstance(x, cx.Field),
    )
    return PointsBatch(frozendict(points), batch.structure, metadata=frozendict(metadata))


def _take_batch(batch: PointsBatch, indices: Array) -> PointsBatch:
    axis, _ = _single_axis_and_size(batch)

    def take(field):
        if not isinstance(field, cx.Field) or axis not in field.named_dims:
            return field
        pos = field.dims.index(axis)
        return cx.Field(jnp.take(field.data, indices, axis=pos), dims=field.dims)

    return _map_batch_fields(batch, take)


def _concat_batches(left: PointsBatch, right: PointsBatch) -> PointsBatch:
    if left.structure != right.structure:
        raise ValueError("Cannot concatenate batches with different structures.")
    axis, _ = _single_axis_and_size(right)

    def concat(a, b):
        if not isinstance(a, cx.Field) or not isinstance(b, cx.Field):
            if a != b:
                raise ValueError("Fixed batch leaves differ during concatenation.")
            return a
        if axis not in a.named_dims:
            return a
        pos = a.dims.index(axis)
        return cx.Field(jnp.concatenate([a.data, b.data], axis=pos), dims=a.dims)

    points = jtu.tree_map(
        concat,
        left.points,
        right.points,
        is_leaf=lambda x: isinstance(x, cx.Field),
    )
    if left.metadata.keys() != right.metadata.keys():
        raise ValueError("Cannot concatenate batches with different metadata fields.")
    metadata = jtu.tree_map(
        concat,
        left.metadata,
        right.metadata,
        is_leaf=lambda x: isinstance(x, cx.Field),
    )
    return PointsBatch(
        frozendict(points),
        left.structure,
        metadata=frozendict(metadata),
    )


def _set_batch_rows(target: PointsBatch, indices: Array, source: PointsBatch) -> PointsBatch:
    axis, _ = _single_axis_and_size(target)

    def set_rows(a, b):
        if not isinstance(a, cx.Field) or not isinstance(b, cx.Field):
            return a
        if axis not in a.named_dims:
            return a
        pos = a.dims.index(axis)
        data = jnp.moveaxis(a.data, pos, 0)
        values = jnp.moveaxis(b.data, pos, 0)
        data = data.at[indices].set(values)
        return cx.Field(jnp.moveaxis(data, 0, pos), dims=a.dims)

    points = jtu.tree_map(
        set_rows,
        target.points,
        source.points,
        is_leaf=lambda x: isinstance(x, cx.Field),
    )
    if target.metadata.keys() != source.metadata.keys():
        raise ValueError("Cannot replace rows from a batch with different metadata fields.")
    metadata = jtu.tree_map(
        set_rows,
        target.metadata,
        source.metadata,
        is_leaf=lambda x: isinstance(x, cx.Field),
    )
    return PointsBatch(
        frozendict(points),
        target.structure,
        metadata=frozendict(metadata),
    )


def _replace_all(
    population: CollocationPopulation,
    batch: PointsBatch,
) -> CollocationPopulation:
    return CollocationPopulation(batch)














__all__ = [
    "AbstractCollocationPolicy",
    "CollocationPolicy",
    "CollocationPopulation",
    "PeriodicCollocation",
    "R3",
    "RARD",
    "with_collocation_policy",
]
