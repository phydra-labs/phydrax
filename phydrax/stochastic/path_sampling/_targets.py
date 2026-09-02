#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Path ensembles, normalized path actions, and reweighting boundaries."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import isfinite
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._sampling._targets import IncrementalMarkovTarget
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ...uq._free_energy import ReducedPotentialSamples
from ._core import (
    FunctionalDynamicsKernel,
    path_trajectory_id,
    PathBuffer,
    StateRegionPlan,
)


def _nonempty(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _active_event_mask(path: PathBuffer, /) -> Array:
    return path.mask.reshape((path.capacity,) + (1,) * len(path.event_shape))


def _finite_active(path: PathBuffer, /) -> Array:
    return jnp.all(
        jnp.where(_active_event_mask(path), jnp.isfinite(path.positions), True)
    )


def _last_state(path: PathBuffer, /) -> Array:
    return path.positions[jnp.clip(path.length - 1, 0, path.capacity - 1)]


class AbstractPathEnsemble(StrictModule):
    """Abstract normalized-support contract for a fixed-capacity path target."""

    ensemble_id: AbstractAttribute[str]
    requires_terminal_hit: AbstractAttribute[bool]

    @abc.abstractmethod
    def contains(self, path: PathBuffer, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def terminal(self, state: Array, direction: Array, /) -> Array:
        raise NotImplementedError


class FixedPathEnsemble(AbstractPathEnsemble, NonTrainableState):
    """Exact-length path ensemble with optional endpoint regions."""

    initial_region: StateRegionPlan | None
    final_region: StateRegionPlan | None
    path_length: int = eqx.field(static=True)
    ensemble_id: str = eqx.field(static=True)
    requires_terminal_hit: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        path_length: int,
        /,
        *,
        initial_region: StateRegionPlan | None = None,
        final_region: StateRegionPlan | None = None,
        ensemble_id: str | None = None,
    ):
        count = int(path_length)
        if count <= 1:
            raise ValueError("path_length must exceed one.")
        if initial_region is not None and not isinstance(initial_region, StateRegionPlan):
            raise TypeError("initial_region must be StateRegionPlan or None.")
        if final_region is not None and not isinstance(final_region, StateRegionPlan):
            raise TypeError("final_region must be StateRegionPlan or None.")
        identity = ensemble_id or canonical_fingerprint(
            {
                "kind": "fixed-path-ensemble-v1",
                "length": count,
                "initial": None if initial_region is None else initial_region.region_id,
                "final": None if final_region is None else final_region.region_id,
            }
        )
        self.initial_region = initial_region
        self.final_region = final_region
        self.path_length = count
        self.ensemble_id = _nonempty(identity, "ensemble_id")

    def contains(self, path: PathBuffer, /) -> Array:
        if not isinstance(path, PathBuffer):
            raise TypeError("path must be a PathBuffer.")
        valid = path.valid() & (path.length == self.path_length) & _finite_active(path)
        if self.initial_region is not None:
            valid = valid & self.initial_region.contains(path.positions[0])
        if self.final_region is not None:
            valid = valid & self.final_region.contains(_last_state(path))
        return valid

    def terminal(self, state: Array, direction: Array, /) -> Array:
        del state, direction
        return jnp.asarray(False)


class FirstPassagePathEnsemble(AbstractPathEnsemble, NonTrainableState):
    """A-to-B first-passage paths with no earlier visit to B."""

    initial_region: StateRegionPlan
    final_region: StateRegionPlan
    minimum_length: int = eqx.field(static=True)
    ensemble_id: str = eqx.field(static=True)
    requires_terminal_hit: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        initial_region: StateRegionPlan,
        final_region: StateRegionPlan,
        /,
        *,
        minimum_length: int = 2,
        ensemble_id: str | None = None,
    ):
        if not isinstance(initial_region, StateRegionPlan) or not isinstance(
            final_region, StateRegionPlan
        ):
            raise TypeError("First-passage endpoints must be StateRegionPlan values.")
        count = int(minimum_length)
        if count < 2:
            raise ValueError("minimum_length must be at least two.")
        identity = ensemble_id or canonical_fingerprint(
            {
                "kind": "first-passage-path-ensemble-v1",
                "initial": initial_region.region_id,
                "final": final_region.region_id,
                "minimum_length": count,
            }
        )
        self.initial_region = initial_region
        self.final_region = final_region
        self.minimum_length = count
        self.ensemble_id = _nonempty(identity, "ensemble_id")

    def contains(self, path: PathBuffer, /) -> Array:
        final_hits = self.final_region.contains(path.positions)
        indices = jnp.arange(path.capacity, dtype=jnp.int32)
        last = path.length - 1
        premature = jnp.any(final_hits & path.mask & (indices < last))
        return (
            path.valid()
            & (path.length >= self.minimum_length)
            & self.initial_region.contains(path.positions[0])
            & self.final_region.contains(_last_state(path))
            & ~premature
        )

    def terminal(self, state: Array, direction: Array, /) -> Array:
        return jnp.where(
            direction > 0,
            self.final_region.contains(state),
            self.initial_region.contains(state),
        )


class InterfacePathEnsemble(AbstractPathEnsemble, NonTrainableState):
    """Paths leaving A, crossing one interface, and terminating in A or B."""

    initial_region: StateRegionPlan
    final_region: StateRegionPlan
    coordinate: Callable[[Array], Array] = eqx.field(static=True)
    interface_value: float = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)
    ensemble_id: str = eqx.field(static=True)
    requires_terminal_hit: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        initial_region: StateRegionPlan,
        final_region: StateRegionPlan,
        coordinate: Callable[[Array], Array],
        interface_value: float,
        /,
        *,
        coordinate_id: str,
        ensemble_id: str | None = None,
    ):
        if not isinstance(initial_region, StateRegionPlan) or not isinstance(
            final_region, StateRegionPlan
        ):
            raise TypeError("Interface endpoints must be StateRegionPlan values.")
        if not callable(coordinate):
            raise TypeError("coordinate must be callable.")
        level = float(interface_value)
        if not isfinite(level):
            raise ValueError("interface_value must be finite.")
        coordinate_identity = _nonempty(coordinate_id, "coordinate_id")
        identity = ensemble_id or canonical_fingerprint(
            {
                "kind": "interface-path-ensemble-v1",
                "initial": initial_region.region_id,
                "final": final_region.region_id,
                "coordinate": coordinate_identity,
                "interface": level.hex(),
            }
        )
        self.initial_region = initial_region
        self.final_region = final_region
        self.coordinate = coordinate
        self.interface_value = level
        self.coordinate_id = coordinate_identity
        self.ensemble_id = _nonempty(identity, "ensemble_id")

    def contains(self, path: PathBuffer, /) -> Array:
        coordinate = jnp.asarray(self.coordinate(path.positions))
        if coordinate.shape != (path.capacity,):
            raise ValueError("Interface coordinate must return one value per path point.")
        coordinate_finite = jnp.all(jnp.where(path.mask, jnp.isfinite(coordinate), True))
        terminal = _last_state(path)
        later = path.mask & (jnp.arange(path.capacity, dtype=jnp.int32) > 0)
        crossed = (coordinate[0] < self.interface_value) & jnp.any(
            later & (coordinate >= self.interface_value)
        )
        initial_hits = self.initial_region.contains(path.positions)
        final_hits = self.final_region.contains(path.positions)
        indices = jnp.arange(path.capacity, dtype=jnp.int32)
        last = path.length - 1
        interior = path.mask & (indices > 0) & (indices < last)
        interior_terminal_hit = jnp.any(interior & (initial_hits | final_hits))
        endpoint = self.initial_region.contains(terminal) | self.final_region.contains(
            terminal
        )
        return (
            path.valid()
            & (path.length >= 2)
            & coordinate_finite
            & self.initial_region.contains(path.positions[0])
            & endpoint
            & crossed
            & ~interior_terminal_hit
        )

    def terminal(self, state: Array, direction: Array, /) -> Array:
        del direction
        return self.initial_region.contains(state) | self.final_region.contains(state)


class MinusPathEnsemble(AbstractPathEnsemble, NonTrainableState):
    """A-to-A excursion crossing the innermost interface."""

    initial_region: StateRegionPlan
    coordinate: Callable[[Array], Array] = eqx.field(static=True)
    interface_value: float = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)
    ensemble_id: str = eqx.field(static=True)
    requires_terminal_hit: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        initial_region: StateRegionPlan,
        coordinate: Callable[[Array], Array],
        interface_value: float,
        /,
        *,
        coordinate_id: str,
        ensemble_id: str | None = None,
    ):
        if not isinstance(initial_region, StateRegionPlan):
            raise TypeError("initial_region must be StateRegionPlan.")
        if not callable(coordinate):
            raise TypeError("coordinate must be callable.")
        level = float(interface_value)
        if not isfinite(level):
            raise ValueError("interface_value must be finite.")
        coordinate_identity = _nonempty(coordinate_id, "coordinate_id")
        identity = ensemble_id or canonical_fingerprint(
            {
                "kind": "minus-path-ensemble-v1",
                "initial": initial_region.region_id,
                "coordinate": coordinate_identity,
                "interface": level.hex(),
            }
        )
        self.initial_region = initial_region
        self.coordinate = coordinate
        self.interface_value = level
        self.coordinate_id = coordinate_identity
        self.ensemble_id = _nonempty(identity, "ensemble_id")

    def contains(self, path: PathBuffer, /) -> Array:
        coordinate = jnp.asarray(self.coordinate(path.positions))
        if coordinate.shape != (path.capacity,):
            raise ValueError("Minus coordinate must return one value per path point.")
        coordinate_finite = jnp.all(jnp.where(path.mask, jnp.isfinite(coordinate), True))
        later = path.mask & (jnp.arange(path.capacity, dtype=jnp.int32) > 0)
        crossed = (coordinate[0] < self.interface_value) & jnp.any(
            later & (coordinate >= self.interface_value)
        )
        indices = jnp.arange(path.capacity, dtype=jnp.int32)
        last = path.length - 1
        interior = path.mask & (indices > 0) & (indices < last)
        interior_initial_hit = jnp.any(
            interior & self.initial_region.contains(path.positions)
        )
        return (
            path.valid()
            & (path.length >= 3)
            & coordinate_finite
            & self.initial_region.contains(path.positions[0])
            & self.initial_region.contains(_last_state(path))
            & crossed
            & ~interior_initial_hit
        )

    def terminal(self, state: Array, direction: Array, /) -> Array:
        del direction
        return self.initial_region.contains(state)


class InterfaceNetworkPlan(StrictModule, NonTrainableState):
    """Ordered interface network shared by TIS and RETIS prepared runtimes."""

    initial_region: StateRegionPlan
    final_region: StateRegionPlan
    coordinate: Callable[[Array], Array] = eqx.field(static=True)
    interfaces: tuple[float, ...] = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_region: StateRegionPlan,
        final_region: StateRegionPlan,
        coordinate: Callable[[Array], Array],
        interfaces: Sequence[float],
        /,
        *,
        coordinate_id: str,
        network_id: str | None = None,
    ):
        if not isinstance(initial_region, StateRegionPlan) or not isinstance(
            final_region, StateRegionPlan
        ):
            raise TypeError("Network endpoints must be StateRegionPlan values.")
        if not callable(coordinate):
            raise TypeError("coordinate must be callable.")
        levels = tuple(float(value) for value in interfaces)
        if (
            not levels
            or any(not isfinite(value) for value in levels)
            or any(
                right <= left for left, right in zip(levels[:-1], levels[1:], strict=True)
            )
        ):
            raise ValueError(
                "interfaces must be finite, non-empty, and strictly increasing."
            )
        coordinate_identity = _nonempty(coordinate_id, "coordinate_id")
        identity = network_id or canonical_fingerprint(
            {
                "kind": "path-interface-network-v1",
                "initial": initial_region.region_id,
                "final": final_region.region_id,
                "coordinate": coordinate_identity,
                "interfaces": [value.hex() for value in levels],
            }
        )
        self.initial_region = initial_region
        self.final_region = final_region
        self.coordinate = coordinate
        self.interfaces = levels
        self.coordinate_id = coordinate_identity
        self.network_id = _nonempty(identity, "network_id")

    @property
    def interface_count(self) -> int:
        return len(self.interfaces)

    def ensemble(self, index: int, /) -> InterfacePathEnsemble:
        selected = int(index)
        if selected < 0 or selected >= self.interface_count:
            raise IndexError("Interface index is outside the network.")
        return InterfacePathEnsemble(
            self.initial_region,
            self.final_region,
            self.coordinate,
            self.interfaces[selected],
            coordinate_id=self.coordinate_id,
        )

    def minus_ensemble(self, /) -> MinusPathEnsemble:
        return MinusPathEnsemble(
            self.initial_region,
            self.coordinate,
            self.interfaces[0],
            coordinate_id=self.coordinate_id,
        )


class AbstractPathAction(StrictModule):
    """Abstract path-space action with explicit normalization semantics."""

    action_id: AbstractAttribute[str]
    normalized: AbstractAttribute[bool]

    @abc.abstractmethod
    def log_weight(self, path: PathBuffer, /) -> Array:
        raise NotImplementedError


class DeterministicPathAction(AbstractPathAction, NonTrainableState):
    """Singular deterministic path action, intentionally not reweightable."""

    kernel: FunctionalDynamicsKernel
    action_id: str = eqx.field(static=True)
    normalized: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        kernel: FunctionalDynamicsKernel,
        /,
        *,
        action_id: str | None = None,
    ):
        if not isinstance(kernel, FunctionalDynamicsKernel):
            raise TypeError("kernel must be FunctionalDynamicsKernel.")
        if kernel.capabilities.stochastic:
            raise ValueError("DeterministicPathAction requires deterministic dynamics.")
        identity = action_id or canonical_fingerprint(
            {
                "kind": "deterministic-path-action-v1",
                "kernel": kernel.kernel_id,
            }
        )
        self.kernel = kernel
        self.action_id = _nonempty(identity, "action_id")

    def log_weight(self, path: PathBuffer, /) -> Array:
        source = path.positions[:-1]
        destination = path.positions[1:]
        values = jax.vmap(self.kernel.transition_log_density, in_axes=(0, 0, None))(
            source, destination, path.direction
        )
        if values.shape != (path.capacity - 1,) or jnp.iscomplexobj(values):
            raise ValueError(
                "Transition log density must return one real scalar per transition."
            )
        transition_mask = path.mask[1:]
        valid = path.valid() & jnp.all(
            jnp.where(transition_mask, jnp.isfinite(values), True)
        )
        return jnp.where(
            valid, jnp.sum(jnp.where(transition_mask, values, 0.0)), -jnp.inf
        )


class NormalizedStochasticPathAction(AbstractPathAction, NonTrainableState):
    """Normalized Markov path action from declared transition densities."""

    kernel: FunctionalDynamicsKernel
    initial_log_density: Callable[[Array], Array] = eqx.field(static=True)
    initial_density_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    normalized: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        kernel: FunctionalDynamicsKernel,
        initial_log_density: Callable[[Array], Array],
        /,
        *,
        initial_density_id: str,
        action_id: str | None = None,
    ):
        if not isinstance(kernel, FunctionalDynamicsKernel):
            raise TypeError("kernel must be FunctionalDynamicsKernel.")
        if (
            not kernel.capabilities.stochastic
            or not kernel.capabilities.normalized_transition_density
        ):
            raise ValueError(
                "NormalizedStochasticPathAction requires normalized stochastic dynamics."
            )
        if not callable(initial_log_density):
            raise TypeError("initial_log_density must be callable.")
        density_identity = _nonempty(initial_density_id, "initial_density_id")
        identity = action_id or canonical_fingerprint(
            {
                "kind": "normalized-stochastic-path-action-v1",
                "kernel": kernel.kernel_id,
                "initial_density": density_identity,
            }
        )
        self.kernel = kernel
        self.initial_log_density = initial_log_density
        self.initial_density_id = density_identity
        self.action_id = _nonempty(identity, "action_id")

    def log_weight(self, path: PathBuffer, /) -> Array:
        values = jax.vmap(self.kernel.transition_log_density, in_axes=(0, 0, None))(
            path.positions[:-1], path.positions[1:], path.direction
        )
        if values.shape != (path.capacity - 1,) or jnp.iscomplexobj(values):
            raise ValueError(
                "Transition log density must return one real scalar per transition."
            )
        transition_mask = path.mask[1:]
        initial = jnp.asarray(self.initial_log_density(path.positions[0]))
        if (
            initial.shape != ()
            or jnp.iscomplexobj(initial)
            or not jnp.issubdtype(initial.dtype, jnp.floating)
        ):
            raise ValueError(
                "Initial path log density must return one real floating scalar."
            )
        finite = jnp.isfinite(initial) & jnp.all(
            jnp.where(transition_mask, jnp.isfinite(values), True)
        )
        total = initial + jnp.sum(jnp.where(transition_mask, values, 0.0))
        return jnp.where(path.valid() & finite, total, -jnp.inf)


class SurrogatePathAction(AbstractPathAction, NonTrainableState):
    """Explicit unnormalized analysis action that cannot cross the FEP boundary."""

    evaluate: Callable[[PathBuffer], Array] = eqx.field(static=True)
    action_id: str = eqx.field(static=True)
    normalized: bool = eqx.field(static=True, default=False)

    def __init__(self, evaluate: Callable[[PathBuffer], Array], /, *, action_id: str):
        if not callable(evaluate):
            raise TypeError("evaluate must be callable.")
        self.evaluate = evaluate
        self.action_id = _nonempty(action_id, "action_id")

    def log_weight(self, path: PathBuffer, /) -> Array:
        return jnp.asarray(self.evaluate(path))


def path_log_target(
    ensemble: AbstractPathEnsemble,
    action: AbstractPathAction,
    path: PathBuffer,
    /,
) -> Array:
    if not isinstance(ensemble, AbstractPathEnsemble):
        raise TypeError("ensemble must implement AbstractPathEnsemble.")
    if not isinstance(action, AbstractPathAction):
        raise TypeError("action must implement AbstractPathAction.")
    value = jnp.asarray(action.log_weight(path))
    if (
        value.shape != ()
        or jnp.iscomplexobj(value)
        or not jnp.issubdtype(value.dtype, jnp.floating)
    ):
        raise ValueError("A path action must return one real floating scalar.")
    return jnp.where(ensemble.contains(path) & jnp.isfinite(value), value, -jnp.inf)


def make_incremental_path_target(
    ensemble: AbstractPathEnsemble,
    action: AbstractPathAction,
    /,
    *,
    refresh_cadence: int = 128,
    cache_tolerance: float = 1.0e-8,
) -> IncrementalMarkovTarget:
    """Expose exact local path ratios through the shared incremental-MH contract."""

    if not isinstance(ensemble, AbstractPathEnsemble) or not isinstance(
        action, AbstractPathAction
    ):
        raise TypeError("ensemble and action must implement path target contracts.")

    def evaluate(path):
        value = path_log_target(ensemble, action, path)
        return value, value

    def propose(current, cache, candidate, payload):
        del current, payload
        value = path_log_target(ensemble, action, candidate)
        return value - cache, value, jnp.isfinite(value)

    def select(current, candidate, accepted):
        return jax.tree_util.tree_map(
            lambda old, new: jnp.where(accepted, new, old), current, candidate
        )

    return IncrementalMarkovTarget(
        initialize=evaluate,
        propose=propose,
        select=select,
        refresh=evaluate,
        target_id=canonical_fingerprint(
            {
                "kind": "incremental-path-target-v1",
                "ensemble": ensemble.ensemble_id,
                "action": action.action_id,
            }
        ),
        refresh_cadence=refresh_cadence,
        cache_tolerance=cache_tolerance,
    )


class ReducedPathPotential(StrictModule, NonTrainableState):
    """Normalized reduced path potential eligible for FEP/BAR/MBAR cross-evaluation."""

    ensemble: AbstractPathEnsemble
    action: NormalizedStochasticPathAction
    potential_id: str = eqx.field(static=True)

    def __init__(
        self,
        ensemble: AbstractPathEnsemble,
        action: NormalizedStochasticPathAction,
        /,
        *,
        potential_id: str | None = None,
    ):
        if not isinstance(ensemble, AbstractPathEnsemble):
            raise TypeError("ensemble must implement AbstractPathEnsemble.")
        if (
            not isinstance(action, NormalizedStochasticPathAction)
            or not action.normalized
        ):
            raise ValueError(
                "Reduced path potentials require a normalized stochastic path action; "
                "deterministic and surrogate actions are unsupported."
            )
        identity = potential_id or canonical_fingerprint(
            {
                "kind": "reduced-path-potential-v1",
                "ensemble": ensemble.ensemble_id,
                "action": action.action_id,
            }
        )
        self.ensemble = ensemble
        self.action = action
        self.potential_id = _nonempty(identity, "potential_id")

    def evaluate(self, path: PathBuffer, /) -> Array:
        return -path_log_target(self.ensemble, self.action, path)


class PathCrossEvaluation(StrictModule, NonTrainableState):
    """Cross-evaluated reduced potentials and their source path identities."""

    samples: ReducedPotentialSamples
    path_ids: tuple[str, ...] = eqx.field(static=True)
    potential_ids: tuple[str, ...] = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)


def cross_evaluate_path_potentials(
    potentials: Sequence[ReducedPathPotential],
    paths: Sequence[PathBuffer],
    origin_states: ArrayLike,
    /,
) -> PathCrossEvaluation:
    """Build the existing FEP/BAR/MBAR matrix only across normalized path laws."""
    from ...uq._free_energy import ReducedPotentialSamples

    potential_values = tuple(potentials)
    path_values = tuple(paths)
    if not potential_values or any(
        not isinstance(value, ReducedPathPotential) for value in potential_values
    ):
        raise TypeError(
            "potentials must be a non-empty sequence of ReducedPathPotential values."
        )
    if not path_values or any(not isinstance(value, PathBuffer) for value in path_values):
        raise TypeError("paths must be a non-empty sequence of PathBuffer values.")
    identities = tuple(path_trajectory_id(path) for path in path_values)
    origins = np.asarray(origin_states, dtype=np.int32)
    if (
        origins.shape != (len(path_values),)
        or np.any(origins < 0)
        or np.any(origins >= len(potential_values))
    ):
        raise ValueError(
            "origin_states must identify one valid source potential per path."
        )
    matrix = jnp.stack(
        [
            jnp.stack([potential.evaluate(path) for path in path_values])
            for potential in potential_values
        ]
    )
    if not bool(jnp.all(jnp.isfinite(matrix))):
        raise ValueError(
            "Cross-evaluation left normalized common support; FEP/BAR/MBAR input is invalid."
        )
    counts = np.bincount(origins, minlength=len(potential_values)).astype(np.int32)
    potential_ids = tuple(value.potential_id for value in potential_values)
    evaluation_id = canonical_fingerprint(
        {
            "kind": "path-cross-evaluation-v1",
            "potentials": list(potential_ids),
            "paths": list(identities),
            "origins": origins.tolist(),
        }
    )
    samples = ReducedPotentialSamples(
        matrix,
        counts,
        origins,
        source_id=evaluation_id,
    )
    return PathCrossEvaluation(samples, identities, potential_ids, evaluation_id)


def path_fep_work(evaluation: PathCrossEvaluation, source: int, target: int, /) -> Array:
    if not isinstance(evaluation, PathCrossEvaluation):
        raise TypeError("evaluation must be PathCrossEvaluation.")
    source_, target_ = int(source), int(target)
    state_count = len(evaluation.potential_ids)
    if source_ < 0 or source_ >= state_count or target_ < 0 or target_ >= state_count:
        raise IndexError("source and target must identify cross-evaluated path states.")
    origins = np.asarray(evaluation.samples.origin_states)
    selected = np.nonzero(origins == source_)[0]
    if selected.size == 0:
        raise ValueError("The requested source has no path samples.")
    difference = evaluation.samples.values[target_] - evaluation.samples.values[source_]
    return difference[jnp.asarray(selected, dtype=jnp.int32)]


__all__ = [
    "AbstractPathAction",
    "AbstractPathEnsemble",
    "cross_evaluate_path_potentials",
    "DeterministicPathAction",
    "FirstPassagePathEnsemble",
    "FixedPathEnsemble",
    "InterfaceNetworkPlan",
    "InterfacePathEnsemble",
    "make_incremental_path_target",
    "MinusPathEnsemble",
    "NormalizedStochasticPathAction",
    "PathCrossEvaluation",
    "path_fep_work",
    "path_log_target",
    "ReducedPathPotential",
    "SurrogatePathAction",
]
