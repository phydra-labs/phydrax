#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import AbstractAttribute, StrictModule
from .._training import TrainingProgress
from ..domain import DomainFunction
from ..sampling.collocation import CausalTimeSlabSchedule
from ._functional_training import FunctionalTrainingPlan, FunctionalTrainingState


class FunctionalWindowAdapter(StrictModule):
    """Typed equation-specific bridge for one sequence of physical time windows."""

    __strict_abstract__ = True

    adapter_id: AbstractAttribute[str]

    @abc.abstractmethod
    def build_solver(
        self,
        previous_solver: Any,
        window_index: int,
        bounds: tuple[Array, Array],
        previous_terminal: Mapping[str, DomainFunction] | None,
        /,
    ) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def terminal_fields(
        self,
        solver: Any,
        window_index: int,
        bounds: tuple[Array, Array],
        /,
    ) -> Mapping[str, DomainFunction]:
        raise NotImplementedError

    @abc.abstractmethod
    def seam_metrics(
        self,
        previous_terminal: Mapping[str, DomainFunction],
        current_solver: Any,
        window_index: int,
        bounds: tuple[Array, Array],
        /,
    ) -> Mapping[str, Array]:
        raise NotImplementedError


class FunctionalTimeWindowPlan(StrictModule):
    """Causal host orchestration over explicit physical time windows."""

    schedule: CausalTimeSlabSchedule
    adapter: FunctionalWindowAdapter
    optimizer: Callable[[int], Any] = eqx.field(static=True)
    training: Callable[[int], FunctionalTrainingPlan | None] = eqx.field(static=True)
    steps: tuple[int, ...] = eqx.field(static=True)
    transfer_optimizer_state: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule: CausalTimeSlabSchedule,
        adapter: FunctionalWindowAdapter,
        optimizer: Callable[[int], Any],
        /,
        *,
        steps: int | Sequence[int],
        training: FunctionalTrainingPlan
        | Callable[[int], FunctionalTrainingPlan | None]
        | None = None,
        transfer_optimizer_state: bool = False,
        plan_id: str | None = None,
    ):
        if not isinstance(schedule, CausalTimeSlabSchedule):
            raise TypeError("schedule must be a CausalTimeSlabSchedule.")
        if not isinstance(adapter, FunctionalWindowAdapter):
            raise TypeError("adapter must implement FunctionalWindowAdapter.")
        if not callable(optimizer):
            raise TypeError("optimizer must be a callable window factory.")
        counts = (
            tuple(int(steps) for _ in range(schedule.slab_count))
            if isinstance(steps, int)
            else tuple(int(value) for value in steps)
        )
        if len(counts) != schedule.slab_count or any(value < 1 for value in counts):
            raise ValueError("steps must provide one positive count per window.")
        if training is None:
            training_factory = lambda index: None
        elif isinstance(training, FunctionalTrainingPlan):
            training_factory = lambda index, value=training: value
        elif callable(training):
            training_factory = training
        else:
            raise TypeError("training must be a plan, callable, or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "functional-time-windows",
                    "schedule": schedule.schedule_id,
                    "adapter": adapter.adapter_id,
                    "steps": counts,
                    "transfer_optimizer_state": bool(transfer_optimizer_state),
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.schedule = schedule
        self.adapter = adapter
        self.optimizer = optimizer
        self.training = training_factory
        self.steps = counts
        self.transfer_optimizer_state = bool(transfer_optimizer_state)
        self.plan_id = identifier


class FunctionalTimeWindowResult(StrictModule):
    solvers: tuple[Any, ...]
    terminal_fields: tuple[frozendict[str, DomainFunction], ...]
    seam_metrics: tuple[frozendict[str, Array], ...]
    boundaries: Array
    successful: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        solvers: Sequence[Any],
        terminal_fields: Sequence[Mapping[str, DomainFunction]],
        seam_metrics: Sequence[Mapping[str, Array]],
        boundaries: Array,
        /,
        *,
        plan_id: str,
    ):
        solvers_ = tuple(solvers)
        terminals = tuple(frozendict(value) for value in terminal_fields)
        seams = tuple(frozendict(value) for value in seam_metrics)
        if not solvers_ or len(terminals) != len(solvers_):
            raise ValueError("Time-window solvers and terminal fields must align.")
        expected_seams = max(len(solvers_) - 1, 0)
        if len(seams) != expected_seams:
            raise ValueError("Time-window seam metrics have an invalid count.")
        finite = all(
            bool(jnp.all(jnp.isfinite(value)))
            for metrics in seams
            for value in metrics.values()
        )
        self.solvers = solvers_
        self.terminal_fields = terminals
        self.seam_metrics = seams
        self.boundaries = jnp.asarray(boundaries)
        self.successful = jnp.asarray(finite)
        self.plan_id = str(plan_id)

    def solver_at(self, time: Array, /):
        value = jnp.asarray(time)
        if value.shape != ():
            raise ValueError("Window queries require one scalar physical time.")
        time_host = float(jax.device_get(value))
        boundaries = tuple(float(value) for value in jax.device_get(self.boundaries))
        if time_host < boundaries[0] or time_host > boundaries[-1]:
            raise ValueError("Window query lies outside the trained physical support.")
        index = min(
            sum(time_host >= boundary for boundary in boundaries[1:-1]),
            len(self.solvers) - 1,
        )
        return self.solvers[index]


def _transfer_training_state(source: Any, target: Any, /):
    state = source.training_state
    if state is None:
        raise ValueError(
            "Optimizer-state transfer requires a source FunctionalTrainingPlan."
        )
    source_structure = jax.tree.structure(source.functions)
    target_structure = jax.tree.structure(target.functions)
    if source_structure != target_structure:
        raise ValueError("Optimizer-state transfer requires identical function PyTrees.")
    transferred = FunctionalTrainingState(
        current_functions=target.functions,
        best_functions=target.functions,
        previous_functions=None,
        optimizer_state=state.optimizer_state,
        key=state.key,
        pseudo_inverse_steps=(),
        term_multipliers=(),
        previous_gradient=None,
        progress=TrainingProgress(),
        run_id=state.run_id,
        gradient_accumulation=state.gradient_accumulation,
        training_seconds=state.training_seconds,
        resumed_from_step=0,
    )
    return eqx.tree_at(
        lambda solver: solver.training_state,
        target,
        transferred,
        is_leaf=lambda value: value is None,
    )


def train_functional_time_windows(
    solver: Any,
    plan: FunctionalTimeWindowPlan,
    /,
) -> FunctionalTimeWindowResult:
    """Train ordered windows while propagating explicit terminal fields."""
    if not isinstance(plan, FunctionalTimeWindowPlan):
        raise TypeError("plan must be a FunctionalTimeWindowPlan.")
    current = solver
    previous_terminal = None
    trained = []
    terminals = []
    seams = []
    for index in range(plan.schedule.slab_count):
        bounds = plan.schedule.bounds(index)
        built = plan.adapter.build_solver(
            current,
            index,
            bounds,
            previous_terminal,
        )
        training = plan.training(index)
        if training is not None and not isinstance(training, FunctionalTrainingPlan):
            raise TypeError("Window training factory returned an invalid plan.")
        if plan.transfer_optimizer_state and training is None:
            raise ValueError(
                "Optimizer-state transfer requires a FunctionalTrainingPlan "
                "for every window."
            )
        if plan.transfer_optimizer_state and index > 0:
            built = _transfer_training_state(current, built)
        current = built.solve(
            num_iter=plan.steps[index],
            optim=plan.optimizer(index),
            training=training,
            resume=plan.transfer_optimizer_state and index > 0,
        )
        terminal = frozendict(plan.adapter.terminal_fields(current, index, bounds))
        if not terminal or any(
            not isinstance(value, DomainFunction) for value in terminal.values()
        ):
            raise TypeError("Window terminal fields must be named DomainFunction values.")
        if previous_terminal is not None:
            seams.append(
                frozendict(
                    plan.adapter.seam_metrics(
                        previous_terminal,
                        current,
                        index,
                        bounds,
                    )
                )
            )
        trained.append(current)
        terminals.append(terminal)
        previous_terminal = terminal
    return FunctionalTimeWindowResult(
        trained,
        terminals,
        seams,
        plan.schedule.boundaries,
        plan_id=plan.plan_id,
    )


__all__ = [
    "FunctionalTimeWindowPlan",
    "FunctionalTimeWindowResult",
    "FunctionalWindowAdapter",
    "train_functional_time_windows",
]
