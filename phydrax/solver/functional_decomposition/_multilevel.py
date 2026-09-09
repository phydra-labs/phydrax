#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import optax

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...domain import (
    LocalFieldFamily,
    partition_of_unity_field,
    SubdomainHierarchy,
)
from .._functional_correction import (
    FunctionalCorrectionProblem,
    prepare_functional_correction,
)
from .._functional_solver import FunctionalSolver


class FunctionalCoarseCorrection(StrictModule):
    """One frozen coarse field plus a trainable local partition-of-unity correction."""

    family: LocalFieldFamily
    correction: FunctionalCorrectionProblem
    field_name: str = eqx.field(static=True)

    def __init__(
        self,
        base_solver: FunctionalSolver,
        field_name: str,
        family: LocalFieldFamily,
        /,
        *,
        epsilon: float = 1.0,
    ):
        if not isinstance(base_solver, FunctionalSolver):
            raise TypeError("base_solver must be a FunctionalSolver.")
        if not isinstance(family, LocalFieldFamily):
            raise TypeError("family must be a LocalFieldFamily.")
        name = str(field_name)
        if name not in base_solver.functions:
            raise KeyError(f"Base solver has no field {name!r}.")
        fine = partition_of_unity_field(family)
        base = base_solver.functions[name]
        if not base.domain.same_support(fine.domain):
            raise ValueError("Coarse and fine fields must share the ambient support.")
        self.family = family
        self.correction = prepare_functional_correction(
            base_solver,
            {name: fine},
            epsilon=epsilon,
        )
        self.field_name = name

    @property
    def training_solver(self) -> FunctionalSolver:
        return self.correction.training_solver

    def finalize(self, trained: FunctionalSolver, /) -> FunctionalSolver:
        """Bind a trained fine correction to the original physical objective."""
        return self.correction.finalize(trained)


class FunctionalHierarchyPlan(StrictModule, NonTrainableState):
    """Per-level fixed work and nonlinear residual scaling."""

    iterations: tuple[int, ...] = eqx.field(static=True)
    epsilons: tuple[float, ...] = eqx.field(static=True)

    def __init__(
        self,
        iterations: Sequence[int],
        /,
        *,
        epsilons: Sequence[float] | None = None,
    ):
        iterations_ = tuple(int(value) for value in iterations)
        if not iterations_ or any(value <= 0 for value in iterations_):
            raise ValueError("iterations must contain positive per-level work counts.")
        epsilons_ = (
            tuple(1.0 for _ in iterations_)
            if epsilons is None
            else tuple(float(value) for value in epsilons)
        )
        if len(epsilons_) != len(iterations_) or any(
            not math.isfinite(value) or value <= 0.0 for value in epsilons_
        ):
            raise ValueError("epsilons must match levels and contain finite positives.")
        self.iterations = iterations_
        self.epsilons = epsilons_


class FunctionalHierarchyResult(StrictModule):
    solver: FunctionalSolver
    level_solvers: tuple[FunctionalSolver, ...]
    hierarchy_id: str = eqx.field(static=True)

    def __init__(
        self,
        solver: FunctionalSolver,
        level_solvers: Sequence[FunctionalSolver],
        /,
        *,
        hierarchy_id: str,
    ):
        self.solver = solver
        self.level_solvers = tuple(level_solvers)
        self.hierarchy_id = str(hierarchy_id)


def train_functional_hierarchy(
    base_solver: FunctionalSolver,
    field_name: str,
    hierarchy: SubdomainHierarchy,
    plan: FunctionalHierarchyPlan,
    optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    /,
    *,
    seed: int = 0,
    jit: bool = True,
) -> FunctionalHierarchyResult:
    """Train arbitrary ordered local correction levels against the full residual."""
    if not isinstance(base_solver, FunctionalSolver):
        raise TypeError("base_solver must be a FunctionalSolver.")
    if not isinstance(hierarchy, SubdomainHierarchy):
        raise TypeError("hierarchy must be a SubdomainHierarchy.")
    if not isinstance(plan, FunctionalHierarchyPlan):
        raise TypeError("plan must be a FunctionalHierarchyPlan.")
    if len(hierarchy.levels) != len(plan.iterations):
        raise ValueError("Hierarchy and training plan level counts must match.")
    name = str(field_name)
    if name not in base_solver.functions:
        raise KeyError(f"Base solver has no field {name!r}.")
    if not base_solver.functions[name].domain.same_support(hierarchy.ambient):
        raise ValueError("Base field and hierarchy ambient domains must match.")

    current = base_solver
    trained_levels = []
    for index, (level, iterations, epsilon) in enumerate(
        zip(hierarchy.levels, plan.iterations, plan.epsilons, strict=True)
    ):
        correction_field = level.field()
        correction = prepare_functional_correction(
            current,
            {name: correction_field},
            epsilon=epsilon,
        )
        trained = correction.training_solver.solve(
            num_iter=iterations,
            optim=optimizer,
            seed=seed + index,
            jit=jit,
            keep_best=False,
            log_every=0,
        )
        current = correction.finalize(trained)
        trained_levels.append(trained)
    return FunctionalHierarchyResult(
        current,
        tuple(trained_levels),
        hierarchy_id=hierarchy.hierarchy_id,
    )


class FunctionalCyclePlan(StrictModule, NonTrainableState):
    """Repeated V- or F-cycle correction work per hierarchy level."""

    cycles: int = eqx.field(static=True)
    descending_iterations: tuple[int, ...] = eqx.field(static=True)
    ascending_iterations: tuple[int, ...] = eqx.field(static=True)
    kind: Literal["v", "f"] = eqx.field(static=True)

    def __init__(
        self,
        cycles: int,
        descending_iterations: Sequence[int],
        /,
        *,
        ascending_iterations: Sequence[int] | None = None,
        kind: Literal["v", "f"] = "v",
    ):
        cycles_ = int(cycles)
        descending = tuple(int(value) for value in descending_iterations)
        ascending = (
            descending
            if ascending_iterations is None
            else tuple(int(value) for value in ascending_iterations)
        )
        if cycles_ <= 0 or not descending or any(value <= 0 for value in descending):
            raise ValueError("Cycle work counts must be positive.")
        if len(ascending) != len(descending) or any(value <= 0 for value in ascending):
            raise ValueError("Ascending work must match hierarchy levels.")
        if kind not in ("v", "f"):
            raise ValueError("kind must be 'v' or 'f'.")
        self.cycles = cycles_
        self.descending_iterations = descending
        self.ascending_iterations = ascending
        self.kind = kind

    def order(self, level_count: int, /) -> tuple[tuple[int, bool], ...]:
        if level_count != len(self.descending_iterations):
            raise ValueError("Cycle plan and hierarchy level counts must match.")
        if self.kind == "v":
            return tuple((index, True) for index in range(level_count)) + tuple(
                (index, False) for index in range(level_count - 2, -1, -1)
            )
        visits = []
        for finest in range(level_count):
            visits.extend((index, True) for index in range(finest + 1))
            visits.extend((index, False) for index in range(finest - 1, -1, -1))
        return tuple(visits)


class FunctionalCycleResult(StrictModule):
    solver: FunctionalSolver
    level_solvers: tuple[FunctionalSolver, ...]
    visits: tuple[str, ...] = eqx.field(static=True)
    cycles: int = eqx.field(static=True)

    def __init__(
        self,
        solver: FunctionalSolver,
        level_solvers: Sequence[FunctionalSolver],
        visits: Sequence[str],
        /,
        *,
        cycles: int,
    ):
        self.solver = solver
        self.level_solvers = tuple(level_solvers)
        self.visits = tuple(str(value) for value in visits)
        self.cycles = int(cycles)


def train_functional_cycles(
    base_solver: FunctionalSolver,
    field_name: str,
    hierarchy: SubdomainHierarchy,
    plan: FunctionalCyclePlan,
    optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
    /,
    *,
    seed: int = 0,
    jit: bool = True,
) -> FunctionalCycleResult:
    """Apply repeated residual-correction V/F cycles to the assembled field."""
    if not isinstance(plan, FunctionalCyclePlan):
        raise TypeError("plan must be a FunctionalCyclePlan.")
    order = plan.order(len(hierarchy.levels))
    current = base_solver
    trained_solvers = []
    visits = []
    visit_index = 0
    for _ in range(plan.cycles):
        for level_index, descending in order:
            level = hierarchy.levels[level_index]
            iterations = (
                plan.descending_iterations[level_index]
                if descending
                else plan.ascending_iterations[level_index]
            )
            correction = prepare_functional_correction(
                current,
                {str(field_name): level.field()},
                epsilon=1.0,
            )
            trained = correction.training_solver.solve(
                num_iter=iterations,
                optim=optimizer,
                seed=seed + visit_index,
                jit=jit,
                keep_best=False,
                log_every=0,
            )
            current = correction.finalize(trained)
            trained_solvers.append(trained)
            visits.append(f"{level.level_id}/{'down' if descending else 'up'}")
            visit_index += 1
    return FunctionalCycleResult(
        current,
        tuple(trained_solvers),
        tuple(visits),
        cycles=plan.cycles,
    )


__all__ = [
    "FunctionalCoarseCorrection",
    "FunctionalCyclePlan",
    "FunctionalCycleResult",
    "FunctionalHierarchyPlan",
    "FunctionalHierarchyResult",
    "train_functional_hierarchy",
    "train_functional_cycles",
]
