#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
from jaxtyping import Array

from .._strict import StrictModule
from ._model import VariableStateValues


class ExactFactorGraphStatus(IntEnum):
    """Portable terminal status for exhaustive finite-state inference."""

    SUCCESS = 0
    INFEASIBLE = 1
    NONFINITE_INPUT = 2
    RESOURCE_LIMIT = 3


class BeliefPropagationStatus(IntEnum):
    """Portable numerical status for sum- or max-product message passing."""

    SUCCESS = 0
    MAXIMUM_STEPS_REACHED = 1
    INFEASIBLE = 2
    NONFINITE_INPUT = 3
    NONFINITE_MESSAGE = 4


class GibbsTransitionStatus(IntEnum):
    """Portable status for one chromatic Gibbs sweep."""

    SUCCESS = 0
    INFEASIBLE_CONDITIONAL = 1
    INVALID_STATE = 2
    NONFINITE_SCORE = 3


class FactorGraphProvenance(StrictModule):
    """Static model, plan, method, and implementation identity."""

    structure_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    implementation: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    configuration: tuple[tuple[str, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        structure_id: str,
        plan_id: str,
        method_id: str,
        implementation: str,
        exact: bool,
        configuration: tuple[tuple[str, str], ...] = (),
    ):
        for name, value in (
            ("structure_id", structure_id),
            ("plan_id", plan_id),
            ("method_id", method_id),
            ("implementation", implementation),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty.")
        self.structure_id = structure_id
        self.plan_id = plan_id
        self.method_id = method_id
        self.implementation = implementation
        self.exact = bool(exact)
        self.configuration = tuple((str(key), str(value)) for key, value in configuration)


class ExactFactorGraphResult(StrictModule):
    """Exact finite-state normalizer, marginals, MAP state, and evidence."""

    log_normalizer: Array
    variable_probabilities: VariableStateValues
    factor_probabilities: tuple[Array, ...]
    map_assignment: Array
    map_log_score: Array
    feasible_configurations: Array
    total_configurations: int = eqx.field(static=True)
    status: Array
    valid: Array
    provenance: FactorGraphProvenance

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(ExactFactorGraphStatus.SUCCESS))

    @property
    def optimal(self) -> Array:
        return self.successful


class BeliefPropagationDiagnostics(StrictModule):
    """Normalized-message fixed-point evidence."""

    initial_residual: Array
    final_residual: Array
    iterations: Array
    support_changes: Array
    factor_evaluations: Array


class GibbsDiagnostics(StrictModule):
    """Transition validity, movement, and optional rank-mixing evidence."""

    invalid_conditional_count: Array
    mean_state_change_fraction: Array
    rhat: Array | None
    bulk_ess: Array | None
    tail_ess: Array | None
    max_rhat: Array
    min_bulk_ess: Array
    min_tail_ess: Array
    mixing_available: bool = eqx.field(static=True)


__all__ = [
    "BeliefPropagationDiagnostics",
    "BeliefPropagationStatus",
    "ExactFactorGraphResult",
    "ExactFactorGraphStatus",
    "FactorGraphProvenance",
    "GibbsDiagnostics",
    "GibbsTransitionStatus",
]
