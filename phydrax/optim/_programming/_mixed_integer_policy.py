#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._branch_and_bound import BranchAndBoundPolicy
from ._policy import ConvexSolvePolicy


MixedIntegerBranchingRule: TypeAlias = Literal["most-fractional"]


class MixedIntegerMethodCapabilities(StrictModule, NonTrainableState):
    """Static geometry, execution, and proof capabilities of one method."""

    linear_program: bool = eqx.field(static=True)
    quadratic_program: bool = eqx.field(static=True)
    conic_program: bool = eqx.field(static=True)
    warm_start: bool = eqx.field(static=True)
    incumbent_start: bool = eqx.field(static=True)
    partial_start: bool = eqx.field(static=True)
    solution_pool: bool = eqx.field(static=True)
    global_cuts: bool = eqx.field(static=True)
    lazy_constraints: bool = eqx.field(static=True)
    incremental_rows: bool = eqx.field(static=True)
    prepared_refresh: bool = eqx.field(static=True)
    infeasibility_certificates: bool = eqx.field(static=True)
    independent_global_bound: bool = eqx.field(static=True)
    exact_arithmetic: bool = eqx.field(static=True)
    deterministic: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_program: bool,
        quadratic_program: bool,
        conic_program: bool,
        warm_start: bool,
        incumbent_start: bool,
        partial_start: bool,
        solution_pool: bool,
        global_cuts: bool,
        lazy_constraints: bool,
        incremental_rows: bool,
        prepared_refresh: bool,
        infeasibility_certificates: bool,
        independent_global_bound: bool,
        exact_arithmetic: bool,
        deterministic: bool,
    ):
        self.linear_program = bool(linear_program)
        self.quadratic_program = bool(quadratic_program)
        self.conic_program = bool(conic_program)
        self.warm_start = bool(warm_start)
        self.incumbent_start = bool(incumbent_start)
        self.partial_start = bool(partial_start)
        self.solution_pool = bool(solution_pool)
        self.global_cuts = bool(global_cuts)
        self.lazy_constraints = bool(lazy_constraints)
        self.incremental_rows = bool(incremental_rows)
        self.prepared_refresh = bool(prepared_refresh)
        self.infeasibility_certificates = bool(infeasibility_certificates)
        self.independent_global_bound = bool(independent_global_bound)
        self.exact_arithmetic = bool(exact_arithmetic)
        self.deterministic = bool(deterministic)


class AbstractMixedIntegerMethod(StrictModule):
    """Algorithm identity and declared capabilities for one discrete method."""

    __strict_abstract__ = True

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def backend(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> MixedIntegerMethodCapabilities:
        raise NotImplementedError

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return ()


class NativeMixedIntegerBranchAndBound(AbstractMixedIntegerMethod):
    """Deterministic host tree with independently audited convex relaxations."""

    relaxation: ConvexSolvePolicy
    tree: BranchAndBoundPolicy
    branching_rule: MixedIntegerBranchingRule = eqx.field(static=True)
    inherit_relaxation_start: bool = eqx.field(static=True)

    def __init__(
        self,
        relaxation: ConvexSolvePolicy | None = None,
        /,
        *,
        tree: BranchAndBoundPolicy | None = None,
        branching_rule: MixedIntegerBranchingRule = "most-fractional",
        inherit_relaxation_start: bool = False,
    ):
        relaxation_ = ConvexSolvePolicy() if relaxation is None else relaxation
        tree_ = BranchAndBoundPolicy() if tree is None else tree
        if not isinstance(relaxation_, ConvexSolvePolicy):
            raise TypeError("relaxation must be a ConvexSolvePolicy.")
        if not isinstance(tree_, BranchAndBoundPolicy):
            raise TypeError("tree must be a BranchAndBoundPolicy.")
        if relaxation_.failure.mode != "status":
            raise ValueError("Node relaxations require status failure mode.")
        if branching_rule != "most-fractional":
            raise ValueError("Only 'most-fractional' branching is supported.")
        if (
            bool(inherit_relaxation_start)
            and not relaxation_.method.capabilities.warm_start
        ):
            raise ValueError(
                "inherit_relaxation_start requires a relaxation method with warm starts."
            )
        self.relaxation = relaxation_
        self.tree = tree_
        self.branching_rule = branching_rule
        self.inherit_relaxation_start = bool(inherit_relaxation_start)

    @property
    def method_id(self) -> str:
        return "native-mixed-integer-branch-and-bound"

    @property
    def backend(self) -> str:
        return self.relaxation.method.backend

    @property
    def capabilities(self) -> MixedIntegerMethodCapabilities:
        relaxation = self.relaxation.method.capabilities
        return MixedIntegerMethodCapabilities(
            linear_program=relaxation.linear_program,
            quadratic_program=relaxation.quadratic_program,
            conic_program=relaxation.conic_program,
            warm_start=relaxation.warm_start,
            incumbent_start=True,
            partial_start=False,
            solution_pool=False,
            global_cuts=False,
            lazy_constraints=False,
            incremental_rows=False,
            prepared_refresh=relaxation.prepared_refresh,
            infeasibility_certificates=relaxation.infeasibility_certificates,
            independent_global_bound=True,
            exact_arithmetic=False,
            deterministic=True,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (
            ("relaxation_policy", self.relaxation.policy_id),
            ("maximum_nodes", str(self.tree.maximum_nodes)),
            ("absolute_gap", repr(self.tree.absolute_gap)),
            ("relative_gap", repr(self.tree.relative_gap)),
            ("branching_rule", self.branching_rule),
            ("inherit_relaxation_start", str(self.inherit_relaxation_start)),
        )


class MixedIntegerCertification(StrictModule, NonTrainableState):
    """Scale-aware primal, integrality, and objective audit tolerances."""

    feasibility: float = eqx.field(static=True)
    integrality: float = eqx.field(static=True)
    objective: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        feasibility: float = 1e-7,
        integrality: float = 1e-7,
        objective: float = 1e-7,
    ):
        values = tuple(float(value) for value in (feasibility, integrality, objective))
        if any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "Mixed-integer audit tolerances must be finite and nonnegative."
            )
        if not 0.0 < values[1] < 0.5:
            raise ValueError("integrality must lie in (0, 0.5).")
        self.feasibility, self.integrality, self.objective = values


class MixedIntegerSolvePolicy(StrictModule):
    """One explicit mixed-integer method plus independent audit tolerances."""

    method: AbstractMixedIntegerMethod
    certification: MixedIntegerCertification
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        method: AbstractMixedIntegerMethod | None = None,
        /,
        *,
        certification: MixedIntegerCertification | None = None,
    ):
        method_ = NativeMixedIntegerBranchAndBound() if method is None else method
        certification_ = (
            MixedIntegerCertification() if certification is None else certification
        )
        if not isinstance(method_, AbstractMixedIntegerMethod):
            raise TypeError("method must be an AbstractMixedIntegerMethod.")
        if not isinstance(certification_, MixedIntegerCertification):
            raise TypeError("certification must be MixedIntegerCertification.")
        self.method = method_
        self.certification = certification_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "mixed-integer-solve-policy",
                "method": method_.method_id,
                "backend": method_.backend,
                "configuration": list(method_.configuration),
                "feasibility": certification_.feasibility,
                "integrality": certification_.integrality,
                "objective": certification_.objective,
            }
        )


__all__ = [
    "AbstractMixedIntegerMethod",
    "MixedIntegerBranchingRule",
    "MixedIntegerCertification",
    "MixedIntegerMethodCapabilities",
    "MixedIntegerSolvePolicy",
    "NativeMixedIntegerBranchAndBound",
]
