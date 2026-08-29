#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._problem import LinearCombinatorialProblem
from ._types import (
    CombinatorialCertification,
    CombinatorialMethodCapabilities,
    CombinatorialResult,
)


class CombinatorialPlan(StrictModule):
    """Validated static execution contract for one native combinatorial solve."""

    certification: CombinatorialCertification
    capabilities: CombinatorialMethodCapabilities
    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    cost_dtype: str = eqx.field(static=True)
    decision_signature: tuple[tuple[str, tuple[int, ...], str], ...] = eqx.field(
        static=True
    )
    feature_signature: tuple[tuple[str, tuple[int, ...], str], ...] = eqx.field(
        static=True
    )
    work_estimate: int = eqx.field(static=True)
    workspace_elements: int = eqx.field(static=True)
    certificate_kind: str = eqx.field(static=True)
    configuration: tuple[tuple[str, str], ...] = eqx.field(static=True)


class AbstractLinearCombinatorialMethod(StrictModule):
    """Exact or approximate method for one declared linear combinatorial problem."""

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> CombinatorialMethodCapabilities:
        raise NotImplementedError

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return ()

    @abc.abstractmethod
    def plan(
        self,
        problem: LinearCombinatorialProblem,
        certification: CombinatorialCertification,
        /,
    ) -> CombinatorialPlan:
        raise NotImplementedError

    @abc.abstractmethod
    def solve(
        self,
        problem: LinearCombinatorialProblem,
        plan: CombinatorialPlan,
        /,
    ) -> CombinatorialResult:
        raise NotImplementedError


def _spec_signature(spec: PyTree[Any], /) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    path_specs, _ = jax.tree_util.tree_flatten_with_path(spec)
    records: list[tuple[str, tuple[int, ...], str]] = []
    for path, value in path_specs:
        if not isinstance(value, jax.ShapeDtypeStruct):
            raise TypeError(
                "combinatorial specifications must contain ShapeDtypeStruct leaves."
            )
        records.append(
            (
                jax.tree_util.keystr(path) or "<root>",
                tuple(int(size) for size in value.shape),
                str(value.dtype),
            )
        )
    return tuple(records)


def make_combinatorial_plan(
    problem: LinearCombinatorialProblem,
    method: AbstractLinearCombinatorialMethod,
    certification: CombinatorialCertification,
    /,
    *,
    work_estimate: int,
    workspace_elements: int,
    certificate_kind: str,
) -> CombinatorialPlan:
    """Build a content-addressed plan after method-specific validation."""

    if not isinstance(problem, LinearCombinatorialProblem):
        raise TypeError("problem must be a LinearCombinatorialProblem.")
    if not isinstance(method, AbstractLinearCombinatorialMethod):
        raise TypeError("method must be an AbstractLinearCombinatorialMethod.")
    if not isinstance(certification, CombinatorialCertification):
        raise TypeError("certification must be a CombinatorialCertification.")
    work = int(work_estimate)
    workspace = int(workspace_elements)
    if work < 0 or workspace < 0:
        raise ValueError(
            "combinatorial work and workspace estimates must be non-negative."
        )
    kind = str(certificate_kind)
    if not kind:
        raise ValueError("certificate_kind must be nonempty.")
    decision_signature = _spec_signature(problem.space.decision_spec())
    feature_signature = _spec_signature(problem.space.feature_spec())
    configuration = tuple((str(key), str(value)) for key, value in method.configuration)
    plan_id = canonical_fingerprint(
        {
            "kind": "native-combinatorial-plan",
            "problem_id": problem.problem_id,
            "structure_id": problem.structure_id,
            "method_id": method.method_id,
            "batch_shape": list(problem.batch_shape),
            "cost_dtype": problem.cost_dtype,
            "decision": [
                {"path": path, "shape": list(shape), "dtype": dtype}
                for path, shape, dtype in decision_signature
            ],
            "features": [
                {"path": path, "shape": list(shape), "dtype": dtype}
                for path, shape, dtype in feature_signature
            ],
            "configuration": dict(configuration),
            "certification": {
                "absolute": certification.absolute,
                "relative": certification.relative,
            },
            "work_estimate": work,
            "workspace_elements": workspace,
            "certificate_kind": kind,
        }
    )
    return CombinatorialPlan(
        certification=certification,
        capabilities=method.capabilities,
        problem_id=problem.problem_id,
        structure_id=problem.structure_id,
        method_id=method.method_id,
        plan_id=plan_id,
        batch_shape=problem.batch_shape,
        cost_dtype=problem.cost_dtype,
        decision_signature=decision_signature,
        feature_signature=feature_signature,
        work_estimate=work,
        workspace_elements=workspace,
        certificate_kind=kind,
        configuration=configuration,
    )


def plan_combinatorial(
    problem: LinearCombinatorialProblem,
    method: AbstractLinearCombinatorialMethod,
    /,
    *,
    certification: CombinatorialCertification | None = None,
) -> CombinatorialPlan:
    """Validate and size one native combinatorial execution without solving it."""

    if not isinstance(problem, LinearCombinatorialProblem):
        raise TypeError("problem must be a LinearCombinatorialProblem.")
    if not isinstance(method, AbstractLinearCombinatorialMethod):
        raise TypeError("method must be an AbstractLinearCombinatorialMethod.")
    selected = CombinatorialCertification() if certification is None else certification
    if not isinstance(selected, CombinatorialCertification):
        raise TypeError("certification must be a CombinatorialCertification or None.")
    return method.plan(problem, selected)


def solve_combinatorial(
    problem: LinearCombinatorialProblem,
    method: AbstractLinearCombinatorialMethod,
    /,
    *,
    certification: CombinatorialCertification | None = None,
) -> CombinatorialResult:
    """Solve one declared linear combinatorial problem with an explicit method."""

    plan = plan_combinatorial(problem, method, certification=certification)
    result = method.solve(problem, plan)
    if not isinstance(result, CombinatorialResult):
        raise TypeError("combinatorial methods must return CombinatorialResult.")
    return jax.tree_util.tree_map(jax.lax.stop_gradient, result)


__all__ = [
    "AbstractLinearCombinatorialMethod",
    "CombinatorialPlan",
    "make_combinatorial_plan",
    "plan_combinatorial",
    "solve_combinatorial",
]
