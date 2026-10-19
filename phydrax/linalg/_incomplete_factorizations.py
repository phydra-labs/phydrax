#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite
from typing import Any

import equinox as eqx
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._trainable import NonTrainableState
from ..typing import checked, parse
from ._costs import PreconditionerCostEstimate
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator
from ._preconditioner_properties import PreconditionerProperties
from ._preconditioners import AbstractPreconditioner, PrecisionCastPreconditioner
from ._preconditioning import AbstractPreconditionerBuilder, PlannedPreconditionerSetup
from ._properties import LinearCapabilityError, LinearResourceLimitError
from ._spaces import coordinate_dtype
from ._sparse_contract import AbstractSparseLinearOperator
from ._sparse_factorizations import (
    _refresh_workspace_bytes,
    prepare_sparse_factor_congruence,
    prepare_sparse_factorization,
    PreparedSparseFactorCongruence,
    PreparedSparseFactorization,
    refresh_sparse_factorization,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationStatus,
)
from ._sparse_ordering import _pattern_identifier, _validated_pattern, SparseOrdering


class SparseFactorizationPreconditioner(AbstractPreconditioner, NonTrainableState):
    """Native factors and conditional correction admission, with raw failure evidence.

    A Cholesky correction is SPD only on its native SUCCESS branch. Failed
    factors remain inspectable and are refused before any triangular action.
    """

    factorization: PreparedSparseFactorization

    @checked
    def __init__(
        self,
        operator: AbstractSparseLinearOperator,
        factorization: PreparedSparseFactorization,
        /,
        *,
        properties: PreconditionerProperties,
        preconditioner_id: str,
    ) -> None:
        if not operator.source.compatible(operator.target):
            raise ValueError(
                "Sparse factor preconditioning requires one compatible space."
            )
        identifier = str(preconditioner_id)
        if not identifier:
            raise ValueError("preconditioner_id must be non-empty.")
        cholesky = factorization.plan.kind == "cholesky"
        expected_positive = cholesky and (
            operator.properties.certifies("positive_definite")
            or operator.properties.certifies("positive_semidefinite")
        )
        if cholesky and not operator.properties.certifies("self_adjoint"):
            raise ValueError(
                "Sparse Cholesky preconditioning requires certified self-adjointness."
            )
        if (
            not properties.linear
            or not properties.stationary
            or properties.self_adjoint != cholesky
            or properties.positive_definite != expected_positive
        ):
            raise ValueError(
                "Sparse factor preconditioner claims must match the factor kind and setup-operator evidence."
            )
        self.space = operator.source
        self.properties = properties
        self.preconditioner_id = identifier
        self.factorization = factorization

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        del iteration
        return self._apply(residual, adjoint=False)

    def apply_adjoint(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        """Apply the conjugate-transposed sparse factor inverse."""
        del iteration
        return self._apply(residual, adjoint=True)

    def _apply(self, residual: PyTree[Any], /, *, adjoint: bool) -> PyTree[Array]:
        coordinates = self.space.flatten(self.space.validate(residual))
        coordinates = eqx.error_if(
            coordinates,
            self.factorization.status != int(SparseFactorizationStatus.SUCCESS),
            "Native sparse correction refused its failed factor; inspect sparse_preconditioner_factorization.",
        )
        solved = self.factorization.solve(coordinates, adjoint=adjoint)
        value = eqx.error_if(
            solved.value,
            solved.status != int(SparseFactorizationStatus.SUCCESS),
            "Sparse factor preconditioner solve failed; inspect factor diagnostics.",
        )
        return self.space.unflatten(value)


class SparseFactorCongruencePreconditioner(AbstractPreconditioner, NonTrainableState):
    """Conditional SPD correction from native LU, not a positivity claim on A."""

    artifact: PreparedSparseFactorCongruence

    def __init__(
        self,
        operator: AbstractSparseLinearOperator,
        artifact: PreparedSparseFactorCongruence,
        /,
        *,
        properties: PreconditionerProperties,
        preconditioner_id: str,
    ) -> None:
        if not operator.source.compatible(operator.target):
            raise ValueError(
                f"SparseFactorCongruencePreconditioner.operator must have compatible source and target spaces; got source={operator.source.space_id!r}, target={operator.target.space_id!r}."
            )
        if not all(
            (
                properties.linear,
                properties.stationary,
                properties.self_adjoint,
                properties.positive_definite,
            )
        ):
            raise ValueError(
                f"SparseFactorCongruencePreconditioner.properties must describe a linear stationary self-adjoint positive-definite action; got linear={properties.linear}, stationary={properties.stationary}, self_adjoint={properties.self_adjoint}, positive_definite={properties.positive_definite}."
            )
        if not preconditioner_id:
            raise ValueError(
                f"SparseFactorCongruencePreconditioner.preconditioner_id must be non-empty; got {preconditioner_id!r}."
            )
        self.space = operator.source
        self.properties = properties
        self.preconditioner_id = preconditioner_id
        self.artifact = artifact

    @property
    def factorization(self) -> PreparedSparseFactorization:
        return self.artifact.factorization

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        del iteration
        # A coordinate Hermitian metric acts on the covector represented by
        # the declared scientific pairing, rather than assuming Euclidean Riesz.
        coordinates = self.space.flatten(self.space.riesz(self.space.validate(residual)))
        coordinates = eqx.error_if(
            coordinates,
            self.artifact.status != int(SparseFactorizationStatus.SUCCESS),
            "Native sparse congruence refused its failed factor; inspect artifact status.",
        )
        solved = self.artifact.solve(coordinates)
        value = eqx.error_if(
            solved.value,
            solved.status != int(SparseFactorizationStatus.SUCCESS),
            "Sparse factor congruence solve failed; inspect artifact diagnostics.",
        )
        return self.space.unflatten(value)


def sparse_preconditioner_factorization(
    action: AbstractPreconditioner | None, /
) -> PreparedSparseFactorization | None:
    """Return native factor evidence through precision and block holders."""
    from ._block_preconditioning import BlockFactorizationPreconditioner

    while isinstance(action, PrecisionCastPreconditioner):
        action = action.inner
    if isinstance(action, BlockFactorizationPreconditioner):
        pivot = sparse_preconditioner_factorization(action.pivot_action)
        schur = sparse_preconditioner_factorization(action.schur_action)
        if pivot is not None and schur is not None:
            raise ValueError(
                f"sparse_preconditioner_factorization.action must contain at most one native block factor; got pivot={type(pivot).__name__}, schur={type(schur).__name__}."
            )
        return schur if schur is not None else pivot
    return (
        action.factorization
        if isinstance(
            action,
            (SparseFactorizationPreconditioner, SparseFactorCongruencePreconditioner),
        )
        else None
    )


def _factor_preconditioner_properties(
    setup_operator: AbstractLinearOperator,
    /,
    *,
    cholesky: bool,
) -> PreconditionerProperties:
    if cholesky and not setup_operator.properties.certifies("self_adjoint"):
        raise ValueError(
            "Sparse Cholesky preconditioning requires certified self-adjointness."
        )
    positive = cholesky and (
        setup_operator.properties.certifies("positive_definite")
        or setup_operator.properties.certifies("positive_semidefinite")
    )
    claims = {
        "linear": True,
        "stationary": True,
        "self_adjoint": cholesky,
        "positive_definite": positive,
    }
    return PreconditionerProperties(
        **claims,
        evidence={name: "construction" for name, claimed in claims.items() if claimed},
    )


class _AbstractSparseFactorizationBuilder(AbstractPreconditionerBuilder):
    @abc.abstractmethod
    def policy(self) -> SparseFactorizationPolicy:
        raise NotImplementedError

    @property
    def builder_id(self) -> str:
        policy = self.policy()
        return canonical_fingerprint(
            {
                "kind": "sparse-factorization-preconditioner-builder",
                "policy": {
                    "kind": policy.kind,
                    "ordering": policy.ordering,
                    "fill_level": policy.fill_level,
                    "drop_tolerance": policy.drop_tolerance,
                    "maximum_fill_per_row": policy.maximum_fill_per_row,
                    "pivot_tolerance": policy.pivot_tolerance,
                    "diagonal_shift": policy.diagonal_shift,
                    "allow_pivot_replacement": policy.allow_pivot_replacement,
                    "replacement_value": policy.replacement_value,
                    "max_factor_nnz": policy.max_factor_nnz,
                    "max_factor_bytes": policy.max_factor_bytes,
                    "max_symbolic_work": policy.max_symbolic_work,
                },
            }
        )

    @property
    def default_refresh(self) -> str:
        return "numeric"

    @checked
    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        if setup_operator.batch_shape or not setup_operator.source.compatible(
            setup_operator.target
        ):
            raise ValueError("Sparse factorization requires an unbatched endomorphism.")
        policy = self.policy()
        cholesky = policy.kind == "cholesky" or (
            policy.kind == "auto"
            and setup_operator.properties.certifies("positive_definite")
        )
        return _factor_preconditioner_properties(
            setup_operator,
            cholesky=cholesky,
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        return self.plan_setup(setup_operator, materialization=materialization).cost

    def plan_setup(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PlannedPreconditionerSetup:
        # The setup operator already stores its sparse values, so nothing is
        # densely materialized; the plan charges the retained factor to
        # SolveResourcePolicy.preconditioner_bytes.
        del materialization
        self.properties_for(setup_operator)
        if not isinstance(setup_operator, AbstractSparseLinearOperator):
            return PlannedPreconditionerSetup(
                PreconditionerCostEstimate(
                    component=self.builder_id,
                    accepted=False,
                    reason="sparse factorization requires canonical sparse operator storage",
                )
            )
        try:
            plan = prepare_sparse_factorization(setup_operator, self.policy())
        except LinearCapabilityError as error:
            return PlannedPreconditionerSetup(
                PreconditionerCostEstimate(
                    component=self.builder_id,
                    accepted=False,
                    reason=str(error),
                )
            )
        itemsize = setup_operator.sparse_storage().values.dtype.itemsize
        factor_entries = plan.factor_indices.size
        cost = PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=plan.factor_bytes,
            preparation_workspace_bytes=factor_entries * itemsize
            + _refresh_workspace_bytes(plan, itemsize),
            apply_workspace_bytes_per_rhs=4 * plan.shape[0] * itemsize,
            accepted=True,
            reason=(
                "fixed-pattern sparse factorization; "
                f"factor_nnz={plan.factor_nnz}, factor_bytes={plan.factor_bytes}, "
                f"factor_storage_bytes_upper={plan.factor_storage_bytes_upper}, "
                f"symbolic_work={plan.symbolic_work}"
            ),
        )
        return PlannedPreconditionerSetup(
            cost, construction=plan, content_id=setup_operator.operator_id
        )

    def prepare_planned(
        self,
        planned: PlannedPreconditionerSetup,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> SparseFactorizationPreconditioner | SparseFactorCongruencePreconditioner:
        properties = self.properties_for(setup_operator)
        if not isinstance(setup_operator, AbstractSparseLinearOperator):
            raise TypeError("Sparse factorization requires a sparse operator.")
        plan = planned.construction_for(setup_operator.operator_id)
        if plan is None:
            return self.prepare(setup_operator, materialization=materialization)
        if not isinstance(plan, SparseFactorizationPlan):
            raise ValueError(
                "Sparse factor preparation requires its planned construction."
            )
        factorization = refresh_sparse_factorization(plan, setup_operator)
        return SparseFactorizationPreconditioner(
            setup_operator,
            factorization,
            properties=properties,
            preconditioner_id=f"{self.builder_id}/{plan.plan_id}",
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> SparseFactorizationPreconditioner | SparseFactorCongruencePreconditioner:
        del materialization
        properties = self.properties_for(setup_operator)
        if not isinstance(setup_operator, AbstractSparseLinearOperator):
            raise TypeError("Sparse factorization requires a sparse operator.")
        plan = prepare_sparse_factorization(setup_operator, self.policy())
        factorization = refresh_sparse_factorization(plan, setup_operator)
        return SparseFactorizationPreconditioner(
            setup_operator,
            factorization,
            properties=properties,
            preconditioner_id=f"{self.builder_id}/{plan.plan_id}",
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> SparseFactorizationPreconditioner | SparseFactorCongruencePreconditioner:
        del materialization
        if not isinstance(preconditioner, SparseFactorizationPreconditioner):
            raise TypeError(
                "Sparse factor refresh requires SparseFactorizationPreconditioner."
            )
        properties = self.properties_for(setup_operator)
        if not isinstance(setup_operator, AbstractSparseLinearOperator):
            raise TypeError("Sparse factorization requires a sparse operator.")
        factorization = refresh_sparse_factorization(
            preconditioner.factorization.plan,
            setup_operator,
        )
        return SparseFactorizationPreconditioner(
            setup_operator,
            factorization,
            properties=properties,
            preconditioner_id=preconditioner.preconditioner_id,
        )


class SparseFactorizationPreconditionerBuilder(
    _AbstractSparseFactorizationBuilder, NonTrainableState
):
    """Prepare sparse inverse or LU congruence actions from one symbolic plan.

    Planned setup retains the factor payload and admission reservation;
    numeric binding reuses that construction for the same operator identity.
    """

    factorization_policy: SparseFactorizationPolicy = eqx.field(static=True)
    form: str = eqx.field(static=True)
    prepared_plan: SparseFactorizationPlan | None
    _prepared_operator_id: str | None = eqx.field(static=True)
    _prepared_space_id: str | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        policy: SparseFactorizationPolicy | None = None,
        /,
        *,
        form: str = "inverse",
        prepared_plan: SparseFactorizationPlan | None = None,
        setup_operator: AbstractSparseLinearOperator | None = None,
    ) -> None:
        if form not in ("inverse", "lu-congruence"):
            raise ValueError(
                f"SparseFactorizationPreconditionerBuilder.form must be 'inverse' or 'lu-congruence'; got {form!r}."
            )
        policy_ = (
            prepared_plan.policy
            if policy is None and prepared_plan is not None
            else SparseFactorizationPolicy()
            if policy is None
            else policy
        )
        if policy_.fill_level is not None:
            raise ValueError("A complete sparse coarse solve requires fill_level=None.")
        if form == "lu-congruence" and (
            policy_.kind != "lu"
            or policy_.allow_pivot_replacement
            or policy_.diagonal_shift != 0.0
        ):
            raise ValueError(
                f"SparseFactorizationPreconditionerBuilder.policy must specify unshifted LU without pivot replacement for lu-congruence; got kind={policy_.kind!r}, diagonal_shift={policy_.diagonal_shift!r}, allow_pivot_replacement={policy_.allow_pivot_replacement!r}."
            )
        self.form = form
        if (prepared_plan is None) != (setup_operator is None):
            raise ValueError(
                f"SparseFactorizationPreconditionerBuilder.setup_operator must be supplied together with prepared_plan; got prepared_plan={type(prepared_plan).__name__}, setup_operator={type(setup_operator).__name__}."
            )
        if prepared_plan is not None:
            if setup_operator is None:
                raise TypeError(
                    f"SparseFactorizationPreconditionerBuilder.setup_operator must be AbstractSparseLinearOperator; got {type(setup_operator).__name__}."
                )
            if not eqx.tree_equal(prepared_plan.policy, policy_):
                raise ValueError(
                    f"SparseFactorizationPreconditionerBuilder.prepared_plan.policy must equal the requested policy; got prepared={prepared_plan.policy!r}, requested={policy_!r}."
                )
            storage, indices, indptr = _validated_pattern(setup_operator)
            if (
                _pattern_identifier(storage.shape, indices, indptr)
                != prepared_plan.input_pattern_id
            ):
                raise ValueError(
                    f"SparseFactorizationPreconditionerBuilder.setup_operator must match the prepared CSR pattern; got pattern_id={_pattern_identifier(storage.shape, indices, indptr)!r}, expected={prepared_plan.input_pattern_id!r}."
                )
            if not setup_operator.source.compatible(setup_operator.target):
                raise ValueError(
                    f"SparseFactorizationPreconditionerBuilder.setup_operator must have compatible source and target spaces; got source={setup_operator.source.space_id!r}, target={setup_operator.target.space_id!r}."
                )
            self._prepared_operator_id = setup_operator.operator_id
            self._prepared_space_id = setup_operator.source.space_id
        else:
            self._prepared_operator_id = None
            self._prepared_space_id = None
        self.prepared_plan = prepared_plan
        self.factorization_policy = policy_

    def policy(self) -> SparseFactorizationPolicy:
        return self.factorization_policy

    @property
    def builder_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "native-sparse-factorization-preconditioner",
                "form": self.form,
                "policy_builder": super().builder_id,
                "prepared_plan": None
                if self.prepared_plan is None
                else self.prepared_plan.plan_id,
                "prepared_operator": self._prepared_operator_id,
                "prepared_space": self._prepared_space_id,
            }
        )

    def _resolved_plan(
        self, operator: AbstractSparseLinearOperator, /
    ) -> SparseFactorizationPlan:
        if self.prepared_plan is None:
            return prepare_sparse_factorization(operator, self.policy())
        if (
            operator.operator_id != self._prepared_operator_id
            or operator.source.space_id != self._prepared_space_id
            or not operator.source.compatible(operator.target)
        ):
            raise ValueError(
                f"SparseFactorizationPreconditionerBuilder.operator must preserve the prepared operator and vector space; got operator_id={operator.operator_id!r}, source={operator.source.space_id!r}, target={operator.target.space_id!r}, expected_operator={self._prepared_operator_id!r}, expected_space={self._prepared_space_id!r}."
            )
        return self.prepared_plan

    def _admit_congruence(self, plan: SparseFactorizationPlan, itemsize: int) -> None:
        if self.form != "lu-congruence":
            return
        required = (
            plan.factor_storage_bytes_upper
            + plan.lu_congruence_storage_bytes_upper(itemsize)
        )
        if required > plan.policy.max_factor_bytes:
            raise LinearResourceLimitError(
                f"SparseFactorizationPolicy.max_factor_bytes must cover LU congruence storage; got limit={plan.policy.max_factor_bytes}, required={required}.",
                resource="sparse_factorization:factor_bytes",
                limit=plan.policy.max_factor_bytes,
                requested=required,
                completed=plan.factor_nnz,
                symbolic_work=plan.symbolic_work,
                storage_bytes_upper=required,
            )

    def properties_for(
        self, setup_operator: AbstractLinearOperator, /
    ) -> PreconditionerProperties:
        if self.form == "inverse":
            return super().properties_for(setup_operator)
        if setup_operator.batch_shape or not setup_operator.source.compatible(
            setup_operator.target
        ):
            raise ValueError(
                f"SparseFactorizationPreconditionerBuilder.setup_operator must be an unbatched endomorphism; got batch_shape={setup_operator.batch_shape}, source={setup_operator.source.space_id!r}, target={setup_operator.target.space_id!r}."
            )
        return PreconditionerProperties(
            linear=True,
            stationary=True,
            self_adjoint=True,
            positive_definite=True,
            evidence={
                name: "construction"
                for name in ("linear", "stationary", "self_adjoint", "positive_definite")
            },
        )

    def _action(
        self,
        operator: AbstractSparseLinearOperator,
        factorization: PreparedSparseFactorization,
        properties: PreconditionerProperties,
        identifier: str,
    ) -> SparseFactorizationPreconditioner | SparseFactorCongruencePreconditioner:
        if self.form == "lu-congruence":
            return SparseFactorCongruencePreconditioner(
                operator,
                prepare_sparse_factor_congruence(factorization),
                properties=properties,
                preconditioner_id=identifier,
            )
        return SparseFactorizationPreconditioner(
            operator,
            factorization,
            properties=properties,
            preconditioner_id=identifier,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> SparseFactorizationPreconditioner | SparseFactorCongruencePreconditioner:
        del materialization
        expected = (
            SparseFactorCongruencePreconditioner
            if self.form == "lu-congruence"
            else SparseFactorizationPreconditioner
        )
        if not isinstance(preconditioner, expected):
            raise TypeError(
                f"SparseFactorizationPreconditionerBuilder.preconditioner must match the builder action form; got {type(preconditioner).__name__}, expected={expected.__name__}."
            )
        if not isinstance(setup_operator, AbstractSparseLinearOperator):
            raise TypeError(
                f"SparseFactorizationPreconditionerBuilder.setup_operator must be AbstractSparseLinearOperator; got {type(setup_operator).__name__}."
            )
        self._admit_congruence(
            preconditioner.factorization.plan,
            setup_operator.sparse_storage().values.dtype.itemsize,
        )
        return self._action(
            setup_operator,
            refresh_sparse_factorization(
                preconditioner.factorization.plan, setup_operator
            ),
            self.properties_for(setup_operator),
            preconditioner.preconditioner_id,
        )

    def plan_setup(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PlannedPreconditionerSetup:
        if not isinstance(setup_operator, AbstractSparseLinearOperator):
            return super().plan_setup(setup_operator, materialization=materialization)
        if self.form == "inverse" and self.prepared_plan is None:
            return super().plan_setup(setup_operator, materialization=materialization)
        self.properties_for(setup_operator)
        try:
            plan = self._resolved_plan(setup_operator)
            self._admit_congruence(
                plan, setup_operator.sparse_storage().values.dtype.itemsize
            )
        except LinearCapabilityError as error:
            return PlannedPreconditionerSetup(
                PreconditionerCostEstimate(
                    component=self.builder_id, accepted=False, reason=str(error)
                )
            )
        coefficient_dtype = setup_operator.sparse_storage().values.dtype
        itemsize = (
            coefficient_dtype.itemsize
            if self.form == "lu-congruence"
            else coordinate_dtype(setup_operator.source).itemsize
        )
        congruence = self.form == "lu-congruence"
        extra_storage = (
            plan.numeric_cache_storage_bytes_for(coefficient_dtype)
            + plan.lu_congruence_storage_bytes_for(coefficient_dtype)
            if congruence
            else 0
        )
        extra_workspace = (
            plan.lu_congruence_refresh_workspace_bytes_upper(itemsize)
            if congruence
            else 0
        )
        cost = PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=plan.factor_bytes + extra_storage,
            preparation_workspace_bytes=plan.factor_nnz * itemsize
            + _refresh_workspace_bytes(plan, itemsize)
            + extra_workspace,
            apply_workspace_bytes_per_rhs=(
                plan.lu_congruence_apply_workspace_bytes_upper(
                    max(itemsize, coordinate_dtype(setup_operator.source).itemsize)
                )
                if congruence
                else 4 * plan.shape[0] * itemsize
            ),
            accepted=True,
            reason=(
                f"native {self.form}; factor_nnz={plan.factor_nnz}, factor_bytes={plan.factor_bytes}, "
                f"factor_storage_bytes_upper={plan.factor_storage_bytes_upper}, symbolic_work={plan.symbolic_work}; "
                f"numeric_preparation_work_units={plan.numeric_substitution_preparation_work_units_upper}; "
                f"extra_preparation_work_units={plan.lu_congruence_preparation_work_units_upper if congruence else 0}; "
                f"apply_work_units={plan.lu_congruence_solve_work_units_upper_for(coefficient_dtype, coordinate_dtype(setup_operator.source)) if congruence else plan.solve_work_units_upper_for(coefficient_dtype, coordinate_dtype(setup_operator.source))}"
            ),
        )
        return PlannedPreconditionerSetup(
            cost, construction=plan, content_id=setup_operator.operator_id
        )

    def prepare_planned(
        self,
        planned: PlannedPreconditionerSetup,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> SparseFactorizationPreconditioner | SparseFactorCongruencePreconditioner:
        if not planned.cost.accepted:
            raise LinearCapabilityError(planned.cost.reason)
        if planned.cost.component != self.builder_id:
            raise ValueError(
                f"SparseFactorizationPreconditionerBuilder.planned.cost.component must match the builder; got {planned.cost.component!r}, expected={self.builder_id!r}."
            )
        properties = self.properties_for(setup_operator)
        if not isinstance(setup_operator, AbstractSparseLinearOperator):
            raise TypeError(
                f"SparseFactorizationPreconditionerBuilder.setup_operator must be AbstractSparseLinearOperator; got {type(setup_operator).__name__}."
            )
        plan = planned.construction_for(setup_operator.operator_id)
        if plan is None:
            return self.prepare(setup_operator, materialization=materialization)
        if not isinstance(plan, SparseFactorizationPlan):
            raise ValueError(
                f"SparseFactorizationPreconditionerBuilder.planned.construction must be SparseFactorizationPlan; got {type(plan).__name__}."
            )
        if self.prepared_plan is not None:
            self._resolved_plan(setup_operator)
        # Reserve the planned dtype before allocation; refresh owns refusal of
        # a changed numeric dtype rather than charging it as a new construction.
        self._admit_congruence(plan, np.dtype(plan.value_dtype).itemsize)
        factorization = refresh_sparse_factorization(plan, setup_operator)
        return self._action(
            setup_operator,
            factorization,
            properties,
            f"{self.builder_id}/{plan.plan_id}",
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> SparseFactorizationPreconditioner | SparseFactorCongruencePreconditioner:
        del materialization
        if not isinstance(setup_operator, AbstractSparseLinearOperator):
            raise TypeError(
                f"SparseFactorizationPreconditionerBuilder.setup_operator must be AbstractSparseLinearOperator; got {type(setup_operator).__name__}."
            )
        properties = self.properties_for(setup_operator)
        plan = self._resolved_plan(setup_operator)
        self._admit_congruence(
            plan, setup_operator.sparse_storage().values.dtype.itemsize
        )
        return self._action(
            setup_operator,
            refresh_sparse_factorization(plan, setup_operator),
            properties,
            f"{self.builder_id}/{plan.plan_id}",
        )


class ILUPreconditionerBuilder(_AbstractSparseFactorizationBuilder):
    """Fixed-pattern level-of-fill ILU(k) builder."""

    fill_level: int = eqx.field(static=True)
    ordering: SparseOrdering = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    diagonal_shift: float = eqx.field(static=True)
    allow_pivot_replacement: bool = eqx.field(static=True)
    replacement_value: float = eqx.field(static=True)

    def __init__(
        self,
        fill_level: int = 0,
        /,
        *,
        ordering: SparseOrdering = "natural",
        pivot_tolerance: float = 0.0,
        diagonal_shift: float = 0.0,
        allow_pivot_replacement: bool = False,
        replacement_value: float = 1e-12,
    ) -> None:
        fill = int(fill_level)
        numeric = tuple(
            float(value) for value in (pivot_tolerance, diagonal_shift, replacement_value)
        )
        if fill < 0:
            raise ValueError("fill_level must be non-negative.")
        if any(not isfinite(value) for value in numeric):
            raise ValueError("ILU numeric policies must be finite.")
        if numeric[0] < 0.0 or numeric[1] < 0.0 or numeric[2] <= 0.0:
            raise ValueError("ILU pivot/shift policies are invalid.")
        ordering = parse(ordering, SparseOrdering, "ordering")
        self.fill_level = fill
        self.ordering = ordering
        self.pivot_tolerance = numeric[0]
        self.diagonal_shift = numeric[1]
        self.allow_pivot_replacement = bool(allow_pivot_replacement)
        self.replacement_value = numeric[2]

    def policy(self) -> SparseFactorizationPolicy:
        return SparseFactorizationPolicy(
            "lu",
            ordering=self.ordering,
            fill_level=self.fill_level,
            pivot_tolerance=self.pivot_tolerance,
            diagonal_shift=self.diagonal_shift,
            allow_pivot_replacement=self.allow_pivot_replacement,
            replacement_value=self.replacement_value,
        )


class ILUTPreconditionerBuilder(_AbstractSparseFactorizationBuilder):
    """Thresholded fixed-candidate ILUT builder with an explicit row fill cap."""

    fill_level: int = eqx.field(static=True)
    drop_tolerance: float = eqx.field(static=True)
    maximum_fill_per_row: int = eqx.field(static=True)
    ordering: SparseOrdering = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    diagonal_shift: float = eqx.field(static=True)
    allow_pivot_replacement: bool = eqx.field(static=True)
    replacement_value: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        fill_level: int = 1,
        drop_tolerance: float = 1e-4,
        maximum_fill_per_row: int = 16,
        ordering: SparseOrdering = "natural",
        pivot_tolerance: float = 0.0,
        diagonal_shift: float = 0.0,
        allow_pivot_replacement: bool = False,
        replacement_value: float = 1e-12,
    ) -> None:
        fill = int(fill_level)
        maximum_fill = int(maximum_fill_per_row)
        numeric = tuple(
            float(value)
            for value in (
                drop_tolerance,
                pivot_tolerance,
                diagonal_shift,
                replacement_value,
            )
        )
        if fill < 0 or maximum_fill < 0:
            raise ValueError("ILUT fill level and row cap must be non-negative.")
        if any(not isfinite(value) for value in numeric):
            raise ValueError("ILUT numeric policies must be finite.")
        if any(value < 0.0 for value in numeric[:3]) or numeric[3] <= 0.0:
            raise ValueError("ILUT drop, pivot, or replacement policy is invalid.")
        ordering = parse(ordering, SparseOrdering, "ordering")
        self.fill_level = fill
        self.drop_tolerance = numeric[0]
        self.maximum_fill_per_row = maximum_fill
        self.ordering = ordering
        self.pivot_tolerance = numeric[1]
        self.diagonal_shift = numeric[2]
        self.allow_pivot_replacement = bool(allow_pivot_replacement)
        self.replacement_value = numeric[3]

    def policy(self) -> SparseFactorizationPolicy:
        return SparseFactorizationPolicy(
            "lu",
            ordering=self.ordering,
            fill_level=self.fill_level,
            drop_tolerance=self.drop_tolerance,
            maximum_fill_per_row=self.maximum_fill_per_row,
            pivot_tolerance=self.pivot_tolerance,
            diagonal_shift=self.diagonal_shift,
            allow_pivot_replacement=self.allow_pivot_replacement,
            replacement_value=self.replacement_value,
        )


class IncompleteCholeskyPreconditionerBuilder(_AbstractSparseFactorizationBuilder):
    """Fixed-pattern incomplete Cholesky IC(k) builder."""

    fill_level: int = eqx.field(static=True)
    drop_tolerance: float = eqx.field(static=True)
    maximum_fill_per_row: int | None = eqx.field(static=True)
    ordering: SparseOrdering = eqx.field(static=True)
    pivot_tolerance: float = eqx.field(static=True)
    diagonal_shift: float = eqx.field(static=True)
    allow_pivot_replacement: bool = eqx.field(static=True)
    replacement_value: float = eqx.field(static=True)

    def __init__(
        self,
        fill_level: int = 0,
        /,
        *,
        drop_tolerance: float = 0.0,
        maximum_fill_per_row: int | None = None,
        ordering: SparseOrdering = "natural",
        pivot_tolerance: float = 0.0,
        diagonal_shift: float = 0.0,
        allow_pivot_replacement: bool = False,
        replacement_value: float = 1e-12,
    ) -> None:
        fill = int(fill_level)
        maximum_fill = None if maximum_fill_per_row is None else int(maximum_fill_per_row)
        numeric = tuple(
            float(value)
            for value in (
                drop_tolerance,
                pivot_tolerance,
                diagonal_shift,
                replacement_value,
            )
        )
        if fill < 0 or (maximum_fill is not None and maximum_fill < 0):
            raise ValueError("IC fill level and row cap must be non-negative.")
        if any(not isfinite(value) for value in numeric):
            raise ValueError("IC numeric policies must be finite.")
        if any(value < 0.0 for value in numeric[:3]) or numeric[3] <= 0.0:
            raise ValueError("IC drop, pivot, or replacement policy is invalid.")
        ordering = parse(ordering, SparseOrdering, "ordering")
        self.fill_level = fill
        self.drop_tolerance = numeric[0]
        self.maximum_fill_per_row = maximum_fill
        self.ordering = ordering
        self.pivot_tolerance = numeric[1]
        self.diagonal_shift = numeric[2]
        self.allow_pivot_replacement = bool(allow_pivot_replacement)
        self.replacement_value = numeric[3]

    def policy(self) -> SparseFactorizationPolicy:
        return SparseFactorizationPolicy(
            "cholesky",
            ordering=self.ordering,
            fill_level=self.fill_level,
            drop_tolerance=self.drop_tolerance,
            maximum_fill_per_row=self.maximum_fill_per_row,
            pivot_tolerance=self.pivot_tolerance,
            diagonal_shift=self.diagonal_shift,
            allow_pivot_replacement=self.allow_pivot_replacement,
            replacement_value=self.replacement_value,
        )


def refresh_incomplete_factorization(
    preconditioner: SparseFactorizationPreconditioner,
    operator: AbstractSparseLinearOperator,
    /,
) -> SparseFactorizationPreconditioner:
    """Refresh incomplete factor values while retaining the symbolic pattern."""
    if not isinstance(preconditioner, SparseFactorizationPreconditioner):
        raise TypeError("preconditioner must be SparseFactorizationPreconditioner.")
    properties = _factor_preconditioner_properties(
        operator,
        cholesky=preconditioner.factorization.plan.kind == "cholesky",
    )
    factorization = refresh_sparse_factorization(
        preconditioner.factorization.plan,
        operator,
    )
    return SparseFactorizationPreconditioner(
        operator,
        factorization,
        properties=properties,
        preconditioner_id=preconditioner.preconditioner_id,
    )


__all__ = [
    "ILUPreconditionerBuilder",
    "ILUTPreconditionerBuilder",
    "IncompleteCholeskyPreconditionerBuilder",
    "SparseFactorCongruencePreconditioner",
    "SparseFactorizationPreconditioner",
    "SparseFactorizationPreconditionerBuilder",
    "refresh_incomplete_factorization",
    "sparse_preconditioner_factorization",
]
