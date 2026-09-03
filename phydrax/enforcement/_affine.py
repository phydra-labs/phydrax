#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from ..conditions._evidence import (
    AffineProjectionCertificate,
    ConditionRealizationStamp,
)
from ..conditions._ir import (
    ArrayCodomain,
    codomains_compatible,
    ConditionCodomain,
    FieldCodomain,
    ProductCodomain,
    validate_codomain_value,
)
from ..conditions._lowering import BoundCondition
from ..conditions._relations import Equality
from ..domain import DomainFunction
from ..integration._linear import PreparedLinearReduction
from ..linalg._constraint_operators import (
    ConstraintOperatorPlan,
    PreparedConstraintOperator,
    refresh_constraint_operator,
)
from ..linalg._operators import BlockLinearOperator, FunctionLinearOperator
from ..linalg._spaces import AbstractVectorSpace, ArraySpace, BlockSpace
from ._lifecycle import (
    RealizationLifecyclePhase,
    RealizationLifecycleState,
    record_realization_stamp,
)
from ._realization import (
    AbstractFieldRealization,
    ConditionEvaluationContext,
    FieldRealizationResult,
    RealizationStatus,
)


AffineCompatibility = Literal["strict", "generalized"]
AffineExactnessScope = Literal["continuum", "realization"]


def _identifier(value: str | None, payload: Mapping[str, Any], name: str, /) -> str:
    identifier = canonical_fingerprint(dict(payload)) if value is None else str(value)
    if not identifier:
        raise ValueError(f"{name} must be nonempty.")
    return identifier


def _field_names(values: Sequence[str], /) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Correction field names must be nonempty and unique.")
    return names


def _tree_add(left: Any, right: Any, /) -> Any:
    if isinstance(left, tuple):
        if not isinstance(right, tuple) or len(left) != len(right):
            raise ValueError("Product correction layouts do not match.")
        return tuple(_tree_add(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or tuple(left) != tuple(right):
            raise ValueError("Mapping correction layouts do not match.")
        return frozendict((name, _tree_add(left[name], right[name])) for name in left)
    return left + right


def _tree_sub(left: Any, right: Any, /) -> Any:
    if isinstance(left, tuple):
        if not isinstance(right, tuple) or len(left) != len(right):
            raise ValueError("Product residual layouts do not match.")
        return tuple(_tree_sub(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or tuple(left) != tuple(right):
            raise ValueError("Mapping residual layouts do not match.")
        return frozendict((name, _tree_sub(left[name], right[name])) for name in left)
    return left - right


def _tree_zero(value: Any, /) -> Any:
    if isinstance(value, tuple):
        return tuple(_tree_zero(leaf) for leaf in value)
    if isinstance(value, Mapping):
        return frozendict((name, _tree_zero(leaf)) for name, leaf in value.items())
    if isinstance(value, DomainFunction):
        return 0.0 * value
    if isinstance(value, cx.Field):
        return cx.Field(jnp.zeros_like(value.data), dims=value.dims)
    return jnp.zeros_like(jnp.asarray(value))


def _tree_norm(value: Any, /, *, batch: Any = None, key: Any = None) -> Array | None:
    leaves: list[Array] = []

    def collect(item: Any) -> bool:
        if isinstance(item, tuple):
            return all(collect(leaf) for leaf in item)
        if isinstance(item, Mapping):
            return all(collect(leaf) for leaf in item.values())
        if isinstance(item, DomainFunction):
            if batch is None:
                return False
            array = item(batch, key=key).data
        elif isinstance(item, cx.Field):
            array = item.data
        else:
            array = jnp.asarray(item)
        leaves.append(jnp.ravel(array))
        return True

    if not collect(value):
        return None
    if not leaves:
        return jnp.asarray(0.0)
    squared = sum(jnp.real(jnp.vdot(leaf, leaf)) for leaf in leaves)
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def _codomain_space(
    codomain: ConditionCodomain, sample: Any, path: str, /
) -> AbstractVectorSpace:
    if isinstance(codomain, ArrayCodomain):
        value = validate_codomain_value(codomain, sample, path=path)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            raise TypeError(f"{path} must use a real or complex inexact dtype.")
        return ArraySpace(codomain.shape, dtype=value.dtype)
    if isinstance(codomain, ProductCodomain):
        value = validate_codomain_value(codomain, sample, path=path)
        return BlockSpace(
            tuple(
                _codomain_space(factor, leaf, f"{path}[{index}]")
                for index, (factor, leaf) in enumerate(
                    zip(codomain.factors, value, strict=True)
                )
            ),
            names=tuple(str(index) for index in range(len(codomain.factors))),
        )
    if isinstance(codomain, FieldCodomain):
        raise TypeError(
            "A FieldCodomain has no finite vector space without a realization."
        )
    raise TypeError("Expected a condition codomain.")


def _finite_codomain(codomain: ConditionCodomain, /) -> bool:
    if isinstance(codomain, ArrayCodomain):
        return True
    if isinstance(codomain, ProductCodomain):
        return all(_finite_codomain(factor) for factor in codomain.factors)
    return False


def _condition_kwargs(
    realizations: Mapping[str, Any], condition_id: str, /
) -> frozendict[str, Any]:
    if condition_id not in realizations:
        return frozendict()
    value = realizations[condition_id]
    if isinstance(value, PreparedLinearReduction):
        return frozendict(reduction=value)
    if isinstance(value, Mapping):
        return frozendict(value)
    raise TypeError(
        "A condition realization must be PreparedLinearReduction or a keyword mapping."
    )


def _realization_schema(realizations: Mapping[str, Any], /) -> tuple[Any, ...]:
    values = []
    for name, value in realizations.items():
        if isinstance(value, PreparedLinearReduction):
            token = (value.realization_id, value.schema.target_id, value.schema.exactness)
        elif isinstance(value, Mapping):
            token = tuple(sorted(str(key) for key in value))
        else:
            token = type(value).__qualname__
        values.append((str(name), token))
    return tuple(values)


def _local_values(
    bound: BoundCondition, fields: Mapping[str, Any], /
) -> frozendict[str, Any]:
    values = []
    for field in bound.condition.fields.fields:
        value = (
            fields[field.source] if field.source in fields else bound.source[field.source]
        )
        values.append(
            (
                field.name,
                validate_codomain_value(
                    field.codomain,
                    value,
                    path=f"condition {bound.condition_id!r} source {field.source!r}",
                ),
            )
        )
    return frozendict(values)


class AffineProjectionPolicy(StrictModule):
    """Compatibility and evidence policy for one prepared joint projection."""

    absolute_tolerance: Array
    relative_tolerance: Array
    compatibility: AffineCompatibility = eqx.field(static=True)
    exactness_scope: AffineExactnessScope = eqx.field(static=True)
    verify_projection: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        compatibility: AffineCompatibility = "strict",
        exactness_scope: AffineExactnessScope = "continuum",
        absolute_tolerance: Any = 1e-10,
        relative_tolerance: Any = 1e-8,
        verify_projection: bool = True,
    ):
        if compatibility not in ("strict", "generalized"):
            raise ValueError("compatibility must be 'strict' or 'generalized'.")
        if exactness_scope not in ("continuum", "realization"):
            raise ValueError("exactness_scope must be 'continuum' or 'realization'.")
        absolute = jnp.asarray(absolute_tolerance)
        relative = jnp.asarray(relative_tolerance)
        if absolute.shape or relative.shape or bool(absolute < 0) or bool(relative < 0):
            raise ValueError("Affine tolerances must be nonnegative scalars.")
        self.compatibility = compatibility
        self.exactness_scope = exactness_scope
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.verify_projection = bool(verify_projection)


class LinearCorrectionEvidence(StrictModule):
    """Preparation evidence for a reusable joint right-inverse action."""

    identity_defect: Array
    range_defect: Array
    numeric_version: Array
    solve_evidence: Any
    provider_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)
    exactness_scope: AffineExactnessScope = eqx.field(static=True)
    generalized: bool = eqx.field(static=True)
    has_adjoint: bool = eqx.field(static=True)
    has_derivative: bool = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    nullity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        provider_id: str,
        preparation_id: str,
        condition_ids: Sequence[str],
        field_names: Sequence[str],
        exactness_scope: AffineExactnessScope,
        generalized: bool,
        numeric_version: int,
        identity_defect: Any = 0.0,
        range_defect: Any = 0.0,
        rank: int = 0,
        nullity: int = 0,
        has_adjoint: bool = False,
        has_derivative: bool = False,
        solve_evidence: Any = None,
    ):
        identity = jnp.asarray(identity_defect)
        range_ = jnp.asarray(range_defect)
        version = int(numeric_version)
        rank_, nullity_ = int(rank), int(nullity)
        if identity.shape or range_.shape or version < 0 or rank_ < 0 or nullity_ < 0:
            raise ValueError("Linear correction evidence has invalid scalar metadata.")
        self.provider_id = str(provider_id)
        self.preparation_id = str(preparation_id)
        if not self.provider_id or not self.preparation_id:
            raise ValueError("Correction evidence identifiers must be nonempty.")
        self.condition_ids = tuple(str(value) for value in condition_ids)
        self.field_names = _field_names(field_names)
        self.exactness_scope = exactness_scope
        self.generalized = bool(generalized)
        self.numeric_version = jnp.asarray(version)
        self.identity_defect = identity
        self.range_defect = range_
        self.rank = rank_
        self.nullity = nullity_
        self.has_adjoint = bool(has_adjoint)
        self.has_derivative = bool(has_derivative)
        self.solve_evidence = solve_evidence


class _ConditionBlockAction(StrictModule):
    bound: BoundCondition
    kwargs: frozendict[str, Any]
    source_name: str = eqx.field(static=True)

    def __init__(self, bound, source_name, kwargs, /):
        self.bound = bound
        self.source_name = str(source_name)
        self.kwargs = frozendict(kwargs)

    def __call__(self, value, /):
        local = {}
        for field in self.bound.condition.fields.fields:
            local[field.name] = (
                value
                if field.source == self.source_name
                else _tree_zero(self.bound.values[field.name])
            )
        result = self.bound.operator.linear_action(frozendict(local), **self.kwargs)
        return validate_codomain_value(
            self.bound.codomain,
            result,
            path=f"condition {self.bound.condition_id!r} block action",
        )


class _ConditionBlockTranspose(StrictModule):
    bound: BoundCondition
    kwargs: frozendict[str, Any]
    local_name: str = eqx.field(static=True)

    def __init__(self, bound, local_name, kwargs, /):
        self.bound = bound
        self.local_name = str(local_name)
        self.kwargs = frozendict(kwargs)

    def __call__(self, value, /):
        return self.bound.adjoint_action(value, **self.kwargs)[self.local_name]


class AffineBlockAssembly(StrictModule):
    """Ordered joint condition action with an optional full finite block operator."""

    bound_conditions: tuple[BoundCondition, ...]
    correction_codomains: tuple[ConditionCodomain, ...]
    condition_codomains: tuple[ConditionCodomain, ...]
    realizations: frozendict[str, Any]
    source_space: BlockSpace | None
    target_space: BlockSpace | None
    operator: BlockLinearOperator | None
    correction_fields: tuple[str, ...] = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    exactness_scope: AffineExactnessScope = eqx.field(static=True)
    assembly_id: str = eqx.field(static=True)

    def __init__(
        self,
        bound_conditions: Sequence[BoundCondition],
        correction_fields: Sequence[str],
        /,
        *,
        realizations: Mapping[str, Any] = frozendict(),
        exactness_scope: AffineExactnessScope = "continuum",
    ):
        conditions = tuple(bound_conditions)
        names = _field_names(correction_fields)
        if not conditions or any(
            not isinstance(value, BoundCondition) for value in conditions
        ):
            raise TypeError("bound_conditions must contain BoundCondition values.")
        ids = tuple(value.condition_id for value in conditions)
        if len(set(ids)) != len(ids):
            raise ValueError("Joint affine condition identifiers must be unique.")
        for bound in conditions:
            if not isinstance(bound.relation, Equality):
                raise TypeError("Exact affine projection requires Equality relations.")
            if not bound.operator.capabilities.is_linear:
                raise TypeError(
                    "Exact affine projection requires certified linear operators."
                )
        declarations: dict[str, tuple[ConditionCodomain, Any]] = {}
        for name in names:
            matches = [
                (field.codomain, bound.source[field.source])
                for bound in conditions
                for field in bound.condition.fields.fields
                if field.source == name
            ]
            if not matches:
                raise ValueError(
                    f"Correction field {name!r} is not used by any condition."
                )
            codomain, sample = matches[0]
            if any(not codomains_compatible(codomain, other) for other, _ in matches[1:]):
                raise ValueError(
                    f"Correction field {name!r} has incompatible declarations."
                )
            declarations[name] = (codomain, sample)
        correction_codomains = tuple(declarations[name][0] for name in names)
        condition_codomains = tuple(bound.codomain for bound in conditions)
        realization_values = frozendict(realizations)
        finite = all(_finite_codomain(value) for value in correction_codomains) and all(
            _finite_codomain(value) for value in condition_codomains
        )
        if finite:
            source_space = BlockSpace(
                tuple(
                    _codomain_space(
                        codomain, declarations[name][1], f"correction {name!r}"
                    )
                    for name, codomain in zip(names, correction_codomains, strict=True)
                ),
                names=names,
            )
            target_samples = tuple(
                bound.linear_action(
                    {
                        field.name: _tree_zero(bound.values[field.name])
                        for field in bound.condition.fields.fields
                    },
                    **_condition_kwargs(realization_values, bound.condition_id),
                )
                for bound in conditions
            )
            target_space = BlockSpace(
                tuple(
                    _codomain_space(codomain, sample, f"condition {condition_id!r}")
                    for condition_id, codomain, sample in zip(
                        ids, condition_codomains, target_samples, strict=True
                    )
                ),
                names=ids,
            )
            rows = []
            for bound, target in zip(conditions, target_space.spaces, strict=True):
                kwargs = _condition_kwargs(realization_values, bound.condition_id)
                by_source = {
                    field.source: field for field in bound.condition.fields.fields
                }
                row = []
                for name, source in zip(names, source_space.spaces, strict=True):
                    if name not in by_source:
                        row.append(None)
                        continue
                    field = by_source[name]
                    transpose = (
                        _ConditionBlockTranspose(bound, field.name, kwargs)
                        if bound.operator.capabilities.has_adjoint
                        else None
                    )
                    row.append(
                        FunctionLinearOperator(
                            _ConditionBlockAction(bound, name, kwargs),
                            source=source,
                            target=target,
                            transpose_action=transpose,
                            closure_convert=False,
                            operator_id=canonical_fingerprint(
                                {
                                    "kind": "affine-condition-block",
                                    "condition": bound.condition_id,
                                    "source": name,
                                    "source_space": source.space_id,
                                    "target_space": target.space_id,
                                    "realization": _realization_schema(
                                        realization_values
                                    ),
                                }
                            ),
                        )
                    )
                rows.append(tuple(row))
            operator = BlockLinearOperator(
                tuple(rows),
                source=source_space,
                target=target_space,
                operator_id=canonical_fingerprint(
                    {
                        "kind": "affine-block-operator",
                        "conditions": ids,
                        "fields": names,
                        "source": source_space.space_id,
                        "target": target_space.space_id,
                        "realization": _realization_schema(realization_values),
                    }
                ),
            )
        else:
            source_space = target_space = operator = None
        self.bound_conditions = conditions
        self.correction_fields = names
        self.condition_ids = ids
        self.correction_codomains = correction_codomains
        self.condition_codomains = condition_codomains
        self.realizations = realization_values
        self.source_space = source_space
        self.target_space = target_space
        self.operator = operator
        self.exactness_scope = exactness_scope
        self.assembly_id = canonical_fingerprint(
            {
                "kind": "affine-block-assembly",
                "conditions": tuple(
                    (bound.bound_id, bound.condition.fields.field_spec_id)
                    for bound in conditions
                ),
                "fields": names,
                "finite": finite,
                "source_space": None if source_space is None else source_space.space_id,
                "target_space": None if target_space is None else target_space.space_id,
                "realization": _realization_schema(realization_values),
                "scope": exactness_scope,
            }
        )

    def actions(
        self,
        fields: Mapping[str, Any],
        /,
        *,
        context: ConditionEvaluationContext | None = None,
        key: Any = None,
    ) -> tuple[Any, ...]:
        del context
        values = frozendict(fields)
        return tuple(
            validate_codomain_value(
                bound.codomain,
                bound.operator.linear_action(
                    _local_values(bound, values),
                    key=key,
                    **_condition_kwargs(self.realizations, bound.condition_id),
                ),
                path=f"condition {bound.condition_id!r} action",
            )
            for bound in self.bound_conditions
        )

    def targets(
        self,
        actions: Sequence[Any],
        /,
        *,
        context: ConditionEvaluationContext | None = None,
    ) -> tuple[Any, ...]:
        del context
        values = tuple(actions)
        if len(values) != len(self.bound_conditions):
            raise ValueError("Action count does not match the joint condition count.")
        return tuple(
            validate_codomain_value(
                bound.codomain,
                bound.relation.target,
                path=f"condition {bound.condition_id!r} target",
            )
            if bound.relation.has_target
            else _tree_zero(action)
            for bound, action in zip(self.bound_conditions, values, strict=True)
        )

    def residual(
        self,
        fields: Mapping[str, Any],
        /,
        *,
        context: ConditionEvaluationContext | None = None,
        key: Any = None,
    ) -> tuple[Any, ...]:
        actions = self.actions(fields, context=context, key=key)
        targets = self.targets(actions, context=context)
        return tuple(
            _tree_sub(target, action)
            for target, action in zip(targets, actions, strict=True)
        )

    def add_correction(
        self, fields: Mapping[str, Any], corrections: Mapping[str, Any], /
    ) -> frozendict[str, Any]:
        if tuple(corrections) != self.correction_fields:
            raise ValueError("Correction output must preserve the declared field order.")
        output = dict(fields)
        for name, codomain in zip(
            self.correction_fields, self.correction_codomains, strict=True
        ):
            if name not in fields:
                raise KeyError(f"Missing correction field {name!r}.")
            update = validate_codomain_value(
                codomain, corrections[name], path=f"correction {name!r}"
            )
            output[name] = _tree_add(fields[name], update)
        return frozendict(output)


class PreparedLinearCorrection(StrictModule):
    """Prepared joint lift; numerical factorization never occurs in ``lift``."""

    lift_action: Callable
    adjoint_action: Callable | None
    derivative_action: Callable | None
    operator: PreparedConstraintOperator | None
    evidence: LinearCorrectionEvidence
    field_names: tuple[str, ...] = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)
    exactness_scope: AffineExactnessScope = eqx.field(static=True)
    generalized: bool = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)

    def __init__(
        self,
        lift_action: Callable,
        /,
        *,
        field_names: Sequence[str],
        provider_id: str,
        preparation_id: str,
        numeric_version: int,
        exactness_scope: AffineExactnessScope = "continuum",
        generalized: bool = False,
        evidence: LinearCorrectionEvidence | None = None,
        adjoint_action: Callable | None = None,
        derivative_action: Callable | None = None,
        operator: PreparedConstraintOperator | None = None,
    ):
        if not callable(lift_action):
            raise TypeError("lift_action must be callable.")
        if adjoint_action is not None and not callable(adjoint_action):
            raise TypeError("adjoint_action must be callable or None.")
        if derivative_action is not None and not callable(derivative_action):
            raise TypeError("derivative_action must be callable or None.")
        names = _field_names(field_names)
        version = int(numeric_version)
        if version < 0:
            raise ValueError("numeric_version must be nonnegative.")
        evidence_ = (
            LinearCorrectionEvidence(
                provider_id=provider_id,
                preparation_id=preparation_id,
                condition_ids=(),
                field_names=names,
                exactness_scope=exactness_scope,
                generalized=generalized,
                numeric_version=version,
                has_adjoint=adjoint_action is not None,
                has_derivative=derivative_action is not None,
            )
            if evidence is None
            else evidence
        )
        if not isinstance(evidence_, LinearCorrectionEvidence):
            raise TypeError("evidence must be LinearCorrectionEvidence or None.")
        self.lift_action = lift_action
        self.adjoint_action = adjoint_action
        self.derivative_action = derivative_action
        self.operator = operator
        self.evidence = evidence_
        self.field_names = names
        self.provider_id = str(provider_id)
        self.preparation_id = str(preparation_id)
        self.exactness_scope = exactness_scope
        self.generalized = bool(generalized)
        self.numeric_version = version

    def lift(self, product_residual: tuple[Any, ...], /) -> Any:
        return self.lift_action(product_residual)

    def corrections(self, product_residual: tuple[Any, ...], /) -> frozendict[str, Any]:
        value = self.lift(product_residual)
        if isinstance(value, Mapping):
            if tuple(value) != self.field_names:
                raise ValueError(
                    "A correction mapping must preserve declared field order."
                )
            return frozendict(value)
        if not isinstance(value, tuple) or len(value) != len(self.field_names):
            raise ValueError(
                "A correction lift must return one outer product block per field."
            )
        return frozendict(zip(self.field_names, value, strict=True))

    def adjoint(self, correction: Any, /) -> Any:
        if self.adjoint_action is None:
            raise TypeError("This prepared correction has no certified adjoint action.")
        return self.adjoint_action(correction)

    def derivative(self, *args: Any, **kwargs: Any) -> Any:
        if self.derivative_action is None:
            raise TypeError(
                "This prepared correction has no certified derivative action."
            )
        return self.derivative_action(*args, **kwargs)


class AbstractLinearCorrectionProvider(StrictModule):
    """Preparation contract for a joint linear correction family."""

    @abstractmethod
    def prepare(
        self,
        bound_conditions: Sequence[BoundCondition],
        assembly: AffineBlockAssembly,
        /,
        *,
        correction_fields: Sequence[str],
        realizations: Mapping[str, Any],
        policy: AffineProjectionPolicy,
        numeric_version: int,
    ) -> PreparedLinearCorrection:
        raise NotImplementedError

    @abstractmethod
    def refresh(
        self,
        previous: PreparedLinearCorrection,
        bound_conditions: Sequence[BoundCondition],
        assembly: AffineBlockAssembly,
        /,
        *,
        correction_fields: Sequence[str],
        realizations: Mapping[str, Any],
        policy: AffineProjectionPolicy,
        numeric_version: int,
    ) -> PreparedLinearCorrection:
        del previous
        return self.prepare(
            bound_conditions,
            assembly,
            correction_fields=correction_fields,
            realizations=realizations,
            policy=policy,
            numeric_version=numeric_version,
        )


class _ConstraintLift(StrictModule):
    operator: PreparedConstraintOperator
    strict: bool = eqx.field(static=True)

    def __init__(self, operator, strict, /):
        self.operator, self.strict = operator, bool(strict)

    def __call__(self, residual, /):
        if self.strict:
            return self.operator.strict_right_inverse(residual)
        return self.operator.minimum_norm_lift(residual, check_compatibility=True)


class _ConstraintAdjoint(StrictModule):
    operator: PreparedConstraintOperator

    def __call__(self, value, /):
        return self.operator.right_inverse_adjoint(value)


class ConstraintLinearCorrectionProvider(AbstractLinearCorrectionProvider):
    """Default finite-dimensional provider backed by one prepared constraint operator."""

    plan: ConstraintOperatorPlan | None
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConstraintOperatorPlan | None = None,
        /,
        *,
        provider_id: str | None = None,
    ):
        if plan is not None and not isinstance(plan, ConstraintOperatorPlan):
            raise TypeError("plan must be a ConstraintOperatorPlan or None.")
        self.plan = plan
        self.provider_id = _identifier(
            provider_id,
            {
                "kind": "constraint-linear-correction",
                "plan": None if plan is None else plan.plan_id,
            },
            "provider_id",
        )

    def _bind(
        self,
        prepared: PreparedConstraintOperator,
        assembly: AffineBlockAssembly,
        policy: AffineProjectionPolicy,
        numeric_version: int,
        /,
    ) -> PreparedLinearCorrection:
        generalized = policy.compatibility == "generalized"
        evidence = LinearCorrectionEvidence(
            provider_id=self.provider_id,
            preparation_id=prepared.prepared_id,
            condition_ids=assembly.condition_ids,
            field_names=assembly.correction_fields,
            exactness_scope=policy.exactness_scope,
            generalized=generalized,
            numeric_version=numeric_version,
            identity_defect=prepared.evidence.strict_right_inverse_residual_norm,
            range_defect=prepared.evidence.generalized_right_inverse_residual_norm,
            rank=prepared.rank,
            nullity=prepared.nullity,
            has_adjoint=True,
            solve_evidence=prepared.evidence,
        )
        return PreparedLinearCorrection(
            _ConstraintLift(prepared, not generalized),
            field_names=assembly.correction_fields,
            provider_id=self.provider_id,
            preparation_id=prepared.prepared_id,
            numeric_version=numeric_version,
            exactness_scope=policy.exactness_scope,
            generalized=generalized,
            evidence=evidence,
            adjoint_action=_ConstraintAdjoint(prepared),
            operator=prepared,
        )

    def prepare(
        self,
        bound_conditions,
        assembly,
        /,
        *,
        correction_fields,
        realizations,
        policy,
        numeric_version,
    ) -> PreparedLinearCorrection:
        del bound_conditions, correction_fields, realizations
        if assembly.operator is None:
            raise TypeError(
                "ConstraintLinearCorrectionProvider requires a finite block assembly."
            )
        plan = (
            ConstraintOperatorPlan(
                assembly.operator,
                require_full_row_rank=policy.compatibility == "strict",
            )
            if self.plan is None
            else self.plan
        )
        if not plan.operator.source.compatible(
            assembly.operator.source
        ) or not plan.operator.target.compatible(assembly.operator.target):
            raise ValueError(
                "The supplied constraint plan has incompatible block spaces."
            )
        return self._bind(plan.prepare(), assembly, policy, int(numeric_version))

    def refresh(
        self,
        previous,
        bound_conditions,
        assembly,
        /,
        *,
        correction_fields,
        realizations,
        policy,
        numeric_version,
    ) -> PreparedLinearCorrection:
        del bound_conditions, correction_fields, realizations
        if previous.operator is None or assembly.operator is None:
            raise TypeError(
                "Finite constraint refresh requires prepared and assembled operators."
            )
        refreshed = refresh_constraint_operator(previous.operator, assembly.operator)
        return self._bind(refreshed, assembly, policy, int(numeric_version))


class AffineProjectorEvidence(StrictModule):
    """Structural and numerical evidence retained by a prepared projector."""

    correction: LinearCorrectionEvidence
    condition_evidence: tuple[Any, ...]
    projector_id: str = eqx.field(static=True)
    assembly_id: str = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    correction_fields: tuple[str, ...] = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)


class PreparedAffineProjector(AbstractFieldRealization):
    """Immutable, jointly prepared affine projection over all condition rows."""

    assembly: AffineBlockAssembly
    correction: PreparedLinearCorrection
    provider: AbstractLinearCorrectionProvider
    policy: AffineProjectionPolicy
    evidence: AffineProjectorEvidence
    prepared_id: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)

    def __init__(self, assembly, correction, provider, policy, /, *, numeric_version):
        if not isinstance(assembly, AffineBlockAssembly):
            raise TypeError("assembly must be an AffineBlockAssembly.")
        if not isinstance(correction, PreparedLinearCorrection):
            raise TypeError("correction must be PreparedLinearCorrection.")
        if not isinstance(provider, AbstractLinearCorrectionProvider):
            raise TypeError("provider must be AbstractLinearCorrectionProvider.")
        if correction.field_names != assembly.correction_fields:
            raise ValueError("Prepared correction fields do not match the assembly.")
        if correction.exactness_scope != policy.exactness_scope:
            raise ValueError("Prepared correction exactness scope differs from policy.")
        if correction.generalized != (policy.compatibility == "generalized"):
            raise ValueError("Prepared correction compatibility differs from policy.")
        version = int(numeric_version)
        if correction.numeric_version != version:
            raise ValueError(
                "Prepared correction numeric version differs from the projector."
            )
        tolerance = policy.absolute_tolerance + policy.relative_tolerance
        defect = (
            correction.evidence.range_defect
            if correction.generalized
            else correction.evidence.identity_defect
        )
        if bool(defect > tolerance):
            raise ValueError(
                "Prepared correction does not meet the affine compatibility tolerance."
            )
        self.assembly = assembly
        self.correction = correction
        self.provider = provider
        self.policy = policy
        self.numeric_version = version
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-affine-projector",
                "assembly": assembly.assembly_id,
                "provider": correction.provider_id,
                "preparation": correction.preparation_id,
                "version": version,
            }
        )
        self.evidence = AffineProjectorEvidence(
            correction=correction.evidence,
            condition_evidence=tuple(
                bound.evidence for bound in assembly.bound_conditions
            ),
            projector_id=self.prepared_id,
            assembly_id=assembly.assembly_id,
            condition_ids=assembly.condition_ids,
            correction_fields=assembly.correction_fields,
            numeric_version=version,
        )

    def apply(
        self,
        fields: Mapping[str, Any],
        /,
        *,
        context: ConditionEvaluationContext | None = None,
        key: Any = None,
    ) -> frozendict[str, Any]:
        residual = self.assembly.residual(fields, context=context, key=key)
        corrections = self.correction.corrections(residual)
        return self.assembly.add_correction(fields, corrections)

    def constraint_defect(
        self,
        fields: Mapping[str, Any],
        /,
        *,
        context: ConditionEvaluationContext | None = None,
        batch: Any = None,
        key: Any = None,
    ) -> Array | None:
        residual = self.assembly.residual(fields, context=context, key=key)
        return _tree_norm(residual, batch=batch, key=key)

    def realize(
        self,
        fields: Mapping[str, Any],
        state: RealizationLifecycleState | None = None,
        *,
        context: ConditionEvaluationContext,
    ) -> FieldRealizationResult:
        current = RealizationLifecycleState.initial() if state is None else state
        if context.condition_id not in self.assembly.condition_ids:
            return FieldRealizationResult.failure(
                RealizationStatus.INVALID_INPUT,
                state=current,
                message="Evaluation context condition is not a member of this joint projector.",
                evidence=self.evidence,
            )
        projected = self.apply(fields, context=context, key=context.prng_key)
        defect = self.constraint_defect(projected, context=context, key=context.prng_key)
        residual_norm = jnp.asarray(0.0) if defect is None else defect
        tolerance = self.policy.absolute_tolerance + self.policy.relative_tolerance
        verified = jnp.asarray(True) if defect is None else residual_norm <= tolerance
        source_id = canonical_fingerprint(
            {
                "kind": "affine-projection-source",
                "assembly": self.assembly.assembly_id,
                "generation": current.generation,
                "parameter_revision": context.parameter_revision,
            }
        )
        stamp = ConditionRealizationStamp(
            context.condition_id,
            source_id,
            self.prepared_id,
            self.correction.provider_id,
            quantifier=context.quantifier,
            exact=True,
        )
        certificate = AffineProjectionCertificate(
            stamp,
            residual_norm,
            tolerance,
            verified,
            certificate_id=canonical_fingerprint(
                {
                    "kind": "affine-projection-certificate",
                    "projector": self.prepared_id,
                    "condition": context.condition_id,
                    "source": source_id,
                }
            ),
            rank=self.correction.evidence.rank,
            nullity=self.correction.evidence.nullity,
        )
        if self.policy.verify_projection and not bool(np.asarray(verified)):
            return FieldRealizationResult.failure(
                RealizationStatus.VALIDATION_FAILED,
                state=current,
                message="Prepared affine projection failed its residual tolerance.",
                evidence=certificate,
            )
        coordinates_changed = (
            current.accepted_step != context.accepted_step
            or current.parameter_revision != context.parameter_revision
        )
        ready = RealizationLifecycleState(
            phase=RealizationLifecyclePhase.READY,
            generation=current.generation + int(coordinates_changed),
            accepted_step=context.accepted_step,
            parameter_revision=context.parameter_revision,
            values=current.values,
            source_stamps=current.source_stamps,
            realization_stamp=current.realization_stamp,
        )
        committed = record_realization_stamp(ready, stamp)
        return FieldRealizationResult.success(
            projected,
            state=committed,
            stamp=stamp,
            evidence=certificate,
        )


class ExactAffineProjector(AbstractFieldRealization):
    """Public exact realization backed exclusively by explicitly prepared state."""

    prepared: PreparedAffineProjector

    def __init__(self, prepared: PreparedAffineProjector, /):
        if not isinstance(prepared, PreparedAffineProjector):
            raise TypeError("ExactAffineProjector requires a PreparedAffineProjector.")
        self.prepared = prepared

    @property
    def evidence(self) -> AffineProjectorEvidence:
        return self.prepared.evidence

    def apply(self, fields, /, **kwargs):
        return self.prepared.apply(fields, **kwargs)

    def constraint_defect(self, fields, /, **kwargs):
        return self.prepared.constraint_defect(fields, **kwargs)

    def realize(self, fields, state=None, *, context):
        return self.prepared.realize(fields, state, context=context)


def prepare_affine_projector(
    bound_conditions: Sequence[BoundCondition],
    provider: AbstractLinearCorrectionProvider,
    /,
    *,
    correction_fields: Sequence[str],
    realizations: Mapping[str, Any] = frozendict(),
    policy: AffineProjectionPolicy | None = None,
    numeric_version: int = 0,
) -> PreparedAffineProjector:
    """Assemble and factor one joint projector before any field query."""
    if not isinstance(provider, AbstractLinearCorrectionProvider):
        raise TypeError("provider must be an AbstractLinearCorrectionProvider.")
    policy_ = AffineProjectionPolicy() if policy is None else policy
    if not isinstance(policy_, AffineProjectionPolicy):
        raise TypeError("policy must be AffineProjectionPolicy or None.")
    version = int(numeric_version)
    if version < 0:
        raise ValueError("numeric_version must be nonnegative.")
    conditions = tuple(bound_conditions)
    names = _field_names(correction_fields)
    assembly = AffineBlockAssembly(
        conditions,
        names,
        realizations=realizations,
        exactness_scope=policy_.exactness_scope,
    )
    correction = provider.prepare(
        conditions,
        assembly,
        correction_fields=names,
        realizations=frozendict(realizations),
        policy=policy_,
        numeric_version=version,
    )
    return PreparedAffineProjector(
        assembly, correction, provider, policy_, numeric_version=version
    )


def refresh_affine_projector(
    prepared: PreparedAffineProjector,
    bound_conditions: Sequence[BoundCondition],
    /,
    *,
    realizations: Mapping[str, Any] | None = None,
    numeric_version: int,
) -> PreparedAffineProjector:
    """Refresh numerical state while preserving the complete joint structure."""
    if not isinstance(prepared, PreparedAffineProjector):
        raise TypeError("prepared must be a PreparedAffineProjector.")
    version = int(numeric_version)
    if version <= prepared.numeric_version:
        raise ValueError("Affine refresh requires a strictly newer numeric_version.")
    conditions = tuple(bound_conditions)
    realization_values = (
        prepared.assembly.realizations
        if realizations is None
        else frozendict(realizations)
    )
    assembly = AffineBlockAssembly(
        conditions,
        prepared.assembly.correction_fields,
        realizations=realization_values,
        exactness_scope=prepared.policy.exactness_scope,
    )
    structural = (
        assembly.condition_ids == prepared.assembly.condition_ids
        and assembly.correction_fields == prepared.assembly.correction_fields
        and len(assembly.condition_codomains)
        == len(prepared.assembly.condition_codomains)
        and all(
            codomains_compatible(left, right)
            for left, right in zip(
                assembly.condition_codomains,
                prepared.assembly.condition_codomains,
                strict=True,
            )
        )
        and all(
            codomains_compatible(left, right)
            for left, right in zip(
                assembly.correction_codomains,
                prepared.assembly.correction_codomains,
                strict=True,
            )
        )
    )
    if not structural:
        raise ValueError(
            "Affine refresh must preserve conditions, fields, and codomains."
        )
    correction = prepared.provider.refresh(
        prepared.correction,
        conditions,
        assembly,
        correction_fields=assembly.correction_fields,
        realizations=realization_values,
        policy=prepared.policy,
        numeric_version=version,
    )
    if correction.evidence.rank != prepared.correction.evidence.rank:
        raise ValueError(
            "Affine refresh changed numerical rank; prepare a new structure."
        )
    return PreparedAffineProjector(
        assembly,
        correction,
        prepared.provider,
        prepared.policy,
        numeric_version=version,
    )


__all__ = [
    "AbstractLinearCorrectionProvider",
    "AffineBlockAssembly",
    "AffineProjectionPolicy",
    "AffineProjectorEvidence",
    "ConstraintLinearCorrectionProvider",
    "ExactAffineProjector",
    "LinearCorrectionEvidence",
    "PreparedAffineProjector",
    "PreparedLinearCorrection",
    "prepare_affine_projector",
    "refresh_affine_projector",
]
