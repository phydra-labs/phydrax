#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe

from .._fingerprint import canonical_fingerprint
from .._frozendict import frozendict
from .._strict import StrictModule
from ..domain import Domain, DomainFunction
from ..domain._derivative import (
    DerivativeBackend,
    DerivativeBasis,
    DerivativeMode,
    DerivativeRule,
)
from ..domain._evaluation import BatchEvaluator
from ..linalg._constraint_operators import PreparedConstraintOperator


FiberExactnessScope = Literal["continuum", "realization"]
FiberDerivativeAction = Callable[..., DomainFunction | None]


def _id(value: str | None, payload: dict[str, Any], /) -> str:
    if value is None:
        return canonical_fingerprint(payload)
    identifier = str(value)
    if not identifier:
        raise ValueError("Fiber projection identifiers must be nonempty.")
    return identifier


def _names(values: Sequence[str], /) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Fiber field names must be nonempty and unique.")
    return names


def _scope(value: str, /) -> FiberExactnessScope:
    if value not in ("continuum", "realization"):
        raise ValueError("Exactness scope must be 'continuum' or 'realization'.")
    return value


def _checked(updates: Mapping[str, Any], names: tuple[str, ...], /):
    if not isinstance(updates, Mapping) or tuple(updates) != names:
        raise ValueError("A fiber lift must return its declared ordered field mapping.")
    return frozendict(updates)


def _add(fields: Mapping[str, Any], updates: Mapping[str, Any], /):
    def add_value(left: Any, right: Any, /) -> Any:
        if isinstance(left, tuple):
            if not isinstance(right, tuple) or len(left) != len(right):
                raise ValueError("Fiber product correction layouts do not match.")
            return tuple(add_value(a, b) for a, b in zip(left, right, strict=True))
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or tuple(left) != tuple(right):
                raise ValueError("Fiber mapping correction layouts do not match.")
            return frozendict((name, add_value(left[name], right[name])) for name in left)
        return left + right

    out = dict(fields)
    for name, update in updates.items():
        if name not in fields:
            raise KeyError(f"Unknown correction field {name!r}.")
        out[name] = add_value(fields[name], update)
    return frozendict(out)


def _sub(target: Any, value: Any, /) -> Any:
    if isinstance(target, tuple):
        if not isinstance(value, tuple) or len(target) != len(value):
            raise ValueError("Fiber product layouts do not match.")
        return tuple(_sub(a, b) for a, b in zip(target, value, strict=True))
    if isinstance(target, Mapping):
        if not isinstance(value, Mapping) or tuple(target) != tuple(value):
            raise ValueError("Fiber mapping layouts do not match.")
        return frozendict((name, _sub(target[name], value[name])) for name in target)
    return target - value


def _negate(value: Any, /) -> Any:
    if isinstance(value, tuple):
        return tuple(_negate(block) for block in value)
    if isinstance(value, Mapping):
        return frozendict((name, _negate(block)) for name, block in value.items())
    return -value


def _fiber_product_leaves(value: Any, /) -> tuple[cx.Field, ...]:
    if isinstance(value, cx.Field):
        return (value,)
    if isinstance(value, tuple):
        leaves = tuple(leaf for block in value for leaf in _fiber_product_leaves(block))
    elif isinstance(value, Mapping):
        leaves = tuple(
            leaf for block in value.values() for leaf in _fiber_product_leaves(block)
        )
    else:
        raise TypeError(
            "A realized fiber residual must be a coordax.Field or an ordered product."
        )
    if not leaves:
        raise ValueError("A fiber residual product must contain at least one field.")
    return leaves


def _pack_fiber_product(value: Any, fiber_dims: tuple[str, ...], /) -> cx.Field:
    leaves = _fiber_product_leaves(value)
    axis = len(fiber_dims)
    fiber_shape = tuple(int(size) for size in leaves[0].data.shape[:axis])
    blocks = []
    for leaf in leaves:
        if leaf.dims[:axis] != fiber_dims or any(
            dim is not None for dim in leaf.dims[axis:]
        ):
            raise ValueError(
                "Every product factor must share the prepared leading fiber axes."
            )
        if tuple(int(size) for size in leaf.data.shape[:axis]) != fiber_shape:
            raise ValueError("Fiber product factors have different named-axis sizes.")
        event_shape = tuple(int(size) for size in leaf.data.shape[axis:])
        event_size = prod(event_shape) if event_shape else 1
        blocks.append(jnp.asarray(leaf.data).reshape(fiber_shape + (event_size,)))
    data = blocks[0] if len(blocks) == 1 else jnp.concatenate(tuple(blocks), axis=-1)
    return cx.Field(data, dims=fiber_dims + (None,))


def _same_product_layout(left: Any, right: Any, /) -> bool:
    if isinstance(left, cx.Field):
        return (
            isinstance(right, cx.Field)
            and left.dims == right.dims
            and left.data.shape == right.data.shape
        )
    if isinstance(left, tuple):
        return (
            isinstance(right, tuple)
            and len(left) == len(right)
            and all(_same_product_layout(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and tuple(left) == tuple(right)
            and all(_same_product_layout(left[name], right[name]) for name in left)
        )
    return False


def _factor_layout(value: cx.Field, name: str, /):
    if not isinstance(value, cx.Field) or value.data.ndim < 2:
        raise TypeError(f"{name} must be a matrix-valued coordax.Field.")
    if value.dims[-2:] != (None, None) or any(dim is None for dim in value.dims[:-2]):
        raise ValueError(
            f"{name} must have named fiber axes and two unnamed matrix axes."
        )
    if len(set(value.dims[:-2])) != len(value.dims[:-2]):
        raise ValueError("Fiber named axes must be unique.")
    return (
        tuple(value.data.shape[:-2]),
        int(value.data.shape[-2]),
        int(value.data.shape[-1]),
    )


def _local_mv(matrix: cx.Field, residual: cx.Field, /) -> cx.Field:
    if not isinstance(residual, cx.Field):
        raise TypeError("A fiber residual must be a coordax.Field.")
    fiber_dims = matrix.dims[:-2]
    axis = len(fiber_dims)
    if residual.dims[:axis] != fiber_dims or residual.dims[axis] is not None:
        raise ValueError("Residual axes do not match the prepared fiber factor.")
    if tuple(residual.data.shape[:axis]) != tuple(matrix.data.shape[:-2]):
        raise ValueError("Residual fiber sizes do not match the prepared factor.")
    k, m = (int(size) for size in matrix.data.shape[-2:])
    if int(residual.data.shape[axis]) != m or any(
        dim is not None for dim in residual.dims[axis + 1 :]
    ):
        raise ValueError("Residual event layout is incompatible with the factor.")
    fiber_shape = tuple(int(size) for size in residual.data.shape[:axis])
    rhs_shape = tuple(int(size) for size in residual.data.shape[axis + 1 :])
    b = prod(fiber_shape) if fiber_shape else 1
    r = prod(rhs_shape) if rhs_shape else 1
    matrices = jnp.asarray(matrix.data).reshape((b, k, m))
    rhs = jnp.asarray(residual.data).reshape((b, m, r))
    result = oe.contract("bkm,bmr->bkr", matrices, rhs)
    return cx.Field(
        result.reshape(fiber_shape + (k,) + rhs_shape),
        dims=fiber_dims + (None,) * (1 + len(rhs_shape)),
    )


def _shared_mv(matrix: Any, residual: cx.Field, /) -> cx.Field:
    matrix_ = jnp.asarray(matrix)
    if matrix_.ndim != 2 or not isinstance(residual, cx.Field):
        raise TypeError("A separable lift requires one matrix and one coordax.Field.")
    named = sum(dim is not None for dim in residual.dims)
    if any(dim is None for dim in residual.dims[:named]) or any(
        dim is not None for dim in residual.dims[named:]
    ):
        raise ValueError("Named residual axes must precede unnamed event axes.")
    k, m = (int(size) for size in matrix_.shape)
    if int(residual.data.shape[named]) != m:
        raise ValueError("Separable residual dimension does not match the factor.")
    fiber_shape = tuple(int(size) for size in residual.data.shape[:named])
    rhs_shape = tuple(int(size) for size in residual.data.shape[named + 1 :])
    b = prod(fiber_shape) if fiber_shape else 1
    r = prod(rhs_shape) if rhs_shape else 1
    rhs = jnp.asarray(residual.data).reshape((b, m, r))
    result = oe.contract("km,bmr->bkr", matrix_, rhs)
    return cx.Field(
        result.reshape(fiber_shape + (k,) + rhs_shape),
        dims=residual.dims[:named] + (None,) * (1 + len(rhs_shape)),
    )


def _shared_lift(operator: PreparedConstraintOperator, residual: Any, /) -> cx.Field:
    first = _fiber_product_leaves(residual)[0]
    named = 0
    for dim in first.dims:
        if dim is None:
            break
        named += 1
    fiber_dims = tuple(str(dim) for dim in first.dims[:named])
    packed = (
        residual
        if isinstance(residual, cx.Field)
        else _pack_fiber_product(residual, fiber_dims)
    )
    correction = _shared_mv(operator.right_inverse, packed)
    if operator.evidence.full_row_rank:
        return correction
    fiber_shape = tuple(int(size) for size in packed.data.shape[:named])
    rhs_shape = tuple(int(size) for size in packed.data.shape[named + 1 :])
    count = prod(fiber_shape + rhs_shape) if fiber_shape or rhs_shape else 1
    targets = jnp.moveaxis(packed.data, named, -1).reshape(
        (count, operator.target_space.size)
    )
    compatible = jax.vmap(
        lambda value: operator.is_compatible(operator.target_space.unflatten(value))
    )(targets)
    return cx.Field(
        eqx.error_if(
            correction.data,
            ~jnp.all(compatible),
            "Separable fiber residual is incompatible with the prepared range.",
        ),
        dims=correction.dims,
    )


class BatchedFiberFactor(StrictModule):
    """Stored local right inverses over named residual-domain axes."""

    right_inverse: cx.Field
    constraint: cx.Field | None
    compatibility_tolerance: Any
    evidence: Any
    fiber_dims: tuple[str, ...] = eqx.field(static=True)
    correction_size: int = eqx.field(static=True)
    target_size: int = eqx.field(static=True)
    generalized: bool = eqx.field(static=True)
    factor_id: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)

    def __init__(
        self,
        right_inverse: cx.Field,
        /,
        *,
        constraint: cx.Field | None = None,
        evidence: Any = None,
        generalized: bool = False,
        compatibility_tolerance: Any = 1e-8,
        factor_id: str | None = None,
        numeric_version: int = 0,
    ):
        shape, k, m = _factor_layout(right_inverse, "right_inverse")
        if constraint is not None:
            other_shape, rows, columns = _factor_layout(constraint, "constraint")
            if (
                other_shape != shape
                or rows != m
                or columns != k
                or constraint.dims[:-2] != right_inverse.dims[:-2]
            ):
                raise ValueError("Constraint and right-inverse fiber layouts conflict.")
        version = int(numeric_version)
        tolerance = jnp.asarray(compatibility_tolerance)
        if version < 0:
            raise ValueError("numeric_version must be nonnegative.")
        if tolerance.shape or bool(tolerance < 0):
            raise ValueError("compatibility_tolerance must be a nonnegative scalar.")
        self.right_inverse = right_inverse
        self.constraint = constraint
        self.compatibility_tolerance = tolerance
        self.evidence = evidence
        self.fiber_dims = tuple(str(dim) for dim in right_inverse.dims[:-2])
        self.correction_size = k
        self.target_size = m
        self.generalized = bool(generalized)
        self.numeric_version = version
        self.factor_id = _id(
            factor_id,
            {
                "kind": "fiber-factor",
                "dims": self.fiber_dims,
                "shape": shape,
                "k": k,
                "m": m,
                "generalized": generalized,
                "version": version,
            },
        )

    def apply(self, residual: Any, /) -> cx.Field:
        """Apply stored factors without a query-time factorization."""
        packed = (
            residual
            if isinstance(residual, cx.Field)
            else _pack_fiber_product(residual, self.fiber_dims)
        )
        correction = _local_mv(self.right_inverse, packed)
        if self.constraint is None:
            return correction
        projected = _local_mv(self.constraint, correction)
        defect = jnp.max(jnp.abs(projected.data - packed.data))
        scale = jnp.maximum(jnp.max(jnp.abs(packed.data)), 1.0)
        incompatible = defect > self.compatibility_tolerance * scale
        return cx.Field(
            eqx.error_if(
                correction.data,
                incompatible,
                "Fiber residual is incompatible with the prepared joint range.",
            ),
            dims=correction.dims,
        )

    def right_inverse_defect(self, /):
        if self.constraint is None or self.generalized:
            return None
        product = oe.contract(
            "...mk,...kn->...mn", self.constraint.data, self.right_inverse.data
        )
        return jnp.max(jnp.abs(product - jnp.eye(self.target_size, dtype=product.dtype)))

    def generalized_inverse_defect(self, /):
        if self.constraint is None:
            return None
        matrix = self.constraint.data
        reconstructed = oe.contract(
            "...mk,...kn,...nl->...ml", matrix, self.right_inverse.data, matrix
        )
        return jnp.max(jnp.abs(reconstructed - matrix))


@dataclass(frozen=True, slots=True, eq=False)
class FiberProjectionDerivativeRule(DerivativeRule):
    """A provider derivative including fiber-factor product terms."""

    action: FiberDerivativeAction
    field_name: str

    def derive(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: DerivativeMode,
        backend: DerivativeBackend,
        basis: DerivativeBasis,
        periodic: bool,
    ) -> DomainFunction | None:
        return self.action(
            self.field_name,
            var=var,
            axis=axis,
            order=int(order),
            mode=mode,
            backend=backend,
            basis=basis,
            periodic=bool(periodic),
        )


class AnalyticFiberProjectionUnit(StrictModule):
    """A continuum fiber action, target, and right-inverse lift."""

    action: Callable
    target: Callable
    lift: Callable
    derivative_action: FiberDerivativeAction | None
    residual_domain: Domain
    evidence: Any
    field_names: tuple[str, ...] = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    exactness_scope: FiberExactnessScope = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)

    def __init__(
        self,
        action: Callable,
        target: Callable,
        lift: Callable,
        residual_domain: Domain,
        /,
        *,
        field_names: Sequence[str],
        condition_ids: Sequence[str] = (),
        derivative_action: FiberDerivativeAction | None = None,
        evidence: Any = None,
        exactness_scope: FiberExactnessScope = "continuum",
        unit_id: str | None = None,
        numeric_version: int = 0,
    ):
        if not callable(action) or not callable(target) or not callable(lift):
            raise TypeError("Analytic fiber action, target, and lift must be callable.")
        if not isinstance(residual_domain, Domain):
            raise TypeError("residual_domain must be a Domain.")
        names, ids, version = (
            _names(field_names),
            tuple(str(value) for value in condition_ids),
            int(numeric_version),
        )
        if version < 0 or any(not value for value in ids):
            raise ValueError("Condition IDs and numeric version are invalid.")
        self.action, self.target, self.lift = action, target, lift
        self.derivative_action, self.residual_domain, self.evidence = (
            derivative_action,
            residual_domain,
            evidence,
        )
        (
            self.field_names,
            self.condition_ids,
            self.exactness_scope,
            self.numeric_version,
        ) = names, ids, _scope(exactness_scope), version
        self.unit_id = _id(
            unit_id,
            {
                "kind": "analytic-fiber",
                "fields": names,
                "conditions": ids,
                "domain": residual_domain.labels,
                "version": version,
            },
        )

    def corrections(self, fields: Mapping[str, Any], context: Any, /):
        residual = _sub(self.target(fields, context), self.action(fields, context))
        return _checked(self.lift(residual, context), self.field_names)

    def homogeneous_corrections(self, fields: Mapping[str, Any], context: Any, /):
        return _checked(
            self.lift(_negate(self.action(fields, context)), context), self.field_names
        )


class RealizedFiberProjectionUnit(StrictModule):
    """A fiber unit exact on a fixed residual grid or basis realization."""

    action: Callable
    target: Callable
    synthesis: Callable
    factor: BatchedFiberFactor
    evidence: Any
    field_names: tuple[str, ...] = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    exactness_scope: FiberExactnessScope = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Callable,
        target: Callable,
        synthesis: Callable,
        factor: BatchedFiberFactor,
        /,
        *,
        field_names: Sequence[str],
        condition_ids: Sequence[str] = (),
        evidence: Any = None,
        unit_id: str | None = None,
    ):
        if (
            not callable(action)
            or not callable(target)
            or not callable(synthesis)
            or not isinstance(factor, BatchedFiberFactor)
        ):
            raise TypeError(
                "Realized fiber callables and BatchedFiberFactor are required."
            )
        names, ids = _names(field_names), tuple(str(value) for value in condition_ids)
        self.action, self.target, self.synthesis, self.factor = (
            action,
            target,
            synthesis,
            factor,
        )
        self.evidence, self.field_names, self.condition_ids, self.exactness_scope = (
            evidence,
            names,
            ids,
            "realization",
        )
        self.unit_id = _id(
            unit_id,
            {
                "kind": "realized-fiber",
                "fields": names,
                "conditions": ids,
                "factor": factor.factor_id,
            },
        )

    def _residual(self, fields, batch, context, key, /):
        target = self.target(fields, batch, context, key)
        action = self.action(fields, batch, context, key)
        if not _same_product_layout(target, action):
            raise ValueError(
                "Realized fiber target and action must have identical product layouts."
            )
        return _sub(target, action)

    def corrections(self, fields, batch, context, key=None, /):
        coefficients = self.factor.apply(self._residual(fields, batch, context, key))
        return _checked(
            self.synthesis(coefficients, fields, batch, context, key),
            self.field_names,
        )

    def homogeneous_corrections(self, fields, batch, context, key=None, /):
        action = self.action(fields, batch, context, key)
        _fiber_product_leaves(action)
        return _checked(
            self.synthesis(
                self.factor.apply(_negate(action)), fields, batch, context, key
            ),
            self.field_names,
        )


class SeparableFiberProjectionUnit(StrictModule):
    """An axiswise fiber unit sharing one prepared reduced factorization."""

    action: Callable
    target: Callable
    synthesis: Callable
    operator: PreparedConstraintOperator
    reduction: Any
    evidence: Any
    field_names: tuple[str, ...] = eqx.field(static=True)
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    exactness_scope: FiberExactnessScope = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Callable,
        target: Callable,
        synthesis: Callable,
        operator: PreparedConstraintOperator,
        /,
        *,
        field_names: Sequence[str],
        condition_ids: Sequence[str] = (),
        reduction: Any = None,
        evidence: Any = None,
        exactness_scope: FiberExactnessScope = "realization",
        unit_id: str | None = None,
    ):
        if (
            not callable(action)
            or not callable(target)
            or not callable(synthesis)
            or not isinstance(operator, PreparedConstraintOperator)
        ):
            raise TypeError(
                "Separable fiber callables and PreparedConstraintOperator are required."
            )
        names, ids = _names(field_names), tuple(str(value) for value in condition_ids)
        self.action, self.target, self.synthesis, self.operator = (
            action,
            target,
            synthesis,
            operator,
        )
        (
            self.reduction,
            self.evidence,
            self.field_names,
            self.condition_ids,
            self.exactness_scope,
        ) = reduction, evidence, names, ids, _scope(exactness_scope)
        self.unit_id = _id(
            unit_id,
            {
                "kind": "separable-fiber",
                "fields": names,
                "conditions": ids,
                "operator": operator.prepared_id,
            },
        )

    def _residual(self, fields, batch, context, key, /):
        target = self.target(fields, batch, context, key)
        action = self.action(fields, batch, context, key)
        if not _same_product_layout(target, action):
            raise ValueError(
                "Separable fiber target and action must have identical product layouts."
            )
        return _sub(target, action)

    def corrections(self, fields, batch, context, key=None, /):
        coefficients = _shared_lift(
            self.operator, self._residual(fields, batch, context, key)
        )
        return _checked(
            self.synthesis(coefficients, fields, batch, context, key),
            self.field_names,
        )

    def homogeneous_corrections(self, fields, batch, context, key=None, /):
        action = self.action(fields, batch, context, key)
        _fiber_product_leaves(action)
        return _checked(
            self.synthesis(
                _shared_lift(self.operator, _negate(action)),
                fields,
                batch,
                context,
                key,
            ),
            self.field_names,
        )


FiberProjectionUnit = (
    AnalyticFiberProjectionUnit
    | RealizedFiberProjectionUnit
    | SeparableFiberProjectionUnit
)


class FiberProjectionState(StrictModule):
    """Prepared disjoint fiber units committed atomically from one input mapping."""

    units: tuple[FiberProjectionUnit, ...]
    evidence: tuple[Any, ...]
    prepared_id: str = eqx.field(static=True)
    numeric_version: int = eqx.field(static=True)

    def __init__(
        self,
        units: Sequence[FiberProjectionUnit],
        /,
        *,
        evidence: Sequence[Any] = (),
        prepared_id: str | None = None,
        numeric_version: int = 0,
    ):
        values = tuple(units)
        if not values or any(
            not isinstance(
                unit,
                (
                    AnalyticFiberProjectionUnit,
                    RealizedFiberProjectionUnit,
                    SeparableFiberProjectionUnit,
                ),
            )
            for unit in values
        ):
            raise TypeError("FiberProjectionState requires at least one projection unit.")
        written = tuple(name for unit in values for name in unit.field_names)
        if len(set(written)) != len(written):
            raise ValueError("Coupled fields must be assembled in one joint fiber unit.")
        condition_ids = tuple(
            condition_id for unit in values for condition_id in unit.condition_ids
        )
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError(
                "Intersecting conditions must be fused into one joint fiber unit."
            )
        analytic = tuple(isinstance(unit, AnalyticFiberProjectionUnit) for unit in values)
        if any(analytic) and not all(analytic):
            raise ValueError("Analytic and realized fiber units require separate states.")
        version = int(numeric_version)
        if version < 0:
            raise ValueError("numeric_version must be nonnegative.")
        evidence_ = tuple(evidence)
        if evidence_ and len(evidence_) != len(values):
            raise ValueError("Fiber state evidence must have one entry per unit.")
        self.units = values
        self.evidence = evidence_ or tuple(unit.evidence for unit in values)
        self.numeric_version = version
        self.prepared_id = _id(
            prepared_id,
            {
                "kind": "fiber-state",
                "units": tuple(unit.unit_id for unit in values),
                "version": version,
            },
        )

    def project_analytic(self, fields, context, /):
        updates: dict[str, Any] = {}
        for unit in self.units:
            if not isinstance(unit, AnalyticFiberProjectionUnit):
                raise TypeError("project_analytic requires analytic units only.")
            updates.update(unit.corrections(fields, context))
        return _add(fields, updates)

    def project_batch(self, fields, batch, context, /, *, key=None, **kwargs):
        base = {
            name: value(batch, key=key, **kwargs)
            if isinstance(value, DomainFunction)
            else value
            for name, value in fields.items()
        }
        updates: dict[str, Any] = {}
        for unit in self.units:
            if isinstance(unit, AnalyticFiberProjectionUnit):
                raise TypeError("Analytic units do not require fixed-batch execution.")
            updates.update(unit.corrections(fields, batch, context, key))
        return _add(base, updates)


class _FiberProjectedEvaluator(StrictModule, BatchEvaluator):
    fields: frozendict[str, Any]
    state: FiberProjectionState
    context: Any
    field_name: str = eqx.field(static=True)

    def __init__(self, fields, state, context, field_name, /):
        self.fields, self.state, self.context, self.field_name = (
            frozendict(fields),
            state,
            context,
            str(field_name),
        )

    def __call_batch__(self, batch, /, *, key=None, **kwargs):
        value = self.state.project_batch(
            self.fields, batch, self.context, key=key, **kwargs
        )[self.field_name]
        if not isinstance(value, cx.Field):
            raise TypeError("Fiber BatchEvaluator output must be a coordax.Field.")
        return value


def realized_fiber_functions(
    fields,
    state: FiberProjectionState,
    context,
    /,
    *,
    derivative_actions: Mapping[str, FiberDerivativeAction] = frozendict(),
):
    """Wrap stored fiber factors as batch-aware projected DomainFunctions."""
    out = dict(fields)
    for unit in state.units:
        for name in unit.field_names:
            source = fields[name]
            if not isinstance(source, DomainFunction):
                raise TypeError("Realized fiber fields must be DomainFunctions.")
            rule = (
                None
                if name not in derivative_actions
                else FiberProjectionDerivativeRule(derivative_actions[name], name)
            )
            out[name] = DomainFunction(
                domain=source.domain,
                deps=source.deps,
                func=_FiberProjectedEvaluator(fields, state, context, name),
                metadata=source.metadata,
                derivative_rule=rule,
            )
    return frozendict(out)


__all__ = [
    "AnalyticFiberProjectionUnit",
    "BatchedFiberFactor",
    "FiberProjectionDerivativeRule",
    "FiberProjectionState",
    "RealizedFiberProjectionUnit",
    "SeparableFiberProjectionUnit",
    "realized_fiber_functions",
]
