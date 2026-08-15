#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, cast, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._costs import PreconditionerCostEstimate
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator, AdjointLinearOperator
from ._preconditioner_properties import (
    _preconditioner_properties_payload,
    PreconditionerProperties,
)
from ._preconditioners import _prepared_action_cost, AbstractPreconditioner
from ._preconditioning import (
    _source_cost,
    _validate_setup_operator,
    AbstractPreconditionerBuilder,
    PreconditionerSource,
)
from ._spaces import _coordinate_dtype


SubspaceCorrectionSweep: TypeAlias = Literal["forward", "backward", "symmetric"]


def _source_identifier(source: PreconditionerSource, /) -> tuple[str, str]:
    if isinstance(source, AbstractPreconditioner):
        return "prepared", source.preconditioner_id
    if isinstance(source, AbstractPreconditionerBuilder):
        return "builder", source.builder_id
    raise TypeError("local_solver must be a preconditioner or preconditioner builder.")


def _identifier(value: str | None, payload: dict[str, object], /) -> str:
    identifier = canonical_fingerprint(payload) if value is None else str(value)
    if not identifier:
        raise ValueError("Subspace-correction identifiers must be non-empty.")
    return identifier


class SubspaceCorrectionTerm(StrictModule):
    """Restriction, prolongation, and local approximate-inverse recipe."""

    restriction: AbstractLinearOperator
    prolongation: AbstractLinearOperator
    local_solver: PreconditionerSource
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        restriction: AbstractLinearOperator,
        prolongation: AbstractLinearOperator,
        local_solver: PreconditionerSource,
        /,
    ):
        if not isinstance(restriction, AbstractLinearOperator) or not isinstance(
            prolongation, AbstractLinearOperator
        ):
            raise TypeError("restriction and prolongation must be linear operators.")
        if restriction.batch_shape or prolongation.batch_shape:
            raise ValueError("Subspace-correction transfers must be unbatched.")
        if not restriction.target.compatible(prolongation.source):
            raise ValueError(
                "Restriction target and prolongation source must be the same local space."
            )
        if not restriction.source.compatible(prolongation.target):
            raise ValueError(
                "Restriction source and prolongation target must be the same global space."
            )
        source_kind, source_id = _source_identifier(local_solver)
        if isinstance(local_solver, AbstractPreconditioner) and not (
            local_solver.space.compatible(restriction.target)
        ):
            raise ValueError("A supplied local solver must act on the local space.")
        self.restriction = restriction
        self.prolongation = prolongation
        self.local_solver = local_solver
        self.term_id = canonical_fingerprint(
            {
                "kind": "subspace-correction-term",
                "restriction": restriction.operator_id,
                "prolongation": prolongation.operator_id,
                "local_solver_kind": source_kind,
                "local_solver": source_id,
            }
        )


def _validate_terms(
    terms: tuple[SubspaceCorrectionTerm, ...],
    /,
) -> tuple[SubspaceCorrectionTerm, ...]:
    if not isinstance(terms, tuple):
        raise TypeError("terms must be a tuple of SubspaceCorrectionTerm values.")
    if not terms:
        raise ValueError("Subspace correction requires at least one term.")
    if not all(isinstance(term, SubspaceCorrectionTerm) for term in terms):
        raise TypeError("terms must contain only SubspaceCorrectionTerm values.")
    global_space = terms[0].restriction.source
    for index, term in enumerate(terms):
        if not term.restriction.source.compatible(global_space) or not (
            term.prolongation.target.compatible(global_space)
        ):
            raise ValueError(
                f"Subspace-correction term {index} does not use the shared global space."
            )
    return terms


def _local_setup_operator(
    term: SubspaceCorrectionTerm,
    setup_operator: AbstractLinearOperator,
    /,
) -> AbstractLinearOperator:
    return term.restriction @ setup_operator @ term.prolongation


def _local_setup_operators(
    terms: tuple[SubspaceCorrectionTerm, ...],
    setup_operator: AbstractLinearOperator,
    /,
) -> tuple[AbstractLinearOperator, ...]:
    _validate_setup_operator(setup_operator)
    if not setup_operator.source.compatible(terms[0].restriction.source):
        raise ValueError("The setup operator must act on the terms' global space.")
    operators = tuple(_local_setup_operator(term, setup_operator) for term in terms)
    for index, operator in enumerate(operators):
        if operator.batch_shape or not operator.source.compatible(operator.target):
            raise ValueError(
                f"Derived local setup operator {index} must be an unbatched endomorphism."
            )
    return operators


def _source_properties(
    source: PreconditionerSource,
    setup_operator: AbstractLinearOperator,
    /,
) -> PreconditionerProperties:
    properties = (
        source.properties
        if isinstance(source, AbstractPreconditioner)
        else source.properties_for(setup_operator)
    )
    if not isinstance(properties, PreconditionerProperties):
        raise TypeError("Every local solver must expose PreconditionerProperties.")
    return properties


def _structurally_adjoint_transfers(term: SubspaceCorrectionTerm, /) -> bool:
    restriction = term.restriction
    prolongation = term.prolongation
    return (
        (restriction is prolongation and restriction.properties.certifies("self_adjoint"))
        or (
            isinstance(restriction, AdjointLinearOperator)
            and restriction.operator is prolongation
        )
        or (
            isinstance(prolongation, AdjointLinearOperator)
            and prolongation.operator is restriction
        )
    )


def _resolved_properties(
    terms: tuple[SubspaceCorrectionTerm, ...],
    local_properties: tuple[PreconditionerProperties, ...],
    supplied: PreconditionerProperties | None,
    /,
    *,
    multiplicative_sweep: SubspaceCorrectionSweep | None,
    setup_operator: AbstractLinearOperator,
) -> PreconditionerProperties:
    linear = all(value.certifies("linear") for value in local_properties)
    stationary = all(value.certifies("stationary") for value in local_properties)
    local_self_adjoint = all(
        value.certifies("self_adjoint") for value in local_properties
    )
    adjoint_transfers = all(_structurally_adjoint_transfers(term) for term in terms)
    if multiplicative_sweep is None:
        self_adjoint = linear and local_self_adjoint and adjoint_transfers
    else:
        self_adjoint = (
            multiplicative_sweep == "symmetric"
            and linear
            and local_self_adjoint
            and adjoint_transfers
            and setup_operator.properties.certifies("self_adjoint")
        )
    if supplied is None:
        claims = {
            "linear": linear,
            "stationary": stationary,
            "self_adjoint": self_adjoint,
        }
        return PreconditionerProperties(
            **claims,
            evidence={name: "transformed" for name, claimed in claims.items() if claimed},
        )
    if not isinstance(supplied, PreconditionerProperties):
        raise TypeError("properties must be PreconditionerProperties or None.")
    unsupported = []
    if supplied.linear and not linear:
        unsupported.append("linear")
    if supplied.stationary and not stationary:
        unsupported.append("stationary")
    if supplied.self_adjoint and not self_adjoint:
        unsupported.append("self_adjoint")
    if supplied.positive_definite and not (stationary and self_adjoint):
        unsupported.append("positive_definite")
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            "Supplied subspace-correction properties lack component and structural "
            f"support for: {names}."
        )
    return supplied


def _validate_prepared_local_solver(
    action: AbstractPreconditioner,
    setup_operator: AbstractLinearOperator,
    expected: PreconditionerProperties,
    /,
) -> None:
    if not isinstance(action, AbstractPreconditioner):
        raise TypeError("A local builder must prepare an AbstractPreconditioner.")
    if not action.space.compatible(setup_operator.source):
        raise ValueError("A prepared local solver must act on its local setup space.")
    for name in ("linear", "stationary", "self_adjoint", "positive_definite"):
        if expected.certifies(name) and not action.properties.certifies(name):
            raise ValueError(
                f"Prepared local solver does not certify planned property {name!r}."
            )


def _prepare_terms(
    terms: tuple[SubspaceCorrectionTerm, ...],
    local_operators: tuple[AbstractLinearOperator, ...],
    /,
    *,
    materialization: MaterializationPolicy,
) -> tuple[SubspaceCorrectionTerm, ...]:
    prepared = []
    for term, operator in zip(terms, local_operators, strict=True):
        source = term.local_solver
        expected = _source_properties(source, operator)
        action = (
            source
            if isinstance(source, AbstractPreconditioner)
            else source.prepare(operator, materialization=materialization)
        )
        _validate_prepared_local_solver(action, operator, expected)
        prepared.append(
            SubspaceCorrectionTerm(term.restriction, term.prolongation, action)
        )
    return tuple(prepared)


def _refresh_terms(
    terms: tuple[SubspaceCorrectionTerm, ...],
    previous_terms: tuple[SubspaceCorrectionTerm, ...],
    local_operators: tuple[AbstractLinearOperator, ...],
    /,
    *,
    materialization: MaterializationPolicy,
) -> tuple[SubspaceCorrectionTerm, ...]:
    refreshed = []
    for term, previous, operator in zip(
        terms, previous_terms, local_operators, strict=True
    ):
        previous_action = cast(AbstractPreconditioner, previous.local_solver)
        source = term.local_solver
        expected = _source_properties(source, operator)
        action = (
            source
            if isinstance(source, AbstractPreconditioner)
            else source.refresh(
                previous_action,
                operator,
                materialization=materialization,
            )
        )
        _validate_prepared_local_solver(action, operator, expected)
        refreshed.append(
            SubspaceCorrectionTerm(term.restriction, term.prolongation, action)
        )
    return tuple(refreshed)


def _validate_prepared_terms(
    terms: tuple[SubspaceCorrectionTerm, ...],
    local_operators: tuple[AbstractLinearOperator, ...],
    /,
) -> None:
    for term, operator in zip(terms, local_operators, strict=True):
        if not isinstance(term.local_solver, AbstractPreconditioner):
            raise TypeError("Prepared correction terms require prepared local solvers.")
        _validate_prepared_local_solver(
            term.local_solver,
            operator,
            term.local_solver.properties,
        )


def _prepared_properties(
    terms: tuple[SubspaceCorrectionTerm, ...],
    local_operators: tuple[AbstractLinearOperator, ...],
    supplied: PreconditionerProperties | None,
    /,
    *,
    multiplicative_sweep: SubspaceCorrectionSweep | None,
    setup_operator: AbstractLinearOperator,
) -> PreconditionerProperties:
    return _resolved_properties(
        terms,
        tuple(
            _source_properties(term.local_solver, operator)
            for term, operator in zip(terms, local_operators, strict=True)
        ),
        supplied,
        multiplicative_sweep=multiplicative_sweep,
        setup_operator=setup_operator,
    )


def _add(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x + y, left, right)


def _subtract(left: PyTree[Array], right: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree.map(lambda x, y: x - y, left, right)


class AdditiveSubspaceCorrectionPreconditioner(AbstractPreconditioner):
    """Prepared sum of independent prolongated local corrections."""

    setup_operator: AbstractLinearOperator
    terms: tuple[SubspaceCorrectionTerm, ...]
    local_setup_operators: tuple[AbstractLinearOperator, ...]
    builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        setup_operator: AbstractLinearOperator,
        terms: tuple[SubspaceCorrectionTerm, ...],
        /,
        *,
        properties: PreconditionerProperties | None = None,
        builder_id: str | None = None,
        preconditioner_id: str | None = None,
    ):
        terms_ = _validate_terms(terms)
        local_operators = _local_setup_operators(terms_, setup_operator)
        _validate_prepared_terms(terms_, local_operators)
        properties_ = _prepared_properties(
            terms_,
            local_operators,
            properties,
            multiplicative_sweep=None,
            setup_operator=setup_operator,
        )
        builder_id_ = _identifier(
            builder_id,
            {
                "kind": "direct-additive-subspace-correction",
                "terms": [term.term_id for term in terms_],
                "properties": _preconditioner_properties_payload(properties_),
            },
        )
        self.space = setup_operator.source
        self.properties = properties_
        self.setup_operator = setup_operator
        self.terms = terms_
        self.local_setup_operators = local_operators
        self.builder_id = builder_id_
        self.preconditioner_id = _identifier(
            preconditioner_id,
            {
                "kind": "prepared-additive-subspace-correction",
                "builder": builder_id_,
                "setup_operator": setup_operator.operator_id,
            },
        )

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        residual_ = self.space.validate(residual)
        correction = jax.tree.map(jnp.zeros_like, residual_)
        for term in self.terms:
            local_solver = cast(AbstractPreconditioner, term.local_solver)
            local_residual = term.restriction.mv(residual_)
            local_correction = local_solver.apply(
                local_residual,
                iteration=iteration,
            )
            correction = _add(
                correction,
                term.prolongation.mv(local_correction),
            )
        return correction

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        return _prepared_action_cost(
            self, setup_operator, apply_workspace_multiplier=2 * len(self.terms)
        )


class MultiplicativeSubspaceCorrectionPreconditioner(AbstractPreconditioner):
    """Prepared ordered defect-correction sweep over local subspaces."""

    setup_operator: AbstractLinearOperator
    terms: tuple[SubspaceCorrectionTerm, ...]
    local_setup_operators: tuple[AbstractLinearOperator, ...]
    sweep: SubspaceCorrectionSweep = eqx.field(static=True)
    term_order: tuple[int, ...] = eqx.field(static=True)
    builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        setup_operator: AbstractLinearOperator,
        terms: tuple[SubspaceCorrectionTerm, ...],
        /,
        *,
        sweep: SubspaceCorrectionSweep = "forward",
        properties: PreconditionerProperties | None = None,
        builder_id: str | None = None,
        preconditioner_id: str | None = None,
    ):
        if sweep not in ("forward", "backward", "symmetric"):
            raise ValueError("sweep must be 'forward', 'backward', or 'symmetric'.")
        terms_ = _validate_terms(terms)
        local_operators = _local_setup_operators(terms_, setup_operator)
        _validate_prepared_terms(terms_, local_operators)
        properties_ = _prepared_properties(
            terms_,
            local_operators,
            properties,
            multiplicative_sweep=sweep,
            setup_operator=setup_operator,
        )
        builder_id_ = _identifier(
            builder_id,
            {
                "kind": "direct-multiplicative-subspace-correction",
                "terms": [term.term_id for term in terms_],
                "sweep": sweep,
                "properties": _preconditioner_properties_payload(properties_),
            },
        )
        self.space = setup_operator.source
        self.properties = properties_
        self.setup_operator = setup_operator
        self.terms = terms_
        self.local_setup_operators = local_operators
        self.sweep = sweep
        count = len(terms_)
        if sweep == "forward":
            self.term_order = tuple(range(count))
        elif sweep == "backward":
            self.term_order = tuple(range(count - 1, -1, -1))
        else:
            self.term_order = tuple(range(count)) + tuple(range(count - 2, -1, -1))
        self.builder_id = builder_id_
        self.preconditioner_id = _identifier(
            preconditioner_id,
            {
                "kind": "prepared-multiplicative-subspace-correction",
                "builder": builder_id_,
                "setup_operator": setup_operator.operator_id,
            },
        )

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        residual_ = self.space.validate(residual)
        correction = jax.tree.map(jnp.zeros_like, residual_)
        for index in self.term_order:
            term = self.terms[index]
            local_solver = cast(AbstractPreconditioner, term.local_solver)
            defect = _subtract(
                residual_,
                self.setup_operator.mv(correction),
            )
            local_correction = local_solver.apply(
                term.restriction.mv(defect),
                iteration=iteration,
            )
            correction = _add(
                correction,
                term.prolongation.mv(local_correction),
            )
        return correction

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        return _prepared_action_cost(
            self, setup_operator, apply_workspace_multiplier=3 * len(self.term_order)
        )


def _terms_properties(
    terms: tuple[SubspaceCorrectionTerm, ...],
    local_operators: tuple[AbstractLinearOperator, ...],
    /,
) -> tuple[PreconditionerProperties, ...]:
    return tuple(
        _source_properties(term.local_solver, operator)
        for term, operator in zip(terms, local_operators, strict=True)
    )


def _builder_payload(
    kind: str,
    terms: tuple[SubspaceCorrectionTerm, ...],
    properties: PreconditionerProperties | None,
    /,
    *,
    sweep: SubspaceCorrectionSweep | None,
) -> dict[str, object]:
    return {
        "kind": kind,
        "terms": [term.term_id for term in terms],
        "sweep": sweep,
        "properties": (
            None if properties is None else _preconditioner_properties_payload(properties)
        ),
    }


def _operator_storage_bytes(
    operators: tuple[AbstractLinearOperator, ...],
    /,
) -> int:
    arrays = {}
    for value in jax.tree.leaves(operators):
        if eqx.is_array(value):
            arrays[id(value)] = value
    return sum(int(value.size * value.dtype.itemsize) for value in arrays.values())


def _cost_for_terms(
    builder_id: str,
    terms: tuple[SubspaceCorrectionTerm, ...],
    local_operators: tuple[AbstractLinearOperator, ...],
    /,
    *,
    multiplicative: bool,
    materialization: MaterializationPolicy | None,
) -> PreconditionerCostEstimate:
    storage = _operator_storage_bytes(
        tuple(
            operator
            for term in terms
            for operator in (term.restriction, term.prolongation)
        )
    )
    preparation_workspace = 0
    apply_workspace = 0
    setup_matvecs = 0
    accepted = True
    rejected = []
    global_bytes = (
        terms[0].restriction.source.size
        * _coordinate_dtype(terms[0].restriction.source).itemsize
    )
    for index, (term, operator) in enumerate(zip(terms, local_operators, strict=True)):
        estimate = _source_cost(
            term.local_solver, operator, materialization=materialization
        )
        storage += estimate.storage_bytes
        preparation_workspace += estimate.preparation_workspace_bytes
        apply_workspace += estimate.apply_workspace_bytes_per_rhs
        setup_matvecs += estimate.setup_matvec_count
        local_bytes = operator.source.size * _coordinate_dtype(operator.source).itemsize
        apply_workspace += 2 * local_bytes + global_bytes
        if multiplicative:
            apply_workspace += 2 * global_bytes
        if not estimate.accepted:
            accepted = False
            rejected.append(f"term {index}: {estimate.reason}")
    reason = (
        "subspace-correction local actions and transfer operators"
        if accepted
        else "; ".join(rejected)
    )
    return PreconditionerCostEstimate(
        component=builder_id,
        storage_bytes=storage,
        preparation_workspace_bytes=preparation_workspace,
        apply_workspace_bytes_per_rhs=apply_workspace,
        setup_matvec_count=setup_matvecs,
        accepted=accepted,
        reason=reason,
    )


class AdditiveSubspaceCorrectionBuilder(AbstractPreconditionerBuilder):
    """Build a sum of local corrections from one global setup operator."""

    terms: tuple[SubspaceCorrectionTerm, ...]
    properties: PreconditionerProperties | None
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: tuple[SubspaceCorrectionTerm, ...],
        /,
        *,
        properties: PreconditionerProperties | None = None,
    ):
        terms_ = _validate_terms(terms)
        if properties is not None and not isinstance(
            properties, PreconditionerProperties
        ):
            raise TypeError("properties must be PreconditionerProperties or None.")
        self.terms = terms_
        self.properties = properties
        self._builder_id = canonical_fingerprint(
            _builder_payload(
                "additive-subspace-correction-builder",
                terms_,
                properties,
                sweep=None,
            )
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        local_operators = _local_setup_operators(self.terms, setup_operator)
        return _resolved_properties(
            self.terms,
            _terms_properties(self.terms, local_operators),
            self.properties,
            multiplicative_sweep=None,
            setup_operator=setup_operator,
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        local_operators = _local_setup_operators(self.terms, setup_operator)
        self.properties_for(setup_operator)
        return _cost_for_terms(
            self.builder_id,
            self.terms,
            local_operators,
            multiplicative=False,
            materialization=materialization,
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        local_operators = _local_setup_operators(self.terms, setup_operator)
        properties = self.properties_for(setup_operator)
        terms = _prepare_terms(
            self.terms,
            local_operators,
            materialization=materialization,
        )
        return AdditiveSubspaceCorrectionPreconditioner(
            setup_operator,
            terms,
            properties=properties,
            builder_id=self.builder_id,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, AdditiveSubspaceCorrectionPreconditioner):
            raise TypeError(
                "Additive subspace-correction refresh requires its prepared action."
            )
        if preconditioner.builder_id != self.builder_id:
            raise ValueError("Subspace-correction refresh must preserve its builder ID.")
        local_operators = _local_setup_operators(self.terms, setup_operator)
        properties = self.properties_for(setup_operator)
        terms = _refresh_terms(
            self.terms,
            preconditioner.terms,
            local_operators,
            materialization=materialization,
        )
        return AdditiveSubspaceCorrectionPreconditioner(
            setup_operator,
            terms,
            properties=properties,
            builder_id=self.builder_id,
        )


class MultiplicativeSubspaceCorrectionBuilder(AbstractPreconditionerBuilder):
    """Build an ordered defect-correction sweep over local subspaces."""

    terms: tuple[SubspaceCorrectionTerm, ...]
    properties: PreconditionerProperties | None
    sweep: SubspaceCorrectionSweep = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        terms: tuple[SubspaceCorrectionTerm, ...],
        /,
        *,
        sweep: SubspaceCorrectionSweep = "forward",
        properties: PreconditionerProperties | None = None,
    ):
        if sweep not in ("forward", "backward", "symmetric"):
            raise ValueError("sweep must be 'forward', 'backward', or 'symmetric'.")
        terms_ = _validate_terms(terms)
        if properties is not None and not isinstance(
            properties, PreconditionerProperties
        ):
            raise TypeError("properties must be PreconditionerProperties or None.")
        self.terms = terms_
        self.properties = properties
        self.sweep = sweep
        self._builder_id = canonical_fingerprint(
            _builder_payload(
                "multiplicative-subspace-correction-builder",
                terms_,
                properties,
                sweep=sweep,
            )
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        local_operators = _local_setup_operators(self.terms, setup_operator)
        return _resolved_properties(
            self.terms,
            _terms_properties(self.terms, local_operators),
            self.properties,
            multiplicative_sweep=self.sweep,
            setup_operator=setup_operator,
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        local_operators = _local_setup_operators(self.terms, setup_operator)
        self.properties_for(setup_operator)
        return _cost_for_terms(
            self.builder_id,
            self.terms,
            local_operators,
            multiplicative=True,
            materialization=materialization,
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        local_operators = _local_setup_operators(self.terms, setup_operator)
        properties = self.properties_for(setup_operator)
        terms = _prepare_terms(
            self.terms,
            local_operators,
            materialization=materialization,
        )
        return MultiplicativeSubspaceCorrectionPreconditioner(
            setup_operator,
            terms,
            sweep=self.sweep,
            properties=properties,
            builder_id=self.builder_id,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, MultiplicativeSubspaceCorrectionPreconditioner):
            raise TypeError(
                "Multiplicative subspace-correction refresh requires its prepared action."
            )
        if preconditioner.builder_id != self.builder_id:
            raise ValueError("Subspace-correction refresh must preserve its builder ID.")
        local_operators = _local_setup_operators(self.terms, setup_operator)
        properties = self.properties_for(setup_operator)
        terms = _refresh_terms(
            self.terms,
            preconditioner.terms,
            local_operators,
            materialization=materialization,
        )
        return MultiplicativeSubspaceCorrectionPreconditioner(
            setup_operator,
            terms,
            sweep=self.sweep,
            properties=properties,
            builder_id=self.builder_id,
        )


__all__ = [
    "AdditiveSubspaceCorrectionBuilder",
    "AdditiveSubspaceCorrectionPreconditioner",
    "MultiplicativeSubspaceCorrectionBuilder",
    "MultiplicativeSubspaceCorrectionPreconditioner",
    "SubspaceCorrectionSweep",
    "SubspaceCorrectionTerm",
]
