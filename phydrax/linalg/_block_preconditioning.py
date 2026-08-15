#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax
from jaxtyping import ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from ._costs import _array_tree_storage_bytes, PreconditionerCostEstimate
from ._materialization import MaterializationPolicy
from ._operators import (
    AbstractLinearOperator,
    AdjointLinearOperator,
    BlockLinearOperator,
    FunctionLinearOperator,
    IdentityLinearOperator,
    ScaledLinearOperator,
)
from ._preconditioner_properties import (
    _preconditioner_properties_payload,
    PreconditionerProperties,
)
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import (
    _source_cost,
    AbstractPreconditionerBuilder,
    PreconditionerSource,
)
from ._properties import OperatorCapabilities, OperatorProperties
from ._spaces import _coordinate_dtype, BlockSpace
from ._structured_operators import SchurComplementLinearOperator


BlockFactorizationForm: TypeAlias = Literal["diagonal", "lower", "upper", "ldu"]


def _tree_subtract(left: PyTree, right: PyTree, /) -> PyTree:
    return jax.tree.map(lambda x, y: x - y, left, right)


def _source_identifier(source: PreconditionerSource, /) -> str:
    if isinstance(source, AbstractPreconditioner):
        return source.preconditioner_id
    if isinstance(source, AbstractPreconditionerBuilder):
        return source.builder_id
    raise TypeError(
        "pivot_solver and schur_solver must be AbstractPreconditioner values or "
        "AbstractPreconditionerBuilder values."
    )


def _source_properties(
    source: PreconditionerSource,
    setup_operator: AbstractLinearOperator,
    /,
) -> PreconditionerProperties:
    if isinstance(source, AbstractPreconditioner):
        if not source.space.compatible(setup_operator.source):
            raise ValueError("Prepared component action has an incompatible space.")
        return source.properties
    if isinstance(source, AbstractPreconditionerBuilder):
        return source.properties_for(setup_operator)
    _source_identifier(source)
    raise RuntimeError("Unreachable preconditioner source state.")


def _validate_fixed_action(
    action: AbstractPreconditioner,
    space,
    component: str,
    /,
) -> None:
    if not isinstance(action, AbstractPreconditioner):
        raise TypeError(f"{component} solver must prepare an AbstractPreconditioner.")
    if not action.space.compatible(space):
        raise ValueError(f"{component} action has an incompatible space.")
    if not action.properties.certifies("linear") or not action.properties.certifies(
        "stationary"
    ):
        raise ValueError(
            f"{component} action must certify fixed linear, stationary semantics."
        )


def _prepare_source(
    source: PreconditionerSource,
    setup_operator: AbstractLinearOperator,
    /,
    *,
    materialization: MaterializationPolicy,
    component: str,
) -> AbstractPreconditioner:
    if isinstance(source, AbstractPreconditioner):
        action = source
    elif isinstance(source, AbstractPreconditionerBuilder):
        action = source.prepare(setup_operator, materialization=materialization)
    else:
        _source_identifier(source)
        raise RuntimeError("Unreachable preconditioner source state.")
    _validate_fixed_action(action, setup_operator.source, component)
    return action


def _refresh_source(
    source: PreconditionerSource,
    previous: AbstractPreconditioner,
    setup_operator: AbstractLinearOperator,
    /,
    *,
    materialization: MaterializationPolicy,
    component: str,
) -> AbstractPreconditioner:
    if isinstance(source, AbstractPreconditioner):
        action = source
    elif isinstance(source, AbstractPreconditionerBuilder):
        action = source.refresh(
            previous,
            setup_operator,
            materialization=materialization,
        )
    else:
        _source_identifier(source)
        raise RuntimeError("Unreachable preconditioner source state.")
    _validate_fixed_action(action, setup_operator.source, component)
    return action


def _is_adjoint_pair(
    lower: AbstractLinearOperator,
    upper: AbstractLinearOperator,
    /,
) -> bool:
    return (isinstance(upper, AdjointLinearOperator) and upper.operator is lower) or (
        isinstance(lower, AdjointLinearOperator) and lower.operator is upper
    )


def _zero_operator(space, /) -> AbstractLinearOperator:
    return ScaledLinearOperator(IdentityLinearOperator(space), 0.0)


def _block_components(
    setup_operator: AbstractLinearOperator,
    /,
) -> tuple[
    BlockLinearOperator,
    AbstractLinearOperator,
    AbstractLinearOperator,
    AbstractLinearOperator,
    AbstractLinearOperator,
]:
    if not isinstance(setup_operator, BlockLinearOperator):
        raise TypeError("setup_operator must be a BlockLinearOperator.")
    if (
        len(setup_operator.source.spaces) != 2
        or len(setup_operator.target.spaces) != 2
        or len(setup_operator.blocks) != 2
        or any(len(row) != 2 for row in setup_operator.blocks)
    ):
        raise ValueError("Block factorization requires an exact 2-by-2 block grid.")
    if not setup_operator.source.compatible(setup_operator.target):
        raise ValueError("Block factorization requires a compatible endomorphism.")
    pivot, upper = setup_operator.blocks[0]
    lower, diagonal = setup_operator.blocks[1]
    if pivot is None or upper is None or lower is None:
        raise ValueError(
            "Block factorization requires A00, A01, and A10; only A11 may be zero."
        )
    diagonal_ = (
        _zero_operator(setup_operator.source.spaces[1]) if diagonal is None else diagonal
    )
    return setup_operator, pivot, upper, lower, diagonal_


def _validate_schur_setup(
    operator: AbstractLinearOperator,
    dual_space,
    /,
) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("schur_setup_operator must be an AbstractLinearOperator or None.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("schur_setup_operator must be an unbatched endomorphism.")
    if not operator.source.compatible(dual_space):
        raise ValueError("schur_setup_operator has an incompatible dual space.")


def _component_properties(
    pivot: PreconditionerProperties,
    schur: PreconditionerProperties,
    form: BlockFactorizationForm,
    paired_off_diagonals: bool,
    supplied: PreconditionerProperties | None,
    /,
) -> PreconditionerProperties:
    linear = pivot.certifies("linear") and schur.certifies("linear")
    stationary = pivot.certifies("stationary") and schur.certifies("stationary")
    self_adjoint = (
        linear
        and stationary
        and (
            (
                form == "diagonal"
                and pivot.certifies("self_adjoint")
                and schur.certifies("self_adjoint")
            )
            or (
                form == "ldu"
                and paired_off_diagonals
                and pivot.certifies("self_adjoint")
                and schur.certifies("self_adjoint")
            )
        )
    )
    if supplied is None:
        claims = {
            "linear": linear,
            "stationary": stationary,
            "self_adjoint": self_adjoint,
            "positive_definite": False,
        }
        return PreconditionerProperties(
            **claims,
            evidence={name: "transformed" for name, claimed in claims.items() if claimed},
        )
    if not isinstance(supplied, PreconditionerProperties):
        raise TypeError("properties must be PreconditionerProperties or None.")
    requires_linear = any(
        supplied.certifies(name)
        for name in ("linear", "self_adjoint", "positive_definite")
    )
    requires_stationary = any(
        supplied.certifies(name)
        for name in ("stationary", "self_adjoint", "positive_definite")
    )
    if (requires_linear and not linear) or (requires_stationary and not stationary):
        raise ValueError(
            "Certified block properties require both actions to certify the "
            "corresponding fixed linear semantics."
        )
    if supplied.self_adjoint and not self_adjoint:
        if form in ("lower", "upper"):
            raise ValueError("Triangular block factorization is nonsymmetric.")
        raise ValueError(
            "A self-adjoint block claim requires certified self-adjoint actions "
            "and pairing-aware adjoint off-diagonals for LDU."
        )
    if supplied.positive_definite and not (
        pivot.certifies("positive_definite") and schur.certifies("positive_definite")
    ):
        raise ValueError(
            "A positive-definite block claim requires certified positive-definite "
            "component actions."
        )
    return supplied


def _schur_properties(
    diagonal: AbstractLinearOperator,
    lower: AbstractLinearOperator,
    upper: AbstractLinearOperator,
    pivot: PreconditionerProperties,
    /,
) -> OperatorProperties:
    self_adjoint = (
        diagonal.properties.certifies("self_adjoint")
        and pivot.certifies("linear")
        and pivot.certifies("stationary")
        and pivot.certifies("self_adjoint")
        and _is_adjoint_pair(lower, upper)
    )
    return OperatorProperties(
        self_adjoint=self_adjoint,
        evidence={"self_adjoint": "transformed"} if self_adjoint else {},
    )


class _PlanningSchurLinearOperator(AbstractLinearOperator):
    """Forward-only capability and property envelope for derived Schur setup."""

    diagonal_block: AbstractLinearOperator

    def __init__(
        self,
        diagonal_block: AbstractLinearOperator,
        /,
        *,
        properties: OperatorProperties,
        operator_id: str,
    ):
        self.diagonal_block = diagonal_block
        self.source = diagonal_block.source
        self.target = diagonal_block.target
        self.properties = properties
        self.capabilities = OperatorCapabilities(
            transpose=False,
            adjoint=False,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = operator_id

    def mv(self, vector: PyTree, /) -> PyTree:
        return self.diagonal_block.mv(vector)

    def transpose_mv(self, vector: PyTree, /) -> PyTree:
        del vector
        raise ValueError("Planning Schur operator has no transpose action.")

    def adjoint_mv(self, vector: PyTree, /) -> PyTree:
        del vector
        raise ValueError("Planning Schur operator has no adjoint action.")

    def _materialize(self, /):
        raise ValueError("Planning Schur operator cannot be materialized.")


def _materializable_schur_setup(
    operator: SchurComplementLinearOperator,
    /,
) -> AbstractLinearOperator:
    return FunctionLinearOperator(
        operator.mv,
        source=operator.source,
        target=operator.target,
        properties=operator.properties,
        operator_id=canonical_fingerprint(
            {
                "kind": "materializable-schur-setup",
                "schur_operator": operator.operator_id,
            }
        ),
    )


class BlockFactorizationPreconditioner(AbstractPreconditioner):
    """Prepared approximate inverse for one compatible 2-by-2 block operator."""

    schur_operator: SchurComplementLinearOperator
    schur_action: AbstractPreconditioner
    form: BlockFactorizationForm = eqx.field(static=True)
    space: BlockSpace
    __hash__ = object.__hash__

    def __init__(
        self,
        schur_operator: SchurComplementLinearOperator,
        schur_action: AbstractPreconditioner,
        form: BlockFactorizationForm,
        /,
        *,
        properties: PreconditionerProperties,
        space: BlockSpace | None = None,
        preconditioner_id: str | None = None,
    ):
        if not isinstance(schur_operator, SchurComplementLinearOperator):
            raise TypeError("schur_operator must be a SchurComplementLinearOperator.")
        if not isinstance(schur_action, AbstractPreconditioner):
            raise TypeError("schur_action must be an AbstractPreconditioner.")
        if form not in ("diagonal", "lower", "upper", "ldu"):
            raise ValueError("Unknown block factorization form.")
        pivot_action = schur_operator.inverse_action
        _validate_fixed_action(pivot_action, schur_operator.upper_block.target, "pivot")
        _validate_fixed_action(schur_action, schur_operator.source, "Schur")
        properties_ = _component_properties(
            pivot_action.properties,
            schur_action.properties,
            form,
            _is_adjoint_pair(
                schur_operator.lower_block,
                schur_operator.upper_block,
            ),
            properties,
        )
        space_ = (
            BlockSpace((pivot_action.space, schur_action.space))
            if space is None
            else space
        )
        if not isinstance(space_, BlockSpace) or len(space_.spaces) != 2:
            raise TypeError("space must be a two-component BlockSpace or None.")
        if not space_.spaces[0].compatible(pivot_action.space) or not space_.spaces[
            1
        ].compatible(schur_action.space):
            raise ValueError("space members must match the pivot and Schur actions.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "block-factorization-preconditioner",
                    "form": form,
                    "schur_operator": schur_operator.operator_id,
                    "pivot_action": pivot_action.preconditioner_id,
                    "schur_action": schur_action.preconditioner_id,
                    "properties": _preconditioner_properties_payload(properties_),
                }
            )
            if preconditioner_id is None
            else str(preconditioner_id)
        )
        if not identifier:
            raise ValueError("preconditioner_id must be non-empty.")
        self.schur_operator = schur_operator
        self.schur_action = schur_action
        self.form = form
        self.space = space_
        self.properties = properties_
        self.preconditioner_id = identifier

    @property
    def pivot_action(self) -> AbstractPreconditioner:
        return self.schur_operator.inverse_action

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        block, pivot_setup, upper, lower, diagonal = _block_components(setup_operator)
        if not block.source.compatible(self.space):
            raise ValueError(
                "Block action cost requires a setup operator on the prepared space."
            )
        pivot_cost = _source_cost(
            self.pivot_action,
            pivot_setup,
            materialization=materialization,
        )
        schur_cost = _source_cost(
            self.schur_action,
            self.schur_operator,
            materialization=materialization,
        )
        accepted = pivot_cost.accepted and schur_cost.accepted
        workspace_multiplier = 2 if self.form == "ldu" else 1
        workspace = (
            workspace_multiplier
            * self.space.size
            * _coordinate_dtype(self.space).itemsize
        )
        return PreconditionerCostEstimate(
            component=self.preconditioner_id,
            storage_bytes=(
                pivot_cost.storage_bytes
                + schur_cost.storage_bytes
                + _array_tree_storage_bytes((diagonal, lower, upper))
            ),
            preparation_workspace_bytes=(
                pivot_cost.preparation_workspace_bytes
                + schur_cost.preparation_workspace_bytes
            ),
            apply_workspace_bytes_per_rhs=(
                pivot_cost.apply_workspace_bytes_per_rhs
                + schur_cost.apply_workspace_bytes_per_rhs
                + workspace
            ),
            setup_matvec_count=(
                pivot_cost.setup_matvec_count + schur_cost.setup_matvec_count
            ),
            accepted=accepted,
            reason=(
                "prepared pivot and Schur actions plus retained Schur operator state"
                if accepted
                else "; ".join(
                    estimate.reason
                    for estimate in (pivot_cost, schur_cost)
                    if not estimate.accepted
                )
            ),
        )

    def apply(
        self,
        residual: PyTree,
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> tuple[PyTree, PyTree]:
        first, second = self.space.validate(residual)
        pivot = self.pivot_action
        schur = self.schur_action
        upper = self.schur_operator.upper_block
        lower = self.schur_operator.lower_block
        if self.form == "diagonal":
            return (
                pivot.apply(first, iteration=iteration),
                schur.apply(second, iteration=iteration),
            )
        if self.form == "lower":
            first_solution = pivot.apply(first, iteration=iteration)
            second_rhs = _tree_subtract(second, lower.mv(first_solution))
            return first_solution, schur.apply(second_rhs, iteration=iteration)
        if self.form == "upper":
            second_solution = schur.apply(second, iteration=iteration)
            first_rhs = _tree_subtract(first, upper.mv(second_solution))
            return pivot.apply(first_rhs, iteration=iteration), second_solution
        pivot_rhs = pivot.apply(first, iteration=iteration)
        second_rhs = _tree_subtract(second, lower.mv(pivot_rhs))
        second_solution = schur.apply(second_rhs, iteration=iteration)
        first_correction = pivot.apply(upper.mv(second_solution), iteration=iteration)
        return _tree_subtract(pivot_rhs, first_correction), second_solution


class BlockFactorizationPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare diagonal, triangular, or LDU actions from a 2-by-2 block operator."""

    pivot_solver: PreconditionerSource
    schur_solver: PreconditionerSource
    form: BlockFactorizationForm = eqx.field(static=True)
    schur_setup_operator: AbstractLinearOperator | None
    properties: PreconditionerProperties | None
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        pivot_solver: PreconditionerSource,
        schur_solver: PreconditionerSource,
        form: BlockFactorizationForm,
        *,
        schur_setup_operator: AbstractLinearOperator | None = None,
        properties: PreconditionerProperties | None = None,
    ):
        pivot_id = _source_identifier(pivot_solver)
        schur_id = _source_identifier(schur_solver)
        if form not in ("diagonal", "lower", "upper", "ldu"):
            raise ValueError("form must be 'diagonal', 'lower', 'upper', or 'ldu'.")
        if schur_setup_operator is not None and not isinstance(
            schur_setup_operator, AbstractLinearOperator
        ):
            raise TypeError("schur_setup_operator must be an operator or None.")
        if properties is not None and not isinstance(
            properties, PreconditionerProperties
        ):
            raise TypeError("properties must be PreconditionerProperties or None.")
        if (
            form in ("lower", "upper")
            and properties is not None
            and (properties.self_adjoint or properties.positive_definite)
        ):
            raise ValueError("Triangular block factorization is nonsymmetric.")
        self.pivot_solver = pivot_solver
        self.schur_solver = schur_solver
        self.form = form
        self.schur_setup_operator = schur_setup_operator
        self.properties = properties
        self._builder_id = canonical_fingerprint(
            {
                "kind": "block-factorization-preconditioner-builder",
                "pivot_solver": pivot_id,
                "schur_solver": schur_id,
                "form": form,
                "schur_setup_operator": (
                    None
                    if schur_setup_operator is None
                    else schur_setup_operator.operator_id
                ),
                "properties": (
                    None
                    if properties is None
                    else _preconditioner_properties_payload(properties)
                ),
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "numeric"

    def _planning_schur_setup(
        self,
        diagonal: AbstractLinearOperator,
        lower: AbstractLinearOperator,
        upper: AbstractLinearOperator,
        pivot_properties: PreconditionerProperties,
        /,
        *,
        materializable: bool,
    ) -> tuple[AbstractLinearOperator, bool]:
        if self.schur_setup_operator is not None:
            _validate_schur_setup(self.schur_setup_operator, diagonal.source)
            return self.schur_setup_operator, False
        properties = _schur_properties(
            diagonal,
            lower,
            upper,
            pivot_properties,
        )
        identifier = canonical_fingerprint(
            {
                "kind": "planned-schur-setup",
                "diagonal": diagonal.operator_id,
                "lower": lower.operator_id,
                "upper": upper.operator_id,
                "pivot_solver": _source_identifier(self.pivot_solver),
            }
        )
        if materializable:
            return (
                FunctionLinearOperator(
                    diagonal.mv,
                    source=diagonal.source,
                    target=diagonal.target,
                    properties=properties,
                    operator_id=identifier,
                ),
                True,
            )
        return (
            _PlanningSchurLinearOperator(
                diagonal,
                properties=properties,
                operator_id=identifier,
            ),
            True,
        )

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        _, pivot_setup, upper, lower, diagonal = _block_components(setup_operator)
        pivot_properties = _source_properties(self.pivot_solver, pivot_setup)
        schur_setup, _ = self._planning_schur_setup(
            diagonal,
            lower,
            upper,
            pivot_properties,
            materializable=True,
        )
        schur_properties = _source_properties(self.schur_solver, schur_setup)
        return _component_properties(
            pivot_properties,
            schur_properties,
            self.form,
            _is_adjoint_pair(lower, upper),
            self.properties,
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        block, pivot_setup, upper, lower, diagonal = _block_components(setup_operator)
        self.properties_for(block)
        pivot_cost = _source_cost(
            self.pivot_solver, pivot_setup, materialization=materialization
        )
        pivot_properties = _source_properties(self.pivot_solver, pivot_setup)
        schur_setup, derived_schur = self._planning_schur_setup(
            diagonal,
            lower,
            upper,
            pivot_properties,
            materializable=True,
        )
        schur_cost = _source_cost(
            self.schur_solver, schur_setup, materialization=materialization
        )
        schur_matvecs = schur_cost.setup_matvec_count
        if derived_schur:
            forward_setup, _ = self._planning_schur_setup(
                diagonal,
                lower,
                upper,
                pivot_properties,
                materializable=False,
            )
            forward_cost = _source_cost(
                self.schur_solver,
                forward_setup,
                materialization=materialization,
            )
            if schur_cost.accepted and not forward_cost.accepted:
                schur_matvecs = max(schur_matvecs, diagonal.source.size)
        storage = (
            pivot_cost.storage_bytes
            + schur_cost.storage_bytes
            + _array_tree_storage_bytes((diagonal, lower, upper))
        )
        itemsize = _coordinate_dtype(block.source).itemsize
        workspace = (2 if self.form == "ldu" else 1) * block.source.size * itemsize
        accepted = pivot_cost.accepted and schur_cost.accepted
        reason = (
            "pivot and Schur actions plus retained Schur operator state"
            if accepted
            else "; ".join(
                estimate.reason
                for estimate in (pivot_cost, schur_cost)
                if not estimate.accepted
            )
        )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=storage,
            preparation_workspace_bytes=(
                pivot_cost.preparation_workspace_bytes
                + schur_cost.preparation_workspace_bytes
            ),
            apply_workspace_bytes_per_rhs=(
                pivot_cost.apply_workspace_bytes_per_rhs
                + schur_cost.apply_workspace_bytes_per_rhs
                + workspace
            ),
            setup_matvec_count=pivot_cost.setup_matvec_count + schur_matvecs,
            accepted=accepted,
            reason=reason,
        )

    def _build(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
        previous: BlockFactorizationPreconditioner | None,
    ) -> BlockFactorizationPreconditioner:
        block, pivot_setup, upper, lower, diagonal = _block_components(setup_operator)
        if previous is None:
            pivot_action = _prepare_source(
                self.pivot_solver,
                pivot_setup,
                materialization=materialization,
                component="pivot",
            )
        else:
            pivot_action = _refresh_source(
                self.pivot_solver,
                previous.pivot_action,
                pivot_setup,
                materialization=materialization,
                component="pivot",
            )
        schur_operator = SchurComplementLinearOperator(
            diagonal,
            lower,
            pivot_action,
            upper,
        )
        schur_setup = (
            _materializable_schur_setup(schur_operator)
            if self.schur_setup_operator is None
            else self.schur_setup_operator
        )
        _validate_schur_setup(schur_setup, diagonal.source)
        if previous is None:
            schur_action = _prepare_source(
                self.schur_solver,
                schur_setup,
                materialization=materialization,
                component="Schur",
            )
        else:
            schur_action = _refresh_source(
                self.schur_solver,
                previous.schur_action,
                schur_setup,
                materialization=materialization,
                component="Schur",
            )
        properties = _component_properties(
            pivot_action.properties,
            schur_action.properties,
            self.form,
            _is_adjoint_pair(lower, upper),
            self.properties,
        )
        return BlockFactorizationPreconditioner(
            schur_operator,
            schur_action,
            self.form,
            properties=properties,
            space=block.source,
            preconditioner_id=canonical_fingerprint(
                {
                    "kind": "prepared-block-factorization",
                    "builder": self.builder_id,
                    "setup_operator": block.operator_id,
                    "pivot_action": pivot_action.preconditioner_id,
                    "schur_operator": schur_operator.operator_id,
                    "schur_action": schur_action.preconditioner_id,
                }
            ),
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        self.properties_for(setup_operator)
        return self._build(
            setup_operator,
            materialization=materialization,
            previous=None,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, BlockFactorizationPreconditioner):
            raise TypeError(
                "Block factorization refresh requires a BlockFactorizationPreconditioner."
            )
        if preconditioner.form != self.form:
            raise ValueError("Block factorization refresh must preserve its form.")
        self.properties_for(setup_operator)
        return self._build(
            setup_operator,
            materialization=materialization,
            previous=preconditioner,
        )


__all__ = [
    "BlockFactorizationForm",
    "BlockFactorizationPreconditioner",
    "BlockFactorizationPreconditionerBuilder",
]
