#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import assert_never, final, Literal, NoReturn, TypeAlias

import equinox as eqx
import jax
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._trainable import fixed_field
from ..typing import checked, parse
from ._costs import _array_tree_storage_bytes, PreconditionerCostEstimate
from ._materialization import MaterializationPolicy
from ._named_blocks import BlockSelection, select_block_operator
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
    PreconditionerRefreshPolicy,
    PreconditionerSource,
)
from ._properties import OperatorCapabilities, OperatorProperties
from ._spaces import AbstractVectorSpace, BlockSpace, coordinate_dtype
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
        "pivot_solver and schur_solver must be AbstractPreconditioner values or AbstractPreconditionerBuilder values."
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
    space: AbstractVectorSpace,
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


def _zero_operator(space: AbstractVectorSpace, /) -> AbstractLinearOperator:
    return ScaledLinearOperator(IdentityLinearOperator(space), 0.0)


def _zero_block_action(
    space: AbstractVectorSpace,
    _vector: PyTree[Array],
    /,
) -> PyTree[Array]:
    return space.zeros()


def _zero_block(
    source: AbstractVectorSpace, target: AbstractVectorSpace, /
) -> AbstractLinearOperator:
    # Keep the callable structural and the scientific space leaves dynamic.
    # Refresh must not create a new static closure-converted Jaxpr each time.
    action = eqx.Partial(_zero_block_action, target)
    transpose = eqx.Partial(_zero_block_action, source)
    return FunctionLinearOperator(
        action,
        source=source,
        target=target,
        transpose_action=transpose,
        closure_convert=False,
        operator_id=canonical_fingerprint(
            {
                "kind": "structural-zero-block",
                "source": source.space_id,
                "target": target.space_id,
            }
        ),
    )


def _block_components(
    setup_operator: AbstractLinearOperator,
    /,
    *,
    form: BlockFactorizationForm,
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
    if (
        form == "diagonal"
        and setup_operator.properties.certifies("block_diagonal")
        and upper is None
        and lower is None
    ):
        lower = _zero_block(
            setup_operator.source.spaces[0], setup_operator.target.spaces[1]
        )
        upper = AdjointLinearOperator(lower)
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
    dual_space: AbstractVectorSpace,
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
    # Diagonal composition is a direct sum; paired LDU is a congruence
    # transformation of the two component actions. Both preserve SPD, but
    # only with the already-certified fixed/self-adjoint component semantics.
    # Native factor actions remain conditional on their SUCCESS branch;
    # this transformation neither proves factor success nor bypasses refusal.
    positive_definite = (
        self_adjoint
        and pivot.certifies("positive_definite")
        and schur.certifies("positive_definite")
    )
    if supplied is None:
        claims = {
            "linear": linear,
            "stationary": stationary,
            "self_adjoint": self_adjoint,
            "positive_definite": positive_definite,
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
            "Certified block properties require both actions to certify the corresponding fixed linear semantics."
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
            "A positive-definite block claim requires certified positive-definite component actions."
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
    ) -> None:
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

    def _materialize(self, /) -> NoReturn:
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


def _action_couplings(
    form: BlockFactorizationForm,
    lower: AbstractLinearOperator,
    upper: AbstractLinearOperator,
    /,
) -> tuple[AbstractLinearOperator | None, AbstractLinearOperator | None]:
    """Retain only couplings consumed by the declared execution form."""
    match form:
        case "diagonal":
            return None, None
        case "lower":
            return lower, None
        case "upper":
            return None, upper
        case "ldu":
            return lower, upper
        case _:
            assert_never(form)


class BlockFactorizationPreconditioner(AbstractPreconditioner):
    """Prepared pivot/Schur actions and only execution-required couplings."""

    pivot_action: AbstractPreconditioner
    schur_action: AbstractPreconditioner
    lower_block: AbstractLinearOperator | None
    upper_block: AbstractLinearOperator | None
    schur_operator_id: str = eqx.field(static=True)
    schur_input_ids: tuple[str, str, str, str] = eqx.field(static=True)
    form: BlockFactorizationForm = eqx.field(static=True)
    space: BlockSpace = fixed_field()
    __hash__ = object.__hash__

    @checked
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
    ) -> None:
        form = parse(form, BlockFactorizationForm, "form")
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
        schur_input_ids = (
            schur_operator.diagonal_block.operator_id,
            schur_operator.lower_block.operator_id,
            pivot_action.preconditioner_id,
            schur_operator.upper_block.operator_id,
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "block-factorization-action",
                    "form": form,
                    "schur_operator": schur_operator.operator_id,
                    "schur_input_ids": schur_input_ids,
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
        self.pivot_action = pivot_action
        self.lower_block, self.upper_block = _action_couplings(
            form, schur_operator.lower_block, schur_operator.upper_block
        )
        self.schur_operator_id = schur_operator.operator_id
        self.schur_input_ids = schur_input_ids
        self.schur_action = schur_action
        self.form = form
        self.space = space_
        self.properties = properties_
        self.preconditioner_id = identifier

    @property
    def preparation_failed(self) -> Array:
        return self.pivot_action.preparation_failed | self.schur_action.preparation_failed

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        block, pivot_setup, upper, lower, diagonal = _block_components(
            setup_operator, form=self.form
        )
        if not block.source.compatible(self.space):
            raise ValueError(
                "Block action cost requires a setup operator on the prepared space."
            )
        pivot_cost = _source_cost(
            self.pivot_action,
            pivot_setup,
            materialization=materialization,
        )
        current_input_ids = (
            diagonal.operator_id,
            lower.operator_id,
            self.pivot_action.preconditioner_id,
            upper.operator_id,
        )
        schur_setup = SchurComplementLinearOperator(
            diagonal,
            lower,
            self.pivot_action,
            upper,
            operator_id=(
                self.schur_operator_id
                if current_input_ids == self.schur_input_ids
                else None
            ),
        )
        schur_cost = _source_cost(
            self.schur_action,
            schur_setup,
            materialization=materialization,
        )
        accepted = pivot_cost.accepted and schur_cost.accepted
        workspace_multiplier = 2 if self.form == "ldu" else 1
        workspace = (
            workspace_multiplier * self.space.size * coordinate_dtype(self.space).itemsize
        )
        return PreconditionerCostEstimate(
            component=self.preconditioner_id,
            storage_bytes=(
                pivot_cost.storage_bytes
                + schur_cost.storage_bytes
                + _array_tree_storage_bytes((self.lower_block, self.upper_block))
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
                "prepared pivot and Schur actions plus execution-required couplings"
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
        upper = self.upper_block
        lower = self.lower_block
        match self.form:
            case "diagonal":
                return (
                    pivot.apply(first, iteration=iteration),
                    schur.apply(second, iteration=iteration),
                )
            case "lower":
                if lower is None:
                    raise RuntimeError(
                        "A lower block action requires its lower coupling."
                    )
                first_solution = pivot.apply(first, iteration=iteration)
                second_rhs = _tree_subtract(second, lower.mv(first_solution))
                return first_solution, schur.apply(second_rhs, iteration=iteration)
            case "upper":
                if upper is None:
                    raise RuntimeError(
                        "An upper block action requires its upper coupling."
                    )
                second_solution = schur.apply(second, iteration=iteration)
                first_rhs = _tree_subtract(first, upper.mv(second_solution))
                return pivot.apply(first_rhs, iteration=iteration), second_solution
            case "ldu":
                if lower is None or upper is None:
                    raise RuntimeError("An LDU block action requires both couplings.")
                pivot_rhs = pivot.apply(first, iteration=iteration)
                second_rhs = _tree_subtract(second, lower.mv(pivot_rhs))
                second_solution = schur.apply(second_rhs, iteration=iteration)
                first_correction = pivot.apply(
                    upper.mv(second_solution), iteration=iteration
                )
                return _tree_subtract(pivot_rhs, first_correction), second_solution
            case _:
                assert_never(self.form)


class BlockFactorizationPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare diagonal, triangular, or LDU actions from a 2-by-2 block operator."""

    pivot_solver: PreconditionerSource
    schur_solver: PreconditionerSource
    form: BlockFactorizationForm = eqx.field(static=True)
    schur_setup_operator: AbstractLinearOperator | None = fixed_field()
    properties: PreconditionerProperties | None = fixed_field()
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        pivot_solver: PreconditionerSource,
        schur_solver: PreconditionerSource,
        form: BlockFactorizationForm,
        *,
        schur_setup_operator: AbstractLinearOperator | None = None,
        properties: PreconditionerProperties | None = None,
    ) -> None:
        pivot_id = _source_identifier(pivot_solver)
        schur_id = _source_identifier(schur_solver)
        form = parse(form, BlockFactorizationForm, "form")
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
                "component_property_rule": "fixed-self-adjoint-spd-congruence:structural-zero-block",
                "prepared_state": "execution-required-action",
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
        block_diagonal: bool,
    ) -> tuple[AbstractLinearOperator, bool]:
        if self.schur_setup_operator is not None:
            _validate_schur_setup(self.schur_setup_operator, diagonal.source)
            return self.schur_setup_operator, False
        if self.form == "diagonal" and block_diagonal:
            return diagonal, False
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
        block, pivot_setup, upper, lower, diagonal = _block_components(
            setup_operator, form=self.form
        )
        pivot_properties = _source_properties(self.pivot_solver, pivot_setup)
        schur_setup, _ = self._planning_schur_setup(
            diagonal,
            lower,
            upper,
            pivot_properties,
            materializable=True,
            block_diagonal=block.properties.certifies("block_diagonal"),
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
        block, pivot_setup, upper, lower, diagonal = _block_components(
            setup_operator, form=self.form
        )
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
            block_diagonal=block.properties.certifies("block_diagonal"),
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
                block_diagonal=block.properties.certifies("block_diagonal"),
            )
            forward_cost = _source_cost(
                self.schur_solver,
                forward_setup,
                materialization=materialization,
            )
            if schur_cost.accepted and not forward_cost.accepted:
                schur_matvecs = max(schur_matvecs, diagonal.source.size)
        retained_couplings = _action_couplings(self.form, lower, upper)
        retained_coupling_bytes = _array_tree_storage_bytes(retained_couplings)
        setup_only_bytes = (
            _array_tree_storage_bytes((diagonal, lower, upper)) - retained_coupling_bytes
        )
        storage = (
            pivot_cost.storage_bytes + schur_cost.storage_bytes + retained_coupling_bytes
        )
        itemsize = coordinate_dtype(block.source).itemsize
        workspace = (2 if self.form == "ldu" else 1) * block.source.size * itemsize
        accepted = pivot_cost.accepted and schur_cost.accepted
        reason = (
            "pivot and Schur actions plus execution-required couplings"
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
                + setup_only_bytes
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
        block, pivot_setup, upper, lower, diagonal = _block_components(
            setup_operator, form=self.form
        )
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
        if self.schur_setup_operator is not None:
            schur_setup = self.schur_setup_operator
        elif self.form == "diagonal" and block.properties.certifies("block_diagonal"):
            schur_setup = diagonal
        else:
            schur_setup = _materializable_schur_setup(schur_operator)
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
    ) -> BlockFactorizationPreconditioner:
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
    ) -> BlockFactorizationPreconditioner:
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


def _mapped_properties(
    selection: BlockSelection, properties: PreconditionerProperties, /
) -> PreconditionerProperties:
    claims = {
        "linear": properties.certifies("linear"),
        "stationary": properties.certifies("stationary"),
        "self_adjoint": selection.preserves_pairing
        and properties.certifies("self_adjoint"),
        "positive_definite": selection.preserves_pairing
        and properties.certifies("positive_definite"),
    }
    return PreconditionerProperties(
        **claims,
        evidence={name: "transformed" for name, claimed in claims.items() if claimed},
    )


def _mapped_setup(
    selection: BlockSelection, operator: AbstractLinearOperator, /
) -> BlockLinearOperator:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("setup_operator must be an AbstractLinearOperator.")
    if not operator.source.compatible(operator.target):
        raise ValueError("Mapped block setup must be an endomorphism.")
    return select_block_operator(operator, selection, selection)


@final
class MappedBlockFactorizationPreconditioner(AbstractPreconditioner):
    """Complete canonical restriction/action/prolongation on the native space."""

    selection: BlockSelection = fixed_field()
    action: BlockFactorizationPreconditioner

    def __init__(
        self,
        selection: BlockSelection,
        action: BlockFactorizationPreconditioner,
        /,
    ) -> None:
        if not isinstance(selection, BlockSelection):
            raise TypeError("selection must be a BlockSelection.")
        if not selection.covers_source:
            raise ValueError(
                "Mapped block preconditioning requires a complete selection."
            )
        if not isinstance(action, BlockFactorizationPreconditioner):
            raise TypeError("action must be a BlockFactorizationPreconditioner.")
        if not selection.target.compatible(action.space):
            raise ValueError("Selection target must match the prepared block action.")
        self.selection = selection
        self.action = action
        self.space = selection.source
        self.properties = _mapped_properties(selection, action.properties)
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "mapped-block-factorization-preconditioner",
                "selection": selection.selection_id,
                "action": action.preconditioner_id,
            }
        )

    @property
    def preparation_failed(self) -> Array:
        return self.action.preparation_failed

    def apply(
        self,
        residual: PyTree[ArrayLike],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        selected = self.selection.restrict(residual)
        return self.selection.prolong(self.action.apply(selected, iteration=iteration))

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        selected = _mapped_setup(self.selection, setup_operator)
        cost = self.action.cost_for(selected, materialization=materialization)
        return _mapped_cost(self.preconditioner_id, self.selection, cost)


def _mapped_cost(
    identifier: str, selection: BlockSelection, cost: PreconditionerCostEstimate, /
) -> PreconditionerCostEstimate:
    mapping_workspace = (
        2 * selection.source.size * coordinate_dtype(selection.source).itemsize
    )
    return PreconditionerCostEstimate(
        component=identifier,
        storage_bytes=cost.storage_bytes + _array_tree_storage_bytes(selection),
        preparation_workspace_bytes=cost.preparation_workspace_bytes,
        apply_workspace_bytes_per_rhs=cost.apply_workspace_bytes_per_rhs
        + mapping_workspace,
        setup_matvec_count=cost.setup_matvec_count,
        accepted=cost.accepted,
        reason=cost.reason,
    )


@final
class MappedBlockFactorizationPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Adapt one native complete partition to the existing exact 2-by-2 LDU recipe."""

    selection: BlockSelection = fixed_field()
    builder: BlockFactorizationPreconditionerBuilder
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        selection: BlockSelection,
        builder: BlockFactorizationPreconditionerBuilder,
        /,
    ) -> None:
        if not isinstance(selection, BlockSelection):
            raise TypeError("selection must be a BlockSelection.")
        if not selection.covers_source or len(selection.target.spaces) != 2:
            raise ValueError(
                "Mapped block factorization requires a complete two-member selection."
            )
        if not isinstance(builder, BlockFactorizationPreconditionerBuilder):
            raise TypeError("builder must be a BlockFactorizationPreconditionerBuilder.")
        self.selection = selection
        self.builder = builder
        self._builder_id = canonical_fingerprint(
            {
                "kind": "mapped-block-factorization-preconditioner-builder",
                "selection": selection.selection_id,
                "builder": builder.builder_id,
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> PreconditionerRefreshPolicy:
        return "numeric"

    def properties_for(
        self, setup_operator: AbstractLinearOperator, /
    ) -> PreconditionerProperties:
        selected = _mapped_setup(self.selection, setup_operator)
        return _mapped_properties(self.selection, self.builder.properties_for(selected))

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        selected = _mapped_setup(self.selection, setup_operator)
        cost = self.builder.cost_for(selected, materialization=materialization)
        return _mapped_cost(self.builder_id, self.selection, cost)

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> MappedBlockFactorizationPreconditioner:
        selected = _mapped_setup(self.selection, setup_operator)
        action = self.builder.prepare(selected, materialization=materialization)
        if not isinstance(action, BlockFactorizationPreconditioner):
            raise TypeError(
                "Block factorization builder returned an incompatible action."
            )
        return MappedBlockFactorizationPreconditioner(self.selection, action)

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> MappedBlockFactorizationPreconditioner:
        if not isinstance(preconditioner, MappedBlockFactorizationPreconditioner):
            raise TypeError(
                "Mapped refresh requires a mapped block factorization action."
            )
        if preconditioner.selection.selection_id != self.selection.selection_id:
            raise ValueError(
                "Mapped refresh must preserve the complete coordinate selection."
            )
        selected = _mapped_setup(self.selection, setup_operator)
        action = self.builder.refresh(
            preconditioner.action, selected, materialization=materialization
        )
        if not isinstance(action, BlockFactorizationPreconditioner):
            raise TypeError(
                "Block factorization refresh returned an incompatible action."
            )
        return MappedBlockFactorizationPreconditioner(self.selection, action)


__all__ = [
    "BlockFactorizationForm",
    "BlockFactorizationPreconditioner",
    "BlockFactorizationPreconditionerBuilder",
    "MappedBlockFactorizationPreconditioner",
    "MappedBlockFactorizationPreconditionerBuilder",
]
