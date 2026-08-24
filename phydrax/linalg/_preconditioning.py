#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._assembly import (
    assemble_diagonal,
    assemble_uniform_blocks,
    plan_sparse_assembly,
    SparseAssemblyPlan,
    SparseAssemblyPolicy,
)
from ._costs import PreconditionerCostEstimate
from ._materialization import MaterializationPolicy, materialize
from ._operators import (
    AbstractLinearOperator,
    AdjointLinearOperator,
    BlockLinearOperator,
    ComposedLinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
    IdentityLinearOperator,
    ScaledLinearOperator,
    SumLinearOperator,
    TransposeLinearOperator,
)
from ._preconditioner_properties import (
    _preconditioner_properties_payload,
    PreconditionerProperties,
)
from ._preconditioners import (
    _CostedPreconditioner,
    _prepared_action_cost,
    AbstractPreconditioner,
    BlockDiagonalPreconditioner,
    DiagonalPreconditioner,
    LocalBlockPreconditioner,
    PrecisionCastPreconditioner,
)
from ._properties import LinearCapabilityError
from ._spaces import (
    _coordinate_dtype,
    _has_diagonal_pairing,
    ArraySpace,
    DiagonalPairing,
    EuclideanPairing,
)
from ._sparse_contract import AbstractSparseLinearOperator
from ._structured_operators import LocalBlockDiagonalLinearOperator


PreconditioningSide: TypeAlias = Literal["auto", "left", "right"]
PreconditionerRefreshPolicy: TypeAlias = Literal["frozen", "numeric", "rebuild"]
PreconditionerRefreshKind: TypeAlias = Literal[
    "prepared", "supplied", "reused", "refreshed", "rebuilt"
]


def _dense_materialization_eligibility(
    operator: AbstractLinearOperator,
    policy: MaterializationPolicy | None,
    /,
) -> tuple[bool, str]:
    if not operator.capabilities.materialize:
        return False, "operator does not support required dense materialization"
    if policy is None:
        return True, "dense materialization capability is available"
    if not isinstance(policy, MaterializationPolicy):
        raise TypeError("materialization must be a MaterializationPolicy or None.")
    entries = operator.source.size * operator.target.size
    required_bytes = entries * _coordinate_dtype(operator.source).itemsize
    if entries > policy.max_entries:
        return (
            False,
            f"dense materialization requires {entries} entries, exceeding "
            f"the policy limit {policy.max_entries}",
        )
    if required_bytes > policy.max_bytes:
        return (
            False,
            f"dense materialization requires {required_bytes} bytes, exceeding "
            f"the policy limit {policy.max_bytes}",
        )
    return True, "dense materialization fits the active policy"


def _materialization_matvec_count(
    operator: AbstractLinearOperator,
    /,
) -> int:
    if isinstance(
        operator,
        (
            DenseLinearOperator,
            DiagonalLinearOperator,
            IdentityLinearOperator,
            AbstractSparseLinearOperator,
        ),
    ):
        return 0
    if isinstance(
        operator,
        (ScaledLinearOperator, TransposeLinearOperator, AdjointLinearOperator),
    ):
        return _materialization_matvec_count(operator.operator)
    if isinstance(operator, (SumLinearOperator, ComposedLinearOperator)):
        return _materialization_matvec_count(
            operator.left
        ) + _materialization_matvec_count(operator.right)
    if isinstance(operator, BlockLinearOperator):
        return sum(
            _materialization_matvec_count(block)
            for row in operator.blocks
            for block in row
            if block is not None
        )
    return operator.source.size


class AbstractPreconditionerBuilder(StrictModule):
    """Symbolic recipe that prepares an approximate inverse from a setup operator."""

    @property
    @abc.abstractmethod
    def builder_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def default_refresh(self) -> PreconditionerRefreshPolicy:
        raise NotImplementedError

    @abc.abstractmethod
    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        raise NotImplementedError

    @abc.abstractmethod
    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        raise NotImplementedError

    @abc.abstractmethod
    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        raise NotImplementedError


class DenseInversePreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare an exact dense inverse, principally for small coarse spaces."""

    _builder_id: str = eqx.field(static=True)

    def __init__(self):
        self._builder_id = canonical_fingerprint(
            {"kind": "dense-inverse-preconditioner-builder"}
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> PreconditionerRefreshPolicy:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        _validate_setup_operator(setup_operator)
        positive = setup_operator.properties.certifies("positive_definite")
        claims = {
            "linear": True,
            "stationary": True,
            "self_adjoint": positive,
            "positive_definite": positive,
        }
        return PreconditionerProperties(
            **claims,
            evidence={name: "transformed" for name, claimed in claims.items() if claimed},
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        _validate_setup_operator(setup_operator)
        itemsize = _coordinate_dtype(setup_operator.source).itemsize
        entries = setup_operator.source.size * setup_operator.target.size
        accepted, materialization_reason = _dense_materialization_eligibility(
            setup_operator,
            materialization,
        )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=entries * itemsize,
            preparation_workspace_bytes=entries * itemsize,
            apply_workspace_bytes_per_rhs=setup_operator.source.size * itemsize,
            setup_matvec_count=_materialization_matvec_count(setup_operator),
            accepted=accepted,
            reason=(
                "dense inverse storage and factorization workspace"
                if accepted
                else materialization_reason
            ),
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        matrix = materialize(setup_operator, materialization)
        properties = self.properties_for(setup_operator)
        return BlockDiagonalPreconditioner(
            (matrix,),
            space=setup_operator.source,
            positive_definite=properties.certifies("positive_definite"),
            preconditioner_id=canonical_fingerprint(
                {
                    "kind": "prepared-dense-inverse",
                    "builder": self.builder_id,
                    "setup_operator": setup_operator.operator_id,
                }
            ),
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, BlockDiagonalPreconditioner):
            raise TypeError(
                "Dense inverse refresh requires a BlockDiagonalPreconditioner."
            )
        return self.prepare(setup_operator, materialization=materialization)


class JacobiPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare a damped Jacobi inverse from an operator diagonal."""

    relaxation: float = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(self, *, relaxation: float = 1.0):
        value = float(relaxation)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("relaxation must be finite and positive.")
        self.relaxation = value
        self._builder_id = canonical_fingerprint(
            {"kind": "jacobi-preconditioner-builder", "relaxation": value}
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> PreconditionerRefreshPolicy:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        _validate_setup_operator(setup_operator)
        positive = setup_operator.properties.certifies(
            "positive_definite"
        ) and _has_diagonal_pairing(setup_operator.source)
        claims = {
            "linear": True,
            "stationary": True,
            "self_adjoint": positive,
            "positive_definite": positive,
        }
        return PreconditionerProperties(
            **claims,
            evidence={name: "transformed" for name, claimed in claims.items() if claimed},
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        _validate_setup_operator(setup_operator)
        itemsize = _coordinate_dtype(setup_operator.source).itemsize
        dimension = setup_operator.source.size
        direct_assembly = setup_operator.capabilities.diagonal_assembly
        accepted = True
        materialization_reason = ""
        if not direct_assembly:
            accepted, materialization_reason = _dense_materialization_eligibility(
                setup_operator,
                materialization,
            )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=dimension * itemsize,
            preparation_workspace_bytes=(
                dimension * itemsize
                if direct_assembly
                else dimension * dimension * itemsize
            ),
            apply_workspace_bytes_per_rhs=dimension * itemsize,
            setup_matvec_count=(
                0 if direct_assembly else _materialization_matvec_count(setup_operator)
            ),
            accepted=accepted,
            reason=(
                "Jacobi diagonal extraction and inverse storage"
                if accepted
                else materialization_reason
            ),
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        diagonal = assemble_diagonal(
            setup_operator,
            materialization=materialization,
        )
        properties = self.properties_for(setup_operator)
        return DiagonalPreconditioner(
            diagonal / self.relaxation,
            space=setup_operator.source,
            positive_definite=(
                True if properties.certifies("positive_definite") else None
            ),
            preconditioner_id=canonical_fingerprint(
                {
                    "kind": "prepared-jacobi",
                    "builder": self.builder_id,
                    "setup_operator": setup_operator.operator_id,
                }
            ),
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, DiagonalPreconditioner):
            raise TypeError("Jacobi refresh requires a DiagonalPreconditioner.")
        return self.prepare(setup_operator, materialization=materialization)


class BlockJacobiPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare fixed-size block Jacobi factors from exact canonical blocks."""

    block_size: int = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    assembly: SparseAssemblyPolicy | None
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        block_size: int,
        /,
        *,
        relaxation: float = 1.0,
        assembly: SparseAssemblyPolicy | None = None,
    ):
        size = int(block_size)
        if size < 1:
            raise ValueError("block_size must be positive.")
        relaxation_ = float(relaxation)
        if not np.isfinite(relaxation_) or relaxation_ <= 0.0:
            raise ValueError("relaxation must be finite and positive.")
        if assembly is not None and not isinstance(
            assembly,
            SparseAssemblyPolicy,
        ):
            raise TypeError("assembly must be a SparseAssemblyPolicy or None.")
        self.block_size = size
        self.relaxation = relaxation_
        self.assembly = assembly
        self._builder_id = canonical_fingerprint(
            {
                "kind": "block-jacobi-preconditioner-builder",
                "block_size": size,
                "relaxation": relaxation_,
                "assembly": _sparse_assembly_policy_payload(assembly),
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> PreconditionerRefreshPolicy:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        _validate_setup_operator(setup_operator)
        positive = setup_operator.properties.certifies(
            "positive_definite"
        ) and _has_diagonal_pairing(setup_operator.source)
        claims = {
            "linear": True,
            "stationary": True,
            "self_adjoint": positive,
            "positive_definite": positive,
        }
        return PreconditionerProperties(
            **claims,
            evidence={name: "transformed" for name, claimed in claims.items() if claimed},
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        _validate_setup_operator(setup_operator)
        dimension = setup_operator.source.size
        if dimension % self.block_size:
            return PreconditionerCostEstimate(
                component=self.builder_id,
                accepted=False,
                reason=(
                    f"block size {self.block_size} does not divide operator "
                    f"dimension {dimension}"
                ),
            )
        itemsize = _coordinate_dtype(setup_operator.source).itemsize
        real_itemsize = jnp.empty(
            (),
            dtype=_coordinate_dtype(setup_operator.source),
        ).real.dtype.itemsize
        num_blocks = dimension // self.block_size
        block_bytes = dimension * self.block_size * itemsize
        storage_bytes = (
            block_bytes
            + dimension * jnp.dtype(jnp.int32).itemsize
            + dimension * real_itemsize
            + num_blocks * jnp.dtype(bool).itemsize
        )
        try:
            assembly_plan = _block_jacobi_assembly_plan(
                self,
                setup_operator,
                materialization,
            )
        except LinearCapabilityError as error:
            return PreconditionerCostEstimate(
                component=self.builder_id,
                storage_bytes=storage_bytes,
                preparation_workspace_bytes=2 * block_bytes,
                apply_workspace_bytes_per_rhs=3 * dimension * itemsize,
                accepted=False,
                reason=str(error),
            )
        assembly_workspace = (
            0
            if assembly_plan is None
            else max(
                assembly_plan.cost.output_bytes,
                assembly_plan.cost.recipe_bytes,
                assembly_plan.cost.symbolic_workspace_bytes,
                assembly_plan.cost.numeric_workspace_bytes,
            )
        )
        setup_matvec_count = (
            _materialization_matvec_count(setup_operator)
            if assembly_plan is not None and assembly_plan.uses_materialization
            else 0
        )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=storage_bytes,
            preparation_workspace_bytes=max(
                2 * block_bytes,
                assembly_workspace,
            ),
            apply_workspace_bytes_per_rhs=3 * dimension * itemsize,
            setup_matvec_count=setup_matvec_count,
            reason=(
                f"exact {self.block_size}-coordinate block extraction and "
                "batched factorization"
            ),
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        blocks = assemble_uniform_blocks(
            setup_operator,
            self.block_size,
            policy=self._resolved_assembly_policy(materialization),
        )
        properties = self.properties_for(setup_operator)
        return LocalBlockPreconditioner(
            blocks,
            space=setup_operator.source,
            positive_definite=properties.certifies("positive_definite"),
            relaxation=self.relaxation,
            preconditioner_id=canonical_fingerprint(
                {
                    "kind": "prepared-block-jacobi",
                    "builder": self.builder_id,
                    "setup_operator": setup_operator.operator_id,
                }
            ),
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, LocalBlockPreconditioner):
            raise TypeError("Block Jacobi refresh requires a LocalBlockPreconditioner.")
        return self.prepare(
            setup_operator,
            materialization=materialization,
        )

    def _resolved_assembly_policy(
        self,
        materialization: MaterializationPolicy | None,
        /,
    ) -> SparseAssemblyPolicy:
        if self.assembly is not None:
            return self.assembly
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        return SparseAssemblyPolicy(materialization=materialization_)


def _block_jacobi_assembly_plan(
    builder: BlockJacobiPreconditionerBuilder,
    setup_operator: AbstractLinearOperator,
    materialization: MaterializationPolicy | None,
    /,
) -> SparseAssemblyPlan | None:
    if isinstance(setup_operator, DenseLinearOperator):
        return None
    if (
        isinstance(setup_operator, LocalBlockDiagonalLinearOperator)
        and setup_operator.input_block_size == builder.block_size
        and setup_operator.output_block_size == builder.block_size
    ):
        return None
    return plan_sparse_assembly(
        setup_operator,
        builder._resolved_assembly_policy(materialization),
    )


def _sparse_assembly_policy_payload(
    policy: SparseAssemblyPolicy | None,
    /,
) -> dict[str, object] | None:
    if policy is None:
        return None
    materialization = policy.materialization
    return {
        "max_nnz": policy.max_nnz,
        "max_bytes": policy.max_bytes,
        "max_contributions": policy.max_contributions,
        "max_workspace_bytes": policy.max_workspace_bytes,
        "materialization": (
            None
            if materialization is None
            else {
                "max_entries": materialization.max_entries,
                "max_bytes": materialization.max_bytes,
            }
        ),
    }


PreconditionerSource: TypeAlias = AbstractPreconditioner | AbstractPreconditionerBuilder


def _source_cost(
    source: PreconditionerSource,
    setup_operator: AbstractLinearOperator,
    /,
    *,
    materialization: MaterializationPolicy | None = None,
) -> PreconditionerCostEstimate:
    if isinstance(source, AbstractPreconditioner):
        estimate = (
            source.cost_for(setup_operator, materialization=materialization)
            if isinstance(source, _CostedPreconditioner)
            else _prepared_action_cost(source, setup_operator)
        )
    else:
        estimate = source.cost_for(
            setup_operator,
            materialization=materialization,
        )
    if not isinstance(estimate, PreconditionerCostEstimate):
        raise TypeError("Preconditioner cost_for must return PreconditionerCostEstimate.")
    return estimate


class PreconditioningPolicy(StrictModule):
    """Preconditioner source, setup operator, application side, and refresh contract."""

    preconditioner: AbstractPreconditioner | None
    builder: AbstractPreconditionerBuilder | None
    setup_operator: AbstractLinearOperator | None
    side: PreconditioningSide = eqx.field(static=True)
    refresh_policy: PreconditionerRefreshPolicy = eqx.field(static=True)

    def __init__(
        self,
        source: PreconditionerSource,
        /,
        *,
        setup_operator: AbstractLinearOperator | None = None,
        side: PreconditioningSide = "auto",
        refresh: PreconditionerRefreshPolicy | None = None,
    ):
        if side not in ("auto", "left", "right"):
            raise ValueError("side must be 'auto', 'left', or 'right'.")
        if refresh is not None and refresh not in ("frozen", "numeric", "rebuild"):
            raise ValueError("refresh must be 'frozen', 'numeric', or 'rebuild'.")
        if isinstance(source, AbstractPreconditioner):
            if setup_operator is not None:
                raise ValueError(
                    "setup_operator is only meaningful for a preconditioner builder."
                )
            self.preconditioner = source
            self.builder = None
            self.setup_operator = None
            self.refresh_policy = "frozen" if refresh is None else refresh
            if self.refresh_policy != "frozen":
                raise ValueError("A supplied prepared preconditioner must remain frozen.")
        elif isinstance(source, AbstractPreconditionerBuilder):
            if setup_operator is not None:
                _validate_setup_operator(setup_operator)
            self.preconditioner = None
            self.builder = source
            self.setup_operator = setup_operator
            self.refresh_policy = source.default_refresh if refresh is None else refresh
            if self.refresh_policy not in ("frozen", "numeric", "rebuild"):
                raise ValueError(
                    "Builder default_refresh must be 'frozen', 'numeric', or 'rebuild'."
                )
        else:
            raise TypeError(
                "source must be an AbstractPreconditioner or "
                "AbstractPreconditionerBuilder."
            )
        self.side = side

    def resolve_setup_operator(
        self,
        system_operator: AbstractLinearOperator,
        /,
    ) -> AbstractLinearOperator:
        if self.builder is None:
            raise ValueError("A supplied preconditioner has no setup operator.")
        setup = system_operator if self.setup_operator is None else self.setup_operator
        _validate_setup_operator(setup)
        if not setup.source.compatible(system_operator.source):
            raise ValueError(
                "The preconditioner setup operator must act on the system source space."
            )
        return setup

    def properties_for(
        self,
        system_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        if self.preconditioner is not None:
            if not self.preconditioner.space.compatible(system_operator.source):
                raise ValueError("Preconditioner space must match the operator source.")
            properties = self.preconditioner.properties
        else:
            if self.builder is None:
                raise RuntimeError("Invalid preconditioning policy state.")
            properties = self.builder.properties_for(
                self.resolve_setup_operator(system_operator)
            )
        if not isinstance(properties, PreconditionerProperties):
            raise TypeError(
                "Preconditioner sources must return PreconditionerProperties."
            )
        return properties

    def with_setup_operator(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditioningPolicy:
        if self.builder is None:
            raise ValueError(
                "A supplied preconditioner has no replaceable setup operator."
            )
        return PreconditioningPolicy(
            self.builder,
            setup_operator=setup_operator,
            side=self.side,
            refresh=self.refresh_policy,
        )


class PreconditionerPlan(StrictModule):
    """Deterministic symbolic preconditioning decision owned by a solve plan."""

    policy: PreconditioningPolicy
    side: Literal["left", "right"] = eqx.field(static=True)
    properties: PreconditionerProperties
    space_id: str = eqx.field(static=True)
    cost: PreconditionerCostEstimate
    setup_operator_id: str | None = eqx.field(static=True)
    component_id: str = eqx.field(static=True)
    compute_dtype: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        policy: PreconditioningPolicy,
        system_operator: AbstractLinearOperator,
        /,
        *,
        side: Literal["left", "right"],
        materialization: MaterializationPolicy | None = None,
        compute_dtype: str | None = None,
    ):
        if not isinstance(policy, PreconditioningPolicy):
            raise TypeError("policy must be a PreconditioningPolicy.")
        if not isinstance(system_operator, AbstractLinearOperator):
            raise TypeError("system_operator must be an AbstractLinearOperator.")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'.")
        materialization_ = (
            MaterializationPolicy() if materialization is None else materialization
        )
        if not isinstance(materialization_, MaterializationPolicy):
            raise TypeError("materialization must be a MaterializationPolicy or None.")
        compute_dtype_ = None if compute_dtype is None else jnp.dtype(compute_dtype).name
        properties = policy.properties_for(system_operator)
        setup = (
            system_operator
            if policy.preconditioner is not None
            else policy.resolve_setup_operator(system_operator)
        )
        source = (
            policy.preconditioner if policy.preconditioner is not None else policy.builder
        )
        if source is None:
            raise RuntimeError("Invalid preconditioning policy state.")
        cost = _source_cost(
            source,
            setup,
            materialization=materialization_,
        )
        if not cost.accepted:
            raise ValueError(
                f"Preconditioner {cost.component} is infeasible: {cost.reason}."
            )
        if compute_dtype_ is not None:
            low_itemsize = jnp.dtype(compute_dtype_).itemsize
            high_itemsize = _coordinate_dtype(system_operator.source).itemsize
            dimension = system_operator.source.size
            cost = PreconditionerCostEstimate(
                component=cost.component,
                storage_bytes=dimension * low_itemsize,
                preparation_workspace_bytes=(
                    cost.preparation_workspace_bytes + dimension * low_itemsize
                ),
                apply_workspace_bytes_per_rhs=dimension
                * (high_itemsize + 2 * low_itemsize),
                setup_matvec_count=cost.setup_matvec_count,
                accepted=cost.accepted,
                reason=(
                    f"{cost.reason}; stored/applied in {compute_dtype_} with "
                    "explicit coordinate casts"
                ),
            )
        if policy.preconditioner is not None:
            setup_operator_id = None
            component_id = policy.preconditioner.preconditioner_id
            builder_id = None
        else:
            if policy.builder is None:
                raise RuntimeError("Invalid preconditioning policy state.")
            setup = policy.resolve_setup_operator(system_operator)
            setup_operator_id = setup.operator_id
            component_id = policy.builder.builder_id
            builder_id = policy.builder.builder_id
        payload = {
            "kind": "preconditioner-plan",
            "space": system_operator.source.space_id,
            "component": component_id,
            "builder": builder_id,
            "setup_operator": setup_operator_id,
            "side": side,
            "refresh": policy.refresh_policy,
            "properties": _preconditioner_properties_payload(properties),
            "compute_dtype": compute_dtype_,
            "cost": {
                "storage_bytes": cost.storage_bytes,
                "preparation_workspace_bytes": cost.preparation_workspace_bytes,
                "apply_workspace_bytes_per_rhs": cost.apply_workspace_bytes_per_rhs,
                "setup_matvec_count": cost.setup_matvec_count,
            },
        }
        self.policy = policy
        self.side = side
        self.properties = properties
        self.cost = cost
        self.space_id = system_operator.source.space_id
        self.setup_operator_id = setup_operator_id
        self.component_id = component_id
        self.compute_dtype = compute_dtype_
        self.plan_id = canonical_fingerprint(payload)


class PreparedPreconditioner(StrictModule):
    """Prepared approximate inverse and auditable numeric-refresh state."""

    action: AbstractPreconditioner
    setup_operator: AbstractLinearOperator | None
    plan: PreconditionerPlan
    numeric_version: Any
    built_numeric_version: Any
    refresh_kind: PreconditionerRefreshKind = eqx.field(static=True)

    def __init__(
        self,
        action: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator | None,
        plan: PreconditionerPlan,
        /,
        *,
        numeric_version: Any,
        built_numeric_version: Any,
        refresh_kind: PreconditionerRefreshKind,
    ):
        if not isinstance(plan, PreconditionerPlan):
            raise TypeError("plan must be a PreconditionerPlan.")
        if setup_operator is not None and not isinstance(
            setup_operator, AbstractLinearOperator
        ):
            raise TypeError("setup_operator must be an AbstractLinearOperator or None.")
        if plan.setup_operator_id is None:
            if setup_operator is not None:
                raise ValueError("A supplied action plan cannot own a setup operator.")
        elif (
            setup_operator is None or setup_operator.operator_id != plan.setup_operator_id
        ):
            raise ValueError("Prepared setup operator does not match its plan.")
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        built_version = jnp.asarray(built_numeric_version, dtype=jnp.int32)
        if version.ndim != 0 or built_version.ndim != 0:
            raise ValueError("Preconditioner numeric versions must be scalar.")
        invalid = (version < 0) | (built_version < 0) | (built_version > version)
        version = eqx.error_if(
            version,
            invalid,
            "Preconditioner numeric versions must satisfy "
            "0 <= built_numeric_version <= numeric_version.",
        )
        built_version = eqx.error_if(
            built_version,
            invalid,
            "Preconditioner numeric versions must satisfy "
            "0 <= built_numeric_version <= numeric_version.",
        )
        if refresh_kind not in (
            "prepared",
            "supplied",
            "reused",
            "refreshed",
            "rebuilt",
        ):
            raise ValueError("Unknown preconditioner refresh kind.")
        _validate_prepared_action(action, plan)
        self.action = action
        self.setup_operator = setup_operator
        self.plan = plan
        self.numeric_version = version
        self.built_numeric_version = built_version
        self.refresh_kind = refresh_kind


def _precision_cast_action(
    action: AbstractPreconditioner,
    plan: PreconditionerPlan,
    /,
) -> AbstractPreconditioner:
    if plan.compute_dtype is None:
        return action
    if isinstance(action, PrecisionCastPreconditioner):
        if action.compute_dtype != plan.compute_dtype:
            raise LinearCapabilityError(
                "Prepared preconditioner precision does not match its plan."
            )
        return action
    if not isinstance(action, DiagonalPreconditioner):
        raise LinearCapabilityError(
            "Lower-precision preconditioning currently requires Jacobi diagonal state."
        )
    if not isinstance(action.space, ArraySpace):
        raise LinearCapabilityError(
            "Lower-precision Jacobi requires one ArraySpace coordinate layout."
        )
    compute_dtype = jnp.dtype(plan.compute_dtype)
    pairing = action.space.pairing
    if isinstance(pairing, DiagonalPairing):
        low_pairing = DiagonalPairing(pairing.weights.astype(compute_dtype))
    elif isinstance(pairing, EuclideanPairing):
        low_pairing = EuclideanPairing()
    else:
        raise LinearCapabilityError(
            "Lower-precision Jacobi requires Euclidean or diagonal pairing."
        )
    low_space = ArraySpace(
        action.space.shape,
        dtype=compute_dtype,
        pairing=low_pairing,
    )
    diagonal = jnp.reciprocal(action.inverse_diagonal).astype(compute_dtype)
    lowered = DiagonalPreconditioner(
        diagonal,
        space=low_space,
        positive_definite=action.properties.certifies("positive_definite"),
    )
    return PrecisionCastPreconditioner(
        lowered,
        action.space,
        compute_dtype,
    )


def prepare_preconditioner(
    plan: PreconditionerPlan | None,
    system_operator: AbstractLinearOperator,
    /,
    *,
    materialization: MaterializationPolicy,
    previous: PreparedPreconditioner | None = None,
    numeric_version: Any = 0,
) -> PreparedPreconditioner | None:
    """Prepare or refresh one solve-owned approximate inverse."""
    if plan is None:
        return None
    if previous is not None and previous.plan.plan_id != plan.plan_id:
        raise ValueError("Preconditioner refresh must preserve its symbolic plan.")
    policy = plan.policy
    if policy.preconditioner is not None:
        if previous is not None:
            action = previous.action
        else:
            action = policy.preconditioner
        setup = None
        built_version = (
            jnp.asarray(0, dtype=jnp.int32)
            if previous is None
            else previous.built_numeric_version
        )
        refresh_kind: PreconditionerRefreshKind = (
            "supplied" if previous is None else "reused"
        )
    else:
        if policy.builder is None:
            raise RuntimeError("Invalid preconditioning policy state.")
        setup = policy.resolve_setup_operator(system_operator)
        if previous is None:
            action = policy.builder.prepare(setup, materialization=materialization)
            built_version = numeric_version
            refresh_kind = "prepared"
        elif policy.refresh_policy == "frozen":
            action = previous.action
            built_version = previous.built_numeric_version
            refresh_kind = "reused"
        elif policy.refresh_policy == "numeric":
            previous_action = (
                previous.action.inner
                if isinstance(previous.action, PrecisionCastPreconditioner)
                else previous.action
            )
            action = policy.builder.refresh(
                previous_action,
                setup,
                materialization=materialization,
            )
            built_version = numeric_version
            refresh_kind = "refreshed"
        else:
            action = policy.builder.prepare(setup, materialization=materialization)
            built_version = numeric_version
            refresh_kind = "rebuilt"
    action = _precision_cast_action(action, plan)
    return PreparedPreconditioner(
        action,
        setup,
        plan,
        numeric_version=numeric_version,
        built_numeric_version=built_version,
        refresh_kind=refresh_kind,
    )


def _validate_setup_operator(operator: AbstractLinearOperator, /) -> None:
    if not isinstance(operator, AbstractLinearOperator):
        raise TypeError("setup_operator must be an AbstractLinearOperator.")
    if operator.batch_shape or not operator.source.compatible(operator.target):
        raise ValueError("A setup operator must be an unbatched endomorphism.")


def _validate_prepared_action(
    action: AbstractPreconditioner,
    plan: PreconditionerPlan,
    /,
) -> None:
    if not isinstance(action, AbstractPreconditioner):
        raise TypeError("action must be an AbstractPreconditioner.")
    if action.space.space_id != plan.space_id:
        raise ValueError(
            "Prepared preconditioner space must match the planned source space."
        )
    for property_name in (
        "linear",
        "stationary",
        "self_adjoint",
        "positive_definite",
    ):
        if plan.properties.certifies(property_name) and not action.properties.certifies(
            property_name
        ):
            raise ValueError(
                f"Prepared action does not certify planned property {property_name!r}."
            )


__all__ = [
    "AbstractPreconditionerBuilder",
    "BlockJacobiPreconditionerBuilder",
    "DenseInversePreconditionerBuilder",
    "JacobiPreconditionerBuilder",
    "PreconditionerPlan",
    "PreconditionerRefreshKind",
    "PreconditionerRefreshPolicy",
    "PreconditionerSource",
    "PreconditionerCostEstimate",
    "PreconditioningPolicy",
    "PreconditioningSide",
    "PreparedPreconditioner",
]
