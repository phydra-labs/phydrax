#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any, final, Literal, NoReturn, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked, parse
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
    AbstractVectorSpace,
    ArraySpace,
    DiagonalPairing,
    EuclideanPairing,
)
from ._sparse_contract import AbstractSparseLinearOperator
from ._structured_operators import LocalBlockDiagonalLinearOperator


if TYPE_CHECKING:
    from ._dense_pseudoinverse import DensePseudoinverseFactors
    from ._policies import RankPolicy
    from ._subspaces import LinearSubspace, NullspacePolicy


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
            f"dense materialization requires {entries} entries, exceeding the policy limit {policy.max_entries}",
        )
    if required_bytes > policy.max_bytes:
        return (
            False,
            f"dense materialization requires {required_bytes} bytes, exceeding the policy limit {policy.max_bytes}",
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


@final
class _PlannedConstruction:
    """Opaque builder-owned construction; compared and hashed by identity."""

    __slots__ = ("value",)

    def __init__(self, value: object, /) -> None:
        self.value = value


@final
class PlannedPreconditionerSetup(StrictModule):
    """A builder's cost estimate and the construction that produced it.

    ``plan_setup`` returns it and ``prepare_planned`` consumes it, so a solve
    plan that costs a builder prepares from the same construction instead of
    repeating it. ``content_id`` is the exact setup-operator identity the
    construction is valid for; a builder that cannot match it prepares anew.
    """

    cost: PreconditionerCostEstimate
    construction: _PlannedConstruction | None = eqx.field(static=True)
    content_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        cost: PreconditionerCostEstimate,
        /,
        *,
        construction: object | None = None,
        content_id: str | None = None,
    ) -> None:
        if not isinstance(cost, PreconditionerCostEstimate):
            raise TypeError("cost must be a PreconditionerCostEstimate.")
        if (construction is None) != (content_id is None):
            raise ValueError("A planned construction requires its content identity.")
        self.cost = cost
        self.construction = (
            None if construction is None else _PlannedConstruction(construction)
        )
        self.content_id = content_id

    def construction_for(self, content_id: str | None, /) -> object | None:
        """The planned construction when ``content_id`` is the planned one."""
        if self.construction is None or content_id is None:
            return None
        return self.construction.value if content_id == self.content_id else None


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

    def plan_setup(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PlannedPreconditionerSetup:
        """Cost this builder; builders with expensive setup also keep the construction."""
        return PlannedPreconditionerSetup(
            self.cost_for(setup_operator, materialization=materialization)
        )

    def prepare_planned(
        self,
        planned: PlannedPreconditionerSetup,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        """Prepare from ``plan_setup``'s construction when it matches the operator."""
        del planned
        return self.prepare(setup_operator, materialization=materialization)

    def lowered_cost(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        compute_dtype: str,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        """Cost of storing and applying this action in ``compute_dtype``.

        Only actions that are one fixed diagonal, local-block, or triangular
        solve admit lower precision: the outer solve owns every residual and
        accumulation in coordinate precision and refines the lowered action.
        Actions with inner residuals, polynomial recurrences, or nested
        hierarchies have no such contract and refuse.
        """
        del setup_operator, compute_dtype, materialization
        _refuse_lower_precision(self)

    def prepare_lowered(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        compute_dtype: str,
        materialization: MaterializationPolicy,
        previous: AbstractPreconditioner | None = None,
    ) -> AbstractPreconditioner:
        """Prepare the action stored and applied in ``compute_dtype`` coordinates."""
        del setup_operator, compute_dtype, materialization, previous
        _refuse_lower_precision(self)


def _refuse_lower_precision(builder: AbstractPreconditionerBuilder, /) -> NoReturn:
    raise LinearCapabilityError(
        f"{type(builder).__name__} has no lower-precision accumulation/residual "
        "contract; lower-precision preconditioning supports Jacobi, block "
        "Jacobi, and single-direction Gauss-Seidel builders."
    )


def _lowered_space(space: AbstractVectorSpace, compute_dtype: str, /) -> ArraySpace:
    """Coordinate-identical ArraySpace in a lower compute dtype."""
    if not isinstance(space, ArraySpace):
        raise LinearCapabilityError(
            "Lower-precision preconditioning requires one ArraySpace coordinate layout."
        )
    dtype = jnp.dtype(compute_dtype)
    pairing = space.pairing
    if isinstance(pairing, DiagonalPairing):
        low_pairing: DiagonalPairing | EuclideanPairing = DiagonalPairing(
            pairing.weights.astype(dtype)
        )
    elif isinstance(pairing, EuclideanPairing):
        low_pairing = EuclideanPairing()
    else:
        raise LinearCapabilityError(
            "Lower-precision preconditioning requires Euclidean or diagonal pairing."
        )
    return ArraySpace(space.shape, dtype=dtype, pairing=low_pairing)


class DenseInversePreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare an exact dense inverse, principally for small coarse spaces."""

    _builder_id: str = eqx.field(static=True)

    def __init__(self) -> None:
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


@final
class ProjectedPseudoinversePreconditioner(AbstractPreconditioner, NonTrainableState):
    """Bounded factored coarse solve with declared compatibility and gauge.

    Factor rank, cutoff, conditioning and finiteness remain available in
    ``factors``; kernel residuals record validation against the declared policy.
    No row is removed or replaced, and no pseudoinverse matrix is constructed.
    """

    factors: DensePseudoinverseFactors
    nullspace: NullspacePolicy
    right_kernel: LinearSubspace
    left_kernel: LinearSubspace
    right_kernel_residual: Array
    left_kernel_residual: Array
    tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        setup_operator: AbstractLinearOperator,
        factors: DensePseudoinverseFactors,
        nullspace: NullspacePolicy,
        properties: PreconditionerProperties,
        right_kernel_residual: Array,
        left_kernel_residual: Array,
        tolerance: float,
    ) -> None:
        right, left = nullspace.right, nullspace.left
        if right is None or left is None:
            raise ValueError("Prepared projected coarse solve requires both kernels.")
        identifier = canonical_fingerprint(
            {
                "kind": "projected-pseudoinverse-preconditioner",
                "operator": setup_operator.operator_id,
                "right": right.subspace_id,
                "left": left.subspace_id,
                "compatibility": nullspace.compatibility,
                "gauge": nullspace.gauge,
            }
        )
        self.space = setup_operator.source
        self.factors = factors
        self.nullspace = nullspace
        self.right_kernel = right
        self.left_kernel = left
        self.properties = properties
        self.right_kernel_residual = right_kernel_residual
        self.left_kernel_residual = left_kernel_residual
        self.tolerance = tolerance
        self.preconditioner_id = identifier

    def apply(
        self,
        residual: Any,
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> Any:
        from ._dense_pseudoinverse import apply_pseudoinverse

        coordinates = self.space.flatten(residual)
        incompatible = self.left_kernel.project_coordinates(coordinates)
        if self.nullspace.compatibility == "error":
            coordinates = eqx.error_if(
                coordinates,
                jnp.linalg.norm(incompatible)
                > self.tolerance * (1 + jnp.linalg.norm(coordinates)),
                "Coarse residual is incompatible with the declared left nullspace.",
            )
        compatible = coordinates - incompatible
        value = apply_pseudoinverse(self.factors, compatible)
        # Both supported native gauges select the orthogonal representative.
        value = value - self.right_kernel.project_coordinates(value)
        return self.space.unflatten(value)


@final
class ProjectedPseudoinversePreconditionerBuilder(AbstractPreconditionerBuilder):
    """Native small-coarse-space pseudoinverse with complete declared kernels.

    This is an explicit coarse solver, not permission to materialize a fine
    operator. The supplied materialization budget bounds both the matrix and
    four matrix-sized factor/workspace allocations. Euclidean coordinates make
    Moore--Penrose and declared orthogonal gauges agree without changing units.
    """

    nullspace: NullspacePolicy
    right_kernel: LinearSubspace
    left_kernel: LinearSubspace
    rank_policy: RankPolicy = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        nullspace: NullspacePolicy,
        /,
        *,
        rank_policy: RankPolicy | None = None,
        tolerance: float = 1e-8,
    ) -> None:
        from ._policies import RankPolicy
        from ._subspaces import NullspacePolicy

        if not isinstance(nullspace, NullspacePolicy):
            raise TypeError("nullspace must be NullspacePolicy.")
        right, left = nullspace.right, nullspace.left
        if right is None or left is None:
            raise ValueError(
                "Projected coarse solves require both complete declared kernels."
            )
        policy = RankPolicy() if rank_policy is None else rank_policy
        if not isinstance(policy, RankPolicy) or policy.require_full_rank:
            raise ValueError("rank_policy must allow the declared singular rank.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0:
            raise ValueError("tolerance must be finite and positive.")
        identifier = canonical_fingerprint(
            {
                "kind": "projected-pseudoinverse-builder",
                "right": right.subspace_id,
                "left": left.subspace_id,
                "compatibility": nullspace.compatibility,
                "gauge": nullspace.gauge,
                "relative_cutoff": policy.relative_cutoff,
                "absolute_cutoff": policy.absolute_cutoff,
                "tolerance": tolerance_,
            }
        )
        self.nullspace = nullspace
        self.right_kernel = right
        self.left_kernel = left
        self.rank_policy = policy
        self.tolerance = tolerance_
        self._builder_id = identifier

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> PreconditionerRefreshPolicy:
        return "numeric"

    def properties_for(
        self, setup_operator: AbstractLinearOperator, /
    ) -> PreconditionerProperties:
        _validate_setup_operator(setup_operator)
        space = setup_operator.source
        if not isinstance(space, ArraySpace) or not isinstance(
            space.pairing, EuclideanPairing
        ):
            raise ValueError(
                "Projected pseudoinverse requires Euclidean ArraySpace coordinates."
            )
        for kernel in (self.right_kernel, self.left_kernel):
            if not kernel.space.compatible(space) or kernel.batch_shape:
                raise ValueError(
                    "Declared coarse kernels must match the unbatched operator space."
                )
        claims = {
            "linear": True,
            "stationary": True,
            "self_adjoint": setup_operator.properties.certifies("self_adjoint"),
        }
        return PreconditionerProperties(
            **claims,
            evidence={name: "transformed" for name, value in claims.items() if value},
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        self.properties_for(setup_operator)
        entries = setup_operator.source.size**2
        itemsize = _coordinate_dtype(setup_operator.source).itemsize
        accepted, reason = _dense_materialization_eligibility(
            setup_operator, materialization
        )
        if materialization is not None and (
            4 * entries > materialization.max_entries
            or 4 * entries * itemsize > materialization.max_bytes
        ):
            accepted = False
            reason = (
                "coarse pseudoinverse factors exceed the explicit materialization budget"
            )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=3 * entries * itemsize,
            preparation_workspace_bytes=4 * entries * itemsize,
            apply_workspace_bytes_per_rhs=4 * setup_operator.source.size * itemsize,
            setup_matvec_count=_materialization_matvec_count(setup_operator),
            accepted=accepted,
            reason=reason,
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        from ._dense_pseudoinverse import factor_pseudoinverse

        properties = self.properties_for(setup_operator)
        estimate = self.cost_for(setup_operator, materialization=materialization)
        if not estimate.accepted:
            raise LinearCapabilityError(estimate.reason)
        matrix = materialize(setup_operator, materialization)
        factors = factor_pseudoinverse(matrix, self.rank_policy)
        right, left = self.right_kernel, self.left_kernel
        right_basis = jnp.where(
            jnp.arange(right.capacity) < right.dimension, right.basis, 0
        )
        left_basis = jnp.where(jnp.arange(left.capacity) < left.dimension, left.basis, 0)
        scale = jnp.maximum(1.0, jnp.linalg.norm(matrix))
        right_error = jnp.linalg.norm(matrix @ right_basis) / (
            scale * jnp.maximum(1.0, jnp.linalg.norm(right_basis))
        )
        left_error = jnp.linalg.norm(jnp.conj(matrix.T) @ left_basis) / (
            scale * jnp.maximum(1.0, jnp.linalg.norm(left_basis))
        )
        checked_matrix = eqx.error_if(
            factors.matrix,
            ~factors.finite
            | (right_error > self.tolerance)
            | (left_error > self.tolerance)
            | (factors.rank != matrix.shape[1] - right.dimension)
            | (factors.rank != matrix.shape[0] - left.dimension),
            "Coarse factor rank or kernel residual does not match the complete declared nullspaces.",
        )
        factors = eqx.tree_at(lambda value: value.matrix, factors, checked_matrix)
        return ProjectedPseudoinversePreconditioner(
            setup_operator,
            factors,
            self.nullspace,
            properties,
            right_error,
            left_error,
            self.tolerance,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, ProjectedPseudoinversePreconditioner):
            raise TypeError(
                "Projected pseudoinverse refresh requires its prepared action."
            )
        return self.prepare(setup_operator, materialization=materialization)


class JacobiPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare a damped Jacobi inverse from an operator diagonal."""

    relaxation: float = eqx.field(static=True)
    _builder_id: str = eqx.field(static=True)

    def __init__(self, *, relaxation: float = 1.0) -> None:
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
            "positive_semidefinite"
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
        if properties.certifies("positive_definite"):
            # A PSD setup can be singular while its Jacobi diagonal is strictly
            # positive. Zero coordinates are refused, never regularized.
            diagonal = eqx.error_if(
                diagonal,
                jnp.any(~jnp.isfinite(diagonal))
                | jnp.any(jnp.real(diagonal) <= 0)
                | jnp.any(jnp.imag(diagonal) != 0),
                "A positive Jacobi correction requires a strictly positive real diagonal.",
            )
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

    def lowered_cost(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        compute_dtype: str,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        cost = self.cost_for(setup_operator, materialization=materialization)
        low_itemsize = jnp.dtype(compute_dtype).itemsize
        high_itemsize = _coordinate_dtype(setup_operator.source).itemsize
        dimension = setup_operator.source.size
        return PreconditionerCostEstimate(
            component=cost.component,
            storage_bytes=dimension * low_itemsize,
            preparation_workspace_bytes=(
                cost.preparation_workspace_bytes + dimension * low_itemsize
            ),
            apply_workspace_bytes_per_rhs=dimension * (high_itemsize + 2 * low_itemsize),
            setup_matvec_count=cost.setup_matvec_count,
            accepted=cost.accepted,
            reason=(
                f"{cost.reason}; stored/applied in {compute_dtype} with explicit coordinate casts"
            ),
        )

    def prepare_lowered(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        compute_dtype: str,
        materialization: MaterializationPolicy,
        previous: AbstractPreconditioner | None = None,
    ) -> AbstractPreconditioner:
        del previous
        space = _lowered_space(setup_operator.source, compute_dtype)
        action = self.prepare(setup_operator, materialization=materialization)
        if not isinstance(action, DiagonalPreconditioner):
            raise RuntimeError("Jacobi preparation must return a diagonal action.")
        return DiagonalPreconditioner(
            jnp.reciprocal(action.inverse_diagonal).astype(space.dtype),
            space=space,
            positive_definite=action.properties.certifies("positive_definite"),
        )


class BlockJacobiPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare fixed-size block Jacobi factors from exact canonical blocks.

    ``padding`` declares structurally absent coordinates (for example the
    padding that buckets variable-size local patches into one homogeneous
    block shape). Their rows and columns must be exactly zero in the setup
    operator, which is checked, and their diagonal entries are factored as
    identity; every other block entry is the exact operator entry.
    """

    block_size: int = eqx.field(static=True)
    relaxation: float = eqx.field(static=True)
    assembly: SparseAssemblyPolicy | None
    padding: Array | None
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        block_size: int,
        /,
        *,
        relaxation: float = 1.0,
        assembly: SparseAssemblyPolicy | None = None,
        padding: ArrayLike | None = None,
    ) -> None:
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
        padding_ = None if padding is None else jnp.asarray(padding)
        if padding_ is not None and (
            padding_.ndim != 1 or padding_.dtype != jnp.bool_ or padding_.size % size
        ):
            raise ValueError(
                "padding must be a Boolean coordinate mask divisible into whole blocks."
            )
        self.block_size = size
        self.relaxation = relaxation_
        self.assembly = assembly
        self.padding = padding_
        self._builder_id = canonical_fingerprint(
            {
                "kind": "block-jacobi-preconditioner-builder",
                "block_size": size,
                "relaxation": relaxation_,
                "assembly": _sparse_assembly_policy_payload(assembly),
                "padding": None if padding_ is None else array_tree_fingerprint(padding_),
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
        if dimension % self.block_size or (
            self.padding is not None and self.padding.size != dimension
        ):
            return PreconditionerCostEstimate(
                component=self.builder_id,
                accepted=False,
                reason=(
                    f"block size {self.block_size} does not divide operator dimension "
                    f"{dimension}, or the padding mask does not cover it"
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
            + num_blocks * jnp.dtype(jnp.bool_).itemsize
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
                f"exact {self.block_size}-coordinate block extraction and batched factorization"
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
        if self.padding is not None:
            if self.padding.size != setup_operator.source.size:
                raise ValueError("padding must have one entry per setup coordinate.")
            pad = self.padding.reshape(blocks.shape[:2])
            absent = pad[:, :, None] | pad[:, None, :]
            blocks = eqx.error_if(
                blocks,
                jnp.any(absent & (blocks != 0)),
                "Padded block-Jacobi coordinates must have structurally zero rows and columns.",
            )
            blocks = blocks + (
                pad[:, :, None] & jnp.eye(self.block_size, dtype=jnp.bool_)
            ).astype(blocks.dtype)
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

    def lowered_cost(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        compute_dtype: str,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        cost = self.cost_for(setup_operator, materialization=materialization)
        if not cost.accepted:
            return cost
        low = jnp.dtype(compute_dtype)
        real_low = jnp.empty((), dtype=low).real.dtype.itemsize
        high_itemsize = _coordinate_dtype(setup_operator.source).itemsize
        dimension = setup_operator.source.size
        block_bytes = dimension * self.block_size * low.itemsize
        return PreconditionerCostEstimate(
            component=cost.component,
            storage_bytes=(
                block_bytes
                + dimension * jnp.dtype(jnp.int32).itemsize
                + dimension * real_low
                + (dimension // self.block_size) * jnp.dtype(jnp.bool_).itemsize
            ),
            preparation_workspace_bytes=cost.preparation_workspace_bytes + block_bytes,
            apply_workspace_bytes_per_rhs=dimension * (high_itemsize + 3 * low.itemsize),
            setup_matvec_count=cost.setup_matvec_count,
            reason=(
                f"{cost.reason}; blocks factored and applied in {compute_dtype} "
                "with explicit coordinate casts"
            ),
        )

    def prepare_lowered(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        compute_dtype: str,
        materialization: MaterializationPolicy,
        previous: AbstractPreconditioner | None = None,
    ) -> AbstractPreconditioner:
        del previous
        space = _lowered_space(setup_operator.source, compute_dtype)
        blocks = assemble_uniform_blocks(
            setup_operator,
            self.block_size,
            policy=self._resolved_assembly_policy(materialization),
        )
        properties = self.properties_for(setup_operator)
        # Exact high-precision blocks are rounded once, then factored in the
        # compute dtype; singular rounded blocks are refused by the factor.
        return LocalBlockPreconditioner(
            blocks.astype(space.dtype),
            space=space,
            positive_definite=properties.certifies("positive_definite"),
            relaxation=self.relaxation,
            preconditioner_id=canonical_fingerprint(
                {
                    "kind": "prepared-block-jacobi",
                    "builder": self.builder_id,
                    "setup_operator": setup_operator.operator_id,
                    "compute_dtype": space.dtype.name,
                }
            ),
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
    ) -> None:
        side = parse(side, PreconditioningSide, "side")
        refresh = parse(refresh, PreconditionerRefreshPolicy | None, "refresh")
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
            self.refresh_policy = parse(
                source.default_refresh if refresh is None else refresh,
                PreconditionerRefreshPolicy,
                "default_refresh",
            )
        else:
            raise TypeError(
                "source must be an AbstractPreconditioner or AbstractPreconditionerBuilder."
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
    setup: PlannedPreconditionerSetup | None
    setup_operator_id: str | None = eqx.field(static=True)
    component_id: str = eqx.field(static=True)
    compute_dtype: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        policy: PreconditioningPolicy,
        system_operator: AbstractLinearOperator,
        /,
        *,
        side: Literal["left", "right"],
        materialization: MaterializationPolicy | None = None,
        compute_dtype: str | None = None,
    ) -> None:
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
        planned: PlannedPreconditionerSetup | None = None
        if compute_dtype_ is None and isinstance(source, AbstractPreconditionerBuilder):
            # The costed construction travels with the plan to preparation.
            planned = source.plan_setup(setup, materialization=materialization_)
            cost = planned.cost
        elif compute_dtype_ is None:
            cost = _source_cost(
                source,
                setup,
                materialization=materialization_,
            )
        elif isinstance(source, AbstractPreconditionerBuilder):
            cost = source.lowered_cost(
                setup,
                compute_dtype=compute_dtype_,
                materialization=materialization_,
            )
        else:
            raise LinearCapabilityError(
                "Lower-precision preconditioning prepares its own action; supply a builder."
            )
        if not cost.accepted:
            raise ValueError(
                f"Preconditioner {cost.component} is infeasible: {cost.reason}."
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
        self.setup = planned
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

    @checked
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
    ) -> None:
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
            "Preconditioner numeric versions must satisfy 0 <= built_numeric_version <= numeric_version.",
        )
        built_version = eqx.error_if(
            built_version,
            invalid,
            "Preconditioner numeric versions must satisfy 0 <= built_numeric_version <= numeric_version.",
        )
        refresh_kind = parse(refresh_kind, PreconditionerRefreshKind, "refresh_kind")
        _validate_prepared_action(action, plan)
        self.action = action
        self.setup_operator = setup_operator
        self.plan = plan
        self.numeric_version = version
        self.built_numeric_version = built_version
        self.refresh_kind = refresh_kind


def _prepare_builder_action(
    builder: AbstractPreconditionerBuilder,
    setup: AbstractLinearOperator,
    plan: PreconditionerPlan,
    /,
    *,
    materialization: MaterializationPolicy,
    previous: AbstractPreconditioner | None,
) -> AbstractPreconditioner:
    if plan.compute_dtype is None:
        if previous is None and plan.setup is not None:
            return builder.prepare_planned(
                plan.setup, setup, materialization=materialization
            )
        if previous is None:
            return builder.prepare(setup, materialization=materialization)
        return builder.refresh(previous, setup, materialization=materialization)
    if previous is not None and not isinstance(previous, PrecisionCastPreconditioner):
        raise LinearCapabilityError(
            "Prepared preconditioner precision does not match its plan."
        )
    lowered = builder.prepare_lowered(
        setup,
        compute_dtype=plan.compute_dtype,
        materialization=materialization,
        previous=None if previous is None else previous.inner,
    )
    return PrecisionCastPreconditioner(lowered, setup.source, plan.compute_dtype)


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
            action = _prepare_builder_action(
                policy.builder,
                setup,
                plan,
                materialization=materialization,
                previous=None,
            )
            built_version = numeric_version
            refresh_kind = "prepared"
        elif policy.refresh_policy == "frozen":
            action = previous.action
            built_version = previous.built_numeric_version
            refresh_kind = "reused"
        elif policy.refresh_policy == "numeric":
            action = _prepare_builder_action(
                policy.builder,
                setup,
                plan,
                materialization=materialization,
                previous=previous.action,
            )
            built_version = numeric_version
            refresh_kind = "refreshed"
        else:
            action = _prepare_builder_action(
                policy.builder,
                setup,
                plan,
                materialization=materialization,
                previous=None,
            )
            built_version = numeric_version
            refresh_kind = "rebuilt"
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
    "ProjectedPseudoinversePreconditioner",
    "ProjectedPseudoinversePreconditionerBuilder",
    "PreconditionerCostEstimate",
    "PreconditioningPolicy",
    "PreconditioningSide",
    "PlannedPreconditionerSetup",
    "PreparedPreconditioner",
]
