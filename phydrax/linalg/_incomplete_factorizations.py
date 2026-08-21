#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite
from typing import Any

import equinox as eqx
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from ._costs import _array_tree_storage_bytes, PreconditionerCostEstimate
from ._materialization import MaterializationPolicy
from ._operators import AbstractLinearOperator
from ._preconditioner_properties import PreconditionerProperties
from ._preconditioners import AbstractPreconditioner
from ._preconditioning import AbstractPreconditionerBuilder
from ._sparse_contract import AbstractSparseLinearOperator
from ._sparse_factorizations import (
    prepare_sparse_factorization,
    PreparedSparseFactorization,
    refresh_sparse_factorization,
    SparseFactorizationPolicy,
    SparseFactorizationStatus,
    SparseOrdering,
)


class SparseFactorizationPreconditioner(AbstractPreconditioner):
    """Sparse exact or incomplete factor solve through the preconditioner API."""

    factorization: PreparedSparseFactorization

    def __init__(
        self,
        operator: AbstractSparseLinearOperator,
        factorization: PreparedSparseFactorization,
        /,
        *,
        properties: PreconditionerProperties,
        preconditioner_id: str,
    ):
        if not isinstance(operator, AbstractSparseLinearOperator):
            raise TypeError("operator must be an AbstractSparseLinearOperator.")
        if not isinstance(factorization, PreparedSparseFactorization):
            raise TypeError("factorization must be PreparedSparseFactorization.")
        if not operator.source.compatible(operator.target):
            raise ValueError(
                "Sparse factor preconditioning requires one compatible space."
            )
        identifier = str(preconditioner_id)
        if not identifier:
            raise ValueError("preconditioner_id must be non-empty.")
        if not isinstance(properties, PreconditionerProperties):
            raise TypeError("properties must be PreconditionerProperties.")
        cholesky = factorization.plan.kind == "cholesky"
        expected_positive = cholesky and operator.properties.certifies(
            "positive_definite"
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
                "Sparse factor preconditioner claims must match the factor kind "
                "and setup-operator evidence."
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
        iteration: Array | None = None,
    ) -> PyTree[Array]:
        del iteration
        coordinates = self.space.flatten(self.space.validate(residual))
        solved = self.factorization.solve(coordinates)
        value = eqx.error_if(
            solved.value,
            solved.status != int(SparseFactorizationStatus.SUCCESS),
            "Sparse factor preconditioner solve failed; inspect factor diagnostics.",
        )
        return self.space.unflatten(value)


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
    positive = cholesky and setup_operator.properties.certifies("positive_definite")
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
                },
            }
        )

    @property
    def default_refresh(self) -> str:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        if not isinstance(setup_operator, AbstractLinearOperator):
            raise TypeError("setup_operator must be an AbstractLinearOperator.")
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
        del materialization
        self.properties_for(setup_operator)
        if not isinstance(setup_operator, AbstractSparseLinearOperator):
            return PreconditionerCostEstimate(
                component=self.builder_id,
                accepted=False,
                reason="sparse factorization requires canonical sparse operator storage",
            )
        plan = prepare_sparse_factorization(setup_operator, self.policy())
        itemsize = setup_operator.sparse_storage().values.dtype.itemsize
        factor_entries = int(plan.factor_indices.size)
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=_array_tree_storage_bytes(plan) + factor_entries * itemsize,
            preparation_workspace_bytes=factor_entries * itemsize,
            apply_workspace_bytes_per_rhs=4 * plan.shape[0] * itemsize,
            accepted=True,
            reason="fixed-pattern sparse factorization",
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> SparseFactorizationPreconditioner:
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
    ) -> SparseFactorizationPreconditioner:
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


class SparseFactorizationPreconditionerBuilder(_AbstractSparseFactorizationBuilder):
    """Prepare a complete refreshable sparse LU or Cholesky coarse solve."""

    factorization_policy: SparseFactorizationPolicy = eqx.field(static=True)

    def __init__(
        self,
        policy: SparseFactorizationPolicy | None = None,
        /,
    ):
        policy_ = SparseFactorizationPolicy() if policy is None else policy
        if not isinstance(policy_, SparseFactorizationPolicy):
            raise TypeError("policy must be SparseFactorizationPolicy or None.")
        if policy_.fill_level is not None:
            raise ValueError("A complete sparse coarse solve requires fill_level=None.")
        self.factorization_policy = policy_

    def policy(self) -> SparseFactorizationPolicy:
        return self.factorization_policy


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
    ):
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
        if ordering not in ("natural", "reverse-cuthill-mckee"):
            raise ValueError(f"Unknown sparse ordering {ordering!r}.")
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
    ):
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
        if ordering not in ("natural", "reverse-cuthill-mckee"):
            raise ValueError(f"Unknown sparse ordering {ordering!r}.")
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
    ):
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
        if ordering not in ("natural", "reverse-cuthill-mckee"):
            raise ValueError(f"Unknown sparse ordering {ordering!r}.")
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
    "SparseFactorizationPreconditioner",
    "SparseFactorizationPreconditionerBuilder",
    "refresh_incomplete_factorization",
]
