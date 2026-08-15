#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax

from .._strict import StrictModule


def _array_tree_storage_bytes(value: object, /) -> int:
    """Count distinct resident array buffers once within one immutable artifact."""
    arrays = {id(leaf): leaf for leaf in jax.tree.leaves(value) if eqx.is_array(leaf)}
    return sum(int(array.size * array.dtype.itemsize) for array in arrays.values())


class PreconditionerCostEstimate(StrictModule):
    """Static storage and workspace estimate for one preconditioner source."""

    component: str = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    apply_workspace_bytes_per_rhs: int = eqx.field(static=True)
    setup_matvec_count: int = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        component: str,
        storage_bytes: int = 0,
        preparation_workspace_bytes: int = 0,
        apply_workspace_bytes_per_rhs: int = 0,
        setup_matvec_count: int = 0,
        accepted: bool = True,
        reason: str = "feasible",
    ):
        component_, reason_ = str(component), str(reason)
        if not component_ or not reason_:
            raise ValueError("Preconditioner cost strings must be non-empty.")
        integers = tuple(
            int(value)
            for value in (
                storage_bytes,
                preparation_workspace_bytes,
                apply_workspace_bytes_per_rhs,
                setup_matvec_count,
            )
        )
        if any(value < 0 for value in integers):
            raise ValueError("Preconditioner cost estimates must be non-negative.")
        self.component = component_
        (
            self.storage_bytes,
            self.preparation_workspace_bytes,
            self.apply_workspace_bytes_per_rhs,
            self.setup_matvec_count,
        ) = integers
        self.accepted = bool(accepted)
        self.reason = reason_


class OperatorActionCostEstimate(StrictModule):
    """Resident state and per-right-hand-side scratch for one operator action."""

    operator_id: str = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)
    apply_workspace_bytes_per_rhs: int = eqx.field(static=True)
    operation_class: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        operator_id: str,
        storage_bytes: int,
        apply_workspace_bytes_per_rhs: int,
        operation_class: str,
        exact: bool,
        reason: str,
    ):
        strings = tuple(str(value) for value in (operator_id, operation_class, reason))
        if any(not value for value in strings):
            raise ValueError("Operator action cost strings must be non-empty.")
        storage, workspace = int(storage_bytes), int(apply_workspace_bytes_per_rhs)
        if storage < 0 or workspace < 0:
            raise ValueError("Operator action cost estimates must be non-negative.")
        self.operator_id, self.operation_class, self.reason = strings
        self.storage_bytes = storage
        self.apply_workspace_bytes_per_rhs = workspace
        self.exact = bool(exact)


class LinearCostEstimate(StrictModule):
    """Static resource estimate and eligibility result for one solver candidate."""

    provider: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    existing_storage_bytes: int = eqx.field(static=True)
    additional_matrix_bytes: int = eqx.field(static=True)
    factorization_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    solve_workspace_bytes_per_rhs: int = eqx.field(static=True)
    operator_apply_workspace_bytes_per_rhs: int = eqx.field(static=True)
    krylov_basis_bytes_per_rhs: int = eqx.field(static=True)
    preconditioner_storage_bytes: int = eqx.field(static=True)
    preconditioner_preparation_workspace_bytes: int = eqx.field(static=True)
    preconditioner_apply_workspace_bytes_per_rhs: int = eqx.field(static=True)
    preconditioner_setup_matvec_count: int = eqx.field(static=True)
    recycling_capacity: int = eqx.field(static=True)
    recycling_state_bytes: int = eqx.field(static=True)
    operation_class: str = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        provider: str,
        method: str,
        existing_storage_bytes: int = 0,
        additional_matrix_bytes: int = 0,
        factorization_bytes: int = 0,
        preparation_workspace_bytes: int = 0,
        solve_workspace_bytes_per_rhs: int = 0,
        operator_apply_workspace_bytes_per_rhs: int = 0,
        krylov_basis_bytes_per_rhs: int = 0,
        preconditioner_storage_bytes: int = 0,
        preconditioner_preparation_workspace_bytes: int = 0,
        preconditioner_apply_workspace_bytes_per_rhs: int = 0,
        recycling_capacity: int = 0,
        preconditioner_setup_matvec_count: int = 0,
        recycling_state_bytes: int = 0,
        operation_class: str,
        accepted: bool,
        reason: str,
    ):
        strings = tuple(
            str(value) for value in (provider, method, operation_class, reason)
        )
        if any(not value for value in strings):
            raise ValueError("Cost-estimate strings must be non-empty.")
        integers = tuple(
            int(value)
            for value in (
                existing_storage_bytes,
                additional_matrix_bytes,
                factorization_bytes,
                preparation_workspace_bytes,
                solve_workspace_bytes_per_rhs,
                krylov_basis_bytes_per_rhs,
                operator_apply_workspace_bytes_per_rhs,
                preconditioner_storage_bytes,
                preconditioner_preparation_workspace_bytes,
                preconditioner_apply_workspace_bytes_per_rhs,
                recycling_capacity,
                preconditioner_setup_matvec_count,
                recycling_state_bytes,
            )
        )
        if any(value < 0 for value in integers):
            raise ValueError("Cost-estimate byte counts must be non-negative.")
        self.provider, self.method, self.operation_class, self.reason = strings
        (
            self.existing_storage_bytes,
            self.additional_matrix_bytes,
            self.factorization_bytes,
            self.preparation_workspace_bytes,
            self.solve_workspace_bytes_per_rhs,
            self.krylov_basis_bytes_per_rhs,
            self.operator_apply_workspace_bytes_per_rhs,
            self.preconditioner_storage_bytes,
            self.preconditioner_preparation_workspace_bytes,
            self.preconditioner_apply_workspace_bytes_per_rhs,
            self.recycling_capacity,
            self.preconditioner_setup_matvec_count,
            self.recycling_state_bytes,
        ) = integers
        self.accepted = bool(accepted)


__all__ = [
    "LinearCostEstimate",
    "OperatorActionCostEstimate",
    "PreconditionerCostEstimate",
]
