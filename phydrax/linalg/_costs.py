#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx

from .._strict import StrictModule


class LinearCostEstimate(StrictModule):
    """Static resource estimate and eligibility result for one solver candidate."""

    provider: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    existing_storage_bytes: int = eqx.field(static=True)
    additional_matrix_bytes: int = eqx.field(static=True)
    factorization_bytes: int = eqx.field(static=True)
    preparation_workspace_bytes: int = eqx.field(static=True)
    solve_workspace_bytes_per_rhs: int = eqx.field(static=True)
    krylov_basis_bytes_per_rhs: int = eqx.field(static=True)
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
        krylov_basis_bytes_per_rhs: int = 0,
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
        ) = integers
        self.accepted = bool(accepted)


__all__ = ["LinearCostEstimate"]
