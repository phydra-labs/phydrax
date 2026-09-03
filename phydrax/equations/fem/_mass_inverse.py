#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.fem._generic import FiniteElementDiscretization
from ...linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    OperatorProperties,
    PreparedFactorization,
)


DiscontinuousMassStrategy = Literal[
    "auto",
    "diagonal",
    "affine_scaled",
    "weight_adjusted",
    "exact_batched",
]


class DiscontinuousMassEvidence(StrictModule, NonTrainableState):
    minimum_eigenvalue: Array
    maximum_condition: Array
    positive_definite: Array
    strategies: tuple[str, ...] = eqx.field(static=True)
    resident_factor_bytes: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedDiscontinuousMassInverse(StrictModule):
    discretization: FiniteElementDiscretization
    field_name: str = eqx.field(static=True)
    strategies: tuple[str, ...] = eqx.field(static=True)
    routes: tuple[Array, ...]
    mass_matrices: tuple[Array, ...]
    inverse_diagonals: tuple[Array, ...]
    scales: tuple[Array, ...]
    weight_adjusted_matrices: tuple[Array, ...]
    factorizations: tuple[tuple[PreparedFactorization, ...], ...]
    evidence: DiscontinuousMassEvidence
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        field_name: str,
        volume_rules,
        /,
        *,
        strategy: DiscontinuousMassStrategy = "auto",
        structure_tolerance: float = 1.0e-11,
    ):
        from ...integration._rules import reference_rule_data

        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
        selected_strategy = str(strategy)
        if selected_strategy not in (
            "auto",
            "diagonal",
            "affine_scaled",
            "weight_adjusted",
            "exact_batched",
        ):
            raise ValueError("Unknown discontinuous mass strategy.")
        tolerance = float(structure_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("structure_tolerance must be finite and positive.")
        field = str(field_name)
        field_index = discretization._field_index(field)
        rules = (
            dict(volume_rules)
            if isinstance(volume_rules, Mapping)
            else {discretization.mesh.blocks[0].name: volume_rules}
        )
        if set(rules) != {block.name for block in discretization.mesh.blocks}:
            raise ValueError("One exact mass quadrature rule is required per mesh block.")
        all_routes = []
        all_matrices = []
        all_inverse_diagonals = []
        all_scales = []
        all_weight_adjusted = []
        all_factorizations = []
        strategies = []
        minimum = np.inf
        condition = 0.0
        factor_bytes = 0
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "verified",
            },
        )
        for block_index, block in enumerate(discretization.mesh.blocks):
            element = discretization.elements[field_index][block_index]
            dof_map = discretization.dof_maps[field_index]
            if element.conformity != "L2" or dof_map.association != "cell":
                raise ValueError(
                    "Discontinuous mass inversion requires a cell-local L2 field."
                )
            rule_data = reference_rule_data(rules[block.name])
            if rule_data.cell != element.cell_kind:
                raise ValueError("Mass quadrature must match each FE cell kind.")
            geometry = discretization.evaluate_block_geometry(
                field,
                block_index,
                discretization.default_runtime.coordinates,
                rule_data.points,
                rule_data.weights,
            )
            basis = geometry.basis_values
            physical_weights = geometry.physical_weights
            if basis.ndim == 2:
                matrices = ein.contract(
                    "cq,qi,qj->cij",
                    physical_weights,
                    basis,
                    basis,
                    backend="jax",
                )
                basis_host = np.asarray(basis)
            else:
                matrices = ein.contract(
                    "cq,cqi,cqj->cij",
                    physical_weights,
                    basis,
                    basis,
                    backend="jax",
                )
                basis_host = np.asarray(basis[0])
            matrices_np = np.asarray(matrices)
            eigenvalues = np.linalg.eigvalsh(matrices_np)
            minimum = min(minimum, float(np.min(eigenvalues)))
            condition = max(
                condition,
                float(max(np.linalg.cond(matrix) for matrix in matrices_np)),
            )
            diagonal = np.stack(tuple(np.diag(matrix) for matrix in matrices_np), axis=0)
            diagonal_matrices = np.zeros_like(matrices_np)
            indices = np.arange(diagonal.shape[1])
            diagonal_matrices[:, indices, indices] = diagonal
            diagonal_defect = np.max(np.abs(matrices_np - diagonal_matrices)) / max(
                1.0, float(np.max(np.abs(matrices_np)))
            )
            reference_physical = matrices_np[0]
            trace = np.trace(matrices_np, axis1=-2, axis2=-1)
            scale_values = trace / trace[0]
            affine_defect = max(
                float(
                    np.max(
                        np.abs(
                            matrices_np - scale_values[:, None, None] * reference_physical
                        )
                    )
                )
                / max(1.0, float(np.max(np.abs(matrices_np)))),
                0.0,
            )
            if selected_strategy == "auto":
                block_strategy = (
                    "diagonal"
                    if diagonal_defect <= tolerance
                    else "affine_scaled"
                    if affine_defect <= tolerance
                    else "weight_adjusted"
                )
            else:
                block_strategy = selected_strategy
            if block_strategy == "diagonal" and diagonal_defect > tolerance:
                raise ValueError("Requested diagonal mass is not diagonal.")
            if block_strategy == "affine_scaled" and affine_defect > tolerance:
                raise ValueError(
                    "Requested affine-scaled mass matrices are not proportional."
                )

            block_factorizations: tuple[PreparedFactorization, ...]
            inverse_diagonal = jnp.zeros((0, 0), dtype=matrices.dtype)
            scales = jnp.asarray(scale_values)
            weight_adjusted = jnp.zeros((0, 0, 0), dtype=matrices.dtype)
            if block_strategy == "diagonal":
                inverse_diagonal = 1.0 / jnp.asarray(diagonal)
                block_factorizations = ()
            elif block_strategy == "affine_scaled":
                factor = _factor_mass_matrix(
                    jnp.asarray(reference_physical),
                    properties,
                    field,
                    block.name,
                    "affine-reference",
                )
                block_factorizations = (factor,)
                factor_bytes += int(reference_physical.nbytes)
            elif block_strategy == "weight_adjusted":
                reference_weights = np.asarray(rule_data.weights)
                reference_mass = ein.contract(
                    "q,qi,qj->ij",
                    reference_weights,
                    basis_host,
                    basis_host,
                )
                jacobian = np.asarray(physical_weights) / reference_weights[None, :]
                reciprocal_mass = ein.contract(
                    "cq,qi,qj->cij",
                    reference_weights[None, :] / jacobian,
                    basis_host,
                    basis_host,
                )
                factor = _factor_mass_matrix(
                    jnp.asarray(reference_mass),
                    properties,
                    field,
                    block.name,
                    "weight-adjusted-reference",
                )
                block_factorizations = (factor,)
                weight_adjusted = jnp.asarray(reciprocal_mass)
                factor_bytes += int(reference_mass.nbytes + reciprocal_mass.nbytes)
            else:
                block_factorizations = tuple(
                    _factor_mass_matrix(
                        matrix,
                        properties,
                        field,
                        block.name,
                        f"exact-{cell}",
                    )
                    for cell, matrix in enumerate(matrices)
                )
                factor_bytes += int(matrices_np.nbytes)
            strategies.append(block_strategy)
            all_routes.append(dof_map.cell_dofs[block_index])
            all_matrices.append(matrices)
            all_inverse_diagonals.append(inverse_diagonal)
            all_scales.append(scales)
            all_weight_adjusted.append(weight_adjusted)
            all_factorizations.append(block_factorizations)
        if not np.isfinite(minimum) or minimum <= 0.0 or not np.isfinite(condition):
            raise ValueError(
                "Local discontinuous mass matrices must be positive definite."
            )
        evidence_id = canonical_fingerprint(
            {
                "kind": "discontinuous-mass-evidence",
                "field": field,
                "minimum_eigenvalue": minimum,
                "maximum_condition": condition,
                "strategies": tuple(strategies),
                "resident_factor_bytes": factor_bytes,
            }
        )
        self.discretization = discretization
        self.field_name = field
        self.strategies = tuple(strategies)
        self.routes = tuple(all_routes)
        self.mass_matrices = tuple(all_matrices)
        self.inverse_diagonals = tuple(all_inverse_diagonals)
        self.scales = tuple(all_scales)
        self.weight_adjusted_matrices = tuple(all_weight_adjusted)
        self.factorizations = tuple(all_factorizations)
        self.evidence = DiscontinuousMassEvidence(
            jnp.asarray(minimum),
            jnp.asarray(condition),
            jnp.asarray(True),
            tuple(strategies),
            factor_bytes,
            evidence_id,
        )
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-discontinuous-mass-inverse",
                "discretization": discretization.prepared_id,
                "field": field,
                "strategies": tuple(strategies),
                "factorizations": tuple(
                    tuple(value.factorization_id for value in block)
                    for block in all_factorizations
                ),
                "evidence": evidence_id,
            }
        )

    def _apply_primal(self, residual: ArrayLike, /, *, validate: bool = True) -> Array:
        value = jnp.asarray(residual)
        result = jnp.zeros_like(value)
        for strategy, routes, inverse_diagonal, scales, weight_adjusted, factors in zip(
            self.strategies,
            self.routes,
            self.inverse_diagonals,
            self.scales,
            self.weight_adjusted_matrices,
            self.factorizations,
            strict=True,
        ):
            local = value[routes]
            local_flat = local.reshape((local.shape[0], local.shape[1], -1))
            if strategy == "diagonal":
                solved_flat = local_flat * inverse_diagonal[..., None]
            elif strategy == "affine_scaled":
                solved_flat = (
                    _solve_grouped_factor(factors[0], local_flat, validate=validate)
                    / scales[:, None, None]
                )
            elif strategy == "weight_adjusted":
                first = _solve_grouped_factor(factors[0], local_flat, validate=validate)
                weighted = ein.contract(
                    "cij,cjk->cik", weight_adjusted, first, backend="jax"
                )
                solved_flat = _solve_grouped_factor(
                    factors[0], weighted, validate=validate
                )
            else:
                solved_cells = tuple(
                    _solve_grouped_factor(
                        factor,
                        local_flat[cell : cell + 1],
                        validate=validate,
                    )[0]
                    for cell, factor in enumerate(factors)
                )
                solved_flat = jnp.stack(solved_cells, axis=0)
            result = result.at[routes].set(
                solved_flat.reshape(local.shape), unique_indices=True
            )
        return result

    def apply(self, residual: ArrayLike, /) -> Array:
        return _apply_discontinuous_mass(self, jnp.asarray(residual))


@eqx.filter_custom_jvp
def _apply_discontinuous_mass(
    operator: PreparedDiscontinuousMassInverse,
    residual: Array,
    /,
) -> Array:
    return operator._apply_primal(residual)


@_apply_discontinuous_mass.def_jvp
def _apply_discontinuous_mass_jvp(primals, tangents):
    operator, residual = primals
    _operator_tangent, residual_tangent = tangents
    residual_tangent = (
        jnp.zeros_like(residual) if residual_tangent is None else residual_tangent
    )
    primal = operator._apply_primal(residual)
    tangent = operator._apply_primal(residual_tangent, validate=False)
    return primal, tangent


def _factor_mass_matrix(
    matrix: Array,
    properties: OperatorProperties,
    field: str,
    block: str,
    label: str,
    /,
) -> PreparedFactorization:
    return factorize(
        DenseLinearOperator(
            matrix,
            properties=properties,
            operator_id=canonical_fingerprint(
                {
                    "kind": "local-discontinuous-mass",
                    "field": field,
                    "block": block,
                    "label": label,
                    "matrix": array_tree_fingerprint(np.asarray(matrix)),
                }
            ),
        ),
        FactorizationPolicy("cholesky"),
    )


def _solve_grouped_factor(
    factorization: PreparedFactorization,
    right_hand_sides: Array,
    /,
    *,
    validate: bool = True,
) -> Array:
    cell_count, local_width, component_count = right_hand_sides.shape
    grouped = jnp.transpose(right_hand_sides, (1, 0, 2)).reshape(
        (local_width, cell_count * component_count)
    )
    solved = factorization.solve(grouped)
    value = (
        eqx.error_if(
            solved.value,
            ~solved.successful,
            "Local discontinuous mass factorization failed.",
        )
        if validate
        else solved.value
    )
    return jnp.transpose(
        value.reshape((local_width, cell_count, component_count)),
        (1, 0, 2),
    )


__all__ = [
    "DiscontinuousMassEvidence",
    "DiscontinuousMassStrategy",
    "PreparedDiscontinuousMassInverse",
]
