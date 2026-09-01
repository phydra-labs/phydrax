#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

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


class DiscontinuousMassEvidence(StrictModule, NonTrainableState):
    minimum_eigenvalue: Array
    maximum_condition: Array
    positive_definite: Array
    evidence_id: str = eqx.field(static=True)


class PreparedDiscontinuousMassInverse(StrictModule):
    discretization: FiniteElementDiscretization
    field_name: str = eqx.field(static=True)
    routes: tuple[Array, ...]
    mass_matrices: tuple[Array, ...]
    factorizations: tuple[tuple[PreparedFactorization, ...], ...]
    evidence: DiscontinuousMassEvidence
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        field_name: str,
        volume_rules,
        /,
    ):
        from ...integration._rules import reference_rule_data

        if not isinstance(discretization, FiniteElementDiscretization):
            raise TypeError("discretization must be FiniteElementDiscretization.")
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
        all_factorizations = []
        minimum = np.inf
        condition = 0.0
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
                    "Exact discontinuous mass requires a cell-local L2 field."
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
                matrices = oe.contract(
                    "cq,qi,qj->cij",
                    physical_weights,
                    basis,
                    basis,
                    backend="jax",
                )
            else:
                matrices = oe.contract(
                    "cq,cqi,cqj->cij",
                    physical_weights,
                    basis,
                    basis,
                    backend="jax",
                )
            matrices_np = np.asarray(matrices)
            eigenvalues = np.linalg.eigvalsh(matrices_np)
            minimum = min(minimum, float(np.min(eigenvalues)))
            condition = max(
                condition,
                float(max(np.linalg.cond(matrix) for matrix in matrices_np)),
            )
            block_factorizations = tuple(
                factorize(
                    DenseLinearOperator(
                        matrix,
                        properties=properties,
                        operator_id=canonical_fingerprint(
                            {
                                "kind": "local-discontinuous-mass",
                                "field": field,
                                "block": block.name,
                                "cell": cell,
                                "matrix": array_tree_fingerprint(np.asarray(matrix)),
                            }
                        ),
                    ),
                    FactorizationPolicy("cholesky"),
                )
                for cell, matrix in enumerate(matrices)
            )
            all_routes.append(dof_map.cell_dofs[block_index])
            all_matrices.append(matrices)
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
            }
        )
        self.discretization = discretization
        self.field_name = field
        self.routes = tuple(all_routes)
        self.mass_matrices = tuple(all_matrices)
        self.factorizations = tuple(all_factorizations)
        self.evidence = DiscontinuousMassEvidence(
            jnp.asarray(minimum),
            jnp.asarray(condition),
            jnp.asarray(True),
            evidence_id,
        )
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-discontinuous-mass-inverse",
                "discretization": discretization.prepared_id,
                "field": field,
                "factorizations": tuple(
                    tuple(value.factorization_id for value in block)
                    for block in all_factorizations
                ),
                "evidence": evidence_id,
            }
        )

    def apply(self, residual: ArrayLike, /) -> Array:
        value = jnp.asarray(residual)
        result = jnp.zeros_like(value)
        for routes, factorizations in zip(self.routes, self.factorizations, strict=True):
            local = value[routes]
            local_flat = local.reshape((local.shape[0], local.shape[1], -1))
            solved_cells = []
            for cell, factorization in enumerate(factorizations):
                solved_components = []
                for component in range(local_flat.shape[-1]):
                    solved = factorization.solve(local_flat[cell, :, component])
                    solved_components.append(
                        eqx.error_if(
                            solved.value,
                            ~solved.successful,
                            "Local discontinuous mass factorization failed.",
                        )
                    )
                solved_cells.append(jnp.stack(tuple(solved_components), axis=-1))
            solved_local = jnp.stack(tuple(solved_cells), axis=0).reshape(local.shape)
            result = result.at[routes].set(solved_local)
        return result


__all__ = [
    "DiscontinuousMassEvidence",
    "PreparedDiscontinuousMassInverse",
]
