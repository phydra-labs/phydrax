#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
from jaxtyping import Array

from ...._strict import StrictModule
from ....linalg import DenseLinearOperator, OperatorProperties
from ....linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy
from ._cross_section import ThinWalledSection


class GBTModeFamily(IntEnum):
    RIGID = 0
    GLOBAL = 1
    DISTORTIONAL = 2
    LOCAL = 3


class GBTModeBasis(StrictModule):
    eigenvalues: Array
    modes: Array
    families: Array
    orthogonality_error: Array
    section_id: str


def compute_gbt_modes(
    section: ThinWalledSection,
    /,
    *,
    mode_count: int | None = None,
) -> GBTModeBasis:
    """Construct a section-graph deformation basis for GBT/finite-strip coupling."""
    node_count = section.nodes.shape[0]
    stiffness = jnp.zeros((node_count, node_count), dtype=section.nodes.dtype)
    segment_stiffness = section.thickness**3 / section.widths
    first, second = section.segments[:, 0], section.segments[:, 1]
    stiffness = stiffness.at[first, first].add(segment_stiffness)
    stiffness = stiffness.at[second, second].add(segment_stiffness)
    stiffness = stiffness.at[first, second].add(-segment_stiffness)
    stiffness = stiffness.at[second, first].add(-segment_stiffness)
    operator = DenseLinearOperator(
        stiffness,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_semidefinite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_semidefinite": "construction",
            },
        ),
        operator_id=f"{section.section_id}:gbt-section-modes",
    )
    count = node_count if mode_count is None else min(int(mode_count), node_count)
    result = eigensolve(
        Eigenproblem(operator, problem_id=f"{section.section_id}:gbt-basis"),
        policy=EigenSolvePolicy(DenseEigh(), count=count, which="smallest-algebraic"),
    )
    scale = jnp.maximum(jnp.max(result.eigenvalues), jnp.finfo(stiffness.dtype).tiny)
    normalized = result.eigenvalues / scale
    families = jnp.where(
        normalized <= 1.0e-10,
        int(GBTModeFamily.RIGID),
        jnp.where(
            normalized <= 0.1,
            int(GBTModeFamily.GLOBAL),
            jnp.where(
                normalized <= 0.5,
                int(GBTModeFamily.DISTORTIONAL),
                int(GBTModeFamily.LOCAL),
            ),
        ),
    ).astype(jnp.int32)
    gram = result.eigenvectors.T @ result.eigenvectors
    error = jnp.max(jnp.abs(gram - jnp.eye(count, dtype=gram.dtype)))
    return GBTModeBasis(
        result.eigenvalues,
        result.eigenvectors,
        families,
        error,
        section.section_id,
    )


__all__ = ["GBTModeBasis", "GBTModeFamily", "compute_gbt_modes"]
