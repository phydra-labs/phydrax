#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class RootDOFEvidence(StrictModule):
    """Numerical root and local degree-of-freedom evidence for a design solve.

    The rank is that of the residual Jacobian at the returned state. This is a
    local numerical diagnostic; it is not a global root count or interval
    uniqueness proof.
    """

    problem_id: str = eqx.field(static=True)
    variable_count: int = eqx.field(static=True)
    equation_count: int = eqx.field(static=True)
    jacobian_rank: int = eqx.field(static=True)
    degrees_of_freedom: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    residual_norm: Array
    singular_values: Array
    numerically_finite: bool = eqx.field(static=True)
    root_converged: bool = eqx.field(static=True)
    full_column_rank: bool = eqx.field(static=True)
    regular_root_indicated: bool = eqx.field(static=True)
    scope: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem_id: str,
        coordinates: ArrayLike,
        residual: ArrayLike,
        jacobian: ArrayLike,
        /,
        *,
        residual_tolerance: float,
    ) -> None:
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("problem_id must be a non-empty string.")
        tolerance = float(residual_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive.")
        coordinates_ = np.asarray(coordinates, dtype=np.float64).reshape((-1,))
        residual_ = np.asarray(residual, dtype=np.float64).reshape((-1,))
        jacobian_ = np.asarray(jacobian, dtype=np.float64)
        if jacobian_.shape != (residual_.size, coordinates_.size):
            raise ValueError("The residual Jacobian must align with equations and variables.")
        numerically_finite = bool(
            np.all(np.isfinite(coordinates_))
            and np.all(np.isfinite(residual_))
            and np.all(np.isfinite(jacobian_))
        )
        if numerically_finite:
            singular_values = np.linalg.svd(jacobian_, compute_uv=False)
            largest = float(singular_values[0]) if singular_values.size else 0.0
            rank_tolerance = (
                np.finfo(np.float64).eps * max(jacobian_.shape, default=1) * largest
            )
            rank = int(np.count_nonzero(singular_values > rank_tolerance))
            residual_norm = float(np.linalg.norm(residual_))
        else:
            singular_values = np.empty((0,), dtype=np.float64)
            rank_tolerance = 0.0
            rank = 0
            residual_norm = float(np.linalg.norm(residual_))
        root_converged = numerically_finite and residual_norm <= tolerance
        full_column_rank = numerically_finite and rank == coordinates_.size
        regular_root_indicated = root_converged and full_column_rank
        scope = "pointwise-residual-jacobian-rank"
        evidence_id = canonical_fingerprint(
            {
                "kind": "design-root-dof-evidence",
                "problem": problem_id,
                "coordinates": coordinates_.tolist() if numerically_finite else None,
                "residual": residual_.tolist() if numerically_finite else None,
                "jacobian": jacobian_.tolist() if numerically_finite else None,
                "numerically_finite": numerically_finite,
                "rank_tolerance": rank_tolerance,
                "residual_tolerance": tolerance,
                "scope": scope,
            }
        )
        self.numerically_finite = numerically_finite
        self.problem_id = problem_id
        self.variable_count = coordinates_.size
        self.equation_count = residual_.size
        self.jacobian_rank = rank
        self.degrees_of_freedom = coordinates_.size - rank
        self.rank_tolerance = rank_tolerance
        self.residual_tolerance = tolerance
        self.residual_norm = jnp.asarray(residual_norm, dtype=jnp.float64)
        self.singular_values = jnp.asarray(singular_values, dtype=jnp.float64)
        self.root_converged = root_converged
        self.full_column_rank = full_column_rank
        self.regular_root_indicated = regular_root_indicated
        self.scope = scope
        self.evidence_id = evidence_id


__all__ = ["RootDOFEvidence"]
