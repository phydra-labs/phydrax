#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite Markovian Hall-device steady states with explicit current operators."""

from __future__ import annotations

from math import isfinite, isqrt

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    DensePropertyVerificationPolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve,
    verify_dense_properties,
)


class OpenHallTransportPlan(StrictModule, NonTrainableState):
    liouvillian: Array
    current_operators: Array
    residual_tolerance: float = eqx.field(static=True)
    bath_identity: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        liouvillian: ArrayLike,
        current_operators: ArrayLike,
        bath_identity: str,
        /,
        *,
        residual_tolerance: float = 1.0e-10,
    ):
        generator = np.asarray(liouvillian, dtype=np.complex128)
        currents = np.asarray(current_operators, dtype=np.complex128)
        identifier = str(bath_identity).strip()
        tolerance = float(residual_tolerance)
        dimension = isqrt(generator.shape[0]) if generator.ndim == 2 else 0
        if (
            generator.shape != (dimension * dimension, dimension * dimension)
            or currents.ndim != 3
            or currents.shape[1:] != (dimension, dimension)
            or np.any(~np.isfinite(generator))
            or np.any(~np.isfinite(currents))
            or any(not np.allclose(value, np.conj(value.T)) for value in currents)
            or not identifier
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "Open Hall Liouvillian, currents, bath, or tolerance is invalid."
            )
        self.liouvillian = jnp.asarray(generator)
        self.current_operators = jnp.asarray(currents)
        self.residual_tolerance = tolerance
        self.bath_identity = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "open-hall-transport-plan",
                "arrays": array_tree_fingerprint(
                    {"liouvillian": generator, "currents": currents}
                ),
                "bath_identity": identifier,
                "residual_tolerance": tolerance,
            }
        )


class OpenHallTransportResult(StrictModule, NonTrainableState):
    density_matrix: Array
    currents: Array
    steady_residual: Array
    trace_residual: Array
    positive_semidefinite: Array
    linear_solve: LinearSolveResult
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def solve_open_hall_transport(plan: OpenHallTransportPlan, /) -> OpenHallTransportResult:
    if not isinstance(plan, OpenHallTransportPlan):
        raise TypeError("plan must be OpenHallTransportPlan.")
    dimension = plan.current_operators.shape[-1]
    trace_row = jnp.eye(dimension, dtype=plan.liouvillian.dtype).reshape((-1,))
    matrix = plan.liouvillian.at[-1, :].set(trace_row)
    right = (
        jnp.zeros((dimension * dimension,), dtype=plan.liouvillian.dtype).at[-1].set(1.0)
    )
    linear_solve = solve(
        LinearSystem(DenseLinearOperator(matrix)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    )
    density = linear_solve.value.reshape((dimension, dimension))
    currents = jnp.real(jnp.trace(plan.current_operators @ density, axis1=-2, axis2=-1))
    steady = jnp.sqrt(jnp.sum(jnp.abs(plan.liouvillian @ density.reshape((-1,))) ** 2))
    trace_residual = jnp.abs(jnp.trace(density) - 1.0)
    properties = verify_dense_properties(
        density,
        policy=DensePropertyVerificationPolicy(
            require_hermitian=True,
            require_positive_semidefinite=True,
        ),
    )
    successful = (
        linear_solve.successful
        & properties.successful
        & (steady <= plan.residual_tolerance)
        & (trace_residual <= plan.residual_tolerance)
    )
    return OpenHallTransportResult(
        density,
        currents,
        steady,
        trace_residual,
        properties.positive_semidefinite,
        linear_solve,
        successful,
        plan.plan_id,
        canonical_fingerprint(
            {"kind": "open-hall-transport-result", "plan": plan.plan_id}
        ),
    )


__all__ = [
    "OpenHallTransportPlan",
    "OpenHallTransportResult",
    "solve_open_hall_transport",
]
