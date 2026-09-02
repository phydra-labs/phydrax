#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-mesh finite GKSL integration by certified channel exponentials."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jspla
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..discretization import TemporalMesh
from ..linalg import HermitianSpectrum
from ..operators.quantum._channels import (
    apply_finite_cptp,
    compose_finite_cptp,
    finite_cptp_from_superoperator,
    FiniteCPTPMap,
)


class FiniteLindbladChannelPlan(StrictModule):
    hamiltonian: Callable[[Array], Array] = eqx.field(static=True)
    jumps: Array
    rates: Array
    active_jumps: Array
    slicing: TemporalMesh
    evaluation: Literal["left", "midpoint"] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    jump_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: ArrayLike | Callable[[Array], Array],
        jumps: ArrayLike,
        rates: ArrayLike,
        slicing: TemporalMesh,
        /,
        *,
        active_jumps: ArrayLike | None = None,
        evaluation: Literal["left", "midpoint"] = "midpoint",
        tolerance: float = 1e-8,
        plan_id: str,
    ):
        operators = jnp.asarray(jumps)
        rate_values = jnp.asarray(rates)
        if (
            operators.ndim != 3
            or operators.shape[1] != operators.shape[2]
            or operators.shape[0] < 1
        ):
            raise ValueError("jumps must have shape (capacity, dimension, dimension).")
        capacity, dimension = int(operators.shape[0]), int(operators.shape[1])
        if rate_values.shape not in ((capacity,), (slicing.num_steps, capacity)):
            raise ValueError(
                "rates must have shape (capacity,) or (time_intervals, capacity)."
            )
        mask = (
            jnp.ones((capacity,), dtype=bool)
            if active_jumps is None
            else jnp.asarray(active_jumps, dtype=bool)
        )
        if mask.shape != (capacity,):
            raise ValueError("active_jumps shape must equal jump capacity.")
        if evaluation not in ("left", "midpoint") or tolerance < 0.0:
            raise ValueError("evaluation/tolerance is invalid.")
        if not isinstance(plan_id, str) or not plan_id:
            raise ValueError("plan_id must be nonempty.")
        if callable(hamiltonian):
            function = hamiltonian
        else:
            matrix = jnp.asarray(hamiltonian)
            if matrix.shape != (dimension, dimension):
                raise ValueError("Hamiltonian dimension must match jumps.")
            function = lambda time: matrix
        self.hamiltonian = function
        self.jumps = operators.astype(jnp.result_type(operators.dtype, 1j))
        self.rates = rate_values
        self.active_jumps = mask
        self.slicing = slicing
        self.evaluation = evaluation
        self.tolerance = float(tolerance)
        self.dimension = dimension
        self.jump_capacity = capacity
        self.plan_id = plan_id


class FiniteCPTPIntegrationResult(StrictModule):
    densities: Array
    superoperators: Array
    choi_matrices: Array
    cp_margins: Array
    trace_preservation_residuals: Array
    density_trace_residuals: Array
    density_hermiticity_residuals: Array
    density_minimum_eigenvalues: Array
    valid_steps: Array
    final_map: FiniteCPTPMap
    valid: Array
    plan_id: str = eqx.field(static=True)
    method_claim: str = eqx.field(static=True)


def _adjoint(value: Array, /) -> Array:
    return jnp.conj(value.T)


def _generator_action(
    hamiltonian: Array, jumps: Array, rates: Array, active: Array, density: Array
) -> Array:
    result = -1j * (hamiltonian @ density - density @ hamiltonian)
    for jump, rate, enabled in zip(jumps, rates, active, strict=True):
        adjoint = _adjoint(jump)
        square = adjoint @ jump
        dissipator = jump @ density @ adjoint - 0.5 * (
            square @ density + density @ square
        )
        result = result + jnp.where(enabled, rate * dissipator, 0.0)
    return result


def _liouvillian(
    hamiltonian: Array, jumps: Array, rates: Array, active: Array, dimension: int
) -> Array:
    basis = jnp.eye(dimension * dimension, dtype=hamiltonian.dtype).reshape(
        dimension * dimension, dimension, dimension
    )
    columns = jax.vmap(
        lambda value: _generator_action(hamiltonian, jumps, rates, active, value).reshape(
            -1
        )
    )(basis)
    return columns.T


def _interval_channel(plan: FiniteLindbladChannelPlan, index: int, /) -> FiniteCPTPMap:
    time = (
        plan.slicing.nodes[index]
        if plan.evaluation == "left"
        else plan.slicing.midpoints[index]
    )
    hamiltonian = jnp.asarray(plan.hamiltonian(time)).astype(plan.jumps.dtype)
    if hamiltonian.shape != (plan.dimension, plan.dimension):
        raise ValueError("Hamiltonian callback returned the wrong finite dimension.")
    rates = plan.rates if plan.rates.ndim == 1 else plan.rates[index]
    generator = _liouvillian(
        hamiltonian, plan.jumps, rates, plan.active_jumps, plan.dimension
    )
    superoperator = jspla.expm(plan.slicing.widths[index] * generator)
    return finite_cptp_from_superoperator(
        superoperator,
        plan.dimension,
        plan.dimension,
        tolerance=plan.tolerance,
    )


def integrate_finite_cptp(
    plan: FiniteLindbladChannelPlan,
    initial_density: ArrayLike,
    /,
) -> FiniteCPTPIntegrationResult:
    """Compose per-interval certified finite maps; never clip or renormalize state."""
    if not isinstance(plan, FiniteLindbladChannelPlan):
        raise TypeError("plan must be FiniteLindbladChannelPlan.")
    density = jnp.asarray(initial_density).astype(plan.jumps.dtype)
    if density.shape != (plan.dimension, plan.dimension):
        raise ValueError("initial_density dimension does not match the plan.")
    identity = finite_cptp_from_superoperator(
        jnp.eye(plan.dimension**2, dtype=density.dtype),
        plan.dimension,
        plan.dimension,
        tolerance=plan.tolerance,
    )
    cumulative = identity
    densities = [density]
    channels = []
    valid_steps = []
    still_valid = jnp.asarray(True)
    for index in range(plan.slicing.num_steps):
        channel = _interval_channel(plan, index)
        step_valid = still_valid & channel.valid
        proposed = apply_finite_cptp(channel, density)
        density = jnp.where(step_valid, proposed, jnp.full_like(proposed, jnp.nan))
        cumulative = compose_finite_cptp(channel, cumulative, tolerance=plan.tolerance)
        still_valid = step_valid & cumulative.valid
        densities.append(density)
        channels.append(channel)
        valid_steps.append(step_valid)
    history = jnp.stack(densities)
    hermitian = 0.5 * (history + jnp.conj(jnp.swapaxes(history, -1, -2)))
    minimum = HermitianSpectrum(
        hermitian,
        tolerance=plan.tolerance,
    ).minimum_eigenvalue
    return FiniteCPTPIntegrationResult(
        densities=history,
        superoperators=jnp.stack([channel.superoperator for channel in channels]),
        choi_matrices=jnp.stack([channel.choi_matrix for channel in channels]),
        cp_margins=jnp.stack(
            [channel.evidence.minimum_choi_eigenvalue for channel in channels]
        ),
        trace_preservation_residuals=jnp.stack(
            [channel.evidence.trace_preservation_residual for channel in channels]
        ),
        density_trace_residuals=jnp.abs(jnp.trace(history, axis1=-2, axis2=-1) - 1.0),
        density_hermiticity_residuals=jnp.max(
            jnp.abs(history - jnp.conj(jnp.swapaxes(history, -1, -2))), axis=(-2, -1)
        ),
        density_minimum_eigenvalues=minimum,
        valid_steps=jnp.stack(valid_steps),
        final_map=cumulative,
        valid=still_valid & jnp.all(jnp.isfinite(history)),
        plan_id=plan.plan_id,
        method_claim=(
            "piecewise-constant-exponential-exact-per-interval"
            if plan.evaluation == "left"
            else "finite-order-midpoint-exponential"
        ),
    )


__all__ = [
    "FiniteCPTPIntegrationResult",
    "FiniteLindbladChannelPlan",
    "integrate_finite_cptp",
]
