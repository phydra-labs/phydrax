#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._geometry_precision import GeometryPrecisionPolicy
from .._strict import StrictModule
from ..discretization.spectral import (
    ChannelStokesPlan,
    PreparedChannelStokesSolver,
    PreparedPseudospectralMethod,
    PseudospectralMethodPlan,
)
from ._incompressible import IncompressibleFlowProblem


class ChannelVelocityDiagnostics(StrictModule):
    """Constraint and kinetic-energy evidence for one channel velocity state."""

    kinetic_energy: Array
    divergence_norm: Array
    wall_residual: Array
    finite: Array
    valid: Array



class CompiledChannelFlowDynamics(StrictModule):
    """Dealiased rotational channel nonlinearity plus prepared Stokes solves."""

    problem: IncompressibleFlowProblem
    stokes_plan: ChannelStokesPlan
    spatial_method: PreparedPseudospectralMethod
    horizontal_admissibility: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    source_hash: str = eqx.field(static=True)

    def __init__(
        self,
        problem: IncompressibleFlowProblem,
        stokes_plan: ChannelStokesPlan,
        spatial_method: PreparedPseudospectralMethod,
        /,
    ):
        if not isinstance(problem, IncompressibleFlowProblem):
            raise TypeError("problem must be an IncompressibleFlowProblem.")
        if problem.spatial_dimension != 3:
            raise ValueError("Channel flow requires a three-dimensional problem.")
        if not isinstance(stokes_plan, ChannelStokesPlan):
            raise TypeError("stokes_plan must be a ChannelStokesPlan.")
        if not bool(jnp.array_equal(problem.viscosity, stokes_plan.viscosity)):
            raise ValueError("Problem and channel Stokes viscosities must match exactly.")
        if not isinstance(spatial_method, PreparedPseudospectralMethod):
            raise TypeError("spatial_method must be PreparedPseudospectralMethod.")
        if (
            spatial_method.discretization.prepared_id
            != stokes_plan.discretization.prepared_id
        ):
            raise ValueError("Channel Stokes and pseudospectral discretizations differ.")
        x_axis, _, z_axis = stokes_plan.discretization.axes
        admissible = (~x_axis.modes.nyquist_mask)[:, None] & (~z_axis.modes.nyquist_mask)[
            None, :
        ]
        state_shape = stokes_plan.discretization.modal_shape + (3,)
        identifier = canonical_fingerprint(
            {
                "kind": "compiled-channel-flow-v2",
                "problem": problem.problem_id,
                "stokes_plan": stokes_plan.plan_id,
                "spatial_method": spatial_method.prepared_id,
                "state_shape": list(state_shape),
            }
        )
        self.problem = problem
        self.stokes_plan = stokes_plan
        self.spatial_method = spatial_method
        self.horizontal_admissibility = admissible
        self.state_shape = state_shape
        self.compilation_id = identifier
        self.source_hash = problem.problem_id

    @property
    def discretization(self):
        return self.stokes_plan.discretization

    def validate_state(
        self, state: ArrayLike, /, *, owner: str = "Channel state"
    ) -> Array:
        value = jnp.asarray(state)
        if value.shape != self.state_shape:
            raise ValueError(
                f"{owner} must have shape {self.state_shape}; got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError(f"{owner} must use complex modal coefficients.")
        return value

    def admissible_modes(self, state: ArrayLike, /) -> Array:
        value = self.validate_state(state)
        return value * self.horizontal_admissibility[:, None, :, None]

    def project_state(self, values: ArrayLike, /) -> Array:
        physical = jnp.asarray(values)
        expected = self.discretization.physical_shape + (3,)
        if physical.shape != expected:
            raise ValueError(
                f"Physical channel velocity must have shape {expected}; got {physical.shape}."
            )
        return self.admissible_modes(self.discretization.project(physical))

    def reconstruct_state(self, state: ArrayLike, /) -> Array:
        return self.discretization.reconstruct(self.admissible_modes(state))

    def state_diagnostics(self, state: ArrayLike, /) -> ChannelVelocityDiagnostics:
        value = self.admissible_modes(state)
        physical = self.discretization.reconstruct(value)
        speed_squared = jnp.sum(jnp.real(physical * jnp.conj(physical)), axis=-1)
        kinetic_energy = 0.5 * jnp.sum(
            self.discretization.quadrature_weights * speed_squared
        )
        x_axis, _, z_axis = self.discretization.axes
        kx = (
            2.0
            * jnp.pi
            * x_axis.modes.mode_numbers
            / x_axis.length
        )[:, None, None]
        kz = (
            2.0
            * jnp.pi
            * z_axis.modes.mode_numbers
            / z_axis.length
        )[None, None, :]
        divergence = (
            1j * kx * value[..., 0]
            + self.discretization.modal_derivative(value[..., 1], axis=1)
            + 1j * kz * value[..., 2]
        )
        precision = GeometryPrecisionPolicy()
        divergence_norm = precision.norm(divergence.reshape((-1,)))
        lower = physical[:, 0, :, :]
        upper = physical[:, -1, :, :]
        wall_residual = jnp.maximum(
            precision.norm(
                (lower - self.stokes_plan.lower_wall_velocity).reshape((-1,))
            ),
            precision.norm(
                (upper - self.stokes_plan.upper_wall_velocity).reshape((-1,))
            ),
        )
        finite = jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(physical))
        valid = (
            finite
            & (divergence_norm <= self.stokes_plan.constraint_tolerance)
            & (wall_residual <= self.stokes_plan.constraint_tolerance)
        )
        return ChannelVelocityDiagnostics(
            kinetic_energy=kinetic_energy,
            divergence_norm=divergence_norm,
            wall_residual=wall_residual,
            finite=finite,
            valid=valid,
        )


    def nonlinear(self, time: Array, state: ArrayLike, args: Any = None, /) -> Array:
        value = self.admissible_modes(state)
        dealiasing = self.spatial_method.dealiasing
        evaluation = dealiasing.evaluation
        padded = dealiasing.embed(value)
        velocity = evaluation.reconstruct(padded)
        derivatives = tuple(
            evaluation.modal_derivative(padded, axis=axis) for axis in range(3)
        )
        vorticity_modal = jnp.stack(
            (
                derivatives[1][..., 2] - derivatives[2][..., 1],
                derivatives[2][..., 0] - derivatives[0][..., 2],
                derivatives[0][..., 1] - derivatives[1][..., 0],
            ),
            axis=-1,
        )
        vorticity = evaluation.reconstruct(vorticity_modal)
        result = -dealiasing.project(jnp.cross(vorticity, velocity, axis=-1))
        if self.problem.forcing is not None:
            forcing = self.validate_state(
                self.problem.forcing(jnp.asarray(time), value, args),
                owner="Channel forcing",
            )
            result = result + forcing
        return self.admissible_modes(result)

    def prepare_stokes(self, shift: ArrayLike, /) -> PreparedChannelStokesSolver:
        return self.stokes_plan.prepare(shift)


def compile_channel_flow(
    problem: IncompressibleFlowProblem,
    stokes_plan: ChannelStokesPlan,
    method: PseudospectralMethodPlan,
    /,
) -> CompiledChannelFlowDynamics:
    """Compile one Fourier–Chebyshev–Fourier rotational channel flow."""
    if not isinstance(problem, IncompressibleFlowProblem):
        raise TypeError("problem must be an IncompressibleFlowProblem.")
    if not isinstance(stokes_plan, ChannelStokesPlan):
        raise TypeError("stokes_plan must be a ChannelStokesPlan.")
    if not isinstance(method, PseudospectralMethodPlan):
        raise TypeError("method must be a PseudospectralMethodPlan.")
    prepared = method.prepare(
        stokes_plan.discretization,
        required_polynomial_degree=2,
        nonlinear=True,
    )
    return CompiledChannelFlowDynamics(problem, stokes_plan, prepared)


__all__ = [
    "ChannelVelocityDiagnostics",
    "CompiledChannelFlowDynamics",
    "compile_channel_flow",
]
