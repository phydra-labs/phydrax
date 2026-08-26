#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization.spectral import (
    ChannelStokesPlan,
    PreparedChannelStokesSolver,
    PreparedPseudospectralMethod,
    PseudospectralMethodPlan,
)


class CompiledChannelFlowDynamics(StrictModule):
    """Dealiased rotational channel nonlinearity plus prepared Stokes solves."""

    stokes_plan: ChannelStokesPlan
    spatial_method: PreparedPseudospectralMethod
    forcing: Any
    horizontal_admissibility: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    forcing_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)

    def __init__(
        self,
        stokes_plan: ChannelStokesPlan,
        spatial_method: PreparedPseudospectralMethod,
        /,
        *,
        forcing: Any = None,
        forcing_id: str = "none",
    ):
        if not isinstance(stokes_plan, ChannelStokesPlan):
            raise TypeError("stokes_plan must be a ChannelStokesPlan.")
        if not isinstance(spatial_method, PreparedPseudospectralMethod):
            raise TypeError("spatial_method must be PreparedPseudospectralMethod.")
        if (
            spatial_method.discretization.prepared_id
            != stokes_plan.discretization.prepared_id
        ):
            raise ValueError("Channel Stokes and pseudospectral discretizations differ.")
        if forcing is not None and not callable(forcing):
            raise TypeError("forcing must be callable or None.")
        source_id = "none" if forcing is None else str(forcing_id)
        if not source_id:
            raise ValueError("forcing_id must be non-empty.")
        x_axis, _, z_axis = stokes_plan.discretization.axes
        admissible = (~x_axis.modes.nyquist_mask)[:, None] & (~z_axis.modes.nyquist_mask)[
            None, :
        ]
        state_shape = stokes_plan.discretization.modal_shape + (3,)
        identifier = canonical_fingerprint(
            {
                "kind": "compiled-channel-flow-v1",
                "stokes_plan": stokes_plan.plan_id,
                "spatial_method": spatial_method.prepared_id,
                "forcing": source_id,
                "state_shape": list(state_shape),
            }
        )
        self.stokes_plan = stokes_plan
        self.spatial_method = spatial_method
        self.forcing = forcing
        self.horizontal_admissibility = admissible
        self.state_shape = state_shape
        self.forcing_id = source_id
        self.compilation_id = identifier

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
        if self.forcing is not None:
            forcing = self.validate_state(
                self.forcing(jnp.asarray(time), value, args), owner="Channel forcing"
            )
            result = result + forcing
        return self.admissible_modes(result)

    def prepare_stokes(self, shift: ArrayLike, /) -> PreparedChannelStokesSolver:
        return self.stokes_plan.prepare(shift)


def compile_channel_flow(
    stokes_plan: ChannelStokesPlan,
    method: PseudospectralMethodPlan,
    /,
    *,
    forcing: Any = None,
    forcing_id: str = "none",
) -> CompiledChannelFlowDynamics:
    """Compile one Fourier–Chebyshev–Fourier rotational channel flow."""
    if not isinstance(stokes_plan, ChannelStokesPlan):
        raise TypeError("stokes_plan must be a ChannelStokesPlan.")
    if not isinstance(method, PseudospectralMethodPlan):
        raise TypeError("method must be a PseudospectralMethodPlan.")
    prepared = method.prepare(
        stokes_plan.discretization,
        required_polynomial_degree=2,
        nonlinear=True,
    )
    return CompiledChannelFlowDynamics(
        stokes_plan,
        prepared,
        forcing=forcing,
        forcing_id=forcing_id,
    )


__all__ = ["CompiledChannelFlowDynamics", "compile_channel_flow"]
