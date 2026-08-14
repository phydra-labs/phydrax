#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._euler_maruyama import (
    _euler_maruyama_log_prob,
    _euler_maruyama_mean,
    _euler_maruyama_parameters,
    _euler_maruyama_sample,
    _identifier,
    _scalar,
    EulerMaruyamaParameters,
)
from ._state_space import (
    AbstractTransitionKernel,
    StateSpaceStepContext,
    TransitionSample,
)


if TYPE_CHECKING:
    from ..dynamics import ContinuousSystem
    from ..nn.models import PortHamiltonianVectorField
    from ..solver import WienerTerm


class IsothermalPortHamiltonianDynamics(StrictModule):
    """Itô dynamics satisfying isothermal fluctuation--dissipation balance."""

    field: PortHamiltonianVectorField
    temperature: float = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    constant_structure: bool = eqx.field(static=True)

    def __init__(
        self,
        field: PortHamiltonianVectorField,
        /,
        *,
        temperature: float,
        process_id: str,
    ):
        from ..nn.models import PortHamiltonianVectorField

        if not isinstance(field, PortHamiltonianVectorField):
            raise TypeError("field must be a PortHamiltonianVectorField.")
        if field.control_size is not None:
            raise ValueError(
                "Isothermal equilibrium dynamics do not accept configured control."
            )
        if field.forcing_model is not None:
            raise ValueError(
                "Isothermal equilibrium dynamics do not accept external forcing."
            )
        if not field.dissipative:
            raise ValueError("Isothermal dynamics require nonzero dissipation structure.")
        temperature_value = float(temperature)
        if not math.isfinite(temperature_value) or temperature_value <= 0.0:
            raise ValueError("temperature must be finite and strictly positive.")
        self.field = field
        self.temperature = temperature_value
        self.process_id = _identifier(process_id, owner="process_id")
        self.state_size = field.state_size
        self.constant_structure = (
            field.interconnection_model is None and field.dissipation_model is None
        )

    def _structure_matrix(self, state: Array, /) -> Array:
        return self.field.dissipation_matrix(state) - self.field.interconnection_matrix(
            state
        )

    def _matrix_divergence(self, state: Array, /) -> Array:
        if self.constant_structure:
            return jnp.zeros_like(state)
        basis = jnp.eye(self.state_size, dtype=state.dtype)
        indices = jnp.arange(self.state_size, dtype=jnp.int32)

        def accumulate(total, item):
            direction, index = item
            _, tangent = jax.jvp(
                self._structure_matrix,
                (state,),
                (direction,),
            )
            return total + tangent[:, index], None

        divergence, _ = jax.lax.scan(
            accumulate,
            jnp.zeros_like(state),
            (basis, indices),
        )
        return divergence

    def ito_correction(self, state: ArrayLike, /) -> Array:
        """Return ``temperature * div(R - J)`` exactly by directional AD."""
        state_array = jnp.asarray(state)
        return self.temperature * self._matrix_divergence(state_array)

    def drift(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Evaluate the equilibrium Itô drift."""
        del time, args
        state_array = jnp.asarray(state)
        return self.field(state_array) + self.ito_correction(state_array)

    def diffusion_factor(self, state: ArrayLike, /) -> Array:
        """Return the canonical factor ``sqrt(2 * temperature) L(state)``."""
        factor = self.field.dissipation_factor(jnp.asarray(state))
        scale = jnp.sqrt(jnp.asarray(2.0 * self.temperature, dtype=factor.dtype))
        return scale * factor

    def diffusion(
        self,
        time: ArrayLike,
        state: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        """Evaluate the canonical Wiener coefficient."""
        del time, args
        return self.diffusion_factor(state)

    def diffusion_covariance(self, state: ArrayLike, /) -> Array:
        """Materialize ``2 * temperature * R(state)`` for diagnostics."""
        factor = self.diffusion_factor(state)
        return factor @ factor.T

    def energy_generator(self, state: ArrayLike, /) -> Array:
        """Return the Itô generator applied to the learned energy."""
        state_array = jnp.asarray(state)
        gradient = self.field.energy_gradient(state_array)
        hessian = jax.hessian(self.field.energy)(state_array)
        drift = self.drift(jnp.asarray(0.0), state_array)
        dissipation = self.field.dissipation_matrix(state_array)
        return jnp.vdot(gradient, drift).real + self.temperature * jnp.einsum(
            "ij,ji->", dissipation, hessian
        )

    def stationary_fokker_planck_residual(
        self,
        state: ArrayLike,
        /,
    ) -> Array:
        """Return the Gibbs-density-normalized stationary Fokker--Planck residual."""
        state_array = jnp.asarray(state)

        def normalized_current(value):
            gradient = self.field.energy_gradient(value)
            score = -gradient / self.temperature
            dissipation = self.field.dissipation_matrix(value)
            divergence = self._dissipation_divergence(value)
            drift = self.drift(jnp.asarray(0.0), value)
            return (
                drift
                - self.temperature * divergence
                - self.temperature * (dissipation @ score)
            )

        current = normalized_current(state_array)
        score = -self.field.energy_gradient(state_array) / self.temperature
        current_jacobian = jax.jacfwd(normalized_current)(state_array)
        return -(jnp.trace(current_jacobian) + jnp.vdot(current, score).real)

    def _dissipation_divergence(self, state: Array, /) -> Array:
        if self.field.dissipation_model is None:
            return jnp.zeros_like(state)
        basis = jnp.eye(self.state_size, dtype=state.dtype)
        indices = jnp.arange(self.state_size, dtype=jnp.int32)

        def accumulate(total, item):
            direction, index = item
            _, tangent = jax.jvp(
                self.field.dissipation_matrix,
                (state,),
                (direction,),
            )
            return total + tangent[:, index], None

        divergence, _ = jax.lax.scan(
            accumulate,
            jnp.zeros_like(state),
            (basis, indices),
        )
        return divergence

    def continuous_system(
        self,
        /,
        *,
        system_id: str | None = None,
    ) -> ContinuousSystem:
        """Bind the Itô drift to the canonical continuous-system contract."""
        from ..dynamics import ContinuousSystem, StateLayout

        resolved_id = f"{self.process_id}:ito-drift" if system_id is None else system_id
        return ContinuousSystem(
            self.drift,
            state_layout=StateLayout((self.state_size,)),
            system_id=resolved_id,
        )

    def wiener_term(
        self,
        /,
        *,
        name: str = "thermal",
        basis_id: str | None = None,
    ) -> WienerTerm:
        """Return the canonical thermal Wiener term for differential solvers."""
        from ..solver import WienerTerm

        return WienerTerm(
            name,
            self.diffusion,
            (self.state_size,),
            structure="additive" if self.constant_structure else "general",
            basis_id=basis_id,
        )

    def transition_kernel(
        self,
        /,
        *,
        approximation_id: str = "euler-maruyama",
    ) -> IsothermalPortHamiltonianTransitionKernel:
        """Return a shared-trainable Euler--Maruyama transition kernel."""
        return IsothermalPortHamiltonianTransitionKernel(
            self,
            approximation_id=approximation_id,
        )


class IsothermalPortHamiltonianTransitionKernel(AbstractTransitionKernel):
    """Euler--Maruyama kernel storing one thermodynamic dynamics PyTree."""

    dynamics: IsothermalPortHamiltonianDynamics
    state_shape: tuple[int, ...] = eqx.field(static=True)
    noise_shape: tuple[int, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    noise_size: int = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(
        self,
        dynamics: IsothermalPortHamiltonianDynamics,
        /,
        *,
        approximation_id: str = "euler-maruyama",
    ):
        if not isinstance(dynamics, IsothermalPortHamiltonianDynamics):
            raise TypeError("dynamics must be IsothermalPortHamiltonianDynamics.")
        self.dynamics = dynamics
        self.state_shape = (dynamics.state_size,)
        self.noise_shape = (dynamics.state_size,)
        self.state_size = dynamics.state_size
        self.noise_size = dynamics.state_size
        self.process_id = dynamics.process_id
        self.approximation_id = _identifier(approximation_id, owner="approximation_id")
        self.has_log_density = True

    def parameters(
        self,
        state: ArrayLike,
        t0: ArrayLike,
        t1: ArrayLike,
        context: StateSpaceStepContext,
        /,
    ) -> EulerMaruyamaParameters:
        state_array = jnp.asarray(state)
        if tuple(state_array.shape) != self.state_shape:
            raise ValueError(
                f"state must have shape {self.state_shape}; got {state_array.shape}."
            )
        start = _scalar(t0, owner="t0")
        end = _scalar(t1, owner="t1")
        drift = self.dynamics.drift(start, state_array, context.args)
        coefficient = self.dynamics.diffusion(start, state_array, context.args)
        return _euler_maruyama_parameters(
            state_array,
            drift,
            coefficient,
            end - start,
            state_size=self.state_size,
            valid=context.input_valid,
        )

    def mean(self, state, t0, t1, context, /) -> Array:
        state_array = jnp.asarray(state)
        return _euler_maruyama_mean(
            state_array,
            self.parameters(state_array, t0, t1, context),
            state_shape=self.state_shape,
            state_size=self.state_size,
        )

    def covariance(self, state, t0, t1, context, /) -> Array:
        return self.parameters(state, t0, t1, context).covariance

    def sample(self, key, state, t0, t1, context, /) -> TransitionSample:
        state_array = jnp.asarray(state)
        values, valid = _euler_maruyama_sample(
            key,
            state_array,
            self.parameters(state_array, t0, t1, context),
            state_shape=self.state_shape,
            state_size=self.state_size,
            noise_size=self.noise_size,
        )
        return TransitionSample(
            values=values,
            valid=valid,
            status=jnp.where(valid, 0, 1).astype(jnp.int32),
            process_id=self.process_id,
            approximation_id=self.approximation_id,
        )

    def log_prob(self, next_state, state, t0, t1, context, /) -> Array:
        next_array = jnp.asarray(next_state)
        if tuple(next_array.shape) != self.state_shape:
            raise ValueError(
                f"next_state must have shape {self.state_shape}; got {next_array.shape}."
            )
        state_array = jnp.asarray(state)
        return _euler_maruyama_log_prob(
            next_array,
            state_array,
            self.parameters(state_array, t0, t1, context),
            state_shape=self.state_shape,
            state_size=self.state_size,
        )


__all__ = [
    "IsothermalPortHamiltonianDynamics",
    "IsothermalPortHamiltonianTransitionKernel",
]
