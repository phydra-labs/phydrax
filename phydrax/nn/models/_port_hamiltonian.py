# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from .._base import _AbstractBaseModel, _AbstractStructuredInputModel
from .._keys import EvalKey, fold_in_eval_key
from .._utils import _get_size
from ..parameters import (
    PositiveDefiniteTransform,
    SkewSymmetricTransform,
    TransformedParameter,
)
from ._input_convex import InputConvexNetwork


class PortHamiltonianVectorField(_AbstractStructuredInputModel):
    r"""Structured energy-based vector field.

    The autonomous dynamics are ``dx/dt = (J - R) grad(H(x))`` with an exactly
    skew-symmetric interconnection ``J`` and an optional positive-definite
    dissipation ``R``. A configured control input adds ``G u``. The model only
    evaluates the vector field; time integration remains solver-owned.
    """

    energy: _AbstractBaseModel
    interconnection: TransformedParameter
    dissipation: TransformedParameter | None
    control_matrix: Array | None
    in_size: int | tuple[int, ...]
    out_size: int
    state_size: int
    control_size: int | None

    def __init__(
        self,
        *,
        state_size: int,
        energy: _AbstractBaseModel | None = None,
        energy_width: int = 64,
        energy_depth: int = 3,
        control_size: int | None = None,
        dissipative: bool = True,
        initial_damping: float = 1e-2,
        minimum_dissipation_factor: float = 1e-6,
        interconnection_scale: float = 0.1,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        dimension = int(state_size)
        if dimension <= 0:
            raise ValueError("state_size must be positive.")
        control_dimension = None if control_size is None else int(control_size)
        if control_dimension is not None and control_dimension <= 0:
            raise ValueError("control_size must be positive when supplied.")
        damping = float(initial_damping)
        minimum = float(minimum_dissipation_factor)
        scale = float(interconnection_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("interconnection_scale must be finite and non-negative.")
        if bool(dissipative) and (
            not math.isfinite(damping)
            or damping <= 0.0
            or not math.isfinite(minimum)
            or minimum <= 0.0
            or math.sqrt(damping) <= minimum
        ):
            raise ValueError(
                "Dissipative dynamics require positive finite damping and a smaller "
                "positive minimum_dissipation_factor."
            )

        energy_key, interconnection_key, control_key = jr.split(key, 3)
        if energy is None:
            resolved_energy: _AbstractBaseModel = InputConvexNetwork(
                in_size=dimension,
                width_size=energy_width,
                depth=energy_depth,
                key=energy_key,
            )
        else:
            if not isinstance(energy, _AbstractBaseModel):
                raise TypeError("energy must be a Phydrax scalar neural model or None.")
            if energy.out_size != "scalar":
                raise ValueError("energy model must have scalar output.")
            if _get_size(energy.in_size) != dimension:
                raise ValueError("energy model input size must match state_size.")
            resolved_energy = energy
        self.energy = resolved_energy

        raw_interconnection = scale * jr.normal(
            interconnection_key, (dimension, dimension)
        )
        self.interconnection = TransformedParameter(
            raw_interconnection, SkewSymmetricTransform()
        )
        if dissipative:
            row, column = jnp.tril_indices(dimension)
            factor_diagonal = math.sqrt(damping) - minimum
            raw_diagonal = math.log(math.expm1(factor_diagonal))
            raw_dissipation = jnp.where(
                row == column,
                jnp.asarray(raw_diagonal),
                jnp.zeros_like(row, dtype=float),
            )
            self.dissipation = TransformedParameter(
                raw_dissipation, PositiveDefiniteTransform(minimum)
            )
        else:
            self.dissipation = None

        self.control_matrix = (
            None
            if control_dimension is None
            else jr.normal(control_key, (dimension, control_dimension))
            / jnp.sqrt(float(control_dimension))
        )
        self.in_size = (
            dimension if control_dimension is None else (dimension, control_dimension)
        )
        self.out_size = dimension
        self.state_size = dimension
        self.control_size = control_dimension

    def interconnection_matrix(self, /) -> Array:
        """Return the exactly skew-symmetric physical interconnection matrix."""
        return self.interconnection()

    def dissipation_matrix(self, /) -> Array:
        """Return the positive dissipation matrix, or exact zero when disabled."""
        if self.dissipation is None:
            return jnp.zeros(
                (self.state_size, self.state_size),
                dtype=self.interconnection.raw.dtype,
            )
        return self.dissipation()

    def energy_gradient(
        self,
        state: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Differentiate the learned scalar Hamiltonian with respect to state."""
        return jax.grad(lambda value: self.energy(value, key=fold_in_eval_key(key, 0)))(
            state
        )

    def __call__(
        self,
        x: Array | tuple[Array, Array],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if isinstance(x, tuple):
            if len(x) != 2:
                raise TypeError("Controlled dynamics require (state, control).")
            if self.control_matrix is None:
                raise ValueError("This vector field has no configured control input.")
            state, control = x
        else:
            state = x
            control = None
        state_array = jnp.asarray(state)
        if state_array.shape != (self.state_size,):
            raise ValueError(
                f"state must have shape ({self.state_size},), got {state_array.shape}."
            )
        gradient = self.energy_gradient(state_array, key=key)
        vector_field = (
            self.interconnection_matrix() - self.dissipation_matrix()
        ) @ gradient
        if control is not None:
            control_matrix = self.control_matrix
            if control_matrix is None:
                raise RuntimeError(
                    "Controlled dynamics are missing their control matrix."
                )
            control_array = jnp.asarray(control)
            if control_array.shape != (self.control_size,):
                raise ValueError(
                    f"control must have shape ({self.control_size},), "
                    f"got {control_array.shape}."
                )
            vector_field = vector_field + control_matrix @ control_array
        return vector_field

    def energy_rate(
        self,
        x: Array | tuple[Array, Array],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the instantaneous power ``grad(H) dot dx/dt``."""
        state = x[0] if isinstance(x, tuple) else x
        gradient = self.energy_gradient(jnp.asarray(state), key=key)
        return jnp.vdot(gradient, self(x, key=key)).real

    def dissipation_rate(self, state: Array, /) -> Array:
        """Return the non-negative autonomous dissipation ``grad(H)^T R grad(H)``."""
        gradient = self.energy_gradient(jnp.asarray(state))
        return jnp.vdot(gradient, self.dissipation_matrix() @ gradient).real


__all__ = ["PortHamiltonianVectorField"]
