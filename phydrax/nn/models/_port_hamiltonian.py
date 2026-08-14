# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import math
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ..._doc import DOC_KEY0
from ..._model import AbstractArrayModel
from .._base import _AbstractBaseModel, _AbstractStructuredInputModel
from .._keys import EvalKey, fold_in_eval_key
from .._utils import _get_value_shape
from ..parameters import (
    PackedSkewSymmetricTransform,
    PositiveDefiniteTransform,
    PositiveSemidefiniteTransform,
    PositiveTransform,
    TransformedParameter,
)
from ._input_convex import InputConvexNetwork


DissipationStructure = Literal["positive_definite", "positive_semidefinite"]
_ResolvedDissipationStructure = Literal[
    "none", "positive_definite", "positive_semidefinite"
]


def _inverse_softplus(value: float, /) -> float:
    return value if value > 20.0 else math.log(math.expm1(value))


def _validated_model(
    model: AbstractArrayModel,
    /,
    *,
    state_size: int,
    output_shape: tuple[int, ...],
    name: str,
) -> AbstractArrayModel:
    if not isinstance(model, AbstractArrayModel):
        raise TypeError(f"{name} must be a Phydrax array model.")
    if _get_value_shape(model.in_size) != (state_size,):
        raise ValueError(f"{name} input shape must be ({state_size},).")
    if _get_value_shape(model.out_size) != output_shape:
        raise ValueError(f"{name} output shape must be {output_shape}.")
    return model


class FeatureNormPotential(_AbstractBaseModel):
    r"""Coercive scalar potential ``0.5 ||phi(x)||² + beta ||x||²``."""

    features: AbstractArrayModel
    quadratic: TransformedParameter
    in_size: int
    out_size: Literal["scalar"]

    def __init__(
        self,
        features: AbstractArrayModel,
        /,
        *,
        initial_quadratic: float = 1e-2,
        minimum_quadratic: float = 1e-8,
    ):
        if not isinstance(features, AbstractArrayModel):
            raise TypeError("features must be a Phydrax array model.")
        input_shape = _get_value_shape(features.in_size)
        if len(input_shape) != 1:
            raise ValueError("FeatureNormPotential requires a vector input model.")
        initial = float(initial_quadratic)
        minimum = float(minimum_quadratic)
        if (
            not math.isfinite(initial)
            or not math.isfinite(minimum)
            or minimum < 0.0
            or initial <= minimum
        ):
            raise ValueError(
                "initial_quadratic must be finite and exceed a finite "
                "non-negative minimum_quadratic."
            )
        self.features = features
        self.quadratic = TransformedParameter(
            jnp.asarray(_inverse_softplus(initial - minimum)),
            PositiveTransform(minimum),
        )
        self.in_size = int(input_shape[0])
        self.out_size = "scalar"

    def quadratic_coefficient(self, /) -> Array:
        """Return the strictly positive quadratic-tail coefficient."""
        return self.quadratic()

    def __call__(
        self,
        state: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        state_array = jnp.asarray(state)
        if state_array.shape != (self.in_size,):
            raise ValueError(
                f"state must have shape ({self.in_size},), got {state_array.shape}."
            )
        features = jnp.ravel(
            jnp.asarray(self.features(state_array, key=fold_in_eval_key(key, 0)))
        )
        return (
            0.5 * jnp.vdot(features, features).real
            + self.quadratic() * jnp.vdot(state_array, state_array).real
        )


class PortHamiltonianVectorField(_AbstractStructuredInputModel):
    r"""State-dependent energy-based vector field.

    The autonomous dynamics are ``dx/dt = (J(x) - R(x)) grad(H(x)) + f(x)``.
    ``J`` is exactly skew-symmetric and ``R = L L.T`` is positive
    semidefinite or positive definite by construction. A configured control
    input adds ``G(x) u``. Time integration remains solver-owned.
    """

    energy: AbstractArrayModel
    interconnection: TransformedParameter | None
    interconnection_model: AbstractArrayModel | None
    dissipation: TransformedParameter | None
    dissipation_model: AbstractArrayModel | None
    forcing_model: AbstractArrayModel | None
    control_matrix: Array | None
    control_model: AbstractArrayModel | None
    in_size: int | tuple[int, ...]
    out_size: int
    state_size: int
    control_size: int | None
    minimum_dissipation_factor: float
    dissipation_structure: _ResolvedDissipationStructure

    def __init__(
        self,
        *,
        state_size: int,
        energy: AbstractArrayModel | None = None,
        energy_width: int = 64,
        energy_depth: int = 3,
        interconnection_model: AbstractArrayModel | None = None,
        dissipation_model: AbstractArrayModel | None = None,
        forcing_model: AbstractArrayModel | None = None,
        control_model: AbstractArrayModel | None = None,
        control_size: int | None = None,
        dissipative: bool = True,
        dissipation_structure: DissipationStructure = "positive_definite",
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
        if dissipation_structure not in (
            "positive_definite",
            "positive_semidefinite",
        ):
            raise ValueError(
                "dissipation_structure must be 'positive_definite' or "
                "'positive_semidefinite'."
            )
        if not dissipative and dissipation_model is not None:
            raise ValueError(
                "dissipation_model cannot be supplied when dissipative is False."
            )
        if control_model is not None and control_dimension is None:
            raise ValueError("control_model requires control_size.")

        damping = float(initial_damping)
        minimum = float(minimum_dissipation_factor)
        scale = float(interconnection_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("interconnection_scale must be finite and non-negative.")
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("minimum_dissipation_factor must be finite and positive.")
        if dissipative:
            valid_damping = math.isfinite(damping) and damping >= 0.0
            if dissipation_structure == "positive_definite":
                valid_damping = valid_damping and math.sqrt(damping) > minimum
            if not valid_damping:
                raise ValueError(
                    "initial_damping must be finite and non-negative, and "
                    "positive-definite damping must exceed the minimum factor."
                )

        energy_key, interconnection_key, control_key = jr.split(key, 3)
        if energy is None:
            resolved_energy: AbstractArrayModel = InputConvexNetwork(
                in_size=dimension,
                width_size=energy_width,
                depth=energy_depth,
                key=energy_key,
            )
        else:
            resolved_energy = _validated_model(
                energy,
                state_size=dimension,
                output_shape=(),
                name="energy",
            )
        self.energy = resolved_energy

        skew_size = dimension * (dimension - 1) // 2
        if interconnection_model is not None:
            if skew_size == 0:
                raise ValueError("A one-dimensional interconnection is identically zero.")
            self.interconnection_model = _validated_model(
                interconnection_model,
                state_size=dimension,
                output_shape=(skew_size,),
                name="interconnection_model",
            )
            self.interconnection = None
        else:
            full_interconnection = scale * jr.normal(
                interconnection_key, (dimension, dimension)
            )
            skew_interconnection = 0.5 * (full_interconnection - full_interconnection.T)
            row, column = jnp.tril_indices(dimension, k=-1)
            self.interconnection = TransformedParameter(
                skew_interconnection[row, column],
                PackedSkewSymmetricTransform(),
            )
            self.interconnection_model = None

        factor_size = dimension * (dimension + 1) // 2
        resolved_structure: _ResolvedDissipationStructure = (
            dissipation_structure if dissipative else "none"
        )
        self.dissipation_structure = resolved_structure
        self.minimum_dissipation_factor = minimum
        if dissipation_model is not None:
            self.dissipation_model = _validated_model(
                dissipation_model,
                state_size=dimension,
                output_shape=(factor_size,),
                name="dissipation_model",
            )
            self.dissipation = None
        elif dissipative:
            row, column = jnp.tril_indices(dimension)
            if dissipation_structure == "positive_definite":
                factor_diagonal = math.sqrt(damping) - minimum
                raw_diagonal = _inverse_softplus(factor_diagonal)
                transform = PositiveDefiniteTransform(minimum)
            else:
                raw_diagonal = math.sqrt(damping)
                transform = PositiveSemidefiniteTransform()
            raw_dissipation = jnp.where(
                row == column,
                jnp.asarray(raw_diagonal),
                jnp.zeros_like(row, dtype=float),
            )
            self.dissipation = TransformedParameter(raw_dissipation, transform)
            self.dissipation_model = None
        else:
            self.dissipation = None
            self.dissipation_model = None

        self.forcing_model = (
            None
            if forcing_model is None
            else _validated_model(
                forcing_model,
                state_size=dimension,
                output_shape=(dimension,),
                name="forcing_model",
            )
        )
        self.control_model = (
            None
            if control_model is None
            else _validated_model(
                control_model,
                state_size=dimension,
                output_shape=(dimension * int(control_dimension),),
                name="control_model",
            )
        )
        self.control_matrix = (
            None
            if control_dimension is None or control_model is not None
            else jr.normal(control_key, (dimension, control_dimension))
            / jnp.sqrt(float(control_dimension))
        )
        self.in_size = (
            dimension if control_dimension is None else (dimension, control_dimension)
        )
        self.out_size = dimension
        self.state_size = dimension
        self.control_size = control_dimension

    @property
    def dissipative(self) -> bool:
        """Whether the field has a nonzero-capable dissipation factor."""
        return self.dissipation_structure != "none"

    @property
    def state_dependent_structure(self) -> bool:
        """Whether any structural matrix is generated from the current state."""
        return (
            self.interconnection_model is not None or self.dissipation_model is not None
        )

    @property
    def externally_forced(self) -> bool:
        """Whether nonconservative forcing or a control channel is configured."""
        return self.forcing_model is not None or self.control_size is not None

    def _state(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        if value.shape != (self.state_size,):
            raise ValueError(
                f"state must have shape ({self.state_size},), got {value.shape}."
            )
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        return value

    def _component_output(
        self,
        model: AbstractArrayModel,
        state: Array,
        /,
        *,
        size: int,
        key: EvalKey,
        site: int,
        name: str,
    ) -> Array:
        value = jnp.asarray(model(state, key=fold_in_eval_key(key, site)))
        if value.shape != (size,):
            raise ValueError(f"{name} returned shape {value.shape}; expected ({size},).")
        return value

    def interconnection_matrix(
        self,
        state: Array | None = None,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the exactly skew-symmetric physical interconnection matrix."""
        if self.interconnection_model is None:
            if self.interconnection is None:
                raise RuntimeError("Constant interconnection parameters are missing.")
            return self.interconnection()
        if state is None:
            raise ValueError("state is required for a state-dependent interconnection.")
        state_array = self._state(state)
        size = self.state_size * (self.state_size - 1) // 2
        raw = self._component_output(
            self.interconnection_model,
            state_array,
            size=size,
            key=key,
            site=1,
            name="interconnection_model",
        )
        return PackedSkewSymmetricTransform()(raw)

    def dissipation_factor(
        self,
        state: Array | None = None,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the canonical lower factor ``L`` in ``R = L L.T``."""
        if not self.dissipative:
            dtype = float if state is None else self._state(state).dtype
            return jnp.zeros((self.state_size, self.state_size), dtype=dtype)
        if self.dissipation_model is None:
            if self.dissipation is None:
                raise RuntimeError("Constant dissipation parameters are missing.")
            raw = self.dissipation.raw
            transform = self.dissipation.transform
        else:
            if state is None:
                raise ValueError("state is required for state-dependent dissipation.")
            state_array = self._state(state)
            size = self.state_size * (self.state_size + 1) // 2
            raw = self._component_output(
                self.dissipation_model,
                state_array,
                size=size,
                key=key,
                site=2,
                name="dissipation_model",
            )
            transform = (
                PositiveDefiniteTransform(self.minimum_dissipation_factor)
                if self.dissipation_structure == "positive_definite"
                else PositiveSemidefiniteTransform()
            )
        if isinstance(transform, PositiveDefiniteTransform):
            return transform.factor(raw)
        if isinstance(transform, PositiveSemidefiniteTransform):
            return transform.factor(raw)
        raise RuntimeError("Dissipation parameters have an incompatible transform.")

    def dissipation_matrix(
        self,
        state: Array | None = None,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Materialize the positive dissipation matrix for diagnostics."""
        factor = self.dissipation_factor(state, key=key)
        return factor @ factor.T

    def control_map(
        self,
        state: Array | None = None,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the constant or state-dependent control map ``G``."""
        if self.control_size is None:
            raise ValueError("This vector field has no configured control input.")
        if self.control_model is None:
            if self.control_matrix is None:
                raise RuntimeError("Constant control parameters are missing.")
            return self.control_matrix
        if state is None:
            raise ValueError("state is required for a state-dependent control map.")
        state_array = self._state(state)
        raw = self._component_output(
            self.control_model,
            state_array,
            size=self.state_size * self.control_size,
            key=key,
            site=3,
            name="control_model",
        )
        return raw.reshape((self.state_size, self.control_size))

    def forcing_vector(
        self,
        state: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the learned nonconservative forcing, or exact zero."""
        state_array = self._state(state)
        if self.forcing_model is None:
            return jnp.zeros_like(state_array)
        return self._component_output(
            self.forcing_model,
            state_array,
            size=self.state_size,
            key=key,
            site=4,
            name="forcing_model",
        )

    def energy_gradient(
        self,
        state: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Differentiate the learned scalar Hamiltonian with respect to state."""
        state_array = self._state(state)
        return jax.grad(lambda value: self.energy(value, key=fold_in_eval_key(key, 0)))(
            state_array
        )

    def _split_input(
        self, x: Array | tuple[Array, Array], /
    ) -> tuple[Array, Array | None]:
        if isinstance(x, tuple):
            if len(x) != 2:
                raise TypeError("Controlled dynamics require (state, control).")
            if self.control_size is None:
                raise ValueError("This vector field has no configured control input.")
            state, control = x
        else:
            state = x
            control = None
        state_array = self._state(state)
        if control is None:
            return state_array, None
        control_array = jnp.asarray(control)
        if control_array.shape != (self.control_size,):
            raise ValueError(
                f"control must have shape ({self.control_size},), "
                f"got {control_array.shape}."
            )
        return state_array, control_array

    def _evaluate(
        self,
        state: Array,
        control: Array | None,
        /,
        *,
        key: EvalKey,
    ) -> tuple[Array, Array, Array, Array]:
        gradient = self.energy_gradient(state, key=key)
        interconnection = self.interconnection_matrix(state, key=key)
        factor = self.dissipation_factor(state, key=key)
        external = self.forcing_vector(state, key=key)
        if control is not None:
            external = external + self.control_map(state, key=key) @ control
        dissipative_gradient = factor @ (factor.T @ gradient)
        vector_field = interconnection @ gradient - dissipative_gradient + external
        return gradient, factor, external, vector_field

    def __call__(
        self,
        x: Array | tuple[Array, Array],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        state, control = self._split_input(x)
        return self._evaluate(state, control, key=key)[-1]

    def energy_rate(
        self,
        x: Array | tuple[Array, Array],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the fixed-input state-directional rate ``grad(H) dot dx/dt``."""
        state, control = self._split_input(x)
        gradient, _, _, vector_field = self._evaluate(state, control, key=key)
        return jnp.vdot(gradient, vector_field).real

    def dissipation_rate(
        self,
        state: Array,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return the non-negative dissipation ``||L.T grad(H)||²``."""
        state_array = self._state(state)
        gradient = self.energy_gradient(state_array, key=key)
        factor = self.dissipation_factor(state_array, key=key)
        projected = factor.T @ gradient
        return jnp.vdot(projected, projected).real

    def input_power(
        self,
        x: Array | tuple[Array, Array],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return power supplied by control and nonconservative forcing."""
        state, control = self._split_input(x)
        gradient, _, external, _ = self._evaluate(state, control, key=key)
        return jnp.vdot(gradient, external).real

    def energy_balance_residual(
        self,
        x: Array | tuple[Array, Array],
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        """Return ``energy_rate + dissipation_rate - input_power``."""
        state, control = self._split_input(x)
        gradient, factor, external, vector_field = self._evaluate(state, control, key=key)
        projected = factor.T @ gradient
        return (
            jnp.vdot(gradient, vector_field).real
            + jnp.vdot(projected, projected).real
            - jnp.vdot(gradient, external).real
        )


__all__ = ["DissipationStructure", "FeatureNormPotential", "PortHamiltonianVectorField"]
