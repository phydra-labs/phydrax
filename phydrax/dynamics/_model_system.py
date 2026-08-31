#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import numpy as np
from jaxtyping import Array, ArrayLike

from .._model import AbstractArrayModel
from .._strict import StrictModule
from ._layout import InputLayout, StateLayout
from ._system import ContinuousSystem, DiscreteSystem


def _value_shape(size: int | tuple[int, ...] | Literal["scalar"], /) -> tuple[int, ...]:
    if size == "scalar":
        return ()
    if isinstance(size, int):
        return (int(size),)
    return tuple(int(value) for value in size)


def _structured_shapes(
    size: int | tuple[int, ...] | Literal["scalar"], /
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(size, tuple):
        raise ValueError("Controlled models must declare two structured input sizes.")
    if len(size) != 2:
        raise ValueError("Controlled models must declare exactly two input sizes.")
    return tuple(_value_shape(value) for value in size)


class ContinuousModelVectorField(StrictModule):
    """Adapt one array model to the canonical continuous vector-field signature."""

    model: AbstractArrayModel
    has_input: bool = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        /,
        *,
        state_layout: StateLayout,
        input_layout: InputLayout | None = None,
    ):
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must be an AbstractArrayModel.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        if _value_shape(model.out_size) != state_layout.shape:
            raise ValueError("model output shape must equal the state layout shape.")
        if input_layout is None:
            if _value_shape(model.in_size) != state_layout.shape:
                raise ValueError("model input shape must equal the state layout shape.")
        else:
            if model.input_binding().input_mode != "structured":
                raise ValueError(
                    "Controlled model systems require structured model input."
                )
            declared = _structured_shapes(model.in_size)
            expected = (state_layout.shape, input_layout.shape)
            if declared != expected:
                raise ValueError(
                    "model structured input shapes must equal state and input layouts."
                )
        self.model = model
        self.has_input = input_layout is not None

    def __call__(
        self,
        time: ArrayLike,
        state: Array,
        *arguments: Any,
    ) -> Array:
        del time
        if self.has_input:
            if len(arguments) != 2:
                raise TypeError(
                    "Controlled continuous vector fields require input and args."
                )
            inputs, args = arguments
            del args
            return self.model((state, inputs), key=None)
        if len(arguments) != 1:
            raise TypeError("Autonomous continuous vector fields require args.")
        del arguments
        return self.model(state, key=None)


class DiscreteModelTransition(StrictModule):
    """Adapt one deterministic pointwise array model to a fixed-step transition."""

    model: AbstractArrayModel
    has_input: bool = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    step_rtol: float = eqx.field(static=True)
    step_atol: float = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        /,
        *,
        state_layout: StateLayout,
        input_layout: InputLayout | None = None,
        step_size: float,
        step_rtol: float = 1e-7,
        step_atol: float = 1e-12,
    ):
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must be an AbstractArrayModel.")
        if not isinstance(state_layout, StateLayout):
            raise TypeError("state_layout must be a StateLayout.")
        if input_layout is not None and not isinstance(input_layout, InputLayout):
            raise TypeError("input_layout must be an InputLayout or None.")
        binding = model.input_binding()
        if binding.batch_mode != "pointwise":
            raise ValueError("Discrete model systems require a pointwise model binding.")
        if _value_shape(model.out_size) != state_layout.shape:
            raise ValueError("model output shape must equal the state layout shape.")
        if input_layout is None:
            if _value_shape(model.in_size) != state_layout.shape:
                raise ValueError("model input shape must equal the state layout shape.")
        else:
            if binding.input_mode != "structured":
                raise ValueError(
                    "Controlled model systems require structured model input."
                )
            declared = _structured_shapes(model.in_size)
            expected = (state_layout.shape, input_layout.shape)
            if declared != expected:
                raise ValueError(
                    "model structured input shapes must equal state and input layouts."
                )
        resolved_step = float(step_size)
        relative_tolerance = float(step_rtol)
        absolute_tolerance = float(step_atol)
        if not np.isfinite(resolved_step) or resolved_step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if (
            not np.isfinite(relative_tolerance)
            or relative_tolerance < 0.0
            or not np.isfinite(absolute_tolerance)
            or absolute_tolerance < 0.0
        ):
            raise ValueError("step_rtol and step_atol must be finite and nonnegative.")
        self.model = model
        self.has_input = input_layout is not None
        self.step_size = resolved_step
        self.step_rtol = relative_tolerance
        self.step_atol = absolute_tolerance

    def __call__(
        self,
        coordinate: ArrayLike,
        state: Array,
        *arguments: Any,
    ) -> Array:
        del coordinate
        binding = self.model.input_binding()
        if self.has_input:
            if len(arguments) != 2:
                raise TypeError("Controlled discrete transitions require input and args.")
            inputs, args = arguments
            del args
            point = binding.pack_point((state, inputs))
        else:
            if len(arguments) != 1:
                raise TypeError("Autonomous discrete transitions require args.")
            del arguments
            point = binding.pack_point((state,))
        return binding.call(
            self.model,
            point,
            key=None,
            iter_=None,
            kwargs={},
        )


def continuous_model_system(
    model: AbstractArrayModel,
    /,
    *,
    state_layout: StateLayout,
    input_layout: InputLayout | None = None,
    system_id: str,
) -> ContinuousSystem:
    """Bind a trainable array model into a canonical continuous system."""
    vector_field = ContinuousModelVectorField(
        model,
        state_layout=state_layout,
        input_layout=input_layout,
    )
    return ContinuousSystem(
        vector_field,
        state_layout=state_layout,
        input_layout=input_layout,
        system_id=system_id,
    )


def discrete_model_system(
    model: AbstractArrayModel,
    /,
    *,
    state_layout: StateLayout,
    input_layout: InputLayout | None = None,
    system_id: str,
    step_size: float,
    step_rtol: float = 1e-7,
    step_atol: float = 1e-12,
) -> DiscreteSystem:
    """Bind a deterministic pointwise model as one complete next-state map."""

    transition = DiscreteModelTransition(
        model,
        state_layout=state_layout,
        input_layout=input_layout,
        step_size=step_size,
        step_rtol=step_rtol,
        step_atol=step_atol,
    )
    return DiscreteSystem(
        transition,
        state_layout=state_layout,
        input_layout=input_layout,
        system_id=system_id,
        step_size=step_size,
        step_rtol=step_rtol,
        step_atol=step_atol,
    )


__all__ = [
    "ContinuousModelVectorField",
    "DiscreteModelTransition",
    "continuous_model_system",
    "discrete_model_system",
]
