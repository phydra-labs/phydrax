#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._model import AbstractArrayModel
from ..._numerics._ssp_runge_kutta import (
    ssprk33_step_with_evidence,
    StageTransformResult,
)
from ..._strict import StrictModule
from ...discretization.spectral._coordinates import HermitianSpectralCoordinates
from ...dynamics import DiscreteStepContext, InputLayout, StateLayout
from ...dynamics.identification._neural_transition import (
    AbstractDiscreteModelRolloutTransition,
    DiscreteModelRolloutTransitionResult,
)
from ...equations._learned_stress import PreparedPeriodicLearnedStress
from ...equations._mac_incompressible import CompiledMACIncompressibleDynamics
from ...linalg import LinearSolveControl, LinearSolveStatus
from ...nn.operator.data import FunctionSamples, OperatorBatch
from ...nn.operator.engine import AbstractOperatorModel


class FixedGridStressOperatorModel(AbstractArrayModel):
    """Expose one fixed-grid neural operator through a flat array-model ABI."""

    operator: AbstractOperatorModel
    template: OperatorBatch
    source_name: str = eqx.field(static=True)
    target_name: str = eqx.field(static=True)
    source_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractOperatorModel,
        template: OperatorBatch,
        /,
        *,
        source_name: str,
        target_name: str,
    ):
        if not isinstance(operator, AbstractOperatorModel):
            raise TypeError("operator must be an AbstractOperatorModel.")
        if not isinstance(template, OperatorBatch) or template.case_shape:
            raise ValueError("Fixed-grid operator templates must have no case axes.")
        source = str(source_name)
        target = str(target_name)
        if source not in template.inputs:
            raise KeyError("Fixed-grid operator source is absent from its template.")
        samples = template.input(source)
        if samples.values is None:
            raise ValueError("Fixed-grid operator templates require source values.")
        if (
            len(operator.operator_output_specs) != 1
            or target not in operator.operator_output_specs
        ):
            raise ValueError(
                "Fixed-grid stress adapters require one named operator output."
            )
        output = operator.operator_output_specs[target]
        query = template.require_single_query()
        source_shape = tuple(samples.values.shape)
        target_shape = query.sample_shape + (
            () if output.channels == "scalar" else (int(output.channels),)
        )
        self.operator = operator
        self.template = template
        self.source_name = source
        self.target_name = target
        self.source_shape = source_shape
        self.target_shape = target_shape
        self.in_size = prod(source_shape)
        self.out_size = prod(target_shape)

    def __call__(self, values: Array, /, *, key: Array | None = None) -> Array:
        source = self.template.input(self.source_name)
        reshaped = jnp.asarray(values).reshape(self.source_shape)
        inputs = dict(self.template.inputs)
        inputs[self.source_name] = FunctionSamples(
            values=reshaped,
            axes=source.axes,
            coordinates=source.coordinates,
            quadrature_weights=source.quadrature_weights,
            mask=source.mask,
            topology=source.topology,
            support_id=source.support_id,
            measure_id=source.measure_id,
        )
        batch = OperatorBatch(
            inputs=inputs,
            queries=self.template.queries,
            case_axes=(),
            case_shape=(),
        )
        prediction = self.operator.predict_prevalidated(batch, key=key)
        return prediction.field(self.target_name).values.reshape((self.out_size,))


class _CurrentStressPredictor(StrictModule):
    model: AbstractArrayModel
    key: Array | None
    iteration: Array | None
    output_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        model: AbstractArrayModel,
        key: Array | None,
        iteration: Array | None,
        output_shape: tuple[int, ...],
        /,
    ):
        self.model = model
        self.key = key
        self.iteration = iteration
        self.output_shape = output_shape

    def __call__(self, normalized: Array, args: Any = None, /) -> Array:
        del args
        binding = self.model.input_binding()
        point = binding.pack_point((jnp.asarray(normalized).reshape((-1,)),))
        values = binding.call(
            self.model,
            point,
            key=self.key,
            iter_=self.iteration,
            kwargs={},
        )
        return jnp.asarray(values).reshape(self.output_shape)


class PeriodicLearnedStressRolloutTransition(AbstractDiscreteModelRolloutTransition):
    """SSPRK(3,3) periodic dynamics with learned stress at every stage."""

    prepared_stress: PreparedPeriodicLearnedStress
    coordinates: HermitianSpectralCoordinates
    base_rate: Callable = eqx.field(static=True)
    base_rate_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared_stress: PreparedPeriodicLearnedStress,
        coordinates: HermitianSpectralCoordinates,
        base_rate: Callable[[Array, Array, Array | None], Array],
        /,
        *,
        base_rate_id: str,
        state_layout: StateLayout,
        input_layout: InputLayout | None = None,
        step_size: float,
        step_rtol: float = 1e-7,
        step_atol: float = 1e-12,
    ):
        if not isinstance(prepared_stress, PreparedPeriodicLearnedStress):
            raise TypeError("prepared_stress must be PreparedPeriodicLearnedStress.")
        if not isinstance(coordinates, HermitianSpectralCoordinates):
            raise TypeError("coordinates must be HermitianSpectralCoordinates.")
        if (
            prepared_stress.projector.discretization.prepared_id
            != coordinates.discretization.prepared_id
        ):
            raise ValueError("Learned stress and Hermitian coordinates disagree.")
        if state_layout.shape != (coordinates.coordinate_size,):
            raise ValueError("State layout must contain the real Hermitian coordinates.")
        if not callable(base_rate):
            raise TypeError("base_rate must be callable.")
        rate_id = str(base_rate_id).strip()
        step = float(step_size)
        rtol = float(step_rtol)
        atol = float(step_atol)
        if not rate_id:
            raise ValueError("base_rate_id must be non-empty.")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
            raise ValueError("Step tolerances must be finite and nonnegative.")
        self.prepared_stress = prepared_stress
        self.coordinates = coordinates
        self.base_rate = base_rate
        self.base_rate_id = rate_id
        self.state_layout = state_layout
        self.input_layout = input_layout
        self.step_size = step
        self.step_rtol = rtol
        self.step_atol = atol
        self.transition_id = canonical_fingerprint(
            {
                "kind": "periodic-learned-stress-rollout-transition",
                "learned_stress": prepared_stress.prepared_id,
                "coordinates": coordinates.coordinate_id,
                "base_rate": rate_id,
                "state_layout": state_layout.layout_id,
                "input_layout": None if input_layout is None else input_layout.layout_id,
                "step_size": step,
                "step_rtol": rtol,
                "step_atol": atol,
                "integrator": "ssprk33-stagewise",
            }
        )

    def validate_model(self, model: AbstractArrayModel, /) -> None:
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("Learned stress transitions require AbstractArrayModel.")
        feature_size = prod(self.prepared_stress.binding.plan.feature_schema.shape)
        output_size = prod(self.prepared_stress.binding.plan.output_contract.shape)
        if model.in_size != feature_size or model.out_size != output_size:
            raise ValueError(
                "Learned stress model sizes do not match the prepared feature/output ABI."
            )

    def evaluate(
        self,
        model: AbstractArrayModel,
        context: DiscreteStepContext,
        state: Array,
        inputs: Array | None,
        /,
        *,
        key: Array | None,
        iteration: Array | None,
        control: Any = None,
    ) -> DiscreteModelRolloutTransitionResult:
        if control is not None:
            raise ValueError("Periodic learned stress does not use linear solve control.")
        predictor = _CurrentStressPredictor(
            model,
            key,
            iteration,
            self.prepared_stress.binding.plan.output_contract.shape,
        )
        prepared = eqx.tree_at(
            lambda value: value.binding.predictor,
            self.prepared_stress,
            predictor,
        )

        def vector_field(time, real_state, _args):
            modal = self.coordinates.from_real_coordinates(real_state)
            stage = prepared.evaluate(modal)
            base = jnp.asarray(self.base_rate(time, modal, inputs))
            if base.shape != modal.shape:
                raise ValueError(
                    "Periodic base rate must preserve the modal state shape."
                )
            rate = base + stage.projected_rate
            rate = jnp.where(stage.successful, rate, jnp.full_like(rate, jnp.nan))
            rate = self.coordinates.project(rate)
            return self.coordinates.to_real_coordinates(rate)

        def project_stage(index, time, candidate, _args):
            del index, time
            modal = self.coordinates.from_real_coordinates(candidate)
            projected = self.coordinates.project(
                self.prepared_stress.projector.project(modal)
            )
            selected = self.coordinates.to_real_coordinates(projected)
            correction = jnp.linalg.norm(selected - candidate)
            finite = jnp.all(jnp.isfinite(selected))
            return StageTransformResult(
                selected,
                correction > 0.0,
                finite,
                correction,
            )

        result = ssprk33_step_with_evidence(
            vector_field,
            context.source,
            self.coordinates.validate_coordinates(state),
            context.duration,
            None,
            stage_transform=project_stage,
        )
        finite = jnp.all(jnp.isfinite(result.state))
        successful = result.successful & finite
        accepted = jnp.where(successful, result.state, jnp.zeros_like(result.state))
        return DiscreteModelRolloutTransitionResult(
            result.state,
            accepted,
            training_usable=successful,
            physically_converged=successful,
            status=jnp.where(successful, 0, 1),
            residual=result.correction_norm,
            iterations=jnp.asarray(3, dtype=jnp.int32),
            transition_id=self.transition_id,
        )


class MACLearnedRateRolloutTransition(AbstractDiscreteModelRolloutTransition):
    """Explicit MAC transition with a learned rate and controlled projection."""

    dynamics: CompiledMACIncompressibleDynamics
    coarse_relative_residual: float = eqx.field(static=True)

    def __init__(
        self,
        dynamics: CompiledMACIncompressibleDynamics,
        /,
        *,
        state_layout: StateLayout,
        step_size: float,
        coarse_relative_residual: float = 1.0,
        step_rtol: float = 1e-7,
        step_atol: float = 1e-12,
    ):
        if not isinstance(dynamics, CompiledMACIncompressibleDynamics):
            raise TypeError("dynamics must be CompiledMACIncompressibleDynamics.")
        if dynamics.algebraic_les is not None or dynamics.dynamic_les is not None:
            raise ValueError(
                "MAC learned-rate training requires base dynamics without another LES closure."
            )
        if dynamics.projection.constant_route != "iterative":
            raise ValueError(
                "MAC learned-rate refinement requires an iterative pressure route."
            )
        if dynamics.projection.linear_policy.differentiation.mode != "algorithmic":
            raise ValueError(
                "MAC linear refinement initially requires algorithmic differentiation."
            )
        if state_layout.shape != dynamics.state_shape:
            raise ValueError("MAC learned-rate state layout does not match dynamics.")
        residual = float(coarse_relative_residual)
        step = float(step_size)
        rtol = float(step_rtol)
        atol = float(step_atol)
        if not np.isfinite(residual) or residual <= 0.0:
            raise ValueError(
                "coarse_relative_residual must be finite and strictly positive."
            )
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
            raise ValueError("Step tolerances must be finite and nonnegative.")
        self.dynamics = dynamics
        self.coarse_relative_residual = residual
        self.state_layout = state_layout
        self.input_layout = None
        self.step_size = step
        self.step_rtol = rtol
        self.step_atol = atol
        self.transition_id = canonical_fingerprint(
            {
                "kind": "mac-learned-rate-rollout-transition",
                "dynamics": dynamics.compilation_id,
                "projection": dynamics.projection.plan_id,
                "state_layout": state_layout.layout_id,
                "step_size": step,
                "step_rtol": rtol,
                "step_atol": atol,
                "coarse_relative_residual": residual,
                "integrator": "explicit-euler-projected",
                "differentiation": "algorithmic",
            }
        )

    @property
    def supports_linear_refinement(self) -> bool:
        return True

    def validate_model(self, model: AbstractArrayModel, /) -> None:
        if not isinstance(model, AbstractArrayModel):
            raise TypeError("MAC learned rates require AbstractArrayModel.")
        if (
            model.in_size != self.state_layout.size
            or model.out_size != self.state_layout.size
        ):
            raise ValueError(
                "MAC learned-rate model input and output must match the state layout."
            )

    def evaluate(
        self,
        model: AbstractArrayModel,
        context: DiscreteStepContext,
        state: Array,
        inputs: Array | None,
        /,
        *,
        key: Array | None,
        iteration: Array | None,
        control: Any = None,
    ) -> DiscreteModelRolloutTransitionResult:
        if inputs is not None:
            raise ValueError("MAC learned-rate transitions are initially autonomous.")
        if control is not None and not isinstance(control, LinearSolveControl):
            raise TypeError("control must be a LinearSolveControl or None.")
        current = self.dynamics.validate_state(state)
        binding = model.input_binding()
        learned_coordinates = jnp.asarray(
            binding.call(
                model,
                binding.pack_point((current,)),
                key=key,
                iter_=iteration,
                kwargs={},
            ),
            dtype=current.dtype,
        )
        if learned_coordinates.shape != self.dynamics.state_shape:
            raise ValueError("Learned MAC rate does not match the state shape.")
        operators = self.dynamics.momentum.operators
        learned_rate = tuple(operators.velocity_space.unflatten(learned_coordinates))
        boundary = self.dynamics.boundary_stage(context.source, None)
        base_rate = self.dynamics.unconstrained_rate(context.source, current, None)
        combined = self.dynamics.momentum.boundaries.enforce_rate(
            tuple(
                base + learned
                for base, learned in zip(base_rate, learned_rate, strict=True)
            ),
            boundary,
        )
        projected = self.dynamics.projection.project_rate(
            combined,
            boundary_stage=boundary,
            control=control,
        )
        linear = projected.linear
        if linear is None:
            raise RuntimeError("Iterative MAC projection did not return linear evidence.")
        relative_residual = jnp.asarray(linear.diagnostics.relative_residual)
        incomplete = linear.status == int(LinearSolveStatus.MAXIMUM_STEPS_REACHED)
        approximate = (
            incomplete
            & linear.diagnostics.finite
            & jnp.isfinite(relative_residual)
            & (relative_residual <= self.coarse_relative_residual)
        )
        usable_projection = projected.converged | approximate
        rate = tuple(
            jnp.where(projected.converged, accepted, candidate)
            for accepted, candidate in zip(
                projected.rate,
                projected.candidate_rate,
                strict=True,
            )
        )
        velocity = self.dynamics.unpack_velocity(current)
        candidate_velocity = tuple(
            value + context.duration * derivative
            for value, derivative in zip(velocity, rate, strict=True)
        )
        target_boundary = self.dynamics.boundary_stage(context.target, None)
        candidate_velocity = self.dynamics.momentum.boundaries.enforce(
            candidate_velocity,
            target_boundary,
        )
        candidate = operators.velocity_space.flatten(candidate_velocity)
        finite = (
            boundary.successful
            & target_boundary.successful
            & jnp.all(jnp.isfinite(candidate))
        )
        training_usable = usable_projection & finite
        physically_converged = projected.converged & finite
        accepted = jnp.where(training_usable, candidate, jnp.zeros_like(candidate))
        return DiscreteModelRolloutTransitionResult(
            candidate,
            accepted,
            training_usable=training_usable,
            physically_converged=physically_converged,
            status=linear.status,
            residual=relative_residual,
            iterations=linear.diagnostics.iterations,
            transition_id=self.transition_id,
        )


__all__ = [
    "FixedGridStressOperatorModel",
    "MACLearnedRateRolloutTransition",
    "PeriodicLearnedStressRolloutTransition",
]
