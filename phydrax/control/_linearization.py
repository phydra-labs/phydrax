#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._strict import StrictModule
from ..dynamics import TimeGrid
from ..dynamics._system import ContinuousSystem, DiscreteStepContext, DiscreteSystem
from ..linalg import (
    ArraySpace,
    JacobianLinearOperator,
    MaterializationPolicy,
    materialize,
    prepare_linearization,
    PreparedLinearization,
)
from ..linalg._materialization import _require_materialization_budget
from ._dynamics import DifferentialControlDynamics, DiscreteControlDynamics
from ._qp_compiler import LinearQuadraticControlProblem


ControlSystemType = Literal["continuous", "discrete"]
OutputFunction = Callable[[Array, Array, Array, Any], ArrayLike]


class LinearizationProvenance(StrictModule):
    """Static identity of a local input/output linearization."""

    dynamics_id: str = eqx.field(static=True)
    dynamics_method_id: str = eqx.field(static=True)
    linearization_id: str = eqx.field(static=True)
    system_type: ControlSystemType = eqx.field(static=True)


class AffineControlLinearization(StrictModule):
    r"""Batched dense local model at explicit operating points.

    State columns are canonical local coordinates. Discrete state rows are
    ``inverse_retract(nominal_next, perturbed_next)``: perturbed minus nominal
    next state in the nominal-next chart. They are therefore zero at the
    operating point and the identity transition has an identity local state
    Jacobian. Differential state rows are physical rates pulled into the
    operating state's retraction chart. Output rows retain the output callback's
    ordinary Euclidean coordinates.
    Operating-point values retain their declared point/control shapes.

    The matrices are the dense materialization, under an explicit
    `MaterializationPolicy`, of the same `JacobianLinearOperator` actions that
    `PreparedControlLinearization` exposes without materialization. A failed
    discrete transition is never repaired: its values and matrices are NaN and
    its case is invalid.
    """

    operating_time: Array
    operating_state: Array
    operating_control: Array
    dynamics_value: Array
    output_value: Array
    state_matrix: Array
    control_matrix: Array
    affine_offset: Array
    output_matrix: Array
    feedthrough_matrix: Array
    output_offset: Array
    valid: Array
    provenance: LinearizationProvenance
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_local_size: int = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def A(self) -> Array:
        return self.state_matrix

    @property
    def B(self) -> Array:
        return self.control_matrix

    @property
    def C(self) -> Array:
        return self.output_matrix

    @property
    def D(self) -> Array:
        return self.feedthrough_matrix


class PreparedControlLinearization(StrictModule):
    r"""Matrix-free local model of one control system at one operating point.

    ``dynamics_jacobian`` and ``output_jacobian`` act on one joint coordinate
    vector ``concatenate((local_state, control))`` and equal ``[A B]`` and
    ``[C D]`` of `AffineControlLinearization`, with the same local-coordinate
    conventions. Consumers that need only actions, transposes, or solves use
    the operators directly; ``phydrax.linalg.materialize`` produces dense
    matrices only under an explicit `MaterializationPolicy`. A failed discrete
    transition is never repaired: its values and Jacobian actions are NaN and
    ``valid`` is false.
    """

    operating_time: Array
    operating_state: Array
    operating_control: Array
    dynamics_value: Array
    local_dynamics_value: Array
    output_value: Array
    dynamics_jacobian: JacobianLinearOperator
    output_jacobian: JacobianLinearOperator
    valid: Array
    provenance: LinearizationProvenance
    state_shape: tuple[int, ...] = eqx.field(static=True)
    state_local_size: int = eqx.field(static=True)
    control_shape: tuple[int, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] = eqx.field(static=True)


ControlDynamics = DiscreteControlDynamics | DifferentialControlDynamics


def _physical_shape(value: tuple[int, ...], /, *, owner: str) -> tuple[int, ...]:
    shape = tuple(value)
    if any(size <= 0 for size in shape):
        raise ValueError(f"{owner} dimensions must be positive.")
    return shape


def _inexact(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value)
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype("float64")


def _case_shape(array: Array, physical_shape: tuple[int, ...], /, *, owner: str):
    physical_rank = len(physical_shape)
    if not physical_rank:
        return array.shape
    if (
        array.ndim < physical_rank
        or tuple(array.shape[-physical_rank:]) != physical_shape
    ):
        raise ValueError(
            f"{owner} must end in shape {physical_shape}; got {array.shape}."
        )
    return array.shape[:-physical_rank]


def _system_type(dynamics: ControlDynamics, /) -> ControlSystemType:
    if isinstance(dynamics, DiscreteControlDynamics):
        return "discrete"
    if isinstance(dynamics, DifferentialControlDynamics):
        return "continuous"
    raise TypeError(
        "dynamics must be a DiscreteControlDynamics or DifferentialControlDynamics."
    )


def _validate_context(
    system_type: ControlSystemType,
    target_time: ArrayLike | None,
    step_index: ArrayLike | None,
    linearization_id: str,
    /,
) -> None:
    if not isinstance(linearization_id, str) or not linearization_id:
        raise ValueError("linearization_id must be a non-empty string.")
    if system_type == "discrete":
        if target_time is None or step_index is None:
            raise ValueError(
                "Discrete linearization requires target_time and step_index."
            )
    elif target_time is not None or step_index is not None:
        raise ValueError(
            "Differential linearization does not accept discrete step context."
        )


def _require_exact_geometry(dynamics: ControlDynamics, /) -> None:
    geometry = dynamics.system.state_layout.geometry
    if not geometry.supports_exact_inverse or not geometry.supports_exact_differential:
        raise ValueError(
            "Control linearization requires exact inverse-retraction and retraction-differential geometry."
        )


def _point_model(
    dynamics: ControlDynamics,
    system_type: ControlSystemType,
    time: Array,
    target_time: Array,
    step_index: Array,
    state: Array,
    control: Array,
    /,
    *,
    args: Any,
    output: OutputFunction | None,
    linearization_id: str,
) -> tuple[Array, Array, PreparedLinearization, PreparedLinearization]:
    """Prepare one operating point's local dynamics and output linearizations.

    Both maps act on the joint coordinates ``(local state, control)``, so their
    Jacobians are ``[A B]`` and ``[C D]``. Returns the physical dynamics value,
    its success flag, and the two prepared linearizations.
    """
    system = dynamics.system
    geometry = system.state_layout.geometry
    state_shape = dynamics.state_shape
    control_shape = dynamics.control_shape
    local_shape = jnp.shape(geometry.inverse_retract(state, state))
    local_size = system.state_layout.local_size
    if prod(local_shape) != local_size:
        raise ValueError(
            "State geometry inverse_retract output size must match state_layout.local_size."
        )
    control_size = prod(control_shape)
    joint = ArraySpace(
        (local_size + control_size,), dtype=jnp.result_type(state, control)
    )

    def physical(state_value: Array, control_value: Array) -> tuple[Array, Array]:
        if system_type == "discrete":
            assert isinstance(system, DiscreteSystem)
            result = system.evaluate_result(
                DiscreteStepContext(time, target_time, step_index),
                state_value,
                args,
                inputs=control_value,
            )
            accepted = _inexact(result.accepted_state)
            # A failed transition's accepted state is a rollback, not a local
            # model: poison its value and every derivative instead of repairing.
            poison = jnp.where(result.successful, 1.0, jnp.nan).astype(accepted.dtype)
            value = accepted * poison
            successful = jnp.asarray(result.successful)
        else:
            assert isinstance(system, ContinuousSystem)
            value = _inexact(
                system.evaluate(time, state_value, args, inputs=control_value)
            )
            successful = jnp.asarray(True)
        if value.shape != state_shape:
            raise ValueError(
                f"Control dynamics output must have shape {state_shape}; got {value.shape}."
            )
        return value, successful

    value, successful = physical(state, control)

    def perturbed(delta: Array) -> tuple[Array, Array]:
        local = delta[:local_size].astype(state.dtype).reshape(local_shape)
        control_delta = delta[local_size:].astype(control.dtype).reshape(control_shape)
        return jnp.asarray(geometry.retract(state, local)), control + control_delta

    def local_dynamics(delta: Array) -> tuple[Array, Array]:
        perturbed_state, perturbed_control = perturbed(delta)
        perturbed_value, perturbed_successful = physical(
            perturbed_state, perturbed_control
        )
        if system_type == "discrete":
            coordinates = geometry.inverse_retract(value, perturbed_value)
        else:
            rate = geometry.project_tangent(perturbed_state, perturbed_value)
            coordinates = geometry.retraction_inverse_jvp(state, perturbed_state, rate)
        coordinates = jnp.asarray(coordinates)
        if coordinates.size != local_size:
            raise ValueError(
                "State geometry local dynamics output size must match state_layout.local_size."
            )
        return coordinates.reshape((local_size,)), perturbed_successful

    def local_output(delta: Array) -> Array:
        perturbed_state, perturbed_control = perturbed(delta)
        observed = (
            perturbed_state
            if output is None
            else output(time, perturbed_state, perturbed_control, args)
        )
        return _inexact(observed).reshape((-1,))

    zero = jnp.zeros(joint.shape, dtype=joint.dtype)
    dynamics_linearization = prepare_linearization(
        local_dynamics,
        zero,
        source=joint,
        has_aux=True,
        linearization_id=f"{linearization_id}:dynamics",
    )
    output_linearization = prepare_linearization(
        local_output,
        zero,
        source=joint,
        linearization_id=f"{linearization_id}:output",
    )
    return value, successful, dynamics_linearization, output_linearization


def _output_shape(
    dynamics: ControlDynamics,
    output: OutputFunction | None,
    time: Array,
    state: Array,
    control: Array,
    args: Any,
    /,
) -> tuple[int, ...]:
    if output is None:
        return dynamics.state_shape
    return tuple(
        jax.eval_shape(
            lambda t, x, u: output(t, x, u, args),
            time,
            state,
            control,
        ).shape
    )


def _linearize(
    dynamics: ControlDynamics,
    time: ArrayLike,
    state: ArrayLike,
    control: ArrayLike,
    /,
    *,
    args: Any,
    target_time: ArrayLike | None,
    step_index: ArrayLike | None,
    output: OutputFunction | None,
    linearization_id: str,
    materialization: MaterializationPolicy,
) -> AffineControlLinearization:
    system_type = _system_type(dynamics)
    _validate_context(system_type, target_time, step_index, linearization_id)
    if not isinstance(materialization, MaterializationPolicy):
        raise TypeError("materialization must be a MaterializationPolicy.")
    state_shape = _physical_shape(dynamics.state_shape, owner="state_shape")
    control_shape = _physical_shape(dynamics.control_shape, owner="control_shape")
    _require_exact_geometry(dynamics)
    geometry = dynamics.system.state_layout.geometry
    states = _inexact(state)
    controls = _inexact(control)
    times = _inexact(time)
    state_cases = _case_shape(states, state_shape, owner="state")
    control_cases = _case_shape(controls, control_shape, owner="control")
    if system_type == "discrete":
        target_times = _inexact(target_time)
        step_indices = jnp.asarray(step_index, dtype=jnp.int32)
        case_shape = jnp.broadcast_shapes(
            state_cases,
            control_cases,
            times.shape,
            target_times.shape,
            step_indices.shape,
        )
    else:
        case_shape = jnp.broadcast_shapes(state_cases, control_cases, times.shape)
        target_times = times
        step_indices = jnp.zeros(times.shape, dtype=jnp.int32)
    if any(size <= 0 for size in case_shape):
        raise ValueError("Linearization case dimensions must be positive.")
    states = jnp.broadcast_to(states, case_shape + state_shape)
    controls = jnp.broadcast_to(controls, case_shape + control_shape)
    times = jnp.broadcast_to(times, case_shape)
    target_times = jnp.broadcast_to(target_times, case_shape)
    step_indices = jnp.broadcast_to(step_indices, case_shape)

    point_size = prod(state_shape)
    local_size = dynamics.system.state_layout.local_size
    control_size = prod(control_shape)
    case_count = prod(case_shape) if case_shape else 1
    flat_states = states.reshape((case_count, point_size))
    flat_controls = controls.reshape((case_count, control_size))
    flat_times = times.reshape((case_count,))
    flat_target_times = target_times.reshape((case_count,))
    flat_step_indices = step_indices.reshape((case_count,))
    output_shape = _output_shape(
        dynamics,
        output,
        flat_times[0],
        flat_states[0].reshape(state_shape),
        flat_controls[0].reshape(control_shape),
        args,
    )
    output_size = prod(output_shape) if output_shape else 1
    joint_size = local_size + control_size
    dtype = jnp.result_type(states, controls)
    # One policy bounds each dense Jacobian family over the whole case batch.
    _require_materialization_budget(
        case_count * local_size * joint_size, dtype, materialization
    )
    _require_materialization_budget(
        case_count * output_size * joint_size, dtype, materialization
    )

    def dense_point(t, target_t, index, flat_state, flat_control):
        value, successful, dynamics_linearization, output_linearization = _point_model(
            dynamics,
            system_type,
            t,
            target_t,
            index,
            flat_state.reshape(state_shape),
            flat_control.reshape(control_shape),
            args=args,
            output=output,
            linearization_id=linearization_id,
        )
        return (
            value.reshape((point_size,)),
            successful & dynamics_linearization.auxiliary,
            dynamics_linearization.primal,
            materialize(JacobianLinearOperator(dynamics_linearization), materialization),
            output_linearization.primal,
            materialize(JacobianLinearOperator(output_linearization), materialization),
        )

    (
        dynamics_value,
        successful,
        local_dynamics_value,
        dynamics_jacobian,
        output_value,
        output_jacobian,
    ) = jax.vmap(dense_point)(
        flat_times,
        flat_target_times,
        flat_step_indices,
        flat_states,
        flat_controls,
    )
    state_matrix = dynamics_jacobian[..., :local_size]
    control_matrix = dynamics_jacobian[..., local_size:]
    output_matrix = output_jacobian[..., :local_size]
    feedthrough_matrix = output_jacobian[..., local_size:]
    if geometry.trivial:
        affine_offset = (
            dynamics_value
            - ein.contract("...ij,...j->...i", state_matrix, flat_states)
            - ein.contract("...ij,...j->...i", control_matrix, flat_controls)
        )
        output_offset = (
            output_value
            - ein.contract("...ij,...j->...i", output_matrix, flat_states)
            - ein.contract("...ij,...j->...i", feedthrough_matrix, flat_controls)
        )
    else:
        affine_offset = local_dynamics_value
        output_offset = output_value

    finite_parts = (
        flat_times.reshape((case_count, 1)),
        flat_states,
        flat_controls,
        dynamics_value,
        local_dynamics_value,
        output_value,
        state_matrix,
        control_matrix,
        affine_offset,
        output_matrix,
        feedthrough_matrix,
        output_offset,
    )
    valid = successful
    for part in finite_parts:
        valid = valid & jnp.all(jnp.isfinite(part.reshape((case_count, -1))), axis=-1)

    return AffineControlLinearization(
        operating_time=times,
        operating_state=states,
        operating_control=controls,
        dynamics_value=dynamics_value.reshape(case_shape + state_shape),
        output_value=output_value.reshape(case_shape + output_shape),
        state_matrix=state_matrix.reshape(case_shape + (local_size, local_size)),
        control_matrix=control_matrix.reshape(case_shape + (local_size, control_size)),
        affine_offset=affine_offset.reshape(case_shape + (local_size,)),
        output_matrix=output_matrix.reshape(case_shape + (output_size, local_size)),
        feedthrough_matrix=feedthrough_matrix.reshape(
            case_shape + (output_size, control_size)
        ),
        output_offset=output_offset.reshape(case_shape + (output_size,)),
        valid=valid.reshape(case_shape),
        provenance=LinearizationProvenance(
            dynamics_id=dynamics.dynamics_id,
            dynamics_method_id=dynamics.method_id,
            linearization_id=linearization_id,
            system_type=system_type,
        ),
        state_shape=state_shape,
        state_local_size=local_size,
        control_shape=control_shape,
        output_shape=output_shape,
    )


def prepare_control_linearization(
    dynamics: ControlDynamics,
    time: ArrayLike,
    state: ArrayLike,
    control: ArrayLike,
    /,
    *,
    args: Any = None,
    target_time: ArrayLike | None = None,
    step_index: ArrayLike | None = None,
    output: OutputFunction | None = None,
    linearization_id: str = "jax-forward-jvp",
) -> PreparedControlLinearization:
    """Prepare matrix-free local dynamics and output actions at one point.

    Discrete dynamics require ``target_time`` and ``step_index``; differential
    dynamics refuse them. Nothing is materialized.
    """
    system_type = _system_type(dynamics)
    _validate_context(system_type, target_time, step_index, linearization_id)
    state_shape = _physical_shape(dynamics.state_shape, owner="state_shape")
    control_shape = _physical_shape(dynamics.control_shape, owner="control_shape")
    _require_exact_geometry(dynamics)
    states = _inexact(state)
    controls = _inexact(control)
    times = _inexact(time)
    if states.shape != state_shape or controls.shape != control_shape:
        raise ValueError(
            "prepare_control_linearization requires one operating point with state "
            f"shape {state_shape} and control shape {control_shape}."
        )
    if times.shape:
        raise ValueError("prepare_control_linearization requires a scalar time.")
    if system_type == "discrete":
        target_times = _inexact(target_time)
        step_indices = jnp.asarray(step_index, dtype=jnp.int32)
        if target_times.shape or step_indices.shape:
            raise ValueError(
                "prepare_control_linearization requires a scalar target_time and step_index."
            )
    else:
        target_times = times
        step_indices = jnp.zeros((), dtype=jnp.int32)
    value, successful, dynamics_linearization, output_linearization = _point_model(
        dynamics,
        system_type,
        times,
        target_times,
        step_indices,
        states,
        controls,
        args=args,
        output=output,
        linearization_id=linearization_id,
    )
    output_shape = _output_shape(dynamics, output, times, states, controls, args)
    valid = successful & dynamics_linearization.auxiliary
    for part in (
        times,
        states,
        controls,
        value,
        dynamics_linearization.primal,
        output_linearization.primal,
    ):
        valid = valid & jnp.all(jnp.isfinite(part))
    return PreparedControlLinearization(
        operating_time=times,
        operating_state=states,
        operating_control=controls,
        dynamics_value=value,
        local_dynamics_value=dynamics_linearization.primal,
        output_value=output_linearization.primal.reshape(output_shape),
        dynamics_jacobian=JacobianLinearOperator(
            dynamics_linearization,
            operator_id=f"{dynamics.dynamics_id}:{linearization_id}:dynamics-jacobian",
        ),
        output_jacobian=JacobianLinearOperator(
            output_linearization,
            operator_id=f"{dynamics.dynamics_id}:{linearization_id}:output-jacobian",
        ),
        valid=valid,
        provenance=LinearizationProvenance(
            dynamics_id=dynamics.dynamics_id,
            dynamics_method_id=dynamics.method_id,
            linearization_id=linearization_id,
            system_type=system_type,
        ),
        state_shape=state_shape,
        state_local_size=dynamics.system.state_layout.local_size,
        control_shape=control_shape,
        output_shape=output_shape,
    )


def linearize_discrete_dynamics(
    dynamics: DiscreteControlDynamics,
    time: ArrayLike,
    state: ArrayLike,
    control: ArrayLike,
    /,
    *,
    materialization: MaterializationPolicy,
    args: Any = None,
    target_time: ArrayLike,
    step_index: ArrayLike,
    output: OutputFunction | None = None,
    linearization_id: str = "jax-forward-jvp",
) -> AffineControlLinearization:
    """Densely linearize a discrete transition and optional output at points."""

    if not isinstance(dynamics, DiscreteControlDynamics):
        raise TypeError("dynamics must be a DiscreteControlDynamics.")
    return _linearize(
        dynamics,
        time,
        state,
        control,
        args=args,
        target_time=target_time,
        step_index=step_index,
        output=output,
        linearization_id=linearization_id,
        materialization=materialization,
    )


def linearize_differential_dynamics(
    dynamics: DifferentialControlDynamics,
    time: ArrayLike,
    state: ArrayLike,
    control: ArrayLike,
    /,
    *,
    materialization: MaterializationPolicy,
    args: Any = None,
    output: OutputFunction | None = None,
    linearization_id: str = "jax-forward-jvp",
) -> AffineControlLinearization:
    """Densely linearize a differential vector field and optional output."""

    if not isinstance(dynamics, DifferentialControlDynamics):
        raise TypeError("dynamics must be a DifferentialControlDynamics.")
    return _linearize(
        dynamics,
        time,
        state,
        control,
        args=args,
        target_time=None,
        step_index=None,
        output=output,
        linearization_id=linearization_id,
        materialization=materialization,
    )


def linearize_control_dynamics(
    dynamics: ControlDynamics,
    time: ArrayLike,
    state: ArrayLike,
    control: ArrayLike,
    /,
    *,
    materialization: MaterializationPolicy,
    args: Any = None,
    output: OutputFunction | None = None,
    target_time: ArrayLike | None = None,
    step_index: ArrayLike | None = None,
    linearization_id: str = "jax-forward-jvp",
) -> AffineControlLinearization:
    """Densely linearize discrete or differential dynamics at explicit points."""

    _system_type(dynamics)
    return _linearize(
        dynamics,
        time,
        state,
        control,
        args=args,
        target_time=target_time,
        step_index=step_index,
        output=output,
        linearization_id=linearization_id,
        materialization=materialization,
    )


_BRIDGE_OWNED_OPTIONS = frozenset({"dynamics_bias", "dynamics_id", "time_grid"})


def linear_quadratic_problem_from_discrete_dynamics(
    dynamics: DiscreteControlDynamics,
    time_grid: TimeGrid,
    operating_states: ArrayLike,
    operating_controls: ArrayLike,
    initial_state: ArrayLike,
    state_costs: ArrayLike,
    control_costs: ArrayLike,
    terminal_state_cost: ArrayLike,
    /,
    *,
    materialization: MaterializationPolicy,
    args: Any = None,
    linearization_id: str = "jax-forward-jvp",
    **problem_options: Any,
) -> LinearQuadraticControlProblem:
    """Affine LQ problem from a discrete transition linearized along a trajectory.

    Stage ``t`` is linearized at ``operating_states[..., t, :]`` and
    ``operating_controls[..., t, :]`` with step context ``(times[t],
    times[t + 1], t)``, giving ``x[t+1] = A[t] x[t] + B[t] u[t] + c[t]`` in
    absolute Euclidean coordinates. Every stage linearization must be valid;
    a failed accepted-state transition is refused, never repaired. The
    remaining keyword arguments are the costs, bounds, constraints, and
    ``problem_id`` of `LinearQuadraticControlProblem`; the bridge owns
    ``dynamics_bias``, ``dynamics_id``, and ``time_grid``.
    """
    if not isinstance(dynamics, DiscreteControlDynamics):
        raise TypeError("dynamics must be a DiscreteControlDynamics.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be a TimeGrid.")
    owned = sorted(_BRIDGE_OWNED_OPTIONS.intersection(problem_options))
    if owned:
        raise TypeError(f"{owned} are determined by the discrete linearization.")
    if not dynamics.system.state_layout.geometry.trivial:
        raise ValueError(
            "The linear-quadratic bridge requires Euclidean state geometry; "
            "manifold states need an explicit error-state formulation."
        )
    if len(dynamics.state_shape) != 1 or len(dynamics.control_shape) != 1:
        raise ValueError(
            "The linear-quadratic bridge requires rank-one state and control shapes."
        )
    horizon = time_grid.num_steps
    states = _inexact(operating_states)
    controls = _inexact(operating_controls)
    if (
        states.ndim < 2
        or states.shape[-2:] != (horizon,) + dynamics.state_shape
        or controls.ndim < 2
        or controls.shape[-2:] != (horizon,) + dynamics.control_shape
    ):
        raise ValueError(
            "operating_states and operating_controls must end in "
            f"{(horizon,) + dynamics.state_shape} and "
            f"{(horizon,) + dynamics.control_shape}."
        )
    linearization = linearize_discrete_dynamics(
        dynamics,
        time_grid.times[:-1],
        states,
        controls,
        materialization=materialization,
        args=args,
        target_time=time_grid.times[1:],
        step_index=jnp.arange(horizon, dtype=jnp.int32),
        linearization_id=linearization_id,
    )
    # Host admission boundary: the problem is an eager, fingerprinted artifact.
    failed = np.argwhere(~np.asarray(linearization.valid))
    if failed.size:
        raise ValueError(
            "Discrete linearization failed at case/stage indices "
            f"{failed.tolist()}; failed accepted-state transitions are not repaired."
        )
    return LinearQuadraticControlProblem(
        linearization.state_matrix,
        linearization.control_matrix,
        initial_state,
        state_costs,
        control_costs,
        terminal_state_cost,
        dynamics_bias=linearization.affine_offset,
        time_grid=time_grid,
        dynamics_id=f"{dynamics.dynamics_id}:linearized:{linearization_id}",
        **problem_options,
    )


__all__ = [
    "AffineControlLinearization",
    "ControlSystemType",
    "LinearizationProvenance",
    "PreparedControlLinearization",
    "linear_quadratic_problem_from_discrete_dynamics",
    "linearize_control_dynamics",
    "linearize_differential_dynamics",
    "linearize_discrete_dynamics",
    "prepare_control_linearization",
]
