#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical QP compilation for finite-horizon linear control."""

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._bounds import Bounds
from .._strict import StrictModule
from ..dynamics import TimeGrid
from ..optim._programming import (
    ConvexProgramResult,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    ConvexWarmStart,
    prepare_convex_program,
    PreparedConvexProgram,
    QuadraticProgram,
    refresh_convex_program,
    solve_convex_program,
)
from ..sparse import EdgeRelation, SparseLinearMap
from ._parameterization import PiecewiseConstantControlParameterization
from ._problem import _identifier
from ._trajectory import (
    CONTROL_DYNAMICS_FAILED,
    CONTROL_INFEASIBLE,
    CONTROL_SUCCESS,
    ControlTrajectory,
)


SliceTuple: TypeAlias = tuple[slice, ...]

ControlQPRepresentation: TypeAlias = Literal["dense", "sparse"]


class LinearControlCompilationPolicy(StrictModule):
    """Explicit dense or structural sparse control-QP representation."""

    representation: ControlQPRepresentation = eqx.field(static=True)

    def __init__(self, representation: ControlQPRepresentation = "dense", /):
        if representation not in ("dense", "sparse"):
            raise ValueError("representation must be 'dense' or 'sparse'.")
        self.representation = representation


def _exact_array_shape(
    value: ArrayLike,
    shape: tuple[int, ...],
    name: str,
    /,
    *,
    dtype: jnp.dtype | None = None,
) -> Array:
    array = jnp.asarray(value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real-valued.")
    if dtype is not None:
        return array.astype(dtype)
    if not jnp.issubdtype(array.dtype, jnp.inexact):
        array = array.astype(float)
    return array


def _optional_exact_array(
    value: ArrayLike | None,
    shape: tuple[int, ...],
    name: str,
    /,
    *,
    dtype: jnp.dtype,
) -> Array | None:
    if value is None:
        return None
    return _exact_array_shape(value, shape, name, dtype=dtype)


def _required_array(value: Array | None, name: str, /) -> Array:
    if value is None:
        raise RuntimeError(f"Missing validated {name} array.")
    return value


def _positive_semidefinite_symmetric_part(
    value: Array,
    name: str,
    tolerance: float,
    /,
) -> Array:
    symmetric = 0.5 * (value + jnp.swapaxes(value, -1, -2))
    eigenvalues = jnp.linalg.eigvalsh(symmetric)
    if bool(jnp.any(jnp.isfinite(eigenvalues) & (eigenvalues < -tolerance))):
        raise ValueError(
            f"{name} must be positive semidefinite; indefinite costs are unsupported."
        )
    return symmetric


class LinearQuadraticControlProblem(StrictModule):
    r"""An explicit finite-horizon affine linear-quadratic control problem.

    The stage convention is

    ``x[t+1] = A[t] x[t] + B[t] u[t] + c[t]``

    with cost

    ``x[t]ᵀQ[t]x[t]/2 + u[t]ᵀR[t]u[t]/2 + x[t]ᵀN[t]u[t]``
    ``+ q[t]ᵀx[t] + r[t]ᵀu[t] + d[t]``.

    Every stage array includes its physical stage axis explicitly, after the
    case axes. State bounds include all ``horizon + 1`` state nodes. General
    stage constraints act on ``(x[t], u[t])`` for ``t = 0, ..., horizon - 1``.
    No input is clipped, projected, regularized, or repaired.
    """

    dynamics_matrices: Array
    control_matrices: Array
    initial_state: Array
    state_costs: Array
    control_costs: Array
    terminal_state_cost: Array
    dynamics_bias: Array
    state_control_cross: Array
    state_linear: Array
    control_linear: Array
    stage_constants: Array
    terminal_linear: Array
    terminal_constant: Array
    state_lower_bounds: Array | None
    state_upper_bounds: Array | None
    control_lower_bounds: Array | None
    control_upper_bounds: Array | None
    stage_equality_state_matrix: Array | None
    stage_equality_control_matrix: Array | None
    stage_equality_rhs: Array | None
    stage_inequality_state_matrix: Array | None
    stage_inequality_control_matrix: Array | None
    stage_inequality_rhs: Array | None
    terminal_equality_matrix: Array | None
    terminal_equality_rhs: Array | None
    terminal_inequality_matrix: Array | None
    terminal_inequality_rhs: Array | None
    time_grid: TimeGrid
    case_shape: tuple[int, ...] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_stage_equalities: int = eqx.field(static=True)
    num_stage_inequalities: int = eqx.field(static=True)
    num_terminal_equalities: int = eqx.field(static=True)
    num_terminal_inequalities: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics_matrices: ArrayLike,
        control_matrices: ArrayLike,
        initial_state: ArrayLike,
        state_costs: ArrayLike,
        control_costs: ArrayLike,
        terminal_state_cost: ArrayLike,
        /,
        *,
        dynamics_bias: ArrayLike | None = None,
        state_control_cross: ArrayLike | None = None,
        state_linear: ArrayLike | None = None,
        control_linear: ArrayLike | None = None,
        stage_constants: ArrayLike | None = None,
        terminal_linear: ArrayLike | None = None,
        terminal_constant: ArrayLike | None = None,
        state_lower_bounds: ArrayLike | None = None,
        state_upper_bounds: ArrayLike | None = None,
        control_lower_bounds: ArrayLike | None = None,
        control_upper_bounds: ArrayLike | None = None,
        stage_equality_state_matrix: ArrayLike | None = None,
        stage_equality_control_matrix: ArrayLike | None = None,
        stage_equality_rhs: ArrayLike | None = None,
        stage_inequality_state_matrix: ArrayLike | None = None,
        stage_inequality_control_matrix: ArrayLike | None = None,
        stage_inequality_rhs: ArrayLike | None = None,
        terminal_equality_matrix: ArrayLike | None = None,
        terminal_equality_rhs: ArrayLike | None = None,
        terminal_inequality_matrix: ArrayLike | None = None,
        terminal_inequality_rhs: ArrayLike | None = None,
        time_grid: TimeGrid | None = None,
        problem_id: str = "control:linear-quadratic",
        dynamics_id: str = "control:dynamics:affine-discrete",
    ):
        a = jnp.asarray(dynamics_matrices)
        if a.ndim < 3 or a.shape[-1] != a.shape[-2]:
            raise ValueError(
                "dynamics_matrices must have shape "
                "case_shape + (horizon, state_size, state_size)."
            )
        case_shape = tuple(int(size) for size in a.shape[:-3])
        horizon = int(a.shape[-3])
        state_size = int(a.shape[-1])
        if horizon < 1 or state_size < 1:
            raise ValueError("horizon and state_size must be positive.")
        b = jnp.asarray(control_matrices)
        if (
            b.ndim < 3
            or tuple(b.shape[:-3]) != case_shape
            or int(b.shape[-3]) != horizon
            or int(b.shape[-2]) != state_size
        ):
            raise ValueError(
                "control_matrices must have shape "
                "case_shape + (horizon, state_size, control_size)."
            )
        control_size = int(b.shape[-1])
        if control_size < 1:
            raise ValueError("control_size must be positive.")

        required = (
            a,
            b,
            jnp.asarray(initial_state),
            jnp.asarray(state_costs),
            jnp.asarray(control_costs),
            jnp.asarray(terminal_state_cost),
        )
        if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in required):
            raise TypeError("Linear-quadratic control data must be real-valued.")
        dtype = jnp.result_type(*(value.dtype for value in required), jnp.float32)
        a = _exact_array_shape(
            a,
            case_shape + (horizon, state_size, state_size),
            "dynamics_matrices",
            dtype=dtype,
        )
        b = _exact_array_shape(
            b,
            case_shape + (horizon, state_size, control_size),
            "control_matrices",
            dtype=dtype,
        )
        initial = _exact_array_shape(
            initial_state,
            case_shape + (state_size,),
            "initial_state",
            dtype=dtype,
        )
        q = _exact_array_shape(
            state_costs,
            case_shape + (horizon, state_size, state_size),
            "state_costs",
            dtype=dtype,
        )
        r = _exact_array_shape(
            control_costs,
            case_shape + (horizon, control_size, control_size),
            "control_costs",
            dtype=dtype,
        )
        q_terminal = _exact_array_shape(
            terminal_state_cost,
            case_shape + (state_size, state_size),
            "terminal_state_cost",
            dtype=dtype,
        )
        zeros = lambda shape: jnp.zeros(shape, dtype=dtype)
        c = (
            zeros(case_shape + (horizon, state_size))
            if dynamics_bias is None
            else _exact_array_shape(
                dynamics_bias,
                case_shape + (horizon, state_size),
                "dynamics_bias",
                dtype=dtype,
            )
        )
        cross = (
            zeros(case_shape + (horizon, state_size, control_size))
            if state_control_cross is None
            else _exact_array_shape(
                state_control_cross,
                case_shape + (horizon, state_size, control_size),
                "state_control_cross",
                dtype=dtype,
            )
        )
        q_linear = (
            zeros(case_shape + (horizon, state_size))
            if state_linear is None
            else _exact_array_shape(
                state_linear,
                case_shape + (horizon, state_size),
                "state_linear",
                dtype=dtype,
            )
        )
        r_linear = (
            zeros(case_shape + (horizon, control_size))
            if control_linear is None
            else _exact_array_shape(
                control_linear,
                case_shape + (horizon, control_size),
                "control_linear",
                dtype=dtype,
            )
        )
        constants = (
            zeros(case_shape + (horizon,))
            if stage_constants is None
            else _exact_array_shape(
                stage_constants,
                case_shape + (horizon,),
                "stage_constants",
                dtype=dtype,
            )
        )
        terminal_linear_value = (
            zeros(case_shape + (state_size,))
            if terminal_linear is None
            else _exact_array_shape(
                terminal_linear,
                case_shape + (state_size,),
                "terminal_linear",
                dtype=dtype,
            )
        )
        terminal_constant_value = (
            zeros(case_shape)
            if terminal_constant is None
            else _exact_array_shape(
                terminal_constant,
                case_shape,
                "terminal_constant",
                dtype=dtype,
            )
        )

        state_shape = case_shape + (horizon + 1, state_size)
        control_shape = case_shape + (horizon, control_size)
        state_lower = _optional_exact_array(
            state_lower_bounds,
            state_shape,
            "state_lower_bounds",
            dtype=dtype,
        )
        state_upper = _optional_exact_array(
            state_upper_bounds,
            state_shape,
            "state_upper_bounds",
            dtype=dtype,
        )
        control_lower = _optional_exact_array(
            control_lower_bounds,
            control_shape,
            "control_lower_bounds",
            dtype=dtype,
        )
        control_upper = _optional_exact_array(
            control_upper_bounds,
            control_shape,
            "control_upper_bounds",
            dtype=dtype,
        )

        (
            stage_eq_state,
            stage_eq_control,
            stage_eq_rhs,
            num_stage_equalities,
        ) = self._stage_constraints(
            stage_equality_state_matrix,
            stage_equality_control_matrix,
            stage_equality_rhs,
            case_shape=case_shape,
            horizon=horizon,
            state_size=state_size,
            control_size=control_size,
            dtype=dtype,
            name="stage equality",
        )
        (
            stage_ineq_state,
            stage_ineq_control,
            stage_ineq_rhs,
            num_stage_inequalities,
        ) = self._stage_constraints(
            stage_inequality_state_matrix,
            stage_inequality_control_matrix,
            stage_inequality_rhs,
            case_shape=case_shape,
            horizon=horizon,
            state_size=state_size,
            control_size=control_size,
            dtype=dtype,
            name="stage inequality",
        )
        terminal_eq_matrix, terminal_eq_rhs, num_terminal_equalities = (
            self._terminal_constraints(
                terminal_equality_matrix,
                terminal_equality_rhs,
                case_shape=case_shape,
                state_size=state_size,
                dtype=dtype,
                name="terminal equality",
            )
        )
        terminal_ineq_matrix, terminal_ineq_rhs, num_terminal_inequalities = (
            self._terminal_constraints(
                terminal_inequality_matrix,
                terminal_inequality_rhs,
                case_shape=case_shape,
                state_size=state_size,
                dtype=dtype,
                name="terminal inequality",
            )
        )

        if time_grid is None:
            time_grid = TimeGrid(
                jnp.arange(horizon + 1, dtype=dtype),
                time_id=f"{problem_id}:time",
            )
        elif not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be a TimeGrid or None.")
        if time_grid.num_steps != horizon:
            raise ValueError(
                f"time_grid must contain {horizon + 1} times for this horizon."
            )

        self.dynamics_matrices = a
        self.control_matrices = b
        self.initial_state = initial
        self.state_costs = q
        self.control_costs = r
        self.terminal_state_cost = q_terminal
        self.dynamics_bias = c
        self.state_control_cross = cross
        self.state_linear = q_linear
        self.control_linear = r_linear
        self.stage_constants = constants
        self.terminal_linear = terminal_linear_value
        self.terminal_constant = terminal_constant_value
        self.state_lower_bounds = state_lower
        self.state_upper_bounds = state_upper
        self.control_lower_bounds = control_lower
        self.control_upper_bounds = control_upper
        self.stage_equality_state_matrix = stage_eq_state
        self.stage_equality_control_matrix = stage_eq_control
        self.stage_equality_rhs = stage_eq_rhs
        self.stage_inequality_state_matrix = stage_ineq_state
        self.stage_inequality_control_matrix = stage_ineq_control
        self.stage_inequality_rhs = stage_ineq_rhs
        self.terminal_equality_matrix = terminal_eq_matrix
        self.terminal_equality_rhs = terminal_eq_rhs
        self.terminal_inequality_matrix = terminal_ineq_matrix
        self.terminal_inequality_rhs = terminal_ineq_rhs
        self.time_grid = time_grid
        self.case_shape = case_shape
        self.horizon = horizon
        self.state_size = state_size
        self.control_size = control_size
        self.num_stage_equalities = num_stage_equalities
        self.num_stage_inequalities = num_stage_inequalities
        self.num_terminal_equalities = num_terminal_equalities
        self.num_terminal_inequalities = num_terminal_inequalities
        self.problem_id = _identifier(problem_id, "problem_id")
        self.dynamics_id = _identifier(dynamics_id, "dynamics_id")

    @staticmethod
    def _stage_constraints(
        state_matrix: ArrayLike | None,
        control_matrix: ArrayLike | None,
        rhs: ArrayLike | None,
        *,
        case_shape: tuple[int, ...],
        horizon: int,
        state_size: int,
        control_size: int,
        dtype: jnp.dtype,
        name: str,
    ) -> tuple[Array | None, Array | None, Array | None, int]:
        if rhs is None:
            if state_matrix is not None or control_matrix is not None:
                raise ValueError(f"{name} matrices require a right-hand side.")
            return None, None, None, 0
        rhs_value = jnp.asarray(rhs)
        if rhs_value.ndim != len(case_shape) + 2:
            raise ValueError(f"{name}_rhs must have shape case_shape + (horizon, rows).")
        if tuple(rhs_value.shape[:-2]) != case_shape or rhs_value.shape[-2] != horizon:
            raise ValueError(f"{name}_rhs must have shape case_shape + (horizon, rows).")
        rows = int(rhs_value.shape[-1])
        if rows < 1:
            raise ValueError(f"{name} must contain at least one row per stage.")
        rhs_value = _exact_array_shape(
            rhs_value,
            case_shape + (horizon, rows),
            f"{name}_rhs",
            dtype=dtype,
        )
        expected_state = case_shape + (horizon, rows, state_size)
        expected_control = case_shape + (horizon, rows, control_size)
        state_value = (
            jnp.zeros(expected_state, dtype=dtype)
            if state_matrix is None
            else _exact_array_shape(
                state_matrix,
                expected_state,
                f"{name}_state_matrix",
                dtype=dtype,
            )
        )
        control_value = (
            jnp.zeros(expected_control, dtype=dtype)
            if control_matrix is None
            else _exact_array_shape(
                control_matrix,
                expected_control,
                f"{name}_control_matrix",
                dtype=dtype,
            )
        )
        return state_value, control_value, rhs_value, rows

    @staticmethod
    def _terminal_constraints(
        matrix: ArrayLike | None,
        rhs: ArrayLike | None,
        *,
        case_shape: tuple[int, ...],
        state_size: int,
        dtype: jnp.dtype,
        name: str,
    ) -> tuple[Array | None, Array | None, int]:
        if matrix is None:
            if rhs is not None:
                raise ValueError(
                    f"{name} matrix and right-hand side are required together."
                )
            return None, None, 0
        if rhs is None:
            raise ValueError(f"{name} matrix and right-hand side are required together.")
        matrix_value = jnp.asarray(matrix)
        if matrix_value.ndim != len(case_shape) + 2:
            raise ValueError(
                f"{name}_matrix must have shape case_shape + (rows, state_size)."
            )
        rows = int(matrix_value.shape[-2])
        expected_matrix = case_shape + (rows, state_size)
        expected_rhs = case_shape + (rows,)
        if rows < 1:
            raise ValueError(f"{name} must contain at least one row.")
        return (
            _exact_array_shape(
                matrix_value,
                expected_matrix,
                f"{name}_matrix",
                dtype=dtype,
            ),
            _exact_array_shape(rhs, expected_rhs, f"{name}_rhs", dtype=dtype),
            rows,
        )


LinearQuadraticControlSpecification = LinearQuadraticControlProblem


class LinearControlDecisionLayout(StrictModule):
    """Immutable slices for ``x[0], x[1:H+1], u[0:H]`` decisions."""

    initial_state_slice: slice = eqx.field(static=True)
    state_stage_slices: SliceTuple = eqx.field(static=True)
    control_stage_slices: SliceTuple = eqx.field(static=True)
    all_states_slice: slice = eqx.field(static=True)
    all_controls_slice: slice = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    control_size: int = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)

    def __init__(self, horizon: int, state_size: int, control_size: int, /):
        if horizon < 1 or state_size < 1 or control_size < 1:
            raise ValueError("horizon, state_size, and control_size must be positive.")
        state_end = (horizon + 1) * state_size
        control_end = state_end + horizon * control_size
        self.initial_state_slice = slice(0, state_size)
        self.state_stage_slices = tuple(
            slice(stage * state_size, (stage + 1) * state_size)
            for stage in range(1, horizon + 1)
        )
        self.control_stage_slices = tuple(
            slice(
                state_end + stage * control_size,
                state_end + (stage + 1) * control_size,
            )
            for stage in range(horizon)
        )
        self.all_states_slice = slice(0, state_end)
        self.all_controls_slice = slice(state_end, control_end)
        self.horizon = horizon
        self.state_size = state_size
        self.control_size = control_size
        self.num_variables = control_end

    @property
    def state_slices(self) -> SliceTuple:
        """All state-node slices, including the separately named initial node."""
        return (self.initial_state_slice,) + self.state_stage_slices

    def state_slice(self, stage: int, /) -> slice:
        if not isinstance(stage, int) or not 0 <= stage <= self.horizon:
            raise IndexError(f"state stage must lie in [0, {self.horizon}].")
        return self.state_slices[stage]

    def control_slice(self, stage: int, /) -> slice:
        if not isinstance(stage, int) or not 0 <= stage < self.horizon:
            raise IndexError(f"control stage must lie in [0, {self.horizon}).")
        return self.control_stage_slices[stage]

    def decode(self, primal: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(primal)
        if value.ndim < 1 or int(value.shape[-1]) != self.num_variables:
            raise ValueError(
                f"primal must end in shape ({self.num_variables},); got {value.shape}."
            )
        states = value[..., self.all_states_slice].reshape(
            value.shape[:-1] + (self.horizon + 1, self.state_size)
        )
        controls = value[..., self.all_controls_slice].reshape(
            value.shape[:-1] + (self.horizon, self.control_size)
        )
        return states, controls

    def encode(self, states: ArrayLike, controls: ArrayLike, /) -> Array:
        state_value = jnp.asarray(states)
        control_value = jnp.asarray(controls)
        expected_state_tail = (self.horizon + 1, self.state_size)
        expected_control_tail = (self.horizon, self.control_size)
        if state_value.ndim < 2 or tuple(state_value.shape[-2:]) != expected_state_tail:
            raise ValueError(f"states must end in shape {expected_state_tail}.")
        if (
            control_value.ndim < 2
            or tuple(control_value.shape[-2:]) != expected_control_tail
            or state_value.shape[:-2] != control_value.shape[:-2]
        ):
            raise ValueError(
                "controls must share the state batch and end in shape "
                f"{expected_control_tail}."
            )
        return jnp.concatenate(
            (
                state_value.reshape(state_value.shape[:-2] + (-1,)),
                control_value.reshape(control_value.shape[:-2] + (-1,)),
            ),
            axis=-1,
        )


class LinearControlBoundLayout(StrictModule):
    """Decision-coordinate provenance for native state and control bounds."""

    state_lower_slices: SliceTuple = eqx.field(static=True)
    state_upper_slices: SliceTuple = eqx.field(static=True)
    control_lower_slices: SliceTuple = eqx.field(static=True)
    control_upper_slices: SliceTuple = eqx.field(static=True)

    def __init__(
        self,
        specification: LinearQuadraticControlProblem,
        decision: LinearControlDecisionLayout,
        /,
    ):
        if not isinstance(specification, LinearQuadraticControlProblem):
            raise TypeError("specification must be a LinearQuadraticControlProblem.")
        if not isinstance(decision, LinearControlDecisionLayout):
            raise TypeError("decision must be a LinearControlDecisionLayout.")
        self.state_lower_slices = (
            decision.state_slices if specification.state_lower_bounds is not None else ()
        )
        self.state_upper_slices = (
            decision.state_slices if specification.state_upper_bounds is not None else ()
        )
        self.control_lower_slices = (
            decision.control_stage_slices
            if specification.control_lower_bounds is not None
            else ()
        )
        self.control_upper_slices = (
            decision.control_stage_slices
            if specification.control_upper_bounds is not None
            else ()
        )


class LinearControlConstraintLayout(StrictModule):
    """Immutable row provenance for true equality and polyhedral constraints."""

    initial_condition_slice: slice = eqx.field(static=True)
    dynamics_slices: SliceTuple = eqx.field(static=True)
    stage_equality_slices: SliceTuple = eqx.field(static=True)
    terminal_equality_slice: slice | None = eqx.field(static=True)
    stage_inequality_slices: SliceTuple = eqx.field(static=True)
    terminal_inequality_slice: slice | None = eqx.field(static=True)
    num_equalities: int = eqx.field(static=True)
    num_inequalities: int = eqx.field(static=True)

    def __init__(self, specification: LinearQuadraticControlProblem, /):
        if not isinstance(specification, LinearQuadraticControlProblem):
            raise TypeError("specification must be a LinearQuadraticControlProblem.")
        horizon = specification.horizon
        state_size = specification.state_size
        equality_cursor = state_size
        self.initial_condition_slice = slice(0, state_size)
        self.dynamics_slices = tuple(
            slice(
                equality_cursor + stage * state_size,
                equality_cursor + (stage + 1) * state_size,
            )
            for stage in range(horizon)
        )
        equality_cursor += horizon * state_size
        stage_equalities = specification.num_stage_equalities
        self.stage_equality_slices = tuple(
            slice(
                equality_cursor + stage * stage_equalities,
                equality_cursor + (stage + 1) * stage_equalities,
            )
            for stage in range(horizon)
        )
        equality_cursor += horizon * stage_equalities
        terminal_equalities = specification.num_terminal_equalities
        self.terminal_equality_slice = (
            slice(equality_cursor, equality_cursor + terminal_equalities)
            if terminal_equalities
            else None
        )
        equality_cursor += terminal_equalities
        self.num_equalities = equality_cursor

        stage_inequalities = specification.num_stage_inequalities
        self.stage_inequality_slices = (
            tuple(
                slice(
                    stage * stage_inequalities,
                    (stage + 1) * stage_inequalities,
                )
                for stage in range(horizon)
            )
            if stage_inequalities
            else ()
        )
        inequality_cursor = horizon * stage_inequalities
        terminal_inequalities = specification.num_terminal_inequalities
        self.terminal_inequality_slice = (
            slice(inequality_cursor, inequality_cursor + terminal_inequalities)
            if terminal_inequalities
            else None
        )
        inequality_cursor += terminal_inequalities
        self.num_inequalities = inequality_cursor


def _sparse_map_from_mask(matrix: Array, mask: np.ndarray, /) -> SparseLinearMap:
    rows, columns = np.nonzero(mask)
    relation = EdgeRelation(
        jnp.asarray(columns, dtype=jnp.int32),
        jnp.asarray(rows, dtype=jnp.int32),
        source_size=mask.shape[1],
        target_size=mask.shape[0],
    )
    coefficients = matrix[..., rows, columns]
    return SparseLinearMap(relation, coefficients)


def _control_sparse_operators(
    specification: LinearQuadraticControlProblem,
    decision: LinearControlDecisionLayout,
    constraints: LinearControlConstraintLayout,
    quadratic: Array,
    equality: Array,
    inequality: Array,
    /,
) -> tuple[SparseLinearMap, SparseLinearMap, SparseLinearMap]:
    variables = decision.num_variables
    quadratic_mask = np.zeros((variables, variables), dtype=bool)
    for stage in range(specification.horizon):
        state = decision.state_slice(stage)
        control = decision.control_slice(stage)
        quadratic_mask[state, state] = True
        quadratic_mask[control, control] = True
        quadratic_mask[state, control] = True
        quadratic_mask[control, state] = True
    terminal = decision.state_slice(specification.horizon)
    quadratic_mask[terminal, terminal] = True

    equality_mask = np.zeros((constraints.num_equalities, variables), dtype=bool)
    initial = constraints.initial_condition_slice
    for index in range(specification.state_size):
        equality_mask[
            initial.start + index, decision.initial_state_slice.start + index
        ] = True
    for stage, rows in enumerate(constraints.dynamics_slices):
        previous = decision.state_slice(stage)
        following = decision.state_slice(stage + 1)
        control = decision.control_slice(stage)
        equality_mask[rows, previous] = True
        equality_mask[rows, control] = True
        for index in range(specification.state_size):
            equality_mask[rows.start + index, following.start + index] = True
        if specification.num_stage_equalities:
            stage_rows = constraints.stage_equality_slices[stage]
            equality_mask[stage_rows, previous] = True
            equality_mask[stage_rows, control] = True
    if constraints.terminal_equality_slice is not None:
        equality_mask[constraints.terminal_equality_slice, terminal] = True

    inequality_mask = np.zeros((constraints.num_inequalities, variables), dtype=bool)
    for stage, rows in enumerate(constraints.stage_inequality_slices):
        inequality_mask[rows, decision.state_slice(stage)] = True
        inequality_mask[rows, decision.control_slice(stage)] = True
    if constraints.terminal_inequality_slice is not None:
        inequality_mask[constraints.terminal_inequality_slice, terminal] = True
    return (
        _sparse_map_from_mask(quadratic, quadratic_mask),
        _sparse_map_from_mask(equality, equality_mask),
        _sparse_map_from_mask(inequality, inequality_mask),
    )


class LinearControlQPCompilation(StrictModule):
    """A canonical QP together with lossless control/constraint provenance."""

    quadratic_program: QuadraticProgram
    decision_layout: LinearControlDecisionLayout
    constraint_layout: LinearControlConstraintLayout
    bound_layout: LinearControlBoundLayout
    sparse_quadratic: SparseLinearMap | None
    sparse_equality: SparseLinearMap | None
    sparse_inequality: SparseLinearMap | None
    specification: LinearQuadraticControlProblem
    objective_constant: Array
    compiler_id: str = eqx.field(static=True)
    representation: ControlQPRepresentation = eqx.field(static=True)

    @property
    def qp(self) -> QuadraticProgram:
        return self.quadratic_program

    @property
    def layout(self) -> LinearControlDecisionLayout:
        return self.decision_layout

    def decode(self, primal: ArrayLike, /) -> tuple[Array, Array]:
        return self.decision_layout.decode(primal)


class PreparedLinearControlQP(StrictModule):
    """Compiled control layout paired with reusable prepared convex-program state."""

    compilation: LinearControlQPCompilation
    prepared: PreparedConvexProgram

    def __init__(
        self,
        compilation: LinearControlQPCompilation,
        prepared: PreparedConvexProgram,
        /,
    ):
        if not isinstance(compilation, LinearControlQPCompilation):
            raise TypeError("compilation must be a LinearControlQPCompilation.")
        if not isinstance(prepared, PreparedConvexProgram):
            raise TypeError("prepared must be a PreparedConvexProgram.")
        if prepared.program is not compilation.quadratic_program:
            raise ValueError("Prepared program must be bound to the compilation QP.")
        self.compilation = compilation
        self.prepared = prepared


class LinearControlQPSolution(StrictModule):
    """Decoded QP solution with exact primal arrays and solver provenance."""

    compilation: LinearControlQPCompilation
    qp_result: ConvexProgramResult
    trajectory: ControlTrajectory
    policy: PiecewiseConstantControlParameterization
    parameters: Array
    objective: Array
    valid: Array
    status: Array
    solution_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    @property
    def states(self) -> Array:
        return self.trajectory.states

    @property
    def controls(self) -> Array:
        return self.parameters

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(ConvexProgramStatus.OPTIMAL))


def compile_linear_quadratic_control(
    specification: LinearQuadraticControlProblem,
    /,
    *,
    cost_tolerance: float = 1e-10,
    compilation_policy: LinearControlCompilationPolicy | None = None,
) -> LinearControlQPCompilation:
    """Compile an affine finite-horizon problem without condensing or repair."""
    if not isinstance(specification, LinearQuadraticControlProblem):
        raise TypeError("specification must be a LinearQuadraticControlProblem.")
    tolerance = float(cost_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("cost_tolerance must be finite and non-negative.")
    selected_compilation = (
        LinearControlCompilationPolicy()
        if compilation_policy is None
        else compilation_policy
    )
    if not isinstance(selected_compilation, LinearControlCompilationPolicy):
        raise TypeError(
            "compilation_policy must be a LinearControlCompilationPolicy or None."
        )
    layout = LinearControlDecisionLayout(
        specification.horizon,
        specification.state_size,
        specification.control_size,
    )
    constraints = LinearControlConstraintLayout(specification)
    bound_layout = LinearControlBoundLayout(specification, layout)
    dtype = specification.dynamics_matrices.dtype
    batch = specification.case_shape
    stage_hessian = jnp.concatenate(
        (
            jnp.concatenate(
                (specification.state_costs, specification.state_control_cross),
                axis=-1,
            ),
            jnp.concatenate(
                (
                    jnp.swapaxes(specification.state_control_cross, -1, -2),
                    specification.control_costs,
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )
    stage_hessian = _positive_semidefinite_symmetric_part(
        stage_hessian, "joint stage costs", tolerance
    )
    state_costs = stage_hessian[
        ..., : specification.state_size, : specification.state_size
    ]
    state_control_cross = stage_hessian[
        ..., : specification.state_size, specification.state_size :
    ]
    control_costs = stage_hessian[
        ..., specification.state_size :, specification.state_size :
    ]
    terminal_state_cost = _positive_semidefinite_symmetric_part(
        specification.terminal_state_cost,
        "terminal_state_cost",
        tolerance,
    )
    quadratic = jnp.zeros(
        batch + (layout.num_variables, layout.num_variables), dtype=dtype
    )
    linear = jnp.zeros(batch + (layout.num_variables,), dtype=dtype)

    for stage in range(specification.horizon):
        state_slice = layout.state_slice(stage)
        control_slice = layout.control_slice(stage)
        quadratic = quadratic.at[..., state_slice, state_slice].add(
            state_costs[..., stage, :, :]
        )
        quadratic = quadratic.at[..., control_slice, control_slice].add(
            control_costs[..., stage, :, :]
        )
        cross = state_control_cross[..., stage, :, :]
        quadratic = quadratic.at[..., state_slice, control_slice].add(cross)
        quadratic = quadratic.at[..., control_slice, state_slice].add(
            jnp.swapaxes(cross, -1, -2)
        )
        linear = linear.at[..., state_slice].add(
            specification.state_linear[..., stage, :]
        )
        linear = linear.at[..., control_slice].add(
            specification.control_linear[..., stage, :]
        )
    terminal_state_slice = layout.state_slice(specification.horizon)
    quadratic = quadratic.at[..., terminal_state_slice, terminal_state_slice].add(
        terminal_state_cost
    )
    linear = linear.at[..., terminal_state_slice].add(specification.terminal_linear)

    equality_matrix = jnp.zeros(
        batch + (constraints.num_equalities, layout.num_variables), dtype=dtype
    )
    equality_rhs = jnp.zeros(batch + (constraints.num_equalities,), dtype=dtype)
    identity_state = jnp.eye(specification.state_size, dtype=dtype)
    equality_matrix = equality_matrix.at[
        ..., constraints.initial_condition_slice, layout.initial_state_slice
    ].set(identity_state)
    equality_rhs = equality_rhs.at[..., constraints.initial_condition_slice].set(
        specification.initial_state
    )
    for stage in range(specification.horizon):
        row = constraints.dynamics_slices[stage]
        equality_matrix = equality_matrix.at[..., row, layout.state_slice(stage + 1)].set(
            identity_state
        )
        equality_matrix = equality_matrix.at[..., row, layout.state_slice(stage)].set(
            -specification.dynamics_matrices[..., stage, :, :]
        )
        equality_matrix = equality_matrix.at[..., row, layout.control_slice(stage)].set(
            -specification.control_matrices[..., stage, :, :]
        )
        equality_rhs = equality_rhs.at[..., row].set(
            specification.dynamics_bias[..., stage, :]
        )
        if specification.num_stage_equalities:
            stage_row = constraints.stage_equality_slices[stage]
            equality_matrix = equality_matrix.at[
                ..., stage_row, layout.state_slice(stage)
            ].set(
                _required_array(
                    specification.stage_equality_state_matrix,
                    "stage equality state matrix",
                )[..., stage, :, :]
            )
            equality_matrix = equality_matrix.at[
                ..., stage_row, layout.control_slice(stage)
            ].set(
                _required_array(
                    specification.stage_equality_control_matrix,
                    "stage equality control matrix",
                )[..., stage, :, :]
            )
            equality_rhs = equality_rhs.at[..., stage_row].set(
                _required_array(
                    specification.stage_equality_rhs,
                    "stage equality right-hand side",
                )[..., stage, :]
            )
    if specification.num_terminal_equalities:
        terminal_row = constraints.terminal_equality_slice
        equality_matrix = equality_matrix.at[..., terminal_row, terminal_state_slice].set(
            specification.terminal_equality_matrix
        )
        equality_rhs = equality_rhs.at[..., terminal_row].set(
            specification.terminal_equality_rhs
        )

    inequality_matrix = jnp.zeros(
        batch + (constraints.num_inequalities, layout.num_variables), dtype=dtype
    )
    inequality_rhs = jnp.zeros(batch + (constraints.num_inequalities,), dtype=dtype)
    for stage, row in enumerate(constraints.stage_inequality_slices):
        inequality_matrix = inequality_matrix.at[..., row, layout.state_slice(stage)].set(
            _required_array(
                specification.stage_inequality_state_matrix,
                "stage inequality state matrix",
            )[..., stage, :, :]
        )
        inequality_matrix = inequality_matrix.at[
            ..., row, layout.control_slice(stage)
        ].set(
            _required_array(
                specification.stage_inequality_control_matrix,
                "stage inequality control matrix",
            )[..., stage, :, :]
        )
        inequality_rhs = inequality_rhs.at[..., row].set(
            _required_array(
                specification.stage_inequality_rhs,
                "stage inequality right-hand side",
            )[..., stage, :]
        )
    if specification.num_terminal_inequalities:
        terminal_row = constraints.terminal_inequality_slice
        inequality_matrix = inequality_matrix.at[
            ..., terminal_row, terminal_state_slice
        ].set(specification.terminal_inequality_matrix)
        inequality_rhs = inequality_rhs.at[..., terminal_row].set(
            specification.terminal_inequality_rhs
        )

    lower_bounds = jnp.full(batch + (layout.num_variables,), -jnp.inf, dtype=dtype)
    upper_bounds = jnp.full(batch + (layout.num_variables,), jnp.inf, dtype=dtype)
    if specification.state_lower_bounds is not None:
        lower_bounds = lower_bounds.at[..., layout.all_states_slice].set(
            specification.state_lower_bounds.reshape(
                batch + ((specification.horizon + 1) * specification.state_size,)
            )
        )
    if specification.state_upper_bounds is not None:
        upper_bounds = upper_bounds.at[..., layout.all_states_slice].set(
            specification.state_upper_bounds.reshape(
                batch + ((specification.horizon + 1) * specification.state_size,)
            )
        )
    if specification.control_lower_bounds is not None:
        lower_bounds = lower_bounds.at[..., layout.all_controls_slice].set(
            specification.control_lower_bounds.reshape(
                batch + (specification.horizon * specification.control_size,)
            )
        )
    if specification.control_upper_bounds is not None:
        upper_bounds = upper_bounds.at[..., layout.all_controls_slice].set(
            specification.control_upper_bounds.reshape(
                batch + (specification.horizon * specification.control_size,)
            )
        )

    qp = QuadraticProgram(
        quadratic,
        linear,
        equality_matrix=equality_matrix,
        equality_rhs=equality_rhs,
        inequality_matrix=inequality_matrix,
        inequality_rhs=inequality_rhs,
        bounds=Bounds(lower_bounds, upper_bounds),
        problem_id=f"{specification.problem_id}:qp",
        convexity_evidence="verified",
    )
    objective_constant = (
        jnp.sum(specification.stage_constants, axis=-1) + specification.terminal_constant
    )
    sparse_quadratic = None
    sparse_equality = None
    sparse_inequality = None
    if selected_compilation.representation == "sparse":
        sparse_quadratic, sparse_equality, sparse_inequality = _control_sparse_operators(
            specification,
            layout,
            constraints,
            quadratic,
            equality_matrix,
            inequality_matrix,
        )
    return LinearControlQPCompilation(
        quadratic_program=qp,
        decision_layout=layout,
        constraint_layout=constraints,
        bound_layout=bound_layout,
        sparse_quadratic=sparse_quadratic,
        sparse_equality=sparse_equality,
        sparse_inequality=sparse_inequality,
        specification=specification,
        objective_constant=objective_constant,
        representation=selected_compilation.representation,
        compiler_id="control:qp-compiler:linear-multiple-shooting",
    )


def prepare_linear_quadratic_control(
    specification: LinearQuadraticControlProblem,
    /,
    *,
    policy: ConvexSolvePolicy | None = None,
    cost_tolerance: float = 1e-10,
    compilation_policy: LinearControlCompilationPolicy | None = None,
) -> PreparedLinearControlQP:
    """Compile and prepare one reusable finite-horizon control QP."""

    compilation = compile_linear_quadratic_control(
        specification,
        cost_tolerance=cost_tolerance,
        compilation_policy=compilation_policy,
    )
    prepared = prepare_convex_program(compilation.quadratic_program, policy)
    return PreparedLinearControlQP(compilation, prepared)


def refresh_linear_quadratic_control(
    prepared: PreparedLinearControlQP,
    specification: LinearQuadraticControlProblem,
    /,
    *,
    cost_tolerance: float = 1e-10,
    compilation_policy: LinearControlCompilationPolicy | None = None,
) -> PreparedLinearControlQP:
    """Refresh control coefficients without changing decision or constraint topology."""

    if not isinstance(prepared, PreparedLinearControlQP):
        raise TypeError("prepared must be a PreparedLinearControlQP.")
    selected_compilation = (
        LinearControlCompilationPolicy(prepared.compilation.representation)
        if compilation_policy is None
        else compilation_policy
    )
    compilation = compile_linear_quadratic_control(
        specification,
        cost_tolerance=cost_tolerance,
        compilation_policy=selected_compilation,
    )
    refreshed = refresh_convex_program(
        prepared.prepared,
        compilation.quadratic_program,
    )
    return PreparedLinearControlQP(compilation, refreshed)


def decode_linear_control_solution(
    compilation: LinearControlQPCompilation,
    result: ConvexProgramResult,
    /,
    *,
    solution_id: str | None = None,
) -> LinearControlQPSolution:
    """Decode exactly the primal returned by a canonical QP solver."""
    if not isinstance(compilation, LinearControlQPCompilation):
        raise TypeError("compilation must be a LinearControlQPCompilation.")
    if not isinstance(result, ConvexProgramResult):
        raise TypeError("result must be a ConvexProgramResult.")
    qp = compilation.quadratic_program
    if result.batch_shape != qp.batch_shape:
        raise ValueError("QP result batch shape does not match the compilation.")
    if int(result.primal.shape[-1]) != qp.num_variables:
        raise ValueError("QP result primal dimension does not match the compilation.")
    specification = compilation.specification
    states, controls = compilation.decode(result.primal)
    finite_nodes = jnp.all(jnp.isfinite(states), axis=-1)
    trajectory_valid = result.valid[..., None] & finite_nodes
    control_status = jnp.where(
        result.valid,
        CONTROL_SUCCESS,
        jnp.where(
            result.status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE),
            CONTROL_INFEASIBLE,
            CONTROL_DYNAMICS_FAILED,
        ),
    ).astype(jnp.int32)
    policy_id = f"{specification.problem_id}:qp-policy"
    policy = PiecewiseConstantControlParameterization(
        specification.time_grid,
        (specification.control_size,),
        parameterization_id=policy_id,
    )
    trajectory = ControlTrajectory(
        time_grid=specification.time_grid,
        states=states,
        controls=controls,
        valid=trajectory_valid,
        status=control_status,
        backend_status=result.status,
        case_shape=specification.case_shape,
        state_shape=(specification.state_size,),
        control_shape=(specification.control_size,),
        problem_id=specification.problem_id,
        dynamics_id=specification.dynamics_id,
        control_id=policy_id,
        backend_id=result.backend,
        method_id=f"control:qp:{result.method}",
        discretization_id="control:discrete:exact-affine",
        approximation_id=policy.approximation_id,
    )
    identifier = (
        f"{specification.problem_id}:qp-solution"
        if solution_id is None
        else _identifier(solution_id, "solution_id")
    )
    return LinearControlQPSolution(
        compilation=compilation,
        qp_result=result,
        trajectory=trajectory,
        policy=policy,
        parameters=controls,
        objective=result.objective + compilation.objective_constant,
        valid=result.valid,
        status=result.status,
        solution_id=identifier,
        method_id=f"control:qp:{result.method}",
    )


def solve_prepared_linear_quadratic_control(
    prepared: PreparedLinearControlQP,
    /,
    *,
    warm_start: ConvexWarmStart | None = None,
) -> LinearControlQPSolution:
    """Execute and decode one prepared linear-control QP."""

    if not isinstance(prepared, PreparedLinearControlQP):
        raise TypeError("prepared must be a PreparedLinearControlQP.")
    execution = solve_convex_program(
        prepared.prepared,
        warm_start=warm_start,
    )
    return decode_linear_control_solution(prepared.compilation, execution.result)


def solve_linear_quadratic_control(
    specification: LinearQuadraticControlProblem,
    /,
    *,
    cost_tolerance: float = 1e-10,
    policy: ConvexSolvePolicy | None = None,
    compilation_policy: LinearControlCompilationPolicy | None = None,
) -> LinearControlQPSolution:
    """Compile, solve, and losslessly decode a finite linear control QP."""

    selected = ConvexSolvePolicy() if policy is None else policy
    if not isinstance(selected, ConvexSolvePolicy):
        raise TypeError("policy must be a ConvexSolvePolicy or None.")
    prepared = prepare_linear_quadratic_control(
        specification,
        policy=selected,
        cost_tolerance=cost_tolerance,
        compilation_policy=compilation_policy,
    )
    return solve_prepared_linear_quadratic_control(prepared)


__all__ = [
    "ControlQPRepresentation",
    "LinearControlCompilationPolicy",
    "LinearControlConstraintLayout",
    "LinearControlDecisionLayout",
    "LinearControlQPCompilation",
    "LinearControlBoundLayout",
    "LinearControlQPSolution",
    "LinearQuadraticControlProblem",
    "PreparedLinearControlQP",
    "prepare_linear_quadratic_control",
    "refresh_linear_quadratic_control",
    "solve_prepared_linear_quadratic_control",
    "LinearQuadraticControlSpecification",
    "compile_linear_quadratic_control",
    "decode_linear_control_solution",
    "solve_linear_quadratic_control",
]
