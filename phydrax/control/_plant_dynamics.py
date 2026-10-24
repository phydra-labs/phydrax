"""Control rollouts whose scan carry is the complete transactional plant state."""

from __future__ import annotations

from math import prod
from typing import Any, assert_never, final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..dynamics import TimeGrid
from ..dynamics._plant import (
    _select_runtime_state,
    _validate_runtime_state,
    AbstractDiscretePlant,
    PlantParameters,
    PlantRuntimeState,
    PlantStepContext,
)
from ..dynamics._plant_codec import (
    ControlVectorCodec,
    EncodedControl,
    PlantStateVectorCodec,
)
from ..dynamics._system import DiscreteTransitionEvidence
from ..typing import parse
from ._parameterization import (
    _validate_parameterization_grid,
    AbstractControlParameterization,
)
from ._plant_history import PlantControlHistory, PlantRuntimeHistory
from ._trajectory import CONTROL_DYNAMICS_FAILED, CONTROL_SUCCESS, ControlTrajectory


type PlantHistoryRetention = Literal["endpoint", "full"]
type _Carry = tuple[PlantRuntimeState, PlantRuntimeState, Array, Array, Array]
type _Output = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    PlantRuntimeState | None,
    PlantRuntimeState | None,
]


def _case_where(
    mask: Array, value: Array, replacement: Array, event_shape: tuple[int, ...]
) -> Array:
    return jnp.where(
        mask.reshape(mask.shape + (1,) * len(event_shape)), value, replacement
    )


@final
class PlantControlDynamics(StrictModule):
    """Result-aware control over a complete plant, with explicit history storage.

    Codecs define only the continuous view and command representation. Discrete
    modes, clocks, typed keys and every payload leaf remain in the scan carry.
    """

    plant: AbstractDiscretePlant
    state_codec: PlantStateVectorCodec
    control_codec: ControlVectorCodec
    parameters: PlantParameters
    retention: PlantHistoryRetention = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        plant: AbstractDiscretePlant,
        state_codec: PlantStateVectorCodec,
        control_codec: ControlVectorCodec,
        /,
        *,
        parameters: PlantParameters,
        retention: PlantHistoryRetention = "endpoint",
        maximum_retained_bytes: int = 64 * 1024 * 1024,
        method_id: str = "complete-plant-discrete-control",
    ) -> None:
        if not isinstance(plant, AbstractDiscretePlant):
            raise TypeError("PlantControlDynamics requires an AbstractDiscretePlant.")
        if not isinstance(state_codec, PlantStateVectorCodec) or not isinstance(
            control_codec, ControlVectorCodec
        ):
            raise TypeError("Plant control requires complete state and command codecs.")
        if plant.control_schema is None:
            raise ValueError(
                "PlantControlDynamics requires an explicitly controlled plant."
            )
        if control_codec.schema.case_ndim != 0:
            raise ValueError("Plant control requires an unbatched command codec.")
        if not plant.state_schema.same_intrinsic_contract(state_codec.schema):
            raise ValueError(
                "State codec does not describe the plant's complete intrinsic payload."
            )
        if not plant.control_schema.same_intrinsic_contract(control_codec.schema):
            raise ValueError(
                "Control codec does not describe the complete plant command."
            )
        for codec in (state_codec, control_codec):
            if (
                codec.semantic_id != plant.semantic_provenance.semantic_id
                or codec.numeric_revision_id != plant.numeric_revision.revision_id
                or codec.executable_signature_id != plant.execution_signature.signature_id
            ):
                raise ValueError("Plant control codec identities do not match the plant.")
        if isinstance(maximum_retained_bytes, bool) or not isinstance(
            maximum_retained_bytes, int
        ):
            raise TypeError("maximum_retained_bytes must be an integer.")
        if (
            maximum_retained_bytes < 0
            or not isinstance(method_id, str)
            or not method_id.strip()
        ):
            raise ValueError("History capacity and method identity are invalid.")
        retention_ = parse(retention, PlantHistoryRetention, "retention")
        if not isinstance(parameters, PlantParameters):
            raise TypeError("parameters must be complete PlantParameters.")
        self.plant = plant
        self.state_codec = state_codec
        self.control_codec = control_codec
        self.parameters = parameters
        self.retention = retention_
        self.maximum_retained_bytes = maximum_retained_bytes
        self.method_id = method_id

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.state_codec.layout.shape

    @property
    def control_shape(self) -> tuple[int, ...]:
        return (self.control_codec.size,)

    @property
    def state_dtype(self) -> np.dtype:
        return self.state_codec.point_dtype

    @property
    def dynamics_id(self) -> str:
        return self.plant.semantic_provenance.semantic_id

    def validate_initial_state(
        self, state: PlantRuntimeState, /
    ) -> tuple[PlantRuntimeState, tuple[int, ...]]:
        cases, checked = _validate_runtime_state(self.plant, state)
        if any(size <= 0 for size in cases):
            raise ValueError("Control rollout case dimensions must be positive.")
        if not jax.dtypes.issubdtype(checked.key.dtype, jax.dtypes.prng_key):
            raise TypeError("Plant control requires typed PRNG keys.")
        return checked, cases

    def state_vector(self, state: PlantRuntimeState, /) -> Array:
        cases = state.time.shape
        if not cases:
            return self.state_codec.encode_point(state.payload).vector
        count = prod(cases)
        flat = jax.tree.map(
            lambda value: value.reshape((count,) + value.shape[len(cases) :]),
            state.payload,
        )

        def encode(payload: Any) -> Array:
            return self.state_codec.encode_point(payload).vector

        return jax.vmap(encode)(flat).reshape(cases + self.state_shape)

    def decode_controls(self, values: Array, cases: tuple[int, ...], /) -> Any:
        codec = self.control_codec

        def decode(vector: Array) -> Any:
            return codec.decode_command(
                EncodedControl(
                    vector,
                    semantic_id=codec.semantic_id,
                    numeric_revision_id=codec.numeric_revision_id,
                    schema_id=codec.schema_id,
                    executable_signature_id=codec.executable_signature_id,
                    codec_id=codec.codec_id,
                )
            )

        if not cases:
            return decode(values)
        count = prod(cases)
        flat = values.reshape((count,) + self.control_shape)
        commands = jax.vmap(decode)(flat)
        return jax.tree.map(
            lambda value: value.reshape(cases + value.shape[1:]), commands
        )

    def retained_storage_bytes(self, state: PlantRuntimeState, steps: int, /) -> int:
        cases = state.time.shape
        count = prod(cases)
        key_bytes = (
            jax.random.key_data(state.key).dtype.itemsize
            * jax.random.key_data(state.key).shape[-1]
        )
        state_bytes = (
            self.plant.state_schema.intrinsic_storage_bytes
            + state.time.dtype.itemsize
            + state.step_index.dtype.itemsize
            + key_bytes
        )
        views = (
            (3 * steps + 1) * self.state_codec.layout.size * self.state_dtype.itemsize
            + steps * self.control_codec.size * self.control_codec.command_dtype.itemsize
            + (steps + 1)
            + steps * 10
            + 8
        )
        match self.retention:
            case "endpoint":
                state_copies = 3
            case "full":
                state_copies = 3 + 2 * steps + 1
            case _:
                assert_never(self.retention)
        return count * (state_copies * state_bytes + views)

    def rollout(
        self,
        time_grid: TimeGrid,
        initial_state: PlantRuntimeState,
        parameterization: AbstractControlParameterization,
        coefficients: ArrayLike,
        /,
        *,
        args: PlantParameters | None = None,
        problem_id: str,
    ) -> ControlTrajectory:
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("Plant control rollout requires a TimeGrid.")
        if parameterization.control_shape != self.control_shape:
            raise ValueError(
                "Parameterization control shape does not match the complete command codec."
            )
        source, _ = self.validate_initial_state(initial_state)
        if source.time.dtype != time_grid.times.dtype:
            raise TypeError(
                "Plant clock and control grid must use the same explicit precision."
            )
        source = eqx.tree_at(
            lambda value: value.time,
            source,
            eqx.error_if(
                source.time,
                jnp.any(source.time != time_grid.times[0]),
                "Plant initial clock does not match the rollout grid.",
            ),
        )
        parameterization = _validate_parameterization_grid(parameterization, time_grid)
        retained = self.retained_storage_bytes(source, time_grid.num_steps)
        if retained > self.maximum_retained_bytes:
            raise ValueError(
                f"Plant control retention requires {retained} bytes, exceeding {self.maximum_retained_bytes}."
            )
        parameters = self.parameters if args is None else args
        if not isinstance(parameters, PlantParameters):
            raise TypeError("Plant control args must be PlantParameters or None.")
        return _rollout_plant(
            self,
            time_grid,
            source,
            parameterization,
            jnp.asarray(coefficients),
            parameters,
            problem_id,
            retained,
        )


@eqx.filter_jit
def _rollout_plant(
    dynamics: PlantControlDynamics,
    grid: TimeGrid,
    initial: PlantRuntimeState,
    parameterization: AbstractControlParameterization,
    coefficients: Array,
    parameters: PlantParameters,
    problem_id: str,
    retained: int,
) -> ControlTrajectory:
    cases = initial.time.shape
    point_shape = dynamics.state_shape
    command_shape = dynamics.control_shape
    initial_vector = dynamics.state_vector(initial)
    initial_valid = dynamics.plant.state_schema.finite_mask(initial.payload)
    keep_history = dynamics.retention == "full"

    def step(carry: _Carry, index: Array) -> tuple[_Carry, _Output]:
        current, last_candidate, valid, backend_status, failed = carry
        vector = dynamics.state_vector(current)
        command_vector = parameterization.evaluate(
            coefficients, grid.times[index], case_shape=cases, state=vector
        )
        if command_vector.shape != cases + command_shape:
            raise ValueError(
                "Control parameterization returned an incompatible command shape."
            )
        command_finite = jnp.all(jnp.isfinite(command_vector), axis=-1)
        active = valid & command_finite
        # PlantCore sanitizes numerical input before proposing and preserves its
        # invalid-input status. A failed case keeps its actual clock and key.
        command = dynamics.decode_controls(command_vector, cases)
        duration = grid.times[index + 1] - grid.times[index]
        target_time = jnp.where(valid, grid.times[index + 1], current.time + duration)
        transition = dynamics.plant.step(
            PlantStepContext(current.time, target_time, current.step_index),
            current,
            command,
            parameters,
        )
        accepted = _select_runtime_state(
            dynamics.plant.state_schema, valid, transition.accepted_state, current, cases
        )
        attempted = active & transition.attempted
        candidate = _select_runtime_state(
            dynamics.plant.state_schema,
            attempted,
            transition.candidate_state,
            current,
            cases,
        )
        last_candidate = _select_runtime_state(
            dynamics.plant.state_schema, attempted, candidate, last_candidate, cases
        )
        next_valid = active & transition.successful
        record = valid & ~failed
        next_status = jnp.where(record, transition.status, backend_status).astype(
            jnp.int32
        )
        next_failed = failed | (valid & ~next_valid)
        accepted_vector = dynamics.state_vector(accepted)
        candidate_vector = dynamics.state_vector(candidate)
        unavailable = jnp.full_like(vector, jnp.nan)
        visible = _case_where(active, accepted_vector, unavailable, point_shape)
        candidate_view = _case_where(
            attempted, candidate_vector, unavailable, point_shape
        )
        accepted_view = _case_where(attempted, accepted_vector, unavailable, point_shape)
        visible_control = _case_where(
            valid, command_vector, jnp.full_like(command_vector, jnp.nan), command_shape
        )
        return (accepted, last_candidate, next_valid, next_status, next_failed), (
            visible,
            visible_control,
            next_valid,
            candidate_view,
            accepted_view,
            attempted,
            attempted & transition.successful,
            jnp.where(valid, transition.status, 0).astype(jnp.int32),
            accepted if keep_history else None,
            candidate if keep_history else None,
        )

    (final_state, final_candidate, _, backend_status, failed), outputs = jax.lax.scan(
        step,
        (
            initial,
            initial,
            initial_valid,
            jnp.zeros(cases, dtype=jnp.int32),
            jnp.zeros(cases, dtype=jnp.bool_),
        ),
        jnp.arange(grid.num_steps, dtype=jnp.int32),
    )
    (
        next_views,
        controls,
        valid_steps,
        candidates,
        accepted,
        attempted,
        successful,
        statuses,
        complete_accepted,
        complete_candidates,
    ) = outputs
    axis = len(cases)
    states = jnp.concatenate(
        (jnp.expand_dims(initial_vector, axis), jnp.moveaxis(next_views, 0, axis)),
        axis=axis,
    )
    valid_nodes = jnp.concatenate(
        (initial_valid[..., None], jnp.moveaxis(valid_steps, 0, -1)), axis=-1
    )
    valid_all = jnp.all(valid_nodes, axis=-1)
    status = jnp.where(valid_all, CONTROL_SUCCESS, CONTROL_DYNAMICS_FAILED).astype(
        jnp.int32
    )
    accepted_history = None
    candidate_history = None
    if keep_history:
        if complete_accepted is None or complete_candidates is None:
            raise RuntimeError("Complete plant history was not retained by the scan.")
        nodes = jax.tree.map(
            lambda first, rest: jnp.concatenate((first[None], rest), axis=0),
            initial,
            complete_accepted,
        )
        accepted_history = PlantRuntimeHistory(nodes)
        candidate_history = PlantRuntimeHistory(complete_candidates)
    history = PlantControlHistory(
        initial,
        final_state,
        final_candidate,
        accepted_history,
        candidate_history,
        retained,
    )
    return ControlTrajectory(
        time_grid=grid,
        states=states,
        controls=jnp.moveaxis(controls, 0, axis),
        valid=valid_nodes,
        status=status,
        backend_status=jnp.where(failed | valid_all, backend_status, status),
        transition_evidence=DiscreteTransitionEvidence(
            jnp.moveaxis(candidates, 0, axis),
            jnp.moveaxis(accepted, 0, axis),
            jnp.moveaxis(attempted, 0, -1),
            jnp.moveaxis(successful, 0, -1),
            jnp.moveaxis(statuses, 0, -1),
        ),
        plant_history=history,
        case_shape=cases,
        state_shape=point_shape,
        control_shape=dynamics.control_shape,
        problem_id=problem_id,
        dynamics_id=dynamics.dynamics_id,
        control_id=parameterization.parameterization_id,
        backend_id="backend:jax:complete-plant-scan",
        method_id=dynamics.method_id,
        discretization_id=grid.time_id,
        approximation_id=parameterization.approximation_id,
    )


__all__ = ["PlantControlDynamics", "PlantHistoryRetention"]
