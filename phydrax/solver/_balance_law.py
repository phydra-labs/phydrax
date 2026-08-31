#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._numerics._checkpointed_scan import checkpointed_scan
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._temporal import RealizedTemporalMesh, TemporalMesh
from ..stochastic._realization import (
    CompositeStochasticRealization,
    is_stochastic_realization,
    StochasticRealization,
)
from ._finite_volume_content import FiniteVolumeConservativeContentState
from ._finite_volume_rollout import FiniteVolumeReplayPolicy
from ._finite_volume_runtime import (
    FiniteVolumeRuntimeState,
    FiniteVolumeScheduledAdvanceResult,
    PreparedFiniteVolumeRuntime,
)


class BalanceLawProcessState(StrictModule):
    """Named fixed-structure auxiliary arrays for one source process."""

    values: tuple[Array, ...]
    process_id: str = eqx.field(static=True)
    field_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        process_id: str,
        field_names: tuple[str, ...],
        values: tuple[ArrayLike, ...],
        /,
    ):
        identifier = str(process_id)
        names = tuple(str(name) for name in field_names)
        arrays = tuple(jnp.asarray(value) for value in values)
        if not identifier or len(names) != len(arrays):
            raise ValueError("Process state requires one name per array.")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Process state field names must be unique and non-empty.")
        self.values = arrays
        self.process_id = identifier
        self.field_names = names

    @classmethod
    def empty(cls, process_id: str, /) -> BalanceLawProcessState:
        return cls(process_id, (), ())

    def field(self, name: str, /) -> Array:
        if name not in self.field_names:
            raise KeyError(f"Unknown process-state field {name!r}.")
        return self.values[self.field_names.index(name)]


class BalanceLawProcessAdvance(StrictModule):
    cell_average: Array
    process_state: BalanceLawProcessState
    successful: Array
    source_change: Array
    diagnostics: Any


class AbstractPreparedBalanceLawProcess(StrictModule, NonTrainableState):
    """Prepared deterministic or replayable stochastic balance-law process."""

    process_id: str = eqx.field(static=True)
    requires_realization: bool = eqx.field(static=True)
    realization_name: str | None = eqx.field(static=True)
    differentiability: str = eqx.field(static=True)

    @abc.abstractmethod
    def initialize(
        self, transport_state: FiniteVolumeRuntimeState, args: Any = None, /
    ) -> BalanceLawProcessState:
        raise NotImplementedError

    @abc.abstractmethod
    def step_limit(
        self,
        time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        args: Any = None,
        /,
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def advance(
        self,
        start_time: Array,
        end_time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        realization: Any = None,
        args: Any = None,
        /,
    ) -> BalanceLawProcessAdvance:
        raise NotImplementedError


class AbstractBalanceLawProcessPlan(StrictModule, NonTrainableState):
    process_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def prepare(
        self, runtime: PreparedFiniteVolumeRuntime, /
    ) -> AbstractPreparedBalanceLawProcess:
        raise NotImplementedError


class BalanceLawRuntimeState(StrictModule):
    transport_state: FiniteVolumeRuntimeState
    process_states: tuple[BalanceLawProcessState, ...]
    process_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        transport_state: FiniteVolumeRuntimeState,
        process_states: tuple[BalanceLawProcessState, ...],
        /,
    ):
        if not isinstance(transport_state, FiniteVolumeRuntimeState):
            raise TypeError("transport_state must be FiniteVolumeRuntimeState.")
        states = tuple(process_states)
        if any(not isinstance(state, BalanceLawProcessState) for state in states):
            raise TypeError("process_states must contain BalanceLawProcessState values.")
        identifiers = tuple(state.process_id for state in states)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Balance-law process IDs must be unique.")
        self.transport_state = transport_state
        self.process_states = states
        self.process_ids = identifiers

    @property
    def time(self) -> Array:
        return self.transport_state.time


class BalanceLawAdvanceResult(StrictModule):
    runtime_state: BalanceLawRuntimeState
    transport: FiniteVolumeScheduledAdvanceResult
    accepted: Array
    status: Array
    process_step_limits: Array
    stability_margin: Array
    process_diagnostics: tuple[Any, ...]


class BalanceLawRolloutResult(StrictModule):
    final_state: BalanceLawRuntimeState
    retained_states: Array
    retained_times: Array
    accepted: Array
    statuses: Array
    stability_margins: Array
    temporal_mesh_id: str = eqx.field(static=True)


class PreparedBalanceLawRuntime(StrictModule, NonTrainableState):
    """Symmetric transactional source/transport composition."""

    transport: PreparedFiniteVolumeRuntime
    processes: tuple[AbstractPreparedBalanceLawProcess, ...]
    process_ids: tuple[str, ...] = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport: PreparedFiniteVolumeRuntime,
        processes: tuple[AbstractPreparedBalanceLawProcess, ...],
        /,
    ):
        if not isinstance(transport, PreparedFiniteVolumeRuntime):
            raise TypeError("transport must be PreparedFiniteVolumeRuntime.")
        prepared = tuple(processes)
        if not prepared or any(
            not isinstance(process, AbstractPreparedBalanceLawProcess)
            for process in prepared
        ):
            raise TypeError(
                "processes must contain at least one prepared balance-law process."
            )
        identifiers = tuple(process.process_id for process in prepared)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Prepared balance-law process IDs must be unique.")
        realization_names = tuple(
            process.realization_name
            for process in prepared
            if process.requires_realization
        )
        if any(name is None or not name for name in realization_names) or len(
            set(realization_names)
        ) != len(realization_names):
            raise ValueError(
                "Stochastic processes require unique non-empty realization names."
            )
        self.transport = transport
        self.processes = prepared
        self.process_ids = identifiers
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-balance-law-runtime",
                "transport": transport.runtime_id,
                "processes": list(identifiers),
                "splitting": "symmetric-declared-order",
            }
        )

    def initialize_state(
        self, transport_state: FiniteVolumeRuntimeState, args: Any = None, /
    ) -> BalanceLawRuntimeState:
        states = tuple(
            process.initialize(transport_state, args) for process in self.processes
        )
        return BalanceLawRuntimeState(transport_state, states)

    @staticmethod
    def _realization_component(
        process: AbstractPreparedBalanceLawProcess,
        realization: StochasticRealization | None,
        stochastic_count: int,
        /,
    ) -> Any:
        if not process.requires_realization:
            return None
        if realization is None:
            raise ValueError(
                f"Process {process.process_id!r} requires a stochastic realization."
            )
        if isinstance(realization, CompositeStochasticRealization):
            return realization.component(str(process.realization_name))
        if stochastic_count == 1 and is_stochastic_realization(realization):
            return realization
        raise TypeError(
            "Multiple stochastic processes require CompositeStochasticRealization."
        )

    @staticmethod
    def _with_cell_average(
        transport_state: FiniteVolumeRuntimeState,
        cell_average: Array,
        /,
    ) -> FiniteVolumeRuntimeState:
        old = transport_state.content_state
        average = jnp.asarray(cell_average).reshape(old.conservative_content.shape)
        content = FiniteVolumeConservativeContentState.from_cell_average(
            average,
            old.effective_cell_volumes,
            old.active_cell_mask,
            old.time,
            topology_epoch_id=old.topology_epoch_id,
            geometry_family_id=old.geometry_family_id,
            geometry_layout_id=old.geometry_layout_id,
            geometry_version=old.geometry_version,
            evidence_policy_id=old.evidence_policy_id,
            evidence_version=old.evidence_version,
            precision=old.precision,
        )
        return FiniteVolumeRuntimeState(
            content,
            transport_state.topology_journal,
            transport_state.step_size,
            accepted_step=transport_state.accepted_step,
            last_status=transport_state.last_status,
            controller_state=transport_state.controller_state,
            integrator_state=transport_state.integrator_state,
            output_cursor=transport_state.output_cursor,
            sliding_coupling=transport_state.sliding_coupling,
            sliding_shift=transport_state.sliding_shift,
            sliding_event_id=transport_state.sliding_event_id,
        )

    def advance_prescribed(
        self,
        runtime_state: BalanceLawRuntimeState,
        start_time: ArrayLike,
        end_time: ArrayLike,
        args: Any = None,
        realization: StochasticRealization | None = None,
        /,
    ) -> BalanceLawAdvanceResult:
        if runtime_state.process_ids != self.process_ids:
            raise ValueError("Balance-law runtime state process order changed.")
        start = jnp.asarray(start_time)
        end = jnp.asarray(end_time, dtype=start.dtype)
        step_size = end - start
        step_size = eqx.error_if(
            step_size,
            ~jnp.isfinite(start)
            | ~jnp.isfinite(end)
            | ~jnp.isfinite(step_size)
            | (step_size <= 0.0),
            "Balance-law interval must be finite and increasing.",
        )
        tolerance = 32.0 * jnp.finfo(start.dtype).eps * jnp.maximum(jnp.abs(start), 1.0)
        current_time = eqx.error_if(
            runtime_state.time,
            jnp.abs(runtime_state.time - start) > tolerance,
            "Balance-law state time must equal interval start.",
        )
        del current_time
        stochastic_count = sum(process.requires_realization for process in self.processes)
        original = runtime_state
        average = runtime_state.transport_state.cell_average()
        states = list(runtime_state.process_states)
        limits = tuple(
            jnp.asarray(process.step_limit(start, average, state, args)).reshape(())
            for process, state in zip(self.processes, states, strict=True)
        )
        limits_array = jnp.stack(limits) if limits else jnp.zeros((0,), dtype=start.dtype)
        limit_valid = jnp.all(
            jnp.isfinite(limits_array) | jnp.isinf(limits_array)
        ) & jnp.all(limits_array > 0.0)
        within_limits = limit_valid & jnp.all(step_size <= limits_array)
        midpoint = start + 0.5 * step_size
        successful = within_limits
        first_diagnostics = []
        for index, process in enumerate(self.processes):
            component = self._realization_component(
                process, realization, stochastic_count
            )
            result = process.advance(
                start,
                midpoint,
                average,
                states[index],
                component,
                args,
            )
            average = jnp.where(result.successful, result.cell_average, average)
            states[index] = result.process_state
            successful = successful & result.successful
            first_diagnostics.append(result.diagnostics)

        source_updated = self._with_cell_average(runtime_state.transport_state, average)
        transport = self.transport.advance_prescribed(source_updated, step_size, args)
        successful = successful & transport.accepted
        average = transport.runtime_state.cell_average()
        second_diagnostics = []
        for reverse_index, process in enumerate(reversed(self.processes)):
            index = len(self.processes) - 1 - reverse_index
            component = self._realization_component(
                process, realization, stochastic_count
            )
            result = process.advance(
                midpoint,
                end,
                average,
                states[index],
                component,
                args,
            )
            average = jnp.where(result.successful, result.cell_average, average)
            states[index] = result.process_state
            successful = successful & result.successful
            second_diagnostics.append(result.diagnostics)

        candidate_transport = self._with_cell_average(transport.runtime_state, average)
        candidate = BalanceLawRuntimeState(candidate_transport, tuple(states))
        committed = jax.lax.cond(
            successful,
            lambda _: candidate,
            lambda _: original,
            operand=None,
        )
        minimum_process_limit = jnp.min(limits_array, initial=jnp.inf)
        combined_limit = jnp.minimum(transport.stable_step_size, minimum_process_limit)
        margin = combined_limit / step_size - 1.0
        status = jnp.where(
            successful,
            0,
            jnp.where(within_limits, 2, 1),
        ).astype(jnp.int32)
        diagnostics = tuple(first_diagnostics) + tuple(reversed(second_diagnostics))
        return BalanceLawAdvanceResult(
            runtime_state=committed,
            transport=transport,
            accepted=successful,
            status=status,
            process_step_limits=limits_array,
            stability_margin=margin,
            process_diagnostics=diagnostics,
        )


class ScheduledBalanceLawRolloutPlan(StrictModule, NonTrainableState):
    runtime: PreparedBalanceLawRuntime
    temporal_mesh: TemporalMesh
    replay: FiniteVolumeReplayPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedBalanceLawRuntime,
        temporal_mesh: TemporalMesh,
        /,
        *,
        replay: FiniteVolumeReplayPolicy | None = None,
    ):
        if not isinstance(runtime, PreparedBalanceLawRuntime):
            raise TypeError("runtime must be PreparedBalanceLawRuntime.")
        if not isinstance(temporal_mesh, TemporalMesh):
            raise TypeError("temporal_mesh must be TemporalMesh.")
        if temporal_mesh.role != "internal" or not bool(
            np.all(np.asarray(temporal_mesh.active_intervals))
        ):
            raise ValueError(
                "Balance-law rollout requires an all-active internal temporal mesh."
            )
        replay_ = FiniteVolumeReplayPolicy() if replay is None else replay
        if not isinstance(replay_, FiniteVolumeReplayPolicy):
            raise TypeError("replay must be FiniteVolumeReplayPolicy or None.")
        self.runtime = runtime
        self.temporal_mesh = temporal_mesh
        self.replay = replay_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scheduled-balance-law-rollout",
                "runtime": runtime.runtime_id,
                "temporal_mesh": temporal_mesh.mesh_id,
                "replay": replay_.policy_id,
            }
        )

    @classmethod
    def from_realized_mesh(
        cls,
        runtime: PreparedBalanceLawRuntime,
        realized_mesh: RealizedTemporalMesh,
        /,
        *,
        replay: FiniteVolumeReplayPolicy | None = None,
    ) -> ScheduledBalanceLawRolloutPlan:
        """Construct an exact scheduled replay from one adaptive realization."""
        if not isinstance(realized_mesh, RealizedTemporalMesh):
            raise TypeError("realized_mesh must be RealizedTemporalMesh.")
        count = int(np.asarray(jax.device_get(realized_mesh.count)))
        if count <= 0:
            raise ValueError(
                "A scheduled replay requires at least one accepted interval."
            )
        initial = np.asarray(jax.device_get(realized_mesh.initial_time)).reshape(())
        accepted = np.asarray(jax.device_get(realized_mesh.accepted_times[:count]))
        temporal_mesh = TemporalMesh(
            np.concatenate((initial[None], accepted)),
            role="internal",
            realized=True,
            source_plan_id=realized_mesh.source_plan_id,
        )
        return cls(runtime, temporal_mesh, replay=replay)

    def rollout(
        self,
        initial_state: BalanceLawRuntimeState,
        args: Any = None,
        realization: StochasticRealization | None = None,
        /,
    ) -> BalanceLawRolloutResult:
        if not isinstance(initial_state, BalanceLawRuntimeState):
            raise TypeError("initial_state must be BalanceLawRuntimeState.")

        def step(carry, interval):
            state, active = carry
            start, end = interval

            def execute(_):
                result = self.runtime.advance_prescribed(
                    state, start, end, args, realization
                )
                return (
                    (result.runtime_state, active & result.accepted),
                    (
                        result.runtime_state.transport_state.content_state.conservative_content,
                        result.runtime_state.time,
                        result.accepted,
                        result.status,
                        result.stability_margin,
                    ),
                )

            def skip(_):
                return (
                    (state, active),
                    (
                        state.transport_state.content_state.conservative_content,
                        state.time,
                        jnp.asarray(False),
                        jnp.asarray(2, dtype=jnp.int32),
                        jnp.asarray(jnp.nan, dtype=start.dtype),
                    ),
                )

            return jax.lax.cond(active, execute, skip, operand=None)

        intervals = (self.temporal_mesh.nodes[:-1], self.temporal_mesh.nodes[1:])
        (final, _), outputs = checkpointed_scan(
            step,
            (initial_state, jnp.asarray(True)),
            intervals,
            length=self.temporal_mesh.interval_count,
            mode=self.replay.mode,
            block_size=self.replay.block_size,
        )
        states, times, accepted, statuses, margins = outputs
        return BalanceLawRolloutResult(
            final_state=final,
            retained_states=states,
            retained_times=times,
            accepted=accepted,
            statuses=statuses,
            stability_margins=margins,
            temporal_mesh_id=self.temporal_mesh.mesh_id,
        )


__all__ = [
    "AbstractBalanceLawProcessPlan",
    "AbstractPreparedBalanceLawProcess",
    "BalanceLawAdvanceResult",
    "BalanceLawProcessAdvance",
    "BalanceLawProcessState",
    "BalanceLawRolloutResult",
    "BalanceLawRuntimeState",
    "PreparedBalanceLawRuntime",
    "ScheduledBalanceLawRolloutPlan",
]
