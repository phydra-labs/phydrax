#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any, cast, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import PreparedFiniteVolumeDynamics
from ._constrained_mhd import (
    ConstrainedMHDRunStatus,
    ConstrainedMHDSSPRK3Plan,
    ConstrainedMHDState,
)
from ._finite_volume_content import FiniteVolumeConservativeContentState
from ._finite_volume_runtime import (
    FiniteVolumeRunStatus,
    FiniteVolumeRuntimeState,
    PreparedFiniteVolumeRuntime,
)


BalanceLawTransportState: TypeAlias = FiniteVolumeRuntimeState | ConstrainedMHDState


class BalanceLawSourceView(StrictModule):
    """Cell-average state exposed to transactional source processes."""

    cell_average: Array
    cell_volumes: Array
    active_cell_mask: Array
    time: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_average: Array,
        cell_volumes: Array,
        active_cell_mask: Array,
        time: Array,
        /,
        *,
        component_names: tuple[str, ...],
        transport_id: str,
    ):
        average = jnp.asarray(cell_average)
        volumes = jnp.asarray(cell_volumes)
        active = jnp.asarray(active_cell_mask, dtype=bool)
        time_ = jnp.asarray(time).reshape(())
        names = tuple(str(name) for name in component_names)
        identifier = str(transport_id)
        if (
            average.ndim != 2
            or volumes.shape != (average.shape[0],)
            or active.shape != volumes.shape
            or len(names) != average.shape[-1]
            or any(not name for name in names)
            or len(set(names)) != len(names)
            or not identifier
        ):
            raise ValueError("Balance-law source view structure is invalid.")
        self.cell_average = average
        self.cell_volumes = volumes
        self.active_cell_mask = active
        self.time = time_
        self.component_names = names
        self.transport_id = identifier


class BalanceLawTransportAdvance(StrictModule):
    state: BalanceLawTransportState
    accepted: Array
    status: Array
    stable_step_size: Array
    stability_margin: Array
    diagnostics: Any


class AbstractPreparedBalanceLawTransport(StrictModule, NonTrainableState):
    """Minimal transport contract consumed by the balance-law runtime."""

    dynamics: Any
    component_names: tuple[str, ...] = eqx.field(static=True)
    mutable_component_names: tuple[str, ...] = eqx.field(static=True)
    transport_kind: str = eqx.field(static=True)
    checkpoint_supported: bool = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    @property
    def precision(self):
        return self.dynamics.precision

    @abc.abstractmethod
    def validate_state(self, state: BalanceLawTransportState, /) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def source_view(self, state: BalanceLawTransportState, /) -> BalanceLawSourceView:
        raise NotImplementedError

    @abc.abstractmethod
    def with_source_view(
        self,
        state: BalanceLawTransportState,
        cell_average: Array,
        /,
    ) -> BalanceLawTransportState:
        raise NotImplementedError

    @abc.abstractmethod
    def advance_prescribed(
        self,
        state: BalanceLawTransportState,
        start_time: Array,
        end_time: Array,
        args: Any = None,
        /,
    ) -> BalanceLawTransportAdvance:
        raise NotImplementedError

    @abc.abstractmethod
    def auxiliary_state(self, state: BalanceLawTransportState, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def checkpoint_arrays(
        self, state: BalanceLawTransportState, /
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abc.abstractmethod
    def checkpoint_array_names(self, /) -> frozenset[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def restore_checkpoint(
        self, arrays: Mapping[str, np.ndarray], /
    ) -> BalanceLawTransportState:
        raise NotImplementedError


class PreparedFiniteVolumeBalanceLawTransport(AbstractPreparedBalanceLawTransport):
    runtime: PreparedFiniteVolumeRuntime

    def __init__(self, runtime: PreparedFiniteVolumeRuntime, /):
        if not isinstance(runtime, PreparedFiniteVolumeRuntime):
            raise TypeError("runtime must be PreparedFiniteVolumeRuntime.")
        names = tuple(runtime.dynamics.system.component_names)
        self.runtime = runtime
        self.dynamics = runtime.dynamics
        self.component_names = names
        self.mutable_component_names = names
        self.transport_kind = "finite_volume"
        self.checkpoint_supported = isinstance(
            runtime.dynamics, PreparedFiniteVolumeDynamics
        )
        self.transport_id = canonical_fingerprint(
            {
                "kind": "finite-volume-balance-law-transport",
                "runtime": runtime.runtime_id,
            }
        )

    def validate_state(self, state: BalanceLawTransportState, /) -> None:
        if not isinstance(state, FiniteVolumeRuntimeState):
            raise TypeError("Finite-volume balance transport requires its runtime state.")

    def source_view(self, state: BalanceLawTransportState, /) -> BalanceLawSourceView:
        self.validate_state(state)
        state = cast(FiniteVolumeRuntimeState, state)
        content = state.content_state
        return BalanceLawSourceView(
            content.cell_average(),
            content.effective_cell_volumes,
            content.active_cell_mask,
            state.time,
            component_names=self.component_names,
            transport_id=self.transport_id,
        )

    def with_source_view(
        self,
        state: BalanceLawTransportState,
        cell_average: Array,
        /,
    ) -> FiniteVolumeRuntimeState:
        self.validate_state(state)
        state = cast(FiniteVolumeRuntimeState, state)
        old = state.content_state
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
            state.topology_journal,
            state.step_size,
            accepted_step=state.accepted_step,
            last_status=state.last_status,
            controller_state=state.controller_state,
            integrator_state=state.integrator_state,
            output_cursor=state.output_cursor,
            sliding_coupling=state.sliding_coupling,
            sliding_shift=state.sliding_shift,
            sliding_event_id=state.sliding_event_id,
        )

    def advance_prescribed(
        self,
        state: BalanceLawTransportState,
        start_time: Array,
        end_time: Array,
        args: Any = None,
        /,
    ) -> BalanceLawTransportAdvance:
        self.validate_state(state)
        state = cast(FiniteVolumeRuntimeState, state)
        result = self.runtime.advance_prescribed(state, end_time - start_time, args)
        return BalanceLawTransportAdvance(
            state=result.runtime_state,
            accepted=result.accepted,
            status=result.runtime_state.last_status,
            stable_step_size=result.stable_step_size,
            stability_margin=result.stability_margin,
            diagnostics=result,
        )

    def auxiliary_state(self, state: BalanceLawTransportState, /) -> Array:
        self.validate_state(state)
        state = cast(FiniteVolumeRuntimeState, state)
        return jnp.zeros((0,), dtype=state.time.dtype)

    def checkpoint_arrays(
        self, state: BalanceLawTransportState, /
    ) -> dict[str, np.ndarray]:
        self.validate_state(state)
        state = cast(FiniteVolumeRuntimeState, state)
        return {
            "transport/cell_average": np.asarray(state.cell_average()),
            "transport/time": np.asarray(state.time),
            "transport/step_size": np.asarray(state.step_size),
            "transport/accepted_step": np.asarray(state.accepted_step, dtype=np.int32),
            "transport/last_status": np.asarray(state.last_status, dtype=np.int32),
            "transport/controller_state": np.asarray(state.controller_state),
            "transport/integrator_state": np.asarray(state.integrator_state),
            "transport/output_cursor": np.asarray(state.output_cursor, dtype=np.int32),
        }

    def checkpoint_array_names(self, /) -> frozenset[str]:
        return frozenset(
            {
                "transport/cell_average",
                "transport/time",
                "transport/step_size",
                "transport/accepted_step",
                "transport/last_status",
                "transport/controller_state",
                "transport/integrator_state",
                "transport/output_cursor",
            }
        )

    def restore_checkpoint(
        self, arrays: Mapping[str, np.ndarray], /
    ) -> FiniteVolumeRuntimeState:
        if not isinstance(self.dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError(
                "Balance-law finite-volume checkpoints require stationary structured dynamics."
            )
        status = int(np.asarray(arrays["transport/last_status"]))
        if status not in tuple(int(value) for value in FiniteVolumeRunStatus):
            raise ValueError("Checkpoint finite-volume status is invalid.")
        cell_average = jnp.asarray(arrays["transport/cell_average"]).reshape(
            self.dynamics.discretization.state_shape
        )
        return self.runtime.initialize_state(
            cell_average,
            jnp.asarray(arrays["transport/time"]),
            jnp.asarray(arrays["transport/step_size"]),
            accepted_step=jnp.asarray(arrays["transport/accepted_step"]),
            last_status=status,
            controller_state=jnp.asarray(arrays["transport/controller_state"]),
            integrator_state=jnp.asarray(arrays["transport/integrator_state"]),
            output_cursor=jnp.asarray(arrays["transport/output_cursor"]),
        )


class PreparedConstrainedMHDBalanceLawTransport(AbstractPreparedBalanceLawTransport):
    integrator: ConstrainedMHDSSPRK3Plan

    def __init__(self, integrator: ConstrainedMHDSSPRK3Plan, /):
        if not isinstance(integrator, ConstrainedMHDSSPRK3Plan):
            raise TypeError("integrator must be ConstrainedMHDSSPRK3Plan.")
        dynamics = integrator.spatial.dynamics
        names = tuple(dynamics.system.component_names)
        mutable = tuple(name for name in names if not name.startswith("magnetic_"))
        self.integrator = integrator
        self.dynamics = dynamics
        self.component_names = names
        self.mutable_component_names = mutable
        self.transport_kind = "constrained_mhd"
        self.checkpoint_supported = True
        self.transport_id = canonical_fingerprint(
            {
                "kind": "constrained-mhd-balance-law-transport",
                "integrator": integrator.plan_id,
            }
        )

    def validate_state(self, state: BalanceLawTransportState, /) -> None:
        if not isinstance(state, ConstrainedMHDState):
            raise TypeError("Constrained-MHD balance transport requires its state.")
        self.integrator.spatial.validate_reduced_state(state.cell_state)
        self.integrator.spatial.validate_magnetic_flux(state.magnetic_flux)

    def source_view(self, state: BalanceLawTransportState, /) -> BalanceLawSourceView:
        self.validate_state(state)
        state = cast(ConstrainedMHDState, state)
        full = self.integrator.spatial.full_state(
            state.cell_state, state.magnetic_flux
        ).reshape((-1, len(self.component_names)))
        volumes = self.dynamics.discretization.cell_volumes.reshape((-1,))
        return BalanceLawSourceView(
            full,
            volumes,
            jnp.ones(volumes.shape, dtype=bool),
            state.time,
            component_names=self.component_names,
            transport_id=self.transport_id,
        )

    def with_source_view(
        self,
        state: BalanceLawTransportState,
        cell_average: Array,
        /,
    ) -> ConstrainedMHDState:
        self.validate_state(state)
        state = cast(ConstrainedMHDState, state)
        incoming = self.source_view(state).cell_average
        value = jnp.asarray(cell_average)
        if value.shape != incoming.shape:
            raise ValueError("Constrained-MHD source view shape changed.")
        value = eqx.error_if(
            value,
            jnp.any(value[..., 5:8] != incoming[..., 5:8]),
            "Balance-law source process changed face-owned magnetic state.",
        )
        reduced = value[..., :5].reshape(self.integrator.spatial.cell_shape + (5,))
        return ConstrainedMHDState(
            reduced,
            state.magnetic_flux,
            state.time,
            state.step_size,
            state.accepted_step,
            state.status,
        )

    def advance_prescribed(
        self,
        state: BalanceLawTransportState,
        start_time: Array,
        end_time: Array,
        args: Any = None,
        /,
    ) -> BalanceLawTransportAdvance:
        self.validate_state(state)
        state = cast(ConstrainedMHDState, state)
        result = self.integrator.advance(state, start_time, end_time, args)
        stable = jnp.min(result.diagnostics.stage_stable_steps)
        return BalanceLawTransportAdvance(
            state=result.state,
            accepted=result.accepted,
            status=result.state.status,
            stable_step_size=stable,
            stability_margin=result.diagnostics.stability_margin,
            diagnostics=result,
        )

    def auxiliary_state(self, state: BalanceLawTransportState, /) -> Array:
        self.validate_state(state)
        state = cast(ConstrainedMHDState, state)
        return state.magnetic_flux

    def checkpoint_arrays(
        self, state: BalanceLawTransportState, /
    ) -> dict[str, np.ndarray]:
        self.validate_state(state)
        state = cast(ConstrainedMHDState, state)
        return {
            "transport/cell_state": np.asarray(state.cell_state),
            "transport/magnetic_flux": np.asarray(state.magnetic_flux),
            "transport/time": np.asarray(state.time),
            "transport/step_size": np.asarray(state.step_size),
            "transport/accepted_step": np.asarray(state.accepted_step, dtype=np.int32),
            "transport/status": np.asarray(state.status, dtype=np.int32),
        }

    def checkpoint_array_names(self, /) -> frozenset[str]:
        return frozenset(
            {
                "transport/cell_state",
                "transport/magnetic_flux",
                "transport/time",
                "transport/step_size",
                "transport/accepted_step",
                "transport/status",
            }
        )

    def restore_checkpoint(
        self, arrays: Mapping[str, np.ndarray], /
    ) -> ConstrainedMHDState:
        status = int(np.asarray(arrays["transport/status"]))
        if status not in tuple(int(value) for value in ConstrainedMHDRunStatus):
            raise ValueError("Checkpoint constrained-MHD status is invalid.")
        state = ConstrainedMHDState(
            jnp.asarray(arrays["transport/cell_state"]),
            jnp.asarray(arrays["transport/magnetic_flux"]),
            jnp.asarray(arrays["transport/time"]),
            jnp.asarray(arrays["transport/step_size"]),
            jnp.asarray(arrays["transport/accepted_step"]),
            jnp.asarray(status, dtype=jnp.int32),
        )
        self.validate_state(state)
        return state


def prepare_balance_law_transport(
    transport: AbstractPreparedBalanceLawTransport
    | PreparedFiniteVolumeRuntime
    | ConstrainedMHDSSPRK3Plan,
    /,
) -> AbstractPreparedBalanceLawTransport:
    """Prepare one explicit balance-law transport adapter."""
    if isinstance(transport, AbstractPreparedBalanceLawTransport):
        return transport
    if isinstance(transport, PreparedFiniteVolumeRuntime):
        return PreparedFiniteVolumeBalanceLawTransport(transport)
    if isinstance(transport, ConstrainedMHDSSPRK3Plan):
        return PreparedConstrainedMHDBalanceLawTransport(transport)
    raise TypeError("Unsupported balance-law transport implementation.")


__all__ = [
    "AbstractPreparedBalanceLawTransport",
    "BalanceLawSourceView",
    "BalanceLawTransportAdvance",
    "BalanceLawTransportState",
    "PreparedConstrainedMHDBalanceLawTransport",
    "PreparedFiniteVolumeBalanceLawTransport",
    "prepare_balance_law_transport",
]
