#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._unstructured_dynamics import (
    PreparedUnstructuredFiniteVolumeDynamics,
)
from ..discretization.finite_volume._vof_phase_change import (
    VOFPhaseChangePlan,
    VOFPhaseChangeStageEvaluation,
)
from ._finite_volume_content import FiniteVolumeConservativeContentState
from ._finite_volume_runtime import (
    FiniteVolumeAdvanceResult,
    FiniteVolumeRuntimeState,
    PreparedFiniteVolumeRuntime,
)


class FiniteVolumePhaseChangeStrangResult(StrictModule):
    runtime_state: FiniteVolumeRuntimeState
    transport: FiniteVolumeAdvanceResult
    first_half: VOFPhaseChangeStageEvaluation
    second_half: VOFPhaseChangeStageEvaluation
    attempted_step_size: Array
    exact_transport_step: Array
    finite: Array
    accepted: Array
    method_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.accepted


class FiniteVolumePhaseChangeStrangMethod(StrictModule, NonTrainableState):
    transport_runtime: PreparedFiniteVolumeRuntime
    phase_change: VOFPhaseChangePlan
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport_runtime: PreparedFiniteVolumeRuntime,
        phase_change: VOFPhaseChangePlan,
        /,
    ):
        if not isinstance(transport_runtime, PreparedFiniteVolumeRuntime):
            raise TypeError("transport_runtime must be PreparedFiniteVolumeRuntime.")
        if not isinstance(phase_change, VOFPhaseChangePlan):
            raise TypeError("phase_change must be VOFPhaseChangePlan.")
        dynamics = transport_runtime.dynamics
        if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("Phase-change Strang splitting requires unstructured FV.")
        if dynamics.coupling.vof is None or (
            dynamics.coupling.vof.plan_id != phase_change.vof.plan_id
        ):
            raise ValueError(
                "Transport runtime and phase change must share VOF geometry."
            )
        if dynamics.coupling.phase_change is not None:
            raise ValueError(
                "Strang splitting requires a transport-only runtime; phase change "
                "cannot also be present in its spatial source ledger."
            )
        if phase_change.thermal_diffusion is not None:
            raise ValueError(
                "Strang splitting currently composes local transfer only; thermal "
                "diffusion must use the native stage source path."
            )
        if any(
            component is not None
            for component in (
                dynamics.coupling.motion,
                dynamics.coupling.embedded_boundary,
                dynamics.coupling.amr,
                dynamics.coupling.overset,
                dynamics.coupling.sliding,
                dynamics.coupling.topology_event_id,
            )
        ):
            raise ValueError(
                "Phase-change Strang splitting currently requires static physical geometry."
            )
        self.transport_runtime = transport_runtime
        self.phase_change = phase_change
        self.method_id = canonical_fingerprint(
            {
                "kind": "finite-volume-phase-change-strang",
                "runtime": transport_runtime.runtime_id,
                "phase_change": phase_change.plan_id,
            }
        )

    @staticmethod
    def _with_average(
        runtime_state: FiniteVolumeRuntimeState, average: Array, /
    ) -> FiniteVolumeRuntimeState:
        content = runtime_state.content_state
        updated = FiniteVolumeConservativeContentState.from_cell_average(
            average,
            content.effective_cell_volumes,
            content.active_cell_mask,
            content.time,
            topology_epoch_id=content.topology_epoch_id,
            geometry_family_id=content.geometry_family_id,
            geometry_layout_id=content.geometry_layout_id,
            geometry_version=content.geometry_version,
            evidence_policy_id=content.evidence_policy_id,
            evidence_version=content.evidence_version,
            precision=content.precision,
        )
        return eqx.tree_at(lambda value: value.content_state, runtime_state, updated)

    def _half_source(
        self, average: Array, step_size: Array, content, /
    ) -> VOFPhaseChangeStageEvaluation:
        alpha = average[:, self.phase_change.phase_change.system.alpha_index]
        plic = self.phase_change.vof.reconstruct_stage(
            alpha,
            geometry_layout_id=content.geometry_layout_id,
            geometry_version=content.geometry_version,
        )
        return self.phase_change.evaluate_stage(
            average,
            0.5 * step_size,
            plic,
        )

    def step(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        args: Any = None,
        /,
    ) -> FiniteVolumePhaseChangeStrangResult:
        if not isinstance(runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("runtime_state must be FiniteVolumeRuntimeState.")
        step = runtime_state.step_size
        original = runtime_state.content_state.cell_average()
        first = self._half_source(original, step, runtime_state.content_state)
        first_average = first.result.state
        source_state = self._with_average(runtime_state, first_average)
        transported = self.transport_runtime.advance(source_state, args)
        transported_average = transported.runtime_state.content_state.cell_average()
        second = self._half_source(
            transported_average,
            step,
            transported.runtime_state.content_state,
        )
        tolerance = 32.0 * jnp.finfo(step.dtype).eps * jnp.maximum(jnp.abs(step), 1.0)
        exact_transport = (
            transported.accepted
            & (transported.retries == 0)
            & (jnp.abs(transported.accepted_step_size - step) <= tolerance)
        )
        finite = first.finite & second.finite & jnp.all(jnp.isfinite(second.result.state))
        accepted = finite & first.successful & second.successful & exact_transport
        candidate = self._with_average(transported.runtime_state, second.result.state)
        final_state = jax.tree.map(
            lambda proposed, original: (
                jnp.where(accepted, proposed, original)
                if eqx.is_array(proposed)
                else proposed
            ),
            candidate,
            runtime_state,
        )
        return FiniteVolumePhaseChangeStrangResult(
            final_state,
            transported,
            first,
            second,
            step,
            exact_transport,
            finite,
            accepted,
            self.method_id,
        )


__all__ = [
    "FiniteVolumePhaseChangeStrangMethod",
    "FiniteVolumePhaseChangeStrangResult",
]
