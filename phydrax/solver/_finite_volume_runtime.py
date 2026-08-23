#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    FiniteVolumeAdmissibilityReport,
    FiniteVolumeMethodPlan,
    FluxPositivityPlan,
    PiecewiseConstantReconstruction,
    PreparedFiniteVolumeDynamics,
)


class FiniteVolumeRunStatus(IntEnum):
    SUCCESS = 0
    RECOVERED_REJECTION = 1
    INVALID_INITIAL_STATE = 2
    RETRY_LIMIT_REACHED = 3
    MINIMUM_STEP_REACHED = 4
    NONFINITE_STATE = 5


class FiniteVolumeStepPolicy(StrictModule, NonTrainableState):
    cfl: float = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    minimum_step_size: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cfl: float = 0.45,
        maximum_retries: int = 4,
        reduction_factor: float = 0.5,
        minimum_step_size: float = 1e-12,
    ):
        cfl_ = float(cfl)
        retries = int(maximum_retries)
        reduction = float(reduction_factor)
        minimum = float(minimum_step_size)
        if (
            not np.isfinite(cfl_)
            or cfl_ <= 0.0
            or retries < 0
            or not 0.0 < reduction < 1.0
            or not np.isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError("Finite-volume step policy is invalid.")
        self.cfl = cfl_
        self.maximum_retries = retries
        self.reduction_factor = reduction
        self.minimum_step_size = minimum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-volume-step-policy",
                "cfl": cfl_,
                "maximum_retries": retries,
                "reduction_factor": reduction,
                "minimum_step_size": minimum,
            }
        )


class FiniteVolumeRuntimeState(StrictModule):
    conservative_state: Array
    time: Array
    accepted_step: Array
    step_size: Array
    last_status: Array
    controller_state: Array
    integrator_state: Array
    forcing_state: Array
    random_state: Array
    output_cursor: Array

    def __init__(
        self,
        conservative_state: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        accepted_step: ArrayLike = 0,
        last_status: ArrayLike = FiniteVolumeRunStatus.SUCCESS,
        controller_state: ArrayLike | None = None,
        integrator_state: ArrayLike | None = None,
        forcing_state: ArrayLike | None = None,
        random_state: ArrayLike | None = None,
        output_cursor: ArrayLike = 0,
    ):
        self.conservative_state = jnp.asarray(conservative_state)
        self.time = jnp.asarray(time).reshape(())
        self.accepted_step = jnp.asarray(
            accepted_step, dtype=jnp.int32
        ).reshape(())
        self.step_size = jnp.asarray(step_size).reshape(())
        self.last_status = jnp.asarray(
            last_status, dtype=jnp.int32
        ).reshape(())
        self.controller_state = jnp.asarray(
            () if controller_state is None else controller_state
        )
        self.integrator_state = jnp.asarray(
            () if integrator_state is None else integrator_state
        )
        self.forcing_state = jnp.asarray(
            () if forcing_state is None else forcing_state
        )
        self.random_state = jnp.asarray(
            () if random_state is None else random_state,
            dtype=jnp.uint32,
        )
        self.output_cursor = jnp.asarray(
            output_cursor, dtype=jnp.int32
        ).reshape(())


class FiniteVolumeAdvanceResult(StrictModule):
    runtime_state: FiniteVolumeRuntimeState
    accepted: Array
    retries: Array
    attempted_step_size: Array
    accepted_step_size: Array
    positivity: FiniteVolumeAdmissibilityReport
    accepted_integrated_fluxes: tuple[Array, ...]


class PreparedFiniteVolumeRuntime(StrictModule, NonTrainableState):
    """SSPRK3 runtime with conservative fallback blending and bounded retries."""

    dynamics: PreparedFiniteVolumeDynamics
    fallback_dynamics: PreparedFiniteVolumeDynamics
    positivity: FluxPositivityPlan
    policy: FiniteVolumeStepPolicy
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        positivity: FluxPositivityPlan,
        policy: FiniteVolumeStepPolicy | None = None,
        /,
    ):
        if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("dynamics must be PreparedFiniteVolumeDynamics.")
        if not isinstance(positivity, FluxPositivityPlan):
            raise TypeError("positivity must be FluxPositivityPlan.")
        policy_ = FiniteVolumeStepPolicy() if policy is None else policy
        if not isinstance(policy_, FiniteVolumeStepPolicy):
            raise TypeError("policy must be FiniteVolumeStepPolicy.")
        fallback_method = FiniteVolumeMethodPlan(
            PiecewiseConstantReconstruction(),
            positivity.fallback_flux,
            viscous=dynamics.method.viscous,
            differentiability="branchwise",
        )
        fallback = PreparedFiniteVolumeDynamics(
            dynamics.system,
            dynamics.discretization,
            fallback_method,
            dynamics.boundaries,
            capacity=dynamics.capacity,
            bathymetry=dynamics.bathymetry,
            source=dynamics.source,
        )
        self.dynamics = dynamics
        self.fallback_dynamics = fallback
        self.positivity = positivity
        self.policy = policy_
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-volume-runtime",
                "dynamics": dynamics.dynamics_id,
                "fallback": fallback.dynamics_id,
                "positivity": positivity.plan_id,
                "policy": policy_.policy_id,
            }
        )

    def _limited_euler(
        self,
        time: Array,
        evaluation_state: Array,
        combination_base: Array,
        step_size: Array,
        args: Any,
        /,
    ):
        high_fluxes, _ = self.dynamics.face_fluxes(
            time, evaluation_state, args
        )
        fallback_fluxes, _ = self.fallback_dynamics.face_fluxes(
            time, evaluation_state, args
        )
        high_residual = self.dynamics(
            time, evaluation_state, args
        )
        common_residual = high_residual - self.dynamics._flux_residual(
            high_fluxes
        )
        return self.positivity.limit_face_fluxes(
            self.dynamics.system,
            combination_base,
            high_fluxes,
            fallback_fluxes,
            common_residual,
            step_size,
            self.dynamics.discretization,
        )

    def _candidate(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ):
        first = self._limited_euler(
            time, state, state, step_size, args
        )
        second_base = 0.75 * state + 0.25 * first.state
        second = self._limited_euler(
            time + step_size,
            first.state,
            second_base,
            0.25 * step_size,
            args,
        )
        third_base = (
            (1.0 / 3.0) * state + (2.0 / 3.0) * second.state
        )
        third = self._limited_euler(
            time + 0.5 * step_size,
            second.state,
            third_base,
            (2.0 / 3.0) * step_size,
            args,
        )
        integrated_fluxes = tuple(
            (1.0 / 6.0) * first_flux
            + (1.0 / 6.0) * second_flux
            + (2.0 / 3.0) * third_flux
            for first_flux, second_flux, third_flux in zip(
                first.integrated_fluxes,
                second.integrated_fluxes,
                third.integrated_fluxes,
                strict=True,
            )
        )
        normal_fluxes = tuple(
            integrated / measure[..., None]
            for integrated, measure in zip(
                integrated_fluxes,
                self.dynamics.discretization.face_measures,
                strict=True,
            )
        )
        return type(third)(
            state=third.state,
            report=third.report,
            normal_fluxes=normal_fluxes,
            integrated_fluxes=integrated_fluxes,
            face_blend_factors=third.face_blend_factors,
        )

    def advance(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        args: Any = None,
        /,
    ) -> FiniteVolumeAdvanceResult:
        if not isinstance(runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("runtime_state must be FiniteVolumeRuntimeState.")
        valid = jnp.all(
            self.dynamics.system.admissible(
                runtime_state.conservative_state
            )
        )

        def valid_branch(_):
            return self._advance_valid(runtime_state, args)

        def invalid_branch(_):
            state = FiniteVolumeRuntimeState(
                runtime_state.conservative_state,
                runtime_state.time,
                runtime_state.step_size,
                accepted_step=runtime_state.accepted_step,
                last_status=int(
                    FiniteVolumeRunStatus.INVALID_INITIAL_STATE
                ),
                controller_state=runtime_state.controller_state,
                integrator_state=runtime_state.integrator_state,
                forcing_state=runtime_state.forcing_state,
                random_state=runtime_state.random_state,
                output_cursor=runtime_state.output_cursor,
            )
            report = FiniteVolumeAdmissibilityReport(
                high_order_valid=jnp.asarray(False),
                fallback_valid=jnp.asarray(False),
                blend_factor=jnp.asarray(
                    0.0, dtype=runtime_state.conservative_state.dtype
                ),
                activated=jnp.asarray(False),
                minimum_density=jnp.min(
                    runtime_state.conservative_state[..., 0]
                ),
                limited_state_valid=jnp.asarray(False),
                secondary_reduction_applied=jnp.asarray(False),
                secondary_reduction_factor=jnp.asarray(
                    0.0, dtype=runtime_state.conservative_state.dtype
                ),
            )
            zero_fluxes = tuple(
                jnp.zeros(
                    layout.shape
                    + (self.dynamics.discretization.component_count,),
                    dtype=runtime_state.conservative_state.dtype,
                )
                for layout in self.dynamics.discretization.face_layouts
            )
            return FiniteVolumeAdvanceResult(
                runtime_state=state,
                accepted=jnp.asarray(False),
                retries=jnp.asarray(0, dtype=jnp.int32),
                attempted_step_size=runtime_state.step_size,
                accepted_step_size=jnp.asarray(
                    0.0, dtype=runtime_state.step_size.dtype
                ),
                positivity=report,
                accepted_integrated_fluxes=zero_fluxes,
            )

        return jax.lax.cond(valid, valid_branch, invalid_branch, operand=None)

    def _advance_valid(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        args: Any = None,
        /,
    ) -> FiniteVolumeAdvanceResult:
        original = runtime_state.conservative_state
        stable = self.dynamics.stable_step(
            original, args, cfl=self.policy.cfl
        )
        attempted = jnp.minimum(runtime_state.step_size, stable)
        accepted = jnp.asarray(False)
        accepted_state = original
        accepted_dt = jnp.asarray(0.0, dtype=attempted.dtype)
        retries = jnp.asarray(0, dtype=jnp.int32)
        last_report = FiniteVolumeAdmissibilityReport(
            high_order_valid=jnp.asarray(False),
            fallback_valid=jnp.asarray(False),
            blend_factor=jnp.asarray(0.0, dtype=original.dtype),
            activated=jnp.asarray(False),
            minimum_density=jnp.min(original[..., 0]),
            limited_state_valid=jnp.asarray(False),
            secondary_reduction_applied=jnp.asarray(False),
            secondary_reduction_factor=jnp.asarray(
                0.0, dtype=original.dtype
            ),
        )
        accepted_fluxes = tuple(
            jnp.zeros(
                layout.shape
                + (self.dynamics.discretization.component_count,),
                dtype=original.dtype,
            )
            for layout in self.dynamics.discretization.face_layouts
        )
        current_dt = attempted
        for retry in range(self.policy.maximum_retries + 1):
            candidate = self._candidate(
                runtime_state.time, original, current_dt, args
            )
            finite = jnp.all(jnp.isfinite(candidate.state))
            valid = (
                finite
                & candidate.report.fallback_valid
                & candidate.report.limited_state_valid
            )
            take = (~accepted) & valid
            accepted_state = jnp.where(take, candidate.state, accepted_state)
            accepted_dt = jnp.where(take, current_dt, accepted_dt)
            retries = jnp.where(take, retry, retries)
            last_report = jax.tree.map(
                lambda new, old: jnp.where(take, new, old),
                candidate.report,
                last_report,
            )
            accepted_fluxes = tuple(
                jnp.where(take, new, old)
                for new, old in zip(
                    candidate.integrated_fluxes,
                    accepted_fluxes,
                    strict=True,
                )
            )
            accepted = accepted | take
            current_dt = current_dt * self.policy.reduction_factor

        minimum_reached = current_dt < self.policy.minimum_step_size
        status = jnp.where(
            accepted & (retries > 0),
            int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
            jnp.where(
                accepted,
                int(FiniteVolumeRunStatus.SUCCESS),
                jnp.where(
                    minimum_reached,
                    int(FiniteVolumeRunStatus.MINIMUM_STEP_REACHED),
                    int(FiniteVolumeRunStatus.RETRY_LIMIT_REACHED),
                ),
            ),
        )
        next_state = FiniteVolumeRuntimeState(
            accepted_state,
            runtime_state.time + accepted_dt,
            jnp.where(accepted, accepted_dt, runtime_state.step_size),
            accepted_step=(
                runtime_state.accepted_step + accepted.astype(jnp.int32)
            ),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            forcing_state=runtime_state.forcing_state,
            random_state=runtime_state.random_state,
            output_cursor=runtime_state.output_cursor,
        )
        return FiniteVolumeAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            retries=retries,
            attempted_step_size=attempted,
            accepted_step_size=accepted_dt,
            positivity=last_report,
            accepted_integrated_fluxes=accepted_fluxes,
        )


__all__ = [
    "FiniteVolumeAdvanceResult",
    "FiniteVolumeRunStatus",
    "FiniteVolumeRuntimeState",
    "FiniteVolumeStepPolicy",
    "PreparedFiniteVolumeRuntime",
]
