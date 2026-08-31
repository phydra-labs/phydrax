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
from ..discretization.finite_volume import UpwindConstrainedTransportPlan


class ConstrainedMHDRunStatus(IntEnum):
    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    STABILITY_LIMIT_EXCEEDED = 2
    POSITIVITY_REJECTED = 3
    MAGNETIC_CONSTRAINT_FAILED = 4
    NONFINITE_STATE = 5


class ConstrainedMHDState(StrictModule):
    cell_state: Array
    magnetic_flux: Array
    time: Array
    step_size: Array
    accepted_step: Array
    status: Array


class ConstrainedMHDDiagnostics(StrictModule):
    stage_stable_steps: Array
    stability_margin: Array
    positivity_factors: Array
    magnetic_constraint_before: Array
    magnetic_constraint_after: Array
    magnetic_constraint_change: Array
    fallback_activated: Array
    successful: Array


class ConstrainedMHDStepResult(StrictModule):
    state: ConstrainedMHDState
    accepted: Array
    diagnostics: ConstrainedMHDDiagnostics


class ConstrainedMHDSSPRK3Plan(StrictModule, NonTrainableState):
    """Coupled SSPRK3 update of cell conservation and face magnetic flux."""

    spatial: UpwindConstrainedTransportPlan
    cfl: float = eqx.field(static=True)
    positivity_iterations: int = eqx.field(static=True)
    divergence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial: UpwindConstrainedTransportPlan,
        /,
        *,
        cfl: float = 0.35,
        positivity_iterations: int = 32,
        divergence_tolerance: float = 1e-10,
    ):
        if not isinstance(spatial, UpwindConstrainedTransportPlan):
            raise TypeError("spatial must be UpwindConstrainedTransportPlan.")
        cfl_ = float(cfl)
        iterations = int(positivity_iterations)
        tolerance = float(divergence_tolerance)
        if (
            not np.isfinite(cfl_)
            or not 0.0 < cfl_ <= 1.0
            or iterations <= 0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Constrained-MHD integration controls are invalid.")
        self.spatial = spatial
        self.cfl = cfl_
        self.positivity_iterations = iterations
        self.divergence_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "constrained-mhd-ssprk3",
                "spatial": spatial.plan_id,
                "cfl": cfl_,
                "positivity_iterations": iterations,
                "divergence_tolerance": tolerance,
            }
        )

    def initialize(
        self,
        full_cell_state: ArrayLike,
        magnetic_flux: ArrayLike,
        time: ArrayLike = 0.0,
        /,
        *,
        step_size: ArrayLike | None = None,
    ) -> ConstrainedMHDState:
        full = jnp.asarray(full_cell_state)
        expected = self.spatial.cell_shape + (8,)
        if full.shape != expected:
            raise ValueError(f"Full MHD state must have shape {expected}.")
        magnetic = self.spatial.validate_magnetic_flux(magnetic_flux)
        reduced = full[..., :5]
        synchronized = self.spatial.full_state(reduced, magnetic)
        magnetic_mismatch = jnp.max(jnp.abs(synchronized[..., 5:8] - full[..., 5:8]))
        constraint = self.spatial.magnetic_constraint(magnetic)
        valid = (
            jnp.all(jnp.isfinite(synchronized))
            & jnp.all(self.spatial.dynamics.system.admissible(synchronized))
            & (magnetic_mismatch <= self.divergence_tolerance)
            & (jnp.max(jnp.abs(constraint), initial=0.0) <= self.divergence_tolerance)
        )
        reduced = eqx.error_if(
            reduced,
            ~valid,
            "Initial constrained-MHD state is inadmissible or inconsistent.",
        )
        time_ = jnp.asarray(time, dtype=reduced.dtype).reshape(())
        step_size_ = jnp.asarray(
            jnp.nan if step_size is None else step_size,
            dtype=reduced.dtype,
        ).reshape(())
        return ConstrainedMHDState(
            reduced,
            magnetic,
            time_,
            step_size_,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(int(ConstrainedMHDRunStatus.SUCCESS), dtype=jnp.int32),
        )

    def _admissible(self, cell_state: Array, magnetic_flux: Array, /) -> Array:
        full = self.spatial.full_state(cell_state, magnetic_flux)
        return jnp.all(jnp.isfinite(full)) & jnp.all(
            self.spatial.dynamics.system.admissible(full)
        )

    def _blend(
        self,
        base_cell: Array,
        base_magnetic: Array,
        candidate_cell: Array,
        candidate_magnetic: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        candidate_valid = self._admissible(candidate_cell, candidate_magnetic)

        def body(_, bounds):
            lower, upper = bounds
            midpoint = 0.5 * (lower + upper)
            cell = base_cell + midpoint * (candidate_cell - base_cell)
            magnetic = base_magnetic + midpoint * (candidate_magnetic - base_magnetic)
            valid = self._admissible(cell, magnetic)
            return jnp.where(valid, midpoint, lower), jnp.where(valid, upper, midpoint)

        lower, _ = jax.lax.fori_loop(
            0,
            self.positivity_iterations,
            body,
            (
                jnp.asarray(0.0, dtype=base_cell.dtype),
                jnp.asarray(1.0, dtype=base_cell.dtype),
            ),
        )
        factor = jnp.where(candidate_valid, 1.0, lower)
        return (
            base_cell + factor * (candidate_cell - base_cell),
            base_magnetic + factor * (candidate_magnetic - base_magnetic),
            factor,
        )

    def _euler_stage(
        self,
        time: Array,
        evaluation_cell: Array,
        evaluation_magnetic: Array,
        base_cell: Array,
        base_magnetic: Array,
        increment: Array,
        args: Any,
        /,
    ):
        rate = self.spatial.rate(
            time,
            evaluation_cell,
            evaluation_magnetic,
            args,
            cfl=self.cfl,
        )
        candidate_cell = base_cell + increment * rate.cell_rate
        candidate_magnetic = base_magnetic + increment * rate.magnetic_rate
        cell, magnetic, factor = self._blend(
            base_cell,
            base_magnetic,
            candidate_cell,
            candidate_magnetic,
        )
        return cell, magnetic, factor, rate

    def advance(
        self,
        state: ConstrainedMHDState,
        start_time: ArrayLike,
        end_time: ArrayLike,
        args: Any = None,
        /,
    ) -> ConstrainedMHDStepResult:
        if not isinstance(state, ConstrainedMHDState):
            raise TypeError("state must be ConstrainedMHDState.")
        start = jnp.asarray(start_time, dtype=state.time.dtype).reshape(())
        end = jnp.asarray(end_time, dtype=state.time.dtype).reshape(())
        step = end - start
        tolerance = 32.0 * jnp.finfo(start.dtype).eps * jnp.maximum(jnp.abs(start), 1.0)
        start = eqx.error_if(
            start,
            ~jnp.isfinite(start)
            | ~jnp.isfinite(end)
            | (step <= 0.0)
            | (jnp.abs(state.time - start) > tolerance),
            "Constrained-MHD interval is invalid or state time is stale.",
        )
        del start
        cell_0 = self.spatial.validate_reduced_state(state.cell_state)
        magnetic_0 = self.spatial.validate_magnetic_flux(state.magnetic_flux)
        constraint_before = self.spatial.magnetic_constraint(magnetic_0)
        cell_1, magnetic_1, factor_1, rate_1 = self._euler_stage(
            state.time,
            cell_0,
            magnetic_0,
            cell_0,
            magnetic_0,
            step,
            args,
        )
        base_cell_2 = 0.75 * cell_0 + 0.25 * cell_1
        base_magnetic_2 = 0.75 * magnetic_0 + 0.25 * magnetic_1
        cell_2, magnetic_2, factor_2, rate_2 = self._euler_stage(
            state.time + step,
            cell_1,
            magnetic_1,
            base_cell_2,
            base_magnetic_2,
            0.25 * step,
            args,
        )
        base_cell_3 = (1.0 / 3.0) * cell_0 + (2.0 / 3.0) * cell_2
        base_magnetic_3 = (1.0 / 3.0) * magnetic_0 + (2.0 / 3.0) * magnetic_2
        cell_3, magnetic_3, factor_3, rate_3 = self._euler_stage(
            state.time + 0.5 * step,
            cell_2,
            magnetic_2,
            base_cell_3,
            base_magnetic_3,
            (2.0 / 3.0) * step,
            args,
        )
        stable_steps = jnp.stack(
            (rate_1.stable_step, rate_2.stable_step, rate_3.stable_step)
        )
        stable = jnp.min(stable_steps)
        constraint_after = self.spatial.magnetic_constraint(magnetic_3)
        constraint_change = jnp.max(
            jnp.abs(constraint_after - constraint_before), initial=0.0
        )
        admissible = self._admissible(cell_3, magnetic_3)
        finite = jnp.all(jnp.isfinite(cell_3)) & jnp.all(jnp.isfinite(magnetic_3))
        stable_valid = step <= stable + tolerance
        constraint_valid = constraint_change <= self.divergence_tolerance
        successful = finite & admissible & stable_valid & constraint_valid
        status = jnp.where(
            successful,
            int(ConstrainedMHDRunStatus.SUCCESS),
            jnp.where(
                ~finite,
                int(ConstrainedMHDRunStatus.NONFINITE_STATE),
                jnp.where(
                    ~stable_valid,
                    int(ConstrainedMHDRunStatus.STABILITY_LIMIT_EXCEEDED),
                    jnp.where(
                        ~constraint_valid,
                        int(ConstrainedMHDRunStatus.MAGNETIC_CONSTRAINT_FAILED),
                        int(ConstrainedMHDRunStatus.POSITIVITY_REJECTED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        candidate = ConstrainedMHDState(
            cell_3,
            magnetic_3,
            end,
            step,
            state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
            status,
        )
        rejected = ConstrainedMHDState(
            state.cell_state,
            state.magnetic_flux,
            state.time,
            state.step_size,
            state.accepted_step,
            status,
        )
        accepted_state = jax.lax.cond(
            successful,
            lambda _: candidate,
            lambda _: rejected,
            operand=None,
        )
        diagnostics = ConstrainedMHDDiagnostics(
            stage_stable_steps=stable_steps,
            stability_margin=stable / step - 1.0,
            positivity_factors=jnp.stack((factor_1, factor_2, factor_3)),
            magnetic_constraint_before=constraint_before,
            magnetic_constraint_after=constraint_after,
            magnetic_constraint_change=constraint_change,
            fallback_activated=(
                rate_1.fallback_activated
                | rate_2.fallback_activated
                | rate_3.fallback_activated
            ),
            successful=successful,
        )
        return ConstrainedMHDStepResult(accepted_state, successful, diagnostics)


__all__ = [
    "ConstrainedMHDDiagnostics",
    "ConstrainedMHDRunStatus",
    "ConstrainedMHDSSPRK3Plan",
    "ConstrainedMHDState",
    "ConstrainedMHDStepResult",
]
