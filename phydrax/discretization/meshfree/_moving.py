# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Extensive moving-surface IMEX dynamics on fixed topology epochs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...lifecycle import commit_candidate, TransactionalCandidate
from ...linalg import (
    AbstractLinearOperator,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    solve,
    TolerancePolicy,
)
from ...solver._balance_law_composition import AdditiveIMEXTableau
from ...solver._conservation_temporal import (
    ConservationIMEXMethod,
    ImplicitConservationStageResult,
)
from ...typing import Bool, Dim, Float64, Int32, Scalar
from .._topology_epoch import TopologyEpoch
from ._capacity import ActivePointDim, MeshfreeCapacityMap
from ._shifting import ShiftAmbientDim
from ._surface import SurfaceRefreshResult
from ._surface_geometry import SurfaceGeometryStatus
from ._surface_transfer import PreparedSurfaceTransfer


class MovingHistoryDim(Dim):
    """Fixed checkpoint history capacity."""


class MovingSurfaceStatus(IntEnum):
    ACCEPTED = 0
    TRUST_REFUSED = 1
    TUBE_REFUSED = 2
    INVALID_GEOMETRY = 3
    IMPLICIT_REFUSED = 4
    HISTORY_EXHAUSTED = 5
    NONCONSERVATIVE_DIFFUSION = 6
    INVALID_STEP = 7


@final
class MovingGeometryRefresh(StrictModule):
    __strict_contract__ = True
    points: Float64[ActivePointDim, ShiftAmbientDim]
    measures: Float64[ActivePointDim]
    normals: Float64[ActivePointDim, ShiftAmbientDim]
    # Maps concentration to EXTENSIVE diffusion rate, with zero column sums.
    diffusion: AbstractLinearOperator
    measure_rate: Float64[ActivePointDim]
    trust_valid: Bool[Scalar]
    tube_valid: Bool[Scalar]
    successful: Bool[Scalar]

    @classmethod
    def from_surface(
        cls,
        refresh: SurfaceRefreshResult,
        diffusion: AbstractLinearOperator,
        measure_rate: ArrayLike,
        /,
    ) -> MovingGeometryRefresh:
        """Consume native fixed-support surface refresh without a second fit.

        ``diffusion`` is the conservative extensive operator supplied by the
        native exterior owner, not the generally nonconservative strong local
        Laplace--Beltrami stencil. A sampled tube estimate is not a certificate.
        """
        if not isinstance(refresh, SurfaceRefreshResult) or not isinstance(
            diffusion, AbstractLinearOperator
        ):
            raise TypeError(
                "Moving geometry requires native surface and linear-operator results."
            )
        rate = jnp.asarray(measure_rate, dtype=jnp.float64)
        if rate.shape != refresh.measures.shape:
            raise ValueError(
                "Measure rate must have one entry per refreshed active point."
            )
        trust_valid = jnp.all(
            refresh.status != int(SurfaceGeometryStatus.SUPPORT_INVALID)
        )
        tube_valid = jnp.asarray(refresh.geometry_evidence.tube_certified) & jnp.all(
            refresh.geometry_evidence.tube_valid
        )
        return cls(
            refresh.points.astype(jnp.float64),
            refresh.measures.astype(jnp.float64),
            refresh.normals.astype(jnp.float64),
            diffusion,
            rate,
            trust_valid,
            tube_valid,
            refresh.accepted,
        )


@final
class MovingSurfaceState(StrictModule):
    __strict_contract__ = True
    points: Float64[ActivePointDim, ShiftAmbientDim]
    measures: Float64[ActivePointDim]
    normals: Float64[ActivePointDim, ShiftAmbientDim]
    content: Float64[ActivePointDim]
    time: Float64[Scalar]
    capacity: MeshfreeCapacityMap
    epoch: TopologyEpoch
    history_content: Float64[MovingHistoryDim, ActivePointDim]
    history_measures: Float64[MovingHistoryDim, ActivePointDim]
    history_points: Float64[MovingHistoryDim, ActivePointDim, ShiftAmbientDim]
    history_times: Float64[MovingHistoryDim]
    history_count: Int32[Scalar]
    accepted_steps: Int32[Scalar]

    @property
    def concentration(self) -> Array:
        return self.content / self.measures


@final
class MovingSurfaceEvidence(StrictModule):
    __strict_contract__ = True
    status: Int32[Scalar]
    successful: Bool[Scalar]
    measure_rate_residual: Float64[Scalar]
    diffusion_column_residual: Float64[Scalar]
    content_before: Float64[Scalar]
    content_after: Float64[Scalar]
    implicit_residual: Float64[Scalar]
    implicit_iterations: Int32[Scalar]
    reaction_content: Float64[Scalar]
    conservation_residual: Float64[Scalar]


@final
class MovingSurfaceStepResult(StrictModule):
    state: MovingSurfaceState
    candidate: MovingSurfaceState
    evidence: MovingSurfaceEvidence


@final
class MovingSurfaceCheckpoint(StrictModule):
    state: MovingSurfaceState
    plan_id: str = eqx.field(static=True)


@final
class MovingSurfaceEpochResult(StrictModule):
    __strict_contract__ = True
    state: MovingSurfaceState
    successful: bool = eqx.field(static=True)
    conservation_residuals: Float64[MovingHistoryDim]
    differentiation_available: bool = eqx.field(static=True, default=False)


@final
class MovingSurfacePlan(StrictModule):
    geometry_refresh: Callable[
        [MovingSurfaceState, Array, Array, Any], MovingGeometryRefresh
    ] = eqx.field(static=True)
    reaction: Callable[[Array, Array, Array, Any], Array] = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    tableau: AdditiveIMEXTableau
    epoch: TopologyEpoch
    history_capacity: int = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry_refresh: Callable[
            [MovingSurfaceState, Array, Array, Any], MovingGeometryRefresh
        ],
        reaction: Callable[[Array, Array, Array, Any], Array],
        /,
        *,
        epoch: TopologyEpoch,
        history_capacity: int,
        plan_id: str,
        linear_policy: LinearSolvePolicy | None = None,
        conservation_tolerance: float = 1e-10,
    ) -> None:
        if (
            not callable(geometry_refresh)
            or not callable(reaction)
            or not isinstance(plan_id, str)
            or not plan_id
            or isinstance(history_capacity, bool)
            or not isinstance(history_capacity, (int, np.integer))
            or history_capacity < 2
        ):
            raise ValueError(
                "Moving runtime needs geometry/reaction callbacks, ID, and at least two history slots."
            )
        if history_capacity > np.iinfo(np.int32).max:
            raise ValueError("Moving history capacity must fit native int32 counters.")
        if not isinstance(epoch, TopologyEpoch):
            raise TypeError("Moving runtime must bind one native TopologyEpoch.")
        if not np.isfinite(conservation_tolerance) or conservation_tolerance <= 0:
            raise ValueError("Conservation tolerance must be positive and finite.")
        policy = (
            linear_policy
            if linear_policy is not None
            else LinearSolvePolicy(
                GMRES(),
                tolerance=TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=200),
            )
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a native LinearSolvePolicy.")
        if (
            policy.tolerance.max_steps is not None
            and policy.tolerance.max_steps > np.iinfo(np.int32).max
        ):
            raise ValueError(
                "Native implicit iteration capacity must fit int32 evidence."
            )
        if policy.failure.mode != "status":
            raise ValueError(
                "Moving IMEX requires status-returning native solves for atomic rollback."
            )
        identifier = canonical_fingerprint(
            {
                "kind": "moving-surface",
                "owner": plan_id,
                "history": history_capacity,
                "epoch": epoch.epoch_id,
            }
        )
        tableau = AdditiveIMEXTableau(
            jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
            jnp.asarray([[0.0, 0.0], [0.0, 1.0]]),
            jnp.asarray([0.0, 1.0]),
            jnp.asarray([0.0, 1.0]),
            explicit_weights=jnp.asarray([1.0, 0.0]),
            implicit_parts=(None, 0),
        )
        self.geometry_refresh, self.reaction = geometry_refresh, reaction
        self.history_capacity, self.conservation_tolerance = (
            int(history_capacity),
            float(conservation_tolerance),
        )
        self.linear_policy, self.plan_id = policy, identifier
        self.epoch = epoch
        self.tableau = tableau

    def initialize(
        self,
        points: ArrayLike,
        measures: ArrayLike,
        normals: ArrayLike,
        concentration: ArrayLike,
        capacity: MeshfreeCapacityMap,
        epoch: TopologyEpoch,
        /,
        *,
        time: float = 0.0,
    ) -> MovingSurfaceState:
        if not isinstance(epoch, TopologyEpoch) or epoch.epoch_id != self.epoch.epoch_id:
            raise ValueError(
                "Initial geometry belongs to a different moving topology epoch."
            )
        x, w, n, c = (
            np.asarray(value, dtype=np.float64)
            for value in (points, measures, normals, concentration)
        )
        if (
            x.ndim != 2
            or x.shape[0] != capacity.active_count
            or w.shape != x.shape[:1]
            or c.shape != w.shape
            or n.shape != x.shape
        ):
            raise ValueError(
                "Initial geometry and field must occupy compact active coordinates."
            )
        if (
            not all(np.all(np.isfinite(value)) for value in (x, w, n, c))
            or not np.all(w > 0)
            or not np.isfinite(time)
        ):
            raise ValueError(
                "Initial moving geometry requires finite data and positive active measures."
            )
        m = jnp.asarray(w * c)
        h = self.history_capacity
        history_content = jnp.zeros((h, w.size)).at[0].set(m)
        history_measures = jnp.broadcast_to(jnp.asarray(w), (h, w.size))
        history_points = jnp.broadcast_to(jnp.asarray(x), (h, *x.shape))
        history_times = jnp.full((h,), time)
        return MovingSurfaceState(
            jnp.asarray(x),
            jnp.asarray(w),
            jnp.asarray(n),
            m,
            jnp.asarray(time),
            capacity,
            epoch,
            history_content,
            history_measures,
            history_points,
            history_times,
            jnp.asarray(1, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )

    def step(
        self, state: MovingSurfaceState, step_size: ArrayLike, /, *, args: Any = None
    ) -> MovingSurfaceStepResult:
        if state.epoch.epoch_id != self.epoch.epoch_id:
            raise ValueError(
                "Moving state changed topology epoch; prepare a new fixed-epoch runtime."
            )
        dt = jnp.asarray(step_size, dtype=state.content.dtype)
        if dt.shape != () or state.history_content.shape[0] != self.history_capacity:
            raise ValueError(
                "Step size must be scalar and history layout must match plan."
            )
        geometry = self.geometry_refresh(state, state.time + dt, dt, args)
        if not isinstance(geometry, MovingGeometryRefresh):
            raise TypeError("Geometry callback must return MovingGeometryRefresh.")
        w = geometry.measures
        if w.shape != state.measures.shape or geometry.points.shape != state.points.shape:
            raise ValueError("Device refresh cannot change active capacity or topology.")
        diffusion = geometry.diffusion
        column_residual = jnp.max(jnp.abs(diffusion.transpose_mv(jnp.ones_like(w))))
        conservative = column_residual <= self.conservation_tolerance
        safe_w = jnp.where(w > 0, w, 1)

        def explicit_rhs(time: Array, content: Array, context: Any, /) -> Array:
            return state.measures * self.reaction(
                time, state.points, content / state.measures, context
            )

        def implicit_rhs(time: Array, content: Array, context: Any, /) -> Array:
            return diffusion.mv(content / safe_w)

        def implicit_solver(
            provisional: Array, time: Array, coefficient: Array, context: Any, /
        ) -> ImplicitConservationStageResult:
            operator = FunctionLinearOperator(
                lambda value: value - coefficient * diffusion.mv(value / safe_w),
                source=diffusion.source,
                target=diffusion.target,
                operator_id=f"{self.plan_id}:implicit-content",
            )
            result = solve(LinearSystem(operator), provisional, policy=self.linear_policy)
            return ImplicitConservationStageResult(
                result.value,
                jnp.all(result.successful),
                jnp.max(result.diagnostics.iterations).astype(jnp.int32),
                jnp.max(result.diagnostics.residual_norm),
                jnp.max(result.status).astype(jnp.int32),
                None,
            )

        # Native forward/backward Euler: explicit reaction at incoming content,
        # then implicit diffusion at the new geometry. Geometry dilution never
        # enters as a spurious source; m=M*c is the integrated conserved state.
        method = ConservationIMEXMethod(
            self.tableau,
            explicit_rhs,
            implicit_rhs,
            implicit_solver,
            method_id=self.plan_id,
        )
        integration = method.step(state.time, state.content, dt, args)
        rate_residual = jnp.max(
            jnp.abs((w - state.measures) - dt * geometry.measure_rate)
        )
        geometry_valid = (
            geometry.successful
            & jnp.all(jnp.isfinite(w) & (w > 0))
            & jnp.all(jnp.isfinite(geometry.points))
            & jnp.all(jnp.isfinite(geometry.normals))
        )
        room = state.history_count < self.history_capacity
        valid_step = jnp.isfinite(dt) & (dt > 0)
        accepted = (
            valid_step
            & room
            & geometry_valid
            & geometry.trust_valid
            & geometry.tube_valid
            & conservative
            & integration.successful
        )
        candidate_content = integration.candidate_state
        index = jnp.minimum(state.history_count, self.history_capacity - 1)
        proposed = MovingSurfaceState(
            geometry.points,
            w,
            geometry.normals,
            candidate_content,
            state.time + dt,
            state.capacity,
            state.epoch,
            state.history_content.at[index].set(candidate_content),
            state.history_measures.at[index].set(w),
            state.history_points.at[index].set(geometry.points),
            state.history_times.at[index].set(state.time + dt),
            state.history_count + jnp.asarray(1, dtype=jnp.int32),
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
        )
        status = jnp.where(
            ~valid_step,
            int(MovingSurfaceStatus.INVALID_STEP),
            jnp.where(
                ~room,
                int(MovingSurfaceStatus.HISTORY_EXHAUSTED),
                jnp.where(
                    ~geometry.trust_valid,
                    int(MovingSurfaceStatus.TRUST_REFUSED),
                    jnp.where(
                        ~geometry.tube_valid,
                        int(MovingSurfaceStatus.TUBE_REFUSED),
                        jnp.where(
                            ~geometry_valid,
                            int(MovingSurfaceStatus.INVALID_GEOMETRY),
                            jnp.where(
                                ~conservative,
                                int(MovingSurfaceStatus.NONCONSERVATIVE_DIFFUSION),
                                jnp.where(
                                    ~integration.successful,
                                    int(MovingSurfaceStatus.IMPLICIT_REFUSED),
                                    int(MovingSurfaceStatus.ACCEPTED),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        reaction_content = dt * jnp.sum(explicit_rhs(state.time, state.content, args))
        before, after = jnp.sum(state.content), jnp.sum(candidate_content)
        evidence = MovingSurfaceEvidence(
            status,
            accepted,
            rate_residual,
            column_residual,
            before,
            after,
            integration.maximum_implicit_residual,
            integration.implicit_iterations.astype(jnp.int32),
            reaction_content,
            after - before - reaction_content,
        )
        committed = commit_candidate(
            TransactionalCandidate(state, proposed, evidence, accepted, self.plan_id)
        )
        return MovingSurfaceStepResult(committed.state, proposed, evidence)

    def checkpoint(self, state: MovingSurfaceState, /) -> MovingSurfaceCheckpoint:
        if state.epoch.epoch_id != self.epoch.epoch_id:
            raise ValueError("Cannot checkpoint a different moving topology epoch.")
        return MovingSurfaceCheckpoint(state, self.plan_id)

    def rollback(self, checkpoint: MovingSurfaceCheckpoint, /) -> MovingSurfaceState:
        if checkpoint.plan_id != self.plan_id:
            raise ValueError("Checkpoint belongs to a different moving plan.")
        return checkpoint.state

    def transition_epoch(
        self,
        state: MovingSurfaceState,
        transfer: PreparedSurfaceTransfer,
        target_epoch: TopologyEpoch,
        target_points: ArrayLike,
        target_normals: ArrayLike,
        target_capacity: MeshfreeCapacityMap,
        history_transfers: Sequence[PreparedSurfaceTransfer],
        target_history_points: ArrayLike,
        /,
    ) -> MovingSurfaceEpochResult:
        """Host atomic transition of current field and EVERY geometry history.

        Each history remap has its own old/new measures. Applying the current
        concentration map to historical extensive arrays would be incorrect.
        A failed route returns the original state, without a partial cutover.
        """
        if state.epoch.epoch_id != self.epoch.epoch_id:
            raise ValueError(
                "Transition source belongs to a different moving topology epoch."
            )
        count = int(np.asarray(state.history_count))
        if len(history_transfers) != count:
            raise ValueError(
                "Every committed history requires its own concentration transfer."
            )
        if not np.array_equal(
            np.asarray(transfer.source_measures), np.asarray(state.measures)
        ):
            raise ValueError("Current transfer measures do not match current geometry.")
        current = transfer.epoch_transition(state.epoch, target_epoch).apply(
            state.concentration
        )
        points, normals = np.asarray(target_points), np.asarray(target_normals)
        history_points = np.asarray(target_history_points)
        n = target_capacity.active_count
        if (
            points.shape != (n, state.points.shape[1])
            or normals.shape != points.shape
            or history_points.shape != (count, *points.shape)
        ):
            raise ValueError(
                "Target current/history geometries must match compact target capacity."
            )
        if not all(
            np.all(np.isfinite(value)) for value in (points, normals, history_points)
        ):
            raise ValueError("Target geometry histories must be finite.")
        contents, measures, residuals = [], [], [current.conservation_residual]
        successful = bool(np.asarray(current.successful))
        for index, route in enumerate(history_transfers):
            if not np.array_equal(
                np.asarray(route.source_measures),
                np.asarray(state.history_measures[index]),
            ) or route.target_measures.shape != (n,):
                raise ValueError(
                    "Historical transfer measures/layout do not match its saved geometry."
                )
            result = route.epoch_transition(state.epoch, target_epoch).apply(
                state.history_content[index] / state.history_measures[index]
            )
            contents.append(jax.lax.stop_gradient(result.values * route.target_measures))
            measures.append(route.target_measures)
            residuals.append(result.conservation_residual)
            successful = successful and bool(np.asarray(result.successful))
        if not successful:
            return MovingSurfaceEpochResult(state, False, jnp.stack(residuals))
        h = self.history_capacity
        hc = jnp.zeros((h, n)).at[:count].set(jnp.stack(contents))
        hm = (
            jnp.broadcast_to(transfer.target_measures, (h, n))
            .at[:count]
            .set(jnp.stack(measures))
        )
        hp = (
            jnp.broadcast_to(jnp.asarray(points), (h, *points.shape))
            .at[:count]
            .set(jnp.asarray(history_points))
        )
        target = MovingSurfaceState(
            jax.lax.stop_gradient(jnp.asarray(points)),
            transfer.target_measures,
            jax.lax.stop_gradient(jnp.asarray(normals)),
            jax.lax.stop_gradient(current.values * transfer.target_measures),
            state.time,
            target_capacity,
            target_epoch,
            hc,
            hm,
            hp,
            state.history_times,
            state.history_count,
            state.accepted_steps,
        )
        target = jax.tree.map(jax.lax.stop_gradient, target)
        return MovingSurfaceEpochResult(target, True, jnp.stack(residuals))


__all__ = [
    "MovingSurfaceStatus",
    "MovingGeometryRefresh",
    "MovingSurfaceState",
    "MovingSurfaceEvidence",
    "MovingSurfaceStepResult",
    "MovingSurfaceCheckpoint",
    "MovingSurfaceEpochResult",
    "MovingSurfacePlan",
]
