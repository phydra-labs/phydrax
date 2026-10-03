# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Stage-correct extensive moving-surface IMEX dynamics on fixed topology epochs.

The native additive IMEX method advances one packed state: extensive content
``m``, a stage-quadrature measure witness ``W``, the explicit reaction ledger
and, for integrated motion laws, the point coordinates. Every stage evaluates
geometry, measures, motion, reaction and diffusion at its own stage time and
stage coordinates. ``W`` integrates the measure-rate source identity with the
tableau's explicit weights, so ``W - w(X)`` at the step end is exactly the
discrete geometric-conservation residual of the selected stage rule.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any, assert_never, final, Literal, TypeAlias

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
    ArraySpace,
    bind_numeric,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveTemplate,
    LinearSystem,
    prepare_template,
    solve,
    TolerancePolicy,
)
from ...solver._balance_law_composition import (
    additive_imex_tableau,
    AdditiveIMEXScheme,
    AdditiveIMEXTableau,
)
from ...solver._conservation_temporal import (
    ConservationIMEXMethod,
    ConservationIMEXResult,
    ImplicitConservationStageResult,
)
from ...typing import Bool, checked, Dim, Float64, Int32, parse, Scalar
from .._topology_epoch import TopologyEpoch
from ._capacity import ActivePointDim, MeshfreeCapacityMap
from ._epochs import remap_live_histories
from ._motion import (
    AbstractSurfaceMotionLaw,
    ChartMotion,
    MovingGeometryRefresh,
    SurfaceMotion,
)
from ._shifting import ShiftAmbientDim, SurfaceMeshShift
from ._transfer import PreparedPointTransfer


class MovingHistoryDim(Dim):
    """Bounded rolling live-history capacity."""


class MovingStageDim(Dim):
    """Stages of the selected additive IMEX tableau."""


class MovingTransferDim(Dim):
    """Current state followed by every remapped live history."""


MovingArchiveMode: TypeAlias = Literal["rolling", "acknowledged"]
"""``rolling`` evicts the oldest live history when the window is full;
``acknowledged`` evicts only histories whose lifetime index precedes the
archive cursor and otherwise refuses the step."""

MovingGCLAdmission: TypeAlias = Literal["stage-quadrature", "diagnostic"]
"""``stage-quadrature`` refuses a step whose discrete GCL defect exceeds the
declared rate tolerance; ``diagnostic`` reports it without admission."""

MovingMeasureLaw: TypeAlias = Literal["geometric", "conservative"]
"""Measures of record. ``geometric`` takes the refreshed geometry's measures;
the integrated measure witness is the stage-quadrature GCL check.
``conservative`` takes the integrated witness itself, so concentration is
content over a measure advanced by the same discrete divergence as the
relative content flux (discrete GCL by construction); the refreshed measures
then report the accumulated geometric drift."""


class MovingSurfaceStatus(IntEnum):
    ACCEPTED = 0
    TRUST_REFUSED = 1
    TUBE_REFUSED = 2
    INVALID_GEOMETRY = 3
    IMPLICIT_REFUSED = 4
    HISTORY_EXHAUSTED = 5
    NONCONSERVATIVE_DIFFUSION = 6
    INVALID_STEP = 7
    MOTION_REFUSED = 8
    CONSERVATION_DEFECT = 9
    GCL_REFUSED = 10
    TRANSPORT_REFUSED = 11
    POSITIVITY_REFUSED = 12


@final
class MovingGCLPolicy(StrictModule):
    """Discrete geometric-conservation admission of one stage rule.

    The defect is ``max |w(X_end) - W_end| / (|dt| max w_start)``: the
    relative measure-rate mismatch per unit time between the geometric
    measures and the stage quadrature of ``w (H V_n + div_G u_tau)``. It
    combines the stage rule's ``O(dt^p)`` quadrature error with the spatial
    consistency of measures, curvature and divergence; it is not required to
    vanish.
    """

    admission: MovingGCLAdmission = eqx.field(static=True)
    rate_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        admission: MovingGCLAdmission = "stage-quadrature",
        rate_tolerance: float = 1e-2,
    ) -> None:
        mode = parse(admission, MovingGCLAdmission, "admission")
        if not np.isfinite(rate_tolerance) or rate_tolerance <= 0:
            raise ValueError("GCL rate tolerance must be positive and finite.")
        self.admission = mode
        self.rate_tolerance = float(rate_tolerance)


@final
class MovingSurfaceState(StrictModule):
    """Accepted moving state with a rolling window of live histories.

    History slot ``k`` holds the entry whose lifetime index ``L`` satisfies
    ``L % capacity == k``; the newest entry is the current state with lifetime
    index ``accepted_steps``. Entries with lifetime index below
    ``archive_cursor`` have been acknowledged by the durable archive owner.
    """

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
    archive_cursor: Int32[Scalar]

    @property
    def concentration(self) -> Array:
        return self.content / self.measures

    def live_slots(self) -> np.ndarray:
        """Host ring slots of the live histories, oldest first."""
        count = int(np.asarray(self.history_count))
        newest = int(np.asarray(self.accepted_steps))
        lifetimes = np.arange(newest - count + 1, newest + 1, dtype=np.int64)
        return (lifetimes % self.history_times.shape[0]).astype(np.int32)

    def live_lifetimes(self) -> np.ndarray:
        """Host lifetime indices of the live histories, oldest first."""
        count = int(np.asarray(self.history_count))
        newest = int(np.asarray(self.accepted_steps))
        return np.arange(newest - count + 1, newest + 1, dtype=np.int32)


@final
class MovingStageAdmission(StrictModule):
    """Geometry, motion, diffusion and relative-transport admission at a stage.

    ``transport_outflow_rate`` is ``max_i outflow_i / w_i`` of the relative
    mesh-shift flux (zero without a shift): the outgoing CFL rate whose product
    with the step is the forward-Euler positivity measure of the upwind flux.
    """

    __strict_contract__ = True
    trust_valid: Bool[Scalar]
    tube_valid: Bool[Scalar]
    geometry_valid: Bool[Scalar]
    motion_valid: Bool[Scalar]
    transport_valid: Bool[Scalar]
    diffusion_column_residual: Float64[Scalar]
    transport_outflow_rate: Float64[Scalar]


@final
class MovingSurfaceEvidence(StrictModule):
    """Independent admission statuses and diagnostics of one attempted step.

    ``status`` reports the first refusal in a fixed precedence; the boolean
    fields remain independent so content conservation, GCL consistency,
    geometry trust/tube, motion, relative transport, positivity, history and
    solver success are each visible. ``transport_cfl`` is the step times the
    largest stage outgoing relative-flux rate; ``positivity_admitted`` is the
    a-posteriori sign check of the candidate requested by the plan.
    """

    __strict_contract__ = True
    status: Int32[Scalar]
    successful: Bool[Scalar]
    trust_valid: Bool[Scalar]
    tube_valid: Bool[Scalar]
    geometry_valid: Bool[Scalar]
    motion_valid: Bool[Scalar]
    transport_valid: Bool[Scalar]
    positivity_admitted: Bool[Scalar]
    diffusion_conservative: Bool[Scalar]
    content_conserved: Bool[Scalar]
    gcl_admitted: Bool[Scalar]
    solver_successful: Bool[Scalar]
    history_admitted: Bool[Scalar]
    evicted_lifetime: Int32[Scalar]
    gcl_defect: Float64[Scalar]
    geometric_drift: Float64[Scalar]
    measure_rate_residual: Float64[Scalar]
    diffusion_column_residual: Float64[Scalar]
    transport_cfl: Float64[Scalar]
    minimum_concentration: Float64[Scalar]
    content_before: Float64[Scalar]
    content_after: Float64[Scalar]
    reaction_content: Float64[Scalar]
    conservation_residual: Float64[Scalar]
    implicit_residual: Float64[Scalar]
    implicit_iterations: Int32[Scalar]
    stage_successful: Bool[MovingStageDim]
    stage_iterations: Int32[MovingStageDim]
    stage_status: Int32[MovingStageDim]


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
    conservation_residuals: Float64[MovingTransferDim]
    differentiation_available: bool = eqx.field(static=True, default=False)


def _stage_admission(
    geometry: MovingGeometryRefresh,
    motion: SurfaceMotion,
    shift: SurfaceMeshShift | None,
    /,
) -> MovingStageAdmission:
    measures = geometry.measures
    geometry_valid = (
        geometry.successful
        & jnp.all(jnp.isfinite(measures) & (measures > 0))
        & jnp.all(jnp.isfinite(geometry.points))
        & jnp.all(jnp.isfinite(geometry.normals))
        & jnp.all(jnp.isfinite(geometry.mean_curvature))
    )
    column = jnp.max(jnp.abs(geometry.diffusion.transpose_mv(jnp.ones_like(measures))))
    if shift is None:
        transport_valid = jnp.asarray(True)
        outflow_rate = jnp.asarray(0.0, dtype=jnp.float64)
    else:
        # Outflow depends on the relative flux only, not on the transported values.
        rate = shift.rate(
            jnp.ones_like(measures), geometry.points, motion.relative_velocity
        )
        transport_valid = rate.status == 0
        safe = jnp.where(measures > 0, measures, 1)
        outflow_rate = jnp.max(rate.node_outflow / safe)
    return MovingStageAdmission(
        geometry.trust_valid,
        geometry.tube_valid,
        geometry_valid,
        motion.valid,
        transport_valid,
        column.astype(jnp.float64),
        outflow_rate.astype(jnp.float64),
    )


def _validated_tableau(
    method: AdditiveIMEXScheme | AdditiveIMEXTableau, /
) -> AdditiveIMEXTableau:
    """Admit tableaux whose every stage has one time and reaches stage evidence.

    Geometry, measures and reaction are evaluated at one time per stage, so
    both parts must share their stage nodes (``A_E 1 = A_I 1 = c``); schemes
    with distinct explicit and implicit abscissae are refused rather than
    evaluated at a time that is wrong for one part. The first stage is either
    the incoming state (explicit-only at node zero) or a nonzero-diagonal
    implicit stage; every later stage solves a nonzero-diagonal implicit
    stage, whose solver records its admission.
    """
    tableau = (
        method
        if isinstance(method, AdditiveIMEXTableau)
        else additive_imex_tableau(method)
    )
    if tableau.part_count != 1:
        raise ValueError("Moving surfaces use one implicit diffusion part.")
    nodes = np.asarray(tableau.nodes)
    if not (
        np.allclose(np.sum(np.asarray(tableau.explicit_matrix), axis=1), nodes)
        and np.allclose(np.sum(np.asarray(tableau.implicit_matrix), axis=1), nodes)
    ):
        raise ValueError(
            "Moving stages need one shared stage time for explicit and implicit parts."
        )
    diagonal = np.diag(np.asarray(tableau.implicit_matrix))
    for stage, part in enumerate(tableau.implicit_parts):
        if part is None and (stage > 0 or nodes[0] != 0.0):
            raise ValueError(
                "Only an explicit first stage at node zero may skip the implicit solve."
            )
        if part is not None and diagonal[stage] == 0.0:
            raise ValueError("Every implicit moving stage needs a nonzero diagonal.")
    return tableau


@final
class MovingSurfacePlan(StrictModule):
    """Fixed-epoch moving-surface reaction--diffusion runtime.

    ``geometry(points, time, args)`` returns the native stage geometry,
    ``motion`` is a typed surface motion law and
    ``reaction(time, points, concentration, args)`` the intensive reaction.
    An optional ``shift`` adds a tangential mesh redistribution whose relative
    content flux is integrated by the same native stage rule as the content,
    coordinates and measure witness. ``require_positivity`` refuses a
    candidate with a negative concentration. The implicit content solve is
    planned once as a native template and bound to fresh coefficients at
    every stage.

    ``measure_law`` selects the measures of record (:data:`MovingMeasureLaw`).
    A mesh shift requires ``"conservative"``: its measure source is the
    shift owner's graph volume rate, the same discrete divergence as its
    relative content flux, so a constant concentration stays constant.

    The geometry, reaction and motion callables are PyTree leaves: a callable
    module carrying arrays (for example :class:`SurfaceGeometryProvider`)
    keeps them dynamic. Callables are opaque to identity, so ``plan_id`` and
    each law's ``law_id`` are the explicit scientific identities.
    """

    geometry: Callable[[Array, Array, Any], MovingGeometryRefresh]
    motion: AbstractSurfaceMotionLaw
    reaction: Callable[[Array, Array, Array, Any], ArrayLike]
    tableau: AdditiveIMEXTableau
    linear_template: LinearSolveTemplate
    epoch: TopologyEpoch
    capacity: MeshfreeCapacityMap
    gcl: MovingGCLPolicy
    shift: SurfaceMeshShift | None
    measure_law: MovingMeasureLaw = eqx.field(static=True)
    require_positivity: bool = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    archive: MovingArchiveMode = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    ledger_tolerance: float = eqx.field(static=True)
    implicit_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        geometry: Callable[[Array, Array, Any], MovingGeometryRefresh],
        motion: AbstractSurfaceMotionLaw,
        reaction: Callable[[Array, Array, Array, Any], ArrayLike],
        /,
        *,
        method: AdditiveIMEXScheme | AdditiveIMEXTableau,
        epoch: TopologyEpoch,
        capacity: MeshfreeCapacityMap,
        history_capacity: int,
        plan_id: str,
        archive: MovingArchiveMode = "rolling",
        gcl: MovingGCLPolicy | None = None,
        linear_policy: LinearSolvePolicy | None = None,
        shift: SurfaceMeshShift | None = None,
        require_positivity: bool = False,
        measure_law: MovingMeasureLaw = "geometric",
        conservation_tolerance: float = 1e-10,
        ledger_tolerance: float = 1e-8,
    ) -> None:
        if not plan_id:
            raise ValueError("plan_id must be a non-empty string.")
        if (
            isinstance(history_capacity, bool)
            or not isinstance(history_capacity, (int, np.integer))
            or not 1 <= history_capacity <= np.iinfo(np.int32).max
        ):
            raise ValueError("Moving history capacity must be a positive int32 count.")
        if epoch.partition_id != capacity.mapping_id:
            raise ValueError("Topology epoch partition must be the bound capacity map.")
        archive_ = parse(archive, MovingArchiveMode, "archive")
        law = parse(measure_law, MovingMeasureLaw, "measure_law")
        if shift is not None and isinstance(motion, ChartMotion):
            raise ValueError(
                "A mesh shift moves integrated coordinates; authoritative chart "
                "positions cannot be redistributed."
            )
        if shift is not None and law != "conservative":
            raise ValueError(
                "A mesh shift requires the conservative measure law: its relative "
                "content flux and the measure evolution must share one discrete "
                "divergence."
            )
        if not isinstance(require_positivity, bool):
            raise TypeError("require_positivity must be bool.")
        gcl_ = MovingGCLPolicy() if gcl is None else gcl
        for value in (conservation_tolerance, ledger_tolerance):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Conservation tolerances must be positive and finite.")
        tableau = _validated_tableau(method)
        policy = (
            LinearSolvePolicy(
                GMRES(),
                tolerance=TolerancePolicy(relative=1e-12, absolute=1e-14, max_steps=200),
            )
            if linear_policy is None
            else linear_policy
        )
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
                "motion": motion.law_id,
                "tableau": tableau.tableau_id,
                "history": int(history_capacity),
                "archive": archive_,
                "gcl": [gcl_.admission, gcl_.rate_tolerance],
                "shift": None if shift is None else shift.shift_id,
                "positivity": require_positivity,
                "measure_law": law,
                "epoch": epoch.epoch_id,
            }
        )
        implicit_id = f"{identifier}:implicit-content"
        space = ArraySpace((capacity.active_count,), dtype=np.float64)
        # Coefficient-independent structure: every stage binds the same
        # operator identity and spaces with refreshed numerical coefficients.
        structure = FunctionLinearOperator(
            lambda value: value, source=space, target=space, operator_id=implicit_id
        )
        self.geometry, self.motion, self.reaction = geometry, motion, reaction
        self.tableau = tableau
        self.linear_template = prepare_template(
            LinearSystem(structure, problem_id=implicit_id), policy
        )
        self.epoch, self.capacity, self.gcl = epoch, capacity, gcl_
        self.shift, self.require_positivity = shift, require_positivity
        self.measure_law = law
        self.history_capacity = int(history_capacity)
        self.archive = archive_
        self.conservation_tolerance = float(conservation_tolerance)
        self.ledger_tolerance = float(ledger_tolerance)
        self.implicit_id = implicit_id
        self.plan_id = identifier

    @property
    def integrates_points(self) -> bool:
        """Whether coordinates are time-integrated rather than authoritative."""
        return not isinstance(self.motion, ChartMotion)

    def _stage_points(self, time: Array, packed: Array, count: int) -> Array:
        motion = self.motion
        if isinstance(motion, ChartMotion):
            return motion.positions(time)
        return packed[2 * count + 1 :].reshape((count, -1))

    def _stage_geometry(
        self, time: Array, packed: Array, count: int, args: Any, /
    ) -> tuple[MovingGeometryRefresh, Array]:
        """Stage geometry carrying the measures of record, and the refreshed ones."""
        points = self._stage_points(time, packed, count)
        geometry = self.geometry(points, time, args)
        if not isinstance(geometry, MovingGeometryRefresh):
            raise TypeError("Geometry callback must return MovingGeometryRefresh.")
        if geometry.points.shape != points.shape:
            raise ValueError("Stage geometry cannot change active capacity or layout.")
        refreshed = geometry.measures
        match self.measure_law:
            case "geometric":
                return geometry, refreshed
            case "conservative":
                witness = packed[count : 2 * count]
                return eqx.tree_at(
                    lambda item: item.measures, geometry, witness
                ), refreshed
            case _:
                assert_never(self.measure_law)

    def _evaluate(
        self, time: Array, packed: Array, count: int, args: Any, /
    ) -> tuple[MovingGeometryRefresh, SurfaceMotion]:
        geometry, _ = self._stage_geometry(time, packed, count, args)
        return geometry, self._stage_motion(time, geometry, args)

    def _stage_motion(
        self, time: Array, geometry: MovingGeometryRefresh, args: Any, /
    ) -> SurfaceMotion:
        motion = self.motion.motion(time, geometry, args)
        if self.shift is None:
            return motion
        velocity = self.shift.velocity(geometry.points, geometry.normals)
        # The shift's measure source is its own graph volume rate: the content
        # rate of a unit concentration under the relative velocity, so the
        # measure and the relative content flux share one discrete divergence.
        volume = self.shift.rate(
            jnp.ones_like(geometry.measures),
            geometry.points,
            motion.relative_velocity - velocity,
        ).content_rate
        return motion.with_mesh_shift(velocity, volume)

    def initialize(
        self,
        points: ArrayLike,
        concentration: ArrayLike,
        /,
        *,
        time: float = 0.0,
        args: Any = None,
    ) -> MovingSurfaceState:
        """Admit the initial geometry from the native provider (host boundary)."""
        x = np.asarray(points, dtype=np.float64)
        c = np.asarray(concentration, dtype=np.float64)
        count = self.capacity.active_count
        if x.ndim != 2 or x.shape[0] != count or c.shape != (count,):
            raise ValueError(
                "Initial geometry and field must occupy compact active coordinates."
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(c)):
            raise ValueError("Initial moving geometry and field must be finite.")
        if not np.isfinite(time):
            raise ValueError("Initial time must be finite.")
        start = jnp.asarray(time, dtype=jnp.float64)
        motion = self.motion
        if isinstance(motion, ChartMotion):
            authoritative = np.asarray(motion.positions(start))
            if authoritative.shape != x.shape or not np.allclose(
                authoritative, x, rtol=1e-12, atol=1e-12
            ):
                raise ValueError("Initial points differ from the authoritative chart.")
        geometry = self.geometry(jnp.asarray(x), start, args)
        if not isinstance(geometry, MovingGeometryRefresh):
            raise TypeError("Geometry callback must return MovingGeometryRefresh.")
        admitted = _stage_admission(
            geometry, self._stage_motion(start, geometry, args), self.shift
        )
        if not bool(
            np.asarray(
                admitted.trust_valid
                & admitted.tube_valid
                & admitted.geometry_valid
                & admitted.motion_valid
                & admitted.transport_valid
            )
        ):
            raise ValueError("Initial moving geometry or motion is not admitted.")
        w = geometry.measures
        m = w * jnp.asarray(c)
        h = self.history_capacity
        return MovingSurfaceState(
            geometry.points,
            w,
            geometry.normals,
            m,
            start,
            self.capacity,
            self.epoch,
            jnp.zeros((h, count)).at[0].set(m),
            jnp.broadcast_to(w, (h, count)),
            jnp.broadcast_to(geometry.points, (h, *x.shape)),
            jnp.full((h,), start),
            jnp.asarray(1, jnp.int32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )

    def _integrate(
        self, state: MovingSurfaceState, dt: Array, args: Any, /
    ) -> tuple[ConservationIMEXResult, Array]:
        count = state.content.shape[0]
        space = ArraySpace((count,), dtype=np.float64)
        parts = [
            state.content,
            state.measures,
            jnp.zeros((1,), dtype=state.content.dtype),
        ]
        if self.integrates_points:
            parts.append(state.points.reshape((-1,)))
        initial = jnp.concatenate(parts)

        def explicit_rhs(time: Array, packed: Array, context: Any, /) -> Array:
            geometry, motion = self._evaluate(time, packed, count, context)
            w = geometry.measures
            safe = jnp.where(w > 0, w, 1)
            reaction = w * jnp.asarray(
                self.reaction(time, geometry.points, packed[:count] / safe, context),
                dtype=jnp.float64,
            )
            content = reaction
            if self.shift is not None:
                # Material crossing the moving mesh: the conservative upwind
                # rate of the relative velocity at this stage's geometry.
                content = (
                    content
                    + self.shift.rate(
                        packed[:count] / safe, geometry.points, motion.relative_velocity
                    ).content_rate
                )
            rates = [content, motion.measure_rate, jnp.sum(reaction)[None]]
            if self.integrates_points:
                rates.append(motion.mesh_velocity.reshape((-1,)))
            return jnp.concatenate(rates)

        def implicit_rhs(time: Array, packed: Array, context: Any, /) -> Array:
            geometry, _ = self._evaluate(time, packed, count, context)
            w = geometry.measures
            rate = geometry.diffusion.mv(packed[:count] / jnp.where(w > 0, w, 1))
            return jnp.concatenate((rate, jnp.zeros_like(packed[count:])))

        def implicit_solver(
            provisional: Array, time: Array, coefficient: Array, context: Any, /
        ) -> ImplicitConservationStageResult:
            geometry, motion = self._evaluate(time, provisional, count, context)
            w = geometry.measures
            safe = jnp.where(w > 0, w, 1)
            operator = FunctionLinearOperator(
                lambda value: value - coefficient * geometry.diffusion.mv(value / safe),
                source=space,
                target=space,
                operator_id=self.implicit_id,
            )
            prepared = bind_numeric(
                self.linear_template, LinearSystem(operator, problem_id=self.implicit_id)
            )
            result = solve(prepared, provisional[:count])
            return ImplicitConservationStageResult(
                provisional.at[:count].set(result.value),
                jnp.all(result.successful),
                jnp.max(result.diagnostics.iterations).astype(jnp.int32),
                jnp.max(result.diagnostics.residual_norm),
                jnp.max(result.status).astype(jnp.int32),
                _stage_admission(geometry, motion, self.shift),
            )

        # Content m = w c is the conserved state: geometry dilution never enters
        # as a source, and the packed measure witness integrates w (H V_n +
        # div u_tau) with the same stage rule as the coordinates.
        method = ConservationIMEXMethod(
            self.tableau,
            explicit_rhs,
            implicit_rhs,
            implicit_solver,
            method_id=self.plan_id,
        )
        return method.step(state.time, initial, dt, args), initial

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
        count = state.content.shape[0]
        integration, initial = self._integrate(state, dt, args)
        candidate = integration.candidate_state
        end_time = state.time + dt
        end_geometry, refreshed = self._stage_geometry(end_time, candidate, count, args)
        end_motion = self._stage_motion(end_time, end_geometry, args)
        admissions = [
            record
            for record in integration.stage_evidence
            if isinstance(record, MovingStageAdmission)
        ]
        admissions.append(_stage_admission(end_geometry, end_motion, self.shift))
        if self.tableau.implicit_parts[0] is None:
            # The explicit-only first stage is the incoming state at node zero.
            admissions.append(
                _stage_admission(
                    *self._evaluate(state.time, initial, count, args), self.shift
                )
            )
        trust_valid = jnp.all(jnp.stack([item.trust_valid for item in admissions]))
        tube_valid = jnp.all(jnp.stack([item.tube_valid for item in admissions]))
        geometry_valid = jnp.all(jnp.stack([item.geometry_valid for item in admissions]))
        motion = self.motion
        if isinstance(motion, ChartMotion):
            # Authoritative positions own the coordinates; a state moved by a
            # separate shift or remap no longer lies on this chart.
            geometry_valid = geometry_valid & (
                jnp.max(jnp.abs(motion.positions(state.time) - state.points))
                <= 1e-10 * jnp.maximum(1, jnp.max(jnp.abs(state.points)))
            )
        motion_valid = jnp.all(jnp.stack([item.motion_valid for item in admissions]))
        transport_valid = jnp.all(
            jnp.stack([item.transport_valid for item in admissions])
        )
        column_residual = jnp.max(
            jnp.stack([item.diffusion_column_residual for item in admissions])
        )
        conservative = column_residual <= self.conservation_tolerance
        transport_cfl = jnp.abs(dt) * jnp.max(
            jnp.stack([item.transport_outflow_rate for item in admissions])
        )

        content = candidate[:count]
        witness = candidate[count : 2 * count]
        reaction_content = candidate[2 * count]
        w = end_geometry.measures
        safe_dt = jnp.where(dt != 0, jnp.abs(dt), 1)
        # Geometric law: measures of record are refreshed, the witness is the
        # stage-quadrature check. Conservative law: the witness is the record
        # (defect zero by construction) and the refreshed measures report the
        # accumulated geometric drift of the conservative measures.
        gcl_defect = jnp.max(jnp.abs(w - witness)) / (safe_dt * jnp.max(state.measures))
        geometric_drift = jnp.max(jnp.abs(refreshed - w)) / jnp.max(jnp.abs(w))
        match self.gcl.admission:
            case "stage-quadrature":
                gcl_admitted = gcl_defect <= self.gcl.rate_tolerance
            case "diagnostic":
                gcl_admitted = jnp.asarray(True)
            case _:
                assert_never(self.gcl.admission)
        rate_residual = jnp.max(
            jnp.abs((w - state.measures) - dt * end_motion.measure_rate)
        )
        before, after = jnp.sum(state.content), jnp.sum(content)
        conservation_residual = after - before - reaction_content
        scale = jnp.maximum(
            jnp.maximum(jnp.sum(jnp.abs(state.content)), jnp.sum(jnp.abs(content))),
            jnp.finfo(content.dtype).tiny,
        )
        content_conserved = (
            jnp.abs(conservation_residual) <= self.ledger_tolerance * scale
        )
        minimum_concentration = jnp.min(content / jnp.where(w > 0, w, 1))
        positivity_admitted = (
            minimum_concentration >= 0 if self.require_positivity else jnp.asarray(True)
        )

        h = self.history_capacity
        lifetime = state.accepted_steps + jnp.asarray(1, jnp.int32)
        slot = lifetime % h
        full = state.history_count >= h
        evicted = jnp.where(full, lifetime - h, -1).astype(jnp.int32)
        match self.archive:
            case "rolling":
                history_admitted = jnp.asarray(True)
            case "acknowledged":
                history_admitted = ~full | (evicted < state.archive_cursor)
            case _:
                assert_never(self.archive)
        valid_step = jnp.isfinite(dt) & (dt > 0)
        solver_successful = integration.successful
        # First refusal in causal precedence: a law refusing its own inputs
        # (for example an unsupported bulk query) explains any later geometry
        # defect at the points it would have produced.
        refusals = (
            (~valid_step, MovingSurfaceStatus.INVALID_STEP),
            (~history_admitted, MovingSurfaceStatus.HISTORY_EXHAUSTED),
            (~motion_valid, MovingSurfaceStatus.MOTION_REFUSED),
            (~trust_valid, MovingSurfaceStatus.TRUST_REFUSED),
            (~tube_valid, MovingSurfaceStatus.TUBE_REFUSED),
            (~geometry_valid, MovingSurfaceStatus.INVALID_GEOMETRY),
            (~transport_valid, MovingSurfaceStatus.TRANSPORT_REFUSED),
            (~conservative, MovingSurfaceStatus.NONCONSERVATIVE_DIFFUSION),
            (~solver_successful, MovingSurfaceStatus.IMPLICIT_REFUSED),
            (~content_conserved, MovingSurfaceStatus.CONSERVATION_DEFECT),
            (~gcl_admitted, MovingSurfaceStatus.GCL_REFUSED),
            (~positivity_admitted, MovingSurfaceStatus.POSITIVITY_REFUSED),
        )
        accepted = ~jnp.any(jnp.stack([refused for refused, _ in refusals]))
        status = jnp.select(
            [refused for refused, _ in refusals],
            [jnp.asarray(int(code), jnp.int32) for _, code in refusals],
            jnp.asarray(int(MovingSurfaceStatus.ACCEPTED), jnp.int32),
        )
        proposed = MovingSurfaceState(
            end_geometry.points,
            w,
            end_geometry.normals,
            content,
            end_time,
            state.capacity,
            state.epoch,
            state.history_content.at[slot].set(content),
            state.history_measures.at[slot].set(w),
            state.history_points.at[slot].set(end_geometry.points),
            state.history_times.at[slot].set(end_time),
            jnp.minimum(state.history_count + 1, h).astype(jnp.int32),
            lifetime,
            state.archive_cursor,
        )
        evidence = MovingSurfaceEvidence(
            status,
            accepted,
            trust_valid,
            tube_valid,
            geometry_valid,
            motion_valid,
            transport_valid,
            positivity_admitted,
            conservative,
            content_conserved,
            gcl_admitted,
            solver_successful,
            history_admitted,
            evicted,
            gcl_defect,
            geometric_drift,
            rate_residual,
            column_residual,
            transport_cfl,
            minimum_concentration,
            before,
            after,
            reaction_content,
            conservation_residual,
            integration.maximum_implicit_residual,
            integration.implicit_iterations.astype(jnp.int32),
            integration.stage_successful,
            integration.stage_iterations.astype(jnp.int32),
            integration.stage_status.astype(jnp.int32),
        )
        committed = commit_candidate(
            TransactionalCandidate(state, proposed, evidence, accepted, self.plan_id)
        )
        return MovingSurfaceStepResult(committed.state, proposed, evidence)

    def acknowledge_archive(
        self, state: MovingSurfaceState, through: int, /
    ) -> MovingSurfaceState:
        """Host acknowledgment that every lifetime index below ``through`` is archived."""
        if state.epoch.epoch_id != self.epoch.epoch_id:
            raise ValueError("Archive cursor belongs to a different moving epoch.")
        cursor = int(np.asarray(state.archive_cursor))
        newest = int(np.asarray(state.accepted_steps))
        if (
            isinstance(through, bool)
            or not isinstance(through, (int, np.integer))
            or not cursor <= through <= newest + 1
        ):
            raise ValueError(
                "Archive cursor must advance monotonically within accepted lifetimes."
            )
        return eqx.tree_at(
            lambda item: item.archive_cursor, state, jnp.asarray(through, jnp.int32)
        )

    def checkpoint(self, state: MovingSurfaceState, /) -> MovingSurfaceCheckpoint:
        if state.epoch.epoch_id != self.epoch.epoch_id:
            raise ValueError("Cannot checkpoint a different moving topology epoch.")
        return MovingSurfaceCheckpoint(state, self.plan_id)

    def rollback(self, checkpoint: MovingSurfaceCheckpoint, /) -> MovingSurfaceState:
        if checkpoint.plan_id != self.plan_id:
            raise ValueError("Checkpoint belongs to a different moving plan.")
        return checkpoint.state

    @checked
    def transition_epoch(
        self,
        state: MovingSurfaceState,
        transfer: PreparedPointTransfer,
        target_epoch: TopologyEpoch,
        target_points: ArrayLike,
        target_normals: ArrayLike,
        target_capacity: MeshfreeCapacityMap,
        history_transfers: Sequence[PreparedPointTransfer],
        target_history_points: ArrayLike,
        /,
    ) -> MovingSurfaceEpochResult:
        """Host atomic transition of the current field and EVERY live history.

        ``history_transfers`` and ``target_history_points`` follow
        ``state.live_slots()`` (oldest first). Each history remap has its own
        old/new measures; applying the current concentration map to
        historical extensive arrays would be incorrect. The current field and
        every live history cross the epoch through the native
        ``remap_live_histories`` owner at one host boundary. Ring slots,
        lifetime indices and the archive cursor are preserved. A failed route
        returns the original state without a partial cutover; refused
        (unadmitted) transfers are rejected before any remap. Values
        differentiate through the frozen transfers; geometry, measures and
        the epoch selection carry no derivative.
        """
        if state.epoch.epoch_id != self.epoch.epoch_id:
            raise ValueError(
                "Transition source belongs to a different moving topology epoch."
            )
        slots = state.live_slots()
        count = slots.size
        if len(history_transfers) != count:
            raise ValueError(
                "Every live history requires its own concentration transfer."
            )
        routes = (transfer, *history_transfers)
        if not all(isinstance(route, PreparedPointTransfer) for route in routes):
            raise TypeError(
                "Epoch transfers must be native PreparedPointTransfer routes."
            )
        if not all(route.admitted for route in routes):
            raise ValueError("Every epoch transfer route must be admitted.")
        if not np.array_equal(
            np.asarray(transfer.source_measures), np.asarray(state.measures)
        ):
            raise ValueError("Current transfer measures do not match current geometry.")
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
        for slot, route in zip(slots, history_transfers, strict=True):
            if not np.array_equal(
                np.asarray(route.source_measures),
                np.asarray(state.history_measures[slot]),
            ) or route.target_measures.shape != (n,):
                raise ValueError(
                    "Historical transfer measures/layout do not match its saved geometry."
                )
        remap = remap_live_histories(
            [route.epoch_transition(state.epoch, target_epoch) for route in routes],
            [
                state.concentration,
                *(
                    state.history_content[slot] / state.history_measures[slot]
                    for slot in slots
                ),
            ],
        )
        if not remap.successful:
            return MovingSurfaceEpochResult(state, False, remap.conservation_residuals)
        current, *histories = remap.values
        h = self.history_capacity
        live = jnp.asarray(slots)
        hc = (
            jnp.zeros((h, n))
            .at[live]
            .set(
                jnp.stack(
                    [
                        values * route.target_measures
                        for values, route in zip(
                            histories, history_transfers, strict=True
                        )
                    ]
                )
            )
        )
        stop = jax.lax.stop_gradient
        hm = stop(
            jnp.broadcast_to(transfer.target_measures, (h, n))
            .at[live]
            .set(jnp.stack([route.target_measures for route in history_transfers]))
        )
        hp = (
            jnp.broadcast_to(jnp.asarray(points), (h, *points.shape))
            .at[live]
            .set(jnp.asarray(history_points))
        )
        target = MovingSurfaceState(
            jnp.asarray(points),
            stop(transfer.target_measures),
            jnp.asarray(normals),
            current * transfer.target_measures,
            state.time,
            target_capacity,
            target_epoch,
            hc,
            hm,
            hp,
            state.history_times,
            state.history_count,
            state.accepted_steps,
            state.archive_cursor,
        )
        return MovingSurfaceEpochResult(target, True, remap.conservation_residuals)


__all__ = [
    "MovingArchiveMode",
    "MovingGCLAdmission",
    "MovingGCLPolicy",
    "MovingMeasureLaw",
    "MovingSurfaceStatus",
    "MovingSurfaceState",
    "MovingStageAdmission",
    "MovingSurfaceEvidence",
    "MovingSurfaceStepResult",
    "MovingSurfaceCheckpoint",
    "MovingSurfaceEpochResult",
    "MovingSurfacePlan",
]
