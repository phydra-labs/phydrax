#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prescribed moving charges driving the compatible Maxwell runtime.

A `PrescribedChargeTrajectory` samples the positions of ``P`` point charges on
the Maxwell step grid ``t_n = t_0 + n Δt``. Every step deposits the
tail-to-head charge-conserving current of the path ``r_n → r_{n+1}`` with a
`~phydrax.discretization.pic.ChargeConservingCurrentPlan` and advances the
prepared compatible Maxwell runtime with it, so the discrete continuity
``(ρ_{n+1} − ρ_n)/Δt − δJ = 0`` makes the Maxwell charge cochain follow the
deposited charge exactly.

*Coincident-neutral start.* Each charge is created at ``t_0`` together with a
static compensating charge of opposite sign at the same point, so the run
starts from zero fields and zero charge and satisfies Gauss's law without an
electrostatic solve. The compensator never moves and carries no current; its
charge is ``−ρ(r_0)``, and the prescribed charge cochain is
``ρ(r_n) − ρ(r_0)``.

*Freeze stop.* After the last trajectory sample, and after a charge leaves the
closed box through a nonperiodic face (``boundary_exit`` of the deposit), the
charge stays frozen at its last deposited position: its current vanishes and
its charge is preserved.

Any linear prepared Maxwell runtime is accepted: dispersive, lossy,
heterogeneous, magnetized-plasma, and negative-index media, CPML absorbers, and
PEC/PMC/impedance boundaries including interior conductor supports. The run is
one ``lax.scan`` that reports per-step continuity, Gauss, and magnetic defects
and the power ledger ``ΔW = −∫ E·J dV dt − ∫ P_loss dt`` (field plus material
energy ``W``; boundary, CPML, and material losses ``P_loss``).
"""

from __future__ import annotations

from enum import IntFlag
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import AbstractCellDeRhamComplex, StructuredCochainBridge
from ..discretization.pic._current import (
    ChargeConservingCurrentPlan,
    PICMaxwellCurrentArguments,
)
from ..discretization.pic._types import PICCurrentDepositResult
from ..typing import Bool, checked, Dim, Float64, Int32, parse, Scalar, Scope, Size
from ._maxwell import (
    _fixed_step,
    _PreparedMaxwellFixedStep,
    CompatibleMaxwellState,
    MaxwellCochainLayout,
    PreparedCompatibleMaxwell,
)
from ._maxwell_sources import AbstractMaxwellSourcePlan, MaxwellSourceForcing


class _SampleDim(Dim, minimum=2):
    """Trajectory samples on the Maxwell step grid."""


class _ChargeDim(Dim, minimum=1):
    """Prescribed point charges."""


class _StepDim(Dim, minimum=1):
    """Executed Maxwell steps."""


class _EnergyDim(Dim, minimum=2):
    """Step-end energy samples, one more than the executed steps."""


class _ElectricDim(Dim, minimum=1):
    """Retained electric cochain entities."""


class _VertexDim(Dim, minimum=1):
    """Charge (vertex) cochain entities."""


# Uniform sampling of the trajectory times, relative to the step size.
_TIME_GRID_TOLERANCE = 1.0e-10


class PrescribedChargeTrajectory(StrictModule):
    """Positions of prescribed point charges on a uniform step grid.

    ``times`` are the Maxwell step times ``t_0 + n Δt`` (uniform to a relative
    ``1e-10`` of ``Δt``); ``positions[n, p]`` is charge ``p`` at ``times[n]``.
    Steps beyond the last sample freeze every charge at its last position.
    """

    __strict_contract__ = True

    times: Float64[_SampleDim]
    positions: Float64[_SampleDim, _ChargeDim, Literal[3]]
    sample_count: Size[_SampleDim] = eqx.field(static=True)
    charge_count: Size[_ChargeDim] = eqx.field(static=True)
    start_time: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)

    def __init__(self, times: ArrayLike, positions: ArrayLike, /) -> None:
        host_times = np.asarray(times, dtype=np.float64)
        host_positions = np.asarray(positions, dtype=np.float64)
        if host_times.ndim != 1 or host_times.shape[0] < 2:
            raise ValueError("times must be a vector of at least two samples.")
        if (
            host_positions.ndim != 3
            or host_positions.shape[0] != host_times.shape[0]
            or host_positions.shape[2] != 3
        ):
            raise ValueError("positions must have shape (samples, charges, 3).")
        if not np.all(np.isfinite(host_times)) or not np.all(np.isfinite(host_positions)):
            raise ValueError("Trajectory times and positions must be finite.")
        increments = np.diff(host_times)
        step = float(increments[0])
        if not step > 0.0:
            raise ValueError("Trajectory times must increase.")
        expected = host_times[0] + step * np.arange(host_times.shape[0])
        if np.max(np.abs(host_times - expected)) > _TIME_GRID_TOLERANCE * step:
            raise ValueError("Trajectory times must lie on one uniform step grid.")
        scope = Scope()
        self.sample_count = parse(
            host_times.shape[0], Size[_SampleDim], "sample_count", scope=scope
        )
        self.charge_count = parse(
            host_positions.shape[1], Size[_ChargeDim], "charge_count", scope=scope
        )
        self.times = parse(
            jnp.asarray(host_times), Float64[_SampleDim], "times", scope=scope
        )
        self.positions = parse(
            jnp.asarray(host_positions),
            Float64[_SampleDim, _ChargeDim, Literal[3]],
            "positions",
            scope=scope,
        )
        self.start_time = float(host_times[0])
        self.step_size = step
        self.trajectory_id = canonical_fingerprint(
            {
                "kind": "prescribed-charge-trajectory",
                "times": array_tree_fingerprint(host_times),
                "positions": array_tree_fingerprint(host_positions),
            }
        )

    def targets(self, step_count: int, /) -> Array:
        """Path heads ``r_{min(n+1, T−1)}`` of steps ``n = 0, …, step_count − 1``."""
        index = np.minimum(np.arange(1, step_count + 1), self.sample_count - 1)
        return self.positions[jnp.asarray(index)]


def _default_step_count(
    trajectory: PrescribedChargeTrajectory, step_count: int | None, /
) -> int:
    if step_count is None:
        return trajectory.sample_count - 1
    count = int(step_count)
    if count < 1:
        raise ValueError("step_count must be positive.")
    return count


def _require_compatible(
    current_plan: ChargeConservingCurrentPlan,
    trajectory: PrescribedChargeTrajectory,
    /,
) -> None:
    if not isinstance(current_plan, ChargeConservingCurrentPlan):
        raise TypeError("current_plan must be a ChargeConservingCurrentPlan.")
    if not isinstance(trajectory, PrescribedChargeTrajectory):
        raise TypeError("trajectory must be a PrescribedChargeTrajectory.")
    if current_plan.transfer.species.capacity != trajectory.charge_count:
        raise ValueError(
            "The current plan's particle capacity must equal the prescribed charge count."
        )
    axes = current_plan.transfer.bridge.grid.structured_axes
    lower = np.asarray([float(axis.bounds[0]) for axis in axes])
    upper = np.asarray([float(axis.bounds[1]) for axis in axes])
    bounded = np.asarray([not axis.periodic for axis in axes])
    start = np.asarray(trajectory.positions[0])
    outside = np.logical_and(bounded, np.logical_or(start < lower, start > upper))
    if np.any(outside):
        raise ValueError("Prescribed charges must start inside the closed domain box.")


def _deposit_step(
    current_plan: ChargeConservingCurrentPlan,
    charge: Array,
    step_size: Array,
    position: Array,
    exited: Array,
    target: Array,
    /,
) -> PICCurrentDepositResult:
    """Deposit ``position → target``; exited charges stay frozen."""
    end = jnp.where(exited[:, None], position, target)
    return current_plan.deposit(position, end, step_size, macrocharge=charge)


def _current_support(
    current_plan: ChargeConservingCurrentPlan,
    trajectory: PrescribedChargeTrajectory,
    step_count: int,
    /,
) -> Array:
    """Electric entities any step of the prescribed paths can drive."""
    charge = jnp.ones((trajectory.charge_count,), dtype=jnp.float64)
    step = jnp.asarray(trajectory.step_size, dtype=jnp.float64)
    count = current_plan.transfer.bridge.cochain.cell_counts[1]

    def body(
        carry: tuple[Array, Array, Array], target: Array, /
    ) -> tuple[tuple[Array, Array, Array], None]:
        position, exited, support = carry
        deposit = _deposit_step(current_plan, charge, step, position, exited, target)
        return (
            deposit.deposited_end,
            exited | deposit.boundary_exit,
            support | (deposit.current != 0.0),
        ), None

    initial = (
        trajectory.positions[0],
        jnp.zeros((trajectory.charge_count,), dtype=jnp.bool_),
        jnp.zeros((count,), dtype=jnp.bool_),
    )
    (_, _, support), _ = jax.lax.scan(body, initial, trajectory.targets(step_count))
    return support


class PreparedPrescribedChargeCurrentSource(StrictModule, NonTrainableState):
    """Dynamic prescribed-charge current with a statically certified support.

    ``electric_support`` marks every electric entity the prescribed paths can
    drive over the run; the current is zero elsewhere by construction, which
    Huygens surfaces use to certify ``J = 0`` on their entities.
    """

    __strict_contract__ = True

    electric_support: Bool[_ElectricDim]
    electric_count: Size[_ElectricDim] = eqx.field(static=True)
    magnetic_count: int = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    current_plan_id: str = eqx.field(static=True)
    bridge_id: str = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        electric_support: Array,
        layout: MaxwellCochainLayout,
        /,
        *,
        trajectory_id: str,
        current_plan_id: str,
        bridge_id: str,
        step_count: int,
        source_id: str,
    ) -> None:
        scope = Scope()
        self.electric_count = parse(
            layout.electric_count, Size[_ElectricDim], "electric_count", scope=scope
        )
        self.electric_support = parse(
            electric_support, Bool[_ElectricDim], "electric_support", scope=scope
        )
        self.magnetic_count = layout.magnetic_count
        self.trajectory_id = trajectory_id
        self.current_plan_id = current_plan_id
        self.bridge_id = bridge_id
        self.step_count = step_count
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-prescribed-charge-current-source",
                "source": source_id,
                "layout": layout.layout_id,
                "support": array_tree_fingerprint(np.asarray(electric_support)),
            }
        )

    def sample(self, time: ArrayLike, args: object = None, /) -> MaxwellSourceForcing:
        del time
        if not isinstance(args, PICMaxwellCurrentArguments):
            raise TypeError(
                "The prescribed-charge source requires PICMaxwellCurrentArguments."
            )
        current = jnp.asarray(args.particle_current)
        if current.shape != (self.electric_count,):
            raise ValueError("Prescribed current must match retained electric cochains.")
        return MaxwellSourceForcing(
            current, jnp.zeros((self.magnetic_count,), dtype=current.dtype)
        )

    def validate_runtime(self, prepared: PreparedCompatibleMaxwell, /) -> None:
        bridge = prepared.plan.bridge
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError(
                "Prescribed-charge sources require a structured cochain bridge."
            )
        if bridge.bridge_id != self.bridge_id:
            raise ValueError(
                "The prescribed-charge source was prepared on a different bridge."
            )
        if prepared.layout.polarization != "full_3d":
            raise ValueError("Prescribed charges require a full_3d Maxwell layout.")


class PrescribedChargeCurrentSourcePlan(AbstractMaxwellSourcePlan, NonTrainableState):
    """Maxwell source slot for the current of one prescribed trajectory.

    Preparation replays the deposited paths once to certify the electric
    support; `PrescribedChargeMaxwellPlan` supplies the per-step current.
    """

    trajectory: PrescribedChargeTrajectory
    current_plan: ChargeConservingCurrentPlan
    step_count: int = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        trajectory: PrescribedChargeTrajectory,
        current_plan: ChargeConservingCurrentPlan,
        /,
        *,
        step_count: int | None = None,
    ) -> None:
        _require_compatible(current_plan, trajectory)
        count = _default_step_count(trajectory, step_count)
        self.trajectory = trajectory
        self.current_plan = current_plan
        self.step_count = count
        self.source_id = canonical_fingerprint(
            {
                "kind": "prescribed-charge-current-source-plan",
                "trajectory": trajectory.trajectory_id,
                "current_plan": current_plan.plan_id,
                "step_count": count,
            }
        )

    def prepare(
        self, bridge: AbstractCellDeRhamComplex, layout: MaxwellCochainLayout, /
    ) -> PreparedPrescribedChargeCurrentSource:
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError(
                "Prescribed-charge sources require a structured cochain bridge."
            )
        if bridge.bridge_id != self.current_plan.transfer.bridge.bridge_id:
            raise ValueError("The current plan was prepared on a different bridge.")
        support = _current_support(self.current_plan, self.trajectory, self.step_count)
        return PreparedPrescribedChargeCurrentSource(
            support,
            layout,
            trajectory_id=self.trajectory.trajectory_id,
            current_plan_id=self.current_plan.plan_id,
            bridge_id=bridge.bridge_id,
            step_count=self.step_count,
            source_id=self.source_id,
        )


class PrescribedChargeStatus(IntFlag):
    """Failure and event flags of one prescribed-charge run.

    ``BOUNDARY_EXIT`` is informational: a charge left through a nonperiodic
    face and was frozen at its exit point.
    """

    SUCCESS = 0
    DEPOSIT_FAILED = 1
    CONTINUITY_DEFECT = 2
    GAUSS_DEFECT = 4
    MAGNETIC_DEFECT = 8
    SUPPORT_LEAK = 16
    LEDGER_OPEN = 32
    NONFINITE = 64
    BOUNDARY_EXIT = 128


_FAILURES = (
    PrescribedChargeStatus.DEPOSIT_FAILED
    | PrescribedChargeStatus.CONTINUITY_DEFECT
    | PrescribedChargeStatus.GAUSS_DEFECT
    | PrescribedChargeStatus.MAGNETIC_DEFECT
    | PrescribedChargeStatus.SUPPORT_LEAK
    | PrescribedChargeStatus.LEDGER_OPEN
    | PrescribedChargeStatus.NONFINITE
)


class PrescribedChargeEvidence(StrictModule):
    """Constraint, ledger, support, and exit evidence of one run.

    ``maximum_continuity_defect`` is the largest deposited
    ``|(ρ_{n+1} − ρ_n)/Δt − δJ|`` and ``maximum_constraint_defect`` the largest
    runtime Gauss defect ``|−δD − ρ_Maxwell|``. ``maximum_gauss_defect`` is
    Gauss's law against the *prescribed* charge, ``|−δD − (ρ(r_n) − ρ(r_0))|``,
    on the ``free_vertex_count`` vertices no boundary or CPML entity touches; in
    conducting media it is the induced conduction charge rather than a defect,
    so it carries no status. ``maximum_magnetic_defect`` is the largest
    ``|d(B) − declared magnetic charge|`` and ``maximum_support_leak`` the
    largest current off the certified source support. The ledger defect is
    ``W_N − W_0 − Σ work + Σ losses`` with trapezoidal step losses; its relative
    value is taken against ``Σ |work|``. ``exit_step`` is the step at which a
    charge left the box (``−1`` if never).
    """

    __strict_contract__ = True

    status: Int32[Scalar]
    successful: Bool[Scalar]
    deposit_successful: Bool[Scalar]
    finite: Bool[Scalar]
    maximum_continuity_defect: Float64[Scalar]
    maximum_constraint_defect: Float64[Scalar]
    maximum_gauss_defect: Float64[Scalar]
    maximum_magnetic_defect: Float64[Scalar]
    maximum_support_leak: Float64[Scalar]
    ledger_defect: Float64[Scalar]
    relative_ledger_defect: Float64[Scalar]
    exited: Bool[_ChargeDim]
    exit_step: Int32[_ChargeDim]
    free_vertex_count: int = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)


class PrescribedChargeMaxwellResult(StrictModule):
    """Final state, acquisitions, per-step ledger, and evidence of one run.

    ``field_energy[n]`` is the leapfrog energy at ``t_n``: field plus material
    energy minus ``(Δt²/8)⟨dE_n, ⋆ ∂H/∂B dE_n⟩``, which equals the staggered
    ``½⟨E_n, ⋆D_n⟩ + ½⟨H_{n+1/2}, ⋆B_{n−1/2}⟩`` of linear magnetic media, the
    energy the leapfrog update exchanges exactly with ``−∫ E·J``;
    ``source_work[n]`` the work ``−Δt ⟨(E_n + E_{n+1})/2, ⋆J_{n+1/2}⟩`` the
    prescribed current does on the field in step ``n``; ``dissipated_energy[n]``
    the trapezoidal boundary, CPML, and material loss of that step.
    ``prescribed_charge`` is the final prescribed charge cochain
    ``ρ(r_N) − ρ(r_0)`` and ``final_positions`` the final deposited positions.
    """

    __strict_contract__ = True

    final_state: CompatibleMaxwellState
    observations: tuple[Array, ...]
    final_positions: Float64[_ChargeDim, Literal[3]]
    prescribed_charge: Float64[_VertexDim]
    field_energy: Float64[_EnergyDim]
    source_work: Float64[_StepDim]
    dissipated_energy: Float64[_StepDim]
    evidence: PrescribedChargeEvidence


def _free_vertices(prepared: PreparedCompatibleMaxwell, /) -> np.ndarray:
    """Vertices whose incident edges carry no boundary or CPML action."""
    bridge = prepared.plan.bridge
    if not isinstance(bridge, StructuredCochainBridge):
        raise TypeError(
            "Prescribed-charge vertex support requires a structured cochain bridge."
        )
    affected = np.zeros((prepared.layout.electric_count,), dtype=np.bool_)
    for boundary in prepared.boundaries:
        if boundary.kind != "pmc":
            affected |= np.asarray(boundary.electric_boundary)
    if prepared.pml is not None:
        for term in prepared.pml.electric_terms:
            affected[np.asarray(term.indices)] = True
    vertex_shape = bridge.orientation_shapes[0][0]
    vertices = np.zeros(vertex_shape, dtype=np.bool_)
    axes = bridge.grid.structured_axes
    for axis, edges in enumerate(bridge.unpack(1, jnp.asarray(affected))):
        marked = np.asarray(edges)
        tail = [slice(0, extent) for extent in marked.shape]
        vertices[tuple(tail)] |= marked
        head = np.roll(marked, 1, axis=axis) if axes[axis].periodic else marked
        if axes[axis].periodic:
            vertices |= head
        else:
            shifted = [slice(0, extent) for extent in marked.shape]
            shifted[axis] = slice(1, marked.shape[axis] + 1)
            vertices[tuple(shifted)] |= head
    return ~vertices.reshape((-1,))


class _StepRecord(StrictModule):
    energy: Array
    work: Array
    dissipated: Array
    continuity: Array
    constraint: Array
    gauss: Array
    magnetic: Array
    support_leak: Array
    deposit_successful: Array
    finite: Array
    exits: Array


class _Carry(StrictModule):
    state: CompatibleMaxwellState
    position: Array
    exited: Array
    electric: Array
    loss: Array


class PrescribedChargeMaxwellPlan(StrictModule):
    """Prescribed point charges driving one prepared compatible Maxwell runtime.

    ``prepared_maxwell`` must carry the `PrescribedChargeCurrentSourcePlan` of
    the same ``trajectory``, ``current_plan``, and ``step_count``; the step size
    is the trajectory's and must not exceed the runtime's stable step.
    ``charge`` holds the physical charge of each prescribed particle.
    ``defect_tolerance`` bounds the relative continuity, Gauss-constraint,
    magnetic, and support defects; ``ledger_tolerance`` bounds the relative
    power-ledger defect (an ``O((ωΔt)²)`` quantity of the step-end energy).
    """

    __strict_contract__ = True

    prepared_maxwell: PreparedCompatibleMaxwell
    current_plan: ChargeConservingCurrentPlan
    trajectory: PrescribedChargeTrajectory
    charge: Float64[_ChargeDim]
    neutral_charge: Float64[_VertexDim]
    free_vertices: Bool[_VertexDim]
    electric_support: Array
    fixed_step: _PreparedMaxwellFixedStep
    step_count: int = eqx.field(static=True)
    charge_scale: float = eqx.field(static=True)
    free_vertex_count: int = eqx.field(static=True)
    defect_tolerance: float = eqx.field(static=True)
    ledger_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        prepared_maxwell: PreparedCompatibleMaxwell,
        current_plan: ChargeConservingCurrentPlan,
        trajectory: PrescribedChargeTrajectory,
        charge: ArrayLike,
        /,
        *,
        step_count: int | None = None,
        defect_tolerance: float = 1.0e-9,
        ledger_tolerance: float = 1.0e-2,
    ) -> None:
        _require_compatible(current_plan, trajectory)
        if prepared_maxwell.layout.polarization != "full_3d":
            raise ValueError("Prescribed charges require a full_3d Maxwell layout.")
        bridge = prepared_maxwell.plan.bridge
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("Prescribed-charge runs require a structured cochain bridge.")
        if current_plan.transfer.bridge.bridge_id != bridge.bridge_id:
            raise ValueError(
                "The current plan and Maxwell runtime use different bridges."
            )
        count = _default_step_count(trajectory, step_count)
        sources = tuple(
            source
            for source in prepared_maxwell.sources
            if isinstance(source, PreparedPrescribedChargeCurrentSource)
        )
        if len(sources) != 1:
            raise ValueError(
                "The Maxwell runtime must carry exactly one "
                "PrescribedChargeCurrentSourcePlan."
            )
        source = sources[0]
        if (
            source.trajectory_id != trajectory.trajectory_id
            or source.current_plan_id != current_plan.plan_id
            or source.step_count != count
        ):
            raise ValueError(
                "The Maxwell prescribed-charge source belongs to another trajectory, "
                "current plan, or step count."
            )
        if trajectory.step_size > float(np.asarray(prepared_maxwell.stable_dt)):
            raise ValueError(
                "The trajectory step size exceeds the Maxwell stable step; the "
                "trajectory must be sampled on the Maxwell step grid."
            )
        host_charge = np.asarray(charge, dtype=np.float64)
        if host_charge.shape != (trajectory.charge_count,) or not np.all(
            np.isfinite(host_charge)
        ):
            raise ValueError("charge must hold one finite value per prescribed charge.")
        tolerances = (float(defect_tolerance), float(ledger_tolerance))
        if any(not np.isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Defect and ledger tolerances must be positive and finite.")
        transfer = current_plan.transfer
        neutral = transfer.deposit_macrocharge(
            transfer.build(trajectory.positions[0]), jnp.asarray(host_charge)
        )
        if not bool(neutral.successful):
            raise ValueError("The initial prescribed charge could not be deposited.")
        magnitude = transfer.deposit_macrocharge(
            transfer.build(trajectory.positions[0]), jnp.asarray(np.abs(host_charge))
        )
        free = _free_vertices(prepared_maxwell)
        scope = Scope()
        self.prepared_maxwell = prepared_maxwell
        self.current_plan = current_plan
        self.trajectory = trajectory
        self.charge = parse(
            jnp.asarray(host_charge), Float64[_ChargeDim], "charge", scope=scope
        )
        self.neutral_charge = parse(
            neutral.cochain.astype(jnp.float64),
            Float64[_VertexDim],
            "neutral_charge",
            scope=scope,
        )
        self.free_vertices = parse(
            jnp.asarray(free), Bool[_VertexDim], "free_vertices", scope=scope
        )
        self.electric_support = source.electric_support
        self.fixed_step = _fixed_step(
            prepared_maxwell, jnp.asarray(trajectory.step_size), np.float64
        )
        # Unsigned charge density scale: coincident opposite charges cancel in
        # the neutral cochain but not in the roundoff floor of the defects.
        self.charge_scale = max(
            float(np.max(np.abs(np.asarray(magnitude.cochain)))),
            float(np.finfo(np.float64).tiny),
        )
        self.step_count = count
        self.free_vertex_count = int(np.sum(free))
        self.defect_tolerance, self.ledger_tolerance = tolerances
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prescribed-charge-maxwell-plan",
                "maxwell": prepared_maxwell.prepared_id,
                "current_plan": current_plan.plan_id,
                "trajectory": trajectory.trajectory_id,
                "charge": array_tree_fingerprint(host_charge),
                "step_count": count,
                "defect_tolerance": tolerances[0],
                "ledger_tolerance": tolerances[1],
            }
        )

    def initialize(self, /) -> CompatibleMaxwellState:
        """Coincident-neutral start: zero fields, zero charge, fresh material."""
        return self.prepared_maxwell.initialize()

    def _step(
        self, carry: _Carry, inputs: tuple[Array, Array], /
    ) -> tuple[_Carry, _StepRecord]:
        index, target = inputs
        maxwell = self.prepared_maxwell
        bridge = self.current_plan.transfer.bridge
        dt = self.fixed_step.parameters.step_size
        time = self.trajectory.start_time + index * dt
        deposit = _deposit_step(
            self.current_plan, self.charge, dt, carry.position, carry.exited, target
        )
        current = deposit.current.astype(jnp.float64)
        state = self.fixed_step.step(
            time, carry.state, PICMaxwellCurrentArguments(current, None)
        )
        electric = maxwell.electric_field(state)
        loss = maxwell.loss_power(state).astype(jnp.float64)
        work = -dt * jnp.real(
            jnp.vdot(
                0.5 * (carry.electric + electric),
                bridge.cochain.hodge_star(maxwell.layout.electric_degree, current),
            )
        )
        charge_scale = self.charge_scale
        constraint = maxwell.electric_constraint(state)
        divergence = constraint + state.primary.charge
        prescribed = deposit.end_charge.cochain - self.neutral_charge
        gauss = jnp.where(self.free_vertices, divergence - prescribed, 0.0)
        magnetic = maxwell.magnetic_constraint(state)
        flux_scale = jnp.maximum(
            jnp.max(jnp.abs(state.primary.magnetic_flux)), jnp.finfo(jnp.float64).tiny
        )
        record = _StepRecord(
            energy=maxwell.leapfrog_energy(state, dt).astype(jnp.float64),
            work=work.astype(jnp.float64),
            dissipated=(0.5 * dt * (carry.loss + loss)).astype(jnp.float64),
            continuity=deposit.maximum_continuity_defect * dt / charge_scale,
            constraint=jnp.max(jnp.abs(constraint), initial=0.0) / charge_scale,
            gauss=jnp.max(jnp.abs(gauss), initial=0.0),
            magnetic=jnp.max(jnp.abs(magnetic), initial=0.0) / flux_scale,
            support_leak=jnp.max(
                jnp.abs(jnp.where(self.electric_support, 0.0, current)), initial=0.0
            ),
            deposit_successful=deposit.successful,
            finite=jnp.all(jnp.isfinite(electric)) & jnp.isfinite(work),
            exits=deposit.boundary_exit,
        )
        return (
            _Carry(
                state,
                deposit.deposited_end,
                carry.exited | deposit.boundary_exit,
                electric,
                loss,
            ),
            record,
        )

    def solve(self, /) -> PrescribedChargeMaxwellResult:
        """Run the prescribed charges over ``step_count`` steps in one scan."""
        maxwell = self.prepared_maxwell
        state = self.initialize()
        charge_count = self.trajectory.charge_count
        initial = _Carry(
            state,
            self.trajectory.positions[0],
            jnp.zeros((charge_count,), dtype=jnp.bool_),
            maxwell.electric_field(state),
            maxwell.loss_power(state).astype(jnp.float64),
        )
        steps = jnp.arange(self.step_count, dtype=jnp.int32)

        def body(
            carry: _Carry, inputs: tuple[Array, Array], /
        ) -> tuple[_Carry, _StepRecord]:
            return self._step(carry, inputs)

        final, records = jax.lax.scan(
            body, initial, (steps, self.trajectory.targets(self.step_count))
        )
        energy = jnp.concatenate(
            (
                maxwell.leapfrog_energy(
                    state, self.fixed_step.parameters.step_size
                ).astype(jnp.float64)[None],
                records.energy,
            )
        )
        transfer = self.current_plan.transfer
        final_charge = (
            transfer.deposit_macrocharge(
                transfer.build(final.position), self.charge
            ).cochain.astype(jnp.float64)
            - self.neutral_charge
        )
        evidence = self._evidence(records, energy)
        return PrescribedChargeMaxwellResult(
            final.state,
            maxwell.observe(final.state),
            final.position,
            final_charge,
            energy,
            records.work,
            records.dissipated,
            evidence,
        )

    def _evidence(
        self, records: _StepRecord, energy: Array, /
    ) -> PrescribedChargeEvidence:
        work = jnp.sum(records.work)
        dissipated = jnp.sum(records.dissipated)
        ledger = energy[-1] - energy[0] - work + dissipated
        scale = jnp.maximum(jnp.sum(jnp.abs(records.work)), jnp.finfo(jnp.float64).tiny)
        relative = jnp.abs(ledger) / scale
        exited = jnp.any(records.exits, axis=0)
        exit_step = jnp.where(
            exited, jnp.argmax(records.exits, axis=0).astype(jnp.int32), -1
        )
        continuity = jnp.max(records.continuity)
        constraint = jnp.max(records.constraint)
        magnetic = jnp.max(records.magnetic)
        leak = jnp.max(records.support_leak)
        deposit_successful = jnp.all(records.deposit_successful)
        finite = jnp.all(records.finite) & jnp.all(jnp.isfinite(energy))
        flag = PrescribedChargeStatus
        status = (
            jnp.where(deposit_successful, 0, int(flag.DEPOSIT_FAILED))
            | jnp.where(
                continuity <= self.defect_tolerance, 0, int(flag.CONTINUITY_DEFECT)
            )
            | jnp.where(constraint <= self.defect_tolerance, 0, int(flag.GAUSS_DEFECT))
            | jnp.where(magnetic <= self.defect_tolerance, 0, int(flag.MAGNETIC_DEFECT))
            | jnp.where(leak == 0.0, 0, int(flag.SUPPORT_LEAK))
            | jnp.where(relative <= self.ledger_tolerance, 0, int(flag.LEDGER_OPEN))
            | jnp.where(finite, 0, int(flag.NONFINITE))
            | jnp.where(jnp.any(exited), int(flag.BOUNDARY_EXIT), 0)
        ).astype(jnp.int32)
        return PrescribedChargeEvidence(
            status=status,
            successful=(status & int(_FAILURES)) == 0,
            deposit_successful=deposit_successful,
            finite=finite,
            maximum_continuity_defect=continuity,
            maximum_constraint_defect=constraint,
            maximum_gauss_defect=jnp.max(records.gauss),
            maximum_magnetic_defect=magnetic,
            maximum_support_leak=leak,
            ledger_defect=ledger,
            relative_ledger_defect=relative,
            exited=exited,
            exit_step=exit_step,
            free_vertex_count=self.free_vertex_count,
            step_count=self.step_count,
            step_size=self.trajectory.step_size,
        )


def solve_prescribed_charge_maxwell(
    plan: PrescribedChargeMaxwellPlan, /
) -> PrescribedChargeMaxwellResult:
    """Compiled entry point with stable identity for `PrescribedChargeMaxwellPlan`."""
    if not isinstance(plan, PrescribedChargeMaxwellPlan):
        raise TypeError("plan must be a PrescribedChargeMaxwellPlan.")
    return _solve_compiled(plan)


@eqx.filter_jit
def _solve_compiled(
    plan: PrescribedChargeMaxwellPlan, /
) -> PrescribedChargeMaxwellResult:
    return plan.solve()


__all__ = [
    "PreparedPrescribedChargeCurrentSource",
    "PrescribedChargeCurrentSourcePlan",
    "PrescribedChargeEvidence",
    "PrescribedChargeMaxwellPlan",
    "PrescribedChargeMaxwellResult",
    "PrescribedChargeStatus",
    "PrescribedChargeTrajectory",
    "solve_prescribed_charge_maxwell",
]
