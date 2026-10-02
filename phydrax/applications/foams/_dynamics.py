#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Constrained finite-region pressure dynamics of explicit foams."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import parameter_field, ParameterOwner
from ..._validation import positive_finite_float, positive_integer
from ...bubble_dynamics import BubbleGasState
from ...geometry.multiregion_surface import (
    apply_surface_events,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    PreparedMultiRegionSurface,
    SurfaceEventPassEvidence,
    SurfaceEventPolicy,
    SurfaceEventProposal,
    validate_multiregion_surface,
)
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    DenseSVD,
    FailurePolicy,
    FunctionLinearOperator,
    JacobianLinearOperator,
    LinearSolvePolicy,
    LinearSolveStatus,
    MaterializationPolicy,
    MinimumNormProblem,
    prepare_linearization,
    RankPolicy,
    solve as solve_linear,
)
from ...solver import (
    ConstrainedMechanicalState,
    ConstrainedMechanicsEvidence,
    PreparedSHAKERATTLEPlan,
    SHAKERATTLEPlan,
)
from ...typing import checked, parse
from ._air import RegionPressureAirEvidence, RegionPressureAirPlan
from ._contracts import FoamMaterialPlan
from ._equilibrium import (
    _not_applicable_foam_volume_constraint_basis,
    _wire_coordinates,
    FoamVolumeConstraintBasis,
    prepare_foam_volume_constraint_basis,
)


FoamDynamicsRoute: TypeAlias = Literal["overdamped", "film-inertia"]


class FoamDynamicsStatus(IntEnum):
    """Outcome of one bounded foam dynamics advance."""

    COMPLETED = 0
    MECHANICS_FAILED = 1
    GAS_FAILED = 2
    GEOMETRY_FAILED = 3
    TOPOLOGY_EVENT_ROLLED_BACK = 4
    NONFINITE = 5
    CCD_FAILED = 6


@final
class FoamDynamicsPlan(StrictModule, ParameterOwner):
    """Time integration, physical coefficients, and safety bounds.

    ``friction`` is used by the overdamped energy-gradient route and
    ``areal_mass`` by the film-inertia SHAKE/RATTLE route. The time step is
    reduced by both an edge-displacement quality limit and a face-altitude
    swept-motion limit before any candidate is evaluated.
    On the incompressible air route, ``maximum_constraint_entries`` bounds both
    the preparation/runtime constraint Gram and its native minimum-norm solve;
    the volume Jacobian itself stays matrix-free. Incompressible preparation is
    admitted before derivative actions against
    ``maximum_constraint_rank_actions`` and
    ``maximum_constraint_preparation_bytes``. Compartment gas has no equality
    volume constraints and does not prepare a constraint basis.
    """

    route: FoamDynamicsRoute = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    friction: Array = parameter_field()
    areal_mass: Array = parameter_field()
    maximum_displacement_fraction: float = eqx.field(static=True)
    ccd_safety_fraction: float = eqx.field(static=True)
    minimum_face_area_fraction: float = eqx.field(static=True)
    ccd_candidate_capacity: int = eqx.field(static=True)
    projection_iterations: int = eqx.field(static=True)
    volume_tolerance: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    condition_limit: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    maximum_constraint_entries: int = eqx.field(static=True)
    maximum_constraint_rank_actions: int = eqx.field(static=True)
    maximum_constraint_preparation_bytes: int = eqx.field(static=True)
    linear_policy: LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        route: FoamDynamicsRoute = "overdamped",
        time_step: float,
        steps: int = 1,
        friction: float = 1.0,
        areal_mass: float = 1.0,
        maximum_displacement_fraction: float = 0.1,
        ccd_safety_fraction: float = 0.2,
        minimum_face_area_fraction: float = 0.1,
        ccd_candidate_capacity: int = 100_000,
        projection_iterations: int = 6,
        volume_tolerance: float = 1.0e-9,
        rank_tolerance: float = 1.0e-12,
        condition_limit: float = 1.0e12,
        energy_tolerance: float = 1.0e-10,
        maximum_constraint_entries: int = 1_000_000,
        maximum_constraint_rank_actions: int = 1_024,
        maximum_constraint_preparation_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        route_ = parse(route, FoamDynamicsRoute, "route")
        step = positive_finite_float(time_step, "time_step")
        count = positive_integer(steps, "steps")
        friction_ = positive_finite_float(friction, "friction")
        mass = positive_finite_float(areal_mass, "areal_mass")
        displacement = positive_finite_float(
            maximum_displacement_fraction, "maximum_displacement_fraction"
        )
        ccd = positive_finite_float(ccd_safety_fraction, "ccd_safety_fraction")
        area = positive_finite_float(
            minimum_face_area_fraction, "minimum_face_area_fraction"
        )
        iterations = positive_integer(projection_iterations, "projection_iterations")
        candidate_capacity = positive_integer(
            ccd_candidate_capacity, "ccd_candidate_capacity"
        )
        volume = positive_finite_float(volume_tolerance, "volume_tolerance")
        rank = positive_finite_float(rank_tolerance, "rank_tolerance")
        condition = positive_finite_float(condition_limit, "condition_limit")
        energy = positive_finite_float(energy_tolerance, "energy_tolerance")
        constraint_entries = positive_integer(
            maximum_constraint_entries, "maximum_constraint_entries"
        )
        rank_actions = positive_integer(
            maximum_constraint_rank_actions, "maximum_constraint_rank_actions"
        )
        preparation_bytes = positive_integer(
            maximum_constraint_preparation_bytes,
            "maximum_constraint_preparation_bytes",
        )
        if displacement >= 0.5 or ccd > 1.0 or area >= 1.0:
            raise ValueError(
                "Safety fractions require displacement < 0.5, CCD <= 1, and area < 1."
            )
        if rank >= 1.0 or condition <= 1.0:
            raise ValueError("Rank and condition bounds require rank < 1 < condition.")
        self.route = route_
        self.time_step = step
        self.steps = count
        self.friction = jnp.asarray(friction_, dtype=jnp.float64)
        self.areal_mass = jnp.asarray(mass, dtype=jnp.float64)
        self.maximum_displacement_fraction = displacement
        self.ccd_safety_fraction = ccd
        self.minimum_face_area_fraction = area
        self.projection_iterations = iterations
        self.ccd_candidate_capacity = candidate_capacity
        self.volume_tolerance = volume
        self.rank_tolerance = rank
        self.condition_limit = condition
        self.energy_tolerance = energy
        self.maximum_constraint_entries = constraint_entries
        self.maximum_constraint_rank_actions = rank_actions
        self.maximum_constraint_preparation_bytes = preparation_bytes
        self.linear_policy = LinearSolvePolicy(
            DenseSVD(),
            rank=RankPolicy(relative_cutoff=rank, require_full_rank=True),
            materialization=MaterializationPolicy(
                max_entries=constraint_entries,
                max_bytes=max(8, 8 * constraint_entries),
            ),
            failure=FailurePolicy("status"),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "foam-dynamics-plan",
                "route": route_,
                "time_step": float(step).hex(),
                "steps": count,
                "friction": float(friction_).hex(),
                "areal_mass": float(mass).hex(),
                "maximum_displacement_fraction": float(displacement).hex(),
                "ccd_safety_fraction": float(ccd).hex(),
                "minimum_face_area_fraction": float(area).hex(),
                "projection_iterations": iterations,
                "ccd_candidate_capacity": candidate_capacity,
                "volume_tolerance": float(volume).hex(),
                "rank_tolerance": float(rank).hex(),
                "condition_limit": float(condition).hex(),
                "energy_tolerance": float(energy).hex(),
                "maximum_constraint_entries": constraint_entries,
                "maximum_constraint_rank_actions": rank_actions,
                "maximum_constraint_preparation_bytes": preparation_bytes,
            }
        )


@final
class FoamDynamicsState(StrictModule):
    """Surface kinematics, optional extensive gas state, rim ledger, and time."""

    surface: MultiRegionSurfaceState
    gas: BubbleGasState | None
    unresolved_rim_content: Array
    time: Array

    @checked
    def __init__(
        self,
        surface: MultiRegionSurfaceState,
        /,
        *,
        gas: BubbleGasState | None = None,
        unresolved_rim_content: ArrayLike = 0.0,
        time: ArrayLike = 0.0,
    ) -> None:
        if gas is not None and not isinstance(gas, BubbleGasState):
            raise TypeError("gas must be BubbleGasState or None.")
        rim = jnp.asarray(unresolved_rim_content, dtype=surface.positions.dtype)
        time_ = jnp.asarray(time, dtype=surface.positions.dtype)
        if rim.ndim != 0 or time_.ndim != 0:
            raise ValueError("unresolved_rim_content and time must be scalar.")
        self.surface = surface
        self.gas = gas
        self.unresolved_rim_content = rim
        self.time = time_


@final
class FoamDynamicsEvidence(StrictModule):
    """Conservation, work, constraint, time-step, event, and derivative evidence."""

    status: Array
    accepted: Array
    elapsed_time: Array
    initial_energy: Array
    final_energy: Array
    energy_nonincreasing: Array
    pressure_work: Array
    viscous_dissipation: Array
    projection_work: Array
    volume_residual: Array
    constraint_rank: Array
    constraint_condition: Array
    constraint_linear_status: Array
    minimum_time_step: Array
    geometry_limited: Array
    ccd_limited: Array
    ccd_certified: Array
    minimum_ccd_time_of_impact: Array
    finite: Array
    derivative_available: Array
    topology_changed: Array
    rollback: Array
    air: RegionPressureAirEvidence | None
    event: SurfaceEventPassEvidence | None
    constrained_region_ids: tuple[str, ...] = eqx.field(static=True)
    dependent_region_ids: tuple[str, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


@final
class FoamDynamicsResult(StrictModule):
    """Committed topology/state, region pressures and complete advance evidence."""

    topology: MultiRegionSurfaceTopology
    state: FoamDynamicsState
    pressures: Array
    region_volumes: Array
    evidence: FoamDynamicsEvidence

    @property
    def successful(self) -> bool:
        """Host success decision."""
        return int(self.evidence.status) == FoamDynamicsStatus.COMPLETED


@final
class FoamKinematicStep(StrictModule):
    """One prescribed-velocity SHAKE/RATTLE step on the E6 constraint set.

    This is the fixed-topology composition point for air models whose own
    dynamics supplies a vertex velocity.  The prepared E6 volume and wire
    constraints remain authoritative; no second projection is implemented by
    the air model.
    """

    state: FoamDynamicsState
    elapsed_time: Array
    volume_residual: Array
    mechanics: ConstrainedMechanicsEvidence

    @property
    def accepted(self) -> Array:
        return self.mechanics.accepted


@final
class _FixedStepEvidence(StrictModule):
    status: Array
    accepted: Array
    elapsed_time: Array
    initial_energy: Array
    final_energy: Array
    energy_nonincreasing: Array
    pressure_work: Array
    viscous_dissipation: Array
    projection_work: Array
    volume_residual: Array
    constraint_rank: Array
    constraint_condition: Array
    geometry_limited: Array
    constraint_linear_status: Array
    ccd_limited: Array
    finite: Array
    pressures: Array
    air: RegionPressureAirEvidence | None


def _stack_gas(states: Sequence[BubbleGasState], /) -> BubbleGasState:
    if not states:
        raise ValueError("Cannot stack an empty gas-state sequence.")
    energies = tuple(state.internal_energy for state in states)
    if any(value is None for value in energies):
        raise ValueError("Foam gas compartments must carry internal energy.")
    return BubbleGasState(
        jnp.stack(tuple(state.amount for state in states)),
        jnp.stack(tuple(jnp.asarray(value) for value in energies)),
        jnp.stack(tuple(state.internal for state in states)),
    )


def _gas_at(state: BubbleGasState, index: int, /) -> BubbleGasState:
    if state.internal_energy is None:
        raise ValueError("Foam gas compartments must carry internal energy.")
    return BubbleGasState(
        state.amount[index],
        state.internal_energy[index],
        state.internal[index],
    )


def _prepare_volume_operator(
    surface: PreparedMultiRegionSurface,
    positions: Array,
    slots: Array,
    owner_id: str,
    role: str,
    /,
) -> JacobianLinearOperator:
    linearization = prepare_linearization(
        lambda values: surface.region_volumes(values)[slots],
        positions,
        linearization_id=f"{owner_id}:{role}-volume-linearization",
    )
    return JacobianLinearOperator(
        linearization,
        operator_id=f"{owner_id}:{role}-volume-jacobian",
    )


@final
class PreparedFoamDynamics(StrictModule):
    """Prepared fixed-topology operators and host event boundary for one epoch."""

    surface: PreparedMultiRegionSurface
    material: FoamMaterialPlan
    air: RegionPressureAirPlan
    plan: FoamDynamicsPlan
    constraint_basis: FoamVolumeConstraintBasis
    face_tension: Array
    finite_slots: Array
    constrained_slots: Array
    constrained_rows: tuple[int, ...] = eqx.field(static=True)
    dependent_rows: tuple[int, ...] = eqx.field(static=True)
    constrained_region_ids: tuple[str, ...] = eqx.field(static=True)
    dependent_region_ids: tuple[str, ...] = eqx.field(static=True)
    mobility_mask: Array
    wire_flat_indices: Array
    wire_values: Array
    initial_face_area: Array
    mechanics: PreparedSHAKERATTLEPlan
    kinematic_mechanics: PreparedSHAKERATTLEPlan
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: FoamDynamicsPlan,
        surface: PreparedMultiRegionSurface,
        material: FoamMaterialPlan,
        air: RegionPressureAirPlan,
        state: FoamDynamicsState,
        /,
    ) -> None:
        topology = surface.topology
        state.surface.require_topology(topology)
        finite = topology.finite_region_indices
        air.require_region_count(len(finite))
        if (air.route == "incompressible") != (state.gas is None):
            raise ValueError(
                "Incompressible state must omit gas; compressible state must provide it."
            )
        if state.gas is not None and state.gas.amount.shape != (len(finite),):
            raise ValueError("Gas state must have one entry per finite region.")
        first, second = material.face_tension_indices(topology)
        tension = material.tensions.pair_values(jnp.asarray(first), jnp.asarray(second))
        tension = jnp.where(topology.face_active, tension, 0.0).astype(jnp.float64)
        wire_flat, wire_value_indices = _wire_coordinates(material, topology)
        mask = np.zeros((topology.vertex_capacity * 3,), dtype=np.float64)
        mask[: 3 * topology.vertex_count] = 1.0
        mask[wire_flat] = 0.0
        wire_values = (
            np.zeros((0,), dtype=np.float64)
            if material.wires is None
            else np.asarray(material.wires.positions, dtype=np.float64).reshape(-1)[
                wire_value_indices
            ]
        )
        if air.route == "incompressible":
            positions = np.asarray(state.surface.positions, dtype=np.float64)
            free = np.flatnonzero(mask > 0.0)
            constraint_basis = prepare_foam_volume_constraint_basis(
                surface,
                positions,
                free,
                maximum_constraint_entries=plan.maximum_constraint_entries,
                maximum_rank_check_actions=plan.maximum_constraint_rank_actions,
                maximum_preparation_bytes=plan.maximum_constraint_preparation_bytes,
                rank_relative_tolerance=plan.rank_tolerance,
            )
        else:
            constraint_basis = _not_applicable_foam_volume_constraint_basis(
                topology,
                maximum_constraint_entries=plan.maximum_constraint_entries,
                maximum_rank_check_actions=plan.maximum_constraint_rank_actions,
                maximum_preparation_bytes=plan.maximum_constraint_preparation_bytes,
            )
        independent = constraint_basis.constrained_rows
        dependent = constraint_basis.dependent_rows
        constrained_slots = tuple(finite[row] for row in independent)
        finite_array = jnp.asarray(finite, dtype=jnp.int32)
        constrained_array = jnp.asarray(constrained_slots, dtype=jnp.int32)
        target_full = jnp.zeros((topology.region_capacity,), dtype=jnp.float64)
        if air.target_volumes is not None:
            target_full = target_full.at[finite_array].set(air.target_volumes)
        wire_indices = jnp.asarray(wire_flat, dtype=jnp.int32)
        wire_array = jnp.asarray(wire_values, dtype=jnp.float64)

        def constraint(configuration: Array, _: object) -> Array:
            positions_ = configuration.reshape((-1, 3))
            parts: list[Array] = []
            if constrained_slots:
                parts.append(
                    surface.region_volumes(positions_)[constrained_array]
                    - target_full[constrained_array]
                )
            if wire_flat.size:
                parts.append(configuration[wire_indices] - wire_array)
            return (
                jnp.concatenate(tuple(parts))
                if parts
                else jnp.zeros((0,), configuration.dtype)
            )

        def potential_gradient(configuration: Array, arguments: object) -> Array:
            positions_ = configuration.reshape((-1, 3))
            gradient = jax.grad(surface.surface_energy)(positions_, tension)
            if air.route == "compartment-gas":
                if not isinstance(arguments, BubbleGasState):
                    raise TypeError(
                        "Film-inertia compressible mechanics requires gas arguments."
                    )
                volume_operator = _prepare_volume_operator(
                    surface, positions_, finite_array, surface.prepared_id, "finite"
                )
                volumes = volume_operator.linearization.primal
                evaluation = air.evaluate(volumes, jnp.zeros_like(volumes), arguments)
                gradient = gradient - volume_operator.transpose_mv(evaluation.pressure)
            return gradient.reshape(-1)

        def zero_potential_gradient(configuration: Array, _: object) -> Array:
            return jnp.zeros_like(configuration)

        vertex_area = jnp.sum(surface.slot_areas(state.surface.positions), axis=1)
        mass = plan.areal_mass * jnp.where(vertex_area > 0.0, vertex_area, 1.0)
        mass = jnp.where(topology.vertex_active, mass, 1.0)
        inverse_mass = jnp.repeat(1.0 / mass, 3).astype(jnp.float64)
        mechanics_plan = SHAKERATTLEPlan(
            maximum_projection_steps=plan.projection_iterations,
            constraint_tolerance=plan.volume_tolerance,
            rank_tolerance=plan.rank_tolerance,
            condition_limit=plan.condition_limit,
            maximum_constraint_entries=plan.maximum_constraint_entries,
        )
        self.surface = surface
        self.material = material
        self.air = air
        self.plan = plan
        self.constraint_basis = constraint_basis
        self.face_tension = tension
        self.finite_slots = finite_array
        self.constrained_slots = constrained_array
        self.constrained_rows = independent
        self.dependent_rows = dependent
        self.constrained_region_ids = tuple(
            topology.region_ids[slot] for slot in constrained_slots
        )
        self.dependent_region_ids = tuple(
            topology.region_ids[finite[row]] for row in dependent
        )
        self.mobility_mask = jnp.asarray(mask.reshape((-1, 3)), dtype=jnp.float64)
        self.wire_flat_indices = wire_indices
        self.wire_values = wire_array
        self.initial_face_area = surface.face_areas(state.surface.positions)
        self.mechanics = mechanics_plan.prepare(
            inverse_mass,
            potential_gradient,
            constraint,
        )
        self.kinematic_mechanics = mechanics_plan.prepare(
            inverse_mass,
            zero_potential_gradient,
            constraint,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-foam-dynamics",
                "plan": plan.plan_id,
                "surface": surface.prepared_id,
                "material": material.material_id,
                "air": air.plan_id,
                "constraint_basis": constraint_basis.basis_id,
                "constrained": list(self.constrained_region_ids),
                "dependent": list(self.dependent_region_ids),
            }
        )

    def _energy(self, positions: Array, gas: BubbleGasState | None, /) -> Array:
        energy = self.surface.surface_energy(positions, self.face_tension)
        if gas is not None and gas.internal_energy is not None:
            energy = energy + jnp.sum(gas.internal_energy)
        return energy

    def volume_operator(
        self,
        positions: ArrayLike,
        /,
        *,
        constrained: bool = False,
    ) -> JacobianLinearOperator:
        """Linearize selected region volumes once with exact JVP/VJP actions.

        By default the target contains every finite region in canonical order.
        ``constrained=True`` selects the independent incompressible rows used by
        the prepared E6 volume constraint.
        """
        if not isinstance(constrained, bool):
            raise TypeError("constrained must be a bool.")
        positions_ = jnp.asarray(positions)
        expected = (self.surface.topology.vertex_capacity, 3)
        if positions_.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        if not jnp.issubdtype(positions_.dtype, jnp.floating):
            raise TypeError("positions must have a floating dtype.")
        slots = self.constrained_slots if constrained else self.finite_slots
        role = "constrained" if constrained else "finite"
        owner_id = self.prepared_id if constrained else self.surface.prepared_id
        return _prepare_volume_operator(self.surface, positions_, slots, owner_id, role)

    def _gram_solve(
        self,
        operator: AbstractLinearOperator,
        metric: Array,
        right_hand_side: Array,
        /,
    ) -> tuple[Array, Array, Array, Array, Array]:
        target = operator.target
        success_status = jnp.asarray(int(LinearSolveStatus.SUCCESS), dtype=jnp.int32)
        if target.size == 0:
            return (
                jnp.zeros_like(right_hand_side),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1.0, dtype=metric.dtype),
                success_status,
                jnp.asarray(True),
            )
        if target.size**2 > self.plan.maximum_constraint_entries:
            return (
                jnp.zeros_like(right_hand_side),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(jnp.inf, dtype=metric.dtype),
                jnp.asarray(int(LinearSolveStatus.CAPABILITY_REJECTED), dtype=jnp.int32),
                jnp.asarray(False),
            )

        def gram_action(multiplier: Array) -> Array:
            transpose = operator.transpose_mv(multiplier)
            return operator.mv(metric * transpose)

        gram = FunctionLinearOperator(
            gram_action,
            source=target,
            target=target,
            operator_id=f"{operator.operator_id}:metric-gram",
        )
        result = solve_linear(
            MinimumNormProblem(gram),
            right_hand_side,
            policy=self.plan.linear_policy,
        )
        rank = result.diagnostics.rank
        condition = result.diagnostics.condition_estimate
        successful = (
            result.successful
            & (rank == target.size)
            & jnp.isfinite(condition)
            & (condition <= self.plan.condition_limit)
        )
        return result.value, rank, condition, result.status, successful

    def _mobility(self, positions: Array, /) -> Array:
        area = jnp.sum(self.surface.slot_areas(positions), axis=1)
        return (
            self.mobility_mask
            / (self.plan.friction * jnp.where(area > 0.0, area, 1.0))[:, None]
        )

    def _motion_limit(
        self, positions: Array, velocity: Array, /
    ) -> tuple[Array, Array, Array]:
        topology = self.surface.topology
        edges = jnp.maximum(topology.edges, 0)
        edge_lengths = jnp.linalg.norm(
            positions[edges[:, 1]] - positions[edges[:, 0]], axis=1
        )
        shortest = jnp.min(jnp.where(topology.edge_active, edge_lengths, jnp.inf))
        speed = jnp.max(jnp.linalg.norm(velocity, axis=1))
        geometry_step = (
            self.plan.maximum_displacement_fraction
            * shortest
            / jnp.maximum(speed, jnp.finfo(positions.dtype).tiny)
        )
        faces = jnp.maximum(topology.faces, 0)
        corners = positions[faces]
        side = jnp.stack(
            (
                jnp.linalg.norm(corners[:, 1] - corners[:, 0], axis=1),
                jnp.linalg.norm(corners[:, 2] - corners[:, 1], axis=1),
                jnp.linalg.norm(corners[:, 0] - corners[:, 2], axis=1),
            ),
            axis=1,
        )
        altitude = (
            2.0
            * self.surface.face_areas(positions)
            / jnp.maximum(jnp.max(side, axis=1), jnp.finfo(positions.dtype).tiny)
        )
        minimum_altitude = jnp.min(jnp.where(topology.face_active, altitude, jnp.inf))
        ccd_step = (
            self.plan.ccd_safety_fraction
            * minimum_altitude
            / jnp.maximum(2.0 * speed, jnp.finfo(positions.dtype).tiny)
        )
        step = jnp.minimum(self.plan.time_step, jnp.minimum(geometry_step, ccd_step))
        return step, geometry_step < self.plan.time_step, ccd_step < self.plan.time_step

    def certify_motion(self, start: Array, end: Array, /) -> tuple[bool, float]:
        """Host inclusion-CCD certificate for one fixed-topology motion leg."""
        from ...discretization.contact import (
            collision_free_step_limit,
            CollisionSurfacePlan,
            InclusionCCDPlan,
            PreparedCollisionScene,
            PreparedCollisionSurface,
            selection_collision_operator,
            SweepAndPruneContactSearchPlan,
        )

        topology = self.surface.topology
        count = topology.vertex_count
        before = np.asarray(start[:count], dtype=np.float64)
        after = np.asarray(end[:count], dtype=np.float64)
        if np.array_equal(before, after):
            return True, 1.0
        collision_plan = CollisionSurfacePlan(
            np.asarray(topology.vertex_global_ids[:count], dtype=np.int64),
            ambient_dimension=3,
            faces=topology.host_faces(),
        )
        space = ArraySpace((count, 3), dtype=np.float64)
        collision_surface = PreparedCollisionSurface(
            collision_plan,
            before,
            selection_collision_operator(
                space,
                np.arange(count, dtype=np.int32),
            ),
        )
        scene = PreparedCollisionScene((collision_surface,))
        span = float(
            np.max(np.max(np.concatenate((before, after)), axis=0))
            - np.min(np.min(np.concatenate((before, after)), axis=0))
        )
        search = SweepAndPruneContactSearchPlan(
            edge_vertex_capacity=0,
            edge_edge_capacity=self.plan.ccd_candidate_capacity,
            face_vertex_capacity=self.plan.ccd_candidate_capacity,
            activation_distance=max(1.0e-12 * max(span, 1.0), 1.0e-300),
        )
        epoch = search.build(scene, before, end_positions=after)
        safety = collision_free_step_limit(
            InclusionCCDPlan(),
            scene,
            epoch,
            before,
            after,
        )
        impact = float(safety.minimum_time_of_impact)
        if not np.isfinite(impact):
            impact = 1.0
        successful = (
            bool(epoch.successful)
            and bool(safety.successful)
            and float(safety.step_size) >= 1.0
        )
        return successful, impact

    def _project_volumes(
        self, positions: Array, metric: Array, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        success_status = jnp.asarray(int(LinearSolveStatus.SUCCESS), dtype=jnp.int32)
        if self.air.target_volumes is None or not self.constrained_rows:
            return (
                positions,
                jnp.asarray(0.0, dtype=positions.dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1.0, dtype=positions.dtype),
                success_status,
                jnp.asarray(True),
            )
        targets = self.air.target_volumes[jnp.asarray(self.constrained_rows)]
        rank = jnp.asarray(0, dtype=jnp.int32)
        condition = jnp.asarray(1.0, dtype=positions.dtype)
        linear_status = success_status
        successful = jnp.asarray(True)

        def iteration(
            _: int, carry: tuple[Array, Array, Array, Array, Array]
        ) -> tuple[Array, Array, Array, Array, Array]:
            values, rank_, condition_, linear_status_, successful_ = carry
            operator = self.volume_operator(values, constrained=True)
            residual = operator.linearization.primal - targets
            multiplier, solved_rank, solved_condition, solved_status, solved = (
                self._gram_solve(operator, metric, residual)
            )
            correction = metric * operator.transpose_mv(multiplier)
            retained_status = jnp.where(
                linear_status_ == int(LinearSolveStatus.SUCCESS),
                solved_status,
                linear_status_,
            )
            return (
                values - correction,
                solved_rank,
                solved_condition,
                retained_status,
                successful_ & solved,
            )

        positions, rank, condition, linear_status, successful = jax.lax.fori_loop(
            0,
            self.plan.projection_iterations,
            iteration,
            (positions, rank, condition, linear_status, successful),
        )
        volumes = self.surface.region_volumes(positions)[self.finite_slots]
        scale = jnp.maximum(jnp.abs(self.air.target_volumes), 1.0)
        residual = jnp.max(jnp.abs(volumes - self.air.target_volumes) / scale)
        return (
            positions,
            residual,
            rank,
            condition,
            linear_status,
            successful & (residual <= self.plan.volume_tolerance),
        )

    def _overdamped_step(
        self, state: FoamDynamicsState, /
    ) -> tuple[FoamDynamicsState, _FixedStepEvidence]:
        positions = state.surface.positions
        initial_energy = self._energy(positions, state.gas)
        mobility = self._mobility(positions)
        gradient = jax.grad(self.surface.surface_energy)(positions, self.face_tension)
        pressures_finite = jnp.zeros((self.finite_slots.size,), dtype=positions.dtype)
        constraint_rank = jnp.asarray(0, dtype=jnp.int32)
        constraint_condition = jnp.asarray(1.0, dtype=positions.dtype)
        constraint_linear_status = jnp.asarray(
            int(LinearSolveStatus.SUCCESS), dtype=jnp.int32
        )
        solved = jnp.asarray(True)
        air_evidence: RegionPressureAirEvidence | None = None
        if self.air.route == "compartment-gas":
            if state.gas is None:
                raise RuntimeError("Compressible dynamics lost its gas state.")
            volume_operator = self.volume_operator(positions)
            volumes = volume_operator.linearization.primal
            gas_evaluation = self.air.evaluate(
                volumes, jnp.zeros_like(volumes), state.gas
            )
            pressures_finite = gas_evaluation.pressure
            gradient = gradient - volume_operator.transpose_mv(pressures_finite)
            velocity = -mobility * gradient
        else:
            volume_operator = self.volume_operator(positions, constrained=True)
            right = -volume_operator.mv(mobility * gradient)
            (
                multiplier,
                constraint_rank,
                constraint_condition,
                constraint_linear_status,
                solved,
            ) = self._gram_solve(volume_operator, mobility, right)
            velocity = -mobility * (gradient + volume_operator.transpose_mv(multiplier))
            pressures_finite = pressures_finite.at[
                jnp.asarray(self.constrained_rows, dtype=jnp.int32)
            ].set(-multiplier)
        step, geometry_limited, ccd_limited = self._motion_limit(positions, velocity)
        candidate_positions = positions + step * velocity
        (
            candidate_positions,
            volume_residual,
            projected_rank,
            projected_condition,
            projected_linear_status,
            projected,
        ) = self._project_volumes(candidate_positions, mobility)
        if self.wire_flat_indices.size:
            flat = (
                candidate_positions.reshape(-1)
                .at[self.wire_flat_indices]
                .set(self.wire_values)
            )
            candidate_positions = flat.reshape(candidate_positions.shape)
        candidate_velocity = (candidate_positions - positions) / step
        candidate_gas = state.gas
        pressure_work = jnp.asarray(0.0, dtype=positions.dtype)
        gas_ok = jnp.asarray(True)
        if self.air.route == "compartment-gas":
            if state.gas is None:
                raise RuntimeError("Compressible dynamics lost its gas state.")
            old_volumes = self.surface.region_volumes(positions)[self.finite_slots]
            new_volumes = self.surface.region_volumes(candidate_positions)[
                self.finite_slots
            ]
            volume_rates = (new_volumes - old_volumes) / step
            candidate_gas, air_evidence = self.air.advance(
                old_volumes, volume_rates, state.gas, step
            )
            pressure_work = air_evidence.pressure_work
            gas_ok = air_evidence.admissible
        final_energy = self._energy(candidate_positions, candidate_gas)
        safe_mobility = jnp.where(mobility > 0.0, mobility, 1.0)
        dissipated = step * jnp.sum(
            jnp.where(
                mobility > 0.0,
                candidate_velocity * candidate_velocity / safe_mobility,
                0.0,
            )
        )
        heat_transfer = (
            self.air.gas_law is not None and self.air.gas_law.capabilities.heat_transfer
        )
        energy_nonincreasing = (
            final_energy
            <= initial_energy
            + self.plan.energy_tolerance * jnp.maximum(jnp.abs(initial_energy), 1.0)
        )
        energy_ok = jnp.asarray(True) if heat_transfer else energy_nonincreasing
        areas = self.surface.face_areas(candidate_positions)
        area_ok = jnp.all(
            jnp.where(
                self.surface.topology.face_active,
                areas >= self.plan.minimum_face_area_fraction * self.initial_face_area,
                True,
            )
        )
        finite = (
            jnp.all(jnp.isfinite(candidate_positions))
            & jnp.all(jnp.isfinite(candidate_velocity))
            & jnp.isfinite(final_energy)
        )
        rank = jnp.maximum(constraint_rank, projected_rank)
        condition = jnp.maximum(constraint_condition, projected_condition)
        linear_status = jnp.where(
            constraint_linear_status == int(LinearSolveStatus.SUCCESS),
            projected_linear_status,
            constraint_linear_status,
        )
        accepted = finite & solved & projected & gas_ok & area_ok & energy_ok
        status = jnp.where(
            ~finite,
            int(FoamDynamicsStatus.NONFINITE),
            jnp.where(
                ~(solved & projected),
                int(FoamDynamicsStatus.MECHANICS_FAILED),
                jnp.where(
                    ~gas_ok,
                    int(FoamDynamicsStatus.GAS_FAILED),
                    jnp.where(
                        ~(area_ok & energy_ok),
                        int(FoamDynamicsStatus.GEOMETRY_FAILED),
                        int(FoamDynamicsStatus.COMPLETED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        surface = eqx.tree_at(
            lambda value: (value.positions, value.velocities),
            state.surface,
            (
                jnp.where(accepted, candidate_positions, positions),
                jnp.where(accepted, candidate_velocity, state.surface.velocities),
            ),
        )
        candidate = FoamDynamicsState(
            surface,
            gas=(
                None
                if state.gas is None
                else jax.tree.map(
                    lambda proposed, source: jnp.where(accepted, proposed, source),
                    candidate_gas,
                    state.gas,
                )
            ),
            unresolved_rim_content=state.unresolved_rim_content,
            time=jnp.where(accepted, state.time + step, state.time),
        )
        pressure_full = (
            jnp.zeros((self.surface.topology.region_capacity,), dtype=positions.dtype)
            .at[self.finite_slots]
            .set(pressures_finite)
        )
        evidence = _FixedStepEvidence(
            status=status,
            accepted=accepted,
            elapsed_time=jnp.where(accepted, step, 0.0),
            initial_energy=initial_energy,
            final_energy=jnp.where(accepted, final_energy, initial_energy),
            energy_nonincreasing=energy_nonincreasing,
            pressure_work=pressure_work,
            viscous_dissipation=dissipated,
            projection_work=jnp.asarray(0.0, dtype=positions.dtype),
            volume_residual=volume_residual,
            constraint_rank=rank,
            constraint_condition=condition,
            constraint_linear_status=linear_status,
            geometry_limited=geometry_limited,
            ccd_limited=ccd_limited,
            finite=finite,
            pressures=pressure_full,
            air=air_evidence,
        )
        return candidate, evidence

    def _inertia_step(
        self, state: FoamDynamicsState, /
    ) -> tuple[FoamDynamicsState, _FixedStepEvidence]:
        positions = state.surface.positions
        initial_energy = self._energy(positions, state.gas)
        velocity = state.surface.velocities
        step, geometry_limited, ccd_limited = self._motion_limit(positions, velocity)
        momentum = velocity.reshape(-1) / self.mechanics.inverse_mass
        mechanical = self.mechanics.step(
            ConstrainedMechanicalState(positions.reshape(-1), momentum),
            step,
            args=state.gas,
        )
        candidate_positions = mechanical.state.configuration.reshape(positions.shape)
        candidate_velocity = (
            self.mechanics.inverse_mass * mechanical.state.momentum
        ).reshape(positions.shape)
        candidate_gas = state.gas
        air_evidence: RegionPressureAirEvidence | None = None
        pressure_work = jnp.asarray(0.0, dtype=positions.dtype)
        gas_ok = jnp.asarray(True)
        pressures_finite = jnp.zeros((self.finite_slots.size,), dtype=positions.dtype)
        recovery_rank = mechanical.evidence.constraint_rank
        recovery_condition = jnp.asarray(1.0, dtype=positions.dtype)
        recovery_linear_status = jnp.asarray(
            int(LinearSolveStatus.SUCCESS), dtype=jnp.int32
        )
        recovery_solved = jnp.asarray(True)
        if self.air.route == "compartment-gas":
            if state.gas is None:
                raise RuntimeError("Compressible dynamics lost its gas state.")
            old_volumes = self.surface.region_volumes(positions)[self.finite_slots]
            new_volumes = self.surface.region_volumes(candidate_positions)[
                self.finite_slots
            ]
            volume_rates = (new_volumes - old_volumes) / step
            candidate_gas, air_evidence = self.air.advance(
                old_volumes, volume_rates, state.gas, step
            )
            pressures_finite = air_evidence.pressures
            pressure_work = air_evidence.pressure_work
            gas_ok = air_evidence.admissible
        elif self.constrained_rows:
            gradient = jax.grad(self.surface.surface_energy)(
                candidate_positions, self.face_tension
            ).reshape(-1)
            operator = self.mechanics.constraint_operator(
                candidate_positions.reshape(-1), state.gas
            )
            metric = self.mechanics.inverse_mass
            right = -operator.mv(metric * gradient)
            (
                multiplier,
                recovery_rank,
                recovery_condition,
                recovery_linear_status,
                recovery_solved,
            ) = self._gram_solve(operator, metric, right)
            pressures_finite = pressures_finite.at[
                jnp.asarray(self.constrained_rows, dtype=jnp.int32)
            ].set(-multiplier[: len(self.constrained_rows)])
        final_energy = self._energy(candidate_positions, candidate_gas)
        areas = self.surface.face_areas(candidate_positions)
        area_ok = jnp.all(
            jnp.where(
                self.surface.topology.face_active,
                areas >= self.plan.minimum_face_area_fraction * self.initial_face_area,
                True,
            )
        )
        finite = (
            mechanical.evidence.finite
            & jnp.all(jnp.isfinite(candidate_positions))
            & jnp.all(jnp.isfinite(candidate_velocity))
            & jnp.isfinite(final_energy)
        )
        accepted = mechanical.accepted & recovery_solved & gas_ok & area_ok & finite
        status = jnp.where(
            ~finite,
            int(FoamDynamicsStatus.NONFINITE),
            jnp.where(
                ~(mechanical.accepted & recovery_solved),
                int(FoamDynamicsStatus.MECHANICS_FAILED),
                jnp.where(
                    ~gas_ok,
                    int(FoamDynamicsStatus.GAS_FAILED),
                    jnp.where(
                        ~area_ok,
                        int(FoamDynamicsStatus.GEOMETRY_FAILED),
                        int(FoamDynamicsStatus.COMPLETED),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        surface = eqx.tree_at(
            lambda value: (value.positions, value.velocities),
            state.surface,
            (
                jnp.where(accepted, candidate_positions, positions),
                jnp.where(accepted, candidate_velocity, velocity),
            ),
        )
        candidate = FoamDynamicsState(
            surface,
            gas=(
                None
                if state.gas is None
                else jax.tree.map(
                    lambda proposed, source: jnp.where(accepted, proposed, source),
                    candidate_gas,
                    state.gas,
                )
            ),
            unresolved_rim_content=state.unresolved_rim_content,
            time=jnp.where(accepted, state.time + step, state.time),
        )
        finite_volumes = self.surface.region_volumes(candidate.surface.positions)[
            self.finite_slots
        ]
        volume_residual = (
            jnp.asarray(0.0, dtype=positions.dtype)
            if self.air.target_volumes is None
            else jnp.max(
                jnp.abs(finite_volumes - self.air.target_volumes)
                / jnp.maximum(jnp.abs(self.air.target_volumes), 1.0)
            )
        )
        pressure_full = (
            jnp.zeros((self.surface.topology.region_capacity,), dtype=positions.dtype)
            .at[self.finite_slots]
            .set(pressures_finite)
        )
        energy_scale = jnp.maximum(jnp.abs(initial_energy), 1.0)
        mechanical_linear_status = jnp.where(
            mechanical.evidence.position_linear_status == int(LinearSolveStatus.SUCCESS),
            mechanical.evidence.velocity_linear_status,
            mechanical.evidence.position_linear_status,
        )
        constraint_linear_status = jnp.where(
            mechanical_linear_status == int(LinearSolveStatus.SUCCESS),
            recovery_linear_status,
            mechanical_linear_status,
        )
        constraint_rank = jnp.minimum(mechanical.evidence.constraint_rank, recovery_rank)
        constraint_condition = jnp.maximum(
            mechanical.evidence.constraint_condition, recovery_condition
        )
        evidence = _FixedStepEvidence(
            status=status,
            accepted=accepted,
            elapsed_time=jnp.where(accepted, step, 0.0),
            initial_energy=initial_energy,
            final_energy=jnp.where(accepted, final_energy, initial_energy),
            energy_nonincreasing=jnp.abs(final_energy - initial_energy)
            <= self.plan.energy_tolerance * energy_scale,
            pressure_work=pressure_work,
            viscous_dissipation=jnp.asarray(0.0, dtype=positions.dtype),
            projection_work=mechanical.evidence.projection_work,
            volume_residual=volume_residual,
            constraint_rank=constraint_rank,
            constraint_condition=constraint_condition,
            constraint_linear_status=constraint_linear_status,
            geometry_limited=geometry_limited,
            ccd_limited=ccd_limited,
            finite=finite,
            pressures=pressure_full,
            air=air_evidence,
        )
        return candidate, evidence

    @checked
    def fixed_topology_step(
        self, state: FoamDynamicsState, /
    ) -> tuple[FoamDynamicsState, _FixedStepEvidence]:
        """One differentiable fixed-topology step with atomic rollback."""
        state.surface.require_topology(self.surface.topology)
        if self.plan.route == "overdamped":
            return self._overdamped_step(state)
        return self._inertia_step(state)

    @checked
    def constrained_kinematic_step(
        self,
        state: FoamDynamicsState,
        velocity: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> FoamKinematicStep:
        """Advance a prescribed velocity with the prepared volume/wire constraints."""
        state.surface.require_topology(self.surface.topology)
        if self.air.route != "incompressible":
            raise ValueError(
                "Prescribed-velocity constrained steps require incompressible targets."
            )
        velocity_ = jnp.asarray(velocity, dtype=state.surface.positions.dtype)
        if velocity_.shape != state.surface.positions.shape:
            raise ValueError("velocity must match the surface position shape.")
        step = jnp.asarray(step_size, dtype=state.surface.positions.dtype)
        if step.ndim != 0:
            raise ValueError("step_size must be scalar.")
        momentum = velocity_.reshape(-1) / self.kinematic_mechanics.inverse_mass
        mechanical = self.kinematic_mechanics.step(
            ConstrainedMechanicalState(state.surface.positions.reshape(-1), momentum),
            step,
        )
        positions = mechanical.state.configuration.reshape(state.surface.positions.shape)
        projected_velocity = (
            self.kinematic_mechanics.inverse_mass * mechanical.state.momentum
        ).reshape(velocity_.shape)
        surface = eqx.tree_at(
            lambda value: (value.positions, value.velocities),
            state.surface,
            (
                jnp.where(mechanical.accepted, positions, state.surface.positions),
                jnp.where(
                    mechanical.accepted,
                    projected_velocity,
                    state.surface.velocities,
                ),
            ),
        )
        candidate = FoamDynamicsState(
            surface,
            gas=state.gas,
            unresolved_rim_content=state.unresolved_rim_content,
            time=jnp.where(mechanical.accepted, state.time + step, state.time),
        )
        volumes = self.surface.region_volumes(candidate.surface.positions)[
            self.finite_slots
        ]
        targets = self.air.target_volumes
        if targets is None:
            raise RuntimeError("Incompressible air lost its volume targets.")
        volume_residual = jnp.max(
            jnp.abs(volumes - targets) / jnp.maximum(jnp.abs(targets), 1.0)
        )
        return FoamKinematicStep(
            state=candidate,
            elapsed_time=jnp.where(mechanical.accepted, step, 0.0),
            volume_residual=volume_residual,
            mechanics=mechanical.evidence,
        )

    def _transfer_gas_after_event(
        self,
        source: FoamDynamicsState,
        target_topology: MultiRegionSurfaceTopology,
        target_state: MultiRegionSurfaceState,
        event: SurfaceEventPassEvidence,
        /,
    ) -> tuple[BubbleGasState | None, bool]:
        if source.gas is None:
            return None, True
        lineage = event.lineage
        if lineage is None:
            return (
                source.gas,
                target_topology.region_ids == self.surface.topology.region_ids,
            )
        law = self.air.gas_law
        environment = self.air.environment
        if law is None or environment is None:
            return None, False
        source_finite = self.surface.topology.finite_region_indices
        target_finite = target_topology.finite_region_indices
        source_ids = tuple(
            self.surface.topology.region_ids[slot] for slot in source_finite
        )
        target_ids = tuple(target_topology.region_ids[slot] for slot in target_finite)
        source_index = {name: index for index, name in enumerate(source_ids)}
        parentage = {child: parents for child, parents in lineage.region_parents}
        source_volumes = self.surface.region_volumes(source.surface.positions)[
            self.finite_slots
        ]
        target_surface = PreparedMultiRegionSurface(target_topology, target_state)
        target_volumes = target_surface.region_volumes(target_state.positions)[
            jnp.asarray(target_finite, dtype=jnp.int32)
        ]
        children_by_parent: dict[str, list[int]] = {}
        for child_index, child in enumerate(target_ids):
            parents = parentage.get(child, (child,))
            if len(parents) == 1 and parents[0] != child:
                children_by_parent.setdefault(parents[0], []).append(child_index)
        split_states: dict[int, BubbleGasState] = {}
        for parent, child_indices in children_by_parent.items():
            if parent not in source_index:
                return None, False
            parent_index = source_index[parent]
            split = law.split(
                _gas_at(source.gas, parent_index),
                source_volumes[parent_index],
                target_volumes[jnp.asarray(child_indices, dtype=jnp.int32)],
                environment,
            )
            if not bool(split.admissible):
                return None, False
            for offset, child_index in enumerate(child_indices):
                split_states[child_index] = _gas_at(split.states, offset)
        target_parts: list[BubbleGasState] = []
        for child_index, child in enumerate(target_ids):
            if child_index in split_states:
                target_parts.append(split_states[child_index])
                continue
            parents = parentage.get(child, (child,))
            if len(parents) == 1 and parents[0] in source_index:
                target_parts.append(_gas_at(source.gas, source_index[parents[0]]))
                continue
            indices = tuple(
                source_index[parent] for parent in parents if parent in source_index
            )
            if len(indices) != len(parents):
                return None, False
            stacked = _stack_gas(tuple(_gas_at(source.gas, index) for index in indices))
            merged = law.merge(
                stacked,
                source_volumes[jnp.asarray(indices, dtype=jnp.int32)],
                target_volumes[child_index],
                environment,
            )
            if not bool(merged.admissible):
                return None, False
            target_parts.append(merged.state)
        target_gas = _stack_gas(target_parts)
        amount_ok = bool(
            jnp.abs(jnp.sum(target_gas.amount) - jnp.sum(source.gas.amount))
            <= 1.0e-12 * jnp.maximum(jnp.abs(jnp.sum(source.gas.amount)), 1.0e-300)
        )
        if source.gas.internal_energy is None or target_gas.internal_energy is None:
            return None, False
        energy_ok = bool(
            jnp.abs(
                jnp.sum(target_gas.internal_energy) - jnp.sum(source.gas.internal_energy)
            )
            <= 1.0e-12
            * jnp.maximum(jnp.abs(jnp.sum(source.gas.internal_energy)), 1.0e-300)
        )
        return target_gas, amount_ok and energy_ok

    @checked
    def advance(
        self,
        state: FoamDynamicsState,
        /,
        *,
        events: Sequence[SurfaceEventProposal] = (),
        event_policy: SurfaceEventPolicy | None = None,
    ) -> FoamDynamicsResult:
        """Advance fixed topology, then atomically apply any E3/E4 event pass."""
        state.surface.require_topology(self.surface.topology)
        current = state
        first_energy = self._energy(state.surface.positions, state.gas)
        elapsed = jnp.asarray(0.0, dtype=state.surface.positions.dtype)
        pressure_work = jnp.asarray(0.0, dtype=state.surface.positions.dtype)
        dissipation = jnp.asarray(0.0, dtype=state.surface.positions.dtype)
        projection_work = jnp.asarray(0.0, dtype=state.surface.positions.dtype)
        minimum_step = jnp.asarray(jnp.inf, dtype=state.surface.positions.dtype)
        rank = jnp.asarray(0, dtype=jnp.int32)
        condition = jnp.asarray(1.0, dtype=state.surface.positions.dtype)
        constraint_linear_status = jnp.asarray(
            int(LinearSolveStatus.SUCCESS), dtype=jnp.int32
        )
        accepted = jnp.asarray(True)
        energy_nonincreasing = jnp.asarray(True)
        geometry_limited = jnp.asarray(False)
        ccd_limited = jnp.asarray(False)
        ccd_certified = True
        minimum_ccd_impact = 1.0
        finite = jnp.asarray(True)
        pressures = jnp.zeros(
            (self.surface.topology.region_capacity,), dtype=state.surface.positions.dtype
        )
        air_evidence: RegionPressureAirEvidence | None = None
        status = jnp.asarray(int(FoamDynamicsStatus.COMPLETED), dtype=jnp.int32)
        for _ in range(self.plan.steps):
            source = current
            candidate, step_evidence = self.fixed_topology_step(source)
            rank = jnp.maximum(rank, step_evidence.constraint_rank)
            condition = jnp.maximum(condition, step_evidence.constraint_condition)
            constraint_linear_status = jnp.where(
                constraint_linear_status == int(LinearSolveStatus.SUCCESS),
                step_evidence.constraint_linear_status,
                constraint_linear_status,
            )
            geometry_limited = geometry_limited | step_evidence.geometry_limited
            ccd_limited = ccd_limited | step_evidence.ccd_limited
            finite = finite & step_evidence.finite
            if bool(step_evidence.accepted):
                leg_certified, impact = self.certify_motion(
                    source.surface.positions,
                    candidate.surface.positions,
                )
                minimum_ccd_impact = min(minimum_ccd_impact, impact)
                ccd_certified = ccd_certified and leg_certified
                if not leg_certified:
                    status = jnp.asarray(
                        int(FoamDynamicsStatus.CCD_FAILED), dtype=jnp.int32
                    )
                    accepted = jnp.asarray(False)
                    ccd_limited = jnp.asarray(True)
                    current = source
                    break
            current = candidate
            status = step_evidence.status
            accepted = accepted & step_evidence.accepted
            elapsed = elapsed + step_evidence.elapsed_time
            minimum_step = jnp.minimum(
                minimum_step,
                jnp.where(step_evidence.accepted, step_evidence.elapsed_time, jnp.inf),
            )
            pressure_work = pressure_work + step_evidence.pressure_work
            dissipation = dissipation + step_evidence.viscous_dissipation
            projection_work = projection_work + step_evidence.projection_work
            energy_nonincreasing = (
                energy_nonincreasing & step_evidence.energy_nonincreasing
            )
            pressures = step_evidence.pressures
            air_evidence = step_evidence.air
            if not bool(step_evidence.accepted):
                break
        fixed_topology_accepted = bool(accepted)
        if not fixed_topology_accepted:
            current = state
            elapsed = jnp.zeros_like(elapsed)
            pressure_work = jnp.zeros_like(pressure_work)
            dissipation = jnp.zeros_like(dissipation)
            projection_work = jnp.zeros_like(projection_work)
            minimum_step = jnp.asarray(0.0, dtype=state.surface.positions.dtype)
            energy_nonincreasing = jnp.asarray(True)
        minimum_step = jnp.where(jnp.isfinite(minimum_step), minimum_step, 0.0)
        event_evidence: SurfaceEventPassEvidence | None = None
        topology = self.surface.topology
        topology_changed = False
        event_rollback = False
        if events and fixed_topology_accepted:
            event_result = apply_surface_events(
                topology,
                current.surface,
                events,
                policy=event_policy,
            )
            event_evidence = event_result.evidence
            if event_result.evidence.committed:
                gas, gas_ok = self._transfer_gas_after_event(
                    current,
                    event_result.topology,
                    event_result.state,
                    event_result.evidence,
                )
                if gas_ok:
                    topology = event_result.topology
                    current = FoamDynamicsState(
                        event_result.state,
                        gas=gas,
                        unresolved_rim_content=current.unresolved_rim_content,
                        time=current.time,
                    )
                    topology_changed = True
                else:
                    event_rollback = True
                    status = jnp.asarray(
                        int(FoamDynamicsStatus.TOPOLOGY_EVENT_ROLLED_BACK),
                        dtype=jnp.int32,
                    )
                    accepted = jnp.asarray(False)
        validation = validate_multiregion_surface(topology, current.surface)
        if not validation.accepted:
            topology = self.surface.topology
            current = state
            event_rollback = True
            status = jnp.asarray(int(FoamDynamicsStatus.GEOMETRY_FAILED), dtype=jnp.int32)
            accepted = jnp.asarray(False)
        volumes = (
            self.surface.region_volumes(current.surface.positions)
            if topology is self.surface.topology
            else PreparedMultiRegionSurface(topology, current.surface).region_volumes(
                current.surface.positions
            )
        )
        final_energy = (
            self._energy(current.surface.positions, current.gas)
            if topology is self.surface.topology
            else jnp.asarray(jnp.nan, dtype=state.surface.positions.dtype)
        )
        target = self.air.target_volumes
        volume_residual = (
            jnp.asarray(0.0, dtype=state.surface.positions.dtype)
            if target is None or topology is not self.surface.topology
            else jnp.max(
                jnp.abs(volumes[self.finite_slots] - target)
                / jnp.maximum(jnp.abs(target), 1.0)
            )
        )
        successful = accepted & (status == int(FoamDynamicsStatus.COMPLETED))
        evidence = FoamDynamicsEvidence(
            status=status,
            accepted=successful,
            elapsed_time=elapsed,
            initial_energy=first_energy,
            final_energy=final_energy,
            energy_nonincreasing=energy_nonincreasing,
            pressure_work=pressure_work,
            viscous_dissipation=dissipation,
            projection_work=projection_work,
            volume_residual=volume_residual,
            constraint_rank=rank,
            constraint_condition=condition,
            constraint_linear_status=constraint_linear_status,
            minimum_time_step=minimum_step,
            geometry_limited=geometry_limited,
            ccd_limited=ccd_limited,
            ccd_certified=jnp.asarray(ccd_certified),
            minimum_ccd_time_of_impact=jnp.asarray(
                minimum_ccd_impact, dtype=state.surface.positions.dtype
            ),
            finite=finite,
            derivative_available=jnp.asarray(False),
            topology_changed=jnp.asarray(topology_changed),
            rollback=jnp.asarray(event_rollback) | ~successful,
            air=air_evidence,
            event=event_evidence,
            constrained_region_ids=self.constrained_region_ids,
            dependent_region_ids=self.dependent_region_ids,
            prepared_id=self.prepared_id,
        )
        return FoamDynamicsResult(topology, current, pressures, volumes, evidence)


__all__ = [
    "FoamDynamicsEvidence",
    "FoamDynamicsPlan",
    "FoamKinematicStep",
    "FoamDynamicsResult",
    "FoamDynamicsRoute",
    "FoamDynamicsState",
    "FoamDynamicsStatus",
    "PreparedFoamDynamics",
]
