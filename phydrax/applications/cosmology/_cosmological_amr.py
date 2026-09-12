#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.amr._composite import CompositeAMRCellLayout
from ...discretization.amr._core import (
    BlockHierarchyState,
    BlockHierarchyTopology,
    BlockLevelState,
)
from ...discretization.finite_volume._amr_diffusion import (
    PreparedCompositeAMRDiffusion,
)
from ...linalg import (
    LinearSolvePolicy,
    LinearSolveResult,
    ProjectedPCG,
    solve,
    TolerancePolicy,
)
from ...solver._block_amr_runtime import (
    BlockAMRAdvanceResult,
    BlockAMRRuntimeState,
    PreparedBlockAMRRuntime,
)
from ._particles import CosmologicalParticleState


class AMRParticleLevelAssignment(StrictModule):
    """Finest-leaf routing of particles into one fixed block topology epoch."""

    levels: Array
    block_ids: Array
    block_slots: Array
    global_cell_indices: Array
    local_cell_indices: Array
    composite_cell_indices: Array
    active: Array
    support: Array
    successful: Array
    epoch_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)


class AMRParticleDepositResult(StrictModule):
    """Conservative nearest-cell particle mass deposition on composite leaves."""

    content: tuple[Array, ...]
    density: tuple[Array, ...]
    source_mass: Array
    deposited_mass: Array
    balance_defect: Array
    support: Array
    successful: Array
    epoch_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)


class AMRParticleGatherResult(StrictModule):
    """Matched composite-leaf field gather for a particle assignment."""

    values: Array
    support: Array
    successful: Array
    epoch_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)


class BlockAMRParticleRoutingPlan(StrictModule, NonTrainableState):
    """Route particles to the finest active cell of an N-level block hierarchy.

    The topology is immutable and nondifferentiable. Particle positions remain
    ordinary array inputs; integer route selection is the only piecewise-discrete
    operation. Cosmological domains are periodic and derive their bounds from the
    hierarchy's canonical tensor geometry rather than a duplicated box definition.
    """

    topology: BlockHierarchyTopology
    layout: CompositeAMRCellLayout
    lower_bounds: tuple[float, ...] = eqx.field(static=True)
    box_size: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology,
        /,
        *,
        dtype: Any = np.float64,
    ):
        if not isinstance(topology, BlockHierarchyTopology):
            raise TypeError("topology must be BlockHierarchyTopology.")
        axes = topology.plan.grid.structured_axes
        if not axes or not all(
            axis.periodic and axis.bounds is not None for axis in axes
        ):
            raise ValueError(
                "Cosmological block AMR requires finite periodic tensor axes."
            )
        bounds = tuple(
            tuple(float(value) for value in np.asarray(axis.bounds, dtype=float))
            for axis in axes
        )
        if any(
            len(pair) != 2 or not np.all(np.isfinite(pair)) or pair[1] <= pair[0]
            for pair in bounds
        ):
            raise ValueError("Cosmological block AMR axis bounds are invalid.")
        lower = tuple(pair[0] for pair in bounds)
        lengths = tuple(pair[1] - pair[0] for pair in bounds)
        layout = CompositeAMRCellLayout(topology, dtype=dtype)
        self.topology = topology
        self.layout = layout
        self.lower_bounds = lower
        self.box_size = lengths
        self.plan_id = canonical_fingerprint(
            {
                "kind": "block-amr-particle-routing",
                "epoch": topology.epoch.epoch_id,
                "topology": topology.topology_id,
                "partition": topology.partition_id,
                "layout": layout.layout_id,
                "bounds": [list(pair) for pair in bounds],
            }
        )

    def route(
        self,
        positions: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> AMRParticleLevelAssignment:
        position = jnp.asarray(positions)
        dimension = len(self.box_size)
        if position.ndim != 2 or position.shape[1] != dimension:
            raise ValueError(
                "Particle positions must have shape (particle_count, AMR_dimension)."
            )
        count = position.shape[0]
        active = (
            jnp.ones((count,), dtype=bool)
            if active_mask is None
            else jnp.asarray(active_mask, dtype=bool)
        )
        if active.shape != (count,):
            raise ValueError("active_mask must contain one entry per particle.")

        finite = jnp.all(jnp.isfinite(position), axis=1)
        safe_position = jnp.where(
            jnp.isfinite(position),
            position,
            jnp.asarray(self.lower_bounds, dtype=position.dtype),
        )
        normalized = jnp.mod(
            (safe_position - jnp.asarray(self.lower_bounds, dtype=position.dtype))
            / jnp.asarray(self.box_size, dtype=position.dtype),
            1.0,
        )
        particle_support = active & finite
        levels = jnp.full((count,), -1, dtype=jnp.int32)
        block_ids = jnp.full((count,), -1, dtype=jnp.int32)
        block_slots = jnp.full((count,), -1, dtype=jnp.int32)
        global_cells = jnp.full((count, dimension), -1, dtype=jnp.int32)
        local_cells = jnp.full((count, dimension), -1, dtype=jnp.int32)
        composite_cells = jnp.full((count,), -1, dtype=jnp.int32)

        for level, (level_plan, metadata, global_shape) in enumerate(
            zip(
                self.topology.plan.levels,
                self.topology.levels,
                self.topology.plan.global_cell_shapes,
                strict=True,
            )
        ):
            shape = jnp.asarray(global_shape, dtype=jnp.int32)
            indices = jnp.floor(normalized * shape).astype(jnp.int32)
            indices = jnp.clip(indices, 0, shape - 1)
            block_shape = jnp.asarray(level_plan.block_shape, dtype=jnp.int32)
            logical = indices // block_shape
            lattice = self.topology.plan.block_lattice_shapes[level]
            strides = np.asarray(
                [prod(lattice[axis + 1 :]) for axis in range(dimension)],
                dtype=np.int32,
            )
            candidate_block_ids = (
                self.topology.plan.block_id_offsets[level]
                + jnp.sum(logical * jnp.asarray(strides), axis=1)
            ).astype(jnp.int32)
            lookup = metadata.lookup_block_ids(candidate_block_ids)
            candidate = particle_support & lookup.supported
            local = indices % block_shape
            local_strides = np.asarray(
                [prod(level_plan.block_shape[axis + 1 :]) for axis in range(dimension)],
                dtype=np.int32,
            )
            local_flat = jnp.sum(local * jnp.asarray(local_strides), axis=1)
            composite = (
                self.layout.cell_offsets[level]
                + lookup.group_slots * prod(level_plan.block_shape)
                + local_flat
            ).astype(jnp.int32)
            levels = jnp.where(candidate, level, levels)
            block_ids = jnp.where(candidate, candidate_block_ids, block_ids)
            block_slots = jnp.where(candidate, lookup.group_slots, block_slots)
            global_cells = jnp.where(candidate[:, None], indices, global_cells)
            local_cells = jnp.where(candidate[:, None], local, local_cells)
            composite_cells = jnp.where(candidate, composite, composite_cells)

        routed = (levels >= 0) | ~active
        successful = jnp.all(routed) & jnp.all(finite | ~active)
        support = active & routed
        return AMRParticleLevelAssignment(
            levels=levels,
            block_ids=block_ids,
            block_slots=block_slots,
            global_cell_indices=global_cells,
            local_cell_indices=local_cells,
            composite_cell_indices=composite_cells,
            active=active,
            support=support,
            successful=successful,
            epoch_id=self.topology.epoch.epoch_id,
            topology_id=self.topology.topology_id,
            partition_id=self.topology.partition_id,
        )

    def _require_assignment(self, assignment: AMRParticleLevelAssignment, /) -> None:
        if not isinstance(assignment, AMRParticleLevelAssignment) or (
            assignment.epoch_id != self.topology.epoch.epoch_id
            or assignment.topology_id != self.topology.topology_id
            or assignment.partition_id != self.topology.partition_id
        ):
            raise ValueError(
                "Particle assignment belongs to a different AMR topology epoch."
            )

    def deposit_density(
        self,
        assignment: AMRParticleLevelAssignment,
        masses: ArrayLike,
        /,
    ) -> AMRParticleDepositResult:
        self._require_assignment(assignment)
        mass = jnp.asarray(masses, dtype=self.layout.dtype)
        if mass.shape != assignment.levels.shape:
            raise ValueError("Particle masses must contain one scalar per assignment.")
        safe_cells = jnp.where(
            assignment.composite_cell_indices >= 0,
            assignment.composite_cell_indices,
            0,
        )
        leaf_support = assignment.support & self.layout.flat_leaf_mask[safe_cells]
        finite_mass = jnp.isfinite(mass) & (mass >= 0.0)
        deposited_values = jnp.where(leaf_support & finite_mass, mass, 0.0)
        flat_content = jnp.zeros((self.layout.cell_count,), dtype=self.layout.dtype)
        flat_content = flat_content.at[safe_cells].add(deposited_values)
        flat_density = jnp.where(
            self.layout.flat_leaf_mask,
            flat_content / self.layout.cell_measures,
            0.0,
        )
        content = self.layout.unflatten_cells(flat_content[:, None])
        density = self.layout.unflatten_cells(flat_density[:, None])
        source_mass = jnp.sum(jnp.where(assignment.active, mass, 0.0))
        deposited_mass = jnp.sum(flat_content)
        balance = deposited_mass - source_mass
        tolerance = (
            128.0
            * jnp.finfo(self.layout.dtype).eps
            * jnp.maximum(jnp.abs(source_mass), 1.0)
        )
        successful = (
            assignment.successful
            & jnp.all(finite_mass | ~assignment.active)
            & jnp.all(leaf_support | ~assignment.active)
            & jnp.isfinite(balance)
            & (jnp.abs(balance) <= tolerance)
        )
        return AMRParticleDepositResult(
            content=content,
            density=density,
            source_mass=source_mass,
            deposited_mass=deposited_mass,
            balance_defect=balance,
            support=leaf_support,
            successful=successful,
            epoch_id=self.topology.epoch.epoch_id,
            topology_id=self.topology.topology_id,
            partition_id=self.topology.partition_id,
        )

    def gather(
        self,
        assignment: AMRParticleLevelAssignment,
        fields: tuple[ArrayLike, ...],
        /,
    ) -> AMRParticleGatherResult:
        self._require_assignment(assignment)
        if not isinstance(fields, (tuple, list)) or len(fields) != len(
            self.layout.level_shapes
        ):
            raise ValueError("Composite gather fields need one array per AMR level.")
        arrays = tuple(jnp.asarray(value, dtype=self.layout.dtype) for value in fields)
        component_shape = arrays[0].shape[len(self.layout.level_shapes[0]) :]
        if any(
            array.shape[: len(level_shape)] != level_shape
            or array.shape[len(level_shape) :] != component_shape
            for array, level_shape in zip(arrays, self.layout.level_shapes, strict=True)
        ):
            raise ValueError("Composite gather fields have inconsistent level shapes.")
        flat = jnp.concatenate(
            tuple(
                array.reshape((mask.size,) + component_shape)
                for array, mask in zip(arrays, self.layout.leaf_mask, strict=True)
            ),
            axis=0,
        )
        safe_cells = jnp.where(
            assignment.composite_cell_indices >= 0,
            assignment.composite_cell_indices,
            0,
        )
        support = assignment.support & self.layout.flat_leaf_mask[safe_cells]
        values = flat[safe_cells]
        broadcast = support.reshape(support.shape + (1,) * len(component_shape))
        values = jnp.where(broadcast, values, 0.0)
        active = assignment.active.reshape(
            assignment.active.shape + (1,) * len(component_shape)
        )
        successful = (
            assignment.successful
            & jnp.all(support | ~assignment.active)
            & jnp.all(jnp.isfinite(values) | ~active)
        )
        return AMRParticleGatherResult(
            values=values,
            support=support,
            successful=successful,
            epoch_id=self.topology.epoch.epoch_id,
            topology_id=self.topology.topology_id,
            partition_id=self.topology.partition_id,
        )


class BlockAMRGravityResult(StrictModule):
    """Composite Poisson solution, cell acceleration, and ordinary solve evidence."""

    potential: tuple[Array, ...]
    acceleration: tuple[Array, ...]
    source: tuple[Array, ...]
    solve_result: LinearSolveResult
    mean_density: Array
    source_integral: Array
    interface_flux_conservation_defect: Array
    finite: Array
    successful: Array
    epoch_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)


class BlockAMRParticleGravityResult(StrictModule):
    """Deposited particle density and gathered composite gravity acceleration."""

    assignment: AMRParticleLevelAssignment
    deposited: AMRParticleDepositResult
    gravity: BlockAMRGravityResult
    gathered: AMRParticleGatherResult
    acceleration: Array
    net_force: Array
    successful: Array


class BlockAMRGravityPlan(StrictModule, NonTrainableState):
    """Newtonian gravity on an FV-owned composite block-AMR Poisson operator."""

    operator: PreparedCompositeAMRDiffusion
    routing: BlockAMRParticleRoutingPlan
    acceleration_layout: CompositeAMRCellLayout
    solve_policy: LinearSolvePolicy
    gravitational_constant: float = eqx.field(static=True)
    gravity_argument: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operator: PreparedCompositeAMRDiffusion,
        routing: BlockAMRParticleRoutingPlan,
        /,
        *,
        gravitational_constant: float = 1.0,
        gravity_argument: str | None = None,
        solve_policy: LinearSolvePolicy | None = None,
    ):
        if not isinstance(operator, PreparedCompositeAMRDiffusion):
            raise TypeError("operator must be PreparedCompositeAMRDiffusion.")
        if not isinstance(routing, BlockAMRParticleRoutingPlan):
            raise TypeError("routing must be BlockAMRParticleRoutingPlan.")
        layout = operator.layout
        if layout.component_shape or layout.layout_id != routing.layout.layout_id:
            raise ValueError(
                "Gravity and particle routing need one scalar composite layout."
            )
        if any(
            condition.kind != "periodic"
            for pair in operator.plan.boundaries
            for condition in pair
        ):
            raise ValueError(
                "Cosmological block-AMR gravity requires periodic boundaries."
            )
        coupling = float(gravitational_constant)
        argument = None if gravity_argument is None else str(gravity_argument)
        if not np.isfinite(coupling) or coupling <= 0.0 or argument == "":
            raise ValueError("Block-AMR gravitational coupling is invalid.")
        policy = (
            LinearSolvePolicy(
                ProjectedPCG(),
                tolerance=TolerancePolicy(
                    relative=1.0e-10,
                    absolute=1.0e-12,
                    max_steps=max(200, 4 * layout.space.size),
                ),
            )
            if solve_policy is None
            else solve_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("solve_policy must be LinearSolvePolicy or None.")
        dimension = len(layout.topology.plan.grid.shape)
        self.operator = operator
        self.routing = routing
        self.acceleration_layout = CompositeAMRCellLayout(
            layout.topology,
            component_shape=(dimension,),
            dtype=layout.dtype,
        )
        self.solve_policy = policy
        self.gravitational_constant = coupling
        self.gravity_argument = argument
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cosmological-block-amr-gravity",
                "operator": operator.operator_id,
                "routing": routing.plan_id,
                "gravitational_constant": coupling,
                "gravity_argument": argument,
                "linear_method": policy.method.name,
                "relative_tolerance": policy.tolerance.relative,
                "absolute_tolerance": policy.tolerance.absolute,
                "maximum_steps": policy.tolerance.max_steps,
            }
        )

    def _gravity(self, args: Any, dtype: Any, /) -> Array:
        value = (
            self.gravitational_constant
            if self.gravity_argument is None
            else args[self.gravity_argument]
        )
        coupling = jnp.asarray(value, dtype=dtype).reshape(())
        return eqx.error_if(
            coupling,
            ~jnp.isfinite(coupling) | (coupling <= 0.0),
            "Gravitational constant must be positive and finite.",
        )

    def _acceleration(self, potential: tuple[Array, ...], /) -> tuple[Array, ...]:
        layout = self.operator.layout
        routes = self.operator.plan.routes
        values = layout.flatten_cells(potential)[:, 0]
        dimension = len(layout.topology.plan.grid.shape)
        numerator = jnp.zeros((layout.cell_count, dimension), dtype=layout.dtype)
        denominator = jnp.zeros_like(numerator)
        distance = routes.edge_left_distance + routes.edge_right_distance
        face_acceleration = (
            values[routes.edge_left] - values[routes.edge_right]
        ) / distance
        for axis in range(dimension):
            selected = routes.edge_axis == axis
            area = jnp.where(selected, routes.edge_area, 0.0)
            contribution = area * face_acceleration
            numerator = numerator.at[routes.edge_left, axis].add(contribution)
            numerator = numerator.at[routes.edge_right, axis].add(contribution)
            denominator = denominator.at[routes.edge_left, axis].add(area)
            denominator = denominator.at[routes.edge_right, axis].add(area)
        safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
        acceleration = jnp.where(
            layout.flat_leaf_mask[:, None],
            numerator / safe_denominator,
            0.0,
        )
        return self.acceleration_layout.unflatten_cells(acceleration)

    def solve_density(
        self,
        density: tuple[ArrayLike, ...],
        args: Any = None,
        /,
    ) -> BlockAMRGravityResult:
        layout = self.operator.layout
        density_values = layout.zero_masked(density)
        coupling = self._gravity(args, layout.dtype)
        volume = jnp.sum(jnp.where(layout.flat_leaf_mask, layout.cell_measures, 0.0))
        mean_density = (
            layout.integral(density_values) / volume
            if self.operator.has_constant_nullspace
            else jnp.asarray(0.0, dtype=layout.dtype)
        )
        source = tuple(
            jnp.where(mask, -4.0 * jnp.pi * coupling * (value - mean_density), 0.0)
            for value, mask in zip(density_values, layout.leaf_mask, strict=True)
        )
        right_hand_side = self.operator.prepare_rhs(
            source,
            project=self.operator.has_constant_nullspace,
        )
        solved = solve(
            self.operator.linear_system(compatibility="project", gauge="project"),
            right_hand_side,
            policy=self.solve_policy,
        )
        potential = layout.zero_masked(solved.value)
        if self.operator.has_constant_nullspace:
            potential = self.operator.zero_mean_gauge(potential)
        acceleration = self._acceleration(potential)
        interface_left, interface_right = self.operator.interface_flux_contributions(
            potential
        )
        interface_defect = jnp.max(jnp.abs(interface_left + interface_right), initial=0.0)
        source_integral = layout.integral(right_hand_side)
        potential_finite = jnp.asarray(True)
        for value in potential:
            potential_finite = potential_finite & jnp.all(jnp.isfinite(value))
        acceleration_finite = jnp.asarray(True)
        for value in acceleration:
            acceleration_finite = acceleration_finite & jnp.all(jnp.isfinite(value))
        finite = (
            potential_finite
            & acceleration_finite
            & jnp.all(jnp.isfinite(source_integral))
            & jnp.isfinite(interface_defect)
        )
        successful = (
            solved.successful
            & jnp.all(solved.diagnostics.converged)
            & jnp.all(solved.diagnostics.finite)
            & finite
        )
        return BlockAMRGravityResult(
            potential=potential,
            acceleration=acceleration,
            source=right_hand_side,
            solve_result=solved,
            mean_density=mean_density,
            source_integral=source_integral,
            interface_flux_conservation_defect=interface_defect,
            finite=finite,
            successful=successful,
            epoch_id=layout.topology.epoch.epoch_id,
            topology_id=layout.topology.topology_id,
            partition_id=layout.topology.partition_id,
        )

    def particle_force(
        self,
        positions: ArrayLike,
        masses: ArrayLike,
        args: Any = None,
        /,
        *,
        active_mask: ArrayLike | None = None,
        background_density: tuple[ArrayLike, ...] | None = None,
    ) -> BlockAMRParticleGravityResult:
        assignment = self.routing.route(positions, active_mask=active_mask)
        deposited = self.routing.deposit_density(assignment, masses)
        density = deposited.density
        if background_density is not None:
            background = self.operator.layout.zero_masked(background_density)
            density = tuple(
                particle + ambient
                for particle, ambient in zip(density, background, strict=True)
            )
        gravity = self.solve_density(density, args)
        gathered = self.routing.gather(assignment, gravity.acceleration)
        mass = jnp.asarray(masses, dtype=gathered.values.dtype)
        net_force = jnp.sum(
            jnp.where(assignment.active, mass, 0.0)[:, None] * gathered.values,
            axis=0,
        )
        successful = (
            deposited.successful
            & gravity.successful
            & gathered.successful
            & jnp.all(jnp.isfinite(net_force))
        )
        return BlockAMRParticleGravityResult(
            assignment=assignment,
            deposited=deposited,
            gravity=gravity,
            gathered=gathered,
            acceleration=gathered.values,
            net_force=net_force,
            successful=successful,
        )


class AMREpochResult(StrictModule):
    """Atomic fixed-topology gas, gravity, routing, and particle commit."""

    runtime_state: BlockAMRRuntimeState
    particles: CosmologicalParticleState
    assignment: AMRParticleLevelAssignment
    runtime_accepted: Array
    flux_finite: Array
    gravity_successful: Array
    particle_routed: Array
    scale_consistent: Array
    successful: Array
    epoch_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)


class BlockAMREpochPlan(StrictModule, NonTrainableState):
    """Commit one solver-owned block-AMR interval and particle update atomically.

    A plan is bound to one immutable topology epoch. Regridding is activated
    between plans with a successor prepared runtime; it is never mixed into the
    differentiable numeric commit below.
    """

    runtime: PreparedBlockAMRRuntime
    routing: BlockAMRParticleRoutingPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedBlockAMRRuntime,
        routing: BlockAMRParticleRoutingPlan,
        /,
    ):
        if not isinstance(runtime, PreparedBlockAMRRuntime):
            raise TypeError("runtime must be PreparedBlockAMRRuntime.")
        if not isinstance(routing, BlockAMRParticleRoutingPlan):
            raise TypeError("routing must be BlockAMRParticleRoutingPlan.")
        topology = runtime.dynamics.topology
        if (
            topology.epoch.epoch_id != routing.topology.epoch.epoch_id
            or topology.topology_id != routing.topology.topology_id
            or topology.partition_id != routing.topology.partition_id
        ):
            raise ValueError(
                "Runtime and particle routing must share one topology epoch."
            )
        if runtime.plan.indicator is not None:
            raise ValueError(
                "Cosmological epoch commits require a fixed-topology runtime; "
                "activate regridding between epoch plans."
            )
        self.runtime = runtime
        self.routing = routing
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cosmological-block-amr-epoch",
                "runtime": runtime.prepared_id,
                "routing": routing.plan_id,
                "epoch": topology.epoch.epoch_id,
            }
        )

    def _require_runtime_state(self, state: BlockAMRRuntimeState, /) -> None:
        topology = self.runtime.dynamics.topology
        if not isinstance(state, BlockAMRRuntimeState) or (
            state.hierarchy_state.topology.epoch.epoch_id != topology.epoch.epoch_id
            or state.hierarchy_state.topology.topology_id != topology.topology_id
            or state.hierarchy_state.topology.partition_id != topology.partition_id
        ):
            raise ValueError(
                "Cosmological AMR state belongs to a different topology epoch."
            )

    @staticmethod
    def _select_hierarchy(
        accepted: Array,
        candidate: BlockHierarchyState,
        previous: BlockHierarchyState,
        /,
    ) -> BlockHierarchyState:
        levels = tuple(
            BlockLevelState(
                old.plan,
                old.metadata,
                jnp.where(accepted, new.values, old.values),
            )
            for new, old in zip(candidate.levels, previous.levels, strict=True)
        )
        return BlockHierarchyState(previous.topology, levels)

    def commit(
        self,
        previous_state: BlockAMRRuntimeState,
        advance: BlockAMRAdvanceResult,
        previous_particles: CosmologicalParticleState,
        candidate_particles: CosmologicalParticleState,
        gravity: BlockAMRGravityResult,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> AMREpochResult:
        self._require_runtime_state(previous_state)
        if not isinstance(advance, BlockAMRAdvanceResult) or (
            advance.prepared_id != self.runtime.prepared_id
        ):
            raise ValueError("Block-AMR advance result belongs to a different runtime.")
        self._require_runtime_state(advance.runtime_state)
        if (
            advance.topology_event_request is not None
            or advance.successor_runtime is not None
        ):
            raise ValueError(
                "Activate a topology successor before its cosmological epoch commit."
            )
        if not isinstance(
            previous_particles, CosmologicalParticleState
        ) or not isinstance(candidate_particles, CosmologicalParticleState):
            raise TypeError("Cosmological epoch commit requires particle states.")
        if not isinstance(gravity, BlockAMRGravityResult) or (
            gravity.epoch_id != self.routing.topology.epoch.epoch_id
            or gravity.topology_id != self.routing.topology.topology_id
            or gravity.partition_id != self.routing.topology.partition_id
        ):
            raise ValueError("Gravity result belongs to a different AMR topology epoch.")
        if (
            previous_particles.positions.shape != candidate_particles.positions.shape
            or previous_particles.canonical_momenta.shape
            != candidate_particles.canonical_momenta.shape
            or previous_particles.positions.shape
            != candidate_particles.canonical_momenta.shape
        ):
            raise ValueError(
                "Cosmological particle candidate changed fixed capacity or dimension."
            )

        assignment = self.routing.route(
            candidate_particles.positions,
            active_mask=active_mask,
        )
        flux_finite = jnp.isfinite(advance.composite_conservation_defect)
        for register in advance.flux_registers:
            flux_finite = (
                flux_finite
                & jnp.all(jnp.isfinite(register.coarse_flux))
                & jnp.all(jnp.isfinite(register.fine_flux))
                & jnp.isfinite(register.accumulated_time)
            )
        dtype = candidate_particles.scale_factor.dtype
        tolerance = 64.0 * jnp.finfo(dtype).eps
        previous_scale_consistent = jnp.abs(
            previous_particles.scale_factor
            - previous_state.time.astype(previous_particles.scale_factor.dtype)
        ) <= tolerance * jnp.maximum(jnp.abs(previous_particles.scale_factor), 1.0)
        candidate_scale_consistent = jnp.abs(
            candidate_particles.scale_factor
            - advance.runtime_state.time.astype(candidate_particles.scale_factor.dtype)
        ) <= tolerance * jnp.maximum(jnp.abs(candidate_particles.scale_factor), 1.0)
        scale_consistent = previous_scale_consistent & candidate_scale_consistent
        active = assignment.active[:, None]
        particles_finite = (
            jnp.all(jnp.isfinite(candidate_particles.positions) | ~active)
            & jnp.all(jnp.isfinite(candidate_particles.canonical_momenta) | ~active)
            & jnp.isfinite(candidate_particles.scale_factor)
        )
        runtime_accepted = jnp.asarray(advance.accepted, dtype=bool)
        successful = jnp.all(
            jnp.asarray(
                runtime_accepted
                & flux_finite
                & gravity.successful
                & assignment.successful
                & scale_consistent
                & particles_finite,
                dtype=bool,
            )
        )
        hierarchy = self._select_hierarchy(
            successful,
            advance.runtime_state.hierarchy_state,
            previous_state.hierarchy_state,
        )
        runtime_state = BlockAMRRuntimeState(
            hierarchy,
            previous_state.topology_journal,
            jnp.where(successful, advance.runtime_state.time, previous_state.time),
            accepted_step=jnp.where(
                successful,
                advance.runtime_state.accepted_step,
                previous_state.accepted_step,
            ),
            level_accepted_steps=jnp.where(
                successful,
                advance.runtime_state.level_accepted_steps,
                previous_state.level_accepted_steps,
            ),
            last_status=jnp.where(
                successful,
                advance.runtime_state.last_status,
                previous_state.last_status,
            ),
        )
        particles = CosmologicalParticleState(
            jnp.where(
                successful,
                candidate_particles.positions,
                previous_particles.positions,
            ),
            jnp.where(
                successful,
                candidate_particles.canonical_momenta,
                previous_particles.canonical_momenta,
            ),
            jnp.where(
                successful,
                candidate_particles.scale_factor,
                previous_particles.scale_factor,
            ),
        )
        selected_assignment = self.routing.route(
            particles.positions,
            active_mask=active_mask,
        )
        topology = self.routing.topology
        return AMREpochResult(
            runtime_state=runtime_state,
            particles=particles,
            assignment=selected_assignment,
            runtime_accepted=runtime_accepted,
            flux_finite=flux_finite,
            gravity_successful=gravity.successful,
            particle_routed=assignment.successful,
            scale_consistent=scale_consistent,
            successful=successful,
            epoch_id=topology.epoch.epoch_id,
            topology_id=topology.topology_id,
            partition_id=topology.partition_id,
        )


__all__ = [
    "AMREpochResult",
    "AMRParticleDepositResult",
    "AMRParticleGatherResult",
    "AMRParticleLevelAssignment",
    "BlockAMREpochPlan",
    "BlockAMRGravityPlan",
    "BlockAMRGravityResult",
    "BlockAMRParticleGravityResult",
    "BlockAMRParticleRoutingPlan",
]
