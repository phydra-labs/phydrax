#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._tensor_index import PreparedTensorIndexSpace, TensorIndexLayout
from ..spatial import SparseBlockTopologyPlan, SparseBlockTopologyState
from ._collision import (
    BGKCollisionPlan,
    collide_detailed,
    LatticeBoltzmannCollisionDiagnostics,
    LatticeBoltzmannCollisionPlan,
    macroscopic_raw_moments,
    prepare_lattice_boltzmann_collision,
    PreparedLatticeBoltzmannCollision,
    quadratic_equilibrium,
)
from ._lattice import LatticeBoltzmannVelocitySet
from ._precision import LatticeBoltzmannPrecisionPolicy


class SparseLatticeBoltzmannDiagnostics(StrictModule):
    """Compact-fluid conservation and admissibility evidence."""

    minimum_density: Array
    minimum_population: Array
    maximum_mach: Array
    total_mass: Array
    mass_defect: Array
    total_momentum: Array
    finite: Array


class SparseLatticeBoltzmannState(StrictModule):
    """Populations on one fixed sparse fluid topology."""

    populations: Array
    topology: SparseBlockTopologyState
    fluid_valid: Array
    prepared_id: str = eqx.field(static=True)


class SparseLatticeBoltzmannStepResult(StrictModule):
    """Transactional collide-stream result on a fixed sparse topology."""

    candidate_state: SparseLatticeBoltzmannState
    accepted_state: SparseLatticeBoltzmannState
    density: Array
    velocity: Array
    diagnostics: SparseLatticeBoltzmannDiagnostics
    collision_diagnostics: LatticeBoltzmannCollisionDiagnostics
    successful: Array


class SparseLatticeBoltzmannTransitionEvidence(StrictModule):
    """Conservative logical-cell alignment for one sparse geometry epoch."""

    retained_cells: Array
    activated_cells: Array
    retired_cells: Array
    activated_mass: Array
    retired_mass: Array
    successful: Array


class SparseLatticeBoltzmannTransitionResult(StrictModule):
    """Prepared sparse geometry and state after an explicit epoch transition."""

    prepared: PreparedSparseLatticeBoltzmann
    state: SparseLatticeBoltzmannState
    evidence: SparseLatticeBoltzmannTransitionEvidence


class SparseLatticeBoltzmannPlan(StrictModule, NonTrainableState):
    """Static sparse collide-stream plan over explicit fluid-cell IDs."""

    index_space: PreparedTensorIndexSpace
    velocity_set: LatticeBoltzmannVelocitySet
    collision: LatticeBoltzmannCollisionPlan
    precision: LatticeBoltzmannPrecisionPolicy
    block_shape: tuple[int, ...] = eqx.field(static=True)
    block_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        index_space: PreparedTensorIndexSpace,
        velocity_set: LatticeBoltzmannVelocitySet,
        block_shape: Sequence[int],
        block_capacity: int,
        /,
        *,
        collision: LatticeBoltzmannCollisionPlan | None = None,
        precision: LatticeBoltzmannPrecisionPolicy | None = None,
    ) -> None:
        if not isinstance(index_space, PreparedTensorIndexSpace):
            raise TypeError("index_space must be PreparedTensorIndexSpace.")
        if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
            raise TypeError("velocity_set must be LatticeBoltzmannVelocitySet.")
        if velocity_set.dimension != len(index_space.axis_names):
            raise ValueError("Velocity-set and tensor-index dimensions must match.")
        collision_ = BGKCollisionPlan() if collision is None else collision
        precision_ = LatticeBoltzmannPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, LatticeBoltzmannPrecisionPolicy):
            raise TypeError("precision must be LatticeBoltzmannPrecisionPolicy.")
        blocks = tuple(int(size) for size in block_shape)
        capacity = int(block_capacity)
        if len(blocks) != velocity_set.dimension or any(size <= 0 for size in blocks):
            raise ValueError("block_shape must match the lattice dimension.")
        if capacity <= 0:
            raise ValueError("block_capacity must be positive.")
        prepare_lattice_boltzmann_collision(collision_, velocity_set, precision_)
        self.index_space = index_space
        self.velocity_set = velocity_set
        self.collision = collision_
        self.precision = precision_
        self.block_shape = blocks
        self.block_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sparse-lattice-boltzmann-plan",
                "index_space": index_space.prepared_id,
                "lattice": velocity_set.lattice_id,
                "collision": collision_.collision_id,
                "precision": precision_.policy_id,
                "block_shape": list(blocks),
                "block_capacity": capacity,
                "boundary": "halfway-bounceback-on-missing-fluid-neighbor",
            }
        )

    def prepare(self, fluid_cell_ids: ArrayLike, /) -> PreparedSparseLatticeBoltzmann:
        return PreparedSparseLatticeBoltzmann(self, fluid_cell_ids)


class PreparedSparseLatticeBoltzmann(StrictModule, NonTrainableState):
    """Prepared static sparse fluid cells, streaming routes, and collision."""

    plan: SparseLatticeBoltzmannPlan
    layout: TensorIndexLayout
    topology_plan: SparseBlockTopologyPlan
    topology: SparseBlockTopologyState
    fluid_valid: Array
    stream_source_slots: Array
    stream_source_valid: Array
    collision: PreparedLatticeBoltzmannCollision
    coordinates: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SparseLatticeBoltzmannPlan,
        fluid_cell_ids: ArrayLike,
        /,
    ) -> None:
        if not isinstance(plan, SparseLatticeBoltzmannPlan):
            raise TypeError("plan must be SparseLatticeBoltzmannPlan.")
        cells = np.asarray(fluid_cell_ids)
        if cells.ndim != 1 or not np.issubdtype(cells.dtype, np.integer):
            raise TypeError("fluid_cell_ids must be a rank-1 integer array.")
        if cells.size == 0:
            raise ValueError("Sparse LBM requires at least one fluid cell.")
        if np.any(cells < 0) or np.any(cells >= plan.index_space.cells().size):
            raise ValueError("fluid_cell_ids contain out-of-domain cells.")
        if np.unique(cells).size != cells.size:
            raise ValueError("fluid_cell_ids must be unique.")
        cells = np.sort(cells.astype(np.int32))
        layout = plan.index_space.cells()
        topology_plan = SparseBlockTopologyPlan(
            plan.index_space,
            plan.block_shape,
            plan.block_capacity,
            layout=layout,
        )
        topology = topology_plan.build(
            jnp.asarray(cells),
            jnp.ones((cells.size,), dtype=bool),
            stable_site_ids=jnp.asarray(cells),
        )
        if not bool(np.asarray(topology.evidence.successful)):
            raise ValueError(
                "Sparse LBM fluid cells exceed block capacity or index support."
            )
        fluid_lookup = topology.lookup(jnp.asarray(cells))
        storage_capacity = topology_plan.storage_capacity
        fluid_valid = (
            jnp.zeros((storage_capacity,), dtype=bool)
            .at[fluid_lookup.storage_slots]
            .set(fluid_lookup.supported)
        )
        logical_ids = topology.logical_node_ids.reshape((-1,))
        storage_valid = topology.node_valid.reshape((-1,))
        integer, _ = layout.integer_coordinates(logical_ids)
        velocities = jnp.asarray(plan.velocity_set.velocities, dtype=jnp.int32)
        source_coordinates = integer[:, None, :] - velocities[None, :, :]
        source_in_domain = jnp.broadcast_to(
            storage_valid[:, None], source_coordinates.shape[:-1]
        )
        resolved_axes = []
        for axis, size in enumerate(layout.shape):
            coordinate = source_coordinates[..., axis]
            if plan.index_space.periodic_axes[axis]:
                coordinate = jnp.mod(coordinate, size)
            else:
                source_in_domain = source_in_domain & (
                    (coordinate >= 0) & (coordinate < size)
                )
                coordinate = jnp.clip(coordinate, 0, size - 1)
            resolved_axes.append(coordinate)
        resolved_source = jnp.stack(resolved_axes, axis=-1)
        source_ids, source_layout_valid = layout.flat_indices(resolved_source)
        source_lookup = topology.lookup(
            source_ids,
            source_in_domain & source_layout_valid,
        )
        source_slots = source_lookup.storage_slots
        source_fluid = (
            source_lookup.supported & fluid_valid[source_slots] & fluid_valid[:, None]
        )
        coordinates, coordinate_valid = layout.coordinates_at(logical_ids)
        coordinates = jnp.where(
            (storage_valid & coordinate_valid)[:, None], coordinates, 0.0
        )
        collision = prepare_lattice_boltzmann_collision(
            plan.collision,
            plan.velocity_set,
            plan.precision,
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-sparse-lattice-boltzmann",
                "plan": plan.plan_id,
                "fluid_cell_ids": cells.tolist(),
                "topology": topology_plan.plan_id,
                "collision": collision.prepared_id,
            }
        )
        self.plan = plan
        self.layout = layout
        self.topology_plan = topology_plan
        self.topology = topology
        self.fluid_valid = fluid_valid
        self.stream_source_slots = source_slots
        self.stream_source_valid = source_fluid
        self.collision = collision
        self.coordinates = coordinates
        self.prepared_id = prepared_id

    @property
    def storage_capacity(self) -> int:
        return self.topology_plan.storage_capacity

    @property
    def population_shape(self) -> tuple[int, int]:
        return self.storage_capacity, self.plan.velocity_set.population_count

    def initialize_state(
        self,
        density: ArrayLike,
        velocity: ArrayLike,
        /,
    ) -> SparseLatticeBoltzmannState:
        dtype = jnp.dtype(self.plan.precision.population_dtype)
        rho = jnp.asarray(density, dtype=dtype)
        if rho.shape == ():
            rho = jnp.broadcast_to(rho, (self.storage_capacity,))
        if rho.shape != (self.storage_capacity,):
            raise ValueError(
                "density must be scalar or have sparse storage-cell capacity."
            )
        velocity_value = jnp.asarray(velocity, dtype=dtype)
        dimension = self.plan.velocity_set.dimension
        if velocity_value.shape == (dimension,):
            velocity_value = jnp.broadcast_to(
                velocity_value, (self.storage_capacity, dimension)
            )
        if velocity_value.shape != (self.storage_capacity, dimension):
            raise ValueError(
                "velocity must be one vector or one vector per sparse storage cell."
            )
        rho = jnp.where(self.fluid_valid, rho, 1.0)
        velocity_value = jnp.where(self.fluid_valid[:, None], velocity_value, 0.0)
        populations = quadratic_equilibrium(
            rho,
            velocity_value,
            self.plan.velocity_set,
            self.plan.precision,
        )
        populations = self.plan.precision.population(
            jnp.where(self.fluid_valid[:, None], populations, 0.0)
        )
        return SparseLatticeBoltzmannState(
            populations=populations,
            topology=self.topology,
            fluid_valid=self.fluid_valid,
            prepared_id=self.prepared_id,
        )

    def macroscopic(self, state: SparseLatticeBoltzmannState, /) -> tuple[Array, Array]:
        self._validate_state(state)
        density, momentum = macroscopic_raw_moments(
            state.populations,
            self.plan.velocity_set,
            self.plan.precision,
        )
        safe_density = jnp.where(self.fluid_valid & (density > 0.0), density, 1.0)
        velocity = momentum / safe_density[:, None]
        return (
            jnp.where(self.fluid_valid, density, 0.0),
            jnp.where(self.fluid_valid[:, None], velocity, 0.0),
        )

    def step(
        self,
        state: SparseLatticeBoltzmannState,
        relaxation_rate: ArrayLike,
        /,
    ) -> SparseLatticeBoltzmannStepResult:
        self._validate_state(state)
        dtype = state.populations.dtype
        rate = jnp.asarray(relaxation_rate, dtype=dtype)
        if rate.shape == ():
            rate = jnp.broadcast_to(rate, (self.storage_capacity,))
        if rate.shape != (self.storage_capacity,):
            raise ValueError(
                "relaxation_rate must be scalar or have sparse storage-cell capacity."
            )
        rate = eqx.error_if(
            rate,
            jnp.any(
                self.fluid_valid & (~jnp.isfinite(rate) | (rate <= 0.0) | (rate >= 2.0))
            ),
            "Fluid relaxation rates must lie in (0, 2).",
        )
        density_before, velocity_before = self.macroscopic(state)
        safe_density = jnp.where(self.fluid_valid, density_before, 1.0)
        safe_velocity = jnp.where(self.fluid_valid[:, None], velocity_before, 0.0)
        equilibrium = quadratic_equilibrium(
            safe_density,
            safe_velocity,
            self.plan.velocity_set,
            self.plan.precision,
        )
        safe_populations = jnp.where(
            self.fluid_valid[:, None], state.populations, equilibrium
        )
        collision = collide_detailed(
            self.collision,
            safe_populations,
            equilibrium,
            jnp.zeros_like(safe_populations),
            jnp.where(self.fluid_valid, rate, 1.0),
            safe_velocity,
            self.plan.velocity_set,
            self.plan.precision,
        )
        post_collision = collision.candidate_populations
        q = self.plan.velocity_set.population_count
        directions = jnp.arange(q, dtype=jnp.int32)[None, :]
        streamed = post_collision[self.stream_source_slots, directions]
        reflected = post_collision[:, self.plan.velocity_set.opposite]
        candidate_populations = jnp.where(
            self.stream_source_valid,
            streamed,
            reflected,
        )
        candidate_populations = self.plan.precision.population(
            jnp.where(self.fluid_valid[:, None], candidate_populations, 0.0)
        )
        candidate_state = SparseLatticeBoltzmannState(
            populations=candidate_populations,
            topology=self.topology,
            fluid_valid=self.fluid_valid,
            prepared_id=self.prepared_id,
        )
        density, velocity = self.macroscopic(candidate_state)
        finite = (
            jnp.all(
                jnp.where(
                    self.fluid_valid[:, None], jnp.isfinite(candidate_populations), True
                )
            )
            & jnp.all(jnp.where(self.fluid_valid, density > 0.0, True))
            & jnp.all(jnp.isfinite(velocity))
        )
        successful = collision.successful & finite
        accepted_populations = jnp.where(
            successful, candidate_populations, state.populations
        )
        accepted_state = SparseLatticeBoltzmannState(
            populations=accepted_populations,
            topology=self.topology,
            fluid_valid=self.fluid_valid,
            prepared_id=self.prepared_id,
        )
        mass_before = jnp.sum(jnp.where(self.fluid_valid, density_before, 0.0))
        mass_after = jnp.sum(jnp.where(self.fluid_valid, density, 0.0))
        momentum = ein.contract(
            "sq,qd->d",
            jnp.where(self.fluid_valid[:, None], candidate_populations, 0.0),
            jnp.asarray(self.plan.velocity_set.velocities, dtype=dtype),
        )
        sound_speed = jnp.sqrt(
            jnp.asarray(self.plan.velocity_set.sound_speed_squared, dtype=dtype)
        )
        speed = jnp.sqrt(jnp.sum(velocity * velocity, axis=-1))
        diagnostics = SparseLatticeBoltzmannDiagnostics(
            minimum_density=jnp.min(
                jnp.where(self.fluid_valid, density, jnp.inf), initial=jnp.inf
            ),
            minimum_population=jnp.min(
                jnp.where(self.fluid_valid[:, None], candidate_populations, jnp.inf),
                initial=jnp.inf,
            ),
            maximum_mach=jnp.max(
                jnp.where(self.fluid_valid, speed / sound_speed, 0.0), initial=0.0
            ),
            total_mass=mass_after,
            mass_defect=mass_after - mass_before,
            total_momentum=momentum,
            finite=finite,
        )
        return SparseLatticeBoltzmannStepResult(
            candidate_state=candidate_state,
            accepted_state=accepted_state,
            density=density,
            velocity=velocity,
            diagnostics=diagnostics,
            collision_diagnostics=collision.diagnostics,
            successful=successful,
        )

    def refresh_geometry(
        self,
        state: SparseLatticeBoltzmannState,
        fluid_cell_ids: ArrayLike,
        /,
        *,
        activated_density: ArrayLike = 1.0,
        activated_velocity: ArrayLike | None = None,
    ) -> SparseLatticeBoltzmannTransitionResult:
        """Prepare a new fixed topology and align populations by logical cell ID."""
        self._validate_state(state)
        candidate = self.plan.prepare(fluid_cell_ids)
        velocity = (
            jnp.zeros((self.plan.velocity_set.dimension,))
            if activated_velocity is None
            else jnp.asarray(activated_velocity)
        )
        initialized = candidate.initialize_state(activated_density, velocity)
        new_logical_ids = candidate.topology.logical_node_ids.reshape((-1,))
        old_lookup = self.topology.lookup(
            new_logical_ids,
            candidate.fluid_valid,
        )
        retained = (
            candidate.fluid_valid
            & old_lookup.supported
            & self.fluid_valid[old_lookup.storage_slots]
        )
        aligned_populations = jnp.where(
            retained[:, None],
            state.populations[old_lookup.storage_slots],
            initialized.populations,
        )
        old_logical_ids = self.topology.logical_node_ids.reshape((-1,))
        new_lookup = candidate.topology.lookup(
            old_logical_ids,
            self.fluid_valid,
        )
        old_retained = (
            self.fluid_valid
            & new_lookup.supported
            & candidate.fluid_valid[new_lookup.storage_slots]
        )
        retired = self.fluid_valid & ~old_retained
        activated = candidate.fluid_valid & ~retained
        topology_changed = jnp.any(activated) | jnp.any(retired)
        next_topology = eqx.tree_at(
            lambda topology: topology.generation,
            candidate.topology,
            self.topology.generation + topology_changed.astype(jnp.int32),
        )
        candidate = eqx.tree_at(
            lambda prepared: prepared.topology,
            candidate,
            next_topology,
        )
        next_state = SparseLatticeBoltzmannState(
            populations=aligned_populations,
            topology=next_topology,
            fluid_valid=candidate.fluid_valid,
            prepared_id=candidate.prepared_id,
        )
        evidence = SparseLatticeBoltzmannTransitionEvidence(
            retained_cells=jnp.sum(retained, dtype=jnp.int32),
            activated_cells=jnp.sum(activated, dtype=jnp.int32),
            retired_cells=jnp.sum(retired, dtype=jnp.int32),
            activated_mass=jnp.sum(
                jnp.where(
                    activated[:, None],
                    initialized.populations,
                    0.0,
                )
            ),
            retired_mass=jnp.sum(jnp.where(retired[:, None], state.populations, 0.0)),
            successful=jnp.asarray(True),
        )
        return SparseLatticeBoltzmannTransitionResult(
            prepared=candidate,
            state=next_state,
            evidence=evidence,
        )

    def _validate_state(self, state: SparseLatticeBoltzmannState, /) -> None:
        if not isinstance(state, SparseLatticeBoltzmannState):
            raise TypeError("state must be SparseLatticeBoltzmannState.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("Sparse LBM state belongs to another prepared plan.")
        if state.populations.shape != self.population_shape:
            raise ValueError("Sparse LBM population shape is incompatible.")


__all__ = [
    "PreparedSparseLatticeBoltzmann",
    "SparseLatticeBoltzmannDiagnostics",
    "SparseLatticeBoltzmannPlan",
    "SparseLatticeBoltzmannTransitionEvidence",
    "SparseLatticeBoltzmannTransitionResult",
    "SparseLatticeBoltzmannState",
    "SparseLatticeBoltzmannStepResult",
]
