#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._tensor_index import PreparedTensorIndexSpace
from ..particle import ParticleDiscretization, ParticlePrecisionPolicy
from ..spatial import SparseBlockTopologyPlan, SparseBlockTopologyState
from ..splatting import (
    AbstractStructuredSplatAssignment,
    ParticleGridSplatBudget,
    ParticleGridSplatPlan,
    ParticleGridSplatState,
    PreparedParticleGridSplat,
    SplatExecutionPolicy,
    TensorBSplineSplatAssignment,
)
from ._types import FLIPGridToParticleResult, FLIPParticleToGridResult


class SparseFLIPTransferState(StrictModule):
    """Logical particle routes plus complete compact cell/face topologies."""

    cell_routes: ParticleGridSplatState
    face_routes: tuple[ParticleGridSplatState, ...]
    cell_topology: SparseBlockTopologyState
    face_topologies: tuple[SparseBlockTopologyState, ...]
    successful: Array
    transfer_id: str = eqx.field(static=True)


class SparseFLIPParticleTransferPlan(StrictModule, NonTrainableState):
    """Matched compact cell/face particle transfers on a tensor index space."""

    index_space: PreparedTensorIndexSpace
    cell_topology: SparseBlockTopologyPlan
    face_topologies: tuple[SparseBlockTopologyPlan, ...]
    assignment: AbstractStructuredSplatAssignment
    execution: SplatExecutionPolicy
    precision: ParticlePrecisionPolicy
    budget: ParticleGridSplatBudget
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        index_space: PreparedTensorIndexSpace,
        cell_topology: SparseBlockTopologyPlan,
        face_topologies: Sequence[SparseBlockTopologyPlan],
        /,
        *,
        assignment: AbstractStructuredSplatAssignment | None = None,
        execution: SplatExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
        budget: ParticleGridSplatBudget | None = None,
    ) -> None:
        if not isinstance(index_space, PreparedTensorIndexSpace):
            raise TypeError("index_space must be PreparedTensorIndexSpace.")
        faces = tuple(face_topologies)
        if len(faces) != len(index_space.axis_names) or not all(
            isinstance(value, SparseBlockTopologyPlan) for value in faces
        ):
            raise TypeError("One sparse face topology is required per tensor axis.")
        if not isinstance(cell_topology, SparseBlockTopologyPlan):
            raise TypeError("cell_topology must be SparseBlockTopologyPlan.")
        if cell_topology.layout.layout_id != index_space.cells().layout_id:
            raise ValueError("cell_topology must use the index-space cell layout.")
        for axis, topology in zip(index_space.axis_names, faces, strict=True):
            if topology.layout.layout_id != index_space.faces(axis).layout_id:
                raise ValueError("Each face topology must use its axis face layout.")
        assignment_ = (
            TensorBSplineSplatAssignment(1) if assignment is None else assignment
        )
        execution_ = SplatExecutionPolicy() if execution is None else execution
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        budget_ = ParticleGridSplatBudget() if budget is None else budget
        if not isinstance(assignment_, AbstractStructuredSplatAssignment):
            raise TypeError("assignment must be a structured splat assignment.")
        self.index_space = index_space
        self.cell_topology = cell_topology
        self.face_topologies = faces
        self.assignment = assignment_
        self.execution = execution_
        self.precision = precision_
        self.budget = budget_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sparse-flip-particle-transfer-plan",
                "index_space": index_space.prepared_id,
                "cell_topology": cell_topology.plan_id,
                "face_topologies": [value.plan_id for value in faces],
                "assignment": assignment_.assignment_id,
                "execution": execution_.policy_id,
                "precision": precision_.policy_id,
                "budget": budget_.budget_id,
            }
        )

    def prepare(
        self, particles: ParticleDiscretization, /
    ) -> PreparedSparseFLIPParticleTransfer:
        return PreparedSparseFLIPParticleTransfer(self, particles)


class PreparedSparseFLIPParticleTransfer(StrictModule, NonTrainableState):
    """Prepared compact FLIP P2G/G2P transfer without a dense MAC allocation."""

    plan: SparseFLIPParticleTransferPlan
    particles: ParticleDiscretization
    cell: PreparedParticleGridSplat
    faces: tuple[PreparedParticleGridSplat, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SparseFLIPParticleTransferPlan,
        particles: ParticleDiscretization,
        /,
    ) -> None:
        if not isinstance(plan, SparseFLIPParticleTransferPlan):
            raise TypeError("plan must be SparseFLIPParticleTransferPlan.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be ParticleDiscretization.")
        if particles.ambient_dimension != len(plan.index_space.axis_names):
            raise ValueError("FLIP particles and tensor index dimensions must match.")

        def prepared_for(layout):
            return ParticleGridSplatPlan(
                plan.index_space,
                location=plan.index_space.location(layout.offsets),
                assignment=plan.assignment,
                boundary="reject",
                execution=plan.execution,
                precision=plan.precision,
                budget=plan.budget,
            ).prepare(particles)

        cell = prepared_for(plan.index_space.cells())
        faces = tuple(
            prepared_for(plan.index_space.faces(axis))
            for axis in plan.index_space.axis_names
        )
        self.plan = plan
        self.particles = particles
        self.cell = cell
        self.faces = faces
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-sparse-flip-particle-transfer",
                "plan": plan.plan_id,
                "particles": particles.prepared_id,
                "cell": cell.prepared_id,
                "faces": [value.prepared_id for value in faces],
            }
        )

    @property
    def dimension(self) -> int:
        return self.particles.ambient_dimension

    def build(
        self,
        position: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        previous: SparseFLIPTransferState | None = None,
    ) -> SparseFLIPTransferState:
        cell_routes = self.cell.build(position, active_mask=active_mask)
        face_routes = tuple(
            value.build(position, active_mask=active_mask) for value in self.faces
        )
        if previous is None:
            cell_topology = self.plan.cell_topology.build(
                cell_routes.stencil.indices.reshape((-1,)),
                cell_routes.stencil.valid.reshape((-1,)),
            )
            face_topologies = tuple(
                topology.build(
                    routes.stencil.indices.reshape((-1,)),
                    routes.stencil.valid.reshape((-1,)),
                )
                for topology, routes in zip(
                    self.plan.face_topologies, face_routes, strict=True
                )
            )
        else:
            if not isinstance(previous, SparseFLIPTransferState):
                raise TypeError("previous must be SparseFLIPTransferState or None.")
            cell_topology = self.plan.cell_topology.refresh(
                previous.cell_topology,
                cell_routes.stencil.indices.reshape((-1,)),
                cell_routes.stencil.valid.reshape((-1,)),
            ).candidate
            face_topologies = tuple(
                topology.refresh(
                    old,
                    routes.stencil.indices.reshape((-1,)),
                    routes.stencil.valid.reshape((-1,)),
                ).candidate
                for topology, old, routes in zip(
                    self.plan.face_topologies,
                    previous.face_topologies,
                    face_routes,
                    strict=True,
                )
            )
        successful = (
            cell_routes.successful
            & cell_topology.evidence.successful
            & jnp.all(
                jnp.stack(
                    tuple(
                        routes.successful & topology.evidence.successful
                        for routes, topology in zip(
                            face_routes, face_topologies, strict=True
                        )
                    )
                )
            )
        )
        return SparseFLIPTransferState(
            cell_routes=cell_routes,
            face_routes=face_routes,
            cell_topology=cell_topology,
            face_topologies=face_topologies,
            successful=successful,
            transfer_id=self.prepared_id,
        )

    def build_fixed(
        self,
        position: ArrayLike,
        topology: SparseFLIPTransferState,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> SparseFLIPTransferState:
        """Build numeric routes against one already allocated topology."""
        if not isinstance(topology, SparseFLIPTransferState):
            raise TypeError("topology must be SparseFLIPTransferState.")
        cell_routes = self.cell.build(position, active_mask=active_mask)
        face_routes = tuple(
            value.build(position, active_mask=active_mask) for value in self.faces
        )
        cell_stencil = self._mapped_stencil(
            self.cell, cell_routes, topology.cell_topology
        )
        face_stencils = tuple(
            self._mapped_stencil(transfer, routes, state)
            for transfer, routes, state in zip(
                self.faces,
                face_routes,
                topology.face_topologies,
                strict=True,
            )
        )
        successful = (
            topology.successful
            & cell_routes.successful
            & jnp.all(~cell_routes.stencil.valid | cell_stencil.valid)
            & jnp.all(
                jnp.stack(
                    tuple(
                        routes.successful & jnp.all(~routes.stencil.valid | stencil.valid)
                        for routes, stencil in zip(
                            face_routes, face_stencils, strict=True
                        )
                    )
                )
            )
        )
        return SparseFLIPTransferState(
            cell_routes=cell_routes,
            face_routes=face_routes,
            cell_topology=topology.cell_topology,
            face_topologies=topology.face_topologies,
            successful=successful,
            transfer_id=self.prepared_id,
        )

    def particle_to_grid(
        self,
        state: SparseFLIPTransferState,
        velocity: ArrayLike,
        reference_density: ArrayLike,
        /,
        *,
        masses: ArrayLike | None = None,
    ) -> FLIPParticleToGridResult:
        self._validate_state(state)
        values = jnp.asarray(velocity, dtype=self.particles.safe_masses.dtype)
        expected = (self.particles.capacity, self.dimension)
        if values.shape != expected:
            raise ValueError(f"velocity must have shape {expected}.")
        density = jnp.asarray(reference_density, dtype=values.dtype).reshape(())
        density = eqx.error_if(
            density,
            ~jnp.isfinite(density) | (density <= 0.0),
            "reference_density must be positive and finite.",
        )
        masses_ = (
            self.particles.masses.astype(values.dtype)
            if masses is None
            else jnp.asarray(masses, dtype=values.dtype)
        )
        if masses_.shape != (self.particles.capacity,):
            raise ValueError("masses must have particle-capacity shape.")
        particle_volume = masses_ / density
        cell_stencil = self._mapped_stencil(
            self.cell, state.cell_routes, state.cell_topology
        )
        cell_measure = self._measure(
            self.plan.cell_topology, state.cell_topology, values.dtype
        )
        cell = self.cell.deposit_content_mapped(
            state.cell_routes,
            particle_volume,
            cell_stencil,
            cell_measure,
        )
        cell_valid = state.cell_topology.node_valid.reshape((-1,))
        liquid_fraction = jnp.where(
            cell_valid,
            cell.content / jnp.where(cell_valid, cell_measure, 1.0),
            0.0,
        )
        face_mass = []
        face_momentum = []
        face_velocity = []
        face_support = []
        momentum_defect = jnp.asarray(0.0, dtype=values.dtype)
        successful = state.successful & cell.successful
        for axis, (transfer, routes, topology_plan, topology) in enumerate(
            zip(
                self.faces,
                state.face_routes,
                self.plan.face_topologies,
                state.face_topologies,
                strict=True,
            )
        ):
            stencil = self._mapped_stencil(transfer, routes, topology)
            measure = self._measure(topology_plan, topology, values.dtype)
            payload = jnp.stack((masses_, masses_ * values[:, axis]), axis=-1)
            result = transfer.deposit_content_mapped(
                routes,
                payload,
                stencil,
                measure,
            )
            mass = result.content[..., 0]
            momentum = result.content[..., 1]
            scale = jnp.maximum(jnp.max(jnp.abs(mass), initial=0.0), 1.0)
            tolerance = jnp.finfo(mass.dtype).eps * max(16, transfer.route_count) * scale
            support = topology.node_valid.reshape((-1,)) & (mass > tolerance)
            velocity_component = jnp.where(
                support, momentum / jnp.where(support, mass, 1.0), 0.0
            )
            face_mass.append(mass)
            face_momentum.append(momentum)
            face_velocity.append(velocity_component)
            face_support.append(support)
            momentum_defect = jnp.maximum(
                momentum_defect, result.balance.maximum_absolute_balance_defect
            )
            successful = successful & result.successful
        finite = jnp.all(jnp.isfinite(liquid_fraction)) & jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in face_velocity))
        )
        return FLIPParticleToGridResult(
            cell.content,
            liquid_fraction,
            tuple(face_mass),
            tuple(face_momentum),
            tuple(face_velocity),
            tuple(face_support),
            cell.balance.maximum_absolute_balance_defect,
            momentum_defect,
            finite,
            successful & finite,
            "",
            self.prepared_id,
        )

    def grid_to_particle(
        self,
        state: SparseFLIPTransferState,
        old_velocity: Sequence[ArrayLike],
        new_velocity: Sequence[ArrayLike],
        /,
    ) -> FLIPGridToParticleResult:
        self._validate_state(state)
        old = tuple(jnp.asarray(value) for value in old_velocity)
        new = tuple(jnp.asarray(value) for value in new_velocity)
        if len(old) != self.dimension or len(new) != self.dimension:
            raise ValueError("One old/new compact face field is required per axis.")
        pic = []
        increment = []
        supports = []
        for transfer, routes, topology, previous, current in zip(
            self.faces,
            state.face_routes,
            state.face_topologies,
            old,
            new,
            strict=True,
        ):
            stencil = self._mapped_stencil(transfer, routes, topology)
            pic_result = transfer.gather_mapped(routes, current, stencil)
            delta_result = transfer.gather_mapped(routes, current - previous, stencil)
            pic.append(pic_result.values)
            increment.append(delta_result.values)
            supports.append(pic_result.support & delta_result.support)
        pic_values = jnp.stack(tuple(pic), axis=-1)
        increments = jnp.stack(tuple(increment), axis=-1)
        support = jnp.all(jnp.stack(tuple(supports), axis=-1), axis=-1)
        finite = jnp.all(jnp.isfinite(pic_values)) & jnp.all(jnp.isfinite(increments))
        return FLIPGridToParticleResult(
            pic_values,
            increments,
            support,
            finite,
            state.successful & finite,
            "",
            self.prepared_id,
        )

    @staticmethod
    def _measure(
        plan: SparseBlockTopologyPlan,
        topology: SparseBlockTopologyState,
        dtype,
    ) -> Array:
        logical = topology.logical_node_ids.reshape((-1,))
        valid = topology.node_valid.reshape((-1,))
        measure, supported = plan.layout.measure_at(logical)
        return jnp.where(valid & supported, measure.astype(dtype), 1.0)

    @staticmethod
    def _mapped_stencil(
        transfer: PreparedParticleGridSplat,
        routes: ParticleGridSplatState,
        topology: SparseBlockTopologyState,
    ):
        lookup = topology.lookup(routes.stencil.indices, routes.stencil.valid)
        from ..._interpolation import GatherStencil

        return GatherStencil(
            indices=lookup.storage_slots,
            weights=routes.stencil.weights,
            source_size=topology.plan.storage_capacity,
            valid=routes.stencil.valid & lookup.supported,
            support=routes.stencil.support & jnp.all(lookup.supported, axis=-1),
            case_shape=routes.stencil.case_shape,
        )

    def _validate_state(self, state: SparseFLIPTransferState, /) -> None:
        if not isinstance(state, SparseFLIPTransferState):
            raise TypeError("state must be SparseFLIPTransferState.")
        if state.transfer_id != self.prepared_id:
            raise ValueError("Sparse FLIP state belongs to another prepared transfer.")


__all__ = [
    "PreparedSparseFLIPParticleTransfer",
    "SparseFLIPParticleTransferPlan",
    "SparseFLIPTransferState",
]
