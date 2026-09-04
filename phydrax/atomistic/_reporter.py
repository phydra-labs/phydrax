#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from ._frame import (
    AbstractAtomisticTrajectorySinkPlan,
    AtomisticFrame,
    AtomisticFrameFields,
)
from ._sites import AtomisticSiteDomain


class AtomisticReporterPlan(StrictModule, NonTrainableState):
    sink: AbstractAtomisticTrajectorySinkPlan
    stride: int = eqx.field(static=True)
    fields: AtomisticFrameFields = eqx.field(static=True)
    coordinate_domain: AtomisticSiteDomain = eqx.field(static=True)
    reporter_id: str = eqx.field(static=True)

    def __init__(
        self,
        sink,
        /,
        *,
        stride: int = 1,
        fields: AtomisticFrameFields = AtomisticFrameFields.POSITIONS
        | AtomisticFrameFields.CELL
        | AtomisticFrameFields.ENERGY,
        coordinate_domain: AtomisticSiteDomain = AtomisticSiteDomain.DOF_ATOMS,
    ):
        if not isinstance(sink, AbstractAtomisticTrajectorySinkPlan):
            raise TypeError("sink must be an atomistic trajectory sink plan.")
        stride_ = int(stride)
        if stride_ <= 0:
            raise ValueError("Reporter stride must be positive.")
        self.sink = sink
        self.stride = stride_
        self.fields = fields
        self.coordinate_domain = coordinate_domain
        self.reporter_id = canonical_fingerprint(
            {
                "kind": "atomistic-reporter",
                "sink": sink.sink_id,
                "stride": stride_,
                "fields": int(fields),
                "domain": coordinate_domain.value,
            }
        )

    def frame(
        self, dynamics: PreparedAtomisticDynamics, state: AtomisticDynamicsState, /
    ) -> AtomisticFrame:
        if self.coordinate_domain is AtomisticSiteDomain.INTERACTION_SITES:
            site_state = dynamics.interaction_sites(state)
            positions = site_state.positions
            ids = dynamics.system.coordinate_map.plan.sites.site_ids
            velocity = momentum = force = images = None
        else:
            positions = state.kinematics.positions
            ids = dynamics.system.plan.particle_ids
            velocity = (
                dynamics.velocity(state)
                if self.fields & AtomisticFrameFields.VELOCITIES
                else None
            )
            momentum = (
                state.kinematics.momenta
                if self.fields & AtomisticFrameFields.MOMENTA
                else None
            )
            force = (
                state.force.forces if self.fields & AtomisticFrameFields.FORCES else None
            )
            images = (
                state.kinematics.image_counts
                if self.fields & AtomisticFrameFields.IMAGES
                else None
            )
        return AtomisticFrame(
            state.time,
            state.step_index,
            positions,
            ids,
            velocities=velocity,
            momenta=momentum,
            forces=force,
            cell_vectors=state.cell_vectors
            if self.fields & AtomisticFrameFields.CELL and state.cell_vectors.size
            else None,
            image_counts=images,
            energy=jnp.stack(
                (
                    state.energy.kinetic_energy,
                    state.energy.potential_energy,
                    state.energy.total_energy,
                )
            )
            if self.fields & AtomisticFrameFields.ENERGY
            else None,
            valid=state.last_status == 0,
            coordinate_domain=self.coordinate_domain,
            system_id=dynamics.system.prepared_id,
            topology_id=dynamics.system.topology.topology_id,
            units=dynamics.system.plan.units,
            source_id=canonical_fingerprint(
                {
                    "kind": "atomistic-frame",
                    "dynamics": dynamics.prepared_id,
                    "step": int(state.step_index),
                }
            ),
        )

    def write_states(
        self, dynamics: PreparedAtomisticDynamics, states, /, *, append: bool = False
    ) -> int:
        count = 0
        with self.sink.open(append=append) as writer:
            for state in states:
                if int(state.step_index) % self.stride == 0:
                    writer.write(self.frame(dynamics, state))
                    count += 1
        return count


__all__ = ["AtomisticReporterPlan"]
