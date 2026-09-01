#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._structured_cochain import StructuredCochainBridge
from ..particle import ParticlePrecisionPolicy, PreparedChargedParticles
from ..splatting import (
    AbstractStructuredSplatAssignment,
    MultilinearSplatAssignment,
    ParticleGridSplatBudget,
    ParticleGridSplatPlan,
    PreparedParticleGridSplat,
    SplatExecutionPolicy,
)
from ._types import (
    PICChargeDepositResult,
    PICFieldGatherResult,
    PICTransferState,
)


class PICParticleCochainTransferPlan(StrictModule, NonTrainableState):
    """Bind charged particles to exact structured cochain entity locations."""

    bridge: StructuredCochainBridge
    assignment: AbstractStructuredSplatAssignment
    execution: SplatExecutionPolicy
    precision: ParticlePrecisionPolicy
    budget: ParticleGridSplatBudget
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        assignment: AbstractStructuredSplatAssignment | None = None,
        execution: SplatExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
        budget: ParticleGridSplatBudget | None = None,
    ):
        if not isinstance(bridge, StructuredCochainBridge):
            raise TypeError("bridge must be StructuredCochainBridge.")
        assignment_ = MultilinearSplatAssignment() if assignment is None else assignment
        execution_ = SplatExecutionPolicy() if execution is None else execution
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        budget_ = ParticleGridSplatBudget() if budget is None else budget
        if not isinstance(assignment_, AbstractStructuredSplatAssignment):
            raise TypeError("assignment must be a structured splat assignment.")
        self.bridge = bridge
        self.assignment = assignment_
        self.execution = execution_
        self.precision = precision_
        self.budget = budget_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pic-particle-cochain-transfer-plan",
                "bridge": bridge.bridge_id,
                "assignment": assignment_.assignment_id,
                "execution": execution_.policy_id,
                "precision": precision_.policy_id,
                "budget": budget_.budget_id,
            }
        )

    def prepare(
        self, species: PreparedChargedParticles, /
    ) -> PreparedPICParticleCochainTransfer:
        return PreparedPICParticleCochainTransfer(self, species)


class PreparedPICParticleCochainTransfer(StrictModule, NonTrainableState):
    """Prepared endpoint charge deposition and physical E/B gathering."""

    plan: PICParticleCochainTransferPlan
    species: PreparedChargedParticles
    charge: PreparedParticleGridSplat
    electric: tuple[PreparedParticleGridSplat, ...]
    magnetic: tuple[PreparedParticleGridSplat, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self, plan: PICParticleCochainTransferPlan, species: PreparedChargedParticles, /
    ):
        if not isinstance(plan, PICParticleCochainTransferPlan):
            raise TypeError("plan must be PICParticleCochainTransferPlan.")
        if not isinstance(species, PreparedChargedParticles):
            raise TypeError("species must be PreparedChargedParticles.")
        if species.spatial_dimension != plan.bridge.dimension:
            raise ValueError("Particle and cochain spatial dimensions must match.")
        grid = plan.bridge.grid

        def prepared_for(layout):
            location = grid.location(layout.offsets)
            return ParticleGridSplatPlan(
                grid,
                location=location,
                assignment=plan.assignment,
                boundary="reject",
                execution=plan.execution,
                precision=plan.precision,
                budget=plan.budget,
            ).prepare(species.particles)

        charge = prepared_for(grid.vertices())
        dimension = len(grid.shape)
        electric = tuple(
            prepared_for(
                grid.entity_layout(
                    tuple(
                        "interval" if component == axis else "point"
                        for axis in range(dimension)
                    )
                )
            )
            for component in range(dimension)
        )
        magnetic = (
            tuple(prepared_for(grid.faces(axis)) for axis in grid.axis_names)
            if dimension == 3
            else ()
        )
        self.plan = plan
        self.species = species
        self.charge = charge
        self.electric = electric
        self.magnetic = magnetic
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-pic-particle-cochain-transfer",
                "plan": plan.plan_id,
                "species": species.prepared_id,
                "charge": charge.prepared_id,
                "electric": [value.prepared_id for value in electric],
                "magnetic": [value.prepared_id for value in magnetic],
            }
        )

    @property
    def bridge(self) -> StructuredCochainBridge:
        return self.plan.bridge

    def build(
        self,
        position: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> PICTransferState:
        return PICTransferState(
            self.charge.build(position, active_mask=active_mask),
            tuple(
                value.build(position, active_mask=active_mask) for value in self.electric
            ),
            tuple(
                value.build(position, active_mask=active_mask) for value in self.magnetic
            ),
            self.prepared_id,
        )

    def _validate_state(self, state: PICTransferState, /) -> None:
        if (
            not isinstance(state, PICTransferState)
            or state.transfer_id != self.prepared_id
        ):
            raise ValueError("PIC transfer state belongs to another prepared transfer.")

    def deposit_charge(self, state: PICTransferState, /) -> PICChargeDepositResult:
        return self.deposit_macrocharge(state, self.species.charges)

    def deposit_macrocharge(
        self, state: PICTransferState, macrocharge: ArrayLike, /
    ) -> PICChargeDepositResult:
        self._validate_state(state)
        result = self.charge.deposit_content(state.charge, macrocharge)
        cochain = self.bridge.pack(0, (result.density,))
        successful = result.successful & jnp.all(jnp.isfinite(cochain))
        return PICChargeDepositResult(
            result.content,
            result.density,
            cochain,
            result.balance,
            successful,
            self.prepared_id,
        )

    def gather_electric(
        self, state: PICTransferState, electric_cochain: ArrayLike, /
    ) -> PICFieldGatherResult:
        self._validate_state(state)
        integrated = self.bridge.unpack(1, electric_cochain)
        measures = self.bridge.unpack(1, self.bridge.cochain.primal_measures[1])
        physical = tuple(
            value / measure for value, measure in zip(integrated, measures, strict=True)
        )
        gathered = tuple(
            transfer.gather(route, component)
            for transfer, route, component in zip(
                self.electric, state.electric, physical, strict=True
            )
        )
        support = jnp.all(jnp.stack(tuple(value.support for value in gathered)), axis=0)
        values = jnp.stack(tuple(value.values for value in gathered), axis=-1)
        if self.bridge.dimension < 3:
            values = jnp.pad(values, ((0, 0), (0, 3 - self.bridge.dimension)))
        finite = jnp.all(jnp.isfinite(values))
        successful = support.all() & finite
        return PICFieldGatherResult(values, support, finite, successful, self.prepared_id)

    def gather_magnetic(
        self, state: PICTransferState, magnetic_cochain: ArrayLike, /
    ) -> PICFieldGatherResult:
        self._validate_state(state)
        if self.bridge.dimension != 3:
            raise ValueError("Magnetic PIC gather currently requires three dimensions.")
        physical = self.bridge.unpack_face_flux(magnetic_cochain)
        gathered = tuple(
            transfer.gather(route, component)
            for transfer, route, component in zip(
                self.magnetic, state.magnetic, physical, strict=True
            )
        )
        support = jnp.all(jnp.stack(tuple(value.support for value in gathered)), axis=0)
        values = jnp.stack(tuple(value.values for value in gathered), axis=-1)
        finite = jnp.all(jnp.isfinite(values))
        successful = support.all() & finite
        return PICFieldGatherResult(values, support, finite, successful, self.prepared_id)


__all__ = ["PICParticleCochainTransferPlan", "PreparedPICParticleCochainTransfer"]
