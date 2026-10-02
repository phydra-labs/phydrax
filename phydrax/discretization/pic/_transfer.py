#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Particle↔cochain transfer with spline-Whitney shapes of order one to three.

Shape order ``p`` deposits charge with the degree-``p`` tensor cardinal
B-spline on vertices and gathers each oriented field component with the
matching spline-Whitney shape: degree ``p − 1`` along the axes an entity spans
and degree ``p`` across the others (edges for ``E``, faces for ``B``). Order one
is the lowest-order Whitney transfer (multilinear charge; multilinear
interpolation of the edge/face components).
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._chains import PreparedChainQuery
from ...typing import checked, parse
from .._cubical_whitney import CubicalSplineWhitneyKernel, PICShapeOrder
from .._structured_cochain import StructuredCochainBridge
from .._tensor_support import TensorEntityLayout
from ..particle import ParticlePrecisionPolicy, PreparedChargedParticles
from ..splatting import (
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


@final
class PICParticleCochainTransferPlan(StrictModule, NonTrainableState):
    """Bind charged particles to exact structured cochain entity locations."""

    bridge: StructuredCochainBridge
    kernel: CubicalSplineWhitneyKernel
    shape_order: PICShapeOrder = eqx.field(static=True)
    execution: SplatExecutionPolicy
    precision: ParticlePrecisionPolicy
    budget: ParticleGridSplatBudget
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        shape_order: PICShapeOrder = 1,
        execution: SplatExecutionPolicy | None = None,
        precision: ParticlePrecisionPolicy | None = None,
        budget: ParticleGridSplatBudget | None = None,
    ) -> None:
        order = parse(shape_order, PICShapeOrder, "shape_order")
        execution_ = SplatExecutionPolicy() if execution is None else execution
        precision_ = ParticlePrecisionPolicy() if precision is None else precision
        budget_ = ParticleGridSplatBudget() if budget is None else budget
        self.bridge = bridge
        self.kernel = CubicalSplineWhitneyKernel(bridge, order)
        self.shape_order = order
        self.execution = execution_
        self.precision = precision_
        self.budget = budget_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pic-particle-cochain-transfer-plan",
                "bridge": bridge.bridge_id,
                "shape_order": order,
                "execution": execution_.policy_id,
                "precision": precision_.policy_id,
                "budget": budget_.budget_id,
            }
        )

    def prepare(
        self, species: PreparedChargedParticles, /
    ) -> PreparedPICParticleCochainTransfer:
        return PreparedPICParticleCochainTransfer(self, species)


@final
class PreparedPICParticleCochainTransfer(StrictModule, NonTrainableState):
    """Prepared endpoint charge deposition and physical E/B gathering."""

    plan: PICParticleCochainTransferPlan
    species: PreparedChargedParticles
    charge: PreparedParticleGridSplat
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self, plan: PICParticleCochainTransferPlan, species: PreparedChargedParticles, /
    ) -> None:
        if species.spatial_dimension != plan.bridge.dimension:
            raise ValueError("Particle and cochain spatial dimensions must match.")
        grid = plan.bridge.grid

        def prepared_for(layout: TensorEntityLayout) -> PreparedParticleGridSplat:
            location = grid.location(layout.offsets)
            return ParticleGridSplatPlan(
                grid,
                location=location,
                assignment=plan.kernel.assignment(layout),
                boundary="reject",
                execution=plan.execution,
                precision=plan.precision,
                budget=plan.budget,
            ).prepare(species.particles)

        charge = prepared_for(grid.vertices())
        self.plan = plan
        self.species = species
        self.charge = charge
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-pic-particle-cochain-transfer",
                "plan": plan.plan_id,
                "species": species.prepared_id,
                "charge": charge.prepared_id,
                "kernel": plan.kernel.kernel_id,
            }
        )

    @property
    def bridge(self) -> StructuredCochainBridge:
        return self.plan.bridge

    @property
    def kernel(self) -> CubicalSplineWhitneyKernel:
        return self.plan.kernel

    def build(
        self,
        position: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> PICTransferState:
        charge = self.charge.build(position, active_mask=active_mask)
        geometry = self.plan.precision.geometry(jnp.asarray(position))
        if self.plan.execution.geometry_ad == "frozen":
            geometry = jax.lax.stop_gradient(geometry)
        active = charge.source_active_mask

        def prepared_query(degree: int) -> PreparedChainQuery:
            query = self.kernel.evaluate(geometry, degree)
            return eqx.tree_at(
                lambda value: (value.valid, value.successful),
                query,
                (query.valid & active[:, None], query.successful | ~active),
            )

        return PICTransferState(
            charge,
            prepared_query(1),
            prepared_query(2) if self.bridge.dimension == 3 else None,
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
        values = state.electric.gather(jnp.asarray(electric_cochain))
        support = state.electric.successful
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
        if state.magnetic is None:
            raise ValueError("Magnetic routes require a three-dimensional bridge.")
        components = state.magnetic.gather(jnp.asarray(magnetic_cochain))
        values = jnp.stack(
            (components[:, 2], -components[:, 1], components[:, 0]), axis=-1
        )
        support = state.magnetic.successful
        finite = jnp.all(jnp.isfinite(values))
        successful = support.all() & finite
        return PICFieldGatherResult(values, support, finite, successful, self.prepared_id)


__all__ = [
    "PICParticleCochainTransferPlan",
    "PreparedPICParticleCochainTransfer",
]
