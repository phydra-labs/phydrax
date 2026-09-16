#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._adm_exchange import (
    ADMGridGeometry,
    combine_stress_energy_projections,
    StressEnergyProjection,
)
from ...solver._grmhd_ct import GRMHDCTState
from ...solver._grrmhd_runtime import (
    FixedGridGRRMHDIMEXPlan,
    GRRMHDStageProposal,
    GRRMHDState,
)
from ...solver._relativistic_finite_volume import ValenciaFiniteVolumeStageGeometry
from ._coupled_runtime import Z4cMatterCoupledRuntime
from ._matter_coupling import (
    ConservationLedger,
    CoupledParticipantStatus,
    CoupledStageAddress,
    FloorLedger,
    HorizonFluxLedger,
    MatterCouplingPolicy,
    MatterStageProposal,
)


class GRRMHDCouplingArguments(StrictModule):
    composition: Any
    transport_extinction: Any
    model_args: Any


class GRRMHDZ4cStageAdapter(StrictModule, NonTrainableState):
    """Bind the native GRRMHD stage proposal to the Z4c coupling protocol."""

    runtime: FixedGridGRRMHDIMEXPlan
    stage_geometry: Callable = eqx.field(static=True)
    stage_geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: FixedGridGRRMHDIMEXPlan,
        stage_geometry: Callable,
        /,
        *,
        stage_geometry_id: str,
        topology_id: str,
    ) -> None:
        if not isinstance(runtime, FixedGridGRRMHDIMEXPlan):
            raise TypeError("runtime must be FixedGridGRRMHDIMEXPlan.")
        if not callable(stage_geometry):
            raise TypeError("stage_geometry must be callable.")
        geometry_id = str(stage_geometry_id).strip()
        topology = str(topology_id).strip()
        if not geometry_id or not topology:
            raise ValueError("Stage-geometry and topology identities must be non-empty.")
        self.runtime = runtime
        self.stage_geometry = stage_geometry
        self.stage_geometry_id = geometry_id
        self.topology_id = topology
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "z4c-grrmhd-stage-adapter",
                "runtime": runtime.plan_id,
                "stage_geometry": geometry_id,
                "topology": topology,
            }
        )

    @staticmethod
    def _arguments(args: Any, /) -> GRRMHDCouplingArguments:
        if not isinstance(args, GRRMHDCouplingArguments):
            raise TypeError("GRRMHD coupling requires GRRMHDCouplingArguments.")
        return args

    def geometry_for(
        self,
        geometry: ADMGridGeometry,
        address: CoupledStageAddress,
        args: Any,
        /,
    ) -> ValenciaFiniteVolumeStageGeometry:
        arguments = self._arguments(args)
        stage = self.stage_geometry(address, arguments.model_args)
        if not isinstance(stage, ValenciaFiniteVolumeStageGeometry):
            raise TypeError(
                "stage_geometry callback must return ValenciaFiniteVolumeStageGeometry."
            )
        if (
            stage.cell.geometry_lineage_id != geometry.geometry_lineage_id
            or stage.cell.topology_id != geometry.topology_id
        ):
            raise ValueError("GRRMHD stage and Z4c ADM geometry identities differ.")
        checked_time = eqx.error_if(
            stage.time,
            stage.cell.snapshot_token != geometry.snapshot_token,
            "GRRMHD stage and Z4c ADM snapshot tokens differ.",
        )
        return eqx.tree_at(lambda value: value.time, stage, checked_time)

    def stress_energy_at_stage(
        self,
        state: GRRMHDState,
        geometry: ADMGridGeometry,
        address: CoupledStageAddress,
        args: Any,
        /,
    ) -> StressEnergyProjection:
        del address
        arguments = self._arguments(args)
        if not isinstance(state, GRRMHDState):
            raise TypeError("GRRMHD coupling matter state must be GRRMHDState.")
        full = self.runtime.material_transport.constrained_transport.full_state(
            state.material_state, state.constrained_transport.magnetic_flux
        )
        recovery = self.runtime.material_transport.system.recover(
            full, geometry, arguments.composition
        )
        material = self.runtime.material_transport.system.stress_energy(
            recovery.primitive,
            geometry,
            arguments.composition,
            conserved=full,
        )
        moments = state.radiation_state / geometry.sqrt_det_spatial_metric[..., None]
        radiation = self.runtime.radiation_transport.system.stress_energy_projection(
            moments[..., 0], moments[..., 1:], geometry
        )
        return combine_stress_energy_projections((material, radiation))

    @staticmethod
    def _combine_ct(
        first_weight: float,
        first: GRMHDCTState,
        second_weight: float,
        second: GRMHDCTState,
        /,
    ) -> GRMHDCTState:
        return GRMHDCTState(
            first_weight * first.magnetic_flux + second_weight * second.magnetic_flux,
            first_weight * first.vector_potential
            + second_weight * second.vector_potential,
            first_weight * first.gauge_scalar + second_weight * second.gauge_scalar,
        )

    def _recurrence(
        self,
        base: GRRMHDState,
        working: GRRMHDState,
        address: CoupledStageAddress,
        /,
    ) -> tuple[GRRMHDState, ArrayLike]:
        step = address.step_size
        if address.stage_id == 0:
            return base, step
        if address.stage_id == 1:
            return (
                GRRMHDState(
                    0.75 * base.material_state + 0.25 * working.material_state,
                    self._combine_ct(
                        0.75,
                        base.constrained_transport,
                        0.25,
                        working.constrained_transport,
                    ),
                    0.75 * base.radiation_state + 0.25 * working.radiation_state,
                    address.stage_time,
                    0.25 * step,
                    working.accepted_step,
                    working.status,
                ),
                0.25 * step,
            )
        return (
            GRRMHDState(
                base.material_state / 3.0 + 2.0 * working.material_state / 3.0,
                self._combine_ct(
                    1.0 / 3.0,
                    base.constrained_transport,
                    2.0 / 3.0,
                    working.constrained_transport,
                ),
                base.radiation_state / 3.0 + 2.0 * working.radiation_state / 3.0,
                address.stage_time,
                2.0 * step / 3.0,
                working.accepted_step,
                working.status,
            ),
            2.0 * step / 3.0,
        )

    def propose_matter_stage(
        self,
        base: GRRMHDState,
        working: GRRMHDState,
        geometry: ADMGridGeometry,
        stress_energy: StressEnergyProjection,
        address: CoupledStageAddress,
        args: Any,
        /,
    ) -> MatterStageProposal:
        arguments = self._arguments(args)
        recurrence_base, increment = self._recurrence(base, working, address)
        stage = self.geometry_for(geometry, address, arguments)
        proposal: GRRMHDStageProposal = self.runtime.propose_stage(
            recurrence_base,
            working,
            address.stage_time,
            increment,
            stage,
            arguments.composition,
            transport_extinction=arguments.transport_extinction,
        )
        candidate = eqx.tree_at(
            lambda value: value.time,
            proposal.candidate,
            address.step_end_time,
        )
        source = proposal.source.ledger
        conservation = ConservationLedger(
            jnp.asarray(0.0, dtype=candidate.time.dtype),
            jnp.max(jnp.abs(source.energy_defect), initial=0.0),
            jnp.max(
                jnp.abs(source.momentum_defect),
                axis=tuple(range(source.momentum_defect.ndim - 1)),
            ),
            jnp.max(
                jnp.abs(
                    self.runtime.material_transport.constrained_transport.magnetic_divergence(
                        candidate.constrained_transport.magnetic_flux
                    )
                ),
                initial=0.0,
            ),
        )
        zero = jnp.asarray(0.0, dtype=candidate.time.dtype)
        floor = FloorLedger(zero, zero, jnp.zeros((3,), dtype=zero.dtype), 0)
        horizon = HorizonFluxLedger(
            zero, zero, jnp.zeros((3,), dtype=zero.dtype), zero, zero
        )
        evidence = CoupledParticipantStatus(
            candidate.status,
            proposal.finite,
            proposal.converged,
            proposal.physically_valid,
            proposal.qualified,
            proposal.derivative_valid,
        )
        return MatterStageProposal(
            candidate,
            stress_energy,
            address,
            conservation,
            floor,
            horizon,
            evidence,
            matter_kind="grrmhd",
        )

    def coupled_runtime(
        self,
        geometry_at_stage: Callable,
        propose_z4c_stage: Callable,
        policy: MatterCouplingPolicy,
        /,
        *,
        z4c_runtime_id: str,
    ) -> Z4cMatterCoupledRuntime:
        return Z4cMatterCoupledRuntime(
            geometry_at_stage,
            self.stress_energy_at_stage,
            propose_z4c_stage,
            self.propose_matter_stage,
            policy,
            topology_id=self.topology_id,
            matter_kind="grrmhd",
            z4c_runtime_id=z4c_runtime_id,
            matter_runtime_id=self.runtime.plan_id,
        )


__all__ = [
    "GRRMHDCouplingArguments",
    "GRRMHDZ4cStageAdapter",
]
