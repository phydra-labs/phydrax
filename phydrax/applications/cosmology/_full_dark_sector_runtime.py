#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transactional composition of the full relativistic dark-sector closure.

Physics remains owned by its native plans and states. This module supplies the
single stage address, stress-energy sum, cross-owner admission ledger, atomic
commit boundary, and distributed restart/output binding.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import SingleDeviceSharding
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle._relativistic_stress_transfer import (
    RelativisticParticleState,
)
from ...equations._dark_radiation_moments import (
    DarkRadiationBoltzmannHierarchyPlan,
    DarkRadiationFourForce,
    DarkRadiationHierarchyState,
)
from ...lifecycle._chunk_repository import ArtifactRepository, ChunkEncoding
from ...lifecycle._distributed_checkpoint import (
    assemble_distributed_checkpoint_manifest,
    ProcessCheckpointPublication,
    publish_process_checkpoint,
    restore_global_array_from_checkpoint,
)
from ...lifecycle._event_graph_repository import (
    EpochCommitReceipt,
    EventGraphEpochManifest,
    GlobalEntity,
    GlobalEvent,
    GlobalEventEdge,
    GlobalWorkItem,
)
from ...lifecycle._models import CheckpointManifest
from ...metrix._adm_exchange import StressEnergyProjection
from ...particle_physics._bound_states import (
    DarkBoundStateSpectrum,
    RadiativeCapturePlan,
)
from ...particle_physics._dark_shower import DarkShowerEpochPlan
from ...particle_physics._decay_cascade import DarkDecayCascadePlan
from ...particle_physics._hadronization import (
    DarkClusterHadronizationPlan,
    DarkStringFragmentationPlan,
)
from ...particle_physics._matching_runtime import ProviderExecutionChain
from ...solver._dark_radiation_packets import (
    DarkRadiationPacketPlan,
    DarkRadiationPacketState,
)
from ...solver._dark_sector_epoch_runtime import (
    DarkSectorEpochPlan,
    DarkSectorEpochResult,
    DarkSectorEpochState,
    DarkSectorRunCoordinator,
)
from ..curved_spacetime_qft._coherent_transport import (
    CoherentDensityMatrixState,
    CoherentTransportPlan,
)
from ..curved_spacetime_qft._gauge_covariant_wigner import (
    GaugeCovariantWignerPlan,
    GaugeCovariantWignerState,
)
from ..curved_spacetime_qft._kadanoff_baym import (
    KadanoffBaymTransportPlan,
    KBMemoryState,
)
from ..curved_spacetime_qft._off_shell_transport import (
    OffShellTransportPlan,
    QuasiparticleOffShellState,
)
from ..numerical_relativity._einstein_vlasov import (
    EinsteinVlasovMatterPlan,
    EinsteinVlasovMatterResult,
    EinsteinVlasovMatterState,
)
from ..relativistic_scattering._matrix_element_revision import MatrixElementRevision
from ..relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from ._dark_radiation_transport import DarkRadiationGravitySource
from ._quantum_dark_kinetics import QuantumDarkKineticsPlan, QuantumKineticState
from ._thermal_dark_rates import ThermalDarkRatePlan
from ._weak_field_relativistic_pm import (
    WeakFieldRelativisticPMPlan,
    WeakFieldStressResult,
)


GravityPlan = WeakFieldRelativisticPMPlan | EinsteinVlasovMatterPlan
GravityState = RelativisticParticleState | EinsteinVlasovMatterState
RadiationPlan = DarkRadiationPacketPlan | DarkRadiationBoltzmannHierarchyPlan
RadiationState = DarkRadiationPacketState | DarkRadiationHierarchyState
HadronizationPlan = DarkStringFragmentationPlan | DarkClusterHadronizationPlan


def _identifier(value: str, name: str, /) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty identifier.")
    return normalized


def _identifiers(
    values: Sequence[str], name: str, /, *, allow_empty: bool = False
) -> tuple[str, ...]:
    result = tuple(_identifier(value, name) for value in values)
    if (not allow_empty and not result) or len(set(result)) != len(result):
        raise ValueError(f"{name} values must be distinct and valid.")
    return result


def _scalar(value: ArrayLike, name: str, /, *, dtype=None) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return result


def _real_scalar(value: ArrayLike, name: str, /, *, dtype=None) -> Array:
    result = _scalar(value, name, dtype=dtype)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    if not eqx.is_inexact_array(result):
        result = result.astype("float64")
    return result


def _nonnegative_tolerance(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


def _matrix_revision_id(revision: MatrixElementRevision, /) -> str:
    if not isinstance(revision, MatrixElementRevision):
        raise TypeError("matrix_element_revision must be MatrixElementRevision.")
    return revision.revision_id


class FullDarkSectorStageToken(StrictModule, NonTrainableState):
    """One exact frame/geometry/epoch/revision address shared by every owner."""

    geometry_snapshot_token: Array
    frame_token: Array
    time: Array
    scale_factor: Array
    epoch_sequence: int = eqx.field(static=True)
    epoch_plan_id: str = eqx.field(static=True)
    capacity_revision_id: str = eqx.field(static=True)
    species_revision_id: str = eqx.field(static=True)
    topology_revision_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_lineage_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    matrix_element_revision_id: str = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(
        self,
        frame: LocalRelativisticFramePlan,
        epoch_state: DarkSectorEpochState,
        matrix_element_revision: MatrixElementRevision,
        /,
    ):
        if not isinstance(frame, LocalRelativisticFramePlan):
            raise TypeError("frame must be LocalRelativisticFramePlan.")
        if not isinstance(epoch_state, DarkSectorEpochState):
            raise TypeError("epoch_state must be DarkSectorEpochState.")
        revision = _matrix_revision_id(matrix_element_revision)
        plan = epoch_state.plan
        realization = frame.realization_id()
        frame_time = jnp.asarray(frame.time).reshape((-1,))
        frame_scale = jnp.asarray(frame.scale_factor).reshape((-1,))
        time = eqx.error_if(
            frame_time[0],
            jnp.any(frame_time != frame_time[0]),
            "A composed full dark-sector stage requires one shared coordinate time.",
        )
        scale_factor = eqx.error_if(
            frame_scale[0],
            jnp.any(frame_scale != frame_scale[0]),
            "A composed full dark-sector stage requires one shared scale factor.",
        )
        self.geometry_snapshot_token = frame.frame_token
        self.frame_token = jnp.asarray(frame.frame_token, dtype=jnp.int32).reshape(())
        self.time = time
        self.scale_factor = scale_factor
        self.epoch_sequence = epoch_state.epoch_sequence
        self.epoch_plan_id = plan.plan_id
        self.capacity_revision_id = plan.capacity_revision_id
        self.species_revision_id = plan.species_revision_id
        self.topology_revision_id = plan.topology_revision_id
        self.topology_id = frame.geometry.topology_id
        self.geometry_lineage_id = frame.geometry.geometry_lineage_id
        self.frame_id = frame.frame_id
        self.frame_realization_id = realization
        self.unit_contract_id = frame.units.contract_id
        self.matrix_element_revision_id = revision
        self.stage_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-stage-token",
                "epoch_sequence": self.epoch_sequence,
                "epoch_plan": self.epoch_plan_id,
                "capacity_revision": self.capacity_revision_id,
                "species_revision": self.species_revision_id,
                "topology_revision": self.topology_revision_id,
                "topology": self.topology_id,
                "geometry_lineage": self.geometry_lineage_id,
                "frame": self.frame_id,
                "frame_realization": realization,
                "units": self.unit_contract_id,
                "matrix_element_revision": revision,
            }
        )


class NamedStressEnergyComponent(StrictModule):
    """One named owner projection plus its independent admission ledgers."""

    projection: StressEnergyProjection
    conservation_defect: Array
    constraint_defect: Array
    gauge_defect: Array
    entropy_production: Array
    unitarity_defect: Array
    evidence_valid: Array
    component_name: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_name: str,
        projection: StressEnergyProjection,
        /,
        *,
        conservation_defect: ArrayLike,
        constraint_defect: ArrayLike,
        gauge_defect: ArrayLike,
        entropy_production: ArrayLike,
        unitarity_defect: ArrayLike,
        evidence_valid: ArrayLike,
        evidence_id: str,
    ):
        if not isinstance(projection, StressEnergyProjection):
            raise TypeError("projection must be StressEnergyProjection.")
        dtype = projection.energy_density.dtype
        ledgers = tuple(
            _real_scalar(value, name, dtype=dtype)
            for value, name in (
                (conservation_defect, "conservation_defect"),
                (constraint_defect, "constraint_defect"),
                (gauge_defect, "gauge_defect"),
                (entropy_production, "entropy_production"),
                (unitarity_defect, "unitarity_defect"),
            )
        )
        evidence = _scalar(evidence_valid, "evidence_valid", dtype=jnp.bool_)
        self.projection = projection
        (
            self.conservation_defect,
            self.constraint_defect,
            self.gauge_defect,
            self.entropy_production,
            self.unitarity_defect,
        ) = ledgers
        self.evidence_valid = jax.lax.stop_gradient(evidence)
        self.component_name = _identifier(component_name, "component_name")
        self.evidence_id = _identifier(evidence_id, "evidence_id")

    @property
    def finite(self) -> Array:
        return self.projection.all_active_valid & jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        self.conservation_defect,
                        self.constraint_defect,
                        self.gauge_defect,
                        self.entropy_production,
                        self.unitarity_defect,
                    )
                )
            )
        )


class FullDarkSectorStressAssembly(StrictModule):
    components: tuple[NamedStressEnergyComponent, ...]
    total: StressEnergyProjection
    component_active: Array
    stage_consistent: Array
    finite: Array
    successful: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)
    assembly_id: str = eqx.field(static=True)


def assemble_full_dark_sector_stress(
    components: Sequence[NamedStressEnergyComponent],
    stage: FullDarkSectorStageToken,
    /,
) -> FullDarkSectorStressAssembly:
    """Assemble exactly one ADM projection from named co-stage components."""

    values = tuple(components)
    if not values or any(
        not isinstance(value, NamedStressEnergyComponent) for value in values
    ):
        raise TypeError("components must contain NamedStressEnergyComponent values.")
    if not isinstance(stage, FullDarkSectorStageToken):
        raise TypeError("stage must be FullDarkSectorStageToken.")
    names = tuple(value.component_name for value in values)
    if len(set(names)) != len(names):
        raise ValueError("Stress-energy component names must be unique.")
    first = values[0].projection
    static_identity = (
        first.geometry_lineage_id,
        first.convention_id,
        first.scale_id,
        first.topology_id,
        first.leading_shape,
        first.energy_density.dtype,
    )
    for value in values[1:]:
        projection = value.projection
        candidate = (
            projection.geometry_lineage_id,
            projection.convention_id,
            projection.scale_id,
            projection.topology_id,
            projection.leading_shape,
            projection.energy_density.dtype,
        )
        if candidate != static_identity:
            raise ValueError(
                "Stress-energy components do not share one grid/frame contract."
            )
    if first.topology_id != stage.topology_id:
        raise ValueError("Stress-energy topology differs from the shared stage token.")
    stage_consistent = jnp.all(
        jnp.stack(
            tuple(
                jnp.asarray(
                    projection.projection.snapshot_token == stage.geometry_snapshot_token
                )
                for projection in values
            )
        )
    )
    energy = eqx.error_if(
        jnp.stack(tuple(value.projection.energy_density for value in values)),
        ~stage_consistent,
        "Full dark-sector stress components do not share the stage snapshot token.",
    )
    active = jnp.stack(tuple(value.projection.active for value in values))
    valid = jnp.stack(tuple(value.projection.valid for value in values))
    momentum = jnp.stack(tuple(value.projection.momentum_covector for value in values))
    stress = jnp.stack(tuple(value.projection.stress_covariant for value in values))
    projection_defect = jnp.stack(
        tuple(value.projection.projection_defect for value in values)
    )
    conservation_defect = jnp.stack(
        tuple(value.projection.conservation_defect for value in values)
    )
    total_active = jnp.any(active, axis=0)
    total_valid = total_active & jnp.all((~active) | valid, axis=0)
    total_energy = jnp.sum(jnp.where(active, energy, 0.0), axis=0)
    total_momentum = jnp.sum(jnp.where(active[..., None], momentum, 0.0), axis=0)
    total_stress = jnp.sum(jnp.where(active[..., None, None], stress, 0.0), axis=0)
    total_projection_defect = jnp.sum(jnp.where(active, projection_defect, 0.0), axis=0)
    total_conservation_defect = jnp.sum(
        jnp.where(active, conservation_defect, 0.0), axis=0
    )
    projection_id = canonical_fingerprint(
        {
            "kind": "full-dark-sector-total-stress-energy",
            "stage": stage.stage_id,
            "components": [
                [value.component_name, value.projection.projection_id] for value in values
            ],
        }
    )
    total = StressEnergyProjection(
        total_energy,
        total_momentum,
        total_stress,
        total_active,
        total_valid,
        total_projection_defect,
        total_conservation_defect,
        snapshot_token=stage.geometry_snapshot_token,
        geometry_lineage_id=first.geometry_lineage_id,
        convention_id=first.convention_id,
        scale_id=first.scale_id,
        topology_id=first.topology_id,
        projection_id=projection_id,
    )
    component_active = jnp.stack(
        tuple(jnp.any(value.projection.active) for value in values)
    )
    finite = (
        jnp.all(jnp.stack(tuple(value.finite for value in values)))
        & total.all_active_valid
    )
    successful = (
        finite
        & stage_consistent
        & jnp.all(jnp.stack(tuple(value.evidence_valid for value in values)))
    )
    assembly_id = canonical_fingerprint(
        {
            "kind": "full-dark-sector-stress-assembly",
            "stage": stage.stage_id,
            "projection": projection_id,
            "components": list(names),
            "evidence": [value.evidence_id for value in values],
        }
    )
    return FullDarkSectorStressAssembly(
        values,
        total,
        component_active,
        stage_consistent,
        finite,
        successful,
        names,
        stage.stage_id,
        assembly_id,
    )


def weak_gravity_stress_component(
    result: WeakFieldStressResult, /
) -> NamedStressEnergyComponent:
    if not isinstance(result, WeakFieldStressResult):
        raise TypeError("result must be WeakFieldStressResult.")
    return NamedStressEnergyComponent(
        "matter-gravity",
        result.stress,
        conservation_defect=jnp.maximum(
            jnp.max(result.stress.conservation_defect), result.force_relative_defect
        ),
        constraint_defect=result.constraint_defect,
        gauge_defect=result.gauge_defect,
        entropy_production=0.0,
        unitarity_defect=0.0,
        evidence_valid=result.successful,
        evidence_id=result.source_id,
    )


def einstein_vlasov_stress_component(
    result: EinsteinVlasovMatterResult, /
) -> NamedStressEnergyComponent:
    if not isinstance(result, EinsteinVlasovMatterResult):
        raise TypeError("result must be EinsteinVlasovMatterResult.")
    evidence = result.evidence
    return NamedStressEnergyComponent(
        "matter-gravity",
        result.endpoint_stress.projection,
        conservation_defect=evidence.source_exchange_defect,
        constraint_defect=evidence.adm_constraint_linf,
        gauge_defect=0.0,
        entropy_production=0.0,
        unitarity_defect=0.0,
        evidence_valid=result.successful & evidence.qualified,
        evidence_id=result.endpoint_stress.source_id,
    )


def radiation_stress_component(
    source: DarkRadiationGravitySource, /
) -> NamedStressEnergyComponent:
    if not isinstance(source, DarkRadiationGravitySource):
        raise TypeError("source must be DarkRadiationGravitySource.")
    exchange = source.exchange
    return NamedStressEnergyComponent(
        "radiation",
        source.projection,
        conservation_defect=jnp.max(jnp.abs(exchange.balance_residual)),
        constraint_defect=0.0,
        gauge_defect=0.0,
        entropy_production=0.0,
        unitarity_defect=0.0,
        evidence_valid=jnp.all(exchange.exact_opposite),
        evidence_id=source.source_id,
    )


class FullDarkSectorStageLedger(StrictModule):
    component_conservation: Array
    component_constraint: Array
    component_gauge: Array
    component_entropy_production: Array
    component_unitarity: Array
    component_evidence_valid: Array
    total_conservation: Array
    total_constraint: Array
    total_gauge: Array
    total_entropy_production: Array
    total_unitarity: Array
    four_force_residual: Array
    stress_valid: Array
    exact_four_force_cancellation: Array
    entropy_admissible: Array
    finite: Array
    successful: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)


class FullDarkSectorRuntimePlan(StrictModule, NonTrainableState):
    """Exact cross-owner profile; it contains no alternate physics implementation."""

    units: RelativisticUnitContract
    gravity: GravityPlan
    epoch: DarkSectorEpochPlan
    quantum: QuantumDarkKineticsPlan
    thermal: ThermalDarkRatePlan
    coherent: CoherentTransportPlan
    off_shell: OffShellTransportPlan
    kadanoff_baym: KadanoffBaymTransportPlan
    wigner: GaugeCovariantWignerPlan
    shower: DarkShowerEpochPlan
    hadronization: HadronizationPlan
    bound_states: DarkBoundStateSpectrum
    radiative_capture: RadiativeCapturePlan
    decay_cascade: DarkDecayCascadePlan
    radiation: RadiationPlan
    matrix_element_revision: MatrixElementRevision = eqx.field(static=True)
    provider_chain: ProviderExecutionChain | None = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    gauge_tolerance: float = eqx.field(static=True)
    unitarity_tolerance: float = eqx.field(static=True)
    entropy_tolerance: float = eqx.field(static=True)
    profile_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    revision_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        units: RelativisticUnitContract,
        gravity: GravityPlan,
        epoch: DarkSectorEpochPlan,
        quantum: QuantumDarkKineticsPlan,
        thermal: ThermalDarkRatePlan,
        coherent: CoherentTransportPlan,
        off_shell: OffShellTransportPlan,
        kadanoff_baym: KadanoffBaymTransportPlan,
        wigner: GaugeCovariantWignerPlan,
        shower: DarkShowerEpochPlan,
        hadronization: HadronizationPlan,
        bound_states: DarkBoundStateSpectrum,
        radiative_capture: RadiativeCapturePlan,
        decay_cascade: DarkDecayCascadePlan,
        radiation: RadiationPlan,
        matrix_element_revision: MatrixElementRevision,
        /,
        *,
        provider_chain: ProviderExecutionChain | None = None,
        conservation_tolerance: float = 1.0e-8,
        constraint_tolerance: float = 1.0e-8,
        gauge_tolerance: float = 1.0e-8,
        unitarity_tolerance: float = 1.0e-8,
        entropy_tolerance: float = 1.0e-12,
    ):
        expected = (
            (units, RelativisticUnitContract, "units"),
            (epoch, DarkSectorEpochPlan, "epoch"),
            (quantum, QuantumDarkKineticsPlan, "quantum"),
            (thermal, ThermalDarkRatePlan, "thermal"),
            (coherent, CoherentTransportPlan, "coherent"),
            (off_shell, OffShellTransportPlan, "off_shell"),
            (kadanoff_baym, KadanoffBaymTransportPlan, "kadanoff_baym"),
            (wigner, GaugeCovariantWignerPlan, "wigner"),
            (shower, DarkShowerEpochPlan, "shower"),
            (bound_states, DarkBoundStateSpectrum, "bound_states"),
            (radiative_capture, RadiativeCapturePlan, "radiative_capture"),
            (decay_cascade, DarkDecayCascadePlan, "decay_cascade"),
        )
        for value, kind, name in expected:
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be {kind.__name__}.")
        if not isinstance(
            gravity, (WeakFieldRelativisticPMPlan, EinsteinVlasovMatterPlan)
        ):
            raise TypeError("gravity must be a weak-field or Einstein–Vlasov plan.")
        if not isinstance(
            hadronization, (DarkStringFragmentationPlan, DarkClusterHadronizationPlan)
        ):
            raise TypeError("hadronization must be an explicit native dark model plan.")
        if not isinstance(
            radiation, (DarkRadiationPacketPlan, DarkRadiationBoltzmannHierarchyPlan)
        ):
            raise TypeError("radiation must be a packet or Boltzmann-hierarchy plan.")
        if provider_chain is not None and not isinstance(
            provider_chain, ProviderExecutionChain
        ):
            raise TypeError("provider_chain must be ProviderExecutionChain or None.")
        revision_id = _matrix_revision_id(matrix_element_revision)
        contract = units.contract_id
        if provider_chain is not None and any(
            record.unit_contract_id != contract
            or record.frame_id != quantum.frame.frame_id
            or record.frame_realization_id != quantum.frame_realization_id
            or record.matrix_element_revision.revision_id != revision_id
            for record in provider_chain.records
        ):
            raise ValueError(
                "Pinned provider executions differ from the runtime units, frame "
                "realization, or matrix-element revision."
            )
        owner_units = (
            quantum.units.contract_id,
            coherent.unit_contract_id,
            thermal.artifact.unit_contract_id,
            off_shell.unit_contract_id,
            kadanoff_baym.unit_contract_id,
            wigner.unit_contract_id,
            shower.units.contract_id,
            hadronization.units.contract_id,
            bound_states.units.contract_id,
            decay_cascade.units.contract_id,
            (
                radiation.units.contract_id
                if isinstance(radiation, DarkRadiationPacketPlan)
                else radiation.unit_contract_id
            ),
            (
                gravity.units.contract_id
                if isinstance(gravity, WeakFieldRelativisticPMPlan)
                else gravity.stress.units.contract_id
            ),
        )
        if any(value != contract for value in owner_units):
            raise ValueError("All full dark-sector owners must share one unit contract.")
        if (
            shower.runtime_plan.plan_id != epoch.plan_id
            or hadronization.runtime_plan.plan_id != epoch.plan_id
            or bound_states.runtime_plan.plan_id != epoch.plan_id
            or decay_cascade.runtime_plan.plan_id != epoch.plan_id
            or kadanoff_baym.epoch.plan_id != epoch.plan_id
            or (
                isinstance(radiation, DarkRadiationPacketPlan)
                and radiation.epoch_plan_id != epoch.plan_id
            )
        ):
            raise ValueError(
                "All unbounded owners must use the shared finite epoch plan."
            )
        particle_species_ids = {
            shower.species.table_id,
            hadronization.species.table_id,
            bound_states.species.table_id,
            decay_cascade.species.table_id,
        }
        if particle_species_ids != {epoch.species_revision_id}:
            raise ValueError(
                "Shower, hadronization, bound-state, decay, and epoch species revisions differ."
            )
        quantum_compatible = (
            coherent.quantum_support.species_plan_ids == quantum.species_plan_ids
            and coherent.quantum_support.frame_id == quantum.frame.frame_id
            and coherent.quantum_support.unit_contract_id == contract
        )
        quantum_compatible = quantum_compatible and (
            thermal.artifact.species_plan_ids == quantum.species_plan_ids
        )
        if not quantum_compatible:
            raise ValueError("Quantum/coherent support identities disagree.")
        if off_shell.support_id != coherent.support_id:
            raise ValueError("Coherent and off-shell support identities disagree.")
        if kadanoff_baym.off_shell.plan_id != off_shell.plan_id:
            raise ValueError("Kadanoff–Baym must consume the exact off-shell plan.")
        if radiative_capture.spectrum.spectrum_id != bound_states.spectrum_id:
            raise ValueError(
                "Radiative capture must consume the exact bound-state spectrum."
            )
        if wigner.support_id != coherent.support_id:
            raise ValueError(
                "Gauge-covariant Wigner support differs from quantum support."
            )
        frame_ids = (
            quantum.frame.frame_id,
            coherent.frame_id,
            off_shell.frame_id,
            kadanoff_baym.frame_id,
            thermal.artifact.frame_id,
            wigner.frame_id,
            shower.frame.frame_id,
            hadronization.frame.frame_id,
            bound_states.frame.frame_id,
            decay_cascade.frame.frame_id,
            radiation.frame_id,
        )
        frame_realizations = (
            quantum.frame_realization_id,
            coherent.frame_realization_id,
            off_shell.frame_realization_id,
            kadanoff_baym.frame_realization_id,
            thermal.artifact.frame_realization_id,
            wigner.frame_realization_id,
            shower.frame_realization_id,
            hadronization.frame_realization_id,
            bound_states.frame_realization_id,
            decay_cascade.frame_realization_id,
            (
                radiation.initial_frame_realization_id
                if isinstance(radiation, DarkRadiationBoltzmannHierarchyPlan)
                else quantum.frame_realization_id
            ),
        )
        if len(set(frame_realizations)) != 1:
            raise ValueError("Stage-bound owners must share one exact frame realization.")
        if len(set(frame_ids)) != 1:
            raise ValueError("All stage-bound owners must share one exact local frame.")
        tolerances = tuple(
            _nonnegative_tolerance(value, name)
            for value, name in (
                (conservation_tolerance, "conservation_tolerance"),
                (constraint_tolerance, "constraint_tolerance"),
                (gauge_tolerance, "gauge_tolerance"),
                (unitarity_tolerance, "unitarity_tolerance"),
                (entropy_tolerance, "entropy_tolerance"),
            )
        )
        gravity_id = (
            gravity.plan_id
            if isinstance(gravity, WeakFieldRelativisticPMPlan)
            else gravity.runtime_id
        )
        hadronization_id = hadronization.profile_id
        radiation_id = radiation.plan_id
        profiles = (
            ("gravity", gravity_id),
            ("epoch", epoch.plan_id),
            ("quantum", quantum.plan_id),
            ("thermal", thermal.plan_id),
            ("coherent", coherent.plan_id),
            ("off-shell", off_shell.plan_id),
            ("kadanoff-baym", kadanoff_baym.plan_id),
            ("gauge-covariant-wigner", wigner.plan_id),
            ("shower", shower.plan_id),
            ("hadronization", hadronization_id),
            ("bound-states", bound_states.spectrum_id),
            ("radiative-capture", radiative_capture.plan_id),
            ("decay-cascade", decay_cascade.plan_id),
            ("radiation", radiation_id),
        ) + (
            ()
            if provider_chain is None
            else (("external-provider-chain", provider_chain.chain_id),)
        )
        revisions = (
            ("matrix-element", revision_id),
            ("species", epoch.species_revision_id),
            ("topology", epoch.topology_revision_id),
            ("thermal-artifact", thermal.artifact.artifact_id),
            ("capacity", epoch.capacity_revision_id),
            ("compile", epoch.compile_signature_id),
            ("shower-model", shower.model_revision_id),
            ("hadronization-model", hadronization.model_revision_id),
            ("bound-state-model", bound_states.model_revision_id),
            ("decay-model", decay_cascade.model_revision_id),
        )
        self.units = units
        self.gravity = gravity
        self.epoch = epoch
        self.quantum = quantum
        self.coherent = coherent
        self.thermal = thermal
        self.off_shell = off_shell
        self.kadanoff_baym = kadanoff_baym
        self.wigner = wigner
        self.shower = shower
        self.hadronization = hadronization
        self.bound_states = bound_states
        self.radiative_capture = radiative_capture
        self.decay_cascade = decay_cascade
        self.radiation = radiation
        self.matrix_element_revision = matrix_element_revision
        self.provider_chain = provider_chain
        (
            self.conservation_tolerance,
            self.constraint_tolerance,
            self.gauge_tolerance,
            self.unitarity_tolerance,
            self.entropy_tolerance,
        ) = tolerances
        self.profile_ids = profiles
        self.revision_ids = revisions
        self.plan_id = canonical_fingerprint(
            {
                "kind": "full-relativistic-dark-sector-runtime-plan",
                "units": contract,
                "profiles": [list(value) for value in profiles],
                "revisions": [list(value) for value in revisions],
                "tolerances": list(tolerances),
            }
        )

    def ledger(
        self,
        assembly: FullDarkSectorStressAssembly,
        exchange: DarkRadiationFourForce,
        /,
    ) -> FullDarkSectorStageLedger:
        if not isinstance(assembly, FullDarkSectorStressAssembly):
            raise TypeError("assembly must be FullDarkSectorStressAssembly.")
        if not isinstance(exchange, DarkRadiationFourForce):
            raise TypeError("exchange must be DarkRadiationFourForce.")
        if (
            exchange.frame_id != self.quantum.frame.frame_id
            or exchange.frame_realization_id != self.quantum.frame_realization_id
            or exchange.unit_contract_id != self.units.contract_id
        ):
            raise ValueError(
                "Four-force frame/unit identity differs from the runtime plan."
            )
        values = assembly.components
        conservation = jnp.stack(tuple(value.conservation_defect for value in values))
        constraint = jnp.stack(tuple(value.constraint_defect for value in values))
        gauge = jnp.stack(tuple(value.gauge_defect for value in values))
        entropy = jnp.stack(tuple(value.entropy_production for value in values))
        unitarity = jnp.stack(tuple(value.unitarity_defect for value in values))
        evidence = jnp.stack(tuple(value.evidence_valid for value in values))
        four_force_residual = exchange.radiation_four_force + exchange.matter_four_force
        exact_force = jnp.all(exchange.exact_opposite) & jnp.all(
            four_force_residual == 0.0
        )
        total_conservation = jnp.sum(conservation) + jnp.max(
            jnp.abs(four_force_residual), initial=0.0
        )
        total_constraint = jnp.max(constraint, initial=0.0)
        total_gauge = jnp.max(gauge, initial=0.0)
        total_entropy = jnp.sum(entropy)
        total_unitarity = jnp.max(unitarity, initial=0.0)
        entropy_admissible = jnp.all(entropy >= -self.entropy_tolerance)
        finite = jnp.all(
            jnp.isfinite(
                jnp.concatenate(
                    (
                        conservation,
                        constraint,
                        gauge,
                        entropy,
                        unitarity,
                        four_force_residual.reshape((-1,)),
                    )
                )
            )
        )
        successful = (
            finite
            & assembly.successful
            & exact_force
            & entropy_admissible
            & jnp.all(evidence)
            & (total_conservation <= self.conservation_tolerance)
            & (total_constraint <= self.constraint_tolerance)
            & (total_gauge <= self.gauge_tolerance)
            & (total_unitarity <= self.unitarity_tolerance)
        )
        evidence_ids = tuple(value.evidence_id for value in values)
        ledger_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-stage-ledger",
                "plan": self.plan_id,
                "assembly": assembly.assembly_id,
                "exchange": exchange.exchange_id,
                "components": list(assembly.component_names),
                "evidence": list(evidence_ids),
            }
        )
        return FullDarkSectorStageLedger(
            conservation,
            constraint,
            gauge,
            entropy,
            unitarity,
            evidence,
            total_conservation,
            total_constraint,
            total_gauge,
            total_entropy,
            total_unitarity,
            four_force_residual,
            assembly.successful,
            exact_force,
            entropy_admissible,
            finite,
            successful,
            assembly.component_names,
            evidence_ids,
            ledger_id,
        )


class FullDarkSectorCompositeState(StrictModule):
    """Composition of exact owner states; no field duplicates their physics."""

    gravity: GravityState
    epoch: DarkSectorEpochState
    quantum: QuantumKineticState
    coherent: CoherentDensityMatrixState
    off_shell: QuasiparticleOffShellState
    kb_memory: KBMemoryState
    wigner: GaugeCovariantWignerState
    radiation: RadiationState
    stage: FullDarkSectorStageToken

    def __init__(
        self,
        gravity: GravityState,
        epoch: DarkSectorEpochState,
        quantum: QuantumKineticState,
        coherent: CoherentDensityMatrixState,
        off_shell: QuasiparticleOffShellState,
        kb_memory: KBMemoryState,
        wigner: GaugeCovariantWignerState,
        radiation: RadiationState,
        stage: FullDarkSectorStageToken,
        /,
    ):
        expected = (
            (epoch, DarkSectorEpochState, "epoch"),
            (quantum, QuantumKineticState, "quantum"),
            (coherent, CoherentDensityMatrixState, "coherent"),
            (off_shell, QuasiparticleOffShellState, "off_shell"),
            (kb_memory, KBMemoryState, "kb_memory"),
            (wigner, GaugeCovariantWignerState, "wigner"),
            (stage, FullDarkSectorStageToken, "stage"),
        )
        for value, kind, name in expected:
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be {kind.__name__}.")
        if not isinstance(
            gravity, (RelativisticParticleState, EinsteinVlasovMatterState)
        ):
            raise TypeError("gravity must be a weak- or full-gravity owner state.")
        if not isinstance(
            radiation, (DarkRadiationPacketState, DarkRadiationHierarchyState)
        ):
            raise TypeError("radiation must be a packet or hierarchy owner state.")
        self.gravity = gravity
        self.epoch = epoch
        self.quantum = quantum
        self.coherent = coherent
        self.off_shell = off_shell
        self.kb_memory = kb_memory
        self.wigner = wigner
        self.radiation = radiation
        self.stage = stage


class FullDarkSectorStageEvidence(StrictModule):
    ledger: FullDarkSectorStageLedger
    epoch_complete: Array
    epoch_conserved: Array
    epoch_not_rolled_back: Array
    state_consistent: Array
    evidence_complete: Array
    finite: Array
    successful: Array
    evidence_id: str = eqx.field(static=True)


class FullDarkSectorStageCandidate(StrictModule):
    source_state: FullDarkSectorCompositeState
    proposed_state: FullDarkSectorCompositeState
    epoch_result: DarkSectorEpochResult
    stress: FullDarkSectorStressAssembly
    radiation_source: DarkRadiationGravitySource
    evidence: FullDarkSectorStageEvidence
    candidate_id: str = eqx.field(static=True)


class FullDarkSectorStageCommit(StrictModule):
    state: FullDarkSectorCompositeState
    stress: FullDarkSectorStressAssembly
    evidence: FullDarkSectorStageEvidence
    receipt: EpochCommitReceipt | None = eqx.field(static=True)
    committed: bool = eqx.field(static=True)
    rolled_back: bool = eqx.field(static=True)
    commit_id: str = eqx.field(static=True)


def _gravity_particles(state: GravityState, /) -> RelativisticParticleState:
    return state if isinstance(state, RelativisticParticleState) else state.particles


def _state_consistency(
    plan: FullDarkSectorRuntimePlan, state: FullDarkSectorCompositeState, /
) -> Array:
    stage = state.stage
    particles = _gravity_particles(state.gravity)
    static_compatible = (
        state.epoch.plan.plan_id == plan.epoch.plan_id
        and stage.epoch_plan_id == plan.epoch.plan_id
        and stage.capacity_revision_id == plan.epoch.capacity_revision_id
        and stage.species_revision_id == plan.epoch.species_revision_id
        and stage.topology_revision_id == plan.epoch.topology_revision_id
        and stage.matrix_element_revision_id
        == _matrix_revision_id(plan.matrix_element_revision)
        and stage.unit_contract_id == plan.units.contract_id
        and particles.topology_id == stage.topology_id
        and particles.frame_id == stage.frame_id
        and particles.frame_lineage_id == stage.geometry_lineage_id
        and state.quantum.unit_contract_id == stage.unit_contract_id
        and state.quantum.frame_id == stage.frame_id
        and state.quantum.frame_realization_id == stage.frame_realization_id
        and state.quantum.support_id == plan.coherent.support_id
        and state.coherent.plan_id == plan.coherent.plan_id
        and state.coherent.frame_id == stage.frame_id
        and state.coherent.frame_realization_id == stage.frame_realization_id
        and state.off_shell.plan_id == plan.off_shell.plan_id
        and state.off_shell.frame_id == stage.frame_id
        and state.off_shell.frame_realization_id == stage.frame_realization_id
        and state.kb_memory.plan_id == plan.kadanoff_baym.plan_id
        and state.kb_memory.frame_realization_id == stage.frame_realization_id
        and state.wigner.plan_id == plan.wigner.plan_id
        and state.wigner.frame_id == stage.frame_id
        and state.wigner.frame_realization_id == stage.frame_realization_id
        and state.radiation.frame_id == stage.frame_id
        and state.radiation.frame_realization_id == stage.frame_realization_id
        and state.radiation.unit_contract_id == stage.unit_contract_id
        and (
            isinstance(state.gravity, RelativisticParticleState)
            == isinstance(plan.gravity, WeakFieldRelativisticPMPlan)
        )
        and (
            isinstance(state.radiation, DarkRadiationPacketState)
            == isinstance(plan.radiation, DarkRadiationPacketPlan)
        )
    )
    dynamic = (
        jnp.asarray(static_compatible)
        & (state.epoch.epoch_sequence == stage.epoch_sequence)
        & jnp.all(particles.frame_token == stage.frame_token)
        & jnp.all(state.quantum.frame_token == stage.frame_token)
        & jnp.all(state.coherent.frame_token == stage.frame_token)
        & jnp.all(state.off_shell.frame_token == stage.frame_token)
        & jnp.all(state.kb_memory.frame_token == stage.frame_token)
        & jnp.all(state.wigner.frame_token == stage.frame_token)
        & jnp.all(state.radiation.frame_token == stage.frame_token)
        & jnp.all(
            state.quantum.frame.geometry.snapshot_token == stage.geometry_snapshot_token
        )
        & jnp.all(particles.time == stage.time)
        & jnp.all(state.quantum.time == stage.time)
        & jnp.all(state.coherent.time == stage.time)
        & jnp.all(state.coherent.frame_time == stage.time)
        & jnp.all(state.off_shell.time == stage.time)
        & jnp.all(state.off_shell.frame_time == stage.time)
        & jnp.all(state.kb_memory.frame_time == stage.time)
        & jnp.all(state.wigner.frame_time == stage.time)
        & jnp.all(particles.scale_factor == stage.scale_factor)
        & jnp.all(state.quantum.frame.scale_factor == stage.scale_factor)
        & jnp.all(state.coherent.frame_scale_factor == stage.scale_factor)
        & jnp.all(state.off_shell.frame_scale_factor == stage.scale_factor)
        & jnp.all(state.kb_memory.frame_scale_factor == stage.scale_factor)
        & jnp.all(state.wigner.frame_scale_factor == stage.scale_factor)
        & jnp.all(state.radiation.scale_factor == stage.scale_factor)
        & jnp.all(state.kb_memory.epoch_sequence == state.epoch.epoch_sequence)
    )
    if isinstance(state.radiation, DarkRadiationPacketState):
        dynamic = (
            dynamic
            & jnp.all(state.radiation.coordinate_time == stage.time)
            & jnp.all(state.radiation.epoch_sequence == state.epoch.epoch_sequence)
        )
    return dynamic


def propose_full_dark_sector_stage(
    plan: FullDarkSectorRuntimePlan,
    source_state: FullDarkSectorCompositeState,
    proposed_state: FullDarkSectorCompositeState,
    epoch_result: DarkSectorEpochResult,
    components: Sequence[NamedStressEnergyComponent],
    radiation_source: DarkRadiationGravitySource,
    /,
) -> FullDarkSectorStageCandidate:
    """Build one global candidate; no state or durable graph is committed here."""

    if not isinstance(plan, FullDarkSectorRuntimePlan):
        raise TypeError("plan must be FullDarkSectorRuntimePlan.")
    if not isinstance(source_state, FullDarkSectorCompositeState) or not isinstance(
        proposed_state, FullDarkSectorCompositeState
    ):
        raise TypeError("source_state and proposed_state must be composite owner states.")
    if not isinstance(epoch_result, DarkSectorEpochResult):
        raise TypeError("epoch_result must be DarkSectorEpochResult.")
    if not isinstance(radiation_source, DarkRadiationGravitySource):
        raise TypeError("radiation_source must be DarkRadiationGravitySource.")
    if epoch_result.state is not proposed_state.epoch:
        raise ValueError(
            "Proposed composite state must contain the exact epoch result state."
        )
    stress = assemble_full_dark_sector_stress(components, proposed_state.stage)
    radiation_components = tuple(
        value for value in stress.components if value.component_name == "radiation"
    )
    if (
        len(radiation_components) != 1
        or radiation_components[0].projection.projection_id
        != radiation_source.projection.projection_id
    ):
        raise ValueError(
            "The named radiation stress must be the exact radiation owner source."
        )
    exchange = radiation_source.exchange
    ledger = plan.ledger(stress, exchange)
    state_consistent = (
        _state_consistency(plan, proposed_state)
        & (exchange.endpoint_time == proposed_state.stage.time)
        & (exchange.frame_token == proposed_state.stage.frame_token)
        & (exchange.frame_realization_id == proposed_state.stage.frame_realization_id)
        & (radiation_source.endpoint_time == proposed_state.stage.time)
        & (radiation_source.frame_token == proposed_state.stage.frame_token)
        & (
            radiation_source.frame_realization_id
            == proposed_state.stage.frame_realization_id
        )
    )
    epoch_complete = epoch_result.complete & ~epoch_result.backpressured
    epoch_conserved = epoch_result.conservation_ok
    epoch_not_rolled_back = ~epoch_result.rolled_back
    evidence_complete = jnp.all(ledger.component_evidence_valid)
    finite = ledger.finite & state_consistent
    successful = (
        finite
        & ledger.successful
        & epoch_complete
        & epoch_conserved
        & epoch_not_rolled_back
        & evidence_complete
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "full-dark-sector-stage-evidence",
            "plan": plan.plan_id,
            "stage": proposed_state.stage.stage_id,
            "ledger": ledger.ledger_id,
            "epoch_evidence": list(epoch_result.evidence_ids),
        }
    )
    evidence = FullDarkSectorStageEvidence(
        ledger,
        epoch_complete,
        epoch_conserved,
        epoch_not_rolled_back,
        state_consistent,
        evidence_complete,
        finite,
        successful,
        evidence_id,
    )
    candidate_id = canonical_fingerprint(
        {
            "kind": "full-dark-sector-stage-candidate",
            "plan": plan.plan_id,
            "stage": proposed_state.stage.stage_id,
            "evidence": evidence_id,
            "stress": stress.assembly_id,
            "exchange": exchange.exchange_id,
        }
    )
    return FullDarkSectorStageCandidate(
        source_state,
        proposed_state,
        epoch_result,
        stress,
        radiation_source,
        evidence,
        candidate_id,
    )


def commit_full_dark_sector_stage(
    plan: FullDarkSectorRuntimePlan,
    coordinator: DarkSectorRunCoordinator,
    candidate: FullDarkSectorStageCandidate,
    /,
    *,
    entities: Sequence[GlobalEntity] = (),
    events: Sequence[GlobalEvent] = (),
    edges: Sequence[GlobalEventEdge] = (),
    work_items: Sequence[GlobalWorkItem] = (),
    committed_at: int | None = None,
) -> FullDarkSectorStageCommit:
    """Atomically publish a successful epoch or return the exact source state."""

    if not isinstance(plan, FullDarkSectorRuntimePlan):
        raise TypeError("plan must be FullDarkSectorRuntimePlan.")
    if not isinstance(coordinator, DarkSectorRunCoordinator):
        raise TypeError("coordinator must be DarkSectorRunCoordinator.")
    if not isinstance(candidate, FullDarkSectorStageCandidate):
        raise TypeError("candidate must be FullDarkSectorStageCandidate.")
    if coordinator.plan.plan_id != plan.epoch.plan_id:
        raise ValueError("Coordinator and full runtime use different epoch plans.")
    entities_ = tuple(entities)
    events_ = tuple(events)
    edges_ = tuple(edges)
    work_items_ = tuple(work_items)
    if any(not isinstance(value, GlobalEntity) for value in entities_):
        raise TypeError("entities must contain GlobalEntity values.")
    if any(not isinstance(value, GlobalEvent) for value in events_):
        raise TypeError("events must contain GlobalEvent values.")
    if any(not isinstance(value, GlobalEventEdge) for value in edges_):
        raise TypeError("edges must contain GlobalEventEdge values.")
    if any(not isinstance(value, GlobalWorkItem) for value in work_items_):
        raise TypeError("work_items must contain GlobalWorkItem values.")
    if any(not value.provenance_ids for value in entities_):
        raise ValueError("Every committed global entity requires provenance identities.")
    if any(not value.evidence_ids for value in events_):
        raise ValueError("Every committed global event requires evidence identities.")
    stage = candidate.proposed_state.stage
    if any(
        value.frame_id != stage.frame_id
        or value.frame_realization_id != stage.frame_realization_id
        or value.unit_contract_id != stage.unit_contract_id
        for value in entities_
    ):
        raise ValueError("Global entities do not bind the exact stage frame realization.")
    revision_id = _matrix_revision_id(plan.matrix_element_revision)
    allowed_revision_ids = {
        revision_id,
        plan.shower.model_revision_id,
        plan.hadronization.model_revision_id,
        plan.bound_states.model_revision_id,
        plan.decay_cascade.model_revision_id,
    }
    if any(
        value.model_revision_id not in allowed_revision_ids
        for value in (*events_, *work_items_)
    ):
        raise ValueError(
            "Event/work records do not bind a declared stage model revision."
        )
    committed = bool(np.asarray(candidate.evidence.successful))
    receipt = None
    if committed:
        receipt = coordinator.commit_epoch(
            candidate.epoch_result,
            entities=entities_,
            events=events_,
            edges=edges_,
            work_items=work_items_,
            matrix_element_revision_id=revision_id,
            evidence_ids=(candidate.evidence.evidence_id,),
            committed_at=committed_at,
        )
    state = candidate.proposed_state if committed else candidate.source_state
    commit_id = canonical_fingerprint(
        {
            "kind": "full-dark-sector-stage-commit",
            "candidate": candidate.candidate_id,
            "committed": committed,
            "epoch_receipt": None if receipt is None else receipt.receipt_id,
        }
    )
    return FullDarkSectorStageCommit(
        state,
        candidate.stress,
        candidate.evidence,
        receipt,
        committed,
        not committed,
        commit_id,
    )


class FullDarkSectorOutputBundle(StrictModule):
    state: FullDarkSectorCompositeState
    stress: FullDarkSectorStressAssembly
    ledger: FullDarkSectorStageLedger
    epoch_manifest: EventGraphEpochManifest
    finite: Array
    successful: Array
    profile_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    revision_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_lineage_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    output_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: FullDarkSectorRuntimePlan,
        commit: FullDarkSectorStageCommit,
        /,
    ):
        if not isinstance(plan, FullDarkSectorRuntimePlan):
            raise TypeError("plan must be FullDarkSectorRuntimePlan.")
        if not isinstance(commit, FullDarkSectorStageCommit) or not commit.committed:
            raise ValueError("Output requires one atomically committed full stage.")
        if commit.receipt is None:
            raise ValueError("Committed output requires a durable epoch receipt.")
        manifest = commit.receipt.manifest
        stage = commit.state.stage
        if (
            manifest.plan_id != plan.epoch.plan_id
            or manifest.capacity_revision_id != plan.epoch.capacity_revision_id
            or manifest.species_revision_id != plan.epoch.species_revision_id
            or manifest.topology_revision_id != plan.epoch.topology_revision_id
            or manifest.matrix_element_revision_id
            != _matrix_revision_id(plan.matrix_element_revision)
            or manifest.epoch_sequence != stage.epoch_sequence
        ):
            raise ValueError("Output epoch identities differ from the committed stage.")
        finite = commit.evidence.finite
        successful = finite & commit.evidence.successful
        self.state = commit.state
        self.stress = commit.stress
        self.ledger = commit.evidence.ledger
        self.epoch_manifest = manifest
        self.finite = finite
        self.successful = successful
        self.profile_ids = plan.profile_ids
        self.revision_ids = plan.revision_ids
        self.topology_id = stage.topology_id
        self.geometry_lineage_id = stage.geometry_lineage_id
        self.frame_id = stage.frame_id
        self.frame_realization_id = stage.frame_realization_id
        self.unit_contract_id = stage.unit_contract_id
        self.stage_id = stage.stage_id
        self.checkpoint_id = manifest.checkpoint_id
        self.output_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-output-bundle",
                "plan": plan.plan_id,
                "profiles": [list(value) for value in plan.profile_ids],
                "revisions": [list(value) for value in plan.revision_ids],
                "topology": self.topology_id,
                "geometry_lineage": self.geometry_lineage_id,
                "epoch": manifest.epoch_manifest_id,
                "frame": self.frame_id,
                "frame_realization": self.frame_realization_id,
                "units": self.unit_contract_id,
                "stage": self.stage_id,
                "checkpoint": self.checkpoint_id,
                "stress": commit.stress.assembly_id,
                "ledger": commit.evidence.ledger.ledger_id,
            }
        )


class FullDarkSectorResourceEvidence(StrictModule, NonTrainableState):
    resident_array_count: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    checkpoint_bytes: int = eqx.field(static=True)
    packet_capacity: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    product_capacity: int = eqx.field(static=True)
    radiation_capacity: int = eqx.field(static=True)
    work_capacity: int = eqx.field(static=True)
    frontier_capacity: int = eqx.field(static=True)
    fixed_capacity: bool = eqx.field(static=True)
    durable_unbounded_epochs: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @classmethod
    def from_state(
        cls,
        plan: FullDarkSectorRuntimePlan,
        state: FullDarkSectorCompositeState,
        /,
    ) -> "FullDarkSectorResourceEvidence":
        if not isinstance(plan, FullDarkSectorRuntimePlan) or not isinstance(
            state, FullDarkSectorCompositeState
        ):
            raise TypeError("Resource evidence requires the typed plan and state.")
        leaves = tuple(
            value for value in jax.tree.leaves(state) if isinstance(value, jax.Array)
        )
        resident = sum(value.size * np.dtype(value.dtype).itemsize for value in leaves)
        epoch = plan.epoch
        payload = resident
        identity = canonical_fingerprint(
            {
                "kind": "full-dark-sector-resource-evidence",
                "plan": plan.plan_id,
                "resident_array_count": len(leaves),
                "resident_bytes": resident,
                "checkpoint_bytes": payload,
                "capacities": [
                    epoch.packet_capacity,
                    epoch.event_capacity,
                    epoch.product_capacity,
                    epoch.radiation_capacity,
                    epoch.work_capacity,
                    epoch.frontier_capacity,
                ],
                "fixed_capacity": True,
                "durable_unbounded_epochs": True,
            }
        )
        return cls(
            len(leaves),
            resident,
            payload,
            epoch.packet_capacity,
            epoch.event_capacity,
            epoch.product_capacity,
            epoch.radiation_capacity,
            epoch.work_capacity,
            epoch.frontier_capacity,
            True,
            True,
            identity,
        )


class FullDarkSectorCheckpointPlan(StrictModule, NonTrainableState):
    """Distributed state checkpoint with exact profile/revision/stage binding."""

    runtime: FullDarkSectorRuntimePlan
    stage: FullDarkSectorStageToken
    epoch_manifest_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)
    analysis_plan_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: FullDarkSectorRuntimePlan,
        stage: FullDarkSectorStageToken,
        /,
        *,
        epoch_manifest_id: str,
    ):
        if not isinstance(runtime, FullDarkSectorRuntimePlan):
            raise TypeError("runtime must be FullDarkSectorRuntimePlan.")
        if not isinstance(stage, FullDarkSectorStageToken):
            raise TypeError("stage must be FullDarkSectorStageToken.")
        epoch_manifest = _identifier(epoch_manifest_id, "epoch_manifest_id")
        if (
            stage.epoch_plan_id != runtime.epoch.plan_id
            or stage.matrix_element_revision_id
            != _matrix_revision_id(runtime.matrix_element_revision)
        ):
            raise ValueError("Checkpoint stage identities differ from the runtime.")
        self.runtime = runtime
        self.stage = stage
        self.epoch_manifest_id = epoch_manifest
        self.analysis_plan_id = runtime.plan_id
        self.numeric_revision_id = stage.matrix_element_revision_id
        self.execution_plan_id = runtime.epoch.compile_signature_id
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-distributed-checkpoint",
                "runtime": runtime.plan_id,
                "profiles": [list(value) for value in runtime.profile_ids],
                "revisions": [list(value) for value in runtime.revision_ids],
                "topology": stage.topology_id,
                "geometry_lineage": stage.geometry_lineage_id,
                "topology_revision": stage.topology_revision_id,
                "epoch_manifest": epoch_manifest,
                "epoch_sequence": stage.epoch_sequence,
                "frame": stage.frame_id,
                "frame_realization": stage.frame_realization_id,
                "units": stage.unit_contract_id,
                "stage": stage.stage_id,
            }
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-checkpoint-plan",
                "checkpoint": self.checkpoint_id,
                "analysis": self.analysis_plan_id,
                "numeric_revision": self.numeric_revision_id,
                "execution": self.execution_plan_id,
            }
        )

    def validate_state(self, state: FullDarkSectorCompositeState, /) -> None:
        if not isinstance(state, FullDarkSectorCompositeState):
            raise TypeError("state must be FullDarkSectorCompositeState.")
        if state.stage.stage_id != self.stage.stage_id:
            raise ValueError("Checkpoint state does not match the exact stage identity.")
        if not bool(np.asarray(_state_consistency(self.runtime, state))):
            raise ValueError("Checkpoint state is not stage-consistent.")

    def _validate_parent_manifest(
        self, parent_manifest: CheckpointManifest | None, /
    ) -> None:
        if parent_manifest is None:
            return
        if not isinstance(parent_manifest, CheckpointManifest):
            raise TypeError("parent_manifest must be a CheckpointManifest or None.")
        if (
            not parent_manifest.complete
            or parent_manifest.analysis_plan_id != self.analysis_plan_id
            or parent_manifest.numeric_revision_id != self.numeric_revision_id
            or parent_manifest.execution_plan_id != self.execution_plan_id
            or parent_manifest.checkpoint_id == self.checkpoint_id
        ):
            raise ValueError(
                "Parent checkpoint is not the compatible current runtime checkpoint."
            )

    def publish(
        self,
        repository: ArtifactRepository,
        state: FullDarkSectorCompositeState,
        /,
        *,
        writer_id: str,
        attempt_id: str | None = None,
        encoding: ChunkEncoding = "identity",
        parent_manifest: CheckpointManifest | None = None,
    ) -> ProcessCheckpointPublication:
        self.validate_state(state)
        self._validate_parent_manifest(parent_manifest)
        return publish_process_checkpoint(
            repository,
            self.checkpoint_id,
            self.execution_plan_id,
            state,
            analysis_plan_id=self.analysis_plan_id,
            numeric_revision_id=self.numeric_revision_id,
            writer_id=writer_id,
            attempt_id=attempt_id,
            topology_epoch=self.stage.epoch_sequence,
            encoding=encoding,
            parent_manifest=parent_manifest,
        )

    def assemble(
        self,
        publications: Sequence[ProcessCheckpointPublication],
        /,
        *,
        expected_process_count: int,
        parent_manifest: CheckpointManifest | None = None,
        diagnostic_ids: Sequence[str] = (),
    ) -> CheckpointManifest:
        self._validate_parent_manifest(parent_manifest)
        return assemble_distributed_checkpoint_manifest(
            self.checkpoint_id,
            self.analysis_plan_id,
            self.numeric_revision_id,
            self.execution_plan_id,
            publications,
            expected_process_count=expected_process_count,
            parent_manifest=parent_manifest,
            diagnostic_ids=diagnostic_ids,
        )

    def restore(
        self,
        repository: ArtifactRepository,
        manifest: CheckpointManifest,
        target_template: FullDarkSectorCompositeState,
        /,
    ) -> FullDarkSectorCompositeState:
        """Restore arrays directly into target shardings; static identities cannot change."""

        self.validate_state(target_template)
        if (
            not isinstance(manifest, CheckpointManifest)
            or not manifest.complete
            or manifest.checkpoint_id != self.checkpoint_id
            or manifest.analysis_plan_id != self.analysis_plan_id
            or manifest.numeric_revision_id != self.numeric_revision_id
            or manifest.execution_plan_id != self.execution_plan_id
        ):
            raise ValueError(
                "Checkpoint manifest identities differ from the restore plan."
            )
        flattened, structure = jax.tree_util.tree_flatten_with_path(target_template)
        inventory: dict[str, tuple[tuple[int, ...], str]] = {}
        for shard in manifest.shards:
            metadata = dict(shard.metadata)
            path = metadata.get("array_path")
            shape = metadata.get("global_shape")
            dtype = metadata.get("dtype")
            if path is None or shape is None or dtype is None:
                raise ValueError("Checkpoint shard lacks array reconstruction metadata.")
            record = (tuple(json.loads(shape)), dtype)
            if path in inventory and inventory[path] != record:
                raise ValueError("Checkpoint array inventory is inconsistent.")
            inventory[path] = record
        expected_paths = tuple(
            jax.tree_util.keystr(path) or "<root>" for path, _ in flattened
        )
        if set(inventory) != set(expected_paths):
            raise ValueError(
                "Checkpoint array inventory differs from the composite state."
            )
        restored = []
        for path, leaf in flattened:
            array_path = jax.tree_util.keystr(path) or "<root>"
            expected = (tuple(leaf.shape), np.dtype(leaf.dtype).str)
            if inventory[array_path] != expected:
                raise ValueError(
                    "Checkpoint shape or dtype differs from the target state."
                )
            sharding = (
                leaf.sharding
                if isinstance(leaf, jax.Array)
                else SingleDeviceSharding(jax.devices()[0])
            )
            restored.append(
                restore_global_array_from_checkpoint(
                    repository, manifest, array_path, sharding
                )
            )
        result = jax.tree.unflatten(structure, restored)
        self.validate_state(result)
        return result


__all__ = [
    "FullDarkSectorCheckpointPlan",
    "FullDarkSectorCompositeState",
    "FullDarkSectorOutputBundle",
    "FullDarkSectorResourceEvidence",
    "FullDarkSectorRuntimePlan",
    "FullDarkSectorStageCandidate",
    "FullDarkSectorStageCommit",
    "FullDarkSectorStageEvidence",
    "FullDarkSectorStageLedger",
    "FullDarkSectorStageToken",
    "FullDarkSectorStressAssembly",
    "NamedStressEnergyComponent",
    "assemble_full_dark_sector_stress",
    "commit_full_dark_sector_stage",
    "einstein_vlasov_stress_component",
    "propose_full_dark_sector_stage",
    "radiation_stress_component",
    "weak_gravity_stress_component",
]
