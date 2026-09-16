#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit adapters and gravity coupling for dark-radiation transport."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._dark_radiation_moments import (
    CosmologicalMultigroupM1System,
    DarkRadiationBoltzmannHierarchyPlan,
    DarkRadiationConversionReceipt,
    DarkRadiationFourForce,
    DarkRadiationHierarchyState,
    DarkRadiationVETResult,
)
from ...metrix._adm_exchange import StressEnergyProjection
from ...solver._dark_radiation_packets import (
    DarkRadiationPacketAdmissionResult,
    DarkRadiationPacketPlan,
    DarkRadiationPacketState,
)
from ..relativistic_scattering._unit_contract import LocalRelativisticFramePlan
from ._dark_radiation import DarkRadiationLedger, DarkRadiationLedgerPlan


_M1_CHECKPOINT_FORMAT = "phydrax-dark-radiation-m1-distributed"


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty stripped string.")
    return value


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    result = tuple(_identifier(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values.")
    return result


class DarkRadiationTransportProfile(StrictModule, NonTrainableState):
    """Auditable support/refusal/differentiation contract for one profile."""

    representation: str = eqx.field(static=True)
    supported_physics: tuple[str, ...] = eqx.field(static=True)
    refusal_conditions: tuple[str, ...] = eqx.field(static=True)
    differentiation_policy: str = eqx.field(static=True)
    production_evidence: tuple[str, ...] = eqx.field(static=True)
    checkpoint_product: str = eqx.field(static=True)
    output_product: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        representation: str,
        supported_physics: Sequence[str],
        refusal_conditions: Sequence[str],
        differentiation_policy: str,
        production_evidence: Sequence[str],
        /,
        *,
        checkpoint_product: str,
        output_product: str,
    ):
        kind = _identifier(representation, "representation")
        support = _identifiers(supported_physics, "supported_physics")
        refusals = _identifiers(refusal_conditions, "refusal_conditions")
        differentiation = _identifier(differentiation_policy, "differentiation_policy")
        evidence = _identifiers(production_evidence, "production_evidence")
        checkpoint = _identifier(checkpoint_product, "checkpoint_product")
        output = _identifier(output_product, "output_product")
        if not support or not refusals or not evidence:
            raise ValueError(
                "Dark-radiation profile support, refusals, and evidence are required."
            )
        if checkpoint == output:
            raise ValueError("Checkpoint and analysis-output products must be distinct.")
        self.representation = kind
        self.supported_physics = support
        self.refusal_conditions = refusals
        self.differentiation_policy = differentiation
        self.production_evidence = evidence
        self.checkpoint_product = checkpoint
        self.output_product = output
        self.profile_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-transport-profile",
                "representation": kind,
                "supported_physics": support,
                "refusal_conditions": refusals,
                "differentiation_policy": differentiation,
                "production_evidence": evidence,
                "checkpoint_product": checkpoint,
                "output_product": output,
            }
        )


class DarkRadiationSourceAdapterEvidence(StrictModule, NonTrainableState):
    ledger_valid: Array
    frame_admissible: Array
    support_complete: Array
    four_momentum_defect: Array
    admitted_count: Array
    successful: Array
    adapter_id: str = eqx.field(static=True)


class DarkRadiationSourceAdapterResult(StrictModule, NonTrainableState):
    admission: DarkRadiationPacketAdmissionResult
    evidence: DarkRadiationSourceAdapterEvidence
    source_ledger_id: str = eqx.field(static=True)
    target_state_id: str = eqx.field(static=True)


class DarkRadiationLedgerSourceAdapter(StrictModule, NonTrainableState):
    """Typed reaction/decay ledger to packet-state boundary.

    The legacy ledger owns emission accounting only. This adapter explicitly
    declares coordinate-to-tetrad conversion and all transport-only state; it
    never treats the ledger itself as a transport state.
    """

    ledger_plan: DarkRadiationLedgerPlan
    packet_plan: DarkRadiationPacketPlan
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        ledger_plan: DarkRadiationLedgerPlan,
        packet_plan: DarkRadiationPacketPlan,
        /,
    ):
        if not isinstance(ledger_plan, DarkRadiationLedgerPlan):
            raise TypeError("ledger_plan must be DarkRadiationLedgerPlan.")
        if not isinstance(packet_plan, DarkRadiationPacketPlan):
            raise TypeError("packet_plan must be DarkRadiationPacketPlan.")
        if ledger_plan.capacity > packet_plan.capacity:
            raise ValueError(
                "Packet transport capacity is smaller than source ledger capacity."
            )
        if ledger_plan.dimension != 3:
            raise ValueError(
                "Relativistic dark-radiation transport requires three dimensions."
            )
        if ledger_plan.speed_of_light != packet_plan.physical_light_speed:
            raise ValueError("Ledger and transport physical light speeds differ.")
        if ledger_plan.position_unit != "comoving-length":
            raise ValueError("Ledger positions must explicitly use comoving-length.")
        self.ledger_plan = ledger_plan
        self.packet_plan = packet_plan
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-ledger-source-adapter",
                "ledger_plan": ledger_plan.plan_id,
                "packet_plan": packet_plan.plan_id,
                "source_basis": "coordinate-physical-(E,p)",
                "target_basis": "local-tetrad-(E,c*p)",
            }
        )

    def adapt(
        self,
        ledger: DarkRadiationLedger,
        target: DarkRadiationPacketState,
        frame: LocalRelativisticFramePlan,
        generation: ArrayLike,
        stokes: ArrayLike,
        optical_depth_threshold: ArrayLike,
        rng_keys: ArrayLike,
        owner: ArrayLike,
        weight: ArrayLike,
        /,
        *,
        target_state_id: str,
    ) -> DarkRadiationSourceAdapterResult:
        if not isinstance(frame, LocalRelativisticFramePlan):
            raise TypeError("frame must be LocalRelativisticFramePlan.")
        if frame.units.contract_id != self.packet_plan.units.contract_id:
            raise ValueError("Source adapter frame and packet plan use different units.")
        ledger_valid = self.ledger_plan.valid(ledger)
        active = ledger.active_mask
        coordinate_four = frame.units.assemble_four_momentum(
            ledger.physical_energy, ledger.physical_momentum
        )
        local_four = frame.coordinate_to_local(coordinate_four)
        energy = local_four[:, 0]
        groups = jnp.searchsorted(self.packet_plan.group_edges, energy, side="right") - 1
        support_complete = jnp.all(
            ~active
            | (
                (groups >= 0)
                & (groups < self.packet_plan.group_edges.size - 1)
                & (energy >= self.packet_plan.group_edges[0])
                & (energy < self.packet_plan.group_edges[-1])
            )
        )
        parents = ledger.parent_ids
        admission = self.packet_plan.admit(
            target,
            ledger.packet_ids,
            ledger.species_ids,
            parents,
            ledger.source_event_ids,
            generation,
            ledger.comoving_position,
            local_four,
            groups,
            stokes,
            optical_depth_threshold,
            rng_keys,
            owner,
            weight,
            active & support_complete & ledger_valid,
        )
        stored_coordinate = frame.local_to_coordinate(local_four)
        defect = jnp.where(active[:, None], stored_coordinate - coordinate_four, 0.0)
        successful = (
            ledger_valid
            & jnp.all(frame.admissible)
            & support_complete
            & jnp.all(defect == 0.0)
            & admission.successful
        )
        source_ledger_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-source-ledger",
                "ledger_plan": self.ledger_plan.plan_id,
                "packet_ids": array_tree_fingerprint(ledger.packet_ids),
                "active": array_tree_fingerprint(active),
                "frame": frame.frame_id,
            }
        )
        evidence = DarkRadiationSourceAdapterEvidence(
            ledger_valid,
            jnp.all(frame.admissible),
            support_complete,
            defect,
            jnp.sum(active),
            successful,
            self.adapter_id,
        )
        return DarkRadiationSourceAdapterResult(
            admission,
            evidence,
            source_ledger_id,
            _identifier(target_state_id, "target_state_id"),
        )


class DarkRadiationPacketMomentResult(StrictModule, NonTrainableState):
    moment_state: Array
    group_stokes: Array
    group_spectrum: Array
    active_cell: Array
    receipt: DarkRadiationConversionReceipt
    frame_token: Array
    observer_coordinates: Array
    coordinate_time: Array
    scale_factor: Array
    source_state_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    target_state_id: str = eqx.field(static=True)


def average_packets_to_m1(
    packet_plan: DarkRadiationPacketPlan,
    packet_state: DarkRadiationPacketState,
    m1: CosmologicalMultigroupM1System,
    cell_indices: ArrayLike,
    cell_volumes: ArrayLike,
    /,
    *,
    target_state_id: str,
) -> DarkRadiationPacketMomentResult:
    """Explicit volume/frequency/angular average with a conservation receipt."""

    if not isinstance(packet_plan, DarkRadiationPacketPlan):
        raise TypeError("packet_plan must be DarkRadiationPacketPlan.")
    if not isinstance(m1, CosmologicalMultigroupM1System):
        raise TypeError("m1 must be CosmologicalMultigroupM1System.")
    if m1.group_count != packet_plan.group_edges.size - 1 or not bool(
        np.array_equal(np.asarray(m1.group_edges), np.asarray(packet_plan.group_edges))
    ):
        raise ValueError("Packet and M1 frequency groups must match exactly.")
    if m1.physical_light_speed != packet_plan.physical_light_speed:
        raise ValueError("Packet and M1 physical light speeds differ.")
    packet_state = eqx.error_if(
        packet_state,
        ~packet_plan.valid(packet_state),
        "Packet-to-M1 averaging requires a valid packet state.",
    )
    cells = jnp.asarray(cell_indices, dtype=jnp.int32)
    volumes = jnp.asarray(cell_volumes, dtype=packet_state.weight.dtype)
    if cells.shape != (packet_plan.capacity,) or volumes.ndim != 1:
        raise ValueError("Packet cell indices or cell volumes have invalid shape.")
    cell_count = volumes.size
    active = packet_state.active_mask
    packet_state = eqx.error_if(
        packet_state,
        ~jnp.all(~active | ((cells >= 0) & (cells < cell_count)))
        | ~jnp.all(jnp.isfinite(volumes))
        | ~jnp.all(volumes > 0.0),
        "Packet-to-M1 cell support is invalid.",
    )
    safe_cells = jnp.where(active, cells, 0)
    safe_groups = jnp.where(active, packet_state.frequency_group, 0)
    segment = safe_cells * m1.group_count + safe_groups
    segment_count = cell_count * m1.group_count
    energy = packet_state.weight * packet_state.tetrad_four_momentum[:, 0]
    cp = packet_state.tetrad_four_momentum[:, 1:]
    flux = packet_plan.physical_light_speed * packet_state.weight[:, None] * cp
    contributions = jnp.concatenate((energy[:, None], flux), axis=-1)
    contributions = jnp.where(active[:, None], contributions, 0.0)
    grouped = jax.ops.segment_sum(contributions, segment, segment_count)
    grouped = grouped.reshape((cell_count, m1.group_count, m1.group_width))
    grouped = grouped / volumes[:, None, None]
    moment_state = grouped.reshape((cell_count, -1))
    weighted_stokes = packet_state.weight[:, None] * packet_state.stokes
    stokes_sum = jax.ops.segment_sum(
        jnp.where(active[:, None], weighted_stokes, 0.0), segment, segment_count
    ).reshape((cell_count, m1.group_count, 4))
    spectrum = grouped[..., 0]
    active_cell = (
        jax.ops.segment_sum(active.astype(jnp.int32), safe_cells, cell_count) > 0
    )
    source_integral = jnp.sum(
        jnp.where(
            active[:, None],
            packet_state.weight[:, None] * packet_state.tetrad_four_momentum,
            0.0,
        ),
        axis=0,
    )
    target_energy = jnp.sum(spectrum * volumes[:, None])
    target_cp = (
        jnp.sum(grouped[..., 1:] * volumes[:, None, None], axis=(0, 1))
        / packet_plan.physical_light_speed
    )
    target_integral = jnp.concatenate((target_energy[None], target_cp))
    target_id = _identifier(target_state_id, "target_state_id")
    receipt = DarkRadiationConversionReceipt(
        source_integral,
        target_integral,
        source_state_id=packet_state.epoch_manifest_id,
        target_state_id=target_id,
        source_representation="relativistic-packets",
        target_representation="multigroup-m1",
        operation="volume-frequency-angular-average",
        differentiation_policy="stop-gradient-discrete-membership",
    )
    return DarkRadiationPacketMomentResult(
        moment_state,
        stokes_sum,
        spectrum,
        active_cell,
        receipt,
        packet_state.frame_token,
        packet_state.observer_coordinates,
        packet_state.coordinate_time,
        packet_state.scale_factor,
        packet_state.epoch_manifest_id,
        packet_state.frame_realization_id,
        target_id,
    )


class DarkRadiationHierarchyLinearizationResult(StrictModule, NonTrainableState):
    hierarchy: DarkRadiationHierarchyState
    receipt: DarkRadiationConversionReceipt
    background_spectrum: Array
    perturbation_spectrum: Array


def linearize_m1_to_hierarchy(
    hierarchy_plan: DarkRadiationBoltzmannHierarchyPlan,
    m1_group_energy: ArrayLike,
    background_group_energy: ArrayLike,
    group_to_momentum_weights: ArrayLike,
    /,
    *,
    scale_factor: ArrayLike,
    conformal_time: ArrayLike,
    source_state_id: str,
    target_state_id: str,
) -> DarkRadiationHierarchyLinearizationResult:
    """Explicit homogeneous-background linearization; never an implicit cast."""

    if not isinstance(hierarchy_plan, DarkRadiationBoltzmannHierarchyPlan):
        raise TypeError("hierarchy_plan must be DarkRadiationBoltzmannHierarchyPlan.")
    energy = jnp.asarray(m1_group_energy)
    background = jnp.asarray(background_group_energy, dtype=energy.dtype)
    mapping = jnp.asarray(group_to_momentum_weights, dtype=energy.dtype)
    if energy.ndim != 2 or background.shape != energy.shape[1:]:
        raise ValueError(
            "M1 linearization expects (k, group) energy and group background."
        )
    if energy.shape[0] != hierarchy_plan.wave_numbers.size or mapping.shape != (
        background.size,
        hierarchy_plan.momentum_nodes.size,
    ):
        raise ValueError("M1-to-hierarchy group/momentum mapping has invalid shape.")
    background = eqx.error_if(
        background,
        ~jnp.all(background > 0.0),
        "Hierarchy linearization background must be positive.",
    )
    perturbation = energy - background[None, :]
    monopole = contract(
        "kg,gq->kq", perturbation / background[None, :], mapping, backend="jax"
    )
    intensity = (
        jnp.zeros(hierarchy_plan.shape, dtype=energy.dtype).at[..., 0].set(monopole)
    )
    hierarchy = hierarchy_plan.initialize(
        intensity,
        scale_factor=scale_factor,
        conformal_time=conformal_time,
        state_id=target_state_id,
    )
    source_integral = jnp.sum(perturbation, axis=-1)
    reconstructed = contract("kq,gq->kg", monopole, mapping, backend="jax")
    target_integral = jnp.sum(reconstructed * background[None, :], axis=-1)
    receipt = DarkRadiationConversionReceipt(
        source_integral,
        target_integral,
        source_state_id=source_state_id,
        target_state_id=target_state_id,
        source_representation="multigroup-m1",
        target_representation="linear-boltzmann-hierarchy",
        operation="background-linearization-and-momentum-projection",
        differentiation_policy="differentiable-fixed-quadrature",
    )
    return DarkRadiationHierarchyLinearizationResult(
        hierarchy, receipt, background, perturbation
    )


class DarkRadiationGravitySource(StrictModule, NonTrainableState):
    """Co-temporal gravity projection and opposite-sign material exchange."""

    projection: StressEnergyProjection
    exchange: DarkRadiationFourForce
    spectrum: Array
    polarization: Array
    source_state_id: str = eqx.field(static=True)
    endpoint_time: Array
    frame_token: Array
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        projection: StressEnergyProjection,
        exchange: DarkRadiationFourForce,
        spectrum: ArrayLike,
        polarization: ArrayLike,
        /,
        *,
        source_state_id: str,
        endpoint_time: ArrayLike,
        frame_token: ArrayLike,
        frame_id: str,
        unit_contract_id: str,
        frame_realization_id: str,
    ):
        if not isinstance(projection, StressEnergyProjection):
            raise TypeError("projection must be StressEnergyProjection.")
        if not isinstance(exchange, DarkRadiationFourForce):
            raise TypeError("exchange must be DarkRadiationFourForce.")
        source = _identifier(source_state_id, "source_state_id")
        frame = _identifier(frame_id, "frame_id")
        units = _identifier(unit_contract_id, "unit_contract_id")
        realization = _identifier(frame_realization_id, "frame_realization_id")
        time = jnp.asarray(endpoint_time)
        token = jnp.asarray(frame_token)
        if time.shape != () or token.shape != () or exchange.endpoint_time.shape != ():
            raise ValueError(
                "Gravity-source endpoint time and frame_token must be scalar."
            )
        if (
            exchange.source_state_id != source
            or exchange.frame_id != frame
            or exchange.frame_realization_id != realization
            or exchange.unit_contract_id != units
        ):
            raise ValueError(
                "Gravity projection and four-force source identities differ."
            )
        if not bool(
            np.asarray((exchange.endpoint_time == time) & (exchange.frame_token == token))
        ):
            raise ValueError("Gravity projection and four-force are not co-temporal.")
        self.projection = projection
        self.exchange = exchange
        self.spectrum = jnp.asarray(spectrum)
        self.polarization = jnp.asarray(polarization)
        self.source_state_id = source
        self.endpoint_time = time
        self.frame_token = token
        self.frame_id = frame
        self.frame_realization_id = realization
        self.unit_contract_id = units
        self.source_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-gravity-source",
                "source_state": source,
                "projection": projection.projection_id,
                "exchange": exchange.exchange_id,
                "frame": frame,
                "frame_realization": realization,
                "units": units,
            }
        )


def packet_gravity_source(
    plan: DarkRadiationPacketPlan,
    state: DarkRadiationPacketState,
    frame: LocalRelativisticFramePlan,
    exchange: DarkRadiationFourForce,
    physical_volume: ArrayLike,
    /,
) -> DarkRadiationGravitySource:
    """Deposit local packet stress-energy into one exact ADM frame snapshot."""

    state = eqx.error_if(
        state,
        ~plan.valid(state),
        "Packet gravity source requires a valid packet state.",
    )
    if (
        state.frame_id != frame.frame_id
        or state.frame_realization_id != frame.realization_id()
        or frame.units.contract_id != plan.units.contract_id
    ):
        raise ValueError("Packet gravity source frame identity mismatch.")
    state = eqx.error_if(
        state,
        state.frame_token != frame.frame_token,
        "Packet gravity source frame token mismatch.",
    )
    volume = jnp.asarray(physical_volume, dtype=state.weight.dtype)
    if volume.shape != ():
        raise ValueError("Packet gravity source volume must be scalar.")
    volume = eqx.error_if(
        volume,
        ~jnp.isfinite(volume) | (volume <= 0.0),
        "Packet gravity source volume must be finite and positive.",
    )
    active = state.active_mask
    weighted = jnp.where(active, state.weight, 0.0)
    four = state.tetrad_four_momentum
    energy = jnp.sum(weighted * four[:, 0]) / volume
    momentum = jnp.sum(weighted[:, None] * four[:, 1:], axis=0) / (
        volume * plan.physical_light_speed
    )
    denominator = jnp.where(four[:, 0] > 0.0, four[:, 0], 1.0)
    stress = (
        contract(
            "n,ni,nj,n->ij",
            weighted,
            four[:, 1:],
            four[:, 1:],
            1.0 / denominator,
            backend="jax",
        )
        / volume
    )
    symmetry = jnp.max(jnp.abs(stress - jnp.swapaxes(stress, -1, -2)))
    projection_id = canonical_fingerprint(
        {
            "kind": "dark-radiation-packet-stress-energy",
            "plan": plan.plan_id,
            "source_state": state.epoch_manifest_id,
            "frame": frame.frame_id,
        }
    )
    projection = StressEnergyProjection(
        energy,
        momentum,
        stress,
        frame.geometry.active,
        frame.admissible & plan.valid(state),
        symmetry,
        jnp.asarray(0.0, dtype=energy.dtype),
        snapshot_token=frame.geometry.snapshot_token,
        geometry_lineage_id=frame.geometry.geometry_lineage_id,
        convention_id=frame.geometry.convention_id,
        scale_id=frame.geometry.scale_id,
        topology_id=frame.geometry.topology_id,
        projection_id=projection_id,
    )
    group_count = plan.group_edges.size - 1
    safe_group = jnp.where(state.active_mask, state.frequency_group, 0)
    spectrum = (
        jax.ops.segment_sum(weighted * four[:, 0], safe_group, group_count) / volume
    )
    polarization = jnp.sum(weighted[:, None] * state.stokes, axis=0) / volume
    return DarkRadiationGravitySource(
        projection,
        exchange,
        spectrum,
        polarization,
        source_state_id=state.epoch_manifest_id,
        endpoint_time=state.coordinate_time,
        frame_token=state.frame_token,
        frame_id=frame.frame_id,
        unit_contract_id=plan.units.contract_id,
        frame_realization_id=state.frame_realization_id,
    )


def m1_gravity_source(
    system: CosmologicalMultigroupM1System,
    state: ArrayLike,
    frame: LocalRelativisticFramePlan,
    exchange: DarkRadiationFourForce,
    group_stokes: ArrayLike,
    /,
    *,
    source_state_id: str,
) -> DarkRadiationGravitySource:
    values = jnp.asarray(state)
    projection = system.stress_energy_projection(
        values, frame.geometry, source_state_id=source_state_id
    )
    groups = system._groups(values)
    spectrum = groups[..., 0]
    polarization = jnp.asarray(group_stokes, dtype=values.dtype)
    if exchange.source_state_id != source_state_id:
        raise ValueError("M1 gravity source and exchange use different source states.")
    return DarkRadiationGravitySource(
        projection,
        exchange,
        spectrum,
        polarization,
        source_state_id=source_state_id,
        frame_token=frame.frame_token,
        endpoint_time=exchange.endpoint_time,
        frame_id=frame.frame_id,
        frame_realization_id=frame.realization_id(),
        unit_contract_id=frame.units.contract_id,
    )


def hierarchy_gravity_source(
    plan: DarkRadiationBoltzmannHierarchyPlan,
    state: DarkRadiationHierarchyState,
    linearization_receipt: DarkRadiationConversionReceipt,
    frame: LocalRelativisticFramePlan,
    exchange: DarkRadiationFourForce,
    background_energy_density: ArrayLike,
    mode_to_geometry: ArrayLike,
    wave_directions: ArrayLike,
    /,
) -> DarkRadiationGravitySource:
    """Reconstruct linear hierarchy stress only with an explicit receipt."""

    if not isinstance(plan, DarkRadiationBoltzmannHierarchyPlan):
        raise TypeError("plan must be DarkRadiationBoltzmannHierarchyPlan.")
    if not isinstance(state, DarkRadiationHierarchyState):
        raise TypeError("state must be DarkRadiationHierarchyState.")
    if not isinstance(linearization_receipt, DarkRadiationConversionReceipt):
        raise TypeError("linearization_receipt must be DarkRadiationConversionReceipt.")
    if (
        linearization_receipt.target_state_id != state.state_id
        or linearization_receipt.target_representation != "linear-boltzmann-hierarchy"
    ):
        raise ValueError("Hierarchy gravity requires its linearization receipt.")
    state = eqx.error_if(
        state,
        ~linearization_receipt.accepted,
        "Hierarchy gravity requires an accepted linearization receipt.",
    )
    if (
        state.frame_id != frame.frame_id
        or state.frame_realization_id != frame.realization_id()
    ):
        raise ValueError("Hierarchy gravity frame realization mismatch.")
    state = eqx.error_if(
        state,
        state.frame_token != frame.frame_token,
        "Hierarchy gravity frame token mismatch.",
    )
    output = plan.line_of_sight(state)
    mapping = jnp.asarray(mode_to_geometry, dtype=state.intensity.dtype)
    directions = jnp.asarray(wave_directions, dtype=state.intensity.dtype)
    background = jnp.asarray(background_energy_density, dtype=state.intensity.dtype)
    if mapping.shape != frame.geometry.leading_shape + (plan.wave_numbers.size,):
        raise ValueError("mode_to_geometry must map every k mode to each ADM lane.")
    if directions.shape != (plan.wave_numbers.size, 3):
        raise ValueError("wave_directions must have shape (k, 3).")
    if background.shape not in ((), frame.geometry.leading_shape):
        raise ValueError("background_energy_density must be scalar or match ADM lanes.")
    direction_norm = jnp.sqrt(jnp.sum(directions**2, axis=-1))
    normalized = (
        directions / jnp.where(direction_norm > 0.0, direction_norm, 1.0)[:, None]
    )
    delta = contract("...k,k->...", mapping, output.density_contrast, backend="jax")
    velocity_mode = output.velocity_divergence / jnp.where(
        plan.wave_numbers > 0.0, plan.wave_numbers, 1.0
    )
    velocity = contract(
        "...k,k,ki->...i", mapping, velocity_mode, normalized, backend="jax"
    )
    anisotropy = contract(
        "...k,k,ki,kj->...ij",
        mapping,
        output.anisotropic_stress,
        normalized,
        normalized,
        backend="jax",
    )
    identity = jnp.eye(3, dtype=state.intensity.dtype)
    energy = background * (1.0 + delta)
    physical_c = jnp.asarray(float(frame.units.speed_of_light), dtype=energy.dtype)
    momentum = background[..., None] * velocity / physical_c
    stress = energy[..., None, None] * identity / 3.0 + background[..., None, None] * (
        anisotropy
        - jnp.trace(anisotropy, axis1=-2, axis2=-1)[..., None, None] * identity / 3.0
    )
    symmetry = jnp.max(jnp.abs(stress - jnp.swapaxes(stress, -1, -2)), axis=(-2, -1))
    finite = (
        jnp.isfinite(energy)
        & jnp.all(jnp.isfinite(momentum), axis=-1)
        & jnp.all(jnp.isfinite(stress), axis=(-2, -1))
        & (energy >= 0.0)
    )
    projection_id = canonical_fingerprint(
        {
            "kind": "dark-radiation-hierarchy-stress-energy",
            "plan": plan.plan_id,
            "source_state": state.state_id,
            "linearization_receipt": linearization_receipt.receipt_id,
            "frame_realization": state.frame_realization_id,
        }
    )
    projection = StressEnergyProjection(
        energy,
        momentum,
        stress,
        frame.geometry.active,
        finite & frame.admissible,
        symmetry,
        jnp.zeros_like(energy),
        snapshot_token=frame.frame_token,
        geometry_lineage_id=frame.geometry.geometry_lineage_id,
        convention_id=frame.geometry.convention_id,
        scale_id=frame.geometry.scale_id,
        topology_id=frame.geometry.topology_id,
        projection_id=projection_id,
    )
    polarization = jnp.stack((output.polarization_e, output.polarization_b), axis=-1)
    return DarkRadiationGravitySource(
        projection,
        exchange,
        output.density_contrast,
        polarization,
        source_state_id=state.state_id,
        endpoint_time=state.conformal_time,
        frame_token=state.frame_token,
        frame_id=state.frame_id,
        frame_realization_id=state.frame_realization_id,
        unit_contract_id=state.unit_contract_id,
    )


def vet_stress_energy_projection(
    result: DarkRadiationVETResult,
    energy_density: ArrayLike,
    flux_covector: ArrayLike,
    frame: LocalRelativisticFramePlan,
    /,
    *,
    source_state_id: str,
) -> StressEnergyProjection:
    """Couple an accepted VET tensor to gravity without converting to M1."""

    if not isinstance(result, DarkRadiationVETResult):
        raise TypeError("result must be DarkRadiationVETResult.")
    energy = jnp.asarray(energy_density)
    flux = jnp.asarray(flux_covector, dtype=energy.dtype)
    if result.eddington_tensor.shape != energy.shape + (
        3,
        3,
    ) or flux.shape != energy.shape + (3,):
        raise ValueError("VET stress inputs must match tensor lanes.")
    if result.evidence.source_state_id != source_state_id:
        raise ValueError("VET result and stress source identities differ.")
    physical_c = jnp.asarray(float(frame.units.speed_of_light), dtype=energy.dtype)
    stress = energy[..., None, None] * result.eddington_tensor
    symmetry = jnp.max(jnp.abs(stress - jnp.swapaxes(stress, -1, -2)), axis=(-2, -1))
    valid = (
        result.evidence.accepted & (energy >= 0.0) & jnp.all(jnp.isfinite(flux), axis=-1)
    )
    projection_id = canonical_fingerprint(
        {
            "kind": "dark-radiation-vet-stress-energy",
            "tensor": result.evidence.tensor_id,
            "source_state": source_state_id,
            "frame_realization": frame.realization_id(),
        }
    )
    return StressEnergyProjection(
        energy,
        flux / physical_c**2,
        stress,
        frame.geometry.active,
        valid & frame.admissible,
        symmetry,
        jnp.zeros_like(energy),
        snapshot_token=frame.frame_token,
        geometry_lineage_id=frame.geometry.geometry_lineage_id,
        convention_id=frame.geometry.convention_id,
        scale_id=frame.geometry.scale_id,
        topology_id=frame.geometry.topology_id,
        projection_id=projection_id,
    )


class DarkRadiationM1Checkpoint(StrictModule, NonTrainableState):
    state: Array
    reflux_register: Array
    global_cell_ids: Array
    owner: Array
    group_edges: Array
    collective_count: Array
    reflux_residual: Array
    epoch_sequence: Array
    frame_token: Array
    observer_coordinates: Array
    coordinate_time: Array
    scale_factor: Array
    plan_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    epoch_manifest_id: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: ArrayLike,
        reflux_register: ArrayLike,
        global_cell_ids: ArrayLike,
        owner: ArrayLike,
        group_edges: ArrayLike,
        collective_count: ArrayLike,
        reflux_residual: ArrayLike,
        epoch_sequence: ArrayLike,
        /,
        *,
        frame: LocalRelativisticFramePlan,
        system_id: str,
        topology_id: str,
        partition_id: str,
        epoch_manifest_id: str,
    ):
        if not isinstance(frame, LocalRelativisticFramePlan):
            raise TypeError("frame must be LocalRelativisticFramePlan.")
        values = jnp.asarray(state)
        reflux = jnp.asarray(reflux_register, dtype=values.dtype)
        identifiers = jnp.asarray(global_cell_ids, dtype=jnp.int64)
        owners = jnp.asarray(owner, dtype=jnp.int32)
        edges = jnp.asarray(group_edges, dtype=values.dtype)
        collectives = jnp.asarray(collective_count, dtype=jnp.int64)
        residual = jnp.asarray(reflux_residual, dtype=values.dtype)
        epoch = jnp.asarray(epoch_sequence, dtype=jnp.int64)
        frame_token = jnp.asarray(frame.frame_token)
        observer = jnp.asarray(frame.observer_coordinates, dtype=values.dtype)
        time = jnp.asarray(frame.time, dtype=values.dtype)
        scale = jnp.asarray(frame.scale_factor, dtype=values.dtype)
        if (
            values.ndim != 2
            or reflux.shape != values.shape
            or identifiers.shape != values.shape[:1]
            or owners.shape != identifiers.shape
        ):
            raise ValueError("M1 checkpoint fixed arrays have inconsistent shapes.")
        if residual.shape != values.shape or collectives.shape != () or epoch.shape != ():
            raise ValueError("M1 checkpoint evidence arrays have invalid shapes.")
        if (
            frame_token.shape != ()
            or not bool(np.asarray(jnp.all(jnp.isfinite(observer))))
            or not bool(np.asarray(jnp.all(jnp.isfinite(time))))
            or not bool(np.asarray(jnp.all(jnp.isfinite(scale) & (scale > 0.0))))
        ):
            raise ValueError("M1 checkpoint frame realization is invalid.")
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("M1 checkpoint group edges are invalid.")
        system = _identifier(system_id, "system_id")
        topology = _identifier(topology_id, "topology_id")
        partition = _identifier(partition_id, "partition_id")
        manifest = _identifier(epoch_manifest_id, "epoch_manifest_id")
        self.state = values
        self.reflux_register = reflux
        self.global_cell_ids = identifiers
        self.owner = owners
        self.group_edges = edges
        self.collective_count = collectives
        self.reflux_residual = residual
        self.epoch_sequence = epoch
        self.frame_token = frame_token
        self.observer_coordinates = observer
        self.coordinate_time = time
        self.scale_factor = scale
        self.system_id = system
        self.topology_id = topology
        self.partition_id = partition
        self.frame_id = frame.frame_id
        self.frame_realization_id = frame.realization_id()
        self.unit_contract_id = frame.units.contract_id
        self.epoch_manifest_id = manifest
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-m1-checkpoint-plan",
                "system": system,
                "topology": topology,
                "partition": partition,
                "capacity": values.shape[0],
                "components": values.shape[1],
                "groups": np.asarray(edges).tolist(),
                "frame": frame.frame_id,
                "unit_contract": frame.units.contract_id,
            }
        )
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "dark-radiation-m1-checkpoint",
                "plan": self.plan_id,
                "epoch_manifest": manifest,
                "epoch_sequence": int(np.asarray(epoch)),
                "global_cell_ids": array_tree_fingerprint(identifiers),
                "state": array_tree_fingerprint(values),
                "reflux_register": array_tree_fingerprint(reflux),
                "reflux_residual": array_tree_fingerprint(residual),
                "frame_realization": self.frame_realization_id,
                "frame_token": array_tree_fingerprint(frame_token),
                "observer_coordinates": array_tree_fingerprint(observer),
                "coordinate_time": array_tree_fingerprint(time),
                "scale_factor": array_tree_fingerprint(scale),
            }
        )


def write_dark_radiation_m1_checkpoint(
    path: str | Path, checkpoint: DarkRadiationM1Checkpoint, /
) -> Path:
    if not isinstance(checkpoint, DarkRadiationM1Checkpoint):
        raise TypeError("checkpoint must be DarkRadiationM1Checkpoint.")
    arrays: dict[str, object] = {}
    specification = pack_array_tree("checkpoint", checkpoint, arrays)
    payload_id = canonical_fingerprint(
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "specification": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    return write_array_archive(
        path,
        manifest={
            "format": _M1_CHECKPOINT_FORMAT,
            "plan_id": checkpoint.plan_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "specification": specification,
            "payload_id": payload_id,
        },
        arrays=arrays,
    )


def read_dark_radiation_m1_checkpoint(
    path: str | Path, template: DarkRadiationM1Checkpoint, /
) -> DarkRadiationM1Checkpoint:
    if not isinstance(template, DarkRadiationM1Checkpoint):
        raise TypeError("template must be DarkRadiationM1Checkpoint.")
    template_arrays: dict[str, object] = {}
    specification = pack_array_tree("checkpoint", template, template_arrays)
    inventory = {
        name: (np.asarray(value).shape, np.asarray(value).dtype)
        for name, value in template_arrays.items()
    }
    manifest, arrays = read_array_archive(path, expected_inventory=inventory)
    if (
        set(manifest)
        != {"format", "plan_id", "checkpoint_id", "specification", "payload_id", "arrays"}
        or manifest["format"] != _M1_CHECKPOINT_FORMAT
        or manifest["plan_id"] != template.plan_id
        or manifest["checkpoint_id"] != template.checkpoint_id
        or manifest["specification"] != specification
    ):
        raise ValueError("Dark-radiation M1 checkpoint identity does not match.")
    payload_id = canonical_fingerprint(
        {
            "checkpoint_id": manifest["checkpoint_id"],
            "specification": manifest["specification"],
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    if payload_id != manifest["payload_id"]:
        raise ValueError("Dark-radiation M1 checkpoint payload is corrupt.")
    restored = unpack_array_tree(manifest["specification"], arrays, template)
    if (
        not isinstance(restored, DarkRadiationM1Checkpoint)
        or restored.checkpoint_id != template.checkpoint_id
    ):
        raise ValueError("Restored dark-radiation M1 checkpoint is invalid.")
    return restored


def packet_transport_profile() -> DarkRadiationTransportProfile:
    return DarkRadiationTransportProfile(
        "relativistic-packets",
        (
            "flrw-geodesic-redshift",
            "bounded-absorption-scattering-polarization",
            "exact-material-four-force",
            "durable-epoch-continuation",
        ),
        (
            "event-capacity-exhaustion",
            "frequency-support-exhaustion",
            "frame-or-unit-mismatch",
            "invalid-polarization-coherency",
        ),
        "pathwise-continuous-stop-gradient-events",
        (
            "four-force-balance",
            "mass-shell-residual",
            "lineage-completeness",
            "checkpoint-restart-identity",
        ),
        checkpoint_product="dark-radiation-packet-restart",
        output_product="dark-radiation-packet-analysis",
    )


def multigroup_m1_profile() -> DarkRadiationTransportProfile:
    return DarkRadiationTransportProfile(
        "multigroup-m1",
        (
            "cosmological-conservative-group-redshift",
            "imex-matter-exchange",
            "amr-reflux",
            "distributed-checkpoint",
        ),
        (
            "m1-crossing-beam-risk",
            "realizability-failure",
            "topology-or-checkpoint-mismatch",
        ),
        "differentiable-away-from-realizability-boundary",
        (
            "physical-c-reduced-c-identity",
            "reflux-conservation-residual",
            "matter-four-force-balance",
        ),
        checkpoint_product="dark-radiation-m1-restart",
        output_product="dark-radiation-m1-analysis",
    )


def boltzmann_hierarchy_profile() -> DarkRadiationTransportProfile:
    return DarkRadiationTransportProfile(
        "linear-boltzmann-hierarchy",
        (
            "free-streaming",
            "metric-source",
            "collision-and-self-interaction",
            "line-of-sight-polarization",
        ),
        (
            "nonlinear-perturbation-regime",
            "multipole-closure-unqualified",
            "quadrature-support-mismatch",
        ),
        "differentiable-fixed-capacity-linear-system",
        (
            "terminal-multipole-bound",
            "tight-coupling-evidence",
            "linearization-receipt",
        ),
        checkpoint_product="dark-radiation-hierarchy-restart",
        output_product="dark-radiation-line-of-sight-analysis",
    )


def vet_research_profile() -> DarkRadiationTransportProfile:
    return DarkRadiationTransportProfile(
        "variable-eddington-tensor-research",
        (
            "fixed-angular-quadrature-formal-solve",
            "shadow-preserving-eddington-tensor",
        ),
        (
            "angular-quadrature-capacity-exhaustion",
            "formal-solve-nonconvergence",
            "production-qualification-not-established",
        ),
        "implicit-fixed-iteration-research-only",
        (
            "iteration-residual-history",
            "tensor-lag-identity",
            "shadow-contrast",
        ),
        checkpoint_product="dark-radiation-vet-research-restart",
        output_product="dark-radiation-vet-research-analysis",
    )


__all__ = [
    "DarkRadiationGravitySource",
    "DarkRadiationHierarchyLinearizationResult",
    "DarkRadiationLedgerSourceAdapter",
    "DarkRadiationM1Checkpoint",
    "DarkRadiationPacketMomentResult",
    "DarkRadiationSourceAdapterEvidence",
    "DarkRadiationSourceAdapterResult",
    "DarkRadiationTransportProfile",
    "average_packets_to_m1",
    "boltzmann_hierarchy_profile",
    "hierarchy_gravity_source",
    "linearize_m1_to_hierarchy",
    "m1_gravity_source",
    "multigroup_m1_profile",
    "packet_gravity_source",
    "packet_transport_profile",
    "read_dark_radiation_m1_checkpoint",
    "vet_research_profile",
    "vet_stress_energy_projection",
    "write_dark_radiation_m1_checkpoint",
]
