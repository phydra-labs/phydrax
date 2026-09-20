#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Model-declared, fixed-capacity dark-sector parton shower epochs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum, StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from ..solver._dark_sector_epoch_runtime import (
    admit_dark_sector_work,
    DarkSectorEpochPlan,
    DarkSectorEpochResult,
    DarkSectorEpochState,
)
from ._events import ParticleEventBatch
from ._identity import ParticleRole
from ._species import ParticleSpeciesTable


class DarkShowerOrdering(StrEnum):
    TRANSVERSE_MOMENTUM = "transverse-momentum"
    VIRTUALITY = "virtuality"
    ANGLE = "angle"


class DarkColorRule(StrEnum):
    FUNDAMENTAL_EMISSION = "fundamental-emission"
    ANTIFUNDAMENTAL_EMISSION = "antifundamental-emission"
    ADJOINT_SPLIT = "adjoint-split"
    PAIR_PRODUCTION = "pair-production"


class DarkSplittingKernelKind(StrEnum):
    FERMION_VECTOR = "fermion-vector"
    VECTOR_FERMION_PAIR = "vector-fermion-pair"
    VECTOR_SELF = "vector-self"


class DarkShowerProposalStatus(IntEnum):
    UNUSED = 0
    ACCEPTED = 1
    NO_EMITTER = 2
    NO_CHANNEL = 3
    VETOED = 4
    CAPACITY_BACKPRESSURE = 5
    INVALID_EVENT = 6


class DarkSplittingChannel(StrictModule, NonTrainableState):
    """One declared dark-model splitting and its certified veto envelope."""

    parent_pdg_id: int = eqx.field(static=True)
    daughter_pdg_ids: tuple[int, int] = eqx.field(static=True)
    kernel_kind: DarkSplittingKernelKind = eqx.field(static=True)
    color_rule: DarkColorRule = eqx.field(static=True)
    kernel_coefficient: float = eqx.field(static=True)
    envelope_coefficient: float = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        parent_pdg_id: int,
        daughter_pdg_ids: tuple[int, int],
        /,
        *,
        kernel_kind: DarkSplittingKernelKind | str,
        color_rule: DarkColorRule | str,
        kernel_coefficient: float,
        envelope_coefficient: float,
    ):
        daughters = tuple(daughter_pdg_ids)
        kind = DarkSplittingKernelKind(kernel_kind)
        rule = DarkColorRule(color_rule)
        coefficient = float(kernel_coefficient)
        envelope = float(envelope_coefficient)
        if len(daughters) != 2:
            raise ValueError("daughter_pdg_ids must contain exactly two species.")
        minimum_envelope = (
            2.0 * coefficient
            if kind is DarkSplittingKernelKind.FERMION_VECTOR
            else coefficient
            if kind is DarkSplittingKernelKind.VECTOR_FERMION_PAIR
            else 2.125 * coefficient
        )
        if (
            not math.isfinite(coefficient)
            or coefficient <= 0.0
            or not math.isfinite(envelope)
            or envelope < minimum_envelope
        ):
            raise ValueError(
                "The veto envelope must be finite and no smaller than the analytic kernel bound."
            )
        self.parent_pdg_id = int(parent_pdg_id)
        self.daughter_pdg_ids = daughters
        self.kernel_kind = kind
        self.color_rule = rule
        self.kernel_coefficient = coefficient
        self.envelope_coefficient = envelope
        self.channel_id = canonical_fingerprint(
            {
                "kind": "dark-splitting-channel",
                "parent": self.parent_pdg_id,
                "daughters": list(daughters),
                "kernel": kind.value,
                "color_rule": rule.value,
                "coefficient": coefficient,
                "envelope": envelope,
            }
        )


class DarkShowerEpochPlan(StrictModule, NonTrainableState):
    """A bounded compiled shower segment continued by the shared epoch runtime."""

    runtime_plan: DarkSectorEpochPlan
    species: ParticleSpeciesTable
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    frame_realization_id: str = eqx.field(static=True)
    channels: tuple[DarkSplittingChannel, ...]
    ordering: DarkShowerOrdering = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    tune_id: str = eqx.field(static=True)
    alpha_reference: float = eqx.field(static=True)
    reference_scale: float = eqx.field(static=True)
    beta0: float = eqx.field(static=True)
    infrared_cutoff: float = eqx.field(static=True)
    maximum_scale: float = eqx.field(static=True)
    z_bounds: tuple[float, float] = eqx.field(static=True)
    proposal_capacity: int = eqx.field(static=True)
    provider_status: int = eqx.field(static=True)
    support_scope: str = eqx.field(static=True)
    refusal_modes: tuple[str, ...] = eqx.field(static=True)
    differentiation_mode: str = eqx.field(static=True)
    production_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime_plan: DarkSectorEpochPlan,
        species: ParticleSpeciesTable,
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        channels: Sequence[DarkSplittingChannel],
        /,
        *,
        ordering: DarkShowerOrdering | str,
        model_id: str,
        model_revision_id: str,
        tune_id: str,
        alpha_reference: float,
        reference_scale: float,
        beta0: float,
        infrared_cutoff: float,
        maximum_scale: float,
        z_bounds: tuple[float, float],
        proposal_capacity: int,
        production_evidence_ids: Sequence[str],
        provider_status: int = 1,
    ):
        if not isinstance(runtime_plan, DarkSectorEpochPlan):
            raise TypeError("runtime_plan must be DarkSectorEpochPlan.")
        if not isinstance(species, ParticleSpeciesTable):
            raise TypeError("species must be ParticleSpeciesTable.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be RelativisticUnitContract.")
        if species.energy_unit.unit_id != units.energy_unit.unit_id:
            raise ValueError(
                "species and shower must share the exact relativistic energy unit."
            )
        if (
            not isinstance(frame, LocalRelativisticFramePlan)
            or frame.units.contract_id != units.contract_id
        ):
            raise ValueError("frame must use the exact relativistic unit contract.")
        if units.convention.metric_signature != "mostly_minus":
            raise ValueError(
                "The native event shower requires the explicit mostly-minus convention."
            )
        channels_ = tuple(channels)
        if not channels_ or any(
            not isinstance(value, DarkSplittingChannel) for value in channels_
        ):
            raise TypeError("channels must contain DarkSplittingChannel values.")
        channel_ids = tuple(value.channel_id for value in channels_)
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("Dark splitting channels must be unique.")
        if runtime_plan.work_width < 7:
            raise ValueError(
                "runtime_plan work/frontier widths must hold the dark-shower continuation schema."
            )
        if species.capacity > 127:
            raise ValueError(
                "Dark-shower continuation species tables are limited to 127 int8 slots."
            )
        labels = tuple(
            str(value).strip() for value in (model_id, model_revision_id, tune_id)
        )
        if any(not value for value in labels):
            raise ValueError("Model, revision, and tune identities must be non-empty.")
        alpha = float(alpha_reference)
        reference = float(reference_scale)
        beta = float(beta0)
        evidence = tuple(str(value).strip() for value in production_evidence_ids)
        if (
            not evidence
            or any(not value for value in evidence)
            or len(set(evidence)) != len(evidence)
        ):
            raise ValueError(
                "production_evidence_ids must contain distinct non-empty identities."
            )
        cutoff = float(infrared_cutoff)
        maximum = float(maximum_scale)
        z_minimum, z_maximum = map(float, z_bounds)
        if (
            not all(
                math.isfinite(value)
                for value in (
                    alpha,
                    reference,
                    beta,
                    cutoff,
                    maximum,
                    z_minimum,
                    z_maximum,
                )
            )
            or alpha <= 0.0
            or reference <= 0.0
            or beta < 0.0
            or not 0.0 < cutoff < maximum
            or not 0.0 < z_minimum < z_maximum < 1.0
            or int(proposal_capacity) < 1
        ):
            raise ValueError(
                "Dark shower scales, coupling, support, and capacity are invalid."
            )
        landau_denominator = 1.0 + alpha * beta * math.log(cutoff / reference) / (
            2.0 * math.pi
        )
        if landau_denominator <= 0.0:
            raise ValueError(
                "The declared shower support crosses the one-loop Landau pole."
            )
        ids = np.asarray(species.pdg_ids)
        active = np.asarray(species.active)
        masses = np.asarray(species.rest_energies)
        charges = np.asarray(species.charges)
        by_id = {
            int(identifier): (float(mass), float(charge))
            for identifier, mass, charge, present in zip(
                ids, masses, charges, active, strict=True
            )
            if present
        }
        for channel in channels_:
            referenced = (channel.parent_pdg_id, *channel.daughter_pdg_ids)
            if any(identifier not in by_id for identifier in referenced):
                raise ValueError(
                    "Every splitting species must exist in the active species table."
                )
            if any(by_id[identifier][0] != 0.0 for identifier in referenced):
                raise ValueError(
                    "The native collinear dark shower supports massless declared partons only."
                )
            parent_charge = by_id[channel.parent_pdg_id][1]
            daughter_charge = sum(by_id[value][1] for value in channel.daughter_pdg_ids)
            if not math.isclose(
                parent_charge, daughter_charge, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "Each dark splitting channel must conserve declared charge."
                )
        self.runtime_plan = runtime_plan
        self.species = species
        self.units = units
        self.frame = frame
        self.frame_realization_id = frame.realization_id()
        self.channels = channels_
        self.ordering = DarkShowerOrdering(ordering)
        self.model_id, self.model_revision_id, self.tune_id = labels
        self.alpha_reference = alpha
        self.reference_scale = reference
        self.beta0 = beta
        self.infrared_cutoff = cutoff
        self.maximum_scale = maximum
        self.z_bounds = (z_minimum, z_maximum)
        self.proposal_capacity = int(proposal_capacity)
        self.provider_status = int(provider_status)
        self.support_scope = "declared-massless-dark-partons"
        self.refusal_modes = (
            "massive-parton-collinear-splitting",
            "generic-qcd",
            "uncertified-veto-envelope",
            "non-mostly-minus-event-record",
        )
        self.differentiation_mode = "piecewise-stopped-veto"
        self.production_evidence_ids = evidence
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bounded-dark-shower-epoch-plan",
                "runtime_plan": runtime_plan.plan_id,
                "species": species.table_id,
                "units": units.contract_id,
                "frame": frame.frame_id,
                "frame_realization": self.frame_realization_id,
                "channels": list(channel_ids),
                "ordering": self.ordering.value,
                "model": self.model_id,
                "revision": self.model_revision_id,
                "tune": self.tune_id,
                "coupling": [alpha, reference, beta],
                "support": [cutoff, maximum, z_minimum, z_maximum],
                "proposal_capacity": self.proposal_capacity,
                "provider_status": self.provider_status,
                "support_scope": self.support_scope,
                "refusal_modes": list(self.refusal_modes),
                "differentiation": self.differentiation_mode,
                "production_evidence": list(evidence),
            }
        )


class DarkShowerEpochResult(StrictModule, NonTrainableState):
    events: ParticleEventBatch
    proposal_status: Array
    proposal_scales: Array
    proposal_fractions: Array
    proposal_kernel: Array
    proposal_envelope: Array
    accepted: Array
    frontier: Array
    branch_scales: Array
    complete: Array
    backpressured: Array
    finite: Array
    plan_id: str = eqx.field(static=True)
    runtime_plan_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)


def running_dark_coupling(plan: DarkShowerEpochPlan, scale: ArrayLike, /) -> Array:
    """One-loop model coupling on the plan's explicitly certified support."""

    if not isinstance(plan, DarkShowerEpochPlan):
        raise TypeError("plan must be DarkShowerEpochPlan.")
    scale_ = jnp.asarray(scale)
    denominator = 1.0 + plan.alpha_reference * plan.beta0 * jnp.log(
        scale_ / plan.reference_scale
    ) / (2.0 * jnp.pi)
    value = plan.alpha_reference / denominator
    supported = (
        (scale_ >= plan.infrared_cutoff)
        & (scale_ <= plan.maximum_scale)
        & (denominator > 0.0)
    )
    return jnp.where(supported, value, jnp.nan)


def dark_splitting_kernel(channel: DarkSplittingChannel, z: ArrayLike, /) -> Array:
    if not isinstance(channel, DarkSplittingChannel):
        raise TypeError("channel must be DarkSplittingChannel.")
    z_ = jnp.asarray(z)
    if channel.kernel_kind is DarkSplittingKernelKind.FERMION_VECTOR:
        shape = (1.0 + z_ * z_) / (1.0 - z_)
    elif channel.kernel_kind is DarkSplittingKernelKind.VECTOR_FERMION_PAIR:
        shape = z_ * z_ + (1.0 - z_) ** 2
    else:
        shape = 2.0 * (z_ / (1.0 - z_) + (1.0 - z_) / z_ + z_ * (1.0 - z_))
    return channel.kernel_coefficient * shape


def certified_splitting_envelope(channel: DarkSplittingChannel, z: ArrayLike, /) -> Array:
    if not isinstance(channel, DarkSplittingChannel):
        raise TypeError("channel must be DarkSplittingChannel.")
    z_ = jnp.asarray(z)
    return channel.envelope_coefficient / (z_ * (1.0 - z_))


def _integrated_kernel(
    channel: DarkSplittingChannel, lower: float, upper: float
) -> float:
    if channel.kernel_kind is DarkSplittingKernelKind.FERMION_VECTOR:
        primitive = lambda value: (
            -0.5 * value * value - value - 2.0 * math.log(1.0 - value)
        )
    elif channel.kernel_kind is DarkSplittingKernelKind.VECTOR_FERMION_PAIR:
        primitive = lambda value: (2.0 / 3.0) * value**3 - value * value + value
    else:
        primitive = lambda value: (
            2.0
            * (
                -math.log(1.0 - value)
                + math.log(value)
                - 2.0 * value
                + 0.5 * value * value
                - value**3 / 3.0
            )
        )
    return channel.kernel_coefficient * (primitive(upper) - primitive(lower))


def sudakov_no_emission_probability(
    plan: DarkShowerEpochPlan,
    channel: DarkSplittingChannel,
    upper_scale: ArrayLike,
    lower_scale: ArrayLike,
    /,
) -> Array:
    """Analytic one-loop no-emission factor over the declared z support."""

    if channel.channel_id not in tuple(value.channel_id for value in plan.channels):
        raise ValueError("channel is not owned by this shower plan.")
    upper = jnp.asarray(upper_scale)
    lower = jnp.asarray(lower_scale, dtype=upper.dtype)
    z_integral = _integrated_kernel(channel, *plan.z_bounds)
    logarithm = jnp.log(upper / lower)
    if plan.beta0 == 0.0:
        scale_integral = plan.alpha_reference * logarithm / (2.0 * jnp.pi)
    else:
        coefficient = plan.alpha_reference * plan.beta0 / (2.0 * jnp.pi)
        upper_denominator = 1.0 + coefficient * jnp.log(upper / plan.reference_scale)
        lower_denominator = 1.0 + coefficient * jnp.log(lower / plan.reference_scale)
        scale_integral = jnp.log(upper_denominator / lower_denominator) / plan.beta0
    supported = (
        (lower >= plan.infrared_cutoff) & (upper <= plan.maximum_scale) & (upper >= lower)
    )
    return jnp.where(supported, jnp.exp(-z_integral * scale_integral), jnp.nan)


def _channel_tables(plan: DarkShowerEpochPlan):
    kind = tuple(DarkSplittingKernelKind).index
    rule = tuple(DarkColorRule).index
    return (
        jnp.asarray([value.parent_pdg_id for value in plan.channels], dtype=jnp.int32),
        jnp.asarray([value.daughter_pdg_ids for value in plan.channels], dtype=jnp.int32),
        jnp.asarray(
            [kind(value.kernel_kind) for value in plan.channels], dtype=jnp.int32
        ),
        jnp.asarray([rule(value.color_rule) for value in plan.channels], dtype=jnp.int32),
        jnp.asarray([value.kernel_coefficient for value in plan.channels]),
        jnp.asarray([value.envelope_coefficient for value in plan.channels]),
    )


def _kernel_from_table(kind, coefficient, z):
    fermion = coefficient * (1.0 + z * z) / (1.0 - z)
    pair = coefficient * (z * z + (1.0 - z) ** 2)
    self_split = 2.0 * coefficient * (z / (1.0 - z) + (1.0 - z) / z + z * (1.0 - z))
    return jnp.where(kind == 0, fermion, jnp.where(kind == 1, pair, self_split))


def evolve_dark_shower_epoch(
    plan: DarkShowerEpochPlan,
    events: ParticleEventBatch,
    proposal_uniforms: ArrayLike,
    /,
) -> DarkShowerEpochResult:
    """Apply one finite shower epoch without partial branch or vertex publication.

    ``proposal_uniforms`` has shape ``(event_capacity, proposal_capacity, 4)``.
    Its lanes select channel, ordering scale, splitting fraction, and veto test.
    A capacity failure publishes neither daughter and marks durable continuation.
    """

    if not isinstance(plan, DarkShowerEpochPlan):
        raise TypeError("plan must be DarkShowerEpochPlan.")
    if not isinstance(events, ParticleEventBatch):
        raise TypeError("events must be ParticleEventBatch.")
    uniforms = jnp.asarray(proposal_uniforms, dtype=events.momenta.dtype)
    event_capacity, particle_capacity = events.particle_active.shape
    events.vertex_active.shape[1]
    if uniforms.shape != (event_capacity, plan.proposal_capacity, 4):
        raise ValueError("proposal_uniforms has incompatible fixed capacity.")
    uniforms = eqx.error_if(
        uniforms,
        jnp.any((uniforms < 0.0) | (uniforms >= 1.0) | ~jnp.isfinite(uniforms)),
        "proposal_uniforms must be finite and lie in [0, 1).",
    )
    if events.momentum_unit_symbol != plan.units.energy_unit.symbol:
        raise ValueError("Event momentum units do not match the shower unit contract.")

    parents, daughters, kinds, rules, coefficients, envelopes = _channel_tables(plan)
    pids = events.pdg_ids
    roles = events.roles
    provider_status = events.provider_status
    momenta = events.momenta
    rest = events.rest_energies
    occupied = events.particle_active
    mothers = events.mother_indices
    production_vertex = events.production_vertex_indices
    end_vertex = events.end_vertex_indices
    color = events.color_flow
    vertices = events.production_vertices
    vertex_occupied = events.vertex_active
    frontier = occupied & (roles == int(ParticleRole.OUTGOING))
    branch_scales = jnp.where(frontier, plan.maximum_scale, 0.0).astype(momenta.dtype)
    proposal_status = jnp.zeros((event_capacity, plan.proposal_capacity), dtype=jnp.int32)
    proposal_scales = jnp.zeros_like(proposal_status, dtype=momenta.dtype)
    proposal_fractions = jnp.zeros_like(proposal_status, dtype=momenta.dtype)
    proposal_kernel = jnp.zeros_like(proposal_status, dtype=momenta.dtype)
    proposal_envelope = jnp.zeros_like(proposal_status, dtype=momenta.dtype)
    accepted = jnp.zeros_like(proposal_status, dtype=jnp.bool_)
    backpressured = jnp.zeros((event_capacity,), dtype=jnp.bool_)
    event_indices = jnp.arange(event_capacity, dtype=jnp.int32)

    carry = (
        pids,
        roles,
        provider_status,
        momenta,
        rest,
        occupied,
        mothers,
        production_vertex,
        end_vertex,
        color,
        vertices,
        vertex_occupied,
        frontier,
        branch_scales,
        proposal_status,
        proposal_scales,
        proposal_fractions,
        proposal_kernel,
        proposal_envelope,
        accepted,
        backpressured,
    )

    def proposal_step(index, state):
        (
            pids_,
            roles_,
            provider_status_,
            momenta_,
            rest_,
            occupied_,
            mothers_,
            production_vertex_,
            end_vertex_,
            color_,
            vertices_,
            vertex_occupied_,
            frontier_,
            branch_scales_,
            statuses_,
            scales_,
            fractions_,
            kernels_,
            bounds_,
            accepted_,
            backpressured_,
        ) = state
        proposal = uniforms[:, index]
        eligible = frontier_ & (branch_scales_ > plan.infrared_cutoff)
        eligible_scale = jnp.where(eligible, branch_scales_, -jnp.inf)
        emitter = jnp.argmax(eligible_scale, axis=1).astype(jnp.int32)
        has_emitter = jnp.any(eligible, axis=1) & events.event_active & events.valid
        parent_pid = pids_[event_indices, emitter]
        compatible = parent_pid[:, None] == parents[None, :]
        channel_weight = jnp.where(compatible, envelopes[None, :], 0.0)
        total_weight = jnp.sum(channel_weight, axis=1)
        target = proposal[:, 0] * total_weight
        cumulative = jnp.cumsum(channel_weight, axis=1)
        channel_index = jnp.argmax(cumulative > target[:, None], axis=1).astype(jnp.int32)
        has_channel = total_weight > 0.0
        upper = branch_scales_[event_indices, emitter]
        candidate_scale = (
            plan.infrared_cutoff * (upper / plan.infrared_cutoff) ** proposal[:, 1]
        )
        z = plan.z_bounds[0] + (plan.z_bounds[1] - plan.z_bounds[0]) * proposal[:, 2]
        selected_kind = kinds[channel_index]
        selected_coefficient = coefficients[channel_index]
        selected_envelope_coefficient = envelopes[channel_index]
        kernel = _kernel_from_table(selected_kind, selected_coefficient, z)
        bound = selected_envelope_coefficient / (z * (1.0 - z))
        veto_accept = proposal[:, 3] * bound <= kernel

        empty_rank = jnp.cumsum((~occupied_).astype(jnp.int32), axis=1) - 1
        first_mask = (~occupied_) & (empty_rank == 0)
        second_mask = (~occupied_) & (empty_rank == 1)
        first_slot = jnp.argmax(first_mask, axis=1).astype(jnp.int32)
        second_slot = jnp.argmax(second_mask, axis=1).astype(jnp.int32)
        has_particles = jnp.sum(~occupied_, axis=1) >= 2
        vertex_rank = jnp.cumsum((~vertex_occupied_).astype(jnp.int32), axis=1) - 1
        vertex_mask = (~vertex_occupied_) & (vertex_rank == 0)
        vertex_slot = jnp.argmax(vertex_mask, axis=1).astype(jnp.int32)
        has_vertex = jnp.any(~vertex_occupied_, axis=1)
        capacity_ok = has_particles & has_vertex
        physical_accept = has_emitter & has_channel & veto_accept
        publish = physical_accept & capacity_ok & ~backpressured_
        pressure = physical_accept & ~capacity_ok
        backpressured_ = backpressured_ | pressure

        selected_daughters = daughters[channel_index]
        parent_momentum = momenta_[event_indices, emitter]
        first_momentum = z[:, None] * parent_momentum
        second_momentum = (1.0 - z)[:, None] * parent_momentum
        new_tag = jnp.max(jnp.abs(color_), axis=(1, 2)).astype(jnp.int32) + index + 1
        parent_color = color_[event_indices, emitter]
        selected_rule = rules[channel_index]
        first_color = jnp.stack((new_tag, jnp.zeros_like(new_tag)), axis=1)
        second_color = jnp.stack((parent_color[:, 0], new_tag), axis=1)
        anti = selected_rule == 1
        adjoint = selected_rule == 2
        pair = selected_rule == 3
        first_color = jnp.where(
            anti[:, None],
            jnp.stack((jnp.zeros_like(new_tag), new_tag), axis=1),
            first_color,
        )
        second_color = jnp.where(
            anti[:, None], jnp.stack((new_tag, parent_color[:, 1]), axis=1), second_color
        )
        first_color = jnp.where(
            adjoint[:, None],
            jnp.stack((parent_color[:, 0], new_tag), axis=1),
            first_color,
        )
        second_color = jnp.where(
            adjoint[:, None],
            jnp.stack((new_tag, parent_color[:, 1]), axis=1),
            second_color,
        )
        first_color = jnp.where(
            pair[:, None],
            jnp.stack((parent_color[:, 0], jnp.zeros_like(new_tag)), axis=1),
            first_color,
        )
        second_color = jnp.where(
            pair[:, None],
            jnp.stack((jnp.zeros_like(new_tag), parent_color[:, 1]), axis=1),
            second_color,
        )

        def scatter_if(array, slot, values):
            old = array[event_indices, slot]
            return array.at[event_indices, slot].set(
                jnp.where(publish.reshape((-1,) + (1,) * (values.ndim - 1)), values, old)
            )

        pids_ = scatter_if(pids_, first_slot, selected_daughters[:, 0])
        pids_ = scatter_if(pids_, second_slot, selected_daughters[:, 1])
        roles_ = scatter_if(
            roles_,
            first_slot,
            jnp.full((event_capacity,), int(ParticleRole.OUTGOING), dtype=jnp.int32),
        )
        intermediate_role = jnp.asarray(
            int(ParticleRole.INTERMEDIATE), dtype=roles_.dtype
        )
        roles_ = roles_.at[event_indices, emitter].set(
            jnp.where(publish, intermediate_role, roles_[event_indices, emitter])
        )
        provider_status_ = scatter_if(
            provider_status_,
            first_slot,
            jnp.full((event_capacity,), plan.provider_status, dtype=jnp.int32),
        )
        provider_status_ = scatter_if(
            provider_status_,
            second_slot,
            jnp.full((event_capacity,), plan.provider_status, dtype=jnp.int32),
        )
        momenta_ = scatter_if(momenta_, first_slot, first_momentum)
        momenta_ = scatter_if(momenta_, second_slot, second_momentum)
        rest_ = scatter_if(
            rest_, first_slot, jnp.zeros((event_capacity,), dtype=rest_.dtype)
        )
        rest_ = scatter_if(
            rest_, second_slot, jnp.zeros((event_capacity,), dtype=rest_.dtype)
        )
        occupied_ = scatter_if(
            occupied_, first_slot, jnp.ones((event_capacity,), dtype=jnp.bool_)
        )
        occupied_ = scatter_if(
            occupied_, second_slot, jnp.ones((event_capacity,), dtype=jnp.bool_)
        )
        child_mothers = jnp.stack((emitter, jnp.full_like(emitter, -1)), axis=1)
        mothers_ = scatter_if(mothers_, first_slot, child_mothers)
        mothers_ = scatter_if(mothers_, second_slot, child_mothers)
        production_vertex_ = scatter_if(production_vertex_, first_slot, vertex_slot)
        production_vertex_ = scatter_if(production_vertex_, second_slot, vertex_slot)
        end_vertex_ = end_vertex_.at[event_indices, emitter].set(
            jnp.where(publish, vertex_slot, end_vertex_[event_indices, emitter])
        )
        color_ = scatter_if(color_, first_slot, first_color)
        color_ = scatter_if(color_, second_slot, second_color)
        vertex_position = jnp.zeros((event_capacity, 4), dtype=vertices_.dtype)
        old_vertex = vertices_[event_indices, vertex_slot]
        vertices_ = vertices_.at[event_indices, vertex_slot].set(
            jnp.where(publish[:, None], vertex_position, old_vertex)
        )
        old_vertex_active = vertex_occupied_[event_indices, vertex_slot]
        vertex_occupied_ = vertex_occupied_.at[event_indices, vertex_slot].set(
            jnp.where(publish, True, old_vertex_active)
        )
        frontier_ = frontier_.at[event_indices, emitter].set(
            jnp.where(publish, False, frontier_[event_indices, emitter])
        )
        frontier_ = scatter_if(
            frontier_, first_slot, jnp.ones((event_capacity,), dtype=jnp.bool_)
        )
        frontier_ = scatter_if(
            frontier_, second_slot, jnp.ones((event_capacity,), dtype=jnp.bool_)
        )
        branch_scales_ = scatter_if(branch_scales_, first_slot, candidate_scale)
        branch_scales_ = scatter_if(branch_scales_, second_slot, candidate_scale)
        status = jnp.where(
            ~events.event_active | ~events.valid,
            int(DarkShowerProposalStatus.INVALID_EVENT),
            jnp.where(
                ~has_emitter,
                int(DarkShowerProposalStatus.NO_EMITTER),
                jnp.where(
                    ~has_channel,
                    int(DarkShowerProposalStatus.NO_CHANNEL),
                    jnp.where(
                        pressure,
                        int(DarkShowerProposalStatus.CAPACITY_BACKPRESSURE),
                        jnp.where(
                            publish,
                            int(DarkShowerProposalStatus.ACCEPTED),
                            int(DarkShowerProposalStatus.VETOED),
                        ),
                    ),
                ),
            ),
        )
        statuses_ = statuses_.at[:, index].set(status)
        scales_ = scales_.at[:, index].set(candidate_scale)
        fractions_ = fractions_.at[:, index].set(z)
        kernels_ = kernels_.at[:, index].set(kernel)
        bounds_ = bounds_.at[:, index].set(bound)
        accepted_ = accepted_.at[:, index].set(publish)
        return (
            pids_,
            roles_,
            provider_status_,
            momenta_,
            rest_,
            occupied_,
            mothers_,
            production_vertex_,
            end_vertex_,
            color_,
            vertices_,
            vertex_occupied_,
            frontier_,
            branch_scales_,
            statuses_,
            scales_,
            fractions_,
            kernels_,
            bounds_,
            accepted_,
            backpressured_,
        )

    carry = jax.lax.fori_loop(0, plan.proposal_capacity, proposal_step, carry)
    (
        pids,
        roles,
        provider_status,
        momenta,
        rest,
        occupied,
        mothers,
        production_vertex,
        end_vertex,
        color,
        vertices,
        vertex_occupied,
        frontier,
        branch_scales,
        proposal_status,
        proposal_scales,
        proposal_fractions,
        proposal_kernel,
        proposal_envelope,
        accepted,
        backpressured,
    ) = carry
    updated = eqx.tree_at(
        lambda value: (
            value.pdg_ids,
            value.roles,
            value.provider_status,
            value.momenta,
            value.rest_energies,
            value.particle_active,
            value.mother_indices,
            value.production_vertex_indices,
            value.end_vertex_indices,
            value.color_flow,
            value.production_vertices,
            value.vertex_active,
        ),
        events,
        (
            pids,
            roles,
            provider_status,
            momenta,
            rest,
            occupied,
            mothers,
            production_vertex,
            end_vertex,
            color,
            vertices,
            vertex_occupied,
        ),
    )
    eligible_remaining = frontier & (branch_scales > plan.infrared_cutoff)
    complete = ~jnp.any(eligible_remaining, axis=1) & ~backpressured
    finite = jnp.all(
        jnp.where(occupied[..., None], jnp.isfinite(momenta), True), axis=(1, 2)
    ) & jnp.all(proposal_kernel <= proposal_envelope)
    return DarkShowerEpochResult(
        updated,
        proposal_status,
        proposal_scales,
        proposal_fractions,
        proposal_kernel,
        proposal_envelope,
        accepted,
        frontier,
        branch_scales,
        complete,
        backpressured,
        finite,
        plan.plan_id,
        plan.runtime_plan.plan_id,
        plan.frame_realization_id,
    )


def stage_dark_shower_continuation(
    plan: DarkShowerEpochPlan,
    result: DarkShowerEpochResult,
    state: DarkSectorEpochState,
    /,
) -> DarkSectorEpochResult:
    """Atomically admit every unfinished branch to the shared durable runtime."""

    if not isinstance(plan, DarkShowerEpochPlan):
        raise TypeError("plan must be DarkShowerEpochPlan.")
    if not isinstance(result, DarkShowerEpochResult):
        raise TypeError("result must be DarkShowerEpochResult.")
    if not isinstance(state, DarkSectorEpochState):
        raise TypeError("state must be DarkSectorEpochState.")
    if result.plan_id != plan.plan_id or state.plan.plan_id != plan.runtime_plan.plan_id:
        raise ValueError("Shower result, shower plan, and runtime state do not match.")
    continuation = np.asarray(
        result.frontier
        & (result.branch_scales > plan.infrared_cutoff)
        & result.events.event_active[:, None]
    )
    lanes = np.argwhere(continuation)
    values = np.zeros(
        (lanes.shape[0], plan.runtime_plan.work_width),
        dtype=plan.runtime_plan.value_dtype,
    )
    statuses = np.zeros((lanes.shape[0],), dtype=np.int8)
    species_ids = np.asarray(plan.species.pdg_ids)
    species_active = np.asarray(plan.species.active)
    slot_by_id = {
        int(identifier): int(slot)
        for slot, (identifier, active) in enumerate(
            zip(species_ids, species_active, strict=True)
        )
        if active
    }
    work_ids: list[str] = []
    for work_index, (event_index, particle_index) in enumerate(lanes):
        event_index_ = int(event_index)
        particle_index_ = int(particle_index)
        pdg_id = int(result.events.pdg_ids[event_index_, particle_index_])
        if pdg_id not in slot_by_id:
            raise ValueError("An unfinished shower branch has an undeclared species.")
        values[work_index, :4] = np.asarray(
            result.events.momenta[event_index_, particle_index_]
        )
        values[work_index, 4] = float(result.branch_scales[event_index_, particle_index_])
        values[work_index, 5] = event_index_
        values[work_index, 6] = particle_index_
        statuses[work_index] = slot_by_id[pdg_id]
        work_ids.append(
            canonical_fingerprint(
                {
                    "kind": "dark-shower-continuation-work",
                    "shower_plan": plan.plan_id,
                    "event": [
                        int(result.events.event_ids[event_index_]),
                        int(result.events.subevent_ids[event_index_]),
                    ],
                    "particle_slot": particle_index_,
                    "frame_realization": plan.frame_realization_id,
                }
            )
        )
    admission = admit_dark_sector_work(
        state,
        tuple(work_ids),
        values,
        work_status=statuses,
    )
    return DarkSectorEpochResult(
        admission.state,
        complete=True,
        backpressured=admission.backpressured,
        rolled_back=admission.refused,
        evidence_ids=(plan.plan_id, *plan.production_evidence_ids),
    )


__all__ = [
    "DarkColorRule",
    "DarkShowerEpochPlan",
    "DarkShowerEpochResult",
    "DarkShowerOrdering",
    "DarkShowerProposalStatus",
    "DarkSplittingChannel",
    "DarkSplittingKernelKind",
    "certified_splitting_envelope",
    "dark_splitting_kernel",
    "evolve_dark_shower_epoch",
    "stage_dark_shower_continuation",
    "running_dark_coupling",
    "sudakov_no_emission_probability",
]
