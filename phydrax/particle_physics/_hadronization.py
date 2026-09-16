#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Declared native dark-string and dark-cluster hadronization profiles."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

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
from ..solver._dark_sector_epoch_runtime import DarkSectorEpochPlan
from ._species import ParticleSpeciesTable


class DarkHadronizationStatus(IntEnum):
    SUCCESS = 0
    NO_CHARGE_CHANNEL = 1
    BELOW_THRESHOLD = 2
    INVALID_COLOR_SINGLET = 3
    NONFINITE_INPUT = 4
    UNKNOWN_SPECIES = 5


class DarkHadronPairChannel(StrictModule, NonTrainableState):
    pdg_ids: tuple[int, int] = eqx.field(static=True)
    relative_weight: float = eqx.field(static=True)
    spectrum_label: str = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        pdg_ids: tuple[int, int],
        relative_weight: float,
        /,
        *,
        spectrum_label: str,
    ):
        identifiers = tuple(int(value) for value in pdg_ids)
        weight = float(relative_weight)
        label = str(spectrum_label).strip()
        if (
            len(identifiers) != 2
            or not math.isfinite(weight)
            or weight <= 0.0
            or not label
        ):
            raise ValueError(
                "A hadron-pair channel requires two species, positive weight, and spectrum label."
            )
        self.pdg_ids = identifiers
        self.relative_weight = weight
        self.spectrum_label = label
        self.channel_id = canonical_fingerprint(
            {
                "kind": "dark-hadron-pair-channel",
                "pdg_ids": list(identifiers),
                "relative_weight": weight,
                "spectrum_label": label,
            }
        )


class DarkClusterFissionChannel(StrictModule, NonTrainableState):
    daughter_rest_energies: tuple[float, float] = eqx.field(static=True)
    daughter_charges: tuple[float, float] = eqx.field(static=True)
    relative_weight: float = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        daughter_rest_energies: tuple[float, float],
        daughter_charges: tuple[float, float],
        relative_weight: float,
        /,
    ):
        masses = tuple(float(value) for value in daughter_rest_energies)
        charges = tuple(float(value) for value in daughter_charges)
        weight = float(relative_weight)
        if (
            len(masses) != 2
            or len(charges) != 2
            or any(not math.isfinite(value) or value < 0.0 for value in masses)
            or any(not math.isfinite(value) for value in charges)
            or not math.isfinite(weight)
            or weight <= 0.0
        ):
            raise ValueError("Cluster fission channel parameters are invalid.")
        self.daughter_rest_energies = masses
        self.daughter_charges = charges
        self.relative_weight = weight
        self.channel_id = canonical_fingerprint(
            {
                "kind": "dark-cluster-fission-channel",
                "rest_energies": list(masses),
                "charges": list(charges),
                "relative_weight": weight,
            }
        )


def _validate_common(
    runtime_plan,
    species,
    units,
    frame,
    channels,
    labels,
    evidence_ids,
):
    if not isinstance(runtime_plan, DarkSectorEpochPlan):
        raise TypeError("runtime_plan must be DarkSectorEpochPlan.")
    if not isinstance(species, ParticleSpeciesTable):
        raise TypeError("species must be ParticleSpeciesTable.")
    if not isinstance(units, RelativisticUnitContract):
        raise TypeError("units must be RelativisticUnitContract.")
    if species.energy_unit.unit_id != units.energy_unit.unit_id:
        raise ValueError(
            "species and hadronization must share the exact relativistic energy unit."
        )
    if (
        not isinstance(frame, LocalRelativisticFramePlan)
        or frame.units.contract_id != units.contract_id
    ):
        raise ValueError("frame must use the exact relativistic unit contract.")
    if units.convention.metric_signature != "mostly_minus":
        raise ValueError(
            "Native dark hadronization requires the explicit mostly-minus convention."
        )
    channels_ = tuple(channels)
    if not channels_ or any(
        not isinstance(value, DarkHadronPairChannel) for value in channels_
    ):
        raise TypeError("channels must contain DarkHadronPairChannel values.")
    labels_ = tuple(str(value).strip() for value in labels)
    evidence = tuple(str(value).strip() for value in evidence_ids)
    if (
        any(not value for value in labels_)
        or not evidence
        or any(not value for value in evidence)
    ):
        raise ValueError("Profile identities and production evidence must be explicit.")
    if len(set(evidence)) != len(evidence):
        raise ValueError("Production evidence identities must be unique.")
    ids = np.asarray(species.pdg_ids)
    active = np.asarray(species.active)
    if any(
        identifier not in set(ids[active].tolist())
        for channel in channels_
        for identifier in channel.pdg_ids
    ):
        raise ValueError(
            "Every produced dark hadron must exist in the active species table."
        )
    return channels_, labels_, evidence


class DarkStringFragmentationPlan(StrictModule, NonTrainableState):
    """Two-body terminal fragmentation for one explicitly named dark string model."""

    runtime_plan: DarkSectorEpochPlan
    species: ParticleSpeciesTable
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    frame_realization_id: str = eqx.field(static=True)
    channels: tuple[DarkHadronPairChannel, ...]
    model_id: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    tune_id: str = eqx.field(static=True)
    string_tension: float = eqx.field(static=True)
    longitudinal_shape: tuple[float, float] = eqx.field(static=True)
    production_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    supports_generic_qcd: bool = eqx.field(static=True)
    support_scope: str = eqx.field(static=True)
    refusal_modes: tuple[str, ...] = eqx.field(static=True)
    differentiation_mode: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime_plan: DarkSectorEpochPlan,
        species: ParticleSpeciesTable,
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        channels: Sequence[DarkHadronPairChannel],
        /,
        *,
        model_id: str,
        model_revision_id: str,
        tune_id: str,
        string_tension: float,
        longitudinal_shape: tuple[float, float],
        production_evidence_ids: Sequence[str],
    ):
        channels_, labels, evidence = _validate_common(
            runtime_plan,
            species,
            units,
            frame,
            channels,
            (model_id, model_revision_id, tune_id),
            production_evidence_ids,
        )
        tension = float(string_tension)
        shape = tuple(float(value) for value in longitudinal_shape)
        if (
            not math.isfinite(tension)
            or tension <= 0.0
            or len(shape) != 2
            or any(not math.isfinite(value) or value < 0.0 for value in shape)
        ):
            raise ValueError(
                "String tension and declared longitudinal shape are invalid."
            )
        self.runtime_plan = runtime_plan
        self.species = species
        self.units = units
        self.frame = frame
        self.frame_realization_id = frame.realization_id()
        self.channels = channels_
        self.model_id, self.model_revision_id, self.tune_id = labels
        self.string_tension = tension
        self.longitudinal_shape = shape
        self.production_evidence_ids = evidence
        self.supports_generic_qcd = False
        self.support_scope = "declared-dark-string-two-body-terminal-fragmentation"
        self.refusal_modes = (
            "generic-qcd",
            "unconnected-color-graph",
            "non-mostly-minus-event-record",
        )
        self.differentiation_mode = "piecewise-stopped-random-choice"
        self.profile_id = canonical_fingerprint(
            {
                "kind": "native-declared-dark-string-fragmentation",
                "runtime_plan": runtime_plan.plan_id,
                "species": species.table_id,
                "units": units.contract_id,
                "frame": frame.frame_id,
                "frame_realization": self.frame_realization_id,
                "channels": [value.channel_id for value in channels_],
                "model": labels[0],
                "revision": labels[1],
                "tune": labels[2],
                "string_tension": tension,
                "longitudinal_shape": list(shape),
                "production_evidence": list(evidence),
                "generic_qcd": False,
                "support_scope": self.support_scope,
                "refusal_modes": list(self.refusal_modes),
                "differentiation": self.differentiation_mode,
            }
        )


class DarkClusterHadronizationPlan(StrictModule, NonTrainableState):
    """Declared dark-cluster fission and terminal decay, never a generic-QCD claim."""

    runtime_plan: DarkSectorEpochPlan
    species: ParticleSpeciesTable
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    frame_realization_id: str = eqx.field(static=True)
    decay_channels: tuple[DarkHadronPairChannel, ...]
    fission_channels: tuple[DarkClusterFissionChannel, ...]
    model_id: str = eqx.field(static=True)
    model_revision_id: str = eqx.field(static=True)
    tune_id: str = eqx.field(static=True)
    fission_threshold: float = eqx.field(static=True)
    production_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    supports_generic_qcd: bool = eqx.field(static=True)
    support_scope: str = eqx.field(static=True)
    refusal_modes: tuple[str, ...] = eqx.field(static=True)
    differentiation_mode: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime_plan: DarkSectorEpochPlan,
        species: ParticleSpeciesTable,
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        decay_channels: Sequence[DarkHadronPairChannel],
        fission_channels: Sequence[DarkClusterFissionChannel],
        /,
        *,
        model_id: str,
        model_revision_id: str,
        tune_id: str,
        fission_threshold: float,
        production_evidence_ids: Sequence[str],
    ):
        decays, labels, evidence = _validate_common(
            runtime_plan,
            species,
            units,
            frame,
            decay_channels,
            (model_id, model_revision_id, tune_id),
            production_evidence_ids,
        )
        fissions = tuple(fission_channels)
        threshold = float(fission_threshold)
        if any(not isinstance(value, DarkClusterFissionChannel) for value in fissions):
            raise TypeError(
                "fission_channels must contain DarkClusterFissionChannel values."
            )
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("fission_threshold must be finite and positive.")
        self.runtime_plan = runtime_plan
        self.species = species
        self.units = units
        self.frame = frame
        self.frame_realization_id = frame.realization_id()
        self.decay_channels = decays
        self.fission_channels = fissions
        self.model_id, self.model_revision_id, self.tune_id = labels
        self.fission_threshold = threshold
        self.production_evidence_ids = evidence
        self.supports_generic_qcd = False
        self.support_scope = "declared-dark-cluster-two-body-fission-decay"
        self.refusal_modes = (
            "generic-qcd",
            "undeclared-multiplicity",
            "non-mostly-minus-event-record",
        )
        self.differentiation_mode = "piecewise-stopped-random-choice"
        self.profile_id = canonical_fingerprint(
            {
                "kind": "native-declared-dark-cluster-hadronization",
                "runtime_plan": runtime_plan.plan_id,
                "species": species.table_id,
                "units": units.contract_id,
                "frame": frame.frame_id,
                "frame_realization": self.frame_realization_id,
                "decays": [value.channel_id for value in decays],
                "fissions": [value.channel_id for value in fissions],
                "model": labels[0],
                "revision": labels[1],
                "tune": labels[2],
                "fission_threshold": threshold,
                "production_evidence": list(evidence),
                "generic_qcd": False,
                "support_scope": self.support_scope,
                "refusal_modes": list(self.refusal_modes),
                "differentiation": self.differentiation_mode,
            }
        )


class DarkHadronizationEvidence(StrictModule, NonTrainableState):
    input_four_momentum: Array
    output_four_momenta: Array
    output_pdg_ids: Array
    output_charges: Array
    channel_probabilities: Array
    selected_channel: Array
    four_momentum_residual: Array
    charge_residual: Array
    finite: Array
    status: Array
    parent_entity_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(DarkHadronizationStatus.SUCCESS)


class DarkClusterFissionResult(StrictModule, NonTrainableState):
    input_four_momentum: Array
    daughter_four_momenta: Array
    daughter_rest_energies: Array
    daughter_charges: Array
    channel_probabilities: Array
    selected_channel: Array
    four_momentum_residual: Array
    charge_residual: Array
    finite: Array
    status: Array
    parent_entity_id: str = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _species_arrays(table: ParticleSpeciesTable, channels):
    ids = np.asarray(table.pdg_ids)
    masses = np.asarray(table.rest_energies)
    charges = np.asarray(table.charges)
    active = np.asarray(table.active)
    by_id = {
        int(identifier): (float(mass), float(charge))
        for identifier, mass, charge, present in zip(
            ids, masses, charges, active, strict=True
        )
        if present
    }
    return (
        jnp.asarray([channel.pdg_ids for channel in channels], dtype=jnp.int32),
        jnp.asarray(
            [[by_id[value][0] for value in channel.pdg_ids] for channel in channels]
        ),
        jnp.asarray(
            [[by_id[value][1] for value in channel.pdg_ids] for channel in channels]
        ),
        jnp.asarray([channel.relative_weight for channel in channels]),
    )


def _two_body_momenta(units, parent, masses, azimuth_uniform, polar_uniform):
    invariant_squared = units.lorentz_scalar(parent, parent)
    invariant_mass = jnp.sqrt(jnp.maximum(invariant_squared, 0.0))
    m1, m2 = masses[0], masses[1]
    first = invariant_squared - (m1 + m2) ** 2
    second = invariant_squared - (m1 - m2) ** 2
    magnitude = jnp.sqrt(jnp.maximum(first * second, 0.0)) / jnp.maximum(
        2.0 * invariant_mass, 1e-30
    )
    cos_theta = 2.0 * polar_uniform - 1.0
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta * cos_theta, 0.0))
    phi = 2.0 * jnp.pi * azimuth_uniform
    direction = jnp.asarray(
        (sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta)
    )
    p_rest = magnitude * direction
    e1 = jnp.sqrt(m1 * m1 + magnitude * magnitude)
    e2 = jnp.sqrt(m2 * m2 + magnitude * magnitude)
    beta = parent[1:] / jnp.maximum(parent[0], 1e-30)
    beta_squared = jnp.sum(beta * beta)
    gamma = parent[0] / jnp.maximum(invariant_mass, 1e-30)

    def boost(energy, spatial):
        projection = jnp.sum(beta * spatial)
        coefficient = jnp.where(
            beta_squared > 0.0,
            (gamma - 1.0) * projection / beta_squared + gamma * energy,
            0.0,
        )
        return jnp.concatenate(
            ((gamma * (energy + projection))[None], spatial + coefficient * beta)
        )

    return jnp.stack((boost(e1, p_rest), boost(e2, -p_rest))), invariant_mass


def _pair_probabilities(
    units, parent, masses, charges, weights, parent_charge, suppression
):
    invariant_squared = units.lorentz_scalar(parent, parent)
    invariant_mass = jnp.sqrt(jnp.maximum(invariant_squared, 0.0))
    threshold = masses[:, 0] + masses[:, 1]
    charge_match = jnp.isclose(
        jnp.sum(charges, axis=1), parent_charge, rtol=0.0, atol=1e-10
    )
    open_channel = invariant_mass >= threshold
    first = invariant_squared - threshold * threshold
    second = invariant_squared - (masses[:, 0] - masses[:, 1]) ** 2
    phase_space = jnp.sqrt(jnp.maximum(first * second, 0.0)) / jnp.maximum(
        2.0 * invariant_mass, 1e-30
    )
    raw = jnp.where(
        charge_match & open_channel, weights * phase_space * suppression(masses), 0.0
    )
    total = jnp.sum(raw)
    return jnp.where(total > 0.0, raw / total, 0.0), charge_match, open_channel


def _select(probabilities, uniform):
    total = jnp.sum(probabilities)
    return jnp.argmax(jnp.cumsum(probabilities) > uniform * total).astype(jnp.int32)


def _fragment_dark_string_total(
    plan,
    parent,
    parent_charge,
    color_singlet,
    random,
    parent_id,
    finite_input,
    species_valid,
):
    channel_ids, masses, charges, weights = _species_arrays(plan.species, plan.channels)
    probabilities, charge_match, open_channel = _pair_probabilities(
        plan.units,
        parent,
        masses,
        charges,
        weights,
        parent_charge,
        lambda values: jnp.exp(
            -jnp.pi * jnp.sum(values * values, axis=1) / plan.string_tension
        ),
    )
    selected = _select(probabilities, random[0])
    output, _ = _two_body_momenta(
        plan.units, parent, masses[selected], random[1], random[2]
    )
    selected_ids = channel_ids[selected]
    selected_charges = charges[selected]
    finite = finite_input & jnp.all(jnp.isfinite(output))
    status = jnp.where(
        ~finite,
        int(DarkHadronizationStatus.NONFINITE_INPUT),
        jnp.where(
            ~species_valid,
            int(DarkHadronizationStatus.UNKNOWN_SPECIES),
            jnp.where(
                ~color_singlet,
                int(DarkHadronizationStatus.INVALID_COLOR_SINGLET),
                jnp.where(
                    ~jnp.any(charge_match),
                    int(DarkHadronizationStatus.NO_CHARGE_CHANNEL),
                    jnp.where(
                        ~jnp.any(charge_match & open_channel),
                        int(DarkHadronizationStatus.BELOW_THRESHOLD),
                        int(DarkHadronizationStatus.SUCCESS),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    momentum_residual = jnp.sum(output, axis=0) - parent
    charge_residual = jnp.sum(selected_charges) - parent_charge
    evidence_id = canonical_fingerprint(
        {
            "kind": "dark-string-fragmentation-evidence",
            "profile": plan.profile_id,
            "parent": parent_id,
            "production_evidence": list(plan.production_evidence_ids),
        }
    )
    return DarkHadronizationEvidence(
        parent,
        output,
        selected_ids,
        selected_charges,
        probabilities,
        selected,
        momentum_residual,
        charge_residual,
        finite,
        status,
        parent_id,
        plan.profile_id,
        plan.frame_realization_id,
        evidence_id,
    )


def fragment_dark_string(
    plan: DarkStringFragmentationPlan,
    endpoint_four_momenta: ArrayLike,
    endpoint_pdg_ids: tuple[int, int],
    endpoint_color_flow: ArrayLike,
    uniforms: ArrayLike,
    /,
    *,
    parent_entity_id: str,
) -> DarkHadronizationEvidence:
    """Fragment one declared color-singlet dark string into an exact two-body state."""

    if not isinstance(plan, DarkStringFragmentationPlan):
        raise TypeError("plan must be DarkStringFragmentationPlan.")
    endpoints = jnp.asarray(endpoint_four_momenta)
    colors = jnp.asarray(endpoint_color_flow, dtype=jnp.int32)
    random = jnp.asarray(uniforms, dtype=endpoints.dtype)
    if endpoints.shape != (2, 4) or colors.shape != (2, 2) or random.shape != (3,):
        raise ValueError(
            "String endpoints, color flow, and uniforms have incompatible shapes."
        )
    random = eqx.error_if(
        random,
        jnp.any((random < 0.0) | (random >= 1.0) | ~jnp.isfinite(random)),
        "uniforms must be finite and lie in [0, 1).",
    )
    parent_id = str(parent_entity_id).strip()
    if not parent_id:
        raise ValueError("parent_entity_id must be non-empty.")
    ids = np.asarray(plan.species.pdg_ids)
    charges_table = np.asarray(plan.species.charges)
    active = np.asarray(plan.species.active)
    charge_by_id = {
        int(identifier): float(charge)
        for identifier, charge, present in zip(ids, charges_table, active, strict=True)
        if present
    }
    if any(int(value) not in charge_by_id for value in endpoint_pdg_ids):
        raise ValueError(
            "String endpoint species must exist in the declared species table."
        )
    parent_charge = sum(charge_by_id[int(value)] for value in endpoint_pdg_ids)
    parent = jnp.sum(endpoints, axis=0)
    color_singlet = (
        (colors[0, 0] > 0)
        & (colors[0, 1] == 0)
        & (colors[1, 0] == 0)
        & (colors[1, 1] == colors[0, 0])
    )
    return _fragment_dark_string_total(
        plan,
        parent,
        parent_charge,
        color_singlet,
        random,
        parent_id,
        jnp.all(jnp.isfinite(endpoints)),
        jnp.asarray(True),
    )


def fragment_dark_string_chain(
    plan: DarkStringFragmentationPlan,
    parton_four_momenta: ArrayLike,
    parton_pdg_ids: ArrayLike,
    parton_color_flow: ArrayLike,
    parton_active: ArrayLike,
    uniforms: ArrayLike,
    /,
    *,
    parent_entity_id: str,
) -> DarkHadronizationEvidence:
    """Fragment one fixed-capacity connected dark-color chain, including kinks."""

    if not isinstance(plan, DarkStringFragmentationPlan):
        raise TypeError("plan must be DarkStringFragmentationPlan.")
    momenta = jnp.asarray(parton_four_momenta)
    pdg_ids = jnp.asarray(parton_pdg_ids, dtype=jnp.int32)
    colors = jnp.asarray(parton_color_flow, dtype=jnp.int32)
    active = jnp.asarray(parton_active, dtype=bool)
    random = jnp.asarray(uniforms, dtype=momenta.dtype)
    if (
        momenta.ndim != 2
        or momenta.shape[1:] != (4,)
        or pdg_ids.shape != (momenta.shape[0],)
        or colors.shape != (momenta.shape[0], 2)
        or active.shape != (momenta.shape[0],)
        or random.shape != (3,)
    ):
        raise ValueError("Dark string chain arrays have incompatible fixed shapes.")
    random = eqx.error_if(
        random,
        jnp.any((random < 0.0) | (random >= 1.0) | ~jnp.isfinite(random)),
        "uniforms must be finite and lie in [0, 1).",
    )
    parent_id = str(parent_entity_id).strip()
    if not parent_id:
        raise ValueError("parent_entity_id must be non-empty.")
    matches = (pdg_ids[:, None] == plan.species.pdg_ids[None, :]) & plan.species.active[
        None, :
    ]
    found = jnp.any(matches, axis=1)
    species_valid = jnp.all(~active | found)
    indices = jnp.argmax(matches, axis=1)
    parton_charges = plan.species.charges[indices]
    parent_charge = jnp.sum(jnp.where(active, parton_charges, 0.0))
    parent = jnp.sum(jnp.where(active[:, None], momenta, 0.0), axis=0)

    entry_active = jnp.repeat(active, 2)
    tags = colors.reshape((-1,))
    tag_counts = jnp.sum(
        (tags[:, None] == tags[None, :]) & entry_active[None, :] & (tags[:, None] != 0),
        axis=1,
    )
    balanced = jnp.all(~entry_active | (tags == 0) | (tag_counts == 2)) & jnp.any(active)
    adjacency = jnp.any(
        (colors[:, :, None, None] == colors[None, None, :, :])
        & (colors[:, :, None, None] != 0),
        axis=(1, 3),
    )
    root = jnp.argmax(active)
    reached = jnp.arange(active.shape[0]) == root

    def connect(_, current):
        return current | jnp.any(current[:, None] & adjacency, axis=0)

    reached = jax.lax.fori_loop(0, active.shape[0], connect, reached)
    connected = jnp.all(~active | reached)
    finite_input = jnp.all(jnp.where(active[:, None], jnp.isfinite(momenta), True))
    return _fragment_dark_string_total(
        plan,
        parent,
        parent_charge,
        balanced & connected,
        random,
        parent_id,
        finite_input,
        species_valid,
    )


def decay_dark_cluster(
    plan: DarkClusterHadronizationPlan,
    cluster_four_momentum: ArrayLike,
    cluster_charge: ArrayLike,
    uniforms: ArrayLike,
    /,
    *,
    parent_entity_id: str,
) -> DarkHadronizationEvidence:
    """Decay one terminal dark cluster using normalized declared spectrum weights."""

    if not isinstance(plan, DarkClusterHadronizationPlan):
        raise TypeError("plan must be DarkClusterHadronizationPlan.")
    parent = jnp.asarray(cluster_four_momentum)
    charge = jnp.asarray(cluster_charge, dtype=parent.dtype)
    random = jnp.asarray(uniforms, dtype=parent.dtype)
    if parent.shape != (4,) or charge.shape != () or random.shape != (3,):
        raise ValueError(
            "Cluster momentum, charge, and uniforms have incompatible shapes."
        )
    random = eqx.error_if(
        random,
        jnp.any((random < 0.0) | (random >= 1.0) | ~jnp.isfinite(random)),
        "uniforms must be finite and lie in [0, 1).",
    )
    parent_id = str(parent_entity_id).strip()
    if not parent_id:
        raise ValueError("parent_entity_id must be non-empty.")
    channel_ids, masses, charges, weights = _species_arrays(
        plan.species, plan.decay_channels
    )
    probabilities, charge_match, open_channel = _pair_probabilities(
        plan.units,
        parent,
        masses,
        charges,
        weights,
        charge,
        lambda values: jnp.ones((values.shape[0],), dtype=values.dtype),
    )
    selected = _select(probabilities, random[0])
    output, _ = _two_body_momenta(
        plan.units, parent, masses[selected], random[1], random[2]
    )
    selected_charges = charges[selected]
    finite = jnp.all(jnp.isfinite(parent)) & jnp.all(jnp.isfinite(output))
    status = jnp.where(
        ~finite,
        int(DarkHadronizationStatus.NONFINITE_INPUT),
        jnp.where(
            ~jnp.any(charge_match),
            int(DarkHadronizationStatus.NO_CHARGE_CHANNEL),
            jnp.where(
                ~jnp.any(charge_match & open_channel),
                int(DarkHadronizationStatus.BELOW_THRESHOLD),
                int(DarkHadronizationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    momentum_residual = jnp.sum(output, axis=0) - parent
    charge_residual = jnp.sum(selected_charges) - charge
    evidence_id = canonical_fingerprint(
        {
            "kind": "dark-cluster-decay-evidence",
            "profile": plan.profile_id,
            "parent": parent_id,
            "production_evidence": list(plan.production_evidence_ids),
        }
    )
    return DarkHadronizationEvidence(
        parent,
        output,
        channel_ids[selected],
        selected_charges,
        probabilities,
        selected,
        momentum_residual,
        charge_residual,
        finite,
        status,
        parent_id,
        plan.profile_id,
        plan.frame_realization_id,
        evidence_id,
    )


def fission_dark_cluster(
    plan: DarkClusterHadronizationPlan,
    cluster_four_momentum: ArrayLike,
    cluster_charge: ArrayLike,
    uniforms: ArrayLike,
    /,
    *,
    parent_entity_id: str,
) -> DarkClusterFissionResult:
    """Fission one above-threshold cluster; both daughters publish atomically."""

    if not isinstance(plan, DarkClusterHadronizationPlan):
        raise TypeError("plan must be DarkClusterHadronizationPlan.")
    if not plan.fission_channels:
        raise ValueError("The cluster profile declares no fission channels.")
    parent = jnp.asarray(cluster_four_momentum)
    charge = jnp.asarray(cluster_charge, dtype=parent.dtype)
    random = jnp.asarray(uniforms, dtype=parent.dtype)
    if parent.shape != (4,) or charge.shape != () or random.shape != (3,):
        raise ValueError(
            "Cluster momentum, charge, and uniforms have incompatible shapes."
        )
    random = eqx.error_if(
        random,
        jnp.any((random < 0.0) | (random >= 1.0) | ~jnp.isfinite(random)),
        "uniforms must be finite and lie in [0, 1).",
    )
    parent_id = str(parent_entity_id).strip()
    if not parent_id:
        raise ValueError("parent_entity_id must be non-empty.")
    masses = jnp.asarray(
        [value.daughter_rest_energies for value in plan.fission_channels],
        dtype=parent.dtype,
    )
    charges = jnp.asarray(
        [value.daughter_charges for value in plan.fission_channels], dtype=parent.dtype
    )
    weights = jnp.asarray(
        [value.relative_weight for value in plan.fission_channels], dtype=parent.dtype
    )
    probabilities, charge_match, open_channel = _pair_probabilities(
        plan.units,
        parent,
        masses,
        charges,
        weights,
        charge,
        lambda values: jnp.ones((values.shape[0],), dtype=values.dtype),
    )
    selected = _select(probabilities, random[0])
    output, invariant_mass = _two_body_momenta(
        plan.units, parent, masses[selected], random[1], random[2]
    )
    selected_charges = charges[selected]
    finite = jnp.all(jnp.isfinite(parent)) & jnp.all(jnp.isfinite(output))
    above_fission = invariant_mass >= plan.fission_threshold
    status = jnp.where(
        ~finite,
        int(DarkHadronizationStatus.NONFINITE_INPUT),
        jnp.where(
            ~jnp.any(charge_match),
            int(DarkHadronizationStatus.NO_CHARGE_CHANNEL),
            jnp.where(
                ~above_fission | ~jnp.any(charge_match & open_channel),
                int(DarkHadronizationStatus.BELOW_THRESHOLD),
                int(DarkHadronizationStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    momentum_residual = jnp.sum(output, axis=0) - parent
    charge_residual = jnp.sum(selected_charges) - charge
    result_id = canonical_fingerprint(
        {
            "kind": "dark-cluster-fission-result",
            "profile": plan.profile_id,
            "parent": parent_id,
            "production_evidence": list(plan.production_evidence_ids),
        }
    )
    return DarkClusterFissionResult(
        parent,
        output,
        masses[selected],
        selected_charges,
        probabilities,
        selected,
        momentum_residual,
        charge_residual,
        finite,
        status,
        parent_id,
        plan.profile_id,
        plan.frame_realization_id,
        result_id,
    )


__all__ = [
    "DarkClusterFissionChannel",
    "DarkClusterFissionResult",
    "DarkClusterHadronizationPlan",
    "DarkHadronPairChannel",
    "DarkHadronizationEvidence",
    "DarkHadronizationStatus",
    "DarkStringFragmentationPlan",
    "decay_dark_cluster",
    "fission_dark_cluster",
    "fragment_dark_string",
    "fragment_dark_string_chain",
]
