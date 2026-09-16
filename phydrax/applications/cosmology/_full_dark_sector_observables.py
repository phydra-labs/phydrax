#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed analysis products for one committed full dark-sector stage.

These classes contain derived observables, never alternate physics states. Metric
and stress data retain the existing ADM owners, while instrument response is
always delegated to :mod:`phydrax.observation`.
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ...observation import CoordinateLayout, LinearObservationPlan, TheoryVector
from ._full_dark_sector_runtime import FullDarkSectorStageLedger


def _identifier(value: str, name: str, /) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty identifier.")
    return normalized


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    result = tuple(_identifier(value, name) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} values must be nonempty and distinct.")
    return result


def _real_array(value: ArrayLike, name: str, /, *, dtype=None) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    if not eqx.is_inexact_array(result):
        result = result.astype(float)
    return result


def _scalar_flag(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value, dtype=bool)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return result


def _scalar(value: ArrayLike, name: str, /, *, dtype=None) -> Array:
    result = _real_array(value, name, dtype=dtype)
    if result.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return result


def _theory_vector(
    values: Sequence[Array], prefix: str, product_id: str, /
) -> TheoryVector:
    flattened = tuple(jnp.asarray(value).reshape((-1,)) for value in values)
    if not flattened:
        raise ValueError("A full dark-sector theory vector cannot be empty.")
    vector = jnp.concatenate(flattened)
    labels = tuple(f"{prefix}:{index}" for index in range(vector.size))
    return TheoryVector(vector, CoordinateLayout(labels), product_id)


class MetricStressObservables(StrictModule):
    """Stage-exact metric and total stress-energy observables."""

    geometry: ADMGridGeometry
    total_stress_energy: StressEnergyProjection
    constraint_defect: Array
    gauge_defect: Array
    frame_token: Array
    finite: Array
    successful: Array
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: ADMGridGeometry,
        total_stress_energy: StressEnergyProjection,
        constraint_defect: ArrayLike,
        gauge_defect: ArrayLike,
        /,
        *,
        frame_id: str,
        frame_realization_id: str,
        source_ids: Sequence[str],
    ):
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if not isinstance(total_stress_energy, StressEnergyProjection):
            raise TypeError("total_stress_energy must be StressEnergyProjection.")
        constraint = _real_array(constraint_defect, "constraint_defect")
        gauge = _real_array(gauge_defect, "gauge_defect", dtype=constraint.dtype)
        if constraint.size == 0 or gauge.size == 0:
            raise ValueError("Metric constraint and gauge evidence must be nonempty.")
        compatible = total_stress_energy.compatible_with(geometry)
        finite = (
            geometry.all_active_valid
            & total_stress_energy.all_active_valid
            & jnp.all(jnp.isfinite(constraint))
            & jnp.all(jnp.isfinite(gauge))
        )
        sources = _identifiers(source_ids, "metric/stress source ID")
        self.geometry = geometry
        self.total_stress_energy = total_stress_energy
        self.constraint_defect = constraint
        self.gauge_defect = gauge
        self.frame_token = jax.lax.stop_gradient(geometry.snapshot_token)
        self.finite = finite
        self.successful = finite & compatible
        self.frame_id = _identifier(frame_id, "frame_id")
        self.frame_realization_id = _identifier(
            frame_realization_id, "frame_realization_id"
        )
        self.source_ids = sources
        self.product_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-metric-stress-observables",
                "frame": self.frame_id,
                "frame_realization": self.frame_realization_id,
                "geometry": geometry.geometry_lineage_id,
                "topology": geometry.topology_id,
                "projection": total_stress_energy.projection_id,
                "sources": list(sources),
            }
        )

    def as_theory_vector(self, /) -> TheoryVector:
        projection = self.total_stress_energy
        return _theory_vector(
            (
                self.geometry.alpha,
                self.geometry.beta_contravariant,
                self.geometry.spatial_metric,
                self.geometry.extrinsic_curvature,
                projection.energy_density,
                projection.momentum_covector,
                projection.stress_covariant,
                self.constraint_defect,
                self.gauge_defect,
            ),
            "metric-stress",
            self.product_id,
        )


class EventShowerHadronizationObservables(StrictModule):
    """Fixed-bin event, shower, hadronization, and decay products."""

    event_counts: Array
    event_weights: Array
    shower_spectrum: Array
    shower_bin_edges: Array
    hadron_yields: Array
    hadron_species_ids: Array
    decay_yields: Array
    decay_species_ids: Array
    unitarity_defect: Array
    active_bins: Array
    finite: Array
    successful: Array
    event_manifest_id: str = eqx.field(static=True)
    matrix_element_revision_id: str = eqx.field(static=True)
    provider_chain_id: str = eqx.field(static=True)
    shower_profile_id: str = eqx.field(static=True)
    hadronization_profile_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        event_counts: ArrayLike,
        event_weights: ArrayLike,
        shower_spectrum: ArrayLike,
        shower_bin_edges: ArrayLike,
        hadron_yields: ArrayLike,
        hadron_species_ids: ArrayLike,
        decay_yields: ArrayLike,
        decay_species_ids: ArrayLike,
        unitarity_defect: ArrayLike,
        active_bins: ArrayLike,
        /,
        *,
        event_manifest_id: str,
        matrix_element_revision_id: str,
        provider_chain_id: str,
        shower_profile_id: str,
        hadronization_profile_id: str,
        source_ids: Sequence[str],
        successful: ArrayLike,
    ):
        counts = _real_array(event_counts, "event_counts").reshape((-1,))
        weights = _real_array(event_weights, "event_weights", dtype=counts.dtype).reshape(
            (-1,)
        )
        shower = _real_array(
            shower_spectrum, "shower_spectrum", dtype=counts.dtype
        ).reshape((-1,))
        edges = _real_array(
            shower_bin_edges, "shower_bin_edges", dtype=counts.dtype
        ).reshape((-1,))
        hadrons = _real_array(hadron_yields, "hadron_yields", dtype=counts.dtype).reshape(
            (-1,)
        )
        decays = _real_array(decay_yields, "decay_yields", dtype=counts.dtype).reshape(
            (-1,)
        )
        hadron_ids = jax.lax.stop_gradient(
            jnp.asarray(hadron_species_ids, dtype=jnp.int32).reshape((-1,))
        )
        decay_ids = jax.lax.stop_gradient(
            jnp.asarray(decay_species_ids, dtype=jnp.int32).reshape((-1,))
        )
        active = jax.lax.stop_gradient(
            jnp.asarray(active_bins, dtype=bool).reshape((-1,))
        )
        unitarity = _scalar(unitarity_defect, "unitarity_defect", dtype=counts.dtype)
        if (
            counts.size == 0
            or weights.shape != counts.shape
            or shower.size == 0
            or edges.shape != (shower.size + 1,)
            or active.shape != shower.shape
            or hadrons.shape != hadron_ids.shape
            or decays.shape != decay_ids.shape
        ):
            raise ValueError("Event/shower/hadronization observable shapes disagree.")
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in (
                        counts,
                        weights,
                        shower,
                        edges,
                        hadrons,
                        decays,
                        unitarity,
                    )
                )
            )
        )
        identities = tuple(
            _identifier(value, name)
            for value, name in (
                (event_manifest_id, "event_manifest_id"),
                (matrix_element_revision_id, "matrix_element_revision_id"),
                (provider_chain_id, "provider_chain_id"),
                (shower_profile_id, "shower_profile_id"),
                (hadronization_profile_id, "hadronization_profile_id"),
            )
        )
        sources = _identifiers(source_ids, "event source ID")
        self.event_counts = counts
        self.event_weights = weights
        self.shower_spectrum = shower
        self.shower_bin_edges = edges
        self.hadron_yields = hadrons
        self.hadron_species_ids = hadron_ids
        self.decay_yields = decays
        self.decay_species_ids = decay_ids
        self.unitarity_defect = unitarity
        self.active_bins = active
        self.finite = finite
        self.successful = finite & _scalar_flag(successful, "successful")
        (
            self.event_manifest_id,
            self.matrix_element_revision_id,
            self.provider_chain_id,
            self.shower_profile_id,
            self.hadronization_profile_id,
        ) = identities
        self.source_ids = sources
        self.product_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-event-shower-hadronization-observables",
                "identities": list(identities),
                "sources": list(sources),
                "shower_bins": int(shower.size),
                "hadron_species": array_tree_fingerprint(hadron_ids),
                "decay_species": array_tree_fingerprint(decay_ids),
            }
        )

    def as_theory_vector(self, /) -> TheoryVector:
        return _theory_vector(
            (
                self.event_counts,
                self.event_weights,
                jnp.where(self.active_bins, self.shower_spectrum, 0.0),
                self.hadron_yields,
                self.decay_yields,
                self.unitarity_defect,
            ),
            "event-shower-hadronization",
            self.product_id,
        )


class QuantumCoherenceObservables(StrictModule):
    """Quantum occupation, coherence, spectral, and entropy diagnostics."""

    occupation: Array
    coherence_real: Array
    coherence_imaginary: Array
    spectral_coordinates: Array
    spectral_density: Array
    entropy_density: Array
    particle_number: Array
    charge: Array
    trace_defect: Array
    kms_defect: Array
    frame_token: Array
    finite: Array
    successful: Array
    quantum_profile_id: str = eqx.field(static=True)
    coherent_profile_id: str = eqx.field(static=True)
    off_shell_profile_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        occupation: ArrayLike,
        coherence_real: ArrayLike,
        coherence_imaginary: ArrayLike,
        spectral_coordinates: ArrayLike,
        spectral_density: ArrayLike,
        entropy_density: ArrayLike,
        particle_number: ArrayLike,
        charge: ArrayLike,
        trace_defect: ArrayLike,
        kms_defect: ArrayLike,
        /,
        *,
        quantum_profile_id: str,
        coherent_profile_id: str,
        off_shell_profile_id: str,
        frame_id: str,
        frame_token: ArrayLike,
        frame_realization_id: str,
        unit_contract_id: str,
        source_ids: Sequence[str],
        successful: ArrayLike,
    ):
        occupation_ = _real_array(occupation, "occupation")
        coherence_r = _real_array(
            coherence_real, "coherence_real", dtype=occupation_.dtype
        )
        coherence_i = _real_array(
            coherence_imaginary, "coherence_imaginary", dtype=occupation_.dtype
        )
        coordinates = _real_array(
            spectral_coordinates, "spectral_coordinates", dtype=occupation_.dtype
        ).reshape((-1,))
        spectral = _real_array(
            spectral_density, "spectral_density", dtype=occupation_.dtype
        )
        entropy = _real_array(entropy_density, "entropy_density", dtype=occupation_.dtype)
        number = _scalar(particle_number, "particle_number", dtype=occupation_.dtype)
        charge_ = _scalar(charge, "charge", dtype=occupation_.dtype)
        trace = _scalar(trace_defect, "trace_defect", dtype=occupation_.dtype)
        kms = _scalar(kms_defect, "kms_defect", dtype=occupation_.dtype)
        if (
            occupation_.size == 0
            or coherence_r.shape != coherence_i.shape
            or coherence_r.size == 0
            or coordinates.size == 0
            or spectral.shape[-1] != coordinates.size
            or entropy.size == 0
        ):
            raise ValueError("Quantum/coherence/spectral observable shapes disagree.")
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in (
                        occupation_,
                        coherence_r,
                        coherence_i,
                        coordinates,
                        spectral,
                        entropy,
                        number,
                        charge_,
                        trace,
                        kms,
                    )
                )
            )
        )
        identities = tuple(
            _identifier(value, name)
            for value, name in (
                (quantum_profile_id, "quantum_profile_id"),
                (coherent_profile_id, "coherent_profile_id"),
                (off_shell_profile_id, "off_shell_profile_id"),
                (frame_id, "frame_id"),
                (frame_realization_id, "frame_realization_id"),
                (unit_contract_id, "unit_contract_id"),
            )
        )
        sources = _identifiers(source_ids, "quantum source ID")
        self.occupation = occupation_
        self.coherence_real = coherence_r
        self.coherence_imaginary = coherence_i
        self.spectral_coordinates = coordinates
        self.spectral_density = spectral
        self.entropy_density = entropy
        self.particle_number = number
        self.charge = charge_
        self.trace_defect = trace
        self.kms_defect = kms
        self.frame_token = jax.lax.stop_gradient(
            _scalar(frame_token, "frame_token", dtype=jnp.int32)
        )
        self.finite = finite
        self.successful = finite & _scalar_flag(successful, "successful")
        (
            self.quantum_profile_id,
            self.coherent_profile_id,
            self.off_shell_profile_id,
            self.frame_id,
            self.frame_realization_id,
            self.unit_contract_id,
        ) = identities
        self.source_ids = sources
        self.product_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-quantum-coherence-observables",
                "identities": list(identities),
                "sources": list(sources),
                "spectral_coordinates": array_tree_fingerprint(coordinates),
            }
        )

    def as_theory_vector(self, /) -> TheoryVector:
        return _theory_vector(
            (
                self.occupation,
                self.coherence_real,
                self.coherence_imaginary,
                self.spectral_density,
                self.entropy_density,
                self.particle_number,
                self.charge,
                self.trace_defect,
                self.kms_defect,
            ),
            "quantum-coherence",
            self.product_id,
        )


class RadiationObservables(StrictModule):
    """Radiation spectrum, Stokes polarization, ΔN_eff, and deposition."""

    spectral_coordinates: Array
    spectrum: Array
    stokes: Array
    delta_neff: Array
    energy_deposition: Array
    momentum_deposition: Array
    four_force_residual: Array
    frame_token: Array
    finite: Array
    successful: Array
    radiation_profile_id: str = eqx.field(static=True)
    packet_profile_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        spectral_coordinates: ArrayLike,
        spectrum: ArrayLike,
        stokes: ArrayLike,
        delta_neff: ArrayLike,
        energy_deposition: ArrayLike,
        momentum_deposition: ArrayLike,
        four_force_residual: ArrayLike,
        /,
        *,
        radiation_profile_id: str,
        packet_profile_id: str,
        frame_id: str,
        frame_token: ArrayLike,
        frame_realization_id: str,
        unit_contract_id: str,
        source_ids: Sequence[str],
        successful: ArrayLike,
    ):
        coordinates = _real_array(spectral_coordinates, "spectral_coordinates").reshape(
            (-1,)
        )
        spectrum_ = _real_array(spectrum, "spectrum", dtype=coordinates.dtype).reshape(
            (-1,)
        )
        stokes_ = _real_array(stokes, "stokes", dtype=coordinates.dtype)
        delta = _scalar(delta_neff, "delta_neff", dtype=coordinates.dtype)
        energy = _real_array(
            energy_deposition, "energy_deposition", dtype=coordinates.dtype
        )
        momentum = _real_array(
            momentum_deposition, "momentum_deposition", dtype=coordinates.dtype
        )
        residual = _real_array(
            four_force_residual, "four_force_residual", dtype=coordinates.dtype
        )
        if (
            coordinates.size == 0
            or spectrum_.shape != coordinates.shape
            or stokes_.shape != (coordinates.size, 4)
            or energy.size == 0
            or momentum.shape != energy.shape + (3,)
            or residual.shape != energy.shape + (4,)
        ):
            raise ValueError("Radiation observable shapes disagree.")
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in (
                        coordinates,
                        spectrum_,
                        stokes_,
                        delta,
                        energy,
                        momentum,
                        residual,
                    )
                )
            )
        )
        identities = tuple(
            _identifier(value, name)
            for value, name in (
                (radiation_profile_id, "radiation_profile_id"),
                (packet_profile_id, "packet_profile_id"),
                (frame_id, "frame_id"),
                (frame_realization_id, "frame_realization_id"),
                (unit_contract_id, "unit_contract_id"),
            )
        )
        sources = _identifiers(source_ids, "radiation source ID")
        self.spectral_coordinates = coordinates
        self.spectrum = spectrum_
        self.stokes = stokes_
        self.delta_neff = delta
        self.energy_deposition = energy
        self.momentum_deposition = momentum
        self.four_force_residual = residual
        self.frame_token = jax.lax.stop_gradient(
            _scalar(frame_token, "frame_token", dtype=jnp.int32)
        )
        self.finite = finite
        self.successful = finite & _scalar_flag(successful, "successful")
        (
            self.radiation_profile_id,
            self.packet_profile_id,
            self.frame_id,
            self.frame_realization_id,
            self.unit_contract_id,
        ) = identities
        self.source_ids = sources
        self.product_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-radiation-observables",
                "identities": list(identities),
                "sources": list(sources),
                "spectral_coordinates": array_tree_fingerprint(coordinates),
            }
        )

    def as_theory_vector(self, /) -> TheoryVector:
        return _theory_vector(
            (
                self.spectrum,
                self.stokes,
                self.delta_neff,
                self.energy_deposition,
                self.momentum_deposition,
                self.four_force_residual,
            ),
            "radiation",
            self.product_id,
        )


class FullDarkSectorLedgerObservables(StrictModule):
    """Component and global conservation/constraint/gauge/entropy evidence."""

    component_conservation: Array
    component_constraint: Array
    component_gauge: Array
    component_entropy_production: Array
    component_unitarity: Array
    global_conservation: Array
    global_constraint: Array
    global_gauge: Array
    global_entropy_production: Array
    global_unitarity: Array
    evidence_valid: Array
    finite: Array
    successful: Array
    component_names: tuple[str, ...] = eqx.field(static=True)
    source_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_conservation: ArrayLike,
        component_constraint: ArrayLike,
        component_gauge: ArrayLike,
        component_entropy_production: ArrayLike,
        component_unitarity: ArrayLike,
        global_conservation: ArrayLike,
        global_constraint: ArrayLike,
        global_gauge: ArrayLike,
        global_entropy_production: ArrayLike,
        global_unitarity: ArrayLike,
        evidence_valid: ArrayLike,
        /,
        *,
        component_names: Sequence[str],
        source_evidence_ids: Sequence[str],
        successful: ArrayLike,
    ):
        names = _identifiers(component_names, "ledger component name")
        component = tuple(
            _real_array(value, name).reshape((-1,))
            for value, name in (
                (component_conservation, "component_conservation"),
                (component_constraint, "component_constraint"),
                (component_gauge, "component_gauge"),
                (component_entropy_production, "component_entropy_production"),
                (component_unitarity, "component_unitarity"),
            )
        )
        expected = (len(names),)
        if any(value.shape != expected for value in component):
            raise ValueError("Component ledgers must match component_names.")
        globals_ = tuple(
            _scalar(value, name, dtype=component[0].dtype)
            for value, name in (
                (global_conservation, "global_conservation"),
                (global_constraint, "global_constraint"),
                (global_gauge, "global_gauge"),
                (global_entropy_production, "global_entropy_production"),
                (global_unitarity, "global_unitarity"),
            )
        )
        evidence = jnp.asarray(evidence_valid, dtype=bool).reshape((-1,))
        if evidence.shape != expected:
            raise ValueError("evidence_valid must match component_names.")
        finite = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(value)) for value in (*component, *globals_))
            )
        )
        evidence_ids = _identifiers(source_evidence_ids, "source evidence ID")
        if len(evidence_ids) != len(names):
            raise ValueError("Every ledger component requires one evidence identity.")
        (
            self.component_conservation,
            self.component_constraint,
            self.component_gauge,
            self.component_entropy_production,
            self.component_unitarity,
        ) = component
        (
            self.global_conservation,
            self.global_constraint,
            self.global_gauge,
            self.global_entropy_production,
            self.global_unitarity,
        ) = globals_
        self.evidence_valid = evidence
        self.finite = finite
        self.successful = (
            finite & jnp.all(evidence) & _scalar_flag(successful, "successful")
        )
        self.component_names = names
        self.source_evidence_ids = evidence_ids
        self.product_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-ledger-observables",
                "components": list(names),
                "evidence": list(evidence_ids),
            }
        )

    @classmethod
    def from_stage_ledger(
        cls, ledger: FullDarkSectorStageLedger, /
    ) -> "FullDarkSectorLedgerObservables":
        if not isinstance(ledger, FullDarkSectorStageLedger):
            raise TypeError("ledger must be FullDarkSectorStageLedger.")
        return cls(
            ledger.component_conservation,
            ledger.component_constraint,
            ledger.component_gauge,
            ledger.component_entropy_production,
            ledger.component_unitarity,
            ledger.total_conservation,
            ledger.total_constraint,
            ledger.total_gauge,
            ledger.total_entropy_production,
            ledger.total_unitarity,
            ledger.component_evidence_valid,
            component_names=ledger.component_names,
            source_evidence_ids=ledger.evidence_ids,
            successful=ledger.successful,
        )

    def as_theory_vector(self, /) -> TheoryVector:
        return _theory_vector(
            (
                self.component_conservation,
                self.component_constraint,
                self.component_gauge,
                self.component_entropy_production,
                self.component_unitarity,
                self.global_conservation,
                self.global_constraint,
                self.global_gauge,
                self.global_entropy_production,
                self.global_unitarity,
            ),
            "ledgers",
            self.product_id,
        )


class FullDarkSectorObservableBundle(StrictModule):
    metric_stress: MetricStressObservables
    event_shower_hadronization: EventShowerHadronizationObservables
    quantum_coherence: QuantumCoherenceObservables
    radiation: RadiationObservables
    ledgers: FullDarkSectorLedgerObservables
    finite: Array
    successful: Array
    stage_id: str = eqx.field(static=True)
    epoch_manifest_id: str = eqx.field(static=True)
    output_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric_stress: MetricStressObservables,
        event_shower_hadronization: EventShowerHadronizationObservables,
        quantum_coherence: QuantumCoherenceObservables,
        radiation: RadiationObservables,
        ledgers: FullDarkSectorLedgerObservables,
        /,
        *,
        stage_id: str,
        epoch_manifest_id: str,
    ):
        products = (
            metric_stress,
            event_shower_hadronization,
            quantum_coherence,
            radiation,
            ledgers,
        )
        expected = (
            MetricStressObservables,
            EventShowerHadronizationObservables,
            QuantumCoherenceObservables,
            RadiationObservables,
            FullDarkSectorLedgerObservables,
        )
        if any(not isinstance(value, kind) for value, kind in zip(products, expected)):
            raise TypeError(
                "Full dark-sector observable bundle contains a wrong product."
            )
        stage = _identifier(stage_id, "stage_id")
        epoch = _identifier(epoch_manifest_id, "epoch_manifest_id")
        if (
            event_shower_hadronization.event_manifest_id != epoch
            or len(
                {
                    metric_stress.frame_id,
                    quantum_coherence.frame_id,
                    radiation.frame_id,
                }
            )
            != 1
            or len(
                {
                    metric_stress.frame_realization_id,
                    quantum_coherence.frame_realization_id,
                    radiation.frame_realization_id,
                }
            )
            != 1
            or quantum_coherence.unit_contract_id != radiation.unit_contract_id
        ):
            raise ValueError(
                "Observable products do not share the epoch and exact frame realization."
            )
        frame_consistent = (
            metric_stress.frame_token == quantum_coherence.frame_token
        ) & (metric_stress.frame_token == radiation.frame_token)
        finite = eqx.error_if(
            jnp.all(jnp.stack(tuple(value.finite for value in products))),
            ~frame_consistent,
            "Observable products do not share one frame token.",
        )
        successful = finite & jnp.all(
            jnp.stack(tuple(value.successful for value in products))
        )
        self.metric_stress = metric_stress
        self.event_shower_hadronization = event_shower_hadronization
        self.quantum_coherence = quantum_coherence
        self.radiation = radiation
        self.ledgers = ledgers
        self.finite = finite
        self.successful = successful
        self.stage_id = stage
        self.epoch_manifest_id = epoch
        self.output_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-observable-bundle",
                "stage": stage,
                "epoch_manifest": epoch,
                "products": [value.product_id for value in products],
            }
        )

    def theory_vectors(self, /) -> tuple[TheoryVector, ...]:
        return (
            self.metric_stress.as_theory_vector(),
            self.event_shower_hadronization.as_theory_vector(),
            self.quantum_coherence.as_theory_vector(),
            self.radiation.as_theory_vector(),
            self.ledgers.as_theory_vector(),
        )


class ObservedFullDarkSectorBundle(StrictModule):
    metric_stress: TheoryVector
    event_shower_hadronization: TheoryVector
    quantum_coherence: TheoryVector
    radiation: TheoryVector
    ledgers: TheoryVector
    source_output_id: str = eqx.field(static=True)
    observation_plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class FullDarkSectorObservationPlan(StrictModule, NonTrainableState):
    """Composition of five existing linear observation owners."""

    metric_stress: LinearObservationPlan
    event_shower_hadronization: LinearObservationPlan
    quantum_coherence: LinearObservationPlan
    radiation: LinearObservationPlan
    ledgers: LinearObservationPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric_stress: LinearObservationPlan,
        event_shower_hadronization: LinearObservationPlan,
        quantum_coherence: LinearObservationPlan,
        radiation: LinearObservationPlan,
        ledgers: LinearObservationPlan,
        /,
    ):
        plans = (
            metric_stress,
            event_shower_hadronization,
            quantum_coherence,
            radiation,
            ledgers,
        )
        if any(not isinstance(value, LinearObservationPlan) for value in plans):
            raise TypeError("Every observation route must be LinearObservationPlan.")
        self.metric_stress = metric_stress
        self.event_shower_hadronization = event_shower_hadronization
        self.quantum_coherence = quantum_coherence
        self.radiation = radiation
        self.ledgers = ledgers
        self.plan_id = canonical_fingerprint(
            {
                "kind": "full-dark-sector-observation-plan",
                "owners": [value.plan_id for value in plans],
            }
        )

    def apply(
        self, bundle: FullDarkSectorObservableBundle, /
    ) -> ObservedFullDarkSectorBundle:
        if not isinstance(bundle, FullDarkSectorObservableBundle):
            raise TypeError("bundle must be FullDarkSectorObservableBundle.")
        sources = bundle.theory_vectors()
        plans = (
            self.metric_stress,
            self.event_shower_hadronization,
            self.quantum_coherence,
            self.radiation,
            self.ledgers,
        )
        observed = tuple(plan.apply(source) for plan, source in zip(plans, sources))
        product_id = canonical_fingerprint(
            {
                "kind": "observed-full-dark-sector-bundle",
                "source": bundle.output_id,
                "plan": self.plan_id,
                "products": [value.product_id for value in observed],
            }
        )
        return ObservedFullDarkSectorBundle(
            *observed,
            bundle.output_id,
            self.plan_id,
            product_id,
        )


__all__ = [
    "EventShowerHadronizationObservables",
    "FullDarkSectorLedgerObservables",
    "FullDarkSectorObservableBundle",
    "FullDarkSectorObservationPlan",
    "MetricStressObservables",
    "ObservedFullDarkSectorBundle",
    "QuantumCoherenceObservables",
    "RadiationObservables",
]
