#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native hard event through calorimeter response and weighted analysis."""

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def main() -> None:
    scattering = phx.applications.relativistic_scattering
    detector = phx.applications.detector
    calorimetry = detector.calorimetry
    analysis = phx.applications.collider_analysis
    catalogue = phx.particle_physics.ParticleCatalogueReference(
        source_id="example-pdg",
        provider_release="example",
        checksum="example-checksum",
        citation_url="https://pdg.lbl.gov/",
    )
    incoming = (
        scattering.Particle(
            "electron",
            mass=0.0,
            charge=-1.0,
            spin_twice=1,
            antiparticle="positron",
            statistics="fermion",
        ),
        scattering.Particle(
            "positron",
            mass=0.0,
            charge=1.0,
            spin_twice=1,
            antiparticle="electron",
            statistics="fermion",
        ),
    )
    outgoing = (
        scattering.Particle(
            "muon",
            mass=0.0,
            charge=-1.0,
            spin_twice=1,
            antiparticle="antimuon",
            statistics="fermion",
        ),
        scattering.Particle(
            "antimuon",
            mass=0.0,
            charge=1.0,
            spin_twice=1,
            antiparticle="muon",
            statistics="fermion",
        ),
    )
    process = scattering.ScatteringProcess("example", incoming, outgoing)
    hard = scattering.HardProcessPlan(
        process,
        scattering.BeamPlan((11, -11), 10.0),
        scattering.ScalePlan(10.0, 10.0, scheme_id="fixed-example"),
        outgoing_pdg_ids=(13, -13),
        matrix_element_id="constant-reference",
        coupling_scheme_id="example",
    ).prepare(lambda incoming_momenta, outgoing_momenta: jnp.asarray(1.0))
    event_plan = phx.particle_physics.ParticleEventPlan(
        catalogue=catalogue,
        momentum_unit=phx.units.GIGAELECTRONVOLT,
        length_unit=phx.units.MILLIMETER,
        time_unit=phx.units.NANOSECOND,
        event_capacity=4,
        particle_capacity=4,
        vertex_capacity=1,
        provider_status_namespace="native-qed",
    )
    produced = scattering.produce_hard_events(
        hard,
        jnp.asarray([[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.8, 0.9]]),
        event_plan,
    )
    conditions = detector.DetectorConditions(
        magnetic_field=jnp.zeros(3),
        electric_field=jnp.zeros(3),
        momentum_unit=phx.units.GIGAELECTRONVOLT,
        length_unit=phx.units.MILLIMETER,
        time_unit=phx.units.NANOSECOND,
        geometry_id="two-cell",
        material_id="example-material",
        field_id="zero-field",
        alignment_id="nominal",
        calibration_id="unit-gain",
        validity_interval=(0, 1),
    )
    geometry = calorimetry.CalorimeterGeometry(
        cell_ids=jnp.asarray([10, 11]),
        channel_ids=jnp.asarray([0, 1]),
        layer_ids=jnp.asarray([0, 1]),
        subdetector_ids=jnp.asarray([0, 0]),
        material_ids=jnp.asarray([0, 0]),
        readout_ids=jnp.asarray([0, 0]),
        centroids=jnp.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        volumes=jnp.ones(2),
        active=jnp.ones(2, dtype=bool),
        dead=jnp.zeros(2, dtype=bool),
        senders=jnp.asarray([0, 1]),
        receivers=jnp.asarray([1, 0]),
        conditions_id=conditions.conditions_id,
    )
    hit_energy = produced.events.momenta[:, 2:4, 0]
    hits = detector.SensitiveHitBank(
        event_ids=produced.events.event_ids,
        hit_ids=jnp.broadcast_to(jnp.arange(2), (4, 2)),
        detector_element_ids=jnp.broadcast_to(jnp.arange(2), (4, 2)),
        channel_ids=jnp.broadcast_to(jnp.arange(2), (4, 2)),
        source_step_indices=jnp.broadcast_to(jnp.arange(2), (4, 2)),
        positions=jnp.zeros((4, 2, 3)),
        times=jnp.zeros((4, 2)),
        energies=hit_energy,
        active=jnp.ones((4, 2), dtype=bool),
        conditions_id=conditions.conditions_id,
    )
    incident = jnp.sum(hit_energy, axis=1)
    truth = calorimetry.route_calorimeter_hits(
        geometry,
        hits,
        incident,
        incident,
        leakage_energy=jnp.zeros(4),
        leakage_known=jnp.ones(4, dtype=bool),
        source_id=produced.prepared_id,
    )
    response = calorimetry.apply_calorimeter_response(
        calorimetry.CalorimeterResponsePlan(
            geometry,
            gain=jnp.ones(2),
            noise_standard_deviation=jnp.zeros(2),
            crosstalk=jnp.zeros((2, 2)),
            adc_lsb=0.001,
            threshold=0.0,
            maximum_adc=100_000,
        ),
        truth,
        jr.key(7),
    )
    observed = calorimetry.calorimeter_observables(
        geometry, response.reconstructed_cell_energy
    )
    histogram = analysis.fill_weighted_histogram(
        analysis.HistogramPlan(
            jnp.asarray([0.0, 9.0, 11.0]), observable_id="visible-energy", unit_id="GeV"
        ),
        observed.total_energy,
        produced.events.weights.nominal,
        active=produced.selected,
    )
    if (
        not bool(produced.successful)
        or not bool(truth.successful)
        or not bool(histogram.finite)
    ):
        raise RuntimeError("HEP production example failed qualification checks.")
    print(
        {
            "events": int(produced.ledger.generated_count),
            "sum_weights": histogram.sum_weights.tolist(),
            "energy_residual": truth.deposited_residual.tolist(),
        }
    )


if __name__ == "__main__":
    main()
