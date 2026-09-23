#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)
from phydrax.particle_physics._hadronization import (
    DarkClusterFissionChannel,
    DarkClusterHadronizationPlan,
    DarkHadronPairChannel,
    DarkStringFragmentationPlan,
    decay_dark_cluster,
    fission_dark_cluster,
    fragment_dark_string,
)
from phydrax.particle_physics._identity import ParticleCatalogReference
from phydrax.particle_physics._species import ParticleSpeciesTable
from phydrax.solver._dark_sector_epoch_runtime import DarkSectorEpochPlan
from phydrax.units import COULOMB


def _contracts():
    units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 1, 1, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros(3),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(2),
        chart_id="flat",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="one-cell",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros(4),
        convention=units.convention,
        source_id="observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros(4),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="observer",
        orientation_id="future-right-handed",
    )
    catalog = ParticleCatalogReference(
        source_id="dark-hadrons",
        provider_release="test",
        checksum="checksum",
        citation_url="https://example.test/dark-hadrons",
    )
    species = ParticleSpeciesTable(
        jnp.asarray((100, -100, 200, -200, 201, -201)),
        jnp.asarray((0.0, 0.0, 1.0, 1.0, 2.0, 2.0)),
        jnp.asarray((1.0, -1.0, 1.0, -1.0, 1.0, -1.0)),
        catalog=catalog,
        energy_unit=units.energy_unit,
        charge_unit=COULOMB,
    )
    runtime = DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=4,
        product_capacity=8,
        radiation_capacity=1,
        work_capacity=8,
        frontier_capacity=8,
        packet_width=1,
        event_width=8,
        product_width=4,
        radiation_width=1,
        work_width=8,
        frontier_width=8,
        species_revision_id=species.table_id,
        topology_revision_id="4" * 64,
    )
    return units, frame, runtime, species


def _plans():
    units, frame, runtime, species = _contracts()
    channels = (
        DarkHadronPairChannel((200, -200), 2.0, spectrum_label="light-pair"),
        DarkHadronPairChannel((201, -201), 1.0, spectrum_label="heavy-pair"),
    )
    string = DarkStringFragmentationPlan(
        runtime,
        species,
        units,
        frame,
        channels,
        model_id="dark-string-model",
        model_revision_id="dark-string-r1",
        tune_id="declared-tune",
        string_tension=10.0,
        longitudinal_shape=(0.3, 0.8),
        production_evidence_ids=("string-validation",),
    )
    cluster = DarkClusterHadronizationPlan(
        runtime,
        species,
        units,
        frame,
        channels,
        (
            DarkClusterFissionChannel((2.0, 2.0), (1.0, -1.0), 1.0),
            DarkClusterFissionChannel((3.0, 3.0), (1.0, -1.0), 0.5),
        ),
        model_id="dark-cluster-model",
        model_revision_id="dark-cluster-r1",
        tune_id="declared-cluster-tune",
        fission_threshold=7.0,
        production_evidence_ids=("cluster-validation",),
    )
    return string, cluster


def test_declared_dark_string_spectrum_is_normalized_and_conservative():
    string, _ = _plans()
    result = fragment_dark_string(
        string,
        jnp.asarray(((5.0, 0.0, 0.0, 5.0), (5.0, 0.0, 0.0, -5.0))),
        (100, -100),
        jnp.asarray(((17, 0), (0, 17))),
        jnp.asarray((0.2, 0.3, 0.7)),
        parent_entity_id="string-parent",
        draw_id="string-draw-0",
    )
    assert bool(result.successful)
    assert not string.supports_generic_qcd
    np.testing.assert_allclose(result.channel_probabilities.sum(), 1.0, atol=1e-7)
    np.testing.assert_allclose(result.four_momentum_residual, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.charge_residual, 0.0, atol=1e-12)
    assert result.output_pdg_ids.shape == (2,)
    np.testing.assert_allclose(
        jnp.sum(result.output_four_momenta, axis=0),
        result.input_four_momentum,
        atol=1e-6,
    )
    np.testing.assert_allclose(jnp.sum(result.output_charges), 0.0, atol=1e-12)


def test_dark_cluster_decay_and_fission_normalize_channels_and_preserve_lineage_quantities():
    _, cluster = _plans()
    parent = jnp.asarray((10.0, 1.0, -0.5, 0.25))
    decay = decay_dark_cluster(
        cluster,
        parent,
        jnp.asarray(0.0),
        jnp.asarray((0.4, 0.2, 0.8)),
        parent_entity_id="cluster-parent",
        draw_id="cluster-decay-draw-0",
    )
    fission = fission_dark_cluster(
        cluster,
        parent,
        jnp.asarray(0.0),
        jnp.asarray((0.7, 0.6, 0.1)),
        parent_entity_id="cluster-parent",
        draw_id="cluster-fission-draw-0",
    )
    assert bool(decay.successful)
    assert int(fission.status) == 0
    assert bool(fission.successful)
    np.testing.assert_allclose(decay.channel_probabilities.sum(), 1.0, atol=1e-7)
    np.testing.assert_allclose(fission.channel_probabilities.sum(), 1.0, atol=1e-7)
    np.testing.assert_allclose(decay.four_momentum_residual, 0.0, atol=1e-5)
    np.testing.assert_allclose(fission.four_momentum_residual, 0.0, atol=1e-5)
    np.testing.assert_allclose(decay.charge_residual, 0.0, atol=1e-12)
    np.testing.assert_allclose(fission.charge_residual, 0.0, atol=1e-12)
    for result in (decay, fission):
        np.testing.assert_allclose(
            jnp.sum(result.output_four_momenta, axis=0),
            result.input_four_momentum,
            atol=1e-5,
        )
        np.testing.assert_allclose(jnp.sum(result.output_charges), 0.0, atol=1e-12)
    assert decay.parent_entity_id == fission.parent_entity_id == "cluster-parent"
