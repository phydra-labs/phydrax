#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import hashlib
import json

import jax.numpy as jnp

from benchmarks._runtime import capture_environment, logical_array_bytes, measure_host
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
    fragment_dark_string_chain,
)
from phydrax.particle_physics._identity import ParticleCatalogReference
from phydrax.particle_physics._species import ParticleSpeciesTable
from phydrax.solver._dark_sector_epoch_runtime import DarkSectorEpochPlan
from phydrax.units import COULOMB


def _profiles(chain_capacity: int):
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
        source_id="benchmark-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros(4),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="benchmark-observer",
        orientation_id="future-right-handed",
    )
    runtime = DarkSectorEpochPlan(
        packet_capacity=1,
        event_capacity=chain_capacity,
        product_capacity=2 * chain_capacity,
        radiation_capacity=1,
        work_capacity=chain_capacity,
        frontier_capacity=chain_capacity,
        packet_width=1,
        event_width=8,
        product_width=4,
        radiation_width=1,
        work_width=8,
        frontier_width=8,
        species_revision_id="d" * 64,
        topology_revision_id="e" * 64,
    )
    catalog_payload = b"synthetic-hadron-catalog:(100,-100,101,200,-200,201,-201)"
    catalog = ParticleCatalogReference(
        source_id="synthetic-benchmark-hadrons",
        provider_release="inline-synthetic-fixture",
        checksum=hashlib.sha256(catalog_payload).hexdigest(),
        citation_url="urn:phydrax:benchmark:synthetic-hadrons",
    )
    species = ParticleSpeciesTable(
        jnp.asarray((100, -100, 101, 200, -200, 201, -201)),
        jnp.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0)),
        jnp.asarray((1.0, -1.0, 0.0, 1.0, -1.0, 1.0, -1.0)),
        catalog=catalog,
        energy_unit=units.energy_unit,
        charge_unit=COULOMB,
    )
    channels = (
        DarkHadronPairChannel((200, -200), 2.0, spectrum_label="light"),
        DarkHadronPairChannel((201, -201), 1.0, spectrum_label="heavy"),
    )
    string = DarkStringFragmentationPlan(
        runtime,
        species,
        units,
        frame,
        channels,
        model_id="benchmark-dark-string",
        model_revision_id="benchmark-string-r1",
        tune_id="benchmark-tune",
        string_tension=10.0,
        longitudinal_shape=(0.3, 0.8),
        production_evidence_ids=("benchmark-string-control",),
    )
    cluster = DarkClusterHadronizationPlan(
        runtime,
        species,
        units,
        frame,
        channels,
        (DarkClusterFissionChannel((2.0, 2.0), (1.0, -1.0), 1.0),),
        model_id="benchmark-dark-cluster",
        model_revision_id="benchmark-cluster-r1",
        tune_id="benchmark-cluster-tune",
        fission_threshold=7.0,
        production_evidence_ids=("benchmark-cluster-control",),
    )
    return string, cluster


def benchmark_case(chain_capacity: int):
    string, cluster = _profiles(chain_capacity)
    momenta = jnp.zeros((chain_capacity, 4))
    momenta = momenta.at[0].set((4.0, 0.0, 0.0, 4.0))
    momenta = momenta.at[1].set((2.0, 0.0, 0.0, 0.0))
    momenta = momenta.at[2].set((4.0, 0.0, 0.0, -4.0))
    pdg_ids = (
        jnp.zeros((chain_capacity,), dtype=jnp.int32)
        .at[0]
        .set(100)
        .at[1]
        .set(101)
        .at[2]
        .set(-100)
    )
    colors = jnp.zeros((chain_capacity, 2), dtype=jnp.int32)
    colors = colors.at[0].set((17, 0)).at[1].set((23, 17)).at[2].set((0, 23))
    active = jnp.arange(chain_capacity) < 3
    string_result, string_seconds = measure_host(
        lambda: fragment_dark_string_chain(
            string,
            momenta,
            pdg_ids,
            colors,
            active,
            jnp.asarray((0.3, 0.2, 0.8)),
            parent_entity_id="benchmark-string",
        )
    )
    parent = jnp.asarray((10.0, 1.0, -0.5, 0.25))
    cluster_result, cluster_seconds = measure_host(
        lambda: decay_dark_cluster(
            cluster,
            parent,
            jnp.asarray(0.0),
            jnp.asarray((0.2, 0.4, 0.7)),
            parent_entity_id="benchmark-cluster",
        )
    )
    fission_result, fission_seconds = measure_host(
        lambda: fission_dark_cluster(
            cluster,
            parent,
            jnp.asarray(0.0),
            jnp.asarray((0.4, 0.7, 0.2)),
            parent_entity_id="benchmark-cluster",
        )
    )
    return {
        "axes": {
            "event_count": 2,
            "chain_capacity": chain_capacity,
            "active_partons": 3,
            "hadron_channels": len(string.channels),
        },
        "ids": {
            "string_profile": string.profile_id,
            "cluster_profile": cluster.profile_id,
            "runtime_plan": string.runtime_plan.plan_id,
            "frame_realization": string.frame_realization_id,
        },
        "host_seconds": {
            "string_fragmentation": string_seconds,
            "cluster_decay": cluster_seconds,
            "cluster_fission": fission_seconds,
        },
        "resources": {
            "event_slots": 2,
            "fragmentation_cascade_nodes": int(jnp.sum(active))
            + string_result.output_pdg_ids.shape[0],
            "string_output_products": string_result.output_pdg_ids.shape[0],
            "cluster_output_products": cluster_result.output_pdg_ids.shape[0],
            "fission_output_clusters": fission_result.daughter_rest_energies.shape[0],
            "logical_bytes": logical_array_bytes(
                (string_result, cluster_result, fission_result)
            ),
        },
        "scientific_residuals": {
            "string_momentum": float(
                jnp.max(jnp.abs(string_result.four_momentum_residual))
            ),
            "cluster_momentum": float(
                jnp.max(jnp.abs(cluster_result.four_momentum_residual))
            ),
            "fission_momentum": float(
                jnp.max(jnp.abs(fission_result.four_momentum_residual))
            ),
            "maximum_charge": float(
                jnp.max(
                    jnp.abs(
                        jnp.asarray(
                            (
                                string_result.charge_residual,
                                cluster_result.charge_residual,
                                fission_result.charge_residual,
                            )
                        )
                    )
                )
            ),
        },
        "successful": bool(
            string_result.successful
            & cluster_result.successful
            & (fission_result.status == 0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-capacities", nargs="+", type=int, default=(3, 16, 64))
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 3 for value in arguments.chain_capacities):
        raise ValueError("chain capacities must be at least three.")
    cases = [benchmark_case(value) for value in arguments.chain_capacities]
    payload = {
        "environment": capture_environment().to_dict(),
        "source_kind": "synthetic-benchmark-fixture",
        "cases": cases,
        "passed": all(case["successful"] for case in cases),
    }
    encoded = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    if arguments.output:
        from benchmarks._io import write_json_atomic

        write_json_atomic(arguments.output, payload)
    else:
        print(encoded)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
