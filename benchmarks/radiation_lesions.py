#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Synthetic pinned-format mechanics benchmark, NOT experimental validation.

Run: .venv/bin/python benchmarks/radiation_lesions.py --histories 128 --repeats 3
Exercises actual import -> source routes/geometry -> lesions -> circular clusters
-> scored-dose yield, plus fixed-support JIT expectation and actual optim/UQ fit.
No external transport, spatial chemistry or performance comparison is claimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tracemalloc
from dataclasses import asdict

import jax
import jax.numpy as jnp
from _runtime import (
    capture_environment,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

from phydrax.applications import radiation_biophysics as rad
from phydrax.applications.radiation_biophysics.interchange import (
    dnadamage1_column_payload,
    import_dnadamage1_columns,
    NANOMETER,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import derived_unit, ELECTRONVOLT, JOULE, KILOGRAM, SECOND


def manifest(payload, *, uncertainty=None):
    return ReferenceArtifactManifest(
        "synthetic-radiation-benchmark",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-PHYDRA-synthetic-benchmark",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=True,
        export_permitted=False,
        export_classification="synthetic-benchmark",
        nondimensionalization={"length_m": 1e-9},
        uncertainty=uncertainty,
        lineage_ids=("hand-authored-not-provider-validation",),
    )


def run(histories: int, repeats: int):
    if histories < 2 or repeats < 1:
        raise ValueError("Benchmark requires at least two histories and one repeat.")
    physical = {
        "x": [0.0, 0.0, 2.0] * histories,
        "y": [0.0] * (3 * histories),
        "z": [0.0] * (3 * histories),
        "edep": [10.0, 7.5, 17.5] * histories,
        "diffKin": [20.0] * (3 * histories),
        "volumeName": [1] * (3 * histories),
        "CopyNumber": [0.0, 0.0, 1.0] * histories,
        "EventID": [event for event in range(histories) for _ in range(3)],
    }
    chemistry = {
        "x": [0.0] * histories,
        "y": [0.0] * histories,
        "z": [0.0] * histories,
        "RadName": ["OH"] * histories,
        "EventID": list(range(histories)),
    }
    payload = dnadamage1_column_payload(
        physical, chemistry, range(3 * histories), range(histories)
    )
    rights = manifest(payload)
    artifact = ScientificArtifactEnvelope(
        artifact_kind="synthetic-column-fixture",
        content_digest=rights.checksum,
        producer="PHYDRA-benchmark",
        producer_version="hand-authored",
        build_id="synthetic",
        license_id=rights.license_id,
        resource_id="in-memory",
        status="complete",
    )
    source = rad.RadiationSource(
        artifact,
        (rights,),
        "Geant4-dnadamage1",
        "v11.3.0",
        "synthetic-config",
        ("synthetic-no-transport-RNG",),
        ("synthetic-no-cross-section-table",),
        (),
        "world",
        NANOMETER,
        ELECTRONVOLT,
        SECOND,
        1e-9,
        "synthetic-chemistry",
        "synthetic-scavenging",
    )
    tracemalloc.start()
    imported, import_seconds = measure_host(
        lambda: import_dnadamage1_columns(
            physical,
            chemistry,
            source=source,
            run_id="benchmark",
            fraction_id="single-fraction",
            physical_entry_ids=range(3 * histories),
            chemical_entry_ids=range(histories),
            volume_materials={1: "G4_WATER"},
        )
    )
    geometry = rad.RadiationTargetGeometry(
        (rad.TargetMolecule("plasmid", ("A", "B"), 12, True),),
        (
            rad.TargetSite(
                10, "plasmid", "A", 0, "backbone", (0.0, 0.0, 0.0), 0.25, "G4_WATER"
            ),
            rad.TargetSite(
                20, "plasmid", "B", 11, "backbone", (2.0, 0.0, 0.0), 0.25, "G4_WATER"
            ),
        ),
        (rad.SourceTargetRoute("1:0", 10, 1.0), rad.SourceTargetRoute("1:1", 20, 1.0)),
        "synthetic-derived-geometry",
        NANOMETER,
        "world",
        "scoring",
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (0.0, 0.0, 0.0),
        "scoring-only",
        "hand-authored-spheres",
    )
    prepared, preparation_seconds = measure_synchronized(
        lambda: rad.prepare_radiation_targets(geometry)
    )
    mapping, mapping_timing = measure_repeated(
        lambda: rad.map_radiation_targets(imported.physical, imported.chemical, prepared),
        warmup=1,
        repeats=repeats,
    )
    policy = rad.LesionPolicy(
        "inclusive-17.5eV",
        17.5,
        ELECTRONVOLT,
        1.0,
        (rad.IndirectLesionRule("OH-deoxyribose-damage", "OH", 1.0),),
        1e-9,
        SECOND,
        "synthetic-chemistry",
        "synthetic-scavenging",
    )

    def classify():
        candidates = rad.candidate_radiation_lesions(
            imported.physical, imported.chemical, mapping, geometry, policy
        )
        lesions = rad.realize_radiation_lesions(
            candidates, random_lineage="benchmark-lesion-draws"
        )
        clusters = rad.cluster_radiation_lesions(lesions, geometry, maximum_contour_gap=1)
        return candidates, lesions, clusters

    (candidates, lesions, clusters), classification_timing = measure_repeated(
        classify, warmup=1, repeats=repeats
    )
    reverse = rad.InteractionLedger(source, imported.physical.records[::-1])
    reverse_mapping = rad.map_radiation_targets(reverse, imported.chemical, prepared)
    reverse_candidates = rad.candidate_radiation_lesions(
        reverse, imported.chemical, reverse_mapping, geometry, policy
    )
    reordered = rad.realize_radiation_lesions(
        reverse_candidates, random_lineage="benchmark-lesion-draws"
    )
    order_invariant = reordered == lesions
    primary_keys = tuple(sorted({item.key.history for item in imported.physical.records}))
    exposures = tuple(
        rad.HistoryExposure(history, 1.0, JOULE, 1.0, KILOGRAM, 12, 1)
        for history in primary_keys
    )
    yield_ = rad.radiation_yield(
        lesions, clusters, exposures, observable="DSB", convention="per-Gy-per-Mbp"
    )
    if (
        not order_invariant
        or len(clusters.clusters) != histories
        or any(item.classification != "DSB" for item in clusters.clusters)
    ):
        raise AssertionError("Radiation benchmark failed order/topology invariants.")
    support = rad.prepare_lesion_expectation(candidates, denominator=histories)
    direct = jnp.asarray([support.direct_multiplicity])
    indirect = jnp.asarray([support.indirect_multiplicity])
    mask = jnp.ones(direct.shape, dtype=bool)
    denominators = jnp.asarray([support.denominator])
    logits = jnp.zeros(2)
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(rad.expected_initial_lesion_yield).lower(
            logits, direct, indirect, mask, denominators
        ),
        lambda lowered: lowered.compile(),
    )
    expectation, expectation_timing = measure_repeated(
        lambda: compiled(logits, direct, indirect, mask, denominators),
        warmup=1,
        repeats=repeats,
    )
    # One dual-cause site (.75) and one direct-only site (.5) per primary.
    expectation_error = abs(float(expectation[0]) - 1.25)
    if expectation_error > 1e-6:
        raise AssertionError(
            "Compiled candidate union probability disagrees with exact reference."
        )

    fit_reference = manifest(
        b"synthetic-Gaussian-lesion-yield-benchmark", uncertainty={"declared_sigma": 0.01}
    )
    unit = derived_unit("Gy^-1", ((rad.GRAY, -1),))

    def data(prefix, pairs, offset):
        supports = tuple(
            rad.LesionExpectationSupport((d,), (i,), 1.0, f"{prefix}:{row}")
            for row, (d, i) in enumerate(pairs)
        )
        return rad.RadiationCalibrationData(
            tuple(f"{prefix}-observation:{row}" for row in range(len(pairs))),
            tuple(
                rad.RadiationCondition(
                    f"{prefix}-condition:{row}", 1.0, offset + row, 0.0, 1e-9
                )
                for row in range(len(pairs))
            ),
            supports,
            tuple(1 - 0.7**d * 0.4**i for d, i in pairs),
            (0.01,) * len(pairs),
            unit,
            "per-Gy",
            fit_reference,
            "synthetic",
        )

    training = data("fit", ((1, 0), (0, 1), (1, 1)), 0.0)
    validation = data("heldout", ((1, 2), (0, 3)), 10.0)
    calibration, calibration_seconds = measure_synchronized(
        lambda: rad.calibrate_radiation_lesions(
            training,
            validation,
            initial_logits=jnp.zeros(2),
            prior_mean=jnp.zeros(2),
            prior_standard_deviation=jnp.full(2, 3.0),
        )
    )
    calibration_error = float(
        jnp.max(
            jnp.abs(
                calibration.heldout_predictions - jnp.asarray(validation.observed_yields)
            )
        )
    )
    if calibration_error > 0.003:
        raise AssertionError(
            "Actual native calibration failed held-out synthetic prediction."
        )
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "environment": capture_environment().to_dict(),
        "profile": "synthetic-pinned-source-format-mechanics",
        "histories": histories,
        "physical_records": len(imported.physical.records),
        "chemical_records": len(imported.chemical.records),
        "active_targets": 2,
        "target_capacity": 2,
        "peak_traced_host_bytes": peak_memory,
        "source_digest": rights.checksum,
        "source_adapter_report": imported.report.report_id,
        "import_seconds": import_seconds,
        "prepare_seconds": preparation_seconds,
        "mapping": mapping_timing.to_dict(),
        "classification": classification_timing.to_dict(),
        "expectation_compilation": asdict(compilation),
        "expectation_runtime": expectation_timing.to_dict(),
        "expectation_absolute_error": expectation_error,
        "order_invariant": order_invariant,
        "dsb_clusters": len(clusters.clusters),
        "dsb_yield_per_Gy_per_Mbp": yield_.value,
        "yield_history_sampling_se": yield_.history_sampling_standard_error,
        "dose_uncertainty_known": yield_.normalization_standard_error is not None,
        "calibration_seconds": calibration_seconds,
        "heldout_absolute_error": calibration_error,
        "likelihood_rank": calibration.likelihood_rank,
        "scientific_gates": calibration.gates,
        "scope": "No real source event corpus or experimental validation; no transport/chemistry/repair model.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    print(json.dumps(run(options.histories, options.repeats), indent=2))
