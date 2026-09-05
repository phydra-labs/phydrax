#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Hand-authored contract fixtures: never experimental/provider validation."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from phydrax.applications import radiation_biophysics as rad
from phydrax.applications.radiation_biophysics.interchange import (
    dnadamage1_column_payload,
    import_dnadamage1_columns,
    import_dnadamage1_root,
    NANOMETER,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.interchange import AdapterLoss
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ANGSTROM, ELECTRONVOLT, JOULE, KILOGRAM, MILLIGRAM, SECOND


def source_for(
    payload, *, training=True, artifact_kind="synthetic-external-radiation-columns"
):
    digest = hashlib.sha256(payload).hexdigest()
    rights = ReferenceArtifactManifest(
        "synthetic-radiation-contract-fixture",
        checksum_algorithm="sha256",
        checksum=digest,
        size_bytes=len(payload),
        license_id="LicenseRef-PHYDRA-synthetic-test",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=training,
        export_permitted=False,
        export_classification="test-only",
        nondimensionalization={"length_m": 1e-9, "energy_j": 1.602176634e-19},
        uncertainty=None,
        lineage_ids=("hand-authored-adapter-regression",),
    )
    artifact = ScientificArtifactEnvelope(
        artifact_kind=artifact_kind,
        content_digest=digest,
        producer="PHYDRA-test",
        producer_version="fixture",
        build_id="hand-authored",
        license_id=rights.license_id,
        resource_id="in-memory-test",
        status="complete",
    )
    return rad.RadiationSource(
        artifact,
        (rights,),
        "Geant4-dnadamage1",
        "v11.3.0",
        "synthetic-explicit-config",
        ("hand-authored-no-transport-RNG",),
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


def fixture(
    *, circular=True, direct_probability=1.0, threshold=17.5, extra_reaction=False
):
    physical = {
        "x": [0.0, 0.0, 2.0, 0.0],
        "y": [0.0] * 4,
        "z": [0.0] * 4,
        "edep": [10.0, 7.5, 17.5, 17.5],
        "diffKin": [99.0] * 4,
        "volumeName": [1] * 4,
        "CopyNumber": [0.0, 0.0, 1.0, 0.0],
        "EventID": [0, 0, 0, 1],
    }
    n = 2 if extra_reaction else 1
    chemical = {
        "x": [0.0] * n,
        "y": [0.0] * n,
        "z": [0.0] * n,
        "RadName": ["OH"] * n,
        "EventID": [0] * n,
    }
    source = source_for(dnadamage1_column_payload(physical, chemical, range(4), range(n)))
    imported = import_dnadamage1_columns(
        physical,
        chemical,
        source=source,
        run_id="run",
        fraction_id="fraction",
        physical_entry_ids=range(4),
        chemical_entry_ids=range(n),
        volume_materials={1: "G4_WATER"},
    )
    geometry = rad.RadiationTargetGeometry(
        (rad.TargetMolecule("plasmid", ("A", "B"), 12, circular),),
        (
            rad.TargetSite(
                10, "plasmid", "A", 0, "backbone", (0.0, 0.0, 0.0), 0.25, "G4_WATER"
            ),
            rad.TargetSite(
                20, "plasmid", "B", 11, "backbone", (2.0, 0.0, 0.0), 0.25, "G4_WATER"
            ),
        ),
        (rad.SourceTargetRoute("1:0", 10, 1.0), rad.SourceTargetRoute("1:1", 20, 1.0)),
        "synthetic-target-map",
        NANOMETER,
        "world",
        "target",
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (0.0, 0.0, 0.0),
        "scoring-only",
        "hand-authored-scoring-spheres",
    )
    prepared = rad.prepare_radiation_targets(geometry)
    mapping = rad.map_radiation_targets(imported.physical, imported.chemical, prepared)
    policy = rad.LesionPolicy(
        "test-policy",
        threshold,
        ELECTRONVOLT,
        direct_probability,
        (rad.IndirectLesionRule("OH-deoxyribose-damage", "OH", 1.0),),
        1e-9,
        SECOND,
        "synthetic-chemistry",
        "synthetic-scavenging",
    )
    candidates = rad.candidate_radiation_lesions(
        imported.physical, imported.chemical, mapping, geometry, policy
    )
    lesions = rad.realize_radiation_lesions(candidates, random_lineage="lesion-seed")
    return imported, geometry, mapping, policy, candidates, lesions


def test_event_order_dedup_history_identity_and_cause_union():
    imported, geometry, mapping, policy, candidates, lesions = fixture(
        extra_reaction=True
    )
    reordered = rad.InteractionLedger(
        imported.physical.source,
        imported.physical.records[::-1] + imported.physical.records[:1],
    )
    chemistry = rad.ReactionLedger(
        imported.chemical.source, imported.chemical.records[::-1]
    )
    assert reordered.fingerprint() == imported.physical.fingerprint()
    remapping = rad.map_radiation_targets(
        reordered, chemistry, rad.prepare_radiation_targets(geometry)
    )
    replay = rad.candidate_radiation_lesions(
        reordered, chemistry, remapping, geometry, policy
    )
    repeated = rad.realize_radiation_lesions(replay, random_lineage="lesion-seed")
    assert repeated == lesions
    assert (
        len(lesions.lesions) == 3
    )  # Two distinct primary histories at target 10 survive.
    both = [item for item in lesions.lesions if item.causes == ("direct", "indirect")]
    assert len(both) == 1 and len(both[0].candidate_ids) == 3
    conflict = replace(imported.physical.records[0], deposited_energy=999.0)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        rad.InteractionLedger(
            imported.physical.source, imported.physical.records + (conflict,)
        )
    assert all(
        candidate.deposited_energy_ev == 17.5
        for candidate in candidates.candidates
        if candidate.cause == "direct"
    )


def test_inclusive_deposition_threshold_and_exact_probability_edges():
    _, _, _, _, candidates, _ = fixture(direct_probability=0.5)
    uniforms = tuple((item.candidate_id, 0.5) for item in candidates.candidates)
    realized = rad.realize_radiation_lesions(
        candidates, random_lineage="edges", uniforms=uniforms
    )
    assert len(realized.lesions) == 1
    assert realized.lesions[0].causes == ("indirect",)
    zero_candidates = replace(
        candidates,
        candidates=tuple(
            replace(item, probability=0.0) for item in candidates.candidates
        ),
    )
    zero_draws = tuple((item.candidate_id, 0.0) for item in zero_candidates.candidates)
    assert not rad.realize_radiation_lesions(
        zero_candidates, random_lineage="zero", uniforms=zero_draws
    ).lesions
    _, _, _, _, above, _ = fixture(threshold=17.50001)
    assert all(item.cause == "indirect" for item in above.candidates)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        rad.realize_radiation_lesions(
            candidates,
            random_lineage="invalid",
            uniforms=tuple((item.candidate_id, 1.0) for item in candidates.candidates),
        )


def test_circular_dsb_and_linear_ssb_are_different_topologies():
    _, circle, _, _, _, circular_lesions = fixture(circular=True)
    circular = rad.cluster_radiation_lesions(
        circular_lesions, circle, maximum_contour_gap=1
    )
    assert sorted(item.classification for item in circular.clusters) == ["DSB", "SSB"]
    _, line, _, _, _, linear_lesions = fixture(circular=False)
    linear = rad.cluster_radiation_lesions(linear_lesions, line, maximum_contour_gap=1)
    assert [item.classification for item in linear.clusters] == ["SSB"] * 3
    with pytest.raises(ValueError, match="topology"):
        rad.cluster_radiation_lesions(circular_lesions, line, maximum_contour_gap=1)


def test_transitive_clusters_require_an_actual_opposite_strand_break_pair():
    _, geometry, _, _, candidates, lesions = fixture()
    original = lesions.lesions[0]
    history = original.history
    sites = tuple(
        rad.TargetSite(
            i,
            "plasmid",
            "A" if i < 2 else "B",
            i * 4,
            "base" if i == 1 else "backbone",
            (float(i), 0.0, 0.0),
            0.2,
            "water",
        )
        for i in range(3)
    )
    geometry = replace(
        geometry,
        sites=sites,
        routes=(),
        molecules=(rad.TargetMolecule("plasmid", ("A", "B"), 12, False),),
    )
    candidates = replace(candidates, geometry_id=geometry.fingerprint())
    members = tuple(
        replace(original, lesion_id=str(i), history=history, target_id=i)
        for i in range(3)
    )
    ledger = replace(lesions, lesions=members, candidates=candidates)
    clusters = rad.cluster_radiation_lesions(ledger, geometry, maximum_contour_gap=4)
    assert len(clusters.clusters) == 1  # 0 -> 4 -> 8, not only first-neighbor grouping.
    assert (
        clusters.clusters[0].classification == "SSB-cluster"
    )  # The base bridge is not a DSB.


def test_many_to_many_routes_conserve_deposition_and_units():
    imported, geometry, _, policy, _, _ = fixture()
    geometry = replace(
        geometry,
        routes=(
            rad.SourceTargetRoute("1:0", 10, 0.25),
            rad.SourceTargetRoute("1:0", 20, 0.75),
            rad.SourceTargetRoute("1:1", 20, 1.0),
        ),
    )
    mapping = rad.map_radiation_targets(
        imported.physical, imported.chemical, rad.prepare_radiation_targets(geometry)
    )
    direct = rad.candidate_radiation_lesions(
        imported.physical, imported.chemical, mapping, geometry, policy
    )
    assert [item.target_id for item in direct.candidates if item.cause == "direct"] == [
        20
    ]
    scaled_geometry = replace(
        geometry,
        length_unit=ANGSTROM,
        sites=tuple(
            replace(
                site, center=tuple(x * 10 for x in site.center), radius=site.radius * 10
            )
            for site in geometry.sites
        ),
    )
    scaled_mapping = rad.map_radiation_targets(
        imported.physical,
        imported.chemical,
        rad.prepare_radiation_targets(scaled_geometry),
    )
    assert mapping.hits == scaled_mapping.hits
    with pytest.raises(ValueError, match="sum to one"):
        replace(geometry, routes=(rad.SourceTargetRoute("1:0", 10, 0.5),))
    with pytest.raises(ValueError, match="dimensions"):
        replace(geometry, length_unit=JOULE)
    loss = AdapterLoss(
        "source/backbone",
        "import",
        "dropped",
        "missing target material",
        changes_interpretation=True,
    )
    with pytest.raises(ValueError, match="material"):
        replace(geometry, losses=(loss,))


def test_untimed_chemistry_cannot_be_recut_and_rights_are_admitted():
    imported, geometry, mapping, policy, _, _ = fixture()
    with pytest.raises(ValueError, match="Untimed"):
        rad.candidate_radiation_lesions(
            imported.physical,
            imported.chemical,
            mapping,
            geometry,
            replace(policy, chemistry_endpoint=5e-10),
        )
    with pytest.raises(PermissionError, match="commercial"):
        rad.map_radiation_targets(
            imported.physical,
            imported.chemical,
            rad.prepare_radiation_targets(geometry),
            commercial_use=True,
        )
    assert imported.physical.records[0].carried_energy is None
    assert imported.physical.records[0].kinetic_energy_loss == 99.0


def test_yields_include_zero_histories_and_explicit_mass_basepair_conventions():
    _, geometry, _, _, _, lesions = fixture()
    clusters = rad.cluster_radiation_lesions(lesions, geometry, maximum_contour_gap=1)
    histories = sorted({item.history for item in lesions.lesions})
    histories.append(replace(histories[0], primary_id="zero"))
    exposures = tuple(
        rad.HistoryExposure(history, 1.0, JOULE, 1.0, KILOGRAM, 1_000_000, 2)
        for history in histories
    )
    yield_ = rad.radiation_yield(
        lesions, clusters, exposures, observable="lesions", convention="per-Gy-per-Mbp"
    )
    assert yield_.value == pytest.approx(1.0)
    assert yield_.history_sampling_standard_error == pytest.approx(1 / 3**0.5)
    assert yield_.normalization_standard_error is None
    assert dict(yield_.history_counts)[histories[-1]] == 0
    scaled = tuple(replace(item, mass=1e6, mass_unit=MILLIGRAM) for item in exposures)
    assert rad.radiation_yield(
        lesions, clusters, scaled, observable="DSB", convention="per-Gy-per-molecule"
    ).value == pytest.approx(1 / 6)
    with pytest.raises(ValueError, match="fractions"):
        rad.radiation_yield(
            lesions,
            clusters,
            exposures
            + (
                replace(exposures[0], history=replace(histories[0], fraction_id="later")),
            ),
            observable="lesions",
            convention="per-Gy",
        )


def test_real_uproot_binary_ttree_roundtrip_and_required_semantic_refusal(tmp_path):
    # This is a real binary-provider invocation, but hand-authored scientific data.
    uproot = pytest.importorskip("uproot")
    path = tmp_path / "synthetic-dnadamage1.root"
    with uproot.recreate(path) as root:
        physical = root.mktree(
            "ntuple/ntuple_1",
            {
                "x": "float64",
                "y": "float64",
                "z": "float64",
                "edep": "float64",
                "diffKin": "float64",
                "volumeName": "int32",
                "CopyNumber": "float64",
                "EventID": "int32",
            },
        )
        physical.extend(
            {
                "x": np.asarray([0.0, 2.0]),
                "y": np.zeros(2),
                "z": np.zeros(2),
                "edep": np.asarray([17.5, 18.0]),
                "diffKin": np.asarray([99.0, 101.0]),
                "volumeName": np.asarray([1, 1], dtype=np.int32),
                "CopyNumber": np.asarray([0.0, 1.0]),
                "EventID": np.asarray([3, 4], dtype=np.int32),
            }
        )
        chemistry = root.mktree(
            "ntuple/ntuple_2",
            {
                "x": "float64",
                "y": "float64",
                "z": "float64",
                "RadName": "string",
                "EventID": "int32",
            },
        )
        chemistry.extend(
            {
                "x": np.zeros(1),
                "y": np.zeros(1),
                "z": np.zeros(1),
                "RadName": ["OH"],
                "EventID": np.asarray([3], dtype=np.int32),
            }
        )
    source = source_for(
        path.read_bytes(), artifact_kind="synthetic-external-radiation-ROOT"
    )
    imported = import_dnadamage1_root(
        path,
        source=source,
        run_id="binary-run",
        fraction_id="fraction",
        volume_materials={1: "G4_WATER"},
    )
    assert [record.deposited_energy for record in imported.physical.records] == [
        17.5,
        18.0,
    ]
    assert [record.key.history.primary_id for record in imported.physical.records] == [
        "3",
        "4",
    ]
    assert imported.chemical.records[0].reactants == ("OH", "Deoxyribose")
    assert (
        imported.chemical.records[0].key.history
        == imported.physical.records[0].key.history
    )
    with pytest.raises(ValueError, match="omits required"):
        import_dnadamage1_root(
            path,
            source=source,
            run_id="binary-run",
            fraction_id="fraction",
            volume_materials={1: "G4_WATER"},
            required_semantics=("event_time",),
        )
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes do not match"):
        import_dnadamage1_root(
            path,
            source=source,
            run_id="binary-run",
            fraction_id="fraction",
            volume_materials={1: "G4_WATER"},
        )
