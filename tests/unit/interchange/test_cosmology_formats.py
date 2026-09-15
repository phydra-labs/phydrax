#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib

import h5py
import numpy as np
import pytest

from phydrax.applications.cosmology._halos import SphericalOverdensityMassDefinition
from phydrax.interchange import AdapterStatus
from phydrax.interchange.cosmology import (
    read_concept_snapshot,
    read_hbt_herons_catalog,
    read_pinocchio_catalog,
    read_pinocchio_lineage,
)
from phydrax.qualification import ReferenceArtifactManifest


def _manifest(path, *, commercial=True):
    payload = path.read_bytes()
    return ReferenceArtifactManifest(
        path.name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic-test-data",
        commercial_use_permitted=commercial,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="unclassified-test-fixture",
        nondimensionalization={"length": 1.0, "mass": 1.0, "time": 1.0},
        uncertainty=None,
        lineage_ids=("independent-synthetic-fixture",),
    )


def _concept(path, *, dataset, values, ids=(4, 8)):
    with h5py.File(path, "w") as handle:
        handle.attrs["a"] = 0.5
        handle.attrs["boxsize"] = 10.0
        handle.attrs["unit time"] = "Gyr"
        handle.attrs["unit length"] = "Mpc"
        handle.attrs["unit mass"] = "Msun"
        handle.attrs["Ωcdm"] = 0.25
        group = handle.create_group("components/cdm")
        group.attrs["species"] = "cdm"
        group.attrs["N"] = 2
        group.attrs["mass"] = 3.0
        group.create_dataset("pos", data=np.asarray([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))
        group.create_dataset(dataset, data=np.asarray(values, dtype=float))
        group.create_dataset("ids", data=np.asarray(ids, dtype=np.int64))


def test_concept_momentum_and_velocity_are_canonicalized_by_declared_semantic(tmp_path):
    momentum_path = tmp_path / "momentum.hdf5"
    raw = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _concept(momentum_path, dataset="mom", values=raw)
    momentum = read_concept_snapshot(
        momentum_path,
        _manifest(momentum_path),
        component="cdm",
        position_scale=2.0,
        mass_scale=5.0,
        time_scale=10.0,
        target_length_unit="target-length",
        target_mass_unit="target-mass",
    )
    np.testing.assert_allclose(momentum.snapshot.canonical_momenta, raw)

    velocity_path = tmp_path / "velocity.hdf5"
    _concept(velocity_path, dataset="vel", values=raw)
    with pytest.raises(ValueError, match="requires kinematic_kind"):
        read_concept_snapshot(velocity_path, _manifest(velocity_path), component="cdm")
    velocity = read_concept_snapshot(
        velocity_path,
        _manifest(velocity_path),
        component="cdm",
        kinematic_kind="peculiar_velocity",
        mass_scale=2.0,
        position_scale=4.0,
        time_scale=2.0,
        target_length_unit="target-length",
        target_mass_unit="target-mass",
    )
    np.testing.assert_allclose(
        velocity.snapshot.canonical_momenta,
        (3.0 * 2.0) * 0.5 * raw * (4.0 / 2.0),
    )
    assert velocity.report.status == AdapterStatus.DECLARED_LOSS


def test_concept_rejects_rights_checksum_duplicate_ids_and_ambiguous_units(tmp_path):
    path = tmp_path / "snapshot.hdf5"
    _concept(path, dataset="mom", values=np.ones((2, 3)), ids=(4, 4))
    source = _manifest(path, commercial=False)
    with pytest.raises(PermissionError):
        read_concept_snapshot(path, source, component="cdm", commercial_use=True)
    with pytest.raises(ValueError, match="unique"):
        read_concept_snapshot(path, source, component="cdm")

    valid_path = tmp_path / "changed.hdf5"
    _concept(valid_path, dataset="mom", values=np.ones((2, 3)))
    stale = _manifest(valid_path)
    valid_path.write_bytes(valid_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        read_concept_snapshot(valid_path, stale, component="cdm")

    ambiguous_path = tmp_path / "ambiguous.hdf5"
    _concept(ambiguous_path, dataset="mom", values=np.ones((2, 3)))
    with h5py.File(ambiguous_path, "r+") as handle:
        del handle.attrs["unit time"]
    with pytest.raises(ValueError, match="ambiguous units"):
        read_concept_snapshot(ambiguous_path, _manifest(ambiguous_path), component="cdm")


def test_hbt_fields_memberships_and_relations_are_retained_without_aliasing(tmp_path):
    path = tmp_path / "SubSnap_003.hdf5"
    dtype = np.dtype(
        [
            ("TrackId", "<i8"),
            ("SinkTrackId", "<i8"),
            ("DescendantTrackId", "<i8"),
            ("NestedParentTrackId", "<i8"),
            ("Nbound", "<i8"),
            ("Mbound", "<f8"),
            ("TracerIndex", "<i8"),
            ("MostBoundParticleId", "<i8"),
            ("SnapshotOfBirth", "<i8"),
            ("SnapshotOfSink", "<i8"),
            ("SnapshotOfDeath", "<i8"),
        ]
    )
    rows = np.asarray(
        [
            (10, -1, -1, -1, 2, 4.5, 0, 100, 1, -1, -1),
            (20, 10, 10, 10, 0, 0.0, -1, -1, 2, 3, 3),
        ],
        dtype=dtype,
    )
    variable = h5py.vlen_dtype(np.dtype("<i8"))
    with h5py.File(path, "w") as handle:
        handle.attrs["SnapshotId"] = 3
        handle.create_dataset("Subhalos", data=rows)
        bound = handle.create_dataset("SubhaloParticles", (2,), dtype=variable)
        source = handle.create_dataset("SourceSubhaloParticles", (2,), dtype=variable)
        bound[0], bound[1] = np.asarray([100, 101]), np.asarray([], dtype=np.int64)
        source[0], source[1] = np.asarray([100, 101, 102]), np.asarray([200])
    result = read_hbt_herons_catalog(
        path,
        _manifest(path),
        maximum_halos=4,
        maximum_particles_per_halo=4,
    )

    np.testing.assert_array_equal(result.sidecar.field("Mbound")[:2], [4.5, 0.0])
    assert bool(result.lineage.sink_mask[0, 1])
    assert bool(result.lineage.descendant_mask[0, 1])
    assert result.lineage.sink_track_ids is not result.lineage.descendant_track_ids
    assert (
        result.lineage.snapshots[0].source_membership_ids[0]
        != result.lineage.snapshots[0].bound_membership_ids[0]
    )
    assert any(loss.path == "Subhalos.HostHaloId" for loss in result.report.losses)


def test_hbt_duplicate_tracks_fail_before_projection(tmp_path):
    path = tmp_path / "duplicate.hdf5"
    rows = np.asarray([(7,), (7,)], dtype=np.dtype([("TrackId", "<i8")]))
    with h5py.File(path, "w") as handle:
        handle.attrs["SnapshotId"] = 0
        handle.create_dataset("Subhalos", data=rows)
    with pytest.raises(ValueError, match="unique"):
        read_hbt_herons_catalog(
            path,
            _manifest(path),
            maximum_halos=2,
            maximum_particles_per_halo=1,
        )


def test_pinocchio_catalog_lightcone_and_lineage_stay_approximation_tagged(tmp_path):
    catalog_path = tmp_path / "pinocchio.catalog.out"
    catalog_path.write_text(
        "# documented fixture\n10 20 1 2 3 4 5 6 7 8 9 12\n20 30 2 3 4 5 6 7 8 9 10 14\n"
    )
    catalog = read_pinocchio_catalog(
        catalog_path,
        _manifest(catalog_path),
        maximum_halos=4,
        coordinate_frame="periodic_comoving_box",
        source_position_unit="Mpc/h",
        source_mass_unit="Msun/h",
        source_velocity_unit="km/s",
        mass_definition=SphericalOverdensityMassDefinition(200.0, "mean_matter"),
        scale_factor=0.5,
        box_size=(100.0, 100.0, 100.0),
    )
    assert catalog.report.status == AdapterStatus.DECLARED_LOSS
    assert "approximation" in catalog.sidecar.approximation_tag.lower()

    light_path = tmp_path / "pinocchio.plc.out"
    light_path.write_text("1 0.2 10 20 30 1 2 3 40 45 60 5 0.21\n")
    light = read_pinocchio_catalog(
        light_path,
        _manifest(light_path),
        product_kind="light_cone",
        maximum_halos=2,
        coordinate_frame="observer_relative_comoving",
        source_position_unit="Mpc/h",
        source_mass_unit="Msun/h",
        source_velocity_unit="km/s",
    )
    assert bool(light.light_cone.phase_space_known[0])
    assert "approximation" in light.light_cone.approximation_tag.lower()

    history_path = tmp_path / "pinocchio.histories.out"
    history_path.write_text(
        "# counts\n1 2\n10 1 2 2 5 9 1.0 3.0 2.0\n20 2 1 -1 9 0 -1 4.0 3.0\n"
    )
    lineage = read_pinocchio_lineage(
        history_path, _manifest(history_path), maximum_branches=4
    )
    assert int(lineage.history.sink_group_ids[0]) == 20
    assert int(lineage.history.descendant_group_ids[0]) == -1
    assert lineage.report.status == AdapterStatus.DECLARED_LOSS


def test_pinocchio_duplicate_group_ids_are_rejected(tmp_path):
    path = tmp_path / "duplicate.catalog.out"
    path.write_text("1 20 1 2 3 4 5 6 7 8 9\n1 30 2 3 4 5 6 7 8 9 10\n")
    with pytest.raises(ValueError, match="unique"):
        read_pinocchio_catalog(
            path,
            _manifest(path),
            maximum_halos=2,
            coordinate_frame="periodic_comoving_box",
            source_position_unit="Mpc/h",
            source_mass_unit="Msun/h",
            source_velocity_unit="km/s",
            mass_definition=SphericalOverdensityMassDefinition(200.0, "mean_matter"),
            scale_factor=1.0,
            box_size=(100.0, 100.0, 100.0),
        )
