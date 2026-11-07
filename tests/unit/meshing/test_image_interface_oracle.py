from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from tools.meshing_qualification.scenarios.image_material_profiles import (
    _categorical_source,
    _voxel_interface_oracle,
)


def test_generic_image_frame_never_assumes_anatomical_axes() -> None:
    contract = phx.SpatialCoordinateContract.si()
    affine = phx.imaging.ImageIndexAffine(
        np.eye(4), "voxel-index", contract, phx.imaging.ImageAxisConvention.XYZ
    )
    assert affine.coordinate_contract.spatial_id == contract.spatial_id
    assert affine.to_convention(phx.imaging.ImageAxisConvention.XYZ) is affine
    for convention in (
        phx.imaging.ImageAxisConvention.LPS,
        phx.imaging.ImageAxisConvention.RAS,
    ):
        with pytest.raises(ValueError, match="no declared anatomical orientation"):
            affine.to_convention(convention)
        anatomical = phx.imaging.ImageIndexAffine(
            np.eye(4),
            "voxel-index",
            phx.SpatialCoordinateContract(
                phx.units.METER, coordinate_system=f"cartesian-{convention.value.lower()}"
            ),
            convention,
        )
        expected_legacy_id = canonical_fingerprint(
            {
                "kind": "image-index-affine",
                "matrix": array_tree_fingerprint(np.eye(4)),
                "source_frame": "voxel-index",
                "spatial_contract": anatomical.coordinate_contract.spatial_id,
                "axis_convention": convention.value,
                "voxel_reference": "center",
                "provenance": "normalized",
            }
        )
        assert anatomical.affine_id == expected_legacy_id
        with pytest.raises(ValueError, match="no declared anatomical orientation"):
            anatomical.to_convention(phx.imaging.ImageAxisConvention.XYZ)
        other = (
            phx.imaging.ImageAxisConvention.RAS
            if convention is phx.imaging.ImageAxisConvention.LPS
            else phx.imaging.ImageAxisConvention.LPS
        )
        np.testing.assert_array_equal(
            anatomical.to_convention(other).to_convention(convention).matrix,
            anatomical.matrix,
        )


@pytest.mark.parametrize("fault", ("singular", "homogeneous", "nonfinite"))
def test_categorical_source_refuses_malformed_affine(fault: str) -> None:
    matrix = np.eye(4)
    if fault == "singular":
        matrix[0, :3] = 0.0
    elif fault == "homogeneous":
        matrix[3, 0] = 1.0
    else:
        matrix[0, 0] = np.nan
    with pytest.raises(ValueError):
        _categorical_source(
            "malformed-affine",
            np.array([[[0]], [[1]]], dtype=np.int32),
            ("material-a", "material-b"),
            matrix,
            phx.SpatialCoordinateContract.si(),
        )


def test_categorical_source_refuses_forged_region_revision() -> None:
    contract = phx.SpatialCoordinateContract.si()
    ids = ("material-a", "material-b")
    first = _categorical_source(
        "region-source",
        np.array([[[0]], [[0]], [[1]]], dtype=np.int32),
        ids,
        np.eye(4),
        contract,
    )
    second = _categorical_source(
        "region-source",
        np.array([[[0]], [[1]], [[1]]], dtype=np.int32),
        ids,
        np.eye(4),
        contract,
    )
    assert first.labels.label_volume_id != second.labels.label_volume_id
    with pytest.raises(ValueError, match="revision"):
        phx.geometry.CompartmentMeshingSource(
            first.labels, second.compartments, first.outer_surface, first.interfaces
        )


def test_categorical_source_refuses_forged_region_measures() -> None:
    source = _categorical_source(
        "forged-regions",
        np.array([[[0]], [[0]], [[1]]], dtype=np.int32),
        ("material-a", "material-b"),
        np.eye(4),
        phx.SpatialCoordinateContract.si(),
    )
    measures = source.compartments.compartment_measures
    forged = replace(
        source.compartments,
        compartment_measures=((measures[0][0], measures[0][1] * 2), measures[1]),
    )
    interfaces = replace(source.interfaces, complex=forged)
    with pytest.raises(ValueError, match="measures or observed adjacency"):
        phx.geometry.CompartmentMeshingSource(
            source.labels, forged, source.outer_surface, interfaces
        )


@pytest.mark.parametrize("oblique", (False, True))
def test_categorical_source_cold_archive_and_forgery_refusal(
    tmp_path: Path, oblique: bool
) -> None:
    from phydrax._model._structure import model_archive_projection
    from phydrax.lifecycle._composition_rebind import Composition, CompositionEntry
    from phydrax.meshing._model_codec import (
        restore_model_archive_value,
        write_model_artifact,
    )

    grid = np.indices((4, 4, 4))
    if oblique:
        values = np.where(grid[0] < 2, np.where(grid[1] < 2, 0, 1), 2)
        ids = ("material-a", "material-b", "material-c")
        linear = np.array(((-1.0, 0.2, 0.05), (0.0, 1.0, 0.1), (0.0, 0.0, 1.0)))
    else:
        values = grid[0] >= 2
        ids = ("material-a", "material-b")
        linear = np.eye(3)
    affine = np.eye(4)
    affine[:3, :3] = 0.25 * linear
    affine[:3, 3] = 0.125 + (np.array((0.0, -0.2, 0.1)) if oblique else 0.0)
    source = _categorical_source(
        "cold-image",
        values.astype(np.int32),
        ids,
        affine,
        phx.SpatialCoordinateContract.si(),
    )
    entry = CompositionEntry(
        source,
        entry_id="image/source",
        role="topology",
        owner_id="phydrax.geometry",
        structure_id=source.source_id,
        revision_id=source.source_revision,
        semantics_id="categorical-image-source",
    )
    composition = Composition((entry,), boundary_id="image-cold-restore")
    path = tmp_path / "image.phx"
    write_model_artifact(path, composition, limits=None)
    script = """
import json, sys
from phydrax.lifecycle._composition_rebind import Composition
from phydrax.meshing._model_codec import read_model_artifact, restore_model_archive_value
composition = read_model_artifact(sys.argv[1], expected_type=Composition,
    restore=restore_model_archive_value, limits=None)
source = composition.value("image/source")
print(json.dumps({"source_id": source.source_id, "revision": source.source_revision,
    "composition_id": composition.composition_id,
    "affine": source.labels.asset.spatial_affine.matrix.tolist(),
    "values": source.labels.asset.values.tolist(),
    "scope_ids": source.scope_entity_set_ids,
    "complex_id": source.compartments.complex_id,
    "extraction_id": source.interfaces.extraction_id}))
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    rebuilt = json.loads(child.stdout)
    assert rebuilt["source_id"] == source.source_id
    assert rebuilt["revision"] == source.source_revision
    assert rebuilt["composition_id"] == composition.composition_id
    assert rebuilt["affine"] == affine.tolist()
    assert rebuilt["values"] == values.astype(np.int32).tolist()
    assert rebuilt["scope_ids"] == list(source.scope_entity_set_ids)
    assert rebuilt["complex_id"] == source.compartments.complex_id
    assert rebuilt["extraction_id"] == source.interfaces.extraction_id

    recipe = model_archive_projection(source)
    shifted = affine.copy()
    shifted[0, 3] += 0.125
    forged_affine = replace(source.labels.asset.spatial_affine, matrix=shifted)
    forged_fields = {**recipe.labels.asset_fields, "spatial_affine": forged_affine}
    compartments = recipe.compartments
    swapped = (
        replace(compartments[0], label_ids=compartments[1].label_ids),
        replace(compartments[1], label_ids=compartments[0].label_ids),
        *compartments[2:],
    )
    for forged in (
        recipe._replace(expected_source_revision="forged-revision"),
        recipe._replace(expected_source_id="forged-source"),
        recipe._replace(
            labels=recipe.labels._replace(expected_label_volume_id="forged-labels")
        ),
        recipe._replace(labels=recipe.labels._replace(asset_fields=forged_fields)),
        recipe._replace(compartments=swapped),
    ):
        with pytest.raises((ValueError, TypeError)):
            restore_model_archive_value(forged)


@pytest.mark.parametrize(
    ("index_point", "accepted"),
    (((0.5, 1.0, 1.0), True), ((1.5, 1.0, 1.0), False), ((-0.5, 1.0, 1.0), False)),
)
def test_affine_seed_requires_material_interior(
    index_point: tuple[float, float, float], accepted: bool
) -> None:
    from phydrax.meshing._compartments import prepare_compartment_plc
    from phydrax.meshing._contracts import VolumeMeshingSpec
    from phydrax.meshing._controls import RegionSeed
    from phydrax.meshing._organization import RegionRole
    from phydrax.meshing._volume_generation import _region_seed_points
    from tools.meshing_qualification.scenarios.image_material_profiles import (
        _image_profile_inputs,
        _world_seed,
    )

    source, original, _, _ = _image_profile_inputs("seed-boundaries", oblique=False)
    affine = source.labels.asset.spatial_affine.matrix
    seed = RegionSeed(
        _world_seed(affine, index_point), "material-a", "material-a", RegionRole.SOLID
    )
    specification = VolumeMeshingSpec(
        original.target,
        original.boundary_scope,
        original.fill_strategy,
        size_controls=original.size_controls,
        region_seeds=(seed,),
    )
    complex_ = prepare_compartment_plc(source).complex
    if accepted:
        points, regions = _region_seed_points(
            complex_, specification, affine=affine, affine_source=source
        )
        np.testing.assert_array_equal(points[0], index_point)
        assert complex_.region_ids[int(regions[0])] == "material-a"
    else:
        with pytest.raises(ValueError, match="same declared material interior"):
            _region_seed_points(
                complex_, specification, affine=affine, affine_source=source
            )


def _fixture() -> tuple[
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    list[tuple[tuple[str, np.ndarray], ...]],
]:
    labels = np.array([[[0]], [[1]]], dtype=np.int32)
    ids = ("material-a", "material-b")
    affine = np.array(
        (
            (-0.25, 0.05, 0.0125, 0.2),
            (0.0, 0.25, 0.025, -0.1),
            (0.0, 0.0, 0.25, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    triangles = []
    sides = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for index in np.ndindex(labels.shape):
            for direction in (-1, 1):
                neighbor = list(index)
                neighbor[axis] += direction
                inside = 0 <= neighbor[axis] < labels.shape[axis]
                if inside and labels[tuple(neighbor)] == labels[index]:
                    continue
                if inside and direction < 0:
                    continue
                corners = []
                for u, v in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
                    point = np.array(index, dtype=float)
                    point[axis] += direction * 0.5
                    point[other[0]] += u
                    point[other[1]] += v
                    corners.append(affine[:3, :3] @ point + affine[:3, 3])
                adjacent = [
                    (
                        ids[int(labels[index])],
                        affine[:3, :3] @ np.array(index) + affine[:3, 3],
                    )
                ]
                if inside:
                    adjacent.append(
                        (
                            ids[int(labels[tuple(neighbor)])],
                            affine[:3, :3] @ np.array(neighbor) + affine[:3, 3],
                        )
                    )
                for vertices in ((0, 1, 2), (0, 2, 3)):
                    triangles.append([corners[i] for i in vertices])
                    sides.append(tuple(adjacent))
    return labels, ids, affine, np.asarray(triangles), sides


def test_reflected_sheared_voxel_faces_are_covered() -> None:
    labels, ids, affine, triangles, sides = _fixture()
    evidence = _voxel_interface_oracle(labels, ids, affine, triangles, sides)
    assert evidence["voxel_face_count"] == 11
    distance = evidence["maximum_plane_distance_m"]
    assert isinstance(distance, float)
    assert distance < 1e-10


@pytest.mark.parametrize(
    "fault", ("displacement", "material", "orientation", "missing", "duplicate")
)
def test_voxel_face_oracle_rejects_faults(fault: str) -> None:
    labels, ids, affine, triangles, sides = _fixture()
    if fault == "displacement":
        triangles[0] += np.array((0.01, 0.0, 0.0))
    elif fault == "material":
        label, center = sides[0][0]
        sides[0] = ((ids[1] if label == ids[0] else ids[0], center),)
    elif fault == "orientation":
        index = next(i for i, side in enumerate(sides) if len(side) == 2)
        (first, a), (second, b) = sides[index]
        sides[index] = ((first, b), (second, a))
    elif fault == "missing":
        triangles = triangles[1:]
        sides = sides[1:]
    else:
        triangles = np.concatenate((triangles, triangles[:1]))
        sides.append(sides[0])
    with pytest.raises(RuntimeError):
        _voxel_interface_oracle(labels, ids, affine, triangles, sides)
