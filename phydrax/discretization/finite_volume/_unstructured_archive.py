#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..._array_archive import read_array_archive, write_array_archive
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._cell_complex import polygonal_connectivity, tetrahedral_connectivity
from ._unstructured import UnstructuredFiniteVolumePlan


def _canonical_archive_global_ids(name: str, value: Any, /) -> np.ndarray:
    identifiers = np.asarray(value)
    if identifiers.dtype != np.dtype(np.int64):
        raise ValueError(
            f"Unstructured mesh archive {name} must use canonical signed int64."
        )
    return identifiers


def _archive_payload_id(manifest: Mapping[str, Any], arrays: Mapping[str, Any], /) -> str:
    return canonical_fingerprint(
        {
            "manifest": dict(manifest),
            "arrays": {
                name: array_tree_fingerprint(value)
                for name, value in sorted(arrays.items())
            },
        }
    )


def write_unstructured_fv_archive(
    path: str | Path,
    plan: UnstructuredFiniteVolumePlan,
    /,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Write one canonical fixed-topology unstructured mesh archive."""

    if not isinstance(plan, UnstructuredFiniteVolumePlan):
        raise TypeError("plan must be UnstructuredFiniteVolumePlan.")
    arrays = {
        "vertices": np.asarray(plan.vertices),
        "triangles": np.asarray(plan.triangles, dtype=np.int32),
        "quadrilaterals": np.asarray(plan.quadrilaterals, dtype=np.int32),
        "tetrahedra": np.asarray(plan.tetrahedra, dtype=np.int32),
        "vertex_global_ids": np.asarray(plan.vertex_global_ids, dtype=np.int64),
        "cell_global_ids": np.asarray(plan.cell_global_ids, dtype=np.int64),
    }
    for name, face_indices in zip(plan.patch_names, plan.patch_faces, strict=True):
        arrays[f"patch/{name}"] = np.asarray(face_indices, dtype=np.int32)
    metadata = {
        "archive_kind": "unstructured-finite-volume-mesh",
        "schema_version": 1,
        "topology_id": plan.topology_id,
        "geometry_id": plan.geometry_id,
        "plan_id": plan.plan_id,
        "cell_dimension": plan.cell_dimension,
        "field_name": plan.field_name,
        "component_names": list(plan.component_names),
        "patch_names": list(plan.patch_names),
        "provenance": {"importer": "native"} if provenance is None else dict(provenance),
    }
    metadata["archive_id"] = _archive_payload_id(metadata, arrays)
    return write_array_archive(path, manifest=metadata, arrays=arrays)


def read_unstructured_fv_archive(path: str | Path, /) -> UnstructuredFiniteVolumePlan:
    """Read and identity-validate one canonical unstructured mesh archive."""

    manifest, arrays = read_array_archive(path)
    required_manifest = {
        "archive_kind",
        "schema_version",
        "topology_id",
        "geometry_id",
        "plan_id",
        "archive_id",
        "cell_dimension",
        "field_name",
        "component_names",
        "patch_names",
        "provenance",
        "arrays",
    }
    if set(manifest) != required_manifest:
        raise ValueError("Unstructured mesh archive manifest fields changed.")
    if (
        manifest["archive_kind"] != "unstructured-finite-volume-mesh"
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Unsupported unstructured finite-volume mesh archive.")
    patch_names = tuple(str(name) for name in manifest["patch_names"])
    expected_arrays = {
        "vertices",
        "triangles",
        "quadrilaterals",
        "tetrahedra",
        "vertex_global_ids",
        "cell_global_ids",
        *(f"patch/{name}" for name in patch_names),
    }
    if set(arrays) != expected_arrays:
        raise ValueError("Unstructured mesh archive array inventory changed.")
    metadata = {
        name: value
        for name, value in manifest.items()
        if name not in ("arrays", "archive_id")
    }
    if _archive_payload_id(metadata, arrays) != manifest["archive_id"]:
        raise ValueError("Unstructured mesh archive payload identity changed.")
    vertices = arrays["vertices"]
    triangles = arrays["triangles"]
    quadrilaterals = arrays["quadrilaterals"]
    tetrahedra = arrays["tetrahedra"]
    vertex_global_ids = _canonical_archive_global_ids(
        "vertex_global_ids", arrays["vertex_global_ids"]
    )
    cell_global_ids = _canonical_archive_global_ids(
        "cell_global_ids", arrays["cell_global_ids"]
    )
    dimension = int(manifest["cell_dimension"])
    if dimension == 2:
        connectivity = polygonal_connectivity(
            triangles, quadrilaterals, vertices.shape[0]
        )
        faces = np.asarray(connectivity.edges, dtype=np.int32)
    elif dimension == 3:
        connectivity = tetrahedral_connectivity(tetrahedra, vertices.shape[0])
        faces = np.asarray(connectivity.faces, dtype=np.int32)
    else:
        raise ValueError("Unstructured mesh archive cell dimension is invalid.")
    patches = {
        name: faces[np.asarray(arrays[f"patch/{name}"], dtype=np.int32)]
        for name in patch_names
    }
    plan = UnstructuredFiniteVolumePlan(
        vertices,
        triangles=triangles,
        quadrilaterals=quadrilaterals,
        tetrahedra=tetrahedra,
        vertex_global_ids=vertex_global_ids,
        cell_global_ids=cell_global_ids,
        boundary_patches=patches,
        field_name=manifest["field_name"],
        component_names=tuple(manifest["component_names"]),
    )
    if (
        plan.topology_id != manifest["topology_id"]
        or plan.geometry_id != manifest["geometry_id"]
        or plan.plan_id != manifest["plan_id"]
    ):
        raise ValueError("Unstructured mesh archive identity changed.")
    return plan


__all__ = ["read_unstructured_fv_archive", "write_unstructured_fv_archive"]
