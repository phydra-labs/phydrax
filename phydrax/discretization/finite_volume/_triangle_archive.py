#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..._array_archive import read_array_archive, write_array_archive
from .._triangular import triangle_connectivity
from ._triangle_fv import TriangleFiniteVolumePlan


def write_triangle_fv_archive(
    path: str | Path,
    plan: TriangleFiniteVolumePlan,
    /,
) -> Path:
    if not isinstance(plan, TriangleFiniteVolumePlan):
        raise TypeError("plan must be TriangleFiniteVolumePlan.")
    arrays = {
        "vertices": np.asarray(plan.vertices),
        "triangles": np.asarray(plan.triangles, dtype=np.int32),
    }
    for name, edge_indices in zip(plan.patch_names, plan.patch_edges, strict=True):
        arrays[f"patch/{name}"] = np.asarray(edge_indices, dtype=np.int32)
    return write_array_archive(
        path,
        manifest={
            "archive_kind": "triangle-finite-volume-mesh",
            "schema_version": 1,
            "plan_id": plan.plan_id,
            "field_name": plan.field_name,
            "component_names": list(plan.component_names),
            "patch_names": list(plan.patch_names),
        },
        arrays=arrays,
    )


def read_triangle_fv_archive(path: str | Path, /) -> TriangleFiniteVolumePlan:
    manifest, arrays = read_array_archive(path)
    if (
        manifest.get("archive_kind") != "triangle-finite-volume-mesh"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("Unsupported triangle finite-volume mesh archive.")
    vertices = arrays["vertices"]
    triangles = arrays["triangles"]
    connectivity = triangle_connectivity(triangles, vertices.shape[0])
    edges = np.asarray(connectivity.edges, dtype=np.int32)
    patch_names = tuple(manifest["patch_names"])
    patches = {
        name: edges[np.asarray(arrays[f"patch/{name}"], dtype=np.int32)]
        for name in patch_names
    }
    plan = TriangleFiniteVolumePlan(
        vertices,
        triangles,
        boundary_patches=patches,
        field_name=manifest["field_name"],
        component_names=tuple(manifest["component_names"]),
    )
    if plan.plan_id != manifest["plan_id"]:
        raise ValueError("Triangle finite-volume mesh archive identity changed.")
    return plan


__all__ = ["read_triangle_fv_archive", "write_triangle_fv_archive"]
