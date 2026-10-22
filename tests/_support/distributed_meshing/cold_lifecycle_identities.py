# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Lossless array and rational-coordinate records for cold lifecycle scenarios."""

from typing import Any, TypedDict

import numpy as np
from jax.typing import ArrayLike

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint
from phydrax.discretization import CellMesh, ExactPlcCellGeometrySource
from phydrax.discretization._coordinate_enclosure import CoordinateSourceBank
from phydrax.discretization._exact_algebraic_geometry import rational_coefficient
from phydrax.geometry.multiregion_surface._label_extraction import LabelFieldVolumeBinding


class ArrayRecord(TypedDict):
    dtype: str
    shape: list[int]
    bytes: str


def array_record(value: ArrayLike) -> ArrayRecord:
    """Return dtype, shape, and bytes without rounding numerical values."""
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes": array.tobytes().hex(),
    }


def record_array(record: ArrayRecord) -> np.ndarray:
    """Decode a scenario array using its recorded dtype and shape."""
    return np.frombuffer(
        bytes.fromhex(record["bytes"]), dtype=np.dtype(record["dtype"])
    ).reshape(record["shape"])


def coordinate_record(coordinates: CoordinateSourceBank) -> list[list[list[int]]]:
    """Return exact rational numerators and denominators in coordinate order."""
    return [
        [
            [rational.numerator, rational.denominator]
            for value in row
            for rational in (rational_coefficient(value, role="fixture coordinate"),)
        ]
        for row in coordinates
    ]


def mesh_record(mesh: CellMesh) -> dict[str, Any]:
    return {
        "coordinates": array_record(mesh.coordinates),
        "vertex_global_ids": array_record(mesh.vertex_global_ids),
        "blocks": [
            {
                "global_ids": array_record(block.global_ids),
                "vertices": array_record(block.vertices),
            }
            for block in mesh.blocks
        ],
    }


def plc_source_record(source: ExactPlcCellGeometrySource) -> dict[str, ArrayRecord]:
    return {
        "source_points": array_record(source.source_points),
        "source_triangles": array_record(source.source_triangles),
        "source_segments": array_record(source.source_segments),
        "source_triangle_ids": array_record(source.source_triangle_ids),
        "source_triangle_bounds": array_record(source.source_triangle_bounds),
        "source_segment_ids": array_record(source.source_segment_ids),
        "source_segment_bounds": array_record(source.source_segment_bounds),
        "vertex_strata": array_record(source.vertex_strata),
        "vertex_rows": array_record(source.vertex_rows),
        "vertex_parameters": array_record(source.vertex_parameters),
    }


def image_source_identity(
    source: phx.geometry.CompartmentMeshingSource | LabelFieldVolumeBinding,
) -> dict[str, Any]:
    source.validate_source_integrity()
    identity: dict[str, Any] = {
        "type": f"{type(source).__module__}.{type(source).__qualname__}",
        "source": source.source_id,
        "revision": source.source_revision,
        "interpretation": source.interpretation,
        "coordinates": source.coordinate_contract.spatial_id,
        "interfaces": [list(definition) for definition in source.interface_definitions],
    }
    if isinstance(source, phx.geometry.CompartmentMeshingSource):
        identity.update(
            {
                "values": array_tree_fingerprint(source.labels.asset.values),
                "valid_mask": array_tree_fingerprint(source.labels.asset.valid_mask),
                "affine": array_tree_fingerprint(
                    source.labels.asset.spatial_affine.matrix
                ),
                "labels": source.labels.label_volume_id,
                "ontology": source.labels.ontology.ontology_content_id,
                "complex": source.compartments.complex_id,
                "extraction": source.interfaces.extraction_id,
                "outer": source.outer_surface.model_id,
                "complex_alias": source.interfaces.complex is source.compartments,
            }
        )
    else:
        identity.update(
            {
                "binding": source.binding_id,
                "extraction": array_tree_fingerprint(source.extraction),
                "domain": array_tree_fingerprint(source.domain),
                "lineage_alias": source.lineage is source.extraction.lineage,
                "evidence_alias": source.extraction_evidence
                is source.extraction.evidence,
            }
        )
    return identity
