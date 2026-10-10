#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Curved exact-periodic ADVANCING layers with immutable native core fill.

The source is a new authored revision. Its ``x = 1`` wall and quadratic map
coefficients are constructed from explicit ``x = 0`` orbit representatives by
one unit translation. The independently authored historical source remains a
negative exact-periodicity regression and is never rewritten by this example.
"""

from __future__ import annotations

import json

import phydrax as phx
from tools._hybrid_layer_source import (
    AuthoredPeriodicLayerSource,
    corrected_periodic_layer_source,
    source_identity_record,
)


def generate_curved_periodic_layer_core(
    *,
    capacity: int = 20_000,
    timeout: float = 120.0,
) -> tuple[AuthoredPeriodicLayerSource, phx.meshing.CellMeshingResult]:
    """Author and publish the corrected curved periodic hybrid source."""
    limits = phx.meshing.MeshingLimits(
        maximum_vertices=capacity,
        maximum_edges=16 * capacity,
        maximum_faces=16 * capacity,
        maximum_cells=8 * capacity,
        maximum_connectivity_entries=128 * capacity,
        maximum_data_bytes=4096 * capacity,
        maximum_work_units=128 * capacity,
        maximum_cavity_cells=capacity,
        maximum_geometry_queries=128 * capacity,
        maximum_scratch_bytes=8192 * capacity,
        maximum_wall_seconds=timeout,
    )
    authored = corrected_periodic_layer_source(limits=limits)
    result = (
        phx.meshing.NativeMeshingProvider(authored.options)
        .plan(
            authored.source,
            authored.specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    if result.certification is None or not result.certification.passed:
        raise RuntimeError("The corrected curved periodic source was not certified.")
    return authored, result


def main() -> None:
    authored, result = generate_curved_periodic_layer_core()
    certificate = result.certification
    if certificate is None:
        raise RuntimeError("The corrected source lost certification after publication.")
    print(
        json.dumps(
            {
                "source": source_identity_record(authored),
                "result_id": result.result_id,
                "mesh_id": result.mesh.mesh_id,
                "certification_id": certificate.report_id,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
