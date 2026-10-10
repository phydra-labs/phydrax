#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native BRep projection and high-order coordinate acceptance evidence.

No OCCT or Gmsh is used. The exact sphere supplies a native meshing domain;
native constrained chart triangulation produces the initial mesh and an
independently checked continuous source-chart cover. Coordinate degrees
2, 3, 4, 6 and 10 are independent of the P1 consuming field.

Every projected candidate goes through local validity, global embedding and
continuous two-sided source certification. The example explicitly admits
200,000,000 native query operations for its five independent degree trials.
The report preserves any unresolved scientific or resource gate as rollback,
never replacing it by nodal residuals. Run
``python examples/cad_high_order_curving.py`` from the repository root.
"""

import json

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def main(degrees: tuple[int, ...] = (2, 3, 4, 6, 10)) -> None:
    contract = phx.SpatialCoordinateContract.si()
    model = phx.geometry.brep_sphere(
        1.0,
        coordinate_contract=contract,
        tessellation=phx.geometry.BRepTessellationPolicy(
            linear_deflection=0.15, angular_deflection=0.6
        ),
    )
    projection = phx.geometry.prepare_brep_projection(
        model,
        query_policy=phx.geometry.BRepQueryPolicy(maximum_operations=200_000_000),
    )
    domain = phx.geometry.MeshingDomain.from_brep(model)
    scope = phx.meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        phx.meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.source_indices[2], dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 3, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.7, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        protected_features=(
            phx.meshing.ProtectedFeature(
                scope, phx.meshing.FeatureKind.SURFACE, maximum_deviation=0.08
            ),
        ),
    )
    initial = (
        phx.meshing.NativeMeshingProvider(
            phx.meshing.NativeMeshingOptions("parametric_surface")
        )
        .plan(
            phx.meshing.NativeSurfaceSource(domain),
            specification,
            coordinate_contract=contract,
        )
        .execute()
    )
    if initial.certification is None or not initial.certification.passed:
        raise RuntimeError("Native surface generation did not publish certification.")
    source = initial.certification.request.source
    if source is None:
        raise RuntimeError("Native surface generation did not retain its source query.")
    mesh = initial.mesh
    association = phx.meshing.associate_mesh_vertices(
        mesh,
        projection,
        policy=phx.meshing.AssociationPropagationPolicy(classification_tolerance=1.0e-8),
    )
    reports = []
    for degree in degrees:
        policy = phx.meshing.HighOrderCurvingPolicy(
            degree=degree, relaxation_rounds=0, fidelity_tolerance=0.15
        )
        result = phx.meshing.curve_cell_mesh(
            mesh, association, projection, policy=policy, source=source
        )
        # The actual returned geometry, including rollback, is consumed by FE.
        field = phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        )
        space = phx.discretization.FiniteElementPlan(
            mesh, field, coordinate_spec=result.geometry
        ).prepare()
        mapped = space.evaluate_geometry("u", result.geometry.coordinates)
        area = float(sum(jnp.sum(block.measure) for block in mapped))
        candidate = result.candidate
        if candidate is None:
            raise RuntimeError("Native vertex association prevented projection.")
        if not result.evidence.accepted:
            np.testing.assert_array_equal(
                result.geometry.coordinates, result.straight.coordinates
            )
        reports.append(
            {
                "degree": degree,
                "status": result.status.value,
                "geometry_nodes": result.geometry.coordinates.shape[0],
                "accepted_round": result.accepted_round,
                "candidate_local_valid": candidate.certificate.all_certified,
                "candidate_invalid_cells": candidate.certificate.invalid_count,
                "candidate_unresolved_cells": candidate.certificate.unresolved_count,
                "candidate_node_residual": candidate.maximum_residual,
                "candidate_global_status": candidate.embedding.status,
                "mandatory_certificate_failures": candidate.certification_failures,
                "optimizer_statuses": [
                    status.name for status in result.relaxation_statuses
                ],
                "returned_geometry_area": area,
                "analytic_sphere_area": 4.0 * np.pi,
            }
        )
    print(
        json.dumps(
            {
                "source_revision": model.source_revision,
                "projection": "native_brep",
                "cells": mesh.entity_set(2).count,
                "curving": reports,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
