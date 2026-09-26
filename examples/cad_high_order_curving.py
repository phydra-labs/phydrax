#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Curve Gmsh tetrahedra of OCCT solids onto their exact B-Rep and certify them.

For an OCCT cylinder and sphere: persist the solid as a native BREP and import the
published revision, mesh it with Gmsh into coarse straight P1 tetrahedra, bind OCCT
projectors to the imported revision, classify every mesh vertex on its B-Rep entity
(and derive the edge, face, and cell classes from them), curve the mesh to P2 and P3
geometry, and certify every curved cell with Bernstein determinant bounds. The
exactly integrated volumes of the straight and curved geometry are compared with the
exact solid volumes. Finally a Gmsh second-order output of the sphere is verified
against the B-Rep without moving it.

Requires the optional Gmsh and OCP (OpenCascade) dependencies. Run with
``python examples/cad_high_order_curving.py``.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere

import phydrax as phx


CONTRACT = phx.SpatialCoordinateContract.si()
ASSOCIATION = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1.0e-8)
RADIUS = 1.0
HEIGHT = 2.0


def mesh_solid(source, size, geometry_order):
    """Coarse Gmsh tetrahedra of one closed solid at the requested geometry order."""
    provider = phx.meshing.GmshProvider()
    scope = provider.whole_scope(source, 3)
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(required=("tetrahedron",)),
            geometry_order=geometry_order,
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, size, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    result = provider.plan(source, specification).execute()
    if not result.audit.passed:
        raise RuntimeError("The Gmsh tetrahedral mesh failed its audit.")
    return result


def mapped_volume(mesh, geometry):
    """Volume of the mapped geometry.

    The degree-aware finite-element rule integrates the polynomial Jacobian
    determinant of P1, P2, and P3 tetrahedra exactly.
    """
    field = phx.discretization.FiniteElementFieldSpec(
        "x", phx.discretization.lagrange_element("tetrahedron", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(
        mesh, field, coordinate_spec=geometry
    ).prepare()
    blocks = discretization.evaluate_geometry("x", geometry.coordinates)
    return float(sum(jnp.sum(block.measure) for block in blocks))


def classify(mesh, projection):
    """Vertex classes plus the derived edge, face, and cell classes."""
    vertices = phx.meshing.associate_mesh_vertices(mesh, projection, policy=ASSOCIATION)
    associations = (vertices,) + tuple(
        phx.meshing.associate_mesh_entities(
            mesh, vertices, projection, dimension, policy=ASSOCIATION
        )
        for dimension in (1, 2, 3)
    )
    counts = {
        str(dimension): {
            "rows": association.target_global_ids.shape[0],
            "resolved": int(np.count_nonzero(np.asarray(association.resolved))),
            "ambiguous": int(np.count_nonzero(np.asarray(association.ambiguous))),
            "maximum_residual": float(np.max(np.asarray(association.residuals))),
        }
        for dimension, association in enumerate(associations)
    }
    if any(
        entry["resolved"] != entry["rows"] or entry["ambiguous"]
        for entry in counts.values()
    ):
        raise RuntimeError("Some mesh entity has no unique B-Rep class.")
    if counts["0"]["maximum_residual"] > ASSOCIATION.classification_tolerance:
        raise RuntimeError("A mesh vertex lies off its classified B-Rep entity.")
    return vertices, counts


def certificate_summary(certificate):
    status = np.asarray(certificate.status)
    if not np.all(status == phx.discretization.CellValidityStatus.CERTIFIED_VALID):
        raise RuntimeError("A curved cell is not certified valid.")
    return {
        "certified_valid": certificate.certified_valid_count,
        "invalid": certificate.invalid_count,
        "unresolved": certificate.unresolved_count,
        "minimum_determinant_lower_bound": float(
            np.min(np.asarray(certificate.determinant_lower))
        ),
        "maximum_subdivision_depth": int(np.max(np.asarray(certificate.depth))),
    }


def curve(mesh, association, projection, degree, exact_volume):
    policy = phx.meshing.HighOrderCurvingPolicy(degree=degree)
    curved = phx.meshing.curve_cell_mesh(mesh, association, projection, policy=policy)
    if curved.status is not phx.meshing.HighOrderCurvingStatus.CURVED:
        raise RuntimeError(f"P{degree} curving ended with status {curved.status}.")
    certificate = phx.discretization.certify_cell_geometry_validity(
        curved.geometry, mesh=mesh
    )
    straight = phx.meshing.verify_curved_geometry(
        curved.straight, mesh, association, projection, policy=policy
    )
    if curved.evidence.maximum_residual > policy.residual_tolerance:
        raise RuntimeError(f"P{degree} geometry nodes lie off the B-Rep.")
    if straight.accepted:
        raise RuntimeError("Straight-sided nodes unexpectedly lie on the curved B-Rep.")
    straight_volume = mapped_volume(mesh, curved.straight)
    curved_volume = mapped_volume(mesh, curved.geometry)
    straight_error = abs(straight_volume - exact_volume) / exact_volume
    curved_error = abs(curved_volume - exact_volume) / exact_volume
    if not curved_error < 0.25 * straight_error:
        raise RuntimeError(f"P{degree} curving did not reduce the volume error.")
    return {
        "status": curved.status.value,
        "geometry_nodes": curved.geometry.coordinates.shape[0],
        "relaxation_rounds": len(curved.minimizations),
        "relaxation_statuses": [status.name for status in curved.relaxation_statuses],
        "accepted_round": curved.accepted_round,
        "certificate": certificate_summary(certificate),
        "node_residual": {
            "straight": straight.maximum_residual,
            "curved": curved.evidence.maximum_residual,
        },
        "minimum_scaled_jacobian": {
            "straight": straight.minimum_scaled_jacobian,
            "curved": curved.evidence.minimum_scaled_jacobian,
        },
        "volume": {
            "exact": exact_volume,
            "straight": straight_volume,
            "curved": curved_volume,
            "straight_relative_error": straight_error,
            "curved_relative_error": curved_error,
        },
    }


solids = (
    (
        "cylinder",
        BRepPrimAPI_MakeCylinder(RADIUS, HEIGHT).Shape(),
        np.pi * RADIUS**2 * HEIGHT,
        0.7,
    ),
    (
        "sphere",
        BRepPrimAPI_MakeSphere(RADIUS).Shape(),
        4.0 / 3.0 * np.pi * RADIUS**3,
        0.6,
    ),
)
summary = {}
with TemporaryDirectory(prefix="phydrax-cad-curving-") as temporary:
    for name, shape, exact_volume, size in solids:
        path = Path(temporary) / f"{name}.brep"
        phx.geometry.persist_occt_shape(shape, path, coordinate_contract=CONTRACT)
        model = phx.geometry.import_brep(
            path,
            coordinate_contract=CONTRACT,
            linear_deflection=0.05,
            angular_deflection=0.2,
        )
        source = phx.geometry.BRepSource(model)
        straight = mesh_solid(source, size, 1)
        projection = phx.geometry.prepare_brep_projection(model, path)
        association, counts = classify(straight.mesh, projection)
        summary[name] = {
            "source_revision": model.source_revision,
            "gmsh_version": straight.runtime.actual_version,
            "vertices": straight.mesh.coordinates.shape[0],
            "cells": straight.mesh.entity_set(3).count,
            "association": counts,
            "curving": {
                f"P{degree}": curve(
                    straight.mesh, association, projection, degree, exact_volume
                )
                for degree in (2, 3)
            },
        }
        if name == "sphere":
            second_order = mesh_solid(source, size, 2)
            evidence = phx.meshing.verify_curved_geometry(
                second_order.geometry,
                second_order.mesh,
                phx.meshing.associate_mesh_vertices(
                    second_order.mesh, projection, policy=ASSOCIATION
                ),
                projection,
                policy=phx.meshing.HighOrderCurvingPolicy(degree=2),
            )
            if not evidence.accepted:
                raise RuntimeError("The Gmsh second-order sphere failed verification.")
            volume = mapped_volume(second_order.mesh, second_order.geometry)
            summary[name]["gmsh_second_order"] = {
                "accepted": evidence.accepted,
                "node_residual": evidence.maximum_residual,
                "certificate": certificate_summary(evidence.certificate),
                "volume_relative_error": abs(volume - exact_volume) / exact_volume,
            }
print(json.dumps(summary, indent=2))
