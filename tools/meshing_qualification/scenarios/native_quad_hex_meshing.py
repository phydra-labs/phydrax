#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Generate native quadrilateral/hexahedral meshes and solve Q1/Q2 diffusion.

The scenarios mesh a holed plate, a two-material cavity, a curved polynomial
solid, and a rational quarter-arc extrusion. The PL cases use dual routes;
the curved solid uses balanced and frame integer-grid routes. The rational
surface uses the parametric-source route and retains its source geometry.

The curved solid requests a HARD target size of 0.5, maximum size of 0.6,
and p50/p95 relative tolerance of 0.5 with zero absolute tolerance.
``generate_curved_hexes`` accepts a ``SizeCompliancePolicy`` to choose the
size-compliance tolerances.

Run from the repository root with double precision enabled::

    python -m tools.meshing_qualification --scenario native-quad-hex --corpus-profile quad_plate

The JSON report includes cell counts, geometry evidence, and diffusion errors.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

import numpy as np

import phydrax as phx
from examples.native_tetrahedral_meshing import complex_with_cavity
from phydrax.linalg import LinearSolvePolicy
from phydrax.meshing import CellMeshingResult, SizeCompliancePolicy
from phydrax.meshing.qualification import (
    check_stationary_euler,
    curved_occupied_source,
    generate_source_quads,
    solve_linear_diffusion,
)
from tools.meshing_qualification._controls import QualificationControls


class QuadHexProfileResult(TypedDict):
    family: str
    cells: int
    audit_passed: bool
    compliance_passed: bool
    certification_passed: bool
    manufactured_physical_solution_error: float
    stationary_fv_euler_content_rate: NotRequired[float]


class QuadHexQualificationResult(TypedDict):
    scenario: Literal["native-quad-hex"]
    status: Literal["passed"]
    selected_profiles: tuple[str, ...]
    profiles: dict[str, QuadHexProfileResult]


M = phx.meshing


def generate_quads() -> CellMeshingResult:
    region = phx.geometry.PlanarMeshRegion(
        np.asarray(
            (
                (0.0, 0.0),
                (2.0, 0.0),
                (2.0, 2.0),
                (0.0, 2.0),
                (0.8, 0.8),
                (0.8, 1.2),
                (1.2, 1.2),
                (1.2, 0.8),
            ),
            dtype=np.float64,
        ),
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        feature_id="quad-plate",
    )
    source = M.NativePlanarSource(region, "plate-revision")
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "plate-region",
        np.asarray((0,), dtype=np.int64),
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("quadrilateral",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(
            M.UniformSizeControl(scope, 0.4, strength=M.SizeControlStrength.SOFT),
        ),
    )
    return (
        M.NativeMeshingProvider(M.NativeMeshingOptions("planar_dual_quad"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )


def generate_hexes() -> CellMeshingResult:
    source = M.NativePlcSource(
        complex_with_cavity(), "material-cavity", "cavity-revision"
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "material-cavity-facets",
        np.arange(source.complex.facet_count, dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(3, 3, M.CellFamilyPolicy(required=("hexahedron",))),
        scope,
        M.VolumeFillStrategy.MULTIZONE,
        size_controls=(
            M.UniformSizeControl(scope, 0.4, strength=M.SizeControlStrength.SOFT),
        ),
    )
    return (
        M.NativeMeshingProvider(M.NativeMeshingOptions("plc_dual_hex"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )


def generate_curved_hexes(
    route: Literal["mapped_balanced_grid_hex", "mapped_frame_grid_hex"],
    /,
    *,
    size_compliance: SizeCompliancePolicy | None = None,
) -> CellMeshingResult:
    """Declare a curved polynomial source and a fixed sizing policy."""
    source = curved_occupied_source({(0, 0, 0)}, curvature=0.125)
    domain, reference_mesh = source.domain, source.domain.reference_mesh
    entities = reference_mesh.topology.entities(2)
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        entities.entity_ids,
    )
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("hexahedron",)), geometry_order=2
        ),
        scope,
        M.VolumeFillStrategy.MULTIZONE,
        size_controls=(
            M.UniformSizeControl(
                scope, 0.5, maximum_size=0.6, strength=M.SizeControlStrength.HARD
            ),
        ),
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5)
        if size_compliance is None
        else size_compliance,
    )
    return (
        M.NativeMeshingProvider(M.NativeMeshingOptions(route))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )


def run(args: QualificationControls) -> QuadHexQualificationResult:
    """Qualify selected quad/hex profiles; omitted selection runs every profile."""
    report: dict[str, QuadHexProfileResult] = {}
    cases = (
        ("quad_plate", generate_quads, "quadrilateral", 1, None, None, None),
        (
            "original_rational_surface_quads",
            generate_source_quads,
            "quadrilateral",
            2,
            (1.0, 0.0, 0.0),
            LinearSolvePolicy(
                phx.linalg.DenseLU(),
                tolerance=phx.linalg.TolerancePolicy(relative=1.0e-12, absolute=1.0e-13),
            ),
            8,
        ),
        ("hex_material_cavity", generate_hexes, "hexahedron", 1, None, None, None),
        (
            "balanced_curved_hex",
            lambda: generate_curved_hexes("mapped_balanced_grid_hex"),
            "hexahedron",
            2,
            None,
            None,
            None,
        ),
        (
            "frame_curved_hex",
            lambda: generate_curved_hexes("mapped_frame_grid_hex"),
            "hexahedron",
            2,
            None,
            None,
            None,
        ),
    )
    names = tuple(case[0] for case in cases)
    selected = names if args.corpus_profile is None else args.corpus_profile
    if not selected:
        raise ValueError(
            "quad_hex.corpus_profile must select at least one profile; got ()"
        )
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise ValueError(f"quad_hex.corpus_profile must select {names}; got {unknown}")
    for name, generate, expected_family, degree, direction, policy, order in cases:
        if name not in selected:
            continue
        result = generate()
        mesh = result.mesh
        certification = result.certification
        if certification is None:
            raise RuntimeError(
                f"meshing.certification must be present; got case={name!r}, certification=None"
            )
        if not (
            result.audit.passed and result.compliance.passed and certification.passed
        ):
            raise RuntimeError(
                "quad_hex.publication must pass audit, compliance and certification; "
                f"got case={name!r}, audit={result.audit.passed}, "
                f"compliance={result.compliance.passed}, "
                f"certification={certification.passed}"
            )
        if {block.cell_kind for block in mesh.blocks} != {expected_family}:
            raise RuntimeError(
                f"meshing.cell_kinds must equal {{{expected_family!r}}}; "
                f"got {tuple(block.cell_kind for block in mesh.blocks)!r}"
            )
        report[name] = {
            "family": mesh.blocks[0].cell_kind,
            "cells": sum(block.cell_count for block in mesh.blocks),
            "audit_passed": result.audit.passed,
            "compliance_passed": result.compliance.passed,
            "certification_passed": certification.passed,
            "manufactured_physical_solution_error": solve_linear_diffusion(
                result,
                degree,
                harmonic_direction=direction,
                solve_policy=policy,
                cell_quadrature_order=order,
            ),
        }
        if mesh.topological_dimension == 3:
            report[name]["stationary_fv_euler_content_rate"] = check_stationary_euler(
                result
            )
    return {
        "scenario": "native-quad-hex",
        "status": "passed",
        "selected_profiles": tuple(selected),
        "profiles": report,
    }
