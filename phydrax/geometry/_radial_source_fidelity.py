#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify continuous radial distance and full-sphere surjectivity.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "coordinate_source_expression"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each triangle coordinate owner supplies polynomial source coefficients, or vertex geometry supplies its affine interpolation."
      }
    ],
    "conclusion": "Unavailable coordinate polynomials leave radial fidelity unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._radial_source_fidelity._prepare_radial_cell",
      "locator": "phydrax.geometry._radial_source_fidelity._prepare_radial_cell",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "radial_source_chart_chain_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "A retained chart cover reports complete certified source-chart coverage."
      }
    ],
    "conclusion": "An incomplete or uncertified chart chain leaves the radial-source premise unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._domain_boundary.MeshingDomainBoundarySource.boundary_chart_cover",
      "locator": "phydrax.geometry._domain_boundary.MeshingDomainBoundarySource.boundary_chart_cover",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "radial_source_closed_chain_premise",
      "radial_source_trace_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every mesh facet has multiplicity two and coordinate polynomials agree on each paired facet."
      }
    ],
    "conclusion": "An unclosed incidence chain or mismatched trace leaves radial surjectivity unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._radial_source_fidelity._certify_radial_chain",
      "locator": "phydrax.geometry._radial_source_fidelity._certify_radial_chain",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "radial_source_homotopy_premise",
      "radial_source_degree_one_premise"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "The outward-rounded Bernstein lower bound of the coordinate/proxy radial dot product is positive on each reference triangle."
      },
      {
        "tag": "X",
        "statement": "The affine radial proxy chain has certified degree one about the full-sphere center."
      },
      {
        "tag": "T",
        "statement": "A nonvanishing homotopy preserves the degree of a map to the sphere."
      }
    ],
    "conclusion": "The coordinate map is radially surjective when both premises hold; an unproved premise leaves surjectivity unresolved.",
    "citation": {
      "kind": "mathematical",
      "reference": "Hatcher, Algebraic Topology (2002)",
      "locator": "Section 2.2, degree and homotopy invariance for maps of spheres",
      "url": "https://pi.math.cornell.edu/~hatcher/AT/AT.pdf"
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "boundary_deviation"
    ],
    "premises": [
      {
        "tag": "N",
        "statement": "Bernstein bounds enclose squared radius on every live piece and source frame bounds enclose the sphere radius; outward square-root rounding bounds directed radial distance."
      },
      {
        "tag": "X",
        "statement": "The barycentric-center polynomial evaluation is a source-point separation test."
      }
    ],
    "conclusion": "A point separation above tolerance violates fidelity; insufficient whole-piece distance bounds leave it unresolved.",
    "citation": {
      "kind": "mathematical",
      "reference": "Farin, Curves and Surfaces for CAGD, fifth edition (2002)",
      "locator": "Bernstein-B\u00e9zier representation and convex-hull property",
      "url": "https://www.sciencedirect.com/book/9781558607378/curves-and-surfaces-for-cagd"
    },
    "failure_statuses": [
      "violated",
      "unresolved"
    ]
  },
  {
    "checks": [
      "radial_source_bernstein_capacity",
      "radial_source_chain_capacity",
      "radial_source_piece_capacity",
      "radial_source_subdivision_depth",
      "radial_source_coefficient_work",
      "radial_source_polynomial_storage",
      "radial_source_retained_basis"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Polynomial node counts, chain size, subdivision work, depth, coefficient work and retained/live storage must fit declared request limits."
      }
    ],
    "conclusion": "Exhausted resource capacity leaves radial fidelity unresolved and preserves consumed work evidence.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._radial_source_fidelity.certify_native_radial_fidelity",
      "locator": "phydrax.geometry._radial_source_fidelity.certify_native_radial_fidelity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from jax import Array

import phydrax.geometry._domain_boundary as domain_boundary_owner
import phydrax.geometry._source_fidelity as source_fidelity_owner
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    reserve_polynomial,
)
from phydrax.geometry._embedding_cells import Cell, vertices as embedding_vertices
from phydrax.geometry._embedding_state import (
    EmbeddingState,
    mesh_facets,
    MeshCertificateFinding,
    MeshCertificateLimits,
)
from phydrax.geometry._rational import fraction_endpoint
from phydrax.geometry._source_boundary_records import SourceBoundaryQuery

from .._strict import StrictModule
from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import (
    CellGeometryElement,
    CellGeometrySpec,
    CellVertexGeometryElement,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._exact_algebraic_geometry import rational_coefficient
from ..typing import checked
from ._domain_numerics import full_sphere_frame, sphere_radial_degree_one
from ._mapped_embedding import trace_continuity
from ._mesh_certificates import MeshCertificateBinding


def _radial_bound(
    squared: algebra.Polynomial,
    frame: tuple[np.ndarray, float, float],
    tolerance: float,
    limits: MeshCertificateLimits,
    findings: list[MeshCertificateFinding],
    entity: int,
    progress: _RadialProgress,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    """Enclose the whole cell image with a worklist of live polynomials."""
    pending = [(squared, 0)]
    upper = 0.0
    while pending:
        if progress.pieces >= min(
            limits.maximum_subdivision_pieces, limits.maximum_distance_evaluations
        ):
            findings.append(
                MeshCertificateFinding(
                    "radial_source_piece_capacity", "unresolved", "cell", (entity,)
                )
            )
            return math.inf
        value, depth = pending.pop()
        progress.pieces += 1
        # Live polynomials retain their storage charge; composition and
        # control-net temporaries release theirs at the scope boundary.
        with budget.temporary_scope():
            live = (value, *(polynomial for polynomial, _ in pending))
            numerator_bits, denominator_bits = algebra.coefficient_profile(
                live, budget=budget
            )
            reserve_polynomial(
                0,
                sum(len(polynomial) for polynomial in live),
                2,
                max(numerator_bits, denominator_bits),
                budget=budget,
            )
            lower_squared, upper_squared = algebra.polynomial_bounds(
                value, "simplex", 2, budget=budget
            )
            lower = max(
                0.0, float(np.nextafter(math.sqrt(max(0.0, lower_squared)), -np.inf))
            )
            high = float(np.nextafter(math.sqrt(max(0.0, upper_squared)), np.inf))
            bound = float(
                np.nextafter(max(high - frame[1], frame[2] - lower, 0.0), np.inf)
            )
            witness = rational_coefficient(
                algebra.evaluate(value, (Fraction(1, 3), Fraction(1, 3)), budget=budget),
                role="radial_source_fidelity.witness",
            )
            witness_low = float(
                np.nextafter(
                    math.sqrt(
                        max(0.0, fraction_endpoint(witness, upper=False, saturate=True))
                    ),
                    -np.inf,
                )
            )
            witness_high = float(
                np.nextafter(
                    math.sqrt(
                        max(0.0, fraction_endpoint(witness, upper=True, saturate=True))
                    ),
                    np.inf,
                )
            )
            separation = float(
                np.nextafter(
                    max(frame[1] - witness_high, witness_low - frame[2], 0.0), -np.inf
                )
            )
            if separation > tolerance:
                findings.append(
                    MeshCertificateFinding(
                        "boundary_deviation", "violated", "cell", (entity,)
                    )
                )
                return math.inf
            if bound <= tolerance:
                upper = max(upper, bound)
                continue
            if depth >= limits.maximum_subdivision_depth:
                findings.append(
                    MeshCertificateFinding(
                        "radial_source_subdivision_depth", "unresolved", "cell", (entity,)
                    )
                )
                upper = max(upper, bound)
                continue
            if progress.pieces + len(pending) + 4 > limits.maximum_subdivision_pieces:
                findings.append(
                    MeshCertificateFinding(
                        "radial_source_piece_capacity", "unresolved", "cell", (entity,)
                    )
                )
                return math.inf
            for child in source_fidelity_owner.split_fidelity_polynomial(
                value, budget=budget
            ):
                pending.append((child, depth + 1))
    return upper


class _RadialCell(StrictModule):
    cell: Cell
    corners: np.ndarray
    bound: float


class _RadialChain(StrictModule):
    cells: list[Cell]
    corners: list[np.ndarray]
    bounds: list[float]


@dataclass(slots=True)
class _RadialProgress:
    pieces: int = 0


def _prepare_radial_cell(
    element: CellGeometryElement,
    local: algebra.CoordinateSourceBank,
    vertices: np.ndarray,
    entity: int,
    frame: tuple[np.ndarray, float, float],
    tolerance: float,
    limits: MeshCertificateLimits,
    findings: list[MeshCertificateFinding],
    progress: _RadialProgress,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _RadialCell | None:
    polynomial = algebra.coordinate_polynomials(element, local, budget=budget)
    match element:
        case CellVertexGeometryElement():
            polynomial = source_fidelity_owner.fidelity_affine(local)
        case _:
            pass
    if polynomial is None:
        findings.append(
            MeshCertificateFinding(
                "coordinate_source_expression",
                "unresolved",
                "cell",
                (entity,),
            )
        )
        return None
    degree = max(
        (sum(index) for value in polynomial for index in value),
        default=0,
    )
    if math.comb(2 * degree + 2, 2) > limits.maximum_bernstein_nodes:
        findings.append(
            MeshCertificateFinding(
                "radial_source_bernstein_capacity",
                "unresolved",
                "cell",
                (entity,),
            )
        )
        return None
    proxy = np.asarray(
        [
            algebra.rounded_point(point)
            for point in algebra.corner_images(polynomial, "triangle", budget=budget)
        ],
        dtype=np.float64,
    )
    affine = source_fidelity_owner.fidelity_affine(proxy)
    # Prove equality against the actual rounded proxy coefficients, not only
    # corner values or an affine-degree claim. Missing terms denote exact zero.
    budget.reserve(
        sum(
            len(value) + len(proxy_value)
            for value, proxy_value in zip(polynomial, affine, strict=True)
        )
    )
    same_affine = all(
        all(
            coefficient == proxy_value.get(index, 0)
            for index, coefficient in value.items()
        )
        and all(
            coefficient == value.get(index, 0)
            for index, coefficient in proxy_value.items()
        )
        for value, proxy_value in zip(polynomial, affine, strict=True)
    )
    relative = tuple(
        algebra.add(
            value,
            algebra.constant(-Fraction(float(center)), 2, budget=budget),
            budget=budget,
        )
        for value, center in zip(polynomial, frame[0], strict=True)
    )
    linear = (
        relative
        if same_affine
        else tuple(
            algebra.add(
                value,
                algebra.constant(-Fraction(float(center)), 2, budget=budget),
                budget=budget,
            )
            for value, center in zip(affine, frame[0], strict=True)
        )
    )
    homotopy = algebra.sum_polynomials(
        tuple(
            algebra.multiply(a, b, budget=budget)
            for a, b in zip(relative, linear, strict=True)
        ),
        budget=budget,
    )
    if algebra.polynomial_bounds(homotopy, "simplex", 2, budget=budget)[0] <= 0.0:
        findings.append(
            MeshCertificateFinding(
                "radial_source_homotopy_premise",
                "unresolved",
                "cell",
                (entity,),
            )
        )
        return None
    # Identical maps in this chart and frame have the same exact dot product.
    # Reuse only the polynomial: positivity above and radial bounds below
    # remain independent checks with their own actual work/storage charges.
    squared = (
        homotopy
        if same_affine
        else algebra.sum_polynomials(
            tuple(algebra.multiply(value, value, budget=budget) for value in relative),
            budget=budget,
        )
    )
    bound = _radial_bound(
        squared, frame, tolerance, limits, findings, entity, progress, budget=budget
    )
    budget.retain_basis(polynomial)
    return _RadialCell(
        Cell(
            polynomial,
            "simplex",
            "triangle",
            2,
            tuple(int(v) for v in vertices),
            embedding_vertices("triangle"),
        ),
        proxy,
        bound,
    )


def _prepare_radial_chain(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    elements: tuple[CellGeometryElement, ...],
    routes: tuple[Array, ...],
    frame: tuple[np.ndarray, float, float],
    tolerance: float,
    limits: MeshCertificateLimits,
    findings: list[MeshCertificateFinding],
    progress: _RadialProgress,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _RadialChain:
    cells: list[Cell] = []
    corners: list[np.ndarray] = []
    bounds: list[float] = []
    values = algebra.prepared_coordinate_source_bank(geometry, budget=budget)
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        for row, (vertices, dofs) in enumerate(
            zip(np.asarray(block.vertices), np.asarray(route), strict=True)
        ):
            entity = int(np.asarray(block.global_ids)[row])
            with budget.temporary_scope():
                local = tuple(values[int(dof)] for dof in dofs)
                prepared = _prepare_radial_cell(
                    element,
                    local,
                    vertices,
                    entity,
                    frame,
                    tolerance,
                    limits,
                    findings,
                    progress,
                    budget=budget,
                )
                if prepared is None:
                    break
                cells.append(prepared.cell)
                corners.append(prepared.corners)
                bounds.append(prepared.bound)
        if findings:
            break
    return _RadialChain(cells, corners, bounds)


def _certify_radial_chain(
    mesh: CellMesh,
    chain: _RadialChain,
    frame: tuple[np.ndarray, float, float],
    count: int,
    findings: list[MeshCertificateFinding],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> float:
    with budget.temporary_scope():
        trace = EmbeddingState([], [])
        trace_continuity(trace, mesh, chain.cells, budget=budget)
    if trace.findings:
        findings.extend(
            MeshCertificateFinding(
                "radial_source_trace_premise",
                "unresolved",
                finding.entity_kind,
                finding.entity_ids,
            )
            for finding in trace.findings
        )
        return math.inf
    triangles = np.asarray(chain.corners, dtype=np.float64)
    budget.reserve(5 * count, 256 * triangles.size)
    points, inverse = np.unique(triangles.reshape((-1, 3)), axis=0, return_inverse=True)
    if not sphere_radial_degree_one(frame, points, inverse.reshape((-1, 3))):
        findings.append(
            MeshCertificateFinding(
                "radial_source_degree_one_premise", "unresolved", "mesh"
            )
        )
        return math.inf
    return max(chain.bounds, default=math.inf)


@checked
def certify_native_radial_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    binding: MeshCertificateBinding,
    tolerance: float,
    order: int,
    limits: MeshCertificateLimits,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> source_fidelity_owner.SourceFidelityCertificate | None:
    """Certify radial distance and degree-one coverage of full-sphere maps."""
    match source:
        case domain_boundary_owner.MeshingDomainBoundarySource():
            pass
        case _:
            return None
    if (
        len(source.patches) != 1
        or mesh.topological_dimension != 2
        or mesh.ambient_dimension != 3
        or any(block.cell_kind != "triangle" for block in mesh.blocks)
    ):
        return None
    frame = full_sphere_frame(source.domain, source.patches[0], budget=budget)
    if frame is None:
        return None
    count = sum(block.vertices.shape[0] for block in mesh.blocks)
    findings: list[MeshCertificateFinding] = []
    cover = None
    if source.chart_triangulations:
        cover = source.boundary_chart_cover(limits.maximum_source_samples, budget=budget)
        findings.extend(cover.findings)
        if not cover.complete or cover.semantics != "certified":
            findings.append(
                MeshCertificateFinding(
                    "radial_source_chart_chain_premise", "unresolved", "mesh"
                )
            )
    progress = _RadialProgress()
    bound = math.inf
    capacity = min(limits.maximum_source_samples, limits.maximum_subdivision_pieces)
    if count > capacity or count > limits.maximum_ray_tests:
        findings.append(
            MeshCertificateFinding("radial_source_chain_capacity", "unresolved", "mesh")
        )
    else:
        facets = mesh_facets(mesh, budget=budget)
        if np.any(facets.counts != 2):
            findings.append(
                MeshCertificateFinding(
                    "radial_source_closed_chain_premise", "unresolved", "mesh"
                )
            )

    if not findings:
        elements, routes, _ = geometry.resolve(mesh)
        try:
            with budget.measurement(), budget.activate():
                chain = _prepare_radial_chain(
                    mesh,
                    geometry,
                    elements,
                    routes,
                    frame,
                    tolerance,
                    limits,
                    findings,
                    progress,
                    budget=budget,
                )
                if not findings:
                    bound = _certify_radial_chain(
                        mesh, chain, frame, count, findings, budget=budget
                    )
        except CoordinateEnclosureResourceError as error:
            findings.append(
                MeshCertificateFinding(
                    f"radial_source_{error.resource}", "unresolved", "mesh"
                )
            )
    if bound > tolerance and not findings:
        findings.append(
            MeshCertificateFinding("boundary_deviation", "unresolved", "mesh")
        )
    return source_fidelity_owner.SourceFidelityCertificate(
        binding,
        tuple(findings),
        tolerance=tolerance,
        bounds=source_fidelity_owner.SourceFidelityBounds(
            semantics=("certified", "certified"),
            mesh_to_source=(bound, 0.0),
            source_to_mesh=(bound, 0.0),
            sample_counts=(progress.pieces, count if not findings else 0),
        ),
        sample_order=order,
        coverage=source_fidelity_owner.SourceFidelityCoverage(chart=cover),
    )
