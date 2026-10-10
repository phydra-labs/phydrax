#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Continuous source distance and surjectivity of native spherical coordinate maps.

A complete native spherical carrier has a source-established radial shell. Exact
coordinate polynomials bound the entire mesh image, not just its geometry nodes.
An exact shared-trace chain and a positive-dot homotopy to an independently
checked degree-one affine chain establish that every source ray is attained.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from ..discretization import _coordinate_enclosure as algebra
from ..discretization._cell_geometry import CellGeometrySpec, CellVertexGeometryElement
from ..discretization._cell_mesh import CellMesh
from ._mapped_embedding import _Cell, _trace_continuity, _vertices
from ._mesh_certificates import (
    _EmbeddingState,
    _fidelity_affine,
    _FidelityMap,
    _mesh_facets,
    MeshCertificateBinding,
    MeshCertificateFinding,
    MeshCertificateLimits,
    SourceBoundaryQuery,
    SourceFidelityCertificate,
)
from ._meshing_domain import (
    _full_sphere_frame,
    _sphere_radial_degree_one,
    MeshingDomainBoundarySource,
)


def _radial_bound(
    squared: algebra.Polynomial,
    frame: tuple[np.ndarray, float, float],
    tolerance: float,
    limits: MeshCertificateLimits,
    findings: list[MeshCertificateFinding],
    entity: int,
    pieces: list[int],
    budget: algebra.CoordinateEnclosureBudget,
    /,
) -> float:
    """Bound one complete image with a worklist of explicitly live polynomials."""
    pending = [(squared, 0)]
    upper = 0.0
    while pending:
        if pieces[0] >= min(
            limits.maximum_subdivision_pieces, limits.maximum_distance_evaluations
        ):
            findings.append(
                MeshCertificateFinding(
                    "radial_source_piece_capacity", "unresolved", "cell", (entity,)
                )
            )
            return math.inf
        value, depth = pending.pop()
        pieces[0] += 1
        # Released composition/control-net temporaries must not be counted as
        # resident forever. The pending canonical polynomials remain charged.
        with budget.temporary_scope():
            live = (value, *(polynomial for polynomial, _ in pending))
            numerator_bits, denominator_bits = algebra._coefficient_profile(live)
            algebra._reserve_polynomial(
                0,
                sum(len(polynomial) for polynomial in live),
                2,
                max(numerator_bits, denominator_bits),
            )
            lower_squared, upper_squared = algebra.polynomial_bounds(value, "simplex", 2)
            lower = max(
                0.0, float(np.nextafter(math.sqrt(max(0.0, lower_squared)), -np.inf))
            )
            high = float(np.nextafter(math.sqrt(max(0.0, upper_squared)), np.inf))
            bound = float(
                np.nextafter(max(high - frame[1], frame[2] - lower, 0.0), np.inf)
            )
            witness = algebra.evaluate(value, (Fraction(1, 3), Fraction(1, 3)))
            witness_low = float(
                np.nextafter(
                    math.sqrt(max(0.0, algebra.outward(witness, -math.inf))), -np.inf
                )
            )
            witness_high = float(
                np.nextafter(
                    math.sqrt(max(0.0, algebra.outward(witness, math.inf))), np.inf
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
            if pieces[0] + len(pending) + 4 > limits.maximum_subdivision_pieces:
                findings.append(
                    MeshCertificateFinding(
                        "radial_source_piece_capacity", "unresolved", "cell", (entity,)
                    )
                )
                return math.inf
            for child in _children(value):
                pending.append((child, depth + 1))
    return upper


def _children(value: algebra.Polynomial, /) -> tuple[algebra.Polynomial, ...]:
    from ._mesh_certificates import _split_fidelity_map

    children = _split_fidelity_map(_FidelityMap((value,), 2, 0))
    polynomials: list[algebra.Polynomial] = []
    for child in children:
        coordinate = child.coordinates[0]
        if isinstance(coordinate, algebra.RationalPolynomial):
            raise RuntimeError(
                "A polynomial radial restriction acquired a rational denominator."
            )
        polynomials.append(coordinate)
    return tuple(polynomials)


def certify_native_radial_fidelity(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    source: SourceBoundaryQuery,
    binding: MeshCertificateBinding,
    tolerance: float,
    order: int,
    limits: MeshCertificateLimits,
    /,
) -> SourceFidelityCertificate | None:
    """Prove actual native full-sphere images, leaving other carriers to their owner."""
    if (
        not isinstance(source, MeshingDomainBoundarySource)
        or len(source.patches) != 1
        or mesh.topological_dimension != 2
        or mesh.ambient_dimension != 3
        or any(block.cell_kind != "triangle" for block in mesh.blocks)
    ):
        return None
    frame = _full_sphere_frame(source.domain, source.patches[0])
    if frame is None:
        return None
    count = sum(block.vertices.shape[0] for block in mesh.blocks)
    findings: list[MeshCertificateFinding] = []
    cover = None
    if source.chart_triangulations:
        cover = source.boundary_chart_cover(limits.maximum_source_samples)
        findings.extend(cover.findings)
        if not cover.complete or cover.semantics != "certified":
            findings.append(
                MeshCertificateFinding(
                    "radial_source_chart_chain_premise", "unresolved", "mesh"
                )
            )
    pieces = [0]
    bound = math.inf
    capacity = min(limits.maximum_source_samples, limits.maximum_subdivision_pieces)
    if count > capacity or count > limits.maximum_ray_tests:
        findings.append(
            MeshCertificateFinding("radial_source_chain_capacity", "unresolved", "mesh")
        )
    else:
        facets = _mesh_facets(mesh)
        if np.any(facets.counts != 2):
            findings.append(
                MeshCertificateFinding(
                    "radial_source_closed_chain_premise", "unresolved", "mesh"
                )
            )
    budget = algebra.CoordinateEnclosureBudget(
        limits.maximum_work_units, limits.maximum_scratch_bytes
    )
    if not findings:
        cells: list[_Cell] = []
        corners: list[np.ndarray] = []
        bounds: list[float] = []
        elements, routes, _ = geometry.resolve(mesh)
        try:
            with budget.activate():
                values = geometry.source_coordinates()
                for block, element, route in zip(
                    mesh.blocks, elements, routes, strict=True
                ):
                    for row, (vertices, dofs) in enumerate(
                        zip(np.asarray(block.vertices), np.asarray(route), strict=True)
                    ):
                        entity = int(np.asarray(block.global_ids)[row])
                        with budget.temporary_scope():
                            local = tuple(values[int(dof)] for dof in dofs)
                            polynomial = algebra.coordinate_polynomials(element, local)
                            if isinstance(element, CellVertexGeometryElement):
                                polynomial = _fidelity_affine(local)
                            if polynomial is None:
                                findings.append(
                                    MeshCertificateFinding(
                                        "coordinate_source_expression",
                                        "unresolved",
                                        "cell",
                                        (entity,),
                                    )
                                )
                                break
                            degree = max(
                                (sum(index) for value in polynomial for index in value),
                                default=0,
                            )
                            if (
                                math.comb(2 * degree + 2, 2)
                                > limits.maximum_bernstein_nodes
                            ):
                                findings.append(
                                    MeshCertificateFinding(
                                        "radial_source_bernstein_capacity",
                                        "unresolved",
                                        "cell",
                                        (entity,),
                                    )
                                )
                                break
                            proxy = np.asarray(
                                [
                                    algebra.rounded_point(point)
                                    for point in algebra.corner_images(
                                        polynomial, "triangle"
                                    )
                                ],
                                dtype=np.float64,
                            )
                            affine = _fidelity_affine(proxy)
                            relative = tuple(
                                algebra.add(
                                    value, algebra.constant(-Fraction(float(center)), 2)
                                )
                                for value, center in zip(
                                    polynomial, frame[0], strict=True
                                )
                            )
                            linear = tuple(
                                algebra.add(
                                    value, algebra.constant(-Fraction(float(center)), 2)
                                )
                                for value, center in zip(affine, frame[0], strict=True)
                            )
                            homotopy = algebra.sum_polynomials(
                                tuple(
                                    algebra.multiply(a, b)
                                    for a, b in zip(relative, linear, strict=True)
                                )
                            )
                            if (
                                algebra.polynomial_bounds(homotopy, "simplex", 2)[0]
                                <= 0.0
                            ):
                                findings.append(
                                    MeshCertificateFinding(
                                        "radial_source_homotopy_premise",
                                        "unresolved",
                                        "cell",
                                        (entity,),
                                    )
                                )
                                break
                            squared = algebra.sum_polynomials(
                                tuple(
                                    algebra.multiply(value, value) for value in relative
                                )
                            )
                            bounds.append(
                                _radial_bound(
                                    squared,
                                    frame,
                                    tolerance,
                                    limits,
                                    findings,
                                    entity,
                                    pieces,
                                    budget,
                                )
                            )
                            budget.retain_basis(polynomial)
                            cells.append(
                                _Cell(
                                    polynomial,
                                    "simplex",
                                    "triangle",
                                    2,
                                    tuple(int(v) for v in vertices),
                                    _vertices("triangle"),
                                )
                            )
                            corners.append(proxy)
                    if findings:
                        break
                if not findings:
                    with budget.temporary_scope():
                        trace = _EmbeddingState([], [])
                        _trace_continuity(trace, mesh, cells)
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
                    else:
                        triangles = np.asarray(corners, dtype=np.float64)
                        budget.reserve(5 * count, 256 * triangles.size)
                        points, inverse = np.unique(
                            triangles.reshape((-1, 3)), axis=0, return_inverse=True
                        )
                        if not _sphere_radial_degree_one(
                            frame, points, inverse.reshape((-1, 3))
                        ):
                            findings.append(
                                MeshCertificateFinding(
                                    "radial_source_degree_one_premise",
                                    "unresolved",
                                    "mesh",
                                )
                            )
                        else:
                            bound = max(bounds, default=math.inf)
        except algebra.CoordinateEnclosureResourceError as error:
            findings.append(
                MeshCertificateFinding(
                    f"radial_source_{error.resource}", "unresolved", "mesh"
                )
            )
    if bound > tolerance and not findings:
        findings.append(
            MeshCertificateFinding("boundary_deviation", "unresolved", "mesh")
        )
    return SourceFidelityCertificate(
        binding,
        tuple(findings),
        tolerance=tolerance,
        semantics=("certified", "certified"),
        mesh_to_source=(bound, 0.0),
        source_to_mesh=(bound, 0.0),
        sample_order=order,
        sample_counts=(pieces[0], count if not findings else 0),
        chart_coverage=cover,
    )
