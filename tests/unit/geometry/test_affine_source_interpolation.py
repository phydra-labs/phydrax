"""Exact affine interpolation residuals and the general polynomial route."""

import math
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import Polynomial
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.discretization._surface_chart_records import SurfaceChartWitness
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.geometry._source_fidelity import certify_source_fidelity
from phydrax.geometry._source_interpolation import (
    _exact_surface_interpolation_bound,
    _exact_surface_map_is_rational_affine,
    certify_domain_surface_charts,
)
from phydrax.geometry.brep._patches import LineCurve, RuledSurface


@pytest.mark.parametrize(
    "chart",
    [
        (
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
        (
            (Fraction(1, 3), Fraction(-2, 7)),
            (Fraction(8, 9), Fraction(1, 5)),
            (Fraction(-7, 11), Fraction(5, 13)),
        ),
        ((Fraction(1, 3), Fraction(1, 5)),) * 3,
    ],
)
def test_rational_affine_residual_is_positive_zero(
    chart: tuple[tuple[Fraction, Fraction], ...],
) -> None:
    expressions: tuple[Polynomial, ...] = (
        {(0, 0): Fraction(1, 3), (1, 0): Fraction(7, 11)},
        {(0, 1): Fraction(-2, 9)},
        {(0, 0): Fraction(4, 7), (2, 0): Fraction(0)},
    )
    assert _exact_surface_map_is_rational_affine(expressions)
    bound = _exact_surface_interpolation_bound(expressions, chart)
    assert bound == 0.0
    assert math.copysign(1.0, bound) == 1.0


def test_non_affine_and_radical_maps_keep_general_route() -> None:
    chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    quadratic: tuple[Polynomial, ...] = ({(2, 0): Fraction(1)}, {}, {})
    radical: tuple[Polynomial, ...] = ({(1, 0): RadicalElement(1)}, {}, {})
    assert not _exact_surface_map_is_rational_affine(quadratic)
    assert _exact_surface_interpolation_bound(quadratic, chart) == 0.25
    assert not _exact_surface_map_is_rational_affine(radical)
    assert _exact_surface_interpolation_bound(radical, chart) == 0.0


def _consumer_case(
    height: float,
) -> tuple[
    CellMesh,
    CellGeometrySpec,
    MeshingDomainBoundarySource,
    SurfaceChartWitness,
    SourceBoundaryChartWitness,
]:
    surface = RuledSurface(
        LineCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        LineCurve((0.0, 1.0, 0.0), (1.0, 0.0, height)),
    )
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    corners = np.asarray(surface.evaluate(jnp.asarray(uv)), dtype=np.float64)
    uses = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, LineCurve((1.0, 0.0), (-1.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((0.0, 0.0), (0.0, 1.0)), 0.0, 1.0, sense=-1),
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(surface, (uses,)),),
        (MeshingDomainCurve(0, 1), MeshingDomainCurve(1, 2), MeshingDomainCurve(0, 2)),
        3,
        source_id=f"affine-consumer-{height}",
        source_revision="affine-consumer-revision",
        corner_points=corners,
    )
    mesh = CellMesh(
        corners,
        (CellBlock("triangles", "triangle", np.asarray(((0, 1, 2),), dtype=np.int64)),),
        vertex_global_ids=np.arange(3, dtype=np.int64),
    )
    geometry = CellGeometrySpec.affine(mesh)
    source = MeshingDomainBoundarySource(domain, (0,))
    entity = domain.entity_id(2, 0)
    occurrence = domain.source_occurrences[2][0]
    scope_by_key = {
        domain.source_entity_key_for_scope(2, int(entity_id)): int(entity_id)
        for entity_id in domain.scope_entity_ids(2)
    }
    ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
    witness = SurfaceChartWitness(
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        domain_id=domain.domain_id,
        cell_global_ids=ids,
        patches=np.zeros(ids.shape, dtype=np.int64),
        charts=uv[None, ...],
        source_root_cell_ids=np.full(
            ids.shape, scope_by_key[(entity, occurrence)], dtype=np.int64
        ),
        source_piece_ids=((0, int(domain.source_indices[2][0])),),
        geometry_entity_ids=(entity,),
        occurrence_paths=(occurrence,),
        root_mesh_cell_bindings=(None,),
    )
    chart_witness = SourceBoundaryChartWitness(
        mesh_id=mesh.mesh_id,
        geometry_id=cell_geometry_id(geometry),
        topology_id=mesh.topology_id,
        source_id=source.source_id,
        source_revision=source.source_revision,
        source_query_id=source.source_scope_id,
        chart_dimension=2,
        reference_dimension=2,
        cell_global_ids=ids,
        entity_global_ids=ids,
        local_entity_indices=np.full(ids.shape, -1, dtype=np.int32),
        target_reference_vertices=uv[None, ...],
        source_chart_ids=(entity,),
        source_entity_ids=(entity,),
        occurrence_paths=(occurrence,),
        source_parameters=uv[None, ...],
        orientations=np.ones(ids.shape, dtype=np.int8),
    )
    return mesh, geometry, source, witness, chart_witness


@pytest.mark.parametrize(
    ("height", "certificate_upper"),
    [(0.0, "0x0.0p+0"), (2.0**-20, "0x1.0000000000000p-22")],
)
def test_affine_and_bilinear_certificate(height: float, certificate_upper: str) -> None:
    mesh, geometry, source, witness, _ = _consumer_case(height)
    certificate = certify_source_fidelity(
        mesh, geometry, source, tolerance=0.5, witness=witness
    )
    assert certificate.status == "certified"
    assert certificate.findings == ()
    assert certificate.deviation_bounds.dtype == np.float64
    assert float(certificate.deviation_bounds[0, 0]).hex() == "0x0.0p+0"
    assert float(certificate.deviation_bounds[0, 1]).hex() == certificate_upper
    assert certificate.mesh_to_source_upper.hex() == certificate_upper
    assert certificate.source_to_mesh_upper.hex() == certificate_upper


@pytest.mark.parametrize(
    ("height", "chart_upper"),
    [
        (0.0, "0x1.3988e14092130p-536"),
        (2.0**-20, "0x1.0000000000001p-22"),
    ],
)
def test_affine_and_bilinear_chart_bounds(height: float, chart_upper: str) -> None:
    mesh, geometry, source, _, chart_witness = _consumer_case(height)

    chart = certify_domain_surface_charts(
        mesh,
        geometry,
        source,
        chart_witness,
        np.zeros(1, dtype=np.int64),
        ((0, 0),),
        tolerance=0.5,
    )
    assert chart.findings == ()
    assert chart.deviation_bounds.dtype == np.float64
    assert float(chart.deviation_bounds[0, 0]).hex() == "0x0.0p+0"
    assert float(chart.deviation_bounds[0, 1]).hex() == chart_upper
    assert chart.mesh_to_source_upper.hex() == chart_upper
    assert chart.source_to_mesh_upper.hex() == chart_upper
