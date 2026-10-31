from collections.abc import Mapping
from fractions import Fraction
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._polynomial._cubature import cubature_rule_data
from phydrax.discretization._cell_geometry import (
    CellGeometrySpec,
    PolynomialComposedCellGeometryElement,
    RationalComposedCellGeometryElement,
)
from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization._coordinate_enclosure import (
    compose_expression,
    Expression,
    RationalPolynomial,
)
from phydrax.discretization.fem._reference import FiniteElementSpec, lagrange_element
from phydrax.geometry._common_root_volume_overlap import (
    _adaptive_expression_integral,
    _as_chart_arguments,
    _clip_exact_tetrahedra,
    _parametric_inverse_inclusion,
    CommonRootOverlapError,
    CurvedOverlapIntegrationPolicy,
    MaterialReferenceVolumeWitness,
    PhysicalOverlapTransfer,
    prepare_common_root_volume_refinement,
    prepare_material_reference_volume_overlap,
)
from phydrax.geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
)
from phydrax.geometry._volume_material_map import (
    CommonRootVolumeWitness,
    declare_material_reference,
    EulerianVolumeOverlapChain,
    EulerianVolumeOverlapMap,
    MaterialReferencePolicy,
    MaterialSourceMapCapability,
    prepare_eulerian_volume_overlap,
    prepare_volume_material_map,
    require_polyhedral_carrier_geometry,
    VolumeMaterialDeformation,
    VolumeMaterialMapError,
)
from tests._support.volume_material_map_fixtures import (
    affine_reference_chart,
    make_curved_coarsening_witness,
    make_curved_material_refinement,
    make_material_coverage_pair,
    make_material_source,
    reference_map_witness,
    ROOT_REFERENCE_TETRAHEDRON,
)


ReferencePoint = tuple[Fraction, Fraction, Fraction]
ReferenceTet = tuple[ReferencePoint, ReferencePoint, ReferencePoint, ReferencePoint]
RationalTerms = dict[tuple[int, int, int], Fraction]
ReferenceChart = tuple[Expression, ...]
ReferenceChartBank = tuple[ReferenceChart, ...]


def test_polyhedral_overlap_refuses_nonowning_coordinate_elements() -> None:
    class GeneralCellGeometryElement:
        cell_kind = "tetrahedron"
        conformity = "H1"
        local_dof_count = 4
        element_id = "external-coordinate-map"

    vertices = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = CellMesh(
        vertices,
        (
            CellBlock(
                "tetrahedron",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            ),
        ),
    )
    geometry = CellGeometrySpec(
        {"tetrahedron": GeneralCellGeometryElement()},
        {"tetrahedron": np.asarray(((0, 1, 2, 3),), dtype=np.int32)},
        vertices,
    )
    with pytest.raises(VolumeMaterialMapError, match="owning scalar coordinate map"):
        require_polyhedral_carrier_geometry(mesh, geometry)


def test_eulerian_curved_common_root_refinement_coarsening_and_field_adjoints() -> None:
    root_mesh, root_geometry, _, target_mesh, target_geometry, target_map, witness = (
        make_curved_material_refinement(exact_chart_arguments=True)
    )
    assert witness is not None
    domain, source_coverage, target_coverage = make_material_coverage_pair(
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        region_id="aluminum",
        occurrence_path=("assembly-1", "occurrence-9"),
    )
    source_map = make_material_source(domain)
    refined = prepare_eulerian_volume_overlap(
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        source_map,
        ("solid-cell",),
        ("solid-cell", "solid-cell"),
        ("aluminum",),
        ("aluminum", "aluminum"),
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        common_root=witness,
    )
    assert refined.refinement.succeeded
    assert refined.integration_evidence is not None
    assert refined.integration_evidence.exact_polynomial_integral_count > 0
    np.testing.assert_allclose(
        np.sort(np.asarray(refined.refinement.volumes)),
        (1.0 / 12.0, 1.0 / 12.0),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(refined.refinement.source_measures),
        (1.0 / 6.0,),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(refined.refinement.target_measures),
        (1.0 / 12.0, 1.0 / 12.0),
        rtol=1e-12,
        atol=1e-14,
    )

    extensive = jnp.asarray((6.0,))
    np.testing.assert_allclose(refined.apply_extensive(extensive), (3.0, 3.0), atol=1e-13)
    np.testing.assert_allclose(
        refined.apply_intensive(jnp.asarray((2.5,))), (2.5, 2.5), atol=1e-13
    )
    _, tangent = jax.jvp(
        refined.apply_extensive, (extensive,), (jnp.ones_like(extensive),)
    )
    np.testing.assert_allclose(tangent, (0.5, 0.5), atol=1e-13)
    cotangent = jnp.asarray((1.0, 3.0))
    np.testing.assert_allclose(
        refined.apply_extensive_transpose(cotangent), (2.0,), atol=1e-13
    )
    np.testing.assert_allclose(
        jnp.vdot(refined.apply_intensive(jnp.asarray((1.25,))), cotangent),
        jnp.vdot(jnp.asarray((1.25,)), refined.apply_intensive_transpose(cotangent)),
        atol=1e-13,
    )

    coarsen_witness = make_curved_coarsening_witness(
        root_mesh, root_geometry, target_mesh, target_geometry, target_map
    )
    coarsened = prepare_eulerian_volume_overlap(
        target_mesh,
        target_geometry,
        root_mesh,
        root_geometry,
        source_map,
        ("solid-cell", "solid-cell"),
        ("solid-cell",),
        ("aluminum", "aluminum"),
        ("aluminum",),
        source_coverage=target_coverage,
        target_coverage=source_coverage,
        common_root=coarsen_witness,
    )
    chain = EulerianVolumeOverlapChain((refined, coarsened))
    np.testing.assert_allclose(chain.apply_extensive(extensive), extensive, atol=1e-13)
    np.testing.assert_allclose(
        chain.apply_intensive(jnp.asarray((4.0,))), (4.0,), atol=1e-13
    )
    np.testing.assert_allclose(
        chain.apply_extensive_transpose(jnp.ones((1,))), (1.0,), atol=1e-13
    )
    assert (
        EulerianVolumeOverlapChain.revalidate(chain, chain.maps).chain_id
        == chain.chain_id
    )
    assert (
        EulerianVolumeOverlapMap.revalidate(
            refined,
            root_mesh,
            root_geometry,
            target_mesh,
            target_geometry,
            root_mesh=root_mesh,
            root_geometry=root_geometry,
        ).map_id
        == refined.map_id
    )


def test_curved_eulerian_overlap_actions_use_explicit_map_gating() -> None:
    root_mesh, root_geometry, _, target_mesh, target_geometry, _, witness = (
        make_curved_material_refinement()
    )
    assert witness is not None
    domain, source_coverage, target_coverage = make_material_coverage_pair(
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        region_id="aluminum",
        occurrence_path=("assembly-1", "occurrence-9"),
    )
    eulerian_overlap = prepare_eulerian_volume_overlap(
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        make_material_source(domain),
        ("solid-cell",),
        ("solid-cell", "solid-cell"),
        ("aluminum",),
        ("aluminum", "aluminum"),
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        policy=CommonRefinementPolicy(),
        common_root=witness,
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )
    transfer = PhysicalOverlapTransfer(eulerian_overlap.refinement)
    assert eulerian_overlap.refinement.succeeded
    assert eulerian_overlap.integration_evidence is not None
    assert eulerian_overlap.integration_evidence.witness_id == witness.witness_id
    source_content = jnp.asarray((6.0,))
    target_content = transfer.apply_extensive(source_content)
    np.testing.assert_allclose(target_content, (3.0, 3.0), atol=1e-13)
    action = transfer.apply_extensive_with_evidence(source_content)
    np.testing.assert_allclose(action.absolute_conservation_defect, 0.0, atol=1e-13)
    assert action.source_coverage_defects.shape == (1,)
    assert action.target_coverage_defects.shape == (2,)
    _, tangent = jax.jvp(
        transfer.apply_extensive,
        (source_content,),
        (jnp.ones_like(source_content),),
    )
    np.testing.assert_allclose(tangent, (0.5, 0.5), atol=1e-13)
    cotangent = jnp.asarray((1.0, 3.0))
    np.testing.assert_allclose(
        jnp.vdot(target_content, cotangent),
        jnp.vdot(source_content, transfer.apply_extensive_transpose(cotangent)),
        atol=1e-13,
    )
    source_average = jnp.asarray((2.5,))
    target_average = transfer.apply_intensive(source_average)
    np.testing.assert_allclose(target_average, (2.5, 2.5), atol=1e-13)
    np.testing.assert_allclose(
        jnp.vdot(target_average, cotangent),
        jnp.vdot(source_average, transfer.apply_intensive_transpose(cotangent)),
        atol=1e-13,
    )


def test_material_reference_overlap_uses_whole_rational_images_and_two_jacobians() -> (
    None
):
    root_mesh, root_geometry, _, _, _, _, _ = make_curved_material_refinement()
    source_points = np.asarray(root_mesh.coordinates).copy()
    source_points[:, 0] *= 2.0
    target_points = np.asarray(root_mesh.coordinates).copy()
    target_points[:, 1] *= 3.0
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    source_mesh = CellMesh(
        source_points,
        (CellBlock("source", "tetrahedron", cells, global_ids=(41,)),),
    )
    target_mesh = CellMesh(
        target_points,
        (CellBlock("target", "tetrahedron", cells, global_ids=(52,)),),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    target_geometry = CellGeometrySpec.affine(target_mesh)

    rational_x = RationalPolynomial(
        {
            (1, 0, 0): Fraction(5, 4),
            (2, 0, 0): Fraction(3, 4),
            (1, 1, 0): Fraction(-1, 4),
            (1, 0, 1): Fraction(-1, 4),
        },
        {(0, 0, 0): Fraction(1), (1, 0, 0): Fraction(1)},
    )
    rational_chart: ReferenceChart = (
        rational_x,
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    source_reference_map = reference_map_witness(
        (0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,)
    )
    target_reference_map = reference_map_witness(
        (0,),
        (7,),
        (ROOT_REFERENCE_TETRAHEDRON,),
        chart_arguments=(rational_chart,),
    )
    witness = MaterialReferenceVolumeWitness(
        root_mesh,
        root_geometry,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        {"source": source_reference_map},
        {"target": target_reference_map},
    )
    overlap = prepare_material_reference_volume_overlap(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        witness,
        policy=CommonRefinementPolicy(),
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )

    assert overlap.succeeded
    assert overlap.status is CommonRefinementStatus.SUCCESS
    assert overlap.target_offsets.tolist() == [0, 1]
    assert overlap.source_rows.tolist() == [0]
    np.testing.assert_allclose(
        overlap.reference_overlap_measures, (1.0 / 6.0), atol=1e-14
    )
    np.testing.assert_allclose(overlap.source_overlap_measures, (1.0 / 3.0), atol=1e-14)
    np.testing.assert_allclose(overlap.target_overlap_measures, (1.0 / 2.0), atol=1e-14)
    np.testing.assert_allclose(overlap.source_physical_measures, (1.0 / 3.0), atol=1e-14)
    np.testing.assert_allclose(overlap.target_physical_measures, (1.0 / 2.0), atol=1e-14)
    np.testing.assert_allclose(overlap.source_reference_measures, (1.0 / 6.0), atol=1e-14)
    np.testing.assert_allclose(overlap.target_reference_measures, (1.0 / 6.0), atol=1e-14)
    np.testing.assert_allclose(
        overlap.source_reference_coverage_defects, (0.0,), atol=1e-14
    )
    np.testing.assert_allclose(
        overlap.target_reference_coverage_defects, (0.0,), atol=1e-14
    )
    assert overlap.source_reference_defects == ("0",)
    assert overlap.target_reference_defects == ("0",)
    np.testing.assert_allclose(overlap.source_coverage_defects, (0.0,), atol=1e-14)
    np.testing.assert_allclose(overlap.target_coverage_defects, (0.0,), atol=1e-14)
    assert witness.reference_id
    assert witness.maximum_displacement_bound >= np.sqrt(14.0)

    rational_coordinates = (
        rational_x,
        RationalPolynomial({(0, 1, 0): Fraction(1)}, {(0, 0, 0): Fraction(1)}),
        RationalPolynomial({(0, 0, 1): Fraction(1)}, {(0, 0, 0): Fraction(1)}),
    )
    root_element = root_geometry.elements[0]
    assert isinstance(root_element, FiniteElementSpec)
    composed_element = RationalComposedCellGeometryElement(
        root_element, "tetrahedron", rational_coordinates
    )
    composed_geometry = CellGeometrySpec(
        {"root": composed_element},
        {"root": np.asarray(root_geometry.geometry_dofs[0], dtype=np.int32)},
        root_geometry.coordinates,
    )
    root_reference_map = reference_map_witness((0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,))
    common_root = CommonRootVolumeWitness(
        root_mesh,
        root_geometry,
        root_mesh,
        root_geometry,
        root_mesh,
        composed_geometry,
        {"root": root_reference_map},
        {"root": target_reference_map},
    )
    refinement, integration = prepare_common_root_volume_refinement(
        root_mesh,
        root_geometry,
        root_mesh,
        composed_geometry,
        common_root,
        policy=CommonRefinementPolicy(),
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )
    assert refinement.succeeded
    np.testing.assert_allclose(refinement.volumes, (1.0 / 6.0,), atol=1e-14)
    assert integration.witness_id == common_root.witness_id


def test_material_reference_overlap_certifies_curved_image_boundary() -> None:
    root_mesh, root_geometry, _, _, _, _, _ = make_curved_material_refinement()
    source_points = np.asarray(root_mesh.coordinates).copy()
    source_points[:, 0] *= 2.0
    target_points = np.asarray(root_mesh.coordinates).copy()
    target_points[:, 1] *= 3.0
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    source_mesh = CellMesh(
        source_points,
        (CellBlock("source", "tetrahedron", cells, global_ids=(61,)),),
    )
    target_mesh = CellMesh(
        target_points,
        (CellBlock("target", "tetrahedron", cells, global_ids=(62,)),),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    target_geometry = CellGeometrySpec.affine(target_mesh)
    epsilon = Fraction(1, 4)
    curved_chart: ReferenceChart = (
        {(1, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {
            (0, 0, 1): Fraction(1),
            (1, 1, 0): epsilon,
            (2, 1, 0): -epsilon,
            (1, 2, 0): -epsilon,
            (1, 1, 1): -epsilon,
        },
    )
    source_reference_map = reference_map_witness(
        (0,),
        (7,),
        (ROOT_REFERENCE_TETRAHEDRON,),
        chart_arguments=(curved_chart,),
    )
    target_reference_map = reference_map_witness(
        (0,),
        (7,),
        (ROOT_REFERENCE_TETRAHEDRON,),
        chart_arguments=(curved_chart,),
    )
    witness = MaterialReferenceVolumeWitness(
        root_mesh,
        root_geometry,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        {"source": source_reference_map},
        {"target": target_reference_map},
    )
    overlap = prepare_material_reference_volume_overlap(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        witness,
        policy=CommonRefinementPolicy(),
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )

    assert overlap.succeeded
    assert overlap.status is CommonRefinementStatus.SUCCESS
    assert "curved-image-triangular-injective" in witness.source_chart_witnesses[0]
    np.testing.assert_allclose(
        overlap.reference_overlap_measures, (79.0 / 480.0,), atol=1e-14
    )
    np.testing.assert_allclose(
        overlap.source_reference_measures, (79.0 / 480.0,), atol=1e-14
    )
    np.testing.assert_allclose(
        overlap.target_reference_measures, (79.0 / 480.0,), atol=1e-14
    )
    np.testing.assert_allclose(overlap.source_overlap_measures, (1.0 / 3.0,), atol=1e-14)
    np.testing.assert_allclose(overlap.target_overlap_measures, (1.0 / 2.0,), atol=1e-14)


def test_material_reference_overlap_integrates_unequal_curved_partitions() -> None:
    root_mesh, root_geometry, _, _, _, _, _ = make_curved_material_refinement()
    zero, one, half = Fraction(0), Fraction(1), Fraction(1, 2)
    lower_cell = (
        (zero, zero, zero),
        (one, zero, zero),
        (zero, one, zero),
        (zero, zero, half),
    )
    upper_cell = (
        (zero, zero, half),
        (one, zero, zero),
        (zero, one, zero),
        (zero, zero, one),
    )
    physical_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    connectivity = np.asarray(((0, 1, 2, 3), (3, 1, 2, 4)), dtype=np.int32)
    source_points = physical_points.copy()
    source_points[:, 0] *= 2.0
    target_points = physical_points.copy()
    target_points[:, 1] *= 3.0
    source_mesh = CellMesh(
        source_points,
        (CellBlock("source", "tetrahedron", connectivity, global_ids=(71, 72)),),
    )
    # One coordinate element per block: each target cell has its own polynomial map.
    target_mesh = CellMesh(
        target_points,
        (
            CellBlock("target-lower", "tetrahedron", connectivity[:1], global_ids=(81,)),
            CellBlock("target-upper", "tetrahedron", connectivity[1:], global_ids=(82,)),
        ),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    linear_reference = lagrange_element("tetrahedron", 1)
    physical_x: Mapping[tuple[int, ...], Fraction] = {(1, 0, 0): one}
    physical_y: Mapping[tuple[int, ...], Fraction] = {(0, 1, 0): Fraction(3)}
    physical_z_lower: Mapping[tuple[int, ...], Fraction] = {
        (0, 0, 1): half,
        (1, 1, 1): Fraction(1, 4),
    }
    physical_z_upper: Mapping[tuple[int, ...], Fraction] = {
        (0, 0, 0): half,
        (1, 0, 0): -half,
        (0, 1, 0): -half,
        (0, 0, 1): half,
        (1, 1, 0): Fraction(1, 4),
        (2, 1, 0): Fraction(-1, 4),
        (1, 2, 0): Fraction(-1, 4),
        (1, 1, 1): Fraction(1, 4),
    }
    target_geometry = CellGeometrySpec(
        {
            "target-lower": PolynomialComposedCellGeometryElement(
                linear_reference,
                "tetrahedron",
                (physical_x, physical_y, physical_z_lower),
            ),
            "target-upper": PolynomialComposedCellGeometryElement(
                linear_reference,
                "tetrahedron",
                (physical_x, physical_y, physical_z_upper),
            ),
        },
        {
            "target-lower": np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            "target-upper": np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        },
        np.asarray(root_mesh.coordinates),
    )
    x: Expression = {(1, 0, 0): one}
    y: Expression = {(0, 1, 0): one}
    source_charts: ReferenceChartBank = (
        (x, y, {(0, 0, 1): half}),
        (
            x,
            y,
            {
                (0, 0, 0): half,
                (1, 0, 0): -half,
                (0, 1, 0): -half,
                (0, 0, 1): half,
            },
        ),
    )
    beta_half = Fraction(1, 4)
    target_charts: ReferenceChartBank = (
        (
            x,
            y,
            {(0, 0, 1): half, (1, 1, 1): beta_half},
        ),
        (
            x,
            y,
            {
                (0, 0, 0): half,
                (1, 0, 0): -half,
                (0, 1, 0): -half,
                (0, 0, 1): half,
                (1, 1, 0): beta_half,
                (2, 1, 0): -beta_half,
                (1, 2, 0): -beta_half,
                (1, 1, 1): -beta_half,
            },
        ),
    )
    source_reference_map = reference_map_witness(
        (0, 1), (7, 7), (lower_cell, upper_cell), chart_arguments=source_charts
    )
    target_reference_maps = {
        name: reference_map_witness((0,), (7,), (cell,), chart_arguments=(chart,))
        for name, cell, chart in zip(
            ("target-lower", "target-upper"),
            (lower_cell, upper_cell),
            target_charts,
            strict=True,
        )
    }
    witness = MaterialReferenceVolumeWitness(
        root_mesh,
        root_geometry,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        {"source": source_reference_map},
        target_reference_maps,
    )
    overlap = prepare_material_reference_volume_overlap(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        witness,
        policy=CommonRefinementPolicy(),
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )

    assert overlap.succeeded
    assert overlap.status is CommonRefinementStatus.SUCCESS
    assert overlap.target_offsets.tolist() == [0, 2, 3]
    assert overlap.source_rows.tolist() == [0, 1, 1]
    # Exact monomial moments of the independent endpoint Jacobians on these slabs.
    np.testing.assert_allclose(
        overlap.reference_overlap_measures,
        (1.0 / 12.0, 1.0 / 480.0, 1.0 / 12.0 - 1.0 / 480.0),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        overlap.source_overlap_measures,
        (1.0 / 6.0, 1.0 / 240.0, 13.0 / 80.0),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        overlap.target_overlap_measures,
        (1.0 / 4.0, 1.0 / 160.0, 41.0 / 160.0),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        overlap.source_physical_measures,
        (1.0 / 6.0, 1.0 / 6.0),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        overlap.target_physical_measures,
        (41.0 / 160.0, 41.0 / 160.0),
        atol=1e-14,
    )
    assert np.all(
        np.abs(np.asarray(overlap.source_coverage_defects))
        <= np.asarray(overlap.source_coverage_tolerances)
    )
    assert np.all(
        np.abs(np.asarray(overlap.target_coverage_defects))
        <= np.asarray(overlap.target_coverage_tolerances)
    )


def test_material_reference_overlap_integrates_reparameterized_curved_image() -> None:
    root_mesh, root_geometry, _, _, _, _, _ = make_curved_material_refinement()
    zero, one, half = Fraction(0), Fraction(1), Fraction(1, 2)
    source_vertices = (
        (zero, zero, zero),
        (one, zero, zero),
        (zero, one, zero),
        (zero, zero, half),
    )
    target_vertices = (
        (zero, zero, zero),
        (zero, zero, half),
        (one, zero, zero),
        (zero, one, zero),
    )
    source_chart: ReferenceChart = (
        {(1, 0, 0): one},
        {(0, 1, 0): one},
        {(0, 0, 1): half, (1, 1, 1): Fraction(1, 4)},
    )
    target_chart: ReferenceChart = (
        {(0, 1, 0): one},
        {(0, 0, 1): one},
        {(1, 0, 0): half, (1, 1, 1): Fraction(1, 4)},
    )
    source_mesh = CellMesh(
        np.asarray(source_vertices, dtype=np.float64),
        (
            CellBlock(
                "source",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((94,), dtype=np.int64),
            ),
        ),
    )
    target_mesh = CellMesh(
        np.asarray(target_vertices, dtype=np.float64),
        (
            CellBlock(
                "target",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((95,), dtype=np.int64),
            ),
        ),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    target_geometry = CellGeometrySpec.affine(target_mesh)
    source_map = reference_map_witness(
        (0,), (7,), (source_vertices,), chart_arguments=(source_chart,)
    )
    target_map = reference_map_witness(
        (0,), (7,), (target_vertices,), chart_arguments=(target_chart,)
    )
    witness = MaterialReferenceVolumeWitness(
        root_mesh,
        root_geometry,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        {"source": source_map},
        {"target": target_map},
    )
    overlap = prepare_material_reference_volume_overlap(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        witness,
        policy=CommonRefinementPolicy(),
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )

    exact_image_measure = 1.0 / 12.0 + 1.0 / 480.0
    assert overlap.succeeded
    assert overlap.status is CommonRefinementStatus.SUCCESS
    assert overlap.target_offsets.tolist() == [0, 1]
    assert overlap.source_rows.tolist() == [0]
    assert abs(overlap.reference_overlap_measures[0] - exact_image_measure) <= (
        overlap.reference_overlap_errors[0] + 1e-14
    )
    assert abs(overlap.source_reference_measures[0] - exact_image_measure) <= (
        overlap.source_reference_errors[0] + 1e-14
    )
    assert abs(overlap.target_reference_measures[0] - exact_image_measure) <= (
        overlap.target_reference_errors[0] + 1e-14
    )
    np.testing.assert_allclose(overlap.source_overlap_measures, (1.0 / 12.0,), atol=1e-12)
    np.testing.assert_allclose(overlap.target_overlap_measures, (1.0 / 12.0,), atol=1e-12)
    assert overlap.source_reference_defects == ("interval-certified",)
    assert overlap.target_reference_defects == ("interval-certified",)


def test_material_reference_overlap_integrates_nonnested_curved_partitions() -> None:
    root_mesh, root_geometry, _, _, _, _, _ = make_curved_material_refinement()
    zero, one, half = Fraction(0), Fraction(1), Fraction(1, 2)
    p0, p1, p2, p3 = ROOT_REFERENCE_TETRAHEDRON
    split_bd = (half, zero, half)
    split_bc = (half, half, zero)
    split_cd = (zero, half, half)
    source_cells = (
        (p0, p1, split_bc, split_bd),
        (p0, split_bc, p2, split_bd),
        (p0, p2, p3, split_bd),
    )
    target_cells = (
        (p0, p1, split_bc, split_cd),
        (p0, split_bc, p2, split_cd),
        (p0, p3, p1, split_cd),
    )
    beta = Fraction(1, 16)
    image_x: Expression = {(1, 0, 0): one}
    image_y: Expression = {(0, 1, 0): one}
    image_z: Expression = {
        (0, 0, 1): one,
        (1, 1, 1): beta,
        (2, 1, 1): -beta,
        (1, 2, 1): -beta,
        (1, 1, 2): -beta,
    }
    image_chart: tuple[Expression, Expression, Expression] = (
        image_x,
        image_y,
        image_z,
    )
    source_charts: ReferenceChartBank = tuple(
        tuple(
            compose_expression(
                expression,
                _as_chart_arguments(affine_reference_chart(cell)),
            )
            for expression in image_chart
        )
        for cell in source_cells
    )
    target_charts: ReferenceChartBank = tuple(
        tuple(
            compose_expression(
                expression,
                _as_chart_arguments(affine_reference_chart(cell)),
            )
            for expression in image_chart
        )
        for cell in target_cells
    )
    points = np.asarray((p0, p1, p2, p3, split_bd, split_bc, split_cd), dtype=np.float64)
    source_connectivity = np.asarray(
        ((0, 1, 5, 4), (0, 5, 2, 4), (0, 2, 3, 4)), dtype=np.int32
    )
    target_connectivity = np.asarray(
        ((0, 1, 5, 6), (0, 5, 2, 6), (0, 3, 1, 6)), dtype=np.int32
    )
    source_mesh = CellMesh(
        points,
        (
            CellBlock(
                "source",
                "tetrahedron",
                source_connectivity,
                global_ids=np.asarray((101, 102, 103), dtype=np.int64),
            ),
        ),
    )
    target_mesh = CellMesh(
        points,
        (
            CellBlock(
                "target",
                "tetrahedron",
                target_connectivity,
                global_ids=np.asarray((201, 202, 203), dtype=np.int64),
            ),
        ),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    target_geometry = CellGeometrySpec.affine(target_mesh)
    source_map = reference_map_witness(
        (0, 1, 2), (7, 7, 7), source_cells, chart_arguments=source_charts
    )
    target_map = reference_map_witness(
        (0, 1, 2), (7, 7, 7), target_cells, chart_arguments=target_charts
    )
    witness = MaterialReferenceVolumeWitness(
        root_mesh,
        root_geometry,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        {"source": source_map},
        {"target": target_map},
    )

    def polynomial_sum(*values: RationalTerms) -> RationalTerms:
        result: RationalTerms = {}
        for value in values:
            for exponent, coefficient in value.items():
                result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        return {
            exponent: coefficient
            for exponent, coefficient in result.items()
            if coefficient
        }

    def polynomial_scale(value: RationalTerms, factor: Fraction) -> RationalTerms:
        return {
            exponent: factor * coefficient
            for exponent, coefficient in value.items()
            if factor * coefficient
        }

    def polynomial_product(first: RationalTerms, second: RationalTerms) -> RationalTerms:
        result: RationalTerms = {}
        for first_exponent, first_coefficient in first.items():
            for second_exponent, second_coefficient in second.items():
                exponent = (
                    first_exponent[0] + second_exponent[0],
                    first_exponent[1] + second_exponent[1],
                    first_exponent[2] + second_exponent[2],
                )
                result[exponent] = (
                    result.get(exponent, Fraction(0))
                    + first_coefficient * second_coefficient
                )
        return {
            exponent: coefficient
            for exponent, coefficient in result.items()
            if coefficient
        }

    def exact_tetrahedron_volume(vertices: ReferenceTet) -> Fraction:
        origin = vertices[0]
        first, second, third = (
            tuple(vertices[column + 1][axis] - origin[axis] for axis in range(3))
            for column in range(3)
        )
        determinant = (
            first[0] * (second[1] * third[2] - second[2] * third[1])
            - second[0] * (first[1] * third[2] - first[2] * third[1])
            + third[0] * (first[1] * second[2] - first[2] * second[1])
        )
        return abs(determinant) / 6

    def exact_root_integral(vertices: ReferenceTet) -> Fraction:
        origin = vertices[0]
        edges = tuple(
            tuple(vertices[column + 1][axis] - origin[axis] for axis in range(3))
            for column in range(3)
        )
        coordinates: list[RationalTerms] = []
        for axis in range(3):
            terms: RationalTerms = {}
            terms[(0, 0, 0)] = origin[axis]
            terms[(1, 0, 0)] = edges[0][axis]
            terms[(0, 1, 0)] = edges[1][axis]
            terms[(0, 0, 1)] = edges[2][axis]
            coordinates.append(terms)
        x, y, z = coordinates
        one_poly: RationalTerms = {(0, 0, 0): one}
        determinant = polynomial_sum(
            one_poly,
            polynomial_scale(
                polynomial_product(
                    polynomial_product(x, y),
                    polynomial_sum(
                        one_poly,
                        polynomial_scale(x, Fraction(-1)),
                        polynomial_scale(y, Fraction(-1)),
                        polynomial_scale(z, Fraction(-2)),
                    ),
                ),
                beta,
            ),
        )
        canonical_integral = sum(
            (
                coefficient
                * Fraction(
                    factorial(exponent[0])
                    * factorial(exponent[1])
                    * factorial(exponent[2]),
                    factorial(sum(exponent) + 3),
                )
                for exponent, coefficient in determinant.items()
            ),
            Fraction(0),
        )
        return 6 * exact_tetrahedron_volume(vertices) * canonical_integral

    expected_pairs = []
    for target_row, target_cell in enumerate(target_cells):
        for source_row, source_cell in enumerate(source_cells):
            pieces = _clip_exact_tetrahedra(source_cell, target_cell)
            physical_reference = sum(
                (exact_tetrahedron_volume(piece) for piece in pieces), Fraction(0)
            )
            if physical_reference:
                root_reference = sum(
                    (exact_root_integral(piece) for piece in pieces), Fraction(0)
                )
                expected_pairs.append(
                    (target_row, source_row, root_reference, physical_reference)
                )
    expected_pairs.sort(key=lambda value: (value[0], value[1]))
    expected_root_total = exact_root_integral(ROOT_REFERENCE_TETRAHEDRON)
    assert expected_root_total == Fraction(1, 6)
    assert (
        sum((root_value for _, _, root_value, _ in expected_pairs), Fraction(0))
        == expected_root_total
    )
    source_cell_measures = tuple(exact_tetrahedron_volume(cell) for cell in source_cells)
    target_cell_measures = tuple(exact_tetrahedron_volume(cell) for cell in target_cells)
    assert any(
        0 < volume < min(source_cell_measures[source], target_cell_measures[target])
        for target, source, _, volume in expected_pairs
    )

    overlap = prepare_material_reference_volume_overlap(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        witness,
        policy=CommonRefinementPolicy(),
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )

    expected_counts = tuple(
        sum(target == row for target, _, _, _ in expected_pairs)
        for row in range(len(target_cells))
    )
    expected_offsets = np.concatenate(
        (np.asarray((0,), dtype=np.int64), np.cumsum(expected_counts, dtype=np.int64))
    )
    assert overlap.succeeded
    assert overlap.status is CommonRefinementStatus.SUCCESS
    np.testing.assert_array_equal(overlap.target_offsets, expected_offsets)
    assert overlap.source_rows.tolist() == [source for _, source, _, _ in expected_pairs]
    for index, (_, _, expected_root, expected_physical) in enumerate(expected_pairs):
        assert abs(overlap.reference_overlap_measures[index] - float(expected_root)) <= (
            overlap.reference_overlap_errors[index] + 1e-14
        )
        assert abs(overlap.source_overlap_measures[index] - float(expected_physical)) <= (
            overlap.source_overlap_errors[index] + 1e-14
        )
        assert abs(overlap.target_overlap_measures[index] - float(expected_physical)) <= (
            overlap.target_overlap_errors[index] + 1e-14
        )
    np.testing.assert_allclose(
        overlap.source_physical_measures,
        tuple(float(value) for value in source_cell_measures),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        overlap.target_physical_measures,
        tuple(float(value) for value in target_cell_measures),
        atol=1e-14,
    )
    assert overlap.source_reference_defects == ("interval-certified",) * 3
    assert overlap.target_reference_defects == ("interval-certified",) * 3


def test_material_reference_overlap_coarsens_curved_three_cell_partition() -> None:
    root_mesh, root_geometry, _, _, _, _, _ = make_curved_material_refinement()
    zero, one, half = Fraction(0), Fraction(1), Fraction(1, 2)
    p0, p1, p2, p3 = ROOT_REFERENCE_TETRAHEDRON
    split_bd = (half, zero, half)
    split_bc = (half, half, zero)
    source_cells = (
        (p0, p1, split_bc, split_bd),
        (p0, split_bc, p2, split_bd),
        (p0, p2, p3, split_bd),
    )
    beta = Fraction(1, 16)
    image_x: Expression = {(1, 0, 0): one}
    image_y: Expression = {(0, 1, 0): one}
    image_z: Expression = {
        (0, 0, 1): one,
        (1, 1, 1): beta,
        (2, 1, 1): -beta,
        (1, 2, 1): -beta,
        (1, 1, 2): -beta,
    }
    image_chart: tuple[Expression, Expression, Expression] = (
        image_x,
        image_y,
        image_z,
    )
    source_charts = tuple(
        tuple(
            compose_expression(
                expression,
                _as_chart_arguments(affine_reference_chart(cell)),
            )
            for expression in image_chart
        )
        for cell in source_cells
    )
    points = np.asarray((p0, p1, p2, p3, split_bd, split_bc), dtype=np.float64)
    source_mesh = CellMesh(
        points,
        (
            CellBlock(
                "source",
                "tetrahedron",
                np.asarray(
                    ((0, 1, 5, 4), (0, 5, 2, 4), (0, 2, 3, 4)),
                    dtype=np.int32,
                ),
                global_ids=np.asarray((301, 302, 303), dtype=np.int64),
            ),
        ),
    )
    target_mesh = CellMesh(
        np.asarray((p0, p1, p2, p3), dtype=np.float64),
        (
            CellBlock(
                "target",
                "tetrahedron",
                np.asarray(((0, 1, 2, 3),), dtype=np.int32),
                global_ids=np.asarray((401,), dtype=np.int64),
            ),
        ),
    )
    source_geometry = CellGeometrySpec.affine(source_mesh)
    target_geometry = CellGeometrySpec.affine(target_mesh)
    source_map = reference_map_witness(
        (0, 1, 2), (7, 7, 7), source_cells, chart_arguments=source_charts
    )
    target_map = reference_map_witness(
        (0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,), chart_arguments=(image_chart,)
    )
    witness = MaterialReferenceVolumeWitness(
        root_mesh,
        root_geometry,
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        {"source": source_map},
        {"target": target_map},
    )
    overlap = prepare_material_reference_volume_overlap(
        source_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        witness,
        policy=CommonRefinementPolicy(),
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )

    expected_cell_measures = (1.0 / 24.0, 1.0 / 24.0, 1.0 / 12.0)
    assert overlap.succeeded
    assert overlap.status is CommonRefinementStatus.SUCCESS
    assert overlap.target_offsets.tolist() == [0, 3]
    assert overlap.source_rows.tolist() == [0, 1, 2]
    np.testing.assert_allclose(
        overlap.reference_overlap_measures,
        overlap.source_reference_measures,
        atol=1e-14,
    )
    assert (
        abs(float(np.sum(np.asarray(overlap.reference_overlap_measures))) - 1.0 / 6.0)
        <= float(np.sum(np.asarray(overlap.reference_overlap_errors))) + 1e-12
    )
    np.testing.assert_allclose(
        overlap.source_physical_measures, expected_cell_measures, atol=1e-14
    )
    np.testing.assert_allclose(
        overlap.source_overlap_measures, expected_cell_measures, atol=1e-14
    )
    np.testing.assert_allclose(overlap.target_physical_measures, (1.0 / 6.0,), atol=1e-14)
    np.testing.assert_allclose(
        overlap.target_overlap_measures, expected_cell_measures, atol=1e-14
    )
    assert np.all(
        np.abs(np.asarray(overlap.source_coverage_defects))
        <= np.asarray(overlap.source_coverage_tolerances)
    )
    assert np.all(
        np.abs(np.asarray(overlap.target_coverage_defects))
        <= np.asarray(overlap.target_coverage_tolerances)
    )
    assert overlap.source_reference_defects == ("interval-certified",) * 3
    assert overlap.target_reference_defects == ("interval-certified",)


def test_parametric_reference_inverse_proves_interior_curved_inclusion() -> None:
    one = Fraction(1)
    beta = Fraction(1, 16)
    image_x: Expression = {(1, 0, 0): one}
    image_y: Expression = {(0, 1, 0): one}
    image_z: Expression = {
        (0, 0, 1): one,
        (1, 1, 1): beta,
        (2, 1, 1): -beta,
        (1, 2, 1): -beta,
        (1, 1, 2): -beta,
    }
    image_chart: tuple[Expression, Expression, Expression] = (
        image_x,
        image_y,
        image_z,
    )
    interior_cell = (
        (Fraction(1, 4), Fraction(1, 4), Fraction(1, 4)),
        (Fraction(17, 64), Fraction(1, 4), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(17, 64), Fraction(1, 4)),
        (Fraction(1, 4), Fraction(1, 4), Fraction(17, 64)),
    )
    interior_chart: tuple[Expression, ...] = tuple(
        compose_expression(
            expression,
            _as_chart_arguments(affine_reference_chart(interior_cell)),
        )
        for expression in image_chart
    )
    assert _parametric_inverse_inclusion(
        interior_chart, ROOT_REFERENCE_TETRAHEDRON, image_chart
    )


def test_rational_overlap_integration_stops_at_binary64_resolution() -> None:
    epsilon = Fraction(1, 1 << 60)
    vertices = (
        (Fraction(1), Fraction(0), Fraction(0)),
        (Fraction(1) + epsilon, Fraction(0), Fraction(0)),
        (Fraction(1), epsilon, Fraction(0)),
        (Fraction(1), Fraction(0), epsilon),
    )
    expression = RationalPolynomial(
        {(0, 0, 0): Fraction(1)},
        {(0, 0, 0): Fraction(2), (1, 0, 0): Fraction(1)},
    )
    rule = cubature_rule_data("tetrahedron", 8)
    rule_points = np.asarray(rule.points, dtype=np.float64)
    rule_weights = np.asarray(rule.weights, dtype=np.float64)
    policy = CurvedOverlapIntegrationPolicy(
        relative_tolerance=1.0e-300,
        absolute_tolerance=0.0,
    )
    with pytest.raises(CommonRootOverlapError, match="binary64") as error:
        _adaptive_expression_integral(
            expression, vertices, policy, rule_points, rule_weights
        )
    assert error.value.status is CommonRefinementStatus.PREDICATE_UNCERTAIN
    assert error.value.integration_error is not None
    assert error.value.integration_tolerance is not None
    assert error.value.integration_error > error.value.integration_tolerance
    assert len(error.value.unresolved_reference_tetrahedron) == 4


@pytest.mark.parametrize(
    "coverage",
    (CommonRefinementCoverage.TARGET, CommonRefinementCoverage.PARTIAL),
)
def test_curved_common_root_honors_partial_coverage_modes(
    coverage: CommonRefinementCoverage,
) -> None:
    (
        root_mesh,
        root_geometry,
        source_geometry,
        target_mesh,
        target_geometry,
        _,
        witness,
    ) = make_curved_material_refinement(gap=True)
    assert witness is not None
    refinement, integration = prepare_common_root_volume_refinement(
        root_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        witness,
        policy=CommonRefinementPolicy(coverage=coverage),
        integration_policy=CurvedOverlapIntegrationPolicy(),
    )
    assert refinement.status is CommonRefinementStatus.SUCCESS
    assert integration.source_reference_defects == ("-1/12",)
    assert integration.target_reference_defects == ("0",)
    assert np.asarray(refinement.evidence.source_coverage_defects)[0] < 0.0
    with pytest.raises(CommonRootOverlapError, match="coverage defect"):
        prepare_common_root_volume_refinement(
            root_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            witness,
            policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.SOURCE),
            integration_policy=CurvedOverlapIntegrationPolicy(),
        )


def test_curved_common_root_gap_overlap_and_mismatched_source_refuse() -> None:
    (
        root_mesh,
        root_geometry,
        source_geometry,
        target_mesh,
        target_geometry,
        target_map,
        witness,
    ) = make_curved_material_refinement(gap=True)
    assert witness is not None
    with pytest.raises(CommonRootOverlapError, match="coverage defect"):
        prepare_common_root_volume_refinement(
            root_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            witness,
            policy=CommonRefinementPolicy(),
            integration_policy=CurvedOverlapIntegrationPolicy(),
        )
    source_reference_map = reference_map_witness(
        (0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,)
    )
    material_witness = MaterialReferenceVolumeWitness(
        root_mesh,
        root_geometry,
        root_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        {"root": source_reference_map},
        {target_mesh.blocks[0].name: target_map},
    )
    with pytest.raises(CommonRootOverlapError, match="image coverage defect"):
        prepare_material_reference_volume_overlap(
            root_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            material_witness,
            policy=CommonRefinementPolicy(),
            integration_policy=CurvedOverlapIntegrationPolicy(),
        )
    (
        root_mesh,
        root_geometry,
        source_geometry,
        target_mesh,
        target_geometry,
        target_map,
        witness,
    ) = make_curved_material_refinement(duplicate=True)
    assert witness is not None
    for coverage in CommonRefinementCoverage:
        with pytest.raises(CommonRootOverlapError, match="overlap"):
            prepare_common_root_volume_refinement(
                root_mesh,
                source_geometry,
                target_mesh,
                target_geometry,
                witness,
                policy=CommonRefinementPolicy(coverage=coverage),
                integration_policy=CurvedOverlapIntegrationPolicy(),
            )
    material_witness = MaterialReferenceVolumeWitness(
        root_mesh,
        root_geometry,
        root_mesh,
        source_geometry,
        target_mesh,
        target_geometry,
        {"root": source_reference_map},
        {target_mesh.blocks[0].name: target_map},
    )
    with pytest.raises(CommonRootOverlapError, match="image cells overlap"):
        prepare_material_reference_volume_overlap(
            root_mesh,
            source_geometry,
            target_mesh,
            target_geometry,
            material_witness,
            policy=CommonRefinementPolicy(),
            integration_policy=CurvedOverlapIntegrationPolicy(),
        )
    (
        root_mesh,
        root_geometry,
        bad_source_geometry,
        target_mesh,
        target_geometry,
        target_map,
        _,
    ) = make_curved_material_refinement(mismatched_source=True)
    source_mapping = {
        "root": reference_map_witness((0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,))
    }
    target_mapping = {"refined": target_map}
    with pytest.raises(ValueError, match="not the witnessed restriction"):
        CommonRootVolumeWitness(
            root_mesh,
            root_geometry,
            root_mesh,
            bad_source_geometry,
            target_mesh,
            target_geometry,
            source_mapping,
            target_mapping,
        )


def test_material_occurrence_sides_are_not_inferred_from_entity_names() -> None:
    root_mesh, root_geometry, _, target_mesh, target_geometry, target_map, _ = (
        make_curved_material_refinement()
    )
    domain, source_coverage, target_coverage = make_material_coverage_pair(
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        region_id="aluminum",
        occurrence_path=("assembly-1", "occurrence-9"),
    )
    ambiguous = MaterialSourceMapCapability(
        domain.source_id,
        domain.source_revision,
        domain.domain_id,
        ("solid-cell", "solid-cell"),
        ("aluminum", "aluminum"),
        occurrence_paths=(
            ("assembly-1", "occurrence-9"),
            ("assembly-2", "occurrence-9"),
        ),
    )
    reference = declare_material_reference(
        root_mesh,
        root_geometry,
        ambiguous,
        ("solid-cell",),
        ("aluminum",),
        source_coverage,
        frame_id="material-frame",
        policy=MaterialReferencePolicy.ACCEPTED_CONFIGURATION,
        occurrence_paths=(("assembly-1", "occurrence-9"),),
    )
    source_reference_map = reference_map_witness(
        (0,), (7,), (ROOT_REFERENCE_TETRAHEDRON,)
    )
    deformation = VolumeMaterialDeformation(
        reference,
        root_mesh,
        root_geometry,
        target_mesh,
        target_geometry,
        source_reference_maps={"root": source_reference_map},
        target_reference_maps={"refined": target_map},
        source_coverage=source_coverage,
        target_coverage=target_coverage,
        source_frame_id="material-frame",
        target_frame_id="material-frame",
    )
    with pytest.raises(VolumeMaterialMapError, match="occurrence side must be supplied"):
        prepare_volume_material_map(
            root_mesh,
            root_geometry,
            target_mesh,
            target_geometry,
            deformation,
            ("solid-cell",),
            ("solid-cell", "solid-cell"),
            ("aluminum",),
            ("aluminum", "aluminum"),
            source_coverage=source_coverage,
            target_coverage=target_coverage,
        )
