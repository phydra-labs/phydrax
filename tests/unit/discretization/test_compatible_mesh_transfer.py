#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction
from typing import Any

from phydrax import config as pxconfig


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    CellGeometryRestrictionSource,
    coordinate_lagrange_element,
)
from phydrax.discretization._cell_geometry_validity import cell_geometry_id
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization.fem import (
    discontinuous_element,
    form_element,
    lagrange_element,
    prepare_l2_projection_target,
    prepare_nested_field_transfer,
    prepare_projection_field_transfer,
)
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.geometry import CommonRefinementPolicy, prepare_common_refinement
from tests._support.cell_meshes import (
    kuhn_cube,
    red_refinement,
    tetrahedral_bisection,
    unit_square,
)
from tests._support.coordinate_references import coordinate_reference_budget


D = phx.discretization
lc = phx.lifecycle


# Empirical `test_nested_h1_transfer_reproduces_constants_and_polynomials`: matched max |delta|=1.04361e-14; joint allowance fraction=8.51926e-08, margin=100%.
_NESTED_H1_POLYNOMIAL_ATOL = 1e-13
# Empirical `test_nested_h1_transfer_reproduces_constants_and_polynomials`: matched max |delta|=6.77236e-15; joint allowance fraction=6.77236e-08, margin=100%.
_NESTED_H1_CONSTANT_ATOL = 1e-14
# Empirical `test_nested_discontinuous_transfer_conserves_extensive_content`: matched max |delta|=2.22045e-16; joint allowance fraction=0.00208845, margin=99.7912%.
_NESTED_DG_CONTENT_RTOL = 1e-13
# Empirical `test_nested_discontinuous_transfer_conserves_extensive_content`: matched max |delta|=1.9984e-15; joint allowance fraction=0.0198846, margin=98.0115%.
_NESTED_DG_AREA_RTOL = 1e-13
# Empirical `test_planar_compatible_transfer_preserves_moments_with_commuting_evidence`: matched max |delta|=6.66134e-16; joint allowance fraction=1.27671e-07, margin=100%.
_PLANAR_COMPATIBLE_MOMENT_ATOL = 1e-13
# Empirical `test_tetrahedral_nedelec_transfer_preserves_circulation_and_curl_free_fields`: matched max |delta|=3.94129e-15; joint allowance fraction=1.31376e-07, margin=100%.
_TETRAHEDRAL_CIRCULATION_ATOL = 1e-13
# Empirical `test_tetrahedral_nedelec_transfer_preserves_circulation_and_curl_free_fields`: matched max |delta|=2.66454e-15; joint allowance fraction=0.0266454, margin=97.3355%.
_TETRAHEDRAL_GRADIENT_CURL_ATOL = 1e-13
# Empirical `test_nonnested_hdiv_projection_reproduces_rt0_fields_and_plane_flux`: matched max |delta|=1.38778e-16; joint allowance fraction=3.77779e-05, margin=99.9962%.
_NONNESTED_RT0_FACE_FLUX_ATOL = 1e-12
# Empirical `test_nonnested_hdiv_projection_reproduces_rt0_fields_and_plane_flux`: matched max |delta|=2.22045e-16; joint allowance fraction=0.00166533, margin=99.8335%.
_SOURCE_PLANE_FLUX_RTOL = 1e-13
# Empirical `test_nonnested_hdiv_projection_reproduces_rt0_fields_and_plane_flux`: matched max |delta|=0.000328438; joint allowance fraction=0.0123164, margin=98.7684%.
_PROJECTED_PLANE_FLUX_RTOL = 2e-2
# Empirical `test_nonnested_hcurl_projection_reproduces_nedelec_fields`: matched max |delta|=1.22125e-15; joint allowance fraction=0.00052549, margin=99.9475%.
_NONNESTED_CIRCULATION_ATOL = 1e-12
# Empirical `test_nonnested_planar_compatible_projection_reproduces_lowest_order_fields`: matched max |delta|=6.66134e-16; joint allowance fraction=1.17657e-06, margin=99.9999%.
_NONNESTED_PLANAR_MOMENT_ATOL = 1e-12
# Empirical `test_topology_transaction_projects_compatible_fields_on_nonnested_adaptation`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_TRANSACTION_FLUX_PROJECTION_ATOL = 1e-12
# Empirical `test_nonnested_divergence_commutes_for_arbitrary_moments`: matched max |delta|=9.63674e-14; joint allowance fraction=0.00389802, margin=99.6102%.
_NONNESTED_DIVERGENCE_ATOL = 2e-11
# Empirical `test_nonnested_divergence_commutes_for_arbitrary_moments`: matched max |delta|=9.63674e-14; joint allowance fraction=0.00389802, margin=99.6102%.
_NONNESTED_DIVERGENCE_RTOL = 2e-11
# Empirical `test_nonnested_curl_commutes_with_its_realizable_companion`: matched max |delta|=4.49418e-13; joint allowance fraction=0.000803842, margin=99.9196%.
_NONNESTED_CURL_ATOL = 2e-10
# Empirical `test_nonnested_curl_commutes_with_its_realizable_companion`: matched max |delta|=4.49418e-13; joint allowance fraction=0.000803842, margin=99.9196%.
_NONNESTED_CURL_RTOL = 2e-10
# Empirical `test_nonnested_curl_commutes_with_its_realizable_companion`: matched max |delta|=1.19904e-14; joint allowance fraction=5.3039e-08, margin=100%.
_NONNESTED_CURL_CONTENT_ATOL = 2e-11
# Empirical `test_nonnested_curl_commutes_with_its_realizable_companion`: matched max |delta|=4.44089e-15; joint allowance fraction=6.26953e-09, margin=100%.
_NONNESTED_CURL_DUAL_PAIRING_ATOL = 1e-11
# Empirical `test_epoch_admission_rejects_same_topology_wrong_geometry_and_precision`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_EPOCH_IDENTITY_VALUES_ATOL = 1e-12
# Empirical `test_nested_p1_transfer_on_embedded_surface_reports_physical_area_measures`: matched max |delta|=2.66454e-15; joint allowance fraction=0.266454, margin=73.3546%.
_EMBEDDED_SURFACE_AREA_ATOL = 1e-14
# Empirical `test_nested_p1_transfer_on_embedded_surface_reports_physical_area_measures`: matched max |delta|=1.77636e-15; joint allowance fraction=8.88178e-09, margin=100%.
_EMBEDDED_P1_VALUES_ATOL = 1e-14
# Empirical `test_nested_p1_transfer_on_embedded_surface_reports_physical_area_measures`: matched max |delta|=4.44089e-15; joint allowance fraction=0.444089, margin=55.5911%.
_EMBEDDED_P1_CONTENT_ATOL = 1e-14
# Empirical `test_retained_rational_affine_transfer_preserves_source_map_and_inventory`: matched max |delta|=4.44089e-16; joint allowance fraction=0.0444089, margin=95.5591%.
_RATIONAL_AFFINE_DENSITY_ATOL = 1e-14
# Empirical `test_retained_rational_affine_transfer_preserves_source_map_and_inventory`: matched max |delta|=2.44249e-15; joint allowance fraction=1.83187e-08, margin=100%.
_RATIONAL_AFFINE_CONTENT_ATOL = 1e-14
# Empirical `test_retained_rational_affine_transfer_preserves_source_map_and_inventory`: matched max |delta|=2.22045e-16; joint allowance fraction=0.0222045, margin=97.7796%.
_RATIONAL_AFFINE_HISTORY_ATOL = 1e-14


def _pair(element: Any, source: Any, target: Any) -> tuple[Any, Any]:
    spec = D.FiniteElementFieldSpec("u", element)
    return (
        D.FiniteElementPlan(source, spec).prepare(),
        D.FiniteElementPlan(target, spec).prepare(),
    )


def _epochs(source: Any, target: Any) -> tuple[Any, Any]:
    return (
        D.TopologyEpoch(
            0,
            cell_geometry_id(D.CellGeometrySpec.affine(source)),
            source.topology_id,
            "serial",
        ),
        D.TopologyEpoch(
            1,
            cell_geometry_id(D.CellGeometrySpec.affine(target)),
            target.topology_id,
            "serial",
        ),
    )


def _quadratic(points: Any, args: object) -> Any:
    del args
    x, y = points[..., 0], points[..., 1]
    return 1.5 - 0.7 * x + 2.0 * y + 0.3 * x * x - 1.1 * x * y + 0.4 * y * y


def _affine(points: Any, args: object) -> Any:
    del args
    return 0.25 + 2.0 * points[..., 0] - 3.0 * points[..., 1]


@pytest.mark.parametrize(
    ("degree", "function"),
    ((1, _affine), (2, _quadratic)),
    ids=("p1-affine", "p2-quadratic"),
)
def test_nested_h1_transfer_reproduces_constants_and_polynomials(
    degree: int, function: Any
) -> None:
    source_mesh, target_mesh, parents = red_refinement()
    source, target = _pair(lagrange_element("triangle", degree), source_mesh, target_mesh)

    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")
    polynomial = transfer.transfer.apply(source.project("u", function))
    constant = transfer.transfer.apply(jnp.ones((source.dof_maps[0].global_dof_count,)))

    assert transfer.evidence.passed
    assert transfer.semantics == "nested-interpolation"
    assert transfer.geometry.relation == "exact-restriction"
    assert transfer.transfer.preserves_constants and transfer.transfer.conservative
    np.testing.assert_allclose(
        polynomial, target.project("u", function), atol=_NESTED_H1_POLYNOMIAL_ATOL
    )
    np.testing.assert_allclose(constant, 1.0, atol=_NESTED_H1_CONSTANT_ATOL)


@pytest.mark.parametrize("degree", (0, 1), ids=("dg0", "dg1"))
def test_nested_discontinuous_transfer_conserves_extensive_content(degree: int) -> None:
    source_mesh, target_mesh, parents = red_refinement()
    source, target = _pair(
        discontinuous_element("triangle", degree), source_mesh, target_mesh
    )
    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")
    source_epoch, target_epoch = _epochs(source_mesh, target_mesh)
    transition = transfer.epoch_transition(source_epoch, target_epoch)
    assert isinstance(transition, D.TopologyEpochTransition)
    count = source.dof_maps[0].global_dof_count
    values = jnp.asarray(np.random.default_rng(3).uniform(0.5, 2.0, count))
    corners = np.asarray(source_mesh.coordinates)[
        np.asarray(source_mesh.blocks[0].vertices)
    ]
    edges = corners[:, 1:] - corners[:, :1]
    area = 0.5 * np.sum(
        np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    )

    result = transition.apply(values)
    unit = transition.apply(jnp.ones((count,)))

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.target_content, result.source_content, rtol=_NESTED_DG_CONTENT_RTOL
    )
    np.testing.assert_allclose(unit.source_content, area, rtol=_NESTED_DG_AREA_RTOL)
    np.testing.assert_allclose(unit.target_content, area, rtol=_NESTED_DG_AREA_RTOL)


def _divergent_flux(points: Any, args: object) -> Any:
    del args
    return jnp.stack((0.5 + 2.0 * points[..., 0], -1.0 + 2.0 * points[..., 1]), axis=-1)


def _solenoidal_flux(points: Any, args: object) -> Any:
    del args
    return jnp.broadcast_to(jnp.asarray((1.0, -2.0)), points.shape)


def _circulating_field(points: Any, args: object) -> Any:
    del args
    return jnp.stack((0.3 - 1.5 * points[..., 1], 0.8 + 1.5 * points[..., 0]), axis=-1)


def _gradient_field(points: Any, args: object) -> Any:
    del args
    return jnp.broadcast_to(jnp.asarray((2.0, -0.5)), points.shape)


@pytest.mark.parametrize(
    ("element", "function", "semantics"),
    (
        (
            form_element("triangle", 1, 1, twist="twisted", proxy="flux"),
            _divergent_flux,
            "contravariant-piola",
        ),
        (
            form_element("triangle", 1, 1, twist="twisted", proxy="flux"),
            _solenoidal_flux,
            "contravariant-piola",
        ),
        (
            form_element("triangle", 1, 1, proxy="circulation"),
            _circulating_field,
            "covariant-piola",
        ),
        (
            form_element("triangle", 1, 1, proxy="circulation"),
            _gradient_field,
            "covariant-piola",
        ),
    ),
    ids=("rt0-flux", "rt0-divergence-free", "nedelec-circulation", "nedelec-curl-free"),
)
def test_planar_compatible_transfer_preserves_moments_with_commuting_evidence(
    element: Any, function: Any, semantics: str
) -> None:
    source_mesh, target_mesh, parents = red_refinement()
    source, target = _pair(element, source_mesh, target_mesh)

    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")
    moments = transfer.transfer.apply(source.project("u", function))

    assert transfer.semantics == semantics
    assert transfer.evidence.passed
    assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
    # Midpoint edge moments of the linear field on the target give its
    # flux/circulation functionals of the target space.
    np.testing.assert_allclose(
        moments, target.project("u", function), atol=_PLANAR_COMPATIBLE_MOMENT_ATOL
    )


@pytest.mark.parametrize(
    "element",
    (
        form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 1, family="full", twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 2, family="full", twist="twisted", proxy="flux"),
    ),
    ids=("rt0", "bdm1", "bdm2"),
)
def test_tetrahedral_hdiv_transfer_certifies_nesting_and_divergence_commutation(
    element: Any,
) -> None:
    source_mesh, target_mesh, parents = tetrahedral_bisection()
    source, target = _pair(element, source_mesh, target_mesh)

    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")

    assert transfer.semantics == "contravariant-piola"
    assert transfer.evidence.passed
    for name in ("containment", "continuity", "reproduction", "commuting"):
        assert transfer.evidence.defect(name) <= transfer.evidence.tolerance


def _edge_integrals(mesh: Any, constant: np.ndarray, rotation: np.ndarray) -> Any:
    """Analytic lower-to-higher edge circulations of ``constant + rotation x x``."""

    coordinates = np.asarray(mesh.coordinates)
    edges = np.asarray(mesh.connectivity.edges)
    start, end = coordinates[edges[:, 0]], coordinates[edges[:, 1]]
    values = constant + np.cross(rotation, 0.5 * (start + end))
    return np.sum(values * (end - start), axis=1)


def test_tetrahedral_nedelec_transfer_preserves_circulation_and_curl_free_fields() -> (
    None
):
    source_mesh, target_mesh, parents = tetrahedral_bisection()
    source, target = _pair(
        form_element("tetrahedron", 1, 1, proxy="circulation"), source_mesh, target_mesh
    )
    constant, rotation = np.asarray((0.2, -1.0, 0.4)), np.asarray((1.1, -0.3, 0.6))
    potential = np.asarray(source_mesh.coordinates) @ np.asarray((1.0, 2.0, -1.0))
    connectivity = source_mesh.connectivity
    assert isinstance(connectivity, D.TetrahedralConnectivity)
    edges = np.asarray(connectivity.edges)

    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")
    circulation = transfer.transfer.apply(
        jnp.asarray(_edge_integrals(source_mesh, constant, rotation))
    )
    gradient = transfer.transfer.apply(
        jnp.asarray(potential[edges[:, 1]] - potential[edges[:, 0]])
    )

    assert transfer.semantics == "covariant-piola" and transfer.evidence.passed
    for name in ("containment", "continuity", "reproduction", "commuting"):
        assert transfer.evidence.defect(name) <= transfer.evidence.tolerance
    np.testing.assert_allclose(
        circulation,
        _edge_integrals(target_mesh, constant, rotation),
        atol=_TETRAHEDRAL_CIRCULATION_ATOL,
    )
    # The independent topological incidence of the edge space sees no curl.
    np.testing.assert_allclose(
        D.FiniteElementDeRhamComplex(
            target_mesh, family="trimmed", order=1
        ).exterior_derivative(1, gradient),
        0.0,
        atol=_TETRAHEDRAL_GRADIENT_CURL_ATOL,
    )


def test_incompatible_field_family_is_refused() -> None:
    source_mesh, target_mesh, parents = red_refinement()
    source = D.FiniteElementPlan(
        source_mesh, D.FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    ).prepare()
    target = D.FiniteElementPlan(
        target_mesh,
        D.FiniteElementFieldSpec(
            "u", form_element("triangle", 1, 1, twist="twisted", proxy="flux")
        ),
    ).prepare()

    with pytest.raises(ValueError, match="field_signature.elements"):
        prepare_nested_field_transfer(source, target, parents, field_name="u")


def test_piola_semantics_refuse_mismatched_field_conformity() -> None:
    source_mesh, target_mesh, parents = red_refinement()
    source, target = _pair(lagrange_element("triangle", 1), source_mesh, target_mesh)
    transfer = prepare_nested_field_transfer(source, target, parents, field_name="u")

    with pytest.raises(ValueError, match="FieldTransfer.semantics"):
        D.FieldTransfer(
            source.field_spaces[0],
            target.field_spaces[0],
            transfer.transfer.primal,
            geometry=transfer.geometry,
            properties=D.TransferProperties(semantics="covariant-piola"),
        )


def test_rejected_transfer_leaves_the_accepted_composition_untouched() -> None:
    source_mesh, target_mesh, parents = red_refinement()
    source, target = _pair(
        form_element("triangle", 1, 1, proxy="circulation"), source_mesh, target_mesh
    )
    # A wrong parent mapping: every child claims the first parent.
    wrong = prepare_nested_field_transfer(
        source, target, np.zeros_like(parents), field_name="u"
    )
    source_epoch, target_epoch = _epochs(source_mesh, target_mesh)
    values = source.project("u", _circulating_field)
    transition = wrong.epoch_transition(source_epoch, target_epoch)

    def entry(value: Any, epoch: Any) -> Any:
        return lc.CompositionEntry(
            jnp.asarray(value),
            entry_id="field/u",
            role="physical-state",
            owner_id="test",
            structure_id=epoch.epoch_id,
            revision_id=f"field-u-{epoch.index}",
            semantics_id="edge-circulation",
        )

    composition = lc.Composition((entry(values, source_epoch),), boundary_id="accepted")
    receipt = lc.commit_composition_rebind(
        lc.CompositionRebind(
            composition,
            transports=(
                transition.composition_transport(
                    composition.entry("field/u"),
                    entry(transition.apply(values).values, target_epoch),
                ),
            ),
        ),
        accepted_boundary=True,
    )

    assert not wrong.evidence.passed
    assert wrong.evidence.defect("containment") > wrong.evidence.tolerance
    assert wrong.geometry.relation == "bounded-reconstruction"
    assert not receipt.published
    assert receipt.composition is composition


# Non-nested remeshing: jittered interior vertices make the target cells cross
# the source cells, while the grid plane x = 1/2 stays a union of faces of both
# meshes so plane fluxes are sums of oriented face moments.


def _projection(element: Any, source_mesh: Any, target_mesh: Any) -> tuple[Any, Any]:
    source, target = _pair(element, source_mesh, target_mesh)
    refinement = prepare_common_refinement(
        source_mesh,
        target_mesh,
        policy=CommonRefinementPolicy(overlap_simplices=True),
    )
    return target, prepare_projection_field_transfer(
        source,
        prepare_l2_projection_target(target, field_name="u"),
        refinement,
        field_name="u",
    )


def _face_fluxes(mesh: Any, field: Any) -> np.ndarray:
    """Face fluxes along the right-hand normal of the sorted face vertices.

    The edge-midpoint rule integrates quadratic fields exactly on triangles.
    """

    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.connectivity.faces)]
    area_normals = 0.5 * np.cross(
        corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    )
    midpoints = 0.5 * (corners + np.roll(corners, -1, axis=1))
    return np.sum(np.mean(field(midpoints), axis=1) * area_normals, axis=-1)


def _plane_flux(mesh: Any, fluxes: Any) -> float:
    """Flux in +x through the faces of the plane x = 1/2."""

    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.connectivity.faces)]
    plane = np.all(np.isclose(corners[..., 0], 0.5), axis=1)
    direction = np.sign(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 0]
    )
    return float(np.sum((direction * np.asarray(fluxes))[plane]))


def _rt0_field(points: np.ndarray) -> np.ndarray:
    return np.asarray((0.3, -1.1, 0.7)) + 1.6 * points


def _solenoidal_field(points: np.ndarray) -> np.ndarray:
    # Divergence-free with total flux 4/3 through every plane x = constant.
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack((1.0 + y * y, z * x, y), axis=-1)


def test_nonnested_hdiv_projection_reproduces_rt0_fields_and_plane_flux() -> None:
    source_mesh, target_mesh = (
        kuhn_cube(2, orientation_swap=(0, 1)),
        kuhn_cube(4, 0.3, orientation_swap=(0, 1)),
    )
    _, transfer = _projection(
        form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        source_mesh,
        target_mesh,
    )
    evidence = transfer.evidence

    reproduced = transfer.transfer.apply(
        jnp.asarray(_face_fluxes(source_mesh, _rt0_field))
    )
    solenoidal = _face_fluxes(source_mesh, _solenoidal_field)
    projected = transfer.transfer.apply(jnp.asarray(solenoidal))

    assert transfer.semantics == "contravariant-piola" and evidence.passed
    assert transfer.geometry.relation == "bounded-reconstruction"
    for name in (
        "coverage",
        "reproduction",
        "content",
        "commuting",
        "divergence-content",
    ):
        assert evidence.defect(name) <= evidence.tolerance
    assert evidence.estimate("solve-residual") <= 1e-10
    np.testing.assert_allclose(
        reproduced,
        _face_fluxes(target_mesh, _rt0_field),
        atol=_NONNESTED_RT0_FACE_FLUX_ATOL,
    )
    np.testing.assert_allclose(
        _plane_flux(source_mesh, solenoidal), 4.0 / 3.0, rtol=_SOURCE_PLANE_FLUX_RTOL
    )
    # The interior plane flux of the projected field matches the source flux to
    # the projection's own discretization accuracy.
    np.testing.assert_allclose(
        _plane_flux(target_mesh, projected), 4.0 / 3.0, rtol=_PROJECTED_PLANE_FLUX_RTOL
    )


def test_nonnested_hcurl_projection_reproduces_nedelec_fields() -> None:
    source_mesh, target_mesh = (
        kuhn_cube(2, orientation_swap=(0, 1)),
        kuhn_cube(3, 0.3, orientation_swap=(0, 1)),
    )
    _, transfer = _projection(
        form_element("tetrahedron", 1, 1, proxy="circulation"), source_mesh, target_mesh
    )
    constant, rotation = np.asarray((0.5, -0.2, 1.3)), np.asarray((-0.7, 0.9, 0.4))

    circulation = transfer.transfer.apply(
        jnp.asarray(_edge_integrals(source_mesh, constant, rotation))
    )

    assert transfer.evidence.passed
    assert transfer.evidence.defect("reproduction") <= transfer.evidence.tolerance
    assert transfer.evidence.defect("curl-content") <= transfer.evidence.tolerance
    assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
    assert transfer.evidence.estimate("cellwise-curl-realizability") > 1e-3
    np.testing.assert_allclose(
        circulation,
        _edge_integrals(target_mesh, constant, rotation),
        atol=_NONNESTED_CIRCULATION_ATOL,
    )


@pytest.mark.parametrize(
    ("element", "function"),
    (
        (form_element("triangle", 1, 1, twist="twisted", proxy="flux"), _divergent_flux),
        (form_element("triangle", 1, 1, proxy="circulation"), _circulating_field),
    ),
    ids=("rt0", "nedelec"),
)
def test_nonnested_planar_compatible_projection_reproduces_lowest_order_fields(
    element: Any, function: Any
) -> None:
    source_mesh, target_mesh = (
        unit_square(2, orientation_swap=(0, 1)),
        unit_square(5, 0.3, orientation_swap=(0, 1)),
    )
    source = D.FiniteElementPlan(source_mesh, D.FiniteElementFieldSpec("u", element))
    target, transfer = _projection(element, source_mesh, target_mesh)

    moments = transfer.transfer.apply(source.prepare().project("u", function))

    assert transfer.evidence.passed
    np.testing.assert_allclose(
        moments, target.project("u", function), atol=_NONNESTED_PLANAR_MOMENT_ATOL
    )


def test_nonnested_projection_refuses_mismatched_piola_maps() -> None:
    source_mesh, target_mesh = (
        unit_square(2, orientation_swap=(0, 1)),
        unit_square(3, 0.3, orientation_swap=(0, 1)),
    )
    source, _ = _pair(
        form_element("triangle", 1, 1, twist="twisted", proxy="flux"),
        source_mesh,
        target_mesh,
    )
    _, target = _pair(
        form_element("triangle", 1, 1, proxy="circulation"), source_mesh, target_mesh
    )

    with pytest.raises(
        ValueError, match="scientific form type, twist, and physical proxy"
    ):
        prepare_projection_field_transfer(
            source,
            prepare_l2_projection_target(target, field_name="u"),
            prepare_common_refinement(
                source_mesh,
                target_mesh,
                policy=CommonRefinementPolicy(overlap_simplices=True),
            ),
            field_name="u",
        )


def _bisect(
    source: Any, refine: Any = (), coarsen: Any = (), hierarchy: Any = None
) -> Any:
    return phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(
                np.asarray(refine, dtype=np.int64),
                np.asarray(coarsen, dtype=np.int64),
                hierarchy=hierarchy,
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )


def test_topology_transaction_projects_compatible_fields_on_nonnested_adaptation() -> (
    None
):
    mesh = D.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5))),
        np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)), dtype=np.int32),
        vertex_global_ids=np.asarray((1, 2, 3, 4, 5)),
        cell_global_ids=np.asarray((10, 20, 30, 40)),
    )
    refined = _bisect(
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()), (10,)
    )
    children = np.setdiff1d(
        np.concatenate(
            [np.asarray(block.global_ids) for block in refined.target.mesh.blocks]
        ),
        np.asarray(mesh.blocks[0].global_ids),
    )
    coarsened = _bisect(refined.target, coarsen=children, hierarchy=refined.hierarchy)
    fields = D.FiniteElementFieldSpec(
        "flux", form_element("triangle", 1, 1, twist="twisted", proxy="flux")
    )
    fine = D.FiniteElementPlan(refined.target.mesh, fields).prepare()
    accepted = phx.solver.FiniteElementAcceptedState(
        (fine.project("flux", _divergent_flux),),
        0.0,
        0,
        refined.target.mesh.topology_id,
        "prepared",
        "compiled",
    )
    transaction = phx.solver.FiniteElementTopologyTransaction(
        lambda candidate_mesh, values, materials, lineage, args: True, fields=fields
    )

    result = transaction.execute(accepted, refined.target.mesh, coarsened)
    coarse = D.FiniteElementPlan(result.mesh, fields).prepare()

    assert phx.solver.refinement_parent_cells(coarsened) is None
    assert bool(result.committed) and result.receipt is not None
    assert result.transfers[0].evidence.passed
    # RT0 contains the flux a + 2x, so coarsening by projection is lossless.
    np.testing.assert_allclose(
        result.state.fields[0],
        coarse.project("flux", _divergent_flux),
        atol=_TRANSACTION_FLUX_PROJECTION_ATOL,
    )


def _opposite_cube() -> Any:
    source = kuhn_cube(1, orientation_swap=(0, 1))
    coordinates = np.asarray(source.coordinates).copy()
    coordinates[:, 0] = 1.0 - coordinates[:, 0]
    # Reflection reverses every positive tetrahedron; swap the source local
    # vertices to retain the authored compatible-transfer orientation.
    cells = np.asarray(source.blocks[0].vertices)[:, [1, 0, 2, 3]]
    return D.CellMesh.from_tetrahedra(coordinates, cells)


def _vector_geometry(
    discretization: Any, reference: np.ndarray
) -> tuple[np.ndarray, ...]:
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    geometry = discretization.evaluate_block_geometry(
        "u",
        0,
        discretization.default_runtime.coordinates,
        reference,
        np.ones(reference.shape[0]),
    )
    dof_map = discretization.dof_maps[0]
    transforms = np.asarray(dof_map.cell_transforms[0])
    basis = np.einsum("cqav,cai->cqiv", np.asarray(geometry.basis_values), transforms)
    gradients = np.einsum(
        "cqavd,cai->cqivd", np.asarray(geometry.physical_gradients), transforms
    )
    element = discretization.elements[0][0]
    if element.mapping == "contravariant_piola":
        derivative = np.trace(gradients, axis1=-2, axis2=-1)[..., None]
    else:
        derivative = np.stack(
            (
                gradients[..., 2, 1] - gradients[..., 1, 2],
                gradients[..., 0, 2] - gradients[..., 2, 0],
                gradients[..., 1, 0] - gradients[..., 0, 1],
            ),
            axis=-1,
        )
    return (
        np.asarray(dof_map.cell_dofs[0]),
        basis,
        derivative,
        np.asarray(rule.points),
        np.asarray(rule.weights),
    )


def _overlap_values(
    source: Any,
    target: Any,
    refinement: Any,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    simplices = np.asarray(refinement.simplices)
    entries = np.repeat(
        np.arange(len(refinement.source_cells)), np.diff(refinement.simplex_offsets)
    )
    source_cells = np.asarray(refinement.source_cells)[entries]
    target_cells = np.asarray(refinement.target_cells)[entries]
    frames = np.swapaxes(simplices[:, 1:] - simplices[:, :1], -1, -2)
    physical = simplices[:, :1] + np.asarray(rule.points)[None] @ np.swapaxes(
        frames, -1, -2
    )
    weights = np.abs(np.linalg.det(frames))[:, None] * np.asarray(rule.weights)[None]
    source_corners = np.asarray(source.mesh.coordinates)[
        np.asarray(source.mesh.blocks[0].vertices)
    ]
    source_frames = np.swapaxes(source_corners[:, 1:] - source_corners[:, :1], -1, -2)
    reference = np.linalg.solve(
        source_frames[source_cells, None],
        (physical - source_corners[source_cells, None, 0])[..., None],
    )[..., 0]
    routes, _, derivatives, _, _ = _vector_geometry(source, reference.reshape((-1, 3)))
    points = np.arange(reference.shape[0] * reference.shape[1]).reshape(
        reference.shape[:2]
    )
    derivative = derivatives[source_cells[:, None], points]
    values = np.asarray(
        phx.ein.contract("sqnw,sn->sqw", derivative, coefficients[routes[source_cells]])
    )
    target_corners = np.asarray(target.mesh.coordinates)[
        np.asarray(target.mesh.blocks[0].vertices)
    ]
    target_frames = np.swapaxes(target_corners[:, 1:] - target_corners[:, :1], -1, -2)
    target_reference = np.linalg.solve(
        target_frames[target_cells, None],
        (physical - target_corners[target_cells, None, 0])[..., None],
    )[..., 0]
    return target_cells, target_reference, weights, values


@pytest.mark.parametrize(
    "element",
    (
        form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 1, family="full", twist="twisted", proxy="flux"),
        form_element("tetrahedron", 2, 2, family="full", twist="twisted", proxy="flux"),
    ),
    ids=("rt0", "bdm1", "bdm2"),
)
def test_nonnested_divergence_commutes_for_arbitrary_moments(element: Any) -> None:
    source, target = _pair(
        element, kuhn_cube(1, orientation_swap=(0, 1)), _opposite_cube()
    )
    refinement = prepare_common_refinement(
        source.mesh, target.mesh, policy=CommonRefinementPolicy(overlap_simplices=True)
    )
    transfer = prepare_projection_field_transfer(
        source,
        prepare_l2_projection_target(target, field_name="u"),
        refinement,
        field_name="u",
    )
    values = np.random.default_rng(51).normal(size=source.dof_maps[0].global_dof_count)
    transferred = np.asarray(transfer.transfer.apply(values))
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    routes, _, derivatives, reference, reference_weights = _vector_geometry(
        target, np.asarray(rule.points)
    )
    corners = np.asarray(target.mesh.coordinates)[
        np.asarray(target.mesh.blocks[0].vertices)
    ]
    measures = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))
    degree = element.degree - 1
    tests = (
        np.ones((reference.shape[0], 1))
        if degree == 0
        else np.column_stack((np.ones(reference.shape[0]), reference))
    )
    divergence = np.asarray(
        phx.ein.contract("cqnw,cn->cqw", derivatives, transferred[routes])
    )
    actual = np.asarray(
        phx.ein.contract(
            "cq,qp,cqw->cpw",
            measures[:, None] * reference_weights[None],
            tests,
            divergence,
        )
    )
    cells, points, weights, source_divergence = _overlap_values(
        source, target, refinement, values
    )
    overlap_tests = (
        np.ones(points.shape[:2] + (1,))
        if degree == 0
        else np.concatenate((np.ones(points.shape[:2] + (1,)), points), axis=-1)
    )
    expected = np.zeros_like(actual)
    np.add.at(
        expected,
        cells,
        np.asarray(
            phx.ein.contract("sq,sqp,sqw->spw", weights, overlap_tests, source_divergence)
        ),
    )
    # Independently integrate every target polynomial divergence moment,
    # including the linear moments needed by BDM2, for an arbitrary source.
    np.testing.assert_allclose(
        actual, expected, atol=_NONNESTED_DIVERGENCE_ATOL, rtol=_NONNESTED_DIVERGENCE_RTOL
    )


@pytest.mark.parametrize("degree", (0, 1))
def test_nonnested_curl_commutes_with_its_realizable_companion(degree: int) -> None:
    source, target = _pair(
        form_element("tetrahedron", 1, degree + 1, proxy="circulation"),
        kuhn_cube(1, orientation_swap=(0, 1)),
        _opposite_cube(),
    )
    refinement = prepare_common_refinement(
        source.mesh, target.mesh, policy=CommonRefinementPolicy(overlap_simplices=True)
    )
    prepared = prepare_l2_projection_target(target, field_name="u")
    transfer = prepare_projection_field_transfer(
        source, prepared, refinement, field_name="u"
    )
    values = np.random.default_rng(32).normal(size=source.dof_maps[0].global_dof_count)
    transferred = np.asarray(transfer.transfer.apply(values))
    rule = phx.integration.ReferenceTetrahedronRule(
        phx.integration.GaussLegendreRule(4)
    ).materialize()
    routes, _, curls, _, reference_weights = _vector_geometry(
        target, np.asarray(rule.points)
    )
    corners = np.asarray(target.mesh.coordinates)[
        np.asarray(target.mesh.blocks[0].vertices)
    ]
    measures = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1]))
    weights = measures[:, None] * reference_weights[None]
    local = np.asarray(phx.ein.contract("cq,cqiw,cqjw->cij", weights, curls, curls))
    size = target.dof_maps[0].global_dof_count
    gram = np.zeros((size, size))
    np.add.at(gram, (routes[:, :, None], routes[:, None, :]), local)
    cells, points, overlap_weights, source_curl = _overlap_values(
        source, target, refinement, values
    )
    _, _, overlap_curls, _, _ = _vector_geometry(target, points.reshape((-1, 3)))
    point_indices = np.arange(points.shape[0] * points.shape[1]).reshape(points.shape[:2])
    overlap_curls = overlap_curls[cells[:, None], point_indices]
    load = np.zeros((size,))
    np.add.at(
        load,
        routes[cells],
        np.asarray(
            phx.ein.contract(
                "sq,sqnw,sqw->sn", overlap_weights, overlap_curls, source_curl
            )
        ),
    )
    # Independent curl-space Galerkin projection: nullspace coefficients do not
    # matter, while the resulting physical curl is unique.
    companion = np.linalg.lstsq(gram, load, rcond=1e-12)[0]
    actual = np.asarray(phx.ein.contract("cqiw,ci->cqw", curls, transferred[routes]))
    expected = np.asarray(phx.ein.contract("cqiw,ci->cqw", curls, companion[routes]))
    np.testing.assert_allclose(
        actual, expected, atol=_NONNESTED_CURL_ATOL, rtol=_NONNESTED_CURL_RTOL
    )
    np.testing.assert_allclose(
        np.asarray(phx.ein.contract("cq,cqw->w", weights, actual)),
        np.asarray(phx.ein.contract("sq,sqw->w", overlap_weights, source_curl)),
        atol=_NONNESTED_CURL_CONTENT_ATOL,
    )
    dual = np.random.default_rng(9).normal(size=size)
    np.testing.assert_allclose(
        dual @ transferred,
        values @ np.asarray(transfer.transfer.pullback(dual)),
        atol=_NONNESTED_CURL_DUAL_PAIRING_ATOL,
    )


def test_projection_refuses_different_hdiv_families_with_the_same_mapping() -> None:
    source, _ = _pair(
        form_element("tetrahedron", 2, 1, twist="twisted", proxy="flux"),
        kuhn_cube(1, orientation_swap=(0, 1)),
        _opposite_cube(),
    )
    _, target = _pair(
        form_element("tetrahedron", 2, 1, family="full", twist="twisted", proxy="flux"),
        kuhn_cube(1, orientation_swap=(0, 1)),
        _opposite_cube(),
    )
    refinement = prepare_common_refinement(
        source.mesh, target.mesh, policy=CommonRefinementPolicy(overlap_simplices=True)
    )
    with pytest.raises(ValueError, match=r"field_signature\.elements"):
        prepare_projection_field_transfer(
            source,
            prepare_l2_projection_target(target, field_name="u"),
            refinement,
            field_name="u",
        )


def test_nonnested_piola_projection_refuses_curved_coordinate_maps() -> None:
    mesh = D.CellMesh.from_tetrahedra(
        np.concatenate((np.zeros((1, 3)), np.eye(3))),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    )
    coordinate_element = coordinate_lagrange_element("tetrahedron", 2)
    nodes = np.asarray(coordinate_element.reference_nodes).copy()
    nodes[:, 2] += 0.05 * nodes[:, 0] * (1.0 - nodes[:, 0])
    curved = D.FiniteElementPlan(
        mesh,
        D.FiniteElementFieldSpec(
            "u", form_element("tetrahedron", 1, 2, proxy="circulation")
        ),
        coordinate_spec=D.CellGeometrySpec(
            {"tetrahedra": coordinate_element},
            {"tetrahedra": np.arange(nodes.shape[0], dtype=np.int32)[None]},
            nodes,
        ),
    ).prepare()
    # A common refinement of affine corners cannot authorize dropping a
    # non-affine coordinate map in a compatible field remap.
    with pytest.raises(ValueError, match="require_affine_simplices.coordinate_element"):
        prepare_l2_projection_target(curved, field_name="u")


@pytest.mark.parametrize("mapped", (False, True), ids=("affine", "curved"))
@pytest.mark.parametrize("compatible", (False, True), ids=("scalar", "hcurl"))
def test_epoch_admission_rejects_same_topology_wrong_geometry_and_precision(
    mapped: bool, compatible: bool
) -> None:
    mesh = D.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    if mapped:
        coordinate = coordinate_lagrange_element("triangle", 2)
        values = np.asarray(coordinate.reference_nodes).copy()
        values[:, 0] += 0.125 * values[:, 0] * values[:, 1]
        geometry = D.CellGeometrySpec(
            {"triangles": coordinate},
            {"triangles": np.arange(values.shape[0], dtype=np.int32)[None, :]},
            values,
        )
    else:
        geometry = D.CellGeometrySpec.affine(mesh)
    element = (
        form_element("triangle", 1, 1, proxy="circulation")
        if compatible
        else discontinuous_element("triangle", 0)
    )
    prepared = D.FiniteElementPlan(
        mesh, D.FiniteElementFieldSpec("u", element), coordinate_spec=geometry
    ).prepare()
    transfer = prepare_nested_field_transfer(
        prepared,
        prepared,
        np.asarray((0,), dtype=np.int32),
        field_name="u",
        parent_reference_vertices=np.asarray((((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),)),
        source_geometry=geometry,
        target_geometry=geometry,
    )
    source_epoch = D.TopologyEpoch(
        0, cell_geometry_id(geometry), mesh.topology_id, "serial"
    )
    target_epoch = D.TopologyEpoch(
        1, cell_geometry_id(geometry), mesh.topology_id, "serial"
    )
    accepted = np.linspace(0.2, 1.1, prepared.field_spaces[0].vector_space.size)
    untouched = accepted.copy()
    transition = transfer.epoch_transition(source_epoch, target_epoch)
    result = transition.apply(accepted)
    assert bool(result.successful)
    np.testing.assert_allclose(result.values, accepted, atol=_EPOCH_IDENTITY_VALUES_ATOL)
    for wrong_source in (False, True):
        old = (
            D.TopologyEpoch(0, "stale-coordinate-map", mesh.topology_id, "serial")
            if wrong_source
            else source_epoch
        )
        new = (
            target_epoch
            if wrong_source
            else D.TopologyEpoch(1, "stale-coordinate-map", mesh.topology_id, "serial")
        )
        with pytest.raises(ValueError, match="represented geometry"):
            transfer.epoch_transition(old, new)
        with pytest.raises(ValueError, match="epoch_transition.endpoints.geometry_id"):
            if isinstance(transition, D.TopologyEpochTransition):
                D.TopologyEpochTransition(
                    old,
                    new,
                    transition.transfer,
                    transition.source_measures,
                    transition.target_measures,
                    measure_defect_bound=transition.measure_defect_bound,
                )
            else:
                D.FieldEpochTransition(
                    old,
                    new,
                    transition.transfer,
                    evidence_passed=True,
                    evidence_id=transfer.evidence.evidence_id,
                )
    with pytest.raises(TypeError, match="dtype"):
        transition.apply(accepted.astype(np.float32))
    np.testing.assert_array_equal(accepted, untouched)


def test_nested_p1_transfer_on_embedded_surface_reports_physical_area_measures() -> None:
    # Three unit right triangles of a tetrahedron corner in R^3; the first is
    # bisected through its hypotenuse midpoint. Measures are surface areas.
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    source_mesh = D.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2), (0, 2, 3), (0, 3, 1)), dtype=np.int32)
    )
    refined = np.concatenate((points, ((0.5, 0.5, 0.0),)))
    target_mesh = D.CellMesh.from_triangles(
        refined, np.asarray(((0, 1, 4), (0, 4, 2), (0, 2, 3), (0, 3, 1)), dtype=np.int32)
    )
    field = D.FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    source = D.FiniteElementPlan(
        source_mesh, field, coordinate_spec=D.CellGeometrySpec.affine(source_mesh)
    ).prepare()
    target = D.FiniteElementPlan(
        target_mesh, field, coordinate_spec=D.CellGeometrySpec.affine(target_mesh)
    ).prepare()
    reference = np.asarray(
        (
            ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5)),
            ((0.0, 0.0), (0.5, 0.5), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        )
    )
    transfer = prepare_nested_field_transfer(
        source,
        target,
        np.asarray((0, 0, 1, 2)),
        field_name="u",
        parent_reference_vertices=reference,
    )
    assert transfer.evidence.passed
    source_measures = np.asarray(transfer.source_measures)
    target_measures = np.asarray(transfer.target_measures)
    np.testing.assert_allclose(
        source_measures.sum(), 1.5, rtol=0.0, atol=_EMBEDDED_SURFACE_AREA_ATOL
    )
    np.testing.assert_allclose(
        target_measures.sum(), 1.5, rtol=0.0, atol=_EMBEDDED_SURFACE_AREA_ATOL
    )

    def linear(coordinates: np.ndarray) -> np.ndarray:
        return 2.0 + coordinates[:, 0] - 3.0 * coordinates[:, 2]

    values = linear(np.asarray(source.dof_maps[0].dof_coordinates))
    mapped = np.asarray(transfer.transfer.apply(jnp.asarray(values)))
    np.testing.assert_allclose(
        mapped,
        linear(np.asarray(target.dof_maps[0].dof_coordinates)),
        atol=_EMBEDDED_P1_VALUES_ATOL,
    )
    # Independent inventory: each unit right triangle has area 1/2 and a linear
    # integrand integrates to area times its vertex mean.
    exact = sum(
        0.5 * np.mean(linear(points[list(cell)]))
        for cell in ((0, 1, 2), (0, 2, 3), (0, 3, 1))
    )
    np.testing.assert_allclose(
        source_measures @ values, exact, rtol=0.0, atol=_EMBEDDED_P1_CONTENT_ATOL
    )
    np.testing.assert_allclose(
        target_measures @ mapped, exact, rtol=0.0, atol=_EMBEDDED_P1_CONTENT_ATOL
    )


def _retained_rational_affine_pair(
    budget: CoordinateEnclosureBudget, *, action_shift: float = 0.0
) -> tuple[
    D.FiniteElementDiscretization,
    D.FiniteElementDiscretization,
    D.CellGeometrySpec,
    D.CellGeometrySpec,
    np.ndarray,
]:
    # The source's middle vertex is exactly 1/3 on the authored segment.
    # Its retained midpoint with -1 is -1/3, not the floating-point sum
    # (-1 + float(1/3))/2, which lies one carrier ulp farther left.
    points = np.asarray(((-1.0, 0.0), (1.0 / 3.0, 0.0), (0.0, 1.0)))
    source_mesh = D.CellMesh.from_triangles(
        points, np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    exact = D.ExactPlcCellGeometrySource(
        np.asarray(((-1.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        np.empty((0, 3), dtype=np.int64),
        np.asarray(((0, 1),), dtype=np.int64),
        np.asarray((0, 1, 0), dtype=np.int64),
        np.asarray((-1, 0, -1), dtype=np.int64),
        np.asarray(((0.0, 0.0), (2.0 / 3.0, 0.0), (0.0, 0.0))),
        domain_source_id="retained-rational-affine-source",
        domain_source_revision="authored-segment",
        source_triangle_ids=np.empty((0,), dtype=np.int64),
        source_triangle_bounds=np.empty((0,), dtype=np.float64),
        source_segment_ids=np.asarray((7,), dtype=np.int64),
        source_segment_bounds=np.asarray((1e-15,), dtype=np.float64),
        vertex_parameter_ratios=(
            ((0, 1), (0, 1)),
            ((2, 3), (0, 1)),
            ((0, 1), (0, 1)),
        ),
        budget=budget,
    )
    source_geometry = D.CellGeometrySpec.plc(source_mesh, exact, budget=budget)
    source_element = source_geometry.elements[0]
    assert isinstance(source_element, FiniteElementSpec)
    midpoint = Fraction(-1, 3) + Fraction(4, 3) * Fraction(action_shift)
    target_points = np.concatenate((points, ((float(midpoint), 0.0),)))
    target_mesh = D.CellMesh(
        target_points,
        (
            D.CellBlock(
                "left",
                "triangle",
                np.asarray(((0, 3, 2),), dtype=np.int32),
                global_ids=np.asarray((10,), dtype=np.int64),
            ),
            D.CellBlock(
                "right",
                "triangle",
                np.asarray(((3, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((11,), dtype=np.int64),
            ),
        ),
    )
    weights = (0.5 - action_shift, 0.5 + action_shift, 0.0)
    actions = (
        np.asarray(((1.0, 0.0, 0.0), weights, (0.0, 0.0, 1.0))),
        np.asarray((weights, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    )
    target_geometry = D.CellGeometrySpec(
        {
            block.name: BarycentricCellGeometryElement(source_element, action)
            for block, action in zip(target_mesh.blocks, actions, strict=True)
        },
        {block.name: np.asarray(((0, 1, 2),)) for block in target_mesh.blocks},
        source_geometry.coordinates,
        restriction_source=CellGeometryRestrictionSource(
            cell_geometry_id(source_geometry),
            source_mesh.topology_id,
            {block.name: np.asarray((0,)) for block in target_mesh.blocks},
            {block.name: np.asarray(((0, 1, 2),)) for block in target_mesh.blocks},
            block_source_blocks={block.name: "triangles" for block in target_mesh.blocks},
        ),
        exact_source=exact,
        budget=budget,
    )
    field = D.FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    reference = np.asarray(
        (
            ((0.0, 0.0), (0.5, 0.0), (0.0, 1.0)),
            ((0.5, 0.0), (1.0, 0.0), (0.0, 1.0)),
        )
    )
    return (
        D.FiniteElementPlan(source_mesh, field, coordinate_spec=source_geometry).prepare(
            budget=budget
        ),
        D.FiniteElementPlan(target_mesh, field, coordinate_spec=target_geometry).prepare(
            budget=budget
        ),
        source_geometry,
        target_geometry,
        reference,
    )


def test_retained_rational_affine_transfer_preserves_source_map_and_inventory() -> None:
    with coordinate_reference_budget() as budget:
        source, target, source_geometry, target_geometry, reference = (
            _retained_rational_affine_pair(budget)
        )
        identities = (
            source.mesh.mesh_id,
            target.mesh.mesh_id,
            cell_geometry_id(source_geometry),
            cell_geometry_id(target_geometry),
        )
        coordinates = np.asarray(target.mesh.coordinates).copy()
        bank = np.asarray(target_geometry.coordinates).copy()
        target_element = target_geometry.elements[0]
        assert isinstance(target_element, BarycentricCellGeometryElement)
        represented = (
            np.asarray(
                target_element.tabulate(
                    coordinate_lagrange_element("triangle", 1).reference_nodes
                )[0]
            )
            @ bank[np.asarray(target_geometry.geometry_dofs[0])]
        )
        assert float(represented[0, 1, 0]).hex() == "-0x1.5555555555556p-2"
        assert float(coordinates[3, 0]).hex() == "-0x1.5555555555555p-2"
        transfer = prepare_nested_field_transfer(
            source,
            target,
            np.asarray((0, 0)),
            field_name="u",
            parent_reference_vertices=reference,
            source_geometry=source_geometry,
            target_geometry=target_geometry,
            budget=budget,
        )
        assert transfer.evidence.passed
        assert dict(transfer.evidence.defects)["geometry"] == 0.0
        assert transfer.transfer.conservative
        assert (
            transfer.transfer.preserves_constants and transfer.transfer.preserves_linear
        )
        assert transfer.geometry.source_geometry_id == identities[2]
        assert transfer.geometry.target_geometry_id == identities[3]
        transition = transfer.epoch_transition(
            D.TopologyEpoch(0, identities[2], source.mesh.topology_id, "serial"),
            D.TopologyEpoch(1, identities[3], target.mesh.topology_id, "serial"),
        )
        assert isinstance(transition, D.TopologyEpochTransition)
        density = transition.apply(jnp.full((source.dof_maps[0].global_dof_count,), 2.0))
        assert bool(density.successful)
        np.testing.assert_allclose(
            density.values, 2.0, rtol=0.0, atol=_RATIONAL_AFFINE_DENSITY_ATOL
        )
        np.testing.assert_allclose(
            density.source_content, 4.0 / 3.0, atol=_RATIONAL_AFFINE_CONTENT_ATOL
        )
        np.testing.assert_allclose(
            density.target_content, 4.0 / 3.0, atol=_RATIONAL_AFFINE_CONTENT_ATOL
        )
        history = source.dof_maps[0].dof_coordinates[:, 0] + 0.5
        np.testing.assert_allclose(
            transfer.transfer.apply(history),
            target.dof_maps[0].dof_coordinates[:, 0] + 0.5,
            rtol=0.0,
            atol=_RATIONAL_AFFINE_HISTORY_ATOL,
        )
        np.testing.assert_array_equal(target.mesh.coordinates, coordinates)
        np.testing.assert_array_equal(target_geometry.coordinates, bank)
        assert identities == (
            source.mesh.mesh_id,
            target.mesh.mesh_id,
            cell_geometry_id(source_geometry),
            cell_geometry_id(target_geometry),
        )
        # The rounded-carrier projection route still cannot drop the retained map.
        with pytest.raises(ValueError, match="FE geometry differs"):
            prepare_l2_projection_target(target, field_name="u")


def test_retained_affine_transfer_refuses_nonzero_source_map_within_rounding_slack() -> (
    None
):
    with coordinate_reference_budget() as budget:
        source, target, source_geometry, target_geometry, reference = (
            _retained_rational_affine_pair(budget, action_shift=2.0**-50)
        )
        with pytest.raises(ValueError, match="preserve the source map algebraically"):
            prepare_nested_field_transfer(
                source,
                target,
                np.asarray((0, 0)),
                field_name="u",
                parent_reference_vertices=reference,
                source_geometry=source_geometry,
                target_geometry=target_geometry,
                budget=budget,
            )


def test_retained_affine_transfer_refuses_stale_target_corner_carrier() -> None:
    with coordinate_reference_budget() as budget:
        _, target, _, target_geometry, _ = _retained_rational_affine_pair(budget)
        points = np.asarray(target.mesh.coordinates).copy()
        points[3, 0] = np.nextafter(points[3, 0], -np.inf)
        with pytest.raises(ValueError, match="match the bound mesh"):
            D.FiniteElementPlan(
                target.mesh.with_coordinates(
                    points, numeric_version="stale-corner-carrier"
                ),
                D.FiniteElementFieldSpec("u", lagrange_element("triangle", 1)),
                coordinate_spec=target_geometry,
            ).prepare(budget=budget)


@pytest.mark.parametrize("payload", ((), (3,), (2, 3)))
def test_sparse_topology_transfer_compiled_actions_match_eager_payloads(
    payload: tuple[int, ...],
) -> None:
    from phydrax.discretization.fem import vertex_interpolation_transfer

    transfer = vertex_interpolation_transfer(
        np.asarray(((0, 1), (2, 0), (1, 2), (0, 2)), dtype=np.int32),
        np.asarray(((0.3, 0.7), (1.0, 0.0), (0.25, 0.75), (0.5, 0.5)), dtype=np.float64),
        np.ones((4, 2), dtype=np.bool_),
        source_size=3,
        source_topology_id="payload-source",
        target_topology_id="payload-target",
    )
    rng = np.random.default_rng(785)
    values = rng.normal(size=(3,) + payload)
    duals = rng.normal(size=(4,) + payload)
    eager_primal = transfer.primal.mv_block(values.reshape((3, -1))).reshape(
        (4,) + payload
    )
    eager_transpose = transfer.primal.transpose_mv_block(duals.reshape((4, -1))).reshape(
        (3,) + payload
    )
    np.testing.assert_array_equal(transfer.apply(values), eager_primal)
    np.testing.assert_array_equal(transfer.pullback(duals), eager_transpose)


@pytest.mark.parametrize("retained_target", (False, True), ids=("parent", "retained"))
def test_reference_coarsening_authenticates_exact_affine_corner_images(
    retained_target: bool,
) -> None:
    from phydrax.discretization._cell_geometry_transition_types import (
        NestedReferenceWitnesses,
    )

    with coordinate_reference_budget() as budget:
        coarse, fine, coarse_geometry, fine_geometry, reference = (
            _retained_rational_affine_pair(budget)
        )
        if retained_target:
            coarse, coarse_geometry = fine, fine_geometry
            reference = np.broadcast_to(
                np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
                (2, 3, 2),
            )
        witnesses = NestedReferenceWitnesses(
            np.asarray((10, 11), dtype=np.int64),
            np.asarray((10, 11) if retained_target else (0, 0), dtype=np.int64),
            reference,
        )
        transfer = prepare_nested_field_transfer(
            fine,
            coarse,
            None,
            field_name="u",
            source_geometry=fine_geometry,
            target_geometry=coarse_geometry,
            coarsening_witnesses=witnesses,
            budget=budget,
        )
        assert transfer.evidence.passed
        assert transfer.transfer.conservative
        assert transfer.transfer.preserves_constants
        fine_points = np.asarray(fine.dof_maps[0].dof_coordinates)
        coarse_points = np.asarray(coarse.dof_maps[0].dof_coordinates)
        values = fine_points[:, 0] + 0.5 * fine_points[:, 1] + 1.0
        expected = coarse_points[:, 0] + 0.5 * coarse_points[:, 1] + 1.0
        np.testing.assert_allclose(
            transfer.transfer.apply(values),
            expected,
            rtol=0.0,
            atol=_RATIONAL_AFFINE_HISTORY_ATOL,
        )
        source_content = np.asarray(transfer.source_measures) @ values
        target_content = np.asarray(transfer.target_measures) @ np.asarray(
            transfer.transfer.apply(values)
        )
        np.testing.assert_allclose(
            target_content,
            source_content,
            rtol=0.0,
            atol=_RATIONAL_AFFINE_CONTENT_ATOL,
        )
