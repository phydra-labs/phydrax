# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Geometry-authorized meshfree k-forms: exactness, topology, consumers, refusals."""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from examples.meshfree_higher_forms import (
    chart_for,
    holed_square,
    planar_grid_mesh,
    realize,
    solid_torus,
    sphere_mesh,
    unit_sphere,
)
from phydrax.discretization import CellMesh
from phydrax.discretization.meshfree import (
    ComplexDomainIdentity,
    ComplexGeometryAuthority,
    MeshfreeCellComplexPlan,
    MeshfreeComplexAdmissionError,
    MeshfreeComplexPolicy,
    PreparedMeshfreeCellComplex,
    radius_clique_complex,
    RadiusCliquePolicy,
)
from phydrax.exterior import (
    ComplexBoundary,
    DiscreteForm,
    integrate_form,
    trace_evidence,
    trace_map,
    validate_de_rham_commutation,
    validate_harmonic_cohomology,
)
from phydrax.linalg import harmonic_subspace, hodge_laplacian
from phydrax.metrix import DifferentialForm
from phydrax.solver import HodgeLaplacePlan


def _square_identity() -> ComplexDomainIdentity:
    return ComplexDomainIdentity("square", "domain", measure=9.0, betti=(1, 0, 0))


@pytest.fixture(scope="module")
def square() -> PreparedMeshfreeCellComplex:
    return realize(planar_grid_mesh(1, hole=False), _square_identity())


@pytest.fixture(scope="module")
def holed() -> PreparedMeshfreeCellComplex:
    return holed_square(1)


@pytest.fixture(scope="module")
def slab() -> PreparedMeshfreeCellComplex:
    return solid_torus(1)


@pytest.fixture(scope="module")
def sphere() -> PreparedMeshfreeCellComplex:
    return unit_sphere(1)


def _linear_coefficients(dimension: int, degree: int) -> Callable[[Array], Array]:
    slopes = jnp.arange(1.0, comb(dimension, degree) + 1.0, dtype=jnp.float64)

    def coefficients(point: Array) -> Array:
        return slopes * (1.0 + point[0] - 0.5 * point[-1]) + 0.25 * point[1]

    return coefficients


def _node_norm(complex_: PreparedMeshfreeCellComplex, values: Array) -> Array:
    return jnp.sum(complex_.node_weights[:, None] * values * values)


@pytest.mark.parametrize("name", ["square", "slab"])
def test_linear_forms_are_reproduced_with_exact_hodge_norms(
    name: str, request: pytest.FixtureRequest
) -> None:
    """Patch test: P1 forms reproduce exactly and M_k integrates |ω|² exactly."""
    complex_: PreparedMeshfreeCellComplex = request.getfixturevalue(name)
    dimension = complex_.cochain.dimension
    for degree in range(dimension + 1):
        coefficients = _linear_coefficients(dimension, degree)
        form = DifferentialForm(coefficients, chart=chart_for(dimension), degree=degree)
        cochain = integrate_form(form, complex_.bridge)
        exact = jax.vmap(coefficients)(complex_.nodes)
        np.testing.assert_allclose(
            complex_.reconstruct(cochain), exact, atol=1e-10, rtol=1e-10
        )
        np.testing.assert_allclose(
            complex_.sample(degree, exact).values, cochain.values, atol=1e-10, rtol=1e-10
        )
        hodge_norm = jnp.vdot(
            cochain.values, complex_.cochain.hodge_star(degree, cochain.values)
        )
        np.testing.assert_allclose(hodge_norm, _node_norm(complex_, exact), rtol=1e-10)
        assert bool(complex_.evidence.degrees[degree].hodge_admitted)
        assert float(complex_.evidence.degrees[degree].reproduction_residual) < 1e-9


def test_hodge_duality_pairs_complementary_degrees(
    square: PreparedMeshfreeCellComplex,
) -> None:
    """|ω|² over M_k equals |⋆ω|² over M_{n-k} for linear planar forms."""
    chart = square.bridge.chart

    def pair(degree: int, coefficients: Callable[[Array], Array]) -> Array:
        form = DifferentialForm(coefficients, chart=chart, degree=degree)
        values = integrate_form(form, square.bridge).values
        return jnp.vdot(values, square.cochain.hodge_star(degree, values))

    scalar = pair(0, lambda x: jnp.asarray([1.0 + x[0] - 2.0 * x[1]]))
    density = pair(2, lambda x: jnp.asarray([1.0 + x[0] - 2.0 * x[1]]))
    one_form = pair(1, lambda x: jnp.asarray([x[1] + 1.0, 2.0 - x[0]]))
    rotated = pair(1, lambda x: jnp.asarray([x[0] - 2.0, x[1] + 1.0]))
    np.testing.assert_allclose(scalar, density, rtol=1e-11)
    np.testing.assert_allclose(one_form, rotated, rtol=1e-11)


def test_codifferential_is_the_metric_adjoint_of_d(
    slab: PreparedMeshfreeCellComplex,
) -> None:
    key = np.random.default_rng(3)
    counts = slab.cochain.cell_counts
    for degree in range(3):
        u = jnp.asarray(key.standard_normal(counts[degree]))
        v = jnp.asarray(key.standard_normal(counts[degree + 1]))
        left = jnp.vdot(
            slab.cochain.exterior_derivative(degree, u),
            slab.cochain.hodge_star(degree + 1, v),
        )
        right = jnp.vdot(
            u, slab.cochain.hodge_star(degree, slab.cochain.codifferential(degree + 1, v))
        )
        np.testing.assert_allclose(left, right, rtol=1e-7)


@pytest.mark.parametrize("name", ["holed", "slab", "sphere"])
def test_cubic_interpolation_commutes_with_d(
    name: str, request: pytest.FixtureRequest
) -> None:
    """Stokes on authoritative cells: R(dω) = d(Rω) to quadrature round-off."""
    complex_: PreparedMeshfreeCellComplex = request.getfixturevalue(name)
    ambient = complex_.bridge.chart.dimension
    for degree in range(complex_.cochain.dimension):
        components = comb(ambient, degree)
        form = DifferentialForm(
            lambda x, n=components: (
                (x[0] ** 3 - x[0] * x[-1] ** 2 + 0.5)
                * jnp.arange(1.0, n + 1.0, dtype=jnp.float64)
            ),
            chart=complex_.bridge.chart,
            degree=degree,
        )
        evidence = validate_de_rham_commutation(form, complex_.bridge, tolerance=1e-11)
        assert bool(evidence.valid), float(evidence.relative_residual)


def test_sampled_moments_commute_exactly_on_reproduced_polynomials(
    square: PreparedMeshfreeCellComplex,
) -> None:
    x, y = square.nodes[:, 0], square.nodes[:, 1]
    potential = (2.0 + 3.0 * x - y)[:, None]
    gradient = jnp.broadcast_to(jnp.asarray([3.0, -1.0]), (x.shape[0], 2))
    derived = square.cochain.exterior_derivative(0, square.sample(0, potential).values)
    np.testing.assert_allclose(derived, square.sample(1, gradient).values, atol=1e-11)
    circulation = jnp.stack((-y, x), axis=1)
    np.testing.assert_allclose(
        square.cochain.exterior_derivative(1, square.sample(1, circulation).values),
        square.sample(2, jnp.full((x.shape[0], 1), 2.0)).values,
        atol=1e-11,
    )


def test_orientation_reversal_changes_basis_signs_only(
    holed: PreparedMeshfreeCellComplex,
) -> None:
    mesh = planar_grid_mesh(1, hole=True)
    identity = holed.plan.authority.identity
    base = holed
    counts = base.cochain.cell_counts
    flips = (
        np.ones(counts[0]),
        np.where(np.arange(counts[1]) % 3 == 0, -1.0, 1.0),
        -np.ones(counts[2]),
    )
    reversed_ = MeshfreeCellComplexPlan(
        ComplexGeometryAuthority(mesh, identity, orientation=flips),
        chart=chart_for(2),
    ).prepare()
    form = DifferentialForm(
        lambda x: jnp.asarray([x[1] ** 2, x[0] * x[1]]), chart=base.bridge.chart, degree=1
    )
    original = integrate_form(form, base.bridge).values
    flipped = integrate_form(form, reversed_.bridge).values
    np.testing.assert_allclose(flipped, flips[1] * original, atol=1e-13)
    np.testing.assert_allclose(
        reversed_.cochain.hodge_star(1, flipped),
        flips[1] * base.cochain.hodge_star(1, original),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        reversed_.cochain.exterior_derivative(1, flipped),
        flips[2] * base.cochain.exterior_derivative(1, original),
        atol=1e-12,
    )
    single = np.ones(counts[2])
    single[0] = -1.0
    with pytest.raises(MeshfreeComplexAdmissionError, match="incoherent"):
        ComplexGeometryAuthority(
            mesh, identity, orientation=(flips[0], np.ones(counts[1]), single)
        )


@pytest.mark.parametrize(
    ("name", "boundary"),
    [
        ("holed", "absolute"),
        ("holed", "relative"),
        ("slab", "absolute"),
        ("slab", "relative"),
        ("sphere", "absolute"),
    ],
)
def test_harmonic_dimensions_equal_exact_betti_numbers(
    name: str, boundary: ComplexBoundary, request: pytest.FixtureRequest
) -> None:
    complex_: PreparedMeshfreeCellComplex = request.getfixturevalue(name)
    for degree in range(complex_.cochain.dimension + 1):
        _, report = validate_harmonic_cohomology(
            complex_.cochain, degree, boundary=boundary, tolerance=1e-7
        )
        assert bool(report.complete), (
            degree,
            report.harmonic_rank,
            report.exact_dimension,
        )
    if boundary == "absolute":
        ranks = tuple(
            validate_harmonic_cohomology(complex_.cochain, degree, tolerance=1e-7)[
                1
            ].exact_dimension
            for degree in range(complex_.cochain.dimension + 1)
        )
        assert ranks == complex_.evidence.betti


def test_stokes_identity_and_trace_orientation(
    holed: PreparedMeshfreeCellComplex,
) -> None:
    chart = holed.bridge.chart
    form = DifferentialForm(
        lambda x: jnp.asarray([-x[1] * x[0] ** 2, x[0] + x[1] ** 3]),
        chart=chart,
        degree=1,
    )
    values = integrate_form(form, holed.bridge).values
    mapping = trace_map(holed.cochain)
    interior = jnp.sum(holed.cochain.exterior_derivative(1, values))
    boundary = jnp.sum(mapping.maps[1].mv(values))
    np.testing.assert_allclose(interior, boundary, rtol=1e-12, atol=1e-12)
    # Admission makes top cells uniformly oriented; that sign orients both sums.
    areas = integrate_form(
        DifferentialForm(lambda x: jnp.asarray([1.0]), chart=chart, degree=2),
        holed.bridge,
    ).values
    orientation = float(jnp.sign(areas[0]))
    assert bool(jnp.all(jnp.sign(areas) == orientation))
    # Green: ∫(∂x(x + y³) − ∂y(−y x²)) = ∫(1 + x²) over [0,3]² minus [1,2]².
    np.testing.assert_allclose(
        orientation * interior, (9.0 + 27.0) - (1.0 + 7.0 / 3.0), rtol=1e-12
    )
    scalar = integrate_form(
        DifferentialForm(lambda x: jnp.asarray([x[0] * x[1]]), chart=chart, degree=0),
        holed.bridge,
    ).values
    assert bool(trace_evidence(mapping, (scalar,)).successful)
    interior_facet = ~np.asarray(holed.cochain.boundary_masks[1])
    selected = np.zeros_like(interior_facet)
    selected[np.flatnonzero(interior_facet)[0]] = True
    with pytest.raises(ValueError, match="exactly one active incident"):
        trace_map(holed.cochain, boundary_mask=selected)


def test_relative_and_absolute_boundary_complexes_differ(
    holed: PreparedMeshfreeCellComplex,
) -> None:
    absolute = holed.cochain.hilbert_complex(boundary="absolute")
    relative = holed.cochain.hilbert_complex(boundary="relative")
    boundary_edges = int(np.count_nonzero(np.asarray(holed.cochain.boundary_masks[1])))
    assert boundary_edges == holed.evidence.boundary_facet_count > 0
    assert relative.space(1).size == absolute.space(1).size - boundary_edges
    assert relative.space(2).size == absolute.space(2).size


def test_mixed_hodge_laplace_consumes_top_degree_forms(
    holed: PreparedMeshfreeCellComplex,
) -> None:
    hilbert = holed.cochain.hilbert_complex(boundary="absolute")
    harmonic = harmonic_subspace(hilbert, 2, expected_dimension=0)
    expected = jnp.sin(jnp.arange(hilbert.space(2).size, dtype=jnp.float64))
    source = hodge_laplacian(hilbert, 2).mv(expected)
    plan = HodgeLaplacePlan(
        holed.cochain, 2, boundary="absolute", formulation="mixed", harmonic=harmonic
    )
    result = plan.solve(source)
    assert bool(result.successful)
    error = jnp.sqrt(hilbert.space(2).inner(result.u - expected, result.u - expected))
    assert float(error) < 1e-6 * float(
        jnp.sqrt(hilbert.space(2).inner(expected, expected))
    )


def test_reconstruct_refuses_foreign_forms(
    square: PreparedMeshfreeCellComplex, holed: PreparedMeshfreeCellComplex
) -> None:
    foreign = DiscreteForm(
        holed.cochain.realization_id,
        holed.cochain.form_type(1),
        jnp.zeros((holed.cochain.cell_counts[1],)),
    )
    with pytest.raises(ValueError, match="primal form of this realization"):
        square.reconstruct(foreign)
    with pytest.raises(ValueError, match="values must have shape"):
        square.sample(1, jnp.zeros((square.nodes.shape[0], 3)))


def test_unsupported_authority_and_identity_are_refused(
    sphere: PreparedMeshfreeCellComplex,
) -> None:
    mesh = planar_grid_mesh(1, hole=True)
    with pytest.raises(TypeError, match="requires a CellMesh"):
        ComplexGeometryAuthority(mesh.topology, _square_identity())  # ty: ignore[invalid-argument-type]
    clique = radius_clique_complex(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        1.5,
        policy=RadiusCliquePolicy(maximum_pairs=8),
    )
    with pytest.raises(TypeError, match="research-only"):
        MeshfreeCellComplexPlan(clique, chart=chart_for(2))  # ty: ignore[invalid-argument-type]
    filled = planar_grid_mesh(1, hole=False)
    with pytest.raises(
        MeshfreeComplexAdmissionError, match="Betti numbers .* do not represent"
    ):
        ComplexGeometryAuthority(
            filled,
            ComplexDomainIdentity(
                "square-with-hole", "domain", measure=9.0, betti=(1, 1, 0)
            ),
        )
    with pytest.raises(
        MeshfreeComplexAdmissionError, match="measure .* does not represent"
    ):
        ComplexGeometryAuthority(
            filled,
            ComplexDomainIdentity(
                "square-with-hole", "domain", measure=8.0, betti=(1, 0, 0)
            ),
        )
    with pytest.raises(MeshfreeComplexAdmissionError, match="not supported"):
        ComplexGeometryAuthority(
            sphere_mesh(1),
            ComplexDomainIdentity("ball", "domain", measure=4.0, betti=(1, 0, 0)),
        )
    assert sphere.evidence.fidelity == "geometry-authorized"
    assert sphere.evidence.boundary_facet_count == 0


def test_hodge_positivity_failure_is_refused() -> None:
    mesh = CellMesh.from_triangles(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.asarray([[0, 1, 2]])
    )
    identity = ComplexDomainIdentity("triangle", "domain", measure=0.5, betti=(1, 0, 0))
    with pytest.raises(MeshfreeComplexAdmissionError, match="positive-definite"):
        MeshfreeCellComplexPlan(
            ComplexGeometryAuthority(mesh, identity),
            chart=chart_for(2),
            policy=MeshfreeComplexPolicy(polynomial_degree=0, stabilization=0.0),
        ).prepare()


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        (MeshfreeComplexPolicy(maximum_patch_entities=4), "maximum_patch_entities"),
        (MeshfreeComplexPolicy(maximum_workset_entries=64), "maximum_workset_entries"),
    ],
    ids=["patch-entities", "workset"],
)
def test_capacity_refusals(policy: MeshfreeComplexPolicy, message: str) -> None:
    with pytest.raises(MeshfreeComplexAdmissionError, match=message):
        realize(planar_grid_mesh(1, hole=False), _square_identity(), policy=policy)


def test_radius_clique_route_is_bounded_abstract_research() -> None:
    angles = 2.0 * np.pi * np.arange(16) / 16
    cloud = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    complex_ = radius_clique_complex(
        cloud, 0.5, policy=RadiusCliquePolicy(maximum_pairs=64)
    )
    assert complex_.fidelity == "abstract-research"
    assert tuple(
        complex_.betti.dimension(degree)
        for degree in range(complex_.topology.dimension + 1)
    ) == (1, 1)
    assert complex_.simplex_counts == (16, 16)
    filled = radius_clique_complex(
        cloud, 0.8, policy=RadiusCliquePolicy(maximum_pairs=128)
    )
    assert filled.topology.dimension == 2
    assert filled.betti.dimension(1) == 1
    with pytest.raises(MeshfreeComplexAdmissionError, match="maximum_simplices"):
        radius_clique_complex(
            cloud, 0.8, policy=RadiusCliquePolicy(maximum_pairs=128, maximum_simplices=20)
        )
    with pytest.raises(MeshfreeComplexAdmissionError, match="maximum_work"):
        radius_clique_complex(
            cloud, 0.8, policy=RadiusCliquePolicy(maximum_pairs=128, maximum_work=10)
        )
