import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import CellMesh
from phydrax.discretization._topology import EntitySelection
from phydrax.discretization.fem._generic import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.geometry import MeshRegion
from phydrax.linalg import (
    DifferentiationPolicy,
    FailurePolicy,
    FGMRES,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from phydrax.operators.integral.layer_potential._galerkin3d import (
    LaplaceSingleLayerDP0GalerkinPolicy3D,
)
from phydrax.operators.integral.layer_potential._scalar_calderon3d import (
    prepare_scalar_calderon_dp0_3d,
)
from phydrax.solver._fem_bem_scalar import (
    prepare_scalar_laplace_fem_bem_3d,
)


_TETRA_VERTICES = jnp.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
)
_TETRA_FACES = jnp.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=jnp.int32)


@pytest.fixture(scope="module")
def tetra_coupling():
    mesh = CellMesh.from_tetrahedra(
        _TETRA_VERTICES,
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
    )
    field = FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1))
    fem = FiniteElementPlan(mesh, field).prepare(numeric_version="fem-bem-test")
    surface = MeshRegion(
        _TETRA_VERTICES,
        _TETRA_FACES,
        feature_id="matching-fem-bem-tetrahedron",
    )
    bem_policy = LaplaceSingleLayerDP0GalerkinPolicy3D(
        regular_order=3,
        singular_order=3,
        near_order=3,
        near_ratio=1.0,
        absolute_tolerance=2.0e-3,
        relative_tolerance=2.0e-3,
        target_block_size=4,
        source_block_size=4,
    )
    calderon = prepare_scalar_calderon_dp0_3d(
        surface,
        policy=bem_policy,
        numeric_version="fem-bem-test",
    )
    facets = mesh.topology.entity_sets[2]
    exterior = EntitySelection.from_subset(facets, "boundary")
    linear = LinearSolvePolicy(
        FGMRES(restart=12, stagnation_iterations=12),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )
    prepared = prepare_scalar_laplace_fem_bem_3d(
        fem,
        surface,
        calderon,
        interface_selection=exterior,
        linear=linear,
    )
    return fem, surface, calderon, prepared


def _pair(values, covectors):
    return sum(jnp.vdot(value, covector) for value, covector in zip(values, covectors))


def test_matching_interface_has_exact_trace_orientation_and_conormal_flux(
    tetra_coupling,
):
    fem, surface, _, prepared = tetra_coupling
    interface = prepared.interface
    gradient = jnp.asarray((0.75, -0.5, 0.25), dtype=fem.stiffness.source.dtype)
    coefficients = 1.25 + fem.default_runtime.coordinates @ gradient
    expected_trace = jnp.mean(
        coefficients[jnp.asarray(interface.fem_vertex_indices)[surface.faces]], axis=1
    )
    expected_conormal = interface.outward_normals @ gradient

    np.testing.assert_allclose(interface.interface_coordinates, surface.vertices)
    np.testing.assert_allclose(
        interface.trace(coefficients), expected_trace, atol=2.0e-12
    )
    np.testing.assert_allclose(
        interface.conormal(coefficients), expected_conormal, atol=2.0e-12
    )
    assert interface.normal_convention.startswith("normal points from the FEM interior")
    assert float(interface.minimum_orientation_cosine) > 0.0
    assert float(interface.coordinate_max_error) == 0.0
    assert float(interface.flux_closure_norm) < 2.0e-12
    assert abs(float(interface.integrated_flux(expected_conormal))) < 2.0e-12


def test_interface_and_coupled_block_have_exact_transposes(tetra_coupling):
    _, _, _, prepared = tetra_coupling
    interface = prepared.interface
    fem_vector = jnp.asarray((0.2, -0.4, 0.7, 1.1))
    boundary_vector = jnp.asarray((-0.3, 0.8, 0.1, -0.6))

    actions = (
        (interface.trace_operator, fem_vector, boundary_vector),
        (interface.boundary_load_operator, boundary_vector, fem_vector),
        (interface.conormal_operator, fem_vector, boundary_vector),
    )
    for operator, source, target in actions:
        np.testing.assert_allclose(
            jnp.vdot(operator.mv(source), target),
            jnp.vdot(source, operator.transpose_mv(target)),
            atol=2.0e-12,
        )

    primal = (fem_vector, boundary_vector)
    dual = (
        jnp.asarray((-0.9, 0.3, 0.5, 0.2)),
        jnp.asarray((0.6, -0.2, 0.4, 0.9)),
    )
    np.testing.assert_allclose(
        _pair(prepared.operator.mv(primal), dual),
        _pair(primal, prepared.operator.transpose_mv(dual)),
        atol=2.0e-10,
    )


def test_manufactured_tetrahedron_volume_source_solves_end_to_end(tetra_coupling):
    _, _, calderon, prepared = tetra_coupling
    manufactured_interior = jnp.asarray((0.3, -0.15, 0.4, 0.8))
    boundary_right = -prepared.exterior_trace_relation.mv(
        prepared.interface.trace(manufactured_interior)
    )
    boundary_solution = solve(
        LinearSystem(
            calderon.single_layer,
            problem_id="manufactured-fem-bem-boundary-conormal",
        ),
        boundary_right,
        policy=prepared.linear_policy,
    )
    assert bool(boundary_solution.successful)
    manufactured_conormal = boundary_solution.value
    manufactured_load = prepared.stiffness_operator.mv(
        manufactured_interior
    ) - prepared.interface.boundary_load(manufactured_conormal)
    source_solution = solve(
        LinearSystem(
            prepared.mass_operator,
            problem_id="manufactured-fem-bem-volume-source",
        ),
        manufactured_load,
        policy=prepared.linear_policy,
    )
    assert bool(source_solution.successful)
    source_coefficients = source_solution.value

    result = prepared.solve(source_coefficients)

    assert bool(result.valid)
    assert not prepared.operator.capabilities.materialize
    assert not calderon.assembly_report.materializable
    assert result.spatial_dimension == 3
    assert result.pde == "scalar interior Poisson / homogeneous decaying exterior Laplace"
    assert "Johnson-Nedelec" in result.formulation
    assert "continuum certification" in result.non_goals[-1]
    assert jnp.all(jnp.isfinite(result.bem_quadrature_maximum_errors))
    np.testing.assert_allclose(
        result.volume_load, prepared.mass_operator.mv(source_coefficients), atol=2.0e-11
    )
    np.testing.assert_allclose(
        result.interior_coefficients, manufactured_interior, atol=2.0e-7
    )
    np.testing.assert_allclose(
        result.exterior_conormal, manufactured_conormal, atol=2.0e-7
    )
    assert float(result.relative_block_residual) < 2.0e-8
    assert float(result.interface_equation_defect) < 2.0e-8
    assert abs(float(result.flux_balance_defect)) < 2.0e-8
    exterior_value, reports = result.evaluate_exterior(jnp.asarray((2.0, 2.0, 2.0)))
    assert jnp.isfinite(exterior_value)
    assert all(bool(report.pde_membership_valid) for report in reports)


def test_mismatched_interface_is_rejected(tetra_coupling):
    fem, surface, calderon, _ = tetra_coupling
    mismatched_vertices = _TETRA_VERTICES.at[0, 0].add(1.0e-3)
    mismatched = MeshRegion(
        mismatched_vertices,
        _TETRA_FACES,
        feature_id=surface.feature_id,
    )

    with pytest.raises(ValueError, match="coordinates do not bijectively match"):
        prepare_scalar_laplace_fem_bem_3d(
            fem,
            mismatched,
            calderon,
        )
