from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import DerivativeRoute, DerivativeSurface
from phydrax.discretization import CellMesh
from phydrax.discretization._topology import EntitySelection
from phydrax.discretization.fem._generic import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.geometry import MeshRegion
from phydrax.linalg import (
    DifferentiationMode,
    DifferentiationPolicy,
    FailurePolicy,
    FGMRES,
    LinearDerivativeSolvePolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
    TolerancePolicy,
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
def tetra_coupling() -> Any:
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


def _pair(values: Any, covectors: Any) -> Any:
    return sum(jnp.vdot(value, covector) for value, covector in zip(values, covectors))


def test_matching_interface_has_exact_trace_orientation_and_conormal_flux(
    tetra_coupling: Any,
) -> None:
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


def test_interface_and_coupled_block_have_exact_transposes(tetra_coupling: Any) -> None:
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


def test_manufactured_tetrahedron_volume_source_solves_end_to_end(
    tetra_coupling: Any,
) -> None:
    fem, _, calderon, prepared = tetra_coupling
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
    # The discretization owner's P1 stiffness is independent of the product's
    # cell-gradient conductivity route.
    _, owner_stiffness = fem.assemble_field_operators("u", fem.default_runtime)
    manufactured_load = owner_stiffness.mv(
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


def test_mismatched_interface_is_rejected(tetra_coupling: Any) -> None:
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


_BIPYRAMID_VERTICES = np.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    )
)
_BIPYRAMID_CELLS = np.asarray(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=np.int32)
_BIPYRAMID_FACES = np.asarray(
    ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)), dtype=np.int32
)
_POINT = {
    "volume_source_coefficients": jnp.asarray((0.3, -0.2, 0.5, 0.1, 0.4)),
    "dirichlet_jump": jnp.linspace(-0.1, 0.2, 6),
    "conormal_jump": jnp.linspace(0.3, -0.2, 6),
    "conductivity": jnp.asarray((1.5, 0.7)),
}
_DIRECTIONS = {
    "volume_source_coefficients": jnp.linspace(1.0, -1.0, 5),
    "dirichlet_jump": jnp.linspace(-0.5, 0.5, 6),
    "conormal_jump": jnp.linspace(0.2, -0.3, 6),
    "conductivity": jnp.asarray((0.3, -0.4)),
}


def _outward_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    # The bipyramid is convex, so a face is outward when its normal points away
    # from the vertex centroid.
    centroid = np.mean(vertices, axis=0)
    oriented = []
    for face in faces:
        first, second, third = vertices[face]
        normal = np.cross(second - first, third - first)
        outward = normal @ (np.mean(vertices[face], axis=0) - centroid) > 0.0
        oriented.append(face if outward else face[[0, 2, 1]])
    return np.asarray(oriented, dtype=np.int32)


def _reference_stiffness(
    vertices: np.ndarray, cells: np.ndarray, conductivity: np.ndarray
) -> np.ndarray:
    """Dense P1 stiffness of a piecewise-constant conductivity from barycentric gradients."""
    matrix = np.zeros((vertices.shape[0], vertices.shape[0]))
    for cell, kappa in zip(cells, conductivity, strict=True):
        affine = np.concatenate((np.ones((4, 1)), vertices[cell]), axis=1)
        gradients = np.linalg.solve(affine, np.eye(4))[1:].T
        volume = abs(np.linalg.det(affine)) / 6.0
        matrix[np.ix_(cell, cell)] += kappa * volume * gradients @ gradients.T
    return matrix


def _tight_policy(
    mode: DifferentiationMode, *, max_steps: int | None = None
) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        FGMRES(restart=24, stagnation_iterations=24),
        tolerance=TolerancePolicy(relative=1.0e-13, absolute=0.0, max_steps=max_steps),
        differentiation=DifferentiationPolicy(mode),
        derivative_solve=LinearDerivativeSolvePolicy(
            relative_tolerance=1.0e-12, absolute_tolerance=1.0e-14
        ),
        failure=FailurePolicy("status"),
    )


class _Bipyramid:
    def __init__(self) -> None:
        mesh = CellMesh.from_tetrahedra(
            jnp.asarray(_BIPYRAMID_VERTICES), jnp.asarray(_BIPYRAMID_CELLS)
        )
        field = FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1))
        self.fem = FiniteElementPlan(mesh, field).prepare(numeric_version="bipyramid")
        self.surface = MeshRegion(
            jnp.asarray(_BIPYRAMID_VERTICES),
            jnp.asarray(_outward_faces(_BIPYRAMID_VERTICES, _BIPYRAMID_FACES)),
            feature_id="matching-fem-bem-bipyramid",
        )
        self.calderon = prepare_scalar_calderon_dp0_3d(
            self.surface,
            policy=LaplaceSingleLayerDP0GalerkinPolicy3D(
                regular_order=3,
                singular_order=3,
                near_order=3,
                near_ratio=1.0,
                absolute_tolerance=2.0e-3,
                relative_tolerance=2.0e-3,
                target_block_size=4,
                source_block_size=4,
            ),
            numeric_version="bipyramid",
        )
        self.exterior = EntitySelection.from_subset(
            mesh.topology.entity_sets[2], "boundary"
        )

    def prepare(self, mode: DifferentiationMode, *, max_steps: int | None = None) -> Any:
        return prepare_scalar_laplace_fem_bem_3d(
            self.fem,
            self.surface,
            self.calderon,
            interface_selection=self.exterior,
            linear=_tight_policy(mode, max_steps=max_steps),
        )


@pytest.fixture(scope="module")
def bipyramid() -> _Bipyramid:
    return _Bipyramid()


def _observe(prepared: Any, arguments: dict[str, Any]) -> Any:
    result = prepared.solve(
        arguments["volume_source_coefficients"],
        dirichlet_jump=arguments["dirichlet_jump"],
        conormal_jump=arguments["conormal_jump"],
        conductivity=arguments["conductivity"],
    )
    return jnp.concatenate((result.interior_coefficients, result.exterior_conormal))


def _along(prepared: Any, argument: str, step: Any) -> Any:
    arguments = dict(_POINT)
    arguments[argument] = arguments[argument] + step * _DIRECTIONS[argument]
    return _observe(prepared, arguments)


def test_cell_gradient_stiffness_is_the_piecewise_constant_p1_stiffness(
    bipyramid: _Bipyramid,
) -> None:
    prepared = bipyramid.prepare("none")
    conductivity = np.asarray(_POINT["conductivity"])
    vector = np.linspace(-0.4, 0.9, 5)

    np.testing.assert_allclose(
        prepared.stiffness(conductivity).mv(jnp.asarray(vector)),
        _reference_stiffness(_BIPYRAMID_VERTICES, _BIPYRAMID_CELLS, conductivity)
        @ vector,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        prepared.stiffness().mv(jnp.asarray(vector)),
        _reference_stiffness(_BIPYRAMID_VERTICES, _BIPYRAMID_CELLS, np.ones(2)) @ vector,
        atol=1.0e-13,
    )


def test_conductivity_and_transmission_jumps_reproduce_a_manufactured_state(
    bipyramid: _Bipyramid,
) -> None:
    prepared = bipyramid.prepare("none")
    policy = prepared.linear_policy
    interior = jnp.asarray((0.3, -0.2, 0.5, 0.1, 0.4))
    conductivity = np.asarray(_POINT["conductivity"])
    dirichlet_jump = _POINT["dirichlet_jump"]
    conormal_jump = _POINT["conormal_jump"]
    # Exterior row: (1/2 - K)(P0 gamma u - g_D) + V phi = 0.
    conormal = solve(
        LinearSystem(
            bipyramid.calderon.single_layer,
            problem_id="manufactured-fem-bem-jump-conormal",
        ),
        -prepared.exterior_trace_relation.mv(
            prepared.interface.trace(interior) - dirichlet_jump
        ),
        policy=policy,
    )
    assert bool(conormal.successful)
    # Interior row: a_kappa(u, v) - <phi, gamma v> = (f, v) + <g_N, gamma v>.
    load = (
        _reference_stiffness(_BIPYRAMID_VERTICES, _BIPYRAMID_CELLS, conductivity)
        @ np.asarray(interior)
        - prepared.interface.boundary_load(conormal.value)
        - prepared.interface.boundary_load(conormal_jump)
    )
    source = solve(
        LinearSystem(prepared.mass_operator, problem_id="manufactured-fem-bem-source"),
        load,
        policy=policy,
    )
    assert bool(source.successful)

    result = prepared.solve(
        source.value,
        dirichlet_jump=dirichlet_jump,
        conormal_jump=conormal_jump,
        conductivity=conductivity,
    )

    assert bool(result.valid)
    np.testing.assert_allclose(result.interior_coefficients, interior, atol=1.0e-10)
    np.testing.assert_allclose(result.exterior_conormal, conormal.value, atol=1.0e-10)
    np.testing.assert_allclose(
        result.exterior_dirichlet_trace,
        prepared.interface.trace(interior) - dirichlet_jump,
        atol=1.0e-10,
    )
    assert float(result.relative_block_residual) < 1.0e-11
    assert abs(float(result.flux_balance_defect)) < 1.0e-11
    exterior_value, reports = result.evaluate_exterior(jnp.asarray((2.0, 2.0, 2.0)))
    assert jnp.isfinite(exterior_value)
    assert all(bool(report.pde_membership_valid) for report in reports)


@pytest.mark.parametrize("argument", tuple(_POINT), ids=tuple(_POINT))
def test_solution_map_jvp_matches_central_differences(
    bipyramid: _Bipyramid, argument: str
) -> None:
    prepared = bipyramid.prepare("mathematical")
    step = 1.0e-5
    reference = (
        np.asarray(_along(prepared, argument, step))
        - np.asarray(_along(prepared, argument, -step))
    ) / (2.0 * step)

    _, tangent = jax.jvp(lambda value: _along(prepared, argument, value), (0.0,), (1.0,))

    assert prepared.derivative_capability.admits(argument)
    assert np.max(np.abs(reference)) > 1.0e-2
    np.testing.assert_allclose(tangent, reference, rtol=1.0e-6, atol=1.0e-8)


def test_solution_map_vjp_is_the_transpose_of_the_jvp(bipyramid: _Bipyramid) -> None:
    prepared = bipyramid.prepare("mathematical")
    names = tuple(_POINT)

    def observe(*values: Any) -> Any:
        return _observe(prepared, dict(zip(names, values, strict=True)))

    primals = tuple(_POINT[name] for name in names)
    directions = tuple(_DIRECTIONS[name] for name in names)
    _, tangent = jax.jvp(observe, primals, directions)
    weights = jnp.linspace(-1.0, 1.0, tangent.size)
    _, pullback = jax.vjp(observe, *primals)
    cotangents = pullback(weights)

    np.testing.assert_allclose(
        jnp.vdot(weights, tangent),
        sum(
            jnp.vdot(cotangent, direction)
            for cotangent, direction in zip(cotangents, directions, strict=True)
        ),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_rhs_only_admits_data_and_refuses_conductivity_derivatives(
    bipyramid: _Bipyramid,
) -> None:
    rhs_only = bipyramid.prepare("rhs-only")
    mathematical = bipyramid.prepare("mathematical")
    source = _POINT["volume_source_coefficients"]
    conductivity = _POINT["conductivity"]
    weights = jnp.linspace(0.5, -1.0, 5)

    def objective(prepared: Any, values: Any, kappa: Any) -> Any:
        result = prepared.solve(values, conductivity=kappa)
        return result.interior_coefficients @ weights

    capability = rhs_only.derivative_capability
    assert capability.derivative_contract.route is DerivativeRoute.IMPLICIT
    assert capability.require("dirichlet_jump") is DerivativeSurface.SOLVER_ARGUMENT
    np.testing.assert_allclose(
        jax.grad(lambda values: objective(rhs_only, values, conductivity))(source),
        jax.grad(lambda values: objective(mathematical, values, conductivity))(source),
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="derivative-unsupported"):
        jax.grad(lambda kappa: objective(rhs_only, source, kappa))(conductivity)
    with pytest.raises(ValueError, match="mathematical"):
        capability.require("conductivity")


@pytest.mark.parametrize("argument", tuple(_POINT), ids=tuple(_POINT))
def test_none_policy_refuses_runtime_derivatives(
    bipyramid: _Bipyramid, argument: str
) -> None:
    prepared = bipyramid.prepare("none")
    result = prepared.solve(
        _POINT["volume_source_coefficients"],
        conductivity=_POINT["conductivity"],
    )

    assert bool(result.valid)
    assert not bool(result.derivative_valid)
    assert result.derivative_capability.derivative_contract.route is (
        DerivativeRoute.STOPPED
    )
    with pytest.raises(ValueError, match="derivative-unsupported"):
        jax.jvp(lambda value: _along(prepared, argument, value), (0.0,), (1.0,))


def test_algorithmic_differentiation_is_refused_at_preparation(
    bipyramid: _Bipyramid,
) -> None:
    with pytest.raises(ValueError, match="'algorithmic' Krylov derivatives"):
        bipyramid.prepare("algorithmic")


def test_failed_solve_keeps_primal_status_and_poisons_derivatives(
    bipyramid: _Bipyramid,
) -> None:
    prepared = bipyramid.prepare("mathematical", max_steps=1)
    source = _POINT["volume_source_coefficients"]

    result = prepared.solve(source)
    _, tangent = jax.jvp(
        lambda values: prepared.solve(values).interior_coefficients,
        (source,),
        (_DIRECTIONS["volume_source_coefficients"],),
    )

    assert not bool(result.valid)
    assert not bool(result.derivative_valid)
    assert np.all(np.isfinite(result.interior_coefficients))
    assert np.all(np.isnan(tangent))


def test_nonpositive_conductivity_is_not_accepted_or_differentiated(
    bipyramid: _Bipyramid,
) -> None:
    """The linear solve itself succeeds; only the owner refuses the conductivity.

    Every derivative-bearing output, including the raw linear result and the
    residual evidence, must carry the refusal rather than a finite tangent.
    """
    prepared = bipyramid.prepare("mathematical")
    source = _POINT["volume_source_coefficients"]
    conductivity = jnp.asarray((-1.0, 1.0))

    def observe(kappa: Any) -> tuple[Any, ...]:
        result = prepared.solve(source, conductivity=kappa)
        return (
            result.interior_coefficients,
            result.linear_result.value,
            result.relative_block_residual,
            result.interface_equation_defect,
            result.flux_balance_defect,
            result.conormal_mismatch_norm,
        )

    result = prepared.solve(source, conductivity=conductivity)
    _, tangents = jax.jvp(observe, (conductivity,), (_DIRECTIONS["conductivity"],))

    assert bool(result.linear_result.successful)
    assert not bool(result.valid)
    assert not bool(result.derivative_valid)
    for tangent in jax.tree.leaves(tangents):
        assert np.all(np.isnan(tangent))


def test_prepared_structure_derivatives_are_refused(bipyramid: _Bipyramid) -> None:
    capability = bipyramid.prepare("mathematical").derivative_capability

    assert capability.require("conductivity") is DerivativeSurface.PHYSICAL_PARAMETER
    for name in ("geometry", "kernel", "quadrature", "exterior_conductivity"):
        assert not capability.admits(name)
        with pytest.raises(ValueError, match=f"derivative-unsupported.*'{name}'"):
            capability.require(name)


def test_compiled_conductivity_gradient_rebinds_without_host_synchronization(
    bipyramid: _Bipyramid,
) -> None:
    prepared = bipyramid.prepare("mathematical")
    weights = jnp.linspace(-1.0, 1.0, 11)

    def objective(kappa: Any) -> Any:
        arguments = dict(_POINT)
        arguments["conductivity"] = kappa
        return _observe(prepared, arguments) @ weights

    compiled = jax.jit(jax.grad(objective))
    for scale in (1.0, 1.25):
        conductivity = scale * _POINT["conductivity"]
        np.testing.assert_allclose(
            compiled(conductivity),
            jax.grad(objective)(conductivity),
            rtol=1.0e-10,
            atol=1.0e-13,
        )
