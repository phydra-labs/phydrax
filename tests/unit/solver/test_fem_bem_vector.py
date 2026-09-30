from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    FGMRES,
    LinearDerivativeSolvePolicy,
    LinearSolvePolicy,
    OperatorProperties,
    TolerancePolicy,
    transpose,
)
from phydrax.operators.integral.layer_potential._elasticity3d import (
    ElasticitySingleLayerDP0Policy3D,
    prepare_elasticity_single_layer_dp0_3d,
)
from phydrax.solver._fem_bem_vector import (
    ElasticityFEMBEMInterfaceQualification3D,
    prepare_elasticity_fem_bem_3d,
    vector_fem_bem_support_report,
)


_VERTICES = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)
_FIXED_STRUCTURE = (
    "geometry",
    "interior_operator",
    "kernel",
    "quadrature",
    "trace_maps",
)


def _prepare_elasticity_bem(dtype: Any) -> Any:
    return prepare_elasticity_single_layer_dp0_3d(
        phx.geometry.MeshRegion(jnp.asarray(_VERTICES, dtype=dtype), _FACES),
        shear_modulus=2.0,
        poisson_ratio=0.25,
        policy=ElasticitySingleLayerDP0Policy3D(
            regular_order=3,
            singular_order=3,
            absolute_tolerance=1.0,
            relative_tolerance=1.0,
            max_face_count=8,
            max_matrix_bytes=4096,
            max_preparation_workspace_bytes=1024 * 1024,
        ),
    )


@pytest.fixture(scope="module")
def elasticity_bem() -> Any:
    return _prepare_elasticity_bem(jnp.float32)


@pytest.fixture
def elasticity_bem_f64() -> Iterator[Any]:
    with jax.enable_x64(True):
        yield _prepare_elasticity_bem(jnp.float64)


def _qualified_blocks(
    elasticity_bem: Any, *, orientation: Any = "outward-from-fem-interior"
) -> Any:
    boundary = elasticity_bem.weak_operator.source
    interior = ArraySpace(
        (5,),
        dtype=boundary.structure().dtype,
        space_id="synthetic-qualified-vector-h1-interior",
    )
    interior_operator = DenseLinearOperator(
        8.0 * jnp.eye(interior.size, dtype=boundary.structure().dtype),
        source=interior,
        target=interior,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id="synthetic-qualified-symmetric-elasticity-interior",
    )
    trace_matrix = jnp.arange(
        boundary.size * interior.size, dtype=boundary.structure().dtype
    ).reshape((boundary.size, interior.size)) / (50.0 * boundary.size * interior.size)
    trace_operator = DenseLinearOperator(
        trace_matrix,
        source=interior,
        target=boundary,
        operator_id="synthetic-exact-signed-calderon-trace",
    )
    conormal_operator = transpose(trace_operator)
    qualification = ElasticityFEMBEMInterfaceQualification3D(
        "synthetic-matching-tetrahedron-interface",
        interior.space_id,
        boundary.space_id,
        trace_operator.operator_id,
        conormal_operator.operator_id,
        elasticity_bem.weak_operator.operator_id,
        orientation=orientation,
        provider_ids=(
            "synthetic-qualified-vector-H1-provider",
            "exact-signed-interface-map-provider",
        ),
        precision_evidence=(str(boundary.structure().dtype), "synthetic-exact-input"),
        resource_evidence=(
            ("interior_unknowns", interior.size),
            ("interface_unknowns", boundary.size),
        ),
        error_evidence=(
            "synthetic maps are supplied exactly in canonical coordinates",
            "no continuum discretization error estimate",
        ),
    )
    return interior_operator, trace_operator, conormal_operator, qualification


def test_symmetric_elasticity_block_executes_and_solves_manufactured_state(
    elasticity_bem: Any,
) -> None:
    blocks = _qualified_blocks(elasticity_bem)
    linear = LinearSolvePolicy(
        DenseLU(),
        differentiation=DifferentiationPolicy("none"),
        failure=FailurePolicy("status"),
    )
    prepared = prepare_elasticity_fem_bem_3d(
        *blocks[:3],
        # ty: ignore[too-many-positional-arguments]
        elasticity_bem,
        blocks[3],
        linear=linear,
    )
    interior_exact = jnp.asarray([0.2, -0.1, 0.3, -0.25, 0.15])
    traction_exact = jnp.linspace(
        -0.15,
        0.2,
        elasticity_bem.weak_operator.source.size,
        dtype=interior_exact.dtype,
    )
    right_hand_side = prepared.operator.mv((interior_exact, traction_exact))

    result = prepared.solve(*right_hand_side)

    assert bool(result.valid)
    assert float(result.relative_block_residual) < 2.0e-5
    assert float(result.symmetry_defect) < 1.0e-6
    assert jnp.allclose(
        result.interior_displacement, interior_exact, rtol=2e-4, atol=2e-5
    )
    assert jnp.allclose(result.boundary_traction, traction_exact, rtol=2e-4, atol=2e-5)
    assert result.normal_convention == "outward-from-fem-interior"
    assert result.continuum_certified is False
    assert "Costabel symmetric" in result.formulation


def test_symmetric_elasticity_block_preserves_exact_transpose(
    elasticity_bem: Any,
) -> None:
    blocks = _qualified_blocks(elasticity_bem)
    # ty: ignore[too-many-positional-arguments]
    prepared = prepare_elasticity_fem_bem_3d(*blocks[:3], elasticity_bem, blocks[3])
    vector = (
        jnp.linspace(-0.3, 0.4, prepared.interior_operator.source.size),
        jnp.linspace(0.2, -0.1, prepared.bem.weak_operator.source.size),
    )

    forward = prepared.operator.mv(vector)
    transposed = prepared.operator.transpose_mv(vector)

    assert prepared.operator.properties.certifies("self_adjoint")
    assert jnp.allclose(transposed[0], forward[0], rtol=1e-6, atol=1e-7)
    assert jnp.allclose(transposed[1], forward[1], rtol=1e-6, atol=1e-7)


def test_elasticity_coupling_rejects_space_mismatch(elasticity_bem: Any) -> None:
    interior_operator, trace_operator, _, qualification = _qualified_blocks(
        elasticity_bem
    )
    boundary = elasticity_bem.weak_operator.source
    wrong_boundary = ArraySpace(
        boundary.structure().shape,
        dtype=boundary.structure().dtype,
        space_id="wrong-interface-space-with-the-same-dimension",
    )
    wrong_trace = DenseLinearOperator(
        trace_operator.matrix,
        source=interior_operator.source,
        target=wrong_boundary,
        operator_id=trace_operator.operator_id,
    )
    wrong_conormal = transpose(wrong_trace)
    wrong_qualification = ElasticityFEMBEMInterfaceQualification3D(
        qualification.interface_id,
        qualification.interior_space_id,
        wrong_boundary.space_id,
        wrong_trace.operator_id,
        wrong_conormal.operator_id,
        qualification.bem_operator_id,
        orientation=qualification.orientation,
        provider_ids=qualification.provider_ids,
        precision_evidence=qualification.precision_evidence,
        resource_evidence=qualification.resource_evidence,
        error_evidence=qualification.error_evidence,
    )

    with pytest.raises(ValueError, match="Boundary space"):
        prepare_elasticity_fem_bem_3d(
            interior_operator,
            wrong_trace,
            wrong_conormal,
            elasticity_bem,
            wrong_qualification,
        )


def test_elasticity_coupling_rejects_orientation_mismatch(elasticity_bem: Any) -> None:
    blocks = _qualified_blocks(elasticity_bem, orientation="outward-from-bem-exterior")

    with pytest.raises(ValueError, match="orientation"):
        # ty: ignore[too-many-positional-arguments]
        prepare_elasticity_fem_bem_3d(*blocks[:3], elasticity_bem, blocks[3])


def test_elasticity_coupling_rejects_unpaired_conormal_map(elasticity_bem: Any) -> None:
    interior_operator, trace_operator, _, qualification = _qualified_blocks(
        elasticity_bem
    )
    independent_conormal = DenseLinearOperator(
        trace_operator.matrix.T,
        source=trace_operator.target,
        target=trace_operator.source,
        operator_id="independent-conormal-copy",
    )
    independent_qualification = ElasticityFEMBEMInterfaceQualification3D(
        qualification.interface_id,
        qualification.interior_space_id,
        qualification.boundary_space_id,
        qualification.trace_operator_id,
        independent_conormal.operator_id,
        qualification.bem_operator_id,
        orientation=qualification.orientation,
        provider_ids=qualification.provider_ids,
        precision_evidence=qualification.precision_evidence,
        resource_evidence=qualification.resource_evidence,
        error_evidence=qualification.error_evidence,
    )

    with pytest.raises(ValueError, match="exact PHYDRAX algebraic transpose pair"):
        prepare_elasticity_fem_bem_3d(
            interior_operator,
            trace_operator,
            independent_conormal,
            elasticity_bem,
            independent_qualification,
        )


def test_vector_support_report_classifies_maxwell_routes() -> None:
    report = vector_fem_bem_support_report()
    assert report.matching_maxwell_supported
    assert report.nonmatching_maxwell_supported
    assert report.caller_built_periodic_maxwell_supported
    assert not report.automatic_periodic_maxwell_supported
    assert not report.continuum_certified


def _rhs_only_policy(
    *, relative: float = 1.0e-12, max_steps: int | None = None
) -> LinearSolvePolicy:
    return LinearSolvePolicy(
        FGMRES(restart=30, stagnation_iterations=30),
        tolerance=TolerancePolicy(relative=relative, absolute=0.0, max_steps=max_steps),
        differentiation=DifferentiationPolicy("rhs-only"),
        derivative_solve=LinearDerivativeSolvePolicy(
            relative_tolerance=1.0e-12, absolute_tolerance=0.0
        ),
        failure=FailurePolicy("status"),
    )


def _prepare(elasticity_bem: Any, linear: LinearSolvePolicy) -> Any:
    blocks = _qualified_blocks(elasticity_bem)
    return prepare_elasticity_fem_bem_3d(
        *blocks[:3],
        # ty: ignore[too-many-positional-arguments]
        elasticity_bem,
        blocks[3],
        linear=linear,
    )


def _loads(prepared: Any) -> tuple[Any, Any]:
    dtype = prepared.bem.weak_operator.source.structure().dtype
    return (
        jnp.linspace(0.1, 0.5, prepared.interior_operator.source.size, dtype=dtype),
        jnp.linspace(-0.2, 0.3, prepared.bem.weak_operator.source.size, dtype=dtype),
    )


def _directions(prepared: Any, seed: int) -> tuple[Any, Any]:
    generator = np.random.default_rng(seed)
    return (
        jnp.asarray(generator.standard_normal(prepared.interior_operator.source.size)),
        jnp.asarray(generator.standard_normal(prepared.bem.weak_operator.source.size)),
    )


def _solution(prepared: Any, interior_load: Any, boundary_load: Any) -> Any:
    result = prepared.solve(interior_load, boundary_load)
    return result.interior_displacement, result.boundary_traction


@pytest.mark.parametrize("argument", ["interior_load", "boundary_load"])
def test_rhs_only_load_jvp_matches_central_finite_differences(
    elasticity_bem_f64: Any, argument: str
) -> None:
    prepared = _prepare(elasticity_bem_f64, _rhs_only_policy())
    loads = _loads(prepared)
    interior_direction, boundary_direction = _directions(prepared, 7)
    tangent = (
        (interior_direction, jnp.zeros_like(boundary_direction))
        if argument == "interior_load"
        else (jnp.zeros_like(interior_direction), boundary_direction)
    )
    result = prepared.solve(*loads)

    _, derivative = jax.jvp(
        lambda interior, boundary: _solution(prepared, interior, boundary),
        loads,
        tangent,
    )
    # The solution map is affine in the loads; the central difference of two
    # converged primal solves is exact up to the solve tolerance.
    step = 1.0e-2
    plus = _solution(prepared, loads[0] + step * tangent[0], loads[1] + step * tangent[1])
    minus = _solution(
        prepared, loads[0] - step * tangent[0], loads[1] - step * tangent[1]
    )

    assert prepared.derivative_capability.admits(argument)
    assert bool(result.valid) and bool(result.derivative_valid)
    for actual, upper, lower in zip(derivative, plus, minus, strict=True):
        reference = (np.asarray(upper) - np.asarray(lower)) / (2.0 * step)
        np.testing.assert_allclose(np.asarray(actual), reference, rtol=1e-7, atol=1e-8)


def test_rhs_only_load_vjp_is_the_transpose_of_the_jvp(
    elasticity_bem_f64: Any,
) -> None:
    prepared = _prepare(elasticity_bem_f64, _rhs_only_policy())
    loads = _loads(prepared)
    direction = _directions(prepared, 11)
    generator = np.random.default_rng(13)
    cotangent = tuple(
        jnp.asarray(generator.standard_normal(load.shape)) for load in loads
    )

    def solve(interior: Any, boundary: Any) -> Any:
        return _solution(prepared, interior, boundary)

    _, forward = jax.jvp(solve, loads, direction)
    _, pullback = jax.vjp(solve, *loads)
    reverse = pullback(cotangent)
    transpose_pairing = sum(
        float(jnp.vdot(value, tangent))
        for value, tangent in zip(reverse, direction, strict=True)
    )
    forward_pairing = sum(
        float(jnp.vdot(weight, value))
        for weight, value in zip(cotangent, forward, strict=True)
    )

    assert np.isclose(transpose_pairing, forward_pairing, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("mathematical", "'mathematical' is unsupported: .*fixed prepared operators"),
        ("algorithmic", "'algorithmic' is unsupported: the unrolled Krylov"),
    ],
)
def test_unqualified_differentiation_modes_are_refused_at_preparation(
    elasticity_bem: Any, mode: Any, message: str
) -> None:
    linear = LinearSolvePolicy(
        FGMRES(restart=30, stagnation_iterations=30),
        differentiation=DifferentiationPolicy(mode),
    )

    with pytest.raises(ValueError, match=message):
        _prepare(elasticity_bem, linear)


@pytest.mark.parametrize("argnum", [0, 1])
def test_stopped_route_refuses_load_gradients(elasticity_bem: Any, argnum: int) -> None:
    prepared = _prepare(
        elasticity_bem,
        LinearSolvePolicy(DenseLU(), differentiation=DifferentiationPolicy("none")),
    )
    loads = _loads(prepared)
    result = prepared.solve(*loads)

    def objective(interior: Any, boundary: Any) -> Any:
        solved = prepared.solve(interior, boundary)
        return jnp.sum(solved.interior_displacement) + jnp.sum(solved.boundary_traction)

    assert bool(result.valid)
    assert not bool(result.derivative_valid)
    assert not prepared.derivative_capability.admits("interior_load")
    with pytest.raises(ValueError, match="derivative-unsupported"):
        jax.grad(objective, argnums=argnum)(*loads)


def test_rejected_solve_poisons_load_derivatives(elasticity_bem_f64: Any) -> None:
    prepared = _prepare(
        elasticity_bem_f64, _rhs_only_policy(relative=1.0e-14, max_steps=1)
    )
    loads = _loads(prepared)
    direction = _directions(prepared, 17)

    result = prepared.solve(*loads)
    _, derivative = jax.jvp(
        lambda interior, boundary: _solution(prepared, interior, boundary),
        loads,
        direction,
    )

    assert not bool(result.valid)
    assert not bool(result.derivative_valid)
    assert bool(jnp.all(jnp.isfinite(result.interior_displacement)))
    assert all(bool(jnp.all(jnp.isnan(value))) for value in derivative)


def test_derivative_capability_refuses_fixed_prepared_structure(
    elasticity_bem_f64: Any,
) -> None:
    prepared = _prepare(elasticity_bem_f64, _rhs_only_policy())
    capability = prepared.derivative_capability

    assert capability.require("interior_load") is phx.DerivativeSurface.SOLVER_ARGUMENT
    assert capability.require("boundary_load") is phx.DerivativeSurface.SOLVER_ARGUMENT
    assert capability.derivative_contract.route is phx.DerivativeRoute.IMPLICIT
    for name in _FIXED_STRUCTURE:
        assert not capability.admits(name)
    with pytest.raises(ValueError, match="derivative-unsupported.*Kelvin.*Lame"):
        capability.require("kernel")
    with pytest.raises(ValueError, match="derivative-unsupported.*hypersingular"):
        capability.require("interior_operator")
    with pytest.raises(ValueError, match="derivative-unsupported.*trace map"):
        capability.require("trace_maps")
