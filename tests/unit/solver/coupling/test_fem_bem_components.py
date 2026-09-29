#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Existing 3-D FEM--BEM products published verbatim as spatial coupled components."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import CellMesh
from phydrax.discretization._topology import EntitySelection
from phydrax.discretization.fem._generic import FiniteElementFieldSpec, FiniteElementPlan
from phydrax.discretization.fem._reference import lagrange_element
from phydrax.geometry import MeshRegion
from phydrax.linalg import (
    ArraySpace,
    DenseLinearOperator,
    DenseLU,
    DifferentiationPolicy,
    FailurePolicy,
    FGMRES,
    LinearSolvePolicy,
    OperatorProperties,
    TolerancePolicy,
    transpose,
)
from phydrax.operators.integral.layer_potential._elasticity3d import (
    ElasticitySingleLayerDP0Galerkin3D,
    ElasticitySingleLayerDP0Policy3D,
    prepare_elasticity_single_layer_dp0_3d,
)
from phydrax.operators.integral.layer_potential._galerkin3d import (
    LaplaceSingleLayerDP0GalerkinPolicy3D,
)
from phydrax.operators.integral.layer_potential._scalar_calderon3d import (
    prepare_scalar_calderon_dp0_3d,
)
from phydrax.solver._fem_bem_scalar import (
    prepare_scalar_laplace_fem_bem_3d,
    PreparedScalarLaplaceFEMBEM3D,
)
from phydrax.solver._fem_bem_vector import (
    ElasticityFEMBEMInterfaceQualification3D,
    prepare_elasticity_fem_bem_3d,
    PreparedElasticityFEMBEM3D,
)
from phydrax.solver.coupling import (
    AbstractSpatialComponent,
    CoupledSolution,
    ElasticityFEMBEMArguments,
    ElasticityFEMBEMComponent,
    ScalarLaplaceFEMBEMArguments,
    ScalarLaplaceFEMBEMComponent,
)


cpl = phx.solver.coupling

# Convex bipyramid of two affine tetrahedra matched to its closed six-face surface
# (the scalar product's own fixture in tests/unit/solver/test_fem_bem_scalar.py).
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
_SOURCE = (0.3, -0.2, 0.5, 0.1, 0.4)
_DIRICHLET_JUMP = tuple(np.linspace(-0.1, 0.2, 6))
_CONORMAL_JUMP = tuple(np.linspace(0.3, -0.2, 6))
_CONDUCTIVITY = (1.5, 0.7)

_TETRA_VERTICES = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
_TETRA_FACES = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))

_DENSE = LinearSolvePolicy(DenseLU())


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


def _reference_stiffness(conductivity: np.ndarray) -> np.ndarray:
    """Dense P1 stiffness of a piecewise-constant conductivity from barycentric gradients."""
    vertices = _BIPYRAMID_VERTICES
    matrix = np.zeros((vertices.shape[0], vertices.shape[0]))
    for cell, kappa in zip(_BIPYRAMID_CELLS, conductivity, strict=True):
        affine = np.concatenate((np.ones((4, 1)), vertices[cell]), axis=1)
        gradients = np.linalg.solve(affine, np.eye(4))[1:].T
        volume = abs(np.linalg.det(affine)) / 6.0
        matrix[np.ix_(cell, cell)] += kappa * volume * gradients @ gradients.T
    return matrix


@pytest.fixture(scope="module")
def scalar_product() -> PreparedScalarLaplaceFEMBEM3D:
    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(_BIPYRAMID_VERTICES), jnp.asarray(_BIPYRAMID_CELLS)
    )
    field = FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 1))
    fem = FiniteElementPlan(mesh, field).prepare(numeric_version="bipyramid")
    surface = MeshRegion(
        jnp.asarray(_BIPYRAMID_VERTICES),
        jnp.asarray(_outward_faces(_BIPYRAMID_VERTICES, _BIPYRAMID_FACES)),
        feature_id="matching-fem-bem-bipyramid",
    )
    calderon = prepare_scalar_calderon_dp0_3d(
        surface,
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
    return prepare_scalar_laplace_fem_bem_3d(
        fem,
        surface,
        calderon,
        interface_selection=EntitySelection.from_subset(
            mesh.topology.entity_sets[2], "boundary"
        ),
        linear=LinearSolvePolicy(
            FGMRES(restart=24, stagnation_iterations=24),
            tolerance=TolerancePolicy(relative=1.0e-13, absolute=0.0),
            differentiation=DifferentiationPolicy("none"),
            failure=FailurePolicy("status"),
        ),
    )


def _elasticity_blocks(
    bem: ElasticitySingleLayerDP0Galerkin3D,
) -> tuple[
    DenseLinearOperator,
    DenseLinearOperator,
    phx.linalg.AbstractLinearOperator,
    ElasticityFEMBEMInterfaceQualification3D,
]:
    # The caller-qualified synthetic blocks of the elasticity product's own tests.
    boundary = bem.weak_operator.source
    dtype = boundary.structure().dtype
    interior = ArraySpace(
        (5,), dtype=dtype, space_id="synthetic-qualified-vector-h1-interior"
    )
    interior_operator = DenseLinearOperator(
        8.0 * jnp.eye(interior.size, dtype=dtype),
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
    trace_matrix = jnp.arange(boundary.size * interior.size, dtype=dtype).reshape(
        (boundary.size, interior.size)
    ) / (50.0 * boundary.size * interior.size)
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
        bem.weak_operator.operator_id,
        orientation="outward-from-fem-interior",
        provider_ids=(
            "synthetic-qualified-vector-H1-provider",
            "exact-signed-interface-map-provider",
        ),
        precision_evidence=(str(dtype), "synthetic-exact-input"),
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


@pytest.fixture(scope="module")
def elasticity_product() -> PreparedElasticityFEMBEM3D:
    bem = prepare_elasticity_single_layer_dp0_3d(
        MeshRegion(
            jnp.asarray(_TETRA_VERTICES, dtype=jnp.float64),
            jnp.asarray(_TETRA_FACES, dtype=jnp.int32),
        ),
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
    interior, trace, conormal, qualification = _elasticity_blocks(bem)
    return prepare_elasticity_fem_bem_3d(
        interior,
        trace,
        conormal,
        bem,
        qualification,
        linear=LinearSolvePolicy(
            DenseLU(),
            differentiation=DifferentiationPolicy("none"),
            failure=FailurePolicy("status"),
        ),
    )


def _solve_alone(
    component: AbstractSpatialComponent,
    arguments: ScalarLaplaceFEMBEMArguments | ElasticityFEMBEMArguments,
    policy: LinearSolvePolicy,
) -> CoupledSolution:
    plan = cpl.CoupledProblemPlan(
        "fem-bem-alone", components=(component,), bindings=(), laws=()
    )
    prepared = cpl.prepare_coupled_problem(plan)
    assert prepared.execution == "linear"
    return cpl.solve_coupled_problem(
        prepared, arguments={component.name: arguments}, policy=policy
    )


def _norm(blocks: tuple[Array, ...]) -> float:
    return float(np.sqrt(sum(np.sum(np.asarray(block) ** 2) for block in blocks)))


def _random_blocks(
    spaces: tuple[phx.linalg.AbstractVectorSpace, ...], seed: int
) -> tuple[Array, ...]:
    generator = np.random.default_rng(seed)
    return tuple(
        space.unflatten(jnp.asarray(generator.standard_normal(space.size)))
        for space in spaces
    )


def test_scalar_component_publishes_the_product_blocks_and_identity(
    scalar_product: PreparedScalarLaplaceFEMBEM3D,
) -> None:
    component = ScalarLaplaceFEMBEMComponent("exterior", scalar_product)
    operator = scalar_product.operator

    assert component.owner_id == scalar_product.prepared_id
    assert component.space == "full"
    assert component.state_space.names == ("interior_field", "exterior_conormal")
    assert component.row_space.names == ("interior_field", "exterior_conormal")
    assert component.state_space.compatible(operator.source)
    assert component.row_space.compatible(operator.target)
    for field, space in zip(component.fields, operator.source.spaces, strict=True):
        assert field.state_block == field.row_block == field.name
        assert field.constraint is None
        assert component.field_space_id(field.name) == space.space_id
        np.testing.assert_array_equal(
            np.asarray(component.lift(field.name, None)), np.zeros(space.size)
        )
    assert component.nullspace(None) is None
    assert component.boundary_impositions() == ()


@pytest.mark.parametrize(
    "jumps_and_conductivity",
    [False, True],
    ids=["unit-conductivity-no-jumps", "cell-conductivity-with-jumps"],
)
def test_scalar_single_component_plan_reproduces_the_product_solve(
    scalar_product: PreparedScalarLaplaceFEMBEM3D, jumps_and_conductivity: bool
) -> None:
    extra = (
        {
            "dirichlet_jump": jnp.asarray(_DIRICHLET_JUMP),
            "conormal_jump": jnp.asarray(_CONORMAL_JUMP),
            "conductivity": jnp.asarray(_CONDUCTIVITY),
        }
        if jumps_and_conductivity
        else {}
    )
    source = jnp.asarray(_SOURCE)
    arguments = ScalarLaplaceFEMBEMArguments(source, **extra)
    component = ScalarLaplaceFEMBEMComponent("exterior", scalar_product)

    # The Calderon blocks are matrix-free actions without dense materialization, so
    # the coupled problem is solved with the product's own FGMRES policy.
    solution = _solve_alone(component, arguments, scalar_product.linear_policy)
    native = scalar_product.solve(source, **extra)
    native_state = (native.interior_coefficients, native.exterior_conormal)
    residual = component.residual(native_state, arguments)
    load = scalar_product.right_hand_side(
        source,
        dirichlet_jump=extra.get("dirichlet_jump"),
        conormal_jump=extra.get("conormal_jump"),
    )

    assert bool(native.valid)
    assert bool(solution.native_successful)
    assert bool(solution.accepted)
    (certificate,) = solution.components
    assert certificate.owner_id == scalar_product.prepared_id
    assert bool(certificate.accepted)
    # Both FGMRES solves (relative tolerance 1e-13) act on the same
    # well-conditioned 11-unknown block; they agree far below 1e-10.
    np.testing.assert_allclose(
        np.asarray(solution.field("exterior", "interior_field")),
        np.asarray(native.interior_coefficients),
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(solution.field("exterior", "exterior_conormal")),
        np.asarray(native.exterior_conormal),
        rtol=0.0,
        atol=1.0e-10,
    )
    # The product's own solution satisfies the published original equations.
    assert _norm(residual) <= 1.0e-11 * _norm(load)


def test_scalar_nonpositive_conductivity_is_not_accepted(
    scalar_product: PreparedScalarLaplaceFEMBEM3D,
) -> None:
    source = jnp.asarray(_SOURCE)
    conductivity = jnp.asarray((1.5, -0.7))
    component = ScalarLaplaceFEMBEMComponent("exterior", scalar_product)

    solution = _solve_alone(
        component,
        ScalarLaplaceFEMBEMArguments(source, conductivity=conductivity),
        scalar_product.linear_policy,
    )

    assert not bool(scalar_product.solve(source, conductivity=conductivity).valid)
    assert not bool(solution.accepted)
    assert not bool(solution.components[0].accepted)


def test_scalar_raw_argument_derivatives_are_refused(
    scalar_product: PreparedScalarLaplaceFEMBEM3D,
) -> None:
    """A conductivity passed as a raw argument has no parameter route: no silent gradient."""
    source = jnp.asarray(_SOURCE)
    component = ScalarLaplaceFEMBEMComponent("exterior", scalar_product)

    def total(conductivity: Array) -> Array:
        solution = _solve_alone(
            component,
            ScalarLaplaceFEMBEMArguments(source, conductivity=conductivity),
            scalar_product.linear_policy,
        )
        return jnp.sum(solution.field("exterior", "interior_field"))

    conductivity = jnp.asarray(_CONDUCTIVITY)
    primal = _solve_alone(
        component,
        ScalarLaplaceFEMBEMArguments(source, conductivity=conductivity),
        scalar_product.linear_policy,
    )

    assert bool(primal.accepted)
    assert primal.derivative_capability.admitted == ()
    with pytest.raises(ValueError, match="admits derivatives only through parameter"):
        jax.grad(total)(conductivity)
    with pytest.raises(ValueError, match="admits derivatives only through parameter"):
        jax.jvp(total, (conductivity,), (jnp.ones_like(conductivity),))


def test_scalar_published_operator_is_the_product_operator_with_exact_transpose(
    scalar_product: PreparedScalarLaplaceFEMBEM3D,
) -> None:
    component = ScalarLaplaceFEMBEMComponent("exterior", scalar_product)
    spaces = scalar_product.operator.source.spaces
    state = _random_blocks(spaces, 3)
    covector = _random_blocks(spaces, 5)
    conductivity = np.asarray(_CONDUCTIVITY)
    published = component.linear_operator(None)
    rebound = component.linear_operator(
        ScalarLaplaceFEMBEMArguments(
            jnp.asarray(_SOURCE), conductivity=jnp.asarray(conductivity)
        )
    )
    product_image = scalar_product.operator.mv(state)
    rebound_image = rebound.mv(state)

    for block, reference in zip(published.mv(state), product_image, strict=True):
        np.testing.assert_array_equal(np.asarray(block), np.asarray(reference))
    # A runtime conductivity changes only the interior stiffness, by the
    # independent barycentric-gradient reference difference.
    stiffness_change = (
        _reference_stiffness(conductivity) - _reference_stiffness(np.ones(2))
    ) @ np.asarray(state[0])
    np.testing.assert_allclose(
        np.asarray(rebound_image[0]) - np.asarray(product_image[0]),
        stiffness_change,
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_array_equal(
        np.asarray(rebound_image[1]), np.asarray(product_image[1])
    )
    for operator in (published, rebound):
        forward = sum(
            float(jnp.vdot(y, image))
            for y, image in zip(covector, operator.mv(state), strict=True)
        )
        backward = sum(
            float(jnp.vdot(image, x))
            for image, x in zip(operator.transpose_mv(covector), state, strict=True)
        )
        assert forward == pytest.approx(backward, rel=1.0e-12, abs=1.0e-13)


def test_elasticity_component_publishes_the_product_blocks_and_identity(
    elasticity_product: PreparedElasticityFEMBEM3D,
) -> None:
    component = ElasticityFEMBEMComponent("elastic", elasticity_product)
    operator = elasticity_product.operator

    assert component.owner_id == elasticity_product.prepared_id
    assert component.state_space.names == ("interior_displacement", "boundary_traction")
    assert component.row_space.names == ("interior_displacement", "boundary_traction")
    assert component.state_space.compatible(operator.source)
    assert component.row_space.compatible(operator.target)
    for field, space in zip(component.fields, operator.source.spaces, strict=True):
        assert field.state_block == field.row_block == field.name
        assert component.field_space_id(field.name) == space.space_id
    assert component.nullspace(None) is None
    assert component.boundary_impositions() == ()


def test_elasticity_single_component_plan_reproduces_the_product_solve(
    elasticity_product: PreparedElasticityFEMBEM3D,
) -> None:
    interior_load = jnp.linspace(
        0.1, 0.5, elasticity_product.interior_operator.source.size
    )
    boundary_load = jnp.linspace(
        -0.2, 0.3, elasticity_product.bem.weak_operator.source.size
    )
    arguments = ElasticityFEMBEMArguments(interior_load, boundary_load)
    component = ElasticityFEMBEMComponent("elastic", elasticity_product)

    solution = _solve_alone(component, arguments, _DENSE)
    native = elasticity_product.solve(interior_load, boundary_load)
    native_state = (native.interior_displacement, native.boundary_traction)
    residual = component.residual(native_state, arguments)

    assert bool(native.valid)
    assert bool(solution.native_successful)
    assert bool(solution.accepted)
    assert solution.components[0].owner_id == elasticity_product.prepared_id
    # Both solves are dense LU factorizations of the same 17-unknown symmetric
    # float64 block; they agree to rounding.
    np.testing.assert_allclose(
        np.asarray(solution.field("elastic", "interior_displacement")),
        np.asarray(native.interior_displacement),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(solution.field("elastic", "boundary_traction")),
        np.asarray(native.boundary_traction),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert _norm(residual) <= 1.0e-12 * _norm((interior_load, boundary_load))


def test_elasticity_published_operator_is_the_product_operator_with_exact_transpose(
    elasticity_product: PreparedElasticityFEMBEM3D,
) -> None:
    component = ElasticityFEMBEMComponent("elastic", elasticity_product)
    spaces = elasticity_product.operator.source.spaces
    state = _random_blocks(spaces, 7)
    covector = _random_blocks(spaces, 9)
    published = component.linear_operator(None)

    for block, reference in zip(
        published.mv(state), elasticity_product.operator.mv(state), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(block), np.asarray(reference))
    forward = sum(
        float(jnp.vdot(y, image))
        for y, image in zip(covector, published.mv(state), strict=True)
    )
    backward = sum(
        float(jnp.vdot(image, x))
        for image, x in zip(published.transpose_mv(covector), state, strict=True)
    )
    assert forward == pytest.approx(backward, rel=1.0e-12, abs=1.0e-13)


def test_components_accept_only_their_prepared_product(
    scalar_product: PreparedScalarLaplaceFEMBEM3D,
    elasticity_product: PreparedElasticityFEMBEM3D,
) -> None:
    with pytest.raises(TypeError, match="PreparedScalarLaplaceFEMBEM3D"):
        ScalarLaplaceFEMBEMComponent("exterior", elasticity_product)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="PreparedElasticityFEMBEM3D"):
        ElasticityFEMBEMComponent("elastic", scalar_product)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="PreparedScalarLaplaceFEMBEM3D"):
        ScalarLaplaceFEMBEMComponent("exterior", scalar_product.operator)  # ty: ignore[invalid-argument-type]


def test_components_refuse_another_argument_kind(
    scalar_product: PreparedScalarLaplaceFEMBEM3D,
    elasticity_product: PreparedElasticityFEMBEM3D,
) -> None:
    scalar = ScalarLaplaceFEMBEMComponent("exterior", scalar_product)
    elastic = ElasticityFEMBEMComponent("elastic", elasticity_product)
    scalar_arguments = ScalarLaplaceFEMBEMArguments(jnp.asarray(_SOURCE))
    elastic_arguments = ElasticityFEMBEMArguments(
        jnp.zeros(elasticity_product.interior_operator.source.size),
        jnp.zeros(elasticity_product.bem.weak_operator.source.size),
    )

    with pytest.raises(TypeError, match="ScalarLaplaceFEMBEMArguments"):
        scalar.linear_operator(elastic_arguments)
    with pytest.raises(TypeError, match="ScalarLaplaceFEMBEMArguments"):
        scalar.residual(scalar.state_space.zeros(), None)
    with pytest.raises(TypeError, match="ElasticityFEMBEMArguments"):
        elastic.linear_operator(scalar_arguments)
    with pytest.raises(TypeError, match="ElasticityFEMBEMArguments"):
        elastic.residual(elastic.state_space.zeros(), scalar_arguments)
    with pytest.raises(TypeError, match="ScalarLaplaceFEMBEMArguments"):
        _solve_alone(scalar, elastic_arguments, scalar_product.linear_policy)
