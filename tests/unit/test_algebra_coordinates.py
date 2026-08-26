import jax.numpy as jnp

import phydrax as phx


def test_complex_and_quaternion_coordinate_maps_preserve_public_values():
    complex_algebra = phx.metrix.algebra.ComplexAlgebraSpec()
    complex_coordinates = phx.linalg.AlgebraCoordinatePlan(
        complex_algebra,
        public_storage="native_complex",
        public_dtype=jnp.complex128,
    ).prepare((3,))
    complex_value = jnp.asarray([1.0 + 2.0j, -0.5 + 0.3j, 4.0 - 1.0j])
    quaternion = phx.metrix.algebra.QuaternionAlgebraSpec()
    quaternion_coordinates = phx.linalg.AlgebraCoordinatePlan(
        quaternion,
        public_storage="real_coordinates",
        public_dtype=jnp.float64,
    ).prepare((2,))
    quaternion_value = jnp.arange(8.0).reshape((2, 4))

    assert complex_coordinates.coordinate_space.shape == (2, 3)
    assert jnp.array_equal(
        complex_coordinates.from_real_coordinates(
            complex_coordinates.to_real_coordinates(complex_value)
        ),
        complex_value,
    )
    assert quaternion_coordinates.coordinate_space.shape == (4, 2)
    assert jnp.array_equal(
        quaternion_coordinates.from_real_coordinates(
            quaternion_coordinates.to_real_coordinates(quaternion_value)
        ),
        quaternion_value,
    )
    assert complex_coordinates.defect(complex_value) == 0.0
    assert quaternion_coordinates.defect(quaternion_value) == 0.0


def test_hermitian_spectral_coordinates_implement_shared_map_without_id_change():
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(8),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    coordinates = phx.discretization.HermitianSpectralCoordinates(space)
    state = space.project(jnp.sin(2.0 * jnp.pi * space.axes[0].nodes))
    real = coordinates.to_real_coordinates(state)

    assert isinstance(coordinates, phx.linalg.AbstractRealCoordinateMap)
    assert coordinates.evidence.domain_kind == "constrained_subspace"
    assert coordinates.evidence.norm_relation == "isometry"
    assert (
        coordinates.coordinate_id
        == "640385aa11eec59bde842286dc5f73423b831c33f72b86201e223d6bf9d0d7b3"
    )
    assert jnp.allclose(coordinates.from_real_coordinates(real), state, atol=1e-12)


def test_real_operator_lifts_componentwise_and_complexifies_without_materialization():
    source = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    matrix = jnp.asarray([[2.0, -1.0], [0.5, 3.0]])
    operator = phx.linalg.DenseLinearOperator(matrix, source=source, target=source)
    quaternion = phx.metrix.algebra.QuaternionAlgebraSpec()
    algebra_space = phx.linalg.AlgebraArraySpace((2,), quaternion, dtype=jnp.float64)
    lifted = phx.linalg.lift_real_operator_to_algebra(operator, algebra_space)
    values = jnp.arange(8.0).reshape((2, 4))
    complexified = phx.linalg.complexify_real_operator(
        operator,
        complex_dtype=jnp.complex128,
    )
    complex_value = jnp.asarray([1.0 + 2.0j, -0.5 + 0.3j])

    assert jnp.allclose(lifted(values), matrix @ values)
    assert jnp.allclose(complexified(complex_value), matrix @ complex_value)


def test_cochain_hodge_uses_shared_complexified_real_action():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(2),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    cochain = phx.discretization.StructuredCochainBridge(grid).cochain
    value = jnp.arange(cochain.cell_counts[0], dtype=float) + 1j * jnp.linspace(
        -0.5, 0.5, cochain.cell_counts[0]
    )
    space = cochain.field_spaces[0].vector_space
    expected = space.riesz(jnp.real(value)) + 1j * space.riesz(jnp.imag(value))

    assert jnp.array_equal(cochain.apply_hodge(0, value), expected)
    assert jnp.allclose(cochain.solve_hodge(0, expected), value)
