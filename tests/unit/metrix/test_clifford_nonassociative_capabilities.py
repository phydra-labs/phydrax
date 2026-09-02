#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_coordinate_metric_clifford_relation_associativity_and_inverse():
    metric = jnp.asarray([[1.0, 0.2], [0.2, -1.0]])
    field = phx.metrix.clifford.CliffordMetricField(
        lambda coordinates: metric + 0.0 * coordinates[0],
        dimension=2,
        signature=(1, 1),
        field_id="nondiagonal-cl11",
    )
    product = phx.metrix.clifford.PreparedCliffordMetricProduct(field)
    basis = jnp.eye(4)
    coordinates = jnp.asarray([0.0])
    for first in range(2):
        for second in range(2):
            left = product(coordinates, basis[1 << first], basis[1 << second])
            right = product(coordinates, basis[1 << second], basis[1 << first])
            expected = jnp.zeros((4,)).at[0].set(2.0 * metric[first, second])
            assert jnp.allclose(left + right, expected)

    a = basis[0] + 0.3 * basis[1] - 0.2 * basis[2]
    b = basis[0] - 0.4 * basis[1]
    c = basis[2] + basis[3]
    assert jnp.allclose(
        product(coordinates, product(coordinates, a, b), c),
        product(coordinates, a, product(coordinates, b, c)),
        atol=1e-6,
    )
    inverse = phx.metrix.clifford.invert_multivector(product, coordinates, a)
    assert bool(inverse.valid)
    assert inverse.left_residual < 1e-6
    assert inverse.right_residual < 1e-6


def test_pin_and_spin_require_unit_versors_with_audited_adjoint_action():
    field = phx.metrix.clifford.CliffordMetricField(
        lambda coordinates: jnp.eye(2) + 0.0 * coordinates[0],
        dimension=2,
        signature=(2, 0),
        field_id="euclidean-cl2-membership",
    )
    product = phx.metrix.clifford.PreparedCliffordMetricProduct(field)
    coordinates = jnp.asarray([0.0])
    basis = jnp.eye(4)

    pin = phx.metrix.clifford.PinElement(product, coordinates, basis[1], parity=1)
    assert bool(pin.valid)
    assert pin.vector_residual < 1e-7
    assert pin.metric_residual < 1e-7

    angle = jnp.asarray(0.37)
    rotor = jnp.cos(angle / 2.0) * basis[0] + jnp.sin(angle / 2.0) * basis[3]
    spin = phx.metrix.clifford.SpinElement(product, coordinates, rotor, tolerance=1e-6)
    assert bool(spin.valid)
    assert spin.norm_residual < 1e-6
    assert spin.vector_residual < 1e-6
    assert spin.metric_residual < 1e-6
    scalar_i = 1j * basis[0].astype(jnp.complex64)
    scalar_phase = jnp.exp(0.23j) * basis[0].astype(jnp.complex64)
    assert not bool(
        phx.metrix.clifford.PinElement(product, coordinates, scalar_i, parity=0).valid
    )
    assert not bool(
        phx.metrix.clifford.SpinElement(product, coordinates, scalar_phase).valid
    )
    assert not bool(
        phx.metrix.clifford.SpinElement(
            product, coordinates.astype(jnp.complex64) + 0.1j, basis[0]
        ).valid
    )

    complex_metric_field = phx.metrix.clifford.CliffordMetricField(
        lambda point: (
            jnp.eye(2, dtype=jnp.complex64)
            + 0.1j * jnp.ones((2, 2), dtype=jnp.complex64)
            + 0.0 * point[0]
        ),
        dimension=2,
        signature=(2, 0),
        field_id="complex-euclidean-cl2-membership",
    )
    complex_metric_product = phx.metrix.clifford.PreparedCliffordMetricProduct(
        complex_metric_field
    )
    assert not bool(
        phx.metrix.clifford.SpinElement(
            complex_metric_product, coordinates, basis[0]
        ).valid
    )

    mixed_invertible = basis[0] + 0.25 * basis[1]
    inverse = phx.metrix.clifford.invert_multivector(
        product, coordinates, mixed_invertible
    )
    assert bool(inverse.valid)
    assert not bool(
        phx.metrix.clifford.PinElement(
            product, coordinates, mixed_invertible, parity=0
        ).valid
    )
    assert not bool(
        phx.metrix.clifford.PinElement(
            product, coordinates, 2.0 * basis[0], parity=0
        ).valid
    )
    assert not bool(
        phx.metrix.clifford.PinElement(product, coordinates, basis[1], parity=0).valid
    )
    assert not bool(phx.metrix.clifford.SpinElement(product, coordinates, basis[1]).valid)


def test_conformal_null_embedding_and_projective_radical_rejection():
    model = phx.metrix.clifford.ConformalCliffordModel(3)
    embedded = model.embed(jnp.asarray([0.2, -0.3, 0.4]))
    assert model.null_residual(embedded) < 1e-6
    projective = phx.metrix.clifford.ProjectiveCliffordModel()
    projective.require_invertible(jnp.asarray([0.0]))
    try:
        projective.require_invertible(jnp.asarray([1.0]))
    except ValueError as error:
        assert "radical" in str(error)
    else:
        raise AssertionError("Projective radical inversion must fail closed.")


def test_unit_octonion_geometry_moufang_brackets_and_algebra_matrix_semantics():
    octonion = phx.metrix.algebra.OctonionAlgebraSpec()
    product = octonion.prepare_product(backend="sparse")
    geometry = phx.metrix.algebra.UnitOctonionStateGeometry(product)
    operations = phx.metrix.algebra.MoufangLoopOperations(geometry)
    basis = jnp.eye(8)
    identity = basis[0]
    assert bool(geometry.contains(identity))
    assert jnp.allclose(operations.multiply(identity, basis[3]), basis[3])
    assert operations.moufang_residual(basis[1], basis[2], basis[4]) < 1e-6
    assert jnp.allclose(
        operations.multiply(basis[3], operations.inverse(basis[3])), identity
    )

    complex_identity = 1j * identity.astype(jnp.complex64)
    assert not bool(geometry.contains(complex_identity))
    with pytest.raises(ValueError, match="real unit octonions"):
        operations.multiply(complex_identity, identity)
    with pytest.raises(ValueError, match="real unit octonions"):
        operations.inverse(complex_identity)
    with pytest.raises(TypeError, match="real octonion coordinates"):
        geometry.project_tangent(identity, 1j * basis[1].astype(jnp.complex64))

    noncanonical = phx.metrix.algebra.MulticomplexAlgebraSpec(3).prepare_product(
        backend="sparse"
    )
    try:
        phx.metrix.algebra.UnitOctonionStateGeometry(noncanonical)
    except TypeError as error:
        assert "canonical audited octonion" in str(error)
    else:
        raise AssertionError("Noncanonical eight-dimensional products must be rejected.")

    witness = octonion.properties.claim("associative").witness
    operands = tuple(basis[octonion.basis_index(value)] for value in witness)
    left = phx.metrix.algebra.BracketingPlan(((0, 1), 2), operand_count=3)
    right = phx.metrix.algebra.BracketingPlan((0, (1, 2)), operand_count=3)
    assert not jnp.allclose(
        left.evaluate(product, operands), right.evaluate(product, operands)
    )

    quaternion = phx.metrix.algebra.QuaternionAlgebraSpec().prepare_product(
        backend="sparse"
    )
    layout = phx.metrix.algebra.AlgebraMatrixLayout(quaternion, 2, 2)
    matrix_product = phx.metrix.algebra.AlgebraMatrixProductPlan(layout, layout)
    matrix = jnp.zeros(layout.shape).at[0, 0, 0].set(1.0).at[1, 1, 0].set(1.0)
    assert jnp.allclose(matrix_product(matrix, matrix), matrix)
    solve = phx.metrix.algebra.algebra_left_solve(
        quaternion,
        jnp.asarray([1.0, 1.0, 0.0, 0.0]),
        jnp.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    assert bool(solve.valid)
    spectrum = phx.metrix.algebra.AlgebraRegularSpectrum(quaternion, jnp.eye(4)[1])
    assert spectrum.side == "left"
    assert bool(spectrum.valid)
