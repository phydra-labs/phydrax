import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.algebraic import PolynomialScaling, SparsePolynomialSupport
from phydrax.conditions import ArrayCodomain, FieldSpec, ProductFieldSpec
from phydrax.enforcement import finite_feature_linear_representation
from phydrax.enforcement._polynomial_representation import (
    casimir_isotypic_blocks,
    DeclaredReductivePolynomialAction,
    extract_equivariant_subspace,
    extract_invariant_subspace,
    FinitePolynomialAction,
    lower_polynomial_enforcement_operator,
    lower_polynomial_linear_representation,
    polynomial_action_constraints,
    scaling_polynomial_action,
    weighted_polynomial_action,
)
from phydrax.linalg import ArraySpace, DenseLinearOperator


def _quadratic_support():
    return SparsePolynomialSupport(
        ("x", "y", "z"),
        ("q",),
        (0, 0, 0, 0, 0, 0),
        (
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ),
    )


def _so3_action(support):
    generators = np.asarray(
        (
            ((0, 0, 0), (0, 0, -1), (0, 1, 0)),
            ((0, 0, 1), (0, 0, 0), (-1, 0, 0)),
            ((0, -1, 0), (1, 0, 0), (0, 0, 0)),
        ),
        dtype="int64",
    )
    structure = np.zeros((3, 3, 3), dtype="int64")
    structure[0, 1, 2] = structure[1, 2, 0] = structure[2, 0, 1] = 1
    structure[1, 0, 2] = structure[2, 1, 0] = structure[0, 2, 1] = -1
    return DeclaredReductivePolynomialAction(
        support,
        generators,
        structure,
        generator_labels=("Jx", "Jy", "Jz"),
    )


def test_so3_quadratics_split_into_scalar_and_traceless_casimir_blocks():
    support = _quadratic_support()
    action = _so3_action(support)

    scalar = extract_invariant_subspace(
        action,
        verification_tolerance=1e-5,
        rejection_tolerance=1e-3,
    )
    blocks = casimir_isotypic_blocks(
        action,
        verification_tolerance=1e-5,
        rejection_tolerance=1e-3,
        labels=("scalar", "traceless-quadratic"),
    )

    assert action.evidence.exact_relations
    assert scalar.dimension == 1
    assert tuple(block.dimension for block in blocks) == (1, 5)
    assert all(block.evidence.candidate_isotypic for block in blocks)
    assert all(not block.evidence.irreducibility_proven for block in blocks)

    exponents = np.asarray(support.exponents)
    diagonal = np.asarray([np.count_nonzero(row == 2) == 1 for row in exponents])
    scalar_vector = np.asarray(scalar.basis[:, 0])
    assert np.allclose(scalar_vector[~diagonal], 0.0, atol=1e-5)
    assert np.allclose(
        np.abs(scalar_vector[diagonal]),
        np.abs(scalar_vector[diagonal][0]),
        atol=1e-5,
    )
    traceless = np.asarray(blocks[1].basis)
    assert np.allclose(np.sum(traceless[diagonal], axis=0), 0.0, atol=1e-5)


def test_finite_cyclic_complex_weight_classes_partition_monomials():
    support = SparsePolynomialSupport(
        ("z",),
        ("p",),
        (0, 0, 0, 0, 0, 0),
        ((0,), (1,), (2,), (3,), (4,), (5,)),
    )
    omega = np.exp(2j * np.pi / 3)
    table = np.fromfunction(lambda left, right: (left + right) % 3, (3, 3), dtype="int64")
    variable_actions = np.asarray([[[omega**power]] for power in range(3)])

    classes = []
    for weight in range(3):
        equation_actions = np.asarray(
            [[[omega ** (weight * power)]] for power in range(3)]
        )
        action = FinitePolynomialAction(
            support,
            variable_actions,
            table,
            equation_actions=equation_actions,
            element_labels=("1", "g", "g2"),
            verification_tolerance=1e-5,
            rejection_tolerance=1e-3,
        )
        block = extract_equivariant_subspace(
            action,
            verification_tolerance=1e-5,
            rejection_tolerance=1e-3,
            label=f"weight-{weight}",
        )
        assert action.evidence.complex_valued
        assert float(action.evidence.maximum_relation_residual) < 1e-5
        assert block.evidence.verified
        assert block.dimension == 2
        classes.append(block)

    projectors = [
        np.asarray(block.basis)
        @ np.conj(np.asarray(block.basis).T)
        @ np.diag(np.asarray(block.metric_weights))
        for block in classes
    ]
    assert np.allclose(sum(projectors), np.eye(support.term_count), atol=1e-5)


def test_invalid_finite_generators_are_rejected_and_near_relations_are_ambiguous():
    support = SparsePolynomialSupport(("x",), ("p",), (0,), ((0,),))
    table = np.asarray(((0, 1), (1, 0)), dtype="int64")

    with pytest.raises(ValueError, match="do not represent"):
        FinitePolynomialAction(
            support,
            np.asarray(([[1.0]], [[2.0]])),
            table,
            verification_tolerance=1e-8,
            rejection_tolerance=1e-4,
        )

    near_action = FinitePolynomialAction(
        support,
        np.asarray(([[1.0]], [[-1.0 + 1e-6]])),
        table,
        verification_tolerance=1e-8,
        rejection_tolerance=1e-4,
    )
    near_constants = extract_invariant_subspace(
        near_action,
        verification_tolerance=1e-8,
        rejection_tolerance=1e-4,
    )

    assert near_action.evidence.status == "ambiguous"
    assert near_constants.dimension == 1
    assert near_constants.evidence.status == "ambiguous"
    with pytest.raises(ValueError, match="verified polynomial basis"):
        lower_polynomial_enforcement_operator(
            DenseLinearOperator(jnp.eye(1)), near_constants
        )


def test_weight_and_scaling_actions_lower_to_equivariant_enforcement_maps():
    support = SparsePolynomialSupport(
        ("x",),
        ("p",),
        (0, 0, 0, 0),
        ((0,), (1,), (2,), (3,)),
    )
    weighted = weighted_polynomial_action(
        support,
        (1,),
        equation_weights=(2,),
    )
    weighted_basis = extract_equivariant_subspace(
        weighted,
        verification_tolerance=1e-5,
        rejection_tolerance=1e-3,
    )
    scaling = PolynomialScaling((2.0,), (4.0,))
    scaled = scaling_polynomial_action(support, scaling)
    scaled_basis = extract_equivariant_subspace(
        scaled,
        verification_tolerance=1e-5,
        rejection_tolerance=1e-3,
    )

    assert weighted_basis.dimension == 1
    assert scaled_basis.dimension == 1
    assert np.argmax(np.abs(np.asarray(weighted_basis.basis[:, 0]))) == 2
    assert np.argmax(np.abs(np.asarray(scaled_basis.basis[:, 0]))) == 2

    dtype = scaled_basis.basis.dtype
    full_map = DenseLinearOperator(jnp.eye(support.term_count, dtype=dtype))
    restricted_map = lower_polynomial_enforcement_operator(full_map, scaled_basis)
    coefficients = restricted_map.mv(jnp.asarray([3.0], dtype=dtype))
    residual = polynomial_action_constraints(scaled).matrix @ coefficients
    assert np.allclose(residual, 0.0, atol=1e-5)

    field_spec = ProductFieldSpec(
        (FieldSpec("p", ArrayCodomain.from_shape((support.term_count,), dtype=dtype)),)
    )
    coefficient_space = ArraySpace((support.term_count,), dtype=dtype)
    base = finite_feature_linear_representation(
        field_spec,
        coefficient_space,
        coefficient_space,
        lambda values: values["p"],
        lambda values, value: {"p": value},
        lambda value: {"p": value},
        lambda bound: None,
        support_ids=(support.support_id,),
    )
    lowered = lower_polynomial_linear_representation(base, scaled_basis)
    synthesized = lowered.synthesize(jnp.asarray([2.0], dtype=dtype))["p"]
    assert np.allclose(
        polynomial_action_constraints(scaled).matrix @ synthesized,
        0.0,
        atol=1e-5,
    )
    assert lowered.coefficient_space.shape == (1,)
    assert not lowered.certificate.round_trip_exact
