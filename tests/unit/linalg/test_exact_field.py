from fractions import Fraction
from math import lcm

import numpy as np
import pytest

from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.linalg._exact import (
    determinant_exact,
    ExactInconsistentSystemError,
    ExactRankDeficientSystemError,
    factor_exact,
    factor_exact_rational_algebraic,
    factor_exact_sparse,
    matmul_exact,
    solve_exact,
    solve_exact_sparse,
)


def test_exact_solve_pivots_on_first_exact_nonzero_row() -> None:
    matrix = (
        (Fraction(0), Fraction(2)),
        (Fraction(1), Fraction(3)),
    )
    right = ((Fraction(1),), (Fraction(2),))

    assert solve_exact(matrix, right) == [[Fraction(1, 2)], [Fraction(1, 2)]]
    assert determinant_exact(matrix) == Fraction(-2)


def test_prepared_rational_factorization_solves_algebraic_multiple_rhs() -> None:
    root = RadicalElement.sqrt(Fraction(5))
    matrix = (
        (Fraction(1), Fraction(1)),
        (Fraction(1), Fraction(-1)),
    )
    factorization = factor_exact_rational_algebraic(matrix, RadicalElement)

    solution = factorization.solve(
        (
            (root, Fraction(0)),
            (Fraction(1), Fraction(2)),
        )
    )

    assert solution == [
        [(root + Fraction(1)) / 2, Fraction(1)],
        [(root - Fraction(1)) / 2, Fraction(-1)],
    ]
    assert factorization.evidence == factor_exact(matrix).evidence
    assert factorization.solve(((Fraction(1),), (Fraction(0),))) == [
        [Fraction(1, 2)],
        [Fraction(1, 2)],
    ]
    with pytest.raises(ValueError, match="singular"):
        factor_exact_rational_algebraic(
            (
                (Fraction(1), Fraction(2)),
                (Fraction(2), Fraction(4)),
            ),
            RadicalElement,
        )


def test_exact_solve_refuses_singular_and_malformed_systems() -> None:
    with pytest.raises(ValueError, match="singular"):
        solve_exact(
            ((Fraction(1), Fraction(2)), (Fraction(2), Fraction(4))),
            ((Fraction(1),), (Fraction(2),)),
        )
    assert (
        determinant_exact(((Fraction(1), Fraction(2)), (Fraction(2), Fraction(4)))) == 0
    )
    with pytest.raises(ValueError, match="square"):
        solve_exact(((Fraction(1), Fraction(2)),), ((Fraction(1),),))
    with pytest.raises(ValueError, match="rows"):
        solve_exact(((Fraction(1),),), ())


def test_exact_empty_determinant_is_the_multiplicative_identity() -> None:
    assert determinant_exact(()) == Fraction(1)


def test_exact_sparse_solve_reuses_one_full_rank_factorization_for_many_rhs() -> None:
    matrix = (
        {0: Fraction(1)},
        {1: Fraction(1)},
        {0: Fraction(1), 1: Fraction(1)},
    )
    right = (
        (Fraction(2), Fraction(0)),
        (Fraction(0), Fraction(2)),
        (Fraction(2), Fraction(2)),
    )

    solution, evidence = solve_exact_sparse(matrix, right, 2, row_ids=(40, 41, 42))

    assert solution == [[Fraction(2), Fraction(0)], [Fraction(0), Fraction(2)]]
    assert evidence.rank == 2
    assert evidence.pivot_rows == (40, 41)
    assert evidence.pivot_columns == (0, 1)


def test_exact_sparse_solve_retains_inconsistency_residual() -> None:
    with pytest.raises(ExactInconsistentSystemError) as captured:
        solve_exact_sparse(
            ({0: Fraction(1)}, {0: Fraction(1)}),
            ((Fraction(1),), (Fraction(2),)),
            1,
            row_ids=(7, 8),
        )

    error = captured.value
    assert error.evidence.rank == 1
    assert error.evidence.pivot_rows == (7,)
    assert error.transformed_row_indices == (1,)
    assert error.residuals == ((Fraction(1),),)
    assert error.transformed_row_ids == (8,)


def test_exact_sparse_solve_reports_rank_deficiency_without_least_squares() -> None:
    with pytest.raises(ExactRankDeficientSystemError) as captured:
        solve_exact_sparse(
            ({0: Fraction(1), 1: Fraction(2)}, {0: Fraction(2), 1: Fraction(4)}),
            ((Fraction(1),), (Fraction(2),)),
            2,
        )

    assert captured.value.evidence.rank == 1
    assert captured.value.evidence.pivot_columns == (0,)


def test_exact_sparse_rank_factorization_returns_canonical_nullspace() -> None:
    matrix = (
        {0: Fraction(1), 1: Fraction(2)},
        {0: Fraction(2), 1: Fraction(4)},
        {1: Fraction(1), 2: Fraction(1)},
    )

    factorization = factor_exact_sparse(matrix, 3, row_ids=(40, 41, 42))

    assert factorization.evidence.right_count == 0
    assert factorization.evidence.rank == 2
    assert factorization.evidence.pivot_rows == (42, 40)
    assert factorization.evidence.pivot_columns == (2, 0)
    assert factorization.nullspace_basis() == ((Fraction(-2), Fraction(1), Fraction(-1)),)
    assert factor_exact_sparse(matrix, 3, row_ids=(40, 41, 42)).evidence == (
        factorization.evidence
    )
    assert (
        factor_exact_sparse(({0: Fraction(1)}, {1: Fraction(1)}), 2).nullspace_basis()
        == ()
    )


def test_exact_sparse_rank_factorization_preserves_radical_field() -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    root_two = RadicalElement.sqrt(Fraction(2))
    matrix = (
        {0: root_two, 1: root_two},
        {0: 2 * root_two, 1: 2 * root_two},
    )

    factorization = factor_exact_sparse(matrix, 2)

    assert factorization.evidence.rank == 1
    assert factorization.nullspace_basis() == ((Fraction(-1), Fraction(1)),)


def test_exact_sparse_solve_uses_algebraic_field_pivots() -> None:
    from phydrax.discretization._exact_algebraic_geometry import (
        ExactScalar,
        RadicalElement,
    )

    root_two = RadicalElement.sqrt(Fraction(2))
    matrix: tuple[dict[int, ExactScalar], ...] = (
        {0: root_two, 1: Fraction(1)},
        {0: Fraction(1), 1: Fraction(1)},
    )
    right: tuple[tuple[ExactScalar, ...], ...] = ((Fraction(1),), (Fraction(0),))
    solution, evidence = solve_exact_sparse(
        matrix,
        right,
        2,
    )

    assert evidence.rank == 2
    assert solution == [[root_two + 1], [-(root_two + 1)]]


def test_exact_sparse_solve_rejects_inexact_float_arithmetic() -> None:
    with pytest.raises(TypeError, match="exact field scalars"):
        solve_exact_sparse(({0: 0.5},), ((Fraction(1),),), 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="exact field scalars"):
        solve_exact_sparse(({0: np.float32(0.5)},), ((Fraction(1),),), 1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="exact field scalars"):
        solve_exact_sparse(({0: Fraction(1)},), ((np.float64(1.0),),), 1)  # ty: ignore[invalid-argument-type]


def test_prepared_exact_factor_reuses_lu_for_multiple_right_hand_sides() -> None:
    matrix = (
        (Fraction(0), Fraction(2)),
        (Fraction(1), Fraction(3)),
    )
    right = (
        (Fraction(1), Fraction(0)),
        (Fraction(2), Fraction(2)),
    )
    factorization = factor_exact(matrix)

    assert factorization.determinant == Fraction(-2)
    assert factorization.evidence.rank == 2
    assert factorization.evidence.pivot_rows == (1, 0)
    assert factorization.solve(right) == [
        [Fraction(1, 2), Fraction(2)],
        [Fraction(1, 2), Fraction(0)],
    ]
    with pytest.raises(ValueError, match="singular"):
        factor_exact(((Fraction(1), Fraction(2)), (Fraction(2), Fraction(4))))


def test_prepared_exact_factor_preserves_radical_field_solves() -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    root_two = RadicalElement.sqrt(Fraction(2))
    factorization = factor_exact(((root_two, Fraction(1)), (Fraction(1), root_two)))

    assert factorization.determinant == Fraction(1)
    assert factorization.evidence.rank == 2
    assert factorization.solve(
        ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    ) == [[root_two, Fraction(-1)], [Fraction(-1), root_two]]


def _random_rational_system(
    size: int, width: int, seed: int
) -> tuple[list[list[Fraction]], list[list[Fraction]]]:
    generator = np.random.default_rng(seed)
    numerators = generator.integers(-40, 41, (size, size)).tolist()
    denominators = generator.integers(1, 30, (size, size)).tolist()
    matrix = [
        [Fraction(n, d) for n, d in zip(row, column, strict=True)]
        for row, column in zip(numerators, denominators, strict=True)
    ]
    right_numerators = generator.integers(-9, 10, (size, width)).tolist()
    right_denominators = generator.integers(1, 7, (size, width)).tolist()
    right = [
        [Fraction(n, d) for n, d in zip(row, column, strict=True)]
        for row, column in zip(right_numerators, right_denominators, strict=True)
    ]
    return matrix, right


@pytest.mark.parametrize(("size", "width", "seed"), [(24, 3, 0), (48, 48, 1), (64, 5, 2)])
def test_rational_lifting_equals_fraction_elimination(
    size: int, width: int, seed: int
) -> None:
    from phydrax.linalg._exact import _solve_rational_lifting

    matrix, right = _random_rational_system(size, width, seed)
    reference = factor_exact(matrix).solve(right)

    lifted = _solve_rational_lifting(matrix, right)

    assert lifted == reference
    assert all(type(value) is Fraction for row in lifted or () for value in row)
    assert solve_exact(matrix, right) == reference


def test_rational_lifting_solves_a_504_row_system_exactly() -> None:
    from phydrax.linalg._exact import _solve_rational_lifting

    size = 504
    generator = np.random.default_rng(504)
    integers = generator.integers(-3, 4, (size, size))
    integers[np.arange(size), np.arange(size)] = 4 * size
    matrix = [[Fraction(int(value)) for value in row] for row in integers]
    right = [
        [Fraction(int(row), 7), Fraction(int(row) % 5 - 2, 3)] for row in range(size)
    ]

    solution = _solve_rational_lifting(matrix, right)

    assert solution is not None
    integer_rows = integers.tolist()
    for column in range(2):
        # Clear the column's common denominator; the residual is then integral.
        denominator = lcm(*(row[column].denominator for row in solution))
        numerators = [
            row[column].numerator * (denominator // row[column].denominator)
            for row in solution
        ]
        for row in range(size):
            assert (
                Fraction(
                    sum(
                        value * numerator
                        for value, numerator in zip(
                            integer_rows[row], numerators, strict=True
                        )
                    ),
                    denominator,
                )
                == right[row][column]
            )


def test_rational_lifting_changes_prime_when_singular_modulo_the_first() -> None:
    from phydrax.linalg._exact import _lifting_primes, _solve_rational_lifting

    size = 24
    prime = _lifting_primes(size)[0]
    generator = np.random.default_rng(24)
    integers = generator.integers(-5, 6, (size, size)).tolist()
    integers[0][0] = 0
    base = determinant_exact([[Fraction(value) for value in row] for row in integers])
    cofactor = determinant_exact(
        [[Fraction(value) for value in row[1:]] for row in integers[1:]]
    )
    assert base.denominator == 1
    assert cofactor.denominator == 1
    assert cofactor.numerator % prime
    # det = base + a00 * cofactor; pick a00 with det = 0 mod prime, det != 0.
    integers[0][0] = -base.numerator * pow(cofactor.numerator, -1, prime) % prime
    matrix = [[Fraction(value) for value in row] for row in integers]
    determinant = determinant_exact(matrix)
    assert determinant != 0
    assert determinant.numerator % prime == 0
    identity = [
        [Fraction(int(row == column)) for column in range(size)] for row in range(size)
    ]

    assert _solve_rational_lifting(matrix, identity) == factor_exact(matrix).solve(
        identity
    )


def test_rational_lifting_defers_singular_systems_to_exact_elimination() -> None:
    from phydrax.linalg._exact import _solve_rational_lifting

    matrix, right = _random_rational_system(32, 2, 3)
    matrix[5] = [2 * value for value in matrix[3]]

    assert _solve_rational_lifting(matrix, right) is None
    with pytest.raises(ValueError, match="singular"):
        solve_exact(matrix, right)


def test_rational_lifting_matches_fraction_elimination_on_form_basis_pairing() -> None:
    from phydrax.discretization.fem._form_elements import FormBasis
    from phydrax.linalg._exact import _solve_rational_lifting

    basis = FormBasis(3, 1, 3, "full", "untwisted")
    generators = basis.exact_generators
    moments = basis.exact_functional_moments
    size = len(moments)
    pairing = [
        [
            sum(
                (
                    moments[dof][monomial][component]
                    * generators[monomial][component][column]
                    for monomial in range(len(generators))
                    for component in range(len(generators[monomial]))
                ),
                Fraction(0),
            )
            for column in range(size)
        ]
        for dof in range(size)
    ]
    identity = [
        [Fraction(int(row == column)) for column in range(size)] for row in range(size)
    ]

    assert size >= 24
    assert _solve_rational_lifting(pairing, identity) == factor_exact(pairing).solve(
        identity
    )


def test_exact_matrix_product_equals_fraction_sums() -> None:
    matrix, right = _random_rational_system(17, 5, 7)
    right[3] = [Fraction(0)] * 5
    matrix[2] = [Fraction(0)] * 17

    product = matmul_exact(matrix, right)

    assert product == tuple(
        tuple(
            sum(
                (matrix[row][inner] * right[inner][column] for inner in range(17)),
                Fraction(0),
            )
            for column in range(5)
        )
        for row in range(17)
    )
    assert all(type(value) is Fraction for row in product for value in row)
    with pytest.raises(ValueError, match="incompatible"):
        matmul_exact(matrix, right[:-1])
