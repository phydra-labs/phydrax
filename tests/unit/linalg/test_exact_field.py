from fractions import Fraction

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
    assert factorization.solve(
        ((Fraction(1),), (Fraction(0),))
    ) == [[Fraction(1, 2)], [Fraction(1, 2)]]
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
    assert factorization.nullspace_basis() == (
        (Fraction(-2), Fraction(1), Fraction(-1)),
    )
    assert factor_exact_sparse(matrix, 3, row_ids=(40, 41, 42)).evidence == (
        factorization.evidence
    )
    assert factor_exact_sparse(
        ({0: Fraction(1)}, {1: Fraction(1)}), 2
    ).nullspace_basis() == ()


def test_exact_sparse_rank_factorization_preserves_radical_field() -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    root_two = RadicalElement.sqrt(Fraction(2))
    matrix = (
        {0: root_two, 1: root_two},
        {0: 2 * root_two, 1: 2 * root_two},
    )

    factorization = factor_exact_sparse(matrix, 2)

    assert factorization.evidence.rank == 1
    assert factorization.nullspace_basis() == (
        (Fraction(-1), Fraction(1)),
    )


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
