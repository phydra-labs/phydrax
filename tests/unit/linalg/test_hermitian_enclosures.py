from fractions import Fraction

import numpy as np
import pytest

from phydrax.linalg._hermitian_spectral import (
    hermitian_exp_enclosure,
    hermitian_log_enclosure,
    HermitianFunctionEnclosure,
)


def test_log_enclosure_covers_repeated_and_nonrepeated_spectra() -> None:
    for matrix in (
        np.diag(np.asarray((2.0, 4.0), dtype=np.float64)),
        2.0 * np.eye(17, dtype=np.float64),
    ):
        enclosure = hermitian_log_enclosure(matrix)
        center = np.asarray(enclosure.matrix, dtype=np.float64)
        expected = np.diag(np.log(np.diag(matrix)))
        assert np.linalg.norm(expected - center, ord=2) <= float(enclosure.error)
        assert enclosure.error >= 0


def test_exp_enclosure_covers_the_nominal_matrix_exponential_and_input_ball() -> None:
    logarithm = HermitianFunctionEnclosure(
        (
            (Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
        Fraction(1, 1000),
    )

    center, error = hermitian_exp_enclosure(logarithm)
    expected = np.diag(np.exp(np.asarray((0.0, 1.0), dtype=np.float64)))

    assert np.linalg.norm(
        expected - np.asarray(center, dtype=np.float64), ord=2
    ) <= float(error)
    assert error >= 0


def test_log_enclosure_refuses_nonpositive_and_unrepresented_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        hermitian_log_enclosure(np.diag(np.asarray((0.0, 1.0), dtype=np.float64)))
    with pytest.raises(ValueError, match="symmetric"):
        hermitian_log_enclosure(np.asarray(((2.0, 0.1), (0.0, 2.0)), dtype=np.float64))
    with pytest.raises(ValueError, match="binary64"):
        hermitian_log_enclosure(np.asarray(((Fraction(1, 3),),), dtype=object))


def test_exp_enclosure_refuses_invalid_radius_and_asymmetry() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        hermitian_exp_enclosure(
            HermitianFunctionEnclosure(((Fraction(0),),), Fraction(-1))
        )
    with pytest.raises(ValueError, match="symmetric"):
        hermitian_exp_enclosure(
            HermitianFunctionEnclosure(
                ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(0))),
                Fraction(0),
            )
        )
