#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from fractions import Fraction

import numpy as np
import pytest

from phydrax.algebraic._exact import (
    EliminateArguments,
    ExactSparsePolynomialSystem,
    ExactSymbolicOperation,
    GF,
    NormalFormArguments,
    plan_exact_symbolic,
    QQ,
    UnivariateResultantArguments,
    ZZ,
)
from phydrax.algebraic._system import SparsePolynomialSupport


def _support():
    return SparsePolynomialSupport(
        ("x", "y"),
        ("f",),
        np.asarray([0, 0, 0]),
        np.asarray([[0, 0], [0, 1], [2, 0]]),
    )


def test_exact_coefficient_normalization_is_canonical_and_float_free():
    support = _support()
    left = ExactSparsePolynomialSystem(
        support,
        ("-0", Fraction(2, 4), "-6/8"),
        QQ,
    )
    right = ExactSparsePolynomialSystem(support, (0, "1/2", Fraction(-3, 4)), QQ)

    assert left.coefficients == ("0", "1/2", "-3/4")
    assert left.canonical_coefficients == right.coefficients
    assert left.coefficient_values() == (Fraction(0), Fraction(1, 2), Fraction(-3, 4))
    assert left.system_id == right.system_id
    assert left.support is support
    assert not isinstance(left.coefficients, np.ndarray)
    with pytest.raises(ValueError, match="ZZ coefficients"):
        ExactSparsePolynomialSystem(support, (1, "1/2", 3), ZZ)
    with pytest.raises(TypeError, match="floating-point"):
        ExactSparsePolynomialSystem(support, (1, 0.5, 3), QQ)


def test_finite_field_admission_normalizes_residues_and_invertible_rationals():
    field = GF(5)

    assert field.label == "GF(5)"
    assert field.normalize(-1) == "4"
    assert field.normalize("7/3") == "4"
    assert field.parse("-6") == 4
    assert GF(5).domain_id == field.domain_id
    with pytest.raises(ValueError, match="denominator is zero"):
        field.normalize("1/5")
    with pytest.raises(ValueError, match="must be prime"):
        GF(15)


def test_joint_coo_construction_keeps_coefficients_aligned_after_canonicalization():
    exact = ExactSparsePolynomialSystem.from_coo(
        ("x",),
        ("f",),
        np.asarray([0, 0, 0]),
        np.asarray([[2], [0], [1]]),
        ("2", "3", "5"),
        ZZ,
    )

    assert np.asarray(exact.support.exponents).tolist() == [[0], [1], [2]]
    assert exact.coefficients == ("3", "5", "2")


def test_operation_plans_reject_mismatched_or_nonunivariate_inventory():
    system = ExactSparsePolynomialSystem(_support(), (1, 2, 3), QQ)
    dividend = ExactSparsePolynomialSystem(
        SparsePolynomialSupport(("x", "y"), ("g",), [0], [[1, 0]]),
        (1,),
        QQ,
    )

    normal = plan_exact_symbolic(
        system,
        ExactSymbolicOperation.NORMAL_FORM,
        NormalFormArguments(dividend),
    )
    assert normal.arguments.polynomial.system_id == dividend.system_id
    eliminate = plan_exact_symbolic(
        system,
        ExactSymbolicOperation.ELIMINATE,
        EliminateArguments((0,)),
    )
    assert eliminate.arguments.variable_indices == (0,)
    with pytest.raises(ValueError, match="exactly one system variable"):
        plan_exact_symbolic(
            system,
            ExactSymbolicOperation.RESULTANT_UNIVARIATE,
            UnivariateResultantArguments((0, 1)),
        )
