#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import os
from pathlib import Path

import pytest

from phydrax._external_runtime import PinnedExecutable
from phydrax.algebraic import (
    ExactSparsePolynomialSystem,
    ExactSymbolicOperation,
    ExactSymbolicStatus,
    execute_exact_symbolic,
    plan_exact_symbolic,
    prepare_exact_symbolic,
    QQ,
    SparsePolynomialSupport,
    UnivariateDiscriminantArguments,
)
from phydrax.backends.macaulay2 import Macaulay2Environment, Macaulay2Provider


pytestmark = pytest.mark.macaulay2_live


def _provider():
    required = (
        "PHYDRAX_MACAULAY2_EXECUTABLE",
        "PHYDRAX_MACAULAY2_SHA256",
        "PHYDRAX_MACAULAY2_VERSION",
    )
    missing = tuple(name for name in required if not os.environ.get(name))
    if missing:
        pytest.skip("live Macaulay2 capability is absent: " + ", ".join(missing))
    path = Path(os.environ["PHYDRAX_MACAULAY2_EXECUTABLE"]).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = os.environ["PHYDRAX_MACAULAY2_SHA256"]
    if digest != expected:
        pytest.fail("The live Macaulay2 executable does not match its explicit pin.")
    executable = PinnedExecutable(
        str(path),
        expected,
        os.environ["PHYDRAX_MACAULAY2_VERSION"],
        os.environ.get("PHYDRAX_MACAULAY2_LICENSE", "GPL-3.0-only"),
        "https://macaulay2.com/",
    )
    return Macaulay2Provider(Macaulay2Environment(executable))


def _quadratic():
    support = SparsePolynomialSupport(
        ("x",),
        ("f",),
        (0, 0),
        ((0,), (2,)),
    )
    return ExactSparsePolynomialSystem(support, ("-1", "1"), QQ)


def test_live_macaulay2_groebner_and_discriminant():
    provider = _provider()
    system = _quadratic()

    basis = prepare_exact_symbolic(
        plan_exact_symbolic(system, ExactSymbolicOperation.GROEBNER_BASIS),
        provider,
    )
    basis_result = execute_exact_symbolic(basis)
    assert basis_result.status is ExactSymbolicStatus.SUCCESS
    assert basis_result.output.domain.domain_id == QQ.domain_id
    assert basis_result.output.variable_count == 1

    discriminant = prepare_exact_symbolic(
        plan_exact_symbolic(
            system,
            ExactSymbolicOperation.DISCRIMINANT_UNIVARIATE,
            UnivariateDiscriminantArguments(0, 0),
        ),
        provider,
    )
    discriminant_result = execute_exact_symbolic(discriminant)
    assert discriminant_result.status is ExactSymbolicStatus.SUCCESS
    assert discriminant_result.output.coefficient_values() == (4,)
