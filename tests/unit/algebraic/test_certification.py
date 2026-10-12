#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _square_root_system() -> Any:
    return phx.algebraic.SparsePolynomialSystem.from_coo(
        ("x",),
        ("equation",),
        (0, 0),
        ((0,), (2,)),
        jnp.asarray((-2.0, 1.0)),
    )


def test_certification_scenario_1() -> None:
    certificate = phx.algebraic.smale_alpha_certificate(
        _square_root_system(), jnp.asarray([jnp.sqrt(2.0)])
    )

    assert certificate.approximate_root
    assert certificate.alpha_upper < 1.0e-10
    assert certificate.beta < 1.0e-10
    certificate = phx.algebraic.krawczyk_certificate(
        _square_root_system(),
        jnp.asarray([1.4142]),
        jnp.asarray([0.01]),
    )

    assert certificate.unique_root
    assert certificate.image_lower[0] > certificate.center[0] - certificate.radius[0]
    assert certificate.image_upper[0] < certificate.center[0] + certificate.radius[0]
    square_roots = phx.algebraic.isolate_univariate_real_roots((-2, 0, 1))
    repeated = phx.algebraic.isolate_univariate_real_roots((1, -2, 1))

    assert len(square_roots) == 2
    assert square_roots[0].upper < 0 < square_roots[1].lower
    assert repeated == (phx.algebraic.ExactRealRootInterval(Fraction(1), Fraction(1), 2),)


def test_krawczyk_dense_admission_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    system = _square_root_system()
    center = np.asarray([1.4142])
    radius = np.asarray([0.01])
    required = 8 * (10 + 12)
    certificate = phx.algebraic.krawczyk_certificate(
        system, center, radius, maximum_dense_workspace_bytes=required
    )
    default = phx.algebraic.krawczyk_certificate(system, center, radius)
    assert certificate.dense_workspace_bytes == required
    assert certificate.maximum_dense_workspace_bytes == required
    assert certificate.certificate_id == default.certificate_id
    inverse = 1.0 / (2 * center[0])
    image_center = center[0] - inverse * (center[0] ** 2 - 2)
    image_radius = 2 * inverse * radius[0] ** 2
    np.testing.assert_allclose(certificate.image_lower, [image_center - image_radius])
    np.testing.assert_allclose(certificate.image_upper, [image_center + image_radius])

    def refuse_dense_jacobian(_system: Any, _point: Any) -> None:
        raise AssertionError("Dense Jacobian evaluated before workspace admission.")

    monkeypatch.setattr(type(system), "jacobian", refuse_dense_jacobian)
    with pytest.raises(ValueError, match="dense workspace requires 176 bytes"):
        phx.algebraic.krawczyk_certificate(
            system, center, radius, maximum_dense_workspace_bytes=required - 1
        )


@pytest.mark.parametrize("budget", (0, -1, True, 1.5))
def test_krawczyk_refuses_invalid_dense_budget(budget: int | float) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        phx.algebraic.krawczyk_certificate(
            _square_root_system(),
            np.asarray([1.4142]),
            np.asarray([0.01]),
            maximum_dense_workspace_bytes=budget,  # ty: ignore[invalid-argument-type]
        )
