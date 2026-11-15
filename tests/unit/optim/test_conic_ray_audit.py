#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.optim._programming._audit import audit_conic_dual_infeasibility_ray


def test_signed_equality_ray_and_normalization() -> None:
    program = phx.optim.ConicProgram(
        None,
        jnp.zeros(1, dtype=jnp.float64),
        jnp.asarray([[1.0], [1.0]], dtype=jnp.float64),
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        phx.optim.ProductCone((phx.optim.ZeroCone(1), phx.optim.NonnegativeCone(1))),
    )
    zero = jnp.zeros(1, dtype=jnp.float64)
    audit = audit_conic_dual_infeasibility_ray(
        program, jnp.asarray([-4.0, 4.0], dtype=jnp.float64), zero, zero, tolerance=1e-6
    )
    assert audit.valid
    assert audit.equality_ray.shape == (0,)
    np.testing.assert_allclose(audit.inequality_ray, [-1.0, 1.0])
    np.testing.assert_allclose(audit.objective, -1.0)
    np.testing.assert_allclose(audit.residual_norm, 0.0)


@pytest.mark.parametrize(
    ("rhs", "candidate", "tolerance", "expected_residual"),
    (
        pytest.param((1.0, 2.0), (1.0, -1.0), 1e-6, 1.0, id="dual-cone"),
        pytest.param((1.0, 0.0), (-1.0, 0.5), 0.1, 0.5, id="stationarity"),
        pytest.param((1.0, 0.0), (-0.1, 0.1), 0.1, 0.0, id="strict-margin"),
    ),
)
def test_original_ray_refusal_boundaries(
    rhs: tuple[float, float],
    candidate: tuple[float, float],
    tolerance: float,
    expected_residual: float,
) -> None:
    program = phx.optim.ConicProgram(
        None,
        jnp.zeros(1, dtype=jnp.float64),
        jnp.asarray([[1.0], [1.0]], dtype=jnp.float64),
        jnp.asarray(rhs, dtype=jnp.float64),
        phx.optim.ProductCone((phx.optim.ZeroCone(1), phx.optim.NonnegativeCone(1))),
    )
    zero = jnp.zeros(1, dtype=jnp.float64)
    audit = audit_conic_dual_infeasibility_ray(
        program,
        jnp.asarray(candidate, dtype=jnp.float64),
        zero,
        zero,
        tolerance=tolerance,
    )
    assert not bool(audit.valid)
    np.testing.assert_allclose(audit.residual_norm, expected_residual, atol=0.0)


def test_nonfinite_ray_retains_failed_evidence() -> None:
    program = phx.optim.ConicProgram(
        None,
        jnp.zeros(1, dtype=jnp.float64),
        jnp.asarray([[1.0]], dtype=jnp.float64),
        jnp.asarray([-1.0], dtype=jnp.float64),
        phx.optim.NonnegativeCone(1),
    )
    zero = jnp.zeros(1, dtype=jnp.float64)
    audit = audit_conic_dual_infeasibility_ray(
        program, jnp.asarray([jnp.inf], dtype=jnp.float64), zero, zero, tolerance=1e-6
    )
    assert not bool(audit.valid)
    assert not bool(jnp.isfinite(audit.objective))


def test_bound_terms_and_shared_normalization() -> None:
    program = phx.optim.ConicProgram(
        None,
        jnp.zeros(1, dtype=jnp.float64),
        jnp.asarray([[1.0]], dtype=jnp.float64),
        jnp.asarray([0.0], dtype=jnp.float64),
        phx.optim.NonnegativeCone(1),
        bounds=phx.optim.Bounds(1.0, jnp.inf),
    )
    audit = audit_conic_dual_infeasibility_ray(
        program,
        jnp.asarray([2.0], dtype=jnp.float64),
        jnp.asarray([2.0], dtype=jnp.float64),
        jnp.zeros(1, dtype=jnp.float64),
        tolerance=1e-6,
    )
    assert audit.valid
    np.testing.assert_allclose(audit.inequality_ray, [1.0])
    np.testing.assert_allclose(audit.lower_bound_ray, [1.0])
    np.testing.assert_allclose(audit.objective, -1.0)
    np.testing.assert_allclose(audit.residual_norm, 0.0)
