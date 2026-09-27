#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


cl = phx.metrix.clifford


def test_monogenic_scenario_1() -> None:
    algebra = cl.CliffordAlgebraSpec((1, 1, 1))
    first = phx.equations.MonogenicPolynomialBasis(algebra, 2)
    second = phx.equations.MonogenicPolynomialBasis(algebra, 2)
    expected_rank = algebra.blade_count * (1 + 2 + 3)

    assert first.rank == expected_rank
    assert first.basis_id == second.basis_id
    assert first.certificate.certificate_id == second.certificate.certificate_id
    assert first.certificate.equation_family == "dirac"
    assert first.certificate.representation_id == algebra.algebra_id
    assert float(first.certificate.construction_residual) == 0.0
    algebra = cl.CliffordAlgebraSpec((1, 1))
    basis = phx.equations.MonogenicPolynomialBasis(algebra, 3)
    point = jnp.asarray([0.2, -0.35])

    for axis in range(2):
        expected = jax.jacfwd(basis.evaluate)(point)[..., axis]
        assert jnp.allclose(basis.evaluate_partial(point, axis, 1), expected)
        expected_second = jax.jacfwd(jax.jacfwd(basis.evaluate))(point)[..., axis, axis]
        assert jnp.allclose(
            basis.evaluate_partial(point, axis, 2),
            expected_second,
        )
    algebra = cl.CliffordAlgebraSpec((1, 1, 1))
    resources = phx.equations.TrefftzResourceBudget(maximum_rank=10)
    with pytest.raises(ValueError, match="exceeds its resource budget"):
        phx.equations.MonogenicPolynomialBasis(
            algebra,
            2,
            resources=resources,
        )
    algebra = cl.CliffordAlgebraSpec((1, -1))
    layout = cl.CliffordBladeLayout.full(algebra)
    # ty: ignore[invalid-argument-type]
    domain = phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0))
    field = domain.Function("x")(
        lambda x: jnp.asarray([x[0] ** 2 + 3.0 * x[1] ** 2, 0.0, 0.0, 0.0])
    )
    first = phx.operators.clifford_dirac(field, algebra, layout)
    second = phx.operators.clifford_dirac(first, algebra, layout)
    batch = domain.component().sample(phx.domain.PointSampling(5), key=jr.key(8))
    expected = jnp.broadcast_to(jnp.asarray([-4.0, 0.0, 0.0, 0.0]), (5, 4))

    assert jnp.allclose(second(batch).data, expected)
    algebra = cl.CliffordAlgebraSpec((1, 1))
    basis = phx.equations.MonogenicPolynomialBasis(algebra, 2)
    model = phx.equations.LinearMonogenicField(
        basis,
        initial_scale=0.2,
        key=jr.key(2),
    )
    # ty: ignore[invalid-argument-type]
    domain = phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0))
    field = domain.Model("x")(model)
    batch = domain.component().sample(phx.domain.PointSampling(12), key=jr.key(3))
    # ty: ignore[invalid-argument-type]
    report = phx.equations.audit_trial_space(field, batch)

    assert bool(report.valid)
    assert float(report.maximum_residual) < 1e-11
    with pytest.raises(TypeError, match="no TrialSpaceCertificate"):
        # ty: ignore[invalid-argument-type]
        phx.equations.audit_trial_space(2.0 * field, batch)


def test_degenerate_monogenic_basis_and_dirac_are_rejected() -> None:
    algebra = cl.CliffordAlgebraSpec((1, 0))
    with pytest.raises(ValueError, match="nondegenerate"):
        phx.equations.MonogenicPolynomialBasis(algebra, 1)
