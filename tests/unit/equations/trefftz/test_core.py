#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable
from phydrax.equations.trefftz._polynomial import _critical_construction_audit


def _laplacian(function, point):
    return jnp.trace(jax.hessian(function)(point))


def test_similarity_and_resource_contracts_fail_closed():
    normalization = phx.equations.SimilarityNormalization([1.0, -2.0], 3.0)
    assert jnp.allclose(normalization(jnp.asarray([4.0, 1.0])), jnp.ones((2,)))
    with pytest.raises(ValueError, match="positive"):
        phx.equations.SimilarityNormalization([0.0, 0.0], 0.0)
    with pytest.raises(ValueError, match="vector"):
        phx.equations.SimilarityNormalization([[0.0, 0.0]], 1.0)

    budget = phx.equations.TrefftzResourceBudget(maximum_rank=2)
    with pytest.raises(ValueError, match="resource budget"):
        phx.equations.HarmonicPolynomialBasis(2, 2, resources=budget)

def test_construction_audits_are_paired_and_fail_each_block_independently():
    with pytest.raises(ValueError, match="block 1 failed"):
        _critical_construction_audit(
            (1e-12, 1e-8),
            (1e-5, 1e-10),
            (0, 1),
            construction="Synthetic",
        )
    residual, tolerance = _critical_construction_audit(
        (1e-12, 5e-9),
        (1e-5, 1e-8),
        (0, 1),
        construction="Synthetic",
    )
    assert residual == 5e-9
    assert tolerance == 1e-8


def test_canonical_harmonic_basis_is_deterministic_and_exact():
    first = phx.equations.HarmonicPolynomialBasis(3, 4)
    second = phx.equations.HarmonicPolynomialBasis(3, 4)
    assert first.rank == 25
    assert first.basis_id == second.basis_id
    for left, right in zip(
        first.coefficient_blocks,
        second.coefficient_blocks,
        strict=True,
    ):
        assert jnp.array_equal(left, right)
    assert all(
        float(values[-1]) > 0.0 for values in first.singular_value_blocks
    )
    assert len(first.construction_residuals) == first.maximum_degree + 1
    assert all(
        float(residual) <= float(tolerance)
        for residual, tolerance in zip(
            first.construction_residuals,
            first.construction_tolerances,
            strict=True,
        )
    )

    model = phx.equations.LinearTrefftzField(
        first,
        initial_scale=0.2,
        key=jr.key(1),
    )
    point = jnp.asarray([0.2, -0.3, 0.4])
    assert jnp.allclose(_laplacian(model, point), 0.0, atol=2e-11, rtol=2e-11)

    trainable, fixed = partition_trainable(model)
    assert trainable.coefficients is not None
    assert trainable.basis is None
    assert fixed.coefficients is None
    assert fixed.basis is first


def test_almansi_basis_is_polyharmonic_and_order_one_matches_harmonic_rank():
    harmonic = phx.equations.HarmonicPolynomialBasis(3, 2)
    order_one = phx.equations.PolyharmonicAlmansiBasis(3, 1, 2)
    assert harmonic.rank == order_one.rank

    basis = phx.equations.PolyharmonicAlmansiBasis(3, 2, (2, 1))
    model = phx.equations.LinearTrefftzField(
        basis,
        initial_scale=0.1,
        key=jr.key(2),
    )
    point = jnp.asarray([0.15, -0.2, 0.35])

    def once(value):
        return _laplacian(model, value)

    assert jnp.allclose(_laplacian(once, point), 0.0, atol=2e-10, rtol=2e-10)
    assert dict(basis.certificate.equation_parameters)["order"] == 2
    assert basis.constituent_certificate_ids == tuple(
        child.certificate.certificate_id for child in basis.harmonic_bases
    )
    assert len(basis.construction_residuals) == basis.order


def test_helmholtz_basis_canonicalizes_orientation_and_is_exact():
    directions = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    basis = phx.equations.HelmholtzPlaneWaveBasis(3, 2.5, directions)
    model = phx.equations.LinearTrefftzField(
        basis,
        initial_scale=0.1,
        key=jr.key(3),
    )
    point = jnp.asarray([0.1, 0.25, -0.4])
    residual = _laplacian(model, point) + 2.5**2 * model(point)
    assert jnp.allclose(residual, 0.0, atol=2e-11, rtol=2e-11)

    with pytest.raises(ValueError, match="duplicate or antipodal"):
        phx.equations.HelmholtzPlaneWaveBasis(
            3,
            1.0,
            jnp.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="positive"):
        phx.equations.HelmholtzPlaneWaveBasis(3, 0.0, directions)


def test_bound_trial_metadata_provenance_audit_and_enforcement_guard():
    domain = phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0))
    model = phx.equations.LinearTrefftzField(
        phx.equations.HarmonicPolynomialBasis(2, 2),
        initial_scale=0.1,
        key=jr.key(4),
    )
    field = domain.Model("x")(model)
    certificate = phx.equations.trial_space_certificate(field)
    assert certificate.equation_family == "laplace"
    assert "trial_space_certificate" not in (2.0 * field).metadata

    batch = domain.component().sample(phx.domain.PointSampling(16), key=jr.key(5))
    report = phx.equations.audit_trial_space(field, batch)
    assert bool(report.valid)
    assert report.certificate_id == certificate.certificate_id

    boundary = domain.component({"x": phx.domain.Boundary()})
    with pytest.raises(ValueError, match="certified exact PDE trial field"):
        phx.enforcement.enforce_dirichlet(field, boundary, target=0.0)

    condition = phx.conditions.Dirichlet("u", boundary, target=0.0)
    with pytest.raises(ValueError, match="certified exact PDE trial fields"):
        phx.enforcement.compile(
            {"u": field},
            (phx.enforcement.EnforcementSpec(condition),),
        )

    source = phx.integration.fixed(
        phx.integration.materialize(
            phx.integration.mean_over(boundary),
            phx.domain.PointSampling(16),
            key=jr.key(6),
        )
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(phx.terms.ResidualPenalty(condition, source),),
    )
    assert solver.discretization_bundle.records[0].artifact_kind == "exact-pde-trial-space"


def test_direct_linear_trial_space_solve_recovers_harmonic_boundary_field():
    domain = phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0))
    model = phx.equations.LinearTrefftzField(
        phx.equations.HarmonicPolynomialBasis(2, 1)
    )
    field = domain.Model("x")(model)
    target = domain.Function("x")(lambda x: 0.5 + 1.25 * x[0] - 0.75 * x[1])
    boundary = domain.component({"x": phx.domain.Boundary()})
    condition = phx.conditions.Dirichlet("u", boundary, target=target)
    source = phx.integration.fixed(
        phx.integration.materialize(
            phx.integration.mean_over(boundary),
            phx.domain.PointSampling(64),
            key=jr.key(7),
        )
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(phx.terms.ResidualPenalty(condition, source),),
    )
    result = phx.solver.solve_linear_trial_space(solver, key=jr.key(8))
    assert bool(result.valid)
    assert float(result.final_residual_norm) < 1e-10
    assert result.coefficient_count == 3

    points = jnp.asarray(
        [
            [-0.4, 0.3],
            [0.1, -0.2],
            [0.7, 0.5],
        ]
    )
    batch = domain.component().points(points)
    learned = jnp.asarray(result.solver["u"](batch).data)
    truth = 0.5 + 1.25 * points[:, 0] - 0.75 * points[:, 1]
    assert jnp.allclose(learned, truth, atol=1e-10, rtol=1e-10)


def test_direct_linear_solver_rejects_nonfixed_realizations():
    domain = phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0))
    field = domain.Model("x")(
        phx.equations.LinearTrefftzField(
            phx.equations.HarmonicPolynomialBasis(2, 1)
        )
    )
    boundary = domain.component({"x": phx.domain.Boundary()})
    condition = phx.conditions.Dirichlet("u", boundary, target=0.0)
    source = phx.integration.per_step(
        phx.integration.mean_over(boundary),
        phx.domain.PointSampling(16),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(phx.terms.ResidualPenalty(condition, source),),
    )
    with pytest.raises(TypeError, match="fixed integration"):
        phx.solver.solve_linear_trial_space(solver)
