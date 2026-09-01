#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import optax
import pytest

import phydrax as phx


def test_native_gmres_uses_low_preconditioner_and_basis_with_high_residual():
    la = phx.linalg
    matrix = jnp.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    problem = la.LinearSystem(la.DenseLinearOperator(matrix))
    policy = la.LinearSolvePolicy(
        la.GMRES(restart=2),
        tolerance=la.TolerancePolicy(relative=1e-10, max_steps=10),
        preconditioning=la.PreconditioningPolicy(la.JacobiPreconditionerBuilder()),
        precision=la.MixedPrecisionPolicy(
            preconditioner_dtype=jnp.float32,
            krylov_dtype=jnp.float32,
        ),
        differentiation=la.DifferentiationPolicy("none"),
    )

    result = la.solve(
        problem,
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        policy=policy,
    )

    assert result.value.dtype == jnp.float64
    assert jnp.allclose(result.value, jnp.linalg.solve(matrix, jnp.asarray([1.0, 2.0])))
    evidence = result.provenance.effective_precision
    assert evidence.preconditioner_dtype == "float32"
    assert evidence.krylov_dtype == "float32"
    assert evidence.residual_dtype == "float64"
    estimate = la.plan(problem, policy).candidates[-1]
    assert estimate.preconditioner_storage_bytes == 2 * jnp.dtype(jnp.float32).itemsize
    assert estimate.preconditioner_preparation_workspace_bytes == 24
    assert estimate.preconditioner_apply_workspace_bytes_per_rhs == 32
    assert estimate.krylov_basis_bytes_per_rhs == 88


def test_functional_precision_is_scoped_to_standard_optax():
    domain = phx.domain.ScalarInterval(0.0, 1.0, label="t")
    parameter = domain.Parameter(2.0)
    objective = phx.terms.IntegralFunctional.from_operator(
        source=phx.integration.per_step(
            phx.integration.over(domain.component()),
            phx.integration.AdaptiveQuadraturePlan(),
        ),
        operator=lambda value: (value - 1.0) ** 2,
        objective_vars="u",
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": parameter},
        terms=(objective,),
    )
    precision = phx.solver.FunctionalPrecisionPolicy("highest")

    trained = solver.solve(
        num_iter=2,
        optim=optax.sgd(0.1),
        keep_best=False,
        jit=True,
        log_every=0,
        precision=precision,
    )

    assert jnp.allclose(trained.loss(), 0.4096, rtol=1e-8, atol=1e-10)
    assert trained.precision.policy_id == precision.policy_id
    assert trained.precision_evidence is not None
    assert all(
        record.precision_evidence_id == trained.precision_evidence.evidence_id
        for record in trained.discretization_bundle.records
    )

    with pytest.raises(ValueError, match="standard Optax"):
        solver.solve(
            num_iter=1,
            optim=phx.optim.kfac(learning_rate=0.1),
            log_every=0,
            precision=precision,
        )
