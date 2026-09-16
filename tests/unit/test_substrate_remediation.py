#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._balance_ledger import BalanceTerm, evaluate_balance
from phydrax.linalg import DensePropertyVerificationPolicy, verify_dense_properties


def test_dense_property_verification_and_local_root():
    evidence = verify_dense_properties(
        jnp.asarray(((2.0, 0.5), (0.5, 1.0))),
        policy=DensePropertyVerificationPolicy(require_positive_definite=True),
    )
    assert bool(evidence.successful)
    assert evidence.properties.positive_definite

    root, diagnostics = phx.nonlinear.VectorLocalRootPlan(
        2,
        plan_id="remediation-root",
    ).solve_with_diagnostics(
        lambda value: jnp.asarray((value[0] ** 2 - 4.0, value[1] - 3.0)),
        jnp.asarray((1.0, 0.0)),
    )
    np.testing.assert_allclose(root, jnp.asarray((2.0, 3.0)), atol=1e-8)
    assert bool(diagnostics.converged)


def test_transaction_balance_runtime_and_affine_evolution():
    candidate = phx.lifecycle.TransactionalCandidate(
        jnp.asarray((1.0,)),
        jnp.asarray((2.0,)),
        jnp.asarray(True),
        jnp.asarray(True),
        "source",
    )
    committed = phx.lifecycle.commit_candidate(candidate)
    np.testing.assert_allclose(committed.state, jnp.asarray((2.0,)))

    balance = evaluate_balance(
        "energy",
        jnp.asarray(3.0),
        (
            BalanceTerm("input", jnp.asarray(5.0), -1, "source", "J"),
            BalanceTerm("output", jnp.asarray(2.0), 1, "sink", "J"),
        ),
        absolute_tolerance=1e-12,
        relative_tolerance=1e-12,
    )
    assert bool(balance.successful)

    identity = phx.qualification.QualificationRuntimeIdentity(
        "build", "environment", "jax", "single-device", "float64"
    )
    assert identity.compatible(identity)

    evolution = phx.dynamics.PreparedAffineLinearEvolution(
        jnp.zeros((1, 1)),
        jnp.asarray((2.0,)),
    ).step(jnp.asarray((1.0,)), jnp.asarray(0.5))
    np.testing.assert_allclose(evolution.value, jnp.asarray((2.0,)), atol=1e-8)
    assert bool(evolution.successful)
