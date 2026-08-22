#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def test_mpax_capabilities_are_lazy_and_specific():
    availability = phx.backends.mpax_availability()
    capabilities = availability.capabilities

    assert capabilities.backend == "mpax"
    assert capabilities.execution == "device"
    assert capabilities.supports("optimization.linear-program")
    assert capabilities.supports("optimization.quadratic-program")
    assert not capabilities.supports("optimization.conic-program")
    assert availability.requirement == "install a compatible mpax==0.2.4 distribution"
    method_capabilities = phx.optim.MPAXraPDHG().capabilities
    assert method_capabilities.dense
    assert not method_capabilities.sparse
    assert not method_capabilities.matrix_free


def test_missing_mpax_raises_selected_backend_error():
    availability = phx.backends.mpax_availability()
    if availability.available:
        pytest.skip("MPAX is installed in this environment.")
    with pytest.raises(phx.backends.BackendUnavailableError, match="mpax"):
        phx.backends.prepare_mpax()


def test_mpax_methods_validate_problem_kind_before_provider_import():
    problem = phx.optim.QuadraticProgram(jnp.eye(1), jnp.zeros(1))
    policy = phx.optim.ConvexSolvePolicy(phx.optim.MPAXr2HPDHG())

    with pytest.raises(ValueError, match="does not support QPs"):
        phx.optim.plan_convex_program(problem, policy)


def test_unrolled_mpax_requires_finite_bounded_iteration_capacity():
    with pytest.raises(ValueError, match="bounded finite budget"):
        phx.backends.MPAXPlan("rapdhg", unroll=True, iteration_limit=100_000)
