import jax.numpy as jnp

import phydrax as phx
from phydrax._numerics import SmolyakIndexSet


def test_cid02_signed_estimators_preserve_sign_and_neyman_allocation():
    estimator = phx.integration.AdaptiveStratifiedEstimator(jnp.asarray([0.5, 0.5]))
    allocation = estimator.next_allocation(jnp.asarray([1.0, 16.0]), 20)
    assert int(allocation[1]) > int(allocation[0])
    value = estimator.signed_reduce(jnp.asarray([-2.0, 2.0]), jnp.asarray([0.5, 0.5]))
    assert value == 0.0


def test_cid04_bounded_breakpoint_discovery_reports_jump():
    plan = phx.integration.BreakpointDiscoveryPlan(
        33, 4, 3, jump_threshold=3.0, defect_threshold=3.0
    )
    evidence, evaluations = phx.integration.discover_breakpoints(
        lambda x: jnp.where(x < 0.2, -1.0, 1.0)[:, None],
        jnp.asarray([-1.0, 1.0]),
        plan,
    )
    assert int(evaluations) == 45
    assert bool(jnp.any(evidence.active))
    assert float(jnp.min(jnp.abs(evidence.points[evidence.active] - 0.2))) < 0.1


def test_cid05_nd_adaptive_cubature_integrates_signed_polynomial():
    plan = phx.integration.AdaptiveCubaturePlan(
        4, absolute_tolerance=1e-8, max_cells=16, throw=False
    )
    estimate = phx.integration.adaptive_cubature_callable(
        lambda x: (x[:, 0] * x[:, 1] + x[:, 2] ** 2)[:, None],
        plan,
        precision=phx.integration.IntegrationPrecisionPolicy(),
    )
    assert jnp.allclose(estimate.value, jnp.asarray([16.0 / 3.0]), atol=1e-7)

    index_set = SmolyakIndexSet(2, ((0, 0), (1, 0)))
    assert index_set.frontier().candidates == ((0, 1), (2, 0))
    assert index_set.add((0, 1)).indices == ((0, 0), (0, 1), (1, 0))


def test_cid07_declared_quantile_transport_round_trip_and_empirical_rejection():
    normal = phx.domain.ProbabilityDomain(phx.uq.Normal(1.0, 2.0), label="z")
    reference = jnp.asarray([-1.0, 0.0, 1.0])
    physical = normal.reference_transport.from_reference(reference)
    assert jnp.allclose(normal.reference_transport.to_reference(physical), reference)

    empirical = phx.domain.ProbabilityDomain(
        phx.uq.EmpiricalDistribution(jnp.asarray([0.0, 1.0])), label="e"
    )
    try:
        _ = empirical.reference_transport
    except ValueError as error:
        assert "no declared exact reference transport" in str(error)
    else:
        raise AssertionError("Empirical laws must not acquire a fabricated transport")
