import jax
import jax.numpy as jnp
import pytest

from phydrax.metrix._chart import CoordinateChart
from phydrax.metrix._metric_domain import MetricDomainEvidence, MetricDomainStatus


def test_signed_margin_classifies_each_domain_lane_without_clipping():
    chart = CoordinateChart("radial", ("t", "r"))
    margins = jnp.asarray([-1.0, 0.0, 0.01, 0.2, jnp.nan])

    def classify(value):
        return MetricDomainEvidence.from_margin(
            value,
            chart=chart,
            domain_id="radial-exterior",
            boundary_tolerance=0.05,
        )

    evidence = jax.jit(classify)(margins)

    assert jnp.array_equal(evidence.margin, margins, equal_nan=True)
    assert jnp.array_equal(
        evidence.status,
        jnp.asarray(
            [
                MetricDomainStatus.OUTSIDE,
                MetricDomainStatus.NEAR_BOUNDARY,
                MetricDomainStatus.NEAR_BOUNDARY,
                MetricDomainStatus.VALID,
                MetricDomainStatus.NONFINITE,
            ],
            dtype=jnp.int32,
        ),
    )
    assert jnp.array_equal(evidence.inside, jnp.asarray([False, True, True, True, False]))
    assert jnp.array_equal(evidence.valid, jnp.asarray([False, True, True, True, False]))
    assert jnp.array_equal(
        evidence.derivative_valid,
        jnp.asarray([False, False, False, True, False]),
    )


def test_provider_status_remains_distinct_from_numeric_membership():
    chart = CoordinateChart("ingoing", ("v", "r", "theta", "phi"))
    evidence = MetricDomainEvidence(
        jnp.asarray([True, False]),
        jnp.asarray([False, False]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([MetricDomainStatus.REJECTED, MetricDomainStatus.REJECTED]),
        chart=chart,
        domain_id="ingoing-ring-exclusion",
    )

    assert jnp.array_equal(evidence.finite, jnp.asarray([True, True]))
    assert jnp.array_equal(evidence.inside, jnp.asarray([True, True]))
    assert jnp.array_equal(evidence.physically_valid, jnp.asarray([True, False]))
    assert not jnp.any(evidence.qualified)


def test_domain_evidence_validates_static_identity_and_aligned_shapes():
    chart = CoordinateChart("radial", ("t", "r"))
    with pytest.raises(ValueError, match="domain_id"):
        MetricDomainEvidence.from_margin(jnp.ones(2), chart=chart, domain_id="")
    with pytest.raises(ValueError, match="near_boundary must have shape"):
        MetricDomainEvidence(
            jnp.ones(2, dtype="bool"),
            jnp.ones(3, dtype="bool"),
            jnp.ones(2),
            jnp.zeros(2, dtype=jnp.int32),
            chart=chart,
            domain_id="radial",
        )
    with pytest.raises(TypeError, match="integer dtype"):
        MetricDomainEvidence(
            jnp.ones(2, dtype="bool"),
            jnp.zeros(2, dtype="bool"),
            jnp.ones(2),
            jnp.asarray([0.0, 1.5]),
            chart=chart,
            domain_id="radial",
        )


def test_invalid_tolerance_is_reported_per_lane_instead_of_repairing_margin():
    chart = CoordinateChart("radial", ("t", "r"))
    evidence = MetricDomainEvidence.from_margin(
        jnp.asarray([0.25, 0.5]),
        chart=chart,
        domain_id="radial",
        boundary_tolerance=-0.1,
    )

    assert jnp.array_equal(evidence.margin, jnp.asarray([0.25, 0.5]))
    assert not jnp.any(evidence.valid)
    assert jnp.all(evidence.status == int(MetricDomainStatus.REJECTED))
