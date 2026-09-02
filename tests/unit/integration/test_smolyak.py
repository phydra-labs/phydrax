#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.integration import _sparse_grid as sparse_grid_module


def _product_intervals(dimension):
    factors = tuple(
        phx.domain.ScalarInterval(0.0, 1.0, label=f"x{axis}") for axis in range(dimension)
    )
    return phx.domain.ProductDomain(*factors)


def test_conventional_sparse_grid_growth_and_diagnostics_in_eight_dimensions():
    domain = _product_intervals(8)
    plan = phx.integration.SparseGridPlan(8, 3)
    realization = phx.integration.materialize(
        phx.integration.over(domain.component()),
        plan,
    )
    estimate = phx.integration.reduce(1.0, realization)

    assert realization.batch.batch.weights.data.size == 145
    assert realization.batch.previous is not None
    assert realization.batch.previous.weights.data.size == 17
    assert estimate.successful
    assert estimate.value.data == pytest.approx(1.0, abs=1e-12)
    assert estimate.diagnostics.num_unique_nodes == 145
    assert estimate.diagnostics.previous_num_unique_nodes == 17
    assert estimate.diagnostics.num_terms == 45
    assert estimate.diagnostics.axis_rules == ("clenshaw-curtis",) * 8
    assert "rules-clenshaw-curtis" in estimate.provenance.realization


def test_sparse_plan_accepts_real_anisotropy_and_validates_rule_contract():
    plan = phx.integration.SparseGridPlan(
        3,
        4,
        anisotropy=(0.5, 1.25, 2.0),
        axis_rules=("clenshaw-curtis", "gauss-hermite", "clenshaw-curtis"),
    )

    assert plan.anisotropy == (0.5, 1.25, 2.0)
    assert plan.axis_rules == (
        "clenshaw-curtis",
        "gauss-hermite",
        "clenshaw-curtis",
    )
    with pytest.raises(ValueError, match="Unsupported Smolyak axis rule"):
        phx.integration.SparseGridPlan(1, 2, axis_rules="leja")
    with pytest.raises(ValueError, match="one rule per dimension"):
        phx.integration.SparseGridPlan(
            2,
            2,
            axis_rules=("clenshaw-curtis",),
        )


def test_gauss_hermite_integrates_shifted_normal_moments():
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Normal(2.0, 3.0),
        label="z",
    )
    function = probability.Function("z")(
        lambda z: jnp.stack((jnp.ones_like(z), z, z**2, z**3))
    )
    estimate = phx.integration.integrate(
        function,
        phx.integration.over(probability.component()),
        phx.integration.SparseGridPlan(1, 4, axis_rules="gauss-hermite"),
    )

    expected = jnp.asarray([1.0, 2.0, 13.0, 62.0])
    assert estimate.successful
    assert jnp.allclose(jnp.asarray(estimate.value.data), expected, atol=1e-11)


def test_gauss_hermite_uses_lognormal_reference_transform():
    probability = phx.domain.ProbabilityDomain(
        phx.uq.LogNormal(0.1, 0.2),
        label="z",
    )
    estimate = phx.integration.integrate(
        probability.Function("z")(lambda z: z),
        phx.integration.over(probability.component()),
        phx.integration.SparseGridPlan(1, 6, axis_rules="gauss-hermite"),
    )

    assert estimate.successful
    assert estimate.value.data == pytest.approx(
        float(jnp.exp(0.1 + 0.5 * 0.2**2)),
        rel=1e-11,
    )


def test_mixed_physical_uniform_and_normal_axes_preserve_measure_semantics():
    x = phx.domain.ScalarInterval(0.0, 2.0, label="x")
    u = phx.domain.ProbabilityDomain(phx.uq.Uniform(-1.0, 1.0), label="u")
    z = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")
    domain = phx.domain.ProductDomain(x, u, z)
    function = domain.Function("x", "u", "z")(lambda x, u, z: x + u + z**2)
    estimate = phx.integration.integrate(
        function,
        phx.integration.over(domain.component()),
        phx.integration.SparseGridPlan(
            3,
            4,
            axis_rules=(
                "clenshaw-curtis",
                "clenshaw-curtis",
                "gauss-hermite",
            ),
        ),
    )

    assert estimate.successful
    assert estimate.value.data == pytest.approx(4.0, abs=1e-11)


def test_gaussian_sparse_grid_supports_normalized_density_targets():
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Normal(0.0, 1.0),
        label="z",
    )
    target = phx.integration.normalized_density(
        phx.integration.over(probability.component()),
        probability.Function("z")(lambda z: 0.2 * z),
    )
    estimate = phx.integration.integrate(
        probability.Function("z")(lambda z: z),
        target,
        phx.integration.SparseGridPlan(1, 8, axis_rules="gauss-hermite"),
    )

    assert estimate.successful
    assert estimate.value.data == pytest.approx(0.2, abs=1e-10)


def test_sparse_grid_rejects_incompatible_factor_rule_pairs():
    interval = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    normal = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="z")

    with pytest.raises(TypeError, match="requires a probability factor"):
        phx.integration.materialize(
            phx.integration.over(interval.component()),
            phx.integration.SparseGridPlan(1, 2, axis_rules="gauss-hermite"),
        )
    with pytest.raises(ValueError, match="bounded probability support"):
        phx.integration.materialize(
            phx.integration.over(normal.component()),
            phx.integration.SparseGridPlan(1, 2),
        )


def test_sparse_grid_preserves_complex_vector_outputs_under_jit_reduction():
    interval = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    function = interval.Function("x")(
        lambda x: jnp.stack((x + 1j * x**2, x**4 - 2j * x**3))
    )
    realization = phx.integration.materialize(
        phx.integration.over(interval.component()),
        phx.integration.SparseGridPlan(1, 4),
    )

    value = jax.jit(lambda: phx.integration.reduce(function, realization).value.data)()

    assert jnp.allclose(value, jnp.asarray([2j / 3.0, 0.4 + 0j]), atol=1e-11)


def test_builtin_probability_reference_transforms_round_trip():
    distributions = (
        phx.uq.Uniform(-2.0, 4.0),
        phx.uq.Normal(1.0, 3.0),
        phx.uq.LogNormal(0.2, 0.4),
    )
    references = (
        jnp.asarray([-0.8, 0.0, 0.9]),
        jnp.asarray([-2.0, 0.0, 1.5]),
        jnp.asarray([-2.0, 0.0, 1.5]),
    )
    for index, (distribution, reference) in enumerate(
        zip(distributions, references, strict=True)
    ):
        domain = phx.domain.ProbabilityDomain(distribution, label=f"z{index}")
        assert jnp.allclose(
            domain.reference_transport.to_reference(
                domain.reference_transport.from_reference(reference)
            ),
            reference,
            atol=1e-12,
        )

    empirical = phx.domain.ProbabilityDomain(
        phx.uq.EmpiricalDistribution(jnp.asarray([0.0, 1.0])),
        label="e",
    )
    with pytest.raises(ValueError, match="no declared exact reference transport"):
        _ = empirical.reference_transport


@pytest.mark.parametrize(
    ("capacity", "expected_status"),
    (
        ({"max_indices": 1, "max_nodes": 100}, "maximum-indices"),
        ({"max_indices": 100, "max_nodes": 1}, "maximum-nodes"),
    ),
)
def test_adaptive_sparse_grid_rejects_frontier_before_materialization(
    monkeypatch,
    capacity,
    expected_status,
):
    interval = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    function = interval.Function("x")(lambda x: x**2)
    materialized_index_counts = []
    original = sparse_grid_module._materialize_level

    def instrumented(target, plan, level, /, *, index_set=None):
        materialized_index_counts.append(len(index_set.indices))
        return original(target, plan, level, index_set=index_set)

    monkeypatch.setattr(sparse_grid_module, "_materialize_level", instrumented)
    result = phx.integration.prepare_adaptive_sparse_grid(
        function,
        phx.integration.over(interval.component()),
        phx.integration.AdaptiveSparseGridPlan(
            1,
            max_rounds=2,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
            **capacity,
        ),
    )

    assert materialized_index_counts == [1]
    assert result.epochs[-1].status == expected_status
    assert result.epochs[-1].selected is None
    assert result.diagnostics.accepted_indices == 1
    assert result.diagnostics.num_unique_nodes == 1


def test_adaptive_sparse_grid_accepts_an_exactly_in_cap_refinement(monkeypatch):
    interval = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
    function = interval.Function("x")(lambda x: x**2)
    materialized_node_counts = []
    original = sparse_grid_module._materialize_level

    def instrumented(target, plan, level, /, *, index_set=None):
        batch = original(target, plan, level, index_set=index_set)
        materialized_node_counts.append(int(batch.weights.data.size))
        return batch

    monkeypatch.setattr(sparse_grid_module, "_materialize_level", instrumented)
    result = phx.integration.prepare_adaptive_sparse_grid(
        function,
        phx.integration.over(interval.component()),
        phx.integration.AdaptiveSparseGridPlan(
            1,
            max_indices=3,
            max_nodes=3,
            max_rounds=2,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        ),
    )

    assert materialized_node_counts == [1, 3]
    assert result.epochs[0].status == "accepted"
    assert result.epochs[-1].status == "maximum-nodes"
    assert result.estimate.value.data == pytest.approx(2.0 / 3.0, abs=1e-12)
