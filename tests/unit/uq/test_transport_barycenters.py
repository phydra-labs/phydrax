#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _measure(points, weights, *, provenance="uq-barycenter"):
    return phx.integration.discrete(
        jnp.asarray(points, dtype=float),
        cx.Field(jnp.asarray(weights, dtype=float), dims=("atom",)),
        axes="atom",
        normalized=True,
        provenance=provenance,
    )


def _problem():
    first = _measure([[-1.0], [1.0]], [0.5, 0.5], provenance="forecast-a")
    second = _measure([[0.0], [2.0]], [0.5, 0.5], provenance="forecast-b")
    support = _measure([[-0.5], [0.5], [1.5]], [0.2, 0.5, 0.3])
    return first, second, support


def test_uq_fixed_support_aggregation_retains_native_solution():
    first, second, support = _problem()
    aggregation = phx.uq.aggregate_transport_barycenter(
        (first, second),
        support,
        measure_weights=jnp.asarray([0.25, 0.75]),
        cost=phx.transport.SquaredEuclideanCost(),
        solver=phx.transport.SinkhornBarycenter(
            0.5,
            max_iterations=400,
            tolerance=1e-8,
            check_every=2,
        ),
    )

    assert aggregation.converged
    assert aggregation.transport.converged
    assert aggregation.measure.target_mass == 1.0
    assert aggregation.measure.provenance == "uq-transport-barycenter"
    assert aggregation.transport.problem.provenance.measures == (
        "forecast-a",
        "forecast-b",
    )


def test_uq_free_support_aggregation_keeps_local_search_provenance():
    first = _measure([[0.0]], [1.0])
    second = _measure([[2.0]], [1.0])
    initialization = _measure([[0.4]], [1.0], provenance="uq-explicit-initial")
    inner = phx.transport.SinkhornBarycenter(
        0.2,
        max_iterations=50,
        tolerance=1e-10,
        check_every=1,
    )
    aggregation = phx.uq.aggregate_free_support_transport_barycenter(
        (first, second),
        initialization,
        measure_weights=jnp.asarray([0.5, 0.5]),
        cost=phx.transport.WeightedSquaredEuclideanCost([2.0]),
        solver=phx.transport.FreeSupportBarycenter(
            inner,
            max_iterations=4,
            tolerance=1e-9,
        ),
    )

    assert aggregation.converged
    assert jnp.allclose(
        aggregation.transport.barycenter.problem.support_points,
        jnp.asarray([[1.0]]),
    )
    assert aggregation.transport.provenance.initialization == "uq-explicit-initial"
    assert len(aggregation.transport.inner_results) == 4
    assert aggregation.measure.provenance == "uq-free-support-transport-barycenter"


def test_barycenter_objective_term_is_composable_and_returns_native_diagnostics():
    first, second, support = _problem()
    problem = phx.transport.fixed_support_barycenter_problem(
        (first, second),
        support,
        measure_weights=jnp.asarray([0.5, 0.5]),
        cost=phx.transport.SquaredEuclideanCost(),
    )
    term = phx.terms.BarycenterObjectiveTerm(
        lambda _: problem,
        phx.transport.SinkhornBarycenter(
            0.5,
            max_iterations=400,
            tolerance=1e-8,
            check_every=2,
        ),
        weight=2.5,
        objective_vars=("field",),
        label="ensemble-law-barycenter",
    )
    evaluation = term.term_evaluation({})

    assert evaluation.value.shape == ()
    assert evaluation.diagnostics.converged
    assert jnp.allclose(
        evaluation.value,
        2.5 * evaluation.diagnostics.objective,
    )
    assert term.objective_vars == ("field",)
    assert term.label == "ensemble-law-barycenter"


def test_barycenter_training_objective_rejects_nonconvergence():
    first, second, support = _problem()
    problem = phx.transport.fixed_support_barycenter_problem(
        (first, second),
        support,
        measure_weights=jnp.asarray([0.5, 0.5]),
        cost=phx.transport.SquaredEuclideanCost(),
    )
    term = phx.terms.BarycenterObjectiveTerm(
        lambda _: problem,
        phx.transport.SinkhornBarycenter(
            0.01,
            max_iterations=1,
            tolerance=0.0,
            check_every=1,
        ),
    )

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="did not converge"):
        evaluation = term.term_evaluation({})
        jax.block_until_ready(evaluation.value)


def test_scientific_barycenter_symbols_are_public():
    uq_symbols = {
        "FreeSupportTransportBarycenterAggregationResult",
        "TransportBarycenterAggregationResult",
        "aggregate_free_support_transport_barycenter",
        "aggregate_transport_barycenter",
    }
    assert uq_symbols <= set(phx.uq.__all__)
    assert "BarycenterObjectiveTerm" in phx.terms.__all__
