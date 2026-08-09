#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _batch(weights, *, cases=2):
    quadrature = jnp.asarray(weights, dtype=float)
    nodes = jnp.linspace(0.0, 1.0, quadrature.shape[0])
    coordinates = jnp.broadcast_to(
        nodes[None, :, None],
        (cases, nodes.shape[0], 1),
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=jnp.broadcast_to(
            quadrature,
            (cases, quadrature.shape[0]),
        ),
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.zeros((cases, quadrature.shape[0])),
        coordinates=coordinates,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"state": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(cases,),
    )


def _predictive(samples, batch, *, sample_dim):
    return phx.uq.operator_predictive_from_samples(
        jnp.asarray(samples, dtype=float),
        batch,
        phx.nn.operator.OperatorOutputSpec("scalar"),
        sample_axes=(phx.uq.SampleAxis(sample_dim, "process"),),
        field_name="output",
        query_name="query",
    )


def _measure(points, weights, *, provenance):
    return phx.integration.discrete(
        jnp.asarray(points, dtype=float),
        cx.Field(jnp.asarray(weights, dtype=float), dims=("atom",)),
        axes="atom",
        normalized=True,
        provenance=provenance,
    )


def test_raw_predictive_sinkhorn_divergence_retains_all_three_solves_and_gradients():
    source = jnp.asarray([[0.0, 0.2], [1.0, 1.2], [2.0, 2.2]])
    identity = phx.uq.predictive_sinkhorn_divergence(source, source, epsilon=1.0)
    translated = phx.uq.predictive_sinkhorn_divergence(
        source,
        source + 0.7,
        epsilon=1.0,
    )
    gradient = jax.grad(
        lambda shift: phx.uq.predictive_sinkhorn_divergence(
            source,
            source + shift,
            epsilon=1.0,
        ).value
    )(jnp.asarray(0.4))

    assert identity.converged & translated.converged
    assert jnp.allclose(identity.value, 0.0, atol=1e-12)
    assert translated.value > 0.0
    assert translated.cross.problem.shape == (3, 3)
    assert translated.source_self.problem.shape == (3, 3)
    assert translated.target_self.problem.shape == (3, 3)
    assert jnp.isfinite(gradient)


def test_operator_transport_metrics_keep_physical_cases_independent_and_replay_keys():
    batch = _batch([0.1, 0.2, 0.7])
    left_samples = jnp.asarray(
        [
            [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            [[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]],
        ]
    )
    right_samples = left_samples.at[:, 1, :].add(jnp.asarray([0.2, 0.5, 1.0]))
    left = _predictive(left_samples, batch, sample_dim="left")
    right = _predictive(right_samples, batch, sample_dim="right")

    sinkhorn = phx.uq.operator_ensemble_sinkhorn_divergence(
        left,
        right,
        epsilon=1.0,
        reduction="none",
    )
    sliced = phx.uq.operator_ensemble_sliced_wasserstein(
        left,
        right,
        num_projections=8,
        key=jr.key(4),
        reduction="none",
    )
    replay = phx.uq.operator_ensemble_sliced_wasserstein(
        left,
        right,
        num_projections=8,
        key=jr.key(4),
        reduction="none",
    )

    assert sinkhorn.method == "sinkhorn-divergence"
    assert sinkhorn.per_case.shape == (2,)
    assert jnp.all(sinkhorn.transport.converged)
    assert jnp.allclose(sinkhorn.per_case[0], 0.0, atol=1e-12)
    assert sinkhorn.per_case[1] > 0.0
    assert sliced.sliced.projection_distances.shape == (2, 8)
    assert jnp.allclose(sliced.per_case[0], 0.0, atol=1e-12)
    assert sliced.per_case[1] > 0.0
    assert jnp.array_equal(sliced.value, replay.value)
    assert jnp.array_equal(sliced.sliced.projections, replay.sliced.projections)


def test_operator_transport_uses_quadrature_scaled_whole_events():
    batch = _batch([0.01, 0.09, 0.9])
    left_samples = jnp.zeros((3, 2, 3))
    right_samples = left_samples.at[..., 0].set(1.0)
    left = _predictive(left_samples, batch, sample_dim="left_measure")
    right = _predictive(right_samples, batch, sample_dim="right_measure")
    quadrature = phx.uq.operator_ensemble_sinkhorn_divergence(
        left,
        right,
        measure="quadrature",
        epsilon=1.0,
        reduction="none",
    )
    uniform = phx.uq.operator_ensemble_sinkhorn_divergence(
        left,
        right,
        measure="uniform",
        epsilon=1.0,
        reduction="none",
    )

    assert jnp.all(quadrature.per_case < uniform.per_case)
    assert jnp.allclose(
        quadrature.per_case / uniform.per_case,
        0.03,
        rtol=1e-6,
        atol=1e-8,
    )


def test_optimal_transport_ensemble_transform_preserves_weighted_mean_and_gradients():
    particles = jnp.asarray([[0.0], [1.0], [3.0]])
    weights = jnp.asarray([0.1, 0.2, 0.7])
    result = phx.uq.optimal_transport_ensemble_transform(
        particles,
        weights,
        epsilon=1.0,
    )
    replay = phx.uq.optimal_transport_ensemble_transform(
        particles,
        weights,
        epsilon=1.0,
    )
    gradient = jax.grad(
        lambda values: jnp.sum(
            phx.uq.optimal_transport_ensemble_transform(
                values,
                weights,
                epsilon=1.0,
            ).particles
            ** 2
        )
    )(particles)

    assert result.transport.converged
    assert jnp.array_equal(result.particles, replay.particles)
    assert jnp.allclose(result.source_mean, jnp.asarray([2.3]))
    assert jnp.allclose(result.transformed_mean, result.source_mean, atol=1e-9)
    assert jnp.allclose(result.mean_error, 0.0, atol=1e-9)
    assert jnp.all(jnp.isfinite(gradient))
    assert "transport" not in phx.uq.ResamplingMethod.__args__


def test_batched_particle_transform_preserves_case_and_particle_layout():
    particles = jnp.asarray(
        [
            [[0.0], [1.0], [3.0]],
            [[2.0], [3.0], [5.0]],
        ]
    )
    weights = jnp.asarray([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
    result = phx.uq.optimal_transport_ensemble_transform(
        particles,
        weights,
        particle_axis=1,
        epsilon=1.0,
    )

    assert result.particles.shape == particles.shape
    assert result.transport.source_potential.shape == (2, 3)
    assert result.transport.diagnostics.status.shape == (2,)
    assert jnp.all(result.transport.converged)
    assert jnp.allclose(result.transformed_mean, result.source_mean, atol=1e-8)


def test_transport_functional_terms_return_scalar_values_and_native_diagnostics():
    solver = phx.transport.Sinkhorn(
        1.0,
        max_iterations=500,
        tolerance=1e-8,
        check_every=5,
    )
    target = _measure([[1.0], [2.0]], [0.5, 0.5], provenance="reference")
    source = _measure([[0.0], [1.0]], [0.5, 0.5], provenance="source")
    reference = phx.transport.prepare_sinkhorn_reference(
        target,
        cost=phx.transport.SquaredEuclideanCost(),
        solver=solver,
    )
    spatial = phx.terms.SpatialSinkhornDivergenceTerm(lambda _: source, reference)
    empirical = phx.terms.EmpiricalSinkhornDivergenceTerm(
        jnp.asarray([[0.0], [1.0]]),
        reference,
    )
    sliced = phx.terms.SlicedWassersteinTerm(
        jnp.asarray([[0.0], [1.0]]),
        jnp.asarray([[1.0], [2.0]]),
        projections=jnp.ones((4, 1)),
    )
    quantile = phx.terms.SoftQuantileFunctional(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.asarray([0.0, 1.0, 2.0]),
        epsilon=0.2,
    )

    spatial_evaluation = spatial.term_evaluation({})
    empirical_evaluation = empirical.term_evaluation({})
    sliced_evaluation = sliced.term_evaluation({})
    quantile_evaluation = quantile.term_evaluation({})

    assert spatial_evaluation.value.shape == ()
    assert empirical_evaluation.value.shape == ()
    assert spatial_evaluation.diagnostics.converged
    assert empirical_evaluation.diagnostics.converged
    assert jnp.allclose(spatial_evaluation.value, empirical_evaluation.value)
    assert jnp.allclose(sliced_evaluation.value, 1.0)
    assert jnp.allclose(quantile_evaluation.value, 0.0, atol=1e-28)
    assert quantile_evaluation.diagnostics["quantiles"].shape == (3,)


def test_transport_terms_reject_nonconverged_training_solves():
    target = _measure([[0.0], [1.0]], [0.5, 0.5], provenance="reference")
    solver = phx.transport.Sinkhorn(
        0.05,
        max_iterations=1,
        tolerance=1e-12,
        check_every=1,
    )
    reference = phx.transport.prepare_sinkhorn_reference(
        target,
        cost=phx.transport.SquaredEuclideanCost(),
        solver=solver,
    )
    source = _measure([[4.0], [8.0]], [0.9, 0.1], provenance="source")
    term = phx.terms.SpatialSinkhornDivergenceTerm(lambda _: source, reference)

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="did not converge"):
        evaluation = term.term_evaluation({})
        jax.block_until_ready(evaluation.value)
