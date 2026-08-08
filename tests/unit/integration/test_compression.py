#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_moment_compression_is_reusable_and_preserves_named_ancestry():
    axis = "sample"
    coordinates = jnp.linspace(-1.0, 1.0, 33)
    samples = cx.Field(coordinates, dims=(axis,))
    log_weights = cx.Field(jnp.log(jnp.arange(1.0, 34.0)), dims=(axis,))
    ancestry = cx.Field(jnp.arange(100, 133, dtype=jnp.int32), dims=(axis,))
    features = cx.Field(
        jnp.stack((coordinates, coordinates**2), axis=1),
        dims=(axis, "feature"),
    )
    target = phx.integration.weighted(
        samples,
        log_weights,
        ancestry=ancestry,
        sample_axes=axis,
    )
    source = phx.integration.materialize(target)

    compressed = phx.integration.compress(
        source,
        phx.coresets.MomentRecombination(),
        features=features,
    )
    source_estimate = phx.integration.reduce(lambda value: value**2, source)
    compressed_estimate = phx.integration.reduce(lambda value: value**2, compressed)

    assert compressed.batch.num_samples <= 3
    assert jnp.allclose(
        compressed_estimate.value.data,
        source_estimate.value.data,
        atol=1e-11,
        rtol=1e-11,
    )
    selected_coordinates = compressed.batch.samples.data
    source_indices = jnp.searchsorted(coordinates, selected_coordinates)
    assert jnp.array_equal(
        compressed.batch.ancestry_ids.data,
        ancestry.data[source_indices],
    )
    assert compressed_estimate.provenance.method == "compressed"
    assert isinstance(
        compressed_estimate.diagnostics,
        phx.integration.CompressedIntegrationDiagnostics,
    )


def test_compression_preserves_nonnormalized_target_mass():
    samples = jnp.linspace(0.0, 2.0, 25)
    target = phx.integration.weighted(
        samples,
        jnp.zeros((25,)),
        normalized=False,
        target_mass=jnp.asarray(7.0),
    )
    source = phx.integration.materialize(target)
    compressed = phx.integration.compress(
        source,
        phx.coresets.MomentRecombination(),
        features=jnp.stack((samples, samples**2), axis=1),
    )

    source_estimate = phx.integration.reduce(lambda value: value**2, source)
    compressed_estimate = phx.integration.reduce(lambda value: value**2, compressed)

    assert jnp.isclose(compressed.target.target_mass, 7.0)
    assert jnp.allclose(
        compressed_estimate.value.data,
        source_estimate.value.data,
        atol=1e-11,
    )


def test_compression_lowers_a_named_discrete_point_measure():
    domain = phx.domain.Interval1d(0.0, 1.0)
    layout = phx.domain.SampleLayout((("x",),))
    points = domain.component().sample(
        phx.domain.PointSampling(16, layout=layout, design="halton"),
        key=jr.key(11),
    )
    axis = points.points["x"].dims[0]
    coordinates = points.points["x"].data[:, 0]
    weights = cx.Field(jnp.linspace(1.0, 2.0, 16), dims=(axis,))
    target = phx.integration.discrete(
        points,
        weights,
        axes=axis,
        normalized=True,
    )
    source = phx.integration.materialize(target)
    features = cx.Field(
        jnp.stack((coordinates, coordinates**2), axis=1),
        dims=(axis, "feature"),
    )
    compressed = phx.integration.compress(
        source,
        phx.coresets.MomentRecombination(),
        features=features,
    )

    @domain.Function("x")
    def square(x):
        return x[0] ** 2

    source_estimate = phx.integration.reduce(square, source)
    compressed_estimate = phx.integration.reduce(square, compressed)

    assert compressed.batch.num_samples <= 3
    assert jnp.allclose(
        compressed_estimate.value.data,
        source_estimate.value.data,
        atol=1e-11,
    )


def test_compression_preserves_proposal_support_failure():
    samples = jnp.linspace(0.0, 1.0, 9)
    target = phx.integration.weighted(
        samples,
        jnp.zeros((9,)),
        support_valid=jnp.asarray(False),
    )
    compressed = phx.integration.compress(
        phx.integration.materialize(target),
        phx.coresets.MomentRecombination(),
    )
    estimate = phx.integration.reduce(lambda value: value, compressed)

    assert not bool(compressed.batch.support_valid)
    assert int(estimate.status) == int(
        phx.integration.IntegrationStatus.PROPOSAL_SUPPORT_FAILURE
    )


@pytest.mark.parametrize("identifier", ["stratum_ids", "pair_ids", "replicate_ids"])
def test_compression_rejects_unpreserved_sample_grouping(identifier):
    samples = jnp.linspace(0.0, 1.0, 12)
    identifiers = jnp.arange(12, dtype=jnp.int32) // 2
    target = phx.integration.weighted(
        samples,
        jnp.zeros((12,)),
        **{identifier: identifiers},
    )

    with pytest.raises(ValueError, match="compressed"):
        phx.integration.compress(
            phx.integration.materialize(target),
            phx.coresets.MomentRecombination(),
        )
