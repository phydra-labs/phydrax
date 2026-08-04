import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def _stable_process():
    return phx.stochastic.SymmetricStableLevyProcess(
        1.3,
        jnp.asarray([0.25, 0.4]),
        drift=jnp.asarray([0.1, -0.2]),
        process_id="two-component-stable",
    )


def test_levy_series_extension_and_batch_growth_preserve_path_prefixes():
    process = _stable_process()
    base = phx.stochastic.LevyProcessRealization.from_process(
        process,
        jr.key(40),
        support=(0.0, 1.0),
        max_terms=32,
        sample_shape=(3,),
    )
    extended = base.extend(64)
    wider = phx.stochastic.LevyProcessRealization.from_process(
        process,
        jr.key(40),
        support=(0.0, 1.0),
        max_terms=32,
        sample_shape=(5,),
    )
    base_series = base.series(process)
    extended_series = extended.series(process)
    wider_series = wider.series(process)

    assert base.coupling_id == extended.coupling_id == wider.coupling_id
    assert base.realization_id != extended.realization_id
    assert jnp.array_equal(
        base_series.arrival_levels,
        extended_series.arrival_levels[..., : base.max_terms],
    )
    assert jnp.array_equal(
        base_series.times,
        extended_series.times[..., : base.max_terms],
    )
    assert jnp.array_equal(
        base_series.jumps,
        extended_series.jumps[..., : base.max_terms, :],
    )
    assert jnp.array_equal(base_series.jumps, wider_series.jumps[:3])


def test_levy_series_interval_queries_are_additive_and_cutoff_complete():
    process = _stable_process()
    realization = phx.stochastic.LevyProcessRealization.from_process(
        process,
        jr.key(41),
        support=(0.0, 1.0),
        max_terms=128,
        sample_shape=(8,),
    )
    series = realization.series(process)
    cutoff = 0.05
    halves = series.increments(
        jnp.asarray([0.0, 0.5]),
        jnp.asarray([0.5, 1.0]),
        cutoff=cutoff,
    )
    whole = series.increments(
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        cutoff=cutoff,
    )[:, 0]
    with_drift = realization.truncated_increments(
        process,
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        cutoff=cutoff,
    )[:, 0]

    assert jnp.all(series.complete_above(cutoff))
    assert jnp.allclose(jnp.sum(halves, axis=1), whole, rtol=0.0, atol=1e-12)
    assert jnp.allclose(with_drift - whole, process.drift, rtol=0.0, atol=1e-12)


def test_symmetric_stable_contract_matches_declared_characteristic_exponent():
    process = _stable_process()
    frequency = jnp.asarray([[0.7, -0.3], [1.2, 0.4]])
    expected = 1j * (frequency @ process.drift) - jnp.sum(
        process.scale**process.alpha * jnp.abs(frequency) ** process.alpha,
        axis=-1,
    )
    covariance = process.small_jump_covariance(0.1)

    assert jnp.allclose(process.characteristic_exponent(frequency), expected)
    assert covariance.shape == (2, 2)
    assert jnp.all(jnp.diag(covariance) > 0.0)
    assert jnp.array_equal(covariance, jnp.diag(jnp.diag(covariance)))


def test_levy_realization_composes_with_other_global_drivers():
    process = _stable_process()
    levy = phx.stochastic.LevyProcessRealization.from_process(
        process,
        jr.key(42),
        support=(0.0, 1.0),
        max_terms=32,
        sample_shape=(4,),
    )
    gaussian = levy.gaussian_realization()
    composite = phx.stochastic.CompositeStochasticRealization(
        {"levy": levy, "small-jump": gaussian}
    )

    assert phx.stochastic.is_stochastic_realization(levy)
    assert composite.sample_shape == (4,)
    assert composite.support == (0.0, 1.0)
    assert len(composite.path_labels) == 4
    assert composite.component("levy") is levy
