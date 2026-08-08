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
from phydrax._trainable import partition_trainable


def _small_mlp(key, *, width=6):
    return phx.nn.models.MLP(
        in_size=2,
        out_size=1,
        width_size=width,
        depth=1,
        key=key,
    )


def test_homogeneous_and_heterogeneous_ensembles_share_predictive_contract():
    homogeneous = phx.uq.HomogeneousFunctionEnsemble.from_factory(
        _small_mlp,
        num_members=4,
        key=jr.key(0),
        source_dim="member",
    )
    heterogeneous = phx.uq.HeterogeneousFunctionEnsemble(
        (_small_mlp(jr.key(1), width=4), _small_mlp(jr.key(2), width=8)),
        source_dim="member",
    )
    x = jnp.asarray([0.1, 0.2])

    homogeneous_prediction = homogeneous.predict(x, key=jr.key(3))
    heterogeneous_prediction = heterogeneous.predict(x, key=jr.key(4))

    assert homogeneous_prediction.samples.dims == ("member", None)
    assert homogeneous_prediction.samples.data.shape == (4, 1)
    assert heterogeneous_prediction.samples.data.shape == (2, 1)


def test_ensemble_predictions_record_or_raise_for_invalid_members():
    ensemble = phx.uq.HeterogeneousFunctionEnsemble(
        (
            lambda value, *, key: jnp.asarray(value) + 1.0,
            lambda value, *, key: jnp.full_like(jnp.asarray(value), jnp.nan),
        ),
        source_dim="member",
    )
    recorded = ensemble.predict(jnp.asarray([1.0, 2.0]), key=jr.key(4))

    assert jnp.array_equal(recorded.valid.data, jnp.asarray([True, False]))
    assert jnp.array_equal(recorded.mean().data, jnp.asarray([2.0, 3.0]))
    with pytest.raises(FloatingPointError, match="invalid realizations"):
        ensemble.predict(
            jnp.asarray([1.0, 2.0]),
            key=jr.key(4),
            valid_policy="raise",
        )


def test_randomized_prior_is_structurally_nontrainable_and_members_are_independent():
    model = phx.uq.RandomizedPriorModel(
        _small_mlp(jr.key(5)),
        _small_mlp(jr.key(6)),
        beta=0.5,
    )
    trainable, _ = partition_trainable(model)
    learned_leaves = jax.tree_util.tree_leaves(
        eqx.filter(model.learned, eqx.is_inexact_array)
    )
    trainable_leaves = jax.tree_util.tree_leaves(trainable)

    assert len(trainable_leaves) == len(learned_leaves)
    assert jnp.isfinite(model(jnp.ones((2,))).all())
    ensemble = phx.uq.randomized_prior_ensemble(
        _small_mlp,
        num_members=3,
        key=jr.key(7),
        beta=0.5,
    )
    prediction = ensemble.predict(jnp.ones((2,)), key=jr.key(8))
    assert jnp.var(prediction.samples.data, axis=0).item() > 0.0


def test_distribution_moments_probability_domain_and_joint_qmc_design():
    normal = phx.uq.Normal(1.0, 2.0)
    lognormal = phx.uq.LogNormal(0.2, 0.4)
    empirical = phx.uq.EmpiricalDistribution(
        jnp.asarray([0.0, 2.0]), probabilities=jnp.asarray([0.25, 0.75])
    )
    assert normal.mean == 1.0 and normal.variance == 4.0
    assert lognormal.mean > 0.0 and lognormal.variance > 0.0
    assert jnp.allclose(empirical.mean, 1.5)

    domain = phx.domain.ProbabilityDomain(normal, label="coefficient")
    draws = domain.sample(256, sampler="sobol_scrambled", key=jr.key(9))
    assert domain.measure == 1.0
    assert draws.shape == (256,)
    with pytest.raises(ValueError, match="unbounded"):
        domain.fixed("start")

    batch = phx.uq.sample_joint(
        {"a": normal, "b": phx.uq.Uniform(-1.0, 1.0)},
        num_samples=256,
        key=jr.key(10),
    )
    assert batch.values["a"].shape == (256,)
    assert not jnp.array_equal(batch.values["a"], batch.values["b"])


def test_lognormal_log_prob_is_safe_outside_support():
    distribution = phx.uq.LogNormal(0.0, 1.0)
    assert distribution.log_prob(-1.0) == -jnp.inf


def test_propagation_chunking_is_deterministic_and_records_invalid_draws():
    samples = phx.uq.sample_joint(
        {"x": phx.uq.Uniform(-1.0, 1.0), "y": phx.uq.Normal(0.0, 1.0)},
        num_samples=64,
        key=jr.key(11),
    )
    function = lambda x, y: jnp.asarray([x + y, x * y])

    whole = phx.uq.propagate(function, samples)
    chunked = phx.uq.propagate(function, samples, batch_size=7)
    assert jnp.array_equal(whole.samples.data, chunked.samples.data)
    assert whole.samples.data.shape == (64, 2)

    invalid = phx.uq.propagate(lambda x, y: jnp.log(x - 2.0), samples)
    assert not jnp.any(invalid.valid.data)
    with pytest.raises(FloatingPointError, match="invalid samples"):
        phx.uq.propagate(lambda x, y: jnp.log(x - 2.0), samples, valid_policy="raise")


def test_propagation_rejects_field_dimension_changes():
    samples = phx.uq.sample_joint(
        {"x": phx.uq.Uniform(0.0, 1.0)},
        num_samples=8,
        key=jr.key(13),
    )
    calls = 0

    def changing_dims(x):
        nonlocal calls
        calls += 1
        dim = "x" if calls == 1 else "y"
        return cx.Field(jnp.asarray([x]), dims=(dim,))

    with pytest.raises(ValueError, match="dimensions changed"):
        phx.uq.propagate(changing_dims, samples)


def test_sobol_jansen_matches_ishigami_reference_indices():
    distributions = {
        "x1": phx.uq.Uniform(-jnp.pi, jnp.pi),
        "x2": phx.uq.Uniform(-jnp.pi, jnp.pi),
        "x3": phx.uq.Uniform(-jnp.pi, jnp.pi),
    }

    def ishigami(x1, x2, x3):
        return jnp.sin(x1) + 7.0 * jnp.sin(x2) ** 2 + 0.1 * x3**4 * jnp.sin(x1)

    result = phx.uq.sobol_indices(
        ishigami,
        distributions,
        num_samples=4096,
        key=jr.key(12),
        batch_size=257,
    )
    expected_first = jnp.asarray([0.3139, 0.4424, 0.0])
    expected_total = jnp.asarray([0.5576, 0.4424, 0.2437])

    assert jnp.allclose(result.first_order.data, expected_first, atol=0.04)
    assert jnp.allclose(result.total_order.data, expected_total, atol=0.04)


def test_sobol_rejects_unknown_output_reduction():
    with pytest.raises(ValueError, match="reduce_output"):
        phx.uq.sobol_indices(
            lambda x: x,
            {"x": phx.uq.Uniform(0.0, 1.0)},
            num_samples=8,
            key=jr.key(14),
            reduce_output="median",
        )


def test_fit_ensemble_returns_deterministic_member_diagnostics_and_indexed_failures():
    class FittedMember(eqx.Module):
        value: jax.Array
        training_diagnostics: dict[str, jax.Array]

        def __call__(self, x, *, key=None):
            return self.value + x

    class Trainer:
        def solve(self, *, seed):
            return FittedMember(
                jnp.asarray(float(seed % 17)),
                {"final_loss": jnp.asarray(1.0)},
            )

    first = phx.uq.fit_ensemble(
        lambda key: Trainer(),
        num_members=3,
        key=jr.key(15),
        homogeneous=False,
        return_diagnostics=True,
    )
    second = phx.uq.fit_ensemble(
        lambda key: Trainer(),
        num_members=3,
        key=jr.key(15),
        homogeneous=False,
        return_diagnostics=True,
    )

    assert isinstance(first, phx.uq.EnsembleFitResult)
    assert first.ensemble.num_members == 3
    assert tuple(member.member_index for member in first.members) == (0, 1, 2)
    assert tuple(member.seed for member in first.members) == tuple(
        member.seed for member in second.members
    )
    assert len({member.seed for member in first.members}) == 3
    assert first.total_duration_seconds >= 0.0
    assert all("final_loss" in member.training_diagnostics for member in first.members)

    class FailingTrainer:
        def solve(self, *, seed):
            raise ValueError("training failed")

    with pytest.raises(phx.uq.EnsembleFitError) as error:
        phx.uq.fit_ensemble(
            lambda key: FailingTrainer(),
            num_members=2,
            key=jr.key(16),
            homogeneous=False,
            return_diagnostics=True,
        )
    assert error.value.member_index == 0
    assert error.value.completed == ()
    assert isinstance(error.value.__cause__, ValueError)
