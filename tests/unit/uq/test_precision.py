#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_predictive_storage_and_summary_precision_are_independent():
    policy = phx.uq.PredictivePrecisionPolicy(
        storage_dtype="float32",
        summary_dtype="float64",
    )
    predictive = phx.uq.PredictiveField(
        cx.Field(
            jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64),
            dims=("draw", "x"),
        ),
        (phx.uq.SampleAxis("draw", "epistemic"),),
        precision=policy,
    )

    assert predictive.samples.data.dtype == jnp.float32
    assert predictive.mean().data.dtype == jnp.float64
    assert predictive.variance().data.dtype == jnp.float64
    assert predictive.quantile(0.5).data.dtype == jnp.float64
    assert predictive.precision_evidence.evidence_id


def test_particle_state_statistics_and_decisions_use_distinct_dtypes(tmp_path):
    observations = phx.stochastic.ObservationSequence(
        jnp.asarray([0.5, 1.0]),
        jnp.asarray([[1.0], [2.0]]),
    )
    model = phx.stochastic.StateSpaceModel(
        phx.stochastic.GaussianStatePrior(
            jnp.asarray([0.0]),
            jnp.asarray([[1.0]]),
            state_shape=(1,),
        ),
        phx.stochastic.LinearGaussianTransitionKernel(
            jnp.asarray([[1.0]]),
            jnp.asarray([[0.2]]),
            state_shape=(1,),
        ),
        phx.stochastic.LinearGaussianObservationModel(
            jnp.asarray([[1.0]]),
            jnp.asarray([[0.5]]),
            state_shape=(1,),
            observation_shape=(1,),
        ),
        model_id="precision-state-space",
    )
    problem = phx.stochastic.StateSpaceProblem(
        model,
        observations,
        initial_time=0.0,
        problem_id="precision-particle-problem",
    )
    precision = phx.uq.ParticlePrecisionPolicy(
        state_storage_dtype="float32",
        statistics_dtype="float64",
        decision_dtype="float64",
    )

    result = phx.uq.bootstrap_particle_filter(
        jr.key(3),
        problem,
        num_particles=128,
        precision=precision,
    )

    assert result.particles.dtype == jnp.float32
    assert result.log_weights.dtype == jnp.float64
    assert result.effective_sample_sizes.dtype == jnp.float64
    assert result.precision_evidence.evidence_id

    checkpoint = phx.uq.write_particle_filter_checkpoint(
        tmp_path / "particle-state",
        problem,
        result.final_state,
    )
    restored = phx.uq.read_particle_filter_checkpoint(
        checkpoint,
        problem,
        num_particles=128,
        precision=precision,
    )
    assert restored.particles.dtype == jnp.float32
    assert restored.log_weights.dtype == jnp.float64
    assert restored.precision.policy_id == precision.policy_id

    with pytest.raises(phx.uq.CheckpointCompatibilityError, match="precision policy"):
        phx.uq.read_particle_filter_checkpoint(
            checkpoint,
            problem,
            num_particles=128,
        )
