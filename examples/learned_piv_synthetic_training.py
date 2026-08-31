#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


base = jnp.linspace(0.0, 1.0, 64).reshape((8, 8, 1))
first = jnp.stack((base, jnp.flip(base, axis=0)))
second = jnp.roll(first, 1, axis=2)
target = jnp.zeros((2, 8, 8, 2)).at[..., 1].set(1.0)
dataset = phx.velocimetry.piv.LearnedPIVDataset(
    first,
    second,
    target_forward_rc=target,
    partition="training",
)
plan = phx.velocimetry.piv.LearnedDensePIVPlan(
    (8, 8),
    level_count=2,
    search_radius=1,
    cost_volume_chunk_size=4,
)
model = phx.velocimetry.piv.CorrelationPyramidPIV(
    plan,
    feature_channels=3,
    refinement_channels=4,
    key=jr.key(12),
)
loss = phx.velocimetry.piv.MultiScaleRobustPIVLoss(
    scale_weights=(0.5, 1.0),
    supervised_weight=1.0,
    photometric_weight=0.1,
    consistency_weight=0.1,
    smoothness_weight=0.01,
)
configuration = phx.velocimetry.piv.LearnedPIVTrainingConfig(
    maximum_steps=1,
    batch_size=1,
    learning_rate=1e-3,
    loss=loss,
    jit=False,
)
fit = phx.velocimetry.piv.fit_learned_piv(
    model,
    dataset,
    configuration,
    key=jr.key(22),
)
prediction = fit.model(fit.model.plan.prepare(first[0], second[0]))

print("training loss", fit.evidence.total_loss)
print("gradient norm", fit.evidence.gradient_norm)
print("valid prediction fraction", float(jnp.mean(prediction.valid)))
print("finite", bool(jnp.all(jnp.isfinite(prediction.displacement_rc))))
