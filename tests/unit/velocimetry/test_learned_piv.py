#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.velocimetry.imaging._types import (
    DenseDisplacementField2D,
    ImageGeometry2D,
)
from phydrax.velocimetry.imaging._warp import image_coordinates
from phydrax.velocimetry.piv._learned_model import (
    CorrelationPyramidPIV,
    LearnedDensePIVPlan,
)
from phydrax.velocimetry.piv._learned_primitives import (
    backward_warp_2d,
    build_cost_volume_2d,
    CostVolumePlan,
    MultiScaleRobustPIVLoss,
    PIV_LOSS_INSUFFICIENT_SUPPORT,
    resize_displacement_2d,
)
from phydrax.velocimetry.piv._learned_qualification import qualify_learned_piv
from phydrax.velocimetry.piv._learned_training import (
    fit_learned_piv,
    LearnedPIVDataset,
    LearnedPIVTrainingConfig,
)


def _small_model(key=jr.key(4)):
    plan = LearnedDensePIVPlan(
        (8, 8),
        level_count=2,
        search_radius=1,
        cost_volume_chunk_size=4,
    )
    return CorrelationPyramidPIV(
        plan,
        feature_channels=3,
        refinement_channels=4,
        key=key,
    )


def test_backward_warp_has_pullback_sign_and_nonperiodic_support():
    image = jnp.broadcast_to(jnp.arange(5.0)[None, :], (4, 5))
    displacement = jnp.zeros((4, 5, 2)).at[..., 1].set(1.0)

    warped = backward_warp_2d(image, displacement)

    np.testing.assert_allclose(warped.values[:, 1:], image[:, :-1])
    np.testing.assert_array_equal(warped.values[:, 0], 0.0)
    np.testing.assert_array_equal(warped.valid[:, 1:], True)
    np.testing.assert_array_equal(warped.valid[:, 0], False)


def test_displacement_resize_scales_row_and_column_components_independently():
    displacement = jnp.ones((2, 4, 2)) * jnp.asarray((1.0, 2.0))

    resized = resize_displacement_2d(displacement, (4, 2))

    np.testing.assert_allclose(resized[..., 0], 2.0)
    np.testing.assert_allclose(resized[..., 1], 1.0)


def test_cost_volume_channel_offsets_are_first_to_second_row_column_candidates():
    reference = jnp.zeros((5, 5, 1)).at[2, 2, 0].set(1.0)
    target = jnp.zeros((5, 5, 1)).at[2, 3, 0].set(1.0)
    plan = CostVolumePlan(1, chunk_size=4)

    volume = build_cost_volume_2d(reference, target, plan)
    positive_column = int(
        jnp.where(jnp.all(plan.offsets_rc == jnp.asarray((0, 1)), axis=-1))[0][0]
    )
    zero = int(jnp.where(jnp.all(plan.offsets_rc == jnp.asarray((0, 0)), axis=-1))[0][0])

    np.testing.assert_allclose(volume.values[2, 2, positive_column], 1.0)
    np.testing.assert_allclose(volume.values[2, 2, zero], 0.0)
    np.testing.assert_array_equal(volume.offsets_rc, plan.offsets_rc)
    assert volume.plan_id == plan.plan_id
    assert bool(volume.valid[2, 2, positive_column])


def test_all_invalid_multiscale_loss_is_zero_with_finite_zero_gradient():
    image = jnp.arange(36.0).reshape((6, 6, 1))
    displacement = jnp.zeros((6, 6, 2))
    invalid = jnp.zeros((6, 6), dtype=bool)
    loss = MultiScaleRobustPIVLoss(
        supervised_weight=1.0,
        photometric_weight=1.0,
        consistency_weight=1.0,
        smoothness_weight=1.0,
    )

    def objective(candidate):
        return loss(
            image,
            image,
            (candidate,),
            (-candidate,),
            first_valid=invalid,
            second_valid=invalid,
            target_forward_rc=jnp.zeros_like(candidate),
            target_valid=invalid,
        ).total

    value, gradient = jax.value_and_grad(objective)(displacement)
    evidence = loss(
        image,
        image,
        (displacement,),
        (-displacement,),
        first_valid=invalid,
        second_valid=invalid,
        target_forward_rc=jnp.zeros_like(displacement),
        target_valid=invalid,
    )

    np.testing.assert_allclose(value, 0.0)
    np.testing.assert_allclose(gradient, 0.0)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert not bool(evidence.valid)
    assert int(evidence.status) == PIV_LOSS_INSUFFICIENT_SUPPORT


def test_correlation_pyramid_initialization_and_prediction_are_deterministic():
    first = jnp.linspace(0.0, 1.0, 64).reshape((8, 8, 1))
    second = jnp.roll(first, 1, axis=1)
    model_a = _small_model(jr.key(9))
    model_b = _small_model(jr.key(9))

    prediction_a = model_a(model_a.plan.prepare(first, second))
    prediction_b = model_b(model_b.plan.prepare(first, second))

    np.testing.assert_allclose(prediction_a.displacement_rc, prediction_b.displacement_rc)
    np.testing.assert_array_equal(prediction_a.valid, prediction_b.valid)
    leaves_a = jax.tree_util.tree_leaves(eqx.filter(model_a, eqx.is_array))
    leaves_b = jax.tree_util.tree_leaves(eqx.filter(model_b, eqx.is_array))
    assert len(leaves_a) == len(leaves_b)
    for first_leaf, second_leaf in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_allclose(first_leaf, second_leaf)


def test_model_objective_has_finite_gradients_and_training_is_reproducible():
    base = jnp.linspace(0.0, 1.0, 64).reshape((8, 8, 1))
    first_images = jnp.stack((base, jnp.flip(base, axis=0)))
    second_images = jnp.roll(first_images, 1, axis=2)
    target = jnp.zeros((2, 8, 8, 2)).at[..., 1].set(1.0)
    dataset = LearnedPIVDataset(
        first_images,
        second_images,
        target_forward_rc=target,
        partition="training",
    )
    loss = MultiScaleRobustPIVLoss(
        scale_weights=(0.5, 1.0),
        supervised_weight=1.0,
        photometric_weight=0.1,
        consistency_weight=0.1,
        smoothness_weight=0.01,
    )
    config = LearnedPIVTrainingConfig(
        maximum_steps=1,
        batch_size=1,
        learning_rate=1e-3,
        loss=loss,
        jit=False,
    )
    model_a = _small_model(jr.key(12))
    model_b = _small_model(jr.key(12))

    fit_a = fit_learned_piv(model_a, dataset, config, key=jr.key(22))
    fit_b = fit_learned_piv(model_b, dataset, config, key=jr.key(22))

    np.testing.assert_allclose(fit_a.evidence.total_loss, fit_b.evidence.total_loss)
    np.testing.assert_allclose(fit_a.evidence.gradient_norm, fit_b.evidence.gradient_norm)
    assert bool(jnp.all(jnp.isfinite(fit_a.evidence.total_loss)))
    assert bool(jnp.all(jnp.isfinite(fit_a.evidence.gradient_norm)))
    trained_a = jax.tree_util.tree_leaves(eqx.filter(fit_a.model, eqx.is_array))
    trained_b = jax.tree_util.tree_leaves(eqx.filter(fit_b.model, eqx.is_array))
    for first_leaf, second_leaf in zip(trained_a, trained_b, strict=True):
        np.testing.assert_allclose(first_leaf, second_leaf)


def test_held_out_qualification_returns_neutral_canonical_dense_fields():
    geometry = ImageGeometry2D((8, 8), geometry_id="held-out-geometry")
    image = jnp.linspace(0.0, 1.0, 64).reshape((1, 8, 8, 1))
    target = jnp.zeros((1, 8, 8, 2))
    dataset = LearnedPIVDataset(
        image,
        image,
        target_forward_rc=target,
        geometry=geometry,
        scenario_ids=("held-out-still-particles",),
        partition="held-out",
    )
    model = _small_model(jr.key(31))

    qualification = qualify_learned_piv(model, dataset)
    direct = model(model.plan.prepare(image[0], image[0]))
    field = qualification.fields[0]

    assert isinstance(field, DenseDisplacementField2D)
    assert field.geometry_id == geometry.geometry_id
    np.testing.assert_allclose(field.positions_rc, image_coordinates(geometry))
    np.testing.assert_allclose(field.displacement_rc, direct.displacement_rc)
    np.testing.assert_array_equal(field.valid, direct.valid)
    assert qualification.scenario_ids == ("held-out-still-particles",)
    assert qualification.endpoint_error.shape == (1,)
    assert bool(jnp.all(jnp.isfinite(qualification.endpoint_error)))
