#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.applications.skeletal_muscle import (
    skeletal_muscle_quantity,
    SkeletalMuscleQuantitySpec,
)
from phydrax.applications.skeletal_muscle.personalization import (
    SkeletalMultimodalLikelihoodPlan,
    SkeletalObservationChannel,
)
from phydrax.uq import (
    CompositePosteriorLikelihood,
    FixedObservationLikelihood,
    GaussianLikelihood,
    ParameterSpace,
)


def _plan():
    force = SkeletalObservationChannel(
        "force",
        "observed_force",
        "load-cell-session",
        jnp.asarray((100, 120, 140)),
        jnp.asarray(2.0),
        jnp.asarray((True, True, True)),
    )
    emg = SkeletalObservationChannel(
        "surface-emg",
        "surface_electric_potential",
        "electrode-session",
        jnp.asarray((1.0e-4, jnp.nan, 5.0e-5)),
        jnp.asarray((1.0e-5, jnp.nan, 1.0e-5)),
        jnp.asarray((True, False, True)),
    )
    return SkeletalMultimodalLikelihoodPlan((force, emg))


def _predictions():
    base_force = jnp.asarray((100.0, 120.0, 140.0))
    base_emg = jnp.asarray((1.0e-4, jnp.nan, 5.0e-5))
    return {
        "force": lambda value: value["force_scale"] * base_force,
        "surface-emg": lambda value: base_emg,
    }


def test_multimodal_plan_assembles_core_likelihood_terms_and_posterior():
    plan = _plan()
    predictions = _predictions()
    terms = plan.likelihood_terms(predictions)

    assert all(isinstance(term, FixedObservationLikelihood) for term in terms)
    assert all(isinstance(term.likelihood, GaussianLikelihood) for term in terms)
    assert tuple(term.label for term in terms) == plan.channel_ids
    assert tuple(term.target.shape for term in terms) == ((3,), (2,))
    assert (
        plan.channels[0].quantity_id
        == skeletal_muscle_quantity("observed_force").quantity_id
    )
    assert (
        plan.channels[1].quantity_id
        == skeletal_muscle_quantity("surface_electric_potential").quantity_id
    )

    space = ParameterSpace(
        {"force_scale": jnp.asarray(1.0)},
        log_prior=lambda value: -0.5 * ((value["force_scale"] - 1.0) / 0.2) ** 2,
    )
    posterior = plan.posterior(space, predictions)
    value, gradient = posterior.validate()

    assert isinstance(posterior.log_likelihood_fn, CompositePosteriorLikelihood)
    assert posterior.parameter_space is space
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient["force_scale"])
    assert jnp.abs(gradient["force_scale"]) < 1.0e-10
    assert jnp.isclose(
        posterior.log_density(posterior.initial_position),
        posterior.log_likelihood(space.initial) + space.log_prior(space.initial),
    )
    predicted = posterior.predict(posterior.initial_position)
    assert tuple(predicted) == plan.channel_ids
    assert jnp.isnan(predicted["surface-emg"][1])

    compiled = eqx.filter_jit(posterior.log_density)(posterior.initial_position)
    assert jnp.isclose(compiled, value)
    hessian = jax.hessian(posterior.negative_log_density)(posterior.initial_position)
    assert hessian["force_scale"]["force_scale"] > 0.0


def test_integer_observations_keep_float_uncertainty_and_gradients():
    channel = SkeletalObservationChannel(
        "integer-observation",
        skeletal_muscle_quantity("relative_isometric_force"),
        "integer-sensor",
        jnp.asarray((1, 2), dtype=jnp.int32),
        0.5,
        jnp.asarray((True, True)),
    )
    plan = SkeletalMultimodalLikelihoodPlan((channel,))
    space = ParameterSpace(
        {"location": jnp.asarray(0.0)},
        log_prior=lambda value: jnp.zeros_like(value["location"]),
    )
    posterior = plan.posterior(
        space,
        {
            "integer-observation": lambda value: jnp.broadcast_to(
                value["location"],
                (2,),
            )
        },
    )
    _, gradient = posterior.validate()

    assert jnp.issubdtype(channel.values.dtype, jnp.floating)
    assert jnp.issubdtype(channel.standard_uncertainty.dtype, jnp.floating)
    assert jnp.all(channel.standard_uncertainty == 0.5)
    assert jnp.isfinite(gradient["location"])
    assert gradient["location"] != 0.0


def test_masked_nonfinite_data_scale_and_prediction_are_inactive():
    channel = SkeletalObservationChannel(
        "masked",
        "observed_force",
        "partially-valid-sensor",
        jnp.asarray((2.0, jnp.nan)),
        jnp.asarray((0.5, jnp.nan)),
        jnp.asarray((True, False)),
    )
    plan = SkeletalMultimodalLikelihoodPlan((channel,))
    space = ParameterSpace(
        {"location": jnp.asarray(2.0)},
        log_prior=lambda value: -0.5 * (value["location"] - 2.0) ** 2,
    )
    posterior = plan.posterior(
        space,
        {"masked": lambda value: jnp.stack((value["location"], jnp.asarray(jnp.nan)))},
    )
    value, gradient = posterior.validate()

    assert jnp.all(jnp.isfinite(channel.values))
    assert jnp.all(jnp.isfinite(channel.standard_uncertainty))
    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient["location"])
    assert gradient["location"] == 0.0


def test_channels_reject_unknown_quantities_complex_data_and_incomplete_maps():
    with pytest.raises(KeyError, match="Unknown skeletal-muscle quantity"):
        SkeletalObservationChannel(
            "unknown",
            "unregistered_quantity",
            "sensor",
            jnp.asarray((1.0,)),
            1.0,
            jnp.asarray((True,)),
        )
    unregistered_spec = SkeletalMuscleQuantitySpec(
        "unregistered_force",
        "force",
        skeletal_muscle_quantity("observed_force").unit,
        sign_convention="positive measurement-axis direction",
        support_association="force transducer samples",
        reference_configuration="unregistered test asset",
    )
    with pytest.raises(KeyError, match="Unknown skeletal-muscle quantity"):
        SkeletalObservationChannel(
            "unregistered-spec",
            unregistered_spec,
            "sensor",
            jnp.asarray((1.0,)),
            1.0,
            jnp.asarray((True,)),
        )
    with pytest.raises(TypeError, match="must be real"):
        SkeletalObservationChannel(
            "complex",
            "observed_force",
            "sensor",
            jnp.asarray((1.0 + 1.0j,)),
            1.0,
            jnp.asarray((True,)),
        )

    plan = _plan()
    with pytest.raises(ValueError, match="exactly the planned channel IDs"):
        plan.likelihood_terms({"force": _predictions()["force"]})
