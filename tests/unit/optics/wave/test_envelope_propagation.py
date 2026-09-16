#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import FourierAxisSpec, TensorGridPlan
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import (
    AdaptiveEnvelopePolicy,
    envelope_spectrum,
    envelope_time_values,
    EnvelopeNonlinearResponsePlan,
    EnvelopePropagationPlan,
    evaluate_envelope_nonlinear_response,
    PlaneFieldSpace,
    prepare_envelope_nonlinear_response,
    prepare_envelope_propagation,
    propagate_envelope,
    propagate_envelope_adaptive,
    PulseEnvelopeField,
    PulseTimeSpace,
)


def _plane():
    grid = TensorGridPlan(
        (FourierAxisSpec(2), FourierAxisSpec(2)), axis_names=("u", "v")
    ).prepare(jnp.asarray([[-1.0, -1.0], [1.0, 1.0]]))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "periodic-cell")


def _time(size=256, half_width=20.0):
    grid = TensorGridPlan((FourierAxisSpec(size),), axis_names=("time",)).prepare(
        jnp.asarray([[-half_width], [half_width]])
    )
    return PulseTimeSpace(grid, topology="periodic-cell")


def _field(time_space, values, *, carrier=100.0):
    plane = _plane()
    return PulseEnvelopeField(
        plane,
        time_space,
        jnp.broadcast_to(values, plane.shape + time_space.shape),
        carrier,
        0.0,
        polarization="scalar",
    )


def test_linear_envelope_propagation_matches_exact_spectral_phase():
    time = _time(128, 10.0)
    values = jnp.exp(-0.5 * (time.coordinates / 1.5) ** 2).astype(jnp.complex128)
    field = _field(time, values)
    response = prepare_envelope_nonlinear_response(
        EnvelopeNonlinearResponsePlan(
            0.0,
            source_id="zero-nonlinearity-control",
        ),
        time,
    )
    prepared = prepare_envelope_propagation(
        EnvelopePropagationPlan(
            time,
            {2: 0.8},
            step_count=4,
            maximum_spectral_edge_fraction=1e-5,
            maximum_refinement_error=1e-11,
        ),
        response,
    )
    distance = 0.4
    result = propagate_envelope(prepared, field, distance)
    initial_spectrum = jnp.where(
        prepared.dealias_mask.reshape((1, 1, -1)),
        envelope_spectrum(field.values),
        0.0j,
    )
    expected = envelope_time_values(
        initial_spectrum
        * jnp.exp(prepared.linear_generator.reshape((1, 1, -1)) * distance)
    )
    np.testing.assert_allclose(result.field.values, expected, rtol=2e-12, atol=2e-12)
    assert bool(result.evidence.accepted)
    np.testing.assert_allclose(
        result.evidence.fixed_step_refinement_error, 0.0, atol=1e-13
    )


def test_fundamental_nls_soliton_preserves_intensity_and_adaptive_path():
    time = _time(256, 20.0)
    values = (1.0 / jnp.cosh(time.coordinates)).astype(jnp.complex128)
    field = _field(time, values)
    response = prepare_envelope_nonlinear_response(
        EnvelopeNonlinearResponsePlan(
            1.0,
            source_id="normalized-kerr-control",
        ),
        time,
    )
    prepared = prepare_envelope_propagation(
        EnvelopePropagationPlan(
            time,
            {2: -1.0},
            step_count=32,
            maximum_spectral_edge_fraction=1e-7,
            maximum_refinement_error=2e-5,
        ),
        response,
    )
    result = propagate_envelope(prepared, field, 0.2)
    np.testing.assert_allclose(
        jnp.abs(result.field.values) ** 2,
        jnp.abs(field.values) ** 2,
        rtol=2e-3,
        atol=2e-4,
    )
    assert bool(result.evidence.accepted)
    adaptive = propagate_envelope_adaptive(
        prepared,
        field,
        0.2,
        AdaptiveEnvelopePolicy(
            relative_tolerance=2e-6,
            absolute_tolerance=1e-8,
            initial_step=0.02,
            minimum_step=1e-5,
            maximum_step=0.05,
            maximum_attempts=100,
        ),
    )
    assert bool(adaptive.adaptive.successful)
    np.testing.assert_allclose(adaptive.adaptive.final_distance, 0.2, atol=1e-12)
    np.testing.assert_allclose(
        jnp.abs(adaptive.field.values) ** 2,
        jnp.abs(field.values) ** 2,
        rtol=3e-3,
        atol=3e-4,
    )


def test_delayed_raman_and_self_steepening_response_is_explicit_and_finite():
    time = _time(128, 10.0)
    coordinates = time.coordinates
    kernel = jnp.where(coordinates >= 0.0, jnp.exp(-coordinates / 0.8), 0.0)
    response = prepare_envelope_nonlinear_response(
        EnvelopeNonlinearResponsePlan(
            1.2,
            raman_fraction=0.18,
            raman_response=kernel,
            self_steepening=True,
            source_id="causal-exponential-raman-control",
        ),
        time,
    )
    values = jnp.exp(-0.5 * coordinates**2).astype(jnp.complex128)
    offsets = jnp.asarray(2.0 * np.pi * np.fft.fftfreq(time.size, d=time.sample_spacing))
    evaluation = evaluate_envelope_nonlinear_response(
        response,
        jnp.broadcast_to(values, (2, 2, time.size)),
        offsets,
        100.0,
    )
    assert bool(evaluation.evidence.finite)
    assert float(evaluation.evidence.delayed_energy) > 0.0
    assert jnp.all(jnp.isfinite(evaluation.source_spectrum))
