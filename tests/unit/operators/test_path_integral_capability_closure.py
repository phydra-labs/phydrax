# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import pytest

from phydrax.discretization import TemporalMesh
from phydrax.geometry import Sphere
from phydrax.operators.path_integral import (
    CompactU1GaugeMeasure,
    estimate_exchange_observable,
    ExchangePathPlan,
    interval_heat_kernel,
    killed_path_mask,
    periodic_path_action,
    PeriodicPathPlan,
    prepare_path_boundary_schedule,
    PreparedGeometryPathKernel,
    real_time_kernel_from_noise,
    RealTimePathIntegralPlan,
    source_feynman_kac_from_paths,
    source_feynman_kac_from_stochastic_paths,
    wilson_loop,
)
from phydrax.stochastic import StochasticPathEnsembleResult


def test_regulated_real_time_is_finite_slice_with_phase_evidence():
    mesh = TemporalMesh.uniform(0.0, 0.2, 2, role="path")
    plan = RealTimePathIntegralPlan(
        mesh,
        mass=1.0,
        regulator=0.5,
        num_paths=4,
    )
    noise = jnp.zeros((4, 2, 1))
    result = real_time_kernel_from_noise(
        noise,
        jnp.array([0.0]),
        jnp.array([0.1]),
        lambda q, t: jnp.asarray(0.0),
        plan=plan,
    )
    assert bool(result.valid)
    assert result.covariance.shape == (2, 2)
    assert result.claim == "regulated-finite-slice-only"
    assert float(result.regulator) == 0.5


def test_source_feynman_kac_separates_terminal_and_duhamel_terms():
    mesh = TemporalMesh.uniform(0.0, 1.0, 2, role="path")
    paths = jnp.zeros((8, 3, 1))
    result = source_feynman_kac_from_paths(
        lambda x, t: jnp.asarray(0.0),
        lambda x, t: jnp.asarray(2.0),
        paths,
        slicing=mesh,
    )
    assert jnp.allclose(result.estimate.value, 2.0)
    assert jnp.allclose(result.terminal_term, 0.0)
    assert jnp.allclose(result.source_term, 2.0)


def test_interval_images_periodic_ring_and_finite_u1_invariants():
    reflecting = PreparedGeometryPathKernel(
        0.0,
        1.0,
        behavior="reflecting",
        image_capacity=8,
    )
    kernel = interval_heat_kernel(reflecting, 0.3, 0.7, 0.2)
    assert bool(kernel.valid)
    assert bool(kernel.mass_conserving_boundary)
    assert kernel.omitted_tail_bound >= 0.0

    periodic = PeriodicPathPlan(
        4,
        1.0,
        zero_mode="fixed-centroid",
        fixed_centroid=(0.0,),
    )
    beads = jnp.array([[-1.0], [0.0], [1.0], [0.0]])
    action = periodic_path_action(beads, lambda value: jnp.sum(value**2), plan=periodic)
    rotated = periodic_path_action(
        jnp.roll(beads, 1, axis=0), lambda value: jnp.sum(value**2), plan=periodic
    )
    assert jnp.allclose(action, rotated)

    plaquette_edge = jnp.array([[1, 1, 1]], dtype=jnp.int32)
    vertex_edge = jnp.array([[-1, 0, 1], [1, -1, 0], [0, 1, -1]], dtype=jnp.int32)
    gauge = CompactU1GaugeMeasure(plaquette_edge, vertex_edge, beta=0.7)
    links = jnp.array([0.2, -0.4, 0.1])
    transformed = gauge.gauge_transform(links, jnp.array([0.3, -0.2, 0.1]))
    assert jnp.allclose(gauge.action(links), gauge.action(transformed))
    assert jnp.allclose(jnp.abs(wilson_loop(links, plaquette_edge[0])), 1.0)


def test_finite_exchange_sector_reports_sign_collapse_without_repair():
    plan = ExchangePathPlan(
        jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        statistics="fermion",
        require_full_enumeration=True,
    )
    result = estimate_exchange_observable(
        jnp.array([1.0, 2.0]),
        jnp.zeros((2,)),
        plan.characters,
        plan=plan,
    )
    assert not bool(result.valid)
    assert result.sector == "full-enumeration"


def test_single_active_signed_ratio_sample_has_no_uncertainty_evidence():
    plan = ExchangePathPlan(
        jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        statistics="fermion",
        require_full_enumeration=True,
    )
    result = estimate_exchange_observable(
        jnp.array([2.0, 123.0]),
        jnp.array([0.0, 1000.0]),
        jnp.array([1.0, 0.0]),
        plan=plan,
    )

    assert jnp.allclose(result.value, 2.0)
    assert not bool(result.valid)
    assert jnp.isnan(result.standard_error)


def test_signed_ratio_uncertainty_has_inverse_sqrt_sample_scaling():
    plan = ExchangePathPlan(
        jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
        statistics="fermion",
        require_full_enumeration=True,
    )
    signs_four = jnp.array([1.0, 1.0, 1.0, -1.0])
    signs_eight = jnp.tile(signs_four, 2)
    four_values = jnp.sqrt(3.0 / 8.0) * jnp.array([1.0, -1.0, 0.0, 0.0])
    eight_values = jnp.sqrt(7.0 / 16.0) * jnp.tile(jnp.array([1.0, -1.0, 0.0, 0.0]), 2)
    four = estimate_exchange_observable(
        four_values,
        jnp.zeros_like(four_values),
        signs_four,
        plan=plan,
    )
    eight = estimate_exchange_observable(
        eight_values,
        jnp.zeros_like(eight_values),
        signs_eight,
        plan=plan,
    )

    assert jnp.allclose(four.standard_error, 1.0 / jnp.sqrt(4.0))
    assert jnp.allclose(eight.standard_error, 1.0 / jnp.sqrt(8.0))
    assert jnp.allclose(four.standard_error / eight.standard_error, jnp.sqrt(2.0))


def test_path_boundary_convention_keeps_negative_interior_until_outward_exit():
    mask = killed_path_mask(
        jnp.array(
            [
                [-1.0, -0.25, 0.0, 0.1],
                [-1.0, -0.5, -0.1, -0.01],
            ]
        )
    )
    assert jnp.array_equal(
        mask,
        jnp.array(
            [
                [True, True, True, False],
                [True, True, True, True],
            ]
        ),
    )

    geometry = Sphere((0.0, 0.0, 0.0), 1.0, feature_id="path-domain").compile()
    absorbing = prepare_path_boundary_schedule(
        geometry,
        "absorbing",
        lambda time, state, args: jnp.array([1.0, 0.0, 0.0]),
        maximum_events=1,
        plan_id="absorbing-outward-crossing",
    )
    absorbing_result = absorbing.localize(
        lambda time, args: jnp.array([time, 0.0, 0.0]),
        jnp.array([[0.5, 1.5]]),
    )
    assert absorbing.events[0].direction == 1
    assert int(absorbing_result.event_count) == 1
    assert bool(absorbing_result.terminal[0])
    assert jnp.allclose(absorbing_result.event_times[0], 1.0, atol=1e-8)

    def specular_state(time, args):
        del args
        return jnp.array([time, 0.0, 0.0, 1.0, 0.0, 0.0])

    def kinetic_field(time, state, args):
        del time, args
        return jnp.concatenate((state[3:], jnp.zeros((3,))))

    specular = prepare_path_boundary_schedule(
        geometry,
        "specular",
        kinetic_field,
        maximum_events=1,
        plan_id="specular-outward-crossing",
    )
    specular_result = specular.localize(
        specular_state,
        jnp.array([[0.5, 1.5]]),
    )
    assert specular.events[0].direction == 1
    assert int(specular_result.event_count) == 1
    assert jnp.allclose(specular_result.event_times[0], 1.0, atol=1e-8)
    assert jnp.allclose(
        specular_result.event_states_after[0, 3:],
        jnp.array([-1.0, 0.0, 0.0]),
        atol=1e-8,
    )


def _stochastic_paths(times):
    return StochasticPathEnsembleResult(
        solution=None,
        states=jnp.zeros((2, 3, 1)),
        times=times,
        path_valid=jnp.ones((2,), dtype=bool),
        status=jnp.zeros((2,), dtype=jnp.int32),
        accepted_steps=jnp.ones((2,), dtype=jnp.int32),
        rejected_steps=jnp.zeros((2,), dtype=jnp.int32),
        temporal_evidence=None,
        event_mask=None,
        path_count=2,
        plan_id="shared-time-plan",
        prepared_id="shared-time-prepared",
        result_id="shared-time-result",
        realization_id="shared-time-realization",
        coupling_id="shared-time-coupling",
        approximation_kind="finite-keyed-path-ensemble",
    )


def test_adaptive_feynman_kac_extracts_and_validates_common_time_axis():
    common = jnp.array([0.0, 0.5, 1.0])
    result = source_feynman_kac_from_stochastic_paths(
        _stochastic_paths(jnp.broadcast_to(common, (2, 3))),
        lambda state, time: jnp.asarray(0.0),
        lambda state, time: jnp.asarray(1.0),
    )
    assert bool(result.valid)
    assert jnp.allclose(result.source_estimate.estimate.value, 1.0)

    inconsistent = jnp.array([[0.0, 0.5, 1.0], [0.0, 0.6, 1.0]])
    with pytest.raises(ValueError, match="common time axis"):
        source_feynman_kac_from_stochastic_paths(
            _stochastic_paths(inconsistent),
            lambda state, time: jnp.asarray(0.0),
            lambda state, time: jnp.asarray(1.0),
        )
