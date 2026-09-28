#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import get_args

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax import DimensionalScaleContract, RelativityScaleContract
from phydrax.discretization.pic import (
    PIC_CODE_RELATIVITY,
    RelativisticPusher,
    RelativisticPushPlan,
)


_METHODS: tuple[RelativisticPusher, ...] = get_args(RelativisticPusher)
_DRIFT_EXACT: tuple[RelativisticPusher, ...] = ("vay", "higuera-cary")


def _relativity(speed_of_light: int) -> RelativityScaleContract:
    return RelativityScaleContract(DimensionalScaleContract.si(), 1, speed_of_light, 1, 1)


def _numpy_boris(
    proper: np.ndarray,
    electric: np.ndarray,
    magnetic: np.ndarray,
    specific_charge: np.ndarray,
    active: np.ndarray,
    step: float,
    light: float,
) -> np.ndarray:
    # Relativistic Boris (Birdsall & Langdon 1991, section 15-4) in proper velocity.
    result = np.zeros_like(proper)
    for index in range(proper.shape[0]):
        if not active[index]:
            continue
        half = 0.5 * step * specific_charge[index]
        u_minus = proper[index] + half * electric[index]
        gamma_minus = np.sqrt(1.0 + u_minus @ u_minus / light**2)
        t = half * magnetic[index] / gamma_minus
        s = 2.0 * t / (1.0 + t @ t)
        u_plus = u_minus + np.cross(u_minus + np.cross(u_minus, t), s)
        result[index] = u_plus + half * electric[index]
    return result


def _trajectory(
    plan: RelativisticPushPlan,
    initial: Array,
    electric: Array,
    magnetic: Array,
    specific_charge: Array,
    step: float,
    count: int,
) -> Array:
    active = jnp.ones((initial.shape[0],), dtype=jnp.bool_)

    def advance(proper: Array, _: None) -> tuple[Array, Array]:
        pushed = plan.push(proper, electric, magnetic, specific_charge, active, step)
        return pushed.proper_velocity, pushed.proper_velocity

    _, history = jax.lax.scan(advance, initial, xs=None, length=count)
    return history


def test_boris_matches_independent_reference_in_declared_speed_of_light() -> None:
    light = 3.0
    generator = np.random.default_rng(20260927)
    proper = generator.normal(scale=2.0, size=(5, 3))
    electric = generator.normal(size=(5, 3))
    magnetic = generator.normal(scale=1.5, size=(5, 3))
    specific_charge = np.asarray([-1.0, 2.5, 0.7, -3.0, 1.0])
    active = np.asarray([True, True, False, True, True])
    step = 0.37
    plan = RelativisticPushPlan(_relativity(3), method="boris")

    result = plan.push(proper, electric, magnetic, specific_charge, active, step)

    expected = _numpy_boris(
        proper, electric, magnetic, specific_charge, active, step, light
    )
    velocity = expected / np.sqrt(1.0 + np.sum(expected**2, axis=-1) / light**2)[:, None]
    np.testing.assert_allclose(result.proper_velocity, expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(result.velocity, velocity, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(
        result.maximum_speed,
        np.max(np.linalg.norm(velocity[active], axis=-1)),
        rtol=1e-14,
    )
    np.testing.assert_array_equal(result.proper_velocity[2], np.zeros((3,)))
    assert bool(result.finite) and bool(result.subluminal) and bool(result.successful)


def test_vay_and_higuera_cary_hold_exact_e_cross_b_drift_where_boris_drifts() -> None:
    # Crossed fields E = v_d B y-hat, B = z-hat with v_d < c: a particle moving at
    # u = gamma_d v_d x-hat feels E + v x B = 0 and its proper velocity is steady.
    # dt resolves the relativistic gyration (Omega dt / gamma = 0.01) but not the
    # rest-mass cyclotron frequency (Omega dt = 100), the regime of Vay (2008).
    gamma = 1.0e4
    drift = np.sqrt(1.0 - 1.0 / gamma**2)
    initial = jnp.asarray([[gamma * drift, 0.0, 0.0]])
    electric = jnp.asarray([[0.0, drift, 0.0]])
    magnetic = jnp.asarray([[0.0, 0.0, 1.0]])
    specific_charge = jnp.asarray([-1.0])
    relativity = PIC_CODE_RELATIVITY

    for method in _DRIFT_EXACT:
        plan = RelativisticPushPlan(relativity, method=method)
        history = _trajectory(
            plan, initial, electric, magnetic, specific_charge, 100.0, 200
        )
        np.testing.assert_allclose(
            history,
            jnp.broadcast_to(initial, history.shape),
            rtol=0.0,
            atol=1.0e-12 * gamma,
            err_msg=method,
        )

    # Boris evaluates the magnetic rotation at gamma(u + qE dt / 2m), so the drift
    # frame acquires a spurious transverse proper velocity of order c / 4.
    boris = RelativisticPushPlan(relativity, method="boris")
    history = _trajectory(boris, initial, electric, magnetic, specific_charge, 100.0, 200)
    assert float(jnp.max(jnp.abs(history[:, 0, 1]))) > 0.1


@pytest.mark.parametrize("method", _METHODS, ids=_METHODS)
def test_pure_magnetic_push_conserves_energy(method: RelativisticPusher) -> None:
    generator = np.random.default_rng(7)
    initial = jnp.asarray(generator.normal(scale=4.0, size=(6, 3)))
    magnetic = jnp.asarray(generator.normal(scale=2.0, size=(6, 3)))
    specific_charge = jnp.asarray([-1.0, 1.0, 3.0, -0.5, 2.0, -4.0])
    plan = RelativisticPushPlan(_relativity(2), method=method)

    history = _trajectory(
        plan,
        initial,
        jnp.zeros_like(initial),
        magnetic,
        specific_charge,
        0.3,
        500,
    )

    np.testing.assert_allclose(
        jnp.sum(history**2, axis=-1),
        jnp.broadcast_to(jnp.sum(initial**2, axis=-1), history.shape[:2]),
        rtol=1e-12,
    )


def test_higuera_cary_map_preserves_phase_space_volume() -> None:
    electric = jnp.asarray([[0.3, -0.2, 0.5]])
    magnetic = jnp.asarray([[0.4, 1.1, -0.7]])
    specific_charge = jnp.asarray([2.0])
    active = jnp.asarray([True])
    points = jnp.asarray([[0.9, -2.0, 1.5], [0.0, 0.1, 0.0], [30.0, -12.0, 4.0]])

    def jacobian_determinants(method: RelativisticPusher) -> Array:
        plan = RelativisticPushPlan(_relativity(2), method=method)

        def advance(proper: Array) -> Array:
            return plan.push(
                proper[None], electric, magnetic, specific_charge, active, 0.7
            ).proper_velocity[0]

        return jax.vmap(lambda point: jnp.linalg.det(jax.jacfwd(advance)(point)))(points)

    np.testing.assert_allclose(jacobian_determinants("higuera-cary"), 1.0, atol=1e-13)
    # Fault adequacy: the Vay map is not volume preserving (Higuera & Cary 2017).
    assert float(jnp.max(jnp.abs(jacobian_determinants("vay") - 1.0))) > 1e-3


@pytest.mark.parametrize("method", _METHODS, ids=_METHODS)
def test_relativistic_gyration_period(method: RelativisticPusher) -> None:
    # c = 3 from the scale contract, q/m = -2, |B| = 1.5, gamma = 5.
    light = 3.0
    gamma = 5.0
    specific = -2.0
    field = 1.5
    period = 2.0 * np.pi * gamma / (abs(specific) * field)
    step = period / 400.0
    count = 4000
    initial = jnp.asarray([[light * np.sqrt(gamma**2 - 1.0), 0.0, 0.0]])
    plan = RelativisticPushPlan(_relativity(3), method=method)

    history = _trajectory(
        plan,
        initial,
        jnp.zeros_like(initial),
        jnp.asarray([[0.0, 0.0, field]]),
        jnp.asarray([specific]),
        step,
        count,
    )

    phase = np.unwrap(
        np.concatenate(
            [
                [0.0],
                np.arctan2(np.asarray(history[:, 0, 1]), np.asarray(history[:, 0, 0])),
            ]
        )
    )
    # Negative charge gyrates counterclockwise about +B.
    assert phase[-1] > 0.0
    np.testing.assert_allclose(2.0 * np.pi * count * step / phase[-1], period, rtol=1e-4)


def test_push_plan_refuses_undeclared_scale_and_method() -> None:
    with pytest.raises(TypeError, match="RelativityScaleContract"):
        RelativisticPushPlan(1.0, method="boris")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="method"):
        RelativisticPushPlan(
            PIC_CODE_RELATIVITY,
            method="leapfrog",  # ty: ignore[invalid-argument-type]
        )
