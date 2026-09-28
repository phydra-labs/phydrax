#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.population_balance as pb


DIAMETERS = np.geomspace(1.0e-3, 8.0e-3, 5)
FIRST_MOMENT_TOLERANCE = 256.0 * np.finfo(np.float64).eps


def _liquid(dissipation_rate: float = 1.0) -> pb.TurbulentBubblyLiquid:
    return pb.TurbulentBubblyLiquid(998.0, 0.072, 1.0e-6, dissipation_rate)


def _plan() -> pb.BubbleSectionalPlan:
    return pb.BubbleSectionalPlan(DIAMETERS, breakage_order=12)


def _reference_eddy_integral(scale: float, minimum_ratio: float) -> float:
    """Composite Gauss–Legendre on geometric panels of [ξmin, 1]."""
    nodes, weights = np.polynomial.legendre.leggauss(30)
    edges = np.geomspace(minimum_ratio, 1.0, 401)
    left, right = edges[:-1, None], edges[1:, None]
    points = 0.5 * (right - left) * nodes + 0.5 * (right + left)
    integrand = (
        (1.0 + points) ** 2
        * points ** (-11.0 / 3.0)
        * np.exp(-scale * points ** (-11.0 / 3.0))
    )
    return float(np.sum(0.5 * (right - left) * weights * integrand))


def _advance(
    plan: pb.BubbleSectionalPlan,
    kernel: np.ndarray,
    frequency: np.ndarray | None,
    daughters: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    solver = pb.ConservativeSectionalSolver.create(
        np.asarray(plan.volumes),
        kernel,
        breakage_frequency_s_inv=frequency,
        daughter_number=daughters,
    )
    state = pb.SectionalPopulationState.create(np.full(DIAMETERS.size, 2.0e7))
    orders = jnp.asarray((0, 1))
    initial = np.asarray(solver.moments(state, orders))
    for _ in range(2):
        step = solver.advance(state, 2.0e-3)
        assert bool(step.successful)
        state = step.state
    return initial, np.asarray(solver.moments(state, orders))


@pytest.mark.parametrize("minimum_ratio", (1.0e-3, 0.03, 0.2, 0.6, 0.95))
def test_luo_svendsen_closed_form_matches_quadrature(minimum_ratio: float) -> None:
    scales = np.array((1.0e-6, 1.0e-2, 0.4, 1.0, 3.0, 20.0, 150.0))
    closed = np.asarray(pb.luo_svendsen_eddy_integral(scales, minimum_ratio))
    reference = np.array([_reference_eddy_integral(b, minimum_ratio) for b in scales])
    np.testing.assert_allclose(closed, reference, rtol=1.0e-10, atol=0.0)


def test_luo_svendsen_closed_form_derivative_matches_finite_difference() -> None:
    scale, ratio, step = 0.8, 0.05, 1.0e-5
    derivative = float(jax.grad(pb.luo_svendsen_eddy_integral)(scale, ratio))
    reference = (
        _reference_eddy_integral(scale + step, ratio)
        - _reference_eddy_integral(scale - step, ratio)
    ) / (2.0 * step)
    assert derivative == pytest.approx(reference, rel=1.0e-7)


def test_coalescence_kernels_are_symmetric_nonnegative_and_supported() -> None:
    plan = _plan()
    liquid = pb.TurbulentBubblyLiquid(
        998.0, 0.072, 1.0e-6, 1.0, integral_length_scale=0.2
    )
    rise = np.linspace(0.2, 0.25, DIAMETERS.size)
    for result in (
        pb.PrinceBlanchCoalescenceKernel(liquid, shear_rate=4.0).evaluate(plan),
        pb.LehrCoalescenceKernel(liquid, 0.15).evaluate(plan, rise_velocities=rise),
    ):
        kernel = np.asarray(result.kernel)
        np.testing.assert_array_equal(kernel, kernel.T)
        assert np.all(kernel > 0.0)
        assert np.all(np.asarray(result.efficiency) <= 1.0)
        assert float(result.symmetry_residual) == 0.0
        assert bool(result.evidence.successful)
        assert int(result.evidence.status) == pb.BubbleKernelStatus.SUCCESS


def test_prince_blanch_efficiency_decreases_with_dissipation() -> None:
    plan = _plan()
    efficiencies = [
        np.asarray(
            pb.PrinceBlanchCoalescenceKernel(_liquid(eps)).evaluate(plan).efficiency
        )
        for eps in (0.1, 1.0, 10.0)
    ]
    assert np.all(efficiencies[0] > efficiencies[1])
    assert np.all(efficiencies[1] > efficiencies[2])


def test_prince_blanch_laminar_shear_adds_orthokinetic_collisions() -> None:
    plan, shear = _plan(), 3.0
    liquid = _liquid()
    plain = pb.PrinceBlanchCoalescenceKernel(liquid).evaluate(plan)
    sheared = pb.PrinceBlanchCoalescenceKernel(liquid, shear_rate=shear).evaluate(plan)
    radii = 0.5 * DIAMETERS
    orthokinetic = 4.0 / 3.0 * (radii[:, None] + radii[None, :]) ** 3 * shear
    np.testing.assert_allclose(
        np.asarray(sheared.kernel) - np.asarray(plain.kernel),
        orthokinetic * np.asarray(plain.efficiency),
        rtol=1.0e-12,
    )


def test_lehr_rate_saturates_at_critical_approach_velocity() -> None:
    plan, critical = _plan(), 0.05
    results = [
        pb.LehrCoalescenceKernel(_liquid(eps), 0.2, critical_velocity=critical).evaluate(
            plan
        )
        for eps in (1.0, 10.0)
    ]
    for result in results:
        velocity = critical * np.asarray(result.collision_frequency / result.kernel)
        assert np.all(velocity > critical)
        np.testing.assert_allclose(
            np.asarray(result.efficiency), critical / velocity, rtol=1.0e-12
        )
    np.testing.assert_allclose(
        np.asarray(results[0].kernel), np.asarray(results[1].kernel), rtol=1.0e-12
    )


def test_coalescence_conserves_gas_volume_and_reduces_number() -> None:
    plan = _plan()
    for result in (
        pb.PrinceBlanchCoalescenceKernel(_liquid()).evaluate(plan),
        pb.LehrCoalescenceKernel(_liquid(), 0.15).evaluate(plan),
    ):
        initial, final = _advance(plan, np.asarray(result.kernel), None, None)
        assert final[0] < initial[0]
        assert abs(final[1] - initial[1]) <= FIRST_MOMENT_TOLERANCE * initial[1]


def test_luo_svendsen_breakage_conserves_gas_volume_and_increases_number() -> None:
    plan = _plan()
    result = pb.LuoSvendsenBreakageKernel(_liquid(), 0.1).evaluate(plan)
    frequency = np.asarray(result.frequency)
    daughters = np.asarray(result.daughter_number)
    volumes = np.asarray(plan.volumes)
    assert bool(result.evidence.successful)
    assert np.all(frequency > 0.0) and np.all(daughters >= 0.0)
    assert float(result.daughter_volume_residual) <= FIRST_MOMENT_TOLERANCE
    np.testing.assert_allclose(volumes @ daughters, volumes, rtol=FIRST_MOMENT_TOLERANCE)
    initial, final = _advance(plan, np.zeros((DIAMETERS.size,) * 2), frequency, daughters)
    assert final[0] > initial[0]
    assert abs(final[1] - initial[1]) <= FIRST_MOMENT_TOLERANCE * initial[1]


def test_evidence_refuses_sizes_outside_inertial_subrange_and_weber_support() -> None:
    plan = _plan()
    result = pb.PrinceBlanchCoalescenceKernel(_liquid(1.0e-7)).evaluate(plan)
    assert float(result.evidence.kolmogorov_length) > DIAMETERS[0]
    assert not bool(result.evidence.inertial_subrange)
    assert int(result.evidence.status) == pb.BubbleKernelStatus.OUTSIDE_SUPPORT
    assert not bool(result.evidence.successful)

    liquid = _liquid()
    weber = np.asarray(liquid.weber_numbers(jnp.asarray(DIAMETERS)))
    narrow = (float(weber[1]), 2.0 * float(weber[-1]))
    breakage = pb.LuoSvendsenBreakageKernel(liquid, 0.1, weber_support=narrow).evaluate(
        plan
    )
    assert float(breakage.evidence.minimum_weber) == pytest.approx(weber[0], rel=1.0e-14)
    assert not bool(breakage.evidence.weber_within_support)
    assert int(breakage.evidence.status) == pb.BubbleKernelStatus.OUTSIDE_SUPPORT
    bounded = pb.TurbulentBubblyLiquid(
        998.0, 0.072, 1.0e-6, 1.0, integral_length_scale=4.0e-3
    )
    lehr = pb.LehrCoalescenceKernel(bounded, 0.1).evaluate(plan)
    assert not bool(lehr.evidence.inertial_subrange)
    assert not bool(lehr.evidence.successful)


def test_evidence_reports_exponents_beyond_float_range_without_clipping() -> None:
    plan = _plan()
    liquid = _liquid(1.0e3)
    drainage = pb.PrinceBlanchCoalescenceKernel(
        liquid, initial_film_thickness=1.0, critical_film_thickness=1.0e-300
    ).evaluate(plan)
    dilute = pb.LehrCoalescenceKernel(liquid, 1.0e-9).evaluate(plan)
    for result in (drainage, dilute):
        evidence = result.evidence
        assert float(evidence.maximum_exponent_argument) > 745.0
        assert bool(evidence.exponent_out_of_range) and bool(evidence.finite)
        assert int(evidence.status) == pb.BubbleKernelStatus.EXPONENT_OUT_OF_RANGE
        assert bool(evidence.successful)
        assert np.any(np.asarray(result.kernel) == 0.0)

    # ε = 1e-5: η ≈ 0.56 mm and λ_min ≈ 6.4 mm, so these bubbles break, but the
    # surface-energy exponent near f = 1/2 exceeds 745 and those rates are zero.
    large = pb.BubbleSectionalPlan(np.geomspace(6.6e-3, 1.2e-2, 4), breakage_order=8)
    breakage = pb.LuoSvendsenBreakageKernel(_liquid(1.0e-5), 0.1).evaluate(large)
    assert float(breakage.evidence.maximum_exponent_argument) > 745.0
    assert bool(breakage.evidence.exponent_out_of_range)
    assert bool(breakage.evidence.successful)
    assert np.any(np.asarray(breakage.partial_rates) == 0.0)
    assert np.all(np.asarray(breakage.frequency) > 0.0)


def test_invalid_declarations_are_refused() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        pb.BubbleSectionalPlan(np.array((1.0e-3, 1.0e-3, 2.0e-3)))
    with pytest.raises(ValueError, match="surface_tension"):
        pb.TurbulentBubblyLiquid(998.0, -0.072, 1.0e-6, 1.0)
    with pytest.raises(ValueError, match="gas_volume_fraction"):
        pb.LehrCoalescenceKernel(_liquid(), 0.55, maximum_packing_fraction=0.5)
    with pytest.raises(ValueError, match="critical_film_thickness"):
        pb.PrinceBlanchCoalescenceKernel(
            _liquid(), initial_film_thickness=1.0e-8, critical_film_thickness=1.0e-4
        )
    with pytest.raises(ValueError, match="rise_velocities"):
        pb.LehrCoalescenceKernel(_liquid(), 0.1).evaluate(
            _plan(), rise_velocities=np.ones(2)
        )
