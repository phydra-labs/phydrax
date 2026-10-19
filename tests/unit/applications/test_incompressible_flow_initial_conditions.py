from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0

from phydrax.applications.incompressible_flow import (
    embed_planar_velocity,
    PeriodicFlowFamily,
    PeriodicFlowInitialMeasure,
    PiecewiseConstantVorticityFamily,
    PowerLawVelocityFamily,
    restrict_periodic_flow_initial_state,
    ShearLayerFamily,
    SinusoidalVortexSheetFamily,
    TaylorGreenFamily,
    VortexPatchFamily,
)
from phydrax.applications.incompressible_flow._initial_conditions import (
    _mollifier_hankel,
    _shear_profile,
)
from phydrax.discretization import (
    AxisDomain,
    FourierBasisPlan,
    PaddingDealiasingPlan,
    PseudospectralMethodPlan,
    TensorSpectralPlan,
)
from phydrax.discretization.spectral import PeriodicLerayProjector
from phydrax.discretization.spectral._space import TensorSpectralDiscretization
from phydrax.equations import (
    compile_periodic_incompressible_flow,
    IncompressibleFlowProblem,
)
from phydrax.execution import MemoryBudgetError
from phydrax.stochastic import PeriodicPowerLawFieldPlan, PowerLawSpectralAmplitude


def _space(
    count: int, dimension: int = 2, length: float = 1.0
) -> TensorSpectralDiscretization:
    return TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(dimension))
    ).prepare(tuple(AxisDomain.periodic(0.3, 0.3 + length) for _ in range(dimension)))


def _families() -> list[PeriodicFlowFamily]:
    return [
        TaylorGreenFamily(2),
        ShearLayerFamily(
            profile="smooth",
            thickness_range=(0.02, 0.04),
            perturbation_amplitude=0.01,
            perturbation_mode_range=(1, 3),
            interface_shift_range=(-0.02, 0.02),
            oversampling=2,
        ),
        ShearLayerFamily(
            profile="discontinuous",
            thickness_range=None,
            perturbation_amplitude=0.01,
            perturbation_mode_range=(1, 3),
            interface_shift_range=(-0.02, 0.02),
            oversampling=2,
        ),
        VortexPatchFamily(radius=0.15, center=(0.7, 0.8), strength=2.0),
        PiecewiseConstantVorticityFamily(partition=(3, 4), value_range=(-1.0, 1.0)),
        SinusoidalVortexSheetFamily(
            mollifier_radius=0.08,
            base_amplitude=0.2,
            perturbation_count=3,
            perturbation_amplitude_range=(0.0, 0.003125),
            quadrature_points=48,
        ),
        PowerLawVelocityFamily(
            PeriodicPowerLawFieldPlan(
                PowerLawSpectralAmplitude(1.5), component_shape=(2,), solenoidal=True
            )
        ),
    ]


@pytest.mark.parametrize("family", _families())
def test_family_reality_divergence_replay_and_nested_gather(
    family: PeriodicFlowFamily,
) -> None:
    fine, middle, coarse = _space(12), _space(9), _space(6)
    measure = PeriodicFlowInitialMeasure(family)
    first = measure.prepare(fine, member_chunk_size=1)
    second = measure.prepare(fine, member_chunk_size=3)
    key = jax.random.key(415)
    state = first.realize(key, np.array([8, 2, 4], dtype=np.int64))
    other = second.realize(key, np.array([4, 8], dtype=np.int64))
    np.testing.assert_allclose(
        other.velocity, np.asarray(state.velocity)[[2, 0]], atol=2e-15
    )
    assert state.measure_id == other.measure_id
    assert state.coupling_id == other.coupling_id
    assert first.prepared_id != second.prepared_id
    for a, b in zip(
        jax.tree.leaves(other.parameters), jax.tree.leaves(state.parameters), strict=True
    ):
        np.testing.assert_array_equal(a, np.asarray(b)[[2, 0]])
    for velocity in state.velocity:
        assert fine.imaginary_leakage(velocity) < 3e-14
        assert PeriodicLerayProjector(fine).divergence_norm(velocity) < 2e-13
    direct = restrict_periodic_flow_initial_state(state, coarse)
    staged = restrict_periodic_flow_initial_state(
        restrict_periodic_flow_initial_state(state, middle), coarse
    )
    np.testing.assert_array_equal(direct.velocity, staged.velocity)
    np.testing.assert_array_equal(
        direct.removed_coefficient_energy, staged.removed_coefficient_energy
    )
    for evidence in (
        state.removed_coefficient_energy,
        state.projection_removed_energy,
        state.removed_mean_vorticity,
        state.quadrature_residual,
    ):
        assert evidence.shape == (3,)
        assert np.all(np.isfinite(evidence))


@pytest.mark.parametrize("dimension", [2, 3])
def test_taylor_green_exact_modal_synthesis(dimension: int) -> None:
    space = _space(7, dimension, length=2.0)
    state = (
        PeriodicFlowInitialMeasure(TaylorGreenFamily(dimension, amplitude=1.7))
        .prepare(space)
        .realize(jax.random.key(2), np.array([0], dtype=np.uint32))
    )
    angle = jnp.meshgrid(
        *(2 * jnp.pi * (a.nodes - 0.3) / a.length for a in space.axes),
        indexing="ij",
    )
    z = jnp.cos(angle[2]) if dimension == 3 else 1.0
    expected = jnp.stack(
        (
            1.7 * jnp.sin(angle[0]) * jnp.cos(angle[1]) * z,
            -1.7 * jnp.cos(angle[0]) * jnp.sin(angle[1]) * z,
        )
        + (() if dimension == 2 else (jnp.zeros_like(angle[0]),)),
        axis=-1,
    )
    np.testing.assert_allclose(space.reconstruct(state.velocity[0]), expected, atol=3e-15)


def _vorticity(space: TensorSpectralDiscretization, velocity: jax.Array) -> np.ndarray:
    return np.asarray(
        space.modal_derivative(velocity[..., 1], axis=0)
        - space.modal_derivative(velocity[..., 0], axis=1)
    )


def test_vortex_patch_against_independent_disk_quadrature() -> None:
    space = _space(9, length=2.0)
    radius, strength = 0.24, 1.8
    center = (0.9, 1.1)
    state = (
        PeriodicFlowInitialMeasure(
            VortexPatchFamily(radius=radius, center=center, strength=strength)
        )
        .prepare(space)
        .realize(jax.random.key(3), np.array([0], dtype=np.uint32))
    )
    omega = _vorticity(space, state.velocity[0])
    # Polar trapezoid + Gauss--Legendre radial quadrature, independent of J1.
    roots, weights = np.polynomial.legendre.leggauss(96)
    r = (roots + 1) * radius / 2
    angle = np.arange(384, dtype=np.float64) * (2 * np.pi / 384)
    x = center[0] - 0.3 + r[:, None] * np.cos(angle)
    y = center[1] - 0.3 + r[:, None] * np.sin(angle)
    for i, j in [(1, 0), (0, 2), (2, 3), (8, 1)]:
        modes = [np.asarray(a.modes.mode_numbers) for a in space.axes]
        phase = np.exp(-1j * np.pi * (modes[0][i] * x + modes[1][j] * y))
        expected = (
            strength
            / 2
            * np.sum(phase * (weights * r * radius / 2)[:, None])
            * (2 * np.pi / 384)
        )
        np.testing.assert_allclose(omega[i, j], expected, atol=2e-15)
    np.testing.assert_allclose(
        state.removed_mean_vorticity, strength * np.pi * radius**2 / 4, atol=1e-16
    )
    assert omega[0, 0] == 0


def test_piecewise_boxes_against_independent_cell_quadrature() -> None:
    space = _space(9, length=2.0)
    state = (
        PeriodicFlowInitialMeasure(
            PiecewiseConstantVorticityFamily(partition=(3, 4), value_range=(-2.0, 3.0))
        )
        .prepare(space)
        .realize(jax.random.key(4), np.array([0], dtype=np.uint32))
    )
    values = np.asarray(state.parameters["cell_values"])[0]
    roots, weights = np.polynomial.legendre.leggauss(16)
    x = (np.arange(3, dtype=np.float64)[:, None] + (roots + 1) / 2) / 3
    y = (np.arange(4, dtype=np.float64)[:, None] + (roots + 1) / 2) / 4
    omega = _vorticity(space, state.velocity[0])
    modes = [np.asarray(a.modes.mode_numbers) for a in space.axes]
    for i, j in [(1, 0), (0, 2), (2, 3), (8, 1)]:
        phase = np.exp(
            -2j
            * np.pi
            * (modes[0][i] * x[:, None, :, None] + modes[1][j] * y[None, :, None, :])
        )
        integrand = (
            values[:, :, None, None]
            * phase
            * weights[None, None, :, None]
            * weights[None, None, None, :]
        )
        expected = 2 * np.sum(integrand) / (4 * values.size)
        # Integrate within each constant cell, never across its discontinuities.
        # The 16-point Gauss remainder is below 7e-29 for these modes/values.
        # gamma_(term count + 32) bounds reduction and per-term float64 work.
        operations = integrand.size + 32
        epsilon = np.finfo(np.float64).eps
        gamma = operations * epsilon / (1 - operations * epsilon)
        allowance = 2 * gamma * np.sum(np.abs(integrand)) / (4 * values.size) + 7e-29
        np.testing.assert_allclose(omega[i, j], expected, rtol=0, atol=allowance)
    np.testing.assert_allclose(state.removed_mean_vorticity, np.mean(values), atol=1e-15)


def test_mollifier_normalization_and_sheet_quadrature_refinement() -> None:
    # Direct published truncated powers, independent of runtime polynomial.
    def psi(r: float) -> float:
        return (
            80
            / (7 * np.pi)
            * (
                max(r + 1, 0) ** 3
                - 4 * max(r + 0.5, 0) ** 3
                + 6 * max(r, 0) ** 3
                - 4 * max(r - 0.5, 0) ** 3
                + max(r - 1, 0) ** 3
            )
        )

    integral = quad(lambda r: 2 * np.pi * r * psi(r), 0, 1, points=[0.5])[0]
    np.testing.assert_allclose(integral, 1.0, atol=1e-14)
    residuals = []
    states = []
    for q in (12, 24, 48):
        family = SinusoidalVortexSheetFamily(
            mollifier_radius=0.08,
            base_amplitude=0.2,
            perturbation_count=3,
            perturbation_amplitude_range=(0.0, 0.003125),
            quadrature_points=q,
        )
        prepared = PeriodicFlowInitialMeasure(family).prepare(
            _space(12), member_chunk_size=1
        )
        np.testing.assert_allclose(prepared.mollifier_normalization, 1.0, atol=1e-14)
        state = prepared.realize(jax.random.key(11), np.array([0], dtype=np.uint32))
        residuals.append(float(state.quadrature_residual[0]))
        states.append(state)
    assert residuals[2] < residuals[1] < residuals[0]
    np.testing.assert_array_equal(
        states[0].parameters["alpha"], states[2].parameters["alpha"]
    )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_planar_embedding_energy_and_dynamics_identity(axis: int) -> None:
    planar = _space(7)
    state = (
        PeriodicFlowInitialMeasure(TaylorGreenFamily(2))
        .prepare(planar)
        .realize(jax.random.key(1), np.array([0], dtype=np.uint32))
    )
    domains = [a.domain for a in planar.axes]
    domains.insert(axis, AxisDomain.periodic(-0.2, 1.8))
    target = TensorSpectralPlan(tuple(FourierBasisPlan(7) for _ in range(3))).prepare(
        tuple(domains)
    )
    embedded = embed_planar_velocity(state, target, invariant_axis=axis)
    np.testing.assert_allclose(
        np.sum(np.abs(embedded.velocity) ** 2),
        2 * np.sum(np.abs(state.velocity) ** 2),
        atol=1e-15,
    )
    assert np.all(np.asarray(embedded.velocity)[..., axis] == 0)
    method = PseudospectralMethodPlan(dealiasing=PaddingDealiasingPlan(2))
    two = compile_periodic_incompressible_flow(
        IncompressibleFlowProblem(2, 0.03), planar, method
    )
    three = compile_periodic_incompressible_flow(
        IncompressibleFlowProblem(3, 0.03), target, method
    )
    rate2 = two(jnp.asarray(0.0), state.velocity[0], None)
    rate3 = three(jnp.asarray(0.0), embedded.velocity[0], None)
    for component, target_component in enumerate(i for i in range(3) if i != axis):
        selected = np.take(np.asarray(rate3)[..., target_component], 0, axis=axis)
        np.testing.assert_allclose(
            selected, np.sqrt(2) * np.asarray(rate2)[..., component], atol=2e-14
        )


def test_refusal_matrix() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        ShearLayerFamily(
            profile="smooth",
            thickness_range=(0.01, 0.02),
            perturbation_amplitude=-0.03,
            perturbation_mode_range=(1, 12),
            interface_shift_range=(0.0, 0.0),
            oversampling=2,
        )
    with pytest.raises(ValueError, match="thickness"):
        ShearLayerFamily(
            profile="discontinuous",
            thickness_range=(0.01, 0.02),
            perturbation_amplitude=0.01,
            perturbation_mode_range=(1, 2),
            interface_shift_range=(0.0, 0.0),
            oversampling=2,
        )
    with pytest.raises(ValueError, match="radius"):
        PeriodicFlowInitialMeasure(
            VortexPatchFamily(radius=0.5, center=(0.5, 0.5), strength=1.0)
        ).prepare(_space(8))
    with pytest.raises(ValueError, match="solenoidal"):
        PowerLawVelocityFamily(
            PeriodicPowerLawFieldPlan(
                PowerLawSpectralAmplitude(1.0), component_shape=(2,)
            )
        )
    measure = PeriodicFlowInitialMeasure(TaylorGreenFamily(2))
    with pytest.raises(MemoryBudgetError) as error:
        measure.prepare(_space(8), maximum_bytes=1)
    assert error.value.budget == "maximum_bytes"
    assert error.value.required_bytes > error.value.limit_bytes == 1
    prepared = measure.prepare(_space(8), member_chunk_size=1)
    for indices in ([], [-1], [2**32]):
        with pytest.raises(ValueError):
            prepared.realize(jax.random.key(1), np.array(indices, dtype=np.int64))
    with pytest.raises(TypeError):
        prepared.realize(jax.random.key(1), np.array([1.2], dtype=np.float64))
    state = prepared.realize(jax.random.key(1), np.array([0], dtype=np.uint32))
    with pytest.raises(ValueError, match="refinement"):
        restrict_periodic_flow_initial_state(state, _space(9))
    with pytest.raises(ValueError):
        restrict_periodic_flow_initial_state(state, _space(6, length=2.0))
    with pytest.raises(ValueError):
        embed_planar_velocity(state, _space(8, 3, length=2.0), invariant_axis=2)


def test_parameter_workspaces_refused_before_drawing() -> None:
    shear = ShearLayerFamily(
        profile="discontinuous",
        thickness_range=None,
        perturbation_amplitude=0.0,
        perturbation_mode_range=(1, 10**8),
        interface_shift_range=(0.0, 0.0),
        oversampling=2,
    )
    sheet = SinusoidalVortexSheetFamily(
        mollifier_radius=0.1,
        base_amplitude=0.2,
        perturbation_count=10**8,
        perturbation_amplitude_range=(0.0, 0.0),
        quadrature_points=8,
    )
    space = _space(7)
    for family in (shear, sheet):
        with pytest.raises(MemoryBudgetError, match="maximum_bytes"):
            PeriodicFlowInitialMeasure(family).prepare(
                space, member_chunk_size=1, maximum_bytes=2**20
            )


def test_shear_half_open_perturbation_amplitudes() -> None:
    # Test the displacement envelope with Poseidon Eq. (45)'s k-weighted law.
    family = ShearLayerFamily(
        profile="smooth",
        thickness_range=(0.1, 0.1),
        perturbation_amplitude=0.025,
        perturbation_mode_range=(10, 10),
        interface_shift_range=(0.0, 0.0),
        oversampling=2,
    )
    prepared = PeriodicFlowInitialMeasure(family).prepare(_space(7), member_chunk_size=2)
    state = prepared.realize(jax.random.key(827), np.arange(8, dtype=np.uint32))
    alpha = np.asarray(state.parameters["alpha"])
    assert np.all((alpha >= 0.0) & (alpha < 1.0))
    assert np.all(family.perturbation_amplitude * np.sum(alpha, axis=1) < 0.25)
    np.testing.assert_array_equal(state.parameters["mode_count"], 10)


@pytest.mark.parametrize("shift", [0.25, 0.65, 1.2, -2.7])
def test_shear_arbitrary_finite_displacement_admitted(shift: float) -> None:
    ShearLayerFamily(
        profile="discontinuous",
        thickness_range=None,
        perturbation_amplitude=0.3,
        perturbation_mode_range=(10, 10),
        interface_shift_range=(shift, shift),
        oversampling=2,
    )


def test_smooth_shear_periodic_seam_and_unwrapped_fault() -> None:
    epsilon = 1e-12
    seams = np.array([epsilon, 1 - epsilon], dtype=np.float64)
    shifts = np.array([0.2, 0.65, 1.2], dtype=np.float64)
    displaced = seams[None, :] + shifts[:, None]
    for rho in (0.1, 0.3):
        actual = np.asarray(
            _shear_profile(jnp.asarray(displaced), jnp.asarray(rho), "smooth")
        )
        wrapped = np.mod(displaced, 1.0)
        expected = np.where(
            wrapped <= 0.5,
            np.tanh((wrapped - 0.25) / rho),
            np.tanh((0.75 - wrapped) / rho),
        )
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
        np.testing.assert_allclose(actual[:, 0], actual[:, 1], atol=3e-11, rtol=0)
        unwrapped = np.where(
            displaced <= 0.5,
            np.tanh((displaced - 0.25) / rho),
            np.tanh((0.75 - displaced) / rho),
        )
        assert np.max(np.abs(unwrapped[:, 0] - unwrapped[:, 1])) > 0.1


def _published_mollifier(r: float) -> float:
    # Literal truncated powers from Poseidon Eq. (49), independent of preparation.
    return (
        80
        / (7 * np.pi)
        * (
            max(r + 1, 0) ** 3
            - 4 * max(r + 0.5, 0) ** 3
            + 6 * max(r, 0) ** 3
            - 4 * max(r - 0.5, 0) ** 3
            + max(r - 1, 0) ** 3
        )
    )


def _complex_quad(integrand: Callable[[float], complex]) -> complex:
    real = quad(lambda x: float(integrand(x).real), 0.0, 1.0, epsabs=1e-12)[0]
    imag = quad(lambda x: float(integrand(x).imag), 0.0, 1.0, epsabs=1e-12)[0]
    return complex(real, imag)


def test_vectorized_hankel_matches_independent_adaptive_quadrature() -> None:
    wave = np.array([0.0, 1.0, 7.0, 43.0, 129.0, 317.0, 600.0], dtype=np.float64)
    radius = 0.2
    weights, normalization = _mollifier_hankel(wave, radius, 2**20)
    expected = np.array(
        [
            quad(
                lambda r: 2 * np.pi * r * _published_mollifier(r) * j0(k * radius * r),
                0.0,
                1.0,
                points=[0.5],
                epsabs=1e-13,
            )[0]
            for k in wave
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(weights, expected, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(normalization, 1.0, atol=2e-14)
    with pytest.raises(MemoryBudgetError, match="maximum_bytes"):
        _mollifier_hankel(wave, radius, 1)


def test_vortex_sheet_against_continuum_quadrature_and_arclength_fault() -> None:
    length = 2.0
    space = _space(9, length=length)
    family = SinusoidalVortexSheetFamily(
        mollifier_radius=0.09,
        base_amplitude=0.2,
        perturbation_count=3,
        perturbation_amplitude_range=(0.0, 0.015),
        quadrature_points=256,
    )
    state = (
        PeriodicFlowInitialMeasure(family)
        .prepare(space, member_chunk_size=1)
        .realize(jax.random.key(610), np.array([7], dtype=np.uint32))
    )
    alpha = np.asarray(state.parameters["alpha"])[0]
    beta = np.asarray(state.parameters["beta"])[0]
    omega = _vorticity(space, state.velocity[0])
    volume = length**2

    def graph(t: float) -> tuple[float, float]:
        angles = 2 * np.pi * (t + beta)
        g = (
            0.5
            + family.base_amplitude * np.sin(2 * np.pi * t)
            + np.sum(alpha * np.sin(angles))
        )
        derivative = (
            2
            * np.pi
            * (
                family.base_amplitude * np.cos(2 * np.pi * t)
                + np.sum(alpha * np.cos(angles))
            )
        )
        return float(g), float(derivative)

    modes = np.asarray(space.axes[0].modes.mode_numbers)
    faults = []
    for nx, ny in ((2, 0), (0, 1), (1, 2), (-2, 1)):
        magnitude = 2 * np.pi * np.hypot(nx, ny) / length
        kernel = quad(
            lambda r: (
                2
                * np.pi
                * r
                * _published_mollifier(r)
                * j0(magnitude * family.mollifier_radius * r)
            ),
            0.0,
            1.0,
            points=[0.5],
            epsabs=1e-13,
        )[0]

        def curve_integrand(t: float, *, arclength: bool) -> complex:
            g, derivative = graph(t)
            density = length * (np.sqrt(1 + derivative**2) if arclength else 1.0)
            return complex(density * np.exp(-2j * np.pi * (nx * t + ny * g)))

        expected = (
            kernel
            * _complex_quad(lambda t: curve_integrand(t, arclength=True))
            / np.sqrt(volume)
        )
        missing_arclength = (
            kernel
            * _complex_quad(lambda t: curve_integrand(t, arclength=False))
            / np.sqrt(volume)
        )
        i = int(np.flatnonzero(modes == nx)[0])
        j = int(np.flatnonzero(modes == ny)[0])
        np.testing.assert_allclose(omega[i, j], expected, atol=2e-12, rtol=2e-12)
        faults.append(abs(expected - missing_arclength))
    assert max(faults) > 1e-3
    expected_mean = quad(
        lambda t: length * np.sqrt(1 + graph(t)[1] ** 2) / volume, 0.0, 1.0, epsabs=1e-12
    )[0]
    np.testing.assert_allclose(state.removed_mean_vorticity, expected_mean, atol=2e-12)
    assert omega[0, 0] == 0


def test_displaced_shear_against_numpy_fft_leray_and_sign_fault() -> None:
    count, oversampling, length = 9, 8, 1.7
    space = _space(count, length=length)
    family = ShearLayerFamily(
        profile="smooth",
        thickness_range=(0.07, 0.07),
        perturbation_amplitude=0.035,
        perturbation_mode_range=(3, 3),
        interface_shift_range=(0.015, 0.015),
        oversampling=oversampling,
    )
    state = (
        PeriodicFlowInitialMeasure(family)
        .prepare(space, member_chunk_size=1)
        .realize(jax.random.key(742), np.array([5], dtype=np.uint32))
    )
    alpha = np.asarray(state.parameters["alpha"])[0]
    beta = np.asarray(state.parameters["beta"])[0]
    rho = float(state.parameters["thickness"][0])
    xi = float(state.parameters["interface_shift"][0])
    q = int(state.parameters["mode_count"][0])
    fine_count = count * oversampling
    t = np.arange(fine_count, dtype=np.float64) / fine_count
    sigma = xi + family.perturbation_amplitude * np.sum(
        alpha[:q] * np.sin(2 * np.pi * t[:, None] * np.arange(1, q + 1) - beta[:q]),
        axis=1,
    )
    fine_modes = np.rint(np.fft.fftfreq(fine_count) * fine_count).astype(np.int64)
    coarse_modes = np.asarray(space.axes[0].modes.mode_numbers)
    indices = np.array(
        [np.flatnonzero(fine_modes == n)[0] for n in coarse_modes], dtype=np.int64
    )
    kx, ky = np.meshgrid(
        2 * np.pi * coarse_modes / length,
        2 * np.pi * coarse_modes / length,
        indexing="ij",
    )
    squared = kx**2 + ky**2
    inverse = np.divide(1.0, squared, out=np.zeros_like(squared), where=squared > 0)

    def oracle(sign: float) -> tuple[np.ndarray, float, float]:
        y = np.mod(t[None, :] + sign * sigma[:, None], 1.0)
        horizontal = np.where(
            y <= 0.5, np.tanh((y - 0.25) / rho), np.tanh((0.75 - y) / rho)
        )
        fine = length * np.fft.fft2(horizontal) / fine_count**2
        retained = np.zeros((count, count, 2), dtype=np.complex128)
        retained[..., 0] = fine[np.ix_(indices, indices)]
        longitudinal = kx * retained[..., 0] + ky * retained[..., 1]
        velocity = (
            retained
            - np.stack((kx, ky), axis=-1) * inverse[..., None] * longitudinal[..., None]
        )
        tail = float(np.sum(np.abs(fine) ** 2) - np.sum(np.abs(retained) ** 2))
        projection = float(np.sum(np.abs(retained - velocity) ** 2))
        return velocity, tail, projection

    expected, tail, projection = oracle(1.0)
    fault, _, _ = oracle(-1.0)
    np.testing.assert_allclose(state.velocity[0], expected, atol=3e-14, rtol=3e-13)
    np.testing.assert_allclose(state.removed_coefficient_energy, tail, atol=3e-14)
    np.testing.assert_allclose(state.projection_removed_energy, projection, atol=3e-14)
    assert np.max(np.abs(expected - fault)) > 1e-3


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_planar_embedding_coupled_restriction_identity(axis: int) -> None:
    fine, coarse = _space(9), _space(6)
    prepared = PeriodicFlowInitialMeasure(
        PiecewiseConstantVorticityFamily(partition=(3, 4), value_range=(-1.0, 1.0))
    ).prepare(fine, member_chunk_size=1)
    state = prepared.realize(jax.random.key(941), np.array([3, 7], dtype=np.uint32))
    coarse_state = restrict_periodic_flow_initial_state(state, coarse)

    def target(
        space: TensorSpectralDiscretization, count: int
    ) -> TensorSpectralDiscretization:
        domains = [a.domain for a in space.axes]
        domains.insert(axis, AxisDomain.periodic(-0.2, 1.8))
        return TensorSpectralPlan(
            tuple(FourierBasisPlan(count) for _ in range(3))
        ).prepare(tuple(domains))

    fine3, coarse3 = target(fine, 9), target(coarse, 6)
    embedded_fine = embed_planar_velocity(state, fine3, invariant_axis=axis)
    embedded_coarse = embed_planar_velocity(coarse_state, coarse3, invariant_axis=axis)
    restricted = restrict_periodic_flow_initial_state(embedded_fine, coarse3)
    assert embedded_fine.measure_id == embedded_coarse.measure_id == restricted.measure_id
    assert (
        embedded_fine.coupling_id == embedded_coarse.coupling_id == restricted.coupling_id
    )
    assert embedded_fine.discretization_id != embedded_coarse.discretization_id
    assert restricted.discretization_id == embedded_coarse.discretization_id
    np.testing.assert_array_equal(restricted.velocity, embedded_coarse.velocity)
    np.testing.assert_array_equal(
        restricted.member_indices, embedded_coarse.member_indices
    )
    longer_domains = [a.domain for a in fine3.axes]
    longer_domains[axis] = AxisDomain.periodic(-0.2, 2.8)
    longer = fine3.plan.prepare(tuple(longer_domains))
    assert (
        embed_planar_velocity(state, longer, invariant_axis=axis).measure_id
        != restricted.measure_id
    )


@pytest.mark.parametrize("value", [True, "1"])
def test_initial_measure_scalar_kind_refusals(value: object) -> None:
    with pytest.raises(TypeError):
        TaylorGreenFamily(value)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        TaylorGreenFamily(2, amplitude=value)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        VortexPatchFamily(radius=value, center=(0.5, 0.5), strength=1.0)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        PiecewiseConstantVorticityFamily(partition=(value, 2), value_range=(-1.0, 1.0))  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        ShearLayerFamily(
            profile="smooth",
            thickness_range=(value, 0.1),  # ty: ignore[invalid-argument-type]
            perturbation_amplitude=0.0,
            perturbation_mode_range=(1, 2),
            interface_shift_range=(0.0, 0.0),
            oversampling=2,
        )
    with pytest.raises(TypeError):
        SinusoidalVortexSheetFamily(
            mollifier_radius=0.1,
            base_amplitude=0.2,
            perturbation_count=value,  # ty: ignore[invalid-argument-type]
            perturbation_amplitude_range=(0.0, 0.01),
            quadrature_points=16,
        )
    measure = PeriodicFlowInitialMeasure(TaylorGreenFamily(2))
    with pytest.raises(TypeError):
        measure.prepare(_space(7), maximum_bytes=value)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        measure.prepare(_space(7), member_chunk_size=value)  # ty: ignore[invalid-argument-type]
    state = measure.prepare(_space(7), member_chunk_size=1).realize(
        jax.random.key(11), np.array([0], dtype=np.uint32)
    )
    with pytest.raises(TypeError):
        embed_planar_velocity(state, _space(7, 3), invariant_axis=value)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        restrict_periodic_flow_initial_state(state, _space(5), maximum_bytes=value)  # ty: ignore[invalid-argument-type]


def test_initial_measure_typed_byte_refusals() -> None:
    space = _space(7)
    measure = PeriodicFlowInitialMeasure(TaylorGreenFamily(2))
    prepared = measure.prepare(space, member_chunk_size=1)
    bounded = measure.prepare(
        space, member_chunk_size=1, maximum_bytes=prepared.member_bytes
    )
    with pytest.raises(MemoryBudgetError) as realization_error:
        bounded.realize(jax.random.key(11), np.array([0, 1], dtype=np.uint32))
    assert realization_error.value.budget == "maximum_bytes"
    assert realization_error.value.required_bytes == 2 * prepared.member_bytes
    assert realization_error.value.limit_bytes == prepared.member_bytes
    state = prepared.realize(jax.random.key(11), np.array([0], dtype=np.uint32))
    with pytest.raises(MemoryBudgetError) as restriction_error:
        restrict_periodic_flow_initial_state(state, _space(5), maximum_bytes=1)
    assert restriction_error.value.budget == "maximum_bytes"
    assert restriction_error.value.required_bytes == 2 * state.velocity.nbytes
    assert restriction_error.value.limit_bytes == 1


def test_vortex_sheet_reported_residual_matches_numpy_trapezoid_sums() -> None:
    space = _space(7, length=1.7)
    family = SinusoidalVortexSheetFamily(
        mollifier_radius=0.09,
        base_amplitude=0.2,
        perturbation_count=3,
        perturbation_amplitude_range=(0.0, 0.015),
        quadrature_points=8,
    )
    state = (
        PeriodicFlowInitialMeasure(family)
        .prepare(space, member_chunk_size=1)
        .realize(jax.random.key(610), np.array([7], dtype=np.uint32))
    )
    alpha, beta = (np.asarray(state.parameters[name])[0] for name in ("alpha", "beta"))
    nx, ny = np.meshgrid(
        *(np.asarray(a.modes.mode_numbers) for a in space.axes), indexing="ij"
    )
    lengths = tuple(float(a.length) for a in space.axes)
    magnitude = np.hypot(2 * np.pi * nx / lengths[0], 2 * np.pi * ny / lengths[1])
    kernels = np.array(
        [
            quad(
                lambda r: (
                    2
                    * np.pi
                    * r
                    * _published_mollifier(r)
                    * j0(k * family.mollifier_radius * r)
                ),
                0,
                1,
                points=[0.5],
                epsabs=1e-13,
            )[0]
            for k in magnitude.ravel()
        ],
        dtype=np.float64,
    ).reshape(space.modal_shape)

    def modal_sum(points: int) -> np.ndarray:
        x = np.arange(points, dtype=np.float64) / points
        angle = 2 * np.pi * (x[:, None] + beta[None, :])
        graph = (
            0.5
            + family.base_amplitude * np.sin(2 * np.pi * x)
            + np.sum(alpha * np.sin(angle), axis=1)
        )
        slope = (
            2
            * np.pi
            * (
                family.base_amplitude * np.cos(2 * np.pi * x)
                + np.sum(alpha * np.cos(angle), axis=1)
            )
        )
        weights = np.sqrt(lengths[0] ** 2 + (lengths[1] * slope) ** 2) / points
        phase = np.exp(-2j * np.pi * (nx[..., None] * x + ny[..., None] * graph))
        coefficients = (
            np.sum(phase * weights, axis=-1) * kernels / np.sqrt(np.prod(lengths))
        )
        return np.where(magnitude > 0, coefficients, 0.0)

    coarse = modal_sum(family.quadrature_points)
    refined = modal_sum(2 * family.quadrature_points)
    expected = np.linalg.norm(refined - coarse)
    assert expected > 1e-3
    np.testing.assert_allclose(
        state.quadrature_residual[0], expected, atol=2e-13, rtol=2e-12
    )
