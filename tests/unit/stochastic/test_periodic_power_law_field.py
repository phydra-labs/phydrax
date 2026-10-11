from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0

from phydrax.discretization import AxisDomain, FourierBasisPlan, TensorSpectralPlan
from phydrax.discretization.spectral import (
    HermitianSpectralCoordinates,
    PeriodicLerayProjector,
)
from phydrax.discretization.spectral._space import TensorSpectralDiscretization
from phydrax.execution import MemoryBudgetError
from phydrax.stochastic import (
    FourierCoefficientLaw,
    FourierSeriesBasis,
    PeriodicPowerLawFieldPlan,
    PowerLawSpectralAmplitude,
)
from phydrax.stochastic._periodic_power_law_field import _tensor_trigonometric_modal


def _space(
    count: int, dimension: int = 2, length: float = 2 * np.pi
) -> TensorSpectralDiscretization:
    return TensorSpectralPlan(
        tuple(FourierBasisPlan(count) for _ in range(dimension))
    ).prepare(tuple(AxisDomain.periodic(0.3, 0.3 + length) for _ in range(dimension)))


@pytest.mark.parametrize("basis", ["complex_exponential", "tensor_trigonometric"])
@pytest.mark.parametrize("law", ["normal", "uniform"])
def test_member_subset_is_bitwise_independent(
    basis: FourierSeriesBasis, law: FourierCoefficientLaw
) -> None:
    prepared = PeriodicPowerLawFieldPlan(
        PowerLawSpectralAmplitude(1.25), basis=basis, law=law
    ).prepare(_space(8), member_chunk_size=3)
    key = jax.random.key(817)
    full = prepared.sample(key, np.arange(7, dtype=np.int64))
    subset = prepared.sample(key, np.asarray([6, 1, 4], dtype=np.int64))
    np.testing.assert_array_equal(
        subset.coefficients, np.asarray(full.coefficients)[[6, 1, 4]]
    )
    np.testing.assert_array_equal(
        subset.removed_coefficient_energy,
        np.asarray(full.removed_coefficient_energy)[[6, 1, 4]],
    )
    assert full.coupling_id == subset.coupling_id
    assert full.coefficient_realization_id == subset.coefficient_realization_id


@pytest.mark.parametrize("basis", ["complex_exponential", "tensor_trigonometric"])
def test_chunk_policy_preserves_member_fields(basis: FourierSeriesBasis) -> None:
    plan = PeriodicPowerLawFieldPlan(
        PowerLawSpectralAmplitude(1.7), basis=basis, component_shape=(2,), solenoidal=True
    )
    space = _space(7)
    first, second = (
        plan.prepare(space, member_chunk_size=1),
        plan.prepare(space, member_chunk_size=4),
    )
    indices = np.arange(5, dtype=np.uint32)
    left, right = (
        first.sample(jax.random.key(129), indices),
        second.sample(jax.random.key(129), indices),
    )
    tolerance = 64 * np.finfo(left.coefficients.real.dtype).eps
    np.testing.assert_allclose(
        left.coefficients, right.coefficients, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(
        left.removed_coefficient_energy,
        right.removed_coefficient_energy,
        rtol=tolerance,
        atol=tolerance,
    )
    assert first.field_id == second.field_id
    assert first.prepared_id != second.prepared_id


@pytest.mark.parametrize("basis", ["complex_exponential", "tensor_trigonometric"])
def test_restriction_is_bitwise_transitive_with_cumulative_energy(
    basis: FourierSeriesBasis,
) -> None:
    fine, middle, coarse = _space(12), _space(9), _space(6)
    prepared = PeriodicPowerLawFieldPlan(
        PowerLawSpectralAmplitude(1.5), basis=basis, component_shape=(2,), solenoidal=True
    ).prepare(fine)
    initial = prepared.sample(jax.random.key(7), np.arange(3, dtype=np.int32))
    intermediate = prepared.restrict(initial, middle)
    staged = prepared.restrict(intermediate, coarse)
    direct = prepared.restrict(initial, coarse)
    np.testing.assert_array_equal(staged.coefficients, direct.coefficients)
    np.testing.assert_array_equal(
        staged.removed_coefficient_energy, direct.removed_coefficient_energy
    )
    lost = np.sum(np.abs(initial.coefficients) ** 2, axis=(1, 2, 3)) - np.sum(
        np.abs(direct.coefficients) ** 2, axis=(1, 2, 3)
    )
    np.testing.assert_allclose(
        direct.removed_coefficient_energy,
        initial.removed_coefficient_energy + lost,
        atol=1e-12,
    )
    assert direct.coupling_id == initial.coupling_id
    assert direct.coefficient_realization_id == initial.coefficient_realization_id
    assert direct.discretization_id == coarse.prepared_id


@pytest.mark.parametrize("count", [7, 8])
@pytest.mark.parametrize("basis", ["complex_exponential", "tensor_trigonometric"])
def test_exact_reality_and_zero_forbidden_modes(
    count: int, basis: FourierSeriesBasis
) -> None:
    space = _space(count)
    prepared = PeriodicPowerLawFieldPlan(
        PowerLawSpectralAmplitude(1.0), basis=basis
    ).prepare(space)
    result = prepared.sample(jax.random.key(89), np.asarray([19], dtype=np.int32))
    coordinates = HermitianSpectralCoordinates(space)
    assert float(coordinates.reality_defect(result.coefficients[0])) == 0.0
    leakage = float(space.imaginary_leakage(result.coefficients[0]))
    if count % 2 == 0:
        assert leakage == 0.0
    else:
        # The canonical odd-length complex FFT has roundoff, although the modal
        # field's imaginary part is mathematically identically zero.
        assert leakage <= 8 * np.finfo(np.float64).eps
    assert result.coefficients[0, 0, 0] == 0
    if count % 2 == 0:
        np.testing.assert_array_equal(result.coefficients[:, count // 2, :], 0)
        np.testing.assert_array_equal(result.coefficients[:, :, count // 2], 0)


@pytest.mark.parametrize("basis", ["complex_exponential", "tensor_trigonometric"])
@pytest.mark.parametrize("law", ["normal", "uniform"])
def test_spectrum_matches_expected_moment_with_five_sigma(
    basis: FourierSeriesBasis, law: FourierCoefficientLaw
) -> None:
    # All retained nonzero modes, not a selected trio. Normal complex power is
    # exponential with normalized variance 1. For uniform latents mu4=9/5:
    # Var(|z|²)=1+(mu4-3)*sum|w_j|^4. The complex basis has two weights 1/sqrt(2);
    # tensor trig has 2**r weights of magnitude 2**(-r/2) at r active axes.
    members = 8192
    space = _space(7)
    prepared = PeriodicPowerLawFieldPlan(
        PowerLawSpectralAmplitude(1.4, scale=0.7), basis=basis, law=law
    ).prepare(space, member_chunk_size=16)
    result = prepared.sample(jax.random.key(7325), np.arange(members, dtype=np.int32))
    modes = np.fft.fftfreq(7) * 7
    kx, ky = np.meshgrid(modes, modes, indexing="ij")
    squared_wave = kx**2 + ky**2
    retained = squared_wave > 0
    expectation = (2 * np.pi) ** 2 * 0.7**2 * squared_wave[retained] ** (-1.4)
    active_axes = ((kx != 0).astype(np.int32) + (ky != 0))[retained]
    fourth_weight_sum = (
        np.full(expectation.shape, 0.5, dtype=np.float64)
        if basis == "complex_exponential"
        else 2.0 ** (-active_axes)
    )
    variance = (
        np.ones_like(expectation)
        if law == "normal"
        else 1.0 + (9 / 5 - 3) * fourth_weight_sum
    )
    tolerance = 5 * expectation * np.sqrt(variance / members)
    observed = np.mean(np.abs(np.asarray(result.coefficients)[:, retained]) ** 2, axis=0)
    assert np.all(np.abs(observed - expectation) <= tolerance), (observed, expectation)
    # A five-percent amplitude defect must fail this same acceptance criterion.
    faulty_observed = 1.05**2 * observed
    assert not np.all(np.abs(faulty_observed - expectation) <= tolerance)


@pytest.mark.parametrize("dimension", [2, 3], ids=["2d", "3d"])
@pytest.mark.parametrize("hurst", [0.25, 0.5, 0.75], ids=["H025", "H050", "H075"])
def test_analytic_expected_increment_exponent(dimension: int, hurst: float) -> None:
    # Infinite-volume spectral expectation, not a false exact power-law claim for
    # a finite torus. Radial measure k^(d-1), density A(k)^2, and 1-J0(kr)
    # (2D) or 1-sinc(kr) (3D). Dimensionless bounds transform with radius.
    amplitude = PeriodicPowerLawFieldPlan.fractional_gaussian(
        hurst, spatial_dimension=dimension
    ).amplitude

    def expectation(radius: float) -> float:
        def integrand(log_t: float) -> float:
            t = np.exp(log_t)
            k = t / radius
            density = amplitude.scale**2 * k ** (-2 * amplitude.exponent)
            angular = j0(t) if dimension == 2 else np.sinc(t / np.pi)
            return float(k**dimension * density * (1 - angular))

        # A relative-only error target preserves the same quadrature admission
        # for the two expectations, whose magnitudes differ by r**(2H).
        value, _ = quad(integrand, -8, 6, epsabs=0.0, epsrel=1e-10, limit=1000)
        return value

    slope = np.log(expectation(0.2) / expectation(0.1)) / np.log(2)
    np.testing.assert_allclose(slope, 2 * hurst, rtol=0.0, atol=1e-8)


@pytest.mark.parametrize("count", [7, 8])
@pytest.mark.parametrize("kind", ["cos", "sin"])
def test_single_tensor_trig_latent_reproduces_analytic_field(
    count: int, kind: Literal["cos", "sin"]
) -> None:
    space = _space(count)
    alpha = jnp.zeros(
        ((count - 1) // 2 + 1, 2, (count - 1) // 2 + 1, 2), dtype=jnp.float64
    )
    alpha = alpha.at[1, 0 if kind == "cos" else 1, 2, 1].set(0.75)
    coefficients = _tensor_trigonometric_modal(alpha, space.modal_shape) * (2 * np.pi)
    x, y = np.meshgrid(
        np.asarray(space.axes[0].nodes) - 0.3,
        np.asarray(space.axes[1].nodes) - 0.3,
        indexing="ij",
    )
    first = np.cos(x) if kind == "cos" else np.sin(x)
    expected = 2 * 0.75 * first * np.sin(2 * y)
    np.testing.assert_allclose(space.reconstruct(coefficients), expected, atol=2e-14)


@pytest.mark.parametrize("dimension", [2, 3])
def test_solenoidal_members_are_divergence_free(dimension: int) -> None:
    space = _space(7, dimension)
    prepared = PeriodicPowerLawFieldPlan.fractional_gaussian(
        0.5, spatial_dimension=dimension, component_shape=(dimension,), solenoidal=True
    ).prepare(space)
    result = prepared.sample(jax.random.key(10), np.arange(3, dtype=np.int32))
    divergence = jax.vmap(PeriodicLerayProjector(space).divergence)(result.coefficients)
    np.testing.assert_allclose(divergence, 0.0, atol=2e-14)
    assert np.all(np.asarray(result.removed_coefficient_energy) > 0)


@pytest.mark.parametrize(
    "indices,error",
    [
        (np.asarray([], dtype=np.int32), ValueError),
        (np.asarray([[-1]], dtype=np.int32), ValueError),
        (np.asarray([-1], dtype=np.int32), ValueError),
        (np.asarray([2**32], dtype=np.int64), ValueError),
        (np.asarray([1.0], dtype=np.float64), TypeError),
    ],
)
def test_member_address_refusal(indices: np.ndarray, error: type[Exception]) -> None:
    prepared = PeriodicPowerLawFieldPlan(PowerLawSpectralAmplitude(1)).prepare(_space(4))
    with pytest.raises(error):
        prepared.sample(jax.random.key(1), indices)


def test_legacy_keys_are_refused() -> None:
    prepared = PeriodicPowerLawFieldPlan(PowerLawSpectralAmplitude(1)).prepare(_space(4))
    with pytest.raises(TypeError):
        prepared.sample(jax.random.PRNGKey(1), np.asarray([0], dtype=np.int32))


@pytest.mark.parametrize("target", ["refinement", "domain", "rank"])
def test_incompatible_restriction_refusal(target: str) -> None:
    prepared = PeriodicPowerLawFieldPlan(PowerLawSpectralAmplitude(1)).prepare(_space(6))
    realization = prepared.sample(jax.random.key(1), np.asarray([0], dtype=np.int32))
    space = (
        _space(8)
        if target == "refinement"
        else _space(4, length=1.0)
        if target == "domain"
        else _space(4, dimension=3)
    )
    with pytest.raises(ValueError):
        prepared.restrict(realization, space)


def test_prepare_resource_admission_refuses_before_sampling() -> None:
    with pytest.raises(MemoryBudgetError) as failure:
        PeriodicPowerLawFieldPlan(PowerLawSpectralAmplitude(1)).prepare(
            _space(8), maximum_sample_bytes=1
        )
    assert failure.value.budget == "maximum_sample_bytes"
    assert failure.value.limit_bytes == 1
    assert failure.value.required_bytes > 1


def test_sample_resource_admission_refuses_padded_members() -> None:
    plan, space = PeriodicPowerLawFieldPlan(PowerLawSpectralAmplitude(1)), _space(4)
    unconstrained = plan.prepare(space, member_chunk_size=2)
    prepared = plan.prepare(
        space, member_chunk_size=2, maximum_sample_bytes=2 * unconstrained.member_bytes
    )
    with pytest.raises(MemoryBudgetError) as failure:
        prepared.sample(jax.random.key(1), np.arange(3, dtype=np.int32))
    assert failure.value.budget == "maximum_sample_bytes"
    assert failure.value.required_bytes == 4 * prepared.member_bytes
    assert failure.value.limit_bytes == 2 * prepared.member_bytes


@pytest.mark.parametrize(
    "exponent,scale,cutoff", [(np.nan, 1.0, None), (1.0, 0.0, None), (1.0, 1.0, -1.0)]
)
def test_amplitude_parameter_refusal(
    exponent: float, scale: float, cutoff: float | None
) -> None:
    with pytest.raises(ValueError):
        PowerLawSpectralAmplitude(exponent, scale=scale, maximum_wavenumber=cutoff)


def test_radial_cutoff_and_zero_mode() -> None:
    amplitude = PowerLawSpectralAmplitude(1.5, scale=2, maximum_wavenumber=3)
    np.testing.assert_allclose(
        amplitude(jnp.asarray([0.0, 2.0, 3.0, 4.0], dtype=jnp.float64)),
        [0, 2 * 2**-1.5, 2 * 3**-1.5, 0],
    )


@pytest.mark.parametrize("dimension,components", [(1, (1,)), (2, ()), (3, (2,))])
def test_solenoidal_shape_refusal(dimension: int, components: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        PeriodicPowerLawFieldPlan(
            PowerLawSpectralAmplitude(1), component_shape=components, solenoidal=True
        ).prepare(_space(4, dimension))


def test_nonperiodic_basis_refusal() -> None:
    from phydrax.discretization import LegendreBasisPlan

    space = TensorSpectralPlan((LegendreBasisPlan(4),)).prepare(
        (AxisDomain.interval(-1, 1),)
    )
    with pytest.raises(ValueError):
        PeriodicPowerLawFieldPlan(PowerLawSpectralAmplitude(1)).prepare(space)


def test_non_member_separable_rng_is_refused() -> None:
    prepared = PeriodicPowerLawFieldPlan(PowerLawSpectralAmplitude(1)).prepare(_space(4))
    with pytest.raises(ValueError, match="threefry2x32"):
        prepared.sample(jax.random.key(1, impl="rbg"), np.asarray([0], dtype=np.int32))


@pytest.mark.parametrize("hurst", [0.0, 1.0, np.nan])
def test_fractional_hurst_refusal(hurst: float) -> None:
    with pytest.raises(ValueError):
        PeriodicPowerLawFieldPlan.fractional_gaussian(hurst, spatial_dimension=2)


@pytest.mark.parametrize("chunk,cap", [(0, 10000), (1, 0)])
def test_execution_policy_refusal(chunk: int, cap: int) -> None:
    with pytest.raises(ValueError):
        PeriodicPowerLawFieldPlan(PowerLawSpectralAmplitude(1)).prepare(
            _space(4), member_chunk_size=chunk, maximum_sample_bytes=cap
        )


@pytest.mark.parametrize("wrong", [True, "0.5", 0.5j])
def test_scientific_parameter_kinds_are_refused(wrong: object) -> None:
    with pytest.raises(TypeError):
        PowerLawSpectralAmplitude(wrong)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        PowerLawSpectralAmplitude(1.0, scale=wrong)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        PowerLawSpectralAmplitude(1.0, maximum_wavenumber=wrong)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        PeriodicPowerLawFieldPlan.fractional_gaussian(
            wrong,  # ty: ignore[invalid-argument-type]
            spatial_dimension=2,
        )
    with pytest.raises(TypeError):
        PeriodicPowerLawFieldPlan.fractional_gaussian(
            0.5,
            spatial_dimension=wrong,  # ty: ignore[invalid-argument-type]
        )


def test_plan_identity_owns_science_not_execution_policy() -> None:
    amplitude = PowerLawSpectralAmplitude(1.25)
    plan = PeriodicPowerLawFieldPlan(amplitude)
    identical = PeriodicPowerLawFieldPlan(PowerLawSpectralAmplitude(1.25))
    assert plan.plan_id == identical.plan_id
    assert plan.plan_id != PeriodicPowerLawFieldPlan(amplitude, law="uniform").plan_id
    assert (
        plan.plan_id
        != PeriodicPowerLawFieldPlan(amplitude, basis="tensor_trigonometric").plan_id
    )
    assert (
        plan.plan_id != PeriodicPowerLawFieldPlan(amplitude, component_shape=(2,)).plan_id
    )
    assert (
        plan.plan_id
        != PeriodicPowerLawFieldPlan(
            amplitude, component_shape=(2,), solenoidal=True
        ).plan_id
    )
    assert plan.prepare(_space(4)).field_id != plan.prepare(_space(5)).field_id
