from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import (
    AxisDomain,
    FourierBasisPlan,
    SineBasisPlan,
    SpectralVanishingViscosityPlan,
    SpectralViscosityKernel,
    TensorSpectralDiscretization,
    TensorSpectralPlan,
)


def _space(count: int, *, length: float = 2.0 * np.pi) -> TensorSpectralDiscretization:
    return TensorSpectralPlan(
        (FourierBasisPlan(count), FourierBasisPlan(count)), axis_names=("x", "y")
    ).prepare((AxisDomain.periodic(0.0, length),) * 2)


def _plan(
    kernel: SpectralViscosityKernel = "tadmor_step",
) -> SpectralVanishingViscosityPlan:
    return SpectralVanishingViscosityPlan(
        strength=0.4,
        order=2.0,
        cutoff_scale=0.5,
        cutoff_exponent=0.3,
        kernel=kernel,
        highpass_exponent=4 if kernel == "exponential_highpass" else None,
    )


@pytest.mark.parametrize(
    "count", [7, 8, 13, 16], ids=["odd7", "even8", "odd13", "even16"]
)
@pytest.mark.parametrize("kernel", ["tadmor_step", "exponential_highpass"])
def test_resolution_scaling_and_modal_profile(
    count: int, kernel: SpectralViscosityKernel
) -> None:
    space = _space(count, length=3.0)
    prepared = _plan(kernel).prepare(space)
    resolution = (count - 1) // 2
    scaled = 0.4 * resolution**-3
    cutoff = 0.5 * resolution**0.3
    numbers = np.rint(np.fft.fftfreq(count) * count)
    squared = numbers[:, None] ** 2 + numbers[None, :] ** 2
    profile = (
        (np.sqrt(squared) > cutoff)
        if kernel == "tadmor_step"
        else -np.expm1(-((np.sqrt(squared) / cutoff) ** 4))
    )
    expected = -scaled * ((2.0 * np.pi / 3.0) ** 2 * squared) ** 2 * profile
    if count % 2 == 0:
        expected[count // 2, :] = 0.0
        expected[:, count // 2] = 0.0
    np.testing.assert_allclose(prepared.decay_rate, expected, atol=1e-13)
    assert prepared.resolution == resolution
    assert prepared.scaled_strength == pytest.approx(scaled)
    assert prepared.cutoff_mode == pytest.approx(cutoff)
    assert prepared.maximum_decay_rate == pytest.approx(float(np.max(-expected)))
    assert prepared.active_mode_count == np.count_nonzero(expected < 0.0)
    assert prepared.operator_scaling_admissible == (kernel == "tadmor_step")
    assert not prepared.cited_convergence_supported
    assert (
        prepared.support_reason
        == "Operator scaling alone does not establish cited rough-data convergence: the initial-data mollification and full-method hypotheses are not certified."
    )


def test_step_kernel_exactly_vanishes_below_cutoff() -> None:
    prepared = SpectralVanishingViscosityPlan(
        strength=0.4,
        order=1.0,
        cutoff_scale=1.0,
        cutoff_exponent=0.4,
        kernel="tadmor_step",
    ).prepare(_space(16))
    assert prepared.decay_rate[0, 0] == 0.0
    assert prepared.decay_rate[1, 0] == 0.0
    assert prepared.decay_rate[2, 0] == 0.0
    assert prepared.decay_rate[3, 0] < 0.0


@pytest.mark.parametrize(
    "theta,admissible",
    [(0.49, True), (0.5, False), (1.0, False)],
    ids=["strict-interior", "boundary", "outside"],
)
def test_operator_scaling_is_not_a_convergence_claim(
    theta: float, admissible: bool
) -> None:
    prepared = SpectralVanishingViscosityPlan(
        strength=0.1,
        order=1.0,
        cutoff_scale=0.25,
        cutoff_exponent=theta,
        kernel="tadmor_step",
    ).prepare(_space(8))
    assert prepared.operator_scaling_admissible == admissible
    assert not prepared.cited_convergence_supported


@pytest.mark.parametrize(
    "parameter,value,error",
    [
        ("strength", 0.0, ValueError),
        ("strength", -0.1, ValueError),
        ("strength", np.inf, ValueError),
        ("order", 0.9, ValueError),
        ("order", np.nan, ValueError),
        ("cutoff_scale", 0.0, ValueError),
        ("cutoff_exponent", 0.0, ValueError),
        ("cutoff_exponent", 1.1, ValueError),
        ("strength", True, TypeError),
        ("strength", "0.1", TypeError),
        ("order", True, TypeError),
        ("order", "1.0", TypeError),
        ("cutoff_scale", True, TypeError),
        ("cutoff_scale", "1.0", TypeError),
        ("cutoff_exponent", True, TypeError),
        ("cutoff_exponent", "0.4", TypeError),
    ],
    ids=[
        "zero-strength",
        "negative-strength",
        "infinite-strength",
        "low-order",
        "nan-order",
        "zero-cutoff",
        "zero-theta",
        "large-theta",
        "bool-strength",
        "string-strength",
        "bool-order",
        "string-order",
        "bool-cutoff",
        "string-cutoff",
        "bool-theta",
        "string-theta",
    ],
)
def test_parameter_refusals(
    parameter: Literal["strength", "order", "cutoff_scale", "cutoff_exponent"],
    value: float | str,
    error: type[Exception],
) -> None:
    values: dict[str, float | str] = dict(
        strength=0.1, order=1.0, cutoff_scale=1.0, cutoff_exponent=0.4
    )
    values[parameter] = value
    with pytest.raises(error):
        SpectralVanishingViscosityPlan(
            strength=values["strength"],  # ty: ignore[invalid-argument-type]
            order=values["order"],  # ty: ignore[invalid-argument-type]
            cutoff_scale=values["cutoff_scale"],  # ty: ignore[invalid-argument-type]
            cutoff_exponent=values["cutoff_exponent"],  # ty: ignore[invalid-argument-type]
            kernel="tadmor_step",
        )


@pytest.mark.parametrize(
    "kernel,exponent,error",
    [
        ("tadmor_step", 2, ValueError),
        ("exponential_highpass", None, ValueError),
        ("exponential_highpass", 1, ValueError),
        ("exponential_highpass", True, TypeError),
        ("exponential_highpass", "2", TypeError),
        ("tadmor_step", True, TypeError),
        ("tadmor_step", "2", TypeError),
    ],
    ids=[
        "step-exponent",
        "missing-exponent",
        "low-exponent",
        "bool-exponent",
        "string-exponent",
        "step-bool-exponent",
        "step-string-exponent",
    ],
)
def test_kernel_exponent_refusals(
    kernel: SpectralViscosityKernel,
    exponent: int | str | None,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        SpectralVanishingViscosityPlan(
            strength=0.1,
            order=1.0,
            cutoff_scale=0.5,
            cutoff_exponent=0.4,
            kernel=kernel,
            highpass_exponent=exponent,  # ty: ignore[invalid-argument-type]
        )


def test_unknown_kernel_refused() -> None:
    with pytest.raises(ValueError):
        SpectralVanishingViscosityPlan(
            strength=0.1,
            order=1.0,
            cutoff_scale=0.5,
            cutoff_exponent=0.4,
            kernel="gaussian",  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    "case", ["anisotropic-length", "anisotropic-count", "nonfourier", "undamped"], ids=str
)
def test_preparation_refuses_incompatible_spaces(case: str) -> None:
    if case == "anisotropic-length":
        space = TensorSpectralPlan(
            (FourierBasisPlan(8), FourierBasisPlan(8)), axis_names=("x", "y")
        ).prepare((AxisDomain.periodic(0.0, 1.0), AxisDomain.periodic(0.0, 2.0)))
    elif case == "anisotropic-count":
        space = TensorSpectralPlan(
            (FourierBasisPlan(8), FourierBasisPlan(10)), axis_names=("x", "y")
        ).prepare((AxisDomain.periodic(0.0, 1.0),) * 2)
    elif case == "nonfourier":
        space = TensorSpectralPlan(
            (SineBasisPlan(8), FourierBasisPlan(8)), axis_names=("x", "y")
        ).prepare((AxisDomain.interval(0.0, 1.0), AxisDomain.periodic(0.0, 1.0)))
    else:
        space = _space(8)
    plan = (
        _plan()
        if case != "undamped"
        else SpectralVanishingViscosityPlan(
            strength=0.1,
            order=1.0,
            cutoff_scale=100.0,
            cutoff_exponent=0.4,
            kernel="tadmor_step",
        )
    )
    with pytest.raises(ValueError):
        plan.prepare(space)


def test_dissipation_matches_analytic_physical_quadrature() -> None:
    space = _space(16)
    prepared = _plan().prepare(space)
    x, y = jnp.meshgrid(space.axes[0].nodes, space.axes[1].nodes, indexing="ij")
    field = jnp.stack((jnp.sin(3.0 * y), 0.5 * jnp.cos(2.0 * x)), axis=-1)
    coefficients = space.project(field)
    rates = space.reconstruct(prepared.rate(coefficients))
    direct = -space.integral(jnp.sum(field * rates, axis=-1))
    # Independent analytic torus integrals: each sine/cosine square has V/2.
    scaled = 0.4 * 7.0**-3
    expected = scaled * (3.0**4 + 0.25 * 2.0**4) * (2.0 * np.pi) ** 2 / 2.0
    np.testing.assert_allclose(prepared.dissipation(coefficients), expected, rtol=2e-14)
    np.testing.assert_allclose(direct, expected, rtol=2e-14)


def test_scientific_identity_uses_resolved_parameters() -> None:
    canonical = _plan()
    numpy_values = SpectralVanishingViscosityPlan(
        strength=np.float64(0.4),
        order=np.float64(2.0),
        cutoff_scale=np.float64(0.5),
        cutoff_exponent=np.float64(0.3),
        kernel="tadmor_step",
    )
    assert canonical.plan_id == numpy_values.plan_id
    assert (
        canonical.prepare(_space(8)).prepared_id
        != canonical.prepare(_space(16)).prepared_id
    )
