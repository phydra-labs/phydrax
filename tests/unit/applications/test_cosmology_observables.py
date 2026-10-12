from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


NATIVE_DIFFERENTIATION = phx.DerivativeContract.smooth(
    (
        phx.DerivativeSurface.INPUT,
        phx.DerivativeSurface.MODEL_PARAMETER,
        phx.DerivativeSurface.PHYSICAL_PARAMETER,
        phx.DerivativeSurface.STORED_VALUES,
    )
)


cosmology = phx.applications.cosmology


def _context() -> Any:
    scale = cosmology.CosmologyScaleContract(
        cosmology.CODE_COSMOLOGY_SCALE.length_unit,
        cosmology.CODE_COSMOLOGY_SCALE.mass_unit,
        cosmology.CODE_COSMOLOGY_SCALE.time_unit,
    )
    background = cosmology.FLRWBackground(1.0, 1.0, scale=scale)
    growth = cosmology.FLRWGrowthPlan(jnp.geomspace(1.0e-2, 1.0, 32)).solve(background)
    provenance = cosmology.CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="survey-test-power",
        numerical_policy_id="survey-test-grid",
        physics_policy_id="linear-total-matter",
        scale_id=scale.scale_id,
        source_kind="native",
        differentiation=NATIVE_DIFFERENTIATION,
    )
    k = jnp.geomspace(0.05, 2000.0, 512)
    power = cosmology.MatterPowerTable(
        # ty: ignore[invalid-argument-type]
        [0.5, 1.0],
        k,
        jnp.ones((2, k.size)),
        cosmology.MatterPowerDescriptor("total_matter", "total_matter"),
        scale,
        provenance,
        background.realization,
    )
    return background, growth, power


def test_cosmology_observables_scenario_1() -> None:
    background, _, power = _context()
    distance = cosmology.FLRWDistancePlan(light_speed=1.0, order=64)
    grid = cosmology.RadialGrid(jnp.linspace(0.1, 0.9, 48))
    distribution = cosmology.RedshiftDistribution(
        grid, jnp.exp(-(((grid.redshifts - 0.5) / 0.15) ** 2)), "bin-0"
    )
    first = cosmology.LinearDensityTracer(distribution, 1.0)
    second = cosmology.LinearDensityTracer(distribution, 2.0)
    # ty: ignore[invalid-argument-type]
    plan = cosmology.LimberAngularPowerPlan([10, 20, 50], 2)
    prediction = plan.predict(background, distance, power, (first, second))
    assert bool(prediction.successful)
    np.testing.assert_allclose(prediction.values[2], 4.0 * prediction.values[0])
    np.testing.assert_allclose(prediction.values[1], 2.0 * prediction.values[0])

    lensing = cosmology.LensingConvergenceTracer(distribution)
    # ty: ignore[invalid-argument-type]
    lensing_plan = cosmology.LimberAngularPowerPlan([20, 50], 1)
    lensing_prediction = lensing_plan.predict(background, distance, power, (lensing,))
    assert bool(lensing_prediction.successful)
    assert jnp.all(lensing_prediction.values >= 0.0)
    background, growth, power = _context()
    distance = cosmology.FLRWDistancePlan(light_speed=1.0, order=64)
    plan = cosmology.LinearRSDMultipolePlan(jnp.asarray([0.1, 0.2, 0.5]), mu_order=64)
    result = plan.predict(
        background,
        background,
        distance,
        growth,
        power,
        2.0,
        0.5,
    )
    np.testing.assert_allclose(result.alpha_perpendicular, 1.0, rtol=1e-12)
    np.testing.assert_allclose(result.alpha_parallel, 1.0, rtol=1e-12)
    np.testing.assert_allclose(result.monopole, 4.0 + 4.0 / 3.0 + 1.0 / 5.0, rtol=1e-10)
    np.testing.assert_allclose(result.quadrupole, 8.0 / 3.0 + 4.0 / 7.0, rtol=1e-10)
    np.testing.assert_allclose(result.hexadecapole, 8.0 / 35.0, rtol=1e-10)
    assert bool(result.successful)


@pytest.mark.parametrize("node_count", (3, 17, 257))
def test_limber_lensing_matches_dense_nonuniform_reference(node_count: int) -> None:
    background, _, power = _context()
    distance = cosmology.FLRWDistancePlan(light_speed=1.0, order=64)
    redshifts = 0.1 + 0.8 * np.linspace(0.0, 1.0, node_count) ** 1.4
    grid = cosmology.RadialGrid(redshifts)
    distribution = cosmology.RedshiftDistribution(
        grid, np.exp(-(((redshifts - 0.5) / 0.2) ** 2)), "dense-reference"
    )
    tracer = cosmology.LensingConvergenceTracer(distribution)
    plan = cosmology.LimberAngularPowerPlan(jnp.asarray([20, 50]), 1)
    prediction = plan.predict(background, distance, power, (tracer,))
    radial = np.asarray(
        distance.evaluate(background, grid.redshifts).radial_comoving_distance
    )
    weights = np.asarray(grid.weights)
    density = np.asarray(distribution.values)
    efficiency = np.sum(
        weights[None, :]
        * density[None, :]
        * np.maximum(radial[None, :] - radial[:, None], 0.0)
        / radial[None, :],
        axis=1,
    )
    hubble = np.asarray(background.hubble(jnp.asarray(1.0 / (1.0 + redshifts))))
    coefficient = 1.5 * background.matter_density * background.hubble_constant**2
    kernel = coefficient * radial * (1.0 + redshifts) * efficiency
    expected = np.sum(weights * kernel**2 / (hubble * radial**2))
    assert bool(prediction.successful)
    np.testing.assert_allclose(prediction.values, expected, rtol=1e-12, atol=1e-16)


@pytest.mark.parametrize("node_count", (3, 17, 257))
def test_limber_efficiency_zero_radius_dense_reference(node_count: int) -> None:
    from phydrax.applications.cosmology._observables import _lensing_efficiency

    radial = np.linspace(0.0, 1.0, node_count) ** 1.7
    weighted_density = np.linspace(0.2, 1.0, node_count) / node_count
    safe = np.where(radial > 0.0, radial, 1.0)
    expected = np.sum(
        np.maximum(radial[None, :] - radial[:, None], 0.0)
        / safe[None, :]
        * weighted_density[None, :],
        axis=1,
    )
    np.testing.assert_allclose(
        _lensing_efficiency(jnp.asarray(radial), jnp.asarray(weighted_density)),
        expected,
        rtol=1e-12,
        atol=1e-16,
    )


def test_limber_dense_oracle_kills_inclusive_density_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jax import Array

    from phydrax.applications.cosmology import _observables

    original = _observables._lensing_efficiency

    def inclusive_density_tail(radial: Array, weighted_density: Array, /) -> Array:
        # Making both tails inclusive cancels exactly; mutate only the density tail.
        return original(radial, weighted_density) + weighted_density

    monkeypatch.setattr(_observables, "_lensing_efficiency", inclusive_density_tail)
    with pytest.raises(AssertionError):
        test_limber_lensing_matches_dense_nonuniform_reference(17)
