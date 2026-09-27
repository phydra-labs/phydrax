#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.equations._relativistic_radiation import GRGrayM1RadiationSystem
from phydrax.equations._relativistic_radiation_interaction import (
    ConstantGRGrayOpacityPlan,
    GRGrayRadiationInteractionPlan,
)
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.units import KILOGRAM


def _scale() -> Any:
    return RelativityScaleContract.geometric(KILOGRAM)


def _nonunit_scale() -> Any:
    return RelativityScaleContract(
        DimensionalScaleContract.si(),
        gravitational_constant=1,
        speed_of_light=4,
        reduced_planck_constant=1,
        boltzmann_constant=1,
    )


def _geometry(
    scale: Any, convention: Any, *, alpha: Any = 1.0, shift: Any = (0.0, 0.0, 0.0)
) -> Any:
    dtype = jnp.float32
    identity = jnp.eye(3, dtype=dtype)
    return ADMGridGeometry(
        jnp.asarray(alpha, dtype=dtype),
        jnp.asarray(shift, dtype=dtype),
        identity,
        identity,
        jnp.asarray(1.0, dtype=dtype),
        jnp.zeros((3, 3), dtype=dtype),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id="single-cell",
        geometry_lineage_id="flat-single-cell",
    )


def test_relativistic_radiation_scenario_1() -> None:
    scale = _scale()
    convention = RelativityConvention.canonical()
    geometry = _geometry(scale, convention)
    system = GRGrayM1RadiationSystem(scale, convention, reduced_light_speed=1.0)

    pressure, reduced, qualified = jax.jit(
        lambda energy, flux: (
            system.closure(energy, flux, geometry).pressure_tensor,
            system.closure(energy, flux, geometry).reduced_flux,
            system.closure(energy, flux, geometry).qualified,
        )
    )(jnp.asarray(3.0), jnp.zeros(3))
    np.testing.assert_allclose(pressure, jnp.eye(3), rtol=1.0e-6, atol=1.0e-7)
    np.testing.assert_allclose(reduced, 0.0)
    assert bool(qualified)
    projection = system.stress_energy_projection(jnp.asarray(3.0), jnp.zeros(3), geometry)
    np.testing.assert_allclose(projection.stress_covariant, jnp.eye(3), rtol=1.0e-6)
    assert projection.geometry_lineage_id == geometry.geometry_lineage_id
    assert bool(projection.compatible_with(geometry))
    assert bool(projection.valid)

    streaming = system.closure(jnp.asarray(3.0), jnp.asarray((3.0, 0.0, 0.0)), geometry)
    np.testing.assert_allclose(streaming.pressure_tensor[0, 0], 3.0, rtol=1.0e-6)
    np.testing.assert_allclose(streaming.pressure_tensor[1:, 1:], 0.0, atol=1.0e-6)
    assert bool(streaming.physically_valid)
    assert not bool(streaming.derivative_valid)
    scale = _scale()
    convention = RelativityConvention.canonical()
    geometry = _geometry(scale, convention, alpha=0.5, shift=(0.1, 0.0, 0.0))
    system = GRGrayM1RadiationSystem(scale, convention, reduced_light_speed=1.0)

    lower, upper = system.coordinate_characteristic_bounds(
        jnp.asarray((1.0, 0.0, 0.0)), geometry
    )

    np.testing.assert_allclose(lower, -0.6, rtol=1.0e-6)
    np.testing.assert_allclose(upper, 0.4, rtol=1.0e-6)
    scale = _scale()
    convention = RelativityConvention.canonical()
    geometry = _geometry(scale, convention)
    system = GRGrayM1RadiationSystem(scale, convention)
    thin = GRGrayRadiationInteractionPlan(
        system,
        ConstantGRGrayOpacityPlan(),
        radiation_constant=1.0,
    )
    coupled = GRGrayRadiationInteractionPlan(
        system,
        ConstantGRGrayOpacityPlan(
            planck_absorption=2.0,
            scattering=3.0,
        ),
        radiation_constant=1.0,
    )
    arguments = (
        jnp.asarray(2.0),
        jnp.asarray((0.25, 0.0, 0.0)),
        jnp.asarray(1.0),
        jnp.zeros(3),
        jnp.asarray(1.0),
        geometry,
    )

    transparent = thin.matter_exchange(*arguments)
    exchange = coupled.matter_exchange(*arguments)

    np.testing.assert_allclose(transparent.radiation_energy_source, 0.0)
    np.testing.assert_allclose(transparent.radiation_flux_source, 0.0)
    np.testing.assert_allclose(
        exchange.radiation_energy_source + exchange.matter_energy_source, 0.0
    )
    np.testing.assert_allclose(
        exchange.radiation_flux_source + exchange.matter_momentum_source,
        jnp.zeros(3),
    )
    assert float(exchange.radiation_energy_source) < 0.0
    assert float(exchange.radiation_flux_source[0]) < 0.0
    assert bool(exchange.converged)
    assert bool(exchange.qualified)
    scale = _nonunit_scale()
    convention = RelativityConvention.canonical()
    geometry = _geometry(scale, convention)
    system = GRGrayM1RadiationSystem(
        scale,
        convention,
        reduced_light_speed=1.0,
    )
    interaction = GRGrayRadiationInteractionPlan(
        system,
        ConstantGRGrayOpacityPlan(
            planck_absorption=2.0,
            scattering=3.0,
        ),
        radiation_constant=1.0,
    )
    energy = jnp.asarray(2.0)
    flux = jnp.asarray((0.25, 0.0, 0.0))

    projection = system.stress_energy_projection(energy, flux, geometry)
    lower, upper = system.coordinate_characteristic_bounds(
        jnp.asarray((1.0, 0.0, 0.0)), geometry
    )
    rest_exchange = interaction.matter_exchange(
        energy, flux, jnp.asarray(1.0), jnp.zeros(3), jnp.asarray(1.0), geometry
    )
    moving_exchange = interaction.matter_exchange(
        energy,
        jnp.zeros(3),
        jnp.asarray(1.0),
        jnp.asarray((2.0, 0.0, 0.0)),
        jnp.asarray(1.0),
        geometry,
    )
    physical_closure = system.closure(energy, jnp.asarray((2.0, 0.0, 0.0)), geometry)

    np.testing.assert_allclose(projection.momentum_covector, flux / 4.0)
    np.testing.assert_allclose((lower, upper), (-1.0, 1.0))
    np.testing.assert_allclose(physical_closure.reduced_flux, 0.25)
    assert bool(system.admissible(jnp.asarray((2.0, 4.0, 0.0, 0.0))))
    np.testing.assert_allclose(rest_exchange.radiation_energy_source, -8.0)
    np.testing.assert_allclose(
        rest_exchange.radiation_flux_source, jnp.asarray((-5.0, 0.0, 0.0))
    )
    np.testing.assert_allclose(
        rest_exchange.radiation_momentum_source,
        jnp.asarray((-1.25, 0.0, 0.0)),
    )
    np.testing.assert_allclose(
        rest_exchange.radiation_momentum_source + rest_exchange.matter_momentum_source,
        jnp.zeros(3),
    )
    assert bool(moving_exchange.physically_valid)
    assert bool(moving_exchange.qualified)
