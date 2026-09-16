#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax._physical import RelativityScaleContract
from phydrax.equations._relativistic_angular_radiation import (
    DiscreteOrdinatesRadiationPlan,
    MonteCarloRadiationClosurePlan,
    VariableEddingtonTensorClosurePlan,
)
from phydrax.equations._relativistic_multigroup_radiation import (
    GRMultigroupM1RadiationSystem,
    GRMultigroupRadiationInteractionPlan,
)
from phydrax.equations._relativistic_neutrino import (
    GRNeutrinoInteractionPlan,
    GRNeutrinoM1System,
)
from phydrax.equations._relativistic_radiation_interaction import (
    ConstantGRGreyOpacityPlan,
    GRGreyRadiationInteractionPlan,
)
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.units import KILOGRAM


def _contracts():
    return RelativityScaleContract.geometric(KILOGRAM), RelativityConvention.canonical()


def _geometry(scale, convention, shape=(2,)):
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    return ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id="line",
        geometry_lineage_id="flat-line",
    )


def _absorbing_interaction(system):
    return GRGreyRadiationInteractionPlan(
        system,
        ConstantGRGreyOpacityPlan(
            planck_absorption=1.0,
            planck_emission=0.0,
            rosseland_transport=1.0,
        ),
    )


def test_multigroup_and_neutrino_exchange_balance_energy_momentum_and_lepton_number():
    scale, convention = _contracts()
    geometry = _geometry(scale, convention)
    multigroup = GRMultigroupM1RadiationSystem(
        scale, convention, jnp.asarray((1.0, 2.0, 4.0))
    )
    interaction = GRMultigroupRadiationInteractionPlan(
        multigroup,
        tuple(_absorbing_interaction(group) for group in multigroup.groups),
    )
    moments = jnp.zeros((2, 2, 4)).at[..., 0].set(jnp.asarray(((2.0, 1.0), (1.5, 0.5))))
    exchange = interaction.matter_exchange(
        multigroup.flatten_groups(moments),
        jnp.ones(2),
        jnp.zeros((2, 3)),
        jnp.ones(2),
        geometry,
    )

    np.testing.assert_allclose(exchange.energy_balance_residual, 0.0)
    np.testing.assert_allclose(exchange.momentum_balance_residual, 0.0)
    assert bool(jnp.all(exchange.qualified))

    neutrinos = GRNeutrinoM1System(scale, convention, jnp.asarray((1.0, 2.0, 4.0)))
    species_interactions = tuple(
        GRMultigroupRadiationInteractionPlan(
            species,
            tuple(_absorbing_interaction(group) for group in species.groups),
        )
        for species in neutrinos.species_systems
    )
    neutrino_interaction = GRNeutrinoInteractionPlan(neutrinos, species_interactions)
    species_moments = jnp.zeros((2, 3, 2, 4))
    species_moments = species_moments.at[..., 0].set(1.0e-3)
    species_moments = species_moments.at[:, 0, :, 0].set(2.0)
    neutrino_exchange = neutrino_interaction.matter_exchange(
        neutrinos.flatten_moments(species_moments),
        jnp.ones(2),
        jnp.zeros((2, 3)),
        jnp.ones(2),
        2.0 * jnp.ones(2),
        0.4 * jnp.ones(2),
        geometry,
    )

    np.testing.assert_allclose(neutrino_exchange.energy_balance_residual, 0.0)
    np.testing.assert_allclose(neutrino_exchange.momentum_balance_residual, 0.0)
    np.testing.assert_allclose(neutrino_exchange.lepton_balance_residual, 0.0)
    assert bool(jnp.all(neutrino_exchange.electron_fraction_source > 0.0))
    assert bool(jnp.all(neutrino_exchange.qualified))


def test_vet_discrete_ordinates_and_monte_carlo_recover_isotropic_pressure():
    scale, convention = _contracts()
    geometry = _geometry(scale, convention)
    directions = jnp.asarray(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        )
    )
    discrete = DiscreteOrdinatesRadiationPlan(scale, convention, directions, jnp.ones(6))
    discrete_result = discrete.evaluate(3.0 * jnp.ones((2, 6)), geometry)

    np.testing.assert_allclose(discrete_result.energy_density, 3.0)
    np.testing.assert_allclose(discrete_result.flux_vector, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(
        discrete_result.pressure_tensor,
        jnp.broadcast_to(jnp.eye(3), (2, 3, 3)),
        atol=1.0e-7,
    )
    assert bool(jnp.all(discrete_result.qualified))

    vet = VariableEddingtonTensorClosurePlan(scale, convention)
    vet_result = vet.evaluate(
        3.0 * jnp.ones(2),
        jnp.zeros((2, 3)),
        jnp.broadcast_to(jnp.eye(3) / 3.0, (2, 3, 3)),
        geometry,
    )
    np.testing.assert_allclose(
        vet_result.pressure_tensor, discrete_result.pressure_tensor
    )
    assert bool(jnp.all(vet_result.qualified))

    monte_carlo = MonteCarloRadiationClosurePlan(scale, convention)
    packet_directions = jnp.broadcast_to(directions, (2, 6, 3))
    monte_carlo_result = monte_carlo.evaluate(
        0.5 * jnp.ones((2, 6)),
        packet_directions,
        jnp.ones(2),
        geometry,
    )
    np.testing.assert_allclose(
        monte_carlo_result.angular.pressure_tensor,
        discrete_result.pressure_tensor,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(monte_carlo_result.effective_packet_count, 6.0)
    assert bool(jnp.all(monte_carlo_result.qualified))
