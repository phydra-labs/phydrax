#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.operators.quantum._two_particle_green import (
    fermionic_crossing_evidence,
    FermionicTwoParticleChannelConvention,
    MatsubaraTwoParticleGreenFunction,
    rechannel_matsubara_two_particle,
    route_matsubara_two_particle,
)


def _crossing_closed_green():
    convention = FermionicTwoParticleChannelConvention("particle-hole-direct")
    labels = jnp.arange(-2, 3, dtype=jnp.int32)
    route = convention.route(
        labels[:, None, None], labels[None, :, None], labels[None, None, :]
    )
    values = (route[..., 0] - route[..., 2]) * (route[..., 1] - route[..., 3])
    return MatsubaraTwoParticleGreenFunction(
        convention,
        5.0,
        labels,
        labels,
        labels,
        values[..., None, None, None, None].astype(complex),
        fermion_label_minimum=-2,
        fermion_label_count=5,
    )


def test_creation_annihilation_order_channel_routes_and_wraps_are_explicit():
    direct = FermionicTwoParticleChannelConvention("particle-hole-direct")
    crossed = FermionicTwoParticleChannelConvention("particle-hole-crossed")
    pair = FermionicTwoParticleChannelConvention("particle-particle")

    assert direct.operator_order == (
        "annihilation",
        "creation",
        "annihilation",
        "creation",
    )
    np.testing.assert_array_equal(direct.route(2, -1, 1), [1, -1, 1, 3])
    np.testing.assert_array_equal(crossed.route(2, -1, 1), [3, -1, -3, 1])
    np.testing.assert_array_equal(pair.route(0, -1, 0), [-1, 0, 0, -1])

    routing = route_matsubara_two_particle(
        direct,
        2,
        -1,
        1,
        fermion_label_minimum=-2,
        fermion_label_count=4,
    )
    assert bool(routing.exact)
    np.testing.assert_array_equal(routing.wraps, [0, 0, 0, 1])
    np.testing.assert_array_equal(routing.wrapped_external_labels, [1, -1, 1, -1])


def test_crossing_and_cyclic_rechanneling_preserve_the_same_four_external_legs():
    green = _crossing_closed_green()
    evidence = fermionic_crossing_evidence(green)

    assert bool(evidence.satisfied)
    assert evidence.coverage_fraction > 0.0
    np.testing.assert_allclose(evidence.annihilation_exchange_residual, 0.0)
    np.testing.assert_allclose(evidence.creation_exchange_residual, 0.0)
    np.testing.assert_allclose(evidence.simultaneous_exchange_residual, 0.0)

    particle_particle = rechannel_matsubara_two_particle(
        green, FermionicTwoParticleChannelConvention("particle-particle")
    )
    round_trip = rechannel_matsubara_two_particle(
        particle_particle,
        FermionicTwoParticleChannelConvention("particle-hole-direct"),
    )
    np.testing.assert_array_equal(round_trip.values, green.values)


def test_two_particle_payload_is_rejected_before_oversized_allocation_contract():
    labels = jnp.arange(2)
    values = jnp.zeros((2, 2, 2, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="maximum_elements"):
        MatsubaraTwoParticleGreenFunction(
            FermionicTwoParticleChannelConvention("particle-hole-direct"),
            2.0,
            labels,
            labels,
            labels,
            values,
            maximum_elements=4,
        )
