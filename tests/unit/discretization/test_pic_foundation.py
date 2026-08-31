#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _pic_transfer(*, particle_count=2):
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count), jnp.ones((particle_count,)), ambient_dimension=3
    ).prepare()
    charged = phx.discretization.ChargedParticlePlan(
        -jnp.ones((particle_count,)), "electrons"
    ).prepare(particles)
    transfer = phx.discretization.pic.PICParticleCochainTransferPlan(bridge).prepare(charged)
    return bridge, charged, transfer


def test_charged_particles_reuse_stable_support_and_validate_specific_charge():
    _, charged, _ = _pic_transfer()
    assert charged.particles.capacity == 2
    np.testing.assert_allclose(charged.specific_charge, -1.0)
    assert charged.prepared_id

    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.asarray([1.0, 2.0]), ambient_dimension=3
    ).prepare()
    with pytest.raises(ValueError, match="specific charge"):
        phx.discretization.ChargedParticlePlan(
            jnp.asarray([-1.0, -1.0]), "invalid"
        ).prepare(particles)


def test_relativistic_boris_preserves_proper_speed_in_magnetic_field():
    pusher = phx.discretization.pic.RelativisticBorisPlan()
    initial = jnp.asarray([[0.2, -0.1, 0.05]])
    result = pusher.push(
        initial,
        jnp.zeros_like(initial),
        jnp.asarray([[0.0, 0.0, 0.8]]),
        jnp.asarray([-1.0]),
        jnp.asarray([True]),
        0.05,
    )
    assert result.successful
    np.testing.assert_allclose(
        jnp.sum(result.proper_velocity**2), jnp.sum(initial**2), rtol=2e-13, atol=2e-13
    )
    assert result.maximum_speed < 1.0


def test_pic_charge_and_field_transfer_preserve_layout_and_fixed_route_ad():
    bridge, charged, transfer = _pic_transfer()
    position = jnp.asarray([[0.2, 0.3, 0.4], [0.7, 0.6, 0.5]])
    routes = transfer.build(position)
    deposited = transfer.deposit_charge(routes)
    assert deposited.successful
    assert deposited.cochain.shape == (bridge.cochain.cell_counts[0],)
    np.testing.assert_allclose(
        jnp.sum(deposited.content), jnp.sum(charged.charges), atol=1e-13
    )

    electric_components = tuple(
        jnp.full(shape, axis + 1.0)
        for axis, shape in enumerate(bridge.orientation_shapes[1])
    )
    electric = bridge.pack_edge_circulation(electric_components)
    gathered = transfer.gather_electric(routes, electric)
    assert gathered.successful
    np.testing.assert_allclose(gathered.values, jnp.asarray([[1.0, 2.0, 3.0]] * 2))

    objective = lambda current: jnp.sum(
        transfer.deposit_charge(transfer.build(current)).density**2
    )
    gradient = jax.jit(jax.grad(objective))(position)
    assert jnp.all(jnp.isfinite(gradient))


def test_whitney_current_satisfies_continuity_across_cell_and_periodic_seam():
    bridge, _, transfer = _pic_transfer()
    current = phx.discretization.pic.ChargeConservingCurrentPlan(transfer)
    start = jnp.asarray([[0.24, 0.35, 0.45], [0.98, 0.65, 0.55]])
    end = jnp.asarray([[0.27, 0.34, 0.47], [1.01, 0.66, 0.54]])
    apply = jax.jit(lambda left, right, dt: current.deposit(left, right, dt))
    result = apply(start, end, jnp.asarray(0.02))
    assert result.successful
    assert not result.capacity_overflow
    assert result.current.shape == (bridge.cochain.cell_counts[1],)
    assert result.maximum_continuity_defect < 1e-9
