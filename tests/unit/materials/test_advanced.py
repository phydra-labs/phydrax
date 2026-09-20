import jax.numpy as jnp

import phydrax as phx


def test_material_property_rve_and_transfer():
    table = phx.materials.TabulatedProperty((300.0, 400.0), (10.0, 20.0), "Pa")
    assert jnp.isclose(table.evaluate(350.0), 15.0)
    rve = phx.materials.RepresentativeVolumeElement.create((1.0, 3.0), (0, 1))
    assert jnp.isclose(rve.average(jnp.asarray((2.0, 4.0))), 3.5)
    matrix = jnp.asarray(((0.5, 0.5), (0.25, 0.75)))
    field = phx.materials.transfer_material_field(matrix, jnp.asarray((2.0, 4.0)))
    assert jnp.allclose(field, jnp.asarray((3.0, 3.5)))


def test_qmom_two_node_reproduces_moments():
    nodes = jnp.asarray((1.0, 3.0))
    weights = jnp.asarray((2.0, 1.0))
    moments = jnp.asarray([jnp.sum(weights * nodes**k) for k in range(4)])
    result = phx.population_balance.qmom_two_node(moments)
    assert bool(result.realizable)
    assert jnp.allclose(result.nodes, nodes)
    assert jnp.allclose(result.weights, weights)


def test_conservative_spatial_transfer_preserves_phase_inventory():
    state = phx.materials.MaterialState(
        jnp.asarray((300.0, 500.0)),
        jnp.asarray((1.0e5, 2.0e5)),
        jnp.asarray(((1.0, 0.0), (0.0, 1.0))),
    )
    field = phx.materials.SpatialMaterialField.create(
        jnp.asarray(((0.0,), (1.0,))), jnp.asarray((1.0, 1.0)), state
    )
    transfer = phx.materials.ConservativeMaterialTransfer.create(
        jnp.asarray(((0.75, 0.25), (0.25, 0.75))),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray((1.0, 1.0)),
    )
    target = field.transfer(jnp.asarray(((0.25,), (0.75,))), transfer)
    assert jnp.allclose(
        field.integral(state.phase_fractions),
        target.integral(target.state.phase_fractions),
    )
    assert jnp.allclose(target.state.temperature_k, jnp.asarray((350.0, 450.0)))


def test_spatial_icme_and_fe2_expose_balance_diagnostics():
    state = phx.materials.MaterialState(
        jnp.asarray((300.0, 400.0)),
        jnp.asarray((1.0e5, 1.0e5)),
        jnp.asarray(((1.0, 0.0), (0.5, 0.5))),
    )
    field = phx.materials.SpatialMaterialField.create(
        jnp.asarray(((0.0,), (1.0,))), jnp.asarray((1.0, 2.0)), state
    )
    model = phx.materials.SpatialICMEModel.create(
        jnp.asarray((100.0, 200.0)),
        jnp.asarray((1.0, 1.0)),
        homogenization="hill",
    )
    step = model.advance(
        field,
        jnp.asarray((350.0, 450.0)),
        jnp.asarray(((0.0, 1.0), (0.0, 1.0))),
        1.0,
    )
    assert bool(step.field.state.admissible)
    assert jnp.allclose(step.phase_balance_residual, 0, atol=1e-6)
    assert jnp.all(step.effective_properties > 100.0)

    fe2 = phx.materials.LinearFE2Plan.create(
        jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
        jnp.asarray((jnp.eye(2) * 2.0, jnp.eye(2) * 4.0)),
        jnp.asarray((1.0, 3.0)),
    ).evaluate(jnp.asarray((0.1, 0.2)))
    assert jnp.allclose(fe2.average_stress, jnp.asarray((0.35, 0.7)))
    assert jnp.allclose(fe2.consistent_tangent, jnp.eye(2) * 3.5)
    assert jnp.isclose(fe2.micro_residual_norm, 0)
