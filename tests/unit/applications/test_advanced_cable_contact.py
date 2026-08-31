from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax as phx


mn = phx.applications.solid_mechanics.member_network


def test_generalized_channels_and_orientation_transport():
    layout = mn.GeneralizedDOFLayout(
        (
            mn.GeneralizedDOFChannel(
                "warping", (3,), constrained=jnp.asarray((True, False, False))
            ),
            mn.GeneralizedDOFChannel("mode", (2,)),
        )
    )
    reduced = layout.reduce(
        {"warping": jnp.asarray((0.0, 1.0, 2.0)), "mode": jnp.asarray((3.0, 4.0))}
    )
    expanded = layout.expand(
        reduced,
        {"warping": jnp.asarray((0.0,)), "mode": jnp.empty((0,))},
    )
    assert jnp.allclose(expanded.channel("warping"), jnp.asarray((0.0, 1.0, 2.0)))

    points = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 1.0, 0.0)))
    field, evidence = mn.parallel_transport_orientations(
        points, jnp.asarray((0.0, 0.0, 1.0))
    )
    assert evidence is None
    assert field.frames.shape == (2, 3, 3)
    assert jnp.all(field.director_margin > 0.0)


def test_elastic_catenary_slack_straight_and_loaded_regimes():
    slack_reference = mn.ElasticCatenaryReference(1.0, 100.0, jnp.zeros((3,)))
    slack = mn.solve_elastic_catenary(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((0.9, 0.0, 0.0)),
        slack_reference,
    )
    assert slack.regime == int(mn.CatenaryRegime.SLACK)
    assert slack.minimum_tension == pytest.approx(0.0)

    taut = mn.solve_elastic_catenary(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((1.1, 0.0, 0.0)),
        slack_reference,
    )
    assert taut.valid
    assert taut.regime == int(mn.CatenaryRegime.ZERO_DISTRIBUTED_LOAD)
    assert taut.minimum_tension == pytest.approx(10.0, rel=1.0e-5)

    loaded_reference = mn.ElasticCatenaryReference(
        1.1, 100.0, jnp.asarray((0.0, -1.0, 0.0))
    )
    loaded = mn.solve_elastic_catenary(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        loaded_reference,
    )
    assert loaded.valid
    assert loaded.sag > 0.0
    assert loaded.minimum_tension > 0.0
    assert (
        jnp.linalg.norm(
            loaded.start_force
            + loaded.end_force
            - 1.1 * loaded_reference.distributed_load
        )
        <= 1.0e-6
    )


def test_contact_friction_saddle_and_connection_energy():
    contact = mn.NodePlaneContact(
        (0,),
        jnp.asarray(((0.0, 0.0, 0.0),)),
        jnp.asarray(((0.0, 1.0, 0.0),)),
        friction_coefficient=jnp.asarray((0.5,)),
    )
    state = mn.evaluate_node_plane_contact(
        contact,
        jnp.asarray(((0.0, -0.1, 0.0),)),
        jnp.asarray((10.0,)),
        jnp.asarray(((3.0, 0.0, 0.0),)),
    )
    assert state.active[0]
    assert state.sticking[0]
    assert state.gap[0] == pytest.approx(-0.1)

    saddle = mn.CableSaddleContact(0, 1, 1, friction_coefficient=0.2)
    saddle_state = mn.evaluate_cable_saddle(
        saddle,
        jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))),
        0,
        2,
        10.0,
        10.0,
    )
    assert saddle_state.wrap_angle == pytest.approx(jnp.pi / 2.0)
    assert not saddle_state.sliding

    block = mn.LinearConnectionSpringBlock(
        ((0, 1),),
        jnp.asarray(((10.0, 10.0),)),
        jnp.asarray(((5.0,),)),
    )
    assert block.translation_stiffness[0, 0] == pytest.approx(10.0)
