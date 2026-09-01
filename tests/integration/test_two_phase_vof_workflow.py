#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _two_phase(*, surface_tension=0.0, body=None):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=1000.0,
        gas_density=10.0,
        surface_tension=surface_tension,
    )
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        maximum_iterations=200,
    ).prepare()
    x = (jnp.arange(8) + 0.5) / 8
    y = (jnp.arange(8) + 0.5) / 8
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    alpha = jnp.where((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.2**2, 1.0, 0.0)
    velocity = (
        jnp.full(discretization.face_layouts[0].shape, 0.01),
        jnp.zeros(discretization.face_layouts[1].shape),
    )
    state = two_phase.initial_state(alpha, velocity)
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(
        two_phase, body=body
    )
    return two_phase, method, method.initial_continuation(state)


def test_plic_and_two_phase_state_are_bounded():
    two_phase, _, continuation = _two_phase()
    view = two_phase.view(continuation.state)

    assert bool(view.plic.valid)
    assert bool(view.topology.valid)
    assert jnp.all((view.alpha >= 0.0) & (view.alpha <= 1.0))
    assert jnp.all(view.density > 0.0)


def test_consistent_vof_step_preserves_phase_volume_and_divergence():
    two_phase, method, continuation = _two_phase()
    initial_volume = jnp.sum(continuation.state.liquid_content)

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.005),
        None,
    )
    final_volume = jnp.sum(result.accepted_state.state.liquid_content)

    assert bool(result.successful)
    np.testing.assert_allclose(final_volume, initial_volume, atol=1e-10)
    assert result.accepted_state.evidence.divergence_residual <= 1e-7
    assert result.accepted_state.evidence.alpha_minimum >= -1e-12
    assert result.accepted_state.evidence.alpha_maximum <= 1.0 + 1e-12


def test_balanced_capillarity_and_moving_body_are_finite():
    body = phx.applications.two_phase_flow.TwoPhaseMovingBodyPlan(
        (0.5, 0.5), 0.1, velocity=(0.0, 0.0), penalty=0.5
    )
    _, method, continuation = _two_phase(surface_tension=0.072, body=body)

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.001),
        None,
    )

    assert bool(result.successful)
    assert jnp.isfinite(result.accepted_state.ledger.capillary_work)
    assert jnp.isfinite(result.accepted_state.ledger.body_work)
    assert jnp.isfinite(result.accepted_state.evidence.clsvof_correction)


def test_two_phase_checkpoint_round_trip(tmp_path):
    two_phase, method, continuation = _two_phase()
    target = tmp_path / "two-phase.chk"

    phx.applications.two_phase_flow.write_two_phase_checkpoint(
        target,
        two_phase,
        method,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        continuation,
    )
    time, step, restored = phx.applications.two_phase_flow.read_two_phase_checkpoint(
        target, two_phase, method, continuation
    )

    np.testing.assert_allclose(time, 0.0)
    assert int(step) == 0
    np.testing.assert_allclose(
        restored.state.liquid_content,
        continuation.state.liquid_content,
    )
