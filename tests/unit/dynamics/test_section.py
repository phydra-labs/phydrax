#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _oscillator_evolution():
    layout = phx.dynamics.StateLayout((2,), component_names=("x", "velocity"))
    system = phx.dynamics.ContinuousSystem(
        lambda time, state, args: jnp.asarray([state[1], -state[0]]),
        state_layout=layout,
        system_id="harmonic-oscillator",
    )
    return layout, phx.solver.DiffraxEvolution(
        system,
        rtol=1e-10,
        atol=1e-12,
        max_steps=4096,
    )


def test_evolution_refined_oriented_sections_and_return_map():
    layout, evolution = _oscillator_evolution()
    grid = phx.dynamics.TimeGrid(
        jnp.linspace(0.0, 5.0 * jnp.pi, 43), time_id="section-grid"
    )
    trajectory = phx.dynamics.evolve(evolution, jnp.asarray([1.0, 0.0]), grid)
    section = phx.dynamics.analysis.AffineSection(
        jnp.asarray([1.0, 0.0]),
        state_layout=layout,
    )

    crossings = phx.dynamics.analysis.find_section_crossings(
        trajectory,
        section,
        direction="positive",
        refinement="evolution",
        evolution=evolution,
        max_crossings=4,
        coordinate_tolerance=1e-11,
        section_tolerance=1e-10,
    )
    return_map = phx.dynamics.analysis.section_return_map(crossings)

    assert int(crossings.count) == 2
    assert not bool(crossings.overflow)
    np.testing.assert_allclose(
        np.asarray(crossings.coordinates[:2]),
        np.asarray([1.5 * np.pi, 3.5 * np.pi]),
        atol=2e-8,
    )
    np.testing.assert_allclose(
        np.asarray(crossings.states[:2]),
        np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        atol=2e-8,
    )
    assert bool(return_map.valid[0])
    np.testing.assert_allclose(
        np.asarray(return_map.return_intervals[0]), 2.0 * np.pi, atol=2e-8
    )


def test_callable_sections_preserve_case_axes_and_report_overflow():
    time = jnp.linspace(0.0, 6.0 * jnp.pi, 241)
    phases = jnp.asarray([0.0, 0.3])
    states = jnp.stack(
        (
            jnp.cos(time[None, :] + phases[:, None]),
            -jnp.sin(time[None, :] + phases[:, None]),
        ),
        axis=-1,
    )
    layout = phx.dynamics.StateLayout((2,), component_names=("x", "velocity"))
    data = phx.dynamics.TrajectoryData(
        jnp.broadcast_to(time, (2, time.size)),
        states,
        state_layout=layout,
        case_axes=("phase",),
        case_axis_roles=("parameter",),
        source_id="oscillator-cases",
    )
    section = phx.dynamics.analysis.CallableSection(
        lambda coordinate, state, args: state[0],
        state_layout=layout,
        section_id="x-zero",
    )

    crossings = phx.dynamics.analysis.find_section_crossings(
        data,
        section,
        direction="any",
        max_crossings=2,
        section_tolerance=1e-12,
    )

    assert crossings.coordinates.shape == (2, 2)
    np.testing.assert_array_equal(np.asarray(crossings.detected_count), [6, 6])
    np.testing.assert_array_equal(np.asarray(crossings.count), [2, 2])
    np.testing.assert_array_equal(np.asarray(crossings.overflow), [True, True])
    assert bool(jnp.all(jnp.abs(crossings.section_values) < 1e-9))
