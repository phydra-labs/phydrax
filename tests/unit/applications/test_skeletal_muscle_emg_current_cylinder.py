#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import iv, ivp

from phydrax.applications.skeletal_muscle.electromyography import (
    Farina2004CylindricalConductorPlan,
    PereiraBotelho2019FiberCurrentPlan,
)
from phydrax.applications.skeletal_muscle.fibers import (
    PrescribedFiberStimulusSchedule,
    SkeletalFiberBundlePlan,
    SkeletalFiberBundleState,
)


def _source(nodes=17, *, radius=25.0e-6, radial_position=0.015):
    schedule = PrescribedFiberStimulusSchedule(
        jnp.zeros((0,)),
        jnp.zeros((0,)),
        jnp.zeros((0,)),
        jnp.zeros((0, 1, nodes), dtype=bool),
    )
    fiber = SkeletalFiberBundlePlan(("f0",), nodes, [40.0], [0.05], schedule).prepare()
    positions = jnp.zeros((1, nodes, 3)).at[..., 0].set(radial_position)
    positions = positions.at[0, :, 2].set(jnp.linspace(-0.02, 0.02, nodes))
    source = PereiraBotelho2019FiberCurrentPlan(
        ("f0",),
        positions,
        [radius],
        geometry_source_id="manufactured-straight-fiber",
        geometry_license="CC0-1.0 manufactured numerical fixture",
    ).prepare(fiber)
    state = fiber.initialize()
    voltage_mV = 40.0 * jnp.cos(jnp.linspace(0, 2 * jnp.pi, nodes))[None, :]
    state = SkeletalFiberBundleState(1.0, state.values.at[..., 0].set(voltage_mV))
    return fiber, source, state


def _observe(source, state):
    prior = source.initialize()
    candidate = source.propose(
        prior,
        state,
        fiber_prepared_id=source.fiber_prepared_id,
        geometry_id=source.plan.geometry_id,
    )
    assert bool(candidate.evidence.successful)
    return candidate.commit(prior, state)


def _cylinder(source, *, sigma=(0.2, 0.2, 0.2, 0.2), angular=2, axial=3, sizes=None):
    return Farina2004CylindricalConductorPlan(
        [0.03, 0.035, 0.04],
        sigma,
        [[0.0, -0.006], [0.1, 0.009]],
        [[0.002, 0.003], [0.002, 0.003]] if sizes is None else sizes,
        [[1.0, -1.0]],
        ("e0", "e1"),
        ("bipolar",),
        axial_period_m=0.2,
        longitudinal_modes=axial,
        angular_modes=angular,
        coordinate_frame_id="manufactured-cylinder-z",
        material_source_id="manufactured-homogeneous",
        electrode_source_id="manufactured-passive-rectangles",
    ).prepare(source, coordinate_frame_id="manufactured-cylinder-z")


def test_source_cosine_has_correct_sign_si_scale_and_sealed_terminal_balance():
    _, source, state = _source()
    accepted = _observe(source, state)
    z = np.asarray(source.plan.positions_m[0, :, 2])
    h = z[1] - z[0]
    w = 2 * np.pi / (z[-1] - z[0])
    sigma = float(source.intracellular_conductivity_S_per_m[0])
    area = np.pi * float(source.plan.radius_m[0]) ** 2
    voltage = np.asarray(accepted.membrane_voltage_V[0])
    # Exact discrete cosine eigenvalue including the half-width end volumes.
    expected_line = -sigma * area * 4 * np.sin(w * h / 2) ** 2 / h**2 * voltage
    np.testing.assert_allclose(
        accepted.transmembrane_line_current_A_per_m[0],
        expected_line,
        rtol=2e-5,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.sum(accepted.transmembrane_current_A, axis=-1), 0.0, atol=1e-14
    )
    np.testing.assert_allclose(
        accepted.transmembrane_line_current_A_per_m * source.control_length_m,
        accepted.transmembrane_current_A,
        rtol=1e-6,
    )


def test_current_transaction_rejects_nonfinite_foreign_and_stale_voltage():
    _, source, state = _source()
    prior = source.initialize()
    candidate = source.propose(
        prior,
        state,
        fiber_prepared_id=source.fiber_prepared_id,
        geometry_id=source.plan.geometry_id,
    )
    changed = SkeletalFiberBundleState(state.time_ms, state.values.at[0, 3, 0].add(1.0))
    rejected = candidate.commit(prior, changed)
    np.testing.assert_array_equal(
        rejected.transmembrane_current_A, prior.transmembrane_current_A
    )
    assert int(rejected.accepted_observations) == 0
    bad = SkeletalFiberBundleState(state.time_ms, state.values.at[0, 2, 0].set(jnp.nan))
    for input_state, geometry in (
        (bad, source.plan.geometry_id),
        (state, "foreign-geometry"),
    ):
        proposal = source.propose(
            prior,
            input_state,
            fiber_prepared_id=source.fiber_prepared_id,
            geometry_id=geometry,
        )
        assert not bool(proposal.evidence.successful)
        rollback = proposal.commit(prior, input_state)
        np.testing.assert_array_equal(
            rollback.transmembrane_current_A, prior.transmembrane_current_A
        )
        assert int(rollback.accepted_observations) == 0
    committed = candidate.commit(prior, state)
    replay = candidate.commit(committed, state)
    assert int(replay.accepted_observations) == 1


def test_source_metric_and_duplicate_ownership_are_not_inferred():
    fiber, source, _ = _source()
    with pytest.raises(ValueError, match="monodomain metric"):
        PereiraBotelho2019FiberCurrentPlan(
            ("f0",),
            source.plan.positions_m * 2,
            source.plan.radius_m,
            geometry_source_id="scaled",
            geometry_license="CC0-1.0",
        ).prepare(fiber)
    with pytest.raises(ValueError, match="one owner"):
        PereiraBotelho2019FiberCurrentPlan(
            ("f0", "f0"),
            jnp.repeat(source.plan.positions_m, 2, axis=0),
            [1e-5, 1e-5],
            geometry_source_id="duplicated",
            geometry_license="CC0-1.0",
        )


def test_cylinder_matches_homogeneous_neumann_green_function_and_aperture():
    _, source, state = _source()
    cylinder = _cylinder(source)
    n = np.arange(-2, 3)[:, None]
    k = (2 * np.pi / 0.2 * np.r_[np.arange(-3, 0), np.arange(1, 4)])[None, :]
    r, outer, sigma = 0.015, 0.04, 0.2
    q = np.abs(k)
    # Wronskian identity gives the insulated homogeneous cylinder surface Green function.
    green = iv(np.abs(n), q * r) / (sigma * q * outer * ivp(np.abs(n), q * outer))
    np.testing.assert_allclose(
        cylinder.radial_transfer_ohm_m[0], green[2:, 3:], rtol=3e-5, atol=1e-9
    )
    accepted = _observe(source, state)
    expected = []
    for theta_e, z_e in np.asarray(cylinder.plan.electrode_centers):
        aperture = np.sinc(n * 0.002 / (2 * np.pi * outer)) * np.sinc(
            k * 0.003 / (2 * np.pi)
        )
        value = 0.0
        for z, current in zip(
            np.asarray(source.plan.positions_m[0, :, 2]),
            np.asarray(accepted.transmembrane_current_A[0]),
        ):
            value += (
                current
                * np.sum(green * aperture * np.cos(n * theta_e + k * (z_e - z)))
                / (2 * np.pi * 0.2)
            )
        expected.append(value)
    prior = cylinder.initialize()
    candidate = cylinder.propose(prior, accepted)
    assert bool(candidate.evidence.successful)
    output = candidate.commit(prior, accepted)
    np.testing.assert_allclose(
        output.contact_potential_V, expected, rtol=3e-5, atol=1e-12
    )
    np.testing.assert_allclose(
        output.lead_voltage_V[0], expected[0] - expected[1], rtol=3e-5, atol=1e-12
    )


def test_cylinder_rejects_non_neutral_current_and_stale_geometry():
    _, source, state = _source()
    cylinder = _cylinder(source)
    accepted = _observe(source, state)
    prior = cylinder.initialize()
    nonneutral = eqx.tree_at(
        lambda x: x.transmembrane_current_A,
        accepted,
        accepted.transmembrane_current_A.at[0, 0].add(1e-5),
    )
    bad = cylinder.propose(prior, nonneutral)
    assert not bool(bad.evidence.successful)
    np.testing.assert_array_equal(
        bad.commit(prior, nonneutral).lead_voltage_V, prior.lead_voltage_V
    )
    candidate = cylinder.propose(prior, accepted)
    changed = eqx.tree_at(
        lambda x: x.accepted_observations, accepted, accepted.accepted_observations + 1
    )
    assert int(candidate.commit(prior, changed).accepted_observations) == 0
    with pytest.raises(ValueError, match="registered frame"):
        cylinder.plan.prepare(source, coordinate_frame_id="new-committed-geometry-frame")


def test_cylinder_runtime_jit_and_aperture_suppression():
    _, source, state = _source()
    cylinder = _cylinder(source)
    accepted = _observe(source, state)
    prior = cylinder.initialize()
    plain = cylinder.propose(prior, accepted)
    compiled = eqx.filter_jit(cylinder.propose)(prior, accepted)
    np.testing.assert_allclose(
        compiled.candidate_state.lead_voltage_V, plain.candidate_state.lead_voltage_V
    )
    assert bool(compiled.evidence.successful)
    wider = _cylinder(source, sizes=[[0.002, 0.1], [0.002, 0.1]])
    assert float(jnp.max(jnp.abs(wider.contact_lead_field_ohm))) < float(
        jnp.max(jnp.abs(cylinder.contact_lead_field_ohm))
    )


def test_cylinder_rejects_collinear_fiber_foldback():
    fiber, source, _ = _source(nodes=3)
    folded = source.plan.positions_m.at[0, :, 2].set(jnp.asarray([0.0, 0.02, 0.0]))
    folded_source = PereiraBotelho2019FiberCurrentPlan(
        ("f0",),
        folded,
        source.plan.radius_m,
        geometry_source_id="manufactured-foldback",
        geometry_license="CC0-1.0",
    ).prepare(fiber)
    with pytest.raises(ValueError, match="foldback"):
        _cylinder(folded_source)
