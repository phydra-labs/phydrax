#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import zipfile

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax.applications.cardiovascular.electrophysiology._activation import (
    activation_observation_result,
    ActivationObservationPlan,
    ChordConductionVelocityPlan,
    commit_activation_observation,
    evaluate_activation_observation,
    evaluate_chord_conduction_velocity,
    initialize_activation_observation,
)
from phydrax.applications.cardiovascular.electrophysiology._aliev_panfilov import (
    AlievPanfilovParameters,
    AlievPanfilovState,
    AlievPanfilovStatus,
    evaluate_aliev_panfilov,
)
from phydrax.applications.cardiovascular.electrophysiology._monodomain import (
    CellStimulusPulse,
    CellwiseDiffusivity,
    monodomain_state_identity,
    MonodomainStatus,
    PhenomenologicalMonodomainPlan,
    read_monodomain_checkpoint,
    run_monodomain_steps,
    write_monodomain_checkpoint,
)
from phydrax.discretization import (
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)


def _tetra_slab(cube_count: int = 2):
    coordinates = []
    for i in range(cube_count + 1):
        for j in range(2):
            for k in range(2):
                coordinates.append((float(i), float(j), float(k)))

    def vertex(i, j, k):
        return 4 * i + 2 * j + k

    tetrahedra = []
    for i in range(cube_count):
        v000 = vertex(i, 0, 0)
        v001 = vertex(i, 0, 1)
        v010 = vertex(i, 1, 0)
        v011 = vertex(i, 1, 1)
        v100 = vertex(i + 1, 0, 0)
        v101 = vertex(i + 1, 0, 1)
        v110 = vertex(i + 1, 1, 0)
        v111 = vertex(i + 1, 1, 1)
        tetrahedra.extend(
            (
                (v000, v100, v110, v111),
                (v000, v110, v010, v111),
                (v000, v010, v011, v111),
                (v000, v011, v001, v111),
                (v000, v001, v101, v111),
                (v000, v101, v100, v111),
            )
        )
    return jnp.asarray(coordinates), jnp.asarray(tetrahedra, dtype=jnp.int32)


def _fem(cube_count: int = 2):
    coordinates, tetrahedra = _tetra_slab(cube_count)
    mesh = CellMesh.from_tetrahedra(coordinates, tetrahedra)
    return FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("activation", lagrange_element("tetrahedron", 1)),
    ).prepare()


def _parameters(**updates):
    values = {
        "a": 0.05,
        "b": 0.15,
        "k": 8.0,
        "epsilon0": 0.002,
        "mu1": 0.2,
        "mu2": 0.3,
        "tau": 12.9,
    }
    values.update(updates)
    return AlievPanfilovParameters(
        values["a"],
        values["b"],
        values["k"],
        values["epsilon0"],
        values["mu1"],
        values["mu2"],
        values["tau"],
    )


def _runtime(
    *,
    cube_count: int = 2,
    dt_ms: float = 0.02,
    amplitude_per_ms: float = 1.0,
    stop_ms: float = 0.2,
):
    fem = _fem(cube_count)
    fibers = jnp.tile(jnp.asarray(((1.0, 0.0, 0.0),)), (6 * cube_count, 1))
    diffusivity = CellwiseDiffusivity.from_fibers(fibers, 0.2, 0.05)
    pulse = CellStimulusPulse(tuple(range(6)), 0.0, stop_ms, amplitude_per_ms)
    plan = PhenomenologicalMonodomainPlan(
        fem, diffusivity, _parameters(), pulses=(pulse,)
    )
    return plan.prepare(dt_ms)


def test_exact_dimensional_aliev_panfilov_rates_and_singularity_evidence():
    parameters = _parameters()
    state = AlievPanfilovState(jnp.asarray((0.2, 0.4)), jnp.asarray((0.1, 0.3)))
    source = jnp.asarray((0.01, -0.02))
    candidate = evaluate_aliev_panfilov(
        parameters, state, activation_source_per_ms=source
    )
    u = np.asarray(state.activation)
    r = np.asarray(state.recovery)
    expected_u = (
        parameters.k * u * (u - parameters.a) * (1.0 - u) - u * r
    ) / parameters.tau + np.asarray(source)
    expected_r = (
        (parameters.epsilon0 + parameters.mu1 * r / (u + parameters.mu2))
        * (-r - parameters.k * u * (u - parameters.b - 1.0))
        / parameters.tau
    )
    np.testing.assert_allclose(candidate.rates.activation_per_ms, expected_u)
    np.testing.assert_allclose(candidate.rates.recovery_per_ms, expected_r)
    assert bool(candidate.evidence.successful)

    singular = evaluate_aliev_panfilov(
        parameters,
        AlievPanfilovState(jnp.asarray((-parameters.mu2,)), jnp.asarray((0.1,))),
    )
    assert int(singular.evidence.singular_count) == 1
    assert int(singular.evidence.status) & int(
        AlievPanfilovStatus.RECOVERY_DENOMINATOR_SINGULAR
    )
    assert not bool(singular.evidence.successful)
    assert bool(jnp.all(singular.rates.activation_per_ms == 0.0))
    assert bool(jnp.all(singular.rates.recovery_per_ms == 0.0))


def test_diffusivity_is_fiber_sign_invariant_and_ids_cover_coefficients():
    fibers = jnp.asarray(((1.0, 2.0, 0.0), (-1.0, 0.0, 1.0)))
    positive = CellwiseDiffusivity.from_fibers(fibers, (0.2, 0.3), 0.05)
    reversed_ = CellwiseDiffusivity.from_fibers(-fibers, (0.2, 0.3), 0.05)
    changed = CellwiseDiffusivity.from_fibers(fibers, (0.21, 0.3), 0.05)
    np.testing.assert_array_equal(positive.tensor_mm2_per_ms, reversed_.tensor_mm2_per_ms)
    assert positive.diffusivity_id == reversed_.diffusivity_id
    assert positive.diffusivity_id != changed.diffusivity_id
    base_id = _parameters().parameter_id
    for name, value in (
        ("a", 0.051),
        ("b", 0.151),
        ("k", 8.1),
        ("epsilon0", 0.0021),
        ("mu1", 0.21),
        ("mu2", 0.31),
        ("tau", 13.0),
    ):
        assert base_id != _parameters(**{name: value}).parameter_id


def test_p1_row_sum_lumping_selected_cell_l2_projection_and_half_open_pulse():
    coordinates = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    mesh = CellMesh.from_tetrahedra(
        coordinates, jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)
    )
    fem = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("activation", lagrange_element("tetrahedron", 1)),
    ).prepare()
    zero_diffusion = CellwiseDiffusivity(jnp.zeros((1, 3, 3)))
    plan = PhenomenologicalMonodomainPlan(
        fem,
        zero_diffusion,
        _parameters(),
        pulses=(CellStimulusPulse((0,), 0.0, 0.1, 1.25),),
    )
    runtime = plan.prepare(0.1)
    np.testing.assert_allclose(runtime.lumped_mass, np.full(4, 1.0 / 24.0))
    np.testing.assert_allclose(runtime.stimulus_projection_per_ms[0], 1.25)

    at_start = runtime.initialize(jnp.zeros(4), jnp.zeros(4))
    proposed_start = runtime.evaluate(at_start)
    assert bool(proposed_start.evidence.successful)
    assert bool(jnp.all(runtime.split(proposed_start.proposed)[0] > 0.0))

    at_stop = runtime.initialize(jnp.zeros(4), jnp.zeros(4), time_ms=0.1, step_index=1)
    proposed_stop = runtime.evaluate(at_stop)
    assert bool(jnp.all(runtime.split(proposed_stop.proposed)[0] == 0.0))
    with pytest.raises(ValueError, match="aligned"):
        plan.prepare(0.06)


def test_diffusion_bound_and_stage_commit_fail_closed():
    runtime = _runtime(cube_count=1, dt_ms=0.02, stop_ms=0.2)
    diffusion_only = PhenomenologicalMonodomainPlan(
        runtime.plan.discretization,
        runtime.plan.diffusivity,
        runtime.plan.reaction,
    )
    with pytest.raises(ValueError, match="diffusion-only bound"):
        diffusion_only.prepare(runtime.diffusion_step_limit_ms * 1.01)

    singular_values = runtime.geometry.combine(
        (
            jnp.full((runtime.plan.node_count,), -runtime.plan.reaction.mu2),
            jnp.zeros((runtime.plan.node_count,)),
        )
    )
    singular_state = type(runtime.initialize(jnp.zeros(8), jnp.zeros(8)))(
        singular_values,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        runtime.runtime_id,
    )
    rejected = runtime.evaluate(singular_state)
    assert int(rejected.evidence.status) & int(MonodomainStatus.REACTION_STAGE_1_FAILURE)
    committed = runtime.commit(rejected, singular_state)
    np.testing.assert_array_equal(committed.values, singular_state.values)

    clean = runtime.initialize(jnp.zeros(8), jnp.zeros(8))
    candidate = runtime.evaluate(clean)
    different = runtime.initialize(jnp.full(8, 0.01), jnp.zeros(8))
    mismatched = runtime.commit(candidate, different)
    np.testing.assert_array_equal(mismatched.values, different.values)
    assert int(mismatched.step_index) == 0

    other_plan = PhenomenologicalMonodomainPlan(
        runtime.plan.discretization,
        runtime.plan.diffusivity,
        _parameters(mu1=0.21),
        pulses=runtime.plan.pulses,
    )
    other_runtime = other_plan.prepare(runtime.dt_ms)
    foreign = other_runtime.initialize(jnp.zeros(8), jnp.zeros(8))
    with pytest.raises(ValueError, match="does not match"):
        runtime.evaluate(foreign)
    foreign_candidate = other_runtime.evaluate(foreign)
    with pytest.raises(ValueError, match="must match"):
        runtime.commit(foreign_candidate, clean)


def test_branchwise_online_activation_and_directed_chord_velocity():
    plan = ActivationObservationPlan(4, (0, 2, 3), threshold=0.5)
    state = initialize_activation_observation(
        plan, jnp.asarray((0.1, 0.0, 0.7, 0.2)), time_ms=0.0
    )
    np.testing.assert_array_equal(state.activated, (False, True, False))
    first = evaluate_activation_observation(
        plan, state, jnp.asarray((0.6, 0.0, 0.8, 0.4)), 2.0
    )
    state = commit_activation_observation(first, state)
    second = evaluate_activation_observation(
        plan, state, jnp.asarray((0.9, 0.0, 0.75, 0.8)), 4.0
    )
    state = commit_activation_observation(second, state)
    result = activation_observation_result(plan, state)
    np.testing.assert_allclose(result.activation_times_ms, (1.6, 0.0, 2.5))
    np.testing.assert_array_equal(result.activated, (True, True, True))
    assert bool(result.successful)

    chord = ChordConductionVelocityPlan(plan, 2, 3, 1.5)
    velocity = evaluate_chord_conduction_velocity(chord, result)
    assert bool(velocity.successful)
    assert float(velocity.transit_time_ms) == pytest.approx(2.5)
    assert float(velocity.velocity_mm_per_ms) == pytest.approx(0.6)

    nonmonotone = evaluate_activation_observation(plan, state, jnp.ones(4), 4.0)
    preserved = commit_activation_observation(nonmonotone, state)
    assert not bool(nonmonotone.evidence.successful)
    np.testing.assert_array_equal(
        preserved.activation_times_ms, state.activation_times_ms
    )


def test_tetra_slab_propagates_and_checkpoint_restart_replays_identically(tmp_path):
    runtime = _runtime(cube_count=2, dt_ms=0.02, amplitude_per_ms=1.5, stop_ms=0.2)
    initial = runtime.initialize(
        jnp.zeros(runtime.plan.node_count), jnp.zeros(runtime.plan.node_count)
    )
    uninterrupted = run_monodomain_steps(runtime, initial, 20)
    assert bool(uninterrupted.successful)
    activation, _ = runtime.split(uninterrupted.state)
    assert float(jnp.max(activation[8:])) > 0.0

    prefix = run_monodomain_steps(runtime, initial, 7)
    checkpoint = tmp_path / "monodomain.phx"
    archive = write_monodomain_checkpoint(runtime, prefix.state, checkpoint)
    restored = read_monodomain_checkpoint(runtime, checkpoint)
    resumed = run_monodomain_steps(runtime, restored, 13)
    assert resumed.state_id == uninterrupted.state_id
    assert monodomain_state_identity(runtime, restored) == prefix.state_id
    np.testing.assert_array_equal(resumed.state.values, uninterrupted.state.values)
    assert archive.manifest.checkpoint_id

    other_runtime = runtime.plan.prepare(0.01)
    with pytest.raises(ValueError, match="does not match"):
        read_monodomain_checkpoint(other_runtime, checkpoint)

    with zipfile.ZipFile(checkpoint, "a") as container:
        container.writestr("unexpected-member", b"corruption")
    with pytest.raises(ArrayArchiveCorruptionError):
        read_monodomain_checkpoint(runtime, checkpoint)
