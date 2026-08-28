#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _particles(ids, dimension=1, *, active_mask=None):
    count = len(ids)
    masses = np.ones((count,))
    if active_mask is not None:
        masses = np.where(np.asarray(active_mask), masses, np.nan)
    return phx.discretization.ParticleSetPlan(
        ids,
        masses,
        ambient_dimension=dimension,
        active_mask=active_mask,
    ).prepare()


def _stable_pairs(state):
    valid = np.asarray(state.pair_relation.valid, dtype=bool)
    left = np.asarray(state.pair_relation.left_particle_ids)[valid]
    right = np.asarray(state.pair_relation.right_particle_ids)[valid]
    return set(zip(left.tolist(), right.tolist(), strict=True))


def test_cell_list_prepares_static_cell_and_neighbor_resources():
    particles = _particles(range(6))
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.CellListParticleNeighborhoodPlan(
        0.26,
        4,
        12,
        box,
    ).prepare(particles)

    assert prepared.cell_shape == (3,)
    assert jnp.all(prepared.cell_widths >= 0.26)
    assert prepared.cell_count == 3
    assert prepared.neighbor_cell_capacity == 3
    assert prepared.candidate_slot_count == 6 * 3 * 4
    assert prepared.pair_capacity == 12
    assert prepared.backend == "cell_edge_list"
    assert prepared.resource_evidence_id == prepared.preparation.report_id

    one_cell = phx.discretization.CellListParticleNeighborhoodPlan(
        2.0,
        6,
        15,
        box,
    ).prepare(particles)
    assert one_cell.cell_shape == (1,)
    assert one_cell.neighbor_cell_capacity == 1
    assert int(jnp.sum(one_cell.neighbor_cells.valid)) == 1


def test_cell_list_sorting_preserves_logical_identity_and_matches_brute_pairs():
    particles = _particles([40, 10, 30, 20])
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.CellListParticleNeighborhoodPlan(
        0.3,
        4,
        6,
        box,
    ).prepare(particles)
    position = jnp.asarray([[0.82], [0.08], [0.56], [0.31]])
    state = prepared.build(position)

    assert state.successful
    assert jnp.array_equal(state.storage_to_logical, jnp.asarray([1, 3, 2, 0]))
    assert jnp.array_equal(
        state.logical_to_storage[state.storage_to_logical], jnp.arange(4)
    )
    assert int(jnp.sum(state.cell_counts)) == 4
    assert _stable_pairs(state) == {(10, 20), (10, 40), (20, 30), (30, 40)}
    assert np.all(
        np.asarray(state.pair_relation.left_particle_ids)[
            np.asarray(state.pair_relation.valid)
        ]
        < np.asarray(state.pair_relation.right_particle_ids)[
            np.asarray(state.pair_relation.valid)
        ]
    )


def test_cell_list_handles_periodic_seams_and_nonperiodic_domain_status():
    particles = _particles([0, 1], dimension=2)
    box = phx.discretization.ParticleBox(
        [0.0, 0.0],
        [1.0, 1.0],
        periodic_axes=(True, False),
    )
    prepared = phx.discretization.CellListParticleNeighborhoodPlan(
        0.2,
        4,
        1,
        box,
    ).prepare(particles)

    seam = prepared.build(jnp.asarray([[0.05, 0.5], [0.95, 0.5]]))
    assert seam.successful
    assert int(seam.pair_count) == 1

    periodic_outside = prepared.build(jnp.asarray([[1.05, 0.5], [-0.05, 0.5]]))
    assert periodic_outside.successful
    assert int(periodic_outside.pair_count) == 1

    upper_boundary = prepared.build(jnp.asarray([[0.05, 1.0], [0.95, 0.5]]))
    assert upper_boundary.domain_violation
    assert int(upper_boundary.domain_violation_count) == 1
    assert not upper_boundary.successful


def test_cell_and_pair_overflow_are_independent_and_fail_closed():
    particles = _particles(range(4))
    box = phx.discretization.ParticleBox([0.0], [1.0])
    clustered = jnp.asarray([[0.10], [0.11], [0.12], [0.13]])

    cell_limited = phx.discretization.CellListParticleNeighborhoodPlan(
        0.25,
        2,
        6,
        box,
    ).prepare(particles)
    cell_state = cell_limited.build(clustered)
    assert cell_state.cell_overflow
    assert int(cell_state.cell_overflow_count) == 2
    assert not cell_state.pair_overflow
    assert not cell_state.successful
    assert jnp.all(jnp.isfinite(cell_state.pair_relation.left_particle_ids))

    pair_limited_plan = phx.discretization.CellListParticleNeighborhoodPlan(
        0.25,
        4,
        2,
        box,
    )
    pair_state = pair_limited_plan.prepare(particles).build(clustered)
    assert not pair_state.cell_overflow
    assert pair_state.pair_overflow
    assert int(pair_state.candidate_pair_count) == 6
    assert int(pair_state.pair_overflow_count) == 4
    assert int(pair_state.pair_count) == 2
    assert not pair_state.successful

    method = phx.discretization.BarotropicSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1),
        0.1,
    )
    compiled = phx.equations.compile_barotropic_sph_problem(
        phx.equations.BarotropicFluidProblemIR(
            "overflow-fluid",
            phx.equations.TaitBarotropicMaterial(1.0, 1.0),
        ),
        particles,
        method,
        neighborhood=pair_limited_plan,
    )
    with pytest.raises(Exception, match="neighborhood construction failed"):
        compiled.dynamics.density(clustered).block_until_ready()


def test_cell_list_enforces_candidate_resource_guard_before_runtime():
    particles = _particles(range(8), dimension=2)
    box = phx.discretization.ParticleBox([0.0, 0.0], [1.0, 1.0])
    plan = phx.discretization.CellListParticleNeighborhoodPlan(
        0.2,
        8,
        64,
        box,
        maximum_candidate_slots=100,
    )
    with pytest.raises(ValueError, match="candidate slots"):
        plan.prepare(particles)


def test_cell_list_runtime_build_is_filter_jittable():
    particles = _particles(range(8))
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.CellListParticleNeighborhoodPlan(
        0.3,
        4,
        24,
        box,
    ).prepare(particles)
    position = (jnp.arange(8, dtype=float) + 0.5)[:, None] / 8.0

    eager = prepared.build(position)
    compiled = eqx.filter_jit(prepared.build)(position)

    assert compiled.successful
    assert jnp.array_equal(compiled.pair_relation.valid, eager.pair_relation.valid)
    assert jnp.array_equal(
        compiled.pair_relation.left_particle_ids,
        eager.pair_relation.left_particle_ids,
    )
    assert jnp.array_equal(
        compiled.pair_relation.right_particle_ids,
        eager.pair_relation.right_particle_ids,
    )
