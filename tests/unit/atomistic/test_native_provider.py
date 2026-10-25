#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native learned-model provider over its single neighborhood bundle."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests._support.atomistic_bundles import neighborhood_bundle_plan, role_capacity
from tests._support.mace_deployment import (
    bundle_resources,
    electronvolt_units,
    finite_frame,
    graph_execution,
    provider_plan,
    tiny_mace,
)


CELL = np.array([[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]])
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
MOVED = WATER + np.array([[0.02, -0.01, 0.0], [0.0, 0.015, -0.02], [0.0, 0.0, 0.01]])
STRAIN = np.eye(3) + 0.01 * np.array([[1.0, 0.3, 0.0], [0.3, -0.5, 0.2], [0.0, 0.2, 0.8]])


def _system(*, periodic: bool, cell: np.ndarray = CELL) -> Any:
    structure = phx.atomistic.AtomicStructure(
        jnp.asarray([8, 1, 1]),
        jnp.asarray(WATER),
        jnp.asarray([15.999, 1.008, 1.008]),
        electronvolt_units().scale,
        cell=jnp.asarray(cell) if periodic else None,
        periodic_axes=jnp.asarray([True, True, True]) if periodic else None,
    )
    return phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, electronvolt_units()
    ).prepare()


def _image_capacity(width: int, degree: int) -> Any:
    return phx.discretization.ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=width,
        maximum_degree=degree,
        maximum_images=343,
        maximum_case_edges=width,
    )


def _resources(*, grown: bool) -> dict[str, Any]:
    """A ladder whose first entries cannot hold the periodic water graph."""
    small = (_image_capacity(8, 8),), (role_capacity(8, 4),)
    large = (_image_capacity(1024, 64),), (role_capacity(1024, 64),)
    search, roles = (small[0] + large[0], small[1] + large[1]) if grown else small
    return {
        **bundle_resources(),
        "search_capacities": search,
        "role_capacities": roles,
    }


def test_periodic_provider_serves_energy_forces_and_current_cell_stress() -> None:
    model = tiny_mace(seed=3)
    provider = provider_plan(model).prepare(_system(periodic=True))
    assert provider.stress_available
    strained_cell = CELL @ STRAIN.T
    strained_positions = WATER @ STRAIN.T
    value = provider.evaluate_state(strained_positions, strained_cell, None)
    assert bool(value.evaluation.successful)
    assert (
        value.program.stress_convention
        is phx.atomistic.AtomisticStressConvention.CAUCHY_TENSION_POSITIVE
    )
    np.testing.assert_array_equal(value.request.cell_vectors, strained_cell)
    # The same physical surface prepared with the current cell as reference.
    reference = (
        provider_plan(model)
        .prepare(_system(periodic=True, cell=strained_cell))
        .evaluate_state(strained_positions, None, None)
    )
    np.testing.assert_allclose(
        value.evaluation.energy, reference.evaluation.energy, rtol=1e-10
    )
    np.testing.assert_allclose(
        value.evaluation.stress, reference.evaluation.stress, rtol=1e-8, atol=1e-11
    )
    step = 1.0e-5
    volume = abs(float(np.linalg.det(strained_cell)))
    for row, column in ((0, 0), (1, 2)):
        direction = np.zeros((3, 3))
        direction[row, column] = direction[column, row] = 1.0 if row == column else 0.5

        def energy(sign: float, direction: np.ndarray = direction) -> float:
            deformation = np.eye(3) + sign * step * direction
            return float(
                provider.evaluate_state(
                    strained_positions @ deformation.T,
                    strained_cell @ deformation.T,
                    None,
                ).evaluation.energy
            )

        derivative = (energy(1.0) - energy(-1.0)) / (2.0 * step)
        np.testing.assert_allclose(
            value.evaluation.stress[row, column],
            derivative / volume,
            rtol=1e-4,
            atol=1e-8,
        )
    shift = np.zeros_like(WATER)
    shift[1, 0] = step
    upper = provider.evaluate_state(strained_positions + shift, strained_cell, None)
    lower = provider.evaluate_state(strained_positions - shift, strained_cell, None)
    np.testing.assert_allclose(
        value.evaluation.forces[1, 0],
        -(float(upper.evaluation.energy) - float(lower.evaluation.energy)) / (2 * step),
        rtol=1e-4,
        atol=1e-8,
    )


def test_finite_provider_evaluates_in_its_explicit_frame() -> None:
    provider = provider_plan(tiny_mace(seed=4)).prepare(_system(periodic=False))
    assert not provider.stress_available
    value = provider.evaluate_state(WATER, None, None)
    assert bool(value.evaluation.successful) and value.evaluation.stress is None
    assert value.request.cell_vectors is None and value.request.image_counts is None
    shifted = provider.evaluate_state(WATER + 0.3, None, None)
    np.testing.assert_allclose(
        shifted.evaluation.energy, value.evaluation.energy, rtol=1e-10
    )
    with pytest.raises(ValueError, match="no cell vectors"):
        provider.evaluate_state(WATER, CELL, None)


def test_retained_state_reuses_its_epoch_and_rebuilds_beyond_the_skin() -> None:
    provider = provider_plan(tiny_mace()).prepare(_system(periodic=True))
    first = provider.evaluate_state(WATER, None, None)
    reused = provider.evaluate_state(MOVED, None, first.request.neighborhoods)
    np.testing.assert_array_equal(
        reused.request.neighborhoods.epochs, first.request.neighborhoods.epochs
    )
    np.testing.assert_allclose(
        reused.evaluation.energy,
        provider.evaluate_state(MOVED, None, None).evaluation.energy,
        rtol=1e-10,
    )
    far = WATER.copy()
    far[1, 0] += 0.6
    rebuilt = provider.evaluate_state(far, None, first.request.neighborhoods)
    assert bool(rebuilt.evaluation.successful)
    assert np.all(
        np.asarray(rebuilt.request.neighborhoods.epochs)
        > np.asarray(first.request.neighborhoods.epochs)
    )


def test_frozen_epoch_binding_never_rebuilds() -> None:
    provider = provider_plan(tiny_mace()).prepare(_system(periodic=True))
    request = provider.prepare_request(WATER, None, None)
    strained = provider.bind_request(
        MOVED @ STRAIN.T, CELL @ STRAIN.T, request.neighborhoods
    )
    refreshed = provider.prepare_request(
        MOVED @ STRAIN.T, CELL @ STRAIN.T, request.neighborhoods
    )
    assert bool(strained.neighborhoods.successful[0])
    np.testing.assert_allclose(
        provider.evaluate_request(strained).evaluation.energy,
        provider.evaluate_request(refreshed).evaluation.energy,
        rtol=1e-12,
    )
    far = WATER.copy()
    far[1, 0] += 0.6
    stale = provider.bind_request(far, None, request.neighborhoods)
    neighborhoods = stale.neighborhoods
    np.testing.assert_array_equal(neighborhoods.stale, [True])
    np.testing.assert_array_equal(neighborhoods.capacity_failure, [False])
    np.testing.assert_array_equal(neighborhoods.epochs, request.neighborhoods.epochs)
    assert not bool(provider.evaluate_request(stale).evaluation.successful)


def test_capacity_failure_is_recovered_on_the_declared_ladder() -> None:
    model = tiny_mace(seed=5)
    system = _system(periodic=True)
    provider = provider_plan(model, resources=_resources(grown=True)).prepare(system)
    failed = provider.evaluate_state(WATER, None, None)
    np.testing.assert_array_equal(failed.request.neighborhoods.capacity_only, [True])
    assert not bool(failed.evaluation.successful)
    recovery = provider.evaluate_recovering(WATER, None, None)
    assert recovery.capacity_growths >= 1 and not recovery.exhausted
    assert bool(recovery.evaluation.evaluation.successful)
    assert recovery.provider.bundle.prepared_id != provider.bundle.prepared_id
    assert recovery.provider.bundle.plan.plan_id == provider.bundle.plan.plan_id
    expected = provider_plan(model).prepare(system).evaluate_state(WATER, None, None)
    np.testing.assert_allclose(
        recovery.evaluation.evaluation.energy, expected.evaluation.energy, rtol=1e-12
    )
    # Recovery continues an accepted state of the grown resources.
    continued = recovery.provider.evaluate_recovering(
        MOVED, None, recovery.evaluation.request.neighborhoods
    )
    assert continued.capacity_growths == 0
    assert bool(continued.evaluation.evaluation.successful)

    exhausted = (
        provider_plan(model, resources=_resources(grown=False))
        .prepare(system)
        .evaluate_recovering(WATER, None, None)
    )
    assert exhausted.exhausted and exhausted.capacity_growths == 0
    assert not bool(exhausted.evaluation.evaluation.successful)
    # Scientific failures are returned unchanged, never retried as capacity.
    nonfinite = WATER.copy()
    nonfinite[0, 0] = np.nan
    scientific = provider.evaluate_recovering(nonfinite, None, None)
    assert scientific.capacity_growths == 0 and not scientific.exhausted
    assert not bool(scientific.evaluation.evaluation.successful)


def test_provider_plan_admits_only_explicit_complete_resources() -> None:
    model = tiny_mace()
    requirements = phx.atomistic.LearnedGraphPotentialTerm(
        model
    ).requirements.neighborhoods
    other = phx.atomistic.LearnedGraphPotentialTerm(
        model, name="other"
    ).requirements.neighborhoods
    periodic = neighborhood_bundle_plan(requirements, **bundle_resources())
    with pytest.raises(ValueError, match="explicitly"):
        phx.atomistic.NativeAtomisticProviderPlan(model, graph_execution())
    with pytest.raises(ValueError, match="cover exactly"):
        phx.atomistic.NativeAtomisticProviderPlan(
            model,
            graph_execution(),
            periodic_resources=neighborhood_bundle_plan(other, **bundle_resources()),
        )
    with pytest.raises(ValueError, match="frame"):
        phx.atomistic.NativeAtomisticProviderPlan(
            model,
            graph_execution(),
            periodic_resources=neighborhood_bundle_plan(
                requirements, frame=finite_frame(), **bundle_resources()
            ),
        )
    plan = phx.atomistic.NativeAtomisticProviderPlan(
        model, graph_execution(), periodic_resources=periodic
    )
    with pytest.raises(ValueError, match="no finite resources"):
        plan.prepare(_system(periodic=False))
    provider = plan.prepare(_system(periodic=True))
    assert provider.provider_id == plan.prepare(_system(periodic=True)).provider_id
    retrained = phx.atomistic.NativeAtomisticProviderPlan(
        tiny_mace(seed=9),
        graph_execution(),
        periodic_resources=neighborhood_bundle_plan(requirements, **bundle_resources()),
    )
    assert retrained.model_revision_id != plan.model_revision_id
    assert retrained.prepare(_system(periodic=True)).provider_id != provider.provider_id


def test_born_oppenheimer_dynamics_over_the_native_provider() -> None:
    system = _system(periodic=True)
    provider = provider_plan(tiny_mace()).prepare(system)
    plan = phx.atomistic.BornOppenheimerVelocityVerletPlan(system, provider, 0.1)
    state = plan.initialize(jnp.asarray(WATER), velocity=jnp.zeros_like(WATER))
    assert bool(state.successful)
    step = plan.step(state)
    assert bool(step.successful)
    np.testing.assert_allclose(
        step.final_evaluation.energy,
        provider.evaluate(system, step.state.positions, None).energy,
        rtol=1e-12,
    )
    assert jax.tree.structure(step.state) == jax.tree.structure(state)
