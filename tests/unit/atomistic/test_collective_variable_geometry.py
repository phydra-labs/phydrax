"""Metric, image-branch and alignment controls for native collective variables."""

from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.atomistic import PreparedAtomisticSystem
from phydrax.atomistic._dynamics import VelocityVerletPlan
from phydrax.atomistic._topology import MolecularTopologyPlan
from phydrax.atomistic.sampling import (
    aligned_deviation,
    CollectiveVariableGeometry,
    CollectiveVariableImageBranch,
    CollectiveVariableKind,
    CollectiveVariablePlan,
    CollectiveVariableProgram,
    CollectiveVariableStatus,
    HarmonicBiasForm,
)
from tests._support.atomistic_dynamics_fixtures import atomistic_system, periodic_cell
from tests._support.atomistic_sampling_fixtures import (
    bias_term,
    biased_dynamics,
    cv_program,
    kabsch_msd,
    nve_table,
    start,
)


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


def test_periodic_torsion_bias_uses_the_metric_across_the_seam() -> None:
    # Baseline defect 1: raw subtraction gave 38.23 instead of 0.01.
    system = atomistic_system([0, 1, 2, 3], [1.0] * 4)
    phi = -math.pi + 0.05
    positions = jnp.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [math.cos(phi), math.sin(phi), 1.0],
        ]
    )
    variables = cv_program(system, CollectiveVariableKind.TORSION, [0, 1, 2, 3])
    form = HarmonicBiasForm(np.asarray([math.pi - 0.05]), 2.0)
    term = bias_term(system, variables, form).prepare(system)
    energy, evaluation, ok = term.bias(
        CollectiveVariableGeometry(positions), term.initial_term_state(jnp.float64)
    )
    np.testing.assert_allclose(float(evaluation.values[0]), phi, atol=1e-12)
    np.testing.assert_allclose(float(energy), 0.5 * 2.0 * 0.1**2, rtol=1e-10)
    assert bool(ok)


def _straddling_system(
    *, topology: MolecularTopologyPlan | None = None
) -> PreparedAtomisticSystem:
    return atomistic_system(
        [0, 1, 2],
        [1.0, 1.0, 1.0],
        cell=periodic_cell(10.0 * np.eye(3)),
        topology=topology,
    )


def test_group_cvs_use_the_unwrapped_molecular_branch() -> None:
    # Baseline defect 2: the wrapped pair has COM x = 5 (distance 2 from atom 2,
    # expected 3) and Rg 4.8 (expected 0.2).  Atom 2 sits off the half-box
    # separation: a COM separation of exactly L/2 is image-ambiguous and refused.
    system = _straddling_system()
    wrapped = jnp.asarray([[9.8, 5.0, 5.0], [0.2, 5.0, 5.0], [7.0, 5.0, 5.0]])
    unwrapped = wrapped.at[1, 0].set(10.2)
    geometry = CollectiveVariableGeometry(
        wrapped, unwrapped_positions=unwrapped, cell=system.cell
    )
    com = CollectiveVariablePlan(
        CollectiveVariableKind.CENTER_OF_MASS_DISTANCE, [0, 1, 2], parameters=[2]
    ).prepare(system)
    gyration = CollectiveVariablePlan(
        CollectiveVariableKind.RADIUS_OF_GYRATION, [0, 1]
    ).prepare(system)
    evaluations = ((com.evaluate(geometry), 3.0), (gyration.evaluate(geometry), 0.2))
    for evaluation, expected in evaluations:
        assert bool(evaluation.successful)
        np.testing.assert_allclose(float(evaluation.value), expected, atol=1e-12)
    half_box = CollectiveVariableGeometry(
        wrapped.at[2, 0].set(5.0),
        unwrapped_positions=unwrapped.at[2, 0].set(5.0),
        cell=system.cell,
    )
    ambiguous = com.evaluate(half_box)
    assert not bool(ambiguous.successful)
    assert int(ambiguous.status) & int(CollectiveVariableStatus.IMAGE_AMBIGUOUS)
    blind = CollectiveVariableGeometry(wrapped, cell=system.cell)
    for refused in (com.evaluate(blind), gyration.evaluate(blind)):
        assert not bool(refused.successful)
        assert int(refused.status) & int(CollectiveVariableStatus.BRANCH_UNCERTIFIED)


def test_runtime_bias_binds_the_accepted_unwrapped_branch() -> None:
    # Baseline defect 2 through the dynamics program: the term context must carry
    # the accepted unwrapped coordinates of a pair straddling the boundary.
    system = _straddling_system()
    program = CollectiveVariableProgram(
        (
            CollectiveVariablePlan(
                CollectiveVariableKind.CENTER_OF_MASS_DISTANCE, [0, 1, 2], parameters=[2]
            ).prepare(system),
            CollectiveVariablePlan(
                CollectiveVariableKind.RADIUS_OF_GYRATION, [0, 1]
            ).prepare(system),
        )
    )
    term = bias_term(system, program, HarmonicBiasForm(np.asarray([4.0, 0.0]), 2.0))
    dynamics = biased_dynamics(system, term, VelocityVerletPlan(1.0e-3))
    unwrapped = jnp.asarray([[9.8, 5.0, 5.0], [10.2, 5.0, 5.0], [7.0, 5.0, 5.0]])
    state = start(dynamics, nve_table(dynamics), unwrapped)
    # The accepted state really is wrapped across the boundary.
    np.testing.assert_allclose(float(state.kinematics.positions[1, 0]), 0.2, atol=1e-12)
    # COM of {9.8, 10.2} is 10, distance 3; Rg 0.2:  0.5 * 2 * ((3 - 4)^2 + 0.2^2).
    np.testing.assert_allclose(
        float(state.force.potential_energy), 0.5 * 2.0 * (1.0**2 + 0.2**2), rtol=1e-12
    )


def test_topology_branch_rebuilds_bonds_and_refuses_winding_closures() -> None:
    topology = MolecularTopologyPlan(bonds=[[0, 1]], bond_type_ids=[0])
    system = _straddling_system(topology=topology)
    wrapped = jnp.asarray([[9.8, 5.0, 5.0], [0.2, 5.0, 5.0], [5.0, 5.0, 5.0]])
    gyration = CollectiveVariablePlan(
        CollectiveVariableKind.RADIUS_OF_GYRATION,
        [0, 1],
        branch=CollectiveVariableImageBranch.TOPOLOGY,
    ).prepare(system)
    value = gyration.evaluate(CollectiveVariableGeometry(wrapped, cell=system.cell))
    assert bool(value.successful)
    np.testing.assert_allclose(float(value.value), 0.2, atol=1e-12)
    with pytest.raises(ValueError, match="bond-connected"):
        CollectiveVariablePlan(
            CollectiveVariableKind.RADIUS_OF_GYRATION,
            [0, 1, 2],
            branch=CollectiveVariableImageBranch.TOPOLOGY,
        ).prepare(system)


def test_triclinic_minimum_image_matches_exhaustive_enumeration() -> None:
    vectors = np.asarray([[4.0, 0.0, 0.0], [1.7, 3.6, 0.0], [-1.1, 0.9, 3.8]])
    system = atomistic_system([0, 1], [1.0, 1.0], cell=periodic_cell(vectors))
    rng = np.random.default_rng(3)
    geometry = CollectiveVariableGeometry(
        jnp.zeros((2, 3)), unwrapped_positions=jnp.zeros((2, 3)), cell=system.cell
    )
    for _ in range(20):
        displacement = rng.uniform(-6.0, 6.0, size=3)
        image, margin = geometry.minimum_image(jnp.asarray(displacement))
        candidates = [
            displacement + np.asarray(shift) @ vectors
            for shift in product(range(-3, 4), repeat=3)
        ]
        shortest = min(np.linalg.norm(candidate) for candidate in candidates)
        np.testing.assert_allclose(float(jnp.linalg.norm(image)), shortest, atol=1e-12)
        assert np.isfinite(float(margin))


def test_aligned_msd_matches_independent_kabsch_and_keeps_regularity() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(6, 3))
    moved = reference @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T + 0.05 * rng.normal(
        size=(6, 3)
    )
    evaluation = aligned_deviation(jnp.asarray(moved), jnp.asarray(reference))
    np.testing.assert_allclose(
        float(evaluation.msd), kabsch_msd(moved, reference), rtol=1e-10
    )
    assert bool(evaluation.msd_regular) and bool(evaluation.rmsd_regular)
    gradient = jax.grad(lambda x: aligned_deviation(x, jnp.asarray(reference)).msd)(
        jnp.asarray(moved)
    )
    step = 1e-6
    direction = rng.normal(size=moved.shape)
    finite = (
        kabsch_msd(moved + step * direction, reference)
        - kabsch_msd(moved - step * direction, reference)
    ) / (2.0 * step)
    np.testing.assert_allclose(float(jnp.sum(gradient * direction)), finite, rtol=1e-6)
    hessian = jax.hessian(lambda x: aligned_deviation(x, jnp.asarray(reference)).msd)(
        jnp.asarray(moved)
    )
    assert bool(jnp.all(jnp.isfinite(hessian)))
    zero = aligned_deviation(jnp.asarray(reference), jnp.asarray(reference))
    assert float(zero.rmsd) == 0.0
    assert bool(zero.msd_regular) and not bool(zero.rmsd_regular)
    assert int(zero.status) & int(CollectiveVariableStatus.ZERO_DISTANCE)


def test_zero_length_distance_is_reported_not_differentiated() -> None:
    system = atomistic_system([0, 1], [1.0, 1.0])
    distance = CollectiveVariablePlan(CollectiveVariableKind.DISTANCE, [0, 1]).prepare(
        system
    )
    result = distance.evaluate(CollectiveVariableGeometry(jnp.zeros((2, 3))))
    assert not bool(result.regular)
    assert int(result.status) & int(CollectiveVariableStatus.ZERO_LENGTH)
    gradient = jax.grad(lambda x: distance.evaluate(CollectiveVariableGeometry(x)).value)(
        jnp.zeros((2, 3))
    )
    assert bool(jnp.all(jnp.isfinite(gradient)))
