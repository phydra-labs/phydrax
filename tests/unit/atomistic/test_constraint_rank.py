#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Constraint rank admission, refusal of rank-changing geometries and DOF counting."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.atomistic._constraints import (
    DistanceConstraintPlan,
    PreparedDistanceConstraints,
)
from phydrax.atomistic._dynamics import (
    AtomisticStepRejectionReason,
    PreparedAtomisticDynamics,
    VelocityVerletPlan,
)
from phydrax.atomistic._system import PreparedAtomisticSystem
from phydrax.atomistic._thermal import NoseHooverChainParameters, NoseHooverChainPlan
from phydrax.atomistic._thermodynamic import PreparedThermodynamicStateTable
from phydrax.atomistic._topology import MolecularTopologyPlan
from tests._support.atomistic_dynamics_fixtures import (
    assert_tree_equal,
    atomistic_system,
    ExternalFieldTerm,
    measure,
    prepare_dynamics,
    row,
    step_detailed,
    table,
)


# Water-like O-H-H triangle (O mass 16) with all three distances constrained.
# Binary-exact geometry: |OH| = 1.25 (0.75^2 + 1^2 = 1.5625), |HH| = 1.5, so
# every squared distance equals its squared target exactly and the initial
# projection moves nothing.
WATER = np.asarray([[0.75, 1.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
WATER_MASSES = np.asarray([16.0, 1.0, 1.0])
WATER_PAIRS = [[0, 1], [0, 2], [1, 2]]
COLLINEAR = np.asarray([[0.75, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
SQUARE = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
TETRAHEDRON = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
INADMISSIBLE = (eqx.EquinoxRuntimeError, jax.errors.JaxRuntimeError)


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


def _distances(positions: np.ndarray, pairs: list[list[int]]) -> list[float]:
    return [float(np.linalg.norm(positions[i] - positions[j])) for i, j in pairs]


def _rigid_system(
    positions: np.ndarray, masses: np.ndarray, pairs: list[list[int]]
) -> PreparedAtomisticSystem:
    topology = MolecularTopologyPlan(
        constraints=pairs, constraint_distances=_distances(positions, pairs)
    )
    return atomistic_system(list(range(positions.shape[0])), masses, topology=topology)


def _water(
    *, field: str, step_size: float, gradient: np.ndarray | None = None
) -> tuple[
    PreparedAtomisticSystem,
    PreparedDistanceConstraints,
    PreparedAtomisticDynamics,
    PreparedThermodynamicStateTable,
]:
    system = _rigid_system(WATER, WATER_MASSES, WATER_PAIRS)
    constraints = DistanceConstraintPlan(
        maximum_iterations=64, tolerance=1.0e-10
    ).prepare(system)
    coefficients = np.zeros((3, 3)) if gradient is None else gradient
    dynamics = prepare_dynamics(
        system,
        (ExternalFieldTerm("field", field, coefficients),),
        VelocityVerletPlan(step_size),
        constraints=constraints,
    )
    thermodynamic = table(dynamics, row(measure(system), "nve"))
    return system, constraints, dynamics, thermodynamic


def test_water_triangle_is_regular_and_steps_on_its_manifold() -> None:
    _, constraints, dynamics, thermodynamic = _water(field="zero", step_size=0.01)
    assert constraints.independent_constraint_count == 3
    evidence = constraints.rank_evidence(jnp.asarray(WATER))
    assert bool(evidence.regular)
    np.testing.assert_array_equal(evidence.component_rank, [3])
    # Host SVD of the mass-weighted Jacobian of this triangle gives 0.4790048.
    assert float(evidence.minimum_relative_singular_value) == pytest.approx(
        0.4790048, rel=1e-6
    )
    velocity = np.random.default_rng(61).normal(size=(3, 3))
    state = dynamics.initialize_state(
        jnp.asarray(WATER), thermodynamic, velocity=jnp.asarray(velocity), key=jr.key(61)
    )
    evaluation = step_detailed(dynamics, state, thermodynamic)
    assert bool(evaluation.successful)
    after = evaluation.accepted_state
    # The plan tolerance bounds both residuals of the accepted projection.
    assert float(after.constraint_position_residual) <= 1.0e-10
    assert float(after.constraint_velocity_residual) <= 1.0e-10
    assert bool(constraints.rank_evidence(after.kinematics.positions).regular)


def test_collinear_geometry_is_irregular_and_a_step_reaching_it_is_rejected() -> None:
    _, constraints, _, _ = _water(field="zero", step_size=1.0)
    evidence = constraints.rank_evidence(jnp.asarray(COLLINEAR))
    assert not bool(evidence.regular)
    # Collinear gradients: d(OH1) - d(OH2) is parallel to d(H1H2), rank 2 of 3.
    np.testing.assert_array_equal(evidence.component_rank, [2])

    # A uniform field on O only: E = 32 y_O, F_O = -32 e_y.  With dt = 1 and
    # m_O = 16 the half kick gives p_O = -16 e_y and the drift moves O exactly
    # from y = 1 to y = 0 (all products are exact binary fractions), so the
    # proposal is collinear, |OH| = 0.75 cannot reach 1.25 on the line, and
    # SHAKE corrections (parallel to the displacements) stay on it.
    gradient = np.zeros((3, 3))
    gradient[0, 1] = 32.0
    _, constraints, dynamics, thermodynamic = _water(
        field="linear", step_size=1.0, gradient=gradient
    )
    units = dynamics.system.plan.units
    assert units.kinetic_to_energy == 1.0
    state = dynamics.initialize_state(
        jnp.asarray(WATER),
        thermodynamic,
        velocity=jnp.zeros((3, 3)),
        key=jr.key(62),
    )
    evaluation = dynamics.step_detailed(state, thermodynamic)
    assert not bool(evaluation.successful)
    assert int(evaluation.rejection_reasons) & int(
        AtomisticStepRejectionReason.CONSTRAINT
    )
    candidate = np.asarray(evaluation.candidate_state.kinematics.positions)
    np.testing.assert_array_equal(candidate[:, 1:], 0.0)
    assert not bool(constraints.rank_evidence(jnp.asarray(candidate)).regular)
    # Rejection commits nothing.
    assert_tree_equal(evaluation.accepted_state, state)


def test_over_constrained_rigid_sets_are_irregular() -> None:
    pairs = [list(pair) for pair in combinations(range(4), 2)]
    # Four atoms have only six distinct pairs: a seventh constraint is a
    # duplicate unordered pair and is refused by the topology itself.
    with pytest.raises(ValueError, match="duplicate unordered pairs"):
        MolecularTopologyPlan(
            constraints=[*pairs, [1, 0]],
            constraint_distances=[*_distances(SQUARE, pairs), 1.0],
        )
    # All six distances of a planar square: out-of-plane motion is invisible to
    # every constraint, so the in-plane rank is 2 * 4 - 3 = 5 < 6.
    square = _rigid_system(SQUARE, np.ones(4), pairs)
    constraints = DistanceConstraintPlan().prepare(square)
    assert constraints.independent_constraint_count == 6
    evidence = constraints.rank_evidence(jnp.asarray(SQUARE))
    assert not bool(evidence.regular)
    np.testing.assert_array_equal(evidence.component_rank, [5])
    dynamics = prepare_dynamics(
        square,
        (ExternalFieldTerm("field", "zero", np.zeros((4, 3))),),
        VelocityVerletPlan(0.01),
        constraints=constraints,
    )
    thermodynamic = table(dynamics, row(measure(square), "nve"))
    with pytest.raises(INADMISSIBLE, match="not admissible"):
        dynamics.initialize_state(
            jnp.asarray(SQUARE), thermodynamic, velocity=jnp.zeros((4, 3)), key=jr.key(63)
        )
    # The same six distances on a nondegenerate tetrahedron are independent.
    tetrahedron = _rigid_system(TETRAHEDRON, np.ones(4), pairs)
    evidence = (
        DistanceConstraintPlan()
        .prepare(tetrahedron)
        .rank_evidence(jnp.asarray(TETRAHEDRON))
    )
    assert bool(evidence.regular)
    np.testing.assert_array_equal(evidence.component_rank, [6])


def test_component_beyond_materialization_limit_is_refused_at_prepare() -> None:
    chain = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    system = _rigid_system(chain, np.ones(3), [[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="spans 3 atoms"):
        DistanceConstraintPlan(maximum_component_atoms=2).prepare(system)
    assert DistanceConstraintPlan(maximum_component_atoms=3).prepare(system) is not None


def test_table_degrees_of_freedom_use_the_independent_constraint_count() -> None:
    system, constraints, _, thermodynamic = _water(field="zero", step_size=0.01)
    assert thermodynamic.constraint_rank == constraints.independent_constraint_count == 3
    assert thermodynamic.degrees_of_freedom == 3 * 3 - 3
    nose_hoover = prepare_dynamics(
        system,
        (ExternalFieldTerm("field", "zero", np.zeros((3, 3))),),
        NoseHooverChainPlan(0.01, NoseHooverChainParameters(0.5)),
        constraints=constraints,
    )
    restricted = table(
        nose_hoover,
        row(measure(system, restriction="center-of-mass"), "nvt", temperature=1.0),
    )
    assert (
        restricted.degrees_of_freedom
        == 3 * 3 - constraints.independent_constraint_count - 3
    )
    velocity = np.random.default_rng(64).normal(size=(3, 3))
    state = nose_hoover.initialize_state(
        jnp.asarray(WATER), restricted, velocity=jnp.asarray(velocity), key=jr.key(64)
    )
    momenta = np.asarray(state.kinematics.momenta)
    kinetic = 0.5 * float(np.sum(momenta * momenta / WATER_MASSES[:, None]))
    boltzmann = system.plan.units.boltzmann_constant
    # A three-term host sum against the ledger kinetic energy: rtol 1e-12.
    np.testing.assert_allclose(
        nose_hoover.diagnostics(state, restricted).temperature,
        2.0 * kinetic / (restricted.degrees_of_freedom * boltzmann),
        rtol=1e-12,
    )
