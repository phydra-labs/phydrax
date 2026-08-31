#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.fem import FiniteElementMortarMetricData
from phydrax.equations.fem import (
    ALEMetricState,
    BR1ViscousPlan,
    ConservativeModalLimiter,
    derived_mortar_entropy_defect,
    DGSEMCharacteristicBoundaryPlan,
    entropy_stable_wall_evidence,
    HPOverintegrationPolicy,
    LocalTimeSteppingPlan,
    MovingMortarMetricPlan,
    PositivityLimiter,
    SplitFormPolicy,
    SubcellFiniteVolumePlan,
    TemporalHPBudget,
    TroubledCellEvidence,
    WellBalancedSourceLedger,
)


def test_physical_boundaries_split_forms_viscosity_and_well_balancing():
    state = jnp.asarray((1.0, 2.0, 0.0, 4.0))
    normal = jnp.asarray((1.0, 0.0))
    wall = DGSEMCharacteristicBoundaryPlan("slip-wall").exterior_state(
        state, state, normal
    )
    np.testing.assert_allclose(np.asarray(wall), (1.0, -2.0, 0.0, 4.0))
    inflow = DGSEMCharacteristicBoundaryPlan("inflow").exterior_state(
        state, state * 2.0, normal
    )
    np.testing.assert_allclose(np.asarray(inflow), np.asarray(state * 2.0))

    evidence = entropy_stable_wall_evidence(
        jnp.asarray(((1.0, 0.0, 0.0, 2.0),)),
        jnp.zeros((1, 4)),
        jnp.ones((1, 4)),
        jnp.asarray(((1.0, 0.0),)),
    )
    assert bool(evidence.passed)
    ledger = WellBalancedSourceLedger(jnp.ones((3, 2)), jnp.ones((3, 2)))
    assert ledger.balance_error == 0.0

    derivative = jnp.asarray(((-1.0, 1.0), (-1.0, 1.0)))
    viscous = BR1ViscousPlan((derivative,), penalty=0.25)
    values = jnp.asarray((((0.0,), (1.0,)),))
    gradient = viscous.gradient(values)
    np.testing.assert_allclose(np.asarray(gradient[..., 0, 0]), ((1.0, 1.0),))
    split = SplitFormPolicy("skew-symmetric").combine(jnp.ones((2,)), jnp.full((2,), 3.0))
    np.testing.assert_allclose(np.asarray(split), 2.0)
    assert HPOverintegrationPolicy((4, 3), (2, 2), 2).quadrature_counts[0] >= 5


def test_shock_sensor_limiters_and_subcell_transfer_preserve_invariants():
    evidence = TroubledCellEvidence(
        jnp.asarray((0.1, 2.0)),
        jnp.asarray((0.2, 0.5)),
        jnp.asarray((0.1, 0.3)),
    )
    np.testing.assert_array_equal(np.asarray(evidence.troubled), (False, True))
    coefficients = jnp.asarray(((1.0, 0.8, 0.4), (2.0, 1.0, 0.5)))
    limited = ConservativeModalLimiter(1.0).apply(coefficients, evidence.troubled)
    np.testing.assert_allclose(np.asarray(limited[:, 0]), np.asarray(coefficients[:, 0]))
    assert float(limited[1, -1]) < float(coefficients[1, -1])

    states = jnp.asarray(
        (
            ((1.0, 0.0, 2.5), (0.1, 0.0, 0.01)),
            ((2.0, 0.0, 4.0), (1.5, 0.0, 3.0)),
        )
    )
    average = jnp.mean(states, axis=1)

    def pressure(value):
        return 0.4 * (value[..., -1] - 0.5 * value[..., 1] ** 2 / value[..., 0])

    positive, theta = PositivityLimiter(1.0e-3, 1.0e-3).apply(states, average, pressure)
    assert jnp.min(positive[..., 0]) >= 1.0e-3
    assert jnp.min(pressure(positive)) >= 1.0e-3 - 1.0e-12
    assert jnp.all(theta <= 1.0)

    nodes = np.linspace(-1.0, 1.0, 4)[:, None]
    subcells = np.linspace(-1.0, 1.0, 7)[:, None]
    plan = SubcellFiniteVolumePlan(nodes, subcells)
    constant = jnp.ones((1, 4, 2))
    projected = plan.project(constant)
    reconstructed = plan.reconstruct(projected)
    np.testing.assert_allclose(np.asarray(reconstructed), 1.0, atol=2.0e-12)
    advanced = plan.advance(
        projected,
        jnp.zeros_like(projected),
        jnp.zeros_like(projected),
        0.1,
        jnp.ones(projected.shape[:-1]),
    )
    np.testing.assert_allclose(np.asarray(advanced), np.asarray(projected))


def test_ale_moving_mortars_local_time_and_space_time_budget_close():
    coordinates = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    velocity = jnp.asarray(((0.1, 0.0), (0.1, 0.0)))
    ale = ALEMetricState(
        coordinates,
        velocity,
        jnp.ones((2,)),
        jnp.asarray((0.2, -0.2)),
        jnp.asarray((-0.2, 0.2)),
    )
    np.testing.assert_allclose(np.asarray(ale.temporal_gcl_defect), 0.0)

    metric = FiniteElementMortarMetricData(
        coordinates,
        jnp.ones((2,)),
        jnp.asarray(((0.0, 1.0), (0.0, 1.0))),
        jnp.asarray(((0.0, -1.0), (0.0, -1.0))),
    )
    moved = MovingMortarMetricPlan().update(metric, velocity, 0.5)
    np.testing.assert_allclose(
        np.asarray(moved.physical_coordinates), np.asarray(coordinates + 0.5 * velocity)
    )

    time = LocalTimeSteppingPlan(jnp.asarray((0.1, 0.05)), jnp.asarray((0, 1)))
    np.testing.assert_allclose(np.asarray(time.level_steps), (0.1, 0.05))
    np.testing.assert_allclose(
        np.asarray(time.reflux(jnp.asarray((3.0,)), jnp.asarray(((1.0,), (2.0,))))),
        0.0,
    )
    budget = TemporalHPBudget(
        jnp.asarray((0.2, 0.01)),
        jnp.asarray((0.1, 0.3)),
        jnp.asarray((0.01, 0.01)),
        0.15,
    )
    np.testing.assert_array_equal(np.asarray(budget.refine_space), (True, False))
    np.testing.assert_array_equal(np.asarray(budget.refine_time), (False, True))


def test_derived_entropy_defect_uses_declared_thermodynamic_values():
    left = jnp.asarray(((1.0, 2.0), (0.5, -0.25)))
    right = jnp.asarray(((2.0, 3.0), (1.5, 0.75)))
    flux = 0.5 * (left + right)
    left_potential = 0.5 * jnp.sum(left**2, axis=-1)
    right_potential = 0.5 * jnp.sum(right**2, axis=-1)
    defect = derived_mortar_entropy_defect(
        left,
        right,
        left,
        right,
        flux,
        left_potential,
        right_potential,
    )
    np.testing.assert_allclose(np.asarray(defect), 0.0, atol=2.0e-14)
