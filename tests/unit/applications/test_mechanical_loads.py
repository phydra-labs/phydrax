#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import pytest

from phydrax.applications.solid_mechanics._loads import (
    ClosedSurfacePressure,
    CompositeMechanicalLoad,
    CurrentBodyForce,
    GeneralFollowerLoad,
    MechanicalLoadSemantics,
    MechanicalLoadState,
    PneumaticPressure,
    ReferenceDeadBodyForce,
)
from phydrax.equations._mechanical_load_action import MechanicalLoadAction
from phydrax.integration._deformed_measure import DeformedMeasurePlan
from phydrax.nn.parameters import ParameterSubspace


def _triangle_action(load):
    reference = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    gathers = jnp.asarray(((0, 1, 2),), dtype=jnp.int32)
    basis = jnp.asarray(((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),))
    gradients = jnp.asarray((((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)),))
    gradients = gradients[:, None, :, :]
    measure = DeformedMeasurePlan("volume", jnp.asarray(((0.5,),)))
    return MechanicalLoadAction(
        load,
        reference,
        gathers,
        basis,
        gradients,
        measure,
        action_id="one-triangle-load",
    )


def _tetrahedron_surface(scale=1.0):
    root_three = jnp.sqrt(3.0)
    centroids = jnp.asarray(
        (
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.0, 1.0 / 3.0, 1.0 / 3.0),
            (1.0 / 3.0, 0.0, 1.0 / 3.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0),
        )
    )
    normals = jnp.asarray(
        (
            (1.0 / root_three, 1.0 / root_three, 1.0 / root_three),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )
    measures = jnp.asarray((root_three / 2.0, 0.5, 0.5, 0.5))
    plan = DeformedMeasurePlan(
        "surface",
        measures,
        reference_normal=normals,
        plan_id="closed-oriented-tetrahedron-measure",
    )
    deformation = scale * jnp.broadcast_to(jnp.eye(3), (4, 3, 3))
    return centroids, scale * centroids, plan.evaluate(deformation)


def _tetrahedron_action(load):
    reference = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    basis = jnp.asarray(
        (
            (0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0 / 3.0),
            (1.0 / 3.0, 1.0 / 3.0, 0.0, 1.0 / 3.0),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        )
    )
    gradient = jnp.asarray(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    gradients = jnp.broadcast_to(gradient, (1, 4, 4, 3))
    root_three = jnp.sqrt(3.0)
    normals = jnp.asarray(
        (
            (1.0 / root_three, 1.0 / root_three, 1.0 / root_three),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )[None, :, :]
    measures = jnp.asarray(((root_three / 2.0, 0.5, 0.5, 0.5),))
    plan = DeformedMeasurePlan(
        "surface",
        measures,
        reference_normal=normals,
        plan_id="tetrahedron-action-surface",
    )
    return MechanicalLoadAction(
        load,
        reference,
        jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32),
        basis,
        gradients,
        plan,
        action_id="tetrahedron-pressure-action",
    )


def test_mechanical_semantics_refuse_uncertified_or_open_potential_routing():
    with pytest.raises(ValueError, match="certified"):
        MechanicalLoadSemantics(
            "body",
            "reference",
            "reference",
            "current",
            "potential",
            potential_certified=False,
        )
    with pytest.raises(ValueError, match="closure and orientation"):
        MechanicalLoadSemantics(
            "boundary",
            "current",
            "current",
            "current",
            "potential",
            potential_certified=True,
        )


def test_dead_load_potential_gradient_is_the_assembled_load_residual():
    action = _triangle_action(ReferenceDeadBodyForce(jnp.asarray((2.0, -3.0))))
    state = MechanicalLoadState(1.5)
    current = action.reference_coordinates + jnp.asarray(
        ((0.1, -0.2), (0.2, 0.05), (-0.1, 0.3))
    )
    evaluation = action.evaluate(current, state)
    potential_gradient = jax.grad(
        lambda coordinates: action.potential(coordinates, state)
    )(current)
    np.testing.assert_allclose(
        potential_gradient, evaluation.residual, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        evaluation.external_force,
        jnp.broadcast_to(jnp.asarray((0.5, -0.75)), (3, 2)),
    )
    assert bool(evaluation.valid)


def test_closed_and_pneumatic_pressure_follow_current_volume():
    reference, current, measure = _tetrahedron_surface(scale=2.0)
    state = MechanicalLoadState()
    closed = ClosedSurfacePressure(
        3.0,
        closure_id="tetrahedron-closed",
        orientation_id="tetrahedron-outward",
    )
    fixed = closed.evaluate(reference, current, measure, state)
    expected_volume = 8.0 / 6.0
    np.testing.assert_allclose(
        jnp.sum(fixed.potential_density * measure.current_measure),
        -3.0 * expected_volume,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(fixed.total_force_density**2, axis=-1)), 3.0
    )

    pneumatic = PneumaticPressure(
        6.0,
        1.0 / 6.0,
        exponent=1.0,
        closure_id="tetrahedron-closed",
        orientation_id="tetrahedron-outward",
    )
    gas = pneumatic.evaluate(reference, current, measure, state)
    np.testing.assert_allclose(
        jnp.sqrt(jnp.sum(gas.total_force_density**2, axis=-1)), 0.75
    )
    expected_potential = -(6.0 / 6.0) * jnp.log(8.0)
    np.testing.assert_allclose(
        jnp.sum(gas.potential_density * measure.current_measure),
        expected_potential,
        rtol=1e-6,
        atol=1e-6,
    )
    assert bool(gas.valid)


def test_closed_pressure_FE_potential_gradient_matches_external_residual():
    pressure = ClosedSurfacePressure(
        2.5,
        closure_id="tetrahedron-closed",
        orientation_id="tetrahedron-outward",
    )
    action = _tetrahedron_action(pressure)
    state = MechanicalLoadState()
    current = 1.2 * action.reference_coordinates
    residual = action.residual(current, state)
    gradient = jax.grad(lambda coordinates: action.potential(coordinates, state))(current)
    np.testing.assert_allclose(gradient, residual, rtol=1e-6, atol=1e-6)


def test_composite_preserves_components_and_conservative_routing():
    first = ReferenceDeadBodyForce(jnp.asarray((1.0, 0.0)), load_id="first")
    second = ReferenceDeadBodyForce(jnp.asarray((0.0, 2.0)), load_id="second")
    composite = CompositeMechanicalLoad((first, second))
    action = _triangle_action(composite)
    evaluation = action.evaluate(action.reference_coordinates, MechanicalLoadState())
    assert evaluation.load.component_ids == ("first", "second")
    assert len(evaluation.load.component_force_densities) == 2
    assert evaluation.load.semantics.conservativity == "potential"

    mixed = CompositeMechanicalLoad(
        (
            CurrentBodyForce(jnp.asarray((1.0, 0.0)), load_id="current-first"),
            CurrentBodyForce(jnp.asarray((0.0, 1.0)), load_id="current-second"),
        )
    )
    assert mixed.semantics.conservativity == "virtual_work"
    assert not mixed.semantics.potential_certified


def test_general_follower_action_keeps_its_nonsymmetric_tangent():
    matrix = jnp.asarray(((0.0, 2.0), (-1.0, 0.5)))

    def law(reference, current, measure, state, args):
        del reference, measure, state, args
        return oe.contract("ij,...j->...i", matrix, current)

    load = GeneralFollowerLoad(
        law,
        support="body",
        measure_frame="reference",
        load_id="nonsymmetric-linear-follower",
    )
    action = _triangle_action(load)
    current = action.reference_coordinates + 0.1
    tangent = action.tangent(current, MechanicalLoadState())
    assert not np.allclose(tangent, tangent.T)
    with pytest.raises(ValueError, match="nonconservative"):
        action.potential(current, MechanicalLoadState())


def test_neural_mechanical_load_prepares_physical_virtual_work_pullback():
    action = _triangle_action(ReferenceDeadBodyForce(jnp.asarray((1.0, -2.0))))
    functions = {"coordinates": action.reference_coordinates + 0.05}

    def trace(root, reference_coordinates, args):
        del reference_coordinates, args
        return root["coordinates"]

    subspace = ParameterSubspace(functions, eqx.is_inexact_array)
    prepared = action.prepare_neural_virtual_work(
        functions,
        trace,
        subspace,
        MechanicalLoadState(),
        trace_id="triangle-coordinate-field",
    )
    residual = prepared.problem.residual_function(prepared.initial_state, None)
    assert residual.shape == prepared.initial_state.shape
    assert jnp.all(jnp.isfinite(residual))
    assert prepared.formulation == "virtual-work"
