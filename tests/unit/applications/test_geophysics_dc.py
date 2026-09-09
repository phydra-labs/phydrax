#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.geophysics import (
    DCElectricalObservationPlan,
    ElectricalSurvey,
    ElectrodePatch,
    FinitePatchDCPlan,
    LogConductivity,
)
from phydrax.discretization import CellMesh
from phydrax.observation import CholeskyCovarianceAction, CoordinateLayout
from phydrax.units import MILLIMETER, MILLIVOLT
from phydrax.uq import ParameterSpace


def _patches(mesh, faces):
    face_lookup = {
        tuple(sorted(face)): row
        for row, face in enumerate(np.asarray(mesh.connectivity.faces))
    }
    return tuple(
        ElectrodePatch(f"electrode-{index}", [face_lookup[tuple(sorted(face))]])
        for index, face in enumerate(faces)
    )


def _tetrahedron(scale=1.0):
    return CellMesh.from_tetrahedra(
        scale
        * jnp.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32),
    )


def _survey(mesh, current_scale=1.0):
    patches = _patches(mesh, ((0, 2, 3), (0, 1, 3), (0, 1, 2), (1, 2, 3)))
    currents = current_scale * jnp.asarray(
        [[1.0, -1.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0], [0.0, 1.0, -1.0, 0.0]]
    )
    receivers = jnp.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )
    return ElectricalSurvey(patches, currents, receivers, jnp.asarray([0, 0, 1, 2]))


def test_patch_integrated_current_conservation_reciprocity_and_minimum_norm_gauge():
    mesh = _tetrahedron()
    prepared = FinitePatchDCPlan(mesh, _survey(mesh), batch_size=2).prepare()
    bound = prepared.bind_conductivity(1.0)
    result = bound.solve()

    assert bool(result.successful)
    assert jnp.allclose(
        prepared.electrode_areas, jnp.asarray([0.5, 0.5, 0.5, np.sqrt(3) / 2]), atol=1e-12
    )
    rhs = prepared.current_load(prepared.plan.survey.currents[0])
    assert jnp.allclose(rhs, jnp.asarray([0.0, -1 / 3, 1 / 3, 0.0]), atol=1e-12)
    assert jnp.allclose(
        result.potentials[0], jnp.asarray([0.0, -2.0, 2.0, 0.0]), atol=1e-9
    )
    for current, potential in zip(
        prepared.plan.survey.currents, result.potentials, strict=True
    ):
        load = prepared.current_load(current)
        assert jnp.allclose(jnp.sum(load), 0.0, atol=1e-12)
        assert jnp.allclose(
            bound.linear_solve.problem.operator.mv(potential), load, atol=1e-9
        )
    assert jnp.allclose(jnp.mean(result.potentials, axis=1), 0.0, atol=1e-10)
    assert jnp.allclose(
        result.voltages, jnp.asarray([4 / 3, 2 / 3, 2 / 3, -2 / 3]), atol=1e-9
    )
    assert jnp.allclose(result.voltages[1], result.voltages[2], atol=1e-10)
    shifted = jax.vmap(prepared.electrode_potentials)(result.potentials + 17.0)
    assert jnp.allclose(prepared.voltages(shifted), result.voltages, atol=1e-10)
    assert jnp.allclose(bound.predict(), result.voltages, atol=1e-10)


def test_conductivity_current_and_three_dimensional_length_scaling():
    mesh = _tetrahedron()
    prepared = FinitePatchDCPlan(mesh, _survey(mesh), batch_size=1).prepare()
    base = prepared.predict(1.0)
    assert jnp.allclose(prepared.predict(3.0), base / 3, atol=1e-9)
    current_scaled = FinitePatchDCPlan(mesh, _survey(mesh, current_scale=-2.0)).prepare()
    assert jnp.allclose(current_scaled.predict(1.0), -2 * base, atol=1e-9)
    larger = _tetrahedron(scale=2.0)
    assert jnp.allclose(
        FinitePatchDCPlan(larger, _survey(larger)).prepare().predict(1.0),
        base / 2,
        atol=1e-9,
    )
    millimetres = _tetrahedron(scale=1000.0)
    assert jnp.allclose(
        FinitePatchDCPlan(millimetres, _survey(millimetres), length_unit=MILLIMETER)
        .prepare()
        .predict(1.0),
        base,
        atol=1e-9,
    )


def test_spd_tensor_current_field_and_reciprocity():
    mesh = _tetrahedron()
    prepared = FinitePatchDCPlan(mesh, _survey(mesh)).prepare()
    tensor = jnp.asarray([[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]])
    result = prepared.bind_conductivity(tensor).solve()
    assert jnp.allclose(
        result.potentials[0], jnp.asarray([-2 / 7, -2.0, 18 / 7, -2 / 7]), atol=1e-9
    )
    assert jnp.allclose(result.voltages[0], 32 / 21, atol=1e-9)
    assert jnp.allclose(result.voltages[1], result.voltages[2], atol=1e-9)
    with pytest.raises(Exception, match="positive"):
        prepared.predict(jnp.diag(jnp.asarray([1.0, 1.0, -1.0])))
    with pytest.raises(Exception, match="symmetric"):
        prepared.predict(jnp.asarray([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(Exception, match="positive"):
        prepared.predict(0.0)


def test_heterogeneous_log_conductivity_implicit_jvp_vjp_and_signed_posterior():
    mesh = CellMesh.from_tetrahedra(
        jnp.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        jnp.asarray([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=jnp.int32),
    )
    patches = _patches(mesh, ((0, 2, 3), (0, 1, 3), (0, 1, 4), (1, 2, 4)))
    patterns = jnp.asarray([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    receivers = jnp.asarray(
        [[0.0, 0.0, 1.0, -1.0], [1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]]
    )
    survey = ElectricalSurvey(patches, patterns, receivers, jnp.asarray([0, 1, 0]))
    dc = FinitePatchDCPlan(mesh, survey, batch_size=1).prepare()
    parameterization = LogConductivity(jnp.asarray([1.0, 2.0]))
    predict = lambda parameters: dc.predict(parameterization(parameters))
    parameters, direction = jnp.asarray([0.1, -0.2]), jnp.asarray([0.3, -0.4])
    value, tangent = jax.jvp(predict, (parameters,), (direction,))
    epsilon = 2e-5
    finite_difference = (
        predict(parameters + epsilon * direction)
        - predict(parameters - epsilon * direction)
    ) / (2 * epsilon)
    assert jnp.allclose(tangent, finite_difference, rtol=3e-5, atol=1e-8)
    assert jnp.allclose(value[0], value[1], atol=1e-9)
    cotangent = jnp.asarray([0.2, -0.7, 0.4])
    _, pullback = jax.vjp(predict, parameters)
    assert jnp.allclose(
        jnp.sum(pullback(cotangent)[0] * direction),
        jnp.sum(cotangent * tangent),
        atol=1e-8,
    )
    assert jnp.allclose(jax.jit(predict)(parameters), value, atol=1e-9)

    covariance = CholeskyCovarianceAction(
        0.05 * jnp.eye(3), CoordinateLayout(("cross-a", "cross-b", "signed"))
    )
    observed = predict(jnp.asarray([0.3, -0.1]))
    assert float(observed[-1]) < 0
    observation = DCElectricalObservationPlan(dc, parameterization, observed, covariance)
    parameter_space = ParameterSpace(
        parameters, log_prior=lambda position: -0.5 * jnp.sum(position**2)
    )
    posterior = observation.posterior(parameter_space)
    expected = observation.field_observation.log_likelihood(value) - 0.5 * jnp.sum(
        parameters**2
    )
    assert jnp.allclose(posterior.log_density(parameters), expected, atol=1e-10)
    gradient = jax.grad(posterior.log_density)(parameters)
    density_fd = (
        posterior.log_density(parameters + epsilon * direction)
        - posterior.log_density(parameters - epsilon * direction)
    ) / (2 * epsilon)
    assert jnp.allclose(jnp.sum(gradient * direction), density_fd, rtol=3e-5, atol=1e-7)
    assert jnp.allclose(posterior.predict(parameters), value, atol=1e-9)
    millivolt_observation = DCElectricalObservationPlan(
        dc,
        parameterization,
        1000 * observed,
        CholeskyCovarianceAction(50 * jnp.eye(3), covariance.layout),
        voltage_unit=MILLIVOLT,
    )
    assert jnp.allclose(
        millivolt_observation.log_likelihood(parameters),
        observation.log_likelihood(parameters),
        atol=1e-9,
    )


def test_invalid_physical_surveys_and_disconnected_bodies_are_rejected():
    mesh = _tetrahedron()
    survey = _survey(mesh)
    with pytest.raises(ValueError, match="balanced"):
        ElectricalSurvey(
            survey.patches, [[1.0, 0.0, 0.0, 0.0]], [[1.0, -1.0, 0.0, 0.0]], [0]
        )
    with pytest.raises(ValueError, match="balanced"):
        ElectricalSurvey(
            survey.patches, [[1.0, -1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [0]
        )
    invalid_patches = (ElectrodePatch("absent", [999]), *survey.patches[1:])
    with pytest.raises(ValueError, match="absent"):
        FinitePatchDCPlan(
            mesh,
            ElectricalSurvey(
                invalid_patches,
                survey.currents,
                survey.receiver_weights,
                survey.source_indices,
            ),
        )
    with pytest.raises(ValueError, match="overlap"):
        FinitePatchDCPlan(
            mesh,
            ElectricalSurvey(
                (
                    ElectrodePatch("duplicate", survey.patches[1].facet_indices),
                    *survey.patches[1:],
                ),
                survey.currents,
                survey.receiver_weights,
                survey.source_indices,
            ),
        )
    disconnected = CellMesh.from_tetrahedra(
        jnp.concatenate((mesh.coordinates, mesh.coordinates + 3.0), axis=0),
        jnp.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32),
    )
    with pytest.raises(ValueError, match="Disconnected"):
        FinitePatchDCPlan(disconnected, survey)
