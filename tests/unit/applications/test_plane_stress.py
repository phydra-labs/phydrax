#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._frozendict import frozendict
from phydrax.applications.solid_mechanics._material_point import (
    NeoHookeanMPMConstitutivePlan,
)
from phydrax.applications.solid_mechanics._mixed_hyperelastic import (
    MixedHyperelasticLaw,
)
from phydrax.applications.solid_mechanics._mpm_plane_stress import (
    PlaneStressMPMConstitutivePlan,
)
from phydrax.applications.solid_mechanics._plane_stress import (
    BlockDiagonalPlaneStressReductionPlan,
    CoupledPlaneStressIncompressiblePlan,
    PlaneStressFailure,
)
from phydrax.applications.solid_mechanics._plane_stress_fields import (
    plane_stress_hyperelastic_form,
    plane_stress_hyperelastic_response,
)
from phydrax.operators.mechanics import (
    NeoHookeanLaw,
    NeoHookeanParameters,
)


def _law():
    return NeoHookeanLaw(NeoHookeanParameters.from_shear_bulk(3.0, 11.0))


def _isochoric_energy(deformation):
    return 1.5 * (jnp.sum(deformation * deformation) - 3.0)


def _jacobian_constraint(deformation):
    return jnp.linalg.det(deformation) - 1.0


def test_plane_stress_root_implicit_derivative_and_schur_tangent():
    plan = BlockDiagonalPlaneStressReductionPlan()
    law = _law()
    deformation = jnp.asarray([[1.12, 0.06], [0.02, 0.93]])
    response = plan.evaluate(deformation, law)

    assert bool(response.successful)
    assert response.failure == int(PlaneStressFailure.OK)
    assert abs(response.residual) < 1.0e-9
    assert response.kinematics.thickness_stretch > 0.0

    numerical_sensitivity = jax.jacfwd(
        lambda value: plan.evaluate(value, law).kinematics.log_thickness_stretch
    )(deformation)
    numerical_tangent = jax.jacfwd(lambda value: plan.evaluate(value, law).first_piola)(
        deformation
    )
    np.testing.assert_allclose(
        response.log_stretch_sensitivity,
        numerical_sensitivity,
        rtol=2.0e-7,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        response.condensed_tangent,
        numerical_tangent,
        rtol=3.0e-7,
        atol=3.0e-8,
    )


def test_reference_thickness_scales_areal_response_but_not_closure_root():
    plan = BlockDiagonalPlaneStressReductionPlan()
    law = _law()
    deformation = jnp.asarray([[1.08, 0.04], [0.01, 0.96]])
    unit = plan.evaluate(deformation, law, reference_thickness=1.0)
    thick = plan.evaluate(deformation, law, reference_thickness=2.75)

    np.testing.assert_allclose(
        thick.kinematics.thickness_stretch,
        unit.kinematics.thickness_stretch,
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        thick.reference_energy_density, 2.75 * unit.reference_energy_density
    )
    np.testing.assert_allclose(thick.first_piola, 2.75 * unit.first_piola)
    np.testing.assert_allclose(thick.condensed_tangent, 2.75 * unit.condensed_tangent)
    np.testing.assert_allclose(
        thick.kinematics.current_thickness,
        2.75 * unit.kinematics.current_thickness,
    )


def test_plane_stress_batches_and_reports_bracket_and_input_failures():
    law = _law()
    plan = BlockDiagonalPlaneStressReductionPlan()
    batch = jnp.asarray(
        [
            [[1.04, 0.02], [0.01, 0.98]],
            [[1.18, -0.03], [0.04, 0.91]],
            [[0.95, 0.06], [0.02, 1.09]],
        ]
    )
    response = plan.evaluate(batch, law, reference_thickness=jnp.asarray([1.0, 2.0, 0.5]))

    assert response.first_piola.shape == (3, 2, 2)
    assert response.condensed_tangent.shape == (3, 2, 2, 2, 2)
    assert response.bracket_residual.shape == (3, 2)
    assert jnp.all(response.successful)
    assert jnp.max(jnp.abs(response.residual)) < 1.0e-9

    no_bracket = BlockDiagonalPlaneStressReductionPlan(None, (-0.01, 0.01)).evaluate(
        1.5 * jnp.eye(2), law
    )
    assert not bool(no_bracket.successful)
    assert no_bracket.failure == int(PlaneStressFailure.NO_BRACKET)
    assert jnp.all(jnp.isfinite(no_bracket.bracket_residual))

    invalid = plan.evaluate(jnp.asarray([[-1.0, 0.0], [0.0, 1.0]]), law)
    assert not bool(invalid.successful)
    assert invalid.failure == int(PlaneStressFailure.INVALID_INPUT)
    assert not bool(jnp.isfinite(invalid.reference_energy_density))


def test_field_and_fe_adapters_match_point_reduction_and_h0():
    law = _law()
    plan = BlockDiagonalPlaneStressReductionPlan()
    h0 = 1.7
    displacement_gradient = jnp.asarray([[0.08, 0.03], [0.01, -0.05]])
    deformation = jnp.eye(2) + displacement_gradient
    expected = plan.evaluate(deformation, law, reference_thickness=h0)
    geometry = phx.domain.GeometryDomain(
        phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
    )

    @geometry.Function("x")
    def displacement(x):
        return displacement_gradient @ x

    fields = plane_stress_hyperelastic_response(
        displacement,
        law,
        plan,
        reference_thickness=h0,
    )
    point = frozendict({"x": cx.Field(jnp.asarray((0.2, -0.4)), dims=(None,))})
    np.testing.assert_allclose(
        fields.energy(point).data, expected.reference_energy_density
    )
    np.testing.assert_allclose(fields.first_piola(point).data, expected.first_piola)
    np.testing.assert_allclose(fields.cauchy(point).data, expected.cauchy_stress)
    np.testing.assert_allclose(
        fields.thickness_stretch(point).data,
        expected.kinematics.thickness_stretch,
    )
    assert bool(fields.successful(point).data)
    assert fields.failure(point).data == int(PlaneStressFailure.OK)

    action = plane_stress_hyperelastic_form(
        "u", law, plan, reference_thickness=h0
    ).actions[0]
    executor_gradient = displacement_gradient.T[None, None]
    values = jnp.zeros((1, 1, 2))
    points = jnp.zeros((1, 1, 2))

    def energy(gradient):
        return jnp.sum(action.density(values, gradient, points, None))

    np.testing.assert_allclose(
        energy(executor_gradient), expected.reference_energy_density
    )
    np.testing.assert_allclose(
        jax.grad(energy)(executor_gradient)[0, 0],
        expected.first_piola.T,
        rtol=3.0e-7,
        atol=3.0e-8,
    )


def test_plane_stress_fe_form_compiles_at_identity():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = jnp.asarray([[0, 1, 3], [1, 2, 3]], dtype=jnp.int32)
    mesh = phx.discretization.CellMesh.from_triangles(vertices, cells)
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
        component_shape=(2,),
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = plane_stress_hyperelastic_form(
        "u", _law(), BlockDiagonalPlaneStressReductionPlan()
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    residual = compiled.residual(compiled.state_space.zeros())

    for leaf in jax.tree.leaves(residual):
        np.testing.assert_allclose(leaf, 0.0, atol=2.0e-10)


def test_mpm_reduction_matches_pointwise_energy_stress_root_and_tangent():
    law = _law()
    plan = BlockDiagonalPlaneStressReductionPlan()
    deformation = jnp.asarray([[[1.12, 0.06], [0.02, 0.93]]])
    point = plan.evaluate(deformation, law)
    material = PlaneStressMPMConstitutivePlan(
        NeoHookeanMPMConstitutivePlan(3),
        reduction=plan,
    )
    parameters = law.parameters
    history = material.initialize_state((1,), deformation.dtype)
    density = jnp.asarray((2.0,))
    response = material.evaluate(deformation, history, density, parameters, 0.0, 0.01)
    linearized = material.evaluate_linearized(
        deformation, history, density, parameters, 0.0, 0.01
    )

    assert bool(response.successful[0])
    assert bool(linearized.tangent_successful[0])
    np.testing.assert_allclose(
        response.reference_energy_density, point.reference_energy_density
    )
    np.testing.assert_allclose(response.first_piola, point.first_piola)
    np.testing.assert_allclose(
        response.diagnostics["out_of_plane_stretch"],
        point.kinematics.thickness_stretch,
    )
    np.testing.assert_allclose(
        response.diagnostics["plane_stress_residual"], point.residual, atol=1.0e-10
    )
    np.testing.assert_allclose(
        linearized.algorithmic_tangent,
        point.condensed_tangent,
        rtol=3.0e-7,
        atol=3.0e-8,
    )


@pytest.mark.parametrize("bulk_modulus", [None, 25.0])
def test_coupled_plane_stress_incompressibility_solves_both_equations_and_tangent(
    bulk_modulus,
):
    law = MixedHyperelasticLaw(
        _isochoric_energy,
        _jacobian_constraint,
        bulk_modulus=bulk_modulus,
    )
    plan = CoupledPlaneStressIncompressiblePlan(
        None,
        (-2.0, 2.0),
        (-100.0, 100.0),
        _jacobian_constraint,
        bulk_modulus,
    )
    deformation = jnp.asarray([[1.2, 0.04], [0.02, 0.9]])
    response = plan.evaluate(deformation, law, reference_thickness=1.6)

    assert bool(response.successful)
    assert response.failure == int(PlaneStressFailure.OK)
    assert jnp.max(jnp.abs(response.residual)) < 1.0e-9
    assert response.thickness_stretch > 0.0
    embedded = (
        jnp.zeros((3, 3))
        .at[:2, :2]
        .set(deformation)
        .at[2, 2]
        .set(response.thickness_stretch)
    )
    np.testing.assert_allclose(
        law.constraint(embedded, response.pressure), 0.0, atol=1.0e-9
    )
    np.testing.assert_allclose(
        law.first_piola(embedded, response.pressure)[2, 2], 0.0, atol=1.0e-9
    )

    step = 2.0e-5
    directions = jnp.eye(4).reshape((4, 2, 2))
    columns = []
    for direction in directions:
        plus = plan.evaluate(deformation + step * direction, law, reference_thickness=1.6)
        minus = plan.evaluate(
            deformation - step * direction, law, reference_thickness=1.6
        )
        columns.append((plus.first_piola - minus.first_piola) / (2.0 * step))
    finite_difference = jnp.stack(columns, axis=-1).reshape((2, 2, 2, 2))
    np.testing.assert_allclose(
        response.condensed_tangent,
        finite_difference,
        rtol=3.0e-4,
        atol=3.0e-5,
    )


def test_coupled_plane_stress_batches_and_scalar_plan_rejects_mixed_law():
    law = MixedHyperelasticLaw(_isochoric_energy, _jacobian_constraint)
    coupled = CoupledPlaneStressIncompressiblePlan(
        None,
        (-2.0, 2.0),
        (-100.0, 100.0),
        _jacobian_constraint,
    )
    deformation = jnp.asarray(
        [
            [[1.2, 0.04], [0.02, 0.9]],
            [[0.95, -0.02], [0.03, 1.08]],
        ]
    )
    response = coupled.evaluate(
        deformation,
        law,
        reference_thickness=jnp.asarray((1.0, 2.0)),
    )

    assert response.residual.shape == (2, 2)
    assert response.condensed_tangent.shape == (2, 2, 2, 2, 2)
    assert jnp.all(response.successful)
    assert jnp.max(jnp.abs(response.residual)) < 1.0e-9

    bounded_out = CoupledPlaneStressIncompressiblePlan(
        None,
        (0.1, 0.2),
        (-0.01, 0.01),
        _jacobian_constraint,
    ).evaluate(deformation[0], law)
    assert not bool(bounded_out.successful)
    assert bounded_out.failure == int(PlaneStressFailure.MAX_STEPS)
    with pytest.raises(TypeError, match="mixed pressure laws"):
        BlockDiagonalPlaneStressReductionPlan().evaluate(jnp.eye(2), law)
