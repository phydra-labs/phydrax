#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax as phx


sm = phx.applications.solid_mechanics


def _certified_properties(*, positive_definite=False):
    evidence = {"self_adjoint": "verified"}
    if positive_definite:
        evidence.update(
            {
                "positive_definite": "verified",
                "positive_semidefinite": "verified",
            }
        )
    return phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_definite=positive_definite,
        evidence=evidence,
    )


def test_mechanics_wrapper_keeps_one_authoritative_nonlinear_root():
    space = phx.linalg.ArraySpace((2,), space_id="mechanics-physical-root")
    root = phx.nonlinear.NonlinearSystemProblem(
        lambda state, args: state - args,
        state_space=space,
        residual_space=space,
        validity=lambda state, residual, auxiliary, args: jnp.linalg.norm(residual) < 1.0,
        problem_id="mechanics-root",
    )
    mechanics = sm.MechanicsEquilibriumProblem(
        root,
        realization_id="load-realization-0",
        provenance_id="assembled-residual-0",
        admissibility=lambda state, residual, auxiliary, args: jnp.all(state > -1.0),
        admissibility_id="positive-or-small-state",
    )
    state = jnp.asarray((0.5, 0.25))
    residual, auxiliary = mechanics.evaluate(state, jnp.zeros(2))

    assert mechanics.root_problem is root
    assert mechanics.state_space is root.state_space
    assert mechanics.residual_space is root.residual_space
    assert mechanics.valid(state, residual, auxiliary, jnp.zeros(2))
    assert not mechanics.admissible(jnp.asarray((-2.0, 0.0)), jnp.zeros(2))


def test_physical_stability_refuses_field_parameter_hessian_semantics():
    parameter_space = phx.linalg.ArraySpace(
        (2,),
        space_id="selected-field-parameters",
    )
    root = phx.nonlinear.NonlinearSystemProblem(
        lambda state, args: state,
        state_space=parameter_space,
        residual_space=parameter_space,
        problem_id="neural-stationarity-root",
    )
    equilibrium = sm.MechanicsEquilibriumProblem(
        root,
        realization_id="field-realization-0",
        provenance_id="parameter-stationarity-0",
        root_coordinates="field-parameters",
    )
    parameter_hessian = phx.linalg.DenseLinearOperator(
        jnp.eye(2),
        source=parameter_space,
        target=parameter_space,
        properties=_certified_properties(),
        operator_id="parameter-hessian",
    )

    with pytest.raises(ValueError, match="field-parameter space"):
        sm.PhysicalStaticStabilityProblem(
            equilibrium,
            parameter_space,
            parameter_hessian,
            tangent_provenance_id="parameter-hessian-is-not-physical",
        )

    physical_space = phx.linalg.ArraySpace((2,), space_id="physical-displacement")
    stiffness = phx.linalg.DenseLinearOperator(
        jnp.diag(jnp.asarray((2.0, 5.0))),
        source=physical_space,
        target=physical_space,
        properties=_certified_properties(),
        operator_id="physical-tangent",
    )
    mass = phx.linalg.DenseLinearOperator(
        jnp.diag(jnp.asarray((1.0, 3.0))),
        source=physical_space,
        target=physical_space,
        properties=_certified_properties(positive_definite=True),
        operator_id="physical-mass",
    )
    static = sm.PhysicalStaticStabilityProblem(
        equilibrium,
        physical_space,
        stiffness,
        tangent_provenance_id="physical-tangent-assembly-0",
    )
    dynamic = sm.DynamicStabilityProblem(
        equilibrium,
        physical_space,
        stiffness,
        mass,
        stiffness_provenance_id="physical-stiffness-assembly-0",
        mass_provenance_id="physical-mass-assembly-0",
    )

    assert static.eigenvalue_quantity == "physical-tangent-curvature"
    assert static.as_eigenproblem().operator is stiffness
    assert dynamic.eigenvalue_quantity == "squared-angular-frequency"
    assert dynamic.as_generalized_eigenproblem().metric_operator is mass
