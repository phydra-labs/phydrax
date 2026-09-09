#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel


class _ParentProductModel(AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.in_size = 2
        self.out_size = "scalar"

    def __call__(self, values, /, *, key=None):
        del key
        return values[0] * values[1]


def _path():
    low = phx.fidelity.FidelityLevelSpec(
        "low",
        problem_id="poisson",
        observable_id="u",
        model_id="low-pinn",
        approximation_id="low",
        observable_contract_id="scalar-field",
    )
    high = phx.fidelity.FidelityLevelSpec(
        "high",
        problem_id="poisson",
        observable_id="u",
        model_id="high-pinn",
        approximation_id="high",
        observable_contract_id="scalar-field",
    )
    return phx.fidelity.FidelityHierarchy(
        (low, high),
        (phx.fidelity.FidelityRelation("low", "high"),),
        target_level_id="high",
    ).linear_path()


def _fixed_observation(geometry, target):
    component = geometry.component()
    batch = component.points({"x": jnp.asarray([-0.5, 0.0, 0.5])})
    condition = phx.conditions.Observation(
        "u",
        component,
        geometry.Function()(target),
    )
    return phx.terms.ResidualPenalty(
        condition,
        phx.integration.fixed(
            phx.integration.from_samples(
                phx.integration.mean_over(component),
                batch,
            )
        ),
    )


def test_fidelity_pinn_stage_freezes_parent_and_binds_stage_identity():
    geometry = phx.domain.Interval1d(-1.0, 1.0)
    parent_solver = phx.solver.FunctionalSolver(
        functions={"u": geometry.Parameter(1.0)},
        terms=(_fixed_observation(geometry, 1.0),),
    )
    parent = phx.solver.bind_fidelity_pinn_level(
        _path(),
        "low",
        parent_solver,
    )
    stage = phx.solver.prepare_fidelity_pinn_stage(
        parent,
        "high",
        {"u": geometry.Parameter(0.5)},
        (_fixed_observation(geometry, 1.05),),
        epsilon=0.1,
        replacement_functions={"coefficient": geometry.Parameter(2.0)},
    )

    assert stage.source_level_id == "low"
    assert stage.target_level_id == "high"
    assert stage.training_solver.functions["coefficient"] is not None
    assert any(
        record.artifact_kind == "fidelity-pinn-stage"
        and record.artifact_id == stage.stage_id
        for record in stage.training_solver.discretization_bundle.records
    )
    result = stage.finalize(stage.training_solver)
    batch = geometry.component().points({"x": jnp.asarray([0.0])})
    assert jnp.allclose(result.functions["u"](batch).data, 1.05)
    assert result.source_result_id == parent.result_id

    with pytest.raises(ValueError, match="does not match"):
        stage.finalize(parent_solver)


def test_parent_conditioned_correction_propagates_coordinate_derivatives():
    geometry = phx.domain.Interval1d(-3.0, 3.0)
    parent = geometry.Function("x")(lambda x: x)
    correction = phx.solver.condition_fidelity_correction(
        parent,
        _ParentProductModel(),
        correction_id="quadratic-parent-correction",
    )

    assert jnp.allclose(correction.func(jnp.asarray(2.0)), 4.0)
    derivative = phx.operators.differential.partial_n(
        correction,
        var="x",
        order=1,
    )
    assert jnp.allclose(derivative.func(jnp.asarray(2.0)), 4.0)
