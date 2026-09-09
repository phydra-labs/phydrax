#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import optax

import phydrax as phx


def _poisson_dataset():
    low = phx.fidelity.FidelityLevelSpec(
        "low",
        problem_id="poisson",
        observable_id="u",
        model_id="biased-poisson",
        approximation_id="low",
        observable_contract_id="scalar-field",
    )
    high = phx.fidelity.FidelityLevelSpec(
        "high",
        problem_id="poisson",
        observable_id="u",
        model_id="target-poisson",
        approximation_id="high",
        observable_contract_id="scalar-field",
    )
    hierarchy = phx.fidelity.FidelityHierarchy(
        (low, high),
        (phx.fidelity.FidelityRelation("low", "high"),),
        target_level_id="high",
    )
    points = jnp.linspace(-1.0, 1.0, 15)
    cases = tuple(
        phx.fidelity.FidelityCaseSpec(
            {"x": point},
            case_id=f"case-{index}",
            split_group_id=f"physical-{index}",
        )
        for index, point in enumerate(points)
    )
    evaluations = []
    for index, point in enumerate(points):
        evaluations.append(
            phx.fidelity.FidelityEvaluation(
                0.8 * point * point,
                case_id=f"case-{index}",
                pair_id=f"pair-{index}",
                level_id="low",
                evaluator_id="poisson-observations",
                valid=True,
                cost=1.0,
                cost_unit="relative",
            )
        )
        if index % 2 == 0:
            evaluations.append(
                phx.fidelity.FidelityEvaluation(
                    point * point,
                    case_id=f"case-{index}",
                    pair_id=f"pair-{index}",
                    level_id="high",
                    evaluator_id="poisson-observations",
                    valid=True,
                    cost=20.0,
                    cost_unit="relative",
                )
            )
    return phx.fidelity.FidelityDataset(hierarchy, cases, tuple(evaluations))


def test_staged_multifidelity_pinn_improves_held_out_target_error():
    dataset = _poisson_dataset()
    split = phx.fidelity.split_fidelity_dataset(
        dataset,
        train_fraction=0.6,
        validation_fraction=0.2,
        seed=11,
        requirements=phx.fidelity.FidelitySplitRequirements(
            train={"low": 2, "high": 2},
            validation={"low": 1, "high": 1},
            test={"high": 1},
        ),
    )
    geometry = phx.domain.Interval1d(-1.0, 1.0)
    component = geometry.component()
    low_train = phx.terms.prepare_fidelity_observation_penalty(
        split.train,
        "low",
        field="u",
        component=component,
    )
    low_validation = phx.terms.prepare_fidelity_observation_penalty(
        split.validation,
        "low",
        field="u",
        component=component,
    )
    x_squared = geometry.Function("x")(lambda x: x * x)
    low_scale = geometry.Parameter(0.4)
    low_solver = phx.solver.FunctionalSolver(
        functions={"u": low_scale * x_squared},
        terms=(low_train.term,),
        evaluation_terms=(low_validation.term,),
    ).solve(
        num_iter=80,
        optim=optax.adam(0.08),
        seed=4,
        log_every=0,
    )
    parent = phx.solver.bind_fidelity_pinn_level(
        dataset.hierarchy.linear_path(),
        "low",
        low_solver,
        training_observations=(low_train,),
        validation_observations=(low_validation,),
    )
    high_train = phx.terms.prepare_fidelity_observation_penalty(
        split.train,
        "high",
        field="u",
        component=component,
    )
    high_validation = phx.terms.prepare_fidelity_observation_penalty(
        split.validation,
        "high",
        field="u",
        component=component,
    )
    high_test = phx.terms.prepare_fidelity_observation_penalty(
        split.test,
        "high",
        field="u",
        component=component,
    )
    pde_condition = phx.conditions.Residual(
        "u",
        component,
        lambda u: phx.operators.differential.laplacian(u, var="x") - 2.0,
        label="target-poisson",
    )
    pde_realization = phx.integration.materialize(
        phx.integration.mean_over(component),
        phx.domain.PointSampling(32),
        key=jr.key(7),
    )
    pde = phx.terms.ResidualPenalty(
        pde_condition,
        phx.integration.fixed(pde_realization),
    )
    correction_scale = geometry.Parameter(0.0)
    stage = phx.solver.prepare_fidelity_pinn_stage(
        parent,
        "high",
        {"u": correction_scale * x_squared},
        (pde,),
        training_observations=(high_train,),
        validation_observations=(high_validation,),
        epsilon=1.0,
    )
    trained = stage.training_solver.solve(
        num_iter=100,
        optim=optax.adam(0.06),
        seed=5,
        log_every=0,
    )
    result = stage.finalize(trained)
    evaluation = phx.solver.evaluate_fidelity_pinn(
        result,
        (high_test,),
        physics_terms=(pde,),
    )

    parent_prediction = parent.functions["u"](high_test.batch).data
    parent_rmse = jnp.sqrt(jnp.mean((parent_prediction - high_test.targets) ** 2))
    assert evaluation.target_data_rmse < parent_rmse
    assert evaluation.target_data_rmse < 0.03
    assert evaluation.physics_losses[0] < 0.03
    assert not set(result.training_group_ids) & set(result.validation_group_ids)
