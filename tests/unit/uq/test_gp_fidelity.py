#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _model_and_state():
    low = phx.fidelity.FidelityLevelSpec(
        "low",
        problem_id="response",
        observable_id="qoi",
        model_id="coarse",
        approximation_id="coarse",
        observable_contract_id="scalar",
    )
    high = phx.fidelity.FidelityLevelSpec(
        "high",
        problem_id="response",
        observable_id="qoi",
        model_id="fine",
        approximation_id="fine",
        observable_contract_id="scalar",
    )
    hierarchy = phx.fidelity.FidelityHierarchy(
        (low, high),
        (phx.fidelity.FidelityRelation("low", "high"),),
        target_level_id="high",
    )
    path = hierarchy.linear_path()
    points = (-1.0, 0.0, 1.0)
    cases = tuple(
        phx.fidelity.FidelityCaseSpec(
            jnp.asarray([point]),
            case_id=f"case-{index}",
        )
        for index, point in enumerate(points)
    )
    evaluations = (
        phx.fidelity.FidelityEvaluation(
            jnp.asarray(-0.8),
            case_id="case-0",
            pair_id="case-0",
            level_id="low",
            evaluator_id="observations",
            valid=True,
            cost=1.0,
            cost_unit="relative",
        ),
        phx.fidelity.FidelityEvaluation(
            jnp.asarray(0.0),
            case_id="case-1",
            pair_id="case-1",
            level_id="low",
            evaluator_id="observations",
            valid=True,
            cost=1.0,
            cost_unit="relative",
        ),
        phx.fidelity.FidelityEvaluation(
            jnp.asarray(1.1),
            case_id="case-2",
            pair_id="case-2",
            level_id="high",
            evaluator_id="observations",
            valid=True,
            cost=100.0,
            cost_unit="relative",
        ),
    )
    dataset = phx.fidelity.FidelityDataset(hierarchy, cases, evaluations)
    model = phx.uq.FidelityGaussianProcess(path, dataset)
    kernel = phx.uq.AutoregressiveFidelityKernel(
        path,
        (
            phx.kernels.SquaredExponentialKernel(length_scale=0.7),
            phx.kernels.SquaredExponentialKernel(length_scale=0.7),
        ),
        transfer_coefficients=jnp.asarray([1.0]),
    )
    state = phx.uq.MultiOutputGaussianProcessLikelihoodState(
        kernel=kernel,
        noise_scale=jnp.asarray([0.03, 0.03]),
        jitter=1e-8,
    )
    return model, state


def test_autoregressive_fidelity_gp_is_psd_and_conditions_target():
    model, state = _model_and_state()
    covariance = state.kernel.matrix(model.design, model.design)
    assert jnp.min(jnp.linalg.eigvalsh(covariance)) > -1e-9

    condition = model.condition_target(jnp.asarray([[-0.5], [0.5]]), state=state)
    assert condition.mean.shape == (2,)
    assert condition.covariance.shape == (2, 2)
    assert jnp.all(condition.variance >= 0.0)
    assert jnp.all(jnp.isfinite(condition.mean))


def test_cost_aware_acquisition_selects_informative_low_fidelity(tmp_path):
    model, state = _model_and_state()
    policy = phx.uq.TargetVarianceAcquisitionPolicy(
        model.path,
        jnp.linspace(-1.0, 1.0, 9)[:, None],
        jnp.asarray([1.0, 100.0]),
        batch_size=1,
        cost_unit="relative",
    )
    result = phx.uq.select_fidelity_acquisition(
        model,
        state,
        jnp.asarray([[0.45], [0.45]]),
        ("low", "high"),
        policy,
    )

    assert result.selected_indices.tolist() == [0]
    assert result.selected_level_ids == ("low",)
    assert result.final_target_variance < result.initial_target_variance

    target_result = model.condition_target_result(
        jnp.asarray([[-0.25], [0.25]]),
        state=state,
    )
    path = tmp_path / "fidelity-gp.phx"
    phx.uq.export_result(target_result, path)
    archive = phx.uq.read_result_archive(path)
    assert archive.kind == "fidelity_gaussian_process"
    assert archive.metadata["target_level_id"] == "high"
