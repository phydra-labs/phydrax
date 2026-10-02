#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.ml import MLBatch
from phydrax.ml.decomposition import POD, SubspaceModel


def _prepared_affine_rom() -> Any:
    full = phx.linalg.ArraySpace((2,), dtype=jnp.float64, space_id="affine-test-full")
    basis = phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(
            full,
            jnp.eye(2, dtype=jnp.float64),
            orthonormal=True,
            subspace_id="affine-test-subspace",
        ),
        role="state",
        state_contract_id="two-component-state",
        support_id="fixed-support",
        measure_id="euclidean-measure",
        geometry_id="fixed-geometry",
        source_artifact_ids=("affine-test-snapshots",),
    )
    reduction = phx.rom.trial_test_reduction_from_bases(basis)
    dual = phx.linalg.DualSpace(full)
    operator = phx.linalg.DenseLinearOperator(
        jnp.eye(2, dtype=jnp.float64),
        source=full,
        target=dual,
        operator_id="identity-operator",
    )
    output = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="qoi-space")
    observation = phx.linalg.DenseLinearOperator(
        jnp.asarray([[0.0, 1.0]], dtype=jnp.float64),
        source=full,
        target=output,
        operator_id="second-component-qoi",
    )
    problem = phx.rom.AffineLinearROMProblem(
        reduction,
        (operator,),
        (
            jnp.asarray([1.0, 0.0], dtype=jnp.float64),
            jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        ),
        operator_term_ids=("identity",),
        right_hand_side_term_ids=("constant", "parameter"),
        observations=(("qoi", observation),),
        source_artifact_ids=("affine-test-operator-family",),
    )
    coefficients = phx.rom.ArrayAffineCoefficientMap(
        jnp.asarray([[0.0]], dtype=jnp.float64),
        jnp.asarray([1.0], dtype=jnp.float64),
        jnp.asarray([[0.0], [1.0]], dtype=jnp.float64),
        jnp.asarray([1.0, 0.0], dtype=jnp.float64),
        operator_term_ids=("identity",),
        right_hand_side_term_ids=("constant", "parameter"),
        lower=jnp.asarray([0.0], dtype=jnp.float64),
        upper=jnp.asarray([2.0], dtype=jnp.float64),
        input_contract_id="mu-array",
        unit_contract_id="dimensionless-mu",
        support_id="mu-in-zero-two",
    )
    return phx.rom.prepare_affine_linear_rom(problem, coefficients)


def test_affine_rom_contracts() -> None:
    model = _prepared_affine_rom()
    level = phx.fidelity.FidelityLevelSpec(
        "rom",
        problem_id="linear-problem",
        observable_id="state",
        model_id=model.model_id,
        approximation_id="affine-galerkin",
        observable_contract_id=model.reduction.trial_state_contract_id,
    )
    evaluator = phx.rom.AffineLinearROMFidelityEvaluator(
        model,
        level,
        cost=1.0,
        observable="state",
    )
    evaluation = evaluator(
        phx.fidelity.FidelityCaseSpec(
            jnp.asarray([1.0], dtype=jnp.float64),
            case_id="query",
        )
    )

    assert bool(evaluation.valid)
    np.testing.assert_allclose(evaluation.observable, np.asarray((1.0, 1.0)))
    assert evaluation.artifact_id == model.model_id

    unsupported = evaluator(
        phx.fidelity.FidelityCaseSpec(
            jnp.asarray([3.0], dtype=jnp.float64),
            case_id="unsupported",
        )
    )
    assert not bool(unsupported.valid)
    assert unsupported.result.solve_result is None
    model = _prepared_affine_rom()
    level = phx.fidelity.FidelityLevelSpec(
        "rom-qoi",
        problem_id="linear-problem",
        observable_id="qoi",
        model_id=model.model_id,
        approximation_id="affine-galerkin",
        observable_contract_id=model.observations[0].observation_id,
    )
    evaluation = phx.rom.AffineLinearROMFidelityEvaluator(
        model,
        level,
        cost=1.0,
        observable="qoi",
    )(
        phx.fidelity.FidelityCaseSpec(
            jnp.asarray([1.25], dtype=jnp.float64),
            case_id="qoi-query",
        )
    )

    np.testing.assert_allclose(evaluation.observable, np.asarray([1.25]))
    assert evaluation.result.reconstructed_state is None
    model = _prepared_affine_rom()
    wrong_observable = phx.fidelity.FidelityLevelSpec(
        "wrong-observable",
        problem_id="linear-problem",
        observable_id="qoi",
        model_id=model.model_id,
        approximation_id="affine-galerkin",
        observable_contract_id=model.observations[0].observation_id,
    )
    with pytest.raises(ValueError, match="observable_id"):
        phx.rom.AffineLinearROMFidelityEvaluator(
            model,
            wrong_observable,
            cost=1.0,
            observable="state",
        )

    wrong_contract = phx.fidelity.FidelityLevelSpec(
        "wrong-contract",
        problem_id="linear-problem",
        observable_id="state",
        model_id=model.model_id,
        approximation_id="affine-galerkin",
        observable_contract_id="another-state-contract",
    )
    with pytest.raises(ValueError, match="observable contract"):
        phx.rom.AffineLinearROMFidelityEvaluator(
            model,
            wrong_contract,
            cost=1.0,
            observable="state",
        )


def test_array_pod_rom_current_physical_basis_and_support() -> None:
    snapshots = jnp.array(
        [[-3.0, 0.2, 0.1], [-1.0, 1.0, -0.3], [1.0, -0.4, 0.2], [3.0, 0.1, 0.4]]
    )
    metric = jnp.array([0.5, 2.0, 3.0])
    model = (
        POD(1, physical_weights=metric, centered=True)
        .fit_batch(MLBatch(snapshots))
        .as_trainable()
    )
    assert isinstance(model, SubspaceModel)
    current = model.prediction_only(weighted_components=jnp.array([[0.0, 1.0, 0.0]]))
    artifact, offset = phx.rom.reduced_basis_from_subspace_model(
        current,
        role="state",
        state_contract_id="temperature",
        support_id="volume-support",
        measure_id="volume-measure",
        geometry_id="mesh-geometry",
        source_artifact_ids=("snapshots",),
    )
    assert artifact.state_contract_id == "temperature"
    assert artifact.support_id == "volume-support"
    assert artifact.measure_id == "volume-measure"
    assert artifact.geometry_id == "mesh-geometry"
    assert jnp.allclose(artifact.subspace.basis, current.components.T, atol=1e-12)
    assert jnp.allclose(offset, current.offset, atol=1e-12)
    excluded = (
        POD(1, physical_weights=jnp.array([1.0, 1.0, 0.0]))
        .fit_batch(MLBatch(snapshots))
        .as_trainable()
    )
    assert isinstance(excluded, SubspaceModel)
    with pytest.raises(ValueError, match="positive physical measure"):
        phx.rom.reduced_basis_from_subspace_model(
            excluded,
            role="state",
            state_contract_id="temperature",
            support_id="volume-support",
            measure_id="volume-measure",
            geometry_id="mesh-geometry",
            source_artifact_ids=("snapshots",),
        )
