#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _prepared_affine_rom():
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


def test_affine_rom_evaluator_propagates_reduced_state_and_validity():
    model = _prepared_affine_rom()
    level = phx.fidelity.FidelityLevelSpec(
        "rom",
        problem_id="linear-problem",
        observable_id="state",
        model_id=model.model_id,
        approximation_id="affine-galerkin",
        observable_contract_id="state-vector",
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


def test_affine_rom_evaluator_exposes_prepared_observation_without_reconstruction():
    model = _prepared_affine_rom()
    level = phx.fidelity.FidelityLevelSpec(
        "rom-qoi",
        problem_id="linear-problem",
        observable_id="qoi",
        model_id=model.model_id,
        approximation_id="affine-galerkin",
        observable_contract_id="scalar-qoi",
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
