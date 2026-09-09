import jax.numpy as jnp

from phydrax.finance.core import FinanceEvidenceBinding, PricingLaw
from phydrax.finance.valuation import (
    assess_tensor_valuation_candidate,
    deep_bsde_independent_validation,
    DeepBSDEApplicability,
    tensor_independent_validation,
    TensorCausalityEvidence,
    TensorSupportEvidence,
    TensorValuationApplicability,
)
from phydrax.solver._deep_bsde import DeepBSDEResult
from phydrax.stochastic import BSDEPathBatch
from phydrax.tensor_train import TensorizedGrid, TensorTrain
from phydrax.terms import DeepBSDERollout, DeepBSDEShootingDiagnostics


def _law():
    return PricingLaw(
        "candidate-q",
        "synthetic candidate law",
        "candidate-factors",
        "left-continuous-market-filtration",
        "candidate-measure",
        "candidate-numeraire",
        "candidate-collateral",
    )


def _binding():
    return FinanceEvidenceBinding(
        ("data-evidence",),
        ("model-evidence",),
        ("numerical-evidence",),
        ("use-evidence",),
    )


def _deep_result(path_id="validation-path"):
    paths = BSDEPathBatch(
        jnp.asarray([0.0, 1.0]),
        jnp.zeros((2, 2, 1)),
        jnp.zeros((2, 1, 1)),
        sample_shape=(2,),
        state_shape=(1,),
        noise_shape=(1,),
        path_id=path_id,
        process_id="candidate-process",
    )
    rollout = DeepBSDERollout(
        jnp.zeros((2, 1)),
        jnp.zeros((2, 2, 1)),
        jnp.zeros((2, 1, 1, 1)),
        jnp.zeros((2, 1, 1)),
        jnp.zeros((2, 1, 1)),
        jnp.zeros((2, 1)),
        jnp.zeros((2, 1)),
        jnp.ones((2,), dtype=bool),
        paths,
    )
    diagnostics = DeepBSDEShootingDiagnostics(
        jnp.asarray(0.0),
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(True),
    )
    return DeepBSDEResult(
        None,
        rollout,
        diagnostics,
        "candidate-problem",
        "candidate-process",
        "initial-value",
        "control",
    )


def _deep_applicability(training_path_id):
    return DeepBSDEApplicability(
        _law(),
        contract_id="candidate-contract",
        problem_id="candidate-problem",
        process_id="candidate-process",
        support_id="candidate-support",
        training_path_id=training_path_id,
        training_independence_id="training-cluster",
        max_validation_paths=8,
        max_time_steps=4,
        terminal_rmse_tolerance=1e-6,
        constraint_tolerance=1e-6,
    )


def test_deep_bsde_validation_cannot_be_relabelled_from_training():
    result = _deep_result()
    separated = deep_bsde_independent_validation(
        _deep_applicability("training-path"),
        result,
        validation_independence_id="validation-cluster",
        constraint_residual=0.0,
        baseline_error=0.0,
        baseline_id="analytic-baseline",
    )
    reused = deep_bsde_independent_validation(
        _deep_applicability("validation-path"),
        result,
        validation_independence_id="training-cluster",
        constraint_residual=0.0,
        baseline_error=0.0,
        baseline_id="analytic-baseline",
    )

    assert separated.independent
    assert bool(separated.valid)
    assert not reused.independent
    assert not bool(reused.valid)


def _tensor_fixture(dense, relative_tolerance):
    law = _law()
    grid = TensorizedGrid.uniform(((0.0, 1.0), (0.0, 1.0)), (2, 2))
    approximation = TensorTrain.from_dense(
        jnp.asarray(dense, dtype=float),
        max_ranks=2,
        relative_tolerance=relative_tolerance,
    )
    applicability = TensorValuationApplicability(
        law,
        grid,
        contract_id="tensor-contract",
        domain_id="tensor-domain",
        support_id="tensor-support",
        training_independence_id="tensor-training",
        route="tt",
        max_ranks=(2,),
        validation_tolerance=1e-5,
        reconstruction_tolerance=1e-5,
        maximum_core_bytes=4096,
        maximum_validation_points=4,
    )
    indices = jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    validation = tensor_independent_validation(
        applicability,
        approximation,
        indices,
        jnp.asarray(dense, dtype=float).reshape((-1,)),
        validation_id="tensor-validation",
        validation_independence_id="tensor-holdout",
    )
    support = TensorSupportEvidence(
        0.0,
        domain_id="tensor-domain",
        support_id="tensor-support",
        factor_layout_id=law.factor_layout_id,
        evidence_id="tensor-support-evidence",
        tolerance=0.0,
    )
    causality = TensorCausalityEvidence(
        0.0,
        filtration_id=law.filtration_id,
        evidence_id="tensor-causality-evidence",
        tolerance=0.0,
    )
    return law, applicability, approximation, validation, support, causality


def test_tensor_reconstruction_and_rank_evidence_fail_closed():
    dense = jnp.asarray([[1.0, 2.0], [2.0, 4.0]])
    law, applicability, approximation, validation, support, causality = _tensor_fixture(
        dense, 1e-6
    )
    accepted = assess_tensor_valuation_candidate(
        applicability,
        approximation,
        law,
        validation,
        support,
        causality,
        _binding(),
    )

    assert bool(accepted.accepted)
    assert not accepted.evidence.rank.rank_saturated
    assert jnp.allclose(accepted.reconstruct(max_entries=4), dense, atol=1e-5)

    missing_validation = assess_tensor_valuation_candidate(
        applicability,
        approximation,
        law,
        None,
        support,
        causality,
        _binding(),
    )
    mismatched_support = TensorSupportEvidence(
        0.0,
        domain_id="other-domain",
        support_id="tensor-support",
        factor_layout_id=law.factor_layout_id,
        evidence_id="other-support-evidence",
        tolerance=0.0,
    )
    domain_mismatch = assess_tensor_valuation_candidate(
        applicability,
        approximation,
        law,
        validation,
        mismatched_support,
        causality,
        _binding(),
    )
    assert not bool(missing_validation.accepted)
    assert not bool(domain_mismatch.accepted)

    saturated = TensorTrain.from_dense(jnp.eye(2), max_ranks=2, relative_tolerance=0.0)
    saturated_candidate = assess_tensor_valuation_candidate(
        applicability,
        saturated,
        law,
        tensor_independent_validation(
            applicability,
            saturated,
            jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
            jnp.eye(2).reshape((-1,)),
            validation_id="saturated-validation",
            validation_independence_id="saturated-holdout",
        ),
        support,
        causality,
        _binding(),
    )
    assert saturated_candidate.evidence.rank.rank_saturated
    assert not bool(saturated_candidate.accepted)
