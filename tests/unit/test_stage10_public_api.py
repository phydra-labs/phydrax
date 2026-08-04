import pytest

import phydrax as phx


PUBLIC_STAGE10 = {
    phx.stochastic: (
        "AtomicStochasticRealization",
        "CompositeStochasticRealization",
        "realization_independence_labels",
        "MartingaleProblem",
        "MartingaleIncrements",
        "StoppingIndices",
        "martingale_increments",
        "stopped_martingale_increments",
        "predictable_bracket_increments",
        "quadratic_covariation",
        "ObservationSequence",
        "AbstractStatePrior",
        "GaussianStatePrior",
        "CategoricalStatePrior",
        "DistributionStatePrior",
        "AbstractObservationModel",
        "CallableObservationModel",
        "GaussianObservationModel",
        "LinearGaussianObservationModel",
        "AbstractTransitionKernel",
        "CallableTransitionKernel",
        "MarginalTransitionKernel",
        "LinearGaussianTransitionKernel",
        "DifferentialTransitionKernel",
        "JumpTransitionKernel",
        "JumpDifferentialTransitionKernel",
        "FiniteStateTransitionKernel",
        "StateSpaceModel",
        "StateSpaceProblem",
        "state_space_key",
        "BSDEPathBatch",
        "BSDEProblem",
        "BSDEEvaluation",
        "evaluate_bsde",
        "autodiff_bsde_control",
        "semilinear_pde_residual",
        "JumpBSDEProblem",
        "JumpBSDEEvaluation",
        "evaluate_jump_bsde",
    ),
    phx.uq: (
        "martingale_diagnostics",
        "quadratic_variation_diagnostics",
        "jump_compensator_diagnostics",
        "martingale_validation_report",
        "initialize_kalman_filter",
        "kalman_filter_step",
        "kalman_filter",
        "rts_smoother",
        "sample_kalman_smoother_paths",
        "kalman_innovation_diagnostics",
        "initialize_particle_filter",
        "particle_filter_step",
        "bootstrap_particle_filter",
        "sample_particle_ancestry_paths",
        "sample_particle_backward_paths",
        "particle_filter_predictive",
        "particle_filter_diagnostics",
        "initialize_ensemble_filter",
        "ensemble_filter_step",
        "ensemble_transform_kalman_filter",
        "ensemble_kalman_smoother",
        "ensemble_filter_predictive",
        "ensemble_filter_diagnostics",
        "write_filter_checkpoint",
        "read_filter_checkpoint",
        "export_result",
        "read_result_archive",
    ),
    phx.solver: (
        "CoupledFBSDEProblem",
        "CoupledFBSDEResult",
        "solve_coupled_fbsde_explicit",
    ),
    phx.objectives: ("BSDEObjective",),
}


@pytest.mark.parametrize(
    ("namespace", "symbol"),
    [
        (namespace, symbol)
        for namespace, symbols in PUBLIC_STAGE10.items()
        for symbol in symbols
    ],
)
def test_stage10_symbols_are_public_and_declared(namespace, symbol):
    assert symbol in namespace.__all__
    assert getattr(namespace, symbol) is not None
