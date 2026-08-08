import pytest

import phydrax as phx


PUBLIC_HIGH_DIMENSIONAL_API = {
    "equations": {
        "analyze_randomized_compilation",
        "CompiledRandomizedPDETerm",
        "compile_pde_randomized_term",
        "RandomizedCompilationReport",
        "RandomizedDifferentialMethod",
        "RandomizedDifferentialPlan",
        "RandomizedNodeCoupling",
    },
    "terms": {
        "BatchSampler",
        "deep_bsde_rollout",
        "deep_bsde_shooting_diagnostics",
        "DeepBSDEPredictor",
        "DeepBSDERollout",
        "DeepBSDESamplingMode",
        "DeepBSDEShootingDiagnostics",
        "DeepBSDEShootingTerm",
        "deep_splitting_labels",
        "DeepSplittingLabelBatch",
        "DeepSplittingLabelProvider",
        "DeepSplittingPredictor",
        "DeepSplittingRegressionDiagnostics",
        "DeepSplittingRegressionTerm",
        "FeynmanKacRegressionDiagnostics",
        "FeynmanKacRegressionTerm",
        "LabelProvider",
        "RandomizedResidualBatch",
        "RandomizedResidualDiagnostics",
        "RandomizedResidualLossMode",
        "RandomizedResidualTerm",
        "RandomizedResidualSamples",
        "RandomizedResidualSamplingMode",
        "ResidualEvaluator",
        "ScoreMatchingBatch",
        "ScoreMatchingDiagnostics",
        "ScoreMatchingMethod",
        "ScoreMatchingTerm",
        "ScoreMatchingPolicy",
        "ScoreMatchingSamplingMode",
        "ScoreSampleProvider",
    },
    "operators": {
        "coordinate_divergence_samples",
        "coordinate_second_derivative_samples",
        "DimensionOperatorEstimate",
        "DimensionOperatorSamples",
        "DimensionSamplingMode",
        "DimensionSamplingPolicy",
        "dimension_sum_samples",
        "estimate_dimension_sum",
        "stochastic_divergence_samples",
        "StochasticOperatorSamples",
        "stochastic_trace_samples",
    },
    "solver": {
        "DeepBSDEResult",
        "DeepSplittingDiagnostics",
        "DeepSplittingInterpolation",
        "DeepSplittingResult",
        "DeepSplittingSamplingMode",
        "DeepSplittingSolution",
        "DeepPicardDiagnostics",
        "DeepPicardInitialSource",
        "DeepPicardResult",
        "PicardSourceContext",
        "solve_deep_picard",
        "solve_deep_bsde",
        "solve_deep_splitting",
        "StructuredPicardSource",
        "StructuredSourceBuilder",
    },
    "stochastic": {
        "FeynmanKacControlTargetMode",
        "FeynmanKacLabelBatch",
        "feynman_kac_label_diagnostics",
        "FeynmanKacLabelDiagnostics",
        "FeynmanKacPathBatch",
        "FeynmanKacRefreshMode",
        "FeynmanKacSamplingMode",
        "FeynmanKacSamplingPlan",
        "FeynmanKacTimeWeighting",
        "query_feynman_kac_labels",
        "sample_feynman_kac_paths",
        "trajectory_node_feynman_kac_labels",
        "trajectory_state_time_measure",
        "trajectory_state_time_samples",
        "TrajectoryStateTimeMode",
        "TrajectoryStateTimeSamples",
    },
}


@pytest.mark.parametrize(
    ("namespace_name", "symbol"),
    [
        (namespace_name, symbol)
        for namespace_name, symbols in PUBLIC_HIGH_DIMENSIONAL_API.items()
        for symbol in sorted(symbols)
    ],
)
def test_high_dimensional_symbols_are_public(namespace_name, symbol):
    namespace = getattr(phx, namespace_name)

    assert getattr(namespace, symbol) is not None
    assert symbol in namespace.__all__


def test_dimension_estimators_are_public_from_differential_namespace():
    for symbol in PUBLIC_HIGH_DIMENSIONAL_API["operators"]:
        assert getattr(phx.operators.differential, symbol) is getattr(
            phx.operators, symbol
        )
        assert symbol in phx.operators.differential.__all__
