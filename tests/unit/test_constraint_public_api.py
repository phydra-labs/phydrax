import phydrax as phx


PUBLIC_TERM_API = {
    "AbstractEvaluatedScalarTerm",
    "AbstractFlowMatchingMetric",
    "AbstractSamplingTerm",
    "AbstractScalarTerm",
    "BSDETerm",
    "BarycenterObjectiveTerm",
    "BatchSampler",
    "CochainResidualTerm",
    "DeepBSDEPredictor",
    "DeepBSDERollout",
    "DeepBSDESamplingMode",
    "DeepBSDEShootingDiagnostics",
    "DeepBSDEShootingTerm",
    "DeepSplittingLabelBatch",
    "DeepSplittingLabelProvider",
    "DeepSplittingPredictor",
    "DeepSplittingRegressionDiagnostics",
    "DeepSplittingRegressionTerm",
    "DenseOverlapClassificationTerm",
    "DenseSiteClassificationBatch",
    "DenseSiteClassificationTerm",
    "DenoisingScoreDataProvider",
    "DenoisingScoreMatchingBatch",
    "DenoisingScoreMatchingDiagnostics",
    "DenoisingScoreMatchingTerm",
    "DenoisingScoreSamplingMode",
    "DenoisingScoreWeighting",
    "DifferentialPhysicsInformedOperatorTerm",
    "EmpiricalSinkhornDivergenceTerm",
    "EuclideanFlowMatchingMetric",
    "FeynmanKacRegressionDiagnostics",
    "FeynmanKacRegressionTerm",
    "AbstractFlowMatchingMetric",
    "EuclideanFlowMatchingMetric",
    "FlowEndpointProvider",
    "FlowMatchingBatch",
    "FlowMatchingDiagnostics",
    "UniformTimeSamplingPolicy",
    "FlowMatchingSamplingMode",
    "FlowMatchingTerm",
    "RiemannianFlowMatchingMetric",
    "GraphClassificationReduction",
    "GraphClassificationTarget",
    "GraphClassificationTargetEncoding",
    "GraphClassificationTerm",
    "GraphTrajectoryClassificationSignal",
    "GraphTrajectoryClassificationTerm",
    "GraphSupervisedTerm",
    "GraphTarget",
    "GraphTargetInterpolation",
    "GraphTrajectorySignal",
    "GraphTrajectorySupervisedTerm",
    "IntegralFunctional",
    "LabelProvider",
    "ManifoldFlowMatchingMetric",
    "MomentPenalty",
    "ObservationPenalty",
    "OperatorDatasetTerm",
    "PhysicsInformedOperatorTerm",
    "RaggedTimeSeriesClassificationBatch",
    "RaggedTimeSeriesClassificationTerm",
    "RaggedSeriesSupervisedBatch",
    "RaggedSeriesSupervisedTerm",
    "RaggedTimeSeriesBatch",
    "RaggedTimeSeriesDataTerm",
    "RaggedTimeSeriesInterpolation",
    "RandomizedMomentBatch",
    "RandomizedMomentDiagnostics",
    "RandomizedMomentPenalty",
    "RandomizedResidualBatch",
    "RandomizedResidualDiagnostics",
    "RandomizedResidualLossMode",
    "RandomizedResidualSamples",
    "RandomizedResidualSamplingMode",
    "RandomizedResidualTerm",
    "ResidualEvaluator",
    "ResidualPenalty",
    "RiemannianFlowMatchingMetric",
    "ScoreMatchingBatch",
    "ScoreMatchingDiagnostics",
    "ScoreMatchingMethod",
    "ScoreMatchingPolicy",
    "ScoreMatchingSamplingMode",
    "ScoreMatchingTerm",
    "ScoreSampleProvider",
    "SlicedWassersteinTerm",
    "SoftQuantileFunctional",
    "SpatialSinkhornDivergenceTerm",
    "SpatialUnbalancedSinkhornDivergenceTerm",
    "SupervisedClassificationTerm",
    "SupervisedFocalClassificationTerm",
    "SupervisedOrdinalClassificationTerm",
    "SupervisedSoftClassificationTerm",
    "SupervisedDatasetBatch",
    "SupervisedDatasetTerm",
    "SupervisedLikelihoodTerm",
    "TermEvaluation",
    "TrajectoryCaseClassificationBatch",
    "TrajectoryCaseClassificationTerm",
    "TrajectoryClassificationMeasure",
    "TrajectoryCaseDataBatch",
    "TrajectoryCaseDataTerm",
    "TrajectoryCaseTime",
    "TrajectorySignal",
    "TrajectorySignalInterpolation",
    "cochain_residual_field",
    "deep_bsde_rollout",
    "deep_bsde_shooting_diagnostics",
    "deep_splitting_labels",
    "evaluate",
    "operator_term_suite",
    "ricci_flat_kahler_term",
    "CompiledModalResidualTerm",
    "EigenspaceAction",
    "FactorizedVariationalEigenspaceResult",
    "FormDensity",
    "InvariantSubspaceResidual",
    "InvariantSubspaceResidualEvaluation",
    "InvariantSubspaceResidualResult",
    "ModalObservationTerm",
    "ModalTimeProvider",
    "VariationalEigenspace",
    "VariationalEigenspaceEvaluation",
    "VariationalEigenspaceResult",
    "factorized_variational_eigenspace",
    "free_boundary_term_suite",
    "implicit_interface_penalty",
    "implicit_phase_penalty",
    "AdversarialEvaluation",
    "EnergyTarget",
    "ImplicitGenerator",
    "FrozenTopologyTerm",
    "PersistentContrastiveDivergence",
    "PersistentEnergyState",
    "wasserstein_adversarial_evaluation",
}

LEGACY_TERMS = {
    "AbstractDomainSamplingTerm",
    "AbstractWeightedTerm",
    "ContinuousDirichletBoundaryTerm",
    "ContinuousFokkerPlanckTerm",
    "ContinuousInitialTerm",
    "ContinuousIntegralInteriorTerm",
    "ContinuousKolmogorovTerm",
    "ContinuousNeumannBoundaryTerm",
    "ContinuousODETerm",
    "ContinuousPointwiseInteriorTerm",
    "ContinuousRobinBoundaryTerm",
    "DiscreteDirichletBoundaryTerm",
    "DiscreteInitialTerm",
    "DiscreteInteriorDataTerm",
    "DiscreteNeumannBoundaryTerm",
    "DiscreteODETerm",
    "IntegralEqualityTerm",
    "PointSetTerm",
    "ResidualTerm",
}


def test_constraint_packages_are_the_only_root_constraint_surfaces():
    assert {"conditions", "terms", "enforcement"} <= set(phx.__all__)
    assert phx.conditions is not None
    assert phx.terms is not None
    assert phx.enforcement is not None
    assert "constraints" not in phx.__all__
    assert not hasattr(phx, "constraints")
    assert "objectives" not in phx.__all__
    assert not hasattr(phx, "objectives")


def test_conditions_expose_canonical_names_and_grouped_catalogs():
    required = {
        "Dirichlet",
        "Initial",
        "Moment",
        "Neumann",
        "Observation",
        "Residual",
        "Robin",
        "cfd",
        "conservation",
        "electromagnetics",
        "solids",
        "stochastic",
        "thermal",
    }
    legacy = {"MomentCondition", "ObservationCondition", "ResidualCondition"}

    assert required <= set(phx.conditions.__all__)
    assert all(getattr(phx.conditions, name) is not None for name in required)
    assert legacy.isdisjoint(phx.conditions.__all__)
    assert all(not hasattr(phx.conditions, name) for name in legacy)


def test_terms_public_surface_exactly_matches_the_supported_catalog():
    assert len(phx.terms.__all__) == len(set(phx.terms.__all__))
    assert set(phx.terms.__all__) == PUBLIC_TERM_API
    assert {
        name for name in vars(phx.terms) if not name.startswith("_")
    } == PUBLIC_TERM_API
    assert all(getattr(phx.terms, name) is not None for name in PUBLIC_TERM_API)


def test_terms_do_not_expose_legacy_or_duplicate_physics_catalogs():
    assert LEGACY_TERMS.isdisjoint(phx.terms.__all__)
    assert all(not hasattr(phx.terms, name) for name in LEGACY_TERMS)
    assert not any(
        name.startswith(("Continuous", "Discrete")) for name in phx.terms.__all__
    )


def test_collocation_policy_attachment_is_not_a_parallel_term_surface():
    assert "with_collocation_policy" not in phx.sampling.collocation.__all__
    assert not hasattr(phx.sampling.collocation, "with_collocation_policy")
