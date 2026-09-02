#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure-aware deterministic, adaptive, and stochastic integration."""

from .._axis_factorization import (
    AxisContractionPlan,
    AxisContractionResult,
    AxisFactor,
    AxisFactorizedField,
    AxisGather,
    AxisProductTerm,
    contract_axis_factors,
)
from ._adaptive_callable import adaptive_interval_callable, adaptive_triangle_callable
from ._adaptive_cubature import adaptive_cubature_callable, integrate_adaptive_cubature
from ._adaptive_signed import (
    AdaptiveImportanceEstimator,
    AdaptiveSignedDiagnostics,
    AdaptiveSignedEstimator,
    AdaptiveSignedPopulation,
    AdaptiveStratifiedEstimator,
)
from ._api import from_samples, integrate, IntegrationRealization, materialize, reduce
from ._atlas import (
    AtlasIntegrationResult,
    AtlasIntegrationTarget,
    AtlasPatchQuadrature,
    integrate_atlas_scalar,
)
from ._batches import (
    IntegrationBatch,
    MappedIntegrationBatch,
    PointIntegrationBatch,
    SeparableIntegrationBatch,
    WeightedSampleBatch,
)
from ._bayesian_quadrature import GaussianKernelMean
from ._breakpoints import discover_breakpoints
from ._calabi_yau import (
    integrate_projective_samples,
    projective_measure_target,
    ProjectiveIntegralResult,
    ProjectiveMeasureKind,
    ProjectiveMeasureTarget,
)
from ._calibration import calibrate, MeasureCalibrationDiagnostics
from ._compression import compress, MeasureCompressionDiagnostics
from ._deformed_measure import (
    DeformedMeasureKind,
    DeformedMeasurePlan,
    DeformedMeasureState,
)
from ._diffrax_collocation import (
    DiffraxCollocationDiagnostics,
    DiffraxCollocationQuadraturePlan,
    integrate_diffrax_collocation,
    materialize_diffrax_collocation,
)
from ._discrete_support import spatial_measure
from ._empirical_cubature import (
    empirical_cubature,
    EmpiricalCubatureDiagnostics,
    EmpiricalCubaturePlan,
)
from ._estimates import (
    AdaptiveCubatureDiagnostics,
    AdaptiveCubaturePartition,
    AdaptivePartition,
    AdaptiveQuadratureDiagnostics,
    AdaptiveTriangleDiagnostics,
    AdaptiveTrianglePartition,
    AntitheticDiagnostics,
    BayesianQuadratureDiagnostics,
    DiscoveredBreakpoints,
    FixedQuadratureDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
    MappedIntegrationDiagnostics,
    MonteCarloDiagnostics,
    ProductIntegrationDiagnostics,
    RandomizedQMCDiagnostics,
    SparseGridDiagnostics,
    StratifiedDiagnostics,
    WeightedSampleDiagnostics,
)
from ._execution import (
    adaptive,
    AdaptiveIntegration,
    caller,
    CallerIntegration,
    fixed,
    FixedIntegration,
    IntegrationSource,
    per_step,
    PerStepIntegration,
    resolve_integration,
)
from ._factorized import (
    factorized_bilinear_form,
    factorized_inner_product,
    FactorizedBilinearEvaluation,
    FactorizedBilinearTerm,
)
from ._kernel_mean_bq import (
    FixedBayesianQuadratureDesign,
    prepare_kernel_mean_bayesian_quadrature,
    PreparedKernelMeanBayesianQuadrature,
    reduce_kernel_mean_bayesian_quadrature,
    SequentialBayesianQuadratureDesign,
)
from ._kernel_means import (
    AbstractKernelMean,
    FiniteFeatureKernelMean,
    FiniteMeasureKernelMean,
    IntervalKernelMean,
)
from ._markov import markov_chain_measure
from ._multilevel import (
    advance_multilevel,
    finalize_multilevel,
    initialize_multilevel,
    MLMCErrorLedger,
    MultilevelDiagnostics,
    MultilevelEstimatorState,
    MultilevelRealization,
    MultilevelResultArchive,
    MultilevelSampleBatch,
    read_multilevel_checkpoint,
    read_multilevel_result,
    write_multilevel_checkpoint,
    write_multilevel_result,
)
from ._plans import (
    AdaptiveCubaturePlan,
    AdaptiveQuadraturePlan,
    AdaptiveSparseGridPlan,
    AdaptiveTrianglePlan,
    AntitheticDesign,
    BayesianQuadraturePlan,
    BreakpointDiscoveryPlan,
    CellQuadraturePlan,
    ControlVariateEstimator,
    FixedQuadraturePlan,
    IIDDesign,
    ImportanceSamplingPlan,
    IntegrationPlan,
    LatinHypercubeDesign,
    MonteCarloPlan,
    MultilevelMonteCarloPlan,
    ProductIntegrationPlan,
    QuasiMonteCarloPlan,
    RandomizedQMCDesign,
    SampleMeanEstimator,
    SelfNormalizedEstimator,
    SparseGridPlan,
    StratifiedDesign,
    StratifiedMonteCarloPlan,
)
from ._precision import IntegrationPrecisionPolicy
from ._product import ProductIntegrationRealization
from ._riemannian import (
    MetricMeasureNormalization,
    normalize_metric_measure,
    riemannian_boundary_target,
)
from ._rules import (
    ClenshawCurtisRule,
    CubatureRule,
    GaussHermiteRule,
    GaussianCubatureRule,
    GaussKronrodRule,
    GaussLegendreRule,
    GaussLobattoLegendreRule,
    interval_rule_data,
    IntervalRule,
    ProbabilityRule,
    reference_rule_data,
    ReferenceCellData,
    ReferenceHexahedronRule,
    ReferenceIntervalRule,
    ReferencePrismRule,
    ReferencePyramidRule,
    ReferenceQuadrilateralRule,
    ReferenceRule,
    ReferenceTetrahedronRule,
    ReferenceTriangleRule,
    TanhSinhRule,
)
from ._sparse_grid import (
    AdaptiveSparseGridDiagnostics,
    AdaptiveSparseGridResult,
    prepare_adaptive_sparse_grid,
    SparseGridRealization,
)
from ._status import IntegrationStatus, status_message
from ._targets import (
    ComponentTarget,
    density,
    DensityTarget,
    discrete,
    DiscreteMeasureTarget,
    expectation,
    IntegrationTarget,
    mapped,
    MappedTarget,
    mean_over,
    multilevel,
    MultilevelSampler,
    MultilevelTarget,
    normalized_density,
    over,
    ProbabilityTarget,
    weighted,
    WeightedSampleTarget,
)
from ._transformations import (
    MeasureTransformationRecord,
    TransformedIntegrationDiagnostics,
)


_SPLITTING_EXPORTS = frozenset(
    {
        "adaptive_multilevel_splitting",
        "AdaptiveMultilevelSplittingPlan",
        "AdaptiveMultilevelSplittingResult",
        "AdaptiveSplittingBranchRequest",
        "AdaptiveSplittingDiagnostics",
        "AdaptiveSplittingEnsembleResult",
        "AdaptiveSplittingStatus",
        "InitialPathSampler",
        "PathBranchSampler",
        "replicate_adaptive_multilevel_splitting",
    }
)


def __getattr__(name: str):
    if name in _SPLITTING_EXPORTS:
        from . import _splitting as module

        exports = {
            "adaptive_multilevel_splitting": module.adaptive_multilevel_splitting,
            "AdaptiveMultilevelSplittingPlan": module.AdaptiveMultilevelSplittingPlan,
            "AdaptiveMultilevelSplittingResult": module.AdaptiveMultilevelSplittingResult,
            "AdaptiveSplittingBranchRequest": module.AdaptiveSplittingBranchRequest,
            "AdaptiveSplittingDiagnostics": module.AdaptiveSplittingDiagnostics,
            "AdaptiveSplittingEnsembleResult": module.AdaptiveSplittingEnsembleResult,
            "AdaptiveSplittingStatus": module.AdaptiveSplittingStatus,
            "InitialPathSampler": module.InitialPathSampler,
            "PathBranchSampler": module.PathBranchSampler,
            "replicate_adaptive_multilevel_splitting": module.replicate_adaptive_multilevel_splitting,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_SURROGATE_EXPORTS = frozenset(
    {
        "smolyak_surrogate_expectation",
        "SmolyakInputSampler",
        "SmolyakProbabilityInputSampler",
        "SmolyakSurrogateHierarchyAdapter",
    }
)


_splitting_getattr = __getattr__


def __getattr__(name: str):
    if name in _SURROGATE_EXPORTS:
        from . import _surrogate as module

        exports = {
            "smolyak_surrogate_expectation": module.smolyak_surrogate_expectation,
            "SmolyakInputSampler": module.SmolyakInputSampler,
            "SmolyakProbabilityInputSampler": module.SmolyakProbabilityInputSampler,
            "SmolyakSurrogateHierarchyAdapter": module.SmolyakSurrogateHierarchyAdapter,
        }
        return exports[name]
    return _splitting_getattr(name)


__all__ = [
    "AxisContractionPlan",
    "AxisContractionResult",
    "AxisFactor",
    "AxisFactorizedField",
    "AxisGather",
    "AxisProductTerm",
    "AtlasIntegrationResult",
    "AtlasIntegrationTarget",
    "AtlasPatchQuadrature",
    "integrate_atlas_scalar",
    "AdaptivePartition",
    "adaptive_cubature_callable",
    "AdaptiveCubatureDiagnostics",
    "AdaptiveCubaturePartition",
    "AdaptiveCubaturePlan",
    "AdaptiveImportanceEstimator",
    "adaptive_interval_callable",
    "AdaptiveIntegration",
    "AdaptiveSignedDiagnostics",
    "AdaptiveSignedEstimator",
    "AdaptiveSignedPopulation",
    "AdaptiveSparseGridDiagnostics",
    "AdaptiveSparseGridPlan",
    "AdaptiveSparseGridResult",
    "AdaptiveStratifiedEstimator",
    "AdaptiveQuadratureDiagnostics",
    "AdaptiveQuadraturePlan",
    "AdaptiveTriangleDiagnostics",
    "AdaptiveTrianglePartition",
    "AdaptiveTrianglePlan",
    "adaptive_triangle_callable",
    "AdaptiveMultilevelSplittingPlan",
    "AdaptiveMultilevelSplittingResult",
    "adaptive_multilevel_splitting",
    "AdaptiveSplittingBranchRequest",
    "AdaptiveSplittingDiagnostics",
    "AdaptiveSplittingStatus",
    "AdaptiveSplittingEnsembleResult",
    "ProjectiveIntegralResult",
    "ProjectiveMeasureKind",
    "ProjectiveMeasureTarget",
    "integrate_projective_samples",
    "projective_measure_target",
    "advance_multilevel",
    "finalize_multilevel",
    "initialize_multilevel",
    "AntitheticDesign",
    "AntitheticDiagnostics",
    "BayesianQuadratureDiagnostics",
    "BayesianQuadraturePlan",
    "BreakpointDiscoveryPlan",
    "CellQuadraturePlan",
    "CallerIntegration",
    "ClenshawCurtisRule",
    "CubatureRule",
    "ComponentTarget",
    "ControlVariateEstimator",
    "DensityTarget",
    "DiscreteMeasureTarget",
    "DiscoveredBreakpoints",
    "DiffraxCollocationDiagnostics",
    "DiffraxCollocationQuadraturePlan",
    "FactorizedBilinearEvaluation",
    "FactorizedBilinearTerm",
    "FixedQuadratureDiagnostics",
    "FixedIntegration",
    "FixedQuadraturePlan",
    "GaussKronrodRule",
    "GaussianCubatureRule",
    "GaussHermiteRule",
    "GaussLegendreRule",
    "GaussianKernelMean",
    "GaussLobattoLegendreRule",
    "IIDDesign",
    "InitialPathSampler",
    "LatinHypercubeDesign",
    "WeightedSampleDiagnostics",
    "ImportanceSamplingPlan",
    "IntegrationBatch",
    "IntegrationEstimate",
    "contract_axis_factors",
    "factorized_bilinear_form",
    "factorized_inner_product",
    "IntegrationPlan",
    "IntegrationPrecisionPolicy",
    "MeasureCalibrationDiagnostics",
    "MeasureCompressionDiagnostics",
    "MeasureTransformationRecord",
    "IntegrationProvenance",
    "IntegrationStatus",
    "DeformedMeasureKind",
    "DeformedMeasurePlan",
    "DeformedMeasureState",
    "IntegrationRealization",
    "IntegrationSource",
    "IntegrationTarget",
    "IntervalRule",
    "ProbabilityRule",
    "MappedIntegrationBatch",
    "MappedIntegrationDiagnostics",
    "MappedTarget",
    "MetricMeasureNormalization",
    "normalize_metric_measure",
    "riemannian_boundary_target",
    "ReferenceCellData",
    "ReferenceHexahedronRule",
    "ReferenceIntervalRule",
    "ReferenceQuadrilateralRule",
    "MonteCarloDiagnostics",
    "MonteCarloPlan",
    "MLMCErrorLedger",
    "MultilevelDiagnostics",
    "MultilevelEstimatorState",
    "MultilevelMonteCarloPlan",
    "MultilevelRealization",
    "MultilevelResultArchive",
    "MultilevelSampleBatch",
    "MultilevelSampler",
    "MultilevelTarget",
    "PointIntegrationBatch",
    "PathBranchSampler",
    "ProbabilityTarget",
    "ProductIntegrationDiagnostics",
    "ProductIntegrationPlan",
    "ProductIntegrationRealization",
    "PerStepIntegration",
    "QuasiMonteCarloPlan",
    "RandomizedQMCDesign",
    "RandomizedQMCDiagnostics",
    "ReferenceRule",
    "ReferencePrismRule",
    "ReferencePyramidRule",
    "ReferenceTetrahedronRule",
    "ReferenceTriangleRule",
    "SampleMeanEstimator",
    "SelfNormalizedEstimator",
    "SeparableIntegrationBatch",
    "SmolyakInputSampler",
    "SmolyakProbabilityInputSampler",
    "smolyak_surrogate_expectation",
    "SmolyakSurrogateHierarchyAdapter",
    "SparseGridDiagnostics",
    "SparseGridPlan",
    "SparseGridRealization",
    "spatial_measure",
    "StratifiedDesign",
    "StratifiedDiagnostics",
    "StratifiedMonteCarloPlan",
    "TanhSinhRule",
    "TransformedIntegrationDiagnostics",
    "WeightedSampleBatch",
    "WeightedSampleTarget",
    "adaptive",
    "caller",
    "calibrate",
    "compress",
    "discrete",
    "density",
    "fixed",
    "discover_breakpoints",
    "from_samples",
    "expectation",
    "multilevel",
    "read_multilevel_checkpoint",
    "read_multilevel_result",
    "replicate_adaptive_multilevel_splitting",
    "integrate",
    "integrate_adaptive_cubature",
    "integrate_diffrax_collocation",
    "interval_rule_data",
    "markov_chain_measure",
    "mapped",
    "materialize",
    "mean_over",
    "per_step",
    "resolve_integration",
    "prepare_adaptive_sparse_grid",
    "materialize_diffrax_collocation",
    "normalized_density",
    "over",
    "reduce",
    "reference_rule_data",
    "status_message",
    "write_multilevel_checkpoint",
    "write_multilevel_result",
    "weighted",
    "AbstractKernelMean",
    "FiniteFeatureKernelMean",
    "FiniteMeasureKernelMean",
    "IntervalKernelMean",
    "FixedBayesianQuadratureDesign",
    "SequentialBayesianQuadratureDesign",
    "PreparedKernelMeanBayesianQuadrature",
    "prepare_kernel_mean_bayesian_quadrature",
    "reduce_kernel_mean_bayesian_quadrature",
    "EmpiricalCubatureDiagnostics",
    "EmpiricalCubaturePlan",
    "empirical_cubature",
]
