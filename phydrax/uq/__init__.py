#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native uncertainty-quantification tools for Phydrax."""

from ._checkpoint import (
    CheckpointCompatibilityError,
    CheckpointCorruptionError,
    CheckpointError,
)
from ._conformal import FunctionalConformal, NormalizedConformal, SplitConformal
from ._diagnostics import (
    MCMCConvergenceError,
    MCMCConvergenceReport,
    MCMCConvergenceThresholds,
    MCMCDiagnostics,
)
from ._discrepancy import (
    ExactGaussianProcessDiscrepancy,
    ExactGaussianProcessFactor,
    GaussianProcessCondition,
    GaussianProcessConditioner,
    MultiOutputGaussianProcessCondition,
    MultiOutputGaussianProcessDiscrepancy,
    SparseGaussianProcessDiscrepancy,
    SparseGaussianProcessFactor,
)
from ._discrepancy_diagnostics import (
    discrepancy_identifiability_report,
    DiscrepancyIdentifiabilityReport,
    DiscrepancyIdentifiabilityThresholds,
)
from ._distributions import (
    AbstractDistribution,
    EmpiricalDistribution,
    LogNormal,
    Normal,
    Uniform,
)
from ._eki import (
    EnsembleKalmanConvergenceError,
    EnsembleKalmanDiagnostics,
    EnsembleKalmanResult,
    fit_eki,
)
from ._ensemble import (
    EnsembleFitError,
    EnsembleFitResult,
    EnsembleMemberDiagnostics,
    fit_ensemble,
    FrozenModel,
    HeterogeneousFunctionEnsemble,
    HomogeneousFunctionEnsemble,
    randomized_prior_ensemble,
    RandomizedPriorModel,
)
from ._laplace import fit_laplace, LaplaceCurvatureError, LaplaceResult
from ._laplax_backend import StructuredLaplaceResult
from ._likelihoods import (
    AbstractLikelihood,
    GaussianLikelihood,
    GaussianLocationScaleLikelihood,
    StudentTLikelihood,
)
from ._map import find_map, MAPConvergenceError, MAPResult
from ._mcmc import MCMCChainWarmup, MCMCResult, sample_hmc, sample_nuts
from ._metrics import (
    calibration_error,
    energy_score,
    ensemble_crps,
    gaussian_crps,
    GaussianScaleCalibrator,
    interval_coverage,
    interval_width,
    negative_log_likelihood,
    pinball_loss,
    student_t_crps,
)
from ._pathfinder import fit_pathfinder, PathfinderResult
from ._posterior import (
    AbstractBijector,
    ExpBijector,
    IdentityBijector,
    ParameterSpace,
    ParameterSubspace,
    PosteriorProblem,
    SigmoidIntervalBijector,
)
from ._posterior_terms import (
    AbstractPosteriorTerm,
    CompositePosteriorLikelihood,
    FixedConstraintLikelihood,
    FixedObservationLikelihood,
    FixedResidualLikelihood,
    GaussianProcessMarginalLikelihood,
)
from ._predictive import (
    PredictionInterval,
    PredictiveField,
    SampleAxis,
    UncertaintySource,
)
from ._propagation import propagate, RandomSampleBatch, sample_joint
from ._result_export import (
    decode_parameter_name,
    encode_parameter_name,
    export_result,
    read_result_archive,
    to_arviz,
    UQResultArchive,
)
from ._sensitivity import sobol_indices, SobolResult
from ._smc import sample_tempered_smc, TemperedSMCResult
from ._whitening import GaussianPriorWhitening


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointCorruptionError",
    "CheckpointError",
    "sobol_indices",
    "SobolResult",
    "AbstractDistribution",
    "EmpiricalDistribution",
    "LogNormal",
    "Normal",
    "Uniform",
    "propagate",
    "RandomSampleBatch",
    "sample_joint",
    "FunctionalConformal",
    "NormalizedConformal",
    "SplitConformal",
    "EnsembleKalmanConvergenceError",
    "EnsembleKalmanDiagnostics",
    "EnsembleKalmanResult",
    "fit_eki",
    "EnsembleFitError",
    "EnsembleFitResult",
    "EnsembleMemberDiagnostics",
    "fit_ensemble",
    "FrozenModel",
    "HeterogeneousFunctionEnsemble",
    "HomogeneousFunctionEnsemble",
    "randomized_prior_ensemble",
    "RandomizedPriorModel",
    "AbstractLikelihood",
    "GaussianLikelihood",
    "GaussianLocationScaleLikelihood",
    "StudentTLikelihood",
    "GaussianScaleCalibrator",
    "calibration_error",
    "energy_score",
    "ensemble_crps",
    "gaussian_crps",
    "interval_coverage",
    "interval_width",
    "negative_log_likelihood",
    "pinball_loss",
    "student_t_crps",
    "PredictionInterval",
    "PredictiveField",
    "SampleAxis",
    "UncertaintySource",
    "AbstractBijector",
    "ExpBijector",
    "IdentityBijector",
    "SigmoidIntervalBijector",
    "ParameterSpace",
    "ParameterSubspace",
    "PosteriorProblem",
    "GaussianPriorWhitening",
    "AbstractPosteriorTerm",
    "CompositePosteriorLikelihood",
    "FixedConstraintLikelihood",
    "FixedObservationLikelihood",
    "FixedResidualLikelihood",
    "GaussianProcessMarginalLikelihood",
    "find_map",
    "MAPConvergenceError",
    "MAPResult",
    "MCMCChainWarmup",
    "MCMCConvergenceError",
    "MCMCConvergenceReport",
    "MCMCConvergenceThresholds",
    "MCMCDiagnostics",
    "MCMCResult",
    "sample_hmc",
    "sample_nuts",
    "PathfinderResult",
    "fit_pathfinder",
    "TemperedSMCResult",
    "sample_tempered_smc",
    "LaplaceCurvatureError",
    "LaplaceResult",
    "StructuredLaplaceResult",
    "fit_laplace",
    "ExactGaussianProcessDiscrepancy",
    "ExactGaussianProcessFactor",
    "GaussianProcessCondition",
    "GaussianProcessConditioner",
    "MultiOutputGaussianProcessCondition",
    "MultiOutputGaussianProcessDiscrepancy",
    "SparseGaussianProcessDiscrepancy",
    "SparseGaussianProcessFactor",
    "DiscrepancyIdentifiabilityReport",
    "DiscrepancyIdentifiabilityThresholds",
    "discrepancy_identifiability_report",
    "UQResultArchive",
    "decode_parameter_name",
    "encode_parameter_name",
    "export_result",
    "read_result_archive",
    "to_arviz",
]
