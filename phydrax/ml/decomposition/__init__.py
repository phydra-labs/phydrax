"""Native differentiable matrix and latent representation learning."""

from ._cross import (
    CCA,
    CCAModel,
    CrossDecompositionDiagnostics,
    PLS,
    PLSModel,
)
from ._factorization import (
    DictionaryLearning,
    FactorizationDiagnostics,
    NMF,
    NMFModel,
    SparseCoding,
    SparseCodingModel,
)
from ._incremental import (
    IncrementalPCA,
    IncrementalPCAModel,
)
from ._latent import (
    FactorAnalysis,
    FactorAnalysisModel,
    ICA,
    ICAModel,
    LatentDecompositionDiagnostics,
)
from ._physical_pod import PhysicalPODPlan, PhysicalPODResult
from ._subspace import (
    PCA,
    POD,
    SubspaceDiagnostics,
    SubspaceGradientTarget,
    SubspaceModel,
    TruncatedSVD,
)


__all__ = [
    "CCA",
    "CCAModel",
    "CrossDecompositionDiagnostics",
    "DictionaryLearning",
    "FactorAnalysis",
    "FactorAnalysisModel",
    "FactorizationDiagnostics",
    "ICA",
    "ICAModel",
    "IncrementalPCA",
    "IncrementalPCAModel",
    "LatentDecompositionDiagnostics",
    "NMF",
    "NMFModel",
    "PhysicalPODPlan",
    "PhysicalPODResult",
    "PCA",
    "PLS",
    "PLSModel",
    "POD",
    "SparseCoding",
    "SparseCodingModel",
    "SubspaceDiagnostics",
    "SubspaceGradientTarget",
    "SubspaceModel",
    "TruncatedSVD",
]
