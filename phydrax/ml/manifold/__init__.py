"""Native differentiable manifold learning with explicit hard topology contracts."""

from ._common import build_neighbor_graph, ManifoldDiagnostics, NeighborhoodGraph
from ._lle import (
    LLEVariant,
    LocallyLinearEmbeddingModel,
    LocallyLinearEmbeddingRecipe,
)
from ._spectral import (
    IsomapModel,
    IsomapRecipe,
    MDSMethod,
    MultidimensionalScalingModel,
    MultidimensionalScalingRecipe,
    SpectralEmbeddingModel,
    SpectralEmbeddingRecipe,
)
from ._stochastic import (
    FuzzyGraphEmbeddingModel,
    FuzzyGraphEmbeddingRecipe,
    TSNEModel,
    TSNERecipe,
)


__all__ = [
    "FuzzyGraphEmbeddingModel",
    "FuzzyGraphEmbeddingRecipe",
    "IsomapModel",
    "IsomapRecipe",
    "LLEVariant",
    "LocallyLinearEmbeddingModel",
    "LocallyLinearEmbeddingRecipe",
    "MDSMethod",
    "ManifoldDiagnostics",
    "MultidimensionalScalingModel",
    "MultidimensionalScalingRecipe",
    "NeighborhoodGraph",
    "SpectralEmbeddingModel",
    "SpectralEmbeddingRecipe",
    "TSNEModel",
    "TSNERecipe",
    "build_neighbor_graph",
]
