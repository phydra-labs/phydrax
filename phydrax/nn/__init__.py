#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""
# Neural networks

Phydrax provides composable model components for PDE learning, including MLPs,
separable models, and latent contraction models over product domains.

## Highlights

- `MLP` and `Linear` for dense models.
- `Separable` and `SeparableMLP` for pointwise separable and coord-separable inputs.
- `LatentContractionModel` for product-domain factorization.
- `LatentExecutionPolicy` for structured execution fallback behavior.

!!! example
    ```python
    import jax
    import phydrax as phx

    model = phx.nn.MLP(in_size=2, out_size="scalar", width_size=32, depth=2, key=jax.random.key(0))
    y = model(jax.numpy.array([0.1, 0.2]))
    ```
"""

from . import (
    activations,
    models,
)

# Re-export objects from submodules
from .activations import (  # noqa: F401
    AdaptiveActivation,
    Stan,
)
from .models import (  # noqa: F401
    add_model_loss,
    ComplexOutputModel,
    ConcatenatedModel,
    DeepONet,
    EquinoxModel,
    EquinoxStructuredModel,
    ExplicitFourierFeatureEmbeddings,
    FeynmaNN,
    FNO1d,
    FNO2d,
    GraphModel,
    GraphRolloutModel,
    HybridFourierFeatureEmbeddings,
    KAN,
    LatentContractionModel,
    LatentExecutionPolicy,
    Linear,
    MagnitudeDirectionModel,
    MaskedSeriesPoolingModel,
    MLP,
    ModelWithLoss,
    ModifiedMLP,
    MultiscaleFourierFeatureEmbeddings,
    RaggedSeriesBatchInput,
    RaggedSeriesModel,
    RandomFourierFeatureEmbeddings,
    Separable,
    SeparableFeynmaNN,
    SeparableKAN,
    SeparableMLP,
    SeparableModifiedMLP,
    Sequential,
    TrainableFourierFeatureEmbeddings,
)


__all__ = [
    # subpackages
    "activations",
    "models",
    # activations exports
    "AdaptiveActivation",
    "Stan",
    # models exports
    "ComplexOutputModel",
    "ConcatenatedModel",
    "ExplicitFourierFeatureEmbeddings",
    "EquinoxModel",
    "EquinoxStructuredModel",
    "HybridFourierFeatureEmbeddings",
    "GraphModel",
    "GraphRolloutModel",
    "MagnitudeDirectionModel",
    "MaskedSeriesPoolingModel",
    "MultiscaleFourierFeatureEmbeddings",
    "RandomFourierFeatureEmbeddings",
    "TrainableFourierFeatureEmbeddings",
    "RaggedSeriesBatchInput",
    "RaggedSeriesModel",
    "SeparableMLP",
    "SeparableModifiedMLP",
    "SeparableKAN",
    "KAN",
    "Linear",
    "MLP",
    "ModifiedMLP",
    "ModelWithLoss",
    "FeynmaNN",
    "DeepONet",
    "FNO1d",
    "FNO2d",
    "Sequential",
    "LatentExecutionPolicy",
    "LatentContractionModel",
    "Separable",
    "SeparableFeynmaNN",
    "add_model_loss",
]
