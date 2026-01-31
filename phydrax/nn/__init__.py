#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""
# Neural networks

Phydrax provides composable model components for PDE learning, including MLPs,
separable models, and latent contraction models over product domains.

## Highlights

- `MLP` and `Linear` for dense models.
- `Separable` and `SeparableMLP` for coord-separable inputs.
- `LatentContractionModel` for product-domain factorization.

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
    ComplexOutputModel,
    ConcatenatedModel,
    DeepONet,
    EquinoxModel,
    EquinoxStructuredModel,
    FeynmaNN,
    FNO1d,
    FNO2d,
    KAN,
    LatentContractionModel,
    Linear,
    MagnitudeDirectionModel,
    MLP,
    RandomFourierFeatureEmbeddings,
    Separable,
    SeparableFeynmaNN,
    SeparableMLP,
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
    "EquinoxModel",
    "EquinoxStructuredModel",
    "MagnitudeDirectionModel",
    "RandomFourierFeatureEmbeddings",
    "SeparableMLP",
    "KAN",
    "Linear",
    "MLP",
    "FeynmaNN",
    "DeepONet",
    "FNO1d",
    "FNO2d",
    "LatentContractionModel",
    "Separable",
    "SeparableFeynmaNN",
]
