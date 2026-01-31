#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from .architectures._deeponet import DeepONet
from .architectures._feynmann import FeynmaNN
from .architectures._fno import FNO1d, FNO2d
from .architectures._kan import KAN
from .architectures._mlp import MLP
from .architectures._separable_feynmann import SeparableFeynmaNN
from .architectures._separable_mlp import SeparableMLP
from .embeddings._fourier import (
    RandomFourierFeatureEmbeddings,
)
from .layers._linear import Linear
from .wrappers._complex_output import ComplexOutputModel
from .wrappers._concatenated import ConcatenatedModel
from .wrappers._equinox import (
    EquinoxModel,
    EquinoxStructuredModel,
)
from .wrappers._magnitude_direction import (
    MagnitudeDirectionModel,
)
from .wrappers._separable_wrappers import (
    LatentContractionModel,
    Separable,
)


__all__ = [
    "ComplexOutputModel",
    "EquinoxModel",
    "EquinoxStructuredModel",
    "RandomFourierFeatureEmbeddings",
    "KAN",
    "Linear",
    "MLP",
    "ConcatenatedModel",
    "MagnitudeDirectionModel",
    "DeepONet",
    "SeparableMLP",
    "SeparableFeynmaNN",
    "FeynmaNN",
    "FNO1d",
    "FNO2d",
    "LatentContractionModel",
    "Separable",
]
