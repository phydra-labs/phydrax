# Embeddings

Input feature maps for coordinate-based learning. All Fourier embeddings use
angular wavevectors and emit cosine features followed by sine features. Selected
raw coordinates and a constant feature may be appended when a problem combines
periodic and nonperiodic inputs.

## Choosing a Fourier basis

Use the narrowest spectral prior justified by the problem:

1. `ExplicitFourierFeatureEmbeddings` for known periods, forcing frequencies,
   eigenmodes, or dispersion relations.
2. `MultiscaleFourierFeatureEmbeddings` for deterministic broadband coverage.
3. `HybridFourierFeatureEmbeddings` for a guaranteed deterministic core with a
   random exploratory tail.
4. `RandomFourierFeatureEmbeddings` when the spectrum is unknown or the input
   dimension makes deterministic coverage impractical.
5. `TrainableFourierFeatureEmbeddings` for experimental unrestricted frequency
   learning. High-order PDE derivatives can make this option poorly conditioned.

Fixed embeddings stop gradients through wavevectors and phases. The trainable
embedding leaves wavevector gradients enabled while keeping phases fixed.

## Explicit and periodic features

```python
import phydrax as phx

embedding = phx.nn.ExplicitFourierFeatureEmbeddings.from_periodic_modes(
    in_size=2,
    coordinate=0,
    period=2.0,
    modes=range(1, 11),
    passthrough=(1,),
    include_constant=True,
)
```

The example encodes the first coordinate with ten exact harmonics, passes the
second coordinate through unchanged, and appends a constant feature.

::: phydrax.nn.ExplicitFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - from_periodic_modes
            - __call__

::: phydrax.nn.MultiscaleFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - __call__

::: phydrax.nn.HybridFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - __call__

::: phydrax.nn.RandomFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - __call__

::: phydrax.nn.TrainableFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - __call__
