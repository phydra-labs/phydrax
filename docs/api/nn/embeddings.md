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

embedding = phx.nn.layers.ExplicitFourierFeatureEmbeddings.from_periodic_modes(
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

## Certified periodic construction

`from_periodic_modes` certifies `coordinate` as exactly periodic with `period`;
the constructor accepts the same declaration as
`periodic_inputs={flat_index: period}`. A declaration is accepted only when every
wavevector component of that input equals the canonical float64 representation
`2*pi*n/period` for an exactly representable integer `n`. Near-integer
wavevectors are not exact evidence and are refused, rather than rounded onto
the lattice. Modes must be distinct positive integers no greater than `2**53`;
the resulting frequencies and all certified feature data must remain finite.
The guarantee is a construction identity, not bitwise equality of floating-point
trigonometric evaluations (especially at very large arguments). Modes sharing
a common divisor `g > 1` remain certified for `period`, but they can only
represent functions of period `period / g`.

A certified input cannot also be passed through: a raw coordinate feature is
not periodic, so `passthrough` containing a certified index is refused. Other
inputs may be passed through freely.

Replacing certified fixed frequency leaves or introducing nonfinite feature
data causes execution to fail, including under JIT, rather than executing with
a stale certificate. Finite changes to uncertified frequency columns preserve
the declared periodic directions.

The embedding attaches a `phx.nn.PeriodicInputCertificate` to a bound field
under the `"periodic_input_certificate"` metadata key. The certificate lists
`(flat_index, period)` pairs and the model's declared regularity, which bounds
the derivative orders whose periodicity it claims (`supports_order`).
`phx.nn.models.Sequential` whose first stage is a certified embedding inherits
the certificate with the composite's regularity, for example `C^0` behind a
ReLU MLP. A composite with undeclared regularity or unbound stochastic
realizations carries no certificate. In particular, active dropout must be
disabled with `eqx.nn.inference_mode` before certifying the complete field;
switching back to active dropout removes the current evidence. Random,
trainable, multiscale, and hybrid embeddings never certify periodicity.

Bound certificates require flat, pointwise packing with an input size matching
the dense domain dependencies in their declared order. `Sequential` honors
each pointwise stage's key and iteration invocation contract; array outputs do
not implicitly unpack into downstream structured-input stages. User metadata
may be added with `with_metadata`, but existing source-bound evidence cannot
be replaced there. Rebind a new model to change its certificate.

```python
periodic = phx.nn.layers.ExplicitFourierFeatureEmbeddings.from_periodic_modes(
    in_size=1, coordinate=0, period=2.0, modes=range(1, 9)
)
mlp = phx.nn.models.MLP(
    in_size=periodic.out_size, out_size="scalar", width_size=32, depth=2
)
model = phx.nn.models.Sequential((periodic, mlp))
u = phx.domain.Interval1d(0.0, 2.0).Model("x")(model)
certificate = u.metadata["periodic_input_certificate"]
```

Arithmetic, `phx.operators.pullback`, and the coordinate differential operators
(`grad`, `partial_x`, `laplacian`, ...) drop the certificate because they do
not preserve the certified construction claim.
Hard periodic enforcement consumes it through
`phx.enforcement.prepare_periodic_projection(..., route="construction")`, which
maps each identified coordinate to its flat model input index in the field's
dependency order.

::: phydrax.nn.layers.ExplicitFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - from_periodic_modes
            - __call__

::: phydrax.nn.PeriodicInputCertificate
    options:
        members:
            - period_of
            - supports_order

::: phydrax.nn.layers.MultiscaleFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - __call__

::: phydrax.nn.layers.HybridFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - __call__

::: phydrax.nn.layers.RandomFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - __call__

::: phydrax.nn.layers.TrainableFourierFeatureEmbeddings
    options:
        members:
            - __init__
            - __call__
