# Layers

Low-level model building blocks.

!!! note
    Key notes:

    - `Linear` supports Random Weight Factorization (RWF) or one explicit
      shape-preserving physical weight transform.
    - `ComplexLinear` and `LowRankComplexLinear` keep trainable leaves real while
      evaluating exact complex-affine maps. The low-rank layer records spectral
      initializer truncation evidence and materializes its dense weight only on request.
    - `SpectralNeuron` evaluates one explicit zero-based ordered eigenvalue of a
      trainable affine real-symmetric matrix pencil. Increasing and decreasing
      coordinates use exact positive- and negative-semidefinite coefficients.
    - `Dropout(mode="feature")` shares one feature/channel mask over leading field axes.
    - Named LeCun, He/Kaiming, and Glorot/Xavier initializers follow JAX's
      post-truncation target-variance definitions; orthogonal initialization
      factors only the smaller rectangular orientation.
    - `inference_mode` switches every inference-aware leaf in mixed
      Phydrax/Equinox trees.
    - `AdaptiveResidual` starts exactly at the identity when `alpha=0`.
    - Recurrent cells consume a canonical `RecurrentBatch`; serial and associative
      execution share one reset and padding contract.
    - `MeasureNormalizedConvND` separates learned signed kernels from non-negative
      physical quadrature and observation masks.

`MeasureNormalizedConvND` keeps zero extension as its ordinary default and
offers an explicit circular mode for periodic same-size fields. Circular
extension requires an odd effective kernel, modularly wraps sanitized measured
values and masked measure—even when the halo exceeds the grid—and performs a
valid convolution with the full periodic stencil denominator. The target mask
is applied on the original unpadded grid. Circular latent extension must not be
used as a substitute for physical boundary-value enforcement.

::: phydrax.nn.layers.Linear
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.layers.ComplexLinear
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.layers.LowRankComplexLinear
    options:
        members:
            - __init__
            - __call__
            - materialize_weight

::: phydrax.nn.layers.LowRankComplexLinearInitializationReport

---

## Spectral neuron

`SpectralNeuron` evaluates the ascending eigenvalue selected by `eigen_index`
from `A(x) = A₀ + Σᵢ xᵢAᵢ`. The index is mandatory and zero-based. The layer
preserves leading axes, returns a scalar per input case, and exposes the full
ordered spectrum, evaluated pencil, and effective coefficient matrices without
exposing raw factor coordinates.

`monotonicity` is declared in flattened layer-input coordinates. An
`"increasing"` feature has `Aᵢ ⪰ 0`; a `"decreasing"` feature has `Aᵢ ⪯ 0`;
`"free"` imposes only real symmetry. The smallest selected eigenvalue is
globally concave, the largest is globally convex, and a one-dimensional matrix
is both. These guarantees apply to the layer coordinates, so bounds and
monotonicity must be rescaled explicitly when an upstream preprocessor changes
physical units.

Fresh initialization records a conservative selected-eigengap certificate over
the declared box `‖x‖∞ ≤ initialization_radius`. The report is initialization
evidence, not a persistent invariant after parameter updates. Dense evaluation
costs `O(nd² + d³)` per input case. At an eigenvalue crossing the forward value
remains defined, but a unique derivative does not.

All coefficient matrices are jointly invariant under a common orthogonal basis
change. Raw entries therefore are not basis-independent explanations; use the
cluster-projector inspection API instead. The construction follows
[arXiv:2608.08003](https://arxiv.org/abs/2608.08003).

::: phydrax.nn.layers.SpectralNeuron
    options:
        members:
            - __init__
            - __call__
            - matrix_pencil
            - eigenvalues
            - materialize_coefficients

::: phydrax.nn.layers.SpectralNeuronInitializationReport

---
::: phydrax.nn.layers.SineLayer
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.layers.AdaptiveResidual
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.layers.MeasureNormalizedConvND
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.layers.RecurrentBatch

---

::: phydrax.nn.layers.RecurrentResult

---

::: phydrax.nn.layers.AbstractRecurrentCell

---

::: phydrax.nn.layers.AffineRecurrence

---

::: phydrax.nn.layers.run_recurrent

---

::: phydrax.nn.layers.run_affine_recurrence

## Interface feature lift

`InterfaceFeatureLift` keeps an evolving interface out of downstream network
weights. It can append locally normalized signed distance, its absolute-value
cusp, and a compact side indicator to the original coordinates. Declare
`distance_semantics="signed_distance"` only when the supplied callable already
has that numerical contract.

::: phydrax.nn.layers.InterfaceFeatureLift

---

## Recurrent substrate

`RecurrentBatch` is the single packed-sequence contract. `valid` marks usable
samples, `reset` starts independent segments, and optional `time` stores physical
sample coordinates. Invalid samples preserve state and emit zero. Within one
packed batch, a later valid sample must declare a reset, and resets on invalid
samples are rejected; the runtime never guesses a segment boundary.
Low-level runners distinguish the streaming
`initial_state` entering a chunk from the canonical state used by resets; an
explicit `reset_state` overrides the latter when required.

`run_recurrent` executes every cell serially. Affine cells can instead use
`run_affine_recurrence`, whose serial and associative routes compose the same
transition monoid. Nonlinear cells may opt into `run_causal_recurrent`, which
uses the certified causal nonlinear solver while preserving the identical
padding, reset, explicit-key, initial-state, output, and continuation semantics.
`RNNCell`, `GRUCell`, `LSTMCell`, and `StackedRecurrentCell` support the adapter.

Causal execution is never selected automatically. It can require more work and
memory than `lax.scan`, especially for short sequences or wide states.
`CausalRecurrentConfig` makes nonconvergence either an error or an explicit
recorded serial fallback. A converged result uses the exact implicit recurrence
adjoint even when its forward direction used a quasi-Newton approximation.

`LinearRecurrentUnit` parameterizes stable complex-conjugate modes with real
input/output maps. `SelectiveStateSpaceBlock` combines reset-aware causal
convolution with input-dependent affine state transitions.
`WeightSpaceRecurrence` applies a diagonal stable recurrence to one explicit
parameter vector; it never materializes a dense parameter-by-parameter matrix.

::: phydrax.nn.layers.CausalRecurrentConfig

---

::: phydrax.nn.layers.CausalRecurrentResult

---

::: phydrax.nn.layers.run_causal_recurrent


::: phydrax.nn.layers.RNNCell

---

::: phydrax.nn.layers.GRUCell

---

::: phydrax.nn.layers.LSTMCell

---

::: phydrax.nn.layers.StackedRecurrentCell

---

::: phydrax.nn.layers.LinearRecurrentUnit
    options:
        members:
            - __init__
            - eigenvalues
            - initial_state
            - evaluate_with_state

---

::: phydrax.nn.layers.ResetAwareCausalConv1D
    options:
        members:
            - __init__
            - initial_state
            - evaluate_with_state
            - __call__

---

::: phydrax.nn.layers.SelectiveStateSpaceBlock
    options:
        members:
            - __init__
            - initial_state
            - evaluate_with_state
            - __call__

---

::: phydrax.nn.layers.SelectiveStateSpaceState

---

::: phydrax.nn.layers.WeightSpaceRecurrence
    options:
        members:
            - __init__
            - retention
            - initial_state
            - evaluate_with_state

---

::: phydrax.nn.layers.WeightSpaceState

---

---


::: phydrax.nn.layers.Dropout
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.layers.inference_mode

