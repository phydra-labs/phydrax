# Layers

Low-level model building blocks.

!!! note
    Key notes:

    - `Linear` supports Random Weight Factorization (RWF) or one explicit
      shape-preserving physical weight transform.
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
::: phydrax.nn.layers.Linear
    options:
        members:
            - __init__
            - __call__

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

