# Architectures

Common end-to-end model families (dense, separable, polynomial, and complex-valued).

!!! note
    Key notes:

    - `MLP` is a standard feed-forward network with optional residual connection.
    - `KAN` replaces activations with polynomial edge functions.
    - `FeynmaNN` builds complex hidden states with a sum-over-paths block.
    - `MLP`, `KAN`, `FeynmaNN`, `FNO1d`, and `FNO2d` support `scan=True` to use
      a scan-over-depth execution path when topology is compatible.
    - `scan=True` is primarily a compile-time optimization for deeper repeated blocks.

::: phydrax.nn.MLP
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.KAN
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.FeynmaNN
    options:
        members:
            - __init__
            - __call__

---

## Operator networks

Operator-learning models typically consume **structured inputs** from product domains, e.g.
\((\text{data}, x)\) for a dataset factor and a spatial geometry.

Phydrax's `Domain.Model(...)` wrapper will pass dependency arguments to models as either:

- **point inputs** (dense sampling): coordinate arrays with leading batch axes, or
- **coord-separable inputs**: tuples of 1D coordinate axes (from `CoordSeparableBatch`).

`supports_structured_input()` and default domain packing are distinct. Dense models
and separable coordinate models such as `SeparableMLP(in_size=d)` default to flat
point packing, so `domain.Model("x", "t")(model)` passes a vector of size `d`.
Operator models such as `DeepONet` and `FNO` default to structured packing. Pass
`input_mode="structured"` or `structured=True` to force tuple packing, and
`input_mode="flat"` to force MLP-style concatenation.

!!! note
    Input conventions:

    - `DeepONet` expects `(branch_input, coords...)`, where `coords` are either
      `coord_dim` separate 1D axes (grid mode) or a single array with trailing dimension `coord_dim` (point mode).
    - `FNO1d` expects `(grid_values, x_axis)` and requires grid evaluation (`x_axis` must be 1D with length > 1).
    - `FNO2d` expects `(grid_values, x_axis, y_axis)` and requires grid evaluation (`x_axis`, `y_axis` 1D with length > 1).
    - These operator models are minimal implementations intended as building blocks; for
      production use you may want additional features (padding/dealiasing, normalization,
      richer positional encodings, batching utilities, etc.).

::: phydrax.nn.DeepONet
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.FNO1d
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.FNO2d
    options:
        members:
            - __init__
            - __call__
