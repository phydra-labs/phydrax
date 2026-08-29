# Structured models

Models that exploit product-domain structure via low-rank factorization.

!!! note
    Key notes:

    - `Separable` implements a CP-style expansion $u=\sum_\ell\prod_i g_i^\ell$.
    - `LatentContractionModel` generalizes this to named factor models and flexible inputs.
    - `LatentExecutionPolicy` controls grouped-vs-flat planning preferences and fallback behavior.
      Supported topology modes are `grouped`, `flat`, `best_effort_flat`, and `strict_flat`.
    - `SeparableMLP`, `SeparableModifiedMLP`, `SeparableKAN`, and
      `SeparableFeynmaNN` declare a blockwise flat `ModelBinding`; callers bind
      them with `Domain.Model(...)` without execution-mode flags.
    - `LatentContractionModel` declares an axis binding when explicit
      `factor_inputs` are present and a structured blockwise binding otherwise.
      `LatentExecutionPolicy` governs grouped, flat, and fallback planning; the
      model contract does not change at call time.
    - For `LatentContractionModel`, `partial_n` / `dt_n` / `laplacian` can use an
      exact latent-factor derivative contraction path under `backend="jet"`; if that
      path is unavailable, execution falls back according to
      `LatentExecutionPolicy.fallback`.
    - `factorize_axes(...)` and `factorize_axis_batch(...)` expose the same
      latent products, including selected partials, as
      `integration.AxisFactorizedField`. Factorized integration can then assemble
      bilinear forms without materializing the global tensor grid.
    - The separable model families forward `scan` to their internal scalar submodels.
    - Use `key=None` for deterministic inference/export. Fan-out models split real
      evaluation keys for stochastic children, but propagate `None` without creating
      PRNG operations.

::: phydrax.nn.models.Separable
    options:
        members:
            - __init__
            - __call__
            - factorize_axes

---

::: phydrax.nn.models.SeparableMLP
    options:
        members:
            - __init__
            - __call__
            - factorize_axes

---
::: phydrax.nn.models.SeparableModifiedMLP
    options:
        members:
            - __init__
            - __call__

---


::: phydrax.nn.models.SeparableKAN
    options:
        members:
            - __init__
            - __call__

`edge_basis` is forwarded to every coordinate KAN, so fixed spline grids and
per-layer basis schedules retain the same scan compatibility as `KAN`.

---

::: phydrax.nn.models.SeparableFeynmaNN
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.models.LatentContractionModel
    options:
        members:
            - __init__
            - __call__
            - factorize_axis_batch

---

::: phydrax.nn.models.LatentExecutionPolicy
    options:
        members:
            - __init__

---

::: phydrax.nn.models.ConcatenatedModel
    options:
        members:
            - __init__
            - __call__
