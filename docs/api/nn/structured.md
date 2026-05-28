# Structured models

Models that exploit product-domain structure via low-rank factorization.

!!! note
    Key notes:

    - `Separable` implements a CP-style expansion $u=\sum_\ell\prod_i g_i^\ell$.
    - `LatentContractionModel` generalizes this to named factor models and flexible inputs.
    - `LatentExecutionPolicy` controls grouped-vs-flat planning preferences and fallback behavior.
      Supported topology modes are `grouped`, `flat`, `best_effort_flat`, and `strict_flat`.
    - `SeparableMLP`, `SeparableKAN`, and `SeparableFeynmaNN` are drop-in pointwise
      replacements for dense vector models in `Domain.Model(...)`: by default their
      dependencies are flattened into one vector, then each scalar coordinate is sent
      to its own internal model. Use `input_mode="structured"` or `structured=True`
      when you intentionally want tuple/coord-separable input packing.
    - `LatentContractionModel` supports layout hints `auto`, `dense_points`,
      `coord_separable`, `hybrid`, and `full_tensor`.
    - Any automatic fallback can be configured to warn, error, or stay silent.
    - For `LatentContractionModel`, `partial_n` / `dt_n` / `laplacian` can use an
      exact latent-factor derivative contraction path under `backend="jet"`; if that
      path is unavailable, execution falls back according to
      `LatentExecutionPolicy.fallback`.
    - `SeparableMLP`, `SeparableKAN`, and `SeparableFeynmaNN` forward `scan` to
      their internal scalar submodels.
    - Use `key=None` for deterministic inference/export. Fan-out models split real
      evaluation keys for stochastic children, but propagate `None` without creating
      PRNG operations.

::: phydrax.nn.Separable
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.SeparableMLP
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.SeparableKAN
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.SeparableFeynmaNN
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.LatentContractionModel
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.LatentExecutionPolicy
    options:
        members:
            - __init__

---

::: phydrax.nn.ConcatenatedModel
    options:
        members:
            - __init__
            - __call__
