# Structured models

Models that exploit product-domain structure via low-rank factorization.

!!! note
    Key notes:

    - `Separable` implements a CP-style expansion $u=\sum_\ell\prod_i g_i^\ell$.
    - `LatentContractionModel` generalizes this to named factor models and flexible inputs.
    - `LatentExecutionPolicy` controls grouped-vs-flat planning preferences and fallback behavior.
      Supported topology modes are `grouped`, `flat`, `best_effort_flat`, and `strict_flat`.
    - `LatentContractionModel` supports layout hints `auto`, `dense_points`,
      `coord_separable`, `hybrid`, and `full_tensor`.
    - Any automatic fallback can be configured to warn, error, or stay silent.
    - For `LatentContractionModel`, `partial_n` / `dt_n` / `laplacian` can use an
      exact latent-factor derivative contraction path under `backend="jet"`; if that
      path is unavailable, execution falls back according to
      `LatentExecutionPolicy.fallback`.

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
