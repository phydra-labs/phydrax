# Structured models

Models that exploit product-domain structure via low-rank factorization.

!!! note
    Key notes:

    - `Separable` implements a CP-style expansion $u=\sum_\ell\prod_i g_i^\ell$.
    - `LatentContractionModel` generalizes this to named factor models and flexible inputs.

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

::: phydrax.nn.ConcatenatedModel
    options:
        members:
            - __init__
            - __call__
