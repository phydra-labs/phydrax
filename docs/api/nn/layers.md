# Layers

Low-level model building blocks.

!!! note
    Key notes:

    - `Linear` supports Random Weight Factorization (RWF) and optional complex parameters.
    - `Dropout(mode="feature")` shares one feature/channel mask over leading field axes.

::: phydrax.nn.layers.Linear
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.layers.Dropout
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.layers.inference_mode
