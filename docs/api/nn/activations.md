# Activations

Trainable nonlinearities.

!!! note
    Key notes:

    - `Stan` is a self-scalable tanh: $\operatorname{tanh}(x)\,(1+\beta x)$.
    - `AdaptiveActivation` wraps $\sigma$ as $x\mapsto\sigma(ax)$.

::: phydrax.nn.Stan
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.AdaptiveActivation
    options:
        members:
            - __init__
            - __call__
