# Activations

Trainable nonlinearities and the declared value regularity of known activations.

!!! note
    Key notes:

    - `Stan` is a self-scalable tanh: $\operatorname{tanh}(x)\,(1+\beta x)$.
    - `AdaptiveActivation` wraps $\sigma$ as $x\mapsto\sigma(ax)$.
    - `activation_regularity` matches known activation callables by identity and
      returns their `DerivativeRegularity`; unknown callables are undeclared (`None`).

::: phydrax.nn.activations.Stan
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.activations.AdaptiveActivation
    options:
        members:
            - __init__
            - __call__

---

::: phydrax.nn.activations.squared_relu

---

::: phydrax.nn.activations.activation_regularity
