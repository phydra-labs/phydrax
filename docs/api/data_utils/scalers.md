# Scalers

Scalers are immutable modules for normalizing arrays before model training or
inference and mapping transformed outputs back to their original scale.

::: phydrax.data_utils.scalers.AffineScaler
    options:
        members:
            - __init__
            - transform
            - inverse_transform

---

::: phydrax.data_utils.scalers.MinMaxScaler
    options:
        members:
            - __init__
            - transform
            - inverse_transform

---

::: phydrax.data_utils.scalers.MaxAbsScaler
    options:
        members:
            - __init__
            - transform
            - inverse_transform

---

::: phydrax.data_utils.scalers.StdScaler
    options:
        members:
            - __init__
            - transform
            - inverse_transform

---

::: phydrax.data_utils.scalers.NormScaler
    options:
        members:
            - __init__
            - transform
            - inverse_transform

---

::: phydrax.data_utils.scalers.scaler_transform_fn
