# Probability distributions and propagation

## Distributions

::: phydrax.uq.Uniform
    options:
        members:
            - __init__
            - sample
            - icdf
            - log_prob
            - to_reference
            - from_reference

---

::: phydrax.uq.Normal
    options:
        members:
            - __init__
            - sample
            - icdf
            - log_prob

            - to_reference
            - from_reference
---

::: phydrax.uq.LogNormal
    options:
        members:
            - __init__
            - sample
            - icdf
            - log_prob

            - to_reference
            - from_reference
---

::: phydrax.uq.EmpiricalDistribution
    options:
        members:
            - __init__
            - sample
            - icdf
            - log_prob

## Domains and joint designs

::: phydrax.domain.ProbabilityDomain
    options:
        members:
            - __init__
            - sample
            - fixed
            - supports_reference_transform
            - reference_measure
            - to_reference
            - from_reference

---
::: phydrax.domain.ReferenceDistribution

---


::: phydrax.uq.RandomSampleBatch
    options:
        members:
            - __init__

---

::: phydrax.uq.sample_joint

---

::: phydrax.uq.propagate
