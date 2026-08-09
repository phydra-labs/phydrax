# Predictive results

::: phydrax.uq.SampleAxis
    options:
        members:
            - __init__

---

::: phydrax.uq.PredictiveField
    options:
        members:
            - __init__
            - mean
            - variance
            - std
            - quantile
            - interval
            - epistemic_variance
            - input_variance
            - observation_variance
            - process_variance
            - numerical_variance
            - decompose_variance
            - total_variance

---

::: phydrax.uq.PredictionInterval
    options:
        members:
            - __init__


## Transport diagnostics and deterministic particle transforms

`predictive_sinkhorn_divergence` treats each stochastic realization as one complete
vector event. The ensemble transform replaces normalized weighted particles with an
equal-weight barycentric ensemble and retains the coupling and mean error.

::: phydrax.uq.predictive_sinkhorn_divergence

---

::: phydrax.uq.OptimalTransportEnsembleTransformResult

---

::: phydrax.uq.optimal_transport_ensemble_transform