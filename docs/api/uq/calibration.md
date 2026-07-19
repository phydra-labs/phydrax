# Likelihoods, scores, and calibration

## Observation likelihoods

::: phydrax.uq.GaussianLikelihood
    options:
        members:
            - __init__
            - log_prob
            - sample

---

::: phydrax.uq.GaussianLocationScaleLikelihood
    options:
        members:
            - __init__
            - scale_from_raw
            - log_prob
            - sample

---

::: phydrax.uq.StudentTLikelihood
    options:
        members:
            - __init__
            - log_prob
            - sample

---

::: phydrax.constraints.SupervisedLikelihoodConstraint
    options:
        members:
            - __init__
            - sample
            - observed_batch
            - log_prob
            - loss

## Proper scores and diagnostics

::: phydrax.uq.negative_log_likelihood

---

::: phydrax.uq.gaussian_crps

---

::: phydrax.uq.student_t_crps

---

::: phydrax.uq.ensemble_crps

---

::: phydrax.uq.energy_score

---

::: phydrax.uq.pinball_loss

---

::: phydrax.uq.interval_coverage

---

::: phydrax.uq.interval_width

---

::: phydrax.uq.calibration_error

## Scale and conformal calibration

::: phydrax.uq.GaussianScaleCalibrator
    options:
        members:
            - __init__
            - fit
            - transform

---

::: phydrax.uq.SplitConformal
    options:
        members:
            - __init__
            - calibrate
            - interval

---

::: phydrax.uq.NormalizedConformal
    options:
        members:
            - __init__
            - calibrate
            - interval

---

::: phydrax.uq.FunctionalConformal
    options:
        members:
            - __init__
            - calibrate
            - interval
