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

::: phydrax.terms.SupervisedLikelihoodTerm
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


## Stochastic-process scores and diagnostics

Process diagnostics retain realization, physical-case, time, and event axes.
Trajectory scores require complete shared-path samples. Jump summaries use
only paths with successful event status. Numerical variance requires explicit
paired coarse/fine evidence.

::: phydrax.uq.horizon_score_diagnostics

---

::: phydrax.uq.HorizonScoreDiagnostics

---

::: phydrax.uq.trajectory_score_diagnostics

---

::: phydrax.uq.TrajectoryScoreDiagnostics

---

::: phydrax.uq.observable_rank_diagnostics

---

::: phydrax.uq.pit_diagnostics

---

::: phydrax.uq.UniformRankDiagnostics

---

::: phydrax.uq.temporal_moment_diagnostics

---

::: phydrax.uq.TemporalMomentDiagnostics

---

::: phydrax.uq.semigroup_mc_diagnostics

---

::: phydrax.uq.SemigroupMonteCarloDiagnostics

---

::: phydrax.uq.jump_event_diagnostics

---

::: phydrax.uq.JumpEventDiagnostics

---

::: phydrax.uq.first_passage_diagnostics

---

::: phydrax.uq.FirstPassageDiagnostics

---

::: phydrax.uq.paired_refinement_uncertainty

---

::: phydrax.uq.PairedNumericalUncertainty

---

::: phydrax.uq.predictive_variance_decomposition

---

::: phydrax.uq.PredictiveVarianceDecomposition

## Process calibration, shift evaluation, and retention

`ProcessValidationSplit` uses disjoint physical-case identities. Calibration
fits only the calibration partition and evaluation uses the test partition.
Reports retain both raw and calibrated statistics. The retention report
combines statistical bounds with replay and provenance gates; call
`raise_for_failure()` when a failed gate must stop promotion.

::: phydrax.uq.ProcessValidationSplit

---

::: phydrax.uq.HorizonScaleCalibrator

---

::: phydrax.uq.ProcessConformalCalibrator

---

::: phydrax.uq.process_calibration_report

---

::: phydrax.uq.ProcessCalibrationReport

---

::: phydrax.uq.process_shift_evaluation_matrix

---

::: phydrax.uq.ProcessShiftEvaluationMatrix

---

::: phydrax.uq.ProcessRetentionThresholds

---

::: phydrax.uq.process_retention_report

---

::: phydrax.uq.ProcessRetentionReport