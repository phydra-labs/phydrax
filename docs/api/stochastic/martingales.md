# Martingale problems, stopping, and diagnostics

`MartingaleProblem` declares an observable and its generator action on one
`StochasticTrajectory`. `martingale_increments` evaluates the compensated increment

$$
M_{t_{i+1}}-M_{t_i}
= \phi(X_{t_{i+1}})-\phi(X_{t_i})
- \int_{t_i}^{t_{i+1}} \mathcal L\phi(X_s)\,ds.
$$

Left, midpoint, and trapezoid quadrature are explicit. Stopping indices truncate the
same trajectory; they do not create an unrelated path. Predictable brackets,
quadratic covariation, generator carré du champ terms, and finite-activity jump
compensators use the same interval alignment.

Statistical diagnostics live in `phydrax.uq`. They aggregate by realization-independence
cluster, so antithetic pairs and other coupled paths are not counted as independent
samples. `MartingaleValidationReport` combines moment, quadratic-variation, and optional
jump-compensator checks without replacing any failed component with a fabricated pass.

## Problems and increments

::: phydrax.stochastic.MartingaleProblem

---

::: phydrax.stochastic.MartingaleIncrements

---

::: phydrax.stochastic.martingale_increments

---

::: phydrax.stochastic.stopped_martingale_increments

---

::: phydrax.stochastic.first_stopping_indices

---

::: phydrax.stochastic.StoppingIndices

## Generator and bracket helpers

::: phydrax.stochastic.carre_du_champ

---

::: phydrax.stochastic.combined_generator_observable

---

::: phydrax.stochastic.jump_generator_observable

---

::: phydrax.stochastic.predictable_bracket_increments

---

::: phydrax.stochastic.quadratic_covariation

## Objectives and diagnostics

::: phydrax.stochastic.martingale_moment_loss

---

::: phydrax.uq.martingale_diagnostics

---

::: phydrax.uq.quadratic_variation_diagnostics

---

::: phydrax.uq.jump_compensator_diagnostics

---

::: phydrax.uq.martingale_validation_report
