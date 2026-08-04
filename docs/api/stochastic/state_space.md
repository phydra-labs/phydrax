# State-space models and transition adapters

## Contract

A `StateSpaceProblem` combines an explicit prior, transition kernel, observation model,
and `ObservationSequence`. The observation sequence owns physical case axes, timestamps,
missingness masks, schedule validity, stable case IDs, and a sequence ID. A transition
sample carries values, validity, status, and a process ID. Filtering algorithms therefore
consume one common contract without guessing whether an array axis denotes a case,
particle, ensemble member, or state component.

`state_space_key` derives semantic subkeys from the root key, operation, case ID, step,
and optional member. Batch and streaming execution consequently replay the same draws.
Changing an unrelated case, observation mask, or execution order does not silently
renumber an existing random stream.

## Priors

::: phydrax.stochastic.AbstractStatePrior

---

::: phydrax.stochastic.GaussianStatePrior

---

::: phydrax.stochastic.CategoricalStatePrior

---

::: phydrax.stochastic.DistributionStatePrior

## Observation models and schedules

::: phydrax.stochastic.ObservationSequence

---

::: phydrax.stochastic.AbstractObservationModel

---

::: phydrax.stochastic.CallableObservationModel

---

::: phydrax.stochastic.GaussianObservationModel

---

::: phydrax.stochastic.LinearGaussianObservationModel

## Transition kernels

`MarginalTransitionKernel` wraps a finite-interval marginal law. Use a pathwise adapter
when a transition must preserve one driver realization, event stream, or cocycle across
the interval.

::: phydrax.stochastic.AbstractTransitionKernel

---

::: phydrax.stochastic.TransitionSample

---

::: phydrax.stochastic.CallableTransitionKernel

---

::: phydrax.stochastic.MarginalTransitionKernel

---

::: phydrax.stochastic.LinearGaussianTransitionKernel

---

::: phydrax.stochastic.DifferentialTransitionKernel

---

::: phydrax.stochastic.JumpTransitionKernel

---

::: phydrax.stochastic.JumpDifferentialTransitionKernel

---

::: phydrax.stochastic.FiniteStateTransitionKernel

---

::: phydrax.stochastic.PathwiseTransitionKernel

## Model and problem

::: phydrax.stochastic.StateSpaceModel

---

::: phydrax.stochastic.StateSpaceProblem

---

::: phydrax.stochastic.state_space_key
