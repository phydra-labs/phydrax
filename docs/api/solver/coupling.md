# Partitioned coupling

The public API lives under `phydrax.solver.coupling`. See the
[partitioned coupling guide](../../guides_partitioned_coupling.md) for numerical,
transactional, transfer, waveform, and differentiation contracts.

## Ports, participants, and exchanges

::: phydrax.solver.coupling.CouplingPort

---

::: phydrax.solver.coupling.CouplingTransferRequirement

---

::: phydrax.solver.coupling.CouplingExchange

---

::: phydrax.solver.coupling.CouplingSubsystemCapabilities

---

::: phydrax.solver.coupling.AbstractCouplingSubsystem

---

::: phydrax.solver.coupling.CallableCouplingSubsystem

---

::: phydrax.solver.coupling.CouplingSubsystemResult

## Graph preparation and refresh

::: phydrax.solver.coupling.CouplingGraph

---

::: phydrax.solver.coupling.CouplingStagePlan

---

::: phydrax.solver.coupling.CouplingResourcePolicy

---

::: phydrax.solver.coupling.CouplingResourceEstimate

---

::: phydrax.solver.coupling.CouplingPreparationReport

---

::: phydrax.solver.coupling.PreparedCoupling

---

::: phydrax.solver.coupling.prepare_coupling

---

::: phydrax.solver.coupling.refresh_coupling

## Numerical and differentiation policies

::: phydrax.solver.coupling.CouplingSweep

---

::: phydrax.solver.coupling.CouplingTolerance

---

::: phydrax.solver.coupling.ExplicitCouplingPolicy

---

::: phydrax.solver.coupling.ImplicitCouplingPolicy

---

::: phydrax.solver.coupling.CouplingDifferentiationPolicy

## Window execution and evidence

::: phydrax.solver.coupling.CouplingWindow

---

::: phydrax.solver.coupling.CouplingState

---

::: phydrax.solver.coupling.CouplingWindowDiagnostics

---

::: phydrax.solver.coupling.CouplingProvenance

---

::: phydrax.solver.coupling.CouplingWindowResult

---

::: phydrax.solver.coupling.CouplingStatus

---

::: phydrax.solver.coupling.coupling_status_message

---

::: phydrax.solver.coupling.advance_coupling_window

## Fixed-window rollout

::: phydrax.solver.coupling.CouplingProblem

---

::: phydrax.solver.coupling.CouplingRolloutPlan

---

::: phydrax.solver.coupling.CouplingSolution

---

::: phydrax.solver.coupling.solve_coupling

## Fixed-capacity waveforms, adaptive windows, and epochs

::: phydrax.solver.coupling.CouplingWaveformPlan

---

::: phydrax.solver.coupling.CouplingWaveformGrid

---

::: phydrax.solver.coupling.CouplingWaveform

---

::: phydrax.solver.coupling.BarycentricCouplingTemporalTransfer

---

::: phydrax.solver.coupling.CouplingWaveformAdaptationPolicy

---

::: phydrax.solver.coupling.adapt_coupling_waveform_grid

---

::: phydrax.solver.coupling.AdaptiveCouplingWindowPolicy

---

::: phydrax.solver.coupling.AdaptiveCouplingRolloutPlan

---

::: phydrax.solver.coupling.rollout_adaptive_coupling

---

::: phydrax.solver.coupling.PreparedCouplingEpoch

---

::: phydrax.solver.coupling.CouplingEpochTransitionPlan

---

::: phydrax.solver.coupling.transition_coupling_epoch

---

::: phydrax.solver.coupling.FixedGridSubcyclingSubsystem
