# Path-integral operators

!!! note
    For mathematical scope, diagnostics, convergence, and examples, see
    [Guides → Euclidean path integrals and Feynman–Kac expectations](../../guides_path_integrals.md).

!!! warning
    Real-time support is a strictly positive-regulator, finite-slice
    oscillatory integral with phase-cancellation evidence. It does not claim
    an unregulated limit, automatic slice convergence, or field-theory
    universality.

::: phydrax.discretization.TemporalMesh

---

::: phydrax.operators.PathIntegralEstimate

---

::: phydrax.operators.brownian_bridge_from_noise

---

::: phydrax.operators.sample_brownian_bridge

---

::: phydrax.operators.kinetic_action

---

::: phydrax.operators.potential_action

---

::: phydrax.operators.discrete_euclidean_action

---

::: phydrax.operators.free_euclidean_kernel

---

::: phydrax.operators.euclidean_kernel_from_noise

---

::: phydrax.operators.euclidean_kernel

---

::: phydrax.operators.euclidean_kernel_function

---

::: phydrax.operators.diffusion_paths_from_noise

---

::: phydrax.operators.sample_diffusion_paths

---

::: phydrax.operators.feynman_kac_from_paths

---

::: phydrax.operators.feynman_kac_expectation

---

::: phydrax.operators.first_exit_index

---

::: phydrax.operators.first_exit_time

---

::: phydrax.operators.survival_probability

---

## Regulated real time and geometry

::: phydrax.operators.path_integral.RealTimePathIntegralPlan

::: phydrax.operators.path_integral.real_time_kernel_from_noise

::: phydrax.operators.path_integral.RealTimeRegulatorContinuation

::: phydrax.operators.path_integral.PreparedGeometryPathKernel

::: phydrax.operators.path_integral.interval_heat_kernel

::: phydrax.operators.path_integral.prepare_path_boundary_schedule

The exact reflecting Brownian density route is limited to prepared affine
interval image kernels. General prepared geometry supports killing, and
kinetic states with explicit velocity support specular resets. Curved-domain
overdamped reflection is not inferred from a specular reset.

## Source, periodic, gauge, and exchange measures

::: phydrax.operators.path_integral.source_feynman_kac_from_paths

::: phydrax.operators.path_integral.source_feynman_kac_from_stochastic_paths

::: phydrax.operators.path_integral.PeriodicPathPlan

::: phydrax.operators.path_integral.estimate_path_partition_function

::: phydrax.operators.path_integral.CompactU1GaugeMeasure

::: phydrax.operators.path_integral.ExchangePathPlan

Absolute partition estimates require a named reference with known log
partition. Gauge measures are finite compact U(1), and exchange estimates are
limited to the admitted permutation table. Fermionic average-sign collapse is
reported rather than repaired.
