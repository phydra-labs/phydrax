# Integral field operators

!!! note
    Global measure-aware quadrature and sampling live under
    [`phydrax.integration`](../integration.md). See
    [Integrals and measures](../../guides_integrals.md) for the target/plan/estimate
    workflow.

The operators below construct or reduce domain fields inside larger operator
expressions. They are retained for local, nonlocal, spatial, and convolutional field
transforms. New global integrals should use `phydrax.integration.integrate`.

`time_convolution` is a deterministic field operator. It maps a declared fixed
`IntervalRule` onto each causal interval, is exactly zero at the time-domain
start, and records its rule in `DomainFunction.metadata`. Randomized inner
integrals must retain independent realizations and use an estimator-aware
randomized term; they are not exposed as an averaged field that can be squared
silently.

::: phydrax.operators.integral

---

::: phydrax.operators.mean

---

::: phydrax.operators.integrate_interior

---

::: phydrax.operators.integrate_boundary

---

::: phydrax.operators.spatial_integral

---

::: phydrax.operators.local_integral

---

::: phydrax.operators.local_integral_ball

---

::: phydrax.operators.nonlocal_integral

---

::: phydrax.operators.time_convolution

## Free-space convolution on bounded grids

`FreeSpaceConvolutionPlan` is the one Hockney doubled-grid FFT substrate for
open-boundary convolutions on bounded uniform cell grids. It owns the kernel
family (`FreeSpaceKernel`): cell-integrated Coulomb and Newton Green functions
(Qiang, Lidia, Ryne, Limborg-Deprey 2006), the point-sampled softened Newton
kernel, and the point-sampled Biot–Savart kernel. `gradient=True` prepares the
derivative kernels under the same sampling rule so fields are convolved rather
than finite-differenced. `FreeSpaceVortexFFTPlan` (vortex methods),
`IsolatedCartesianGravityPlan` (`phydrax.solver.advanced`), and
`SpaceChargeIGFPlan` (`phydrax.applications.accelerator`) are thin owners of
their scientific meaning over this substrate.

::: phydrax.operators.FreeSpaceConvolutionPlan
    options:
      show_root_heading: true

::: phydrax.operators.FreeSpaceConvolutionResult
    options:
      show_root_heading: true

## Three-dimensional multipole translations

::: phydrax.operators.LaplaceMultipolePlan3D
    options:
      show_root_heading: true

::: phydrax.operators.PreparedLaplaceMultipole3D
    options:
      show_root_heading: true

::: phydrax.operators.LaplaceMultipoleEvaluation3D
    options:
      show_root_heading: true

::: phydrax.operators.HelmholtzMultipolePlan3D
    options:
      show_root_heading: true

::: phydrax.operators.ModifiedHelmholtzMultipolePlan3D
    options:
      show_root_heading: true

::: phydrax.operators.evaluate_laplace_layer_multipole_3d
    options:
      show_root_heading: true

::: phydrax.operators.prepare_laplace_qbx_far_local_3d
    options:
      show_root_heading: true
