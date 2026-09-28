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
kernel, the point-sampled Biot–Savart kernel, and `"tabulated"` kernels whose
caller-supplied `kernel_table` holds the kernel at every on-grid displacement
(for Green functions without reflection symmetry; trailing axes give several
fields from one source). `gradient=True` prepares the
derivative kernels so fields are convolved rather than finite-differenced:
point-sampled derivatives for the sampled kernels and, for the integrated
kernels, the exact field of the density that is linear between sites along the
derivative axis and cell-constant across it,
`K_a(m) = [P(m + ½e_a) − P(m − ½e_a)]/h_a` with `P` the exact cell integral.
The cell average of `∂_a g` would drop the near-zone field of the density
gradient inside a cell; on cells longer than the source is wide (relativistic
rest frames) that underestimates `E_z` by 40 % at aspect ratio 100 and 60 % at
1000, while the face-difference kernel stays second order at any aspect ratio.
`FreeSpaceVortexFFTPlan` (vortex methods),
`IsolatedCartesianGravityPlan` (`phydrax.solver.advanced`),
`SpaceChargeIGFPlan`, and the `"3d-steady-igf"` `CSRPlan`
(`phydrax.applications.accelerator`) are thin owners of their scientific
meaning over this substrate.

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
