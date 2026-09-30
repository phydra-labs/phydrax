# Modal transforms and direct solvers

## Fast linear transforms

::: phydrax.linalg.AbstractLinearTransform

---

::: phydrax.linalg.FFTLinearTransform

---

::: phydrax.linalg.RealTrigonometricTransform

---

::: phydrax.linalg.SimilarityScaledLinearTransform

---

::: phydrax.linalg.TensorLinearTransform

## Weighted and irregular modal bases

`ModalTransform` and `SpectralDecomposition` expose an `analysis_metric` native
linear operator on their full coordinates. Diagonal routes still accept
`quadrature_weights` and resolve that metric to a diagonal operator. For a
non-diagonal Gram matrix, supply `analysis_metric=` instead of weights: analysis
is the metric adjoint of synthesis, and orthonormality is measured with the full
operator. `quadrature_weights` is then `None`; `probability_measure` refuses the
request because an off-diagonal Gram operator is not a pointwise measure.
The transform identity includes the metric operator identity. Product Laplacian
bases compose factor metrics by the native Kronecker operator, retaining
off-diagonal pairings rather than multiplying their diagonals.
`EigenbasisDiscretization` is specifically a point-value discretization with a
pointwise scalar quadrature integral. It therefore admits only spectral plans
with `quadrature_weights`; operator-Gram plans remain available for native modal
analysis/synthesis, product spectra and spectral kernels, but are refused by
this point-measure wrapper rather than assigned a fictitious diagonal measure.


::: phydrax.discretization.ModalTransform

---

::: phydrax.discretization.OperatorSpectrum

---

::: phydrax.discretization.trigonometric_modal_transform

---
::: phydrax.discretization.TensorModalTransform

---

Spherical coefficient layouts and Laplace--Beltrami actions are owned by
`SphericalSpectralDiscretization`; they are not attached to arbitrary dense modal
transforms.

::: phydrax.discretization.SpectralDecomposition

---

::: phydrax.linalg.TransformDiagonalRepresentation

---

::: phydrax.linalg.TransformDiagonalSolvePlan

---

::: phydrax.linalg.PreparedTransformDiagonalSolve

---

::: phydrax.linalg.TransformDiagonalSolveResult

---

::: phydrax.linalg.LowRankBoundaryCorrectionPlan

## Global collocation

::: phydrax.discretization.ChebyshevCollocation
