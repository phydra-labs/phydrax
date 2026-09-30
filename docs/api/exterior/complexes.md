# De Rham and Hilbert complexes

## Realization protocol

`AbstractDeRhamComplex` binds `dimension`, `primal_twist` and `realization_id` to
`hilbert_complex(*, boundary="absolute")`. Obtain vector spaces with
`realization.hilbert_complex(boundary=...).space(k)`; there is no competing
realization `space()` vocabulary. Cell realizations add topology and boundary
masks through `phydrax.discretization.AbstractCellDeRhamComplex`.

`DiscreteForm(realization_id, form_type, values, /)` carries a single degree-space
vector and its scientific identity. Primal/dual placement follows its twist
relative to the realization's `primal_twist`, not its array shape. Batch numerical
calls with `jax.vmap`.

| Operation | Meaning |
|---|---|
| `exterior_derivative(k, values, boundary=...)` | d_k |
| `codifferential(k, values, boundary=...)` | Hilbert adjoint of d_(k−1) |
| `hodge_laplacian(k, values, boundary=..., part=...)` | dδ, δd or their sum |
| `hodge_star(k, values)` | Metric Riesz map to dual cochains |
| `inverse_hodge_star(k, values)` | Inverse metric Riesz action |
| `dual_exterior_derivative(k, values)` | Signed dual transpose |

For cell cochains d_k = B_(k+1)ᵀ and δ_k = M_(k−1)⁻¹ d_(k−1)ᴴ M_k.
Physical divergence is −δ. Complex adjoints include conjugation. The discrete star
is a Riesz map, not the smooth coefficient complement rotation.

`ComplexBoundary` is `"absolute"` or `"relative"`; calculus calls consistently use
`boundary=`, not a policy object. Relative means the active-coordinate subcomplex:
restricted incidence and restricted RMRᴴ pairing. Full-coordinate calls restrict
inputs and zero-extend outputs. Boundary/inactive DOFs do not enter harmonic
counts or spectra. On a closed cell complex, both choices agree. Periodic spectral
realizations refuse relative boundary rather than pretending it is supported.

Cell realizations supply `active_indices(degree, boundary=)` in their own DOF
layout. Spectral lifting uses these indices, including higher-order finite-element
moments; topology entity counts are not substituted for FE DOF counts. Non-cell
realizations retain their compact native modal layout.

Top-degree d and degree-zero δ raise. Complete Δ omits a missing part; explicitly
requesting that missing lower/upper part raises.

::: phydrax.exterior.AbstractDeRhamComplex

::: phydrax.exterior.DiscreteForm

::: phydrax.exterior.ComplexBoundary

::: phydrax.discretization.AbstractCellDeRhamComplex

## Metric and constitutive separation

`DiagonalHodge(weights, /)` and
`SparseHodge(rows, columns, upper_values, size, /, *, policy=None)` belong to
discretization. They encode positive metric pairings only. SparseHodge accepts the
owning `LinearSolvePolicy`, with native PCG/Jacobi as the default; it does not add a
second `cg`/`cholesky` selector. `valid` is device evidence and `admit()` is the
host refusal boundary. `restrict(active)` builds the restricted pairing;
`refresh(values)` retains its fixed layout.

Material coefficients, including complex lossy coefficients, are separate
constitutive V→Dual(V) operators or weighted coordinate Gram forms. They never
enter an SPD metric pairing. The semantic dual view is not certified
self-adjoint/positive-definite as a coordinate endomorphism. Coordinate conversion
owns those certifications for weak mass, stiffness, mixed matrices and eigen
pencils.

Dynamic values and geometry derivatives stay on device. Explicit binding
`numeric_revision` metadata stays stable inside compiled loops; refresh numerical
leaves and preparation without stale factors or per-step static reconstruction.
Host CHOLMOD/SuperLU is non-JIT, never runs in scan, and is admitted only under an
explicit eager preparation contract.

## Native linear algebra

`HilbertComplex(spaces, differentials, /, *, complex_id)` takes tuples and requires
an explicit identity. Differentials must support transpose and match degree-space
identities. Realizations supply explicit linear operator ids.

`ComplexMap(source, target, maps, /, *, map_id, degree_offset=0)` describes a
commuting degreewise map. Probe evidence reports actual commutation defects.
Harmonic calculations and Hodge decomposition respect the chosen restricted
complex; exact-potential solves need harmonic information at k and k−1. Prepare
repeated decomposition work rather than refactoring a mixed system on each call.

::: phydrax.linalg.HilbertComplex

::: phydrax.linalg.ComplexMap

::: phydrax.linalg.complex_map_evidence

::: phydrax.linalg.complex_nilpotency_evidence

::: phydrax.linalg.codifferential

::: phydrax.linalg.hodge_laplacian

::: phydrax.linalg.mass_form

::: phydrax.linalg.stiffness_form

::: phydrax.linalg.harmonic_subspace

`hodge_laplacian_eigenbasis` solves the native generalized weak pencil using the
full Riesz Gram operator. If M-orthonormal eigenvectors are U, synthesis is
S = sqrt(trace(M)) U and its analysis metric is G = M / trace(M); analysis is
SᴴG. Cell bases zero-extend S, G and analysis into the realization's DOF layout.
This preserves diagonal probability-measure normalization while not inventing
pointwise weights for a sparse non-diagonal Gram. The canonical
`SpectralDecomposition.eigen_solve` retains native status, convergence and residual
diagnostics; failed native solves are refused. Reports use the full metric
orthonormality residual and record actual zero-eigenvalue canonicalization.
`hodge_sector_spectra` retains the native solve results for its harmonic, exact
and coexact sectors, consumed directly by `HodgeSpectralKernel`.

::: phydrax.exterior.hodge_laplacian_eigenbasis

::: phydrax.exterior.hodge_sector_spectra

## Realizations

- Cell cochains: `CochainDiscretization` over canonical topology and metric Hodges.
- Finite elements: `FiniteElementDeRhamComplex` and `form_element`; top-degree
  density is distinct from unrelated scalar discontinuous elements.
- Splines: `SplineDeRhamComplex` over the IGA owner.
- Fourier: `FourierDeRhamComplex(space, nyquist_policy="zero-self-conjugate")`,
  compact admissible modal coordinates and Hermitian Parseval pairing.
- Sphere: `SphericalDeRhamComplex(space)`, normalized poloidal/toroidal coordinates
  and Betti numbers (1, 0, 1).

GraphIR is a diagonal-Hodge lowering, not another scientific complex. MAC exposes
its existing face/cell Hilbert slice. SBP remains a separate collocated stack.
Generic directed multigraphs, negative-cotan metric Hodges, discrete Lorentzian
pairings and SUSY lattice p-forms are not admitted by these contracts.
