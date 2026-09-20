# Periodic condensed-matter models

Phydrax represents a periodic linear operator once: a directed `EdgeRelation`, one integer lattice translation per edge, and one numeric block per edge. Reverse edges exchange source and target, negate the translation, and carry the adjoint block for a Hermitian family. Host preparation coalesces duplicates and checks this involution. JAX evaluation, matrix-free action, and first or second reciprocal derivatives then use fixed shapes. Dense Bloch matrices and finite COO realizations are explicit bounded requests; sparse action never materializes them.

## Reciprocal supports

`ReciprocalMeshPlan` and `ReciprocalPathPlan` are bound to a `PeriodicCell`. Fractional coordinates therefore have the cell lattice rank, not an assumed dimension of three. A Monkhorst–Pack mesh is constructed with

```python
mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (nk1, nk2))
```

for a rank-two cell. Positive weights sum to one and points must be unique modulo reciprocal lattice translations. `ReciprocalConnectivityPlan.regular(mesh)` creates explicit directed links, reverse links, reciprocal wrap shifts, and oriented plaquettes. Topology and MMN data consume the prepared connectivity and refuse a different cell identity.

## Orbital convention and generalized pencil

`PeriodicOrbitalBasisPlan` fixes orbital labels and order, fractional centers, spin order, fermionic statistics, cell length unit, and either the lattice or atomic Bloch gauge. In lattice gauge, a family is evaluated with `exp(+2π i k·R)`. Atomic gauge additionally applies the center phase `exp[2π i k·(τ_b−τ_a)]`; its derivative includes the derivative of this phase.

`PeriodicOrbitalPencilPlan` binds independent Hamiltonian and overlap families. `PeriodicOrbitalPencilPlan.orthonormal(...)` is the only route that constructs `S=I`; generalized input is never silently orthogonalized at import. Evaluation returns `H(k)`, `S(k)`, both fractional derivatives, the minimum eigenvalue of `S`, its condition number, and a success predicate. Positivity or condition failure is a refusal at the eigenspectrum boundary.

Hubbard parameters, reference populations, and ionic energy live in `PeriodicHubbardMeanFieldPlan`, not in the one-particle pencil. `NativePeriodicSCFPlan` consumes this separate field plan.

## Spectrum and observables

`PeriodicSpectrumPlan` uses the Phydrax generalized self-adjoint eigensolver for `H C = S C ε`. It returns raw residuals and `C† S C` metric residuals. It does not expose a private eigensolver.

A Gaussian `PeriodicDensityOfStatesPlan` retains its integrated state count. `PeriodicProjectedDOSPlan` requires named disjoint orbital groups and uses the generalized Mulliken partition `Re[C* (S C)]`; grouped weights sum to the metric norm. No unnamed Euclidean projection is substituted for generalized `S`.

`PeriodicVelocityPlan` evaluates the full interband matrix

```text
C_n† [dH − (ε_n + ε_m)dS/2] C_m
```

and transforms fractional derivatives through the reciprocal basis. The result is physical Cartesian velocity in meters per second after explicit energy- and length-unit conversion and division by the reduced Planck constant. Diagonal entries reduce to `C_n†(dH−ε_n dS)C_n`. A degeneracy mask identifies cluster blocks whose individual diagonal values are gauge dependent.

`fermi_surface_evidence` returns every regular-mesh cell and band that brackets the selected Fermi energy, including corner energies and unresolved/Lifshitz masks. It does not silently interpolate a cell whose crossing touches a corner or is otherwise unresolved.

The bounded `ChebyshevMomentPlan` is candidate functionality. It currently accepts only pencils whose evaluated overlap is orthonormal `S=I` on the declared mesh and uses matrix-free family actions. Its moments do not imply an eigenspectrum, DOS convergence, or release status.

## Cross-k topology

Values of `S(k)` do not define overlaps between distinct k points. Generalized topology therefore requires a `PeriodicCrossKConnection` or imported MMN matrices. The analytic identity connection is available only for structurally orthonormal pencils. `PeriodicOverlapBundle` retains raw overlaps and singular values before constructing normalized links, and refuses rank-deficient links.

`PeriodicBandManifold` refuses a manifold that is not isolated at every sampled point. `PeriodicWilsonPlan` accepts one explicit closed oriented link sequence and returns its raw link determinant phases, Wilson matrix, eigenphases, Zak phase, link singular values, and unitarity residuals. These phases are hybrid-center evidence; they are not localized Wannier functions.

`PeriodicChernPlan` accumulates oriented plaquette phases and returns the raw Chern value, nearest integer, quantization residual, minimum direct gap, and minimum link singular value. A caller that claims mesh convergence should require `PeriodicChernRefinementEvidence`; missing or failed required refinement is rejected. Discrete invariant selection and connectivity are host-side and are not differentiable operations.

## Finite boundaries and disorder

`PeriodicFiniteBoundaryPlan` selects exactly one route:

- `open(shape)` drops translations crossing every boundary;
- `periodic(shape)` wraps every axis;
- `twisted(shape, twists)` wraps with explicit radian phases;
- `slab(shape, open_axis)` opens one axis and wraps the others.

`PeriodicFiniteOrbitalPlan` realizes both H and S in sparse COO form. Ordering is C-order cell index, then the declared orbital order. `PrescribedPeriodicDisorder` adds fixed onsite shifts in exactly that order and retains its source identity; it does not sample an ensemble. `finite_layer_populations` uses the finite overlap metric and reports its partition residual.

## Wannier90 interchange

HR and MMN readers accept bytes, never paths or auto-detected formats. `PeriodicSourceContext` must bind the exact byte digest and size, rights identity, source and parents, cell, orbital order and centers, gauge, spin order, statistics, and units.

The HR reader requires exactly `nrpts × num_wann²` unique records, complete reverse translations, finite values, and adjoint reverse blocks. Degeneracy factors are applied exactly once when lowering into the canonical family; raw blocks and degeneracies remain available.

The MMN reader requires exact coverage of one caller-prepared reciprocal connectivity. It does not invent k points, weights, windows, a band manifold, or topology, and does not polar-project or clip imported links. Both adapters reject oversized, malformed, truncated, duplicated, nonfinite, digest-mismatched, or context-free input.

## Lifecycle artifacts

`write_periodic_artifact_archive` persists numeric leaves for canonical family
states, pencil and spectrum results, IFC2/IFC3 and finite-displacement evidence,
and bounded single-site DMFT results. `read_periodic_artifact_archive` requires an
exact matching caller-prepared template and `ArrayArtifactProvenance`; providers,
force evaluators, impurity callables, and other executable objects are not
serialized. See the [production-evidence guide](guides_condensed_matter_production_evidence.md).

## Resource and scientific scope

Every host-expanding route admits a caller-visible maximum before allocation: reciprocal points and links, family edges, dense entries, finite entries, eigenpairs, DOS kernels, velocity matrices, Fermi cells, topology links, and interchange bytes/records. Exceeding a limit raises an explicit resource error. Default values are conservative execution policy, not supported-capacity or release claims.

This layer supplies bounded numerical primitives and raw evidence. It does not provide automatic high-symmetry paths, Wannier localization or disentanglement, symmetry indicators, many-body lowering, material accuracy, thermodynamic-limit inference, or a production release claim. Release requires separately governed qualification and runtime evidence.
