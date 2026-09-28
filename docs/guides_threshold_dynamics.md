# Threshold dynamics for multiphase capillary geometry

`phydrax.threshold_dynamics` evolves many-cell capillary geometry — grain
networks, dry foams, multiphase droplets — as a hard label field. Every site
carries one label; one step convolves the label indicators with a heat kernel and
reassigns every site to its cheapest label. The method is the
Merriman–Bence–Osher (MBO) scheme in its multiphase, arbitrary-tension form by
Esedoglu and Otto, with arbitrary mobilities by the two-kernel construction of
Salvador and Esedoglu, exact label volumes by the auction dynamics of Jacobs,
Merkurjev and Esedoglu, and a sparse candidate-label route in the spirit of
Elsey, Esedoglu and Smereka.

It complements, and does not replace, the explicit multiregion surfaces of
`phydrax.geometry.multiregion_surface` (film-resolved foams) and the diffuse
phase-field models of `phydrax.applications.phase_field`.

## Interface tensions and mobilities

Labels are stable identifiers. `InterfaceTensionMatrix` and
`InterfaceMobilityMatrix` (`phydrax.interfacial_transport`, documented with
[surface thin films](api/surface_thin_films.md)) hold `sigma_ij` and `mu_ij` for
every unordered pair of distinct labels, either as a dense symmetric matrix
(`structure="pairwise"`) or as one shared value (`structure="uniform"`, needed
when the label count makes an `L x L` matrix a resource defect). Interface
`Gamma_ij` stores energy `sigma_ij |Gamma_ij|` and moves with normal velocity
`mu_ij sigma_ij kappa`. A soap film carries the effective tension of both of
its surfaces. `admissibility()` reports, on device:

- the triangle inequality `sigma_ik <= sigma_ij + sigma_jk` (required: without it
  the multiphase energy is not lower semicontinuous and a third phase wets the
  interface);
- conditional negative semidefiniteness of `sigma` on the sum-zero subspace, from
  the native `linalg.verify_dense_properties` spectrum (the sufficient condition
  for unconditional energy dissipation).

## Model and kernels

For a physical step `dt` the reduced kernel time of pair `ij` is
`tau_ij = mu_ij sigma_ij dt` (a squared length). The pair kernel is

```text
K_ij = a_ij G(tau_alpha) + b_ij G(tau_beta),
a_ij sqrt(tau_alpha) + b_ij sqrt(tau_beta) = sigma_ij          (tension)
a_ij / sqrt(tau_alpha) + b_ij / sqrt(tau_beta) = sigma_ij / tau_ij   (mobility)
```

with `G(tau) = exp(tau Laplacian)`. `kernel_form="single-gaussian"` requires one
shared `tau_ij` and is the Esedoglu–Otto scheme `K_ij = sigma_ij G(tau) / sqrt(tau)`;
`"two-gaussian"` uses `tau_alpha = min tau_ij` and
`tau_beta = max(max tau_ij, 2 tau_alpha)`, which keeps `a_ij, b_ij >= 0`. With
potentials `psi_i = sum_j K_ij * u_j` a step assigns

```text
l(x) <- argmin_{i active} psi_i(x)     (ties: lowest label index)
```

and the discrete Lyapunov energy is

```text
E(u) = (sqrt(pi) / 2) sum_x m(x) psi_{l(x)}(x)  ->  sum_{i<j} sigma_ij |Gamma_ij|.
```

`E` is non-increasing for every `dt` when both coefficient matrices are
conditionally negative semidefinite and the route's discrete kernel is positive
semidefinite in the site measure. Only then is `dissipation_admitted` true and an
energy increase beyond roundoff a failure (`ENERGY_INCREASE`, rolled back).
`ThresholdDynamicsPlan` refuses zero tensions, triangle-inequality violations,
nonpositive mobilities and single-kernel inputs with inconsistent
`mu_ij sigma_ij`; tensions, mobilities and `dt` remain dynamic leaves and are
re-certified inside every step (`INADMISSIBLE_PARAMETERS` otherwise).

## Routes

| Route | Sites | Kernel | Labels per site |
|---|---|---|---|
| `PeriodicGridHeatKernel` | periodic uniform grid (1–3D) | exact Fourier multiplier `exp(-tau |k|^2)` of the canonical `TensorSpectralDiscretization` | every declared label |
| `SparseLabelGrid` | active voxels of a periodic Morton box in aligned bricks | box stencil of the exact 1D periodic heat kernels | `candidate_capacity` labels of the brick halo |
| `MeshHeatKernel` | mesh vertices, lumped P1 mass | native Taylor action of `exp(-tau M^{-1} K)` | every declared label |

Dense routes are admitted by `ThresholdDynamicsResourcePolicy` (about eight
`site x label` working arrays). When tension and mobility are uniform, their two
kernel coefficients remain two structured off-diagonal scalars through heat
evaluation; no `label x label` coefficient array is formed. Nonuniform routes
report and admit the retained pair-time and two coefficient matrices before
those arrays are allocated, and `working_bytes` includes that storage.

The sparse route groups the labels of every brick halo with a case-local
`KeyGroupPlan`, accumulates the stencil only into those candidate slots, and
never forms a `site x global-label` array: storage and work scale with sites,
stencil size and candidate capacity, independent of the declared label count.
It reports candidate overflow (`CANDIDATE_OVERFLOW`, rolled back), the truncated
kernel mass and the smallest eigenvalue factor of the truncated periodic
stencil; step evidence combines the current and candidate evaluations so a
current-only overflow remains visible. Dissipation is admitted only when the
symbol factor is nonnegative. A stencil is exact when its distinct periodic
residues cover the period (`2 r + 1 >= resolution` for this construction), and
then its reported truncated mass is exactly zero.

The mesh route preserves native matrix-function status and error together with
operator/prepared provenance, convergence, derivative validity, iterations,
matvec work, retained storage and workspace in `HeatActionEvidence`. Current and
candidate actions are aggregated without dropping those diagnostics, and their
errors enter the energy comparison. There is no implicit-Euler heat
approximation.

On the mesh route, sites are the vertices of a planar triangle mesh, a
triangulated surface in R^3 or a tetrahedral mesh, each weighted by its lumped
P1 mass `m_i`; `G(tau) = exp(-tau M^{-1} K)` with the P1 stiffness `K` (the
cotangent Laplacian on triangles) and `M = diag(m)`. `M^{-1} K` is self-adjoint and
positive semidefinite in the `m`-weighted inner product, boundaries carry the
natural zero-flux condition, and the native scaled Taylor action is prepared once
and applied to one label column per call. The default policy estimates the
operator norm: its error is the observed Taylor tail (an estimate, not a
certified bound). Stored-coordinate analytic bounds grow like
`exp(tau ||M^{-1} K||_1)` and refuse (`RESOURCE_EXHAUSTED`, surfaced as
`HEAT_ACTION_FAILED`) at the resolved kernel times `tau >~ h^2` threshold
dynamics needs. Simplices must be consistently oriented (tetrahedra positively).

The sphere qualification converts a discrete cap area to `cos(theta)` with
`1 - 2 A_cap / A_mesh`, where `A_mesh = sum_i m_i` is the area of that same
chordal polyhedron. Using the continuum denominator `4 pi` with a polyhedral
numerator adds a geometry-normalization bias before the heat action is tested.
The campaign separately varies the Taylor tolerance,
compares lumped and consistent P1 mass for one step, varies `tau` at fixed mesh,
and jointly refines `tau` and `h`. In the critical `h = O(tau)` regime, a bounded
error is reported without a convergence order unless the measured sequence is
monotone; Misiats and Yip show that this regime can contain pinning, depinning,
and grid anisotropy.

## Accuracy, resolution and pinning

- Two-phase MBO is first order in `tau` when the grid spacing is `o(tau)`;
  multiphase junction motion is half order. On a grid the interface snaps to
  sites every step: when the per-step displacement `tau kappa` falls below half a
  cell, interfaces pin. `resolution_ratio = sqrt(tau_alpha) / h` below
  `minimum_resolution_ratio` commits with status `UNDER_RESOLVED`; pinning by low
  curvature (`tau kappa < h / 2`) is a property of the configuration and is not
  detected.
- Measured on the periodic route (qualification tool): the curve-shortening rate
  `dA/dt = -2 pi` has relative error 2.3% (128^2, dt 4e-3), 2.0% (256^2,
  2e-3), 0.9% (512^2, 1e-3) and 0.6% (1024^2, 5e-4) under joint refinement;
  refining `dt` alone at fixed `h` does not improve it. Herring angles for
  tensions `(1, 1, sqrt 2)` are within 4.4 degrees at 256^2 and 3.5 degrees at
  512^2 of `(90, 135, 135)`.

## Exact volumes

`LabelVolumeConstraint(lower_counts, upper_counts=None)` replaces the argmin by
the assignment minimizing `sum_x psi_{l(x)}(x)` under integer site counts
`lower_i <= n_i <= upper_i` (exact when `upper_counts` is omitted), solved by
`phydrax.combinatorial.CapacitatedAuctionPlan` over the per-site candidate slots
(forward auction for upper bounds, reverse auction for lower bounds, static
epsilon scaling, bounded rounds, explicit duality-gap certificate). Declared
counts must be nonnegative and exactly representable as signed int32; they are
validated before device conversion. Volumes are exact only because every site
carries the same measure; routes with unequal site measure are refused (in
practice the mesh route, whose lumped masses are never bitwise equal).
Dissipation is claimed only when the current state satisfies the bounds; the
energy comparison admits the auction's duality gap. A failed auction
(`VOLUME_CONSTRAINT_FAILED`) rolls the step back with no partial assignment.

Auction prices are numerical duals. Their physical normalization is a research
gate that the qualification tool checks: at rest, the interface condition
`psi_i + p_i = psi_j + p_j` gives `p_j - p_i = sigma_ij kappa / sqrt(pi)`, so

```text
P_i = -sqrt(pi) p_i   (up to one additive gauge)
```

satisfies the Laplace law. The measured ratio `sqrt(pi) (p_out - p_in) R / sigma`
is within a few percent of one for circles and spheres when `tau kappa`
exceeds the grid spacing, and degrades (to about 0.75) when interfaces pin.
Single-step prices oscillate within the assignment's dual interval by a few
percent.

## Gas-diffusive dry-foam coarsening

Because the normalization validates, `GasDiffusionCoarsening(prepared, permeance)`
evolves incompressible gas cells: each step reads cell pressures from the prices,
measures film areas isotropically as
`A_ij = (sqrt(pi) / sqrt(tau)) m sum_x u_i G(tau) u_j`, prescribes the next
integer volumes `V + dt k sum_j A_ij (P_j - P_i)` (largest-remainder rounding
conserves the total exactly) and advances the films by one volume-constrained
step. Films move with velocity `mu (sigma kappa - [P])`, so permeance and film
mobility act in series, `k_eff = k mu / (k + mu)`: an isolated circular cell
loses area at `2 pi sigma k_eff`, and isotropic 2D dry foams follow von Neumann's
law `dA/dt = (pi sigma k_eff / 3)(n - 6)`. The quasi-static foam limit is
`mu >> k`. Gas compressibility, film drainage and rupture are not modelled.

## Labels, epochs, statuses and evidence

A label that loses all its sites becomes inactive, never re-nucleates, and
advances `LabelFieldState.epoch`; simultaneous extinctions share one epoch.
Every state carries its ordered stable `label_ids` plus route, preparation and
site-layout identities. Construction refuses out-of-range labels or an owning
label marked inactive. `potentials`, `energy`, `step`, `run` and gas coarsening
refuse a foreign identity or site shape before numerical execution, so an
equal-shaped state cannot be reinterpreted under reordered materials.

Threshold steps are not differentiable, and no derivative is claimed across
epochs. `step` returns one `ThresholdDynamicsStepResult`; `run` carries the
potentials of the committed state into the next step (one heat combination per
step), warm-starts the auction, halts at the first rolled-back step and stacks
`ThresholdDynamicsEvidence` along a leading step axis. Statuses in increasing
severity: `SUCCESS`, `UNDER_RESOLVED` (both commit), `ENERGY_INCREASE`,
`VOLUME_CONSTRAINT_FAILED`, `NONFINITE`, `HEAT_ACTION_FAILED`,
`INADMISSIBLE_PARAMETERS`, `CANDIDATE_OVERFLOW` (all roll back).

## Seeding an explicit multiregion surface

`LabelFieldSurfaceExtractionPlan`, re-exported here as the threshold integration
facade, is owned by `phydrax.geometry.multiregion_surface`. It maps stable C
label IDs to explicit E region IDs through deterministic, junction-conforming
multi-label marching tetrahedra on a Freudenthal cube split. Uniform states bind
by their 3D shape; sparse states bind with
`SparseLabelGrid.site_coordinates` and the route's full grid shape. The plan
requires explicit spacing, origin, ambient identity, surface capacities and a
host validation policy.

Extraction is seed/repair, not coupled evolution. It requires the state's
ordered label identity to match the extraction plan and fingerprints one
`LabelFieldState` epoch together with its binding, route, source preparation and
site identities. Those identities remain explicit in
`LabelFieldSurfaceLineage`. Extraction resolves multi-label cells in a canonical
tetrahedral order, then requires E's orientation, incidence, valence,
watertightness, capacity and collision certificates. Only an accepted result
exposes a new explicit topology, state and `PreparedMultiRegionSurface`; those
objects become the sole geometry authority. Later threshold steps do not update
them, and later E motion/events do not update the label field. Periodic C
geometry is not unwrapped during this conversion.

## Qualification

`tools/threshold_dynamics_qualification.py` runs, at feasible CPU scale, the
circle and sphere rate convergence, Herring angles, the price-normalization gate,
isolated-bubble gas diffusion, von Neumann–Mullins trends for grain growth and
foam coarsening, 2D/3D grain topology against Euler and the steady-state mean
face count 13.766 of Mason, Lazar, MacPherson and Srolovitz, equal-volume
Kelvin/Weaire–Phelan cell costs (5.306 and 5.288), sparse-route scaling in sites
and declared labels, and the mesh route on a sphere. Its resource probe also
executes a 50,000-label/four-site uniform dense route, records the avoided
quadratic coefficient bytes, and verifies pre-decomposition refusal of
nonuniform coefficient storage above policy. Candidate profiles come from
`threshold_dynamics_candidate_profiles()`.

The Kelvin seed is an explicit 2 x 2 x 2 BCC supercell whose 16 periodic
Voronoi regions each have 14 faces. The Weaire–Phelan seed is the explicit A15
cubic cell with two 12-faced and six 14-faced regions. An independent replicated
Voronoi construction validates their seed topology and periodic volume before
dynamics; the auction then enforces equal discrete region volumes at every
committed step. For unit tension, threshold energy `E` counts each shared film
once, while the sum of cell surface areas counts it twice. At each spatial
refinement the campaign evaluates the committed geometry at three resolved heat
kernel widths and linearly extrapolates the heat-content perimeter to zero
kernel width; this removes the finite-kernel underestimation without modifying
the dynamics or its safety checks. The reported mean cell cost is
`(2 E/N) / (V_box/N)^(2/3)`, with no boundary correction because the Fourier
kernel and site measure are periodic.

The completion campaign records:

- 2D grain statistics at `512^2`: 100/100 commits, 3000 to 223 grains,
  mean side count `5.964 +/- 0.081` (standard error), maximum 26 of 32
  candidates and no overflow;
- finest extrapolated costs `5.2752 +/- 0.0108` for Kelvin at `64^3` and
  `5.2730 +/- 0.0063` for Weaire–Phelan at `50^3`, respectively 0.58% and
  0.28% from the references. The observed 0.0021 gap is much smaller than the
  0.0406 successive-refinement uncertainty, so no ordering is claimed;
- sphere-mesh errors `0.0290`, `0.0183`, and `0.00486` at 642, 2562, and
  10242 vertices under fixed-final-time joint refinement. The local observed
  orders are 0.67 and 1.91, but no single asymptotic order is claimed.
  Tightening the Taylor tolerance from `1e-6` to `1e-10` changes no labels,
  and the one-step consistent-mass reference matches the lumped-mass labels
  and error; the remaining variation is the critical-regime time/space
  thresholding error.

## Nonclaims

- Capillarity-only threshold dynamics is not a complete grain-growth model: no
  anisotropic or misorientation-dependent energies, solute drag, particle
  pinning, recrystallization or nucleation.
- The 0.3% Kelvin/Weaire–Phelan difference is claimed only if the finest-grid
  cost intervals, using the preceding refinement change as uncertainty, are
  disjoint. Otherwise both absolute costs are compared and their ordering remains
  a nonclaim.
- Auction prices are pressures only under the documented normalization and
  resolution regime; a pinned interface gives no pressure claim.
- Volume constraints hold for equal site measure only; weighted site volumes are
  a separate method.
- The sparse route's truncated stencil is a different discrete kernel from the
  Gaussian; its positivity is certified per step, not assumed.

## References

- B. Merriman, J. Bence, S. Osher, *Motion of multiple junctions: a level set
  approach*, J. Comput. Phys. 112 (1994).
- S. Esedoglu, F. Otto, *Threshold dynamics for networks with arbitrary surface
  tensions*, Comm. Pure Appl. Math. 68 (2015).
- T. Salvador, S. Esedoglu, *A simplified threshold dynamics algorithm for
  isotropic surface energies*, J. Sci. Comput. 79 (2019).
- M. Jacobs, E. Merkurjev, S. Esedoglu, *Auction dynamics: a volume constrained
  MBO scheme*, J. Comput. Phys. 354 (2018).
- M. Elsey, S. Esedoglu, P. Smereka, *Diffusion generated motion for grain growth
  in two and three dimensions*, J. Comput. Phys. 228 (2009).
- D. P. Bertsekas, D. A. Castañón, *The auction algorithm for the transportation
  problem*, Ann. Oper. Res. 20 (1989).
- J. von Neumann (1952) and W. W. Mullins, J. Appl. Phys. 27 (1956), on the
  `(n - 6)` law.
- J. K. Mason, E. A. Lazar, R. D. MacPherson, D. J. Srolovitz, *Geometric and
  topological properties of the canonical grain-growth microstructure*, Phys.
  Rev. E 92, 063308 (2015).
- D. Weaire, R. Phelan, *A counter-example to Kelvin's conjecture on minimal
  surfaces*, Phil. Mag. Lett. 69, 107 (1994).
- L. Acuña Valverde, *Heat content estimates over sets of finite perimeter*,
  J. Math. Anal. Appl. 441 (2016), 104–120.
- O. Misiats, N. K. Yip, *Convergence of space-time discrete threshold dynamics
  to anisotropic motion by mean curvature*, Discrete Contin. Dyn. Syst. 36
  (2016), 6377–6411.
