# Meshfree: end-to-end production and beyond closure plan

## 0. Execution contract and evidence boundary

- Baseline: `dev`, merge `85ad69ac5`, PR #383, implementation commit `4f4dfa02e`.
- This document is a plan, not an implementation or qualification result. Research inspected source, checked-in artifacts, the PR, and public-symbol references; no examples, tests, or benchmarks were executed for this plan.
- Implementation requires a fresh worktree under `/Users/lgleyzer/PHYDRA/phydra-labs/.worktrees/meshfree-closure`, branched from the then-current branch. Copy the authoritative ignored `AGENTS.md` into it before implementation. All code, tests, generated data, benchmarks, and implementation documentation stay there. The plan itself is the only repository change authorized by this request.
- Reconcile any intervening user changes before implementation; never undo them. Record the actual implementation baseline and compare only compatible measured artifacts.
- Complete every phase below. Dependency ordering is not scope reduction. Do not stop after preparation or the first application example and call that closure.
- Proposed new files are explicitly marked **new**. Existing paths denote ownership, not permission to invent a second convention.
- Public changes use clean cutovers. Resolve references with the repository LSP before changing each exported symbol; migrate every package consumer, example, benchmark, qualification driver, test, facade, and document. Remove superseded implementations and aliases rather than preserving parallel APIs.
- Retain native ownership: sparse relations in `phydrax.sparse`; solves, rank/property evidence, preconditioning, and multigrid in `phydrax.linalg`; conic optimization in `phydrax.optim`; nonlinear solves in `phydrax.nonlinear`; chart/tensor geometry in `geometry`/`metrix`; time stepping in `solver`; execution in `phydrax.execution`; transactions/restart in `lifecycle`; calibration in `uq`; release authority in `qualification`.
- New Strict modules are final by default, validate before assignment, use nominal `phydrax.typing` contracts and scientific axis identities, and declare `__strict_contract__ = True` where required. Numerical leaves are dynamic; scientific metadata is static. No internal generation suffixes or schema-version fields.
- Complexity changes are measured before and after at changed symbols. Split by validation, preparation, execution, acceptance, and transaction boundaries; preserve floating-point/reduction order unless a deliberately qualified numerical change requires otherwise.

### What “complete” means

Three different milestones must remain separate:

1. **Implementation closure:** every scope item has an actual public end-to-end path, documented admissibility/refusal, migrated callers, exercised smoke scenario, and targeted regression coverage.
2. **Qualification closure:** the exact declared numerical, derivative, performance, provider, scientific, and operational envelopes have retained evidence. Unavailable hardware, external references, or measurements are explicit blockers, never silently smaller support or a pass.
3. **Production release:** exact support tuples, current trusted evidence, transitive dependencies, and independent release authorization pass the canonical registry/trust process. Code cannot authorize its own release. `docs/CAPABILITY_LIFECYCLE.md:17-33` governs the terminology.

The end-to-end implementation must be finished even when an external release prerequisite remains unavailable. Report the precise remaining qualification/release blocker; do not manufacture signatures or mark research methods released.

## 1. Scope ledger: nothing left implicit

| ID | Required closure | Owning phases |
|---|---|---|
| C01 | Evidence propagation, identity/policy consistency, meaningful status and derivative refusal | P0, P13, P15 |
| C02 | Bounded compiled approximation, reusable neighborhoods, periodic addressing, precision | P1, P12 |
| C03 | Scalable exact/relaxed signed and nonnegative metrics; original-equation and optimality audits | P2, P13 |
| C04 | Mixed boundaries, anisotropic diffusion, coupled elliptic systems, boundary stability | P3, P4 |
| C05 | Multilevel/block/nullspace/cycle/smoother completeness and fine-system cost evidence | P4 |
| C06 | Moving bulk clouds, material/ALE semantics, transient nonlinear PDEs | P5 |
| C07 | Higher-order conservative transport, inflow/outflow, CFL/positivity, stabilization | P5 |
| C08 | Curves and surfaces, higher-order charts, authoritative quadrature, open/sharp patches | P6 |
| C09 | Covariant vector/tensor surface operators and PDEs | P6, P10 |
| C10 | Selectable high-order moving-surface integration and typed geometry-motion laws | P7 |
| C11 | Conservative, constant-reproducing, positive transfer with explicit feasibility | P8 |
| C12 | Physical surface split/merge events, not merely sample resampling | P8 |
| C13 | Higher exterior degrees, orientation/Hodge/boundary/commuting-transfer semantics | P9 |
| C14 | Compatible incompressible flow, Lagrangian GMLS flow, SPH interoperability | P10 |
| C15 | Meshfree elasticity, vector coupling, fluid/solid/surface work exchange | P10, P11 |
| C16 | Material interfaces, real boundary traces, meshfree–FEM/FV and monolithic exchange | P3, P11 |
| C17 | CPU/GPU precision/resource envelopes; distributed arbitrary point ownership | P12, P16 |
| C18 | Fixed-support shape sensitivities, fixed-active KKT, frozen-remap adjoints, nonsmooth contracts | P13 |
| C19 | Error-driven p/h/support adaptivity, certified learned laws/corrections, calibration, hybrid Schwarz | P14 |
| C20 | Long-running production runtime, durable restart/migration, retained evidence and release | P15, P16 |

Permanent mathematical boundaries are not missing implementations: arbitrary samples cannot certify global reach; arbitrary clouds cannot guarantee stencil or positive-metric feasibility; arbitrary changing topology need not have a smooth derivative; conformal coverage needs its stated statistical assumptions. The public APIs must implement these distinctions and refuse unsupported claims.

## 2. Corrections to the preliminary brainstorm

These corrections are binding design decisions.

1. **CG convergence does not certify global rank.** Local SVD, sparse pivot profiles, and Lanczos estimates are not interchangeable with a complete rank/nullspace certificate. An exact forward metric may be admitted by its full residual and minimum-norm KKT audit without claiming global rank; a mathematical geometry derivative needs stronger regularity evidence.
2. **Matching Betti numbers does not establish geometric fidelity.** A Rips/clique complex is an abstract combinatorial realization. Production higher-form PDEs need an authoritative oriented complex tied to geometry, interpolation, measures, and an independently verified consistency route.
3. **A graph solenoidal projection is not Navier–Stokes.** A dense radius graph may contain many artificial cycle modes. Physical velocity reconstruction, boundary closure, pressure stability, viscosity, transport, and time integration must be qualified separately.
4. **The signed-metric wall is not proven to be solely ordering.** Source verifies per-edge quadratic Schur-product enumeration, rank-elimination work, natural-order factorization, and an all-pairs radius witness. Measure these phases separately; no universal fill law or predicted speedup follows from source inspection.
5. **Do not change minimum norm through preconditioning.** With `w = sqrt(Phi) z`, the objective is `||z||`. Arbitrary right preconditioning changes that norm. Native LSMR currently rejects preconditioning; extend its owner only with objective-preserving semantics.
6. **Converged root and contraction certificate are distinct.** Current Lipschitz-law tests deliberately admit a converged root without a contraction certificate. Preserve that distinction, and add an explicit scientific acceptance policy only where the consumer requires uniqueness/coercivity.
7. **Boundary quadrature is not automatically needed in strong collocation.** It is required for integrated natural loads, weak/SBP forms, trace pairings, and flux accounting. Do not silently turn collocated Neumann rows into a different weak problem.
8. **One-sided support does not require ghost points everywhere.** Use actual labeled-side points when unisolvent. An explicitly selected ghost route needs its own scientific extension law and evidence.
9. **Rolling temporal history differs from an archive.** Long runs must retain all live mathematical histories and stage/controller state, not indefinitely append every accepted state to a small fixed array. Archive output is bounded and durable through the existing runtime.
10. **Derivatives of values across a frozen remap are valid.** Integer topology decisions remain fixed, but the remap’s coordinate transpose/Hilbert adjoint can propagate field sensitivities. Do not stop every numerical derivative just because an epoch changed.
11. **No blanket O(N) promise.** Fixed-capacity actions can have controlled storage while search, ordering, rank, conic solves, and factorization remain superlinear. Report capacities, algorithmic work, measured scaling, and refusal.
12. **Current surface operators already use padded `lax.map`.** Reuse that pattern; the host bulk stencil preparation is the eager chunk loop needing cutover. Do not rebuild an already vectorized surface row loop.

## 3. Dependencies and integration ownership

| Phase | Depends on | Principal result |
|---|---|---|
| P0 | baseline | Trustworthy consumer contracts and migration ledger |
| P1 | P0 | Shared compiled local approximation and support refresh |
| P2 | P1 | All-equation scalable metric/constraint solves |
| P3 | P1, P2 | Boundary/system/trace-ready spatial problems |
| P4 | P2, P3 | Reusable scalable preconditioning |
| P5 | P1–P4 | Bulk transient/ALE/conservative transport |
| P6 | P1–P3 | Generalized intrinsic geometry and tensor PDE operators |
| P7 | P4–P6 | Stage-correct moving-surface runtime |
| P8 | P2, P5–P7 | Complete remap and physical topology transaction |
| P9 | P2, P3, P6 | Geometry-authorized higher-form realization |
| P10 | P3–P7 | Physical flow and mechanics workflows; P9 for higher-form variants |
| P11 | P3, P6–P10 | Mixed-method/vector/monolithic coupled workflows |
| P12 | P1–P4; consumers integrate as ready | Real accelerator/distributed execution |
| P13 | P1, P2, P6–P11 | Honest end-to-end sensitivity contracts |
| P14 | P4, P8, P11, P13 | Adaptive, learned, calibrated, hybrid workflows |
| P15 | P0, P5, P7, P8, P11–P14 | Durable long-running execution and migration |
| P16 | every phase | Qualification artifacts, exact release dossier and blockers |

After shared contracts are frozen, independent work may run concurrently: P3/P4/P6 after P2; spatial distribution during P5–P11; derivative substrate work alongside consumers; geometry and UQ/reference curation independently. One integration owner controls shared facades, `_views.py`, solver result types, native linalg policies, and qualification declarations. Children skip checks mid-flight; the integration owner runs each selected final check once after integration.

## P0. Contracts, evidence, and identity

### Files and changes

- `phydrax/solver/_fixed_step.py`: preserve native method evidence through retry selection, fixed-step advance, rollout, and solution. Keep failed-attempt evidence distinguishable from accepted-state evidence. Introduce a bounded retention policy, not unbounded arrays for every attempt.
- `phydrax/solver/coupling/_method_participants.py`: extend `_NativeStep`, `_SubstepCarry`, and participant results so real method evidence survives substeps and rollback. An evidence reducer must be method-owned or declared; generic summation/max of arbitrary fields is prohibited.
- `phydrax/solver/_partitioned_coupling_types.py` and the coupling evaluation/runtime consumers: carry participant evidence through window sweeps and terminal results. When heterogeneous participant PyTrees differ, prepare a static structure and ownership map rather than force one undifferentiated scalar.
- `phydrax/solver/coupling/_surface_exchange.py`: publish query coverage, actual amount transfer, status, nonlinear diagnostics, and exact-vs-relaxed metric evidence into the surviving result stream.
- `phydrax/discretization/_point_cloud_pde.py`: bind tolerance, assembly/materialization/preconditioning policies, boundary semantics, and precision into plan identity. Preserve exact source/numeric revision identity in refreshed plans.
- `phydrax/discretization/meshfree/_stencils.py`: make method-specific policy applicability explicit. GMLS owns a weight kernel; PHS owns its radial power. Cleanly migrate configuration so changing an ignored kernel does not pretend to change PHS numerics.
- `phydrax/discretization/_point_cloud.py`: make refreshable-neighborhood capability truthful once P1 implements refresh; do not retain metadata that advertises a nonexistent method.
- `phydrax/discretization/meshfree/_exterior_transport.py`: separate raw candidate diagnostics from accepted values and define derivative validity for failed CFL/positivity/metric admission. No clamping and no successful gradient through a refused public state.
- `phydrax/discretization/meshfree/_profiles.py`: preserve low-level 1-D support and explicitly add its qualified tuple rather than disallow working 1-D code to fit a 2-D/3-D profile. Keep dimensional claims granular.

### Regression and completion

Use the actual exchange method through a multi-substep participant, retry/rollback, and fixed-step rollout. Verify a physical amount balance, a refused attempt, retained diagnostic evidence, and unchanged accepted state. This is not a mock forwarding test. Extend existing fixed-step/coupling tests plus `tests/integration/test_meshfree_bulk_surface_workflow.py`.

Identity tests must exercise stale policy/revision refusal through a real consumer, not pin source text or the spelling of fingerprint dictionaries. Preserve current coercivity-versus-convergence tests.

## P1. Compiled local preparation and support lifecycle

### Files and changes

- `phydrax/discretization/meshfree/_stencils.py`: split host declaration/admission from the pure numerical fit; add a stable module-level compiled chunk entry point; pad final chunks and map fixed-size chunks on device. Build all requested functionals once, share each fit/factorization, reduce reports on device, and synchronize only at the declared host admission boundary.
- The same file: add `PreparedLocalStencils` numeric refresh on frozen `RowRelation`, with coordinate/functional/basis identity, row rank/condition/amplification/moment evidence, and status-returning refusal. Retain GMLS/PHS operation order where unchanged. Do not change SVD rank thresholds as a performance shortcut.
- `phydrax/discretization/meshfree/_neighbors.py`: accept a canonical addressing/support declaration, physical periodic cell, per-axis periodicity, active/stable point identities, candidate/chunk/pair capacities, and effective precision. Fixed kNN and fixed-radius routes remain distinct.
- `phydrax/discretization/spatial/_neighbor_query.py`: add a bounded annulus/shell completeness witness. For a declared admissible displacement delta, certify inclusion/exclusion around radius plus/minus 2 delta; a missed owner, threshold tie, or capacity overflow refuses. It need not find the largest possible global trust radius to provide a valid conservative trust bound.
- `phydrax/discretization/meshfree/_exterior.py`: replace the all-pairs radius-gap loop with that witness, including periodic image semantics. Charge query work separately from metric symbolic work. Never infer “no excluded pairs” merely because a capacity-truncated query returned none.
- `phydrax/discretization/_point_cloud.py`: add a status-returning fixed-support refresh result retaining anchored reference geometry, stable IDs, measures, boundary data, relation, and dynamic weights. New geometry/support discovery belongs to an epoch boundary, not to derivative application.
- `phydrax/discretization/_point_cloud_view.py`: use the shared compiled fit with admitted fixed candidates; explicitly rebind queries after owner revision changes. Keep polynomial and positive Shepard routes distinct.
- `phydrax/discretization/meshfree/_surface.py`, `_surface_geometry.py`, `_multilevel.py`, `_surface_transfer.py`: consume the same prepared fit/refresh substrate instead of adding independent fitting loops.
- `phydrax/discretization/meshfree/_capacity.py`: support stable-ID active/storage maps for bulk as well as surfaces. Computational padding must not create physical zero-mass nodes in positive spaces.

### Acceptance

Both approximation methods, 1-D/2-D/3-D, degrees 2–4, arbitrary admitted functionals, fixed support refresh, partial/masked rows, and irregular final chunks. Check independent nonpolynomial derivatives as well as polynomial reproduction; actual weighted transpose/Hilbert adjoint; coordinate JVP/VJP under a strict support margin; duplicate/tie/rank/condition/overflow refusal. Compare chunk capacities for numerically equivalent results, not a particular internal batching layout.

A smoke changes coefficients, coordinates within trust, RHS, and step size dynamically and records lowering/compilation reuse. Benchmarks separate cold compilation from preparation and warm refresh; cached small-N timing is not a speedup claim.

## P2. Scalable metrics and native constrained numerics

### Mathematical contract

The existing exact signed objective is minimum weighted norm under every original moment equation:

`minimize 0.5 wᵀ Phi⁻¹ w, subject to A w = b`.

Set `S = sqrt(Phi)`, `B = A S`, `w = S z`; solve exact minimum Euclidean norm `B z = b` from zero start. Row scaling can aid exact compatible execution only if the original residual and objective are audited. Preserve the original relaxed objective exactly, including its original row scaling: `0.5 ||z||² + rho/2 ||D⁻¹(A S z − b)||²`. Represent it with native stacked actions and a diagonal identity term. No explicit inverse, Schur matrix, or rank-profile pass is required merely to execute this forward action.

Feasible forward convergence, minimum-norm optimality, full rank, and derivative regularity are separate facts. Report what is unavailable. A stalled iterate alone is not an infeasibility proof.

### Files and changes

- `phydrax/linalg/_problems.py`: extend `MinimumNormProblem` to legitimate consistent rectangular cases with redundant rows, including source size below target size. Reject damping for the exact constrained objective. Do not silently route it as unconstrained least squares.
- `phydrax/linalg/_plans.py`, `backends/_native_krylov.py`, `_runtime.py`: retain zero-start LSMR/GeneralizedLSMR and true constraint residual semantics. Add a numerical minimum-norm stationarity witness/dual residual using native actions, without materializing a global nullspace. Distinguish incompatible RHS with a validated left-null/dual witness from unclassified nonconvergence or resource refusal.
- `phydrax/linalg/_structured_operators.py`, `_operators.py`: reuse diagonal/composed/`StackedLinearOperator` actions for metric transforms and regularization. Do not add a meshfree-local solver wrapper.
- `phydrax/linalg/_rank.py`, `_sparse_rank.py`: retain the pivot-profile nonclaim; add exact scope, revision, cutoff/gap, complete-nullspace or bounded rank-revealing evidence where available. A certificate can refuse its own work/materialization limit. Local blocks cannot stand in for global certificates.
- `phydrax/linalg/_policies.py`, `_costs.py`, `_results.py`: charge original/stacked rows, forward/adjoint action counts, transformed scratch, Krylov and certificate storage, and report original constraint/optimality residuals separately. Extend native rectangular preconditioning only through a formulation preserving the requested norm; keep unsupported right transforms refused.
- `phydrax/discretization/meshfree/_exterior_metric.py`: replace the exact signed Schur-pair arrays and unconditional sparse Cholesky route with the all-equation transformed solve. Replace relaxed signed execution with the mathematically equivalent scaled stacked problem. Delete the superseded Schur storage and comments; keep all original moment, sign, amplification, status, and slack audits.
- The same file and `_exterior.py`: parameterize declared polynomial moment degree, deterministic multi-index order, row scaling, and RHS from the owning differential functional. Higher moments can be infeasible with a symmetric/nonnegative edge metric; retain a precise refusal or explicitly requested relaxation, never truncate the moments.
- `phydrax/optim/_programming/_native_hsd.py`, `_native_conic.py`: reuse native matrix-free conic directions for nonnegative metrics. Complete original-coordinate KKT, complementarity, and primal/dual infeasibility evidence at the scale used by consumers. Numerical inability to construct a certificate is not labeled infeasible.
- `phydrax/discretization/meshfree/_conservation_solve.py`, `_exterior.py`: replace mandatory large background/coercivity factorization with a policy-selected native solve/property assessment. A property audit may be unavailable while a separately admitted root converges. Charge any explicitly selected factorization against shared resource limits.

### Sparse ordering, when an actual factor is requested

- **New** `phydrax/linalg/_sparse_ordering.py`: own stable graph-ordering preparation, validated permutations/inverses, IDs, work bounds, and deterministic tie breaking.
- `phydrax/linalg/_sparse_lu_analysis.py`: reuse/extract its existing COLAMD-like quotient-graph logic instead of creating a second minimum-degree implementation.
- `phydrax/linalg/_sparse_factorizations.py`: accept the prepared ordering; provide deterministic AMD and graph/geometric nested-dissection plans. Extend closed selector parsing once, retain natural/RCM when deliberately selected, and refresh without recomputing ordering on unchanged structure.
- Ordering is not a fallback that makes every factor affordable. Benchmark actual fill, symbolic work, and factor bytes. Keep matrix-free metrics matrix-free even when ordering improves other consumers.

### Acceptance

Independent bounded dense SVD/QP reference for weighted optimum; redundant compatible and incompatible equations; exact versus relaxed equality/slack/objective; nonnegative feasible/infeasible/weakly complementary cases; unchanged objective under admitted transforms; resource refusal before allocation. Exercise Q8 at 256 and 512 without automatically increasing the old symbolic budget or introducing a dense path. Rank/certificate/derivative evidence may honestly be unavailable; law/source sensitivities must still use their own admitted fixed-metric contract.

Extend `test_exterior_metric.py`, `test_conservation_solve.py`, `tests/unit/linalg/test_sparse_rank.py`, native minimum-norm/LSMR tests, `tests/unit/optim/test_sparse_hsd.py`, and sparse factor-resource tests. Add **new** `tests/unit/linalg/test_sparse_ordering.py` for deterministic externally usable permutations, original solve residuals, fill/work refusal, and refresh behavior.

## P3. Boundary, elliptic, reconstruction, and SBP closure

### Files and changes

- `phydrax/discretization/_point_cloud_pde.py`: revise `PointBoundaryPlan` to one canonical per-entity declaration of boundary kind, component, values/Robin data, side, normal, and physical measure. Cleanly migrate the scalar `kind` constructor. Mixed Dirichlet/Neumann/Robin rows, periodic boundaries, and multiple disconnected Neumann components have explicit gauge/nullspace and compatibility semantics.
- The same file: extend `PointDiffusionOperator` to declared scalar or symmetric tensor diffusivity using native property evidence. Collocated product-rule and quadrature-adjoint forms remain separate. Native positivity/conditioning evidence must reach the solve result; no symmetrization/jitter repair by default.
- Keep `PointCloudPoissonPlan` the canonical scalar elliptic plan. Replace the restrictive preconditioner string with the native linear/preconditioning policy and prepared hierarchy contract, migrating all callers. Add a rectangular oversampled route with explicit independent source and target support/row measures through native least squares; its residual and boundary weighting differ from square collocation and must be declared.
- **New** `phydrax/discretization/meshfree/_systems.py`: assemble coupled block equations, component-specific boundaries, and nonlinear spatial residuals through native `BlockSpace`, named blocks, sparse assembly, and prepared nonlinear differentiation. This is not a vector-valued alias for scalar Poisson.
- **New** `phydrax/discretization/meshfree/_boundary.py`: own side-labeled point support, one-sided admission, boundary/interface quadrature, and prepared boundary actions. Consume `geometry/_atlas.py`, `_cubature.py`, `geometry/surface/_contracts.py`, and `discretization/_side_actions.py`; raw point coordinates never fabricate facets or material-side identity.
- `phydrax/discretization/_point_cloud_view.py`, `_views.py`: publish actual derivatives of the admitted reconstructed field, with order/regularity and support evidence. Distinguish evaluation of a discrete differential field from differentiating a query-dependent MLS interpolant; only advertise the interpretation the native request contract represents. Keep explicit surface intrinsic derivatives distinct from unsupported ambient extensions.
- `phydrax/discretization/_point_cloud_pde.py` and `_boundary.py`: keep `point_sbp_report` as a full sparse coefficient audit. Add an explicitly selected constrained derivative preparation enforcing polynomial reproduction and `M D + Dᵀ M = B` for a supplied positive cubature and boundary form using the native sparse optimization owner. It may be infeasible; do not relabel arbitrary dissipative derivatives as SBP. Boundary/SAT penalties follow an actual admitted identity and discrete energy contract.

### Acceptance and tests

Manufactured variable anisotropic diffusion on irregular 2-D/3-D clouds, mixed labels/corner ownership, discontinuous material coefficients with one-sided queries, gauge/compatibility for each disconnected component, independent original physical residuals, and boundary-normal orientation. Oversampled fits must converge on a nonpolynomial oracle and refuse bad row weighting/rank; high-order one-sided rows must use real samples.

Extend `test_point_cloud_poisson.py`, `test_prepared_field_queries.py`, `test_discrete_field_views.py`, and `test_side_actions.py`; add **new** meshfree `test_boundary.py` and `test_systems.py`. Tests distinguish channel-wise payload application from a genuinely coupled block PDE. SBP tests check the full coefficient/energy identity and an independently manufactured boundary flux, not a finite probe advertised as a certificate.

## P4. Multilevel and preconditioning closure

### Files and changes

- `phydrax/discretization/meshfree/_multilevel.py`: reuse compiled P1 transfers; replace serial row dispatch with a prepared stable-ID graph coarsening plan. A deterministic MIS may run bounded device rounds; nonconvergence or no progress is explicit. Do not promise that every boundary-heavy cloud coarsens.
- The same file: parameterize reproduction degree, retained boundary/feature sets, scalar versus coupled component spaces, and complete near-nullspace candidates. Exact retained-node inclusion and stiffness-coordinate restriction semantics stay intact. General rigid-body/constant/component kernels replace the hard-coded one-dimensional constant-only validation when mathematically declared.
- `phydrax/linalg/_multigrid.py`, `_multigrid_setup.py`: retain V/W/F/full ownership; expose the selected native cycle through the meshfree preparation. Numeric refresh reuses symbolic products when dependencies truly match, and rebuilds at a declared topology boundary otherwise.
- `phydrax/linalg/_multigrid_smoothers.py`, `_preconditioning.py`, `_block_preconditioning.py`: prepare block-Jacobi, symmetric/ordered Gauss–Seidel, ILU/ILUT, and admitted polynomial smoothers with actual property requirements. GPU parallel smoothers are explicit choices, not a silent replacement of a nonsymmetric solver. Generalize lower-precision preconditioning beyond Jacobi only where accumulation/residual and refinement semantics are supported.
- `phydrax/discretization/_point_cloud_pde.py`, `_systems.py`: allow meshfree hierarchy, native smoothed aggregation, ILU, Schwarz, or supplied native builders via one owning policy. Use PCG/MINRES only with native properties, GMRES for fixed nonsymmetric preconditioning, FGMRES for flexible actions. Solve the original equation regardless of accelerator choice.

### Acceptance and evidence

Scalar and elasticity/block transfers, affine/higher-degree reproduction, multiple complete nullspaces, boundary-heavy no-progress, V/W/F/full, refresh/reprepare, and coarse resource refusal. Track grid/operator complexity and near-nullspace defects. Compare identical fine equations at multiple capacities with native ILU and smoothed aggregation; report true residual, total setup/compile/solve/refresh cost and logical/physical memory. No gate requires multigrid to beat ILU on every cloud; claimed scaling profiles must meet their own recorded thresholds.

Extend `test_multilevel.py`, native multigrid-lifecycle tests, sparse preconditioner resource tests, and `benchmarks/meshfree_multilevel.py`.

## P5. Bulk transient and material/ALE workflows

### Files and changes

- `phydrax/discretization/_point_cloud.py`: fixed-support motion uses P1 refresh; support/population changes use a staged topology epoch and native field transfer. The physical measure is named: Eulerian/ALE quadrature volume is not material mass.
- **New** `phydrax/discretization/meshfree/_evolution.py`: own semidiscrete spatial problem assembly for advection, diffusion, reaction, nonlinear constitutive terms, and selected stabilization. Publish native temporal problem/state/capacity contracts; no local time integration loop.
- `phydrax/discretization/meshfree/_exterior_transport.py`: separate instantaneous conservative edge flux/rate from time integration. Add geometric velocity-to-integrated-volume-flux ownership, boundary inflow/outflow, high-order reconstruction, and paired limited antidiffusive fluxes. Use native sparse gather/transpose; limiter decisions remain explicit differentiability boundaries.
- **New** `phydrax/discretization/meshfree/_transport.py`: prepare the substantial high-/low-order flux and bounds/CFL admission invariant, not a wrapper renaming the old kernel. Surface relative advection consumes this same owner.
- `phydrax/solver/_conservation_temporal.py`, `_balance_law_composition.py`, `_temporal_method.py`: extend only missing capability/result evidence; reuse native SSP-RK, IMEX, Rosenbrock/BDF, implicit stage solves, and adaptive controllers. Positivity of a spatial flux is not positivity of every temporal method.
- `phydrax/discretization/meshfree/_stabilization.py`: compose hyperviscosity with the spatial residual and explicit spectral estimate/evidence. Estimated radii are not rigorous CFL bounds. Do not automatically turn stabilization on after a failure.
- `phydrax/discretization/particle/_core.py`, `_population.py`, `_verlet.py`: consume their mass, persistent-ID/lineage and cached-neighbor contracts for genuine material-particle motion. For a quadrature cloud provide an explicit measure/coordinate map, rather than relabeling quadrature as particle mass.

### Acceptance and tests

**New** `test_bulk_evolution.py` and `test_transport.py`: advected nonpolynomial pulse with inflow, diffusion/reaction MMS, anisotropic/nonlinear implicit solve, mass and positivity under the declared spatial/temporal CFL, unaccepted candidate without clipping, periodic seam motion, fixed-support trust exhaustion, and full-step rollback. Verify translation and affine dilation GCL, material-minus-mesh sign, and an adaptive accepted-grid trajectory with retained solver/controller status.

**New** `examples/meshfree_bulk_advection_diffusion.py`: run the actual semidiscrete problem with at least two temporal methods, an accepted motion refresh, a refused capacity/support candidate, and original mass/error evidence. Keep independent spatial and temporal refinement campaigns.

## P6. Intrinsic geometry, open/sharp patches, vector/tensor operators

### Files and changes

- `phydrax/discretization/meshfree/_surface_geometry.py`: generalize to explicitly declared intrinsic dimension and embedding, covering curves in R2/R3 and sheets in R3. Use metrix rank/codimension/metric evidence; never infer dimension from equal array sizes. Implicit, authoritative chart, and sample-estimate sources remain distinct.
- `phydrax/metrix/_chart.py`, `_atlas.py`, `_embedded.py`, `_connection.py`, `_operators.py`, `_tensor.py`, `_patchwise.py`: reuse chart transitions, tangent frames, connections, variance, and covariant transformations. Extend these owners if an actually reusable operation is absent; no hand-coded second Christoffel/tensor calculus in meshfree.
- `phydrax/discretization/meshfree/_surface_geometry.py`: parameterize sampled height-fit degree and oversampling, retaining independent fit/normal/curvature errors. Sample-only sources have local diagnostics, not invented global tube/topology certificates. Frame switches need gauge-invariant outputs and explicit regularity.
- `phydrax/discretization/meshfree/_surface_quadrature.py`: add geometry-authorized chart/Jacobian cubature using `geometry/_atlas.py`, `_cubature.py`, `geometry/surface/_high_order.py`, and `_g1_multipatch.py`. Retain tangent-Voronoi as an estimate; compile bounded polygon clipping where useful. Supplied total area is not an independent area test.
- `phydrax/discretization/meshfree/_surface.py`: assemble dimension-general scalar operators with native metric solves and chart stencils. Open boundaries use one-sided physical supports plus declared boundary quadrature; sharp seams use separate smooth patches with oriented side/interface data, never average normals across a crease.
- **New** `phydrax/discretization/meshfree/_surface_pde.py`: own tangential vector and typed tensor operators with connection terms and genuine component coupling. Bochner and Hodge Laplacians are explicitly different. Use native block/saddle operators for tangency and incompressibility constraints rather than treating each Cartesian component as an unrelated scalar.
- `phydrax/discretization/meshfree/_surface.py` reconstruction: publish intrinsic derivative requests on declared charts; still refuse invented ambient volumetric derivatives or facet cells for a raw sample source.

### Acceptance and tests

Circle and space curve, sphere and torus, authoritative open planar/curved patch, oriented crease with two sides, and anisotropic sampling. Compare independent length/area/curvature and actual geometry gradients, including a source-program change with equal samples. Quadrature gets its own refinement/error campaign.

Extend existing geometry/quadrature/operator tests; add **new** `test_surface_charts.py` and `test_surface_pde.py`. Test chart/gauge covariance, tangency, known vector spherical harmonics, metric compatibility, tensor re-expression, weak Green/boundary identity, and Bochner–Hodge curvature distinction. Non-Euclidean source support must carry the owning metric and geodesic/local-support premises or refuse; Euclidean distance alone cannot be advertised as intrinsic-distance certification.

**New** `examples/meshfree_open_surface_diffusion.py` and `examples/meshfree_surface_vector_pde.py` exercise actual solves, boundaries, and error/evidence.

## P7. Stage-correct moving-surface dynamics

### Files and changes

- `phydrax/discretization/meshfree/_moving.py`: accept native tableau/method selection, prepare stage operator/solver templates once, and refresh numerical coefficients. Remove per-stage cold solve preparation. Geometry, measures, rates, reaction, and relative flux must be evaluated at the method’s actual stage states/times.
- The same file: define a discrete GCL admission policy appropriate to the chosen stage rule. The existing endpoint measure-rate residual is diagnostic, not a universal zero-error requirement; a nonlinear measure history may legitimately produce an O(dt²) endpoint difference. Conservation of content, GCL consistency, geometry trust/tube, and solver success remain independent statuses.
- **New** `phydrax/discretization/meshfree/_motion.py`: typed prescribed velocity, authoritative evolving level-set/chart motion, curvature-normal flow, and bulk-driven motion. Own the material/mesh normal-tangent split and measure-rate source identity; reuse geometric/metrix and nonlinear owners.
- `phydrax/discretization/meshfree/_shifting.py`: shifting is a mesh velocity with relative advection, not a physical normal-motion law. Advance its flux through the P5 native temporal composition rather than assuming a single Euler correction is high-order.
- `_moving.py`: replace append-only lifetime history with a bounded rolling live-history set, stage/controller state, and an archive cursor. Preserve the current ability to transport every still-live history independently. Production archive retention belongs to P15.

### Acceptance and tests

Prescribed dilation, translation/rotation, nonuniform normal motion with an independent measure rate, shrinking circle/sphere curvature flow before singularity, tangential mesh redistribution, and bulk-driven geometry. Spatial, temporal, quadrature, and geometry errors are separated. Force tube/trust, nonconservative diffusion, implicit-solve, and archive/history-capacity refusals; verify atomic rollback and all status evidence.

Extend `test_moving_surface.py`, `test_surface_shifting.py`, `examples/meshfree_moving_surface_reaction_diffusion.py`, and `benchmarks/meshfree_moving_surface.py`. Add a second-/higher-order IMEX refinement study and long-run smoke exceeding the former append-only history capacity.

## P8. Joint transfer, resampling, and physical topology

### Joint transfer invariant

For concentration transfer T, test/request separately `T 1 = 1`, `w_newᵀ T = w_oldᵀ`, coefficient nonnegativity, and declared reproduction moments. If total source and target measure differ, simultaneous conservation and constant preservation is generally impossible: summing the two equations proves the obstruction. Material dilation is not disguised as a same-surface resampling. Local support can make an otherwise globally feasible transport problem infeasible.

### Files and changes

- `phydrax/discretization/meshfree/_surface_transfer.py`: replace misleading unconstrained “high-order” claims with one explicit constraint/objective declaration, retaining distinct requests for conservative signed, conservative-positive, and joint conservative/constant/moment routes. Solve minimum-change corrections with native rectangular solves or sparse conic optimization. Audit every requested equation, sign, coverage, correction, and objective; retain infeasibility evidence or unresolved provider failure separately.
- Generalize the reusable point-support transfer portion into **new** `phydrax/discretization/meshfree/_transfer.py`, used by bulk and surfaces; keep `SurfaceTransferPlan` only for its real surface/chart/measure-specific preparation, not as an alias to a generic plan.
- `phydrax/discretization/_transfer.py`, `_topology_epoch.py`: preserve coordinate dual versus Hilbert adjoint, source/target measure/identity, and event semantics. Admit derivatives with respect to values on a frozen transition; geometric/topological selection derivatives have their separate contract.
- `phydrax/discretization/meshfree/_resampling.py`: prepare bounded deterministic sample repair, quality indicators, projection, and capacity proposals. It remains distinct from physical surface topology change.
- **New** `phydrax/discretization/meshfree/_epochs.py`: own one complete staged dependency transaction for geometry, measures, points, routes, stencils, metric, hierarchy, all live histories, coupling queries, predictors/controller state, and derivative capabilities. Use `lifecycle/_composition_rebind.py` and `_transaction.py`, not a private rebuild registry. A failure in any history/dependency returns the source composition unchanged.
- `phydrax/geometry/multiregion_surface/_events.py`, `_topology_transitions.py`, `_transaction.py`, `_transfers.py`: remain the physical event authority. Consume committed split/merge/pinch/region lineage and CCD/volume/geometry evidence in a meshfree epoch adapter. Do not manufacture a physical event from insertion/deletion of sample points.

### Acceptance and tests

Joint feasible positive/constant/conservative transfer; unequal-area obstruction; sparse support infeasibility; uncovered source; high-moment/positivity conflict; signed high-order route; dual/adjoint/value JVP; repeated remap with independently measured error; all-history failure rollback. Preserve existing tests proving conservation does not imply constant reproduction.

Extend `test_surface_transfer.py`, `test_surface_resampling.py`, `test_moving_surface.py`; add **new** `test_epochs.py` and `tests/integration/test_meshfree_surface_topology_workflow.py`. Reuse native multiregion event fixtures to exercise an actual physical split and merge, field/region lineage, state continuation, and derivative boundary, without reimplementing their numerical oracle.

## P9. Higher exterior degrees with geometry authority

### Files and changes

- **New** `phydrax/discretization/meshfree/_complex.py`: prepare a meshfree approximation/reconstruction over a supplied oriented `CellComplexTopology` and geometry/measure authority, with degree-specific spaces, interpolation, sparse Hodges, relative boundaries, and evidence. Keep the current graph-only `_exterior.py` honest about top degree 1.
- `phydrax/discretization/_cell_complex.py`, `_cochain.py`, `_cochain_hodge.py`, `_boundary_complex.py`, `_cell_de_rham.py`: own incidence, restriction, pairing, and d/codifferential semantics. Local reconstructed form moments bind these native owners; they do not create a second discrete exterior algebra.
- `phydrax/exterior/_de_rham.py`, `_traces.py`, `_products.py`, `_cohomology.py`, `_spectra.py`: consume the new realization and publish applicable operators through the canonical interfaces. Preserve degree, twist, orientation, primal/dual, and fiber identity.
- Existing meshing/simplicial geometry owners provide constrained/domain or restricted-surface complexes. An alpha route must carry domain/surface restriction and degeneracy/provider evidence; an unconstrained point triangulation does not prove it represents the requested domain.
- A bounded radius-clique route may also be implemented in `_complex.py` for abstract research, with explicit simplex/work capacities, orientation, d-squared-zero and topological evidence. It never receives a continuum-fidelity profile solely from matching Betti numbers.
- `phydrax/topology`: use the native exact chain/homology owner for abstract ranks/generators. Metric Hodge positivity is checked separately, and polynomial/form consistency separately again.

### Acceptance and tests

Authoritative 2-D and 3-D complexes with manufactured differential forms, commuting interpolation `d I = I d`, orientation reversal, absolute/relative boundary conditions, Stokes identities, Hodge duality, and harmonic spaces. Include a domain with a hole and a curved closed sheet. Exercise a cochain heat or compatible Maxwell action to prove degree 2/3 is used by a real consumer. Refuse unsupported domain identity, Hodge positivity, simplex capacity, or trace orientation.

Add **new** `test_complex.py`, `tests/integration/test_meshfree_higher_forms_workflow.py`, and `examples/meshfree_higher_forms.py`; extend affected exterior realization tests and qualification. Abstract-clique and geometry-authorized records stay separate.

## P10. Physical incompressible flow and mechanics

### Compatible pressure owner

- `phydrax/solver/_compatible_systems.py`: cut over `CompatibleIncompressibleProjection` and `CompatibleVariableDensityProjection` from retained dense pseudoinverses to prepared native operator solves. Accept the canonical cochain realization rather than require a multidimensional structured bridge merely to execute degrees 0/1. Preserve structured consumers through migration to the owner, not a compatibility adapter.
- Projection results add original pressure/divergence residual, compatibility/gauge/nullspace and native solver status, work, precision, and derivative evidence. Variable density owns its edge coefficient interpolation and positivity evidence.
- LSP reference checks for this plan found exports in `phydrax/solver/__init__.py` and consumers in `tests/unit/discretization/test_fd_production.py` and `test_fd_completion.py`; update `docs/api/discretization/finite_difference.md` too. Re-run references at implementation because the repository may change.

### Full physical workflows

- **New** `phydrax/solver/_meshfree_incompressible.py`: compose conservative momentum transport, viscous stress, body forces, reconstruction, pressure projection, and native temporal acceptance. Qualify physical velocity spaces and pressure inf-sup/spurious modes; an arbitrary graph cycle is not accepted as a physical flow mode. Surface Stokes uses the covariant strain/divergence pair and native saddle solve.
- **New** `phydrax/solver/_meshfree_lagrangian.py`: compose particle state/mass/population/Verlet owners with GMLS reconstruction and an admitted pressure correction. The existing `particle/_iisph.py`, `_dfsph.py`, `_barotropic_sph.py`, `_sph_operators.py`, and `_pairwise.py` remain SPH/kernel-particle owners. Add an adapter only for real exchange of reconstructions/pressure/fields; no second IISPH or DFSPH loop.
- Quadrature-volume GMLS ALE and material-mass particle flow are explicit distinct problem representations. Transfer between them uses a measure-aware declared relation, not equal coordinate shapes.
- **New** `phydrax/discretization/meshfree/_mechanics.py`: prepare coupled strain/stress/divergence and traction actions. Reuse `operators/mechanics/_linear_elasticity.py`, `_finite_strain.py`, and existing solid-mechanics constitutive/native nonlinear owners. Implement linear elasticity and an admitted smooth finite-strain constitutive route with native stiffness/Jacobian and energy/work evidence; scalar Laplacians are not an elasticity substitute.
- Use P4 rigid-body/nullspace and mixed-block preconditioners. Near-incompressible mechanics needs an admitted mixed formulation and pressure stability, not an arbitrary large penalty.

### Acceptance and tests

Taylor–Green/periodic divergence and energy, bounded-domain Stokes manufactured velocity/pressure/traction, variable-density compatibility, spurious cycle/checkerboard detection, and surface Stokes with tangent constraint. Pressure and velocity refinement are separate. Particle/GMLS interoperability checks mass, density, divergence, momentum and pair torque only where the declared discretization actually claims them.

Elasticity patch plus independent nonpolynomial displacement/stress, rigid-body zero strain, traction balance, strain energy, near-incompressible mixed response, finite-strain tangent/adjoint, and failure rollback.

Add **new** `test_incompressible.py`, `test_lagrangian.py`, `test_mechanics.py`, `tests/integration/test_meshfree_flow_workflow.py`, and `test_meshfree_mechanics_workflow.py`. Add real `examples/meshfree_incompressible_flow.py`, `meshfree_lagrangian_flow.py`, and `meshfree_elasticity.py`.

## P11. Mixed-method traces, monolithic exchange, and work-consistent coupling

### Files and changes

- `phydrax/solver/coupling/_meshfree_components.py`: publish coupled fields, block residuals/capacities, and nonlinear prepared linearization when supported. Raw point-cloud components stay non-facet; a boundary-authorized component publishes native trace capability through the same canonical component contract with fail-closed capability admission.
- P3 `_boundary.py` plus `discretization/_side_actions.py` and `_views.py`: supply oriented side traces, conormal/traction actions, trace Gram/Riesz spaces, and independent inverse-trace/penalty evidence where a coupling law requires it. No facet identity inferred from cloud shape.
- `phydrax/solver/coupling/_surface_exchange.py`: generalize exchange to reconstruction-capable FEM/FV bulk and meshfree surface while retaining exact query/capacity/measure identity, signed-vs-positive deposition distinctions, and per-window lag. Relocation/epoch transactions rebind all consumers atomically.
- `phydrax/solver/coupling/_prepared_problem.py`, `_acceptance.py`, `_transient.py`: use existing native block/DAE/nonlinear coupling for monolithic transient exchange and admit implicit sensitivities only after P13 supplies forward/KKT/adjoint evidence. No new meshfree Newton loop.
- `_interfaces.py`, `_interface_quadrature.py`, `_laws.py`, and existing FEM interface/FV side-trace owners: consume matching/nonmatching boundary-authorized meshfree traces for mortar/Nitsche/conservative flux exchange. Preserve one-sided material support and jump-condition semantics.
- P10 vector components: use native traction/displacement or velocity/force port laws for a small fluid–solid–surface workflow. Work uses the correct dual pairing, not equal scalar sums of vector coefficients.

### Acceptance and tests

Meshfree–FEM and meshfree–FV interface MMS with displacement/solution continuity and flux jumps; material labels and orientation reversal; conservative bulk/surface amount exchange; positive deposition separately; monolithic nonlinear transient solve and adjoint; stale same-shaped source refusal; moving site/epoch cutover. A vector fluid/solid/surface exchange scenario checks integrated force, boundary work and total balance.

Extend `test_meshfree_components.py`, `test_surface_exchange.py`, `test_meshfree_bulk_surface_workflow.py`, and native coupled nonlinear/transient tests. Add **new** `tests/integration/test_meshfree_mixed_method_workflow.py` and `test_meshfree_fluid_structure_workflow.py`, with executable mixed-method and fluid/structure examples. Schwarz in P14 accelerates these original equations; it does not redefine the coupling physics.

## P12. Precision, accelerators, and distributed point relations

### Files and changes

- **New** `phydrax/discretization/meshfree/_precision.py`: bind native precision roles for geometry/coefficient/basis, fit/factorization, compute/accumulation, residual/certification, communication, checkpoint and output. Reuse `_precision.py`, linalg and particle precision policies. Remove unconditional float64 conversions that defeat an explicitly admitted float32 profile, but never silently lower certification precision.
- `phydrax/discretization/meshfree/_neighbors.py`, `_stencils.py`, `_surface_geometry.py`, `_exterior_metric.py`, `_point_cloud_view.py`: use the resolved precision/backend plan, actual supported local SVD/solve kernels and bounded spatial distance backend. Pallas is selected only where implemented; other stages remain native JAX rather than pretend the entire solver is Pallas.
- **New** `phydrax/discretization/spatial/_distributed_relations.py`: own stable global IDs, owner/process maps, arbitrary uneven partition capacities, owner-local Morton/cell indices, shell expansion, periodic images, global deterministic candidate merge, and kNN/radius completeness. Query owner bounds and remote evidence; do not replicate every target or assume a local result is globally nearest.
- `phydrax/discretization/spatial/_plane_distributed.py`: migrate its existing all-target distributed query to this owner, retaining its public scientific behavior via clean caller migration.
- **New** `phydrax/discretization/meshfree/_distributed.py`: bind row/edge relations to owned rows and halo source columns; exactly-once edge ownership; forward gather and transpose sum; interpolation/operator/field/history migration; capacity/status; execution-group/partition identities.
- `phydrax/_execution_array.py`, `_execution_runtime.py`, `_execution_resources.py`, `_execution_workset.py`: compose NamedSharding, real process-local ingress, collectives, bounded worksets, and placement/resource evidence. Extend only shared missing transport/resource invariants.
- `phydrax/discretization/particle/_distributed.py`, `_distributed_runtime.py`: compose the shared point ownership/packet substrate where particle semantics apply; eliminate assumptions about periodic axis-0 slabs from new arbitrary-owner routes, without changing unrelated existing slab behavior by accident.
- `phydrax/linalg` distributed actions and P4 hierarchy: global norms/compatibility reductions and cross-rank adjoints; coarse-level agglomeration is a prepared ownership plan. Do not gather a global dense pressure/operator on one device as an undisclosed coarse solve.

### Acceptance and tests

CPU and real GPU float32/float64 where supported; mixed-precision preconditioner with high-precision true residual; dtype/rank warnings tested under `strict_jax` for owning surfaces. Adversarial near ties must use certification precision rather than changed neighbor identity from an unnoticed downcast.

Uneven 2-D/3-D partitions, nonperiodic and multi-axis periodic cases, kNN shell crossing at least three owners, pair-once radius edges, incomplete halo/missing-owner/candidate overflow, transpose duality and global conservation, migration overflow rollback, and restart under changed ownership. Numerical reduction order/reproducibility class is explicit.

Add **new** `test_distributed_relations.py`, meshfree `test_distributed.py`, and `tests/integration/test_meshfree_distributed_workflow.py`; update existing distributed-plane/particle/atomistic consumers reached by reference analysis. Forced CPU devices prove only functional distributed parity. Actual multi-host GPU performance requires actual multi-host GPU runs.

## P13. Sensitivities and nonsmooth event contracts

### Files and changes

- P1 `_stencils.py`, `_point_cloud.py`, `_point_cloud_view.py`: publish fixed-support coordinate/source/functional sensitivities with local rank and conditioning evidence. JVP and VJP must be genuine derivatives of the published mathematical map. No masked finite zeros for failed public solves.
- `phydrax/optim/_programming/_conic_sensitivity.py`, `_matrix_free_conic_sensitivity.py`, `_policy.py`: reuse existing projection-KKT sensitivity machinery for nonnegative metrics/transfer. Add strict-complementarity, active-set, full KKT regularity and original-coordinate residual evidence. A forward provider currently declaring no implicit differentiation must have its owner capability updated only after this route is implemented and qualified.
- `_exterior_metric.py`: bind that evidence. Exact signed derivatives require constant-rank/compatible-tangent admission; relaxed regularized derivatives have a distinct all-equation contract. Krylov condition estimates alone authorize neither.
- `_surface_transfer.py`, `_transfer.py`, `_epochs.py`, `_moving.py`: admit state-value tangents/adjoints through a frozen transfer and propagate every live history’s cotangent. Remove the blanket state `stop_gradient` only where the frozen-value contract is valid; geometry/event selection remains separate and explicit.
- `solver/coupling/_prepared_problem.py`, `_acceptance.py`, `_transient.py`: enable actual nonlinear coupled implicit primal/adjoint sensitivities after complete original-equation forward/linearized acceptance; preserve failure guards and static/reprepare refusal.
- Smooth fixed-radius GMLS: implement a bounded candidate-envelope route containing the union of neighbors over the admitted trajectory. Use a fixed physical support radius/scale and a compact kernel with the requested derivative order vanishing at the cutoff. Verify positive weighted design rank throughout. This can admit a zero-weight neighbor entering/leaving the physical support without differentiating integer selection; rebuilding the candidate envelope or changing the unknown population is still an event. Do not extend the claim to ordinary kNN or PHS support changes.
- Genuine active-set changes: publish directional/generalized derivative only through the explicitly selected native generalized-conic policy, with its interpretation and residual evidence; weak complementarity has no default smooth gradient. Genuine split/merge/topology selection has no universal smooth derivative. For smooth transversal event timing with a supplied reset map, compose the existing native event/saltation owner if available and admit only its hypotheses; otherwise return a precise derivative-unavailable result.

### Acceptance and tests

Central differences and JVP/VJP duality for fixed support, exact-compatible tangents, relaxed metrics, strict-active KKT, nonlinear coupled laws/source/boundaries, and frozen multi-history remap. Deliberately cross a kNN tie, candidate capacity, rank loss, weak complementarity and topology event; verify named refusal/NaN sensitivities or the explicitly requested directional contract, never a plausible silent gradient.

Add a smooth-radius entering/leaving-neighbor test with an independent polynomial/nonpolynomial oracle and two-sided derivatives at the cutoff. It must pass without a changed candidate envelope and fail explicitly when that envelope is insufficient.

Extend meshfree derivative tests, `tests/unit/optim/test_conic_sensitivity.py`, `test_quadratic_program.py`, native nonlinear trial-validity tests, and coupled derivative tests. Do not copy the implementation into a numerical oracle.

## P14. Adaptivity, learned corrections, calibration, and hybrid Schwarz

### Files and changes

- **New** `phydrax/discretization/meshfree/_adaptivity.py`: own residual, flux-jump, p-versus-p+1 and geometry/support indicators; stable-ID deterministic marking; bounded point insertion/removal, p changes and support changes. Estimates are indicators unless an explicitly scoped reliability/efficiency bound is established; local fit residual is not a PDE error bound. Accept/reject through P8 complete composition rebind.
- `phydrax/discretization/meshfree/_constitutive.py`: extend scalar laws to genuine coupled component representations and convex potential gradients, using existing `InputConvexNetwork`/`PartiallyInputConvexNetwork` and `nn/operator/representations/_o3.py`/`layers/_o3_tensor_product.py`. O(3) equivariance is not automatic for an arbitrary multivariate ICNN: use invariant convex construction or validated equivariant structure and expose its certificate.
- The same file and `_conservation_solve.py`: preserve immutable external context unless a whole-state monotonicity proof includes state-dependent features. Prove coupled output Lipschitz/strong monotonicity rather than concatenating scalar estimates. Preserve source/boundary/law parameter roles and MODEL authorization.
- **New** `phydrax/discretization/meshfree/_corrections.py`: own a learned candidate correction constrained against the full moment operator. Project with native weighted rectangular/conic actions; test moments, sign margins, energy and flux parity separately. Undirected positive metric coefficients are not antisymmetric fluxes. If a learned correction breaks positivity/coercivity, refuse or solve an explicitly selected constrained correction, never clip and claim exact moments survived.
- `_coverage.py`: stay train-fitted empirical feature support. Calibration uses `phydrax/uq/_conformal.py`, `_process_validation.py`, `_metrics.py`, `_predictive.py`; no second conformal framework under meshfree. Independent complete cases/trajectories define train/calibration/test splits. Report exchangeability, nominal and measured predictive coverage, width, drift, and source/regime identity separately from Mahalanobis support.
- **New** `phydrax/discretization/meshfree/_schwarz.py`: prepare meaningful stable-ID overlapping patches, restrictions, prolongations and partition of unity for `linalg/_subspace_correction.py` `SubspaceCorrectionTerm` and native additive/multiplicative builders. Bounded local solves and optional coarse correction are native linalg; no private Schwarz iteration.
- `solver/coupling/_meshfree_components.py`, `_acceptance.py`: use those accelerators for a meshfree-near-boundary/FEM-or-FV-interior overlap solve against the original coupled residual and work/amount balance. Explicit overlap/interface geometry supplies the transfer, not shape coincidence.

### Acceptance and tests

An adaptive nonpolynomial bulk boundary-layer and curved-surface PDE with fixed total work/accuracy comparisons; accepted and rejected point/support/degree transactions; no fabricated reliable bound. Learned multi-component parity/frame covariance and energy-gradient consistency; null-moment correction with positivity/coercivity conflict cases; failed primal/adjoint training rejection.

A hybrid meshfree–FEM/FV overlap solve converges to the independently checked original problem. Calibration uses disjoint complete cases, finite-sample quantile boundaries, and a separately labeled shifted-geometry test; no distribution-free guarantee under arbitrary drift.

Add **new** `test_adaptivity.py`, `test_corrections.py`, `test_schwarz.py`, `tests/integration/test_meshfree_adaptive_learning_workflow.py`, and `test_meshfree_hybrid_workflow.py`; extend constitutive/conservation/learned-training and existing UQ owner tests only where their public behavior changes. Add executable `examples/meshfree_adaptive_learning.py` and `meshfree_hybrid_calibrated.py` with independent analytic/physical error and explicit support/calibration status.

## P15. Durable runtime, restart, and resource/evidence retention

### Files and changes

- `phydrax/solver/_production_runtime.py`: bind meshfree temporal participants to bounded compiled segments, exact output schedule, observer state, accepted and rejected evidence, and checkpoint/output backpressure. Retain rolling mathematical history and native controller/iteration state; archive the requested trajectory through existing output owners.
- `phydrax/solver/_runtime_lifecycle.py`: encode cloud/geometry/topology/partition identities, stable point/lineage IDs, measure realization, support/stencil/metric/hierarchy revisions, boundary/interface/query sources, method/controller/precision, RNG addressing, live-history and evidence/archive cursor. Validate complete restored Strict values once before use.
- `phydrax/lifecycle/_composition_rebind.py`, `_transaction.py`, `_models.py`, `_distributed_checkpoint.py`, `_chunk_repository.py`: reuse staged rebuild/transport receipts, durable manifests, addressable-shard publication, and bounded staging. Same-topology and explicit ownership/epoch migrations remain distinct restart relations.
- **New** `phydrax/discretization/meshfree/_runtime.py`: only if needed, own the substantial meshfree-specific checkpoint inventory/rebind dependency contract; delegate encoding, persistence, trust, output and transactions to the native owners. It must not be a parallel checkpoint runtime.
- **New** `phydrax/_execution_sampling.py`: shared host/process and device-memory measurement with method, timestamp, baseline/peak scope, sampling uncertainty, unavailable fields, and provider identity. Integrate with `_execution_resources.py` and benchmark owners. CPU RSS, device allocator reservation, live buffer bytes and compiler estimates are different measurements.
- `phydrax/_execution_plan.py`, `_execution_resources.py`: hard resource ceilings come from declared capacity/work/storage admission plus trustworthy available memory/placement evidence. A sampled peak is not a universal upper bound. Unknown required evidence refuses a hard certified envelope rather than making up free memory.
- Qualification/runtime identities capture source/build/compiler/provider/device/topology/precision and validated resource/measurement policy. Never embed mutable timestamps or diagnostic NaNs as scientific identity.

### Acceptance and tests

Interrupt/restart a long moving/adaptive/coupled run after an accepted step and around a rejected epoch. Match uninterrupted accepted state, live histories, RNG, schedules, solver/controller and evidence cursors under the declared reproducibility class. Restart into a changed valid partition through an explicit migration receipt; stale source/program/geometry/capacity/precision and truncated or foreign checkpoint payloads refuse before consumer use.

Resource refusal covers candidates, local fit matrices, moments, constraints, rank, fill, Krylov workspaces, simplex populations, halos/migration packets, history, compilation and checkpoint/output staging. No storage bucket silently increases.

Add **new** `tests/integration/test_meshfree_runtime_restart.py` and focused meshfree checkpoint/resource tests. Extend generic runtime evidence/restart tests only for newly changed shared behavior. This phase requires actual CLI runs and observations of continuation/rollback, not only serialization unit tests.

## P16. Qualification, docs, manifests, and release dossier

### Qualification producers

- `tools/meshfree_qualification.py`: retain existing Q1–Q8 names, add finite selectable campaigns below, validate requested support/capacity/dimension/precision before execution, and distinguish passed/failed/unexecuted/blocked through the owning evidence vocabulary.
- Replace the current “observed rate > 0” convergence gate with declared per-method/geometry/derivative-order thresholds and at least four resolutions where claiming order. Derivative accuracy normally depends on polynomial degree minus derivative order, geometry order, and sampling; do not assume sphere superconvergence on irregular torus samples. Report measured spacing/fill estimates and their provenance, not nominal N alone.
- Add expected-refusal campaigns separately from accepted-performance campaigns. A refused large workload is useful resource evidence but does not pass a target that requires a successful large solve.
- `phydrax/discretization/meshfree/_profiles.py`: declare granular exact support tuples for new workflows, method choices, boundaries, geometry authorities, derivative classes, providers/devices, capacities, precision, temporal methods and reproducibility. Do not lump every method/provider into one opaque “bounded” string.
- `phydrax/qualification/_builtin_catalog.py`, `_catalog.py`, `_registry.py`, `_campaign.py`, `_criterion.py`, `_evidence.py`, `_promotion.py`, `_trust.py`: use existing typed evidence/profile/campaign/promotion machinery. Add a meshfree-specific criterion-to-artifact mapping only where needed; no campaign directly sets `released=True` or supplies independent sign-off.

### Finite campaign matrix

| Campaign | Required accepted/refused scenarios | Primary proof |
|---|---|---|
| Q1–Q2 revised | strong approximation, mixed/tensor/oversampled elliptic, dimensions 1–3 | independent derivatives/solutions; original residual; actual order |
| Q3 revised | scalar/block hierarchy, multiple nullspaces and cycle/smoother choices | reproduction, fine residual, setup/solve/refresh cost |
| Q4 revised | exact/relaxed signed/nonnegative and higher-moment metrics | original constraints, optimum, KKT, sign/feasibility |
| Q5 revised | circle/space curve/sphere/torus/open/sharp patch | independent geometry/quadrature and scalar/vector/tensor PDEs |
| Q6 revised | stage-correct motion, curvature, all live-history resampling | temporal/GCL/content error, rollback, continued run |
| Q7 revised | meshfree/FEM/FV exchange and monolithic/vector coupling | amount/force/work, original nonlinear residual, lag |
| Q8 revised | learned scalar/coupled laws, seen/unseen cases | implicit primal/adjoint, support/failed-state rejection |
| Q9 | bulk ADR/ALE/limited transport/nonlinear transient | spatial/temporal order, CFL/positivity, conservation |
| Q10 | physical incompressible and Lagrangian/particle interoperability | pressure/velocity error, divergence, modes, energy/mass |
| Q11 | elasticity, surface Stokes/tensor PDE and fluid–solid coupling | rigid modes, stress/traction/energy/work and mixed stability |
| Q12 | geometry-authorized higher forms; abstract clique as separate research | commuting/Stokes/d-squared-zero/Hodge/harmonic evidence |
| Q13 | joint transfer and actual physical split/merge | feasibility/obstruction, lineage, all-history transaction |
| Q14 | fixed-support/strict-active/generalized/frozen-remap sensitivity | finite differences, duality, event refusal and contract identity |
| Q15 | adaptive learned/calibrated/hybrid workflows | original error/work, rollback, invariance, held-out case diagnostics |
| Q16 | actual CPU/GPU/sharded/multi-host workloads | completeness/duality/migration, precision and measured resource |
| Q17 | long-run interrupt/restart/changed ownership | replay class, history/RNG/evidence, source/trust refusals |
| Dossier | independent physical references, rights, operations and release | current trusted evidence and independent decision |

These are new campaign selections, not schema generations. No Q record gains release status because another Q record passed.

### Capacity and hardware campaign contract

- Baseline regression: repeat existing sizes 12/24 learned, 25/49 conservative, 64/128 bulk/multilevel, 48/64 motion, 256/512 surface with identical original equations or disclose deliberate changes.
- Mandatory accepted Q8 expansion: 256/512, not merely an expected refusal at those sizes.
- Strong derivative/search/action scaling: 1,024, 4,096, 16,384, 65,536, 262,144 and 1,048,576 points, varying controlling candidate/support/chunk capacity as well as N.
- Bulk elliptic/transient/block and meshfree metric scaling: 256, 1,024, 4,096 and 16,384 points with explicit dimension-specific k, degree, radius, and nnz. Add larger accepted rows only when their resource admission and hardware permit; preserve declared refusal rows outside the envelope.
- Intrinsic surface and moving/epoch: 256, 512, 1,024, 2,048 and 4,096; resolve spatial, temporal, chart and quadrature errors independently. Physical event/higher-complex rows additionally vary edge/face/cell capacities; N alone is not their controlling complexity.
- Hardware rows: CPU float64, CPU float32 where scientifically supported, actual GPU float32/float64, mixed precision, multi-device single host, actual multi-host CPU/GPU. Unsupported provider dtype is an explicit support refusal. Forced CPU devices cannot substitute for GPU/multi-host performance.
- Budgets are supplied and recorded before execution, not inflated inside a failed run. Hardware insufficient for a mandatory accepted row is a qualification blocker. It must not disappear from the requested campaign or be recategorized as a pass.

### Benchmark files

Extend `benchmarks/meshfree_scaling.py`, `meshfree_multilevel.py`, `meshfree_exterior.py`, and `meshfree_moving_surface.py` for changed paths and compatible baseline handling. Add **new** `benchmarks/meshfree_closure.py` to select new physical/distributed/adjoint/adaptive/restart workloads using the same evidence utilities rather than copying four result writers.

Record preparation phases (search, geometry, local fit, rank/certificate, conic, ordering/fill, hierarchy, transfer), lowering, compilation, first/warm actions, solve, numeric refresh, JVP/VJP, full epoch commit, communication/migration, restart and output. Record argument/output/temporary/code compiler bytes, retained arrays including callable captures, host and device measurements with uncertainty, halo/checkpoint/output staging and logical allocation bounds. Include actual geometry/reproduction/constraint/error/status evidence beside timing.

Only compare before/after when the supplied baseline has the same mathematical workload and compatible source/environment/provider/precision support. Otherwise report unavailable comparison. Fit observed slopes over at least three compatible sizes; no inferred complexity or improvement from two warmups.

### Public documentation and examples

Update in each implementing phase, not only at the end:

- `docs/guides_meshfree.md`, `docs/api/discretization/meshfree.md`.
- `docs/guides_discretization.md`, `docs/guides_solver_substrates.md`.
- `docs/guides_numerical_interoperability.md`, `docs/guides_partitioned_coupling.md`, `docs/api/solver/coupling.md`.
- Affected linalg/execution/lifecycle/metrix/exterior/UQ/particle API pages.
- `docs/api/discretization/finite_difference.md` for projection cutover.
- `docs/examples/index.md`, `mkdocs.yml`, and `CHANGELOG.md` after actual paths and smoke proof exist.
- Existing examples: `point_cloud_poisson.py`, `meshfree_conservative_diffusion.py`, `meshfree_multilevel_poisson.py`, `meshfree_surface_laplace_beltrami.py`, `meshfree_moving_surface_reaction_diffusion.py`, `meshfree_bulk_surface_exchange.py`, `meshfree_learned_edge_flux.py`. Do not leave a second policy vocabulary in a familiar example.
- Facades: `phydrax/discretization/meshfree/__init__.py`, `phydrax/discretization/__init__.py`, relevant spatial/linalg/solver/coupling facades. Explicit lazy exports and runtime-resolvable contract annotations preserve import boundaries.

Regenerate, never hand-edit, public API/capability/closure/portfolio/source inventories from actual owner declarations. Candidate/research/released dispositions follow the existing lifecycle; the global policy itself does not need rewriting to accommodate meshfree.

## 4. Test selection, smoke proof, and static verification

### Selection rules

- Select the minimal conservative union from changes since the last merge into `dev`, not merely the last edited file. Include native-owner and consumer regressions whose contracts actually changed. Do not run realistically unaffected surfaces.
- Integration owner runs selected tests with `-n auto`; provider/device-topology cases use a separate dedicated process. Do not parallelize an initialized collective/provider case in a way that invalidates its evidence.
- A full suite is required only if the implementation changes global collection/configuration, shared fixtures or suite architecture. Broad implementation scope alone does not justify every unrelated application test.
- New permanent tests cover plausible consumer-visible numerical failures, boundaries, transitions, resource refusal, rollback, precedence and identity semantics. No source/import/copy/wiring/mock-echo tests. Remove existing incidental wording or implementation-pinning tests when encountered; do not re-pin them.
- Keep each independent failure scenario collectable; parameterize genuine method/geometry/precision matrices with diagnostic IDs. Deterministic bounded domains, independent NumPy/analytic/FEM references, and only the needed provider skips.
- Owning dtype/rank/warning tests use `strict_jax`; unrelated interoperability retains its native policy. Deliberate static misuse pairs exact line-local `ty` diagnostics with runtime refusal when reachable.

### Phase verification packages

Each phase's acceptance section specifies its runtime smoke and permanent test additions. Native owner changes also run the existing affected linalg, optim, geometry, nonlinear, solver, execution, lifecycle and UQ contract tests. In particular:

- P0 changes generic fixed-step/coupling evidence, so include retry, production/partitioned integration, not just the meshfree example.
- P2 changes minimum-norm/LSMR and sparse ordering, so include original native rectangular, rank-deficient/inconsistent, implicit derivative and factor-resource cases.
- P9 changes realization interfaces, so include native cochain and exterior consumers, not only a meshfree higher-form test.
- P10 projection cutover includes the two FD projection tests and their physical residual expectations.
- P12 shared spatial/particle transport includes existing distributed-plane/particle/atomistic consumers reached by reference analysis.
- P13 conic/nonlinear sensitivity changes include failed-lane batching/JVP/VJP and fixed-active/generic policy cases.
- P15 generic runtime changes include output/retry/evidence retention and restart/backpressure consumers.

Permanent tests do not replace smoke runs. Run every new executable workflow, observe its original physical error/status/ledger, exercise at least one refusal, and continue after rollback. Performance changes run their relevant phase-separated benchmark once after integration. UI is not part of this plan; CLI/runtime output is the actual changed surface.

### Static and generated contracts

Commands below are planned verification, not executed evidence:

```sh
python -m tools.generate_public_api_manifest --output docs/data/public_api.json
python -m tools.generate_capability_inventory
python -m tools.check_public_api_manifest --root . --manifest docs/data/public_api.json
python -m tools.check_capability_consistency --root .
python -m tools.check_import_boundaries --root .
python tools/check_typing.py check
python -m tools.audit_selectors
python -m tools.audit_contract_candidates --limit 100
python -m tools.check_installed_typing
```

Use the configured Ruff formatter/linter on changed files and pinned ty/annotation audit for first-party code. Installed typing must respect the actual package `requires-python`/wheel support; do not claim an installation on unsupported Python solely because an older instruction mentions it. Measure McCabe/cognitive complexity, NLOC and parameter counts at materially changed symbols before/after; do not hide severe functions behind repository averages. All named helpers have truthful parameter/return annotations and avoid compatibility casts/suppressions.

Generator defaults have been inspected: capability output includes `capabilities.json`, `application_portfolios.json`, `capability_closure.json`, `source_absorption.json`, and their generated Markdown views. Do not forget generated side outputs.

## 5. Cutover, qualification blockers, and final delivery

### Cutover completeness

For every changed public symbol/policy, retain a migration ledger containing old owner/API, canonical replacement, LSP reference result, migrated package consumers, examples/tools/benchmarks, tests and docs, and generated data. Final acceptance verifies behavior, not that the ledger matches source text. Remove obsolete Schur preparation, dense compatible projection storage, old ignored policy knobs, and superseded standalone helpers after the new path is exercised. Do not remove unrelated user changes.

Runtime cache and restore invalidation is explicit for changed structure, solver/resource policy, support, degree, geometry authority, partition, temporal method, law metadata and scientific measure. No same-shaped or same-display-name shortcut admits a stale object. Preserve deterministic stable ordering and RNG addressing.

### External prerequisites that code cannot invent

- Actual GPU/multi-host device inventory, initialized collective capability, and trustworthy measurement support for those mandatory qualification rows.
- Independent physical/reference cases with source identity, tolerances, provenance and usable rights. Manufactured fields prove numerical behavior but not scientific validation.
- Approved resource envelopes and reproducibility criteria, not an automatically enlarged budget.
- Current independent gate/role/signature/trust and release-decision records. Use the generic qualification machinery, not a domain-specific cardiovascular release API merely because `docs/RELEASE.md` uses cardiovascular examples.
- Source-backed domain/surface authority where physical topology, open/sharp patches, global reach, or higher-form fidelity is claimed.

These prerequisites do not block implementing and exercising all reachable numerical/runtime paths. They block the exact claim that needs them, and must remain visible in the final dossier. An unresolved mathematical support case is a documented refusal or research profile, not a fake fallback.

### End-to-end exit checklist

1. Every C01–C20 item is mapped to an implemented public path, exercised scenario, documented contract and retained verification record; no phase silently deferred.
2. Existing Q1–Q8 paths still work or have an explicitly qualified clean numerical cutover; Q8 256/512 accepted-target results are retained or explicitly blocked, never presented as passed by refusal.
3. Bulk, curve/surface, vector/tensor, flow/mechanics, physical topology, higher-form, mixed-method, adaptive/learned/calibrated, distributed and long-run restart examples are executable with real numerical results and failure evidence.
4. Original equations, geometric/physical measures, boundary/interface identities, signs, conservation, stability/positivity/coercivity, and derivative regularity reach consumer results without inferred success.
5. No hidden dense runtime fallback, all-pairs radius witness, Python per-point numerical iteration, per-stage cold factor preparation, automatic policy relaxation, dropped history, or silently suppressed failure remains on changed paths. Each retained host/static/provider loop or bounded dense result has its owner rationale and measured cost.
6. Targeted tests, actual smoke workflows, phase-separated performance, typing/selector/import audits and generated contracts have exercised evidence; unavailable hardware/reference/release authority is reported precisely.
7. Qualification and release status are accurate at each exact support tuple. An implementation-complete or qualified candidate is not labeled production without independent trusted release.
8. Final delivery includes the implementation worktree/branch, capability/API changes, actual observed checks/artifacts, and any remaining external qualification blockers. No unperformed benchmark, speedup, rank proof, topology certificate, calibration guarantee or release is asserted.
