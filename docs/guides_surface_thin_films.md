# Surface thin films

`phydrax.interfacial_transport` evolves thin liquid films on triangulated
manifold surfaces: lubrication drainage with disjoining pressure, symmetric
two-interface surfactant transport, plug-flow (extensional) films with
Marangoni forcing, and transport on moving surfaces. Every route stores
extensive vertex content, exchanges it only through antisymmetric edge fluxes,
and reports status and evidence instead of repairing a failed candidate.

## Prepared film geometry

`prepare_film_surface(mesh)` builds a `PreparedFilmSurface` from a manifold
`TriangleMesh`:

- `FilmSurfaceTopology` (host, once): canonical edges `(low, high)`, the
  oriented vertex-to-edge `EdgeRelation` with coboundary signs `(-1, +1)`,
  incident faces, boundary vertices/edges, `topology_id` and `operator_id`.
- Geometry (pure JAX): `DDGOperators` (cotangent stiffness, P1 gradients),
  barycentric dual areas `vertex_area`, vertex normals, and
  `curvature_squared = kappa_1^2 + kappa_2^2` from the normal-cycle estimate
  `S_v = A_v^-1 sum_e theta_e (l_e/2) e e^T` (Cohen-Steiner & Morvan, SoCG 2003).
  The estimate is exactly zero on planar meshes, including boundary vertices.
- `FilmSurfaceEvidence`: minimum cotangent conductance, the number of negative
  conductances, minimum dual/face areas, and `conductance_admissible`
  (all conductances nonnegative up to roundoff, i.e. an M-matrix stiffness).
- `refresh(coordinates)` recomputes the numeric geometry for identical
  topology and increments the dynamic `geometry_revision`; it never rebuilds
  connectivity and can run inside compiled loops.

Edge fluxes are positive from `edges[:, 0]` to `edges[:, 1]`;
`edge_divergence` returns the net outflow per vertex and sums to zero.
`edge_area_flux(v)` is the exact P1 area flux through each barycentric dual
edge: inside a face the dual segment between corners `a`, `b` has conormal
`A_f (grad phi_b - grad phi_a)/3` and mean velocity
`5/12 (v_a + v_b) + 1/6 v_c`.

`TriangleTopology` stays manifold. Non-manifold foams use
`PreparedFilmSheetSlots`: one `PreparedFilmSurface` per region pair plus sparse
gather/scatter maps to the owning multiregion sheet slots.

## Lubrication drainage (B1)

State authority is the vertex liquid volume `V_i`; `h_i = V_i / A_i`.
Backward Euler solves the mixed content–pressure system

```text
A h - V^n + dt div F = 0,   F_ij = w_ij m_ij (Phi_i - Phi_j)
Phi = sigma_eff (K h / A - (kappa_1^2 + kappa_2^2) h) - Pi(h)
      - rho g.x  [- rho (g.n) h for supported films]
```

with the thickness unknown `ln h` (a positive-domain iteration). The
`FilmMobilityLaw` selects `h^3/(12 mu)` (`"immobile-free-film"`, symmetric
free film, `sigma_eff = sigma/2`), `h^3/(3 mu)` (`"one-sided-substrate"`,
`sigma_eff = sigma`) or `(h^3 + 3 b h^2)/(3 mu)` (`"navier-slip"`). The edge
mobility is the entropy mean `(h_j - h_i)/(G'(h_j) - G'(h_i))`, `G'' = 1/m`
(Zhornitskaya & Bertozzi, SIAM J. Numer. Anal. 37, 2000; Grün & Rumpf,
Numer. Math. 87, 2000). `film_capillary_pressure` exposes the capillary part.

Each plan prepares the native sparse Jacobian
(`compile_sparse_jacobian`) and its symbolic sparse-LU pattern once. A
standalone solve refreshes the numeric factor at its initial guess and uses
it as a frozen right preconditioner for FGMRES. A bounded nearby-state
recurrence may instead refresh one numeric factor before the compiled loop
and reuse it; the current exact Jacobian still supplies every FGMRES action,
so the factor affects convergence rather than the equation,
and failure remains explicit. Native `NewtonTrustRegion` is used because
Armijo line search stalls on large nonlinear steps. The inner solves use a
constant forcing term equal to the declared linear tolerance. Newton steps,
trust-region attempts and FGMRES iterations (90 per solve) are bounded. A
budget that runs out is reported as `SOLVE_FAILED` with the native nonlinear
status. The hard 6x6, 30 % perturbation at `dt = 0.5` is outside the recorded
qualification campaign; no successful hard-step claim is made. The
committed volume is recomputed in flux form from the converged solution, so
totals are conserved to roundoff.

Boundary policies are `"no-flux"` or `"fixed-thickness"` (a reservoir such as
a Plateau border). `SurfaceLubricationStepResult.boundary_exchange_m3` is the
per-vertex exchange, positive into the film and zero off the boundary; its sum
is `SurfaceFilmEvidence.boundary_exchange_m3`.

`SurfaceFilmEvidence.positivity_guaranteed` holds only when conductances are
admissible, the positive-domain Newton solve converged, the result is finite
and the committed volume is positive. Otherwise the step reports the first
failed premise as a `FilmStepStatus` and keeps the previous state; there is no
clipping. `dissipation_guaranteed` additionally requires a convex discrete
energy (no curvature term, non-increasing `Pi`, stabilizing normal gravity)
and no-flux boundaries; the observed `energy_change_j` and `entropy_change`
are always reported. `rupture_mask` flags thickness below
`rupture_thickness_m`; lubrication never mutates topology.

### Disjoining pressure

`Pi > 0` is repulsive; `W(h) = int_h^inf Pi` with `W' = -Pi`.

| Law | `Pi(h)` |
|---|---|
| `VanDerWaalsDisjoiningPressure(A)` | `-A / (6 pi h^3)` |
| `DoubleLayerDisjoiningPressure(c, psi, T, eps_r, valence=z)` | `64 n k T tanh^2(z e psi / 4kT) exp(-kappa h)` |
| `ShortRangeRepulsionPressure(P0, l, exponent=n)` | `P0 (l/h)^n` |
| `CompositeDisjoiningPressure(laws)` | sum |

The double layer is the weak-overlap result (Israelachvili, *Intermolecular
and Surface Forces*, 3rd ed.). `black_film_equilibrium(law, P_c, bracket)`
solves `Pi(h) = P_c` in `ln h` with native Brent. The bracket selects the
branch (common or Newton black film); `stable` reports `dPi/dh < 0`; a bracket
without a sign change reports `BlackFilmStatus.NO_ROOT_IN_BRACKET`.

## Symmetric two-interface surfactant (B2)

`LangmuirSurfactantLaw(sigma_0, T, Gamma_inf)`, `AdsorptionKinetics(k_a, k_d,
Gamma_inf)` and `CoxVoinovWettingLaw` are strict modules whose physical
coefficients are `parameter_field` leaves. Each law describes one interface:
`sigma = sigma_0 + R T Gamma_inf ln(1 - Gamma/Gamma_inf)` and
`gibbs_elasticity = -Gamma dsigma/dGamma = R T Gamma_inf Gamma/(Gamma_inf - Gamma)`.
The checked methods refuse states at or beyond capacity; solvers use
`evaluate`, which returns an admissibility mask.

Both leaflets of a symmetric film share `Gamma = N/A` (`N` per interface). The
film carries `2N`, total tension `2 sigma`, Marangoni force `2 grad sigma`
and exchange `2 A j`. `SymmetricFilmSurfactantPlan` applies optional explicit
donor-cell advection by a tangential velocity (Courant number evidence;
`COURANT_LIMIT` above one), then an implicit diffusion/adsorption step through
`CoupledBulkSurfaceTransport` with surface area `2A`, conductance `2 D_s w`
and bulk volume equal to the liquid volume.

`CoupledBulkSurfaceTransport` is the sparse bulk–surface owner: symmetric edge
conductances, a sparse surface-to-bulk partition of unity, Langmuir kinetics
(or `None` for insoluble surfactant), and backward Euler through the same
prepared Newton route. Committed amounts are recomputed in flux form; a
candidate with negative amounts or full coverage is rejected. Nonfinite liquid
volume is reported as `NONFINITE` before the positivity check, including on
the insoluble route, and the state is left unchanged.

## Plug-flow films and Marangoni waves (B3)

```text
rho h Du/Dt = 2 grad sigma + div(N_T + 2 tau_BS) + rho h P g - C_air (u - u_air)
N_T = 2 mu h (D + (div u) P)            (Trouton sheet stress; Howell 1996)
tau_BS = 2 mu_s D + (kappa_s - mu_s)(div u) P   (rheology.boussinesq_scriven_stress)
```

The linear Marangoni wave speed is `c_M^2 = 2 E_s / (rho h)` (Chomaz, J.
Fluid Mech. 442, 2001), with the single-interface `E_s`. `SurfacePlugFlowPlan`
advances extensive volume, surfactant, optional dissolved amount and
tangential momentum with `solver.ConservationIMEXMethod`: a first-order
forward–backward Euler tableau whose implicit part is split into two
block-sequential parts (`AdditiveIMEXTableau(..., explicit_weights=...,
implicit_parts=(None, 0, 1))`):

1. explicit stage: conservative transport with `u^n`, selected by
   `transport_scheme` (`FilmTransportScheme`): `"donor-cell"` (default,
   positivity for outflow Courant numbers up to one) or `"limited-muscl"`
   (edge-based MUSCL: the upwind-extended difference from the lumped P1
   vertex gradient, minmod-limited against the central difference; admitted
   up to Courant one half, positivity not guaranteed). The limited scheme
   removes the donor-cell numerical viscosity `|u| dx / 2`, which otherwise
   dominates resolved vortical film flow (a Gaussian bump advected 60 steps
   at Courant 0.9 on a 60 x 30 grid keeps 47 % of its peak against 19 %);
2. implicit elastic/viscous part in `(u', Gamma')` through the prepared
   native Newton solve; the surfactant compression is corrected in flux form,
   so the Marangoni wave has no explicit stability limit, and momentum is
   committed as `P* + dt F(u', Gamma')`;
3. implicit diffusion/adsorption part through the symmetric-film surfactant
   route, started from the stage-2 value.

Each implicit stage commits a conservative update evaluated at its nonlinear
iterate, so conservation does not depend on the solve tolerance. The tableau
is stiffly accurate and stage 3 reuses the stage-2 value, so constrained
momentum components (walls at rest) are committed exactly. Input momentum must
be tangent to the film within a dtype-scaled residual tolerance; a finite
normal component rejects the step without projecting or committing it.
Per-stage solver evidence remains in `ConservationIMEXResult`.
`SurfaceFilmEvidence` reports the terminal stage's nonlinear status,
iterations, residual and convergence, while
`PlugFlowEvidence.terminal_nonlinear_stage` identifies that stage as
`PlugFlowNonlinearStage.ELASTIC` or `.SURFACTANT`.

The vertex Marangoni force is `2 sum_f (A_f/3) grad_f sigma`; the viscous force
is the exact adjoint of the face strain rate, so viscous dissipation
`sum_f A_f N:D` is nonnegative. `PlugFlowEvidence` reports momentum change
versus external impulse (equal to roundoff on planar films), the tangential
input residual and tolerance, surfactant residual, viscous and drag
dissipation, kinetic and surfactant free-energy changes, and the transport and
Marangoni Courant numbers.

### Fixed-sphere gravity equilibrium

Huang et al. (*Chemomechanical simulation of soap film flow on spherical
bubbles*, ACM TOG 39(4), 2020, eqs. 33–36) balance gravity against the
Marangoni stress of a bubble at rest, `(M/eta) dGamma/dtheta = g sin theta`.
Their model has no surface diffusion, no viscosity at rest and a linear
tension law. With `Gamma/eta` materially constant from a uniform start, the
balance gives `eta ~ exp(-(g/M) cos theta)`. In dimensional form the exponent
is `a = rho g R h_0 / (2 R T Gamma_0)` (`h_0` the total thickness,
`Gamma_0` per interface). Their eq. 36 as printed normalizes over the polar
angle (`int_0^pi dtheta`), which does not conserve the liquid volume: at
`a = 1` it loses 7 %. The volume-conserving constant is `a / sinh a`.

The plug-flow model shares the rest balance
`2 grad sigma(Gamma) + rho h g_t = 0`, the symmetric factor two, and
`Gamma/h` materially constant when `surface_diffusivity_m2_s = 0` and there
is no exchange. It uses the Langmuir tension, so its exact rest profile is
the logistic

```text
logit(Gamma / Gamma_inf) = C - rho g z / (2 R T Gamma_0 / h_0),   Gamma = (Gamma_0/h_0) h
```

with `C` fixed by the liquid volume. It reduces to Huang's exponential for
`Gamma << Gamma_inf`: at `Gamma_0/Gamma_inf = 1 %` the two profiles differ by
0.57 % (area-weighted relative L2). The permanent test and the qualification
campaign relax a uniform bubble (`R = 2 cm`, `h_0 = 1 um`, `a = 1`, linear air
drag `0.05 kg/(m^2 s)`, `dt = 5 ms`) and compare the thickness with this
profile. The campaign is bounded at 1 s and, after at least 0.4 s, stops only
after two consecutive 0.1 s windows change both thickness and concentration
by at most `1e-3` in area-weighted relative L2 and the peak speed by at most
`1e-3` of `rho h_0 g / C_air`. Two discretization effects remain, and both
shrink under refinement:

- A vertex-collocated rest state must satisfy two tangential force equations
  per vertex with one scalar unknown. The discrete force field is therefore
  not exactly a gradient, and a steady circulation remains, balanced by drag:
  3.7 % and 1.1 % of the drainage speed `rho h g / C_air` at 42 and 162
  vertices.
- The IMEX split transports volume with `u^n` explicitly and corrects the
  surfactant compression with `u'`, so `Gamma/h` drifts by `O(dt)`. On 42
  vertices the drift is 0.73 % at `dt = 5 ms` and 0.18 % at `1.25 ms`; it is
  0.45 % at `dt = 5 ms` on 162 vertices.

### Open and wall boundaries

`PlugFlowBoundary(surface, {kind: vertex_ids}, inflow_...)` declares one
`PlugFlowBoundaryKind` per boundary edge: `"no-flux"`, `"no-slip"`,
`"free-slip"`, `"inflow"` or `"outflow"`. An edge takes kind `k` when both
endpoints are in set `k`; every boundary edge needs exactly one kind, so
corner vertices may belong to two sets. Without a boundary every edge is
`no-flux` with a traction-free velocity.

- Open edges carry the exact P1 half-edge area flux
  `phi_a = (l nu / 2) . (3 u_a + u_b) / 4` with the outward conormal
  `l nu = -2 A_f grad phi_c`, which closes the discrete divergence theorem
  cell by cell. Transport is donor cell with the prescribed exterior
  density on inflow and the interior (zero-gradient) density on outflow and
  backflow; the implicit compression correction includes the same halves.
- Velocity constraints replace the constrained components of the implicit
  momentum rows by `Q (u - u_D)`: the tangent projector on held vertices
  (`no-slip` at rest, `inflow` at the prescribed velocity; precedence
  no-slip, inflow, free-slip) and `nu nu^T` with the averaged wall conormal
  on free-slip vertices. The commit and `initial_state` apply the same
  projection.
- `PlugFlowEvidence.boundary_impulse_n_s` collects open-edge momentum
  transport, the constraint impulse and the net boundary tension, so the
  momentum change equals the external plus boundary impulse on planar films.
  `PlugFlowEvidence.boundary` (`PlugFlowBoundaryEvidence`) holds per-vertex
  volume, surfactant and momentum outflow and the constraint force, which
  includes the line tension `oint phi_i 2 sigma nu ds` omitted by the
  strong-form Marangoni force; the force of the film on a wall is its
  negative sum over the wall vertices. `SurfaceFilmEvidence.boundary_exchange_m3`
  is the net inflow.
- A uniform film at the gravity/drag terminal velocity stays uniform to
  roundoff between inflow and outflow edges with free-slip walls, and a film
  at rest in a no-slip frame pulls each side inward with `2 sigma` per length
  (`tests/unit/interfacial_transport/test_surface_plug_flow_boundary.py`).
  The soap-film tunnel application composes these boundaries.

## Moving surfaces (B4)

Three motions are distinct. The *mesh motion* moves every vertex along the
straight path `x(t) = x^n + t w`, `w = (x^{n+1} - x^n)/dt`, and defines the
discrete surface and its dual measures at every instant. The *material
motion* `u` moves the film. *Surface deformation* is the change of the
discrete surface itself: a tangential mesh motion that keeps the discrete
surface fixed deforms nothing, while normal or non-rigid motion stretches it.

`SurfaceMeshMotion(source, target_coordinates, dt)` holds both geometries,
`w` with its split `w = (w.n) n + w_t` at midpoint vertex normals, and the
exact dual-area rate `area_rate_m2_s`. The barycentric measure
`A_i = sum_f |f|/3` obeys the lumped ESFEM transport property
`dA_i/dt = sum_f |f| div_f(w)/3`. Along straight paths the doubled face-area
vector `N_f(t) = N_0 + t N_1 + t^2 N_2` is quadratic, so the rate integrates in
closed form to `(|N_f(dt)| - |N_f(0)|)/2`. It is evaluated from the source
geometry and `w`, not from the target. The discrete geometric conservation
law `A^{n+1} - A^n = dt * area_rate` therefore holds to roundoff for every
motion: non-homothetic, non-planar and non-rigid alike. A midpoint-rule rate
(`dt * sum_f |f| div_f(w)/3` at the midpoint geometry) is exact only when
every triangle stays in its plane, and leaves a 4e-7 relative residual for a
5 % non-homothetic wobble of a level-3 icosphere.

`SurfaceMotionEvidence.status` is a `SurfaceMotionStatus`:
`INADMISSIBLE_INPUT` for a non-finite target or step, `ORIENTATION_REVERSED`,
`INADMISSIBLE_MEASURES`, or `GCL_VIOLATED` when the per-vertex residual
relative to the magnitude of its face-area terms exceeds `64 eps`. The last
one catches, for example, coordinates translated by 1e9 m, where
cancellation destroys the target measures. A refused motion must not be
committed.

`transport(content, u)` moves extensive content with the relative velocity:
the in-face part of `u - w` crosses the barycentric dual edges at the
midpoint geometry in antisymmetric donor-cell fluxes, so totals are conserved
to roundoff. Each barycentric subcell is a material region of its face's
affine mesh motion, so a Lagrangian film (`u = w`) keeps every cell's content
while its measure changes by exactly `dt * area_rate`. `material_velocity(u_t)`
builds `u = u_t + (w.n) n` for material that follows the surface normally.
`SurfaceTransportResult` carries the candidate, the outflow Courant number
and a `SurfaceMotionStatus`: the motion status, then `COURANT_LIMIT` above
one and `NONFINITE`. `content` is the candidate only when the status is
`ACCEPTED`; otherwise it is the input content, and the caller stays on the
source geometry. Refreshed lubrication geometry is rebound with
`PreparedSurfaceLubrication.with_surface` without recompiling.

A uniform areal density stays uniform only under a uniform material
dilatation. With zero material velocity, a tangential mesh motion of a fixed
surface keeps it uniform to roundoff: vertices slide along the straight grid
lines of the ruled surface `z = 0.1 sin(2 pi x)`, so the mesh deforms
non-homothetically while the discrete surface stays fixed. Arbitrary surface
deformation changes the density by the physical dilatation
`A^n / (A^n + dt * area_rate)` of a Lagrangian cell, and that exact
finite-step relation is what the tests check on a sphere and a wavy plane.
Fixed-topology motion is differentiable. `SurfaceEpochTransfer` moves
extensive content across a remesh with sparse nonnegative routes that split
each source cell completely, `TopologyEpoch` lineage, and
`derivative_available = False`.

## Qualification and nonclaims

`tools/surface_thin_film_qualification.py` runs the complete refinement
campaign and records environment, sequences, errors, statuses, and resources.
The completed matrix and current relative errors are:

| Campaign | Resolution sequence | Relative-error sequence | Observed orders |
| --- | --- | --- | --- |
| Planar levelling | 10x10 / 20x20 / 40x40 | 6.195 % / 1.611 % / 0.4091 % | 2.083 / 2.048 |
| Spherical-harmonic decay | icosphere levels 1 / 2 / 3 / 4 | 17.343 % / 4.555 % / 1.164 % / 0.2972 % | 1.981 / 1.982 / 1.973 |
| Marangoni quarter period | 99 / 195 / 387 vertices | 0.6210 % / 0.1618 % / 0.04672 % | 3.967 / 3.625 |
| Fixed-sphere gravity equilibrium | icosphere levels 1 / 2 / 3 | 5.191 % / 1.362 % / 0.4217 % | 1.983 / 1.703 |

Every gravity row reached the declared steady-state gate with accepted
statuses; the level-3 row used 120 steps (0.6 s simulated time), with a
2.13e-5 final-window velocity change relative to drainage speed. Its volume
and surfactant residuals were 3.31e-16 and 1.31e-16 relative, respectively.

The geometric-conservation campaign checks a tangential slide on a fixed ruled
surface, spherical scaling, and arbitrary non-homothetic motion of a sphere
and a wavy plane. All reach a relative GCL residual of at most 1.95e-16
against a 1.42e-14 tolerance. The midpoint-rule rate misses by 4.41e-7
(wobble) and 3.57e-5 (arbitrary motion). Gauss–Legendre integrals of the
instantaneous lumped rate converge to the closed form: 3.57e-5, 8.08e-9,
and 1.12e-15 with 1, 2, and 4 nodes. A 1e9 m translation is refused with
`GCL_VIOLATED`.

- Sparse-LU resources: the symbolic plan stores the factor pattern and its
  column index, not one record per elimination update. The 40x40 planar row
  has 3,362 unknowns, 380,644 factor nonzeros, and a 16,488,328-byte factor
  plan. Icosphere level 4 has 5,124 unknowns, 1,392,360 factor nonzeros, and a
  56,862,676-byte factor plan. Gravity level 3 has 1,926 unknowns, 398,898
  factor nonzeros, a 16,907,896-byte factor plan, and 20,268,714 logical
  prepared bytes. These are the resources recorded by the completed campaign,
  not evidence that the hard `dt = 0.5` perturbation succeeds.

- Positivity is claimed only under the listed premises; failures are
  reported, never clipped.
- A uniform film is exactly invariant on planar meshes. On curved meshes the
  per-vertex normal-cycle curvature varies around irregular vertices, so a
  uniform film drifts at the discretization level.
- No rupture topology change, no marginal-regeneration claim and no
  self-similar rupture exponent are claimed.
- Plug flow: `dissipation_guaranteed` is false for the IMEX splitting;
  total momentum is conserved only on planar films; collocated vertex
  velocities can carry grid-scale modes that viscosity damps. The
  fixed-sphere gravity equilibrium is qualified against the model's own
  Langmuir rest profile (Huang et al.'s exponential in the dilute limit), up
  to the discretization-level circulation and `O(dt)` `Gamma/h` drift above.
- Two symmetric leaflets only; asymmetric leaflets are not represented.
- The IMEX step is first order; the block-sequential implicit parts add a
  first-order splitting error between the elastic and exchange blocks.
- Moving surfaces: the discrete GCL is exact to roundoff for every motion.
  Uniform density is preserved only by tangential mesh motion of a fixed
  discrete surface; surface deformation changes it by the physical
  dilatation. The relative flux of `u - w` is evaluated once at the midpoint
  geometry (first order in time for `u != w`). Remesh routes are supplied by
  the caller.
