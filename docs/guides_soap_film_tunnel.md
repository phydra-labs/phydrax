# Soap-film tunnel

`phydrax.applications.soap_film_tunnel` simulates the quasi-two-dimensional
hydrodynamics of a flowing soap film: the vertical, gravity-driven soap-film
tunnel in which a film falls between two wires, reaches a terminal speed at
which air friction balances its weight, and flows past an obstacle that
pierces it (Couder, Chomaz & Rabaud, Physica D 37, 1989; Gharib & Derango,
Physica D 37, 1989; Kellay, Wu & Goldburg, Phys. Rev. Lett. 74, 1995;
Rutgers, Wu & Daniel, Rev. Sci. Instrum. 72, 2001). The application composes
the symmetric plug-flow film route of
[surface thin films](guides_surface_thin_films.md) with inflow, outflow, wire
and obstacle-rim boundaries. It is not a 3D foam or a 3D Navier–Stokes model.

## Model

The film occupies the plane `z = 0`, `0 <= x <= L`, `0 <= y <= W`. The
tunnel axis `+x` points downstream (downward in a vertical tunnel). Film
thickness `h`, surface concentration `Gamma` on both interfaces and the
tangential velocity `u` obey the plug-flow equations of
`SurfacePlugFlowPlan` (Chomaz, J. Fluid Mech. 442, 2001):

```text
rho h Du/Dt = 2 grad sigma(Gamma) + div(N_T + 2 tau_BS) + rho h g e_x - C u
Dh/Dt = -h div u,   DGamma/Dt = -Gamma div u + D_s lap Gamma
```

- Compressibility: thickness and surfactant variations change the surface
  tension; the Marangoni elasticity `E = -Gamma dsigma/dGamma` of the
  Langmuir law gives the elastic wave speed `c_M = sqrt(2 E / (rho h))`. The
  film Mach number `u / c_M` plays the role of the acoustic Mach number of a
  2D compressible gas (Kim & Mandre, Phys. Rev. Fluids 2, 082001, 2017).
- Viscosity: the Trouton sheet stress `2 mu h (D + (div u) P)` and twice the
  Boussinesq–Scriven interfacial stress give the 2D kinematic viscosity
  `nu_2D = (mu h + 2 mu_s) / (rho h)`; the Reynolds number is
  `Re = U d / nu_2D` with the obstacle diameter `d`.
- Drive: gravity `g` along `+x` and linear air drag `C u` per film area. A
  uniform film has the terminal velocity `u_T = rho h g / C` and relaxes to
  it on the time `rho h / C`. Rutgers et al. measured a laminar
  boundary-layer air drag that grows like `u^{3/2}`; the linear coefficient
  is the declared closure (for example the secant `tau(U) / U` at the
  operating speed), not that law.

`SoapFilmTunnelPlan.terminal_velocity_m_s` and `SoapFilmTunnelScales`
(`prepared.scales`) report `u_T`, `rho h / C`, `c_M`, the inflow film Mach
number, `nu_2D`, `Re`, the blockage ratio `d / W`, and the cell Reynolds
number `U dx / nu_2D` at the rim spacing.

## Geometry and mesh

`SoapFilmTunnelGeometry(L, W, mesh_size_m=..., obstacle_diameter_m=...,
obstacle_center_m=..., rim_segments=...)` describes the channel and an
optional circular hole whose rim is a polygon of `rim_segments` chords.
`triangulate()` uses the native constrained Delaunay owner
(`phydrax.geometry.ConstrainedDelaunayTriangulation`, Ruppert/Chew refinement
to `minimum_angle_degrees`, largest area `sqrt(3)/4 mesh_size^2`; it requires
the optional `phydrax-meshcore` library, installed as `phydrax[meshcore]` or
named by `PHYDRAX_MESHCORE_LIBRARY`). The result, `SoapFilmTunnelMesh`, keeps
the triangulation evidence and the inlet, outlet, wire and rim vertex sets,
read from the preserved input-segment labels, not from coordinate
tolerances. Refinement leaves no boundary segment encroached and keeps free
edges Delaunay, so the cotangent conductances are nonnegative;
`PreparedSoapFilmTunnel` audits this through `PreparedFilmSurface` and
refuses an inadmissible mesh.

## Boundaries

`PreparedSoapFilmTunnel` declares a `PlugFlowBoundary` (see the thin-film
guide for the discrete half-edge fluxes and constraint rows):

| Part | Kind | Meaning |
|---|---|---|
| inlet `x = 0` | `inflow` | prescribed speed `U e_x`, thickness, `Gamma` (and dissolved concentration); donor-cell inflow of every content |
| outlet `x = L` | `outflow` | donor-cell outflow (zero gradient on backflow), traction-free film |
| wires `y = 0, W` | `no-slip` (default) or `free-slip` | closed to transport; the film sticks to (or slides along) the wires |
| obstacle rim | `no-slip` | closed; the film sticks to the cylinder |

Corner vertices take the precedence no-slip, inflow, free-slip.

## Stepping and evidence

```python
import jax.numpy as jnp

prepared = plan.prepare()
state = prepared.initial_state()            # uniform inflow film
result = prepared.run(state, jnp.asarray(step_size), steps)
estimate = prepared.strouhal(result, transient_s=...)
reference = prepared.cylinder_wake_reference(estimate.mean_drag_coefficient)
```

`run` is backed by one stable module-level `eqx.filter_jit` entry point
containing one `jax.lax.scan(..., unroll=1)`. It does not construct a JIT
wrapper at runtime or unroll Python steps into the trace. Every evidence sample
is public output, so checkpoint rematerialization would not reduce the declared
trajectory. `step` advances one plug-flow step; `run` scans a fixed number of
steps and stacks `SoapFilmTunnelEvidence`:

- inlet inflow and outlet outflow rates of liquid volume and surfactant
  (both interfaces plus dissolved);
- the film owner's volume and surfactant residuals after boundary exchange
  and the momentum residual (change minus external and boundary impulse),
  all roundoff for accepted planar steps;
- `obstacle_force_n` and `wire_force_n`: the step-averaged force the film
  exerts on the rim and wires, from the constraint reactions including the
  rim line tension `2 sigma` (a film pulling uniformly on a closed rim exerts
  no net force);
- `film_mach_number = max |u| / c_M` with the local `c_M` of the attempted
  candidate, the candidate maximum speed and minimum thickness;
- status, transport and Marangoni Courant numbers, and terminal nonlinear
  stage, status, iterations, residual and convergence.

A rejected step (for example `COURANT_LIMIT` or `SOLVE_FAILED`) keeps the
previous state and time. Its evidence is computed from `film.candidate_state`,
so candidate diagnostics preserve nonfinite attempted values rather than
substituting the rolled-back state.

`strouhal` estimates the shedding frequency from the lift after a transient:
the centered lift coefficient must exceed `minimum_lift_coefficient`,
periods are counted between upward zero crossings that follow a descent below
minus half the amplitude, and `St = f d / U`. The frequency and Strouhal
standard errors propagate the sample standard error of the observed cycle
periods; they do not include mesh, time-step or closure uncertainty.
`StrouhalEstimate.status` is `SHEDDING`, `NO_SHEDDING`,
`INSUFFICIENT_RECORD` (fewer than two periods) or `REJECTED_STEPS`; drag and
lift coefficients use `rho h U^2 d / 2`.

## Reference values

For 2D circular-cylinder wakes the unconfined Strouhal number rises from about
0.12 at the onset of shedding (`Re ≈ 47`) through 0.16–0.19 for
`Re = 100–200` and stays near 0.2 up to `Re ~ 10^5` (Roshko, NACA TN 2913,
1953; Williamson, Annu. Rev. Fluid Mech. 28, 1996). Soap-film tunnels
reproduce this range (Vorobieff & Ecke, Phys. Rev. E 60, 2953, 1999).

The example has non-negligible blockage `B = d/W = 1/6`.
`cylinder_wake_reference(C_D)` therefore reports both the unconfined
Williamson parallel-shedding fit and the Allen--Vincenti wall-interference
comparison

```text
U'/U = 1 + C_D B/4 + 0.82 B^2,
Re' = Re U'/U,
St_blockage = (U'/U) St_Williamson(Re').
```

The last expression converts the fitted corrected-speed Strouhal number back
to the tunnel's inflow-speed convention. The formula follows its use for
low-Re cylinder CFD by Decuyper et al., Mech. Syst. Signal Process. 98,
209–230 (2018), who trace it to Allen & Vincenti (NACA TR 782, 1944).
The NACA report derives a general two-dimensional wind-tunnel wall correction,
not a soap-film calibration; at `B = 1/6` this is an explicit comparison gate,
not calibrated truth or a combined uncertainty bound.

For scale, Masroor, Yang & Stremler (Data in Brief 41, 107819, 2022)
report a gravity-driven film experiment with `Re ≈ 235`, `d/W ≈ 0.064`,
`h ≈ 3 um` and elastic Mach about 0.2. The example is instead `Re = 150`,
`d/W = 0.167`, `h = 1.5 um` and inflow film Mach 0.337; it is a
model campaign, not a parameter-matched reproduction of that experiment.

## Numerical route and limits

- The film step is the plug-flow owner's IMEX step. The tunnel defaults to
  `transport_scheme="limited-muscl"`: donor-cell transport adds a numerical
  viscosity of about `|u| dx / 2`, which at affordable meshes exceeds the
  declared film viscosity many times over (cell Reynolds numbers of 10–30)
  and suppresses shedding. The limited scheme admits outflow Courant
  numbers up to one half and does not guarantee positivity; a violated
  premise rejects the step (`COURANT_LIMIT`, `POSITIVITY_VIOLATED`).
- Every accepted step solves the implicit elastic/viscous block with the
  prepared native Newton–FGMRES solve. Its symbolic sparse-LU pattern of the
  one-ring Jacobian is prepared once; each standalone solve refreshes only
  its numeric values. Factor-plan storage scales with factor nonzeros, and
  nonlinear and linear iteration limits bound every solve.
- The obstacle rim is a polygon; the Strouhal estimate depends on rim and
  wake resolution, blockage and the numerical dissipation of the wake, and
  is reported with its evidence rather than calibrated.

## Qualification campaign

`tools/soap_film_tunnel_qualification.py` runs a terminal-flow invariant and,
without `--smoke`, cylinder wakes at rim/mesh pairs `(24, 0.5 d)`,
`(32, 0.4 d)` and `(48, 0.3 d)`. Each wake retains six requested shedding
periods after an equal transient at `Re = 150`. The report keeps preparation,
lowering, executable compilation, first execution and repeated warm execution
separate, along with compiler memory, logical retained bytes, cycle-count
uncertainty, film Mach, flux/momentum ledgers and the blockage comparison.

The phase-separated runtime smoke on a 265-vertex empty channel (Apple M1 Max,
CPU backend, heavy shared-machine load) took 39.33 s for host preparation,
5.98 s for lowering, 6.19 s for executable compilation, 27.3 ms for first
execution and 22.7 ms warm. Compiler-estimated arguments, output and
temporaries totaled 7.24 MB. This is plumbing evidence that compilation is
bounded; the cylinder rows are the controlling-capacity evidence.


## Nonclaims

- Quasi-2D plug flow only: no 3D foam, no thickness-resolved (Poiseuille)
  film flow, no film rupture or hole growth, no air flow computation.
- Linear air drag only; the `u^{3/2}` laminar boundary-layer drag of
  Rutgers et al. is not represented.
- No derivative claims through preparation (host triangulation and
  validation) and no qualified Strouhal or drag values: the
  `soap-film-tunnel.planar-plug-flow` profile is a candidate until the
  cylinder-wake campaign (`tools/soap_film_tunnel_qualification.py`) passes.
- Film Mach numbers near or above one (Marangoni shocks, Kim & Mandre 2017)
  are reported but not qualified.
