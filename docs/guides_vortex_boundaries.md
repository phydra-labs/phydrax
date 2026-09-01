# Vortex panels, walls, and moving bodies

Boundary methods in Phydrax separate three different models:

1. source/vortex panels impose inviscid normal-flow conditions;
2. boundary-sheet transfer creates resolved near-wall vortex carriers through an
   explicit accepted transaction;
3. remeshing redistributes existing integrated strength while checking conservation.

They share prepared geometry and evidence, but one does not silently stand in for
another. In particular, a no-penetration panel solve is not a no-slip viscous wall
model.

The executable `examples/vortex_panel_cylinder.py` applies the public panel
surface to a stationary cylinder and then realizes the same reference topology
under a rigid motion.

## Geometry and sign convention

The 2-D panel operator consumes an explicitly closed, consistently oriented
polygonal contour. Reference panels are immutable; a runtime realization maps
their points, tangents, and normals to world coordinates. Counter-clockwise outer
contours receive outward right normals.

The flow-specific kernel evaluates source and vortex sheet velocity, not the scalar
Laplace value returned by the general boundary-layer-potential API. Its influence
assembly retains both normal and tangential traces. Singular self traces and their
sign convention are explicit; callers must not repair diagonal entries after
preparation.

A result identity binds the reference panelization, world realization, trace policy,
targets, and strengths. This prevents stationary-body influence evidence from being
reported for a different moving-body state.

## No-penetration and circulation constraints

At collocation point \(x_i\), an inviscid body solve enforces

\[
 n_i \mathbin{\cdot}
 \left(u_\infty(x_i) + u_{\mathrm{body\ sheet}}(x_i)
       - u_{\mathrm{body}}(x_i)\right) = 0.
\]

The linear system may include an explicit total-circulation constraint and declared
Kutta metadata. A Kutta condition is never inferred from two array endpoints. The solve
is routed through Phydrax linear algebra and retains its rank/status evidence,
constraint residual, normal residual, and circulation defect.

The stationary circular-cylinder case is a useful inviscid qualification because it
has an analytic tangential velocity and pressure coefficient. Agreement there
qualifies the stated panel convention and resolution; it does not validate viscous
separation or drag.

## Moving-body realization

A rigid realization supplies translation, rotation, linear velocity, and angular
velocity. Numeric panel coordinates and surface velocity change while reference panel
connectivity and source identity remain fixed. In two dimensions the local rigid
surface velocity is

\[
 u_b(x) = V + \Omega\,\hat{z}\times(x-x_c).
\]

The boundary condition uses relative normal flow. Simply moving collocation points
without the body surface velocity solves a different problem and is unsupported.

Pressure loads retain the density, incident velocity, and residual evidence used
to construct them. A finite force is not accepted when the boundary solve or
pressure reconstruction fails. Panel loads are inviscid loads; skin friction and
impulse-force equivalence are not inferred.

## Boundary-sheet transfer

A boundary-sheet-to-particle transition is an explicit state update. The plan declares
wall offsets, generated core size, target particle slots, candidate capacity, and
acceptance policy before execution. A candidate reports:

- circulation supplied by the sheet and circulation inserted into active slots;
- candidate and accepted counts;
- offset and wall-clearance status;
- finite core/position/strength status;
- capacity and atomic-acceptance status.

Insufficient capacity or an invalid wall location fails closed. The pre-transition
state remains the accepted state; partial insertion is not presented as a successful
wall treatment. This transfer supplies discrete vorticity near the wall but does not
claim a universal turbulent boundary-layer closure.

## Compact diffusion near a wall

`WallCorrectedPSEPlan` provides explicit mirror and one-sided support policies.
Its evidence separates conservative bulk exchange from induced or prescribed
wall flux. `BoundaryIntegralVorticityFluxPlan2D` solves normal and tangential
residuals before atomic sheet-to-particle transfer.

## Conservative remeshing

Fixed-grid remeshing maps a candidate set of active particles onto preallocated target
slots. The remeshing evidence distinguishes:

- total integrated-strength defect;
- first spatial-moment defect;
- assignment partition and support defect;
- target capacity and finite-state status;
- candidate versus accepted state identity.

The update is atomic. When a requested conservation condition or capacity is not met,
the candidate may be inspected but cannot replace the accepted state. Remeshing changes
the discrete carrier layout, so it is outside a smooth ODE right-hand side and outside
the derivative of a preceding fixed-topology step.

Conservation of circulation and first moment is necessary but not sufficient for field
accuracy. Core size, grid spacing, transfer order, boundary clearance, and higher
moments remain method choices.

## Coupling boundaries and wakes

Panel bodies, free vortices, and lifting wakes may be composed only through explicit
field and state-transition contracts:

- panel-bound circulation and free-vortex circulation retain separate ownership;
- body motion updates world geometry and surface velocity before assembly;
- a wake or wall transfer commits only after the enclosing step is accepted;
- velocity evaluations identify their complete source collection and self mappings;
- loads and pose exchange are explicit in coupled rigid-body workflows.

No solver automatically creates separation points, changes wake connectivity, or moves
vorticity across a body. Such a topology or modeling decision must be supplied by a
prepared policy with acceptance evidence.

## Completed boundary capability and irreducible limits

Native `BoundaryPanelization2D` and `SurfacePanelization3D` adapters provide
source, vortex, doublet, combined, near-corrected, moving/deforming, pressure,
impulse, Blasius, and added-mass contracts. Boundary-integral no-slip flux and
the MAC immersed hybrid are separate qualified wall routes. Complete remeshing
supports 2-D/3-D, degree one through three, obstacle clearance, and epochal
capacity transitions.

Phydrax still does not claim a universal near-wall turbulence law or exact force
recovery from an under-resolved sheet. Reduced separation criteria are named,
parameterized models; resolved viscous separation comes from the wall solve.

## Primary method references

- Hess and Smith, "Calculation of potential flow about arbitrary bodies," *Progress in
  Aeronautical Sciences* 8 (1967), pp. 1--138.
- Morton, "The generation and decay of vorticity," *Geophysical & Astrophysical Fluid
  Dynamics* 28 (1984),
  [doi:10.1080/03091928408230368](https://doi.org/10.1080/03091928408230368).
- Koumoutsakos, Leonard, and Pepin, "Boundary conditions for viscous vortex methods,"
  *Journal of Computational Physics* 113 (1994),
  [doi:10.1006/jcph.1994.1117](https://doi.org/10.1006/jcph.1994.1117).
- Beale and Majda, "High order accurate vortex methods with explicit velocity
  kernels," *Journal of Computational Physics* 58 (1985),
  [doi:10.1016/0021-9991(85)90176-7](https://doi.org/10.1016/0021-9991(85)90176-7).
