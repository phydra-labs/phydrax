# Porous-media flow, heat, and chemistry

`phydrax.applications.porous_media` provides conservative fixed-geometry porous,
surface, fracture-network, phase-equilibrium, thermal, and reactive physics. It owns
physical constitutive laws and coupled residuals; generic nonlinear solves, implicit
calculus, sparse operators, and field transfers remain in their existing modules.

## Hybrid tensor diffusion

`HybridMimeticDiffusion` adds the missing unstructured scalar/tensor diffusion
operator. It retains one potential per cell and one shared global potential per face.
For each admissible star-shaped polyhedral or tetrahedral cell it combines:

- an affine-consistent reconstructed gradient;
- the complete symmetric positive-definite three-dimensional material tensor;
- a positive normal-distance stabilization of non-affine face traces;
- one owner-oriented integrated face rate.

The constructor verifies positive volumes, outward closure, star-shaped distances,
and the first geometric moment identity. `HybridDiffusionBoundary` supplies
Dirichlet potential, integrated outward Neumann rate, or Robin conductance and target.
Interior faces cannot carry boundary laws. Every unsteady or steady balance consumes
the same local face rates, so interior contributions cancel exactly.

This is not TPFA. No tensor or mesh is declared TPFA-admissible merely because its
cells came from a Voronoi mesher. Coercivity also does not imply monotonicity or a
discrete maximum principle.

## Richards flow

`RichardsPlan` uses pressure in pascals as its primary variable. Its Darcy volume rate
is

```text
q = -(k_r / mu) K (grad(p) - rho g)
```

and its conserved liquid mass is

```text
M = cell_volume × pore_fraction(p) × saturation(p) × density(p, T).
```

The backward-Euler root contains every cell pressure and shared face pressure. Cell
rows are mass rates; face rows enforce mass continuity or the declared physical
boundary law. `PorousBoundaryConditions` uses pressure in Pa, integrated outward mass
rate in kg/s, and leakage conductance in kg/(Pa s).

`PorousMaterial` supports scalar, cellwise scalar, or full SPD intrinsic permeability,
explicit fluid and pore compressibility, thermal expansion, and temperature-dependent
viscosity. `VanGenuchtenMualem` provides nonhysteretic capillary retention and liquid
relative permeability. Saturated and residual endpoints are exact: no hidden storage
or mobility floor is introduced. A closed saturated incompressible component without
pressure anchoring is reported as ill posed.

Successful roots receive native implicit JVP/VJP derivatives. Failed roots retain an
inspectable candidate but do not commit state or provide valid derivatives.

## Coupled water and sensible heat

`CoupledWaterHeatPlan` solves cell/face pressure and temperature in one root. It uses:

- fixed-skeleton and liquid sensible-energy storage;
- full-tensor dry/saturated effective conduction;
- the exact accepted liquid mass rate for advected enthalpy;
- explicit inflow temperature for every exterior liquid inflow.

Storage uses sensible internal energy; advective transport uses liquid enthalpy with a
common reference temperature. Temperature-dependent density or viscosity therefore
feeds back into the same pressure-temperature residual. The model does not imply
latent heat, vapor transport, freezing, kinetic energy, elastic work, or local thermal
nonequilibrium.

## Surface storage and runoff

`BoundarySurfaceTrace` binds explicit volume boundary faces to surface cells with
parent cell, area, center, normal, and edge adjacency. A positive exchange is outward
from the volume and inward to the surface; `volume_content_rate` applies its exact
opposite.

`OrthogonalDiffusiveWaveSurfacePlan` stores liquid volume and sensible energy on
upward-facing projected surface cells. Its lateral Manning rate is admitted only for
projected-orthogonal two-point geometry. Donor budgets limit shared rates before the
update; inventories are not clipped afterward.

`SurfaceRichardsPlan` couples pressure and surface depth through wet/dry
Fischer–Burmeister complementarity. `SurfaceWaterHeatPlan` additionally couples
pressure, temperature, surface water, and surface sensible energy in one root. Wet
cells enforce pressure/head and temperature continuity; dry cells retain the declared
subsurface thermal boundary. Derivatives are qualified only away from wet/dry
transitions.

## Conservative component transport

`ComponentTransport` accepts the already accepted owner-oriented water-volume face
rates. It advances conserved component inventories with upstream concentration and,
when supplied, hybrid SPD dispersion. Inward exterior water requires explicit upstream
component concentrations. Sources and boundary rates are in physical integrated units.

`MassActionSystem` solves local aqueous equilibrium in a declared primary/secondary
basis with ideal or bounded Davies activities. Formation constants use natural-log
form, charges are explicit, and an optional charge-balance replacement records which
component total is open. Only successful nonsingular roots have implicit derivatives.

`MineralKinetics` advances stoichiometric bounded extents. Exhaustion is enforced in
the solved extent rather than by clipping mineral inventory after reaction.
`FractureMatrixExchange` supplies aperture-resolved two-dimensional fracture storage,
explicit parent cell/global identities, hydraulic exchange, and equal-and-opposite
component sources. It is a fixed resolved mixed-dimensional relation, not an arbitrary
DFN generator.

## Mesh qualification

`qualify_vorocrust_porous_mesh` consumes an existing successful VoroCrust
`CellMeshingResult`, a compatible `GeospatialContract`, and optional explicit surface
faces. It prepares and verifies the numerical geometry used by hybrid diffusion. It
does not claim retained generators, Delaunay duals, material labels, TPFA consistency,
or differentiable meshing.

## Execution and qualification

Representative commands:

```text
python examples/geophysics/porous_infiltration.py
python benchmarks/porous_media.py --warmup 1 --repeats 5
```

The benchmark reports compilation, warm execution, nonlinear success, physical
residual, global mass/energy balance, and implicit tangent/adjoint agreement.

## Multiphase component and total-energy conservation

`MultiphaseConservationPlan` separates phase volume rates from the component-mass and
enthalpy rates they carry. A prepared state records saturation, phase density,
phase composition, phase internal energy, rock internal energy, porosity, component
inventory, energy inventory, and the exact plan identity. Saturations and phase
compositions must close to one; densities and porosity must remain physical.

`MultiphaseFaceFluxes` contains the accepted upstream phase properties and integrated
owner-oriented face rates. `residual` computes local and global component/energy
balances without reconstructing a hidden phase flux. States or fluxes from another
plan are rejected.

`RachfordRiceFlashPlan` handles single-liquid, two-phase, and single-vapor states
without fabricating an absent phase. `CompositionalFlashPlan` evaluates a caller-owned
identified K-value function before applying the same phase-state logic. Phase
appearance is a branch boundary; derivative availability records that fact.

## Hysteresis and dynamic capillarity

`BrooksCoreyRetention` supplies a bounded drainage or imbibition curve.
`HystereticRetentionPlan` records the active branch and reversal pressure/saturation.
Scanning curves that leave the physical saturation envelope by more than floating
roundoff fail instead of being silently clipped. A reversal invalidates the local
smooth derivative flag.

`DynamicCapillaryPressure` adds an explicit pressure-rate relaxation term. It is not
implicitly enabled by a Richards or multiphase plan.

## Freeze/thaw, vapor, and atmosphere

`FreezeThawMaterial` provides sensible plus latent enthalpy with an optional smooth
mushy interval. Zero-width phase change is intentionally nondifferentiable.
`VaporEquilibrium` supplies saturation pressure and equilibrium vapor density over its
positive-temperature domain.

`AtmosphericExchangePlan` reports evaporation/condensation, sensible heat, latent heat,
net radiation, and limiter activity. Positive evaporation is limited by available
water over the step. Negative evaporation is condensation and is not incorrectly
limited by current liquid storage.

These are separate constitutive and boundary pieces. `CoupledWaterHeatPlan` remains a
liquid sensible-heat model until a caller explicitly composes the additional phase and
vapor terms.

## General surface flow and wells

`UnstructuredShallowWaterPlan` implements hydrostatic reconstruction and a Rusanov
flux on an explicitly oriented two-dimensional surface mesh. Interior, wall, outflow,
and prescribed-inflow edges are distinct. Rainfall and infiltration use integrated
cell budgets; outgoing rates are limited before the update so water volume is not
clipped afterward. State identity prevents reuse on another surface plan.

`WellCompletionPlan` maps named completion cells to pressure-driven component rates.
`WellControl` selects rate or pressure operation, and rate control switches to a
pressure bound explicitly when required. The result reports the realized control,
completion rates, component rates, and success; there is no hidden well-index or
control fallback.

## High-ionic-strength and monolithic reaction transport

`SITActivityModel` and `PitzerInteractionModel` have explicit ionic-strength domains
and interaction matrices. `RedoxEquilibrium`, `HenryGasEquilibrium`, and
`IonExchangeEquilibrium` keep their activities, pressure, charge, and equivalent-pool
semantics separate.

`MonolithicReactiveTransportPlan` solves cell transport inventory, aqueous
mass-action equilibrium, and optional bounded mineral extents in one global root on
fixed accepted water rates. It supports the cell-only transport layout; a hybrid
face-trace chemistry layout is rejected rather than dropped. Successful output
reports species, component and mineral inventories, face component fluxes, the global
component balance, and the native nonlinear result.

`MixedDimensionalFractureNetworkPlan` provides explicit matrix-cell, fracture, and
intersection storage plus matrix/fracture and fracture/intersection exchange.
Every internal exchange is equal and opposite. It is a fixed network realization,
not an automatic discrete-fracture-network generator.

## Production capability matrix

| Family | Conserved or solved quantity | Smooth derivative scope | Explicit boundary |
| --- | --- | --- | --- |
| Richards | Liquid mass with pressure primary | Successful nonsingular fixed-layout root | Single liquid phase |
| Coupled water/heat | Liquid mass and sensible energy | Successful fixed-layout root | No implicit latent/vapor term |
| Multiphase | Named component mass and total energy | Fixed accepted phase state | Phase appearance is a branch |
| Flash | Rachford–Rice material split | Away from phase-state margins | Caller supplies compositional K model |
| Hysteresis | Saturation branch/reversal state | Away from reversals and bounds | No hidden post-step clipping |
| Cryosphere/atmosphere | Enthalpy and explicit boundary exchanges | Smooth mushy/unlimited branch only | No atmospheric transport solver |
| Diffusive-wave surface | Water and sensible-energy storage | Away from wet/dry transitions | Projected-orthogonal geometry only |
| Shallow water | Water volume and momentum | Positive un-limited branch | Fixed unstructured surface topology |
| Wells | Completion volume/component rates | Fixed active control branch | Explicit rate/pressure switching |
| Split reactive | Component and mineral inventory | Successful local/root solves | Operator-split coupling |
| Monolithic reactive | Global component inventory and equilibrium | Successful global root | Cell-only transport unknowns |
| Fracture network | Matrix/fracture/intersection component and energy inventory | Fixed network/exchange topology | No generated or evolving DFN |

Analytic qualification covers closed balances, equilibrium states, phase fractions,
hysteresis reversal flags, condensation versus storage-limited evaporation, well
control, aqueous activity finiteness, monolithic equilibrium, and mixed-dimensional
exchange. These repository checks are not field calibration.

The solid-Earth benchmark and qualification commands are listed in the
[native geophysics guide](guides_geophysics.md).
