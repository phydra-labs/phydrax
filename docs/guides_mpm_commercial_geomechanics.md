# Commercial MPM geomechanics and coupled fields

## General plane stress

`GeneralPlaneStressMPMConstitutivePlan` augments a two-dimensional in-plane
deformation with a three-component transverse director and solves:

```text
P[:,3] = 0.
```

It stores the director in material history and returns the full implicit Schur
condensation of the three-dimensional algorithmic tangent. The scalar isotropic
`P33 = 0` closure remains the cheaper specialization.

`MPMMaterialOrientation` and `OrientedMPMConstitutivePlan` rotate deformation,
stress, and tangent actions through one validated proper-orthogonal material frame.

## Pressure-dependent materials

Commercial geomechanical plans are independent models:

- `DruckerPragerMPMConstitutivePlan`: smooth pressure-dependent yield with separate
  friction and dilation angles and a nonsymmetric non-associated tangent.
- `MohrCoulombMPMConstitutivePlan`: multiplicative principal-stress face/edge/apex
  returns with all-plane admissibility and nonassociated dilation.
- `ModifiedCamClayMPMConstitutivePlan`: Borja–Tamagnini finite-strain hyperelastic
  Modified Cam-Clay with associated flow and natural-volume hardening.
- `NorSandMPMConstitutivePlan`: modern image-state log surface, limited
  hardening, pressure-dependent hypoelasticity and work-conjugate 3-D flow.

Each response reports yield residual, plastic multiplier, branch/corner state,
dissipation, admissibility, material step recommendation, and algorithmic tangent.
No incremental-potential or symmetric-positive-definite claim is made for
non-associated flow.

### Drucker–Prager conventions and initialization

Kirchhoff and Cauchy stresses are tension-positive; pressure
`p = -trace(tau)/3` is compression-positive. With `s = tau + p I`,
`q = sqrt(3/2 * s:s)`, the outer Mohr–Coulomb fit is
`M_phi = 6 sin(phi)/(3 - sin(phi))` and
`k = 6 c cos(phi)/(3 - sin(phi)) + H alpha`. Yield is
`f = q - M_phi p - k`; confinement therefore strengthens the material.
The potential is `g = q - M_psi p`, with the same fit for the dilation angle.
Positive dilation produces plastic volumetric expansion, not compaction.
The smooth return has `delta_gamma = f_trial/(3 G + K M_phi M_psi + H)`,
`q_new = q_trial - 3 G delta_gamma`, and
`p_new = p_trial + K M_psi delta_gamma`.

Branch codes are `0` elastic, `1` cone, `2` apex, `4` tension, `5` corner,
and `-1` failure. Optional `tensile_strength` imposes the mean-stress cutoff
`-p - tensile_strength <= 0`. A zero-dilation trial beyond the apex without a
cutoff is explicitly inadmissible, not silently repaired.

J2 and Drucker–Prager history is the six-component plastic metric
`(11, 22, 33, 23, 13, 12)` followed by equivalent plastic strain, not nine
components of `Fp`. Use `initialize_state` for the identity metric.
For prestress at `F = I`, call
`material.initialize_state_from_stress(kirchhoff_stress, parameters,
equivalent_plastic_strain=0.0)`. The returned `PrestressedMaterialState` contains
`state`, `admissible`, and `yield_value`. It constructs an elastic metric from
`epsilon_e = dev(tau)/(2 G) + trace(tau)/(9 K) I` using the native Hermitian
exponential. Check `admissible` before committing; traced inadmissible stress
returns evidence rather than raising.

`NonlocalSofteningPlan(*, viscosity=0.0)` combines caller-provided local and
nonlocal history with optional rate regularization. It does not perform a spatial
nonlocal solve or own a characteristic length; those belong to the caller's
nonlocal discretization. Local negative hardening is not presented as mesh objective.

### Mohr–Coulomb ordered return

`MohrCoulombParameters(G, K, c, phi, psi, H=0)` uses radians and
`c(alpha) = c + H alpha`: `H` hardens cohesion, not the entire yield intercept.
With tension-positive ordered principal Kirchhoff stresses `tau1 >= tau2 >= tau3`,
yield is `(tau1-tau3) + (tau1+tau3) sin(phi) - 2 c(alpha) cos(phi) <= 0`.
The potential replaces `phi` by `psi`. Positive `psi` produces volumetric
plastic expansion; `psi = phi` is associated flow.

The closed-form algorithm first returns to the main plane, then tests left
(`tau1 = tau2`) and right (`tau2 = tau3`) edges, then the apex. Ordering,
nonnegative multipliers, and all six plane inequalities certify admission.
The algorithmic structure follows de Souza Neto, Perić and Owen (2008),
*Computational Methods for Plasticity*, §8.2; the module docstring derives
the formulas, including the `2 H cos(phi)` hardening factor. No tension
cutoff or generic multisurface root is used.

History is seven entries: the packed plastic metric
`Cp_inverse(11,22,33,23,13,12)` followed by `alpha`, the accumulated **sum
of plane multipliers**, not a von Mises equivalent strain. Initialize the
identity metric with `initialize_state`; do not pass zero history.
Prestress uses `initialize_state_from_stress(tau, parameters,
equivalent_plastic_strain=alpha)` and returns `PrestressedMaterialState`;
check `admissible` before committing. The keyword supplies the MC hardening
scalar under the stated multiplier convention.

Branch codes are `0` elastic, `1` face, `2` edge, `3` apex, `-1` failure.
Diagnostic `edge_side` is `-1` left, `+1` right, and zero otherwise.
`plane_yield_values`, ordered trial/returned stresses, plastic increment,
total multiplier, and `strict_branch` provide local evidence.
`evaluate_linearized` returns finite consistent first-order tangents at
repeated spectra on strict branches and reports invalid tangents on branch
boundaries. Failed returns preserve committed history and recommend cutback.
Recoverable energy is elastic Hencky energy; the reported dissipation is
plastic stress work. Cohesion hardening is phenomenological, not an additional
stored-energy model.


### Modified Cam-Clay energy and initialization

This is the two-invariant, multiplicative finite-strain model of Borja and
Tamagnini (1998), *Cam-Clay plasticity, Part III: Extension of the infinitesimal
model to include finite strains*, CMAME 155, 73–95,
[DOI: 10.1016/S0045-7825(97)00141-2](https://doi.org/10.1016/S0045-7825(97)00141-2).
It is not a constant-bulk-modulus or pressure-dependent hypoelastic MCC variant.

Construct `ModifiedCamClayParameters(mu0, alpha0, M, lambda_raw, kappa_raw)`.
The last two inputs are conventional void-ratio versus log-pressure slopes.
Normalize exactly once using the fixed initial void ratio:
`lambda_tilde = lambda_raw/(1 + e0)`,
`kappa_tilde = kappa_raw/(1 + e0)`. Do not use the current void ratio or pass
already-normalized slopes. Require `lambda_raw > kappa_raw > 0`, `M > 0`,
`mu0 >= 0`, `alpha0 >= 0`, and positive effective shear stiffness.

With tension-positive elastic Hencky strain `epsilon_e = 0.5 log(Fe Fe.T)`,
`a = trace(epsilon_e)`, `d = dev(epsilon_e)`, and
`b = sqrt(2/3 * d:d)`, the reference elastic volume strain is fixed at
`epsilon_v0 = 0`. The exact stored energy per reference volume and its stresses are:

```text
R = kappa_tilde
B = p0 exp(-a/R)
mu = mu0 + alpha0 B
psi = B R + (3/2) mu b^2
tau = -p I + 2 mu d
p = B [1 + 3 alpha0 b^2/(2 R)]
q = 3 mu b
f = q^2/M^2 + p(p - pc) <= 0
```

The shear-induced pressure term is part of the energy derivative, not an optional
correction. `has_free_energy` is true. `mu0` is the additive constant modulus:
at the zero-shear reference the total shear modulus is `mu0 + alpha0 p0`.
`p0` is the initial mean Kirchhoff pressure, distinct from the hardening pressure
`pc`; it is never inferred from `pc`. Plastic compaction
`delta_epsilon_v_p < 0` increases `pc` through
`pc_new = pc_old exp(-delta_epsilon_v_p/(lambda_tilde - kappa_tilde))`.

The no-argument `ModifiedCamClayMPMConstitutivePlan()` has no hidden reference
fields. Its width-nine history is `Cp_inverse(6), pc, p0, e0`, with metric order
`(11, 22, 33, 23, 13, 12)` and off-diagonals stored once.
`initialize_state` returns the identity metric and zero reference scalars; that
state is deliberately inadmissible for evaluation. Prestress initialization is
required, including for an initially hydrostatic material:

```python
material = phx.applications.solid_mechanics.ModifiedCamClayMPMConstitutivePlan()
parameters = phx.applications.solid_mechanics.ModifiedCamClayParameters(
    5400.0, 0.0, 1.05, 0.234, 0.0324
)
tau0 = -90.0 * jnp.eye(3, dtype=jnp.float64)[None, :, :]
initialized = material.initialize_state_from_stress(
    tau0, parameters,
    preconsolidation_pressure=180.0,
    initial_void_ratio=0.8,
)
# Commit initialized.state only when initialized.admissible is true.
```

All stresses and moduli in this example use kPa. Initialization accepts batched
3×3 effective Kirchhoff stresses at `F = I`, sets `p0 = -trace(tau0)/3`, and
inverts the hyperelastic energy to recover the metric (including shear-pressure
coupling for nonhydrostatic stress). The returned `PrestressedMaterialState`
reports `state`, `admissible`, and `yield_value`. In particular, an initial
deviatoric stress requires `pc >= p + q^2/(M^2 p)`, not merely `pc >= p`.
Effective Cauchy stresses must first be converted to Kirchhoff stresses.

The soil domain refuses `p <= 0`, nonpositive reference/hardening pressure,
invalid elastic metrics, and failed local solves; there are no pressure floors.
Failure retains committed history and returns explicit evidence.
Diagnostics include `p`/`mean_pressure`, `q`/`equivalent_stress`,
`preconsolidation_pressure`, `reference_pressure`, `initial_void_ratio`,
`elastic_volume_strain`, `elastic_shear_strain`, `yield_residual`,
`plastic_multiplier`, `root_successful`, and `tangent_successful`.


### NorSand: modern log surface and hypoelastic stress memory

`NorSandMPMConstitutivePlan` implements the modern Jefferies–Been log-surface
baseline, with Jefferies–Shuttle (2002) work-conjugate 3-D principal flow.
Its pressure-dependent shear modulus is
`G = G_ref (p/p_ref)^m`, with `K = 2 G (1+nu)/(3(1-2nu))`.
This is a hypoelastic rate law: **`has_free_energy=False`**; the zero energy
response field is not a recoverable energy or an implicit potential.

```python
import jax.numpy as jnp

from phydrax.applications.solid_mechanics import (
    NorSandMPMConstitutivePlan, NorSandParameters,
)

# All stresses/moduli in kPa; Gamma is the CSL intercept at 1 kPa.
parameters = NorSandParameters(
    20000.0, 100.0, 0.4, 0.2, 1.0, 0.02, 1.3, 0.3, 3.0, 250.0, 250.0,
    critical_state_reference_pressure=1.0,
)
material = NorSandMPMConstitutivePlan()
initial = material.initialize_state_from_stress(
    -100.0 * jnp.eye(3), parameters,
    initial_void_ratio=0.8, overconsolidation_ratio=2.0,
)
# Check initial.admissible before accepting initial.state.
```

Parameter order is reference shear modulus, modulus reference pressure,
pressure exponent, Poisson ratio, CSL intercept, CSL slope, TC critical
friction ratio, volumetric coupling N, TC dilatancy coefficient, H0 and Hy.
The CSL pressure reference is independent of the modulus reference and must
scale with all stresses when changing units.

The selected equations are `psi=e-ec(p)`, `psi_i=e-ec(pi)`,
`Mi_tc=Mtc-N chi_i abs(psi_i)`, `Mi=Mi_tc g(theta)`,
`f=q/(p Mi)-1+ln(p/pi)`, and source compression-positive
`Dp=Mi-q/p`. Image-pressure hardening is
`d ln(pi)=d gamma H g(theta) (p/pi)^2 (pi_max/p-pi/p)`,
`pi_max=p exp(-chi_i psi_i/Mi_tc)` and `H=H0-Hy psi`.
The documented image-coefficient guard is enabled:
`chi_i=chi_tc/(1-lambda chi_tc/Mtc)` below the 0.2 ratio threshold,
and `1.25 chi_tc` above it. Negative H and nonpositive image friction
are explicit inadmissible states, never clipped.

History width 23 stores packed `Cp_inverse`, packed Kirchhoff stress,
previous F, current void ratio and image pressure. Mandatory prestress
initialization chooses a **preloaded reference** with initial elastic
metric I; it does not claim that hypoelastic stress uniquely determines an
initial elastic strain. OCR multiplies the image pressure found from the
modern yield equation. OCR below one is preserved and reported inadmissible
if outside the surface. Pressure `p <= 0` is outside the model domain.

Incremental polar rotation transports committed stress and elastic metric.
Native spectral logs give the corotated elastic Hencky increment; endpoint
moduli and the principal five-unknown return are coupled through a scalar
pressure root. The metric is reconstructed with the native spectral exp
and native small solves. `evaluate_linearized` differentiates this discrete
update. Branch codes are 0 elastic, 1 outer-surface plastic and -1 failure;
failures retain the entire committed history. Diagnostics expose pressure,
image pressure, hardening limit, geometric limit stress ratio, dilatancy,
Lode factor, plastic multiplier, yield/pressure-root residuals and tangent
status. TC/TE principal flow is interpolated according to Jefferies–Shuttle
Eqs. 14–16, not obtained from an arbitrary Lode-dependent yield gradient.
Separate inner-cap hardening, cyclic/PSR terms and extra softening are absent.

Spectral inputs are normalized by the modulus reference pressure, so native
confluent eigenvalue tests are dimensionless. Newton stresses and plastic
shear use pressure and elastic-strain scales respectively; residual tolerances
are `max(1e-10, 64*machine_epsilon)`. Float32 and float64 admission are tested.

At `q=0`, `lode_defined=False` and no deviatoric plastic direction is invented.
Hydrostatic loading outside the outer surface is rejected: this outer-only
baseline does not implement inner-cap flow. Tangent status is false at the
loading/unloading switch or an active image-strength absolute-value kink;
an inactive image-strength kink does not invalidate an elastic tangent.
Only first-order derivatives are promised. Near repeated TE stresses the
principal map is generally C1 but not C2, so tangent finite-difference
qualification uses step refinement rather than assuming second-order error.

**Oracle scope and unresolved workbook parity.** The CC BY 4.0 subset in
`tests/fixtures/geomechanics/norsand4ai.json` credits Ozelim, Casagrande and
Cavalcante, dataset DOI [10.5281/zenodo.8170537](https://doi.org/10.5281/zenodo.8170537),
article DOI [10.5194/gmd-17-3175-2024](https://doi.org/10.5194/gmd-17-3175-2024).
It records the exact member paths, zero-based row selection, attributes,
units and retrieval command. The 19.9 GB archive was not downloaded:
HTTP Range reads fetched its ZIP64 directory and selected compressed HDF5 members.
The independent NumPy rate-form driver solves TC consistency and drained
confinement/undrained volume constraints without calling the JAX integrator.
The primary curve oracle is this independent integration of **exactly the
published image-state equations**. With axial log-strain increments of
1/6400, the maximum p/q error normalized by initial pressure through 1%
strain is 0.0141612 for dense undrained sand, 0.0103597 for loose drained
sand and 0.0117330 for loose undrained sand. The test bounds this
backward-Euler integration error by 0.025 and void-ratio error by 0.0005;
dense drained loading continues to 2%, with error 0.0156394, and includes the
actual phase change to dilation.


Workbook parity is **not established**. Its Dp fields are lagged (the first
two recorded rows can have identical Dp despite different eta), so Dp+eta
is not a reliable endpoint Mi. For Par_0_0, the recorded initial image ratio
is `exp(eta/Mtc-1)=0.362004727` despite OCR=0.8552; its recorded maximum ratio
0.982188106 matches current-state `exp(-chi_tc psi/Mtc)`, not the frozen
image-state expression. The released HDF5 curves do not expose the workbook
cell formulas. Independent integrations through 1% strain with the minus-Hy
law gave maximum (p,q) errors normalized by p0 of (0.029114,0.087342)
for drained Par_0_0 and (0.157130,0.208510) for undrained Par_0_0.
Those discrepancies are **not** attributed solely to finite-strain effects,
and no broad tolerance disguises them as model qualification. Changing only
the H sign did not resolve the discrepancy; chi threshold parity cannot be
inferred from these materials, whose ratios are below 0.2. The frozen modern
law is retained rather than inventing a hybrid from incomplete diagnostics.

Late TC workbook yield diagnostics satisfy the modern log surface with
`Mi=Mtc-N chi_tc abs(current psi)`, and its recorded maximum-image ratio
matches `exp(-chi_tc current psi/Mi)`. Independent integrations of these
inferred current-state approximations fit dense Par_0_2 through 1% with
normalized (p,q) errors (0.003046,0.003664) drained and
(0.000551,0.019614) undrained when using `H0-Hy psi`; the plus sign
gave larger errors for these dense curves. But loose undrained Par_0_0
still differs by (0.126755,0.170755). Those diagnostic identifications
do not establish the complete workbook equations, and are not mixed into
the published image-state implementation. The fixture provenance records
this distinction. Dense comparisons have a separately named small-strain
envelope; no passing tolerance is asserted on the known loose mismatch.
The published image-state implementation is compared separately with **all**
recorded Par_0_2 rows through 1% strain (20 drained and 46 undrained rows).
Measured normalized (p,q) errors are (0.0061251,0.0129029) drained and
(0.0004716,0.0193573) undrained; the comparison envelope is 0.025.
This is a measured cross-implementation envelope including workbook
current-state, endpoint integration and strain-driver differences, **not**
exact workbook parity or experimental validation. In kPa/m/s calculations,
mass density must be in tonne/m³; Pa/m/s uses kg/m³.

References: Jefferies (1993), [10.1680/geot.1993.43.1.91](https://doi.org/10.1680/geot.1993.43.1.91);
Jefferies & Shuttle (2002), [10.1680/geot.2002.52.9.625](https://doi.org/10.1680/geot.2002.52.9.625);
Jefferies & Been, second edition (2016), [10.1201/b19114](https://doi.org/10.1201/b19114).
The modern equation-number cross-check is
[Itasca 9.8 Eqs. 9–19](https://docs.itascasoftware.com/itasca980/common/models/norsand/doc/modelnorsand.html);
these are not asserted to be the book's equation numbers.

## Typed coupled fields

`MPMPhysicalFieldPlan` distinguishes mechanical velocity, pore pressure, saturation,
temperature, damage, and species. Contact velocity slots are not reused for scalar
physics.

`PreparedMPMCoupledFieldOperator` implements:

```text
S p_dot + alpha div(u_dot) - div(k/mu grad p) = source

rho c T_dot - div(k_T grad T) = chi plastic_dissipation + source
```

It returns Darcy and heat fluxes, pore-pressure and temperature residuals, and the
Biot/thermal effective-stress correction. Dirichlet and flux boundary conditions are
one prepared `MPMCoupledBoundaryPlan`. JVP and transpose actions include pressure and
temperature cross-dependence.

Application validation remains material, drainage, thermal, geometry, loading, and
observable specific.

## Saturated double-point solid–water dynamics

`DoublePointMPMPlan` couples effective-stress MPM skeleton particles to separate
FLIP water particles through one matrix-free mixed operator `D = [D_s, D_f]`.
The phase divergences and pressure-force pullbacks are exact algebraic adjoints,
with physical deposited phase masses and cell-volume pairing for physical
pressure. Accepted pressure is applied once before lagged implicit drag; one
mixed pressure-increment solve follows drag. Do not add pore pressure to the
material's effective skeleton stress or apply a second solid pressure force.

A physical reservoir/free-surface pressure datum is mandatory; datum-free
configurations are refused. Fractions are not clipped, stationary phase
component masks are part of the operator, and both phases roll back together
when an admission, solve, or material update fails. This path requires
Cartesian dense single-field USL MPM; axisymmetric, sparse, multifield, and
other stress-update schedules are refused.

See the [double-point MPM guide](guides_mpm_double_point.md) for the pressure
multiplier's units/sign, drag action–reaction, consumer evidence, the real
`examples/double_point_mpm.py` step, and the incompressible `alpha=1`, `S=0`
Darcy/Terzaghi limits. The typed finite-storage scalar-field operator above
is a distinct model, not an additional pressure solve for this transaction.
