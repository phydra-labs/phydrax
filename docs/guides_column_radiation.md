# Conservative grey column radiation

`ColumnRadiationPlan` solves actual upward/downward radiative transfer through a
stack of atmospheric layers. Shortwave scattering and absorption, thermal
emission and absorption, surface reflection, and surface thermal emission are
coupled through interface fluxes. This is not relaxation to a prescribed
radiative-equilibrium temperature.

## Declared scientific scope

The model is a **plane-parallel, two-band grey reference approximation**:

- Shortwave uses the hemispheric two-stream closure, including coherent
  scattering and an explicit scattering asymmetry moment.
- Incoming `solar_down` is a **diffuse hemispheric downward flux at the top of
  atmosphere**. It is not direct beam-normal irradiance. The model does not
  calculate zenith-angle attenuation, direct-to-diffuse redistribution, or
  spectral solar absorption. A caller that maps actual sunlight onto this
  diffuse boundary is making an additional approximation, which must be stated.
- Longwave uses grey LTE absorption/emission, with hemispheric diffusivity 2.
  Each layer is homogeneous and isothermal. Longwave scattering by condensate
  is neglected, explicitly, rather than included in an absorption coefficient
  while claiming a scattering calculation.
- All thermal Planck emission belongs to the longwave channel; atmospheric and
  surface shortwave thermal emission are neglected. This separation is intended
  for cool atmospheric/surface states, not incandescent objects.
- No molecular lines, correlated-k table, pressure broadening, aerosol size
  distribution, Mie calculation, cloud fractional overlap, lateral radiation,
  refraction, or radiative time storage is implied.
- Falling rain/snow are explicitly **optically transparent precipitation**.
  Their mass is excluded from dry gas opacity and is not assigned cloud
  area-per-mass coefficients. Their thermodynamic energy and settling physics
  remain active in the caller; this is only an optical approximation.
- Required mass coefficients are supplied by the user. **There are no default
  atmospheric opacities and no hidden data provider.** A reference tag is a
  provenance declaration, not certification that chosen coefficients represent
  an actual atmosphere.

Gas, vapor, cloud liquid, and cloud ice masses determine optical depth at every evaluation.
Temperature determines LTE source flux, not a temperature-dependent opacity.
The dry coefficient describes the entire non-water gas mixture. Scaling it can
represent an explicitly calibrated grey greenhouse perturbation, but is not an
implicit mapping from CO₂ concentration to logarithmic forcing.

### Primary derivation

The implemented shortwave equations are the hemispheric closure in
[Heng, Mendonça & Lee (2014), *Analytical Models of Exoplanetary Atmospheres. II.
Radiative Transfer via the Two-Stream Approximation*](https://arxiv.org/html/1405.0026),
especially equations 11, 19, 23, and 66–69. Their sections 2.4–2.5 derive the
conservative-scattering and blackbody-emission constraints; section 3.2 derives
the anisotropic coefficients. The [full manuscript](https://arxiv.org/pdf/1405.0026)
was consulted. No coefficients or accuracy claims are imported from its
exoplanet applications. The closure is also identified in that primary paper
with [Toon et al. (1989)](https://doi.org/10.1029/JD094iD13p16287).

## Explicit inputs and native parameters

```python
import equinox as eqx
import jax.numpy as jnp
from phydrax.applications.atmosphere import (
    ColumnOpticalProperties,
    ColumnRadiationPlan,
)

# Illustrative numerical coefficients only: not observations or a fitted Earth model.
optics = ColumnOpticalProperties(
    shortwave_absorption=(2e-6, 0.003, 0.15, 0.1),
    shortwave_scattering=(1e-5, 0.0, 50.0, 30.0),
    shortwave_asymmetry=(0.0, 0.0, 0.8, 0.7),
    longwave_absorption=(5e-5, 0.07, 20.0, 15.0),
    reference_id="illustrative-grey-coefficients-not-measured",
)
plan = ColumnRadiationPlan(
    optics,
    surface_albedo=0.19,
    surface_emissivity=0.94,
    shortwave_absorption_scale=1.0,
    shortwave_scattering_scale=1.0,
    longwave_absorption_scale=1.0,
)
result = eqx.filter_jit(plan.evaluate)(
    temperature=jnp.array([230.0, 260.0, 285.0]),
    layer_mass=jnp.array([2000.0, 3000.0, 5000.0]),
    vapor_mass=jnp.array([1.0, 5.0, 20.0]),
    liquid_mass=jnp.array([0.0, 0.02, 0.0]),
    ice_mass=jnp.array([0.01, 0.0, 0.0]),
    surface_temperature=jnp.asarray(295.0),
    solar_down=jnp.asarray(350.0),
)
# Check successful before committing any reservoir update.
```

Every optical vector has four entries in **(dry, vapor, liquid, ice)** order.
Absorption/scattering coefficients are m²/kg **of that species**, not of total
moist mass. Asymmetry is dimensionless and lies in [−1, 1]; zero is isotropic,
one is the no-backscatter limiting case of this closure. The layer mass includes
all water phases; dry mass is layer mass minus vapor, cloud liquid/ice, and
falling rain/snow masses. Pass the latter using optional `rain_mass=0.0` and
`snow_mass=0.0` keywords, each a scalar per-layer value or an `[..., n]` array.
The fixed `optics.precipitation_optics == "transparent"` declaration is included
in the optical fingerprint. Do not add precipitation to the cloud arguments or
omit it while retaining its contribution to total layer mass: either would
misclassify its opacity.

`ColumnOpticalProperties` is immutable native `NonTrainableState`, containing
explicit reference-tagged numeric coefficients. Native `partition_trainable`
keeps this object fixed. `ColumnRadiationPlan` is not a `NonTrainableState`:
its three scale vectors, surface albedo, and surface emissivity are ordinary
trainable JAX array leaves. Each scale accepts a scalar or a four-species vector
and is stored as a four-species array. Scales multiply the corresponding fixed
mass coefficients; zero disables the corresponding interaction without a
special alternate solver. Surface albedo/emissivity may have broadcast batch
axes. Use `eqx.tree_at` to provide differentiated calibration values to an
already prepared plan; construction validates concrete prepared values.

The plan identifier describes the fixed optics, constants, and closure. It does
**not** identify the current values of trainable parameters. Scientific
checkpoints must therefore retain the full plan/parameters, not just this tag.
SI Stefan–Boltzmann constant 5.670374419e−8 W m⁻² K⁻⁴ is fixed, not trainable.

## Layer order, shape, and budgets

Layer arrays have shape `[..., n]`, with `n >= 1`, in **top-to-bottom order**.
All five layer arrays must have the same final-axis length. Leading batch axes
broadcast with the surface and incident-boundary arrays. Interface arrays have
shape `[..., n+1]`: interface 0 is top of atmosphere; interface n is the surface.
All temperatures are K, all layer masses are kg/m², and all returned fluxes and
transfers are W/m². No unit conversion occurs.

`upward_flux` and `downward_flux` are positive directional magnitudes summed over
SW and LW. The separately reported arrays are:

- `shortwave_upward_flux`, `shortwave_downward_flux`;
- `longwave_upward_flux`, `longwave_downward_flux`.

With `net_up = upward_flux - downward_flux`, transfers are defined only by these
same interface fluxes:

```text
heating          = net_up[..., 1:] - net_up[..., :-1]
surface_heating  = -net_up[..., -1]
space_heating    =  net_up[..., 0]
budget_residual  = sum(heating, axis=-1) + surface_heating + space_heating
```

`heating` is layer power per horizontal area, **not K/s**. Dividing by an
appropriate layer thermal capacity or updating a conservative energy state is
the integrator's responsibility. `space_heating` is signed net energy escaping
at TOA, including incoming solar energy with negative sign. Thus the external
reservoir ledger receives `dt * space_heating`, atmosphere receives
`dt * heating`, and surface receives `dt * surface_heating` exactly once.

`budget_residual` is a signed **unnormalized W/m²** residual. It is not divided by
a large total internal energy. Its smallness establishes discrete conservation,
not accuracy of grey optics relative to a real atmosphere.

## Transfer equations and stable solution

For each layer let A be its SW absorption optical depth and let
S = sum((1−gₛ) κₛ,scatter scaleₛ mₛ) be its transport scattering depth.
Both are dimensionless. On a normalized downward layer coordinate, the SW
fluxes obey

```text
a = 2 A + S,  b = S
dD/dx = -a D + b U
dU/dx = -b D + a U
```

This is the cited hemispheric closure expressed directly in absorption and
transport scattering depths. It avoids division by total extinction or
single-scattering albedo when the layer is transparent. With k² = a²−b², the
homogeneous slab's reflection and transmission are

```text
den = cosh(k) + a sinh(k)/k
R = b sinh(k)/k / den
T = 1 / den
Q = [cosh(k)-1 + (a-b) sinh(k)/k] / den
```

Q is absorptance. The code evaluates these expressions with decaying scaled
exponentials for large k and an even power series in k² near zero. This is a
numerical evaluation of the same analytic solution, not a clipping of optical
state. In a purely scattering slab, R = S/(1+S), T = 1/(1+S), Q = 0. In a purely
absorbing slab, R = 0 and T = exp(−2A).

Layers are added upward from the surface reflection condition Uₙ = α Dₙ,
then a downward scan reconstructs every interface. Both effective reflectivity
and its complement are carried separately. This avoids the cancellation of
`1-R` in a thick conservative slab over a perfectly reflecting surface. Cost and
storage are linear in layer count; there is no dense global transfer matrix or
iterative multiple-reflection convergence criterion.

Longwave absorption depth is L = sum(κₛ,LW scaleₛ mₛ). For the homogeneous LTE
layer define t = exp(−2L), e = (1−t) σ T_layer⁴. Then

```text
D_bottom = t D_top + e
U_top    = t U_bottom + e
U_surface = ε σ T_surface^4 + (1-ε) D_surface
```

The source factor uses `expm1` to retain small optical-depth accuracy. The surface
obeys grey Kirchhoff absorption/emission balance. LW albedo is 1−ε and need not
equal the independently specified shortwave albedo α.

The optional `longwave_down` keyword supplies explicit downward TOA longwave
irradiation (default zero, empty space). An isothermal column and equal-temperature
blackbody bath have zero thermal exchange. An isothermal atmosphere facing
empty space instead radiates and cools at its top; the model does not falsely
impose zero heating merely because temperatures are isothermal.

## Invalid states and differentiation

Prepared invalid coefficient shapes, nonfinite/negative opacities, out-of-range
asymmetry/surface parameters, or blank source tags raise `ValueError`. Inconsistent
runtime layer axes or nonbroadcastable shapes also raise native shape errors.

For each runtime batch column, `successful` requires positive finite layer and
surface temperatures, finite nonnegative masses, total water no greater than
layer mass, finite nonnegative incident fluxes/scales, valid optical parameters,
and finite resulting fluxes/transfers. Zero-mass transparent layers are allowed.
Any invalid layer rejects **the entire column**. Every numeric output of that
column is zeroed together while `successful=False`; other batch columns remain
independent. These zeros are rejected-output sentinels, never a physical fallback.
Check flags; do not infer success from a zero residual. Extreme finite values
that overflow representable arithmetic are rejected, not capped or clipped.

JAX differentiation traverses the optical-mass contractions and both interface
solves. The transparent and conservative-scattering limits avoid a spurious
`sqrt(0)` derivative singularity. Physical domain boundaries still restrict which
finite-difference perturbations are admissible; a central perturbation through
negative absorption is invalid. Derivatives of rejected states carry no physical
meaning. Use `eqx.filter_jit` for bound methods of the array-bearing plan, or a JAX
compiled function that receives the plan as a PyTree argument.

Increasing vapor LW opacity over a warmer surface can reduce outgoing LW and
increase surface back-radiation in this model. Scattering clouds can increase
reflected SW while cloud LW absorption increases back-radiation. Neither fact
establishes an unconditional sign for **net cloud forcing**. The qualification
cases compare the actual band-separated fluxes rather than assigning a universal
warming/cooling sign.

## Reproducible numerical evidence

```bash
PYTHONPATH=. python tools/column_radiation_qualification.py --layers 12
```

The tool emits JSON with the explicit optical tag, baseline W/m² transfers,
independent fundamental-matrix slab flux error, transparent and opaque-blackbody
errors, homogeneous-layer composition errors, and maximum unnormalized budget
residual in W/m². It also emits a six-step centered derivative epsilon sweep
against native JAX derivatives, with full derivative/error matrices and physical
control units. Controls cover vapor and liquid masses, surface temperature, LW
absorption calibration, SW scattering calibration, and incoming solar flux.
It fails on rejected perturbed states or stated numerical tolerances. It makes
no spectral/observational accuracy claim.

Dedicated regression cases additionally defend thick conservative scattering
over a perfect reflector, exact LTE bath equilibrium, shortwave/longwave forcing
separation, cloud/water transfer changes, top-to-bottom batch rejection, and a
finite analytic optical-zero derivative. These are independent limiting-model
checks, not comparisons to fabricated reference weather or climate data.
