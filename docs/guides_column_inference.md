# Interactive-column calibration and intervention design

`phydrax.applications.geophysics._inference` binds the actual
`InteractiveMoistColumnPlan` to native geophysical observation products, bounded
least squares, fixed-step replay, UQ information actions and pairing-aware SVD.
It is not a surrogate climate simulator, assimilation filter or independent UQ
engine. Synthetic evidence here qualifies the numerical inverse workflow; it
**does not validate Earth climate, inferred Earth parameters or global posterior
coverage**.

## Runnable physical twin

From the repository root:

```bash
JAX_ENABLE_X64=1 PYTHONPATH=. python examples/interactive_column_inference.py
JAX_ENABLE_X64=1 PYTHONPATH=. python tools/column_inference_qualification.py
```

Both accept `--steps` and `--maximum-steps`. The example additionally accepts
`--skip-gradient-audit`; the qualification deliberately requires the audit. The
small deterministic twin uses two fixed-volume, top-to-bottom atmospheric layers,
prognostic vapor and falling rain, native grey radiation, ventilated surface
exchange, conservative interlayer mixing and an interactive liquid slab. Six
jointly fitted physical parameters are:

- longwave absorption scale;
- shortwave absorption scale;
- sensible-heat transfer coefficient;
- moisture transfer coefficient;
- background diffusivity;
- rain/snow evaporation timescale.

The default observations are exact physical-model means. Explicit synthetic
instrument variances still define the stipulated observation-error likelihood;
they are not estimates of empirical sensor errors. `twin_data(...,
noise_seed=integer)` instead adds reproducible Gaussian instrument noise. Optical
coefficients, initial conditions, forcing and error choices carry an explicit
synthetic provenance label. Rain/snow retain the owning radiation plan's
transparent-precipitation convention.

The workflow contrasts a deliberately confounded single TOA net-flux sample
with complementary TOA/surface radiation, atmospheric temperature and specific
humidity profiles, slab temperature/water inventory, and cumulative precipitation.
A single scalar constrains at most one combination of the six parameters; it
cannot separately identify greenhouse opacity, solar absorption and turbulent
or microphysical response. Complementary observations provide additional local
combinations. The qualification checks the measured rank, rather than assuming
that more observations necessarily produce identifiability.

After calibration, prospective greenhouse-opacity, solar, slab-capacity and
targeted-profile experiments are ranked. Only their forcing, observation
operators and declared noise enter design. Greenhouse and solar outcomes are
then generated separately and scored as **held-out physical predictions**. Their
values never enter optimization or candidate selection. The example returns
physical prediction vectors, errors, noise-normalized error, local uncertainty,
coverage diagnostics and lineage IDs. Exact-mean twin coverage is not an
empirical coverage study.

## User observation products and time support

Prepare real user-supplied products through the existing
`prepare_geophysical_observations` interface; no new file-provider or provenance
inference layer is introduced. A `ColumnObservationBinding` associates one
`GeophysicalObservationOperator` with a physical signal and numerical sample
times. `ColumnExperiment` gives the initial physical state, calendar/epoch, fixed
step, number of steps, forcing and an explicit source label. `ColumnObservationData`
requires the corresponding `PreparedGeophysicalObservations`, an explicit
`provenance`, and either `role="calibration"` or `role="holdout"`.

Supported signal semantics are strict:

| Signal | Quantity kind | Sign | Reference |
| --- | --- | --- | --- |
| `toa_net_upward_flux` | `radiative_flux` | `upward` | `absolute` |
| `surface_net_downward_flux` | `radiative_flux` | `downward` | `absolute` |
| `temperature` | `temperature` | `positive` | `absolute` |
| `specific_humidity` | `specific_humidity` | `positive` | `absolute` |
| `surface_temperature` | `temperature` | `positive` | `absolute` |
| `surface_water_mass` | `water_mass_per_area` | `positive` | `absolute` |
| `precipitated_water` | `precipitation_amount` | `positive` | `experiment-initial-state` |

Specific humidity is vapor divided by total atmospheric mass, including
condensate. Fluxes are instantaneous reevaluations of the owning radiation plan
at the observed state. Temperature/humidity source arrays have the actual layer
count; scalar boundary/inventory signals have shape `(1,)`. Native transfer
operators may select or interpolate these explicitly fixed supports. Quantity
unit conversion is performed before applying the operator.

Only grid-aligned instantaneous observations are supported by the existing
geophysical observation operator. Calendar, epoch, temporal support, quantity
kind, reference and signs must match; there is no nearest-time snap. The column
clock is seconds from the named epoch; binding times are in the clock's declared
units. The precipitation signal samples the native **cumulative ledger state
relative to experiment initialization** at a point time. It is neither an
instantaneous precipitation rate nor an unlabeled interval accumulation. Mean
fluxes, satellite averaging kernels and interval products must be represented
by an appropriate independently supplied observation operator/support; this
point-time binding rejects them rather than treating them as point samples.

### Explicit covariance and masks

`ColumnObservationData(..., covariance=R)` accepts a joint covariance in flattened
order: binding, time, then native target-field order. Cross-time/cross-signal
correlations may be specified. Its marginal variances must match the native
prepared measurement-plus-representativeness error variances on accepted
observations. If `R` is omitted, independence is explicitly assumed using those
native variances. QC/missing observations are removed using a principal
covariance submatrix; missing values are not replacement evidence. Accepted
covariance must be strictly positive definite: no hidden jitter or repair occurs.
Whitening uses native linalg, and the covariance is a native `uq.DenseCovariance`.

## Parameter coordinates and local identifiability

`ColumnParameterSpace(names, scales, lower, upper)` binds existing numeric leaves;
physical parameters satisfy `theta = scales * z`. These nondimensional scales
are **not a prior**. All native mixing/microphysics timescales, diffusivity,
mixing length, critical Richardson number, cloud threshold, fall speeds and
surface/optical coefficients are available. Scalar optical parameters set all
four species' calibration scales equally; species contrast remains in the
fixed, provenance-tagged optical properties. Active radiation or exchange is
required to fit its coefficients. Bounds are physical, explicit and enforced by
native `optim.BoundedLevenbergMarquardt`; initial values outside bounds raise
instead of being clipped.

```python
from phydrax.optim import OptimizationTermination

problem = ColumnCalibrationProblem(plan, space, (training_data,))
result = problem.calibrate(
    initial_physical_parameters,
    termination=OptimizationTermination(maximum_steps=24),
)
info = result.information
```

The native termination policy is exposed explicitly. Optimality is the norm of
the whitened residual gradient in dimensionless parameter coordinates, not a
raw physical-unit residual tolerance. Choose its accuracy consistently with
observation uncertainty and forward floating-point resolution. The example
reports objective, initial/final optimality, final step and the stopping
threshold; native unsuccessful statuses are never relabeled as success.

The deterministic float64 twin explicitly uses absolute optimality `1e-7`.
At its numerically recovered optimum, measured diagnostics were objective
`1.27e-20`, maximum whitened residual `1.14e-10`, and gradient norm `2.42e-8`.
The example's tolerance accommodates that measured normalized numerical
resolution without changing the native core default, observation likelihood,
or optimizer status. The full selected policy and actual stopping diagnostics
are emitted; no extra initial Jacobian or heuristic operation-count allowance
is computed.

The native optimizer result is retained. `result.successful` requires optimizer
success, finite residuals and physically regular derivatives. Failed column
admission gives nonfinite residuals rather than a spurious exact fit to an
unchanged/zero-output state. No observation-driven state analysis increment is
written into the physical mass, energy or forcing ledger.

`information()` differentiates the actual fixed-step physical prediction via
JVP. SVD acts on the noise-whitened sensitivity in dimensionless parameter
coordinates. `combinations` contains orthonormal parameter directions **by row**;
`identifiable`, `rank` and `singular_values` expose rank/confounding at the
explicit `rank_rtol`. `unidentifiable_projector` identifies directions that the
observations do not locally constrain.

`covariance_on_identifiable_subspace` is the local linear Gaussian covariance
**conditional on holding null directions fixed**. Its zero entries in null
directions are not zero uncertainty. This is not an inverse-Fisher assertion of
a globally Gaussian posterior. Physical covariance of identifiable directions
uses `diag(scales) @ covariance @ diag(scales)`. Active bounds are reported;
unconstrained Gaussian design/scoring is invalid at a bound. Local rank can
change with initial state, forcing, observation placement and parameter point.

`problem.fisher_action(theta, direction)` uses native UQ JVP/VJP actions and
retains physical derivative validity. `column_local_information` is also usable
on an analytic whitened map; the qualification checks a diagonal identifiable
Gaussian limit and a rank-one sum with an unconstrained contrast.

## Derivative validity and numerical resolution

The native fixed-step adapter accumulates `step.derivative_valid` across the
entire physical trajectory, including phase, source availability, surface donor,
stability and admission boundaries. It does not rely only on terminal
temperature diagnostics. `column_gradient_audit` compares the true JVP against
central **actual model perturbations**, checking both parameter bounds and
physical flags. It additionally rejects stencils that change sampled phase,
saturation, condensate/fallout-presence or cloud-threshold regimes. Errors remain
visible even when a stencil is invalid; they must not be cited as a gradient
certificate. No nonsmooth branch is secretly smoothed.

The audit also runs the same experiment on explicit `dt/2` and `dt/4` grids and
reports changes in the JVP at the same physical observation times. This tests
sensitivity to the first-order physical splitting, not merely autodiff versus
finite differences of one discretization. Physical/time-step rejections remain
rejections; the binding never substeps or repairs state automatically. The
qualification requires valid regular-state stencils, accurate epsilon agreement
and improving time-step refinement.

## Prospective interventions and conditional local information gain

`ColumnDesignCandidate` contains an experiment and declared observation noise,
**not observed outcomes**. `design_column_intervention` requires an explicit SPD
`reference_covariance` in dimensionless coordinates. It forms the local
conditional covariance `(C_ref^-1 + F_cal)^-1` and evaluates native UQ mutual
information `0.5 log det(I + C_cond^(1/2) F_candidate C_cond^(1/2))` through actual
candidate forward sensitivities. The reference uncertainty is not inferred from
held-out outcomes and is not an implicit regularization constant. Every result
labels the local-linear-Gaussian approximation; all-invalid candidates return
`chosen=-1`, never a fabricated informative experiment.

`ColumnIntervention` supports solar, LW opacity and dry slab-capacity multipliers.
Changing capacity is a **pre-experiment preparation at fixed initial slab
water/temperature**. Its energy difference is reported as `preparation_energy`,
not inserted into time-integrated physical heating. This is not an in-flight
unaccounted slab heat source. A candidate with different bindings implements a
targeted-observation intervention.

`score_column_holdout` rejects calibration-role products, repeated product IDs
and reuse of the calibration experiment. Prediction uncertainty propagates the
local identifiable covariance plus declared measurement covariance. If the
prediction sees any unconstrained parameter direction its marginal standard
deviation is infinite and global coverage is invalid. Nonsmooth trajectories
and active bounds also invalidate the Gaussian interpretation. Reported coverage
is only a diagnostic conditional on this local model and the stipulated errors;
it does not establish calibration of a global posterior or Earth prediction.

## Artifacts and physical continuation

`save_column_inference(path, problem, result)` saves accepted parameters and
identifiability evidence using the native array archive, and saves the actual
terminal column through its native `save_checkpoint`. Artifacts bind the
structural model, numeric plan leaves, initial inventories, parameter
scales/bounds, observation identities/masks/covariance/provenance and experiment
forcing/time/intervention. The native checkpoint verifies exact numeric physical
parameters, clocks and budgets on reload; an evidence digest also binds the
terminal state and inference arrays.

`load_column_inference(path, problem)` requires matching model/parameter/data
lineage and returns `parameters`, the effective physical `plan`, exact native
`state`, evidence and manifest. Calling the returned plan's `step` continues the
actual physical trajectory. This is **not optimizer warm-state continuation**:
a caller may explicitly restart native optimization from the saved bounded
parameter point. The qualification compares a resumed physical step against an
uninterrupted one exactly. No alternate restart format or schema version is
introduced.
