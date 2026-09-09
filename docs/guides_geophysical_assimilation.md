# Geophysical observations and ensemble assimilation

The geophysical application layer prepares physical observation semantics and binds an existing numerical model to the native `phydrax.stochastic.StateSpaceProblem`. It does not implement another integrator, state-layout abstraction, or ensemble filter. Use the existing `phydrax.uq.ensemble_transform_kalman_filter`, `ensemble_filter_step`, `ensemble_kalman_smoother`, and ensemble checkpoint functions on the returned problem.

## Observation support and physical quantities

`GeophysicalObservationOperator` owns a native `FieldTransfer`, a `GeophysicalQuantity`, a `GeophysicalTimeSpec`, and an instantaneous `TemporalSupport`. The source and target are exact `DiscreteFieldSpace` objects. A transfer prepared by an existing discretization can be used directly, including a conservative grid remap when that native transfer genuinely supplies one. A transfer's properties are not upgraded by calling it an observation.

`prepare_tensor_observation_operator` prepares a sparse multilinear `FunctionLinearOperator` and its exact transpose for a scalar point-value tensor field. It uses at most 2 to the power of the coordinate rank corner contributions per observation; it does not allocate an observation-by-state dense matrix.

- **Stations:** supply one coordinate vector per source axis, each containing the station coordinates.
- **Grids:** supply flattened meshgrid coordinates and `kind="grid"`. The resulting observation axis is flat; the caller retains the grid shape for display. Grid point sampling is not cell averaging or a conservative transfer.
- **Vertical profiles:** include the fixed pressure or height axis and supply point coordinates for all source axes, with `kind="profile"`. The vertical axis uses increasing source coordinates, in the same units as the target coordinates. This is linear interpolation in the supplied coordinate, not an implicit log-pressure interpolation. Descending pressure arrays must be reordered together with their data. Hybrid columns whose pressure changes with surface pressure must first be reconstructed by their native owner onto the declared fixed-pressure support; supplying pressure values as hybrid-index values is incorrect.
- **Longitude:** an explicit `periods={"longitude": 360.0}` wraps a degree-valued axis. The source contains a single period without a duplicate endpoint. No other coordinate is extrapolated, and a singleton axis accepts only its exact location.

The source support ID must match the coordinate support. Axis names must exactly match the tensor layout. Coordinates are expressed in the source coordinate units: no coordinate-system transformation, spherical metric, pole treatment or unit guessing is performed. The operator content, coordinate values, periods, field identities, quantity and temporal semantics participate in preparation identity.

`prepare_geophysical_observations` accepts one physical case with exact shape `(time, *target_field_shape)`. Numerical times use the specified clock; ISO strings are encoded using its calendar and epoch. Times are strictly increasing. Calendar, epoch and time-unit identity must agree with the operator and the numerical model. The callback advances to these observation times; the preparation never silently chooses a nearby model step.

Only instantaneous point-time observations are supported by this adapter. Mean, accumulation, minimum and maximum interval measurements are rejected, even when their array shapes match. An instantaneous model evaluation cannot substitute for an interval operator.

## Missing values, quality control and errors

Preparation requires the observation quantity and the target support identity explicitly. Quantity compatibility checks physical kind, sign and reference configuration as well as dimensions. Local names and display units may differ. Values and standard deviations are converted into the operator's unit before assimilation; an absolute temperature is not interchangeable with an anomaly.

`GeophysicalObservationPolicy` makes rejection decisions explicit:

1. `availability` declares which measurements exist; `qc_accept` declares external quality-control acceptance. Both must be boolean arrays of the exact data shape.
2. Available, externally accepted nonfinite values raise by default. `nonfinite="mask"` explicitly permits their rejection.
3. Optional bounds, in the **operator's units**, reject out-of-range measurements.
4. Measurement standard deviations must be finite and positive on accepted observations. Representativeness standard deviations must be finite and nonnegative. Scalars or arrays exactly matching the observation shape are supported.
5. Measurement and representativeness errors are assumed independent and their variances are added. All accepted total variances must be finite and positive. This adapter supplies diagonal observation covariance; correlated errors are not silently discarded from an accepted covariance input.

Unavailable or rejected values may contain NaNs and missing errors. Preparation records `unavailable` and `qc_rejected`, replaces masked values with finite placeholders, and passes the actual missingness mask to the native filter. Placeholders are never assimilated as observations. All-missing times still advance the model and apply the native requested inflation.

## Fixed-layout numerical models and uncertainty axes

`prepare_geophysical_assimilation` accepts:

- a native `dynamics.StateLayout` and an explicit source `DiscreteFieldSpace` covering its entire array; or
- a native `equations.DiscreteStateLayout` and the name of one of its existing fields.

Exact field identity is checked, not just shape. Manifold state geometries are rejected: the native Euclidean ensemble transform is not a manifold filter. The callback is `transition(key, state, t0, t1, context)` and returns a same-shape array or a native `TransitionSample`. For a `TransitionSample`, the process identity must equal the declared `model_id`, and validity/status are scalar. The numerical model owns time integration and solver failures. `model_time.unit` is also the callback time unit.

Every run supplies a `GeophysicalEnsembleLineage`, selecting a label from each of five `GeophysicalEnsembleAxis` objects:

| Axis kind | Meaning |
|---|---|
| `initial_condition` | Initial-condition distribution family; native members are draws from this selected prior. |
| `scenario` | External forcing/scenario case. |
| `parameter` | Selected physical parameter case. |
| `structural` | Selected equations/closure/model structure. |
| `internal_stochastic` | Unresolved process-noise family; members and steps have distinct streams. |

Run scenario, parameter and structural selections as separate problems. Do not flatten them into one exchangeable ETKF member axis. The selection declares semantics; the caller must actually supply its corresponding model, prior, forcing and parameters. The adapter does not synthesize these physical alternatives from label strings.

The native filter derives keys by case, step and member. The adapter independently namespaces initial-condition draws and internal stochastic transitions. Changing internal-noise labels does not resample the prior. Reordering or adding available axis labels does not change keys for an existing selected coordinate. Case identity includes scenario/parameter/structural selections; restart compatibility includes the complete selected lineage, time, observation content, native layout, prior content, numerical model and approximation identity. Model IDs and `args` must describe changed callback physics; callable source code is not automatically fingerprinted.

The returned native problem can be used directly:

```python
from phydrax.uq import (
    ensemble_transform_kalman_filter,
    ensemble_kalman_smoother,
)

filtered = ensemble_transform_kalman_filter(
    root_key, problem, ensemble_size=24, inflation=1.0,
    raise_on_failure=True,
)
smoothed = ensemble_kalman_smoother(filtered)
```

The filter uses its existing deterministic, mean-preserving symmetric square-root transform. Anomaly inflation uses the existing native implementation. No perturbed observations or independently implemented gain are added here. No localized-filter capability is claimed: a geographic distance taper cannot simply be inserted into this global member-space transform and still be described as the same mathematically justified solver.

## Physical analysis budgets and restart

`GeophysicalAnalysisInventory` declares a linear inventory on the exact packed native state. Its weights include the model owner's physical quadrature, area/volume, layer mass, capacity and conversion factors; its quantity specifies the resulting unit. For example, temperature-anomaly weights can be column area times areal heat capacity, producing heat anomaly in joules.

`geophysical_analysis_increments(filtered, inventories)` returns forecast inventory, analysis inventory, their memberwise difference, and the ensemble-mean increment at each observation time. These increments are physical analysis impulses. They must be added as analysis sources/sinks to the model's physical ledger, not hidden or described as a time-integration conservation error. No balancing correction is silently applied. Weights do not validate the model's physical derivation; nonlinear inventories must be evaluated directly by the native model owner on forecast and analysis states.

Use the existing `write_ensemble_filter_checkpoint` and `read_ensemble_filter_checkpoint` for pickle-free streaming restart. Keep the complete observation schedule and its identity when reopening a checkpoint. A checkpoint stores the native root key and absolute next step index; restarting a sliced schedule at step zero changes PRNG lineage and is rejected when identities differ. Resume with `ensemble_filter_step(problem, restored_state)`.

## Runnable qualification

From the worktree, with the project environment activated:

```console
PYTHONPATH=. python examples/geophysical_assimilation.py
PYTHONPATH=. python tools/geophysical_assimilation_qualification.py
python -m pytest tests/unit/applications/test_geophysical_observations.py -n auto
```

The example is a genuine linear physical twin: two equal-area temperature-anomaly reservoirs undergo radiative relaxation and conservative inter-column heat exchange. A biased initial ensemble is assimilated against alternating complete/partially missing noisy observations, then forecast for two additional days. The script reports open-loop and analysis errors, covariance contraction, smoothing, joule-valued analysis impulses, physical-budget residual and bitwise native checkpoint restart equality. Its stochastic increments represent unresolved heat forcing and are not claimed to be the exact transition of a continuously forced Ornstein–Uhlenbeck model.

The qualification independently compares the first ensemble analysis against the analytic linear Gaussian posterior calculated from that same finite ensemble's mean and covariance, then checks the physical twin's error, spread, missingness, budget and restart criteria. Tests additionally defend periodic/vertical interpolation and transpose pairing, quantity conversion, explicit QC/error handling, semantic rejections, partial-missing linear Gaussian updates, all-missing inflation and axis-key separation. These are small scientific qualifications, not claims of global weather forecast skill, nonlinear observational retrieval support or production-scale assimilation throughput.
