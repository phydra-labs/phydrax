# Geophysical operator learning and verification

Geophysical learning is a host-side binding to `nn.operator`. It does not add a
training engine, time integrator, or an external-model emulation layer. Data,
normalization, execution plans, model weights, portable artifacts and recurrent
control routing remain native operator objects.

## Finite-interval column closure

`ColumnClosureBinding` binds an `OperatorTask` to an ordered tuple of
`GeophysicalQuantity` objects, scalar `state_fields` and `target_fields`, an
ordered `level_ids` tuple, a `vertical_id`, and `interval_seconds`. Levels are in
physical top-to-bottom order. Supply `HybridPressureCoordinate.coordinate_id`
for a hybrid-pressure vertical identity and `layer_mass(surface_pressure)` for
the physical quadrature. A different variable or vertical order is a different
binding, even when all shapes happen to agree.

The native task metadata must explicitly contain:

```python
metadata={
    "geophysical_target_kind": "interval_increment",
    "geophysical_step_seconds": 600.0,
}
```

Each quantity binds one scalar source and one scalar target of the same physical
dimension. Native source/query names come from the task fields. The column query
is a point cloud with one dimension, the ordered level index; its quadrature is
the actual layer mass in kg/m², not uniform level counting. The index is not a
pressure coordinate: add separately named pressure/geometry forcing sources if
the trained closure depends on changing pressure geometry. `vertical_id` must
identify that physical coordinate rather than a numerical grid nickname.

`column_closure_dataset(binding, before, after, resolved_increment, ...)` accepts
arrays in **(case, level, variable)** order and computes exactly:

```text
closure_increment = after - before - resolved_increment
```

The resolved increment is the already integrated contribution of resolved
processes over the same interval. Neither it nor the closure target is an
instantaneous RHS. Arbitrary RHS tasks, a different deployment interval, and
forcing shifted by one interval are rejected. Bounds are explicit (case, 2)
physical seconds; every forcing source must have exactly the same bounds. Use
the geophysical clock to convert archive timestamps before constructing them.
Forcing means, accumulations and instantaneous values must first be prepared
according to their own `TemporalSupport`; this adapter never silently averages,
integrates, or converts one support kind to another.

The dataset retains native case provenance with **scenario, model and member**
identities. `GeophysicalLearningExperiment.prepare` defaults to native grouped
splitting on **one joint simulation identity**, derived from the full
`(scenario, model, member)` tuple. The individual axes are not each held out:
same-model forecasts are valid, and local member labels such as `r1` may repeat
under different scenarios or models. All cases from the same complete simulation
stay together by default, so at least three independent simulations are needed.
The original axes remain in native provenance alongside the derived
`geophysical_simulation` identity.

The evaluation question is explicit. Pass a native `OperatorSplitPolicy` to
choose a different guarantee: `group_by=("model",)` requests held-out models;
`group_by=("scenario",)` requests held-out scenarios. Several native keys protect
each selected key independently; to protect a joint identity, select its one
prepared composite key. Explicit policies are not replaced by a universal
held-out-all-axes rule. All constructions reject shared case IDs and overlap in
the declared group identities, including experiments built from existing splits.

Chronological forecasting within one simulation is also supported. Put the
window/block identity in native case `identities`, its numeric support bounds in
native `order`, then choose the window group and chronological coordinate:

```python
experiment = GeophysicalLearningExperiment.prepare(
    task, dataset,
    policy=OperatorSplitPolicy(group_by=("window",), order_by="start"),
    temporal_bounds=("start", "end"),
)
```

This permits a model and member to occur in all three partitions while keeping
selected windows disjoint. Native grouped chronology checks order-coordinate
ranges; the explicit `temporal_bounds` additionally rejects target windows that
extend across a train/validation/test boundary even if their start times sort.
Adjacent intervals may share an endpoint. Supply the complete input/target window
support, not just an initialization timestamp, when overlap could leak future
information. If `temporal_bounds` is omitted, only the selected native groups and
point-order coordinates are protected; the adapter does not infer an unrecorded
window duration. A completely ungrouped random case split is not accepted.

`experiment.fit(model, steps=..., output_field_map=...)` calls native
`fit_operator` with train-only normalization, physical quadrature weighting and
held-out validation. `steps` specifies optimizer updates: the adapter gives the
native trainer enough epochs to reach that budget instead of silently stopping
after one data pass. It returns the native `OperatorFitResult`, including
`completed_steps`. Native training
checkpoints, custom loss terms, and rollout curricula remain available through
`fit_operator` directly on the prepared split; the host convenience method does
not duplicate those controls. Its cases must be active and equally weighted;
physical layer/area quadrature remains nonuniform. No validation or test case is
used to fit normalizers.

### Deployment and budget admission

`deploy_column_closure` consumes a real `TrainedOperator` with the binding's exact
task fingerprint. It predicts the closure using the beginning-of-interval
sources, and combines it with the explicitly supplied `resolved_increment`.
The caller supplies:

- actual layer mass for the deployment column;
- the required mass-integrated **closure-only** budget for each ordered variable;
- the native physical domain's admission callable on the proposed complete state.

The native `project_operator_conservation` projects each variable onto that
budget with a constant-in-layer correction. This is a transparent, conservative
correction, not a positivity clip. `ColumnClosureAdmission` reports raw and
corrected budgets, budget corrections, the full local correction, quantity IDs
and budget units. For temperature the budget unit is K kg/m²: divide an energy
budget by the appropriate heat capacity explicitly. Coupled enthalpy/water
constraints must be prepared in the physical owner; independently prescribing a
temperature integral is not a substitute for nonlinear moist thermodynamics.

Projection is followed by the supplied domain admission check and a budget
residual check. `require_state()` refuses an inadmissible proposal. A correction
that conserves a budget but produces negative water or invalid temperature is
rejected, never silently clipped or committed. The physics owner remains
responsible for committing the admitted state and any compensating reservoir
exchange; the report exposes the correction needed for that accounting.

## Native autoregressive forecast and climate experiments

The accepted execution object is a native `TrainedOperator`, including one
restored with `load_trained_operator`. Its task declares:

```python
metadata={
    "geophysical_target_kind": "next_state",
    "geophysical_step_seconds": 3600.0,
}
```

The trained task's `problem.rollout_steps` bounds the complete horizon. Every
`role='both'` field must have exactly one native `OperatorRolloutRoute`. The
route's output name is the architecture output (often `output`), while
`task_field` is the physical task field. Every source-only field must have an
explicit `GeophysicalForcingSchedule`, even if constant. Every target has an
ordered physical quantity binding. Fields may contain homogeneous physical
channels, but mixed-unit channels need separate task fields/scales rather than
pretending their units are interchangeable.

`NativeGeophysicalForecast` uses
`autoregressive_operator_rollout_routes` without implementing a second stepper.
Forcing schedules become native `OperatorRolloutControlRoute` objects, evaluated
at the global recurrent step. Schedules contain prepared `FunctionSamples`, a
geophysical `time_id`, and exact numeric clock bounds for every interval. Both
the start and end must match `initialization + step * dt`; an equally long but
shifted forcing interval is invalid. A schedule has one entry per requested
step. It is never extrapolated or recycled.

`GeophysicalForecastRequest` carries the clock, ISO initialization, complete
step horizon, scenario, model, experiment and ordered member identities. The
initial native batch must have `case_axes=('member',)` and matching member count.
`GeophysicalForecastProduct.field(name)` has **(lead, member, native spatial
axes..., native channels...)** order, alongside physical lead seconds and ISO
valid times. These are experiment products, not an implicit claim of forecast
skill or climate equilibration.

A continuation retains the complete native physical batch, next global step,
original random key and original request. Its adapter identity binds model
weights, normalization, task, routes, quantities and the full forcing schedule.
Use `adapter.resume(product.continuation, steps=...)` repeatedly. The initial
time does not reset and random/control steps do not restart at zero. Changing
weights or forcing invalidates continuation. Any recurrent memory/history must
be represented explicitly in native recurrent task fields; hidden mutable model
state is outside this accepted schema. Continuation is an in-memory host product;
there is no extra disk checkpoint format. Native model artifacts remain portable.

The native SFNO supports recurrence on its exact real, spin-zero spherical grid.
It remains an experimental architecture with all-valid grid requirements. The
small example is autonomous scalar diffusion; it does not imply that SFNO's
single selected input automatically consumes other task sources. A conditioned
forecast must use a native architecture that actually consumes all its declared
conditioning inputs. External artifact execution is intentionally not supported
by this adapter: no arbitrary callable, fake native wrapper, framework loader,
or foreign weather-model schema is accepted.

## Physical verification

`geophysical_forecast_metrics` requires explicit arrays:

- truth: **(lead, time, area, layer, variable)**;
- ensemble: **(lead, member, time, area, layer, variable)**;
- mask: the full truth shape, independently marking each variable;
- area weights: (area,); time weights: (time,);
- layer weights: (layer,) or (lead, time, area, layer).

Use physical cell areas, layer masses, and interval durations as appropriate.
Use a singleton layer for surface fields. The metric measure is their product;
no cosine-latitude approximation or implicit equal-level averaging is inserted.
Masked NaN storage is neutralized before ensemble pair differences. Nonfinite
*active* values remain invalid according to native `ml.metrics` status rules.
Empty support is not treated as a zero error.

RMSE and signed bias retain (lead, member); CRPS retains lead. **RMSE and CRPS have
the field's unit, not its square.** The endpoint bias trend is reported per
second when two or more leads are available. There is no default multivariate
average across temperature, wind and humidity. Supply one positive physical
scale per variable to request the dimensionless native energy score; all
variables must then share active event support.

Anomaly correlation uses a supplied `GeophysicalClimatology`. The fixed temporal
climatology fitter uses identified training records and explicit time weights;
verification rejects overlap with those case IDs. Its content/provenance identity
is included in the report. This fixed climatology is not automatically a seasonal
or calendar-day climatology. A seasonal application must prepare the appropriate
reference explicitly rather than mixing calendar phases.

`geophysical_extreme_reliability` reports weighted exceedance Brier score and
reliability-bin probability, observed frequency and mass. Its threshold is in
physical field units, Brier is dimensionless, and empty bins have NaN frequency
and zero mass. It assumes equally weighted members. Reliability is descriptive;
small verification samples do not imply calibrated extreme-event skill.

`geophysical_spectral_rmse` uses native complex-capable `ml.metrics` on explicit,
compatible spectral coefficients and mode weights. Obtain coefficients from the
same native spherical or spatial transform and normalization. Do not apply an
FFT to a flattened sphere. Leads/members precede the last mode axis and remain
unreduced. Spectral normalization and transform provenance are supplied by the
native transform owner, not inferred from array shape.

## Runnable qualification

From the worktree root, using its configured environment:

```sh
PYTHONPATH=. python examples/geophysical_operator_forecast.py --steps 40
PYTHONPATH=. python tools/geophysical_learning_qualification.py --steps 40
PYTHONPATH=. python -m pytest -n auto tests/unit/applications/test_geophysical_learning.py
```

The example trains a small **native SFNO** on exact finite-interval l=1 spherical
heat-equation solutions, saves/restores its real portable artifact, produces an
autoregressive forecast, compares restarted and uninterrupted recurrence, and
computes weighted RMSE/CRPS in kelvin. It also trains a native DeepONet column
mixing closure, deploys it with an explicit zero closure heat budget, and reports
physical admission and the actual budget correction. The qualification runner
requires both training objectives to decrease and checks restart/budget errors;
it prints measured runtime and physical diagnostics. It is deliberately not a
benchmark against operational weather models, a moist-column skill claim, or a
long-horizon climate stability qualification.
