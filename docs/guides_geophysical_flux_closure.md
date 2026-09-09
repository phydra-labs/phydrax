# Conservative learned column interface fluxes

The conservative flux surface is an alternative to arbitrary finite-interval
state-increment prediction. It reuses native `OperatorTask`, `OperatorDataset`,
`GeophysicalLearningExperiment`, operator architectures, normalization and
trained-operator artifacts. There is no second optimizer, rollout engine,
checkpoint format, conservation projection, or hidden recurrent state.

## Physical contract

`column_flux_tasks(training_id, thermodynamics, ...)` produces two native scalar
tasks. Keeping them separate retains their different physical dimensions:

| Output field | SI rate | Meaning |
|---|---|---|
| `vapor_mass_flux` | kg m^-2 s^-1 | Vapor-phase water moving downward |
| `total_energy_flux` | W m^-2 | Total transported energy, already including transported water energy |

Both tasks bind the same training identity and `MoistThermodynamicPlan.plan_id`,
positive-downward interface sign, top-to-bottom layer ordering, and closed zero
external fluxes. These are interval-mean rates, not an instantaneous learned ODE
right-hand side. The names, units, reference, phase and boundary semantics are
part of the task fingerprint and survive native artifact reload.

There are exactly `layers - 1` learned interfaces. Append zero at each boundary.
For either flux F, layer j receives the extensive increment

```
increment[j] = dt * (F[j] - F[j + 1])
F[0] = F[layers] = 0
```

Each interior transfer appears once positively and once negatively. Global
cancellation is structural, up to floating-point summation, not a posteriori
budget repair. `ColumnFluxAdmission.inventory_residual` exposes that roundoff.
There is no last-layer residual deposition, mass redistribution, or clipping.

The closure moves **vapor only**. Cloud liquid, cloud ice, rain and snow cannot
supply its donor inventory. `admit_column_flux` checks outgoing vapor before
inflow: a layer cannot fund simultaneous outflow from an incoming transfer.
All condensate still contributes to caloric temperature admission. A future
all-species transport model requires a different explicit task contract; it
cannot be relabeled as this vapor closure.

## Measures, interval and runtime binding

`column_flux_space(dry_mass, support_id)` prepares a native scalar cell-average
field space and physical dry-mass measure in kg/m². `ColumnFluxBinding` holds
those exact identities, the closure interval in seconds, resolution in meters,
regime and forcing identities, and both native tasks. Nonuniform positive layer
masses are supported. Alternate units are rejected rather than guessed.

The native input fields are vapor mass per dry mass, total energy per dry mass,
interval seconds, resolution meters, and an explicitly supplied dimensionless
forcing control. Samples carry the actual physical dry-mass quadrature. Query
coordinates are ordered interior dry-mass fractions; their quadrature is unit
horizontal interface area. Query quadrature is not a layer-inventory measure.

`binding.batch(vapor_mass, total_energy, forcing=...)` accepts `(case, layer)`
inventories. It and deployment verify runtime coordinates, physical measures,
source support, interior query geometry and interval/resolution conditioning.
`deploy_column_flux((water_artifact, energy_artifact), binding, batch, ...)`
requires the expected artifact IDs, both matching native task fingerprints, and
the actual thermodynamic plan and dry masses. Hidden output-projection pipelines
and external model adapters are not admitted.

The input batch describes the interval's conditional initial state. The
inventories supplied for admission are the resolved coarse endpoint to which
the unresolved correction is applied. This is an explicit finite-interval
splitting contract, not a claim that a frozen learned flux is valid under any
arbitrary ordering of nonlinear column physics.

Call `admission.require_inventories()` before committing a batch. Failed
proposals remain inspectable but are never silently turned into clipped states.
For online physical stepping, pass admitted `water_flux` and `energy_flux` rates
to `InteractiveMoistColumnPlan.step`. Its optional hooks take scalar zero or
`(layers - 1,)` rates with the same sign and units. Native column admission
checks the complete process sequence and rejects atomically. Include cloud
liquid plus rain as `liquid_mass`, and cloud ice plus snow as `ice_mass`, when
using the standalone admission function.

Physical continuation remains the native column state/checkpoint. The stateless
closure does not need another recurrent checkpoint. Reload both native operator
artifacts and retain the expected artifact IDs and binding at resume; changing
a training task, thermodynamic reference or expected artifact identity is an
error. Applications requiring history must route it through explicit native
continuation instead of a universal hidden recurrent vector.

## Conservative conditional labels

`ConservativeColumnTransfer` wraps a prepared native `FieldTransfer` and its
exact source/target `DiscreteMeasure` values. Both fields must be scalar
cell-average spaces with matching support and entity identities. The transfer
must declare conservation, positivity and constant preservation. A native
transposed operator action verifies the physical measure pairing without
building a dense matrix. Constant preservation is checked directly; negative
transferred vapor is rejected.

Extensive inventories are divided by source dry masses, transferred as specific
inventories and multiplied by target dry masses. This is deliberately not an
arithmetic average of layer masses or energies.

`conditional_column_flux_target` requires:

- the coarse initial state equals the conservatively restricted fine initial state;
- both endpoints describe the same declared interval and physical forcing;
- the transfer target and thermodynamic reference agree with the runtime binding;
- the resulting conditional correction has zero external inventory to numerical precision.

The combined conditional increment is restricted fine endpoint minus resolved
coarse endpoint. A nonzero external inventory discrepancy cannot be represented
by this closed-interface closure and is rejected, not projected away. The
unique interior flux follows from the negative prefix sum divided by dt.

When both fine and coarse reference endpoints are available, the target keeps
three distinct arrays:

1. `combined_increment`: discrete fine-minus-coarse endpoint discrepancy;
2. `unresolved_increment`: reference fine-minus-coarse endpoint discrepancy;
3. `numerical_increment`: combined minus unresolved.

Training uses the unresolved part in this case. Otherwise provenance explicitly
says `combined_unseparated`; no separation is invented. Reference endpoints must
have the same initial state, forcing and interval, and their source is identified
by the caller's `reference_id`. The example uses refined temporal integration
at each spatial resolution. It does **not** eliminate spatial truncation error
or establish a continuum/real-atmosphere reference.

`column_flux_datasets` carries resolution, closure interval, forcing, regime,
transfer/binding/reference identities and interval bounds in native case
provenance. Native split policies can select `resolution`, `closure_interval`,
`forcing` or simulation identities. Evaluation on independently prepared native
datasets is also supported, including genuinely different numbers of layers.
Merely supplying dt or dx to a model is not evidence of transfer skill.

## Runnable scientific qualification

From the repository root:

```sh
PYTHONPATH=. python examples/geophysical_flux_closure.py --steps 200
PYTHONPATH=. python tools/geophysical_flux_closure_qualification.py --steps 200
```

The example evolves the existing native `conservative_vertical_mixing` law on
fine and coarse vertical grids, with paired vapor/total-energy transport and
positive explicit transport timesteps. The coarse resolved model omits 75% of
the diffusivity. Actual native integral-branch DeepONets learn the missing
interface rates using `GeophysicalLearningExperiment`; integral branches allow
variable sensor counts rather than hiding a fixed flattened grid.

The example saves and reloads native artifacts, then evaluates independent
members, an eight-layer withheld grid (training has six layers), a withheld
75-second interval (training has 60 seconds), a withheld diffusivity forcing,
and a joint holdout. No held-out data fits normalization or optimizer state.
It compares trained operators, original random weights with the same train-only
normalization, and the no-closure resolved model. The example intentionally does
not use dt/dx conditioning as evidence for generalization: it measures errors
on separately evolved supports.

The JSON report includes mass-weighted vapor/energy RMSE, skill relative to no
closure, the separately measured temporal numerical discrepancy, rejected-case
rates, zero correction rates, inventory residuals, task support and simulation
case identities, training losses, completed updates, native artifact reload
error, and native physical column checkpoint/resume error with learned fluxes.
Rejected cases are counted and retain the resolved baseline when reporting
consumer-visible RMSE. The qualification fails if either trained channel does
not improve on both baselines on each declared evaluation set, if trained
proposals reject, or if inventory/artifact/holdout contracts fail.

`native_artifact_roundtrip` reports each named channel separately in its SI unit:
maximum absolute error, relative and machine-epsilon-scaled L-infinity errors,
and the maximum pointwise native-allclose tolerance ratio. Training uses compiled
execution while native artifact reload defaults to eager execution; both actual
execution strategies are reported, not silently forced to agree. Task, weight
and normalization identities must agree exactly. Original and restored weights
are additionally compared under the same eager execution route and must produce
identical predictions. Only the compiled-versus-eager comparison uses the native
artifact regression convention (`rtol=1e-5`, `atol=1e-8` in each channel's SI
unit). Material differences fail with the channel's full diagnostics. Physical
column checkpoint continuation retains its separate exact-equality requirement.

The physical checkpoint comparison explicitly sets `mixing_length=0` and zero
rain/snow fall speeds to isolate learned vapor transport. Otherwise the native
convective closure activates even with zero background diffusivity, duplicating
transport and imposing an unrelated turbulent CFL. Learned rates and their
physical interval are never reduced to make the continuation pass.

This is a controlled closed-column diffusion qualification. It is not evidence
of turbulence universality, a moist convection closure, cloud/precipitation
transport skill, stable long-range climate feedbacks, or real-Earth forecast
skill. Those claims require additional real process and observational evidence.
