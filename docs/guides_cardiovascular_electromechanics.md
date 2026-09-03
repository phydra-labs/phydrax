# Cardiovascular active mechanics and electromechanics

PhydraX separates local contraction, the constitutive active-mechanics route, and EP↔mechanics orchestration. This boundary is intentional: evaluating an `ActiveStressPlan` or `ActiveStrainPlan` from a prescribed signal is **active mechanics only**. It is electromechanics only when an `OneWayElectromechanicsPlan` or `BidirectionalElectromechanicsPlan` connects typed electrophysiology and mechanics ports through `phydrax.solver.coupling`.

All kernel values use the cardiovascular kernel scale: time in ms, stress in kPa, voltage in mV, length in mm, mass in mg, and volume in mm³. Cytosolic calcium is supplied in mM by a compatible cardiac ionic model. Fiber stretch and activation are dimensionless. Fiber extension is positive; active shortening is represented by a fiber active stretch below one, while the optional `shortening_velocity` argument is positive during shortening.

## Explicit contraction fidelities

The four contraction fidelities are distinct plan types, not a string-selected mode.

| Fidelity | Plan | Required drive | Internal state |
|---|---|---|---|
| Prescribed tension | `PrescribedTensionContractionPlan` | active tension in kPa | accepted prescribed tension |
| Activation-driven first order | `ActivationDrivenContractionPlan` | dimensionless activation | bounded first-order activation |
| Ca-driven first order | `CalciumDrivenFirstOrderContractionPlan` | live cytosolic Ca in mM | Hill target and first-order activation |
| Land-class length/velocity | `LandLengthVelocityContractionPlan` | live cytosolic Ca, fiber stretch, shortening velocity | Ca binding, distortion, previous stretch, activation |

The first-order routes use the exact update over a constant target in a step,

`a[n+1] = a* + (a[n] - a*) exp(-Δt / τ)`,

with separate activation and relaxation time constants where applicable. The calcium target is a bounded Hill response. The Land-class route additionally applies length-dependent calcium sensitivity and a relaxing velocity distortion. Its purpose is an inspectable Land-class length/velocity response; it does not claim to reproduce an external implementation.

Every route is prepared against a fixed `ContractionState` shape. `candidate` returns `ContractionCandidate` and `ContractionEvidence`; `commit` accepts the candidate only when its finite/bounds/model-compatibility evidence succeeds. A failed candidate rolls back leaf-wise to the previous state. `checkpoint` and `restore` retain the prepared-plan identity and reject a checkpoint from another plan.

### Live-Ca contract

For calcium-driven contraction, prefer `PreparedContraction.candidate_from_reaction`. It requires a concrete `CardiacReactionEvaluation`, consumes `calcium_cytosol_mM`, checks `reaction.model_id` against the contraction plan's `ionic_model_id`, requires the `mM` unit contract, and incorporates `reaction.valid` into fail-closed evidence. A Land plan therefore cannot silently consume an arbitrary activation field or calcium from an incompatible ionic model.

```python
state = cv.mechanics.ContractionState.resting((cell_count,))
land = cv.mechanics.LandLengthVelocityContractionPlan(
    100.0,
    5.0e-4,
    ionic_model_id=ionic_model.model_id,
)
prepared = cv.mechanics.prepare_contraction(land, state)
candidate = prepared.candidate_from_reaction(
    state,
    reaction_evaluation,
    fiber_stretch,
    dt_ms,
    shortening_velocity=positive_shortening_rate_per_ms,
)
state = prepared.commit(candidate)
```

## Active stress and active strain remain separate

`ActiveStressPlan` constructs a symmetric fiber/sheet active Cauchy stress and the corresponding first Piola stress. The reference fiber/sheet frame is orthonormalized once during preparation. Runtime evidence records current-frame norm/orthogonality residuals, symmetry, active power, finiteness, and success.

`ActiveStrainPlan` instead constructs an isochoric active distortion `Fₐ` and returns the elastic deformation `Fₑ = F Fₐ⁻¹`. Fiber shortening is accompanied by equal transverse stretches so `det(Fₐ) = 1`. Evidence records the determinant, frame, minimum-stretch, and multiplicative-reconstruction residuals.

Neither plan advances electrophysiology, transfers a field, or uses the word electromechanics in its claim. Combine its committed response with a passive cardiac material in the mechanics residual. Do not add active stress and active strain simultaneously unless the scientific model deliberately defines that combination.

## Typed EP↔mechanics ports

Three typed port descriptors prevent quantity or direction ambiguity:

- `ActivationEPToMechanicsPort`: dimensionless EP activation output to mechanics activation input.
- `CalciumEPToMechanicsPort`: live cytosolic Ca output to a compatible calcium-driven contraction input; the ionic model ID and unit are part of the contract.
- `StretchMechanicsToEPPort`: dimensionless mechanics fiber-stretch output to an EP stretch input.

Each descriptor binds exact `DiscreteFieldSpace` instances. If source and target use different vector-space identities, an explicit prepared `FieldTransfer` is mandatory. Forward activation/Ca transfers must declare constant and positivity preservation. Stretch transfers must declare constant preservation. Transfer IDs are retained in preparation evidence. There is no nearest-neighbor or same-shape fallback.

This permits, for example, an EP nodal mesh and a distinct mechanics quadrature mesh. Build the interpolation/restriction from the PhydraX discretization substrate and pass it to the typed descriptor. The coupling graph then applies that operator; application code does not duplicate transfer logic.

## One-way multirate coupling

`OneWayElectromechanicsPlan` constructs a directed EP→mechanics `CouplingGraph` with an explicit Gauss–Seidel sweep ordered as electrophysiology then mechanics. `ElectromechanicsCadence(ep_substeps, mechanics_substeps=1)` declares the work per physical coupling window. Participant callbacks receive the exact count and return it as `completed_substeps`; a mismatch fails the participant and causes native coupling rollback.

The callbacks return `ElectricalWindowCandidate` and `MechanicalWindowCandidate`. They own their domain integration over the supplied `CouplingWindow`; the electromechanics layer only adapts those results to `solver.coupling.CouplingSubsystemResult`. Consequently, there is one transactional executor: `solver.coupling.solve_coupling`.

Use one-way coupling when deformation does not affect EP during the simulated interval. The mechanics callback may use either active stress or active strain. The plan's contraction type and forward port must agree: activation ports require activation-driven contraction; Ca ports require Ca-driven first-order or Land-class contraction with the same ionic model ID and calcium unit.

## Bidirectional partitioned coupling

`BidirectionalElectromechanicsPlan` adds the mechanics→EP stretch exchange and constructs an implicit native fixed-point policy over both interface fields. Absolute and relative tolerances are attached to the physical target ports. The native solver records interface residual norms, participant statuses/evaluations, coupling iterations, terminal window status, convergence, and accepted state.

The algorithm is fixed-topology over a prepared run. Differentiation policy is explicit (`none`, `algorithmic`, or a compatible native implicit route); the plan does not differentiate across mesh/topology changes. A failure in a participant, non-finite interface value, exhausted solve, or failed interface certification leaves the accepted `CouplingState` at the previous window boundary.

The cadence controls domain work, while the coupling window controls interface synchronization. Refine these independently:

1. double EP substeps at fixed coupling window and compare the chosen observables;
2. reduce the coupling window and inspect interface-residual and work evidence;
3. refine the field transfer and verify constant/positivity properties and observable convergence;
4. keep topology fixed inside each differentiable prepared epoch.

## Checkpoints and restart

The default rollout retains the full trajectory. For production runs, pass `CouplingRolloutPlan(retention="checkpoints", checkpoint_stride=k)`. The resulting native `CouplingState` is the restart payload: it contains accepted participant states, accepted target exchange fields, physical time, local window index, and stable subsystem/exchange IDs.

```python
run = prepared.solve()
checkpoint = run.solution.final_state
continued = prepared.restart(checkpoint, next_end_time_ms).solve()
```

`restart` validates subsystem and exchange identities, carries only accepted states/fields into a new native `CouplingProblem`, and starts at the checkpoint's physical time. The restarted problem's `window_index` is local to the new rollout; physical time and stable IDs provide global restart identity. Never restart from an uncommitted participant candidate.

## Evidence interpretation

`ElectromechanicsPreparationEvidence` records transfer IDs, forward quantity, cadence, fixed-topology status, bidirectionality, and differentiation policy. `ElectromechanicsEvidence` retains native interface residuals, participant status/evaluation counts, coupling iterations, terminal statuses, cadence ratio, declared participant substep work per window, work-accounting completeness, success, and rollback. Callback `work` is supplied to native per-window diagnostics; retained rollout evidence exposes native participant evaluation counts and declared cadence work as portable work measures.

A successful active-stress or active-strain evaluation is not evidence of electrical coupling. Electromechanics qualification additionally requires the typed port/transfer preparation evidence and a successful native coupling trajectory.

## Cube qualification benchmark

`benchmarks/cardiovascular_electromechanics.py` builds a unit-cube transfer from eight EP corner nodes to a configurable tensor mechanics grid, plus a mechanics-to-EP corner restriction. It executes:

- one-way activation transfer and mechanics response;
- bidirectional activation/stretch fixed-point coupling;
- doubled-cadence refinement;
- checkpoint restart for another physical window.

Run it with:

```console
python benchmarks/cardiovascular_electromechanics.py \
  --points-per-axis 3 --cadence 4 --windows 4
```

The JSON record includes both transfer IDs, one-way and bidirectional success, maximum interface residual, cadence-refinement difference, restart success/time, and wall-clock measurements. Qualification should require success, residual below the declared tolerance, a refinement difference appropriate to the model/discretization, and exact restart time.
