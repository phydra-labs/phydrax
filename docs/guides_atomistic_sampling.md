# Enhanced atomistic sampling

Collective variables, biases, adaptive histories, thermodynamic states, multistate
assignments and free-energy analysis are separate typed layers. Physical coordinates
never silently change thermodynamic labels, bias histories or reduced-potential
conventions. The [sampling cookbook](cookbook/atomistic_sampling.md) runs the complete
path end to end.

## Collective variables, metric and geometry

`CollectiveVariablePlan` declares a coordinate domain, stable indices, parameters, a
metric and an image branch. Built-in variables include:

- distance, angle and torsion;
- center-of-mass separation and radius of gyration;
- coordination and native-contact similarity;
- aligned RMSD;
- volume and density;
- path coordinates.

Every evaluation consumes one `CollectiveVariableGeometry`, which carries:

- the wrapped and accepted unwrapped (material) coordinates;
- the current, possibly strained, cell vectors;
- bound neighborhood views.

The evaluation reports `successful`, `regular`, a `CollectiveVariableStatus` word and
branch margins.

Point-pair vectors use the cell's certified minimum image. Groups use one of these
`CollectiveVariableImageBranch` options:

| Branch | How a group is placed |
|---|---|
| `UNWRAPPED` | The accepted unwrapped branch. |
| `TOPOLOGY` | A bond spanning-tree reconstruction with checked cycle closures. |
| `ANCHORED` | Minimum images from an anchor, under a declared extent certificate. |

An ambiguous image or an uncertified branch is an explicit failure. It is never
silently resolved by rounding. Aligned RMSD uses a native spectral alignment and retains
its eigengap. Zero-length, collinear and zero-distance configurations keep a finite
value but report that their first derivative does not exist.

Bias forms use the metric displacement, not raw subtraction. A periodic umbrella takes
the nearest-image displacement and reports its seam margin. A periodic wall or flat
bottom must declare its branch, or it refuses. `HillKernel.WRAPPED` is an explicitly
truncated image sum with a reported tail bound. It is not an exactly periodic kernel.

## Biases are Hamiltonian terms

A bias is an `AtomisticBiasTerm` inside the same `AtomisticPotentialProgram` as the
physical terms. Every force kick, constraint projection, force rebase, Monte Carlo cell
trial, stress evaluation and reduced-potential cross evaluation therefore sees the same
definition. The term is placed in its own force group, so estimators can request
physical forces explicitly.

The available forms are:

| Form | Bias |
|---|---|
| `HarmonicBiasForm` | Scalar, diagonal or full positive-semidefinite stiffness with native spectral evidence. |
| `FlatBottomBiasForm` | Flat bottom on a declared branch. |
| `WallBiasForm` | One-sided wall on a declared branch. |
| `HillHistoryBiasForm` | Metadynamics or configuration-kernel history. |
| `IntegratedProfileBiasForm` | Frozen ABF profile, used as `V_bias = -A`. |
| `LearnedFreeEnergyBiasForm` | Committee-gated learned bias. |

Every form has one evaluation contract,
`form.evaluate(values, metrics, controls, history, owner_key) -> BiasFormEvaluation`,
which carries `energy`, `successful`, the form's own `status` word and, for aligned
configuration kernels, `minimum_relative_gap` (`None` otherwise). A history plan's sole
evaluation is `plan.evaluate(values, metrics, state, owner_key) ->
AtomisticBiasHistoryEvaluation` (`energy`, `successful`, `status`, `active_kernels`,
`minimum_relative_gap`). Scalar consumers read `.energy`; there is no separate
energy-only form or history method. The term's Hamiltonian contribution remains
`PreparedAtomisticBiasTerm.energy(context)`, and `bias(geometry, state)` returns
`(energy, CV evaluation, successful)` including the form's admission.

Cell-dependent biases share the strain map of the program. Their stress is the
tension-positive strain stress `sym(dE/dstrain) / V`, taken at fixed fractional
coordinates and integer images. For example, a volume umbrella `k (V - V0)^2 / 2` on
`CollectiveVariableKind.CELL_VOLUME` contributes `k (V - V0) I`. That is a term stress,
not a pressure: the instantaneous internal pressure also includes the kinetic momentum
flux and the stress of every other term.

Controls (umbrella centers), history views, the history owner key and an energy coupling
are read-only `AtomisticBiasTermState` values, bound to the term's declared slot `name` and
authenticated by its exact `term_id` (dynamic identity words). They change
only at committed boundaries, through `with_bias_controls` and `with_bias_history`. Each
change advances the revision and invalidates force caches. A moving restraint
(`center_rate`) books protocol work. A changing history books history work. Both are
evaluated at the committed configuration, under a fixed boundary-state convention.

`AtomisticDynamicsPlan(..., force_ports=...)` admits typed nonconservative force ports.
A runtime with ports reports `conservative_force_definition=False` and refuses every
energy-based consumer:

- `AtomisticCanonicalSamplingQualification`, so no multistate runtime or analysis exists
  for it, even with one state;
- multi-state thermodynamic tables (cross-evaluated reduced potentials);
- Monte Carlo volume moves;
- integrator-owned cell methods;
- RESPA force-group splitting;
- switching work.

## Adaptive histories

`AtomisticBiasHistoryPlan` declares one bounded kernel history. Its settings are:

| Setting | Meaning |
|---|---|
| `interval` | Cadence in committed steps or iterations. |
| `ramp_depositions` | Ramp of new kernels to full height. |
| `bias_factor` and `temperature` | Well-tempered damping `k_B T (gamma - 1)`, at the history owner's temperature. |
| `ownership` | `SHARED`, `WALKER` (lane) or `STATE` (row `owner_keys`). |
| `owner_count` | Number of owner keys. |
| `storage` | Storage policy (see below). |
| `kernel` | Kernel type (see below). |

Its arrays are fixed-capacity dynamic state. Force evaluations hold the history fixed.
Kernels are deposited only at declared committed boundaries, never during a rejected
proposal, cross evaluation, force prime or rebase.

A simultaneous multi-walker event works as follows:

- Every height is computed against the pre-event history.
- Kernels are inserted in a stable walker order.
- The event is all-or-nothing.

Storage policies are explicit:

- `TERMINAL` refuses the event that would overflow. In `PreparedBiasedDynamics` this
  refusal is latched: the step carrying `BiasedStepReason.HISTORY_EXHAUSTED`, and every
  later step, commits nothing (no step, RNG draw, deposit or counter). This lasts until
  the host transaction `grow_bias_history_capacity` (or `grow_bias_history`) copies
  kernels, owners and counters into a larger buffer. The scientific identity is kept;
  only the capacity identity changes. Nothing grows inside a compiled loop.
- `FIFO` overwrites the oldest kernels. Once a kernel has been overwritten, the usual
  metadynamics profile formula no longer applies, and `metadynamics_profile` reports
  that refusal.

Kernel types:

| Kernel | Behavior |
|---|---|
| `NEAREST_IMAGE` | Keeps the source seam behavior. |
| `WRAPPED` | Truncated image sum with a reported tail bound. |
| `ALIGNED_MSD` | Fixed-correspondence configuration kernel. Each active kernel (valid, selected by the owner key and with a positive ramp weight) is aligned once; that alignment supplies its value and its admission. An active kernel whose optimal superposition is degenerate (no simple top key-matrix eigenvalue) or nonfinite fails the bias evaluation, and `AtomisticBiasEvidence` reports the kernels' `form_status` and smallest `alignment_relative_gap`. An exact zero MSD is regular for this kernel. Inactive, padding and zero-ramp slots never enter the admission. It has no CV-grid profile and is analyzed by frozen-history reweighting. |

Resume and warm start are distinct:

- **Resume** restores the exact checkpointed state, through
  `write_biased_dynamics_checkpoint(path, runtime, state)` and
  `read_biased_dynamics_checkpoint(path, runtime, template)` (dynamics, term states with
  histories, ABF bins and controls, and the work ledgers, authenticated against the
  runtime's identities), or through the multistate checkpoint.
- **Warm start** (`warm_start_bias_history(plan, frozen)`) declares a new run's initial
  condition from a frozen history of the same scientific definition, with ramps
  completed and counters restarted. It is not a continuation.

`PreparedBiasedDynamics(dynamics, thermodynamic, *, physical_force_groups,
walker_index=0, adaptive=True, barostat=None, barostat_interval=None)` composes the
native dynamics runtime and adds no integrator. It discovers the bias terms in the
dynamics program and the ABF ports in its force ports. `physical_force_groups` must
exclude every bias group. Together with the bias groups, they must cover the program.
One committed iteration runs in this order:

1. Propagate with `H_n` (term states fixed).
2. Run an eligible Monte Carlo cell move, still with `H_n`.
3. Run committed-boundary events: ABF samples from the physical groups, due hills
   (heights against the pre-event history) and moving centers.
4. Commit the event's term states (ABF counts and counters, deposited kernels, moving
   centers). If the applied force changed (a term revision advanced), re-prime the
   forces at the fixed configuration; otherwise the force cache stays valid. The
   energy jump is split into `history_work` and `protocol_work`, both evaluated at the
   committed configuration.
5. Publish.

`step` returns a `BiasedStepEvaluation` with `accepted_state`, `successful`,
`BiasedStepReason` flags in `reasons`, `deposited` and the work increments. Any failure
rejects the whole iteration, and nothing is deposited on a rejected, probe, rebase or
prime call. `adaptive=False` freezes every history and control. `freeze(state)` returns
authenticated `FrozenBiasHistory` values keyed by the history terms' slot names.

## ABF: force-only and integrated

`AtomisticABFPlan(variables, /, *, minimum, maximum, bins, temperature, min_samples=200,
full_samples=None, max_force=None, measure=ABFBaseMeasure.CONFIGURATIONAL,
maximum_closure_atoms=64)` declares one estimator:

- half-open bins `[minimum, maximum)` per CV; a periodic CV is first mapped to the
  branch `[minimum, minimum + period)`. Preparation precomputes each axis's
  `bins + 1` uniform edges once in the system coordinate dtype
  (`PreparedABF.bin_edges`) and refuses bounds that exceed that precision or edges
  that collapse in it. A sample on an exact interior edge belongs to the bin on its
  right; `maximum` itself is out of range, as is anything below `minimum`. An
  out-of-range sample is counted in `out_of_range`, never recorded in an edge bin, and
  nothing is applied outside the range;
- the per-bin ramp, zero below `min_samples` and full at `full_samples` (default
  `2 * min_samples`);
- an optional cap `max_force` on every applied `dA/dxi` component;
- the base measure `ABFBaseMeasure`, which must match the dynamics: constraints
  touching the CV closure require `CONSTRAINED_DYNAMICS`, none require
  `CONFIGURATIONAL`;
- `temperature`, which must equal every thermostat temperature of the table that
  `PreparedBiasedDynamics` runs.

ABF CVs are explicit-index CVs (no neighborhood requirements), because they are
evaluated at force-port boundaries where bundle views are not bound.

The estimator is the conditional mean force of the thermodynamic-integration ABF
formulation (Darve and Pohorille, *J. Chem. Phys.* 115, 9169 (2001); Ciccotti, Kapral
and Vanden-Eijnden, *ChemPhysChem* 6, 1809 (2005)). With `J = [J_xi; J_sigma]` the
Jacobian of the CVs and of the closure's holonomic constraints, the dual fields
`B = J^T (J J^T)^{-1} E` satisfy `J_xi b_a = e_a` and `J_sigma b_a = 0`, and

$$
\frac{\partial A}{\partial \xi_a}
= \Big\langle \nabla U \cdot b_a - k_B T\, \nabla \cdot b_a
- k_B T\, b_a \cdot \nabla \log w \;\Big|\; \xi \Big\rangle ,
$$

where `w` is the density of the declared `ABFBaseMeasure` on the closure coordinates
(`ABFBaseMeasure.density`). `CONFIGURATIONAL` is Lebesgue measure (`w = 1`).
`CONSTRAINED_DYNAMICS` is the configurational marginal of the canonical measure on the
constrained phase space, which RATTLE with a constraint-projected thermostat samples:
integrating the momenta tangent to the constraints gives
`delta(sigma) sqrt(det G_sigma) exp(-U / k_B T) dq` with
`G_sigma = J_sigma M^{-1} J_sigma^T`, so `w = det(G_sigma)^{1/2}` and the Fixman term is
`-(k_B T / 2) b_a . grad log det G_sigma`. A constrained plan binds this density in its
`abf_id`, so a history accumulated under another density never resumes under it. For a
pair distance in three dimensions the estimator reduces to
`-(F_j - F_i) . u / 2 - 2 k_B T / r`, so the radial reference of a two-atom system is
`A(r) = U(r) - 2 k_B T log r`.

The implementation evaluates one prepared coarea linearization of the dual fields on
the closure (CV support plus every atom of a touching constraint component). The fields
come from one native multi-right-hand-side solve of the Gram system `J J^T` (no explicit
inverse). The divergence is a stream of exact diagonal JVPs of that linearization, with
no Hessian of `U`. The Fixman term uses one prepared solve of the constraint metric
for the directional log-determinant actions, with no explicit inverse.

`PreparedABF.sample(geometry, physical_forces, /, *, maximum_condition=1e10)` must receive
forces of the physical Hamiltonian only; `PreparedBiasedDynamics` takes them from the
explicitly requested physical force groups, never from total forces that include a
bias. The returned `ABFSample` carries `values`, `bin_index`, `in_range`, `regular`,
`mean_force`, `gram_condition`, `status`, and the solve evidence:

- `dual_solve`: an `ABFLinearSolveEvidence` for the Gram solve;
- `fixman_solve`: an `ABFLinearSolveEvidence` for the metric solve, or `None` for an
  unconstrained closure.

`ABFLinearSolveEvidence` holds the native per-right-hand-side `status` and
`derivative_valid`, plus the solver `diagnostics` and `provenance`; its `successful`
property requires every right-hand side to succeed with a valid derivative. A sample is
`regular` only if the CV evaluation is successful and regular, every solve is
successful, the Gram condition is below `maximum_condition` and the mean force is
finite. An irregular sample (for example dependent CVs) has `mean_force` NaN, is
counted in `irregular` when it is in range (otherwise in `out_of_range`), and is never
recorded as a zero force.

`PreparedABF.accumulate` records an in-range regular sample. `mean_force(state)` is NaN
in empty bins. `applied_gradient(state)` is the ramped and capped estimate, zero where
nothing applies. The state `revision` advances only when the applied force of the
sampled bin changes; counts, sums and the `out_of_range`/`irregular` counters are
committed at every successful step regardless, so samples below the ramp threshold
accumulate.

ABF has two realizations:

- **Force-only.** `AtomisticABFForcePort(abf, name=...)` applies
  `F = sum_a g_a grad xi_a` in the current bin, where `g` is `applied_gradient`. This is
  the sign of `V_bias = -A` locally: the bias force is `+dA/dxi grad xi`. It is
  evaluated as one pullback of `g` through the CV map on the closure coordinates; no
  CV-by-coordinate Jacobian is formed. A CV program whose values have another dtype than
  the coordinates pulls back one exact output-basis row at a time and accumulates
  `g_a grad xi_a` in the coordinate dtype, so `g` is never narrowed. The port is a
  typed nonconservative force port with no energy. Its runtime refuses multistate
  tables, Monte Carlo volume moves, integrator-owned cell methods, RESPA force-group
  splitting and switching work.
- **Integrated.** `abf_profile(abf, state, /, *, minimum_samples=None)` integrates the
  mean-force field of a fixed history over its largest connected supported component,
  with gauge `A = 0` at the first node, and reports support, an integrability residual
  and `periodic_axes` (axes whose bins close a CV period; their last and first bins are
  neighbours through the seam). The least-squares gradient problem is solved by one
  native matrix-free PCG solve of the grid-graph normal equations that stops only on
  the true residual: the finite-difference incidence is a masked shift action, so
  storage stays proportional to the grid and no dense incidence or normal matrix is
  formed. `integration_solve`
  retains the native status, derivative validity, diagnostics (iterations, residual,
  convergence) and provenance; a failed status, including non-convergence within the
  declared step budget, refuses the profile even if its values are finite. It is
  `None` when no solve is needed (an isolated supported node or a metadynamics
  profile).
  `IntegratedProfileBiasForm(abf_profile(...))` turns an admitted profile
  into the conservative term `V_bias = -A`, with cubic Hermite interpolation for one CV
  and native multilinear interpolation for several. A periodic axis must close its
  period and be supported in every bin; queries are reduced to the period and the
  wrapping interval is interpolated through the seam, so energy and force are
  continuous across it. A periodic CV whose profile covers only part of its period, or
  has unsupported bins, is refused rather than clamped at the seam. The form also
  refuses a one-CV profile with fewer than two supported nodes and a multi-CV profile
  without full-grid support. On a nonperiodic axis the energy is constant outside the
  node range. It is conservative because the profile is fixed, and it takes part in
  ordinary energy workflows.

`metadynamics_profile` and `abf_profile` return diagnostic `AdaptiveProfileResult`
values. These carry assumptions (base measure, storage, ownership, temperature,
`adaptive-history:not-equilibrium-samples` or `adaptive-history:convergence-not-certified`)
and refusals (`formula_admitted` is false when any refusal applies). They are never
equilibrium `FreeEnergyResult` values: an adaptive history samples a time-dependent
Hamiltonian, so the profile is not equilibrium evidence.

Validation scope: the pair-distance radial reference `A(r) = U(r) - 2 kT log r` is the
declared independent control for the sign and geometric correction. The constrained
measure is controlled independently by three equal masses with two rigid unit bonds
and no energy, whose canonical end-to-end marginal is
`p(r) ~ r sqrt(4 - cos^2 theta)` with `cos theta = 1 - r^2 / 2`. Phase-probe
means of the estimator matched on four- and eight-atom systems with two CVs. No
convergence rate, accuracy on larger systems or speed is claimed.

## Phase-space measure and thermodynamic states

`AtomisticPhaseSpaceMeasurePlan` identifies the common elements of one sampling target:

- system, topology and particle support;
- masses and coordinate map;
- constraints, units and cell convention.

Every state in one compiled table must share that measure.

`AtomisticThermodynamicStatePlan` declares NVE, NVT, NPT or NPH intensives, an ordered
Hamiltonian-control vector and an optional `AtomisticBiasRowBinding`. The binding
carries each bias term's row controls, history `owner_keys`, energy `couplings` and
history declarations.

A row's `bias_id` fingerprints all of this and enters its potential ID. Every row must
bind the same bias terms. A multistate table must bind every stateful bias term.
Coupling `0` keeps a term in the program but realizes the unbiased row of the same
Hamiltonian.

`PreparedThermodynamicStateTable` lowers the rows to arrays of inverse temperature,
temperature, pressure, controls and masks. Dynamic assignments select rows with JAX
gathers. BAOAB and the isotropic Monte Carlo barostat obtain temperature and pressure
from this table.

`AtomisticCanonicalSamplingQualification` binds each multistate run to exact
target-distribution evidence or a finite declared sampling-bias bound. Approximate
kernels may produce diagnostic estimates, but they never receive successful equilibrium
status. It refuses runtimes with nonconservative force ports, and records
`fixed_row_hamiltonians` (no row declares an adaptive history); distribution consumers
call `require_fixed_row_hamiltonians(consumer)` rather than issuing their own
certificate.

## Bias-aware multistate execution

`AtomisticMultistatePlan` binds one prepared dynamics runtime, a table, stable replica
IDs and at most one assignment policy:

- **`AtomisticReplicaExchangePlan`** runs alternating disjoint neighbor exchanges with
  one replica per state. Exchange across bias rows is window exchange.
- **`AtomisticSAMSPlan`** allows distinct replica and state counts.
- **No policy** keeps fixed labels. Unoccupied rows, such as an unbiased target, are
  then still cross-evaluated.

Every history term must be bound by the rows, with one common declaration in every row:

- `"adaptive"`: a live history that deposits at committed iterations.
- `"frozen:<digest>"`: a fixed Hamiltonian, whose history never changes.

History owners resolve per term (`PreparedAtomisticMultistate.bias_owner_resolutions`):
a shared history has one owner; a state-owned or frozen walker-owned history evaluates
the owner key its row declares, so every row, including an unbiased target, is one fixed
Hamiltonian under complete cross evaluation (analyze a frozen multi-walker history with
one row per state and owner); an adaptive walker-owned history evaluates each replica
lane's own kernels, so its rows depend on the sample column and are never analysis rows.

The following are refused:

- an unbound history term (even when there is only one state);
- mixed declarations across rows;
- a declaration on a term that has no history;
- a well-tempered history whose rows do not all sit at its owner temperature;
- a state-owned or frozen walker-owned history whose row `owner_keys` reach
  `owner_count`; an adaptive walker-owned history with fewer owners than replicas, or
  whose rows declare a nonzero owner key.

One atomic iteration runs in this order:

1. Propagate with `H_n`.
2. Run an eligible cell proposal with `H_n`.
3. Evaluate the complete reduced-potential matrix, including each row's actual bias at
   each configuration and cell.
4. Decide the assignment move.
5. Rebase every lane under its new row (controls, declared owner and coupling).
6. Deposit adaptive kernels, after the assignment move. A state-owned kernel is tagged
   with the row owner key (`AtomisticBiasRowBinding.owner_keys`) of the label selected
   by `AtomisticMultistatePlan(..., history_label=...)`: `BiasHistoryLabel.ORIGIN`
   (default, the row whose Hamiltonian sampled the configuration) or `TARGET` (the
   post-move row).
7. Refresh forces if a history changed.
8. Commit.

Valid rejected proposals are successful no-ops. An invalid propagation, required cross
evaluation, rebase or deposit, or a terminal capacity refusal, rolls the whole iteration
back without consuming counters.

Combined or walker-dependent Hamiltonian moves have no derived detailed-balance
acceptance and are not admitted. Asynchronous exchange is not provided.

## Adaptive versus equilibrium samples

Draws taken while SAMS weights adapt, or while any history is declared `"adaptive"`, are
recorded but are never `sample_active`. Adaptive evidence is never equilibrium
evidence: the sampled Hamiltonian changes along the run, and an
`AtomisticCanonicalSamplingQualification` qualifies the kernel of a fixed row, not an
advancing history or adapting SAMS weights.

`AtomisticMultistateEquilibriumSamples(result)` is the single admission of segment draws
for every reduced-potential consumer, for any state count. It refuses unsuccessful
segments, active draws marked `sams_adapting` or `bias_adapting`, frozen declarations
that do not authenticate their content (`bias_history_digests()`), and any active draw
whose evaluated history owner differs from its row's declared owner or comes from a
lane-owned history (`authenticate_bias_owners()`, also run on checkpoint write/read).
`reduced_potential_dataset_from_multistate` builds the MBAR dataset from it. No
time-dependent reweighting formula is applied to adaptive draws. To analyze them as
equilibrium samples, freeze the history first.

## Frozen-history authentication and per-draw evidence

`FrozenBiasHistory(plan, state)` fingerprints every kernel array and counter. Its
`declaration` is `"frozen:<digest>"`. Labels never authenticate content. The content is
checked against the declaration at every stage:

- At initialization, every lane's bound history must reproduce the digest, and every
  lane must bind identical content.
- `AtomisticMultistateSegmentResult.bias_history_digests()` checks the end-of-segment
  content, the per-draw revisions and the deposits:
  - Frozen: the revision is constant and nothing is deposited.
  - Adaptive: the last committed revision equals the bound `written` counter.
- The multistate checkpoint manifest stores the declarations and digests in its
  payload identity, and recomputes them on read.
- UQ conversion repeats the segment authentication.

Each committed iteration records these arrays in the segment result (padding is
`False`/`-1`):

| Field | Shape | Meaning |
|---|---|---|
| `bias_adapting` | (capacity, R) | Draw was taken under an adaptive history. |
| `bias_deposited` | (capacity, H) | History received this iteration's event. |
| `bias_exhausted` | (capacity,) | Attempted iteration was refused by terminal capacity; it ends the segment. |
| `bias_owner_keys` | (capacity, R, B) | History owner each replica's bias terms evaluated. |
| `bias_history_revision` | (capacity, H) | History `written` counter after the draw. |

The static fields `bias_term_slots` and `bias_history_term_slots` (declared term names)
name the axes. The host properties `bias_term_ids` and `bias_history_term_ids` decode the
exact term digests, and `bias_history_declarations` decodes the `"adaptive"` or
`"frozen:<digest>"` declarations.

## Freeze-to-analysis

A standard freeze-to-analysis workflow runs as follows:

1. Freeze the adaptive history with `PreparedBiasedDynamics.freeze` or
   `FrozenBiasHistory`.
2. Declare `frozen.declaration` in every analysis row.
3. Add an unbiased target row: the same terms at coupling `0`.
4. Run fixed-Hamiltonian segments.
5. Convert with `reduced_potential_dataset_from_multistate` and estimate with
   `multistate_bennett_acceptance_ratio`. MBAR estimates the unsampled target row
   through cross evaluation.

No second estimator or self-issued equilibrium certificate is introduced.

## Switching and Hamiltonian transitions

`AlchemicalSwitchingPlan` requires both endpoints to share one exact row bias identity.
It refuses adaptive histories and nonconservative force ports. The execution
authenticates every frozen history on all initial and final states.

The resulting `AlchemicalSwitchingRecord` carries the following, taken from the table
rows rather than from caller labels:

- `bias_ids`;
- `bias_history_digests`.

`reduced_work_dataset_from_alchemical_switching(..., bias_ids=...)` requires the
caller's `bias_ids` to equal `record.bias_ids` exactly.

`PreparedAtomisticDynamics.transition_hamiltonian` moves an accepted state to a
different scalar Hamiltonian. It books the energy jump as external work, and keys its
lineage by `potential.scientific_id`. `rebind_hamiltonian` re-resources the same
scientific Hamiltonian (for example, after a capacity ladder).

`initialize_state_detailed(..., cell_vectors=...)` returns an explicit
`INVALID_INITIAL_STATE` evaluation rather than raising on numerical failure.

## Authenticated free-energy analysis

`ReducedPotentialDataset`, `ReducedWorkDataset` and `ThermodynamicDerivativeDataset` are
the estimator boundaries. They bind dimensionless observations to:

- ordered state, potential and bias IDs;
- inverse temperatures and the reduced-potential convention;
- the phase-space measure;
- the sampling qualification;
- producer, run, lineage, active masks and coverage.

`FreeEnergySelectionPlan` records burn-in, thinning, correlation diagnostics and
synchronous time blocks. FEP, BAR, TI and MBAR return a `FreeEnergyResult` with:

- a fixed gauge;
- full covariance;
- overlap and connectivity;
- effective sample sizes;
- separate numerical and statistical status.

A finite value without successful support and covariance evidence is not a qualified
estimate.

## Current limits

- There is no unqualified time-dependent reweighting of adaptive draws. Freeze first.
- There is no asynchronous exchange. There are no combined or walker-dependent
  Hamiltonian moves without a derived acceptance.
- Mixed-temperature well-tempered histories and FIFO-overwritten histories do not
  support the usual profile formula.
- Force-only ABF has no energy, so it cannot enter multi-state tables, Monte Carlo
  volume moves, integrator-owned cell methods, RESPA splitting or switching work. Use a
  frozen `IntegratedProfileBiasForm` there.
- `AdaptiveProfileResult` profiles from `abf_profile` and `metadynamics_profile` are
  diagnostics of an adaptive run, not equilibrium free energies.
- Small smokes, the cookbook and short benchmarks demonstrate contracts only. They are
  not free-energy convergence evidence, and no sampling speed is claimed.

## Reproducibility

Random paths are addressed by experiment, repeat, replica, action and committed counter.
An exact resume of the same random path requires all of the following to be unchanged:

- phase-space measure and thermodynamic table;
- numerical operator schedule and coordinate map;
- control layout;
- bias identities and authenticated history content;
- continuation watermark.

Hidden retries, partial iteration commits, deposits on probes and stale pre-exchange
forces are not permitted.
