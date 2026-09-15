# Enhanced atomistic sampling

Collective variables, biases, thermodynamic states, multistate assignments, and
free-energy analysis are separate typed layers. Physical coordinates never silently
change thermodynamic labels, bias history, or reduced-potential conventions.

## Collective variables and biases

`CollectiveVariablePlan` defines coordinate domains, stable indices, parameters, and
metrics. Built-in variables include distances, angles, torsions, center-of-mass
separation, radius of gyration, coordination, native-contact similarity, aligned RMSD,
volume, density, and path coordinates. Periodic metrics retain branch margins and every
evaluation carries explicit success evidence.

Static equilibrium biases belong to the declared Hamiltonian and thermodynamic state.
Moving, metadynamics, and adaptive-biasing-force histories remain fixed-capacity dynamic
state. Those adaptive histories are not ordinary equilibrium samples: FEP, BAR, and MBAR
admission requires a separately qualified time-dependent reweighting contract.

## Phase-space measure and thermodynamic states

`AtomisticPhaseSpaceMeasurePlan` identifies the common system, topology, particle
support, masses, coordinate map, constraints, units, and cell convention. Every state in
one compiled table must share that measure.

`AtomisticThermodynamicStatePlan` declares NVE, NVT, or NPT intensives and an ordered
Hamiltonian-control vector. `PreparedThermodynamicStateTable` lowers the declared states
to numeric arrays of inverse temperature, temperature, pressure, controls, masks, and
stable state IDs. Dynamic assignments select rows with JAX gathers; no traced value
indexes Python objects.

BAOAB and the isotropic Monte Carlo barostat obtain temperature and pressure from this
table. Numerical plans own step size, friction, proposal width, cadence, and realization
identity—not a second copy of the target distribution.

`AtomisticCanonicalSamplingQualification` binds each multistate run to exact
target-distribution evidence or a finite declared sampling-bias bound. Approximate
kernels may produce diagnostic estimates, but their free-energy results do not receive
successful equilibrium status.

## Multistate execution

`AtomisticMultistatePlan` binds one prepared dynamics runtime, a thermodynamic-state
table, stable replica IDs, and exactly one assignment policy:

- `AtomisticReplicaExchangePlan` performs alternating disjoint neighbor exchanges and
  requires one replica per state.
- `AtomisticSAMSPlan` supports distinct replica and state counts and persists its
  adaptive log-weight state.
  Samples emitted while SAMS adaptation is active remain explicit but inactive for
  equilibrium free-energy analysis.

One iteration propagates under the current assignment, applies any scheduled valid
barostat move, evaluates the complete dimensionless reduced-potential matrix, records
sample origins, decides assignment moves, and rebases force and energy caches under the
accepted assignment. Valid rejected exchange or barostat proposals are successful
no-ops and consume their committed action counter. Any invalid propagation or required
cross evaluation rolls the complete iteration back without consuming counters.

`AtomisticMultistateSegmentPlan` executes a fixed number of iterations into bounded,
masked arrays. `AtomisticMultistateSegmentResult` retains state-major reduced
potentials, coverage, replica/state assignments, chain/draw/repeat/dependence lineage,
transition evidence, and a continuation watermark. Immutable segments provide
unbounded history without allocating a whole campaign on device.

The multistate checkpoint writer stores one segment and its successor continuation in
one authenticated payload. Restoration validates the runtime, thermodynamic table,
predecessor, watermark, counters, and array contents before allowing another segment.

## Authenticated free-energy analysis

`ReducedPotentialDataset`, `ReducedWorkDataset`, and
`ThermodynamicDerivativeDataset` are the estimator boundaries. They bind dimensionless
observations to ordered state and potential IDs, per-state inverse temperatures and
reduced-potential convention, phase-space measure, sampling-qualification identity,
exactness or finite bias bound, producer, run, sample lineage, active masks, and
coverage.

`FreeEnergySelectionPlan` records burn-in, thinning, correlation diagnostics, and
synchronous time blocks. Coupled replicas are resampled jointly within a repeat.
FEP, BAR, TI, and MBAR return `FreeEnergyResult`, including a fixed gauge, full
covariance, derived pair differences and errors, overlap/connectivity, raw and adjusted
effective sample sizes, solver evidence, and separate numerical and statistical status.
A finite value without successful support and covariance evidence is not a qualified
estimate.

Use `reduced_potential_dataset_from_multistate` to convert a committed segment without
losing its state, measure, or lineage identities.

## Reproducibility

Random paths are addressed by experiment, repeat, replica, action, and committed
counter. A replay requires the same phase-space measure, thermodynamic table, numerical
operator schedule, coordinate map, control layout, bias identities, and continuation
watermark. Hidden retries, partial iteration commits, and stale pre-exchange forces are
not permitted.
