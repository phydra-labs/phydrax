# Enhanced atomistic sampling

Collective variables, biases, replica labels, and free-energy estimators are separate typed
layers. A physical state never silently changes thermodynamic labels or bias history.

## Collective variables

`CollectiveVariablePlan` defines the coordinate domain, indices, parameters, and metric.
Built-in variables include distances, angles, torsions, center-of-mass separation, radius
of gyration, coordination, native-contact similarity, aligned RMSD, volume, density, and
path progress/distance. Periodic metrics provide wrapped differences for torsions and other
cyclic variables. Every evaluation includes a branch margin and success flag.

`AbstractCollectiveVariableProgram` is the execution boundary consumed by biases.
`ModelCollectiveVariableProgram` composes a frozen array model after any existing CV
feature program, preserving position derivatives and fixed output metrics. This is the
bridge for canonical slow coordinates learned by the variational-kinetics runtime.

## Bias plans

`AtomisticBiasPlan` represents one static harmonic, flat-bottom, wall, moving,
umbrella, metadynamics, or adaptive-biasing-force plan over a
`CollectiveVariableProgram`. Its fixed-capacity history is stored in
`AtomisticBiasState`. Only an accepted dynamics step advances schedules, deposits a
hill, or updates ABF statistics; rejected proposals leave history unchanged.
Checkpoint the physical state and bias state together.

`LearnedFreeEnergyBiasPlan` holds a gauge-aligned committee of scalar free-energy
models. Each member is shifted to zero at one declared reference coordinate before
averaging or computing disagreement. A smooth uncertainty taper multiplies the scalar
bias energy; forces differentiate the complete tapered energy and are never blended
after differentiation.

`RestrainedMeanForcePlan` estimates finite-stiffness free-energy gradients from
restrained windows. `fit_free_energy_model` trains a scalar model against those
gradients with inverse-uncertainty weighting. The finite-restraint approximation and
source window identity remain explicit.

## Replica ensembles

`AtomisticReplicaEnsemblePlan` separates replica slots from thermodynamic labels.
Exchange proposals swap labels rather than coordinate arrays and record a
deterministic ledger. The same acceptance rule supports temperature, Hamiltonian,
lambda, and umbrella exchange when the reduced-potential matrix contains all cross
evaluations.

## Free energy

`ReducedPotentialSamples` is the estimator boundary. FEP, thermodynamic integration, BAR,
and MBAR consume reduced potentials and return estimates, uncertainties, overlap, effective
sample sizes, and convergence status. Treat a finite estimate without overlap or
convergence as unsuccessful.

## Reproducibility

Use stable plan identities in checkpoints, split random keys by replica and accepted step,
and persist exchange ledgers and metadynamics/ABF state. A replay must use the same
coordinate map, collective-variable metrics, label schedule, and reduced-potential adapter.
