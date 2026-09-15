# Native alchemical free energy

This guide defines the scientific contracts shared by controlled atomistic Hamiltonians,
multistate sampling, free-energy estimators, and protocol composition. It does not create
a second molecule, topology, unit, trajectory, scheduler, or storage model.

## Measure and state identity

Every multistate calculation starts from one `AtomisticPhaseSpaceMeasurePlan`. Its
identity binds the prepared system, stable particle support, topology, masses, mobility,
coordinate map, constraints, cell convention, and `AtomisticUnitSystem`. States with
different support, masses, constraints, or coordinate measure cannot be mixed by the
current estimators.

`AtomisticThermodynamicStatePlan` declares an NVE, NVT, or NPT state and its ordered
Hamiltonian controls. `PreparedThermodynamicStateTable` lowers all states to numeric
inverse-temperature, temperature, pressure, control, ensemble, and validity arrays. The
ordered state IDs are scientific order; execution worksets may canonicalize task order
only when they scatter results back to that state order.

Reduced potentials are dimensionless. NVT states use the declared inverse temperature
and potential energy. NPT states additionally use the same pressure, volume-coordinate,
and scaled-entity measure convention as the barostat. Momenta are marginalized; any
state-dependent kinetic or constraint normalization must be supplied explicitly rather
than hidden in an energy matrix.

## Sampling and persistence

`AtomisticMultistatePlan` binds the prepared dynamics, thermodynamic table, stable
replica IDs, lineage, and either neighbor replica exchange or SAMS. Every required
state-by-replica reduced potential is evaluated after propagation and before assignment
updates. Exchanges swap state labels. An accepted state change rebases force, kinetic,
potential, total-energy, and auxiliary cache state before the next force kick.

`AtomisticMultistateSegmentPlan` bounds device memory. A segment result records reduced
potentials, coverage, sample activity, origins, chain/draw/repeat/dependence lineage,
assignment decisions, and the successor continuation watermark. The checkpoint writer
commits the segment and successor continuation as one authenticated payload. A valid
rejected proposal consumes its random-action counter; a rolled-back invalid iteration
consumes none.

Only kernels carrying a declared target-distribution qualification may support an exact
equilibrium claim. Finite-step BAOAB results retain their numerical-kernel identity and
must not be described as exact merely because replica exchange succeeded.

## Analysis boundary

Use the UQ bridge functions to build authenticated datasets:

- `reduced_potential_dataset_from_multistate` for committed equilibrium segments;
- `reduced_potential_dataset_from_alchemical_evaluation` for explicit controlled-state
  cross evaluation;
- `reduced_work_dataset_from_alchemical_switching` for oriented switching work.

`FreeEnergySelectionPlan` binds burn-in, thinning, correlation diagnostics, and joint
time blocks to one dataset. Replica lanes coupled by exchange are resampled
synchronously. FEP, BAR, TI, and MBAR return a common `FreeEnergyResult` with full
covariance, overlap/connectivity, effective sample sizes, solver evidence, and separate
numerical and statistical status. Incomplete active MBAR coverage is rejected.

## Sign conventions

All elementary values use destination minus source.

For environment decoupling, define `D_env = f_off,env - f_on,env`. Protocol plans use:

- neutral absolute solvation: `D_vac - D_solv + C_solv`;
- standard absolute binding: `D_solv - D_complex + C_restraint + C_symmetry`;
- mapped relative solvation B minus A: `T_solv(A to B) - T_vac(A to B)`;
- mapped relative binding B minus A: `T_complex(A to B) - T_solv(A to B)`.

Corrections are typed result objects with explicit identities and covariance. A missing,
failed, differently normalized, or differently mapped leg invalidates the composite
result. One-repeat between-repeat dispersion is unavailable rather than zero.

## Supported controlled physics

The controlled Hamiltonian uses the canonical force-field scalar and currently admits
only routes whose endpoint behavior can be made complete and exact. Supported controls
cover prepared harmonic bonded routes, Lennard-Jones interactions, direct Coulomb
interactions, stable-ID regions/mappings, and cell-aware force/control derivatives.

Preparation rejects state-dependent masses, constraints, virtual-site geometry,
charge-changing controlled regions, unsupported force-field routes, and reciprocal or
tail terms whose real, reciprocal, self, background, exception, and correction pieces
cannot change together. `RegionMaskedPotential` is not a reciprocal-space alchemical
partition.

## Protocol admission

The protocol value plans consume already prepared native force fields and qualified leg
results. They do not infer protonation, tautomer, mapping, force field, partial charges,
solvent composition, box, ions, or restraint anchors.

Initial solvation and binding contracts are neutral-only. Charged transformations remain
unsupported until a qualified co-alchemical counterion or explicit background and
finite-size correction contract exists. Absolute binding additionally requires an
independently validated restraint and standard-state correction result. Separated and
mapped relative protocols retain their leg and mapping identities rather than reducing
them to one anonymous scalar.

## Campaign inference

`FreeEnergyEdgeObservation` represents an oriented contrast and either a joint influence
basis or declared covariance. `FreeEnergyNetworkPlan` performs gauge-fixed generalized
least squares, reports connectedness and rank, and evaluates covariance-aware cycle
residuals. Zero off-diagonal covariance requires explicit independence evidence.

Automatic network selection is not implied. Mapping scores are planning evidence, not
measurement uncertainty; a future planner must preserve connectivity, redundancy,
cost, and failure robustness.

Sparse cross-evaluation uses the separate `SparseReducedPotentialDataset` and explicit
pairwise BAR edges. `sparse_pairwise_free_energy_network` requires every edge's source
and destination observations to be jointly covered and requires edge dependence-group
sets to be disjoint before using diagonal-covariance network inference. Shared sparse
samples are rejected rather than treated as independent; dense MBAR remains strictly
dense.

## Reproducible benchmark

Run the bounded native runtime and controlled-Hamiltonian smoke benchmark with:

```console
python -m benchmarks.multistate_free_energy --smoke
```

The checked evidence is `benchmarks/multistate_free_energy.json`. It separates lowering,
compilation, synchronized steady execution, logical array bytes, cross-evaluation
throughput, estimator diagnostics, physical success, and environment identity. It is a
numerical/runtime certificate, not a molecular force-field or experimental-accuracy
claim.
