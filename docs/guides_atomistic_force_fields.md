# Atomistic force fields

PhydraX represents a classical force field as an immutable `AtomisticForceFieldPlan`.
Preparation validates every term against one prepared atomistic system and returns one
`PreparedAtomisticForceField` with a stable identity and an auditable preparation report.

## Coordinate domains

Dynamics state stores physical degrees of freedom. `AtomisticCoordinateMapPlan` realizes
interaction sites for an evaluation. Identity maps preserve ordinary atomistic systems;
weighted and local-frame virtual sites derive extra coordinates without adding masses or
momenta. Forces on derived sites return to degrees of freedom through the coordinate-map
pullback, so force conservation and differentiation use one path.

Periodic virtual sites use the native bounded nearest-image stencil, including
for skew cells and rank-two lattices embedded in three Cartesian dimensions.
Cached fractional coordinates select integer images only; parent displacements
remain Cartesian, preserving transverse coordinates and force/cell derivatives.
Holding those fractions fixed does not detach a parent's force contribution.
Nonfinite active parent charts report unsuccessful realization. Lower-rank
lattices do not acquire a three-dimensional stress volume from this support.


## Terms and nonbonded policy

A bundle may combine bonded terms, torsion series, CMAP, pair overrides, Morse,
Buckingham, tabulated pairs, reaction-field electrostatics, and dispersion treatment.
`AtomisticNonbondedPolicy` is the single source of truth for exclusions, 1-4 scaling,
cutoffs, switching, combining rules, and reciprocal-space choices. Do not duplicate those
rules in individual potentials.

```python
force_field = phx.atomistic.AtomisticForceFieldPlan(
    terms=(
        phx.atomistic.MorsePotential(depth, width, equilibrium, cutoff),
        phx.atomistic.BuckinghamPotential(a, b, c, cutoff),
    ),
    provenance=phx.atomistic.AtomisticForceFieldProvenance(
        source="project-parameters",
        source_version="2026.1",
    ),
)
prepared = force_field.prepare(system)
```

Preparation is fail-closed: unsupported term kinds, mismatched capacities, invalid tables,
and incompatible policies raise before compilation. Runtime evaluation reports a success
flag in addition to energy and forces.

## Dispersion

Force-field dispersion terms are the fixed-coefficient definitions. They include
Lennard-Jones pairs with `LennardJonesDispersionCorrection`, Tang-Toennies-damped
`DampedDispersionPotential`, and `LennardJonesPMEPotential`. LJ-PME requires a symmetric,
positive semidefinite per-type `C6` matrix, factors it with square-root eigenvalues, and
refuses an indefinite matrix.

Conventional D3(BJ) (`D3BJPotential`) and FourierD3 (`FourierD3Potential`) are separate
methods that depend on the coordination number. FourierD3 uses signed low-rank `C6` factors
of the D3 reference block and keeps negative eigenvalues. It is never routed through LJ-PME.
Neither method includes an ATM three-body term, and PhydraX bundles no D3 reference tables.
See [Atomistic dispersion](guides_atomistic_dispersion.md) for the following:

- the table-rights contract;
- the conventional and modified coordination-number definitions;
- the AIMNet2 Tkatchenko-Scheffler stage;
- the error budget and derivative matrix.

## Long-range electrostatics benchmark scope

`benchmarks/atomistic_long_range.py` measures actual native programs phase by phase:
host preparation, lowering, compilation, first and warm execution, compiler bytes,
sampled memory, and retained bytes. Refused, failed, and out-of-memory rows keep their
exception. Rows cover the following:

- point charges: direct, DSF, Ewald, PME, and the slab boundary of both reciprocal
  methods;
- degree-two point and Gaussian multipoles on the finite, Ewald, and PME routes;
- induced-dipole polarization (PCG or fixed-order TCG) on the same routes. The
  Hamiltonian is the induced functional only; no permanent-permanent energy is added;
- `learned-charge-pme` and `learned-charge-ewald`: a native AIMNet2 term publishes atom
  charges through `AtomisticFieldReference` into the periodic Coulomb term of one
  program. The Coulomb term uses the `uniform-background` convention because AIMNet2
  publishes a nonzero `CHARGE_RESIDUAL`. The model is a random source-layout fixture, so
  these rows are chain-rule cost and consistency evidence, not accuracy evidence.

Every family runs the canonical `AtomisticDynamicsPlan` MD step. Periodic multipole and
polarization rows record their real-space route as an axis:

- `--multipole-real-space explicit-image-reference` is the bounded explicit image sum,
  quadratic in atoms;
- `--multipole-real-space neighborhood` is the term-owned `IMAGE_PAIR_SUM` requirement
  served by a declared cell-list image source.

Periodic polarization rows record their stability certificate. Select it with
`--polarization-stability dense-spectrum` (the default; it adds a dense eigensolve to
every evaluation) or `--polarization-stability assumed-positive-definite`.

The default `--workloads` set is `energy`, `efs`, `hessian-vector`,
`coefficient-refresh`, `cell-refresh`, and `md-step`. Two workloads run only when you
select them:

- `efs-workset`: full-program E/F/S through `ExecutionWorksetPlan` and
  `evaluate_execution_worksets_filter_vmap` over one shared preparation. Positions and
  current cells form the item lane. `--workset-batches` and
  `--workset-cells equal distinct` select the batches, and `--workset-bucket-capacity`
  selects padding. Each item is compared with its standalone evaluation. Programs
  prepared at different reference cells are recorded as separate preparations and are
  never stacked.
- `parameter-gradient` (learned rows): model-parameter gradients of an E/F/S loss and of
  the Coulomb term energy alone.

Learned `efs` rows also record the model-alone and held-charge programs, the nonzero
charge-response forces and stress, and an independent uniform-background Ewald of the
predicted charges. Learned `coefficient-refresh` separates model-parameter refresh from
total-charge-condition refresh.

Independent host references cover the following:

- converged Ewald, converged jointly in energy, forces, and analytic strain derivative;
- the DSF and direct pair sums;
- definitional finite multipoles;
- dense finite point polarization;
- `tests/_support/atomistic_multipole_reference.py`, for finite Gaussian polarization and
  for periodic point and Gaussian multipoles and polarization. Its bounds are explicit
  `--moment-reference-*` arguments.

TCG rows differ from the stationary reference by their declared truncation. Mesh rows are
not interpolation certificates. Measurements describe only the recorded devices.
Distributed reciprocal execution and host-sync audits are separate campaigns.

A converged independent reference does not make a finite-cutoff native row
converged. Check real-space truncation and reciprocal/mesh convergence separately,
especially for broad Gaussian sites. A tighter native reciprocal comparison that
shares the same real cutoff cannot detect the shared real-space truncation error.

## Rigid water

`SETTLEPlan` provides an analytic three-site water position projection and momentum tangent
projection. It is appropriate only for the geometry encoded by the plan; use general
distance constraints for other molecules.

## Interchange

Use `phydrax.atomistic.interchange` for OpenMM, OpenFF Interchange, and ParmEd boundaries.
Adapters return an `AtomisticInterchangeReport`; persist it with force-field provenance so
unsupported or approximated content cannot be hidden.
