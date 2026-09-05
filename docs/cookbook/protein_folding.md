# Run an admitted protein force-field handoff, then separate inference controls

This recipe starts with the retained, source-pinned **1L2Y** structure and an
actual **Amber14 protein ff14SB** parameterization. It executes native
conservative energy/forces and a short NVE trajectory. The later recipes execute
joint inference, enthalpy estimators, rotamer inference, coordinate training, and
mixed/cotranslational mechanics, but explicitly do **not** substitute those
numerical demonstrations for experimental folding qualification.

See the [guide](../guides_protein_folding.md),
[public API](../api/applications/protein_folding.md), and
[source and rights disposition](../biophysical_sources.md) for the full contracts.

## 1. Prepare the environment and retain the source

Run the commands from the repository root with Python 3.11–3.13 and the local
package installed. The physical handoff additionally needs OpenMM 8.2–8.x;
the repository's `atomistic-interop` extra includes it. In a selected virtual
environment, installation is:

```bash
python -m pip install -e '.[atomistic-interop]'
```

Dependency installation may access a package index. The benchmark itself does
not download a PDB, run an external prediction service, or fetch model weights.
The package enables JAX 64-bit arithmetic by default; retain the emitted JAX,
backend, device and precision evidence when comparing numerical results.

Keep these existing repository files together:

- `tests/fixtures/protein_folding/1L2Y.pdb`
- `tests/fixtures/protein_folding/1L2Y.source.json`

The pinned PDB SHA-256 is
`5d1bbb545a312dfff1ae1e64b6d8addecb2f561ddc4011aeb5bee9d1dfcd4438`.
The raw file retains all 38 deposited NMR models. Selecting model 1 does not
convert it into a population or delete the other models. The structure source
is admitted as wwPDB CC0 core coordinate data; the parameter artifact is the
installed OpenMM `amber14/protein.ff14SB.xml` with separate recorded force-field
data provenance. These permissions are distinct from the package's own license.
Review them for your requested use rather than copying the fixture's declarations
to unrelated data or parameters.

## 2. Execute the real all-atom workflow

```bash
python benchmarks/protein_binding.py --steps 4 --repeats 3
```

Use this **script path**, not `python -m benchmarks.protein_binding`: its benchmark
runtime helper is imported from the script directory. The retained PDB is the
default. `--pdb` may name another local copy of the same pinned bytes, not an
arbitrary replacement protein under the 1L2Y benchmark claim.

The complete workflow in that entrypoint does the following:

1. Checks the pinned source bytes, retains all model records, selects model 1
   explicitly, and checks a bijection between original PDB serials and OpenMM
   particles.
2. Creates construct chain A, sequence `NLYIQWLKDGGPSSGRPPPS`, and explicit
   author-number-to-`ResidueKey` mapping. It resolves canonical atom names without
   overwriting source-row identities.
3. Supplies `ResolvedProteinChemistry` with the existing explicit hydrogens,
   residue chemical-state declarations, `NH3+`/`COO-` termini, and source lineage.
   It does not add hydrogens, guess pH, or repair missing atoms.
4. Parameterizes the supplied topology with OpenMM Amber14 ff14SB,
   `NoCutoff`, no constraints, and no center-of-mass remover. It records the
   installed parameter file's checksum and rights separately from the PDB.
5. Calls public `interchange.bind_protein_openmm` with the full source-record
   map and explicit native electronvolt/ångström/dalton/femtosecond units.
   The native adapter's bounded NoCutoff Lennard–Jones approximation is admitted
   explicitly with a 100 Å cutoff, and its warning remains in the report.
6. Builds the dense native neighborhood, evaluates the native conservative
   potential, and compares energy/forces at the **same initial configuration**
   against OpenMM's Reference platform.
7. Prepares declared bond/clash/chirality/peptide geometry screening and executes
   `workflows.run_protein_dynamics` with `VelocityVerletPlan(0.05)`, zero initial
   velocity, explicit velocity units, and a fixed PRNG key. Four steps cover
   0.2 fs of this isolated model, not an equilibration or a folding event.
8. Emits JSON with preparation, compilation, repeated synchronized execution,
   logical/compiled memory evidence, native/geometry success, energy range, and
   distinct raw, parameterized, and trajectory artifact IDs. A failed native
   force evaluation, rollout, or final geometry raises after emitting evidence.

### Interpret the output

| JSON field | What to inspect |
|---|---|
| `claim` | Must remain `caller-parameterized-native-handoff-not-folding-accuracy` |
| `capacity`, `active_atoms` | The admitted all-atom fixture has 304 atoms; missing material must not become padding |
| `retained_raw_models` | All 38 raw models remain in the source artifact |
| `reference_energy_abs_error_eV` | Absolute same-configuration native/OpenMM potential-energy difference |
| `reference_force_max_abs_error_eV_per_angstrom` | Maximum Cartesian force-component difference, not a structure RMSD |
| `native_force_successful`, `short_nve_successful`, `geometry_successful` | Separate native numerical and declared-geometry gates |
| `nve_total_energy_range_eV` | Short-run energy range for this step size/initial state, not long-term stability |
| `interchange_warnings` | Retain the finite-cutoff approximation; do not erase it after a successful comparison |
| `raw_artifact`, `parameterized_artifact`, `trajectory_artifact` | Different lineage stages, not interchangeable source identifiers |
| `environment`, `compilation`, `execution_seconds`, `memory`, `compiler` | Context for reproducing timing and precision; memory categories are not interchangeable |

An observed execution of this admitted workflow passed with **304 atoms**, all
**38 raw models retained**, maximum force difference
**7.374656441072602 × 10⁻¹⁴ eV/Å**, and energy difference
**5.329070518200751 × 10⁻¹⁵ eV**. These are numerical handoff observations for the
pinned configuration, not a tolerance guarantee across versions/devices and not
experimental model accuracy. Source/force-field uncertainty is unknown (`None`),
not zero.

The 100 Å bound exceeds this fixture's initial extent; it does not make a finite
native cutoff equivalent to unbounded NoCutoff for arbitrary future coordinates.
This isolated profile includes no solvent, ion atmosphere, barostat, or
experimental condition calibration. More steps do not remove those limitations.

## 3. Move from the repository example to your own physical inputs

The reusable public path is:

```python
from phydrax.applications import protein_folding as protein
from phydrax.atomistic.interchange import read_pdb_atom_records, select_pdb_model
```

Use the entrypoint's preparation sequence as a complete example, but supply your
own raw artifact and rights, explicit construct/source map, resolved chemistry,
parameterized force field, exact units, stable IDs, and justified geometry
bounds. `protein.bind_protein` accepts a prepared native field directly;
`protein.interchange.bind_protein_openmm` is the optional conversion route.
Neither is an automatic protein preparation service.

For NVT, the public dynamics workflow accepts a caller-configured native
`BAOABLangevinPlan` instead of `VelocityVerletPlan`. Keep the target temperature,
friction, step-size, PRNG and unit evidence with the result. This cookbook's
observed 1L2Y result is NVE, not an NVT calibration. Conservative sampling biases
must already be in the potential and named by `bias_id`; such trajectories must
not enter unbiased kinetic analysis.

Refuse unsupported chemistry rather than weakening the inventory: caps,
disulfides, PTMs, covalent ligands, solvent/ions, virtual DOFs, periodicity, and
multichain physical binding are outside the admitted P1 profile. Missing or
unresolved atoms require a separately source-supported realization, not zero
coordinates or padding.

## 4. Execute P2 joint experimental-model inference controls

The base package suffices for the remaining analytical controls; OpenMM is not
used by these commands.

```bash
python -m benchmarks.protein_unfolding_inference --repeats 3
```

This module constructs synthetic reversible two-state thermal × denaturant data
with two channel baselines, prepares one named-parameter problem, executes
native nonlinear least squares, compares held-out synthetic signal, and checks
that an isotherm retains non-identifiable thermal directions. Inspect
`case.fit_solver_successful`, `case.successful`, the held-out error, and
identifiability evidence; a low residual alone is not acceptance.

To also exercise actual posterior sampling:

```bash
python -m benchmarks.protein_unfolding_inference --repeats 3 --posterior-samples 100
```

The optional output retains posterior chain/draw and divergence information.
This small run demonstrates the NUTS path, not guaranteed chain convergence.
Fit covariance is a local approximation and is not a substitute for posterior
draws. Real experimental use additionally needs calibrated errors/covariance,
rights-cleared references with uncertainty, and reversible equilibrium evidence.
The public family also includes monomer/dimer three-state models, repeat transfer
models, chevrons, and parallel paths; their exact observation and temperature
restrictions are listed in the guide/API, not implied by this two-state benchmark.

## 5. Execute P3 enthalpy and closure controls

```bash
python benchmarks/protein_thermodynamics.py --samples 1024 --replicas 4 --repeats 3
```

This script generates analytical independent sampling controls, evaluates
paired-state enthalpy, fits a heat-capacity slope, propagates an explicitly
**synthetic** melting uncertainty, and evaluates a constant-offset FEP control.
Inspect `delta_h_error_kJ_per_mol`, `delta_h_standard_errors`,
`delta_cp_error_kJ_per_mol_kelvin`, `linear_model_residuals`,
`closed_g_standard_errors`, and `constant_offset_fep_error`. Its
`experimental_closure_dependency` identifies the synthetic control in this run;
the field name does not make it measured protein data.

To make a physical paired-state claim, replace the controls with independently
equilibrated folded/unfolded **whole-box** enthalpy replicas with matched
composition/pressure/parameters, at least two independent replicas per
state/temperature, justified correlation-time bounds and complete blocks, and
at least three temperatures for a slope. A real free-energy closure also needs
a measured melting point or closed thermodynamic cycle and its covariance,
inside the sampled temperature interval. Enthalpy alone, one ligand Kd, or the
short NVE trajectory above cannot provide that closure.

Likewise, a free-energy estimate or configuration reweighting cannot establish
unbiased path kinetics. VAMP/VAC/TICA/MSM/committor consumers require physical
time calibration, uniform valid lags, independent basins/shooting outcomes,
reset-aware trajectories, and the unbiased-data conditions in the guide.

## 6. Execute P4 exact versus implicit-BP rotamer controls

```bash
python benchmarks/protein_rotamer_free_energy.py --sizes 3 6 9 --tolerances 1e-5 1e-9 --repeats 3
```

The script compiles **analytical, uncalibrated single-site rotamer tables** on
trees and loops. It compares exact enumeration with the implicit-BP scalar,
marginals and force derivatives, retaining root residual, iteration count,
contraction bound, branch status, compiler evidence and timings. Inspect each
`results[].rows[]` entry rather than reporting one aggregate speed as universal.

Tree normalizers are exact when inference succeeds. Loop values are Bethe
approximations: reducing the root tolerance does not eliminate approximation
error. A converged root without the contraction/derivative gate is not an
admitted conservative force branch. `biological_acceptance_gate` remains open
for rights-cleared calibrated parameters and independent measured validation.
This is not an all-atom or experimentally qualified folding demonstration.

## 7. Execute G1 training and retain proposal failures

For a self-contained actual native training/sampling workflow:

```bash
python -m benchmarks.biophysical_coordinate_generation --steps 200 --repeats 3
```

This prepares an original four-atom analytic deformation corpus, performs an
explicit train/validation split, trains the conditional velocity, samples via
the native ODE owner, canonicalizes separately, and screens every proposal.
Inspect training/held-out flow losses, coordinate error, `solver_status`,
`accepted_fraction`, and `all_samples_retained`. Four atoms are not a complete
protein residue. The condition variable is an analytic deformation parameter,
not temperature or biological time. No pretrained weights or external provider
are invoked.

For actual admitted source coordinates rather than the analytical fixture:

```bash
python -m benchmarks.biophysical_coordinate_generation_pdb \
  --pdb tests/fixtures/protein_folding/1L2Y.pdb \
  --source tests/fixtures/protein_folding/1L2Y.source.json \
  --steps 200 --repeats 3
```

This is **same-construct reconstruction** using all 38 source-pinned NMR models
with an explicitly heavy-atom coordinate ABI. It omits hydrogens without imputing
them and is not the Amber14 all-atom physical realization. The models share one
experiment/refinement and are correlated. A model-index holdout is therefore
only a reconstruction diagnostic, not a biologically independent predictor
split. Inspect `correlated_holdout_flow_loss`,
`nearest_correlated_holdout_coordinate_rmse_angstrom`, `solver_status`,
`geometry_accepted_fraction`, `all_raw_models_retained`, and the reported
`open_gates`.

In the 200-step, 30-fit/8-correlated-holdout reconstruction run, all eight generated
samples were retained and **zero passed geometry qualification**. Lower training
loss did not produce qualified protein structures. Do not turn the successful ODE
solver status into a successful structure-prediction claim.

In both runs, low flow loss and sparse geometry acceptance do not establish
physical validity, confidence calibration, equilibrium sampling, or folding
accuracy. All proposal failures remain visible. Generative time is 0 to 1
pseudotime. Coordinate likelihood and Boltzmann weights are unavailable on this
gauge-fixed support. A scientifically qualified model requires a rights-cleared
independent corpus, meaningful leakage-resistant splits, chemistry coverage,
compute evidence, and independent predictive/confidence calibration.

For user-supplied learned-provider outputs, use
`generation.CoordinateProviderProvenance` and `import_protein_hypotheses` with
separate output, weight and prepared-input rights, raw digest-bound sources,
MSA/template lineage, and explicit egress authorization where applicable.
Training and model save/load preserve inherited restrictions. Neither these
benchmarks nor an upstream code license authorizes restricted weights/output
training or commercial export.

## 8. Execute X1 mixed-resolution and X2 activation mechanics

```bash
python -m benchmarks.protein_nucleic_hybrid --particles 16 --steps 20 --repeats 3
python benchmarks/protein_cotranslation.py --residues 6 --steps 8 --repeats 3
```

The hybrid entrypoint uses package-style benchmark imports and must run as a
module; the cotranslation entrypoint uses a local runtime import and must run
as a script. Both use caller-defined **synthetic/reference-conditioned**
coefficients, not Amber14 protein chemistry or biological calibration.

For hybrid mechanics, inspect force/torque balance, fixed-support reactions,
reference longitudinal stiffness error, support-map identity, and `refinement`
rows. The model retains separate Cartesian protein and rigid nucleotide state,
with conservative cross-site forces and native force/torque pullback. Its KDK
integrator has finite-step rotational error and no heat bath. A successful
mechanics evaluation is not a calibrated protein–nucleotide binding prediction.

For cotranslation, inspect `active_sizes`, `accepted_native_steps`,
`inserted_mass`, `insertion_work`, `conservation_balance`,
`ledger_balance_residual`, and the explicit
`biological_timing_calibrated: false`. Each codon activates a stable-ID residue
at an atomic epoch boundary; dormant material is not active physics. Native MD
segments and insertion ledgers are separate. Do not form a lag pair across a
topology switch or interpret synthetic reduced-time dwells as translation rates.
The public cursor/checkpoint interface supports epoch-aware replay and retains
the preactivation state on rejected insertion.

## What this cookbook does and does not qualify

The 1L2Y/Amber14 run proves a particular real parameterized force handoff and
short admitted native trajectory. The other commands execute their actual
numerical workflows; they are not fake forward-only placeholders, but their
scientific inputs remain analytical, correlated, or reference-conditioned.

Do not combine these outputs into a claim of experimental folding accuracy,
absolute folding free energy without experimental closure, unbiased kinetic
rates from configuration weights, pretrained prediction, biological translation
time, or unrestricted commercial rights. Each such claim requires its own
source-supported conditions, uncertainty and requested-use authorization.
Unknown uncertainty stays `None`; synthetic error bars do not supply missing
experimental evidence.
