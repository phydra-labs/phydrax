# Host interchange and inspection

Phydrax inspection records are explicit host-side projections of accepted or candidate
scientific products. They are not numerical fields, JAX transformation inputs,
checkpoints, or persistence formats. Product-owned adapters must preserve support,
layout, representation, validity, status, and provenance identities and must report
semantic loss through `AdapterReport`.

`HostInspectionField` stores read-only host values and an explicit validity mask at one
cell, face, vertex, particle, marker, or point location. It never infers units,
resamples data, or converts between point values, averages, moments, and coefficients.
`HostInspectionFrame` groups uniquely named fields for exactly one candidate or
accepted state. `HostInspectionConversion` couples a frame to its interchange report.

Candidate and accepted frames are separate objects. A renderer or asynchronous
publisher may consume them, but renderer handles and serialization methods are outside
the inspection contract.

::: phydrax.interchange.AdapterReport

---

::: phydrax.interchange.HostInspectionField

---

::: phydrax.interchange.HostInspectionFrame

---

::: phydrax.interchange.HostInspectionConversion

## Bilby result import

`read_bilby_result_json` admits only bounded current plain JSON beneath an
explicit trusted root. It does not decode pickles, reconstruct Python objects,
or execute Bilby.

::: phydrax.interchange.ImportedBilbyResult

::: phydrax.interchange.read_bilby_result_json

## Geospatial qualification

::: phydrax.interchange.GeospatialContract

::: phydrax.interchange.GeospatialTransform

::: phydrax.interchange.QualifiedGeospatialGrid

::: phydrax.interchange.export_geospatial_grid

::: phydrax.interchange.render_geospatial_grid

## SEG-Y

::: phydrax.interchange.SEGYRev1IEEEProfile

::: phydrax.interchange.DecodedSEGY

::: phydrax.interchange.decode_segy_bytes

::: phydrax.interchange.read_segy

## Executed coordinate and time transforms

::: phydrax.interchange.CoordinateTransformPlan

::: phydrax.interchange.CoordinateTransformResult

::: phydrax.interchange.execute_coordinate_transform

::: phydrax.interchange.LeapSecondTable

::: phydrax.interchange.TimeReferenceContract

::: phydrax.interchange.TimeTransform

::: phydrax.interchange.convert_time

## Borehole and planetary coordinates

::: phydrax.interchange.BoreholeTrajectory

::: phydrax.interchange.BoreholeInterval

::: phydrax.interchange.PreparedBoreholeSampling

::: phydrax.interchange.ReferenceBodyContract

::: phydrax.interchange.PlanetaryCoordinateContract

## SEG-Y revision 2

::: phydrax.interchange.SEGYRev2IEEEProfile

::: phydrax.interchange.decode_segy_rev2_bytes

::: phydrax.interchange.decode_segy_rev2_resource

::: phydrax.interchange.read_segy_rev2

## Raster and array formats

::: phydrax.interchange.read_geotiff_grid

::: phydrax.interchange.read_cf_netcdf_grid

::: phydrax.interchange.read_consolidated_zip_zarr_grid

## Seismic waveforms and metadata

::: phydrax.interchange.QualifiedWaveformTrace

::: phydrax.interchange.QualifiedWaveformCollection

::: phydrax.interchange.read_miniseed3

::: phydrax.interchange.read_sac

::: phydrax.interchange.StationXMLMetadata

::: phydrax.interchange.read_stationxml

## Electrical and electromagnetic formats

::: phydrax.interchange.ElectricalTabularSurvey

::: phydrax.interchange.read_electrical_survey_csv

::: phydrax.interchange.MTImpedanceData

::: phydrax.interchange.read_edi_impedance

::: phydrax.interchange.read_emtf_xml_impedance

## Potential-field and geodetic formats

::: phydrax.interchange.ICGEMGravityModel

::: phydrax.interchange.read_icgem_gfc

::: phydrax.interchange.GeomagneticHarmonicModel

::: phydrax.interchange.read_geomagnetic_coefficients

::: phydrax.interchange.SINEXPositionSolution

::: phydrax.interchange.read_sinex_positions

::: phydrax.interchange.RINEXObservationData

::: phydrax.interchange.read_rinex_observations

## Well-log formats

::: phydrax.interchange.QualifiedWellLog

::: phydrax.interchange.read_las_curve

## Optional OpticStudio boundary

The OpticStudio adapter is host-only, lazily imports the optional ZOSPy package,
opens only owned standalone sessions, and rejects unsupported sequential features
before mutating the vendor system.

::: phydrax.interchange.opticstudio.OpticStudioBackend

---

::: phydrax.interchange.opticstudio.OpticStudioAnalysisRequest

---

::: phydrax.interchange.opticstudio.export_sequential_to_opticstudio

---

::: phydrax.interchange.opticstudio.run_opticstudio_analysis

## External design-analysis execution

The aerodynamic and device-design adapters are available as direct modules:

```python
from phydrax.interchange import (
    dafoam,
    geant4_detector_design,
    hfss_design,
    xfoil,
)
```

They are concrete, host-only boundaries around mutable external engines. They are not
Phydrax-native state solvers, do not run under `jit`, `vmap`, `grad`, or another JAX
transformation, and do not provide a mock, cached-success, surrogate-gradient, or
universal-runner fallback. Even an argument-free engine call in a JAX trace is refused;
traced inputs are refused in ordinary host calls. Use the returned detached values only
after the external call has completed.

Every executable is a `PinnedExecutable`: the caller supplies its path, lowercase
SHA-256 digest, release, and license identifier; `source_url` records optional
provenance. The executable bytes are checked for each run, but this does not
transitively pin dynamic libraries or make the process a sandbox. Runs use private
directories with bounded exact-byte inputs, logs, and named outputs and positive
finite timeouts. A launch error, timeout, nonzero
exit, absent output, changed executable, or resource-bound violation raises with the
bounded run evidence; it never becomes an optimization penalty or a skipped success.

The repository's parser and boundary tests are not live engine qualifications. The
local development host used for this work has no DAFoam runtime, Windows
AEDT/HFSS installation, or initialized ECCE/Fun4All/Geant4 runtime. Their absence is
a runtime-verification blocker only and **not** a successful qualification. A result
supports only the exact engine, source, dependencies, realization, and profile recorded
by its artifact; third-party tutorial or published results are not Phydrax
certification.

### DAFoam 4.0.3 steady serial primal and total adjoint

`phydrax.interchange.dafoam` accepts a prepared OpenFOAM case as exact bytes under
`0/`, `constant/`, and `system/`; a nonempty geometry-source declaration; canonical
DAFoam options; and every declared design vector. The steady aerodynamic profile is
limited to `DASimpleFoam`, `DARhoSimpleFoam`, or `DARhoSimpleCFoam`, reverse AD,
Krylov/PETSc adjoints, `discipline="aero"`, explicitly declared
`primalMinResTol`, `primalMinResTolDiff`, `checkMeshThreshold`, functions, design
surfaces, and solver/function-participating `volCoord`, `patchVelocity`, `patchVar`,
or `field` inputs. `volCoord` means the complete ordered volume-coordinate vector;
the adapter does not invent a CAD or FFD map. The worker requires `MPI.COMM_WORLD`
size one.

Create a `DAFoamRuntime` with `pin_dafoam_runtime`. The reference API is pinned to
[DAFoam v4.0.3](https://github.com/mdolab/dafoam/tree/v4.0.3), and the runtime
requires exactly DAFoam `4.0.3`, a pinned Python executable, an explicit DAFoam
license identifier, the
package directory containing `pyDAFoam.py`, and hashes for both `pyDAFoam.py` and
`mphys/mphys_dafoam.py` plus a built native extension. Supply dependency roots that
cover the used OpenFOAM/DAFoam libraries and OpenMDAO, MPhys, PETSc, MPI,
`mpi4py`, `petsc4py`, and IDWarp installations. Those selected Python/native files
are hashed and rechecked in the worker; system libraries outside the supplied roots
remain unpinned. Runtime identity is bounded by default to 100,000 files and 8 GiB.

`run_dafoam(..., total_adjoint=False)` returns external function values only after
native `checkMesh` and `primalFail`/finite-state acceptance.
`total_adjoint=True` additionally runs OpenMDAO reverse `compute_totals` one
functional at a time and returns `DAFoamTotalDerivative` records plus per-functional
`DAFoamAdjointEvidence`. Each record carries state, realized-mesh, declared-design,
and adjoint hashes, PETSc convergence reason, iteration count, residual norm, and
failure text. The worker verifies that state, mesh, and design did not change between
the accepted primal and each total-adjoint evaluation, and the host verifies those
identities again. This is same-realization derivative evidence, not differentiation
through a host process. A failed mesh/primal exposes no function values; a failed or
nonfinite adjoint exposes no total derivatives. `DAFoamResult.require_acceptance()`
raises `DAFoamConvergenceError` rather than accepting either case.

Case bytes and the canonical request have individual hashes and a request identity;
the default combined run-artifact bound is 128 MiB. The qualification entrypoint
generates its NACA0012 case from the public four-digit equation, runs the real steady
primal and totals, then compares the angle total against two independent same-case
central finite-difference solves. Its acceptance bound is
`1e-4 + 0.03 * abs(central_difference)`; the modest generated mesh is not a
mesh-independent aerodynamic-accuracy claim.

```console
PYTHONPATH=. python tools/aerodynamic_design_adapter_benchmarks.py dafoam \
  --python /path/to/python --python-version 3.x.y \
  --package-root /path/to/dafoam/package \
  --dependency-root /path/to/pinned/dependencies
```

::: phydrax.interchange.dafoam
    options:
      members: true

### XFOIL 6.9/6.99 independent operating points and polars

`phydrax.interchange.xfoil` supports only real XFOIL `6.9`/`6.99` viscous,
type-1 fixed-Reynolds/fixed-Mach operating points. `XFOILOperatingPoint` declares
angle in degrees, positive Reynolds number, subsonic Mach number, positive
`ncrit`, and top/bottom transition positions. Geometry is 20–240 finite,
unit-chord `(x, y)` nodes ordered trailing-edge to leading-edge to trailing-edge.
The adapter writes those exact nodes, uses `PCOP` rather than repaneling or
normalizing, saves the realized nodes, and rejects a changed realization.

`run_xfoil_point` starts a fresh executable and boundary layer, without warm start
or retry. It accepts a point only when native PACC output contains exactly one
finite seven-column row with the requested rounded header conditions and angle,
nonnegative drag, and transition positions in `[0, 1]`; a clean process exit is
not convergence. A missing row is represented as
`XFOILPointResult(coefficients=None, converged=False, ...)`, never zeros, NaNs, or
a penalty, and `require_convergence()` raises. `run_xfoil_polar` runs 1–256 such
processes independently in request order, so failure or boundary-layer history at
one point cannot continue into another. It retains each child artifact, exact
operating point, geometry SHA-256, executable identity, and aggregate convergence.
The default per-point output bound is 8 MiB. No derivative capability is claimed.
The executable hash does not cover its dynamic libraries; the inherited host
environment remains trusted and must be part of the operator's qualification. The
benchmark records the XFOIL distribution URL as provenance, but it does not claim a
detached source-tree identity; the executable bytes and observed 6.9/6.99 PACC
header are the execution evidence.

```console
PYTHONPATH=. python tools/aerodynamic_design_adapter_benchmarks.py xfoil \
  --executable /path/to/xfoil --version 6.99 --timeout 120
```

::: phydrax.interchange.xfoil
    options:
      members: true

### HFSS eigenmode/EPR and Q3D

`phydrax.interchange.hfss_design` implements the QDesignOptimizer `0.2.0`
coupled-transmon tutorial at
[Git commit `d4f6ada5ada59b786df1006d53f8f148b364364e`](https://github.com/202Q-lab/QDesignOptimizer/tree/d4f6ada5ada59b786df1006d53f8f148b364364e).
It requires an exclusive licensed Windows host, Python 3.11 or 3.12, and AEDT
2021 R2 or 2022 R2. The caller supplies
separate pinned Python and `ansysedt.exe` executables, exact unit-bearing upstream
design-variable strings, a nonempty source license, and a local checkout whose
selected license, project, lockfile, package, and tutorial bytes match that commit.
The observed Python and AEDT releases must equal their executable declarations.
An already-running AEDT desktop, a different COM-opened executable, multiple AEDT
desktops, or loss of process ownership fails qualification.

The exact Poetry-lock dependency profile is:

| Distribution | Version | Distribution | Version |
| --- | --- | --- | --- |
| `gdstk` | `0.9.62` | `geopandas` | `1.1.2` |
| `gmsh` | `4.11.1` | `ipykernel` | `7.1.0` |
| `ipython` | `9.10.0` | `matplotlib` | `3.10.8` |
| `numpy` | `1.26.4` | `pandas` | `2.3.3` |
| `pint` | `0.24.4` | `psutil` | `6.1.1` |
| `pyaedt` | `0.23.0` | `pyepr-quantum` | `1.0.0` |
| `pygments` | `2.14.0` | `pyside6` | `6.10.2` |
| `pywin32` | `308` | `pyyaml` | `6.0.3` |
| `qdarkstyle` | `3.1` | `quantum-metal` | `0.7.4` |
| `qutip` | `5.2.3` | `scipy` | `1.15.3` |
| `scqubits` | `4.3.1` | `shapely` | `2.0.7` |

Every distribution version must agree with the detached lockfile. Wheel `METADATA`,
`RECORD`, optional `direct_url.json`, and the installed file manifest/content are
hashed; absent files, more than 100,000 files, or more than 2 GiB of recorded content
per distribution fail. The Python executable, AEDT executable, detached worker,
selected QDesignOptimizer files, complete selected source-tree identity, and
dependency lock identity are recorded. The AEDT executable hash does not claim to
content-pin the installation's dynamic DLLs.

`HFSSDesignProfile` selects tutorial group 1 or 2, exactly two named disjoint mode
windows, at least two adaptive passes, explicit frequency/capacitance convergence
tolerances, and uniquely named targets. The real workflow runs HFSS Eigenmode,
pyEPR, and Q3D; it never substitutes a driven S-parameter analysis. Supported target
quantities are EPR `frequency_hz`, eigenmode `kappa_hz`, EPR `chi_hz`,
EPR `participation`, and signed `capacitance_ff`. Outputs are finite target values,
normalized residuals, labels, mesh-evidence identity, artifacts, and run evidence;
adaptive convergence is numerical evidence, not statistical uncertainty.

Mode identity is the pinned tutorial's upstream target-frequency rank within two
disjoint owner windows, checked independently against eigenmode and EPR frequencies.
It is **not** field-overlap tracking: a crossing, ambiguity, or departure from its
window is rejected. Mesh identity hashes `request.json`, eigenmode and capacitance
adaptive histories, and exported mesh-statistics files. It explicitly does not claim
element-connectivity identity. Both solvers need at least two measured passes and a
finite final percentage delta within the declared tolerance; EPR Kerr and Q3D
capacitance matrices must be finite and symmetric, participation must lie in
`[0, 1]`, and every requested target must exist. Nonfinite, unconverged, identity-
mismatched, dependency-mismatched, or incomplete results raise
`DeviceQualificationError`; they are not returned as residual penalties.

```console
python tools/device_design_adapter_benchmarks.py \
  --engine hfss --config hfss-run.json --output hfss-evidence
```

The JSON config and its relative paths are bounded and traversal-safe. It requires
`source={directory,commit,license_id}`, complete `PinnedExecutable` objects named
`python` and `aedt`, `profile={group,mode_windows_hz,targets,max_passes,`
`frequency_tolerance_percent,capacitance_tolerance_percent}`, and a
`design_variables` mapping. Dependencies are never installed or downloaded. The
default timeout is 3,600 seconds and the default combined artifact bound is 64 MiB.
The command exits nonzero and records failed/unavailable evidence when the runtime
cannot be used.

::: phydrax.interchange.hfss_design
    options:
      members: true

### Fun4All/Geant4 detector tracking and reconstruction

`phydrax.interchange.geant4_detector_design` implements the real
[GYM4DetectorDesign workflow](https://github.com/wmdataphys/GYM4DetectorDesign/tree/bbfd1b8dd10dc36b28ab118aac8af5dbace79296)
pinned to commit `bbfd1b8dd10dc36b28ab118aac8af5dbace79296`: pi-minus generation,
Fun4All/Geant4 geometry and fast Kalman tracking, followed by the upstream ROOT
double-Gaussian resolution reconstruction. It requires a pinned ROOT executable;
exact settings, design, and sPHENIX ROOT field-map bytes; and an initialized
ECCE/Fun4All/Geant4/ROOT host or container with runtime data, an explicit source
license, and a nonempty
`runtime_build_id` identifying that environment/container. The worker records the
observed ROOT and Geant4 releases, random seed, and loaded-library paths. The ROOT
hash alone does not pin runtime libraries, Geant4 data, or calibration resources.

Only `geometry_profile="corrected-silicon-units"` is admissible. Before execution,
the adapter snapshots the pinned `AllMacros` source and performs five mandatory,
ordered, exact-count corrections on detached bytes:

1. in `G4_Barrel_EIC.C`, change
   `void BarrelSetup(PHG4Reco* g4Reco)` to
   `double BarrelSetup(PHG4Reco* g4Reco)` once;
2. in `G4_Barrel_EIC.C`, replace both occurrences of
   `pitch / 10000. / sqrt(12.)` with `pitch / sqrt(12.)` so micrometre pitch is
   not converted twice;
3. in `G4_FST_EIC.C`, replace
   `Form("SI_L%i_THICKNESS", j + 1)]*Units::um` with
   `Form("SI_L%i_THICKNESS", j + 1)] * 9.37 / 100.` once;
4. in `G4_FST_EIC.C`, replace
   `Form("SI_L%i_THICKNESS", j)]*Units::um` with
   `Form("SI_L%i_THICKNESS", j)] * 9.37 / 100.` once; and
5. in `G4_TrackingSupport.C`, replace
   `Form("SI_L%i_THICKNESS", ilyr)] * Units::um` with
   `Form("SI_L%i_THICKNESS", ilyr)] * 9.37 / 100.` once.

Each stage records its path, exact before/after text, required replacement count,
reason, and before/after SHA-256 digests. Execution is refused if a source context
does not exactly match the pin or any known-bad expression remains. The artifact
retains both complete original and corrected source-file identities, both source-tree
identities, the correction ledger, geometry and field-map hashes, runtime identity,
seed, and engine evidence. The corrected geometry semantics are explicit: positions,
radii, and barrel lengths are cm; pitch and disk thickness are micrometres; barrel
thickness is percent radiation length using silicon `X0 = 9.37 cm`.

`GYMDetectorProfile` fixes the reconstruction support to 50 eta/momentum bins over
eta `[-3.4, 3.4]` and momentum `[1, 20]` GeV/c, PDG `-211`, explicit events and
31-bit seed, complete physically ordered geometry, a minimum reconstructed-track
count, and a maximum reduced chi-square. Geant4 overlap diagnostics or abnormal
Fun4All event-loop completion fail before reconstruction. Reconstruction then
requires every bin exactly once, finite positive fitted widths/errors, no histogram
overflow/underflow, sufficient tracks, consistent measured binomial error, and
reduced chi-square within the declared bound. Missing, nonfinite, or unqualified
physics raises with both bounded run records; values are never clipped, imputed, or
turned into penalties.

Accepted outputs include all qualified bins, the aggregated momentum-resolution
percent with propagated disjoint-bin fit diagnostic, Kalman inefficiency with
binomial error, elapsed time, and both simulation/reconstruction artifacts. The
upstream fit error is a chi-square-scaled fit diagnostic, not calibrated observation
noise. The legacy inverse-thickness sum combines reciprocals of unlike native units
(barrel percent-X0 and disk micrometres); it is retained only as
`upstream_inverse_thickness_diagnostic`, explicitly listed in `excluded_costs`, and
is never an accepted objective, cost, or training penalty.

```console
python tools/device_design_adapter_benchmarks.py \
  --engine geant4 --config detector-run.json \
  --output detector-evidence --repeats 3
```

The detector config requires `source={directory,commit,license_id}`, a complete
`root` executable pin, `settings_file`, `design_file`, `field_map_file`, and
`profile={events,seed,runtime_build_id,geometry_profile,minimum_tracks_per_bin,`
`maximum_reduced_chi_squared}`. Repeats increment the declared seed and report
sample repeatability only after every independent run is accepted. The default
timeout is 3,600 seconds and the combined input/log/output bound is 256 MiB. Missing
engines or dependencies yield nonzero failed/unavailable qualification, never a skip.

::: phydrax.interchange.geant4_detector_design
    options:
      members: true

## Optional energy execution

These host-only boundaries require explicit runtime/version/license provenance.
They do not execute inside JAX transformations or provide surrogate derivatives.
FMI support is synchronous FMI 2.0 Co-Simulation; HELICS support is value
federation. See [the energy interchange guide](../guides_energy_interchange.md)
for lifecycle, capabilities, unsupported features, and ownership.

::: phydrax.interchange.energy_runtime
    options:
      members: true

::: phydrax.interchange.fmi
    options:
      members: true

::: phydrax.interchange.helics
    options:
      members: true
