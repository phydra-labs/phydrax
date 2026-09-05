# External radiation records to initial-lesion evidence

This recipe runs the actual R1/R2 repository workflow: pinned-format column import,
source-linked geometry mapping, direct/indirect candidates, lesion realization,
circular contour clustering, scored-dose yield, and native probability calibration
with held-out predictions. The data are **hand-authored synthetic mechanics
fixtures**, not a real Geant4 campaign or experimental DNA-damage reference.

## Prerequisites and command

Use Python 3.11–3.13 with this checkout installed and a working supported JAX backend.
Run from the repository root, not from `docs/`:

```bash
python -m pip install -e .
python benchmarks/radiation_lesions.py --histories 128
```

For explicit timing repetitions, the observed smoke command was:

```bash
python benchmarks/radiation_lesions.py --histories 128 --repeats 3
```

The benchmark needs at least two histories and one repeat. It prints JSON to
standard output, including the execution environment, timings, result counts,
adapter identity, fit diagnostics, and scientific gates. No external provider or
ROOT package is required for this **column** route. There is no benchmark flag
that downloads a campaign, runs transport, or supplies experimental qualification.

## What the command actually computes

1. Build the exact Geant4 dnadamage1 `v11.3.0` physical and chemical column sets,
   canonical derivative bytes, source envelope, and synthetic rights manifest.
   Preserve original entry IDs and independent primary `EventID` values.
2. Import 3 physical deposition records and 1 selected OH/deoxyribose damage record
   per history using `import_dnadamage1_columns`. The source has nm/eV units and
   explicitly synthetic configuration, RNG/table lineage, chemistry, and scavenging
   metadata; those labels do not claim real transport.
3. Prepare two backbone scoring targets on opposite strands of a declared circular
   12-base-pair plasmid. Physical volume/copy routes allocate deposition to targets;
   the chemistry record uses coordinate membership.
4. Apply the inclusive 17.5 eV cumulative **deposition** threshold. The two records
   at one site contribute 10 + 7.5 eV; the other site receives 17.5 eV. `diffKin`
   remains kinetic-energy loss and is not used as deposition or carried energy.
   The selected indirect reaction retains its own cause at the first site.
5. Realize candidates with stable addressed draws, form contour clusters with gap
   one (including circular closure), and repeat after physical-record reversal.
   A dual-cause site is one initial lesion, not two sites.
6. Calculate DSB yield with explicitly synthetic `HistoryExposure` values: 1 J
   into 1 kg per history, 12 duplex base pairs, and one molecule. These values
   exercise denominator semantics; they are not reconstructed experimental dose.
7. Compile the fixed-support expected initial-lesion union map. At direct and
   indirect probabilities one half, the dual-cause site's expectation is 0.75
   and the direct-only site's is 0.5, giving 1.25 per history. This is an analytic
   synthetic check, not an expected DSB-cluster model.
8. Fit two cause probabilities using separate synthetic supports and Gaussian
   observations, with a proper logit prior; predict held-out physical conditions.
   Transport, chemistry, thresholding, and clustering are not differentiated.

## Read the output without promoting the claim

| JSON field | Interpretation |
|---|---|
| `profile`, `scope`, `source_digest`, `source_adapter_report` | Retain the synthetic profile, source bytes, declared adapter losses, and scientific boundary with results. |
| `histories`, `physical_records`, `chemical_records`, `dsb_clusters`, `order_invariant` | Workload and history/topology/order checks. |
| `dsb_yield_per_Gy_per_Mbp` | Yield under the declared synthetic scored-mass/base-pair normalization, not a literature-calibrated DSB yield. |
| `yield_history_sampling_se` | Independent-primary sampling error only; repeated deterministic fixtures need not show sampling variation. |
| `dose_uncertainty_known` | False when normalization uncertainty is `None`; zero history variation does not supply missing dose UQ. |
| `expectation_absolute_error` | Difference from the independent candidate-union reference on fixed support. |
| `heldout_absolute_error`, `likelihood_rank` | Synthetic held-out fit error and likelihood identifiability, not experimental accuracy. |
| `scientific_gates` | Missing experimental training/held-out observations and independent stage qualification remain visible. |
| `import_seconds`, `prepare_seconds`, `mapping`, `classification`, `expectation_compilation`, `expectation_runtime`, `calibration_seconds` | Distinct host/orchestration, lowering/compilation, and repeated execution costs. Compare only matched environments/workloads. |
| `peak_traced_host_bytes` | Python traced host allocation peak, not whole-process RSS or accelerator memory. |

The recorded 128-history smoke produced **384 physical records, 128 chemical
records, and 128 DSB clusters**, with `order_invariant=true`. Fixed-support
expectation absolute error was zero; held-out synthetic maximum absolute error was
`9.983941036906252e-6`, and likelihood rank was two. These are observed smoke
results, not performance guarantees for other hardware or biological accuracy.
The source evidence also includes actual binary ROOT-reader exercise with a
synthetic pinned-format fixture; the benchmark command above itself uses columns.

## Move to a real retained ROOT file

The optional reader is real, not a stub or transport wrapper:

```bash
python -m pip install -e '.[radiation-interop]'
```

This installs `uproot>=5.6,<6`. The application root does not import `uproot`;
`import_dnadamage1_root` loads it only when invoked. A real-data invocation uses
these existing public arguments after the caller has admitted the actual source:

```python
from phydrax.applications.radiation_biophysics.interchange import import_dnadamage1_root

# source: RadiationSource bound to this file's actual bytes and retained run artifacts.
# volume_materials: the retained external run's integer volume-code -> material table.
imported = import_dnadamage1_root(
    "retained-dnadamage1.root",
    source=source,
    run_id="retained-run",
    fraction_id="fraction-1",
    volume_materials=volume_materials,
    required_semantics=("event_identity", "primary_history", "deposited_energy"),
)
```

This real-data fragment has deliberate **external prerequisites**; the earlier
benchmark command is the self-contained runnable recipe. Do not substitute the
benchmark's synthetic source manifest for real-file admission. Supply the actual
checksum/size, engine/revision, configuration, frame, nm/eV units, source-table/RNG
lineage, endpoint/chemistry/scavenging metadata, and use rights. The reader expects
`ntuple/ntuple_1` and `ntuple/ntuple_2`; the [guide](../guides_radiation_biophysics.md)
lists every exact column. ROOT bytes and a retained-column derivative need separate
byte manifests and retained governing parent rights.

Requesting unavailable event time, track/parent/process/species, or carried energy
must fail rather than invent values. The chemical tree is only selected damage
reactions: it cannot qualify complete radiolysis or time-dependent G-values.
Untimed reactions cannot be filtered to another chemistry endpoint. Mapping requires
explicit source-to-target identity, contour alignment, coordinate transform, material
policy, and overlap/outside decisions. Whole-scored-mass dose must be supplied
independently of target-only energy deposition.

## Experimental completion gates

Before treating results as qualified, retain rights-cleared real provider and
experimental campaign artifacts with known uncertainty. Supply independent
`RadiationStageEvidence` for transport, chemical-G, target reactions, and lesion
yields; use experimental training and held-out lesion observations with disjoint
physical conditions and a declared acceptance criterion. Unknown reference
uncertainty is `None`, not an invented zero, and cannot support quantitative
comparison. Finite Laplace posterior intervals from a prior are not likelihood rank.

The delivered synthetic workflow and reader do not supply that corpus. No native
transport, spatial-radiolysis, repair, survival, clinical, or commercial-readiness
claim follows from a successful command. Consult the
[API](../api/applications/radiation_biophysics.md) and
[source dispositions](../biophysical_sources.md) before reusing external artifacts.
