# Biophysical sources and qualification

The delivered surfaces are [protein folding](guides_protein_folding.md),
[nucleic-acid biophysics](guides_nucleic_acid_biophysics.md),
[radiation biophysics](guides_radiation_biophysics.md), and
[single-cell systems biology](guides_systems_biology.md). They compose existing
numerical owners; there is no restored broad bioinformatics package, universal
biomolecule superclass, or second MD, SSA, BP, quantum, training, or UQ engine.

## What numerical completion establishes

A successful compiled result establishes only its declared support, model, precision,
resource bounds, and numerical acceptance criteria. It does not establish experimental
accuracy or authorize use of external inputs. Biological identities, source mappings,
units, model applicability, uncertainty, and requested-use rights remain explicit.
Unknown reference uncertainty is `None`, not a manufactured zero. Quantitative
qualification requires known uncertainty independently of rights admission.

| Capability | Executed evidence and scientific boundary |
|---|---|
| Protein physical handoff | Retained 1L2Y model 1, caller-parameterized Amber14/ff14SB through OpenMM, native conservative forces and short NVE. This is a physical handoff, not a folding-accuracy study. |
| Experimental protein inference | Synthetic joint-channel held-out prediction and identifiable/nonidentifiable parameter combinations. An admissible experimental dataset with calibrated uncertainty is still required for experimental validation. |
| Paired-state protein thermodynamics | Matched analytical replica control, correlation-aware estimation, slope and explicit reference closure. Actual equilibrated folded/unfolded ensembles and experimental closure measurements are not supplied by that control. |
| Conditional rotamer potential | Exact/tree/loopy numerical and coordinate-gradient evidence. Cleared, calibrated protein parameter tables and a meaningful mini-protein validation campaign are required before a protein-folding claim. |
| Nucleotide observations | Rigid/order invariance, sparse/dense eRMSD, torsion and pucker evidence. Published and smoothed descriptors have distinct identities; arbitrary nonideal geometry is not declared experimentally valid. |
| Chemical mapping | Actual retained RMDB observations, native fitting and withheld-construct scoring. The simple two-feature model has poor withheld agreement under the supplied errors; successful optimization does not qualify it experimentally. |
| Rigid nucleotide models | Energy/wrench, conservation, timestep and conditional kinetic-statistics controls with independently declared analytical coefficients. Published-scale duplex observables, independently cleared parameter calibration, and an experimental CG clock remain gates. |
| Secondary-structure CTMC | Detailed balance, closed-generator/transient, empirical SSA and event-hitting evidence. Nearest-neighbor tables, conditions and freely adjustable kinetic prefactors are separate source inputs; unit labels do not calibrate a clock. |
| Electronic-site execution | Analytic coherent/dephasing, density and trajectory evidence for declared site parameters and environment. Electronic populations are neither atomic charges nor lesion probabilities; environment/structure calibration is separate. |
| Single-cell scenarios | Exact piecewise-constant latent/count-assay controls and held-out count-moment prediction. Count-derived drift is not a path derivative, physical lineage, or calibrated biological clock. Conditional fit covariance is not a posterior. |
| Radiation lesions and calibration | Real optional ROOT reading of a source-pinned synthetic format fixture, history-preserving mapping/clustering, yield conventions and synthetic held-out calibration. A real external event corpus and independent transport, chemical-G, target-reaction and lesion-yield campaigns are not supplied. |
| Native coordinate proposals | Actual native training, sampling, portable-weight admission and reconstruction diagnostics. No independently qualified generative corpus, pretrained weights, equilibrium weights, or pretrained-performance claim is supplied. |
| Hybrid mechanics | Reference-conditioned Cartesian/rigid conservative coupling and explicit cross-site parameters. It does not imply de novo folding or sequence-specific recognition. |
| Co-translational activation | Complete bounded nascent-chain protocol, insertion sources, epoch rollback and replay. Codon timing, ribosome interactions and non-native folding physics require independent calibration. |

Transport and spatial radiolysis remain external by design. Native low-energy track
transport, cross-section databases, whole-nucleus geometry production, repair, survival,
and clinical prediction are outside this program, not placeholder implementations.
Smooth exogenous hazards and actual cell division likewise require their own qualified
forward models; a sampled schedule or a branch-shaped plot does not supply them.

## Retained data and honest predictive evidence

- `tests/fixtures/protein_folding/1L2Y.pdb` retains all 38 raw NMR models. The adjacent
  source record binds the CC0 source, citation, byte count and SHA-256
  `5d1bbb545a312dfff1ae1e64b6d8addecb2f561ddc4011aeb5bee9d1dfcd4438`.
- The corresponding 154-heavy-atom coordinate reconstruction used 30 fit models,
  eight correlated diagnostic models and 200 training steps. Training loss fell,
  but none of the eight retained generated proposals passed geometry qualification.
  This is a failed structural qualification, not evidence of a working folding
  predictor; the rejected samples and missing independent-corpus gates remain visible.
  Selecting a model creates a view; NMR model counts are not thermodynamic populations.
- `tests/fixtures/nucleic_acid_biophysics/TODEX_DMS_0000.rdat` retains the CC0 RMDB
  source with SHA-256
  `2a597de8277f0965543340210381b0ff6debe4f406c67a445571460c355cc2b5`.
  Negative processed reactivities, mutant constructs and supplied errors are retained.
  In the 12-training/4-withheld-construct benchmark, 540 withheld measurements gave
  RMSE 0.7701 and χ² per observation 230.30 for the simple two-feature model.
  This is evidence against adequacy of that observation model under those errors,
  not a reason to discard data, suppress uncertainty, or advertise accurate structure
  inference. The deposited pairing is a designed hypothesis, not solved pairing truth.
- Synthetic radiation columns exercise the source-pinned ledger adapter and
  downstream analysis; separate binary ROOT fixtures exercise the actual optional
  reader. Neither constitutes a real transport corpus. Synthetic fluorescence,
  thermodynamic, kinetic, mechanical and transcript controls are explicitly labelled.

Code, raw data, parameter tables, model weights, derived outputs and provider execution
have separate rights. A published equation does not automatically authorize copying a
repository's tables or training on restricted outputs. No external predictor runtime,
weights, or database is implicitly downloaded. All proposal samples and parent
restrictions remain available; confidence is provider/model-specific.

## Disposition of all assessed sources

These are conceptual references and explicit exclusions, not a statement that their
code, models, tables or datasets were imported. The same disposition applies to future
parameter admission and qualification campaigns.

| Source | Disposition |
|---|---|
| [PhyFold](https://github.com/jamshaidwarraich/PhyFold) | Reject a phenomenological PDE as molecular folding; retain observation/condition discipline in protein experiments. |
| [PMC paired-state thermodynamic method](https://pmc.ncbi.nlm.nih.gov/articles/PMC10751793/) | Explicit paired-state enthalpy estimator and experimental free-energy closure; no universal constant-heat-capacity or exclusion rule. |
| [ProteinUnfolding2D](https://github.com/KULL-Centre/ProteinUnfolding2D) | Joint temperature–denaturant observation law with shared populations and separate channel nuisances. |
| [HP_model](https://github.com/TommyGiak/HP_model) | Pedagogical combinatorial idea only; not a physical folding model or delivered core dependency. |
| [PyFolding](https://github.com/quantumjot/PyFolding) | Independently implemented named equilibrium, kinetic and transfer-matrix models. |
| [ColabFold](https://github.com/sokrypton/ColabFold) | External hypothesis/MSA/template and requested-use provenance boundary; no runtime clone or implicit service call. |
| [Upside](https://github.com/sosnicklab/upside-md) | Conditional rotamer free-energy concept on native PGM/energy owners; no copied engine or calibrated tables. |
| [SimpleFold](https://github.com/apple/ml-simplefold) | Conditional coordinate-model design input; restricted weights excluded absent authorization. |
| [CG-SimTK](https://github.com/obrien-lab/cg_simtk_protein_folding) | Reference-conditioned coarse protocols, activation and entanglement observations; not imported time calibration. |
| [JosephPB/Protein](https://github.com/JosephPB/Protein) | Defective lattice implementation rejected; no core dependency. |
| [AWS quantum examples](https://github.com/awslabs/quantum-computing-exploration-for-drug-discovery-on-aws) | No protein-folding or quantum-advantage claim; generic discrete examples do not establish either. |
| [PolyFold](https://github.com/Bhattacharya-Lab/PolyFold) | Interval-distance reconstruction and physical/chirality diagnostics through native composition. |
| [AlphaFold3](https://github.com/google-deepmind/alphafold3) | Typed external input/hypothesis/confidence boundary; no restricted weights or output distillation. |
| [SPQR](https://github.com/srnas/spqr) | Oriented base and conformer semantics; no uncalibrated replacement score presented as a validated model. |
| [oxDNA](https://github.com/lorenzo-rovigatti/oxdna) | Rigid nucleotide energy-model family and explicit condition/parameter artifacts; no copied runtime or automatically cleared calibration tables. |
| [rna-tools](https://github.com/mmagnus/rna-tools) | Explicit normalization, mapping loss and qualification; toolbox/CLI aggregation excluded. |
| [Barnaba](https://github.com/srnas/barnaba) | Base frames, published eRMSD, torsion, pucker and named geometric observations. |
| [VeloSim](https://github.com/PeterZZQ/VeloSim) | Native exact scenario, latent state and independent assay separation. |
| [VeloDyn](https://github.com/calico/velodyn) | Evidence-bound inferred field semantics; no arbitrary embedding path called physical time or energy. |
| [ATOM-1](https://www.biorxiv.org/content/10.1101/2023.12.13.571579v1.full) | Chemical-mapping supervision and qualification concepts; not an RNA-velocity source and no unavailable checkpoint dependency. |
| [dnadamage1](https://gitlab.cern.ch/geant4/geant4/-/tree/master/examples/extended/medical/dna/dnadamage1) | Source-pinned external stage/event/lesion semantics; classifier defaults are not adopted as biological truth. |
| [QuantumDNA](https://github.com/dehe1011/QuantumDNA) | Electronic-site and source-parameter compilation into native quantum execution; not quantum-computing optimization. |
| [Multistrand](https://github.com/DNA-and-Natural-Algorithms-Group/multistrand) | Secondary-state, rate and detailed-balance semantics on native CTMC execution. |
| [gMicroMC](https://github.com/utaresearch/gMicroMC) | External staged ledgers and validation requirements; no copied CUDA transport/runtime/tables. |
| [Geant4-DNA collection](https://gitlab.cern.ch/geant4/geant4/-/tree/master/examples/extended/medical/dna) | External transport/chemistry references and model-specific validation; no implied native track engine. |
| [PDBDNAConv](https://github.com/fkgw1228/PDBDNAConv) | Derived target geometry with explicit many-to-many source correspondence. |
| [ANM-oxDNA](https://github.com/sulcgroup/anm-oxdna) | Existing elastic-network protein plus independent cross-site rigid-nucleotide mechanics; reference conditioning remains explicit. |

## Reproducing numerical evidence

The application guides and cookbooks give the exact supported commands and dependency
profiles. Benchmarks use `benchmarks/_runtime.py` to separate lowering/compilation from
synchronized repeated execution where the native boundary permits it, and retain
capacity, memory, numerical errors and environment evidence. Run physical-unit
profiles with `JAX_ENABLE_X64=true`; choose a declared backend. No matched-hardware,
matched-model superiority over another engine is claimed from these controls.
