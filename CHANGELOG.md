# Changelog

## Unreleased

### Added
- `phydrax.typing.checked` checks a function's annotated arguments against their
  input contracts: nominal runtime classes, callables, and Phydrax tensor and
  metadata forms, in declaration order and one dimension `Scope` per call, after
  Python's own binding errors. Selectors, conversion inputs, scalars,
  containers, and protocols stay static-only and remain with their owner;
  values are forwarded unchanged and returns are not checked. Constructors and
  methods across the package now declare argument kinds once in their
  signatures instead of restating them in `isinstance` guards; module-level
  scientific functions keep their guards because their code is
  content-addressed. Argument-kind errors now name the function and argument
  and are raised before the body's own validation.
  `tools/audit_contract_candidates.py --signatures` reports guards that a
  checked signature has made redundant.
  The weak signature registry releases discarded dynamic owner types and
  wrapped functions instead of retaining them through cached plans.
- Meshfree examples use the canonical public discretization, coupling, and
  metric facades rather than private implementation imports.
- `PosteriorProblem` retains numerical state in prediction, observation-variance,
  observation-sampling, and Gauss–Newton residual callbacks as dynamic PyTree
  leaves. Array-bearing bound methods no longer trigger a static-array warning,
  and updated callback data remains visible under compiled evaluation and
  differentiation.
- Prepared meshfree approximation in `phydrax.discretization.meshfree`: bounded
  certified neighborhoods, batched GMLS/PHS-RBF-FD, enforced stencil admission,
  sparse point-cloud elliptic solves, explicit hyperviscosity, and native
  meshfree Galerkin hierarchy preparation.
- Conservative graph metric preparation, intrinsic surface approximation,
  extensive moving-surface IMEX execution, tangential shifting, deterministic
  resampling, measured conservative/positive epoch transfers, capacity maps,
  and native bulk–surface exchange publication.
- Certified local constitutive edge laws, feature-coverage evidence, MODEL-bound
  implicit nonlinear solves/training, and phase-separated meshfree qualification
  drivers. Numerical implementation and scientific release remain distinct;
  meshfree profiles are unreleased candidates.
- Native bounded sparse row-rank profiles and projected coarse pseudoinverse
  preconditioning. Sparse homogeneous conic directions remain matrix-free and
  terminate against original-coordinate KKT/complementarity evidence.
- Clean meshfree API cutover: local stencil policies/functionals replace the old
  stencil wrapper, `PointDiffusionOperator` replaces dense dissipative matrices,
  and `PointCloudPoissonPlan` replaces the dense Poisson helper. Unused point
  conormal/distribution metadata and compatibility names are removed.
- Negative-offset positive transforms are refused when they cannot certify
  input-convex constitutive potentials.
- Native exterior calculus in `phydrax.exterior`: explicit form degree, twist, fiber,
  and proxy semantics; shared exterior algebra; de Rham integration, Whitney chains,
  induced boundary traces, numerical form products, and coefficient systems.
- Paired `HilbertComplex` and `ComplexMap` substrates, compound matrices with
  singular-minor derivatives, prepared Hodge decomposition, harmonic evidence,
  coordinate weak forms, mixed Hodge–Laplace recipes, and auxiliary-space solvers.
- Compatible finite-element form families and reconstruction, public spline complexes,
  Fourier and spherical complexes, and canonical AMR transfer evidence.
- Forms-aware PDE validation, serialization, tokens, and smooth/discrete lowering,
  with executable Hodge–Laplace, cavity, learned-Hodge, and solenoidal-field examples.
- Phase-separated exterior, cochain, FE, Maxwell, spectral, spline, MAC, and PIC
  benchmarks with compiler-memory and retained-state evidence.
- Numerical interoperability across methods and pipelines
  (`docs/guides_numerical_interoperability.md`). Existing scientific owners
  stay authoritative; prepared bindings connect them.
  - Interface bindings: `phydrax.meshing.MeshInterfaceAttachment`
    (certified `GeometryAssociation` witness and B-Rep normal side),
    `MeshAssembly.require_attachment`, `GeometryAssociation.target_rows`,
    `SubdomainCover.revision`, and `phydrax.solver.coupling.InterfaceBinding`,
    `InterfaceEndpoint`, `InterfaceSource`, `PairedSupportAttachment`,
    `SheetViewAttachment`, `InterfaceIncidence`, `EmbeddedMeaning` for
    two-sided, junction, overlap and embedded incidences with revision,
    orientation and witness checks.
  - Prepared field queries and side actions: `PreparedFieldQuery`
    (`PreparedFieldReconstruction.prepare_query`), `PreparedTraceAction`,
    `PreparedFluxAction`, `SideActionDescriptor`, `SideGatherRoute`,
    `BoundaryImposition`, `PreparedNonlinearFaceTrace`, with providers for
    FE/SEM, explicit polygon H1, VEM (exact edge traces versus labeled
    projected interiors), IGA (`iga.prepare_isogeometric_field_reconstruction`,
    public-layout side traces), FD/SBP (`SBPGridNorm`, `SBPClosureEvidence`,
    `SBPNormKind`, `SBPNormLayout`), global spectral (`SpectralFaceRoute`),
    finite volumes (cell-average and face-state representations) and point
    clouds; boundary trace-space capabilities `BoundaryTraceSpaceCapability`
    and `CauchyTraceCapability` for 2-D/3-D scalar Galerkin, RWG and
    Buffa–Christiansen spaces.
  - Native 2-D Laplace Galerkin boundary operator on closed straight-panel
    polygons: `ClosedPolygonalCurve2D`, `prepare_scalar_laplace_galerkin_2d`
    (weak V and K, P1/DP0 trace spaces, exterior relation, per-class
    singular quadrature evidence), `prepare_exterior_laplace_dirichlet_2d`,
    `solve_exterior_laplace_dirichlet_2d`,
    `prepare_boundary_trace_projection_2d`.
  - Spatial coupled problems in `phydrax.solver.coupling`:
    `CoupledProblemPlan`, `prepare_coupled_problem`,
    `PreparedCoupledProblem`, `solve_coupled_problem`, `CoupledSolution`
    (original component residuals and interface defects certified; native,
    accepted and derivative validity reported separately);
    `VariationalComponent`, `GalerkinBoundaryComponent` (declared
    `far_field="bounded"|"decaying"`), `ReducedComponent`,
    `ScalarLaplaceFEMBEMComponent`, `ElasticityFEMBEMComponent`;
    `ScalarTransmissionLaw` with `MatchingElimination`, `MortarImposition`
    and `NitscheImposition` (certified trace-inverse constants via
    `certify_trace_inverse`, `CompiledFiniteElementProblem.prepare_pointwise_flux`
    and `certify_flux_stability`); `BoundaryIntegralTransmissionLaw`
    (bordered Johnson–Nédélec); `ConservativeFluxLaw`, `IntegralPortLaw`,
    `FieldTransferLaw`, `MonotoneInterfaceConductance`; exact 2-D common
    refinement `prepare_interface_quadrature`; coupled kernel detection
    through law-owned unknowns with `CoupledGauge`.
  - Spectral-element, virtual-element and boundary-element transmission in
    one coupled solve (`examples/sem_vem_bem_transmission.py`) with FE,
    polygon and IGA substitutions under the same law.
  - Parameters, observations and derivatives: `ParameterBinding`,
    `RuntimeInput`, `FieldPointObservation`, `FieldBoundaryObservation`,
    `FieldFluxObservation`, `PreparedCoupledProblem.bind_arguments` and
    `derivative_capability`, `phydrax.OwnerDerivativeCapability`; exact RHS,
    boundary-data and coefficient derivative routes for the 3-D scalar and
    elasticity FEM–BEM products and the 2-D exterior solve.
  - Temporal coupling: `CouplingMeasurement` inventory functionals,
    `CouplingTemporalConversion`, native participants
    `FixedStepCouplingParticipant`, `DAECouplingParticipant`,
    `SteadyResponseCouplingParticipant`, `PartitionedCouplingDeclaration`,
    `lower_partitioned_coupling`, `CouplingStatus.UNRELIABLE_ERROR_ESTIMATE`,
    `FixedPointProblem(evaluation_work=True)` and
    `NonlinearResult.evaluation_work`; host execution for FMI participants
    (`prepare_host_coupling`, `advance_host_coupling_window`,
    `solve_host_coupling`, FMI2 `FMICouplingBinding`,
    `FMICouplingParticipant`, `FMIVariableBinding`, `FMIUnit`,
    `fmi_unit_definition`).
  - Block coordinates and transient systems: `BlockSelection`,
    `CoordinateBlock`, `BlockRestrictionLinearOperator`,
    `BlockProlongationLinearOperator`, `MappedBlockLinearOperator`,
    `select_block_operator`, `assemble_block_operator`,
    `phydrax.solver.DAECoordinateAdapter`, exact static condensation and
    coupled transient problems.
  - Consumers: `prepare_coupled_transition`, `PreparedCoupledTransition`,
    `CoupledObservationPort`, `phydrax.stochastic.CoupledTransitionKernel`,
    `phydrax.optim.StateDesignComponentAdmission`.
  - Lifecycle: `phydrax.lifecycle` composition rebind transaction
    (`Composition`, `CompositionEntry`, `CompositionTransport`,
    `CompositionRebind`, `commit_composition_rebind`) with owner hooks for
    FE, FV, topology epochs, coupling, restart, distribution, training and
    worksets; `PreparedPlateauBorder.reprepare_after_events`,
    `multiregion_topology_epoch`.
  - Execution: bounded coupled lane worksets (`CoupledExecutionPolicy`,
    `PreparedCoupledExecution`, `LaneWorkset`, `LaneWorksetEstimate`);
    PIC capability records (`PICFieldSolverCapability`,
    `PICCapabilityRecord`, `PICFieldSolverCapabilities`),
    `pic_distribution_support`, `hand_off_pic_state`.
  - Declared implicit derivative route
    `LinearDerivativeSolvePolicy(route="primal-factors")`
    (`DerivativeSolveRoute`): tangent and adjoint solves reuse the primal
    DenseLU/DenseCholesky factors with residual acceptance; the default
    `"krylov"` route is unchanged.
  - Measurement noise: `restrict_observation_covariance`,
    `MeasurementNoiseModel`, `MeasurementWhitening`.
  - Qualification runner `tools/numerical_interoperability_qualification.py`
    and benchmark `benchmarks/numerical_interoperability.py`.
  - Examples: `numerical_interface_binding`, `prepared_field_observations`,
    `exterior_laplace_galerkin`, `coupled_scalar_regions`,
    `nitsche_transmission`, `sem_vem_bem_transmission`,
    `coupled_inverse_problem`, `coupled_learned_interface`,
    `mixed_method_time_coupling`, `named_block_dae`,
    `coupled_transient_fields`, `coupled_control`,
    `coupled_data_assimilation`, `coupled_rom_swap`,
    `hybrid_pinn_classical`, `adaptive_fe_fv_rebind`,
    `foam_junction_rebind`, `fmi_host_coupling`, `coupled_lane_worksets`,
    `pic_field_solver_substitution`, `pic_field_handoff`.
- Full-wave FEL `phydrax.applications.accelerator.fel.FELFullWavePlan`
  (`FELFullWaveBeam`, `FELFullWaveSeed`, `FELFullWaveTracks`,
  `FELFullWaveHuygens`, `FELFullWaveResult`, `FELFullWaveLedger`,
  `FELFullWaveEvidence`, `FELFullWaveFrameEvidence`, `FELFullWaveStatus`,
  `PreparedFELFullWave`): boosted-frame staggered PSATD PIC with a gather-only
  undulator, PML-open boost axis, a B5 antenna seed boosted onto the grid,
  A1 track radiation or boosted Huygens far fields, and a lab ledger of beam,
  grid-field, PML-escaped, and antenna-injected energy. Seeded small-signal
  gain matches `FELPlan` within 5 % (+3.6 %), spontaneous Huygens spectra match
  A1 times the track's spline deposit factor within 6 %, prebunched emission
  matches KMR within 3 % and scales as `(I J₁(a))²`, and the boosted run
  matches a lab-frame PIC run within 5 %. Example
  `examples/free_electron_laser_full_wave.py`, benchmark
  `benchmarks/fel_full_wave.py`.
- Cartesian PSATD drives the B5 `SampledPlaneCurrentAntennaPlan`:
  `SpectralMaxwellPlan(..., antennas=(...))` prepares each plan as
  `phydrax.solver.maxwell.spectral.PreparedSpectralPlaneAntenna`, electric
  and magnetic TFSF sheet currents (with the convective terms of a sheet
  moving along its normal) on a band-limited normal delta, flat to a quarter of
  the Nyquist wavenumber and empty above half of it. Declared sheet charges live
  in `SpectralMaxwellState.antenna_charge`/`antenna_magnetic_charge`;
  `SpectralMaxwellDiagnostics.antenna_work` is each antenna's work on the
  field. `BoostedFramePlan.boost_antenna` output now seeds boosted PSATD runs
  (stationary on the Galilean grid). The exact propagators
  (`exact_interval`, `window_integral`, `apply_source`) take a magnetic
  current, and `SpectralOperators.magnetic_charge_field` inverts `∇⁺·B`.
  Measured: one-way ratio below `1e-12`, antenna work equal to the injected
  energy to `3e-14`, β = −0.9 Doppler centroid within `6e-5`, Gaussian waist
  within `0.6%` and Gouy phase within `5e-3` rad, and a γ_b = 2 LWFA seeded by
  a boosted antenna within `0.3%` of the lab gain with no NCI rejection.
- Pinned external provider oracles for the radiation routes, placed next to
  the route each one checks. Each adapter builds its input deck from a Phydrax
  plan and runs a caller-pinned executable through `run_pinned_command` with
  byte-capped file artifacts. Output comes back in plan units through the
  plan's `ElectromagneticScaleContract`. Each result reports the provider
  version, executable and output digests, the license, and an `AdapterReport`
  of declared losses. Configurations outside the supported subset are refused
  before the provider runs.
  - SRW: `run_srw`, `srw_input`, `read_srw_output`, `SRWProvider`,
    `SRWFieldMapSource`, `SRWFieldRepresentation`, `SRWFieldSpectrum`, and
    `SRWSpectrumResult` (`phydrax.electromagnetics`). SRW computes
    single-electron spectra from A1 trajectories or X1a field maps, returned
    in the A1 convention.
  - UFGC and Symphony: `run_ufgc`, `ufgc_input`, `read_ufgc_output`,
    `UFGCProvider`, `UFGCCoefficients`, `UFGCResult`, `run_symphony`,
    `symphony_input`, `read_symphony_output`, `SymphonyProvider`,
    `SymphonyCoefficients`, and `SymphonyResult`. They compute C2 emission and
    absorption coefficients. Both codes are GPL, so each runs only through a
    pinned interpreter plus a digest-pinned library or module.
  - WarpX, Smilei, and PIConGPU (`phydrax.solver`): `PICOracleScenario`,
    `PICOracleCode`, `pic_oracle_case`, `PICOracleCase`, `PICOracleSpecies`,
    `PICOracleSpectralSolver`, `PICOracleFields`, `PICOracleResult`,
    `WarpXProvider`, `warpx_input`, `run_warpx`, `read_pic_oracle_openpmd`,
    `SmileiProvider`, `smilei_input`, `run_smilei`, `read_smilei_fields`,
    `PIConGPUProvider`, `picongpu_input`, `run_picongpu`, `PICOracleTrack`,
    `PICOracleTrackResult`, `run_warpx_track`, `read_pic_oracle_track`,
    `PICOracleLaser`, `PICOracleWakefieldCase`, `pic_oracle_wakefield_case`,
    `PICOracleModalFields`, `PICOracleWakefieldResult`,
    `warpx_preroll_steps`, `run_warpx_wakefield`, and `read_warpx_wakefield`.
    Each scenario is derived from the plan passed in:
    - `periodic-yee-plasma` (WarpX, Smilei, PIConGPU) and
      `periodic-psatd-plasma` (WarpX) come from the fully periodic
      `ElectromagneticPICPlan`.
    - `single-particle-radiation` (WarpX) takes one recorded particle in a
      uniform static external field. Its track enters through the openPMD
      particle reader as a `ChargedTrajectory` for A1.
    - `laser-wakefield-stage` (WarpX RZ PSATD) comes from the
      quasi-cylindrical P3 plan. Its mode-resolved fields enter through the
      openPMD `thetaMode` reader.

    WarpX and PIConGPU fields are imported through the openPMD mesh reader.
    PIConGPU runs through a pinned per-setup build driver.
  - Puffin: `run_puffin`, `puffin_input`, `read_puffin_power`,
    `PuffinProvider`, `PuffinGaussianSeed`, and `PuffinResult`
    (`phydrax.applications.accelerator.fel`). Puffin runs the 1-D
    unaveraged FEL against `FELTimeDependentPlan`.
  - elegant: `run_elegant_csr`, `elegant_csr_input`, `elegant_csr_bunch`,
    `ElegantCSRProvider`, and `ElegantCSRResult`
    (`phydrax.applications.accelerator`). elegant runs `CSRCSBEND`/`CSRDRIFT`
    tracking against the X4 1-D models.
  - Geant4 through geant4_pybind: `run_geant4_shower`,
    `geant4_shower_input`, `read_geant4_shower`, `Geant4ShowerResult`,
    `run_geant4_cherenkov`, `geant4_cherenkov_input`,
    `read_geant4_cherenkov`, `Geant4CherenkovResult`, `run_geant4_optical`,
    `geant4_optical_input`, `read_geant4_optical`, `Geant4OpticalResult`,
    `Geant4Provider`, and `Geant4ElectromagneticPhysics`
    (`phydrax.applications.detector`). Geant4 computes M1 shower depth
    profiles, M2 Cherenkov yield, cone, and polarization, and M2 transport of
    optical photons through planar polished UNIFIED stacks with absorption and
    Rayleigh scattering. In Geant4 11.4.p01 the in-plane polarization
    component changes sign at every dielectric crossing. The adapter reports
    this as a declared loss and does not correct it.

  New interchange catalog formats: `sdds` (binary elegant particle files,
  read and write), `puffin-integrated`, and `smilei-fields` (both read-only).
  openPMD `fileBased` series now accept the zero-padded `%0<N>T` iteration
  placeholder that WarpX and PIConGPU write; exactly one placeholder is
  required. The `thetaMode` mesh reader also accepts `axisLabels`
  `("z", "r")` (WarpX RZ) and transposes those records to `[mode, r, z]`.
  Every tiny reference output under `tests/data/providers/` was
  produced by the real provider, and its provenance is recorded next to it.
  Validated releases, licenses, and environment variables are listed in
  `docs/guides_charged_particle_radiation.md#external-provider-oracles`.
- Cross-route consistency release matrix for charged-particle radiation.
  `tests/integration/test_radiation_cross_route.py` compares independent
  routes on shared scenarios at two or more resolutions. Each tolerance is set
  from a measured error coefficient. The rows are: PIC-tracked electron
  through `PICTrackRecorder` and A1 against A1 on the exact helix (A1 ↔ P);
  one axisymmetric TM pulse on the cochain, Cartesian-PSATD, and
  quasi-cylindrical field solvers against the closed form (P1 ↔ P2 ↔ P3);
  nonlinear-Compton energy loss against classical Landau–Lifshitz as `χ → 0`
  (Q1 ↔ Q2); thermal cyclotron harmonic emissivity against Jüttner-averaged A1
  helices (C2 ↔ A1); and Cherenkov photon yield against the moving-charge
  Poynting flux over `ħω` (M2 ↔ B2). The A1 ↔ B1, B2 ↔ B4, X4 steady ↔
  retarded-mesh and X5 ↔ X6 rows reference their milestone tests. The new
  candidate profile `radiation.cross-route-release-matrix`
  (`phydrax.qualification`) names every row's test node IDs as required gates
  and depends on the member route profiles. New candidate profiles:
  `electromagnetics.cold-plasma-dielectric`, `accelerator.wake-impedance`,
  `accelerator.space-charge-igf`, and
  `optics.transport.optical-photon-transport`. The guide section is
  `docs/guides_charged_particle_radiation.md#cross-route-release-matrix`.
- Relativistic self-field initialization for electromagnetic PIC.
  `ElectromagneticPICPlan.initialize(..., self_fields="relativistic-per-species",
  drifts=None)` gives every species drifting along a grid axis (explicit drift or
  the mean velocity of its active particles) the lab-frame field of its
  rest-frame Coulomb field: `−∇·(ε(∇⊥ + γ⁻² ê∥∂∥)φ) = ρ` with the native PCG
  path, `E = −(∇⊥φ + γ⁻² ê∥∂∥φ)`, `B = d(βφ/c) = β × E/c`, superposed over
  species. New optional field-solver protocol `PICRelativisticSelfFields`
  (implemented by `CochainMaxwellPICFieldSolver`) returns
  `PICRelativisticFieldResult` with the Gauss residual and `|∇·B|` (zero by
  construction). A γ = 10 Gaussian bunch matches the boosted analytic field to
  1.8% at 32³ with `B = β × E/c` to 3.5%, and radiates 1.4% of the energy
  through surrounding planes that a static `B = 0` start radiates.
- Lorentz-boosted-frame PIC. `phydrax.solver.BoostedFramePlan(frame,
  lab_domain)` binds a pure boost `LorentzFrame` along one grid axis to a lab
  `phydrax.geometry.Box`: exact event, field, and proper-velocity transforms;
  `boost_particles` (lab-resting plasma becomes density `γ_b n` drifting at
  `−v_b`, beams get the velocity addition and a ballistic drift onto the
  boosted slice); `boost_external_field` (`BoostedExternalField`, gather-only,
  never deposited; a boosted static undulator is a sub-luminal magnetic-type
  field); `boost_antenna` (moving `SampledPlaneCurrentAntennaPlan`).
  `prepare(pic, *, snapshots, nci_...)` returns `PreparedBoostedFrame` over a
  Galilean (comoving with the boosted plasma) or standard
  `PreparedSpectralMaxwell`: lab vacuum fields (lasers) sampled on the initial
  slice and Gauss-projected, a mandatory high-`|k|` NCI guard that rejects
  steps, back-transformed lab field and particle snapshots
  (`BoostedSnapshotPlan`) in a bounded ring, lab-frame track conversion with
  per-lane lab times for trajectory radiation (`lab_trajectory`), boosted
  Huygens far fields relabeled into the lab only on standard grids with zero
  surface current in vacuum (`lab_far_field`), and a `"boosted-frame"` restart
  component. Refusals: rotated or oblique frames, non-comoving Galilean
  velocities, grids short of the contracted lab domain, unboosted external
  fields, recorders streaming boosted-frame radiation, non-vacuum media,
  overflowing snapshot rings, plasma on a standard grid, vacuum fields over
  particles. Qualification candidate `pic.boosted-frame`; example
  `examples/boosted_frame_lwfa.py`; benchmark `benchmarks/boosted_frame.py`.
- Open, dispersive, and magnetized self-consistent electromagnetic PIC.
  `CochainMaxwellPICFieldSolver` accepts bounded axes, CPML absorbers,
  PEC/PMC/impedance boundaries, and every linear passive medium (lossy
  conductors, Lorentz–Drude electric and magnetic poles, magnetized cold
  plasma); nonlinear or active media and an electrostatic permittivity that
  differs from the medium's instantaneous response are refused. Grounded
  (Dirichlet) initialization puts the induced wall charge on the fixed
  vertices. New optional protocols `PICEnergyAccounting` (`PICFieldEnergy`
  split into electric, leapfrog-corrected magnetic, and medium energy plus the
  source-free loss power) and `PICOpenDomain` (periodic/bounded axes, box, and
  per-species wall inset of the spline stencil). `PICEnergyLedger` gains
  `material`, `dissipated` (trapezoidal conduction, pole damping, collision,
  impedance, and CPML loss), and `exited`; `ElectromagneticPICDiagnostics`
  gains `exit` (`PICExitLedger`: absorbed charge, mass, kinetic energy),
  `medium_charge`, and `charge_ledger_defect`;
  `ElectromagneticPICPlan.synchronized_energy` returns a `PICEnergySnapshot`
  at the state's integer time. `PICBoundaryKind.PERIODIC` marks periodic
  particle axes; particle faces are validated against the field box.
  `PreparedCompatibleMaxwell.leapfrog_energy` and `loss_power` (moved from the
  prescribed-charge runtime). Qualification candidate
  `pic.dispersive-self-consistent`; example
  `examples/dispersive_pic_cherenkov.py`.
- Quasi-cylindrical spectral PIC. `QuasiCylindricalMaxwellPlan` on a
  `phydrax.discretization.pic.QuasiCylindricalGrid` advances azimuthal modes
  `m = 0, …, M` whose circular components are transformed by
  `phydrax.discretization.SharedGridHankelPlan` (orders `m − 1, m, m + 1` on
  one k-grid from the zeros of `J_m`, pseudoinverse analysis with rank and
  residual evidence) into the Cartesian-equivalent spectrum, so the P2
  standard, Galilean, and averaged-Galilean PSATD propagators apply exactly
  (constant-J; `"update-with-rho"` or `"spectral-correction"`,
  `"vay-deposition"` refused). `AzimuthalTransferPlan` deposits and gathers
  modes with `e^{−imθ}` weights (×2 for `m > 0`), mirror folding, and
  near-axis-corrected volumes for shapes 1–3. The prepared solver is a PIC
  field solver with `PICSpectralSymbol`, `PICHuygensSampling` (closed
  cylinders, standard variant, no antennas), `PICWindowShift` along `z` with
  a spectral Gauss projection, `PICGalileanGrid`, `PICRestartState`, and
  `PICGaussProjection`; `RadialDampingPlan` absorbs in an outer radial layer
  without touching the divergences, and `QuasiCylindricalAntennaPlan` injects
  per-mode one-way sheet currents with declared sheet charges
  (`phydrax.optics.wave.pulse_envelope_quasi_cylindrical_antenna` builds it
  from a `PulseEnvelopeField`). `fbpic_laser_wakefield` runs a pinned
  external FBPIC interpreter (`FBPICProvider`) as a laser-wakefield oracle;
  the on-axis wake matches FBPIC 0.27.0 to 5% relative `L²`. Benchmark
  `benchmarks/quasi_cylindrical_pic.py`.
- Distributed explicit electromagnetic PIC and lifecycle restart.
  `DistributedPICFieldSolver(base, mesh, guard_cells=, particle_margin=,
  identity_tiles=)` runs the periodic 3-D cochain, reduced 1-D/2-D, or
  Cartesian PSATD solver over a static block decomposition of the leading grid
  axes on a one- to three-axis device mesh (slabs, pencils, blocks;
  `phydrax.discretization.pic.PICDomainDecomposition`) aligned with per-device
  particle slot blocks; per-device deposits are halo-accumulated axis by axis
  through one plane-level `DistributedHaloPlan` per decomposed axis (a uniform
  mean-current mode is summed), gathers read owned cells plus guards exchanged
  axis by axis (edge and corner guards included), and preparation refuses
  guards narrower than the transfer footprint or non-window-local deposits
  (the continuity-projected reduced 2-D current is not window-local).
  Global-FFT PSATD runs its slab or pencil transforms on the PIC mesh;
  local-guarded PSATD (Kirchen et al. 2020) runs per device with the plan's
  guard cells exchanged through the halo substrate and local finite-order
  transforms of each device's subdomains (Vay deposition or update-with-ρ),
  with `PreparedSpectralMaxwell.guard_truncation(dt)` as stencil-truncation
  evidence; `PreparedSpectralMaxwell.guarded_update`/`complete_advance` and
  `SpectralLocalUpdate` split the PSATD step into its device-local update and
  its evidence. It shares its base solver's identity, implements
  `PICGaussProjection` through its base, and reports `PICDistributedEvidence`
  (`distributed=True`, mesh shape, spectral guards; cochain Maxwell
  capabilities with `distributed` and `spatial_distribution`).
  `DistributedElectromagneticPICPlan` initializes with caller-order
  identities, installs a `DistributedPICExecutor`
  (`ElectromagneticPICPlan.executor`, a new `AbstractPICParticleExecutor`
  hook that also exchanges particles before the population stage;
  `ElectromagneticPICDiagnostics.exchange`), and migrates particles with their
  slot-aligned process state through fixed-capacity `ppermute` packets to every
  mesh neighbor including diagonals (`PICMigrationPlan`,
  `PICMigrationEvidence`, `PICSlotGroup`), rejecting the whole step on any
  packet, reach, or slot overflow (`PICRejectionReason.MIGRATION`).
  Creation-stage, population-stage, stateful, and charge-redistributing
  processes implement the new `PICDistributedProcess` protocol
  (`PICProcessStatePartition`, `PICProcessBank`) — the QED cascade with
  polarization, particle merging and splitting, field and impact ionization —
  and run per device; they allocate through `allocate_particles`
  (`PICProcessContext.allocator`, `AbstractPICParticleAllocator`), and the
  distributed `PICIdentityAllocator` gives created particles
  decomposition-independent identities without communication (per-call
  identity reservations striped by identity tile), so `N`-device cascades and
  resampling reproduce one-device identities, lineage, and counts bitwise. The
  ionization plans' `apply` gain an optional `allocator` keyword. The previous
  refusals of local-guarded PSATD and of creation, stateful, and
  charge-redistributing processes are removed. `PICRestartPlan` composes the
  per-component restart leaves (species with identities, field with absorber
  and material memory, boundary ledger, recorders, process states, field
  history, moving-window epoch, auxiliary `PICRestartState` owners) into a
  `PICRestartManifest` published and restored through lifecycle
  addressable-shard checkpoints: same-topology restarts continue bitwise; a
  different mesh repartitions particles, slot-aligned process state, and
  process banks exactly and is a `"tolerance"` restart admitted by
  `TopologyRestartPolicy`. Benchmark `benchmarks/pic_distributed.py` covers
  slabs, blocks, and local-guarded PSATD. Candidate qualification profile
  `pic.distributed`.
- Polarization- and spin-resolved strong-field QED. `QEDPolarizationModel`
  (`"unpolarized"`, `"photon-polarized"`, `"spin-and-photon-polarized"`) on
  `NonlinearComptonPlan` and `NonlinearBreitWheelerPlan` selects the Seipt–King
  LCFA rates in the spin-quantization-axis scheme: `QEDTable(..., polarization=
  "positive" | "negative")` (`QEDTablePolarization`) tabulates the polarized
  components whose average is the unpolarized table; `channel_spectrum` exposes
  the four spin/photon channels of each process; leptons carry identity-keyed
  rest-frame polarization vectors (`QEDSpinState`) with spin-flip sampling and
  the no-emission evolution, reproducing Sokolov–Ternov polarization
  (`8/(5√3)`); photons carry linear Stokes parameters rotated to the local
  field basis, weighting pair creation and evolving by vacuum dichroism, and
  pairs are created with sampled spins. `QEDCascadeState` gains `polarization`
  (`QEDPolarizationState`, `None` when unpolarized) with escaped Stokes
  histograms; `QEDCascadeProcess` gains `magnetic_moment_anomaly`
  (`ELECTRON_MAGNETIC_MOMENT_ANOMALY`, CODATA 2022), `polarization_reference`
  and `polarize`. `RelativisticPushPlan.precess` integrates
  Thomas–Bargmann–Michel–Telegdi precession consistently with the Boris, Vay
  and Higuera–Cary pushes. `PICProcessContext` gains `pusher`,
  `step_start_proper_velocity` and `grid_velocity`.
  `phydrax.special.synchrotron_h(x) = x K_{1/3}(x)`. `tools/qed_tables.py`
  builds and verifies all six tables. Qualification candidate
  `pic.polarized-qed`. The unpolarized model keeps its arithmetic, fingerprints
  and random streams; polarized tables, plans and processes carry their
  polarization in their fingerprints.
- Creation-stage PIC processes run on Galilean field grids: the refusal of a
  nonzero `PICGalileanGrid.grid_velocity` is lifted, and the QED cascade drifts
  its photons at `c k̂ − v_grid` in grid coordinates.
- Frequency-domain moving charges in media. `phydrax.electromagnetics`
  `UniformMotionFieldPlan` (with `UniformMotionMedium`, `UniformMotionField`,
  `UniformMotionEvidence`, `UniformMotionGeometry`) evaluates the exact
  `exp(-iωt)` field of a point (`K₀`, `K₁` of `sρ`) or line (`exp(−s|η|)`) charge
  in uniform motion through a homogeneous isotropic dispersive passive medium,
  with the passive branch `Im k_ρ ≥ 0` (outgoing `H⁽¹⁾` above the Cherenkov
  threshold, reversed phase in lossless negative-index media) and the radial
  spectral energy flux. `phydrax.solver.maxwell` `MaxwellMovingChargePlan`
  integrates `q v̂ exp(iωs/v)` exactly on Whitney edge and node forms (discrete
  continuity to roundoff; periodic paths require `ωL/v ∈ 2πℤ`), and
  `FrequencyMovingChargePlan` solves `SourceFormulation`
  `"total-field"`/`"scattered-field"` (analytic incident field of a declared
  homogeneous background; contrast and conductor surfaces radiate) with the
  Krylov or sparse-direct `FrequencyMaxwellOperator`, stretched coordinates, and
  `MaxwellBoundaryPlan` conductors, reporting `FrequencyMovingChargeEvidence`
  (branch, `γβλ` bound-field reach vs absorber clearance, continuity, source
  distance, open endpoints, convergence). `fourier_modal.MovingLineChargeSource`
  (Bloch wavevector `(ω/v) d̂ + k_⊥ ê_⊥`), `moving_line_charge_excitation`, and
  `MovingPointChargeQuadrature`/`MovingPointChargeIntegral` (point charges by
  cosine-mapped Gauss–Kronrod `k_⊥` panels with `Γ = √(ω²/(β²γ²c²) + k_⊥²)`
  truncation and Kronrod–Gauss evidence). Validated against Frank–Tamm, the
  2-D line-charge Cherenkov flux, below-threshold nulls, Ginzburg–Frank
  transition radiation, the Smith–Purcell relation, and published SI
  Smith–Purcell energies (Szczepkowicz, Schächter & England 2020). Examples
  `examples/maxwell_cherenkov_frequency_domain.py` and
  `examples/smith_purcell_fourier_modal.py`; qualification candidate
  `electromagnetics.frequency-moving-charge`.
- Strong-field QED Monte Carlo and cascades in PIC
  (`phydrax.discretization.pic`): `QEDTable` (host-prepared, fingerprinted
  nonlinear Compton `K(χ)`/`P(χ, ξ)` and Breit–Wheeler `T(χ_γ)`/`P(χ_γ, ξ)`
  from the F5 synchrotron kernels with quadrature, interpolation, monotone-CDF
  and truncation evidence), `NonlinearComptonPlan` (`QEDEmissionModel`
  `"lcfa"`/`"improved-lcfa"` after Di Piazza et al. 2019; optical-depth Monte
  Carlo with a per-particle event-probability cap and bounded adaptive
  subcycling; identity-addressed `(step, id_hi, id_lo, event)` randomness;
  declared `QEDConservation` with the `O(m²c⁴/ε)` defect handed to the field),
  `NonlinearBreitWheelerPlan`, `QEDPhotonSpeciesPlan` (ballistic photon bank
  with F6 identities, escape box and polar-angle × energy histograms), and
  `QEDCascadeProcess` (creation-stage process: Compton emission then pair
  creation into electron/positron species with lineage before drift and
  deposit; atomic capacity refusal; occupancy merge requests; per-event LCFA
  validity, infrared, one-step trident and photon splitting flags in
  `QEDEventFlag`/`QEDCascadeEvidence`). `tools/qed_tables.py` generates,
  records and bitwise-verifies table sets. Guide
  `docs/guides_strong_field_qed.md`, example `examples/pic_qed_cascade.py`,
  benchmark `benchmarks/pic_qed_cascade.py`, qualification candidate
  `pic.qed-cascade`.
- PIC process capabilities: the `"creation"` process stage (after momentum
  processes, before the drift, with pointwise charge preservation verified by
  redeposition), stateful processes (`AbstractPICProcess.stateful`,
  `initialize_state`, `shift_frame`, `PICProcessContext.state`,
  `PICProcessResult.state`, restart components `process/{index}`), field
  probes (`PICFieldProbe`, `PICFieldProbeSample`, `field_probe`), and
  `PICProcessRadiation.field_exchange_energy`/`created_rest_energy`.
  `ParticleMergePlan(minimum_occupancy=...)` gates merging on a species'
  occupancy (the QED merge trigger).
- Prescribed moving charges in compatible Maxwell media
  (`phydrax.solver.maxwell`): `PrescribedChargeTrajectory` (positions on the
  uniform Maxwell step grid), `PrescribedChargeCurrentSourcePlan` (certified
  electric support so Huygens boxes admit it), `PrescribedChargeMaxwellPlan`,
  `solve_prescribed_charge_maxwell`, `PrescribedChargeMaxwellResult`,
  `PrescribedChargeEvidence`, `PrescribedChargeStatus`. Coincident-neutral
  start (static compensating charge, zero initial field), freeze after the last
  sample or a boundary exit, one `lax.scan` with continuity, Gauss (against the
  prescribed charge), magnetic, support-leak, and exit evidence and a power
  ledger of leapfrog energy, source work, and the runtime's semi-discrete
  losses. Any linear runtime: dispersive, lossy, heterogeneous, magnetized
  plasma, negative-index, CPML, PEC/PMC/impedance. Qualification candidates
  `electromagnetics.moving-charge-cherenkov`, `-transition`, `-smith-purcell`.
  Example `examples/cherenkov_prescribed_charge.py` (CFL ≤ 0.45), benchmark
  `benchmarks/prescribed_charge_maxwell.py` (`--cfl`, default 0.225); both
  close the CPML power ledger. Checked against B2b's
  `FrequencyMovingChargePlan`: Cherenkov and Smith–Purcell Bloch harmonics
  (fields and spectral Poynting per order) agree at second order in Δt, the
  time-domain Smith–Purcell orders obey `λ = (d/n)(1/β − cos θ)` on the Yee
  branch, and slit diffraction radiation matches the scattered-field
  reference and the Kazantsev–Surdutovich two-edge limit.
- `ChargeConservingCurrentPlan` supports nonperiodic axes: paths are clipped at
  the closed box and `PICCurrentDepositResult` gains `deposited_end` and
  `boundary_exit`; the continuity success scale includes the `ε|ρ|/Δt`
  roundoff floor. Spline orders refuse stencils beyond an open face.
- `MaxwellBoundaryPlan(kind, support=mask)` applies PEC/PMC/impedance to an
  explicit interior entity mask (conductor plates, gratings, screens); Huygens
  boxes refuse boundary-constrained surface entities.
- Compatible Maxwell tracks every non-curl magnetic forcing (source magnetic
  currents, CPML stretching, magnetic conductivity) as declared magnetic charge,
  so full-3-D CPML and magnetically conductive runs elide the global magnetic
  projection (previously the per-half-step projection solve failed); only PMC
  still projects. Both magnetic half kicks and the electric kick now see
  boundary-constrained fields, making PEC/PMC leapfrog runs exactly
  energy-conserving. `ConductiveMaxwellConstitutivePlan` no longer reports
  `magnetic_closedness_preserving=False`.
- Cartesian PSATD/Galilean spectral PIC field solver
  (`phydrax.solver.maxwell.spectral`): `SpectralMaxwellPlan` on a periodic
  uniform 3-D bridge prepares `PreparedSpectralMaxwell`, an
  `AbstractPreparedPICFieldSolver` for `ElectromagneticPICPlan` with the exact
  φ-function propagator (`variant` standard/galilean/averaged-galilean,
  `time_dependency` constant-j/linear-j/multi-j, `charge_conservation`
  spectral-correction/vay-deposition/update-with-rho, infinite- or finite-order
  `stencil` on collocated or staggered `grid`, `decomposition` global-fft or
  local-guarded), the two-step split-field `SpectralPMLPlan`,
  `SpectralHuygensBoxPlan` far-field sampling, and the NCI monitor
  `SpectralNCIMonitorPlan` with the Godfrey–Vay–Haber linear reference
  `godfrey_vay_growth_rate`. Incompatible combinations are refused at
  construction. New core protocol `phydrax.solver.PICGalileanGrid`: particle
  positions are grid coordinates, the PIC runtime drifts them by
  `(v − v_grid)Δt` and samples external fields at the lab position. Guide
  `docs/guides_spectral_pic.md`, benchmark `benchmarks/spectral_maxwell.py`.
- Time-dependent averaged free-electron laser
  (`phydrax.applications.accelerator.fel`): `FELTimeDependentPlan` composes an
  `FELPlan` and couples its slices through slippage inside the Strang sequence
  `T S(Δz/2) F(Δz) S(Δz/2) T` with `F = D X`; `FELSlippageRoute`
  `"commensurate"` (exact integer rolls, non-integer step slips refused) or
  `"spectral"` (exact Fourier phase ramp), `FELWindowBoundary` `"periodic"`
  (infinitely long beam; reduces to `FELPlan` for a uniform beam) or `"open"`
  with head padding, exit-energy ledger, and padding evidence. SASE starts from
  identity-addressed Fawley shot noise of `I Δζ/(ec)` electrons per slice;
  `FELPulseSeed` seeds from optics `PulseEnvelopeField`s on exact slots with an
  exact carrier ramp; `FELPrebunching` runs `FELModulator` energy modulation and
  `SymplecticMapPlan` chicanes for HGHG/EEHG; per-slice X2 wakes and X3 space
  charge (`FELSpaceCharge`). `FELTimeDependentResult` reports temporal power,
  pulse energy, per-slice bunching, `FELSpectrum` (spectral energy, spikes,
  coherence time, bandwidth), `FELTimeDependentLedger`, and
  `FELTimeDependentEvidence`; `FELStatus` gains `WINDOW_TRUNCATED`,
  `SEED_UNREPRESENTED`, `SPACE_CHARGE_REFUSED`, and `SLIPPAGE_UNRESOLVED`. Pinned Genesis 1.3 version 4
  oracle `run_genesis4`/`genesis4_input`/`Genesis4GaussianSeed`/`Genesis4Result`.
  `SpaceChargeIGFPlan.evaluate` returns the traceable `SpaceChargeIGFKick`
  arrays (used by `kick`), and uniform splat stencils take their spacing from
  the prepared coordinates so both run under `jit`. Qualification candidate
  `accelerator.fel-time-dependent`, example
  `examples/free_electron_laser_time_dependent.py`, benchmark
  `benchmarks/fel_time_dependent.py`.
- Coherent synchrotron radiation (`phydrax.applications.accelerator`):
  `CSRPlan(model, lattice, scale, grid, ...)` with `CSRModel =
  Literal["1d-steady", "1d-transient-shielded", "3d-steady-igf",
  "3d-retarded-mesh"]` on a planar drift/arc `CSRLattice` with pole faces:
  the Saldin–Derbenev steady wake with a cell-integrated kernel; the exact
  retarded 1-D line-charge model of Mayes and Hoffstaetter with entrance and
  exit transients, previous elements, density history (bunch compression), and
  parallel-plate image series with half-weighted truncation evidence; the
  steady 3-D Green functions (the Cai–Ding 2020 longitudinal potential and
  closed-form exact-Lorentz transverse potentials, which give the residual
  centripetal force `−2qλ/(4πε₀ρ)` of both 3-D routes) integrated over cells
  and convolved on the doubled grid; and a retarded 3-D mesh of the
  smooth deposited density over its history. Retarded times use the native
  bracketed `scalar_root`; the straight-line space-charge field is subtracted
  pairwise. `CSRState` (bounded density ring and per-particle potential memory
  so `−ΔqΦ` telescopes), `CSRWake`, `CSRKickResult`, `CSREvidence`
  (`CSRStatus` refusals and Derbenev/steady-length validity bits),
  `CSRResources`/`CSRResourceEstimate`/`CSRResourceError`, and split-step
  tracking `CSRTrackingPlan`/`track_csr`/`CSRTrackingResult`. Pinned external
  oracles `ocelot_csr_tracking`/`OcelotCSRProvider` and
  `pycsr3d_longitudinal_wake`/`PyCSR3DProvider`. Qualification candidates
  `accelerator.csr-1d` and `accelerator.csr-3d`; example
  `examples/csr_chicane.py`; benchmark `benchmarks/csr_models.py`.
- `FreeSpaceConvolutionPlan("tabulated", grid, kernel_table=...)`
  (`phydrax.operators`): caller-supplied kernels at every on-grid displacement
  for Green functions without reflection symmetry, with trailing component
  axes; existing kernel plan identities are unchanged.
- Radiation reaction (`phydrax.discretization.pic`): `RadiationReactionPlan(model,
  scale, physical_charge, physical_mass, *, tables, maximum_chi, minimum_gamma,
  maximum_relative_step_loss, minimum_scale_separation)` with models
  `landau-lifshitz-reduced`, `landau-lifshitz` (field gradients from the
  order-one gather plus staggered-history time derivatives),
  `quantum-corrected-landau-lifshitz` and `stochastic-fokker-planck` (Niel et al.
  2018, identity-addressed Wiener increments); `RadiationReactionTables` (`g(χ)`,
  `h(χ)` from `phydrax.special` synchrotron kernels), `RadiationReactionResult`
  (radiated energy, χ, critical frequency, scale separation, Fokker–Planck drift
  and diffusion), `RadiationReactionFlag`, and the momentum-stage
  `RadiationReactionProcess` claiming `subgrid-reaction` ownership. PIC core:
  `PICEnergyLedger.radiated`, `PICRejectionReason.RADIATION_OWNERSHIP`,
  `PICProcessRadiation` on `PICProcessLedger.radiation`, `PICProcessContext`
  field gradients/time derivatives and grid cutoff, `AbstractPICProcess.
  requires_field_derivatives` and `validate_run`, and
  `phydrax.solver.PICFieldHistory` in `ElectromagneticPICState.field_history`
  (restart component `field-history`, shifted by moving windows). Detector
  `ChargedPropagationPlan(radiation_reaction=, radiation_key=)` with
  `ChargedPropagationResult.radiated_energy_history` and
  `radiation_flags_history`. Qualification candidate `pic.radiation-reaction`;
  example `examples/pic_radiation_reaction.py`; benchmark
  `benchmarks/pic_radiation_reaction.py`.
- Plasma rays and mode-resolved polarized transfer: `phydrax.optics.geometric`
  `DispersionRayPlan(hamiltonian, step_size, step_count, method=)` integrates
  rays of any `AbstractDispersionHamiltonian` with the implicit-midpoint
  symplectic scheme (midpoint solved by `VectorLocalRootPlan`, Cayley tangent
  map) or, for `AbstractSeparableDispersionHamiltonian`, kick–drift–kick;
  `DispersionRayResult`/`DispersionRayEvidence` report Hamiltonian drift,
  symplectic residual, root convergence, refractive-index extremes and
  per-ray `DispersionRayStatus` (launch, support, resonance, qualifying
  cutoff). `ColdPlasmaProfile` and `ColdPlasmaHamiltonian` trace one cold-plasma
  mode (Stix quartic in Cartesian form, normalized to the group light path,
  mode fixed at launch by the `PlasmaWaveMode` label);
  `ColdPlasmaHamiltonian.sample_path` returns a `ColdPlasmaRayPath` with the
  ray refractive index `n_r²`, mode Stokes vectors and Faraday rotation in a
  parallel-transported basis, mode-coupling parameter and quasi-transverse
  flags. `phydrax.applications.radiation_transport.PlasmaRayTransferPlan`
  transports `S/n_r²` in the weak (independent modes) or strong (coupled
  Stokes) mode-coupling limit (`ModeCouplingLimit`,
  `PlasmaRayTransferResult`, `PlasmaRayTransferStatus`) on the canonical
  `PolarizedRadiativeTransferPlan`, and
  `magnetobremsstrahlung_path_coefficients` composes `MagnetobremsstrahlungPlan`
  per segment into `PlasmaPathCoefficients`. Candidate qualification
  `electromagnetics.plasma-rays`; example `examples/plasma_ray_transfer.py`.
- PIC shapes, filters, binning, and numerical-Cherenkov guard:
  `PICParticleCochainTransferPlan(bridge, *, shape_order=1|2|3)`
  (`PICShapeOrder`) selects spline-Whitney transfer; orders two and three
  deposit degree-`p` B-spline charge, gather edge/face components with degree
  `p − 1` along spanned axes, and `ChargeConservingCurrentPlan` integrates the
  spline-Whitney path integrals exactly (knot-lattice splitting plus
  `⌈3p/2⌉`-point Gauss–Legendre) with continuity to roundoff, reduced through
  `phydrax.sparse` in canonical cell-binned order so the current is invariant
  to particle slot order. Order one is numerically unchanged.
  `TensorBSplineSplatAssignment` accepts per-axis degree tuples.
  `phydrax.solver.PICFilterPlan` (binomial passes plus compensation) acts
  identically on charge, current, and the gather field through the new
  optional `PICTensorLayout` solver capability (cochain and reduced solvers),
  commutes with the divergence on periodic axes, mirrors with component parity
  at nonperiodic walls, and reports `PICFilterContinuityReport` (interior and
  wall-normal commutation defects, Gauss-initialization residual) in
  `ElectromagneticPICPlan.filter_reports`.
  `phydrax.discretization.pic.PICCellBinningPlan`/`PICCellBins` give
  bounded-work cell-sorted binning in canonical `(cell, identity)` order.
  `phydrax.solver.PICCherenkovGuard` (`ElectromagneticPICPlan(...,
  cherenkov_guards=...)`) enforces a B3 `CherenkovRegimePlan`: numerical-only
  emission is refused at construction and steps off the audited step size or
  drift speed are rejected with `PICRejectionReason.NUMERICAL_CHERENKOV`.
- openPMD meshes and PIC output (`phydrax.interchange`): `OpenPMDMeshRecord`
  and `OpenPMDMeshIteration` carry the openPMD 1.1.0 `E`, `B`, `J`, and `rho`
  mesh records in `cartesian` and `thetaMode` (`m=<M>;imag=+`, `2M - 1` mode
  planes) geometry with per-component staggering `position` and record
  `timeOffset`, in the units of a bound `ElectromagneticScaleContract`
  (`unitSI`, `gridUnitSI`, `timeUnitSI`, `unitDimension` checked).
  `read_openpmd_meshes_hdf5(resource, OpenPMDMeshImportPolicy(iteration,
  records=), scale=)` preflights the whole HDF5 tree and decoded-byte budget
  before any payload read, canonicalizes Cartesian axes to `x, y, z` (C or F
  `dataOrder`), and refuses with `OpenPMDMeshError`; `write_openpmd_meshes_hdf5`
  publishes one iteration exclusively as a `fileBased` series member.
  `OpenPMDPICLayout(plan, scale, particle_masses)`,
  `write_openpmd_pic_state`, and `read_openpmd_pic_state` export and rebuild
  `ElectromagneticPICState` for the cochain and reduced field solvers (fields
  with solver staggering, `J` and momenta at `timeOffset -dt/2`, particles in
  identity order with `parentId` lineage, wall charge from `rho`); slot
  bookkeeping, charge-transition history, boundary ledgers, and recorder
  states are declared losses. `OpenPMDPICStreamWriter` is the bounded per-step
  output component: atomic exclusive publication per due iteration, with
  iteration-count and total-byte budgets refused before publishing.
  ADIOS2 BP4 is an optional provider route through a pinned openPMD-api
  `openpmd-pipe` (`OpenPMDADIOS2Provider`, `convert_openpmd_hdf5_to_adios2`,
  `read_openpmd_adios2`, and the writer's `provider=`). Catalog format
  `openpmd-mesh`. The shared openPMD base gains `fileBased` root metadata and
  shared constant-component, identity, and ED-PIC weighting helpers, which
  the particle-track adapter now uses.
- Macroparticle resampling as PIC population processes
  (`phydrax.discretization.pic`): `ParticleMergePlan(method="vranic-momentum-cell")`
  merges packets of one charge state in spherical momentum cells of crowded
  cells (`maximum_per_cell`) into exact pairs conserving charge, mass,
  momentum, energy, and the charge dipole (Vranic et al. 2015);
  `ParticleSplitPlan` splits the heaviest particles of sparse cells
  (`minimum_per_cell`) into `2d` axis-pair children. Both bin through
  `PICCellBinningPlan`, order packets, reductions, and identity assignment by
  cell and global identity (slot-order invariant), give products fresh
  identities with the lowest merged or the split particle as parent, refuse
  events beyond free capacity or allocation width with counted evidence, and
  report `ParticleResamplingEvidence` (conservation defects, momentum and
  spatial second-moment distortion, occupancy before/after,
  `ParticleResamplingStatus`). New optional field-solver capability
  `PICGaussProjection` (`PICGaussProjectionResult`, route
  `"cochain-poisson"`/`"spectral-poisson"`) is implemented by the cochain,
  reduced, and tetrahedral PIC solvers; processes declaring
  `AbstractPICProcess.redistributes_charge` make `ElectromagneticPICPlan`
  Gauss-project the field onto the redeposited charge after the population
  stage, reported as `ElectromagneticPICDiagnostics.gauss_projection` with
  divergence before/after. `PICProcessResult.evidence` carries process
  evidence to `ElectromagneticPICDiagnostics.process_evidence`. Example
  `examples/pic_resampling.py`.
- Time-independent averaged free-electron laser
  (`phydrax.applications.accelerator.fel`): `FELUndulatorLattice` of
  `FELUndulatorSegment`s wrapping X1a `InsertionDeviceField`s (stepwise taper,
  natural and smooth focusing, thin break quadrupoles and phase shifters);
  `FELBeamSlices` and `FELLoading` quiet-start beamlets (`M ≥ 2h_max`) with
  identity-addressed Fawley shot noise; `FELPlan.solve` integrates the KMR
  period-averaged equations with harmonic `[JJ]_h` coupling
  (`undulator_coupling_factors`) by a Strang split of exact betatron motion,
  optics angular-spectrum diffraction (or a one-dimensional field), and an RK4
  source kick; `FELResult` reports power, bunching, harmonic frequencies and
  far field, `FELGainEvidence` (fitted gain length, saturation, Pierce/1-D/Ming
  Xie `fel_scaling_estimate`), an `FELEnergyLedger`, and `FELStatus` evidence.
  X2 longitudinal wakes apply per slice through `FELWakeLoss`. Qualification
  candidate `accelerator.fel-averaged`, guide `guides_free_electron_lasers.md`,
  example `examples/free_electron_laser_averaged.py`, benchmark
  `benchmarks/fel_averaged.py`.
- Identity-addressed PIC particle tracks (`phydrax.discretization.pic`):
  `PICTrackRecorder` implements the PIC recorder protocol for declared
  `(id_hi, id_lo)` identities per species, located among active slots every
  accepted step, so tracks survive slot permutation/migration, reused slots
  never continue a previous occupant, and births/deaths toggle lane activity
  with per-lane first/last step, activation count, parent lineage,
  duplicate-identity and charge/mass-transition evidence and per-sample slot
  and incarnation provenance (`PICTrackRecorderState`). Samples carry the
  time-centered proper velocity `(u^{k−1/2} + u^{k+1/2})/2` at `x^k` (one step
  behind the run); reduced 1-D/2-D runs drift unresolved coordinates with
  `u/γ`. `AbstractPICRecorder` gains `validate_run(species, relativity)`
  (called by `ElectromagneticPICPlan` at construction; the track recorder
  refuses other species plans or pusher relativity) and
  `shift_frame(state, axis, distance)` (called by `PICMovingWindowPlan.shift`;
  the track recorder accumulates the window offset so moving-window tracks
  stay in the lab frame). `PICTrackBuffer` is a fixed-capacity ring with
  `TrackOverflowPolicy` `"refuse"`/`"keep-latest"` and `dropped_samples`;
  `to_charged_trajectory(state, scale)` returns a `ChargedTrajectory` whose
  charge × multiplicity is the macrocharge. An optional
  `PreparedTrajectoryRadiation` streams every sample into the far-field
  accumulator inside the run, with or without stored tracks
  (`finalize_radiation`); the recorder state is a restart component.
- Storage-ring radiation (`phydrax.applications.accelerator`):
  `RingLattice` of `RingElement` drifts, quadrupoles, combined-function
  sector bends, and thin RF cavities repeated over a periodic cell;
  `RingRadiationPlan` solves the periodic first-order optics and integrates
  the radiation integrals `I₁…I₅` (Gauss–Kronrod on exact in-bend optics),
  giving the damping partition, damping times, `U₀`, and the equilibrium
  emittance and energy spread in the bound `ElectromagneticScaleContract`;
  unstable optics, open rings, `J_x ≤ 0`/`J_s ≤ 0`, sub-ultrarelativistic
  references, and quantum parameters above the classical limit raise
  `RingOpticsError`. `SynchrotronPhotonSpectrum` samples the classical
  photon-number spectrum `F(ξ)/ξ` from `phydrax.special.synchrotron_f`.
  `RadiativeRingTrackingPlan`/`track_radiative_ring` track an
  `AcceleratorBunch` with `"none"`, `"classical"` (mean loss), or
  `"stochastic"` (particle-identity-addressed Poisson photon emission)
  radiation, thin RF at the synchronous phase, bounded memory, and
  per-turn moment, emittance, energy-ledger, overflow, and loss evidence.
  Example `examples/storage_ring_radiation.py`; candidate qualification
  profile `accelerator.ring-radiation`.
- One-way plane antennas (`phydrax.solver.maxwell`):
  `SampledPlaneCurrentAntennaPlan` launches sampled tangential rest-frame
  envelopes `Re[A(τ) e^{−iω₀τ}]` from an electric node-plane sheet
  `K = s â × H'` and a magnetic sheet `K_m = −s â × E'` half a cell upstream
  (total-field/scattered-field placement), implementing the prepared Maxwell
  source contract. `beta` moves the sheet along its normal in the vacuum of a
  declared `ElectromagneticScaleContract` as a smoothed moving TFSF boundary:
  quadratic B-spline weights over the three nearest node planes, one exact
  lattice pair per plane driven by the Lorentz-transformed incident fields at
  each event's rest-frame retarded time (`phydrax.boost_event`), plus the
  convective currents `−D ∂Θ/∂t`, `−B ∂Θ/∂t` of the moving boundary (Doppler
  factor `γ(1 + sβ)`; backward leakage second order in the cell size: 7.9e-4 at
  24 and 1.1e-4 at 48 cells per wavelength for `β = 0.2`).
  `SampledPlaneAntennaEvidence` reports aperture support, emitted carrier and
  resolution, Lorentz factor, active window, touched planes, and the magnetic
  sheet's discrete divergence (its declared surface magnetic charge).
  `MaxwellAntennaWorkObserverPlan` streams the work the sheets do on the field.
  Optics adapters `phydrax.optics.wave.pulse_envelope_antenna`,
  `openpmd_laser_envelope_antenna`, and `sample_focused_gaussian_pulse_envelope`
  (paraxial Gaussian sampled upstream of its waist). The prepared source
  contract gains `validate_runtime(prepared)`, called for every source by
  `PreparedCompatibleMaxwell`; antennas refuse media other than the declared
  homogeneous medium and CPML on their support, and Huygens boxes refuse
  antennas reaching their surface. Candidate capability
  `electromagnetics.one-way-antenna`. Example
  `examples/maxwell_antenna_gaussian_beam.py`.
- Maxwell media and discrete-dispersion audit (`phydrax.solver.maxwell`):
  `MaxwellLorentzPoles` (spatial `(poles, entities)` strengths with masked
  energy/dissipation), magnetic Lorentz poles (magnetization ADE) on
  `LorentzDrudeMaxwellConstitutivePlan`,
  `MagnetizedColdPlasmaMaxwellConstitutivePlan` (multi-species
  `J̇ = ε₀ωₚ²E + J×ω_c − νJ` on vertex-collocated currents with exact
  exponential gyration, `MagnetizedColdPlasmaState`),
  `CompatibleMaxwellDispersionAudit` (exact one-step Bloch map of the executed
  update in a `MaxwellMaterialRegion`, `MaxwellDispersionResult`, cyclotron
  resonance shift), and `CherenkovRegimePlan`/`CherenkovRegimeEvidence`
  (physical versus numerical Cherenkov resonance). Candidate capability
  `electromagnetics.discrete-dispersion-audit`.
- Maxwell frequency response (`phydrax.solver.maxwell`): every constitutive law
  implements `frequency_response(ω)` (`AbstractMaxwellFrequencyResponse`,
  `DiagonalMaxwellFrequencyResponse`, `InstantaneousMaxwellFrequencyResponse`,
  `MagnetizedColdPlasmaFrequencyResponse`) and declares `auxiliary_degrees`.
  `FrequencyMaxwellOperator(..., stretching=MaxwellCPMLPlan)` applies CFS
  coordinate stretching on the CPML profile, with
  `power_ledger` → `FrequencyMaxwellPowerLedger` (source, material, and
  absorbed power).
- PIC field-solver protocol (`phydrax.solver`): `AbstractPreparedPICFieldSolver`
  with core `deposit`, `advance`, and `gather(derivative_order)` (order one
  returns exact forward-mode spatial gradients of the solver's interpolant);
  the deposit↔Gauss pairing is verified numerically when a PIC plan is
  prepared (`ElectromagneticPICPlan.pairing_defect`). Optional capability
  protocols `PICSpectralSymbol`, `PICHuygensSampling`, `PICMultiDeposit`,
  `PICWindowShift`, `PICRestartState`; prescribed fields through the core
  `ExternalFieldSource`. Implementations `CochainMaxwellPICFieldSolver`,
  `ReducedMaxwellPICFieldSolver`, `UnstructuredMaxwellPICFieldSolver`.
  `PICPrecisionPolicy`, `AbstractPICFieldFilter`, `PICRestartCheckpoint`
  (per-component restart with owner-identity admission), `PICWindowInjection`.
- PIC species and processes (`phydrax.discretization.pic`): `PICSpeciesPlan`
  (runtime population with persistent identities and charge model),
  `AbstractPICProcess` with `"momentum"`/`"population"` stages and
  `PICProcessLedger`, `AbstractPICRecorder`, `RadiationOwnership`;
  `collisions.CoulombCollisionProcess`, `collisions.BackgroundCollisionProcess`,
  `ionization.FieldIonizationProcess`, `ionization.ImpactIonizationProcess`;
  `PICRejectionReason.PROCESS`. PIC qualification provider
  `phydrax.solver._pic_qualification` (electrostatic, cochain electromagnetic,
  and pusher gates) called by `tools/pic_qualification.py`.
- Magnetobremsstrahlung (`phydrax.electromagnetics`):
  `MagnetobremsstrahlungPlan` gives mode-resolved emission and absorption of a
  gyrotropic population into both `ColdPlasmaDielectric` modes from `n_σ(ω, θ)`
  and the mode polarization, with an exact `"harmonic-sum"` route (harmonic set
  from the resonance ellipse/hyperbola ∩ momentum support, including `s ≤ 0`)
  and an exact-Bessel `"continuous-harmonic"` route; explicit status for
  evanescent modes, the resonance cone, harmonic capacity and anomalous
  Doppler; Stokes emission and a propagation matrix with the cold-plasma
  Faraday terms. Distributions `AbstractGyrotropicDistribution`,
  `ThermalJuttnerDistribution`, `PowerLawDistribution`, `KappaDistribution`,
  `TabulatedGyrotropicDistribution`. `ThermalFreeFreeModel` (Born thermal Gaunt
  factor, `born_thermal_gaunt`) and `GrayMeanOpacities` Planck/Rosseland means;
  `FaradayCoefficients.mode_stokes`. Qualification candidate
  `electromagnetics.magnetobremsstrahlung`; example
  `examples/magnetobremsstrahlung_emission.py`.
- Optical photons from charged steps (`phydrax.optics.transport`):
  `ChargedOpticalSteps` (straight steps, speed linear in path, start times,
  optical medium, deposit, charge number, multiplicity, parent identity,
  completeness) built by `charged_steps_from_transport` from
  `ChargedParticleTransportPlan` step banks or by
  `charged_steps_from_trajectory` from `ChargedTrajectory` lanes.
  `CherenkovEmission` samples the dispersive Frank–Tamm spectrum on declared
  wavelength nodes, the emission speed by exact inverse CDF above threshold,
  the cone `cos θ = 1 / (β n(λ))` at the local speed, and polarization along
  `p × (p × v)`; `ScintillationEmission` applies yield, Birks quenching,
  rise/decay time components, and tabulated emission spectra.
  `OpticalPhotonSourcePlan` and `emit_optical_photons` draw Poisson counts
  keyed by parent identity, step, and ordinal into one fixed-capacity
  `OpticalPhotonEmission` bank (launch with `ExplicitPhotonSource`) with
  consecutive photon identities, parent lineage, per-parent count and energy
  ledgers, and `OpticalEmissionStatus` flags that refuse the whole bank
  atomically. `phydrax.equations.cherenkov_step_spectral_yield` is the single
  dispersive Frank–Tamm owner, exact along linear-speed steps. Candidate
  profile `optics.transport.charged-step-optical-sources`; example
  `examples/cherenkov_water_detector.py`.
- Kinetic plasma dielectric (`phydrax.electromagnetics`):
  `KineticPlasmaDielectric` gives the hot magnetized susceptibility of
  drifting bi-Maxwellian species with `Z = i√π·wofz` on Landau's contour for
  every `Im ω` and the causal `sgn(k∥)` continuation, exact `k∥ = 0`
  (Bernstein) and `k⊥ = 0` limits, first-omitted-harmonic truncation
  evidence, and a weakly relativistic Shkarofsky model (lowest Larmor order,
  `F_q(z, a)` from closed forms in `Z` continued from `Im ω > 0`).
  `KineticDispersionProblem` solves the electromagnetic or electrostatic
  dispersion relation for complex `ω` with `VectorLocalRootPlan` along a
  wavenumber path with secant branch continuation and per-point
  convergence, conditioning and branch-jump status.
  `RelativisticWeakGrowthPlan` integrates the relativistic anti-Hermitian
  susceptibility of an arbitrary `AbstractGyrotropicDistribution` over the
  resonance ellipse for the weak growth of `ColdPlasmaDielectric` modes, with
  `LossConeDistribution`, `RingDistribution` and `HorseshoeDistribution`
  maser drivers. Candidate qualification `electromagnetics.kinetic-dispersion`;
  example `examples/kinetic_plasma_dispersion.py`.
- Accelerator field maps and insertion devices
  (`phydrax.applications.accelerator`): exact-vacuum analytic
  `InsertionDeviceField` (planar/helical undulators and wigglers with matched
  tanh terminations and `resonance(scale, γ)` giving `K`, the on-axis
  fundamental, and the critical frequency), `DipoleBendField` with analytic
  fringes, `TabulatedFieldMap` on the native multilinear gather with reported
  interpolation-error estimates, and `FieldMapBeamline` binding one
  `ElectromagneticScaleContract`. `FieldMapTrackingPlan`/`track_field_map`
  convert a bunch between entrance/exit planes and lab time per
  `AcceleratorConvention`, push with `RelativisticPushPlan`, and return the
  exit bunch, a per-lane lab-time `ChargedTrajectory` for trajectory
  radiation, and field-support/exit-orbit evidence; apertures beyond an
  analytic model's support and lanes leaving field support are refused.
  Core capability `phydrax.discretization.pic.ExternalFieldSource`
  (`external_fields(positions, times) -> ExternalFieldSample`) is implemented by
  every element. Example `examples/undulator_radiation.py`; candidate
  qualification profile `accelerator.insertion-device-radiation`.
- openPMD particle tracks (`phydrax.interchange`):
  `read_openpmd_particle_tracks_hdf5(resource, OpenPMDParticleTrackImportPolicy(
  OpenPMDParticleTrackSelection(species, iterations=, identities=)), scale=)`
  follows one openPMD 1.1.0 species through an increasing iteration range and
  returns a `ChargedTrajectory` (lanes in ascending `uint64` id order, split into
  `(hi, lo)` words; weighting as multiplicity; momentum/mass as proper velocity
  through ED-PIC `macroWeighted`/`weightingPower`), the per-lane rest masses, and
  an `AdapterReport`. `unitSI` and `unitDimension` convert through the bound
  `ElectromagneticScaleContract.unit_si_map()`. Missing or repeated ids,
  nonmonotonic times, per-particle charge/mass/weighting changes, truncated
  records, unit-dimension mismatches, and resource overflow are refused with
  `OpenPMDParticleTrackError` before any payload read where the check is
  structural. `write_openpmd_particle_tracks_hdf5` encodes shared-time,
  fully active lanes in scale units (declared loss for `proper_accelerations`).
  Catalog format `openpmd-particle-tracks`; no openPMD-api dependency.
- Detector tracks as radiation lanes:
  `phydrax.applications.detector.charged_trajectory(plan, tracks, result,
  scale)` returns a `ChargedTrajectory` from one constant-field propagation.
  It prepends the initial sample at time 0 and uses row-major
  `(event_id, track_id)` lane identities. Charges are per-particle charges in
  `scale.charge_unit` with multiplicity one, and proper velocities are
  `u = p c²/E₀`. It refuses a pusher relativity scale whose units or exact speed
  of light differ from `scale`, and refuses non-float64 kinematics.
  `ChargedPropagationResult` gains `active_history`, which records whether each
  step committed a state of an existing track. A constant-field helix radiates
  its cyclotron line at `|q|B/(γm)`.
- Electromagnetic shower matter-process closure: optional governed XCOM
  nuclear/electron-field Bethe--Heitler pair channels now create typed
  electron/positron daughters in fixed-capacity shower stacks;
  charged-particle bremsstrahlung selects the unchanged bounded diagnostic
  route, an independent screened Bethe--Heitler/Tsai spectrum, or
  `SeltzerBergerBremsstrahlungTable`, with optional LPM and Ter–Mikaelian
  suppression. Public reference kernels cover Møller/Bhabha delta-ray
  kinematics, positron annihilation in flight, fluorescence/Auger relaxation,
  and the Longo--Sestili shower profile. `FoilStackTransitionRadiationPlan`
  implements Garibian/Cherry formation-zone interference with absorption;
  `ChargedStepRadiationPlan` consumes M1a step banks for foil transition
  radiation and Frank--Tamm Cherenkov yields with atomic capacity refusal and
  an energy ledger. `tools/import_nist_seltzer_berger.py` imports only
  caller-supplied SHA-256-pinned NIST rows; no NIST data are fabricated or
  bundled. NIST XCOM/ESTAR/Seltzer--Berger data are documented as United
  States public domain under 17 U.S.C. § 105, while Geant4 remains a separately
  pinned provider-only oracle.
- Near-zone Liénard–Wiechert fields (`phydrax.electromagnetics`):
  `LienardWiechertFieldPlan(scale, *, history, exclusion_radius,
  interpolation, resources)` and `PreparedLienardWiechertField.evaluate(
  trajectory, observer_events[O, 4])` return the electric, magnetic,
  velocity-field, and acceleration-field parts of point charges at any
  distance, plus retarded times `[O, P]`. The monotone retarded condition is
  located by a fixed `⌈log₂(T − 1)⌉` index bisection and solved on the Hermite
  segment (`"hermite-cubic"` or `"hermite-quintic"`) by the native TOMS748
  `scalar_root`, differentiated implicitly through `lax.custom_root`; `κ`
  uses the cancellation-free form. `history="refuse"` or
  `"inertial-extrapolation"` decides retarded times before the window; later
  retarded times, nonmonotone or superluminal samples, and superluminal
  interpolants are unsupported (NaN, never zero). `LienardWiechertEvidence`
  reports per-observer and per-pair `LienardWiechertStatus` bits, support,
  resolution, derivative-valid masks, excluded and absent (inactive) charge,
  retardation factor, observer-time root residual, Lorentz-consistency
  mismatch, and the resource estimate; execution runs in bounded
  observer×particle chunks refused above `LienardWiechertResources` limits
  (`LienardWiechertResourceError`). Candidate profile
  `electromagnetics.near-zone-lienard-wiechert`,
  `examples/lienard_wiechert_near_field.py`,
  `benchmarks/lienard_wiechert_fields.py`, and guide/API sections.
- Vacuum trajectory radiation (`phydrax.electromagnetics`):
  `TrajectoryRadiationPlan(scale, observers, angular_frequencies, *, coherence,
  route, emission, form_factor, bunch_sigma, observer_time_window,
  quadrature_order, resources)` computes far-field spectra of sampled
  `ChargedTrajectory` lanes (float64 per-lane times, positions, proper
  velocities and optional accelerations, charges, multiplicities, activity,
  `(hi, lo)` identities; float32 refused) for `RadiationObserverPlan`
  directions with the `exp(-iωt)` convention and `d²W/(dω dΩ) = ε₀c|rẼ|²/π`,
  prefactors from `ElectromagneticScaleContract`, and the stable retardation
  factor `(1/γ² + |n×β|²)/(1 + n·β)`. Routes `"segment-exact"`,
  `"segment-hermite"` (fourth order), and `"node-gridded"` (batched Type-3
  NUFFT with a reported absolute floor); coherence `"coherent"`,
  `"incoherent"`, `"gaussian-form-factor"`, `"tabulated-form-factor"`; results
  carry field spectra, coherency, Stokes (`U = 2Re(R1R2*)`, `V = -2Im(R1R2*)`)
  and `TrajectoryRadiationEvidence` (`TrajectoryRadiationStatus` bits,
  `resolved`, `derivative_valid`, retardation/phase/amplitude/window-edge
  metrics, gridded floor, resource estimate). Execution is bounded by
  `TrajectoryRadiationResources` (`TrajectoryRadiationResourceError`);
  `PreparedTrajectoryRadiation` offers `evaluate`, streaming
  `initialize`/`accumulate`/`finalize` equal to offline evaluation, and the
  coherent observer-time `waveform`. Candidate profile
  `electromagnetics.vacuum-trajectory-radiation`,
  `tools/electromagnetic_radiation_qualification.py`,
  `benchmarks/trajectory_radiation.py`,
  `examples/trajectory_cyclotron_synchrotron_radiation.py`, the
  charged-particle radiation guide, `docs/api/electromagnetics.md` (which now
  also documents `MaxwellFrequencySystem`/`MaxwellFrequencyResult` in place of
  the module dump in the advanced omniphysics page), and the electromagnetic
  radiation source ledger.
- Maxwell Huygens surfaces and far fields (`phydrax.solver.maxwell`): phasors
  follow `exp(-iωt)`. `MaxwellSpectralAcquisition(angular_frequencies, *, sign:
  FourierExponentSign, measure: MaxwellSpectralMeasure, start_time, stop_time)`
  owns every spectral observer's convention; `"time-integral"` applies the
  trapezoid rule over consecutive in-window samples (carrying the previous
  payload and time in `DFTObserverState`) and `"sample-mean"` reproduces the
  former windowed DFT mean. `DFTObserverPlan(probe, acquisition)` replaces
  `DFTObserverPlan(probe, angular_frequencies, *, start_time, stop_time)`;
  callers pass `sign="negative", measure="sample-mean"` for the former values.
  Observers gain `validate_runtime(prepared)`, called by
  `PreparedCompatibleMaxwell`. `MaxwellHuygensBoxPlan(bridge, lower_nodes,
  upper_nodes, acquisition, exterior)` samples a closed `full_3d` box at
  surface-cell centers (tangential `E` from edge circulations, `H` from the four
  straddling faces, both through `SparseLinearMap` gathers) and refuses CPML
  overlap, sources on the surface, dynamic PIC currents, and any constitutive
  law other than the declared lossless `HomogeneousMaxwellExterior`.
  `MaxwellHuygensSurfacePlan(hodge, faces, acquisition, exterior)` does the same
  on closed oriented interior tetrahedral face sets with Whitney
  reconstruction; `MaxwellHuygensSampler` is the capability protocol.
  `MaxwellFarFieldPlan(directions, reference_axis, exterior).evaluate(phasors)`
  returns `MaxwellFarFieldResult` with `field_spectrum[F, D, 2]` on
  `(θ̂, φ̂)`, coherency, Stokes (`U = 2 Re(F_θ F_φ*)`, `V = −2 Im(F_θ F_φ*)`),
  and one-sided `spectral_energy = εc|F|²/π`; `spectral_poynting_energy`
  integrates `(1/π) Re ∫ (Ẽ × H̃*)·n̂ dS`. Signs are fixed by a Hertzian-dipole
  test (pattern, absolute energy, grid convergence, nested boxes, Poynting
  balance). `MaxwellNearToFarPlan` is removed. `TetrahedralMaxwellHodge` now
  records `vertices`, `tetrahedra`, `permittivity`, and `inverse_permeability`,
  and its Whitney assembly uses the correct barycentric gradients `J⁻¹`
  (previously `J⁻ᵀ`, which mis-weighted the edge and face mass matrices on
  cells with non-symmetric Jacobians); unstructured Maxwell results change on
  such meshes. Candidate capability `electromagnetics.maxwell-far-field`.
  Example `examples/maxwell_dipole_far_field.py`, benchmark
  `benchmarks/maxwell_far_field.py`.
- `phydrax.electromagnetics.ColdPlasmaDielectric(scale, *, densities,
  charge_numbers, mass_ratios, magnetic_field, collision_frequencies,
  continuation_steps, polarization_tolerance)`: cold magnetized multi-species
  plasma bound to an `ElectromagneticScaleContract`. `stix_parameters` and
  `dielectric_tensor` return Stix `S`, `D`, `P`, `R`, `L` (complex with
  collisions, `Im n² > 0` absorbing); `refractive_indices(ω, θ)` returns both
  roots of `A n⁴ − B n² + C = 0` in a `ColdPlasmaWaveResult` with branch
  identity (`PlasmaWaveMode` `RIGHT`/`LEFT` by continuation from `θ = 0`,
  `ORDINARY`/`EXTRAORDINARY` from `θ = π/2`, certified by continuing the
  pole-free discriminant root with reported separation and per-step turn),
  unit polarization vectors, `K = iE_x/E_y`, longitudinal components, QL/QT
  discriminant terms, and `ColdPlasmaWaveStatus` bits for evanescent,
  resonant, degenerate, ambiguous, and polarization-undefined roots.
  `characteristic_frequencies` returns every zero of `R`, `L`, `P`, `S` with
  residuals, `resonance_cone` returns `tan²θ_res = −P/S`, and
  `faraday_coefficients` returns Faraday rotation and conversion per unit
  length (`FaradayCoefficients`) from the two mode indices and polarizations.
  Angular frequencies and angles must be float64. See
  `docs/guides_plasma_waves_and_emission.md` and
  `examples/cold_plasma_waves.py`.
- Optical photon Monte Carlo: `phydrax.optics.transport` is one spectral,
  polarized owner. `OpticalMonteCarloPlan(surfaces, medium, *, relativity,
  maximum_interactions, variance_reduction, surface_model, detector_response,
  branch_capacity, photon_batch_size, ...)`, `prepare_optical_monte_carlo`, and
  `simulate_optical_photons(prepared, source, key)` transport
  `OpticalPhotonState` packets carrying position, direction, a transverse axis
  with a unit complex Jones vector on `(e1, direction × e1)`, vacuum
  wavelength, time (`n · distance / c` from the bound
  `RelativityScaleContract`), weight, and a persistent `(id_hi, id_lo)`
  identity allocated consecutively by `launch_optical_photons(...,
  first_identity=...)` with the particle-population reserved-identity rule.
  Scattering and interfaces rotate the Jones vector into the scattering-plane
  and `(s, p)` frames; `maximum_polarization_defect` reports unit-norm and
  transversality. Variance reduction is the explicit
  `OpticalVarianceReduction(interface_branching=Literal["stochastic",
  "expected-split"], roulette_threshold, roulette_survival_probability)`.
  Media, surfaces, detectors, and sources are the `OpticalMedium`,
  `OpticalSurfaceModel`, `OpticalDetectorResponse`, and `OpticalPhotonSource`
  protocols; the tissue configuration implements them with
  `TissueOpticalMedium` (`mu_a`, `mu_s`, Henyey–Greenstein `g`, real `n`),
  `ScalarFresnelSurfaceModel`, `UnitDetectorResponse`, and
  `ExplicitPhotonSource`. Guide `docs/guides_optical_photon_transport.md`,
  example `examples/optical_tissue_reflectance.py`; tests reproduce the MCML
  van de Hulst slab and Giovanelli semi-infinite references.
  `TissueTransportPlan`, `TissueTransportCoefficients`,
  `PreparedTissueTransport`, `TissueTransportResult`, `TissueTransportStatus`,
  `TissueTransportTallies`, `prepare_tissue_transport`, and
  `simulate_tissue_transport` are removed with `optics/transport/_tissue.py`.
  Tissue outputs are re-baselined: random draws are now `derive_key(root,
  SampleAddress(purpose), interaction, id_hi, id_lo, branch)` on the photon
  identity instead of `fold_in` chains on caller `photon_ids`, so per-photon
  histories and finite-sample tallies for a given seed differ from earlier
  releases while their expectations are unchanged; keys must be typed JAX
  keys.
- Spectral optical media (`phydrax.optics.transport`):
  `SpectralOpticalMedium(wavelengths, refractive_indices, absorption_lengths,
  *, rayleigh, henyey_greenstein, mie, wavelength_shifter)` tabulates media on
  one vacuum-wavelength grid with native piecewise-linear interpolation of
  inverse lengths and refuses wavelengths outside the grid.
  `RayleighScattering` samples the polarized dipole law (Cardano polar
  inverse, exact Stokes-conditioned azimuth, Jones `(S1 J_s, S2 J_p)`);
  `HenyeyGreensteinScattering` is the scalar specialization shared with
  `TissueOpticalMedium`; `MieParticles` prepares Lorenz–Mie series per node
  (Wiscombe `N_stop`, Lentz-started downward `D_n(m x)`, Riccati–Bessel
  recurrences, complex `m = n + iκ`) with exact inverse-CDF angular tables,
  runtime `S1`/`S2` at the sampled angle, and `MieTableEvidence` (truncation
  share, Lentz lengths, table normalization/asymmetry residuals, refusal above
  `table_tolerance`); `WavelengthShifter` re-emits isotropically and
  unpolarized with a quantum yield, an exact piecewise-linear emission
  spectrum, and an exponential or delta delay. `lorenz_mie(size_parameter,
  relative_index, cosines)` returns `LorenzMieResult` efficiencies,
  amplitudes, coefficients, and evidence; tests reproduce the Wiscombe MIEV0
  cases. The `OpticalMedium` protocol gains `spectral_support`, and
  `OpticalScatteringSample` gains post-event `wavelengths` and `delays`;
  unsupported lanes stop with `OpticalTransportStatus.SPECTRAL_SUPPORT_EXCEEDED`
  and their weight reported as truncated. Example
  `examples/optical_spectral_media.py`.
- Optical surfaces and photodetection (`phydrax.optics.transport`):
  `UnifiedSurfaceModel(finishes, *, sigma_alpha, specular_spike,
  specular_lobe, backscatter, reflectivity, facet_attempts)` with the
  `OpticalSurfaceFinish = Literal["polished", "ground",
  "polished-front-painted", "ground-front-painted"]` selector is the polarized
  Fresnel boundary (complex `(s, p)` amplitudes of the refractive-interface
  owner applied to the Jones vector, so total-internal-reflection phase and
  ellipticity are carried) and the Geant4 UNIFIED rough surface (Gaussian
  micro-facet tilt, spike/lobe/backscatter/Lambertian branches, front paints,
  surface absorption). `OpticalPhotodetector`, `PhotodetectionPlan`, and
  `detect_optical_arrivals` apply QE(λ), collection efficiency, transit time
  and Gaussian transit-time spread, a gamma single-photoelectron charge, and
  Poisson dark counts to the transport's new `OpticalDetectorArrivals` and
  return the detector `SensitiveHitBank` consumed by `digitize_sensitive_hits`
  (`PhotodetectionResult`, per-event `PhotodetectionStatus`). Protocol
  cutover: `OpticalSurfaceModel.interact(hit, keys)` receives identity-keyed
  per-lane keys and gains `validate_surfaces(surfaces)` (called by
  `OpticalMonteCarloPlan`); `OpticalSurfaceHit` gains `frame_axes` and
  `surface_ids`; `OpticalSurfaceInteraction` gains `reflected_axes` and
  `transmitted_axes`, and `reflectance + transmittance < 1` is now absorbed
  at the surface instead of forced into transmission. `OpticalMonteCarloPlan`
  gains `detector_arrival_capacity`, `OpticalTransportResult` gains
  `detector_arrivals`, and `OpticalTransportStatus` gains
  `DETECTOR_ARRIVAL_CAPACITY_EXHAUSTED`. Tests cover Maxwell boundary
  conditions, the Born–Wolf TIR phase, reciprocity, facet and branch
  statistics, binomial/Poisson/TTS/SPE moments, a light guide, and a
  Lambertian cavity. Example `examples/optical_light_guide_photodetection.py`.
- `phydrax.operators.FreeSpaceConvolutionPlan(kernel, grid, *, softening,
  gradient)`: one Hockney doubled-grid FFT substrate for open-boundary
  convolutions on bounded uniform cell grids with `FreeSpaceKernel =
  Literal["coulomb-igf", "newton-igf", "newton-softened", "biot-savart"]`. The
  `*-igf` kernels are the cell-integrated Green functions of Qiang, Lidia,
  Ryne, and Limborg-Deprey (2006) in closed form; `gradient=True` prepares
  derivative kernels under the same sampling rule so fields are convolved, not
  finite-differenced. `FreeSpaceVortexFFTPlan` and
  `IsolatedCartesianGravityPlan` now own only their scientific semantics over
  this substrate; their duplicated Hockney code is deleted.
  `FreeSpaceVortexFFTPlan` now applies the cell measure to
  `vorticity_density` (its velocity was previously the convolution of the raw
  cell values), takes `velocity_gradient` at construction instead of in
  `evaluate`, and convolves the analytic Biot–Savart kernel derivatives for
  `velocity_gradient` instead of differentiating the padded velocity
  spectrally. Isolated gravity potentials are unchanged.
- `phydrax.applications.accelerator.SpaceChargeIGFPlan(scale, grid, *,
  capacity, maximum_rest_frame_speed)`: native rest-frame space charge.
  `kick(bunch, step_length)` boosts proper velocities with
  `boost_proper_velocity`, deposits macro-charge with multilinear assignment in
  centroid-relative rest-frame coordinates, solves the open-boundary Poisson
  equation with the integrated Coulomb kernel and its integrated gradient,
  back-transforms the field with `boost_fields`, and applies the exact lab
  three-momentum increment `Δ(pc) = q(E + v × B)Δs/β_z`. `SpaceChargeIGFResult`
  reports rest and lab fields, deposited charge, per-particle support, the
  maximum rest-frame speed, and `accepted`. The provider route
  `SpaceChargeKickPlan`/`apply_space_charge_kick` remains distinct.
- Accelerator wakes and impedance replace `LongitudinalWakePlan`,
  `LongitudinalWakeResult`, and `apply_longitudinal_wake` (removed; callers use
  `apply_wake`). `WakeFunctionPlan(kind, zeta_samples, wake_values, *, units,
  causality_convention)` tabulates a causal wake in SI (`WakeKind =
  Literal["longitudinal", "dipolar-x", "dipolar-y", "quadrupolar-x",
  "quadrupolar-y"]`, units `"V/C"`/`"V/C/m"` checked against the kind,
  transverse tables vanishing at zero). `apply_wake(plan, bunch, scale, *,
  bin_count, arrival_time, memory)` binds an `ElectromagneticScaleContract`,
  bins by `zeta` per `AcceleratorConvention`, convolves causally with the
  half-weighted self term, kicks `delta` per physical particle or
  `px/p0`/`py/p0` with source (dipolar) or witness (quadrupolar) offsets, and
  reports bunch energy change, loss factor, causality defect, table truncation,
  and history sufficiency. `WakeMemoryState`/`record_passage` hold a bounded
  ring of passages for multi-bunch/multi-turn wakes inside `lax.scan`.
  `wake_impedance` and `wake_from_impedance` map `W ↔ Z` with the
  `exp(−iωt)` convention through the gridded Type-3 nonuniform Fourier
  transform. `ResonatorWake(R, Q, f_r, *, kind, scale)` gives closed-form
  wakes, impedance, and `loss_factor = ω_r R_s/(2Q)`;
  `ResistiveWallWake(radius, conductivity, length, *, scale)` evaluates Bane–Sands
  short-range wakes with the classical long-range forms beyond `50 s₀`.
- Electromagnetic showers: `phydrax.solver.EMShowerPlan` couples
  `PhotonTransportPlan` and `ChargedParticleTransportPlan` on one voxel
  geometry over bounded generation rounds with fixed-capacity
  `ShowerParticleBatch` launches, identity-ordered compaction of secondaries,
  exact 64-bit child identities `p * stride + e + 1`, atomic capacity/identity
  refusal, and an `EMShowerResult` whose ledger closes
  `primary = deposited + escaped + truncated + stack_remainder` and reports
  photon↔charged transfers, per-generation counts, and every transport result
  and launch batch. `SecondaryStackSpec`/`SecondaryParticleStack` attach
  fixed-capacity per-history secondary stacks: `PhotonTransportPlan(...,
  electron_stack=...)` records photoelectrons (K-shell Sauter angular
  distribution) and Compton recoil electrons (Klein–Nishina kinematics) and
  reports `secondary_electron_energy`; `ChargedParticleTransportPlan(...,
  photon_stack=...)` turns bremsstrahlung tallies into recorded secondary
  photons (leading-order Tsai angle). A full stack reports
  `SECONDARY_CAPACITY_EXHAUSTED` and tallies the refused secondary as
  truncated. `ChargedParticleTransportPlan(..., step_bank_capacity=...)`
  records a `ChargedStepBank` (pre/post position, pre/post `beta`, deposit,
  material, `unrecorded_count`). `PhotonTransportPlan(...,
  compton_kinematics=...)` selects `"free-electron"` or
  `"impulse-approximation"` Doppler broadening from the optional per-material
  `compton_profile_j0` of `RadiationCrossSectionLibrary`. New
  `phydrax.equations` samplers: `sample_sauter_cosine`,
  `compton_electron_cosine`, `sample_compton_profile_momentum`,
  `doppler_scattered_energy`, `sample_bremsstrahlung_photon`. A history keeps
  its first failure status; event/step capacity is reported only for
  otherwise healthy histories. Example `examples/em_shower_slab.py`.
- Photon and charged transport histories are now addressed by persistent
  `(id_hi, id_lo)` identity words through `SampleAddress`/`derive_key`:
  `simulate(..., identities=(hi, lo))` replaces `history_ids=`, results carry
  `id_hi`/`id_lo` instead of `history_ids`, and keys must be typed
  `jax.random.key` values. Random streams, plan IDs (`random_addressing`,
  `step_bank_capacity`, `compton_kinematics`), and consequently sampled
  outcomes change. `PlanarXRayDetectorResult` carries
  `transport_id_hi`/`transport_id_lo`, and sensitive-hit `event_ids` pack both
  words as `int64`.
- `phydrax.special.synchrotron_f(x) = x ∫ₓ^∞ K_{5/3}(t) dt` and
  `phydrax.special.synchrotron_g(x) = x K_{2/3}(x)` for float64 `x >= 0`:
  small-`x` series, twelve log-`x` Chebyshev panels on `[0.25, 64)`, and
  asymptotic series, with analytic all-order custom JVPs from the closed Bessel
  recurrence system. `tools/synchrotron_function_tables.py` regenerates the
  committed table module reproducibly from 30-digit mpmath quadrature and
  records the uniform relative-error bound `3.6e-15`.
- Nonuniform Fourier transforms gain a `"gridded"` Type-1/2 route: the
  exponential-of-semicircle kernel (Barnett, Magland, af Klinteberg 2019) with
  twofold oversampling, width and `beta` chosen from the requested
  `tolerance` split over axes, an FFT, and deconvolution by the kernel Fourier
  transform evaluated with Gauss–Legendre quadrature. The new
  `NonuniformFourierType3Plan` shares the kernel and a gridded Type-2 inner
  transform, and reports a per-target `supported` mask for its declared
  source and target boxes. Gridded preparation reports
  `NonuniformFourierGridEvidence` (requested tolerance, kernel width, fine
  grid, grid bytes, chunk stencil entries), refuses grids above
  `maximum_grid_points` with `NonuniformFourierResourceError`, and refuses a
  tolerance the prepared dtype cannot resolve. `PreparedNonuniformFourier` now
  requires an explicit real `dtype`.
- `phydrax.ElectromagneticScaleContract` composes `RelativityScaleContract` with exact
  `elementary_charge`, `electron_mass`, `vacuum_permittivity`, a charge unit, and
  `constant_set_id`. `si()` gives CODATA 2022 values (`"codata-2022"`). It also
  provides exact `vacuum_permeability`, `vacuum_impedance`, `fine_structure`,
  `classical_electron_radius`, and `schwinger_field`, plus `code_units(...)`, which
  requires every constant explicitly. `unit_si_map()` returns openPMD `unitSI` and
  `unitDimension` pairs. The contract has a canonical `scale_id` fingerprint and
  round-trips through `to_dict`/`from_dict`.
- Core special relativity in `phydrax`: `FourMomentum`, `minkowski_dot`,
  `lower_four_vector`, `MINKOWSKI_METRIC`, and `LorentzFrame` move from
  `phydrax.applications.relativistic_scattering` (re-exports removed; import them
  from `phydrax`). New traced `boost_matrix`, `boost_event`,
  `boost_proper_velocity`, `boost_fields` (`F' = Λ F Λᵀ`), `boost_wavevector`
  (Doppler and aberration), and `transform_spectral_energy` with
  `LorentzSpectralTransform` evidence, which refuses media and truncated
  emission (`SpectralEmissionCompleteness`). `LorentzFrame.boost`, hadronization
  two-body decays, and `BMSFrameTransformation` now use `boost_matrix` /
  `boost_wavevector`; `BMSFrameTransformation.lorentz_matrix` is removed.
- Persistent particle identity: `ParticlePopulationState` adds `id_hi`/`id_lo`
  (`uint32` words of a 64-bit global ID), `parent_hi`/`parent_lo` lineage with a
  `has_parent` mask, and the `next_id_hi`/`next_id_lo` population counter.
  Identity survives deactivation, slot reuse, and slot permutation. Each creating
  event numbers its particles consecutively in request order (slot order for
  `initialize` and `update_particle_population`, ascending event ID for
  `allocate`). New `ParticleAllocationRequest(..., parents=(hi, lo))`,
  `assign_particle_identities`, and `ParticlePopulationStatus.IDENTITY_EXHAUSTED`.
  Field- and impact-ionization electrons record the ionized ion as parent. FLIP
  split children record their cell's receiver as parent. `derive_key` now folds each
  index as one exact `uint32` word. It refuses host integers outside `[0, 2**32)`,
  booleans, non-integer dtypes, and non-scalar indices, so particle keys
  `derive_key(root, address, step, id_hi, id_lo, event)` follow the particle rather
  than its slot.
- Bounded on-disk provider artifacts: `run_pinned_command(..., artifacts=
  PinnedFileOutputs(destination, requests, maximum_total_bytes))` declares file
  outputs as `PinnedFileRequest(path, maximum_bytes)`. After a successful command,
  every declared file is checked against its own cap and the shared total cap. The
  whole set is then exclusively published under the caller-owned destination and
  returned as `PinnedFileArtifact` (location, size, SHA-256, envelope) through
  `PinnedRunResult.file_artifacts` / `file_artifact(path)`; the run evidence records
  each digest. A missing or oversize file, or a destination collision, refuses the
  whole set, publishes nothing, and raises `ExternalRuntimeError`. A failed command
  publishes nothing. File artifacts do not pass through memory, so provider output
  can exceed `max_output_bytes`.
- `phydrax.interchange._openpmd_base` is the shared openPMD HDF5 owner.
  Per-profile standard revisions pin openPMD 1.1.0 for meshes and particles and the
  2.0.0 LaserEnvelope draft at commit `0957997`. Root validation reads 1.x
  extension bitmasks (ED-PIC) and 2.0 extension names, and refuses other standard
  versions as unsupported. It also provides iteration time and `timeUnitSI`, record
  units (`unitSI`/`unitDimension` via `OpenPMDUnit.to_si`/`from_si`),
  revision-dependent grid units (1.1.0 scalar `gridUnitSI`, 2.0 per-axis
  `gridUnitSI` + `gridUnitDimension`), and a bounded structural preflight: depth,
  node, attribute, and decoded-byte budgets, no link aliases, soft/external links,
  or unadmitted filters, and no payload read. The LaserEnvelope adapter now uses
  it: stored code units are converted through `unitSI` and `gridUnitSI`, and
  unsupported-semantic classification uses `OpenPMDUnsupportedError` instead of
  message matching.
- Credible structured two-phase flow: `MACBalancedCapillaryOperator`,
  `StructuredPLICPlan`, `HeightFunctionCurvaturePlan`, reduced-gravity and
  variable-viscosity support, absolute-pressure/interface-geometry accessors,
  the public `alpha_bound_tolerance` admission rule, conservation/work
  evidence, the `advanced_rising_bubble.py` workflow, and Hysing qualification
  tooling.
- `phydrax.bubble_dynamics` with clean, coated, thermal, compressible,
  viscoelastic, dissolving, pinned-surface, cloud, retarded-coupling,
  Bjerknes-force, and far-field-emission routes; public examples cover acoustic
  bubbles, contrast-agent inversion, surface nanobubbles, and bubble clouds.
  `phydrax.equations.NobleAbelStiffenedGasMaterial` supplies the complete
  thermodynamic-material contract used by bubble gas laws.
- Bubble population and acoustic media support:
  `PrinceBlanchCoalescenceKernel`, `LehrCoalescenceKernel`,
  `LuoSvendsenBreakageKernel`, `BubblyMediumDispersionPlan`,
  `solve_bubbly_medium_dispersion`, and `wood_sound_speed`.
- Manifold thin-film transport in `phydrax.interfacial_transport`: prepared
  DDG film surfaces; DLVO disjoining-pressure and black-film equilibria;
  conservative lubrication; symmetric surfactant, Marangoni, and plug-flow
  dynamics; open/wall boundaries; and fixed-topology moving-surface and
  multiregion sheet-slot transfer with explicit status and ledger evidence.
- Single-film optical appearance through
  `phydrax.optics.wave.ThinFilmInterferencePlan`,
  `phydrax.rendering.SpectralColorimetryPlan`,
  `ThinFilmAppearancePlan`, and support-aware `thin_film_surface_colors`.
  `phydrax.artifacts` exposes the existing fail-closed
  `ExternalArtifactPolicy`/`admit_external_artifact` boundary required for
  hash-verified external illuminants. Runnable thin-film and foam-iridescence
  examples retain rejected samples as NaN with owner status instead of
  fabricating colors.
- `phydrax.threshold_dynamics` with periodic Fourier, sparse-label, and mesh
  heat actions; exact/bounded label volumes; gas-diffusive coarsening; and
  explicit energy, extinction, capacity, and rollback evidence. Label states
  bind ordered material, route, preparation, and site identities; uniform
  pair coefficients remain structured without quadratic materialization;
  nonuniform coefficient storage is resource-admitted; sparse exactness uses
  distinct periodic residues and rollback evidence retains current/candidate
  overflow; mesh heat evidence preserves native provenance, convergence, work,
  and storage diagnostics. The shared `InterfaceTensionMatrix`/
  `InterfaceMobilityMatrix` contracts and native `CapacitatedAuctionPlan` use
  stable label identity and certified assignment gaps.
- Resolved bubbly-flow identity, gas-compartment thermodynamics, mixed MAC
  projection, multi-marker transport, drainage-gated coalescence, variable
  surface tension/Marangoni forcing, and evidence-only reduced/resolved
  exchange records in `phydrax.applications.two_phase_flow`.
- N-color color-gradient lattice Boltzmann flow with ordered component IDs,
  pairwise tension/recoloring, `ColorGradientInterfacialFields`, and
  antisymmetric `NearContactRepulsionPlan` work and momentum evidence.
- `phydrax.geometry.multiregion_surface` and
  `phydrax.applications.foams`: labeled non-manifold surfaces, exact validation,
  conservative sparse transfers, volume-constrained equilibrium, transactional
  remeshing/T1/pinch/merge/split events, overdamped and SHAKE/RATTLE dynamics,
  deterministic rupture, circulation-slot vortex air, and Plateau-border
  drainage.
- `phydrax.applications.soap_film_tunnel` for gravity-driven planar film flow
  with meshcore-constrained geometry, open/wire boundaries, film-Mach and
  conservation evidence, cylinder-wake references, and Strouhal uncertainty.
- Cross-track transactions: hard-label-to-multiregion surface extraction,
  biomembrane remeshing through the shared surface-event owner, conservative
  rupture-to-border transfer, support-aware foam rendering, and explicit
  reduced/resolved bubble request and failure records.
- Meshing workflow examples `adaptive_bisection_heat.py`,
  `anisotropic_metric_adaptation.py`, `ale_conservative_remesh.py`,
  `cad_high_order_curving.py`, `boundary_layer_core_mesh.py`, and
  `delaunay_voronoi.py`, linked from the new Workflows section of the meshing
  guide, whose code blocks now run as one sequence. `tools/meshing_qualification.py`
  adds the `bisection`, `device-adaptation`, `metric-adaptation`, `gmsh-metric`,
  `supermesh`, `remap-p1`, `cad-curving`, `boundary-layer`,
  `gmsh-boundary-layer`, `predicates`, `triangulation`, `distribution`,
  `forest-amr`, `omega-h`, and `omega-h-distributed` scenarios (source/target
  identities, `QualificationRuntimeIdentity` with the meshcore build, quality
  before/after, conservation and transfer properties, resources, and explicit
  missing-dependency records); `tools/meshing_benchmarks.py` adds the
  `host-bisection`, `local-metric`, `optimization`, `predicates`,
  `boundary-layer`, `high-order-certification`, `provider-worker`, and
  `repartition` cases.
- Boundary layers: `phydrax.meshing.BoundaryLayerControl` binds a `LayerSchedule`
  to a wall scope with an explicit `BoundaryLayerRoute` (`EXACT_SWEEP`,
  `CAD_EXTRUSION`, `ADVANCING`, `PROVIDER`), `BoundaryLayerCollisionPolicy`
  (`FAIL`, `TERMINATE_LOCALLY`, `REDUCE_THICKNESS`, `MERGE`), and
  `BoundaryLayerCornerPolicy` (`FAN`, `SMOOTH`, `REJECT`).
  `prepare_boundary_layers` grows native advancing layers from triangle or
  quadrilateral walls: dihedral ridge/corner classification, visibility-optimal
  column directions (simplex QP through `phydrax.optim`), fan columns and corner
  patches at convex features, stretch-bounded concave corners, curvature and
  BVH-nearest medial proximity limits, weighted Laplacian smoothing, pyramid and
  tetrahedron termination templates, transition pyramids on quadrilateral caps,
  and per-layer Bernstein validity plus exact-predicate intersection
  certification; failures carry the offending wall vertices and locations.
  `BoundaryLayerMesh` exposes the certified cells, the exact cap, and
  `BoundaryLayerEvidence` (thickness measured as exact wall distance; the
  `layer_active` mask marks requested layers carried by at least one surviving
  column, and layers carried by none report NaN thicknesses and growth rates).
  `GmshProvider.fill_boundary_layer_core` fills the core with the cap and outer
  boundary fixed (bitwise node and exact face-conformity checks) and returns
  `boundary-layer`/`core` zones with `wall`, `layer-core-interface`, and `outer`
  patches. `GmshProvider` realizes ADVANCING and PROVIDER (boundary-layer
  extrusion) controls around closed BRep walls, lowers planar PROVIDER controls to
  a Gmsh `BoundaryLayer` field with measured first-layer thickness, growth, and
  layer-count compliance (`SurfaceMeshingSpec.layer_controls`), and
  `prepare_boundary_layer_extrusion` partitions exact CAD extrusions of planar
  walls into an `EXACT_SWEEP` slab (`BoundaryLayerExtrusion`).
- Device adaptive simplex epochs: `phydrax.discretization.MaskedSimplexMesh`
  (capacity-bucketed simplex layout with activity masks and packed sibling
  half-facets) and `AdaptiveSimplexState` with module-level compiled
  `refine_adaptive_simplex` / `coarsen_adaptive_simplex` (conformity closure in
  one `while_loop`, prefix-sum ID issue, FILTERED_DEVICE child validity) keyed
  by the static `AdaptiveSimplexLayout` of an `AdaptiveSimplexPolicy` bucket.
  Status flags accumulate in `AdaptiveSimplexState.status_flags` /
  `DeviceMetricState.status_flags`: a failed call (capacity, closure bound,
  protected conflict, invalid geometry) rolls every array back but records its
  terminal flags, every later call on that state is refused on device
  (collectively on parts), and the commits raise the mapped `MeshingFailure`
  (RESOURCE_EXHAUSTED, INVALID_SPECIFICATION, QUALITY_REJECTED) instead of
  committing a failed epoch as unchanged. `NEEDS_HOST_RESOLUTION` is resolved at
  commit by exact host orientation of every committed cell, which rejects the
  epoch unless all cells are certified positive; pass-limited coarsening commits
  as PASS_LIMIT. A source-preserving request whose operations were all rejected,
  stalled, or pass-limited keeps that status rather than becoming the converged
  UNCHANGED status.
  `phydrax.meshing.prepare_adaptive_simplex` / `commit_adaptive_simplex` bind a
  certified source to the device and commit one `MeshAdaptationResult` with one
  transfer; the new `MeshAdaptationRoute.DEVICE_BISECTION` commits the meshes,
  lineage, and hierarchy of `NATIVE_BISECTION` byte for byte, and
  `DEVICE_METRIC_2D` runs the planar metric passes on device.
  `partition_adaptive_simplex`, `refine_adaptive_simplex_parts`, and
  `commit_partitioned_adaptive_simplex` refine owned cells per device inside one
  `shard_map` with part-independent IDs; a terminal flag on any part rejects the
  whole epoch. `MeshAdaptationPolicy` gains
  `device_policy`, and its `predicate_mode` now defaults to EXACT for host routes
  and FILTERED_DEVICE for device routes. Host and device bisection share one set
  of Maubach templates.
- Masked solver routes on `MaskedSimplexMesh`: `MaskedFiniteElementPlan`,
  `assemble_masked_finite_element` (P1 mass/stiffness with pinned identity rows on
  inactive DOFs), `constrain_masked_dofs`, `evaluate_masked_fv_geometry`,
  `masked_fv_flux_divergence`, and `evaluate_masked_fv_conservation` (capacity-route
  stage ledger); compiled identity is the layout signature, never the active count.
- Galerkin L2 projection of a scalar Lagrange field between non-matching triangle
  or tetrahedron meshes. `phydrax.discretization.prepare_l2_projection_target`
  prepares the `PreparedL2ProjectionTarget` once per target field: the exact target
  mass, its reverse Cuthill-McKee symbolic Cholesky plan and numeric factor (status
  and pivot diagnostics), a condition estimate, and the target DOF measures.
  `prepare_l2_projection_transfer(source, prepared_target, refinement, field_name=...)`
  assembles only the exact mixed mass on the overlap simplices of a
  `PreparedCommonRefinement` and forms the `FiniteElementL2Projection` primal
  (`M_T^{-1} B`, pullback `B^T M_T^{-1}`, payload axes as one multi-right-hand-side
  block); one target artifact serves every source field and refinement.
  `refresh_l2_projection_target` refactors moved target geometry with an unchanged
  DOF structure through the native sparse factorization refresh, reusing the
  symbolic plan and the module-level compiled factorization, condition, and
  target-solve kernels. Prepared target and projection constructors are
  preparation-owned, preventing same-shape masses, factors, or mixed operators
  from being recombined without their scientific binding. Constant/linear
  preservation and conservation are claimed from the certified coverage. The
  `remap` benchmark case warms the original action before its refresh snapshot,
  then reports cold and refreshed target preparation, both first and warmed
  actions, actual refresh compilations, factor and plan bytes, and retained
  mixed-mass bytes.
  `FiniteElementTopologyTransfer.primal` now accepts any unbatched linear operator
  (claims certified through actions, positivity only from sparse coefficients,
  `action_condition` scaling), and `apply`/`pullback` carry payload axes as one
  column block.
- Fixed-topology mesh motion selects an explicit `FiniteElementMeshMotionRoute`
  through `FiniteElementMeshMotionPolicy(route=...)`: `HARMONIC` (graph Laplacian),
  `LINEAR_ELASTICITY` (matrix-free finite-element elasticity with Jacobian stiffening
  `E = (J_max / J)**chi`), `WINSLOW` (inverse harmonic map with an inversion barrier,
  implicit Newton–Krylov), `MMPDE` (Huang–Russell moving-mesh PDE equilibrium for a
  monitor, SER pseudo-transient continuation with relaxation time `tau`), and
  `PRESCRIBED`. `FiniteElementMotionExtension` owns the prepared route and returns
  `FiniteElementMotionExtensionResult` with native solver status; every route is
  differentiable in the boundary displacement and MMPDE monitor parameters.
  `FiniteElementMeshMotionPlan.realize(..., monitor=...)` drives MMPDE and
  `certify(coordinates)` returns the host-epoch Bernstein certificate.
- `MotionValidityPlan`, `MotionValidityPolicy`, `MotionValidityEvidence`, and
  `MotionValidityStatus` are the single traced accept/reject owner of fixed-topology
  motion (signed corner Jacobians of every standard cell kind, finiteness,
  displacement). FE mesh motion, `FixedConnectivityMotionPlan`, and
  `VariablePatchGeometryPlan` consume it; `FiniteElementGeometryEvidence` is removed
  and `FiniteElementMeshMotionPolicy` takes `validity=MotionValidityPolicy(...)`.
- `FixedConnectivityMotionPlan` and `VariablePatchGeometryPlan` take
  `motion_policy`/`monitor`: non-prescribed routes move only the boundary through the
  supplied map and extend it with the shared motion route, keeping mesh velocity an
  exact time JVP so the GCL holds by construction.
- `MeshMotionMonitor`, `MeshMotionMonitorPolicy`, `MeshMotionAssessment`,
  `MeshMotionDecision`, `MeshMotionAdvance`, and `advance_mesh_motion` assess moved
  meshes (certified Jacobian ratio, quality degradation, displacement, boundary
  residual), relocate/untangle through `optimize_cell_mesh`, and escalate to a
  `MetricMeshAdaptation` whose transition feeds the solver transaction.
- Target-matrix mesh optimization supports prism, pyramid, and polygon corner frames
  and polyhedral face-centroid star simplices; metric-alignment targets use each
  corner's ideal-cell frame.
- Riemannian mesh metrics are owned by `phydrax/meshing/_metric.py`
  (`MeshMetricField` moved from sizing). `MeshMetricField` certifies its declared
  size and anisotropy bounds at construction on the `verify_dense_properties`
  spectrum (dtype-epsilon, condition-scaled roundoff only; violations raise) and
  carries no gradation bound: requested gradation belongs to
  `MetricGradationPolicy` or provider options (`MmgOptions.gradation` is Mmg's
  hgrad with or without a metric), and `size_field_metric(field, scope)` takes no
  gradation. `normalize_mesh_metric(metric, *, policy,
  adjacency, coordinates, vertex_volumes)` takes an explicit
  `MetricNormalizationPolicy` (size and anisotropy bounds, target complexity
  `sum_i V_i sqrt(det M_i)` met by the native bracketed root, gradation, and
  opt-in symmetrization/indefinite projection of untrusted `MeshMetricSamples`)
  and returns `MetricNormalizationEvidence` with every clamp and repair count.
  `grade_mesh_metric` enforces edge-length-aware physical or metric-space growth
  by an exact minimum-first relaxation (scalar) or Alauzet grow-and-intersect
  sweeps (anisotropic). `MetricGradationStatus` distinguishes convergence,
  hard-bound conflict, and sweep exhaustion; success additionally requires the
  a-posteriori maximum violation to meet tolerance, and
  `MetricGradationError` withholds an executable field on failure. Added
  `combine_mesh_metrics` (canonical simultaneous-reduction intersection
  returning `MetricCombinationResult`, whose field exists only when
  `MetricCombinationEvidence` reports no hard-bound conflict and whose
  permutation-independent identity uses canonically sorted input IDs),
  `interpolate_mesh_metric`, `metric_edge_lengths`, and `lp_metric_from_hessian`
  (Loseille-Alauzet `L^p` metric with explicit indefinite/zero Hessian
  handling). All spectral and SPD decisions use `phydrax.linalg`; adaptation
  requests, `BackgroundMetricControl`, and the Mmg and Omega_h plans accept only
  a certified `MeshMetricField`.
- `phydrax.discretization.fem` adds superconvergent patch recovery for P1/P2
  Lagrange fields on simplices: `prepare_gradient_recovery` (fixed-capacity vertex
  patches, two-ring enlargement of singular or ill-conditioned fits, one batched
  patch factorization), `recover_gradient`, `recover_hessian` (recovery of the
  recovered gradient, symmetric part with asymmetry evidence),
  `recovery_error_estimate` (Zienkiewicz-Zhu indicators), and
  `dual_weighted_residual_indicators` (enriched adjoint solved through
  `phydrax.linalg.solve_adjoint`), with `FiniteElementRecoveryEvidence`.
- Native meshing core `phydrax-meshcore` (`native/meshcore`, C++20 plain C ABI
  loaded through `phydrax._meshcore`, optional extra `phydrax[meshcore]` or
  `PHYDRAX_MESHCORE_LIBRARY`): Shewchuk adaptive exact predicates with
  index-ordered symbolic perturbation, exactly classified convex polygon,
  tetrahedron, and halfspace clipping moments, and exact Delaunay, regular, and
  constrained Delaunay (Ruppert/Chew) triangulations. `phydrax.geometry` adds
  `orient2d`/`orient3d`/`incircle`/`insphere` with `PredicateMode` (`FILTERED`,
  `FILTERED_DEVICE`, `EXACT`), `DelaunayTriangulation`,
  `ConstrainedDelaunayTriangulation`, `VoronoiDiagram`, `PowerDiagram`,
  `DiagramCells`, and `TriangulationEvidence`; `GeometryPrecisionPolicy` gains
  `predicate_mode`. Convex-polygon and tetrahedron intersections drop their
  extended-precision stages and the polygon `tolerance` argument: orientation
  decisions go through the policy's predicates (exact with meshcore, otherwise
  unresolved signs return `UNCERTAIN_PREDICATE`), `PredicateEvidence` reports
  the predicate mode and counts, and polygon areas are correctly rounded.
  `phydrax-meshcore` is released in lockstep with phydrax (`phydrax[meshcore]`
  pins the identical version); the loader reports a library of another release
  or one missing a bound C ABI symbol as `MeshcoreUnavailableError`. No C++
  exception crosses the C ABI (a refused allocation is `CAPACITY_EXCEEDED`, a
  rejected argument `INVALID_ARGUMENT`, anything else `INTERNAL_ERROR`) and
  counts whose row offsets overflow are `INVALID_ARGUMENT`; CTest covers
  allocator failure and runs under ASan/UBSan with `-DPHX_MC_SANITIZE=ON`.
- `phydrax.geometry.prepare_common_refinement` certifies the common refinement
  (supermesh) of two `CellMesh` instances of any 2D/3D block mix: convex pieces
  or exactly certified vertex cones per cell, float64 BVH broad phase, batched
  meshcore clipping, and CSR rows grouped by target cell with overlap measures,
  first moments, optional second moments and exact simplex partitions, certified
  cell measures, and global IDs (`PreparedCommonRefinement`). Failures are
  statuses, never repairs (`CommonRefinementStatus`: invalid geometry,
  uncertain predicates, native failures, double coverage, coverage gaps against
  `CommonRefinementCoverage`, resource refusal under `CommonRefinementPolicy`
  candidate/accepted-pair/memory limits), with `CommonRefinementEvidence`.
  meshcore adds `polygon_intersection_simplices` and
  `tetrahedron_intersection_simplices` (C entry points
  `phx_mc_*_intersection_simplices`), which return the moment fans as exact
  simplex partitions. `tools/meshing_benchmarks.py --case supermesh` scales the
  build with the cell count.
- Implicit, OpenVDB, Poisson, and Manifold surface providers: `ImplicitMeshingPlan.execute`
  runs in process and publishes `result.geometry.coordinates` as the JAX
  realization, so result observables differentiate with respect to design
  parameters along the fixed route; `discover_implicit_surface` is vectorized
  (batched lattice field, one-program ITP root isolation, 256-configuration
  dual-cell tables, batched QEF small solves, one batched orientation gradient)
  and takes self-intersection candidates from BVH overlap of reachable dual-cell
  boxes instead of all face pairs; implicit fingerprints hash array bytes.
  `ManifoldProvider.execute(operands, operation, *, vertex_properties=...)` performs
  n-ary booleans and publishes transferred vertex properties per face corner
  with barycentric transfer evidence. `OpenVDBProvider` ingests voxel bricks in
  bulk and offers `OpenVDBLevelSetRebuild`. `PoissonReconstructionSpec` adds
  `density_trim_quantile`, `threads`, and `boundary` (`PoissonBoundaryCondition`),
  and results publish per-vertex sample density. The benchmark case
  `implicit-discovery` scales discovery with lattice resolution.
- Added the machine-learning interoperability qualification runner
  `tools/ml_interoperability_qualification.py`. It runs the scenarios of the 24
  gates in `tests/integration/test_ml_interoperability_qualification.py` (all gates
  or a `--gate` subset, optionally in parallel) and binds each gate's outcome into
  one content-addressed `phydrax.qualification` record chain: support tuple,
  zero-unqualified-scenario criterion, campaign start, raw observation, campaign
  observation, and evidence, validated for causality. The report uses a logical
  clock, so an unchanged build, environment, and outcome set reproduce it byte for
  byte.
- Completed the machine-learning interoperability gates. G4 drives continuous and
  fixed-step discrete dynamics from one bound learned vector field and trains a
  `NeuralFeedbackPolicy` through a differentiable rollout whose gradient matches
  finite differences. G5 exposes a neural operator's proposal and the native
  Newton correction of a finite-element problem as field views. G9 feeds one
  learned face closure the finite-element facet traces of a field view and refuses
  a field with a mismatched port. The new G21 shows DEM and reactive CFD-DEM replay
  mismatches invalidating cotangents while preserving the primal and the forward
  replay evidence.
- Added the guide `docs/guides_ml_interoperability.md`: array roles, explicit
  freeze, model state, lanes, intrinsic execution contracts versus bound authority,
  ports, regularity, precision, randomness, objective admission, proposals versus
  authoritative results, external tiers, and the qualification gates.
- Added solver objectives in `phydrax.solver`: `SolverObjective` (implicit
  solution maps), `RolloutObjective` (unrolled rollouts), and
  `AlgorithmicWorkObjective` (fixed-work solver iterations with the
  `algorithmic_work_loss` residual-reduction loss and a precision-aware stopped
  floor), all built on `AbstractSolverObjective`. Each holds the trained component
  as a separate PARAMETER child and binds it into a fixed prepared solve per
  evaluation, with a frozen realization, declared route and objective kind, an
  explicit `AcceptedResultPolicy`, and failure reduction into support and
  rejection counts. `SolverObjectiveEvaluation` reports value, support, primal
  status, derivative evidence, and binding identity.
- Added `train_components(tree, objectives, optimizer=..., steps=..., key=...)`,
  which trains components through solver objectives on the one accepted-update
  training kernel with the `FunctionalSolver.solve` optimizer union and optional
  checkpoint resume; mixed-authority trees need one compatible objective per
  authority group. It returns `ComponentTrainingResult`.
- Added `phydrax.uq.posterior_problem_from_solver_objective`, a fail-closed
  residual-valued posterior over a component's parameters for EKI and
  distribution-evolution consumers; derivative-free and gradient consumers are
  never switched silently.
- Added fixed-trip native Krylov: `DifferentiationPolicy("algorithmic")` on native
  PCG, projected PCG, GMRES, and FGMRES runs the same gated steps in static-length
  scans. Gram-Schmidt and Givens loops run the static restart length with
  exact-zero masks, each FGMRES restart cycle and Arnoldi step is a checkpoint
  unit, and PCG uses square-root block checkpointing, so reverse mode
  differentiates the executed iteration across restart boundaries. Iterates,
  iteration counts, and status match the unchanged early-exit route used by every
  other mode. Benchmark: `benchmarks/linalg_fixed_trip_krylov.py`.
- Added initial-guess providers: the ACCELERATOR slot
  `phydrax.linalg.AbstractInitialGuessProvider` with `HistoryInitialGuess`,
  `LearnedInitialGuess`, and one `InitialGuessDiagnostics` type.
  `solve(..., initial_guess=provider)` compares the proposal's true residual with
  the native zero guess on device, keeps the zero guess unless the proposal is
  finite and strictly better, stops the selected guess, and reports
  `LinearSolveResult.initial_guess`. `phydrax.nonlinear.select_initial_state`
  applies the same rule to nonlinear initial states with domain validity.
- Added `phydrax.nonlinear.implicit_fixed_point_result`: stopped Picard/Anderson
  primal iterations, a custom root on `g(x, theta) - x`, required tangent and
  adjoint policies for `I - dg/dx`, C1 and branch-margin admission of mapping
  components, and no switch to Newton.
- Added `phydrax.continuation.accepted_point_sensitivity`, the implicit derivative
  of one accepted branch point with its continuation coordinate fixed.
- Added differentiable receding-horizon MPC:
  `phydrax.control.prepare_receding_horizon_mpc_sensitivity` runs the audited
  `RecedingHorizonMPC.solve` with cold-started dense windows, prepares one
  `PreparedQPSensitivity` per window, and composes them through the exact affine
  state handoffs into `PreparedMPCSensitivity` with `jvp`/`vjp` over
  `LinearQuadraticControlProblem`-shaped tangents. Only dense compilations with
  zero solver regularization and no warm start are admitted (`DensePrimalDualQP`
  active-set or barrier KKT, or `MPAXraPDHG(unroll=True)` algorithmic), and the
  complete derivative is refused unless every window is valid, OPTIMAL, and
  regular.
- Added `phydrax.control.prepare_control_linearization`, a matrix-free
  `PreparedControlLinearization` whose `[A B]` and `[C D]` Jacobians are
  `JacobianLinearOperator` values, and
  `phydrax.control.linear_quadratic_problem_from_discrete_dynamics`, which
  linearizes a Euclidean discrete transition along an operating trajectory into a
  `LinearQuadraticControlProblem` and refuses failed transitions.
- Added arbitrary-normal finite-volume face closures: the neutral
  `AbstractFaceClosurePlan` slot, `ArbitraryNormalFaceClosurePlan` evaluated with a
  `FaceFluxContext` (unit normal, positive face measure, grid-normal velocity,
  Cartesian axis, geometry and frame identity), and the optional construction-certified
  `SymmetrizedFaceClosure`. Structured, mapped, block-AMR, triangle, unstructured,
  moving, and overset owners apply one closure at every face site; a learned closure
  trains inside the prepared dynamics as the only parameter lane. Euler,
  compressible Navier--Stokes, and homogeneous-mixture gas systems implement the
  explicit `AbstractNormalFrameSystem` capability used by face-normal-frame closures.
- Added `LearnedStepCorrection`, a learned accepted-step transform: a model
  proposes a correction of the native fixed-step candidate, native checks
  (finiteness, support, declared conservation invariants, lower bounds, and a
  stability bound relative to the native increment) admit it as one
  transaction, and a rejected proposal keeps the native candidate with
  `LearnedStepCorrectionReason` bits in the new `transform_admissibility`
  evidence of fixed-step results, rollouts, and solutions. It never retries,
  never uses the coarse residual as an accuracy certificate, and trains
  through checkpointed fixed-step rollouts with frozen admission decisions.
- Added `MODEL`-authority constitutive slots `AbstractConstitutiveModel` and
  `phydrax.equations.fem.AbstractLocalImplicitMaterial`. `ConstitutiveModel` and
  `LocalImplicitMaterial` are their fixed analytic implementations;
  `LearnedConstitutiveModel` and `LearnedLocalImplicitMaterial` hold a learned
  model child bound to the slot with unit-carrying ports, admit only models with
  classical `C^1` regularity, deterministic randomness, and a declared precision
  contract, return per-site admissibility headers, and poison derivatives
  (including the exact-JVP consistent tangent) at invalid sites.
  `MaterialIntegrationPlan` accepts any constitutive slot implementation and is
  neutral, so learned laws train through implicit mechanics.
  `AbstractTransportClosure`, `AbstractMPMConstitutivePlan`, and
  `AbstractImplicitMPMConstitutivePlan` are neutral `MODEL` slots.
- Frozen learned providers gain an explicit trainable counterpart:
  `LearnedClosureBindingPlan.as_trainable_binding()` returns a
  `TrainableLearnedClosureBinding`, and
  `LearnedChemicalTransitionPlan.as_trainable_binding()` returns a
  `TrainableLearnedChemicalTransitionPlan`; both keep the artifact's ABI,
  schema, normalizer, manifests, and identities, and never modify the artifact.
  The artifacts are now `ExplicitFreeze` holders. `PreparedSpectralDriftHook`
  holds its binding (constructor takes the binding instead of a predictor and
  `binding_id`), and a conservative-face `ArbitraryNormalFaceClosurePlan` holds
  its binding as the correction child, so in either deployment a frozen
  predictor stays fixed and a trainable one trains.
- Solver, state-space, control, and meshing extension points are now owner
  component slots. `AbstractPreconditioner` and `AbstractNonlinearUpdate` are
  neutral `ACCELERATOR` slots whose composites (precision cast, multigrid
  levels, subspace-correction terms, block factorizations) keep learned
  children PARAMETER; `AbstractTransitionKernel` and `AbstractObservationModel`
  are `MODEL` slots; `AbstractControlParameterization` and the new
  `phydrax.meshing.AbstractMeshProposer` are `DECISION` slots.
  `FunctionNonlinearUpdate` is the canonical callable and learned update: models
  in its callable module are bound to the update slot
  (`component_contracts()`), its capabilities derive from their execution and
  derivative contracts, and success still requires a finite proposal that the
  original problem accepts. Added `phydrax.stochastic.ModelObservationLocation`
  (learned observation location with unchanged ensemble-transform numerics),
  `phydrax.control.NeuralFeedbackPolicy` (learned state feedback), and
  `phydrax.meshing.LearnedMeshProposer` (learned marking, size, and metric
  proposals certified only by the native projection).
- Added discrete field views: `PreparedFieldReconstruction` owns exact
  coordinate evaluation, support geometry, value port, regularity, evidenced
  maximum derivative order, trace policy, pointwise query evidence, and the
  exact coefficient transpose; `DiscreteFieldFunctionView` binds coefficients to
  an explicit equivalent `GeometryDomain` and exposes `as_domain_function()`
  (an exact derivative rule that refuses unsupported orders and invalid queries)
  and side-bound `trace(points, side=..., cell_ids=...)`. Finite-element fields
  are evaluated from native tabulation and DOF routes at arbitrary points located
  by `PreparedSimplicialCellLocator` (which now reports every containing cell and
  accepts cell masks) or an explicit `AbstractCellLocator`; `C^0` facet
  gradients require an owner, neighbor, or average trace. Sums with other fields
  require matching value ports (units, frame, axes). FE point interpolation moved
  to its own module and gained arbitrary component shapes and exact physical
  derivatives.
- Added `TensorSpectralDiscretization.evaluate` and `derivative_at` for
  arbitrary-point canonical synthesis (prepared normalization, sign, and mode
  ordering; periodic wrapping; out-of-box queries fail) and
  `prepare_spectral_field_reconstruction` for smooth spectral field views.
- Added `NeuralImplicitRegion`: a certified neural implicit region whose weights
  live in the geometry `DesignState`, with explicit bounds, negative-inside sign
  margins, constructed or declared Lipschitz and evaluation-error bounds,
  discovered topology identity (`ImplicitRegionTopology`), normals only for
  evidenced `C^1` networks with gradient margins, and `recertify` that rejects
  topology changes. Geometry domains remain FIXED.
- Added model execution contracts (`ModelExecutionContract`,
  `ExecutionCapabilities`, `ComponentPrecisionContract`, `RandomnessContract`)
  separate from bound component contracts (`AbstractComponentSlot`,
  `ComponentContract`, `ComponentBinding`). Network, fitted ML, Trefftz,
  layer-potential, and Equinox-wrapped families declare regularity from their
  layers and activations; learned chemistry, learned stress, MHD closures, and
  learned transitions return admissibility headers and derivative contracts.
- Added one canonical derivative vocabulary at the package root: derivative
  surfaces, gradient levels, routes, regularity with polynomial-degree algebra,
  `DerivativeContract` meet/compose/admission, branch-differentiation policies,
  objective kinds, component authorities, capability evidence requirements, and
  construction certificates.
- Added semantic value ports (`ValuePort`, `ModelPorts`, `PortMapping`,
  `resolve_port_mapping`) with derived views for ML schemas, operator fields and
  queries, dynamics state/input layouts, domains, discrete field spaces,
  geophysical fields, and closure schemas.
- Linear solve results and temporal differentiation evidence report canonical
  derivative contracts; input-convex networks emit a construction certificate.
- Added native quantum Hall workflows spanning Haldane, Kane--Mele, and
  Hofstadter lattices; Chern, time-reversal Z2, ribbon, and Bott topology;
  matrix-free projected-sphere pseudopotential spectra and gaps; monopole-sphere
  Landau-level-mixing VMC; conserved-charge cylinder DMRG; periodic-lead
  multi-terminal transport; and finite-width Coulomb form factors.
- Added a native resource-bounded scaled-Taylor exponential action with
  operator-only plan/prepare/refresh reuse, fixed-capacity differentiable
  execution, explicit norm and truncation evidence, and matrix-free augmented
  exponential/phi combinations for affine and semilinear evolution.
- Added a three-dimensional compressible kinetic production surface with
  positive guided D3Q39, entropic D3Q343, filtered thermal D3Q33, full-range
  quasi-equilibrium Prandtl closure, safeguarded entropy roots, explicit KBC
  variants, integral-frame remapping, population-axis partitioning, bounded
  worksets and storage policies, AMR/mapped/moving-geometry contracts,
  species/radiation/ablation coupling, checkpoints, IREE export, VTK output,
  and bounded rendering/video adapters.
- Added `ArtifactBindingIdentity`, the one binding identity of a frozen or
  published model: its `SemanticProvenance`, the `NumericRevision` of its
  dynamic content, and its `ExecutableSignature`, recorded together. Functional
  checkpoints, operator training checkpoints, and native operator artifacts
  record it at publication and recompute it on load, failing closed on any
  mismatch; `ScientificArtifactEnvelope` and lifecycle `ModelManifest` accept
  it whole. `ExecutableSignature(static_callables=...)` and
  `PoolExecutionSignature(static_callables=...)` identify callables compiled into
  an executable through `callable_payload`, so statically held weights are part
  of the executable while dynamic weights change only the numeric revision
  (qualification gate G22).
- Plugin registration is public once at the root: `OperatorArchitectureCodec`,
  `register_operator_architecture_codec`, `operator_architecture_codec`,
  `operator_architecture_codec_for`, `register_artifact_value`,
  `artifact_value`, and `artifact_value_id`. Lookups are exact type or object
  identity with no base-class, name, or entry-point fallback.
- External model tiers: every external invocation is admitted against declared
  `ExecutionCapabilities` before it runs (host-only refuses jit/vmap/grad/jvp/vjp)
  at `ExternalOperatorAdapter` (now requiring capabilities and an
  `ArtifactBindingIdentity`), `OperatorExecutionPlan` (a compiled strategy
  requires jit), `OperatorContextModel`, `IREEExecutable`, and the new
  `phydrax.export.HostInferenceAdapter` and `load_onnx` (`phydrax[onnx-inference]`,
  optional DLPack transport). `phydrax.nn.models.FunctionalJAXAdapter` holds
  PARAMETER and MODEL_STATE lanes with an explicit inference mode; the
  `EquinoxModel` `StateIndex` error names it. Staged external adjoints
  `ExternalPrimalStage`/`ExternalAdjointAction` refuse replay mismatches;
  `DAFoamAdjointAction` adds `DAFoamDesignVariableKind`, canonical ordering,
  shape-preserving totals, and `replay_id`; providers without an adjoint report
  derivative-free alternatives (gates G6, G7, G13).
- Added the `phydrax.graph.facet_adjacency` bridge (`FacetAdjacency`): finite-volume
  owner/neighbor cells and finite-element interior facets become an
  `EdgeRelation` with a shared topology ID and `GraphIR.from_edge_relation`.

### Changed
- Smooth forms carry explicit twist and an always-present coefficient axis.
  Metric Hodge stars no longer require an orientation; converting between twisted
  and untwisted forms requires an explicitly declared orientation.
- Cochain Hodges are dynamic diagonal or sparse metric operators. Relative complexes
  restrict both differentials and pairings before inversion, and numerical refresh
  reuses admitted sparse structure and stable binding identities.
- Graph cochain payloads lower the canonical discretization; spectral and harmonic
  algorithms use native linear algebra. Operator-learning field payloads persist
  canonical form identity and reject obsolete cochain specifications.
- Native Adam defaults use a single explicit-dtype Optax boundary; callers supplying
  their own external optimizer retain their provider contract.
- The integrate conversion's time unit is declared once on
  `CouplingGraph(time_unit=...)` / `PartitionedCouplingDeclaration(time_unit=...)`.
- `AbstractCouplingLaw.prepare` receives `interface_owners`; law runtime
  inputs are declared through `runtime_inputs` and must be parameter-bound.
- `CouplingPort` physical ports use `measurement=CouplingMeasurement`;
  `measure`, `measure_unit` and `temporal_transfer` are removed. Conservative
  transfers are certified by `L_target P = L_source`; ledgers carry one row
  per component (`CouplingState.budget_row_ids`); the cell-average-only
  restriction is gone. Endpoint/waveform conversions must be declared on
  `CouplingExchange(temporal=...)`.
- `MeasurementComparisonResult` reports `noise_model`, `whitening`, a
  factor-backed `whitened_residual`, `quadratic`, `logdet_covariance`,
  `log_likelihood`, `active_value_count` and `active_set_consistent`;
  `standardized_residual` is removed. Data without uncertainty compare as
  unquantified; `MeasurementComparisonPlan(reference_scale=...)` declares
  least-squares weighting; correlated covariances must be declared on the
  active values.
- `DistributedPICFieldSolver(...)` is replaced by
  `distribute_pic_field_solver(...)`, which publishes exactly the protocols
  its configuration supports.
- `SideRepresentation` adds `cell-average` and `face-state`;
  `PreparedIsogeometricDiscretization.prepare_side_trace` acts on the public
  control layout.
- `FactorizedVirtualElementOperator.materialize_buckets` is replaced by
  `local_tensors`.
- `FunctionalSolver.solve(optim=...)` is annotated to accept Phydrax
  least-squares and scalar iterative methods.
- Non-selector aliases introduced in this release use canonical Python 3.12
  `type` statements. Closed `Literal` selectors retain runtime
  `typing.TypeAlias` values so `typing.get_args` remains the canonical source
  of members; the selector and structural-contract audits now recognize both
  syntaxes. CSR and PSATD preparation are split into validation,
  geometry/execution, and kernel phases, and distributed PIC no longer
  constructs a fresh inner `jax.jit` wrapper on each map call.
- `SpectralHuygensBoxPlan` samples each tangential `E` component at its own
  staggered edge points (no interpolation) and interpolates the paired `H`
  across the face with the fourth-order stencil `(−1, 9, 9, −1)/16`: the
  Hertzian-dipole far-field error drops from 4.6 % to 1.5 % at nine cells per
  wavelength. Huygens boxes must stay two cells clear of a PML, and the `J = 0`
  check watches the tangential face edges and the normal edges crossing
  them. `PreparedSpectralHuygensBox` takes a `current_gather`.
- The PSATD PML keeps the Gauss law: the divergence its split-field damping
  creates is booked as `SpectralMaxwellState.absorber_charge`/
  `absorber_magnetic_charge`, supported in
  `PreparedSpectralMaxwell.absorber_support` (the layers dilated by the stencil
  half-width for finite order; for infinite order the global divergence is
  confined to the layers by a static curl-free correction). The spectral
  constraints are full-grid residuals against the Gauss plus declared absorber
  and antenna charges (no interior mask), and `project_gauss` keeps the
  declared charges. Before, the interior Gauss residual grew by spectral
  leakage of the layers' divergence (to `3e-3` of the charge density in 300
  steps) and failed PIC constraint tolerances; it now stays below `1e-13`,
  with the reflection unchanged. Infinite-order PMLs need a layer at least two
  cells thick.
- `SampledPlaneCurrentAntennaPlan` no longer requires a nonperiodic normal
  axis; the compatible (cochain) preparation refuses one.
- `PICFieldAdvance.charge` is the start Gauss charge moved by the step's
  current alone; the PIC particle↔field charge check is incremental, so
  induced wall, conduction, plasma, and CPML charge the field holds is not a
  defect, and the deposit↔Gauss pairing uses the current-driven charge.
- `PICOpenBoundaryPlan.apply` requires `kinetic_energy=(start, end)`; an
  absorbed particle records the energy interpolated to its hit fraction
  (relativistic `(γ − 1)mc²` from the PIC runtime).
- `ownership="diagnostic-only"` is refused for electromagnetic PIC: the
  self-consistent field solver claims the resolved radiation.
- Reduced Maxwell nonperiodic axes use the skew-adjoint zero-pad difference
  pair (lower wall reads zero, forward difference reads zero beyond the last
  cell) in curl, divergence, and charge; the reduced 2-D charge follows the
  Gauss charge of the stepped field. Reduced 2-D PIC with bounded axes pairs
  to roundoff, initializes and projects through the per-axis eigenbasis of the
  reduced Poisson operator, and `ReducedPICTransferPlan` projects 2-D currents
  with that operator (`laplacian_bases`).
- Time-dependent FEL space charge is evaluated from the current particles at
  every step as the kick `K(Δz)` at the center of the Strang step (replacing
  the constant per-slice rate computed once at the entrance).
  `FELSpaceCharge(bunch=None, *, transverse, harmonics=0, radial_extent=0.0,
  radial_cells=100, azimuthal_modes=0, transverse_tolerance=1e-2)` combines
  intra-slice harmonic longitudinal space charge (the Genesis 1.3 version 4
  short-range model in the mean-motion frame `γ_z = γ_r/√(1 + a_w²)`: a radial
  finite-volume solve for the grid model, the uniform-disk reduction
  `1 − 2I₁K₁` for the one-dimensional model) with the X3 bunch-scale field
  (every particle at its slice center, boosted with `γ_z`; open windows,
  capacity slices × particles). The X3 transverse kick (∝ 1/γ_z²) is applied
  or omitted by the `FELTransverseSpaceCharge` selector; its accumulated
  ratio to the entrance rms `u⊥` is reported, and an omitted kick above
  tolerance raises the new `FELStatus.TRANSVERSE_SPACE_CHARGE_OMITTED`.
  `FELTimeDependentResult.space_charge_rate` is replaced by
  `space_charge_gain[z, s]`; `FELTimeDependentEvidence` gains
  `space_charge_cells_per_sigma` and `transverse_space_charge_ratio`.
  `run_genesis4` maps longitudinal space charge onto Genesis `&efield`.
- `SpaceChargeIGFPlan.kick`/`evaluate` take `frame_lorentz_factor` (a
  quasi-static frame other than the bunch reference) and refuse bunches whose
  rest-frame rms size falls below `minimum_cells_per_sigma` (default 3) grid
  cells on any axis; `SpaceChargeIGFResult`/`SpaceChargeIGFKick` report
  `cells_per_sigma` and `resolved`.
- `FrequencyMaxwellOperator` takes `boundaries=` (`MaxwellBoundaryPlan`,
  including `support` masks: perfect-conductor identity rows, zeroed PMC `H`,
  impedance conduction `YE`) and `solve(method=...)` with
  `FrequencyMaxwellSolveMethod` `"krylov"` (GMRES, default) or `"direct"`
  (exact sparse assembly by structural coloring of the curl-curl pattern,
  reusable `sparse_coloring()`, native sparse LU). `FrequencyMaxwellPowerLedger`
  gains `boundary_power`; the Hermitian eigen path refuses perfect-conductor and
  impedance boundaries.
- `ElectromagneticPICState` gains the required trailing `processes` tuple of
  process states (`ElectromagneticPICPlan.initialize_process_states`); the
  moving window shifts it and openPMD import restarts it (declared adapter
  loss). `PICEnergyLedger` gains trailing `created_rest_energy` and
  `field_exchange`, and its `defect` is
  `total + radiated + created_rest_energy − field_exchange − previous_total`.
  `PICProcessStage` gains `"creation"`.
- Compatible Maxwell tracks declared source magnetic charge: source magnetic
  currents (Huygens, mode, antenna sheets) carry physical surface divergence,
  so `MaxwellAuxiliaryState` gains `magnetic_charge` on the magnetic-divergence
  cochain, advanced with the `B` half kicks (`∂q_m/∂t = −d(M)`);
  `PreparedCompatibleMaxwell.pack(..., magnetic_charge=None)` and
  `magnetic_charge_count` expose it. The magnetic Gauss law is `d(B) = q_m`:
  `magnetic_constraint(state)` and the diagnostics report `d(B) − q_m`, and the
  minimum-norm projection (still run for non-preserving materials, boundaries,
  or CPML) targets `d(B) = q_m`. Sources no longer carry
  `magnetic_closedness_preserving` (removed from the prepared source contract,
  `PreparedMaxwellSource`, `MaxwellPairedCurrentSourcePlan`,
  `MaxwellHuygensSourcePlan`, and `PreparedPICMaxwellCurrentSource`), so source
  magnetic currents never trigger the global projection; a fully 3-D antenna beam
  on a 48×48×64 grid runs projection-free with the declared-charge defect at
  roundoff. Runtimes with an elided projection no longer prepare the
  minimum-norm solver: under the automatic policy `pack` checks that the initial
  flux matches its declared charge (a traced error otherwise; use the `"project"`
  mode to project a non-closed initial flux) instead of projecting it.
- `RelativityScaleContract.si()` now declares ħ as exact SI h/(2π), stored as a
  40-significant-digit rational (`1.054571817646156391262428003302280744723e-34`,
  float64-correctly rounded) instead of the truncated `1.054571817e-34`. SI ħ,
  α (now CODATA 2022 α⁻¹ = 137.035999177(21) within 1σ; previously ~4σ off),
  Schwinger field and every ħ-derived quantity shift by ~6.1e-10 relative, and
  the SI relativity/electromagnetic `scale_id` fingerprints (and any artifact
  fingerprint embedding them) change.
- `PICEnergyLedger` gains `radiated` (after `magnetic_field`) and its `defect`
  is `total + radiated − previous_total`; `ElectromagneticPICState` gains the
  required trailing `field_history`; `ChargedPropagationResult` gains
  `radiated_energy_history` and `radiation_flags_history`, and the
  `ChargedPropagationPlan` identity includes its radiation reaction.
- Graded-index rays are the isotropic specialization of dispersion rays:
  `RefractiveIndexHamiltonian(field)` (`H = ½(|p|² − n²)`) and
  `GradedIndexRayPlan` (now a `StrictModule` binding it to kick–drift–kick).
  `GradedIndexRayState`, `GradedIndexRayEvidence`, `GradedIndexRayResult` and
  `PreparedGradedIndexRay` are replaced by `DispersionRayState`,
  `DispersionRayEvidence` (`field_covered` → `medium_covered`, plus root,
  index and status evidence), `DispersionRayResult` and
  `PreparedDispersionRay`; `CurvedSchlierenPlan` takes a
  `PreparedDispersionRay` and checks the Hamiltonian's coordinate frame.
  Geometric and optical lengths integrate `|∂H/∂p|` and `p·∂H/∂p`, which
  differ from the previous `n`-based trapezoid by the Hamiltonian drift;
  graded-index plan identities change. `RayFanPlan` is a `StrictModule`.
- PIC filter protocol: `AbstractPICFieldFilter` operations take the prepared
  solver (`filter_charge(solver, charge)`, `filter_current(solver, current)`,
  `filter_field(solver, field)`), and `validate_solver` is replaced by
  `continuity_report(solver) -> PICFilterContinuityReport`.
  `PICParticleCochainTransferPlan` replaces its `assignment` argument with
  `shape_order`; transfer and current plan identities include the shape order.
- `SymplecticMapPlan` checks symplecticity with the form of its
  `AcceleratorConvention`: `δ` is conjugate to the positive-early longitudinal
  coordinate, so `"positive-late"` maps with dispersion carry the opposite
  longitudinal sign, and conventions without a `"positive-late"`/`"positive-early"`
  sign or the canonical momentum normalization are refused. Field-map
  tracking and wake kicks share the same convention validation.
- `LorentzDrudeMaxwellConstitutivePlan(electric_poles, /, *, magnetic_poles,
  permittivity_infinity, permeability_infinity)` takes `MaxwellLorentzPoles`
  (the positional frequency/damping/strength arrays and the `permeability`
  keyword are removed); `DispersiveMaxwellState` gains `magnetization` and
  `magnetization_velocity`; `drude_maxwell_constitutive` accepts spatial plasma
  frequencies and takes `permeability_infinity`. Plan fingerprints include
  the strengths.
- `FrequencyMaxwellOperator` takes a `StructuredCochainBridge` or cochain,
  uses the law's `frequency_response`, requires `ω > 0`, and no longer takes
  `material_state`; `dissipated_power` is replaced by `power_ledger`.
  Lorentz–Drude, magnetic-loss conductive, and linear-gain laws now support
  the frequency domain; the Hermitian eigen path refuses stretched, lossy, or
  dispersive responses.
- `MaxwellCPMLPlan.prepare(bridge, layout, wave_speed)`: the CPML profile uses
  `σ_max = (m + 1) c ln(1/R) / (2 L)` with the physical layer thickness `L` and
  the runtime's wave-speed bound `c`. The old per-cell `σ_max` ignored grid
  spacing and wave speed, so time-domain CPML results and term fingerprints change.
- `energy_rate` of dispersive constitutive laws is the rate of the complete
  stored energy, so `power_balance_residual` closes with material dissipation.
- `ElectromagneticPICPlan(solver, *, species, processes, boundaries, recorders,
  filters, ownership, precision, ...)` is the single explicit electromagnetic
  PIC runtime over any PIC field solver, with species as
  `ParticlePopulationState` + persistent identities. It replaces the old
  cochain-only constructor, `ReducedElectromagneticPICPlan`/`State`/`Result`
  and `UnstructuredElectromagneticPICPlan`/`State`/`Result`.
  `ElectromagneticPICState` carries `species`, `field`, `boundaries`,
  `wall_charge`, `recorders`; `ElectromagneticPICDiagnostics.particle_maxwell_charge_defect`
  is renamed `particle_field_charge_defect` and gains `process_charge_defect`,
  `processes`, `field_successful`, `process_successful`. The cochain example is
  bitwise unchanged. `PICMovingWindowPlan(pic, axis)` shifts through
  `PICWindowShift` and operates on `ElectromagneticPICState`.
  `ChargeConservingCurrentPlan.deposit` accepts runtime `macrocharge` and
  `active_mask`. `SemiImplicitPICPlan` (ECSIM) remains a separate runtime.
- `tools/pic_qualification.py` report: `boris_speed_defect` is renamed
  `pusher_speed_defect` (worst over every `RelativisticPusher`); adds
  `electromagnetic_continuity_defect`, `particle_field_charge_defect`,
  `gauss_defect`, `deposit_gauss_pairing_defect`.
- Thermal synchrotron moved to `phydrax.electromagnetics`:
  `ThermalSynchrotronModel`, `ThermalSynchrotronCoefficients`,
  `ThermalSynchrotronDomain`, `ThermalSynchrotronEvidence`,
  `ThermalSynchrotronReferenceEvidence` and `ThermalSynchrotronUnitContract` are
  no longer exported by `phydrax.applications.astrophysics`; the model and unit
  contract bind the CODATA 2022 SI `ElectromagneticScaleContract` instead of a
  `RelativityScaleContract` (model and unit fingerprints change; values are
  unchanged). `invariant_synchrotron_coefficients`/`InvariantSynchrotronCoefficients`
  are renamed `invariant_emission_coefficients`/`InvariantEmissionCoefficients`
  and also accept SI `ThermalFreeFreeCoefficients`. Both gain `gray_means`.
- `ThermalBremsstrahlungGrayOpacityPlan` and `ThermalSynchrotronGrayOpacityPlan`
  (`phydrax.applications.compact_objects`) bind an `ElectromagneticScaleContract`
  and compute Planck and Rosseland means of the spectral free–free and MNY96
  coefficients; `emission_prefactor`, `rosseland_ratio`, `radiation_constant`
  and the temperature-window arguments are removed and qualification is the
  spectral support of every mean.
- The PIC pusher is now a relativistic pusher family: `RelativisticBorisPlan` is
  replaced by `RelativisticPushPlan(relativity, method=...)` with
  `RelativisticPusher = Literal["boris", "vay", "higuera-cary"]`, and
  `BorisPushResult` is renamed `RelativisticPushResult` (same fields). The speed
  of light comes from the plan's `RelativityScaleContract`; callers holding an
  `ElectromagneticScaleContract` pass its `relativity`. `"vay"` (Vay 2008) and
  `"higuera-cary"` (Higuera–Cary 2017) keep the E×B drift exact at any γ, and
  Higuera–Cary preserves phase-space volume; `"boris"` is numerically
  unchanged. `phydrax.discretization.pic.PIC_CODE_RELATIVITY` declares the c = 1
  code-unit scale used by electrostatic, electromagnetic, reduced, and
  unstructured PIC plans and by `ChargedPropagationPlan` when no pusher is
  supplied. `ChargedPropagationPlan` takes `pusher=` instead of
  `speed_of_light=`. Pusher and dependent plan fingerprints change.
- Documentation capability boundaries corrected: the particle-in-cell guide now
  states that the full 3-D `ElectromagneticPICPlan` requires every axis periodic
  and therefore cannot carry CPML (`PreparedMaxwellCPML` rejects CPML on periodic
  axes); the advanced particle-grid guide restricts electromagnetic-PIC CPML
  support to reduced 1-D/2-D PIC; the accelerator guide documents
  `LongitudinalWakePlan` as a prescribed, unitless binned-convolution research
  primitive rather than qualified wakefield support.
- Private electromagnetic constants now come from
  `ElectromagneticScaleContract.si()` (CODATA 2022), so several numeric results change
  slightly:
  - Vacuum permittivity changes from 8.8541878128e-12 to 8.8541878188e-12 F/m
    (relative change 6.8e-10). This affects chemistry optical response, SBS, optics
    wave nonlinear/material polarization, geophysical layered EM,
    `semiconductor.VACUUM_PERMITTIVITY_SI` and materials, GR microphysics, and quantum
    Hall Coulomb energies.
  - Vacuum permeability is now 1/(ε₀c²) = 1.25663706127e-6 H/m. It was previously
    4π×10⁻⁷ in `electrical_machines.VACUUM_PERMEABILITY` and
    `geophysics.VACUUM_PERMEABILITY_H_M` (potential fields), and 1.25663706212e-6 in
    layered/magnetotelluric geophysics and London superconductivity.
    `tokamak.DEFAULT_VACUUM_PERMEABILITY_H_M` keeps its value but is now derived.
  - Electron mass in the optics-wave plasma material response changes from the
    CODATA 2018 value 9.1093837015e-31 kg to 9.1093837139e-31 kg. Photon and
    charged-particle transport use an electron rest energy of 510998.95069 eV, derived
    as mₑc²/e, instead of 510998.95 eV.
  - Any fingerprint that records these values changes. This includes geophysical
    layered-earth defaults and the optics-wave plasma response `electron_mass`.
- The energy-specific external runner is now generic: `run_energy_command` →
  `run_pinned_command`, `EnergyRunResult` → `PinnedRunResult` (new
  `file_artifacts` field), `EnergyOutput` → `PinnedOutput`, `EnergyRuntimeError`
  → `ExternalRuntimeError`, and `pin_energy_executable` → `pin_executable`. Every
  caller (backends, interchange adapters, applications, rendering, tools, tests,
  guides) is migrated. Run and output envelopes now use the artifact kinds
  `pinned-command-run`/`pinned-command-output`, so run artifact IDs change.
- Two-phase VOF now advances geometric Weymouth--Yue PLIC transport,
  mass-consistent momentum, implicit variational viscosity, balanced
  capillary/gravity forcing, and variable-density projection in that order.
  Structured PLIC reports exact facet centroids, and height functions use a
  bounded primary-facet quadratic fallback with explicit
  support/rank/condition/resource evidence. Pressure, curvature, Courant,
  capillary-step, viscosity, gravity, surface-energy, and work diagnostics
  reach the consumer.
- `ColorGradientLBMMethod`, state, runtime parameters, diagnostics, and
  initialization now use an ordered static set of two or more components;
  binary flow is the two-component case of the same pairwise API.
- `CoupledBulkSurfaceTransport` now advances extensive amounts on sparse
  topology, and surface plug flow composes `ConservationIMEXMethod`.
  `SurfaceMeshMotion` uses the exact finite-step dual-area GCL and rejects
  inadmissible motion or Courant violations transactionally.
- Sparse factorization stores factor structure rather than padded elimination
  schedules, derives numeric update targets at runtime, and charges actual
  retained and transient work. Film Newton solves use the declared accurate
  inexact-Newton tolerance rather than a stalled Eisenstat--Walker forcing cap.
- `SHAKERATTLEPlan` is a static plan whose masses and constraint callbacks bind
  in `prepare`; every migrated consumer receives rank, condition, projection
  work, and rollback evidence through `ConstrainedMechanicsEvidence`.
- `BiomembranePlan` declares remesh region/species identity and capacity, and
  `PreparedBiomembrane.propose_remesh` now accepts canonical shared
  `EdgeSplitProposal`, `EdgeCollapseProposal`, and `EdgeFlipProposal`
  transactions with sparse vertex and face transfers.
- New bubble, film, threshold, color-gradient, multiregion, foam, optics,
  rendering, and soap-film-tunnel qualification profiles are registered as
  unreleased candidates. Passing a campaign does not independently authorize
  release.
- Candidate-profile registration now uses canonical package owners:
  `phydrax.discretization.lattice_boltzmann` owns color-gradient profiles,
  multiregion geometry owns geometry/extraction/remeshing/topology profiles and
  foam mechanics owns only `foams.*` profiles.
- Plug-flow evidence identifies the terminal nonlinear stage explicitly;
  threshold label states/evidence expose their complete contract fields, and
  two-phase checkpoints bind the bubbly parameter-realization identity.
- Python support now targets 3.12 (`>=3.12,<3.13`). Runtime, optional, test,
  QA, and Python build dependency floors target the newest jointly resolvable
  stable releases, with ceilings at the next compatible release boundary. Both
  uv locks are refreshed.
- The accidentally reintroduced `chex` and `coordax` runtime requirements are
  removed; their former roles remain owned by native substrates.
- JAX 0.11 execution uses its final-style JAXPR and transpose contracts without
  the removed pxla primitive registry. OCP 8 collection and TopoDS downcast
  APIs replace their removed predecessors, and benchmark named arrays import
  `AxisArray` directly from the native axis substrate.
- Public runtime type aliases use Python 3.12 `type` statements, so Griffe 2
  resolves their package-owned identities without chasing `typing` internals.
- Static typing is restored across package, test, example, and qualification
  surfaces under NumPy 2.5 and JAX 0.11. Seventy-one obsolete suppressions are
  removed, and host metadata, dtype, optional-value, and provider boundaries
  now carry explicit contracts without changing numerical or evidence semantics.
- `SweptLayerControl` and `LayerTerminationPolicy` are replaced by
  `BoundaryLayerControl(wall, schedule, route=EXACT_SWEEP, volume_scope=...,
  cap_scope=...)`; `VolumeMeshingSpec.layer_controls` accepts
  `BoundaryLayerControl` values only.
- Polygon cells are certified without assuming star-shapedness: a planar polygon
  is `CERTIFIED_VALID` iff its vertices are finite and distinct, every edge
  clears the scale-aware floor, its boundary is simple, it is counterclockwise,
  and its area clears the determinant floor;
  embedded polygons must lie within the new
  `CellValidityPolicy.relative_planarity_tolerance` of their Newell plane and be
  simple in projection. `CellGeometrySpec.affine` binds polygon blocks to
  `CellVertexGeometryElement`. `phydrax.geometry` adds
  `segment_intersections_2d` (`SegmentIntersectionStatus`/`Result`) and
  `polygon_simplicity_2d` (`PolygonSimplicityStatus`/`Result`, bounded BVH edge
  broad phase with candidate-capacity evidence); the predicate owner moves to
  the private top-level `phydrax._geometry_predicates`.
  `CellMeshAuditPolicy.self_intersection` now defaults to `REJECT`,
  `maximum_intersection_candidates` bounds streamed polygon and triangle
  candidates, polygon loops are ear-clipped exactly for the self-intersection
  check, and `CellMeshAuditReport` reports `evaluated_checks` and
  `skipped_checks`.
- Automatic finite-volume remap consumes the canonical common refinement:
  `prepare_unstructured_conservative_remap(source, target, *, provenance, policy)`
  returns `PreparedUnstructuredConservativeRemap` (refinement, its
  `CommonRefinementStatus`/evidence, reason, and a plan only on success),
  replacing `build_unstructured_conservative_remap`, its private AABB/clipping
  loops, Mapping/attribute limit parsing, and
  `UnstructuredConservativeRemapBuildResult`/`Evidence`/`Status`.
  New bound-preserving second-order remap `UnstructuredSecondOrderRemapPlan`
  (vertex-stencil least-squares gradients as a sparse map, exact overlap
  first-moment integration, Barth-Jespersen `UnstructuredRemapLimiter`, bounded
  global conservation restoration, `UnstructuredSecondOrderRemapResult` evidence).
  Topology-event transactions take `remap_policy` instead of
  `remap_tolerance`/`remap_limits` and report `automatic_remap`.
- External meshing providers run as persistent native library-API workers
  (`native/providers/{mmg,omega_h,tioga,vorocrust}`, moved out of
  `phydrax/meshing/providers/native`; shared protocol headers in
  `native/providers/common`; wheels install them as `phydrax/native/providers`
  and `phydrax.meshing.providers.native_provider_source_path(provider)` locates
  them in either layout). Arrays travel through a binary, checksummed,
  memory-mappable exchange directory (`phydrax._external_exchange`: one NPY file
  per array plus a canonical manifest of name, dtype, shape, and payload
  SHA-256). `phydrax._external_runtime.NativeWorker` keeps one bounded worker
  process per provider session (reuse across calls, `close()`/context manager,
  per-call wall time, exchange byte bounds, address-space or peak-RSS memory
  limits, call-count lifetime) and records the worker's exact runtime identity
  once at startup; failures surface as `MeshingFailure` with the worker log tail
  and raw evidence on `__cause__`. No provider spawns a version process per call.
  Worker calls, `close`, and `abort` are serialized by a reentrant lock, and a
  provider's session creation, replacement, calls, and close by another taken
  first, so concurrent callers share one session safely. Collective workers
  (ParMmg, Omega_h, TIOGA) run every local parse, allocation, provider phase,
  extraction, and output stage through the same ordered rank agreement before
  the next collective call, so one rank's failure reaches every peer as the
  lowest failing rank's error instead of mismatching collectives.
- Mmg runs only through its library worker (the pymmg/medit executable route and
  the `meshing-mmg` extra are removed). `MmgProvider.adapt(source:
  CellMeshingResult, ...)` returns `MmgAdaptationResult`: multi-block/multi-zone
  region and facet-patch references are retained by name, required entities and
  ridges are honored, isotropic metrics travel as scalar sizes and others as
  tensors, `MmgLevelSet` discretizes level sets, `MmgLagrangianMotion` moves
  meshes when Mmg is built with ELAS (otherwise refused), declared vertex fields
  are P1-interpolated with `MmgFieldTransfer` evidence, and
  `-DPHYDRAX_MMG_WITH_PARMMG=ON` builds a collective ParMmg worker (ParMmg with
  its CMake package export: upstream `d2eddc5` or later; the 1.5.0 release
  installs none).
- Omega_h: `OmegaHProvider.execute(source: CellMeshingResult, metric, *, options,
  fields, ranks, gather, limits)` preserves blocks, zones, patches, and labels as
  Omega_h class IDs, passes `OmegaHOptions` AdaptOpts targets verbatim, transfers
  declared `OmegaHField`s (LINEAR vertex, CONSERVE cell) with integral evidence,
  and rejects an output metric outside the input field's hard size or anisotropy
  bounds instead of widening them. It returns vectorized per-rank
  `OmegaHPartition` ownership, ghost, and global-ID arrays; the global carrier
  and `MeshDistribution` are assembled only for serial runs or `gather=True`.
  `OmegaHProvider(timeout=)` is removed (use `MeshingLimits.maximum_wall_seconds`).
- TIOGA runs as a persistent collective worker: `TiogaProvider.move(previous,
  coordinates)` updates moving parts in the resident `TiogaRegistration` without
  restarting, donor records are parsed vectorized from CSR arrays,
  `TiogaOptions.timeout_seconds` is removed, and the license metadata is
  BSD-3-Clause. VoroCrust extraction is a persistent worker returning binary
  arrays assembled with NumPy and a vectorized alias merge; `VoroCrustProvider`'s
  second argument is the optional `worker`, preflight bounds are per emitted
  polytope, and `relative_merge_tolerance` defaults to 1e-9.
- Worker executables are located through `PHYDRAX_MMG_WORKER`,
  `PHYDRAX_OMEGA_H_WORKER`, `PHYDRAX_TIOGA_WORKER`, and
  `PHYDRAX_VOROCRUST_WORKER` (replacing `PHYDRAX_OMEGA_H_EXECUTABLE`,
  `PHYDRAX_TIOGA_EXECUTABLE`, and the VoroCrust extractor path).
- Mesh coordinate optimization runs through `phydrax.optim.minimize`
  (`ProjectedLBFGS` default, `NewtonTrustRegion` optional) on one stable compiled
  route with fixed vertices eliminated (bit-identical) and coordinate boxes
  enforced by projection. `TargetMatrixOptimizationPlan` selects a
  `MeshQualityObjective` (generalized Knupp shape, shape plus size, metric
  alignment, or the Gram-determinant energy) on triangle, tetrahedron,
  quadrilateral, and hexahedron corners and takes `coordinate_bounds`,
  `termination`, `method`, and `MeshUntanglingPolicy` instead of step-size and
  projection callbacks. Inverted input runs explicit Escobar untangling accepted
  only at zero inversions with a passing audit; failures return unmodified
  coordinates with a `MeshOptimizationStatus`, and `MeshOptimizationResult`
  carries the native `MinimizationResult`. `optimize_cell_geometry_coordinates`
  uses the same route and returns `CellGeometryOptimizationResult`
  (`optimizer_status`, `converged`).
- Mesh optimization no longer reports non-convergence as optimized: `OPTIMIZED`
  requires a converged native minimization, a valid (inversion-free, audited)
  non-converged iterate is `NONCONVERGED` and not accepted by default, and
  `TargetMatrixOptimizationPlan(accept_valid_nonconverged=True)` explicitly
  commits it as `VALID_NONCONVERGED`. `MeshOptimizationResult.accepted` is true
  exactly for `OPTIMIZED` and `VALID_NONCONVERGED`, the only statuses carrying a
  certified `result`. Untangling stages follow the same rule
  (`MeshUntanglingEvidence.optimizer_statuses`, `converged`). Mesh-motion
  relocation (bounded by `MeshMotionMonitorPolicy.relocation_termination`) fails
  on non-convergence unless
  `MeshMotionMonitorPolicy(accept_valid_nonconverged_relocation=True)`, and a
  non-converged curving relaxation cannot replace the accepted geometry unless
  `HighOrderCurvingPolicy(accept_valid_nonconverged_relaxation=True)`;
  `HighOrderCurvingResult` records `relaxation_statuses` and `accepted_round`, and
  a refused valid candidate rolls back as `ROLLED_BACK_NONCONVERGED`.
- Size resolution measures proximity gaps with exact BVH nearest queries
  (`normals` replaces supplied gap samples), grades hard growth limits by edge
  length through the metric owner, records the gradation evidence in
  `SizeResolutionReport`, binds `ResolvedSizeField.sample_entity_ids`, and
  compiles fields into metric constraints with `size_field_metric`. Mesh
  proposals project sizes and metrics through the metric owner and expose
  `MeshProposalProjection.metric_evidence`.
- Cell incidence construction is vectorized: polygonal, tetrahedral, hexahedral,
  mixed standard-block, and face-loop polyhedral connectivity, cell-block
  duplicate checks, hexahedral face tensor permutations, and `TriangleTopology`
  half-edge twins, boundary loops, face components, and vertex incidence use
  sorted canonical keys and array passes instead of per-cell Python loops.
  `OrientedIncidence` and `EntitySet` check uniqueness with sorted row keys
  instead of `np.unique` over rows. Entity ordering, global IDs, orientation
  signs, topology IDs, and the first reported defect are unchanged. `CellMesh`
  builds tetrahedral and hexahedral connectivity once, and
  `CellMesh.with_coordinates` reuses the prepared connectivity and topology. The
  unused `hexahedral_cell_complex` helper is removed. The benchmark case
  `incidence` scales construction with the cell count.
- The uniform full-depth `SparseLevelOctree`/`SparseLevelOctreePlan`/
  `SparseLevelOctreeEvidence` layout is replaced by the adaptive linear octree
  `AdaptiveOctreePlan(address_plan, leaf_capacity=..., balanced=False,
  separation_padding=0.0, u_capacity=..., v_capacity=..., w_capacity=...,
  x_capacity=...).prepare(points) -> AdaptiveOctree` in
  `phydrax.discretization.spatial`. Host preparation subdivides Morton-sorted
  points level by level until leaves hold at most `leaf_capacity` points,
  optionally enforces 2:1 leaf balance, and stores only the refined nodes and
  their child blocks (empty siblings are leaves, so `locate` finds the leaf of
  any domain point) with parent, child span, point span, level, and cell
  geometry. Adaptive-FMM U/V/W/X lists are target-major `EdgeRelation` routes
  with row offsets, bounded capacities, and `overflow`/`required_routes`
  evidence; `separation_padding` keeps one cell of clearance for displaced
  sources. `LaplaceMultipolePlan3D`, the radial Helmholtz plans, and
  `VortexFMMPlan` with `execution="level_octree"` now prepare this tree over
  the reference sources (leaf capacity `source_leaf_occupancy`/`leaf_capacity`),
  locate targets at evaluation, add X-list P2L and W-list M2P routes, and report
  `p2l_count`/`m2p_count`. `far_local` and `prepare_laplace_qbx_far_local_3d`
  return far-only local expansions centered at the requested target centers.
- Packed BVHs have one host construction entry point, `prepare_bvh(bbox_min,
  bbox_max, policy=BVHBuildPolicy(kind, leaf_size, sah_bins), dtype=...)`, with
  `BVHBuildKind.MEDIAN` (level-synchronous median splits), `MORTON` (Karras radix
  tree over sorted 30- or 63-bit Morton codes of box centers), and `SAH` (binned
  surface-area heuristic); ties are ordered by item index and narrower storage
  dtypes round bounds outward. `build_packed_bvh` and `build_point_bvh` are
  removed (points use `prepare_bvh(points, points)`). `PackedBVH` is level-ordered,
  stores its item boxes, and `refit_packed_bvh_bounds(bvh, item_min, item_max)`
  returns the refitted hierarchy with one vectorized, differentiable update per
  level (`reduce_packed_bvh_nodes` exposes the level sweep). New queries:
  `bvh_nearest_items` (exact k nearest items), `bvh_hierarchical_sum`
  (Barnes-Hut style cut sums), `bvh_overlap_pairs` (JAX simultaneous traversal
  with pair capacity, int64 `count`, and `overflow`), and `bvh_overlap_pairs_host` /
  `bvh_overlap_pair_blocks` (complete exact host pairs, sorted or in bounded
  blocks). `query_host_aabb_overlaps` uses the host pair search instead of an
  all-pairs loop, with unchanged statuses, limits, and ordering, and
  `HostAabbOverlapBvh` carries its prepared `packed_bvh`. `TriangleBVH` takes a
  `BVHBuildPolicy`, uses `max_depth + 2` traversal stacks, and adds `refit`,
  `nearest_faces`, exact `winding_number` (hierarchical closing fans), and the
  approximate `fast_winding_number` (Barill et al. dipoles), both returning a
  `WindingNumberResult` with its `WindingNumberRoute`. `MeshRegion` distance,
  closest-point, and inside queries refit that hierarchy instead of forming
  query-by-face arrays, and `triangle_query_evidence(points, selected_triangles,
  closest_points, second_distance_squared)` takes the two nearest distances
  rather than dense per-face closest points. Benchmark cases `lbvh` and
  `overlap-pairs` scale construction, refit, kNN, and pair search.
- `MortonNeighborQueryPlan` and `MortonRadiusRelationPlan` prune with coarse
  Morton cells instead of a plane schedule: sources are sorted once by Morton
  code, each target visits its cell's `3^d` stencil at an adaptive level, and
  distances run only over a fixed `maximum_candidates` buffer in bounded target
  chunks (`target_chunk_size`), so no target-by-source product is formed. k-NN
  selections are certified against the visited region and retry once at a
  coarser level. Results carry a per-target `status`
  (`MortonNeighborQueryStatus`); evidence reports `complete`, `sources_valid`,
  `overflow_rows`, and `uncertified_rows` instead of plane-node counts. Radius
  results expose `logical_cell_slots`, `cell_counts`, and `cell_offsets`. The
  `maximum_nodes`, `maximum_leaf_occupancy`, `coarsening_factor`, and
  `target_top_nodes` options are removed from these plans,
  `DistributedMortonNeighborQueryPlan`, and `MortonTreeParticleNeighborhoodPlan`
  (which gain `maximum_candidates`); `PeriodicFoFFinderPlan` replaces
  `morton_maximum_nodes` and `morton_leaf_occupancy` with
  `morton_maximum_candidates`.
- `DDGOperators` accumulates cotangent edge weights over the triangle topology's
  half-edge-to-edge map instead of a dense face-corner-by-edge match.
  `phydrax.graph.mesh_cotangent_weights` and `mesh_to_cotangent_graph` now take
  their edge weights and lumped vertex masses from `DDGOperators` on a validated
  `TriangleMesh`, so degenerate faces, duplicate faces, and non-manifold edges
  raise `ValueError` instead of silently receiving zero cotangent weight.
- The dense control linearizations `linearize_discrete_dynamics`,
  `linearize_differential_dynamics`, and `linearize_control_dynamics` are built
  from `PreparedLinearization` and `JacobianLinearOperator` and require a
  `materialization: MaterializationPolicy` that bounds each dense Jacobian family
  over the whole case batch. A failed discrete transition now yields NaN matrices
  instead of a finite zero Jacobian.
- The native dense QP interior-point kernel, its independent audit, and the
  active-set KKT tangent solve are compiled once per static program layout, so
  repeated solves and sensitivities at fixed structure (such as MPC windows of one
  topology) no longer retrace and recompile.
- Removed `LinearSolveHistory`, `LinearSolveHistoryPolicy`, `solve_with_history`,
  `HistoryLinearSolveResult`, `LinearInitialGuessDiagnostics`, and
  `LinearInitialGuessStrategy`. Use `HistoryInitialGuess(operator, family_id,
  strategy=...)` as a solve `initial_guess`; `at_time(t)` sets the extrapolation
  target, and the unused reorthogonalization control is gone.
- `phydrax.lifecycle.NumericRevision` is removed; `phydrax.NumericRevision` is the
  one numeric-content identity. Lifecycle ancestry is the new
  `lifecycle.RevisionLineage`, which references canonical semantic and revision
  IDs plus a label, metadata, and one parent lineage. Lineage archives store the
  revision's numeric content and recompute it on open; archives with the old
  numeric-revision record are refused. ROM models and archives, IGA compatible
  qualification and transfer plans, Fourier-modal revisions, and atomistic label
  sets use canonical revisions (`fourier_modal_numeric_revision` drops `label`,
  `qualify_compatible_complex` recomputes the complex revision,
  `ReducedBasisArtifact` drops `numeric_revision`, and IGA transfer plans record
  `source_content_id`/`target_content_id`).
- Atomistic potential identity derives from the current PARAMETER lane:
  `atomistic_potential_revision` replaces `checkpoint_atomistic_potential`, and
  `parameter_state_tree`, the patched `parameter_state_id`/`potential_id` fields,
  and `AtomisticProvenance.parameter_state_id`/`potential_id` are removed
  (provenance records `potential_revision_id`).
- `ExecutionWorksetCheckpoint` and `restore_execution_workset_checkpoint` require
  the `numeric_revisions` bound into the items and refuse a restore under other
  revisions. Operator training checkpoints require declared array roles and take
  `static_callables`.
- `OperatorArchitectureCodec` and `register_operator_architecture_codec` are no
  longer re-exported from `phydrax.nn.operator.training`.
- Phydrax-native fixed-topology message passing (MeshGraphNet, attention,
  kernel and neural operators, equivariant, relational, hypergraph, simplicial,
  DEC harmonic projection, cluster pooling, GCN/SAGE/GIN, `MessagePassing`)
  gathers and reduces over sparse `EdgeRelation` routes with inert masked
  routes; the jraph-compatible family keeps segment aggregators.
  `GraphKernelIntegral` and `GraphNeuralOperator` take `reduction=` instead of
  `aggregate_fn`, and `MessagePassing.aggregate` is removed.
- `DAFoamTotalDerivative.values` is a read-only array in the design variable's
  shape, and `run_dafoam` request identity includes design shapes.
- Every native trainer (functional solvers including KFAC, evolution, windows,
  decomposition, variational Monte Carlo and Calabi-Yau; operator fitting;
  discrete, variational, and neural-CDE identification; kinetic rollout
  closures; atomistic and free-energy fitting; flows, variational inference,
  sparse GPs, buffered state space, targeted maps, SING, PGM; Stefan and
  learned PIV) runs through one accepted-update training kernel. Attempts end
  accepted, rule-rejected (only rule-authorized state commits), or nonfinite
  (full rollback); unsuccessful and nonfinite updates are never committed.
  Parameters train only under objectives their component authority admits.
- Training and execution random keys use semantic sample addresses; training
  checkpoints persist the kernel state and previous checkpoint formats fail
  closed. `FunctionalUpdateKernel`, `training_key`, and `TrainingController`
  key handling are removed.
- Numerical flux plans, face reconstructions, and slope limiters are neutral
  `DISCRETIZATION` component slots. Mapped structured finite volumes admit any
  arbitrary-normal flux (including HLLC and the all-speed fluxes) and refuse
  axis-only fluxes at preparation; face closures now apply on mapped geometry.
- `ConservativeFaceClosurePlan` and its Cartesian-axis correction ABI are replaced
  by `ArbitraryNormalFaceClosurePlan`; corrections receive a `FaceFluxContext`
  instead of an axis. `LearnedClosureBindingPlan.bind_conservative_faces` verifies the
  predictor's `conservative_face_numeric_revision` before binding.
- The conservation-source callable type `SourceFunction` has one owner shared by the
  structured, block-AMR, triangle, unstructured, and SBP dynamics.
- `FieldCertificate` owns the Lipschitz upper bound, evaluation-error bound, and
  topology identity; `ExactSDFEnclosureCertificate` carries only its field
  certificate and requires those bounds on it.
- `PreparedFiniteElementPointInterpolation` and
  `prepare_finite_element_point_interpolation` live in the FE point-evaluation
  module; rigid attachments check their nodal vector-field layout themselves.
  `InterpolationTransposeEvidence` is exported from `phydrax.discretization`
  only.
- Models with intrinsic ports require an explicit `PortMapping` at
  `Domain.Model`, model systems, operator context/execution plans, geophysical
  bindings, and learned-stress bindings. Operator output name maps are replaced
  by port mappings; fitted ML schemas can be built from owner ports.
- Derivative planning admits field regularity before tracing: proven
  degeneracy (for example a ReLU network with linear output under a Laplacian)
  is rejected, and `FunctionalSolver(regularity_policy=...)` declares whether
  almost-everywhere and undeclared regularity are admitted.
- Nonlinear precision policies compose declared component error floors,
  excluding accelerators, and reject unreachable tolerances. Implicit roots
  and Newton preparation admit component randomness only when deterministic,
  in inference state, or bound to a `FrozenRealization`.
- Trainability is declared, not inferred from dtype. Arrays carry an
  `ArrayRole` (parameter, fixed, model state) from field declarations
  (`parameter_field`, `fixed_field`, `model_state_field`), `ParameterOwner`
  model bases, and terminal `NonTrainableState`/`ExplicitFreeze` markers.
  Every training entry runs `require_parameter_roles`, which rejects
  unclassified arrays, parameters hidden under a fixed ancestor, and callables
  capturing undeclared arrays. `partition_parameters` returns parameter,
  model-state, and fixed lanes; `partition_trainable`, `combine_trainable`,
  and the trainable-leaf predicates are removed.
- Learned-component slot bases (numerical fluxes, reconstructions, limiters,
  face closures, step and stage transforms, transport and MPM constitutive
  plans) and their method/dynamics containers are neutral; built-in analytic
  leaves remain fixed. Frozen artifacts (`FrozenModel`, `TrainedOperator`,
  operator correction bindings) are explicit freezes. Operator batches,
  context sources, normalizers, scalers, and fitted ML statistics are fixed.
- Accepted-step and SSP stage transforms are `DISCRETIZATION` component
  slots, and discrete model rollout transitions are a `MODEL` component slot;
  built-in transitions are fixed. Fixed-step methods validate every direct
  accepted-step transform result centrally (structure, dtype, scalar
  evidence), and a failed transform can no longer change the candidate.
  Learned DAE defects enter through the residual and the existing implicit
  and adaptive acceptance lifecycle; `DAESolvePolicy` stays a fixed policy.
- `ParameterSubspace` selections are role declarations; worksets and
  ensembles map lanes through a declared `LaneLayout`, independent of roles.
  `FunctionalSolver.partition_functions` returns three lanes.
- `EquinoxModel` rejects stateful Equinox modules.
- ML fitting uses the canonical derivative vocabulary: `FitResult` exposes
  `derivative_contract`, `derivative_admission`, `require_derivative`, and model
  ports; `fit(..., derivative_request=...)` replaces the ML gradient request.
  The ML-specific gradient contract types are removed.
- Fitted ML executables retain feature/target schemas (with optional physical
  dimensions) and ports. Native ML artifacts persist the canonical contract,
  schemas, ports, and semantic/numeric/executable identity; `load_ml_model`
  returns the bound executable and previous artifact records fail closed.
- Scientific artifact evidence uses `DerivativeContract`; the local
  differentiation contract and `DerivativeAvailability` are removed.
- Branch sensitivity, conservation, spectral, particle, reconstruction, and
  filter differentiation policies are unified as
  `BranchDifferentiationPolicy`; each owner accepts its supported subset.
  `HybridSensitivityMode` and the owner-specific policy types are removed.
  Fingerprints that hashed the previous policy spellings change.
- Public API inventory now follows canonical access paths rather than private
  implementation module names, traverses without a depth cutoff, and includes
  explicitly supported lazy scientific leaves. Documentation directives,
  examples, and capability declarations now fail closed when they reference a
  private or undeclared path.
- Added the supported `StrictModule` extension import and public diagnostics
  module; completed explicit HFSS, Geant4 detector, discrete-velocity, and
  multiwavelet exports; removed the unintended `ein.get_symbol` re-export.
- Replaced required ModePy nodal preparation, Matfree least-squares and low-rank
  routes, Polars CSV ingestion, TensorBoard scalar writing, ASDEX structural
  tracing, Evosax distribution search, SymPy exact algebra, and FlowJAX density
  models with Phydrax-native substrates. Triangle import and planar boundary
  extraction no longer require Trimesh or Shapely; PyVista, ArviZ, and Manifold
  are explicit optional providers. Build123d was replaced by the directly
  consumed no-VTK OCCT binding.
- Matrix-function results now expose portable status, nested diagnostics, and
  typed method/operator/plan provenance. Affine, activation, and semilinear
  exponential updates share one augmented action rather than constructing
  independent exponential and phi projections.
- Blockwise model bindings now declare their output layout
  (`dependency_axes`, `dependency_subset`, or `axis_array`). Output axes come
  from that declaration, the vmap schedule, or coordinate dependencies; array
  sizes only validate a declaration and never establish axis identity.
- Domain-backed integration targets accept a `DomainFunction` or a constant.
  A raw callable now raises `TypeError` naming `domain.Function(*labels)(f)`;
  raw-callable engines (mapped, breakpoint, external, adaptive callable) keep
  their callable interface.
- Callable identities in residual relaxation, cochain residuals,
  astrodynamics and GR events, Maxwell sources, probabilistic ODE drifts,
  variational Monte Carlo, and robot environments use the canonical
  callable payload. Opaque callables require explicit semantic and numeric
  identifiers; `repr`, source-location, and type-name identities are removed.
- Triangle and unstructured finite-volume methods accept any
  arbitrary-normal numerical flux; moving and overset unstructured routes
  require the new `AbstractArbitraryNormalALENumericalFluxPlan`.

### Removed
- Duplicate exterior-basis tables, graph cochain calculus/spectral/harmonic carriers,
  diagonal-only cochain metric state, specialized RT/BDM/Nédélec factories, compatible
  tensor and spline Piola/transfer shims, and redundant gauge path implementations.
- Orphaned adaptive-campaign runner/profiler references to unavailable drivers.
  Typed JSONL scientific analysis and promotion-evidence contracts remain supported.
- Removed `TwoPhaseImplicitViscosity`, `TwoPhaseViscousResult`, the local
  two-phase viscosity module, obsolete total-energy/limiter/body residual
  fields, and the unused `MultiphaseFLIPPlan.surface_tension` argument.
- Removed the rolled-stencil `thin_film_pressure`, dense
  `transfer_surface_content`/`surface_transfer_balance`,
  `CoupledBulkSurfaceTransport.create`, its dense generator matrices, and
  `SurfaceTransportResult.positivity_preserving`; moving-surface transport now
  exposes candidate/accepted state and an explicit status.
- Removed color-gradient LBM red/blue state, density, mass, defect,
  scalar-tension, and scalar-contact-angle fields; component and canonical-pair
  arrays are now the only API.
- Removed `BiomembraneRemeshOperation`, the forwarding
  `propose_split`/`propose_collapse`/`propose_flip` methods, local remesh and
  triangle-intersection implementations, and dense remesh transfer matrices.

### Fixed
- Tetrahedral H(div) reference-face routing and nonvacuous shared-normal continuity.
- Physical magnetic-flux gathering in passive and dispersive PIC media.
- Large-prime cup-product accumulation and topology identity checks, repeated
  cellular-sheaf attaching occurrences, and point-cloud topology fingerprints.
- Sparse topology admission under compile-time evaluation, reusable traced sparse
  storage, native complex strict-dtype solves, batched tiny linear algebra, and
  generalized eigensolver dependent-direction handling.
- Scientific FE trace identity for selected boundary closures, form spectral lifting,
  full-Gram spectral normalization, and low-precision Adam bias correction.
- High-dimensional cubic tensor-form duality uses stable separable interval
  preparation instead of an ill-conditioned multidimensional monomial solve.
  Piola mapping reuses factors across basis and fiber right-hand sides.
- Full-coordinate FE differential/Laplacian actions and compact relative PIC
  Poisson projection preserve principal metric restriction and physical charge.
- Native restarted Krylov solves retain unused iteration budget after a projected
  convergence nomination fails the actual residual check.
- IAS15 prepares real Jacobi–Radau stages rather than depending on complex-valued
  polynomial-root storage. Real orbital recurrences retain their declared dtype.
- Compiler evidence no longer claims estimates are unavailable when official cost
  and memory analysis supplied them.
- Newton implicit roots no longer fail with a `lax.cond` pytree mismatch in
  DAE replay and large-offset BDF increments.
- `DAEContinuation` carries the modified-Newton refresh triggers, so
  continued adaptive windows reproduce uninterrupted runs.
- Default simplicial point location no longer reports `LOCATION_FAILED` for
  interior points of small FE meshes (BVH point queries test item boxes).
- Field algebra between a discrete field view and a `DomainFunction` under
  `jit` no longer converts traced domain bounds.
- B-Rep projection bounding boxes work with OCP 8 (`CornerMin`/`CornerMax`).
- `prepare_scalar_calderon_3d` gives its P1 space its own identity (complex
  kernels no longer crash); Buffa–Christiansen barycentric refinement uses
  the correct midpoint vertex indices.
- The VEM sparse realization no longer runs host validation inside jitted
  paths.
- Partitioned fixed-point windows replay the Gauss–Seidel sweep in their
  final certification evaluation; adaptive windows without a reliable error
  estimate no longer report success.
- `TopologyEpochTransition.composition_transport` tolerances scale with the
  transported magnitudes.
- Coupled certificates (component rows, law defects, 2-D exterior
  compatibility, transient rows) scale by uncancelled term magnitudes, so
  exact states whose owner terms cancel are accepted.
- The Riesz-identified coupled `LinearSystem` and lane actions publish the
  exact coordinate transpose `M^{-T}`.
- Interface bindings are audited against each law side's trace sites and
  normals; interface quadrature enforces a per-facet tiling; matching
  elimination refuses or imposes strong crosspoints; certified trace-inverse
  constants refuse flux/energy pencils whose kernels disagree.
- Coupled derivatives refuse raw runtime arguments that no parameter binding
  supplies and solver-argument declarations that structurally enter the
  operator; `refuse_derivative_dependencies` refuses every derivative order;
  FEM–BEM results guard every derivative-bearing output.
- Lane worksets certify custom differentiation rules and host callbacks in
  their executable signatures and refuse uncertified higher-order
  derivatives; `solve_coupled_problem` defaults to DenseLU with
  `route="primal-factors"` when a dense plan fits.
- Partitioned coupling: exact waveform integration, steady-response
  checkpoints, native step-index continuation, conservative certification of
  dimension-changing transfers, scale-aware ledger tolerances, exact host
  window grids, per-participant host rollback, and FMI time-unit checks.
- Lifecycle refreshes cannot rebind state onto new structure; topology epoch
  transitions honor owner-certified conservation bounds; exchange transports
  verify epoch provenance.
- Curved simplicial cells use certified Bernstein bounds for BVH pruning;
  prepared queries keep native complex coefficient dtypes and publish a
  realified operator for real-output complex reconstructions.
- `PhysicalPODPlan` decides its rank above a relative numerical-rank floor of
  the method of snapshots.
- Observation ports respect validity masks and transient capacity terms;
  dense-precision covariance restriction is symmetrized; the coupled
  transition kernel's log density uses an exact triangular factor.
- `solve_factored_matrix_equation` certifies the Frobenius residual on the
  QR-reduced low-rank core instead of a squared Gram trace, so exact
  low-rank ADI solutions are no longer reported as
  `RESIDUAL_TOLERANCE_NOT_MET`.
- The strict documentation build renders runtime type aliases
  (`Literal[...]`, parameterized generics, PEP 695 `type` aliases with
  forward references) through the `RuntimeTypeAliases` griffe extension
  instead of failing alias resolution.
- The PIC deposit↔Gauss preparation certificate no longer divides numerical
  roundoff by an almost-zero charge change when a high-capacity periodic probe
  uniformly fills the grid. It subtracts the same charge-scaled roundoff floor
  used by runtime continuity evidence; a 512-slot probe on 64 cells now certifies
  at roundoff while deliberately nonconserving deposits are still refused.
- Public radiation documentation and the QED cascade example now import bounded
  resources and `StrictModule` from public facades. Generated external-provider
  scripts use explicit NumPy dtypes, and the Geant4 electromagnetic constructor
  is selected by a fail-closed exhaustive branch instead of dynamic attribute
  lookup.
- Axis domains, axis materialization, tensor-grid bounds, and broadcasted
  coordinates request JAX's canonical available real precision rather than
  forcing unavailable float64 values. x64-disabled executions now remain
  float32 without precision-truncation warnings. Cherenkov regime preparation
  also validates and caches continuum permeability once instead of synchronizing
  once per frequency during evaluation.
- The rank-deficient height-function fallback regression evaluated a
  `StructuredPLICReconstruction` of one volume-fraction field against a
  different `alpha`, so the reconstruction's Youngs-supported mixed cells fell
  outside the query's interface band and `CurvatureEvidence` correctly refused
  the inconsistent interface delta before any fallback evidence was produced.
  The regression now reconstructs the same diffuse plane it evaluates: with
  every primary facet the fallback fit has full rank, and with only one column
  of collinear facets it reports rank one and `UNDERRESOLVED`.
- Sparse LU/Cholesky symbolic factorization took an active `MaterializationPolicy`,
  a dense-materialization permission, as its factor ceilings: `max_entries`
  capped factor nonzeros, `max_bytes` capped retained bytes, and
  `4 · max_entries` capped symbolic work. A matrix-free Krylov solve that forbids
  dense materialization (`max_entries=1, max_bytes=16`) therefore refused its own
  sparse ILU preconditioner ("factor_bytes requires 6499, exceeding limit 16"),
  breaking the semiconductor small-signal, sensitivity, and circuit operating-point
  solves. The default `MaterializationPolicy` of every solve also silently halved
  the sparse factor's own nonzero ceiling. `prepare_sparse_factorization` and
  `factorize_sparse` no longer take `materialization=`; fill is bounded by the
  `SparseFactorizationPolicy` ceilings, and a solve plan charges the retained
  factor to `SolveResourcePolicy.preconditioner_bytes`, refusing above it.
- `phydrax.linalg.evaluate_pfaffian` no longer reports skew matrices whose
  largest entry exceeds `1 / finfo.tiny` (2^1022 in float64, 2^126 in float32)
  as singular. XLA lowers the internal normalization by that scale to a
  multiplication by its reciprocal, which is subnormal and flushed to zero on
  CPU, zeroing every pivot. The normalization scale is now capped at
  `1 / finfo.tiny`, so normalized entries stay below 4; values, signs, log
  magnitudes, and derivatives are unchanged for all other inputs.
- `jax.linearize` and `prepare_linearization` of vmapped implicit roots no
  longer raise `NotImplementedError` for `jax._src.hijax.VmapOf`. JAX 0.11's
  `lax.custom_root` batches into a HiJAX primitive without a linearization rule,
  which broke implicit MPM plane-stress steps and semiconductor equilibrium
  Newton Jacobians/sparse derivative plans. Package roots (plane stress,
  semiconductor thermodynamics, moist adjustment, Liénard–Wiechert retardation,
  FEM local materials, local root plans, nonlinear/optim implicit results,
  linalg implicit solves, hybrid events) now use a Phydrax custom-JVP
  implicit-function rule with the same values and derivative contract;
  `jax.lax.custom_root` is a banned API.
- `CompatibleMaxwell1DPlan`/`CompatibleMaxwell2DPlan.stable_dt` and the
  high-order TENO `TENOQualification` residuals and `passed` flag are host
  Python scalars. They were NumPy scalars in static fields, so every
  construction warned "A JAX array is being set as static" (NumPy scalars are
  array leaves to Equinox). Values and fingerprints are unchanged.
- Spectral Huygens boxes on `grid="collocated"` are refused. The collocated
  half-cell spectral centering of deposited edge currents leaves current on
  every node along each current's axis, so no surface is current-free, and
  far fields depended on the source's sub-cell position (a 20–50 % dipole
  error that did not converge). The old `J = 0` check inspected only the edge
  current. This caused the full-wave FEL's Huygens/A1 drop to 0.76 after a
  half-cell origin shift.
- Compiled quadratic/cubic spline PIC steps rejected every step for
  CONTINUITY on larger periodic grids (e.g. 24×24×12, 48×48×24) when particles
  sat exactly on spline knots: XLA recomputed the normalized coordinate of
  `_uniform_axis_stencil` per fusion, so a knot particle's support base and its
  weights came from different roundings and its charge shifted by one vertex.
  The coordinate is now materialized once. The electromagnetic PIC continuity
  and particle↔field charge gates are relative to the new
  `PICFieldDeposit.continuity_scale` (unsigned charge-rate magnitude, reported
  as `ElectromagneticPICDiagnostics.continuity_scale`) with tolerance
  `max(continuity_tolerance, 64 ε)` instead of an absolute `1e-9`;
  `ChargeConservingCurrentPlan` drops its dimensional `max(1, ·)` floor and
  reports `PICCurrentDepositResult.continuity_scale`.
- The PIC deposit↔Gauss pairing check runs one module-level compiled probe per
  species instead of eager op-by-op dispatch. Plan preparation over an already
  prepared solver: 32×16×16 PSATD 9.8 s → 1.5 s, 24×24×12 quadratic Yee
  5.5 s → 1.6 s; cold first build including compilation 29.6 s → 9.3 s and
  16.8 s → 10.7 s.
- Compatible Maxwell current↔medium coupling is second order. The Lorentz–Drude
  and magnetic-pole ADE advanced with the pre-constraint `D`/`B`, so PEC (PMC)
  walls drove a spurious O(Δt) wall polarization every step; conduction and
  impedance-wall currents were sampled at the kick start (forward Euler) and
  magnetic conduction at `H_n` and `H_{n+½}`, leaving an O(Δt) endpoint term
  `−(Δt/4)σ(|E_N|² − |E_0|²)` against the trapezoidal loss ledger. Media now
  advance with the boundary-constrained fluxes, and conduction uses the
  predictor-midpoint field (magnetic conduction the trapezoid of the two half
  kicks). PIC energy ledgers now fall 4.0× per Δt halving for Lorentz–Drude
  (was 2.74 → 2.45 → 2.25), negative-index (2.89 → 2.33), magnetized plasma, and
  conductors (2.09 → 2.02); prescribed-charge ledgers 4.0× for conductors (was
  4.5 → 3.6 → 3.2) and impedance walls. Updates without dispersive media,
  conduction, or impedance walls are bitwise unchanged.
- `CochainElectrostaticPlan` solves with native PCG by default and certifies a
  relative residual. Lineax CG stops on an elementwise max-norm test while the
  linear runtime certifies the Hodge 2-norm residual, so non-neutral charge on
  bounded 3-D grids of 12³ and more returned `RESIDUAL_TOO_LARGE` (32³: 2.5e-9
  against a 1e-10 threshold). `tolerance` is now relative to the Hodge norm of
  the assembled right-hand side (charge plus Neumann source minus the Dirichlet
  lift, floored at the runtime's roundoff level) with no absolute term, in both
  the linear policy and the plan's own `converged` check: with SI permittivities
  the Dirichlet-driven right-hand side is ~1e-10, so an absolute tolerance
  accepted PCG after a few iterations and the semiconductor detector weighting
  potential came out `[1, 0.667, 0.333, 0, …]` instead of `1 − x/L`. The lift
  now carries Dirichlet values on fixed vertices only (free-vertex values are
  not constraints and would cancel the right-hand side to roundoff).
- `examples/dispersive_pic_cherenkov.py` evidence is consistent: quadratic
  shapes (order-one Whitney gathers jump at each face the beam crosses, a
  first-order self-field work defect that left the ledger 23% open), runs at
  `Δt` and `Δt/2` whose ledgers fall 4.45× within the Richardson second-order
  bound, and compares the first-harmonic cone (43.26°) with the dispersion
  audit's discrete cone at the same `Δt` (43.10°) as well as the continuum
  (42.50°).
- `NonlinearComptonPlan` optical-depth Monte Carlo is an unbiased renewal
  process: after a crossing, the next target is a fresh draw minus the
  overshoot, and a second crossing in one subcycle stays pending instead of
  being discarded. Before this fix the emission rate and radiated energy were
  biased low by `(1 − e^{−p})/p` (−4.9 % at the default
  `maximum_event_probability = 0.1`), including for the polarized and spin
  models. `NonlinearBreitWheelerPlan` carries the overshoot the same way after
  a refused draw.
- Order-one cochain PIC gathers E and B with the lowest-order Whitney forms
  (degree zero along the entity's own axes) instead of multilinear
  interpolation between entity midpoints, so the gathered field does exactly
  the work of the charge-conserving current; `TensorBSplineSplatAssignment`
  admits per-axis degree zero.
- `phydrax.sparse` structural sparsity detection
  (`compile_sparse_jacobian(..., compiler="auto")`) shared one const/bound state
  across every invocation of a nested jaxpr. JAX binds one jaxpr object for
  repeated calls of a function (every `jnp.where` shares `_where`), so an index
  array resolved at one call site was read back as the static index of another
  call whose index was unknown, and gathers/scatters there produced a wrong
  pattern. The Maxwell curl-curl operator (Equinox bounds checks in the
  codifferential) lost most entries: the assembled operator had relative error ~1.
  Every nested jaxpr invocation (`jit`, custom JVP/VJP calls, `cond` branches,
  `while`/`scan` bodies) now owns its scope, seeded only from its captured
  constants and actual arguments, and publishes known results to the call site;
  `cond` with a statically known index traces only the taken branch, and
  `reduce_or`/`reduce_and`/`reduce_xor`/`unvmap_any`/`unvmap_max` propagate known
  values. Statically passing `eqx.error_if` checks therefore keep checked indices
  exact instead of dense. The scatter position map is vectorized.
  `FrequencyMaxwellOperator(...).solve(method="direct")` now uses the traced
  pattern; its explicit incidence-product pattern (and the diagonal-Hodge
  restriction it imposed) is removed.
- Host `SparseLU` (`provider="auto"`/`"scipy-superlu"`) estimated its factor as
  the dense `n²` bound (1.4 GB for 9k complex unknowns), so the default
  `SolveResourcePolicy` refused realistic grids. Planning now owns the column
  ordering through `analyze_sparse_lu` (approximate minimum degree of `AᵀA` from
  the rows of `A`, column elimination tree, Gilbert–Ng–Peyton column counts),
  bounds `nnz(L + U)` by George–Ng's `2 nnz(chol((AQ)ᵀ(AQ)))` for every pivot
  sequence, and SuperLU factors `A[:, Q]` with its natural ordering. On 2-D/3-D
  Laplacians and curl-curl operators the bound is 1.7–2.1× the executed fill and
  the owned ordering's fill is 0.85–1.07× SuperLU's COLAMD. The plan carries
  `LinearSolvePlan.sparse_lu_analysis` (`SparseLUSymbolicAnalysis`), and the
  factorization budget still refuses above the symbolic bound. The moving-charge
  tests and the frequency-domain Cherenkov example no longer raise the budget.
- Fourier-modal Maxwell layers composed a mixed `[E_left, H_right] ↦ [E_right,
  H_left]` boundary relation, and reconstructed interior and interface fields by
  inverting its `H_right ↦ H_left` block. That block decays like the evanescent
  transmission, and the mixed relation has poles at sub-slab cavity resonances.
  Metallic and perfect-conductor-like patterned layers, even under plane-wave
  excitation, and dielectric gratings at ≥15 harmonics under the moving-charge
  Bloch wavevector `ω/v` (condition number 4.5·10¹⁵) therefore failed with
  "Linear solve failed". `BoundaryRelation` is now the power-wave scattering
  relation `s11, s12, s21, s22`, with `f, g = (e ± Jh)/2`. It is contractive for
  passive slabs and composed by the Redheffer star product.
  `AffineBoundaryRelation` carries `forward_source`/`backward_source`.
  Solve-time boundary fields and `fields_in_layer` close each plane from both
  sides with prefix and suffix relations; continuous layers retain
  `segment_suffix_boundaries`. The shared dense solve is compiled once per
  shape, which makes a 41-harmonic solve about 0.4 s instead of 20–40 s.
- `VectorFourierFactorizationPlan` applied the inverse rule to the component
  tangent to the interfaces and Laurent's rule to the normal one, the reverse of
  Li's rules. Lamellar TE and TM efficiencies therefore did not converge to the
  correct values. It now uses `⟦1/ε⟧⁻¹ + (⟦ε⟧ − ⟦1/ε⟧⁻¹)⟦t tᴴ⟧`. A metallic
  lamellar grating (index 0.22 + 6.71i) now reproduces the Lalanne–Hugonin (2000)
  inverse-rule efficiencies to 10⁻⁵ at 21, 41, and 81 harmonics, and a lossless
  ε = −10⁴ grating conserves energy to 10⁻⁹. The silica Smith–Purcell grating
  converges by 21 harmonics (21 versus 61 agree to 3·10⁻⁵). Its point-charge
  energy is 0.8 % below Szczepkowicz–Schächter–England (2020), and its line-charge
  up/down split matches the cochain frequency-domain solver.
- Integrated-Green-function field kernels (`FreeSpaceConvolutionPlan`
  `"coulomb-igf"`/`"newton-igf"` with `gradient=True`, used by
  `SpaceChargeIGFPlan`) were the cell average of `∇g`, the exact field of a
  cell-constant density: it has no self-cell field and drops the near-zone
  field of the density gradient inside a cell. On cells longer than the
  source is wide (relativistic rest frames) that zone carries most of `E_z`:
  a Gaussian bunch at four cells per σ on every axis read `E_z` 9 % low at
  aspect ratio 10, 42 % at 100, and 62 % at 1000, converging only
  logarithmically. The kernels are now `[P(m + ½e_a) − P(m − ½e_a)]/h_a`, the
  exact field of the density linear between sites along the derivative axis
  (the multilinear deposit) and cell-constant across it, equal to direct
  integration of that density to 10⁻⁹ at aspect ratios 1–1000 and
  second-order accurate (1–4 % at four cells per σ) at any aspect ratio.
- The time-domain CFS-CPML of `CompatibleMaxwellPlan` and the reduced 1-D/2-D
  Maxwell blocks read their convolution memories half a step off the kick's
  time level (the electric kick used `ψ(t_{n+1})`, the opening magnetic half
  kick `ψ(t_n + Δt/2)`), a first-order error that made normal-incidence
  reflection proportional to `Δt` (1.3e-3 at CFL 0.9 for a 15-cell layer
  calibrated to 1e-4) and kept the prescribed-charge CPML power ledger from
  converging. Each memory now advances by two exponential half-step
  recursions per step and every kick reads it at its own time level; the same
  layer reflects 4.4e-5 independent of `Δt`, and the CPML ledger converges at
  second order (relative defect 8.5e-3, 2.0e-3, 4.9e-4 as `Δt` halves).
  `PreparedMaxwellCPML.bind_coefficients(step_size)` binds the half-step
  recursion of one leapfrog step, and `apply_magnetic_start`/`apply_magnetic_end`
  replace `apply_magnetic` (`PreparedReducedMaxwellCPML` likewise replaces
  `apply` with `apply_electric`/`apply_magnetic_start`/`apply_magnetic_end`).
  The reduced CPML now takes its profile from the shared calibration
  `σ_max = (m + 1) c ln(1/R) / (2d)` (it ignored wave speed and spacing and
  was twice as strong) and grades electric memories at nodes rather than cell
  centers, which lowers its reflection from 1.4e-2 to 3.6e-4 for a 15-cell
  layer; `PreparedReducedMaxwellCPML` takes the grid spacing and wave speed.
- Complex `phydrax.special.kv`/`kve` were a pure power-series connection
  formula (relative error ~1e-9 at `|z| = 0.5`, 0.7 at `|z| = 10`, 1e8 at
  `|z| = 20`; 1e6 at `|z| = 40` on the imaginary axis). For `Re z >= 0`
  (including the imaginary axis), `|Re v| <= 128.5`, and `|Im v| <= 1`, they
  now use Temme's series (`|z| <= 2`) or Temme's CF2 continued fraction in
  Steed form, then masked upward order recurrence, all with fixed trip counts;
  `kve` is computed directly in scaled form instead of `exp(z) * kv`, and the
  argument JVP uses `K_v' = (v/z) K_v - K_{v+1}`. Relative error against
  mpmath is below 2e-15 over `|z| in [1e-3, 200]`, `|arg z| <= pi/2`. The real
  `kv`/`kve` order derivatives, which route through this continuation, inherit
  the accuracy. Left-half-plane arguments keep the previous series.
- Strict dtype promotion failures: native restarted GMRES and the Arnoldi
  orthogonality check compared `int64` index ranges with `int32` step counts
  (reached by dense `VectorLocalRootPlan` tangent solves above dimension three
  and by `PolarizedRadiativeTransferPlan` matrix-exponential actions).
- `ParticlePopulationPlan.allocate` sorted invalid requests with an `int64`
  sentinel cast into the request's `int32` event-ID dtype (wrapping to `-1`,
  so padding rows came first), and unused request rows scattered back into
  slot 0, erasing an allocation made there; event IDs now widen before
  sorting and only allocated rows write.
- Lorentz–Drude ADE damping was applied twice per half step (effective `2γ`);
  the oscillator half step is now a symmetric kick-drift-kick with
  Crank–Nicolson damping and matches the continuum `ε(ω)` to second order.
- `tools/pic_qualification.py --smoke` crashed on the missing
  `ElectrostaticPICDiagnostics.continuity_defect`; the diagnostics now report the
  integrated continuity residual `|Σ ⋆0(ρ_{n+1} − ρ_n)|/Δt`.
- Tetrahedral electromagnetic PIC passed Whitney edge flow directly as Maxwell
  current; charge and current now enter Maxwell through the inverse Hodge stars
  with the physical sign, so Maxwell's Gauss charge follows the deposited
  charge. The reduced 1-D Maxwell Gauss/charge divergence gives the nonperiodic
  lower wall zero flux, matching the PIC continuity operator; reduced 2-D fields
  with a nonperiodic axis (not paired) are refused by the pairing check.
- Corrected structured VOF capillarity, pressure accumulation, density
  averaging, PLIC offset residuals, height-function fallback support, and
  continuation dtypes; the force now produces the intended pressure jump and
  restoring dynamics without post-transport volume-fraction clipping.
- Native PCG and ProjectedPCG convergence now requires the true residual;
  recurrence-residual hits trigger confirmation and residual replacement rather
  than accepting a drifted Krylov residual.
- Film Newton convergence no longer stalls on large drainage steps because its
  inner forcing matches the declared linear tolerance. Sparse ILU/IC dropping
  also preserves the x64 loop-carry dtype and avoids full-pattern marker
  allocation per pivot.
- Exact finite topology predicates no longer leave already separated triangle
  pairs unresolved; zero-energy biomembrane remesh candidates use a finite
  execution-dtype scale instead of producing `0/0` evidence.
- Langmuir surfactant updates and inflows now reject capacity crossings,
  nonpositive tension, and nonfinite states atomically; constant-tension
  vertex forces also cancel exactly in the momentum ledger.
- Bubble-cloud FMM overlap candidates now cover the complete declared contact
  support, while resolved bubbly-flow and foam dynamics roll failed retries,
  pressure solves, CCD steps, and event passes back without committing
  compartment, ledger, topology, or lineage changes. Active multiregion face
  normals are unit normals, and event lineage records parentless created faces
  explicitly.
- Batched native PCG confirms nominated true residuals with one shared action
  stream, and compartment pressure lanes use one multi-right-hand-side solve
  while preserving scalar projection and derivative semantics.
- Vortex-sheet curvature preparation now stores a bounded edge relation instead
  of a vertex-by-region materialization, with exact retained-byte evidence and
  fail-closed relation-capacity admission.
- Capacitated-assignment canonicalization rejects out-of-range integer slots
  before int32 narrowing, and audit residuals use overflow-safe saturation so
  large deficits, including exactly `2**32`, cannot wrap to feasible zero.
- Sparse LU/Cholesky symbolic fill enforces incremental factor-nnz,
  retained-byte, and symbolic-work ceilings before dictionaries grow; factor
  builders honor active materialization ceilings with observed/limit evidence,
  and sparse solve status gives `NONFINITE` precedence over `ZERO_PIVOT`.
- Incompressible foam dynamics and foam equilibrium now prepare a bounded
  topology-first volume-constraint basis with explicit rank-action/byte
  evidence and resource refusal. Compartment-gas dynamics records a canonical
  not-applicable zero-work basis status and retains only its matrix-free volume
  pressure operator. Open boundary-boundary film interiors use the same
  tangential quotient as closed flat films, preserving the catenoid equilibrium
  route under current JAX.
- `phydrax.meshing.evaluate_cell_quality` of hexahedra, prisms, and pyramids
  can be first evaluated under `jax.jit`/`eqx.filter_jit`: the cached exact
  measure quadrature holds host NumPy nodal gradients instead of converting a
  JAX tabulation, which raised `TracerArrayConversionError` inside a trace.
- Simplex Lagrange elements of order 3 and higher (`lagrange_element` on
  triangles and tetrahedra, `SimplexNodalFamily.finite_element`) order edge DOFs
  along their reference edge direction, so `FiniteElementDofMap` routes every
  shared-edge DOF to one physical node on arbitrarily oriented meshes;
  previously edges such as the triangle's `(2, 0)` edge were listed in reverse.
- `PreparedHelmholtzMultipole3D.p2l` pairs the outgoing Hankel radial factor with
  `conj(Y)` as its documented local convention states; it previously also
  conjugated the Hankel factor.
- Derivative admission refuses every request on a `STOPPED` route, including
  hard-tree fits and meets of differing routes; differentiated regularity keeps
  the conditions it was admitted under.
- Solver-objective gradients, ML influence functions, particle Fisher and
  genealogical scores, and neural implicit design states differentiate only the
  declared parameter lane; influence solves use native linear algebra and report
  rank, conditioning, and per-sample status. Linear-Gaussian priors, transition
  parameterizations, and observation models declare their roles.
- Training preflight rejects functions that read array-valued module globals
  and hidden arrays beneath plain fixed state; only explicit freezes authorize
  hidden artifact state, and such functions no longer receive a stateless
  identity. Trainable provider bindings require a parameter leaf; lane layouts
  are canonically ordered.
- Training checkpoints bind the digest of every lane and cursor; objective
  callables may hold only fixed arrays; zero-support objectives commit no
  model-state transition.
- Relative component error floors require a declared residual scale before a
  nonlinear tolerance can be certified. Batched QP sensitivity reports
  regularity per case, and MPC sensitivity refusals name the failing case and
  window.
- Models that declare ports must be bound with owner ports: feedback policies,
  observation locations, step corrections, mesh proposers, rollout transitions,
  and other learned slots derive them from their layouts. Model execution
  contracts can record declared capabilities, which satisfy only
  declaration-level requirements.
- Discrete field transposes and duality evidence refuse invalid query routes;
  external operator adapters derive their binding from the manifest; blockwise
  dependency subsets are validated against declared dependencies.
- The public API manifest discovers lazily exported submodules by module spec,
  so it no longer depends on import history; it now lists 22 previously omitted
  public modules.
- Neural implicit geometry reports sampled sign and topology evidence only and
  no longer claims certified topology or reliable sign; its design state holds
  parameter leaves only.
- The environment now matches the declared `equinox==0.13.8` pin. Declared
  `eqx.AbstractVar` fields are enforced as abstract, strict modules no longer
  carry an undeclared initialization flag in their pytree state, and domain
  geometries no longer declare phantom `adf` fields; `error_if` failures now
  surface as `EquinoxRuntimeError` in eager and compiled calls.
- JAX dtype promotion that the style refactor changed from weak `float` to
  strong `float64` again preserves single-precision and complex-single inputs.
- Frozen-dict leaves are addressed by mapping key in pytree paths; empty
  frozen dicts keep their layout through structural updates.
- Corrected latent defects across finite-volume, finite-element, IGA, DEM,
  compatible systems, Bayesian quadrature, circuit small-signal convention,
  dense classification, tribology saturation, native LSMR stopping, variational
  inequality and higher-order root linear budgets, filter IPM budgets,
  optimistix integration, ArviZ export, property verification tolerances,
  multigrid coarse precision, chaos RQA validity, unitary propagation tangents,
  graph value enforcement, operator benchmark primary sources, broken pairwise
  iteration, and the `imageio.v3` import; stale tests and benchmark evidence
  were updated to the current contracts.
- Discrete-velocity pull offsets are integers, and the learned-energy and IREE
  export evidence is regenerated against current artifacts.
- The training-boundary check for hidden arrays inspects dataclass fields and
  declared slots.
- Composite derivative rules no longer hold copies of their operand fields.
  Arithmetic expressions, transposes, gated and weighted boundary blends,
  interior anchor corrections, ragged time-series ansätze and corrections,
  trajectory signals, discrete field views, fiber projections, and frozen
  correction fields now derive their rule on demand from their evaluator's own
  operands (`DomainFunction.derivative_rule` returns the explicit rule, else the
  evaluator-derived one; the stored rule is `explicit_derivative_rule`).
  Derivatives therefore use the current parameters after a training update
  instead of stale construction-time copies, each parameter is a single visible
  leaf, and such fields pass `require_parameter_roles`.
- `eqx.AbstractVar[...]` / `eqx.AbstractClassVar[...]` declarations on strict
  modules are now actually abstract. Under `from __future__ import annotations`
  Equinox silently treated them as concrete dataclass fields, so abstract
  attributes were never enforced and every subclass implementing one with a
  property carried a phantom field that broke flatten/unflatten round trips
  (filter specs, parameter subspaces). Four concrete classes that never
  implemented a declared attribute now do: `IntegrationAxisSpec.n`,
  `FeasibleParameterization.scope`, `PreparedGeneralForceFieldTerm.force_group`,
  and `ChemicalJumpProcess.process_id`.
- Strict modules no longer store a freeze flag in their instance dictionary.
  Equinox flattened it as wrapper metadata, so every unflattened copy gained
  `__name__`/`__qualname__` set to a sentinel and `eqx.filter_jit` of
  `eqx.filter_vmap(module)` failed with "__name__ must be set to a string
  object". Deleting a strict module attribute still raises.
- Mapped finite-volume dynamics and wave-propagation plans now refuse a face
  closure at preparation instead of silently ignoring it.
- DEM and reactive checkpointed replay VJPs invalidate the returned cotangent
  when the replay does not match; the primal and replay evidence are kept.
- Implicit root differentiation checks the nonlinear method's capability even
  when explicit tangent and adjoint policies are supplied.
- Hardened the quantum Hall, compressible kinetic, and scaled-Taylor additions:
  scientific owner identities and charge rosters now fail closed, transport and
  SCBA retain native solve status, finite-support means require convex-support
  evidence, kinetic remap/AMR/geometry operations expose truthful transactions,
  and exact-norm Taylor actions enforce their forward truncation bound.
- Geometry domains now preserve exact, estimated, and unknown mass evidence,
  reject scalar-to-diagonal coercion outside one dimension, and distinguish
  interior from boundary-measure capabilities. Von Mises stress now requires
  an explicit convention outside its 2D/3D physical envelope.
- Corrected rectangular rank-deficient pseudoinverse derivatives; certified
  mathematical linear and nonlinear solution-map derivatives against primal
  and tangent residuals; made regular QP sensitivities bidirectional and
  fail-closed at weak or singular active sets; made derivative planning
  preserve mixed variables and reserve Jet for explicit requests; added a
  native Pallas pair-acceleration JVP, explicit ML gradient admission, and
  shared invalid-derivative guards for finite diagnostic fallbacks.
- Preserved tiny Gegenbauer parameters through the first recurrence step by
  avoiding cancellation in the `2*alpha` coefficient. The isolated `z=1`
  polylogarithm branch was removed from its differentiated contract; callers
  use `zeta` for that value identity instead of receiving a false finite
  argument derivative or zero higher derivative. Polylogarithm primals and
  custom JVPs now compute only required quantities and select bounded
  streaming or term-axis series routes from static shape.
- Benchmark runtime fingerprints now record `NPROC`, the worker-count input
  consumed by XLA's CPU thread-pool sizing.

### Added
- Added `phydrax.typing`, a closed structural contract language shared by static
  checkers and runtime boundaries. Nominal `Dim`/`VariadicDim` size variables,
  `AnyDim`, `AnyShape`, `Scalar`, and `Broadcast[D]` build JAX tensor forms
  (`Float64[ComponentDim]`, statically `jax.Array`) and host forms
  (`HostFloat64[ComponentDim]`, statically a dtype-carrying NumPy array);
  `Size[D]`, `Identifier`, `Identifiers[D]`, `PRNGKey` (typed keys only),
  `Literal` selectors, `Enum`s, optionals, unions, and fixed tuples complete the
  grammar. `Scope` binds dimensions with provenance and union rollback; `parse`
  validates and canonicalizes selector literals; `as_array` and `as_host_array`
  convert once under an explicit casting policy; `validate` checks every
  contract field of a module. Checks read only kind, rank, extent, and dtype
  metadata: they add no JAX operations and synchronize nothing. See the typing
  guide.
- The Phydrax wheel ships the PEP 561 `py.typed` marker, so type checkers use its
  inline annotations. `python -m tools.check_installed_typing` builds the wheel,
  installs it for Python 3.12 and checks consumer fixtures against the pinned ty.
- `tools/audit_host_sync.py` reports `bool`/`int`/`float`/`numpy.asarray`/
  `numpy.array`/`jax.device_get` calls in package code, classified as static-shape
  reads, host preparation, explicit safe points, external-provider code, output
  observation, bodies of JAX-transformed functions (resolved by lexical scope),
  or unclassified.
- `SamplingMPCRealizations` and `SamplingMPCPlan` opt into structural contracts:
  their weights, support masks, and model counts share one nominal model axis.
  `tools/audit_contract_candidates.py` ranks strict modules whose static metadata
  a contract could state.
- Private validation helpers that were behaviorally identical copies of shared
  host validators (canonical and normalized identifiers, finite positive floats,
  positive and nonnegative integers) were replaced by the shared owners in
  `phydrax._validation`; accepted values, exception categories, and messages are
  unchanged.
- Closed selectors across the package are declared once as `TypeAlias`
  `Literal` aliases and validated with `phydrax.typing.parse`: duplicated option
  tables and inline membership checks were removed, equality chains over a whole
  alias became exhaustive `match` statements, and constructors store the
  canonical literal (for example a `numpy.str_` selector is stored as `str`).
  Unsupported selector values still raise `ValueError`; non-string selector
  values now raise `TypeError`, and selector error messages name the parameter.
  `tools/audit_selectors.py` reports remaining duplication.
- Domain geometry annotations use `phydrax.typing` forms with nominal point and
  spatial dimensions; `GeometryTransitionResult` opts into structural contracts.
  jaxtyping dtype and shape forms are refused by lint in package code.
- PRNG key parameters and fields are annotated with `phydrax.typing.PRNGKey`, the
  typed scalar key; seeded internal draws in Lyapunov, covariant-vector, and chaos
  analysis, Riemannian density flows, multistart optimization, process
  tomography, and stochastic immersed forcing create typed keys with
  `jax.random.key`, producing the same random streams. Modules that store a key
  (training kernel state and keys, stochastic, differential-evolution, control,
  design, and MAP search results, evolution-strategy payloads and rules, adaptive
  integrands, and functional-decomposition problems) opt into structural
  contracts, so their stored key must be one typed scalar key; legacy
  `uint32[2]` keys raise `TypeError`. `jaxtyping.Key` and
  `jaxtyping.PRNGKeyArray` are refused by lint in package code.
- Package code imports `Array` from `jax` and `ArrayLike` from `jax.typing`, their
  canonical owners; importing them from `jaxtyping` is refused by lint.
- Strict modules that declare `__strict_contract__ = True` (inherited by
  subclasses) check their `phydrax.typing` field contracts, read-only, after
  Equinox construction and `__check_init__`; `phydrax.typing.validate` checks an
  opted-in module and the opted-in modules it contains. Model array-recipe
  restores and operator artifact models are validated once after complete
  reconstruction, and fitted ML schema binding uses `equinox.tree_at` instead of
  raw object reconstruction. `ChemicalComponentCatalog` opts in.
- `ChemicalComponentCatalog` declares its stored fields with `phydrax.typing`
  forms and checks name counts and array extents through one binding scope; its
  accepted inputs, identity, and exception categories are unchanged.
- Battery OED criteria, generic experiment-design criteria, and axis
  discretization bases and primary entities are parsed against their `Literal`
  aliases, removing duplicated option tuples; non-string selector values now
  raise `TypeError`, unsupported selector strings still raise `ValueError`.
- Added a common file-resource substrate with descriptor-safe resident and
  seekable admission, bounded resource sets and external archives,
  crash-consistent file and bundle publication, strict document/NumPy/HDF5
  preflight, deterministic format capability introspection, lifecycle-backed
  finite-element persistence, canonical material-point and replay archives,
  and staged validation for mesh, surface, imaging, deployment, and domain
  interchange routes.
- Added dimension-generic balls, orthotopes, straight extrusions,
  arbitrary-ambient low-dimensional simplices, codimension-one boundary frames,
  bounded ND cochains, cell-list neighborhoods, interpolation, wavelet/Fourier
  resource policies, and explicit planar curl operators.
- Added planar modified-Helmholtz, Stokes, elasticity, and periodic spectral
  boundary kernels; topology-generic compact U(1); native PIC qualification;
  bounded spatial DFN lanes; ND effective-mass/Poisson confinement; planar
  implicit-curve discovery; and oriented 3D crack-surface/front quadrature.
- Added 21 exact fixed-topology omniphysics candidate tuples spanning spatial
  materials and manufacturing, conservative population and interface transport,
  coupled electrohydrodynamics, smart and chemo-mechanics, EHL and thermal
  systems, membranes, catalysis, optomechanics, acoustics, electrochemistry, and
  equation-oriented processes. Added full engineering application workflows,
  pinned source and license records, independent controls, three refinement
  campaigns, analytic application validations, and exact runtime-provider
  evidence. Implementation closure is separate from release: retained evidence
  covers one float64 `jax-cpu-arm64` host and explicitly does not qualify
  distributed execution, experimental validity, legal approval, or release.
- Replaced the disconnected Metrix Gaussian/RDP sketches and private
  Riemannian optimizer with a first-class research `phydrax.privacy` control
  plane. Privacy units, adjacency and trust assumptions, Google DP Accounting
  event composition, provider-coupled JAX Privacy DP-SGD, restricted exact
  resume, public-safe certificates, operator-artifact binding, release-root
  budget ledgers, qualification nonclaims, focused tests, and an end-to-end
  benchmark now share one fail-closed contract. The current JAX PRNG profile is
  explicitly not public-release authorized.
- Added a native prepared Pfaffian lifecycle with signed-log, skew,
  singularity, resource, refresh, and differentiation evidence; factor-owned
  LU determinant evaluation; fixed-capacity exact determinant/Pfaffian
  low-rank update sequences; paired-fermion Pfaffian-Jastrow amplitudes; and
  exact local targets for Pfaffian-Jastrow and periodic determinant VMC.
  Markov targets now distinguish same-target cache refresh from intentional
  numeric rebinding, allowing parameter updates and checkpoint restoration to
  reconstruct caches without stale state. Supersymmetric lattice Pfaffian
  evidence now consumes the native factorization.
- Added an optional application-owned W&B training sink with bounded typed scalar
  and lifecycle delivery, coordinator-safe iteration sessions, and fail-open
  provider errors. Functional gradient, Evosax, and KFAC runs now expose the same
  checkpoint-aware session boundary without automatic authentication, network
  setup, or artifact upload.
- Completed semantic axis-key alignment and unified axis contraction ownership;
  promoted local implicit roots into the nonlinear substrate; added request-driven
  dense matrix property verification, canonical validation and PyTree algebra,
  transactional commit and balance-ledger carriers, qualification runtime identity,
  and prepared affine linear interval evolution.
- Closed the bounded computational-frontier program across matrix-irrep quantum
  sectors, mixed/spinning and certified conformal bootstrap, Pfaffian and
  fermionic-BFSS supersymmetric workflows, harmonic/global Calabi–Yau geometry,
  nonlinear spherical Einstein–scalar AdS, many-body and matrix fuzzy spaces,
  native SM/MSSM spectra with SLHA2 semantics, general defects and multimode
  envelopes, and native nonzero-spin/finite-complex spin foams. Added shared
  claim/resource/archive contracts, validated decimal intervals, ordered limit
  studies, exact regression controls, qualification evidence, benchmarks, and
  explicit permanent scientific nonclaims. Defect potentials and projective
  variety families reuse the canonical sparse-polynomial system introduced
  below rather than defining a second support representation.
- Added canonical sparse polynomial systems with exact multigrading and scaling-
  symmetry analysis, isolated-root and positive-dimensional provider evidence,
  fixed-mode AC power-flow enumeration, polynomial-image implicitization,
  restricted exact symbolic operations, moment/SOS compilation, quotient-
  algebra roots, equivariant polynomial bases, symmetric tensor decomposition,
  and prepared G1 multi-patch spline constraints. Optional Julia and Macaulay2
  execution remains explicitly pinned, bounded, host-only, and fail-closed.
- Added a research-only medical-radiation spine: shared radiation quantities and
  units, multi-reference/uncertainty-bearing medical images, strict read-only
  DICOM CT/NM/PET/RT profiles, source-pinned diagnostic-photon material data,
  HU calibration and deterministic material-basis spectral CT, artifact-only
  external radiation score profiles, native time-activity/S-value internal
  dosimetry, and exact deterministic/stochastic circulating-blood dose. All
  capability profiles remain unreleased pending their named scientific gates.
- Added the full relativistic dark-sector profile lattice: explicit unit/tetrad
  contracts, stress-energy particle transfer, scalar/vector/tensor weak-field PM,
  Einstein--Vlasov Z4c coupling, durable semantically unbounded event epochs,
  immutable runtime matrix-element adaptation, model-specific dark showers,
  hadronization, bound states and decay cascades, Bose/Pauli quantum kinetics,
  rights-qualified thermal/HTL/LPM rates, coherent density-matrix and off-shell
  Kadanoff--Baym/Wigner transport, packet/M1/VET/hierarchy dark radiation, and one
  fully coupled stress-energy runtime with distributed checkpoint, observables,
  inference, claim and source-rights evidence.
- Closed the production polymer-physics surface with immutable primitive-path
  snapshots, native and Z1+ entanglement routes, uncertainty-bearing estimators,
  particle/tube/slip-spring/GLaMM reptation, matrix-operator RPY and confined FIB
  hydrodynamics, lubrication and complete stress contributions, evolving
  Lees–Edwards and Kraynik–Reinelt cells, peculiar-momentum SLLOD, driven-work
  and LAOS analysis, exact support admission, composite replay checkpoints,
  qualification gates, bounded smokes, and performance benchmarks.
- Added a conservative general-relativistic radiation-MHD suite: standalone
  boundary-aware gray and multigroup M1 transport, implicit four-force GRRMHD,
  dynamic Z4c coupling, neutrino lepton exchange, resistive and force-free
  transitions, physical opacity/photon-number and electron/pair plasma evolution,
  VET/discrete-ordinates/Monte-Carlo closures, polarized feedback, ingoing-Kerr
  torus/fast-light products, AMR/distribution/restart/production integration, and
  qualification tests, examples, and benchmark evidence.
- Added native reacting-flow closure: governed transport properties, constrained
  equilibrium and jumps, CEMA, conservative spatial low-Mach SDC/projection,
  synchronized AMR chemistry, ALE/source/scheduling ledgers, learned
  stoichiometric transitions with exact fallback, campaigns, smoke, and benchmark.
- Added native radiation-transport closure: governed cross sections and spectra,
  voxel photon KERMA with event export, diagnostic detector workflow, slab
  multigroup discrete ordinates, charged condensed histories, hybrid IMC/DDMC,
  correlated-k/polarized experiments, campaigns, smoke, and benchmark.
- Added scale-explicit superconductivity ownership and candidates for planar
  London/Pearl, local-U(1) GL/TDGL, equilibrium Riccati quasiclassics, retarded
  spectroscopy, cable current-sharing/quench/protection, and evidence-only
  fidelity bridges, while preserving existing BdG support coordinates.
- Added bounded finite-group character sectors for canonical quantum lattices,
  including explicit monomial site/local actions, fermionic permutation signs,
  complete group closure, normalized orbit embeddings, exact invariance and
  Hermiticity audits, fixed reduced routes, archive support, qualification
  evidence, and construction/compilation/steady-action benchmarks.
- Added typed four-scalar conformal data and crossing-basis gauges,
  general-dimensional finite global-block recursion with truncation and Casimir
  evidence, exact-decimal polynomial matrix programs, finite sampled PSD
  audits, and a bounded pinned SDPB preprocessing/solver path with exact summary
  parsing, functional reconstruction, independent audit, qualification, and
  benchmark evidence.
- Added a regulated two-dimensional twisted N=(2,2) SYM production path:
  invertible real coordinates for independent complex links, a minimal
  pseudofermion Dirac protocol, antisymmetric geometric Kähler–Dirac action,
  exact `M†M + mu² I` regulator construction, bounded structural spectral
  interval, generated quarter-power rational pseudofermions, nested RHMC,
  tiny-volume Pfaffian/Ward/phase-overlap evidence, qualification, and
  preparation/trajectory benchmarks.
- Added independent Calabi–Yau metric qualification with disjoint sample
  ancestry, weighted held-out residual distributions, ESS, batch uncertainty,
  positivity/chart/pivot evidence, optional explicit Ricci audits, and evidence-
  bound frozen artifacts. Added transverse fixed-support complex-structure
  families, provenance-gated sampled Weil–Petersson/Yukawa integrals, general
  matrix-valued Chern-character forms, sampled characteristic-number evidence,
  qualification, and evidence-cost benchmarks.
- Added a source-clean conformal Einstein–AdS reference surface: the complete
  four-dimensional vacuum metric-conformal zero-quantity ledger, analytic
  constant-curvature AdS control, explicit generalized-wave/conformal gauge,
  timelike boundary and corner audits, a fixed-background reflecting conformal
  scalar runtime with normal-mode/energy evidence, conformally coupled scalar
  stress, provenance-bound scalar and holographic stress observables,
  qualification, source ledger, and benchmarks. This is not nonlinear
  dynamical AdS gravity or automatic holographic renormalization.
- Added exact finite SU(2) coupled-sector compilation with Condon–Shortley
  transforms, projector and operator-invariance evidence, plus a bounded
  two-particle lowest-Landau-level fuzzy-sphere application with explicit
  exchange statistics, complete pair-spin pseudopotentials, rotational
  evidence, labeled spectra, qualification, docs, and flux-scaling benchmarks.
- Added bounded two-dimensional Virasoro references: caller-derived
  second-order BPZ hypergeometric blocks with explicit branch/source and
  series-tail/nome evidence, plus exact `c=1/2` Ising four-spin identity and
  energy blocks with channel-summed crossing qualification. These APIs do not
  claim generic Virasoro recursion or a continuum 2D bootstrap.
- Added particle-spectrum semantics with explicit perturbative profiles,
  independent numerical/physical/provider/warning statuses, unit-bearing
  observables and running trajectories; bounded strict SLHA parsing and
  preservation; pinned external calculator execution; and a native log-scale
  RK4/Newton boundary-value workflow with analytic qualification and resolution
  benchmarks. No model-specific supersymmetric loop corrections are inferred.
- Added a bounded stationary double-well kink solve with analytic profile,
  energy, sector and stability evidence, plus scalar-envelope GNLSE response
  and propagation with explicit beta/loss, Kerr, optional causal Raman and
  self-steepening, fixed interaction-picture RK4, adaptive step doubling,
  spectral/refinement/work evidence, qualification, docs, and benchmarks.
- Added explicitly research-only quantum-geometry controls: expanded finite
  SU(2)/BF identities, fixed-graph Gauss-invariant spin-network bases and area
  values, complete finite-cutoff Lorentzian EPRL semantic admission, a pinned
  external process protocol, and the native analytic zero-spin SL(2,C) B4
  booster with quadrature/tail evidence, qualification, docs, and benchmarks.
- Closed bounded dark-matter production profiles with signed scientific claims,
  typed output/restart contracts, correlated mixed-component initial conditions,
  shared and distributed wave/particle/gas gravity, periodic finite-difference and
  pure-complex distributed AMR wave evolution, contact and isolated-wave profiles,
  differential/anisotropic, unequal-weight, frequent and spherical gravothermal SIDM,
  reversible multistate reactions with dark-radiation accounting, and native
  observables/inference. Every profile carries explicit support, capacity,
  conservation, rollback, provenance, distribution and differentiation evidence;
  automatic regime conversion, relativistic reactions, resolved 2-to-n radiation,
  and wave/Hamilton--Jacobi representation switching remain fail-closed nonclaims.
- Added an end-to-end polymer-physics stack: energy-shifted WCA and generic FENE
  atomistics, Kremer–Grest qualification, polymer conformation/scattering and
  equilibrium-rheology evidence, constant-mobility Brownian and discrete-FDT GLE
  runtimes, live chromatin–atomistic coupling, native isotropic PRISM with four
  closures and continuation, periodic linear/branched SCFT with implicit
  derivatives/cell/symmetry continuation, bounded partial-saddle and
  complex-Langevin FTS, deterministic material recipes and stable-ID topology
  lowering, admitted construction adapters, nonperiodic and explicit-winding
  periodic reaction epochs, network observables, cross-representation theory
  vectors, candidate qualification campaigns, smokes, benchmarks, and guides.
- Added a coupled phase-field multiphysics closure with a single ownership
  graph and total energy/entropy/conservation ledger; thermodynamically
  consistent nonisothermal grand-potential phases; calibrated anti-trapping
  transport; prefix-stable transactional nucleation; coherent small- and
  finite-strain mechanics; power-canceling Model-H flow; fixed-charge and
  fixed-voltage dielectric coupling; electrochemical flux, Maxwell stress, and
  Joule heat; power-adjoint transfers; coupled fixed-step/checkpoint identities;
  exact profiles; PFHub-style qualification; and compiled flagship performance
  evidence. Duplicate storage, incomplete exchange ownership, failed event
  inventory, Gauss-law defects, incompressibility defects, or global ledger
  failure reject atomically.
- Closed the extended phase-field production surface with registered
  discrete-gradient potentials, complete energy/work/source ledgers,
  heterogeneous FE blocks, wetting and imposed boundary fluxes, periodic H1
  constraints, anisotropic Onsager mobility, dense grand-potential evolution,
  fixed-capacity active phase IDs, accepted hp/AMR epochs, replayable spatial
  Wiener forcing, distributed ownership and checkpoint identities, exact
  candidate support profiles, integrated qualification, and compiled
  performance evidence. Unsupported potential laws, capacity overflow,
  inconsistent periodicity, failed transfers, and invalid physical ledgers now
  reject explicitly without state repair.
- Replaced the aspirational profile-based ROM facade and truth-backed online
  evaluation with content-bound physical basis artifacts, shared case partitions,
  explicit trial/test reductions, reduced-only affine online assembly, native
  fidelity and archive integration, physical-norm audit, scoped coercive
  certification, restricted polynomial operator inference, full-residual
  Galerkin/LSPG references, and distinct DEIM, GNAT, and ECSW hyperreduction
  contracts. Gravitational-wave ROQ now consumes a role-explicit basis directly,
  and cardiovascular truth cohorts retain their canonical operator targets rather
  than a parallel ROM truth payload.
- Closed the ROM production-and-beyond capability graph with governed maturity
  declarations, resource/admission/cost contracts, portable deployment bundles,
  physical vector-space POD and snapshot manifests, rectangular and transient
  projection, trace-qualified lifts, SCM/greedy/primal-dual evidence, selected
  residual plans and thin GNAT, fixed-reference geometry atlases, quadratic and
  coordinate-conditioned charts, sensor-history estimation and assimilation,
  spectral-submanifold identification, balanced and rational control reduction,
  symplectic and port-Hamiltonian projection, and immutable active-learning,
  enrichment, distributed-basis, and out-of-core correlation artifacts.
- Added conservative phase-change physics: analytic solid/liquid enthalpy
  inversion with implicit MAC mushy resistance, binary-alloy enthalpy/solute
  coupling, bounded Antoine saturation curves, homogeneous-equilibrium
  barotropic cavitation, pressure- and heat-driven two-material VOF transfer,
  conservative thermal diffusion, moved-stage PLIC reconstruction, and
  phase-aware overset fluxes with explicit conservation, admissibility,
  failure, and derivative-event evidence.
- Added a fail-closed micro/nanoflow capability stack: shared model-admission
  evidence; cell-local NTC DSMC with distinct VHS/VSS scattering, accepted-pair
  chemistry, physical walls/reservoirs, moments, and conservative continuum
  exchange; first-order continuum slip/jump/thermal-creep walls; one-way
  finite-radius particle transport; MAC-native PNP and resolved/thin-EDL
  electroosmosis; hydraulic DAE components; accepted-step atomistic nanoflow
  observers and immutable closure artifacts; and an end-to-end candidate DLD
  workflow with exact circular-post geometry, LBM flow admission, outlet metrics,
  empirical screening, and robustness evidence.
- Added the condensed-matter production-evidence layer without an umbrella physics
  capability: concrete immutable array archives now retain canonical periodic
  family/pencil/spectrum/IFC/DMFT, direct quantum-sector and TPQ/response, and
  semiconductor detector results against matching caller-prepared structure and
  exact source/profile/unit provenance. Added maturity-neutral owner/application-leaf
  profiles and disjoint campaigns for every implemented periodic, lattice/phonon,
  Green/embedding, quantum, spectroscopy, magnetic-resonance, semiconductor,
  soft-matter, and bounded frontier slice; an exact derived baseline ledger plus
  a separate frontier inventory; a public end-to-end smoke example; and a
  failure-preserving qualification/benchmark orchestrator. No release
  index, authority key, signed evidence, or capacity claim is shipped.
- Replaced compile-per-step phase-field helpers with prepared convex-split
  Allen–Cahn and mixed Cahn–Hilliard finite-element methods, one canonical
  binary free-energy model, interface-resolution admission, physical
  energy/dissipation and cumulative-mass gates, exact failed-step rollback,
  fixed-step production/checkpoint composition, scientific qualification, and
  compiled performance evidence. The qualified profile is closed, float64,
  single-device P1 triangles on one fixed homogeneous cell block; wetting,
  imposed boundary work, periodic constraints, AMR, multiphase thermodynamics,
  and distributed execution remain explicit nonclaims.
- Replaced the array-only replica and free-energy surface with phase-space-bound
  thermodynamic state tables, closed atomistic operator schedules, fixed-capacity
  kernel-qualified replica-exchange and SAMS segments, atomic continuation
  checkpoints, canonical controlled Hamiltonians, authenticated dense and sparse
  reduced-potential/work/derivative datasets, covariance-qualified FEP/BAR/TI/MBAR,
  pairwise sparse and full network inference, native switching execution, and explicit
  neutral solvation, binding, separated-topology, and mapped-relative
  protocol contrasts. The former endpoint-only alchemy, scalar alchemical term,
  anonymous work/energy estimators, and replica-state APIs were removed.
- Added bounded periodic wave-dark-matter Schrödinger--Poisson evolution,
  cosmologically normalized rare elastic SIDM, path-batched jump/guard composition,
  manifest-qualified terrestrial and solar dark-matter transport, continuum-plus-line
  indirect yields, species-resolved exotic energy deposition, stable halo lineage,
  rights-checked cosmology interchange, and exact-grid external matter-power products
  with explicit support, conservation, failure, differentiation, and provenance
  evidence.
- Added fixed-shape scalar Lanczos/Jacobi and matrix-valued block-Jacobi
  continued-fraction evaluation over complex shift families. Scalar forms retain
  explicit terminal-resolvent closure, lane-local truncation evidence, bound
  projection provenance, and shift/tail differentiation. Matrix fractions
  preserve explicit noncommutative upper/lower coupling order, accept terminal
  self-energies, isolate singular/nonfinite shifts, and report per-level inverse
  evidence.
- Added a native evidence-bounded laser-domain stack: coupled four-dimensional
  paraxial resonator closure and Gaussian-mode lowering, passive geometric-ray
  volume attenuation with per-medium/surface deposition ledgers, explicit
  finite/periodic pulse-time spaces and slowly varying electric-field envelopes,
  exact envelope/carrier-field bridging, direct different-grid Fresnel
  propagation, a bounded HDF5 profile for the pinned upcoming openPMD
  LaserEnvelope draft, passive bidirectional coupled-mode scattering, frozen
  semiconductor gain/index response, reduced deterministic and replayable
  stochastic traveling-wave laser dynamics with Fabry-Perot and distributed-
  grating threshold modes, certified finite-radius Bessel-zero Hankel
  transforms, scalar axisymmetric carrier-resolved propagation, and causal
  Raman, multiphoton-ionization, and Drude-current responses. Full transient
  drift-diffusion/laser coupling, nonzero cylindrical azimuthal order,
  magnetized long-pulse plasma, and laser-processing CFD remain explicit
  nonclaims.
- Added provider-complete HEP production profiles with bounded particle-event
  truth and signed-weight accounting, LHEF/HepMC/ROOT-profile interchange,
  native two-body hard-event generation, collision pileup composition, typed
  detector transport/hit/digit/tracking records, cell-explicit calorimeter
  response and sparse flow-matching fast simulation, collider analysis and
  likelihood primitives, accelerator beamline/collective contracts, and
  finite-density B/Q/S lattice, canonical, reweighting, HRG, critical-provider,
  and qualified equation-of-state tables. Native and external capabilities,
  derivative validity, overflow, support, provenance, and scientific nonclaims
  remain explicit.
- Extended HEP beyond the reference production slice with authoritative ragged
  host events and bounded packing evidence, operational conditions/exposure,
  process normalization and systematic-source semantics, shared governed
  binned/unbinned statistics, Awkward-ready columnar and framework contexts,
  native reference/fuzzy jets, calibration/vertex/particle-flow reconstruction,
  offline event-building/trigger/buffer replay, collider-theory/EFT prediction,
  ring optics/tracking and wakes, QCD transport and heavy-ion evidence,
  neutrino oscillation/rate workflows, coherent flavor mixing, clean-room
  trapped-particle phase-transition bubbles, fixed-target/LLP acceptance,
  distributed workload snapshots, and preservation bundles. Live controls,
  official experiment certification, and provider-owned production engines
  remain structurally external.
- Added dense fixed-rank TRG and HOTRG for uniform square-lattice partition
  tensors, including positive-semidefinite pair-weight lowering, exact
  plan/prepare/refresh identities, static resource admission, terminal
  partition accounting, local truncation and precision evidence, an Onsager
  qualification campaign, a compiled benchmark, and a runnable Ising example.
- Closed the production block-AMR geometry and execution seam with a canonical
  bucketed hierarchy/resource preflight, deterministic fill-aware patch
  clustering, high-order mapped metrics, explicit nonconforming mortar geometry
  and conservative fluxes, piecewise-linear two- and three-dimensional
  multivalued cut components, certified adaptive implicit sampling, polyhedral
  finite-volume lowering, physical common-refinement transfer, component-aware
  small-cell redistribution, unstructured viscous fluxes, disconnected-nullspace
  diffusion, moving swept-volume/content transactions with bounded multi-event
  localization, cut-complex cochains, commuting topology transfer and
  reflux-curl, finite executable signature caching, process-local live
  execution-group sharding, explicit derivative modes, and
  topology-reconstructing portable checkpoint/output artifacts.
- Added production-and-frontier QFT closure across improved compact-gauge
  actions and updates, lattice fermions and RHMC, distributed QCD ownership,
  observables, archives and continuum studies, Hamiltonian gauge sectors,
  fermionic Fock algebra, thermal DLR/Green-function and diagrammatic EFT
  workflows, relativistic scattering and VEGAS events, Gaussian/atomic and
  semiclassical QED, variable-sector VMC, exact learned and complex-weight
  methods, functional RG and nonequilibrium fields, supersymmetric lattice
  references, anyonic tensor categories, conformal bootstrap, and
  curved-spacetime QFT. Optional external lattice providers remain explicit
  availability probes when their packages are absent.
- Added explicit streaming three-term multi-shift Lanczos execution for
  self-adjoint shifted families below a certified spectral lower bound, with
  direct original-system residuals, conditional/certified forward-error
  evidence, complete resource accounting, and retained-projection execution as
  the default. Rational actions now propagate solve-error bounds through
  distinct pseudofermion refresh/action/force/acceptance policies and RHMC
  results; checked square solves expose `StabilityLowerBound` forward-error
  bounds. Streaming intentionally excludes preconditioning, warm starts,
  Arnoldi, nonreal shifts, and recurrence differentiation.
- Added physically certified scaled root systems with conservative stopping-limit
  conversion, matrix-free setup/adjoint propagation, complex-state support, and
  content identities; corrected Type-I/II Anderson damping, Hermitian secants,
  direct regularized Type-II least squares, bounded histories, initial fixed-point
  success, and exact work evidence; added explicit fixed-point-to-root conversion
  and independently certified singular/scaled root qualification cases. The former
  `PreparedScaledRoot` and `prepare_scaled_root` surface is replaced by
  `ScaledRootSystem` and `scale_root`.
- Added prepared native RA34PW2 solves and immutable record-once scheduled replay
  with exact accepted-step provenance, checkpoint-policy evidence, fail-closed
  weighted-RMS adequacy, and explicit schedule refresh.
- Corrected adaptive RA34PW2 error control to scale the componentwise embedded
  defect before weighted-RMS reduction. The unified `solve_rosenbrock` now owns
  fixed and explicitly requested adaptive execution; the separate adaptive entry
  point was removed.
- Added a finite lattice-field platform with cochain scalar `phi4` actions,
  exact local action caches, topology-native compact U(1), ordered non-Abelian
  boundary paths, matrix U(N)/SU(N) gauge links, Wilson actions, flat-torus
  and product-Haar Hamiltonian Monte Carlo, raw correlated-observable
  diagnostics, exact finite Z2 Gauss sectors, compact prefix-square MPOs, and
  open Schwinger-chain local/MPO/background-flux lowerings. The
  `CompactU1GaugeMeasure` constructor now consumes one canonical
  `CellComplexTopology` instead of duplicate dense incidence arrays.
- Added physical CAD revision and association identities, explicit B-Rep and
  planar partition results, generic region and patch controls, scheduled swept
  layers, planar-band evidence, coordinate-bound CAD persistence, and layout
  interchange/process-stack boundaries. Meshing providers now consume explicit
  topology-derived scopes; Gmsh coordinates belong to the B-Rep source, and
  fTetWild rejects unsupported patch controls.
- Added compact Morton execution planes, exact stable-ID k-nearest and radius
  queries, a Morton particle-neighborhood realization, deterministic
  capacity-evidenced FoF alternatives, distributed exact top-k merging, and an
  explicit Pallas distance kernel with native differentiation.
- Generalized the free-space Cartesian particle FMM through order seven with
  power-of-two coefficient scaling, capacity-evidenced Morton dual traversal,
  exact near completion, deterministic or compensated reductions, a
  whole-operator rematerializing VJP, and an explicit Pallas near-pair kernel.
- Added fixed-envelope bipartite Morton plane traversal and optional plane-dual
  spherical Laplace, Helmholtz, modified-Helmholtz, and vortex FMM execution;
  preserved layer/QBX far-field adapters; added wave-resolution route limits,
  high-order TreePM short-range Cartesian FMM, and an explicit
  capacity-evidenced screened-radius Ewald real-space route.
- Added native gravitational-wave inference with canonical one-sided detector
  spectra, interferometer response, declared waveform providers, normalized
  network likelihoods, physical parameter plans, nested-sampling preparation,
  phase/distance/time/calibration marginalization and joint reconstruction,
  exact-qualified relative binning, empirical-interpolation reduced-order
  quadrature, multibanding, posterior reweighting, hierarchical population and
  selection terms, simulation-based calibration, portable result context,
  bounded Bilby JSON import, an executable recovery example, qualification,
  and smoke/standard benchmark scenarios.
- Added normalized aligned-spin numerical-relativity polynomial-EIM mode artifacts
  with strict support/frame identities, caller-asserted content provenance, explicit
  unauthenticated/unqualified status, native spin-minus-two synthesis, geometric-to-SI
  evaluation, and nonprecessing mode symmetry; canonical one-sided PSD-weighted
  fixed-time overlaps and discrete phase/time-maximized matches; and a bounded
  UIB2016v2 aligned-binary remnant mass/spin fit. External JaxNRSur/Ripple runtimes,
  model HDF5, PSD/QNM tables, and phenomenological waveform coefficients remain
  unbundled and unclaimed.
- Added a layered black-hole closure across exact charted geometry and snapshot-bound
  ADM exchange; stationary/extended thermodynamics; angular perturbations and fixed-
  substep Schwarzschild Riccati/log-amplitude radial matching with generic
  $V/f=z^2W(z)$ order-12 infinity recurrence (finite Regge--Wheeler coefficients and
  recursive exact-rational Zerilli coefficients), independently gated Chebyshev ODE
  residuals, and strict axial/polar isospectral QNM regression; real scattering/
  superradiance and exact-state-bound Hawking spectra
  and bounded evaporation; accretion, plasma and first-order self-force; relativistic
  EOS/GRHD and periodic all-active GRMHD/CT, force-free, resistive and gray-M1 systems;
  bounded GR rays, typed polarized ray paths, exact chart/path/snapshot-bound midpoint
  sampling, MNY96 Stokes-I/K2 evidence with reference-unqualified polarization/Faraday,
  Jy images/interferometry and fixed-branch
  inference; Z4c/coupling; separately typed MOTS/apparent, isolated/dynamical and
  Hamilton-evolved offline event horizons; corrected characteristic Psi4 and
  harmonic-exactness BMS products; formulation-aware fixed-capacity block AMR; typed
  distributed restart; committed output/checkpoint receipts; complete artifact rights;
  resolver-evidenced production limits; exact runtime-manifest-bound qualification
  profiles; synthetic examples/benchmarks; focused guides and curated API pages.
  Technical evidence does not imply observational validity, a released production
  profile, external rights, or PNPL deployment authorization.
- Expanded the computational-chemistry candidate surface to general Gaussian
  shells/integrals, RHF/UHF/ROHF/GHF and moving-grid DFT response, bounded
  post-HF and representation-correct excited manifolds, vibronic/anharmonic
  spectroscopy, internal-coordinate reaction workflows, multipolar adaptive
  QM/MM, periodic Ewald/FFTDF/GDF/spin SCF, phonons/QHA/transport, and GW/BSE.
- Kept released chemistry support tuples narrow while adding separate candidate
  support dependencies and leakage-controlled qualification campaigns.
- Replaced paired tensor Gauss--Legendre adaptive cubature with native
  Genz--Malik and nested tensor Gauss--Kronrod rules, integrand-directed
  refinement, parent/children consistency errors, compensated signed
  reductions, strict point batching, physical axis breakpoints, and mixed
  scalar/`HyperRectangle` target support. `AdaptiveCubaturePlan` now takes one
  rule whose dimension owns the flattened coordinate layout; the former
  integer-first `low_rule`/`high_rule` constructor is removed.
- Extended the experimental learned total-energy D2V path with a
  pressure-consistent analytic particle stress, explicit learned support,
  portable frozen-model artifacts, atomic fixed-step D2V17 spatial transport,
  coupled population boundaries and force-work sources, leakage-safe physical
  rollout training and checkpoints, finite-volume-owned hybrid shock
  transactions, ordered multi-output DVM export, conservative D2V37 departure
  transport, and isolated matched-thermal research primitives. Exact energy,
  particle stress, learned constitutive flux, boundary/source ledgers, and
  failure rollback remain separately evidenced; no entropy theorem or
  paper-equivalent production shock claim is made.
- Added finite-molecule computational chemistry with explicit electronic state,
  model-chemistry, provider, property, and unit identities; loss-audited
  QCSchema, ASE, PySCF, and QCEngine boundaries; shared potential-energy
  surfaces; host-gradient geometry optimization; molecular Hessians, projected
  normal modes, RRHO thermochemistry, IR line strengths, lifecycle archives,
  atomistic Born--Oppenheimer adapters, qualification, and benchmarks.
- Added fixed-recipe selective reliability with pure dense OOF assembly,
  empirical-mass tie-block risk curves and weighted midranks, group-safe paired
  locked-test loss inference, and direct scalar interval diagnostics.
- Added inspectable temporal differentiation evidence; split composite,
  pushforward, and checkpointed pullback routes for Diffrax-backed evolution;
  matrix-free state and argument Jacobian actions; local derivative
  certification; argument-native shadowing problems; segmented JAX NILSS;
  discrete NILSAS with stored or recomputed adjoints and flow-neutral
  constraints; generic shadowing qualification; and sensitivity benchmarks.
- Added native fixed-capacity Cartesian block AMR with atomic host topology
  compilation, conservative epoch transition, source-classified FillPatch,
  block finite-volume ledgers, N-level SSPRK scheduling, composite scalar
  diffusion, exact-transpose packed distribution, fixed-epoch AD boundaries,
  and canonical stable-block checkpoint/HDF5-XDMF integration through the
  existing finite-volume lifecycle. Qualification remains limited to exact
  captured profiles and does not claim general AMR or external-library parity.
- Extended block AMR with sparse ancestry transfers, finite variable-patch
  shape buckets, bounded canonical node/edge/face/cell complexes, exact signed
  entity gather/scatter and commuting transfers, traceable mapped/ALE metrics,
  two-dimensional apertured embedded geometry, accepted-boundary moving-body
  topology transactions, and explicit variable-patch placement/restart.
- Added native orthonormal scalar spherical Legendre and angular/Cartesian
  spherical-harmonic evaluation with pole-safe derivatives, stable Cartesian
  normalization, lane-local zero/nonfinite refusal, and table-free evaluation
  of spherical spectral coefficients at runtime Cartesian directions.
- Added frame-explicit spin-weighted spherical point evaluation, regular and
  irregular solid harmonics, and matrix-free solid-harmonic synthesis over the
  canonical spherical mode layout.
- Added generalized integer-degree Gegenbauer values, modes-last Vandermonde
  evaluation, parameter derivatives, and private prepared quadrature,
  differentiation, and basis-connection resources.
- Added differentiated Riemann and Hurwitz zeta functions, principal complex
  dilogarithm and Spence functions, and a bounded general-order polylogarithm
  contract that refuses unsupported lanes.
- Added complete prepared three-dimensional Laplace, outgoing Helmholtz, and
  modified-Helmholtz multipole pipelines with all hierarchical passes, exact
  near completion, capacity/truncation evidence, and layer/QBX adapters.
- Added shared private spherical-Bessel sequences, migrated the cosmology
  radial table, and removed the geophysical digital-Hankel dependency on
  direct JAX fixed-order Bessel evaluation without adding public aliases.
- Added a prepared affine-simplex geometry map, scaled tiny-matrix determinant,
  side-specific interpolation fills, and one shared exact affine-Gaussian
  interval discretization kernel.
- Added native axis arrays and layout plans, native wavelet and orthogonal-polynomial
  kernels, native structured curvature, prepared finite Fourier transforms, and
  explicit active-set and barrier-KKT quadratic-program sensitivities.
- Added a bootstrap-safe distributed execution substrate with provider-neutral
  resource policies and plans, process-symmetric JAX execution groups,
  process-local batching, grouped worksets, ownership-aware distributed PCG,
  addressable-shard checkpoint/restart, cooperative failure scopes, ranked
  Slurm/Kubernetes launch, optional rank-local MPI collectives, and
  qualification-gated vendor profiles.
- Added a native causal-inference substrate with role-free observations and study
  designs; distinct DAG, ADMG, MAG, PDAG, CPDAG, and PAG semantics; randomized,
  adjustment, and finite general-ID/IDC functionals; exact finite and structural
  causal models with immutable interventions and abductive counterfactuals;
  cross-fitted g-computation/IPW/AIPW with fail-closed overlap and uncertainty;
  Fisher-Z, G², kernel CI, PC-Stable, conservative FCI, and bounded
  equivalence-class GES discovery; causal UQ experiment selection, conditional
  qualification, portable results, examples, documentation, and benchmarks.
- Added fixed-capacity active-key worklists, prepared target-grouped relation
  execution, virtual tensor index spaces, and sparse block topologies. Particle
  cell lists now store occupied cells only; sparse voxels and AMR metadata
  reuse canonical key grouping; Gaussian rasterization has a tile-major
  realization; and compact LBM, explicit/implicit/multifield MPM, phase-field,
  and transactional FLIP paths preserve logical IDs, capacity evidence, and
  branchwise differentiation.
- Added a default-silent Loguru event substrate with privacy-bounded text and
  canonical JSONL sinks, context-local correlation, structured training and
  provider/runtime events, explicit host/JAX telemetry snapshots, and
  process-safe sink ownership. Functional solver file logging now uses
  process-level `phydrax.logging` sinks and `log_every` defaults to zero.
- Added a transform-safe iteration execution substrate with bounded pure observers,
  typed domain records, explicit device stop rules, deterministic host sessions,
  capability-checked terminal/output/step/attempt/inner-iteration granularity, and
  checkpointed host cursors across production and training lifecycles.
- Added experimental PyTree-native conflict-free optimizer-update alignment
  with exact small active sets, canonical dual-QP fallback, real/complex cone
  evidence, and checkpointed gradient--update mismatch statistics across
  standard-Optax functional and neural-operator training.
- Added native functional domain decomposition with exact multidimensional,
  nonuniform, and periodic Cartesian topology; mapped-cover evidence; sparse
  partition-of-unity and side-aware broken fields; typed local references; physical
  value/flux/transmission, mortar, Nitsche, and augmented interface coupling; joint,
  persistent block, colored, relaxed trace-state, generalized, and bounded-staleness
  Schwarz training; arbitrary staged and V/F-cycle correction hierarchies; local
  Riesz residual norms; dense, KFAC, and matrix-free local curvature; transactional
  h refinement/coarsening and trainable positive-width partitions; explicit device
  placement and POU/Schwarz collectives; hybrid participants; restart; and
  content-identified deployment artifacts.
- Added multimodal acquisition collections, clock calibration, time-dependent
  frame routing, and bounded sample selection across external measurements.
- Added Hamiltonian graded-index rays with tangent-flow and caustic evidence,
  coherent thin-screen/multislice/Helmholtz Schlieren, matched CT projectors and
  reconstruction, and complex Cartesian/non-Cartesian MRI encoding.
- Added time-resolved surface, atmospheric, multipath, and stochastic LiDAR
  waveforms; bounded E57 and ROS admission; weather/FMCW radar profiles; and
  calibrated sonar waveform/beamforming/XTF contracts.
- Added task-bound neural-operator residual preconditioning, operator-informed
  Galerkin coarse-space lowering, immutable on-policy residual corpora, and a
  deterministic original-residual benchmark. Learned actions retain explicit
  field-transfer, resource, provenance, and FGMRES-only reliability contracts.
- Added a governed nuclear substrate with stable nuclide identities, processed-data
  provenance, canonical energy groups, material compositions, typed multigroup
  sources/fluxes, conservative fusion reactions, and fixed-network activation.
- Added axisymmetric tokamak conventions, bounded EQDSK and scoped IMAS
  interchange, nested flux-surface geometry, implicit core/current transport,
  fixed/free-boundary equilibrium, active/passive circuit coupling, transactional
  plants, shot governance, and fusion-to-activation workflows.
- Added scoped one-dimensional multigroup reactor diffusion, criticality, delayed
  neutron kinetics, candidate qualification profiles, synthetic qualification,
  examples, and performance benchmarks. All scientific profiles remain unreleased.
- Added a modality-neutral measurement substrate for physical quantities,
  sample supports, validity, uncertainty, acquisition identity, governed
  provenance, derivation lineage, and compatible observed/predicted comparison.
- Added generic scientific image supports and cameras, exact dynamic-surface
  rendering, LiDAR scan/point/range operators, and differentiable straight-ray
  Schlieren, knife-edge, and background-oriented image formation.
- Promoted camera, image-sampling, Gaussian raster, and photometric operators
  from velocimetry to their general imaging and rendering owners.
- Added production aerothermodynamics contracts and exact profiles spanning ionized
  multitemperature gas, per-reaction thermal control, implicit thermochemical source
  solves, ambipolar/electrostatic plasma coupling, and non-LTE multigroup radiation.
- Added persistent gas--surface chemistry, catalytic/plasma wall exchange, porous
  material response, conjugate heating, fixed-connectivity recession, and conservative
  gas/material topology remap.
- Added a native fixed-capacity DSMC substrate with VSS/VHS collisions, internal-mode
  relaxation, bounded chemistry, gas--surface exchange, production stepping, and
  fixed/dynamic continuum--DSMC coupling.
- Added named SST/DDES/IDDES closures, multicomponent DG admissibility filtering,
  high-enthalpy AMR/ALE evidence, distributed ownership ledgers, rights-bound
  validation campaigns, qualification tooling, and performance benchmarks.
- Added rights-checked medical-image assets with exact RAS/LPS, units, time,
  labels, tensors, NIfTI interchange, and audited image/mesh transfer.
- Added semantic compartment extraction and real multi-surface fTetWild meshing
  with exclusive cell zones and certified internal interfaces.
- Added metric networks, conservative embedded measure transfer, coupled
  3D–1D–0D transport, complete tetrahedral BDM₂/DG₁ Stokes with normal-flow and
  resistance constraints, CSF/PVS flow preparation, image-space inversion,
  identifiability evidence, and neurofluid workflows.
- Added exact-system high-speed flow composition: canonical mixture-safe viscous
  finite volume, diffusive source ledgers, physical wall capabilities, reconstructed
  surface loads and heat flux, normal/oblique-shock and expansion references, mapped
  airfoil grids, fixed-lift roots, buffet spectra/shock tracking, and rights-bound
  operator datasets.
- Added complete SA-neg-noft2 transport, explicit wall-distance and wall contracts,
  two-temperature neutral-gas states with thermal-mode energy, coupled fixed-work
  chemistry/relaxation, and hysteretic gradient-length Knudsen evidence.
- Added separate low-fidelity transonic small-disturbance and subsonic panel pressure
  policies. Neither route is presented as viscous, reacting, or supersonic CFD.
- Added leakage-controlled `ScientificCampaign` membership and
  unit/aggregation-exact `ScientificClaimProfile` evaluation, plus four narrow
  biophysical candidate profiles that remain explicitly unreleased.
- Added offline MegaScale/Tsuboyama protein-stability admission, grouped
  baseline and environment predictors, covariance-aware double-mutant challenge,
  and a fixed-construct periodic internal-coordinate decoder with unfiltered
  geometry evidence.
- Added source-pinned strand-displacement raw-trace admission, independent
  reporter calibration, effective-versus-mechanistic locked comparison,
  externally mapped conditional RNA ensembles, finite-support reweighting and
  thermodynamic closure, and content-addressed experimental design plans.
- Added zero-preserving timed radiation histories and plasmid-gel assessment,
  plus four-channel single-cell pulse/chase admission, identifiability, and
  held-out assessment. Repository application lanes report failed or
  inconclusive scientific gates without turning synthetic controls into
  experimental qualification; prospective qualification requires new
  acquisition after a frozen plan.
- Added a finance substrate with immutable point-in-time market and contract
  semantics, distinct physical/pricing/stress laws, curve and model calibration,
  analytic and stochastic valuation, econometrics, portfolio risk, credit/XVA,
  execution control, martingale transport, replay, archive, and qualification.
- Added accepted fixed-realization state-response pullbacks, blockwise physical and
  transpose evidence, all-at-once final recertification, heterogeneous multipoint
  state/design composition, simulation-anchored target matching, and frozen
  latent-to-physical state-design parameterization with mandatory final reanalysis.
- Added exact finite-outcome expected-information design with immutable log-space
  belief updates and bounded enumeration, plus correlated constrained noisy
  two/three-objective batch Bayesian optimization with pending-aware hypervolume.
- Added finite-horizon MAC/Boussinesq thermofluid material topology design and
  linear planar magnetostatic machine design with independently assembled torque,
  conservation, topology, bounds, and final-physics evidence.
- Added source- and executable-pinned XFOIL, DAFoam, HFSS Eigenmode/EPR/Q3D, and
  Fun4All/Geant4 detector-design adapters. External convergence, geometry/mode,
  derivative, dependency, correction, and artifact identities fail closed.
- Added native multi-fidelity SciML workflows with acyclic target-aware model
  hierarchies, sparse heterogeneous observations, leakage-safe physical-case
  splits, portable corpus archives, coupled fidelity MLMC, autoregressive
  Gaussian processes, target-information-per-cost acquisition, field correction
  operators, and fallback-free ROM evaluators.
- Added native staged multi-fidelity PINNs over `FunctionalSolver`: target-aware
  grouped split requirements, fixed heterogeneous observation penalties,
  frozen-parent additive and parent-conditioned corrections, continuous field
  transfers, target-owned replacement parameters, target-only evaluation
  evidence, and checkpoint-bound stage identities. Composed fidelity neural
  fields explicitly refuse KFAC until they expose an affine curvature layout.
- Added provenance-safe closure/operator deployment, conflict-free objective
  gradients, solver-interleaved periodic and MAC learning transitions,
  validation-driven native-Krylov refinement, terminal-physics flow matching,
  and conservative learned particle exchange bindings.
- Added a full-gated `CfCCell` with explicit elapsed-event semantics, packed
  physical-time execution, context-preserving recurrent stacks, fail-closed
  timed causal dispatch, and capacity-controlled irregular-event qualification.
- Added native heterogeneous neural-network execution with linear-storage cable
  solves, numerical event-time sensitivities, physical point neurons, bounded
  delayed routing, lifetime-safe plasticity, artificial recurrent spiking cells,
  population-code fitting, delayed regional dynamics, persistent BOLD
  observations, and bounded SONATA semantic interchange.
- Added explicit geospatial, vertical, time, borehole, and planetary coordinate
  contracts; pinned offline PROJ execution; and bounded SEG-Y 1/2, miniSEED 3, SAC,
  StationXML, raster, electrical, EM, potential-field, geodetic, and LAS adapters.
- Added 3D finite-patch, complete-electrode, singularity-subtracted point-electrode,
  line-current, and 2.5D DC resistivity; passive spectral/time-domain IP; borehole,
  marine, and mixed-dimensional casing workflows.
- Added free-space and spherical-harmonic gravity/magnetics, terrain/trend/continuation
  processing, constant/variable-density acoustic and isotropic/anisotropic elastic
  waves, CPML/free surfaces, spectral-element/AMR backends, FWI, RTM, traveltime,
  earthquake-source, surface-wave, ambient-noise, and HVSR workflows.
- Added layered, 3D tetrahedral H(curl), MT, implicit TDEM, and passive dispersive
  full-wave GPR electromagnetics, plus H(curl), shifted-Helmholtz, and true
  constrained-pressure-residual preconditioner composition.
- Added conservative multiphase component/energy flow, phase appearance,
  hysteresis/dynamic capillarity, freeze/thaw, vapor/atmosphere exchange,
  unstructured shallow water, wells, high-ionic-strength chemistry, monolithic
  reaction transport, and 3D/2D/1D/0D fracture networks.
- Added mixed Biot poromechanics, damage/contact/rate-state fault mechanics,
  earthquake cycles, GNSS/InSAR/tilt/strain observations, spherical geodynamics,
  calibrated petrophysics/geology, joint MAP/ensemble/pCN/time-lapse workflows,
  scalable covariance/prior/design actions, distributed halos, topology epochs,
  exact restart/resource refusal, and governed external-oracle/field qualification.
- Added native balanced power studies with separate physical/control contracts,
  sparse AC power flow, AC/DC optimization, explicit dynamic-machine and fault
  models, and bounded power-case interchange.
- Added reduced building energy models, explicit environmental boundaries,
  geometry/radiation enrichment, EPW forcing, native HVAC control, identifiable
  calibration, and held-out prediction.
- Added unit-aware carrier dispatch, independent storage capacities and physical
  chronology, investments/scenarios, and original-model conservation replay.
  Shared quantity-aware intervals compose the existing units and series substrates.
- Added oriented thermal capacitances, conductors, heat-conversion laws, material
  mixing, and provider-backed fluid heat exchange through native acausal DAEs.
- Added pinned host-only energy execution, FMI co-simulation and HELICS sessions,
  real external-reference routes, and cross-domain workflows with lifecycle
  archives and scenario-scoped qualification evidence.
- Added focused protein-folding, nucleic-acid-biophysics, and radiation-biophysics
  applications, plus exact single-cell transcript scenarios under systems biology;
  no broad bioinformatics namespace or duplicate simulation engines.
- Added explicit protein chemistry/force-field binding, joint thermodynamic and
  kinetic observation inference, paired-state enthalpy estimation, conditional
  rotamer free energies, rights-bound coordinate proposals, mixed Cartesian/rigid
  mechanics, and transactional co-translational activation.
- Added directed nucleotide identity, base-frame/eRMSD/torsion observations,
  chemical-mapping inference and reconstruction, rigid nucleotide model families,
  reversible secondary-structure CTMCs, and native electronic-site execution.
- Added source-pinned external radiation ledgers, mapped direct/indirect lesions,
  topology-aware clusters, explicit yield normalization, and staged calibration
  with held-out uncertainty and separate likelihood-rank evidence.
- Added a neutral loss-aware PDB record reader, identity-validated rigid site-load
  bindings, conditional rigid velocity heat baths, material insertion epochs, and
  event-exact CTMC hitting/absorption analysis. Scientific parameter, data, rights,
  and calibration gates remain explicit.
- Added a bounded semiconductor model family: named extensive carrier/energy/trap
  state layouts; aligned Boltzmann/Fermi–Dirac band thermodynamics; incomplete
  ionization; generalized Scharfetter–Gummel heterojunction transport; explicit
  sheet/dipole/thermionic interfaces; local high-field, electrothermal,
  carrier-energy, impact-ionization, WKB tunneling, and dynamic-trap ledgers.
  Added effective-mass Schrödinger–Poisson and density-gradient confinement,
  selected-source coherent transport with analytic semi-infinite leads and bound
  poles, conserving optical-phonon SCBA, refinable lead-memory transients,
  screened quantum response/noise, and explicit quantum/classical reservoir
  matching. Added leakage-safe correlated calibration campaigns, held-out
  prediction, provenance/rights archives, fail-closed empirical qualification,
  classical/quantum regressions, guides, API references, and benchmarks. No
  named-foundry calibration is claimed without authorized measurements.
- Corrected shared electrokinetic Poisson/Gauss and charge-continuity conventions
  across PNP, Maxwell, and PIC; stabilized Bernoulli differentiation at equilibrium.
  Sparse numeric factor refresh now preserves prepared route coalescing inside
  JAX loops, natural continuation preserves declared corrector spaces, and circuit
  operating-point solves honor declared residual scales.
- Added native geophysical quantity/storage bindings, explicit model calendars
  and interval support, pressure-coordinate measures, and intrinsic spherical
  tangent gradient/divergence/curl and Helmholtz wind inversion.
- Added deterministic gas-reservoir/multilayer-energy-balance climate scenarios,
  equilibrium-paired compressible dry columns/slices, conservative moist columns,
  and a fixed-step hydrostatic global spectral reference with explicit physical
  approximation, budget, rejection, and restart contracts.
- Added physically typed interval-integral coupling, slab/hydrostatic/Boussinesq
  ocean adapters, optional CF NetCDF/Zarr/GRIB interchange, native semantic
  archives, operator forecast/closure adapters, weighted climate diagnostics,
  and native ensemble-filter observation preparation.
- Repaired IMEX custom validators, zero-diagonal/zero-step derivatives and complex
  state promotion. Removed metadata-only balance-law integration-mode labels;
  finite updates remain process-owned and accepted source/transport/coupling
  inventory accounting is retained transactionally across retries and restart.
- Added forward and higher derivatives for fixed recursive spherical transforms,
  preserved shared geometry during native operator minibatch collation, and
  restored structured dynamics shapes for prescribed finite-volume advancement.
- Added independently checked balanced atmospheric references and dimensional
  energy/torque diagnostics; conservative gray column radiation; state-dependent
  wet-surface heat and water exchange; explicit conservative spectral water
  limiting with measured redistribution; and a restartable finite-rate moist
  column with cloud, rain, snow, sedimentation, and unresolved-mechanics evidence.
- Replaced phasewise anomaly contraction with a joint simplex/spectral
  projection that preserves the represented total-water field, measures
  phase adjustment against physical conversion, and closes moist enthalpy.
  Added energy-neutral angular-momentum projection, bounded TOA/surface flux
  preconditioning, and exact sequential checkpoint spinup segments. Raw
  corrections, lifetime intervention, stationary samples, and the distinct
  deterministic near-fixed-equilibrium gate remain explicit.
- Added paired conservative vapor/energy flux learning across declared resolution,
  interval, regime, and forcing supports, plus bounded physical-column calibration,
  identifiability analysis, held-out intervention scoring, and local experiment
  design. Synthetic qualifications remain explicitly separate from Earth-system
  validation and operational forecast skill.
- Added solver-neutral meshing specifications, revision-bound scopes, physical
  coordinate contracts, audits, quality metrics, staged provider results,
  interchange, topology lineage, and constrained fixed-topology optimization.
  Promoted reference-cell and coordinate geometry ownership to discretization;
  Gmsh construction now lives in `phydrax.meshing` rather than FEM.
- Added real optional Gmsh, Mmg, fTetWild, Manifold, Poisson, OpenVDB,
  VoroCrust, Omega_h, and TIOGA provider paths, including periodic/high-order
  CAD meshing, qualified straight layer sweeps, polyhedral output, and MPI
  adaptation/overset ownership evidence.
- Added compact mesh assemblies, shared `CellPartition` distribution lowering,
  conformal/periodic/contact/overset overlays, and safety-projected learned
  proposals with audited transactional acceptance.
- Added a source-pinned Potvin--Fuglevand 2017 sustained-isometric skeletal
  motor-unit population with exact quantity semantics, trainable numeric parameters,
  explicit central/peripheral fatigue mechanisms, hard-branch differentiation
  evidence, transactional rollback, a generic discrete-dynamics view, qualification,
  examples, documentation, and scaling benchmarks.
- Expanded the skeletal-muscle platform with source-named stochastic motor-unit
  discharge/twitch force, macroscopic fatigue/recovery, physical force calibration,
  complete Shorten fast-twitch cellular kinetics, structured moving-geometry
  one-dimensional fibers, sparse endplate territories, De Groote--Fregly
  explicit/implicit musculotendon dynamics, bounded analytic routes with
  three-dimensional lateral-cylinder wrapping, GASAM, prescribed-stress
  Heidlauf--Röhrle, and pinned idealized Almonacid muscle--aponeurosis continuum
  routes, physical fiber-current and ideal cylindrical-conductor EMG observations,
  feline spindle proprioception, Uchida--Umberger energetics, conservative retained
  heat with scalar heterogeneous Pennes fields, multimodal UQ, causal exact surrogate
  replay, immutable external-model interchange, and deterministic execution
  worksets/checkpoints. Each route retains one force owner and explicit
  source/data/hardware gates; numerical source equivalence is not biological or
  anatomical validation.
- Added a mathematically differentiable JAX-CPU sparse LU provider and a
  single-right-hand-side Krylov path that avoids unsupported batching of
  provider-backed preconditioner actions.
- Made high-order simplex nodal tabulation JAX-traceable through an equivalent
  Bernstein modal basis, enabling Taylor--Hood mixed finite-element closure
  conversion without NumPy tracer conversion.
- Added provenance-complete LES equations and production owners: static and dynamic
  periodic Fourier/MAC compilation; transactional dynamic ETDRK and projected MAC
  stepping; guarded static ETDRK and frozen MAC IMEX/SBDF2; enforced channel SBDF2
  stability with optional equilibrium-traction walls; accepted-step stochastic MAC
  inflow; static, buoyant, periodic-dynamic, and true-no-slip low-Re KSGS; learned
  stress Fourier/MAC divergence backends; device-resident distributed slab/pencil
  full-flow ETDRK/SSPRK production; pressure-stepped tetrahedral low-Mach restart
  continuation with conservative face-work KSGS production and explicit enthalpy
  thermalization; neglected or transported-SGS-energy Favre gas flow; fixed immersed
  MAC IMEX/SBDF2 ledgers; and collision-local LBM Smagorinsky. The public cutover
  removes `SmagorinskyLESClosure` and `MACStepRestriction` in favor of prepared LES
  plans and `MACLESStepRestriction`. `PeriodicModalTurbulenceStatisticsPlan` now
  binds compiled static or dynamic dynamics, and `PeriodicSpectralProductionPlan`
  takes `(dynamics, method, statistics, case, ...)` with an
  initial-condition-bound `PeriodicSpectralProductionCase`. Qualification outputs
  use the generic evidence spine, remain unsigned/unreleased candidates, and retain
  external base-profile release dependencies.
- Added a native optics platform spanning fixed-shape ray intersections,
  Snell/Fresnel interfaces, planar camera stacks, sequential/paraxial and bounded
  non-sequential tracing, sampled angular-spectrum propagation, thin and coherent
  field actions, dispersion laws, Maxwell/pupil adapters, differential Gaussian
  beamlets, pupil/PSF/OTF/MTF analysis, atmospheric and statistical-AO models,
  carrier-resolved nonlinear propagation, tissue radiative transport,
  fixed-frequency guided electromagnetic and elastic modes, SBS overlaps, and an
  optional host-only OpticStudio adapter. The Fourier-modal Maxwell boundary was
  hardened simultaneously with outward reference distances, directional periodic
  bases, physical unit-flux normalization, independent terminal-power auditing,
  and segment-aware continuous-layer dense fields.
- Added `phydrax.units`, an exact rational dimension algebra and immutable
  multiplicative unit catalog for explicit host-boundary conversion, coherent
  domain constants, content-addressed provenance, and raw-array prepared
  execution.
- Released the first public `phydrax.applications.battery` surface for the exact
  passive-sign, prescribed-current/rest lumped thermal ECM support tuple, with
  fixed positive R/C and capacity parameters, bounded OCV and entropic laws,
  native ODE execution, fail-closed status/termination and conservation
  ledgers, an external-evidence release-profile builder, generic trusted-index
  admission, a deterministic simulation plus generic scalar-optimization
  example, and guide/API documentation. This release makes no hysteresis, fade,
  circuit, multi-cell, pack, full-order electrochemistry, safety, lifetime,
  fast-charge, regulatory, or commercial-readiness claim.
- Added `phydrax.series`, a coordinate-neutral ordered-series substrate with
  shared or per-series masked supports, node- and edge-aligned numerical
  PyTrees, lazy reset-safe pair views, and explicit reconstruction policies;
  canonical trajectory data, sampled dynamics and state-space inputs,
  trajectory signals, and scalar cosmology histories now compose it without
  erasing their domain-specific validity or provenance contracts.
- Added reset-safe VAMP/VAC/TICA and Markov-state kinetics, exact full-batch
  variational encoders and model-backed collective variables, gauge-aligned
  learned free-energy biases, immutable atomistic learning campaigns,
  non-element molecular coarse beads with fixed-map force matching, and exact
  targeted free-energy maps with FlowJAX and alchemical endpoint adapters.
- Added the native rigid and soft robotics platform: explicit-root bounded URDF
  adaptation; four-space state geometry and true-dual mechanics; complete atomic
  plant, codec, checkpoint, and replay contracts; spatial native rods; PCS/GVS
  basis, reconstruction, materials, reduced dynamics, and integrators; atomic
  contact-free tendon and circular-capsule plane/self-contact plants; capstan,
  pressure, intrinsic-strain, variable-stiffness, and affine-magnetic actuator
  evaluators; continuum IK, observations, calibration, fixed-mode co-design, and
  sampling MPC; floating and fixed-topology rigid–soft composition; fixed-mesh
  FEM and fixed-topology MPM profiles; and an MJX-JAX complete-state plant gated
  by an exact 3.12.x provider pair and closed feature manifest. Added
  self-contained tendon/contact examples and a deterministic soft-rod benchmark
  matrix with evidence, residual, work, and storage reporting. Support claims
  remain limited to each exact declared capability tuple and runtime evidence.
- Added first-class surfel discretizations with stable point ownership,
  validated oriented tangent footprints, physical surface quadrature,
  boundary-atlas and simplicial materialization, Morton primitive bounds,
  bounded ray queries, and confidence-aware local sparse-voxel projection.
- Added deterministic explicit lowest-order H1 elements on conformingly
  segmented star-shaped polygons, with transported witness-fan condensation,
  exact trace and affine-reproduction evidence, component-aware constraints,
  reconstruction, differentiable geometry refresh, and capability-selected
  dense matrix-free local functional execution.
- Added the canonical `phydrax.applications.cardiovascular` platform with
  explicit, duplicate-free anatomy, electrophysiology, mechanics, circulation,
  hemodynamics, observations, and personalization facades; cross-domain
  quantity/case, fixed-capacity execution, checkpoint/replay, distributed
  reference, and fail-closed G0--G7 release contracts; harmonic cardiac
  coordinates and ventricular microstructure; phenomenological and physical
  monodomain, bidomain, eikonal, Purkinje/pacing, regional, and named cellular
  electrophysiology routes; passive/active mechanics, electromechanics,
  sarcomere, growth, and unloading workflows; 0D/1D circulation, coronary,
  valve, device, oxygen, fixed-wall flow, ALE/immersed FSI, and leaflet routes;
  observation, multimodal likelihood, inverse/design, cohort, surrogate
  refusal, learning, and native reanalysis contracts; public generic ownership
  for tensor diffusion, bounded array archives, and lifecycle support-bundle
  authorization; a hard-failing public end-to-end example, focused
  cross-domain integration tests, complete guide/API navigation, and bounded
  qualification/benchmark indexes. All supported claims remain limited to the
  exact declared research and engineering support tuple: no clinical,
  diagnostic, treatment, regulated-device, regulatory, or commercial-readiness
  claim is made.
- Added canonical Morton addressing, fixed-capacity sparse point hierarchies,
  traversed Barnes--Hut gravity, sparse occupied-level Cartesian and vortex
  FMM, brick-backed sparse voxel fields and qualified geometry sampling,
  atomic balanced dyadic adaptation with conservative field transfer, and
  explicit coarse/fine finite-volume lowering.
- Added `phydrax.signal` with explicit-axis differentiable windows and framing,
  finite direct/FFT convolution, causal FIR state, raw and aligned polyphase
  rate conversion, fixed-capacity causal streaming resampling, periodic Fourier
  resampling, and public fixed discrete wavelet transforms.
- Added research-tier conditional-affine chemical transitions with exact
  directional mass-action certification, inverse-free exponential/phi actions,
  reaction-shared positive rate correction, stoichiometric extent
  reconstruction, staged operator losses, portable artifacts, and explicit
  local `DiscreteSystem` deployment without clipping or hidden fallback.
- Added native order-two radial Laguerre, Fourier--Laguerre, Wigner, and
  Wigner--Laguerre transforms with physical `r**2 dr` normalization, together
  with resource-bounded exact directional ball wavelets and immutable ragged
  multiresolution coefficients.
- Added `phydrax.ein` as the package-wide optimized contraction boundary with
  native named JAX rearrangement, reduction, and repetition.
- Added log-stable numerator/support gradient accumulation shared by operator,
  discrete-dynamics, and standard-Optax functional training. Operator case
  measures now remain exact across weighted, masked, uneven, lazy, and sharded
  microbatches; optimizer, target, reporting, and checkpoint state advance only
  at accepted positive-support update boundaries.
- Added `phydrax.control.games` finite-horizon affine linear-quadratic
  full-state feedback Nash policies with explicit player control ownership,
  per-player quadratic values, case batching, differentiable nonsymmetric
  dense-LU solves, diagnostic-only rank SVDs, and independent curvature,
  stationarity, Bellman, conditioning, linear-status, and causal-failure
  evidence without regularization, pseudoinverses, clipping, or fallback.
- Added deterministic nonlinear game evaluation, physical/dimensionless nominal
  Nash residuals, exact-cost local quadratic policy suggestions, and
  residual-globalized finite-horizon iLQ with fixed-capacity plan/prepare/refresh
  execution and local nominal-stationarity evidence.
- Added explicit player-local, player-owned-coupled, and shared game-constraint
  ownership; sampled feasibility and multiplier layouts; convex open-loop
  variational equilibria with common shared multipliers; generic open-loop GNEs
  with player-specific shared-multiplier copies and optional bounded unilateral
  best-response audits; private nonlinear open-loop KKT; and fixed-active-set
  feedback quasi-Nash local models.
- Added prepared-noise stochastic feedback rollout, empirical-risk and paired-policy
  evidence, exact additive- and multiplicative-noise LQ control and feedback-Nash
  games, centralized observation-before-action Gaussian-belief LQG, and frozen-policy
  fitted Bellman evaluation with a BSDE bridge that keeps physical actions separate
  from martingale integrands.
- Added single-agent and player-owned open-loop stochastic-maximum-principle
  residual evidence, bounded one-dimensional HJB and zero-sum HJBI references,
  branch-explicit coupled-HJB policy iteration, and frozen-training/disjoint-holdout
  policy-game SAA with local empirical stationarity and cluster provenance.
- Added supplied frozen-law response evaluation, independently induced-law MFG
  fixed-point candidates, finite-scenario conditional common-noise MFG candidates,
  constrained individual/aggregate-generic/aggregate-variational MFG KKT evidence,
  finite-population continuation with complete numerical and simultaneous
  statistical deviation bounds, and MFC planner stationarity with explicit
  analytic or finite-particle measure-externality evidence.
- Added finite-state common-information pure-prescription Bayesian backward
  induction and an exact finite-state, finite-population empirical-law-lattice
  master-equation reference with Bellman, action-minimum, simplex, and discrete
  neighbor-transfer evidence.
- Added runnable nonlinear feedback, constrained open-loop, open-loop VE,
  stochastic feedback, additive LQG game, HJBI reference, mean-field fixed-point,
  and finite-state common-information examples. Each new game/control family
  retains its exact solution concept and does not silently repair inputs or fall
  back to a universal combined solver.
- Added qualified circuit-QED mode reduction and device assembly, one-to-one
  dressed-state tracking, sampled I/Q controls, leakage-aware gate metrics,
  exact-state local product formulas with reversible gradients, and exact
  heterogeneous MPO lowering without dense-Hamiltonian fallbacks.
- Added content-identified homogeneous Helmholtz thermodynamics with canonical
  component/phase-occurrence identity, explicit gas reference pressure,
  ideal-mixture calorics, Peng--Robinson residual properties and exhaustive
  roots, ideal-gas Gibbs equilibrium, tangent-plane stability, fixed two-phase
  TP flash, and frozen-composition homogeneous-mixture Euler flow.
- Added typed thermofluid components lowered to the acausal DAE substrate,
  immutable compressor maps/design calibration, role-specific spectral
  radiation coefficients and conservative matter exchange, and physical
  molecular-velocity kinetics with positive discrete Maxwellians, BGK,
  Shakhov, Maxwell walls, kinetic-breakdown evidence, and deterministic
  synthetic correction.
- Added a native real spectral-neuron layer with explicit ordered-eigenvalue
  selection, exact coordinate monotonicity constraints, fresh-initialization
  eigengap evidence, and invariant cluster-aware inspection.
- Added content-identified local quantum observables, lower-only Pauli-rotation
  program templates, grouped dense expectations, exact parameter-shift
  Jacobians, dense circuit feature models, fidelity kernels, variational binary
  classification, and native IQP/data-reuploading benchmark workloads without
  external quantum-framework dependencies or hidden normalization.
- Added native advanced-biophysics capability families: exact fixed-capacity
  path-space sampling and rare-event analysis; differentiable cable
  electrophysiology; dynamic particle relations and active biopolymers;
  Helfrich membranes and vertex tissues with transactional topology epochs;
  residual-gated polarizable and alchemical atomistics; compartmental systems
  biology with assertion provenance; and experiment-facing biophysical
  observation and qualification models.
- Added fixed-capacity open-boundary tensor-network completion with shared
  precision-aware SVD evidence, canonical QR sweeps, MPO construction/algebra,
  MPS action and compression, reusable bra–MPO–ket environments, network-native
  MPO Frobenius/Hermiticity diagnostics, capacity-bounded dense materialization,
  prepared two-site DMRG with truncation-aware Galerkin convergence, and
  one-site projector-splitting matrix-product TDVP.
- Added representation-specific MPS and locally purified `QuantumProgram`
  plan/prepare/refresh execution with template-state structure contracts,
  explicit nearest-neighbor routes, fixed bond and purification capacities,
  CP/PSD construction evidence, and observable norm, trace, and truncation
  loss without hidden SWAPs or normalization.
- Added immutable ordinary labeled-contraction structures with explicit output
  ordering, native contraction path/resource planning, fixed-signature
  prepare/refresh execution, precision provenance, and concrete prepared MPS
  and MPO inner-product consumers.
- Added static Abelian U(1), Z_n, and product-charge tensor layouts with
  oriented fixed-capacity legs, immutable tuple-block storage, separate MPS/MPO
  representations, blockwise environments and canonicalization,
  symmetry-preserving TEBD, and deterministic global cross-sector truncation
  evidence without fermionic or non-Abelian overclaims.
- Added the production tensor-network envelope: exact support tuples, conservative
  resource admission, execution manifests, bounded pickle-free archives,
  accepted-boundary checkpoint/replay, cancellation supervision, redacted
  telemetry, release qualification, and strict interchange/provenance records.
- Added production finite-chain methods with prepared environments, local/string
  MPO construction, variational compression, reduced/correlation/entanglement
  observables, projected-residual and variance-qualified finite DMRG, prepared
  one-/two-site finite TDVP, thermal purification, excited-state/response
  workflows, injective uniform states, VUMPS, and uniform tangent response.
- Added domain-neutral tensor trains with TT-SVD/rounding error bounds, tensorized
  grids and quantics, bounded deterministic TT-cross, structured Cartesian
  operators, ALS/AMEn solves, weighted completion, block eigenproblems, and an
  explicit tensor-train neural linear layer.
- Added canonical quantum instruments and experiments with exact branches,
  addressed replayable shots and feed-forward, route/decomposition ledgers,
  fixed-grid controls, MPO/LPDO Lindbladian workflows, Stinespring process
  learning/digital twins, and bounded service/interchange records.
- Added arbitrary-incidence tensor-network topology with traces, hyperedges and
  scalar nodes; inspectable deterministic schedules, full live-memory admission,
  exact slicing/checkpoints/reverse execution, placement and multi-device slice
  execution, finite PEPS/PEPO, boundary-MPS, CTMRG, simple/full updates, exact
  tree messages, loopy BP, circuit topology, and binary MERA.
- Added production Abelian block planning/algebra/solvers/open systems, explicit
  fermion grading and mode-order/Jordan-Wigner routes, and a separate SU(2)
  representation-category layer with deterministic CG/6j/F moves,
  multiplet-complete truncation, reduced MPS/MPO, DMRG, and TDVP.
- Added certified exact/bounded sharp-geometry realization with compatible
  MAC/FLIP/VOF coupling, explicit fixed-step and host-inspection adapters, and
  bounded nonconservative MacCormack transport for periodic passive tracers.
- Added a stateful functional-training runtime with canonical measure-weighted
  residual roots, named residual blocks, finite-width empirical neural tangent
  operators and diagnostics, pseudo-transient and causal residual transforms,
  gradient-norm and NTK-trace balancing, gradient alignment evidence, fixed
  evaluation selection, exact checkpoint/resume, named-axis sharding, physical
  time-window orchestration, and exact nonlinear defect correction.
- Added a Phydrax-native SOAP optimizer with resource-bounded per-axis Shampoo
  covariances, Adam moments in adaptive orthogonal bases, periodic QR refresh,
  independent moment/preconditioner dtypes, decoupled weight decay, JIT-safe
  mixed precision, and exact functional-training checkpoint state.
- Closed FVS-01–FVS-06 with mapped periodic viscous seam evidence, globally
  conservative multiblock positivity, explicit MAC continuation/checkpoint restore,
  canonical polyhedral finite-volume geometry, stage-refreshed moving WLSQ and
  fixed-combinatorics remap derivatives, arbitrary-normal/content-form entropy,
  mapped/ALE shallow-water balance, equilibrium WENO-Z/open/geostrophic routes,
  multilayer/Exner physics, shoreline event evidence, and LPP-resolved sub-float32
  storage.
- Added bounded stochastic capability families: multiplicative and affine-Hausdorff
  SING with explicit surrogate/audit semantics; finite coupled SPDE and
  particle/sparse-grid/separated Fokker–Planck approximations; represented-positive
  normalized densities and replayable stochastic boundaries; intrinsic Stratonovich
  and fixed-route rough preparation; finite-degree Wiener-signature certification
  with error/refinement evidence; finite GW/assignment/Gaussian-component/learned
  optimal transport; prepared finite diffusion bridges; and measure-explicit
  Riemannian/injective/conditional/eventful/hybrid/trajectory/finite-field flow laws.
  These additions do not claim infinite-dimensional execution, generic
  high-dimensional density solution, global GW/Monge optimality, exact mixture W2,
  continuum bridge exactness, path-space density, or densities for
  surjective/noninvertible routes.
- Added static leading-batch dense and shared-pattern sparse factorization artifacts,
  batched dense matrix-function and stochastic actions, explicit
  cross-dtype/complex sparse derivative and dual/Riesz/cotangent Hessian contracts,
  and threshold-certified bounded numerical inertia; Spineax zero-inertia remains
  unqualified.
- Added bounded public-JAX precision rewrite and finite-workload selection evidence,
  scalar FP8 and portable OCP-style MXFP8/MXFP6/MXFP4 formats with exact payload
  accounting, portable block-scaled contraction, deterministic local optimizer-state
  compression only (without communication or collectives), and complete complex
  parameter/optimizer/typed-RNG/auxiliary/checkpoint interchange.
- APP-01 added fixed-capacity single-pair close-encounter regularization with
  KS/Sundman evidence and rollback; clean-cutover
  `TLEPropagationPlan`/`TLEPropagationResult` with static near/deep SDP4 resonance
  routes; self-generated scalar Einstein–Boltzmann
  CDM/baryon/photon-polarization/massless-relic evolution with cold+baryon/total and
  unlensed TT/TE/EE products; and checksum-verified offline
  leap/EOP/gravity/ephemeris/IAU assets. Removed `Sgp4Plan`/`Sgp4Result` and the
  duplicate `EinsteinBoltzmannPlan`/`NativeBoltzmannResult` path.
- APP-02 added polar-cap/tripolar/equiangular-cubed-sphere hydrostatic mosaics,
  `TEOS10GSW75EOS`, explicit wet/dry epoch and saltation evidence, fixed/adaptive
  `ExternalModeSubcyclePolicy` schedules, and passive `TrajectoryData`
  lowering/advection. Replaced `split_substeps` and migrated continuation
  initialization to include the prepared ocean.
- APP-03 retains PR #235 as canonical for
  capillarity/wave/rigid/hydroelastic/two-phase/PLIC/contact-angle/graph-rezone, and
  adds explicit VOF wet/dry/moving-contact/surface-piercing/body-contact/breaking/
  overturning event evidence plus conservative two-phase remesh epochs.
- APP-04 and bio-specific APP-05 remain intentionally superseded by PR #232;
  bioinformatics stays removed and generic learned artifacts remain SNM-owned.
- Added held-out calibrated MC-dropout intervals; explicit frozen residual-noise
  mappings; proper/improper complex Gaussian observation laws; SWAG/SVGP state;
  overlap-gated Flow-NUTS evidence; structured kinetic actions; scheduled
  SGHMC/pSGLD; audited factor/operator minibatches; bounded nested plans; dense-exact
  causal HMC mass; and buffered particle-boundary evidence.
- Added bounded finite-atlas preparation and regular-level-set/immersion evidence;
  post-processing Gaussian and private Riemannian SGD; native complex optimizer
  leaves; fixed-rank density strata and explicit rank transitions; fixed-root
  Calabi–Yau moduli with mechanically gated non-proof certificates; exact bounded
  point-cloud/multiparameter/zigzag/cup/sheaf/spectral-page topology;
  coordinate-metric Clifford and nonassociative/G2/algebra-matrix families;
  finite-chart divisors, operator-specific analytic networks, and gauge transport;
  principal complex special-function continuation with Bessel order derivatives;
  and certified-tail compact homogeneous kernels with non-PD geodesic radial gating.
- K3/quintic trained checkpoints remain explicitly excluded qualification assets:
  existing constructors/solve/freeze/evaluate APIs ship no checkpoint, downloader,
  registry, format, or schema.
- Added bounded MPC-01–MPC-05 particle/mechanics closures: `CellMesh`
  barycentric/compact splats and conservative epochs; conserved wet DEM barrier
  reservoirs with periodic-envelope/stress evidence; fixed-budget LBVH, nonmatching
  tetrahedral hydroelastic patches, and Reynolds-film VI contact; equilibrium
  wall-vortex injection, uncertainty-bounded load recovery, solver-owned
  hybrid-event replay, and Helmholtz compressible augmentation; and atomic
  runtime-capacity SPH emission with exact source ledgers.
- Expanded spectral and structured linear execution with real JAX-mesh
  full-complex slab/pencil FFTs, horizontal-partitioned channel actions,
  resource-bounded layout/transposition evidence, partition-aware Thomas/SPIKE/PCR
  line solves, and invariant-extruded multiblock PCG whose block-direct factors are
  preconditioners rather than global direct solves. Line partition algebra remains
  in-process unless a higher-level communication owner is supplied; multi-host launch
  and scaling evidence are not inferred.
- Added generalized MAC pressure and control: frozen
  `-D(beta G_h p)` actions with Robin lifts and geometry epochs; exact
  transform/hybrid direct eligibility; PCG for symmetric positive actions; FGMRES for
  stabilized nonsymmetric traction; separate collective distributed projection; and
  prescribed-pressure-gradient, bulk-velocity, and frozen-density mass-flux
  method-stage response control with rank, conditioning, resource, residual, and
  atomic rollback evidence.
- Added exact candidate ownership for prescribed-marker, free-rigid,
  fixed-topology-sharp, deformable/contact, LBM-body, and resolved CFD–DEM immersed
  regimes. Two-phase runtime admission binds existing owners, support tuples, ranks,
  resources, derivative scope, motion/topology/geometry epochs, distributed
  reductions, gap state, sharp measures, and load provenance without route fallback.
- Added smooth, finite-volume, all-speed, shock-resolving, and slow-growth
  compressible-flow application policy over the canonical all-species homogeneous
  Helmholtz gas state. Added canonical mixture Navier–Stokes transport, normal
  characteristics, entropy evidence, full-species forcing/budgets/Favre statistics,
  relative-Mach all-speed HLL, and pressure-sensor/admissibility generic-HLL fallback.
  DGSEM/BR1, nodal-DG/LDG, structured/mapped WENO-Z/TENO/MP5, and Peng–Robinson
  phase-equilibrium ownership remain distinct. Temporal and modeled-spatial slow
  growth freeze one baseflow snapshot per parent step and retain
  `claims_spatial_dns=false`.
- Added reacting-flow mixture-averaged and bounded dense Stefan–Maxwell transport,
  fixed-schedule Strang and iterative-trapezoidal chemistry with atomic rollback,
  full-species reactive statistics/closure targets, and a separate low-Mach divergence
  constraint over the canonical component/schema/Helmholtz/mechanism/Euler owners.
  Chemical sources preserve full chemical total energy with a zero energy RHS; heat
  release is diagnostic. Cantera import/reference is an explicit host-only,
  non-differentiable, feature-gated boundary with explicit gas standard pressure.
- Added LBM nondimensional operating envelopes, per-device resource preflight,
  exact deployment compatibility, C0/C1/C2/C3 and conjugate-thermal unsigned
  candidate profiles, and passive sensible-energy CHT with equal-and-opposite
  interface heat rates and rollback. Profile evaluation does not set `released=true`,
  and host/device declarations do not by themselves establish multi-host execution.
- Added closure-data state/trajectory identity, symbolic filters and conservative
  alignment, deterministic target lineage, complete content-addressed chunk coverage,
  leakage-safe partitioning, train-only normalizer provenance, and artifact-bound
  conservative-face/spectral-drift deployment. Invalid spectral drift returns an
  explicit typed zero-drift fallback record rather than changing behavior invisibly.
- Added finite exact QL/CE2 and GQL/GCE2 statistical dynamics, independent-real
  barotropic beta-plane cumulant coordinates, dense/factor covariance evidence,
  bounded segmented NILSS, continuation, logical shard layouts, and semantic-preserving
  restart redistribution. Logical shard helpers are in-process array algebra, not a
  multi-device statistical solver.
- Added platform closure for exact support/evidence matrices and unreleased
  candidates, resolved run identities, pure forward configuration migration,
  transactional POSIX/conditional-object repositories, direct range-based
  topology restart, durable reference-service orchestration, Slurm/Kubernetes and
  identity provider boundaries, support bundles, process-local secret handles, and
  explicit signing/trust rotation. Asymmetric JWT/X.509 and Ed25519 require optional
  cryptography; no entry claims signed candidate evidence or universal certification.
- Added declared-incidence acausal DAE structural reduction; bounded DAE
  reset/consistency/regularity/manifold stages; generalized-pencil/Hopf
  continuation; prepared case-axis iLQR; Radau/multiphase/complementarity/
  stochastic/manifold transcription audits; continuous-path and finite-box
  optimality certificates; typed adaptive stochastic-delay interpolation;
  archived-primal delay backsolve adjoints; exact exponential/certified-tail memory;
  canonical hybrid event tape/replay/log-Jacobians; and fixed-capacity whole-solve
  segment evidence.
- Completed QPV-01–QPV-08 with positive-regulator finite-slice real-time paths;
  canonical adaptive/source/geometry evidence; periodic/U(1)/exchange measures;
  root HMC, chunks, proposal adaptation, and incremental caches; adaptive/symmetric
  and finite-subspace Cayley TDVP; resource-admitted
  electronic/periodic/no-pair/stochastic-trace routes; PR #236 canonical
  `QuantumProgram` measurement, bounded control, and tensor execution; canonical
  finite CPTP maps/integration; and finite
  Fock/HEOM/compression/steady-state/identifiability certificates. All claims are
  finite, truncation-aware, and fail closed.
- Wave C added revision-checked affine CAD-to-FEM meshing and prepared FE cell maps;
  root rank-r `PeriodicCell` and face-defined `PolyhedralConnectivity` with
  polynomial three-dimensional VEM consumers and hp adaptation; Laplace DP0
  three-dimensional kernel-independent FMM/H/H² with exact prepared near blocks;
  continuous-P1/DP0 scalar Calderón, stable dual spaces, mortar traces, nonmatching
  FEM–BEM, screen-junction, modified-Helmholtz, and displacement-discontinuity
  products; bounded finite-image periodic Maxwell and rank-two periodic free-surface
  products; fixed-history elasticity/Maxwell FEM–BEM CQ node-family controllers;
  fixed-topology nonlinear/viscous-potential and second-order QTF products; bounded
  contact/fracture operator lifecycle; and epoch-bound prepared capacitance with
  explicit topology transitions, fixed-epoch coordinate JVPs, and a
  rank/shape-certified stable dual Calderón preconditioner. These claims do not cross
  epoch or pair-class changes and do not extend outside certified shape-regular dual
  families.
- Added optimization/search/calibration capabilities: sparse public `ConicProgram`
  with native-device and sparse Clarabel routes; bounded CVXPY and explicit MPAX
  representations; public finite top-k/Pareto/adaptive reducers; bounded
  mixed-integer convex search; mixed/Pareto differential evolution with guarded
  validity; covariant/interval/group/subset calibration contracts; weighted
  PAV/typed ordering; relaxed Bernoulli/top-k; inverse-logit evidence; and prepared
  CSG continuation. Added direct `SparseStorage`-to-BCOO MPAX sparse lowering for
  zero/nonnegative cones; matrix-free conic JVP/VJP through
  `JacobianLinearOperator` with matching verified `StabilityLowerBound` and
  selected-projection `ConicGeneralizedDerivativePolicy` evidence; canonical
  conic exact/interval/group relative-entropy calibration; KFAC logical
  block-axis/kind/complex-Cartesian/sharing metadata and structured layout
  lowering; and a clean cutover from the legacy private finite reducer, with
  control, UQ, and tool callers migrated to the public surface.
  KKT inertia now consumes canonical `linalg.InertiaEvidence`.
- Closed CID-01–CID-12 with typed affine trace enforcement, signed adaptive
  populations, lazy ragged pooling, bounded breakpoint discovery, N-D adaptive
  cubature, adaptive Smolyak integration/interpolation, typed probability reference
  transports, GTA-evidenced nonuniform scaled cubature, matrix-free Type-1 scattered
  Fourier fitting, mixed Fourier/Chebyshev/Legendre reconstruction, trainable
  B-spline banks, and certified rational/trainable KAN topology transitions.
- Expanded bounded GP/BQ/coreset capabilities with rational/separable state-space GPs
  (SHO, stable CARMA, sums, repeated/derivative/spatial rows, exact associative
  covariance filtering, and certified Bernoulli/Poisson Laplace sites);
  fixed-capacity iterative and UQI action-space computation-aware inference;
  finite-coordinate signature-path functionals; mixed constrained q-batch Bayesian
  optimization with native GP fitting; exact interval/finite-measure/finite-feature
  kernel means and sequential BQ; and native operator-case/query, trajectory-block,
  and empirical-cubature adapters.
- EPC-01–EPC-04 added bianisotropic/patterned-port continuous-z Fourier modal
  execution with TO-PML, finite-aperture fields, and harmonic epochs; nonperiodic
  reduced PIC/Maxwell, curved FE location, and finite-phase/ALE FLIP; integer-ratio
  multilevel LBM AMR, prepared replay, and forward-VJP IREE; and fixed-capacity
  higher-order/adaptive coupling waveforms, windows, and topology epochs.
- Fixed-capacity per-collocation Diffrax quadrature now participates in the canonical
  `IntegrationPlan`/`materialize`/`reduce` lifecycle with solver-identity and failure
  evidence.
- SNM-01–SNM-17 added arbitrary-query wavelets and directional scattering, point
  O(d) CNO, multi-source frames and coefficient flows, checksummed first-party
  FNO/DeepONet weights, complete recurrence/rollout, boundary-aware masked CNO/UNO,
  replayable Galerkin/characteristics, attention replacement/anchors, modal
  discovery/recovery, generalized residual layouts, transformed complex alias-aware
  low rank, constrained polyconvex/Onsager wrappers, CID collocation with typed
  integral rewrite and target/causal workflows, and soft/learned-cutpoint ordinal
  classification. Tier promotion remains excluded.
- Added deterministic dense local quantum programs with explicit mixed-dimensional
  Hilbert layouts, ordered local unitary and Kraus contractions, CP-by-construction
  and trace-preservation evidence, resource-bounded plan/prepare/refresh execution,
  physicality diagnostics, and fixed-schema JIT, batching, and gradient contracts.
- Added prepared harmonic-balance planning, resource evidence, numeric refresh,
  and provenance around the existing Fourier-collocation circuit residual and
  matrix-free native nonlinear solve.
- Closed the single-process production high-order conservation surface with
  normal-first boundary capabilities, typed boundary traces, method-neutral
  stage ledgers, generalized operator programs, affine/weight-adjusted/exact
  mass strategies, arbitrary-order prism/pyramid references, transformed
  periodicity, generalized SBP entropy flux differencing, entropy-compatible
  mortars and boundary contracts, generic entropy-diffusion viscous DG,
  shape-generic filtering, conservative subcell/correction ledgers, hp/ALE/GCL,
  IMEX/LTS, resource and sensitivity evidence, durable run transactions, and
  byte-bounded exactly-once output.
- Added beyond-core reacting multispecies flow, ideal-MHD and shallow-water
  entropy pairs, LES/RANS and wall closures, CAD curvature adaptation,
  conservative sliding/cut-cell/overset coupling, and frozen-event reverse-time
  topology adjoints. MPI and distributed execution remain intentionally outside
  this closure.
- Added high-order conservation completion with physical DGSEM boundaries,
  conservative SSP-stage entropy filtering, tensor LDG Navier–Stokes, stable
  simplex and hybrid references, exact-mass nodal DG, mixed 2-D/3-D mortars,
  high-order mesh import, cost-aware distributed phases, runtime checkpoints,
  exact-time schedules, streaming triggers, and bounded asynchronous publication.
- Added advanced hydrodynamics with corrected graph-stage timing, pressure/reference
  semantics, unified mapped kinetic and boundary ownership, truthful work ledgers,
  variational surface tension, coherent wave forcing/absorption, vertical rezoning,
  shoreline handoff events, submerged rigid/modal coupling, and a separate
  conservative incompressible two-phase VOF/CLSVOF product with variable-density
  projection, capillarity, moving-body forcing, topology evidence, restart, output,
  qualification, examples, and benchmarks.
- Added executable geometry-state validity, exact analytic extrusion and revolution,
  dense host-discovered fixed-topology implicit surfaces with normal-gauge projection
  and regularized native QEF realization, and graph-harmonic finite-element mesh
  motion with signed-Jacobian evidence, safe rejected-trial fallback, qualification,
  examples, benchmarks, and explicit topology-refresh boundaries.
- Added prepared periodic Fourier-shell statistics with continuum FFT normalization,
  Hermitian mode accounting, DC/Nyquist/final-edge policies, measured one-epoch matter
  power, auto/cross spectra, phase-sensitive spectral discrepancy, Parseval evidence,
  and inverse particle-field realization through existing splat, covariance,
  optimization, and sensitivity substrates.
- Added fixed-topology one-phase free-surface ALE hydrodynamics with graph
  geometry, extensive mapped momentum/scalars, conservative kinematic GCL,
  nonorthogonal mapped Hodge, mixed pressure projection, strongly coupled
  second-order stepping, accepted work ledgers, strict restart/output,
  qualification, examples, and benchmarks.
- Added hydrostatic primitive-equation ocean modeling with prognostic free surface,
  extensive layer transports and tracer inventories, implicit and split-explicit
  external modes, z-star and partial-cell geometry, freshwater volume sources,
  Flather/radiation boundaries, conservative wetting/drying, beta-plane and bounded
  latitude-longitude metrics, checked vertical implicit mixing, nonlinear seawater
  thermodynamics, Ri/KPP-like/TKE/Redi-GM closures, accepted ledgers, restart/output,
  qualification scenarios, examples, and benchmarks.
- Extended marker-flow coupling from the fixed uniform baseline to a shared
  stage-inverse KKT contract with explicit route state, physical-boundary correction,
  rank/condition gates, multiple regularized kernels, deterministic/compensated
  transpose reduction, variable-density and SPD variable-viscosity stages, nonuniform
  and mapped transfer, accepted-time rigid backward-Euler/midpoint, monolithic FE
  Newmark, native joint/contact adapters, resolved-subtraction lubrication, composite
  AMR impulse reflux, distributed single-owner transfer, conservative marker topology
  epochs, divergence-free projected transfer, sharp cut-cell and immersed-interface
  families, moving sharp epochs, fluctuating inertial and overdamped FIB methods,
  complete checkpoint/replay/output/runtime records, canonical examples, qualification,
  and scaling benchmarks. Advanced mapped, AMR, distributed, contact, sharp, and
  stochastic families retain explicit case-specific qualification gates rather than a
  blanket production claim.
- Added an authoritative surface and boundary-integral platform with checked
  linear solves, scalar Calderón formulations, periodic scalar kernels,
  FEM–BEM coupling, finite-depth potential-flow hydrodynamics, adaptive and
  block execution, portable archives, static elasticity and Stokes kernels,
  convolution quadrature, RWG Maxwell support, and fail-closed commercial
  qualification evidence.
- Reconciled cosmology, astrodynamics, and astrophysical-observation foundations:
  dimensional scales, artifacts and derivative capabilities, labeled observation/
  covariance/likelihood algebra, direct and hierarchical particle gravity, KDK
  transactions, ratio-two AMR mechanics, and event replay now have core owners.
  Domain applications retain comoving/canonical/scale-factor, physical epoch/frame/
  encounter, and instrument-specific semantics. Removed the astrodynamics nominal FMM
  and TreePM names that did not implement those algorithms.
- Added bounded maximal native cosmology profiles: fixed-layout thermodynamics/scalar
  transfer/line-of-sight algebra, global S3 manifold/KDK/harmonic Poisson/particle
  transfer, typed multi-release survey composition, deterministic FoF/unbinding/M200m/
  substructure/merger products, dynamic replayable stochastic stellar feedback,
  two-level ratio-two AMR, a shared Morton particle octree, isolated Barnes--Hut,
  uniform Cartesian FMM, and BH-short-range single-device TreePM. Each profile records
  explicit unsupported physics, topology, approximation, capacity, distribution, and
  communication boundaries.
- Added experimental granular micro--macro completion: fitted finite-volume
  capillary bridges with analytic energy and fit margins, radius-derived contact
  envelopes, conserved film/bridge inventory and exposed-area evaporation,
  balance-audited particle and interaction-segment continuum fields, sparse
  multilevel polydisperse neighborhoods, and dense-authority deforming periodic
  DEM cells with mixed stress/strain control and cell-work rollback.
- Added commercial Material Point Method closure contracts: exact claim tuples and
  executable support decisions, intended-use and G0--G7 release evidence, durable
  atomic checkpoint generations, HDF5/XDMF/VTK output, host-side supervision and
  observability, PIC/FLIP/blended/APIC transfer and independent advection, affine and
  post-advection MUSL, vector-root anisotropic plane stress, non-associated
  Drucker--Prager and Mohr--Coulomb plus Modified Cam-Clay, typed porothermal fields,
  simultaneous K-way contact with essential constraints and shared rigid reactions,
  topology-aware moving-domain and compact implicit actions, deterministic execution
  and capacity certificates, distributed ownership/global transactions, conservative
  particle lifecycle and ratio-two AMR, and evidence-tagged branchwise, event-aware,
  generalized, surrogate, stochastic, or nondifferentiable derivative products.
- Added full nonlinear solid-mechanics closure: canonical finite-strain laws,
  safeguarded plane stress, mixed incompressibility, conservative and follower
  loads, transactional continuation, physical bifurcation/selection, current-
  geometry contact, sharp and diffuse fracture, state-certified topology
  optimization, and parameter-measure-aware amortized operator learning.
- Added explicit state/adjoint acceptance evidence, prepared neural-field
  stationarity and virtual-work roots, separate physical static/dynamic
  stability contracts, and accepted-state continuation checkpoints/replay.
- Added three-dimensional Cartesian rigid-lid Boussinesq ocean process modeling
  with linear temperature-salinity reference physics, weighted-skew f-plane
  Coriolis, directional scalar diffusion, conservative surface scalar fluxes,
  impermeable surface stress, coupled fail-closed SSPRK3, accepted budgets,
  strict restart/output archives, qualification scenarios, examples, and
  benchmarks. Hardened coupled MAC scalar CFL, boundary-stage propagation,
  buoyancy exchange evidence, and rotation/stratification step restrictions.
- Added native fixed-topology partitioned multiphysics coupling with exact typed
  participant ports, direct and paired field transfers, deterministic SCC plans,
  explicit Jacobi/Gauss–Seidel sweeps, physically certified implicit interface
  roots, atomic rollback, fixed-window replay, fixed-grid waveform/subcycling
  contracts, resource and work evidence, and explicit algorithmic or implicit
  differentiation semantics.
- Added the fail-closed `iga.tensor` R1 isogeometric foundation for regular
  untrimmed full-dimensional 1D/2D/3D polynomial and NURBS maps; anisotropic and
  independent geometry/field grids; direct-tensor and extracted-Bernstein
  realizations; explicit common integration overlays; self-periodic traces;
  h-, p-, and k-refinement transfer; scalar, vector, and mixed H1 fields; linear
  elasticity, thermoelasticity, and generalized eigenspaces; fixed-topology
  differentiable numeric refresh; and immutable native restart lineage. Added
  exact allow-listed support tuples, deterministic per-case qualification
  manifests, an unreleased capability-profile producer, public examples,
  S1 migration fixtures, and record-only performance producers. Sampled map
  evidence remains neither a global injectivity certificate nor BRep/CAD
  support, and no capability is released without separately signed gate evidence.
- Added the representation-independent `phydrax.variational` functional substrate,
  DomainFunction bindings, and prepared-local value/first-variation/Hessian
  execution for coupled finite-element and isogeometric potentials.
- Unified `IntegralFunctional`, `VariationalEigenspace`, and
  `InvariantSubspaceResidual` on typed integration sources; fixed randomized
  objectives now require explicit realizations.
- Added `LocalFunctionalAction`, `finite_element_form_from_functional`, and
  `compile_finite_element_functional` alongside representation-bound
  `CellEnergyAction` and `FiniteElementFunctional` adapters.
- Routed the portable Neo-Hookean functional, DomainFunction operators,
  finite elements, and material points through the canonical finite-strain law.
- Closed remaining advanced-cosmology boundaries with projected canonical physical
  state, content-addressed products/artifacts, dependency-aware derivatives, shared
  observation/covariance likelihood algebra, concrete pinned precision-process
  wrappers, one-loop SPT, calibrated 200m halo/galaxy foundations, release-locked
  survey likelihood contracts, primordial H/He microphysics, local-curvature evidence,
  low-resolution CMB sky/TOD/mapmaking, periodic Ewald qualification, snapshots, and
  distributed-PM feasibility. Native full Boltzmann/CMB parity, global curved N-body,
  generic surveys, stochastic feedback, and production tree gravity remain explicit
  non-goals rather than fallbacks.
- Added a common chemical species and phase schema, NASA and polynomial species
  thermodynamics, prepared deterministic and stochastic mechanisms, native stiff
  reactors, extended rate laws, YAML interchange, and calibration coordinates;
  added compatible Poisson--Nernst--Planck transport, reactive electrodes,
  electrohydrodynamic and multiphase electrolyte coupling; and added compact
  Q-tensor Landau--de Gennes, Beris--Edwards, anchoring, active, chiral,
  electrostatic, and electrolytic liquid-crystal dynamics.
- Added one fixed-capacity runtime particle-population authority with activity,
  mass, incarnation-safe slot reuse, deterministic allocation/deactivation, DEM
  lifecycle migration, and runtime particle-splat masks.
- Added advanced PIC capabilities: integer charge states, conservative binary and
  background collisions, impact/field ionization, 1D3V/2D3V compatible Maxwell,
  reduced PIC current projection, open particle ledgers, CPML compatibility,
  integer moving windows, affine simplicial electrostatic/Whitney-current PIC,
  conductor KKT coupling, unstructured electromagnetic PIC, and matrix-free
  semi-implicit particle response with bounded Gauss correction.
- Added advanced FLIP capabilities: deterministic fixed-pool reseeding, particle
  level-set and ghost-fluid geometry, sharp capillary pressure jumps, moving-solid
  cut-cell and particle collision ledgers, free-surface viscous measures,
  variational symmetric-strain viscosity, and two-phase one-velocity FLIP.
- Added fixed-geometry 3D Laplace DP0 surface Galerkin capacitance solves with
  explicit weak/strong maps, bounded singular and near-pair quadrature,
  nonmaterializable blocked actions, immutable conductor selections, physical
  charge integration, and reuse of the existing direct and QBX layer evaluators.
- Added fixed-capacity two- and three-dimensional vortex methods with
  Gaussian free-space direct fields, periodic vortex-in-cell inversion,
  conservative particle-strength exchange, classic stretching, regularized
  filaments, steady and unsteady lifting surfaces, rigid polygonal vortex
  panels, boundary-sheet transfer, conservative remeshing, explicit rVPM and
  relaxation operators, nonlinear polar closure, fixed-tree acceleration,
  actuator/rigid/stochastic/learned workflows, qualification evidence, and
  fixed-topology differentiation contracts.
- Closed the native vortex capability boundaries with typed source/target and
  capability contracts, dynamic-core formulations, periodic Ewald and
  free-space FFT authorities, corrected P3M, hierarchical 2-D/3-D FMM,
  transactional populations and epoch replay, shared ring/sheet wakes,
  multi-surface lifting and complete loads, native 2-D/3-D panels, no-slip and
  immersed wall coupling, rigid/flexible FSI, rotor/actuator/control/acoustic
  workflows, stochastic ensembles, constrained learned reconstruction and
  assimilation, portable checkpoints/exports, and explicit sharding evidence.
- Added fixed-topology material-measure immersed-boundary coupling on uniform
  unit-density MAC grids: local cubic B-spline marker routes, force/torque/work
  certificates, exact prescribed pressure-plus-marker projection, IMEX-Euler and
  SBDF2 execution, explicitly separate penalty CFD–DEM, generic free rigid-body
  coupling, fixed FE marker H/H* maps, synchronized deformable coupling, and
  fixed-routing implicit sensitivities. Variable density, mapped/AMR/distributed
  markers, remeshing, contact extensions, fluctuating hydrodynamics,
  divergence-free interpolation, and sharp-interface changes remain unsupported.
- Added field-valued logarithmic compressible Neo-Hookean reference energy,
  line-search-safe nonfinite integral propagation, and an experimental matched
  neural-variational/finite-element hyperelastic qualification.
- Added advanced cosmology contracts: curved/CPL FLRW geometry and distances,
  realization-safe semantic transfer/power products, process-isolated linear-theory
  interoperation, neutrino component algebra, model-card power corrections,
  adiabatic gas--particle shared gravity, analytic halo foundations, Limber/RSD
  predictions, canonical CMB spectra, and bounded periodic force qualification.
  Periodic LPT/PM remains flat-only; calibrated external spectra and production
  distributed gravity remain explicit provider/qualification boundaries.
- Closed the declared astronomy capability boundaries with exact astronomical time
  instants and routes, IERS Earth orientation, compiled frame graphs, pinned
  artifacts and Chebyshev ephemerides, CCSDS/TLE products, high-fidelity force and
  light-time models, adaptive Gauss--Radau IAS15, analytical/DSST propagation,
  bounded event and maneuver schedules, encounter and hierarchical gravity,
  coupled variable-mass vehicles, tracking/variational/orbit-determination/mission
  products, calibrated WCS imaging, surveys, radiative transfer, waveform and
  exoplanet operators, native early-universe/Boltzmann/nonlinear cosmology, and
  compact-object EOS/TOV models. Provider discovery, network access, external data
  redistribution, and smooth-gradient claims across discrete topology remain
  intentionally excluded.
- Extended Material Point Method with explicit USF/USL-minus/MUSL schedules,
  fixed-capacity adaptive realization and scheduled replay, constitutive capability
  and algorithmic-tangent contracts, isotropic plane stress, multiplicative
  finite-strain J2 plasticity, uGIMP/cpGIMP/CPDI/CPDI2 particle domains, rigid and
  two-field Coulomb contact, material/velocity-field identity state, active-block
  semantics, compact block storage, dense matrix-free implicit roots, AT2 diffuse
  fracture, and separate field-partition/CPIC sharp-fracture paths. Each family
  carries transactional rollback, branch/topology evidence, qualification artifacts,
  compatibility limits, and explicit differentiation semantics.
- Added one- and two-dimensional Cartesian wet/dry shallow-water finite volumes with
  exact dry-state semantics, prepared static bathymetry, Chen--Noelle hydrostatic HLL
  face contributions, equilibrium-aware MUSCL reconstruction, SSPRK-stage conservative
  positivity, accepted one-sided bed-integral evidence, f/beta-plane Coriolis forcing,
  renderer-neutral observables, output support, qualification cases, and benchmarks.
  Removed the unqualified one-dimensional shallow-water f-wave path.
- Added native experimental velocimetry with mask-aware multipass and ensemble
  PIV, explicit peak/validation/replacement evidence, calibrated physical
  conversion, pinhole/distorted/refractive camera rigs, robust calibration and
  triangulation, conflict-free multi-view particle reconstruction, streaming and
  globally refined PTV tracks, frozen-association smoothing, radiometric
  particle-image formation, residual-image Lagrangian refinement, deterministic
  synthetic qualification, optional learned dense displacement, canonical
  archives, and explicit-loss ecosystem adapters.
- Added fixed-population compatible particle-in-cell dynamics over stable charged
  particle supports: measure-aware endpoint charge, physical cochain E/B gather,
  matrix-free compatible electrostatics, relativistic Boris stepping, periodic
  cubical-Whitney trajectory current with discrete-continuity evidence, and
  transactional coupling to the existing compatible Maxwell runtime.
- Added constant-density fixed-population free-surface FLIP over prepared
  particle splats and MAC grids: cell and staggered-face mass/momentum transfer,
  runtime atmospheric pressure projection, bounded velocity extrapolation,
  explicit PIC/FLIP grid-delta blending, problem compilation, fixed-step
  rollback, and complete transfer/projection/energy evidence.
- Added `phydrax.circuit`: block-valued typed wave ports, dense and matrix-free
  hierarchical scattering, grounded dense/sparse MNA, causal implicit element laws,
  native DAE/DC/continuation/descriptor analysis, rational macromodels, periodic
  analysis, correlated noise, metrology/de-embedding, field/electrothermal coupling,
  SPICE and restricted behavioral interchange, certified learned dissipative laws,
  Touchstone I/O, and thin native optimization/UQ adapters.
- Generalized compatible time-domain Maxwell to explicit full-3D, TEz, and TMz
  cochain roles; added resource-preflighted final-state runs, sparse magnetic
  constraint projection with proved elision, boundary-packed CPML, prepared paired
  electric/magnetic sources and mode ports, harmonic-defect evidence, scalar
  geometry material assembly, and independent case batching.
- Added sparse metric-aware conic density filtering with explicit fixed-region
  semantics, finite-beta tanh projection, differentiable composed transforms, and
  separate forward-only hard thresholding.
- Added explicit cosmological length/mass/time scales, parameter-differentiable flat
  FLRW backgrounds, native first/second Lagrangian growth, immutable expansion/growth
  and linear-power products, state-ready 1LPT/2LPT, and transactional periodic
  scale-factor particle-mesh rollout. The cosmological path reuses the existing
  particle discretization, splat, self-gravity, and typed PM force evaluation;
  synchronized baryon/particle orchestration remains distinct from physical coupling.
- Added `phydrax.applications.astrodynamics`: explicit scale, two-part epoch, and
  frame contexts; Cartesian and modified-equinoctial states; bounded universal
  Kepler propagation with implicit JVP; fixed-capacity multi-revolution Lambert
  branches; pure force composition; adaptive and symplectic propagation; hybrid
  orbital events; provenance-bearing time/frame/ephemeris products; third-body and
  J2--J4 gravity; direct and nearly-Keplerian N-body dynamics; CR3BP; rigid
  spacecraft, finite-burn, reaction-wheel, and orbit-measurement contracts; and
  host-only coordinate, SPICE, and SGP4 adapters. No provider discovery, data
  download, close-encounter regularization, DSST, or adaptive IAS15 is implied.
- Added native astrophysical observation operators for observer projection,
  polynomial limb-darkened circular occultation, photon-counting bandpasses,
  transit count likelihood composition, binned and image responses,
  frequency-domain detector likelihoods, ordered ray transfer, and static complex
  field sequences. Contacts, event/branch selection, provider loading, and capacity
  changes remain explicit non-smooth boundaries.
- Added provenance-bearing CMB angular-power tables with explicit `Cl`/`Dl`
  conversion and fixed response-window Gaussian likelihood composition. Spectrum
  generation and experiment data remain external.
- Consolidated kinetic multiphysics around one thermodynamic closure for energy,
  variational derivative, symmetric stress, and explicit force representation.
  Added auditable kinetic field/stage manifests, exact portable checkpoints,
  production prepared sharding, signed-distance geometry-to-link compilation,
  parabolic and Womersley targets, collision-aware ratio-two AMR transfer with
  half-time interface data, and graduated scientific qualification evidence.
- Added enhanced conforming scalar virtual elements of qualified degree one
  through three on arbitrary-arity polygonal cell blocks, including certified
  H1/L2 projectors, explicit stabilization, functional trace constraints,
  matrix-free and sparse execution, fixed-topology geometry differentiation,
  projected reconstruction, mass-matrix DAEs, and generalized eigenproblems.
- Added static three-dimensional fixed, ball, and hinge rigid-body graphs with
  globally coupled mass-metric SO(3) pose projection, full velocity KKT projection,
  implicit root derivatives, physical position/velocity residual certification,
  multiplier warm starts, and fail-closed candidate/accepted transitions. Contact,
  friction, compliance, motors, dynamic topology, two-dimensional joints, and PBD
  compatibility remain outside this contract.
- Extended constrained mechanics with native planar fixed/ball joints, dimension-aware
  prismatic and distance joints, canonical stable row/coordinate layouts, physical
  compliant/dissipative laws, bounded effort motors and servos, unilateral joint
  limits, hard velocity restitution, exact planar/spatial Coulomb-cone impulses,
  irreversible joint breakage, and fixed-capacity topology transactions. Added
  transactional implicit Newmark volumetric FEM, mixed pressure gauges,
  rigid--deformable attachment KKT operators, objective two-/three-dimensional
  Cosserat rods and triangular membrane/bending shells. Replaced the partial
  particle-local deformable-contact routes and 2-D penalty workflow with
  exact-map collision surfaces, dense/sweep-and-prune candidate epochs,
  area-weighted physical barrier contact, conservative inclusion CCD, T3/T4
  inversion limits, static and transactional Newmark solves, lagged Coulomb
  friction, fixed-route sensitivities, and direct rod/shell collision-surface
  adapters. Explicit rigid--MPM weld/penalty/impulse coupling retains separate
  branch, rank, energy, route, and rollback certificates.
- Extended the contact substrate with an ordered guarantee lattice,
  roundoff-directed certified swept-AABB CCD, cached and fully compiled
  fixed-shape candidate filters, per-vertex separation, nonlinear/independent
  participant kinematics, rigid/articulated/point/MPM adapters, high-order
  proxy error inflation, implicit geometry, cubic and rigid sweep trajectories,
  closed-surface geometric-contact filters, deterministic triangle-overlap
  mortar quadrature, equal-pressure tetrahedral patch extraction, distributed
  route ownership/halo exchange, and remeshing state transfer. Added composable
  material-pair closure with barrier/geometric/compliant/adhesive normal laws,
  static/dynamic, anisotropic, and rate-state friction, irreversible
  wear/cohesive evolution, smooth force assembly, hard Coulomb-cone impact,
  projected, SAP, semismooth, and primal-dual cone solvers, rolling/spinning
  resistance, mortar, one-sided/unbiased Nitsche, mesh tying,
  cross-discretization coupling, hydroelastic patches, periodic/homogenized
  rough contact, thermal/electrical/mass flux, lubrication, contact-graph
  preconditioning, and fixed-branch closure/cone/mortar derivatives.
- Added epochal particle-capacity growth with stable structured interaction
  identities, transactional state migration, fixed-pool insertion and fragmentation
  retries, segmented replay, and transition pullbacks. Added multidimensional
  body-frame particle interiors, conservative unstructured transport, local
  coarse/fine AMR, boundary-face exchange, and native sparse implicit conversion.
  Added feature-certified superquadric triangle-wall contact with canonical shared
  feature ownership, wall histories, reactions, wear observables, and explicit
  feature curvature. Added matrix-free monolithic fluid-particle Newton coupling
  with momentum, heat, species, reaction, contact/radiative sources, route
  certificates, block preconditioning, atomic rollback, and implicit sensitivity.
- Added native atomistic dynamics with complete unit identities,
  position-independent prepared systems, stable-ID molecular topology and pair
  exceptions, composable classical/learned scalar-energy programs, dense and
  triclinic cell/Verlet execution, momentum-form NVE and BAOAB NVT,
  SHAKE/RATTLE constraints, stress, direct Ewald and B-spline PME, isotropic
  NPT moves, bounded replayable trajectories, exact checkpoints, hybrid and
  RESPA composition, Born–Oppenheimer provider boundaries, ring polymers with
  PILE, and variance-constrained semi-grand transitions. Dense graph resources
  are now explicit execution-plan identity rather than learned architecture identity.
- Extended the atomistic runtime with interaction-site coordinate maps and virtual-site
  force pullback; native force-field bundles, terms, policies, SETTLE, and OpenMM/OpenFF/
  ParmEd adapters; typed frames, H5MD/XYZ reporting, rerun, MDAnalysis, i-PI, and PACKMOL
  boundaries; collective variables, static/adaptive biases, replica exchange, FEP/TI/BAR/
  MBAR; committee uncertainty and deterministic acquisition; advanced thermostats,
  anisotropic pressure control, rigid and Brownian dynamics; polarization, multipoles,
  implicit solvent, advanced quantum-nuclear estimators; walls, manifold constraints,
  active/DPD and EAM/SW/Tersoff models; and distributed atomistic execution.
- Added fixed-capacity explicit Material Point Method dynamics for plane-strain and
  three-dimensional Neo-Hookean solids: nodal quadratic B-splines, matched APIC
  transfer, first-Piola reference-volume forces, transactional USL updates,
  support-halo and prescribed-velocity boundaries, acoustic/advective/force step
  evidence, full/step/block replay, final/checkpoint/trajectory retention, and
  piecewise-versus-frozen gradient reports. Corrected logarithmic Neo-Hookean
  parameter naming so its volumetric coefficient is Lamé lambda, with an explicit
  physical shear/bulk constructor.
- Added bounded and periodic unit-density MAC incompressible dynamics with static
  no-slip wall closure, face-dual velocity coordinates, symmetry-preserving momentum
  transport, conservative explicit viscosity, transform-or-iterative stage
  projection, fixed-step SSPRK composition, short-horizon differentiation, step
  restrictions, and complete constraint and kinetic-energy diagnostics. Hardened
  singular transform solves so masked pressure nullspaces retain finite reverse-mode
  derivatives.
- Extended the MAC flow substrate with dynamic no-slip/free-slip/inflow/pressure/open
  boundary closures, named conservative scalars and Boussinesq exchange, iterative,
  transform, hybrid-line and IMEX/SBDF2 viscous solves, conservative variable-density
  face momentum, dual-measure resolved IB–DEM coupling, transactional adaptivity and
  replay, short-horizon and least-squares-shadowing sensitivities, explicit sharded
  pressure CG, compatible mapped/ALE geometry, and conservative nondifferentiable
  remesh epochs. Every path exposes its mass, momentum, energy, residual, topology,
  differentiation, resource, and fail-closed acceptance evidence.
- Added exact fixed-temporal finite-volume replay with full, step, or block
  rematerialization; transactional balance-law source composition and persistence;
  periodic Newtonian and particle-mesh gravity; replayable Hermitian spectral
  Ornstein--Uhlenbeck forcing; implicitly differentiated tabulated cooling;
  conservative trainable face closures; and periodic Cartesian constrained MHD with
  integrated cochain face fluxes, edge circulations, coupled stage positivity, and
  HLLD-to-HLL fallback evidence.
- Added bounded adaptive balance-law realization with process-aware step limits,
  transactional retry rollback, fixed-capacity decision journals, and exact scheduled
  replay of accepted temporal meshes. Added global OU realizations whose innovations
  obey the OU semigroup under interval subdivision, including antithetic coupling.
- Unified ordinary finite-volume and constrained-MHD source composition behind one
  prepared balance-law transport contract. Gravity, cooling, and OU forcing now compose
  with face-flux MHD under the same adaptive realization, scheduled replay, rollback,
  component-ownership checks, and portable checkpoint semantics.
- Added dimension-generic constrained-MHD layouts, primitive PLM/WENO/TENO/MP5
  reconstruction, HLL-UCT, accepted face/edge integral ledgers, physical boundary
  policies, dual-energy and CTU support, non-ideal and AMR cochain operators, bounded
  gravity, exact cooling coordinates, modal forcing, thermochemistry, radiation
  moments, cosmological workflows, field inference, and structure-preserving closures.
- Added fixed-rank randomized Nyström preconditioning with auditable sketch and
  refresh evidence; Diffrax-backed neural Galerkin evolution over fixed physical
  field metrics with rectangular or Gram tangent solves and saved-node audits;
  backward Diffrax characteristic tracing with macro-step neural projection; and
  mass-preserving fixed-support residual-attention collocation with explicit ESS,
  KFAC, and controlled-policy contracts.
- Added native athermal lattice-Boltzmann flow on uniform isotropic cell grids:
  certified D2Q9/D3Q19 velocity sets, BGK/TRT collision with collision-coupled
  Guo forcing, periodic and frozen halfway-wall link routing, fixed tangential
  moving walls, explicit physical/lattice scaling and precision evidence,
  fail-closed fixed-step integration, differentiable runtime controls, and a
  memory-bounded generic fixed-step rollout with final/checkpoint/trajectory
  retention.
- Expanded kinetic methods with D3Q27, prepared moment bases and advanced collision
  families, staged open/curved/moving-wall ownership, explicit local implicit forcing,
  geometry epochs and conservative transfers, multiblock and ratio-2 refinement
  contracts, color-gradient/free-energy/thermal/species/reactive distributions,
  certified D2V17 and off-lattice D2V37 smooth-compressible methods, fixed FV/kinetic
  interfaces, sharded and AA/fused execution, block reverse replay, and a forward-only
  stable-tuple IREE export contract. Advanced paths report capability, conservation,
  realizability, equivalence, and qualification evidence without extending the
  qualified low-Mach baseline by implication.
- Added reciprocal-lattice harmonic discretization with true one-dimensional and
  oblique two-dimensional periodicity, selected FFT analysis/synthesis,
  pairwise-difference material convolution, translation covariance, resource
  preflight, and Gamma-containing Brillouin-zone rules.
- Added `phydrax.solver.maxwell.fourier_modal`: full-tensor periodic finite layers,
  homogeneous ports, differentiable boundary-field cascade propagation, direct,
  inverse, and local-frame Fourier factorization, a nondifferentiable modal reference
  backend, stable scattering composition, named electric/magnetic current planes,
  multi-RHS and Brillouin source semantics, interior field reconstruction,
  diffraction-order far fields, explicit refresh, convergence, resource, diagnostic,
  status, and provenance contracts.
- Extended native low-rank adaptation with rank-stabilized scaling, exact
  adapter-artifact reconstruction, and composition with frozen random-weight
  factorization coordinates.
- Added field-certificate-aware geometry-to-material rasterization for
  Fourier-modal Maxwell, with sharp and differentiable compact-Heaviside paths,
  fixed subpixel sampling, fill-fraction evidence, and material identities.
- Added certified finite-box Method of Moving Asymptotes, constrained
  reduced-adjoint state/design optimization, sparse physical-radius density
  filtering, SIMP compliance topology optimization, and independent
  reference-discretization reanalysis.
- Added native force-density structural design for tension, compression, and
  mixed-sign pin-jointed systems with sparse coordinate or orthonormal affine
  restraints, reciprocal GraphIR conversion, stable external IDs, prepared
  linear/nonlinear refresh, weighted-Laplacian Newton preconditioning, fixed,
  line, self-weight, traction, follower-pressure, and pneumatic load laws with a
  component ledger, mathematical solution derivatives, reduced and structured
  force/support/load design, pure geometry/force observables, same-topology
  batches, per-graph evidence, mechanism/self-stress spectra, supplied-rigidity
  tangent stability, and continuation bridges.
- Added member-network constitutive verification over force-density topology:
  stress-free reference states, exact tension-only cable active sets,
  corotational frame and discrete-rod bending, surface hinges, local and global
  buckling, nonlinear continuation bridges, prestress fabrication/actuation
  evidence, staged construction replay, continuous and catalog sizing, and
  explicit certified/failed/incomplete structural verdicts.
- Added advanced structural evidence: generalized coordinate channels, explicit
  section-orientation fields, semirigid connections and nonlinear supports,
  extensible catenaries, cable/saddle contact, warping beam and bracing energy,
  fiber-section plasticity transactions, imperfections, collapse and dynamics,
  thin-walled GBT/finite-strip/shell-submodel evidence, exact precedence
  branch-and-bound, standards clauses, reliability, calibration, evidence
  acquisition, and immutable structural-twin snapshots.
- Added pickle-free StableHLO/IREE inference export with matched optional
  compiler/runtime versions, in-process compilation and loading, exact
  shape/dtype ABI checks, checksummed manifests, and native parity evidence.
- Added full-rank Euclidean VP/VE score diffusion with structured diagonal Gaussian
  laws, exact perturbation marginals, weighted denoising score matching, replayable
  reverse-time SDE sampling, probability-flow composition, per-realization Diffrax
  initial states, and memory-linear diagonal Wiener coefficients. Replaced the
  flow-specific `FlowMatchingPolicy` with shared `UniformTimeSamplingPolicy`.
- Extended generative transport with stable array/PyTree/complex event coordinates,
  block-operator Wiener noise, full and Hausdorff Gaussian factor laws,
  matrix/state-dependent Itô reversal, exactness-labeled guidance, discrete Gaussian
  and categorical diffusion, coefficient-space field/path diffusion, intrinsic
  manifold and complex diffusion, latent/graph/atomistic compositions, persistent
  energy training, normalized autoregressive laws, and sample-only adversarial
  objectives. Every family retains explicit measure, geometry, approximation, and
  density capabilities rather than sharing a universal model facade.
- Generalized matrix-free quantum local actions through
  `AbstractLocalQuantumOperator` and evidence-rich `LocalOperatorEstimate`, with
  a clean migration of discrete VMC/TDVP while preserving the connected-action
  algorithm. Added finite nonperiodic Born--Oppenheimer
  `ElectronicCoulombHamiltonian`, validated Bohr/Hartree reference conversion,
  exact and chunked-exact coordinate kinetic traces, singularity statuses without
  distance clipping, replayable electronic walkers, an exactly corrected
  state-dependent proposal, and a full-generalized-determinant
  `phydrax.nn.quantum.FermiNet` with same-spin antisymmetry, sparsity-aware
  scaled log envelopes, higher-order-correct zero/subnormal signed products,
  polynomial singular-term determinant derivatives, coefficient-aware
  nonzero-product mixture shifts with coefficient- and singularity-reactivation
  fallbacks, a positive physical decay floor, and determinant mixtures
  differentiable at zero coefficients, under an explicit four-electron ceiling.
  Electronic VMC
  folds local statuses into validity and reuses persistent chains, matrix-free
  score/Gram stochastic reconfiguration, training lifecycle, linear solves,
  diagnostics, statuses, and checkpoints. Added H/He/H₂ tests,
  documentation, and a fixed multi-seed benchmark campaign with predeclared
  statistical/chemical gates and provenance; periodic, relativistic, and
  stochastic-trace electrons remain unsupported.
- Added exact scalar temporal Matérn-3/2 and Matérn-5/2 Gaussian processes
  through content-addressed continuous state-space compilation, origin-shifted
  stable irregular train/query schedules, exact missing/query masks, bounded
  stationary long-gap discretization, sequential square-root filtering and
  reverse-scan RTS smoothing,
  dense-parity parameter gradients, active-observation marginal likelihoods,
  linear-storage predictive marginals, explicit compute precision,
  prepared/evaluated identity and failure provenance, portable result export,
  and complete retained-storage scaling benchmarks.
- Added integration-native fixed-design Bayesian quadrature for normalized scalar
  Gaussian targets with analytic squared-exponential kernel means, optional
  kernel scaling, content-bound Gaussian targets, separate observation noise and
  solve regularization, true evaluation-stage dtype placement, scale-normalized
  prepared `phydrax.linalg` conditioning, reusable PyTree/field reductions,
  overflow-stable analytic means, posterior-SD diagnostics, dtype-aware variance
  validity, explicit target/contraction/solve/resource failure boundaries, and
  an analytic Gaussian benchmark against IID and
  randomized QMC. The posterior SD is explicitly model uncertainty, not a
  deterministic or frequentist error bound.
- Added `phydrax.atomistic` and `phydrax.nn.atomistic.PaiNNPotential` for finite
  nonperiodic molecular research: scale-identified atomic structures and padded
  batches reuse material-particle identities and `GraphIR`; resource-guarded
  case-isolated dense neighborhoods fail closed without truncation; invariant
  energies yield conservative forces with typed status, diagnostics, precision,
  and provenance; energy-only, force-only, and joint training retain fitted
  training-only normalization, selection, restart, and complete histories; and
  local-NPZ rMD17 parsing/splitting plus a fingerprinted multi-seed benchmark
  tool require explicit data provenance. Periodic execution, stress, long-range
  electrostatics, direct-force heads, ASE integration, and molecular-dynamics
  stability claims remain outside this capability.
- Added labeled nonintrusive polynomial chaos for independent scalar Uniform and
  Normal inputs: preflight-guarded graded total-degree multiindices and
  sample-by-feature projection storage, stable normalized Legendre/Hermite tensor
  bases, content-addressed measure-honoring product-integration projection, diagnosed
  exact/least-squares regression with complete solver-policy identity, immutable
  array/Field/PyTree expansions, coefficient moments and first/total Sobol effects,
  portable fit evidence, and a matched-model-call benchmark campaign.
- Added resource-planned `O3TensorProductPlan`/`O3TensorProduct` layers and a
  drop-in `phydrax.nn.atomistic.NequIPPotential`. Independently derived
  Cartesian Clebsch–Gordan maps cover legal scalar/pseudoscalar,
  vector/pseudovector, and symmetric-traceless tensor/pseudotensor paths through
  degree two with per-instruction radial weights, masked finite-molecule
  aggregation, species-conditioned self connections, parity-safe gates, and the
  existing conservative prediction/training contracts. The rMD17 campaign now
  records matched PaiNN-versus-NequIP seeds, errors, equivariance defects,
  timing, memory, parameters, neighborhood work, gates, summaries, and
  provenance. High-degree irreps, MACE/symmetric contraction, periodic systems,
  stress, long range, and molecular-dynamics claims remain out of scope.
- Added experimental two- and three-dimensional soft-sphere DEM with rigid
  translational/angular state, collision-free stable pair keys, persistent
  Cundall--Strack contact history, linear spring--dashpot and Hertz--Mindlin
  contact families, exact-signed-distance barriers, dense/cell-list execution,
  structured fail-closed fixed stepping, contact qualification, an executable
  settling example, and dense/cell performance evidence.
- Added source-resolved accepted-step DEM energy/work ledgers, explicit rejection
  reasons, qualification artifacts, certified Verlet caching, fused/reference
  pair reductions, radius-class filtering, replay/checkpointed VJPs, sharp and
  smooth sensitivity contracts, inverse/UQ qualification, and transverse
  hybrid-event saltation.
- Added compositional normal/cohesion/tangential/rotational DEM contact history,
  elastic rolling–torsional resistance, finite-range DMT cohesion, conservative
  capillary bridge lifecycle, near-contact lubrication, bilinear elasto-plastic
  normal response, elastic half-space multicontact correction, and conservative
  contact heat exchange.
- Added prescribed force/torque servo barriers, certified analytic contact
  curvature, curved Hertz walls, facet traction/work/heat observables, Finnie
  wear accumulation and geometry commits, SO(2)/SO(3) rigid bodies, immutable
  sphere-clump templates, triangle walls, elastic/damageable bond graphs,
  fixed-pool topology events, convex SAT contact, certified sphere-to-implicit
  contact, and support-map superquadric contact and dynamics.
- Added conservative slab/cylindrical/spherical internal shell meshes, typed
  species/phase/element thermochemistry, polynomial heat-capacity inversion,
  heat/species transport, stoichiometric Arrhenius networks, evaporation,
  shrinking-core conversion, reference Rosenbrock and structured tridiagonal
  solvers, morphology, fragmentation, radiation, and process operations.
- Added conservative particle-grid transfer, unresolved Stokes CFD–DEM,
  work-adjoint resolved immersed-boundary coupling, reactive continuum heat and
  species exchange, atomic Strang/iterated reactive CFD–DEM windows, generic
  hybrid-event sensitivity, replay/checkpointed VJPs, UQ, compositional support
  claims, executable examples, qualification campaigns, and performance evidence.
- Generalized fixed-step problems and solutions to mixed-dtype array PyTrees
  while preserving the existing array-valued SSPRK contract.
- Added all-coordinate tensor spectral PDE residual compilation with explicit
  full-closure versus retained-projection semantics, polynomial closure
  dealiasing, exactness and resource evidence, physical quadrature norms,
  external hard-condition contracts, and targetless operator fitting through
  `SpectralPDEResidualLoss`.
- Added native high-order quadrilateral/hexahedral finite elements with explicit
  nodal representations, GLL reference actions, dense and sum-factorized
  workset kernels, mapped cell/facet metrics, high-order CG/DG routing,
  mass-policy-aware rates, tensor SBP and periodic mapped DGSEM conservation,
  p-multigrid, Schwarz/FDM and auxiliary preconditioning, two-sided mortars,
  fixed-capacity hp transactions, and backend-neutral distributed ownership.
- Added operational nonconforming tensor-hp epochs with stable refinement forests,
  isotropic quad/hex h-refinement and coarsening, 2:1 closure, anisotropic p
  buckets, curved parent-map inheritance, H1 master-trace constraints, asymmetric
  DG mortar worksets, role-correct h/p transfers, atomic solver transactions,
  adaptive indicators and budgets, hp condensation/multigrid, inherited
  partition ownership, and certified entropy-compatible DGSEM mortars.
- Completed the single-host spectral-hp stack with native epoch compilation,
  anisotropic h and geometry-order adaptation, robust viscous/shock/ALE policies,
  tensor de Rham complexes, simplex/prism/pyramid references, nonlinear hp
  solvers, CAD/unfitted geometry, frozen-schedule adjoints, semantic caches,
  high-order output/import adapters, and complete public examples and guidance.
- Added implicit tensor-modal neural fields with Hermitian real-field projection,
  explicit modal input scaling and resource bounds, optional positive exponential
  decay and prepared-basis modulation, masked modal observations, and direct
  residual training against compiled coefficient-resident spectral dynamics.
- Added native low-rank adaptation for exact real `Linear` weight paths,
  factor-only `ParameterSubspace` training through `fit_operator` and Optax
  `FunctionalSolver`, safe scan fallback, explicit KFAC rejection, pure dense
  deployment merging, and checksum-validated adapter artifacts bound to the
  complete base model content and structure.
- Added deterministic fixed-step learned discrete systems with lazy
  mask/reset/control-aware trajectory windows, supervised, reference-branch,
  and residual rollout objectives, evidence-weighted gradient accumulation,
  exact update-boundary resume, and full/prefix/chunk causal equivalence.
- Added task-bound recurrent neural-operator training with one pipeline-safe
  physical state route, named future targets, supervised and residual rollout
  losses, route-aware deployed continuation, and instance-authoritative
  pointwise/finite/global/unknown dependency-support evidence.
- Added advanced computational topology: exact cellular and filtered maps,
  induced maps and cone audits, extended and temporal field topology, diagram
  features and certified matching, rational and integral class algebra,
  harmonic-period constraints, exact Morse cancellation, structured cubical
  analysis, local homology, certified implicit covers, and Conley homology
  index-pair workflows.
- Added `phydrax.pgm`: immutable finite-discrete factor graphs over native
  bipartite `GraphIR` topology; dense, sparse-enumerated, structured, and open
  capability-declared kernels; explicit precision/resource evidence; directed linear
  forest propagation; synchronous, Gauss--Seidel, accelerated, and qualified implicit
  loopy BP; same-topology and heterogeneous graph batches; bounded variable
  elimination, junction trees, normalized laws, smooth dual MAP bounds, and
  perturb-and-MAP estimates; systematic/random/block/tempered/qualified-cluster
  sampling with online reducers; persistent CD/SML, pseudolikelihood, Bethe, and exact
  EM objectives; and pickle-free graph/BP/Gibbs checkpoints. Added general PyTree
  conditional update programs under `phydrax.sampling.conditional` and factor-graph
  reverse denoising, adaptive mixing control, and hybrid embeddings under
  `phydrax.transport.discrete`.
- Added `phydrax.topology`: compact active subcomplexes and relative pairs,
  exact prime-field homology with cycle/cocycle representatives, exact rational
  Betti dimensions, explicit cell-vertex supports, lower/upper-star filtrations,
  ordinary and induced-relative persistent homology, natural and fixed-capacity
  diagrams, frozen-order endpoint derivatives, fail-closed resource evidence,
  and exact-nullity validation of metric cochain harmonic kernels.
- Added explicit bounded, periodic, half-line, and real-line spectral domains;
  endpoint-correct tensor measures; canonical modal transfers; rational
  Chebyshev line and half-line bases; linear trace constraints; exact periodic
  Hilbert transforms; physical modal-tail diagnostics; homogeneous
  cross-resolution eigen and eigenspace evidence; pairing-aware resolvent scans;
  and original-residual-certified polynomial eigenproblems.
- Added a native linear-combinatorial substrate with separate logical decisions
  and objective features, content-addressed plans, deterministic ties, portable
  statuses, and independent certificates. Added exact streamed finite,
  fixed-cardinality, primal-dual Hungarian assignment, and signed-cost DAG path
  oracles, plus explicit one-extra-solve blackbox surrogate pullbacks.
- Added a method-neutral structured nonlinear optimization spine with
  topology/numeric prepare-refresh lifecycles, exact sparse Jacobian and
  Lagrangian-Hessian reuse, portable primal/dual warm starts, independently
  certified native and Ipopt results, a unified dense/matrix-free/sparse
  `PrimalDualInteriorPoint`, truthful provider-backed KKT preparation, fixed-width
  root and structured-solve pools, generic PyTree/state-design/multiple-shooting
  compilers, fixed-active sensitivities and continuation, and optional
  Spineax/cuDSS sparse LDLT with numerical refactorization, reported inertia,
  and explicit resource release.
- Added an end-to-end free-boundary SciML substrate: differentiable compact
  Heaviside/delta and coarea calculus; level-set phase, normal, curvature,
  velocity, and Eikonal operators; discontinuity-aware coordinate lifts;
  implicit phase/interface functionals; Stefan, jump, kinematic,
  Gibbs--Thomson, Young--Laplace, and traction conditions; causal time slabs
  and narrow-band adaptive collocation.
- Added explicit-front, implicit-level-set, reference-map, and relaxed
  probabilistic Stefan workflows with common collocation, optimization, and
  representation comparison. Added free-boundary operator contracts,
  Jacobian/pullback/GCL evidence, VOF/PLIC and SPH adapters, and
  residual-controlled hybrid rollouts.
- Added interface predictive uncertainty, residual/diversity acquisition,
  bounded test-time context adaptation, phase geometry and masked interface
  distance evidence, plus exact Stefan, Mullins--Sekerka, topology-event,
  Hysing bubble, Turek--Hron FSI, obstacle, fracture, and
  trajectory-disjoint OOD benchmark contracts.
- Added certified variational eigenspaces across continuous, learned-operator,
  factorized high-dimensional, and discrete quantum workflows: basis-invariant
  block Rayleigh objectives, native reduced Ritz extraction and full-space
  residuals, learned trial-space warm starts, product-factor bilinear assembly
  without global tensor materialization, and mixture-sampled multi-state VMC
  with overlap/Hamiltonian evidence, span conditioning, Ritz modes, stochastic
  reconfiguration, and explicit failure statuses.
  Added a self-adjoint strong-form `InvariantSubspaceResidual` for neural trial
  fields, with projected reduced operators, basis-invariant residual Grams,
  generalized positive metrics, complex/vector pairings, continuous residual
  modes, absolute/relative residual evidence, and explicit rejection of
  collapsed, indefinite, or non-Hermitian trial systems.
- Added JAX-native direct collocation for explicit controlled systems and
  input-aware state-shaped DAEs, with fixed/nonuniform or optimized-duration
  meshes, backward-Euler and midpoint transcription, interval controls, shared
  optimized parameter spaces, bound-form path and trajectory constraints,
  physical scaling, exact sparse Jacobians and optional Lagrangian Hessians,
  explicitly selected native-dense or low-level sparse-Ipopt execution,
  independent KKT recertification, typed decisions/layouts/results, and
  non-certifying off-grid defect audits.
- Hardened direct collocation with canonical typed sparse-Ipopt evidence,
  callback/conversion counts, exact status mapping, topology-valid warm starts,
  optional `cyipopt` packaging, exact/limited-memory qualification artifacts,
  per-interval off-grid evidence, explicit nested h-refinement and primal
  transfer, controlled-DAE input policies and causal replay, and a fingerprinted
  eight-family graduation/regression campaign.
- Added a first-class material-particle discretization with stable physical IDs,
  static active selections, physical mass measures, explicit precision/execution
  policies, periodic pair geometry, budgeted canonical dense neighborhoods,
  fail-closed fixed-capacity cell-list neighborhoods, exact solver-relation
  `GraphIR` views, and equal/opposite pair accumulation. Added normalized
  Wendland C2 and cubic-spline SPH kernels, a Tait barotropic energy closure,
  conservative fixed-h summation-density SPH, complete conservation and step
  diagnostics, and native separable-Hamiltonian compilation through
  `StormerVerlet`.
- Added first-order weakly compressible SPH with explicit summation- and
  continuity-density state layouts, pair-once continuity and conservative
  pressure operators, symmetric Morris physical viscosity, external
  acceleration power accounting, SSPRK33/SSPRK54 lowering, complete
  energy/dissipation diagnostics, periodic shear qualification, and dense versus
  cell-list scaling evidence.
- Added particle assemblies and bipartite relations, fixed-step programs and
  accepted-step transforms, geometry-derived wall particles, free-surface
  detection and atmospheric pressure policies, first-order kernel correction,
  delta-SPH diffusion, Monaghan artificial viscosity, Shepard density
  renormalization, transport velocity, adaptive smoothing length and grad-h,
  reciprocal multiphase WCSPH, and fail-closed IISPH/DFSPH projection steps with
  explicit residual and work evidence.
- Added explicit experimental/qualified/production/certified particle-method
  maturity, evidence-backed claims, dimensionless original density/divergence
  residuals, pressure and boundary constraint diagnostics, and a production gate
  that separates finite execution from numerical qualification.
- Added native multi-population cell execution, specialized batched 1D--3D
  local solves, safeguarded adaptive-h roots, production boundary and interface
  geometry evidence, stateful stabilization controls, assembled projection
  oracles and complementarity diagnostics, reference halo/migration semantics,
  benchmark registries, qualification artifacts, support matrices, and replay
  packets for commercial particle-method hardening.
- Added experimental measure-aware particle-grid splatting with multilinear
  and degree-one through degree-three tensor B-spline assignments over nodal,
  cell, face, and edge layouts; extensive content and density outputs; weighted
  intensive reconstruction; adjoint grid gather; route gradients and moments;
  periodic and explicit reject/drop boundaries; piecewise or frozen geometry
  differentiation; static resource budgets; precision evidence; and
  fast/deterministic/compensated accumulation with independent balance,
  partition, reproduction, and gradient diagnostics.
- Made the finite-element local-action/workset program authoritative for scalar
  and product-space residuals; replaced split mixed subproblems with one compiled
  problem; added executable SIPG/Nitsche/nullspace handling, conservative upwind
  facets, Darcy, Maxwell, generated lowest-order primal HDG, and Taylor-Hood
  forms; connected high-order cell-local simplex/tensor and hexahedron Q1
  execution; and bound smoothed-elasticity actions to the same form program.
  Added accepted field/material/topology transactions, deterministic local T3
  marking/refinement/coarsening and transfer data, local DWR indicators, and
  executable phase-field, CPFEM, persistent-pair contact, fracture, and fixed-
  crack XFEM application workflows. Distributed execution remains out of scope.
- Added accepted linear-solve histories, exact FEM diagonal data, pairing-aware
  p-transfer roles and p-level planning, collocated tensor-product actions,
  staged DG traces, explicit quadrature evidence, one-ring patch and low-order
  auxiliary preconditioners, an incompressible pressure-correction workflow, and
  conservative multirate DG traces. The execution and preconditioning design is
  informed by libParanumal and its published high-order solver algorithms while
  remaining Phydrax-native.
- Added first-class exact-sampling round-sphere spectral discretizations with
  explicit S2FFT mode layouts, physical area measures, matrix-free
  Laplace--Beltrami actions, complete-degree real eigenbases and spatial noise,
  radius-aware addition-theorem kernels, resource/precision provenance, and one
  shared prepared-space contract for SFNO.
- Added native computation-aware scalar Gaussian processes with fixed, normalized
  block-sparse, and pseudo-input action policies; bounded sparse kernel-action
  contraction; reusable projected factors and low-storage conditioners; diagonal
  prediction; a full-data ELBO; numerical/resource diagnostics; and exact,
  conservative-covariance, differentiation, and scaling qualification gates.
- Added JAX-native static nested slice sampling over `PosteriorProblem` with
  weighted posterior quadrature, stochastic evidence-shrinkage uncertainty,
  insertion-rank and constrained-kernel diagnostics, semantic replay keys,
  portable checkpoints/results, and predictive integration.
- Added exact mathematical complex parameter interchange for dense and low-rank
  holomorphic layers, HMLPs, polynomial and constrained frame coefficients,
  meromorphic coefficients, and trainable pole locations while retaining real
  Cartesian trainable leaves, destination dtype/sharding, and affine membership
  evidence.
- Generalized representation-preserving holomorphic constraints around reusable
  real-coordinate frames, target-independent functional operators, batched
  minimum-norm lifts, coupled outputs, and target-specific affine maps. Added
  exact nonlinear cardinal projection plus named Goursat and plane-elasticity
  boundary functionals with explicit gauge and nullspace evidence.
- Added query-holomorphic DeepONet trunks with fixed or source-dependent hard
  targets, analytic conditional jets, and real harmonic operator adapters.
- Added exact finite Fourier circle traces, explicit contour/period functionals,
  fixed-pole meromorphic frames with domain-clearance evidence, and reduced
  variable-projection fitting for trainable pole locations.
- Added several-complex-variable multi-indices, analytic multijets for
  polynomial frames, holomorphic MLPs, and product potentials, pluriharmonic
  real-field wrappers, and metric-invariant holomorphic Kähler gauges.
- Replaced the scalar triangular P1 vertical slice with a shared computational
  cell mesh and generic fitted finite-element substrate: triangle P1/P2,
  quadrilateral Q1, tetrahedron P1, global DOF maps, fixed-topology geometry
  differentiation, dual-space weak residuals, affine Dirichlet constraints,
  compensated functionals, sparse affine lowering, and native linear,
  nonlinear, and DAE adapters.
- Completed the fitted finite-element runtime contracts with dynamic geometry
  and coefficient refresh, component and mixed `BlockSpace` fields, selected
  cell/exterior/interior domains with native rules, user residual/energy/
  bilinear/facet kernels, execution and accumulation policies, sparse lifecycle,
  lagged and adjoint operator factories, nullspace-aware linear solves, dynamic
  DAE/second-order/eigen adapters, curved coordinate elements, P0/RT0/Nedelec0,
  local/HDG condensation, material transactions and checkpoints, hierarchy/error
  evidence, embedded/enriched bases, partitioned DOFs, halo semantics, and FE IO.
- Added shared twofold compensated conservation accounting across structured,
  triangle, unstructured, spectral, and SBP conservation diagnostics, plus
  finite-volume ledgers, remap, overset/sliding, multiblock, small-cell, and
  diffusion certificates. Spectral diagnostics now honor their declared reduction
  dtype before accumulation.
- Reworked generic continuation around exact terminal coordinates, complete
  `phydrax.nonlinear` correctors, prepared Newton/trust-region reuse, explicit
  state/residual geometry, canonical real-coordinate maps for complex, algebra, and
  constrained spectral states, tangent predictors, full bordered tangents, curvature
  rejection, full-augmented event localization, execution-coordinate stability, and
  geometry-aware bifurcation evidence and normal forms.
- Added independently selectable tangent and adjoint linear policies for exact
  implicit root derivatives, plus prepared lagged-linear nonlinear updates that
  refresh structure-preserving operators, retain complete failure evidence, and
  certify every accepted root against the original physical residual.
- Added coefficient-resident global Fourier, sine, cosine, Chebyshev, Legendre,
  constrained, and mixed tensor spectral spaces; explicit padding/filter dealiasing;
  modal PDE and periodic conservation lowering; entropy diagnostics;
  conjugacy-preserving modal spatial noise; internal-linalg Galerkin, boundary-lift,
  and generalized tau formulations; and diagonal ETDRK2/4 integration with shared
  stable phi-three matrix actions.
- Added dealiased periodic incompressible Fourier dynamics, Hermitian real analysis
  coordinates, tensor spectral symmetry actions, primitive Fourier--Chebyshev--Fourier
  channel Stokes solves with pressure-gradient or bulk-flux control, fixed-step
  channel SBDF2 integration, shared-runtime periodic/Floquet analysis, relative
  invariant residuals, bounded evolution observation, portable spectral state
  artifacts, recurrence seeding, and finite-horizon edge tracking.
- Hardened incompressible spectral workflows around one shared flow problem,
  semidiscrete energy-balance diagnostics, constraint-valid channel initialization,
  fail-closed SBDF2 histories, and reproducible qualification artifacts. Added
  structured periodic compact first/second derivatives and staggered interpolation
  without dense line solves; periodic diagonal-norm SBP flux differencing with a
  reusable symmetric entropy-conservative Euler volume flux; and a geometry/solver
  split for structured MAC pressure projection with exact transform and refreshed
  variable-coefficient linalg routes.
- Added explicit real/imaginary Diffrax state packing for complex ODE, split, CDE,
  and stochastic paths. Public states remain complex; temporal evidence records the
  doubled real backend shape, dtype, tolerance geometry, and policy, while native and
  reject strategies remain explicit.
- Added exact finite real algebra specifications for real, complex, quaternion,
  octonion, Cayley--Dickson, and multicomplex families; three-valued law evidence,
  resource-bounded sparse/dense products, shared real-coordinate maps, algebra-valued
  spaces and operators, Diffrax algebra state policies, and unit complex/quaternion
  state geometries.
- Added exact and numerical commutator, Jordan-product, and associator operations;
  explicit left/right regular-action operators; resource-bounded algebra derivation
  spaces; and an octonion-derived G2 bridge with local metric, torsion, Ricci, and
  infinitesimal invariance diagnostics.
- Added canonical-complex holomorphic construction dependencies, spectrally
  initialized low-rank complex-affine layers, per-layer factorized
  `HolomorphicMLP` plans, certified independent branch bundles, same-coordinate
  holomorphic product potentials with exact Taylor-convolution jets, multiplicative
  gauge diagnostics, and a deterministic separability benchmark.
- Expanded compatible electromagnetics around conservative electric-displacement and
  magnetic-flux cochains: canonical structured/unstructured calculus, diagonal and
  metric-Hermitian constitutive maps, conductivity, Lorentz/Drude ADEs, Kerr/Pockels,
  gyrotropy, active/saturable gain, PEC/PMC/impedance/interface policies, periodic and
  Bloch calculus, electromagnetic CPML, probes/DFT/energy/Poynting observers, modal and
  near-to-far outputs, time/frequency/reversible adjoints, tetrahedral Whitney Hodge
  assembly, distribution metadata, and complete energy/constraint/CFL evidence.
- Added certified point-cloud differential functionals and dissipative Poisson
  execution, sixth/eighth-order smooth-exact TENO, explicit stabilization filters,
  cochain multirate scheduling, a backend-neutral lowered operator program with JAX
  and NumPy parity, neutral data interchange schemas, and runtime-integration
  guardrails for external research packages.
- Added public callable adaptive interval and triangle engines that reuse
  `AdaptiveQuadraturePlan`, `AdaptiveTrianglePlan`, `IntegrationPrecisionPolicy`,
  `IntegrationEstimate`, bounded partitions, statuses, and error-kind diagnostics.
  Specialized evaluators can now keep singularity classification and correction
  orchestration separate without duplicating the adaptive refinement subsystem.
- Expanded boundary layers with an explicit representation/discretization/evaluator
  split, global-error-aware adaptive near/self panel evaluation, declared corner
  topology and Kress/dyadic partitions, outgoing 2D Helmholtz kernels and explicit
  Brakhage--Werner CFIE assembly reports, target-associated 2D local expansions,
  triangular 3D surface panels, and a corrected near/far direct backend contract.
- Added target-centered Duffy self integration for 3D surface layers, coefficient-
  quadrature 3D QBX with continuous signed-distance clearance, outgoing 3D
  Helmholtz fields, and explicit Duffy-based 3D CFIE assembly reports. The
  reference near/far backend remains explicitly direct; no FMM claim is attached.
- Split the analytic sphere boundary atlas into two trimmed reference triangles,
  preserving full-sphere measure under triangular 3D panel quadrature and sampling.
- Added a fixed-topology Laplace multipole treecode reference with truncation
  estimates and direct/multipole work accounting; the production FMM path and
  global QBX coupling are recorded separately below.
- Added genuine 2D Laplace M2M/M2L/L2L translations and global QBX/FMM coupling
  with prepared target associations, continuous expansion clearance, separate FMM
  truncation, coefficient-quadrature, and local expansion error evidence.
- Added a metric-dependent Clifford algebra substrate with canonical blade layouts,
  sparse/dense prepared geometric, exterior, and contraction products, involutions,
  resource evidence, differential-form bridges, finite and standalone metric
  isometries, outermorphism actions, and exhaustive algebra-automorphism audits.
- Added flat constant-metric Dirac operators and exact-rational monogenic polynomial
  Trefftz fields with analytic partial derivatives, algebraic trial certificates,
  boundary-only linear fitting, and independent Dirac residual audits.
- Added complete-grade Clifford neural representations, grade-wise equivariant
  linear and geometric-product layers, Euclidean invariant gating, operator field
  schemas, sampled equivariance evidence, and non-promoted differential-context
  benchmark scenarios for incompressible flow, entropy-aware Euler, and Maxwell
  fields.
- Added likelihood-backed binary and multiclass empirical classification terms
  with encoded target schemas, case masks, positive statistical sample weights,
  posterior-compatible raw log probabilities, classification diagnostics, and a
  gathered categorical hard-label kernel that avoids one-hot target allocation.
- Expanded classification with posterior-compatible independent multilabel and
  fixed-threshold ordinal likelihoods; soft-target and focal objectives; explicit
  target-event masks; differentiable sigmoid/softmax/expectation field transforms;
  Dice, Jaccard, and Tversky overlap scores; and dense-grid, regular/irregular
  trajectory, graph-entity, and neural-operator classification. Structured terms
  retain geometry masks and physical measures, operator schemas use canonical
  JSON-safe ordered class names, and zero-weight objectives bypass evaluation.

- Added explicit convex entropy pairs with Euler mathematical-entropy factories,
  entropy-variable and flux compatibility validation, relative-entropy diagnostics,
  and volume-weighted structured/mapped finite-volume entropy evidence. Compiler
  integration rejects viscous, triangle, and modern unstructured pair diagnostics
  until those contributions have separate certified contracts.
- Promoted `h5py>=3.16.0` to a core dependency so finite-volume checkpoint and
  restart persistence is available in the default installation.
- Added domain-aware Legendre geometry with explicit primal/dual supports,
  conjugate and Fenchel--Young operations, representative validation, and direct
  dual translations. Added fixed-step mirror descent over mixed trainable
  PyTrees, including FunctionalSolver diagnostics and documented
  exponential-family KL and simplex exponentiated-gradient identities.
- Added a sequential Gaussian-process MAP initializer for expensive bounded posterior
  objectives. The UQ-owned search normalizes unconstrained positions, treats surrogate
  noise in raw negative-log-density units, records complete evaluated-point and
  fallback evidence, composes with state-space global/local MAP and Laplace workflows,
  and preserves the existing differential-evolution result contract.

- Added regular, first-order conic primal JVP/VJP operators over audited
  `ConicProgram` executions, with cached dense projection-KKT Jacobians,
  native-bound cotangents, projection/linear regularity evidence, and exact
  numeric binding. Added real scaled-triangle PSD, exponential, and standard
  power cones with JAX-native primal/dual projections and direct Clarabel
  mappings. Fixed versus interval bounds now participate in conic structure
  identity.
- Added certified finite Trefftz trial spaces for nD harmonic,
  polyharmonic-Almansi, and homogeneous Helmholtz fields, with deterministic
  exact-rational harmonic bases, resource preflight, sampled PDE audits,
  enforcement safety, provenance, and direct fixed-boundary least-squares
  fitting.

- Added real-parameter complex affine layers, polynomial and exponential
  holomorphic potentials, certified 2D harmonic, biharmonic, and plane-elastic
  representations, plus prepared 2D Laplace single/double boundary layers and
  an interior Dirichlet boundary-integral solve. Holomorphic coverage and parameter
  linearity propagate into physical certificates, distinguishing linear finite
  subspaces from nonlinear finite parametric families. Layer fields retain algebraic
  PDE exactness off singular support; continuous-boundary target admissibility is
  validated before residual audits, while target-clearance and panel/trace/BC
  approximation evidence remain separate.
- Added a single-device unstructured finite-volume stack over canonical cell complexes:
  triangle, quadrilateral, mixed polygonal, and affine tetrahedral geometry; stable
  topology/geometry/global identities; normal Rusanov-HLL-HLLC fluxes; general
  cell-polynomial and CWENO/WENO-Z reconstruction; explicit viscous triangle closure;
  shared SSPRK positivity/retry; matrix-free backward Euler; momentum-weighted
  Rhie--Chow pressure correction; schema-versioned mesh, case, checkpoint, HDF5/XDMF,
  and VTK persistence; fixed-connectivity GCL diagnostics and conservative remap;
  polygonal embedded-boundary clipping and PLIC/VOF transport; fixed-capacity two-level
  AMR; conservative overset interpolation; and accepted-step periodic sliding overlap.
  Tetrahedral reconstruction/dynamics qualification is degree-one affine only; degree-two
  k-exact and WENO qualification remains limited to the tested 2-D geometries.

- Added explicit stage flux-rate and accepted content-integral ledgers, conservative
  content state, epoch/event transactions, automatic certified AABB/polygon/tetra
  remap artifacts, embedded small-cell stabilization, two-material EOS/system
  foundations, capillarity/contact-angle evidence, and fail-closed rejection of
  unintegrated two-material PLIC runtime coupling.

- Added native-precision open-system campaign records, integrity-checked
  artifacts, semantic-variate replay evidence, fail-closed promotion policies,
  frozen campaign-matrix tooling, and cross-campaign graduation with permanent
  unsupported-claim provenance.

- Replaced Boolean archive qualification with exact campaign
  deserialization and reproduction, added derived physicality/capacity gates,
  adaptive preconditioned HEOM, eventful multi-event MPS jumps, analytic Padé
  residues, direct-memory Choi certification, active-memory refit, and separate
  neural projection-audit semantics.
  Campaign orchestration and graduation now live under developer tooling rather
  than the public solver API.

- Added connected VMC neural trajectory execution, seeded disjoint process
  tomography designs, count-aware held-out refit evidence, and pre-fit/post-fit
  recovery gates for sequential-process and causal-memory campaigns.

- Added fail-closed open-system evidence with quantified approximation
  thresholds, exact pseudomode initial states, process initial-state
  physicality, semantic trajectory keys, fixed-step probability guards,
  Gaussian convention checks, versioned artifact tooling, generic jump-solver
  adapters, certified local Kraus preparation, BDF HEOM diagnostics, and causal
  process simulation.

- Hardened open-system workflows with shared scalar-root event refinement,
  semantic trajectory checkpoints, environment MPS contractions,
  nonnormalizing TEBD, LPDO Strang evolution, scaled/implicit HEOM, matched
  spin-boson evidence, causal process tomography, and neural no-jump TDVP.

- Added event-driven quantum jumps, MPS canonicalization and TEBD, MPS jump
  trajectories, locally purified Kraus evolution, HEOM continuation,
  non-Markovian cross-representation diagnostics, adaptive Fock continuation,
  fermionic Gaussian dynamics, process-comb causality, and neural jump
  projection.

- General nonlinear root families with certified scalar bracketing,
  safeguarded Newton/Halley, chord and limited-memory Broyden, DF-SANE,
  pseudo-transient continuation, vector Halley, capability-selected and robust
  attempt graphs, Type-I/II Anderson, Steffensen acceleration, exact nested work
  budgets, scaling, mixed precision, batched small-system kernels, explicit
  sharding semantics, and first/second-order solution maps.
- Block residual/factor graphs with robust losses, manifold parameter blocks,
  route and Schur planning, traditional/subspace dogleg, dogbox,
  trust-reflective bounds, variable projection, POUNDERS, and incremental
  add/remove/relinearization evidence; plus BOBYQA, COBYQA, deterministic
  multistart, and independently recertified SciPy, NLopt, Ipopt, and Ceres
  boundaries.
- Scaled constrained models, SQP BFGS/SR1/exact Hessian choices, a native filter
  interior-point method with restoration, KKT inertia/null/range planning, fixed
  active-set and barrier sensitivities, frozen peer/corpus manifests,
  cross-family performance-profile campaigns, and solver graduation/regression
  gates.
- Authoritative physical optimization certificates now demote false-success
  POUNDERS and interior-point exits; condensed interior-point KKT systems reuse
  one factorization across predictor/corrector right-hand sides. Root
  polyalgorithms reuse residual and prepared Newton evidence across attempts.
  Residual-graph plans now execute dense, LSMR, or Schur routes with block-local
  robust curvature and explicit clipping evidence. Nonlinear comparisons keep
  backend claims separate from mathematics, enforce frozen runner identity and
  initial fingerprints, record cold/warm/steady phases in flat JSON, and form
  family-compatible performance profiles.
- Added Gaussian bosonic Lindblad dynamics, quantum-jump ensembles, adaptive
  bosonic Fock spaces, pseudomode/reaction-coordinate embeddings, HEOM,
  memory-kernel and TCL evolution, tensor-network states and truncation
  evidence, and process-tensor MPO contracts.
- Unified nonlinear and optimization model/direction/certificate precision,
  routed dense root, interpolation, Schur, KKT, and sensitivity systems through
  `phydrax.linalg`, and retained nested execution evidence. Added temporal,
  integration, geometry, and Hermitian precision to Gaussian, trajectory, HEOM,
  memory-kernel, Fock, and process-tensor paths, plus an explicit
  `TensorNetworkPrecisionPolicy` for storage, contraction, factorization,
  accumulation, certification, and output roles.

- Added projective-line Calabi–Yau campaign preparation, residue and induced
  hypersurface geometry, positivity-globalized Kähler-potential solving,
  Hermitian spectral/Sylvester infrastructure, faithful Bures density geometry,
  SLD quantum Fisher actions, mixed-state tomography, fixed-rank/Uhlmann
  primitives, and finite-dimensional Lindblad channel evolution.

- Estimator-aware `RandomizedMomentPenalty` with U-statistic,
  independent-product, and explicit plug-in modes; deterministic causal
  convolution and Caputo field-operator provenance; integral/nonlocal physics
  guidance; and an accuracy, bias, and performance benchmark campaign.
- Added end-to-end weighted geometric diffusion semantics, right-trivialized
  unitary propagation, abelian metric-DEC gauge fields, matrix-free Fisher and
  Hessian operators, geodesic manifold flow matching, explicit atlas covers and
  patch integration, CP^n Fubini–Study references, Dolbeault/Chern/Berry
  calculus, projective hypersurfaces, Kähler-potential Monge–Ampère operators,
  and Ricci-flat Kähler optimization composition.
- Expanded `phydrax.metrix` with immersion validation and Riemannian map
  geometry; correct tensor-density covariant derivatives; weighted metric
  measures and intrinsic hypersurface normals; exact and numerical endpoint
  geodesics with Fréchet statistics and transport/flow-matching adapters;
  complex-projective, unitary, special-unitary, and Hermitian-positive-definite
  manifolds; real-coordinate almost-complex, Hermitian, Kähler, atlas, and local
  SU(n) diagnostics; Hessian and exponential-family information geometry;
  vector-bundle gauge curvature; metric cochain Hodge assembly; anisotropic
  horizontal cometrics; and fixed-step Störmer–Verlet integration.

- Capability-checked temporal integration with complete Diffrax configuration
  provenance; additive KenCarp/Sil3 IMEX; native SSPRK3/SSPRK54, endpoint theta,
  variable-step BDF1--BDF5, matrix-free RA34PW2 Rosenbrock-W, generalized-alpha,
  fixed-ratio partitioned RK2/RK3, and one- through three-stage Gauss--Legendre
  implicit RK with collocation dense output.
- `phydrax.transport.continuous` endpoint couplings, linear probability
  interpolants, status-preserving continuous sampling, exact Euclidean continuous-flow
  densities, and uncertainty-bearing Hutchinson density estimates; plus
  `FlowMatchingTerm` and fixed-query quadrature-aware operator velocity metrics.

- Prepared finite nonlinear updates with typed application status, hard work
  controls, refreshable plans, additive/multiplicative/residual-optimal
  composition, Armijo Richardson and typed NGMRES outer methods, FAS/Picard/Newton
  updates, nonlinear Schwarz/Gauss--Seidel decomposition, and ASPIN with
  independently certified physical roots.
- Strict box-preserving semismooth variational inequalities with prepared
  topology-preserving refresh; matrix-free Steihaug--Toint quadratic trust
  regions; large-scale unconstrained and bounded Newton trust-region methods;
  and bound-aware Gauss--Newton and Levenberg--Marquardt residual optimization.
- Positive certified Xiao--Gimbutas, Lebedev, periodic, radial, and Duffy
  cubature with content identity and bounded storage; measure-matched fixed
  Gauss--Hermite expectations; geometry-owned native disk, circle, ball,
  sphere, and triangle-surface maps; mixed product plans; and static-capacity
  differentiable adaptive triangle refinement with explicit paired-rule error,
  partition, evaluation-budget, and terminal-status evidence.
- Positive total-degree standard-normal cubature through degree five, including
  grouped probability-product lowering; static-capacity Markov cubature for weak
  Itô and Stratonovich law propagation; positive polynomial recombination with
  frozen-support continuous derivatives; signature-certified piecewise-linear
  Wiener controls; weighted-measure result interoperability; explicit resource,
  moment, rank, positivity, and terminal-status diagnostics; and a compiled
  accuracy/performance benchmark harness.
- Cross-domain executable precision contracts with strict content-addressed
  request, resolution, nested evidence, and resource-assumption records.
  Finite differences separate coefficient, field, accumulation, certification,
  communication, checkpoint, AMR, distributed-halo, multigrid, and adjoint
  placement. Structured finite volume separates state, reconstruction, flux,
  conservative reduction/decision, output, and checkpoint precision across
  dynamics, SSPRK runtime, AMR, rollouts, HDF5, and restart. Integration
  separates evaluation, accumulation, decision, and output precision across
  fixed, mapped, adaptive, stochastic, product, weighted, MLMC, atlas,
  Riemannian, and projective execution. Neural operators retain master
  parameters, transient compute views, scoped matmul/FFT precision, dynamic
  loss-scale state, and persisted effective evidence. Spatial noise, SPDE
  composition, predictive summaries, bootstrap particles, native
  GMRES/FGMRES basis storage, Jacobi preconditioning, nonlinear Newton and
  globalization decisions, native/SSP temporal integration, flow-matching and
  manifold reductions, Hermitian spectra/Sylvester solves, quantum tomography,
  Calabi--Yau campaigns, randomized estimator objectives, and experimental
  standard-Optax `FunctionalSolver` contractions expose matching policies and
  evidence. Information geometry composes geometry precision with the linear
  runtime instead of raw dense solves. Precision-sensitive persistence formats
  reject incompatible contracts and retain effective evidence. The precision
  benchmark reports accuracy, storage, runtime, and evidence across FD, finite
  volume, integration, linear, nonlinear, temporal, geometry, Hermitian,
  operator, and UQ domains.
- `phydrax.discretization`: canonical entity/topology/support, finite-measure,
  DOF/field-space, plan/preparation, transfer, bundle, hierarchy, temporal, tensor,
  spectral, metric-cochain, conforming P1 finite-element, and conservative
  first-order finite-volume contracts. Strong, variational, and conservation
  compilers now retain complete discretization provenance; adaptive DAE results add
  their realized accepted-step mesh.
- Shared normalized proposals and fixed-kernel persistent Metropolis--Hastings chains
  with semantic key addressing, exact asymmetric Hastings correction, chain-preserving
  transition evidence, target refresh after parameter changes, and direct lowering to
  correlated `WeightedSampleTarget` measures that never claim IID uncertainty.
- Pairing-aware `EmpiricalGramLinearOperator` geometry with normalized nonnegative
  weights, sample centering, masking through zero weights, damping, rank/ESS evidence,
  complex adjoint/transpose actions, existing linear-runtime interoperability, and a
  single numerical implementation reused by UQ empirical Fisher actions.
- Discrete variational Monte Carlo with stable log-magnitude/unit-phase amplitudes,
  explicit real/holomorphic/nonholomorphic parameter modes, fixed-capacity connected
  operators, matrix-free local energies, persistent walkers, centered score geometry,
  damped SR updates, frozen-model final evaluation, complete status histories,
  documentation, tests, and compile/steady-state/storage benchmarks. JaQMC, jQMC, and
  Quantax are acknowledged as design references; this implementation is independent
  and Phydrax-native.
- Frozen-chain VMC diagnostics now report rank-normalized R-hat and bulk/tail ESS;
  pickle-free checkpoints preserve exact model/walker/key continuation; finite signed
  permutation groups provide validated symmetry-sector amplitude projection; and
  fixed-step real/imaginary-time TDVP reuses the same persistent sampling and
  pairing-aware score geometry. The scale benchmark now profiles sampler, connected
  local-energy, geometry, solve, storage, and end-to-end costs for periodic 8/12/16-site
  transverse-field Ising chains. Masked nonfinite feature and connection payloads are
  selected to safe values before Gram or local-energy multiplication.

- Industrial structured finite differences: exact point/interval entity layouts;
  masked variable-width Fornberg banks with consistency, adjoint, conservation, and
  stability evidence; manufactured convergence studies; stage-cached dynamic
  Dirichlet/Neumann/Robin and conforming interface programs; conservative scalar,
  diagonal, and tensor diffusion/advection lowering; diagonal-norm SBP orders
  2/4/6/8 with compatible second derivatives and SAT coupling; discrete-curl mapped
  geometry; oriented conforming and 2:1 multiblock mortars; geometric multigrid with
  Jacobi, red-black, and line smoothers; compact interior kernels, fused CSE pipelines,
  and multidimensional collective halos; WENO-Z/TENO/MP5, characteristic and
  multispecies Euler, ideal MHD, unsplit multidimensional fluxes, positivity, and
  entropy policies; entity-aware AMR halo/transfer/subcycling/reflux/regrid/migration;
  portable checkpoints, exact discrete adjoints, resource/precision preflight,
  structured cochains, and compatible Maxwell, MHD induction, elasticity,
  variable-density projection, poroelasticity, and thermoelasticity. Certified
  FFT/DCT/DST direct solves and directional split-field acoustic PML remain integrated
  with the same provenance.
- Structured finite volume now binds cell-average and directional face spaces directly
  to tensor support; supports uniform/nonuniform Cartesian and stationary mapped
  geometry, typed physical boundaries, piecewise-constant/MUSCL/WENO-Z/TENO/MP5 and
  characteristic reconstruction, Rusanov/HLL/HLLC/Roe and entropy fluxes, normal and
  transverse wave propagation, hydrostatic wet/dry shallow water, multidimensional
  split/unsplit execution, Euler/multispecies/MHD systems, positivity and
  differentiability policies, conservative diffusion and compressible viscous fluxes,
  MAC pressure projection, matrix-free linearization, conforming/nested multiblock
  fluxes, and fixed-capacity AMR synchronization with integrated reflux.
- Structured finite-volume runtime hardening adds immutable ideal/stiffened-gas
  materials and constant/Sutherland/Prandtl transport closures; material-owned viscous
  and mapped-viscous fluxes; slip, no-slip thermal, supersonic, characteristic, and
  far-field boundaries; one prepared halo authority; Einfeldt-HLL fallback blending;
  bounded SSPRK retry/status runtime; versioned case, precision, checksum checkpoint,
  optional HDF5/XDMF output, differentiable scan/rematerialization rollout, quantitative
  verification contracts and CLI, and NamedSharding decomposition with scaling
  benchmarks.



- `phydrax.weighting` exact and quadratically reconciled relative-entropy moment
  calibration for dense, sparse, and matrix-free feature actions, with affine-rank
  reduction, audited convergence/regularity evidence, warm starts, and implicit
  target/prior derivatives. The mathematical formulation follows Barratt, Angeris,
  and Boyd's *Optimal Representative Sample Weighting*; the public `cvxgrp/rsw`
  and Apache-2.0 `andytimm/rswjax` packages are acknowledged as design
  inspiration, while this implementation is independent and Phydrax-native.
- Finite-measure calibration in `phydrax.integration`, shared calibration/coreset
  lowering, and ordered transformation diagnostics that preserve physical mass,
  masks, named axes, ancestry, support validity, execution keys, and provenance
  while invalidating inapplicable inherited integration-error bounds.
- Dense/sparse exact/soft calibration benchmarks with separate setup, first
  compilation, steady-state, cold-nearby, and warm-nearby timing and numerical
  evidence.
- Newton--Krylov now passes its adaptive forcing tolerance into each inner linear
  solve, so forcing policy changes actual Krylov work under eager and compiled
  execution.

- Learned function frames with masked, quadrature- and channel-metric-aware
  projection, explicit rank and residual evidence, reusable source encodings,
  arbitrary-query reconstruction, frozen inference, portable artifacts, and a
  research-tier operator benchmark composition.
- Scalar Tikhonov damping for dense SVD least-squares solves and a direct
  weighted least-squares path that reuses one factorization for coefficients,
  rank diagnostics, residuals, and differentiation.
- Native SING natural-gradient variational smoothing for additive-noise latent
  SDEs, with Gaussian information-chain algebra, deterministic and fixed-sample
  Gaussian expectations, irregular masked schedules, per-case backtracking,
  coherent posterior paths, fixed-posterior ELBO gradients, portable archives,
  diagnostics, tests, documentation, benchmarks, and explicit numerical status.
- Certified causal nonlinear recurrence with associative temporal linear solves,
  exact implicit adjoints, dense and quasi-Newton linearizations, and ELK-style
  Levenberg--Marquardt damping; plus opt-in recurrent-layer and fixed-trajectory HMC
  consumers with explicit convergence and fallback diagnostics.
- Normalized mean-field and FlowJAX reverse-KL variational inference with deterministic
  checkpoint replay, full-path Gaussian Markov state-space VI, reusable amortized
  encoders, and inverse-inclusion-weighted buffered target windows.
- Normalized latent-path and parameterized state-space density contracts that preserve
  the existing prior, transition, observation, schedule, mask, physical-time, and
  exogenous-input model hierarchy.
- JAX-compiled bootstrap particle filtering, retained initial genealogy, complete-model
  `O(TN)` genealogical scores, replaceable SG-MCMC gradient estimators, and
  complete-sequence particle-driven SGLD/SGNHT.
- `phydrax.nonlinear` contracts for algebraic systems, Newton line-search and
  trust-region roots, nonlinear GMRES and preconditioning, fixed-point acceleration,
  full-approximation multigrid cycles, variational inequalities, semismooth
  complementarity solves, and implicit root derivatives with explicit failure status.
- Native regular index-one differential-algebraic systems with explicit structural
  roles and scales, consistent initialization contracts, prepared fixed/adaptive
  BDF1--BDF5 integration, guarded cross-step numerical reuse, segmented continuation,
  local regularity evidence, frozen accepted-grid JVP/VJP replay with bounded
  checkpoint memory, status-rich trajectory evidence, semidiscrete implicit PDE
  compilation, and canonical identification adapters.
- Reusable nonlinear Newton preparation/refresh/solve lifecycles, adaptive
  Eisenstat--Walker forcing, explicit Jacobian refresh policies, hard aggregate
  inner-linear work budgets, and physical-root certification for transformed
  nonlinear preconditioners.
- Dynamic per-invocation controls for prepared native Krylov solves, plus
  capability-checked mixed-precision dense LU with pre-factorization condition
  screening, high-precision residual certification, iterative refinement, and
  requested/effective precision evidence.
- Matrix-free low-rank ADI lifecycles for factored continuous Lyapunov equations,
  including fixed-capacity factors, rank/truncation evidence, per-shift convergence,
  exact low-rank residual certification, numeric refresh, and factor-versus-dense
  storage accounting.
- Typed nonlinear optimization with matrix-free Newton--Krylov and trust regions,
  nonlinear conjugate gradients and strong-Wolfe search, Gauss--Newton,
  Levenberg--Marquardt, deterministic finite-difference least squares, proximal
  gradient/Newton methods and built-in functionals, filter/SOC SQP, primal--dual
  predictor--corrector KKT solves, state/design adjoints, stochastic risks and
  decomposition, explicit Optimistix interoperation, and `FunctionalSolver`
  integration.
- Canonical linear and quadratic programs with native variable bounds, typed solve and
  differentiation policies, reusable plan/prepare/bind/refresh lifecycles, explicit
  warm starts, independently audited KKT and infeasibility/recession certificates,
  portable status, and complete numeric provenance.
- Public zero, nonnegative, second-order, rotated second-order, and product-cone
  programs, with optional MPAX 0.2.4 LP/QP and Clarabel 0.11.1 conic execution behind
  lazy provider lifecycles and original-coordinate audits.
- Dense/structural-sparse linear-control compilation, reusable numeric refresh,
  explicit receding-horizon warm-start shifting, and affine stage/terminal SOCP
  constraints.
- Reproducible LP/QP/SOCP advanced-solver cases and independent certificates, plus
  control-horizon campaigns for sparse storage and cold-versus-warm MPC evidence.
- `phydrax.continuation` contracts for generic parameterized residual curves,
  arbitrary physical-parameter PyTrees, natural and pseudo-arclength
  predictor/corrector methods, reusable bordered solves, adaptive rejection,
  event localization, dense and Krylov stability evidence, explicit branch switching,
  fold/Hopf/pitchfork extended systems and certificates, normal forms, homotopies, and
  metric-aware root deflation.
- Standard and generalized nonsymmetric eigenproblem lifecycles with dense Schur/QZ
  and native restarted-Arnoldi/Krylov--Schur methods, standard, shift-invert, and
  Cayley transforms, homogeneous finite/infinite classification, paired left/right
  eigenvectors, residual and conditioning evidence, resource accounting, and
  isolated-eigenvalue derivatives.
- Lazy optional PETSc KSP/SNES, SLEPc EPS, PyAMGCL, and NVIDIA AmgX backends with
  dependency-free package import, explicit capability probes and lifecycle plans,
  sparse or matrix-free execution contracts, independently verified residual/status
  evidence, numeric refresh, transfer accounting, and explicit GPU/collective release.
- Canonical sparse triangular analysis and numeric solve, provider-neutral sparse
  factorization plans, incomplete Cholesky/ILU/ILUT preconditioner builders, and
  explicit sparse-provider availability/capability inspection.
- Array-backed finite axes and lazy Cartesian products, exact streaming exhaustive reduction, deterministic finite MAP screening, exact finite control-catalog search, portable MAP candidate archives, and a dense-oracle benchmark harness; independently implemented with [Brutax](https://github.com/michael-0brien/brutax) acknowledged as design inspiration.
- Typed preconditioner builders, prepared actions, planning costs, refresh provenance, and materialization-aware resource rejection.
- Additive and multiplicative subspace correction, Chebyshev smoothing, block factorization, and immutable multigrid hierarchy preparation.
- Native fixed-capacity BlockGMRES and BlockCG with explicit right-hand-side layouts, shared-subspace diagnostics, and block-aware differentiation.
- Immutable GCRO-DR recycling state with explicit reuse and rebuild policy.
- Standard and generalized Hermitian eigensolve plans with LOBPCG and thick-restarted Lanczos, residual/status diagnostics, refresh, and isolated-eigenvalue differentiation.
- Exact Galerkin and smoothed-aggregation hierarchy builders, deterministic diagnostics, transfer reuse, and optional PyAMG conversion.
- Exact diagonal and uniform local-block assembly, local block operators and factorizations, block-Jacobi preparation, and native Kronecker-sum direct solves.
- Policy-bounded canonical sparse assembly plans with reusable prepare/refresh recipes for sparse algebraic operator graphs.
- Deduplicated resident operator-state and per-right-hand-side action-workspace estimates, propagated into solve candidate costs.
- Symbolic `LinearSolveTemplate` planning with separate numeric binding, scoped kernel and spectral-interval certificates, and quotient-space `ProjectedPCG`.
- Bounded dense standard/generalized Hermitian eigensolves and pairing-aware singular-value decompositions with plan/prepare/refresh lifecycles and restricted scalar differentiation.
- Arbitrary-base `BasePlusLowRankLinearOperator` Woodbury solves with reusable base state, correction conditioning, resources, status, and provenance.
- Two-sided equilibration, iterative refinement, and resilient solve lifecycles that verify residuals in the original coordinates.
- Explicit structure compilation for diagonal, permutation, tridiagonal, triangular, banded, DCT-diagonal, and FFT-diagonal operators, including refresh-safe transform-diagonal operators.
- Numerically bound reusable Arnoldi/Lanczos projections, shared-basis shifted solve families, and partial-fraction rational matrix-function actions.
- Adaptive fixed-capacity stochastic trace and log-determinant estimation with separate statistical and projection-error evidence.
- General dense real/complex Schur eigensolves, nonnormal spectral observables, Riesz spectral subspaces, and first-order projector derivatives.
- Generalized, Sylvester, and continuous/discrete Lyapunov matrix-equation lifecycles built on the shared linear runtime.
- An advanced JSON benchmark harness covering Krylov reuse, shifted/rational actions, matrix equations, spectral projectors, low-rank updates, resilience, and adaptive spectral estimation.
- A schema-validated advanced-solver benchmark package with deterministic problem
  generators, independent original and refreshed certificates, explicit setup/
  compilation/preparation/solve/differentiation/refresh/verification and transfer
  accounting, reproducible JSON comparison, and lazy Phydrax, JAX, Lineax, Optimistix,
  SciPy, PyAMG, AMGCL, PETSc, and SLEPc adapters.
- Reusable full self-adjoint spectra with exact cluster-safe projector,
  density-kernel, and Loewner spectral-function derivatives for standard and
  generalized real or complex problems.
- Native batched dense Hermitian eigensolves and batched self-adjoint spectral
  calculus with per-member diagnostics, status, provenance, and batch-scaled
  resource accounting.

### Changed
- The pinned ty and complete-annotation gate cover every first-party Python file
  in `phydrax`, tests, tools, examples, benchmarks, and `mkdocstrings_setup.py`.
  Selector auditing resolves imported `Literal` aliases, rejects discarded
  selector parses, and reaches zero duplicated validation tables or dispatches.
  Typing-introduced internal narrowing checks remain active under optimized Python
  and raise explicit internal-invariant errors instead of relying on `assert`.
- Randomized PDE compilation validates `loss_mode` against the shared
  `RandomizedResidualLossMode` alias; uncertainty-source lists in predictive fields,
  variance decomposition orders, and process retention are parsed against
  `UncertaintySource` (the `UNCERTAINTY_SOURCES` tuple is removed); and
  `phydrax.domain` exports `RaggedSeriesWindowSampling`.
- The supported precision dtype names (`RealPrecisionDType`,
  `ComplexPrecisionDType`, `ScalarPrecisionDType`, `precision_dtype_name`,
  `real_precision_dtype_name`, `complex_precision_dtype`) have one dependency-free
  owner that also classifies dtypes by exact dtype or category; `phydrax.precision`
  exports them unchanged. Precision dtype arguments are typed as `DTypeLike`.
- Static checkers now type every `StrictModule` construction through the concrete
  class's generated or custom `__init__`. The runtime metaclass `__call__`, which
  still performs the abstract/final refusal and the freeze transition, is hidden
  from checkers because its `-> Any` signature erased every constructor call.
  Concrete modules that inherit an abstract owner's custom constructor declare a
  checker-only `__init__` alias, matching Equinox's runtime constructor choice.
- Typing is gated by the pinned ty `0.0.84` through `tools/check_typing.py`
  (`report`, `check`). ty analyzes Python 3.12 semantics, ignores the dataclass
  field-order rule that does not apply to Equinox constructors, types optional
  providers as `Any` at their import boundary, and honors only `ty: ignore[rule]`
  suppressions. The package is ty-clean, and every function, method, and nested
  helper annotates all parameters and its return type, as enforced by the pinned
  Ruff `0.16.9` `ANN` rules; `check` fails on any diagnostic. Non-ty checker
  comments (`type: ignore` and similar) were removed and are rejected.
- Package annotations were corrected throughout: constructor inputs that are
  converted with `jnp.asarray` or `numpy.asarray` accept array-like values, fields
  and parameters formerly typed `object` carry their real types, bare `Callable`
  and untruthful `Any` were replaced by parameterized callables, protocols, and
  type variables, closed selectors use their `Literal` aliases, and loop carries
  are typed. Runtime behavior is unchanged except for the fixes listed below.
- Removed additional scaling bottlenecks across bounded BVH traversal,
  filtered execution worksets, sparse conic control and particle transport,
  block-local contact, matrix-free DFN Newton updates, reduced-space
  functional RG, bounded tree split evaluation, direct 3-D panel influence,
  streamed hydroelastic transfer, cached belief-propagation cavities,
  multi-RHS electromagnetic solves, and prepared atomistic neighbor slots.
- Replaced the under-specified generic astrophysics frequency-response and
  detector-network helpers with the canonical gravitational-wave data,
  response, waveform, and likelihood contracts; Welch PSD estimation now
  supports Tukey windows and bias-corrected median averaging.
- Reworked performance-critical numerical paths around stable compiled entry
  points, homogeneous vectorization, bounded BVH and local-stencil routing,
  neighborhood-backed many-body potentials, genuinely sparse control programs,
  batched particle and inference solves, device recurrences, direct panel and
  Choi assembly, reusable tensor environments, and linear-memory Hawkes
  likelihood evaluation.
- Routed contracted coordinate derivatives through exact JVP actions, reused
  prepared primal/JVP/VJP linearizations, preserved matrix-free hydrodynamic
  and manifold solves through `phydrax.linalg`, and moved package contractions
  and runtime interpolation behind their native substrate boundaries.
- Added supplied exact, Gauss--Newton, and approximate Hessian actions to
  `MinimizationProblem`, and centralized Gaussian moment conditioning and
  weighted effective-sample-size reductions.
- Restored implicit scalar and least-squares differentiation after iteration
  runtimes began returning lifecycle evidence alongside their accepted run.
- Replaced training and finite-search callbacks and fixed-step diagnostic callbacks
  with explicit iteration plans, sinks, and separate host control. Native Krylov,
  Newton, scalar optimization, fixed-step, Markov, Hamiltonian, continuation, and
  functional-decomposition execution now expose owner-typed iteration evidence;
  delegated backends report only faithfully available terminal or saved-output data.
- Compressible case identity now binds the exact physical system, including transport
  and auxiliary state. Equation-owned diffusion replaces the legacy `D + 2`
  material-only FV path, and viscous conservation diagnostics include total
  `inviscid - diffusive` boundary flux plus gradient-coupled sources.
- Reacting flow now uses `ThermochemistryProcessPlan` through
  `PreparedBalanceLawRuntime`. Removed the duplicate `ReactiveStrangPlan` and
  `ReactiveIMEXPlan` state machines and the incomplete algebraic SA/SST turbulence
  closures without compatibility aliases.
- Scientific candidate and analytical evidence now leaves campaign-start and
  campaign-observation bindings explicitly empty until approved criteria,
  resolved execution, and raw observations pass the shared causality validator.
- Removed the nonfunctional ROM `MultifidelityControlVariateProfile` and
  `MultifidelityMLMCProfile` declarations and their unused nested truth-sample
  fields. Control variates and MLMC now remain with integration, while ROMs enter
  fidelity workflows only through executable reduced evaluations.
- Multilevel Monte Carlo plans now distinguish a finest-level estimand from a
  continuum-limit estimand. Limit claims require either three refinement levels
  or an explicit terminal bias bound; result evidence records the chosen
  estimand.
- Reference manifests can retain unknown uncertainty as `None`; quantitative
  qualification consumers explicitly require known uncertainty instead of treating
  missing metadata as a zero-error reference.
- Fixed-support PGM preparation now supports runtime numeric factor tables for
  exact and implicit-BP inference without host value checks inside JIT.
- Physical scale, atomistic, electrophysiology, cardiovascular,
  skeletal-muscle, PDE, and neural-operator contracts now share canonical
  dimension and unit definitions.
  Unit-bearing content IDs and serialized payloads are intentionally replaced;
  ambiguous legacy float-dimension and ID-only unit artifacts are rejected.
- Periodic Fourier resampling is now owned by `phydrax.signal` with trailing-axis
  defaults and explicit spatial axes for channel-last neural operators. Removed
  the `_interpolation.fourier_resample` and
  `phydrax.nn.operator.architectures.spectral_resample` exposure paths without
  compatibility aliases.
- Replaced the independent multispecies and reacting-Euler thermodynamic
  conventions with `HomogeneousMixtureEulerSystem` and full chemical-energy
  conservation. Removed the legacy reacting-flow classes without aliases.
- Consolidated scalar absorption-emission transfer under `RayTransferPlan`,
  made polarized propagation valid for singular operators, and replaced the
  overclaimed gray flux-limited API with explicit linear diffusion using
  separate transport-extinction and absorption coefficients.
- Replaced the legacy isotropic plane-stress MPM, mixed volumetric-constraint,
  contact workflow, sharp-fracture workflow, and compliance-only topology APIs
  with their explicit clean-cutover contracts. No deprecated aliases remain.
- Neo-Hookean field stress operators now name Lamé's first parameter `lambda_`
  instead of incorrectly describing the same coefficient as bulk modulus
  `kappa`; the old keyword is removed in one clean cutover.
- Neural-operator autoregression now requires a task-bound physical state route
  and the deployed normalization/constraint pipeline. The raw callable/advance
  rollout, standalone autoregressive loss, and teacher-forcing schedule were
  removed in one clean cutover.
- CNO and UNO now have periodic-Fourier semantics: circular measure-aware local
  convolution, endpoint-exclusive sine/cosine coordinate features, periodic
  uniform axes, and new semantic architecture identities. Nonperiodic and
  legacy artifact routes are rejected rather than reinterpreted.
- Benchmark tooling now shares one synchronized PyTree timing runtime, normalized
  software/hardware fingerprints, raw duration distributions, official XLA
  cost/memory evidence, atomic artifact writes, and environment-checked bootstrap
  comparisons. Operator reports separate lowering, compilation, first execution,
  and steady samples and no longer relabel process allocator high-water state as
  operation-local peak memory.
- Compatible Maxwell state now stores electric displacement `D`, magnetic flux `B`,
  charge, material/boundary auxiliary state, and observer state. `E` and `H` are
  constitutive outputs. Construction uses `CompatibleMaxwellPlan(...).prepare()` and
  `PreparedCompatibleMaxwell`; the scalar-only direct `CompatibleMaxwellDynamics`
  path and E/B-primary state were removed.
- `MomentPenalty` now rejects resampled stochastic integration rather than
  silently optimizing a variance-biased squared estimate. `time_convolution`
  now accepts a deterministic `IntervalRule`; randomized QMC and importance
  modes were removed from the field-valued operator.
  Caputo field operators now use direct deterministic Gauss--Jacobi or
  Gauss--Legendre evaluation for both supported order intervals; stochastic
  sampler and endpoint-regularization arguments were removed.
- Finite-volume ownership is now structured and face-first. The triangular generic
  `FiniteVolumePlan`, system-specific reconstruction dynamics, and the
  `phydrax.discretization.reconstruction` owner were removed. Physical conservation
  systems now live in `phydrax.equations`, conservative face operators live in
  `phydrax.discretization.finite_volume`, time advancement lives in `phydrax.solver`,
  and `FDAMRSubcyclingPlan` is now `ConservativeAMRSubcyclingPlan`.

- Orthogonal-polynomial evaluation and Gaussian rule construction now pass through
  one private convention boundary. Hermite and Laguerre KAN identity/default
  initialization now represent the intended affine map, standard-normal Hermite
  rules own their probability normalization, invalid Legendre rule kinds fail
  explicitly, and functional collocation reuses canonical Chebyshev--Lobatto data.
- Numerical axis specifications, tensor/spectral methods, temporal path slicing,
  spatial-noise bases, cochain field semantics, and Laplacian spectral bases now have
  one canonical owner. Old `phydrax.domain`, `phydrax.solver`, `phydrax.operators`,
  `phydrax.graph`, and `phydrax.metrix` aliases were removed rather than deprecated:
  use `phydrax.discretization`, and use `phydrax.stochastic.SpatialNoiseBasis` for
  spatial stochastic forcing. `StochasticCouplingPlan` now owns a generic
  `DiscretizationHierarchy`.

- Spectral representations now split reusable `ModalTransform` objects from
  operator-specific `OperatorSpectrum` values. `SpectralDecomposition` pairs them
  where one API needs both, while `TransformDiagonalRepresentation` supports
  finite-difference, pseudospectral, graph, manifold, and covariance modal symbols.
- DeepONet now accepts generalized coordinate-evaluated basis trunks, exposes
  explicit output-bias control, and preserves existing pointwise and POD
  behavior while supporting projection branches and frozen nested models.
- Associative Gaussian-chain primitives now live in the lower-level linear-algebra
  implementation while their existing `phydrax.uq` public names remain unchanged.
- Nonlinear algebraic systems now have one public owner, `phydrax.nonlinear`, and
  generic continuation/bifurcation workflows have one public owner,
  `phydrax.continuation`; obsolete optimization and dynamics-analysis continuation
  exports were removed rather than retained as aliases.

- Linear solve planning now treats preconditioning as an explicit prepared subsystem with compatibility, memory, workspace, and setup-action budgets.
- Nonlinear optimizer runtimes now keep callable refresh structure static, preserve
  float32 and accepted-state carries under JIT, distinguish all-nonfinite globalization
  failures, reject unenforceable adapter evaluation budgets, and aggregate nested
  refresh and derivative diagnostics without negative sentinel arithmetic.
- Operator property evidence is propagated conservatively through preconditioners, block solvers, eigensolvers, and multigrid construction.
- RHS-width, GCRO-DR extraction, and multigrid setup/reuse resources are now rejected and reported before numerical work, including transformed prepared solves and dependency-invalidated hierarchy refreshes.
- Galerkin and smoothed-aggregation hierarchies now retain sparse assembly recipes and refresh coarse coefficients without rebuilding symbolic routes or silently densifying downstream levels.
- The linear benchmark harnesses now cover block-Jacobi preparation, sparse assembly planning/refresh/action, structured Kronecker-sum solves, and the advanced reusable and higher-operator lifecycles against dense, exact, finite-difference, or invariant references.
- The advanced-solver benchmark now provides algorithmically matched explicit-dense
  Newton-LU and matrix-free Newton-GMRES root cases for Phydrax and Optimistix, plus a
  semilinear sparse-PDE case covering Phydrax sparse-Jacobian preparation, symbolic
  reuse, numeric refresh, and native Jacobi-preconditioned PCG. It separately measures
  compiled root solves, implicit-root derivative compilation/execution, numeric
  refresh, refreshed solves, and refreshed verification; campaigns preserve float64
  inputs and canonical problem identities across adapters.
- Tensor contractions across the package, tests, benchmarks, and tools now consistently use `phydrax.ein.contract` instead of direct `jax.numpy.einsum` calls.
- Closure-converted matrix-free SVD, eigensolve, spectral-projector, density-kernel, and spectral-function derivatives now support filtered JVP, reverse mode, and JIT.
- Named truncated-normal initializers now produce their conventional target variance, while rectangular orthogonal initialization avoids max-dimension square samples.
- `phydrax.nn.layers.inference_mode` now switches every inference-aware Equinox or Phydrax leaf in mixed model trees.

### Fixed
- Failed `phydrax.typing.parse` and conversion operations roll back every dimension
  binding introduced into a shared `Scope`, while preserving bindings established
  by earlier successful operations.
- Imported closed selectors are parsed and stored canonically across the package;
  equal NumPy/Python selector scalars now produce identical scientific identities,
  and wrong-kind selectors raise `TypeError`. Operator classification losses also
  store the canonical common reduction selectors returned by `parse`.
- The certified implicit adjoint of Neural Galerkin evolution works with the default
  rectangular tangent formulation. `NeuralTangentSolvePolicy` gains
  `adjoint_linear_policy` for the self-adjoint damped normal system; it defaults to
  the Gram policy, or for the rectangular formulation to PCG with positive damping
  and MINRES without damping, instead of reusing the least-squares policy.
- Factor-graph numerical execution accepts graphs, prepared belief-propagation and
  Gibbs plans, elimination/junction plans, and normalized laws as traced arguments.
  `DiscreteFactorGraph` owns immutable host topology for shape/routing decisions
  alongside fixed device routes; elimination no longer duplicates that topology,
  forest decoding stores scope positions at preparation, and Gibbs, MAP,
  pseudolikelihood, causal enumeration and discrete reverse kernels no longer
  convert traced structure or data to Python/NumPy. Same-structure numeric parameter
  refreshes reuse compiled programs. Invalid assignments remain outside support;
  invalid evidence and reverse observations raise `equinox.EquinoxRuntimeError` in
  eager and compiled execution.
- Durable service restoration and the Kubernetes scheduler check stored JSON types
  exactly: missing fields, wrong kinds, unsupported literals, and non-integer
  counts raise `IntegrityError` instead of being coerced or split.
- Reduced-order-model capability declarations name their maturity
  (`internal`, `experimental`, `candidate`, ...) instead of storing the enumeration
  integer; the capability inventory is regenerated.
- `MACPressureOperatorSpec`, its reports and results, and
  `TwistedSYMCoordinateEvidence` store their coefficient and coordinate
  fingerprints as canonical fingerprint strings instead of mutable dictionaries in
  static fields.
- `NILSASPlan` stores the canonical memory-mode literal, so an equal NumPy string
  yields the same plan.
- Defects exposed by the complete static typing pass are fixed, each with a
  regression test:
  - Calls to attributes or keywords that do not exist now use the owning API:
    semiconductor confinement mode counting and multi-block tetrahedral face
    scopes; IGA solid, rod, and shell certificate gates (`accepted`); vector
    convolution-quadrature transpose and adjoint actions; unstructured shallow-water snapshots; distributed periodic
    LES production results; dark-sector packet-radiation frames; particle-physics
    provider records; ROM execution requirements; PSE and vortex-diffusion
    periodic boxes; panel doublet fields; Sinkhorn ordering; dyadic finite-volume
    runtimes and positivity limiting; fixed-design and non-Gaussian Bayesian
    quadrature refusals; finite-feature kernel means; spectral dealiasing and
    eigen-resolution refusals; collective-variable cell checks; and the KAN,
    DEM-batch, SPH multiphase, and particle-morphology refusals.
  - Operator-property evidence is declared with its recognized kinds in the
    neural-Galerkin certified backsolve, self-adjoint Krylov stability
    analysis, defect scattering, and multi-asset correlation factors.
  - Implicit interface and phase penalties integrate the squared residual
    function; residual-penalty quadratic reduction and variational boundary
    normals honor their contracts; point-jet and linear-reduction actions and
    narrow-band interface collocation use their default keys.
  - Finite-element tensor paths build shape tuples correctly (collocated
    pairwise flux, sum-factorized tensor diffusion), cross-block facets without
    prepared references validate, and constrained `value_and_residual` uses the
    constraint map of the compiled problem.
  - Restored Kalman and particle-filter checkpoints hold `int32` step indices;
    lidar waveform plans, ensemble reweighting, and two-puncture tuning report
    array-valued evidence; linear-refinement plateaus and thermal dark-rate
    qualification record Python booleans; numerical-relativity AMR transfer
    identities are hashable strings.
  - Least-squares dogleg and POUNDERS steps accept `termination=None`;
    Lennard-Jones PME terms and directed-graph terms without a cutoff are refused
    at preparation; rational spectral axes refuse arbitrary-point synthesis;
    finite-density domains require a temperature pair; dense conic sensitivity
    refuses sparse tangents; GPR refuses envelope-free current sources; affine
    enforcement refuses cross-field point jets without a certified lifting;
    Pennes thermal and cardiovascular high-order geometry refuse unsupported
    meshes and elements with their contract errors; NPZ archive export refuses a
    payload named `allow_pickle`; adaptive boundary targets refuse combined
    fields without self-correction; cardiovascular supports request the
    displacement value; implicit-curve evidence stores the projection evidence;
    Gmsh surface and non-semantic volume runs bind their zones before auditing;
    projected semismooth VI steps use the prepared Jacobian coordinates;
    Fresnel axes without quadrature weights use the grid-point measure;
    two-site tensor-network gates refuse scalars with the shape error;
    battery reference consensus reports missing uncertainty coverage; and
    strand-displacement workbooks skip rows without sample cells.
- Canonical identifier validation raises `TypeError` for a value that is not a
  string and keeps `ValueError` for empty or whitespace-padded strings. This
  applies to every constructor that validates identifiers through the shared
  identifier contract, including admissibility headers, discrete field views,
  finite-volume geometry protocols, facet adjacency, BEM array archives, privacy
  definitions, and qualification runtime identities.
- `ChemicalComponentCatalog` raises `TypeError` for non-integer `charges`,
  matching `element_composition`; the charge shape is still checked first.
- `VortexFlexibleCouplingPlan.step` no longer passes the unsupported `t0`/`t1`
  keywords to `SecondOrderDifferentialProblem`, which made every flexible
  structural step raise `TypeError`; the two-node `TimeGrid` remains the sole
  owner of the step interval.
- `PreparedScalarFastProvider3D.as_linear_operator` no longer passes the
  unsupported `adjoint_action` keyword to `FunctionLinearOperator`, which made the
  conversion raise `TypeError`; the adjoint is derived from the transpose action
  and the space pairings.
- Polynomial eigensolves over `BlockSpace`, `DualSpace`, `TensorProductSpace`,
  and `AxisArraySpace` sources no longer fail with `AttributeError`: residual
  norms use the space's own `inner` instead of a `pairing` that only array and
  PyTree spaces carry.
- IGA `prepare_tensor_transfer` no longer raises `AttributeError` on every call;
  it reads the stencil's `indices`.
- `DifferentialAlgebraicSystem.trial_valid` passes JAX arrays for time, state,
  and state rate to the `DAETrialValidity` callback, as its contract declares,
  also when time is a Python scalar.
- Quantum Hall `charge_gap` no longer raises `AttributeError`; it reads each
  `HaldaneSpherePlan`'s `twice_monopole_flux`.
- Ensemble reweighting reports the non-finite-whitening dual objective as a JAX
  array of the dual dtype instead of the Python float `inf`.
- Restored packaged standalone CMake build/install contracts for the Omega_h,
  TIOGA, and VoroCrust bridges, including exact external revision reporting and
  usable installed runtime linkage; corrected TIOGA's provider license metadata.
- Reduced ROM evaluations now compute QoIs from the reduced state and declared
  QoI vector instead of copying a supplied truth QoI into the ROM result.
- Acausal structural matching now prefers assignments that avoid unnecessary
  differentiation, preserving index-one physical flow/state equations instead of
  differentiating algebraic connections selected by lexical matching.
- Native Krylov norm evaluation now preserves finite reverse derivatives through
  zero happy-breakdown residuals without perturbing positive norms.
- OpenMM import preserves multiple Fourier components on each unique topological
  torsion using native series potentials, including native serialization and
  OpenMM energy/force round-trip export.
- Atomistic wrapped/unwrapped coordinate contexts retain bonded force and curvature
  derivatives instead of freezing supplied coordinate representations.
- Pairing-aware SVD uses operator-norm backward-error certification, avoiding false
  failures on numerically accurate null singular modes without relaxing tolerances.
- Implicit BP removes the fixed inner linear absolute-tolerance floor so tighter
  requested nonlinear convergence can be attained with native failure evidence.
- Native implicit DAE stages and bordered event roots now solve in increment
  coordinates, preserving small Newton corrections and accurately solved rates
  on large physical state offsets across fixed/adaptive execution and replay.
  Adaptive stage stopping is capped by the outer residual and constraint
  certificates, and higher-order steps preserve BDF ratio bounds without
  stranding order-one tails at output boundaries.
- Circuit ECM ledgers use nonuniform piecewise-quadratic saved-sample quadrature
  without crossing held-current jumps. Adaptive circuit execution is restored
  without weakening scientific acceptance limits.
- Real-coordinate Diffrax execution packs structured PyTree vector-field outputs
  before entering the array backend instead of coercing the public state container.
- Neo-Hookean finite-element forms now derive their residual from cell energy
  and support explicit two-dimensional plane strain as well as three-dimensional
  kinematics.
- Causal interval clustering now uses the Jacobian of the original reference
  coordinate, and zero-duration convolution and Caputo evaluations return exact
  zeros without evaluating singular kernels.
- Spectral, cochain, point-cloud, Maxwell, finite-volume, Clifford, and entropy
  contracts now preserve numeric identity and dtype, validate selectors and evidence,
  use exact Hodge pairings, apply directional CPML forcing, and fail closed when an
  advertised decomposition or eigenproblem is unsupported.
- Open-system solvers now enforce complete positivity and physical process evidence,
  preserve tensor precision and canonical gauges, process every event up to explicit
  capacity, align comparison time grids, propagate truncation failures, and mint
  verified campaigns only through provenance-bound artifact reproduction.
- Conic projections and sensitivities now classify canonical bounds, respect
  materialization budgets, avoid overflow in symmetric/fixed-bound arithmetic, and
  robustly handle low-precision PSD, exponential, and power-cone edge cases through
  the Phydrax linear-algebra substrate.
- Classification now preserves scalar and batch axes, validates vocabularies, labels,
  focal policies, masks, and operator semantics, excludes zero-support observations,
  and uses stable ordinal tails and positive-class binary overlap.
- Layer-potential QBX/FMM paths now use target normals and declared source reference
  triangles, share polynomial density reconstruction, propagate quadrature failures,
  report bounded omitted tails, bind source provenance, and route dense solves through
  `phydrax.linalg`. Bounded GP MAP maps avoid overflow, use disjoint design streams,
  and report source-bound, correctly normalized benchmark timings.
- Incompressible spectral workflows now bind every callable identity, preflight
  Hermitian-coordinate, recurrence, and channel-factor resources, compose reflected
  translations correctly, preserve linear-algebra precision, certify autonomous-flow
  neutral multipliers, latch bounded-observer failures, and verify spectral artifact
  content fingerprints on read.

- Masked BSDE and deep-splitting losses now sanitize inactive residuals before
  nonlinear reductions, Flower sanitizes masked source and normalization state,
  and ragged-series pooling selects inactive latent values before reduction.
