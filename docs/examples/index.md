# Examples

This section collects public [Marimo](https://marimo.io) notebooks and directly runnable repository scripts.

## Meshfree spatial and temporal workflows

Run these checkout examples from the repository root with explicit float64
support. The commands select implementations, not production release authority.

```text
JAX_ENABLE_X64=1 python -m examples.point_cloud_poisson
JAX_ENABLE_X64=1 python -m examples.meshfree_bulk_advection_diffusion
JAX_ENABLE_X64=1 python -m examples.meshfree_multilevel_poisson
JAX_ENABLE_X64=1 python -m examples.meshfree_conservative_diffusion
JAX_ENABLE_X64=1 python -m examples.meshfree_surface_laplace_beltrami
JAX_ENABLE_X64=1 python -m examples.meshfree_open_surface_diffusion
JAX_ENABLE_X64=1 python -m examples.meshfree_surface_vector_pde
JAX_ENABLE_X64=1 python -m examples.meshfree_moving_surface_reaction_diffusion
JAX_ENABLE_X64=1 python -m examples.meshfree_surface_topology_events
JAX_ENABLE_X64=1 python -m examples.meshfree_higher_forms
JAX_ENABLE_X64=1 python -m examples.meshfree_incompressible_flow
JAX_ENABLE_X64=1 python -m examples.meshfree_lagrangian_flow
JAX_ENABLE_X64=1 python -m examples.meshfree_elasticity
JAX_ENABLE_X64=1 python -m examples.meshfree_bulk_surface_exchange
JAX_ENABLE_X64=1 python -m examples.meshfree_mixed_method_coupling
JAX_ENABLE_X64=1 python -m examples.meshfree_fluid_structure
JAX_ENABLE_X64=1 python -m examples.meshfree_adaptive_learning
JAX_ENABLE_X64=1 python -m examples.meshfree_hybrid_calibrated
JAX_ENABLE_X64=1 python -m examples.meshfree_learned_edge_flux
JAX_ENABLE_X64=1 python -m examples.meshfree_learned_metric_correction
```

The examples cover original-equation residuals, boundary/trace identity,
conservation, native solver evidence, and admitted versus refused updates.
Physical surface events require their multiregion authority. Higher-form PDEs
require an oriented geometry-authorized complex; an abstract clique complex is
a separate research route. Material particle mass and quadrature volume are
distinct representations.

`examples.meshfree_production_restart` and
`examples.meshfree_production_epochs` provide checkpoint/restart CLI workflows;
use their `--help` for repository and interruption controls. They retain live
history, source/support identity, and migration receipts instead of treating
same-shaped data as a compatible restart.

See [Meshfree solvers](../guides_meshfree.md) and the
[meshfree API](../api/discretization/meshfree.md) for precision, support,
geometry, boundary, capacity, derivative, and refusal contracts.
Qualification campaigns are independently selectable through
`python -m tools.meshfree_qualification --help`. Automated numerical evidence,
including forced-CPU distributed checks, does not authorize a GPU, multi-host,
scientific-validation, or production-release claim.

## Black holes and numerical relativity

```text
python examples/kerr_horizon_thermodynamics.py
python examples/kerr_shadow_rays.py
python examples/polarized_fast_light_grrt.py
python examples/qnm_scattering_hawking.py
python examples/compact_object_accretion.py
python examples/fixed_grid_z4c.py
python examples/binary_black_hole_extraction.py
python examples/simulation_product_visibility.py
```

These are bounded public-API demonstrations. The stationary thermodynamics script
does not claim a dynamical horizon. The Kerr ray fan records capture/escape but does
not render. Fast-light GRRT binds an exact `GRRayResult` and metric to
`PolarizedRayPath`, requires the snapshot and path chart IDs to match, prepares
active/valid segment-midpoint sampling bound to both path and snapshot, and performs
invariant transfer. Only the numerical path/transfer and MNY96 Stokes-$I$ support are
qualified; active polarization/Faraday and the composite polarized prediction remain
reference-unqualified. The QNM, real-frequency scattering, and Hawking products remain
separate, and its illustrative scattering amplitudes/tail evidence stay unqualified.
The accretion script builds Michel and Fishbone--Moncrief initial data without
evolution. The Z4c script advances two periodic fixed-grid steps. The binary script
produces Brill--Lindquist candidate-surface/null/quasilocal diagnostics and
finite-radius $\Psi_4$ multipoles, not a certified MOTS, evolved binary, or asymptotic
waveform. The visibility script performs no external I/O or rendering.

See [Black-hole sources, provenance, rights, and qualification](../black_hole_sources.md)
for the separate benchmark and eleven-profile qualification lanes. Example output is
not production qualification or PNPL deployment authorization.

## Nuclear and tokamak workflow

```text
python examples/tokamak_fusion_activation.py
python tools/nuclear_tokamak_qualification.py
python benchmarks/nuclear_tokamak.py --warmup 1 --repeats 5
```

The example runs a synthetic imported-geometry core-transport, D-T source,
explicit neutron-response, and activation step. The qualification tool uses
analytic synthetic controls. Neither constitutes a reactor-safety, operational
tokamak, evaluated-data, or external-neutronics qualification.

## Square-lattice tensor renormalization

```text
python examples/ising_tensor_renormalization.py
python tools/tensor_renormalization_qualification.py
python benchmarks/tensor_renormalization.py --repeats 3
```

The example lowers real positive-semidefinite Ising pair weights into a uniform
square tensor and estimates the thermodynamic-limit logarithmic partition
density with fixed-rank HOTRG. The qualification checks both TRG and HOTRG
against Onsager references; local discarded weights are not a global error
bound.

## Wave Equation (1D)

A tutorial notebook showing PCI enforced overlays, latent-factorized modeling, and efficient JVP-based differential operators for the 1D wave equation, with comparisons to the [Nvidia PhysicsNeMo](https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/foundational/1d_wave_equation.html) implementation.

- Public notebook: [wave1d](https://static.marimo.app/static/wave1d-ul81)

## Coupled Spring-Mass ODE

A benchmark notebook for the coupled 3-DOF spring-mass system in matrix form, with normalized-time training, exact initial-condition enforcement, and comparison context against the [NVIDIA PhysicsNeMo spring-mass example](https://docs.nvidia.com/physicsnemo/25.11/physicsnemo-sym/user_guide/foundational/ode_spring_mass.html).

- Public notebook: [spring-mass-ode](https://static.marimo.app/static/spring-mass-ode-xuq3)

## Battery equation models and admission

These checkout-only scripts import repository support modules. Run them from the
repository root with that root on the import path:

```text
PYTHONPATH=. python examples/battery_simulation_and_optimization.py --development
PYTHONPATH=. python examples/battery_circuit_ecm.py --development
PYTHONPATH=. python examples/battery_spme.py --development
```

These scripts explicitly select unreleased development candidates. They execute
native current/rest trajectories and check model status and conservation. The
thermal ECM example also composes generic bounded optimization and re-executes
the accepted current; that local check is not authenticated independent release
replay. The SPMe example uses self-authored equation data, not fitted measurements.

Circuit ECM and SPMe production mode requires externally retained signed deployment proofs,
separately provisioned trust roots, and an executor-signed runtime byte
attestation. Historical ECM artifacts cannot authorize the changed distribution.
See [thermal ECM equations](../guides_battery.md) and
[battery production admission](../guides_battery_production.md).

## Linear-quadratic feedback game

```text
python examples/lq_nash_game.py
```

The script solves a two-player affine finite-horizon full-state feedback Nash
game, checks curvature, rank, conditioning, stationarity, and Bellman
evidence, replays the joint affine policy through the physical control
contract, and compares both direct discrete payoffs with their initial value
functions.

## Nonlinear, constrained, and stochastic game scripts

```text
python examples/nonlinear_feedback_game.py
python examples/open_loop_variational_game.py
python examples/constrained_open_loop_game.py
python examples/stochastic_feedback_control.py
python examples/lqg_feedback_game.py
python examples/hjbi_reference_game.py
python examples/mean_field_game.py
python examples/common_information_game.py
```

`nonlinear_feedback_game.py` runs residual-globalized iLQ and independently
recomputes the accepted local nominal stationarity residual. It does not relabel
that evidence as an exact nonlinear feedback-Nash result.

`open_loop_variational_game.py` solves a convex shared-resource VE with one common
shared multiplier. `constrained_open_loop_game.py` solves a nonlinear
opponent-dependent private open-loop KKT system; its result is local KKT evidence,
not a feedback or global GNE certificate.

`stochastic_feedback_control.py` evaluates one unchanged feedback policy on
disjoint prepared training and holdout noise, preserving path, coupling, and
independence-cluster provenance. `lqg_feedback_game.py` exercises the exact
additive-noise, full-state LQG feedback-Nash recursion and per-player value trace
corrections.

`hjbi_reference_game.py` evaluates declared lower and upper zero-sum action orders
and reports the finite-grid residual, refinement, and Isaacs-gap gates.
`mean_field_game.py` keeps frozen-law response evaluation separate from the
independently induced-law fixed-point check and reports the empirical-law evidence
ceiling.

`common_information_game.py` performs pure-prescription Bayesian backward
induction over a finite public state, then queries each player's policy with only
that public state and the player's own private type.

## Shallow-water scripts

The wet/dry and rotating-flow paths have directly runnable qualification examples:

```text
python examples/shallow_water_wet_dry.py
python examples/rotating_shallow_water.py
```

The first reports stage acceptance, minimum depth, mass defect, and wet-cell count.
The second exercises identified f/beta-plane forcing and reports mass and momentum
norm diagnostics. See [Shallow water](../guides_shallow_water.md).

## Ocean process scripts

The Cartesian rigid-lid Boussinesq product has directly runnable examples:

```text
python examples/ocean_inertial_oscillation.py
python examples/ocean_stratified_adjustment.py
python examples/ocean_surface_flux_column.py
```

They exercise weighted-skew f-plane rotation, state-dependent stratification bounds,
directional T/S diffusion, and conservative surface heat flux. See
[Cartesian ocean process modeling](../guides_ocean.md).

## Hydrostatic and coastal ocean scripts

```text
python examples/hydrostatic_external_wave.py
python examples/hydrostatic_wetdry_freshwater.py
python examples/hydrostatic_spherical_thermodynamics.py
```

These exercise prognostic free surface, implicit and split-explicit external modes,
freshwater volume, conservative wetting/drying, partial/z-star geometry,
latitude-longitude metrics, nonlinear seawater thermodynamics, and vertical closures.
See [Hydrostatic primitive-equation ocean modeling](../guides_hydrostatic_ocean.md).

## One-phase free-surface hydrodynamics

```text
python examples/free_surface_ale_wave.py
```

The script exercises graph ALE geometry, extensive mapped momentum, mixed
pressure projection, coupled surface kinematics, scalar GCL, and accepted work
evidence. See
[One-phase free-surface ALE hydrodynamics](../guides_free_surface_ale_hydrodynamics.md).

## Advanced and two-phase hydrodynamics

```text
python examples/advanced_capillary_wave.py
python examples/advanced_rigid_hydroelastic_body.py
python examples/advanced_two_phase_vof.py
python examples/passive_tracer_maccormack.py
python examples/advanced_rising_bubble.py
```

These exercise variational graph capillarity, coherent wave forcing and absorption,
mapped rigid/modal coupling, conservative two-phase VOF flow, and an explicitly
nonconservative bounded periodic passive tracer. See
[Advanced hydrodynamics](../guides_advanced_hydrodynamics.md),
[Structured finite volume](../guides_finite_volume.md), and
[Two-phase hydrodynamics](../guides_two_phase_hydrodynamics.md).

## Bubble dynamics and resolved bubbly flow

```text
python examples/advanced_acoustic_bubble.py
python examples/advanced_contrast_agent_microbubble.py
python examples/advanced_surface_nanobubble.py
python examples/advanced_bubble_cloud.py
python examples/advanced_bubble_coalescence.py
python examples/advanced_thermocapillary_droplet.py
python examples/advanced_bubble_evidence_exchange.py
python examples/advanced_lbm_emulsion.py
```

- `advanced_acoustic_bubble.py` exercises a Keller--Miksis air bubble with
  boundary-layer thermal gas, tone-burst forcing, and linear-response evidence.
- `advanced_contrast_agent_microbubble.py` compares clean and Marmottant-shell
  responses and recovers shell elasticity and viscosity by bounded inverse fitting.
- `advanced_surface_nanobubble.py` evaluates pinned Lohse--Zhang stability and
  free, pinned, and unpinned Epstein--Plesset dissolution.
- `advanced_bubble_cloud.py` composes heterogeneous cloud dynamics, far-field
  emission, Bjerknes forces, dense/FMM coupling, and retarded propagation.
- `advanced_bubble_coalescence.py` runs marker-resolved near-contact drainage and
  gated coalescence with separate pressure and contact-work ledgers.
- `advanced_thermocapillary_droplet.py` transports material temperature and
  compares Marangoni migration with the Young--Goldstein--Block control.
- `advanced_bubble_evidence_exchange.py` creates an explicitly unaccepted
  resolved-to-reduced request that retains missing impulse and flow-field
  invariants rather than performing an automatic model handoff.
- `advanced_lbm_emulsion.py` compares merging and pairwise-repelled N-color
  emulsions while reporting mass, momentum, and near-contact-work evidence.

The rising-bubble script in the preceding hydrodynamics block runs the Hysing TC1
workflow and reports circularity, centroid, rise speed, volume, and solver evidence.
The Hysing route remains a candidate until every declared campaign gate passes.

## Surface films and interference color

```text
python examples/advanced_bubble_film_drainage.py
python examples/advanced_soap_film_tunnel.py
python examples/advanced_thin_film_iridescence.py
```

- `advanced_bubble_film_drainage.py` drains a spherical soap film with DLVO
  disjoining pressure and reports conservation, positivity, and black-film
  equilibrium evidence.
- `advanced_soap_film_tunnel.py` runs a gravity-driven cylinder wake at
  Reynolds number 150 with flux ledgers, film-Mach evidence, Strouhal
  estimation, and an interference-color image; it requires `phydrax-meshcore`.
- `advanced_thin_film_iridescence.py` renders an air--soap--air thickness and
  viewing-angle sweep through Airy interference, spectral colorimetry, and sRGB
  encoding, with fringe-sampling, energy, gamut, and white-balance evidence.

## Threshold dynamics and explicit foam geometry

```text
python examples/advanced_foam_coarsening.py
python examples/advanced_grain_growth_3d.py
python examples/advanced_threshold_surface_seeding.py
python examples/advanced_double_bubble.py
python examples/advanced_catenoid_collapse.py
python examples/advanced_foam_burst.py
python examples/advanced_foam_bubble_oscillation.py
python examples/advanced_plateau_border_drainage.py
python examples/advanced_foam_iridescence.py
python examples/foam_junction_rebind.py
python examples/advanced_biomembrane_remeshing.py
```

- `advanced_foam_coarsening.py` compares unconstrained threshold coarsening,
  exact capacitated label volumes, and gas-diffusive von Neumann evolution.
- `advanced_grain_growth_3d.py` runs capillarity-only grain growth with sparse
  candidate labels and explicit stencil-support and dissipation evidence.
- `advanced_threshold_surface_seeding.py` converts a hard-label field into a
  collision-certified multiregion surface and prepares the foam equilibrium route.
- `advanced_double_bubble.py` relaxes an unequal soap-film double bubble and
  compares pressures and geometry with the closed-form reference.
- `advanced_catenoid_collapse.py` continues to the catenoid stability limit,
  then demonstrates certified remeshing, pinch, region split, and disk relaxation.
- `advanced_foam_burst.py` deletes an accepted thin sheet, conserves liquid and
  gas ledgers through the merge lineage, and advances the merged constrained cell.
- `advanced_foam_bubble_oscillation.py` advances a volume-constrained quadrupole
  with circulation-slot vortex air, regularized FMM, and bounded direct-error evidence.
- `advanced_plateau_border_drainage.py` couples per-sheet film operators to a
  physical Plateau-border network with liquid, surfactant, and evaporation ledgers.
- `advanced_foam_iridescence.py` renders drained and ruptured foam surfaces with
  support masks and verifies that rendering leaves the physical states unchanged.
- `foam_junction_rebind.py` drains three double-bubble sheets into a declared
  Plateau-border junction, splits a border through one accepted composition
  rebind with explicit sheet and border content transports, renders the
  published thickness one way, and refuses a rupture without a border rule.
- `advanced_biomembrane_remeshing.py` sends a membrane face flip through the
  shared certified surface-event transaction with sparse conservative transfers.

## Particle physics scripts

The repository includes directly runnable scripts for the fixed-capacity particle stack:

```text
python examples/discrete_element_method.py
python examples/material_point_method.py
python examples/material_point_schedules.py
python examples/material_point_materials.py
python examples/material_point_domains_sparse.py
python examples/material_point_contact_fracture.py
python examples/material_point_implicit.py
python examples/material_point_commercial_runtime.py
python examples/material_point_commercial_mechanics.py
python examples/material_point_commercial_scale.py
python examples/electrostatic_pic.py
python examples/electromagnetic_pic.py
python examples/pic_field_solver_substitution.py
python examples/pic_field_handoff.py
python examples/flip_dam_break.py
python examples/wet_granular_bridge.py
python examples/superquadric_collision.py
python examples/particle_internal_heating.py
python examples/particle_radial_drying.py
python examples/reactive_cfd_dem.py
python examples/prescribed_immersed_cylinder.py
```

Each script prints its acceptance flag and balance, geometry, constitutive,
contact, nonlinear, or topology evidence for the exercised route.

## Atomistic ecosystem scripts

The atomistic examples exercise the native force-field, trajectory-interchange,
enhanced-sampling, and committee-uncertainty paths:

```text
python examples/atomistic_force_field.py
python examples/atomistic_virtual_sites.py
python examples/atomistic_interop.py
python examples/atomistic_ipi.py
python examples/atomistic_sampling.py
python examples/atomistic_uncertainty.py
```

Each script is self-contained, uses stable prepared plans, and fails if the exercised
runtime contract is unsuccessful.

## Velocimetry scripts

The native image-measurement stack includes deterministic, directly runnable
workflows:

```text
python examples/piv_synthetic_translation.py
python examples/ptv_calibrated_stereo.py
python examples/stb_synthetic_particles.py
python examples/learned_piv_synthetic_training.py
python examples/velocimetry_interop.py
```

The scripts report measurement validity and scientific error evidence. They
distinguish image displacement from physical velocity and reconstructed
particle identities from latent synthetic particle IDs.

## Skeletal-muscle motor units

The public sustained-isometric motor-unit example is directly runnable:

```text
python examples/skeletal_muscle_motor_units.py
```

It reports source and plan identities, relative force, capacity loss, recruitment,
and complete transition success. The output is not force in newtons or a dynamic
musculotendon simulation. Qualification and benchmark entry points are:

```text
python tools/skeletal_muscle_motor_unit_qualification.py
python benchmarks/skeletal_muscle_motor_units.py
```

The remaining source-bounded skeletal and provider examples are:

```text
python examples/skeletal_motor_units_fuglevand_1993.py
python examples/skeletal_fatigue_liu_2002.py
python examples/skeletal_force_calibration.py
python examples/skeletal_muscle_fast_twitch.py
python examples/skeletal_muscle_fibers.py
python examples/skeletal_muscle_motor_territories.py
python examples/skeletal_musculotendon_de_groote_fregly_2016.py
python examples/skeletal_muscle_continuum.py
python examples/skeletal_muscle_proprioception.py
python examples/skeletal_muscle_emg.py
python examples/skeletal_muscle_energetics.py
python examples/skeletal_muscle_multimodal_uq.py
python examples/skeletal_muscle_control_replay.py
python examples/skeletal_muscle_interchange_worksets.py
python examples/robotics_fixed_body_muscle_route.py
python examples/robotics_analytic_wrap.py
python examples/robotics_mjx_muscle_projection.py
```

Qualification entry points:

```text
python tools/skeletal_motor_units_qualification.py
python tools/skeletal_fatigue_qualification.py
python tools/skeletal_force_calibration_qualification.py
python tools/skeletal_muscle_cell_qualification.py
python tools/skeletal_muscle_fiber_qualification.py
python tools/qualify_de_groote_fregly_2016.py
python tools/qualify_fixed_body_route.py
python tools/qualify_analytic_route_wrap.py
python tools/qualify_mjx_muscle_projection.py
python tools/skeletal_muscle_continuum_qualification.py
python tools/skeletal_proprioception_qualification.py
python tools/skeletal_muscle_emg_qualification.py
python tools/skeletal_muscle_energetics_qualification.py
python tools/skeletal_muscle_interchange_qualification.py
```

Benchmark entry points:

```text
python benchmarks/skeletal_motor_units.py
python benchmarks/skeletal_fatigue.py
python benchmarks/skeletal_force_calibration.py
python benchmarks/skeletal_muscle_cell.py
python benchmarks/skeletal_muscle_fibers.py
python benchmarks/skeletal_muscle_musculotendon.py
python benchmarks/skeletal_muscle_continuum.py
python benchmarks/skeletal_muscle_proprioception.py
python benchmarks/skeletal_muscle_emg.py
python benchmarks/skeletal_muscle_energetics.py
python benchmarks/skeletal_muscle_execution_worksets.py
python benchmarks/robotics_fixed_body_routes.py
python benchmarks/robotics_analytic_wrap.py
python benchmarks/robotics_mjx_muscle_projection.py
```

## Numerical interoperability

```text
python examples/numerical_interface_binding.py
python examples/prepared_field_observations.py
python examples/exterior_laplace_galerkin.py
python examples/coupled_scalar_regions.py
python examples/nitsche_transmission.py
python examples/sem_vem_bem_transmission.py
python examples/coupled_inverse_problem.py
python examples/coupled_learned_interface.py
python examples/mixed_method_time_coupling.py
python examples/named_block_dae.py
python examples/coupled_transient_fields.py
python examples/coupled_control.py
python examples/coupled_data_assimilation.py
python examples/coupled_rom_swap.py
python examples/hybrid_pinn_classical.py
python examples/adaptive_fe_fv_rebind.py
python examples/fmi_host_coupling.py
python examples/coupled_lane_worksets.py
python benchmarks/numerical_interoperability.py --smoke
```

Each script exercises one family of
[Numerical interoperability](../guides_numerical_interoperability.md) and prints
its evidence and refusals:

- `numerical_interface_binding.py` binds two independently meshed parts of one
  B-Rep to one interface and refuses a stale part revision.
- `prepared_field_observations.py` observes a refreshed finite-element field
  through one prepared query, pulls a sensor residual back through the exact
  transpose, and reports a rejected sensor under masked coverage.
- `exterior_laplace_galerkin.py` recovers an off-center dipole field with the
  2-D Galerkin boundary operator under panel refinement.
- `coupled_scalar_regions.py` couples P1 triangles and polygonal virtual elements
  across a nonmatching interface with a side-trace mortar.
- `nitsche_transmission.py` solves a diffusivity jump on nonmatching P1 and P2
  refinements with certified Nitsche penalties and couples a virtual-element
  region one-sidedly.
- `sem_vem_bem_transmission.py` solves one Laplace field across spectral,
  virtual, and boundary elements in one certified solve, substitutes each volume
  method, and prints the refinement campaigns.
- `coupled_inverse_problem.py` infers a conductivity and a wall heat flux of the
  finite-element / virtual-element plate from observations with declared
  uncertainty through `SolverObjective` and `train_components`.
- `coupled_learned_interface.py` trains a learned conductivity field through the
  accepted solve and a learned preconditioner through fixed FGMRES work, and
  shows that neither the preconditioner nor an untrusted initial guess changes
  the accepted solution.
- `mixed_method_time_coupling.py` couples a backward-Euler finite-element solid
  and an SSPRK(3,3) finite-volume fluid across a nonmatching interface through
  declared temporal conversions and a certified conservative map.
- `named_block_dae.py` preconditions a reduced thermal DAE by its named
  differential and algebraic blocks and checks the analytic decay.
- `coupled_transient_fields.py` integrates finite-element / virtual-element heat
  conduction with a moving lift as one index-one DAE, continues an adaptive run
  across windows, and refuses the mortar-multiplier declaration.
- `coupled_control.py` rolls out, linearizes, and runs a dense MPC on the wall
  heat flux of a coupled transition against host references.
  `coupled_data_assimilation.py` runs the ensemble transform Kalman filter on the
  same transition against the exact linear-Gaussian Kalman filter. Both use
  `examples/_coupled_heat_transient.py`.
- `coupled_rom_swap.py` replaces the finite-element region of the plate by POD
  Galerkin models of increasing rank under unchanged declarations.
- `hybrid_pinn_classical.py` couples a surrogate network region to a
  finite-element region through an accepted state-design response and one
  Dirichlet–Neumann step.
- `adaptive_fe_fv_rebind.py` refines one side of the mixed-method coupling at an
  accepted boundary through one lifecycle composition rebind and continues the
  windows.
- `fmi_host_coupling.py` couples a compiled thermal-zone FMU to a native node
  under explicit host orchestration; without FMPy or a C compiler it reports the
  missing prerequisite.
- `coupled_lane_worksets.py` evaluates a twelve-strip mortar chain per owner and
  as bounded lane worksets and checks both against the exact harmonic field.

`foam_junction_rebind.py` (threshold dynamics and explicit foam geometry) and
`pic_field_solver_substitution.py` and `pic_field_handoff.py` (particle physics)
are listed with their owners above. The benchmark rows of
`benchmarks/numerical_interoperability.py` (selected with `--rows`) record
preparation, compilation, runtime, and retained-byte evidence separately from the
qualification scenarios.

## Meshing

Public facade workflows:

```text
python examples/meshing_native.py
python examples/adaptive_bisection_heat.py
python examples/adaptive_device_simplex.py
python examples/anisotropic_metric_adaptation.py
python examples/ale_conservative_remesh.py
python examples/cad_high_order_curving.py
python examples/boundary_layer_core_mesh.py
python examples/delaunay_voronoi.py
python examples/native_surface_meshing.py
python examples/native_tetrahedral_meshing.py
python examples/native_multiblock_meshing.py
python examples/native_polyhedral_meshing.py
python -m tools.meshing_qualification --scenario bisection
python -m tools.meshing_benchmarks --case host-bisection --resolution 64 --repeats 1
```

Bounded repository integration recipes additionally include
`native_quad_hex_meshing.py`, `curved_native_transfer.py`, and
`hp_metric_order_adaptation.py`. The qualification-only
`tools/native_moving_overset.py` exercises source-aware motion and restart.
`tools/layer_core_lifecycle_continuation.py` consumes an already published
source archive and therefore also requires its path and expected content ID.
Those tools use internal source/archive owners; they do not enlarge the public
API.

The [meshing guide](../guides_meshing.md#workflows) records each route's exact
source, certification, resource, periodic, tolerance, and evidence boundaries.
Native generation, triangulation, supermesh, and remap routes require the
separately packaged meshcore library (`phydrax[meshcore]` or
`PHYDRAX_MESHCORE_LIBRARY`). `cad_high_order_curving.py` and
`boundary_layer_core_mesh.py` are now native and do not require Gmsh or OCP.
`meshing_omega_h.py` remains an explicit optional Omega_h comparison; Gmsh, Mmg,
fTetWild, Manifold, OpenVDB, Poisson, VoroCrust, TIOGA, METIS, and OCCT/OCP are
likewise used only by their named provider or interchange boundaries. An absent
optional provider is `missing-dependency`, never a native pass or product
failure.

## Medical imaging and neurofluid transport

```text
python examples/image_mesh_transfer.py
python examples/neurofluid_transport.py
python -m tools.neurofluid_qualification
python -m tools.neurofluid_benchmarks --size 32 --queries 4096
```

The image example proves affine scalar reproduction through the consistent P1
projection. The neurofluid example advances a closed bulk/network/reservoir
system and verifies total mass. Qualification uses only synthetic de-identified
artifacts; it is not clinical validation.

## Cardiovascular platform

The public end-to-end script uses only canonical facades:

```text
python examples/cardiovascular_platform.py
```

It binds quantity and case identities, constructs harmonic cardiac coordinates
and ventricular microstructure, advances phenomenological electrophysiology,
observes activation, replays a checkpoint, compares circulation and observation
pressure--volume work, and confirms that incomplete commercial evidence is
refused. The model and synthetic cube are research demonstrations, not clinical
or commercial qualification.

Cardiovascular qualification tools:

```text
python tools/cardiovascular_geometry_qualification.py
python tools/cardiovascular_high_order_qualification.py
python tools/cardiovascular_ep_foundation_qualification.py
python tools/cardiovascular_advanced_ep_qualification.py
python tools/cardiovascular_mechanics_qualification.py
python tools/cardiovascular_circulation_qualification.py
python tools/cardiovascular_hemodynamics_qualification.py
python tools/cardiovascular_observation_qualification.py
python tools/cardiovascular_personalization_qualification.py
python tools/cardiovascular_learning_qualification.py
python tools/cardiovascular_runtime_qualification.py
python tools/cardiovascular_release_qualification.py
```


## Omniphysics candidate workflows

Run the public examples and the retained qualification campaign:

```text
python examples/materials_homogenization.py
python examples/ded_single_track.py
python examples/electroviscoelastic_drop.py
python examples/process_flash_recycle.py
python benchmarks/omniphysics_production_qualification.py
```

The benchmark retains independent controls, refinement trends, analytic application references, and exact provider facts. These candidate demonstrations confer no distributed, experimental, safety, or release claim.
