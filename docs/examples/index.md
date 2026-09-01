# Examples

This section links to reproducible, runnable Phydrax examples as public [Marimo](https://marimo.io) notebooks.

## Wave Equation (1D)

A tutorial notebook showing PCI enforced overlays, latent-factorized modeling, and efficient JVP-based differential operators for the 1D wave equation, with comparisons to the [Nvidia PhysicsNeMo](https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/foundational/1d_wave_equation.html) implementation.

- Public notebook: [wave1d](https://static.marimo.app/static/wave1d-ul81)

## Coupled Spring-Mass ODE

A benchmark notebook for the coupled 3-DOF spring-mass system in matrix form, with normalized-time training, exact initial-condition enforcement, and comparison context against the [NVIDIA PhysicsNeMo spring-mass example](https://docs.nvidia.com/physicsnemo/25.11/physicsnemo-sym/user_guide/foundational/ode_spring_mass.html).

- Public notebook: [spring-mass-ode](https://static.marimo.app/static/spring-mass-ode-xuq3)

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
```

These exercise variational graph capillarity, coherent wave forcing and absorption,
mapped rigid/modal coupling, and conservative two-phase VOF flow. See
[Advanced hydrodynamics](../guides_advanced_hydrodynamics.md) and
[Two-phase hydrodynamics](../guides_two_phase_hydrodynamics.md).

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
