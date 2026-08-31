# Examples

This section links to reproducible, runnable Phydrax examples as public [Marimo](https://marimo.io) notebooks.

## Wave Equation (1D)

A tutorial notebook showing PCI enforced overlays, latent-factorized modeling, and efficient JVP-based differential operators for the 1D wave equation, with comparisons to the [Nvidia PhysicsNeMo](https://docs.nvidia.com/physicsnemo/latest/physicsnemo-sym/user_guide/foundational/1d_wave_equation.html) implementation.

- Public notebook: [wave1d](https://static.marimo.app/static/wave1d-ul81)

## Coupled Spring-Mass ODE

A benchmark notebook for the coupled 3-DOF spring-mass system in matrix form, with normalized-time training, exact initial-condition enforcement, and comparison context against the [NVIDIA PhysicsNeMo spring-mass example](https://docs.nvidia.com/physicsnemo/25.11/physicsnemo-sym/user_guide/foundational/ode_spring_mass.html).

- Public notebook: [spring-mass-ode](https://static.marimo.app/static/spring-mass-ode-xuq3)

## Particle physics scripts

The repository includes directly runnable scripts for the fixed-capacity particle stack:

```text
python examples/discrete_element_method.py
python examples/material_point_method.py
python examples/electrostatic_pic.py
python examples/electromagnetic_pic.py
python examples/flip_dam_break.py
python examples/wet_granular_bridge.py
python examples/superquadric_collision.py
python examples/particle_internal_heating.py
python examples/particle_radial_drying.py
python examples/reactive_cfd_dem.py
```

Each script prints its acceptance flag and the balance or geometry residuals
that qualify the exercised route. The material-point example additionally
reports transfer mass defect and minimum deformation Jacobian.

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
