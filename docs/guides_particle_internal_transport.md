# Particle internal transport

Phydrax represents unresolved particle interiors as fixed-capacity conservative radial finite-volume batches. This path is nondistributed and independent of the DEM contact integrator; reactive CFD–DEM composes both prepared dynamics explicitly.

## Radial shell geometry

`RadialShellMeshPlan` supports `SLAB`, `CYLINDER`, and `SPHERE`. The plan fixes normalized shell faces. `PreparedRadialShellMesh.metrics(outer_scale)` computes exact dimensional cell measures, face measures, centroid locations, and centroid-to-centroid distances for every particle.

The measure identities are exact up to floating-point roundoff:

```text
slab total measure       = L
cylinder total measure   = πR²
sphere total measure     = 4πR³ / 3
```

A changing `outer_scale` recomputes all metrics. No reference-volume approximation is retained after morphology changes.

## Homogeneous batches

`ParticleInternalBatchPlan` maps selected global particle slots to one homogeneous internal model:

- one radial geometry and shell count;
- one species schema width;
- one optional shrinking-core front count;
- fixed global-to-local and local-to-global owner maps.

Use multiple batches for different materials or internal discretizations. The global particle pool remains fixed-capacity; batch membership is static. Dynamic activity is carried by state.

```text
batch_plan = phx.discretization.ParticleInternalBatchPlan(
    owner_indices,
    phx.discretization.RadialShellMeshPlan(
        phx.discretization.ParticleInternalGeometry.SPHERE,
        cell_count=8,
    ),
    species_count=3,
)
batch = batch_plan.prepare(particles)
```

## Extensive state

`ParticleInternalBatchState` stores shell internal energy and species amount as extensive quantities. Porosity, internal surface area, outer scale, reaction-front coordinates, and activity complete the state. Signed internal energy is valid because material reference energies are arbitrary; species amount, surface area, and scale retain physical positivity constraints.

`initialize_particle_conversion_state` combines batches and initializes an accepted-step ledger. `ParticleConversionStateGeometry` adds continuous extensive tangents while preserving discrete ownership, activity, IDs, and batch structure.

## Conservative transport

`ParticleTransportMaterialPlan` provides phase-weighted thermal conductivity and species diffusivity. `evaluate_particle_transport` uses one finite-volume flux on each interior face and one explicit surface exchange channel. Interior contributions cancel pairwise. Diagnostics report:

- internal energy closure;
- per-species internal closure;
- entropy production;
- explicit stability restriction;
- reconstructed thermodynamic state.

Boundary values are supplied with `ParticleTransportBoundary`. Heat and species transfer coefficients are independent; prescribed source rates remain separate channels.

### Reusable radial species action

`RadialSpeciesTransportPlan(species_count)` separates the static species width from runtime material and geometry values. Bind it once to a `PreparedRadialShellMesh` with `prepare_radial_species_transport(plan, mesh)`. The prepared action accepts species amounts with trailing `(shell, species)` axes and preserves every leading batch axis.

The radial branch of `evaluate_particle_transport` uses this action directly. Its established `boundary_species_rate` remains inward-positive, while the reusable action reports the integrated outer rate as outward-positive.


At each evaluation, pass `outer_scale`, shell `storage_measure`, `cell_diffusivity`, outward-positive `outer_molar_flux`, and an `active_mask`. Storage may be shared across leading entries only with shape `(shell,)`; otherwise it must have exact shape `(..., shell)`. Diffusivity may have shape `(species,)`, `(shell, species)`, `(..., 1, species)`, or `(..., shell, species)`. Outer flux may have shape `(species,)` or exact shape `(..., species)`. Other implicit broadcasting is rejected.

`RadialSpeciesTransportResult` exposes concentrations, face molar-flux densities, shell amount rates, outward-positive integrated outer amount rates, an explicit diffusion limit, conservation defects, and per-entry validity. The origin face flux is exactly zero. Each active entry satisfies

```text
sum(shell amount rate) + outward outer amount rate = conservation defect ≈ 0.
```

Inactive finite entries return zero concentrations, fluxes, rates, outer rates, and defects, with an infinite explicit time-step limit. A nonpositive storage measure or radius, or a negative diffusivity, invalidates only an active entry; zero diffusivity is valid and gives an infinite local diffusion limit. The explicit limit is the frozen-coefficient diffusion diagonal bound: it is a stability estimate, not an accuracy guarantee, and state-dependent or nonlinear diffusivity must be supplied again on every evaluation. Prescribed outer flux is forcing and does not enter that bound. Runtime radii and diffusivities remain ordinary array inputs, so the same prepared topology works under JIT, batching, and parameter differentiation.

## Solver backends

Compile through `ParticleConversionProblemIR` and `compile_particle_conversion_problem`. Then select:

- `REFERENCE_ROSENBROCK`: a general implicit reference route;
- `STRUCTURED_NATIVE`: native radial tridiagonal or multidimensional sparse transport solves plus local source updates.

Both routes return `ParticleConversionStepResult`. Candidate and accepted states are distinct. Any solver, thermodynamic, transport, reaction, phase, admissibility, or balance failure rejects the whole conversion step.

Run `examples/particle_internal_heating.py` for a complete shell-heating workflow. `tools/particle_conversion_qualification.py` checks exact shell measures, balance closure, and agreement between the two solver routes.

## Limitations

Batch membership and shell count are static. The implementation is single-process. General multidimensional intraparticle meshes, moving internal interfaces beyond the scalar shrinking-core coordinate, and monolithic fluid–particle Newton systems are not implemented.
