# Adaptive multidimensional particle interiors

Particle interiors now share one mesh contract across radial, two-dimensional, and three-dimensional finite-volume descriptions.

## Mesh contract

`AbstractParticleInternalMeshPlan` prepares a fixed-capacity mesh with cell measures, face measures, owner/neighbour routes, boundary faces, stable identities, and geometry evidence. Implementations include:

- `RadialShellMeshPlan`;
- `UnstructuredParticleInternalMeshPlan` for triangles, quadrilaterals, and tetrahedra.

`ParticleInternalBatchPlan` uses `cell_capacity`; conversion state remains extensive in internal energy and species amount.

Unstructured meshes are defined in body coordinates. Scale changes update cell and face Jacobians. Three-dimensional boundary traces map face centers and normals through the body quaternion without changing body-frame measures.

## Conservative transport

Every active interior face is evaluated once. Heat and species fluxes are scattered with opposite signs to owner and neighbour cells. Boundary transfer is distributed over active boundary faces and deposits exact opposite extensive content into the continuum.

The conversion solver dispatches to:

- the radial tridiagonal backend for radial meshes;
- a native unstructured implicit solve through `phydrax.linalg` for multidimensional meshes;
- the reference Rosenbrock route for authority comparisons.

## Local AMR

`ParticleInternalAMRState` stores coarse and fine extensive fields plus per-particle active-leaf masks. `ParticleInternalAdaptationPolicy` uses separate refine/coarsen thresholds and dwell hysteresis.

`adapt_particle_internal_mesh` conservatively transfers:

- internal energy;
- every species amount;
- pore volume;
- internal reactive surface area.

Reaction progress is transferred with bounded conservative averaging. Overflow returns a growth request and preserves the old state.

Coarse/fine accepted flux mismatches are applied once through `apply_particle_internal_flux_correction` and the existing unstructured AMR register.

## Surface coupling

`ParticleBoundaryTrace` exposes boundary-face positions, normals, measures, and owner cells. Continuum exchange distributes source to boundary cells. Contact heat selects the boundary cell aligned with the resolved contact direction. Radiation is distributed by boundary-face area.

## Differentiation

Transport and remap values remain differentiable under a fixed active-leaf route. Refine/coarsen selection and overflow decisions are stopped-gradient topology events recorded in the adaptation route digest.

Run `examples/adaptive_catalyst_pellet.py` and `tools/particle_internal_amr_qualification.py`.
