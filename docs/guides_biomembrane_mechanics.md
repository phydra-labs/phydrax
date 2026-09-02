# Biomembrane mechanics

PhydraX biomembranes are closed, outward-oriented triangular two-manifolds with stable vertex and face identifiers. `BiomembranePlan` owns topology, constitutive parameters, species kinetics, and fixed capacities. Calling `prepare(reference_positions)` performs exhaustive host-side edge and vertex-link manifold checks, orientation, nondegeneracy, oriented-normal, scale-aware self-intersection, and intrinsic-Delaunay transport-stencil checks before constructing a `PreparedBiomembrane` runtime. Static-topology evaluation, differentiation, surface transport, and thermal stepping are fixed-shape JAX programs.

## Mechanical model

For vertex dual areas $A_i$, face areas $A_f$, discrete twice-mean curvature $2H_i$, angle-defect Gaussian curvature $K_i$, and enclosed oriented volume $V$, the conservative potential is

$$
E = \frac12 \sum_i A_i\kappa_i(2H_i-C_i)^2
  + \sum_i A_i\bar\kappa_i K_i
  + \frac12\sum_f k_f\frac{(A_f-A_f^0)^2}{A_f^0}
  + \frac{k_A}{2A_*}(A-A_*)^2
  + \frac{k_V}{2V_*}(V-V_*)^2
  + \sigma A-pV+E_{\mathrm{adh}}.
$$

The spontaneous curvature can depend on surface composition,
$C_i=C_i^0+\sum_s \chi_s c_{is}$. The adhesion term is a smooth finite-range plane potential. Positive pressure acts outward because its potential is $-pV$. `PreparedBiomembrane.evaluate` computes the conservative nodal force as exactly `-jax.grad(total_energy)`. Active normal traction is assembled by consistent face-area-vector quadrature, reported separately, and then added to `force`; a constant traction therefore has exactly zero resultant on a closed mesh. Geometry evidence includes local/global area and volume residuals, minimum face area and oriented vertex-normal magnitude, finite/orientation status, and normalized conservative net-force and net-torque residuals.

The Gaussian term uses the closed-mesh angle defect. For constant Gaussian rigidity it evaluates the discrete Gauss–Bonnet invariant. The differentiable path is conditional on fixed topology; remeshing is an explicit nondifferentiable event.

## Surface species

`BiomembraneState.species_mass` stores nodal amounts, not concentrations. Every state also carries its static `prepared_id`; a runtime rejects states from another topology epoch even when capacities happen to match. Current concentration is amount divided by current barycentric dual area. `diffuse_react` uses the signed intrinsic-Delaunay cotangent finite-volume flux and applies the exactly column-conservative execution-dtype reaction matrix to concentrations. Flux is added with equal and opposite endpoint contributions.

The result contains both candidate and accepted states. A nonfinite, nonconservative, or negative candidate fails closed and returns the input amounts as `accepted_state`. `BiomembraneTransportEvidence` reports per-species before/after amounts, total conservation residual, minimum candidate amount, finite/positivity/conservation status, and acceptance. Choose an explicit step size small enough for the positivity limit of the explicit transport update.

## Thermal overdamped dynamics

`thermal_step` advances each coordinate with

$$
\Delta x_i=M_iF_i\Delta t+\sqrt{2k_BT M_i\Delta t}\,\xi_i,
\qquad \xi_i\sim\mathcal N(0,I).
$$

The supplied PRNG key is folded with both a stable preparation tag and `step_index`. Equal preparation, key, and index therefore produce an identical increment; changing topology changes the stream identity. Evidence exposes deterministic and stochastic increments and the expected coordinate variance. Invalid candidate geometry or nonfinite data rolls back to the exact source state.

## Immersed-boundary fluid composition

`couple_immersed_boundary` directly accepts the existing `ImmersedBoundaryForcingPlan`. Current dual areas become marker measures, membrane positions become marker positions, and the requested membrane velocity is the no-slip target. `membrane_force` is the hydrodynamic load and is exactly the negative of the force spread to the fluid; `mechanical_force` is the membrane evaluation load and `total_force` is their explicit sum. The nested forcing result retains interpolation convergence, partition-of-unity, work, and force-ledger evidence.

```python
from phydrax.applications.cellular_mechanics import BiomembranePlan

plan = BiomembranePlan(
    faces,
    bending_rigidity=1.0,
    spontaneous_curvature=0.0,
    global_area_modulus=20.0,
    volume_modulus=20.0,
    species_diffusivity=(0.05, 0.01),
    reaction_matrix=((-0.2, 0.1), (0.2, -0.1)),
)
membrane = plan.prepare(reference_positions)
state = membrane.state(species_mass=species_amount)
evaluation = membrane.evaluate(state)
transport = membrane.diffuse_react(state, 1.0e-3)
```

## Transactional remeshing

Split, collapse, and flip operations are host-side candidate/evaluation/commit transactions:

1. `propose_split`, `propose_collapse`, or `propose_flip` resolves an edge by its two stable vertex IDs.
2. The proposal constructs a candidate only if exhaustive edge and vertex-link manifold, opposite-edge-orientation, scaled positive-volume, oriented-normal, intrinsic-Delaunay, and shared-entity-aware triangle-intersection guards pass.
3. Nodal species amounts, area-integrated material fields, and per-face local rest area are transferred by a conservative nearest-support measure map. Existing entity IDs survive; split entities receive monotone new IDs; parent-ID arrays record lineage. Removed collapse-patch material remains local instead of being distributed over the whole surface.
4. `evaluate_remesh` reports finite signed and relative area, volume, energy, species, and material jumps and applies caller-selected jump limits. Its identity fingerprints both source and candidate states, so evidence cannot be reused after composition changes.
5. `commit_remesh` requires valid source and candidate states and returns the candidate only when every guard and limit accepts it. Rejection returns the exact source preparation and state objects.

A successful transaction always has a new `prepared_id`, even when the represented surface is geometrically unchanged. Compiled functions must be prepared again after commit because their capacities and topology are intentionally static.

```python
proposal = membrane.propose_split(state, (vertex_id_a, vertex_id_b))
evidence = membrane.evaluate_remesh(
    proposal,
    maximum_relative_area_jump=0.02,
    maximum_relative_volume_jump=0.02,
    maximum_relative_energy_jump=0.05,
)
transaction = membrane.commit_remesh(proposal, evidence)
membrane, state = transaction.prepared, transaction.state
```

## Evidence and differentiation contract

All compiled results carry `prepared_id` and explicit finite/valid/successful status. Status must be checked before consuming a candidate. Conservative mechanics and fixed-topology diffusion are differentiable with respect to coordinates and numerical state. Topology selection, remesh guards, remesh commit, stable-ID allocation, and self-intersection classification are host decisions and are not differentiated. No gradient is claimed across a remesh event.
