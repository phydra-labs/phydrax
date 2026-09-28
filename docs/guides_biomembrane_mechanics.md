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

`BiomembraneState.species_mass` stores nodal amounts, not concentrations;
`BiomembranePlan.species_ids` gives every species axis an explicit scientific
identity. Every state also carries its static `prepared_id`; a runtime rejects
states from another topology epoch even when capacities happen to match.
Current concentration is amount divided by current barycentric dual area.
`diffuse_react` uses the signed intrinsic-Delaunay cotangent finite-volume flux
and applies the exactly column-conservative execution-dtype reaction matrix to
concentrations. Flux is added with equal and opposite endpoint contributions.

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

Biomembrane mechanics remains owned by `BiomembranePlan`; only its topology
transaction and field transfer use
`phydrax.geometry.multiregion_surface`. Preparation creates an explicit finite
interior/ambient pair under the `manifold_two_region` validation profile. That
profile refuses extra regions, borders, non-manifold edges, singular vertex
fans, disconnected finite regions, and self-intersections. The default
`MultiRegionSurfaceCapacityPlan` reserves headroom for subsequent events; pass
an explicit `remesh_capacity` when a workflow needs a different fixed resource
bound.

Split, collapse, and flip are shared `EdgeSplitProposal`,
`EdgeCollapseProposal`, and `EdgeFlipProposal` transactions:

1. `propose_remesh` converts the current coordinates, species amounts, and
   area-integrated vertex mechanics fields to one capacity-shaped shared event
   state.
2. E3 resolves stable vertex IDs, applies its link/feature/normal guards,
   restores enclosed volume, certifies every motion by inclusion CCD, validates
   the complete candidate with exact predicates, and commits or returns the
   unchanged source surface.
3. One sparse sheet-slot route transfers every nodal extensive field. A sparse
   face route transfers reference area and reference-area-weighted local
   modulus. Intensive mechanics coefficients are reconstructed by dividing
   transferred content by the candidate dual measure. No dense
   `N_new`-by-`N_old` transfer is formed.
4. The proposal exposes shared pass status, exact validation, conservative
   transfer evidence, stable `MultiRegionSurfaceLineage`, and sparse vertex and
   face epoch transitions. `evaluate_remesh` additionally reports membrane
   area, volume, energy, species, and material jumps under caller-selected
   limits.
5. `commit_remesh` returns the new `PreparedBiomembrane` epoch only when both
   shared certification and membrane mechanics admission pass. Any failure
   returns the exact source preparation and state objects.

Prepared geometry and sparse routes are reused for every proposal in one
epoch. A successful transaction advances the topology epoch and prepares the
new fixed-topology mechanics runtime. Event evidence always reports
`derivative_available=False`; fixed-topology mechanics and transport remain
differentiable separately.

```python
from phydrax.geometry.multiregion_surface import EdgeSplitProposal

proposal = membrane.propose_remesh(
    state, EdgeSplitProposal((vertex_id_a, vertex_id_b))
)
evidence = membrane.evaluate_remesh(
    proposal,
    maximum_relative_area_jump=0.02,
    maximum_relative_volume_jump=0.02,
    maximum_relative_energy_jump=0.05,
)
transaction = membrane.commit_remesh(proposal, evidence)
membrane, state = transaction.prepared, transaction.state
```

The executable smoke is `examples/advanced_biomembrane_remeshing.py`.
`tools/biomembrane_remesh_qualification.py` exercises all three event kinds,
conservation, sparse-route scaling, epoch evidence, and rollback.
`benchmarks/biomembrane_remesh_scaling.py` records lowering, compilation,
warmed execution, compiler memory, retained sparse-route memory, and host
transaction time across declared vertex capacities.

## Evidence and differentiation contract

All compiled results carry `prepared_id` and explicit finite/valid/successful status. Status must be checked before consuming a candidate. Conservative mechanics and fixed-topology diffusion are differentiable with respect to coordinates and numerical state. Topology selection, remesh guards, remesh commit, stable-ID allocation, and self-intersection classification are host decisions and are not differentiated. No gradient is claimed across a remesh event.
