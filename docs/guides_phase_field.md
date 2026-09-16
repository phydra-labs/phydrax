# Phase-field production

`phydrax.applications.phase_field` provides prepared finite-element Allen–Cahn,
Cahn–Hilliard, and dense grand-potential evolution with physical accepted-step
ledgers. The package also supplies wetting and driven boundary laws, periodic
constraints, arbitrary registered bulk potentials, heterogeneous blocks, tensor
mobility, fixed-capacity active phase storage, accepted hp/AMR transactions,
replayable stochastic forcing, and distributed ownership plans.

Nonisothermal solidification, anti-trapping, nucleation, mechanics, Model-H
flow, and electrostatic/electrochemical work are covered by the
[coupled phase-field multiphysics guide](guides_phase_field_multiphysics.md).

## Governing runtime contract

Every candidate reports the complete balance

$$
R_E = F_{n+1}-F_n + D_{\mathrm{bulk}}+D_{\mathrm{wall}}
      -W_{\mathrm{boundary}}-W_{\mathrm{source}}-W_{\mathrm{noise}}.
$$

Every conserved component separately reports

$$
R_M = M_{n+1}-M_0-M_{\mathrm{volume\ source}}-M_{\mathrm{boundary\ flux}}.
$$

A candidate is promoted only when its nonlinear root, constitutive laws, energy
ledger, component balances, capacity evidence, and numerical state all pass. Failure
retains the exact prior accepted state. No clipping, phase deletion, resampling, or
post-solve mass repair is used.

## Bulk potentials and time laws

`AbstractBulkFreeEnergy` now includes a discrete derivative and a current-state
incremental density. Registered potentials therefore own both their physical density
and their energy-compatible time law.

Built-in routes include:

- `DoubleWellFreeEnergy` with `ConvexSplitDoubleWellLaw`;
- `PolynomialBulkFreeEnergy` with an exact algebraic discrete gradient;
- `CallableBulkFreeEnergy`, which requires explicit density, derivative, discrete
  derivative, incremental density, domain, and stable identity;
- `DiscreteGradientBulkLaw` for arbitrary contract-compliant potentials.

An opaque callable without a discrete law is not a production potential.

```python
potential = phx.equations.PolynomialBulkFreeEnergy(
    (0.25, 0.0, -0.5, 0.0, 0.25)
)
model = phx.applications.phase_field.BinaryPhaseFieldModel(
    phx.equations.BinaryThermodynamicParameters(1.0, 0.05),
    closure=phx.equations.BinaryPhaseThermodynamicClosure(potential),
)
```

## Heterogeneous finite-element blocks

Prepared binary methods iterate every mesh block. `LocalGeometry` exposes stable block
and entity metadata to local functional densities; previous-state quadrature buffers
are selected by block identity. Quadrature, energy, mass, dissipation, and resolution
evidence are accumulated in deterministic block order.

Supported scalar H1 cell families are triangle, quadrilateral, tetrahedron,
hexahedron, prism, and pyramid when a compatible Lagrange element and quadrature route
exist. Mixed Cahn–Hilliard fields must use the same global layout.

`FiniteElementDiscretization.cell_block_domain(name)` exposes one exact global cell
selection for block-local actions.

## Wetting and imposed boundary work

`PhaseFieldBoundaryPlan` assigns phase physics to named exterior-facet sets without
creating a second mesh-boundary ownership convention.

Surface laws include:

- `PolynomialSurfaceEnergy`;
- `YoungAngleSurfaceEnergy`;
- `PrescribedMicrotractionEnergy`.

For wall energy $\gamma_w$ the natural condition is generated from the same
functional:

$$
\kappa\,\mathbf n\!\cdot\!\nabla\phi
 + \frac{\partial\gamma_w}{\partial\phi}=0.
$$

Time-dependent surface potentials contribute explicit boundary work to the accepted
ledger. `PrescribedPhaseFieldFlux` adds a Cahn–Hilliard boundary load, cumulative mass
source, and chemical work. Positive prescribed flux enters the modeled domain.

`BinaryThermodynamicParameters` contains bulk and gradient coefficients only.
Wetting is an explicit boundary law. Free-energy LBM exposes its numerical wall
strength separately from shared bulk thermodynamics.

## Periodic finite elements

`periodic_constraint` lowers `FiniteElementPeriodicFacetPair` metadata into a sparse
master-coordinate prolongation. It:

- maps owner coordinates through the declared affine transform;
- requires a bijective match on the neighbor facet;
- forms canonical equivalence classes by stable DOF ID;
- certifies injective reduced coordinates and constant reproduction;
- composes with the existing FE constraint substrate.

Scalar and componentwise-identity H1 phase fields are supported. Nonidentity component
transforms are rejected by the phase-field constraint route. A facet cannot
simultaneously be physical and periodic.

## Anisotropic mobility

`AbstractPhaseFieldMobility` evaluates one Onsager operator and its positivity
evidence. Implementations include:

- `ScalarPhaseFieldMobility`;
- `TensorPhaseFieldMobility`;
- `CallableTensorPhaseFieldMobility`.

For Cahn–Hilliard evolution,

$$
\mathbf j=-\mathbf M\nabla\mu,
\qquad
D=\Delta t\int_\Omega \nabla\mu\cdot\mathbf M\nabla\mu\,d\Omega.
$$

Tensor laws are symmetrized only after rejecting material antisymmetry above
roundoff tolerance. Negative spectrum rejects preparation or the attempted step.
Allen–Cahn kinetics remains scalar; a spatial tensor is not silently reinterpreted as
a local kinetic coefficient.

## Dense grand-potential evolution

`QuadraticGrandPotentialPhase` implements a thermodynamically consistent phase law:

$$
\omega(\boldsymbol\mu)
 = \omega_0 - \mathbf c_0\!\cdot\!\boldsymbol\mu
 - \tfrac12\boldsymbol\mu^T\boldsymbol\chi\boldsymbol\mu,
$$

with composition $\mathbf c=\mathbf c_0+\boldsymbol\chi\boldsymbol\mu$ and Helmholtz
density $f=\omega+\boldsymbol\mu\cdot\mathbf c$.

`GrandPotentialMaterialCatalog` requires distinct phase identities and one compatible
component basis. `GrandPotentialMixtureModel` combines phase interpolation, barrier
energy, gradient energy, phase kinetics, and component mobility.

`GrandPotentialFEMPlan` evolves vector phase logits and diffusion potentials on the
same FE mesh. Softmax phase weights preserve a dense simplex. Accepted steps gate
physical energy and every integrated component.

```python
phase_a = phx.applications.phase_field.QuadraticGrandPotentialPhase(
    "alpha", 0.0, c_alpha, chi_alpha
)
phase_b = phx.applications.phase_field.QuadraticGrandPotentialPhase(
    "beta", 0.0, c_beta, chi_beta
)
catalog = phx.applications.phase_field.GrandPotentialMaterialCatalog(
    (phase_a, phase_b)
)
model = phx.applications.phase_field.GrandPotentialMixtureModel(
    catalog,
    barrier_scale=barrier,
    gradient_coefficient=kappa,
    kinetic_coefficient=kinetic,
    mobility=mobility,
)
method = phx.applications.phase_field.GrandPotentialFEMPlan(model).prepare(
    discretization, "phase_logits", "diffusion_potential"
)
```

The qualified thermodynamic closure is isothermal. Temperature evolution, latent
heat, anti-trapping current, nucleation, elasticity, flow, and electrostatics require
separate coupled ledgers.

## Active multiphase storage

`ActivePhaseStoragePlan` stores fixed-capacity phase IDs and values per FE coordinate:

```text
phase_ids, values, active, dwell: (global_dofs, local_capacity)
```

IDs are canonical and `-1` is the only inactive sentinel. `KeyGroupPlan` constructs a
cell-local phase union in stable `(phase_id, DOF-slot)` order. Capacity overflow is
evidence and rejects the candidate; no phase is dropped.

`from_dense` and `dense` provide an exact reference bridge. `transition` aligns old and
candidate cell groups by logical phase ID and reports activation, pruning, topology
change, and capacity success. Global catalog size and local storage capacity remain
separate quantities.

## Accepted hp/AMR evolution

`PhaseFieldAdaptivityPlan` computes interface-gradient indicators and performs atomic
local T3 refinement using native adaptation maps and primal transfer. Cahn–Hilliard
transfers preserve the original mass reference and cumulative boundary source.
Transfer mass, energy, resolution, and finite-state evidence determine whether the
candidate epoch commits.

`PhaseFieldHPTransactionPlan` specializes the native fixed-capacity hp transaction,
mass projection, constraint, mortar, and rollback substrate. Structural changes occur
only after an accepted physical step.

`PhaseFieldAdaptiveEpoch` binds method, accepted state, and epoch identity. Failed
adaptation retains the complete source epoch.

## Replayable stochastic forcing

`PhaseFieldNoisePlan` maps a global `WienerRealization` through a finite-rank spatial
basis. Retries and subdivided intervals query the same global path. The realization
and basis identities enter the method identity.

Allen–Cahn accepts general basis increments. Cahn–Hilliard requires positive
conservation weights and removes the weighted constant mode, giving pathwise zero
mass increment under the declared discrete pairing.

The accepted ledger records realized stochastic work rather than requiring pathwise
energy decrease. `PhaseFieldNoisePlan.thermal` applies the constant-mobility factor
$\sqrt{2k_BTM}$; state-dependent stochastic mobility requires an additional drift law
and is not inferred.

## Distributed execution

`DistributedPhaseFieldPlan` composes native cost-aware cell partitioning,
owned/halo worksets, exactly-once facet ownership, and real JAX named-axis
collectives. It exposes distributed linear-operator wrapping and exactly-once cell
reductions.

`DistributedPhaseFieldCheckpointManifest` uses global topology, geometry, cell, DOF,
method, stochastic, and active-storage identities. Partition count is execution
provenance rather than physical identity, allowing a future restart to choose a new
partition after reconstructing global state.

The current released execution evidence covers the native partition/reference route.
A multihost support profile remains unreleased until exercised on an actual
multi-process JAX mesh.

## Stationary topological kink

`DoubleWellKinkPlan` is a separate finite one-dimensional stationary problem
using the canonical `DoubleWellFreeEnergy`. It fixes a uniform interval,
gradient coefficient, opposite vacuum Dirichlet traces, Newton/backtracking
work, residual tolerance, and dense-reference capacity. The solve reports the
Euler–Lagrange residual, gradient and bulk energy, total energy, boundary
residual, oriented topological sector, center residual, translational-mode
residual, stability spectrum, and negative-mode count.

The analytic tanh profile and energy `2 sqrt(2 kappa a) / 3` provide independent
controls. This finite interval result is not a universal soliton type,
infinite-domain proof, higher-dimensional defect model, or dynamical scattering
solver.

## Production identities and profiles

`phase_field_candidate_profiles()` declares exact support tuples for:

- deterministic general binary evolution;
- stochastic adaptive binary evolution;
- active grand-potential evolution;
- the fully integrated flagship combination.

`phase_field_released_profiles()` binds accepted, time-bounded release evidence
to those exact tuples. The integrated qualification campaign emits released
profile records only after every section, including a real two-device JAX
collective, passes. A multihost profile remains separate.

Prepared binary and grand-potential methods implement `AbstractFixedStepMethod` and
compose with `ProductionRunPlan` and durable checkpoints. Method identity binds model,
evolution law, mobility, boundary plan, noise realization, FE compilation, topology,
geometry, and precision.

## Qualification and benchmarks

Core binary qualification:

```console
PYTHONPATH=. JAX_ENABLE_X64=1 python \
  tools/phase_field_production_qualification.py
```

Integrated closure qualification:

```console
PYTHONPATH=. JAX_ENABLE_X64=1 \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
python tools/phase_field_extended_qualification.py
```

Performance evidence:

```console
PYTHONPATH=. JAX_ENABLE_X64=1 python \
  benchmarks/phase_field_production.py
PYTHONPATH=. JAX_ENABLE_X64=1 python \
  benchmarks/phase_field_extended.py
```

The integrated campaign covers arbitrary polynomial evolution, multiple blocks,
wetting, boundary work and mass flux, periodic constraints, tensor mobility, dense
grand-potential evolution, active storage, accepted AMR, stochastic replay,
distributed ownership, and exact capability-profile construction.

## Explicit nonclaims

The closure does not claim:

- opaque potentials without a discrete evolution law;
- nonidentity phase-component periodic transforms;
- simultaneous physical and periodic ownership of one facet;
- nonisothermal or anti-trapping solidification;
- nucleation;
- mechanics, flow, or electrostatic coupling;
- global classical differentiability through retries, active-set changes, AMR, or
  repartitioning;
- bitwise equivalence across different collective reduction layouts;
- multihost execution without an actual multihost qualification campaign.

Diffuse fracture remains in `phydrax.applications.fracture`; its irreversibility and
history contracts are different from phase evolution.

## General topological defects

`PolynomialDefectPotential` gives scalar or multifield stationary defects a
finite, source-identified potential. `MappedInfiniteDefectPlan` uses a rational
compactification of the real line, fixed vacuum traces, Newton/backtracking,
the full stability Hessian, translation-mode evidence, energy, and topological
charge. Radial defects, warm-started coefficient continuation, and damped
multifield Klein–Gordon scattering are separate plans. Relativistic scattering
does not reuse diffusive phase-field time evolution.
