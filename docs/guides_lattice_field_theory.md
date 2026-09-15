# Lattice field theory

Phydrax represents a finite regulated lattice model by composing canonical topology, field-space, geometry, action, sampler, and observable owners. It does not define a universal quantum-field state or solver.

The initial platform covers:

- real scalar Euclidean `phi4` actions;
- compact angle-valued U(1) Wilson measures;
- fundamental-representation U(N) and SU(N) Wilson actions;
- local Metropolis updates where an exact local action cache exists;
- fixed-step Hamiltonian Monte Carlo on a flat torus or product compact matrix Lie group;
- raw-scale autocorrelation diagnostics for sampled observables.

The action is always finite dimensional. Continuum, infinite-volume, gauge-fixing, dynamical-fermion, and QCD claims require separate approximation evidence and are not inferred.

## Scalar fields on cochains

A scalar field is a real degree-zero cochain. The action

```text
S[phi] =
    kinetic_scale / 2 * (d0 phi)^T H1 (d0 phi)
  + sum_v dual_volume_v * (mass_squared * phi_v^2 / 2
                            + quartic_coupling * phi_v^4 / 4)
```

uses the prepared cochain exterior derivative and Hodge/Riesz operator.

```python
import jax
import jax.numpy as jnp
import phydrax as phx

plan = phx.discretization.TensorGridPlan(
    (
        phx.discretization.UniformCellAxisSpec(8, periodic=True),
        phx.discretization.UniformCellAxisSpec(8, periodic=True),
    ),
    axis_names=("x", "y"),
)
grid = plan.prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
cochain = phx.discretization.StructuredCochainBridge(grid).cochain

action = phx.operators.path_integral.Phi4LatticeAction(
    cochain,
    kinetic_scale=1.0,
    mass_squared=-0.4,
    quartic_coupling=0.8,
)
field = jnp.zeros(action.configuration_shape)
value = action.action(field)
```

Negative `mass_squared` is valid when a positive quartic term bounds the action. With zero quartic coupling, the current all-vertices-dynamic representation claims normalizability only for positive `mass_squared`.

`prepare_local_phi4_action` requires a diagonal degree-one Hodge map. A non-diagonal Hodge map remains valid for full-action evaluation and Hamiltonian sampling, but does not claim site-local action differences.

```python
local_action = phx.operators.path_integral.prepare_local_phi4_action(action)
target = phx.operators.path_integral.incremental_target_from_lattice_action(
    local_action,
    refresh_cadence=32,
)
proposal = phx.sampling.SingleCoordinateGaussianProposal(0.2)
kernel = phx.sampling.MetropolisHastings(proposal)
initial = jnp.zeros((4,) + action.configuration_shape)
state = kernel.initialize(target, initial)
```

Every candidate cache is pure. Rejection selects the untouched current cache; periodic refresh recomputes the exact action and invalidates a drifted cache.

## Per-configuration observables

`LatticeObservablePlan` records topology, field-space, normalization, output shape, and real/complex semantics independently from ensemble reduction.

```python
plans = phx.operators.path_integral.phi4_observable_plans(action)
values = tuple(
    phx.operators.path_integral.evaluate_lattice_observable(plan, field)
    for plan in plans
)
```

Susceptibility and Binder ratios are constructed from sampled magnetization moments. They are not single-configuration observables.

`phi4_pair_correlation_plan` accepts explicit source/target pairs and weights, so structured translation averages and arbitrary graph pair averages share one contract.

## Correlated-chain diagnostics

`correlated_observable_diagnostics` accepts real samples with leading `(chain, draw)` axes. It computes zero-padded FFT autocovariances and Geyer's initial-monotone paired-sequence window.

The reported convention is

```text
tau_int = 1 + 2 * sum_{lag > 0} rho_lag
effective_sample_size = total_draws / tau_int
```

```python
diagnostics = phx.uq.correlated_observable_diagnostics(
    observable_samples,
    policy=phx.uq.CorrelatedObservablePolicy(max_lag=256),
)
```

This raw-scale diagnostic complements rank-normalized R-hat and bulk/tail ESS. Complex observables must be diagnosed through separately declared real and imaginary components.

## Compact U(1)

`CompactU1GaugeMeasure` consumes one validated `CellComplexTopology`; callers do not pass duplicate dense incidence matrices.

```python
topology = phx.discretization.polygonal_cell_complex(
    jnp.asarray([[0, 1, 2]]),
    None,
    3,
)
measure = phx.operators.path_integral.CompactU1GaugeMeasure(
    topology,
    beta=0.7,
)
links = jnp.zeros((measure.num_edges,))
```

Link angles use `FlatTorusStateGeometry` and principal coordinates in `[-pi, pi)`. The reduced action is

```text
S_reduced = -beta * sum_p cos(theta_p)
```

The conventional Wilson action is returned by `canonical_action`; its difference from the reduced action is recorded by `evidence.additive_constant`.

Local wrapped updates compose with the generic Markov lifecycle:

```python
target = phx.operators.path_integral.incremental_target_from_lattice_action(
    measure,
    refresh_cadence=32,
)
proposal = phx.sampling.SingleCoordinatePeriodicProposal(
    2.0 * jnp.pi,
    0.6,
)
kernel = phx.sampling.MetropolisHastings(proposal)
state = kernel.initialize(target, jnp.zeros((4, measure.num_edges)))
```

The U(1) `wilson_loop` function uses integer oriented-edge coefficients. Abelian multiplication commutes, so it does not require the ordered non-Abelian path representation.

## Ordered paths and matrix gauge links

Non-Abelian holonomy depends on traversal order. `prepare_cell_boundary_paths` recovers and validates one ordered closed traversal for every selected two-cell. `CellBoundaryPathPlan` proves that each ordered path reproduces the canonical oriented boundary incidence.

```python
boundaries = phx.discretization.prepare_cell_boundary_paths(topology)
group = phx.metrix.SpecialUnitaryGroup(2)
link_space = phx.graph.MatrixGaugeLinkSpace(topology, group)
links = link_space.identity()
holonomy = phx.graph.path_holonomy(link_space, links, boundaries.paths)
```

For an oriented edge from `x` to `y`, gauge transformations use

```text
U_x_to_y -> g_x U_x_to_y inverse(g_y)
```

Open-path holonomy is covariant at its endpoints. The normalized trace of a closed path is invariant.

`MatrixGaugeLinkSpace` references one canonical `DiscreteFieldSpace` and applies `PointwiseStateGeometry(LieGroupStateGeometry(...))` over the edge axis. It does not introduce another field or topology hierarchy.

## Matrix Wilson actions

```python
wilson = phx.operators.path_integral.WilsonGaugeAction(
    link_space,
    boundaries,
    plaquette_couplings=0.8,
)
```

The supported action is the fundamental normalized-trace convention

```text
S_reduced = -sum_p beta_p * real(trace(U_p)) / N
```

A scalar coupling broadcasts to all selected plaquettes; an explicit vector supplies anisotropic couplings. The action stores the conventional additive constant separately.

`WilsonGaugeAction` has an exact local cache, but the initial platform does not expose an SU(N) local proposal or heatbath. Its local transition is available for independently normalized future proposal kernels and for cache qualification.

## Compact-group Hamiltonian Monte Carlo

A lattice action lowers to a geometric log target explicitly:

```python
target = phx.operators.path_integral.compact_geometric_target_from_lattice_action(
    wilson
)
kernel = phx.sampling.prepare_compact_group_hamiltonian_kernel(
    target,
    step_size=0.08,
    leapfrog_steps=4,
)
state = phx.sampling.initialize_compact_group_hamiltonian_state(
    kernel,
    jnp.stack((link_space.identity(), link_space.identity())),
)
result = phx.sampling.sample_compact_group_hamiltonian(
    kernel,
    state,
    key=jax.random.key(0),
    num_draws=100,
)
```


The sampler supports only:

- `FlatTorusStateGeometry` relative to normalized flat angular measure;
- pointwise left/body `LieGroupStateGeometry` relative to normalized product Haar measure.

For a Lie-group coordinate basis `T_a = hat(e_a)`, momentum uses the exact Gram matrix

```text
G_ab = real(trace(adjoint(T_a) T_b))
```

The implementation factorizes one algebra-size matrix and broadcasts it over links. It never materializes a global link-by-link mass matrix.

The frozen production sampler reports acceptance probability, Hamiltonian error, divergence, nonfinite state, membership failure, and used leapfrog steps. `adapt_compact_group_hamiltonian` performs a finite Robbins-Monro step-size warmup and returns a frozen kernel. NUTS, learned metrics, force splitting, gauge fixing, and distributed link halos are outside the current claim.

## Qualification and nonclaims

`benchmarks/lattice_field_theory.py` exercises:

- a free scalar ensemble against the exact inverse quadratic covariance;
- interacting scalar full/local target agreement and magnetization autocorrelation;
- a one-plaquette U(1) expectation against `I1(beta) / I0(beta)`;
- SU(2) and SU(3) Wilson actions and group HMC;
- chain, membership, energy, and runtime evidence.

A finite lattice result is not a continuum or infinite-volume result. Users must vary lattice spacing and volume independently and preserve chain diagnostics for every ensemble. Gauge redundancy is retained; no implicit gauge fixing or Faddeev-Popov factor is applied.

## Hamiltonian Z2 gauge theory

Hamiltonian gauge-model recipes live in
`phydrax.applications.lattice_field`; the application layer composes canonical
topology, quantum-register, local-Hamiltonian, tensor-network, and solver
owners.

`Z2GaugeModel` uses one qubit per edge in the electric computational basis:

```text
G_v = product of Z_e over edges incident to v
B_p = product of X_e over the boundary of p
H = -h sum_e Z_e - K sum_p B_p
```

```python
model = phx.applications.lattice_field.Z2GaugeModel(
    topology,
    electric_coupling=0.4,
    magnetic_coupling=1.2,
)
sector = phx.applications.lattice_field.prepare_z2_gauss_sector(model)
hamiltonian = phx.applications.lattice_field.z2_gauge_hamiltonian(model)
```

`prepare_z2_gauss_sector` enumerates computational basis states only below an
explicit resource limit. GF(2) arithmetic is host-side topology/constraint
analysis; runtime states remain ordinary JAX arrays and qubits. Redundant
vertex constraints are counted by exact binary rank. `z2_homology` uses the
canonical topology package with `PrimeField(2)` for topological labels.

This is a finite-register reference path, not a scalable gauge-invariant MPS
sector implementation.

## Open Schwinger chains

`SchwingerChainModel` implements the open 1+1-dimensional staggered spin
encoding

```text
Q_j = (Z_j + (-1)^j I) / 2
L_n = L_left + L_external,n + sum_{j <= n} Q_j
H_electric = a g^2 / 2 * sum_n L_n^2
```

with nearest-neighbor hopping `(X X + Y Y) / (4 a)` and reduced staggered
mass `m (-1)^j Z_j / 2`.

```python
schwinger = phx.applications.lattice_field.SchwingerChainModel(
    6,
    lattice_spacing=0.4,
    mass=0.3,
    gauge_coupling=0.8,
    left_boundary_flux=0.1,
)
local = phx.applications.lattice_field.schwinger_local_hamiltonian(schwinger)
mpo = phx.applications.lattice_field.schwinger_mpo(schwinger)
```

The small-chain local expansion has quadratic term count and an explicit
`maximum_terms` guard. The electric MPO uses
`build_prefix_quadratic_mpo`; its prefix-square bond dimension is at most
three independently of chain length.

`schwinger_background_schedule` lowers knot-valued external fluxes to
`FixedStructureMPOCoefficients`. `schwinger_local_background_schedule`
provides the corresponding piecewise-constant `FixedGridLocalHamiltonian`
for resource-admitted small chains. Existing product-formula and TDVP solvers
remain the only evolution owners.

The elimination is explicitly open-boundary and one-dimensional. It does not
claim periodic, higher-dimensional, or non-Abelian link elimination.

`benchmarks/lattice_hamiltonian.py` cross-checks the Z2 Gauss sector,
Hamiltonian-constraint commutators, dense/local/MPO Schwinger agreement,
constant-bond prefix construction, DMRG, TDVP, and Gauss-law reconstruction.
