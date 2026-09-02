# Tensor platform

Phydrax tensor computation is a family of explicit representations, not one universal
array. Finite quantum chains, tensor trains, locally purified states, conserved-charge
tensors, arbitrary networks, and projected entangled-pair states share private numerical
kernels only where their mathematical contracts are genuinely identical.

## Representation selection

Choose the representation from the problem, not from a fallback hierarchy:

- finite MPS/MPO for open one-dimensional quantum systems;
- LPDO for positive semidefinite mixed states represented by purification;
- uniform MPS/MPO for an injective finite unit cell in the thermodynamic limit;
- tensor train/operator for general discrete tensors, functions, and structured SciML;
- Abelian tensors for exact U(1), cyclic, or product-charge conservation;
- graded tensors only when exchange signs are part of the algebra;
- reduced SU(2) tensors only when irrep/fusion-tree structure is explicit;
- labelled topology and contraction schedules for arbitrary finite networks;
- finite rectangular PEPS/PEPO for two-dimensional open-boundary workloads.

Unsupported combinations fail during planning. Phydrax never changes representation,
precision, rank, planner, device, normalization, or physicality route silently.

## Plans, preparation, refresh, and execution

Persistent workloads follow four boundaries:

1. **Plan** fixes topology, shapes, capacities, method, precision, resources, and evidence
   layout.
2. **Prepare** binds numerical operators and validates structural/physical contracts.
3. **Refresh** changes numerical leaves only when all structural identities still match.
4. **Execute** performs fixed-shape JAX computation and returns value plus evidence.

Host effects such as archive I/O, admission, checkpoint publication, telemetry, and
provider communication remain outside transformed numerical functions.

## Precision and approximation

Tensor-network precision separates storage, contraction, factorization, accumulation,
decision/certification, and output roles. Every approximation reports its own evidence:

- retained and available rank;
- local and certified global discarded weight when a proof applies;
- canonical/gauge residual;
- local eigensolver or matrix-function residual;
- projected/Galerkin residual;
- energy variance;
- norm or trace drift;
- CP, TP, PSD, covariance, causality, charge, grading, or category residual as relevant;
- planned and observed resource use.

Accumulated local loss is not described as a global error bound unless the algorithmic
assumptions establish one. Rank and bond dimension are resource capacities, not accuracy
claims.

## Finite chains

Finite-chain workflows include local-term and string MPO construction, compression,
expectations, reduced states, correlations, entanglement, transfer diagnostics, DMRG,
real/imaginary-time TDVP, thermal purification, excited states, and time-domain response.

DMRG acceptance requires more than an energy plateau: local solver status, true projected
residual, variance, canonical residual, Hermiticity evidence, and truncation floor remain
separate. TDVP records site/bond exponential errors, tangent projection, normalization,
and time-order defects. A time-dependent Hamiltonian has a fixed operator structure and
refreshes coefficients only.

Finite-temperature purification starts from the infinite-temperature physical-ancilla
state and evolves to half inverse temperature. Raw normalization provides the partition
weight; normalized energy and thermodynamic identities are reported separately.

## Uniform chains

Uniform states use a finite unit cell and transfer normalization rather than a finite
global norm. The initial released scope is injective states with an isolated dominant
transfer eigenvalue. Transfer fixed-point residuals, left/right normalization, center
equations, projected VUMPS residual, energy/variance density, and correlation length are
part of every result. Noninjective or unresolved transfer spectra produce typed refusal.

## Tensor trains and quantics

`phydrax.tensor_train` is domain-neutral and does not inherit quantum-state normalization.
TT-SVD and rounding report per-cut loss and the root-sum-square Frobenius bound where the
canonical theorem applies. Tensorized-grid and quantics layouts explicitly state radix,
axis order, significance order, interleaving, endpoint, and coordinate conventions.

TT-cross is a bounded oracle algorithm. It reports queried fibers, pivot conditioning,
evaluation count, rank history, and held-out estimator; it does not claim a uniform error
bound. Structured tensor-train operators cover declared Cartesian shifts, Kronecker
sums/products, finite-difference operators, diagonals, and boundary treatments. ALS and
AMEn-like solvers recompute the true global residual because compression makes local
truncating actions inexact.

## Quantum experiments and open systems

`QuantumProgram` remains the deterministic local-map IR. Instruments and
`QuantumExperimentProgram` add finite outcomes, bounded classical state, static
feed-forward, and addressed typed randomness. Shot replay is independent of batching.
Pure MPS branches reject multi-Kraus mixed outcomes; LPDO branches preserve positivity by
purified construction and report CP, TP, trace, bond, and purification evidence.

Hardware compilation records logical mapping, native decomposition, SWAP or interval
routing, and approximation cost. No hardware or provider call occurs inside numerical
execution. Time-dependent controls have a fixed grid and basis. Process-learning
workflows use native Stinespring/Stiefel parameterizations and bind held-out likelihood,
gauge, physicality, and checkpoint lineage.

## Symmetry and statistics

Abelian group charge, fermionic grading, and non-Abelian representation categories are
separate layers.

An Abelian tensor block is legal only when the oriented charge sum equals the declared
total charge. Allocation capacities are static; active support is a fixed-length dynamic
vector. Basis order, allocation, support, and numerical-value identities are distinct.
Global sector truncation includes every singular value, including modes beyond an
individual sector capacity.

A fermion grading is an explicit homomorphism into Z2. It generates Koszul signs for odd
exchanges and never follows merely from a Z2 charge label. Mode order and permutation
plans are structural. Jordan-Wigner and native graded lowering are explicit alternatives.

Reduced SU(2) tensors store degeneracy data with fusion-tree metadata and deterministic
Clebsch-Gordan/recoupling identities. Truncation retains complete multiplets. Generic
non-Abelian or anyonic support is not inferred from SU(2).

## Arbitrary networks and contraction

Topology, planning, and execution are separate immutable artifacts. Topology supports
explicit outputs, traces/diagonals, hyperedges, scalar nodes, and deterministic builders.
A contraction schedule is an inspectable SSA-style DAG with liveness, exact shapes,
FLOPs, live memory, objective, planner provenance, and deterministic ID.

Slicing has a fixed mixed-radix order and deterministic accumulation policy. Resource
admission includes resident operands, live intermediates, outputs, accumulators, transfer
buffers, and safety reserve. Multi-device execution begins with independent exact slices;
distributed failure returns no accepted aggregate.

## PEPS, CTMRG, trees, and MERA

Finite rectangular OBC PEPS/PEPO is distinct from a generic topology object. Exact small
contraction is the reference route. Boundary-MPS contraction, CTMRG, simple update, and
full update expose separate approximation claims and evidence. Simple update never
claims a full environment. Full update never falls back to simple update after a failed
solve.

Tree tensor networks use exact message passing. Loopy tensor-network belief propagation
reports residual and Bethe-style diagnostics without a global error bound. Binary MERA
maintains isometries through native manifold operations.

## Production runtime

A support tuple binds representation, geometry, symmetry/statistics, algorithm, sizes,
capacities, precision, device/mesh, differentiation, determinism, and resource policy.
Production admission verifies the tuple before allocation. Checkpoints are accepted only
at declared atomic boundaries and bind structure, plan, policy, numerical revision, key
schedule, histories, evidence, and parent digest.

Telemetry contains IDs, status, durations, resource aggregates, and residual summaries;
it excludes tensor values, customer paths, wire names, physical parameters, and secrets.
Archives are pickle-free and bounded before payload allocation. Optional interchange
rejects ambiguous basis, leg order, gauge, category, or physicality metadata.

## Maturity and limitations

Every capability is experimental, qualified, or released for a named support tuple.
CPU, accelerator, multi-device, and multi-host support are qualified independently. A
successful run outside a published support tuple does not promote that tuple.

The platform does not claim infinite-dimensional execution, data-dependent output ranks
inside JIT, derivatives through rank/path/branch selection, general PEPS error bounds,
generic non-Abelian categories, arbitrary anyons, or automatic best-method selection.
