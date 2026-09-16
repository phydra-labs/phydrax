# Quantum lattices

The quantum-lattice package separates a finite physical specification from conserved-sector coordinates and from each execution target. It does not select a backend.

## Conventions

- `QuantumLatticeSpecification.spaces` is the canonical tensor-product site order.
- Every fermionic specification also carries one `FermionModeOrder`. Fermionic signs are computed from that order, not inferred from graph edges or term order.
- `QuantumLatticeTerm.factors` is an ordered operator product. The rightmost factor acts first. `add_adjoint=True` represents `c M + conj(c) M†` and is the preferred construction for hopping terms.
- Local matrices use row-output, column-input coordinates. A `LocalOperatorPlan` certifies one integral charge change on every nonzero matrix element.
- Boson `cutoff` is the retained local dimension, so its occupations are `0, ..., cutoff - 1`.
- A `SectorChargeMap` names the only source and target sectors of an operator. Compilation refuses terms whose charge changes do not all equal that map.

## Compile and act in a direct sector

```python
import numpy as np
from phydrax.operators.quantum import FermionModeOrder
from phydrax.operators.quantum.lattice import (
    FixedCardinalityFermionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    prepare_quantum_lattice,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)

order = FermionModeOrder(("left", "right"))
left = LocalSpacePlan.fermion("left", "left")
right = LocalSpacePlan.fermion("right", "right")
create = np.array([[0.0, 0.0], [1.0, 0.0]])
annihilate = create.T
hop = QuantumLatticeTerm(
    (
        LocalOperatorPlan(left, "create", create, (1,)),
        LocalOperatorPlan(right, "annihilate", annihilate, (-1,)),
    ),
    coefficient=-1.0,
    add_adjoint=True,
    label="nearest-neighbor-hop",
)
specification = QuantumLatticeSpecification(
    (left, right), (hop,), fermion_mode_order=order
)
prepared = prepare_quantum_lattice(
    specification,
    QuantumLatticeResourcePolicy(
        maximum_terms=4,
        maximum_factors_per_term=2,
        maximum_branches_per_input=16,
        maximum_sector_dimension=32,
        maximum_workspace_bytes=100_000,
    ),
)
basis = FixedCardinalityFermionBasis(
    order,
    1,
    resources=SectorBasisResourcePolicy(
        maximum_dimension=32,
        maximum_table_bytes=10_000,
    ),
)
operator = QuantumSectorOperator(prepared, SectorChargeMap(basis, basis, 0))
```

`FixedCardinalityFermionBasis`, `FixedSpinProjectionBasis`, and `FixedBosonNumberBasis` calculate their dimensions from bounded dynamic-programming tables. `coordinate(index)` and `rank(coordinate)` are direct. `QuantumSectorOperator.mv` iterates only the admitted source sector, scatters directly into the target sector, and declares no dense-materialization capability.

Use `phydrax.linalg.eigen.Eigenproblem(operator)` and the normal `phydrax.linalg.eigen.eigensolve` planning/execution API for ground or excited states. There is no quantum-lattice eigensolver wrapper.

## Finite-group orbit sectors

`FiniteGroupActionPlan` projects one existing direct charge sector into a
one-dimensional unitary character sector. A
`MonomialConfigurationGenerator` declares a complete site permutation, local
state bijections, unit phases, its finite order, and any fermionic site order.
Preparation enumerates the complete admitted group action, verifies the
declared generator orders and character, and constructs normalized orbit
embeddings. A character that is incompatible with a stabilizer produces no
basis vector rather than an artificial zero-norm state.

`prepare_quantum_orbit_sector_operator` first verifies that the compiled
Hamiltonian commutes with every prepared group action over the complete direct
sector. It then projects and coalesces fixed source-to-target routes. Runtime
action uses only those routes; it does not enumerate the ambient tensor-product
basis and intentionally has no dense-materialization capability. Complex
momentum phases and fermionic permutation signs are retained explicitly.

The initial support tuple is deliberately bounded to finite groups,
one-dimensional characters, direct conserved sectors, and endomorphisms.
Higher-dimensional irreducible representations and cross-character operators
require separate basis-gauge and multiplicity contracts. A finite symmetry
reduction is not a thermodynamic-limit or phase-identification claim.


## Explicit targets

Targets are selected by importing their owner:

- `phydrax.solver._quantum_lattice.lower_quantum_lattice_to_local_hamiltonian`
- `phydrax.tensor_network._quantum_lattice.lower_quantum_lattice_to_mpo`
- `phydrax.operators.quantum.lattice.lower_quantum_lattice_to_vmc`

The LocalHamiltonian target admits each bounded local support matrix before constructing it. The MPO target inserts Jordan--Wigner parity strings using the specification's mode order and admits its exact bond/tensor storage before construction. The VMC target exposes `H[current, connected]`; it conjugates the compiler's outgoing `H[connected, current]` values only after requiring a self-adjoint sector operator.

## Periodic one-particle bridge

`periodic_finite_to_fermion_lattice` accepts only a typed `PeriodicFiniteRealization`, an exact `FermionModeOrder`, realization labels equal to that order, and a mandatory `FermionInteractionPlan`. An empty interaction tuple is an explicit noninteracting statement and still requires units and provenance. The bridge coalesces sparse one-particle entries, verifies Hermiticity, and creates CAR terms. It never infers interactions, sectors, spin order, or a tensor-network/VMC target from a one-particle matrix.

## Canonical TPQ and response

`phydrax.solver._thermal_pure_quantum` uses fixed-sector random-phase probes and native Lanczos matrix-exponential actions under explicit retained-state and Krylov-workspace byte limits. Results retain every raw probe and thermal vector. Partition and observable estimates are ratios of probe sums; jackknife errors are computed from the same retained probe records. A beta-zero random-phase probe has exact squared norm equal to the sector dimension.

`phydrax.operators.quantum._response` defines explicit source/target probe, zero-temperature, and finite-temperature plans. `phydrax.solver._quantum_response` provides:

- zero-temperature retarded response through the existing shifted-system solver, with raw solve residuals, spectral moments, and positivity evidence;
- finite-temperature TPQ forward and reverse time correlations, probe-level statistical errors, moments, spectral positivity, and canonical-sector KMS evidence including the source/target partition-function ratio.

A finite response plan requires symmetric time and frequency grids, a caller-supplied nonnegative symmetric time window, and explicit result/workspace limits. TPQ and response require the Phydrax Lanczos matrix-function policy at a stopped differentiation boundary because probe choice, Krylov rank, and stopping are not differentiable physics. A zero-temperature plan can require physical frequency-window coverage. Insufficient or oversized windows are refused rather than clipped.

## Candidate stochastic evidence

`SignFreeStochasticCandidatePlan` admits fixed chain, draw, state-width, observable, expansion-order, autocorrelation-lag, and raw-byte counts. `assess_sign_free_stochastic_candidate` retains every raw chain state, expansion order, observable, phase/sign, and acceptance decision and reports the order histogram, phase autocorrelation, signed effective samples, and reweighted covariance. It succeeds only for finite unit phases consistent with the sign-free tolerance and sufficient effective samples.

This is a method-specific candidate control. It is not a stochastic solver, an unrestricted scaling claim, a thermodynamic-limit claim, or a solution to the sign problem. Candidate profiles returned by `quantum_lattice_candidate_profiles` are unreleased and carry no qualification evidence.

## Lifecycle artifacts

Direct bases, prepared finite-group actions, orbit bases, and matrix-free
sector operators use `write_quantum_lattice_artifact_archive`; TPQ and
zero-/finite-temperature results use `write_quantum_result_archive`. Their read
counterparts require the matching caller-prepared type and structure,
preserving raw probes, correlations, residuals, masks, PRNG state, and
source/target IDs without serializing an operator provider. See the
[production-evidence guide](guides_condensed_matter_production_evidence.md).

## Evidence utilities

`tools/cm_quantum_lattice_qualification.py` emits one tiny candidate-only CAR,
charge, Hermiticity, matrix-free, and VMC connection record.
`tools/cm_quantum_symmetry_qualification.py` retains the complete direct and
character-sector matrices, embeddings, spectra, invariance residuals, and
projection residuals for a three-site periodic spin control. Both include the
exact unreleased profile declarations and are not release evidence.

`benchmarks/cm_quantum_lattice.py` measures direct-sector construction and
explicit target lowerings. `benchmarks/cm_quantum_symmetry.py` separates group
closure, orbit preparation, reduced-route construction, JAX compilation, and
steady reduced action. `benchmarks/cm_quantum_thermal_response.py` records TPQ
and response costs together with raw numerical/statistical errors, positivity,
moments, and KMS residuals. No benchmark is a scientific release gate.
