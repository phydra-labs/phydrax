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
- uniform square tensors for fixed-rank TRG/HOTRG partition-function coarse graining;
- tensor train/operator for general discrete tensors, functions, and structured SciML;
- Abelian tensors for exact U(1), cyclic, or product-charge conservation;
- graded tensors only when exchange signs are part of the algebra;
- reduced SU(2) tensors only when irrep/fusion-tree structure is explicit;
- labeled topology and contraction schedules for arbitrary finite networks;
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

### Explicit local executors and path planning

`phydrax.tensor_network` owns topology, schedules, precision, admission, and
scientific evidence. `prepare_contraction(plan, operands, execution=...)` selects
a local executor; installing optional libraries never changes the default
`"native"`. Ordinary `phydrax.ein.contract` remains the exact upstream
opt-einsum function: there is no global interception or provider switch.

The execution request has exactly these constructor controls:

```python
tn.ContractionExecutionPolicy(
    "native",  # alternatively "cutensor" or "cutensornet"
    maximum_workspace_bytes=2**28,
    maximum_cache_bytes=2**30,
    maximum_cache_entries=16,
    maximum_batch_size=64,
    maximum_derivative_bytes=2**31,
)
```

All bounds are positive integers below `2**63`; `maximum_cache_entries` must
also be at most 4096. `policy_id` is computed, not a constructor argument.
These defaults describe the API, not a memory recommendation for every job.
The runnable example below deliberately requests smaller external budgets.

`plan_contraction` accepts **one** of `optimizer="greedy"`/`"optimal"`, an
imported `path`, or a `ContractionPathPlanner` passed as `path_planner`.
Leaving all three unspecified selects native greedy planning. Imported paths
index the evolving live operand list, not permanent topology node numbers.
Every path is validated and replayed into a native schedule with native
shape/FLOP/liveness/resource admission. Path provenance is separate from
executor identity.

```python
tn.ContractionPathPlanner(
    maximum_planning_seconds=30.0,
    maximum_search_iterations=64,
    maximum_workspace_bytes=2**28,
    seed=0,
)
```

This request selects shape-only cuTensorNet planning with a FLOP objective,
without automatic slicing. It runs in a spawned worker under a whole-worker
wall deadline; it passes neither numerical arrays nor live handles to that
worker. Iterations must be in `[1, 10000]`, workspace a positive integer below
`2**63`, time finite and positive, and seed an integer in `[0, 2**31)`.
`planner_id` is computed. The distinct `ContractionPlannerPolicy` controls
native search (`maximum_search_states`, `maximum_planning_seconds`), not vendor
search iterations.

Changing only `execution` while retaining `plan` compares executors on the
**same path**. Replanning with `path_planner` changes the algorithmic path;
execute that new plan natively or with an explicitly selected provider, but
do not call the difference a kernel-only speedup. Neither route promises a
faster result.

### Runnable native workflow with an optional local CUDA executor

Save this block as a Python script. Running it without arguments uses only the
native executor and native greedy planner. `--provider cutensor` or
`--provider cutensornet` explicitly selects a CUDA executor; the corresponding
optional provider and native component must be installed (see
[installation](installation.md)). `--plan-with-cutensornet` independently
requests optional path planning, even with `--provider native`. Missing
providers, unsupported hardware, or exceeded budgets are errors, not silent
fallbacks. The process guard is required for spawned planning.

```python
import argparse

import jax
import jax.numpy as jnp
from phydrax import tensor_network as tn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", choices=("native", "cutensor", "cutensornet"), default="native"
    )
    parser.add_argument("--plan-with-cutensornet", action="store_true")
    args = parser.parse_args()

    structure = tn.ContractionStructure(
        (
            tn.ContractionOperand(
                "left", (tn.ContractionLeg("row", 4), tn.ContractionLeg("k", 6))
            ),
            tn.ContractionOperand(
                "center", (tn.ContractionLeg("k", 6), tn.ContractionLeg("l", 5))
            ),
            tn.ContractionOperand(
                "right", (tn.ContractionLeg("l", 5), tn.ContractionLeg("column", 3))
            ),
        ),
        ("row", "column"),
    )
    resources = tn.ContractionResourcePolicy(
        maximum_operand_elements=1024,
        maximum_intermediate_elements=1024,
        maximum_output_elements=128,
        maximum_workspace_bytes=2**26,
        maximum_flops=1_000_000,
        maximum_schedule_steps=32,
    )
    path_planner = (
        tn.ContractionPathPlanner(
            maximum_planning_seconds=10.0,
            maximum_search_iterations=16,
            maximum_workspace_bytes=2**20,
            seed=0,
        )
        if args.plan_with_cutensornet
        else None
    )
    plan = tn.plan_contraction(
        structure,
        dtype="float32",
        resources=resources,
        planner=tn.ContractionPlannerPolicy(
            maximum_search_states=10_000, maximum_planning_seconds=10.0
        ),
        path_planner=path_planner,
    )
    operands = (
        jnp.arange(24, dtype=jnp.float32).reshape(4, 6) / 24,
        jnp.arange(30, dtype=jnp.float32).reshape(6, 5) / 30,
        jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 15,
    )
    if args.provider != "native":
        # Explicit placement; provider preparation never transfers implicitly.
        device = jax.devices("gpu")[0]
        operands = tuple(jax.device_put(value, device) for value in operands)
    execution = tn.ContractionExecutionPolicy(
        args.provider,
        maximum_workspace_bytes=2**20,
        maximum_cache_bytes=2**20,
        maximum_cache_entries=4,
        maximum_batch_size=4,
        maximum_derivative_bytes=2**26,
    )
    prepared = tn.prepare_contraction(plan, operands, execution=execution)
    result = tn.execute_contraction(prepared)
    sliced_plan = tn.plan_sliced_contraction(
        plan,
        ("k",),
        batch_size=2,
        resources=tn.SlicingResourcePolicy(
            maximum_slices=16,
            maximum_batch_elements=4096,
            maximum_checkpoint_bytes=2**16,
        ),
    )
    binding = tn.prepare_sliced_contraction(prepared, sliced_plan)
    sliced = tn.execute_sliced_contraction(binding, mode="batched")
    print(result.value)
    print(result.evidence.provider, result.evidence.accepted)
    print(sliced.value, sliced.evidence.accepted)


if __name__ == "__main__":
    main()
```

To import a fixed path instead of searching, replace the planning argument
`path_planner=path_planner` with `path=((0, 1), (0, 1))` and leave `optimizer`
unspecified. The first step combines `left` and `center`; after removing them,
the live list contains `right` followed by their result.

### Preparation, differentiation, and resource boundaries

`prepare_contraction` and `refresh_contraction` are host-only content-addressed
boundaries; traced operands are rejected. Refresh preserves the plan and
executor, rebinds current arrays, and advances `numeric_version`. Numerical
values are never reused from a provider cache. For sliced execution, explicitly
bind the current prepared contraction using
`prepare_sliced_contraction(prepared, sliced_plan)` after preparation or refresh.
This admits the residual executor once with the same execution policy; the
residual preserves the original path rather than launching another search.
`execute_sliced_contraction(binding, ...)` takes that binding, not a plan and
an unbound operand tuple.

Prepared ordinary contractions admit `float32`, `float64`, `complex64`, and
`complex128` (fp32/fp64/c64/c128). This is an intended dtype admission contract,
**not hardware qualification** of every dtype, transform, topology, or device.
Precision roles remain explicit: optional executors require equal resolved
multiply and accumulation dtypes and refuse wider accumulation rather than
silently relabeling it. cuTENSOR admits unary/binary SSA steps; cuTensorNet
fixed-path execution requires binary evolving-live-list paths. Unsupported
multi-input steps are not silently split.

Native JAX execution keeps compositional differentiation. Optional executors
use native-owned multilinear JVP/VJP rules, bounded serial device-local
batching, and derivatives through order two; higher derivative requests are
refused. Complex contraction is bilinear, with no implicit Hermitian
conjugation. Differentiate dynamic numerical leaves of an already prepared
object, not host preparation, path/rank selection, or fingerprint construction.
`execute_slice_assignment(binding, assignment)` is the pure local kernel for
JAX transformations. `execute_sliced_contraction`, checkpoint construction,
checkpoint merge, and distribution are host orchestration, **not
differentiable APIs**.

`maximum_workspace_bytes` on the execution policy bounds external provider
workspace; `maximum_cache_bytes` bounds retained external device scratch/cache
across its partition, while `maximum_cache_entries` bounds descriptor count.
`maximum_batch_size` limits mapped lanes and `maximum_derivative_bytes`
admits provider transformation payloads, including retained network/layout/cast
buffers. These are not a hard bound on XLA-managed native autodiff memory.
The native plan's separate `ContractionResourcePolicy` still admits the whole
contraction; a small native workspace limit can reject an otherwise valid
external-cache budget. Inspect `external_workspace_bound_bytes`,
`external_cache_bound_bytes`, `vendor_host_bound_bytes`, and
`derivative_contract` in execution evidence. Opaque vendor host descriptor
storage has an **unknown** byte bound (`None`), not a zero-byte allocation.
Release and shutdown retain the same accounting contract. Native cleanup verifies
the owning CUDA device, drains submitted work, checks every vendor destruction
status, and restores the prior device. If scratch or opaque objects cannot be
shown released, the partition stays poisoned and charged; no later idempotent
release can manufacture successful teardown evidence.

The optional route is for explicitly prepared ordinary dense contractions.
Abelian charge blocks, fermionic grading, reduced SU(2) tensors, and TRG/HOTRG
remain with their native representation/algorithm owners; installing an
executor neither redirects those algorithms nor implicitly densifies tensors.
Local executor support does not qualify multi-GPU peer transfers, collectives,
or multi-host operation.

Source contracts: [planning, preparation, refresh, and evidence](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_contraction.py),
[execution/planner policy fields](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_contraction_provider.py),
[provider admission and memory evidence](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_cutensor.py),
[native-owned transformation rules](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_contraction_ad.py),
[bounded shape-only planner](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_cutensornet.py),
[sliced binding and host boundary](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_slicing.py), and
[native TRG/HOTRG execution](https://github.com/phydra-labs/phydrax/blob/main/phydrax/tensor_network/_tensor_renormalization.py).

## Tensor renormalization

`UniformSquareTensor` represents one rank-four tensor repeated on an infinite
square lattice with axes `(up, right, down, left)`. Dense `TRGMethod` and
`HOTRGMethod` execution uses fixed retained ranks, preplanned contractions,
explicit normalization and terminal corrections, local discarded-weight
evidence, and fail-closed positive-real partition semantics. This is a
thermodynamic-limit partition-density route, not general infinite-network
execution or a global truncation-error certificate. See
[Tensor renormalization](guides_tensor_renormalization.md).

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

The platform does not claim arbitrary infinite-network execution, data-dependent
output ranks inside JIT, derivatives through rank/path/branch selection, general
PEPS error bounds, generic non-Abelian categories, arbitrary anyons, or automatic
best-method selection.
