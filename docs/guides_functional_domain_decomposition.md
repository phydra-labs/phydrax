# Functional domain decomposition

Phydrax represents neural domain decomposition through solver-neutral covers, local
`DomainFunction` families, explicit interface traces, and typed training strategies.
The public API describes the mathematics directly rather than selecting a paper by
name.

## Decomposition axes

Four choices are independent:

- **cover:** overlapping or non-overlapping local domains;
- **field assembly:** partition of unity or a side-aware broken field;
- **coupling:** value, flux, overlap, or user-authored transmission residuals;
- **training:** joint, block Jacobi, block Gauss–Seidel, or Schwarz-local updates.

The combinations recover common methods:

| Method family | Phydrax construction |
| --- | --- |
| FBPINN | overlapping cover + partition-of-unity field + joint training |
| cPINN | broken field + value/physical-flux interface terms |
| XPINN | broken space-time field + authored state/residual interface terms |
| DeepDDM/Schwarz | broken local fields + pair terms + Schwarz training |
| two-level local PINN | frozen coarse field + partition-of-unity correction |

Residual continuity is a numerical coupling condition; it is not automatically a
physical conservation law. Flux conditions require an explicit physical flux.

## Covers and pairings

`cartesian_subdomain_cover` partitions `ScalarInterval`, `Interval1d`, and
`HyperRectangle` factors. Scalar counts create uniform partitions; explicit
`AxisPartition` values create nonuniform tensor-product partitions with independent
overlap and periodicity. It constructs:

- exact local patch domains in one, two, or three dimensions;
- ambient/local coordinate maps;
- fixed C2 tensor-product compact windows when overlap is positive;
- exact face parameterizations, measures, and canonical normals;
- shared and periodic paired supports;
- exact coverage and maximum-overlap evidence.

```python
cover = phx.domain.cartesian_subdomain_cover(
    domain,
    "x",
    4,
    overlap_fraction=0.2,
)
```

A general `SubdomainCover` can be assembled from `SubdomainPatch` and
`PairedSupport`. Every pairing owns one sampling component and maps it into both
patches. Both traces therefore consume the same points and integration weights.
The optional normal is oriented from the left patch to the right patch.

`SubdomainCover.audit(points)` is sampled evidence. It never upgrades a user-defined
cover to an exact proof. Built-in Cartesian covers expose exact structural evidence.

## Local fields

`LocalFieldFamily` binds one logical field to one local `DomainFunction` per patch.
Each parameter array has one canonical owner.

```python
local_fields = {}
for patch, key in zip(cover.patches, jr.split(key, len(cover.patches)), strict=True):
    model = phx.nn.models.MLP(
        in_size=1,
        out_size="scalar",
        width_size=16,
        depth=2,
        key=key,
    )
    local_fields[patch.patch_id] = patch.domain.Model("x")(model)

family = phx.domain.LocalFieldFamily("u", cover, local_fields)
```

`normalized_patch_coordinate(patch, label)` returns an unclipped affine local
coordinate in `[-1, 1]` on the patch. It may be used to pull a reference-domain model
onto physical coordinates. Clipping is intentionally absent because it would corrupt
boundary derivatives.

## Partition-of-unity fields

`partition_of_unity_field(family)` constructs the ambient field

```text
u(x) = sum_i chi_i(x) u_i(x) / sum_i chi_i(x).
```

Only active local functions are executed. The prepared maximum overlap bounds the
fixed JAX route. Evaluation returns nonfinite output rather than silently truncating
if runtime multiplicity exceeds that capacity.

Apply the PDE operator to the assembled field. Blending already evaluated local PDE
residuals is generally wrong, including for linear differential operators because
window derivatives contribute.

A partition-of-unity problem uses ordinary global terms:

```python
problem = phx.solver.FunctionalDecompositionProblem.partition_of_unity(
    family,
    terms=(pde_term, boundary_term),
)
plan = phx.solver.FunctionalDecompositionPlan(
    phx.solver.JointDecompositionTraining(2000),
    required_window_regularity=2,
)
prepared = phx.solver.prepare_functional_decomposition(problem, plan)
result = phx.solver.solve_functional_decomposition(
    prepared,
    optax.adam(1.0e-3),
)
```

`result.global_field` is an ordinary ambient `DomainFunction`.

## Broken fields and interfaces

A broken problem exposes stable local solver names with
`family.field_name(patch_id)`. Local and pair terms receive explicit scopes.

```python
jump = phx.conditions.SubdomainValueJump(
    family.field_name(pairing.left_patch_id),
    family.field_name(pairing.right_patch_id),
    pairing,
)
pair_term = phx.solver.ScopedFunctionalTerm(
    phx.terms.ResidualPenalty(jump, interface_source),
    phx.solver.PairScope(pairing.pairing_id),
)
problem = phx.solver.FunctionalDecompositionProblem.broken(
    family,
    terms=(*local_terms, pair_term),
)
```

Available interface conditions:

- `SubdomainValueJump`;
- `SubdomainFluxJump` with left and right physical flux callbacks;
- `SubdomainTransmission` for a general paired residual;
- `subdomain_overlap_consistency` for a codimension-zero pairing.

`SubdomainFluxJump` differentiates or otherwise forms each local flux before pulling
it onto the interface. It then contracts the flux difference with the canonical
normal.

`result.broken` preserves local values and side-specific traces. Conversion to an
ambient pointwise field requires an explicit ownership policy:

```python
field = result.broken.as_domain_function(ownership="first")
```

At an unresolved interface this policy chooses the first canonical patch. It does
not claim continuity.

## Block and Schwarz training

`BlockDecompositionTraining` minimizes incident patch, pair, and global terms while
updating one exact local parameter subtree. Every patch owns an independent Optax
state.

- Jacobi reads one immutable sweep snapshot and merges disjoint local updates.
- Gauss–Seidel lets later patches observe earlier updates from the same sweep.
- Colored scheduling updates one deterministic parameter-conflict color at a time.
- Active and fixed patch IDs support explicit local schedules.

Block training supports broken and partition-of-unity fields. POU parameter ownership
is resolved inside the assembled evaluator. `IntegrationOwnership` supplies a
separate non-negative partition of the physical residual measure.

`SchwarzDecompositionTraining` requires broken fields and paired terms. Preparation
creates fixed interface batches. `SchwarzTraceState` retains outgoing traces,
relaxed incoming targets, interface revisions, and the maximum fixed-point defect.
Relaxation acts on trace values, never neural parameter vectors.

```python
plan = phx.solver.FunctionalDecompositionPlan(
    phx.solver.SchwarzDecompositionTraining(
        sweeps=20,
        inner_iterations=50,
        sweep="gauss-seidel",
        relaxation=0.7,
        interface_tolerance=1.0e-6,
    ),
    trace_points=128,
)
```

Block and Schwarz strategies use the shared `FunctionalUpdateKernel`, preserving
one optimizer state per local parameter block. Standard Optax transformations are
supported; transformations requiring update-time line-search arguments remain a
separate backend route.

Use `max_sweeps` to stop at an accepted sweep boundary. Save and restore the exact
local parameters and optimizer states with
`save_functional_decomposition_checkpoint` and
`load_functional_decomposition_checkpoint`, then pass the restored state back to
`solve_functional_decomposition`.

## Coarse correction

`FunctionalCoarseCorrection` freezes an existing global solution and trains one
partition-of-unity correction through Phydrax's exact nonlinear defect-correction
substrate.

```python
correction = phx.solver.FunctionalCoarseCorrection(
    coarse_solver,
    "u",
    fine_family,
)
trained = correction.training_solver.solve(
    num_iter=1000,
    optim=optax.adam(1.0e-3),
)
physical_solver = correction.finalize(trained)
```

The base field is unchanged. The physical solver evaluates the complete coarse plus
fine field against the original unscaled residual objective.

## Advanced coupling and local residuals

`MortarInterfacePenalty` projects a fixed paired value jump into a user-supplied
interface basis and applies the inverse discrete Gram matrix. Singular mortar bases
are rejected.

`NitscheInterfaceFunctional` implements the symmetric scalar consistency and
stabilization functional for authored physical fluxes. Its penalty scale is explicit;
Phydrax does not infer a stable scale from an arbitrary strong residual.

`AugmentedValueConstraint` owns fixed interface multipliers, primal residual, dual
residual, and accepted multiplier updates. `solve_augmented_interface` alternates
functional minimization and multiplier updates.

`LocalTestSpace` and `LocalizedResidualNorm` provide hp-VPINN-style fixed local test
spaces. The residual moments are weighted by the inverse discrete Gram matrix, so a
nonorthonormal basis is not silently treated as orthonormal.

## Hierarchies, adaptation, sharding, and deployment

`SubdomainHierarchy` holds arbitrary ordered partition-of-unity correction levels.
`train_functional_hierarchy` trains them coarse-to-fine against the complete
nonlinear residual. `FunctionalCyclePlan` adds repeated V- and F-cycle visit orders.

`prepare_adaptive_topology_transaction` validates a candidate cover and transfers
the represented ambient field through a frozen field-preserving pullback. A topology
change commits only when coverage and transfer-error gates pass.
`refine_axis_partition` and `coarsen_axis_partition` provide explicit h changes.
`TrainableAxisPartition` parameterizes cell widths through a positive simplex with a
strict minimum width; materialization remains an accepted host-side topology epoch.

Periodic Cartesian partitions use wrapped local supports and fixed periodic
coordinate maps, so overlapping POU windows remain smooth across the seam.

`FunctionalDecompositionShardingPlan` assigns every patch to an explicit JAX device.
`place_local_field_family` places local array trees, while
`place_schwarz_trace_state` moves incoming traces to their target devices and reports
cross-device payload bytes. `distributed_pou_collective` uses an actual device-axis
sum, and `distributed_schwarz_exchange` routes fixed-shape traces through a device
collective.

`SchwarzTraceQuantity` exchanges authored value, derivative, flux, traction, or
characteristic fields. `aitken_relax_trace_state` applies bounded Aitken
delta-squared acceleration in trace space. `solve_asynchronous_schwarz` executes a
deterministic patch schedule against explicitly bounded stale trace revisions.

`HybridFunctionalDecomposition` accepts trainable functional participants and fixed
external/numerical field participants. `DecompositionDeploymentArtifact` preserves
the cover, parameter content, local fields, and explicit POU or broken-owner assembly
through a content-identified archive.

`LocalCurvaturePlan` supplies a bounded exact dense local Newton route through
Phydrax linear algebra. `solve_local_kfac` and `solve_overlap_kfac` train broken-field
local blocks with native KFAC. `matrix_free_gauss_newton_step` applies the damped
normal operator through JVP/VJP products and a Phydrax conjugate-gradient solve,
without assembling a Jacobian or Hessian.

## Evidence and limitations

`FunctionalDecompositionEvidence` reports:

- final training objective;
- independently authored evaluation objective;
- per-patch values;
- per-pair values and maximum pair defect;
- cover verification;
- certification against an optional tolerance.

Certification requires evaluation terms. Completing optimizer work alone does not
certify a PDE solution.

Current limitations:

- wrapped periodic overlap is currently available for Cartesian interval/box factors;
- arbitrary geometry requires explicit patch and pairing maps;
- mortar and augmented constraints currently use fixed interface designs;
- Nitsche support is the symmetric scalar authored-flux form;
- asynchronous Schwarz uses a deterministic bounded-staleness host schedule rather
  than nondeterministic multi-host progress;
- local and overlap KFAC operate on explicit broken-field blocks; an assembled POU
  KFAC layout is not yet available;
- trainable partitions materialize at accepted host epochs rather than changing
  compiled topology inside an optimizer update;
- collectives use fixed-shape JAX device axes and do not provide process-failure
  recovery;
- automatic physical-flux inference remains intentionally unsupported.

## References

- [Finite Basis Physics-Informed Neural Networks](https://arxiv.org/abs/2107.07871)
- [FBPINNs as a Schwarz domain decomposition method](https://arxiv.org/abs/2211.05560)
- [Multilevel domain-decomposition PINN architectures](https://arxiv.org/abs/2306.05486)
- [Parallel cPINN and XPINN](https://arxiv.org/abs/2104.10013)
- [DeepDDM](https://arxiv.org/abs/2004.04884)
