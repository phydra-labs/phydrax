# Commercial MPM transfers and schedules

`ExplicitMPMMethodPlan` owns independent velocity-transfer, position-advection,
and stress-update schedule plans.

## Velocity transfer

Available MPM-specific plans:

```text
PICTransferPlan
FLIPTransferPlan
PICFLIPTransferPlan
APICTransferPlan
```

They do not reuse electromagnetic PIC or free-surface FLIP solver state.

```text
v_PIC = sum_i N_ip v_i_new

v_FLIP = v_p_old + sum_i N_ip (v_i_new - v_i_before)

v_new = pic_fraction v_PIC + (1 - pic_fraction) v_FLIP
```

APIC retains its affine `C` state. PIC/FLIP/blended plans set `C = 0` and exclude
affine momentum from P2G. `PICFLIPTransferPlan.pic_fraction` is fingerprinted with
the displayed convention.

Position advection is independently selected:

```text
PICAdvectionPlan
TransferredVelocityAdvectionPlan
MidpointAdvectionPlan
```

This prevents FLIP momentum update from silently choosing its own advection law.

## Schedule closure

In addition to USF, USL-minus, and classical translational MUSL:

- `AffineMUSLMPMSchedule` includes updated APIC affine momentum in its second
  pre-advection P2G.
- `PostAdvectionMUSLMPMSchedule` advances particle position, updates assignment
  input, rebuilds routes, repeats P2G/constraints, and records the second topology
  digest.

Every second grid reconstruction reapplies the complete simultaneous field,
rigid, and essential constraint transaction.

## Multifield transfer

A field plan owns one `KWayMPMContactPlan`. Each field receives independent mass,
momentum, force, mass-gradient, and transfer reconstruction. USF and all MUSL
variants update every field under the same accepted schedule. Pairwise
`project_two_field_contact` is no longer a public authority.

## Evidence

Transfer/schedule evidence includes mass, momentum, affine/angular behavior,
PIC dissipation, FLIP grid-delta behavior, second-P2G balance, second route digest,
constraint work/dissipation, field leakage, and schedule identity.

Energy and balance totals use the shared ordered twofold compensated reduction.
Its CUDA execution keeps the accumulation recurrence inside native GPU kernels,
including the common tangent and reverse actions. Block expansion, particle and
chunk ordering, and compensation are unchanged; this is not a replacement by an
unordered parallel sum. CPU and other platforms retain reference execution.
Compiler-bound execution artifacts must be reprepared when their primitive
identities change; particle state, transfer conventions, and physical evidence
retain their existing representations.

Small-matrix solves and local post-moment algebra use bounded elementwise
contractions so JAX can fuse value, diagnostic, and derivative calculations.
The same native small-solve owner handles refinement and conditioning for every caller.
Unused transfer fields can be eliminated by consumer compilation; failure
checks are not replaced by an assumed diagonal moment matrix. Axisymmetric,
B-bar, F-bar, masked-route, and inactive-particle semantics remain explicit.
Last-bit differences from generic GEMM reductions may change whether a
refinement correction is accepted; the reported count describes the executed
algorithm, not a historical branch decision.

Single-case Cartesian moment formation has a native CUDA streaming lowering.
Each GPU lane owns one particle/component and accumulates routes in storage
order, with no materialized route-product tensors. The admitted profile uses
float32 or float64, int32 indices, at most 27 routes and at most three physical
components/axes, on explicitly resolved Ampere-or-newer targets. The existing
boolean mask is read directly. Unused output moments and their input producers
are removed by the owning primitive's dead-code rule.

CPU, unknown or unsupported targets, wider stencils/axes, mixed precisions,
batched transformations, axisymmetric cases and B-bar retain their reference
execution. JVP, VJP and higher derivatives use the native JAX reference.
Streaming uses ordered scalar accumulation,
so last-bit agreement with the reference reduction tree is not promised.
Conditioning, refinement, failure status, and inactive-particle masks remain
owned by the existing APIC and small-solve contracts. Compiler and structural
sparsity admission recognize only the exact owned moment primitive.
