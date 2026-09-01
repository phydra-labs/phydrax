# Isogeometric support and qualification

The `IGA.Core.Tensor` capability profile is an allow-list, not a claim that every
Cartesian product of the rows below is supported. A run is inside the R1 profile
only when its exact `SupportTuple` appears in a passing qualification manifest.
An axis value appearing in this table does not make unlisted combinations valid.
Unsupported combinations fail during planning or preparation; Phydrax does not
silently lower them to a different basis, interface, precision, or backend.

## R1 axis vocabulary

| Axis | R1 values | Qualification boundary |
| --- | --- | --- |
| geometry | `full_dim_1d_poly`, `full_dim_2d_nurbs`, `full_dim_3d_nurbs` | Regular, untrimmed, full-dimensional tensor-product maps only. Geometry checks are sampled evidence, not a global injectivity proof. |
| space | `scalar_H1`, `vector_H1`, `mixed_H1` | Geometry and each field may use independent tensor grids. No H(div), H(curl), shell, or manifold pullback is implied. |
| basis | `direct_tensor`, `extracted_bernstein` | Both are explicit realizations. Selection never changes the mathematical space. |
| formulation | `mass`, `diffusion_reaction`, `source`, `linear_elasticity`, `thermoelasticity`, `generalized_eigen` | Only the exact field and coefficient contracts exercised by an allow-listed case are qualified. |
| interface | `none`, `periodic_self` | `periodic_self` identifies opposite traces of one patch. Multipatch coupling is outside R1. |
| backend | `cpu`, `gpu` | A backend is supported only for tuples whose manifest was produced on that backend. There is no backend inheritance. |
| precision | `float64` | R1 has no float32 or mixed-precision support tuple. |
| distributed | `single` | No multi-process or distributed decomposition is claimed. |
| derivative | `Q0`, `Q1`, `Q2` | The tuple records the highest qualified derivative contract for that exact case; lower orders must still be listed or implied by the registry's exact policy, never by callers. |
| restart | `none`, `native` | `native` means checksum-validated Phydrax lifecycle manifests with immutable numeric-revision lineage. It is not an interchange format. |

## Canonical R1 case families

Every qualification output records one manifest per case. Each manifest contains
`case_id`, the exact `SupportTuple`, `producer_id`, plan/prepared/numeric/execution
identities, `qualification_policy_id`, immutable input and artifact digests,
metrics, per-gate status and threshold, and diagnostics. It contains no schema or
version field.

| Case family | Contract exercised |
| --- | --- |
| `core.affine.1d` | Full-dimensional 1D polynomial geometry and scalar H1 mass/diffusion/source execution. |
| `core.affine.2d.anisotropic` | Independent per-axis degrees and spans on an affine 2D map. |
| `core.affine.3d` | Full-dimensional 3D tensor topology, geometry, integration, and scalar execution. |
| `core.rational.quarter_annulus` | Positive-weight rational 2D geometry and exact physical-coordinate reproduction. |
| `core.rational.cylinder_or_quarter_pipe` | Positive-weight rational 3D geometry. The producer records which geometry variant it executed. |
| `core.independent_geometry_field_overlay` | Independent geometry and field grids integrated through an explicit common overlay. |
| `core.periodic_fourier` | Self-periodic trace identification and Fourier-mode mass/diffusion behavior. |
| `core.transfer.h_insert` | H-refinement transfer and field preservation. |
| `core.transfer.degree_elevate` | Degree-elevation transfer and field preservation. |
| `core.transfer.k` | Combined degree elevation and knot insertion with explicit transfer lineage. |
| `core.direct_extracted_parity` | Direct tensor and extracted Bernstein realizations of one space agree to the declared tolerance. |
| `core.matrixfree_partial_assembled_parity` | Matrix-free and partial-assembly actions agree for a fixed probe. This does not claim a public global sparse backend. |
| `core.elasticity.patch.2d` | Vector-H1 small-strain linear-elasticity patch behavior. |
| `core.elasticity.patch.3d` | Three-dimensional vector-H1 small-strain linear-elasticity patch behavior. |
| `core.eigen.laplacian_cluster` | Generalized mass/stiffness eigenspace with cluster/subspace evidence rather than eigenvector-sign comparisons. |
| `core.restart.numeric_refresh` | Immutable numeric refresh, checkpoint ancestry, identity checks, checksum validation, and replay. |
| `core.geometry_negative.W` | Nonpositive or inadmissibly small rational denominator is rejected. |
| `core.geometry_negative.rank` | Rank-deficient geometry Jacobian is rejected. |
| `core.geometry_negative.orientation` | Orientation failure is rejected where orientation is defined by the profile. |

The transfer rows are representation-preservation evidence, not local adaptive
spline support. The direct/extracted row proves parity for its declared tuple,
not universal parity for every degree, coefficient type, or backend. The
performance producer is record-only: elapsed time, compilation time, logical
bytes, and operation rates are observations and never release gates.

## S1 migration boundary

The former S1 workflow is retained as a migration fixture: a regular untrimmed
2D single patch, equal clamped axis grids, positive mean-one-gauge NURBS weights,
one isoparametric scalar H1 field, explicit Gauss quadrature, homogeneous strong
trace constraints, and matrix-free sum-factorized diffusion/source execution.
The fixture must preserve its coefficient layout, exact quadratic Poisson result,
prepared execution, runtime numeric refresh, geometry diagnostics, and natural
boundary measure. R1 documentation and examples use the R1 topology/atlas/
realization vocabulary; the fixture prevents semantic drift rather than keeping
a second implementation stack or deprecated public aliases.

## Failure and remediation

| Failure stage | Evidence | Required remediation |
| --- | --- | --- |
| request/profile lookup | No exact support tuple is allow-listed. | Change the request to a listed tuple or qualify a new tuple. Do not relabel an existing manifest. |
| topology construction | Degree, knot multiplicity, coefficient shape, periodic identification, or axis data are inconsistent. | Rebuild `SplineSpanTopology` with coherent immutable axis topology. Numeric refresh cannot repair topology. |
| atlas construction | Patch dimension, control-net shape, weights, or physical dimension disagree with topology. | Correct the `PatchAtlas` input and create a new plan. Do not reshape or broadcast control data implicitly. |
| geometry preparation | Rational denominator, rank, or orientation gate fails. | Repair control points/weights or refine the evidence policy explicitly and re-run qualification. Never reuse evidence from another numeric revision. |
| overlay preparation | Geometry and field partitions do not admit the declared common integration overlay. | Build an explicit `IntegrationOverlay` covering both partitions; do not integrate on either side's cells by assumption. |
| realization preparation | Extraction connectivity/operator shape disagrees with the space topology. | Recompute `ExtractedBernsteinRealization` from the exact field topology, or select `DirectTensorRealization` explicitly. |
| trace/interface preparation | Periodic traces have incompatible coefficient layouts or orientation. | Make the identified traces compatible and rebuild the constraint/trace route. Do not truncate or pad coefficients. |
| transfer preparation | Source/target lineage or preservation gates fail. | Rebuild the `TransferPlan`/`RefinementTransaction` from the actual source topology; retain the old field until the transaction commits. |
| execution | Plan, prepared, numeric, or execution identity differs from the manifest. | Prepare and execute the exact declared plan/revision, then produce new evidence. IDs are evidence, not user-editable labels. |
| restart | Checkpoint is incomplete, a shard digest differs, or analysis/execution identity mismatches. | Reject the checkpoint, recover an intact ancestor, and emit a new immutable `NumericRevision` and child checkpoint. Never patch a prior manifest. |
| release gate | Any case/gate is failed, missing, stale, unsigned, or from an unlisted environment. | Keep the profile unreleased, run the deterministic producer for every required tuple, review diagnostics, and let the release pipeline sign assembled evidence. |

## Evidence and release discipline

`tools/iga_r1_qualification.py` is the numeric evidence producer and
`tools/iga_r1_benchmarks.py` is the record-only timing producer. The
qualification producer writes deterministic JSON with nonfinite values rejected.
It never signs evidence and never marks `IGA.Core.Tensor` released.
`tools/iga_r1_profile.py` assembles only an unreleased candidate profile from
complete, passing manifests. Release status belongs to the qualification
registry and requires separately produced `ReleaseGateEvidence`; editing JSON or
running the profile producer cannot create it.
