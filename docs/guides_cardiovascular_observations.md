# Cardiovascular observation metadata and operators

The cardiovascular observation layer turns normalized host arrays into explicit,
fixed-shape JAX operators. It does **not** parse DICOM or NIfTI, infer patient
coordinate conventions, remove protected health information (PHI), or estimate a
deformation. Ingestion code must normalize those concerns before constructing the
records described here.

## Coordinate and time metadata

`SpatialFrame` names a patient-space coordinate system and declares either
`SpatialConvention.LPS` or `SpatialConvention.RAS`. Coordinates and affine
translations are in millimetres. `SpatialAffine` maps a final `(i, j, k)` voxel
index axis into that patient frame:

```python
import numpy as np
from phydrax.applications.cardiovascular import observations as cvobs

lps = cvobs.SpatialFrame("scanner-lps", cvobs.SpatialConvention.LPS)
affine = cvobs.SpatialAffine(
    np.array(
        [
            [1.25, 0.0, 0.0, -80.0],
            [0.0, 1.25, 0.0, -96.0],
            [0.0, 0.0, 8.0, -40.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    "cine-voxel-ijk",
    lps,
)

world_mm = affine.index_to_world(np.array([[10.0, 20.0, 3.0]]))
indices = affine.world_to_index(world_mm)
```

`to_convention` applies the explicit LPS/RAS sign change to the world affine. It
does not relabel an unchanged matrix. This makes round trips testable and keeps
patient axes out of array-layout assumptions.

If normalized qform and sform matrices are both available, resolve them with
`SpatialAffine.from_qform_sform`. The resolver accepts one valid form or two forms
that agree within `conflict_tolerance_mm`. It refuses conflicting forms rather
than silently choosing one:

```python
resolved = cvobs.SpatialAffine.from_qform_sform(
    qform_mm=qform,
    sform_mm=sform,
    source_frame_id="lge-voxel-ijk",
    target_frame=lps,
)
```

`TimeBase` stores strictly increasing sample times in milliseconds. Use
`TimeBase.uniform` only when acquisition timing is genuinely uniform; otherwise
pass the normalized timestamps directly. `is_uniform`, `interval_ms`, and
`duration_ms` expose host-side timing facts without reconstructing them inside a
compiled calculation.

## De-identification and data rights

Every `MedicalImageAsset` requires both a `DeidentificationIdentity` and a
`DataRightsIdentity`. Admission fails unless direct identifiers, burned-in
annotations, and facial features have all been handled, and unless the requested
`intended_use` appears in the explicit rights grant. Common direct-identifier
metadata keys are rejected recursively even when the de-identification flags are
true.

```python
deid = cvobs.DeidentificationIdentity(
    "deid-run-23",
    "subject-pseudonym-0042",
    "site-protocol-a",
    True,   # direct identifiers removed
    True,   # burned-in annotations removed
    True,   # facial features removed
)
rights = cvobs.DataRightsIdentity(
    "rights-grant-7",
    "institutional-research-grant",
    ("research",),
    "site-data-controller",
)
asset = cvobs.MedicalImageAsset(
    "cine-series-4",
    "cine-mri",
    normalized_pixels,
    affine,
    timebase,
    deid,
    rights,
    "signal-intensity",
    "arbitrary-unit",
    valid_mask=valid_pixels,
    metadata={"series_description": "short-axis cine"},
)
```

Arrays are defensive, read-only host copies. Invalid samples may contain nonfinite
sentinels only where `valid_mask` is false. `content_id` binds the array contents,
mask, affine, timebase, rights, de-identification identity, and safe metadata.
This identity is not a claim that upstream source files have been archived.

`ObservationRecord` is the smaller host channel consumed by personalization
adapters. It carries `record_id`, `modality`, `values`, `valid_mask`, `quantity`,
`unit`, and optional `frame_id`, `timebase_id`, and `asset_id`. Likelihood code
can convert its values and mask to JAX arrays while leaving rights and PHI checks
at the host boundary.

## Prepared spatial and temporal sampling

Sampling follows a plan/prepare/evaluate split. Host preparation fixes topology,
route indices, route weights, support, and a stable plan identity. The prepared
operator then has a fixed-shape, JAX-compatible action.

- `VoxelObservationPlan` lowers patient-space query points through a
  `SpatialAffine` to trilinear voxel routes.
- `P1ObservationPlan` locates fixed points in a tetrahedral mesh and constructs
  barycentric P1 routes. Supplying `cell_indices` avoids host point search when an
  authoritative containing-cell map already exists.
- `SurfaceObservationPlan` constructs triangular-surface P1 routes and checks
  distance from each fixed point to its candidate face.
- `ElectrodeObservationPlan` represents explicit electrode, lead, or reference
  combinations over a source potential space. Signed lead weights are retained;
  they are not normalized.
- `TimeObservationPlan` constructs piecewise-linear routes over an explicit
  `TimeBase`.
- `ObservationSamplingPlan` is the low-level fixed-width sparse route contract.

```python
plan = cvobs.VoxelObservationPlan(
    asset.values.shape[:3],
    asset.spatial_affine,
    query_points_lps_mm,
    require_complete_coverage=False,
)
operator = plan.prepare()
candidate = operator.apply(asset.values, source_mask=asset.valid_mask)
```

`candidate.evidence.support` is the exact query mask after geometric support and
source-mask handling. The evidence also records `covered_count`, `query_count`,
`coverage_fraction`, `complete_coverage`, `finite`, and `successful`. With
`require_complete_coverage=False`, at least one supported, finite query is needed
for success; with it enabled, every query must be supported. Unsupported output
entries are zero but never masquerade as observed values because support remains
explicit.

The prepared operator is linear for a fixed mask. `transpose(cotangent)` returns
the exact source-space transpose action, and `jvp(values, tangent)` returns the
primal candidate plus the exact tangent action. These are the supported routes
for adjoint objectives and parameter sensitivities; downstream code should not
materialize a second dense observation matrix.

## Cine timing

`CineTimingPlan` maps one non-endpoint-duplicated cardiac cycle to periodic phase,
using an explicit end-diastolic time and cycle length in milliseconds. Preparation
computes circular Voronoi frame durations. Their sum equals the declared cycle
length even for irregular timing.

```python
cine_timing = cvobs.CineTimingPlan(
    timebase,
    cycle_length_ms=860.0,
    end_diastolic_time_ms=12.0,
).prepare()
timing = cine_timing.evaluate()
```

The result preserves acquisition-order times, phases in `[0, 1)`, and frame
durations. Evidence reports the largest circular phase gap, phase coverage,
phase uniqueness, finiteness, and success. `phase_at(dynamic_times_ms)` applies
the same periodic reference inside JAX calculations.

## Deformation registration evidence

A `RegistrationEvaluationPlan` evaluates a displacement field already estimated
by an imaging-registration method. It fixes reference points, reference and target
frame IDs, map direction, a minimum admissible Jacobian determinant, and whether
inverse-consistency or uncertainty evidence is mandatory.
Only `RegistrationDirection.REFERENCE_TO_TARGET` is admitted by this
reference-point evaluator; reverse registration must use a separate plan with
its own reference support. Every `evaluate` call must supply both runtime frame
IDs. Missing identities are rejected, and mismatches produce unsuccessful
evidence.


```python
registration = cvobs.RegistrationEvaluationPlan(
    reference_points_mm,
    "end-diastole-lps",
    "end-systole-lps",
    require_inverse_consistency=True,
    require_uncertainty=True,
).prepare()

candidate = registration.evaluate(
    displacement_mm,
    displacement_gradient,
    inverse_displacement_at_deformed_mm=reverse_displacement_at_forward_points,
    displacement_standard_deviation_mm=displacement_std_mm,
    reference_frame_id="end-diastole-lps",
    target_frame_id="end-systole-lps",
)
checkpoint = registration.commit(candidate)
```

The displacement convention is `deformed_points = reference_points +
displacement`; therefore the deformation gradient is `I + displacement_gradient`.
The reverse displacement supplied for inverse consistency must already be sampled
at the forward-deformed points. The evaluator does not hide another interpolation
or change topology during differentiation.

Evidence includes:

- explicit reference/target frame matches;
- Jacobian determinant and folding mask, count, and fraction;
- inverse-consistency availability, RMS and maximum residual in millimetres, and
  tolerance outcome;
- uncertainty availability, RMS scale, and non-negative finite validity; and
- aggregate finite and fail-closed success flags.

Only a successful host-evaluated candidate can be committed to a
`RegistrationCheckpoint`. Registration remains an observation/measurement
operation. It does not define passive material response, active stress, force,
energy, equilibrium, or any other mechanics law.

## Green–Lagrange and Eulerian strain

`StrainEvaluationPlan` fixes sample shape, reference frame, and a typed
`StrainMeasure`. `GREEN_LAGRANGE` returns reference-configuration strain
`0.5 * (F.T @ F - I)`. `EULERIAN` returns current-configuration Euler–Almansi
strain `0.5 * (I - F**(-T) @ F**(-1))`. The standalone
`green_lagrange_strain` and `eulerian_strain` functions expose the same tensor
calculations.

```python
strain_evaluator = cvobs.StrainEvaluationPlan(
    displacement_gradient.shape[:-2],
    "end-diastole-lps",
    cvobs.StrainMeasure.GREEN_LAGRANGE,
    require_uncertainty=True,
).prepare()
strain = strain_evaluator.evaluate(
    candidate.deformation_gradient,
    deformation_gradient_standard_deviation=gradient_std,
    reference_frame_id="end-diastole-lps",
)
```
Every strain evaluation requires the runtime `reference_frame_id`; omitting it
is rejected and a mismatch fails the candidate evidence.


When independent deformation-gradient standard deviations are supplied, the
prepared evaluator propagates them through the selected finite-strain map with
exact JVPs and reports tensor standard deviations. This is first-order,
independent-input propagation, not a claim of a complete posterior covariance.
Evidence keeps reference-frame agreement, determinant, folding, invertibility,
symmetry residual, uncertainty validity, finiteness, and aggregate success
explicit. A folded or singular deformation never receives successful strain
evidence.
