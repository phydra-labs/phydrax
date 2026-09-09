# Cardiovascular observation metadata and operators

Cardiovascular observation plans consume shared normalized image and observation
contracts. Raw DICOM/NIfTI parsing, PHI admission, physical coordinates, and
rights belong to `phydrax.imaging`; modality-specific signal interpretation and
likelihoods remain here.

## Coordinate and time metadata

`ImageIndexAffine` maps voxel indices to an exact `SpatialCoordinateContract` and
declares RAS/LPS plus voxel-center semantics:

```python
import numpy as np
import phydrax as phx

contract = phx.SpatialCoordinateContract(
    phx.units.MILLIMETER,
    coordinate_system="cartesian-lps",
    reference_frame="patient-space",
)
affine = phx.imaging.ImageIndexAffine(
    np.eye(4),
    "cine-voxel-index",
    contract,
    phx.imaging.ImageAxisConvention.LPS,
)
time_axis = phx.measurement.SampleTimeAxis.uniform(
    "cine-clock", 20, 40.0, phx.units.MILLISECOND
)
```

`to_convention` performs an actual RAS/LPS reflection and `to_unit` scales the
three world rows. `SampleTimeAxis` supports any time `UnitDefinition`;
cardiovascular plans explicitly convert to milliseconds.

## De-identification and data rights

`MedicalImageAsset` requires complete `DeidentificationEvidence`, an
offline-verifiable `ReferenceArtifactManifest`, an `ImageFieldSpec`, an explicit
derivation record, and an exact validity mask. PHI keys are rejected recursively.
Arrays are defensive,
read-only host copies. See [Medical imaging](guides_imaging.md).

`phydrax.observation.ObservationRecord` is the smaller normalized host channel
consumed by personalization adapters. It carries optional `frame_id`,
`time_axis_id`, and `asset_id`; it does not duplicate image governance.

## Prepared spatial and temporal sampling

Shared fixed-shape plans live in `phydrax.spatial_sampling`:

- `VoxelObservationPlan` builds trilinear voxel routes;
- `P1ObservationPlan` builds tetrahedral barycentric routes;
- `SurfaceObservationPlan` builds triangular-surface routes;
- `TimeObservationPlan` builds piecewise-linear routes over `SampleTimeAxis`;
- `ObservationSamplingPlan` is the low-level sparse route contract.

`ElectrodeObservationPlan` remains re-exported from the cardiovascular namespace
because electrode semantics are modality-specific.

```python
operator = phx.spatial_sampling.VoxelObservationPlan(
    asset.values.shape[:3],
    asset.spatial_affine,
    query_points_lps_mm,
    require_complete_coverage=False,
).prepare()
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

The shared `phydrax.imaging.RegistrationEvaluationPlan` evaluates a displacement
field already estimated by an imaging-registration method. It fixes reference
points, reference and target frame IDs, map direction, a minimum admissible
Jacobian determinant, and required inverse-consistency/uncertainty evidence.
Only `RegistrationDirection.REFERENCE_TO_TARGET` is admitted by this
reference-point evaluator; reverse registration must use a separate plan with
its own reference support. Every `evaluate` call must supply both runtime frame
IDs. Missing identities are rejected, and mismatches produce unsuccessful
evidence.


```python
registration = phx.imaging.RegistrationEvaluationPlan(
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
