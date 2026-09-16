# Scientific and medical imaging

`phydrax.imaging` owns generic two-dimensional image supports, calibrated cameras,
nonperiodic image sampling, Schlieren/BOS image formation, and strict medical
image preparation. External-data authority and physical quantity semantics come
from [`phydrax.measurement`](guides_measurement.md). Plotting and interactive
viewers remain external.

`ImagePlaneSupport` fixes row-down/column-right image coordinates and implements
the shared sample-support contract. `ImageAsset` binds that support to a governed
`MeasurementAsset`. Generic camera models now live under
`phydrax.imaging.camera`; state-to-image operators live under
[`phydrax.rendering`](guides_rendering.md).

## Medical coordinates and time



`ImageIndexAffine` maps voxel indices into one `SpatialCoordinateContract` and
records RAS/LPS convention plus voxel-center/corner semantics. Affine translation
and linear terms use the contract's `UnitDefinition`. `to_convention` performs the
RAS/LPS reflection; `to_unit` converts all physical rows. Conflicting valid NIfTI
qform/sform matrices are refused.

`SampleTimeAxis` stores strictly increasing samples with an explicit time unit.
Cardiovascular plans convert it to milliseconds at their application boundary;
brain tracer studies may retain seconds or hours without another time type.

## Admission

Every `MedicalImageAsset` requires:

- complete `DeidentificationEvidence`;
- one or more governed `ReferenceArtifactManifest` values;
- an `ImageFieldSpec` containing `QuantitySpec`, `ValueLayout`, and sampling semantics;
- an explicit `DerivationRecord`;
- a validity mask over sample sites;
- optional independent standard uncertainty and quality flags that remain separate
  from validity.

Only valid samples must be finite. Invalid values are retained but never silently
filled. Categorical arrays require integer storage. Probability arrays require a
nonnegative unit simplex. `DiffusionTensorImage` requires symmetric positive
semidefinite tensors and an explicit component frame.

`NibabelImageProvider` is lazy and available with `phydrax[imaging-nifti]`. It
verifies the source file against its reference manifest, resolves qform/sform,
preserves values and affine on export, and requires export rights. NIfTI is
represented in RAS coordinates; LPS assets are deliberately reframed for export.

Medical admission composes a canonical `MeasurementAsset` but keeps PHI refusal,
LPS/RAS, voxel geometry, and medical-tool execution as stricter imaging rules.

## Image and mesh transfer

- `VoxelObservationPlan` prepares trilinear gathers and exact transposes.
- `ImageToP1ProjectionPlan` samples tetrahedral quadrature and solves the
  consistent P1 mass system.
- `ConservativeVoxelCellTransfer` consumes exact voxel/cell overlap measures and
  supplies primal, dual-pullback, and weighted-adjoint actions.
- `LabelImageTransferPlan` performs nearest categorical transfer.
- `ProbabilityImageTransferPlan` preserves the probability simplex.
- `TensorImageTransferPlan` supports Euclidean or log-Euclidean interpolation and
  explicit proper-rotation reorientation.

Transfers are bound to source and target identities. Coverage, partition of
unity, constant reproduction, conservation, and adjoint defects are returned as
evidence. Geometry intersection code prepares overlap measures; the compiled
runtime stores only sparse routes.

## Segmentation and compartments

`LabelOntology` decouples semantic identities from integer storage. A
`SegmentationProcessingPlan` applies explicit connected-component, small-component,
or hole-filling operations and reports changed voxels, physical measure, and
component counts. No fixed clipping plane or unrecorded smoothing exists.

`build_compartment_complex` derives physical measures and six-neighbor adjacency.
`extract_compartment_surfaces` emits each declared interface once with a stable
orientation. The default extractor is exact with respect to voxel cells and is
therefore staircase geometry; smoothing/remeshing is a separate revisioned stage.

```python
import numpy as np

import phydrax as phx

contract = phx.SpatialCoordinateContract(
    phx.units.MILLIMETER,
    coordinate_system="cartesian-lps",
    reference_frame="patient-space",
)
affine = phx.imaging.ImageIndexAffine(
    np.eye(4), "voxel-index", contract, phx.imaging.ImageAxisConvention.LPS
)
```

## Registration evidence

`RegistrationEvaluationPlan` admits externally estimated reference-to-target
displacements only after frame matching, positive deformation Jacobians,
optional inverse-consistency, and optional uncertainty checks. A committed
`RegistrationCheckpoint` is an observation artifact, not a mechanics solution.

Run `python examples/image_mesh_transfer.py` for an affine-exact image-to-P1
projection.

## DICOM and CT material calibration

`phydrax.imaging.interchange` contains bounded, read-only profile readers for
regular legacy/Enhanced CT, NM counts, PET activity concentration, RT Plan
metadata, closed-planar RT Structure Set contours, and linked or explicitly
unlinked RT Dose. It has no generic reader, writer, network client,
de-identifier, contour rasterizer, registration, or treatment executor. See the
[DICOM interchange guide](guides_dicom_interchange.md).

`HUToMaterialCalibration` is a separate governed derivation from a CT-number
image to density and ordered material fractions. It is piecewise linear only
inside the source-pinned calibration interval and never clamps or extrapolates.
CT number admission itself does not assert density, composition, attenuation,
or diagnosis.

The calibration retains its own `ReferenceArtifactManifest` in every derived
asset. Scalar density uncertainty may be propagated from independent HU
uncertainty through the local slope. Material-fraction components share the same
HU source and are correlated, so they do not receive a false diagonal
`IndependentStandardUncertainty`; their covariance remains explicitly
unrepresented.
