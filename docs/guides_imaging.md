# Medical imaging

`phydrax.imaging` owns normalized host-side medical-image arrays, exact physical
coordinates, data admission, and quantity-aware transfer. It does not parse raw
DICOM at model execution time and never places PHI-bearing metadata in JAX state.

## Coordinates and time

`ImageIndexAffine` maps voxel indices into one `SpatialCoordinateContract` and
records RAS/LPS convention plus voxel-center/corner semantics. Affine translation
and linear terms use the contract's `UnitDefinition`. `to_convention` performs the
RAS/LPS reflection; `to_unit` converts all physical rows. Conflicting valid NIfTI
qform/sform matrices are refused.

`ImageTimeAxis` stores strictly increasing samples with an explicit time unit.
Cardiovascular plans convert it to milliseconds at their application boundary;
brain tracer studies may retain seconds or hours without another time type.

## Admission

Every `MedicalImageAsset` requires:

- complete `DeidentificationEvidence`;
- a `ReferenceArtifactManifest` with checksum, license, rights, and lineage;
- an `ImageValueLayout` with quantity, unit, component shape, and transformation kind;
- a validity mask over sample sites.

Only valid samples must be finite. Invalid values are retained but never silently
filled. Categorical arrays require integer storage. Probability arrays require a
nonnegative unit simplex. `DiffusionTensorImage` requires symmetric positive
semidefinite tensors and an explicit component frame.

`NibabelImageProvider` is lazy and available with `phydrax[imaging-nifti]`. It
verifies the source file against its reference manifest, resolves qform/sform,
preserves values and affine on export, and requires export rights. NIfTI is
represented in RAS coordinates; LPS assets are deliberately reframed for export.

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
