# DICOM research interchange

`phydrax.imaging.interchange` is a bounded, read-only admission layer for a small set of explicitly named DICOM profiles. It does not expose a generic DICOM object model, discover files, contact a DICOM service, de-identify data, write DICOM, rasterize contours, register images, or execute a treatment plan.

Install the optional parser with the `imaging-dicom` extra. The parser remains lazy: importing `phydrax.imaging` does not import `pydicom`.

## Admission boundary

Every reader accepts caller-supplied `BoundedResource` bytes, matching `ReferenceArtifactManifest` records, research-ready `DeidentificationEvidence`, and a finite `DICOMResourcePolicy`. Readers verify the exact source bytes, enforce SOP-specific requirements, remove non-allowlisted metadata, record transformations and losses in `DICOMImportReport`, and construct research-only results.

`DICOMObjectIdentity`, `DICOMReference`, and `DICOMReferenceGraph` retain de-identified Study, Series, SOP, and Frame-of-Reference identity. Linked RT profiles require a closed graph; unresolved or class-mismatched references are errors rather than warnings.

## Supported profiles

| Profile | Entry point | Output boundary |
|---|---|---|
| Legacy CT Image Storage series | `read_dicom_legacy_ct_image_series` | Regular LPS CT-number `MedicalImageAsset` |
| Enhanced CT | `read_dicom_enhanced_ct_image` | Regular LPS CT-number `MedicalImageAsset` |
| NM counts | `read_dicom_nuclear_medicine_counts_image` | Counts, never inferred activity |
| PET activity concentration | `read_dicom_pet_activity_concentration_image_series` | Explicit activity concentration and frame timing |
| RT Plan | `read_dicom_rt_plan_metadata` | Host-only identity/reference metadata; no delivery model |
| RT Structure Set | `read_dicom_rt_structure_set_closed_planar` | Closed planar contours; no implicit rasterization |
| Unlinked RT Dose | `read_dicom_rt_dose_unlinked` | Explicitly unlinked physical or relative score grid |
| Plan-linked RT Dose | `read_dicom_rt_dose_linked_plan` | Dose grid with a closed RT reference graph |

Each function implements one profile. There is no SOP autodetection entry point.

## Geometry and values

CT and RT Dose use DICOM patient LPS coordinates and voxel-center semantics. Slice order comes from physical positions, not filenames or `InstanceNumber`. Stored-pixel rescale and `DoseGridScaling` are applied exactly once. A frame stack is admitted only when its orientation, spacing, displacement, scaling, and dimensions form one representable regular affine lattice. Irregular stacks are refused; readers never resample silently.

CT output is CT number only. It is not density, elemental composition, attenuation, or diagnosis. Use `HUToMaterialCalibration` separately with a source-pinned calibration.

NM counts, PET activity concentration, physical absorbed dose, dose to water, dose to medium, kerma, and relative dose are distinct quantities. A DICOM tag or profile must establish the quantity; equal array shapes or units never do. Relative RT Dose remains dimensionless. The RT Dose readers never infer a water/medium basis.

## PHI and resource refusal

Readers require caller-provided de-identification evidence and reject known identifying/private metadata outside the technical allowlist. They are not anonymizers. Raw bytes remain governed by their manifests and access policy.

`DICOMResourcePolicy` bounds instance, frame, row, column, voxel, decoded-byte, element, sequence-depth, contour, and contour-point counts. Unsupported transfer syntax, malformed UID, unknown units, inconsistent geometry, missing calibration, unresolved linked reference, and resource excess fail closed.

## Qualification status

The synthetic fixtures prove deterministic parser/profile behavior only. They do not validate a scanner, reconstruction, activity calibration, treatment plan, treatment delivery, or clinical workflow. All imaging DICOM capability profiles remain unreleased until independent source, conformance, coordinate, and transfer evidence closes their named gates.
