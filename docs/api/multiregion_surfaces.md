# Multiregion surfaces

Labeled non-manifold triangle complexes for soap films and dry foams. See the
[multiregion surfaces guide](../guides_multiregion_surfaces.md).

## Contracts and topology

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceCapacityPlan

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceCounts

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceCapacityEvidence

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceCapacityError

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceTopology

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceState

## Validation

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceValidationPolicy

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceStatus

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceEvidence

::: phydrax.geometry.multiregion_surface.MultiRegionSurfacePreparationError

::: phydrax.geometry.multiregion_surface.validate_multiregion_surface

## Prepared geometry

::: phydrax.geometry.multiregion_surface.PreparedMultiRegionSurface

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceGeometry

::: phydrax.geometry.multiregion_surface.JunctionWedges

## Views

::: phydrax.geometry.multiregion_surface.multiregion_cell_complex

::: phydrax.geometry.multiregion_surface.multiregion_sheet_views

::: phydrax.geometry.multiregion_surface.MultiRegionSheetView

::: phydrax.geometry.multiregion_surface.MultiRegionSheetViews

## Transfers

::: phydrax.geometry.multiregion_surface.ConservativeFieldTransfer

::: phydrax.geometry.multiregion_surface.ExtensiveTransferEvidence

::: phydrax.geometry.multiregion_surface.BoundedFieldReconstruction

::: phydrax.geometry.multiregion_surface.IntensiveReconstructionEvidence

## Topology events

::: phydrax.geometry.multiregion_surface.apply_surface_events

::: phydrax.geometry.multiregion_surface.apply_surface_burst

::: phydrax.geometry.multiregion_surface.SurfaceBurstResult


`SurfaceEventProposal` accepts `EdgeSplitProposal`, `EdgeCollapseProposal`,
`EdgeFlipProposal`, `T1PopProposal`, `PinchProposal`, `MergeProposal`,
`RegionSplitProposal`, or `BurstProposal`.

::: phydrax.geometry.multiregion_surface.SurfaceEventPolicy

::: phydrax.geometry.multiregion_surface.SurfaceEventKind

::: phydrax.geometry.multiregion_surface.SurfaceEventStatus

::: phydrax.geometry.multiregion_surface.SurfaceEventPassStatus

::: phydrax.geometry.multiregion_surface.SurfaceEventRecord

::: phydrax.geometry.multiregion_surface.SurfaceEventPassEvidence

::: phydrax.geometry.multiregion_surface.SurfaceEventPassResult

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceLineage

## Quality remeshing

::: phydrax.geometry.multiregion_surface.MultiRegionRemeshPlan

`MultiRegionRemeshOperation` is the closed selector `"split" | "collapse" | "flip"`.

::: phydrax.geometry.multiregion_surface.MultiRegionRemeshFlags

::: phydrax.geometry.multiregion_surface.remesh_edge_flags

::: phydrax.geometry.multiregion_surface.propose_remesh

::: phydrax.geometry.multiregion_surface.EdgeSplitProposal

::: phydrax.geometry.multiregion_surface.EdgeCollapseProposal

::: phydrax.geometry.multiregion_surface.EdgeFlipProposal

## Physical transitions

::: phydrax.geometry.multiregion_surface.T1PopProposal

::: phydrax.geometry.multiregion_surface.PinchProposal

::: phydrax.geometry.multiregion_surface.MergeProposal

::: phydrax.geometry.multiregion_surface.SurfaceMergePolicy

::: phydrax.geometry.multiregion_surface.RegionSplitProposal

::: phydrax.geometry.multiregion_surface.SurfaceEventSearch

::: phydrax.geometry.multiregion_surface.propose_t1_pops

::: phydrax.geometry.multiregion_surface.propose_pinches

::: phydrax.geometry.multiregion_surface.propose_merges

## Seeds

::: phydrax.geometry.multiregion_surface.MultiRegionSurfaceSeed

::: phydrax.geometry.multiregion_surface.seed_from_vertex_tissue

::: phydrax.geometry.multiregion_surface.seed_sphere

::: phydrax.geometry.multiregion_surface.seed_double_bubble

::: phydrax.geometry.multiregion_surface.seed_catenoid

## Hard-label extraction

::: phydrax.geometry.multiregion_surface.LabelFieldSurfaceExtractionPlan

::: phydrax.geometry.multiregion_surface.PreparedLabelFieldSurfaceExtraction

::: phydrax.geometry.multiregion_surface.LabelFieldSurfaceExtractionResult

::: phydrax.geometry.multiregion_surface.LabelFieldSurfaceExtractionEvidence

::: phydrax.geometry.multiregion_surface.LabelFieldSurfaceLineage

::: phydrax.geometry.multiregion_surface.LabelFieldSurfaceExtractionStatus

## Candidate profiles

::: phydrax.geometry.multiregion_surface.multiregion_surface_candidate_profiles
