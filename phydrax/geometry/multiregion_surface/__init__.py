#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Labeled non-manifold multiregion surfaces (soap-film and dry-foam complexes)."""

from ._contracts import (
    DRY_FOAM_EDGE_VALENCE,
    DRY_FOAM_VERTEX_REGIONS,
    MultiRegionCoordinateDtype,
    MultiRegionDomain,
    MultiRegionIndexDtype,
    MultiRegionKind,
    MultiRegionSurfaceCapacityError,
    MultiRegionSurfaceCapacityEvidence,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceCounts,
    MultiRegionSurfaceEvidence,
    MultiRegionSurfacePreparationError,
    MultiRegionSurfaceStatus,
    MultiRegionSurfaceValidationPolicy,
    MultiRegionValidationProfile,
)
from ._events import (
    MultiRegionSurfaceLineage,
    SurfaceEventKind,
    SurfaceEventPassEvidence,
    SurfaceEventPassResult,
    SurfaceEventPassStatus,
    SurfaceEventPolicy,
    SurfaceEventRecord,
    SurfaceEventStatus,
)
from ._geometry import (
    JunctionWedges,
    MultiRegionSurfaceGeometry,
    PreparedMultiRegionSurface,
)
from ._label_extraction import (
    LabelFieldSurfaceExtractionEvidence,
    LabelFieldSurfaceExtractionPlan,
    LabelFieldSurfaceExtractionResult,
    LabelFieldSurfaceExtractionRoute,
    LabelFieldSurfaceExtractionStatus,
    LabelFieldSurfaceLineage,
    PreparedLabelFieldSurfaceExtraction,
)
from ._profiles import multiregion_surface_candidate_profiles
from ._remesh import (
    EdgeCollapseProposal,
    EdgeFlipProposal,
    EdgeSplitProposal,
    MultiRegionRemeshFlags,
    MultiRegionRemeshOperation,
    MultiRegionRemeshPlan,
    propose_remesh,
    remesh_edge_flags,
)
from ._seeding import (
    MultiRegionSurfaceSeed,
    seed_catenoid,
    seed_double_bubble,
    seed_from_vertex_tissue,
    seed_sphere,
)
from ._state import MultiRegionSurfaceState
from ._topology import MultiRegionSurfaceTopology
from ._topology_transitions import (
    MergeProposal,
    PinchProposal,
    propose_merges,
    propose_pinches,
    propose_t1_pops,
    RegionSplitProposal,
    SurfaceEventSearch,
    SurfaceMergePolicy,
    T1PopProposal,
)
from ._transaction import (
    apply_surface_burst,
    apply_surface_events,
    SurfaceBurstResult,
    SurfaceEventProposal,
)
from ._transfers import (
    BoundedFieldReconstruction,
    ConservativeFieldTransfer,
    ExtensiveTransferEvidence,
    IntensiveReconstructionEvidence,
)
from ._validation import validate_multiregion_surface
from ._views import (
    multiregion_cell_complex,
    multiregion_sheet_views,
    MultiRegionSheetView,
    MultiRegionSheetViews,
)


__all__ = [
    "DRY_FOAM_EDGE_VALENCE",
    "DRY_FOAM_VERTEX_REGIONS",
    "BoundedFieldReconstruction",
    "ConservativeFieldTransfer",
    "EdgeCollapseProposal",
    "EdgeFlipProposal",
    "EdgeSplitProposal",
    "ExtensiveTransferEvidence",
    "IntensiveReconstructionEvidence",
    "LabelFieldSurfaceExtractionEvidence",
    "LabelFieldSurfaceExtractionPlan",
    "LabelFieldSurfaceExtractionResult",
    "LabelFieldSurfaceExtractionRoute",
    "LabelFieldSurfaceExtractionStatus",
    "LabelFieldSurfaceLineage",
    "JunctionWedges",
    "MergeProposal",
    "MultiRegionCoordinateDtype",
    "MultiRegionDomain",
    "MultiRegionIndexDtype",
    "MultiRegionKind",
    "MultiRegionRemeshFlags",
    "MultiRegionRemeshOperation",
    "MultiRegionRemeshPlan",
    "MultiRegionSheetView",
    "MultiRegionSheetViews",
    "MultiRegionSurfaceCapacityError",
    "MultiRegionSurfaceCapacityEvidence",
    "MultiRegionSurfaceCapacityPlan",
    "MultiRegionSurfaceCounts",
    "MultiRegionSurfaceEvidence",
    "MultiRegionSurfaceGeometry",
    "MultiRegionSurfaceLineage",
    "MultiRegionSurfacePreparationError",
    "MultiRegionSurfaceSeed",
    "MultiRegionSurfaceState",
    "MultiRegionSurfaceStatus",
    "MultiRegionSurfaceTopology",
    "MultiRegionSurfaceValidationPolicy",
    "MultiRegionValidationProfile",
    "PinchProposal",
    "PreparedLabelFieldSurfaceExtraction",
    "PreparedMultiRegionSurface",
    "RegionSplitProposal",
    "SurfaceBurstResult",
    "SurfaceEventKind",
    "SurfaceEventPassEvidence",
    "SurfaceEventPassResult",
    "SurfaceEventPassStatus",
    "SurfaceEventPolicy",
    "SurfaceEventProposal",
    "SurfaceEventRecord",
    "SurfaceEventSearch",
    "SurfaceEventStatus",
    "SurfaceMergePolicy",
    "T1PopProposal",
    "apply_surface_burst",
    "apply_surface_events",
    "multiregion_surface_candidate_profiles",
    "multiregion_cell_complex",
    "multiregion_sheet_views",
    "propose_merges",
    "propose_pinches",
    "propose_remesh",
    "propose_t1_pops",
    "remesh_edge_flags",
    "seed_catenoid",
    "seed_double_bubble",
    "seed_from_vertex_tissue",
    "seed_sphere",
    "validate_multiregion_surface",
]
