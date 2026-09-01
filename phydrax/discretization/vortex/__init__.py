#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._advanced import (
    PedrizzettiRelaxationPlan3D,
    ReformulatedVPMPlan3D,
    ReformulatedVPMRate3D,
    VortexRelaxationResult3D,
)
from ._capabilities import (
    VortexDiffusionCapabilities,
    VortexVelocityCapabilities,
)
from ._capacity import *  # noqa: F403
from ._capacity import __all__ as _capacity_all
from ._checkpoint import *  # noqa: F403
from ._checkpoint import __all__ as _checkpoint_all
from ._compatibility import (
    vortex_property_requirements,
    VortexPropertyRequirements,
    VortexVelocityCompatibility,
)
from ._diffusion_complete import *  # noqa: F403
from ._diffusion_complete import __all__ as _diffusion_complete_all
from ._export import *  # noqa: F403
from ._export import __all__ as _export_all
from ._filament import (
    OrientedFilamentGeometry,
    VortexFilamentState,
    VortexFilamentTopology,
)
from ._formulations_complete import *  # noqa: F403
from ._formulations_complete import __all__ as _formulations_complete_all
from ._hybrid_derivatives import *  # noqa: F403
from ._hybrid_derivatives import __all__ as _hybrid_derivatives_all
from ._interfaces import (
    AbstractPreparedVortexDiffusion,
    AbstractPreparedVortexVelocity,
    AbstractVortexDiffusionPlan,
    AbstractVortexVelocityPlan,
    DEFAULT_VORTEX_FIELD_REQUEST,
    VortexDiffusionDiagnostics,
    VortexDiffusionEvaluation,
    VortexFieldRequest,
    VortexVelocityDiagnostics,
    VortexVelocityEvaluation,
)
from ._lifting import LiftingSurfacePlan, PreparedLiftingSurface
from ._lifting_complete import *  # noqa: F403
from ._lifting_complete import __all__ as _lifting_complete_all
from ._method import (
    BackgroundVortexVelocity,
    InviscidVortexDiffusionPlan,
    PreparedInviscidVortexDiffusion,
    PreparedVortexParticleDynamics,
    VortexParticleDiagnostics,
    VortexParticleMethodPlan,
    VortexParticleStepRestriction,
)
from ._particle import (
    VortexParticleProperties,
    VortexParticleState,
    VortexParticleStateLayout,
)
from ._population import *  # noqa: F403
from ._population import __all__ as _population_all
from ._precision import VortexPrecisionPolicy
from ._remesh import ConservativeVortexRemeshPlan2D, VortexRemeshResult2D
from ._remesh_complete import *  # noqa: F403
from ._remesh_complete import __all__ as _remesh_complete_all
from ._replay import *  # noqa: F403
from ._replay import __all__ as _replay_all
from ._ring_sheet import *  # noqa: F403
from ._ring_sheet import __all__ as _ring_sheet_all
from ._source import VortexSourceState, VortexTargetState
from ._wake import VortexWakePlan, VortexWakeState, VortexWakeTransition
from ._wall import (
    BoundarySheetParticleTransferPlan2D,
    BoundarySheetParticleTransferResult,
    WallVortexPoolState,
)
from ._wall_diffusion import *  # noqa: F403
from ._wall_diffusion import __all__ as _wall_diffusion_all


__all__ = [
    "AbstractPreparedVortexDiffusion",
    "AbstractPreparedVortexVelocity",
    "AbstractVortexDiffusionPlan",
    "AbstractVortexVelocityPlan",
    "BoundarySheetParticleTransferPlan2D",
    "BoundarySheetParticleTransferResult",
    "BackgroundVortexVelocity",
    "ConservativeVortexRemeshPlan2D",
    "DEFAULT_VORTEX_FIELD_REQUEST",
    "InviscidVortexDiffusionPlan",
    "LiftingSurfacePlan",
    "OrientedFilamentGeometry",
    "PedrizzettiRelaxationPlan3D",
    "PreparedInviscidVortexDiffusion",
    "PreparedLiftingSurface",
    "PreparedVortexParticleDynamics",
    "ReformulatedVPMPlan3D",
    "ReformulatedVPMRate3D",
    "VortexDiffusionCapabilities",
    "VortexDiffusionDiagnostics",
    "VortexDiffusionEvaluation",
    "VortexFieldRequest",
    "VortexFilamentState",
    "VortexFilamentTopology",
    "VortexParticleDiagnostics",
    "VortexParticleMethodPlan",
    "VortexParticleProperties",
    "VortexParticleState",
    "VortexParticleStateLayout",
    "VortexPrecisionPolicy",
    "VortexPropertyRequirements",
    "VortexParticleStepRestriction",
    "VortexRelaxationResult3D",
    "VortexSourceState",
    "VortexTargetState",
    "VortexRemeshResult2D",
    "VortexVelocityDiagnostics",
    "VortexVelocityEvaluation",
    "VortexVelocityCapabilities",
    "VortexVelocityCompatibility",
    "VortexWakePlan",
    "VortexWakeState",
    "VortexWakeTransition",
    "WallVortexPoolState",
    "vortex_property_requirements",
]

__all__ += [
    name
    for name in (
        *_capacity_all,
        *_checkpoint_all,
        *_diffusion_complete_all,
        *_export_all,
        *_formulations_complete_all,
        *_hybrid_derivatives_all,
        *_lifting_complete_all,
        *_population_all,
        *_remesh_complete_all,
        *_replay_all,
        *_ring_sheet_all,
        *_wall_diffusion_all,
    )
    if name not in __all__
]
