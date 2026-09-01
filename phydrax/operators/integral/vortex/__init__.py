#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._acceleration import FixedClusterDiagnostics2D, FixedClusterVortexPlan2D
from ._core_families import (
    RosenheadVortexKernel2D,
    RosenheadVortexKernel3D,
    SingularVortexKernel2D,
    SingularVortexKernel3D,
    VortexCoreEvaluation,
)
from ._direct2d import (
    DirectVortexResourceEvidence,
    GaussianDirectDiagnostics2D,
    GaussianDirectVortexPlan2D,
    PreparedGaussianDirectVortex2D,
)
from ._direct3d import (
    DirectVortexResourceEvidence3D,
    GaussianErfDirectVortexPlan3D,
    PreparedGaussianErfDirectVortex3D,
)
from ._ewald import *  # noqa: F403
from ._ewald import __all__ as _ewald_all
from ._filament3d import (
    FilamentVelocityDiagnostics,
    FilamentVelocityEvaluation,
    PreparedFilamentVelocity3D,
    regularized_filament_velocity_3d,
)
from ._fmm_complete import *  # noqa: F403
from ._fmm_complete import __all__ as _fmm_complete_all
from ._free_space_mesh import *  # noqa: F403
from ._free_space_mesh import __all__ as _free_space_mesh_all
from ._gaussian2d import (
    gaussian_vortex_kernel_2d,
    gaussian_vortex_velocity_2d,
    gaussian_vortex_velocity_gradient_2d,
    gaussian_vortex_vorticity_2d,
    GaussianVortexKernelEvaluation2D,
)
from ._gaussian3d import GaussianErfKernelEvaluation3D, GaussianErfVortexKernel3D
from ._morton import *  # noqa: F403
from ._morton import __all__ as _morton_all
from ._p3m import *  # noqa: F403
from ._p3m import __all__ as _p3m_all
from ._panel_complete import *  # noqa: F403
from ._panel_complete import __all__ as _panel_complete_all
from ._panels2d import (
    constant_panel_velocity_2d,
    FlowPanelGeometry2D,
    panel_influence_matrix_2d,
    RigidPanelMotion2D,
)
from ._panels3d_complete import *  # noqa: F403
from ._panels3d_complete import __all__ as _panels3d_complete_all
from ._particle_mesh import (
    PeriodicVortexInCellDiagnostics,
    PeriodicVortexInCellPlan,
    PreparedPeriodicVortexInCell,
)
from ._pse import (
    GaussianParticleStrengthExchangePlan,
    ParticleStrengthExchangeEvidence,
    PreparedGaussianParticleStrengthExchange,
)
from ._ring_field import *  # noqa: F403
from ._ring_field import __all__ as _ring_field_all
from ._sharding import *  # noqa: F403
from ._sharding import __all__ as _sharding_all


__all__ = [
    "FixedClusterDiagnostics2D",
    "FixedClusterVortexPlan2D",
    "DirectVortexResourceEvidence",
    "DirectVortexResourceEvidence3D",
    "FilamentVelocityDiagnostics",
    "FilamentVelocityEvaluation",
    "FlowPanelGeometry2D",
    "GaussianDirectDiagnostics2D",
    "GaussianDirectVortexPlan2D",
    "GaussianErfDirectVortexPlan3D",
    "GaussianErfKernelEvaluation3D",
    "GaussianErfVortexKernel3D",
    "GaussianParticleStrengthExchangePlan",
    "GaussianVortexKernelEvaluation2D",
    "ParticleStrengthExchangeEvidence",
    "PeriodicVortexInCellDiagnostics",
    "PeriodicVortexInCellPlan",
    "PreparedFilamentVelocity3D",
    "PreparedGaussianDirectVortex2D",
    "PreparedGaussianErfDirectVortex3D",
    "PreparedGaussianParticleStrengthExchange",
    "PreparedPeriodicVortexInCell",
    "RosenheadVortexKernel2D",
    "RosenheadVortexKernel3D",
    "RigidPanelMotion2D",
    "SingularVortexKernel2D",
    "SingularVortexKernel3D",
    "VortexCoreEvaluation",
    "constant_panel_velocity_2d",
    "gaussian_vortex_kernel_2d",
    "gaussian_vortex_velocity_2d",
    "gaussian_vortex_velocity_gradient_2d",
    "gaussian_vortex_vorticity_2d",
    "panel_influence_matrix_2d",
    "regularized_filament_velocity_3d",
]

__all__ += [
    name
    for name in (
        *_ewald_all,
        *_fmm_complete_all,
        *_free_space_mesh_all,
        *_morton_all,
        *_p3m_all,
        *_panel_complete_all,
        *_panels3d_complete_all,
        *_ring_field_all,
        *_sharding_all,
    )
    if name not in __all__
]
