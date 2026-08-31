#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._acceleration import BarnesHutDiagnostics2D, BarnesHutVortexPlan2D
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
from ._filament3d import (
    FilamentVelocityDiagnostics,
    FilamentVelocityEvaluation,
    PreparedFilamentVelocity3D,
    regularized_filament_velocity_3d,
)
from ._gaussian2d import (
    gaussian_vortex_kernel_2d,
    gaussian_vortex_velocity_2d,
    gaussian_vortex_velocity_gradient_2d,
    gaussian_vortex_vorticity_2d,
    GaussianVortexKernelEvaluation2D,
)
from ._gaussian3d import GaussianErfKernelEvaluation3D, GaussianErfVortexKernel3D
from ._panels2d import (
    constant_panel_velocity_2d,
    FlowPanelGeometry2D,
    panel_influence_matrix_2d,
    RigidPanelMotion2D,
)
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


__all__ = [
    "BarnesHutDiagnostics2D",
    "BarnesHutVortexPlan2D",
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
    "RigidPanelMotion2D",
    "constant_panel_velocity_2d",
    "gaussian_vortex_kernel_2d",
    "gaussian_vortex_velocity_2d",
    "gaussian_vortex_velocity_gradient_2d",
    "gaussian_vortex_vorticity_2d",
    "panel_influence_matrix_2d",
    "regularized_filament_velocity_3d",
]
