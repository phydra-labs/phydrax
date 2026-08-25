"""Prepared boundary layer representations, discretizations, and evaluators."""

from ._acceleration import (
    AbstractLayerBackend,
    DirectNearFarReferenceBackend2D,
    LayerBackendEvaluation2D,
)
from ._core import (
    AbstractLayerKernel,
    BoundaryCornerTopology2D,
    BoundaryOperatorAssemblyReport,
    BoundaryPanelization2D,
    BoundaryPanelPartition2D,
    KernelActionSide,
    LayerDiscretizationReport,
    LayerPotentialTargetReport,
)
from ._corner_block import CornerBlockInversePreconditioner2D
from ._evaluation import (
    evaluate_layer_potential,
    LayerEvaluationPlan2D,
    LayerEvaluationReport,
    LayerEvaluationResult,
)
from ._helmholtz2d import (
    HelmholtzCombinedField2D,
    HelmholtzLayerKernel2D,
    HelmholtzLayerPotential2D,
)
from ._helmholtz3d import (
    HelmholtzCombinedField3D,
    HelmholtzLayerKernel3D,
    HelmholtzLayerPotential3D,
)
from ._laplace2d import (
    double_layer_principal_value_matrix,
    LaplaceLayerKernel2D,
    LaplaceLayerPotential2D,
)
from ._laplace3d import (
    evaluate_laplace_layer_3d,
    LaplaceLayerKernel3D,
    LaplaceLayerPotential3D,
)
from ._qbx2d import evaluate_qbx_2d, QBXEvaluation2D
from ._qbx3d import evaluate_qbx_3d, QBXEvaluation3D
from ._quadrature2d import (
    AdaptiveLayerEvaluation2D,
    classify_panel_interactions_2d,
    evaluate_laplace_single_layer_self_panel_2d,
    PanelInteractionReport2D,
)
from ._quadrature3d import (
    evaluate_double_layer_self_triangle_3d,
    evaluate_single_layer_self_triangle_3d,
)
from ._surface3d import SurfacePanelization3D, SurfaceTargetReport3D
from ._treecode2d import LaplaceTreecodeBackend2D, LaplaceTreecodeEvaluation2D


__all__ = [
    "AbstractLayerKernel",
    "BoundaryPanelization2D",
    "double_layer_principal_value_matrix",
    "KernelActionSide",
    "LaplaceLayerKernel2D",
    "LaplaceLayerPotential2D",
    "LayerDiscretizationReport",
    "LayerEvaluationPlan2D",
    "LayerEvaluationReport",
    "LayerEvaluationResult",
    "LayerPotentialTargetReport",
    "evaluate_layer_potential",
    "AdaptiveLayerEvaluation2D",
    "PanelInteractionReport2D",
    "classify_panel_interactions_2d",
    "evaluate_laplace_single_layer_self_panel_2d",
    "BoundaryCornerTopology2D",
    "BoundaryPanelPartition2D",
    "HelmholtzCombinedField2D",
    "HelmholtzLayerKernel2D",
    "HelmholtzLayerPotential2D",
    "BoundaryOperatorAssemblyReport",
    "QBXEvaluation2D",
    "evaluate_qbx_2d",
    "LaplaceLayerKernel3D",
    "LaplaceLayerPotential3D",
    "SurfacePanelization3D",
    "SurfaceTargetReport3D",
    "evaluate_laplace_layer_3d",
    "evaluate_double_layer_self_triangle_3d",
    "evaluate_single_layer_self_triangle_3d",
    "QBXEvaluation3D",
    "evaluate_qbx_3d",
    "CornerBlockInversePreconditioner2D",
    "DirectNearFarReferenceBackend2D",
    "HelmholtzCombinedField3D",
    "HelmholtzLayerKernel3D",
    "HelmholtzLayerPotential3D",
    "AbstractLayerBackend",
    "LayerBackendEvaluation2D",
    "LaplaceTreecodeBackend2D",
    "LaplaceTreecodeEvaluation2D",
]
