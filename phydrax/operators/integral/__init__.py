from importlib import import_module
from typing import Any

#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from . import layer_potential, multipole as multipole, vortex as vortex
from ._batch_ops import integral, integrate_boundary, integrate_interior, mean
from ._convolution_quadrature import __all__ as _convolution_quadrature_all
from ._diffrax_collocation import DiffraxCollocationIntegralOperator
from ._free_space_convolution import __all__ as _free_space_convolution_all
from ._local_ops import local_integral, local_integral_ball
from ._spatial_ops import nonlocal_integral, spatial_integral
from ._time_convolution import time_convolution
from .layer_potential import (
    __all__ as _layer_potential_all,
    AbstractLayerBackend,
    AbstractLayerKernel,
    AdaptiveLayerEvaluation2D,
    BoundaryCornerTopology2D,
    BoundaryOperatorAssemblyReport,
    BoundaryPanelization2D,
    BoundaryPanelPartition2D,
    BoundaryTraceProjection2D,
    BoundaryTraceRepresentation2D,
    BoundaryTraceSpace2D,
    classify_panel_interactions_2d,
    ClosedPolygonalCurve2D,
    CurveTraversal2D,
    DirectNearFarReferenceBackend2D,
    double_layer_principal_value_matrix,
    evaluate_double_layer_self_triangle_3d,
    evaluate_global_qbx_fmm_2d,
    evaluate_laplace_layer_3d,
    evaluate_layer_potential,
    evaluate_qbx_2d,
    evaluate_qbx_3d,
    evaluate_single_layer_self_triangle_3d,
    ExteriorFarField2D,
    ExteriorLaplaceDirichletResult2D,
    GlobalQBXFMMEvaluation2D,
    HelmholtzCombinedField2D,
    HelmholtzCombinedField3D,
    HelmholtzLayerKernel2D,
    HelmholtzLayerKernel3D,
    HelmholtzLayerPotential2D,
    HelmholtzLayerPotential3D,
    KernelActionSide,
    LaplaceFMMBackend2D,
    LaplaceFMMEvaluation2D,
    LaplaceLayerKernel2D,
    LaplaceLayerKernel3D,
    LaplaceLayerPotential2D,
    LaplaceLayerPotential3D,
    LaplaceSingleLayerDP0AssemblyReport3D,
    LaplaceSingleLayerDP0Galerkin3D,
    LaplaceSingleLayerDP0GalerkinPolicy3D,
    LaplaceTreecodeBackend2D,
    LaplaceTreecodeEvaluation2D,
    LayerBackendEvaluation2D,
    LayerDiscretizationReport,
    LayerEvaluationPlan2D,
    LayerEvaluationReport,
    LayerEvaluationResult,
    LayerPotentialTargetReport,
    PanelInteractionReport2D,
    prepare_boundary_trace_projection_2d,
    prepare_exterior_laplace_dirichlet_2d,
    prepare_laplace_single_layer_dp0_3d,
    prepare_scalar_laplace_galerkin_2d,
    PreparedExteriorLaplaceDirichlet2D,
    QBXEvaluation2D,
    QBXEvaluation3D,
    RCIPPreconditioner2D,
    ScalarBoundarySpaces2D,
    ScalarLaplaceFieldEvaluation2D,
    ScalarLaplaceGalerkin2D,
    ScalarLaplaceGalerkinPolicy2D,
    ScalarLaplaceGalerkinReport2D,
    ScalarTraceConvention2D,
    ScalarTraceSide2D,
    solve_exterior_laplace_dirichlet_2d,
    SurfacePanelization3D,
    SurfaceTargetReport3D,
    TraceProjectionResult2D,
)
from .multipole import __all__ as _multipole_all
from .vortex import __all__ as _vortex_all


_FACADE_EXPORT_MODULES = (
    "._convolution_quadrature",
    "._free_space_convolution",
    ".layer_potential",
    ".multipole",
    ".vortex",
)


def __getattr__(name: str) -> Any:
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AbstractLayerKernel",
    "BoundaryPanelization2D",
    "double_layer_principal_value_matrix",
    "DiffraxCollocationIntegralOperator",
    "integral",
    "integrate_boundary",
    "integrate_interior",
    "local_integral",
    "KernelActionSide",
    "LaplaceLayerKernel2D",
    "LaplaceLayerPotential2D",
    "layer_potential",
    "LayerDiscretizationReport",
    "LayerEvaluationPlan2D",
    "LayerEvaluationReport",
    "LayerEvaluationResult",
    "LayerPotentialTargetReport",
    "LaplaceSingleLayerDP0AssemblyReport3D",
    "LaplaceSingleLayerDP0Galerkin3D",
    "LaplaceSingleLayerDP0GalerkinPolicy3D",
    "prepare_laplace_single_layer_dp0_3d",
    "ClosedPolygonalCurve2D",
    "CurveTraversal2D",
    "ScalarTraceConvention2D",
    "ScalarTraceSide2D",
    "BoundaryTraceRepresentation2D",
    "BoundaryTraceSpace2D",
    "ScalarBoundarySpaces2D",
    "ScalarLaplaceGalerkinPolicy2D",
    "ScalarLaplaceGalerkinReport2D",
    "ScalarLaplaceGalerkin2D",
    "ScalarLaplaceFieldEvaluation2D",
    "prepare_scalar_laplace_galerkin_2d",
    "ExteriorFarField2D",
    "PreparedExteriorLaplaceDirichlet2D",
    "ExteriorLaplaceDirichletResult2D",
    "prepare_exterior_laplace_dirichlet_2d",
    "solve_exterior_laplace_dirichlet_2d",
    "BoundaryTraceProjection2D",
    "TraceProjectionResult2D",
    "prepare_boundary_trace_projection_2d",
    "evaluate_layer_potential",
    "AdaptiveLayerEvaluation2D",
    "PanelInteractionReport2D",
    "classify_panel_interactions_2d",
    "BoundaryCornerTopology2D",
    "BoundaryPanelPartition2D",
    "HelmholtzCombinedField2D",
    "HelmholtzLayerKernel2D",
    "HelmholtzLayerPotential2D",
    "LaplaceLayerKernel3D",
    "LaplaceLayerPotential3D",
    "SurfacePanelization3D",
    "SurfaceTargetReport3D",
    "evaluate_laplace_layer_3d",
    "evaluate_single_layer_self_triangle_3d",
    "evaluate_double_layer_self_triangle_3d",
    "QBXEvaluation3D",
    "evaluate_qbx_3d",
    "DirectNearFarReferenceBackend2D",
    "AbstractLayerBackend",
    "LayerBackendEvaluation2D",
    "BoundaryOperatorAssemblyReport",
    "QBXEvaluation2D",
    "evaluate_qbx_2d",
    "HelmholtzCombinedField3D",
    "HelmholtzLayerKernel3D",
    "RCIPPreconditioner2D",
    "HelmholtzLayerPotential3D",
    "LaplaceTreecodeBackend2D",
    "LaplaceTreecodeEvaluation2D",
    "LaplaceFMMBackend2D",
    "LaplaceFMMEvaluation2D",
    "local_integral_ball",
    "GlobalQBXFMMEvaluation2D",
    "evaluate_global_qbx_fmm_2d",
    "mean",
    "nonlocal_integral",
    "spatial_integral",
    "time_convolution",
]
__all__ += [
    name
    for name in (
        *_convolution_quadrature_all,
        *_free_space_convolution_all,
        *_layer_potential_all,
    )
    if name not in __all__
]

__all__ += [name for name in _vortex_all if name not in __all__]
__all__ += [name for name in _multipole_all if name not in __all__]
