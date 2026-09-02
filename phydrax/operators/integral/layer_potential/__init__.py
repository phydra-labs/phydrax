"""Prepared boundary layer representations, discretizations, and evaluators."""

from ._acceleration import (
    AbstractLayerBackend,
    DirectNearFarReferenceBackend2D,
    LayerBackendEvaluation2D,
)
from ._adaptive_boundary import *  # noqa: F403
from ._adaptive_boundary import __all__ as _adaptive_boundary_all
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
from ._displacement_discontinuity3d import *  # noqa: F403
from ._displacement_discontinuity3d import __all__ as _displacement_discontinuity_all
from ._elasticity3d import *  # noqa: F403
from ._elasticity3d import __all__ as _elasticity_all
from ._evaluation import (
    evaluate_layer_potential,
    LayerEvaluationPlan2D,
    LayerEvaluationReport,
    LayerEvaluationResult,
)
from ._fast_provider import *  # noqa: F403
from ._fast_provider import __all__ as _fast_provider_all
from ._fmm2d import LaplaceFMMBackend2D, LaplaceFMMEvaluation2D
from ._free_surface_green3d import *  # noqa: F403
from ._free_surface_green3d import __all__ as _free_surface_green_all
from ._free_surface_hydrodynamics3d import *  # noqa: F403
from ._free_surface_hydrodynamics3d import __all__ as _free_surface_hydrodynamics_all
from ._galerkin3d import (
    LaplaceSingleLayerDP0AssemblyReport3D,
    LaplaceSingleLayerDP0Galerkin3D,
    LaplaceSingleLayerDP0GalerkinPolicy3D,
    prepare_laplace_single_layer_dp0_3d,
)
from ._global_qbx_fmm2d import evaluate_global_qbx_fmm_2d, GlobalQBXFMMEvaluation2D
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
from ._hierarchical3d import *  # noqa: F403
from ._hierarchical3d import __all__ as _hierarchical3d_all
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
from ._maxwell3d import *  # noqa: F403
from ._maxwell3d import __all__ as _maxwell_all
from ._modified_helmholtz3d import *  # noqa: F403
from ._modified_helmholtz3d import __all__ as _modified_helmholtz_all
from ._periodic_core3d import *  # noqa: F403
from ._periodic_core3d import __all__ as _periodic_core_all
from ._periodic_free_surface3d import *  # noqa: F403
from ._periodic_free_surface3d import __all__ as _periodic_free_surface_all
from ._periodic_helmholtz3d import *  # noqa: F403
from ._periodic_helmholtz3d import __all__ as _periodic_helmholtz_all
from ._periodic_laplace3d import *  # noqa: F403
from ._periodic_laplace3d import __all__ as _periodic_laplace_all
from ._periodic_maxwell_boundary3d import *  # noqa: F403
from ._periodic_maxwell_boundary3d import __all__ as _periodic_maxwell_boundary_all
from ._periodic_modified_helmholtz3d import *  # noqa: F403
from ._periodic_modified_helmholtz3d import (
    __all__ as _periodic_modified_helmholtz_all,
)
from ._periodic_vector3d import *  # noqa: F403
from ._periodic_vector3d import __all__ as _periodic_vector_all
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
from ._qualification import *  # noqa: F403
from ._qualification import __all__ as _qualification_all
from ._rcip import RCIPPreconditioner2D
from ._scalar_calderon3d import *  # noqa: F403
from ._scalar_calderon3d import __all__ as _scalar_calderon_all
from ._scalar_conforming3d import *  # noqa: F403
from ._scalar_conforming3d import __all__ as _scalar_conforming_all
from ._scalar_formulations3d import *  # noqa: F403
from ._scalar_formulations3d import __all__ as _scalar_formulations_all
from ._scalar_interfaces3d import *  # noqa: F403
from ._scalar_interfaces3d import __all__ as _scalar_interfaces_all
from ._scalar_screens3d import *  # noqa: F403
from ._scalar_screens3d import __all__ as _scalar_screens_all
from ._scalar_trace import *  # noqa: F403
from ._scalar_trace import __all__ as _scalar_trace_all
from ._stokes3d import *  # noqa: F403
from ._stokes3d import __all__ as _stokes_all
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
    "LaplaceSingleLayerDP0AssemblyReport3D",
    "LaplaceSingleLayerDP0Galerkin3D",
    "LaplaceSingleLayerDP0GalerkinPolicy3D",
    "prepare_laplace_single_layer_dp0_3d",
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
    "RCIPPreconditioner2D",
    "DirectNearFarReferenceBackend2D",
    "HelmholtzCombinedField3D",
    "HelmholtzLayerKernel3D",
    "HelmholtzLayerPotential3D",
    "AbstractLayerBackend",
    "LayerBackendEvaluation2D",
    "LaplaceTreecodeBackend2D",
    "LaplaceTreecodeEvaluation2D",
    "LaplaceFMMBackend2D",
    "LaplaceFMMEvaluation2D",
    "GlobalQBXFMMEvaluation2D",
    "evaluate_global_qbx_fmm_2d",
]

__all__ += [
    name
    for name in (
        *_adaptive_boundary_all,
        *_displacement_discontinuity_all,
        *_elasticity_all,
        *_fast_provider_all,
        *_hierarchical3d_all,
        *_free_surface_green_all,
        *_free_surface_hydrodynamics_all,
        *_maxwell_all,
        *_modified_helmholtz_all,
        *_periodic_core_all,
        *_periodic_free_surface_all,
        *_periodic_maxwell_boundary_all,
        *_periodic_helmholtz_all,
        *_periodic_laplace_all,
        *_periodic_modified_helmholtz_all,
        *_qualification_all,
        *_scalar_conforming_all,
        *_scalar_calderon_all,
        *_scalar_formulations_all,
        *_scalar_trace_all,
        *_stokes_all,
    )
    if name not in __all__
]
__all__ += [
    name
    for name in (*_periodic_vector_all, *_scalar_interfaces_all)
    if name not in __all__
]
__all__ += [name for name in _scalar_screens_all if name not in __all__]
