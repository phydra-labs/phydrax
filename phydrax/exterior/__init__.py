"""Canonical exterior types, kernels, and prepared de Rham calculus.

Core types and algebra load independently of smooth and discrete realizations.
"""

from importlib import import_module
from typing import Any, TYPE_CHECKING


_FACADE_EXPORT_MODULES: dict[str, str] = {
    "FiberProduct": "._form_type",
    "FormProxy": "._form_type",
    "FormPullbackRule": "._form_type",
    "FormTwist": "._form_type",
    "FormType": "._form_type",
    "FormValueSpec": "._form_type",
    "axes_bitmap": "._basis",
    "bitmap_axes": "._basis",
    "exterior_indices": "._basis",
    "wedge_sign": "._basis",
    "wedge_table": "._basis",
    "derivative_table": "._basis",
    "interior_table": "._basis",
    "complement_table": "._basis",
    "wedge": "._algebra",
    "interior": "._algebra",
    "exterior_derivative_from_jacobian": "._algebra",
    "hodge_star": "._algebra",
    "inner": "._algebra",
    "pullback": "._algebra",
    "to_twisted": "._algebra",
    "to_untwisted": "._algebra",
    "hodge_square_sign": "._algebra",
    "codifferential_sign": "._algebra",
    "vector_to_form": "._algebra",
    "form_to_vector": "._algebra",
    "map_reference_values": "._algebra",
    "AbstractDeRhamComplex": "._complex",
    "ComplexBoundary": "._complex",
    "DiscreteForm": "._complex",
    "CellParameterization": "._de_rham",
    "DeRhamBridge": "._de_rham",
    "DeRhamCommutationEvidence": "._de_rham",
    "simplicial_parameterizations": "._de_rham",
    "structured_parameterizations": "._de_rham",
    "integrate_form": "._de_rham",
    "validate_de_rham_commutation": "._de_rham",
    "metric_dual_hodges": "._de_rham",
    "HodgeSpectrumPolicy": "._spectra",
    "hodge_laplacian_eigenbasis": "._spectra",
    "HodgeSectorSpectra": "._spectra",
    "hodge_sector_spectra": "._spectra",
    "HodgeCohomologyReport": "._cohomology",
    "validate_harmonic_cohomology": "._cohomology",
    "harmonic_kernel_certificate": "._cohomology",
    "HarmonicClassFrame": "._cohomology",
    "prepare_harmonic_class_frame": "._cohomology",
    "HodgeSubspaceTracking": "._cohomology",
    "AbstractChainIntegrationKernel": "._chains",
    "PreparedChainQuery": "._chains",
    "SegmentWeight": "._chains",
    "TraceEvidence": "._traces",
    "trace_evidence": "._traces",
    "trace_map": "._traces",
    "CoefficientSystem": "._coefficients",
    "CurvatureEvidence": "._coefficients",
    "WhitneyProductPlan": "._products",
    "LieDerivativeMethod": "._products",
    "ProductVectorField": "._products",
    "cochain_cup_product": "._products",
    "whitney_wedge": "._products",
    "interior_product": "._products",
    "lie_derivative": "._products",
    "twisted_differential": "._coefficients",
    "curvature_evidence": "._coefficients",
    "bloch_coefficient_system": "._coefficients",
    "orientation_coefficient_system": "._coefficients",
    "sheaf_laplacian": "._coefficients",
}

__all__ = list(_FACADE_EXPORT_MODULES)

if TYPE_CHECKING:
    from ._algebra import (
        codifferential_sign as codifferential_sign,
        exterior_derivative_from_jacobian as exterior_derivative_from_jacobian,
        form_to_vector as form_to_vector,
        hodge_square_sign as hodge_square_sign,
        hodge_star as hodge_star,
        inner as inner,
        interior as interior,
        map_reference_values as map_reference_values,
        pullback as pullback,
        to_twisted as to_twisted,
        to_untwisted as to_untwisted,
        vector_to_form as vector_to_form,
        wedge as wedge,
    )
    from ._basis import (
        axes_bitmap as axes_bitmap,
        bitmap_axes as bitmap_axes,
        complement_table as complement_table,
        derivative_table as derivative_table,
        exterior_indices as exterior_indices,
        interior_table as interior_table,
        wedge_sign as wedge_sign,
        wedge_table as wedge_table,
    )
    from ._chains import (
        AbstractChainIntegrationKernel as AbstractChainIntegrationKernel,
        PreparedChainQuery as PreparedChainQuery,
        SegmentWeight as SegmentWeight,
    )
    from ._coefficients import (
        bloch_coefficient_system as bloch_coefficient_system,
        CoefficientSystem as CoefficientSystem,
        curvature_evidence as curvature_evidence,
        CurvatureEvidence as CurvatureEvidence,
        orientation_coefficient_system as orientation_coefficient_system,
        sheaf_laplacian as sheaf_laplacian,
        twisted_differential as twisted_differential,
    )
    from ._cohomology import (
        harmonic_kernel_certificate as harmonic_kernel_certificate,
        HarmonicClassFrame as HarmonicClassFrame,
        HodgeCohomologyReport as HodgeCohomologyReport,
        HodgeSubspaceTracking as HodgeSubspaceTracking,
        prepare_harmonic_class_frame as prepare_harmonic_class_frame,
        validate_harmonic_cohomology as validate_harmonic_cohomology,
    )
    from ._complex import (
        AbstractDeRhamComplex as AbstractDeRhamComplex,
        ComplexBoundary as ComplexBoundary,
        DiscreteForm as DiscreteForm,
    )
    from ._de_rham import (
        CellParameterization as CellParameterization,
        DeRhamBridge as DeRhamBridge,
        DeRhamCommutationEvidence as DeRhamCommutationEvidence,
        integrate_form as integrate_form,
        metric_dual_hodges as metric_dual_hodges,
        simplicial_parameterizations as simplicial_parameterizations,
        structured_parameterizations as structured_parameterizations,
        validate_de_rham_commutation as validate_de_rham_commutation,
    )
    from ._form_type import (
        FiberProduct as FiberProduct,
        FormProxy as FormProxy,
        FormPullbackRule as FormPullbackRule,
        FormTwist as FormTwist,
        FormType as FormType,
        FormValueSpec as FormValueSpec,
    )
    from ._products import (
        cochain_cup_product as cochain_cup_product,
        interior_product as interior_product,
        lie_derivative as lie_derivative,
        LieDerivativeMethod as LieDerivativeMethod,
        ProductVectorField as ProductVectorField,
        whitney_wedge as whitney_wedge,
        WhitneyProductPlan as WhitneyProductPlan,
    )
    from ._spectra import (
        hodge_laplacian_eigenbasis as hodge_laplacian_eigenbasis,
        hodge_sector_spectra as hodge_sector_spectra,
        HodgeSectorSpectra as HodgeSectorSpectra,
        HodgeSpectrumPolicy as HodgeSpectrumPolicy,
    )
    from ._traces import (
        trace_evidence as trace_evidence,
        trace_map as trace_map,
        TraceEvidence as TraceEvidence,
    )


def __getattr__(name: str) -> Any:
    module = _FACADE_EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
