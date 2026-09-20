#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Differentiable finite-dimensional Euclidean path-integral operators."""

from importlib import import_module

from ._action import (
    discrete_euclidean_action,
    kinetic_action,
    potential_action,
)
from ._diffusion import diffusion_paths_from_noise, sample_diffusion_paths
from ._estimate import PathIntegralEstimate
from ._euclidean import (
    euclidean_kernel,
    euclidean_kernel_from_noise,
    free_euclidean_kernel,
)
from ._exchange import (
    estimate_exchange_observable,
    exchange_path_action,
    ExchangePathEstimate,
    ExchangePathPlan,
)
from ._feynman_kac import (
    AdaptiveFeynmanKacEstimate,
    feynman_kac_expectation,
    feynman_kac_from_paths,
    source_feynman_kac_from_paths,
    source_feynman_kac_from_stochastic_paths,
    SourceFeynmanKacEstimate,
)
from ._first_passage import first_exit_index, first_exit_time, survival_probability
from ._function import euclidean_kernel_function
from ._geometry import (
    GeometryKernelEstimate,
    interval_heat_kernel,
    killed_path_mask,
    prepare_path_boundary_schedule,
    PreparedGeometryPathKernel,
    specular_reflect,
    SpecularReflectionResult,
)
from ._improved_gauge import __all__ as _improved_gauge_all
from ._lattice_action import (
    AbstractIncrementalLatticeAction,
    AbstractLatticeEuclideanAction,
    compact_geometric_target_from_lattice_action,
    full_target_from_lattice_action,
    incremental_target_from_lattice_action,
    lattice_action_local_gradient,
    LatticeActionEvidence,
    LatticeReferenceMeasure,
)
from ._lattice_fermion import __all__ as _lattice_fermion_all
from ._lattice_gauge import (
    CompactU1GaugeMeasure,
    initialize_u1_gauge_state,
    U1GaugeState,
    wilson_loop,
    wrap_u1,
)
from ._lattice_observable import (
    evaluate_lattice_observable,
    LatticeObservableKind,
    LatticeObservableNormalization,
    LatticeObservablePlan,
    LatticeObservableValue,
    phi4_observable_plans,
    phi4_pair_correlation_plan,
)
from ._periodic import (
    estimate_path_partition_function,
    PathPartitionEstimate,
    periodic_path_action,
    PeriodicPathPlan,
)
from ._pseudofermion import __all__ as _pseudofermion_all
from ._pseudofermion_operator import AbstractPseudofermionDiracOperator
from ._rational_approximation import __all__ as _rational_approximation_all
from ._real_time import (
    continue_real_time_regulator_from_noise,
    OscillatoryPathIntegralEstimate,
    real_time_kernel,
    real_time_kernel_from_noise,
    RealTimeContinuationResult,
    RealTimePathIntegralPlan,
    RealTimeRegulatorContinuation,
)
from ._sampling import (
    brownian_bridge_from_noise,
    sample_brownian_bridge,
)
from ._scalar_lattice import (
    LocalPhi4LatticeAction,
    Phi4ActionCache,
    Phi4LatticeAction,
    prepare_local_phi4_action,
)
from ._smearing import __all__ as _smearing_all
from ._wilson_gauge import (
    GaugeLinkProposalPayload,
    WilsonGaugeAction,
    WilsonGaugeActionCache,
)


_FACADE_EXPORT_MODULES = (
    "._improved_gauge",
    "._lattice_fermion",
    "._pseudofermion",
    "._rational_approximation",
    "._smearing",
)


def __getattr__(name: str):
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
    *_improved_gauge_all,
    *_lattice_fermion_all,
    *_pseudofermion_all,
    *_rational_approximation_all,
    *_smearing_all,
    "AbstractIncrementalLatticeAction",
    "AbstractLatticeEuclideanAction",
    "AbstractPseudofermionDiracOperator",
    "AdaptiveFeynmanKacEstimate",
    "CompactU1GaugeMeasure",
    "ExchangePathEstimate",
    "ExchangePathPlan",
    "GeometryKernelEstimate",
    "GaugeLinkProposalPayload",
    "LatticeActionEvidence",
    "LatticeObservableKind",
    "LatticeObservableNormalization",
    "LatticeObservablePlan",
    "LatticeObservableValue",
    "LatticeReferenceMeasure",
    "LocalPhi4LatticeAction",
    "OscillatoryPathIntegralEstimate",
    "PathPartitionEstimate",
    "PeriodicPathPlan",
    "PreparedGeometryPathKernel",
    "RealTimeContinuationResult",
    "RealTimePathIntegralPlan",
    "RealTimeRegulatorContinuation",
    "Phi4ActionCache",
    "Phi4LatticeAction",
    "SourceFeynmanKacEstimate",
    "WilsonGaugeAction",
    "WilsonGaugeActionCache",
    "SpecularReflectionResult",
    "U1GaugeState",
    "euclidean_kernel_function",
    "PathIntegralEstimate",
    "brownian_bridge_from_noise",
    "diffusion_paths_from_noise",
    "discrete_euclidean_action",
    "estimate_exchange_observable",
    "estimate_path_partition_function",
    "exchange_path_action",
    "evaluate_lattice_observable",
    "compact_geometric_target_from_lattice_action",
    "full_target_from_lattice_action",
    "incremental_target_from_lattice_action",
    "euclidean_kernel",
    "feynman_kac_expectation",
    "feynman_kac_from_paths",
    "continue_real_time_regulator_from_noise",
    "first_exit_index",
    "first_exit_time",
    "euclidean_kernel_from_noise",
    "free_euclidean_kernel",
    "kinetic_action",
    "lattice_action_local_gradient",
    "initialize_u1_gauge_state",
    "interval_heat_kernel",
    "killed_path_mask",
    "periodic_path_action",
    "phi4_observable_plans",
    "phi4_pair_correlation_plan",
    "prepare_local_phi4_action",
    "prepare_path_boundary_schedule",
    "potential_action",
    "sample_brownian_bridge",
    "real_time_kernel",
    "real_time_kernel_from_noise",
    "sample_diffusion_paths",
    "source_feynman_kac_from_paths",
    "source_feynman_kac_from_stochastic_paths",
    "specular_reflect",
    "survival_probability",
    "wilson_loop",
    "wrap_u1",
]
