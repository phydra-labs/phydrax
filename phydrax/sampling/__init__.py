#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reference designs, proposals, and stateful Markov sampling."""

from importlib import import_module

from .._sampling import (
    AbstractChainSampleResult,
    AbstractProposal,
    adapt_hamiltonian_kernel,
    adapt_proposal_scale,
    AdaptiveProposalState,
    AntitheticDesign,
    CallableProposal,
    design_capabilities,
    design_name,
    design_signature,
    DesignCapabilities,
    FullMarkovTarget,
    GaussianRandomWalkProposal,
    HaltonDesign,
    HamiltonianAdaptationPlan,
    HamiltonianAdaptationResult,
    HamiltonianChainState,
    HamiltonianSampleResult,
    HammersleyDesign,
    IIDDesign,
    IncrementalMarkovTarget,
    IncrementalTargetProposal,
    initialize_hamiltonian_state,
    initialize_proposal_adaptation,
    LatinHypercubeDesign,
    MarkovChunkPlan,
    MarkovChunkResult,
    MarkovSampleResult,
    MarkovState,
    MarkovTargetState,
    MarkovTransitionInfo,
    materialize_design,
    MetropolisHastings,
    prepare_hamiltonian_kernel,
    PreparedHamiltonianKernel,
    ProposalMove,
    RandomizedQMCDesign,
    resolve_design,
    RobbinsMonroScalePolicy,
    sample_hamiltonian,
    sample_markov,
    sample_markov_chunked,
    SingleCoordinateGaussianProposal,
    SingleCoordinatePeriodicProposal,
    SingleCoordinateProposalPayload,
    SobolDesign,
)
from . import collocation, conditional
from ._compact_group_hamiltonian import (
    adapt_compact_group_hamiltonian,
    CompactGeometricTarget,
    CompactGroupHamiltonianAdaptationResult,
    CompactGroupHamiltonianChainState,
    CompactGroupHamiltonianIterationMetrics,
    CompactGroupHamiltonianSampleResult,
    initialize_compact_group_hamiltonian_state,
    prepare_compact_group_hamiltonian_kernel,
    PreparedCompactGroupHamiltonianKernel,
    sample_compact_group_hamiltonian,
)
from ._exact_learned import __all__ as _exact_learned_all
from ._gauge_updates import __all__ as _gauge_updates_all
from ._rhmc import __all__ as _rhmc_all
from ._split_group_dynamics import __all__ as _split_group_dynamics_all


_FACADE_EXPORT_MODULES = (
    "._exact_learned",
    "._gauge_updates",
    "._rhmc",
    "._split_group_dynamics",
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
    *_exact_learned_all,
    *_gauge_updates_all,
    *_rhmc_all,
    *_split_group_dynamics_all,
    "AbstractChainSampleResult",
    "AdaptiveProposalState",
    "AbstractProposal",
    "collocation",
    "CompactGeometricTarget",
    "CompactGroupHamiltonianAdaptationResult",
    "CompactGroupHamiltonianChainState",
    "CompactGroupHamiltonianIterationMetrics",
    "CompactGroupHamiltonianSampleResult",
    "conditional",
    "AntitheticDesign",
    "CallableProposal",
    "FullMarkovTarget",
    "DesignCapabilities",
    "HaltonDesign",
    "GaussianRandomWalkProposal",
    "HammersleyDesign",
    "IIDDesign",
    "HamiltonianAdaptationPlan",
    "HamiltonianAdaptationResult",
    "HamiltonianChainState",
    "HamiltonianSampleResult",
    "IncrementalMarkovTarget",
    "IncrementalTargetProposal",
    "initialize_hamiltonian_state",
    "initialize_compact_group_hamiltonian_state",
    "initialize_proposal_adaptation",
    "MarkovSampleResult",
    "MarkovState",
    "MarkovTransitionInfo",
    "MetropolisHastings",
    "MarkovChunkPlan",
    "MarkovChunkResult",
    "MarkovTargetState",
    "PreparedHamiltonianKernel",
    "prepare_compact_group_hamiltonian_kernel",
    "PreparedCompactGroupHamiltonianKernel",
    "ProposalMove",
    "SingleCoordinateGaussianProposal",
    "SingleCoordinatePeriodicProposal",
    "SingleCoordinateProposalPayload",
    "prepare_hamiltonian_kernel",
    "LatinHypercubeDesign",
    "sample_compact_group_hamiltonian",
    "adapt_compact_group_hamiltonian",
    "RandomizedQMCDesign",
    "RobbinsMonroScalePolicy",
    "sample_hamiltonian",
    "sample_markov_chunked",
    "SobolDesign",
    "design_capabilities",
    "adapt_hamiltonian_kernel",
    "adapt_proposal_scale",
    "design_name",
    "materialize_design",
    "design_signature",
    "resolve_design",
    "sample_markov",
]
