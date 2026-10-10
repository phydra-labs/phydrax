#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from importlib import import_module
from typing import Any, TYPE_CHECKING

from ._adaptation import (
    adapt_proposal_scale,
    AdaptiveProposalState,
    initialize_proposal_adaptation,
    RobbinsMonroScalePolicy,
)
from ._addressing import derive_key, SampleAddress
from ._chain import AbstractChainSampleResult
from ._chunks import (
    MarkovChunkIterationMetrics,
    MarkovChunkPlan,
    MarkovChunkResult,
    sample_markov_chunked,
)
from ._designs import (
    host_design,
    host_design_factory,
    materialize_design,
    seed_from_key,
)


_HAMILTONIAN_EXPORTS = frozenset(
    (
        "adapt_hamiltonian_kernel",
        "HamiltonianAdaptationPlan",
        "HamiltonianAdaptationResult",
        "HamiltonianChainState",
        "HamiltonianIterationMetrics",
        "HamiltonianSampleResult",
        "initialize_hamiltonian_state",
        "prepare_hamiltonian_kernel",
        "PreparedHamiltonianKernel",
        "sample_hamiltonian",
    )
)


if TYPE_CHECKING:
    from ._hamiltonian import (
        adapt_hamiltonian_kernel,
        HamiltonianAdaptationPlan,
        HamiltonianAdaptationResult,
        HamiltonianChainState,
        HamiltonianIterationMetrics,
        HamiltonianSampleResult,
        initialize_hamiltonian_state,
        prepare_hamiltonian_kernel,
        PreparedHamiltonianKernel,
        sample_hamiltonian,
    )
from ._markov import (
    MarkovIterationMetrics,
    MarkovSampleResult,
    MarkovState,
    MarkovTransitionInfo,
    MetropolisHastings,
    sample_markov,
)
from ._proposals import (
    AbstractProposal,
    CallableProposal,
    GaussianRandomWalkProposal,
    ProposalMove,
    SingleCoordinateGaussianProposal,
    SingleCoordinatePeriodicProposal,
    SingleCoordinateProposalPayload,
    SingleElectronSphereProposal,
    SphereElectronProposalPayload,
)
from ._targets import (
    FullMarkovTarget,
    IncrementalMarkovTarget,
    IncrementalTargetProposal,
    MarkovTargetState,
)
from ._transports import UnitCubeTransport
from ._types import (
    AntitheticDesign,
    design_capabilities,
    design_name,
    design_signature,
    DesignCapabilities,
    DesignLike,
    DesignName,
    HaltonDesign,
    HammersleyDesign,
    IIDDesign,
    LatinHypercubeDesign,
    normalize_design_name,
    RandomizedQMCDesign,
    resolve_design,
    SobolDesign,
    UnitDesign,
)


def __getattr__(name: str) -> Any:
    if name not in _HAMILTONIAN_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("._hamiltonian", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _HAMILTONIAN_EXPORTS)


__all__ = [
    "AbstractChainSampleResult",
    "AdaptiveProposalState",
    "AbstractProposal",
    "AntitheticDesign",
    "CallableProposal",
    "DesignCapabilities",
    "DesignLike",
    "DesignName",
    "HaltonDesign",
    "HamiltonianIterationMetrics",
    "FullMarkovTarget",
    "GaussianRandomWalkProposal",
    "HammersleyDesign",
    "IIDDesign",
    "HamiltonianAdaptationPlan",
    "HamiltonianAdaptationResult",
    "HamiltonianChainState",
    "HamiltonianSampleResult",
    "IncrementalMarkovTarget",
    "IncrementalTargetProposal",
    "MarkovIterationMetrics",
    "MarkovSampleResult",
    "MarkovState",
    "MarkovTransitionInfo",
    "MetropolisHastings",
    "MarkovChunkIterationMetrics",
    "MarkovChunkPlan",
    "MarkovChunkResult",
    "MarkovTargetState",
    "LatinHypercubeDesign",
    "RandomizedQMCDesign",
    "UnitCubeTransport",
    "SampleAddress",
    "PreparedHamiltonianKernel",
    "ProposalMove",
    "SingleCoordinateGaussianProposal",
    "SingleElectronSphereProposal",
    "SphereElectronProposalPayload",
    "SingleCoordinatePeriodicProposal",
    "SingleCoordinateProposalPayload",
    "RobbinsMonroScalePolicy",
    "SobolDesign",
    "design_signature",
    "UnitDesign",
    "derive_key",
    "adapt_hamiltonian_kernel",
    "adapt_proposal_scale",
    "design_capabilities",
    "design_name",
    "host_design",
    "host_design_factory",
    "initialize_hamiltonian_state",
    "initialize_proposal_adaptation",
    "materialize_design",
    "normalize_design_name",
    "resolve_design",
    "seed_from_key",
    "prepare_hamiltonian_kernel",
    "sample_markov",
    "sample_hamiltonian",
    "sample_markov_chunked",
]
