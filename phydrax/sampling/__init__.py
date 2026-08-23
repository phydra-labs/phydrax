#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reference designs, proposals, and stateful Markov sampling."""

from .._sampling import (
    AbstractProposal,
    AntitheticDesign,
    CallableProposal,
    DESIGN_ALGORITHM_VERSION,
    design_capabilities,
    design_name,
    design_signature,
    DesignCapabilities,
    GaussianRandomWalkProposal,
    HaltonDesign,
    HammersleyDesign,
    IIDDesign,
    LatinHypercubeDesign,
    MarkovSampleResult,
    MarkovState,
    MarkovTransitionInfo,
    materialize_design,
    MetropolisHastings,
    RandomizedQMCDesign,
    resolve_design,
    sample_markov,
    SobolDesign,
)
from . import collocation


__all__ = [
    "AbstractProposal",
    "collocation",
    "AntitheticDesign",
    "CallableProposal",
    "DESIGN_ALGORITHM_VERSION",
    "DesignCapabilities",
    "HaltonDesign",
    "GaussianRandomWalkProposal",
    "HammersleyDesign",
    "IIDDesign",
    "MarkovSampleResult",
    "MarkovState",
    "MarkovTransitionInfo",
    "MetropolisHastings",
    "LatinHypercubeDesign",
    "RandomizedQMCDesign",
    "SobolDesign",
    "design_capabilities",
    "design_name",
    "materialize_design",
    "design_signature",
    "resolve_design",
    "sample_markov",
]
