#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Operation-level C01–C22 closure matrices for the native atomistic plan.

Each scope-ledger row of ``docs/plans/native-atomistic-capabilities.md`` is one
:class:`~phydrax.qualification.CapabilityClosureMatrix` named
``atomistic.c01`` … ``atomistic.c22``. Its requirements are the operation
obligations that need independent resolution: implementation, numerical,
derivative, resource, hardware/topology, continuation, scientific, rights and
review evidence are never merged into one success flag.

Requirements bind the existing capability declarations and their exact
candidate tuples in :mod:`._profiles` and :mod:`._mace_profiles`; this module
is not a second profile store. Every resolution is one of:

``candidate``
    The native implementation is present in this tree but has no retained
    observed evidence. A registered profile or source inspection is not
    evidence, so this is implemented-but-unqualified.
``missing`` with ``external-prerequisite-…``
    Qualification is blocked by a named external prerequisite (weight or data
    rights, physical hardware, independent authority).
``implemented``
    Only when the caller supplies observed retained evidence for that exact
    requirement. Release is never authorized here.

Required native behavior is never an external prerequisite or a nonclaim: an
obligation is external only when it names a prerequisite outside this tree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..discretization.spectral._qualification import DISTRIBUTED_SPECTRAL_CAPABILITY
from ..qualification._closure_matrix import CapabilityClosureMatrix
from ..qualification._closure_requirement import (
    CapabilityClosureRequirement,
    CapabilityGapResolution,
)
from ..qualification._closure_taxonomy import (
    CapabilityDepth,
    CarrierRepresentation,
    ClosureDisposition,
    CouplingLocation,
    ExecutionRegime,
    PhysicsField,
    TopologyRegime,
    WorkflowClass,
)
from ._profiles import (
    AIMNET2_CAPABILITY,
    atomistic_candidate_declarations,
    CAMPAIGN_CAPABILITY,
    CRYSTAL_CAPABILITY,
    DISPERSION_CAPABILITY,
    DISTILLATION_CAPABILITY,
    DOMAIN_CAPABILITY,
    DYNAMICS_CAPABILITY,
    ELECTROSTATICS_CAPABILITY,
    GENERATION_CAPABILITY,
    INTERCHANGE_CAPABILITY,
    LATENT_CAPABILITY,
    MACE_EXTENDED_CAPABILITY,
    MACELES_CAPABILITY,
    MULTIPOLE_CAPABILITY,
    NEIGHBORHOOD_CAPABILITY,
    REACTION_PATH_CAPABILITY,
    RECIPROCAL_CAPABILITY,
    RELAXATION_CAPABILITY,
    ROUTE_REDUCTION_CAPABILITY,
    SAMPLING_CAPABILITY,
    SPLINE_CAPABILITY,
    TRAINING_CAPABILITY,
    TRAINING_DISTRIBUTION_CAPABILITY,
    UMA_CAPABILITY,
)


_MACE = "atomistic.mace-inference"
_MODELS = (
    _MACE,
    MACE_EXTENDED_CAPABILITY,
    MACELES_CAPABILITY,
    AIMNET2_CAPABILITY,
    UMA_CAPABILITY,
)
_ALL = "all-atomistic-capabilities"

NATIVE_UNQUALIFIED = "native-implementation-present-observed-evidence-required"
OBSERVED = "observed-retained-evidence-release-not-authorized"
EXTERNAL_PREFIX = "external-prerequisite-"


@dataclass(frozen=True, slots=True)
class _Obligation:
    """One independently resolved obligation; ``prerequisite`` marks it external."""

    name: str
    scope: str
    depth: CapabilityDepth
    dimensions: tuple[str, ...]
    capabilities: tuple[str, ...] | None
    prerequisite: str | None
    producers: tuple[str, ...] | None


def _native(
    name: str,
    scope: str,
    depth: CapabilityDepth,
    *dimensions: str,
    capabilities: tuple[str, ...] | None = None,
    producers: tuple[str, ...] | None = None,
) -> _Obligation:
    return _Obligation(name, scope, depth, dimensions, capabilities, None, producers)


def _external(
    name: str,
    scope: str,
    prerequisite: str,
    depth: CapabilityDepth,
    *dimensions: str,
    capabilities: tuple[str, ...] | None = None,
    producers: tuple[str, ...] = (),
) -> _Obligation:
    return _Obligation(
        name, scope, depth, dimensions, capabilities, prerequisite, producers
    )


@dataclass(frozen=True, slots=True)
class _Row:
    ledger: str
    capabilities: tuple[str, ...]
    fields: tuple[PhysicsField, ...]
    regimes: tuple[ExecutionRegime, ...]
    workflows: tuple[WorkflowClass, ...]
    producers: tuple[str, ...]
    sources: tuple[str, ...]
    obligations: tuple[_Obligation, ...]
    couplings: tuple[CouplingLocation, ...] = (CouplingLocation.BULK,)
    topologies: tuple[TopologyRegime, ...] = (TopologyRegime.FIXED,)
    carriers: tuple[CarrierRepresentation, ...] = (CarrierRepresentation.ATOMISTIC,)


_D = CapabilityDepth
_IMPL = ("implementation", "numerical")
_IMPL_DERIVATIVE = (*_IMPL, "derivative")
_CHEMISTRY = (PhysicsField.CHEMICAL_SPECIES,)
_THERMAL = (PhysicsField.CHEMICAL_SPECIES, PhysicsField.THERMAL)
_CELL = (*_THERMAL, PhysicsField.SOLID_MECHANICS)
_TOOLKIT = ("nvalchemi-toolkit",)
_OPS = ("nvalchemi-toolkit-ops",)
_BOTH = (*_TOOLKIT, *_OPS)
_MODEL_SOURCES = (
    "aimnetcentral",
    "e3nn-code",
    "fairchem",
    "les",
    "mace-code",
    "mace-code-0.3.15",
    *_TOOLKIT,
)
_DISTRIBUTED_PRODUCERS = (
    "benchmarks/distributed_atomistic_scaling.py",
    "tools/atomistic_distributed_qualification.py",
    "tools/atomistic_distributed_smoke.py",
)
_CAMPAIGN_PRODUCERS = ("benchmarks/atomistic_dynamics_campaigns.py",)
_TRAINING_PRODUCERS = ("benchmarks/atomistic_training_streaming.py",)
_SAMPLING_PRODUCERS = ("benchmarks/atomistic_enhanced_sampling.py",)
_PERFORMANCE_PRODUCERS = (
    "benchmarks/atomistic_crystal_dedup.py",
    "benchmarks/atomistic_crystal_packing.py",
    "benchmarks/atomistic_dynamics_campaigns.py",
    "benchmarks/atomistic_enhanced_sampling.py",
    "benchmarks/atomistic_model_families.py",
    "benchmarks/atomistic_periodic_generation.py",
    "benchmarks/atomistic_training_streaming.py",
    "benchmarks/distributed_atomistic_scaling.py",
    "tools/particle_neighbor_scaling.py",
)
_MULTI_HOST = "multi-host-cluster-access"

_LEDGER = (
    _Row(
        "C01",
        (*_MODELS, TRAINING_CAPABILITY),
        (*_CHEMISTRY, PhysicsField.ELECTRONIC_QUANTUM),
        (ExecutionRegime.STATIC,),
        (WorkflowClass.FORWARD, WorkflowClass.CALIBRATION),
        (),
        ("aimnetcentral", "fairchem", "mace-code", *_TOOLKIT),
        (
            _native(
                "typed-conditioning-property-ports",
                "species and material identity, units, conditioning, requested properties and typed model ports",
                _D.ANALYTIC_CONTROL,
                *_IMPL,
            ),
            _native(
                "parameter-roles-derivative-requests",
                "trainable parameter roles and admitted coordinate, strain and parameter derivative requests",
                _D.ANALYTIC_CONTROL,
                "derivative",
            ),
        ),
    ),
    _Row(
        "C02",
        _MODELS,
        (*_CHEMISTRY, PhysicsField.ELECTRONIC_QUANTUM),
        (ExecutionRegime.STATIC,),
        (WorkflowClass.FORWARD,),
        ("benchmarks/atomistic_model_families.py",),
        _MODEL_SOURCES,
        (
            _native(
                "native-model-extension-contract",
                "user-defined native energy and property models through the canonical model contract",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "mace-family-architectures",
                "standard, attention/nonlinear, joint-embedding and latent-Ewald MACE families",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL_DERIVATIVE,
                capabilities=(_MACE, MACE_EXTENDED_CAPABILITY, MACELES_CAPABILITY),
            ),
            _native(
                "aimnet2-architecture",
                "AIMNet2 single-channel and NSE charge equilibration with long-range composition",
                _D.SPATIAL_COUPLED,
                *_IMPL_DERIVATIVE,
                capabilities=(AIMNET2_CAPABILITY,),
            ),
            _native(
                "uma-architecture",
                "UMA eSCN-MD backbone, mixture of linear experts, merged preparation and direct heads",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL_DERIVATIVE,
                capabilities=(UMA_CAPABILITY,),
            ),
            _native(
                "model-resource-performance",
                "phase-separated model-family resource and throughput envelope",
                _D.SPATIAL_SINGLE_PHYSICS,
                "resource-performance",
            ),
            _native(
                "mace-source-checkpoint-fidelity",
                "named MACE releases under admitted weights against an isolated source oracle",
                _D.SPATIAL_SINGLE_PHYSICS,
                "numerical",
                capabilities=(_MACE,),
                producers=("tools/mace_checkpoint_campaign.py",),
            ),
            _external(
                "aimnet2-uma-maceles-source-checkpoint-fidelity",
                "named AIMNet2, UMA and MACELES checkpoints against isolated source oracles",
                "admitted-pretrained-weights",
                _D.SPATIAL_SINGLE_PHYSICS,
                "numerical",
                capabilities=(AIMNET2_CAPABILITY, MACELES_CAPABILITY, UMA_CAPABILITY),
            ),
            _external(
                "pretrained-chemical-accuracy",
                "chemical accuracy of pretrained models on independent reference data",
                "admitted-pretrained-weights-and-reference-data",
                _D.SCIENTIFICALLY_VALIDATED,
                "scientific-validation",
            ),
        ),
    ),
    _Row(
        "C03",
        (INTERCHANGE_CAPABILITY, *_MODELS),
        _CHEMISTRY,
        (ExecutionRegime.STATIC, ExecutionRegime.TRANSIENT),
        (WorkflowClass.DEPLOYMENT, WorkflowClass.RESTART_REPLAY),
        ("tools/mace_checkpoint_campaign.py",),
        _MODEL_SOURCES,
        (
            _native(
                "native-artifact-conversion",
                "trusted source checkpoints converted into native arrays and registered recipes",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
                capabilities=(INTERCHANGE_CAPABILITY,),
            ),
            _native(
                "fresh-environment-artifact-restore",
                "converted artifacts restored without numerical source packages",
                _D.SPATIAL_SINGLE_PHYSICS,
                "operations-restart",
                capabilities=(INTERCHANGE_CAPABILITY,),
            ),
            _native(
                "multi-model-simulation-training-restart",
                "complete multi-model dynamics and training continuation",
                _D.SPATIAL_COUPLED,
                "operations-restart",
                capabilities=(
                    INTERCHANGE_CAPABILITY,
                    DYNAMICS_CAPABILITY,
                    TRAINING_CAPABILITY,
                ),
                producers=(),
            ),
            _external(
                "pretrained-weight-admission",
                "per-checkpoint lawful admission of pretrained weights for conversion",
                "per-checkpoint-weight-rights-admission",
                _D.SEMANTIC,
                "rights-security",
                capabilities=(INTERCHANGE_CAPABILITY,),
            ),
        ),
    ),
    _Row(
        "C04",
        (NEIGHBORHOOD_CAPABILITY,),
        _CHEMISTRY,
        (ExecutionRegime.STATIC, ExecutionRegime.TRANSIENT),
        (WorkflowClass.FORWARD, WorkflowClass.RESTART_REPLAY),
        ("tools/particle_neighbor_scaling.py",),
        _OPS,
        (
            _native(
                "cell-list-and-cluster-tile-routes",
                "prepared cell-list and Morton cluster-tile image routes",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "dense-reference-route",
                "bounded dense all-pairs image reference route",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "tiled-reference-route",
                "prepared scalar and tile-bounded image reference traversal",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "role-bundle-cutoff-tiers-rebuild",
                "independent role resources, cutoff tiers, layouts, certified rebuild and distinct capacity failure",
                _D.SPATIAL_SINGLE_PHYSICS,
                "implementation",
                "operations-restart",
            ),
            _native(
                "differentiable-pair-actions",
                "fixed-topology differentiable custom pair actions",
                _D.SPATIAL_SINGLE_PHYSICS,
                "derivative",
                capabilities=(NEIGHBORHOOD_CAPABILITY, ROUTE_REDUCTION_CAPABILITY),
            ),
            _native(
                "neighbor-resource-performance",
                "search, rebuild and role-view resource envelope",
                _D.SPATIAL_SINGLE_PHYSICS,
                "resource-performance",
            ),
        ),
        topologies=(TopologyRegime.FIXED, TopologyRegime.TOPOLOGY_TRANSITION),
    ),
    _Row(
        "C05",
        (ROUTE_REDUCTION_CAPABILITY, SPLINE_CAPABILITY),
        _CHEMISTRY,
        (ExecutionRegime.STATIC,),
        (WorkflowClass.FORWARD,),
        (),
        _OPS,
        (
            _native(
                "segmented-route-reduction",
                "masked segmented sum, mean, extrema, dot, multiply, matvec and RMS with adjoints",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL_DERIVATIVE,
                capabilities=(ROUTE_REDUCTION_CAPABILITY,),
            ),
            _native(
                "particle-grid-spline-transfer",
                "bounded tensor B-spline spread, gather, gradient and channel transfer",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL_DERIVATIVE,
                capabilities=(SPLINE_CAPABILITY,),
            ),
        ),
        couplings=(CouplingLocation.BULK, CouplingLocation.PARTICLE_FIELD),
        carriers=(CarrierRepresentation.ATOMISTIC, CarrierRepresentation.PARTICLE),
    ),
    _Row(
        "C06",
        (ELECTROSTATICS_CAPABILITY,),
        (*_CHEMISTRY, PhysicsField.ELECTRIC),
        (ExecutionRegime.STATIC,),
        (WorkflowClass.FORWARD,),
        (),
        _OPS,
        (
            _native(
                "direct-and-damped-shifted-coulomb",
                "finite direct, screened and damped-shifted-force point-charge Coulomb",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL_DERIVATIVE,
            ),
            _native(
                "ewald-and-particle-mesh-ewald",
                "periodic Ewald and PME with declared boundary, background and influence",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL_DERIVATIVE,
            ),
            _native(
                "slab-correction",
                "admitted Yeh-Berkowitz slab semantics for two periodic axes",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL_DERIVATIVE,
            ),
            _native(
                "predicted-charge-field-response",
                "conservative response through predicted charges",
                _D.SPATIAL_COUPLED,
                "derivative",
                capabilities=(ELECTROSTATICS_CAPABILITY, AIMNET2_CAPABILITY),
            ),
        ),
        couplings=(
            CouplingLocation.BULK,
            CouplingLocation.BOUNDARY,
            CouplingLocation.PARTICLE_FIELD,
        ),
    ),
    _Row(
        "C07",
        (MULTIPOLE_CAPABILITY, LATENT_CAPABILITY, RECIPROCAL_CAPABILITY),
        (*_CHEMISTRY, PhysicsField.ELECTRIC),
        (ExecutionRegime.STATIC,),
        (WorkflowClass.FORWARD,),
        (),
        (*_OPS, "les"),
        (
            _native(
                "multipole-basis-electrostatics",
                "point and Gaussian multipoles to quadrupole with exact bases, self, exclusion and geometry",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL_DERIVATIVE,
                capabilities=(MULTIPOLE_CAPABILITY,),
            ),
            _native(
                "induced-dipole-response",
                "owner polarization response with declared solver sensitivity",
                _D.SPATIAL_COUPLED,
                *_IMPL_DERIVATIVE,
                capabilities=(MULTIPOLE_CAPABILITY, RECIPROCAL_CAPABILITY),
            ),
            _native(
                "latent-charge-ewald",
                "latent charges and latent Ewald energy distinct from physical charges",
                _D.SPATIAL_COUPLED,
                *_IMPL_DERIVATIVE,
                capabilities=(LATENT_CAPABILITY,),
            ),
        ),
        couplings=(CouplingLocation.BULK, CouplingLocation.PARTICLE_FIELD),
    ),
    _Row(
        "C08",
        (DISPERSION_CAPABILITY,),
        (*_CHEMISTRY, PhysicsField.ELECTRONIC_QUANTUM),
        (ExecutionRegime.STATIC,),
        (WorkflowClass.FORWARD,),
        (),
        (*_BOTH, "grimme-dftd3"),
        (
            _native(
                "conventional-d3bj",
                "two-body D3(BJ) with truncated coordination and admitted cutoff policies",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL_DERIVATIVE,
            ),
            _native(
                "fourier-d3",
                "FourierD3 modified counting and signed low-rank C6 decomposition",
                _D.SPATIAL_COUPLED,
                *_IMPL_DERIVATIVE,
            ),
            _external(
                "d3-element-table-fidelity",
                "element fidelity under the published D3 reference tables",
                "admitted-d3-reference-tables",
                _D.SPATIAL_SINGLE_PHYSICS,
                "numerical",
                "rights-security",
            ),
        ),
        couplings=(CouplingLocation.BULK, CouplingLocation.PARTICLE_FIELD),
    ),
    _Row(
        "C09",
        (DYNAMICS_CAPABILITY,),
        _CELL,
        (ExecutionRegime.TRANSIENT, ExecutionRegime.STOCHASTIC),
        (WorkflowClass.FORWARD, WorkflowClass.RESTART_REPLAY),
        ("tools/atomistic_dynamics_benchmarks.py",),
        _TOOLKIT,
        (
            _native(
                "fixed-cell-ensembles",
                "NVE and NVT integrators and thermostats over learned and mixed programs",
                _D.SPATIAL_COUPLED,
                *_IMPL,
            ),
            _native(
                "variable-cell-ensembles",
                "NPT and NPH barostats with learned dynamic-cell state",
                _D.SPATIAL_COUPLED,
                *_IMPL_DERIVATIVE,
            ),
            _native(
                "constraints-and-failure-separation",
                "constraints with distinct capacity, constraint and scientific failure",
                _D.SPATIAL_COUPLED,
                "implementation",
            ),
            _native(
                "dynamics-continuation",
                "complete controller, cell and force-cache continuation",
                _D.SPATIAL_COUPLED,
                "operations-restart",
            ),
            _native(
                "ensemble-measure-validation",
                "independent equilibrium-measure references for admitted ensembles",
                _D.SCIENTIFICALLY_VALIDATED,
                "scientific-validation",
            ),
        ),
    ),
    _Row(
        "C10",
        (RELAXATION_CAPABILITY, REACTION_PATH_CAPABILITY),
        _CELL,
        (ExecutionRegime.STATIC,),
        (WorkflowClass.OPTIMIZATION,),
        _CAMPAIGN_PRODUCERS,
        _TOOLKIT,
        (
            _native(
                "coordinate-cell-relaxation",
                "device-resident batched FIRE2 and L-BFGS coordinate and cell relaxation",
                _D.SPATIAL_COUPLED,
                *_IMPL,
                capabilities=(RELAXATION_CAPABILITY,),
            ),
            _native(
                "grouped-neb-climbing-image",
                "grouped periodic fixed-cell NEB and climbing-image paths",
                _D.SPATIAL_COUPLED,
                *_IMPL,
                capabilities=(REACTION_PATH_CAPABILITY,),
            ),
        ),
    ),
    _Row(
        "C11",
        (CAMPAIGN_CAPABILITY,),
        _THERMAL,
        (ExecutionRegime.TRANSIENT, ExecutionRegime.MULTIRATE),
        (WorkflowClass.FORWARD, WorkflowClass.RESTART_REPLAY),
        _CAMPAIGN_PRODUCERS,
        _TOOLKIT,
        (
            _native(
                "heterogeneous-cohort-refill-stages",
                "heterogeneous bounded cohorts, inflight refill, stages, force reprime and whole groups",
                _D.SPATIAL_COUPLED,
                *_IMPL,
            ),
            _native(
                "backpressure-exact-publication",
                "bounded backpressure, exactly-once publication and continuation",
                _D.SPATIAL_COUPLED,
                "operations-restart",
            ),
            _native(
                "multi-process-stage-transport",
                "observed single-host multi-process stage transport",
                _D.ADAPTIVE_DISTRIBUTED,
                "hardware-topology",
                producers=_DISTRIBUTED_PRODUCERS,
            ),
        ),
    ),
    _Row(
        "C12",
        (DOMAIN_CAPABILITY, RECIPROCAL_CAPABILITY),
        _CHEMISTRY,
        (ExecutionRegime.STATIC, ExecutionRegime.TRANSIENT),
        (WorkflowClass.FORWARD, WorkflowClass.RESTART_REPLAY),
        _DISTRIBUTED_PRODUCERS,
        _BOTH,
        (
            _native(
                "owner-local-domain-execution",
                "owner-local halo and graph-partition execution with layer halos and reverse cotangents",
                _D.ADAPTIVE_DISTRIBUTED,
                *_IMPL_DERIVATIVE,
                capabilities=(DOMAIN_CAPABILITY,),
            ),
            _native(
                "model-family-domain-parity",
                "AIMNet2, UMA and MACELES domain execution against single-device references",
                _D.ADAPTIVE_DISTRIBUTED,
                "numerical",
                "derivative",
                capabilities=(
                    DOMAIN_CAPABILITY,
                    AIMNET2_CAPABILITY,
                    MACELES_CAPABILITY,
                    UMA_CAPABILITY,
                ),
            ),
            _native(
                "migration-domain-checkpoint",
                "owner migration and domain checkpoint publication and restore",
                _D.ADAPTIVE_DISTRIBUTED,
                "operations-restart",
                capabilities=(DOMAIN_CAPABILITY,),
            ),
            _native(
                "slab-reciprocal-distribution",
                "distributed slab FFT and mode-chunk reciprocal execution",
                _D.ADAPTIVE_DISTRIBUTED,
                *_IMPL_DERIVATIVE,
                capabilities=(RECIPROCAL_CAPABILITY,),
            ),
            _native(
                "pencil-reciprocal-distribution",
                "particle patches bound to the canonical pencil spectral layout",
                _D.ADAPTIVE_DISTRIBUTED,
                *_IMPL_DERIVATIVE,
                capabilities=(RECIPROCAL_CAPABILITY, DISTRIBUTED_SPECTRAL_CAPABILITY),
            ),
            _native(
                "domain-dispersion-reciprocal-terms",
                "LJ-PME and FourierD3 reciprocal terms composed into owner-local domain execution",
                _D.ADAPTIVE_DISTRIBUTED,
                *_IMPL_DERIVATIVE,
                capabilities=(
                    DOMAIN_CAPABILITY,
                    RECIPROCAL_CAPABILITY,
                    DISPERSION_CAPABILITY,
                ),
            ),
            _native(
                "single-host-multi-process-execution",
                "observed one-host multi-process domain execution",
                _D.ADAPTIVE_DISTRIBUTED,
                "hardware-topology",
            ),
            _external(
                "physical-multi-device-execution",
                "observed execution across multiple physical accelerators",
                "multiple-physical-accelerators",
                _D.ADAPTIVE_DISTRIBUTED,
                "hardware-topology",
                producers=_DISTRIBUTED_PRODUCERS,
            ),
            _external(
                "physical-multi-host-execution",
                "observed execution across multiple hosts",
                _MULTI_HOST,
                _D.ADAPTIVE_DISTRIBUTED,
                "hardware-topology",
                producers=_DISTRIBUTED_PRODUCERS,
            ),
        ),
        topologies=(TopologyRegime.FIXED, TopologyRegime.TOPOLOGY_TRANSITION),
    ),
    _Row(
        "C13",
        (TRAINING_DISTRIBUTION_CAPABILITY, DOMAIN_CAPABILITY, CAMPAIGN_CAPABILITY),
        _CHEMISTRY,
        (ExecutionRegime.STATIC, ExecutionRegime.MULTIRATE),
        (WorkflowClass.CALIBRATION, WorkflowClass.RESTART_REPLAY),
        _DISTRIBUTED_PRODUCERS,
        _TOOLKIT,
        (
            _native(
                "data-parallel-training",
                "data-parallel atomistic training with global numerator and support reduction",
                _D.ADAPTIVE_DISTRIBUTED,
                *_IMPL,
                capabilities=(TRAINING_DISTRIBUTION_CAPABILITY,),
            ),
            _native(
                "data-by-domain-training",
                "combined data and owner axes, both larger than one",
                _D.ADAPTIVE_DISTRIBUTED,
                *_IMPL,
                "hardware-topology",
                capabilities=(TRAINING_DISTRIBUTION_CAPABILITY, DOMAIN_CAPABILITY),
            ),
            _native(
                "stage-by-domain-campaign",
                "domain execution inside a backpressured multistage campaign",
                _D.ADAPTIVE_DISTRIBUTED,
                *_IMPL,
                "operations-restart",
                "hardware-topology",
                capabilities=(CAMPAIGN_CAPABILITY, DOMAIN_CAPABILITY),
            ),
            _native(
                "distributed-training-continuation",
                "distributed training sections, loader cursors and global commit",
                _D.ADAPTIVE_DISTRIBUTED,
                "operations-restart",
                capabilities=(TRAINING_DISTRIBUTION_CAPABILITY,),
            ),
            _external(
                "physical-multi-host-training",
                "observed multi-host distributed training and stage pipelines",
                _MULTI_HOST,
                _D.ADAPTIVE_DISTRIBUTED,
                "hardware-topology",
                producers=_DISTRIBUTED_PRODUCERS,
            ),
        ),
    ),
    _Row(
        "C14",
        (TRAINING_CAPABILITY,),
        _CHEMISTRY,
        (ExecutionRegime.STATIC,),
        (WorkflowClass.CALIBRATION, WorkflowClass.RESTART_REPLAY),
        _TRAINING_PRODUCERS,
        _TOOLKIT,
        (
            _native(
                "objectives-optimizers-schedules",
                "streaming objectives, residual forms, schedules and optimizers",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "energy-force-stress-property-supervision",
                "energy, force, stress and additional-property supervision",
                _D.SPATIAL_SINGLE_PHYSICS,
                "derivative",
            ),
            _native(
                "validation-ema-freezing-adaptation-accumulation",
                "validation, EMA selection, freezing, low-rank adaptation and accumulation",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "training-continuation-rollback",
                "failed-update rollback and mid-epoch and accumulation continuation",
                _D.SPATIAL_SINGLE_PHYSICS,
                "operations-restart",
            ),
        ),
    ),
    _Row(
        "C15",
        (DISTILLATION_CAPABILITY,),
        _CHEMISTRY,
        (ExecutionRegime.STATIC, ExecutionRegime.TRANSIENT),
        (WorkflowClass.CALIBRATION, WorkflowClass.RESTART_REPLAY),
        (),
        _TOOLKIT,
        (
            _native(
                "offline-native-teacher-distillation",
                "offline distillation from frozen native teachers",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "on-policy-replay",
                "on-policy labeling, bounded replay and phase continuation",
                _D.SPATIAL_COUPLED,
                "implementation",
                "operations-restart",
                capabilities=(DISTILLATION_CAPABILITY, CAMPAIGN_CAPABILITY),
            ),
            _native(
                "embedding-hvp-matching",
                "embedding and mixed-order coordinate HVP matching",
                _D.SPATIAL_SINGLE_PHYSICS,
                "derivative",
            ),
            _native(
                "distribution-objective-measure",
                "distribution objectives with admitted measure normalization",
                _D.SPATIAL_COUPLED,
                "numerical",
            ),
        ),
    ),
    _Row(
        "C16",
        (TRAINING_CAPABILITY,),
        _CHEMISTRY,
        (ExecutionRegime.STATIC,),
        (WorkflowClass.CALIBRATION, WorkflowClass.RESTART_REPLAY),
        _TRAINING_PRODUCERS,
        _TOOLKIT,
        (
            _native(
                "atom-edge-budget-batching",
                "atom and edge budget batching",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "weighted-mixing-storage-boundaries",
                "weighted source mixing and structure-store boundaries",
                _D.SPATIAL_SINGLE_PHYSICS,
                "implementation",
                "operations-restart",
            ),
            _native(
                "bounded-prefetch-staging-reporting",
                "bounded prefetch, staging, reporting and profiling",
                _D.SPATIAL_SINGLE_PHYSICS,
                "resource-performance",
            ),
        ),
    ),
    _Row(
        "C17",
        (SAMPLING_CAPABILITY,),
        _CELL,
        (ExecutionRegime.TRANSIENT, ExecutionRegime.STOCHASTIC),
        (WorkflowClass.FORWARD, WorkflowClass.RESTART_REPLAY),
        _SAMPLING_PRODUCERS,
        _TOOLKIT,
        (
            _native(
                "metric-correct-collective-variables",
                "metric and image-branch correct collective variables",
                _D.SPATIAL_COUPLED,
                *_IMPL_DERIVATIVE,
            ),
            _native(
                "umbrella-wall-metadynamics-histories",
                "umbrella, wall and metadynamics biases with owned bounded histories",
                _D.SPATIAL_COUPLED,
                *_IMPL,
                "operations-restart",
            ),
            _native(
                "abf-sign-measure",
                "adaptive biasing force with correct sign and base measure",
                _D.SPATIAL_COUPLED,
                *_IMPL,
            ),
            _native(
                "dynamic-cell-bias-response",
                "bias strain response under dynamic cells",
                _D.SPATIAL_COUPLED,
                "derivative",
            ),
            _native(
                "sampling-measure-validation",
                "independent ensemble-measure references for biased sampling",
                _D.SCIENTIFICALLY_VALIDATED,
                "scientific-validation",
            ),
        ),
    ),
    _Row(
        "C18",
        (SAMPLING_CAPABILITY, DYNAMICS_CAPABILITY),
        _THERMAL,
        (ExecutionRegime.STOCHASTIC,),
        (WorkflowClass.FORWARD, WorkflowClass.RESTART_REPLAY),
        _SAMPLING_PRODUCERS,
        _TOOLKIT,
        (
            _native(
                "bias-aware-replica-exchange",
                "bias-aware temperature and window replica exchange",
                _D.SPATIAL_COUPLED,
                *_IMPL,
            ),
            _native(
                "history-ownership-thermodynamic-rebinding",
                "walker, state and shared history ownership with thermodynamic rebinding",
                _D.SPATIAL_COUPLED,
                "implementation",
                "operations-restart",
            ),
            _native(
                "reweighting-boundaries",
                "explicit reweighting boundaries of frozen histories",
                _D.SPATIAL_COUPLED,
                "numerical",
            ),
            _native(
                "exchange-detailed-balance-validation",
                "independent detailed-balance and exchange-measure references",
                _D.SCIENTIFICALLY_VALIDATED,
                "scientific-validation",
            ),
        ),
    ),
    _Row(
        "C19",
        (GENERATION_CAPABILITY,),
        _CHEMISTRY,
        (ExecutionRegime.STOCHASTIC,),
        (WorkflowClass.FORWARD, WorkflowClass.RESTART_REPLAY),
        ("benchmarks/atomistic_periodic_generation.py",),
        _TOOLKIT,
        (
            _native(
                "conditional-periodic-samplers",
                "conditional custom, periodic reverse-SDE and probability-flow samplers",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "resumable-bounded-streams",
                "bounded semantic-ordinal streams with exactly-once publication and continuation",
                _D.SPATIAL_SINGLE_PHYSICS,
                "operations-restart",
            ),
            _native(
                "distributed-generation-quotas",
                "global ordinal work and accepted-output quotas across an execution group",
                _D.ADAPTIVE_DISTRIBUTED,
                *_IMPL,
                "operations-restart",
                "hardware-topology",
                producers=("tools/atomistic_generation_distributed_smoke.py",),
            ),
            _external(
                "physical-multi-host-generation",
                "observed multi-host placed generation",
                _MULTI_HOST,
                _D.ADAPTIVE_DISTRIBUTED,
                "hardware-topology",
                producers=("tools/atomistic_generation_distributed_smoke.py",),
            ),
        ),
    ),
    _Row(
        "C20",
        (CRYSTAL_CAPABILITY,),
        (*_CHEMISTRY, PhysicsField.SOLID_MECHANICS),
        (ExecutionRegime.STATIC, ExecutionRegime.STOCHASTIC),
        (WorkflowClass.FORWARD, WorkflowClass.OPTIMIZATION, WorkflowClass.RESTART_REPLAY),
        (
            "benchmarks/atomistic_crystal_dedup.py",
            "benchmarks/atomistic_crystal_packing.py",
        ),
        (*_TOOLKIT, "spglib"),
        (
            _native(
                "exact-settings-asymmetric-unit",
                "exact Hall settings, metric constraints and bounded site-orbit enumeration",
                _D.ANALYTIC_CONTROL,
                *_IMPL,
            ),
            _native(
                "special-site-expansion-constrained-continuation",
                "admitted special-site actions, coset expansion, rational Z' and symmetry-preserving retraction",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "general-position-packing-contact-relief",
                "randomized general-position molecular packing with multi-image contact relief",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "symmetry-relaxation-ranking",
                "symmetry-constrained physical relaxation and same-definition ranking",
                _D.SPATIAL_COUPLED,
                *_IMPL_DERIVATIVE,
                capabilities=(CRYSTAL_CAPABILITY, RELAXATION_CAPABILITY),
            ),
            _native(
                "bounded-equivalence-deduplication",
                "bounded certified equivalence and deduplication",
                _D.SPATIAL_SINGLE_PHYSICS,
                *_IMPL,
            ),
            _native(
                "reconstructible-packing-recipe",
                "persisted scientific packing recipe reconstructed in a fresh process",
                _D.SPATIAL_SINGLE_PHYSICS,
                "implementation",
                "operations-restart",
                capabilities=(CRYSTAL_CAPABILITY, GENERATION_CAPABILITY),
            ),
            _external(
                "space-group-prior-and-atomic-data",
                "weighted space-group priors and packaged radius and volume data",
                "admitted-crystallographic-prior-data",
                _D.SEMANTIC,
                "rights-security",
            ),
            _external(
                "crystal-prediction-scientific-validation",
                "polymorph recovery against independent reference structures",
                "independent-reference-polymorph-data",
                _D.SCIENTIFICALLY_VALIDATED,
                "scientific-validation",
            ),
        ),
    ),
    _Row(
        "C21",
        (_ALL,),
        _CHEMISTRY,
        (ExecutionRegime.STATIC, ExecutionRegime.TRANSIENT, ExecutionRegime.STOCHASTIC),
        (WorkflowClass.FORWARD, WorkflowClass.RESTART_REPLAY),
        (*_CAMPAIGN_PRODUCERS, "tools/atomistic_distributed_qualification.py"),
        _TOOLKIT,
        (
            _native(
                "failure-resource-separation",
                "consumer-visible capacity, scientific and domain failure with resource evidence",
                _D.SPATIAL_COUPLED,
                "implementation",
            ),
            _native(
                "provenance-semantic-rng",
                "provenance and semantic ordinal RNG addressing independent of placement",
                _D.SPATIAL_COUPLED,
                *_IMPL,
            ),
            _native(
                "observation-cancellation",
                "observation and cancellation without partial publication",
                _D.SPATIAL_COUPLED,
                "implementation",
            ),
            _native(
                "crash-nonpublication-full-restart",
                "controlled failure never publishes partial state; full same-profile and migrated restart",
                _D.SPATIAL_COUPLED,
                "operations-restart",
            ),
        ),
    ),
    _Row(
        "C22",
        (_ALL,),
        _CHEMISTRY,
        (ExecutionRegime.STATIC,),
        (WorkflowClass.QUALIFICATION, WorkflowClass.DEPLOYMENT),
        ("tools/check_capability_consistency.py", "tools/check_public_api_manifest.py"),
        _BOTH,
        (
            _native(
                "caller-docs-api-capability-cutover",
                "complete caller, test, docs, example, public API and capability cutover",
                _D.SEMANTIC,
                "implementation",
            ),
            _native(
                "measured-performance-campaigns",
                "phase-separated end-to-end performance campaigns",
                _D.SPATIAL_COUPLED,
                "resource-performance",
                producers=_PERFORMANCE_PRODUCERS,
            ),
            _external(
                "independent-review",
                "independent scientific and implementation review",
                "independent-reviewer",
                _D.SCIENTIFICALLY_VALIDATED,
                "independent-review",
            ),
            _external(
                "signed-release-authorization",
                "signed support-profile release authorization",
                "signed-release-authority",
                _D.SCIENTIFICALLY_VALIDATED,
                "release-authorization",
            ),
        ),
    ),
)


def _requirement_name(row: _Row, obligation: _Obligation, /) -> str:
    return f"atomistic-{row.ledger.lower()}-{obligation.name}"


def atomistic_closure_requirement_names() -> tuple[str, ...]:
    """Return every C01–C22 obligation name accepted as an evidence key."""
    return tuple(
        _requirement_name(row, obligation)
        for row in _LEDGER
        for obligation in row.obligations
    )


def atomistic_closure_matrices(
    observed_evidence: Mapping[str, Sequence[str]] | None = None, /
) -> tuple[CapabilityClosureMatrix, ...]:
    """Classify C01–C22; only observed retained evidence resolves an obligation.

    ``observed_evidence`` maps an obligation name to the content addresses of
    its retained observed artifacts. No resolution is release authorized.
    """
    observed = {} if observed_evidence is None else observed_evidence
    names = set(atomistic_closure_requirement_names())
    for name, ids in observed.items():
        if isinstance(ids, str):
            raise TypeError(f"Evidence for {name!r} must be a sequence of identifiers.")
        if name not in names or not tuple(ids):
            raise ValueError(
                "Observed evidence must name nonempty known atomistic closure obligations."
            )
    every = tuple(
        sorted(
            {_MACE, *(value.capability for value in atomistic_candidate_declarations())}
        )
    )
    matrices: list[CapabilityClosureMatrix] = []
    for row in _LEDGER:
        requirements: list[CapabilityClosureRequirement] = []
        resolutions: list[CapabilityGapResolution] = []
        for obligation in row.obligations:
            name = _requirement_name(row, obligation)
            capabilities = (
                row.capabilities
                if obligation.capabilities is None
                else obligation.capabilities
            )
            if capabilities == (_ALL,):
                capabilities = every
            requirement = CapabilityClosureRequirement.create(
                name,
                minimum_depth=obligation.depth,
                physical_fields=row.fields,
                carriers=row.carriers,
                coupling_locations=row.couplings,
                execution_regimes=row.regimes,
                topology_regimes=row.topologies,
                workflow_classes=row.workflows,
                required_benchmarks=row.producers
                if obligation.producers is None
                else obligation.producers,
                required_evidence_dimensions=obligation.dimensions,
                source_ids=row.sources,
                rationale=f"{row.ledger}: {obligation.scope}",
            )
            evidence = tuple(observed.get(name, ()))
            if evidence:
                disposition = ClosureDisposition.IMPLEMENTED
                depth = obligation.depth
                rationale = OBSERVED
            elif obligation.prerequisite is None:
                disposition = ClosureDisposition.CANDIDATE
                depth = CapabilityDepth.SEMANTIC
                rationale = NATIVE_UNQUALIFIED
            else:
                disposition = ClosureDisposition.MISSING
                depth = CapabilityDepth.SEMANTIC
                rationale = f"{EXTERNAL_PREFIX}{obligation.prerequisite}"
            requirements.append(requirement)
            resolutions.append(
                CapabilityGapResolution.create(
                    requirement.requirement_id,
                    disposition,
                    depth,
                    capability_ids=capabilities,
                    evidence_ids=evidence,
                    rationale=rationale,
                    release_authorized=False,
                )
            )
        matrices.append(
            CapabilityClosureMatrix.create(
                f"atomistic.{row.ledger.lower()}", requirements, resolutions
            )
        )
    return tuple(matrices)


__all__ = [
    "EXTERNAL_PREFIX",
    "NATIVE_UNQUALIFIED",
    "OBSERVED",
    "atomistic_closure_matrices",
    "atomistic_closure_requirement_names",
]
