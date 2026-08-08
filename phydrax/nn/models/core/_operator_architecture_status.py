#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Maturity metadata for public neural-operator architectures."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Literal, TypeAlias


OperatorArchitectureTier: TypeAlias = Literal["stable", "experimental", "research"]
OperatorArchitectureConfiguration: TypeAlias = tuple[tuple[str, object], ...]

from ._encoded_operator import AbstractEncodedOperatorModel
from ._operator import OperatorBatch
from ._operator_capabilities import (
    ConfiguredOperatorContract,
    OperatorCapabilitySpec,
    OperatorCompatibilityReport,
    OperatorProblemSpec,
    OperatorTrainingEvidence,
    OperatorTrainingRequirement,
)


_ALL_GEOMETRIES = (
    "abstract",
    "tensor_grid",
    "point_cloud",
    "cell_complex",
    "graph",
    "simplicial",
    "sphere",
    "manifold",
)
_ALL_REPRESENTATIONS = (
    "generic_channels",
    "scalar",
    "pseudoscalar",
    "vector",
    "covector",
    "tensor",
)


def _capabilities_for(name: str, architecture: str, /) -> OperatorCapabilitySpec:
    if architecture == "HOFNO":
        return OperatorCapabilitySpec(
            source_geometries=("tensor_grid",),
            query_geometries=("tensor_grid",),
            spatial_dimensions=(1, 2, 3),
            source_query_relations=("coincident",),
            axis_requirement="periodic_uniform",
            quadrature="unused",
            masks="all_valid_only",
            topology="unused",
            resolution_transfer=True,
            autoregressive_rollout=True,
        )
    if architecture == "FNO":
        return OperatorCapabilitySpec(
            source_geometries=("tensor_grid",),
            query_geometries=("tensor_grid",),
            spatial_dimensions=(1, 2, 3),
            source_query_relations=("coincident",),
            axis_requirement="uniform",
            quadrature="unused",
            masks="supported",
            topology="unused",
            resolution_transfer=True,
            autoregressive_rollout=True,
        )
    if architecture == "DeepONet":
        return OperatorCapabilitySpec(
            source_geometries=(
                "abstract",
                "tensor_grid",
                "point_cloud",
                "graph",
                "simplicial",
            ),
            query_geometries=("tensor_grid", "point_cloud", "graph", "simplicial"),
            spatial_dimensions=(1, 2, 3),
            source_query_relations=("coincident", "independent"),
            requires_fixed_query=name == "PODDeepONet",
            quadrature="optional",
            masks="supported",
            topology="optional",
            resolution_transfer=True,
        )
    if architecture == "CochainNeuralOperator":
        return OperatorCapabilitySpec(
            source_geometries=("cell_complex",),
            query_geometries=("cell_complex",),
            spatial_dimensions=(1, 2, 3),
            source_query_relations=("coincident", "shared_topology"),
            quadrature="physical_required",
            masks="supported",
            topology="required",
            cochains="required",
            cochain_sides=("primal",),
            input_representations=_ALL_REPRESENTATIONS,
            output_representations=_ALL_REPRESENTATIONS,
            multiple_queries=True,
            resolution_transfer=True,
        )
    if architecture == "GraphNeuralOperator":
        return OperatorCapabilitySpec(
            source_geometries=("graph", "simplicial"),
            query_geometries=("graph", "simplicial"),
            spatial_dimensions=(1, 2, 3),
            source_query_relations=("coincident", "independent"),
            quadrature="physical_required",
            masks="supported",
            topology="required",
            resolution_transfer=True,
            input_representations=_ALL_REPRESENTATIONS,
            output_representations=_ALL_REPRESENTATIONS,
        )
    if architecture == "SFNO":
        return OperatorCapabilitySpec(
            source_geometries=("sphere", "tensor_grid"),
            query_geometries=("sphere", "tensor_grid"),
            spatial_dimensions=(2,),
            source_query_relations=("coincident",),
            quadrature="physical_required",
            masks="supported",
            topology="unused",
            resolution_transfer=True,
        )
    if architecture == "ManifoldSpectralOperator":
        return OperatorCapabilitySpec(
            source_geometries=("manifold", "graph", "simplicial"),
            query_geometries=("manifold", "graph", "simplicial"),
            spatial_dimensions=(2, 3),
            source_query_relations=("coincident",),
            requires_fixed_query=True,
            quadrature="physical_required",
            masks="all_valid_only",
            topology="required",
            resolution_transfer=False,
        )
    if architecture in (
        "GINO",
        "GeometryInformedFlower",
        "RIGNO",
        "GAOT",
        "EqGINO",
        "Transolver",
        "GNOT",
        "GreenKernelOperator",
        "LocalDifferentialOperator",
        "LocalGlobalOperator",
        "LocalIntegralOperator",
        "CoordinateConditionedOperator",
    ):
        representations = (
            _ALL_REPRESENTATIONS
            if architecture == "EqGINO"
            else ("generic_channels", "scalar")
        )
        return OperatorCapabilitySpec(
            source_geometries=("tensor_grid", "point_cloud", "graph", "simplicial"),
            query_geometries=("tensor_grid", "point_cloud", "graph", "simplicial"),
            spatial_dimensions=(1, 2, 3),
            source_query_relations=("coincident", "independent"),
            quadrature="physical_required",
            masks="supported",
            topology="optional",
            input_representations=representations,
            encode_once_decode_many=architecture
            in ("EqGINO", "Transolver", "CoordinateConditionedOperator"),
            output_representations=representations,
            resolution_transfer=True,
        )
    if architecture == "DiagonalStateSpaceMixer":
        return OperatorCapabilitySpec(
            source_geometries=("tensor_grid", "point_cloud"),
            query_geometries=("tensor_grid", "point_cloud"),
            spatial_dimensions=(1,),
            source_query_relations=("coincident",),
            axis_requirement="none",
            quadrature="unused",
            masks="supported",
            topology="unused",
            resolution_transfer=False,
        )
    if architecture == "LaplaceTemporalOperator":
        return OperatorCapabilitySpec(
            source_geometries=("tensor_grid", "point_cloud"),
            query_geometries=("tensor_grid", "point_cloud"),
            spatial_dimensions=(1,),
            source_query_relations=("coincident", "independent"),
            quadrature="optional",
            masks="supported",
            topology="unused",
            resolution_transfer=True,
        )
    if architecture in (
        "CNO",
        "UNO",
        "Flower",
        "IFNO",
        "AxialFactorizedFNO",
        "WaveletNeuralOperator",
        "MultiwaveletOperator",
        "Poseidon",
        "DPOT",
    ):
        return OperatorCapabilitySpec(
            source_geometries=("tensor_grid",),
            query_geometries=("tensor_grid",),
            spatial_dimensions=(1, 2, 3),
            source_query_relations=("coincident",),
            axis_requirement="uniform",
            quadrature="optional",
            masks="supported",
            topology="unused",
            resolution_transfer=architecture
            in (
                "CNO",
                "UNO",
                "Flower",
                "IFNO",
                "Poseidon",
            ),
            autoregressive_rollout=architecture
            in (
                "CNO",
                "UNO",
                "Flower",
                "IFNO",
                "DPOT",
                "Poseidon",
            ),
        )
    if architecture == "ConditionalFlowFunctionOperator":
        return OperatorCapabilitySpec(
            source_geometries=_ALL_GEOMETRIES,
            query_geometries=("tensor_grid", "point_cloud"),
            spatial_dimensions=(1, 2, 3),
            source_query_relations=("coincident", "independent"),
            requires_fixed_query=True,
            quadrature="optional",
            masks="supported",
            topology="unused",
            input_representations=_ALL_REPRESENTATIONS,
            output_representations=_ALL_REPRESENTATIONS,
        )
    if architecture in ("UPT", "ABUPT"):
        return OperatorCapabilitySpec(
            source_geometries=_ALL_GEOMETRIES,
            query_geometries=_ALL_GEOMETRIES,
            spatial_dimensions=(1, 2, 3),
            source_query_relations=("coincident", "independent"),
            quadrature="optional",
            masks="supported",
            topology="optional",
            input_representations=_ALL_REPRESENTATIONS,
            encode_once_decode_many=True,
            output_representations=_ALL_REPRESENTATIONS,
            resolution_transfer=True,
            multiple_queries=architecture == "ABUPT",
            autoregressive_rollout=True,
        )
    return OperatorCapabilitySpec(
        source_geometries=_ALL_GEOMETRIES,
        query_geometries=_ALL_GEOMETRIES,
        spatial_dimensions=(1, 2, 3),
        source_query_relations=("coincident", "independent"),
        quadrature="optional",
        masks="supported",
        topology="optional",
        input_representations=_ALL_REPRESENTATIONS,
        encode_once_decode_many=architecture in ("CoDANO", "InContextOperator"),
        multiple_queries=architecture == "CoDANO",
        output_representations=_ALL_REPRESENTATIONS,
    )


def _training_for(name: str, architecture: str, /) -> OperatorTrainingRequirement:
    if architecture in ("Poseidon", "DPOT"):
        return OperatorTrainingRequirement(
            regime="pretrained_system",
            pretrained_weights_required=True,
            corpus_description="large multi-PDE pretraining corpus",
            claim_scope=f"published {architecture} pretrained-system behavior",
        )
    if architecture == "InContextOperator":
        return OperatorTrainingRequirement(
            regime="task_distribution",
            corpus_description="diverse prompted operator-task distribution",
            claim_scope="in-context adaptation to unseen prompted operators",
        )
    if architecture == "CoDANO":
        return OperatorTrainingRequirement(
            regime="task_distribution",
            corpus_description="multiple coupled PDE systems",
            claim_scope="few-shot cross-system multiphysics transfer",
        )
    return OperatorTrainingRequirement(
        regime="task_specific",
        claim_scope=f"task-specific {name} operator learning",
    )


@dataclass(frozen=True, slots=True)
class OperatorArchitectureStatus:
    """Immutable maturity and recommendation status for an operator architecture."""

    name: str
    architecture: str
    configuration: OperatorArchitectureConfiguration
    tier: OperatorArchitectureTier
    recommendation_eligible: bool
    evidence: str
    capabilities: OperatorCapabilitySpec
    training: OperatorTrainingRequirement


def _status(
    name: str,
    architecture: str,
    tier: OperatorArchitectureTier,
    evidence: str,
    /,
    *,
    configuration: OperatorArchitectureConfiguration = (),
) -> OperatorArchitectureStatus:
    return OperatorArchitectureStatus(
        name=name,
        architecture=architecture,
        configuration=configuration,
        tier=tier,
        recommendation_eligible=tier == "stable",
        evidence=evidence,
        capabilities=_capabilities_for(name, architecture),
        training=_training_for(name, architecture),
    )


_OPERATOR_ARCHITECTURE_STATUSES = {
    "FNO": _status(
        "FNO",
        "FNO",
        "stable",
        "Native batching, resolution transfer, and spectral factorization are regression tested.",
    ),
    "HOFNO": _status(
        "HOFNO",
        "HOFNO",
        "experimental",
        "Implementation and numerical alias-suppression contracts are verified; "
        "decision-grade higher-order benchmark evidence remains incomplete.",
    ),
    "TFNO": _status(
        "TFNO",
        "FNO",
        "stable",
        "Tucker factorization uses the validated public FNO factorization path.",
        configuration=(("factorization", "tucker"),),
    ),
    "DeepONet": _status(
        "DeepONet",
        "DeepONet",
        "stable",
        "Branch-trunk evaluation, multiple inputs, and chunked queries are regression tested.",
    ),
    "MIONet": _status(
        "MIONet",
        "DeepONet",
        "stable",
        "MIONet is the validated multiple-branch product-fusion DeepONet configuration.",
        configuration=(("branch", "mapping"), ("fusion", "product")),
    ),
    "PODDeepONet": _status(
        "PODDeepONet",
        "DeepONet",
        "stable",
        "The fixed-basis trunk configuration has deterministic projection and batching checks.",
        configuration=(("trunk", "pod_basis"),),
    ),
    "GINO": _status(
        "GINO",
        "GINO",
        "research",
        "Per-case geometry, measure-aware graph transfer, latent FNO execution, "
        "and five-seed capacity-matched remeshing/geometry benchmarks are covered. "
        "Results show regime-dependent GINO/Flower tradeoffs rather than uniform "
        "dominance, so the architecture remains research tier.",
    ),
    "GeometryInformedFlower": _status(
        "GeometryInformedFlower",
        "GeometryInformedFlower",
        "research",
        "Weighted point-set transfer, explicit occupancy/SDF support, conditioned "
        "latent Flower execution, diagnostics, gradients, and physical source/query "
        "conservation have focused checks. Five-seed capacity-matched decision "
        "benchmarks cover remeshing, extrapolation, boundary shifts, sensor damage, "
        "and conservative transport. They support the resolution-consistent default, "
        "but native mesh topology and verified upstream Flower parity remain absent.",
    ),
    "RIGNO": _status(
        "RIGNO",
        "RIGNO",
        "research",
        "Per-case regional sampling, measure-aware latent graph processing, and "
        "arbitrary query decoding are regression tested; family parity and "
        "decision-grade benchmark evidence remain pending.",
    ),
    "GAOT": _status(
        "GAOT",
        "GAOT",
        "research",
        "Multiscale measure-aware graph attention, patchwise transformer execution, "
        "and arbitrary-query decoding are regression tested; family parity and "
        "decision-grade benchmark evidence remain pending.",
    ),
    "CNO": _status(
        "CNO",
        "CNO",
        "experimental",
        "Odd-grid and native-batch checks exist; broader validation is still pending.",
    ),
    "Flower": _status(
        "Flower",
        "Flower",
        "research",
        "Paper-faithful defaults plus resolution-consistent transitions, physical "
        "nonuniform grids, masks, independent queries, conditioned/probabilistic "
        "routes, conservation, diagnostics, and gradients have focused checks; "
        "decision-grade multi-seed transport and wave evidence remains pending.",
    ),
    "GraphNeuralOperator": _status(
        "GraphNeuralOperator",
        "GraphNeuralOperator",
        "experimental",
        "GraphIR batching, measure-aware messages, and source-to-query transfer "
        "are tested; decision-grade geometry benchmarks remain pending.",
    ),
    "SFNO": _status(
        "SFNO",
        "SFNO",
        "experimental",
        "Spherical resolution checks exist; broader validation is still pending.",
    ),
    "LocalDifferentialOperator": _status(
        "LocalDifferentialOperator",
        "LocalDifferentialOperator",
        "experimental",
        "Focused local-differential checks exist; broader validation is still pending.",
    ),
    "LocalGlobalOperator": _status(
        "LocalGlobalOperator",
        "LocalGlobalOperator",
        "experimental",
        "The local-global composition is public but lacks stable-tier validation breadth.",
    ),
    "LocalIntegralOperator": _status(
        "LocalIntegralOperator",
        "LocalIntegralOperator",
        "experimental",
        "Focused local-integral checks exist; broader validation is still pending.",
    ),
    "OperatorAttention": _status(
        "OperatorAttention",
        "OperatorAttention",
        "experimental",
        "Shape and measure-aware checks exist; broader validation is still pending.",
    ),
    "SliceAttention": _status(
        "SliceAttention",
        "SliceAttention",
        "experimental",
        "Shape and measure-aware checks exist; broader validation is still pending.",
    ),
    "AxialOperatorAttention": _status(
        "AxialOperatorAttention",
        "AxialOperatorAttention",
        "experimental",
        "Shape checks exist; broader validation is still pending.",
    ),
    "CodomainAttention": _status(
        "CodomainAttention",
        "CodomainAttention",
        "experimental",
        "The public attention layer lacks stable-tier validation breadth.",
    ),
    "UNO": _status(
        "UNO",
        "UNO",
        "research",
        "The research architecture has focused numerical checks but limited validation breadth.",
    ),
    "LaplaceTemporalOperator": _status(
        "LaplaceTemporalOperator",
        "LaplaceTemporalOperator",
        "research",
        "Pole stability and causality are checked, but the architecture remains research-stage.",
    ),
    "IFNO": _status(
        "IFNO",
        "IFNO",
        "experimental",
        "Shared-weight spectral iteration, static execution, and convergence diagnostics "
        "have focused regression coverage; decision-grade benchmarks remain pending.",
    ),
    "AxialFactorizedFNO": _status(
        "AxialFactorizedFNO",
        "AxialFactorizedFNO",
        "experimental",
        "Sequential one-axis spectral transforms and factorized weights have focused "
        "regression coverage; broad PDE validation remains pending.",
    ),
    "WaveletNeuralOperator": _status(
        "WaveletNeuralOperator",
        "WaveletNeuralOperator",
        "research",
        "Exact multiresolution reconstruction is checked, but decision-grade "
        "cross-family operator benchmarks remain pending.",
    ),
    "MultiwaveletOperator": _status(
        "MultiwaveletOperator",
        "MultiwaveletOperator",
        "research",
        "Polynomial multiwavelet analysis and synthesis have focused numerical checks; "
        "broad PDE validation remains pending.",
    ),
    "ManifoldSpectralOperator": _status(
        "ManifoldSpectralOperator",
        "ManifoldSpectralOperator",
        "research",
        "Laplace-eigenbasis execution and discretization checks exist; validation "
        "across independently meshed manifolds remains pending.",
    ),
    "CoordinateConditionedOperator": _status(
        "CoordinateConditionedOperator",
        "CoordinateConditionedOperator",
        "research",
        "Nonlinear function-conditioned query decoding has focused checks; "
        "decision-grade NOMAD-style comparisons remain pending.",
    ),
    "UPT": _status(
        "UPT",
        "UPT",
        "research",
        "Latent token encoding, processing, and arbitrary-query decoding have focused "
        "checks; broad scientific validation remains pending.",
    ),
    "ABUPT": _status(
        "ABUPT",
        "ABUPT",
        "research",
        "Typed anchored branches and bidirectional latent interactions have focused "
        "checks; broad scientific validation remains pending.",
    ),
    "CoDANO": _status(
        "CoDANO",
        "CoDANO",
        "research",
        "Typed heterogeneous field fusion and codomain attention have focused checks; "
        "multiphysics decision benchmarks remain pending.",
    ),
    "EqGINO": _status(
        "EqGINO",
        "EqGINO",
        "research",
        "O(3)-equivariant transfer and representation checks exist; broad "
        "geometry-dependent PDE validation remains pending.",
    ),
    "InContextOperator": _status(
        "InContextOperator",
        "InContextOperator",
        "research",
        "Prompt masking, demonstration conditioning, and query decoding have focused "
        "checks; in-context task-distribution benchmarks remain pending.",
    ),
    "GaussianFunctionOperator": _status(
        "GaussianFunctionOperator",
        "GaussianFunctionOperator",
        "research",
        "Coherent diagonal-plus-low-rank function distributions and likelihoods are "
        "checked; calibrated operator-UQ benchmarks remain pending.",
    ),
    "ConditionalFlowFunctionOperator": _status(
        "ConditionalFlowFunctionOperator",
        "ConditionalFlowFunctionOperator",
        "experimental",
        "Conditional FlowJAX residual densities are supported on shared fixed query "
        "geometries; arbitrary-query and resolution-transfer claims are excluded.",
    ),
    "Poseidon": _status(
        "Poseidon",
        "Poseidon",
        "research",
        "Native multiscale scOT execution, time conditioning, masks, and gradients "
        "have focused checks; pretrained foundation-model evidence is not bundled.",
    ),
    "DPOT": _status(
        "DPOT",
        "DPOT",
        "research",
        "AFNO execution, autoregressive history semantics, and denoising corruption "
        "have focused checks; large-scale pretraining evidence is not bundled.",
    ),
    "Transolver": _status(
        "Transolver",
        "Transolver",
        "research",
        "Quadrature-aware physical slices and arbitrary-query decoding have focused "
        "checks; decision-grade PDE benchmarks remain pending.",
    ),
    "TransolverPlusPlus": _status(
        "TransolverPlusPlus",
        "Transolver",
        "research",
        "Overlapping normalized physical-slice memberships are implemented and "
        "distinguished from hard slices; broad Transolver++ validation remains pending.",
        configuration=(("slice_top_k", "greater_than_one"),),
    ),
    "GNOT": _status(
        "GNOT",
        "GNOT",
        "research",
        "Heterogeneous source encoders, measure-aware cross-attention, and learned "
        "source gates have focused checks; broad multiphysics validation remains pending.",
    ),
    "DiagonalStateSpaceMixer": _status(
        "DiagonalStateSpaceMixer",
        "DiagonalStateSpaceMixer",
        "research",
        "Stable conjugate poles, exact irregular-time input integration, masked "
        "ragged schedules, and equivalent recurrent, dense, and associative "
        "execution have focused checks; broad sequence benchmarks remain pending.",
    ),
    "KoopmanTemporalOperator": _status(
        "KoopmanTemporalOperator",
        "KoopmanTemporalOperator",
        "research",
        "Stable latent evolution, semigroup consistency, and nonperiodic query times "
        "have focused checks; broad temporal PDE validation remains pending.",
    ),
    "CochainNeuralOperator": _status(
        "CochainNeuralOperator",
        "CochainNeuralOperator",
        "research",
        "PhydraX-native metric DEC routes, cochain field semantics, and exact harmonic "
        "projection have focused checks; this is a reconstruction and is not claimed "
        "to reproduce the unpublished upstream TNO implementation.",
    ),
    "GreenKernelOperator": _status(
        "GreenKernelOperator",
        "GreenKernelOperator",
        "research",
        "Separate volume-forcing and boundary-kernel paths use physical quadrature; "
        "geometry-family and boundary-condition benchmarks remain pending.",
    ),
}

OPERATOR_ARCHITECTURE_STATUSES = MappingProxyType(_OPERATOR_ARCHITECTURE_STATUSES)


def _normalize_architecture_name(name: str, /) -> str:
    if not isinstance(name, str):
        raise ValueError(
            f"Operator architecture name must be a string; got {type(name).__name__}."
        )
    return "".join(
        "plus" if character == "+" else character
        for character in name.casefold()
        if character.isalnum() or character == "+"
    )


_ALIAS_TARGETS = {
    "fourierneuraloperator": "FNO",
    "higherorderfno": "HOFNO",
    "higherorderfourierneuraloperator": "HOFNO",
    "tuckerfno": "TFNO",
    "tuckerfactorizedfno": "TFNO",
    "tensorizedfno": "TFNO",
    "deepoperatornetwork": "DeepONet",
    "multipleinputoperatornetwork": "MIONet",
    "poddeeponet": "PODDeepONet",
    "properorthogonaldecompositiondeeponet": "PODDeepONet",
    "graphoperator": "GraphNeuralOperator",
    "resolutioninvariantgraphneuraloperator": "RIGNO",
    "geometryawareoperatortransformer": "GAOT",
    "geometryinformedflowerlearned": "GeometryInformedFlower",
    "domainconditionedflower": "GeometryInformedFlower",
    "conservativegeometryinformedflower": "GeometryInformedFlower",
    "graphneuraloperator": "GraphNeuralOperator",
    "multiinputoperatornetwork": "MIONet",
    "convolutionalneuraloperator": "CNO",
    "flowers": "Flower",
    "warpdriveneuralpdesolver": "Flower",
    "sphericalfourierneuraloperator": "SFNO",
    "sphericalfno": "SFNO",
    "localdifferential": "LocalDifferentialOperator",
    "localglobal": "LocalGlobalOperator",
    "localintegral": "LocalIntegralOperator",
    "operatormultiheadattention": "OperatorAttention",
    "axialattention": "AxialOperatorAttention",
    "ushapedneuraloperator": "UNO",
    "laplace": "LaplaceTemporalOperator",
    "laplaceneuraloperator": "LaplaceTemporalOperator",
    "implicitfourierneuraloperator": "IFNO",
    "axialfno": "AxialFactorizedFNO",
    "factorizedfno": "AxialFactorizedFNO",
    "wno": "WaveletNeuralOperator",
    "multiwavelettransformoperator": "MultiwaveletOperator",
    "mwt": "MultiwaveletOperator",
    "manifoldneuraloperator": "ManifoldSpectralOperator",
    "nomad": "CoordinateConditionedOperator",
    "universalphysicstransformer": "UPT",
    "anchoredbrancheduniversalphysicstransformer": "ABUPT",
    "codomainattentionneuraloperator": "CoDANO",
    "equivariantgino": "EqGINO",
    "iconoperator": "InContextOperator",
    "incontextoperatornetwork": "InContextOperator",
    "gaussianoperator": "GaussianFunctionOperator",
    "conditionalflowoperator": "ConditionalFlowFunctionOperator",
    "scot": "Poseidon",
    "denoisingpretrainedoperatortransformer": "DPOT",
    "transolverplusplus": "TransolverPlusPlus",
    "transolverpp": "TransolverPlusPlus",
    "generalneuraloperatortransformer": "GNOT",
    "koopmanneuraloperator": "KoopmanTemporalOperator",
    "greenneuraloperator": "GreenKernelOperator",
}
_ALIAS_TARGETS.update(
    {
        _normalize_architecture_name(canonical_name): canonical_name
        for canonical_name in _OPERATOR_ARCHITECTURE_STATUSES
    }
)
_OPERATOR_ARCHITECTURE_ALIASES = MappingProxyType(_ALIAS_TARGETS)


def operator_architecture_status(name: str, /) -> OperatorArchitectureStatus:
    """Return maturity metadata for a normalized architecture name or alias."""

    normalized = _normalize_architecture_name(name)
    canonical_name = _OPERATOR_ARCHITECTURE_ALIASES.get(normalized)
    if canonical_name is None:
        raise ValueError(f"Unknown operator architecture {name!r}.")
    return OPERATOR_ARCHITECTURE_STATUSES[canonical_name]


def _configured_values(
    status: OperatorArchitectureStatus,
    configuration: Mapping[str, Any] | Sequence[tuple[str, Any]] | None,
    /,
) -> OperatorArchitectureConfiguration:
    additions = (
        ()
        if configuration is None
        else tuple(
            sorted(configuration.items())
            if isinstance(configuration, Mapping)
            else configuration
        )
    )
    names = tuple(str(name) for name, _ in additions)
    if len(set(names)) != len(names):
        raise ValueError("Configured operator parameter names must be unique.")
    merged = list(status.configuration)
    known = dict(merged)
    for raw_name, value in additions:
        name = str(raw_name)
        if name in known and known[name] != value:
            raise ValueError(
                f"Configuration {name!r} conflicts with registered value {known[name]!r}."
            )
        if name not in known:
            merged.append((name, value))
            known[name] = value
    return tuple(merged)


def operator_architecture_contract(
    name: str,
    /,
    *,
    configuration: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
) -> ConfiguredOperatorContract:
    """Resolve one configured architecture to its runtime and training contract."""

    status = operator_architecture_status(name)
    configured = _configured_values(status, configuration)
    capabilities = status.capabilities
    return ConfiguredOperatorContract(
        architecture=status.architecture,
        configuration=configured,
        capabilities=capabilities,
        training=status.training,
    )


def _reconcile_instance_contract(
    model: Any,
    contract: ConfiguredOperatorContract,
    /,
) -> ConfiguredOperatorContract:
    """Reconcile declared capabilities with the concrete runtime protocols."""
    capability = contract.capabilities
    requires_fixed_query = capability.requires_fixed_query
    if type(model).__name__ == "DeepONet":
        requires_fixed_query = type(model.trunk).__name__ == "PODBasis"
    reconciled = replace(
        capability,
        requires_fixed_query=requires_fixed_query,
        encode_once_decode_many=isinstance(model, AbstractEncodedOperatorModel),
    )
    if reconciled == capability:
        return contract
    return replace(contract, capabilities=reconciled)


def operator_instance_contract(model: Any, /) -> ConfiguredOperatorContract:
    """Derive one runtime contract exclusively from a concrete model instance."""
    model_type = type(model).__name__
    if model_type == "PDEConditionedOperator":
        wrapped = model.operator.operator_contract
        capability = wrapped.capabilities
        input_name = model.input_name
        if input_name in capability.global_condition_sources:
            raise ValueError(
                f"PDE condition source {input_name!r} already exists in the wrapped contract."
            )
        maximum_sources = capability.maximum_sources
        contract = ConfiguredOperatorContract(
            architecture="PDEConditionedOperator",
            configuration=wrapped.configuration
            + (
                ("wrapped_architecture", wrapped.architecture),
                ("condition_input", input_name),
            ),
            capabilities=replace(
                capability,
                global_condition_sources=capability.global_condition_sources
                + (input_name,),
                minimum_sources=capability.minimum_sources + 1,
                maximum_sources=(
                    None if maximum_sources is None else maximum_sources + 1
                ),
            ),
            training=wrapped.training,
        )
        return _reconcile_instance_contract(model, contract)
    if model_type == "GaussianFunctionOperator":
        wrapped = model.base.operator_contract
        contract = ConfiguredOperatorContract(
            architecture="GaussianFunctionOperator",
            configuration=wrapped.configuration
            + (
                ("wrapped_architecture", wrapped.architecture),
                ("out_channels", model.out_size),
                ("factor_rank", model.factor_rank),
                ("min_scale", model.min_scale),
                ("factor_scale", model.factor_scale),
                ("scale_mode", model.scale_mode),
                ("fixed_scale", model.fixed_scale),
                ("uncertainty_source", model.uncertainty_source),
            ),
            capabilities=wrapped.capabilities,
            training=wrapped.training,
        )
        return _reconcile_instance_contract(model, contract)
    if model_type == "ConditionalFlowFunctionOperator":
        wrapped = model.location_model.operator_contract
        supported_queries = tuple(
            geometry
            for geometry in wrapped.capabilities.query_geometries
            if geometry in ("tensor_grid", "point_cloud")
        )
        contract = ConfiguredOperatorContract(
            architecture="ConditionalFlowFunctionOperator",
            configuration=wrapped.configuration
            + (
                ("wrapped_architecture", wrapped.architecture),
                ("flow_type", type(model.flow).__name__),
                ("event_size", int(model.active_indices.shape[0])),
                ("condition_inputs", tuple(model.conditioner.encoders)),
                ("condition_size", model.conditioner.condition_size),
                ("query_fingerprint", model.reference_query_fingerprint),
                ("uncertainty_source", model.uncertainty_source),
            ),
            capabilities=replace(
                wrapped.capabilities,
                query_geometries=supported_queries,
                requires_fixed_query=True,
                multiple_queries=False,
                resolution_transfer=False,
                topology="unused",
            ),
            training=wrapped.training,
        )
        return _reconcile_instance_contract(model, contract)
    if model_type == "ExternalOperatorAdapter":
        manifest = model.manifest
        contract = operator_architecture_contract(
            manifest.architecture,
            configuration=(
                ("model_version", manifest.model_version),
                ("revision", manifest.revision),
            ),
        )
        return _reconcile_instance_contract(model, contract)

    if model_type == "CochainNeuralOperator":
        contract = operator_architecture_contract(
            "CochainNeuralOperator",
            configuration=(
                ("active_degrees", model.active_degrees),
                ("width", model.width),
                ("depth", model.depth),
                ("boundary_policy", model.boundary_policy),
                ("routes", model.routes.enabled_routes),
            ),
        )
        return _reconcile_instance_contract(
            model,
            replace(contract, field_specs=model.fields),
        )

    architecture = {
        "EquivariantGeometryOperator": "EqGINO",
        "NativeGraphOperator": "GraphNeuralOperator",
    }.get(model_type, model_type)
    configuration: tuple[tuple[str, object], ...] = ()
    if model_type == "DiagonalStateSpaceMixer":
        configuration = (
            ("state_size", model.state_size),
            ("input_integration", model.input_integration),
            ("execution", model.execution),
            ("discretization", model.discretization),
            ("approximation", model.approximation),
            ("source_key", model.source_key),
            ("method_id", model.method_id),
        )
    if model_type in ("FNO", "AxialFactorizedFNO", "IFNO"):
        configuration = (
            ("n_modes", model.n_modes),
            ("width", model.width),
            ("factorization", model.factorization),
            ("rank", model.factorization_rank),
            ("axial", model.axial_factorization),
            ("coordinate_embedding", model.coordinate_embedding),
            ("domain_padding", model.domain_padding),
            ("source_key", model.source_key),
            ("scan", model.scan),
        )
    if model_type == "HOFNO":
        configuration = (
            ("n_modes", model.n_modes),
            ("width", model.width),
            ("depth", len(model.blocks)),
            ("interaction_order", model.interaction_order),
            ("factor_bias", model.factor_bias),
            ("spectral_channel_mixing", model.spectral_channel_mixing),
            ("aliasing", model.aliasing),
            ("ffn_expansion", model.ffn_expansion),
            ("norm_epsilon", model.norm_epsilon),
            ("coordinate_embedding", model.coordinate_embedding),
            ("domain_padding", model.domain_padding),
            ("source_key", model.source_key),
            ("scan", model.scan),
        )
    contract = operator_architecture_contract(architecture, configuration=configuration)
    return _reconcile_instance_contract(model, contract)


def validate_operator_architecture(
    name: str,
    batch: OperatorBatch,
    /,
    *,
    configuration: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
    problem: OperatorProblemSpec | None = None,
    training_evidence: OperatorTrainingEvidence | None = None,
    fields: Sequence[Any] = (),
) -> OperatorCompatibilityReport:
    """Validate one configured architecture against a concrete operator problem."""

    contract = operator_architecture_contract(name, configuration=configuration)
    return contract.validate(
        batch,
        problem=problem,
        training_evidence=training_evidence,
        fields=fields,
    )


__all__ = [
    "operator_architecture_contract",
    "operator_instance_contract",
    "OPERATOR_ARCHITECTURE_STATUSES",
    "OperatorArchitectureStatus",
    "OperatorArchitectureTier",
    "operator_architecture_status",
    "validate_operator_architecture",
]
