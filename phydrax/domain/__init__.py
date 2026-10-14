#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

r"""
# Domains

Domains describe the geometry and coordinate systems for PDE problems. They are
used to sample points, build product structures, and define functions with
explicit coordinate labels.

## Building blocks

- Scalar domains like `Interval1d`, `ScalarInterval`, and `TimeInterval`.
- Compiled geometry adapters via `GeometryDomain`; sources live in `phydrax.geometry`.
- Product domains via the `@` operator, e.g. $\Omega = \Omega_x \times \Omega_t$.
- Dataset domains for operator learning and ragged row-indexed trajectories.
- `DomainFunction` wrappers that carry domain metadata.

## Structured sampling

Sampling returns `PointBatch` or `GridBatch` objects that retain axis information.
This enables operators and constraints to preserve shape semantics without manual
broadcasting.

!!! example
    ```python
    import phydrax as phx

    geom = phx.domain.Interval1d(0.0, 1.0)
    time = phx.domain.TimeInterval(0.0, 1.0)
    domain = geom @ time

    component = domain.component({"t": phx.domain.FixedStart()})
    sampling = phx.domain.PointSampling(
        16, layout=phx.domain.SampleLayout((("x",),))
    )
    batch = component.sample(sampling)
    ```
"""

from typing import Any, TYPE_CHECKING

from .._deferred_exports import deferred_exports


# Interval factors remain domain objects; geometric sources live under
# ``phydrax.geometry`` and adapt through ``GeometryDomain``.


if TYPE_CHECKING:
    from . import graph
    from .graph import (
        as_cochain_field,
        BoundaryEdges,
        BoundaryNodes,
        cochain_form_type,
        CochainCellRegion,
        CochainCells,
        Edges,
        EdgeSet,
        EdgeType,
        Globals,
        GRAPH_ENTITY_OFFSET_KEY,
        GRAPH_SAMPLE_INDEX_KEY,
        GraphBatch,
        GraphDatasetDomain,
        GraphDomain,
        GraphTrajectoryDatasetDomain,
        InterfaceEdges,
        InteriorNodes,
        Nodes,
        NodeSet,
        NodeType,
    )


# Graph domains compose the graph substrate, which itself depends on domains;
# they load on first use so either package can be imported first.
_GRAPH_EXPORTS = frozenset(
    {
        "as_cochain_field",
        "BoundaryEdges",
        "BoundaryNodes",
        "cochain_form_type",
        "CochainCellRegion",
        "CochainCells",
        "Edges",
        "EdgeSet",
        "EdgeType",
        "Globals",
        "GRAPH_ENTITY_OFFSET_KEY",
        "GRAPH_SAMPLE_INDEX_KEY",
        "GraphBatch",
        "GraphDatasetDomain",
        "GraphDomain",
        "GraphTrajectoryDatasetDomain",
        "InterfaceEdges",
        "InteriorNodes",
        "Nodes",
        "NodeSet",
        "NodeType",
    }
)


_DECOMPOSITION_EXPORTS = frozenset(
    {
        "AxisPartition",
        "BoxPartition",
        "BrokenField",
        "CartesianCoverPlan",
        "CoverAdapterEvidence",
        "IntegrationOwnership",
        "IntegrationOwnershipEvidence",
        "MappedCoverEvidence",
        "MappedCoverValidationPlan",
        "LocalFieldFamily",
        "LocalFieldRef",
        "PairedSupport",
        "PairedSupportEvidence",
        "PeriodicIdentification",
        "PairingTopology",
        "PreparedFieldRouting",
        "SubdomainCover",
        "SubdomainCoverEvidence",
        "SubdomainPatch",
        "broken_field",
        "cartesian_subdomain_cover",
        "cover_integration_ownership",
        "SubdomainHierarchy",
        "SubdomainLevel",
        "decomposition",
        "normalized_patch_coordinate",
        "geometry_subdomain_patch",
        "partition_of_unity_family",
        "partition_of_unity_field",
        "physical_boundary",
        "prepare_field_routing",
        "validate_atlas_cover_adapter",
        "validate_cell_partition_cover",
        "validate_mapped_cover",
    }
)


_NORMALIZED_DENSITY_EXPORTS = frozenset(
    {
        "density_normalization_evidence",
        "DensityNormalizationEvidence",
        "normalize_density_field",
        "NormalizedDensityField",
    }
)


def _facade_getattr(name: str) -> Any:
    if name == "graph" or name in _GRAPH_EXPORTS:
        from importlib import import_module

        module = import_module(f"{__name__}.graph")
        return module if name == "graph" else getattr(module, name)
    if name in _DECOMPOSITION_EXPORTS:
        from importlib import import_module

        module = import_module(f"{__name__}.decomposition")
        return module if name == "decomposition" else module.__dict__[name]
    if name in _NORMALIZED_DENSITY_EXPORTS:
        from . import _normalized_density as module

        exports = {
            "density_normalization_evidence": module.density_normalization_evidence,
            "DensityNormalizationEvidence": module.DensityNormalizationEvidence,
            "normalize_density_field": module.normalize_density_field,
            "NormalizedDensityField": module.NormalizedDensityField,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "ModelBinding": ".._model",
    "ModelEvaluator": ".._model",
    "AbstractGeometry": "._base",
    "EnforcementGateMethod": "._base",
    "GeometryTransitionKind": "._base",
    "GeometryTransitionResult": "._base",
    "ComponentSum": "._components",
    "DomainComponent": "._components",
    "CoordinateSpec": "._coordinate",
    "DATASET_INDEX_KEY": "._dataset",
    "DatasetDomain": "._dataset",
    "CallbackDerivativeRule": "._derivative",
    "DerivativeBackend": "._derivative",
    "DerivativeBasis": "._derivative",
    "DerivativeMode": "._derivative",
    "DerivativeRule": "._derivative",
    "Domain": "._domain",
    "JointFactor": "._domain",
    "BatchEvaluator": "._evaluation",
    "FunctionBinding": "._evaluation",
    "PointwiseEvaluator": "._evaluation",
    "FactorComponent": "._factor_component",
    "BinaryFieldEvaluator": "._function",
    "DomainFunction": "._function",
    "expectation_field": "._function",
    "sigmoid_field": "._function",
    "softmax_field": "._function",
    "SwapAxesFieldEvaluator": "._function",
    "UnaryFieldEvaluator": "._function",
    "GeometryDomain": "._geometry",
    "HyperRectangle": "._hyperrectangle",
    "irregular_trajectory_default_quadrature_total_weight": "._irregular_trajectory_dataset",
    "IrregularTrajectoryDatasetDomain": "._irregular_trajectory_dataset",
    "BaseMeasure": "._measure",
    "EstimatedMass": "._measure",
    "ExactMass": "._measure",
    "Mass": "._measure",
    "require_exact_mass": "._measure",
    "UnknownMass": "._measure",
    "ConcatenatedModelEvaluator": "._model_function",
    "construct_reference_transport": "._probability",
    "ContinuousScalarQuantileLaw": "._probability",
    "open_unit_interval": "._probability",
    "ProbabilityDomain": "._probability",
    "ReferenceMeasure": "._probability",
    "ReferenceTransport": "._probability",
    "ReferenceTransportEvidence": "._probability",
    "ReferenceTransportProvider": "._probability",
    "ProductDomain": "._product_domain",
    "RAGGED_SERIES_INDEX_KEY": "._ragged_series_dataset",
    "RaggedSeriesDatasetDomain": "._ragged_series_dataset",
    "RaggedSeriesSampling": "._ragged_series_dataset",
    "RaggedSeriesWindowSampling": "._ragged_series_dataset",
    "reference_transport": "._reference",
    "DensityReference": "._referenced_density",
    "ReferencedDensityField": "._referenced_density",
    "with_riemannian_measure": "._riemannian_measure",
    "with_volume_density": "._riemannian_measure",
    "with_weighted_riemannian_measure": "._riemannian_measure",
    "AbstractScalarDomain": "._scalar",
    "ScalarInterval": "._scalar",
    "Boundary": "._selection",
    "CoordinateFace": "._selection",
    "CoordinateFaceSide": "._selection",
    "Fixed": "._selection",
    "FixedEnd": "._selection",
    "FixedStart": "._selection",
    "Interior": "._selection",
    "Selection": "._selection",
    "SelectionSpec": "._selection",
    "SeparatedLogDensityField": "._separated_density",
    "AxisSampling": "._structure",
    "GridBatch": "._structure",
    "GridSampling": "._structure",
    "NumPoints": "._structure",
    "PointBatch": "._structure",
    "Points": "._structure",
    "PointSampling": "._structure",
    "SampleLayout": "._structure",
    "SamplingPlan": "._structure",
    "TimeInterval": "._time",
    "TRAJECTORY_CASE_INDEX_KEY": "._trajectory_dataset",
    "trajectory_default_quadrature_total_weight": "._trajectory_dataset",
    "TRAJECTORY_TIME_INDEX_KEY": "._trajectory_dataset",
    "TrajectoryDatasetDomain": "._trajectory_dataset",
    "Interval1d": ".geometry1d",
}


__getattr__, __dir__ = deferred_exports(
    __name__, globals(), _DEFERRED_EXPORTS, fallback=_facade_getattr
)


if TYPE_CHECKING:
    from .._model import ModelBinding, ModelEvaluator
    from ._base import (
        AbstractGeometry,
        EnforcementGateMethod,
        GeometryTransitionKind,
        GeometryTransitionResult,
    )
    from ._components import ComponentSum, DomainComponent
    from ._coordinate import CoordinateSpec
    from ._dataset import DATASET_INDEX_KEY, DatasetDomain
    from ._derivative import (
        CallbackDerivativeRule,
        DerivativeBackend,
        DerivativeBasis,
        DerivativeMode,
        DerivativeRule,
    )
    from ._domain import Domain, JointFactor
    from ._evaluation import BatchEvaluator, FunctionBinding, PointwiseEvaluator
    from ._factor_component import FactorComponent
    from ._function import (
        BinaryFieldEvaluator,
        DomainFunction,
        expectation_field,
        sigmoid_field,
        softmax_field,
        SwapAxesFieldEvaluator,
        UnaryFieldEvaluator,
    )
    from ._geometry import GeometryDomain
    from ._hyperrectangle import HyperRectangle
    from ._irregular_trajectory_dataset import (
        irregular_trajectory_default_quadrature_total_weight,
        IrregularTrajectoryDatasetDomain,
    )
    from ._measure import (
        BaseMeasure,
        EstimatedMass,
        ExactMass,
        Mass,
        require_exact_mass,
        UnknownMass,
    )
    from ._model_function import ConcatenatedModelEvaluator
    from ._probability import (
        construct_reference_transport,
        ContinuousScalarQuantileLaw,
        open_unit_interval,
        ProbabilityDomain,
        ReferenceMeasure,
        ReferenceTransport,
        ReferenceTransportEvidence,
        ReferenceTransportProvider,
    )
    from ._product_domain import ProductDomain
    from ._ragged_series_dataset import (
        RAGGED_SERIES_INDEX_KEY,
        RaggedSeriesDatasetDomain,
        RaggedSeriesSampling,
        RaggedSeriesWindowSampling,
    )
    from ._reference import reference_transport
    from ._referenced_density import DensityReference, ReferencedDensityField
    from ._riemannian_measure import (
        with_riemannian_measure,
        with_volume_density,
        with_weighted_riemannian_measure,
    )
    from ._scalar import AbstractScalarDomain, ScalarInterval
    from ._selection import (
        Boundary,
        CoordinateFace,
        CoordinateFaceSide,
        Fixed,
        FixedEnd,
        FixedStart,
        Interior,
        Selection,
        SelectionSpec,
    )
    from ._separated_density import SeparatedLogDensityField
    from ._structure import (
        AxisSampling,
        GridBatch,
        GridSampling,
        NumPoints,
        PointBatch,
        Points,
        PointSampling,
        SampleLayout,
        SamplingPlan,
    )
    from ._time import TimeInterval
    from ._trajectory_dataset import (
        TRAJECTORY_CASE_INDEX_KEY,
        trajectory_default_quadrature_total_weight,
        TRAJECTORY_TIME_INDEX_KEY,
        TrajectoryDatasetDomain,
    )
    from .geometry1d import Interval1d


__all__ = [
    # subpackages
    "graph",
    "decomposition",
    # time domain
    "AbstractGeometry",
    "AbstractScalarDomain",
    "ScalarInterval",
    "EnforcementGateMethod",
    "GeometryTransitionKind",
    "GeometryTransitionResult",
    "TimeInterval",
    "CoordinateSpec",
    "Domain",
    "CoverAdapterEvidence",
    "JointFactor",
    "AxisPartition",
    "BoxPartition",
    "BrokenField",
    "CartesianCoverPlan",
    "IntegrationOwnership",
    "IntegrationOwnershipEvidence",
    "MappedCoverEvidence",
    "MappedCoverValidationPlan",
    "LocalFieldFamily",
    "LocalFieldRef",
    "PairedSupport",
    "PairedSupportEvidence",
    "PeriodicIdentification",
    "PairingTopology",
    "PreparedFieldRouting",
    "SubdomainCover",
    "SubdomainCoverEvidence",
    "SubdomainPatch",
    "broken_field",
    "SubdomainHierarchy",
    "SubdomainLevel",
    "cartesian_subdomain_cover",
    "cover_integration_ownership",
    "geometry_subdomain_patch",
    "normalized_patch_coordinate",
    "partition_of_unity_field",
    "partition_of_unity_family",
    "physical_boundary",
    "prepare_field_routing",
    "validate_mapped_cover",
    "validate_atlas_cover_adapter",
    "validate_cell_partition_cover",
    # product domains / structure
    "ProbabilityDomain",
    "ReferenceMeasure",
    "ReferenceTransport",
    "ReferenceTransportEvidence",
    "ReferenceTransportProvider",
    "ContinuousScalarQuantileLaw",
    "construct_reference_transport",
    "open_unit_interval",
    "ProductDomain",
    "FactorComponent",
    "BaseMeasure",
    "ExactMass",
    "EstimatedMass",
    "Mass",
    "require_exact_mass",
    "UnknownMass",
    "reference_transport",
    "as_cochain_field",
    "cochain_form_type",
    "BatchEvaluator",
    "ModelBinding",
    "ModelEvaluator",
    "CallbackDerivativeRule",
    "DerivativeBackend",
    "DerivativeBasis",
    "DerivativeMode",
    "DerivativeRule",
    "FunctionBinding",
    "PointwiseEvaluator",
    "BinaryFieldEvaluator",
    "ConcatenatedModelEvaluator",
    "DomainFunction",
    "expectation_field",
    "sigmoid_field",
    "softmax_field",
    "SwapAxesFieldEvaluator",
    "UnaryFieldEvaluator",
    "DatasetDomain",
    "DATASET_INDEX_KEY",
    "IrregularTrajectoryDatasetDomain",
    "irregular_trajectory_default_quadrature_total_weight",
    "RAGGED_SERIES_INDEX_KEY",
    "RaggedSeriesDatasetDomain",
    "RaggedSeriesSampling",
    "RaggedSeriesWindowSampling",
    "TrajectoryDatasetDomain",
    "TRAJECTORY_CASE_INDEX_KEY",
    "TRAJECTORY_TIME_INDEX_KEY",
    "trajectory_default_quadrature_total_weight",
    "GraphBatch",
    "GraphDatasetDomain",
    "GraphDomain",
    "GraphTrajectoryDatasetDomain",
    "GRAPH_ENTITY_OFFSET_KEY",
    "GRAPH_SAMPLE_INDEX_KEY",
    "HyperRectangle",
    "NumPoints",
    "Points",
    "GeometryDomain",
    "AxisSampling",
    "SampleLayout",
    "SamplingPlan",
    "PointSampling",
    "GridSampling",
    "PointBatch",
    "GridBatch",
    # components
    "Selection",
    "Interior",
    "Boundary",
    "CoordinateFace",
    "CoordinateFaceSide",
    "Fixed",
    "FixedStart",
    "FixedEnd",
    "SelectionSpec",
    "DomainComponent",
    "CochainCellRegion",
    "DensityReference",
    "ReferencedDensityField",
    "DensityNormalizationEvidence",
    "NormalizedDensityField",
    "density_normalization_evidence",
    "normalize_density_field",
    "SeparatedLogDensityField",
    "with_riemannian_measure",
    "with_volume_density",
    "with_weighted_riemannian_measure",
    "CochainCells",
    "ComponentSum",
    "BoundaryEdges",
    "BoundaryNodes",
    "EdgeSet",
    "EdgeType",
    "Nodes",
    "Edges",
    "Globals",
    "InteriorNodes",
    "InterfaceEdges",
    "NodeSet",
    "NodeType",
    # geometry1d exports
    "Interval1d",
]
