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

from . import graph
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
    SwapAxesFieldEvaluator,
    UnaryFieldEvaluator,
)
from ._geometry import GeometryDomain
from ._grid import (
    AbstractAxisSpec,
    AxisDiscretization,
    broadcasted_grid,
    CosineAxisSpec,
    FourierAxisSpec,
    GridSpec,
    LegendreAxisSpec,
    NestedDyadicAxisSpec,
    SineAxisSpec,
    UniformAxisSpec,
)
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
    open_unit_interval,
    ProbabilityDomain,
    ReferenceDistribution,
)
from ._product_domain import ProductDomain
from ._ragged_series_dataset import (
    RAGGED_SERIES_INDEX_KEY,
    RaggedSeriesDatasetDomain,
    RaggedSeriesSampling,
)
from ._reference import reference_transport
from ._riemannian_measure import with_riemannian_measure
from ._scalar import AbstractScalarDomain, ScalarInterval
from ._selection import (
    Boundary,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
    Selection,
    SelectionSpec,
)
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

# Interval factors remain domain objects; geometric sources live under
# ``phydrax.geometry`` and adapt through ``GeometryDomain``.
from .geometry1d import Interval1d
from .graph import (
    as_cochain_field,
    BoundaryEdges,
    BoundaryNodes,
    cochain_field_spec,
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


__all__ = [
    # subpackages
    "graph",
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
    "JointFactor",
    # product domains / structure
    "ProbabilityDomain",
    "ReferenceDistribution",
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
    "cochain_field_spec",
    "BatchEvaluator",
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
    "SwapAxesFieldEvaluator",
    "UnaryFieldEvaluator",
    "DatasetDomain",
    "DATASET_INDEX_KEY",
    "IrregularTrajectoryDatasetDomain",
    "irregular_trajectory_default_quadrature_total_weight",
    "RAGGED_SERIES_INDEX_KEY",
    "RaggedSeriesDatasetDomain",
    "RaggedSeriesSampling",
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
    # grid/basis specs
    "AbstractAxisSpec",
    "AxisDiscretization",
    "GridSpec",
    "broadcasted_grid",
    "NestedDyadicAxisSpec",
    "UniformAxisSpec",
    "FourierAxisSpec",
    "SineAxisSpec",
    "CosineAxisSpec",
    "LegendreAxisSpec",
    # components
    "Selection",
    "Interior",
    "Boundary",
    "Fixed",
    "FixedStart",
    "FixedEnd",
    "SelectionSpec",
    "DomainComponent",
    "CochainCellRegion",
    "with_riemannian_measure",
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
