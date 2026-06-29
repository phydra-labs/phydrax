#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""
# Domains

Domains describe the geometry and coordinate systems for PDE problems. They are
used to sample points, build product structures, and define functions with
explicit coordinate labels.

## Building blocks

- Scalar domains like `Interval1d`, `ScalarInterval`, and `TimeInterval`.
- Geometry in 2D and 3D (`Square`, `Sphere`, `Geometry2DFromCAD`, etc.).
- Product domains via the `@` operator, e.g. $\\Omega = \\Omega_x \\times \\Omega_t$.
- Dataset domains for operator learning and ragged row-indexed trajectories.
- `DomainFunction` wrappers that carry domain metadata.

## Structured sampling

Sampling returns `PointsBatch` or `CoordSeparableBatch` objects that retain
axis information. This enables operators and constraints to preserve
shape semantics without manual broadcasting.

!!! example
    ```python
    import phydrax as phx

    geom = phx.domain.Interval1d(0.0, 1.0)
    time = phx.domain.TimeInterval(0.0, 1.0)
    domain = geom @ time

    component = domain.component({"t": phx.domain.FixedStart()})
    structure = phx.domain.ProductStructure((("x",),))
    batch = component.sample(num_points=16, structure=structure)
    ```
"""

from . import (
    geometry1d,
    geometry2d,
    geometry3d,
    graph,
)
from ._components import (
    Boundary,
    ComponentSpec,
    DomainComponent,
    DomainComponentUnion,
    Fixed,
    FixedEnd,
    FixedStart,
    Interior,
    VarComponent,
)
from ._dataset import DatasetDomain
from ._function import DomainFunction
from ._grid import (
    AbstractAxisSpec,
    AxisDiscretization,
    CosineAxisSpec,
    FourierAxisSpec,
    GridSpec,
    LegendreAxisSpec,
    SineAxisSpec,
    UniformAxisSpec,
)
from ._hyperrectangle import HyperRectangle
from ._model_function import structured
from ._product_domain import ProductDomain
from ._scalar import ScalarInterval
from ._structure import (
    CoordSeparableBatch,
    PointsBatch,
    ProductStructure,
    QuadratureBatch,
)
from ._time import TimeInterval
from ._trajectory_dataset import (
    TrajectoryDatasetDomain,
)

# Re-export geometry submodule objects (unary domains)
from .geometry1d import Interval1d  # noqa: F401
from .geometry2d import (  # noqa: F401
    Circle,
    Ellipse,
    Geometry2DFromCAD,
    Geometry2DFromPointCloud,
    Polygon,
    Rectangle,
    Square,
    Triangle,
)
from .geometry3d import (  # noqa: F401
    Cone,
    Cube,
    Cuboid,
    Cylinder,
    Ellipsoid,
    Geometry3DFromCAD,
    Geometry3DFromDEM,
    Geometry3DFromLidarScene,
    Geometry3DFromPointCloud,
    Sphere,
    Torus,
    Wedge,
)
from .graph import (
    BoundaryEdges,
    BoundaryNodes,
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
    "geometry1d",
    "geometry2d",
    "geometry3d",
    # time domain
    "ScalarInterval",
    "TimeInterval",
    # product domains / structure
    "ProductDomain",
    "DomainFunction",
    "structured",
    "DatasetDomain",
    "TrajectoryDatasetDomain",
    "GraphBatch",
    "GraphDatasetDomain",
    "GraphDomain",
    "GraphTrajectoryDatasetDomain",
    "GRAPH_ENTITY_OFFSET_KEY",
    "GRAPH_SAMPLE_INDEX_KEY",
    "HyperRectangle",
    "ProductStructure",
    "PointsBatch",
    "QuadratureBatch",
    "CoordSeparableBatch",
    # grid/basis specs
    "AbstractAxisSpec",
    "AxisDiscretization",
    "GridSpec",
    "UniformAxisSpec",
    "FourierAxisSpec",
    "SineAxisSpec",
    "CosineAxisSpec",
    "LegendreAxisSpec",
    # components
    "VarComponent",
    "Interior",
    "Boundary",
    "Fixed",
    "FixedStart",
    "FixedEnd",
    "ComponentSpec",
    "DomainComponent",
    "DomainComponentUnion",
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
    # geometry2d exports
    "Geometry2DFromCAD",
    "Geometry2DFromPointCloud",
    "Circle",
    "Ellipse",
    "Polygon",
    "Rectangle",
    "Square",
    "Triangle",
    # geometry3d exports
    "Geometry3DFromCAD",
    "Geometry3DFromDEM",
    "Geometry3DFromLidarScene",
    "Geometry3DFromPointCloud",
    "Cone",
    "Cube",
    "Cuboid",
    "Cylinder",
    "Ellipsoid",
    "Sphere",
    "Torus",
    "Wedge",
]
