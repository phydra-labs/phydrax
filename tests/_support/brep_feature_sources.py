"""Low-level analytic B-Rep sources with deliberately unnormalized carrier gauges."""

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.geometry import BRepModel, BRepTessellationPolicy
from phydrax.geometry.brep._constructors import Builder, finish
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._patches import LineCurve, PlanePatch


_CONTRACT = SpatialCoordinateContract.si()
_LAZY = BRepTessellationPolicy(realize=False)


def unnormalized_wedge() -> BRepModel:
    """Return a 3-4-5 wedge with integer plane and line coefficients."""

    polygon = ((0.0, 0.0), (0.0, -5.0), (4.0, -5.0), (4.0, -3.0))
    builder = Builder()
    vertices = [
        builder.vertex(np.asarray((x, y, z), dtype=np.float64))
        for z in (0.0, 4.0)
        for x, y in polygon
    ]
    bottom, top, vertical = [], [], []
    for i in range(4):
        j = (i + 1) % 4
        for shift, bank in ((0, bottom), (4, top)):
            a, b = vertices[i + shift], vertices[j + shift]
            bank.append(
                builder.edge(
                    LineCurve(
                        builder.vertices[a], builder.vertices[b] - builder.vertices[a]
                    ),
                    0.0,
                    1.0,
                    a,
                    b,
                )
            )
        vertical.append(
            builder.edge(
                LineCurve(builder.vertices[i], (0.0, 0.0, 4.0)), 0.0, 1.0, i, i + 4
            )
        )
    with coordinate_source_request((polygon,)) as budget:
        for z, edges, orientation in ((0.0, bottom, -1), (4.0, top, 1)):
            uses = []
            for i, edge in enumerate(edges):
                j = (i + 1) % 4
                uses.append(
                    builder.coedge(
                        edge,
                        1,
                        LineCurve(
                            polygon[i], np.asarray(polygon[j]) - np.asarray(polygon[i])
                        ),
                        budget=budget,
                    )
                )
            builder.face(
                PlanePatch((0.0, 0.0, z), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                np.asarray(((0.0, -5.0), (4.0, 0.0)), dtype=np.float64),
                [uses],
                "cap",
                None,
                orientation,
            )
        for i in range(4):
            j = (i + 1) % 4
            uses = [
                builder.coedge(
                    bottom[i], 1, LineCurve((0.0, 0.0), (1.0, 0.0)), budget=budget
                ),
                builder.coedge(
                    vertical[j], 1, LineCurve((1.0, 0.0), (0.0, 1.0)), budget=budget
                ),
                builder.coedge(
                    top[i], -1, LineCurve((0.0, 1.0), (1.0, 0.0)), budget=budget
                ),
                builder.coedge(
                    vertical[i], -1, LineCurve((0.0, 0.0), (0.0, 1.0)), budget=budget
                ),
            ]
            builder.face(
                PlanePatch(
                    builder.vertices[i],
                    builder.vertices[j] - builder.vertices[i],
                    (0.0, 0.0, 4.0),
                ),
                np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
                [uses],
                "side",
                None,
                1,
            )
        return finish(
            builder,
            solid=True,
            coordinate_contract=_CONTRACT,
            source_id="wedge-fixture",
            tessellation=_LAZY,
            budget=budget,
        )
