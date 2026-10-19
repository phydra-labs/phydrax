"""Angular carrier bounds retain their own arithmetic across physical unit scales."""

from pathlib import Path

import numpy as np
import pytest

from phydrax._external_resource import ResourceLimits
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._meshing_domain import DomainCurveMap
from phydrax.geometry.brep import (
    brep_cylinder,
    BRepModel,
    BRepTessellationPolicy,
    CylinderPatch,
    LineCurve,
)
from phydrax.interchange import CadImportPolicy, decode_step_bytes, write_step
from phydrax.units import METER, MILLIMETER


def _cylinder_angular_boxes(model: BRepModel) -> np.ndarray:
    geometry = model.require_geometry()
    boxes = []
    for face, patch in enumerate(model.patches):
        if not isinstance(patch, CylinderPatch):
            continue
        for loop in geometry.face_loops[face]:
            edges = [geometry.coedge_edges[coedge] for coedge in loop]
            ranges = np.asarray(geometry.edge_ranges)[edges]
            pcurves = tuple(geometry.pcurves[coedge] for coedge in loop)
            boundary = DomainCurveMap(
                (patch,) * len(loop),
                pcurves,
                ranges[:, 0],
                ranges[:, 1],
                tuple(geometry.coedge_endpoint_roots[coedge] for coedge in loop),
            )
            assert np.all(np.isfinite(boundary.endpoint_errors))
            assert np.all(boundary.endpoint_errors >= 0.0)
            for pcurve, (first, last) in zip(pcurves, ranges, strict=True):
                box = pcurve.bounding_box(float(first), float(last))
                admitted = patch.validate_parameter_box(box)
                lower, upper = patch.derivative_bounds(admitted, order=1)
                assert lower.shape == upper.shape == (3, 2)
                assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
                boxes.append(box[:, 0])
    assert boxes
    return np.stack(boxes)


@pytest.mark.parametrize(
    "scale", [2.0**-20, 1.0, 2.0**20], ids=["small", "unit", "large"]
)
def test_native_cylinder_step_import_keeps_angular_bounds_unit_invariant(
    scale: float, tmp_path: Path
) -> None:
    source = brep_cylinder(
        scale,
        2.0 * scale,
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    path = tmp_path / "cylinder.step"
    written = write_step(source, path)
    assert written.report.valid
    data = path.read_bytes()
    limits = ResourceLimits(1 << 24, 64, 200_000, 4_000_000, 0)
    models = []
    angular_boxes = []
    for unit in (METER, MILLIMETER):
        # This pins source import and the real domain's angular/derivative
        # admission, independently of full surface-realization quality.
        realization = BRepTessellationPolicy(realize=False)
        imported = decode_step_bytes(
            data,
            CadImportPolicy(
                SpatialCoordinateContract(unit), limits, tessellation=realization
            ),
        )
        assert imported.report.valid
        assert imported.model.mesh_faces.shape[0] == 0
        models.append(imported.model)
        angular_boxes.append(_cylinder_angular_boxes(imported.model))
    # The source angular arithmetic is identical: converting the other chart
    # coordinate from metres to millimetres must not alter even one bound bit.
    np.testing.assert_array_equal(angular_boxes[0], angular_boxes[1])
    reference = LineCurve((0.0, 0.0), (1.0, 0.0)).bounding_box(0.0, 2.0 * np.pi)[:, 0]
    for model, boxes in zip(models, angular_boxes, strict=True):
        spans = boxes[:, 1] - boxes[:, 0]
        full = spans > np.pi
        assert np.any(full)
        np.testing.assert_array_equal(
            boxes[full], np.broadcast_to(reference, boxes[full].shape)
        )
        assert model.topology.num_solids == 1


@pytest.mark.parametrize("height", [2.0**-20, 1.0, 2.0**20])
def test_mixed_angular_length_line_box_uses_coordinate_local_roundoff(
    height: float,
) -> None:
    angular = LineCurve((0.0, height), (1.0, 0.0)).bounding_box(0.0, 2.0 * np.pi)[:, 0]
    reference = LineCurve((0.0, 0.0), (1.0, 0.0)).bounding_box(0.0, 2.0 * np.pi)[:, 0]
    np.testing.assert_array_equal(angular, reference)
    assert angular[0] <= 0.0
    assert angular[1] >= 2.0 * np.pi
