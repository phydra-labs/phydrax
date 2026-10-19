"""Native STEP cylinders retain relative realization quality across length scales.

The reference is the cylinder x² + y² = r², 0 <= z <= 2r, with two disk
caps. Neither source patch evaluators nor mesher-produced samples define the
oracle. Relative realization preserves public connectivity and dimensionless
geometry, deviation, and normal enclosures across all six cases.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from phydrax._external_resource import ResourceLimits
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep import brep_cylinder, BRepTessellationPolicy, CylinderPatch
from phydrax.interchange import CadImportPolicy, decode_step_bytes, write_step
from phydrax.units import METER, MILLIMETER, UnitDefinition


_RELATIVE_DEFLECTION = 0.05
_NORMAL_ANGLE = 0.4
# The fixture's admission limits and policy defaults are unchanged.
_IMPORT_LIMITS = ResourceLimits(1 << 24, 64, 200_000, 4_000_000, 0)
_UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2.0


def _gamma(operations: int) -> float:
    """Return the standard binary64 forward-error accumulation bound."""
    return operations * _UNIT_ROUNDOFF / (1.0 - operations * _UNIT_ROUNDOFF)


# Normalize before geometry arithmetic: coordinates have magnitude <= 2.
# A 17-significant-digit STEP round trip, unit conversion, normalization,
# analytic coordinate evaluation, hypot, and atan2 are budgeted 128 roundings
# (including four ulps per elementary-function evaluation). Multiplying by
# the coordinate bound 2 gives an absolute dimensionless comparison bound.
# This is not a source/meshing acceptance tolerance; those comparisons below
# use the requested limits without any slack.
_POINT_ROUNDOFF = 2.0 * _gamma(128)


@pytest.fixture(
    scope="module",
    params=[2.0**-20, 1.0, 2.0**20],
    ids=["small", "unit", "large"],
)
def cylinder_step_data(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[float, bytes]:
    """Share one SI source serialization between its two target unit systems."""
    radius = float(request.param)
    source = brep_cylinder(
        radius,
        2.0 * radius,
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    path: Path = tmp_path_factory.mktemp("cylinder-scale") / "cylinder.step"
    written = write_step(source, path)
    assert written.report.valid
    assert source.mesh_faces.shape == (0, 3)
    return radius, path.read_bytes()


@pytest.fixture(scope="module")
def unit_meter_reference(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[np.ndarray, np.ndarray]:
    """Retain one public unit-meter realization for topology correspondence."""
    source = brep_cylinder(
        1.0,
        2.0,
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    path = tmp_path_factory.mktemp("cylinder-scale-reference") / "cylinder.step"
    written = write_step(source, path)
    assert written.report.valid
    imported = decode_step_bytes(
        path.read_bytes(),
        CadImportPolicy(
            SpatialCoordinateContract(METER),
            _IMPORT_LIMITS,
            tessellation=BRepTessellationPolicy(
                linear_deflection=_RELATIVE_DEFLECTION,
                angular_deflection=_NORMAL_ANGLE,
            ),
        ),
    )
    assert imported.report.valid
    return (
        np.asarray(imported.model.mesh_vertices, dtype=np.float64),
        np.asarray(imported.model.mesh_faces, dtype=np.int64),
    )


def _canonical_connectivity(faces: np.ndarray) -> np.ndarray:
    """Remove triangle emission order and cyclic starts, retaining orientation."""
    starts = np.argmin(faces, axis=1)
    columns = (starts[:, None] + np.arange(3, dtype=np.int64)) % 3
    rotated = np.take_along_axis(faces, columns, axis=1)
    rows = np.lexsort((rotated[:, 2], rotated[:, 1], rotated[:, 0]))
    return rotated[rows]


def _cylinder_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    lateral: np.ndarray,
    deviation: np.ndarray,
    normal: np.ndarray,
) -> dict[str, float]:
    """Check an independent unit-radius cylinder and return normalized metrics."""
    corners = vertices[faces]
    radii = np.hypot(corners[..., 0], corners[..., 1])
    heights = corners[..., 2]
    side = corners[lateral]
    caps = corners[~lateral]
    assert side.shape[0] > 0 and caps.shape[0] > 0
    assert np.all(np.isfinite(vertices))
    assert np.all(radii <= 1.0 + _POINT_ROUNDOFF)
    assert np.all(heights >= -_POINT_ROUNDOFF)
    assert np.all(heights <= 2.0 + _POINT_ROUNDOFF)
    assert np.max(np.abs(radii[lateral] - 1.0)) <= _POINT_ROUNDOFF
    cap_height = np.mean(caps[..., 2], axis=1)
    assert (
        np.max(np.minimum(np.abs(cap_height), np.abs(cap_height - 2.0)))
        <= _POINT_ROUNDOFF
    )
    assert np.max(np.abs(caps[..., 2] - cap_height[:, None])) <= _POINT_ROUNDOFF

    # Independent manifold counting prevents an empty, open, duplicated, or
    # disconnected partial cylinder from passing pointwise geometry checks.
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges, uses = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    assert np.all(uses == 2)
    assert np.unique(directed, axis=0).shape[0] == directed.shape[0]
    assert vertices.shape[0] - edges.shape[0] + faces.shape[0] == 2
    assert np.unique(faces).size == vertices.shape[0]

    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    assert np.all(twice_area > 0.0)
    assert np.all(
        np.sum(cross[lateral, :2] * np.mean(side[..., :2], axis=1), axis=1) > 0.0
    )
    assert np.all(cross[~lateral, 2] * (cap_height - 1.0) > 0.0)

    # atan2(det, dot), rather than acos(dot), avoids sqrt(eps) loss near
    # coincident radial normals. All three corner pairs test normal turn.
    first = side[:, [0, 1, 2], :2]
    second = side[:, [1, 2, 0], :2]
    turns = np.abs(
        np.arctan2(
            first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0],
            np.sum(first * second, axis=-1),
        )
    )
    assert np.all(turns <= normal[lateral, None] + _POINT_ROUNDOFF)
    assert np.max(turns) <= _NORMAL_ANGLE + _POINT_ROUNDOFF

    # Convex combinations on lateral facets have radial deficit; the analytic
    # cylinder gives their distance directly. Cap samples stay in planar disks.
    weights = np.asarray(
        ((1 / 3, 1 / 3, 1 / 3), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)),
        dtype=np.float64,
    )
    samples = weights[None] @ corners
    sample_radius = np.hypot(samples[..., 0], samples[..., 1])
    sampled_distance = np.zeros(sample_radius.shape, dtype=np.float64)
    sampled_distance[lateral] = np.abs(sample_radius[lateral] - 1.0)
    sampled_distance[~lateral] = np.abs(samples[~lateral, :, 2] - cap_height[:, None])
    assert np.all(sampled_distance <= deviation[:, None] + _POINT_ROUNDOFF)

    # The lateral cylinder is isometric to (theta,z). Unwrap each triangle
    # locally (its angular span is < pi), then measure its independent curved
    # reference area; the full lateral cover has area 4*pi.
    theta = np.arctan2(side[..., 1], side[..., 0])
    delta = theta[:, 1:] - theta[:, :1]
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    dz = side[:, 1:, 2] - side[:, :1, 2]
    reference_area = 0.5 * np.abs(delta[:, 0] * dz[:, 1] - delta[:, 1] * dz[:, 0])
    # gamma(128 + 32*N) bounds local evaluation plus a worst-case serial
    # reduction of N triangles; the absolute summands are accumulated explicitly.
    reduction_roundoff = _gamma(128 + 32 * faces.shape[0])
    area_roundoff = reduction_roundoff * (1.0 + float(np.sum(reference_area)))
    assert abs(float(np.sum(reference_area)) - 4.0 * np.pi) <= area_roundoff

    # F(theta,z)=(cos(theta),sin(theta),z) has ||F_theta_theta||=1.
    # Each secant's remainder is <= delta_theta²/2. The cross-product
    # perturbation inequality gives a triangulation-dependent area enclosure,
    # valid even for skinny triangles (no assumed aspect-ratio quality).
    lengths = np.hypot(delta, dz)
    remainder = 0.5 * delta**2
    distortion = 0.5 * float(
        np.sum(
            lengths[:, 0] * remainder[:, 1]
            + lengths[:, 1] * remainder[:, 0]
            + remainder[:, 0] * remainder[:, 1]
        )
    )
    side_area = 0.5 * float(np.sum(twice_area[lateral]))
    cap_area = 0.5 * float(np.sum(twice_area[~lateral]))
    area_roundoff += reduction_roundoff * (1.0 + side_area + cap_area + distortion)
    assert abs(side_area - 4.0 * np.pi) <= distortion + area_roundoff
    # Boundary chord spans <= angle enclose the concentric disk of radius
    # cos(angle/2). Both caps and the closed radial shell are inscribed.
    inner_radius = np.cos(_NORMAL_ANGLE / 2.0)
    assert 2.0 * np.pi * inner_radius**2 - area_roundoff <= cap_area
    assert cap_area <= 2.0 * np.pi + area_roundoff
    volume_terms = (
        np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
    )
    volume = float(np.sum(volume_terms))
    volume_roundoff = reduction_roundoff * (1.0 + float(np.sum(np.abs(volume_terms))))
    assert 2.0 * np.pi * inner_radius**2 - volume_roundoff <= volume
    assert volume <= 2.0 * np.pi + volume_roundoff
    return {
        "area_over_radius_squared": side_area + cap_area,
        "volume_over_radius_cubed": volume,
        "max_deviation_over_radius": float(np.max(deviation)),
        "max_normal_bound_radians": float(np.max(normal)),
        "max_sampled_deviation_over_radius": float(np.max(sampled_distance)),
        "max_radial_normal_turn_radians": float(np.max(turns)),
        "lateral_area_enclosure_radius": distortion + area_roundoff,
        "geometry_roundoff_bound": _POINT_ROUNDOFF,
        "area_roundoff_bound": area_roundoff,
        "volume_roundoff_bound": volume_roundoff,
    }


@pytest.mark.parametrize(
    ("target_unit", "conversion"),
    [(METER, 1.0), (MILLIMETER, 1000.0)],
    ids=["meter", "millimeter"],
)
def test_native_cylinder_step_realization_is_scale_equivariant(
    cylinder_step_data: tuple[float, bytes],
    unit_meter_reference: tuple[np.ndarray, np.ndarray],
    target_unit: UnitDefinition,
    conversion: float,
    request: pytest.FixtureRequest,
) -> None:
    source_radius, data = cylinder_step_data
    target_radius = source_radius * conversion
    policy = BRepTessellationPolicy(
        linear_deflection=_RELATIVE_DEFLECTION * target_radius,
        angular_deflection=_NORMAL_ANGLE,
    )
    imported = decode_step_bytes(
        data,
        CadImportPolicy(
            SpatialCoordinateContract(target_unit), _IMPORT_LIMITS, tessellation=policy
        ),
    )
    assert imported.report.valid
    model = imported.model
    assert model.report.coordinate_contract == SpatialCoordinateContract(target_unit)
    assert model.report.linear_deflection == policy.linear_deflection
    assert model.report.angular_deflection == policy.angular_deflection
    assert model.topology.num_solids == 1
    vertices = np.asarray(model.mesh_vertices, dtype=np.float64)
    faces = np.asarray(model.mesh_faces, dtype=np.int64)
    assert vertices.shape[0] > 0 and faces.shape[0] > 0
    deviation = np.asarray(model.tessellation_deviation_bounds, dtype=np.float64)
    normal = np.asarray(model.tessellation_normal_bounds, dtype=np.float64)
    assert deviation.shape == normal.shape == (faces.shape[0],)
    assert np.all(np.isfinite(deviation)) and np.all(np.isfinite(normal))
    assert np.all(deviation >= 0.0) and np.all(normal >= 0.0)
    # Consumer-visible native certificates must satisfy the requested bounds.
    assert np.max(deviation) <= policy.linear_deflection
    assert np.max(normal) <= policy.angular_deflection
    lateral_patches = np.asarray(
        [isinstance(patch, CylinderPatch) for patch in model.patches], dtype=np.bool_
    )
    lateral = lateral_patches[np.asarray(model.triangle_face_ids, dtype=np.int64)]
    metrics = _cylinder_metrics(
        vertices / target_radius, faces, lateral, deviation / target_radius, normal
    )
    metrics.update(
        {
            "source_radius_meters": source_radius,
            "target_radius": target_radius,
            "area": metrics["area_over_radius_squared"] * target_radius**2,
            "volume": metrics["volume_over_radius_cubed"] * target_radius**3,
            "max_deviation": float(np.max(deviation)),
        }
    )
    # Pytest/JUnit consumer evidence exposes every case without writing product
    # artifacts or requiring a particular incidental mesh ordering.
    request.node.user_properties.append(
        ("cylinder_scale_metrics", json.dumps(metrics, sort_keys=True))
    )
    print(json.dumps(metrics, sort_keys=True))
    reference_vertices, reference_faces = unit_meter_reference
    # Coordinate correspondence fixes the scientific meaning of public vertex
    # indices. Comparing two normalized evaluations doubles their individual
    # absolute roundoff bound; there is no relative or meshing-quality slack.
    np.testing.assert_allclose(
        vertices / target_radius,
        reference_vertices,
        rtol=0.0,
        atol=2.0 * _POINT_ROUNDOFF,
    )
    np.testing.assert_array_equal(
        _canonical_connectivity(faces), _canonical_connectivity(reference_faces)
    )
