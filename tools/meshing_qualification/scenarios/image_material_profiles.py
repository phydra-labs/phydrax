from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product

import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._cell_complex import TetrahedralConnectivity
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.geometry._affine_plc_boundary import AffinePlcBoundarySource
from phydrax.geometry._affine_plc_domain import AffinePlcDomain
from phydrax.geometry._planar_primitives import area2, intersection
from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
)
from phydrax.meshing._controls import RegionSeed
from phydrax.meshing._organization import MeshZone, RegionRole
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing._volume_generation import NativeVolumeSchedule
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


def _world_seed(
    index_to_world: np.ndarray, coordinates: tuple[float, float, float], /
) -> np.ndarray:
    homogeneous = np.asarray((*coordinates, 1.0), dtype=np.float64)
    return (index_to_world @ homogeneous)[:3]


def _categorical_source(
    name: str,
    values: np.ndarray,
    label_ids: tuple[str, ...],
    matrix: np.ndarray,
    contract: SpatialCoordinateContract,
    /,
) -> phx.geometry.CompartmentMeshingSource:
    """Bind the authored voxel labels, exact affine and complete box exterior."""
    affine = phx.imaging.ImageIndexAffine(
        matrix, "voxel-index", contract, phx.imaging.ImageAxisConvention.XYZ
    )
    payload = values.tobytes()
    source_manifest = phx.qualification.ReferenceArtifactManifest(
        "qualification-image",
        checksum_algorithm="sha256",
        checksum=sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"length": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("synthetic",),
    )
    asset = phx.imaging.MedicalImageAsset(
        name,
        "segmentation",
        values,
        affine,
        phx.imaging.ImageFieldSpec.named(
            "segmentation", phx.units.ONE, phx.measurement.ValueKind.CATEGORICAL
        ),
        phx.imaging.DeidentificationEvidence(
            "synthetic-deidentification", "synthetic", "qualification", True, True, True
        ),
        (source_manifest,),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="native-image-material",
        ),
    )
    labels = phx.imaging.LabelVolume(
        asset,
        phx.imaging.LabelOntology(
            name,
            "synthetic",
            "1",
            tuple(
                phx.imaging.LabelDefinition(i, label, label)
                for i, label in enumerate(label_ids)
            ),
        ),
    )
    definitions = tuple(
        phx.geometry.CompartmentDefinition(
            label,
            (label,),
            "solid",
            allowed_neighbor_ids=tuple(other for other in label_ids if other != label),
        )
        for label in label_ids
    )
    pairs = {
        tuple(sorted((int(values[index]), int(values[neighbor]))))
        for index in np.ndindex(values.shape)
        for axis in range(3)
        if index[axis] + 1 < values.shape[axis]
        for neighbor in (tuple(index[i] + (i == axis) for i in range(3)),)
        if values[index] != values[neighbor]
    }
    interfaces = tuple(
        phx.geometry.CompartmentInterfaceDefinition(
            f"{label_ids[a]}:{label_ids[b]}", label_ids[a], label_ids[b], "material"
        )
        for a, b in sorted(pairs)
    )
    complex_ = phx.imaging.build_compartment_complex(labels, definitions, interfaces)
    surfaces = phx.imaging.extract_compartment_surfaces(labels, complex_)
    upper = np.asarray(values.shape, dtype=np.float64) - 0.5
    index_vertices = np.asarray(
        (
            (-0.5, -0.5, -0.5),
            (upper[0], -0.5, -0.5),
            (upper[0], upper[1], -0.5),
            (-0.5, upper[1], -0.5),
            (-0.5, -0.5, upper[2]),
            (upper[0], -0.5, upper[2]),
            (upper[0], upper[1], upper[2]),
            (-0.5, upper[1], upper[2]),
        ),
        dtype=np.float64,
    )
    quads = (
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
        (0, 3, 2, 1),
        (4, 5, 6, 7),
    )
    faces = np.asarray(
        [triangle for a, b, c, d in quads for triangle in ((a, b, c), (a, c, d))],
        dtype=np.int64,
    )
    if np.linalg.det(matrix[:3, :3]) < 0:
        faces = faces[:, (0, 2, 1)]
    outer = phx.geometry.SurfaceModel.from_triangles(
        affine.index_to_world(index_vertices),
        faces,
        phx.geometry.SurfaceMetadata(
            source_id=f"{name}:outer",
            source_revision=labels.label_volume_id,
            coordinate_contract=contract,
            provenance=("categorical-voxel-exterior",),
        ),
    )
    return phx.geometry.CompartmentMeshingSource(labels, complex_, outer, surfaces)


def _voxel_interface_oracle(
    labels: np.ndarray,
    label_ids: tuple[str, ...],
    affine: np.ndarray,
    triangles: np.ndarray,
    sides: list[tuple[tuple[str, np.ndarray], ...]],
    /,
    *,
    tolerance: float = 1.0e-10,
) -> dict[str, object]:
    """Compare all faces to oriented voxel faces using exact planar clipping."""
    linear = affine[:3, :3]
    inverse = np.linalg.inv(linear)
    index_triangles = (triangles - affine[:3, 3]) @ inverse.T
    squares: dict[tuple[int, int, str, str], list[tuple[int, int]]] = {}
    ambient = "ambient"
    for axis in range(3):
        other = tuple(i for i in range(3) if i != axis)
        for plane in range(labels.shape[axis] + 1):
            for u, v in product(
                range(labels.shape[other[0]]), range(labels.shape[other[1]])
            ):
                point = [0, 0, 0]
                point[other[0]], point[other[1]] = u, v
                point[axis] = plane - 1
                lower = ambient if plane == 0 else label_ids[int(labels[tuple(point)])]
                point[axis] = plane
                upper = (
                    ambient
                    if plane == labels.shape[axis]
                    else label_ids[int(labels[tuple(point)])]
                )
                if lower != upper:
                    squares.setdefault((axis, plane, lower, upper), []).append((u, v))

    fragments: dict[
        tuple[int, int, str, str, int, int], list[tuple[tuple[Fraction, Fraction], ...]]
    ] = {}
    max_distance = 0.0
    for triangle, adjacent in zip(index_triangles, sides, strict=True):
        candidates = []
        for axis in range(3):
            plane = int(np.rint(np.mean(triangle[:, axis]) + 0.5))
            distance = float(np.max(np.abs(triangle[:, axis] - (plane - 0.5)))) / float(
                np.linalg.norm(inverse[axis])
            )
            if distance <= tolerance:
                candidates.append((axis, plane, distance))
        if len(candidates) != 1:
            raise RuntimeError(
                "Image interface triangle is not on exactly one authored voxel plane."
            )
        axis, plane, distance = candidates[0]
        max_distance = max(max_distance, distance)
        ordered = [ambient, ambient]
        for label, interior in adjacent:
            coordinate = float((inverse @ (interior - affine[:3, 3]))[axis])
            side = int(coordinate > plane - 0.5)
            if ordered[side] != ambient or label == ambient or coordinate == plane - 0.5:
                raise RuntimeError(
                    "Image interface has inconsistent oriented material sides."
                )
            ordered[side] = label
        key = (axis, plane, ordered[0], ordered[1])
        other = tuple(i for i in range(3) if i != axis)
        polygon = tuple(
            (Fraction(float(point[other[0]])), Fraction(float(point[other[1]])))
            for point in triangle
        )
        triangle_area = area2(polygon) / 2
        if not triangle_area:
            raise RuntimeError("Image interface contains a degenerate triangle.")
        covered = Fraction(0)
        for u, v in squares.get(key, ()):
            box = tuple(
                (Fraction(x, 2), Fraction(y, 2))
                for x, y in (
                    (2 * u - 1, 2 * v - 1),
                    (2 * u + 1, 2 * v - 1),
                    (2 * u + 1, 2 * v + 1),
                    (2 * u - 1, 2 * v + 1),
                )
            )
            fragment = intersection(polygon, box)
            area = area2(fragment) / 2
            if area:
                covered += area
                fragments.setdefault((*key, u, v), []).append(fragment)
        scale = float(np.linalg.norm(np.cross(linear[:, other[0]], linear[:, other[1]])))
        area_tolerance = (
            tolerance * 2.0 * sum(float(np.linalg.norm(linear[:, i])) for i in other)
        )
        if abs(float(covered - triangle_area)) * scale > area_tolerance:
            raise RuntimeError(
                "Image interface leaves its authored material-side voxel faces."
            )
    for key, positions in squares.items():
        axis = key[0]
        other = tuple(i for i in range(3) if i != axis)
        scale = float(np.linalg.norm(np.cross(linear[:, other[0]], linear[:, other[1]])))
        area_tolerance = (
            tolerance * 2.0 * sum(float(np.linalg.norm(linear[:, i])) for i in other)
        )
        for u, v in positions:
            pieces = fragments.get((*key, u, v), [])
            total = sum((area2(piece) / 2 for piece in pieces), Fraction(0))
            if abs(float(total - 1)) * scale > area_tolerance:
                raise RuntimeError(
                    "Image interface does not cover an authored voxel face."
                )
            for first, second in combinations(pieces, 2):
                if float(area2(intersection(first, second)) / 2) * scale > area_tolerance:
                    raise RuntimeError("Image interface has overlapping face coverage.")
    return {
        "method": "oriented voxel faces and exact rational planar clipping",
        "physical_location_tolerance_m": tolerance,
        "maximum_plane_distance_m": max_distance,
        "voxel_face_count": sum(map(len, squares.values())),
        "interface_triangle_count": len(triangles),
        "adjacency_pairs": [
            list(pair)
            for pair in sorted({tuple(sorted((key[2], key[3]))) for key in squares})
        ],
    }


def _mesh_interface_oracle(
    result: phx.meshing.CellMeshingResult,
    cell_regions: np.ndarray,
    region_ids: tuple[str, ...],
    labels: np.ndarray,
    label_ids: tuple[str, ...],
    affine: np.ndarray,
    /,
) -> dict[str, object]:
    mesh = result.mesh
    connectivity = mesh.connectivity
    if isinstance(connectivity, TetrahedralConnectivity):
        faces, cell_faces = connectivity.faces, connectivity.cell_faces
    elif isinstance(connectivity, SimplicialConnectivity) and connectivity.dimension == 3:
        faces, cell_faces = connectivity.entities[2], connectivity.cell_entities[2]
    else:
        raise TypeError("Image interface oracle requires tetrahedral connectivity.")
    faces = np.asarray(faces, dtype=np.int64)
    cell_faces = np.asarray(cell_faces, dtype=np.int64).reshape((-1, 4))
    cells = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if len(cells) != len(cell_regions) or len(cell_faces) != len(cells):
        raise RuntimeError("Image cell and face incidence does not match region labels.")
    centers = points[cells].mean(axis=1)
    incidence: list[list[int]] = [[] for _ in range(len(faces))]
    for cell, rows in enumerate(cell_faces):
        for face in rows:
            incidence[int(face)].append(cell)
    face_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    patches: dict[int, tuple[str, str]] = {}
    for patch in result.patches:
        pair = patch.source_adjacent_region_ids
        if pair is None or pair == (None, None):
            raise RuntimeError("Image patch omitted material adjacency.")
        first, second = (value if value is not None else "ambient" for value in pair)
        for identifier in np.asarray(patch.scope.entity_ids, dtype=np.int64):
            if int(identifier) in patches:
                raise RuntimeError("Image face belongs to multiple material patches.")
            patches[int(identifier)] = (min(first, second), max(first, second))
    selected: list[int] = []
    sides: list[tuple[tuple[str, np.ndarray], ...]] = []
    for row, adjacent in enumerate(incidence):
        if len(adjacent) not in (1, 2):
            raise RuntimeError("Image mesh has nonmanifold face incidence.")
        materials = tuple(region_ids[int(cell_regions[cell])] for cell in adjacent)
        if len(adjacent) == 2 and materials[0] == materials[1]:
            continue
        mesh_pair = tuple(
            sorted((*materials, "ambient") if len(adjacent) == 1 else materials)
        )
        if patches.pop(int(face_ids[row]), None) != mesh_pair:
            raise RuntimeError(
                "Image patch adjacency disagrees with mesh material sides."
            )
        selected.append(row)
        sides.append(
            tuple(
                (region_ids[int(cell_regions[cell])], centers[cell]) for cell in adjacent
            )
        )
    if patches:
        raise RuntimeError("Image patches include non-interface or unknown faces.")
    return _voxel_interface_oracle(
        labels, label_ids, affine, points[faces[selected]], sides
    )


def _fv_transport(
    result: phx.meshing.CellMeshingResult,
    cell_regions: np.ndarray,
    region_ids: tuple[str, ...],
    expected_region_measures: dict[str, float],
    /,
) -> dict[str, object]:
    from phydrax.discretization.finite_volume import (
        PiecewiseConstantReconstruction,
        PreparedUnstructuredFiniteVolumeDynamics,
        RusanovFluxPlan,
        UnstructuredFiniteVolumeBoundarySet,
        UnstructuredFiniteVolumeMethodPlan,
        UnstructuredFiniteVolumePlan,
    )
    from phydrax.equations import (
        compile_conservation_problem,
        ConservationProblemIR,
        EulerSystem,
        IdealGasMaterial,
    )
    from phydrax.solver import FiniteVolumeStepPolicy, PreparedFiniteVolumeRuntime

    system = EulerSystem(3, material=IdealGasMaterial(1.4, 1.0))
    geometry = UnstructuredFiniteVolumePlan(
        np.asarray(result.mesh.coordinates),
        tetrahedra=np.concatenate(
            [np.asarray(block.vertices) for block in result.mesh.blocks], axis=0
        ),
        vertex_global_ids=np.asarray(result.mesh.vertex_global_ids),
        cell_global_ids=np.concatenate(
            [np.asarray(block.global_ids) for block in result.mesh.blocks]
        ),
        field_name="conservative",
        component_names=system.component_names,
    ).prepare()
    patch_names = tuple(geometry.boundary_patch_names)
    boundaries = UnstructuredFiniteVolumeBoundarySet(
        patch_names,
        {name: phx.discretization.ExtrapolationBoundary() for name in patch_names},
    )
    method = UnstructuredFiniteVolumeMethodPlan(
        PiecewiseConstantReconstruction(), RusanovFluxPlan()
    )
    dynamics = compile_conservation_problem(
        ConservationProblemIR(
            "native-image-material-transport", "conservative", system, boundaries
        ),
        geometry,
        method,
    ).dynamics
    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
        raise TypeError(
            "Image material transport did not prepare unstructured FV dynamics."
        )
    runtime = PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(
            3, fallback_flux=phx.discretization.EinfeldtHLLFluxPlan()
        ),
        FiniteVolumeStepPolicy(cfl=0.15),
    )
    identifiers = np.asarray(result.mesh.entity_set(3).entity_ids, dtype=np.int64)
    regions = np.asarray(cell_regions, dtype=np.int64)
    if regions.shape != identifiers.shape:
        raise RuntimeError(
            "Image transport region labels do not align with native cell IDs."
        )
    density_by_region = np.arange(1, len(region_ids) + 1, dtype=np.float64)
    density = density_by_region[regions]
    average = np.column_stack(
        (
            density,
            np.zeros_like(density),
            np.zeros_like(density),
            np.zeros_like(density),
            np.full_like(density, 2.5),
        )
    )
    transport_dt = 0.01
    transport_steps = 5
    initial_state = runtime.initialize_state(average, 0.0, transport_dt)
    initial_content = np.asarray(
        initial_state.content_state.conservative_content, dtype=np.float64
    ).copy()
    state = initial_state
    start_mass = float(np.sum(initial_content[:, 0], dtype=np.float64))
    source_sum = np.zeros((len(system.component_names),), dtype=np.float64)
    boundary_sum = np.zeros_like(source_sum)
    net_sum = np.zeros_like(source_sum)
    positivity_valid = True
    for _ in range(transport_steps):
        step = runtime.advance(state)
        if not bool(np.asarray(step.accepted)):
            raise RuntimeError(
                "Label-field material transport did not accept every native Euler step."
            )
        state = step.runtime_state
        source, boundary, net = step.accepted_flux_integrals.conservation_sums()
        source_sum += np.asarray(source, dtype=np.float64)
        boundary_sum += np.asarray(boundary, dtype=np.float64)
        net_sum += np.asarray(net, dtype=np.float64)
        positivity_valid &= bool(np.asarray(step.positivity.fallback_valid)) and bool(
            np.asarray(step.positivity.limited_state_valid)
        )
    end_content = np.asarray(state.content_state.conservative_content, dtype=np.float64)
    end_mass = float(np.sum(end_content[:, 0], dtype=np.float64))
    expected_mass = sum(
        density_by_region[index] * expected_region_measures[label]
        for index, label in enumerate(region_ids)
    )
    flux_balance_error = abs((end_mass - start_mass) - float(net_sum[0]))
    mass_error = abs(end_mass - expected_mass)
    if (
        mass_error > 1.0e-9
        or abs(start_mass - expected_mass) > 1.0e-10
        or flux_balance_error > 1.0e-9
        or not positivity_valid
        or int(np.asarray(state.accepted_step)) != transport_steps
        or float(np.asarray(state.time)) <= 0.0
        or not np.all(np.isfinite(net_sum))
    ):
        raise RuntimeError(
            f"Labeled material transport failed its independent inventory/flux/positivity oracle: {mass_error=}, {flux_balance_error=}, {expected_mass=}, {start_mass=}, {end_mass=}."
        )
    density_changed = bool(np.any(end_content[:, 0] != initial_content[:, 0]))
    if not density_changed:
        raise RuntimeError(
            "The multi-material Euler solve did not evolve its authored contact field."
        )
    return {
        "system": system.system_id,
        "accepted_step": int(np.asarray(step.runtime_state.accepted_step)),
        "start_time": float(np.asarray(initial_state.time)),
        "end_time": float(np.asarray(state.time)),
        "transport_steps": transport_steps,
        "expected_mass": expected_mass,
        "initial_mass": start_mass,
        "continued_mass": end_mass,
        "absolute_mass_error": mass_error,
        "flux_balance_error": flux_balance_error,
        "source_integral": source_sum.tolist(),
        "boundary_integral": boundary_sum.tolist(),
        "net_integral": net_sum.tolist(),
        "positive_density_changed": density_changed,
        "positivity_valid": positivity_valid,
        "mass_tolerance": 1.0e-9,
    }


def _image_result_evidence(result: phx.meshing.CellMeshingResult, /) -> dict[str, object]:
    """Extract identities from computed acceptance for independent cold replay."""
    accepted_result_record(result)
    report = result.certification
    if report is None or report.coverage is None or report.fidelity is None:
        raise RuntimeError("Image result omitted its coverage or fidelity certificate.")
    inputs = report.inputs
    domain = inputs.domain
    query = inputs.source
    point_witness = inputs.source_coverage_witness
    chart_witness = inputs.fidelity_witness
    if domain is None or query is None or point_witness is None or chart_witness is None:
        raise RuntimeError("Image result omitted retained source and coverage evidence.")
    if not isinstance(domain, AffinePlcDomain):
        raise RuntimeError("Image result omitted its affine PLC domain.")
    if not isinstance(query, AffinePlcBoundarySource):
        raise RuntimeError("Image result omitted its affine PLC boundary query.")
    return {
        "result_id": result.result_id,
        "mesh_id": inputs.mesh_id,
        "topology_id": inputs.topology_id,
        "geometry_id": inputs.geometry_id,
        "domain_id": inputs.domain_id,
        "source_id": inputs.source_id,
        "source_revision": inputs.source_revision,
        "query_id": query.source_query_id,
        "point_witness_id": point_witness.witness_id,
        "chart_witness_id": chart_witness.witness_id,
        "coverage_id": report.coverage.certificate_id,
        "fidelity_id": report.fidelity.certificate_id,
        "report_id": report.report_id,
        "association_ids": [value.association_id for value in result.associations],
        "affine_matrix": np.asarray(domain.matrix).tolist(),
        "affine_offset": np.asarray(domain.offset).tolist(),
        "cell_regions": np.asarray(inputs.cell_regions).tolist(),
    }


def _image_profile_inputs(
    name: str, /, *, oblique: bool
) -> tuple[
    phx.geometry.CompartmentMeshingSource,
    phx.meshing.VolumeMeshingSpec,
    NativeMeshingOptions,
    SpatialCoordinateContract,
]:
    contract = SpatialCoordinateContract.si()
    resolution = 4
    spacing = 1.0 / resolution
    grid = np.indices((resolution,) * 3)
    if oblique:
        source_labels = np.where(
            grid[0] < resolution // 2,
            np.where(grid[1] < resolution // 2, 0, 1),
            2,
        ).astype(np.int32)
        label_ids = ("material-a", "material-b", "material-c")
        linear = np.asarray(
            ((-1.0, 0.2, 0.05), (0.0, 1.0, 0.1), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
    else:
        source_labels = (grid[0] >= resolution // 2).astype(np.int32)
        label_ids = ("material-a", "material-b")
        linear = np.eye(3, dtype=np.float64)
    index_to_world = np.eye(4, dtype=np.float64)
    index_to_world[:3, :3] = spacing * linear
    index_to_world[:3, 3] = (
        0.5 * spacing + np.asarray((0.0, -0.2, 0.1)) if oblique else 0.5 * spacing
    )
    source = _categorical_source(name, source_labels, label_ids, index_to_world, contract)
    scope = MeshingScope.whole(source, 2)
    seeds_index = (
        ((0.5, 0.5, 1.5), (0.5, 2.5, 1.5), (2.5, 1.5, 1.5))
        if oblique
        else ((0.5, 1.5, 1.5), (2.5, 1.5, 1.5))
    )
    specification = phx.meshing.VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(UniformSizeControl(scope, 0.4),),
        region_seeds=tuple(
            RegionSeed(
                _world_seed(index_to_world, point),
                label,
                label,
                RegionRole.SOLID,
            )
            for label, point in zip(label_ids, seeds_index, strict=True)
        ),
    )
    options = NativeMeshingOptions(
        "image_material_tetrahedral",
        volume_schedule=NativeVolumeSchedule(
            radius_edge_bound=8.0,
            minimum_dihedral_degrees=0.0,
        ),
    )
    return source, specification, options, contract


def _run_image_profile(name: str, /, *, oblique: bool) -> dict[str, object]:
    source, specification, options, contract = _image_profile_inputs(
        name, oblique=oblique
    )
    source_labels = np.asarray(source.labels.asset.values)
    label_ids = tuple(item.compartment_id for item in source.compartments.compartments)
    region_ids = label_ids
    index_to_world = source.labels.asset.spatial_affine.matrix
    scope = specification.boundary_scope
    region_counts = {
        label: int(np.count_nonzero(source_labels == index))
        for index, label in enumerate(label_ids)
    }
    cell_volume = float(abs(np.linalg.det(index_to_world[:3, :3])))
    expected_regions = {label: region_counts[label] * cell_volume for label in label_ids}
    result = (
        NativeMeshingProvider(options)
        .plan(source, specification, coordinate_contract=contract)
        .execute()
    )
    accepted = accepted_result_record(result)
    certification = result.certification
    if (
        certification is None
        or certification.inputs.source_id != scope.source_id
        or certification.inputs.source_revision != scope.source_revision
    ):
        raise RuntimeError(
            f"{name} material acceptance is not bound to the categorical source revision."
        )
    witness = certification.inputs.fidelity_witness
    if (
        not isinstance(witness, SourceBoundaryChartWitness)
        or witness.source_id != scope.source_id
        or witness.source_revision != scope.source_revision
        or not witness.target_reference_rational_tokens
        or not witness.source_parameter_rational_tokens
    ):
        raise RuntimeError(f"{name} omitted its exact source-boundary chart witness.")
    material_zones: dict[str, MeshZone] = {}
    for zone in result.zones:
        if not zone.name or zone.name in material_zones:
            raise RuntimeError(
                f"{name} result has an empty or repeated material zone {zone.name!r}."
            )
        material_zones[zone.name] = zone
    if set(material_zones) != set(label_ids):
        raise RuntimeError(
            f"{name} result zones do not cover declared labels: {sorted(material_zones)}."
        )
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in result.mesh.blocks]
    )
    cell_regions = np.asarray(certification.inputs.cell_regions, dtype=np.int64)
    measures = np.asarray(result.quality.evaluation.measures, dtype=np.float64)
    if cell_regions.shape != cell_ids.shape or measures.shape != cell_ids.shape:
        raise RuntimeError(
            f"{name} cell-region and measure ledgers do not align to canonical IDs."
        )
    row_by_cell = {int(cell_id): row for row, cell_id in enumerate(cell_ids)}
    region_measures: dict[str, float] = {}
    for label, zone in material_zones.items():
        zone_ids = np.asarray(zone.scope.entity_ids, dtype=np.int64)
        rows = np.asarray(
            [row_by_cell[int(identifier)] for identifier in zone_ids], dtype=np.int64
        )
        region_index = np.unique(cell_regions[rows])
        if region_index.size != 1 or region_ids[int(region_index[0])] != label:
            raise RuntimeError(
                f"{name} material zone {label!r} disagrees with source region indices."
            )
        region_measures[label] = float(np.sum(measures[rows]))
    total_volume = float(np.sum(measures))
    analytic_volume = float(np.prod(np.asarray(source_labels.shape)) * cell_volume)
    if not np.isclose(total_volume, analytic_volume, rtol=0.0, atol=1.0e-10):
        raise RuntimeError(
            f"{name} tetrahedra integrate {total_volume}, expected {analytic_volume} m^3."
        )
    for label in label_ids:
        if not np.isclose(
            region_measures[label], expected_regions[label], rtol=0.0, atol=1.0e-10
        ):
            raise RuntimeError(
                f"{name} material {label!r} has measure {region_measures[label]}, expected {expected_regions[label]} m^3."
            )
    transport = _fv_transport(result, cell_regions, region_ids, expected_regions)
    interface = _mesh_interface_oracle(
        result, cell_regions, region_ids, source_labels, label_ids, index_to_world
    )
    return {
        "status": "passed",
        "profile": name,
        "source_id": scope.source_id,
        "source_revision": scope.source_revision,
        "coordinate_affine": index_to_world.tolist(),
        "label_ids": list(label_ids),
        "label_voxel_counts": region_counts,
        "independent_region_volumes_m3": expected_regions,
        "measured_region_volumes_m3": region_measures,
        "analytic_total_volume_m3": analytic_volume,
        "measured_total_volume_m3": total_volume,
        "interface_oracle": interface,
        "source_chart_witness": {
            "witness_id": witness.witness_id,
            "source_id": witness.source_id,
            "source_revision": witness.source_revision,
            "chart_piece_count": len(witness.source_chart_ids),
            "target_face_count": len(set(map(int, witness.entity_global_ids))),
        },
        "transport": transport,
        "accepted": accepted,
    }


def native_image_material() -> dict[str, object]:
    meshcore = require_meshcore()
    cases: list[dict[str, object]] = []
    failures: list[str] = []
    for profile, oblique in (
        ("native-image-lifecycle", False),
        ("native-image-lifecycle:oblique-junction-transport", True),
    ):
        try:
            cases.append(_run_image_profile(profile, oblique=oblique))
        except Exception as error:
            failures.append(f"{profile}: {type(error).__name__}: {error}")
    return {
        "status": "failed" if failures else "passed",
        "workflow": "two-slab and oblique/sheared/reflected multi-material image -> native tetrahedral mesh -> FV transport",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#labeled-image",
        "cases": cases,
        "failures": failures,
        "runtime": runtime_identity(),
        "meshcore_identity": meshcore,
    }
