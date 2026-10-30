from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.multiregion_surface import (
    LabelFieldSurfaceExtractionPlan,
    MultiRegionSurfaceValidationPolicy,
)
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
from phydrax.threshold_dynamics import LabelFieldState

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


def _world_seed(
    index_to_world: np.ndarray, coordinates: tuple[float, float, float], /
) -> np.ndarray:
    homogeneous = np.asarray((*coordinates, 1.0), dtype=np.float64)
    return (index_to_world @ homogeneous)[:3]


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
    geometry = UnstructuredFiniteVolumePlan.from_cell_mesh(
        result.mesh, component_names=system.component_names
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


def _run_image_profile(name: str, /, *, oblique: bool) -> dict[str, object]:
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
    extraction_plan = LabelFieldSurfaceExtractionPlan(
        label_ids,
        index_to_world=index_to_world,
        boundary_region_id="ambient",
        validation_policy=MultiRegionSurfaceValidationPolicy(profile="dry_foam"),
    )
    state = LabelFieldState(
        jnp.asarray(source_labels, dtype=jnp.int32),
        jnp.asarray((True,) * len(label_ids)),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float64),
        label_ids=label_ids,
        route_id=f"native-image-material:{name}",
        prepared_id=f"native-image-material:{name}:preparation",
        site_id=f"native-image-material:{name}:sites",
    )
    extraction = extraction_plan.prepare(source_labels.shape).extract(state)
    if (
        not extraction.accepted
        or extraction.topology is None
        or extraction.surface is None
    ):
        raise RuntimeError(
            f"{name} labeled source failed collision-certified extraction."
        )
    topology = extraction.topology
    if topology.region_ids != extraction.lineage.region_ids:
        raise RuntimeError(
            f"{name} extraction topology and source lineage disagree on region identities."
        )
    scope = MeshingScope.whole(extraction, 2)
    seeds_index = (
        ((0.5, 0.5, 1.5), (0.5, 2.5, 1.5), (2.5, 1.5, 1.5))
        if oblique
        else ((0.5, 1.5, 1.5), (2.5, 1.5, 1.5))
    )
    region_counts = {
        name: int(np.count_nonzero(source_labels == index))
        for index, name in enumerate(label_ids)
    }
    cell_volume = float(abs(np.linalg.det(index_to_world[:3, :3])))
    expected_regions = {name: region_counts[name] * cell_volume for name in label_ids}
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
    plan = NativeMeshingProvider(
        NativeMeshingOptions(
            "image_material_tetrahedral",
            volume_schedule=NativeVolumeSchedule(
                radius_edge_bound=8.0,
                minimum_dihedral_degrees=0.0,
            ),
        )
    ).plan(extraction, specification, coordinate_contract=contract)
    result = plan.execute()
    accepted = accepted_result_record(result)
    certification = result.certification
    if (
        certification is None
        or certification.inputs.source_id != scope.source_id
        or certification.inputs.source_revision != scope.source_revision
    ):
        raise RuntimeError(
            f"{name} material acceptance is not bound to the extracted source revision."
        )
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
        if region_index.size != 1 or topology.region_ids[int(region_index[0])] != label:
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
    expected_pairs: set[tuple[str, str]] = set()
    for pair in topology.host_face_labels():
        first, second = (topology.region_ids[int(region)] for region in pair)
        expected_pairs.add((first, second) if first <= second else (second, first))
    actual_pairs: set[tuple[str, str]] = set()
    for patch in result.patches:
        if patch.source_adjacent_region_ids is None:
            raise RuntimeError(f"{name} source patch omitted exact adjacent region IDs.")
        first, second = patch.source_adjacent_region_ids
        if first is None or second is None:
            raise RuntimeError(f"{name} source patch omitted exact adjacent region IDs.")
        actual_pairs.add((first, second) if first <= second else (second, first))
    if actual_pairs != expected_pairs:
        raise RuntimeError(
            f"{name} patches lost source adjacency: {actual_pairs} != {expected_pairs}."
        )
    transport = _fv_transport(result, cell_regions, topology.region_ids, expected_regions)
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
        "source_adjacency_pairs": [list(pair) for pair in sorted(actual_pairs)],
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
