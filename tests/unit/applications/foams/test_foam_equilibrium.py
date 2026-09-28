from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.foams import (
    catenoid_area_ratio,
    catenoid_critical_parameters,
    catenoid_stable_neck_ratio,
    FoamDynamicsPlan,
    FoamDynamicsState,
    FoamEquilibriumPlan,
    FoamEquilibriumResult,
    FoamEquilibriumStatus,
    FoamKKTStatus,
    FoamMaterialPlan,
    FoamWireConstraints,
    PreparedFoamDynamics,
    PreparedFoamEquilibrium,
    RegionPressureAirPlan,
    StandardDoubleBubble,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceSeed,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    MultiRegionSurfaceValidationPolicy,
    PreparedMultiRegionSurface,
    seed_catenoid,
    seed_double_bubble,
    seed_sphere,
)
from phydrax.interfacial_transport import InterfaceTensionMatrix


SIGMA = 0.025


def _prepared(
    seed: MultiRegionSurfaceSeed,
    material: Callable[[MultiRegionSurfaceTopology], FoamMaterialPlan],
    plan: FoamEquilibriumPlan | None = None,
    policy: MultiRegionSurfaceValidationPolicy | None = None,
) -> tuple[MultiRegionSurfaceTopology, MultiRegionSurfaceState, PreparedFoamEquilibrium]:
    topology = seed.topology(seed.capacity_plan(resource_id="foam-test"))
    state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, state, policy=policy)
    equilibrium = PreparedFoamEquilibrium(
        plan or FoamEquilibriumPlan(), surface, material(topology), state
    )
    return topology, state, equilibrium


def _soap(topology: MultiRegionSurfaceTopology) -> FoamMaterialPlan:
    return FoamMaterialPlan.soap_film(topology.region_ids, SIGMA)


@pytest.fixture(scope="module")
def sphere_problem() -> tuple[
    PreparedFoamEquilibrium, MultiRegionSurfaceState, FoamEquilibriumResult, float
]:
    seed = seed_sphere(1.0, subdivisions=1)
    topology, state, equilibrium = _prepared(
        seed, _soap, FoamEquilibriumPlan(method="sqp")
    )
    volume = 4.0 * np.pi / 3.0
    result = equilibrium.solve(state, equilibrium.parameters(jnp.asarray((volume,))))
    return equilibrium, state, result, volume


def test_spherical_soap_bubble_recovers_laplace_pressure(
    sphere_problem: tuple[
        PreparedFoamEquilibrium,
        MultiRegionSurfaceState,
        FoamEquilibriumResult,
        float,
    ],
) -> None:
    equilibrium, _, coarse, volume = sphere_problem
    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    laplace = 4.0 * SIGMA / radius
    evidence = coarse.evidence
    assert int(evidence.status) == FoamEquilibriumStatus.CONVERGED
    assert evidence.route == "sqp-exact-hessian"
    assert int(evidence.kkt_status) == FoamKKTStatus.REGULAR
    assert int(evidence.kkt_positive) == evidence.primal_dimension
    assert int(evidence.kkt_negative) == evidence.constraint_dimension == 7
    assert float(evidence.volume_residual) < 1e-8
    # Discrete virial identity 3 p V = 2 E and zero net force/torque multipliers.
    assert abs(float(evidence.virial_residual)) < 1e-9
    assert float(evidence.gauge_multiplier_norm) < 1e-9
    coarse_error = float(coarse.pressures[0]) / laplace - 1.0
    assert 0.0 < coarse_error < 0.02

    seed = seed_sphere(1.0, subdivisions=2)
    _, state, fine_problem = _prepared(seed, _soap)
    fine = fine_problem.solve(state, fine_problem.parameters(jnp.asarray((volume,))))
    assert fine.evidence.route == "augmented-lagrangian-trust-region"
    assert int(fine.evidence.status) == FoamEquilibriumStatus.CONVERGED
    fine_error = float(fine.pressures[0]) / laplace - 1.0
    assert 0.0 < fine_error < coarse_error / 3.0
    assert float(fine.pressures[1]) == 0.0


def _sphere_fit(points: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = np.hstack((2.0 * points, np.ones((points.shape[0], 1))))
    solution, *_ = np.linalg.lstsq(matrix, np.sum(points**2, axis=1), rcond=None)
    center = solution[:3]
    return center, float(np.sqrt(solution[3] + center @ center))


def _sheet_points(
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    pair: tuple[int, int],
) -> np.ndarray:
    labels = np.sort(topology.host_face_labels(), axis=1)
    faces = topology.host_faces()[np.all(labels == np.sort(pair), axis=1)]
    return np.asarray(state.positions)[np.unique(faces)]


def test_unequal_double_bubble_curvatures_pressures_and_plateau_angle() -> None:
    reference = StandardDoubleBubble(1.0, 0.8, 2.0 * SIGMA)
    seed = seed_double_bubble(1.0, 0.8, ring_points=18)
    topology, state, equilibrium = _prepared(seed, _soap)
    targets = jnp.asarray((reference.volume_first, reference.volume_second))
    result = equilibrium.solve(state, equilibrium.parameters(targets))
    evidence = result.evidence
    assert int(evidence.status) == FoamEquilibriumStatus.CONVERGED
    assert int(evidence.kkt_status) == FoamKKTStatus.REGULAR
    assert abs(float(evidence.virial_residual)) < 1e-9
    np.testing.assert_allclose(
        np.asarray(result.pressures[:2]), reference.pressures, rtol=0.01
    )
    assert float(result.energy) == pytest.approx(reference.energy, rel=0.01)
    center_1, radius_1 = _sphere_fit(_sheet_points(topology, result.state, (0, 2)))
    center_2, radius_2 = _sphere_fit(_sheet_points(topology, result.state, (1, 2)))
    _, radius_p = _sphere_fit(_sheet_points(topology, result.state, (0, 1)))
    assert 1.0 / radius_p == pytest.approx(1.0 / radius_2 - 1.0 / radius_1, rel=0.06)
    # Films meet at 120 degrees: the outer-cap normals at the junction differ by 60.
    junction = np.asarray(result.state.positions)[np.asarray(seed.vertex_set("junction"))]
    normals_1 = (junction - center_1) / radius_1
    normals_2 = (junction - center_2) / radius_2
    angles = np.degrees(np.arccos(np.sum(normals_1 * normals_2, axis=1)))
    np.testing.assert_allclose(angles, 60.0, atol=2.0)


def test_equal_double_bubble_has_flat_wall_and_equal_pressures() -> None:
    reference = StandardDoubleBubble(1.0, 1.0, 2.0 * SIGMA)
    seed = seed_double_bubble(1.0, 1.0, ring_points=18)
    topology, state, equilibrium = _prepared(seed, _soap)
    volumes = jnp.asarray((reference.volume_first, reference.volume_second))
    result = equilibrium.solve(state, equilibrium.parameters(volumes))
    assert int(result.evidence.status) == FoamEquilibriumStatus.CONVERGED
    assert result.evidence.tangential_gauge_dimension > 0
    assert int(result.evidence.kkt_status) == FoamKKTStatus.REGULAR
    assert int(result.evidence.kkt_zero) == 0
    pressures = np.asarray(result.pressures[:2])
    assert pressures[0] == pytest.approx(pressures[1], rel=1e-6)
    assert pressures[0] == pytest.approx(reference.pressures[0], rel=0.01)
    wall = _sheet_points(topology, result.state, (0, 1))
    centered = wall - np.mean(wall, axis=0)
    _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    assert singular[-1] / singular[0] < 1e-6


def _y_junction_seed() -> MultiRegionSurfaceSeed:
    angles = np.radians((0.0, 120.0, 240.0))
    points = [(0.1, -0.05, 0.0), (0.1, -0.05, 1.0)]
    for angle in angles:
        points += [
            (np.cos(angle), np.sin(angle), 0.0),
            (np.cos(angle), np.sin(angle), 1.0),
        ]
    faces, labels = [], []
    for sheet in range(3):
        bottom, top = 2 + 2 * sheet, 3 + 2 * sheet
        faces += [(0, bottom, top), (0, top, 1)]
        labels += [(sheet, (sheet - 1) % 3)] * 2
    return MultiRegionSurfaceSeed(
        np.asarray(points),
        np.asarray(faces),
        np.asarray(labels),
        ("w01", "w12", "w20"),
        ("boundary",) * 3,
        source="y-junction",
        vertex_sets={"wires": tuple(range(2, 8)), "junction": (0, 1)},
    )


def _y_material(
    seed: MultiRegionSurfaceSeed, tensions: np.ndarray
) -> Callable[[MultiRegionSurfaceTopology], FoamMaterialPlan]:
    def material(topology: MultiRegionSurfaceTopology) -> FoamMaterialPlan:
        ids = seed.vertex_set("wires") + seed.vertex_set("junction")
        fixed = np.asarray([[True, True, True]] * 6 + [[False, False, True]] * 2)
        wires = FoamWireConstraints(
            ids, seed.positions[list(ids)], fixed_components=fixed
        )
        return FoamMaterialPlan(
            InterfaceTensionMatrix(topology.region_ids, tensions), wires=wires
        )

    return material


def _sheet_directions(
    result: FoamEquilibriumResult, seed: MultiRegionSurfaceSeed
) -> np.ndarray:
    junction = np.asarray(result.state.positions[0])
    wires = seed.positions[[2, 4, 6]]
    directions = wires[:, :2] - junction[:2]
    return directions / np.linalg.norm(directions, axis=1)[:, None]


def test_unequal_tensions_recover_herring_neumann_angles() -> None:
    root_two = np.sqrt(2.0)
    # Sheet k separates wedge regions (k, k - 1): gamma_0 = sqrt(2), gamma_1 = gamma_2 = 1.
    tensions = np.asarray(((0.0, 1.0, root_two), (1.0, 0.0, 1.0), (root_two, 1.0, 0.0)))
    seed = _y_junction_seed()
    _, state, equilibrium = _prepared(seed, _y_material(seed, tensions))
    result = equilibrium.solve(state, equilibrium.parameters(jnp.zeros((0,))))
    evidence = result.evidence
    assert evidence.route == "unconstrained-trust-region"
    assert int(evidence.status) == FoamEquilibriumStatus.CONVERGED
    assert int(evidence.kkt_status) == FoamKKTStatus.REGULAR
    directions = _sheet_directions(result, seed)

    def angle(first: int, second: int) -> float:
        return float(
            np.degrees(np.arccos(np.clip(directions[first] @ directions[second], -1, 1)))
        )

    # Neumann triangle: cos(theta_12) = (g0^2 - g1^2 - g2^2) / (2 g1 g2) = 0.
    assert angle(1, 2) == pytest.approx(90.0, abs=1e-6)
    assert angle(0, 1) == pytest.approx(135.0, abs=1e-6)
    assert angle(2, 0) == pytest.approx(135.0, abs=1e-6)
    wedges = equilibrium.surface.junction_wedges(result.state.positions)
    triple = np.asarray(wedges.valence) == 3
    np.testing.assert_allclose(
        np.degrees(np.asarray(wedges.angles)[triple][0]), (135.0, 90.0, 135.0), atol=1e-6
    )


def test_implicit_equilibrium_derivatives_match_finite_differences(
    sphere_problem: tuple[
        PreparedFoamEquilibrium,
        MultiRegionSurfaceState,
        FoamEquilibriumResult,
        float,
    ],
) -> None:
    equilibrium, state, result, volume = sphere_problem
    assert bool(result.evidence.derivative_available)

    def pressure(targets: jax.Array, tension: jax.Array) -> jax.Array:
        parameters = equilibrium.parameters(targets, tension_values=tension)
        return equilibrium.implicit_equilibrium(result, parameters).pressures[0]

    targets = jnp.asarray((volume,))
    tension = jnp.asarray(2.0 * SIGMA)
    # One primal solve: linearize once and transpose the linear map for the VJP.
    value, linear = jax.linearize(pressure, targets, tension)
    tangent = linear(jnp.asarray((1.0,)), jnp.asarray(0.3))
    step = 1e-4
    shifted = [
        equilibrium.solve(
            state,
            equilibrium.parameters(
                targets + sign * step, tension_values=tension + sign * 0.3 * step
            ),
        ).pressures[0]
        for sign in (1.0, -1.0)
    ]
    finite_difference = (float(shifted[0]) - float(shifted[1])) / (2.0 * step)
    assert float(value) == pytest.approx(float(result.pressures[0]), rel=1e-10)
    assert float(tangent) == pytest.approx(finite_difference, rel=1e-6)
    transpose = jax.linear_transpose(linear, targets, tension)
    cotangent_targets, cotangent_tension = transpose(jnp.asarray(1.0))
    dual = float(cotangent_targets[0]) * 1.0 + float(cotangent_tension) * 0.3
    assert dual == pytest.approx(float(tangent), rel=1e-10)


def _partitioned_box_seed() -> MultiRegionSurfaceSeed:
    corners = [(x, y, z) for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    ring = [(0.5, y, z) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    points = np.asarray(corners + ring)

    def corner(x: int, y: int, z: int) -> int:
        return 4 * x + 2 * y + z

    def cut(y: int, z: int) -> int:
        return 8 + 2 * y + z

    faces, labels = [], []

    def quad(a: int, b: int, c: int, d: int, left: int, right: int) -> None:
        faces.extend([(a, b, c), (a, c, d)])
        labels.extend([(left, right)] * 2)

    quad(corner(0, 0, 0), corner(0, 0, 1), corner(0, 1, 1), corner(0, 1, 0), 0, 2)
    quad(corner(1, 0, 0), corner(1, 1, 0), corner(1, 1, 1), corner(1, 0, 1), 1, 2)
    for half, (x0, region) in enumerate(((0, 0), (1, 1))):
        del half
        near = (lambda y, z: corner(0, y, z)) if x0 == 0 else (lambda y, z: cut(y, z))
        far = (lambda y, z: cut(y, z)) if x0 == 0 else (lambda y, z: corner(1, y, z))
        quad(near(0, 0), far(0, 0), far(0, 1), near(0, 1), region, 2)
        quad(near(1, 0), near(1, 1), far(1, 1), far(1, 0), region, 2)
        quad(near(0, 0), near(1, 0), far(1, 0), far(0, 0), region, 2)
        quad(near(0, 1), far(0, 1), far(1, 1), near(1, 1), region, 2)
    quad(cut(0, 0), cut(1, 0), cut(1, 1), cut(0, 1), 0, 1)
    return MultiRegionSurfaceSeed(
        points,
        np.asarray(faces),
        np.asarray(labels),
        ("left", "right", "outside"),
        ("finite", "finite", "boundary"),
        source="partitioned-box",
    )


def test_redundant_volume_row_is_removed_with_declared_pressure_gauge() -> None:
    seed = _partitioned_box_seed()

    def material(topology: MultiRegionSurfaceTopology) -> FoamMaterialPlan:
        fixed = np.asarray([[True, True, True]] * 8 + [[False, True, True]] * 4)
        wires = FoamWireConstraints(
            tuple(range(12)), seed.positions, fixed_components=fixed
        )
        return FoamMaterialPlan.soap_film(topology.region_ids, SIGMA, wires=wires)

    topology, state, equilibrium = _prepared(seed, material)
    assert equilibrium.constrained_region_ids == ("left",)
    assert equilibrium.pressure_reference_region_ids == ("right",)
    finite = jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    initial_targets = equilibrium.surface.region_volumes(state.positions)[finite]
    dynamics = PreparedFoamDynamics(
        FoamDynamicsPlan(time_step=1.0e-5),
        equilibrium.surface,
        material(topology),
        RegionPressureAirPlan.incompressible(initial_targets),
        FoamDynamicsState(state),
    )
    basis = dynamics.constraint_basis.evidence
    assert dynamics.constrained_region_ids == equilibrium.constrained_region_ids
    assert dynamics.dependent_region_ids == equilibrium.pressure_reference_region_ids
    assert basis.partition_count == 1
    assert basis.closed_partition_count == 1
    assert basis.constrained_count == basis.numerical_rank == 1
    assert basis.dependent_count == 1
    assert basis.accepted
    result = equilibrium.solve(state, equilibrium.parameters(jnp.asarray((0.4, 0.6))))
    evidence = result.evidence
    assert int(evidence.status) == FoamEquilibriumStatus.CONVERGED
    assert evidence.pressure_reference_region_ids == ("right",)
    np.testing.assert_allclose(
        np.asarray(result.state.positions)[8:12, 0], 0.4, atol=1e-8
    )
    assert float(evidence.dependent_volume_residual) < 1e-8
    # A flat wall carries no pressure jump; the reference region is the gauge zero.
    assert abs(float(result.pressures[0])) < 1e-8
    assert float(result.pressures[1]) == 0.0
    inconsistent = equilibrium.solve(
        state, equilibrium.parameters(jnp.asarray((0.4, 0.5)))
    )
    assert (
        int(inconsistent.evidence.status)
        == FoamEquilibriumStatus.DEPENDENT_TARGETS_INCONSISTENT
    )


def test_catenoid_matches_the_pinned_primary_source_convention() -> None:
    critical, critical_ratio, critical_alpha = catenoid_critical_parameters()
    assert critical * np.tanh(critical) == pytest.approx(1.0, abs=1e-14)
    assert critical_ratio == pytest.approx(0.66274, abs=1e-5)
    assert critical_alpha == pytest.approx(0.55243, abs=1e-5)
    assert catenoid_area_ratio(critical_ratio, critical_alpha) == pytest.approx(
        1.1997, abs=1e-4
    )
    with pytest.raises(ValueError, match="No catenoid"):
        catenoid_stable_neck_ratio(0.7)

    ratio = 0.5
    alpha = catenoid_stable_neck_ratio(ratio)
    seed = seed_catenoid(1.0, ratio, ring_points=24, rows=10, neck_radius=alpha)

    def material(topology: MultiRegionSurfaceTopology) -> FoamMaterialPlan:
        ids = seed.vertex_set("ring-lower") + seed.vertex_set("ring-upper")
        wires = FoamWireConstraints(ids, seed.positions[seed.vertex_indices(ids)])
        return FoamMaterialPlan.soap_film(topology.region_ids, SIGMA, wires=wires)

    topology, state, equilibrium = _prepared(seed, material)
    result = equilibrium.solve(state, equilibrium.parameters(jnp.zeros((0,))))
    assert int(result.evidence.status) == FoamEquilibriumStatus.CONVERGED
    assert result.evidence.tangential_gauge_dimension > 0
    assert int(result.evidence.kkt_status) == FoamKKTStatus.REGULAR
    points = np.asarray(result.state.positions[: topology.vertex_count])
    neck = float(np.min(np.linalg.norm(points[:, :2], axis=1)))
    assert neck == pytest.approx(alpha, rel=0.02)
    area = float(result.energy) / (2.0 * SIGMA) / (2.0 * np.pi)
    assert area == pytest.approx(catenoid_area_ratio(ratio, alpha), rel=0.01)
    assert np.all(np.asarray(result.pressures) == 0.0)


def test_material_and_resource_contracts_refuse_invalid_foams() -> None:
    seed = seed_sphere(1.0, subdivisions=1)
    topology = seed.topology(seed.capacity_plan(resource_id="foam-contracts"))
    state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, state)
    foreign = FoamMaterialPlan.soap_film(("bubble", "outside"), SIGMA)
    with pytest.raises(ValueError, match="lacks surface regions"):
        PreparedFoamEquilibrium(FoamEquilibriumPlan(), surface, foreign, state)
    with pytest.raises(ValueError, match="maximum_dense_dimension"):
        PreparedFoamEquilibrium(
            FoamEquilibriumPlan(method="sqp", maximum_dense_dimension=64),
            surface,
            _soap(topology),
            state,
        )
    unknown = FoamWireConstraints((999,), np.zeros((1, 3)))
    with pytest.raises(ValueError, match="not surface vertices"):
        PreparedFoamEquilibrium(
            FoamEquilibriumPlan(),
            surface,
            FoamMaterialPlan.soap_film(topology.region_ids, SIGMA, wires=unknown),
            state,
        )
