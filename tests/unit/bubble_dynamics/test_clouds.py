import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.bubble_dynamics as bd


AMBIENT = 101325.0
DENSITY = 998.0
SOUND_SPEED = 1481.0


def _model(
    equation: bd.RadialBubbleEquation = "rayleigh_plesset",
    *,
    viscosity: float = 0.0,
    tension: float = 0.0,
    sound_speed: float = SOUND_SPEED,
    interface: bd.AbstractBubbleInterfaceLaw | None = None,
) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        equation,
        bd.PolytropicBubbleGasLaw(1.4),
        bd.NewtonianBubbleLiquidLaw(viscosity),
        bd.CleanBubbleInterfaceLaw(tension) if interface is None else interface,
        bd.BubbleEnvironment(AMBIENT, 293.15),
        liquid_density=DENSITY,
        liquid_sound_speed=sound_speed,
    )


def _pair(
    model: bd.RadialBubbleModel,
    radius: float,
    distance: float,
    **kwargs: object,
) -> bd.BubbleSpeciesGroup:
    return bd.BubbleSpeciesGroup(
        model,
        np.full(2, radius),
        np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        bubble_ids=(0, 1),
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def _upward_crossing_frequency(times: np.ndarray, signal: np.ndarray) -> float:
    rising = np.nonzero((signal[:-1] < 0.0) & (signal[1:] >= 0.0))[0]
    crossings = times[rising] - signal[rising] * (times[rising + 1] - times[rising]) / (
        signal[rising + 1] - signal[rising]
    )
    return 2.0 * np.pi / float(np.mean(np.diff(crossings)))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_two_bubble_normal_modes_follow_the_coupled_frequencies(sign: float) -> None:
    radius, distance = 10.0e-6, 40.0e-6
    omega0 = float(
        bd.minnaert_angular_frequency(radius, AMBIENT, DENSITY, 1.4, surface_tension=0.0)
    )
    times = np.linspace(0.0, 4.0 * 2.0 * np.pi / omega0, 1601)[1:]
    group = _pair(
        _model(),
        radius,
        distance,
        initial_radii=radius * np.array([1.0 + 1.0e-4, 1.0 + sign * 1.0e-4]),
    )
    plan = bd.BubbleCloudPlan(
        (group,),
        bd.ConstantPressureDrive(0.0),
        times,
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
    )
    result = bd.solve_bubble_cloud(plan.prepare())
    assert bool(result.completed)
    signal = np.asarray(result.trajectory.radius[:, 0]) - radius
    measured = _upward_crossing_frequency(np.asarray(result.trajectory.times), signal)
    expected = omega0 / np.sqrt(1.0 + sign * radius / distance)
    assert measured == pytest.approx(expected, rel=2.0e-4)
    # The coupled Rayleigh-Plesset cloud is the Euler-Lagrange system of the
    # potential-flow Lagrangian: liquid kinetic energy change equals wall work.
    evidence = result.evidence
    assert evidence.work_identity_exact
    kinetic = (
        2.0
        * np.pi
        * DENSITY
        * radius**3
        * np.nanmax(np.asarray(result.trajectory.wall_velocity)) ** 2
    )
    assert abs(float(evidence.work_residual)) <= 1.0e-6 * kinetic


def _viscous_pair(radii: tuple[float, float]) -> bd.BubbleCloudResult:
    group = bd.BubbleSpeciesGroup(
        _model("keller_miksis", viscosity=2.0e-2, tension=0.072),
        np.array(radii),
        np.array([[0.0, 0.0, 0.0], [200.0e-6, 0.0, 0.0]]),
        bubble_ids=(0, 1),
    )
    plan = bd.BubbleCloudPlan(
        (group,),
        bd.HarmonicPressureDrive(1.0e4, 2.0 * np.pi * 4.0e5),
        np.linspace(0.0, 40.0e-6, 1601)[1:],
    )
    return bd.solve_bubble_cloud(plan.prepare())


def test_secondary_bjerknes_force_follows_the_resonance_sign_rule() -> None:
    below = _viscous_pair((2.0e-6, 3.0e-6))
    opposite = _viscous_pair((2.0e-6, 30.0e-6))
    for result in (below, opposite):
        assert bool(result.completed)
    toward = []
    for result in (below, opposite):
        forces = bd.mean_bjerknes_forces(result, 20.0e-6, 40.0e-6)
        assert bool(forces.covered)
        secondary = np.asarray(forces.secondary)
        # Both bubbles feel the same interaction sign (mutual attraction or
        # repulsion); equal magnitudes hold only for exactly periodic motion.
        assert secondary[0, 0] * secondary[1, 0] < 0.0
        toward.append(secondary[0, 0])
    # Both below resonance: in phase, attraction; straddling resonance: repulsion.
    assert toward[0] > 0.0
    assert toward[1] < 0.0


def _cloud_state(
    count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    side = int(np.ceil(count ** (1.0 / 3.0)))
    axis = np.arange(side) * 60.0e-6
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )[:count]
    points = lattice + rng.uniform(-10.0e-6, 10.0e-6, size=lattice.shape)
    radii = rng.uniform(6.0e-6, 12.0e-6, size=count)
    return (
        points,
        radii,
        radii * rng.uniform(0.9, 1.1, size=count),
        rng.uniform(-2.0, 2.0, size=count),
    )


def test_dense_coupled_accelerations_match_a_direct_reference() -> None:
    points, radii, initial, velocity = _cloud_state(5, 1)
    model = _model("keller_miksis", viscosity=1.0e-3, tension=0.072)
    drive = bd.HarmonicPressureDrive(5.0e4, 2.0 * np.pi * 3.0e5)
    group = bd.BubbleSpeciesGroup(
        model,
        radii,
        points,
        bubble_ids=tuple(range(5)),
        initial_radii=initial,
        initial_wall_velocities=velocity,
    )
    prepared = bd.BubbleCloudPlan((group,), drive, np.array([1.0e-6])).prepare()
    time = 3.0e-7
    rates = prepared.rates(prepared.initial_state, time)
    assert bool(rates.coupling_successful)
    # Independent reference: isolated-bubble rates of each member give a_i and
    # f_i; the nonsymmetric system a_i R̈_i + Σ R_j² R̈_j/d_ij = f_i − Σ 2R_jṘ_j²/d_ij
    # is solved on the host.
    members = prepared.initial_state.groups[0]
    inertia = np.zeros(5)
    forcing = np.zeros(5)
    for index in range(5):
        member = jax.tree.map(lambda leaf, index=index: leaf[index], members)
        single = model.rates(member, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(time), drive)
        inertia[index] = float(single.inertia_fraction) * initial[index]
        forcing[index] = float(single.acceleration) * inertia[index]
    separation = np.linalg.norm(points[:, None] - points[None], axis=-1)
    kernel = np.where(
        np.eye(5, dtype=bool), 0.0, 1.0 / np.where(np.eye(5, dtype=bool), 1.0, separation)
    )
    matrix = np.diag(inertia) + kernel * initial[None, :] ** 2
    reference = np.linalg.solve(matrix, forcing - kernel @ (2.0 * initial * velocity**2))
    np.testing.assert_allclose(np.asarray(rates.acceleration), reference, rtol=1.0e-11)


def test_fmm_route_matches_the_dense_route_within_its_declared_tolerance() -> None:
    count = 125
    points, radii, initial, velocity = _cloud_state(count, 2)
    corrections = {}
    for route in ("dense", "fmm"):
        group = bd.BubbleSpeciesGroup(
            _model(tension=0.072),
            radii,
            points,
            bubble_ids=tuple(range(count)),
            initial_radii=initial,
            initial_wall_velocities=velocity,
        )
        plan = bd.BubbleCloudPlan(
            (group,),
            bd.HarmonicPressureDrive(2.0e4, 2.0 * np.pi * 1.0e5),
            np.array([1.0e-6]),
            route=route,
            resources=bd.BubbleCloudResourcePolicy(maximum_dense_entries=count * count),
        )
        prepared = plan.prepare()
        rates = prepared.rates(prepared.initial_state, 2.5e-7)
        assert bool(rates.coupling_successful)
        corrections[route] = np.asarray(rates.acceleration - rates.uncoupled_acceleration)
    error = np.max(np.abs(corrections["fmm"] - corrections["dense"]))
    assert error <= FMM_COUPLING_TOLERANCE * np.max(np.abs(corrections["dense"]))


FMM_COUPLING_TOLERANCE = 1.0e-5


def test_fmm_overlap_certificate_matches_dense_contact_semantics() -> None:
    radius = 10.0e-6
    group = bd.BubbleSpeciesGroup(
        _model(),
        np.full(3, radius),
        np.array(
            [[0.0, 0.0, 0.0], [3.0 * radius, 0.0, 0.0], [1.0e-3, 0.0, 0.0]]
        ),
        bubble_ids=(0, 1, 2),
    )
    drive = bd.ConstantPressureDrive(0.0)
    times = np.array([1.0e-8])
    unit_contact = bd.BubbleCloudResourcePolicy(
        overlap_growth_bound=1.1,
        maximum_candidate_pairs=2,
        fmm_depth=2,
        fmm_order=1,
        maximum_coupling_iterations=4,
    )
    nonoverlapping = []
    for route in ("dense", "fmm"):
        prepared = bd.BubbleCloudPlan(
            (group,), drive, times, route=route, resources=unit_contact
        ).prepare()
        nonoverlapping.append(float(prepared.contact_ratio(prepared.initial_state)) > 1.0)
        if route == "fmm":
            assert prepared.candidates is not None
            assert int(prepared.candidates.required) == 0
    assert nonoverlapping == [True, True]

    extended_contact = bd.BubbleCloudResourcePolicy(
        minimum_contact_ratio=2.0,
        overlap_growth_bound=1.1,
        maximum_candidate_pairs=2,
        fmm_depth=2,
        fmm_order=1,
        maximum_coupling_iterations=4,
    )
    statuses = {}
    for route in ("dense", "fmm"):
        result = bd.solve_bubble_cloud(
            bd.BubbleCloudPlan(
                (group,), drive, times, route=route, resources=extended_contact
            ).prepare()
        )
        statuses[route] = int(result.status)
        if route == "fmm":
            assert int(result.evidence.candidate_pairs_required) == 1
    assert statuses == {
        "dense": int(bd.BubbleDynamicsStatus.OVERLAP),
        "fmm": int(bd.BubbleDynamicsStatus.OVERLAP),
    }


def test_fmm_refuses_initial_growth_beyond_its_overlap_certificate() -> None:
    radius = 10.0e-6
    growth_bound = 1.1
    group = _pair(
        _model(),
        radius,
        100.0e-6,
        initial_radii=np.full(2, 1.2 * radius),
    )
    prepared = bd.BubbleCloudPlan(
        (group,),
        bd.ConstantPressureDrive(0.0),
        np.array([1.0e-8]),
        route="fmm",
        resources=bd.BubbleCloudResourcePolicy(
            overlap_growth_bound=growth_bound,
            maximum_candidate_pairs=1,
            fmm_depth=2,
            fmm_order=1,
            maximum_coupling_iterations=4,
        ),
    ).prepare()
    assert prepared.growth_limit is not None
    assert float(prepared.growth_limit) == pytest.approx(growth_bound * radius)

    result = bd.solve_bubble_cloud(prepared)

    assert int(result.status) == int(bd.BubbleDynamicsStatus.VALIDITY_EXCEEDED)
    assert int(result.evidence.accepted_steps) == 0


def test_overlap_capacity_and_conditioning_are_refused() -> None:
    drive = bd.ConstantPressureDrive(0.0)
    times = np.array([1.0e-6])
    overlapping = _pair(_model(), 10.0e-6, 15.0e-6)
    refused = bd.solve_bubble_cloud(
        bd.BubbleCloudPlan((overlapping,), drive, times).prepare()
    )
    assert int(refused.status) == int(bd.BubbleDynamicsStatus.OVERLAP)
    assert int(refused.evidence.accepted_steps) == 0
    np.testing.assert_array_equal(
        np.asarray(refused.terminal_state.groups[0].radius), 10.0e-6
    )
    close = _pair(_model(), 10.0e-6, 21.0e-6)
    strict = bd.BubbleCloudResourcePolicy(maximum_condition_number=1.5)
    conditioned = bd.solve_bubble_cloud(
        bd.BubbleCloudPlan((close,), drive, times, resources=strict).prepare()
    )
    assert int(conditioned.status) == int(bd.BubbleDynamicsStatus.ILL_CONDITIONED)
    many = bd.BubbleSpeciesGroup(
        _model(),
        np.full(70, 1.0e-6),
        np.arange(210.0).reshape(70, 3) * 1.0e-4,
        bubble_ids=tuple(range(70)),
    )
    with pytest.raises(ValueError, match="exceeds the resource policy"):
        bd.BubbleCloudPlan((many,), drive, times, route="dense")


def test_growing_bubbles_stop_at_contact_with_the_overlap_status() -> None:
    group = _pair(_model(tension=0.072), 10.0e-6, 25.0e-6)
    plan = bd.BubbleCloudPlan(
        (group,), bd.ConstantPressureDrive(-5.0e4), np.linspace(0.0, 20.0e-6, 41)[1:]
    )
    result = bd.solve_bubble_cloud(plan.prepare())
    assert int(result.status) == int(bd.BubbleDynamicsStatus.OVERLAP)
    assert float(result.terminal_time) < 20.0e-6
    radius = np.asarray(result.terminal_state.groups[0].radius)
    assert np.sum(radius) == pytest.approx(25.0e-6, rel=1.0e-6)


def test_retarded_coupling_converges_to_incompressible_as_sound_speed_grows() -> None:
    times = np.linspace(0.0, 6.0e-6, 61)[1:]
    drive = bd.HarmonicPressureDrive(3.0e4, 2.0 * np.pi * 2.0e5)
    differences = []
    for sound_speed in (1481.0, 5924.0):
        group = bd.BubbleSpeciesGroup(
            _model(tension=0.072, sound_speed=sound_speed),
            np.array([10.0e-6, 8.0e-6]),
            np.array([[0.0, 0.0, 0.0], [80.0e-6, 0.0, 0.0]]),
            bubble_ids=(0, 1),
        )
        incompressible = bd.solve_bubble_cloud(
            bd.BubbleCloudPlan((group,), drive, times).prepare()
        )
        retarded = bd.solve_bubble_cloud(
            bd.BubbleCloudPlan((group,), drive, times, coupling="retarded").prepare()
        )
        assert bool(incompressible.completed) and bool(retarded.completed)
        assert float(retarded.evidence.maximum_spectral_radius) < 1.0
        differences.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            retarded.trajectory.radius - incompressible.trajectory.radius
                        )
                    )
                )
            )
        )
    # First order in the retardation d/c.
    assert differences[1] < 0.4 * differences[0]


def test_dense_packing_is_refused_by_the_retarded_stability_guard() -> None:
    axis = np.arange(3) * 22.0e-6
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    group = bd.BubbleSpeciesGroup(
        _model(), np.full(27, 10.0e-6), points, bubble_ids=tuple(range(27))
    )
    plan = bd.BubbleCloudPlan(
        (group,), bd.ConstantPressureDrive(0.0), np.array([1.0e-6]), coupling="retarded"
    )
    prepared = plan.prepare()
    condition, definite, spectral = prepared.coupling_spectrum(
        prepared.initial_state, 0.0
    )
    # The instantaneous system stays positive definite while ρ(W) ≥ 1.
    assert bool(definite) and float(spectral) >= 1.0
    result = bd.solve_bubble_cloud(prepared)
    assert int(result.status) == int(bd.BubbleDynamicsStatus.NEUTRAL_UNSTABLE)
    del condition


def test_member_models_batch_into_one_group_like_separate_groups() -> None:
    points = np.array([[0.0, 0.0, 0.0], [30.0e-6, 0.0, 0.0]])
    radii = np.array([5.0e-6, 6.0e-6])
    tensions = (0.03, 0.072)
    batched = bd.BubbleSpeciesGroup(
        [_model(tension=value) for value in tensions],
        radii,
        points,
        bubble_ids=(0, 1),
        initial_radii=1.1 * radii,
    )
    separate = tuple(
        bd.BubbleSpeciesGroup(
            _model(tension=tensions[index]),
            radii[index : index + 1],
            points[index : index + 1],
            bubble_ids=(index,),
            initial_radii=1.1 * radii[index : index + 1],
        )
        for index in range(2)
    )
    drive = bd.ConstantPressureDrive(0.0)
    times = np.array([1.0e-6])
    one = bd.BubbleCloudPlan((batched,), drive, times).prepare()
    two = bd.BubbleCloudPlan(separate, drive, times).prepare()
    np.testing.assert_allclose(
        np.asarray(one.rates(one.initial_state, 0.0).acceleration),
        np.asarray(two.rates(two.initial_state, 0.0).acceleration),
        rtol=1.0e-12,
    )
    with pytest.raises(ValueError, match="share one law structure"):
        bd.BubbleSpeciesGroup(
            [_model(), _model("keller_miksis")], radii, points, bubble_ids=(0, 1)
        )
    with pytest.raises(ValueError, match="smooth interface"):
        bd.BubbleSpeciesGroup(
            _model(interface=bd.MarmottantShell(0.5, 0.02, 0.072, 1.0e-8)),
            radii,
            points,
            bubble_ids=(0, 1),
        )
    with pytest.raises(ValueError, match="unique across the cloud"):
        bd.BubbleCloudPlan((batched, batched), drive, times)
