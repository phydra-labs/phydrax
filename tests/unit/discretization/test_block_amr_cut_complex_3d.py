#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _topology():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(1) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((1, 1, 1), halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (phx.discretization.PatchBucketPlan(signature, 1),),
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0, 0), (1, 1, 1)),),
    )
    return phx.discretization.VariablePatchTopologyCompiler(plan).initial_topology()


def _topology_x_cells(count):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count),
            phx.discretization.UniformCellAxisSpec(1),
            phx.discretization.UniformCellAxisSpec(1),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((count, 1, 1), halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (phx.discretization.PatchBucketPlan(signature, 1),),
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0, 0), (count, 1, 1)),),
    )
    return phx.discretization.VariablePatchTopologyCompiler(plan).initial_topology()


def _identity(points, time, args):
    del time, args
    return points


def _resources(*, components=4):
    return phx.discretization.BlockAMRResourcePlan(
        maximum_components_per_cell=components,
        maximum_apertures_per_face=96,
        maximum_embedded_faces_per_cell=256,
        maximum_mortars=32,
        maximum_redistribution_routes=64,
        maximum_topology_events=16,
        maximum_communication_peers=16,
    )


def test_canonical_hierarchy_preflight_accounts_component_capacity():
    hierarchy = phx.discretization.canonicalize_patch_hierarchy(_topology())
    evidence = _resources(components=3).preflight(
        hierarchy,
        physical_component_count=5,
        dtype=np.float64,
    )

    assert hierarchy.levels[0].leaf_cell_count == 1
    assert evidence.valid
    assert evidence.cell_slots == 1
    assert evidence.control_volume_slots == 3
    assert evidence.reserved_device_bytes > 3 * 5 * 8


def test_three_dimensional_plane_cut_closes_volume_and_faces():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.37,
        "plane-x-0.37",
        7,
    )
    complex_ = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
    ).prepare()

    assert complex_.evidence.valid
    assert complex_.evidence.cut_cell_count == 1
    assert complex_.evidence.multivalued_cell_count == 0
    assert complex_.component_count == 1
    np.testing.assert_allclose(
        np.asarray(complex_.component_volumes)[0],
        0.63,
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(complex_.component_centers)[0],
        np.asarray((0.685, 0.5, 0.5)),
        rtol=2.0e-6,
        atol=2.0e-6,
    )
    assert np.count_nonzero(np.asarray(complex_.face_kinds) == 2) > 0


def test_subcell_piecewise_linear_field_preserves_disconnected_components():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: (points[:, 0] - 0.3) * (points[:, 0] - 0.7),
        "solid-slab",
        9,
    )
    complex_ = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
        subdivision=4,
    ).prepare()

    assert complex_.evidence.valid
    assert complex_.evidence.multivalued_cell_count == 1
    assert complex_.component_count == 2
    volumes = np.sort(
        np.asarray(complex_.component_volumes)[np.asarray(complex_.component_active)]
    )
    np.testing.assert_allclose(volumes, np.asarray((0.34, 0.34)), atol=2.0e-6)
    centers = np.sort(
        np.asarray(complex_.component_centers)[np.asarray(complex_.component_active), 0]
    )
    np.testing.assert_allclose(centers, np.asarray((0.17, 0.83)), atol=2.0e-6)


def test_cut_complex_lowers_to_polyhedral_finite_volume_plan():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.37,
        "plane-fv",
        3,
    )
    complex_ = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
    ).prepare()
    discretization = complex_.finite_volume_plan(component_names=("mass",))
    prepared = discretization.prepare()

    assert prepared.cell_count == 1
    np.testing.assert_allclose(prepared.cell_volumes, jnp.asarray((0.63,)), atol=2.0e-6)


def test_cut_complex_advances_through_public_finite_volume_runtime():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.37,
        "plane-runtime",
        12,
    )
    complex_ = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
    ).prepare()
    system = phx.equations.EulerSystem(3)
    discretization = complex_.finite_volume_plan(
        component_names=system.component_names
    ).prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: (
                phx.discretization.SlipWallBoundary()
                if name.startswith("embedded-")
                else phx.discretization.ExtrapolationBoundary()
            )
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "multivalued-cut-cell-runtime",
        "state",
        system,
        boundaries,
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 0.0, 1.0)),
        discretization.state_shape,
    )
    initial_average = system.primitive_to_conserved(primitive)
    initial = runtime.initialize_state(initial_average, 0.0, 1.0e-4)

    result = runtime.advance(initial)

    assert bool(result.accepted)
    np.testing.assert_allclose(
        result.runtime_state.cell_average(),
        initial_average,
        rtol=2.0e-6,
        atol=2.0e-7,
    )


def test_metric_common_refinement_preserves_content_and_transpose_pairing():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.37,
        "plane-transition",
        14,
    )
    body_set = phx.discretization.EmbeddedLevelSetBodySet((body,))
    source = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        body_set,
        _resources(),
        subdivision=1,
    ).prepare()
    target = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        body_set,
        _resources(),
        subdivision=1,
    ).prepare()
    transition = phx.discretization.MultivaluedCutCellTransition(
        source,
        target,
        tolerance=2.0e-8,
    )
    source_content = jnp.zeros((source.component_capacity, 2))
    source_content = source_content.at[0].set(jnp.asarray((2.0, 3.0)))

    result = transition.apply_content(source_content)

    assert bool(result.successful)
    np.testing.assert_allclose(result.target_total, jnp.asarray((2.0, 3.0)))
    constant = jnp.zeros((source.component_capacity, 1)).at[0].set(4.0)
    np.testing.assert_allclose(
        transition.apply_average(constant)[0],
        jnp.asarray((4.0,)),
    )
    target_cotangent = jnp.zeros((target.component_capacity, 2))
    target_cotangent = target_cotangent.at[0].set(jnp.asarray((5.0, 7.0)))
    pullback = transition.transpose_content(target_cotangent)
    lhs = jnp.vdot(result.target_content, target_cotangent)
    rhs = jnp.vdot(source_content, pullback)
    np.testing.assert_allclose(lhs, rhs, rtol=2.0e-6, atol=2.0e-7)


def test_multivalued_small_cell_redistribution_uses_aperture_neighbor():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.49,
        "small-sliver",
        18,
    )
    complex_ = phx.discretization.MultivaluedCutCellPlan(
        _topology_x_cells(2),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
    ).prepare()
    policy = phx.discretization.EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.1,
        maximum_recipients=4,
    )
    redistribution = phx.discretization.ConservativeSmallCellRedistributionPlan.from_multivalued_cut_complex(
        complex_,
        policy,
    )
    rate = jnp.zeros((complex_.component_count, 1))
    source = int(redistribution.source_cells[0])
    rate = rate.at[source, 0].set(1.0)

    result = redistribution.redistribute_rate(rate)

    assert result.activated
    np.testing.assert_allclose(jnp.sum(result.redistributed_rate), 1.0)
    np.testing.assert_allclose(result.redistributed_rate[source, 0], 0.2, atol=2.0e-6)


def test_polyhedral_viscous_residual_vanishes_for_constant_state():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.37,
        "viscous-plane",
        19,
    )
    complex_ = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
    ).prepare()
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.2),
        3,
    )
    discretization = complex_.finite_volume_plan(
        component_names=system.component_names
    ).prepare()
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.1, -0.05, 0.02, 1.0)),
        discretization.state_shape,
    )
    state = system.primitive_to_conserved(primitive)

    residual = phx.discretization.ViscousFluxPlan().unstructured_residual(
        system,
        jnp.asarray(0.0),
        state,
        discretization,
    )

    np.testing.assert_allclose(residual, 0.0, atol=2.0e-7)


def test_multivalued_composite_diffusion_solves_each_connected_nullspace():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: jnp.ones((points.shape[0],)),
        "full-fluid-diffusion",
        20,
    )
    complex_ = phx.discretization.MultivaluedCutCellPlan(
        _topology_x_cells(2),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
    ).prepare()
    diffusion = phx.discretization.MultivaluedCutCellDiffusionPlan(complex_, 1.0)
    constant = jnp.ones((diffusion.cell_count,))

    np.testing.assert_allclose(diffusion.apply(constant), 0.0, atol=2.0e-7)
    right_hand_side = jnp.asarray((1.0, -1.0))
    result = diffusion.solve(
        right_hand_side,
        relative_tolerance=1.0e-8,
        absolute_tolerance=1.0e-10,
    )

    assert bool(result.successful)
    np.testing.assert_allclose(
        diffusion.apply(result.value),
        right_hand_side,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        jnp.sum(result.value * diffusion.volumes),
        0.0,
        atol=2.0e-7,
    )


def test_moving_cut_cell_transaction_closes_swept_volume_and_content():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - (0.35 + 0.01 * time),
        "moving-plane",
        22,
    )
    cut_plan = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
        subdivision=1,
    )
    moving = phx.solver.MovingMultivaluedCutCellPlan(
        cut_plan,
        tolerance=2.0e-7,
    )
    state = moving.initialize(jnp.asarray(2.0), 0.0)

    result = moving.advance(state, 1.0, jnp.asarray(2.0))

    assert bool(result.accepted)
    np.testing.assert_allclose(
        result.state.cell_average()[result.state.complex.component_active],
        2.0,
        rtol=3.0e-6,
        atol=3.0e-7,
    )
    np.testing.assert_allclose(
        result.evidence.volume_balance_defect,
        0.0,
        atol=2.0e-7,
    )
    np.testing.assert_allclose(
        result.evidence.wall_content_integral,
        2.0 * (result.evidence.target_volume - result.evidence.source_volume),
        rtol=3.0e-6,
        atol=3.0e-7,
    )


def test_cut_complex_cochains_preserve_chain_identity_and_reflux_curl_divergence():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.37,
        "cochain-plane",
        24,
    )
    complex_ = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
    ).prepare()
    cochain = phx.discretization.CutCellCochainPlan(complex_).prepare()
    topology = cochain.topology.topology
    edge_count = topology.entities(1).count
    face_count = topology.entities(2).count
    edge_values = jnp.linspace(-0.3, 0.7, edge_count)
    face_values = jnp.linspace(0.2, 1.1, face_count)
    curl = topology.incidences[1].exterior_derivative().mv(edge_values)
    divergence_of_curl = topology.incidences[2].exterior_derivative().mv(curl)

    np.testing.assert_allclose(divergence_of_curl, 0.0, atol=2.0e-7)
    assert bool(cochain.metrics.valid)
    register = phx.solver.advanced.ElectromotiveForceRegister(
        jnp.zeros((edge_count,)),
        1.0e-3 * edge_values,
        register_id="cut-cell-emf",
    )
    updated, diagnostics = phx.solver.advanced.CutCellCochainSynchronizationPlan(
        cochain
    ).reflux_curl(
        face_values,
        register,
    )

    assert updated.shape == face_values.shape
    np.testing.assert_allclose(
        diagnostics.divergence_after,
        diagnostics.divergence_before,
        atol=2.0e-7,
    )


def test_adaptive_implicit_certificate_detects_corner_invisible_surface():
    def slab(points, time, args):
        del time, args
        return (points[:, 0] - 0.3) * (points[:, 0] - 0.7)

    def interval_bound(lower, upper, time, args):
        del time, args
        candidates = jnp.asarray(
            (
                (lower[0] - 0.3) * (lower[0] - 0.7),
                (upper[0] - 0.3) * (upper[0] - 0.7),
                (jnp.clip(0.5, lower[0], upper[0]) - 0.3)
                * (jnp.clip(0.5, lower[0], upper[0]) - 0.7),
            )
        )
        return jnp.min(candidates), jnp.max(candidates)

    body = phx.discretization.EmbeddedLevelSetBody(
        slab,
        "certified-hidden-slab",
        25,
    )
    adaptive = phx.discretization.AdaptiveImplicitSamplingPlan(
        _topology(),
        _identity,
        "identity-map",
        (
            phx.discretization.CertifiedImplicitBody(
                body,
                interval_bound,
                lambda points, values, time, args: (
                    jnp.any(values > 0.0) & jnp.any(values < 0.0)
                ),
                "piecewise-linear-simplex",
            ),
        ),
        maximum_depth=3,
    )

    prepared = adaptive.prepare()
    complex_ = adaptive.cut_plan(prepared, _resources()).prepare()

    assert prepared.evidence.valid
    assert prepared.evidence.maximum_depth_used >= 1
    assert prepared.subdivision >= 2
    assert complex_.component_count == 2


def test_localized_moving_step_resolves_multiple_enter_exit_events():
    def oscillating_plane(points, time, args):
        del args
        threshold = 0.5 - 2.8 * (time - 0.5) ** 2
        return points[:, 0] - threshold

    body = phx.discretization.EmbeddedLevelSetBody(
        oscillating_plane,
        "oscillating-plane",
        26,
    )
    cut_plan = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
        subdivision=1,
    )
    moving = phx.solver.MovingMultivaluedCutCellPlan(
        cut_plan,
        tolerance=3.0e-7,
    )
    initial = moving.initialize(jnp.asarray(2.0), 0.0)

    result = moving.advance_localized(
        initial,
        1.0,
        jnp.asarray(2.0),
        phx.solver.MovingTopologyLocalizationPlan(
            probe_count=16,
            bisection_iterations=18,
            minimum_event_separation=1.0e-3,
        ),
    )

    assert bool(result.accepted)
    assert len(result.localization.event_times) >= 2
    assert len(result.substeps) == len(result.localization.event_times) + 1
    np.testing.assert_allclose(
        result.state.cell_average()[result.state.complex.component_active],
        2.0,
        rtol=5.0e-6,
        atol=5.0e-7,
    )


def test_cut_cochain_transition_commutes_and_uses_metric_adjoint():
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.37,
        "cochain-transition-plane",
        28,
    )
    complex_ = phx.discretization.MultivaluedCutCellPlan(
        _topology(),
        _identity,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        _resources(),
    ).prepare()
    plan = phx.discretization.CutCellCochainPlan(complex_)
    state = plan.prepare()
    transfer = phx.discretization.CutCellCochainTransferPlan(
        plan,
        plan,
        state,
        state,
    )

    assert transfer.evidence.valid
    assert transfer.evidence.maximum_commuting_defect == 0.0
    for degree, entities in enumerate(state.topology.topology.entity_sets):
        source = jnp.linspace(-0.4, 0.7, entities.count)
        target = jnp.linspace(0.2, 0.9, entities.count)
        mapped = transfer.apply(degree, source)
        pullback = transfer.transpose(degree, target)
        adjoint = transfer.adjoint(degree, target)
        hodge = state.metrics.hodge_stars[degree]
        np.testing.assert_allclose(mapped, source)
        np.testing.assert_allclose(
            jnp.vdot(mapped, target),
            jnp.vdot(source, pullback),
            rtol=2.0e-6,
            atol=2.0e-7,
        )
        np.testing.assert_allclose(
            jnp.vdot(mapped * hodge, target),
            jnp.vdot(source * hodge, adjoint),
            rtol=2.0e-6,
            atol=2.0e-7,
        )


def test_embedded_body_set_supports_tagged_union_and_difference_csg():
    left = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.25,
        "left-solid",
        31,
    )
    right = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: 0.75 - points[:, 0],
        "right-solid",
        32,
    )
    points = jnp.asarray(((0.1, 0.5, 0.5), (0.5, 0.5, 0.5), (0.9, 0.5, 0.5)))

    union, tags = phx.discretization.EmbeddedLevelSetBodySet((left, right)).evaluate(
        points, jnp.asarray(0.0), None
    )
    difference, _ = phx.discretization.EmbeddedLevelSetBodySet(
        (left, right),
        operation="intersection",
        body_signs=(1, -1),
    ).evaluate(points, jnp.asarray(0.0), None)

    np.testing.assert_array_equal(union < 0.0, (True, False, True))
    np.testing.assert_array_equal(tags, (31, 31, 32))
    np.testing.assert_array_equal(difference < 0.0, (True, False, False))
