import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _runtime(
    *,
    capillary=False,
    surface_tension=0.0,
    embedded=False,
    embedded_field=None,
    embedded_field_id=None,
    contact_angle=np.pi / 2.0,
    contact_tolerance=1.0e-8,
    boundary_primitive=None,
):
    vertices = np.asarray(
        [(i / 4.0, j / 2.0) for j in range(3) for i in range(5)], dtype=float
    )
    cells = []
    for j in range(2):
        for i in range(4):
            lower = j * 5 + i
            cells.append((lower, lower + 1, lower + 6, lower + 5))
    eos = phx.equations.TwoMaterialEOSClosure(
        phx.equations.IdealGasMaterial(1.4),
        phx.equations.StiffenedGasMaterial(4.4, 2.0, 1.0),
    )
    system = phx.equations.TwoMaterialVOFSystem(2, eos=eos)
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(cells, dtype=np.int32),
        component_names=system.component_names,
    ).prepare()
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    vof = phx.discretization.UnstructuredVOFPlan(discretization, gradient)
    embedded_boundary = None
    embedded_boundaries = None
    contact_angles = None
    if embedded:
        if embedded_field is None:
            embedded_field = lambda points, args: points[:, 0] + 1.0
            embedded_field_id = "all-fluid-evidence-wall"
        elif not embedded_field_id:
            raise ValueError("Custom embedded fields require an explicit field ID.")
        embedded_boundary = phx.discretization.EmbeddedBoundaryPlan(
            discretization,
            embedded_field,
            field_id=embedded_field_id,
            body_tag=7,
        )
        embedded_metrics = embedded_boundary.prepare()
        embedded_boundaries = phx.discretization.UnstructuredEmbeddedBoundarySet(
            {7: phx.discretization.SlipWallBoundary()}
        )
    capillary_operator = (
        phx.discretization.BalancedCapillaryOperator(
            discretization,
            gradient,
            phx.discretization.SurfaceTensionPolicy(
                surface_tension,
                density_floor=1e-12,
                capillary_cfl=0.5,
            ),
        )
        if capillary
        else None
    )
    if embedded:
        contact_angles = phx.discretization.EmbeddedBoundaryContactAngleSet(
            {
                7: phx.discretization.ContactAngleCondition(
                    7,
                    contact_angle,
                    contact_tolerance,
                    (
                        "runtime-contact:"
                        f"{float(contact_angle).hex()}:"
                        f"{float(contact_tolerance).hex()}"
                    ),
                )
            },
            geometry_id=embedded_metrics.geometry_id,
            plic_id=vof.plan_id,
        )
    coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        embedded_boundary=embedded_boundary,
        embedded_boundaries=embedded_boundaries,
        vof=vof,
        capillarity=capillary_operator,
        contact_angles=contact_angles,
    )
    if boundary_primitive is None:
        boundary_policies = {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        }
    else:
        boundary_state = system.primitive_to_conserved(
            jnp.asarray(boundary_primitive, dtype=jnp.float32)
        )
        boundary_policies = {
            name: phx.discretization.ConstantStateBoundary(boundary_state)
            for name in discretization.boundary_patch_names
        }
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        boundary_policies,
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "two-material-vof-runtime", "state", system, boundaries
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method, coupling=coupling
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(
            fallback_flux=phx.discretization.RusanovFluxPlan()
        ),
    )
    return system, discretization, runtime


def test_two_material_vof_runtime_reconstructs_each_stage_and_advances():
    system, discretization, runtime = _runtime()
    alpha = jnp.where(discretization.cell_centers[:, 0] < 0.5, 0.8, 0.2)
    primitive = jnp.stack(
        (
            jnp.full_like(alpha, 1.2),
            jnp.full_like(alpha, 0.7),
            jnp.zeros_like(alpha),
            jnp.zeros_like(alpha),
            jnp.full_like(alpha, 2.5),
            alpha,
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    runtime_state = runtime.initialize_state(state, 0.0, 1.0e-4)
    result = runtime.advance(runtime_state)
    assert bool(result.accepted)
    average = result.runtime_state.cell_average()
    alpha_new = average[:, system.layout.alpha_index]
    assert jnp.all(jnp.isfinite(average))
    assert jnp.all((alpha_new >= 0.0) & (alpha_new <= 1.0))
    assert jnp.all(system.admissible(average))
    jitted = eqx.filter_jit(runtime.advance)(runtime_state)
    np.testing.assert_allclose(jitted.runtime_state.cell_average(), average)


def test_vof_stage_alpha_changes_stage_apertures():
    system, discretization, runtime = _runtime()
    vof = runtime.dynamics.coupling.vof
    assert vof is not None
    first = vof.reconstruct_stage(
        jnp.where(discretization.cell_centers[:, 0] < 0.5, 0.8, 0.2)
    )
    second = vof.reconstruct_stage(
        jnp.where(discretization.cell_centers[:, 0] < 0.5, 0.6, 0.4)
    )
    assert not jnp.allclose(first.owner_phase_apertures, second.owner_phase_apertures)
    assert jnp.all(first.interface_evidence)
    assert jnp.all(second.interface_evidence)


def test_zero_surface_tension_capillary_runtime_matches_vof_runtime():
    system, discretization, plain = _runtime(capillary=False)
    _, _, capillary = _runtime(capillary=True)
    alpha = jnp.where(discretization.cell_centers[:, 0] < 0.5, 0.8, 0.2)
    primitive = jnp.stack(
        (
            jnp.full_like(alpha, 1.2),
            jnp.full_like(alpha, 0.7),
            jnp.zeros_like(alpha),
            jnp.zeros_like(alpha),
            jnp.full_like(alpha, 2.5),
            alpha,
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    plain_state = plain.initialize_state(state, 0.0, 1.0e-4)
    capillary_state = capillary.initialize_state(state, 0.0, 1.0e-4)
    plain_result = plain.advance(plain_state)
    capillary_result = capillary.advance(capillary_state)
    assert bool(plain_result.accepted & capillary_result.accepted)
    np.testing.assert_allclose(
        capillary_result.runtime_state.cell_average(),
        plain_result.runtime_state.cell_average(),
    )


def test_positive_surface_tension_pure_phases_are_unchanged_eager_and_filter_jit():
    for pure_alpha in (0.0, 1.0):
        system, discretization, runtime = _runtime(
            capillary=True, surface_tension=1.0e8, embedded=True
        )
        alpha = jnp.full((discretization.cell_count,), pure_alpha, dtype=jnp.float32)
        primitive = jnp.stack(
            (
                jnp.full_like(alpha, 1.2),
                jnp.full_like(alpha, 0.7),
                jnp.zeros_like(alpha),
                jnp.zeros_like(alpha),
                jnp.full_like(alpha, 2.5),
                alpha,
            ),
            axis=-1,
        )
        state = system.primitive_to_conserved(primitive)
        runtime_state = runtime.initialize_state(state, 0.0, 1.0e-4)
        initial = runtime_state.cell_average()

        eager = runtime.advance(runtime_state)
        compiled = eqx.filter_jit(runtime.advance)(runtime_state)

        assert bool(eager.accepted)
        assert bool(compiled.accepted)
        assert int(eager.retries) == 0
        assert int(compiled.retries) == 0
        assert eager.embedded is not None
        assert compiled.embedded is not None
        np.testing.assert_allclose(
            eager.embedded.relative_cfl_step * eager.embedded.maximum_relative_rate,
            runtime.policy.cfl,
        )
        np.testing.assert_array_equal(
            compiled.embedded.relative_cfl_step,
            eager.embedded.relative_cfl_step,
        )
        np.testing.assert_array_equal(
            compiled.embedded.maximum_relative_rate,
            eager.embedded.maximum_relative_rate,
        )
        np.testing.assert_allclose(
            eager.runtime_state.cell_average(), initial, rtol=0.0, atol=1e-14
        )
        np.testing.assert_allclose(
            compiled.runtime_state.cell_average(), initial, rtol=0.0, atol=1e-14
        )


def test_capillary_dominated_candidate_preserves_limit_and_hyperbolic_evidence():
    system, discretization, runtime = _runtime(
        capillary=True, surface_tension=1.0e4, embedded=True
    )
    alpha = jnp.full((discretization.cell_count,), 0.5, dtype=jnp.float32)
    primitive = jnp.stack(
        (
            jnp.full_like(alpha, 1.2),
            jnp.full_like(alpha, 0.7),
            jnp.zeros_like(alpha),
            jnp.zeros_like(alpha),
            jnp.full_like(alpha, 2.5),
            alpha,
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    capillarity = runtime.dynamics.coupling.capillarity
    vof = runtime.dynamics.coupling.vof
    assert capillarity is not None
    assert vof is not None
    plic = vof.reconstruct_stage(alpha)
    assert bool(jnp.any(plic.interface_active))
    capillary_limit = capillarity.capillary_step(
        jnp.sqrt(discretization.cell_volumes),
        state[:, 0] + state[:, 1],
        interface_active=plic.interface_active,
    )
    attempted_step = 1.5 * capillary_limit
    runtime_state = runtime.initialize_state(state, 0.0, attempted_step)

    result = runtime.advance(runtime_state)

    assert bool(result.accepted)
    assert int(result.retries) == 1
    assert result.embedded is not None
    evidence = result.embedded
    np.testing.assert_allclose(evidence.relative_cfl_step, capillary_limit, rtol=1e-6)
    assert result.attempted_step_size > evidence.relative_cfl_step
    assert result.accepted_step_size <= evidence.relative_cfl_step
    np.testing.assert_allclose(
        result.accepted_step_size,
        result.attempted_step_size * runtime.policy.reduction_factor,
    )
    np.testing.assert_allclose(
        evidence.maximum_relative_rate,
        jnp.max(jnp.stack(evidence.stage_maximum_relative_rates)),
    )
    hyperbolic_limit = runtime.policy.cfl / evidence.maximum_relative_rate
    assert evidence.relative_cfl_step < hyperbolic_limit

    _, _, hyperbolic_runtime = _runtime(embedded=True)
    hyperbolic_result = hyperbolic_runtime.advance(
        hyperbolic_runtime.initialize_state(state, 0.0, result.accepted_step_size)
    )
    assert bool(hyperbolic_result.accepted)
    assert hyperbolic_result.embedded is not None
    np.testing.assert_allclose(
        evidence.maximum_relative_rate,
        hyperbolic_result.embedded.maximum_relative_rate,
        rtol=1e-6,
    )


def test_embedded_vof_contact_angle_stage_runtime_is_explicit_and_finite():
    system, discretization, _ = _runtime()
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    vof = phx.discretization.UnstructuredVOFPlan(discretization, gradient)
    embedded = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.1,
        field_id="contact-wall",
        body_tag=7,
    )
    metrics = embedded.prepare()
    embedded_boundaries = phx.discretization.UnstructuredEmbeddedBoundarySet(
        {7: phx.discretization.SlipWallBoundary()}
    )
    contact = phx.discretization.EmbeddedBoundaryContactAngleSet(
        {7: phx.discretization.ContactAngleCondition(7, np.pi / 2.0, 1e-8, "contact-90")},
        geometry_id=metrics.geometry_id,
        plic_id=vof.plan_id,
    )
    capillary = phx.discretization.BalancedCapillaryOperator(
        discretization,
        gradient,
        phx.discretization.SurfaceTensionPolicy(
            0.01,
            density_floor=1e-12,
            capillary_cfl=0.5,
        ),
    )
    coupling = phx.discretization.UnstructuredFiniteVolumeCouplingPlan(
        embedded_boundary=embedded,
        embedded_boundaries=embedded_boundaries,
        vof=vof,
        capillarity=capillary,
        contact_angles=contact,
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "embedded-vof-contact", "state", system, boundaries
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method, coupling=coupling
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(
            fallback_flux=phx.discretization.RusanovFluxPlan()
        ),
    )
    alpha = jnp.where(discretization.cell_centers[:, 1] < 0.5, 0.8, 0.2)
    primitive = jnp.stack(
        (
            jnp.full_like(alpha, 1.2),
            jnp.full_like(alpha, 0.7),
            jnp.zeros_like(alpha),
            jnp.zeros_like(alpha),
            jnp.full_like(alpha, 2.5),
            alpha,
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    first = runtime.advance(runtime.initialize_state(state, 0.0, 1.0e-4))
    assert bool(first.accepted)
    second = runtime.advance(first.runtime_state)
    assert bool(second.accepted)
    assert jnp.all(jnp.isfinite(second.runtime_state.cell_average()))
    assert (
        second.runtime_state.content_state.topology_epoch_id
        == first.runtime_state.content_state.topology_epoch_id
    )


def test_boundary_inflow_uses_exterior_composition_and_outflow_uses_owner_plic():
    velocity = 0.2
    system, discretization, runtime = _runtime(
        boundary_primitive=(1.0, 1.0, velocity, 0.0, 2.5, 1.0)
    )
    alpha = jnp.zeros((discretization.cell_count,), dtype=jnp.float32)
    interior_primitive = jnp.stack(
        (
            jnp.ones_like(alpha),
            jnp.ones_like(alpha),
            jnp.full_like(alpha, velocity),
            jnp.zeros_like(alpha),
            jnp.full_like(alpha, 2.5),
            alpha,
        ),
        axis=-1,
    )
    initial = runtime.initialize_state(
        system.primitive_to_conserved(interior_primitive),
        0.0,
        1.0e-4,
    )

    stage_metrics = phx.discretization.lower_static_unstructured_stage_metrics(
        discretization,
        time=initial.time,
        topology_epoch_id=initial.content_state.topology_epoch_id,
    )
    stage = runtime.dynamics.evaluate_stage(
        initial.content_state,
        stage_metrics,
        cfl=runtime.policy.cfl,
    )
    physical_stage = next(
        block for block in stage.ledger.blocks if block.block_kind == "physical"
    )
    stage_flux = np.asarray(physical_stage.flux_rate)
    stage_boundary = np.asarray(physical_stage.active_mask) & (
        np.asarray(physical_stage.neighbour_cells) < 0
    )
    stage_total_mass = stage_flux[:, 0] + stage_flux[:, 1]
    stage_inflow = stage_boundary & (stage_total_mass < 0.0)
    stage_outflow = stage_boundary & (stage_total_mass > 0.0)
    assert np.any(stage_inflow)
    assert np.any(stage_outflow)
    assert np.all(stage_flux[stage_inflow, 0] < 0.0)
    assert np.all(stage_flux[stage_inflow, system.layout.alpha_index] < 0.0)
    np.testing.assert_allclose(
        stage_flux[stage_outflow, 0],
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        stage_flux[stage_outflow, 1],
        stage_total_mass[stage_outflow],
        rtol=1.0e-6,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        stage_flux[stage_outflow, system.layout.alpha_index],
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )

    eager = runtime.advance(initial)
    compiled = eqx.filter_jit(runtime.advance)(initial)

    assert bool(eager.accepted)
    assert bool(compiled.accepted)
    initial_inventory = jnp.sum(
        initial.content_state.conservative_content,
        axis=0,
    )
    for result in (eager, compiled):
        final_inventory = jnp.sum(
            result.runtime_state.content_state.conservative_content,
            axis=0,
        )
        assert float(final_inventory[0] - initial_inventory[0]) > 0.0
        assert (
            float(
                final_inventory[system.layout.alpha_index]
                - initial_inventory[system.layout.alpha_index]
            )
            > 0.0
        )

    assert (
        eager.accepted_flux_integrals.ledger_id
        == compiled.accepted_flux_integrals.ledger_id
    )
    for eager_block, compiled_block in zip(
        eager.accepted_flux_integrals.blocks,
        compiled.accepted_flux_integrals.blocks,
        strict=True,
    ):
        np.testing.assert_allclose(
            compiled_block.flux_integral,
            eager_block.flux_integral,
            rtol=1.0e-12,
            atol=1.0e-30,
        )


def test_active_rotated_contact_failure_rejects_eager_and_filter_jit_before_flux():
    def rotated_wall(points, args):
        del args
        return 0.3 * points[:, 0] + 0.7 * points[:, 1] - 0.35

    system, discretization, runtime = _runtime(
        embedded=True,
        embedded_field=rotated_wall,
        embedded_field_id="strict-rotated-contact-wall",
        contact_angle=np.pi / 3.0,
        contact_tolerance=0.0,
    )
    centers = discretization.cell_centers
    alpha = jnp.clip(
        0.5 + 0.2 * (-0.8 * centers[:, 0] + 0.2 * centers[:, 1]),
        0.05,
        0.95,
    ).astype(jnp.float32)
    primitive = jnp.stack(
        (
            jnp.full_like(alpha, 1.2),
            jnp.full_like(alpha, 0.7),
            jnp.zeros_like(alpha),
            jnp.zeros_like(alpha),
            jnp.full_like(alpha, 2.5),
            alpha,
        ),
        axis=-1,
    )
    initial = runtime.initialize_state(
        system.primitive_to_conserved(primitive),
        0.0,
        1.0e-4,
    )

    for advance in (runtime.advance, eqx.filter_jit(runtime.advance)):
        with pytest.raises(
            Exception,
            match="contact-angle reconstruction evidence failed",
        ):
            advance(initial)
