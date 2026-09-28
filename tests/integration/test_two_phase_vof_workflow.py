#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax.bubble_dynamics as bubble_dynamics
from phydrax._fingerprint import canonical_fingerprint
from phydrax._sharp_measures import exact_sharp_geometry
from phydrax.discretization.finite_volume import CurvatureStatus


def _two_phase(*, surface_tension: Any = 0.0, body: Any = None) -> Any:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=1000.0,
        gas_density=10.0,
        surface_tension=surface_tension,
    )
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        maximum_iterations=200,
    ).prepare()
    x = (jnp.arange(8, dtype=jnp.float64) + 0.5) / 8
    y = (jnp.arange(8, dtype=jnp.float64) + 0.5) / 8
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    alpha = jnp.where((xx - 0.5) ** 2 + (yy - 0.5) ** 2 < 0.2**2, 1.0, 0.0)
    velocity = (
        jnp.full(discretization.face_layouts[0].shape, 0.01),
        jnp.zeros(discretization.face_layouts[1].shape),
    )
    state = two_phase.initial_state(alpha, velocity)
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(
        two_phase, body=body
    )
    return two_phase, method, method.initial_continuation(state)


def _bubbly_plan(
    two_phase: Any,
    *,
    near_contact_pressure: float = 1.0,
    atmosphere_pressure: float = 1.0e5,
    heat_capacity_ratio: float = 1.4,
    surface_tension: float = 0.05,
    coalescence: bool = True,
) -> Any:
    api = phx.applications.two_phase_flow
    identity = api.BubbleComponentPlan(
        two_phase,
        component_capacity=4,
        maximum_rounds=32,
        pair_capacity=8,
    )
    markers = api.MultiMarkerPlan(
        two_phase,
        marker_capacity=2,
        component_capacity=4,
        pair_capacity=8,
        proximity_radius=2,
    )
    compartments = api.BubbleCompartmentPlan(
        bubble_dynamics.CaloricIdealBubbleGasLaw(heat_capacity_ratio),
        bubble_dynamics.BubbleEnvironment(atmosphere_pressure, 300.0),
        capacity=4,
        dimension=2,
    )
    drainage = (
        api.FilmDrainageCoalescencePlan(
            regime="immobile",
            geometry="planar",
            pair_capacity=8,
            id_upper_bound=1_000,
            continuous_viscosity=1.0e-3,
            surface_tension=surface_tension,
            initial_film_thickness=1.0e-6,
            critical_thickness=5.0e-8,
        )
        if coalescence
        else None
    )
    return api.BubblyFlowPlan(
        two_phase,
        identity,
        markers=markers,
        compartments=compartments,
        coalescence=drainage,
        near_contact_pressure=near_contact_pressure,
        atmosphere_pressure=atmosphere_pressure,
    )


def _bubbly_checkpoint_case() -> tuple[Any, Any, Any, Any]:
    two_phase, _, base = _two_phase()
    plan = _bubbly_plan(two_phase)
    alpha = two_phase.alpha(base.state)
    bubbles = plan.initial_state(
        alpha,
        color=jnp.zeros(alpha.shape, dtype=jnp.int32),
        compartment_pressure=plan.atmosphere_pressure,
    )
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(
        two_phase, bubbles=plan
    )
    return (
        two_phase,
        plan,
        method,
        method.initial_continuation(base.state, bubbles=bubbles),
    )


def _qualified_static_two_phase() -> Any:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
            phx.discretization.UniformCellAxisSpec(8, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    cell_fraction = jnp.ones(discretization.cell_shape).at[0, :].set(0.5)
    x_aperture = jnp.ones(discretization.face_layouts[0].shape).at[0, :].set(0.0)
    y_aperture = jnp.ones(discretization.face_layouts[1].shape).at[0, :].set(0.5)
    geometry = exact_sharp_geometry(
        discretization.cell_volumes * cell_fraction,
        discretization.cell_volumes,
        (
            discretization.face_measures[0] * x_aperture,
            discretization.face_measures[1] * y_aperture,
        ),
        discretization.face_measures,
        source_id="static-vof-solid",
        source_fidelity="exact-polytope",
        measure_evidence_id="aligned-plane-cell-clipping",
        support_id=discretization.support.support_id,
        cell_field_id=discretization.cell_space.field_space_id,
        face_field_ids=tuple(
            space.field_space_id for space in discretization.face_spaces
        ),
        operator_id=operators.prepared_id,
        pairing_id=canonical_fingerprint(
            {
                "pressure": operators.pressure_space.space_id,
                "velocity": operators.velocity_space.space_id,
            }
        ),
    )
    material = phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
        liquid_density=1000.0,
        gas_density=10.0,
        liquid_viscosity=0.0,
        gas_viscosity=0.0,
    )
    prepared = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        material,
        maximum_iterations=200,
        geometry=geometry,
    ).prepare()
    return discretization, geometry, prepared


def test_two_phase_vof_workflow_scenario_1() -> None:
    two_phase, _, continuation = _two_phase()
    view = two_phase.view(continuation.state)

    assert bool(view.plic.valid)
    assert bool(view.topology.valid)
    assert jnp.all((view.alpha >= 0.0) & (view.alpha <= 1.0))
    assert jnp.all(view.density > 0.0)
    two_phase, method, continuation = _two_phase()
    initial_volume = jnp.sum(continuation.state.liquid_content)

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.005),
        None,
    )
    final_volume = jnp.sum(result.accepted_state.state.liquid_content)

    assert bool(result.successful)
    np.testing.assert_allclose(final_volume, initial_volume, atol=1e-10)
    assert result.accepted_state.evidence.divergence_residual <= 1e-7
    assert result.accepted_state.evidence.alpha_minimum >= -1e-12
    assert result.accepted_state.evidence.alpha_maximum <= 1.0 + 1e-12
    discretization, geometry, two_phase = _qualified_static_two_phase()
    alpha = jnp.zeros(discretization.cell_shape).at[3:5, 3:5].set(1.0)
    state = two_phase.initial_state(alpha)
    np.testing.assert_allclose(state.liquid_content, geometry.cell_fluid_measure * alpha)
    assert state.geometry_id == geometry.realization_id
    assert two_phase.view(state).geometry_id == geometry.realization_id

    triple_cut = alpha.at[0, 0].set(0.5)
    with pytest.raises(ValueError, match="both solid and liquid-gas PLIC"):
        two_phase.initial_state(triple_cut)
    discretization, geometry, two_phase = _qualified_static_two_phase()
    alpha = jnp.zeros(discretization.cell_shape).at[3:5, 3:5].set(1.0)
    state = two_phase.initial_state(alpha)
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(state)
    initial = jnp.sum(state.liquid_content)

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.001),
        None,
    )

    assert result.successful
    assert result.accepted_state.state.geometry_id == geometry.realization_id
    assert result.accepted_state.evidence.geometry_accepted
    np.testing.assert_allclose(
        jnp.sum(result.accepted_state.state.liquid_content), initial, atol=1.0e-10
    )
    discretization, _, two_phase = _qualified_static_two_phase()
    alpha = jnp.zeros(discretization.cell_shape).at[3:5, 3:5].set(1.0)
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(two_phase.initial_state(alpha))
    stale = eqx.tree_at(
        lambda value: value.state.geometry_epoch,
        continuation,
        continuation.state.geometry_epoch + 1,
    )

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        stale,
        jnp.asarray(0.001),
        None,
    )

    assert not result.candidate_state.evidence.geometry_accepted
    assert result.accepted_state.state.geometry_epoch == stale.state.geometry_epoch
    np.testing.assert_allclose(
        result.accepted_state.state.liquid_content, stale.state.liquid_content
    )


def test_balanced_capillarity_and_moving_body_are_finite() -> None:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(16, periodic=True),
            phx.discretization.UniformCellAxisSpec(16, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
            liquid_density=1000.0, gas_density=10.0, surface_tension=0.072
        ),
    ).prepare()
    samples = (jnp.arange(8) + 0.5) / 8.0
    centers = discretization.cell_centers
    offsets = (samples - 0.5) / 16.0
    x = centers[..., 0][..., None, None] + offsets[None, None, :, None]
    y = centers[..., 1][..., None, None] + offsets[None, None, None, :]
    alpha = jnp.mean((x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.3**2, axis=(-2, -1))
    body = phx.applications.two_phase_flow.TwoPhaseMovingBodyPlan(
        # ty: ignore[invalid-argument-type]
        (0.5, 0.5),
        0.1,
        # ty: ignore[invalid-argument-type]
        velocity=(0.0, 0.0),
        penalty=0.5,
    )
    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(
        two_phase, body=body
    )
    continuation = method.initial_continuation(two_phase.initial_state(alpha))

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.001),
        None,
    )

    evidence = result.accepted_state.evidence
    assert bool(result.successful)
    assert int(evidence.curvature_underresolved_count) == 0
    assert float(evidence.capillary_pressure_jump) > 0.0
    assert jnp.isfinite(result.accepted_state.ledger.capillary_work)
    assert jnp.isfinite(result.accepted_state.ledger.body_work)


def test_near_contact_bubbles_fall_back_and_refuse_unresolved_capillarity() -> None:
    count = 16
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=("two-phase",)
    ).prepare()
    two_phase = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFPlan(
        discretization,
        phx.applications.two_phase_flow.TwoPhaseMaterialPlan(
            liquid_density=1000.0,
            gas_density=10.0,
            surface_tension=0.072,
        ),
    ).prepare()
    samples = (jnp.arange(16) + 0.5) / 16.0
    offsets = (samples - 0.5) / count
    centers = discretization.cell_centers
    x = centers[..., 0][..., None, None] + offsets[None, None, :, None]
    y = centers[..., 1][..., None, None] + offsets[None, None, None, :]
    first = (x - 0.415) ** 2 + (y - 0.5) ** 2 < 0.08**2
    second = (x - 0.585) ** 2 + (y - 0.5) ** 2 < 0.08**2
    alpha = 1.0 - jnp.mean(first | second, axis=(-2, -1))

    geometry = two_phase.interface_geometry(alpha)
    curvature = geometry.curvature
    if curvature is None:
        raise AssertionError("Surface tension must prepare curvature evidence.")
    target = (6, 8)
    assert bool(curvature.fallback_attempted[target])
    assert int(curvature.evidence.status[target]) in (
        int(CurvatureStatus.FALLBACK),
        int(CurvatureStatus.UNDERRESOLVED),
    )

    method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(two_phase)
    continuation = method.initial_continuation(two_phase.initial_state(alpha))
    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(1.0e-4),
        None,
    )
    evidence = result.candidate_state.evidence

    assert not bool(result.successful)
    assert int(evidence.curvature_fallback_count) > 0
    assert int(evidence.curvature_underresolved_count) > 0
    np.testing.assert_allclose(
        result.accepted_state.state.liquid_content,
        continuation.state.liquid_content,
    )


def test_underresolved_interface_refuses_capillary_step() -> None:
    _, method, continuation = _two_phase(surface_tension=0.072)

    result = method.step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(0.001),
        None,
    )

    assert not bool(result.successful)
    assert int(result.candidate_state.evidence.curvature_underresolved_count) > 0
    np.testing.assert_allclose(
        result.accepted_state.state.liquid_content, continuation.state.liquid_content
    )


def test_two_phase_checkpoint_round_trip(tmp_path: Any) -> None:
    two_phase, method, continuation = _two_phase()
    target = tmp_path / "two-phase.chk"

    phx.applications.two_phase_flow.write_two_phase_checkpoint(
        target,
        two_phase,
        method,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        continuation,
    )
    time, step, restored = phx.applications.two_phase_flow.read_two_phase_checkpoint(
        target, two_phase, method, continuation
    )

    np.testing.assert_allclose(time, 0.0)
    assert int(step) == 0
    np.testing.assert_allclose(
        restored.state.liquid_content,
        continuation.state.liquid_content,
    )


@pytest.mark.strict_jax
def test_bubbly_parameter_realization_is_current_and_canonical() -> None:
    two_phase, _, _ = _two_phase()
    plan = _bubbly_plan(two_phase)
    baseline = plan.parameter_realization_id()
    equivalent = _bubbly_plan(two_phase)

    assert equivalent.plan_id == plan.plan_id
    assert equivalent.parameter_realization_id() == baseline

    trained = eqx.tree_at(
        lambda value: value.compartments.law.heat_capacity_ratio,
        plan,
        jnp.asarray(
            1.5,
            dtype=plan.compartments.law.heat_capacity_ratio.dtype,
        ),
    )
    assert trained.plan_id == plan.plan_id
    assert trained.parameter_realization_id() != baseline

    near_then_atmosphere = eqx.tree_at(
        lambda value: value.near_contact_pressure,
        plan,
        jnp.asarray(2.0, dtype=plan.near_contact_pressure.dtype),
    )
    near_then_atmosphere = eqx.tree_at(
        lambda value: value.atmosphere_pressure,
        near_then_atmosphere,
        jnp.asarray(9.0e4, dtype=plan.atmosphere_pressure.dtype),
    )
    atmosphere_then_near = eqx.tree_at(
        lambda value: value.atmosphere_pressure,
        plan,
        jnp.asarray(9.0e4, dtype=plan.atmosphere_pressure.dtype),
    )
    atmosphere_then_near = eqx.tree_at(
        lambda value: value.near_contact_pressure,
        atmosphere_then_near,
        jnp.asarray(2.0, dtype=plan.near_contact_pressure.dtype),
    )
    assert (
        near_then_atmosphere.parameter_realization_id()
        == atmosphere_then_near.parameter_realization_id()
    )

    changed_dtype = eqx.tree_at(
        lambda value: value.near_contact_pressure,
        plan,
        plan.near_contact_pressure.astype(jnp.float32),
    )
    assert changed_dtype.parameter_realization_id() != baseline


def test_bubbly_checkpoint_round_trip_accepts_equivalent_realization(
    tmp_path: Any,
) -> None:
    two_phase, plan, method, continuation = _bubbly_checkpoint_case()
    target = tmp_path / "bubbly-two-phase.chk"
    phx.applications.two_phase_flow.write_two_phase_checkpoint(
        target,
        two_phase,
        method,
        jnp.asarray(0.25),
        jnp.asarray(3, dtype=jnp.int32),
        continuation,
    )
    equivalent_plan = _bubbly_plan(two_phase)
    equivalent_method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(
        two_phase, bubbles=equivalent_plan
    )

    time, step, restored = phx.applications.two_phase_flow.read_two_phase_checkpoint(
        target,
        two_phase,
        equivalent_method,
        continuation,
    )

    assert equivalent_method.method_id == method.method_id
    assert equivalent_plan.parameter_realization_id() == plan.parameter_realization_id()
    np.testing.assert_allclose(time, 0.25)
    assert int(step) == 3
    np.testing.assert_array_equal(
        restored.state.liquid_content,
        continuation.state.liquid_content,
    )


@pytest.mark.parametrize(
    "name",
    ("near-contact", "atmosphere", "nested-gas", "nested-tension"),
)
def test_bubbly_checkpoint_rejects_changed_parameter_realization(
    tmp_path: Any,
    name: str,
) -> None:
    two_phase, _, method, continuation = _bubbly_checkpoint_case()
    target = tmp_path / "bubbly-parameter-mismatch.chk"
    phx.applications.two_phase_flow.write_two_phase_checkpoint(
        target,
        two_phase,
        method,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        continuation,
    )
    if name == "near-contact":
        changed_plan = _bubbly_plan(two_phase, near_contact_pressure=1.25)
    elif name == "atmosphere":
        changed_plan = _bubbly_plan(two_phase, atmosphere_pressure=9.5e4)
    elif name == "nested-gas":
        changed_plan = _bubbly_plan(two_phase, heat_capacity_ratio=1.5)
    elif name == "nested-tension":
        changed_plan = _bubbly_plan(two_phase, surface_tension=0.06)
    else:
        raise ValueError(f"Unknown parameter-realization test case {name!r}.")
    changed_method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(
        two_phase,
        bubbles=changed_plan,
    )

    assert changed_method.method_id == method.method_id
    with pytest.raises(ValueError, match="bubbly parameter realization"):
        phx.applications.two_phase_flow.read_two_phase_checkpoint(
            target,
            two_phase,
            changed_method,
            continuation,
        )


def test_bubbly_checkpoint_retains_structural_mismatch_refusal(tmp_path: Any) -> None:
    two_phase, _, method, continuation = _bubbly_checkpoint_case()
    target = tmp_path / "bubbly-structure-mismatch.chk"
    phx.applications.two_phase_flow.write_two_phase_checkpoint(
        target,
        two_phase,
        method,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        continuation,
    )
    changed_method = phx.applications.two_phase_flow.IncompressibleTwoPhaseVOFMethod(
        two_phase,
        bubbles=_bubbly_plan(two_phase, coalescence=False),
    )

    with pytest.raises(ValueError, match="method identity"):
        phx.applications.two_phase_flow.read_two_phase_checkpoint(
            target,
            two_phase,
            changed_method,
            continuation,
        )
