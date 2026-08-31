#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _bridge(shape):
    dimension = len(shape)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    return phx.discretization.StructuredCochainBridge(grid)


def test_resource_policy_fails_before_projection_and_streaming_matches_steps():
    bridge = _bridge((2, 2, 2))
    with pytest.raises(ValueError, match="resource budget"):
        phx.solver.CompatibleMaxwellPlan(
            bridge,
            resources=phx.solver.maxwell.MaxwellResourcePolicy(maximum_total_bytes=1),
        ).prepare()
    runtime = phx.solver.CompatibleMaxwellPlan(bridge).prepare()
    state = runtime.initialize()
    dt = 0.05 * runtime.stable_dt
    expected = runtime.leapfrog_step(0.0, state, dt)
    expected = runtime.leapfrog_step(dt, expected, dt)
    solved = phx.solver.maxwell.solve_compatible_maxwell(runtime, state, 0.0, dt, 2)
    np.testing.assert_allclose(
        solved.final_state.primary.electric_displacement,
        expected.primary.electric_displacement,
    )
    assert solved.resource_estimate.logical_primary_bytes > 0
    assert solved.step_count == 2


def test_projection_is_sparse_and_elision_is_capability_gated():
    bridge = _bridge((2, 2, 2))
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        magnetic_constraint=phx.solver.maxwell.MaxwellMagneticConstraintPolicy("project"),
    ).prepare()
    assert runtime.magnetic_incidence.sparse_storage().nnz > 0
    assert (
        runtime.magnetic_constraint_solver.problem.operator is runtime.magnetic_incidence
    )
    elided = phx.solver.CompatibleMaxwellPlan(
        bridge,
        magnetic_constraint=phx.solver.maxwell.MaxwellMagneticConstraintPolicy("elide"),
    ).prepare()
    assert elided.magnetic_projection_elided
    with pytest.raises(ValueError, match="closedness evidence"):
        phx.solver.CompatibleMaxwellPlan(
            bridge,
            pml=phx.solver.maxwell.MaxwellCPMLPlan(0),
            magnetic_constraint=phx.solver.maxwell.MaxwellMagneticConstraintPolicy(
                "elide"
            ),
        ).prepare()


def test_genuine_tez_tmz_layouts_and_chain_invariants():
    bridge = _bridge((3, 4))
    tez = phx.solver.CompatibleMaxwellPlan(bridge, polarization="tez").prepare()
    tmz = phx.solver.CompatibleMaxwellPlan(bridge, polarization="tmz").prepare()
    assert tez.layout.electric_degree == 1 and tez.layout.magnetic_degree == 2
    assert tmz.layout.electric_degree == 0 and tmz.layout.magnetic_degree == 1
    assert tmz.initialize().primary.charge.shape == (0,)
    assert tez.magnetic_constraint(tez.initialize()).shape == ()
    scalar = jnp.sin(jnp.arange(tmz.layout.electric_count, dtype=float))
    exact_b = -bridge.exterior_derivative(0, scalar)
    np.testing.assert_allclose(bridge.exterior_derivative(1, exact_b), 0.0, atol=1e-14)


def test_packed_cpml_terms_and_fixed_coefficients_share_exact_support():
    bridge = _bridge((4, 4, 4))
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        pml=phx.solver.maxwell.MaxwellCPMLPlan(1),
    ).prepare()
    assert runtime.pml is not None
    assert runtime.pml.state_elements < bridge.dimension * sum(runtime.primary_counts[:2])
    state = runtime.pml.initialize(dtype=complex)
    assert all(
        memory.ndim == 1 for memory in (*state.electric_memory, *state.magnetic_memory)
    )
    coefficients = runtime.pml.bind_coefficients(0.1, 0.05)
    assert tuple(value.term_id for value in coefficients.electric) == tuple(
        value.term_id for value in runtime.pml.electric_terms
    )


def test_prepared_paired_source_substep_phases_and_charge_continuity():
    bridge = _bridge((2, 2, 2))
    layout = phx.solver.maxwell.MaxwellCochainLayout(bridge)
    source = phx.solver.maxwell.MaxwellPairedCurrentSourcePlan(
        jnp.asarray([0]),
        jnp.asarray([2.0]),
        jnp.asarray([0]),
        jnp.asarray([3.0]),
        angular_frequency=2.0,
    )
    prepared = source.prepare(bridge, layout)
    start = prepared.sample(0.0)
    middle = prepared.sample(0.25)
    np.testing.assert_allclose(
        middle.electric_current[0] / start.electric_current[0], jnp.exp(-0.5j)
    )
    runtime = phx.solver.CompatibleMaxwellPlan(bridge, sources=(source,)).prepare()
    rate = runtime.drift(0.0, runtime.initialize())
    np.testing.assert_allclose(
        rate.charge,
        bridge.codifferential(layout.electric_degree, rate.electric_displacement),
    )


def test_harmonic_defects_and_independent_batch_match_serial():
    bridge = _bridge((2, 2))
    runtime = phx.solver.CompatibleMaxwellPlan(bridge, polarization="tez").prepare()
    frequency = phx.solver.maxwell.FrequencyMaxwellOperator(
        bridge.cochain,
        runtime.layout,
        runtime.constitutive,
        0.4,
    )
    field = jnp.linspace(0.0, 1.0, frequency.size)
    report = frequency.defect(field, frequency.mv(field))
    np.testing.assert_allclose(report.absolute_norm, 0.0, atol=1e-13)
    state = runtime.initialize()
    dt = 0.05 * runtime.stable_dt
    batch = phx.solver.maxwell.prepare_compatible_maxwell_case_batch(
        (runtime, runtime), (state, state), (2,), dt
    )
    solved = phx.solver.maxwell.solve_compatible_maxwell_case_batch(batch, 0.0, 1)
    serial = runtime.leapfrog_step(0.0, state, dt)
    np.testing.assert_allclose(
        solved.final_states.primary.magnetic_flux[0], serial.primary.magnetic_flux
    )
    baseline = solved.final_states.primary.electric_displacement[:, 0]
    jacobian = jax.jacfwd(lambda values: values + baseline)(jnp.ones((2,)))
    np.testing.assert_allclose(jacobian, jnp.eye(2))


def test_scalar_geometry_material_assembly_is_degree_aligned_and_positive():
    bridge = _bridge((3, 3))
    layout = phx.solver.maxwell.MaxwellCochainLayout(bridge, "tez")
    geometry = phx.geometry.Square(center=(0.5, 0.5), side=4.0).compile()
    assembled = phx.solver.maxwell.assemble_scalar_maxwell_material(
        geometry,
        bridge,
        layout,
        inside_permittivity=4.0,
        outside_permittivity=1.0,
        inside_permeability=2.0,
        outside_permeability=1.0,
    )
    assert assembled.constitutive.permittivity.shape == (layout.electric_count,)
    assert assembled.constitutive.permeability.shape == (layout.magnetic_count,)
    np.testing.assert_allclose(assembled.constitutive.permittivity, 4.0)
    np.testing.assert_allclose(assembled.constitutive.permeability, 2.0)


def test_maxwell_resource_estimate_reserves_complex_primary_storage():
    runtime = phx.solver.CompatibleMaxwellPlan(_bridge((2, 2, 2))).prepare()
    expected = np.dtype(np.complex128).itemsize * sum(runtime.primary_counts)
    assert runtime.resource_estimate.logical_primary_bytes == expected


def test_tmz_reversible_step_uses_the_retained_electric_degree():
    bridge = _bridge((3, 3))
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tmz",
    ).prepare()
    potential = jnp.linspace(0.0, 1.0, runtime.layout.electric_count)
    state = runtime.initialize(
        electric_displacement=potential,
        magnetic_flux=-bridge.exterior_derivative(0, potential),
    )
    step_size = 0.01 * runtime.stable_dt
    advanced = runtime.leapfrog_step(0.0, state, step_size)
    reversible = phx.solver.maxwell.MaxwellReversibleAdjointPlan(runtime, 1)
    restored = reversible.inverse_step(step_size, advanced, step_size)
    np.testing.assert_allclose(
        restored.primary.electric_displacement,
        state.primary.electric_displacement,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        restored.primary.magnetic_flux,
        state.primary.magnetic_flux,
        rtol=1e-11,
        atol=1e-11,
    )


def test_mode_observer_demodulates_exp_minus_iwt_fields():
    runtime = phx.solver.CompatibleMaxwellPlan(
        _bridge((2, 2)),
        polarization="tez",
    ).prepare()
    electric_mode = jnp.zeros((runtime.layout.electric_count, 1)).at[0, 0].set(1.0)
    magnetic_mode = jnp.zeros((runtime.layout.magnetic_count, 1)).at[0, 0].set(1.0)
    angular_frequency = 2.5
    observer = phx.solver.maxwell.ModeAmplitudeObserverPlan(
        electric_mode,
        magnetic_mode,
        jnp.asarray([angular_frequency]),
        direction=1,
    ).prepare(runtime.layout)
    observation = observer.initialize()
    for time in (0.2, 0.7):
        phase = jnp.exp(-1j * angular_frequency * time)
        observation = observer.update(
            jnp.asarray(time),
            electric_mode[:, 0] * phase,
            magnetic_mode[:, 0] * phase,
            observation,
        )
    np.testing.assert_allclose(observer.value(observation), 1.0, atol=1e-12)


def test_refresh_rejects_changed_constitutive_state_shape():
    bridge = _bridge((2, 2))
    one_pole = phx.solver.maxwell.LorentzDrudeMaxwellConstitutivePlan(
        jnp.asarray([1.0]),
        jnp.asarray([0.1]),
        jnp.asarray([0.5]),
    )
    two_poles = phx.solver.maxwell.LorentzDrudeMaxwellConstitutivePlan(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0.1, 0.2]),
        jnp.asarray([0.5, 0.25]),
    )
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        constitutive=one_pole,
    ).prepare()
    changed = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        constitutive=two_poles,
    )
    spec = phx.solver.maxwell.CompatibleMaxwellRefreshSpec(
        changed,
        jnp.asarray(0.01),
        "float64",
    )
    with pytest.raises(ValueError, match="executable step signature"):
        phx.solver.maxwell.refresh_compatible_maxwell(runtime, spec)


def test_refresh_rejects_changed_prepared_source_shape():
    bridge = _bridge((2, 2))
    one_entry = phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
        jnp.asarray([0]),
        jnp.asarray([1.0]),
    )
    two_entries = phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
        jnp.asarray([0, 1]),
        jnp.asarray([1.0, 1.0]),
    )
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        sources=(one_entry,),
    ).prepare()
    changed = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        sources=(two_entries,),
    )
    spec = phx.solver.maxwell.CompatibleMaxwellRefreshSpec(
        changed,
        jnp.asarray(0.01),
        "float64",
    )
    with pytest.raises(ValueError, match="executable step signature"):
        phx.solver.maxwell.refresh_compatible_maxwell(runtime, spec)


def test_refresh_rejects_changed_static_boundary_and_source_semantics():
    bridge = _bridge((2, 2))
    pec_runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        boundaries=(phx.solver.maxwell.MaxwellBoundaryPlan("pec"),),
    ).prepare()
    pmc_plan = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        boundaries=(phx.solver.maxwell.MaxwellBoundaryPlan("pmc"),),
    )
    with pytest.raises(ValueError, match="executable step signature"):
        phx.solver.maxwell.refresh_compatible_maxwell(
            pec_runtime,
            phx.solver.maxwell.CompatibleMaxwellRefreshSpec(
                pmc_plan,
                jnp.asarray(0.01),
                "float64",
            ),
        )

    uncontrolled = phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
        jnp.asarray([0]),
        jnp.asarray([1.0]),
    )
    controlled = phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
        jnp.asarray([0]),
        jnp.asarray([1.0]),
        control_key="drive",
    )
    uncontrolled_runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        sources=(uncontrolled,),
    ).prepare()
    controlled_plan = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        sources=(controlled,),
    )
    with pytest.raises(ValueError, match="executable step signature"):
        phx.solver.maxwell.refresh_compatible_maxwell(
            uncontrolled_runtime,
            phx.solver.maxwell.CompatibleMaxwellRefreshSpec(
                controlled_plan,
                jnp.asarray(0.01),
                "float64",
            ),
        )

    class Envelope:
        def __call__(self, time, args):
            del args
            return jnp.cos(time)

    first_envelope = phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
        jnp.asarray([0]),
        jnp.asarray([1.0]),
        envelope=Envelope(),
    )
    second_envelope = phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
        jnp.asarray([0]),
        jnp.asarray([1.0]),
        envelope=Envelope(),
    )
    envelope_runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        sources=(first_envelope,),
    ).prepare()
    envelope_plan = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        sources=(second_envelope,),
    )
    with pytest.raises(ValueError, match="executable step signature"):
        phx.solver.maxwell.refresh_compatible_maxwell(
            envelope_runtime,
            phx.solver.maxwell.CompatibleMaxwellRefreshSpec(
                envelope_plan,
                jnp.asarray(0.01),
                "float64",
            ),
        )
