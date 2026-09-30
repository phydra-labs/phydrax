#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization._cell_de_rham import AbstractCellDeRhamComplex
from phydrax.solver._maxwell import (
    AbstractMaxwellFrequencyResponse,
    CompatibleMaxwellState,
    PreparedCompatibleMaxwell,
)
from phydrax.solver._maxwell_frequency import FrequencyMaxwellSolveResult


def _bridge(shape: tuple[int, ...]) -> phx.discretization.StructuredCochainBridge:
    dimension = len(shape)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    return phx.discretization.StructuredCochainBridge(grid)


@jax.jit
def _compiled_leapfrog(
    runtime: PreparedCompatibleMaxwell,
    state: CompatibleMaxwellState,
    step_size: Array,
) -> CompatibleMaxwellState:
    return runtime.leapfrog_step(0.0, state, step_size)


@jax.jit
def _compiled_frequency_response(
    response: AbstractMaxwellFrequencyResponse, electric: Array, flux: Array
) -> tuple[Array, Array]:
    return response.electric_displacement(electric), response.magnetic_field(flux)


def _incidence_matrix(cochain: AbstractCellDeRhamComplex, degree: int) -> np.ndarray:
    incidence = cochain.topology.incidences[degree]
    relation = incidence.relation
    valid = np.asarray(relation.valid)
    matrix = np.zeros(
        (cochain.cell_counts[degree + 1], cochain.cell_counts[degree]),
        dtype=np.float64,
    )
    np.add.at(
        matrix,
        (
            np.asarray(relation.target_indices)[valid],
            np.asarray(relation.source_indices)[valid],
        ),
        np.asarray(incidence.signs)[valid],
    )
    return matrix


def _unit_source_envelope(time: Array, args: object) -> Array:
    del args
    return jnp.ones_like(time)


def test_maxwell_platform_scenario_1() -> None:
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
    bridge = _bridge((2, 2, 2))
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        magnetic_constraint=phx.solver.maxwell.MaxwellMagneticConstraintPolicy("project"),
    ).prepare()
    flux = jnp.cos(jnp.arange(runtime.layout.magnetic_count, dtype=jnp.float64))
    projected = runtime.pack(
        jnp.zeros((runtime.layout.electric_count,), dtype=jnp.float64), flux
    )
    np.testing.assert_allclose(
        bridge.exterior_derivative(2, projected.primary.magnetic_flux),
        0.0,
        atol=1e-10,
    )
    elided = phx.solver.CompatibleMaxwellPlan(
        bridge,
        magnetic_constraint=phx.solver.maxwell.MaxwellMagneticConstraintPolicy("elide"),
    ).prepare()
    assert elided.magnetic_projection_elided
    # A PMC trace overwrites B, so only projection restores d(B) = q_m.
    with pytest.raises(ValueError, match="closedness evidence"):
        phx.solver.CompatibleMaxwellPlan(
            bridge,
            boundaries=(phx.solver.maxwell.MaxwellBoundaryPlan("pmc"),),
            magnetic_constraint=phx.solver.maxwell.MaxwellMagneticConstraintPolicy(
                "elide"
            ),
        ).prepare()
    bridge = _bridge((3, 4))
    tez = phx.solver.CompatibleMaxwellPlan(bridge, polarization="tez").prepare()
    tmz = phx.solver.CompatibleMaxwellPlan(bridge, polarization="tmz").prepare()
    assert tez.layout.electric_degree == 1 and tez.layout.magnetic_degree == 2
    assert tmz.layout.electric_degree == 0 and tmz.layout.magnetic_degree == 1
    assert tmz.initialize().primary.charge.shape == (0,)
    assert tez.magnetic_constraint(tez.initialize()).shape == ()
    scalar = jnp.sin(jnp.arange(tmz.layout.electric_count, dtype="float64"))
    exact_b = -bridge.exterior_derivative(0, scalar)
    np.testing.assert_allclose(bridge.exterior_derivative(1, exact_b), 0.0, atol=1e-14)


def test_maxwell_platform_scenario_2() -> None:
    bridge = _bridge((4, 4, 4))
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        pml=phx.solver.maxwell.MaxwellCPMLPlan(1),
    ).prepare()
    assert runtime.pml is not None
    assert runtime.pml.state_elements < bridge.dimension * sum(runtime.primary_counts[:2])
    state = runtime.pml.initialize(dtype="complex128")
    assert all(
        memory.ndim == 1 for memory in (*state.electric_memory, *state.magnetic_memory)
    )
    coefficients = runtime.pml.bind_coefficients(0.1)
    assert tuple(value.term_id for value in coefficients.electric) == tuple(
        value.term_id for value in runtime.pml.electric_terms
    )
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
    # Edge zero runs from (0,0,0) to (1/2,0,0). Its dual area is
    # (1/4)^2, length 1/2, and J=2, so it carries 1/4 charge per second
    # from the tail dual volume into the head dual volume.
    expected_charge_rate = jnp.zeros_like(rate.charge).at[0].set(-16.0).at[9].set(8.0)
    np.testing.assert_allclose(rate.charge, expected_charge_rate, atol=1e-13)
    unstructured = phx.solver.maxwell.UnstructuredMaxwellPlan(
        bridge.cochain,
        phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(),
        spectral_upper_bound=1000.0,
        courant_factor=0.9,
    ).prepare()
    stepped = unstructured.step(
        0.0,
        unstructured.initialize(),
        1.0e-3,
        electric_current=jnp.real(start.electric_current),
    )
    np.testing.assert_allclose(
        stepped.primary.charge, 1.0e-3 * expected_charge_rate, atol=1e-13
    )
    np.testing.assert_allclose(unstructured.constraints(stepped)[0], 0.0, atol=1e-13)
    cochain = _bridge((2, 2)).cochain
    categories = [np.zeros((count,), dtype=np.int32) for count in cochain.cell_counts]
    incidence = cochain.topology.incidences[0].relation
    valid = np.asarray(incidence.valid, dtype=np.bool_)
    lower = np.asarray(incidence.source_indices)[valid]
    upper = np.asarray(incidence.target_indices)[valid]
    repeated = next(index for index in np.unique(lower) if np.sum(lower == index) > 1)
    positions = np.flatnonzero(lower == repeated)
    categories[1][upper[positions[0]]] = 5
    categories[1][upper[positions[-1]]] = 1

    partition = phx.solver.CochainRatePartition(cochain, categories)

    for degree, relation in enumerate(cochain.topology.incidences):
        active = np.asarray(relation.relation.valid, dtype=np.bool_)
        sources = np.asarray(relation.relation.source_indices)[active]
        targets = np.asarray(relation.relation.target_indices)[active]
        assert np.all(
            np.abs(
                np.asarray(partition.categories[degree])[sources]
                - np.asarray(partition.categories[degree + 1])[targets]
            )
            <= 1
        )


def _cpml_reflection(width: int, target: float, fraction: float) -> float:
    """Round-trip amplitude reflection of a normally incident TMz pulse pair.

    A modulated Gaussian ``E_z`` (λ = 20 cells) splits into two pulses that
    cross the CPML, reflect off its outer wall, and meet again at the center
    after one interior crossing; the interior energy then is ``|r|²`` of the
    initial energy.
    """
    cells, length = 200, 20.0
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(cells),
            phx.discretization.UniformCellAxisSpec(2, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [length, 0.2]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tmz",
        pml=phx.solver.maxwell.MaxwellCPMLPlan((width, 0), target_reflection=target),
    ).prepare()
    x = np.linspace(-0.5 * length, 0.5 * length, cells + 1)[:, None]
    pulse = np.broadcast_to(np.exp(-((x / 1.5) ** 2)) * np.cos(np.pi * x), (cells + 1, 2))
    electric = bridge.pack(0, (jnp.asarray(pulse),))
    state = runtime.initialize(
        electric_displacement=runtime.constitutive.electric_displacement(electric, None)
    )
    dt = fraction * float(runtime.stable_dt)
    steps = round((length - 2.0 * width * length / cells) / dt)
    final = phx.solver.maxwell.solve_compatible_maxwell(
        runtime, state, 0.0, dt, steps
    ).final_state

    def interior(degree: int, values: Any) -> Any:
        components = []
        for value in bridge.unpack(degree, values):
            index = np.indices(value.shape)[0]
            inside = (index >= width) & (index < cells - width)
            components.append(jnp.where(inside, value, 0.0))
        return bridge.pack(degree, tuple(components))

    inner = runtime.initialize(
        electric_displacement=interior(0, final.primary.electric_displacement),
        magnetic_flux=interior(1, final.primary.magnetic_flux),
    )
    return float(np.sqrt(runtime.energy(inner) / runtime.energy(state)))


def test_time_domain_cpml_reflects_below_its_calibrated_target() -> None:
    # σ_max is calibrated so the continuum round trip reflects R = 1e-4. The
    # discrete layer must stay below it, and the reflection must not depend on
    # Δt: memories read off their kick's time level reflected ∝ Δt (1.3e-3 at
    # CFL 0.9, 6.4e-4 at CFL 0.45 before the kicks were centered).
    coarse, fine = (_cpml_reflection(15, 1e-4, value) for value in (0.9, 0.45))
    assert coarse < 1e-4 and fine < 1e-4
    assert abs(coarse - fine) < 0.05 * fine


def test_maxwell_platform_scenario_3() -> None:
    bridge = _bridge((2, 2, 2))
    plan = phx.solver.maxwell.UnstructuredMaxwellPlan(
        bridge.cochain,
        phx.solver.maxwell.ConductiveMaxwellConstitutivePlan(
            electric_conductivity=0.2,
            magnetic_conductivity=0.0,
        ),
        spectral_upper_bound=1000.0,
        courant_factor=0.9,
    )

    with pytest.raises(ValueError, match="instantaneous lossless"):
        plan.prepare()
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
    conductive = phx.solver.maxwell.ConductiveMaxwellConstitutivePlan(
        electric_conductivity=0.2,
        magnetic_conductivity=0.0,
    ).prepare(bridge.cochain, runtime.layout)
    conductive_frequency = phx.solver.maxwell.FrequencyMaxwellOperator(
        bridge.cochain,
        runtime.layout,
        conductive,
        0.4,
    )
    np.testing.assert_allclose(
        conductive_frequency.mv(field) - frequency.mv(field),
        -1j * 0.4 * 0.2 * field,
        atol=1e-13,
    )
    electric_star = bridge.cochain.hodge_diagonal(runtime.layout.electric_degree)
    ledger = conductive_frequency.power_ledger(field, conductive_frequency.mv(field))
    np.testing.assert_allclose(
        ledger.electric_material,
        0.5 * 0.2 * jnp.sum(electric_star * field**2),
        rtol=1e-12,
    )
    magnetic_loss = phx.solver.maxwell.ConductiveMaxwellConstitutivePlan(
        electric_conductivity=0.2,
        magnetic_conductivity=0.1,
    ).prepare(bridge.cochain, runtime.layout)
    lossy_frequency = phx.solver.maxwell.FrequencyMaxwellOperator(
        bridge.cochain,
        runtime.layout,
        magnetic_loss,
        0.4,
    )
    lossy_ledger = lossy_frequency.power_ledger(field, lossy_frequency.mv(field))
    assert lossy_ledger.magnetic_material > 0.0
    assert lossy_ledger.relative_residual < 1e-12
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


def test_frequency_adjoint_retains_failed_primal_and_adjoint_evidence(
    monkeypatch: Any,
) -> None:
    bridge = _bridge((2, 2))
    runtime = phx.solver.CompatibleMaxwellPlan(bridge, polarization="tez").prepare()
    operator = phx.solver.maxwell.FrequencyMaxwellOperator(
        bridge.cochain,
        runtime.layout,
        runtime.constitutive,
        0.4,
    )
    failed = FrequencyMaxwellSolveResult(
        jnp.zeros((operator.size,), dtype=jnp.complex128),
        jnp.asarray(1.0),
        jnp.asarray(False),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(3, dtype=jnp.int32),
        None,
    )
    monkeypatch.setattr(type(operator), "solve", lambda self, source: failed)
    monkeypatch.setattr(type(operator), "adjoint_solve", lambda self, source: failed)

    result = phx.solver.maxwell.frequency_maxwell_adjoint(
        operator,
        jnp.ones((operator.size,)),
        lambda electric: jnp.sum(jnp.real(electric)),
    )

    assert not bool(result.valid)
    assert result.primal_result is failed
    assert result.adjoint_result is failed
    assert int(result.primal_result.status) == 3
    assert int(result.adjoint_result.status) == 3


def test_maxwell_platform_scenario_4() -> None:
    runtime = phx.solver.CompatibleMaxwellPlan(_bridge((2, 2, 2))).prepare()
    expected = np.dtype(np.complex128).itemsize * sum(runtime.primary_counts)
    assert runtime.resource_estimate.logical_primary_bytes == expected
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


def test_refresh_contracts() -> None:
    bridge = _bridge((2, 2))
    one_pole = phx.solver.maxwell.LorentzDrudeMaxwellConstitutivePlan(
        phx.solver.maxwell.MaxwellLorentzPoles(
            jnp.asarray([1.0]),
            jnp.asarray([0.1]),
            jnp.asarray([0.5]),
        )
    )
    two_poles = phx.solver.maxwell.LorentzDrudeMaxwellConstitutivePlan(
        phx.solver.maxwell.MaxwellLorentzPoles(
            jnp.asarray([1.0, 2.0]),
            jnp.asarray([0.1, 0.2]),
            jnp.asarray([0.5, 0.25]),
        )
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


def _cosine_envelope(time: Any, args: Any) -> Any:
    del args
    return jnp.cos(time)


def test_source_envelope_identity_requires_declared_ids_for_opaque_callables() -> None:
    frequency = 2.0
    for envelope in (
        lambda time, args: jnp.cos(time),
        lambda time, args: jnp.cos(frequency * time),
    ):
        with pytest.raises(TypeError, match="Opaque callables"):
            phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
                jnp.asarray([0]), jnp.asarray([1.0]), envelope=envelope
            )
        with pytest.raises(TypeError, match="Opaque callables"):
            phx.solver.maxwell.MaxwellPairedCurrentSourcePlan(
                jnp.asarray([0]),
                jnp.asarray([1.0]),
                jnp.asarray([0]),
                jnp.asarray([1.0]),
                envelope=envelope,
                envelope_semantic_id="cosine-envelope",
            )
    with pytest.raises(ValueError, match="require an envelope"):
        phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
            jnp.asarray([0]),
            jnp.asarray([1.0]),
            envelope_semantic_id="cosine-envelope",
            envelope_numeric_id="unit-frequency",
        )

    def plan(envelope: Any, **ids: Any) -> Any:
        return phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
            jnp.asarray([0]), jnp.asarray([1.0]), envelope=envelope, **ids
        )

    cosine = plan(
        lambda time, args: jnp.cos(time),
        envelope_semantic_id="cosine-envelope",
        envelope_numeric_id="unit-frequency",
    )
    sine = plan(
        lambda time, args: jnp.sin(time),
        envelope_semantic_id="sine-envelope",
        envelope_numeric_id="unit-frequency",
    )
    retuned = plan(
        lambda time, args: jnp.cos(frequency * time),
        envelope_semantic_id="cosine-envelope",
        envelope_numeric_id="double-frequency",
    )
    plain = plan(_cosine_envelope)
    assert len({cosine.source_id, sine.source_id, retuned.source_id}) == 3
    assert plain.source_id == plan(_cosine_envelope).source_id
    assert plain.source_id != cosine.source_id

    bridge = _bridge((2, 2))

    def maxwell(source: Any) -> Any:
        return phx.solver.CompatibleMaxwellPlan(
            bridge, polarization="tez", sources=(source,)
        )

    def refresh(runtime: Any, source: Any) -> Any:
        return phx.solver.maxwell.refresh_compatible_maxwell(
            runtime,
            phx.solver.maxwell.CompatibleMaxwellRefreshSpec(
                maxwell(source), jnp.asarray(0.01), "float64"
            ),
        )

    cosine_runtime = maxwell(cosine).prepare()
    with pytest.raises(ValueError, match="executable step signature"):
        refresh(cosine_runtime, sine)
    refreshed = refresh(cosine_runtime, retuned)
    state = refreshed.runtime.initialize()
    stepped = refreshed.step(jnp.asarray(0.25), state)
    expected = maxwell(retuned).prepare().leapfrog_step(0.25, state, 0.01)
    np.testing.assert_allclose(
        np.asarray(stepped.primary.electric_displacement),
        np.asarray(expected.primary.electric_displacement),
    )
    plain_runtime = maxwell(plain).prepare()
    refresh(plain_runtime, plan(_cosine_envelope))
    with pytest.raises(ValueError, match="executable step signature"):
        refresh(plain_runtime, cosine)


@pytest.mark.parametrize("complex_fields", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("structured_bridge", [False, True], ids=["cell", "stencil"])
def test_generic_cell_maxwell_preserves_gauss_and_closedness(
    complex_fields: bool, structured_bridge: bool
) -> None:
    bridge = _bridge((2, 2, 2))
    cochain = bridge.cochain
    electric = jnp.sin(jnp.arange(cochain.cell_counts[1], dtype=jnp.float64))
    current = jnp.cos(jnp.arange(cochain.cell_counts[1], dtype=jnp.float64))
    if complex_fields:
        electric = electric.astype(jnp.complex128) * (1.0 + 0.5j)
        current = current.astype(jnp.complex128) * (0.3 - 0.7j)
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge if structured_bridge else cochain,
        constitutive=phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(
            permittivity=2.0, permeability=3.0
        ),
        sources=(
            phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
                jnp.arange(cochain.cell_counts[1], dtype=jnp.int32),
                current,
                envelope=_unit_source_envelope,
            ),
        ),
    ).prepare()
    flux = cochain.exterior_derivative(1, electric, boundary="absolute")
    displacement = 2.0 * electric
    charge = -cochain.codifferential(1, displacement, boundary="absolute")
    state = runtime.pack(displacement, flux, charge)
    expected_energy = 0.5 * jnp.real(
        jnp.vdot(electric, cochain.hodge_diagonal(1) * displacement)
        + jnp.vdot(flux / 3.0, cochain.hodge_diagonal(2) * flux)
    )
    np.testing.assert_allclose(runtime.energy(state), expected_energy, rtol=1e-13)
    dt = 0.1 * runtime.stable_dt
    stepped = _compiled_leapfrog(runtime, state, dt)
    gradient = _incidence_matrix(cochain, 0)
    curl = _incidence_matrix(cochain, 1)
    weights = tuple(np.asarray(cochain.hodge_diagonal(k)) for k in range(3))
    magnetic_half = np.asarray(flux) - 0.5 * float(dt) * (curl @ np.asarray(electric))
    expected_displacement = np.asarray(displacement) + float(dt) * (
        curl.T @ (weights[2] * magnetic_half / 3.0) / weights[1] - np.asarray(current)
    )
    expected_flux = magnetic_half - 0.5 * float(dt) * (
        curl @ (expected_displacement / 2.0)
    )
    expected_charge = np.asarray(charge) + float(dt) * (
        gradient.T @ (weights[1] * np.asarray(current)) / weights[0]
    )
    np.testing.assert_allclose(
        stepped.primary.electric_displacement, expected_displacement, atol=1e-12
    )
    np.testing.assert_allclose(stepped.primary.magnetic_flux, expected_flux, atol=1e-12)
    np.testing.assert_allclose(stepped.primary.charge, expected_charge, atol=1e-12)
    assert stepped.primary.electric_displacement.dtype == electric.dtype
    assert stepped.primary.magnetic_flux.dtype == electric.dtype
    assert stepped.primary.charge.dtype == electric.dtype
    np.testing.assert_allclose(runtime.magnetic_constraint(stepped), 0.0, atol=1e-12)
    np.testing.assert_allclose(runtime.electric_constraint(stepped), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        cochain.exterior_derivative(
            2, stepped.primary.magnetic_flux, boundary="absolute"
        ),
        0.0,
        atol=1e-12,
    )
    assert runtime.layout.electric_form_type == phx.exterior.FormType(3, 1)
    assert runtime.layout.displacement_form_type == phx.exterior.FormType(
        3, 2, twist="twisted"
    )
    assert runtime.layout.magnetic_form_type == phx.exterior.FormType(3, 2)
    assert runtime.layout.magnetic_field_form_type == phx.exterior.FormType(
        3, 1, twist="twisted"
    )


def test_generic_cell_maxwell_refuses_structured_cpml() -> None:
    cochain = _bridge((2, 2, 2)).cochain
    with pytest.raises(ValueError, match="CPML requires"):
        phx.solver.CompatibleMaxwellPlan(
            cochain, pml=phx.solver.maxwell.MaxwellCPMLPlan(1)
        ).prepare()


def _whitney_tetrahedron() -> phx.discretization.FiniteElementDeRhamComplex:
    coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "tetrahedra", "tetrahedron", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    return phx.discretization.FiniteElementDeRhamComplex(mesh, family="trimmed", order=1)


def test_whitney_top_mass_returns_constant_density() -> None:
    complex_ = _whitney_tetrahedron()
    # The degree-three DOF is integrated content, not point density.
    volume = 1.0 / 6.0
    density = 2.75
    np.testing.assert_allclose(
        complex_.hodge_star(3, jnp.asarray((volume * density,))),
        (density,),
        rtol=1e-12,
        atol=1e-12,
    )


def test_unstructured_automatic_cfl_bounds_closed_form_whitney_pencil() -> None:
    complex_ = _whitney_tetrahedron()
    gradients = np.asarray(
        ((-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    moments = (np.ones((4, 4), dtype=np.float64) + np.eye(4, dtype=np.float64)) / 120.0
    mass = np.asarray(
        [
            [
                moments[i, a] * gradients[j] @ gradients[b]
                - moments[i, b] * gradients[j] @ gradients[a]
                - moments[j, a] * gradients[i] @ gradients[b]
                + moments[j, b] * gradients[i] @ gradients[a]
                for a, b in edges
            ]
            for i, j in edges
        ]
    )
    curls = np.asarray([2.0 * np.cross(gradients[i], gradients[j]) for i, j in edges])
    stiffness = curls @ curls.T / 6.0
    factor = np.linalg.cholesky(mass)
    left = np.linalg.solve(factor, stiffness)
    symmetric = np.linalg.solve(factor, left.T).T
    exact_largest = np.linalg.eigvalsh(symmetric)[-1]
    runtime = phx.solver.maxwell.UnstructuredMaxwellPlan(
        complex_,
        phx.solver.maxwell.DiagonalMaxwellConstitutivePlan(),
        courant_factor=0.9,
    ).prepare()
    assert 0.0 < float(runtime.stable_dt) <= 1.8 / np.sqrt(exact_largest)
    state = runtime.initialize()
    current = jnp.linspace(-0.2, 0.3, complex_.cell_counts[1], dtype=jnp.float64)
    stepped = jax.jit(
        lambda carry: runtime.step(
            0.0, carry, 0.1 * runtime.stable_dt, electric_current=current
        )
    )(state)
    np.testing.assert_allclose(runtime.constraints(stepped)[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(runtime.constraints(stepped)[1], 0.0, atol=1e-10)


def test_fe_anisotropic_material_maps_physical_fields_without_metric_weighting() -> None:
    complex_ = _whitney_tetrahedron()
    points = np.asarray(complex_.mesh.coordinates)
    connectivity = phx.discretization.tetrahedral_connectivity(
        np.asarray(((0, 1, 2, 3),), dtype=np.int32), 4
    )
    edges = np.asarray(connectivity.edges)
    faces = np.asarray(connectivity.faces)
    tangents = points[edges[:, 1]] - points[edges[:, 0]]
    area_vectors = 0.5 * np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    epsilon = np.diag(np.asarray((2.0, 3.0, 4.0)))
    inverse_mu = np.diag(np.asarray((0.5, 0.25, 0.125)))
    electric_vector = np.asarray((0.3, -0.7, 1.1))
    magnetic_vector = np.asarray((-0.4, 0.9, 0.25))
    electric = jnp.asarray(tangents @ electric_vector)
    magnetic_flux = jnp.asarray(area_vectors @ magnetic_vector)
    material = phx.solver.maxwell.FiniteElementMaxwellConstitutivePlan(
        permittivity=epsilon, inverse_permeability=inverse_mu
    ).prepare(complex_, phx.solver.maxwell.MaxwellCochainLayout(complex_))
    displacement = material.electric_displacement(electric, None)
    magnetic = material.magnetic_field(magnetic_flux, None)
    np.testing.assert_allclose(
        displacement, tangents @ (epsilon @ electric_vector), rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        magnetic, area_vectors @ (inverse_mu @ magnetic_vector), rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        material.electric_field(displacement, None), electric, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        material.magnetic_flux(magnetic, None), magnetic_flux, rtol=1e-9, atol=1e-10
    )
    hilbert = complex_.hilbert_complex()
    # ∫ E·εE/2 + B·μ⁻¹B/2 over the unit tetrahedron.
    expected_energy = (
        electric_vector @ epsilon @ electric_vector
        + magnetic_vector @ inverse_mu @ magnetic_vector
    ) / 12.0
    np.testing.assert_allclose(
        material.energy(
            displacement, magnetic_flux, None, hilbert.space(1), hilbert.space(2)
        ),
        expected_energy,
        rtol=1e-9,
        atol=1e-10,
    )


def test_cochain_halo_sends_only_incidence_crossings() -> None:
    topology = phx.discretization.simplicial_cell_complex(
        (
            np.asarray(((0,), (1,), (2,)), dtype=np.int32),
            np.asarray(((0, 1), (0, 2), (1, 2)), dtype=np.int32),
            np.asarray(((0, 1, 2),), dtype=np.int32),
        )
    )
    cochain = phx.discretization.CochainDiscretization(
        topology,
        tuple(
            phx.discretization.DiagonalHodge(np.ones(count, dtype=np.float64))
            for count in (3, 3, 1)
        ),
    )
    owners = (
        np.asarray((0, 1, 0), dtype=np.int32),
        np.asarray((0, 0, 1), dtype=np.int32),
        np.asarray((0,), dtype=np.int32),
    )
    partition = phx.discretization.CochainPartition(owners, 2)
    exchange = phx.discretization.CochainHaloExchange(cochain, partition, 0)
    values = jnp.asarray((10.0, 20.0, 30.0))
    target_partitions, payload = exchange.payload(values)
    messages = sorted(
        zip(
            np.asarray(target_partitions).tolist(),
            np.asarray(payload).tolist(),
            strict=True,
        )
    )
    # Vertex 1 is needed by edge 01 on owner 0; vertex 2 by edge 12 on owner 1.
    assert messages == [(0, 20.0), (1, 30.0)]
    with pytest.raises(ValueError, match="every cell coordinate"):
        malformed = phx.discretization.CochainPartition((owners[0][:-1], *owners[1:]), 2)
        phx.discretization.CochainHaloExchange(cochain, malformed, 0)


def test_fe_material_frequency_response_preserves_complex_polarization() -> None:
    complex_ = _whitney_tetrahedron()
    epsilon = np.diag(np.asarray((2.0, 3.0, 4.0)))
    mu = np.diag(np.asarray((3.0, 5.0, 7.0)))
    material = phx.solver.maxwell.FiniteElementMaxwellConstitutivePlan(
        permittivity=epsilon, inverse_permeability=np.linalg.inv(mu)
    ).prepare(complex_, phx.solver.maxwell.MaxwellCochainLayout(complex_))
    response = material.frequency_response(2.0)
    connectivity = phx.discretization.tetrahedral_connectivity(
        np.asarray(((0, 1, 2, 3),), dtype=np.int32), 4
    )
    points = np.asarray(complex_.mesh.coordinates)
    edges = np.asarray(connectivity.edges)
    tangents = points[edges[:, 1]] - points[edges[:, 0]]
    vector = np.asarray((0.3 + 0.2j, -0.7 + 0.9j, 1.1 - 0.4j), dtype=np.complex128)
    electric = jnp.asarray(tangents @ vector)
    expected = tangents @ (epsilon @ vector)
    faces = np.asarray(connectivity.faces)
    areas = 0.5 * np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    magnetic_vector = np.asarray((0.6 - 0.8j, -0.2 + 0.4j, 0.7 + 0.1j))
    flux = jnp.asarray(areas @ magnetic_vector)
    displacement, magnetic = _compiled_frequency_response(response, electric, flux)
    np.testing.assert_allclose(displacement, expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        magnetic, areas @ np.linalg.solve(mu, magnetic_vector), rtol=1e-9, atol=1e-10
    )
    assert displacement.dtype == jnp.complex128
    assert magnetic.dtype == jnp.complex128


def test_compiled_dispersive_frequency_response_matches_complex_transfer_functions() -> (
    None
):
    bridge = _bridge((2, 2, 2))
    layout = phx.solver.maxwell.MaxwellCochainLayout(bridge)
    material = phx.solver.maxwell.LorentzDrudeMaxwellConstitutivePlan(
        phx.solver.maxwell.MaxwellLorentzPoles([1.2, 0.0], [0.4, 0.6], [0.9, 0.5]),
        permittivity_infinity=2.0,
        permeability_infinity=3.0,
        magnetic_poles=phx.solver.maxwell.MaxwellLorentzPoles(
            [1.5, 0.0], [0.2, 0.3], [0.4, 0.7]
        ),
    ).prepare(bridge.cochain, layout)
    omega = 1.1
    electric = jnp.sin(
        jnp.arange(layout.electric_count, dtype=jnp.float64)
    ) + 1j * jnp.cos(jnp.arange(layout.electric_count, dtype=jnp.float64))
    flux = jnp.cos(jnp.arange(layout.magnetic_count, dtype=jnp.float64)) - 0.5j * jnp.sin(
        jnp.arange(layout.magnetic_count, dtype=jnp.float64)
    )
    displacement, magnetic = _compiled_frequency_response(
        material.frequency_response(omega), electric, flux
    )
    epsilon = (
        2.0
        + 0.9 / (1.2**2 - omega**2 - 0.4j * omega)
        + 0.5 / (-(omega**2) - 0.6j * omega)
    )
    mu = (
        3.0
        + 0.4 / (1.5**2 - omega**2 - 0.2j * omega)
        + 0.7 / (-(omega**2) - 0.3j * omega)
    )
    np.testing.assert_allclose(displacement, epsilon * electric, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(magnetic, flux / mu, rtol=1e-12, atol=1e-12)


def test_complex_whitney_curl_and_metric_adjoint_duality() -> None:
    complex_ = _whitney_tetrahedron()
    points = np.asarray(complex_.mesh.coordinates)
    connectivity = phx.discretization.tetrahedral_connectivity(
        np.asarray(((0, 1, 2, 3),), dtype=np.int32), 4
    )
    edges = np.asarray(connectivity.edges)
    faces = np.asarray(connectivity.faces)
    midpoints = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    tangents = points[edges[:, 1]] - points[edges[:, 0]]
    amplitude = 1.0 + 0.3j
    # E = amplitude * (-y/2, x/2, 0) has constant curl amplitude * e_z.
    electric = jnp.asarray(
        amplitude
        * 0.5
        * (-midpoints[:, 1] * tangents[:, 0] + midpoints[:, 0] * tangents[:, 1])
    )
    areas = 0.5 * np.cross(
        points[faces[:, 1]] - points[faces[:, 0]],
        points[faces[:, 2]] - points[faces[:, 0]],
    )
    curl = jax.jit(lambda value: complex_.exterior_derivative(1, value))(electric)
    np.testing.assert_allclose(curl, amplitude * areas[:, 2], atol=1e-12)
    magnetic = jnp.asarray((0.2 + 0.1j, -0.4 + 0.3j, 0.7 - 0.2j, 0.5 + 0.8j))
    adjoint = complex_.codifferential(2, magnetic)
    np.testing.assert_allclose(
        jnp.vdot(curl, complex_.hodge_star(2, magnetic)),
        jnp.vdot(electric, complex_.hodge_star(1, adjoint)),
        rtol=1e-9,
        atol=1e-10,
    )
