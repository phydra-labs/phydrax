from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.cosmology import (
    CosmologicalKDKPlan,
    CosmologyProductProvenance,
    FLRWBackground,
    FLRWGrowthPlan,
    LagrangianPerturbationInitialConditionPlan,
    MatterPowerDescriptor,
    MatterPowerTable,
)
from phydrax.discretization.finite_volume._mhd_boundary import (
    MHDOutflowBoundary,
    PerfectlyConductingWallBoundary,
    PrescribedMHDInflowBoundary,
)
from phydrax.discretization.finite_volume._mhd_closure import (
    ConstrainedMHDClosurePlan,
    StructurePreservingFaceClosurePlan,
)
from phydrax.discretization.finite_volume._mhd_reconstruction import (
    MHDPrimitiveReconstructionPlan,
)
from phydrax.discretization.finite_volume._uct import HLLUCTElectromotivePlan
from phydrax.equations import MultigroupM1RadiationSystem
from phydrax.equations._glm_mhd import GLMIdealMHDSystem
from phydrax.solver._balance_law_composition import (
    AdditiveIMEXTableau,
    BalanceLawCompositionPlan,
)
from phydrax.solver._distributed_mhd import DegreeAwareEntityOwnership
from phydrax.solver._isolated_gravity import IsolatedCartesianGravityPlan
from phydrax.solver._mapped_mhd import (
    MappedALEConstrainedTransportPlan,
    MappedCochainGeometry,
)
from phydrax.solver._mhd_advanced import (
    DualEnergyMHDPlan,
    LocalMHDPositivityPlan,
    MHDCharacteristicReconstructionPlan,
    MHDCTUPredictorPlan,
)
from phydrax.solver._mhd_amr import (
    ConstrainedMHDAMRSynchronizationPlan,
    ElectromotiveForceRegister,
)
from phydrax.solver._modal_forcing import ModalForcingBasis
from phydrax.solver._multiphysics_inference import (
    FieldObservationPlan,
    SimulationSensitivityReport,
    WhitenedFieldInferencePlan,
)
from phydrax.solver._nonideal_mhd import AnisotropicThermalTransportPlan, NonIdealMHDPlan
from phydrax.solver._radiation import GrayLinearRadiationDiffusionPlan
from phydrax.solver._unstructured_mhd import UnstructuredConstrainedTransportPlan


def _mhd_problem(dimension: int, count: int = 6):
    names = tuple("xyz"[:dimension])
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for _ in range(dimension)
        ),
        axis_names=names,
    ).prepare(jnp.stack((jnp.zeros(dimension), jnp.ones(dimension))))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    system = phx.equations.IdealMHDSystem(dimension)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        f"mhd-{dimension}d",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(names),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLDFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    primitive = jnp.zeros(grid.shape + (8,))
    primitive = primitive.at[..., 0].set(1.0)
    primitive = primitive.at[..., 4].set(1.0)
    primitive = primitive.at[..., 5].set(0.2)
    primitive = primitive.at[..., 6].set(0.1)
    full = system.primitive_to_conserved(primitive)
    normal = tuple(primitive[..., 5 + axis] for axis in range(dimension))
    magnetic = bridge.pack_normal_flux(normal)
    return grid, bridge, system, dynamics, full, magnetic


def test_dimension_generic_mhd_and_accepted_integrals():
    for dimension in (1, 2, 3):
        _, bridge, _, dynamics, full, magnetic = _mhd_problem(dimension)
        spatial = phx.discretization.UpwindConstrainedTransportPlan(dynamics, bridge)
        integrator = phx.solver.ConstrainedMHDSSPRK3Plan(spatial, cfl=0.2)
        state = integrator.initialize(full, magnetic, step_size=1e-4)
        result = integrator.advance(state, 0.0, 1e-4)
        assert bool(result.accepted)
        assert len(result.accepted_integrals.face_flux_integrals) == dimension
        np.testing.assert_allclose(
            spatial.magnetic_constraint(result.state.magnetic_flux),
            spatial.magnetic_constraint(state.magnetic_flux),
            atol=1e-12,
        )
        np.testing.assert_allclose(result.state.cell_state, state.cell_state, atol=1e-10)


def test_mhd_reconstruction_and_hll_uct_constant_state():
    _, bridge, _, dynamics, full, magnetic = _mhd_problem(3)
    for method in ("plm", "weno_z", "teno", "mp5"):
        reconstruction = MHDPrimitiveReconstructionPlan(method)
        spatial = phx.discretization.UpwindConstrainedTransportPlan(
            dynamics,
            bridge,
            reconstruction=reconstruction,
            electromotive_plan=HLLUCTElectromotivePlan(),
        )
        rate = spatial.rate(0.0, spatial.layout.reduce_full_state(full), magnetic)
        assert jnp.all(jnp.isfinite(rate.cell_rate))
        np.testing.assert_allclose(rate.cell_rate, 0.0, atol=1e-10)
        np.testing.assert_allclose(rate.magnetic_rate, 0.0, atol=1e-10)


def test_prepared_thermochemistry_conserves_species_invariant():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("A", "B"),
        (
            phx.equations.ChemicalPhaseKind.GAS,
            phx.equations.ChemicalPhaseKind.GAS,
        ),
        jnp.asarray((1.0, 1.0)),
        ("X",),
        jnp.asarray(((1, 1),), dtype=jnp.int32),
        jnp.asarray((0, 0), dtype=jnp.int32),
        gas_standard_pressure=101325.0,
    )
    thermodynamics = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray((10.0, 10.0)),
        jnp.asarray((0.0, 0.0)),
        reference_temperature=300.0,
        minimum_temperature=200.0,
        maximum_temperature=2000.0,
    )
    mechanism = phx.equations.ChemicalMechanismIR(
        "conversion",
        schema,
        thermodynamics,
        (
            phx.equations.ChemicalReactionSpec(
                "A->B",
                {"A": 1.0},
                {"B": 1.0},
                phx.equations.ArrheniusRatePlan(0.5),
            ),
        ),
    ).prepare()
    fields = mechanism.evaluate(
        jnp.asarray((1.0, 0.0)),
        jnp.asarray(500.0),
        jnp.asarray(101325.0),
    )
    np.testing.assert_allclose(fields.element_residual, 0.0, atol=1e-12)


def test_mhd_boundaries_advanced_integrators_and_nonideal_update():
    _, _, system, dynamics, full, magnetic = _mhd_problem(3)
    interior = full[0]
    normal = jnp.full(interior.shape[:-1], 0.2)
    wall = PerfectlyConductingWallBoundary()
    wall_trace = wall.trace(system, interior, normal, 0, "lower", jnp.asarray(0.0))
    wall_primitive = system.conserved_to_primitive(wall_trace.exterior_state)
    np.testing.assert_allclose(wall_primitive[..., 1], 0.0)
    np.testing.assert_allclose(wall_trace.boundary_electromotive, 0.0)
    outflow = MHDOutflowBoundary()
    assert jnp.all(
        jnp.isfinite(
            outflow.trace(
                system, interior, normal, 0, "upper", jnp.asarray(0.0)
            ).exterior_state
        )
    )
    prescribed = PrescribedMHDInflowBoundary(
        system.conserved_to_primitive(interior[0, 0])
    )
    assert jnp.all(
        jnp.isfinite(
            prescribed.trace(
                system, interior, normal, 0, "lower", jnp.asarray(0.0)
            ).exterior_state
        )
    )

    bridge = phx.discretization.StructuredCochainBridge(dynamics.discretization.grid)
    spatial = phx.discretization.UpwindConstrainedTransportPlan(dynamics, bridge)
    reduced = spatial.layout.reduce_full_state(full)
    predictor = MHDCTUPredictorPlan(spatial)
    predicted_cell, predicted_magnetic, _ = predictor.predict(
        jnp.asarray(0.0), reduced, magnetic, jnp.asarray(1e-4)
    )
    assert jnp.all(jnp.isfinite(predicted_cell))
    positivity = LocalMHDPositivityPlan(spatial)
    limited = positivity.apply(reduced, magnetic, predicted_cell, predicted_magnetic)
    assert bool(limited.successful)
    dual = DualEnergyMHDPlan(system.gamma)
    auxiliary = dual.initialize(full)
    synchronized, _, _ = dual.synchronize(full, auxiliary)
    np.testing.assert_allclose(synchronized, full, atol=1e-10)
    characteristic = MHDCharacteristicReconstructionPlan(
        lambda left, right, axis, args: (
            jnp.broadcast_to(jnp.eye(8), left.shape[:-1] + (8, 8)),
            jnp.broadcast_to(jnp.eye(8), left.shape[:-1] + (8, 8)),
            jnp.ones(left.shape[:-1] + (8,)),
        ),
        declared_id="identity-mhd-eigensystem",
    )
    left_characteristic, right_characteristic, _ = characteristic.project(full, full, 0)
    np.testing.assert_allclose(left_characteristic, full)
    np.testing.assert_allclose(right_characteristic, full)

    integrator = phx.solver.ConstrainedMHDSSPRK3Plan(spatial, cfl=0.2)
    state = integrator.initialize(full, magnetic, step_size=1e-4)
    nonideal = NonIdealMHDPlan(spatial, resistivity=1e-3)
    advanced, report = nonideal.advance(state, 1e-4)
    assert bool(report.successful)
    np.testing.assert_allclose(advanced.magnetic_flux, state.magnetic_flux, atol=1e-10)


def test_modal_basis_multirate_and_amr_topology_contracts():
    basis = ModalForcingBasis(
        jnp.asarray([[[1.0]], [[-1.0]]]),
        weights=jnp.asarray([1.0, 0.5]),
    )
    evaluated = basis.evaluate(jnp.asarray([1.0, 2.0]))
    np.testing.assert_allclose(evaluated, 0.0)
    composition = BalanceLawCompositionPlan(
        (1, 4),
        integration_modes=("explicit", "implicit"),
    )
    assert composition.process_subcycles == (1, 4)

    from phydrax.solver._amr_multiphysics import (
        AMRTopologyEpoch,
        AMRTopologyReplayPlan,
    )

    first = AMRTopologyEpoch(
        jnp.asarray([True, False]),
        jnp.asarray([-1, -1]),
        jnp.asarray([0, 0]),
    )
    second = AMRTopologyEpoch(
        jnp.asarray([True, True]),
        jnp.asarray([-1, 0]),
        jnp.asarray([0, 1]),
    )
    replay = AMRTopologyReplayPlan((first, second), (0, 3))
    assert replay.epoch(2).epoch_id == first.epoch_id
    assert replay.epoch(3).epoch_id == second.epoch_id


def test_bounded_one_dimensional_mhd_runtime():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(6, periodic=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    system = phx.equations.IdealMHDSystem(1)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    generic_pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    problem = phx.equations.ConservationProblemIR(
        "bounded-mhd",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet(("x",), (generic_pair,)),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLDFluxPlan(),
        ),
    ).dynamics
    primitive = jnp.zeros((6, 8))
    primitive = primitive.at[:, 0].set(1.0)
    primitive = primitive.at[:, 4].set(1.0)
    primitive = primitive.at[:, 5].set(0.2)
    full = system.primitive_to_conserved(primitive)
    normal_shape = bridge.orientation_shapes[0][0]
    magnetic = bridge.pack_normal_flux((jnp.full(normal_shape, 0.2),))
    wall = PerfectlyConductingWallBoundary()
    boundary_set = phx.solver.advanced.ConstrainedMHDBoundarySet(
        ("x",), {"x": (wall, wall)}
    )
    spatial = phx.discretization.UpwindConstrainedTransportPlan(
        dynamics,
        bridge,
        boundary_set=boundary_set,
    )
    integrator = phx.solver.ConstrainedMHDSSPRK3Plan(spatial, cfl=0.2)
    state = integrator.initialize(full, magnetic, step_size=1e-4)
    result = integrator.advance(state, 0.0, 1e-4)
    assert bool(result.accepted)
    np.testing.assert_allclose(result.state.cell_state, state.cell_state, atol=1e-10)
    np.testing.assert_allclose(
        result.state.magnetic_flux, state.magnetic_flux, atol=1e-10
    )


def test_isolated_gravity_anisotropic_transport_and_imex():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    gravity = IsolatedCartesianGravityPlan(grid, softening=0.05)
    density = jnp.exp(-100.0 * (grid.structured_axes[0].interval_centers - 0.5) ** 2)
    potential, acceleration, evidence = gravity.solve(density)
    assert bool(evidence.finite)
    assert jnp.all(jnp.isfinite(potential))
    assert jnp.all(jnp.isfinite(acceleration))

    conduction = AnisotropicThermalTransportPlan(0.1)
    temperature = 1.0 + 0.01 * jnp.sin(
        2.0 * jnp.pi * grid.structured_axes[0].interval_centers
    )
    material = jnp.ones_like(temperature)
    magnetic = jnp.ones(temperature.shape + (1,))
    advanced, report = conduction.advance(
        temperature, material, magnetic, jnp.asarray(1e-4), (1.0 / 8.0,)
    )
    assert bool(report.successful)
    assert jnp.all(advanced > 0.0)

    tableau = AdditiveIMEXTableau(
        jnp.asarray([[0.0]]),
        jnp.asarray([[1.0]]),
        jnp.asarray([1.0]),
        jnp.asarray([1.0]),
    )
    stepped = tableau.step(
        jnp.asarray([1.0]),
        jnp.asarray(0.0),
        jnp.asarray(0.1),
        lambda state, time, args: jnp.zeros_like(state),
        lambda provisional, time, diagonal_step, args: (
            provisional / (1.0 + diagonal_step)
        ),
    )
    np.testing.assert_allclose(stepped, 1.0 / 1.1, rtol=1e-6)


def test_bounded_gravity_and_conservative_energy_coupling():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem(1)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    problem = phx.equations.ConservationProblemIR(
        "bounded-gravity",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.2, maximum_retries=0),
    )
    transport = phx.solver.prepare_balance_law_transport(runtime)
    gravity = phx.solver.NewtonianSelfGravityPlan(
        0.1,
        boundaries={"x": ("dirichlet", "dirichlet")},
    ).prepare(transport)
    density = 1.0 + 0.1 * jnp.sin(2.0 * jnp.pi * grid.structured_axes[0].interval_centers)
    potential, _, acceleration, solved = gravity.solve_density(density)
    assert bool(solved.converged)
    assert jnp.all(jnp.isfinite(potential))
    assert jnp.all(jnp.isfinite(acceleration))

    primitive = jnp.stack(
        (density, jnp.zeros_like(density), jnp.ones_like(density)), axis=-1
    )
    average = system.primitive_to_conserved(primitive).reshape((8, 3))
    context = phx.solver.advanced.BalanceLawAcceptedStepContext(
        start_time=jnp.asarray(0.0),
        end_time=jnp.asarray(1e-4),
        incoming_cell_average=average,
        provisional_cell_average=average,
        accepted_integrals=None,
        transport_id=transport.transport_id,
    )
    coupling = phx.solver.advanced.ConservativeGravityEnergyCoupling(gravity)
    corrected = coupling.apply(context)
    assert bool(corrected.successful)
    np.testing.assert_allclose(corrected.cell_average, average, atol=1e-10)


def test_exact_cooling_coordinate_round_trip():
    curve = phx.equations.TabulatedCoolingCurve(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([-2.0, -1.0, 1.0]),
        bounds_policy="power_law_extrapolate",
    )
    temperature = jnp.asarray([1.5, 8.0, 40.0])
    coordinate = curve.cooling_coordinate(temperature)
    recovered = curve.temperature_from_cooling_coordinate(coordinate)
    np.testing.assert_allclose(recovered, temperature, rtol=1e-6)


def test_radiation_moments_and_gray_exchange():
    system = MultigroupM1RadiationSystem(2, 2)
    state = jnp.zeros((4, system.group_count * system.group_width))
    state = state.at[:, 0].set(1.0)
    state = state.at[:, system.group_width].set(2.0)
    assert jnp.all(system.admissible(state))
    assert jnp.all(jnp.isfinite(system.physical_flux(state, 0)))

    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    plan = GrayLinearRadiationDiffusionPlan(
        grid,
        transport_extinction=2.0,
        absorption_coefficient=2.0,
    )
    initial = plan.initialize(jnp.ones((4,)), 2.0 * jnp.ones((4,)))
    advanced, diagnostics = plan.advance(initial, 1e-3, 1.2)
    assert bool(diagnostics.successful)
    assert jnp.all(advanced.radiation_energy > 0.0)
    np.testing.assert_allclose(diagnostics.combined_energy_defect, 0.0, atol=1e-12)


def test_glm_unstructured_mapped_and_distributed_cochains():
    glm = GLMIdealMHDSystem(2)
    primitive = jnp.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.2, 0.0, 0.0, 0.0])
    state = glm.primitive_to_conserved(primitive)
    assert bool(glm.admissible(state))

    _, bridge, _, _, _, magnetic = _mhd_problem(3, count=2)
    geometry = MappedCochainGeometry(
        tuple(
            jnp.broadcast_to(jnp.eye(3)[axis], shape + (3,))
            for axis, shape in enumerate(bridge.orientation_shapes[2])
        ),
        tuple(
            jnp.broadcast_to(jnp.eye(3)[axis], shape + (3,))
            for axis, shape in enumerate(bridge.orientation_shapes[1])
        ),
        jnp.ones(bridge.grid.shape),
    )
    mapped = MappedALEConstrainedTransportPlan(bridge, geometry)
    electric = tuple(jnp.zeros(shape + (3,)) for shape in bridge.orientation_shapes[1])
    updated, evidence = mapped.faraday_advance(magnetic, electric, 0.0, 0.1)
    np.testing.assert_array_equal(updated, magnetic)
    assert evidence.constraint_change == 0.0

    unstructured = UnstructuredConstrainedTransportPlan(bridge.cochain, 3)
    unstructured_state = unstructured.initialize(magnetic)
    edge = jnp.zeros((bridge.cochain.cell_counts[1],))
    advanced, report = unstructured.advance(unstructured_state, edge, 0.1)
    np.testing.assert_array_equal(advanced.face_flux, magnetic)
    assert bool(report.successful)

    ownership = DegreeAwareEntityOwnership(bridge.cochain, 2)
    assert ownership.owned_mask(2, 0).shape == magnetic.shape


def test_reflux_curl_preserves_constraint():
    _, bridge, _, _, _, magnetic = _mhd_problem(3, count=2)
    edge_count = bridge.cochain.cell_counts[1]
    register = ElectromotiveForceRegister(
        jnp.zeros((edge_count,)),
        jnp.linspace(0.0, 1e-6, edge_count),
        register_id="test-emf-register",
    )
    plan = ConstrainedMHDAMRSynchronizationPlan(bridge)
    updated, diagnostics = plan.reflux_curl(magnetic, register)
    assert diagnostics.divergence_change < 1e-12
    assert jnp.all(jnp.isfinite(updated))


def test_cosmology_inference_and_closure_contracts():
    background = FLRWBackground(1.0, 0.3)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(1), jnp.ones((1,)), ambient_dimension=1
    ).prepare()
    kdk = CosmologicalKDKPlan(particles, (1.0,))
    state = kdk.initialize(jnp.asarray([[0.25]]), jnp.asarray([[0.0]]), 0.5)
    advanced, diagnostics = kdk.advance(
        background,
        state,
        0.6,
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
    )
    assert bool(diagnostics.successful)
    assert advanced.scale_factor == 0.6

    lpt_particles = phx.discretization.ParticleSetPlan(
        jnp.arange(4), jnp.ones((4,)), ambient_dimension=1
    ).prepare()
    growth = FLRWGrowthPlan(jnp.asarray([0.1, 1.0])).solve(background)
    provenance = CosmologyProductProvenance(
        producer="test",
        producer_version="current",
        model_form_id=background.model_form_id,
        request_id="test-power",
        numerical_policy_id="test-power",
        physics_policy_id="linear-cold-baryon-power",
        scale_id=background.scale.scale_id,
        source_kind="external",
        differentiation="constant",
    )
    power = MatterPowerTable(
        [0.1, 1.0],
        [1.0, 20.0],
        [[1.0e-8, 1.0e-8], [1.0e-6, 1.0e-6]],
        MatterPowerDescriptor("cold_baryon", "cold_baryon", spatial_dimension=1),
        background.scale,
        provenance,
        background.realization,
    )
    lpt = LagrangianPerturbationInitialConditionPlan(lpt_particles, (4,), (1.0,))
    realized = lpt.realize(background, growth, power, jnp.ones((4,)), 0.1)
    assert realized.positions.shape == (4, 1)

    observation = FieldObservationPlan(
        lambda value, args: value,
        jnp.asarray([1.0]),
        phx.observation.CholeskyCovarianceAction(
            jnp.eye(1), phx.observation.CoordinateLayout(("field:0",))
        ),
        observation_id="identity-observation",
    )
    inference = WhitenedFieldInferencePlan(
        lambda field, args: field,
        observation,
        jnp.eye(1),
        plan_id="identity-inference",
    )
    value, gradient = inference.value_and_gradient(jnp.asarray([0.0]))
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    sensitivity = SimulationSensitivityReport.evaluate(
        lambda parameter: parameter**2,
        jnp.asarray([2.0]),
        jnp.asarray([1.0]),
    )
    assert sensitivity.jvp_residual < 1e-3

    face = StructurePreservingFaceClosurePlan(
        lambda left, right, args: jnp.ones(left.shape[:-1]),
        lambda left, right, args: jnp.zeros(left.shape[:-1]),
        closure_id="dissipative-face",
    )
    closure = ConstrainedMHDClosurePlan(
        face,
        lambda left, right, flux, edge, args: jnp.zeros_like(edge),
        closure_id="face-edge",
    )
    left = jnp.ones((2, 8))
    right = 2.0 * left
    corrected, edge, closure_report = closure.apply(
        left,
        right,
        jnp.zeros_like(left),
        jnp.zeros((3,)),
    )
    assert jnp.all(jnp.isfinite(corrected))
    assert jnp.all(edge == 0.0)
    assert not bool(closure_report.fallback_activated)
