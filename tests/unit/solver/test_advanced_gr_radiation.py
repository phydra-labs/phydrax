#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._physical import RelativityScaleContract
from phydrax.applications.numerical_relativity._grrmhd_coupling import (
    GRRMHDCouplingArguments,
    GRRMHDZ4cStageAdapter,
)
from phydrax.applications.numerical_relativity._matter_coupling import (
    CoupledStageAddress,
)
from phydrax.applications.numerical_relativity._production import (
    FixedGridGRRMHDProductionMethod,
    GRRMHDProductionArguments,
)
from phydrax.discretization.finite_volume._structured import FiniteVolumePlan
from phydrax.equations._force_free import GRForceFreeSystem
from phydrax.equations._relativistic_eos import GammaLawEOS
from phydrax.equations._relativistic_mhd import IdealValenciaGRMHDSystem
from phydrax.equations._relativistic_multigroup_radiation import (
    GRMultigroupM1RadiationSystem,
    GRMultigroupRadiationInteractionPlan,
)
from phydrax.equations._relativistic_neutrino import (
    GRNeutrinoInteractionPlan,
    GRNeutrinoM1System,
)
from phydrax.equations._relativistic_radiation import GRGrayM1RadiationSystem
from phydrax.equations._relativistic_radiation_interaction import (
    ConstantGRGrayOpacityPlan,
    GRGrayRadiationInteractionPlan,
)
from phydrax.equations._resistive_grmhd import ResistiveGRMHDOhmicClosure
from phydrax.metrix._adm_exchange import ADMGridGeometry
from phydrax.metrix._spacetime_conventions import RelativityConvention
from phydrax.solver._gr_m1_finite_volume import (
    FixedGridGRM1SSPRK3Plan,
    GRM1BoundaryCondition,
)
from phydrax.solver._gr_multigroup_radiation import (
    FixedGridGRMultigroupM1SSPRK3Plan,
)
from phydrax.solver._gr_neutrino import FixedGridGRNeutrinoM1Plan
from phydrax.solver._gr_polarized_radiation_feedback import (
    GRPolarizedRadiationFeedbackPlan,
    polarized_propagation_matrix,
)
from phydrax.solver._grmhd_ct import GRMHDConstrainedTransportPlan
from phydrax.solver._grmhd_force_free_transition import (
    GRMHDForceFreeTransitionPlan,
)
from phydrax.solver._grmhd_runtime import GRMHDSSPRK3Plan
from phydrax.solver._grrmhd_runtime import FixedGridGRRMHDIMEXPlan
from phydrax.solver._grrmhd_source import GRRMHDImplicitSourcePlan
from phydrax.solver._relativistic_finite_volume import (
    lower_valencia_stage_geometry,
)
from phydrax.solver._resistive_grrmhd_runtime import (
    FixedGridResistiveGRRMHDIMEXPlan,
)
from phydrax.units import KILOGRAM


def _contracts():
    return RelativityScaleContract.geometric(KILOGRAM), RelativityConvention.canonical()


def _geometry(scale, convention, shape=(), *, topology_id="single-cell"):
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    return ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype="bool"),
        jnp.ones(shape, dtype="bool"),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id=topology_id,
        geometry_lineage_id="flat",
    )


def test_periodic_gr_m1_preserves_uniform_stream_and_closes_balance_ledger():
    scale, convention = _contracts()
    system = GRGrayM1RadiationSystem(scale, convention)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    discretization = FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    runtime = FixedGridGRM1SSPRK3Plan(system, discretization)
    geometry = _geometry(
        scale, convention, tuple(grid.shape), topology_id=grid.topology.topology_id
    )
    step = 1.0e-3
    stages = (
        lower_valencia_stage_geometry(discretization, geometry, 0.0),
        lower_valencia_stage_geometry(discretization, geometry, step),
        lower_valencia_stage_geometry(discretization, geometry, 0.5 * step),
    )
    moments = jnp.broadcast_to(jnp.asarray((2.0, 0.25, 0.0, 0.0)), (8, 4))
    state = runtime.initialize(moments, stages[0])
    result = runtime.advance(state, 0.0, step, stages)

    assert bool(result.accepted)
    np.testing.assert_allclose(result.state.radiation_state, state.radiation_state)
    np.testing.assert_allclose(result.attempted_ledger.balance_defect, 0.0, atol=1.0e-7)
    assert bool(result.attempted_ledger.qualified)


def test_m1_vacuum_boundary_removes_incoming_flux_and_reflective_boundary_flips_it():
    scale, convention = _contracts()
    system = GRGrayM1RadiationSystem(scale, convention)
    geometry = _geometry(scale, convention)
    incoming = jnp.asarray((1.0, 0.5, 0.0, 0.0))
    outgoing = jnp.asarray((1.0, -0.5, 0.0, 0.0))

    vacuum = GRM1BoundaryCondition("vacuum")
    reflected = GRM1BoundaryCondition("reflective")
    clipped, clipped_valid = vacuum.exterior(system, incoming, geometry, 0, "lower")
    transmitted, transmitted_valid = vacuum.exterior(
        system, outgoing, geometry, 0, "lower"
    )
    mirrored, mirrored_valid = reflected.exterior(system, incoming, geometry, 0, "lower")

    np.testing.assert_allclose(clipped[1], 0.0)
    np.testing.assert_allclose(transmitted, outgoing)
    np.testing.assert_allclose(mirrored[1], -incoming[1])
    assert bool(clipped_valid & transmitted_valid & mirrored_valid)


def test_force_free_transition_is_hysteretic_and_balances_restoration_energy():
    scale, convention = _contracts()
    geometry = _geometry(scale, convention)
    eos = GammaLawEOS(scale, 4.0 / 3.0, minimum_density=1.0e-12)
    grmhd = IdealValenciaGRMHDSystem(
        eos,
        scale,
        convention=convention,
        maximum_magnetization=1.0e6,
        recovery_iterations=32,
        enthalpy_iterations=32,
    )
    force_free = GRForceFreeSystem(scale, convention)
    plan = GRMHDForceFreeTransitionPlan(
        grmhd,
        force_free,
        enter_magnetization=5.0,
        exit_magnetization=2.0,
    )
    high_primitive = jnp.asarray((1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 3.0))
    high_conserved = grmhd.primitive_to_conserved(high_primitive, geometry)
    state = plan.initialize(high_conserved, geometry)
    entered = plan.transition(state, geometry)

    assert bool(entered.accepted)
    assert bool(entered.state.force_free_mask)
    assert not bool(entered.ledger.restored_grmhd)

    support_primitive = high_primitive.at[7].set(0.2)
    support = grmhd.primitive_to_conserved(support_primitive, geometry)
    restored = plan.transition(entered.state, geometry, material_support=support)

    assert bool(restored.accepted)
    assert not bool(restored.state.force_free_mask)
    assert bool(restored.ledger.restored_grmhd)
    np.testing.assert_allclose(restored.state.grmhd_conserved, support)
    np.testing.assert_allclose(restored.ledger.energy_balance_residual, 0.0)
    assert float(jnp.abs(restored.ledger.reservoir_energy_change)) > 0.0


def test_polarized_absorption_and_faraday_rotation_feed_back_exact_four_momentum():
    scale, convention = _contracts()
    geometry = _geometry(scale, convention)
    plan = GRPolarizedRadiationFeedbackPlan(scale, convention, jnp.asarray((1.0, 1.0)))
    stokes = jnp.asarray(((2.0, 0.4, 0.0, 0.0), (1.0, 0.0, 0.2, 0.0)))
    state = plan.initialize(stokes, jnp.asarray(10.0), jnp.zeros(3), geometry)
    absorption = jnp.asarray(((0.5, 0.0, 0.0, 0.0),) * 2)
    faraday = jnp.asarray(((0.0, 0.0, 1.0),) * 2)
    propagation = polarized_propagation_matrix(absorption, faraday)
    directions = jnp.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    result = plan.advance(
        state,
        0.2,
        jnp.zeros_like(stokes),
        propagation,
        directions,
        geometry,
    )

    assert bool(result.accepted)
    assert float(result.state.matter_energy_density) > float(state.matter_energy_density)
    assert not np.allclose(
        np.asarray(result.state.stokes[..., 1:3]), np.asarray(stokes[..., 1:3])
    )
    np.testing.assert_allclose(result.ledger.energy_balance_residual, 0.0, atol=1.0e-7)
    np.testing.assert_allclose(result.ledger.momentum_balance_residual, 0.0, atol=1.0e-7)
    assert bool(result.stress_energy.valid)


def test_multigroup_and_neutrino_uniform_transport_preserve_all_groups_and_lepton_fraction():
    scale, convention = _contracts()
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    discretization = FiniteVolumePlan(
        grid,
        component_names=(
            "radiation_energy",
            "radiation_flux_x",
            "radiation_flux_y",
            "radiation_flux_z",
        ),
    ).prepare()
    geometry = _geometry(
        scale, convention, tuple(grid.shape), topology_id=grid.topology.topology_id
    )
    step = 1.0e-4
    stages = (
        lower_valencia_stage_geometry(discretization, geometry, 0.0),
        lower_valencia_stage_geometry(discretization, geometry, step),
        lower_valencia_stage_geometry(discretization, geometry, 0.5 * step),
    )
    multigroup = GRMultigroupM1RadiationSystem(
        scale, convention, jnp.asarray((1.0, 2.0, 4.0))
    )

    def transport_for(system):
        return FixedGridGRMultigroupM1SSPRK3Plan(
            system,
            tuple(
                FixedGridGRM1SSPRK3Plan(group, discretization) for group in system.groups
            ),
        )

    multigroup_transport = transport_for(multigroup)
    moments = jnp.zeros((4, 2, 4)).at[..., 0].set(jnp.asarray((1.0, 0.5))[None, :])
    multigroup_state = multigroup_transport.initialize(
        multigroup.flatten_groups(moments), stages[0]
    )
    multigroup_result = multigroup_transport.advance(multigroup_state, 0.0, step, stages)

    assert bool(multigroup_result.accepted)
    np.testing.assert_allclose(
        multigroup_result.state.densitized_moments,
        multigroup_state.densitized_moments,
    )
    identity = jnp.broadcast_to(jnp.eye(3), tuple(grid.shape) + (3, 3))
    scaled_geometry = ADMGridGeometry(
        jnp.ones(tuple(grid.shape)),
        jnp.zeros(tuple(grid.shape) + (3,)),
        4.0 * identity,
        0.25 * identity,
        8.0 * jnp.ones(tuple(grid.shape)),
        jnp.zeros(tuple(grid.shape) + (3, 3)),
        jnp.ones(tuple(grid.shape), dtype="bool"),
        jnp.ones(tuple(grid.shape), dtype="bool"),
        snapshot_token=jnp.asarray(1, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id=grid.topology.topology_id,
        geometry_lineage_id="scaled",
    )
    rejected_stages = (
        lower_valencia_stage_geometry(discretization, geometry, 0.0),
        lower_valencia_stage_geometry(discretization, scaled_geometry, 1.0),
        lower_valencia_stage_geometry(discretization, scaled_geometry, 0.5),
    )
    # A unit interval spans four cell widths, far beyond the M1 stability limit, so
    # the step is rejected on a valid interval.
    rejected = multigroup_transport.advance(
        multigroup_state,
        0.0,
        1.0,
        rejected_stages,
    )
    assert not bool(rejected.accepted)
    assert bool(rejected.stress_energy.compatible_with(rejected_stages[0].cell))
    assert not bool(rejected.stress_energy.compatible_with(rejected_stages[-1].cell))

    neutrinos = GRNeutrinoM1System(scale, convention, jnp.asarray((1.0, 2.0, 4.0)))
    species_transport = tuple(
        transport_for(system) for system in neutrinos.species_systems
    )
    species_interactions = tuple(
        GRMultigroupRadiationInteractionPlan(
            system,
            tuple(
                GRGrayRadiationInteractionPlan(group, ConstantGRGrayOpacityPlan())
                for group in system.groups
            ),
        )
        for system in neutrinos.species_systems
    )
    neutrino_plan = FixedGridGRNeutrinoM1Plan(
        neutrinos,
        species_transport,
        GRNeutrinoInteractionPlan(neutrinos, species_interactions),
    )
    neutrino_moments = jnp.zeros((4, 3, 2, 4)).at[..., 0].set(0.25)
    neutrino_state = neutrino_plan.initialize(
        neutrinos.flatten_moments(neutrino_moments),
        jnp.ones(4),
        0.4 * jnp.ones(4),
        stages[0],
    )
    neutrino_result = neutrino_plan.advance(
        neutrino_state,
        0.0,
        step,
        stages,
        jnp.ones(4),
        jnp.zeros((4, 3)),
        jnp.ones(4),
        jnp.ones(4),
    )

    assert bool(neutrino_result.accepted)
    np.testing.assert_allclose(
        neutrino_result.state.electron_fraction,
        neutrino_state.electron_fraction,
    )
    np.testing.assert_allclose(
        neutrino_result.ledger.lepton_balance_residual, 0.0, atol=1.0e-10
    )


def _coupled_grrmhd_fixture(*, conductivity=0.0):
    scale, convention = _contracts()
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    geometry = _geometry(
        scale, convention, tuple(grid.shape), topology_id=grid.topology.topology_id
    )
    eos = GammaLawEOS(scale, 4.0 / 3.0, minimum_density=1.0e-12)
    material = IdealValenciaGRMHDSystem(
        eos,
        scale,
        convention=convention,
        maximum_magnetization=1.0e6,
        recovery_iterations=32,
        enthalpy_iterations=32,
    )
    ct = GRMHDConstrainedTransportPlan(bridge)
    material_transport = GRMHDSSPRK3Plan(material, ct, cfl=0.2)
    radiation = GRGrayM1RadiationSystem(scale, convention)
    discretization = FiniteVolumePlan(
        grid, component_names=radiation.component_names
    ).prepare()
    radiation_transport = FixedGridGRM1SSPRK3Plan(radiation, discretization, cfl=0.2)
    interaction = GRGrayRadiationInteractionPlan(radiation, ConstantGRGrayOpacityPlan())
    source = GRRMHDImplicitSourcePlan(
        material, interaction, caloric_temperature_scale=1.0
    )
    base = FixedGridGRRMHDIMEXPlan(material_transport, radiation_transport, source)
    resistive = FixedGridResistiveGRRMHDIMEXPlan(
        base,
        ResistiveGRMHDOhmicClosure(scale, convention, conductivity=conductivity),
    )
    primitive = jnp.zeros((4, 8))
    primitive = primitive.at[..., 0].set(1.0)
    primitive = primitive.at[..., 4].set(0.1)
    primitive = primitive.at[..., 5].set(0.2)
    conserved = material.primitive_to_conserved(primitive, geometry)
    magnetic_flux = ct.pack_densitized_face_flux((primitive[..., 5],))
    radiation_moments = jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0, 0.0)), (4, 4))
    stages = (
        lower_valencia_stage_geometry(discretization, geometry, 0.0),
        lower_valencia_stage_geometry(discretization, geometry, 1.0e-5),
    )
    return (
        base,
        resistive,
        conserved,
        radiation_moments,
        magnetic_flux,
        stages,
    )


def test_uniform_grrmhd_imex_step_preserves_equilibrium_and_all_ledgers():
    base, _, conserved, radiation, magnetic_flux, stages = _coupled_grrmhd_fixture()
    state = base.initialize(
        conserved,
        radiation,
        stages[0],
        magnetic_flux=magnetic_flux,
    )
    result = base.advance(state, 0.0, 1.0e-5, stages)

    assert bool(result.accepted)
    np.testing.assert_allclose(
        result.state.material_state, state.material_state, atol=1.0e-7
    )
    np.testing.assert_allclose(
        result.state.radiation_state, state.radiation_state, atol=1.0e-7
    )
    np.testing.assert_allclose(
        result.accepted_ledger.source_energy_defect, 0.0, atol=1.0e-8
    )
    np.testing.assert_allclose(
        result.accepted_ledger.source_momentum_defect, 0.0, atol=1.0e-8
    )


def test_uniform_implicit_four_force_accepts_absorption_and_preserves_total_energy():
    base, _, conserved, radiation, _, stages = _coupled_grrmhd_fixture()
    interaction = GRGrayRadiationInteractionPlan(
        base.radiation_transport.system,
        ConstantGRGrayOpacityPlan(
            planck_absorption=0.5,
            planck_emission=0.0,
            rosseland_transport=0.5,
        ),
    )
    source = GRRMHDImplicitSourcePlan(
        base.material_transport.system,
        interaction,
        caloric_temperature_scale=1.0,
    )
    result = source.advance(conserved, radiation, 0.05, stages[0].cell)

    assert bool(result.accepted)
    assert bool(jnp.all(result.radiation_state[..., 0] < radiation[..., 0]))
    np.testing.assert_allclose(result.ledger.energy_defect, 0.0, atol=1.0e-10)
    np.testing.assert_allclose(result.ledger.momentum_defect, 0.0, atol=1.0e-10)
    assert bool(jnp.all(result.ledger.jacobian_fallback))
    assert not bool(result.derivative_valid)


def test_grrmhd_production_adapter_commits_only_the_accepted_fixed_step():
    base, _, conserved, radiation, magnetic_flux, stages = _coupled_grrmhd_fixture()
    runtime_state = base.initialize(
        conserved,
        radiation,
        stages[0],
        magnetic_flux=magnetic_flux,
    )
    method = FixedGridGRRMHDProductionMethod(base, fixed_step_size=1.0e-5)
    state = method.initialize(runtime_state)
    arguments = GRRMHDProductionArguments(stages, None, 0.0)
    result = method.step(0, 0.0, state, 1.0e-5, arguments)

    assert bool(result.successful)
    assert int(result.accepted_state.runtime_state.accepted_step) == 1
    assert float(result.accepted_state.runtime_state.time) == 1.0e-5


def test_grrmhd_z4c_adapter_combines_material_and_radiation_stress_at_one_stage():
    base, _, conserved, radiation, magnetic_flux, stages = _coupled_grrmhd_fixture()
    state = base.initialize(
        conserved,
        radiation,
        stages[0],
        magnetic_flux=magnetic_flux,
    )
    adapter = GRRMHDZ4cStageAdapter(
        base,
        lambda _address, stage: stage,
        stage_geometry_id="flat-stage",
        topology_id=stages[0].cell.topology_id,
    )
    address = CoupledStageAddress(
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-5),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        topology_id=stages[0].cell.topology_id,
    )
    arguments = GRRMHDCouplingArguments(None, 0.0, stages[0])
    checked = adapter.geometry_for(stages[0].cell, address, arguments)
    projection = adapter.stress_energy_at_stage(state, checked.cell, address, arguments)

    assert checked.stage_geometry_id == stages[0].stage_geometry_id
    assert bool(jnp.all(projection.valid))
    assert bool(jnp.all(projection.energy_density > 0.0))


def test_zero_conductivity_resistive_step_leaves_electric_field_and_charge_unchanged():
    _, resistive, conserved, radiation, magnetic_flux, stages = _coupled_grrmhd_fixture(
        conductivity=0.0
    )
    electric = jnp.broadcast_to(jnp.asarray((0.0, 0.1, 0.0)), (4, 3))
    charge = 0.25 * jnp.ones(4)
    state = resistive.initialize(
        conserved,
        radiation,
        electric,
        charge,
        stages[0],
        magnetic_flux=magnetic_flux,
    )
    result = resistive.advance(state, 0.0, 1.0e-5, stages)

    assert bool(result.accepted)
    np.testing.assert_allclose(result.state.electric_covector, electric)
    np.testing.assert_allclose(result.state.densitized_charge, charge)
    np.testing.assert_allclose(result.ledger.energy_balance_residual, 0.0)
    np.testing.assert_allclose(result.ledger.charge_balance_residual, 0.0)
