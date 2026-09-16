#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _square_mesh():
    return phx.discretization.CellMesh.from_triangles(
        jnp.asarray(
            ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            dtype=jnp.float64,
        ),
        jnp.asarray(((0, 1, 3), (1, 2, 3)), dtype=jnp.int32),
    )


def _two_block_mesh():
    coordinates = jnp.asarray(
        (
            (0.0, 0.0),
            (0.5, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 1.0),
            (1.0, 1.0),
        ),
        dtype=jnp.float64,
    )
    return phx.discretization.CellMesh(
        coordinates,
        (
            phx.discretization.CellBlock(
                "left",
                "triangle",
                jnp.asarray(((0, 1, 3), (1, 4, 3)), dtype=jnp.int32),
                global_ids=jnp.asarray((0, 1)),
            ),
            phx.discretization.CellBlock(
                "right",
                "triangle",
                jnp.asarray(((1, 2, 4), (2, 5, 4)), dtype=jnp.int32),
                global_ids=jnp.asarray((2, 3)),
            ),
        ),
    )


def _binary_model(*, polynomial=False):
    parameters = phx.equations.BinaryThermodynamicParameters(1.0, 1.0)
    if not polynomial:
        return phx.applications.phase_field.BinaryPhaseFieldModel(parameters)
    potential = phx.equations.PolynomialBulkFreeEnergy((0.25, 0.0, -0.5, 0.0, 0.25))
    return phx.applications.phase_field.BinaryPhaseFieldModel(
        parameters,
        closure=phx.equations.BinaryPhaseThermodynamicClosure(potential),
    )


def test_multiblock_discrete_gradient_closes_one_energy_ledger():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _two_block_mesh(),
        phx.discretization.FiniteElementFieldSpec(
            "eta", {"left": element, "right": element}
        ),
    ).prepare()
    method = phx.applications.phase_field.AllenCahnFEMPlan(
        _binary_model(polynomial=True), 1.0
    ).prepare(discretization, "eta")
    initial = method.initialize(
        jnp.asarray((-0.2, 0.1, 0.3, -0.1, 0.2, -0.3), dtype=jnp.float64)
    )

    result = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.01)
    )

    assert bool(result.successful)
    assert isinstance(
        method.model.evolution_law,
        phx.applications.phase_field.DiscreteGradientBulkLaw,
    )
    assert result.evidence.ledger.total_residual <= result.evidence.energy_tolerance
    assert len(method.discretization.mesh.blocks) == 2


def test_periodic_constraint_identifies_both_square_seams():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        phx.discretization.FiniteElementFieldSpec("u", element),
    ).prepare()
    orientation = phx.discretization.facet_orientation_actions("edge")[0]
    pairs = (
        phx.discretization.FiniteElementPeriodicFacetPair(
            0,
            4,
            transform=phx.discretization.FiniteElementPeriodicTransform(
                jnp.eye(2), jnp.asarray((0.0, 1.0)), orientation
            ),
        ),
        phx.discretization.FiniteElementPeriodicFacetPair(
            2,
            3,
            transform=phx.discretization.FiniteElementPeriodicTransform(
                jnp.eye(2), jnp.asarray((1.0, 0.0)), orientation
            ),
        ),
    )
    boundary = phx.discretization.FiniteElementBoundarySet(
        discretization, {}, periodic_pairs=pairs
    )

    constraint = phx.discretization.periodic_constraint(discretization, "u", boundary)
    expanded = constraint.constraint_map.prolongation.mv(jnp.asarray((2.5,)))

    assert isinstance(constraint.constraint_map.reduced_space, phx.linalg.ArraySpace)
    assert constraint.constraint_map.reduced_space.shape == (1,)
    np.testing.assert_array_equal(expanded, jnp.full((4,), 2.5))


def test_wetting_and_time_dependent_microtraction_enter_boundary_ledger():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    wetting = phx.applications.phase_field.YoungAngleSurfaceEnergy(1.0, jnp.pi / 3.0)
    loading = phx.applications.phase_field.PrescribedMicrotractionEnergy(
        lambda points, time, args: jnp.full(points.shape[:-1], 0.2 * time),
        traction_id="linear-microtraction",
    )
    boundary = phx.applications.phase_field.PhaseFieldBoundaryPlan(
        discretization,
        {
            "wetting": ((0,), wetting, None),
            "loaded": ((4,), loading, None),
        },
    )
    method = phx.applications.phase_field.AllenCahnFEMPlan(_binary_model(), 1.0).prepare(
        discretization, "eta", boundary=boundary
    )
    initial = method.initialize(jnp.asarray((-0.2, 0.1, 0.3, -0.1), dtype=jnp.float64))

    result = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.01)
    )

    assert bool(result.successful)
    assert result.evidence.ledger.surface_before != 0.0
    assert result.evidence.boundary_work != 0.0
    assert result.evidence.ledger.closed


def test_tensor_mobility_and_boundary_flux_close_mass_balance():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        (
            phx.discretization.FiniteElementFieldSpec("c", element),
            phx.discretization.FiniteElementFieldSpec("mu", element),
        ),
    ).prepare()
    mobility = phx.applications.phase_field.TensorPhaseFieldMobility(
        jnp.asarray(((1.0, 0.0), (0.0, 0.5)), dtype=jnp.float64)
    )
    flux = phx.applications.phase_field.PrescribedPhaseFieldFlux(
        lambda points, time, args: jnp.full(points.shape[:-1], 1.0e-3),
        flux_id="constant-inflow",
    )
    boundary = phx.applications.phase_field.PhaseFieldBoundaryPlan(
        discretization,
        {"inlet": ((0,), None, flux)},
    )
    method = phx.applications.phase_field.CahnHilliardFEMPlan(
        _binary_model(), mobility
    ).prepare(discretization, "c", "mu", boundary=boundary)
    initial = method.initialize(jnp.asarray((-0.2, 0.1, 0.3, -0.1), dtype=jnp.float64))

    result = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.005)
    )

    assert bool(result.successful)
    assert bool(result.evidence.mobility_successful)
    np.testing.assert_allclose(result.evidence.mass_source, 5.0e-6)
    assert result.evidence.mass_defect <= result.evidence.mass_tolerance
    assert result.evidence.dissipation > 0.0


def test_dense_grand_potential_step_conserves_components():
    phase_a = phx.applications.phase_field.QuadraticGrandPotentialPhase(
        "phase-a", 0.0, jnp.asarray((0.2,)), jnp.asarray(((1.0,),))
    )
    phase_b = phx.applications.phase_field.QuadraticGrandPotentialPhase(
        "phase-b", 0.0, jnp.asarray((0.8,)), jnp.asarray(((1.0,),))
    )
    catalog = phx.applications.phase_field.GrandPotentialMaterialCatalog(
        (phase_a, phase_b)
    )
    model = phx.applications.phase_field.GrandPotentialMixtureModel(
        catalog,
        barrier_scale=0.1,
        gradient_coefficient=0.2,
        kinetic_coefficient=1.0,
        mobility=1.0,
    )
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        (
            phx.discretization.FiniteElementFieldSpec(
                "eta", element, component_shape=(2,)
            ),
            phx.discretization.FiniteElementFieldSpec(
                "mu", element, component_shape=(1,)
            ),
        ),
    ).prepare()
    method = phx.applications.phase_field.GrandPotentialFEMPlan(
        model,
        absolute_energy_tolerance=1.0,
        relative_energy_tolerance=1.0,
        component_tolerance=1.0e-7,
    ).prepare(discretization, "eta", "mu")
    initial = method.initialize(
        jnp.asarray(
            ((1.0, -1.0), (0.5, -0.5), (-0.5, 0.5), (-1.0, 1.0)),
            dtype=jnp.float64,
        ),
        jnp.zeros((4, 1), dtype=jnp.float64),
    )

    result = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(1.0e-3)
    )

    assert bool(result.successful)
    assert bool(result.evidence.components_conserved)
    assert result.evidence.component_defect <= result.evidence.component_tolerance
    assert result.evidence.energy_defect <= result.evidence.energy_tolerance


def test_active_phase_storage_is_dense_equivalent_and_capacity_safe():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    plan = phx.applications.phase_field.ActivePhaseStoragePlan(
        discretization.dof_maps[0].cell_dofs[0],
        5,
        2,
        cell_phase_capacity=4,
    )
    dense = jnp.asarray(
        (
            (0.7, 0.3, 0.0, 0.0, 0.0),
            (0.6, 0.4, 0.0, 0.0, 0.0),
            (0.0, 0.2, 0.8, 0.0, 0.0),
            (0.0, 0.1, 0.9, 0.0, 0.0),
        ),
        dtype=jnp.float64,
    )
    state = plan.from_dense(dense)

    assert bool(state.evidence.successful)
    np.testing.assert_array_equal(plan.dense(state), dense)
    overflow = plan.from_dense(dense.at[0].set(jnp.asarray((0.4, 0.3, 0.3, 0.0, 0.0))))
    assert bool(overflow.evidence.dof_overflow)
    assert not bool(overflow.evidence.successful)


def test_amr_stochastic_replay_and_distributed_ownership_are_identity_safe():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    realization = phx.stochastic.WienerRealization(
        jax.random.key(7),
        (4,),
        support=(0.0, 1.0),
        tolerance=1.0e-5,
        noise_id="phase-field-regression",
    )
    noise = phx.applications.phase_field.PhaseFieldNoisePlan(
        "allen-cahn", realization, 1.0e-3 * jnp.eye(4)
    )
    method = phx.applications.phase_field.AllenCahnFEMPlan(_binary_model(), 1.0).prepare(
        discretization, "eta", noise=noise
    )
    initial = method.initialize(jnp.asarray((-0.2, 0.1, 0.3, -0.1), dtype=jnp.float64))
    first = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.01)
    )
    replay = method.step_detailed(
        jnp.asarray(0), jnp.asarray(0.0), initial, jnp.asarray(0.01)
    )
    np.testing.assert_array_equal(
        first.candidate_state.phase, replay.candidate_state.phase
    )

    epoch = phx.applications.phase_field.PhaseFieldAdaptiveEpoch(method, initial)
    adaptation = phx.applications.phase_field.PhaseFieldAdaptivityPlan(
        gradient_threshold=0.0,
        energy_tolerance=1.0,
    ).refine(epoch)
    assert bool(adaptation.committed)
    assert adaptation.candidate.epoch_index == 1
    assert adaptation.candidate.method.noise is not None
    assert (
        adaptation.candidate.method.noise.realization.realization_id
        == noise.realization.realization_id
    )
    assert adaptation.candidate.method.noise.basis.shape[0] == 5

    distributed = phx.applications.phase_field.DistributedPhaseFieldPlan(
        discretization, 2
    )
    np.testing.assert_allclose(
        distributed.reference_owned_sum(jnp.asarray((1.0, 2.0))),
        3.0,
    )
    manifest = phx.applications.phase_field.DistributedPhaseFieldCheckpointManifest(
        discretization,
        method.method_id,
        stochastic_id=noise.noise_plan_id,
    )
    assert manifest.stochastic_id == noise.noise_plan_id


def test_integrated_capability_profiles_are_exact_unreleased_claims():
    profiles = phx.applications.phase_field.phase_field_candidate_profiles()

    assert tuple(profile.name for profile in profiles) == (
        "phase-field-deterministic-general-binary",
        "phase-field-stochastic-adaptive-binary",
        "phase-field-active-grand-potential",
        "phase-field-fully-integrated-flagship",
    )
    assert not any(profile.released for profile in profiles)
    assert all(profile.capability == "phase-field-evolution" for profile in profiles)
    released = phx.applications.phase_field.phase_field_released_profiles(
        "qualification-artifact",
        reviewer_id="phase-field-reviewer",
        issued_at=1767225600,
        expires_at=1798761600,
    )
    assert all(profile.released for profile in released)
    assert tuple(profile.name for profile in released) == tuple(
        profile.name for profile in profiles
    )
    assert len(released[-1].dependencies) == 2
