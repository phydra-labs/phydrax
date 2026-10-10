#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import hashlib
import json
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications import geophysics as geo, porous_media as porous
from phydrax.observation import CoordinateLayout, DiagonalCovarianceAction
from phydrax.qualification import GeophysicalReferenceRecipe, ReferenceArtifactManifest
from phydrax.series import SampledSeries, SeriesSupport
from phydrax.units import DEGREE, METER
from tools.geophysics_reference_qualification import qualify


def _bridge(shape: Any = (3, 3)) -> Any:
    dimension = len(shape)
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for count in shape),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension]))
    return phx.discretization.StructuredCochainBridge(grid)


def _finite_volume_geometry() -> Any:
    return phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)),
            dtype="float64",
        ),
        tetrahedra=np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    ).prepare()


def _field_space(name: Any, count: Any) -> Any:
    topology = phx.discretization.TensorTopology(("x",), (count,))
    support = phx.discretization.DiscreteSupport(topology, 1, f"{name}-line")
    layout = phx.discretization.TensorDofLayout(("x",), (count,))
    return phx.discretization.DiscreteFieldSpace(
        name,
        support.support_id,
        layout,
        phx.linalg.ArraySpace((count,)),
        representation="basis_coefficient",
        conformity="H1",
    )


def test_governed_external_and_field_references_are_content_verified_and_evidenced(
    tmp_path: Any,
) -> None:
    payload = b"field-reference"
    manifest = ReferenceArtifactManifest(
        "field-reference",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC-BY-4.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="EAR99",
        nondimensionalization={"observable": 1.0},
        uncertainty={"standard-deviation": 0.1},
        lineage_ids=("doi:10.0000/example",),
    )
    assert manifest.verify_bytes(payload) == manifest.manifest_id
    recipe = GeophysicalReferenceRecipe(
        "field",
        "seismic",
        "governed-field-case",
        "pressure-Pa",
        "https://example.invalid/pinned-by-checksum",
        manifest,
        "coordinate-contract",
        time_contract_id="time-contract",
        absolute_tolerance=0.1,
        relative_tolerance=0.0,
        maximum_standardized_rms=1.0,
        maximum_samples=3,
        minimum_valid_samples=2,
    )
    assert (
        GeophysicalReferenceRecipe.from_record(recipe.to_record()).recipe_id
        == recipe.recipe_id
    )
    comparison = recipe.compare(
        # ty: ignore[invalid-argument-type]
        [1.0, 2.05, 1000.0],
        # ty: ignore[invalid-argument-type]
        [1.0, 2.0, 0.0],
        # ty: ignore[invalid-argument-type]
        valid=[True, True, False],
        # ty: ignore[invalid-argument-type]
        standard_deviation=[0.1, 0.1, 0.1],
    )
    assert comparison.passed
    assert comparison.valid_sample_count == 2
    evidence = comparison.evidence(
        ("seismic-forward",),
        build_id="build",
        environment_id="environment",
        backend="cpu",
        topology="serial",
        precision="float64",
        reduction="deterministic",
        replay_id="replay",
        campaign_start_record_ids=("campaign-start",),
        campaign_observation_record_ids=("campaign-observation",),
        reviewer_id="reviewer",
        issued_at=1,
        expires_at=2,
    )
    assert evidence.passed
    with pytest.raises(ValueError, match="checksum"):
        manifest.verify_bytes(b"FIELD-REFERENCE")
    artifact_path = tmp_path / "external-oracle.npz"
    np.savez(
        artifact_path,
        prediction=np.asarray([1.0, 2.001]),
        reference=np.asarray([1.0, 2.0]),
    )
    artifact_bytes = artifact_path.read_bytes()
    oracle_manifest = ReferenceArtifactManifest(
        "external-oracle",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(artifact_bytes).hexdigest(),
        size_bytes=len(artifact_bytes),
        license_id="BSD-3-Clause",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="EAR99",
        nondimensionalization={"observable": 1.0},
        uncertainty=None,
        lineage_ids=("oracle:independent",),
    )
    oracle_recipe = GeophysicalReferenceRecipe(
        "external-oracle",
        "electrical",
        "independent-halfspace",
        "voltage-V",
        "local:external-oracle.npz",
        oracle_manifest,
        "coordinate-contract",
        absolute_tolerance=0.002,
        relative_tolerance=0.0,
        maximum_samples=2,
        minimum_valid_samples=2,
    )
    recipe_path = tmp_path / "external-oracle.json"
    recipe_path.write_text(json.dumps(oracle_recipe.to_record()), encoding="utf-8")

    report = qualify(recipe_path, artifact_path)
    with pytest.raises(ValueError, match="sample bound"):
        # ty: ignore[invalid-argument-type]
        oracle_recipe.compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert report["passed"]


def test_geophysics_production_qualification_scenario_1() -> None:
    source = phx.interchange.bounded_resource_from_bytes(
        b"leap", limits=phx.interchange.ResourceLimits(100, 1, 1, 1, 0)
    ).manifest
    # ty: ignore[invalid-argument-type]
    leaps = phx.interchange.LeapSecondTable([100.0], [11.0], 10.0, source)
    utc = phx.interchange.TimeReferenceContract(
        "utc",
        "origin",
        10.0,
        epoch_nominal_seconds=0.0,
        leap_seconds=leaps,
    )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="positive leap"):
        utc.from_tai(110.5)

    body = phx.interchange.ReferenceBodyContract("oblate", 4e14, 6.4e6, 6.3e6, 0.0, 0.0)
    coordinates = phx.interchange.GeospatialContract(
        phx.SpatialCoordinateContract(METER, reference_frame="oblate-body"),
        horizontal_crs="oblate-body-geographic",
        horizontal_kind="geographic",
        horizontal_axes=("longitude", "latitude"),
        horizontal_units=(DEGREE, DEGREE),
        horizontal_datum="oblate-body",
        epoch_required=False,
        vertical_kind="elevation",
        vertical_positive="up",
        vertical_datum="oblate-body-surface",
        vertical_unit=METER,
        registration="unknown",
        longitude_seam="none",
        longitude_domain=(-180.0, 180.0),
        mask_semantics="none",
    )
    planetary = phx.interchange.PlanetaryCoordinateContract(
        body, coordinates, latitude_kind="planetocentric"
    )
    # ty: ignore[invalid-argument-type]
    pole = planetary.to_body_fixed_cartesian([0.0, 90.0, 0.0])
    np.testing.assert_allclose(pole, [0.0, 0.0, 6.3e6], atol=1e-8)

    debye = geo.DebyeSpectrumConductivity(
        # ty: ignore[invalid-argument-type]
        [1.0, 2.0],
        # ty: ignore[invalid-argument-type]
        [[0.2, 0.1], [0.3, 0.2]],
        # ty: ignore[invalid-argument-type]
        [0.1, 1.0],
    )
    # ty: ignore[invalid-argument-type]
    conductivity = debye.conductivity([1.0, 10.0, 100.0])
    assert conductivity.shape == (3, 2)
    assert jnp.all(jnp.imag(conductivity) <= 0)
    # ty: ignore[invalid-argument-type]
    facies = geo.FaciesProbabilityPlan(("sand", "clay"), [[1.0, 10.0], [3.0, 30.0]])
    # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(facies.mixture([0.0, 0.0]), [2.0, 20.0], rtol=1e-12)
    grid = geo.AcousticGrid((7, 7), (1.0, 1.0))
    # ty: ignore[invalid-argument-type]
    acquisition = geo.SeismicAcquisition(grid, [[3.0, 3.0]], [[4.0, 3.0]])
    forward = geo.ConstantDensityAcousticPlan(grid, 0.1, 4, 2.0)
    source = jnp.zeros((4, 1)).at[0, 0].set(0.1)
    observed = forward.simulate(1.5, acquisition, source).traces.values
    layout = CoordinateLayout(tuple(f"sample-{index}" for index in range(observed.size)))
    covariance = DiagonalCovarianceAction(jnp.ones(observed.size), layout)
    shot = geo.AcousticShot(acquisition, source, observed, covariance)
    inversion = geo.AcousticWaveformInversionPlan(forward, (shot,), replay="full")
    result = inversion.evaluate(jnp.asarray(1.5))
    np.testing.assert_allclose(result.objective, 0.0, atol=1e-14)
    np.testing.assert_allclose(result.gradient, 0.0, atol=1e-12)
    rtm = inversion.rtm(jnp.asarray(1.5))
    np.testing.assert_allclose(rtm.image, 0.0, atol=1e-12)
    projection = geo.AcousticSourceProjectionPlan(
        inversion, 0, source[None, ...], jnp.asarray(1.5)
    ).evaluate()
    # ty: ignore[unresolved-attribute]
    np.testing.assert_allclose(projection.parameters, [1.0], atol=1e-10)

    # ty: ignore[invalid-argument-type]
    spectrum = geo.StandardLinearSolidSpectrum([0.5], [0.1])
    anisotropic = geo.PeriodicAnisotropicViscoelasticPlan(grid, 0.01, 2, spectrum)
    stiffness = geo.ElasticStiffness.isotropic(2.0, 1.0, 2)
    state, observations = anisotropic.simulate(
        1.0,
        stiffness,
        # ty: ignore[invalid-argument-type]
        geo.ElasticAcquisition(grid, [[3.0, 3.0]], [[4.0, 3.0]]),
        jnp.zeros((2, 1, 2)),
    )
    np.testing.assert_allclose(state.velocity_m_s, 0.0)
    np.testing.assert_allclose(observations, 0.0)
    other_grid = geo.AcousticGrid((7, 7), (2.0, 1.0))
    with pytest.raises(ValueError, match="another grid"):
        anisotropic.simulate(
            1.0,
            stiffness,
            # ty: ignore[invalid-argument-type]
            geo.ElasticAcquisition(other_grid, [[3.0, 3.0]], [[4.0, 3.0]]),
            jnp.zeros((2, 1, 2)),
        )
    # ty: ignore[invalid-argument-type]
    layered = geo.LayeredEarthModel([100.0], [0.01, 0.001])
    plan, expected = geo.MagnetotelluricResponsePlan.from_layered(
        layered, jnp.asarray([1.0, 10.0])
    )
    magnetic = jnp.broadcast_to(jnp.eye(2, dtype="complex128"), (2, 2, 2))
    response = plan.evaluate(expected, magnetic)
    assert response.finite
    np.testing.assert_allclose(response.impedance_ohm, expected)

    remote = jnp.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, -1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ],
        dtype="complex128",
    )
    electric = phx.ein.contract("fij,wfj->wfi", expected, remote)
    estimated, coherence = geo.RemoteReferenceMTPlan().estimate(electric, remote, remote)
    np.testing.assert_allclose(estimated, expected, atol=1e-12)
    np.testing.assert_allclose(coherence, 1.0, atol=1e-12)

    # ty: ignore[invalid-argument-type]
    love = geo.LayeredLoveWavePlan([100.0], [2000.0, 2500.0], [1200.0, 2200.0])
    # ty: ignore[invalid-argument-type]
    modes = love.solve([5.0, 10.0])
    assert jnp.any(modes.mode_valid)
    assert jnp.all(
        (modes.phase_velocities_m_s[modes.mode_valid] > 1200.0)
        & (modes.phase_velocities_m_s[modes.mode_valid] < 2200.0)
    )

    support = SeriesSupport(np.arange(16) * 0.1)
    values = jnp.sin(2 * jnp.pi * jnp.arange(16) / 8)
    series = SampledSeries(support, values, series_id="qualified-ambient-noise")
    correlation = geo.AmbientNoiseCorrelationPlan(8, 4, 0.1).evaluate(series, series)
    assert correlation.successful
    assert jnp.argmax(correlation.correlation) == 4
    shifted = SampledSeries(
        SeriesSupport(np.arange(16) * 0.1 + 0.05),
        values,
        series_id="shifted-ambient-noise",
    )
    with pytest.raises(Exception, match="connected uniformly sampled clock"):
        geo.AmbientNoiseCorrelationPlan(8, 4, 0.1).evaluate(series, shifted)
    frequency, ratio = geo.HVSRPlan(0.1).evaluate(2 * values, 2 * values, values)
    assert frequency.shape == ratio.shape
    np.testing.assert_allclose(ratio[2], 2.0, rtol=1e-6)


def test_distributed_operator_topology_transfer_and_physics_preconditioners_are_paired() -> (
    None
):
    matrix = jnp.asarray(
        [
            [2.0, -1.0, 0.0, -1.0, 0.0],
            [-1.0, 3.0, -1.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, 0.0, -1.0],
            [-1.0, -1.0, 0.0, 3.0, -1.0],
            [0.0, 0.0, -1.0, -1.0, 2.0],
        ]
    )
    halo = phx.discretization.DistributedHaloPlan(
        # ty: ignore[invalid-argument-type]
        [0, 0, 0, 1, 1],
        # ty: ignore[invalid-argument-type]
        [[0, 1], [1, 2], [0, 3], [1, 3], [2, 4], [3, 4]],
        2,
    )

    def local_action(part: Any, local: Any, ids: Any, valid: Any, owned: Any) -> Any:
        del part
        global_values = jnp.zeros(5).at[ids].add(jnp.where(valid, local, 0.0))
        return jnp.where(owned, (matrix @ global_values)[ids], 0.0)

    def local_transpose(part: Any, local: Any, ids: Any, valid: Any, owned: Any) -> Any:
        del part
        rows = jnp.zeros(5).at[ids].add(jnp.where(owned, local, 0.0))
        return jnp.where(valid, (matrix.T @ rows)[ids], 0.0)

    operator = phx.discretization.DistributedLocalOperator(
        halo, local_action, local_transpose, operator_name="qualified-chain-laplacian"
    )
    value = jnp.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    cotangent = jnp.asarray([-1.0, 0.5, 2.0, 1.0, -0.25])
    np.testing.assert_allclose(operator.serial_reference(value), matrix @ value)
    np.testing.assert_allclose(
        operator.serial_transpose_reference(cotangent), matrix.T @ cotangent
    )
    np.testing.assert_allclose(
        jnp.vdot(operator.serial_reference(value), cotangent),
        jnp.vdot(value, operator.serial_transpose_reference(cotangent)),
    )
    assert jnp.any(halo.phase_send_valid != halo.phase_receive_valid)

    source_space, target_space = _field_space("coarse", 2), _field_space("fine", 4)
    prolongation = jnp.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    forward = phx.linalg.DenseLinearOperator(
        prolongation,
        source=source_space.vector_space,
        target=target_space.vector_space,
    )
    adjoint = phx.linalg.DenseLinearOperator(
        prolongation.T,
        source=target_space.vector_space,
        target=source_space.vector_space,
    )
    transfer = phx.discretization.FieldTransfer(
        source_space,
        target_space,
        forward,
        dual_pullback_operator=adjoint,
        hilbert_adjoint_operator=adjoint,
        geometry=phx.discretization.TransferGeometryBinding(
            "geometry-0",
            "geometry-1",
            "topology-correspondence",
            source_topology_id="topology-0",
            target_topology_id="topology-1",
            coverage_defect=None,
        ),
        properties=phx.discretization.TransferProperties(
            constant_preserving=True,
            conservative=True,
            adjoint_paired=True,
            differentiable_geometry=False,
            exact_on=("constants",),
        ),
    )
    transition = phx.discretization.TopologyEpochTransition(
        phx.discretization.TopologyEpoch(0, "geometry-0", "topology-0", "serial"),
        phx.discretization.TopologyEpoch(1, "geometry-1", "topology-1", "serial"),
        transfer,
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        # ty: ignore[invalid-argument-type]
        [0.5, 0.5, 0.5, 0.5],
    )
    # ty: ignore[invalid-argument-type]
    transitioned = transition.apply([2.0, 4.0])
    assert transitioned.successful
    np.testing.assert_allclose(transitioned.values, [2.0, 2.0, 4.0, 4.0])
    assert not transitioned.differentiation_available
    wave_state = geo.SecondOrderWaveState(
        jnp.asarray([2.0, 4.0]),
        jnp.asarray([1.0, 3.0]),
        jnp.asarray(2, dtype=jnp.int32),
        "source-wave",
    )
    amr = geo.SeismicAMRTransition(
        transition, transition, "source-wave", "target-wave"
    ).apply(wave_state)
    assert amr.successful
    assert amr.state.plan_id == "target-wave"
    assert not amr.differentiation_available

    edge_space, scalar_space = phx.linalg.ArraySpace((2,)), phx.linalg.ArraySpace((1,))

    system = phx.linalg.DenseLinearOperator(
        # ty: ignore[invalid-argument-type]
        [[2.0, 1.0], [1.0, 3.0]],
        source=edge_space,
        target=edge_space,
    )
    zero_inverse = phx.linalg.OperatorPreconditioner(
        phx.linalg.DenseLinearOperator(
            np.zeros((2, 2)), source=edge_space, target=edge_space
        )
    )
    restriction = phx.linalg.DenseLinearOperator(
        # ty: ignore[invalid-argument-type]
        [[1.0, 0.0]],
        source=edge_space,
        target=scalar_space,
    )
    scalar_inverse = phx.linalg.DiagonalPreconditioner(
        jnp.asarray([2.0]), space=scalar_space, positive_definite=True
    )
    cpr = phx.linalg.porous_cpr_preconditioner(
        system, zero_inverse, restriction, scalar_inverse
    )
    np.testing.assert_allclose(cpr.apply([2.0, 4.0]), [1.0, 0.0])


def test_geophysics_production_qualification_scenario_2() -> None:
    bridge = _bridge()
    pulse = geo.GaussianDerivativeWaveform(10.0, 0.1)
    source = phx.solver.maxwell.MaxwellElectricCurrentSourcePlan(
        # ty: ignore[invalid-argument-type]
        [0],
        # ty: ignore[invalid-argument-type]
        [1.0],
        envelope=pulse,
    )
    # ty: ignore[invalid-argument-type]
    observer = phx.solver.maxwell.FieldProbePlan("electric", [0])
    material = phx.solver.maxwell.LorentzDrudeMaxwellConstitutivePlan(
        phx.solver.maxwell.MaxwellLorentzPoles(
            jnp.asarray([1.0]), jnp.asarray([0.1]), jnp.asarray([0.5])
        )
    )
    runtime = phx.solver.CompatibleMaxwellPlan(
        bridge,
        polarization="tez",
        constitutive=material,
        pml=phx.solver.maxwell.MaxwellCPMLPlan(1),
        sources=(source,),
        observers=(observer,),
    ).prepare()
    result = geo.DispersiveFullWaveGPRPlan(
        runtime, 0.05 * runtime.stable_dt, 3
    ).simulate()
    assert result.finite
    assert result.passive
    np.testing.assert_allclose(pulse(0.1), 0.0, atol=1e-15)
    objective = lambda value: 0.5 * jnp.sum((value - 2.0) ** 2)
    hessian = lambda _parameters, direction: direction
    map_plan = geo.MatrixFreeMAPPlan(
        objective,
        hessian,
        1,
        objective_id="quadratic-map",
        hessian_action_id="identity-hessian",
        maximum_iterations=4,
        gradient_tolerance=1e-10,
    )
    # ty: ignore[invalid-argument-type]
    mapped = map_plan.solve([0.0])
    assert mapped.converged
    np.testing.assert_allclose(mapped.parameters, [2.0], atol=1e-10)

    layout = CoordinateLayout(("observation",))
    # ty: ignore[invalid-argument-type]
    covariance = DiagonalCovarianceAction([0.1], layout)
    ensemble = jnp.asarray([[-1.0], [0.0], [2.0], [4.0]])
    # ty: ignore[invalid-argument-type]
    eki = geo.EnsembleKalmanInversionPlan([1.0], covariance)
    updated = eki.update(ensemble, lambda value: value)
    assert updated.finite
    assert abs(float(jnp.mean(updated.ensemble)) - 1.0) < abs(
        float(jnp.mean(ensemble)) - 1.0
    )

    pcn = geo.PCNSampler(
        lambda value: -0.5 * jnp.sum((value - 1.0) ** 2),
        0.2,
        8,
        log_likelihood_id="unit-gaussian-shift",
        # ty: ignore[invalid-argument-type]
    ).sample(jax.random.key(4), [0.0, 0.0])
    assert jnp.all(jnp.isfinite(pcn.samples))
    assert 0 <= pcn.acceptance_rate <= 1

    epochs = (
        # ty: ignore[invalid-argument-type]
        geo.MonitoringEpoch(0.0, [0.0], covariance, "geometry", "acquisition-0"),
        # ty: ignore[invalid-argument-type]
        geo.MonitoringEpoch(1.0, [1.0], covariance, "geometry", "acquisition-1"),
    )
    monitoring = geo.SequentialMonitoringPlan(
        epochs,
        lambda member, dt, key: (
            member + 0.0 * dt + 0.0 * jax.random.normal(key, member.shape)
        ),
        (lambda value, time: value + 0.0 * time, lambda value, time: value + 0.0 * time),
        # ty: ignore[invalid-argument-type]
        [0.0],
        dynamics_id="identity-dynamics",
        prediction_ids=("identity-observer-0", "identity-observer-1"),
    )
    monitored = monitoring.step(monitoring.initialize(ensemble, jax.random.key(5)))
    assert monitored.successful
    assert int(monitored.state.epoch_index) == 1


def test_monolithic_reactive_spherical_fields_rays_and_geodynamics_close(
    tmp_path: Any,
) -> None:
    geometry = _finite_volume_geometry()
    chemistry = porous.MassActionSystem(
        ("A",),
        (),
        np.empty((0, 1)),
        # ty: ignore[invalid-argument-type]
        [],
        # ty: ignore[invalid-argument-type]
        [0.0],
        reference_concentration=1.0,
    )
    transport = porous.ComponentTransport(geometry, chemistry.primary_names)
    concentrations = jnp.ones((geometry.cell_count, 1))
    volumes = jnp.full((geometry.cell_count,), 0.1)
    inventory = volumes[:, None] * jax.vmap(chemistry.component_totals)(concentrations)
    reactive = porous.MonolithicReactiveTransportPlan(transport, chemistry).step(
        inventory,
        volumes,
        jnp.zeros(geometry.owner_cells.size),
        1.0,
        porous.TransportBoundary(geometry, 1),
        concentrations,
    )
    assert reactive.successful
    np.testing.assert_allclose(reactive.component_balance_mol, 0.0, atol=1e-12)

    exchange = porous.AtmosphericExchangePlan(1.0, 0.1).evaluate(
        270.0, 280.0, 1.0, 0.0, 0.0, 0.0, 1.0
    )
    assert exchange.evaporation_kg_s < 0
    assert not exchange.limited
    fractions = porous.IonExchangeEquilibrium(
        # ty: ignore[invalid-argument-type]
        [1.0, 2.0],
        # ty: ignore[invalid-argument-type]
        [1.0, 2.0],
        # ty: ignore[invalid-argument-type]
    ).equivalent_fractions([[1.0, 4.0], [2.0, 2.0]])
    np.testing.assert_allclose(jnp.sum(fractions, axis=-1), 1.0)

    gravity_file = tmp_path / "monopole.gfc"
    gravity_file.write_text(
        "modelname monopole\n"
        "earth_gravity_constant 4e14\n"
        "radius 6.4e6\n"
        "max_degree 0\n"
        "norm unnormalized\n"
        "tide_system zero_tide\n"
        "end_of_head\n"
        "gfc 0 0 1 0\n"
    )
    gravity_model = phx.interchange.read_icgem_gfc(
        gravity_file.name,
        trusted_root=tmp_path,
        limits=phx.interchange.ResourceLimits(10_000, 2, 100, 100, 1),
    )
    spherical = geo.SphericalHarmonicGravityPlan(gravity_model).evaluate(
        # ty: ignore[invalid-argument-type]
        [[12.8e6, 0.0, 0.0]]
    )
    np.testing.assert_allclose(spherical.potential_m2_s2, [4e14 / 12.8e6])
    np.testing.assert_allclose(
        spherical.acceleration_m_s2[0], [-4e14 / 12.8e6**2, 0.0, 0.0]
    )

    body = phx.interchange.ReferenceBodyContract("body", 4e14, 6.4e6, 6.4e6, 0.0, 0.0)
    radial = geo.RadialBodyModel(
        # ty: ignore[invalid-argument-type]
        [0.0, 3.2e6, 6.4e6],
        # ty: ignore[invalid-argument-type]
        [4000.0, 4000.0, 4000.0],
        # ty: ignore[invalid-argument-type]
        [8000.0, 8000.0, 8000.0],
        # ty: ignore[invalid-argument-type]
        [4000.0, 4000.0, 4000.0],
        # ty: ignore[invalid-argument-type]
        [1000.0, 1000.0, 1000.0],
        # ty: ignore[invalid-argument-type]
        [1000.0, 1000.0, 1000.0],
        # ty: ignore[invalid-argument-type]
        [4.0, 4.0, 4.0],
        body,
    )
    ray = geo.SphericalRayPlan(radial, "P").evaluate(0.0)
    assert ray.successful
    np.testing.assert_allclose(ray.travel_time_s, 2 * 6.4e6 / 8000.0)

    velocity_space = phx.linalg.ArraySpace((2,))
    pressure_space = phx.linalg.ArraySpace((1,))
    temperature_space = phx.linalg.ArraySpace((1,))
    divergence = phx.linalg.DenseLinearOperator(
        # ty: ignore[invalid-argument-type]
        [[1.0, 0.0]],
        source=velocity_space,
        target=pressure_space,
    )
    buoyancy = phx.linalg.DenseLinearOperator(
        # ty: ignore[invalid-argument-type]
        [[0.0], [1.0]],
        source=temperature_space,
        target=velocity_space,
    )
    geodynamics = geo.SphericalThermomechanicalPlan(
        geo.SphericalShellGeometry(3.0e6, 6.4e6, body.body_id),
        velocity_space,
        pressure_space,
        lambda temperature: phx.linalg.DenseLinearOperator(
            # ty: ignore[invalid-argument-type]
            [[2.0, 0.0], [0.0, 3.0]],
            source=velocity_space,
            target=velocity_space,
        ),
        divergence,
        buoyancy,
        lambda velocity, temperature: jnp.zeros_like(temperature),
        1.0,
        momentum_factory_id="constant-viscosity",
        thermal_rate_id="zero-thermal-rate",
    )
    # ty: ignore[invalid-argument-type]
    geodynamic_step = geodynamics.step(geodynamics.initial_state([300.0]), 0.1)
    assert geodynamic_step.successful
    np.testing.assert_allclose(geodynamic_step.incompressibility_residual, 0.0, atol=1e-8)
    np.testing.assert_allclose(geodynamic_step.thermal_residual, 0.0, atol=1e-12)
