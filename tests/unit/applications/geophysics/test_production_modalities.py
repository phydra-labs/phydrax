#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import geophysics as geo


def _unit_tetra():
    return phx.discretization.CellMesh.from_tetrahedra(
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float),
        np.asarray(((0, 1, 2, 3),)),
    )


def _patch_survey(mesh):
    faces = np.asarray(mesh.connectivity.faces)
    lookup = {tuple(sorted(face)): index for index, face in enumerate(faces)}
    patch_faces = ((0, 2, 3), (0, 1, 3), (0, 1, 2), (1, 2, 3))
    patches = tuple(
        geo.ElectrodePatch(f"e{index}", [lookup[tuple(sorted(face))]])
        for index, face in enumerate(patch_faces)
    )
    return geo.ElectricalSurvey(
        patches,
        jnp.asarray(((1.0, -1.0, 0.0, 0.0),)),
        jnp.asarray(((0.0, 0.0, 1.0, -1.0),)),
        jnp.asarray((0,)),
    )


def _hcurl_mesh():
    vertices = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ),
        dtype=float,
    )
    faces = (
        (1, 3, 5),
        (3, 2, 5),
        (2, 4, 5),
        (4, 1, 5),
        (3, 1, 6),
        (2, 3, 6),
        (4, 2, 6),
        (1, 4, 6),
    )
    cells = np.asarray([(0, *face) for face in faces])
    return phx.discretization.CellMesh.from_tetrahedra(vertices, cells)


def test_complete_and_point_electrode_models_are_finite_and_reciprocal():
    mesh = _unit_tetra()
    finite = geo.FinitePatchDCPlan(mesh, _patch_survey(mesh))
    complete = geo.CompleteElectrodeDCPlan(finite, 0.01).prepare()
    result = complete.solve(1.0)
    assert result.successful
    assert result.dissipated_power_W[0] >= 0
    np.testing.assert_allclose(result.current_residuals, 0.0, atol=1e-14)

    point_positions = np.asarray(
        ((0.10, 0.10, 0.10), (0.20, 0.10, 0.10), (0.10, 0.20, 0.10), (0.10, 0.10, 0.20))
    )
    survey = geo.PointElectrodeSurvey(
        point_positions,
        [[1.0, -1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0, -1.0]],
        [0],
    )
    point = geo.PointElectrodeDCPlan(mesh, survey, 1.0).prepare()
    prediction = point.predict(1.0)
    assert prediction.shape == (1,)
    assert jnp.all(jnp.isfinite(prediction))


def test_line_two_point_five_d_and_ip_responses_are_physical():
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=float),
        np.asarray(((0, 1, 2), (0, 2, 3))),
    )
    positions = np.asarray(
        ((0.15, 0.0, 0.15), (0.85, 0.0, 0.15), (0.25, 0.0, 0.75), (0.75, 0.0, 0.75))
    )
    line_survey = geo.InvariantElectricalSurvey(
        positions,
        [[1.0, -1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0, -1.0]],
        [0],
        current_kind="line-current",
    )
    assert jnp.all(jnp.isfinite(geo.LineCurrentDCPlan(mesh, line_survey).predict(1.0)))
    point_positions = positions.copy()
    point_positions[:, 1] = np.asarray((0.0, 0.0, 0.2, -0.2))
    point_survey = geo.InvariantElectricalSurvey(
        point_positions,
        [[1.0, -1.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0, -1.0]],
        [0],
        current_kind="point-current",
    )
    plan = geo.TwoPointFiveDDCPlan.gauss_legendre(
        mesh, point_survey, wavenumber_max_m_inverse=30.0, count=12
    )
    assert jnp.all(jnp.isfinite(plan.predict(1.0)))

    cole = geo.ColeColeConductivity(1.0, 0.2, 0.5, 0.8)
    conductivity = cole.conductivity(jnp.asarray((0.0, 1.0, 100.0)))
    assert jnp.all(jnp.real(conductivity) > 0)
    assert jnp.all(jnp.imag(conductivity) <= 1e-12)
    debye = geo.DebyeSpectrumConductivity(1.0, [0.3, 0.2], [0.1, 1.0])
    memory, current = debye.advance_memory(debye.initial_memory(()), 1.0, 0.1)
    assert jnp.all(memory > 0)
    assert current > 0


def test_free_space_gravity_magnetics_and_continuation_obey_analytic_limits():
    source = geo.GravityQuadratureSource([[0.0, 0.0, 0.0]], [1.0], [0], 1)
    coordinates = phx.interchange.GeospatialContract.local_cartesian(
        phx.SpatialCoordinateContract.si(), vertical_datum="local"
    )
    gravity = geo.FreeSpaceGravityPlan(
        source, [[0.0, 0.0, 10.0]], coordinates, minimum_separation_m=1.0
    )
    result = gravity.evaluate([1000.0])
    np.testing.assert_allclose(
        result.potential_m2_s2[0],
        geo.GRAVITATIONAL_CONSTANT_M3_KG_S2 * 1000 / 10,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.acceleration_m_s2[0, 2],
        -geo.GRAVITATIONAL_CONSTANT_M3_KG_S2 * 1000 / 100,
        rtol=1e-12,
    )
    magnetic = geo.FreeSpaceMagneticPlan(
        source,
        [[0.0, 0.0, 10.0]],
        [0.0, 0.0, 1.0],
        coordinates,
        minimum_separation_m=1.0,
    )
    material = geo.MagneticMaterial(0.1, [0.0, 0.0, 0.0], 1)
    field = magnetic.evaluate(material, [0.0, 0.0, 50_000.0])
    assert field.flux_density_T[0, 2] > 0
    constant = jnp.full((8, 8), 3.0)
    continued = geo.FourierContinuationPlan((8, 8), (1.0, 1.0), 2.0).apply(constant)
    np.testing.assert_allclose(continued, constant, atol=1e-12)


def test_variable_elastic_cpml_travel_and_surface_wave_workflows_execute():
    grid = geo.AcousticGrid((9, 9), (1.0, 1.0))
    acquisition = geo.SeismicAcquisition(grid, [[4.0, 4.0]], [[5.0, 4.0]])
    source = jnp.zeros((4, 1)).at[0, 0].set(0.01)
    variable = geo.VariableDensityAcousticPlan(grid, 0.05, 4, 2.0)
    simulation = variable.simulate(1.5, 1000.0, acquisition, source)
    assert jnp.all(jnp.isfinite(simulation.traces.values))

    elastic_acquisition = geo.ElasticAcquisition(grid, [[4.0, 4.0]], [[5.0, 4.0]])
    elastic = geo.PeriodicIsotropicElasticWavePlan(grid, 0.05, 3, 2.0)
    force = jnp.zeros((3, 1, 2))
    moment = jnp.zeros((3, 1, 3))
    elastic_result = elastic.simulate(1.0, 2.0, 1.0, elastic_acquisition, force, moment)
    assert elastic_result.finite

    cpml = geo.CartesianCPML(
        grid,
        0.05,
        2.0,
        widths=((2, 2), (0, 2)),
    )
    cpml_plan = geo.CPMLVariableDensityAcousticPlan(
        variable, cpml, free_surface_axis=1, free_surface_side="lower"
    )
    cpml_result = cpml_plan.simulate(1.5, 1000.0, acquisition, source)
    assert cpml_result.finite

    graph = geo.TravelTimeGraphPlan(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [[0, 1], [1, 2]]
    )
    travel = graph.solve(2.0, 0)
    np.testing.assert_allclose(travel.travel_times_s, [0.0, 0.5, 1.0])
    rayleigh = geo.HomogeneousRayleighWavePlan(3000.0, 1700.0)
    assert 0.85 * 1700 < rayleigh.phase_velocity_m_s < 1700


def test_layered_frequency_and_time_domain_em_are_passive():
    layered = geo.LayeredEarthModel([100.0], [1.0, 1.0])
    impedance = layered.magnetotelluric_impedance([1.0, 10.0])
    assert jnp.all(jnp.isfinite(impedance))
    assert jnp.all(jnp.real(impedance) > 0)

    mesh = _hcurl_mesh()
    hcurl = phx.discretization.TetrahedralNedelecSpace(mesh)
    connectivity = mesh.connectivity
    free = np.flatnonzero(~np.asarray(connectivity.boundary_edges))
    source = np.zeros((1, hcurl.edge_count))
    source[0, free[0]] = 1.0
    receivers = source.copy()
    survey = geo.FrequencyDomainEMSurvey(source, receivers, [0])
    material = geo.ConductiveEMMaterial(
        1.0,
        geo.electromagnetics.VACUUM_PERMITTIVITY_F_M,
        1.0 / geo.electromagnetics.VACUUM_PERMEABILITY_H_M,
        hcurl.cell_count,
    )
    frequency = geo.FrequencyDomainEMPlan(mesh, survey).solve([2 * np.pi], material)
    assert frequency.successful
    assert frequency.dissipated_power_W[0, 0] >= 0
    time = geo.ImplicitTimeDomainEMPlan(mesh, survey).simulate(
        material, [0.01, 0.01], [[1.0], [0.0]]
    )
    assert time.successful
    assert jnp.all(time.dissipated_energy_J >= 0)
