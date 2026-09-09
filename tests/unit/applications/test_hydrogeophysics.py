#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.applications.geophysics import (
    ArchieSaturationConductivity,
    ElectricalSurvey,
    ElectrodePatch,
    FinitePatchDCPlan,
    HydrogeophysicalPlan,
)
from phydrax.discretization import CellMesh, nested_cell_transfer
from phydrax.discretization.finite_volume import UnstructuredFiniteVolumePlan
from phydrax.interchange import GeospatialContract


def _mesh():
    return CellMesh.from_tetrahedra(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            )
        ),
        np.asarray(((0, 1, 2, 3), (0, 2, 1, 4))),
    )


def _finite_volume(mesh):
    return UnstructuredFiniteVolumePlan(
        np.asarray(mesh.coordinates),
        tetrahedra=np.asarray(mesh.blocks[0].vertices),
    ).prepare()


def _electrical(mesh):
    exterior = np.flatnonzero(np.asarray(mesh.connectivity.boundary_faces))
    patches = tuple(
        ElectrodePatch(f"E{index}", [int(face)])
        for index, face in enumerate(exterior[:4])
    )
    survey = ElectricalSurvey(
        patches,
        jnp.asarray(((1.0, -1.0, 0.0, 0.0),)),
        jnp.asarray(((1.0, -1.0, 0.0, 0.0),)),
        jnp.asarray((0,)),
    )
    return FinitePatchDCPlan(mesh, survey).prepare()


def test_fixed_nested_transfer_and_uncertain_archie_law_compose_into_dc_prediction():
    mesh = _mesh()
    finite_volume = _finite_volume(mesh)
    transfer = nested_cell_transfer(finite_volume, finite_volume, [0, 1])
    geospatial = GeospatialContract.local_cartesian(
        SpatialCoordinateContract.si(), vertical_datum="local-survey-datum"
    )
    electrical = _electrical(mesh)
    plan = HydrogeophysicalPlan(
        electrical,
        transfer,
        finite_volume.cell_volumes,
        finite_volume.cell_volumes,
        source_geometry_id=finite_volume.geometry_id,
        target_geometry_id=finite_volume.geometry_id,
        source_coordinates=geospatial,
        target_coordinates=geospatial,
    )
    porosity = jnp.asarray((0.3, 0.4))
    saturation = jnp.asarray((0.7, 0.8))
    water = porosity * saturation * finite_volume.cell_volumes
    temperature = jnp.asarray((290.0, 300.0))
    calibration = ArchieSaturationConductivity(
        0.5,
        cementation_exponent=2.0,
        saturation_exponent=2.2,
        temperature_coefficient_K_inverse=0.02,
    )
    result = plan.predict(
        porosity,
        water,
        temperature,
        calibration,
        log_discrepancy=jnp.asarray((0.1, -0.05)),
    )
    expected_conductivity = calibration.conductivity(
        porosity,
        saturation,
        temperature,
        log_discrepancy=jnp.asarray((0.1, -0.05)),
    )
    np.testing.assert_allclose(result.conductivity_S_m, expected_conductivity)
    np.testing.assert_allclose(
        result.response, electrical.predict(expected_conductivity), atol=1.0e-10
    )
    np.testing.assert_allclose(result.water_volume_residual_m3, 0.0, atol=1.0e-15)

    def response(discrepancy):
        return plan.predict(
            porosity, water, temperature, calibration, log_discrepancy=discrepancy
        ).response[0]

    direction = jnp.asarray((0.3, -0.2))
    _, tangent = jax.jvp(response, (jnp.zeros(2),), (direction,))
    reverse = jnp.vdot(jax.grad(response)(jnp.zeros(2)), direction)
    np.testing.assert_allclose(tangent, reverse, atol=1.0e-9)
    assert abs(float(tangent)) > 1.0e-8
    with pytest.raises(ValueError, match="geometry change"):
        plan.require_geometry("changed", finite_volume.geometry_id)


def test_nested_transfer_rejects_nonpartitioning_parent_routes():
    finite_volume = _finite_volume(_mesh())
    with pytest.raises(ValueError, match="partition"):
        nested_cell_transfer(finite_volume, finite_volume, [0, 0])
