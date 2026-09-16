#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology._dark_sector_species import DarkSectorSpeciesPlan
from phydrax.applications.cosmology._quantum_dark_kinetics import QuantumKineticState
from phydrax.applications.curved_spacetime_qft._gauge_covariant_wigner import (
    gauge_covariance_evidence,
    GaugeCovariantWignerPlan,
)
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.discretization import (
    OrientedEdgePathPlan,
    polygonal_cell_complex,
    prepare_cell_boundary_paths,
)
from phydrax.graph import MatrixGaugeLinkSpace
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    FundamentalGaugeRepresentation,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
    SpecialUnitaryGroup,
    U1ChargeRepresentation,
    UnitaryGroup,
)


def _support():
    scale = RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1)
    units = RelativisticUnitContract(
        scale, RelativityConvention(metric_signature="mostly_minus")
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(4),
        chart_id="minkowski-cartesian",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="gauge-wigner-observer",
        geometry_lineage_id="flat",
    )
    chart = CoordinateChart("minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        tolerance=1.0e-7,
        source_id="gauge-wigner-observer",
    )
    frame = LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="gauge-wigner-observer",
        orientation_id="right-handed-future",
    )
    species = (
        DarkSectorSpeciesPlan(
            "charged-dark-fermion",
            1.0,
            mass_unit=units.scale.dimensional_scale.mass_unit.unit_id,
            energy_unit=units.energy_unit.unit_id,
        ),
    )
    return QuantumKineticState(
        jnp.asarray([[[0.2, 0.3]]]),
        None,
        species=species,
        statistics=jnp.asarray((-1,), dtype=jnp.int8),
        spatial_active=jnp.asarray((True,)),
        momentum_active=jnp.asarray((True, True)),
        units=units,
        frame=frame,
    )


def _paths(topology):
    boundary = prepare_cell_boundary_paths(topology).paths
    edges = boundary.edge_indices[0]
    signs = boundary.orientations[0]
    return OrientedEdgePathPlan(
        topology,
        jnp.asarray([[edges[0], edges[1]], [edges[0], edges[1]]]),
        jnp.asarray([[signs[0], signs[1]], [signs[0], signs[1]]]),
        valid=jnp.asarray([[True, False], [True, True]]),
        path_names=("nearest", "next-nearest"),
    )


def test_abelian_wilson_line_wigner_transform_is_gauge_covariant():
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    group = UnitaryGroup(1)
    link_space = MatrixGaugeLinkSpace(topology, group)
    representation = U1ChargeRepresentation(group, 2)
    paths = _paths(topology)
    plan = GaugeCovariantWignerPlan(_support(), link_space, representation, paths)
    links = jnp.exp(1.0j * jnp.asarray((0.1, -0.2, 0.3)))[:, None, None]
    bilocal = jnp.asarray((0.8 + 0.1j, 0.4 - 0.2j))[:, None, None]
    phases = jnp.asarray([[1.0, 1.0j], [1.0, -1.0j]])
    weights = jnp.asarray((0.5, 0.5))
    vertex_elements = jnp.exp(1.0j * jnp.asarray((0.2, -0.4, 0.1)))[:, None, None]

    evidence = gauge_covariance_evidence(
        plan,
        links,
        bilocal,
        phases,
        weights,
        vertex_elements,
    )

    assert bool(evidence.covariant)
    assert evidence.covariance_residual < 1.0e-12
    assert evidence.transformed.frame_id == plan.quantum_support.frame_id
    assert evidence.transformed.unit_contract_id == plan.quantum_support.unit_contract_id
    np.testing.assert_array_equal(
        evidence.transformed.frame_token, plan.quantum_support.frame.frame_token
    )
    assert (
        evidence.transformed.frame_realization_id
        == plan.quantum_support.frame.realization_id()
    )


def test_nonabelian_wigner_transport_refuses_complex_fundamental_without_real_owner():
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    group = SpecialUnitaryGroup(2)
    link_space = MatrixGaugeLinkSpace(topology, group)
    representation = FundamentalGaugeRepresentation(group)

    with pytest.raises(TypeError, match="fails closed"):
        GaugeCovariantWignerPlan(
            _support(),
            link_space,
            representation,
            _paths(topology),
        )
