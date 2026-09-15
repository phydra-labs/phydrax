import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.semiconductor import quantum as semiconductor_quantum
from phydrax.applications.semiconductor._detector import (
    DetectorBiasElectrostaticPlan,
    DetectorBiasElectrostaticResult,
    DetectorElectrode,
    DetectorResourcePolicy,
    DetectorWeightingFieldPlan,
    DetectorWeightingFieldResult,
    SemiconductorDetectorPlan,
)
from phydrax.applications.semiconductor._detector_response import (
    DetectorTrajectoryRoute,
    PrescribedShockleyRamoPlan,
)
from phydrax.applications.semiconductor._production_qualification import (
    semiconductor_candidate_profile,
    semiconductor_quantum_transport_campaign,
)
from phydrax.applications.semiconductor._quantities import (
    PERMITTIVITY_UNIT,
    SQUARE_METER,
)
from phydrax.discretization import (
    NonuniformCellAxisSpec,
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.dynamics import StateLayout, TrajectoryData
from phydrax.units import METER


EPSILON = 11.7 * 8.8541878128e-12


def _resources(**changes):
    values = dict(
        maximum_nodes=256,
        maximum_edges=768,
        maximum_electrodes=8,
        maximum_linear_iterations=2000,
        maximum_trajectory_cases=8,
        maximum_trajectory_samples=256,
        maximum_interpolation_routes=2048,
    )
    values.update(changes)
    return DetectorResourcePolicy(**values)


def _interval_bridge(edges=8, *, lower=0.0, upper=1.0):
    grid = TensorGridPlan(
        (UniformCellAxisSpec(edges, periodic=False),), axis_names=("x",)
    ).prepare(jnp.asarray([[lower], [upper]]))
    return StructuredCochainBridge(grid)


def _endpoint_electrodes(bridge):
    x = np.asarray(bridge.cochain.coordinates[0])[:, 0]
    return (
        DetectorElectrode("inner", np.isclose(x, x[0])),
        DetectorElectrode("outer", np.isclose(x, x[-1])),
    )


def _parallel_detector(*, resources=None, length=2.0, area=3.0):
    bridge = _interval_bridge(lower=0.0, upper=length)
    return SemiconductorDetectorPlan(
        bridge,
        _endpoint_electrodes(bridge),
        _resources() if resources is None else resources,
        permittivity=EPSILON,
        node_transverse_measure=area,
        edge_transverse_measure=area,
        permittivity_unit=PERMITTIVITY_UNIT,
        transverse_unit=SQUARE_METER,
    )


def test_parallel_plate_bias_and_weighting_are_distinct_and_capacitance_closes():
    length, area = 2.0, 3.0
    detector = _parallel_detector(length=length, area=area)
    x = detector.bridge.cochain.coordinates[0][:, 0]
    zero = jnp.zeros((detector.node_count,))
    bias = DetectorBiasElectrostaticPlan(detector, jnp.asarray([5.0, -1.0])).solve(zero)
    weighting_plan = DetectorWeightingFieldPlan(detector)
    weighting = weighting_plan.solve()

    assert isinstance(bias, DetectorBiasElectrostaticResult)
    assert isinstance(weighting, DetectorWeightingFieldResult)
    assert not isinstance(bias, DetectorWeightingFieldResult)
    assert not isinstance(weighting, DetectorBiasElectrostaticResult)
    np.testing.assert_allclose(bias.electrostatic.potential, 5.0 - 6.0 * x / length)
    np.testing.assert_allclose(weighting.potentials[0], 1.0 - x / length)
    np.testing.assert_allclose(weighting.potentials[1], x / length)
    expected = EPSILON * area / length
    np.testing.assert_allclose(
        weighting.capacitance,
        [[expected, -expected], [-expected, expected]],
        rtol=2e-8,
        atol=1e-20,
    )
    assert bias.successful
    assert weighting.evidence.certified
    assert weighting.evidence.capacitance_reciprocal
    assert weighting.evidence.complete_electrode_charge_closed
    np.testing.assert_allclose(bias.charge_closure_defect, 0.0, atol=1e-20)

    route = DetectorTrajectoryRoute(StateLayout((1,)), (0,))
    with pytest.raises(TypeError, match="bias results cannot substitute"):
        PrescribedShockleyRamoPlan(weighting_plan, bias, route)


def test_coaxial_logarithmic_weighting_and_capacitance_are_recovered():
    inner, outer, length = 0.4, 2.5, 1.7
    radii = np.geomspace(inner, outer, 17)
    normalized = (radii - inner) / (outer - inner)
    grid = TensorGridPlan(
        (NonuniformCellAxisSpec(normalized, periodic=False),),
        axis_names=("radius",),
    ).prepare(jnp.asarray([[inner], [outer]]))
    bridge = StructuredCochainBridge(grid)
    node_radius = np.asarray(bridge.cochain.coordinates[0])[:, 0]
    edge_log_mean = np.diff(radii) / np.diff(np.log(radii))
    detector = SemiconductorDetectorPlan(
        bridge,
        _endpoint_electrodes(bridge),
        _resources(),
        permittivity=EPSILON,
        node_transverse_measure=2.0 * np.pi * node_radius * length,
        edge_transverse_measure=2.0 * np.pi * edge_log_mean * length,
        transverse_unit=SQUARE_METER,
    )
    weighting = DetectorWeightingFieldPlan(detector).solve()

    expected_potential = np.log(outer / node_radius) / np.log(outer / inner)
    expected_capacitance = 2.0 * np.pi * EPSILON * length / np.log(outer / inner)
    np.testing.assert_allclose(weighting.potentials[0], expected_potential, atol=2e-8)
    np.testing.assert_allclose(
        weighting.capacitance[0, 0], expected_capacitance, rtol=2e-8
    )
    assert weighting.evidence.certified


def _segmented_detector(*, complete=True):
    grid = TensorGridPlan(
        (
            UniformCellAxisSpec(4, periodic=False),
            UniformCellAxisSpec(4, periodic=False),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    bridge = StructuredCochainBridge(grid)
    points = np.asarray(bridge.cochain.coordinates[0])
    x, y = points[:, 0], points[:, 1]
    bottom_left = np.isclose(y, 0.0) & (x <= 0.5)
    bottom_right = np.isclose(y, 0.0) & (x > 0.5)
    top = np.isclose(y, 1.0)
    sides = (
        (np.isclose(x, 0.0) | np.isclose(x, 1.0)) & ~bottom_left & ~bottom_right & ~top
    )
    electrodes = [
        DetectorElectrode("bottom-left", bottom_left),
        DetectorElectrode("bottom-right", bottom_right),
        DetectorElectrode("top", top),
    ]
    if complete:
        electrodes.append(DetectorElectrode("side-guard", sides))
    return SemiconductorDetectorPlan(
        bridge,
        tuple(electrodes),
        _resources(),
        permittivity=EPSILON,
        node_transverse_measure=0.2,
        edge_transverse_measure=0.2,
        transverse_unit=METER,
    )


def test_segmented_complete_basis_and_prescribed_response_close_without_motion():
    detector = _segmented_detector()
    weighting_plan = DetectorWeightingFieldPlan(detector)
    weighting = weighting_plan.solve()
    layout = StateLayout((2,), axes=("position",), component_names=("x", "y"))
    states = jnp.asarray(
        [[0.15, 0.15], [0.30, 0.35], [0.48, 0.55], [0.70, 0.75], [0.85, 0.85]]
    )
    trajectory = TrajectoryData(
        jnp.linspace(0.0, 4.0e-9, states.shape[0]),
        states,
        state_layout=layout,
        source_id="segmented-prescribed-path",
    )
    before = np.asarray(trajectory.states).copy()
    response = PrescribedShockleyRamoPlan(
        weighting_plan,
        weighting,
        DetectorTrajectoryRoute(layout, (0, 1)),
    ).evaluate(trajectory, -1.602176634e-19)

    assert weighting.evidence.certified
    assert response.successful
    np.testing.assert_allclose(
        response.integrated_current,
        response.endpoint_charge_change,
        rtol=1e-12,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        jnp.sum(response.interval_current, axis=-1), 0.0, atol=1e-22
    )
    np.testing.assert_array_equal(trajectory.states, before)
    assert response.trajectory_dataset_id == trajectory.dataset_id
    assert response.sign_convention == (
        "Q_induced=-q*phi_w; i_into_electrode=dQ_induced/dt"
    )


def test_partition_certificate_requires_complete_dirichlet_electrodes():
    detector = _segmented_detector(complete=False)
    weighting_plan = DetectorWeightingFieldPlan(detector)
    weighting = weighting_plan.solve()
    layout = StateLayout((2,))

    assert not detector.complete_dirichlet_electrodes
    assert weighting.evidence.unassigned_boundary_count > 0
    assert not weighting.evidence.partition_of_unity_certified
    assert not weighting.evidence.complete_electrode_charge_closed
    assert not weighting.evidence.certified
    with pytest.raises(ValueError, match="complete-electrode weighting basis"):
        PrescribedShockleyRamoPlan(
            weighting_plan,
            weighting,
            DetectorTrajectoryRoute(layout, (0, 1)),
        )


def test_route_and_resource_failures_are_explicit_and_preallocation_bounded():
    bridge = _interval_bridge(edges=8)
    with pytest.raises(ValueError, match="node count"):
        SemiconductorDetectorPlan(
            bridge,
            _endpoint_electrodes(bridge),
            _resources(maximum_nodes=2),
            permittivity=EPSILON,
            node_transverse_measure=1.0,
            edge_transverse_measure=1.0,
            transverse_unit=SQUARE_METER,
        )

    detector = _parallel_detector(
        resources=_resources(
            maximum_trajectory_samples=3,
            maximum_interpolation_routes=16,
        )
    )
    weighting_plan = DetectorWeightingFieldPlan(detector)
    weighting = weighting_plan.solve()
    layout = StateLayout((1,))
    response_plan = PrescribedShockleyRamoPlan(
        weighting_plan,
        weighting,
        DetectorTrajectoryRoute(layout, (0,)),
    )
    too_long = TrajectoryData(
        jnp.linspace(0.0, 1.0, 4),
        jnp.linspace(0.1, 0.9, 4)[:, None],
        state_layout=layout,
        source_id="resource-overflow-path",
    )
    with pytest.raises(ValueError, match="sample count"):
        response_plan.evaluate(too_long, 1.0)

    admitted = _parallel_detector()
    admitted_weighting_plan = DetectorWeightingFieldPlan(admitted)
    admitted_response_plan = PrescribedShockleyRamoPlan(
        admitted_weighting_plan,
        admitted_weighting_plan.solve(),
        DetectorTrajectoryRoute(layout, (0,)),
    )
    outside = TrajectoryData(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([[0.2], [3.0]]),
        state_layout=layout,
        source_id="out-of-support-path",
    )
    invalid = admitted_response_plan.evaluate(outside, 1.0)
    assert not invalid.route_valid
    assert not invalid.successful


def test_existing_chain_transport_profiles_bind_real_routes_and_exact_nonclaims():
    names_and_symbols = {
        "semiconductor.quantum.chain-landauer.v1": (
            semiconductor_quantum.integrate_coherent,
            "no-displacement-current-or-interacting-transport",
        ),
        "semiconductor.quantum.chain-coherent-ac.v1": (
            semiconductor_quantum.finite_frequency_quantum_response,
            "no-scba-vertex-or-ballistic-dc-from-adiabatic-rate",
        ),
        "semiconductor.quantum.chain-finite-lead-transient.v1": (
            semiconductor_quantum.solve_quantum_transient,
            "no-lindblad-or-general-interacting-transient-negf",
        ),
        "semiconductor.quantum.chain-optical-phonon-scba.v1": (
            semiconductor_quantum.solve_phonon_transport,
            "no-hartree-tadpole-numerical-eta-or-vertex-corrections",
        ),
    }
    profiles = {name: semiconductor_candidate_profile(name) for name in names_and_symbols}
    for name, (symbol, nonclaim) in names_and_symbols.items():
        profile = profiles[name]
        support = dict(profile.support_tuples[0].attributes)
        assert callable(symbol)
        assert support["code_id"].endswith(symbol.__name__)
        assert support["nonclaim"] == nonclaim
        assert not ({"candidate", "maturity", "released", "status"} & set(support))
        assert not profile.released
        assert profile.release_evidence == ()

    landauer = profiles["semiconductor.quantum.chain-landauer.v1"]
    for name in (
        "semiconductor.quantum.chain-coherent-ac.v1",
        "semiconductor.quantum.chain-finite-lead-transient.v1",
        "semiconductor.quantum.chain-optical-phonon-scba.v1",
    ):
        dependency = profiles[name].dependencies
        assert len(dependency) == 1
        assert dependency[0].profile_id == landauer.profile_id
        assert (
            dependency[0].support_tuple_id == landauer.support_tuples[0].support_tuple_id
        )

    for removed_placeholder in (
        "semiconductor.detector.carrier-packet-drift-diffusion.v1",
        "semiconductor.detector.carrier-packet-trapping.v1",
        "semiconductor.detector.carrier-packet-avalanche.v1",
        "semiconductor.detector.circuit-coupled-response.v1",
    ):
        with pytest.raises(ValueError, match="Unknown semiconductor capability"):
            semiconductor_candidate_profile(removed_placeholder)


def test_chain_transport_campaign_keeps_calibration_and_locked_units_disjoint():
    campaign = semiconductor_quantum_transport_campaign()
    roles = {role.name: set(role.case_ids) for role in campaign.roles}
    calibration = roles["calibration"]
    locked = roles["locked_evaluation"]

    assert calibration == {
        "semiconductor-landauer-transparent-calibration",
        "semiconductor-landauer-resonant-calibration",
    }
    assert locked == {
        "semiconductor-coherent-ac-locked",
        "semiconductor-finite-lead-transient-locked",
        "semiconductor-optical-phonon-scba-locked",
        "semiconductor-undetermined-bound-state-refusal",
        "semiconductor-quantum-resource-refusal",
    }
    assert calibration.isdisjoint(locked)
    assert len(campaign.independent_unit_ids) == len(campaign.case_ids)
