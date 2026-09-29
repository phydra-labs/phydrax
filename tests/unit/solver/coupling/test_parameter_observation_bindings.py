#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Parameter and observation bindings of prepared spatial coupled problems.

The FE–VEM heat-conduction plate of ``tests._support.coupled_plate`` has an
x-only analytic temperature for constant conductivities; it is the independent
reference of every observed value.
"""

from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import FacetTraceRule
from phydrax.solver import coupling as cpl
from tests._support import coupled_plate as cp


measurement = phx.measurement
TRUTH = {"conductivity-left": 1.3, "conductivity-right": 0.7, "heat-flux": 0.5}
CONTRACT = phx.SpatialCoordinateContract(phx.units.METER)
TEMPERATURE = measurement.QuantitySpec(
    "plate", "temperature", "temperature", phx.units.KELVIN, "temperature"
)
POINT = measurement.SamplingSemantics(measurement.SpatialSamplingKind.POINT)
ONE_SAMPLE = measurement.IndexSampleSupport((1,), ("wall",))
RULE = FacetTraceRule(points=2)


def _values(**overrides: float) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(overrides.get(key, value)) for key, value in TRUTH.items()}


def _sensors(points: tuple[tuple[float, float], ...]) -> measurement.PointSampleSupport:
    coordinates = np.asarray([(x, y, 0.0) for x, y in points])
    return measurement.PointSampleSupport(
        coordinates, tuple(f"s{index}" for index in range(len(points))), CONTRACT
    )


def _plan(
    plate: cp.CoupledPlate,
    parameters: tuple[cpl.ParameterBinding, ...],
    observations: tuple[cpl.AbstractObservationBinding, ...] = (),
) -> cpl.CoupledProblemPlan:
    return cpl.CoupledProblemPlan(
        "plate",
        components=(plate.triangles, plate.polygons),
        bindings=(plate.binding,),
        laws=(plate.law,),
        parameters=parameters,
        observations=observations,
    )


@pytest.fixture(scope="module")
def plate() -> cp.CoupledPlate:
    return cp.build_plate(0)


@pytest.fixture(scope="module")
def prepared(plate: cp.CoupledPlate) -> cpl.PreparedCoupledProblem:
    millikelvin = phx.units.UnitDefinition(
        "mK",
        phx.units.KELVIN.dimension,
        phx.units.KELVIN.reference_system_id,
        Fraction(1, 1000),
    )
    observations = (
        cpl.FieldPointObservation(
            "masked-sensors",
            "triangles",
            "u",
            quantity=TEMPERATURE,
            support=_sensors(((0.25, 0.5), (1.6, 0.5), (0.75, 0.25))),
            sampling=POINT,
            field_unit=phx.units.KELVIN,
            coverage="masked",
        ),
        cpl.FieldPointObservation(
            "millikelvin-sensor",
            "polygons",
            "u",
            quantity=measurement.QuantitySpec(
                "plate", "temperature", "temperature", millikelvin, "temperature"
            ),
            support=_sensors(((1.5, 0.5),)),
            sampling=POINT,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldBoundaryObservation(
            "right-wall-integral",
            "polygons",
            "u",
            plate.right_wall,
            statistic="integral",
            rule=RULE,
            quantity=measurement.QuantitySpec(
                "plate",
                "wall-temperature-integral",
                "temperature-length",
                phx.units.derived_unit(
                    "K m", ((phx.units.KELVIN, 1), (phx.units.METER, 1))
                ),
                "wall-temperature-integral",
            ),
            support=ONE_SAMPLE,
            sampling=measurement.SamplingSemantics(
                measurement.SpatialSamplingKind.PATH_INTEGRAL
            ),
            field_unit=phx.units.KELVIN,
            length_unit=phx.units.METER,
        ),
        cpl.FieldFluxObservation(
            "left-wall-heat",
            "triangles",
            "u",
            plate.left_wall,
            rule=RULE,
            quantity=measurement.QuantitySpec(
                "plate", "wall-heat", "heat-rate-per-depth", cp.LINE_HEAT_UNIT, "heat"
            ),
            support=ONE_SAMPLE,
            sampling=measurement.SamplingSemantics(
                measurement.SpatialSamplingKind.PATH_INTEGRAL
            ),
            reaction_unit=cp.LINE_HEAT_UNIT,
        ),
    )
    plan = _plan(plate, cp.parameter_bindings(), observations)
    return cpl.prepare_coupled_problem(
        plan, interface_owners=(plate.cover,), parameters=_values()
    )


@pytest.mark.parametrize(
    ("mode", "admitted"),
    (
        ("mathematical", ("conductivity-left", "conductivity-right", "heat-flux")),
        ("rhs-only", ("heat-flux",)),
        ("algorithmic", ()),
        ("none", ()),
    ),
)
def test_derivative_capability_follows_the_solve_differentiation_mode(
    prepared: cpl.PreparedCoupledProblem,
    mode: phx.linalg.DifferentiationMode,
    admitted: tuple[str, ...],
) -> None:
    capability = prepared.derivative_capability(cp.dense_policy(mode))
    refused = dict(capability.refused)

    assert tuple(name for name, _ in capability.admitted) == admitted
    assert {"geometry", "quadrature", "topology"} <= set(refused)
    assert set(refused) | set(admitted) >= set(TRUTH)
    assert (capability.derivative_contract.route is phx.DerivativeRoute.IMPLICIT) == bool(
        admitted
    )
    if mode == "rhs-only":
        assert "rhs-only" in refused["conductivity-left"]


def test_solve_binds_every_refresh_parameter_explicitly(
    prepared: cpl.PreparedCoupledProblem,
) -> None:
    missing = {key: value for key, value in _values().items() if key != "heat-flux"}
    with pytest.raises(ValueError, match="must be bound at every solve"):
        cpl.solve_coupled_problem(prepared, parameters=missing)
    with pytest.raises(ValueError, match="not bound by the plan"):
        cpl.solve_coupled_problem(
            prepared, parameters={**_values(), "emissivity": jnp.asarray(0.3)}
        )
    with pytest.raises(ValueError, match="event shape"):
        cpl.solve_coupled_problem(
            prepared, parameters={**_values(), "heat-flux": jnp.ones((2,))}
        )
    with pytest.raises(ValueError, match="bound parameters"):
        cpl.solve_coupled_problem(
            prepared,
            arguments={"polygons": {"heat-flux": 1.0}},
            parameters=_values(),
        )


def test_observations_carry_their_measurement_semantics(
    prepared: cpl.PreparedCoupledProblem,
) -> None:
    solution = cpl.solve_coupled_problem(
        prepared, parameters=_values(), policy=cp.dense_policy()
    )
    exact = cp.exact_temperature(
        np.asarray([[0.25, 0.5], [0.75, 0.25], [1.5, 0.5], [2.0, 0.5]]), *TRUTH.values()
    )
    masked = solution.observation("masked-sensors")
    millikelvin = solution.observation("millikelvin-sensor")
    integral = solution.observation("right-wall-integral")
    heat = solution.observation("left-wall-heat")

    assert bool(solution.accepted)
    # The second sensor lies in the other region: reported invalid, not zero-valued.
    assert masked.valid_mask.tolist() == [True, False, True]
    np.testing.assert_allclose(
        np.asarray(masked.values)[[0, 2]], exact[:2], rtol=0.0, atol=2.0e-2
    )
    np.testing.assert_allclose(millikelvin.values, 1000.0 * exact[2], rtol=2.0e-2)
    assert millikelvin.unit_id != masked.unit_id
    # The right wall has unit length: its temperature integral is the wall value.
    np.testing.assert_allclose(integral.values, exact[3], rtol=1.0e-9)
    np.testing.assert_allclose(
        heat.values, cp.exact_left_wall_heat(TRUTH["heat-flux"]), rtol=1.0e-9
    )
    assert prepared.observation("masked-sensors").approximation == "exact"
    assert prepared.observation("millikelvin-sensor").approximation == "h1-projection"


@pytest.mark.parametrize(
    ("kind", "statistic"),
    (
        (measurement.SpatialSamplingKind.POINT, "average"),
        (measurement.SpatialSamplingKind.SURFACE_AVERAGE, "integral"),
        (measurement.SpatialSamplingKind.PATH_INTEGRAL, "average"),
    ),
)
def test_boundary_statistics_are_not_identified_with_other_samplings(
    plate: cp.CoupledPlate,
    kind: measurement.SpatialSamplingKind,
    statistic: cpl.BoundaryStatistic,
) -> None:
    with pytest.raises(ValueError, match="distinct measurements"):
        cpl.FieldBoundaryObservation(
            "wall",
            "polygons",
            "u",
            plate.right_wall,
            statistic=statistic,
            rule=RULE,
            quantity=TEMPERATURE,
            support=ONE_SAMPLE,
            sampling=measurement.SamplingSemantics(kind),
            field_unit=phx.units.KELVIN,
            length_unit=phx.units.METER,
        )


def test_densities_contents_and_values_are_not_identified(
    plate: cp.CoupledPlate,
) -> None:
    path = measurement.SamplingSemantics(measurement.SpatialSamplingKind.PATH_INTEGRAL)
    with pytest.raises(ValueError, match="never identified"):
        cpl.FieldFluxObservation(
            "heat",
            "triangles",
            "u",
            plate.left_wall,
            rule=RULE,
            quantity=measurement.QuantitySpec(
                "plate", "wall-flux", "heat-flux", cp.HEAT_FLUX_UNIT, "flux"
            ),
            support=ONE_SAMPLE,
            sampling=path,
            reaction_unit=cp.LINE_HEAT_UNIT,
        )
    with pytest.raises(ValueError, match="never identified"):
        cpl.FieldBoundaryObservation(
            "wall",
            "polygons",
            "u",
            plate.right_wall,
            statistic="integral",
            rule=RULE,
            quantity=TEMPERATURE,
            support=ONE_SAMPLE,
            sampling=path,
            field_unit=phx.units.KELVIN,
            length_unit=phx.units.METER,
        ).prepare({"polygons": plate.polygons})
    with pytest.raises(ValueError, match="never identified"):
        cpl.FieldPointObservation(
            "gradient",
            "triangles",
            "u",
            quantity=TEMPERATURE,
            support=_sensors(((0.5, 0.5),)),
            sampling=POINT,
            field_unit=phx.units.KELVIN,
            derivative=(1, 0),
            length_unit=phx.units.METER,
        )


def test_trace_samples_must_sit_at_the_trace_sites(plate: cp.CoupledPlate) -> None:
    trace = plate.polygons.prepare_side_trace("u", plate.right_wall, rule=RULE)
    sites = np.asarray(trace.sites).reshape((-1, 2))
    shifted = np.column_stack((sites[:, 0], sites[:, 1] + 1.0e-3))

    def observation(points: np.ndarray) -> cpl.FieldBoundaryObservation:
        return cpl.FieldBoundaryObservation(
            "wall-trace",
            "polygons",
            "u",
            plate.right_wall,
            statistic="trace",
            rule=RULE,
            quantity=TEMPERATURE,
            support=_sensors(tuple((x, y) for x, y in points)),
            sampling=POINT,
            field_unit=phx.units.KELVIN,
        )

    prepared_trace = observation(sites).prepare({"polygons": plate.polygons})
    assert prepared_trace.approximation == "exact"
    with pytest.raises(ValueError, match="trace's facet quadrature sites"):
        observation(shifted).prepare({"polygons": plate.polygons})


def _linear_temperature(plate: cp.CoupledPlate) -> dict[tuple[str, str], jnp.ndarray]:
    """``u = 3 x + 1`` K on the triangles: exact in P1, with gradient (3, 0) K/m."""
    return {("triangles", "u"): jnp.asarray(3.0 * plate.triangle_nodes[:, 0] + 1.0)}


def test_point_derivatives_are_in_the_support_coordinate_unit(
    plate: cp.CoupledPlate,
) -> None:
    gradient = measurement.QuantitySpec(
        "plate",
        "temperature-gradient",
        "temperature-gradient",
        phx.units.derived_unit("K/m", ((phx.units.KELVIN, 1), (phx.units.METER, -1))),
        "temperature-gradient",
    )

    def observation(
        length_unit: phx.units.UnitDefinition | None,
    ) -> cpl.FieldPointObservation:
        return cpl.FieldPointObservation(
            "gradient",
            "triangles",
            "u",
            quantity=gradient,
            support=_sensors(((0.3, 0.4),)),
            sampling=POINT,
            field_unit=phx.units.KELVIN,
            derivative=(1, 0),
            length_unit=length_unit,
        )

    # The support coordinates are metres: the derivative is per metre.
    for declared in (None, phx.units.METER):
        prepared = observation(declared).prepare({"triangles": plate.triangles})
        predicted = prepared.evaluate(_linear_temperature(plate), {})
        np.testing.assert_allclose(predicted.values, [3.0], rtol=1.0e-12)
    with pytest.raises(ValueError, match="coordinate contract"):
        observation(phx.units.MILLIMETER)


def test_boundary_length_unit_follows_the_point_support_contract(
    plate: cp.CoupledPlate,
) -> None:
    trace = plate.polygons.prepare_side_trace("u", plate.right_wall, rule=RULE)
    sites = np.asarray(trace.sites).reshape((-1, 2))
    with pytest.raises(ValueError, match="coordinate contract"):
        cpl.FieldBoundaryObservation(
            "wall-trace",
            "polygons",
            "u",
            plate.right_wall,
            statistic="trace",
            rule=RULE,
            quantity=TEMPERATURE,
            support=_sensors(tuple((x, y) for x, y in sites)),
            sampling=POINT,
            field_unit=phx.units.KELVIN,
            length_unit=phx.units.MILLIMETER,
        )


def test_inactive_support_samples_are_never_predicted(plate: cp.CoupledPlate) -> None:
    # An inactive sample may carry non-finite coordinates; a finite one inside the
    # component is still not a measurement and must not be reported valid.
    support = measurement.PointSampleSupport(
        np.asarray([[0.25, 0.5, 0.0], [np.nan, np.nan, np.nan], [0.75, 0.25, 0.0]]),
        ("s0", "s1", "s2"),
        CONTRACT,
        active_mask=np.asarray([True, False, False]),
    )

    def observation(
        coverage: phx.discretization.FieldQueryCoverage,
    ) -> cpl.FieldPointObservation:
        return cpl.FieldPointObservation(
            "partially-active",
            "triangles",
            "u",
            quantity=TEMPERATURE,
            support=support,
            sampling=POINT,
            field_unit=phx.units.KELVIN,
            coverage=coverage,
        )

    prepared = observation("masked").prepare({"triangles": plate.triangles})
    predicted = prepared.evaluate(_linear_temperature(plate), {})

    assert predicted.valid_mask.tolist() == [True, False, False]
    np.testing.assert_allclose(np.asarray(predicted.values)[0], 1.75, rtol=1.0e-12)
    assert not prepared.complete
    with pytest.raises(ValueError, match="inactive samples"):
        observation("complete")


def test_parameter_declarations_are_validated(plate: cp.CoupledPlate) -> None:
    port = cp.conductivity_port("left")
    target = (cpl.RuntimeInput("triangles", "conductivity"),)
    with pytest.raises(ValueError, match="enters the coupled map as"):
        cpl.ParameterBinding(
            "kappa",
            port,
            targets=target,
            role="coefficient",
            derivative=phx.DerivativeSurface.SOLVER_ARGUMENT,
        )
    with pytest.raises(ValueError, match="admits no derivative surface"):
        cpl.ParameterBinding(
            "kappa",
            port,
            targets=target,
            role="coefficient",
            derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
            change="reprepare",
        )
    left, right, flux = cp.parameter_bindings()
    twin = cpl.ParameterBinding(
        "conductivity-twin", port, targets=target, role="coefficient"
    )
    with pytest.raises(ValueError, match="both bind runtime input"):
        cpl.prepare_coupled_problem(
            _plan(plate, (left, right, flux, twin)),
            interface_owners=(plate.cover,),
            parameters={**_values(), "conductivity-twin": jnp.asarray(1.0)},
        )
    with pytest.raises(ValueError, match="reference value of every declared"):
        cpl.prepare_coupled_problem(
            _plan(plate, (left, right, flux)),
            interface_owners=(plate.cover,),
            parameters={"heat-flux": jnp.asarray(0.5)},
        )


def test_a_solver_argument_entering_the_operator_is_refused(
    plate: cp.CoupledPlate,
) -> None:
    left, _, flux = cp.parameter_bindings()
    mislabeled = cpl.ParameterBinding(
        "conductivity-right",
        cp.conductivity_port("right"),
        targets=(cpl.RuntimeInput("polygons", "conductivity"),),
        role="boundary",
        derivative=phx.DerivativeSurface.SOLVER_ARGUMENT,
    )
    with pytest.raises(ValueError, match="enters the coupled operator"):
        cpl.prepare_coupled_problem(
            _plan(plate, (left, mislabeled, flux)),
            interface_owners=(plate.cover,),
            parameters=_values(),
        )


def test_reprepare_parameters_are_fixed_structure(plate: cp.CoupledPlate) -> None:
    left, right, _ = cp.parameter_bindings()
    fixed_flux = cpl.ParameterBinding(
        "heat-flux",
        cp.HEAT_FLUX_PORT,
        targets=(cpl.RuntimeInput("polygons", "heat-flux"),),
        role="boundary",
        change="reprepare",
    )
    prepared = cpl.prepare_coupled_problem(
        _plan(plate, (left, right, fixed_flux)),
        interface_owners=(plate.cover,),
        parameters=_values(**{"heat-flux": 1.5}),
    )
    refresh = {key: value for key, value in _values().items() if key != "heat-flux"}
    solution = cpl.solve_coupled_problem(
        prepared, parameters=refresh, policy=cp.dense_policy()
    )
    reference = cp.exact_temperature(
        plate.triangle_nodes, TRUTH["conductivity-left"], TRUTH["conductivity-right"], 1.5
    )

    np.testing.assert_allclose(
        solution.field("triangles", "u"), reference, rtol=0.0, atol=2.0e-2
    )
    assert "re-preparation" in dict(solution.derivative_capability.refused)["heat-flux"]
    with pytest.raises(ValueError, match="prepare the coupled problem again"):
        cpl.solve_coupled_problem(prepared, parameters=_values())
