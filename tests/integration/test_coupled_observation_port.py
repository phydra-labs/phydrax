#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admission of observation bindings into a coupled-transition observation port.

The plant is the FE-VEM heat transient of ``examples/_coupled_heat_transient.py``
with extra observation bindings. The port's control-output and state-space
location ABIs are plain value vectors, so a binding must predict every sample,
and it must observe the transient state it is evaluated on.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from examples import _coupled_heat_transient as heat
from phydrax.discretization import FacetTraceRule
from phydrax.solver import coupling as cpl


CELLS = 2
LINE_HEAT_UNIT = phx.units.derived_unit("W/m", ((heat.WATT, 1), (phx.units.METER, -1)))


def _points(
    binding_id: str,
    points: tuple[tuple[float, float], ...],
    coverage: phx.discretization.FieldQueryCoverage,
) -> cpl.FieldPointObservation:
    return cpl.FieldPointObservation(
        binding_id,
        "conductor",
        "u",
        quantity=heat.TEMPERATURE,
        support=phx.measurement.PointSampleSupport(
            np.asarray([(x, y, 0.0) for x, y in points]),
            tuple(f"{binding_id}-{index}" for index in range(len(points))),
            heat.CONTRACT,
        ),
        sampling=heat.POINT,
        field_unit=phx.units.KELVIN,
        coverage=coverage,
    )


@pytest.fixture(scope="module")
def transition() -> cpl.PreparedCoupledTransition:
    _, interface = heat._conductor(CELLS)
    extra = (
        # The second sample lies in the insulator: never located on the conductor.
        _points("partly-outside", ((0.55, 0.45), (1.5, 0.5)), "masked"),
        _points("all-inside", ((0.55, 0.45), (0.25, 0.75)), "masked"),
        cpl.FieldFluxObservation(
            "interface-heat",
            "conductor",
            "u",
            interface,
            rule=FacetTraceRule(points=2),
            quantity=phx.measurement.QuantitySpec(
                "coupled-heat",
                "interface-heat",
                "heat-rate-per-depth",
                LINE_HEAT_UNIT,
                "heat",
            ),
            support=phx.measurement.IndexSampleSupport((1,), ("interface",)),
            sampling=phx.measurement.SamplingSemantics(
                phx.measurement.SpatialSamplingKind.PATH_INTEGRAL
            ),
            reaction_unit=LINE_HEAT_UNIT,
        ),
    )
    sensors = heat.sensor_bindings()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(heat, "sensor_bindings", lambda: (*sensors, *extra))
        transient = heat.coupled_heat_transient(CELLS)
    return cpl.prepare_coupled_transition(
        transient,
        step=0.1,
        policy=heat.implicit_euler_policy(),
        parameters={heat.CONTROL: jnp.zeros((2,))},
    )


def test_port_refuses_a_binding_with_never_valid_samples(
    transition: cpl.PreparedCoupledTransition,
) -> None:
    with pytest.raises(ValueError, match="never valid"):
        transition.observation_port(("conductor-sensor", "partly-outside"))


def test_port_admits_a_masked_binding_whose_samples_are_all_located(
    transition: cpl.PreparedCoupledTransition,
) -> None:
    port = transition.observation_port(("all-inside",))
    state = jnp.linspace(0.0, 1.0, transition.state_size)
    parameters = {heat.CONTROL: jnp.asarray([0.4, -0.3])}
    (field,) = port.measure(0.0, state, parameters)

    assert port.observation_size == 2
    assert bool(jnp.all(field.valid_mask))
    np.testing.assert_array_equal(port.values(0.0, state, parameters), field.values)


def test_port_refuses_a_steady_flux_content_binding(
    transition: cpl.PreparedCoupledTransition,
) -> None:
    with pytest.raises(ValueError, match="capacity action"):
        transition.observation_port(("interface-heat",))
