from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from phydrax.closure_data import (
    FlowStateSchema,
    LearnedStressFeatureSchema,
    LearnedStressOutputContract,
)
from phydrax.units import DIMENSIONLESS, ENERGY, MASS, TIME, VELOCITY, VOLUME


def _flow_schema(
    *, units: Any = ("kg/m^3", "m/s", "m/s", "1", "J/kg"), scales: Any = None
) -> Any:
    return FlowStateSchema(
        ("rho", "u", "v", "species", "enthalpy"),
        units,
        (1.2, 10.0, 10.0, 1.0, 1000.0) if scales is None else scales,
        density_name="rho",
        velocity_names=("u", "v"),
        species_names=("species",),
        enthalpy_name="enthalpy",
    )


def _feature_schema(
    *, units: Any = ("1/s", "1/s"), shape: Any = (4, 2), name: Any = "resolved-gradient"
) -> Any:
    return LearnedStressFeatureSchema(
        name=name,
        component_names=("s_xx", "s_xy"),
        component_units=units,
        shape=shape,
        dtype=jnp.float32,
        flow_schema_id="flow-schema",
    )


def _output_contract(
    *, units: Any = "(m/s)^2", shape: Any = (2, 3, 3), discretization: Any = "mesh-32"
) -> Any:
    return LearnedStressOutputContract(
        shape=shape,
        dtype=jnp.float32,
        units=units,
        target_id="deviatoric-specific-stress-target",
        filter_id="box-filter",
        discretization_id=discretization,
        regime="constant-density-incompressible",
    )


def test_closure_ports_scenario_1() -> None:
    schema = _flow_schema()
    port = schema.value_port(representation="nondimensional")
    assert port.semantic_id == f"flow-state-schema:{schema.schema_id}"
    assert port.event_shape == (5,)
    assert port.component_ids == ("rho", "u", "v", "species", "enthalpy")
    assert port.dimensions == (
        MASS / VOLUME,
        VELOCITY,
        VELOCITY,
        DIMENSIONLESS,
        ENERGY / MASS,
    )
    assert port.representation == "nondimensional"
    assert port.normalization_id is not None
    assert port.variance == "neutral"
    assert (port.space_id, port.frame_id, port.axis_keys) == (None, None, None)
    first = _flow_schema().value_port(representation="nondimensional")
    second = _flow_schema().value_port(representation="nondimensional")
    assert first == second
    assert first.port_id == second.port_id
    schema = _flow_schema(units=("kg/m^3", "m/s", "m/s", "1", "BTU/lb"))
    with pytest.raises(ValueError, match="BTU/lb") as error:
        schema.value_port(representation="dimensional")
    assert schema.schema_id in str(error.value)
    assert "'enthalpy'" in str(error.value)
    schema = _flow_schema()
    nondimensional = schema.value_port(representation="nondimensional")
    dimensional = schema.value_port(representation="dimensional")
    assert dimensional.representation == "dimensional"
    assert dimensional.normalization_id is None
    assert dimensional.port_id != nondimensional.port_id
    rescaled = _flow_schema(scales=(1.0, 10.0, 10.0, 1.0, 1000.0)).value_port(
        representation="nondimensional"
    )
    assert rescaled.normalization_id != nondimensional.normalization_id
    assert rescaled.port_id != nondimensional.port_id
    with pytest.raises(ValueError, match="representation"):
        schema.value_port(representation="normalized")
    port = _feature_schema().value_port()
    assert port.semantic_id == "resolved-gradient"
    assert port.event_shape == (2,)
    assert port.component_ids == ("s_xx", "s_xy")
    assert port.dimensions == (DIMENSIONLESS / TIME, DIMENSIONLESS / TIME)
    assert port.representation == "learned-stress-features"
    assert port.variance == "neutral"
    assert port.normalization_id is None
    schema = _feature_schema(units=("1/s", "per-second"))
    with pytest.raises(ValueError, match="per-second") as error:
        schema.value_port()
    assert "resolved-gradient" in str(error.value)
    port = _feature_schema().value_port()
    assert _feature_schema().value_port().port_id == port.port_id
    assert _feature_schema(shape=(8, 8, 2)).value_port().port_id == port.port_id
    assert _feature_schema(name="other-gradient").value_port().port_id != port.port_id
    assert _feature_schema(units=("1/s", "1")).value_port().port_id != port.port_id
    port = _output_contract().value_port()
    target = "deviatoric-specific-stress-target"
    assert port.semantic_id == target
    assert port.event_shape == (3, 3)
    assert port.component_ids == tuple(f"{target}[{index}]" for index in range(9))
    assert port.dimensions == (VELOCITY**2,) * 9
    assert port.representation == "deviatoric-constant-density-specific"
    assert port.space_id == "mesh-32"
    assert port.variance == "neutral"
    contract = _output_contract(units="(ft/min)^2")
    with pytest.raises(ValueError, match="ft/min") as error:
        contract.value_port()
    assert "deviatoric-specific-stress-target" in str(error.value)
    port = _output_contract().value_port()
    assert _output_contract().value_port() == port
    assert _output_contract(shape=(4, 4, 3, 3)).value_port().port_id == port.port_id
    assert _output_contract(units="m^2/s^2").value_port().port_id == port.port_id
    assert _output_contract(units="m/s").value_port().port_id != port.port_id
    assert _output_contract(discretization="mesh-64").value_port().port_id != port.port_id
