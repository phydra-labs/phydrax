from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry import Ball, ParameterId, ParameterSchema, ParameterSpec
from phydrax.geometry.design._schema import ParameterCollector
from phydrax.geometry.process import ProcessStack, StackRegion, StackVoid, ZInterval
from phydrax.geometry.simplicial import PlanarMeshRegion
from phydrax.units import convert_value, METER, MILLIMETER, ONE, SECOND


def test_design_parameter_and_compile_require_explicit_units() -> None:
    with pytest.raises(TypeError, match="unit"):
        ParameterSpec(ParameterId("sample", "radius"), (), "float64", "length")  # ty: ignore[missing-argument]
    with pytest.raises(TypeError, match="coordinate_unit"):
        ParameterCollector()  # ty: ignore[missing-argument]
    with pytest.raises(TypeError, match="coordinate_unit"):
        Ball((0.0, 0.0), 1.0).compile()  # ty: ignore[missing-argument]
    collector = ParameterCollector(METER)
    with pytest.raises(TypeError, match="unit"):
        collector.bind(ParameterId("sample", "radius"), 1.0, role="length")  # ty: ignore[missing-argument]


def test_parameter_units_refuse_incompatible_dimensions_and_shared_units() -> None:
    identifier = ParameterId("sample", "radius")
    with pytest.raises(ValueError, match="dimension"):
        ParameterSpec(identifier, (), "float64", "length", SECOND)
    with pytest.raises(ValueError, match="length"):
        ParameterCollector(ONE)
    collector = ParameterCollector(METER)
    collector.bind(identifier, 1.0, role="length", unit=METER)
    with pytest.raises(ValueError, match="conflicting"):
        collector.bind(identifier, 1.0, role="length", unit=MILLIMETER)


def test_schema_identity_includes_exact_units_without_conversion() -> None:
    identifier = ParameterId("sample", "radius")
    meters = ParameterSchema((ParameterSpec(identifier, (), "float64", "length", METER),))
    millimeters = ParameterSchema(
        (ParameterSpec(identifier, (), "float64", "length", MILLIMETER),)
    )
    assert meters != millimeters
    assert meters.schema_id != millimeters.schema_id
    geometry = Ball((0.0, 0.0), 2.0).scaled(3.0).compile(coordinate_unit=MILLIMETER)
    assert (
        next(spec for spec in geometry.schema.specs if spec.role == "scale").unit == ONE
    )
    length_specs = tuple(
        spec for spec in geometry.schema.specs if spec.role in {"position", "length"}
    )
    assert length_specs
    assert all(spec.unit == MILLIMETER for spec in length_specs)
    radius = next(
        spec for spec in geometry.schema.specs if spec.parameter_id.name == "radius"
    )
    assert float(geometry.state.values[geometry.schema.index(radius.parameter_id)]) == 2.0


def _footprint() -> PlanarMeshRegion:
    return PlanarMeshRegion(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
        ((0, 1, 2, 3),),
        feature_id="unit-footprint",
    )


def test_stack_units_are_required_and_mismatches_refuse() -> None:
    with pytest.raises(TypeError, match="unit"):
        ZInterval(0.0, 1.0)  # ty: ignore[missing-argument]
    with pytest.raises(ValueError, match="length"):
        ZInterval(0.0, 1.0, SECOND)
    contract = SpatialCoordinateContract(MILLIMETER)
    footprint = _footprint()
    region = StackRegion("solid", footprint, ZInterval(0.0, 1.0, MILLIMETER), 0)
    with pytest.raises(ValueError, match="exactly match"):
        ProcessStack(
            contract, (StackRegion("solid", footprint, ZInterval(0.0, 1.0, METER), 0),)
        )
    with pytest.raises(ValueError, match="exactly match"):
        ProcessStack(
            contract,
            (region,),
            (StackVoid("opening", footprint, ZInterval(0.0, 1.0, METER), ("solid",)),),
        )
    converted = convert_value(
        jnp.asarray([0.0, 1.0], dtype=jnp.float64), source=METER, target=MILLIMETER
    )
    interval = ZInterval(float(converted[0]), float(converted[1]), MILLIMETER)
    assert interval.upper == 1000.0
    assert (
        ProcessStack(
            contract, (StackRegion("solid", footprint, interval, 0),)
        ).coordinate_contract
        == contract
    )


def test_process_operand_fingerprints_include_unit_identity() -> None:
    footprint = _footprint()
    meter_region = StackRegion("solid", footprint, ZInterval(0.0, 1.0, METER), 0)
    millimeter_region = StackRegion(
        "solid", footprint, ZInterval(0.0, 1.0, MILLIMETER), 0
    )
    assert meter_region.stack_region_id != millimeter_region.stack_region_id
    assert (
        ProcessStack(SpatialCoordinateContract(METER), (meter_region,)).stack_id
        != ProcessStack(
            SpatialCoordinateContract(MILLIMETER), (millimeter_region,)
        ).stack_id
    )
