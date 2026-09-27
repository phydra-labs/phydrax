#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import json
from typing import Any

import pytest

from phydrax import (
    ModelPorts,
    PortMapping,
    PortProvider,
    resolve_port_mapping,
    ValuePort,
)
from phydrax.axes import AxisKey
from phydrax.units import LENGTH, TIME, VELOCITY


def _velocity(**overrides: Any) -> Any:
    fields = dict(
        event_shape=(2,),
        component_ids=("u", "v"),
        representation="cartesian",
        space_id="channel",
        dimensions=(VELOCITY, VELOCITY),
        frame_id="lab",
        normalization_id="identity",
        axis_keys=(AxisKey("velocity", "component"),),
        variance="contravariant",
    )
    fields.update(overrides)
    # ty: ignore[invalid-argument-type]
    return ValuePort("fluid.velocity", **fields)


def _position(**overrides: Any) -> Any:
    fields = dict(
        event_shape=(2,),
        component_ids=("x", "y"),
        representation="cartesian",
        dimensions=(LENGTH, LENGTH),
        frame_id="lab",
    )
    fields.update(overrides)
    # ty: ignore[invalid-argument-type]
    return ValuePort("space.position", **fields)


def _time() -> Any:
    return ValuePort(
        "time", event_shape=(), component_ids=("t",), representation="physical"
    )


def _bind(
    model_inputs: Any,
    owner_inputs: Any,
    model_outputs: Any,
    owner_outputs: Any,
    **mapping: Any,
) -> Any:
    model = ModelPorts(inputs=model_inputs, outputs=model_outputs)
    owner = ModelPorts(inputs=owner_inputs, outputs=owner_outputs)
    if not mapping:
        mapping = dict(
            inputs=[
                (m.port_id, o.port_id)
                for m, o in zip(model_inputs, owner_inputs, strict=True)
            ],
            outputs=[
                (m.port_id, o.port_id)
                for m, o in zip(model_outputs, owner_outputs, strict=True)
            ],
        )
    return resolve_port_mapping(model, owner, PortMapping(**mapping))


def test_ports_scenario_1() -> None:
    position, time, velocity = _position(), _time(), _velocity()
    evidence = _bind((position, time), (position, time), (velocity,), (velocity,))

    assert evidence.inputs == (
        (position.port_id, position.port_id),
        (time.port_id, time.port_id),
    )
    assert evidence.outputs == ((velocity.port_id, velocity.port_id),)
    assert evidence.dimensions_verified is False
    assert ("input", time.port_id, "dimensions") in evidence.unverified
    assert ("input", position.port_id, "normalization") in evidence.unverified
    assert not any(record[0] == "output" for record in evidence.unverified)

    velocity_only = _bind((velocity,), (velocity,), (velocity,), (velocity,))
    assert velocity_only.unverified == ()
    assert velocity_only.dimensions_verified
    assert velocity_only.axes_verified
    assert velocity_only.frames_verified
    assert velocity_only.normalizations_verified
    assert velocity_only.spaces_verified
    position, time, velocity = _position(), _time(), _velocity()
    evidence = _bind(
        (time,),
        (position, time),
        (velocity,),
        (velocity,),
        inputs=[(time.port_id, time.port_id)],
        outputs=[(velocity.port_id, velocity.port_id)],
    )
    assert evidence.inputs == ((time.port_id, time.port_id),)

    with pytest.raises(ValueError, match="unbound"):
        _bind(
            (position, time),
            (position, time),
            (velocity,),
            (velocity,),
            inputs=[(time.port_id, time.port_id)],
            outputs=[(velocity.port_id, velocity.port_id)],
        )
    with pytest.raises(ValueError, match="unbound"):
        _bind(
            (time,),
            (time,),
            (velocity,),
            (velocity,),
            inputs=[(time.port_id, time.port_id)],
        )
    with pytest.raises(ValueError, match="unknown model input"):
        _bind(
            (time,),
            (position, time),
            (velocity,),
            (velocity,),
            inputs=[(time.port_id, time.port_id), (position.port_id, position.port_id)],
            outputs=[(velocity.port_id, velocity.port_id)],
        )
    with pytest.raises(ValueError, match="unknown owner output"):
        _bind(
            (time,),
            (time,),
            (velocity,),
            (velocity,),
            inputs=[(time.port_id, time.port_id)],
            outputs=[(velocity.port_id, time.port_id)],
        )
    position, time = _position(), _time()
    with pytest.raises(ValueError, match="model port more than once"):
        PortMapping(
            inputs=[(time.port_id, time.port_id), (time.port_id, position.port_id)]
        )
    with pytest.raises(ValueError, match="owner port more than once"):
        PortMapping(
            inputs=[(time.port_id, time.port_id), (position.port_id, time.port_id)]
        )
    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        PortMapping(inputs=[[time.port_id, time.port_id]])
    assert PortMapping(
        inputs=[(time.port_id, time.port_id), (position.port_id, position.port_id)]
    ) == PortMapping(
        inputs=[(position.port_id, position.port_id), (time.port_id, time.port_id)]
    )


def test_ports_scenario_2() -> None:
    mismatches: tuple[tuple[str, ValuePort], ...] = (
        (
            "semantic_id",
            ValuePort(
                "fluid.momentum",
                event_shape=(2,),
                component_ids=("u", "v"),
                representation="cartesian",
                variance="contravariant",
            ),
        ),
        ("component_ids", _velocity(component_ids=("v", "u"))),
        ("event_shape", _velocity(event_shape=(2, 1), axis_keys=None)),
        ("representation", _velocity(representation="polar")),
        ("variance", _velocity(variance="covariant")),
        ("dimensions", _velocity(dimensions=(VELOCITY, TIME))),
        ("axes", _velocity(axis_keys=(AxisKey("velocity", "direction"),))),
        ("frame", _velocity(frame_id="body")),
        ("normalization", _velocity(normalization_id="z-score")),
        ("space", _velocity(space_id="inlet")),
    )
    time, velocity = _time(), _velocity()
    for aspect, owner in mismatches:
        with pytest.raises(
            ValueError,
            match=rf"output port pair 'fluid.velocity'.*{aspect}",
        ):
            _bind((time,), (time,), (velocity,), (owner,))
    time, velocity = _time(), _velocity()
    undeclared_owner = _velocity(dimensions=None)
    evidence = _bind((time,), (time,), (velocity,), (undeclared_owner,))
    assert not evidence.dimensions_verified
    assert [record for record in evidence.unverified if record[0] == "output"] == [
        ("output", velocity.port_id, "dimensions")
    ]

    with pytest.raises(ValueError, match="dimensions mismatch"):
        _bind((time,), (time,), (velocity,), (_velocity(dimensions=(LENGTH, LENGTH)),))
    assert _velocity().port_id == _velocity().port_id
    assert _velocity().port_id != _velocity(frame_id="body").port_id
    assert _velocity().port_id != _velocity(dimensions=None).port_id

    first = _bind((_time(),), (_time(),), (_velocity(),), (_velocity(),))
    second = _bind((_time(),), (_time(),), (_velocity(),), (_velocity(),))
    assert first.binding_fingerprint == second.binding_fingerprint
    relaxed = _bind((_time(),), (_time(),), (_velocity(),), (_velocity(frame_id=None),))
    assert relaxed.binding_fingerprint != first.binding_fingerprint

    ports = ModelPorts(inputs=(_position(), _time()), outputs=(_velocity(),))
    swapped = ModelPorts(inputs=(_time(), _position()), outputs=(_velocity(),))
    assert ports.ports_id != swapped.ports_id
    for port in (_velocity(), _time()):
        record = json.loads(json.dumps(port.to_dict()))
        restored = ValuePort.from_dict(record)
        assert restored.port_id == port.port_id
        assert restored.dimensions == port.dimensions
        assert restored.axis_keys == port.axis_keys

    record = _velocity().to_dict()
    with pytest.raises(ValueError, match="port_id does not match"):
        ValuePort.from_dict(record | {"port_id": _time().port_id})
    with pytest.raises(ValueError, match="port_id does not match"):
        ValuePort.from_dict(record | {"frame_id": "body"})
    with pytest.raises(ValueError, match="canonical fields"):
        ValuePort.from_dict(record | {"schema_version": 1})
    with pytest.raises(ValueError, match="component IDs"):
        _velocity(component_ids=("u",))
    with pytest.raises(ValueError, match="one value per component"):
        _velocity(dimensions=(VELOCITY,))
    with pytest.raises(TypeError):
        _velocity(dimensions=("m/s", "m/s"))
    with pytest.raises(ValueError, match="one value per event axis"):
        _velocity(axis_keys=())
    with pytest.raises(ValueError, match="distinct"):
        _velocity(
            event_shape=(1, 2),
            axis_keys=(AxisKey("velocity", "component"),) * 2,
        )
    with pytest.raises(ValueError, match="variance"):
        _velocity(variance="none")
    with pytest.raises(ValueError):
        _velocity(event_shape=(0,), component_ids=())
    with pytest.raises(ValueError, match="repeat a port"):
        ModelPorts(inputs=(_time(), _time()), outputs=())
    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        resolve_port_mapping(ModelPorts(inputs=(), outputs=()), object(), PortMapping())


def test_port_providers_are_structural() -> None:
    class _Provider:
        def model_ports(self) -> Any:
            return ModelPorts(inputs=(_time(),), outputs=(_velocity(),))

    assert isinstance(_Provider(), PortProvider)
    assert not isinstance(object(), PortProvider)
