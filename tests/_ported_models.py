#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Port-declaring array models for owner-slot port binding tests."""

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax import AbstractArrayModel, ModelPorts, PortMapping, ValuePort
from phydrax.axes import AxisKey
from phydrax.units import DIMENSIONLESS


def full_port(semantic_id: str, event_shape: Sequence[int], /) -> ValuePort:
    """Return a port declaring every optional aspect, so bindings verify fully."""
    shape = tuple(event_shape)
    size = prod(shape)
    return ValuePort(
        semantic_id,
        event_shape=shape,
        component_ids=tuple(f"{semantic_id}[{index}]" for index in range(size)),
        representation="test-values",
        space_id=f"{semantic_id}:space",
        dimensions=(DIMENSIONLESS,) * size,
        frame_id="lab",
        normalization_id="identity",
        axis_keys=tuple(
            AxisKey(semantic_id, f"axis-{axis}") for axis in range(len(shape))
        ),
    )


def in_order(model_ports: ModelPorts, owner_ports: ModelPorts, /) -> PortMapping:
    """Map each model port to the owner port at the same position."""
    return PortMapping(
        inputs=[
            (model.port_id, owner.port_id)
            for model, owner in zip(model_ports.inputs, owner_ports.inputs, strict=True)
        ],
        outputs=[
            (model.port_id, owner.port_id)
            for model, owner in zip(model_ports.outputs, owner_ports.outputs, strict=True)
        ],
    )


def _shape(size: int | tuple[int, ...] | str, /) -> tuple[int, ...]:
    if size == "scalar":
        return ()
    return (size,) if isinstance(size, int) else tuple(size)


class PortedAffine(AbstractArrayModel):
    """Linear flat map declaring intrinsic ports; `out_size` is the output size."""

    weight: jax.Array
    ports: ModelPorts
    in_size: int = eqx.field(static=True)
    out_size: int | tuple[int, ...] | str = eqx.field(static=True)

    def __init__(
        self,
        ports: ModelPorts,
        /,
        *,
        out_size: int | tuple[int, ...] | str,
        weight: jax.Array | None = None,
    ):
        in_size = sum(prod(port.event_shape) for port in ports.inputs)
        out = prod(_shape(out_size))
        self.weight = jnp.zeros((out, in_size)) if weight is None else weight
        self.ports = ports
        self.in_size = in_size
        self.out_size = out_size

    def __call__(self, x, /, *, key=None):
        del key
        return (self.weight @ jnp.ravel(x)).reshape(_shape(self.out_size))

    def model_ports(self) -> ModelPorts:
        return self.ports
