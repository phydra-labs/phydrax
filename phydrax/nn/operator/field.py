#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ...discretization import CochainFieldSpec
from .._utils import _get_size
from .capabilities import OperatorFieldRepresentation
from .data import OperatorOutputSpec
from .representations import (
    CliffordGradeRepresentation,
    TensorFieldLayout,
)


OperatorFieldRole = Literal["source", "target", "both"]


class OperatorFieldSpec(StrictModule):
    """Static physical semantics for one named operator field."""

    name: str
    channels: int | Literal["scalar"]
    role: OperatorFieldRole
    representation: OperatorFieldRepresentation
    source_name: str | None
    query_name: str | None
    output_spec: OperatorOutputSpec | None
    component_names: tuple[str, ...]
    physical_dimension: tuple[float, ...]
    scale: tuple[float, ...]
    offset: tuple[float, ...]
    cochain: CochainFieldSpec | None
    tensor_layout: TensorFieldLayout | None
    clifford_layout: CliffordGradeRepresentation | None
    required: bool

    def __init__(
        self,
        name: str,
        /,
        *,
        channels: int | Literal["scalar"] = "scalar",
        role: OperatorFieldRole = "both",
        representation: OperatorFieldRepresentation | None = None,
        source_name: str | None = None,
        query_name: str | None = None,
        output_spec: OperatorOutputSpec | None = None,
        component_names: Sequence[str] = (),
        physical_dimension: Sequence[float] = (),
        scale: float | Sequence[float] = 1.0,
        offset: float | Sequence[float] = 0.0,
        cochain: CochainFieldSpec | None = None,
        tensor_layout: TensorFieldLayout | None = None,
        clifford_layout: CliffordGradeRepresentation | None = None,
        required: bool = True,
    ):
        resolved_name = str(name)
        if not resolved_name:
            raise ValueError("Operator field name must not be empty.")
        if role not in ("source", "target", "both"):
            raise ValueError("Operator field role must be 'source', 'target', or 'both'.")
        channel_count = _get_size(channels)
        if representation is None:
            if clifford_layout is not None:
                resolved_representation: OperatorFieldRepresentation = (
                    "clifford_multivector"
                )
            elif tensor_layout is not None:
                resolved_representation = "tensor"
            elif channels == "scalar":
                resolved_representation = "scalar"
            else:
                resolved_representation = "generic_channels"
        else:
            resolved_representation = representation
        if resolved_representation not in (
            "generic_channels",
            "scalar",
            "pseudoscalar",
            "vector",
            "covector",
            "tensor",
            "clifford_multivector",
        ):
            raise ValueError("Unknown operator field representation.")
        names = tuple(str(value) for value in component_names)
        if names and (len(names) != channel_count or len(set(names)) != len(names)):
            raise ValueError("component_names must uniquely name every field channel.")
        dimension = tuple(float(value) for value in physical_dimension)
        if any(not jnp.isfinite(value) for value in dimension):
            raise ValueError("physical_dimension exponents must be finite.")

        def channel_values(
            value: float | Sequence[float],
            label: str,
        ) -> tuple[float, ...]:
            if isinstance(value, (int, float)):
                result = (float(value),) * channel_count
            else:
                result = tuple(float(item) for item in value)
            if len(result) != channel_count or any(
                not jnp.isfinite(item) for item in result
            ):
                raise ValueError(f"{label} must provide one finite value per channel.")
            return result

        scales = channel_values(scale, "scale")
        offsets = channel_values(offset, "offset")
        if any(value <= 0.0 for value in scales):
            raise ValueError("Field scales must be strictly positive.")
        if tensor_layout is not None:
            if not isinstance(tensor_layout, TensorFieldLayout):
                raise TypeError("tensor_layout must be a TensorFieldLayout or None.")
            if resolved_representation != "tensor":
                raise ValueError(
                    "Structured tensor layouts require representation='tensor'."
                )
            if tensor_layout.channel_count != channel_count:
                raise ValueError(
                    "Tensor layout width must equal the declared field channel count."
                )
            tensor_layout.validate_affine_normalization(scales, offsets)
        if clifford_layout is not None:
            if not isinstance(clifford_layout, CliffordGradeRepresentation):
                raise TypeError(
                    "clifford_layout must be CliffordGradeRepresentation or None."
                )
            if resolved_representation != "clifford_multivector":
                raise ValueError(
                    "Clifford layouts require representation='clifford_multivector'."
                )
            if clifford_layout.packed_size != channel_count:
                raise ValueError(
                    "Clifford packed width must equal the declared field channel count."
                )
            if tensor_layout is not None or cochain is not None:
                raise ValueError(
                    "Clifford fields cannot also declare tensor or cochain layouts."
                )
            clifford_layout.validate_affine_normalization(scales, offsets)
        elif resolved_representation == "clifford_multivector":
            raise ValueError("Clifford multivector fields require a clifford_layout.")
        if cochain is not None and not isinstance(cochain, CochainFieldSpec):
            raise TypeError("cochain must be a CochainFieldSpec or None.")
        if (
            cochain is not None
            and cochain.cell_orientation == "signed"
            and any(value != 0.0 for value in offsets)
        ):
            raise ValueError(
                "Orientation-signed cochain fields require zero dimensional offsets."
            )
        sources = role in ("source", "both")
        targets = role in ("target", "both")
        resolved_source = (
            resolved_name if sources and source_name is None else source_name
        )
        resolved_query = resolved_name if targets and query_name is None else query_name
        if sources and not resolved_source:
            raise ValueError("Source fields require a source_name.")
        if targets and output_spec is None:
            output_spec = OperatorOutputSpec(
                channels,
                component_names=names if channels != "scalar" else (),
            )
        if not targets and output_spec is not None:
            raise ValueError("Source-only fields cannot define an output specification.")
        if targets and not resolved_query:
            raise ValueError("Target fields require a query_name.")
        classification = None if output_spec is None else output_spec.classification
        if classification is not None:
            if role != "target":
                raise ValueError(
                    "Classification fields are target-only until physical label "
                    "channels are explicitly supported."
                )
            if (
                channels != "scalar"
                or resolved_representation != "scalar"
                or names
                or dimension
                or scales != (1.0,)
                or offsets != (0.0,)
                or cochain is not None
                or tensor_layout is not None
                or clifford_layout is not None
            ):
                raise ValueError(
                    "Classification fields must have no physical channels: use a "
                    "dimensionless scalar field with identity affine semantics and "
                    "no component, cochain, or tensor metadata."
                )
        self.name = resolved_name
        self.channels = channels
        self.representation = resolved_representation
        self.role = role
        self.source_name = None if resolved_source is None else str(resolved_source)
        self.query_name = None if resolved_query is None else str(resolved_query)
        self.output_spec = output_spec
        self.component_names = names
        self.physical_dimension = dimension
        self.scale = scales
        self.offset = offsets
        self.cochain = cochain
        self.tensor_layout = tensor_layout
        self.clifford_layout = clifford_layout
        self.required = bool(required)

    @property
    def channel_count(self) -> int:
        return _get_size(self.channels)

    @property
    def is_source(self) -> bool:
        return self.role in ("source", "both")

    @property
    def is_target(self) -> bool:
        return self.role in ("target", "both")

    @property
    def is_classification(self) -> bool:
        return (
            self.output_spec is not None and self.output_spec.classification is not None
        )

    def nondimensionalize(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        if self.is_classification:
            return array
        scale = jnp.asarray(self.scale, dtype=array.dtype)
        offset = jnp.asarray(self.offset, dtype=array.dtype)
        if self.channels == "scalar":
            return (array - offset[0]) / scale[0]
        if int(array.shape[-1]) != self.channel_count:
            raise ValueError(
                f"Field {self.name!r} expected {self.channel_count} channels; "
                f"got {array.shape[-1]}."
            )
        return (array - offset) / scale

    def dimensionalize(self, values: Array, /) -> Array:
        array = jnp.asarray(values)
        if self.is_classification:
            return array
        scale = jnp.asarray(self.scale, dtype=array.dtype)
        offset = jnp.asarray(self.offset, dtype=array.dtype)
        if self.channels == "scalar":
            return array * scale[0] + offset[0]
        if int(array.shape[-1]) != self.channel_count:
            raise ValueError(
                f"Field {self.name!r} expected {self.channel_count} channels; "
                f"got {array.shape[-1]}."
            )
        return array * scale + offset

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible field specification."""
        output_spec = self.output_spec
        return {
            "name": self.name,
            "channels": self.channels,
            "role": self.role,
            "representation": self.representation,
            "source_name": self.source_name,
            "query_name": self.query_name,
            "output_spec": (None if output_spec is None else output_spec.to_dict()),
            "component_names": list(self.component_names),
            "physical_dimension": list(self.physical_dimension),
            "scale": list(self.scale),
            "offset": list(self.offset),
            "cochain": None if self.cochain is None else self.cochain.to_dict(),
            "tensor_layout": (
                None if self.tensor_layout is None else self.tensor_layout.to_dict()
            ),
            "clifford_layout": (
                None if self.clifford_layout is None else self.clifford_layout.to_dict()
            ),
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "OperatorFieldSpec":
        """Restore a field specification from its canonical dictionary."""
        output_value = value.get("output_spec")
        output_spec = (
            None if output_value is None else OperatorOutputSpec.from_dict(output_value)
        )
        return cls(
            str(value["name"]),
            channels=value.get("channels", "scalar"),
            role=value.get("role", "both"),
            representation=value.get("representation"),
            source_name=value.get("source_name"),
            query_name=value.get("query_name"),
            output_spec=output_spec,
            component_names=value.get("component_names", ()),
            physical_dimension=value.get("physical_dimension", ()),
            scale=value.get("scale", 1.0),
            offset=value.get("offset", 0.0),
            cochain=(
                None
                if value.get("cochain") is None
                else CochainFieldSpec.from_dict(value["cochain"])
            ),
            tensor_layout=(
                None
                if value.get("tensor_layout") is None
                else TensorFieldLayout.from_dict(value["tensor_layout"])
            ),
            clifford_layout=(
                None
                if value.get("clifford_layout") is None
                else CliffordGradeRepresentation.from_dict(value["clifford_layout"])
            ),
            required=bool(value.get("required", True)),
        )


__all__ = [
    "OperatorFieldRole",
    "OperatorFieldSpec",
]
