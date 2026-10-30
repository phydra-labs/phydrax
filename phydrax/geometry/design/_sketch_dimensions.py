#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import checked, parse
from ._schema import _ParameterCollector, DesignState, ParameterId, ParameterSchema


SketchDimensionAttribute: TypeAlias = Literal["distance", "angle", "radius"]


@dataclass(frozen=True, slots=True)
class SketchDimensionBinding:
    """Stable mapping from a dimension-bearing sketch constraint to its state leaf."""

    constraint_index: int
    state_index: int
    parameter_id: ParameterId
    attribute: SketchDimensionAttribute


class SketchDimensionSensitivity(StrictModule):
    """One fixed-topology implicit sensitivity of sketch variables to a dimension."""

    parameter_id: ParameterId = eqx.field(static=True)
    coordinates: Array
    residual_tangent: Array
    root_evidence_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)
    sensitivity_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        parameter_id: ParameterId,
        coordinates: ArrayLike,
        residual_tangent: ArrayLike,
        root_evidence_id: str,
        /,
    ) -> None:
        if not isinstance(parameter_id, ParameterId):
            raise TypeError("parameter_id must be ParameterId.")
        if not isinstance(root_evidence_id, str) or not root_evidence_id:
            raise ValueError("root_evidence_id must be non-empty.")
        coordinates_ = jnp.asarray(coordinates, dtype=jnp.float64).reshape((-1,))
        residual_tangent_ = jnp.asarray(residual_tangent, dtype=jnp.float64).reshape(
            (-1,)
        )
        if not bool(np.all(np.isfinite(np.asarray(coordinates_)))) or not bool(
            np.all(np.isfinite(np.asarray(residual_tangent_)))
        ):
            raise ValueError("Sketch dimension sensitivity must be finite.")
        scope = "local-implicit-fixed-topology"
        self.parameter_id = parameter_id
        self.coordinates = coordinates_
        self.residual_tangent = residual_tangent_
        self.root_evidence_id = root_evidence_id
        self.scope = scope
        self.sensitivity_id = canonical_fingerprint(
            {
                "kind": "sketch-dimension-sensitivity",
                "parameter_id": str(parameter_id),
                "root_evidence": root_evidence_id,
                "coordinates": np.asarray(coordinates_),
                "residual_tangent": np.asarray(residual_tangent_),
                "scope": scope,
            }
        )


def compile_sketch_dimensions(
    entries: Sequence[
        tuple[
            int,
            ParameterId,
            ArrayLike,
            SketchDimensionAttribute,
            tuple[float | None, float | None],
        ],
    ],
    feature_id: str,
    /,
) -> tuple[ParameterSchema, DesignState, tuple[SketchDimensionBinding, ...]]:
    """Compile explicitly named scalar dimensions into a stable DesignState."""
    collector = _ParameterCollector()
    bindings: list[SketchDimensionBinding] = []
    for constraint_index, parameter_id, value, role, bounds in entries:
        if not isinstance(parameter_id, ParameterId):
            raise TypeError("Every sketch dimension requires a ParameterId.")
        if parameter_id.feature_id != feature_id:
            raise ValueError(
                "Sketch dimension ParameterIds must use the sketch feature_id."
            )
        role = parse(role, SketchDimensionAttribute, "role")
        initial = np.asarray(value, dtype=np.float64)
        if initial.shape != () or not np.isfinite(initial):
            raise ValueError("Sketch dimensions must be finite scalar values.")
        binding = collector.bind(
            parameter_id,
            initial,
            role=f"sketch-dimension:{role}",
            physical_scale=1.0,
            bounds=bounds,
            trainable=False,
        )
        bindings.append(
            SketchDimensionBinding(
                constraint_index,
                binding.index,
                parameter_id,
                role,
            )
        )
    schema, state = collector.finish()
    return schema, state, tuple(bindings)


def require_dimension_state(
    state: DesignState,
    schema: ParameterSchema,
    /,
) -> DesignState:
    if not isinstance(state, DesignState):
        raise TypeError("dimensions must be a DesignState.")
    if state.schema != schema:
        raise ValueError("Sketch dimensions do not match the sketch dimension schema.")
    for spec, value in zip(schema.specs, state.values, strict=True):
        value_ = np.asarray(value, dtype=np.float64)
        if value_.shape != () or not np.isfinite(value_):
            raise ValueError("Sketch dimension values must be finite scalars.")
        lower, upper = spec.bounds
        if lower is not None and value_ < lower:
            raise ValueError(
                f"Sketch dimension {spec.parameter_id} is below its lower bound."
            )
        if upper is not None and value_ > upper:
            raise ValueError(
                f"Sketch dimension {spec.parameter_id} is above its upper bound."
            )
        if spec.role == "sketch-dimension:radius" and value_ <= 0.0:
            raise ValueError("Radius dimensions must remain strictly positive.")
    return state


__all__ = [
    "SketchDimensionBinding",
    "SketchDimensionSensitivity",
    "compile_sketch_dimensions",
    "require_dimension_state",
]
