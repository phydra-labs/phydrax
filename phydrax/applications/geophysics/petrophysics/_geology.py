#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class LevelSetInterfacePlan(StrictModule, NonTrainableState):
    basis: Array
    smoothing_width: float = eqx.field(static=True)
    topology_mode: Literal["hard", "smooth"] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: ArrayLike,
        /,
        *,
        smoothing_width: float = 0.0,
    ):
        matrix = jnp.asarray(basis)
        width = float(smoothing_width)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError(
                "Level-set basis must be a nonempty sample-by-parameter matrix."
            )
        matrix = eqx.error_if(
            matrix,
            jnp.any(~jnp.isfinite(matrix)),
            "Level-set basis must be finite.",
        )
        if not np.isfinite(width) or width < 0:
            raise ValueError("Level-set smoothing width must be finite and nonnegative.")
        self.basis, self.smoothing_width = matrix, width
        self.topology_mode = "hard" if width == 0 else "smooth"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "level-set-interface",
                "basis": matrix,
                "smoothing_width": width,
            }
        )

    def signed_distance_proxy(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != (self.basis.shape[1],):
            raise ValueError("Level-set coefficients do not match basis columns.")
        return self.basis @ values

    def phase_fraction(self, coefficients: ArrayLike, /) -> Array:
        level = self.signed_distance_proxy(coefficients)
        if self.smoothing_width == 0:
            return (level >= 0).astype(level.dtype)
        return jnn.sigmoid(level / self.smoothing_width)

    @property
    def differentiable_topology(self) -> bool:
        return self.smoothing_width > 0


class StratigraphicLayerPlan(StrictModule, NonTrainableState):
    interfaces: tuple[LevelSetInterfacePlan, ...]
    layer_names: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        interfaces: tuple[LevelSetInterfacePlan, ...],
        layer_names: tuple[str, ...],
        /,
    ):
        if not interfaces or any(
            not isinstance(value, LevelSetInterfacePlan) for value in interfaces
        ):
            raise TypeError("Stratigraphy requires one or more level-set interfaces.")
        names = tuple(str(value).strip() for value in layer_names)
        if (
            len(names) != len(interfaces) + 1
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Stratigraphic layer names must be unique and bracket interfaces."
            )
        if len({value.basis.shape[0] for value in interfaces}) != 1:
            raise ValueError("Stratigraphic interfaces must share sample support.")
        self.interfaces, self.layer_names = interfaces, names
        self.plan_id = canonical_fingerprint(
            {
                "kind": "stratigraphic-layer-plan",
                "interfaces": [value.plan_id for value in interfaces],
                "layer_names": names,
            }
        )

    def fractions(self, coefficients: tuple[ArrayLike, ...], /) -> Array:
        if len(coefficients) != len(self.interfaces):
            raise ValueError("Stratigraphic coefficient blocks must match interfaces.")
        cumulative = jnp.stack(
            [
                interface.phase_fraction(value)
                for interface, value in zip(self.interfaces, coefficients, strict=True)
            ],
            axis=1,
        )
        # Ordered cumulative fractions must decrease upward without crossing.
        cumulative = eqx.error_if(
            cumulative,
            jnp.any(cumulative[:, 1:] > cumulative[:, :-1] + 1e-10),
            "Stratigraphic interfaces cross on the declared sample support.",
        )
        return jnp.concatenate(
            (
                1.0 - cumulative[:, :1],
                cumulative[:, :-1] - cumulative[:, 1:],
                cumulative[:, -1:],
            ),
            axis=1,
        )


class GeologicalBodyPlan(StrictModule, NonTrainableState):
    sample_positions: Array
    kind: Literal["ellipsoid", "axis-aligned-prism"] = eqx.field(static=True)
    smoothing_width: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_positions: ArrayLike,
        kind: Literal["ellipsoid", "axis-aligned-prism"],
        /,
        *,
        smoothing_width: float = 0.0,
    ):
        positions = jnp.asarray(sample_positions)
        width = float(smoothing_width)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or kind
            not in (
                "ellipsoid",
                "axis-aligned-prism",
            )
        ):
            raise ValueError("Geological body samples/kind are invalid.")
        self.sample_positions = eqx.error_if(
            positions,
            jnp.any(~jnp.isfinite(positions)),
            "Geological body sample positions must be finite.",
        )
        if not np.isfinite(width) or width < 0:
            raise ValueError("Geological body smoothing must be nonnegative finite.")
        self.kind, self.smoothing_width = kind, width
        self.plan_id = canonical_fingerprint(
            {
                "kind": "geological-body-plan",
                "sample_positions_m": positions,
                "body_kind": kind,
                "smoothing_width": width,
            }
        )

    def fraction(self, center_m: ArrayLike, half_axes_m: ArrayLike, /) -> Array:
        center, axes = jnp.asarray(center_m), jnp.asarray(half_axes_m)
        if center.shape != (3,) or axes.shape != (3,):
            raise ValueError("Geological body center and half axes must be 3-vectors.")
        axes = eqx.error_if(
            axes,
            jnp.any(~jnp.isfinite(center))
            | jnp.any(~jnp.isfinite(axes))
            | jnp.any(axes <= 0),
            "Geological body center/half axes must be finite and axes positive.",
        )
        normalized = jnp.abs((self.sample_positions - center) / axes)
        signed = (
            1.0 - jnp.sqrt(jnp.sum(normalized**2, axis=1))
            if self.kind == "ellipsoid"
            else 1.0 - jnp.max(normalized, axis=1)
        )
        if self.smoothing_width == 0:
            return (signed >= 0).astype(signed.dtype)
        return jnn.sigmoid(signed / self.smoothing_width)


class FaciesProbabilityPlan(StrictModule, NonTrainableState):
    facies_names: tuple[str, ...] = eqx.field(static=True)
    property_values: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, facies_names: tuple[str, ...], property_values: ArrayLike, /):
        names = tuple(str(value).strip() for value in facies_names)
        properties = jnp.asarray(property_values)
        if (
            len(names) < 2
            or any(not value for value in names)
            or len(set(names)) != len(names)
            or properties.ndim < 1
            or properties.shape[0] != len(names)
        ):
            raise ValueError("Facies names/property leading axis are invalid.")
        self.facies_names = names
        self.property_values = eqx.error_if(
            properties,
            jnp.any(~jnp.isfinite(properties)),
            "Facies properties must be finite.",
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "facies-probability-plan",
                "facies_names": names,
                "property_values": properties,
            }
        )

    def probabilities(self, logits: ArrayLike, /) -> Array:
        values = jnp.asarray(logits)
        if values.shape[-1] != len(self.facies_names):
            raise ValueError("Facies logits trailing axis is invalid.")
        return jnn.softmax(values, axis=-1)

    def mixture(self, logits: ArrayLike, /) -> Array:
        probabilities = self.probabilities(logits)
        flat_properties = self.property_values.reshape(len(self.facies_names), -1)
        mixed = ein.contract("...f,fp->...p", probabilities, flat_properties)
        return mixed.reshape(probabilities.shape[:-1] + self.property_values.shape[1:])


__all__ = [
    "FaciesProbabilityPlan",
    "GeologicalBodyPlan",
    "LevelSetInterfacePlan",
    "StratigraphicLayerPlan",
]
