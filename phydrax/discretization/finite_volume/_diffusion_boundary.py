#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Face-indexed boundary laws for area-integrated hybrid diffusion rates."""

from __future__ import annotations

from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ._unstructured import UnstructuredFiniteVolumeDiscretization


class HybridDiffusionBoundary(StrictModule):
    """Dirichlet u, outward Neumann rate, or q = conductance*(u-external).

    Conductance and Neumann values are already integrated over the face area.
    Unspecified exterior faces are impermeable. Interior faces cannot carry a law.
    Values may be replaced with ``eqx.tree_at`` for differentiable boundary data.
    """

    kind: Array
    value: Array
    conductance: Array
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: UnstructuredFiniteVolumeDiscretization,
        /,
        *,
        dirichlet: Mapping[int, float] | None = None,
        neumann: Mapping[int, float] | None = None,
        robin: Mapping[int, tuple[float, float]] | None = None,
    ):
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise TypeError("Boundary geometry must be native prepared unstructured FV.")
        exterior = np.asarray(discretization.neighbour_cells) < 0
        count = exterior.size
        kind = np.where(exterior, 2, 0).astype(np.int32)
        value = np.zeros(count)
        conductance = np.zeros(count)
        assigned: set[int] = set()

        def validate_face(face: int) -> int:
            if isinstance(face, bool) or not isinstance(face, (int, np.integer)):
                raise TypeError("Boundary face indices must be integers.")
            face_index = int(face)
            if not 0 <= face_index < count or not exterior[face_index]:
                raise ValueError("Boundary laws require exterior face indices.")
            if face_index in assigned:
                raise ValueError("A face cannot have multiple boundary laws.")
            assigned.add(face_index)
            return face_index

        def assign_scalar(data: Mapping[int, float] | None, code: int) -> None:
            for face, target in ({} if data is None else data).items():
                face_index = validate_face(face)
                if not np.isfinite(target):
                    raise ValueError("Boundary targets must be finite.")
                kind[face_index], value[face_index] = code, target

        assign_scalar(dirichlet, 1)
        assign_scalar(neumann, 2)
        for face, datum in ({} if robin is None else robin).items():
            face_index = validate_face(face)
            transfer, target = datum
            if not np.isfinite(transfer) or transfer <= 0:
                raise ValueError("Robin conductance must be finite and positive.")
            if not np.isfinite(target):
                raise ValueError("Boundary targets must be finite.")
            kind[face_index], value[face_index] = 3, target
            conductance[face_index] = transfer
        self.kind = jnp.asarray(kind)
        self.value = jnp.asarray(value)
        self.conductance = jnp.asarray(conductance)
        self.geometry_id = discretization.geometry_id

    def face_residual(self, face_values: Array, outward_sum: Array) -> Array:
        """Hybrid stationarity on interior faces and the declared exterior law."""
        result = -outward_sum
        result = jnp.where(self.kind == 2, result + self.value, result)
        result = jnp.where(
            self.kind == 3,
            result + self.conductance * (face_values - self.value),
            result,
        )
        return jnp.where(self.kind == 1, face_values - self.value, result)

    def impose_dirichlet(self, face_values: Array) -> Array:
        return jnp.where(self.kind == 1, self.value, face_values)


__all__ = ["HybridDiffusionBoundary"]
