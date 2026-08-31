#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._discretization import LatticeBoltzmannDiscretization
from ._geometry import (
    LatticeBoltzmannGeometryEpoch,
    LatticeBoltzmannGeometryRefresh,
    LatticeBoltzmannGeometryTransaction,
    LatticeBoltzmannTopologyEventRequest,
    prepare_lattice_boltzmann_topology_event,
)
from ._link_geometry import FixedSDFLinkGeometry


MovingSignedDistance: TypeAlias = Callable[[Array, Array, Any], Array]


class MovingSDFEvaluation(StrictModule, NonTrainableState):
    geometry: FixedSDFLinkGeometry
    boundary_fraction: Array
    boundary_normals: Array
    time: Array
    evaluation_id: str = eqx.field(static=True)


class MovingSDFUpdate(StrictModule, NonTrainableState):
    evaluation: MovingSDFEvaluation
    refresh: LatticeBoltzmannGeometryRefresh | None
    transaction: LatticeBoltzmannGeometryTransaction | None
    topology_changed: Array
    update_id: str = eqx.field(static=True)


class MovingSDFGeometryPlan(StrictModule, NonTrainableState):
    """Host-prepared moving SDF with numeric refresh or accepted-step transfer.

    The signed-distance callable returns either one field with shape ``grid.shape``
    or one field per named body with shape ``grid.shape + (body_count,)``. Positive
    values denote fluid. Cell classification is nondifferentiable; fixed-branch link
    fractions remain explicit numeric data between accepted topology events.
    """

    discretization: LatticeBoltzmannDiscretization
    signed_distance: MovingSignedDistance
    body_names: tuple[str, ...] = eqx.field(static=True)
    sdf_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: LatticeBoltzmannDiscretization,
        signed_distance: MovingSignedDistance,
        /,
        *,
        sdf_id: str,
        body_names: Sequence[str] = ("body",),
    ):
        if not isinstance(discretization, LatticeBoltzmannDiscretization):
            raise TypeError("discretization must be LatticeBoltzmannDiscretization.")
        if not callable(signed_distance):
            raise TypeError("signed_distance must be callable.")
        names = tuple(str(value) for value in body_names)
        identifier = str(sdf_id)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("body_names must be unique nonempty values.")
        if not identifier:
            raise ValueError("sdf_id must be nonempty.")
        self.discretization = discretization
        self.signed_distance = signed_distance
        self.body_names = names
        self.sdf_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "moving-sdf-lattice-boltzmann-geometry",
                "discretization": discretization.prepared_id,
                "sdf_id": identifier,
                "body_names": names,
            }
        )

    def _evaluate_fields(
        self, time: Array, parameters: Any, /
    ) -> tuple[np.ndarray, np.ndarray]:
        coordinates = self.discretization.grid.points.reshape(
            self.discretization.grid.shape + (self.discretization.velocity_set.dimension,)
        )
        values = np.asarray(self.signed_distance(time, coordinates, parameters))
        shape = self.discretization.grid.shape
        if values.shape == shape:
            if len(self.body_names) != 1:
                raise ValueError("One SDF field requires exactly one body name.")
            union = values
            labels = np.where(union > 0.0, -1, 0).astype(np.int32)
        elif values.shape == shape + (len(self.body_names),):
            union = np.min(values, axis=-1)
            closest = np.argmin(values, axis=-1).astype(np.int32)
            labels = np.where(union > 0.0, -1, closest).astype(np.int32)
        else:
            raise ValueError("Moving SDF output has an incompatible grid/body shape.")
        if np.any(~np.isfinite(union)):
            raise ValueError("Moving SDF output must be finite.")
        return union, labels

    def _normals(
        self,
        signed_distance: np.ndarray,
        geometry: FixedSDFLinkGeometry,
        /,
    ) -> tuple[np.ndarray, np.ndarray]:
        spacing = float(self.discretization.cell_size)
        components = []
        for axis, periodic in enumerate(self.discretization.periodic):
            if periodic:
                derivative = (
                    np.roll(signed_distance, -1, axis=axis)
                    - np.roll(signed_distance, 1, axis=axis)
                ) / (2.0 * spacing)
            else:
                derivative = np.gradient(
                    signed_distance, spacing, axis=axis, edge_order=1
                )
            components.append(derivative)
        gradient = np.stack(components, axis=-1)
        magnitude = np.sqrt(np.sum(gradient**2, axis=-1))
        unit = gradient / np.where(magnitude > 0.0, magnitude, 1.0)[..., None]
        blocked = np.isfinite(np.asarray(geometry.link_fraction))
        velocities = np.asarray(
            self.discretization.velocity_set.velocities, dtype=np.float64
        )
        lengths = np.sqrt(np.sum(velocities**2, axis=-1))
        fallback = velocities / np.where(lengths > 0.0, lengths, 1.0)[:, None]
        cell_normals = np.broadcast_to(
            unit[..., None, :],
            blocked.shape + (self.discretization.velocity_set.dimension,),
        )
        degenerate = magnitude <= 0.0
        normals = np.where(
            blocked[..., None],
            np.where(degenerate[..., None, None], fallback, cell_normals),
            0.0,
        )
        fractions = np.where(blocked, np.asarray(geometry.link_fraction), 0.0)
        return fractions, normals

    def evaluate(self, time: Array, parameters: Any = None, /) -> MovingSDFEvaluation:
        time_ = jnp.asarray(time)
        if time_.shape != ():
            raise ValueError("Moving SDF time must be scalar.")
        signed_distance, labels = self._evaluate_fields(time_, parameters)
        geometry = FixedSDFLinkGeometry(
            self.discretization,
            signed_distance,
            body_labels=labels,
            body_names=self.body_names,
        )
        fractions, normals = self._normals(signed_distance, geometry)
        evaluation_id = canonical_fingerprint(
            {
                "kind": "moving-sdf-lattice-boltzmann-evaluation",
                "plan": self.plan_id,
                "time": float(time_),
                "geometry": geometry.geometry_id,
            }
        )
        return MovingSDFEvaluation(
            geometry,
            jnp.asarray(fractions),
            jnp.asarray(normals),
            time_,
            evaluation_id,
        )

    def initialize(
        self, time: Array, parameters: Any = None, /
    ) -> tuple[LatticeBoltzmannGeometryEpoch, MovingSDFEvaluation]:
        evaluation = self.evaluate(time, parameters)
        epoch = LatticeBoltzmannGeometryEpoch.from_mask(
            self.discretization,
            evaluation.geometry.fluid_mask,
            source_id=evaluation.evaluation_id,
            boundary_fraction=evaluation.boundary_fraction,
            boundary_normals=evaluation.boundary_normals,
        )
        return epoch, evaluation

    def update(
        self,
        accepted: LatticeBoltzmannGeometryEpoch,
        time: Array,
        accepted_step: int,
        parameters: Any = None,
        /,
    ) -> MovingSDFUpdate:
        if not isinstance(accepted, LatticeBoltzmannGeometryEpoch):
            raise TypeError("accepted must be LatticeBoltzmannGeometryEpoch.")
        if accepted.discretization.prepared_id != self.discretization.prepared_id:
            raise ValueError("Accepted geometry belongs to a different discretization.")
        evaluation = self.evaluate(time, parameters)
        topology_changed = not np.array_equal(
            np.asarray(accepted.fluid_mask),
            np.asarray(evaluation.geometry.fluid_mask),
        )
        if not topology_changed:
            refresh = accepted.refresh_numeric(
                boundary_fraction=evaluation.boundary_fraction,
                boundary_normals=evaluation.boundary_normals,
            )
            return MovingSDFUpdate(
                evaluation,
                refresh,
                None,
                jnp.asarray(False),
                canonical_fingerprint(
                    {
                        "kind": "moving-sdf-numeric-refresh",
                        "source": accepted.epoch_id,
                        "target": refresh.epoch.epoch_id,
                        "evaluation": evaluation.evaluation_id,
                    }
                ),
            )
        request = LatticeBoltzmannTopologyEventRequest(
            accepted,
            evaluation.geometry.fluid_mask,
            int(accepted_step),
            source_id=evaluation.evaluation_id,
        )
        transaction = prepare_lattice_boltzmann_topology_event(
            accepted,
            request,
            boundary_fraction=evaluation.boundary_fraction,
            boundary_normals=evaluation.boundary_normals,
        )
        return MovingSDFUpdate(
            evaluation,
            None,
            transaction,
            jnp.asarray(True),
            canonical_fingerprint(
                {
                    "kind": "moving-sdf-topology-update",
                    "source": accepted.epoch_id,
                    "request": request.request_id,
                    "transaction": transaction.transaction_id,
                    "evaluation": evaluation.evaluation_id,
                }
            ),
        )


__all__ = [
    "MovingSDFEvaluation",
    "MovingSDFGeometryPlan",
    "MovingSDFUpdate",
    "MovingSignedDistance",
]
