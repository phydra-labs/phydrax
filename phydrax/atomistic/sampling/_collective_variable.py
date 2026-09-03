#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ... import linalg as la
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from .._sites import AtomisticSiteDomain
from .._system import PreparedAtomisticSystem


class CollectiveVariableKind(StrEnum):
    DISTANCE = "distance"
    ANGLE = "angle"
    TORSION = "torsion"
    CENTER_OF_MASS_DISTANCE = "center-of-mass-distance"
    RADIUS_OF_GYRATION = "radius-of-gyration"
    COORDINATION = "coordination"
    CONTACT_SIMILARITY = "contact-similarity"
    ALIGNED_RMSD = "aligned-rmsd"
    CELL_VOLUME = "cell-volume"
    DENSITY = "density"
    PATH_PROGRESS = "path-progress"
    PATH_DISTANCE = "path-distance"


class CollectiveVariableMetric(StrictModule, NonTrainableState):
    periodic: bool = eqx.field(static=True)
    period: float | None = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(self, /, *, period: float | None = None):
        period_ = None if period is None else float(period)
        if period_ is not None and period_ <= 0.0:
            raise ValueError("CV period must be positive.")
        self.periodic = period_ is not None
        self.period = period_
        self.metric_id = canonical_fingerprint({"kind": "cv-metric", "period": period_})

    def difference(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        delta = jnp.asarray(left) - jnp.asarray(right)
        if self.period is None:
            return delta
        period = jnp.asarray(self.period, dtype=delta.dtype)
        return jnp.mod(delta + 0.5 * period, period) - 0.5 * period


class CollectiveVariableEvaluation(StrictModule):
    value: Array
    branch_margin: Array
    successful: Array
    cv_id: str = eqx.field(static=True)


class AbstractCollectiveVariablePlan(StrictModule, NonTrainableState):
    cv_id: AbstractAttribute[str]
    metric: AbstractAttribute[CollectiveVariableMetric]

    @abc.abstractmethod
    def prepare(self, system: PreparedAtomisticSystem, /) -> "PreparedCollectiveVariable":
        raise NotImplementedError


class CollectiveVariablePlan(AbstractCollectiveVariablePlan):
    kind: CollectiveVariableKind = eqx.field(static=True)
    indices: Array
    parameters: Array
    reference: Array
    domain: AtomisticSiteDomain = eqx.field(static=True)
    metric: CollectiveVariableMetric
    cv_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: CollectiveVariableKind,
        indices: ArrayLike,
        /,
        *,
        parameters: ArrayLike = (),
        reference: ArrayLike = (),
        domain: AtomisticSiteDomain = AtomisticSiteDomain.DOF_ATOMS,
        metric: CollectiveVariableMetric | None = None,
    ):
        if not isinstance(kind, CollectiveVariableKind):
            raise TypeError("kind must be CollectiveVariableKind.")
        index = np.asarray(indices)
        if index.ndim not in (1, 2) or not np.issubdtype(index.dtype, np.integer):
            raise TypeError("CV indices must be rank-one or rank-two integers.")
        params = np.asarray(parameters, dtype=float)
        ref = np.asarray(reference, dtype=float)
        if (
            not isinstance(domain, AtomisticSiteDomain)
            or np.any(~np.isfinite(params))
            or np.any(~np.isfinite(ref))
        ):
            raise ValueError("CV domain, parameters, and reference must be valid.")
        exact_indices = {
            CollectiveVariableKind.DISTANCE: 2,
            CollectiveVariableKind.ANGLE: 3,
            CollectiveVariableKind.TORSION: 4,
        }
        if kind in exact_indices and (
            index.ndim != 1 or index.size != exact_indices[kind]
        ):
            raise ValueError(f"{kind.value} requires {exact_indices[kind]} indices.")
        if kind is CollectiveVariableKind.CENTER_OF_MASS_DISTANCE:
            if params.size != 1 or params[0] != int(params[0]):
                raise ValueError("Center-of-mass distance requires one integer split.")
            split = int(params[0])
            if index.ndim != 1 or not 0 < split < index.size:
                raise ValueError("Center-of-mass groups must both be non-empty.")
        if kind is CollectiveVariableKind.RADIUS_OF_GYRATION and index.size == 0:
            raise ValueError("Radius of gyration requires at least one index.")
        if kind is CollectiveVariableKind.COORDINATION:
            if index.size == 0 or index.size % 2 or params.size != 2:
                raise ValueError("Coordination requires pairs, distance, and power.")
            if params[0] <= 0.0 or params[1] <= 0.0:
                raise ValueError("Coordination distance and power must be positive.")
        if kind is CollectiveVariableKind.CONTACT_SIMILARITY:
            if (
                index.size == 0
                or index.size % 2
                or ref.size != index.size // 2
                or params.size != 1
                or params[0] <= 0.0
            ):
                raise ValueError("Contact similarity requires pairs, targets, and scale.")
        if kind is CollectiveVariableKind.ALIGNED_RMSD:
            if index.ndim != 1 or index.size < 3 or ref.shape != (index.size, 3):
                raise ValueError("Aligned RMSD requires three or more reference points.")
        if kind in (
            CollectiveVariableKind.PATH_PROGRESS,
            CollectiveVariableKind.PATH_DISTANCE,
        ):
            frame_size = index.size * 3
            if (
                index.ndim != 1
                or index.size == 0
                or ref.size < 2 * frame_size
                or ref.size % frame_size
                or params.size != 1
                or params[0] <= 0.0
            ):
                raise ValueError(
                    "Path CV requires at least two images and positive scale."
                )
        metric_ = (
            CollectiveVariableMetric(
                period=2.0 * np.pi if kind is CollectiveVariableKind.TORSION else None
            )
            if metric is None
            else metric
        )
        self.kind = kind
        self.indices = jnp.asarray(index, dtype=jnp.int32)
        self.parameters = jnp.asarray(params)
        self.reference = jnp.asarray(ref)
        self.domain = domain
        self.metric = metric_
        self.cv_id = canonical_fingerprint(
            {
                "kind": "atomistic-cv",
                "cv_kind": kind.value,
                "indices": array_tree_fingerprint(index),
                "parameters": array_tree_fingerprint(params),
                "reference": array_tree_fingerprint(ref),
                "domain": domain.value,
                "metric": metric_.metric_id,
            }
        )

    def prepare(self, system: PreparedAtomisticSystem, /) -> "PreparedCollectiveVariable":
        capacity = (
            system.capacity
            if self.domain is not AtomisticSiteDomain.INTERACTION_SITES
            else system.coordinate_map.plan.sites.capacity
        )
        if self.indices.size and (
            int(jnp.min(self.indices)) < 0 or int(jnp.max(self.indices)) >= capacity
        ):
            raise ValueError("CV index exceeds selected coordinate domain.")
        if self.domain is AtomisticSiteDomain.INTERACTION_SITES and self.kind in (
            CollectiveVariableKind.CENTER_OF_MASS_DISTANCE,
            CollectiveVariableKind.RADIUS_OF_GYRATION,
            CollectiveVariableKind.DENSITY,
        ):
            raise ValueError("Mass-weighted CVs require the physical DOF atom domain.")
        return PreparedCollectiveVariable(self, system)


class PreparedCollectiveVariable(StrictModule, NonTrainableState):
    plan: CollectiveVariablePlan
    system: PreparedAtomisticSystem
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan, system, /):
        self.plan = plan
        self.system = system
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-cv", "plan": plan.cv_id, "system": system.prepared_id}
        )

    def evaluate(
        self, positions: ArrayLike, /, *, cell=None, cell_vectors=None
    ) -> CollectiveVariableEvaluation:
        dof = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        coordinates = (
            self.system.coordinate_map.realize(dof, cell=cell).positions
            if self.plan.domain is AtomisticSiteDomain.INTERACTION_SITES
            else dof
        )
        index = self.plan.indices
        kind = self.plan.kind
        margin = jnp.asarray(jnp.inf, dtype=dof.dtype)
        success = jnp.asarray(True)
        if kind is CollectiveVariableKind.DISTANCE:
            vector = coordinates[index[0]] - coordinates[index[1]]
            if cell is not None:
                vector = cell.minimum_image(vector)
            value = jnp.sqrt(jnp.sum(vector * vector))
            margin = value
        elif kind is CollectiveVariableKind.ANGLE:
            left = coordinates[index[0]] - coordinates[index[1]]
            right = coordinates[index[2]] - coordinates[index[1]]
            if cell is not None:
                left, right = cell.minimum_image(left), cell.minimum_image(right)
            cross_norm = jnp.sqrt(jnp.sum(jnp.cross(left, right) ** 2))
            value = jnp.arctan2(cross_norm, jnp.sum(left * right))
            margin = jnp.minimum(jnp.sqrt(jnp.sum(left**2)), jnp.sqrt(jnp.sum(right**2)))
            success = margin > 0.0
        elif kind is CollectiveVariableKind.TORSION:
            points = coordinates[index[:4]]
            b0, b1, b2 = (
                points[0] - points[1],
                points[2] - points[1],
                points[3] - points[2],
            )
            if cell is not None:
                b0, b1, b2 = (
                    cell.minimum_image(b0),
                    cell.minimum_image(b1),
                    cell.minimum_image(b2),
                )
            axis_norm = jnp.sqrt(jnp.sum(b1**2))
            axis = b1 / jnp.where(axis_norm > 0.0, axis_norm, 1.0)
            v = b0 - jnp.sum(b0 * axis) * axis
            w = b2 - jnp.sum(b2 * axis) * axis
            value = jnp.arctan2(jnp.sum(jnp.cross(axis, v) * w), jnp.sum(v * w))
            margin = jnp.minimum(
                axis_norm, jnp.minimum(jnp.sqrt(jnp.sum(v**2)), jnp.sqrt(jnp.sum(w**2)))
            )
            success = margin > 0.0
        elif kind is CollectiveVariableKind.CENTER_OF_MASS_DISTANCE:
            split = int(self.plan.parameters.reshape((-1,))[0])
            first, second = index[:split], index[split:]
            mass = self.system.plan.masses
            c1 = jnp.sum(mass[first, None] * coordinates[first], axis=0) / jnp.sum(
                mass[first]
            )
            c2 = jnp.sum(mass[second, None] * coordinates[second], axis=0) / jnp.sum(
                mass[second]
            )
            vector = c1 - c2
            if cell is not None:
                vector = cell.minimum_image(vector)
            value = jnp.sqrt(jnp.sum(vector**2))
        elif kind is CollectiveVariableKind.RADIUS_OF_GYRATION:
            selected = coordinates[index]
            mass = self.system.plan.masses[index]
            center = jnp.sum(mass[:, None] * selected, axis=0) / jnp.sum(mass)
            value = jnp.sqrt(
                jnp.sum(mass * jnp.sum((selected - center) ** 2, axis=-1)) / jnp.sum(mass)
            )
        elif kind is CollectiveVariableKind.COORDINATION:
            pairs = index.reshape((-1, 2))
            vector = coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
            if cell is not None:
                vector = cell.minimum_image(vector)
            distance = jnp.sqrt(jnp.sum(vector**2, axis=-1))
            r0, power = self.plan.parameters[:2]
            ratio = distance / r0
            value = jnp.sum(1.0 / (1.0 + ratio**power))
            margin = jnp.min(jnp.abs(distance - r0))
        elif kind is CollectiveVariableKind.CONTACT_SIMILARITY:
            pairs = index.reshape((-1, 2))
            vector = coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
            if cell is not None:
                vector = cell.minimum_image(vector)
            distance = jnp.sqrt(jnp.sum(vector**2, axis=-1))
            target = self.plan.reference.reshape((-1,))
            scale = self.plan.parameters.reshape(())
            value = jnp.mean(jnp.exp(-(((distance - target) / scale) ** 2)))
        elif kind is CollectiveVariableKind.ALIGNED_RMSD:
            selected = coordinates[index]
            reference = self.plan.reference.reshape(selected.shape).astype(selected.dtype)
            centered = selected - jnp.mean(selected, axis=0)
            centered_reference = reference - jnp.mean(reference, axis=0)
            covariance = contract("ni,nj->ij", centered, centered_reference)
            result = la.svd.svd(
                la.svd.SVDProblem(la.DenseLinearOperator(covariance)),
                policy=la.svd.SVDSolvePolicy(count=3),
            )
            left = jnp.asarray(result.left_vectors)
            right = jnp.asarray(result.right_vectors)
            provisional = left @ right.T
            determinant = jnp.sum(
                provisional[0] * jnp.cross(provisional[1], provisional[2])
            )
            correction = (
                jnp.eye(3, dtype=selected.dtype)
                .at[2, 2]
                .set(jnp.where(determinant >= 0.0, 1.0, -1.0))
            )
            rotation = jax.lax.stop_gradient(left @ correction @ right.T)
            aligned = centered @ rotation
            mean_squared = jnp.mean(jnp.sum((aligned - centered_reference) ** 2, axis=-1))
            value = jnp.where(
                mean_squared > 0.0,
                jnp.sqrt(jnp.where(mean_squared > 0.0, mean_squared, 1.0)),
                0.0,
            )
            margin = jnp.min(result.singular_values)
            success = result.successful
        elif kind is CollectiveVariableKind.CELL_VOLUME:
            vectors = (
                None
                if self.system.cell is None and cell_vectors is None
                else self.system.cell.vectors
                if cell_vectors is None
                else jnp.asarray(cell_vectors)
            )
            if vectors is None:
                raise ValueError("Cell-volume CV requires cell vectors.")
            value = jnp.abs(jnp.sum(vectors[0] * jnp.cross(vectors[1], vectors[2])))
        elif kind is CollectiveVariableKind.DENSITY:
            vectors = (
                None
                if self.system.cell is None and cell_vectors is None
                else self.system.cell.vectors
                if cell_vectors is None
                else jnp.asarray(cell_vectors)
            )
            if vectors is None:
                raise ValueError("Density CV requires cell vectors.")
            volume = jnp.abs(jnp.sum(vectors[0] * jnp.cross(vectors[1], vectors[2])))
            value = jnp.sum(self.system.plan.masses[self.system.active_mask]) / volume
        else:
            selected = coordinates[index]
            references = self.plan.reference.reshape((-1,) + selected.shape).astype(
                selected.dtype
            )
            distances = jnp.sqrt(
                jnp.mean(
                    jnp.sum((references - selected[None, ...]) ** 2, axis=-1), axis=-1
                )
            )
            scale = self.plan.parameters.reshape(())
            weights = jnp.exp(-scale * distances**2)
            if kind is CollectiveVariableKind.PATH_PROGRESS:
                value = jnp.sum(
                    jnp.arange(distances.size, dtype=distances.dtype) * weights
                ) / jnp.sum(weights)
            else:
                value = -jnp.log(jnp.sum(weights)) / scale
            pair_difference = jnp.abs(distances[:, None] - distances[None, :])
            margin = jnp.min(
                jnp.where(
                    jnp.eye(distances.size, dtype=bool),
                    jnp.inf,
                    pair_difference,
                )
            )
        success = success & jnp.all(jnp.isfinite(value))
        return CollectiveVariableEvaluation(
            jnp.asarray(value), margin, success, self.prepared_id
        )


class CollectiveVariableProgram(StrictModule, NonTrainableState):
    variables: tuple[PreparedCollectiveVariable, ...]
    program_id: str = eqx.field(static=True)

    def __init__(self, variables, /):
        values = tuple(variables)
        if not values or any(
            not isinstance(value, PreparedCollectiveVariable) for value in values
        ):
            raise TypeError("variables must contain prepared collective variables.")
        self.variables = values
        self.program_id = canonical_fingerprint(
            {"kind": "cv-program", "variables": [value.prepared_id for value in values]}
        )

    def evaluate(self, positions, /, **kwargs):
        evaluations = tuple(
            value.evaluate(positions, **kwargs) for value in self.variables
        )
        return jnp.stack(tuple(value.value for value in evaluations)), jnp.all(
            jnp.stack(tuple(value.successful for value in evaluations))
        )


__all__ = [
    "AbstractCollectiveVariablePlan",
    "CollectiveVariableEvaluation",
    "CollectiveVariableKind",
    "CollectiveVariableMetric",
    "CollectiveVariablePlan",
    "CollectiveVariableProgram",
    "PreparedCollectiveVariable",
]
