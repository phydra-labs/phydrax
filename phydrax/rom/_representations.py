#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ..discretization import FieldTransfer
from ..linalg import AbstractVectorSpace
from ._basis import ReducedBasisArtifact


class ReferencePhysicalRepresentation(StrictModule, NonTrainableState):
    """Qualified bidirectional physical/reference field representation."""

    physical_to_reference: FieldTransfer
    reference_to_physical: FieldTransfer
    geometry_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    maximum_round_trip_defect: float = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_to_reference: FieldTransfer,
        reference_to_physical: FieldTransfer,
        probe_vectors: ArrayLike,
        /,
        *,
        geometry_id: str,
        topology_id: str,
        tolerance: float = 1.0e-8,
    ):
        if not isinstance(physical_to_reference, FieldTransfer) or not isinstance(
            reference_to_physical, FieldTransfer
        ):
            raise TypeError("Reference representation requires two FieldTransfer values.")
        if (
            physical_to_reference.source.field_space_id
            != reference_to_physical.target.field_space_id
            or physical_to_reference.target.field_space_id
            != reference_to_physical.source.field_space_id
        ):
            raise ValueError(
                "Reference transfer directions must reverse the same field spaces."
            )
        probes = jnp.asarray(probe_vectors)
        physical_space = physical_to_reference.source.vector_space
        if probes.ndim != 2 or probes.shape[1] != physical_space.size:
            raise ValueError(
                "probe_vectors must have shape (probes, physical space size)."
            )
        transferred = physical_to_reference.primal_operator.mv_block(
            jnp.swapaxes(probes, 0, 1)
        )
        returned = reference_to_physical.primal_operator.mv_block(transferred)
        defect = float(np.max(np.abs(np.asarray(jnp.swapaxes(returned, 0, 1) - probes))))
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ < 0.0 or defect > tolerance_:
            raise ValueError(
                "Reference transfer round-trip exceeds the declared tolerance."
            )
        geometry = str(geometry_id)
        topology = str(topology_id)
        if not geometry or not topology:
            raise ValueError("Geometry and topology IDs must be non-empty.")
        self.physical_to_reference = physical_to_reference
        self.reference_to_physical = reference_to_physical
        self.geometry_id = geometry
        self.topology_id = topology
        self.maximum_round_trip_defect = defect
        self.representation_id = canonical_fingerprint(
            {
                "kind": "reference-physical-representation",
                "physical_to_reference": physical_to_reference.transfer_id,
                "reference_to_physical": reference_to_physical.transfer_id,
                "geometry": geometry,
                "topology": topology,
                "round_trip_defect": defect,
            }
        )


class ReducedBasisAtlasArtifact(StrictModule, NonTrainableState):
    bases: tuple[ReducedBasisArtifact, ...]
    centers: Array
    radii: Array
    transition_matrices: Array
    maximum_cycle_defect: Array
    parameter_contract_id: str = eqx.field(static=True)
    atlas_id: str = eqx.field(static=True)

    def __init__(
        self,
        bases: Sequence[ReducedBasisArtifact],
        centers: ArrayLike,
        radii: ArrayLike,
        transition_matrices: ArrayLike,
        /,
        *,
        parameter_contract_id: str,
        maximum_cycle_tolerance: float = 1.0e-6,
    ):
        charts = tuple(bases)
        if not charts:
            raise ValueError("A basis atlas requires at least one chart.")
        if any(not isinstance(chart, ReducedBasisArtifact) for chart in charts):
            raise TypeError("Atlas charts must be ReducedBasisArtifact values.")
        binding = {
            (chart.support_id, chart.measure_id, chart.geometry_id) for chart in charts
        }
        if len(binding) != 1:
            raise ValueError(
                "Atlas charts must share reference support, measure, and geometry."
            )
        center = jnp.asarray(centers)
        radius = jnp.asarray(radii)
        transitions = jnp.asarray(transition_matrices)
        count = len(charts)
        rank = charts[0].rank
        if any(chart.rank != rank for chart in charts):
            raise ValueError("Initial basis atlas requires one common reduced rank.")
        if center.ndim != 2 or center.shape[0] != count or radius.shape != (count,):
            raise ValueError("Atlas centers and radii must align with chart count.")
        if transitions.shape != (count, count, rank, rank):
            raise ValueError("Atlas transition matrices have invalid shape.")
        if np.any(np.asarray(radius) <= 0.0):
            raise ValueError("Atlas radii must be positive.")
        identity = jnp.eye(rank, dtype=transitions.dtype)
        cycle = jnp.max(
            jnp.abs(
                jax.vmap(
                    lambda row, reverse: jax.vmap(
                        lambda left, right: left @ right - identity
                    )(row, reverse)
                )(transitions, jnp.swapaxes(transitions, 0, 1))
            )
        )
        tolerance = float(maximum_cycle_tolerance)
        if float(np.asarray(cycle)) > tolerance:
            raise ValueError("Atlas transition cycle defect exceeds tolerance.")
        parameter = str(parameter_contract_id)
        if not parameter:
            raise ValueError("parameter_contract_id must be non-empty.")
        self.bases = charts
        self.centers = center
        self.radii = radius
        self.transition_matrices = transitions
        self.maximum_cycle_defect = cycle
        self.parameter_contract_id = parameter
        self.atlas_id = canonical_fingerprint(
            {
                "kind": "reduced-basis-atlas",
                "bases": [chart.artifact_id for chart in charts],
                "parameter_contract": parameter,
                "content": array_tree_fingerprint(
                    {"centers": center, "radii": radius, "transitions": transitions}
                )["sha256"],
            }
        )

    def route(self, parameters: ArrayLike, /) -> tuple[int, Array]:
        value = jnp.asarray(parameters)
        if value.shape != self.centers.shape[1:]:
            raise ValueError("Atlas parameters have invalid shape.")
        distances = jnp.linalg.norm(self.centers - value[None, :], axis=-1)
        normalized = distances / self.radii
        index = int(np.argmin(np.asarray(normalized)))
        if float(np.asarray(normalized[index])) > 1.0:
            raise ValueError("Parameters lie outside every atlas chart.")
        return index, normalized[index]

    def transition(self, source: int, target: int, coordinates: ArrayLike, /) -> Array:
        return self.transition_matrices[int(source), int(target)] @ jnp.asarray(
            coordinates
        )


class AbstractReferenceStateChart(StrictModule, NonTrainableState):
    latent_space: AbstractAttribute[AbstractVectorSpace]
    reference_space: AbstractAttribute[AbstractVectorSpace]
    support_id: AbstractAttribute[str]
    geometry_id: AbstractAttribute[str]
    chart_id: AbstractAttribute[str]

    @abc.abstractmethod
    def decode(self, latent: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def jvp(self, latent: ArrayLike, tangent: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def vjp(self, latent: ArrayLike, cotangent: ArrayLike, /) -> Array:
        raise NotImplementedError


class QuadraticStateChart(AbstractReferenceStateChart):
    offset: Array
    linear_basis: Array
    quadratic_basis: Array
    latent_space: AbstractVectorSpace
    reference_space: AbstractVectorSpace
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)
    symmetric_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(
        self,
        latent_space: AbstractVectorSpace,
        reference_space: AbstractVectorSpace,
        offset: ArrayLike,
        linear_basis: ArrayLike,
        quadratic_basis: ArrayLike,
        /,
        *,
        support_id: str,
        geometry_id: str,
    ):
        if not isinstance(latent_space, AbstractVectorSpace) or not isinstance(
            reference_space, AbstractVectorSpace
        ):
            raise TypeError("Chart spaces must be AbstractVectorSpace values.")
        rank = latent_space.size
        pairs = tuple(
            (left, right) for left in range(rank) for right in range(left, rank)
        )
        offset_ = jnp.asarray(offset)
        linear = jnp.asarray(linear_basis)
        quadratic = jnp.asarray(quadratic_basis)
        if (
            offset_.shape != (reference_space.size,)
            or linear.shape != (reference_space.size, rank)
            or quadratic.shape != (reference_space.size, len(pairs))
        ):
            raise ValueError("Quadratic chart arrays have invalid shape.")
        support = str(support_id)
        geometry = str(geometry_id)
        if not support or not geometry:
            raise ValueError("Chart support and geometry IDs must be non-empty.")
        self.offset = offset_
        self.linear_basis = linear
        self.quadratic_basis = quadratic
        self.latent_space = latent_space
        self.reference_space = reference_space
        self.support_id = support
        self.geometry_id = geometry
        self.symmetric_pairs = pairs
        self.chart_id = canonical_fingerprint(
            {
                "kind": "quadratic-state-chart",
                "latent_space": latent_space.space_id,
                "reference_space": reference_space.space_id,
                "support": support,
                "geometry": geometry,
                "content": array_tree_fingerprint(
                    {"offset": offset_, "linear": linear, "quadratic": quadratic}
                )["sha256"],
            }
        )

    def _monomials(self, latent: Array) -> Array:
        return jnp.stack(
            tuple(latent[left] * latent[right] for left, right in self.symmetric_pairs)
        )

    def decode(self, latent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        if value.shape != (self.latent_space.size,):
            raise ValueError("Latent coordinate shape is invalid.")
        return (
            self.offset
            + self.linear_basis @ value
            + self.quadratic_basis @ self._monomials(value)
        )

    def jvp(self, latent: ArrayLike, tangent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        direction = jnp.asarray(tangent)
        return jax.jvp(self.decode, (value,), (direction,))[1]

    def vjp(self, latent: ArrayLike, cotangent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        covector = jnp.asarray(cotangent)
        _, pullback = jax.vjp(self.decode, value)
        return pullback(covector)[0]


class CoordinateConditionedStateChart(AbstractReferenceStateChart):
    """Fixed decoder queried on one declared reference support."""

    decoder: object = eqx.field(static=True)
    query_points: Array
    latent_space: AbstractVectorSpace
    reference_space: AbstractVectorSpace
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    decoder_id: str = eqx.field(static=True)
    chart_id: str = eqx.field(static=True)

    def __init__(
        self,
        decoder,
        query_points: ArrayLike,
        latent_space: AbstractVectorSpace,
        reference_space: AbstractVectorSpace,
        /,
        *,
        decoder_id: str,
        support_id: str,
        geometry_id: str,
    ):
        if not callable(decoder):
            raise TypeError("decoder must be callable.")
        if not isinstance(latent_space, AbstractVectorSpace) or not isinstance(
            reference_space, AbstractVectorSpace
        ):
            raise TypeError("Chart spaces must be AbstractVectorSpace values.")
        query = jnp.asarray(query_points)
        if query.ndim != 2:
            raise ValueError("query_points must have shape (points, coordinates).")
        identifiers = tuple(str(value) for value in (decoder_id, support_id, geometry_id))
        if any(not value for value in identifiers):
            raise ValueError("Decoder, support, and geometry IDs must be non-empty.")
        probe = jnp.zeros((latent_space.size,), dtype=query.dtype)
        decoded = jnp.asarray(decoder(probe, query)).reshape((-1,))
        if decoded.shape != (reference_space.size,):
            raise ValueError("Decoder output does not match the reference-space size.")
        self.decoder = decoder
        self.query_points = query
        self.latent_space = latent_space
        self.reference_space = reference_space
        self.decoder_id = identifiers[0]
        self.support_id = identifiers[1]
        self.geometry_id = identifiers[2]
        self.chart_id = canonical_fingerprint(
            {
                "kind": "coordinate-conditioned-state-chart",
                "decoder": identifiers[0],
                "latent_space": latent_space.space_id,
                "reference_space": reference_space.space_id,
                "support": identifiers[1],
                "geometry": identifiers[2],
                "query": array_tree_fingerprint(query)["sha256"],
            }
        )

    def decode(self, latent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        if value.shape != (self.latent_space.size,):
            raise ValueError("Latent coordinate shape is invalid.")
        return jnp.asarray(self.decoder(value, self.query_points)).reshape((-1,))

    def jvp(self, latent: ArrayLike, tangent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        direction = jnp.asarray(tangent)
        return jax.jvp(self.decode, (value,), (direction,))[1]

    def vjp(self, latent: ArrayLike, cotangent: ArrayLike, /) -> Array:
        value = jnp.asarray(latent)
        _, pullback = jax.vjp(self.decode, value)
        return pullback(jnp.asarray(cotangent))[0]


__all__ = [
    "AbstractReferenceStateChart",
    "CoordinateConditionedStateChart",
    "QuadraticStateChart",
    "ReducedBasisAtlasArtifact",
    "ReferencePhysicalRepresentation",
]
