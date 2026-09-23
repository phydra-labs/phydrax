#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Multipolar, mutually polarizable, adaptive, and periodic multilevel QM/MM."""

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import (
    AtomisticUnitSystem,
    PermanentMultipoleSiteData,
    PolarizationOperatorPlan,
    PolarizationScaleData,
    PreparedPolarizationOperator,
)
from ...discretization import TopologyEpoch
from ...ein import contract
from ...linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    prepare,
    solve,
)
from .._context import PermanentMultipoleEmbeddingState, PolarizableEmbeddingState
from .._surface import (
    AbstractPreparedPotentialEnergySurface,
    PotentialEnergySurfaceCapabilities,
    PotentialEnergySurfaceEvaluation,
)
from ._core import EmbeddedRegionEvaluation, PreparedQuantumRegion, QMMMEvaluation


def multipole_embedding_for_region(
    region: PreparedQuantumRegion,
    positions: ArrayLike,
    multipoles: PermanentMultipoleSiteData,
    /,
) -> PermanentMultipoleEmbeddingState:
    if not isinstance(region, PreparedQuantumRegion) or not isinstance(
        multipoles, PermanentMultipoleSiteData
    ):
        raise TypeError(
            "Region multipole embedding requires typed region and multipoles."
        )
    coordinate = jnp.asarray(positions)
    capacity = region.plan.system.particle_ids.size
    if coordinate.shape != (capacity, 3) or multipoles.site_capacity != capacity:
        raise ValueError(
            "Full-system positions and multipoles must share fixed capacity."
        )
    indices = jnp.asarray(region.mm_indices)
    selected = PermanentMultipoleSiteData(
        multipoles.charges[indices],
        multipoles.dipoles[indices],
        multipoles.quadrupoles[indices],
        multipoles.polarizabilities[indices],
        multipoles.damping[indices],
    )
    return PermanentMultipoleEmbeddingState(
        jnp.asarray(region.plan.system.particle_ids)[indices],
        coordinate[indices],
        selected,
        region.plan.system.units,
    )


class PolarizableEmbeddedRegionEvaluation(StrictModule, NonTrainableState):
    embedded: EmbeddedRegionEvaluation
    electric_field_at_sites: Array
    successful: Array
    provider_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        embedded: EmbeddedRegionEvaluation,
        electric_field_at_sites: ArrayLike,
        successful: ArrayLike,
        provider_id: str,
        /,
    ):
        if not isinstance(embedded, EmbeddedRegionEvaluation):
            raise TypeError("embedded must be EmbeddedRegionEvaluation.")
        field = jnp.asarray(
            electric_field_at_sites, dtype=embedded.point_charge_forces.dtype
        )
        provider = str(provider_id).strip()
        if field.shape != embedded.point_charge_forces.shape or not provider:
            raise ValueError("Polarizable embedded field must align with MM sites.")
        valid = (
            jnp.asarray(successful, dtype=jnp.bool_)
            & embedded.successful
            & jnp.all(jnp.isfinite(field))
        )
        self.embedded = embedded
        self.electric_field_at_sites = field
        self.successful = valid
        self.provider_id = provider
        self.result_id = canonical_fingerprint(
            {
                "kind": "polarizable-embedded-region-evaluation",
                "embedded": embedded.result_id,
                "provider": provider,
                "successful": bool(valid),
                "field": array_tree_fingerprint(np.asarray(field)),
            }
        )


class AbstractPolarizableEmbeddedRegionProvider(StrictModule, NonTrainableState):
    provider_id: eqx.AbstractVar[str]
    region_system_id: eqx.AbstractVar[str]
    unit_system_id: eqx.AbstractVar[str]
    conservative: eqx.AbstractVar[bool]

    @abc.abstractmethod
    def evaluate(
        self,
        region_positions: ArrayLike,
        permanent_embedding: PermanentMultipoleEmbeddingState,
        induced_dipoles: ArrayLike,
        /,
    ) -> PolarizableEmbeddedRegionEvaluation:
        raise NotImplementedError


PolarizableEmbeddedEvaluator = Callable[
    [ArrayLike, PermanentMultipoleEmbeddingState, ArrayLike],
    PolarizableEmbeddedRegionEvaluation,
]


class CallablePolarizableEmbeddedRegionProvider(
    AbstractPolarizableEmbeddedRegionProvider
):
    evaluator: PolarizableEmbeddedEvaluator = eqx.field(static=True)
    declared_provider_id: str = eqx.field(static=True)
    region_system_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluator: PolarizableEmbeddedEvaluator,
        provider_id: str,
        region_system_id: str,
        units: AtomisticUnitSystem,
        /,
        *,
        conservative: bool = True,
    ):
        if not callable(evaluator) or not isinstance(units, AtomisticUnitSystem):
            raise TypeError("Polarizable provider requires callable evaluator and units.")
        declared = str(provider_id).strip()
        region = str(region_system_id).strip()
        if not declared or not region:
            raise ValueError("Polarizable provider identities must be non-empty.")
        self.evaluator = evaluator
        self.declared_provider_id = declared
        self.region_system_id = region
        self.unit_system_id = units.unit_system_id
        self.conservative = bool(conservative)
        self.provider_id = canonical_fingerprint(
            {
                "kind": "callable-polarizable-embedded-provider",
                "declared_provider": declared,
                "region_system": region,
                "unit_system": units.unit_system_id,
                "conservative": self.conservative,
            }
        )

    def evaluate(
        self,
        region_positions: ArrayLike,
        permanent_embedding: PermanentMultipoleEmbeddingState,
        induced_dipoles: ArrayLike,
        /,
    ) -> PolarizableEmbeddedRegionEvaluation:
        result = self.evaluator(region_positions, permanent_embedding, induced_dipoles)
        if (
            not isinstance(result, PolarizableEmbeddedRegionEvaluation)
            or result.provider_id != self.declared_provider_id
        ):
            raise ValueError("Polarizable provider changed result type or identity.")
        return PolarizableEmbeddedRegionEvaluation(
            result.embedded,
            result.electric_field_at_sites,
            result.successful,
            self.provider_id,
        )


class MutualPolarizationResult(StrictModule, NonTrainableState):
    qmmm: QMMMEvaluation
    polarizable_embedding: PolarizableEmbeddingState
    mutual_residual: Array
    iterations: Array
    successful: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self, qmmm, polarizable_embedding, mutual_residual, iterations, successful, /
    ):
        if not isinstance(qmmm, QMMMEvaluation) or not isinstance(
            polarizable_embedding, PolarizableEmbeddingState
        ):
            raise TypeError(
                "Mutual polarization result requires typed QM/MM and embedding states."
            )
        residual = jnp.asarray(mutual_residual).reshape(())
        self.qmmm = qmmm
        self.polarizable_embedding = polarizable_embedding
        self.mutual_residual = residual
        self.iterations = jnp.asarray(iterations, dtype=jnp.int32).reshape(())
        self.successful = jnp.asarray(successful, dtype=jnp.bool_).reshape(())
        self.result_id = canonical_fingerprint(
            {
                "kind": "mutual-polarization-result",
                "qmmm": qmmm.result_id,
                "embedding": polarizable_embedding.embedding_id,
                "iterations": int(self.iterations),
                "successful": bool(self.successful),
                "residual": float(residual),
            }
        )


class MutualPolarizableQMMMSurface(AbstractPreparedPotentialEnergySurface):
    region: PreparedQuantumRegion
    classical_partition: AbstractPreparedPotentialEnergySurface
    quantum_provider: AbstractPolarizableEmbeddedRegionProvider
    multipoles: PermanentMultipoleSiteData
    polarization_operator: PreparedPolarizationOperator
    damping: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(
        self,
        region: PreparedQuantumRegion,
        classical_partition: AbstractPreparedPotentialEnergySurface,
        quantum_provider: AbstractPolarizableEmbeddedRegionProvider,
        multipoles: PermanentMultipoleSiteData,
        polarization_operator: PolarizationOperatorPlan,
        /,
        *,
        scaling: PolarizationScaleData | None = None,
        damping: float = 0.5,
        residual_tolerance: float = 1.0e-8,
        maximum_iterations: int = 100,
    ):
        if (
            not isinstance(region, PreparedQuantumRegion)
            or not isinstance(classical_partition, AbstractPreparedPotentialEnergySurface)
            or not isinstance(quantum_provider, AbstractPolarizableEmbeddedRegionProvider)
            or not isinstance(multipoles, PermanentMultipoleSiteData)
            or not isinstance(polarization_operator, PolarizationOperatorPlan)
        ):
            raise TypeError(
                "Mutual QM/MM requires typed region, surfaces, multipoles, and operator."
            )
        system = region.plan.system
        if (
            classical_partition.system_id != system.system_id
            or quantum_provider.region_system_id != region.region_system.system_id
            or quantum_provider.unit_system_id != system.units.unit_system_id
            or multipoles.site_capacity != system.particle_ids.size
        ):
            raise ValueError("Mutual QM/MM components do not share system identities.")
        if not classical_partition.capabilities.forces:
            raise ValueError(
                "Mutual polarizable QM/MM requires a force-capable classical partition."
            )
        indices = jnp.asarray(region.mm_indices)
        selected = PermanentMultipoleSiteData(
            multipoles.charges[indices],
            multipoles.dipoles[indices],
            multipoles.quadrupoles[indices],
            multipoles.polarizabilities[indices],
            multipoles.damping[indices],
        )
        if scaling is not None and scaling.direct.shape != (indices.size, indices.size):
            raise ValueError("Polarization scaling must align with MM sites.")
        prepared_operator = polarization_operator.prepare(selected, scaling=scaling)
        damping_ = float(damping)
        tolerance = float(residual_tolerance)
        maximum = int(maximum_iterations)
        if (
            not isfinite(damping_)
            or not 0.0 < damping_ <= 1.0
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or maximum <= 0
        ):
            raise ValueError(
                "Mutual polarization damping, tolerance, or limit is invalid."
            )
        self.region = region
        self.classical_partition = classical_partition
        self.quantum_provider = quantum_provider
        self.multipoles = multipoles
        self.polarization_operator = prepared_operator
        self.damping = damping_
        self.residual_tolerance = tolerance
        self.maximum_iterations = maximum
        self.system_id = system.system_id
        self.units = system.units
        self.provider_id = canonical_fingerprint(
            {
                "kind": "mutual-polarizable-qmmm-provider",
                "quantum": quantum_provider.provider_id,
                "classical": classical_partition.provider_id,
            }
        )
        self.surface_id = canonical_fingerprint(
            {
                "kind": "mutual-polarizable-qmmm-surface",
                "region": region.prepared_id,
                "classical": classical_partition.surface_id,
                "quantum": quantum_provider.provider_id,
                "multipoles": multipoles.multipole_id,
                "polarization_operator": prepared_operator.prepared_id,
                "damping": damping_,
                "residual_tolerance": tolerance,
                "maximum_iterations": maximum,
            }
        )
        self.capabilities = PotentialEnergySurfaceCapabilities(
            forces=True,
            conservative=classical_partition.capabilities.conservative
            and quantum_provider.conservative,
        )

    def evaluate_components(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> MutualPolarizationResult:
        coordinate = jnp.asarray(positions)
        region_positions = self.region.realize(coordinate)
        permanent = multipole_embedding_for_region(
            self.region, coordinate, self.multipoles
        )
        mm_positions = permanent.positions
        site_count = mm_positions.shape[0]
        induced = jnp.zeros((site_count, 3), dtype=coordinate.dtype)
        residual = jnp.asarray(jnp.inf, dtype=coordinate.dtype)
        zero = jnp.zeros_like(induced)
        operator_at_zero = self.polarization_operator.apply(
            mm_positions, zero, cell_vectors=cell_vectors
        )
        matrix = jax.jacfwd(
            lambda value: self.polarization_operator.apply(
                mm_positions,
                value.reshape((site_count, 3)),
                cell_vectors=cell_vectors,
            ).action.reshape((-1,))
        )(zero.reshape((-1,)))
        prepared_linear = prepare(
            LeastSquaresProblem(DenseLinearOperator(matrix)),
            LinearSolvePolicy(DenseSVD()),
        )
        converged = False
        iteration_successful = True
        completed = 0
        quantum = None
        for iteration in range(self.maximum_iterations):
            quantum = self.quantum_provider.evaluate(region_positions, permanent, induced)
            right = (operator_at_zero.p_field + quantum.electric_field_at_sites).reshape(
                (-1,)
            )
            linear = solve(prepared_linear, right)
            target = linear.value.reshape((site_count, 3))
            updated = (1.0 - self.damping) * induced + self.damping * target
            candidate_residual = jnp.max(jnp.abs(updated - induced), initial=0.0)
            iteration_valid = bool(
                linear.successful
                & quantum.successful
                & jnp.all(jnp.isfinite(updated))
                & jnp.isfinite(candidate_residual)
            )
            completed = iteration + 1
            if not iteration_valid:
                iteration_successful = False
                residual = jnp.asarray(
                    jnp.finfo(coordinate.dtype).max, dtype=coordinate.dtype
                )
                break
            induced = updated
            residual = candidate_residual
            if bool(residual <= self.residual_tolerance):
                converged = True
                break
        if quantum is None:
            raise RuntimeError("Polarizable QM/MM iteration did not execute.")
        if converged:
            quantum = self.quantum_provider.evaluate(region_positions, permanent, induced)
            iteration_successful = iteration_successful and bool(quantum.successful)
        classical = self.classical_partition.evaluate(coordinate, cell_vectors)
        if converged and iteration_successful and bool(classical.successful):
            frozen_induced = jax.lax.stop_gradient(induced)

            def polarization_function(mm_coordinate):
                operator = self.polarization_operator.apply(
                    mm_coordinate,
                    frozen_induced,
                    cell_vectors=cell_vectors,
                )
                return 0.5 * contract(
                    "nd,nd->", frozen_induced, operator.action
                ) - contract("nd,nd->", frozen_induced, operator.p_field)

            polarization_energy, polarization_gradient = jax.value_and_grad(
                polarization_function
            )(mm_positions)
            full_quantum_force = self.region.pullback(quantum.embedded.region_forces)
            site_force = quantum.embedded.point_charge_forces - polarization_gradient
            full_site_force = (
                jnp.zeros_like(coordinate)
                .at[jnp.asarray(self.region.mm_indices)]
                .add(site_force)
            )
            total_energy = (
                classical.energy + quantum.embedded.energy + polarization_energy
            )
            total_forces = classical.forces + full_quantum_force + full_site_force
            successful = jnp.isfinite(total_energy) & jnp.all(jnp.isfinite(total_forces))
        else:
            polarization_energy = jnp.asarray(jnp.nan, dtype=coordinate.dtype)
            site_force = jnp.full_like(mm_positions, jnp.nan)
            total_energy = jnp.asarray(jnp.nan, dtype=coordinate.dtype)
            total_forces = jnp.full_like(coordinate, jnp.nan)
            successful = jnp.asarray(False)
        qmmm = QMMMEvaluation(
            total_energy,
            classical.energy,
            quantum.embedded.energy,
            polarization_energy,
            total_forces,
            successful,
            (classical.source_result_id, quantum.result_id),
            point_charge_forces=site_force,
        )
        polarizable = PolarizableEmbeddingState(
            permanent,
            induced,
            residual,
            self.polarization_operator.prepared_id,
        )
        return MutualPolarizationResult(
            qmmm,
            polarizable,
            residual,
            completed,
            successful,
        )

    def evaluate(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> PotentialEnergySurfaceEvaluation:
        result = self.evaluate_components(positions, cell_vectors)
        return PotentialEnergySurfaceEvaluation(
            result.qmmm.total_energy,
            result.qmmm.forces,
            None,
            result.successful,
            provider_id=self.provider_id,
            source_result_id=result.result_id,
        )


AdaptiveWeightFunction = Callable[[Array], Array]


class AdaptiveQMMMEvaluation(StrictModule, NonTrainableState):
    evaluation: PotentialEnergySurfaceEvaluation
    weights: Array
    dominant_partition: Array
    partition_entropy: Array
    topology_epoch: TopologyEpoch
    successful: Array
    result_id: str = eqx.field(static=True)


class AdaptivePartitionedQMMMSurface(AbstractPreparedPotentialEnergySurface):
    partitions: tuple[AbstractPreparedPotentialEnergySurface, ...]
    weight_function: AdaptiveWeightFunction = eqx.field(static=True)
    partition_ids: tuple[str, ...] = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(
        self,
        partitions: Sequence[AbstractPreparedPotentialEnergySurface],
        weight_function: AdaptiveWeightFunction,
        partition_ids: Sequence[str],
        /,
    ):
        surfaces = tuple(partitions)
        ids = tuple(str(value).strip() for value in partition_ids)
        if (
            not surfaces
            or any(
                not isinstance(value, AbstractPreparedPotentialEnergySurface)
                for value in surfaces
            )
            or not callable(weight_function)
        ):
            raise TypeError(
                "Adaptive QM/MM requires surfaces and a differentiable weight function."
            )
        if (
            len(ids) != len(surfaces)
            or len(set(ids)) != len(ids)
            or any(not value for value in ids)
        ):
            raise ValueError(
                "Adaptive QM/MM partition identities must be unique and aligned."
            )
        system_id = surfaces[0].system_id
        units = surfaces[0].units
        if any(
            value.system_id != system_id
            or value.units.unit_system_id != units.unit_system_id
            for value in surfaces
        ):
            raise ValueError(
                "Adaptive QM/MM partition surfaces must share system and units."
            )
        if any(not value.capabilities.forces for value in surfaces):
            raise ValueError("Adaptive QM/MM requires force-capable partition surfaces.")
        self.partitions = surfaces
        self.weight_function = weight_function
        self.partition_ids = ids
        self.system_id = system_id
        self.units = units
        self.provider_id = canonical_fingerprint(
            {"kind": "adaptive-partitioned-qmmm-provider", "partitions": ids}
        )
        self.surface_id = canonical_fingerprint(
            {
                "kind": "adaptive-partitioned-qmmm-surface",
                "partitions": [value.surface_id for value in surfaces],
                "partition_ids": ids,
            }
        )
        self.capabilities = PotentialEnergySurfaceCapabilities(
            forces=True,
            conservative=all(value.capabilities.conservative for value in surfaces),
            differentiable=False,
        )

    def evaluate_partition(
        self,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None = None,
        /,
        *,
        epoch_index: int = 0,
    ) -> AdaptiveQMMMEvaluation:
        coordinate = jnp.asarray(positions)
        weights = jnp.asarray(self.weight_function(coordinate))
        if weights.shape != (len(self.partitions),):
            raise ValueError("Adaptive weights must provide one value per partition.")
        if bool(jnp.any(weights < 0.0)) or not bool(
            jnp.isclose(jnp.sum(weights), 1.0, atol=1.0e-10)
        ):
            raise ValueError(
                "Adaptive partition weights must be non-negative and sum to one."
            )
        values = tuple(
            surface.evaluate(coordinate, cell_vectors) for surface in self.partitions
        )
        energies = jnp.stack(tuple(value.energy for value in values))
        forces = jnp.stack(tuple(value.forces for value in values))
        weight_jacobian = jax.jacfwd(self.weight_function)(coordinate)
        blended_energy = jnp.dot(weights, energies)
        blended_forces = contract("p,pnd->nd", weights, forces) - contract(
            "p,pnd->nd", energies, weight_jacobian
        )
        dominant = jnp.argmax(weights)
        partition_id = canonical_fingerprint(
            {
                "kind": "adaptive-qmmm-partition",
                "ids": self.partition_ids,
                "dominant": int(dominant),
                "weights": array_tree_fingerprint(np.asarray(weights)),
            }
        )
        epoch = TopologyEpoch(
            epoch_index,
            canonical_fingerprint(
                {
                    "kind": "adaptive-qmmm-geometry",
                    "coordinates": array_tree_fingerprint(np.asarray(coordinate)),
                }
            ),
            self.system_id,
            partition_id,
        )
        successful = jnp.all(
            jnp.stack(tuple(value.successful for value in values))
        ) & jnp.all(jnp.isfinite(blended_forces))
        evaluation = PotentialEnergySurfaceEvaluation(
            blended_energy,
            blended_forces,
            None,
            successful,
            provider_id=self.provider_id,
            source_result_id=epoch.epoch_id,
        )
        entropy = -jnp.sum(jnp.where(weights > 0.0, weights * jnp.log(weights), 0.0))
        return AdaptiveQMMMEvaluation(
            evaluation,
            weights,
            dominant,
            entropy,
            epoch,
            successful,
            canonical_fingerprint(
                {
                    "kind": "adaptive-qmmm-evaluation",
                    "surface": self.surface_id,
                    "epoch": epoch.epoch_id,
                    "sources": [value.source_result_id for value in values],
                }
            ),
        )

    def evaluate(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> PotentialEnergySurfaceEvaluation:
        return self.evaluate_partition(positions, cell_vectors).evaluation


class PeriodicMultilevelQMMMSurface(AbstractPreparedPotentialEnergySurface):
    levels: tuple[AbstractPreparedPotentialEnergySurface, ...]
    coefficients: tuple[float, ...] = eqx.field(static=True)
    state_id: str = eqx.field(static=True)
    system_id: str = eqx.field(static=True)
    units: AtomisticUnitSystem
    provider_id: str = eqx.field(static=True)
    surface_id: str = eqx.field(static=True)
    capabilities: PotentialEnergySurfaceCapabilities

    def __init__(
        self,
        levels: Sequence[AbstractPreparedPotentialEnergySurface],
        coefficients: Sequence[float],
        state_id: str,
        /,
    ):
        surfaces = tuple(levels)
        coefficients_ = tuple(float(value) for value in coefficients)
        state = str(state_id).strip()
        if (
            not surfaces
            or len(surfaces) != len(coefficients_)
            or any(
                not isinstance(value, AbstractPreparedPotentialEnergySurface)
                for value in surfaces
            )
        ):
            raise ValueError("Multilevel QM/MM surfaces and coefficients must align.")
        if any(not isfinite(value) for value in coefficients_) or not state:
            raise ValueError("Multilevel coefficients and state identity are invalid.")
        system_id = surfaces[0].system_id
        units = surfaces[0].units
        if any(
            value.system_id != system_id
            or value.units.unit_system_id != units.unit_system_id
            for value in surfaces
        ):
            raise ValueError("Multilevel QM/MM surfaces must share system and units.")
        if any(not value.capabilities.forces for value in surfaces):
            raise ValueError(
                "Periodic multilevel QM/MM requires force-capable level surfaces."
            )
        self.levels = surfaces
        self.coefficients = coefficients_
        self.state_id = state
        self.system_id = system_id
        self.units = units
        self.provider_id = canonical_fingerprint(
            {"kind": "periodic-multilevel-qmmm-provider", "state": state}
        )
        self.surface_id = canonical_fingerprint(
            {
                "kind": "periodic-multilevel-qmmm-surface",
                "state": state,
                "levels": [value.surface_id for value in surfaces],
                "coefficients": coefficients_,
            }
        )
        self.capabilities = PotentialEnergySurfaceCapabilities(
            forces=True,
            conservative=all(value.capabilities.conservative for value in surfaces),
        )

    def evaluate(
        self, positions: ArrayLike, cell_vectors: ArrayLike | None = None, /
    ) -> PotentialEnergySurfaceEvaluation:
        if cell_vectors is None:
            raise ValueError("Periodic multilevel QM/MM requires cell vectors.")
        values = tuple(level.evaluate(positions, cell_vectors) for level in self.levels)
        coefficients = jnp.asarray(self.coefficients)
        energy = jnp.dot(coefficients, jnp.stack(tuple(value.energy for value in values)))
        forces = contract(
            "l,lnd->nd",
            coefficients,
            jnp.stack(tuple(value.forces for value in values)),
        )
        successful = jnp.all(jnp.stack(tuple(value.successful for value in values)))
        return PotentialEnergySurfaceEvaluation(
            energy,
            forces,
            None,
            successful,
            provider_id=self.provider_id,
            source_result_id=canonical_fingerprint(
                {
                    "kind": "periodic-multilevel-qmmm-evaluation",
                    "surface": self.surface_id,
                    "sources": [value.source_result_id for value in values],
                }
            ),
        )


__all__ = [
    "AbstractPolarizableEmbeddedRegionProvider",
    "AdaptivePartitionedQMMMSurface",
    "AdaptiveQMMMEvaluation",
    "CallablePolarizableEmbeddedRegionProvider",
    "MutualPolarizableQMMMSurface",
    "MutualPolarizationResult",
    "PeriodicMultilevelQMMMSurface",
    "PolarizableEmbeddedRegionEvaluation",
    "multipole_embedding_for_region",
]
