#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class LocalMHDEntityFactors(StrictModule):
    face_factors: tuple[Array, ...]
    edge_factors: tuple[Array, ...]


class LocalMHDPositivityResult(StrictModule):
    cell_state: Array
    magnetic_flux: Array
    cell_factors: Array
    magnetic_factor: Array
    successful: Array


class LocalMHDPositivityPlan(StrictModule, NonTrainableState):
    spatial: object
    iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, spatial, /, *, iterations: int = 24):
        from ..discretization.finite_volume import UpwindConstrainedTransportPlan

        count = int(iterations)
        if not isinstance(spatial, UpwindConstrainedTransportPlan) or count <= 0:
            raise ValueError("Local MHD positivity plan is invalid.")
        self.spatial = spatial
        self.iterations = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "local-mhd-positivity",
                "spatial": spatial.plan_id,
                "iterations": count,
            }
        )

    def entity_factors(self, cell_factors: Array, /) -> LocalMHDEntityFactors:
        factors = jnp.asarray(cell_factors)
        if factors.shape != self.spatial.cell_shape:
            raise ValueError("Cell positivity factors do not match the MHD grid.")
        face_factors = []
        for axis, grid_axis in enumerate(self.spatial.bridge.grid.structured_axes):
            if grid_axis.periodic:
                face = jnp.minimum(factors, jnp.roll(factors, -1, axis=axis))
            else:
                lower = jnp.take(factors, jnp.asarray([0]), axis=axis)
                upper = jnp.take(
                    factors, jnp.asarray([factors.shape[axis] - 1]), axis=axis
                )
                interior = jnp.minimum(
                    jnp.take(factors, jnp.arange(factors.shape[axis] - 1), axis=axis),
                    jnp.take(factors, jnp.arange(1, factors.shape[axis]), axis=axis),
                )
                face = jnp.concatenate((lower, interior, upper), axis=axis)
            face_factors.append(face)
        edge_factors = []
        if self.spatial.layout.dimension >= 2:
            for orientation in combinations(
                range(self.spatial.layout.dimension),
                int(self.spatial.layout.electromotive_degree),
            ):
                transverse = tuple(
                    axis
                    for axis in range(self.spatial.layout.dimension)
                    if axis not in orientation
                )
                edge = factors
                for axis in transverse:
                    if self.spatial.bridge.grid.structured_axes[axis].periodic:
                        edge = jnp.minimum(edge, jnp.roll(edge, -1, axis=axis))
                    else:
                        lower = jnp.take(edge, jnp.asarray([0]), axis=axis)
                        upper = jnp.take(
                            edge, jnp.asarray([edge.shape[axis] - 1]), axis=axis
                        )
                        interior = jnp.minimum(
                            jnp.take(
                                edge,
                                jnp.arange(edge.shape[axis] - 1),
                                axis=axis,
                            ),
                            jnp.take(
                                edge,
                                jnp.arange(1, edge.shape[axis]),
                                axis=axis,
                            ),
                        )
                        edge = jnp.concatenate((lower, interior, upper), axis=axis)
                edge_factors.append(edge)
        return LocalMHDEntityFactors(tuple(face_factors), tuple(edge_factors))

    def limit_integrals(
        self,
        low_face_integrals: tuple[Array, ...],
        high_face_integrals: tuple[Array, ...],
        low_edge_integrals: Array,
        high_edge_integrals: Array,
        cell_factors: Array,
        /,
    ) -> tuple[tuple[Array, ...], Array, LocalMHDEntityFactors]:
        entity = self.entity_factors(cell_factors)
        face = tuple(
            low + factor[..., None] * (high - low)
            for low, high, factor in zip(
                low_face_integrals,
                high_face_integrals,
                entity.face_factors,
                strict=True,
            )
        )
        if self.spatial.layout.dimension == 1:
            edge = jnp.zeros((0,), dtype=low_edge_integrals.dtype)
        else:
            degree = int(self.spatial.layout.electromotive_degree)
            low_components = self.spatial.bridge.unpack(degree, low_edge_integrals)
            high_components = self.spatial.bridge.unpack(degree, high_edge_integrals)
            edge = self.spatial.bridge.pack(
                degree,
                tuple(
                    low + factor * (high - low)
                    for low, high, factor in zip(
                        low_components,
                        high_components,
                        entity.edge_factors,
                        strict=True,
                    )
                ),
            )
        return face, edge, entity

    def apply(
        self,
        low_cell: Array,
        low_magnetic: Array,
        high_cell: Array,
        high_magnetic: Array,
        /,
    ) -> LocalMHDPositivityResult:
        low_full = self.spatial.full_state(low_cell, low_magnetic)
        if not jnp.all(self.spatial.dynamics.system.admissible(low_full)):
            raise ValueError("Low-order MHD state must be admissible.")

        def body(_, bounds):
            lower, upper = bounds
            middle = 0.5 * (lower + upper)
            cell = low_cell + middle[..., None] * (high_cell - low_cell)
            magnetic_factor = jnp.min(middle)
            magnetic = low_magnetic + magnetic_factor * (high_magnetic - low_magnetic)
            valid = self.spatial.dynamics.system.admissible(
                self.spatial.full_state(cell, magnetic)
            )
            return jnp.where(valid, middle, lower), jnp.where(valid, upper, middle)

        shape = low_cell.shape[:-1]
        lower, _ = jax.lax.fori_loop(
            0,
            self.iterations,
            body,
            (
                jnp.zeros(shape, dtype=low_cell.dtype),
                jnp.ones(shape, dtype=low_cell.dtype),
            ),
        )
        magnetic_factor = jnp.min(lower)
        cell = low_cell + lower[..., None] * (high_cell - low_cell)
        magnetic = low_magnetic + magnetic_factor * (high_magnetic - low_magnetic)
        successful = jnp.all(
            self.spatial.dynamics.system.admissible(
                self.spatial.full_state(cell, magnetic)
            )
        )
        return LocalMHDPositivityResult(
            cell_state=cell,
            magnetic_flux=magnetic,
            cell_factors=lower,
            magnetic_factor=magnetic_factor,
            successful=successful,
        )


class DualEnergyMHDState(StrictModule):
    material_internal_energy: Array
    entropy_density: Array


class DualEnergyMHDPlan(StrictModule, NonTrainableState):
    gamma: float = eqx.field(static=True)
    switch_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, gamma: float, /, *, switch_fraction: float = 1e-3):
        gamma_ = float(gamma)
        fraction = float(switch_fraction)
        if gamma_ <= 1.0 or not 0.0 < fraction < 1.0:
            raise ValueError("Dual-energy MHD controls are invalid.")
        self.gamma = gamma_
        self.switch_fraction = fraction
        self.plan_id = canonical_fingerprint(
            {"kind": "dual-energy-mhd", "gamma": gamma_, "switch_fraction": fraction}
        )

    def initialize(self, full_state: Array, /) -> DualEnergyMHDState:
        density = full_state[..., 0]
        kinetic = 0.5 * jnp.sum(full_state[..., 1:4] ** 2, axis=-1) / density
        magnetic = 0.5 * jnp.sum(full_state[..., 5:8] ** 2, axis=-1)
        internal = full_state[..., 4] - kinetic - magnetic
        entropy = (self.gamma - 1.0) * internal / density**self.gamma
        return DualEnergyMHDState(internal, density * entropy)

    def synchronize(
        self,
        full_state: Array,
        auxiliary: DualEnergyMHDState,
        /,
    ) -> tuple[Array, DualEnergyMHDState, Array]:
        density = full_state[..., 0]
        kinetic = 0.5 * jnp.sum(full_state[..., 1:4] ** 2, axis=-1) / density
        magnetic = 0.5 * jnp.sum(full_state[..., 5:8] ** 2, axis=-1)
        total_internal = full_state[..., 4] - kinetic - magnetic
        use_auxiliary = total_internal < self.switch_fraction * full_state[..., 4]
        entropy_internal = (
            auxiliary.entropy_density * density ** (self.gamma - 1.0) / (self.gamma - 1.0)
        )
        internal = jnp.where(use_auxiliary, entropy_internal, total_internal)
        synchronized = full_state.at[..., 4].set(kinetic + magnetic + internal)
        next_auxiliary = DualEnergyMHDState(
            internal,
            density * (self.gamma - 1.0) * internal / density**self.gamma,
        )
        return synchronized, next_auxiliary, use_auxiliary


class MHDCTUPredictorPlan(StrictModule, NonTrainableState):
    spatial: object
    predictor_id: str = eqx.field(static=True)

    def __init__(self, spatial, /):
        self.spatial = spatial
        self.predictor_id = canonical_fingerprint(
            {"kind": "mhd-ctu-half-step-predictor", "spatial": spatial.plan_id}
        )

    def predict(
        self,
        time: Array,
        cell_state: Array,
        magnetic_flux: Array,
        step_size: Array,
        args=None,
        /,
    ) -> tuple[Array, Array, object]:
        rate = self.spatial.rate(time, cell_state, magnetic_flux, args)
        return (
            cell_state + 0.5 * step_size * rate.cell_rate,
            magnetic_flux + 0.5 * step_size * rate.magnetic_rate,
            rate,
        )


class MHDCharacteristicReconstructionPlan(StrictModule, NonTrainableState):
    eigensystem: Callable = eqx.field(static=True)
    declared_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, eigensystem: Callable, /, *, declared_id: str):
        if not callable(eigensystem) or not declared_id:
            raise ValueError("MHD characteristic reconstruction metadata is invalid.")
        self.eigensystem = eigensystem
        self.declared_id = str(declared_id)
        self.plan_id = canonical_fingerprint(
            {"kind": "mhd-characteristic-reconstruction", "declared_id": declared_id}
        )

    def project(
        self,
        left_state: Array,
        right_state: Array,
        axis: int,
        args=None,
        /,
    ) -> tuple[Array, Array, Array]:
        left_matrix, _right_matrix, speeds = self.eigensystem(
            left_state, right_state, int(axis), args
        )
        left_characteristic = oe.contract("...ij,...j->...i", left_matrix, left_state)
        right_characteristic = oe.contract("...ij,...j->...i", left_matrix, right_state)
        return left_characteristic, right_characteristic, speeds


__all__ = [
    "LocalMHDEntityFactors",
    "DualEnergyMHDPlan",
    "DualEnergyMHDState",
    "LocalMHDPositivityPlan",
    "LocalMHDPositivityResult",
    "MHDCharacteristicReconstructionPlan",
    "MHDCTUPredictorPlan",
]
