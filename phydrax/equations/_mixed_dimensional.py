#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Conservative bulk-network-reservoir transport with fixed sparse couplings."""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..discretization import (
    CellMesh,
    PreparedEmbeddedMeasureTransfer,
    PreparedMetricNetwork,
    TetrahedralConnectivity,
)
from ..ein import contract
from ..linalg import (
    ArraySpace,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)


class BulkDGTransportEvidence(StrictModule):
    minimum_volume: Array
    minimum_transmissibility: Array
    finite: Array
    conservative: Array
    successful: Array


class PreparedBulkDGTransport(StrictModule):
    mesh: CellMesh
    mass: Array
    owner: Array
    neighbour: Array
    volume_flux: Array
    transmissibility: Array
    boundary_owner: Array
    boundary_volume_flux: Array
    boundary_inflow_concentration: Array
    removal_rate: Array
    evidence: BulkDGTransportEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def cell_count(self) -> int:
        return int(self.mass.shape[0])

    def advective_residual(self, concentration: ArrayLike, /) -> Array:
        value = jnp.asarray(concentration)
        if value.shape != self.mass.shape:
            raise ValueError("Bulk concentration must contain one value per cell.")
        upwind = jnp.where(
            self.volume_flux >= 0.0,
            value[self.owner],
            value[self.neighbour],
        )
        flux = self.volume_flux * upwind
        result = jnp.zeros_like(value).at[self.owner].add(flux)
        result = result.at[self.neighbour].add(-flux)
        boundary_trace = jnp.where(
            self.boundary_volume_flux >= 0.0,
            value[self.boundary_owner],
            self.boundary_inflow_concentration,
        )
        return result.at[self.boundary_owner].add(
            self.boundary_volume_flux * boundary_trace
        )

    def diffusive_residual(self, concentration: ArrayLike, /) -> Array:
        value = jnp.asarray(concentration)
        if value.shape != self.mass.shape:
            raise ValueError("Bulk concentration must contain one value per cell.")
        flux = -self.transmissibility * (value[self.neighbour] - value[self.owner])
        result = jnp.zeros_like(value).at[self.owner].add(flux)
        return result.at[self.neighbour].add(-flux)

    def residual(self, concentration: ArrayLike, /) -> Array:
        value = jnp.asarray(concentration)
        return (
            self.advective_residual(value)
            + self.diffusive_residual(value)
            + self.mass * self.removal_rate * value
        )

    def external_loss_rate(self, concentration: ArrayLike, /) -> Array:
        value = jnp.asarray(concentration)
        if value.shape != self.mass.shape:
            raise ValueError("Bulk concentration must contain one value per cell.")
        boundary_trace = jnp.where(
            self.boundary_volume_flux >= 0.0,
            value[self.boundary_owner],
            self.boundary_inflow_concentration,
        )
        return jnp.sum(self.boundary_volume_flux * boundary_trace) + jnp.vdot(
            self.mass * self.removal_rate, value
        )

    def total_mass(self, concentration: ArrayLike, /) -> Array:
        value = jnp.asarray(concentration)
        if value.shape != self.mass.shape:
            raise ValueError("Bulk concentration must contain one value per cell.")
        return jnp.vdot(self.mass, value)


@dataclass(frozen=True, slots=True)
class BulkDGTransportPlan:
    mesh: CellMesh
    porosity: np.ndarray
    diffusivity: np.ndarray
    velocity: np.ndarray
    boundary_volume_flux: np.ndarray | None = None
    boundary_inflow_concentration: np.ndarray | None = None
    removal_rate: np.ndarray | None = None
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, CellMesh) or not isinstance(
            self.mesh.connectivity, TetrahedralConnectivity
        ):
            raise TypeError("Bulk DG0 transport requires a tetrahedral CellMesh.")
        count = sum(block.cell_count for block in self.mesh.blocks)
        porosity = np.asarray(self.porosity, dtype=float)
        diffusivity = np.asarray(self.diffusivity, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        if (
            porosity.shape != (count,)
            or np.any(~np.isfinite(porosity))
            or np.any(porosity <= 0.0)
        ):
            raise ValueError("porosity must be finite and positive per cell.")
        if diffusivity.shape not in ((count,), (count, 3, 3)):
            raise ValueError("diffusivity must be scalar or 3x3 tensor per cell.")
        if np.any(~np.isfinite(diffusivity)):
            raise ValueError("diffusivity must be finite.")
        if diffusivity.ndim == 1:
            if np.any(diffusivity < 0.0):
                raise ValueError("Scalar diffusivity must be non-negative.")
        else:
            symmetric = 0.5 * (diffusivity + np.swapaxes(diffusivity, -1, -2))
            tolerance = (
                128.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(diffusivity))))
            )
            if not np.allclose(
                diffusivity, symmetric, atol=tolerance, rtol=0.0
            ) or np.any(np.linalg.eigvalsh(symmetric) < -tolerance):
                raise ValueError(
                    "Tensor diffusivity must be symmetric positive semidefinite."
                )
            diffusivity = symmetric
        if velocity.shape != (count, 3) or np.any(~np.isfinite(velocity)):
            raise ValueError("velocity must have finite shape (cell_count, 3).")
        connectivity = self.mesh.connectivity
        boundary_count = int(
            np.count_nonzero(np.asarray(connectivity.boundary_faces, dtype=bool))
        )
        boundary_flux = (
            np.zeros((boundary_count,), dtype=float)
            if self.boundary_volume_flux is None
            else np.asarray(self.boundary_volume_flux, dtype=float)
        )
        boundary_inflow = (
            np.zeros((boundary_count,), dtype=float)
            if self.boundary_inflow_concentration is None
            else np.asarray(self.boundary_inflow_concentration, dtype=float)
        )
        removal = (
            np.zeros((count,), dtype=float)
            if self.removal_rate is None
            else np.asarray(self.removal_rate, dtype=float)
        )
        if boundary_flux.shape == ():
            boundary_flux = np.full((boundary_count,), float(boundary_flux))
        if boundary_inflow.shape == ():
            boundary_inflow = np.full((boundary_count,), float(boundary_inflow))
        if removal.shape == ():
            removal = np.full((count,), float(removal))
        if boundary_flux.shape != (boundary_count,) or np.any(
            ~np.isfinite(boundary_flux)
        ):
            raise ValueError("boundary_volume_flux must be finite per boundary face.")
        if (
            boundary_inflow.shape != (boundary_count,)
            or np.any(~np.isfinite(boundary_inflow))
            or np.any(boundary_inflow < 0.0)
        ):
            raise ValueError(
                "boundary_inflow_concentration must be finite and non-negative."
            )
        if (
            removal.shape != (count,)
            or np.any(~np.isfinite(removal))
            or np.any(removal < 0.0)
        ):
            raise ValueError("removal_rate must be finite and non-negative per cell.")
        normalized = (
            ("porosity", porosity),
            ("diffusivity", diffusivity),
            ("velocity", velocity),
            ("boundary_volume_flux", boundary_flux),
            ("boundary_inflow_concentration", boundary_inflow),
            ("removal_rate", removal),
        )
        for name, value in normalized:
            copied = np.array(value, copy=True)
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "bulk-dg0-transport-plan",
                    "mesh": self.mesh.mesh_id,
                    **{name: array_tree_fingerprint(value) for name, value in normalized},
                }
            ),
        )

    def prepare(self) -> PreparedBulkDGTransport:
        connectivity = self.mesh.connectivity
        if not isinstance(connectivity, TetrahedralConnectivity):
            raise TypeError("Bulk DG0 transport requires tetrahedral connectivity.")
        boundary_flux = self.boundary_volume_flux
        boundary_inflow = self.boundary_inflow_concentration
        removal = self.removal_rate
        if boundary_flux is None or boundary_inflow is None or removal is None:
            raise RuntimeError("Normalized bulk boundary/removal data are unavailable.")
        cells = np.concatenate([np.asarray(block.vertices) for block in self.mesh.blocks])
        points = np.asarray(self.mesh.coordinates)
        tetrahedra = points[cells]
        volume = np.abs(np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1])) / 6.0
        centroids = np.mean(tetrahedra, axis=1)
        faces = np.asarray(connectivity.faces)
        cross = np.cross(
            points[faces[:, 1]] - points[faces[:, 0]],
            points[faces[:, 2]] - points[faces[:, 0]],
        )
        cross_norm = np.linalg.norm(cross, axis=1)
        if np.any(cross_norm <= 0.0):
            raise ValueError("Bulk mesh faces must have positive area.")
        face_area = 0.5 * cross_norm
        face_normal = cross / cross_norm[:, None]
        incidents = [[] for _ in faces]
        for cell_index, face_row in enumerate(np.asarray(connectivity.cell_faces)):
            for face_index in face_row:
                incidents[int(face_index)].append(cell_index)
        owners = []
        neighbours = []
        normals = []
        areas = []
        distances = []
        boundary_owners = []
        for face_index, adjacent in enumerate(incidents):
            if len(adjacent) == 1:
                boundary_owners.append(adjacent[0])
                continue
            if len(adjacent) != 2:
                raise ValueError("Bulk mesh faces must have one or two incident cells.")
            owner, neighbour = adjacent
            normal = face_normal[face_index]
            if np.dot(normal, centroids[neighbour] - centroids[owner]) < 0.0:
                normal = -normal
            owners.append(owner)
            neighbours.append(neighbour)
            normals.append(normal)
            areas.append(face_area[face_index])
            distances.append(
                abs(float(np.dot(centroids[neighbour] - centroids[owner], normal)))
            )
        owner = np.asarray(owners, dtype=np.int32)
        neighbour = np.asarray(neighbours, dtype=np.int32)
        normal = np.asarray(normals, dtype=float).reshape((-1, 3))
        area = np.asarray(areas)
        distance = np.asarray(distances)
        boundary_owner = np.asarray(boundary_owners, dtype=np.int32)
        if boundary_owner.shape != boundary_flux.shape:
            raise RuntimeError(
                "Boundary face ordering and boundary data are inconsistent."
            )
        if np.any(distance <= 0.0) or np.any(area <= 0.0):
            raise ValueError("Bulk mesh contains a degenerate interior face route.")
        face_velocity = 0.5 * (self.velocity[owner] + self.velocity[neighbour])
        volume_flux = area * np.sum(face_velocity * normal, axis=1)
        if self.diffusivity.ndim == 1:
            owner_diffusion = self.diffusivity[owner]
            neighbour_diffusion = self.diffusivity[neighbour]
        else:
            owner_diffusion = np.sum(
                normal * contract("fij,fj->fi", self.diffusivity[owner], normal),
                axis=1,
            )
            neighbour_diffusion = np.sum(
                normal * contract("fij,fj->fi", self.diffusivity[neighbour], normal),
                axis=1,
            )
        denominator = owner_diffusion + neighbour_diffusion
        harmonic = np.zeros_like(denominator)
        np.divide(
            2.0 * owner_diffusion * neighbour_diffusion,
            denominator,
            out=harmonic,
            where=denominator > 0.0,
        )
        transmissibility = area * harmonic / distance
        mass = self.porosity * volume
        finite = bool(
            np.all(np.isfinite(volume_flux))
            and np.all(np.isfinite(transmissibility))
            and np.all(np.isfinite(mass))
        )
        evidence = BulkDGTransportEvidence(
            jnp.asarray(volume.min()),
            jnp.asarray(transmissibility.min() if transmissibility.size else 0.0),
            jnp.asarray(finite),
            jnp.asarray(True),
            jnp.asarray(finite and bool(np.all(volume > 0.0))),
        )
        return PreparedBulkDGTransport(
            self.mesh,
            jnp.asarray(mass),
            jnp.asarray(owner),
            jnp.asarray(neighbour),
            jnp.asarray(volume_flux),
            jnp.asarray(transmissibility),
            jnp.asarray(boundary_owner),
            jnp.asarray(boundary_flux),
            jnp.asarray(boundary_inflow),
            jnp.asarray(removal),
            evidence,
            canonical_fingerprint(
                {
                    "kind": "prepared-bulk-dg0-transport",
                    "plan": self.plan_id,
                    "owner": array_tree_fingerprint(owner),
                    "neighbour": array_tree_fingerprint(neighbour),
                    "volume_flux": array_tree_fingerprint(volume_flux),
                    "transmissibility": array_tree_fingerprint(transmissibility),
                    "boundary_owner": array_tree_fingerprint(boundary_owner),
                }
            ),
        )


class PreparedNetworkTransport(StrictModule):
    network: PreparedMetricNetwork
    diffusivity: Array
    volume_flow: Array
    prepared_id: str = eqx.field(static=True)

    @property
    def mass(self) -> Array:
        return self.network.node_measures

    def advective_residual(self, concentration: ArrayLike, /) -> Array:
        flux = self.network.advective_flux(concentration, self.volume_flow)
        return -self.network.divergence(flux)

    def diffusive_residual(self, concentration: ArrayLike, /) -> Array:
        gradient = self.network.gradient(concentration)
        flux = -self.network.areas * self.diffusivity * gradient
        return -self.network.divergence(flux)

    def residual(self, concentration: ArrayLike, /) -> Array:
        return self.advective_residual(concentration) + self.diffusive_residual(
            concentration
        )

    def total_mass(self, concentration: ArrayLike, /) -> Array:
        return self.network.mass(concentration)


@dataclass(frozen=True, slots=True)
class NetworkTransportPlan:
    network: PreparedMetricNetwork
    diffusivity: np.ndarray
    volume_flow: np.ndarray
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.network, PreparedMetricNetwork):
            raise TypeError("network must be PreparedMetricNetwork.")
        count = len(self.network.lengths)
        diffusivity = np.asarray(self.diffusivity, dtype=float)
        flow = np.asarray(self.volume_flow, dtype=float)
        if diffusivity.shape == ():
            diffusivity = np.full((count,), float(diffusivity))
        if (
            diffusivity.shape != (count,)
            or np.any(~np.isfinite(diffusivity))
            or np.any(diffusivity < 0.0)
        ):
            raise ValueError("diffusivity must be finite and non-negative per edge.")
        if flow.shape != (count,) or np.any(~np.isfinite(flow)):
            raise ValueError("volume_flow must be finite per edge.")
        diffusivity = np.array(diffusivity, copy=True)
        flow = np.array(flow, copy=True)
        diffusivity.setflags(write=False)
        flow.setflags(write=False)
        object.__setattr__(self, "diffusivity", diffusivity)
        object.__setattr__(self, "volume_flow", flow)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "network-transport-plan",
                    "network": self.network.network_id,
                    "diffusivity": array_tree_fingerprint(diffusivity),
                    "flow": array_tree_fingerprint(flow),
                }
            ),
        )

    def prepare(self) -> PreparedNetworkTransport:
        return PreparedNetworkTransport(
            self.network,
            jnp.asarray(self.diffusivity),
            jnp.asarray(self.volume_flow),
            self.plan_id,
        )


class PermeabilityExchangePlan(StrictModule):
    transfer: PreparedEmbeddedMeasureTransfer
    coefficients: Array
    exchange_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedEmbeddedMeasureTransfer,
        coefficients: ArrayLike,
        /,
    ):
        if not isinstance(transfer, PreparedEmbeddedMeasureTransfer):
            raise TypeError("transfer must be PreparedEmbeddedMeasureTransfer.")
        coefficient = jnp.asarray(coefficients)
        if coefficient.shape == ():
            coefficient = jnp.broadcast_to(coefficient, transfer.target_measures.shape)
        if coefficient.shape != transfer.target_measures.shape:
            raise ValueError(
                "Exchange coefficients must be scalar or one per target site."
            )
        coefficient = eqx.error_if(
            coefficient,
            jnp.any(~jnp.isfinite(coefficient)) | jnp.any(coefficient < 0.0),
            "Exchange coefficients must be finite and non-negative.",
        )
        self.transfer = transfer
        self.coefficients = coefficient
        self.exchange_id = canonical_fingerprint(
            {
                "kind": "permeability-exchange",
                "transfer": transfer.transfer_id,
                "coefficients": array_tree_fingerprint(coefficient),
            }
        )

    def residuals(
        self, bulk_concentration: ArrayLike, network_concentration: ArrayLike, /
    ) -> tuple[Array, Array, Array]:
        bulk = jnp.asarray(bulk_concentration)
        network = jnp.asarray(network_concentration)
        if network.shape != self.transfer.target_measures.shape:
            raise ValueError("network_concentration must match exchange target sites.")
        average = self.transfer.average(bulk).values
        density = self.coefficients * (average - network)
        integrated = self.transfer.target_measures * density
        bulk_out = self.transfer.dual_pullback(integrated)
        network_out = -integrated
        return bulk_out, network_out, density


class ReservoirCouplingPlan(StrictModule):
    network_indices: Array
    volumes: Array
    coefficients: Array
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        network_indices: ArrayLike,
        volumes: ArrayLike,
        coefficients: ArrayLike,
        /,
        *,
        network_size: int,
    ):
        indices = np.asarray(network_indices)
        volume = np.asarray(volumes, dtype=float)
        coefficient = np.asarray(coefficients, dtype=float)
        if not np.issubdtype(indices.dtype, np.integer) or indices.ndim != 1:
            raise TypeError("network_indices must be one integer vector.")
        if volume.shape != indices.shape or coefficient.shape != indices.shape:
            raise ValueError("Reservoir arrays must have one value per reservoir.")
        if (
            np.any(indices < 0)
            or np.any(indices >= network_size)
            or np.unique(indices).size != len(indices)
        ):
            raise ValueError("Reservoir network indices must be unique and in bounds.")
        if np.any(~np.isfinite(volume)) or np.any(volume <= 0.0):
            raise ValueError("Reservoir volumes must be finite and positive.")
        if np.any(~np.isfinite(coefficient)) or np.any(coefficient < 0.0):
            raise ValueError("Reservoir coefficients must be finite and non-negative.")
        self.network_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.volumes = jnp.asarray(volume)
        self.coefficients = jnp.asarray(coefficient)
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "reservoir-coupling",
                "indices": array_tree_fingerprint(indices),
                "volumes": array_tree_fingerprint(volume),
                "coefficients": array_tree_fingerprint(coefficient),
            }
        )

    def residuals(
        self, network_concentration: ArrayLike, reservoir_concentration: ArrayLike, /
    ) -> tuple[Array, Array, Array]:
        network = jnp.asarray(network_concentration)
        reservoir = jnp.asarray(reservoir_concentration)
        if reservoir.shape != self.volumes.shape:
            raise ValueError("reservoir_concentration must match reservoir count.")
        density = self.coefficients * (network[self.network_indices] - reservoir)
        integrated = self.volumes * density
        network_out = jnp.zeros_like(network).at[self.network_indices].add(integrated)
        return network_out, -integrated, density


class MixedDimensionalTransportState(StrictModule):
    bulk: Array
    network: Array
    reservoirs: Array


class MixedDimensionalSources(StrictModule):
    bulk: Array
    network: Array
    reservoirs: Array


class MixedDimensionalMassLedger(StrictModule):
    bulk_mass: Array
    network_mass: Array
    reservoir_mass: Array
    total_mass: Array
    external_loss_rate: Array
    exchange_defect: Array
    balance_defect: Array
    minimum_concentration: Array
    finite: Array
    successful: Array


class MixedDimensionalStepResult(StrictModule):
    state: MixedDimensionalTransportState
    candidate_state: MixedDimensionalTransportState
    ledger: MixedDimensionalMassLedger
    linear_status: Array
    accepted: Array
    runtime_id: str = eqx.field(static=True)


class PreparedMixedDimensionalTransport(StrictModule):
    bulk: PreparedBulkDGTransport
    network: PreparedNetworkTransport
    exchange: PermeabilityExchangePlan
    reservoirs: ReservoirCouplingPlan
    require_nonnegative: bool = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    @property
    def sizes(self) -> tuple[int, int, int]:
        return self.bulk.cell_count, len(self.network.mass), len(self.reservoirs.volumes)

    def _validate_state(self, state: MixedDimensionalTransportState, /) -> None:
        if not isinstance(state, MixedDimensionalTransportState):
            raise TypeError("state must be MixedDimensionalTransportState.")
        expected = tuple((size,) for size in self.sizes)
        actual = (state.bulk.shape, state.network.shape, state.reservoirs.shape)
        if actual != expected:
            raise ValueError(f"State shapes must be {expected}; got {actual}.")

    def mass_vector(self, dtype) -> Array:
        return jnp.concatenate(
            (
                self.bulk.mass.astype(dtype),
                self.network.mass.astype(dtype),
                self.reservoirs.volumes.astype(dtype),
            )
        )

    def flatten(self, state: MixedDimensionalTransportState, /) -> Array:
        self._validate_state(state)
        return jnp.concatenate((state.bulk, state.network, state.reservoirs))

    def unflatten(self, values: ArrayLike, /) -> MixedDimensionalTransportState:
        array = jnp.asarray(values)
        if array.shape != (sum(self.sizes),):
            raise ValueError("Flattened mixed-dimensional state has the wrong size.")
        bulk_end = self.sizes[0]
        network_end = bulk_end + self.sizes[1]
        return MixedDimensionalTransportState(
            array[:bulk_end], array[bulk_end:network_end], array[network_end:]
        )

    def residual(
        self,
        state: MixedDimensionalTransportState,
        /,
        *,
        include_advection: bool = True,
    ) -> tuple[MixedDimensionalTransportState, Array]:
        self._validate_state(state)
        bulk_transport = (
            self.bulk.diffusive_residual(state.bulk)
            + self.bulk.mass * self.bulk.removal_rate * state.bulk
        )
        network_transport = self.network.diffusive_residual(state.network)
        if include_advection:
            bulk_transport = bulk_transport + self.bulk.advective_residual(state.bulk)
            network_transport = network_transport + self.network.advective_residual(
                state.network
            )
        bulk_exchange, network_exchange, exchange_density = self.exchange.residuals(
            state.bulk, state.network
        )
        reservoir_network, reservoir, _ = self.reservoirs.residuals(
            state.network, state.reservoirs
        )
        return (
            MixedDimensionalTransportState(
                bulk_transport + bulk_exchange,
                network_transport + network_exchange + reservoir_network,
                reservoir,
            ),
            exchange_density,
        )

    def step_backward_euler(
        self,
        state: MixedDimensionalTransportState,
        dt: float,
        /,
        *,
        sources: MixedDimensionalSources | None = None,
        policy: LinearSolvePolicy | None = None,
    ) -> MixedDimensionalStepResult:
        return self._step(state, dt, sources=sources, policy=policy, imex=False)

    def step_imex_euler(
        self,
        state: MixedDimensionalTransportState,
        dt: float,
        /,
        *,
        sources: MixedDimensionalSources | None = None,
        policy: LinearSolvePolicy | None = None,
    ) -> MixedDimensionalStepResult:
        return self._step(state, dt, sources=sources, policy=policy, imex=True)

    def _step(
        self,
        state: MixedDimensionalTransportState,
        dt: float,
        *,
        sources: MixedDimensionalSources | None,
        policy: LinearSolvePolicy | None,
        imex: bool,
    ) -> MixedDimensionalStepResult:
        self._validate_state(state)
        width = float(dt)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("dt must be finite and positive.")
        old = self.flatten(state)
        mass = self.mass_vector(old.dtype)
        if sources is None:
            source = jnp.zeros_like(old)
        else:
            source = jnp.concatenate((sources.bulk, sources.network, sources.reservoirs))
            if source.shape != old.shape:
                raise ValueError("Mixed-dimensional sources must match the state shapes.")
        explicit = jnp.zeros_like(old)
        if imex:
            advective = MixedDimensionalTransportState(
                self.bulk.advective_residual(state.bulk),
                self.network.advective_residual(state.network),
                jnp.zeros_like(state.reservoirs),
            )
            explicit = self.flatten(advective)

        def action(candidate):
            candidate_state = self.unflatten(candidate)
            residual, _ = self.residual(candidate_state, include_advection=not imex)
            return mass * candidate + width * self.flatten(residual)

        space = ArraySpace(old.shape, dtype=old.dtype)
        operator = FunctionLinearOperator(
            action,
            source=space,
            target=space,
            operator_id=f"{self.runtime_id}:{'imex' if imex else 'be'}:{width:.17g}",
        )
        right = mass * old + width * (source - explicit)
        solved = solve(LinearSystem(operator), right, policy=policy)
        candidate = self.unflatten(solved.value)
        accepted = jnp.all(solved.successful) & jnp.all(jnp.isfinite(solved.value))
        initial_total = self.ledger(state).total_mass
        if imex:
            external_loss = jnp.sum(self.bulk.advective_residual(state.bulk)) + jnp.vdot(
                self.bulk.mass * self.bulk.removal_rate, candidate.bulk
            )
        else:
            external_loss = self.bulk.external_loss_rate(candidate.bulk)
        ledger = self.ledger(
            candidate,
            reference_total=initial_total + width * (jnp.sum(source) - external_loss),
        )
        accepted = accepted & ledger.successful
        result_state = MixedDimensionalTransportState(
            jnp.where(accepted, candidate.bulk, state.bulk),
            jnp.where(accepted, candidate.network, state.network),
            jnp.where(accepted, candidate.reservoirs, state.reservoirs),
        )
        return MixedDimensionalStepResult(
            result_state,
            candidate,
            ledger,
            solved.status,
            accepted,
            self.runtime_id,
        )

    def ledger(
        self,
        state: MixedDimensionalTransportState,
        /,
        *,
        reference_total: ArrayLike | None = None,
    ) -> MixedDimensionalMassLedger:
        self._validate_state(state)
        bulk = self.bulk.total_mass(state.bulk)
        network = self.network.total_mass(state.network)
        reservoir = jnp.vdot(self.reservoirs.volumes, state.reservoirs)
        total = bulk + network + reservoir
        external_loss = self.bulk.external_loss_rate(state.bulk)
        bulk_exchange, network_exchange, _ = self.exchange.residuals(
            state.bulk, state.network
        )
        reservoir_network, reservoir_exchange, _ = self.reservoirs.residuals(
            state.network, state.reservoirs
        )
        exchange_defect = jnp.abs(
            jnp.sum(bulk_exchange)
            + jnp.sum(network_exchange)
            + jnp.sum(reservoir_network)
            + jnp.sum(reservoir_exchange)
        )
        reference = total if reference_total is None else jnp.asarray(reference_total)
        balance = jnp.abs(total - reference) / jnp.maximum(1.0, jnp.abs(reference))
        minimum = jnp.minimum(
            jnp.min(state.bulk),
            jnp.minimum(jnp.min(state.network), jnp.min(state.reservoirs)),
        )
        finite = jnp.all(jnp.isfinite(self.flatten(state))) & jnp.isfinite(total)
        positivity = (
            minimum >= -1.0e-12 if self.require_nonnegative else jnp.asarray(True)
        )
        successful = (
            finite & positivity & (exchange_defect <= 1.0e-10) & (balance <= 1.0e-8)
        )
        return MixedDimensionalMassLedger(
            bulk,
            network,
            reservoir,
            total,
            external_loss,
            exchange_defect,
            balance,
            minimum,
            finite,
            successful,
        )

    def diagonal_preconditioner(self, dt: float, /) -> Array:
        width = float(dt)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("dt must be finite and positive.")
        mass = self.mass_vector(self.bulk.mass.dtype)
        exchange_diagonal = jnp.concatenate(
            (
                jnp.zeros_like(self.bulk.mass),
                self.exchange.coefficients * self.exchange.transfer.target_measures,
                self.reservoirs.coefficients * self.reservoirs.volumes,
            )
        )
        return 1.0 / jnp.maximum(
            mass + width * exchange_diagonal, jnp.finfo(mass.dtype).tiny
        )


@dataclass(frozen=True, slots=True)
class MixedDimensionalTransportPlan:
    bulk: BulkDGTransportPlan
    network: NetworkTransportPlan
    exchange_transfer: PreparedEmbeddedMeasureTransfer
    exchange_coefficients: np.ndarray
    reservoir_network_indices: np.ndarray
    reservoir_volumes: np.ndarray
    reservoir_coefficients: np.ndarray
    require_nonnegative: bool = True
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.bulk, BulkDGTransportPlan):
            raise TypeError("bulk must be BulkDGTransportPlan.")
        if not isinstance(self.network, NetworkTransportPlan):
            raise TypeError("network must be NetworkTransportPlan.")
        if not isinstance(self.exchange_transfer, PreparedEmbeddedMeasureTransfer):
            raise TypeError("exchange_transfer must be PreparedEmbeddedMeasureTransfer.")
        if self.exchange_transfer.source_measures.shape != (
            sum(block.cell_count for block in self.bulk.mesh.blocks),
        ):
            raise ValueError("Exchange source must be the bulk DG0 cell field.")
        if (
            self.exchange_transfer.target_measures.shape
            != self.network.network.node_measures.shape
        ):
            raise ValueError("Exchange targets must be the network node field.")
        if not isinstance(self.require_nonnegative, bool):
            raise TypeError("require_nonnegative must be boolean.")
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "mixed-dimensional-transport-plan",
                    "bulk": self.bulk.plan_id,
                    "network": self.network.plan_id,
                    "exchange": self.exchange_transfer.transfer_id,
                    "exchange_coefficients": array_tree_fingerprint(
                        np.asarray(self.exchange_coefficients)
                    ),
                    "reservoir_indices": array_tree_fingerprint(
                        np.asarray(self.reservoir_network_indices)
                    ),
                    "reservoir_volumes": array_tree_fingerprint(
                        np.asarray(self.reservoir_volumes)
                    ),
                    "reservoir_coefficients": array_tree_fingerprint(
                        np.asarray(self.reservoir_coefficients)
                    ),
                    "require_nonnegative": self.require_nonnegative,
                }
            ),
        )

    def prepare(self) -> PreparedMixedDimensionalTransport:
        bulk = self.bulk.prepare()
        network = self.network.prepare()
        exchange = PermeabilityExchangePlan(
            self.exchange_transfer, self.exchange_coefficients
        )
        reservoirs = ReservoirCouplingPlan(
            self.reservoir_network_indices,
            self.reservoir_volumes,
            self.reservoir_coefficients,
            network_size=len(network.mass),
        )
        runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-mixed-dimensional-transport",
                "plan": self.plan_id,
                "bulk": bulk.prepared_id,
                "network": network.prepared_id,
                "exchange": exchange.exchange_id,
                "reservoirs": reservoirs.coupling_id,
                "require_nonnegative": self.require_nonnegative,
            }
        )
        return PreparedMixedDimensionalTransport(
            bulk,
            network,
            exchange,
            reservoirs,
            self.require_nonnegative,
            runtime_id,
        )


__all__ = [
    "BulkDGTransportEvidence",
    "BulkDGTransportPlan",
    "MixedDimensionalMassLedger",
    "MixedDimensionalSources",
    "MixedDimensionalStepResult",
    "MixedDimensionalTransportPlan",
    "MixedDimensionalTransportState",
    "NetworkTransportPlan",
    "PermeabilityExchangePlan",
    "PreparedBulkDGTransport",
    "PreparedMixedDimensionalTransport",
    "PreparedNetworkTransport",
    "ReservoirCouplingPlan",
]
