#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import TensorGridPlan, UniformCellAxisSpec
from ...discretization.finite_volume._hydrostatic_grid import PreparedHydrostaticGrid
from ...discretization.multiblock import (
    BlockInterface,
    InterfaceOrientation,
    MultiblockGridPlan,
    PreparedMultiblockGrid,
)


SphericalMosaicKind: TypeAlias = Literal["polar-cap", "tripolar", "cubed-sphere"]


class SphericalHydrostaticBlock(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    longitude: Array
    latitude: Array
    cartesian_unit: Array
    cell_area: Array
    first_edge_length: Array
    second_edge_length: Array
    covariant_metric: Array
    contravariant_metric: Array
    horizontal_jacobian: Array
    xi_lower_cartesian: Array
    xi_upper_cartesian: Array
    eta_lower_cartesian: Array
    eta_upper_cartesian: Array
    xi_lower_frame: Array
    xi_upper_frame: Array
    eta_lower_frame: Array
    eta_upper_frame: Array
    coriolis: Array
    rest_depth: Array
    vertical_faces: Array
    geometry: PreparedHydrostaticGrid
    block_id: str = eqx.field(static=True)

    def interface_coordinates(self, axis: str, side: str, /) -> Array:
        if axis == "xi":
            return self.xi_lower_cartesian if side == "lower" else self.xi_upper_cartesian
        if axis == "eta":
            return (
                self.eta_lower_cartesian if side == "lower" else self.eta_upper_cartesian
            )
        raise ValueError("Spherical block interface axis is invalid.")

    def interface_frame(self, axis: str, side: str, /) -> Array:
        if axis == "xi":
            return self.xi_lower_frame if side == "lower" else self.xi_upper_frame
        if axis == "eta":
            return self.eta_lower_frame if side == "lower" else self.eta_upper_frame
        raise ValueError("Spherical block interface axis is invalid.")


class SphericalMosaicSeam(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    left_block: str = eqx.field(static=True)
    left_axis: str = eqx.field(static=True)
    left_side: str = eqx.field(static=True)
    right_block: str = eqx.field(static=True)
    right_axis: str = eqx.field(static=True)
    right_side: str = eqx.field(static=True)
    orientation: InterfaceOrientation
    interface: BlockInterface
    vector_rotation: Array
    seam_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        left_block: str,
        left_axis: str,
        left_side: str,
        right_block: str,
        right_axis: str,
        right_side: str,
        /,
        flip: bool = False,
    ):
        axes = (str(left_axis), str(right_axis))
        sides = (str(left_side), str(right_side))
        if any(axis not in ("xi", "eta") for axis in axes) or any(
            side not in ("lower", "upper") for side in sides
        ):
            raise ValueError("Spherical seam axes and sides are invalid.")
        orientation = InterfaceOrientation(1, flips=(flip,))
        interface = BlockInterface(
            name,
            left_block,
            axes[0],
            sides[0],
            right_block,
            axes[1],
            sides[1],
            orientation,
        )
        rotation = jnp.eye(2)
        self.name = str(name)
        self.left_block = str(left_block)
        self.left_axis = axes[0]
        self.left_side = sides[0]
        self.right_block = str(right_block)
        self.right_axis = axes[1]
        self.right_side = sides[1]
        self.orientation = orientation
        self.interface = interface
        self.vector_rotation = rotation
        self.seam_id = canonical_fingerprint(
            {
                "kind": "spherical-hydrostatic-seam",
                "interface": interface.interface_id,
                "orientation": orientation.orientation_id,
                "left_route": [axes[0], sides[0]],
                "right_route": [axes[1], sides[1]],
            }
        )

    def rotate_vector_trace(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        oriented = self.orientation.apply(value, trailing_axes=value.ndim - 1)
        if oriented.shape[-1] != 2:
            raise ValueError(
                "Spherical seam vector traces require two tangent components."
            )
        rotation = self.vector_rotation
        for _ in range(oriented.ndim - rotation.ndim + 1):
            rotation = rotation[..., None, :, :]
        return contract("...ij,...j->...i", rotation, oriented)

    def inverse_rotate_vector_trace(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape[-1] != 2:
            raise ValueError(
                "Spherical seam vector traces require two tangent components."
            )
        rotation = self.vector_rotation
        determinant = (
            rotation[..., 0, 0] * rotation[..., 1, 1]
            - rotation[..., 0, 1] * rotation[..., 1, 0]
        )
        inverse = (
            jnp.stack(
                (
                    jnp.stack(
                        (rotation[..., 1, 1], -rotation[..., 0, 1]),
                        axis=-1,
                    ),
                    jnp.stack(
                        (-rotation[..., 1, 0], rotation[..., 0, 0]),
                        axis=-1,
                    ),
                ),
                axis=-2,
            )
            / determinant[..., None, None]
        )
        for _ in range(value.ndim - inverse.ndim + 1):
            inverse = inverse[..., None, :, :]
        oriented = contract("...ij,...j->...i", inverse, value)
        return self.orientation.inverse(oriented, trailing_axes=value.ndim - 1)


class PreparedHydrostaticMosaicGrid(StrictModule, NonTrainableState):
    topology: PreparedMultiblockGrid
    blocks: tuple[SphericalHydrostaticBlock, ...]
    seams: tuple[SphericalMosaicSeam, ...]
    northern_poles: Array
    kind: SphericalMosaicKind = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    rotation_rate: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def block(self, name: str, /) -> SphericalHydrostaticBlock:
        names = tuple(block.name for block in self.blocks)
        if name not in names:
            raise KeyError(f"Unknown hydrostatic mosaic block {name!r}.")
        return self.blocks[names.index(name)]

    def prepare_ocean(self, block_name: str, /, **plan_kwargs):
        from ._hydrostatic import HydrostaticPrimitiveEquationPlan

        return HydrostaticPrimitiveEquationPlan(
            self.block(block_name).geometry, **plan_kwargs
        ).prepare()

    def prepare_oceans(self, /, **plan_kwargs) -> "PreparedHydrostaticMosaicOcean":
        """Prepare the public, seam-coupled multiblock ocean workflow."""
        return PreparedHydrostaticMosaicOcean(self, plan_kwargs)

    def seam_traces(
        self,
        seam_index: int,
        block_values: dict[str, ArrayLike],
        /,
    ) -> tuple[Array, Array]:
        index = int(seam_index)
        if index < 0 or index >= len(self.seams):
            raise IndexError("Spherical mosaic seam index is out of bounds.")
        seam = self.seams[index]
        left = self.topology.trace(
            seam.left_block,
            seam.left_axis,
            seam.left_side,
            jnp.asarray(block_values[seam.left_block]),
        )
        right = self.topology.trace(
            seam.right_block,
            seam.right_axis,
            seam.right_side,
            jnp.asarray(block_values[seam.right_block]),
        )
        return left, seam.orientation.apply(right)

    def seam_transport_traces(
        self,
        seam_index: int,
        block_states,
        /,
    ) -> tuple[Array, Array]:
        index = int(seam_index)
        if index < 0 or index >= len(self.seams):
            raise IndexError("Spherical mosaic seam index is out of bounds.")
        seam = self.seams[index]

        def trace(block_name: str, axis: str, side: str, /) -> Array:
            block = self.block(block_name)
            state = block_states[block_name]
            axis_index = 0 if axis == "xi" else 1
            transport = jnp.asarray(state.transports[axis_index])
            expected = (
                block.geometry.x_face_shape
                if axis_index == 0
                else block.geometry.y_face_shape
            )
            if transport.shape != expected:
                raise ValueError("Hydrostatic mosaic transport shape is invalid.")
            face_index = 0 if side == "lower" else -1
            return jnp.take(transport, face_index, axis=axis_index)

        left = trace(seam.left_block, seam.left_axis, seam.left_side)
        right = trace(seam.right_block, seam.right_axis, seam.right_side)
        return left, seam.orientation.apply(right, trailing_axes=1)

    def scatter_seam_flux(
        self, seam_index: int, integrated_flux: ArrayLike, /
    ) -> tuple[Array, Array]:
        index = int(seam_index)
        if index < 0 or index >= len(self.seams):
            raise IndexError("Spherical mosaic seam index is out of bounds.")
        flux = jnp.asarray(integrated_flux)
        if flux.ndim < 1:
            raise ValueError("Spherical seam flux must retain its trace axis.")
        trailing = flux.ndim - 1
        return flux, -self.seams[index].orientation.inverse(flux, trailing_axes=trailing)


class HydrostaticMosaicState(StrictModule):
    """Continuation state for every block of one coupled spherical mosaic."""

    blocks: dict[str, Any]


class HydrostaticMosaicAdvance(StrictModule):
    """One conservative multiblock advance and its exchanged seam fluxes."""

    state: HydrostaticMosaicState
    seam_fluxes: tuple[Array, ...]
    successful: Array
    residual: Array


class PreparedHydrostaticMosaicOcean(StrictModule):
    """Hydrostatic solvers coupled through the physical mosaic interfaces."""

    grid: PreparedHydrostaticMosaicGrid
    oceans: tuple[Any, ...]
    methods: tuple[Any, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedHydrostaticMosaicGrid,
        plan_kwargs: Mapping[str, Any],
        /,
    ):
        from ._hydrostatic_step import HydrostaticIMEXMidpointMethod

        if not isinstance(grid, PreparedHydrostaticMosaicGrid):
            raise TypeError("grid must be a PreparedHydrostaticMosaicGrid.")
        kwargs = dict(plan_kwargs)
        oceans = tuple(grid.prepare_ocean(block.name, **kwargs) for block in grid.blocks)
        if grid.seams and any(
            ocean.plan.external_mode == "split-explicit" for ocean in oceans
        ):
            raise ValueError(
                "Split-explicit external subcycles are not available for coupled "
                "mosaics; use the globally coupled implicit external mode."
            )
        if grid.seams and any(ocean.plan.boundaries for ocean in oceans):
            raise ValueError(
                "Coupled mosaic faces are physical interfaces and cannot also "
                "carry single-block open-boundary conditions."
            )
        methods = tuple(HydrostaticIMEXMidpointMethod(ocean) for ocean in oceans)
        self.grid = grid
        self.oceans = oceans
        self.methods = methods
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-hydrostatic-mosaic-ocean",
                "grid": grid.prepared_id,
                "oceans": [ocean.prepared_id for ocean in oceans],
                "methods": [method.method_id for method in methods],
            }
        )

    def ocean(self, block_name: str, /):
        names = self.grid.topology.plan.block_names
        if block_name not in names:
            raise KeyError(f"Unknown hydrostatic mosaic block {block_name!r}.")
        return self.oceans[names.index(block_name)]

    @staticmethod
    def _block_argument(value, name: str, default):
        if value is None:
            return default
        if isinstance(value, Mapping):
            if name not in value:
                return default
            return value[name]
        return value

    def initialize_state(
        self,
        eta: Mapping[str, ArrayLike] | ArrayLike = 0.0,
        /,
        *,
        transports: Mapping[str, tuple[ArrayLike, ArrayLike]] | None = None,
        tracers: Mapping[str, Mapping[str, ArrayLike]] | None = None,
    ) -> HydrostaticMosaicState:
        from ._hydrostatic_step import HydrostaticContinuationState

        states = {}
        for block, ocean in zip(self.grid.blocks, self.oceans, strict=True):
            eta_value = jnp.asarray(
                self._block_argument(eta, block.name, 0.0),
                dtype=block.cell_area.dtype,
            )
            if eta_value.shape == ():
                eta_value = jnp.full(block.geometry.horizontal_shape, eta_value)
            state = ocean.initialize_state(
                eta_value,
                transports=self._block_argument(transports, block.name, None),
                tracers=self._block_argument(tracers, block.name, None),
            )
            states[block.name] = HydrostaticContinuationState.initialize(ocean, state)
        tracer_names = {
            tuple(sorted(value.state.tracer_inventory)) for value in states.values()
        }
        if len(tracer_names) != 1:
            raise ValueError(
                "Every coupled mosaic block must carry the same tracer fields."
            )
        coupled, _ = self._couple(HydrostaticMosaicState(states))
        return coupled

    @staticmethod
    def _replace_transport_trace(
        continuation,
        ocean,
        axis: int,
        side: str,
        trace: Array,
        /,
        *,
        step_size: Array | None = None,
    ):
        from ._hydrostatic import HydrostaticOceanState
        from ._hydrostatic_step import HydrostaticContinuationState

        geometry = ocean.geometry

        location = [slice(None)] * 3
        location[axis] = 0 if side == "lower" else -1
        location_ = tuple(location)
        transports = list(continuation.state.transports)
        previous = transports[axis][location_]
        transports[axis] = transports[axis].at[location_].set(trace)
        eta = continuation.state.eta
        ledger = continuation.ledger
        filtered_eta = continuation.filtered_eta
        if step_size is not None:
            cell_location = [slice(None)] * 2
            cell_location[axis] = 0 if side == "lower" else -1
            cell_location_ = tuple(cell_location)
            outward_sign = -1.0 if side == "lower" else 1.0
            transport_change = jnp.sum(trace - previous, axis=-1)
            eta_correction = (
                -jnp.asarray(step_size, dtype=eta.dtype)
                * outward_sign
                * transport_change
                / geometry.cell_area[cell_location_]
            )
            eta = eta.at[cell_location_].add(eta_correction)
            filtered_eta = filtered_eta.at[cell_location_].add(0.5 * eta_correction)
            volume_correction = jnp.sum(
                geometry.cell_area[cell_location_] * eta_correction
            )
            surface_energy_correction = (
                0.5
                * ocean.plan.reference_density
                * ocean.plan.gravity
                * jnp.sum(
                    geometry.cell_area[cell_location_]
                    * (
                        eta[cell_location_] ** 2
                        - continuation.state.eta[cell_location_] ** 2
                    )
                )
            )
            ledger = eqx.tree_at(
                lambda value: (
                    value.volume_change,
                    value.free_surface_energy_change,
                    value.reconciliation_correction,
                ),
                ledger,
                (
                    ledger.volume_change + volume_correction,
                    ledger.free_surface_energy_change + surface_energy_correction,
                    ledger.reconciliation_correction
                    + jnp.sum(jnp.abs(step_size * transport_change)),
                ),
            )
        ocean_state = HydrostaticOceanState(
            eta,
            (transports[0], transports[1]),
            continuation.state.tracer_inventory,
            continuation.state.tke_inventory,
        )
        return HydrostaticContinuationState(
            ocean_state,
            ledger,
            filtered_eta,
            geometry.depth_integrate((transports[0], transports[1])),
            continuation.subcycle_phase,
            continuation.subcycle_schedule,
        )

    @staticmethod
    def _replace_tracer_inventory_traces(
        continuation,
        axis: int,
        side: str,
        corrections: Mapping[str, Array],
        /,
    ):
        from ._hydrostatic import HydrostaticOceanState
        from ._hydrostatic_step import HydrostaticContinuationState

        cell_location = [slice(None)] * 3
        cell_location[axis] = 0 if side == "lower" else -1
        cell_location_ = tuple(cell_location)
        inventory = dict(continuation.state.tracer_inventory)
        for name, correction in corrections.items():
            inventory[name] = inventory[name].at[cell_location_].add(correction)
        tracer_change = dict(continuation.ledger.tracer_change)
        tracer_source = dict(continuation.ledger.tracer_source)
        for name, correction in corrections.items():
            total = jnp.sum(correction)
            tracer_change[name] = tracer_change[name] + total
            tracer_source[name] = tracer_source[name] + total
        ledger = eqx.tree_at(
            lambda value: (value.tracer_change, value.tracer_source),
            continuation.ledger,
            (tracer_change, tracer_source),
        )
        ocean_state = HydrostaticOceanState(
            continuation.state.eta,
            continuation.state.transports,
            inventory,
            continuation.state.tke_inventory,
        )
        return HydrostaticContinuationState(
            ocean_state,
            ledger,
            continuation.filtered_eta,
            continuation.filtered_barotropic_transport,
            continuation.subcycle_phase,
            continuation.subcycle_schedule,
        )

    @staticmethod
    def _replace_tke_inventory_trace(
        continuation, axis: int, side: str, correction: Array, /
    ):
        from ._hydrostatic import HydrostaticOceanState
        from ._hydrostatic_step import HydrostaticContinuationState

        location = [slice(None)] * 3
        location[axis] = 0 if side == "lower" else -1
        tke = continuation.state.tke_inventory.at[tuple(location)].add(correction)
        state = HydrostaticOceanState(
            continuation.state.eta,
            continuation.state.transports,
            continuation.state.tracer_inventory,
            tke,
        )
        return HydrostaticContinuationState(
            state,
            continuation.ledger,
            continuation.filtered_eta,
            continuation.filtered_barotropic_transport,
            continuation.subcycle_phase,
            continuation.subcycle_schedule,
        )

    @staticmethod
    def _with_ocean_state(continuation, ocean_state, /):
        from ._hydrostatic_step import HydrostaticContinuationState

        return HydrostaticContinuationState(
            ocean_state,
            continuation.ledger,
            continuation.filtered_eta,
            continuation.filtered_barotropic_transport,
            continuation.subcycle_phase,
            continuation.subcycle_schedule,
        )

    def _tracer_trace(
        self,
        continuation,
        block_name: str,
        axis: str,
        side: str,
        tracer_name: str,
        /,
    ) -> Array:
        block = self.grid.block(block_name)
        state = continuation.state
        epoch = block.geometry.metric_epoch(state.eta)
        inventory = state.tracer_inventory[tracer_name]
        concentration = jnp.where(
            epoch.cell_volume > 0.0,
            inventory / jnp.where(epoch.cell_volume > 0.0, epoch.cell_volume, 1.0),
            0.0,
        )
        axis_index = 0 if axis == "xi" else 1
        index = 0 if side == "lower" else -1
        return jnp.take(concentration, index, axis=axis_index)

    def _tke_trace(
        self,
        continuation,
        block_name: str,
        axis: str,
        side: str,
        /,
    ) -> Array:
        block = self.grid.block(block_name)
        state = continuation.state
        epoch = block.geometry.metric_epoch(state.eta)
        concentration = jnp.where(
            epoch.cell_volume > 0.0,
            state.tke_inventory
            / jnp.where(epoch.cell_volume > 0.0, epoch.cell_volume, 1.0),
            0.0,
        )
        return jnp.take(
            concentration,
            0 if side == "lower" else -1,
            axis=0 if axis == "xi" else 1,
        )

    def _correct_tracer_exchange(
        self,
        candidate: HydrostaticMosaicState,
        evaluation: HydrostaticMosaicState,
        step_size: Array,
        /,
        *,
        boundary_traces: Mapping[str, Any] | None = None,
    ) -> HydrostaticMosaicState:
        continuations = dict(candidate.blocks)
        tracer_names = tuple(
            sorted(next(iter(evaluation.blocks.values())).state.tracer_inventory)
        )
        boundaries = (
            self._boundary_traces(evaluation)
            if boundary_traces is None
            else boundary_traces
        )
        tracer_fluxes = {}
        tke_fluxes = {}
        for block, ocean in zip(self.grid.blocks, self.oceans, strict=True):
            state = evaluation.blocks[block.name].state
            epoch = block.geometry.metric_epoch(state.eta)
            view = ocean.view(state)
            tracer_fluxes[block.name] = {}
            for name in tracer_names:
                flux = ocean._horizontal_tracer_fluxes(
                    name,
                    view.tracers[name],
                    state.transports,
                    boundary_values=boundaries[block.name].tracers[name],
                )
                if ocean.plan.mixing.kind == "redi-gm":
                    redi, _ = ocean._redi_gm_fluxes(
                        view.tracers[name],
                        state,
                        epoch,
                        concentration_boundary=boundaries[block.name].tracers[name],
                        density_boundary=boundaries[block.name].density,
                    )
                    flux = (flux[0] + redi[0], flux[1] + redi[1])
                tracer_fluxes[block.name][name] = flux
            tke_concentration = jnp.where(
                epoch.cell_volume > 0.0,
                state.tke_inventory
                / jnp.where(epoch.cell_volume > 0.0, epoch.cell_volume, 1.0),
                0.0,
            )
            tke_fluxes[block.name] = ocean._horizontal_tracer_fluxes(
                "__tke__",
                tke_concentration,
                state.transports,
                boundary_values=boundaries[block.name].tke,
            )

        def face_trace(fluxes: tuple[Array, Array], axis: str, side: str, /) -> Array:
            axis_index = 0 if axis == "xi" else 1
            return jnp.take(
                fluxes[axis_index],
                0 if side == "lower" else -1,
                axis=axis_index,
            )

        for seam in self.grid.seams:
            left_sign = -1.0 if seam.left_side == "lower" else 1.0
            right_sign = -1.0 if seam.right_side == "lower" else 1.0
            left_corrections = {}
            right_corrections = {}
            for name in tracer_names:
                left = face_trace(
                    tracer_fluxes[seam.left_block][name],
                    seam.left_axis,
                    seam.left_side,
                )
                right = seam.orientation.apply(
                    face_trace(
                        tracer_fluxes[seam.right_block][name],
                        seam.right_axis,
                        seam.right_side,
                    ),
                    trailing_axes=1,
                )
                common = 0.5 * (left_sign * left - right_sign * right)
                left_change = left_sign * common - left
                right_change = seam.orientation.inverse(
                    -right_sign * common - right, trailing_axes=1
                )
                left_corrections[name] = -step_size * left_sign * left_change
                right_corrections[name] = -step_size * right_sign * right_change
            continuations[seam.left_block] = self._replace_tracer_inventory_traces(
                continuations[seam.left_block],
                0 if seam.left_axis == "xi" else 1,
                seam.left_side,
                left_corrections,
            )
            continuations[seam.right_block] = self._replace_tracer_inventory_traces(
                continuations[seam.right_block],
                0 if seam.right_axis == "xi" else 1,
                seam.right_side,
                right_corrections,
            )

            left_tke = face_trace(
                tke_fluxes[seam.left_block],
                seam.left_axis,
                seam.left_side,
            )
            right_tke = seam.orientation.apply(
                face_trace(
                    tke_fluxes[seam.right_block],
                    seam.right_axis,
                    seam.right_side,
                ),
                trailing_axes=1,
            )
            common_tke = 0.5 * (left_sign * left_tke - right_sign * right_tke)
            left_tke_correction = -step_size * left_sign * (left_sign * common_tke)
            right_tke_correction = seam.orientation.inverse(
                -step_size * right_sign * (-right_sign * common_tke),
                trailing_axes=1,
            )
            continuations[seam.left_block] = self._replace_tke_inventory_trace(
                continuations[seam.left_block],
                0 if seam.left_axis == "xi" else 1,
                seam.left_side,
                left_tke_correction,
            )
            continuations[seam.right_block] = self._replace_tke_inventory_trace(
                continuations[seam.right_block],
                0 if seam.right_axis == "xi" else 1,
                seam.right_side,
                right_tke_correction,
            )
        return HydrostaticMosaicState(continuations)

    def _enforce_global_tracer_balance(
        self,
        candidate: HydrostaticMosaicState,
        base: HydrostaticMosaicState,
        time: Array,
        step_size: Array,
        args: Any,
        /,
    ) -> HydrostaticMosaicState:
        from ._hydrostatic import HydrostaticOceanState
        from ._hydrostatic_step import HydrostaticContinuationState

        continuations = dict(candidate.blocks)
        names = tuple(sorted(next(iter(base.blocks.values())).state.tracer_inventory))
        epochs = {
            block.name: block.geometry.metric_epoch(
                candidate.blocks[block.name].state.eta
            )
            for block in self.grid.blocks
        }
        total_volume = sum(jnp.sum(epoch.cell_volume) for epoch in epochs.values())
        for name in names:
            initial = sum(
                jnp.sum(value.state.tracer_inventory[name])
                for value in base.blocks.values()
            )
            source = jnp.asarray(0.0, dtype=step_size.dtype)
            for block, ocean in zip(self.grid.blocks, self.oceans, strict=True):
                freshwater = ocean.plan.freshwater.evaluate(
                    time,
                    block.geometry.horizontal_shape,
                    args,
                )
                incoming = (
                    ocean.plan.freshwater.absolute_salinity
                    if name == "absolute_salinity"
                    else (
                        ocean.plan.freshwater.conservative_temperature
                        if name == "conservative_temperature"
                        else 0.0
                    )
                )
                source = source + (
                    step_size * jnp.sum(block.geometry.cell_area * freshwater) * incoming
                )
            current = sum(
                jnp.sum(value.state.tracer_inventory[name])
                for value in continuations.values()
            )
            defect = initial + source - current
            reconciliation_scale = jnp.maximum(
                jnp.maximum(jnp.abs(initial + source), jnp.abs(current)),
                jnp.finfo(step_size.dtype).tiny,
            )
            reconciliation_limit = (
                1.0e-8 * reconciliation_scale
                + 65536.0 * jnp.finfo(step_size.dtype).eps * reconciliation_scale
            )
            bounded_defect = jnp.where(
                jnp.abs(defect) <= reconciliation_limit, defect, 0.0
            )
            for block in self.grid.blocks:
                continuation = continuations[block.name]
                correction = (
                    bounded_defect * epochs[block.name].cell_volume / total_volume
                )
                inventory = dict(continuation.state.tracer_inventory)
                inventory[name] = inventory[name] + correction
                tracer_change = dict(continuation.ledger.tracer_change)
                total = jnp.sum(correction)
                tracer_change[name] = tracer_change[name] + total
                ledger = eqx.tree_at(
                    lambda value: (
                        value.tracer_change,
                        value.reconciliation_correction,
                    ),
                    continuation.ledger,
                    (
                        tracer_change,
                        continuation.ledger.reconciliation_correction
                        + jnp.sum(jnp.abs(correction)),
                    ),
                )
                ocean_state = HydrostaticOceanState(
                    continuation.state.eta,
                    continuation.state.transports,
                    inventory,
                    continuation.state.tke_inventory,
                )
                continuations[block.name] = HydrostaticContinuationState(
                    ocean_state,
                    ledger,
                    continuation.filtered_eta,
                    continuation.filtered_barotropic_transport,
                    continuation.subcycle_phase,
                    continuation.subcycle_schedule,
                )
        return HydrostaticMosaicState(continuations)

    def _boundary_traces(self, state: HydrostaticMosaicState, /) -> dict[str, Any]:
        from ._hydrostatic import (
            _cell_from_faces,
            HydrostaticBoundaryTraces,
        )

        views = {
            name: self.ocean(name).view(continuation.state)
            for name, continuation in state.blocks.items()
        }
        tracer_names = tuple(sorted(next(iter(views.values())).tracers))
        empty = lambda: [[None, None], [None, None]]
        surface = {block.name: empty() for block in self.grid.blocks}
        pressure = {block.name: empty() for block in self.grid.blocks}
        density = {block.name: empty() for block in self.grid.blocks}
        tke = {block.name: empty() for block in self.grid.blocks}
        tracers = {
            block.name: {name: empty() for name in tracer_names}
            for block in self.grid.blocks
        }
        velocity = {block.name: [empty(), empty()] for block in self.grid.blocks}
        cell_velocity = {}
        tke_concentration = {}
        for block in self.grid.blocks:
            view = views[block.name]
            cell_velocity[block.name] = jnp.stack(
                (
                    _cell_from_faces(view.velocity[0], 0, block.geometry.periodic[0]),
                    _cell_from_faces(view.velocity[1], 1, block.geometry.periodic[1]),
                ),
                axis=-1,
            )
            epoch = block.geometry.metric_epoch(state.blocks[block.name].state.eta)
            inventory = state.blocks[block.name].state.tke_inventory
            tke_concentration[block.name] = jnp.where(
                epoch.cell_volume > 0.0,
                inventory / jnp.where(epoch.cell_volume > 0.0, epoch.cell_volume, 1.0),
                0.0,
            )

        def trace(value: Array, axis: str, side: str, /) -> Array:
            return jnp.take(
                value,
                0 if side == "lower" else -1,
                axis=0 if axis == "xi" else 1,
            )

        def assign(
            storage,
            seam: SphericalMosaicSeam,
            left_value: Array,
            right_value: Array,
            /,
        ) -> None:
            left_axis = 0 if seam.left_axis == "xi" else 1
            right_axis = 0 if seam.right_axis == "xi" else 1
            left_side = 0 if seam.left_side == "lower" else 1
            right_side = 0 if seam.right_side == "lower" else 1
            left_trace = trace(left_value, seam.left_axis, seam.left_side)
            right_trace = trace(right_value, seam.right_axis, seam.right_side)
            trailing = right_trace.ndim - 1
            storage[seam.left_block][left_axis][left_side] = seam.orientation.apply(
                right_trace, trailing_axes=trailing
            )
            storage[seam.right_block][right_axis][right_side] = seam.orientation.inverse(
                left_trace, trailing_axes=left_trace.ndim - 1
            )

        for seam in self.grid.seams:
            assign(
                surface,
                seam,
                state.blocks[seam.left_block].state.eta,
                state.blocks[seam.right_block].state.eta,
            )
            assign(
                pressure,
                seam,
                views[seam.left_block].hydrostatic_pressure,
                views[seam.right_block].hydrostatic_pressure,
            )
            assign(
                density,
                seam,
                views[seam.left_block].density,
                views[seam.right_block].density,
            )
            assign(
                tke,
                seam,
                tke_concentration[seam.left_block],
                tke_concentration[seam.right_block],
            )
            for name in tracer_names:
                left_store = {
                    seam.left_block: tracers[seam.left_block][name],
                    seam.right_block: tracers[seam.right_block][name],
                }
                assign(
                    left_store,
                    seam,
                    views[seam.left_block].tracers[name],
                    views[seam.right_block].tracers[name],
                )
            left_vector = trace(
                cell_velocity[seam.left_block],
                seam.left_axis,
                seam.left_side,
            )
            right_vector = trace(
                cell_velocity[seam.right_block],
                seam.right_axis,
                seam.right_side,
            )
            left_ghost = seam.rotate_vector_trace(right_vector)
            right_ghost = seam.inverse_rotate_vector_trace(left_vector)
            left_axis = 0 if seam.left_axis == "xi" else 1
            right_axis = 0 if seam.right_axis == "xi" else 1
            left_side = 0 if seam.left_side == "lower" else 1
            right_side = 0 if seam.right_side == "lower" else 1
            for component in range(2):
                velocity[seam.left_block][component][left_axis][left_side] = left_ghost[
                    ..., component
                ]
                velocity[seam.right_block][component][right_axis][right_side] = (
                    right_ghost[..., component]
                )

        return {
            block.name: HydrostaticBoundaryTraces(
                tuple(tuple(pair) for pair in surface[block.name]),
                tuple(tuple(pair) for pair in pressure[block.name]),
                tuple(tuple(pair) for pair in density[block.name]),
                tuple(
                    tuple(tuple(pair) for pair in component)
                    for component in velocity[block.name]
                ),
                {
                    name: tuple(tuple(pair) for pair in tracers[block.name][name])
                    for name in tracer_names
                },
                tuple(tuple(pair) for pair in tke[block.name]),
            )
            for block in self.grid.blocks
        }

    def _couple(
        self,
        state: HydrostaticMosaicState,
        /,
        *,
        step_size: Array | None = None,
    ) -> tuple[HydrostaticMosaicState, tuple[Array, ...]]:
        if not isinstance(state, HydrostaticMosaicState):
            raise TypeError("state must be a HydrostaticMosaicState.")
        continuations = dict(state.blocks)
        seam_fluxes = []
        for index, seam in enumerate(self.grid.seams):
            raw_states = {
                name: continuation.state for name, continuation in continuations.items()
            }
            left, right = self.grid.seam_transport_traces(index, raw_states)
            left_sign = -1.0 if seam.left_side == "lower" else 1.0
            right_sign = -1.0 if seam.right_side == "lower" else 1.0
            outward = 0.5 * (left_sign * left - right_sign * right)
            left_trace = left_sign * outward
            right_oriented = -right_sign * outward
            right_trace = seam.orientation.inverse(right_oriented, trailing_axes=1)
            continuations[seam.left_block] = self._replace_transport_trace(
                continuations[seam.left_block],
                self.ocean(seam.left_block),
                0 if seam.left_axis == "xi" else 1,
                seam.left_side,
                left_trace,
                step_size=step_size,
            )
            continuations[seam.right_block] = self._replace_transport_trace(
                continuations[seam.right_block],
                self.ocean(seam.right_block),
                0 if seam.right_axis == "xi" else 1,
                seam.right_side,
                right_trace,
                step_size=step_size,
            )
            seam_fluxes.append(outward)
        return HydrostaticMosaicState(continuations), tuple(seam_fluxes)

    def _validate_coupled_input(self, state: HydrostaticMosaicState, /) -> None:
        raw = {name: continuation.state for name, continuation in state.blocks.items()}
        for index, seam in enumerate(self.grid.seams):
            left, right = self.grid.seam_transport_traces(index, raw)
            left_sign = -1.0 if seam.left_side == "lower" else 1.0
            right_sign = -1.0 if seam.right_side == "lower" else 1.0
            mismatch = left_sign * left + right_sign * right
            if (
                not bool(jnp.all(jnp.isfinite(mismatch)))
                or float(jnp.max(jnp.abs(mismatch))) > 1.0e-10
            ):
                raise ValueError(
                    "Hydrostatic mosaic input transports must already be seam coupled."
                )

    def advance(
        self,
        state: HydrostaticMosaicState,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        step_index: ArrayLike = 0,
        args: Any = None,
    ) -> HydrostaticMosaicAdvance:
        from ._hydrostatic import HydrostaticOceanState
        from ._hydrostatic_step import HydrostaticContinuationState

        if not self.grid.seams:
            name = self.grid.blocks[0].name
            result = self.methods[0].step(
                jnp.asarray(step_index),
                jnp.asarray(time),
                state.blocks[name],
                jnp.asarray(step_size),
                args,
            )
            return HydrostaticMosaicAdvance(
                HydrostaticMosaicState({name: result.accepted_state}),
                (),
                result.successful,
                result.residual,
            )

        del step_index
        self._validate_coupled_input(state)
        coupled, initial_fluxes = self._couple(state)
        dt = jnp.asarray(step_size)
        half_dt = 0.5 * dt
        first_traces = self._boundary_traces(coupled)
        midpoint_blocks = {}
        first_evidence = {}
        for block, method in zip(self.grid.blocks, self.methods, strict=True):
            midpoint_state, evidence, _ = method._advance(
                coupled.blocks[block.name].state,
                coupled.blocks[block.name].state,
                jnp.asarray(time),
                half_dt,
                args,
                boundary_traces=first_traces[block.name],
            )
            midpoint_blocks[block.name] = self._with_ocean_state(
                coupled.blocks[block.name], midpoint_state
            )
            first_evidence[block.name] = evidence
        midpoint_flux_evaluation = HydrostaticMosaicState(
            {
                name: self._with_ocean_state(
                    continuation,
                    HydrostaticOceanState(
                        continuation.state.eta,
                        continuation.state.transports,
                        coupled.blocks[name].state.tracer_inventory,
                        coupled.blocks[name].state.tke_inventory,
                    ),
                )
                for name, continuation in midpoint_blocks.items()
            }
        )
        midpoint, _ = self._couple(
            HydrostaticMosaicState(midpoint_blocks), step_size=half_dt
        )
        midpoint = self._correct_tracer_exchange(
            midpoint,
            midpoint_flux_evaluation,
            half_dt,
            boundary_traces=first_traces,
        )
        midpoint = self._enforce_global_tracer_balance(
            midpoint,
            coupled,
            jnp.asarray(time),
            half_dt,
            args,
        )
        midpoint_traces = self._boundary_traces(midpoint)

        candidates = {}
        successful = []
        residuals = []
        for block, method in zip(self.grid.blocks, self.methods, strict=True):
            candidate_state, second_evidence, second_ledger = method._advance(
                coupled.blocks[block.name].state,
                midpoint.blocks[block.name].state,
                jnp.asarray(time) + half_dt,
                dt,
                args,
                boundary_traces=midpoint_traces[block.name],
            )
            continuation = coupled.blocks[block.name]
            ledger = method._add_ledgers(continuation.ledger, second_ledger)
            candidates[block.name] = HydrostaticContinuationState(
                candidate_state,
                ledger,
                0.5 * (continuation.filtered_eta + candidate_state.eta),
                method.ocean.geometry.depth_integrate(candidate_state.transports),
                continuation.subcycle_phase
                + jnp.where(
                    method.ocean.plan.external_mode == "split-explicit",
                    second_evidence.subcycle_schedule.count,
                    jnp.asarray(1, dtype=jnp.int32),
                ),
                second_evidence.subcycle_schedule,
            )
            successful.append(
                first_evidence[block.name].successful & second_evidence.successful
            )
            residuals.append(
                jnp.maximum(
                    first_evidence[block.name].free_surface_residual,
                    second_evidence.free_surface_residual,
                )
            )
        candidate_flux_evaluation = HydrostaticMosaicState(
            {
                name: self._with_ocean_state(
                    continuation,
                    HydrostaticOceanState(
                        continuation.state.eta,
                        continuation.state.transports,
                        midpoint.blocks[name].state.tracer_inventory,
                        midpoint.blocks[name].state.tke_inventory,
                    ),
                )
                for name, continuation in candidates.items()
            }
        )
        candidate, candidate_fluxes = self._couple(
            HydrostaticMosaicState(candidates), step_size=dt
        )
        candidate = self._correct_tracer_exchange(
            candidate,
            candidate_flux_evaluation,
            dt,
            boundary_traces=midpoint_traces,
        )
        candidate = self._enforce_global_tracer_balance(
            candidate,
            coupled,
            jnp.asarray(time) + half_dt,
            dt,
            args,
        )
        for block, ocean in zip(self.grid.blocks, self.oceans, strict=True):
            corrected_state = candidate.blocks[block.name].state
            epoch = block.geometry.metric_epoch(corrected_state.eta)
            view = ocean.view(corrected_state)
            successful.append(
                epoch.valid
                & view.eos_valid
                & view.eos_finite
                & jnp.all(corrected_state.tke_inventory >= 0.0)
                & view.eos_successful
                & jnp.all(jnp.isfinite(corrected_state.tke_inventory))
                & jnp.all(
                    jnp.stack(
                        tuple(
                            jnp.all(jnp.isfinite(value))
                            for value in corrected_state.tracer_inventory.values()
                        )
                    )
                )
            )
        block_successful = jnp.all(jnp.stack(successful))
        candidate_states = {
            name: continuation.state for name, continuation in candidate.blocks.items()
        }
        seam_residual = jnp.asarray(0.0, dtype=dt.dtype)
        seam_scale = jnp.asarray(jnp.finfo(dt.dtype).tiny, dtype=dt.dtype)
        for index, seam in enumerate(self.grid.seams):
            left, right = self.grid.seam_transport_traces(index, candidate_states)
            left_sign = -1.0 if seam.left_side == "lower" else 1.0
            right_sign = -1.0 if seam.right_side == "lower" else 1.0
            seam_residual = jnp.maximum(
                seam_residual,
                jnp.max(jnp.abs(left_sign * left + right_sign * right)),
            )
            seam_scale = jnp.maximum(
                seam_scale,
                jnp.maximum(
                    jnp.max(jnp.abs(left)),
                    jnp.max(jnp.abs(right)),
                ),
            )
        initial_volume = sum(
            jnp.sum(
                block.geometry.metric_epoch(
                    coupled.blocks[block.name].state.eta
                ).cell_volume
            )
            for block in self.grid.blocks
        )
        final_volume = sum(
            jnp.sum(
                block.geometry.metric_epoch(
                    candidate.blocks[block.name].state.eta
                ).cell_volume
            )
            for block in self.grid.blocks
        )
        freshwater_volume = sum(
            dt
            * jnp.sum(
                block.geometry.cell_area
                * ocean.plan.freshwater.evaluate(
                    jnp.asarray(time) + half_dt,
                    block.geometry.horizontal_shape,
                    args,
                )
            )
            for block, ocean in zip(self.grid.blocks, self.oceans, strict=True)
        )
        volume_residual = jnp.abs(final_volume - initial_volume - freshwater_volume)
        volume_scale = jnp.maximum(
            jnp.maximum(jnp.abs(initial_volume), jnp.abs(final_volume)),
            jnp.maximum(
                jnp.abs(freshwater_volume),
                jnp.finfo(dt.dtype).tiny,
            ),
        )
        relative_volume_residual = volume_residual / volume_scale
        relative_tracer_residual = jnp.asarray(0.0, dtype=dt.dtype)
        for name in next(iter(state.blocks.values())).state.tracer_inventory:
            initial_inventory = sum(
                jnp.sum(value.state.tracer_inventory[name])
                for value in state.blocks.values()
            )
            tracer_scale = jnp.maximum(
                jnp.abs(initial_inventory),
                jnp.finfo(dt.dtype).tiny,
            )
            final_inventory = sum(
                jnp.sum(value.state.tracer_inventory[name])
                for value in candidate.blocks.values()
            )
            tracer_source = jnp.asarray(0.0, dtype=dt.dtype)
            for block, ocean in zip(self.grid.blocks, self.oceans, strict=True):
                incoming = (
                    ocean.plan.freshwater.absolute_salinity
                    if name == "absolute_salinity"
                    else (
                        ocean.plan.freshwater.conservative_temperature
                        if name == "conservative_temperature"
                        else 0.0
                    )
                )
                tracer_source = tracer_source + (
                    dt
                    * jnp.sum(
                        block.geometry.cell_area
                        * ocean.plan.freshwater.evaluate(
                            jnp.asarray(time) + half_dt,
                            block.geometry.horizontal_shape,
                            args,
                        )
                    )
                    * incoming
                )
            tracer_scale = jnp.maximum(
                tracer_scale,
                jnp.maximum(
                    jnp.abs(final_inventory),
                    jnp.abs(tracer_source),
                ),
            )
            relative_tracer_residual = jnp.maximum(
                relative_tracer_residual,
                jnp.abs(final_inventory - initial_inventory - tracer_source)
                / tracer_scale,
            )
        relative_seam_residual = seam_residual / seam_scale
        balance_residual = jnp.maximum(
            relative_seam_residual,
            jnp.maximum(
                relative_volume_residual,
                relative_tracer_residual,
            ),
        )
        tolerance = 65536.0 * jnp.finfo(dt.dtype).eps
        all_successful = (
            block_successful
            & jnp.isfinite(relative_seam_residual)
            & jnp.isfinite(relative_volume_residual)
            & jnp.isfinite(relative_tracer_residual)
            & (relative_seam_residual <= tolerance)
            & (relative_volume_residual <= tolerance)
            & (relative_tracer_residual <= tolerance)
        )
        accepted = jax.tree.map(
            lambda proposed, current: jnp.where(all_successful, proposed, current),
            candidate,
            state,
        )
        fluxes = tuple(
            jnp.where(all_successful, proposed, current)
            for proposed, current in zip(candidate_fluxes, initial_fluxes, strict=True)
        )
        return HydrostaticMosaicAdvance(
            accepted,
            fluxes,
            all_successful,
            jnp.maximum(
                jnp.max(jnp.stack(residuals)),
                balance_residual,
            ),
        )


class SphericalHydrostaticMosaicPlan(StrictModule, NonTrainableState):
    kind: SphericalMosaicKind = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    resolution: tuple[int, int] = eqx.field(static=True)
    vertical_faces: Array
    rest_depth: float = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    rotation_rate: float = eqx.field(static=True)
    cap_latitude: float = eqx.field(static=True)
    hemisphere: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SphericalMosaicKind,
        resolution: tuple[int, int],
        vertical_faces: ArrayLike,
        rest_depth: float,
        /,
        *,
        radius: float = 6_371_000.0,
        rotation_rate: float = 7.292115e-5,
        cap_latitude: float = np.deg2rad(60.0),
        hemisphere: Literal["north", "south"] = "north",
    ):
        if kind not in ("polar-cap", "tripolar", "cubed-sphere"):
            raise ValueError("Unknown spherical hydrostatic mosaic kind.")
        shape = tuple(int(value) for value in resolution)
        z = jnp.asarray(vertical_faces, dtype=float)
        depth = float(rest_depth)
        radius_ = float(radius)
        rotation_ = float(rotation_rate)
        cap = float(cap_latitude)
        if (
            len(shape) != 2
            or any(value < 2 for value in shape)
            or (kind == "cubed-sphere" and shape[0] != shape[1])
            or z.ndim != 1
            or z.size < 3
            or bool(jnp.any(~jnp.isfinite(z)))
            or bool(jnp.any(jnp.diff(z) <= 0.0))
            or not np.isfinite(depth)
            or depth <= 0.0
            or not np.isfinite(radius_)
            or radius_ <= 0.0
            or not np.isfinite(rotation_)
            or not np.isfinite(cap)
            or not 0.0 < cap < 0.5 * np.pi
            or hemisphere not in ("north", "south")
        ):
            raise ValueError("Spherical hydrostatic mosaic parameters are invalid.")
        names = {
            "polar-cap": (f"{hemisphere}-polar-cap",),
            "tripolar": (
                "southwest-belt",
                "southeast-belt",
                "northwest-cap",
                "northeast-cap",
            ),
            "cubed-sphere": ("+x", "-x", "+y", "-y", "+z", "-z"),
        }[kind]
        self.kind = kind
        self.block_names = names
        self.resolution = shape
        self.vertical_faces = z
        self.rest_depth = depth
        self.radius = radius_
        self.rotation_rate = rotation_
        self.cap_latitude = cap
        self.hemisphere = hemisphere
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spherical-hydrostatic-mosaic",
                "topology": kind,
                "blocks": list(names),
                "resolution": list(shape),
                "vertical_faces": np.asarray(z).tolist(),
                "rest_depth": depth,
                "radius": radius_,
                "rotation_rate": rotation_,
                "cap_latitude": cap,
                "hemisphere": hemisphere,
            }
        )

    @staticmethod
    def _cartesian(longitude: Array, latitude: Array, /) -> Array:
        cosine = jnp.cos(latitude)
        return jnp.stack(
            (
                cosine * jnp.cos(longitude),
                cosine * jnp.sin(longitude),
                jnp.sin(latitude),
            ),
            axis=-1,
        )

    @staticmethod
    def _normalize(vector: Array, /) -> Array:
        return vector / jnp.sqrt(jnp.sum(vector**2, axis=-1, keepdims=True))

    @classmethod
    def _normalized_blend(cls, left: Array, right: Array, fraction: Array, /) -> Array:
        return cls._normalize((1.0 - fraction) * left + fraction * right)

    @staticmethod
    def _spherical_triangle_area(first: Array, second: Array, third: Array, /) -> Array:
        numerator = jnp.abs(contract("...i,...i->...", first, jnp.cross(second, third)))
        denominator = (
            1.0
            + contract("...i,...i->...", first, second)
            + contract("...i,...i->...", second, third)
            + contract("...i,...i->...", third, first)
        )
        return 2.0 * jnp.arctan2(numerator, denominator)

    def _block(
        self,
        name: str,
        coordinate_map,
        /,
        *,
        periodic_first: bool = False,
        collapse_second_lower: bool = False,
        collapse_second_upper: bool = False,
    ) -> SphericalHydrostaticBlock:
        periodic_first = bool(periodic_first)
        collapse_second_lower = bool(collapse_second_lower)
        collapse_second_upper = bool(collapse_second_upper)
        nx, ny = self.resolution
        xi = (jnp.arange(nx, dtype=float) + 0.5) / nx
        eta = (jnp.arange(ny, dtype=float) + 0.5) / ny
        xi_grid, eta_grid = jnp.meshgrid(xi, eta, indexing="ij")
        points = jnp.stack((xi_grid, eta_grid), axis=-1)
        flat_points = points.reshape((-1, 2))
        cartesian = jax.vmap(coordinate_map)(flat_points).reshape((nx, ny, 3))
        differential = self.radius * jax.vmap(jax.jacfwd(coordinate_map))(
            flat_points
        ).reshape((nx, ny, 3, 2))
        covariant = contract("...ki,...kj->...ij", differential, differential)
        determinant = (
            covariant[..., 0, 0] * covariant[..., 1, 1] - covariant[..., 0, 1] ** 2
        )
        if bool(jnp.any(~jnp.isfinite(determinant))) or bool(jnp.any(determinant <= 0.0)):
            raise ValueError("Spherical mosaic coordinate map is singular.")
        jacobian = jnp.sqrt(determinant)
        contravariant = (
            jnp.stack(
                (
                    jnp.stack((covariant[..., 1, 1], -covariant[..., 0, 1]), axis=-1),
                    jnp.stack((-covariant[..., 1, 0], covariant[..., 0, 0]), axis=-1),
                ),
                axis=-2,
            )
            / determinant[..., None, None]
        )

        def mapped_metric(first: Array, second: Array, /) -> tuple[Array, Array]:
            face_points = jnp.stack(
                jnp.broadcast_arrays(first[:, None], second[None, :]), axis=-1
            )
            face_differential = self.radius * jax.vmap(jax.jacfwd(coordinate_map))(
                face_points.reshape((-1, 2))
            ).reshape(face_points.shape[:-1] + (3, 2))
            face_covariant = contract(
                "...ki,...kj->...ij", face_differential, face_differential
            )
            return face_covariant[..., 0, 0], face_covariant[..., 1, 1]

        xi_faces = (
            jnp.arange(nx, dtype=float) / nx
            if periodic_first
            else jnp.arange(nx + 1, dtype=float) / nx
        )
        eta_faces = jnp.arange(ny + 1, dtype=float) / ny
        x_g11, x_g22 = mapped_metric(xi_faces, eta)
        y_g11, y_g22 = mapped_metric(xi, eta_faces)
        x_center_distance = jnp.sqrt(x_g11) / nx
        y_center_distance = jnp.sqrt(y_g22) / ny
        x_edge_length = jnp.sqrt(x_g22) / ny
        y_edge_length = jnp.sqrt(y_g11) / nx
        if collapse_second_lower:
            y_edge_length = y_edge_length.at[:, 0].set(0.0)
            y_center_distance = y_center_distance.at[:, 0].set(y_center_distance[:, 1])
        if collapse_second_upper:
            y_edge_length = y_edge_length.at[:, -1].set(0.0)
            y_center_distance = y_center_distance.at[:, -1].set(y_center_distance[:, -2])
        first_edge_length = jnp.sqrt(covariant[..., 0, 0]) / nx
        second_edge_length = jnp.sqrt(covariant[..., 1, 1]) / ny

        vertex_xi = jnp.arange(nx + 1, dtype=float) / nx
        vertex_eta = jnp.arange(ny + 1, dtype=float) / ny
        vertex_points = jnp.stack(
            jnp.broadcast_arrays(vertex_xi[:, None], vertex_eta[None, :]), axis=-1
        )
        vertices = jax.vmap(coordinate_map)(vertex_points.reshape((-1, 2))).reshape(
            (nx + 1, ny + 1, 3)
        )
        lower_left = vertices[:-1, :-1]
        lower_right = vertices[1:, :-1]
        upper_right = vertices[1:, 1:]
        upper_left = vertices[:-1, 1:]
        area = self.radius**2 * (
            self._spherical_triangle_area(lower_left, lower_right, upper_right)
            + self._spherical_triangle_area(lower_left, upper_right, upper_left)
        )

        def boundary(first: Array, second: Array, /) -> Array:
            values = jnp.stack(jnp.broadcast_arrays(first, second), axis=-1)
            return jax.vmap(coordinate_map)(values)

        xi_lower = boundary(jnp.zeros_like(eta), eta)
        xi_upper = boundary(jnp.ones_like(eta), eta)
        eta_lower = boundary(xi, jnp.zeros_like(xi))
        eta_upper = boundary(xi, jnp.ones_like(xi))

        def boundary_frame(first: Array, second: Array, /) -> Array:
            values = jnp.stack(jnp.broadcast_arrays(first, second), axis=-1)
            tangent = jax.vmap(jax.jacfwd(coordinate_map))(values)
            first_vector = tangent[..., :, 0]
            second_vector = tangent[..., :, 1]
            inner = jnp.sum(first_vector * second_vector, axis=-1, keepdims=True)
            first_normal = first_vector - second_vector * inner / jnp.sum(
                second_vector**2, axis=-1, keepdims=True
            )
            second_normal = second_vector - first_vector * inner / jnp.sum(
                first_vector**2, axis=-1, keepdims=True
            )
            first_normal = first_normal / jnp.linalg.norm(
                first_normal, axis=-1, keepdims=True
            )
            second_normal = second_normal / jnp.linalg.norm(
                second_normal, axis=-1, keepdims=True
            )
            return jnp.nan_to_num(jnp.stack((first_normal, second_normal), axis=-1))

        xi_lower_frame = boundary_frame(jnp.zeros_like(eta), eta)
        xi_upper_frame = boundary_frame(jnp.ones_like(eta), eta)
        eta_lower_frame = boundary_frame(xi, jnp.zeros_like(xi))
        eta_upper_frame = boundary_frame(xi, jnp.ones_like(xi))
        longitude = jnp.arctan2(cartesian[..., 1], cartesian[..., 0])
        latitude = jnp.arcsin(jnp.clip(cartesian[..., 2], -1.0, 1.0))
        coriolis = 2.0 * self.rotation_rate * cartesian[..., 2]
        depth = jnp.full(area.shape, self.rest_depth, dtype=area.dtype)
        block_id = canonical_fingerprint(
            {
                "kind": "spherical-hydrostatic-block",
                "plan": self.plan_id,
                "name": name,
                "shape": list(area.shape),
                "periodic_first": periodic_first,
            }
        )
        nz = int(self.vertical_faces.size - 1)
        geometry = PreparedHydrostaticGrid(
            horizontal_coordinate="latitude-longitude",
            vertical_coordinate="zstar",
            cell_shape=area.shape + (nz,),
            horizontal_shape=area.shape,
            periodic=(periodic_first, False),
            cell_area=area,
            x_edge_length=x_edge_length,
            y_edge_length=y_edge_length,
            x_center_distance=x_center_distance,
            y_center_distance=y_center_distance,
            covariant_metric=covariant,
            contravariant_metric=contravariant,
            horizontal_jacobian=jacobian,
            reference_vertical_faces=self.vertical_faces,
            reference_layer_fraction=jnp.diff(self.vertical_faces)
            / (self.vertical_faces[-1] - self.vertical_faces[0]),
            rest_depth=depth,
            wet_depth=1.0e-6,
            minimum_partial_fraction=0.2,
            longitude=longitude,
            latitude=latitude,
            coriolis=coriolis,
            radius=self.radius,
            geometry_id=canonical_fingerprint(
                {
                    "kind": "spherical-hydrostatic-block-geometry",
                    "block": block_id,
                }
            ),
        )
        return SphericalHydrostaticBlock(
            name=name,
            longitude=longitude,
            latitude=latitude,
            cartesian_unit=cartesian,
            cell_area=area,
            first_edge_length=first_edge_length,
            second_edge_length=second_edge_length,
            covariant_metric=covariant,
            contravariant_metric=contravariant,
            horizontal_jacobian=jacobian,
            xi_lower_cartesian=xi_lower,
            xi_upper_cartesian=xi_upper,
            eta_lower_cartesian=eta_lower,
            eta_upper_cartesian=eta_upper,
            xi_lower_frame=xi_lower_frame,
            xi_upper_frame=xi_upper_frame,
            eta_lower_frame=eta_lower_frame,
            eta_upper_frame=eta_upper_frame,
            coriolis=coriolis,
            rest_depth=depth,
            vertical_faces=self.vertical_faces,
            geometry=geometry,
            block_id=block_id,
        )

    @staticmethod
    def _with_total_area(
        block: SphericalHydrostaticBlock, total_area: ArrayLike, /
    ) -> SphericalHydrostaticBlock:
        target = jnp.asarray(total_area, dtype=block.cell_area.dtype)
        scaled = block.cell_area * target / jnp.sum(block.cell_area)
        return eqx.tree_at(
            lambda value: (value.cell_area, value.geometry.cell_area),
            block,
            (scaled, scaled),
        )

    def _polar_blocks(self) -> tuple[SphericalHydrostaticBlock, ...]:
        sine_cap = jnp.sin(self.cap_latitude)
        sign = 1.0 if self.hemisphere == "north" else -1.0

        def coordinate_map(point):
            longitude = -jnp.pi + 2.0 * jnp.pi * point[0]
            sine = sine_cap + (1.0 - sine_cap) * point[1]
            latitude = sign * jnp.arcsin(sine)
            return self._cartesian(longitude, latitude)

        block = self._block(
            self.block_names[0],
            coordinate_map,
            periodic_first=True,
            collapse_second_upper=self.hemisphere == "north",
            collapse_second_lower=self.hemisphere == "south",
        )
        area = 2.0 * jnp.pi * self.radius**2 * (1.0 - sine_cap)
        return (self._with_total_area(block, area),)

    def _tripolar_poles(self, /) -> Array:
        pole_latitude = self.cap_latitude + 0.65 * (0.5 * jnp.pi - self.cap_latitude)
        return jnp.stack(
            (
                self._cartesian(-0.75 * jnp.pi, pole_latitude),
                self._cartesian(0.25 * jnp.pi, pole_latitude),
            )
        )

    def _bipolar_cap_map(self, eastern: bool, /):
        """Return one side of a two-pole spherical transfinite cap.

        The lower boundary is the ordinary latitude belt.  The upper boundary
        is the bipolar fold joining two displaced northern poles; the two
        returned cap charts approach that fold from opposite sides.  The map
        stays in Cartesian unit-vector coordinates, so its regular crossing of
        the geographic North Pole never evaluates a longitude singularity.
        """
        cap = jnp.asarray(self.cap_latitude)
        date_pole, central_pole = self._tripolar_poles()
        date_join = self._cartesian(-jnp.pi, cap)
        central_join = self._cartesian(0.0, cap)

        def coordinate_map(point):
            xi, eta = point[0], point[1]
            if eastern:
                longitude = jnp.pi * xi
                lower = self._cartesian(longitude, cap)
                left_join, right_join = central_join, date_join
                left_pole, right_pole = central_pole, date_pole
            else:
                longitude = -jnp.pi + jnp.pi * xi
                lower = self._cartesian(longitude, cap)
                left_join, right_join = date_join, central_join
                left_pole, right_pole = date_pole, central_pole
            fold = self._normalized_blend(left_pole, right_pole, xi)
            left = self._normalized_blend(left_join, left_pole, eta)
            right = self._normalized_blend(right_join, right_pole, eta)
            corners = (
                (1.0 - xi) * (1.0 - eta) * left_join
                + xi * (1.0 - eta) * right_join
                + (1.0 - xi) * eta * left_pole
                + xi * eta * right_pole
            )
            transfinite = (
                (1.0 - eta) * lower
                + eta * fold
                + (1.0 - xi) * left
                + xi * right
                - corners
            )
            return self._normalize(transfinite)

        return coordinate_map

    def _tripolar_blocks(self) -> tuple[SphericalHydrostaticBlock, ...]:
        sine_cap = jnp.sin(self.cap_latitude)

        def belt_map(lower: float):
            def coordinate_map(point):
                longitude = lower + jnp.pi * point[0]
                sine = -1.0 + (1.0 + sine_cap) * point[1]
                return self._cartesian(longitude, jnp.arcsin(sine))

            return coordinate_map

        southwest = self._block(
            "southwest-belt",
            belt_map(-jnp.pi),
            collapse_second_lower=True,
        )
        southeast = self._block(
            "southeast-belt",
            belt_map(0.0),
            collapse_second_lower=True,
        )
        northwest = self._block("northwest-cap", self._bipolar_cap_map(False))
        northeast = self._block("northeast-cap", self._bipolar_cap_map(True))
        belt_area = jnp.pi * self.radius**2 * (1.0 + sine_cap)
        cap_area = jnp.pi * self.radius**2 * (1.0 - sine_cap)
        return (
            self._with_total_area(southwest, belt_area),
            self._with_total_area(southeast, belt_area),
            self._with_total_area(northwest, cap_area),
            self._with_total_area(northeast, cap_area),
        )

    def _cube_blocks(self) -> tuple[SphericalHydrostaticBlock, ...]:
        def cube_map(name: str):
            def coordinate_map(point):
                alpha = -0.25 * jnp.pi + 0.5 * jnp.pi * point[0]
                beta = -0.25 * jnp.pi + 0.5 * jnp.pi * point[1]
                tangent_alpha = jnp.tan(alpha)
                tangent_beta = jnp.tan(beta)
                raw = {
                    "+x": jnp.stack((jnp.asarray(1.0), tangent_alpha, tangent_beta)),
                    "-x": jnp.stack((jnp.asarray(-1.0), -tangent_alpha, tangent_beta)),
                    "+y": jnp.stack((-tangent_alpha, jnp.asarray(1.0), tangent_beta)),
                    "-y": jnp.stack((tangent_alpha, jnp.asarray(-1.0), tangent_beta)),
                    "+z": jnp.stack((-tangent_beta, tangent_alpha, jnp.asarray(1.0))),
                    "-z": jnp.stack((tangent_beta, tangent_alpha, jnp.asarray(-1.0))),
                }[name]
                return self._normalize(raw)

            return coordinate_map

        return tuple(self._block(name, cube_map(name)) for name in self.block_names)

    def _seams(self) -> tuple[SphericalMosaicSeam, ...]:
        if self.kind == "polar-cap":
            return ()
        if self.kind == "tripolar":
            adjacency = (
                (
                    "belt-date-line",
                    "southwest-belt",
                    "xi",
                    "lower",
                    "southeast-belt",
                    "xi",
                    "upper",
                    False,
                ),
                (
                    "belt-central",
                    "southwest-belt",
                    "xi",
                    "upper",
                    "southeast-belt",
                    "xi",
                    "lower",
                    False,
                ),
                (
                    "west-cap",
                    "southwest-belt",
                    "eta",
                    "upper",
                    "northwest-cap",
                    "eta",
                    "lower",
                    False,
                ),
                (
                    "east-cap",
                    "southeast-belt",
                    "eta",
                    "upper",
                    "northeast-cap",
                    "eta",
                    "lower",
                    False,
                ),
                (
                    "cap-date-line",
                    "northwest-cap",
                    "xi",
                    "lower",
                    "northeast-cap",
                    "xi",
                    "upper",
                    False,
                ),
                (
                    "cap-central",
                    "northwest-cap",
                    "xi",
                    "upper",
                    "northeast-cap",
                    "xi",
                    "lower",
                    False,
                ),
                (
                    "tripolar-fold",
                    "northwest-cap",
                    "eta",
                    "upper",
                    "northeast-cap",
                    "eta",
                    "upper",
                    True,
                ),
            )
            return tuple(
                SphericalMosaicSeam(
                    name,
                    left,
                    left_axis,
                    left_side,
                    right,
                    right_axis,
                    right_side,
                    flip=flip,
                )
                for (
                    name,
                    left,
                    left_axis,
                    left_side,
                    right,
                    right_axis,
                    right_side,
                    flip,
                ) in adjacency
            )
        adjacency = (
            ("+x", "xi", "upper", "+y", "xi", "lower", False),
            ("+x", "xi", "lower", "-y", "xi", "upper", False),
            ("+x", "eta", "upper", "+z", "eta", "lower", False),
            ("+x", "eta", "lower", "-z", "eta", "upper", False),
            ("-x", "xi", "lower", "+y", "xi", "upper", False),
            ("-x", "xi", "upper", "-y", "xi", "lower", False),
            ("-x", "eta", "upper", "+z", "eta", "upper", True),
            ("-x", "eta", "lower", "-z", "eta", "lower", True),
            ("+y", "eta", "upper", "+z", "xi", "upper", False),
            ("+y", "eta", "lower", "-z", "xi", "upper", True),
            ("-y", "eta", "upper", "+z", "xi", "lower", True),
            ("-y", "eta", "lower", "-z", "xi", "lower", False),
        )
        return tuple(
            SphericalMosaicSeam(
                f"cube-{left}-{right}",
                left,
                left_axis,
                left_side,
                right,
                right_axis,
                right_side,
                flip=flip,
            )
            for (
                left,
                left_axis,
                left_side,
                right,
                right_axis,
                right_side,
                flip,
            ) in adjacency
        )

    @staticmethod
    def _with_vector_rotation(
        seam: SphericalMosaicSeam,
        blocks: Mapping[str, SphericalHydrostaticBlock],
        /,
    ) -> SphericalMosaicSeam:
        left = blocks[seam.left_block].interface_frame(seam.left_axis, seam.left_side)
        right = seam.orientation.apply(
            blocks[seam.right_block].interface_frame(seam.right_axis, seam.right_side),
            trailing_axes=2,
        )
        overlap = contract("...ki,...kj->...ij", left, right)
        right_inner = contract("...ki,...kj->...ij", right, right)
        right_determinant = (
            right_inner[..., 0, 0] * right_inner[..., 1, 1] - right_inner[..., 0, 1] ** 2
        )
        right_dual = (
            jnp.stack(
                (
                    jnp.stack(
                        (right_inner[..., 1, 1], -right_inner[..., 0, 1]),
                        axis=-1,
                    ),
                    jnp.stack(
                        (-right_inner[..., 1, 0], right_inner[..., 0, 0]),
                        axis=-1,
                    ),
                ),
                axis=-2,
            )
            / right_determinant[..., None, None]
        )
        rotation = contract("...ik,...kj->...ij", overlap, right_dual)
        if bool(jnp.any(~jnp.isfinite(rotation))):
            raise ValueError("Spherical seam normal-frame map is singular.")
        seam_id = canonical_fingerprint(
            {
                "kind": "spherical-hydrostatic-seam",
                "interface": seam.interface.interface_id,
                "rotation": array_tree_fingerprint(rotation),
            }
        )
        rotated = eqx.tree_at(
            lambda value: value.vector_rotation,
            seam,
            rotation,
        )
        object.__setattr__(rotated, "seam_id", seam_id)
        return rotated

    def prepare(self, /) -> PreparedHydrostaticMosaicGrid:
        blocks = {
            "polar-cap": self._polar_blocks,
            "tripolar": self._tripolar_blocks,
            "cubed-sphere": self._cube_blocks,
        }[self.kind]()
        raw_seams = self._seams()
        block_by_name = {block.name: block for block in blocks}
        seams = tuple(
            self._with_vector_rotation(seam, block_by_name) for seam in raw_seams
        )
        logical_blocks = []
        for block in blocks:
            shape = block.cell_area.shape
            logical = TensorGridPlan(
                tuple(UniformCellAxisSpec(size) for size in shape),
                axis_names=("xi", "eta"),
            ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
            logical_blocks.append((block.name, logical))
        block_by_name = {block.name: block for block in blocks}
        interface_coordinates = tuple(
            (
                block_by_name[seam.left_block].interface_coordinates(
                    seam.left_axis, seam.left_side
                ),
                block_by_name[seam.right_block].interface_coordinates(
                    seam.right_axis, seam.right_side
                ),
            )
            for seam in seams
        )
        topology = MultiblockGridPlan(
            logical_blocks,
            tuple(seam.interface for seam in seams),
            geometry_tolerance=2.0e-6,
        ).prepare(interface_coordinates=interface_coordinates)
        return PreparedHydrostaticMosaicGrid(
            topology=topology,
            blocks=blocks,
            seams=seams,
            northern_poles=(
                self._tripolar_poles()
                if self.kind == "tripolar"
                else jnp.empty((0, 3), dtype=blocks[0].cell_area.dtype)
            ),
            kind=self.kind,
            radius=self.radius,
            rotation_rate=self.rotation_rate,
            prepared_id=canonical_fingerprint(
                {
                    "kind": "prepared-spherical-hydrostatic-mosaic",
                    "plan": self.plan_id,
                    "topology": topology.prepared_id,
                    "blocks": [block.block_id for block in blocks],
                    "seams": [seam.seam_id for seam in seams],
                }
            ),
        )


def polar_cap(
    resolution: tuple[int, int], vertical_faces: ArrayLike, rest_depth: float, /, **kwargs
) -> SphericalHydrostaticMosaicPlan:
    return SphericalHydrostaticMosaicPlan(
        "polar-cap", resolution, vertical_faces, rest_depth, **kwargs
    )


def tripolar(
    resolution: tuple[int, int], vertical_faces: ArrayLike, rest_depth: float, /, **kwargs
) -> SphericalHydrostaticMosaicPlan:
    return SphericalHydrostaticMosaicPlan(
        "tripolar", resolution, vertical_faces, rest_depth, **kwargs
    )


def equiangular_cubed_sphere(
    resolution: tuple[int, int], vertical_faces: ArrayLike, rest_depth: float, /, **kwargs
) -> SphericalHydrostaticMosaicPlan:
    return SphericalHydrostaticMosaicPlan(
        "cubed-sphere", resolution, vertical_faces, rest_depth, **kwargs
    )


__all__ = [
    "HydrostaticMosaicAdvance",
    "HydrostaticMosaicState",
    "PreparedHydrostaticMosaicOcean",
    "PreparedHydrostaticMosaicGrid",
    "SphericalHydrostaticBlock",
    "SphericalHydrostaticMosaicPlan",
    "SphericalMosaicKind",
    "SphericalMosaicSeam",
    "equiangular_cubed_sphere",
    "polar_cap",
    "tripolar",
]
