#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._barotropic_beta_plane import BarotropicBetaPlane
from ._cumulants import (
    cumulants_from_ensemble,
    DenseCumulantState,
    ForcingCovariance,
    SecondCumulantLayout,
)
from ._interactions import InteractionPartition
from ._plan import PreparedStatisticalDynamics, QuadraticDynamics, StatisticalDynamicsPlan


class BetaPlaneStatisticalCoordinates(StrictModule, NonTrainableState):
    """Admissible independent real coordinates for beta-plane cumulants."""

    problem: BarotropicBetaPlane
    partition: InteractionPartition
    active_coordinate_indices: Array
    coordinate_modal_indices: Array
    layout: SecondCumulantLayout
    coordinate_size: int = eqx.field(static=True)
    coordinate_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: BarotropicBetaPlane,
        partition: InteractionPartition,
        /,
    ):
        if not isinstance(problem, BarotropicBetaPlane):
            raise TypeError("problem must be a BarotropicBetaPlane.")
        if not isinstance(partition, InteractionPartition):
            raise TypeError("partition must be an InteractionPartition.")
        if partition.state_shape != problem.state_shape:
            raise ValueError("Interaction partition and beta-plane modal shapes differ.")
        coordinates = problem.coordinates
        fixed = np.asarray(coordinates.fixed_indices, dtype=np.int64)
        representatives = np.asarray(coordinates.representative_indices, dtype=np.int64)
        coordinate_modes = np.concatenate((fixed, representatives, representatives))
        admissible = np.asarray(partition.admissibility_mask).reshape((-1,))
        low = np.asarray(partition.low_mask).reshape((-1,))
        active = np.flatnonzero(admissible[coordinate_modes])
        active_modes = coordinate_modes[active]
        active_low = low[active_modes]
        if active.size < 2 or not np.any(active_low) or np.all(active_low):
            raise ValueError(
                "Beta-plane statistical coordinates require non-empty low and high subspaces."
            )
        coordinate_id = canonical_fingerprint(
            {
                "kind": "beta-plane-statistical-coordinates",
                "problem": problem.problem_id,
                "partition": partition.partition_id,
                "active_coordinate_indices": active.tolist(),
                "coordinate_modal_indices": active_modes.tolist(),
            }
        )
        layout = SecondCumulantLayout(
            int(active.size),
            np.flatnonzero(active_low),
            eddy_indices=np.flatnonzero(~active_low),
            layout_id=canonical_fingerprint(
                {
                    "kind": "beta-plane-second-cumulant-layout",
                    "coordinates": coordinate_id,
                }
            ),
        )
        self.problem = problem
        self.partition = partition
        self.active_coordinate_indices = jnp.asarray(active, dtype=jnp.int32)
        self.coordinate_modal_indices = jnp.asarray(active_modes, dtype=jnp.int32)
        self.layout = layout
        self.coordinate_size = int(active.size)
        self.coordinate_id = coordinate_id

    def validate_coordinates(self, values: ArrayLike, /) -> Array:
        coordinates = jnp.asarray(values)
        if coordinates.shape != (self.coordinate_size,):
            raise ValueError(
                f"Statistical coordinates must have shape {(self.coordinate_size,)}; "
                f"got {coordinates.shape}."
            )
        if not jnp.issubdtype(coordinates.dtype, jnp.floating):
            raise TypeError("Beta-plane statistical coordinates must be real-valued.")
        return coordinates

    def from_coordinates(self, values: ArrayLike, /) -> Array:
        active = self.validate_coordinates(values)
        full = (
            jnp.zeros(
                (self.problem.coordinates.coordinate_size,),
                dtype=active.dtype,
            )
            .at[self.active_coordinate_indices]
            .set(active)
        )
        return self.problem.coordinates.from_real_coordinates(full)

    def to_coordinates(self, vorticity: ArrayLike, /) -> Array:
        state = self.problem.project_state(vorticity)
        full = self.problem.coordinates.to_real_coordinates(state)
        return full[self.active_coordinate_indices]

    def ensemble_cumulants(
        self,
        vorticity_members: Sequence[ArrayLike] | ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        tolerance: float = 1.0e-10,
    ) -> DenseCumulantState:
        values = jnp.asarray(vorticity_members)
        if (
            values.ndim != len(self.problem.state_shape) + 1
            or tuple(values.shape[1:]) != self.problem.state_shape
        ):
            raise ValueError("Vorticity ensemble has an incompatible modal shape.")
        coordinates = jax.vmap(self.to_coordinates)(values)
        return cumulants_from_ensemble(
            self.layout,
            coordinates,
            weights=weights,
            mean_subspace_tolerance=tolerance,
            eddy_mean_tolerance=tolerance,
        )


class BetaPlaneCumulantSystem(StrictModule, NonTrainableState):
    """Quadratic beta-plane owner materialized for one declared closure."""

    problem: BarotropicBetaPlane
    partition: InteractionPartition
    coordinates: BetaPlaneStatisticalCoordinates
    linear: Array
    tensor_bytes: int = eqx.field(static=True)
    maximum_tensor_bytes: int = eqx.field(static=True)
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: BarotropicBetaPlane,
        partition: InteractionPartition,
        /,
        *,
        maximum_tensor_bytes: int = 512 * 1024 * 1024,
        maximum_coordinate_dimension: int = 512,
    ):
        coordinates = BetaPlaneStatisticalCoordinates(problem, partition)
        maximum_bytes = int(maximum_tensor_bytes)
        maximum_dimension = int(maximum_coordinate_dimension)
        if maximum_bytes <= 0 or maximum_dimension <= 0:
            raise ValueError("Beta-plane cumulant resource limits must be positive.")
        dimension = coordinates.coordinate_size
        if dimension > maximum_dimension:
            raise MemoryError(
                "Beta-plane cumulant coordinates exceed maximum_coordinate_dimension."
            )
        dtype = np.dtype(
            jnp.empty(
                (), dtype=problem.discretization.plan.precision.coefficient_dtype
            ).real.dtype
        )
        tensor_bytes = (dimension**3 + dimension**2 + dimension) * dtype.itemsize
        if tensor_bytes > maximum_bytes:
            raise MemoryError("Beta-plane cumulant tensor exceeds maximum_tensor_bytes.")
        zero = jnp.zeros((dimension,), dtype=dtype)

        def linear_action(values: Array) -> Array:
            vorticity = coordinates.from_coordinates(values)
            return coordinates.to_coordinates(problem.linear_tendency(vorticity))

        linear = jax.jacfwd(linear_action)(zero)
        self.problem = problem
        self.partition = partition
        self.coordinates = coordinates
        self.linear = linear
        self.tensor_bytes = tensor_bytes
        self.maximum_tensor_bytes = maximum_bytes
        self.system_id = canonical_fingerprint(
            {
                "kind": "beta-plane-cumulant-system",
                "problem": problem.problem_id,
                "partition": partition.partition_id,
                "coordinates": coordinates.coordinate_id,
                "linear": array_tree_fingerprint(linear),
                "tensor_bytes": tensor_bytes,
            }
        )

    @property
    def layout(self) -> SecondCumulantLayout:
        return self.coordinates.layout

    def prepare(
        self,
        forcing: ForcingCovariance,
        /,
        *,
        closure: str,
        time_step: float,
        hermitian_tolerance: float = 1.0e-10,
        psd_tolerance: float = 1.0e-10,
        maximum_state_bytes: int = 512 * 1024 * 1024,
        maximum_workspace_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> PreparedStatisticalDynamics:
        if closure == "ce2":
            interaction_model = "ql"
        elif closure == "gce2":
            interaction_model = "gql"
        else:
            raise ValueError("closure must be 'ce2' or 'gce2'.")
        dtype = self.linear.dtype
        zero = jnp.zeros((self.coordinates.coordinate_size,), dtype=dtype)

        def selected_quadratic_action(values: Array) -> Array:
            vorticity = self.coordinates.from_coordinates(values)
            selected = self.partition.select(
                self.problem.bilinear_tendency,
                vorticity,
                model=interaction_model,
            )
            return self.coordinates.to_coordinates(selected)

        quadratic = 0.5 * jax.jacfwd(jax.jacfwd(selected_quadratic_action))(zero)
        dynamics = QuadraticDynamics(
            jnp.zeros_like(zero),
            self.linear,
            quadratic,
            dynamics_id=canonical_fingerprint(
                {
                    "kind": "beta-plane-selected-quadratic-coordinates",
                    "system": self.system_id,
                    "interaction_model": interaction_model,
                }
            ),
        )
        return StatisticalDynamicsPlan(
            self.layout,
            dynamics,
            forcing,
            closure=closure,
            interaction_model=interaction_model,
            time_step=time_step,
            hermitian_tolerance=hermitian_tolerance,
            psd_tolerance=psd_tolerance,
            maximum_state_bytes=maximum_state_bytes,
            maximum_workspace_bytes=maximum_workspace_bytes,
        ).prepare()


__all__ = ["BetaPlaneCumulantSystem", "BetaPlaneStatisticalCoordinates"]
