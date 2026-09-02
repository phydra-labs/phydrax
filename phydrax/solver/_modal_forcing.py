#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..stochastic import OrnsteinUhlenbeckRealization
from ._balance_law import (
    AbstractBalanceLawProcessPlan,
    AbstractPreparedBalanceLawProcess,
    BalanceLawProcessAdvance,
    BalanceLawProcessState,
)
from ._balance_law_transport import (
    AbstractPreparedBalanceLawTransport,
    BalanceLawSourceView,
)


class ModalForcingBasis(StrictModule, NonTrainableState):
    """Prepared physical vector modes evaluated at conservative cells."""

    vectors: Array
    weights: Array
    mode_ids: tuple[str, ...] = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        vectors: ArrayLike,
        /,
        *,
        weights: ArrayLike | None = None,
        mode_ids: tuple[str, ...] | None = None,
    ):
        values = np.asarray(vectors, dtype=float)
        if values.ndim != 3 or values.shape[0] == 0 or values.shape[-1] not in (1, 2, 3):
            raise ValueError(
                "Modal forcing vectors must have shape (modes, cells, dimension)."
            )
        mode_count = values.shape[0]
        weights_ = (
            np.ones((mode_count,), dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        identifiers = (
            tuple(f"mode-{index}" for index in range(mode_count))
            if mode_ids is None
            else tuple(str(identifier) for identifier in mode_ids)
        )
        if (
            weights_.shape != (mode_count,)
            or np.any(~np.isfinite(values))
            or np.any(~np.isfinite(weights_))
            or np.any(weights_ < 0.0)
            or len(identifiers) != mode_count
            or any(not identifier for identifier in identifiers)
            or len(set(identifiers)) != mode_count
        ):
            raise ValueError("Modal forcing basis metadata is invalid.")
        self.vectors = jnp.asarray(values)
        self.weights = jnp.asarray(weights_)
        self.mode_ids = identifiers
        self.basis_id = canonical_fingerprint(
            {
                "kind": "modal-forcing-basis",
                "vectors": array_tree_fingerprint(values),
                "weights": array_tree_fingerprint(weights_),
                "mode_ids": list(identifiers),
            }
        )

    @property
    def mode_count(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def cell_count(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[2])

    def evaluate(self, coefficients: ArrayLike, /) -> Array:
        value = jnp.asarray(coefficients)
        if value.shape != (self.mode_count,):
            raise ValueError("Modal forcing coefficients have the wrong shape.")
        return ein.contract("m,m,mcj->cj", value, self.weights, self.vectors)


class ModalOUForcingDiagnostics(StrictModule):
    acceleration: Array
    coefficients: Array
    momentum_change: Array
    energy_change: Array
    successful: Array


class ModalOUForcingPlan(AbstractBalanceLawProcessPlan):
    basis: ModalForcingBasis
    correlation_time: float = eqx.field(static=True)
    rms_acceleration: float = eqx.field(static=True)
    realization_name: str = eqx.field(static=True)

    def __init__(
        self,
        basis: ModalForcingBasis,
        /,
        *,
        correlation_time: float = 1.0,
        rms_acceleration: float = 1.0,
        realization_name: str = "modal_ou_forcing",
    ):
        if not isinstance(basis, ModalForcingBasis):
            raise TypeError("basis must be ModalForcingBasis.")
        correlation = float(correlation_time)
        rms = float(rms_acceleration)
        name = str(realization_name)
        if (
            not np.isfinite(correlation)
            or correlation <= 0.0
            or not np.isfinite(rms)
            or rms < 0.0
            or not name
        ):
            raise ValueError("Modal OU forcing parameters are invalid.")
        self.basis = basis
        self.correlation_time = correlation
        self.rms_acceleration = rms
        self.realization_name = name
        self.process_id = canonical_fingerprint(
            {
                "kind": "modal-ou-forcing",
                "basis": basis.basis_id,
                "correlation_time": correlation,
                "rms_acceleration": rms,
                "realization_name": name,
            }
        )

    def prepare(
        self, transport: AbstractPreparedBalanceLawTransport, /
    ) -> PreparedModalOUForcing:
        return PreparedModalOUForcing(self, transport)


class PreparedModalOUForcing(AbstractPreparedBalanceLawProcess):
    plan: ModalOUForcingPlan
    transport: AbstractPreparedBalanceLawTransport
    density_index: int = eqx.field(static=True)
    momentum_indices: tuple[int, ...] = eqx.field(static=True)
    energy_index: int = eqx.field(static=True)

    def __init__(
        self,
        plan: ModalOUForcingPlan,
        transport: AbstractPreparedBalanceLawTransport,
        /,
    ):
        if not isinstance(plan, ModalOUForcingPlan):
            raise TypeError("plan must be ModalOUForcingPlan.")
        if not isinstance(transport, AbstractPreparedBalanceLawTransport):
            raise TypeError("transport must be a prepared balance-law transport.")
        names = transport.component_names
        momentum = tuple(
            index for index, name in enumerate(names) if name.startswith("momentum_")
        )
        if len(momentum) != plan.basis.dimension:
            raise ValueError("Modal basis dimension does not match transport momentum.")
        cell_count = int(np.prod(transport.dynamics.discretization.grid.shape))
        if plan.basis.cell_count != cell_count:
            raise ValueError("Modal basis cell count does not match the transport grid.")
        self.plan = plan
        self.transport = transport
        self.density_index = names.index("density")
        self.momentum_indices = momentum
        self.energy_index = names.index("total_energy")
        self.process_id = canonical_fingerprint(
            {
                "kind": "prepared-modal-ou-forcing",
                "plan": plan.process_id,
                "transport": transport.transport_id,
            }
        )
        self.requires_realization = True
        self.realization_name = plan.realization_name
        self.differentiability = "smooth-discrete-stochastic"
        self.modified_components = tuple(names[index] for index in momentum) + (
            "total_energy",
        )

    def initialize(
        self, source_view: BalanceLawSourceView, args: Any = None, /
    ) -> BalanceLawProcessState:
        del source_view, args
        coefficients = jnp.zeros(
            (self.plan.basis.mode_count,),
            dtype=self.plan.basis.vectors.dtype,
        )
        return BalanceLawProcessState(
            self.process_id,
            ("coefficients",),
            (coefficients,),
        )

    def step_limit(
        self,
        time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        args: Any = None,
        /,
    ) -> Array:
        del time, cell_average, process_state, args
        return jnp.asarray(jnp.inf)

    def advance(
        self,
        start_time: Array,
        end_time: Array,
        cell_average: Array,
        process_state: BalanceLawProcessState,
        realization: Any = None,
        args: Any = None,
        /,
    ) -> BalanceLawProcessAdvance:
        del args
        if not isinstance(realization, OrnsteinUhlenbeckRealization):
            raise TypeError("Modal OU forcing requires OrnsteinUhlenbeckRealization.")
        if realization.sample_shape or realization.noise_shape != (
            self.plan.basis.mode_count,
        ):
            raise ValueError("OU realization shape does not match modal forcing basis.")
        coefficients = process_state.field("coefficients")
        next_coefficients = realization.transition(
            coefficients,
            start_time,
            end_time,
            jnp.asarray(self.plan.correlation_time),
        )
        acceleration = self.plan.basis.evaluate(next_coefficients)
        weighted_rms = jnp.sqrt(jnp.mean(jnp.sum(acceleration**2, axis=-1)))
        scale = jnp.where(
            weighted_rms > 0.0,
            self.plan.rms_acceleration / weighted_rms,
            0.0,
        )
        acceleration = scale * acceleration
        field = jnp.asarray(cell_average)
        density = field[..., self.density_index]
        momentum = field[..., self.momentum_indices]
        step = end_time - start_time
        momentum_change = step * density[..., None] * acceleration
        next_momentum = momentum + momentum_change
        energy_change = (
            jnp.sum(momentum * momentum_change, axis=-1)
            + 0.5 * jnp.sum(momentum_change**2, axis=-1)
        ) / density
        candidate = field.at[..., self.momentum_indices].set(next_momentum)
        candidate = candidate.at[..., self.energy_index].add(energy_change)
        successful = jnp.all(jnp.isfinite(candidate))
        accepted = jnp.where(successful, candidate, field)
        next_state = BalanceLawProcessState(
            self.process_id,
            ("coefficients",),
            (next_coefficients,),
        )
        diagnostics = ModalOUForcingDiagnostics(
            acceleration=acceleration,
            coefficients=next_coefficients,
            momentum_change=accepted[..., self.momentum_indices] - momentum,
            energy_change=accepted[..., self.energy_index]
            - field[..., self.energy_index],
            successful=successful,
        )
        return BalanceLawProcessAdvance(
            cell_average=accepted,
            process_state=next_state,
            successful=successful,
            source_change=accepted - field,
            diagnostics=diagnostics,
        )


__all__ = [
    "ModalForcingBasis",
    "ModalOUForcingDiagnostics",
    "ModalOUForcingPlan",
    "PreparedModalOUForcing",
]
