#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quasi-static free-boundary equilibrium driven by active and passive circuits."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.special import ellipe, ellipk

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...circuit import (
    CoupledInductanceStepResult,
    PreparedCoupledInductance,
)
from ._grad_shafranov import (
    DEFAULT_VACUUM_PERMEABILITY_H_M,
    FixedBoundaryEquilibriumResult,
    PreparedFixedBoundaryGradShafranov,
)


class TokamakWindingRole(StrEnum):
    ACTIVE = "active"
    PASSIVE = "passive"


@dataclass(frozen=True, slots=True)
class AxisymmetricFilamentCoil:
    winding_id: str
    r_m: float
    z_m: float
    turns: int = 1
    polarity: int = 1

    def __post_init__(self) -> None:
        winding = str(self.winding_id).strip()
        radius = float(self.r_m)
        height = float(self.z_m)
        if not winding or winding != self.winding_id:
            raise ValueError("winding_id must be non-empty canonical text.")
        if not np.isfinite(radius) or radius <= 0.0 or not np.isfinite(height):
            raise ValueError("Filament coil coordinates require finite R > 0 and Z.")
        if isinstance(self.turns, bool) or not isinstance(self.turns, Integral):
            raise TypeError("turns must be an integer.")
        turns = int(self.turns)
        polarity = int(self.polarity)
        if turns < 1 or polarity not in (-1, 1):
            raise ValueError("Filament coil turns must be positive and polarity ±1.")
        object.__setattr__(self, "winding_id", winding)
        object.__setattr__(self, "r_m", radius)
        object.__setattr__(self, "z_m", height)
        object.__setattr__(self, "turns", turns)
        object.__setattr__(self, "polarity", polarity)


class PreparedAxisymmetricCoilResponse(StrictModule, NonTrainableState):
    boundary_flux_per_amp_wb_per_rad: Array
    winding_roles: tuple[TokamakWindingRole, ...] = eqx.field(static=True)
    winding_ids: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    response_id: str = eqx.field(static=True)

    def boundary_flux(self, current_a: ArrayLike, /) -> Array:
        current = jnp.asarray(
            current_a, dtype=self.boundary_flux_per_amp_wb_per_rad.dtype
        )
        if current.shape != (len(self.winding_roles),):
            raise ValueError("Winding current must match the coil-response winding axis.")
        return contract("zrw,w->zr", self.boundary_flux_per_amp_wb_per_rad, current)


@dataclass(frozen=True, slots=True)
class AxisymmetricCoilResponsePlan:
    boundary_flux_per_amp_wb_per_rad: np.ndarray
    winding_roles: tuple[TokamakWindingRole, ...]
    source_id: str
    winding_ids: tuple[str, ...] = ()
    response_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = np.array(
            self.boundary_flux_per_amp_wb_per_rad, dtype=np.float64, copy=True
        )
        roles = tuple(self.winding_roles)
        winding_ids = tuple(str(value).strip() for value in self.winding_ids)
        source = str(self.source_id).strip()
        if (
            values.ndim != 3
            or values.shape[0] < 3
            or values.shape[1] < 3
            or values.shape[2] < 1
        ):
            raise ValueError("Coil response must have shape (z, r, winding).")
        if len(roles) != values.shape[2] or any(
            not isinstance(value, TokamakWindingRole) for value in roles
        ):
            raise TypeError("winding_roles must match the coil-response winding axis.")
        if (
            len(winding_ids) != values.shape[2]
            or len(set(winding_ids)) != len(winding_ids)
            or any(not value for value in winding_ids)
        ):
            raise ValueError("winding_ids must uniquely identify the response axis.")
        if np.any(~np.isfinite(values)):
            raise ValueError("Coil response must be finite.")
        interior = values[1:-1, 1:-1]
        if np.any(interior != 0.0):
            raise ValueError(
                "Initial coil response stores boundary flux only; interior entries must be zero."
            )
        if not source or source != self.source_id:
            raise ValueError("source_id must be non-empty canonical text.")
        values.setflags(write=False)
        object.__setattr__(self, "boundary_flux_per_amp_wb_per_rad", values)
        object.__setattr__(self, "winding_roles", roles)
        object.__setattr__(self, "winding_ids", winding_ids)
        object.__setattr__(self, "source_id", source)
        object.__setattr__(
            self,
            "response_id",
            canonical_fingerprint(
                {
                    "kind": "axisymmetric-coil-boundary-response",
                    "values": array_tree_fingerprint(values),
                    "roles": [value.value for value in roles],
                    "winding_ids": list(winding_ids),
                    "source": source,
                }
            ),
        )

    @classmethod
    def from_filament_coils(
        cls,
        r_m: ArrayLike,
        z_m: ArrayLike,
        coils: tuple[AxisymmetricFilamentCoil, ...],
        winding_roles: tuple[TokamakWindingRole, ...],
        /,
        *,
        source_id: str,
        minimum_distance_m: float,
        vacuum_permeability_h_m: float = DEFAULT_VACUUM_PERMEABILITY_H_M,
    ) -> AxisymmetricCoilResponsePlan:
        r = np.asarray(r_m, dtype=np.float64)
        z = np.asarray(z_m, dtype=np.float64)
        coils_ = tuple(coils)
        roles = tuple(winding_roles)
        distance = float(minimum_distance_m)
        permeability = float(vacuum_permeability_h_m)
        if (
            r.ndim != 1
            or z.ndim != 1
            or r.size < 3
            or z.size < 3
            or np.any(~np.isfinite(r))
            or np.any(~np.isfinite(z))
            or np.any(r <= 0.0)
        ):
            raise ValueError(
                "Filament response requires finite R > 0 and rank-one grids."
            )
        if (
            not coils_
            or any(not isinstance(value, AxisymmetricFilamentCoil) for value in coils_)
            or len(roles) != len(coils_)
        ):
            raise TypeError("Coils and winding roles must form one nonempty shared axis.")
        if (
            not np.isfinite(distance)
            or distance <= 0.0
            or not np.isfinite(permeability)
            or permeability <= 0.0
        ):
            raise ValueError(
                "Filament regularity distance and permeability must be positive."
            )
        rr, zz = np.meshgrid(r, z)
        boundary = np.zeros(rr.shape, dtype=bool)
        boundary[0] = True
        boundary[-1] = True
        boundary[:, 0] = True
        boundary[:, -1] = True
        response = np.zeros(rr.shape + (len(coils_),), dtype=np.float64)
        for index, coil in enumerate(coils_):
            separation = np.sqrt((rr - coil.r_m) ** 2 + (zz - coil.z_m) ** 2)
            if np.any(boundary & (separation <= distance)):
                raise ValueError(
                    "A filament coil lies within minimum_distance_m of the boundary."
                )
            k_squared = (
                4.0 * rr * coil.r_m / ((rr + coil.r_m) ** 2 + (zz - coil.z_m) ** 2)
            )
            if np.any(boundary & ((k_squared <= 0.0) | (k_squared >= 1.0))):
                raise ValueError(
                    "Filament Green-function modulus leaves its regular domain."
                )
            safe_k_squared = np.where(boundary, k_squared, 0.5)
            k = np.sqrt(safe_k_squared)
            green = (
                permeability
                / (2.0 * np.pi)
                * np.sqrt(rr * coil.r_m)
                / k
                * (
                    (2.0 - safe_k_squared) * ellipk(safe_k_squared)
                    - 2.0 * ellipe(safe_k_squared)
                )
            )
            response[..., index] = np.where(
                boundary,
                coil.polarity * coil.turns * green,
                0.0,
            )
        return cls(
            response,
            roles,
            source_id,
            tuple(coil.winding_id for coil in coils_),
        )

    def prepare(self) -> PreparedAxisymmetricCoilResponse:
        return PreparedAxisymmetricCoilResponse(
            jnp.asarray(self.boundary_flux_per_amp_wb_per_rad),
            self.winding_roles,
            self.winding_ids,
            self.source_id,
            self.response_id,
        )


class FreeBoundaryTokamakState(StrictModule):
    winding_current_a: Array
    time_s: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, winding_current_a: ArrayLike, time_s: ArrayLike, plan_id: str, /):
        current = jnp.asarray(winding_current_a)
        time = jnp.asarray(time_s, dtype=current.dtype)
        if current.ndim != 1 or time.shape != ():
            raise ValueError(
                "Free-boundary state requires rank-one current and scalar time."
            )
        identity = str(plan_id).strip()
        if not identity:
            raise ValueError("plan_id must be non-empty.")
        self.winding_current_a = current
        self.time_s = time
        self.plan_id = identity


class FreeBoundaryTokamakStepResult(StrictModule):
    candidate_state: FreeBoundaryTokamakState
    accepted_state: FreeBoundaryTokamakState
    circuit: CoupledInductanceStepResult
    equilibrium: FixedBoundaryEquilibriumResult
    boundary_flux_wb_per_rad: Array
    finite: Array
    domain_valid: Array
    sensitivity_valid: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class FreeBoundaryTokamakPlan:
    equilibrium: PreparedFixedBoundaryGradShafranov
    circuits: PreparedCoupledInductance
    coil_response: AxisymmetricCoilResponsePlan
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.equilibrium, PreparedFixedBoundaryGradShafranov):
            raise TypeError("equilibrium must be PreparedFixedBoundaryGradShafranov.")
        if not isinstance(self.circuits, PreparedCoupledInductance):
            raise TypeError("circuits must be PreparedCoupledInductance.")
        if not isinstance(self.coil_response, AxisymmetricCoilResponsePlan):
            raise TypeError("coil_response must be AxisymmetricCoilResponsePlan.")
        expected = (
            self.equilibrium.z_m.size,
            self.equilibrium.r_m.size,
            self.circuits.winding_count,
        )
        if self.coil_response.boundary_flux_per_amp_wb_per_rad.shape != expected:
            raise ValueError(
                "Coil response does not match equilibrium grid and winding count."
            )
        if self.coil_response.winding_ids != self.circuits.winding_ids:
            raise ValueError("Coil response and circuit winding identities disagree.")
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "free-boundary-tokamak-plan",
                    "equilibrium": self.equilibrium.plan_id,
                    "circuits": self.circuits.plan_id,
                    "coil_response": self.coil_response.response_id,
                    "coupling": "quasi-static-prescribed-plasma-current",
                }
            ),
        )

    def prepare(self) -> PreparedFreeBoundaryTokamak:
        return PreparedFreeBoundaryTokamak(
            self.equilibrium,
            self.circuits,
            self.coil_response.prepare(),
            self.plan_id,
        )


class PreparedFreeBoundaryTokamak(StrictModule, NonTrainableState):
    equilibrium: PreparedFixedBoundaryGradShafranov
    circuits: PreparedCoupledInductance
    coil_response: PreparedAxisymmetricCoilResponse
    plan_id: str = eqx.field(static=True)

    def state(
        self, winding_current_a: ArrayLike, time_s: ArrayLike = 0.0, /
    ) -> FreeBoundaryTokamakState:
        state = FreeBoundaryTokamakState(winding_current_a, time_s, self.plan_id)
        if state.winding_current_a.shape != (self.circuits.winding_count,):
            raise ValueError("Free-boundary state current does not match winding count.")
        return state

    def step(
        self,
        state: FreeBoundaryTokamakState,
        winding_voltage_v: ArrayLike,
        toroidal_current_density_a_m2: ArrayLike,
        external_boundary_flux_wb_per_rad: ArrayLike,
        dt_s: ArrayLike,
        /,
    ) -> FreeBoundaryTokamakStepResult:
        if (
            not isinstance(state, FreeBoundaryTokamakState)
            or state.plan_id != self.plan_id
        ):
            raise ValueError("Free-boundary state does not belong to this plan.")
        voltage = jnp.asarray(winding_voltage_v, dtype=state.winding_current_a.dtype)
        if voltage.shape != state.winding_current_a.shape:
            raise ValueError("Winding voltage must match winding current shape.")
        passive = jnp.asarray(
            [
                role is TokamakWindingRole.PASSIVE
                for role in self.coil_response.winding_roles
            ]
        )
        passive_valid = jnp.all(jnp.where(passive, voltage == 0.0, True))
        circuit = self.circuits.step_implicit_euler(
            state.winding_current_a, voltage, dt_s
        )
        coil_boundary = self.coil_response.boundary_flux(circuit.candidate_current_a)
        external = jnp.asarray(
            external_boundary_flux_wb_per_rad, dtype=coil_boundary.dtype
        )
        if external.shape != coil_boundary.shape:
            raise ValueError(
                "External boundary flux must match the equilibrium R-Z grid."
            )
        boundary = external + coil_boundary
        equilibrium = self.equilibrium.solve(toroidal_current_density_a_m2, boundary)
        finite = circuit.finite & equilibrium.finite & jnp.all(jnp.isfinite(boundary))
        domain_valid = circuit.domain_valid & equilibrium.domain_valid & passive_valid
        successful = circuit.successful & equilibrium.successful & finite & passive_valid
        candidate_state = self.state(
            circuit.candidate_current_a, state.time_s + jnp.asarray(dt_s)
        )
        accepted_state = self.state(
            jnp.where(successful, circuit.candidate_current_a, state.winding_current_a),
            jnp.where(successful, state.time_s + jnp.asarray(dt_s), state.time_s),
        )
        return FreeBoundaryTokamakStepResult(
            candidate_state,
            accepted_state,
            circuit,
            equilibrium,
            boundary,
            finite,
            domain_valid,
            successful,
            successful,
        )


__all__ = [
    "AxisymmetricCoilResponsePlan",
    "AxisymmetricFilamentCoil",
    "FreeBoundaryTokamakPlan",
    "FreeBoundaryTokamakState",
    "FreeBoundaryTokamakStepResult",
    "PreparedAxisymmetricCoilResponse",
    "PreparedFreeBoundaryTokamak",
    "TokamakWindingRole",
]
