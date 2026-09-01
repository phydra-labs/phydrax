#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState
from ._transfer import APICGatherResult, gather_apic


class AbstractMPMVelocityTransferPlan(StrictModule, NonTrainableState):
    transfer_name: AbstractAttribute[str]
    requires_affine_state: AbstractAttribute[bool]
    uses_grid_delta: AbstractAttribute[bool]
    maximum_condition: AbstractAttribute[float]
    plan_id: AbstractAttribute[str]


class APICTransferPlan(AbstractMPMVelocityTransferPlan):
    transfer_name: str = eqx.field(static=True)
    requires_affine_state: bool = eqx.field(static=True)
    uses_grid_delta: bool = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, *, maximum_condition: float = 1.0e8):
        condition = float(maximum_condition)
        if not np.isfinite(condition) or condition <= 1.0:
            raise ValueError("maximum_condition must be finite and greater than one.")
        self.transfer_name = "apic"
        self.requires_affine_state = True
        self.uses_grid_delta = False
        self.maximum_condition = condition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-velocity-transfer",
                "name": "apic",
                "maximum_condition": condition,
            }
        )


class PICTransferPlan(AbstractMPMVelocityTransferPlan):
    transfer_name: str = eqx.field(static=True)
    requires_affine_state: bool = eqx.field(static=True)
    uses_grid_delta: bool = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self):
        self.transfer_name = "pic"
        self.requires_affine_state = False
        self.uses_grid_delta = False
        self.maximum_condition = np.inf
        self.plan_id = canonical_fingerprint(
            {"kind": "mpm-velocity-transfer", "name": "pic"}
        )


class FLIPTransferPlan(AbstractMPMVelocityTransferPlan):
    transfer_name: str = eqx.field(static=True)
    requires_affine_state: bool = eqx.field(static=True)
    uses_grid_delta: bool = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self):
        self.transfer_name = "flip"
        self.requires_affine_state = False
        self.uses_grid_delta = True
        self.maximum_condition = np.inf
        self.plan_id = canonical_fingerprint(
            {"kind": "mpm-velocity-transfer", "name": "flip"}
        )


class PICFLIPTransferPlan(AbstractMPMVelocityTransferPlan):
    transfer_name: str = eqx.field(static=True)
    requires_affine_state: bool = eqx.field(static=True)
    uses_grid_delta: bool = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    pic_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, pic_fraction: float, /):
        fraction = float(pic_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("pic_fraction must lie in [0, 1].")
        self.transfer_name = "pic-flip"
        self.requires_affine_state = False
        self.uses_grid_delta = True
        self.maximum_condition = np.inf
        self.pic_fraction = fraction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mpm-velocity-transfer",
                "name": "pic-flip",
                "pic_fraction": fraction,
                "convention": "pic_fraction*PIC+(1-pic_fraction)*FLIP",
            }
        )


class AbstractMPMAdvectionPlan(StrictModule, NonTrainableState):
    advection_name: AbstractAttribute[str]
    plan_id: AbstractAttribute[str]

    @abc.abstractmethod
    def velocity(
        self,
        transferred_velocity: Array,
        pic_velocity: Array,
        previous_velocity: Array,
        /,
    ) -> Array:
        raise NotImplementedError


class PICAdvectionPlan(AbstractMPMAdvectionPlan):
    advection_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self):
        self.advection_name = "pic"
        self.plan_id = canonical_fingerprint({"kind": "mpm-advection", "name": "pic"})

    def velocity(self, transferred_velocity, pic_velocity, previous_velocity, /):
        del transferred_velocity, previous_velocity
        return pic_velocity


class TransferredVelocityAdvectionPlan(AbstractMPMAdvectionPlan):
    advection_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self):
        self.advection_name = "transferred-velocity"
        self.plan_id = canonical_fingerprint(
            {"kind": "mpm-advection", "name": "transferred-velocity"}
        )

    def velocity(self, transferred_velocity, pic_velocity, previous_velocity, /):
        del pic_velocity, previous_velocity
        return transferred_velocity


class MidpointAdvectionPlan(AbstractMPMAdvectionPlan):
    advection_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self):
        self.advection_name = "midpoint"
        self.plan_id = canonical_fingerprint(
            {"kind": "mpm-advection", "name": "midpoint"}
        )

    def velocity(self, transferred_velocity, pic_velocity, previous_velocity, /):
        del pic_velocity
        return 0.5 * (previous_velocity + transferred_velocity)


class MPMVelocityTransferResult(StrictModule):
    velocity: Array
    pic_velocity: Array
    advection_velocity: Array
    velocity_gradient: Array
    affine_velocity: Array
    condition_estimate: Array
    successful: Array


def apply_velocity_transfer(
    transfer: AbstractMPMVelocityTransferPlan,
    advection: AbstractMPMAdvectionPlan,
    routes,
    grid_velocity_before: Array,
    grid_velocity_after: Array,
    particle_velocity: Array,
    active: Array,
    /,
) -> MPMVelocityTransferResult:
    if not isinstance(transfer, AbstractMPMVelocityTransferPlan):
        raise TypeError("transfer must be AbstractMPMVelocityTransferPlan.")
    if not isinstance(advection, AbstractMPMAdvectionPlan):
        raise TypeError("advection must be AbstractMPMAdvectionPlan.")
    dimension = int(grid_velocity_after.shape[-1])
    after: APICGatherResult = gather_apic(
        routes,
        grid_velocity_after.reshape((-1, dimension)),
        active,
        transfer.maximum_condition if np.isfinite(transfer.maximum_condition) else 1.0e30,
    )
    pic = after.velocity
    if isinstance(transfer, APICTransferPlan):
        velocity = pic
        affine = after.affine_velocity
        successful = after.successful
    elif isinstance(transfer, PICTransferPlan):
        velocity = pic
        affine = jnp.zeros_like(after.affine_velocity)
        successful = jnp.all(jnp.isfinite(velocity))
    else:
        before = gather_apic(
            routes,
            grid_velocity_before.reshape((-1, dimension)),
            active,
            1.0e30,
        )
        flip = particle_velocity + (after.velocity - before.velocity)
        if isinstance(transfer, FLIPTransferPlan):
            velocity = flip
        else:
            velocity = transfer.pic_fraction * pic + (1.0 - transfer.pic_fraction) * flip
        affine = jnp.zeros_like(after.affine_velocity)
        successful = (
            after.successful & before.successful & jnp.all(jnp.isfinite(velocity))
        )
    advection_velocity = advection.velocity(velocity, pic, particle_velocity)
    return MPMVelocityTransferResult(
        velocity,
        pic,
        advection_velocity,
        after.velocity_gradient,
        affine,
        after.condition_estimate,
        successful & jnp.all(jnp.isfinite(advection_velocity)),
    )


__all__ = [
    "APICTransferPlan",
    "AbstractMPMAdvectionPlan",
    "AbstractMPMVelocityTransferPlan",
    "FLIPTransferPlan",
    "MPMVelocityTransferResult",
    "MidpointAdvectionPlan",
    "PICAdvectionPlan",
    "PICFLIPTransferPlan",
    "PICTransferPlan",
    "TransferredVelocityAdvectionPlan",
    "apply_velocity_transfer",
]
