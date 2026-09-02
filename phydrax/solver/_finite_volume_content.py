#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..discretization._conservation_ledger import (
    ConservationStageLedger,
)
from ..discretization._fv_precision import FiniteVolumePrecisionPolicy


def _nonempty_identifier(value: str, /, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string.")
    return value


def _component_shape(value: Array, /) -> tuple[int, ...]:
    return tuple(int(size) for size in value.shape[1:])


def _reduction_value(
    precision: FiniteVolumePrecisionPolicy, value: ArrayLike, /
) -> Array:
    return jnp.asarray(value, dtype=precision.reduction_dtype)


def _integer_scalar(value: ArrayLike, /, *, name: str) -> Array:
    array = jnp.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    if not jnp.issubdtype(array.dtype, jnp.integer):
        raise TypeError(f"{name} must have an integer dtype.")
    return array


def _active_cell_mask(value: ArrayLike, /, *, cell_count: int) -> Array:
    mask = jnp.asarray(value)
    if mask.shape != (cell_count,) or mask.dtype != jnp.bool_:
        raise ValueError(
            "active_cell_mask must be a boolean array with exact shape (cell_count,)."
        )
    return mask


class FiniteVolumeConservativeContentState(StrictModule):
    """Immutable content and active-cell ownership for one effective geometry."""

    conservative_content: Array
    effective_cell_volumes: Array
    active_cell_mask: Array
    time: Array
    geometry_version: Array
    evidence_version: Array
    topology_epoch_id: str = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    evidence_policy_id: str = eqx.field(static=True)
    precision: FiniteVolumePrecisionPolicy

    def __init__(
        self,
        conservative_content: ArrayLike,
        effective_cell_volumes: ArrayLike,
        active_cell_mask: ArrayLike,
        time: ArrayLike,
        /,
        *,
        topology_epoch_id: str,
        geometry_family_id: str,
        geometry_layout_id: str,
        geometry_version: ArrayLike,
        evidence_policy_id: str,
        evidence_version: ArrayLike,
        precision: FiniteVolumePrecisionPolicy,
    ):
        if not isinstance(precision, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be a FiniteVolumePrecisionPolicy.")
        content = precision.storage(conservative_content)
        volumes = _reduction_value(precision, effective_cell_volumes)
        time_ = _reduction_value(precision, time)
        precision.validate_state(content)
        if jnp.dtype(volumes.dtype).name != precision.reduction_dtype:
            raise TypeError(
                "Effective cell volume dtype does not match finite-volume "
                "reduction precision."
            )
        if jnp.dtype(time_.dtype).name != precision.reduction_dtype:
            raise TypeError(
                "Finite-volume state time dtype does not match reduction precision."
            )
        if content.ndim < 2 or any(size <= 0 for size in content.shape):
            raise ValueError(
                "conservative_content must have shape (cells, *component_shape) "
                "with non-empty cell and component axes."
            )
        if volumes.shape != (content.shape[0],):
            raise ValueError(
                "effective_cell_volumes must have exact shape (cell_count,)."
            )
        active = _active_cell_mask(active_cell_mask, cell_count=int(content.shape[0]))
        if time_.shape != ():
            raise ValueError("time must be scalar.")
        volumes = eqx.error_if(
            volumes,
            jnp.any(active & (~jnp.isfinite(volumes) | (volumes <= 0.0))),
            "Active effective cell volumes must be finite and strictly positive.",
        )
        volumes = eqx.error_if(
            volumes,
            jnp.any((~active) & (volumes != 0.0)),
            "Inactive effective cell volumes must be exactly zero.",
        )
        trailing = (1,) * (content.ndim - 1)
        active_content = active.reshape((-1,) + trailing)
        content = eqx.error_if(
            content,
            jnp.any((~active_content) & (content != 0.0)),
            "Inactive cells must have exactly zero conservative content.",
        )
        time_ = eqx.error_if(
            time_, ~jnp.isfinite(time_), "Finite-volume state time must be finite."
        )
        self.conservative_content = content
        self.effective_cell_volumes = volumes
        self.active_cell_mask = active
        self.time = time_
        self.geometry_version = _integer_scalar(geometry_version, name="geometry_version")
        self.evidence_version = _integer_scalar(evidence_version, name="evidence_version")
        self.topology_epoch_id = _nonempty_identifier(
            topology_epoch_id, name="topology_epoch_id"
        )
        self.geometry_family_id = _nonempty_identifier(
            geometry_family_id,
            name="geometry_family_id",
        )
        self.geometry_layout_id = _nonempty_identifier(
            geometry_layout_id, name="geometry_layout_id"
        )
        self.evidence_policy_id = _nonempty_identifier(
            evidence_policy_id, name="evidence_policy_id"
        )
        self.precision = precision

    @classmethod
    def from_cell_average(
        cls,
        cell_average: ArrayLike,
        effective_cell_volumes: ArrayLike,
        active_cell_mask: ArrayLike,
        time: ArrayLike,
        /,
        *,
        topology_epoch_id: str,
        geometry_family_id: str,
        geometry_layout_id: str,
        geometry_version: ArrayLike,
        evidence_policy_id: str,
        evidence_version: ArrayLike,
        precision: FiniteVolumePrecisionPolicy,
    ) -> FiniteVolumeConservativeContentState:
        """Form content from active averages; inactive content is exactly zero."""

        if not isinstance(precision, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be a FiniteVolumePrecisionPolicy.")
        average = precision.storage(cell_average)
        volumes = _reduction_value(precision, effective_cell_volumes)
        if average.ndim < 2 or any(size <= 0 for size in average.shape):
            raise ValueError(
                "cell_average must have shape (cells, *component_shape) with "
                "non-empty cell and component axes."
            )
        if volumes.shape != (average.shape[0],):
            raise ValueError(
                "effective_cell_volumes must have exact shape (cell_count,)."
            )
        active = _active_cell_mask(active_cell_mask, cell_count=int(average.shape[0]))
        trailing = (1,) * (average.ndim - 1)
        active_values = active.reshape((-1,) + trailing)
        reduction_average = precision.reduction(average)
        masked_average = jnp.where(
            active_values, reduction_average, jnp.zeros_like(reduction_average)
        )
        masked_volumes = jnp.where(active, volumes, jnp.zeros_like(volumes)).reshape(
            (-1,) + trailing
        )
        content = precision.storage(masked_average * masked_volumes)
        return cls(
            content,
            volumes,
            active,
            time,
            topology_epoch_id=topology_epoch_id,
            geometry_family_id=geometry_family_id,
            geometry_layout_id=geometry_layout_id,
            geometry_version=geometry_version,
            evidence_policy_id=evidence_policy_id,
            evidence_version=evidence_version,
            precision=precision,
        )

    @property
    def cell_count(self) -> int:
        return int(self.conservative_content.shape[0])

    @property
    def component_shape(self) -> tuple[int, ...]:
        return _component_shape(self.conservative_content)

    def cell_average(self) -> Array:
        """Derive active averages, returning exact zero for every inactive cell."""

        trailing = (1,) * (self.conservative_content.ndim - 1)
        active = self.active_cell_mask.reshape((-1,) + trailing)
        safe_volumes = jnp.where(
            self.active_cell_mask,
            self.effective_cell_volumes,
            jnp.ones_like(self.effective_cell_volumes),
        ).reshape((-1,) + trailing)
        average = self.precision.reduction(self.conservative_content) / safe_volumes
        masked_average = jnp.where(active, average, jnp.zeros_like(average))
        return self.precision.storage(masked_average)

    def with_content(
        self,
        conservative_content: ArrayLike,
        /,
        *,
        time: ArrayLike | None = None,
        evidence_version: ArrayLike | None = None,
    ) -> FiniteVolumeConservativeContentState:
        """Return new content while preserving geometry and active-cell ownership."""

        content = jnp.asarray(conservative_content)
        if content.shape != self.conservative_content.shape:
            raise ValueError(
                "Replacement conservative_content must preserve exact cell and "
                "component shapes."
            )
        return FiniteVolumeConservativeContentState(
            content,
            self.effective_cell_volumes,
            self.active_cell_mask,
            self.time if time is None else time,
            topology_epoch_id=self.topology_epoch_id,
            geometry_family_id=self.geometry_family_id,
            geometry_layout_id=self.geometry_layout_id,
            geometry_version=self.geometry_version,
            evidence_policy_id=self.evidence_policy_id,
            evidence_version=(
                self.evidence_version if evidence_version is None else evidence_version
            ),
            precision=self.precision,
        )

    def with_topology_epoch(
        self,
        topology_epoch_id: str,
        /,
        *,
        geometry_family_id: str | None = None,
        geometry_layout_id: str | None = None,
        geometry_version: ArrayLike | None = None,
        evidence_version: ArrayLike | None = None,
    ) -> FiniteVolumeConservativeContentState:
        """Rebind unchanged content to one validated successor topology epoch."""

        return FiniteVolumeConservativeContentState(
            self.conservative_content,
            self.effective_cell_volumes,
            self.active_cell_mask,
            self.time,
            topology_epoch_id=topology_epoch_id,
            geometry_family_id=(
                self.geometry_family_id
                if geometry_family_id is None
                else geometry_family_id
            ),
            geometry_layout_id=(
                self.geometry_layout_id
                if geometry_layout_id is None
                else geometry_layout_id
            ),
            geometry_version=(
                self.geometry_version if geometry_version is None else geometry_version
            ),
            evidence_policy_id=self.evidence_policy_id,
            evidence_version=(
                self.evidence_version if evidence_version is None else evidence_version
            ),
            precision=self.precision,
        )

    def volume_integral(self) -> Array:
        """Return the component-wise integral already represented by content."""

        return jnp.sum(self.precision.reduction(self.conservative_content), axis=0)

    def conservation_change(
        self,
        reference: FiniteVolumeConservativeContentState | ArrayLike,
        /,
    ) -> Array:
        """Return the component-wise change from a state or reference integral."""

        if isinstance(reference, FiniteVolumeConservativeContentState):
            if reference.component_shape != self.component_shape:
                raise ValueError(
                    "Reference state must have the same exact component shape."
                )
            reference_integral = reference.volume_integral()
        else:
            reference_integral = _reduction_value(self.precision, reference)
            if reference_integral.shape != self.component_shape:
                raise ValueError(
                    "Reference integral must have the exact component shape."
                )
        return self.volume_integral() - _reduction_value(
            self.precision, reference_integral
        )


def apply_stage_rate_euler_update(
    state: FiniteVolumeConservativeContentState,
    ledger: ConservationStageLedger,
    local_euler_increment: ArrayLike,
    /,
    *,
    target_time: ArrayLike,
    target_cell_volumes: ArrayLike | None = None,
    target_geometry_version: ArrayLike | None = None,
    target_evidence_version: ArrayLike | None = None,
) -> FiniteVolumeConservativeContentState:
    """Apply one ledger rate for a local increment at an explicit target time.

    The ledger scatter is the complete content rate, including its source
    contribution. ``target_time`` is independent of ``local_euler_increment``.

    The ledger must certify the input state's geometry and evidence identity.
    A changed target geometry requires volumes, geometry version, and evidence
    version together; a static update preserves the input evidence version.
    """

    if not isinstance(state, FiniteVolumeConservativeContentState):
        raise TypeError("state must be FiniteVolumeConservativeContentState.")
    if not isinstance(ledger, ConservationStageLedger):
        raise TypeError("ledger must be ConservationStageLedger.")
    if ledger.topology_epoch_id != state.topology_epoch_id:
        raise ValueError("Stage rate ledger topology does not match the input state.")
    if ledger.geometry_family_id != state.geometry_family_id:
        raise ValueError(
            "Stage rate ledger geometry family does not match the input state."
        )
    if ledger.geometry_layout_id != state.geometry_layout_id:
        raise ValueError(
            "Stage rate ledger geometry layout does not match the input state."
        )
    if ledger.evidence_policy_id != state.evidence_policy_id:
        raise ValueError(
            "Stage rate ledger evidence policy does not match the input state."
        )
    if (
        ledger.cell_count != state.cell_count
        or ledger.component_shape != state.component_shape
    ):
        raise ValueError(
            "Stage rate ledger must match the exact state cell and component shapes."
        )
    target_geometry_fields = (
        target_cell_volumes,
        target_geometry_version,
        target_evidence_version,
    )
    if any(value is not None for value in target_geometry_fields) and not all(
        value is not None for value in target_geometry_fields
    ):
        raise ValueError(
            "target_cell_volumes, target_geometry_version, and "
            "target_evidence_version must be supplied together."
        )

    content_rate = ledger.scatter_content_rate()
    if content_rate.shape != state.conservative_content.shape:
        raise ValueError(
            "Scattered stage content rate must match the exact state cell and "
            "component shapes."
        )
    content_rate = eqx.error_if(
        content_rate,
        ledger.geometry_version != state.geometry_version,
        "Stage rate ledger starting geometry version does not match the input state.",
    )
    content_rate = eqx.error_if(
        content_rate,
        ledger.evidence_version != state.evidence_version,
        "Stage rate ledger starting evidence version does not match the input state.",
    )
    content_rate = eqx.error_if(
        content_rate,
        jnp.any(ledger.active_cell_mask != state.active_cell_mask),
        "Stage rate ledger active-cell mask does not match the input state.",
    )
    increment = _reduction_value(state.precision, local_euler_increment)
    if increment.shape != ():
        raise ValueError("local_euler_increment must be scalar.")
    increment = eqx.error_if(
        increment,
        ~jnp.isfinite(increment),
        "local_euler_increment must be finite.",
    )
    updated_content = state.precision.storage(
        state.precision.reduction(state.conservative_content)
        + increment * _reduction_value(state.precision, content_rate)
    )
    return FiniteVolumeConservativeContentState(
        updated_content,
        (
            state.effective_cell_volumes
            if target_cell_volumes is None
            else target_cell_volumes
        ),
        state.active_cell_mask,
        target_time,
        topology_epoch_id=state.topology_epoch_id,
        geometry_family_id=state.geometry_family_id,
        geometry_layout_id=state.geometry_layout_id,
        geometry_version=(
            state.geometry_version
            if target_geometry_version is None
            else target_geometry_version
        ),
        evidence_policy_id=state.evidence_policy_id,
        evidence_version=(
            state.evidence_version
            if target_evidence_version is None
            else target_evidence_version
        ),
        precision=state.precision,
    )


__all__ = [
    "FiniteVolumeConservativeContentState",
    "apply_stage_rate_euler_update",
]
