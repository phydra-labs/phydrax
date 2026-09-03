#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.spectral._space import TensorSpectralDiscretization


InteractionKind: TypeAlias = Literal["nl", "ql", "gql"]
BilinearAction: TypeAlias = Callable[[Array, Array], Array]


def _conjugate_indices(discretization: TensorSpectralDiscretization, /) -> np.ndarray:
    shape = discretization.modal_shape
    multi = np.indices(shape, dtype=np.int64).reshape((len(shape), -1))
    conjugate_multi = np.stack(
        tuple(
            np.asarray(
                discretization.axes[axis_index].modes.conjugate_indices,
                dtype=np.int64,
            )[multi[axis_index]]
            for axis_index in range(len(shape))
        ),
        axis=0,
    )
    return np.ravel_multi_index(conjugate_multi, shape)


class InteractionPartition(StrictModule, NonTrainableState):
    """A disjoint Hermitian-closed low/high modal partition.

    Forbidden modes are in neither partition.  This makes every projection
    preserve a real physical field and prevents a triad model from reviving a
    filtered Nyquist or gauge mode.
    """

    low_mask: Array
    high_mask: Array
    admissibility_mask: Array
    conjugate_indices: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    low_count: int = eqx.field(static=True)
    high_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        low_mask: ArrayLike,
        conjugate_indices: ArrayLike,
        /,
        *,
        admissibility_mask: ArrayLike | None = None,
        partition_id: str | None = None,
    ):
        low = np.asarray(low_mask, dtype=bool)
        if low.ndim < 1 or low.size < 1:
            raise ValueError("low_mask must be a non-empty modal array.")
        conjugates = np.asarray(conjugate_indices, dtype=np.int64).reshape((-1,))
        if (
            conjugates.shape != (low.size,)
            or np.any(conjugates < 0)
            or np.any(conjugates >= low.size)
            or np.any(conjugates[conjugates] != np.arange(low.size))
        ):
            raise ValueError("conjugate_indices must be an involution over all modes.")
        admissible = (
            np.ones(low.shape, dtype=bool)
            if admissibility_mask is None
            else np.asarray(admissibility_mask, dtype=bool)
        )
        if admissible.shape != low.shape:
            raise ValueError("admissibility_mask and low_mask must have the same shape.")
        flat_low = low.reshape((-1,))
        flat_admissible = admissible.reshape((-1,))
        if np.any(flat_low & ~flat_admissible):
            raise ValueError("The low partition cannot include forbidden modes.")
        if np.any(flat_low != flat_low[conjugates]) or np.any(
            flat_admissible != flat_admissible[conjugates]
        ):
            raise ValueError("Partition masks must be closed under Hermitian conjugacy.")
        high = admissible & ~low
        payload = {
            "kind": "interaction-partition",
            "shape": list(low.shape),
            "low_indices": np.flatnonzero(flat_low).tolist(),
            "admissible_indices": np.flatnonzero(flat_admissible).tolist(),
            "conjugates": conjugates.tolist(),
        }
        identifier = (
            canonical_fingerprint(payload) if partition_id is None else str(partition_id)
        )
        if not identifier:
            raise ValueError("partition_id must be non-empty.")
        self.low_mask = jnp.asarray(low)
        self.high_mask = jnp.asarray(high)
        self.admissibility_mask = jnp.asarray(admissible)
        self.conjugate_indices = jnp.asarray(conjugates, dtype=jnp.int32)
        self.state_shape = tuple(int(size) for size in low.shape)
        self.low_count = int(np.count_nonzero(low))
        self.high_count = int(np.count_nonzero(high))
        self.partition_id = identifier

    @classmethod
    def from_wavenumber_cutoff(
        cls,
        discretization: TensorSpectralDiscretization,
        cutoff: int,
        /,
        *,
        axes: Sequence[int] | None = None,
        admissibility_mask: ArrayLike | None = None,
    ) -> "InteractionPartition":
        if not isinstance(discretization, TensorSpectralDiscretization):
            raise TypeError("discretization must be a TensorSpectralDiscretization.")
        if any(axis.family != "fourier" for axis in discretization.axes):
            raise ValueError("Spectral interaction partitions require Fourier axes.")
        cutoff_ = int(cutoff)
        if cutoff_ < 0:
            raise ValueError("cutoff must be non-negative.")
        selected = (
            tuple(range(len(discretization.axes)))
            if axes is None
            else tuple(int(axis) for axis in axes)
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(axis < 0 or axis >= len(discretization.axes) for axis in selected)
        ):
            raise ValueError("axes must contain unique valid Fourier axes.")
        low = np.ones(discretization.modal_shape, dtype=bool)
        for axis_index in selected:
            axis = discretization.axes[axis_index]
            one_dimensional = np.abs(np.asarray(axis.modes.mode_numbers)) <= cutoff_
            reshape = [1] * len(discretization.axes)
            reshape[axis_index] = axis.mode_count
            low &= np.broadcast_to(one_dimensional.reshape(tuple(reshape)), low.shape)
        admissible = (
            np.ones(discretization.modal_shape, dtype=bool)
            if admissibility_mask is None
            else np.asarray(admissibility_mask, dtype=bool)
        )
        low &= admissible
        return cls(
            low,
            _conjugate_indices(discretization),
            admissibility_mask=admissible,
            partition_id=canonical_fingerprint(
                {
                    "kind": "spectral-interaction-partition",
                    "discretization": discretization.prepared_id,
                    "cutoff": cutoff_,
                    "axes": list(selected),
                    "admissible_count": int(np.count_nonzero(admissible)),
                }
            ),
        )

    @classmethod
    def zonal_mean(
        cls,
        discretization: TensorSpectralDiscretization,
        /,
        *,
        zonal_axis: int = 0,
        admissibility_mask: ArrayLike | None = None,
    ) -> "InteractionPartition":
        return cls.from_wavenumber_cutoff(
            discretization,
            0,
            axes=(zonal_axis,),
            admissibility_mask=admissibility_mask,
        )

    def _validate(self, state: ArrayLike, name: str, /) -> Array:
        value = jnp.asarray(state)
        if value.shape[: len(self.state_shape)] != self.state_shape:
            raise ValueError(
                f"{name} must begin with modal shape {self.state_shape}; got {value.shape}."
            )
        return value

    def _mask(self, mask: Array, value: Array, /) -> Array:
        return value * mask.reshape(
            self.state_shape + (1,) * (value.ndim - len(self.state_shape))
        )

    def low(self, state: ArrayLike, /) -> Array:
        value = self._validate(state, "Partitioned state")
        return self._mask(self.low_mask, value)

    def high(self, state: ArrayLike, /) -> Array:
        value = self._validate(state, "Partitioned state")
        return self._mask(self.high_mask, value)

    def admissible(self, state: ArrayLike, /) -> Array:
        value = self._validate(state, "Partitioned state")
        return self._mask(self.admissibility_mask, value)

    def mask_is_closed(self, /) -> Array:
        low = self.low_mask.reshape((-1,))
        high = self.high_mask.reshape((-1,))
        return jnp.all(low == low[self.conjugate_indices]) & jnp.all(
            high == high[self.conjugate_indices]
        )

    def triad_retained(
        self,
        output_index: int,
        left_index: int,
        right_index: int,
        /,
        *,
        model: InteractionKind,
    ) -> bool:
        indices = (int(output_index), int(left_index), int(right_index))
        if any(index < 0 or index >= int(self.low_mask.size) for index in indices):
            raise IndexError("Triad indices must address flattened partition modes.")
        output, left, right = indices
        admissible = np.asarray(self.admissibility_mask).reshape((-1,))
        low = np.asarray(self.low_mask).reshape((-1,))
        if not (admissible[output] and admissible[left] and admissible[right]):
            return False
        if model == "nl":
            return True
        if model == "ql":
            return bool(
                (low[output] and (low[left] == low[right]))
                or (not low[output] and (low[left] != low[right]))
            )
        if model == "gql":
            return bool(low[output] or (low[left] != low[right]))
        raise ValueError("model must be 'nl', 'ql', or 'gql'.")

    def select(
        self,
        bilinear: BilinearAction,
        state: ArrayLike,
        /,
        *,
        model: InteractionKind,
        interaction_coordinate: ArrayLike = 0.0,
    ) -> Array:
        if not callable(bilinear):
            raise TypeError("bilinear must be callable.")
        value = self.admissible(state)
        coordinate = jnp.asarray(interaction_coordinate, dtype=value.real.dtype)
        if coordinate.shape != ():
            raise ValueError("interaction_coordinate must be scalar.")
        coordinate = eqx.error_if(
            coordinate,
            ~jnp.isfinite(coordinate) | (coordinate < 0.0) | (coordinate > 1.0),
            "interaction_coordinate must lie in [0, 1].",
        )
        nonlinear = self.admissible(bilinear(value, value))
        if model == "nl":
            return nonlinear
        low = self.low(value)
        high = self.high(value)
        cross = bilinear(low, high) + bilinear(high, low)
        if model == "ql":
            low_rhs = self.low(bilinear(low, low) + bilinear(high, high))
            selected = low_rhs + self.high(cross)
        elif model == "gql":
            selected = self.low(nonlinear) + self.high(cross)
        else:
            raise ValueError("model must be 'nl', 'ql', or 'gql'.")
        return selected + coordinate.astype(value.dtype) * (nonlinear - selected)

    def selector(self, model: InteractionKind, /) -> Callable:
        if model not in ("nl", "ql", "gql"):
            raise ValueError("model must be 'nl', 'ql', or 'gql'.")

        def apply(
            bilinear: BilinearAction,
            state: Array,
            interaction_coordinate: ArrayLike,
        ) -> Array:
            return self.select(
                bilinear,
                state,
                model=model,
                interaction_coordinate=interaction_coordinate,
            )

        return apply


class AbstractInteractionModel(StrictModule, NonTrainableState):
    kind: InteractionKind = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def select(
        self,
        partition: InteractionPartition,
        bilinear: BilinearAction,
        state: ArrayLike,
        /,
        *,
        interaction_coordinate: ArrayLike = 0.0,
    ) -> Array:
        raise NotImplementedError


def _select_model(
    model: AbstractInteractionModel,
    partition: InteractionPartition,
    bilinear: BilinearAction,
    state: ArrayLike,
    interaction_coordinate: ArrayLike,
    /,
) -> Array:
    if not isinstance(partition, InteractionPartition):
        raise TypeError("partition must be an InteractionPartition.")
    return partition.select(
        bilinear,
        state,
        model=model.kind,
        interaction_coordinate=interaction_coordinate,
    )


class NonlinearInteractions(AbstractInteractionModel):
    def __init__(self):
        self.kind = "nl"
        self.model_id = canonical_fingerprint({"kind": "triad-selection", "model": "nl"})

    def select(
        self,
        partition: InteractionPartition,
        bilinear: BilinearAction,
        state: ArrayLike,
        /,
        *,
        interaction_coordinate: ArrayLike = 0.0,
    ) -> Array:
        return _select_model(self, partition, bilinear, state, interaction_coordinate)


class QuasilinearInteractions(AbstractInteractionModel):
    def __init__(self):
        self.kind = "ql"
        self.model_id = canonical_fingerprint({"kind": "triad-selection", "model": "ql"})

    def select(
        self,
        partition: InteractionPartition,
        bilinear: BilinearAction,
        state: ArrayLike,
        /,
        *,
        interaction_coordinate: ArrayLike = 0.0,
    ) -> Array:
        return _select_model(self, partition, bilinear, state, interaction_coordinate)


class GeneralizedQuasilinearInteractions(AbstractInteractionModel):
    def __init__(self):
        self.kind = "gql"
        self.model_id = canonical_fingerprint({"kind": "triad-selection", "model": "gql"})

    def select(
        self,
        partition: InteractionPartition,
        bilinear: BilinearAction,
        state: ArrayLike,
        /,
        *,
        interaction_coordinate: ArrayLike = 0.0,
    ) -> Array:
        return _select_model(self, partition, bilinear, state, interaction_coordinate)


class InteractionContinuationStage(StrictModule, NonTrainableState):
    coordinate: float = eqx.field(static=True)
    stage_id: str = eqx.field(static=True)

    def __init__(self, coordinate: float, /, *, stage_id: str | None = None):
        value = float(coordinate)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("Interaction continuation coordinates must lie in [0, 1].")
        identifier = (
            canonical_fingerprint(
                {"kind": "interaction-continuation-stage", "coordinate": value}
            )
            if stage_id is None
            else str(stage_id)
        )
        if not identifier:
            raise ValueError("stage_id must be non-empty.")
        self.coordinate = value
        self.stage_id = identifier


class InteractionContinuationSchedule(StrictModule, NonTrainableState):
    stages: tuple[InteractionContinuationStage, ...]
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinates: Sequence[float | InteractionContinuationStage],
        /,
        *,
        schedule_id: str | None = None,
    ):
        stages = tuple(
            value
            if isinstance(value, InteractionContinuationStage)
            else InteractionContinuationStage(value)
            for value in coordinates
        )
        if not stages:
            raise ValueError("An interaction continuation schedule cannot be empty.")
        values = tuple(stage.coordinate for stage in stages)
        if any(right < left for left, right in zip(values, values[1:])):
            raise ValueError(
                "Interaction continuation coordinates must be nondecreasing."
            )
        identifiers = tuple(stage.stage_id for stage in stages)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Interaction continuation stage identities must be unique.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "interaction-continuation-schedule",
                    "coordinates": list(values),
                    "stages": list(identifiers),
                }
            )
            if schedule_id is None
            else str(schedule_id)
        )
        if not identifier:
            raise ValueError("schedule_id must be non-empty.")
        self.stages = stages
        self.schedule_id = identifier


__all__ = [
    "AbstractInteractionModel",
    "GeneralizedQuasilinearInteractions",
    "InteractionContinuationSchedule",
    "InteractionContinuationStage",
    "InteractionKind",
    "InteractionPartition",
    "NonlinearInteractions",
    "QuasilinearInteractions",
]
