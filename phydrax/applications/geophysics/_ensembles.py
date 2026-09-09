#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Geophysical semantics around the native state-space ensemble algorithms."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import DiscreteFieldSpace
from ...dynamics import StateLayout
from ...ein import contract
from ...equations import DiscreteStateLayout
from ...linalg import ArraySpace, FunctionLinearOperator
from ...metrix import EuclideanStateGeometry
from ...stochastic import (
    AbstractStatePrior,
    CallableTransitionKernel,
    GaussianObservationModel,
    ObservationSequence,
    state_space_key,
    StateSpaceModel,
    StateSpaceProblem,
    StateSpaceStepContext,
    TransitionSample,
)
from ...uq import EnsembleFilterResult
from ._observations import PreparedGeophysicalObservations
from ._quantities import GeophysicalQuantity
from ._time import GeophysicalTimeSpec


_AXIS_KINDS = (
    "initial_condition",
    "scenario",
    "parameter",
    "structural",
    "internal_stochastic",
)


class GeophysicalEnsembleAxis(StrictModule, NonTrainableState):
    """An uncertainty source, not an exchangeable mixture of physical scenarios."""

    kind: str = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    axis_id: str = eqx.field(static=True)

    def __init__(self, kind: str, labels: Sequence[str], /):
        labels_ = tuple(labels)
        if kind not in _AXIS_KINDS:
            raise ValueError(f"Ensemble axis kind must be one of {_AXIS_KINDS}.")
        if (
            not labels_
            or any(not isinstance(label, str) or not label for label in labels_)
            or len(set(labels_)) != len(labels_)
        ):
            raise ValueError("Ensemble axis labels must be unique nonempty strings.")
        self.kind = kind
        self.labels = labels_
        self.axis_id = canonical_fingerprint(
            {"kind": "geophysical-ensemble-axis", "role": kind, "labels": sorted(labels_)}
        )


class GeophysicalEnsembleLineage(StrictModule, NonTrainableState):
    """One coordinate in five explicitly distinct ensemble axes.

    Run scenario/parameter/structural coordinates as separate physical cases.
    ETKF members sample the selected initial-condition distribution; each member
    receives a distinct internal-process stream. Coordinate keys use semantic
    labels, not their position or the set/order of other available labels.
    """

    axes: tuple[GeophysicalEnsembleAxis, ...]
    coordinates: tuple[str, ...] = eqx.field(static=True)
    case_id: str = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self, axes: Sequence[GeophysicalEnsembleAxis], coordinates: Mapping[str, str], /
    ):
        axes_ = tuple(axes)
        if any(not isinstance(axis, GeophysicalEnsembleAxis) for axis in axes_):
            raise TypeError("axes must contain GeophysicalEnsembleAxis values.")
        by_kind = {axis.kind: axis for axis in axes_}
        if (
            len(axes_) != len(_AXIS_KINDS)
            or set(by_kind) != set(_AXIS_KINDS)
            or set(coordinates) != set(_AXIS_KINDS)
        ):
            raise ValueError(
                "Exactly one coordinate for each of the five uncertainty axes is required."
            )
        if any(coordinates[kind] not in by_kind[kind].labels for kind in _AXIS_KINDS):
            raise ValueError(
                "An ensemble coordinate is not present on its declared axis."
            )
        self.axes = tuple(by_kind[kind] for kind in _AXIS_KINDS)
        self.coordinates = tuple(coordinates[kind] for kind in _AXIS_KINDS)
        self.case_id = canonical_fingerprint(
            {
                "kind": "geophysical-ensemble-case",
                "coordinates": {
                    kind: coordinates[kind]
                    for kind in ("scenario", "parameter", "structural")
                },
            }
        )
        self.lineage_id = canonical_fingerprint(
            {"kind": "geophysical-ensemble-lineage", "coordinates": dict(coordinates)}
        )

    def key(
        self,
        root_key: Array,
        kind: str,
        /,
        *,
        step: int | Array = 0,
        member: int | Array = 0,
    ) -> Array:
        if kind not in _AXIS_KINDS:
            raise ValueError("Unknown uncertainty axis.")
        label = self.coordinates[_AXIS_KINDS.index(kind)]
        return state_space_key(
            root_key, f"geophysical-{kind}", label, step, member=member
        )


class _LineagePrior(AbstractStatePrior):
    prior: AbstractStatePrior
    lineage: GeophysicalEnsembleLineage
    state_shape: tuple[int, ...] = eqx.field(static=True)
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    prior_id: str = eqx.field(static=True)
    has_log_density: bool = eqx.field(static=True)

    def __init__(self, prior: AbstractStatePrior, lineage: GeophysicalEnsembleLineage):
        self.prior = prior
        self.lineage = lineage
        self.state_shape = prior.state_shape
        self.batch_shape = prior.batch_shape
        self.has_log_density = prior.has_log_density
        self.prior_id = canonical_fingerprint(
            {
                "kind": "geophysical-prior",
                "prior": prior.prior_id,
                "content": array_tree_fingerprint(prior),
                "initial_condition": lineage.coordinates[0],
            }
        )

    @property
    def location(self) -> Array:
        return self.prior.location

    def sample(self, key: Array, sample_shape: tuple[int, ...] = ()) -> Array:
        return self.prior.sample(self.lineage.key(key, "initial_condition"), sample_shape)

    def log_prob(self, value: ArrayLike, /) -> Array:
        return self.prior.log_prob(value)


def _layout_shape(layout: StateLayout | DiscreteStateLayout) -> tuple[int, ...]:
    if isinstance(layout, StateLayout):
        if not isinstance(layout.geometry, EuclideanStateGeometry):
            raise ValueError(
                "The native ETKF requires Euclidean, fixed-layout state coordinates."
            )
        return layout.shape
    if isinstance(layout, DiscreteStateLayout):
        return layout.state_shape
    raise TypeError("state_layout must be a native StateLayout or DiscreteStateLayout.")


def prepare_geophysical_assimilation(
    state_layout: StateLayout | DiscreteStateLayout,
    observations: PreparedGeophysicalObservations,
    /,
    *,
    source_field: DiscreteFieldSpace | str,
    prior: AbstractStatePrior,
    transition: Callable[
        [Array, Array, Array, Array, StateSpaceStepContext], Array | TransitionSample
    ],
    model_id: str,
    approximation_id: str,
    model_time: GeophysicalTimeSpec,
    initial_time: float,
    lineage: GeophysicalEnsembleLineage,
    args: Any = None,
) -> StateSpaceProblem:
    """Bind a numerical callback to native ETKF/filter/smoother/checkpoint APIs.

    The callback is ``transition(key, state, t0, t1, context)`` and owns all model
    integration. Times use ``model_time.unit``. It must preserve the native
    layout and return an array or native ``TransitionSample`` with ``process_id
    == model_id``. Callback/parameter physics must be represented by model_id,
    approximation_id, lineage and args; callables themselves are not hashed.

    A DiscreteStateLayout selects an existing named PDE field. A StateLayout
    uses a supplied native field space covering its whole array. Exact field
    identities are checked, not merely equal sizes. No orbit-state, generic
    integrator, stochastic dynamics implementation or filter is introduced here.
    """
    shape = _layout_shape(state_layout)
    if not isinstance(observations, PreparedGeophysicalObservations):
        raise TypeError("observations must be prepared geophysical observations.")
    if not isinstance(lineage, GeophysicalEnsembleLineage):
        raise TypeError("lineage must be GeophysicalEnsembleLineage.")
    if (
        not isinstance(prior, AbstractStatePrior)
        or prior.state_shape != shape
        or prior.batch_shape
    ):
        raise ValueError(
            "The prior must have exactly the native state shape and one physical case."
        )
    if (
        not isinstance(model_time, GeophysicalTimeSpec)
        or model_time.time_id != observations.operator.time.time_id
    ):
        raise ValueError(
            "Model and observation calendar, epoch and time unit must agree."
        )
    if (
        not callable(transition)
        or not isinstance(model_id, str)
        or not model_id
        or not isinstance(approximation_id, str)
        or not approximation_id
    ):
        raise ValueError(
            "A numerical callback and nonempty model/approximation identities are required."
        )
    if isinstance(state_layout, DiscreteStateLayout):
        if (
            not isinstance(source_field, str)
            or source_field not in state_layout.field_names
        ):
            raise ValueError(
                "source_field must name a field in the native PDE state layout."
            )
        source = state_layout.field_spaces[state_layout.field_names.index(source_field)]

        def select(state):
            return state_layout.field(state, source_field)
    else:
        if not isinstance(source_field, DiscreteFieldSpace):
            raise TypeError(
                "StateLayout observations require an explicit native source field space."
            )
        source = source_field
        if (
            not isinstance(source.vector_space, ArraySpace)
            or source.vector_space.shape != shape
        ):
            raise ValueError(
                "The source field must cover exactly the native state array."
            )

        def select(state):
            return state

    if source.field_space_id != observations.operator.transfer.source.field_space_id:
        raise ValueError(
            "Observation source field support/representation does not match the model."
        )
    transfer = observations.operator.transfer
    state_space = ArraySpace(shape, dtype=source.vector_space.dtype)
    state_operator = FunctionLinearOperator(
        lambda state: transfer.primal_operator.mv(select(state)),
        source=state_space,
        target=transfer.target.vector_space,
        operator_id=canonical_fingerprint(
            {
                "kind": "geophysical-state-observation",
                "layout": state_layout.layout_id,
                "observation": observations.operator.operator_id,
            }
        ),
    )
    wrapped_prior = _LineagePrior(prior, lineage)

    def forecast(key, state, t0, t1, context):
        value = transition(
            lineage.key(key, "internal_stochastic"), state, t0, t1, context
        )
        array = (
            value.values if isinstance(value, TransitionSample) else jnp.asarray(value)
        )
        if array.shape != shape:
            raise ValueError(
                "Numerical model transition changed the native state layout."
            )
        if isinstance(value, TransitionSample) and (
            value.valid.shape != () or value.status.shape != ()
        ):
            raise ValueError(
                "A single-case transition must return scalar validity and status."
            )
        return value

    kernel = CallableTransitionKernel(
        forecast,
        state_shape=shape,
        process_id=model_id,
        approximation_id=approximation_id,
    )
    observation = GaussianObservationModel(
        lambda state, time, context: state_operator.mv(state),
        lambda time, context: jnp.diag(
            observations.error_variance[context.step_index].reshape(-1)
        ),
        state_shape=shape,
        observation_shape=transfer.target.vector_space.shape,
        observation_id=observations.preparation_id,
    )
    original = observations.sequence
    sequence = ObservationSequence(
        original.times,
        original.values,
        observation_axes=original.observation_axes,
        observation_mask=original.observation_mask,
        case_ids=(lineage.case_id,),
        sequence_id=canonical_fingerprint(
            {"observations": original.sequence_id, "case": lineage.case_id}
        ),
        sensor_id=original.sensor_id,
        discretization_id=original.discretization_id,
    )
    identity = canonical_fingerprint(
        {
            "kind": "geophysical-assimilation",
            "model": model_id,
            "approximation": approximation_id,
            "state_layout": state_layout.layout_id,
            "source_field": source.field_space_id,
            "observations": observations.preparation_id,
            "lineage": lineage.lineage_id,
            "time": model_time.time_id,
            "initial_time": float(initial_time),
            "prior": wrapped_prior.prior_id,
            "args": array_tree_fingerprint(args),
        }
    )
    model = StateSpaceModel(
        wrapped_prior,
        kernel,
        observation,
        model_id=identity,
        parameter_id=lineage.coordinates[_AXIS_KINDS.index("parameter")],
        discretization_id=state_layout.layout_id,
        metadata={
            "geophysical_lineage": lineage.lineage_id,
            "time": model_time.time_id,
            "state_layout": state_layout.layout_id,
        },
    )
    return StateSpaceProblem(
        model, sequence, initial_time=initial_time, problem_id=identity, args=args
    )


class GeophysicalAnalysisInventory(StrictModule, NonTrainableState):
    """A declared linear physical inventory over the packed model state.

    Weights must include all quadrature, area/volume, layer mass and conversion
    factors necessary for the output quantity.unit. Their physical correctness
    is the model owner's responsibility. Nonlinear inventories should instead be
    evaluated directly on native forecast/analysis ensembles by that owner.
    """

    weights: Array
    quantity: GeophysicalQuantity
    name: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    inventory_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        quantity: GeophysicalQuantity,
        weights: ArrayLike,
        state_layout: StateLayout | DiscreteStateLayout,
        /,
    ):
        shape = _layout_shape(state_layout)
        array = np.asarray(weights, dtype=float)
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(quantity, GeophysicalQuantity)
        ):
            raise ValueError("Inventory name and physical quantity are required.")
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(
                "Inventory weights must be finite and match the exact native state shape."
            )
        self.name = name
        self.quantity = quantity
        self.weights = jnp.asarray(array)
        self.layout_id = state_layout.layout_id
        self.state_shape = shape
        self.inventory_id = canonical_fingerprint(
            {
                "kind": "geophysical-analysis-inventory",
                "name": name,
                "quantity": quantity.quantity_id,
                "layout": self.layout_id,
                "weights": array_tree_fingerprint(array),
            }
        )

    def __call__(self, state: ArrayLike, /) -> Array:
        array = jnp.asarray(state)
        rank = len(self.state_shape)
        if array.ndim < rank or (rank and tuple(array.shape[-rank:]) != self.state_shape):
            raise ValueError("Inventory input must end in its native state shape.")
        leading_shape = array.shape[:-rank] if rank else array.shape
        flat = array.reshape(leading_shape + (-1,))
        return contract("...i,i->...", flat, self.weights.reshape(-1))


class GeophysicalAnalysisIncrement(StrictModule, NonTrainableState):
    inventory: GeophysicalAnalysisInventory
    forecast: Array
    analysis: Array
    increment: Array
    mean_increment: Array


def geophysical_analysis_increments(
    result: EnsembleFilterResult,
    inventories: Sequence[GeophysicalAnalysisInventory],
    /,
) -> dict[str, GeophysicalAnalysisIncrement]:
    """Report physical analysis impulses, not fictional model conservation errors.

    Memberwise inventories have shape ``case_shape + (time, member)``. Their
    means expose all physical sources/sinks introduced by analysis. Native ETKF
    anomaly inflation is mean preserving; no conservation correction is applied.
    """
    if not isinstance(result, EnsembleFilterResult):
        raise TypeError("result must be the native EnsembleFilterResult.")
    inventories_ = tuple(inventories)
    if any(
        not isinstance(inventory, GeophysicalAnalysisInventory)
        for inventory in inventories_
    ):
        raise TypeError("inventories must contain GeophysicalAnalysisInventory values.")
    if len({inventory.name for inventory in inventories_}) != len(inventories_):
        raise ValueError("Inventory names must be unique.")
    reports = {}
    for inventory in inventories_:
        if (
            inventory.layout_id != result.problem.model.discretization_id
            or inventory.state_shape != result.state_shape
        ):
            raise ValueError(
                "Analysis inventory and model state layout must match exactly."
            )
        forecast = inventory(result.forecast_ensembles)
        analysis = inventory(result.analysis_ensembles)
        increment = analysis - forecast
        reports[inventory.name] = GeophysicalAnalysisIncrement(
            inventory=inventory,
            forecast=forecast,
            analysis=analysis,
            increment=increment,
            mean_increment=jnp.mean(increment, axis=-1),
        )
    return reports


__all__ = [
    "GeophysicalEnsembleAxis",
    "GeophysicalEnsembleLineage",
    "GeophysicalAnalysisInventory",
    "GeophysicalAnalysisIncrement",
    "prepare_geophysical_assimilation",
    "geophysical_analysis_increments",
]
