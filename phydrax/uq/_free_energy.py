#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Authenticated reduced free-energy observations and qualified estimators."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntFlag
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg._dense_pseudoinverse import (
    factor_pseudoinverse,
)
from ..linalg._policies import RankPolicy
from ._free_energy_kernels import (
    bar_kernel,
    fep_kernel,
    mbar_asymptotic_covariance_kernel,
    mbar_kernel,
    thermodynamic_integration_kernel,
)


_DIMENSIONLESS_UNIT_ID = "1"
WorkKind: TypeAlias = Literal[
    "equilibrium-difference",
    "targeted-map",
    "nonequilibrium-switching",
]
UncertaintyMethod: TypeAlias = Literal["analytic", "block-bootstrap"]


class FreeEnergyStatus(IntFlag):
    """Composable numerical and statistical qualification bits."""

    SUCCESS = 0
    INSUFFICIENT_SAMPLES = 1 << 0
    NONFINITE = 1 << 1
    NONCONVERGED = 1 << 2
    POOR_OVERLAP = 1 << 3
    DISCONNECTED = 1 << 4
    RANK_DEFICIENT = 1 << 5
    INSUFFICIENT_BLOCKS = 1 << 6
    CORRELATION_UNRESOLVED = 1 << 7
    CYCLE_INCONSISTENT = 1 << 8
    UNQUALIFIED_KERNEL = 1 << 9


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string.")
    return value


def _identifiers(
    values: Sequence[str],
    name: str,
    /,
    *,
    count: int | None = None,
    unique: bool = False,
) -> tuple[str, ...]:
    result = tuple(_identifier(value, name) for value in values)
    if count is not None and len(result) != count:
        raise ValueError(f"{name} must contain exactly {count} entries.")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique.")
    return result


def _state_bias_ids(
    values: Sequence[str | None], state_count: int, /
) -> tuple[str | None, ...]:
    raw = tuple(values)
    if not raw:
        return (None,) * state_count
    if len(raw) != state_count:
        raise ValueError("bias_ids must align exactly with state_ids.")
    return tuple(
        None if value is None else _identifier(value, "bias_id") for value in raw
    )


def _floating_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype("float64")
    return array


def _index_array(value: ArrayLike, name: str, /) -> Array:
    host = np.asarray(value)
    if not np.issubdtype(host.dtype, np.integer):
        raise TypeError(f"{name} must contain integer indices.")
    return jnp.asarray(host, dtype=jnp.int32)


def _dimensionless_unit(value: str, /) -> str:
    unit = _identifier(value, "unit_id")
    if unit != _DIMENSIONLESS_UNIT_ID:
        raise ValueError(
            f"Reduced free-energy observations require unit_id={_DIMENSIONLESS_UNIT_ID!r}."
        )
    return unit


def _sampling_qualification(
    qualification_id: str,
    sampling_exact: bool,
    sampling_bias_bound: float,
    /,
) -> tuple[str, bool, float]:
    qualification = _identifier(qualification_id, "qualification_id")
    exact = bool(sampling_exact)
    bias_bound = float(sampling_bias_bound)
    if not math.isfinite(bias_bound) or bias_bound < 0.0:
        raise ValueError("sampling_bias_bound must be finite and non-negative.")
    if exact and bias_bound != 0.0:
        raise ValueError("Exact sampling requires a zero sampling_bias_bound.")
    return qualification, exact, bias_bound


def _validate_sample_identity(
    active: np.ndarray,
    state: np.ndarray,
    chain: np.ndarray,
    draw: np.ndarray,
    repeat: np.ndarray,
    dependence: np.ndarray,
    /,
) -> None:
    active_indices = np.nonzero(active.reshape((-1,)))[0]
    if active_indices.size == 0:
        raise ValueError(
            "A free-energy dataset must contain at least one active observation."
        )
    flattened = tuple(
        np.asarray(value).reshape((-1,))
        for value in (state, chain, draw, repeat, dependence)
    )
    for value in flattened:
        if np.any(value[active_indices] < 0):
            raise ValueError("Active sample identity indices must be non-negative.")
    identities = [
        tuple(int(value[index]) for value in flattened[:4]) for index in active_indices
    ]
    if len(set(identities)) != len(identities):
        raise ValueError(
            "Active observations must have unique (state, chain, draw, repeat) identities."
        )


def _dataset_fingerprint(kind: str, metadata: dict, arrays: dict, /) -> str:
    return canonical_fingerprint(
        {
            "kind": kind,
            "metadata": metadata,
            "arrays": array_tree_fingerprint(arrays),
        }
    )


class ReducedPotentialDataset(StrictModule, NonTrainableState):
    """Dense cross-evaluated, dimensionless reduced potentials.

    Every active configuration is evaluated at every declared state. Inactive
    columns and uncovered cells are canonicalized, while active origin counts
    are always derived rather than accepted as a second authority.
    """

    values: Array
    coverage: Array
    sample_active: Array
    origin_state: Array
    chain_index: Array
    draw_index: Array
    repeat_index: Array
    dependence_group_index: Array
    inverse_temperatures: Array
    state_ids: tuple[str, ...] = eqx.field(static=True)
    potential_ids: tuple[str, ...] = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    producer_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    bias_ids: tuple[str | None, ...] = eqx.field(static=True)
    unit_system_id: str | None = eqx.field(static=True)
    reduced_convention_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        coverage: ArrayLike,
        sample_active: ArrayLike,
        origin_state: ArrayLike,
        chain_index: ArrayLike,
        draw_index: ArrayLike,
        repeat_index: ArrayLike,
        dependence_group_index: ArrayLike,
        /,
        *,
        state_ids: Sequence[str],
        potential_ids: Sequence[str],
        inverse_temperatures: ArrayLike,
        reduced_convention_id: str,
        qualification_id: str,
        sampling_exact: bool,
        sampling_bias_bound: float,
        measure_id: str,
        producer_id: str,
        run_id: str,
        bias_ids: Sequence[str | None] = (),
        unit_system_id: str | None = None,
        unit_id: str,
    ):
        potential = _floating_array(values, "values")
        if potential.ndim != 2 or potential.shape[0] < 2 or potential.shape[1] < 1:
            raise ValueError(
                "values must have shape (states, capacity) with at least two states."
            )
        state_count, capacity = potential.shape
        covered = jnp.asarray(coverage, dtype=jnp.bool_)
        active = jnp.asarray(sample_active, dtype=jnp.bool_)
        origin = _index_array(origin_state, "origin_state")
        chain = _index_array(chain_index, "chain_index")
        draw = _index_array(draw_index, "draw_index")
        repeat = _index_array(repeat_index, "repeat_index")
        dependence = _index_array(dependence_group_index, "dependence_group_index")
        beta = _floating_array(inverse_temperatures, "inverse_temperatures")
        if beta.shape != (state_count,) or not bool(
            jnp.all(jnp.isfinite(beta) & (beta > 0.0))
        ):
            raise ValueError(
                "inverse_temperatures must contain one finite positive beta per state."
            )
        if covered.shape != potential.shape:
            raise ValueError("coverage must have the same shape as values.")
        for name, array in (
            ("sample_active", active),
            ("origin_state", origin),
            ("chain_index", chain),
            ("draw_index", draw),
            ("repeat_index", repeat),
            ("dependence_group_index", dependence),
        ):
            if array.shape != (capacity,):
                raise ValueError(f"{name} must have shape (capacity,).")
        host_active = np.asarray(active)
        host_origin = np.asarray(origin)
        if np.any(
            (host_origin[host_active] < 0) | (host_origin[host_active] >= state_count)
        ):
            raise ValueError(
                "Every active configuration must name one valid origin state."
            )
        if not bool(jnp.all(jnp.where(active[None, :], covered, True))):
            raise ValueError(
                "Dense MBAR coverage requires every active configuration at every state."
            )
        if not bool(jnp.all(jnp.where(active[None, :], jnp.isfinite(potential), True))):
            raise ValueError("Covered reduced potentials must be finite.")
        _validate_sample_identity(
            host_active,
            host_origin,
            np.asarray(chain),
            np.asarray(draw),
            np.asarray(repeat),
            np.asarray(dependence),
        )
        states = _identifiers(state_ids, "state_id", count=state_count, unique=True)
        potentials = _identifiers(potential_ids, "potential_id", count=state_count)
        biases = _state_bias_ids(bias_ids, state_count)
        measure = _identifier(measure_id, "measure_id")
        producer = _identifier(producer_id, "producer_id")
        unit_system = (
            None
            if unit_system_id is None
            else _identifier(unit_system_id, "unit_system_id")
        )
        convention = _identifier(reduced_convention_id, "reduced_convention_id")
        qualification = _identifier(qualification_id, "qualification_id")
        exact = bool(sampling_exact)
        bias_bound = float(sampling_bias_bound)
        if not math.isfinite(bias_bound) or bias_bound < 0.0:
            raise ValueError("sampling_bias_bound must be finite and non-negative.")
        if exact and bias_bound != 0.0:
            raise ValueError("Exact sampling requires a zero sampling_bias_bound.")
        run = _identifier(run_id, "run_id")
        unit = _dimensionless_unit(unit_id)
        canonical_coverage = covered & active[None, :]
        canonical_values = jnp.where(canonical_coverage, potential, 0.0)
        canonical_origin = jnp.where(active, origin, -1)
        canonical_chain = jnp.where(active, chain, -1)
        canonical_draw = jnp.where(active, draw, -1)
        canonical_repeat = jnp.where(active, repeat, -1)
        canonical_dependence = jnp.where(active, dependence, -1)
        metadata = {
            "state_ids": states,
            "potential_ids": potentials,
            "measure_id": measure,
            "producer_id": producer,
            "run_id": run,
            "bias_ids": biases,
            "unit_system_id": unit_system,
            "unit_id": unit,
            "reduced_convention_id": convention,
            "qualification_id": qualification,
            "sampling_exact": exact,
            "sampling_bias_bound": bias_bound.hex(),
        }
        arrays = {
            "inverse_temperatures": beta,
            "values": canonical_values,
            "coverage": canonical_coverage,
            "sample_active": active,
            "origin_state": canonical_origin,
            "chain_index": canonical_chain,
            "draw_index": canonical_draw,
            "repeat_index": canonical_repeat,
            "dependence_group_index": canonical_dependence,
        }
        self.values = canonical_values
        self.coverage = canonical_coverage
        self.sample_active = active
        self.origin_state = canonical_origin
        self.chain_index = canonical_chain
        self.draw_index = canonical_draw
        self.repeat_index = canonical_repeat
        self.dependence_group_index = canonical_dependence
        self.inverse_temperatures = beta
        self.state_ids = states
        self.potential_ids = potentials
        self.measure_id = measure
        self.producer_id = producer
        self.run_id = run
        self.bias_ids = biases
        self.unit_id = unit
        self.unit_system_id = unit_system
        self.reduced_convention_id = convention
        self.qualification_id = qualification
        self.sampling_exact = exact
        self.sampling_bias_bound = bias_bound
        self.dataset_id = _dataset_fingerprint(
            "reduced-potential-dataset", metadata, arrays
        )

    @property
    def state_counts(self) -> Array:
        safe_origin = jnp.where(self.sample_active, self.origin_state, 0)
        return jnp.bincount(
            safe_origin,
            weights=self.sample_active.astype(jnp.int32),
            length=len(self.state_ids),
        ).astype(jnp.int32)

    @property
    def sample_count(self) -> Array:
        return jnp.sum(self.sample_active, dtype=jnp.int32)


class ReducedWorkDataset(StrictModule, NonTrainableState):
    """Authenticated directed reduced work for one ordered pair of states."""

    values: Array
    coverage: Array
    sample_active: Array
    source_state: Array
    destination_state: Array
    chain_index: Array
    draw_index: Array
    repeat_index: Array
    dependence_group_index: Array
    state_ids: tuple[str, str] = eqx.field(static=True)
    potential_ids: tuple[str, str] = eqx.field(static=True)
    measure_ids: tuple[str, str] = eqx.field(static=True)
    producer_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    work_id: str = eqx.field(static=True)
    work_kind: WorkKind = eqx.field(static=True)
    mapping_id: str | None = eqx.field(static=True)
    bias_ids: tuple[str | None, str | None] = eqx.field(static=True)
    unit_system_id: str | None = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        coverage: ArrayLike,
        sample_active: ArrayLike,
        source_state: ArrayLike,
        destination_state: ArrayLike,
        chain_index: ArrayLike,
        draw_index: ArrayLike,
        repeat_index: ArrayLike,
        dependence_group_index: ArrayLike,
        /,
        *,
        state_ids: Sequence[str],
        potential_ids: Sequence[str],
        measure_ids: Sequence[str],
        producer_id: str,
        run_id: str,
        work_id: str,
        work_kind: WorkKind,
        qualification_id: str,
        sampling_exact: bool,
        sampling_bias_bound: float,
        mapping_id: str | None = None,
        bias_ids: Sequence[str | None] = (),
        unit_system_id: str | None = None,
        unit_id: str,
    ):
        work = _floating_array(values, "values")
        if work.ndim != 1 or work.size < 1:
            raise ValueError("values must be a non-empty capacity vector.")
        capacity = work.size
        covered = jnp.asarray(coverage, dtype=jnp.bool_)
        active = jnp.asarray(sample_active, dtype=jnp.bool_)
        source = _index_array(source_state, "source_state")
        destination = _index_array(destination_state, "destination_state")
        chain = _index_array(chain_index, "chain_index")
        draw = _index_array(draw_index, "draw_index")
        repeat = _index_array(repeat_index, "repeat_index")
        dependence = _index_array(dependence_group_index, "dependence_group_index")
        for name, array in (
            ("coverage", covered),
            ("sample_active", active),
            ("source_state", source),
            ("destination_state", destination),
            ("chain_index", chain),
            ("draw_index", draw),
            ("repeat_index", repeat),
            ("dependence_group_index", dependence),
        ):
            if array.shape != (capacity,):
                raise ValueError(f"{name} must have shape (capacity,).")
        host_active = np.asarray(active)
        host_source = np.asarray(source)
        host_destination = np.asarray(destination)
        directed = ((host_source == 0) & (host_destination == 1)) | (
            (host_source == 1) & (host_destination == 0)
        )
        if not np.all(directed[host_active]):
            raise ValueError(
                "Active work must be explicitly oriented between states 0 and 1."
            )
        canonical_covered = active & covered
        if not bool(jnp.all(jnp.where(canonical_covered, jnp.isfinite(work), True))):
            raise ValueError("Covered active reduced work values must be finite.")
        _validate_sample_identity(
            host_active,
            host_source,
            np.asarray(chain),
            np.asarray(draw),
            np.asarray(repeat),
            np.asarray(dependence),
        )
        states_ = _identifiers(state_ids, "state_id", count=2, unique=True)
        states = (states_[0], states_[1])
        potentials_ = _identifiers(potential_ids, "potential_id", count=2)
        potentials = (potentials_[0], potentials_[1])
        measures_ = _identifiers(measure_ids, "measure_id", count=2)
        measures = (measures_[0], measures_[1])
        if work_kind not in (
            "equilibrium-difference",
            "targeted-map",
            "nonequilibrium-switching",
        ):
            raise ValueError("work_kind is not a supported reduced-work definition.")
        mapping = None if mapping_id is None else _identifier(mapping_id, "mapping_id")
        if measures[0] != measures[1] and mapping is None:
            raise ValueError(
                "Work between different measures requires an authenticated mapping_id."
            )
        if work_kind == "targeted-map" and mapping is None:
            raise ValueError("Targeted-map work requires mapping_id.")
        producer = _identifier(producer_id, "producer_id")
        run = _identifier(run_id, "run_id")
        identity = _identifier(work_id, "work_id")
        biases_ = _state_bias_ids(bias_ids, 2)
        biases = (biases_[0], biases_[1])
        unit_system = (
            None
            if unit_system_id is None
            else _identifier(unit_system_id, "unit_system_id")
        )
        qualification, exact, bias_bound = _sampling_qualification(
            qualification_id,
            sampling_exact,
            sampling_bias_bound,
        )
        unit = _dimensionless_unit(unit_id)
        canonical_coverage = covered & active
        canonical_values = jnp.where(canonical_coverage, work, 0.0)
        canonical_source = jnp.where(active, source, -1)
        canonical_destination = jnp.where(active, destination, -1)
        canonical_chain = jnp.where(active, chain, -1)
        canonical_draw = jnp.where(active, draw, -1)
        canonical_repeat = jnp.where(active, repeat, -1)
        canonical_dependence = jnp.where(active, dependence, -1)
        metadata = {
            "state_ids": states,
            "potential_ids": potentials,
            "measure_ids": measures,
            "producer_id": producer,
            "run_id": run,
            "work_id": identity,
            "work_kind": work_kind,
            "mapping_id": mapping,
            "bias_ids": biases,
            "unit_system_id": unit_system,
            "unit_id": unit,
            "qualification_id": qualification,
            "sampling_exact": exact,
            "sampling_bias_bound": bias_bound.hex(),
        }
        arrays = {
            "values": canonical_values,
            "coverage": canonical_coverage,
            "sample_active": active,
            "source_state": canonical_source,
            "destination_state": canonical_destination,
            "chain_index": canonical_chain,
            "draw_index": canonical_draw,
            "repeat_index": canonical_repeat,
            "dependence_group_index": canonical_dependence,
        }
        self.values = canonical_values
        self.coverage = canonical_coverage
        self.sample_active = active
        self.source_state = canonical_source
        self.destination_state = canonical_destination
        self.chain_index = canonical_chain
        self.draw_index = canonical_draw
        self.repeat_index = canonical_repeat
        self.dependence_group_index = canonical_dependence
        self.state_ids = states
        self.measure_ids = measures
        self.potential_ids = potentials
        self.producer_id = producer
        self.run_id = run
        self.work_id = identity
        self.work_kind = work_kind
        self.mapping_id = mapping
        self.bias_ids = biases
        self.unit_system_id = unit_system
        self.unit_id = unit
        self.qualification_id = qualification
        self.sampling_exact = exact
        self.sampling_bias_bound = bias_bound
        self.dataset_id = _dataset_fingerprint("reduced-work-dataset", metadata, arrays)

    @property
    def direction_counts(self) -> Array:
        forward = self.sample_active & (self.source_state == 0)
        reverse = self.sample_active & (self.source_state == 1)
        return jnp.asarray(
            [jnp.sum(forward, dtype=jnp.int32), jnp.sum(reverse, dtype=jnp.int32)]
        )


class ThermodynamicDerivativeDataset(StrictModule, NonTrainableState):
    """Raw complete reduced derivatives along one ordered control path."""

    values: Array
    coverage: Array
    sample_active: Array
    chain_index: Array
    draw_index: Array
    repeat_index: Array
    dependence_group_index: Array
    path_parameter: Array
    quadrature_weights: Array
    state_ids: tuple[str, ...] = eqx.field(static=True)
    potential_ids: tuple[str, ...] = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    producer_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    derivative_id: str = eqx.field(static=True)
    control_path_id: str = eqx.field(static=True)
    quadrature_rule: str = eqx.field(static=True)
    bias_ids: tuple[str | None, ...] = eqx.field(static=True)
    unit_system_id: str | None = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        coverage: ArrayLike,
        sample_active: ArrayLike,
        chain_index: ArrayLike,
        draw_index: ArrayLike,
        repeat_index: ArrayLike,
        dependence_group_index: ArrayLike,
        path_parameter: ArrayLike,
        /,
        *,
        state_ids: Sequence[str],
        potential_ids: Sequence[str],
        measure_id: str,
        producer_id: str,
        run_id: str,
        derivative_id: str,
        control_path_id: str,
        qualification_id: str,
        sampling_exact: bool,
        sampling_bias_bound: float,
        quadrature_rule: str = "trapezoid",
        bias_ids: Sequence[str | None] = (),
        unit_system_id: str | None = None,
        unit_id: str,
    ):
        derivative = _floating_array(values, "values")
        if derivative.ndim != 2 or derivative.shape[0] < 2 or derivative.shape[1] < 1:
            raise ValueError(
                "values must have shape (states, capacity) with at least two states."
            )
        state_count, capacity = derivative.shape
        covered = jnp.asarray(coverage, dtype=jnp.bool_)
        active = jnp.asarray(sample_active, dtype=jnp.bool_)
        chain = _index_array(chain_index, "chain_index")
        draw = _index_array(draw_index, "draw_index")
        repeat = _index_array(repeat_index, "repeat_index")
        dependence = _index_array(dependence_group_index, "dependence_group_index")
        parameter = _floating_array(path_parameter, "path_parameter")
        for name, array in (
            ("coverage", covered),
            ("sample_active", active),
            ("chain_index", chain),
            ("draw_index", draw),
            ("repeat_index", repeat),
            ("dependence_group_index", dependence),
        ):
            if array.shape != derivative.shape:
                raise ValueError(f"{name} must have shape (states, capacity).")
        if parameter.shape != (state_count,):
            raise ValueError("path_parameter must contain one point per state.")
        if not bool(jnp.all(jnp.isfinite(parameter))) or bool(
            jnp.any(jnp.diff(parameter) <= 0.0)
        ):
            raise ValueError("path_parameter must be finite and strictly increasing.")
        if not bool(jnp.all(jnp.where(active, covered, True))):
            raise ValueError("Every active derivative observation must be covered.")
        if not bool(jnp.all(jnp.where(active, jnp.isfinite(derivative), True))):
            raise ValueError("Active complete path derivatives must be finite.")
        counts = np.sum(np.asarray(active), axis=1)
        if np.any(counts == 0):
            raise ValueError(
                "Thermodynamic integration requires observations at every state."
            )
        state_matrix = np.broadcast_to(np.arange(state_count)[:, None], derivative.shape)
        _validate_sample_identity(
            np.asarray(active),
            state_matrix,
            np.asarray(chain),
            np.asarray(draw),
            np.asarray(repeat),
            np.asarray(dependence),
        )
        if quadrature_rule != "trapezoid":
            raise ValueError("Only the explicit trapezoid quadrature rule is supported.")
        increments = jnp.diff(parameter)
        weights = jnp.zeros_like(parameter)
        weights = weights.at[:-1].add(0.5 * increments)
        weights = weights.at[1:].add(0.5 * increments)
        states = _identifiers(state_ids, "state_id", count=state_count, unique=True)
        potentials = _identifiers(potential_ids, "potential_id", count=state_count)
        measure = _identifier(measure_id, "measure_id")
        producer = _identifier(producer_id, "producer_id")
        run = _identifier(run_id, "run_id")
        derivative_identity = _identifier(derivative_id, "derivative_id")
        path_identity = _identifier(control_path_id, "control_path_id")
        biases = _state_bias_ids(bias_ids, state_count)
        unit_system = (
            None
            if unit_system_id is None
            else _identifier(unit_system_id, "unit_system_id")
        )
        qualification, exact, bias_bound = _sampling_qualification(
            qualification_id,
            sampling_exact,
            sampling_bias_bound,
        )
        unit = _dimensionless_unit(unit_id)
        canonical_coverage = covered & active
        canonical_values = jnp.where(canonical_coverage, derivative, 0.0)
        canonical_chain = jnp.where(active, chain, -1)
        canonical_draw = jnp.where(active, draw, -1)
        canonical_repeat = jnp.where(active, repeat, -1)
        canonical_dependence = jnp.where(active, dependence, -1)
        metadata = {
            "state_ids": states,
            "potential_ids": potentials,
            "measure_id": measure,
            "producer_id": producer,
            "run_id": run,
            "derivative_id": derivative_identity,
            "control_path_id": path_identity,
            "derivative_definition": "complete-reduced-control-path-derivative",
            "quadrature_rule": quadrature_rule,
            "bias_ids": biases,
            "unit_system_id": unit_system,
            "unit_id": unit,
            "qualification_id": qualification,
            "sampling_exact": exact,
            "sampling_bias_bound": bias_bound.hex(),
        }
        arrays = {
            "values": canonical_values,
            "coverage": canonical_coverage,
            "sample_active": active,
            "chain_index": canonical_chain,
            "draw_index": canonical_draw,
            "repeat_index": canonical_repeat,
            "dependence_group_index": canonical_dependence,
            "path_parameter": parameter,
            "quadrature_weights": weights,
        }
        self.values = canonical_values
        self.coverage = canonical_coverage
        self.sample_active = active
        self.chain_index = canonical_chain
        self.draw_index = canonical_draw
        self.repeat_index = canonical_repeat
        self.dependence_group_index = canonical_dependence
        self.path_parameter = parameter
        self.quadrature_weights = weights
        self.state_ids = states
        self.potential_ids = potentials
        self.measure_id = measure
        self.producer_id = producer
        self.run_id = run
        self.derivative_id = derivative_identity
        self.control_path_id = path_identity
        self.quadrature_rule = quadrature_rule
        self.bias_ids = biases
        self.unit_system_id = unit_system
        self.unit_id = unit
        self.qualification_id = qualification
        self.sampling_exact = exact
        self.sampling_bias_bound = bias_bound
        self.dataset_id = _dataset_fingerprint(
            "thermodynamic-derivative-dataset", metadata, arrays
        )

    @property
    def state_counts(self) -> Array:
        return jnp.sum(self.sample_active, axis=1, dtype=jnp.int32)


FreeEnergyDataset: TypeAlias = (
    ReducedPotentialDataset | ReducedWorkDataset | ThermodynamicDerivativeDataset
)


class FreeEnergySelectionPlan(StrictModule, NonTrainableState):
    """Deterministic burn-in, thinning, correlation and block policy."""

    burn_in: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    block_length: int | None = eqx.field(static=True)
    maximum_correlation_lag: int | None = eqx.field(static=True)
    minimum_samples_per_state: int = eqx.field(static=True)
    minimum_blocks: int = eqx.field(static=True)
    minimum_overlap: float = eqx.field(static=True)
    uncertainty_method: UncertaintyMethod = eqx.field(static=True)
    bootstrap_replicates: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        burn_in: int = 0,
        stride: int = 1,
        block_length: int | None = None,
        maximum_correlation_lag: int | None = None,
        minimum_samples_per_state: int = 2,
        minimum_blocks: int = 2,
        minimum_overlap: float = 1.0e-3,
        uncertainty_method: UncertaintyMethod = "analytic",
        bootstrap_replicates: int = 0,
    ):
        burn = int(burn_in)
        stride_ = int(stride)
        block = None if block_length is None else int(block_length)
        lag = None if maximum_correlation_lag is None else int(maximum_correlation_lag)
        minimum_samples = int(minimum_samples_per_state)
        blocks = int(minimum_blocks)
        overlap = float(minimum_overlap)
        replicates = int(bootstrap_replicates)
        if burn < 0 or stride_ < 1 or (block is not None and block < 1):
            raise ValueError("burn_in, stride and block_length are invalid.")
        if lag is not None and lag < 1:
            raise ValueError("maximum_correlation_lag must be positive when provided.")
        if minimum_samples < 1 or blocks < 1:
            raise ValueError("Minimum samples and blocks must be positive.")
        if not math.isfinite(overlap) or overlap < 0.0 or overlap >= 1.0:
            raise ValueError("minimum_overlap must be finite and in [0, 1).")
        if uncertainty_method not in ("analytic", "block-bootstrap"):
            raise ValueError(
                "uncertainty_method must be 'analytic' or 'block-bootstrap'."
            )
        if uncertainty_method == "block-bootstrap" and replicates < 2:
            raise ValueError("Block bootstrap requires at least two replicates.")
        if uncertainty_method == "analytic" and replicates != 0:
            raise ValueError(
                "bootstrap_replicates applies only to block-bootstrap uncertainty."
            )
        self.burn_in = burn
        self.stride = stride_
        self.block_length = block
        self.maximum_correlation_lag = lag
        self.minimum_samples_per_state = minimum_samples
        self.minimum_blocks = blocks
        self.minimum_overlap = overlap
        self.uncertainty_method = uncertainty_method
        self.bootstrap_replicates = replicates
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-energy-selection-plan",
                "burn_in": burn,
                "stride": stride_,
                "block_length": block,
                "maximum_correlation_lag": lag,
                "minimum_samples_per_state": minimum_samples,
                "minimum_blocks": blocks,
                "minimum_overlap": overlap.hex(),
                "uncertainty_method": uncertainty_method,
                "bootstrap_replicates": replicates,
                "correlation_window": "geyer-initial-positive-sequence",
                "resampling": "synchronous-repeat-dependence-group-blocks",
            }
        )

    def select(self, dataset: FreeEnergyDataset, /) -> "FreeEnergySelectionEvidence":
        return _select_free_energy_dataset(dataset, self)


class FreeEnergySelectionEvidence(StrictModule, NonTrainableState):
    """Dataset-bound retained observations and synchronous block metadata."""

    retained: Array
    block_index: Array
    block_group_index: Array
    stratum_index: Array
    raw_counts: Array
    retained_counts: Array
    block_counts: Array
    statistical_inefficiency: Array
    correlation_resolved: Array
    correlation_effective_sample_size: Array
    plan: FreeEnergySelectionPlan
    resolved_block_length: int = eqx.field(static=True)
    block_count: int = eqx.field(static=True)
    dataset_kind: str = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)

    def __init__(
        self,
        retained: ArrayLike,
        block_index: ArrayLike,
        block_group_index: ArrayLike,
        stratum_index: ArrayLike,
        raw_counts: ArrayLike,
        retained_counts: ArrayLike,
        block_counts: ArrayLike,
        statistical_inefficiency: ArrayLike,
        correlation_resolved: ArrayLike,
        correlation_effective_sample_size: ArrayLike,
        plan: FreeEnergySelectionPlan,
        /,
        *,
        resolved_block_length: int,
        block_count: int,
        dataset_kind: str,
        dataset_id: str,
    ):
        if not isinstance(plan, FreeEnergySelectionPlan):
            raise TypeError("plan must be FreeEnergySelectionPlan.")
        kept = jnp.asarray(retained, dtype=jnp.bool_)
        block = _index_array(block_index, "block_index")
        group = _index_array(block_group_index, "block_group_index")
        stratum = _index_array(stratum_index, "stratum_index")
        if (
            block.shape != kept.shape
            or group.shape != kept.shape
            or stratum.shape != kept.shape
        ):
            raise ValueError(
                "Selection masks and indices must share one observation shape."
            )
        raw = _index_array(raw_counts, "raw_counts")
        selected = _index_array(retained_counts, "retained_counts")
        blocks = _index_array(block_counts, "block_counts")
        inefficiency = _floating_array(
            statistical_inefficiency, "statistical_inefficiency"
        )
        resolved_correlation = jnp.asarray(correlation_resolved, dtype=jnp.bool_)
        effective = _floating_array(
            correlation_effective_sample_size, "correlation_effective_sample_size"
        )
        if not (
            raw.ndim == 1
            and selected.shape == raw.shape
            and blocks.shape == raw.shape
            and inefficiency.shape == raw.shape
            and resolved_correlation.shape == raw.shape
            and effective.shape == raw.shape
        ):
            raise ValueError(
                "Selection count and correlation summaries must align by stratum."
            )
        if bool(jnp.any(raw < selected)) or bool(jnp.any(selected < 0)):
            raise ValueError("Selection counts are inconsistent.")
        if not bool(jnp.all(jnp.isfinite(inefficiency) & (inefficiency >= 1.0))):
            raise ValueError(
                "Statistical inefficiencies must be finite and at least one."
            )
        resolved = int(resolved_block_length)
        total_blocks = int(block_count)
        if resolved < 1 or total_blocks < 0:
            raise ValueError("Resolved block metadata is invalid.")
        if bool(jnp.any(jnp.where(kept, block < 0, block != -1))):
            raise ValueError(
                "Retained observations must have a block; excluded entries use -1."
            )
        if bool(jnp.any(jnp.where(kept, group < 0, group != -1))):
            raise ValueError(
                "Retained observations must have a group; excluded entries use -1."
            )
        host_kept = np.asarray(kept).reshape((-1,))
        host_block = np.asarray(block).reshape((-1,))
        host_group = np.asarray(group).reshape((-1,))
        host_stratum = np.asarray(stratum).reshape((-1,))
        stratum_count = raw.size
        if np.any(
            (host_stratum[host_kept] < 0) | (host_stratum[host_kept] >= stratum_count)
        ) or np.any(host_stratum[~host_kept] != -1):
            raise ValueError("Selection stratum indices are not canonical.")
        unique_blocks = sorted(set(int(value) for value in host_block[host_kept]))
        if unique_blocks != list(range(total_blocks)):
            raise ValueError("Selection block indices must be contiguous and exhaustive.")
        unique_groups = sorted(set(int(value) for value in host_group[host_kept]))
        if unique_groups and unique_groups != list(range(unique_groups[-1] + 1)):
            raise ValueError("Selection group indices must be contiguous.")
        derived_selected = np.bincount(
            host_stratum[host_kept], minlength=stratum_count
        ).astype(np.int32)
        derived_blocks = np.asarray(
            [
                len(set(host_block[host_kept & (host_stratum == index)].tolist()))
                for index in range(stratum_count)
            ],
            dtype=np.int32,
        )
        if not np.array_equal(
            derived_selected, np.asarray(selected)
        ) or not np.array_equal(derived_blocks, np.asarray(blocks)):
            raise ValueError(
                "Selection counts do not match retained observation metadata."
            )
        if not np.allclose(
            np.asarray(effective),
            np.asarray(selected) / np.asarray(inefficiency),
            rtol=1.0e-6,
            atol=0.0,
        ):
            raise ValueError("Correlation-adjusted sample counts are inconsistent.")
        kind = _identifier(dataset_kind, "dataset_kind")
        identity = _identifier(dataset_id, "dataset_id")
        arrays = {
            "retained": kept,
            "block_index": block,
            "block_group_index": group,
            "stratum_index": stratum,
            "raw_counts": raw,
            "correlation_resolved": resolved_correlation,
            "retained_counts": selected,
            "block_counts": blocks,
            "statistical_inefficiency": inefficiency,
            "correlation_effective_sample_size": effective,
        }
        selection_id = _dataset_fingerprint(
            "free-energy-selection-evidence",
            {
                "dataset_id": identity,
                "plan_id": plan.plan_id,
                "dataset_kind": kind,
                "resolved_block_length": resolved,
                "block_count": total_blocks,
            },
            arrays,
        )
        self.retained = kept
        self.block_index = block
        self.block_group_index = group
        self.stratum_index = stratum
        self.raw_counts = raw
        self.correlation_resolved = resolved_correlation
        self.retained_counts = selected
        self.block_counts = blocks
        self.statistical_inefficiency = inefficiency
        self.correlation_effective_sample_size = effective
        self.plan = plan
        self.resolved_block_length = resolved
        self.block_count = total_blocks
        self.dataset_kind = kind
        self.observation_shape = tuple(kept.shape)
        self.dataset_id = identity
        self.selection_id = selection_id


class FreeEnergyResult(StrictModule, NonTrainableState):
    """Gauge-fixed estimate with full covariance and fail-closed evidence."""

    free_energies: Array
    covariance: Array
    differences: Array
    standard_errors: Array
    overlap: Array
    connectivity: Array
    raw_effective_sample_size: Array
    effective_sample_size: Array
    influence_values: Array
    iterations: Array
    solver_residual: Array
    covariance_rank: Array
    numerical_status: Array
    statistical_status: Array
    state_ids: tuple[str, ...] = eqx.field(static=True)
    gauge_state_id: str = eqx.field(static=True)
    method: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)

    def __init__(
        self,
        free_energies: ArrayLike,
        covariance: ArrayLike,
        overlap: ArrayLike,
        connectivity: ArrayLike,
        raw_effective_sample_size: ArrayLike,
        effective_sample_size: ArrayLike,
        influence_values: ArrayLike,
        iterations: ArrayLike,
        solver_residual: ArrayLike,
        covariance_rank: ArrayLike,
        numerical_status: int | FreeEnergyStatus | ArrayLike,
        statistical_status: int | FreeEnergyStatus | ArrayLike,
        /,
        *,
        state_ids: Sequence[str],
        gauge_state_id: str,
        method: str,
        dataset_id: str,
        selection_id: str,
    ):
        free = _floating_array(free_energies, "free_energies").reshape((-1,))
        states = _identifiers(state_ids, "state_id", count=free.size, unique=True)
        if free.size < 2:
            raise ValueError("A free-energy result requires at least two states.")
        gauge = _identifier(gauge_state_id, "gauge_state_id")
        if gauge not in states:
            raise ValueError("gauge_state_id must identify one result state.")
        gauge_index = states.index(gauge)
        covariance_ = _floating_array(covariance, "covariance")
        overlap_ = _floating_array(overlap, "overlap")
        connectivity_ = jnp.asarray(connectivity, dtype=jnp.bool_)
        raw_ess = _floating_array(raw_effective_sample_size, "raw_effective_sample_size")
        ess = _floating_array(effective_sample_size, "effective_sample_size")
        influence = _floating_array(influence_values, "influence_values")
        shape = (free.size, free.size)
        if (
            covariance_.shape != shape
            or overlap_.shape != shape
            or connectivity_.shape != shape
        ):
            raise ValueError(
                "Covariance, overlap and connectivity must be square over states."
            )
        if raw_ess.shape != free.shape or ess.shape != free.shape:
            raise ValueError("Effective sample sizes must contain one value per state.")
        if influence.ndim != 2 or influence.shape[0] != free.size:
            raise ValueError(
                "influence_values must have shape (states, influence_modes)."
            )
        arrays = (free, covariance_, overlap_, raw_ess, ess, influence)
        if not all(bool(jnp.all(jnp.isfinite(value))) for value in arrays):
            raise ValueError(
                "Free-energy result arrays must be finite; failures use status bits."
            )
        covariance_ = 0.5 * (covariance_ + covariance_.T)
        host_covariance = np.asarray(covariance_)
        scale = max(float(np.max(np.abs(host_covariance), initial=0.0)), 1.0)
        tolerance = 256.0 * np.finfo(host_covariance.dtype).eps * scale
        if float(np.min(np.linalg.eigvalsh(host_covariance))) < -tolerance:
            raise ValueError("covariance must be positive semidefinite.")
        if not np.allclose(
            np.asarray(influence @ influence.T),
            host_covariance,
            atol=tolerance,
            rtol=1e-6,
        ):
            raise ValueError("influence_values must factor the full covariance.")
        if abs(float(np.asarray(free[gauge_index]))) > tolerance:
            raise ValueError("free_energies must use the declared zero gauge.")
        if np.max(np.abs(host_covariance[gauge_index])) > tolerance:
            raise ValueError("The covariance must be zero along the fixed gauge state.")
        if bool(jnp.any(raw_ess < 0.0)) or bool(jnp.any(ess < 0.0)):
            raise ValueError("Effective sample sizes must be non-negative.")
        if bool(jnp.any(overlap_ < 0.0)):
            raise ValueError("Overlap evidence must be non-negative.")
        host_connectivity = np.asarray(connectivity_)
        if not np.array_equal(host_connectivity, host_connectivity.T) or not np.all(
            np.diag(host_connectivity)
        ):
            raise ValueError("Connectivity evidence must be symmetric and reflexive.")
        iteration_array = _index_array(iterations, "iterations").reshape(())
        residual_array = _floating_array(solver_residual, "solver_residual").reshape(())
        rank_array = _index_array(covariance_rank, "covariance_rank").reshape(())
        if int(np.asarray(iteration_array)) < 0:
            raise ValueError("iterations must be non-negative.")
        if not bool(jnp.isfinite(residual_array) & (residual_array >= 0.0)):
            raise ValueError("solver_residual must be finite and non-negative.")
        rank_value = int(np.asarray(rank_array))
        if rank_value < 0 or rank_value > free.size - 1:
            raise ValueError("covariance_rank is incompatible with the fixed gauge.")
        method_ = _identifier(method, "method")
        dataset = _identifier(dataset_id, "dataset_id")
        selection = _identifier(selection_id, "selection_id")
        numerical_value = int(np.asarray(numerical_status))
        statistical_value = int(np.asarray(statistical_status))
        known_status = sum(int(value) for value in FreeEnergyStatus if value)
        if (
            numerical_value < 0
            or statistical_value < 0
            or numerical_value & ~known_status
            or statistical_value & ~known_status
        ):
            raise ValueError("Free-energy status contains unknown bits.")
        numerical = jnp.asarray(numerical_value, dtype=jnp.int32)
        statistical = jnp.asarray(statistical_value, dtype=jnp.int32)
        difference = free[None, :] - free[:, None]
        diagonal = jnp.diag(covariance_)
        difference_variance = jnp.maximum(
            diagonal[:, None] + diagonal[None, :] - 2.0 * covariance_, 0.0
        )
        errors = jnp.sqrt(difference_variance)
        analysis_id = _dataset_fingerprint(
            "free-energy-result",
            {
                "state_ids": states,
                "gauge_state_id": gauge,
                "method": method_,
                "dataset_id": dataset,
                "selection_id": selection,
                "numerical_status": int(np.asarray(numerical)),
                "statistical_status": int(np.asarray(statistical)),
            },
            {
                "free_energies": free,
                "covariance": covariance_,
                "overlap": overlap_,
                "connectivity": connectivity_,
                "raw_effective_sample_size": raw_ess,
                "effective_sample_size": ess,
                "influence_values": influence,
                "iterations": iteration_array,
                "solver_residual": residual_array,
                "covariance_rank": rank_array,
            },
        )
        self.free_energies = free
        self.covariance = covariance_
        self.differences = difference
        self.standard_errors = errors
        self.overlap = overlap_
        self.connectivity = connectivity_
        self.raw_effective_sample_size = raw_ess
        self.effective_sample_size = ess
        self.influence_values = influence
        self.iterations = iteration_array
        self.solver_residual = residual_array
        self.covariance_rank = rank_array
        self.numerical_status = numerical
        self.statistical_status = statistical
        self.state_ids = states
        self.gauge_state_id = gauge
        self.method = method_
        self.dataset_id = dataset
        self.selection_id = selection
        self.analysis_id = analysis_id

    @property
    def successful(self) -> Array:
        return (self.numerical_status == int(FreeEnergyStatus.SUCCESS)) & (
            self.statistical_status == int(FreeEnergyStatus.SUCCESS)
        )


def _dataset_observations(dataset: FreeEnergyDataset, /):
    if isinstance(dataset, ReducedPotentialDataset):
        safe_origin = jnp.clip(dataset.origin_state, 0, len(dataset.state_ids) - 1)
        observable = dataset.values[safe_origin, jnp.arange(dataset.values.shape[1])]
        return (
            observable,
            dataset.sample_active,
            dataset.origin_state,
            dataset.chain_index,
            dataset.draw_index,
            dataset.repeat_index,
            dataset.dependence_group_index,
            "reduced-potential",
            len(dataset.state_ids),
        )
    if isinstance(dataset, ReducedWorkDataset):
        return (
            dataset.values,
            dataset.sample_active,
            dataset.source_state,
            dataset.chain_index,
            dataset.draw_index,
            dataset.repeat_index,
            dataset.dependence_group_index,
            "reduced-work",
            2,
        )
    if isinstance(dataset, ThermodynamicDerivativeDataset):
        state_count, capacity = dataset.values.shape
        stratum = jnp.broadcast_to(
            jnp.arange(state_count, dtype=jnp.int32)[:, None], (state_count, capacity)
        )
        return (
            dataset.values,
            dataset.sample_active,
            stratum,
            dataset.chain_index,
            dataset.draw_index,
            dataset.repeat_index,
            dataset.dependence_group_index,
            "thermodynamic-derivative",
            state_count,
        )
    raise TypeError("dataset must be an authenticated free-energy dataset.")


def _correlation_inefficiency(
    values: np.ndarray,
    retained: np.ndarray,
    strata: np.ndarray,
    chain: np.ndarray,
    draw: np.ndarray,
    repeat: np.ndarray,
    dependence: np.ndarray,
    stratum_count: int,
    maximum_lag: int | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.ones((stratum_count,), dtype=np.float64)
    resolved = np.ones((stratum_count,), dtype=np.bool_)
    for stratum in range(stratum_count):
        selected_state = retained & (strata == stratum)
        if not np.any(selected_state):
            resolved[stratum] = False
            continue
        group_keys = sorted(
            set(
                zip(
                    repeat[selected_state].tolist(),
                    dependence[selected_state].tolist(),
                )
            )
        )
        for repeat_id, dependence_id in group_keys:
            selected = (
                selected_state & (repeat == repeat_id) & (dependence == dependence_id)
            )
            sequences: list[np.ndarray] = []
            for chain_id in sorted(set(chain[selected].tolist())):
                indices = np.nonzero(selected & (chain == chain_id))[0]
                indices = indices[np.argsort(draw[indices], kind="stable")]
                if indices.size:
                    sequences.append(values[indices])
            sample_count = sum(sequence.size for sequence in sequences)
            if sample_count < 2:
                resolved[stratum] = False
                continue
            concatenated = np.concatenate(sequences)
            mean = float(np.mean(concatenated))
            variance_numerator = sum(
                float(np.sum((sequence - mean) ** 2)) for sequence in sequences
            )
            variance = variance_numerator / sample_count
            scale = max(float(np.max(np.abs(concatenated), initial=0.0)), 1.0)
            variance_floor = 128.0 * np.finfo(concatenated.dtype).eps * scale * scale
            if not math.isfinite(variance):
                resolved[stratum] = False
                continue
            if variance <= variance_floor:
                continue
            available_lag = max(sequence.size for sequence in sequences) - 1
            if available_lag < 1:
                resolved[stratum] = False
                continue
            lag_limit = (
                available_lag if maximum_lag is None else min(available_lag, maximum_lag)
            )
            correlations: list[float] = []
            correlation_valid = True
            for lag in range(1, lag_limit + 1):
                numerator = 0.0
                count = 0
                for sequence in sequences:
                    if sequence.size > lag:
                        numerator += float(
                            np.sum((sequence[:-lag] - mean) * (sequence[lag:] - mean))
                        )
                        count += sequence.size - lag
                if count == 0:
                    correlation_valid = False
                    break
                correlation = numerator / (count * variance)
                if not math.isfinite(correlation):
                    correlation_valid = False
                    break
                correlations.append(correlation)
            if not correlation_valid:
                resolved[stratum] = False
                continue
            included = 0.0
            for index in range(0, len(correlations), 2):
                pair_sum = correlations[index]
                if index + 1 < len(correlations):
                    pair_sum += correlations[index + 1]
                if pair_sum <= 0.0:
                    break
                included += pair_sum
            result[stratum] = max(
                result[stratum],
                max(1.0, 1.0 + 2.0 * included),
            )
    return result, resolved


def _select_free_energy_dataset(
    dataset: FreeEnergyDataset,
    plan: FreeEnergySelectionPlan,
    /,
) -> FreeEnergySelectionEvidence:
    if not isinstance(plan, FreeEnergySelectionPlan):
        raise TypeError("plan must be FreeEnergySelectionPlan.")
    (
        values_,
        active_,
        strata_,
        chain_,
        draw_,
        repeat_,
        dependence_,
        kind,
        stratum_count,
    ) = _dataset_observations(dataset)
    shape = active_.shape
    values = np.asarray(values_).reshape((-1,))
    active = np.asarray(active_).reshape((-1,))
    strata = np.asarray(strata_).reshape((-1,))
    chain = np.asarray(chain_).reshape((-1,))
    draw = np.asarray(draw_).reshape((-1,))
    repeat = np.asarray(repeat_).reshape((-1,))
    dependence = np.asarray(dependence_).reshape((-1,))
    retained = active.copy()
    grouping = sorted(
        set(
            zip(
                strata[active].tolist(),
                repeat[active].tolist(),
                dependence[active].tolist(),
                chain[active].tolist(),
            )
        )
    )
    for stratum, repeat_index, dependence_index, chain_index in grouping:
        indices = np.nonzero(
            active
            & (strata == stratum)
            & (repeat == repeat_index)
            & (dependence == dependence_index)
            & (chain == chain_index)
        )[0]
        ordered = indices[np.argsort(draw[indices], kind="stable")]
        retained[ordered] = False
        selected = ordered[plan.burn_in :: plan.stride]
        retained[selected] = True
    if isinstance(dataset, ReducedPotentialDataset):
        potential = np.asarray(dataset.values)
        safe_origin = np.clip(strata, 0, stratum_count - 1)
        origin_value = potential[safe_origin, np.arange(potential.shape[1])]
        inefficiency = np.ones((stratum_count,), dtype=np.float64)
        correlation_resolved = np.asarray(
            [np.any(retained & (strata == state)) for state in range(stratum_count)],
            dtype=np.bool_,
        )
        for target in range(stratum_count):
            work = potential[target] - origin_value
            target_inefficiency, target_resolved = _correlation_inefficiency(
                work,
                retained,
                strata,
                chain,
                draw,
                repeat,
                dependence,
                stratum_count,
                plan.maximum_correlation_lag,
            )
            relevant = np.arange(stratum_count) != target
            inefficiency[relevant] = np.maximum(
                inefficiency[relevant],
                target_inefficiency[relevant],
            )
            correlation_resolved[relevant] &= target_resolved[relevant]
    else:
        inefficiency, correlation_resolved = _correlation_inefficiency(
            values,
            retained,
            strata,
            chain,
            draw,
            repeat,
            dependence,
            stratum_count,
            plan.maximum_correlation_lag,
        )
    resolved_block_length = (
        max(1, int(math.ceil(float(np.max(inefficiency, initial=1.0)))))
        if plan.block_length is None
        else plan.block_length
    )
    block_keys: list[tuple[int, int, int]] = []
    group_keys = sorted(
        set(zip(repeat[retained].tolist(), dependence[retained].tolist()))
    )
    group_lookup = {key: index for index, key in enumerate(group_keys)}
    group_index = np.full(active.shape, -1, dtype=np.int32)
    observation_block_key: dict[int, tuple[int, int, int]] = {}
    for key in group_keys:
        selected = retained & (repeat == key[0]) & (dependence == key[1])
        unique_draws = sorted(set(int(value) for value in draw[selected]))
        draw_rank = {value: index for index, value in enumerate(unique_draws)}
        group_index[selected] = group_lookup[key]
        for index in np.nonzero(selected)[0]:
            block_key = (
                key[0],
                key[1],
                draw_rank[int(draw[index])] // resolved_block_length,
            )
            observation_block_key[int(index)] = block_key
            block_keys.append(block_key)
    unique_blocks = sorted(set(block_keys))
    block_lookup = {key: index for index, key in enumerate(unique_blocks)}
    block_index = np.full(active.shape, -1, dtype=np.int32)
    for index, key in observation_block_key.items():
        block_index[index] = block_lookup[key]
    raw_counts = np.asarray(
        [np.sum(active & (strata == state)) for state in range(stratum_count)],
        dtype=np.int32,
    )
    retained_counts = np.asarray(
        [np.sum(retained & (strata == state)) for state in range(stratum_count)],
        dtype=np.int32,
    )
    block_counts = np.asarray(
        [
            len(set(block_index[retained & (strata == state)].tolist()))
            for state in range(stratum_count)
        ],
        dtype=np.int32,
    )
    effective = retained_counts / inefficiency
    return FreeEnergySelectionEvidence(
        retained.reshape(shape),
        block_index.reshape(shape),
        group_index.reshape(shape),
        np.where(retained, strata, -1).astype(np.int32).reshape(shape),
        raw_counts,
        retained_counts,
        block_counts,
        inefficiency,
        correlation_resolved,
        effective,
        plan,
        resolved_block_length=resolved_block_length,
        block_count=len(unique_blocks),
        dataset_kind=kind,
        dataset_id=dataset.dataset_id,
    )


def _selection(
    dataset: FreeEnergyDataset,
    selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None,
    /,
) -> FreeEnergySelectionEvidence:
    if selection is None:
        evidence = FreeEnergySelectionPlan().select(dataset)
    elif isinstance(selection, FreeEnergySelectionPlan):
        evidence = selection.select(dataset)
    elif isinstance(selection, FreeEnergySelectionEvidence):
        evidence = selection
    else:
        raise TypeError("selection must be FreeEnergySelectionPlan, evidence, or None.")
    observations = _dataset_observations(dataset)
    expected_kind = observations[7]
    expected_shape = tuple(observations[1].shape)
    if (
        evidence.dataset_id != dataset.dataset_id
        or evidence.dataset_kind != expected_kind
        or evidence.observation_shape != expected_shape
    ):
        raise ValueError("Selection evidence is not bound to this exact dataset.")
    return evidence


def _selection_status(
    evidence: FreeEnergySelectionEvidence,
    required: np.ndarray,
    /,
) -> int:
    status = FreeEnergyStatus.SUCCESS
    counts = np.asarray(evidence.retained_counts)
    blocks = np.asarray(evidence.block_counts)
    required_indices = np.nonzero(required)[0]
    if np.any(counts[required_indices] < evidence.plan.minimum_samples_per_state):
        status |= FreeEnergyStatus.INSUFFICIENT_SAMPLES
    if np.any(blocks[required_indices] < evidence.plan.minimum_blocks):
        status |= FreeEnergyStatus.INSUFFICIENT_BLOCKS
    if np.any(~np.asarray(evidence.correlation_resolved)[required_indices]):
        status |= FreeEnergyStatus.CORRELATION_UNRESOLVED
    if evidence.resolved_block_length < int(
        math.ceil(
            float(
                np.max(
                    np.asarray(evidence.statistical_inefficiency)[required_indices],
                    initial=1.0,
                )
            )
        )
    ):
        status |= FreeEnergyStatus.CORRELATION_UNRESOLVED
    return int(status)


def _block_factor(
    observation_influence: Array,
    evidence: FreeEnergySelectionEvidence,
    /,
) -> Array:
    block_count = evidence.block_count
    if block_count == 0:
        return jnp.zeros(
            (observation_influence.shape[0], 0), dtype=observation_influence.dtype
        )
    block = evidence.block_index.reshape((-1,))
    retained = evidence.retained.reshape((-1,))
    safe_block = jnp.where(retained, block, 0)
    factor = jnp.zeros(
        (observation_influence.shape[0], block_count), dtype=observation_influence.dtype
    )
    factor = factor.at[:, safe_block].add(observation_influence * retained[None, :])
    if block_count > 1:
        factor = factor * math.sqrt(block_count / (block_count - 1.0))
    else:
        factor = jnp.zeros_like(factor)
    return factor


def _bootstrap_observation_weights(
    evidence: FreeEnergySelectionEvidence,
    key: ArrayLike | None,
    /,
) -> Array:
    if evidence.plan.uncertainty_method != "block-bootstrap":
        raise ValueError("Bootstrap weights require a block-bootstrap selection plan.")
    if key is None:
        raise ValueError(
            "Block-bootstrap uncertainty requires an explicit JAX random key."
        )
    replicates = evidence.plan.bootstrap_replicates
    block_count = evidence.block_count
    block = np.asarray(evidence.block_index).reshape((-1,))
    group = np.asarray(evidence.block_group_index).reshape((-1,))
    retained = np.asarray(evidence.retained).reshape((-1,))
    groups = sorted(set(int(value) for value in group[retained]))
    keys = jr.split(jnp.asarray(key), max(len(groups), 1))
    multiplicity = jnp.zeros((replicates, block_count), dtype=jnp.float64)
    for key_index, group_id in enumerate(groups):
        group_blocks = sorted(
            set(int(value) for value in block[retained & (group == group_id)])
        )
        count = len(group_blocks)
        draws = jr.randint(keys[key_index], (replicates, count), 0, count)
        local = jnp.sum(jax.nn.one_hot(draws, count, dtype=jnp.float64), axis=1)
        multiplicity = multiplicity.at[:, jnp.asarray(group_blocks, dtype=jnp.int32)].set(
            local
        )
    block_array = evidence.block_index.reshape((-1,))
    retained_array = evidence.retained.reshape((-1,))
    safe_block = jnp.where(retained_array, block_array, 0)
    return multiplicity[:, safe_block] * retained_array[None, :]


def _bootstrap_factor(estimates: Array, /) -> tuple[Array, Array, bool]:
    finite = bool(jnp.all(jnp.isfinite(estimates)))
    safe = jnp.where(jnp.isfinite(estimates), estimates, 0.0)
    centered = safe - jnp.mean(safe, axis=0, keepdims=True)
    factor = centered.T / math.sqrt(max(estimates.shape[0] - 1, 1))
    covariance = factor @ factor.T
    return covariance, factor, finite


def _covariance_rank(covariance: Array, tolerance: float = 0.0, /) -> Array:
    factors = factor_pseudoinverse(
        covariance,
        RankPolicy(relative_cutoff=None if tolerance == 0.0 else tolerance),
        hermitian=True,
    )
    return factors.rank


def _transitive_connectivity(overlap: Array, minimum: float, /) -> Array:
    host = np.asarray(overlap)
    connected = (np.maximum(host, host.T) >= minimum) | np.eye(
        host.shape[0], dtype=np.bool_
    )
    for intermediate in range(host.shape[0]):
        connected |= connected[:, intermediate, None] & connected[None, intermediate, :]
    return jnp.asarray(connected)


def _require_complete_work(dataset: ReducedWorkDataset, /) -> None:
    complete = bool(jnp.all(jnp.where(dataset.sample_active, dataset.coverage, True)))
    if not complete:
        raise ValueError(
            "Free-energy analysis requires coverage for every attempted work sample."
        )


def free_energy_perturbation(
    dataset: ReducedWorkDataset,
    selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None = None,
    /,
    *,
    key: ArrayLike | None = None,
) -> FreeEnergyResult:
    """Estimate state 1 minus state 0 from authenticated 0-to-1 reduced work."""

    if not isinstance(dataset, ReducedWorkDataset):
        raise TypeError("dataset must be ReducedWorkDataset.")
    _require_complete_work(dataset)
    evidence = _selection(dataset, selection)
    retained = evidence.retained
    forward = retained & (dataset.source_state == 0) & (dataset.destination_state == 1)
    count = jnp.sum(forward)
    delta, raw_ess_target, normalized_influence = fep_kernel(
        dataset.values,
        forward,
        jnp.ones_like(dataset.values),
    )
    observation_influence = jnp.stack(
        (jnp.zeros_like(normalized_influence), normalized_influence)
    )
    factor = _block_factor(observation_influence, evidence)
    covariance = factor @ factor.T
    numerical_status = FreeEnergyStatus.SUCCESS
    if int(np.asarray(count)) == 0 or not math.isfinite(float(np.asarray(delta))):
        numerical_status |= FreeEnergyStatus.NONFINITE
        delta = jnp.asarray(0.0, dtype=dataset.values.dtype)
        factor = jnp.zeros_like(factor)
        covariance = jnp.zeros((2, 2), dtype=dataset.values.dtype)
    if evidence.plan.uncertainty_method == "block-bootstrap":
        bootstrap_weight = _bootstrap_observation_weights(evidence, key)
        estimates = jax.vmap(
            lambda weight: fep_kernel(dataset.values, forward, weight)[0]
        )(bootstrap_weight)
        covariance, factor, finite = _bootstrap_factor(
            jnp.stack((jnp.zeros_like(estimates), estimates), axis=1)
        )
        if not finite:
            numerical_status |= FreeEnergyStatus.NONFINITE
    statistical_status = FreeEnergyStatus(
        _selection_status(evidence, np.asarray([True, False]))
    )
    if not dataset.sampling_exact:
        statistical_status |= FreeEnergyStatus.UNQUALIFIED_KERNEL
    inefficiency = evidence.statistical_inefficiency[0]
    adjusted_target = raw_ess_target / inefficiency
    if float(np.asarray(adjusted_target)) < evidence.plan.minimum_samples_per_state:
        statistical_status |= FreeEnergyStatus.POOR_OVERLAP
    overlap_value = jnp.where(count > 0, raw_ess_target / count, 0.0)
    overlap = jnp.asarray([[1.0, overlap_value], [overlap_value, 1.0]])
    connectivity = _transitive_connectivity(overlap, evidence.plan.minimum_overlap)
    if float(np.asarray(overlap_value)) < evidence.plan.minimum_overlap:
        statistical_status |= FreeEnergyStatus.POOR_OVERLAP
    if not bool(np.all(np.asarray(connectivity))):
        statistical_status |= FreeEnergyStatus.DISCONNECTED
    free = jnp.asarray([0.0, delta])
    return FreeEnergyResult(
        free,
        covariance,
        overlap,
        connectivity,
        jnp.asarray([count, raw_ess_target]),
        jnp.asarray([count / inefficiency, adjusted_target]),
        factor,
        jnp.asarray(1),
        jnp.asarray(0.0),
        _covariance_rank(covariance),
        numerical_status,
        statistical_status,
        state_ids=dataset.state_ids,
        gauge_state_id=dataset.state_ids[0],
        method="free-energy-perturbation",
        dataset_id=dataset.dataset_id,
        selection_id=evidence.selection_id,
    )


def bennett_acceptance_ratio(
    dataset: ReducedWorkDataset,
    selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None = None,
    /,
    *,
    maximum_iterations: int = 128,
    tolerance: float = 1.0e-10,
    key: ArrayLike | None = None,
) -> FreeEnergyResult:
    """Solve BAR for explicitly oriented forward and reverse reduced work."""

    if not isinstance(dataset, ReducedWorkDataset):
        raise TypeError("dataset must be ReducedWorkDataset.")
    _require_complete_work(dataset)
    iterations_limit = int(maximum_iterations)
    tolerance_ = float(tolerance)
    if iterations_limit < 1 or not math.isfinite(tolerance_) or tolerance_ <= 0.0:
        raise ValueError("BAR solver controls must be finite and positive.")
    evidence = _selection(dataset, selection)
    retained = evidence.retained
    forward = retained & (dataset.source_state == 0) & (dataset.destination_state == 1)
    reverse = retained & (dataset.source_state == 1) & (dataset.destination_state == 0)
    unit_weight = jnp.ones_like(dataset.values)
    (
        estimate,
        residual,
        iterations,
        forward_probability,
        reverse_probability,
        direction_counts,
    ) = bar_kernel(
        dataset.values,
        forward,
        reverse,
        unit_weight,
        maximum_iterations=iterations_limit,
        tolerance=tolerance_,
    )
    forward_count, reverse_count = direction_counts
    derivative = jnp.sum(
        forward * forward_probability * (1.0 - forward_probability)
    ) + jnp.sum(reverse * reverse_probability * (1.0 - reverse_probability))
    forward_mean = jnp.sum(forward * forward_probability) / jnp.maximum(forward_count, 1)
    reverse_mean = jnp.sum(reverse * reverse_probability) / jnp.maximum(reverse_count, 1)
    delta_influence = jnp.where(
        forward,
        -(forward_probability - forward_mean) / jnp.maximum(derivative, 1.0e-30),
        0.0,
    ) + jnp.where(
        reverse,
        (reverse_probability - reverse_mean) / jnp.maximum(derivative, 1.0e-30),
        0.0,
    )
    observation_influence = jnp.stack((jnp.zeros_like(delta_influence), delta_influence))
    factor = _block_factor(observation_influence, evidence)
    covariance = factor @ factor.T
    numerical_status = FreeEnergyStatus.SUCCESS
    if not bool(np.asarray(residual <= tolerance_)):
        numerical_status |= FreeEnergyStatus.NONCONVERGED
    if int(np.asarray(forward_count)) == 0 or int(np.asarray(reverse_count)) == 0:
        numerical_status |= FreeEnergyStatus.NONFINITE
        estimate = jnp.asarray(0.0, dtype=dataset.values.dtype)
        factor = jnp.zeros_like(factor)
        covariance = jnp.zeros((2, 2), dtype=dataset.values.dtype)
    if evidence.plan.uncertainty_method == "block-bootstrap":
        bootstrap_weight = _bootstrap_observation_weights(evidence, key)
        bootstrap_outputs = jax.vmap(
            lambda weight: bar_kernel(
                dataset.values,
                forward,
                reverse,
                weight,
                maximum_iterations=iterations_limit,
                tolerance=tolerance_,
            )
        )(bootstrap_weight)
        estimates, bootstrap_residuals = bootstrap_outputs[:2]
        covariance, factor, finite = _bootstrap_factor(
            jnp.stack((jnp.zeros_like(estimates), estimates), axis=1)
        )
        if not finite:
            numerical_status |= FreeEnergyStatus.NONFINITE
        if not bool(jnp.all(bootstrap_residuals <= tolerance_)):
            numerical_status |= FreeEnergyStatus.NONCONVERGED
    forward_sum = jnp.sum(forward * forward_probability)
    reverse_sum = jnp.sum(reverse * reverse_probability)
    forward_ess = forward_sum**2 / jnp.maximum(
        jnp.sum(forward * forward_probability**2), 1.0e-300
    )
    reverse_ess = reverse_sum**2 / jnp.maximum(
        jnp.sum(reverse * reverse_probability**2), 1.0e-300
    )
    raw_ess = jnp.asarray([forward_ess, reverse_ess])
    adjusted = raw_ess / evidence.statistical_inefficiency
    overlap_value = jnp.minimum(forward_mean + reverse_mean, 1.0)
    overlap = jnp.asarray([[1.0, overlap_value], [overlap_value, 1.0]])
    connectivity = _transitive_connectivity(overlap, evidence.plan.minimum_overlap)
    statistical_status = FreeEnergyStatus(
        _selection_status(evidence, np.asarray([True, True]))
    )
    if not dataset.sampling_exact:
        statistical_status |= FreeEnergyStatus.UNQUALIFIED_KERNEL
    if (
        float(np.asarray(jnp.min(adjusted))) < evidence.plan.minimum_samples_per_state
        or float(np.asarray(overlap_value)) < evidence.plan.minimum_overlap
    ):
        statistical_status |= FreeEnergyStatus.POOR_OVERLAP
    return FreeEnergyResult(
        jnp.asarray([0.0, estimate]),
        covariance,
        overlap,
        connectivity,
        raw_ess,
        adjusted,
        factor,
        iterations,
        jnp.where(jnp.isfinite(residual), residual, 0.0),
        _covariance_rank(covariance),
        numerical_status,
        statistical_status,
        state_ids=dataset.state_ids,
        gauge_state_id=dataset.state_ids[0],
        method="bennett-acceptance-ratio",
        dataset_id=dataset.dataset_id,
        selection_id=evidence.selection_id,
    )


def thermodynamic_integration(
    dataset: ThermodynamicDerivativeDataset,
    selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None = None,
    /,
    *,
    key: ArrayLike | None = None,
) -> FreeEnergyResult:
    """Integrate raw complete reduced path derivatives with joint block covariance."""

    if not isinstance(dataset, ThermodynamicDerivativeDataset):
        raise TypeError("dataset must be ThermodynamicDerivativeDataset.")
    evidence = _selection(dataset, selection)
    retained = evidence.retained
    (
        free,
        means,
        counts,
        transform,
        observation_influence,
    ) = thermodynamic_integration_kernel(
        dataset.values,
        retained,
        dataset.path_parameter,
    )
    state_count, capacity = dataset.values.shape
    factor = _block_factor(observation_influence, evidence)
    covariance = factor @ factor.T
    numerical_status = FreeEnergyStatus.SUCCESS
    if evidence.plan.uncertainty_method == "block-bootstrap":
        bootstrap_weight = _bootstrap_observation_weights(evidence, key).reshape(
            (evidence.plan.bootstrap_replicates, state_count, capacity)
        )
        bootstrap_count = jnp.sum(bootstrap_weight, axis=2)
        bootstrap_mean = jnp.sum(
            bootstrap_weight * dataset.values[None, :, :], axis=2
        ) / jnp.maximum(bootstrap_count, 1.0)
        estimates = contract("ij,rj->ri", transform, bootstrap_mean)
        covariance, factor, finite = _bootstrap_factor(estimates)
        if not finite or bool(jnp.any(bootstrap_count == 0.0)):
            numerical_status |= FreeEnergyStatus.NONFINITE
    required = np.ones((state_count,), dtype=np.bool_)
    statistical_status = FreeEnergyStatus(_selection_status(evidence, required))
    if not dataset.sampling_exact:
        statistical_status |= FreeEnergyStatus.UNQUALIFIED_KERNEL
    raw_ess = counts.astype(dataset.values.dtype)
    adjusted = raw_ess / evidence.statistical_inefficiency
    connectivity = jnp.ones((state_count, state_count), dtype=jnp.bool_)
    return FreeEnergyResult(
        free,
        covariance,
        jnp.eye(state_count, dtype=dataset.values.dtype),
        connectivity,
        raw_ess,
        adjusted,
        factor,
        jnp.asarray(1),
        jnp.asarray(0.0),
        _covariance_rank(covariance),
        numerical_status,
        statistical_status,
        state_ids=dataset.state_ids,
        gauge_state_id=dataset.state_ids[0],
        method="thermodynamic-integration",
        dataset_id=dataset.dataset_id,
        selection_id=evidence.selection_id,
    )


def multistate_bennett_acceptance_ratio(
    dataset: ReducedPotentialDataset,
    selection: FreeEnergySelectionPlan | FreeEnergySelectionEvidence | None = None,
    /,
    *,
    reference_state: int = 0,
    maximum_iterations: int = 10_000,
    tolerance: float = 1.0e-10,
    rank_tolerance: float = 1.0e-12,
    key: ArrayLike | None = None,
) -> FreeEnergyResult:
    """Solve dense MBAR with a fixed gauge and rank-aware joint covariance."""

    if not isinstance(dataset, ReducedPotentialDataset):
        raise TypeError("dataset must be ReducedPotentialDataset.")
    state_count = len(dataset.state_ids)
    reference = int(reference_state)
    iterations_limit = int(maximum_iterations)
    tolerance_ = float(tolerance)
    rank_tolerance_ = float(rank_tolerance)
    if reference < 0 or reference >= state_count:
        raise ValueError("reference_state must identify one declared state.")
    if (
        iterations_limit < 1
        or not math.isfinite(tolerance_)
        or tolerance_ <= 0.0
        or not math.isfinite(rank_tolerance_)
        or rank_tolerance_ < 0.0
    ):
        raise ValueError("MBAR solver and rank controls are invalid.")
    evidence = _selection(dataset, selection)
    retained = evidence.retained
    sample_weight = retained.astype(dataset.values.dtype)
    free, residual, iterations, counts, weights = mbar_kernel(
        dataset.values,
        dataset.origin_state,
        sample_weight,
        reference_state=reference,
        maximum_iterations=iterations_limit,
        tolerance=tolerance_,
    )
    gram = contract("in,n,jn->ij", weights, sample_weight, weights)
    overlap = gram * counts[None, :]
    connectivity = _transitive_connectivity(overlap, evidence.plan.minimum_overlap)
    covariance = mbar_asymptotic_covariance_kernel(
        weights,
        sample_weight,
        counts,
        reference_state=reference,
        rank_tolerance=rank_tolerance_,
    )
    sampled = np.asarray(counts) > 0.0
    maximum_inefficiency = float(
        np.max(np.asarray(evidence.statistical_inefficiency)[sampled], initial=1.0)
    )
    covariance = covariance * maximum_inefficiency
    covariance_factors = factor_pseudoinverse(
        covariance,
        RankPolicy(relative_cutoff=rank_tolerance_),
        hermitian=True,
    )
    factor = (
        covariance_factors.left_vectors
        * jnp.sqrt(
            jnp.where(
                covariance_factors.retained, covariance_factors.singular_values, 0.0
            )
        )[None, :]
    )
    numerical_status = FreeEnergyStatus.SUCCESS
    if not bool(np.asarray(residual <= tolerance_)):
        numerical_status |= FreeEnergyStatus.NONCONVERGED
    if not bool(jnp.all(jnp.isfinite(free))) or not bool(
        jnp.all(jnp.isfinite(covariance))
    ):
        numerical_status |= FreeEnergyStatus.NONFINITE
        free = jnp.where(jnp.isfinite(free), free, 0.0)
        covariance = jnp.zeros((state_count, state_count), dtype=dataset.values.dtype)
        factor = jnp.zeros((state_count, state_count), dtype=dataset.values.dtype)
    if evidence.plan.uncertainty_method == "block-bootstrap":
        bootstrap_weight = _bootstrap_observation_weights(evidence, key)
        bootstrap_outputs = jax.vmap(
            lambda weight: mbar_kernel(
                dataset.values,
                dataset.origin_state,
                weight,
                reference_state=reference,
                maximum_iterations=iterations_limit,
                tolerance=tolerance_,
            )
        )(bootstrap_weight)
        estimates, bootstrap_residuals = bootstrap_outputs[:2]
        covariance, factor, finite = _bootstrap_factor(estimates)
        if not finite:
            numerical_status |= FreeEnergyStatus.NONFINITE
        if not bool(jnp.all(bootstrap_residuals <= tolerance_)):
            numerical_status |= FreeEnergyStatus.NONCONVERGED
    inverse_ess = jnp.sum(sample_weight[None, :] * weights**2, axis=1)
    raw_ess = jnp.where(
        inverse_ess > 0.0,
        1.0 / jnp.maximum(inverse_ess, 1.0e-300),
        0.0,
    )
    effective = raw_ess / maximum_inefficiency
    required = sampled
    statistical_status = FreeEnergyStatus(_selection_status(evidence, required))
    if not dataset.sampling_exact:
        statistical_status |= FreeEnergyStatus.UNQUALIFIED_KERNEL
    if not bool(np.all(np.asarray(connectivity))):
        statistical_status |= FreeEnergyStatus.DISCONNECTED
    off_diagonal = np.asarray(overlap).copy()
    np.fill_diagonal(off_diagonal, 0.0)
    if (
        state_count > 1
        and float(np.max(off_diagonal, initial=0.0)) < evidence.plan.minimum_overlap
    ):
        statistical_status |= FreeEnergyStatus.POOR_OVERLAP
    return FreeEnergyResult(
        free,
        covariance,
        overlap,
        connectivity,
        raw_ess,
        effective,
        factor,
        iterations,
        jnp.where(jnp.isfinite(residual), residual, 0.0),
        _covariance_rank(covariance, rank_tolerance_),
        numerical_status,
        statistical_status,
        state_ids=dataset.state_ids,
        gauge_state_id=dataset.state_ids[reference],
        method="multistate-bennett-acceptance-ratio",
        dataset_id=dataset.dataset_id,
        selection_id=evidence.selection_id,
    )


__all__ = [
    "FreeEnergyResult",
    "FreeEnergySelectionEvidence",
    "FreeEnergySelectionPlan",
    "FreeEnergyStatus",
    "ReducedPotentialDataset",
    "ReducedWorkDataset",
    "ThermodynamicDerivativeDataset",
    "bennett_acceptance_ratio",
    "free_energy_perturbation",
    "multistate_bennett_acceptance_ratio",
    "thermodynamic_integration",
]
