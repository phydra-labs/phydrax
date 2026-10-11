"""Compiled numerical fuel step shared by structurally identical rods.

`FuelNumericalExecution` compiles one pure step for a homogeneous workset;
`FuelNumericalBinding` binds one case's arrays to it and restores results under
that case's own scientific receipt. Production segments, checkpoints, retry
transactions, and attempt observers stay with each case's simulation.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import fixed_field
from ...discretization.fem import PreparedFiniteElementPointInterpolation
from ...solver import AbstractFixedStepMethod, FixedStepResult
from ._acceptance import FuelAcceptancePolicy, FuelStepEvidence
from ._bundles import FuelModelBundle
from ._execution_structure import (
    array_layout,
    ArrayLayout,
    dynamic_arrays,
    execution_structure,
)
from ._free_volume import FreeVolumeModel
from ._numerical_views import (
    FuelNumericalSpecification,
    NumericalFreeVolumeModel,
    NumericalPointInterpolation,
)
from ._problem import PreparedFuelRod
from ._specification import FuelRodSpecification
from ._state import FuelRodState


def _numerical_view(value: Any, /) -> Any:
    """Replace a host fabrication record by its device-array numerical view."""
    if isinstance(value, FuelRodSpecification):
        return FuelNumericalSpecification(value)
    if isinstance(value, FreeVolumeModel):
        return NumericalFreeVolumeModel(value)
    if isinstance(value, PreparedFiniteElementPointInterpolation):
        role_id = canonical_fingerprint(execution_structure(value))
        return NumericalPointInterpolation(value, role_id)
    return value


def _has_numerical_view(value: Any, /) -> bool:
    return isinstance(
        value,
        (FuelRodSpecification, FreeVolumeModel, PreparedFiniteElementPointInterpolation),
    )


def _numerical_method(method: AbstractFixedStepMethod, /) -> AbstractFixedStepMethod:
    """Substitute numerical views for host records; every other field is kept."""
    return jax.tree.map(_numerical_view, method, is_leaf=_has_numerical_view)


def _result_template(
    state: FuelRodState, evidence: FuelStepEvidence, /
) -> FixedStepResult:
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    flag = jnp.asarray(False, dtype=jnp.bool_)
    integer = jnp.asarray(0, dtype=jnp.int32)
    return FixedStepResult(
        candidate_state=state,
        accepted_state=state,
        successful=flag,
        residual=zero,
        iterations=integer,
        work=integer,
        transform_applied=flag,
        transform_correction_norm=zero,
        terminal_failure=flag,
        evidence=evidence,
    )


def fuel_numerical_execution_id(
    prepared: PreparedFuelRod, /, *, policy: FuelAcceptancePolicy | None = None
) -> str:
    """Structural identity shared by rods that can use one numerical execution."""
    from ._step import FuelStepMethod

    if not isinstance(prepared, PreparedFuelRod):
        raise TypeError("Fuel numerical execution identity requires a prepared rod.")
    return canonical_fingerprint(
        execution_structure(_numerical_method(FuelStepMethod(prepared, policy=policy)))
    )


class FuelNumericalExecution:
    """Compiled pure fuel step for one structurally homogeneous workset.

    Scientific IDs are never rewritten or equated: the template keeps the IDs of
    the rod it was built from, and only numerical arrays cross the JIT entry.
    Each `FuelNumericalBinding` restores results under its own case receipt.
    """

    def __init__(
        self, prepared: PreparedFuelRod, /, *, policy: FuelAcceptancePolicy | None = None
    ) -> None:
        from ._step import FuelStepMethod

        if not isinstance(prepared, PreparedFuelRod):
            raise TypeError("Fuel numerical execution requires a prepared rod.")
        bundle = prepared.admission.bundle
        method = FuelStepMethod(prepared, policy=policy)
        numerical = _numerical_method(method)
        self.policy = method.policy
        self.execution_id = canonical_fingerprint(execution_structure(numerical))
        self._input_layout = array_layout(dynamic_arrays(numerical))
        self._bundle_structure = execution_structure(bundle, topology=False)
        self._bundle_layout = array_layout(dynamic_arrays(bundle))
        method_dynamic, method_static = eqx.partition(numerical, eqx.is_array)
        bundle_dynamic, bundle_static = eqx.partition(bundle, eqx.is_array)
        method_definition = jax.tree.structure(method_dynamic)
        bundle_definition = jax.tree.structure(bundle_dynamic)
        state_definition = jax.tree.structure(prepared.initial_state())
        self.trace_count = 0

        def numerical_step(
            inputs: tuple[Array, ...],
            bundle_inputs: tuple[Array, ...],
            state_arrays: tuple[Array, ...],
            step_index: Array,
            time: Array,
            step_size: Array,
        ) -> tuple[Array, ...]:
            self.trace_count += 1
            numerical_method = eqx.combine(
                jax.tree.unflatten(method_definition, inputs), method_static
            )
            step_bundle = eqx.combine(
                jax.tree.unflatten(bundle_definition, bundle_inputs), bundle_static
            )
            state = jax.tree.unflatten(state_definition, state_arrays)
            result = numerical_method.step(
                step_index, time, state, step_size, step_bundle
            )
            return dynamic_arrays(result)

        self.compiled_step = jax.jit(numerical_step)

    def bind(
        self, prepared: PreparedFuelRod, /, *, policy: FuelAcceptancePolicy | None = None
    ) -> FuelNumericalBinding:
        return FuelNumericalBinding(
            self, prepared, policy=self.policy if policy is None else policy
        )


class FuelNumericalBinding(StrictModule):
    """One case's numerical inputs bound to a shared `FuelNumericalExecution`."""

    owner: FuelNumericalExecution = eqx.field(static=True)
    numerical_inputs: tuple[Array, ...] = fixed_field()
    bundle_inputs: tuple[Array, ...] = fixed_field()
    prepared_id: str = eqx.field(static=True)
    _state_layout: ArrayLayout = eqx.field(static=True)
    _result_layout: ArrayLayout = eqx.field(static=True)
    _result_definition: jax.tree_util.PyTreeDef = eqx.field(static=True)

    def __init__(
        self,
        owner: FuelNumericalExecution,
        prepared: PreparedFuelRod,
        /,
        *,
        policy: FuelAcceptancePolicy,
    ) -> None:
        from ._step import FuelStepMethod

        if not isinstance(owner, FuelNumericalExecution):
            raise TypeError("Fuel numerical binding requires its execution owner.")
        if not isinstance(prepared, PreparedFuelRod):
            raise TypeError("Fuel numerical binding requires a prepared rod.")
        method = FuelStepMethod(prepared, policy=policy)
        numerical = _numerical_method(method)
        if canonical_fingerprint(execution_structure(numerical)) != owner.execution_id:
            raise ValueError(
                "Fuel numerical execution layout, topology, or scientific role mismatch; regroup the workset."
            )
        inputs = dynamic_arrays(numerical)
        if array_layout(inputs) != owner._input_layout:
            raise ValueError(
                "Fuel numerical execution input layout mismatch; regroup the workset."
            )
        state = prepared.initial_state()
        template = _result_template(state, method.evidence_template)
        self.owner = owner
        self.numerical_inputs = inputs
        self.bundle_inputs = dynamic_arrays(prepared.admission.bundle)
        self.prepared_id = prepared.prepared_id
        self._state_layout = array_layout(dynamic_arrays(state))
        self._result_layout = array_layout(dynamic_arrays(template))
        self._result_definition = jax.tree.structure(template)

    def _parameter_inputs(self, parameters: FuelModelBundle, /) -> tuple[Array, ...]:
        if not isinstance(parameters, FuelModelBundle):
            raise TypeError("Fuel numerical parameters require a native fuel bundle.")
        if (
            execution_structure(parameters, topology=False)
            != self.owner._bundle_structure
        ):
            raise ValueError(
                "Fuel numerical parameter source or structural role mismatch."
            )
        inputs = dynamic_arrays(parameters)
        if array_layout(inputs) != self.owner._bundle_layout:
            raise ValueError("Fuel numerical parameter layout mismatch.")
        return inputs

    def restore_result(
        self, arrays: tuple[Array, ...], /, *, parameters: FuelModelBundle | None = None
    ) -> FixedStepResult:
        if array_layout(arrays) != self._result_layout:
            raise ValueError(
                "Fuel numerical result layout differs from its case receipt."
            )
        if parameters is not None:
            self._parameter_inputs(parameters)
        result = jax.tree.unflatten(self._result_definition, arrays)
        if not isinstance(result, FixedStepResult) or not isinstance(
            result.evidence, FuelStepEvidence
        ):
            raise RuntimeError(
                "Fuel numerical result lost its registered result contract."
            )
        if parameters is not None:
            # The parameter bundle shares the bound bundle's structure (checked
            # above); only its scientific receipt differs.
            result = dataclasses.replace(
                result,
                evidence=dataclasses.replace(
                    result.evidence, bundle_id=parameters.bundle_id
                ),
            )
        return result

    def step(
        self,
        step_index: Array,
        time: Array,
        state: FuelRodState,
        step_size: Array,
        /,
        *,
        parameters: FuelModelBundle | None = None,
    ) -> FixedStepResult:
        if not isinstance(state, FuelRodState):
            raise TypeError("Fuel numerical steps require a complete rod checkpoint.")
        if state.prepared_id != self.prepared_id:
            raise ValueError("Fuel state belongs to a different prepared FE identity.")
        state_arrays = dynamic_arrays(state)
        if array_layout(state_arrays) != self._state_layout:
            raise ValueError(
                "Fuel numerical state layout differs from the bound checkpoint."
            )
        bundle_inputs = (
            self.bundle_inputs
            if parameters is None
            else self._parameter_inputs(parameters)
        )
        arrays = self.owner.compiled_step(
            self.numerical_inputs,
            bundle_inputs,
            state_arrays,
            step_index,
            time,
            step_size,
        )
        return self.restore_result(arrays, parameters=parameters)


__all__ = [
    "FuelNumericalExecution",
    "FuelNumericalBinding",
    "fuel_numerical_execution_id",
]
