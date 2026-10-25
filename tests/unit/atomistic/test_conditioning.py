#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._differentiation import (
    ComponentAuthority,
    DerivativeSurface,
    DifferentiationRequest,
)
from phydrax._fingerprint import array_tree_fingerprint
from phydrax._model._ports import ValuePort
from phydrax._strict import StrictModule
from phydrax._trainable import ParameterOwner, partition_parameters
from phydrax.atomistic import (
    AtomisticConditionField,
    AtomisticConditioning,
    AtomisticConditioningSpec,
    AtomisticConditionKind,
    AtomisticConditionLevel,
    AtomisticConditionStatus,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticObservableValue,
    TOTAL_CHARGE_PORT,
)
from phydrax.units import DIMENSIONLESS, ELEMENTARY_CHARGE, METER, UnitDefinition


pytestmark = pytest.mark.strict_jax


@final
class _EvidenceParameters(StrictModule, ParameterOwner):
    weight: Array


def test_output_snapshots_freeze_embedded_parameters_without_stopping_autodiff() -> None:
    evidence = _EvidenceParameters(jnp.asarray(2.0, dtype=jnp.float64))
    model_parameters, _, _ = partition_parameters(evidence)
    np.testing.assert_array_equal(model_parameters.weight, 2.0)
    outputs = AtomisticModelOutputs(
        jnp.asarray([3.0], dtype=jnp.float64),
        jnp.asarray([[3.0]], dtype=jnp.float64),
        jnp.asarray([True], dtype=jnp.bool_),
        evidence=(evidence,),
    )
    parameters, model_state, fixed = partition_parameters(outputs)
    assert parameters is None and model_state is None
    np.testing.assert_array_equal(fixed.energy, np.asarray([3.0], dtype=np.float64))

    def total_energy(value: AtomisticModelOutputs) -> Array:
        return jnp.sum(value.energy)

    gradient = eqx.filter_grad(total_energy)(outputs)
    np.testing.assert_array_equal(gradient.energy, np.asarray([1.0], dtype=np.float64))


def _scalar_port(name: str) -> ValuePort:
    return ValuePort(
        name,
        event_shape=(),
        component_ids=("value",),
        dimensions=(DIMENSIONLESS,),
        representation="declared-scalar",
        normalization_id="identity",
    )


def _spec() -> AtomisticConditioningSpec:
    return AtomisticConditioningSpec(
        (
            AtomisticConditionField(
                "charge",
                TOTAL_CHARGE_PORT,
                unit=ELEMENTARY_CHARGE,
                minimum=-1.0,
                maximum=1.0,
            ),
            AtomisticConditionField(
                "spin",
                _scalar_port("test.source-spin-index"),
                kind=AtomisticConditionKind.CATEGORICAL,
                dtype=jnp.int32,
                categories=(0, 2),
                required=False,
            ),
        )
    )


def test_condition_status_keeps_invalid_cases_isolated_under_jit() -> None:
    conditions = AtomisticConditioning(
        _spec(),
        {
            "charge": np.asarray([0.0, 3.0], dtype=np.float64),
            "spin": np.asarray([0, 1], dtype=np.int32),
        },
        case_count=2,
        atom_capacity=2,
    )
    mask = jnp.asarray([True, False, True, False], dtype=jnp.bool_)
    status = jax.jit(lambda value: value.case_status(mask))(conditions)
    np.testing.assert_array_equal(
        status,
        np.asarray(
            [
                0,
                int(
                    AtomisticConditionStatus.OUT_OF_RANGE
                    | AtomisticConditionStatus.INVALID_CATEGORY
                ),
            ],
            dtype=np.int32,
        ),
    )


def test_explicit_defaults_do_not_invent_optional_conditions() -> None:
    conditions = AtomisticConditioning(
        _spec(),
        {},
        case_count=2,
        atom_capacity=1,
        defaults={"charge": np.asarray(0.0, dtype=np.float64)},
    )
    np.testing.assert_array_equal(
        conditions.value("charge"), np.zeros(2, dtype=np.float64)
    )
    np.testing.assert_array_equal(
        conditions.defaults_used,
        np.asarray([[True, False], [True, False]], dtype=np.bool_),
    )
    assert not conditions.present("spin")
    with pytest.raises(ValueError):
        conditions.value("spin")


def test_missing_required_condition_is_refused() -> None:
    with pytest.raises(ValueError):
        AtomisticConditioning(_spec(), {}, case_count=1, atom_capacity=1)


def test_categorical_condition_refuses_fractional_conversion() -> None:
    with pytest.raises(TypeError):
        AtomisticConditioning(
            _spec(),
            {
                "charge": np.asarray([0.0], dtype=np.float64),
                "spin": np.asarray([0.5], dtype=np.float64),
            },
            case_count=1,
            atom_capacity=1,
        )


def test_inactive_atom_conditions_do_not_poison_case_admission() -> None:
    spec = AtomisticConditioningSpec(
        (
            AtomisticConditionField(
                "feature",
                _scalar_port("test.atomic-feature"),
                level=AtomisticConditionLevel.ATOM,
                minimum=-1.0,
                maximum=1.0,
            ),
        )
    )
    conditions = AtomisticConditioning(
        spec,
        {"feature": np.asarray([0.25, np.nan], dtype=np.float64)},
        case_count=1,
        atom_capacity=2,
    )
    inactive = jnp.asarray([True, False], dtype=jnp.bool_)
    active = jnp.asarray([True, True], dtype=jnp.bool_)
    np.testing.assert_array_equal(
        conditions.case_status(inactive), np.zeros(1, dtype=np.int32)
    )
    status = conditions.case_status(active)
    assert bool(status[0] & int(AtomisticConditionStatus.NONFINITE))


@pytest.mark.parametrize("unit", [None, METER], ids=["missing-unit", "wrong-dimension"])
def test_dimensioned_condition_requires_compatible_unit(
    unit: UnitDefinition | None,
) -> None:
    with pytest.raises(ValueError):
        AtomisticConditionField("charge", TOTAL_CHARGE_PORT, unit=unit)


def test_observable_requests_canonicalize_identity_and_refuse_missing_quantity() -> None:
    left = AtomisticObservableRequest(
        (AtomisticObservable.CHARGES, AtomisticObservable.ENERGY)
    )
    right = AtomisticObservableRequest(
        (AtomisticObservable.ENERGY, AtomisticObservable.CHARGES)
    )
    assert left.request_id == right.request_id
    with pytest.raises(ValueError):
        left.require_supported((AtomisticObservable.ENERGY,), "energy-only model")


def test_available_nonfinite_property_invalidates_only_its_case() -> None:
    field = AtomisticObservableValue(
        AtomisticObservable.NODE_EMBEDDINGS,
        _scalar_port("test.embedding.nonfinite"),
        np.asarray([1.0, np.nan], dtype=np.float64),
        np.asarray([True, True], dtype=np.bool_),
        level=AtomisticConditionLevel.ATOM,
        case_count=2,
        atom_capacity=1,
    )
    outputs = AtomisticModelOutputs(
        np.asarray([1.0, 2.0], dtype=np.float64),
        np.asarray([[1.0], [2.0]], dtype=np.float64),
        np.asarray([True, True], dtype=np.bool_),
        additional=(field,),
    )
    np.testing.assert_array_equal(
        field.available, np.asarray([True, True], dtype=np.bool_)
    )
    np.testing.assert_array_equal(
        field.successful, np.asarray([True, False], dtype=np.bool_)
    )
    np.testing.assert_array_equal(
        outputs.successful, np.asarray([True, False], dtype=np.bool_)
    )


def test_named_embedding_ports_require_unambiguous_selection() -> None:
    first = _scalar_port("test.embedding.first")
    second = _scalar_port("test.embedding.second")
    fields = tuple(
        AtomisticObservableValue(
            AtomisticObservable.NODE_EMBEDDINGS,
            port,
            np.asarray([value], dtype=np.float64),
            np.asarray([True], dtype=np.bool_),
            level=AtomisticConditionLevel.ATOM,
            case_count=1,
            atom_capacity=1,
        )
        for port, value in ((first, 2.0), (second, 3.0))
    )
    outputs = AtomisticModelOutputs(
        np.zeros(1, dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
        np.ones(1, dtype=np.bool_),
        additional=fields,
    )
    with pytest.raises(ValueError):
        outputs.observable(AtomisticObservable.NODE_EMBEDDINGS)
    np.testing.assert_array_equal(
        outputs.observable(AtomisticObservable.NODE_EMBEDDINGS, port=second).values,
        np.asarray([3.0], dtype=np.float64),
    )


def _json(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record, allow_nan=False))


def test_conditioning_record_rebuilds_values_defaults_and_absence() -> None:
    defaulted = AtomisticConditioning(
        _spec(),
        {},
        case_count=2,
        atom_capacity=1,
        defaults={"charge": np.asarray(0.5, dtype=np.float64)},
    )
    record, arrays = defaulted.to_record()
    record = _json(record)
    assert record["bindings"] == ["default", "absent"]
    assert len(arrays) == 1 and arrays[0].shape == ()
    restored = AtomisticConditioning.from_record(record, arrays)
    assert restored.spec.spec_id == defaulted.spec.spec_id
    assert array_tree_fingerprint(restored) == array_tree_fingerprint(defaulted)
    assert not restored.present("spin")
    np.testing.assert_array_equal(restored.defaults_used, defaulted.defaults_used)

    # The same numerical value supplied explicitly is different default evidence.
    explicit = AtomisticConditioning(
        _spec(),
        {
            "charge": np.asarray([0.5, 0.5], dtype=np.float64),
            "spin": np.asarray([0, 2], dtype=np.int32),
        },
        case_count=2,
        atom_capacity=1,
    )
    record, arrays = explicit.to_record()
    assert record["bindings"] == ["value", "value"]
    rebuilt = AtomisticConditioning.from_record(_json(record), arrays)
    assert array_tree_fingerprint(rebuilt) == array_tree_fingerprint(explicit)
    assert array_tree_fingerprint(rebuilt) != array_tree_fingerprint(restored)


def test_conditioning_record_refuses_altered_metadata_and_inventory() -> None:
    conditions = AtomisticConditioning(
        _spec(),
        {"charge": np.asarray([0.25], dtype=np.float64)},
        case_count=1,
        atom_capacity=1,
    )
    record, arrays = conditions.to_record()
    record = _json(record)

    def altered(edit: Callable[[dict[str, Any]], object]) -> dict[str, Any]:
        value = copy.deepcopy(record)
        edit(value)
        return value

    edits: tuple[Callable[[dict[str, Any]], object], ...] = (
        lambda value: value["spec"]["fields"][0].update(maximum=2.0),
        lambda value: value["spec"]["fields"][0]["port"].update(representation="other"),
        lambda value: value["spec"]["fields"][0]["unit"].update(scale_numerator=2),
        lambda value: value["spec"].update(spec_id="0" * 64),
        lambda value: value["spec"]["fields"].reverse(),
        lambda value: value.update(bindings=["absent", "absent"]),
        lambda value: value.update(case_count=True),
    )
    for edit in edits:
        with pytest.raises((TypeError, ValueError)):
            AtomisticConditioning.from_record(altered(edit), arrays)
    with pytest.raises(ValueError, match="inventory"):
        AtomisticConditioning.from_record(record, (*arrays, arrays[0]))
    with pytest.raises(ValueError, match="declared layout"):
        AtomisticConditioning.from_record(record, (arrays[0].astype(np.float32),))
    with pytest.raises(ValueError, match="declared layout"):
        AtomisticConditioning.from_record(record, (np.zeros((2,), dtype=np.float64),))


def test_observable_request_record_rebuilds_derivatives_exactly() -> None:
    request = AtomisticObservableRequest(
        (AtomisticObservable.FORCES, AtomisticObservable.ENERGY),
        derivatives=(
            DifferentiationRequest(
                (DerivativeSurface.INPUT, DerivativeSurface.MODEL_PARAMETER),
                order=2,
                authority=ComponentAuthority.MODEL,
            ),
            DifferentiationRequest((DerivativeSurface.INPUT,)),
        ),
    )
    record = _json(request.to_dict())
    restored = AtomisticObservableRequest.from_dict(record)
    assert restored.request_id == request.request_id
    assert restored.observables == request.observables
    assert restored.derivatives == request.derivatives
    for edit in (
        {"request_id": "0" * 64},
        {"observables": ["energy"]},
        {"observables": ["not-an-observable"]},
        {"derivatives": [{**record["derivatives"][0], "order": True}]},
    ):
        with pytest.raises(ValueError):
            AtomisticObservableRequest.from_dict({**record, **edit})
