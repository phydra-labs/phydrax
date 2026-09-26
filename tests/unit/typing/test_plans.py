from __future__ import annotations

import dataclasses
import gc
import threading
import weakref
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax.typing as pt
from phydrax import StrictModule


if TYPE_CHECKING:
    from phydrax.typing import Float32 as CheckerOnlyFloat


class ComponentDim(pt.Dim, minimum=1):
    pass


class ElementDim(pt.Dim):
    pass


class Catalog(StrictModule):
    names: pt.Identifiers[ComponentDim] = eqx.field(static=True)
    masses: pt.Float64[ComponentDim]
    composition: pt.Int32[ElementDim, ComponentDim]
    basis: Literal["nodal", "modal"] = eqx.field(static=True)
    note: str = eqx.field(static=True)
    count: pt.Size[ComponentDim] = eqx.field(static=True)


def _catalog(**overrides):
    fields = {
        "names": ("h2", "o2"),
        "masses": jnp.asarray((2.0, 32.0)),
        "composition": jnp.asarray(((2, 0), (0, 2)), dtype=jnp.int32),
        "basis": "nodal",
        "note": "free text is static-only",
        "count": 2,
    }
    fields.update(overrides)
    return Catalog(**fields)


def test_class_plans_check_contract_fields_in_declaration_order():
    pt.validate(_catalog())
    with pytest.raises(ValueError, match="masses"):
        pt.validate(_catalog(masses=jnp.asarray((2.0, 32.0, 18.0))))
    with pytest.raises(ValueError, match="count"):
        pt.validate(_catalog(count=3))
    with pytest.raises(TypeError, match="masses"):
        pt.validate(_catalog(masses=jnp.asarray((2, 32), dtype=jnp.int32), count=3))


def test_field_validation_requires_exact_literal_runtime_types():
    import numpy as np

    with pytest.raises(TypeError, match="basis"):
        pt.validate(_catalog(basis=np.str_("nodal")))
    with pytest.raises(ValueError, match="basis"):
        pt.validate(_catalog(basis="spectral"))


def test_classes_without_contract_fields_are_refused():
    class Plain(StrictModule):
        note: str = eqx.field(static=True)

    with pytest.raises(TypeError):
        pt.validate(Plain("x"))


def test_type_checking_only_contract_names_fail_with_field_context():
    class Hidden(StrictModule):
        values: CheckerOnlyFloat[ComponentDim]

    with pytest.raises(TypeError, match="Hidden.values"):
        pt.validate(Hidden(jnp.zeros((1,), dtype=jnp.float32)))


def test_unsupported_contract_placements_fail_with_field_context():
    class Misplaced(StrictModule):
        values: list[pt.Float64[ComponentDim]] = eqx.field(static=True)

    with pytest.raises(TypeError, match="Misplaced.values"):
        pt.validate(Misplaced([]))


def test_inherited_fields_resolve_in_their_defining_module():
    class AbstractHolder(StrictModule):
        masses: pt.Float64[ComponentDim]

    class Holder(AbstractHolder):
        count: pt.Size[ComponentDim] = eqx.field(static=True)

    pt.validate(Holder(jnp.zeros((2,)), 2))
    with pytest.raises(ValueError):
        pt.validate(Holder(jnp.zeros((2,)), 3))


def test_concurrent_first_validation_compiles_one_consistent_plan():
    class Concurrent(StrictModule):
        masses: pt.Float64[ComponentDim]

    instance = Concurrent(jnp.zeros((2,)))
    errors = []

    def run():
        try:
            pt.validate(instance)
        except Exception as error:  # collected for the assertion below
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


def test_plan_cache_does_not_retain_dynamically_defined_classes():
    def define():
        @dataclasses.dataclass(frozen=True)
        class Transient:
            masses: pt.Float64[ComponentDim]

        pt.validate(Transient(jnp.zeros((1,))))
        return weakref.ref(Transient)

    reference = define()
    gc.collect()
    assert reference() is None
