from __future__ import annotations

import threading
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
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
    __strict_contract__ = True

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


def test_contract_fields_are_checked_in_declaration_order():
    pt.validate(_catalog())
    with pytest.raises(ValueError, match="masses"):
        _catalog(masses=jnp.asarray((2.0, 32.0, 18.0)))
    with pytest.raises(ValueError, match="count"):
        _catalog(count=3)
    with pytest.raises(TypeError, match="masses"):
        _catalog(masses=jnp.asarray((2, 32), dtype=jnp.int32), count=3)


@pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
def test_field_validation_requires_exact_literal_runtime_types():
    with pytest.raises(TypeError, match="basis"):
        _catalog(basis=np.str_("nodal"))
    with pytest.raises(ValueError, match="basis"):
        _catalog(basis="spectral")


def test_validate_requires_an_opted_in_strict_module():
    class Plain(StrictModule):
        masses: pt.Float64[ComponentDim]

    with pytest.raises(TypeError):
        pt.validate(Plain(jnp.zeros((1,))))
    with pytest.raises(TypeError):
        pt.validate(object())


def test_opt_in_without_contract_fields_is_refused():
    class Plain(StrictModule):
        __strict_contract__ = True

        note: str = eqx.field(static=True)

    with pytest.raises(TypeError):
        Plain("x")


def test_type_checking_only_contract_names_fail_with_field_context():
    class Hidden(StrictModule):
        __strict_contract__ = True

        values: CheckerOnlyFloat[ComponentDim]

    with pytest.raises(TypeError, match="Hidden.values"):
        Hidden(jnp.zeros((1,), dtype=jnp.float32))


def test_unsupported_contract_placements_fail_with_field_context():
    class Misplaced(StrictModule):
        __strict_contract__ = True

        values: list[pt.Float64[ComponentDim]] = eqx.field(static=True)

    with pytest.raises(TypeError, match="Misplaced.values"):
        Misplaced([])


def test_inherited_fields_resolve_in_their_defining_module():
    class AbstractHolder(StrictModule):
        __strict_contract__ = True

        masses: pt.Float64[ComponentDim]

    class Holder(AbstractHolder):
        count: pt.Size[ComponentDim] = eqx.field(static=True)

    pt.validate(Holder(jnp.zeros((2,)), 2))
    with pytest.raises(ValueError):
        Holder(jnp.zeros((2,)), 3)


def test_concurrent_first_construction_compiles_one_consistent_plan():
    class Concurrent(StrictModule):
        __strict_contract__ = True

        masses: pt.Float64[ComponentDim]

    values = jnp.zeros((2,))
    errors = []

    def run():
        try:
            Concurrent(values)
        except Exception as error:  # collected for the assertion below
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
