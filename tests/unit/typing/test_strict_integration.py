from __future__ import annotations

import copy
import threading
from dataclasses import fields
from typing import Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.typing as pt
from phydrax import StrictModule
from phydrax._model._structure import (
    model_from_array_recipe,
    model_structure_recipe,
    pack_model_array_tree,
)
from phydrax.equations import ChemicalComponentCatalog
from tests._support.assertions import assert_tree_equal


pytestmark = pytest.mark.strict_jax


if TYPE_CHECKING:
    from phydrax.typing import Float32 as CheckerOnlyFloat


class NodeDim(pt.Dim, minimum=1):
    pass


class ComponentDim(pt.Dim):
    pass


class ElementDim(pt.Dim):
    pass


type Basis = Literal["nodal", "modal"]


class Field(StrictModule):
    __strict_contract__ = True

    values: pt.Float64[NodeDim]
    count: pt.Size[NodeDim] = eqx.field(static=True)


class Twin(StrictModule):
    values: jax.Array
    count: int = eqx.field(static=True)


class Checked(StrictModule):
    __strict_contract__ = True

    values: pt.Float64[NodeDim]

    def __check_init__(self) -> None:
        if self.values.ndim == 0:
            raise RuntimeError("owner invariant")


class Pair(StrictModule):
    __strict_contract__ = True

    left: pt.Float64[NodeDim]
    right: pt.Float64[NodeDim] | pt.Float64[NodeDim, ComponentDim]


class Outer(StrictModule):
    __strict_contract__ = True

    inner: Field
    values: pt.Float64[NodeDim]


class AbstractOptedIn(StrictModule):
    __strict_contract__ = True

    values: eqx.AbstractVar[pt.Float64[NodeDim]]


class Concrete(AbstractOptedIn):
    values: pt.Float64[NodeDim]


class Catalog(StrictModule):
    __strict_contract__ = True

    names: pt.Identifiers[ComponentDim] = eqx.field(static=True)
    masses: pt.Float64[ComponentDim]
    composition: pt.Int32[ElementDim, ComponentDim]
    basis: Basis = eqx.field(static=True)
    note: str = eqx.field(static=True)
    count: pt.Size[ComponentDim] = eqx.field(static=True)


def _catalog(
    *,
    names: tuple[str, ...] = ("h2", "o2"),
    masses: jax.Array | None = None,
    composition: jax.Array | None = None,
    basis: Basis = "nodal",
    note: str = "free text is static-only",
    count: int = 2,
) -> Catalog:
    if masses is None:
        masses = jnp.asarray((2.0, 32.0))
    if composition is None:
        composition = jnp.asarray(((2, 0), (0, 2)), dtype=jnp.int32)
    return Catalog(names, masses, composition, basis, note, count)


@pytest.mark.filterwarnings("ignore:A JAX array is being set as static")
def test_construction_fails_in_declaration_order_and_keeps_literal_runtime_types() -> (
    None
):
    field = Field(jnp.zeros((3,)), 3)
    assert field.count == 3
    with pytest.raises(ValueError):
        Field(jnp.zeros((3,)), 4)
    with pytest.raises(TypeError):
        Field(jnp.zeros((3,), dtype=jnp.float32), 3)
    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        Field(np.zeros((3,)), 3)

    pt.validate(_catalog())
    with pytest.raises(ValueError, match="masses"):
        _catalog(masses=jnp.asarray((2.0, 32.0, 18.0)))
    with pytest.raises(ValueError, match="count"):
        _catalog(count=3)
    with pytest.raises(TypeError, match="masses"):
        _catalog(masses=jnp.asarray((2, 32), dtype=jnp.int32), count=3)
    with pytest.raises(TypeError, match="basis"):
        # ty: ignore[invalid-argument-type]
        _catalog(basis=np.str_("nodal"))
    with pytest.raises(ValueError, match="basis"):
        # ty: ignore[invalid-argument-type]
        _catalog(basis="spectral")


def test_strict_integration_scenario_1() -> None:
    class Plain(StrictModule):
        masses: pt.Float64[ComponentDim]

    with pytest.raises(TypeError):
        pt.validate(Plain(jnp.zeros((1,))))
    with pytest.raises(TypeError):
        pt.validate(object())

    class EmptyOptIn(StrictModule):
        __strict_contract__ = True

        note: str = eqx.field(static=True)

    with pytest.raises(TypeError):
        EmptyOptIn("x")

    DirectAnnotation = type(
        "DirectAnnotation",
        (StrictModule,),
        {
            "__module__": __name__,
            "__strict_contract__": True,
            "__annotations__": {"values": pt.Float64[ComponentDim]},
        },
    )
    # ty: ignore[too-many-positional-arguments]
    pt.validate(DirectAnnotation(jnp.zeros((2,))))

    class Hidden(StrictModule):
        __strict_contract__ = True

        values: CheckerOnlyFloat[ComponentDim]

    with pytest.raises(TypeError, match="Hidden.values"):
        Hidden(jnp.zeros((1,), dtype=jnp.float32))

    class Misplaced(StrictModule):
        __strict_contract__ = True

        values: list[pt.Float64[ComponentDim]] = eqx.field(static=True)

    with pytest.raises(TypeError, match="Misplaced.values"):
        Misplaced([])

    class AbstractHolder(StrictModule):
        __strict_contract__ = True

        masses: pt.Float64[ComponentDim]

    class Holder(AbstractHolder):
        count: pt.Size[ComponentDim] = eqx.field(static=True)

    pt.validate(Holder(jnp.zeros((2,)), 2))
    with pytest.raises(ValueError):
        Holder(jnp.zeros((2,)), 3)
    with pytest.raises(ValueError, match="right"):
        Pair(jnp.zeros((2,)), jnp.zeros((3,)))
    with pytest.raises(TypeError, match="left"):
        Pair(jnp.zeros((2,), dtype=jnp.int32), jnp.zeros((5, 5, 5)))
    pair = Pair(jnp.zeros((2,)), jnp.zeros((2, 4)))
    assert pair.right.shape == (2, 4)

    outer = Outer(Field(jnp.zeros((5,)), 5), jnp.zeros((2,)))
    pt.validate(outer)

    Concrete(jnp.zeros((2,)))
    with pytest.raises(TypeError):
        Concrete(jnp.zeros((2,), dtype=jnp.int32))
    with pytest.raises(TypeError):
        # ty: ignore[missing-argument]
        AbstractOptedIn()
    catalog = ChemicalComponentCatalog(
        ("H2", "O2"),
        np.asarray((2.016, 31.998)),
        ("H", "O"),
        np.asarray(((2, 0), (0, 2))),
    )
    recipe = model_structure_recipe(catalog)
    arrays = pack_model_array_tree(catalog, recipe, prefix="model")

    restored = model_from_array_recipe(recipe, arrays, prefix="model")
    assert restored.catalog_id == catalog.catalog_id
    np.testing.assert_array_equal(restored.molar_masses, catalog.molar_masses)

    tampered_recipe = copy.deepcopy(recipe)
    molar_masses = next(
        index
        for index, field in enumerate(fields(type(catalog)))
        if field.name == "molar_masses"
    )
    tampered_recipe["items"][molar_masses]["shape"] = [3]
    tampered_arrays = dict(arrays)
    tampered_arrays["model/000000"] = np.ones((3,), dtype=np.float64)
    with pytest.raises(ValueError, match="molar_masses"):
        model_from_array_recipe(tampered_recipe, tampered_arrays, prefix="model")


def test_structural_validation_has_no_numerical_equations_or_extra_leaves() -> None:
    def build_opted(values: jax.Array) -> jax.Array:
        return Field(values, 3).values

    def build_twin(values: jax.Array) -> jax.Array:
        return Twin(values, 3).values

    opted_jaxpr = jax.make_jaxpr(build_opted)(jnp.zeros((3,)))
    twin_jaxpr = jax.make_jaxpr(build_twin)(jnp.zeros((3,)))
    assert len(opted_jaxpr.jaxpr.eqns) == len(twin_jaxpr.jaxpr.eqns) == 0

    values = jnp.arange(3.0)
    opted_leaves, opted_tree = jax.tree_util.tree_flatten(Field(values, 3))
    twin_leaves, twin_tree = jax.tree_util.tree_flatten(Twin(values, 3))
    assert [leaf.shape for leaf in opted_leaves] == [leaf.shape for leaf in twin_leaves]
    assert opted_tree.num_leaves == twin_tree.num_leaves

    with jax.transfer_guard_device_to_host("disallow"):
        Field(jnp.zeros((3,)), 3)
    with pytest.raises(RuntimeError, match="owner invariant"):
        Checked(jnp.asarray(1.0))


def test_transformations_require_explicit_validation_and_trace_at_construction() -> None:
    field = Field(jnp.zeros((3,)), 3)
    widened = eqx.tree_at(lambda module: module.values, field, jnp.zeros((4,)))
    doubled = jax.tree_util.tree_map(lambda leaf: 2.0 * leaf, field)
    pt.validate(doubled)
    with pytest.raises(ValueError):
        pt.validate(widened)

    outer = Outer(Field(jnp.zeros((5,)), 5), jnp.zeros((2,)))
    with pytest.raises(ValueError, match="Field.values"):
        pt.validate(eqx.tree_at(lambda module: module.inner, outer, widened))

    @eqx.filter_jit
    def build(values: jax.Array) -> Field:
        return Field(values, 3)

    assert build(jnp.zeros((3,))).values.shape == (3,)
    with pytest.raises(ValueError):
        build(jnp.zeros((4,)))

    stacked = eqx.filter_vmap(lambda values: Field(values, 3))(jnp.zeros((2, 3)))
    assert stacked.values.shape == (2, 3)
    with pytest.raises(ValueError):
        pt.validate(stacked)


def test_concurrent_first_construction_compiles_one_consistent_plan() -> None:
    class Concurrent(StrictModule):
        __strict_contract__ = True

        masses: pt.Float64[ComponentDim]

    values = jnp.zeros((2,))
    errors: list[Exception] = []

    def run() -> None:
        try:
            Concurrent(values)
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []

    assert_tree_equal(Concurrent(values), Concurrent(values))
