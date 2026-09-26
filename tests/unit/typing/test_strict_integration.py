import copy

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


class NodeDim(pt.Dim, minimum=1):
    pass


class ComponentDim(pt.Dim):
    pass


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


def test_construction_accepts_and_refuses_structural_contracts():
    field = Field(jnp.zeros((3,)), 3)

    assert field.count == 3
    with pytest.raises(ValueError):
        Field(jnp.zeros((3,)), 4)
    with pytest.raises(TypeError):
        Field(jnp.zeros((3,), dtype=jnp.float32), 3)
    with pytest.raises(TypeError):
        Field(np.zeros((3,)), 3)


def test_structural_checks_add_no_equations_and_no_host_transfers():
    def build(opted):
        def run(values):
            module = Field(values, 3) if opted else Twin(values, 3)
            return module.values

        return run

    opted = jax.make_jaxpr(build(True))(jnp.zeros((3,)))
    twin = jax.make_jaxpr(build(False))(jnp.zeros((3,)))
    assert len(opted.jaxpr.eqns) == len(twin.jaxpr.eqns) == 0
    with jax.transfer_guard_device_to_host("disallow"):
        Field(jnp.zeros((3,)), 3)


def test_leaves_and_static_structure_match_an_unchecked_twin():
    values = jnp.arange(3.0)
    opted = jax.tree_util.tree_flatten(Field(values, 3))
    twin = jax.tree_util.tree_flatten(Twin(values, 3))

    assert [leaf.shape for leaf in opted[0]] == [leaf.shape for leaf in twin[0]]
    assert opted[1].num_leaves == twin[1].num_leaves


def test_owner_check_init_runs_before_structural_checks():
    with pytest.raises(RuntimeError, match="owner invariant"):
        Checked(jnp.asarray(1.0))


def test_the_first_failing_field_in_declaration_order_is_reported():
    with pytest.raises(ValueError, match="right"):
        Pair(jnp.zeros((2,)), jnp.zeros((3,)))
    with pytest.raises(TypeError, match="left"):
        Pair(jnp.zeros((2,), dtype=jnp.int32), jnp.zeros((5, 5, 5)))


def test_failed_union_alternatives_roll_back_their_bindings():
    pair = Pair(jnp.zeros((2,)), jnp.zeros((2, 4)))

    assert pair.right.shape == (2, 4)


def test_nested_modules_bind_dimensions_in_independent_scopes():
    outer = Outer(Field(jnp.zeros((5,)), 5), jnp.zeros((2,)))

    pt.validate(outer)


def test_abstract_opt_in_is_inherited_by_concrete_modules():
    Concrete(jnp.zeros((2,)))
    with pytest.raises(TypeError):
        Concrete(jnp.zeros((2,), dtype=jnp.int32))
    with pytest.raises(TypeError):
        AbstractOptedIn()


def test_transformations_do_not_validate_but_explicit_validation_does():
    field = Field(jnp.zeros((3,)), 3)
    widened = eqx.tree_at(lambda module: module.values, field, jnp.zeros((4,)))
    doubled = jax.tree_util.tree_map(lambda leaf: 2.0 * leaf, field)

    pt.validate(doubled)
    with pytest.raises(ValueError):
        pt.validate(widened)
    outer = Outer(Field(jnp.zeros((5,)), 5), jnp.zeros((2,)))
    with pytest.raises(ValueError, match="Field.values"):
        pt.validate(eqx.tree_at(lambda module: module.inner, outer, widened))


def test_filter_jit_and_filter_vmap_check_at_trace_time():
    @eqx.filter_jit
    def build(values):
        return Field(values, 3)

    assert build(jnp.zeros((3,))).values.shape == (3,)
    with pytest.raises(ValueError):
        build(jnp.zeros((4,)))

    stacked = eqx.filter_vmap(lambda values: Field(values, 3))(jnp.zeros((2, 3)))
    assert stacked.values.shape == (2, 3)
    with pytest.raises(ValueError):
        pt.validate(stacked)


def test_array_recipe_round_trip_validates_the_restored_model():
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

    # A self-consistent recipe whose mass extent disagrees with the static
    # component count restores structurally and is refused by its contract.
    tampered_recipe = copy.deepcopy(recipe)
    tampered_recipe["fields"]["molar_masses"]["shape"] = [3]
    tampered_arrays = dict(arrays)
    tampered_arrays["model/000000"] = np.ones((3,), dtype=np.float64)
    with pytest.raises(ValueError, match="molar_masses"):
        model_from_array_recipe(tampered_recipe, tampered_arrays, prefix="model")
