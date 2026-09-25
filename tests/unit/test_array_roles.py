import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax import ArrayRole
from phydrax.nn.parameters import ParameterSubspace


class Frozen(phx.StrictModule, phx.NonTrainableState):
    value: jax.Array


class Freeze(phx.StrictModule, phx.ExplicitFreeze):
    value: object


class Plain(phx.StrictModule, phx.NonTrainableState):
    value: object


class Owner(phx.StrictModule, phx.ParameterOwner):
    weight: jax.Array
    child: object = None


class AnalyticOwner(phx.StrictModule, phx.ParameterOwner, phx.NonTrainableState):
    coefficient: jax.Array


class Neutral(phx.StrictModule):
    value: object


class Declared(phx.StrictModule):
    parameters: object = phx.parameter_field(default=None)
    fixed: object = phx.fixed_field(default=None)
    state: object = phx.model_state_field(default=None)


class DeclaredOwner(phx.StrictModule, phx.ParameterOwner):
    weight: jax.Array
    data: object = phx.fixed_field()


class ParameterHolder(phx.StrictModule):
    value: object = phx.parameter_field()


class StateHolder(phx.StrictModule):
    value: object = phx.model_state_field()


class Normalized(phx.StrictModule, phx.ParameterOwner):
    weight: jax.Array
    shift: jax.Array = phx.fixed_field()

    def __call__(self, x):
        return self.weight * (x - self.shift)


def _roles(tree):
    resolution = phx.resolve_array_roles(tree)
    return dict(zip(resolution.paths, resolution.roles, strict=True))


def _kinds(tree):
    return {(path, kind) for path, kind, _ in phx.resolve_array_roles(tree).violations}


def test_rule_1_terminal_nodes_freeze_whole_subtrees():
    domain = phx.domain.Interval1d(0.0, 1.0)
    tree = Owner(
        jnp.ones(2),
        (
            Frozen(jnp.ones(3)),
            Freeze(Owner(jnp.ones(4))),
            domain,
        ),
    )

    roles = _roles(tree)

    assert roles[".weight"] is ArrayRole.PARAMETER
    assert roles[".child[0].value"] is ArrayRole.FIXED
    assert roles[".child[1].value.weight"] is ArrayRole.FIXED
    domain_paths = [path for path in roles if path.startswith(".child[2]")]
    assert domain_paths
    assert {roles[path] for path in domain_paths} == {ArrayRole.FIXED}
    assert not phx.resolve_array_roles(tree).violations


def test_rule_2_explicit_field_roles_apply_to_value_subtrees():
    tree = Declared(
        parameters={"a": [jnp.ones(2)], "b": (jnp.ones(1), jnp.arange(2))},
        fixed=Owner(jnp.ones(3)),
        state={"mean": jnp.zeros(2), "count": jnp.zeros((), jnp.int32)},
    )

    roles = _roles(tree)

    assert roles[".parameters['a'][0]"] is ArrayRole.PARAMETER
    assert roles[".parameters['b'][0]"] is ArrayRole.PARAMETER
    assert roles[".parameters['b'][1]"] is ArrayRole.FIXED
    assert roles[".fixed.weight"] is ArrayRole.FIXED
    assert roles[".state['mean']"] is ArrayRole.MODEL_STATE
    assert roles[".state['count']"] is ArrayRole.MODEL_STATE


def test_nearest_explicit_field_role_wins():
    tree = Declared(
        parameters=StateHolder(jnp.zeros(2)),
        fixed=ParameterHolder(jnp.ones(2)),
        state=None,
    )

    roles = _roles(tree)

    assert roles[".parameters.value"] is ArrayRole.MODEL_STATE
    assert roles[".fixed.value"] is ArrayRole.PARAMETER


def test_rule_3_parameter_or_state_field_on_terminal_value_is_a_violation():
    tree = Neutral(
        (
            ParameterHolder(Frozen(jnp.ones(2))),
            StateHolder(Freeze(jnp.ones(2))),
            Declared(fixed=Frozen(jnp.ones(2))),
        )
    )

    assert _kinds(tree) == {
        (".value[0].value", "role-field-on-terminal-value"),
        (".value[1].value", "role-field-on-terminal-value"),
    }
    assert set(_roles(tree).values()) == {ArrayRole.FIXED}


def test_rule_4_containers_inherit_and_owners_default_to_parameter():
    tree = Owner(
        jnp.ones(2),
        {
            "neutral": Neutral([jnp.ones(3), jnp.arange(3)]),
            "declared": DeclaredOwner(jnp.ones(1), (jnp.ones(2),)),
        },
    )

    roles = _roles(tree)

    assert roles[".weight"] is ArrayRole.PARAMETER
    assert roles[".child['neutral'].value[0]"] is ArrayRole.PARAMETER
    assert roles[".child['neutral'].value[1]"] is ArrayRole.FIXED
    assert roles[".child['declared'].weight"] is ArrayRole.PARAMETER
    assert roles[".child['declared'].data[0]"] is ArrayRole.FIXED


def test_rule_5_parameter_owner_below_plain_non_trainable_state_is_a_violation():
    tree = {
        "direct": Plain(Owner(jnp.ones(2))),
        "nested": Plain(Plain([Owner(jnp.ones(2))])),
        "declared": Plain(ParameterHolder(jnp.ones(2))),
    }

    assert _kinds(tree) == {
        ("['direct'].value", "parameter-under-fixed-ancestor"),
        ("['nested'].value.value[0]", "parameter-under-fixed-ancestor"),
        ("['declared'].value.value", "parameter-under-fixed-ancestor"),
    }
    assert set(_roles(tree).values()) == {ArrayRole.FIXED}


def test_rule_5_explicit_freeze_ends_the_audit():
    tree = Plain(
        (
            Freeze(Owner(jnp.ones(2), Plain(Owner(jnp.ones(1))))),
            AnalyticOwner(jnp.ones(3)),
        )
    )

    resolution = phx.resolve_array_roles(tree)

    assert resolution.violations == ()
    assert set(resolution.roles) == {ArrayRole.FIXED}


def test_rule_6_unmarked_inexact_arrays_are_unclassified():
    tree = Neutral({"weight": jnp.ones(2), "index": jnp.arange(2), "rate": 0.5})

    resolution = phx.resolve_array_roles(tree)

    assert resolution.unclassified == (".value['weight']",)
    assert resolution.role_of(".value['weight']") is None
    assert resolution.role_of(".value['index']") is ArrayRole.FIXED
    assert resolution.role_of(".value['rate']") is ArrayRole.FIXED


@pytest.mark.parametrize(
    "field", [phx.parameter_field, phx.fixed_field, phx.model_state_field]
)
def test_role_fields_cannot_be_static(field):
    with pytest.raises(TypeError, match="static"):
        field(static=True)


def test_resolution_order_matches_jax_leaves_and_equinox_partition():
    tree = Owner(
        jnp.ones(2),
        (
            Declared(
                parameters=[jnp.ones(1)],
                fixed={"b": jnp.ones(2), "a": jnp.ones(3)},
                state=jnp.zeros(1),
            ),
            Frozen(jnp.ones(4)),
            "label",
        ),
    )
    resolution = phx.resolve_array_roles(tree)

    expected = tuple(
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]
    )
    assert resolution.paths == expected
    parameters, _ = eqx.partition(tree, resolution.filter_spec(ArrayRole.PARAMETER))
    assert [leaf.shape for leaf in jax.tree_util.tree_leaves(parameters)] == [
        (2,),
        (1,),
    ]


def test_partition_and_combine_round_trip_all_role_lanes():
    tree = Owner(
        jnp.ones(2),
        (
            Declared(parameters=jnp.ones(1), fixed=jnp.ones(2), state=jnp.zeros(3)),
            Frozen(jnp.ones(4)),
            phx.domain.Interval1d(0.0, 1.0),
        ),
    )

    parameters, model_state, fixed = phx.partition_parameters(tree)

    assert [leaf.shape for leaf in jax.tree_util.tree_leaves(parameters)] == [
        (2,),
        (1,),
    ]
    assert [leaf.shape for leaf in jax.tree_util.tree_leaves(model_state)] == [(3,)]
    assert parameters.child[1] is None and model_state.child[2] is None
    assert isinstance(fixed.child[1], Frozen)
    assert eqx.tree_equal(phx.combine_parameters(parameters, model_state, fixed), tree)


def test_partition_rejects_undeclared_trees():
    with pytest.raises(ValueError, match="unclassified|Unclassified"):
        phx.partition_parameters(Neutral(jnp.ones(2)))


def test_require_parameter_roles_names_paths_and_remedies():
    tree = {
        "raw": Neutral(jnp.ones(2)),
        "silent": Plain(Owner(jnp.ones(2))),
    }

    with pytest.raises(ValueError) as error:
        phx.require_parameter_roles(tree, context="unit training")

    message = str(error.value)
    assert message.startswith("unit training:")
    assert "['raw'].value" in message
    assert "['silent'].value [parameter-under-fixed-ancestor]" in message
    for remedy in (
        "parameter_field",
        "fixed_field",
        "EquinoxModel",
        "ParameterSubspace",
        "ExplicitFreeze",
    ):
        assert remedy in message
    clean = Owner(jnp.ones(2))
    assert phx.require_parameter_roles(clean, context="x").unclassified == ()


@dataclasses.dataclass(frozen=True, slots=True)
class _SlottedCoefficients:
    scale: object
    label: str = "coefficients"


def _training_callable(coefficients):
    def loss(x):
        return coefficients.scale * x

    return Neutral(loss)


def test_training_callable_capturing_slotted_dataclass_array_is_rejected():
    tree = _training_callable(_SlottedCoefficients(jnp.ones(2)))

    with pytest.raises(ValueError) as error:
        phx.require_parameter_roles(tree, context="unit training")

    message = str(error.value)
    assert message.startswith("unit training:")
    assert ".value: closure variable 'coefficients' -> attribute 'scale'" in message


def test_training_callable_capturing_slotted_dataclass_scalars_is_accepted():
    tree = _training_callable(_SlottedCoefficients(2.0))

    assert phx.require_parameter_roles(tree, context="x").unclassified == ()


_GLOBAL_WEIGHTS = jnp.array([1.0, 2.0])
_GLOBAL_TABLE = {"weights": (jnp.array([3.0]),)}
_GLOBAL_SCALE = 2.0


def _reads_global_weights(x):
    return x * _GLOBAL_WEIGHTS


def _reads_global_table(x):
    return x * _GLOBAL_TABLE["weights"][0]


def _calls_global_reader(x):
    return _reads_global_weights(x) + 1.0


def _reads_scalar_globals(x):
    return jnp.sin(x) * _GLOBAL_SCALE


class _StaticPlan(phx.StrictModule, phx.NonTrainableState):
    fn: object = eqx.field(static=True)


class _ExplicitStaticPlan(phx.StrictModule, phx.ExplicitFreeze):
    fn: object = eqx.field(static=True)


class _PlanOwner(phx.StrictModule, phx.ParameterOwner):
    weight: jax.Array
    plan: object = phx.fixed_field()


@pytest.mark.parametrize(
    ("fn", "route"),
    [
        (_reads_global_weights, "global '_GLOBAL_WEIGHTS'"),
        (_reads_global_table, "global '_GLOBAL_TABLE' -> ['weights']"),
        (_calls_global_reader, "global '_reads_global_weights' -> global"),
    ],
)
def test_plain_function_reading_global_arrays_is_rejected(fn, route):
    tree = _PlanOwner(jnp.ones(2), _StaticPlan(fn))

    with pytest.raises(ValueError) as error:
        phx.require_parameter_roles(tree, context="unit training")

    assert f".plan.fn: static field -> {route}" in str(error.value)


def test_plain_nontrainable_state_does_not_hide_closure_arrays():
    captured = jnp.ones(2)
    tree = _PlanOwner(jnp.ones(2), _StaticPlan(lambda x: x * captured))

    with pytest.raises(ValueError, match="closure variable 'captured'"):
        phx.require_parameter_roles(tree, context="unit training")


def test_explicit_freeze_authorizes_hidden_provider_state():
    captured = jnp.ones(2)
    for fn in (_reads_global_weights, lambda x: x * captured):
        tree = _PlanOwner(jnp.ones(2), _ExplicitStaticPlan(fn))
        assert phx.require_parameter_roles(tree, context="x").unclassified == ()


def test_plain_function_reading_modules_and_scalar_globals_is_accepted():
    tree = _PlanOwner(jnp.ones(2), _StaticPlan(_reads_scalar_globals))

    assert phx.require_parameter_roles(tree, context="x").unclassified == ()


def _stacked_members():
    members = [
        Normalized(jnp.full(3, 1.0 + index), jnp.full(3, float(index)))
        for index in range(4)
    ]
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *members)


def _serial(model, inputs, layout):
    size = layout.lane_size((model, inputs))
    axes = layout.in_axes((model, inputs))
    outputs = []
    for index in range(size):
        member = jax.tree_util.tree_map(
            lambda leaf, axis: leaf if axis is None else leaf[index],
            (model, inputs),
            axes,
        )
        outputs.append(member[0](member[1]))
    return jnp.stack(outputs)


def test_lane_layout_maps_parameters_and_fixed_arrays():
    model = _stacked_members()
    inputs = jnp.linspace(0.0, 1.0, 3)
    layout = phx.LaneLayout.from_predicate(
        (model, inputs), lambda path: path.startswith("[0]"), kind="member"
    )

    assert layout.mapped_paths == ("[0].shift", "[0].weight")
    assert phx.resolve_array_roles(model).role_of(".shift") is ArrayRole.FIXED
    in_axes = layout.in_axes((model, inputs))
    vmapped = eqx.filter_vmap(lambda m, x: m(x), in_axes=in_axes)(model, inputs)

    assert jnp.allclose(vmapped, _serial(model, inputs, layout))


def test_lane_layout_shares_unmapped_parameters():
    model = _stacked_members()
    shared = eqx.tree_at(lambda m: m.weight, model, jnp.full(3, 2.0))
    inputs = jnp.stack([jnp.linspace(0.0, 1.0, 3) + index for index in range(4)])
    layout = phx.LaneLayout("item", ("[0].shift", "[1]"))

    in_axes = layout.in_axes((shared, inputs))
    vmapped = eqx.filter_vmap(lambda m, x: m(x), in_axes=in_axes)(shared, inputs)

    assert in_axes[0].weight is None
    assert jnp.allclose(vmapped, _serial(shared, inputs, layout))


def test_lane_layout_identity_ignores_declaration_order():
    forward = phx.LaneLayout("item", ("[0].shift", "[1]", "[0].weight"))
    permuted = phx.LaneLayout("item", ("[1]", "[0].weight", "[0].shift"))

    assert forward == permuted
    assert hash(forward) == hash(permuted)
    assert forward.mapped_paths == ("[0].shift", "[0].weight", "[1]")


def test_lane_layout_validates_paths_and_lane_sizes():
    tree = {"a": jnp.ones((3, 2)), "b": jnp.ones((2, 2)), "c": jnp.ones(())}

    with pytest.raises(ValueError, match="share one lane size"):
        phx.LaneLayout("case", ("['a']", "['b']")).in_axes(tree)
    with pytest.raises(ValueError, match="Unknown case lane"):
        phx.LaneLayout("case", ("['missing']",)).in_axes(tree)
    with pytest.raises(ValueError, match="leading lane axis"):
        phx.LaneLayout("case", ("['c']",)).in_axes(tree)
    with pytest.raises(ValueError, match="kind"):
        phx.LaneLayout("batch", ("['a']",))


def test_parameter_subspace_rejects_fixed_and_terminal_selection():
    tree = Owner(
        jnp.ones(2),
        (
            DeclaredOwner(jnp.ones(3), jnp.ones(4)),
            Frozen(jnp.ones(5)),
            Declared(state=jnp.zeros(1)),
        ),
    )

    assert ParameterSubspace.array_leaf_paths(tree) == (".weight", ".child[0].weight")
    for path in (".child[0].data", ".child[1].value", ".child[2].state"):
        with pytest.raises(ValueError, match="cannot select FIXED or MODEL_STATE"):
            ParameterSubspace.from_leaf_paths(tree, (path,))
    with pytest.raises(ValueError, match="cannot select FIXED or MODEL_STATE"):
        ParameterSubspace.from_subtree_paths(tree, (".child[1]",))
    with pytest.raises(ValueError, match="cannot select FIXED or MODEL_STATE"):
        spec = jax.tree_util.tree_map(lambda _: False, tree)
        ParameterSubspace(tree, eqx.tree_at(lambda t: t.child[0].data, spec, True))


def test_parameter_subspace_selection_declares_parameters_and_freezes_complement():
    tree = Neutral(
        {
            "raw": jnp.ones(2),
            "other": jnp.ones(3),
            "owner": DeclaredOwner(jnp.ones(1), jnp.ones(4)),
            "frozen": Frozen(jnp.ones(5)),
        }
    )

    subspace = ParameterSubspace.from_subtree_paths(
        tree, (".value['raw']", ".value['owner']")
    )

    assert subspace.leaf_paths == (".value['owner'].weight", ".value['raw']")
    assert subspace.initial.value["frozen"] is None
    assert subspace.frozen.value["other"].shape == (3,)
    moved = subspace.reconstruct_vector(jnp.zeros(subspace.total_dimension))
    assert jnp.all(moved.value["raw"] == 0.0)
    assert jnp.all(moved.value["other"] == 1.0)
    assert jnp.all(moved.value["owner"].data == 1.0)
    assert eqx.tree_equal(moved.value["frozen"], tree.value["frozen"])
    everything = ParameterSubspace(tree, eqx.is_inexact_array)
    assert everything.leaf_paths == (
        ".value['other']",
        ".value['owner'].weight",
        ".value['raw']",
    )
