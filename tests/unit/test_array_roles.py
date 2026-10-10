import dataclasses
from typing import Any

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

    def __call__(self, x: Any) -> Any:
        return self.weight * (x - self.shift)


def _roles(tree: Any) -> Any:
    resolution = phx.resolve_array_roles(tree)
    return dict(zip(resolution.paths, resolution.roles, strict=True))


def _kinds(tree: Any) -> Any:
    return {(path, kind) for path, kind, _ in phx.resolve_array_roles(tree).violations}


def test_array_roles_scenario_1() -> None:
    domain = phx.domain.Interval1d(0.0, 1.0)
    terminal = Owner(
        jnp.ones(2),
        (
            Frozen(jnp.ones(3)),
            Freeze(Owner(jnp.ones(4))),
            domain,
        ),
    )
    terminal_roles = _roles(terminal)
    assert terminal_roles[".weight"] is ArrayRole.PARAMETER
    assert terminal_roles[".child[0].value"] is ArrayRole.FIXED
    assert terminal_roles[".child[1].value.weight"] is ArrayRole.FIXED
    domain_paths = [path for path in terminal_roles if path.startswith(".child[2]")]
    assert domain_paths
    assert {terminal_roles[path] for path in domain_paths} == {ArrayRole.FIXED}
    assert not phx.resolve_array_roles(terminal).violations

    declared = Declared(
        parameters={"a": [jnp.ones(2)], "b": (jnp.ones(1), jnp.arange(2))},
        fixed=Owner(jnp.ones(3)),
        state={"mean": jnp.zeros(2), "count": jnp.zeros((), jnp.int32)},
    )
    declared_roles = _roles(declared)
    assert declared_roles[".parameters['a'][0]"] is ArrayRole.PARAMETER
    assert declared_roles[".parameters['b'][0]"] is ArrayRole.PARAMETER
    assert declared_roles[".parameters['b'][1]"] is ArrayRole.FIXED
    assert declared_roles[".fixed.weight"] is ArrayRole.FIXED
    assert declared_roles[".state['mean']"] is ArrayRole.MODEL_STATE
    assert declared_roles[".state['count']"] is ArrayRole.MODEL_STATE

    nested = Declared(
        parameters=StateHolder(jnp.zeros(2)),
        fixed=ParameterHolder(jnp.ones(2)),
        state=None,
    )
    nested_roles = _roles(nested)
    assert nested_roles[".parameters.value"] is ArrayRole.MODEL_STATE
    assert nested_roles[".fixed.value"] is ArrayRole.PARAMETER
    terminal_conflicts = Neutral(
        (
            ParameterHolder(Frozen(jnp.ones(2))),
            StateHolder(Freeze(jnp.ones(2))),
            Declared(fixed=Frozen(jnp.ones(2))),
        )
    )
    assert _kinds(terminal_conflicts) == {
        (".value[0].value", "role-field-on-terminal-value"),
        (".value[1].value", "role-field-on-terminal-value"),
    }
    assert set(_roles(terminal_conflicts).values()) == {ArrayRole.FIXED}

    inherited = Owner(
        jnp.ones(2),
        {
            "neutral": Neutral([jnp.ones(3), jnp.arange(3)]),
            "declared": DeclaredOwner(jnp.ones(1), (jnp.ones(2),)),
        },
    )
    inherited_roles = _roles(inherited)
    assert inherited_roles[".weight"] is ArrayRole.PARAMETER
    assert inherited_roles[".child['neutral'].value[0]"] is ArrayRole.PARAMETER
    assert inherited_roles[".child['neutral'].value[1]"] is ArrayRole.FIXED
    assert inherited_roles[".child['declared'].weight"] is ArrayRole.PARAMETER
    assert inherited_roles[".child['declared'].data[0]"] is ArrayRole.FIXED

    fixed_ancestor = {
        "direct": Plain(Owner(jnp.ones(2))),
        "nested": Plain(Plain([Owner(jnp.ones(2))])),
        "declared": Plain(ParameterHolder(jnp.ones(2))),
    }
    assert _kinds(fixed_ancestor) == {
        ("['direct'].value", "parameter-under-fixed-ancestor"),
        ("['nested'].value.value[0]", "parameter-under-fixed-ancestor"),
        ("['declared'].value.value", "parameter-under-fixed-ancestor"),
    }
    assert set(_roles(fixed_ancestor).values()) == {ArrayRole.FIXED}

    explicitly_frozen = Plain(
        (
            Freeze(Owner(jnp.ones(2), Plain(Owner(jnp.ones(1))))),
            AnalyticOwner(jnp.ones(3)),
        )
    )
    resolution = phx.resolve_array_roles(explicitly_frozen)
    assert resolution.violations == ()
    assert set(resolution.roles) == {ArrayRole.FIXED}
    tree = Neutral({"weight": jnp.ones(2), "index": jnp.arange(2), "rate": 0.5})
    resolution = phx.resolve_array_roles(tree)
    assert resolution.unclassified == (".value['weight']",)
    assert resolution.role_of(".value['weight']") is None
    assert resolution.role_of(".value['index']") is ArrayRole.FIXED
    assert resolution.role_of(".value['rate']") is ArrayRole.FIXED

    for field in (phx.parameter_field, phx.fixed_field, phx.model_state_field):
        with pytest.raises(TypeError, match="static"):
            field(static=True)
    ordered = Owner(
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
    resolution = phx.resolve_array_roles(ordered)
    expected_paths = tuple(
        jax.tree_util.keystr(path)
        for path, _ in jax.tree_util.tree_flatten_with_path(ordered)[0]
    )
    assert resolution.paths == expected_paths
    parameters, _ = eqx.partition(
        ordered,
        resolution.filter_spec(ArrayRole.PARAMETER),
    )
    assert [leaf.shape for leaf in jax.tree_util.tree_leaves(parameters)] == [
        (2,),
        (1,),
    ]

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


def test_array_roles_scenario_2() -> None:
    with pytest.raises(ValueError, match="unclassified|Unclassified"):
        phx.partition_parameters(Neutral(jnp.ones(2)))
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
    captured_array = _training_callable(_SlottedCoefficients(jnp.ones(2)))
    with pytest.raises(ValueError) as error:
        phx.require_parameter_roles(captured_array, context="unit training")
    message = str(error.value)
    assert message.startswith("unit training:")
    assert ".value: closure variable 'coefficients' -> attribute 'scale'" in message

    captured_scalar = _training_callable(_SlottedCoefficients(2.0))
    assert phx.require_parameter_roles(captured_scalar, context="x").unclassified == ()


@dataclasses.dataclass(frozen=True, slots=True)
class _SlottedCoefficients:
    scale: object
    label: str = "coefficients"


def _training_callable(coefficients: Any) -> Any:
    def loss(x: Any) -> Any:
        return coefficients.scale * x

    return Neutral(loss)


_GLOBAL_WEIGHTS = jnp.array([1.0, 2.0])
_GLOBAL_TABLE = {"weights": (jnp.array([3.0]),)}
_GLOBAL_SCALE = 2.0


def _reads_global_weights(x: Any) -> Any:
    return x * _GLOBAL_WEIGHTS


def _reads_global_table(x: Any) -> Any:
    return x * _GLOBAL_TABLE["weights"][0]


def _calls_global_reader(x: Any) -> Any:
    return _reads_global_weights(x) + 1.0


def _reads_scalar_globals(x: Any) -> Any:
    return jnp.sin(x) * _GLOBAL_SCALE


class _StaticPlan(phx.StrictModule, phx.NonTrainableState):
    fn: object = eqx.field(static=True)


class _ExplicitStaticPlan(phx.StrictModule, phx.ExplicitFreeze):
    fn: object = eqx.field(static=True)


class _PlanOwner(phx.StrictModule, phx.ParameterOwner):
    weight: jax.Array
    plan: object = phx.fixed_field()


def test_array_roles_scenario_3() -> None:
    cases = (
        (_reads_global_weights, "global '_GLOBAL_WEIGHTS'"),
        (_reads_global_table, "global '_GLOBAL_TABLE' -> ['weights']"),
        (_calls_global_reader, "global '_reads_global_weights' -> global"),
    )
    for function, route in cases:
        tree = _PlanOwner(jnp.ones(2), _StaticPlan(function))
        with pytest.raises(ValueError) as error:
            phx.require_parameter_roles(tree, context="unit training")
        assert f".plan.fn: static field -> {route}" in str(error.value)

    captured = jnp.ones(2)
    hidden = _PlanOwner(jnp.ones(2), _StaticPlan(lambda x: x * captured))
    with pytest.raises(ValueError, match="closure variable 'captured'"):
        phx.require_parameter_roles(hidden, context="unit training")

    for function in (_reads_global_weights, lambda x: x * captured):
        explicitly_frozen = _PlanOwner(
            jnp.ones(2),
            _ExplicitStaticPlan(function),
        )
        assert (
            phx.require_parameter_roles(
                explicitly_frozen,
                context="x",
            ).unclassified
            == ()
        )
    scalar_globals = _PlanOwner(jnp.ones(2), _StaticPlan(_reads_scalar_globals))
    assert phx.require_parameter_roles(scalar_globals, context="x").unclassified == ()
    model = _stacked_members()
    inputs = jnp.linspace(0.0, 1.0, 3)
    mapped = phx.LaneLayout.from_predicate(
        (model, inputs),
        lambda path: path.startswith("[0]"),
        kind="member",
    )
    assert mapped.mapped_paths == ("[0].shift", "[0].weight")
    assert phx.resolve_array_roles(model).role_of(".shift") is ArrayRole.FIXED
    mapped_axes = mapped.in_axes((model, inputs))
    mapped_output = eqx.filter_vmap(
        lambda member, value: member(value),
        in_axes=mapped_axes,
    )(model, inputs)
    assert jnp.allclose(mapped_output, _serial(model, inputs, mapped))

    shared = eqx.tree_at(lambda member: member.weight, model, jnp.full(3, 2.0))
    batched_inputs = jnp.stack([jnp.linspace(0.0, 1.0, 3) + index for index in range(4)])
    shared_layout = phx.LaneLayout("item", ("[0].shift", "[1]"))
    shared_axes = shared_layout.in_axes((shared, batched_inputs))
    shared_output = eqx.filter_vmap(
        lambda member, value: member(value),
        in_axes=shared_axes,
    )(shared, batched_inputs)
    assert shared_axes[0].weight is None
    assert jnp.allclose(
        shared_output,
        _serial(shared, batched_inputs, shared_layout),
    )

    forward = phx.LaneLayout("item", ("[0].shift", "[1]", "[0].weight"))
    permuted = phx.LaneLayout("item", ("[1]", "[0].weight", "[0].shift"))
    assert forward == permuted
    assert hash(forward) == hash(permuted)
    assert forward.mapped_paths == ("[0].shift", "[0].weight", "[1]")

    invalid_tree = {
        "a": jnp.ones((3, 2)),
        "b": jnp.ones((2, 2)),
        "c": jnp.ones(()),
    }
    with pytest.raises(ValueError, match="share one lane size"):
        phx.LaneLayout("case", ("['a']", "['b']")).in_axes(invalid_tree)
    with pytest.raises(ValueError, match="Unknown case lane"):
        phx.LaneLayout("case", ("['missing']",)).in_axes(invalid_tree)
    with pytest.raises(ValueError, match="leading lane axis"):
        phx.LaneLayout("case", ("['c']",)).in_axes(invalid_tree)
    with pytest.raises(ValueError, match="kind"):
        # ty: ignore[invalid-argument-type]
        phx.LaneLayout("batch", ("['a']",))
    declared = Owner(
        jnp.ones(2),
        (
            DeclaredOwner(jnp.ones(3), jnp.ones(4)),
            Frozen(jnp.ones(5)),
            Declared(state=jnp.zeros(1)),
        ),
    )
    assert ParameterSubspace.array_leaf_paths(declared) == (
        ".weight",
        ".child[0].weight",
    )
    for path in (".child[0].data", ".child[1].value", ".child[2].state"):
        with pytest.raises(ValueError, match="cannot select FIXED or MODEL_STATE"):
            ParameterSubspace.from_leaf_paths(declared, (path,))
    with pytest.raises(ValueError, match="cannot select FIXED or MODEL_STATE"):
        ParameterSubspace.from_subtree_paths(declared, (".child[1]",))
    with pytest.raises(ValueError, match="cannot select FIXED or MODEL_STATE"):
        spec = jax.tree_util.tree_map(lambda _: False, declared)
        ParameterSubspace(
            declared,
            eqx.tree_at(lambda tree: tree.child[0].data, spec, True),
        )

    tree = Neutral(
        {
            "raw": jnp.ones(2),
            "other": jnp.ones(3),
            "owner": DeclaredOwner(jnp.ones(1), jnp.ones(4)),
            "frozen": Frozen(jnp.ones(5)),
        }
    )
    subspace = ParameterSubspace.from_subtree_paths(
        tree,
        (".value['raw']", ".value['owner']"),
    )
    assert subspace.leaf_paths == (".value['owner'].weight", ".value['raw']")
    assert subspace.initial.value["frozen"] is None
    assert subspace.frozen.value["other"].shape == (3,)
    moved = subspace.reconstruct_vector(jnp.zeros(subspace.total_dimension))
    assert jnp.all(moved.value["raw"] == 0.0)
    assert jnp.all(moved.value["other"] == 1.0)
    assert jnp.all(moved.value["owner"].data == 1.0)
    # ty: ignore[not-subscriptable]
    assert eqx.tree_equal(moved.value["frozen"], tree.value["frozen"])
    everything = ParameterSubspace(tree, eqx.is_inexact_array)
    assert everything.leaf_paths == (
        ".value['other']",
        ".value['owner'].weight",
        ".value['raw']",
    )


def _stacked_members() -> Any:
    members = [
        Normalized(jnp.full(3, 1.0 + index), jnp.full(3, float(index)))
        for index in range(4)
    ]
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *members)


def _serial(model: Any, inputs: Any, layout: Any) -> Any:
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


def _closure_converted(weights: jax.Array) -> Any:
    return eqx.filter_closure_convert(lambda x: x * weights, jnp.zeros(2))


def test_closure_converted_constants_are_visible_unless_held_statically() -> None:
    converted = _closure_converted(jnp.full(2, 3.0))
    visible = _PlanOwner(jnp.ones(2), converted)
    assert phx.require_parameter_roles(visible, context="x").unclassified == ()

    hidden = _PlanOwner(jnp.ones(2), _StaticPlan(converted))
    with pytest.raises(ValueError, match="jaxpr constant 0"):
        phx.require_parameter_roles(hidden, context="unit training")
