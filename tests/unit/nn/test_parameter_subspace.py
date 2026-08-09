import equinox as eqx
import jax.numpy as jnp
import pytest

from phydrax.nn.parameters import ParameterSubspace


def test_parameter_subspace_selects_and_reconstructs_exact_leaves():
    tree = {
        "encoder": {"weight": jnp.ones((2, 2)), "label": "fixed"},
        "readout": {"weight": jnp.ones((1, 2)), "bias": jnp.zeros((1,))},
    }
    paths = ParameterSubspace.array_leaf_paths(tree)
    selected_paths = tuple(path for path in paths if path.startswith("['readout']"))
    subspace = ParameterSubspace.from_leaf_paths(tree, selected_paths)

    assert subspace.leaf_paths == selected_paths
    assert subspace.total_dimension == 3
    reconstructed = subspace.reconstruct(subspace.initial)
    assert eqx.tree_equal(reconstructed, tree)


def test_parameter_subspace_selects_disjoint_subtrees():
    tree = {
        "branches": (
            {"body": jnp.ones((3, 2)), "head": jnp.ones((1, 3))},
            {"body": jnp.ones((4, 2)), "head": jnp.ones((1, 4))},
        )
    }
    subspace = ParameterSubspace.from_subtree_paths(
        tree,
        ("['branches'][0]['head']", "['branches'][1]['head']"),
    )
    assert subspace.total_dimension == 7
    assert subspace.leaf_paths == (
        "['branches'][0]['head']",
        "['branches'][1]['head']",
    )

    with pytest.raises(ValueError, match="disjoint"):
        ParameterSubspace.from_subtree_paths(
            tree,
            ("['branches'][0]", "['branches'][0]['head']"),
        )
