#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import opt_einsum as oe
import pytest

from phydrax.tensor_network._boundary_mps import (
    BoundaryMPSPolicy,
    contract_peps_boundary_mps,
)
from phydrax.tensor_network._contraction import (
    ContractionPlanCache,
    ContractionPlannerPolicy,
    ContractionResourcePolicy,
    execute_contraction,
    execute_contraction_reverse,
    plan_contraction,
    prepare_contraction,
)
from phydrax.tensor_network._ctmrg import contract_peps_ctmrg, CTMRGPolicy
from phydrax.tensor_network._mera import (
    BinaryMERA,
    contract_mera,
    MERAResourcePolicy,
    update_mera_isometry,
)
from phydrax.tensor_network._network_bp import (
    FactorGraphNetwork,
    FactorTensor,
    NetworkBPPolicy,
    run_network_belief_propagation,
)
from phydrax.tensor_network._peps import contract_peps_exact, PEPS
from phydrax.tensor_network._peps_update import (
    full_update_peps,
    PEPSUpdatePolicy,
    simple_update_peps,
)
from phydrax.tensor_network._placement import (
    create_tensor_network_mesh,
    execute_distributed_slices,
    plan_slice_placement,
)
from phydrax.tensor_network._slicing import (
    checkpoint_slice_ranges,
    execute_sliced_contraction,
    merge_slice_checkpoints,
    mixed_radix_assignments,
    plan_sliced_contraction,
    SlicingResourcePolicy,
)
from phydrax.tensor_network._topology import (
    ContractionLeg,
    ContractionOperand,
    ContractionStructure,
    tree_contraction_structure,
)
from phydrax.tensor_network._tree_network import (
    contract_tree_messages,
    TreeTensorNetwork,
)


def test_arbitrary_incidence_trace_hyperedge_scalar_and_outputs():
    structure = ContractionStructure(
        (
            ContractionOperand(
                "diagonal", (ContractionLeg("x", 3), ContractionLeg("x", 3))
            ),
            ContractionOperand("left-copy", (ContractionLeg("x", 3),)),
            ContractionOperand("right-copy", (ContractionLeg("x", 3),)),
            ContractionOperand("scalar"),
        ),
        (),
    )
    matrix = jnp.arange(9.0, dtype=jnp.float32).reshape((3, 3))
    left = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    right = jnp.asarray([0.5, 1.0, 1.5], dtype=jnp.float32)
    plan = plan_contraction(structure, dtype="float32")
    result = execute_contraction(
        prepare_contraction(
            plan, (matrix, left, right, jnp.asarray(2.0, dtype=jnp.float32))
        )
    )
    assert structure.has_diagonals
    assert structure.has_hyperedges
    assert result.evidence.exact
    assert jnp.allclose(result.value, 2.0 * jnp.sum(jnp.diag(matrix) * left * right))

    output_structure = ContractionStructure(
        (
            ContractionOperand("a", (ContractionLeg("i", 2), ContractionLeg("j", 3))),
            ContractionOperand("b", (ContractionLeg("j", 3), ContractionLeg("k", 4))),
        ),
        ("k", "i"),
    )
    a = jnp.arange(6.0, dtype=jnp.float32).reshape((2, 3))
    b = jnp.arange(12.0, dtype=jnp.float32).reshape((3, 4))
    output = execute_contraction(
        prepare_contraction(plan_contraction(output_structure, dtype="float32"), (a, b))
    )
    assert output.value.shape == (4, 2)
    assert jnp.allclose(output.value, (a @ b).T)


def test_slicing_order_serial_batch_equality_resource_refusal_and_reverse():
    structure = ContractionStructure(
        (
            ContractionOperand("left", (ContractionLeg("i", 2), ContractionLeg("j", 3))),
            ContractionOperand("right", (ContractionLeg("j", 3), ContractionLeg("k", 2))),
        ),
        ("i", "k"),
    )
    left = jnp.arange(6.0, dtype=jnp.float32).reshape((2, 3))
    right = jnp.arange(6.0, dtype=jnp.float32).reshape((3, 2))
    cache = ContractionPlanCache(2)
    cached_first = plan_contraction(structure, dtype="float32", cache=cache)
    cached_second = plan_contraction(structure, dtype="float32", cache=cache)
    assert cached_first is cached_second
    assert cached_first.cost.peak_live_elements >= cached_first.cost.operand_elements
    search_structure = ContractionStructure(
        structure.operands + (ContractionOperand("scale"),),
        structure.outputs,
    )
    with pytest.raises(RuntimeError, match="search"):
        plan_contraction(
            search_structure,
            dtype="float32",
            planner=ContractionPlannerPolicy(maximum_search_states=1),
            optimizer="optimal",
        )
    plan = plan_contraction(structure, dtype="float32")
    prepared = prepare_contraction(plan, (left, right))
    sliced = plan_sliced_contraction(plan, ("j",), batch_size=2)
    serial = execute_sliced_contraction(prepared, sliced, mode="serial")
    batched = execute_sliced_contraction(
        prepared, sliced, mode="batched", logarithmic_scaling=True
    )
    direct = execute_contraction(prepared)
    assert jnp.array_equal(
        mixed_radix_assignments((2, 3)),
        jnp.asarray([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]),
    )
    assert jnp.allclose(serial.value, direct.value)
    assert jnp.allclose(batched.value, direct.value)
    assert serial.evidence.accepted and batched.evidence.accepted
    ranges = checkpoint_slice_ranges(sliced.slice_count, 2)
    partials = tuple(
        execute_sliced_contraction(
            prepared,
            sliced,
            slice_range=range_,
            logarithmic_scaling=True,
        ).checkpoint
        for range_ in ranges
    )
    merged = merge_slice_checkpoints(sliced, partials)
    assert merged.evidence.accepted
    assert jnp.allclose(merged.value, direct.value)
    with pytest.raises(MemoryError, match="maximum_slices"):
        plan_sliced_contraction(
            plan,
            ("j",),
            resources=SlicingResourcePolicy(maximum_slices=2),
        )
    with pytest.raises(MemoryError, match="workspace"):
        plan_contraction(
            structure,
            dtype="float32",
            resources=ContractionResourcePolicy(maximum_workspace_bytes=1),
        )

    scalar_structure = ContractionStructure(
        (
            ContractionOperand("x", (ContractionLeg("q", 3),)),
            ContractionOperand("y", (ContractionLeg("q", 3),)),
        ),
        (),
    )
    x = jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float32)
    y = jnp.asarray([3.0, 5.0, 7.0], dtype=jnp.float32)
    reverse = execute_contraction_reverse(
        prepare_contraction(plan_contraction(scalar_structure, dtype="float32"), (x, y)),
        rematerialization="rematerialize",
    )
    assert jnp.allclose(reverse.operand_cotangents[0], y)
    assert jnp.allclose(reverse.operand_cotangents[1], x)
    assert reverse.reverse_evidence.exact_derivative


def test_single_and_available_multi_device_slice_parity():
    structure = ContractionStructure(
        (
            ContractionOperand("x", (ContractionLeg("s", 4),)),
            ContractionOperand("y", (ContractionLeg("s", 4),)),
        ),
        (),
    )
    x = jnp.arange(4.0, dtype=jnp.float32)
    y = jnp.arange(4.0, dtype=jnp.float32) + 1.0
    original = plan_contraction(structure, dtype="float32")
    prepared = prepare_contraction(original, (x, y))
    sliced = plan_sliced_contraction(original, ("s",), batch_size=2)
    expected = execute_sliced_contraction(prepared, sliced).value

    single = plan_slice_placement(sliced, create_tensor_network_mesh((jax.devices()[0],)))
    single_result = execute_distributed_slices(prepared, single)
    assert single_result.aggregate is not None
    assert jnp.allclose(single_result.aggregate, expected)
    devices = tuple(jax.devices())
    if len(devices) > 1:
        multiple = plan_slice_placement(sliced, create_tensor_network_mesh(devices))
        multiple_result = execute_distributed_slices(prepared, multiple)
        assert multiple_result.aggregate is not None
        assert jnp.allclose(multiple_result.aggregate, expected)


def _product_peps(rows=2, columns=2):
    local = jnp.asarray([1.0, 0.0], dtype=jnp.float32).reshape((1, 1, 1, 1, 2))
    return PEPS(tuple(local for _ in range(rows * columns)), rows, columns)


def test_exact_2x2_peps_boundary_and_ctm_evidence():
    state = _product_peps()
    exact = contract_peps_exact(state)
    boundary = contract_peps_boundary_mps(state, BoundaryMPSPolicy(2))
    ctm = contract_peps_ctmrg(state, CTMRGPolicy(2, 6, tolerance=1e-6))
    assert jnp.allclose(exact.value, 1.0)
    assert exact.evidence.exact and not exact.evidence.global_error_bound_claimed
    assert jnp.allclose(boundary.value, exact.value)
    assert boundary.evidence.exact
    assert ctm.evidence.convergence_mask.shape == (6,)
    assert ctm.evidence.converged
    assert ctm.evidence.exact
    assert not ctm.evidence.global_error_bound_claimed


def test_simple_and_full_peps_updates_are_explicitly_distinct():
    state = _product_peps(1, 2)
    gate = jnp.eye(4, dtype=jnp.float32).reshape((2, 2, 2, 2))
    policy = PEPSUpdatePolicy(2, regularization=1e-3)
    simple = simple_update_peps(state, 0, 0, "right", gate, policy)
    full = full_update_peps(
        state,
        0,
        0,
        "right",
        gate,
        jnp.diag(jnp.asarray([1.0, 2.0], dtype=jnp.float32)),
        policy,
    )
    assert simple.evidence.route == "simple"
    assert full.evidence.route == "full-environment-weighted"
    assert simple.evidence.solver_status == -1
    assert full.evidence.solver_successful
    assert not simple.evidence.global_error_bound_claimed
    assert not full.evidence.global_error_bound_claimed


def test_ttn_messages_equal_direct_exact_contraction():
    structure = tree_contraction_structure(2, bond_dimension=2, physical_dimension=2)
    leaf0 = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    leaf1 = jnp.asarray([[0.5, 1.0], [1.5, 2.0]], dtype=jnp.float32)
    branch = jnp.arange(8.0, dtype=jnp.float32).reshape((2, 2, 2)) / 8.0
    root = jnp.asarray([1.0, -0.5], dtype=jnp.float32)
    result = contract_tree_messages(
        TreeTensorNetwork(structure, (leaf0, leaf1, branch, root))
    )
    expected = oe.contract("ia,jb,abc,c->ij", leaf0, leaf1, branch, root)
    assert result.evidence.exact
    assert result.evidence.accepted
    assert jnp.allclose(result.value, expected)


def test_loopy_bp_residual_and_binary_mera_isometry():
    pair = jnp.asarray([[2.0, 1.0], [1.0, 2.0]], dtype=jnp.float32)
    graph = FactorGraphNetwork(
        {"a": 2, "b": 2, "c": 2},
        (
            FactorTensor("ab", ("a", "b"), pair),
            FactorTensor("bc", ("b", "c"), pair),
            FactorTensor("ca", ("c", "a"), pair),
        ),
    )
    bp = run_network_belief_propagation(
        graph, NetworkBPPolicy(20, tolerance=1e-6, damping=0.2)
    )
    assert bp.evidence.converged
    assert bp.evidence.residual_history[-1] <= 1e-6
    assert not bp.evidence.exact
    assert jnp.allclose(bp.variable_beliefs[0], jnp.asarray([0.5, 0.5]))

    isometry = jnp.eye(4, dtype=jnp.float32)[:, :2].reshape((2, 2, 2))
    disentangler = jnp.eye(4, dtype=jnp.float32).reshape((2, 2, 2, 2))
    mera = BinaryMERA((isometry,), (disentangler,))
    policy = MERAResourcePolicy(isometry_tolerance=1e-5)
    contraction = contract_mera(mera, jnp.asarray([1.0, 0.0], dtype=jnp.float32), policy)
    update = update_mera_isometry(mera, 0, jnp.ones_like(isometry), 0.05, policy)
    assert contraction.evidence.accepted
    assert jnp.allclose(contraction.value, 1.0)
    assert update.evidence.accepted
    assert update.evidence.residual_after <= policy.isometry_tolerance
