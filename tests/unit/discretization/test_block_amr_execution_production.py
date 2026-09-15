#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec

import phydrax as phx


def _hierarchy():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(4),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    signature = phx.discretization.PatchShapeSignature((4,), halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (phx.discretization.PatchBucketPlan(signature, 1),),
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0,), (4,)),),
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(plan).initial_topology()
    return phx.discretization.canonicalize_patch_hierarchy(topology)


def test_signature_policy_generates_aligned_finite_envelopes():
    policy = phx.discretization.PatchSignaturePolicy(
        (2, 2),
        (16, 16),
        halo_width=2,
    )

    signature = policy.shape_signature((5, 3))

    assert signature.envelope_shape == (8, 4)
    assert signature.halo_width == (2, 2)
    assert signature.admits((6, 4))


def test_executable_cache_compiles_then_reuses_exact_signature():
    resources = phx.discretization.BlockAMRResourcePlan(
        maximum_components_per_cell=2,
    )
    cache = phx.discretization.PatchExecutableCachePlan(resources)
    signatures = cache.required_signatures(
        _hierarchy(),
        physical_component_count=3,
        method_id="cache-kernel",
        dtype=np.float64,
    )

    def kernel_factory(signature):
        scale = jnp.asarray(2.0, dtype=jnp.dtype(signature.dtype))
        return lambda values: scale * values

    def sample_factory(signature):
        return ((jnp.zeros(signature.state_shape, dtype=signature.dtype),), {})

    initial = phx.discretization.PatchExecutableCacheState()
    installed = cache.install(initial, signatures, kernel_factory, sample_factory)
    reused = cache.install(
        installed.state,
        signatures,
        kernel_factory,
        sample_factory,
    )

    assert installed.changed
    assert installed.state.generation == 1
    assert not reused.changed
    assert reused.state.cache_id == installed.state.cache_id
    executable = installed.state.lookup(signatures[0])
    assert executable is not None
    values = jnp.ones(signatures[0].state_shape, dtype=signatures[0].dtype)
    np.testing.assert_allclose(executable(values), 2.0)


def _cut_complex():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(1) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    signature = phx.discretization.PatchShapeSignature((1, 1, 1), halo_width=1)
    plan = phx.discretization.VariablePatchHierarchyPlan(
        grid,
        (
            phx.discretization.VariablePatchLevelPlan(
                0,
                (phx.discretization.PatchBucketPlan(signature, 1),),
            ),
        ),
        (phx.discretization.LogicalPatchBox(0, (0, 0, 0), (1, 1, 1)),),
    )
    topology = phx.discretization.VariablePatchTopologyCompiler(plan).initial_topology()
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.37,
        "distributed-plane",
        3,
    )
    resources = phx.discretization.BlockAMRResourcePlan(
        maximum_components_per_cell=2,
        maximum_apertures_per_face=16,
        maximum_embedded_faces_per_cell=32,
    )
    return phx.discretization.MultivaluedCutCellPlan(
        topology,
        lambda points, time, args: points,
        "identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        resources,
    ).prepare()


def test_cut_cell_partition_uses_live_execution_group_sharding():
    complex_ = _cut_complex()
    group = phx.execution.ExecutionRuntime.current().root_group
    partition = phx.discretization.DistributedCutCellPartitionPlan(
        group,
        2,
    ).prepare(complex_)
    canonical = jnp.zeros((complex_.component_capacity, 2))
    canonical = canonical.at[0].set(jnp.asarray((2.0, 3.0)))

    state = partition.pack(canonical)
    restored = partition.unpack(state)
    left, right, active = partition.face_states(state)
    local_indices = partition.local_component_indices()
    local_state = partition.pack_process_local(
        local_indices,
        canonical[jnp.asarray(local_indices, dtype=jnp.int32)],
    )
    local_restored = partition.unpack_process_local(local_state)

    np.testing.assert_allclose(restored, canonical)
    assert state.values.sharding == group.named_sharding(
        PartitionSpec(partition.mesh_axis, None, None)
    )
    assert left.shape == right.shape == (complex_.face_capacity, 2)
    assert int(jnp.count_nonzero(active)) == complex_.face_count
    assert partition.local_component_indices() == (0,)
    np.testing.assert_allclose(
        local_restored,
        canonical[jnp.asarray(local_indices, dtype=jnp.int32)],
    )
    np.testing.assert_allclose(partition.unpack(local_state), canonical)
    assert partition.collective_accept(jnp.asarray(True))


def test_frozen_cut_transition_jvp_and_vjp_are_exact_pair():
    complex_ = _cut_complex()
    transition = phx.discretization.MultivaluedCutCellTransition(
        complex_,
        complex_,
    )
    derivative = phx.discretization.FrozenCutCellTransitionDerivativePlan(
        transition,
        topology_margin=0.1,
    )
    source = jnp.zeros((complex_.component_capacity, 2))
    source = source.at[0].set(jnp.asarray((1.0, 2.0)))
    tangent = jnp.zeros_like(source).at[0].set(jnp.asarray((0.3, -0.2)))
    result = derivative.jvp_content(source, tangent)
    cotangent = jnp.zeros_like(source).at[0].set(jnp.asarray((0.7, 0.4)))
    pullback, evidence = derivative.vjp_content(cotangent)

    assert bool(result.evidence.valid)
    assert bool(evidence.valid)
    np.testing.assert_allclose(
        jnp.vdot(result.tangent, cotangent),
        jnp.vdot(tangent, pullback),
    )


def test_relaxed_hierarchy_derivative_is_explicit_smooth_surrogate():
    plan = phx.discretization.RelaxedHierarchyBlendPlan(
        phx.discretization.BlockAMRDerivativePolicy(
            "relaxed",
            relaxation_temperature=0.2,
        )
    )
    coarse = jnp.asarray(((1.0,), (2.0,)))
    fine = jnp.asarray(((3.0,), (4.0,)))
    indicators = jnp.asarray(((0.0, 1.0), (1.0, 0.0)))

    blended, weights = plan.blend((coarse, fine), indicators)

    np.testing.assert_allclose(jnp.sum(weights, axis=0), 1.0)
    assert blended[0, 0] > coarse[0, 0]
    assert blended[1, 0] < fine[1, 0]
