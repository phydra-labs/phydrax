import math

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


tn = phx.tensor_network
_CRITICAL_BETA = 0.5 * math.log(1.0 + math.sqrt(2.0))
_CRITICAL_LOG_PARTITION = 0.5 * math.log(2.0) + 2.0 * 0.915965594177219 / math.pi


def _ising_tensor(beta: float, *, precision=None):
    spins = jnp.asarray((-1.0, 1.0), dtype=jnp.float64)
    weights = jnp.exp(beta * spins[:, None] * spins[None, :])
    return tn.build_uniform_pair_partition_tensor(
        weights,
        precision=precision,
        positivity_tolerance=1e-6 if precision is not None else 1e-10,
    ).tensor


def _run(tensor, method, *, bond: int, steps: int):
    return tn.run_tensor_renormalization(
        tn.TensorRenormalizationProblem(tensor),
        tn.TensorRenormalizationPolicy(
            method,
            maximum_bond_dimension=bond,
            steps=steps,
        ),
    )


def test_uniform_square_tensor_enforces_oriented_opposite_bonds():
    value = jnp.ones((2, 3, 2, 3), dtype=jnp.float64)
    tensor = tn.UniformSquareTensor(value)
    assert tensor.vertical_bond_dimension == 2
    assert tensor.horizontal_bond_dimension == 3

    with pytest.raises(ValueError, match="up/down"):
        tn.UniformSquareTensor(jnp.ones((2, 3, 4, 3)))
    with pytest.raises(ValueError, match="right/left"):
        tn.UniformSquareTensor(jnp.ones((2, 3, 2, 4)))


def test_pair_partition_builder_handles_anisotropic_diagonal_weights():
    vertical = jnp.diag(jnp.asarray((4.0, 9.0), dtype=jnp.float64))
    horizontal = jnp.diag(jnp.asarray((1.0, 16.0), dtype=jnp.float64))
    onsite = jnp.asarray((2.0, 3.0), dtype=jnp.float64)
    result = tn.build_uniform_pair_partition_tensor(
        vertical,
        horizontal,
        site_weight=onsite,
    )
    expected = jnp.zeros((2, 2, 2, 2), dtype=jnp.float64)
    expected = expected.at[0, 0, 0, 0].set(8.0)
    expected = expected.at[1, 1, 1, 1].set(432.0)
    assert result.evidence.accepted
    assert result.evidence.positive_semidefinite
    assert jnp.allclose(result.tensor.value, expected, atol=1e-12)
    float32 = tn.build_uniform_pair_partition_tensor(
        vertical.astype(jnp.float32),
        horizontal.astype(jnp.float32),
        site_weight=onsite.astype(jnp.float32),
        positivity_tolerance=1e-5,
    )
    assert float32.tensor.value.dtype == jnp.dtype("float32")


def test_pair_partition_builder_rejects_indefinite_weights():
    indefinite = jnp.asarray(((1.0, 2.0), (2.0, 1.0)), dtype=jnp.float64)
    with pytest.raises(eqx.EquinoxRuntimeError, match="positive-semidefinite"):
        result = tn.build_uniform_pair_partition_tensor(indefinite)
        jax.block_until_ready(result.tensor.value)


@pytest.mark.parametrize("method", (tn.TRGMethod(), tn.HOTRGMethod()))
def test_tensor_renormalization_is_exact_for_unit_bond_dimension(method):
    tensor = tn.UniformSquareTensor(jnp.asarray([[[[2.0]]]], dtype=jnp.float64))
    result = _run(tensor, method, bond=1, steps=4)
    assert result.successful
    assert result.diagnostics.exact
    assert not result.diagnostics.global_error_bound_claimed
    assert jnp.allclose(result.log_partition_density, jnp.log(2.0), atol=1e-12)
    assert result.final_tensor.value.shape == (1, 1, 1, 1)


def test_full_rank_one_step_matches_independent_two_tensor_closures():
    value = (
        jax.random.uniform(
            jax.random.key(7),
            (2, 2, 2, 2),
            dtype=jnp.float64,
        )
        + 0.1
    )
    tensor = tn.UniformSquareTensor(value)
    trg = _run(tensor, tn.TRGMethod(), bond=4, steps=1)
    hotrg_vertical = _run(
        tensor,
        tn.HOTRGMethod(first_direction="vertical"),
        bond=4,
        steps=1,
    )
    hotrg_horizontal = _run(
        tensor,
        tn.HOTRGMethod(first_direction="horizontal"),
        bond=4,
        steps=1,
    )
    trg_partition = jnp.einsum("urdl,urdl->", value, value)
    vertical_partition = jnp.einsum("uama,mbub->", value, value)
    horizontal_partition = jnp.einsum("umul,vlvm->", value, value)
    assert jnp.allclose(jnp.exp(2.0 * trg.log_partition_density), trg_partition)
    assert jnp.allclose(
        jnp.exp(2.0 * hotrg_vertical.log_partition_density),
        vertical_partition,
    )
    assert jnp.allclose(
        jnp.exp(2.0 * hotrg_horizontal.log_partition_density),
        horizontal_partition,
    )


@pytest.mark.parametrize("method", (tn.TRGMethod(), tn.HOTRGMethod()))
def test_critical_ising_partition_density_matches_exact_solution(method):
    result = _run(_ising_tensor(_CRITICAL_BETA), method, bond=6, steps=8)
    assert result.successful
    assert jnp.isfinite(result.log_partition_density)
    assert abs(float(result.log_partition_density) - _CRITICAL_LOG_PARTITION) < 5e-3


def test_q2_potts_partition_matches_ising_mapping():
    beta = 2.0 * _CRITICAL_BETA
    pair_weight = jnp.asarray(
        ((math.exp(beta), 1.0), (1.0, math.exp(beta))),
        dtype=jnp.float64,
    )
    tensor = tn.build_uniform_pair_partition_tensor(pair_weight).tensor
    result = _run(tensor, tn.TRGMethod(), bond=6, steps=8)
    reference = beta + _CRITICAL_LOG_PARTITION
    assert result.successful
    assert abs(float(result.log_partition_density) - reference) < 5e-3


def test_hotrg_plan_preserves_anisotropy_and_alternates_directions():
    tensor = tn.UniformSquareTensor(jnp.ones((2, 3, 2, 3), dtype=jnp.float64))
    problem = tn.TensorRenormalizationProblem(tensor)
    plan = tn.plan_tensor_renormalization(
        problem,
        tn.TensorRenormalizationPolicy(
            tn.HOTRGMethod(first_direction="vertical"),
            maximum_bond_dimension=4,
            steps=2,
        ),
    )
    assert tuple(stage.kind for stage in plan.stages) == (
        "hotrg-vertical",
        "hotrg-horizontal",
    )
    assert plan.stages[0].output_shape == (2, 4, 2, 4)
    assert plan.stages[1].output_shape == (4, 4, 4, 4)


def test_prepared_refresh_reuses_plan_but_changes_partition_density():
    tensor = _ising_tensor(0.2)
    problem = tn.TensorRenormalizationProblem(tensor)
    policy = tn.TensorRenormalizationPolicy(
        tn.TRGMethod(),
        maximum_bond_dimension=4,
        steps=4,
    )
    prepared = tn.prepare_tensor_renormalization(problem, policy)
    original = tn.run_tensor_renormalization(prepared)
    scaled_tensor = tn.UniformSquareTensor(
        tensor.value * 1.01,
        precision=tensor.precision,
    )
    refreshed = tn.refresh_tensor_renormalization(
        prepared,
        tn.TensorRenormalizationProblem(scaled_tensor),
    )
    updated = tn.run_tensor_renormalization(refreshed)
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert refreshed.prepared_id == prepared.prepared_id
    assert int(refreshed.numeric_version) == int(prepared.numeric_version) + 1
    assert jnp.allclose(
        updated.log_partition_density - original.log_partition_density,
        jnp.log(1.01),
        atol=1e-10,
    )


def test_tensor_renormalization_reports_numerical_failures():
    nonfinite = tn.UniformSquareTensor(jnp.asarray([[[[jnp.nan]]]], dtype=jnp.float64))
    nonfinite_result = _run(nonfinite, tn.TRGMethod(), bond=1, steps=1)
    assert int(nonfinite_result.diagnostics.status) == int(
        tn.TensorRenormalizationStatus.NONFINITE_INPUT
    )
    assert not nonfinite_result.successful

    signed_value = jnp.zeros((2, 1, 2, 1), dtype=jnp.float64)
    signed_value = signed_value.at[0, 0, 1, 0].set(1.0)
    signed_value = signed_value.at[1, 0, 0, 0].set(-1.0)
    signed_result = _run(
        tn.UniformSquareTensor(signed_value),
        tn.HOTRGMethod(),
        bond=1,
        steps=1,
    )
    assert int(signed_result.diagnostics.status) == int(
        tn.TensorRenormalizationStatus.TERMINAL_PARTITION_INVALID
    )
    assert not signed_result.successful

    complex_partition = tn.UniformSquareTensor(
        jnp.asarray([[[[1.0j]]]], dtype=jnp.complex128)
    )
    with pytest.raises(TypeError, match="real partition tensors"):
        tn.TensorRenormalizationProblem(complex_partition)


def test_tensor_renormalization_refuses_factorization_resources_before_execution():
    problem = tn.TensorRenormalizationProblem(_ising_tensor(0.2))
    resources = tn.TensorRenormalizationResourcePolicy(maximum_factorization_elements=1)
    policy = tn.TensorRenormalizationPolicy(
        tn.TRGMethod(),
        maximum_bond_dimension=4,
        steps=1,
        resources=resources,
    )
    with pytest.raises(MemoryError, match="factorization"):
        tn.plan_tensor_renormalization(problem, policy)


def test_tensor_renormalization_honors_mixed_precision_roles():
    precision = tn.TensorNetworkPrecisionPolicy(
        storage_dtype="float32",
        contraction_dtype="float32",
        factorization_dtype="float64",
        accumulation_dtype="float64",
        decision_dtype="float64",
        output_dtype="float64",
    )
    tensor = _ising_tensor(0.2, precision=precision)
    result = _run(tensor, tn.TRGMethod(), bond=4, steps=3)
    assert result.successful
    assert result.final_tensor.value.dtype == jnp.dtype("float32")
    assert result.log_partition_density.dtype == jnp.dtype("float64")
    assert result.diagnostics.first_discarded_weight_history.dtype == jnp.dtype("float64")
