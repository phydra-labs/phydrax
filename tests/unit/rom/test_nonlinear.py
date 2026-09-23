import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


class _Residual(phx.rom.AbstractResidualProvider):
    state_space: phx.linalg.AbstractVectorSpace
    residual_space: phx.linalg.AbstractVectorSpace
    residual_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(self, state_space):
        self.state_space = state_space
        self.residual_space = phx.linalg.DualSpace(state_space)
        self.residual_id = "quadratic-residual"
        self.support_id = "fixed-support"
        self.geometry_id = "fixed-geometry"

    def residual(self, coordinate, state, state_rate, inputs, /):
        del coordinate, state_rate, inputs
        return jnp.asarray([state[0] ** 2, state[1] ** 2, state[0] + state[1]])


class _StageResidual(phx.rom.AbstractStageResidualProvider):
    state_space: phx.linalg.AbstractVectorSpace
    residual_space: phx.linalg.AbstractVectorSpace
    residual_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(self, state_space):
        self.state_space = state_space
        self.residual_space = phx.linalg.DualSpace(state_space)
        self.residual_id = "manufactured-stage-residual"
        self.support_id = "fixed-support"
        self.geometry_id = "fixed-geometry"

    def residual(
        self,
        source_coordinate,
        target_coordinate,
        source_state,
        target_state,
        inputs,
        /,
    ):
        del inputs
        step = target_coordinate - source_coordinate
        return target_state - source_state - step * jnp.asarray([1.0, 2.0, 0.0])


class _SampledNonlinear(phx.rom.AbstractSampledNonlinearProvider):
    provider_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(self):
        self.provider_id = "sampled-quadratic"
        self.support_id = "fixed-support"
        self.geometry_id = "fixed-geometry"

    def evaluate_selected(self, state, node_indices, inputs, /):
        del inputs
        values = jnp.asarray([state[0] ** 2, state[1] ** 2, 0.0])
        return values[node_indices]


class _SampledStage(phx.rom.AbstractSampledStageResidualProvider):
    provider_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(self):
        self.provider_id = "sampled-stage"
        self.support_id = "fixed-support"
        self.geometry_id = "fixed-geometry"

    def evaluate_selected(
        self,
        source_coordinate,
        target_coordinate,
        source_state,
        target_state,
        node_indices,
        inputs,
        /,
    ):
        del inputs
        step = target_coordinate - source_coordinate
        residual = target_state - source_state - step * jnp.asarray([1.0, 2.0, 0.0])
        return residual[node_indices]


class _Elements(phx.rom.AbstractElementResidualProvider):
    provider_id: str = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    contributions: jnp.ndarray

    def __init__(self, reduction_id, contributions):
        self.provider_id = "manufactured-elements"
        self.reduction_id = reduction_id
        self.support_id = "fixed-support"
        self.geometry_id = "fixed-geometry"
        self.contributions = jnp.asarray(contributions)

    def evaluate_elements(self, state, element_indices, inputs, /):
        del state, inputs
        return self.contributions[element_indices]


def _reduction():
    full = phx.linalg.ArraySpace((3,), dtype=jnp.float64, space_id="nonlinear-full")
    basis = phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(
            full,
            jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64),
            orthonormal=True,
            subspace_id="nonlinear-state-subspace",
        ),
        role="state",
        state_contract_id="three-component-state",
        support_id="fixed-support",
        measure_id="euclidean-measure",
        geometry_id="fixed-geometry",
        source_artifact_ids=("nonlinear-snapshots",),
    )
    return full, phx.rom.trial_test_reduction_from_bases(basis)


def _residual_basis(full, role, source):
    dual = phx.linalg.DualSpace(full)
    return phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(
            dual,
            jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64),
            orthonormal=True,
            subspace_id=f"{role}-subspace",
        ),
        role=role,
        state_contract_id="three-component-residual",
        support_id="fixed-support",
        measure_id="euclidean-measure",
        geometry_id="fixed-geometry",
        source_artifact_ids=(source,),
    )


def test_full_galerkin_and_deim_agree_on_collateral_span():
    full, reduction = _reduction()
    state = jnp.asarray([2.0, 3.0, 0.0], dtype=jnp.float64)
    galerkin = phx.rom.FullResidualGalerkin(reduction, _Residual(full))
    expected = galerkin.residual(0.0, jnp.asarray([2.0, 3.0]))
    provider = _SampledNonlinear()
    deim = phx.rom.prepare_deim(
        reduction,
        _residual_basis(full, "nonlinear-term", "nonlinear-term-snapshots"),
        provider,
    )

    np.testing.assert_allclose(deim.evaluate(provider, state), expected)
    assert deim.node_indices.size == 2


def test_lspg_and_gnat_solve_same_time_discrete_residual():
    full, reduction = _reduction()
    stage = _StageResidual(full)
    lspg = phx.rom.ReducedLSPGProblem(reduction, stage)
    context = phx.rom.LSPGStepContext(
        jnp.asarray(0.0),
        jnp.asarray(0.5),
        jnp.asarray([0.0, 0.0]),
        None,
    )
    full_result = lspg.solve(jnp.asarray([0.1, 0.1]), context)

    sampled = _SampledStage()
    gnat = phx.rom.prepare_gnat(
        _residual_basis(full, "residual", "stage-residual-snapshots"),
        sampled,
    )
    reduced_result = phx.rom.GNATLSPGProblem(lspg, sampled, gnat).solve(
        jnp.asarray([0.1, 0.1]),
        context,
    )

    np.testing.assert_allclose(
        full_result.parameters,
        np.asarray([0.5, 1.0]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        reduced_result.parameters,
        full_result.parameters,
        atol=1e-7,
    )


def test_ecsw_selects_nonnegative_element_quadrature_and_reproduces_target():
    _, reduction = _reduction()
    training_contributions = jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0], [1.0, -1.0]],
        ],
        dtype=jnp.float64,
    )
    target = 2.0 * training_contributions[:, 1, :]
    online_contributions = jnp.asarray(
        [[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]],
        dtype=jnp.float64,
    )
    provider = _Elements(reduction.reduction_id, online_contributions)
    artifact = phx.rom.prepare_ecsw(
        training_contributions,
        target,
        reduction,
        provider,
        phx.rom.ECSWPlan(1),
    )
    result = phx.rom.evaluate_ecsw(
        artifact,
        provider,
        jnp.zeros((3,), dtype=jnp.float64),
    )

    assert jnp.all(artifact.weights >= 0.0)
    np.testing.assert_allclose(result, np.asarray([0.0, 8.0]), atol=1e-5)


def test_thin_gnat_rejects_nonintegral_duplicate_and_out_of_range_nodes():
    full, _ = _reduction()
    residual = _residual_basis(full, "residual", "thin-gnat-snapshots")
    metric = jnp.eye(3, dtype=jnp.float64)

    with pytest.raises(TypeError, match="integers"):
        phx.rom.prepare_thin_gnat(
            residual,
            jnp.asarray([0.9, 1.1]),
            metric,
            provider_id="sampled-residual",
        )
    for nodes, message in (
        (jnp.asarray([-1, 1]), "outside"),
        (jnp.asarray([0, 3]), "outside"),
        (jnp.asarray([0, 0]), "unique"),
    ):
        with pytest.raises(ValueError, match=message):
            phx.rom.prepare_thin_gnat(
                residual,
                nodes,
                metric,
                provider_id="sampled-residual",
            )
