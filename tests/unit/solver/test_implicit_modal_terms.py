import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

import phydrax as phx


class _HeatModeModel(eqx.Module):
    scale: jnp.ndarray
    diffusivity: float = eqx.field(static=True)

    def __call__(self, query, *, key=None):
        del key
        mode, time = query
        amplitude = jnp.where(
            mode == 1.0,
            -0.5j,
            jnp.where(mode == -1.0, 0.5j, 0.0j),
        )
        wave_number = 2.0 * jnp.pi * mode
        return self.scale * amplitude * jnp.exp(-self.diffusivity * wave_number**2 * time)


def _compiled_heat(count=4, diffusivity=0.05):
    x = phx.equations.PDECoordinate(
        "x",
        "space",
        bounds=(0.0, 1.0),
        periodic=True,
    )
    t = phx.equations.PDECoordinate("t", "time")
    pde_field = phx.equations.PDEField("u", coordinates=("x", "t"))
    u = phx.equations.PDEExpression.field("u")
    problem = phx.equations.PDEProblemIR(
        (x, t),
        (pde_field,),
        equations=(
            phx.equations.PDEEquation(
                "heat",
                u.derivative("t"),
                diffusivity * u.laplacian("x"),
            ),
        ),
    )
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(count),),
        axis_names=("x",),
        field_name="u",
    ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
    compiled = phx.equations.compile_semidiscrete_pde(
        problem,
        space,
        phx.discretization.PseudospectralMethodPlan(),
    )
    return space, compiled


def _function(scale, space, *, diffusivity=0.05):
    modal = phx.nn.models.ImplicitModalField(
        _HeatModeModel(jnp.asarray(scale), diffusivity),
        space,
        real_field=True,
    )
    time = phx.domain.ScalarInterval(0.0, 0.1, label="t")
    return modal.as_domain_function(time)


def test_compiled_modal_residual_matches_exact_heat_evolution():
    space, compiled = _compiled_heat()
    function = _function(1.0, space)
    term = phx.terms.CompiledModalResidualTerm(
        compiled,
        function_name="u_hat",
        times=jnp.linspace(0.0, 0.1, 4),
    )

    loss = term.loss({"u_hat": function}, key=jr.key(1))

    assert loss < 1e-24


def test_modal_observation_ignores_unobserved_nonfinite_targets():
    space, _compiled = _compiled_heat()
    function = _function(1.0, space)
    target = function.func(0.0)[None, ...]
    mode_index = int(jnp.argmax(function.func.mode_numbers[:, 0] == 1.0))
    mask = jnp.zeros_like(target, dtype=bool).at[0, mode_index].set(True)
    masked_target = jnp.where(mask, target, jnp.asarray(jnp.nan + 0.0j))
    term = phx.terms.ModalObservationTerm(
        jnp.asarray([0.0]),
        masked_target,
        function_name="u_hat",
        mask=mask,
    )

    assert term.loss({"u_hat": function}, key=jr.key(2)) == 0.0


def test_functional_solver_updates_implicit_modal_parameters():
    space, compiled = _compiled_heat()
    exact = _function(1.0, space)
    trainable = _function(0.0, space)
    target = exact.func(0.0)[None, ...]
    observation = phx.terms.ModalObservationTerm(
        jnp.asarray([0.0]),
        target,
        function_name="u_hat",
        scalar_weight=10.0,
    )
    residual = phx.terms.CompiledModalResidualTerm(
        compiled,
        function_name="u_hat",
        times=jnp.asarray([0.0, 0.05, 0.1]),
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u_hat": trainable},
        terms=(observation, residual),
    )
    initial = solver.loss(key=jr.key(3))

    trained = solver.solve(
        num_iter=1,
        optim=optax.sgd(0.1),
        seed=4,
        jit=False,
        keep_best=False,
        log_every=0,
    )
    final = trained.loss(key=jr.key(3))

    assert final < initial
    assert trained.functions["u_hat"].func.model.scale > 0.0


def test_compiled_modal_residual_rejects_incompatible_discretization():
    _space, compiled = _compiled_heat(count=4)
    other_space, _ = _compiled_heat(count=6)
    term = phx.terms.CompiledModalResidualTerm(
        compiled,
        function_name="u_hat",
        times=jnp.asarray([0.0]),
    )

    with pytest.raises(ValueError, match="different discretizations"):
        term.loss({"u_hat": _function(1.0, other_space)})
