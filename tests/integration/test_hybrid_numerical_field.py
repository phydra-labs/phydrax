#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hybrid PINN/classical response of the two-conductivity plate.

The workflow is ``examples/hybrid_pinn_classical.py``: a P1 finite-element owner
on ``[0, 1]^2`` reads the interface heat flux of a ``SURROGATE`` network on
``[1, 2] x [0, 1]``; the accepted classical response of the interface-continuity
residual ``J`` is pulled back to the network's PARAMETER lane through
``StateDesignComponentAdmission``; one Dirichlet-Neumann step feeds the FE
interface trace back to the network as a fixed numerical field.

Independent references: the host analytic plate temperature
``tests._support.coupled_plate.exact_temperature``, a host NumPy dense solve of
the materialized FE linear system, the FE solution with the exact interface heat
flux (the P1 discretization floor, checked for second-order convergence), and
central finite differences of ``J`` along a random parameter direction evaluated
on those host solves.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest

import phydrax as phx
from examples import hybrid_pinn_classical as hybrid
from tests._support import coupled_plate as cp


@dataclass(frozen=True)
class _Stage:
    binding: phx.ComponentBinding
    linearization: phx.optim.StateDesignLinearization
    response: phx.optim.StateDesignResponseVJP


@dataclass(frozen=True)
class _Hybrid:
    region: hybrid.ClassicalRegion
    problem: phx.optim.StateDesignProblem
    points: hybrid.Collocation
    pretrained: _Stage
    coupled: _Stage


def _stage(
    region: hybrid.ClassicalRegion,
    problem: phx.optim.StateDesignProblem,
    network: hybrid.TemperatureNetwork,
) -> _Stage:
    binding = hybrid.surrogate_binding(network)
    linearization, response = hybrid.classical_response(region, problem, binding)
    return _Stage(binding, linearization, response)


@pytest.fixture(scope="module")
def region() -> hybrid.ClassicalRegion:
    return hybrid.classical_region()


@pytest.fixture(scope="module")
def workflow(region: hybrid.ClassicalRegion) -> _Hybrid:
    problem = hybrid.classical_problem(region)
    points = hybrid.collocation(region)
    network, _, _ = hybrid.train(
        hybrid.pinn_solver(
            hybrid.surrogate_network(jr.key(0)),
            points,
            hybrid.constant_interface_temperature(0.0),
        ),
        hybrid.PRETRAINING_STEPS,
    )
    pretrained = _stage(region, problem, network)
    view = hybrid.classical_field(region, pretrained.linearization.state)
    coupled, _, _ = hybrid.train(
        hybrid.pinn_solver(
            network,
            points,
            hybrid.classical_interface_temperature(view, points.interface),
        ),
        hybrid.COUPLING_STEPS,
    )
    return _Hybrid(region, problem, points, pretrained, _stage(region, problem, coupled))


def _exact(points: np.ndarray) -> np.ndarray:
    return cp.exact_temperature(
        points, hybrid.KAPPA_LEFT, hybrid.KAPPA_RIGHT, hybrid.HEAT_FLUX
    )


def _host_state(
    region: hybrid.ClassicalRegion,
    design: hybrid.TemperatureNetwork,
    held: hybrid.HeldLanes,
) -> np.ndarray:
    """Host dense solve of the materialized FE system with the network's flux load."""
    network = phx.combine_parameters(design, *held)
    system, right = region.problem.linear_system({hybrid.INTERFACE_FLUX: network})
    matrix = phx.linalg.materialize(
        system.operator,
        phx.linalg.MaterializationPolicy(max_entries=1_000_000, max_bytes=16 << 20),
    )
    return np.linalg.solve(np.asarray(matrix), np.asarray(right))


def _host_objective(
    workflow: _Hybrid, design: hybrid.TemperatureNetwork, held: hybrid.HeldLanes
) -> float:
    state = _host_state(workflow.region, design, held)
    value, _ = workflow.problem.value(jnp.asarray(state), design, held)
    return float(value)


def _axpy(
    scale: float, direction: hybrid.TemperatureNetwork, design: hybrid.TemperatureNetwork
) -> hybrid.TemperatureNetwork:
    return jax.tree.map(lambda d, x: x + scale * d, direction, design)


def test_accepted_hybrid_response_matches_host_finite_differences(
    workflow: _Hybrid,
) -> None:
    stage = workflow.pretrained
    response = stage.response
    assert response.adjoint_acceptance is not None
    assert bool(response.state_acceptance.accepted)
    assert bool(response.adjoint_acceptance.accepted)
    assert bool(response.accepted)
    component = response.component
    assert component is not None
    # The admission carried with the response is the one granted to this binding:
    # it returns exactly the design lane the response was differentiated at.
    for admitted, used in zip(
        jax.tree.leaves(component.design(stage.binding)),
        jax.tree.leaves(stage.linearization.design),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(admitted), np.asarray(used))

    held = hybrid.held_lanes(stage.binding)
    design = stage.linearization.design
    # The accepted primal is the host dense solution of the same FE system.
    np.testing.assert_allclose(
        np.asarray(stage.linearization.state),
        _host_state(workflow.region, design, held),
        rtol=1e-10,
        atol=1e-12,
    )

    leaves, structure = jax.tree.flatten(design)
    keys = jr.split(jr.key(7), len(leaves))
    direction = jax.tree.unflatten(
        structure,
        [
            jr.normal(key, leaf.shape, dtype=leaf.dtype)
            for key, leaf in zip(keys, leaves, strict=True)
        ],
    )
    # J is smooth in theta (tanh network, affine FE state): the central difference
    # error is O(step^2) ~ 1e-10 relative; rounding contributes ~ eps J / step.
    step = 1e-5
    forward = _host_objective(workflow, _axpy(step, direction, design), held)
    backward = _host_objective(workflow, _axpy(-step, direction, design), held)
    reference = (forward - backward) / (2.0 * step)
    derivative = sum(
        float(jnp.vdot(gradient, tangent))
        for gradient, tangent in zip(
            jax.tree.leaves(response.design_cotangent),
            jax.tree.leaves(direction),
            strict=True,
        )
    )
    assert abs(reference) > 1e-3
    assert derivative == pytest.approx(reference, rel=1e-6)


# kappa_right u'(1+) of the exact plate: the heat flux the right region delivers.
EXACT_INTERFACE_FLUX = hybrid.SOURCE + hybrid.HEAT_FLUX


def _exact_flux_network() -> hybrid.TemperatureNetwork:
    """The linear network ``u = (s + g) x / kappa_right``, whose interface flux is exact."""
    linear = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        hidden_sizes=(),
        use_final_bias=False,
        key=jr.key(0),
    )
    weight = (
        jnp.zeros_like(linear.layers[0].weight)
        .at[..., 0]
        .set(EXACT_INTERFACE_FLUX / hybrid.KAPPA_RIGHT)
    )
    return hybrid.TemperatureNetwork(
        eqx.tree_at(lambda mlp: mlp.layers[0].weight, linear, weight)
    )


def _nodal(
    region: hybrid.ClassicalRegion, network: hybrid.TemperatureNetwork
) -> np.ndarray:
    """Nodal FE temperature (host dense solve) with ``network``'s interface flux."""
    design, model_state, fixed = phx.partition_parameters(network)
    state = _host_state(region, design, (model_state, fixed))
    return np.asarray(region.problem.expand(jnp.asarray(state)))


def _discretization_error(region: hybrid.ClassicalRegion) -> np.ndarray:
    """P1 nodal error with the EXACT interface heat flux: the FE floor of the mesh.

    P1 is not nodally exact here although ``u`` is an ``x``-only quadratic: the
    diagonal triangulation gives the interface corners ``(1, 0)`` and ``(1, 1)``
    one and two triangles, so their consistent source loads ``s h^2 / 6`` and
    ``s h^2 / 3`` differ from the symmetric ``s h^2 / 4``.
    """
    return _nodal(region, _exact_flux_network()) - _exact(region.nodes)


def _interface_flux_error(network: hybrid.TemperatureNetwork) -> float:
    """``max |kappa_right du_theta/dx - (s + g)|`` densely on the interface ``x = 1``."""
    line = np.stack(
        (np.full(2001, hybrid.INTERFACE_X), np.linspace(0.0, 1.0, 2001)), axis=-1
    )
    gradient = np.asarray(jax.vmap(jax.grad(network))(jnp.asarray(line)))
    return float(
        np.max(np.abs(hybrid.KAPPA_RIGHT * gradient[:, 0] - EXACT_INTERFACE_FLUX))
    )


def _stage_network(stage: _Stage) -> hybrid.TemperatureNetwork:
    network = stage.binding.model
    assert isinstance(network, hybrid.TemperatureNetwork)
    return network


def test_finite_element_floor_converges_at_second_order(
    region: hybrid.ClassicalRegion,
) -> None:
    coarse = float(np.max(np.abs(_discretization_error(region))))
    fine_region = hybrid.classical_region(2 * hybrid.RESOLUTION)
    fine = float(np.max(np.abs(_discretization_error(fine_region))))
    # The pointwise P1 error in 2-D is O(h^2 |log h|): halving h from 1/N divides
    # it by at least 4 log(N) / log(2 N) (0.304 observed against 1/3 at N = 8).
    ratio = 0.25 * np.log(2 * hybrid.RESOLUTION) / np.log(hybrid.RESOLUTION)
    assert fine <= ratio * coarse


def test_dirichlet_neumann_coupling_reaches_the_finite_element_floor(
    workflow: _Hybrid,
) -> None:
    before, after = workflow.pretrained, workflow.coupled
    assert bool(after.response.accepted)
    # Pretraining held the interface at u = 0, so J measures the full interface
    # temperature jump (about ((3 s / 2 + g) / kappa_left)^2 / 2); one
    # Dirichlet-Neumann step closes it to the network's training accuracy.
    assert float(before.response.values) > 1.0
    assert float(after.response.values) < 1e-6
    assert float(after.response.values) < 1e-6 * float(before.response.values)

    region = workflow.region
    exact = _exact(region.nodes)
    floor_error = _discretization_error(region)
    floor = float(np.max(np.abs(floor_error)))
    # Classical side. The P1 stiffness of this right-triangle mesh is an M-matrix
    # (nonnegative inverse) and a unit interface flux loads the exactly
    # representable x / kappa_left, so an interface flux error dg moves the nodal
    # solution by at most max|dg| / kappa_left (discrete maximum principle); the
    # coupled FE error is then bounded by the floor plus that PINN trace term.
    for stage in (before, after):
        nodal = np.asarray(region.problem.expand(stage.linearization.state))
        flux_term = _interface_flux_error(_stage_network(stage)) / hybrid.KAPPA_LEFT
        assert np.max(np.abs(nodal - (exact + floor_error))) <= flux_term
        assert np.max(np.abs(nodal - exact)) <= floor + flux_term
    # Pretraining leaves the FE solution at its discretization floor. Observed:
    # floor 4.13e-3; flux terms 2.1e-5 (pretrained) and 4.6e-3 (coupled, whose
    # zero-mean flux error answers the FE trace error it was fitted to); coupled
    # FE error 2.85e-3.
    pretrained_flux = _interface_flux_error(_stage_network(before)) / hybrid.KAPPA_LEFT
    assert pretrained_flux <= 0.1 * floor

    # Network side. The coupled network was fitted to the pretrained FE trace. The
    # exact trace is constant in y, so the P1 trace error is bounded by the
    # interface nodal errors; the network error is that data error plus its fit
    # to the trace plus the error it reaches with exact interface data (the
    # pretrained network against the exact plate shifted to u(1) = 0).
    on_interface = np.isclose(region.nodes[:, 0], hybrid.INTERFACE_X)
    pretrained_nodal = np.asarray(region.problem.expand(before.linearization.state))
    data_error = float(np.max(np.abs(pretrained_nodal - exact)[on_interface]))
    order = np.argsort(region.nodes[on_interface, 1])
    sites = workflow.points.interface
    trace = np.interp(
        sites[:, 1],
        region.nodes[on_interface, 1][order],
        pretrained_nodal[on_interface][order],
    )
    coupled_network = _stage_network(after)
    fit = float(
        np.max(np.abs(np.asarray(jax.vmap(coupled_network)(jnp.asarray(sites))) - trace))
    )
    probes = np.stack(
        np.meshgrid(np.linspace(1.0, 2.0, 11), np.linspace(0.0, 1.0, 11), indexing="ij"),
        axis=-1,
    ).reshape(-1, 2)
    shift = _exact(np.asarray([[hybrid.INTERFACE_X, 0.0]]))[0]
    own = float(
        np.max(
            np.abs(
                np.asarray(jax.vmap(_stage_network(before))(jnp.asarray(probes)))
                - (_exact(probes) - shift)
            )
        )
    )
    surrogate = float(
        np.max(
            np.abs(
                np.asarray(jax.vmap(coupled_network)(jnp.asarray(probes)))
                - _exact(probes)
            )
        )
    )
    # Observed: data 4.13e-3, fit 1.42e-3, own 3.5e-6, network error 2.57e-3.
    assert own <= 0.1 * data_error
    assert surrogate <= data_error + fit + own


def test_surrogate_is_refused_as_a_solution_map_design() -> None:
    binding = hybrid.surrogate_binding(hybrid.surrogate_network(jr.key(1)))
    with pytest.raises(
        ValueError,
        match=r"surrogate component cannot supply the design of a state-design "
        r"response consumed under \(direct, solution-map\)",
    ):
        phx.optim.StateDesignComponentAdmission(
            binding,
            kind=phx.ObjectiveKind.SOLUTION_MAP,
            policy=hybrid.UNDECLARED_REGULARITY,
        )


def _never_called(*_: object) -> NoReturn:
    raise AssertionError("a refused objective never binds or measures")


def test_implicit_solver_objective_refuses_to_train_the_surrogate() -> None:
    binding = hybrid.surrogate_binding(hybrid.surrogate_network(jr.key(1)))
    objective = phx.solver.SolverObjective(
        None, _never_called, _never_called, objective_id="hybrid-solution-map"
    )
    message = r"no admissible training signal for \['surrogate'\]"
    with pytest.raises(ValueError, match=message):
        objective.evaluate(binding)
    with pytest.raises(ValueError, match=message):
        phx.solver.train_components(
            binding, (objective,), optimizer=optax.adam(1e-3), steps=1, key=jr.key(2)
        )


def _wider_lane(network: hybrid.TemperatureNetwork) -> hybrid.TemperatureNetwork:
    del network
    wider = hybrid.TemperatureNetwork(
        phx.nn.models.MLP(
            in_size=2,
            out_size="scalar",
            width_size=8,
            depth=2,
            activation=jnp.tanh,
            key=jr.key(3),
        )
    )
    parameters, _, _ = phx.partition_parameters(wider)
    return parameters


def _float32_lane(network: hybrid.TemperatureNetwork) -> hybrid.TemperatureNetwork:
    parameters, _, _ = phx.partition_parameters(network)
    return jax.tree.map(lambda leaf: leaf.astype(jnp.float32), parameters)


@pytest.mark.parametrize(
    "lane", [_wider_lane, _float32_lane], ids=("other-structure", "other-dtype")
)
def test_design_outside_the_admitted_lane_is_refused(
    region: hybrid.ClassicalRegion,
    lane: Callable[[hybrid.TemperatureNetwork], hybrid.TemperatureNetwork],
) -> None:
    network = hybrid.surrogate_network(jr.key(1))
    binding = hybrid.surrogate_binding(network)
    with pytest.raises(ValueError, match="not the PARAMETER lane of the admitted"):
        phx.optim.prepare_state_design_linearization(
            hybrid.classical_problem(region),
            lane(network),
            region.problem.state_space.zeros(),
            args=hybrid.held_lanes(binding),
            linear_policy=hybrid.dense_policy(),
            component=hybrid.admit(binding),
        )
