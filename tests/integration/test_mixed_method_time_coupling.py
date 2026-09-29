#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nonmatching FE/FV conjugate heat with independent native time integration.

A P1 finite-element solid on [0, 1]² (backward Euler through one prepared LU
factorization) and a cell-centered finite-volume fluid on [1, 2] × [0, 1]
(SSPRK(3,3) with a face-heat accumulator in its state) meet on x = 1 with four
solid edges against three fluid faces. The solid interface-temperature waveform
reaches the fluid through time interpolation and exact face averages I; the
fluid's whole-window face heat (extensive flux moments) reaches the solid nodal
loads (a counting functional) through the conservative map P = Iᵀ.

Reference: exp(t A) of the coupled semi-discrete system assembled on the host
from hand-written P1 element matrices and the two-point fluid stencil (SciPy),
independent of the coupling runtime, the finite-element owner, and both
integrators. The partitioned scheme is first order in the window (backward Euler
steps and the uniform-rate spending of each window's heat).
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
from jax import Array

import phydrax as phx


cpl = phx.solver.coupling
_DIVISIONS = 4
_CELLS = 3
_SOLID_CAPACITY = 2.0
_SOLID_CONDUCTIVITY = 1.0
_FLUID_CAPACITY = 1.0
_CONDUCTANCE = 0.25
_FINAL_TIME = 0.4
_KEY_IMPL = "threefry2x32"
_HEAT = cpl.CouplingQuantity("heat", phx.units.JOULE, sign_convention="into-solid")
_FLUID_CELL_COUNT = _CELLS * _CELLS
# Fluid faces as (owner, neighbor, unit conductance) and the interface cells;
# cell (i, j) has index 3 i + j, i along x.
_CELL_INDEX = np.arange(_FLUID_CELL_COUNT).reshape(_CELLS, _CELLS)
_OWNERS = np.concatenate((_CELL_INDEX[:-1].reshape(-1), _CELL_INDEX[:, :-1].reshape(-1)))
_NEIGHBORS = np.concatenate((_CELL_INDEX[1:].reshape(-1), _CELL_INDEX[:, 1:].reshape(-1)))
_INTERFACE_CELLS = _CELL_INDEX[0]
# Square cells: unit face conductance 1, and 2 through the half-cell interface gap.
_INTERFACE_UNIT_CONDUCTANCE = 2.0
_CELL_AREA = 1.0 / _FLUID_CELL_COUNT


def _solid_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.linspace(0.0, 1.0, _DIVISIONS + 1)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    vertices = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    index = np.arange(vertices.shape[0]).reshape(x.shape)
    lower = np.stack((index[:-1, :-1], index[1:, :-1], index[1:, 1:]), axis=-1)
    upper = np.stack((index[:-1, :-1], index[1:, 1:], index[:-1, 1:]), axis=-1)
    triangles = np.concatenate((lower.reshape(-1, 3), upper.reshape(-1, 3)))
    return vertices, triangles.astype(np.int32), index[-1]


def _face_averages() -> np.ndarray:
    """Exact fluid-face averages of the solid interface hat functions."""
    nodes = np.linspace(0.0, 1.0, _DIVISIONS + 1)
    edges = np.linspace(0.0, 1.0, _CELLS + 1)
    breaks = np.union1d(nodes, edges)
    matrix = np.zeros((_CELLS, nodes.size))
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        face = np.searchsorted(edges, 0.5 * (left + right)) - 1
        for node in range(nodes.size):
            values = np.interp((left, right), nodes, np.eye(nodes.size)[node])
            matrix[face, node] += 0.5 * (right - left) * (values[0] + values[1])
    return matrix / np.diff(edges)[:, None]


def _field(
    name: str, representation: phx.discretization.FieldRepresentation, count: int, /
) -> phx.discretization.DiscreteFieldSpace:
    layout = phx.discretization.EntityDofLayout(f"{name}/entities", count, count)
    space = phx.linalg.ArraySpace((count,), dtype=jnp.float64, space_id=f"{name}/dofs")
    return phx.discretization.DiscreteFieldSpace(
        name, f"{name}/support", layout, space, representation=representation
    )


_TRACE = _field("solid-trace", "basis_coefficient", _DIVISIONS + 1)
_LOADS = _field("solid-loads", "functional", _DIVISIONS + 1)
_FACE_TEMPERATURE = _field("fluid-face-temperature", "cell_average", _CELLS)
_FACE_HEAT = _field("fluid-face-heat", "flux_moment", _CELLS)


def _transfer(
    source: phx.discretization.DiscreteFieldSpace,
    target: phx.discretization.DiscreteFieldSpace,
    matrix: np.ndarray,
    properties: phx.discretization.TransferProperties,
    /,
) -> phx.discretization.FieldTransfer:
    values = jnp.asarray(matrix)
    return phx.discretization.FieldTransfer(
        source,
        target,
        phx.linalg.FunctionLinearOperator(
            lambda vector: values @ vector,
            source=source.vector_space,
            target=target.vector_space,
            transpose_action=lambda covector: values.T @ covector,
        ),
        dual_pullback_operator=phx.linalg.FunctionLinearOperator(
            lambda covector: values.T @ covector,
            source=target.vector_space,
            target=source.vector_space,
        ),
        properties=properties,
    )


class _SolidBackwardEuler(phx.solver.AbstractFixedStepMethod):
    """(ρc M + Δt k K) Tⁿ⁺¹ = ρc M Tⁿ + Sᵀ ℓ with one prepared factorization."""

    discretization: Any
    factorization: phx.linalg.PreparedLinearSolve
    interface_nodes: Array
    prepared_step: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: Any,
        factorization: phx.linalg.PreparedLinearSolve,
        interface_nodes: Array,
        step: float,
        /,
    ) -> None:
        self.discretization = discretization
        self.factorization = factorization
        self.interface_nodes = interface_nodes
        self.prepared_step = step
        self.method_id = f"solid-backward-euler:{step!r}"

    @property
    def required_step_size(self) -> float:
        return self.prepared_step

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def step(
        self, step_index: Any, time: Any, state: Any, step_size: Any, args: Any, /
    ) -> phx.solver.FixedStepResult:
        del step_index, time
        right_hand_side = _SOLID_CAPACITY * self.discretization.mass(state)
        solved = phx.linalg.solve(
            self.factorization, right_hand_side.at[self.interface_nodes].add(args)
        )
        tolerance = 64 * jnp.finfo(state.dtype).eps * self.prepared_step
        prepared = jnp.abs(step_size - self.prepared_step) <= tolerance
        return phx.solver.FixedStepResult(
            candidate_state=solved.value,
            accepted_state=solved.value,
            successful=jnp.all(solved.successful) & prepared,
            residual=jnp.zeros((), dtype=state.dtype),
            iterations=jnp.asarray(1, dtype=jnp.int32),
            work=jnp.asarray(1, dtype=jnp.int32),
            transform_applied=jnp.asarray(False),
            transform_correction_norm=jnp.zeros((), dtype=state.dtype),
        )


def _waveform_plan(steps: int, name: str, /) -> phx.solver.coupling.CouplingWaveformPlan:
    nodes = tuple(np.linspace(0.0, 1.0, steps + 1))
    return cpl.CouplingWaveformPlan(steps + 1, 1, nodes, plan_id=f"{name}-{steps}")


def _solid(
    window_size: float, steps: int, /
) -> phx.solver.coupling.FixedStepCouplingParticipant:
    vertices, triangles, interface_nodes = _solid_mesh()
    discretization = phx.discretization.FiniteElementPlan(
        phx.discretization.CellMesh.from_triangles(
            jnp.asarray(vertices), jnp.asarray(triangles)
        ),
        phx.discretization.FiniteElementFieldSpec(
            "T", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    step = window_size / steps
    system, _ = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "solid-backward-euler",
            "T",
            (
                phx.equations.MassAction("T", _SOLID_CAPACITY),
                phx.equations.DiffusionAction("T", step * _SOLID_CONDUCTIVITY),
            ),
        ),
        discretization,
    ).linear_system()
    method = _SolidBackwardEuler(
        discretization,
        phx.linalg.prepare(
            system,
            phx.linalg.LinearSolvePolicy(
                phx.linalg.DenseLU(),
                materialization=phx.linalg.MaterializationPolicy(max_entries=4096),
            ),
        ),
        jnp.asarray(interface_nodes, dtype=jnp.int32),
        step,
    )
    load_total = cpl.CouplingMeasurement(
        phx.linalg.FunctionLinearOperator(
            lambda loads: jnp.sum(loads, keepdims=True),
            source=_LOADS.vector_space,
            target=phx.linalg.ArraySpace((1,), dtype=jnp.float64),
            transpose_action=lambda amount: jnp.broadcast_to(amount, (_DIVISIONS + 1,)),
        ),
        phx.units.ONE,
        representation="functional",
        support_id=_LOADS.support_id,
        provenance_id="solid-p1-basis-loads",
        normalization="counting",
    )
    return cpl.FixedStepCouplingParticipant(
        method,
        # Each step spends a uniform share of the window's nodal loads.
        lambda window, views, model_state, key, args: cpl.MethodWindowBinding(
            views[0], model_state
        ),
        lambda native, args: (native[method.interface_nodes],),
        subsystem_id="solid-fe",
        substeps=steps,
        input_ports=(
            cpl.CouplingPort(
                "solid-fe/loads",
                "input",
                _LOADS.vector_space,
                field_space=_LOADS,
                quantity=_HEAT,
                measurement=load_total,
                temporal_kind="interval_integral",
                reference_scale=1.0,
            ),
        ),
        output_ports=(
            cpl.CouplingPort(
                "solid-fe/temperature",
                "output",
                _TRACE.vector_space,
                field_space=_TRACE,
                waveform_plan=_waveform_plan(steps, "solid-steps"),
                reference_scale=1.0,
            ),
        ),
    )


def _fluid_rate(time: Array, state: Array, args: Any) -> Array:
    start, end, interface_start, interface_end, conductance = args
    temperature = state[:_FLUID_CELL_COUNT]
    interface = interface_start + (time - start) / (end - start) * (
        interface_end - interface_start
    )
    flux = conductance * (temperature[_OWNERS] - temperature[_NEIGHBORS])
    # Heat leaving each interface face toward the solid.
    face_heat = (
        conductance
        * _INTERFACE_UNIT_CONDUCTANCE
        * (temperature[_INTERFACE_CELLS] - interface)
    )
    balance = (
        jnp.zeros_like(temperature)
        .at[_OWNERS]
        .add(-flux)
        .at[_NEIGHBORS]
        .add(flux)
        .at[_INTERFACE_CELLS]
        .add(-face_heat)
    )
    return jnp.concatenate((balance / (_FLUID_CAPACITY * _CELL_AREA), face_heat))


def _fluid(
    steps: int, randomness: phx.solver.coupling.MethodParticipantRandomness, /
) -> phx.solver.coupling.FixedStepCouplingParticipant:
    def bind(
        window: phx.solver.coupling.CouplingWindow,
        views: tuple[Any, ...],
        model_state: tuple[Array, Array],
        key: Array | None,
        args: None,
    ) -> phx.solver.coupling.MethodWindowBinding:
        del args
        conductance, applied_steps = model_state
        if key is not None:
            conductance = conductance * jnp.exp(
                0.05 * jax.random.normal(key, (), dtype=jnp.float64)
            )
        start, end = views[0]
        return cpl.MethodWindowBinding(
            (window.start, window.end, start, end, conductance),
            (model_state[0], applied_steps + 1),
        )

    return cpl.FixedStepCouplingParticipant(
        phx.solver.SSPRK33FixedStepMethod(_fluid_rate),
        bind,
        lambda native, args: (),
        subsystem_id="fluid-fv",
        substeps=steps,
        input_ports=(
            cpl.CouplingPort(
                "fluid-fv/temperature",
                "input",
                _FACE_TEMPERATURE.vector_space,
                field_space=_FACE_TEMPERATURE,
                waveform_plan=_waveform_plan(steps, "fluid-steps"),
                reference_scale=1.0,
            ),
        ),
        output_ports=(
            cpl.CouplingPort(
                "fluid-fv/face-heat",
                "output",
                _FACE_HEAT.vector_space,
                field_space=_FACE_HEAT,
                quantity=_HEAT,
                measurement=cpl.CouplingMeasurement.extensive(
                    _FACE_HEAT.vector_space,
                    _FACE_HEAT.support_id,
                    provenance_id="fluid-two-point-face-heat",
                ),
                temporal_kind="interval_integral",
                reference_scale=1.0,
            ),
        ),
        amounts=lambda start, end, args: (
            end[_FLUID_CELL_COUNT:] - start[_FLUID_CELL_COUNT:],
        ),
        randomness=randomness,
        key_impl=_KEY_IMPL if randomness == "carried-key" else None,
    )


def _initial_temperatures() -> tuple[np.ndarray, np.ndarray]:
    vertices, _, _ = _solid_mesh()
    solid = 1.0 + 0.5 * np.cos(np.pi * vertices[:, 1]) * vertices[:, 0]
    fluid = np.tile(0.25 * np.arange(_CELLS) / _CELLS, _CELLS)
    return solid, fluid


def _problem(
    window_size: float,
    solid_steps: int,
    fluid_steps: int,
    /,
    *,
    randomness: phx.solver.coupling.MethodParticipantRandomness = "none",
    key: Array | None = None,
    maximum_steps: int = 60,
) -> phx.solver.coupling.CouplingProblem:
    solid = _solid(window_size, solid_steps)
    fluid = _fluid(fluid_steps, randomness)
    averages = _face_averages()
    exchanges = (
        cpl.CouplingExchange(
            "interface-temperature",
            "solid-fe/temperature",
            "fluid-fv/temperature",
            transfer=_transfer(
                _TRACE,
                _FACE_TEMPERATURE,
                averages,
                phx.discretization.TransferProperties(constant_preserving=True),
            ),
            requirement=cpl.CouplingTransferRequirement(constant_preserving=True),
            temporal=cpl.CouplingTemporalConversion(
                "interpolate", transfer=cpl.BarycentricCouplingTemporalTransfer(1)
            ),
        ),
        cpl.CouplingExchange(
            "interface-heat",
            "fluid-fv/face-heat",
            "solid-fe/loads",
            transfer=_transfer(
                _FACE_HEAT,
                _LOADS,
                averages.T,
                phx.discretization.TransferProperties(conservative=True),
            ),
            requirement=cpl.CouplingTransferRequirement(conservative=True),
            temporal=cpl.CouplingTemporalConversion("window-integral"),
        ),
    )
    policy = cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-11,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=1e-14,
            maximum_steps=maximum_steps,
        ),
        (
            cpl.CouplingTolerance("fluid-fv/temperature", absolute=1e-10),
            cpl.CouplingTolerance("solid-fe/loads", absolute=1e-10),
        ),
        fixed_point_sweep=cpl.CouplingSweep(
            "gauss-seidel", subsystem_order=("fluid-fv", "solid-fe")
        ),
    )
    solid_initial, fluid_initial = _initial_temperatures()
    states = {
        "solid-fe": solid.initial_state(jnp.asarray(solid_initial)),
        "fluid-fv": fluid.initial_state(
            jnp.concatenate((jnp.asarray(fluid_initial), jnp.zeros(_CELLS))),
            model_state=(
                jnp.asarray(_CONDUCTANCE, dtype=jnp.float64),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            key=key,
        ),
    }
    return cpl.lower_partitioned_coupling(
        cpl.PartitionedCouplingDeclaration((solid, fluid), exchanges, policy),
        states,
        t0=0.0,
        t1=_FINAL_TIME,
        window_size=window_size,
    )


def _participant(state: phx.solver.coupling.CouplingState, subsystem_id: str, /) -> Any:
    return state.participant_states[state.subsystem_ids.index(subsystem_id)]


def _temperatures(state: phx.solver.coupling.CouplingState, /) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(_participant(state, "solid-fe").native),
            np.asarray(_participant(state, "fluid-fv").native)[:_FLUID_CELL_COUNT],
        )
    )


def _p1_matrices() -> tuple[np.ndarray, np.ndarray]:
    vertices, triangles, _ = _solid_mesh()
    count = vertices.shape[0]
    mass = np.zeros((count, count))
    stiffness = np.zeros((count, count))
    gradients_reference = np.asarray(((-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0)))
    for triangle in triangles:
        points = vertices[triangle]
        jacobian = np.stack((points[1] - points[0], points[2] - points[0]), axis=1)
        area = 0.5 * abs(np.linalg.det(jacobian))
        gradients = np.linalg.solve(jacobian.T, gradients_reference)
        mass[np.ix_(triangle, triangle)] += area * (np.ones((3, 3)) + np.eye(3)) / 12
        stiffness[np.ix_(triangle, triangle)] += area * gradients.T @ gradients
    return mass, stiffness


def _reference(time: float, /) -> np.ndarray:
    """exp(t A) of the assembled coupled semi-discrete system."""
    _, _, interface_nodes = _solid_mesh()
    mass, stiffness = _p1_matrices()
    trace = np.zeros((interface_nodes.size, mass.shape[0]))
    trace[np.arange(interface_nodes.size), interface_nodes] = 1.0
    fluid_trace = np.zeros((_CELLS, _FLUID_CELL_COUNT))
    fluid_trace[np.arange(_CELLS), _INTERFACE_CELLS] = 1.0
    laplacian = np.zeros((_FLUID_CELL_COUNT, _FLUID_CELL_COUNT))
    np.add.at(laplacian, (_OWNERS, _OWNERS), 1.0)
    np.add.at(laplacian, (_NEIGHBORS, _NEIGHBORS), 1.0)
    np.add.at(laplacian, (_OWNERS, _NEIGHBORS), -1.0)
    np.add.at(laplacian, (_NEIGHBORS, _OWNERS), -1.0)
    average = _face_averages() @ trace
    face = _CONDUCTANCE * _INTERFACE_UNIT_CONDUCTANCE
    generator = np.block(
        [
            [
                -_SOLID_CONDUCTIVITY * stiffness - face * average.T @ average,
                face * average.T @ fluid_trace,
            ],
            [
                face * fluid_trace.T @ average,
                -_CONDUCTANCE * laplacian - face * fluid_trace.T @ fluid_trace,
            ],
        ]
    )
    capacity = scipy.linalg.block_diag(
        _SOLID_CAPACITY * mass, _FLUID_CAPACITY * _CELL_AREA * np.eye(_FLUID_CELL_COUNT)
    )
    count = mass.shape[0] + _FLUID_CELL_COUNT
    rates = np.zeros((count, count))
    rates[...] = np.linalg.solve(capacity, generator)
    return scipy.linalg.expm(time * rates) @ np.concatenate(_initial_temperatures())


def _energies(temperatures: np.ndarray, /) -> tuple[float, float]:
    """Heat content of the solid (∫ ρc T_h) and of the fluid cells."""
    mass, _ = _p1_matrices()
    return (
        float(_SOLID_CAPACITY * np.sum(mass @ temperatures[: mass.shape[0]])),
        float(_FLUID_CAPACITY * _CELL_AREA * np.sum(temperatures[mass.shape[0] :])),
    )


@pytest.mark.parametrize(
    ("solid_steps", "fluid_steps"),
    ((2, 3), (4, 1)),
    ids=("backward-euler-2-ssprk33-3", "backward-euler-4-ssprk33-1"),
)
def test_independent_local_integrators_converge_first_order_in_the_window(
    solid_steps: int, fluid_steps: int
) -> None:
    reference = _reference(_FINAL_TIME)
    errors = []
    for window_size in (0.1, 0.05, 0.025):
        solution = cpl.solve_coupling(_problem(window_size, solid_steps, fluid_steps))
        assert bool(solution.successful)
        assert bool(jnp.all(solution.converged))
        errors.append(np.max(np.abs(_temperatures(solution.final_state) - reference)))

    rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    np.testing.assert_array_less(np.abs(rates - 1.0), 0.2)
    assert errors[-1] < 5e-3


def test_consumed_interface_heat_balances_the_ledger_and_total_energy() -> None:
    solution = cpl.solve_coupling(_problem(0.05, 2, 3))

    final = solution.final_state
    assert bool(solution.successful)
    rows = dict(
        zip(
            final.budget_row_ids,
            np.asarray(final.cumulative_exchange_budget),
            strict=True,
        )
    )
    solid_start, fluid_start = _energies(np.concatenate(_initial_temperatures()))
    solid_end, fluid_end = _energies(_temperatures(final))
    debit, credit = rows["interface-heat"]
    # The ledger records the heat the fluid actually spent and the solid consumed.
    assert abs(debit + credit) <= 64 * np.finfo(np.float64).eps * abs(credit)
    assert abs(credit) > 0.1
    assert solid_end - solid_start == pytest.approx(credit, rel=1e-12, abs=1e-14)
    assert fluid_end - fluid_start == pytest.approx(debit, rel=1e-12, abs=1e-14)
    np.testing.assert_array_equal(rows["interface-temperature"], [0.0, 0.0])
    total = solid_start + fluid_start
    assert abs(solid_end + fluid_end - total) <= 1e-14 * abs(total)


def _assert_work_counts_every_iterate(
    result: phx.solver.coupling.CouplingWindowResult, /
) -> None:
    """Every interface evaluation replays both owners from the checkpoint.

    Per evaluation the fluid takes three SSPRK(3,3) steps of three stage rates and
    the solid two backward Euler steps of one prepared solve each.
    """
    diagnostics = result.diagnostics
    evaluations = int(diagnostics.nonlinear_residual_evaluations) + 1
    per_evaluation = {"fluid-fv": 9, "solid-fe": 2}
    subsystems = result.accepted_state.subsystem_ids
    assert diagnostics.counts_complete
    np.testing.assert_array_equal(
        diagnostics.participant_evaluations, [evaluations] * len(subsystems)
    )
    np.testing.assert_array_equal(
        diagnostics.participant_work,
        [evaluations * per_evaluation[subsystem] for subsystem in subsystems],
    )


def test_rejected_window_rolls_back_carried_key_and_model_state() -> None:
    key = jax.random.key(7, impl=_KEY_IMPL)
    exhausted = _problem(0.05, 2, 3, randomness="carried-key", key=key, maximum_steps=2)
    prepared = exhausted.prepare()
    checkpoint = prepared.reference_state

    rejected = cpl.advance_coupling_window(prepared, checkpoint, 0.05)
    replay = cpl.advance_coupling_window(prepared, checkpoint, 0.05)

    once = jax.random.key_data(jax.random.split(key)[1])
    candidate = _participant(rejected.candidate_state, "fluid-fv")
    assert int(rejected.status) == int(cpl.CouplingStatus.WORK_EXHAUSTED)
    assert bool(eqx.tree_equal(rejected.accepted_state, checkpoint))
    assert int(candidate.accepted_windows) == 1
    assert int(candidate.model_state[1]) == 3
    np.testing.assert_array_equal(candidate.key_data, once)
    assert bool(eqx.tree_equal(replay, rejected))
    # The rejected iterates' work is evidence too, not only accepted work.
    _assert_work_counts_every_iterate(rejected)

    converging = _problem(0.05, 2, 3, randomness="carried-key", key=key).prepare()
    accepted = cpl.advance_coupling_window(converging, converging.reference_state, 0.05)

    committed = _participant(accepted.accepted_state, "fluid-fv")
    assert bool(accepted.successful)
    assert int(accepted.diagnostics.coupling_iterations) > 2
    # Every iterate replayed the checkpoint: the state advanced exactly once.
    assert int(committed.accepted_windows) == 1
    assert int(committed.model_state[1]) == 3
    np.testing.assert_array_equal(committed.key_data, once)
    _assert_work_counts_every_iterate(accepted)
