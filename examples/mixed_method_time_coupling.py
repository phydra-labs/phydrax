#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mixed-method conjugate heat transfer with independent local time integration.

A P1 finite-element solid on [0, 1]² and a cell-centered finite-volume fluid slab
on [1, 2] × [0, 1] meet on the nonmatching interface x = 1 (four solid edges,
three fluid faces). Each owner keeps its native integrator and its own step count
per coupling window:

- the solid takes backward Euler steps through one prepared LU factorization of
  its finite-element operator ρc M + Δt k K, reused for every step;
- the fluid takes SSPRK(3,3) steps and integrates the heat leaving each interface
  face in a native accumulator carried in its state.

The solid publishes its interface-temperature waveform on its own step nodes; the
exchange interpolates it in time onto the fluid step nodes and averages the P1
trace over each fluid face (I). The fluid publishes the whole-window face heat
(flux moments, extensive storage); the conservative map P = Iᵀ turns it into the
consistent nodal loads of the solid basis (a declared counting functional), and
preparation certifies L_solid P = L_fluid through transposed actions. Every window
is solved implicitly; each interface iterate replays one accepted checkpoint.

The independent reference is the matrix exponential of the assembled coupled
semi-discrete system (host SciPy, hand-assembled P1 matrices).
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax import Array

import phydrax as phx


cpl = phx.solver.coupling
SOLID_DIVISIONS = 4
FLUID_CELLS = (3, 3)
SOLID_HEAT_CAPACITY = 2.0
SOLID_CONDUCTIVITY = 1.0
FLUID_HEAT_CAPACITY = 1.0
FLUID_CONDUCTANCE = 0.25
FINAL_TIME = 0.4
WINDOW_SIZES = (0.1, 0.05, 0.025)
# (label, solid backward-Euler steps, fluid SSPRK(3,3) steps) per window.
INTEGRATORS = (
    ("backward Euler x2 | SSPRK(3,3) x3", 2, 3),
    ("backward Euler x4 | SSPRK(3,3) x1", 4, 1),
)
KEY_IMPL = "threefry2x32"
HEAT = cpl.CouplingQuantity("heat", phx.units.JOULE, sign_convention="into-solid")


# Host preparation: meshes, the interface transfer, and the fluid face graph.


def solid_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-triangle mesh of [0, 1]² and its interface vertices on x = 1."""
    coordinates = np.linspace(0.0, 1.0, SOLID_DIVISIONS + 1)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    vertices = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    index = np.arange(vertices.shape[0]).reshape(x.shape)
    lower = np.stack((index[:-1, :-1], index[1:, :-1], index[1:, 1:]), axis=-1)
    upper = np.stack((index[:-1, :-1], index[1:, 1:], index[:-1, 1:]), axis=-1)
    triangles = np.concatenate((lower.reshape(-1, 3), upper.reshape(-1, 3)))
    return vertices, triangles.astype(np.int32), index[-1, :]


def hat_face_averages() -> np.ndarray:
    """Exact fluid-face averages of every piecewise-linear solid interface hat."""
    nodes = np.linspace(0.0, 1.0, SOLID_DIVISIONS + 1)
    edges = np.linspace(0.0, 1.0, FLUID_CELLS[1] + 1)
    breaks = np.union1d(nodes, edges)
    matrix = np.zeros((edges.size - 1, nodes.size))
    identity = np.eye(nodes.size)
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        face = np.searchsorted(edges, 0.5 * (left + right)) - 1
        for node in range(nodes.size):
            values = np.interp((left, right), nodes, identity[node])
            matrix[face, node] += 0.5 * (right - left) * (values[0] + values[1])
    return matrix / np.diff(edges)[:, None]


class FluidGrid(NamedTuple):
    """Two-point conduction faces of the uniform fluid cell grid."""

    owners: Array
    neighbors: Array
    unit_conductances: Array
    interface_cells: Array
    interface_unit_conductance: float
    cell_area: float
    cell_count: int


def fluid_grid() -> FluidGrid:
    nx, ny = FLUID_CELLS
    hx, hy = 1.0 / nx, 1.0 / ny
    cell = np.arange(nx * ny).reshape(nx, ny)
    owners = np.concatenate((cell[:-1, :].reshape(-1), cell[:, :-1].reshape(-1)))
    neighbors = np.concatenate((cell[1:, :].reshape(-1), cell[:, 1:].reshape(-1)))
    conductances = np.concatenate(
        (np.full((nx - 1) * ny, hy / hx), np.full(nx * (ny - 1), hx / hy))
    )
    return FluidGrid(
        jnp.asarray(owners, dtype=jnp.int32),
        jnp.asarray(neighbors, dtype=jnp.int32),
        jnp.asarray(conductances, dtype=jnp.float64),
        jnp.asarray(cell[0, :], dtype=jnp.int32),
        # Interface faces see the cell center at half a cell width.
        hy / (0.5 * hx),
        hx * hy,
        nx * ny,
    )


def field(
    name: str, representation: phx.discretization.FieldRepresentation, count: int, /
) -> phx.discretization.DiscreteFieldSpace:
    layout = phx.discretization.EntityDofLayout(f"{name}/entities", count, count)
    space = phx.linalg.ArraySpace((count,), dtype=jnp.float64, space_id=f"{name}/dofs")
    return phx.discretization.DiscreteFieldSpace(
        name, f"{name}/support", layout, space, representation=representation
    )


SOLID_TRACE = field("solid-interface-trace", "basis_coefficient", SOLID_DIVISIONS + 1)
SOLID_LOADS = field("solid-interface-loads", "functional", SOLID_DIVISIONS + 1)
FLUID_TEMPERATURE = field("fluid-interface-temperature", "cell_average", FLUID_CELLS[1])
FLUID_FACE_HEAT = field("fluid-interface-face-heat", "flux_moment", FLUID_CELLS[1])


def matrix_transfer(
    source: phx.discretization.DiscreteFieldSpace,
    target: phx.discretization.DiscreteFieldSpace,
    matrix: np.ndarray,
    properties: phx.discretization.TransferProperties,
    name: str,
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
            operator_id=f"{name}/action",
        ),
        dual_pullback_operator=phx.linalg.FunctionLinearOperator(
            lambda covector: values.T @ covector,
            source=target.vector_space,
            target=source.vector_space,
            operator_id=f"{name}/pullback",
        ),
        properties=properties,
    )


# Solid owner: backward Euler with one prepared factorization for every step.


class SolidBackwardEuler(phx.solver.AbstractFixedStepMethod):
    """(ρc M + Δt k K) Tⁿ⁺¹ = ρc M Tⁿ + Sᵀ ℓ with interface nodal loads ℓ."""

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
        loads = args
        right_hand_side = SOLID_HEAT_CAPACITY * self.discretization.mass(state)
        right_hand_side = right_hand_side.at[self.interface_nodes].add(loads)
        solved = phx.linalg.solve(self.factorization, right_hand_side)
        # The factorization belongs to one step size; any other step fails.
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


def solid_loads_measurement() -> phx.solver.coupling.CouplingMeasurement:
    """Total heat of consistent nodal loads: every basis load counted once."""
    count = SOLID_DIVISIONS + 1
    return cpl.CouplingMeasurement(
        phx.linalg.FunctionLinearOperator(
            lambda loads: jnp.sum(loads, keepdims=True),
            source=SOLID_LOADS.vector_space,
            target=phx.linalg.ArraySpace((1,), dtype=jnp.float64),
            transpose_action=lambda amount: jnp.broadcast_to(amount, (count,)),
            operator_id="solid-interface-load-total",
        ),
        phx.units.ONE,
        representation="functional",
        support_id=SOLID_LOADS.support_id,
        provenance_id="solid-p1-basis-loads",
        normalization="counting",
    )


def solid_participant(
    window_size: float, substeps: int, /
) -> tuple[phx.solver.coupling.FixedStepCouplingParticipant, Array]:
    vertices, triangles, interface_nodes = solid_mesh()
    mesh = phx.discretization.CellMesh.from_triangles(
        jnp.asarray(vertices), jnp.asarray(triangles)
    )
    temperature = phx.discretization.FiniteElementFieldSpec(
        "T", phx.discretization.lagrange_element("triangle", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, temperature).prepare()
    step = window_size / substeps
    form = phx.equations.FiniteElementForm(
        "solid-backward-euler",
        "T",
        (
            phx.equations.MassAction("T", SOLID_HEAT_CAPACITY),
            phx.equations.DiffusionAction("T", step * SOLID_CONDUCTIVITY),
        ),
    )
    system, _ = phx.equations.compile_finite_element_problem(
        form, discretization
    ).linear_system()
    factorization = phx.linalg.prepare(
        system,
        phx.linalg.LinearSolvePolicy(
            phx.linalg.DenseLU(),
            materialization=phx.linalg.MaterializationPolicy(max_entries=4096),
        ),
    )
    method = SolidBackwardEuler(
        discretization,
        factorization,
        jnp.asarray(interface_nodes, dtype=jnp.int32),
        step,
    )

    def bind(
        window: phx.solver.coupling.CouplingWindow,
        views: tuple[Any, ...],
        model_state: None,
        key: None,
        args: None,
    ) -> phx.solver.coupling.MethodWindowBinding:
        del window, key, args
        # views[0] is this step's uniform share of the window's nodal loads.
        return cpl.MethodWindowBinding(views[0], model_state)

    participant = cpl.FixedStepCouplingParticipant(
        method,
        bind,
        lambda native, args: (native[method.interface_nodes],),
        subsystem_id="solid-fe",
        substeps=substeps,
        input_ports=(
            cpl.CouplingPort(
                "solid-fe/interface-loads",
                "input",
                SOLID_LOADS.vector_space,
                field_space=SOLID_LOADS,
                quantity=HEAT,
                measurement=solid_loads_measurement(),
                temporal_kind="interval_integral",
                reference_scale=1.0,
            ),
        ),
        output_ports=(
            cpl.CouplingPort(
                "solid-fe/interface-temperature",
                "output",
                SOLID_TRACE.vector_space,
                field_space=SOLID_TRACE,
                waveform_plan=cpl.CouplingWaveformPlan(
                    substeps + 1,
                    1,
                    tuple(np.linspace(0.0, 1.0, substeps + 1)),
                    plan_id=f"solid-steps-{substeps}",
                ),
                reference_scale=1.0,
            ),
        ),
    )
    initial = 1.0 + 0.5 * np.cos(np.pi * vertices[:, 1]) * vertices[:, 0]
    return participant, jnp.asarray(initial)


# Fluid owner: SSPRK(3,3) with a face-heat accumulator in its native state.


def fluid_rate(grid: FluidGrid, /) -> Callable[[Array, Array, Any], Array]:
    """Cell temperatures and accumulated face heat of the two-point FV scheme."""

    def rate(time: Array, state: Array, args: Any) -> Array:
        start, end, interface_start, interface_end, conductance = args
        temperature = state[: grid.cell_count]
        fraction = (time - start) / (end - start)
        interface = interface_start + fraction * (interface_end - interface_start)
        flux = (
            conductance
            * grid.unit_conductances
            * (temperature[grid.owners] - temperature[grid.neighbors])
        )
        # Heat leaving each interface face toward the solid.
        face_heat = (
            conductance
            * grid.interface_unit_conductance
            * (temperature[grid.interface_cells] - interface)
        )
        balance = (
            jnp.zeros_like(temperature)
            .at[grid.owners]
            .add(-flux)
            .at[grid.neighbors]
            .add(flux)
            .at[grid.interface_cells]
            .add(-face_heat)
        )
        return jnp.concatenate(
            (balance / (FLUID_HEAT_CAPACITY * grid.cell_area), face_heat)
        )

    return rate


def fluid_participant(
    substeps: int, /, *, randomness: phx.solver.coupling.MethodParticipantRandomness
) -> tuple[phx.solver.coupling.FixedStepCouplingParticipant, Array]:
    grid = fluid_grid()

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
            # A carried-key conductance model: one lognormal draw per step.
            conductance = conductance * jnp.exp(
                0.05 * jax.random.normal(key, (), dtype=jnp.float64)
            )
        interface_start, interface_end = views[0]
        return cpl.MethodWindowBinding(
            (window.start, window.end, interface_start, interface_end, conductance),
            (model_state[0], applied_steps + 1),
        )

    participant = cpl.FixedStepCouplingParticipant(
        phx.solver.SSPRK33FixedStepMethod(fluid_rate(grid)),
        bind,
        lambda native, args: (),
        subsystem_id="fluid-fv",
        substeps=substeps,
        input_ports=(
            cpl.CouplingPort(
                "fluid-fv/interface-temperature",
                "input",
                FLUID_TEMPERATURE.vector_space,
                field_space=FLUID_TEMPERATURE,
                waveform_plan=cpl.CouplingWaveformPlan(
                    substeps + 1,
                    1,
                    tuple(np.linspace(0.0, 1.0, substeps + 1)),
                    plan_id=f"fluid-steps-{substeps}",
                ),
                reference_scale=1.0,
            ),
        ),
        output_ports=(
            cpl.CouplingPort(
                "fluid-fv/face-heat",
                "output",
                FLUID_FACE_HEAT.vector_space,
                field_space=FLUID_FACE_HEAT,
                quantity=HEAT,
                measurement=cpl.CouplingMeasurement.extensive(
                    FLUID_FACE_HEAT.vector_space,
                    FLUID_FACE_HEAT.support_id,
                    provenance_id="fluid-two-point-face-heat",
                ),
                temporal_kind="interval_integral",
                reference_scale=1.0,
            ),
        ),
        # The owner already spent the window's face heat in its accumulator.
        amounts=lambda start, end, args: (
            end[grid.cell_count :] - start[grid.cell_count :],
        ),
        randomness=randomness,
        key_impl=KEY_IMPL if randomness == "carried-key" else None,
    )
    rows = np.arange(FLUID_CELLS[1])
    cells = np.tile(0.25 * rows / FLUID_CELLS[1], FLUID_CELLS[0])
    native = np.concatenate((cells, np.zeros(FLUID_CELLS[1])))
    return participant, jnp.asarray(native)


# Coupled declaration and the independent semi-discrete reference.


def declaration(
    window_size: float,
    solid_substeps: int,
    fluid_substeps: int,
    /,
    *,
    randomness: phx.solver.coupling.MethodParticipantRandomness = "none",
    maximum_steps: int = 60,
) -> tuple[phx.solver.coupling.PartitionedCouplingDeclaration, dict[str, Any]]:
    solid, solid_initial = solid_participant(window_size, solid_substeps)
    fluid, fluid_initial = fluid_participant(fluid_substeps, randomness=randomness)
    averages = hat_face_averages()
    exchanges = (
        cpl.CouplingExchange(
            "interface-temperature",
            "solid-fe/interface-temperature",
            "fluid-fv/interface-temperature",
            transfer=matrix_transfer(
                SOLID_TRACE,
                FLUID_TEMPERATURE,
                averages,
                phx.discretization.TransferProperties(
                    constant_preserving=True, exact_on=("constants",)
                ),
                "solid-trace-face-average",
            ),
            requirement=cpl.CouplingTransferRequirement(constant_preserving=True),
            temporal=cpl.CouplingTemporalConversion(
                "interpolate", transfer=cpl.BarycentricCouplingTemporalTransfer(1)
            ),
        ),
        cpl.CouplingExchange(
            "interface-heat",
            "fluid-fv/face-heat",
            "solid-fe/interface-loads",
            transfer=matrix_transfer(
                FLUID_FACE_HEAT,
                SOLID_LOADS,
                averages.T,
                phx.discretization.TransferProperties(conservative=True),
                "fluid-face-heat-basis-loads",
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
            cpl.CouplingTolerance("fluid-fv/interface-temperature", absolute=1e-10),
            cpl.CouplingTolerance("solid-fe/interface-loads", absolute=1e-10),
        ),
        # The solid consumes the heat the fluid spent in the same sweep, so the
        # exchanged energy balances exactly; the iteration closes the temperature.
        fixed_point_sweep=cpl.CouplingSweep(
            "gauss-seidel", subsystem_order=("fluid-fv", "solid-fe")
        ),
    )
    key = jax.random.key(7, impl=KEY_IMPL) if randomness == "carried-key" else None
    states = {
        "solid-fe": solid.initial_state(solid_initial),
        "fluid-fv": fluid.initial_state(
            fluid_initial,
            model_state=(
                jnp.asarray(FLUID_CONDUCTANCE, dtype=jnp.float64),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            key=key,
        ),
    }
    return cpl.PartitionedCouplingDeclaration((solid, fluid), exchanges, policy), states


def p1_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Independent host assembly of the P1 mass and stiffness matrices."""
    vertices, triangles, _ = solid_mesh()
    count = vertices.shape[0]
    mass = np.zeros((count, count))
    stiffness = np.zeros((count, count))
    local_mass = (np.ones((3, 3)) + np.eye(3)) / 12.0
    reference_gradients = np.asarray(((-1.0, 1.0, 0.0), (-1.0, 0.0, 1.0)))
    for triangle in triangles:
        points = vertices[triangle]
        jacobian = np.stack((points[1] - points[0], points[2] - points[0]), axis=1)
        area = 0.5 * abs(np.linalg.det(jacobian))
        gradients = np.linalg.solve(jacobian.T, reference_gradients)
        mass[np.ix_(triangle, triangle)] += area * local_mass
        stiffness[np.ix_(triangle, triangle)] += area * gradients.T @ gradients
    return mass, stiffness


def reference_solution(initial: np.ndarray, time: float, /) -> np.ndarray:
    """exp(t A) of the coupled semi-discrete system [solid nodes; fluid cells]."""
    _, _, interface_nodes = solid_mesh()
    mass, stiffness = p1_matrices()
    grid = fluid_grid()
    solid_count, fluid_count = mass.shape[0], grid.cell_count
    solid_trace = np.zeros((interface_nodes.size, solid_count))
    solid_trace[np.arange(interface_nodes.size), interface_nodes] = 1.0
    fluid_trace = np.zeros((FLUID_CELLS[1], fluid_count))
    fluid_trace[np.arange(FLUID_CELLS[1]), np.asarray(grid.interface_cells)] = 1.0
    laplacian = np.zeros((fluid_count, fluid_count))
    owners, neighbors = np.asarray(grid.owners), np.asarray(grid.neighbors)
    conductances = np.asarray(grid.unit_conductances)
    np.add.at(laplacian, (owners, owners), conductances)
    np.add.at(laplacian, (neighbors, neighbors), conductances)
    np.add.at(laplacian, (owners, neighbors), -conductances)
    np.add.at(laplacian, (neighbors, owners), -conductances)
    face_average = hat_face_averages() @ solid_trace
    face = FLUID_CONDUCTANCE * grid.interface_unit_conductance
    # Face heat q = g (E T_f - I S T_s) leaves the fluid and loads the solid by Iᵀq.
    generator = np.block(
        [
            [
                -SOLID_CONDUCTIVITY * stiffness - face * face_average.T @ face_average,
                face * face_average.T @ fluid_trace,
            ],
            [
                face * fluid_trace.T @ face_average,
                -FLUID_CONDUCTANCE * laplacian - face * fluid_trace.T @ fluid_trace,
            ],
        ]
    )
    capacity = scipy.linalg.block_diag(
        SOLID_HEAT_CAPACITY * mass,
        FLUID_HEAT_CAPACITY * grid.cell_area * np.eye(fluid_count),
    )
    rates = np.zeros((solid_count + fluid_count, solid_count + fluid_count))
    rates[...] = np.linalg.solve(capacity, generator)
    return scipy.linalg.expm(time * rates) @ initial


def participant(state: phx.solver.coupling.CouplingState, subsystem_id: str, /) -> Any:
    return state.participant_states[state.subsystem_ids.index(subsystem_id)]


def temperatures(solid: Any, fluid: Any, /) -> np.ndarray:
    """Solid nodal temperatures followed by fluid cell temperatures."""
    cells = FLUID_CELLS[0] * FLUID_CELLS[1]
    return np.concatenate((np.asarray(solid.native), np.asarray(fluid.native)[:cells]))


def energies(solid: Any, fluid: Any, /) -> tuple[float, float]:
    """Heat content of the solid (∫ ρc T_h) and of the fluid cells."""
    mass, _ = p1_matrices()
    grid = fluid_grid()
    values = temperatures(solid, fluid)
    return (
        float(SOLID_HEAT_CAPACITY * np.sum(mass @ values[: mass.shape[0]])),
        float(FLUID_HEAT_CAPACITY * grid.cell_area * np.sum(values[mass.shape[0] :])),
    )


def solve(
    window_size: float, solid_substeps: int, fluid_substeps: int, /
) -> tuple[phx.solver.coupling.CouplingSolution, dict[str, Any]]:
    coupled, states = declaration(window_size, solid_substeps, fluid_substeps)
    problem = cpl.lower_partitioned_coupling(
        coupled, states, t0=0.0, t1=FINAL_TIME, window_size=window_size
    )
    return cpl.solve_coupling(problem), states


def convergence() -> None:
    print("convergence to the coupled semi-discrete reference at t =", FINAL_TIME)
    print(f"  {'local integrators':36s} {'window':>8s} {'max error':>11s} {'rate':>6s}")
    for label, solid_substeps, fluid_substeps in INTEGRATORS:
        errors = []
        for window_size in WINDOW_SIZES:
            solution, states = solve(window_size, solid_substeps, fluid_substeps)
            if not bool(solution.successful):
                raise RuntimeError(f"{label} failed a coupling window.")
            final = solution.final_state
            reference = reference_solution(
                temperatures(states["solid-fe"], states["fluid-fv"]), FINAL_TIME
            )
            computed = temperatures(
                participant(final, "solid-fe"), participant(final, "fluid-fv")
            )
            errors.append(float(np.max(np.abs(computed - reference))))
            rate = "" if len(errors) == 1 else f"{np.log2(errors[-2] / errors[-1]):6.3f}"
            print(f"  {label:36s} {window_size:8.4f} {errors[-1]:11.3e} {rate:>6s}")
        rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        if np.any(np.abs(rates - 1.0) > 0.2):
            raise RuntimeError(f"{label} is not first order in the window size.")


def ledger() -> None:
    solution, states = solve(0.05, 2, 3)
    final = solution.final_state
    rows = np.asarray(final.cumulative_exchange_budget)
    solid_start, fluid_start = energies(states["solid-fe"], states["fluid-fv"])
    solid_end, fluid_end = energies(
        participant(final, "solid-fe"), participant(final, "fluid-fv")
    )
    total = solid_start + fluid_start
    energy_error = abs(solid_end + fluid_end - total) / abs(total)
    print("exchange ledger [debit, credit] over", FINAL_TIME, "s:")
    for row_id, (debit, credit) in zip(final.budget_row_ids, rows, strict=True):
        print(f"  {row_id:22s} {debit: .15e} {credit: .15e}")
    print(f"  solid heat change      {solid_end - solid_start: .15e}")
    print(f"  fluid heat change      {fluid_end - fluid_start: .15e}")
    print(f"  relative total-energy error {energy_error:.3e}")
    heat = rows[final.budget_row_ids.index("interface-heat")]
    if energy_error > 1e-13 or abs(heat[0] + heat[1]) > 1e-13 * abs(heat[1]):
        raise RuntimeError("Interface heat is not conserved to roundoff.")


def rollback() -> None:
    window = 0.05
    exhausted, states = declaration(
        window, 2, 3, randomness="carried-key", maximum_steps=2
    )
    prepared = cpl.lower_partitioned_coupling(
        exhausted, states, t0=0.0, t1=FINAL_TIME, window_size=window
    ).prepare()
    checkpoint = prepared.reference_state
    rejected = cpl.advance_coupling_window(prepared, checkpoint, window)
    replay = cpl.advance_coupling_window(prepared, checkpoint, window)
    converging, _ = declaration(window, 2, 3, randomness="carried-key")
    accepting = cpl.lower_partitioned_coupling(
        converging, states, t0=0.0, t1=FINAL_TIME, window_size=window
    ).prepare()
    accepted = cpl.advance_coupling_window(accepting, accepting.reference_state, window)
    fluid_checkpoint = participant(checkpoint, "fluid-fv")
    candidate = participant(rejected.candidate_state, "fluid-fv")
    committed = participant(accepted.accepted_state, "fluid-fv")
    diagnostics = accepted.diagnostics
    evidence = {
        "rejected status": cpl.coupling_status_message(int(rejected.status)),
        "checkpoint retained": bool(eqx.tree_equal(rejected.accepted_state, checkpoint)),
        "candidate model steps": int(candidate.model_state[1]),
        "candidate key advanced": not np.array_equal(
            candidate.key_data, fluid_checkpoint.key_data
        ),
        "replay bitwise identical": bool(eqx.tree_equal(replay, rejected)),
        "accepted status": cpl.coupling_status_message(int(accepted.status)),
        "accepted windows": int(committed.accepted_windows),
        "accepted model steps": int(committed.model_state[1]),
        "participant evaluations": dict(
            zip(
                checkpoint.subsystem_ids,
                np.asarray(diagnostics.participant_evaluations).tolist(),
                strict=True,
            )
        ),
        "participant work": dict(
            zip(
                checkpoint.subsystem_ids,
                np.asarray(diagnostics.participant_work).tolist(),
                strict=True,
            )
        ),
        "work counts complete": diagnostics.counts_complete,
    }
    print("rollback and replay of a rejected implicit window:")
    for name, value in evidence.items():
        print(f"  {name:26s} {value}")
    if (
        int(rejected.status) != int(cpl.CouplingStatus.WORK_EXHAUSTED)
        or not evidence["checkpoint retained"]
        or not evidence["candidate key advanced"]
        or not evidence["replay bitwise identical"]
        or not bool(accepted.successful)
        or evidence["accepted windows"] != 1
        or evidence["accepted model steps"] != 3
        or not diagnostics.counts_complete
    ):
        raise RuntimeError("Transactional rollback evidence failed.")


if __name__ == "__main__":
    convergence()
    ledger()
    rollback()
