#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One-sided mesh adaptation of a nonmatching FE/FV conjugate-heat coupling.

The coupled problem is the one of `mixed_method_time_coupling.py`: a P1
finite-element solid on [0, 1]² (backward Euler, one prepared factorization) and
a cell-centered finite-volume fluid on [1, 2] × [0, 1] (SSPRK(3,3) with a
face-heat accumulator) meet on the nonmatching interface x = 1. The interface
temperature reaches the fluid through exact face averages I; the fluid's
whole-window face heat reaches the solid nodal loads through P = Iᵀ.

At the accepted window boundary t = 0.2 one side is refined and the run continues
on the new composition:

- refining the solid: the P1 temperature crosses through the native FE vertex
  interpolation transfer (certified conservative against the P1 basis
  integrals) bound as a topology-epoch transition; the factorization and the
  solid probe query are reprepared; the fluid is retained bitwise;
- refining the fluid: cell temperatures cross through the native first-order
  common-refinement remap; the face-heat accumulators split to their child faces
  through the interface-face lineage (lengths); the fluid probe is reprepared;
  the solid is retained bitwise.

Both variants reprepare the interface common refinement and the prepared
coupling graph, retain the consumed exchange budgets, and publish one accepted
composition through `phx.lifecycle.commit_composition_rebind`. The independent
reference is the matrix exponential of the assembled coupled semi-discrete
system on each topology (host SciPy, hand-assembled P1 matrices), with the exact
nested prolongation at t = 0.2.
"""

from collections.abc import Callable
from typing import Any, assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax import Array

import phydrax as phx
from phydrax.discretization import SimplicialLocationPolicy
from phydrax.discretization.fem import (
    prepare_finite_element_field_reconstruction,
    vertex_interpolation_transfer,
)


jax.config.update("jax_enable_x64", True)

cpl = phx.solver.coupling
lc = phx.lifecycle
D = phx.discretization
Side: TypeAlias = Literal["solid", "fluid"]

SOLID_DIVISIONS = 4
FLUID_CELLS = (3, 3)
SOLID_HEAT_CAPACITY = 2.0
SOLID_CONDUCTIVITY = 1.0
FLUID_HEAT_CAPACITY = 1.0
FLUID_CONDUCTANCE = 0.25
FINAL_TIME = 0.4
REBIND_TIME = 0.2
WINDOW_SIZES = (0.1, 0.05, 0.025)
SOLID_SUBSTEPS = 2
FLUID_SUBSTEPS = 3
KEY_IMPL = "threefry2x32"
SOLID_PROBE = (0.35, 0.6)
FLUID_PROBE = (1.5, 0.45)
HEAT = cpl.CouplingQuantity("heat", phx.units.JOULE, sign_convention="into-solid")


# Solid owner: P1 triangles, the FE discretization, and its topology epoch.


class SolidOwner(NamedTuple):
    vertices: np.ndarray
    triangles: np.ndarray
    interface: np.ndarray
    discretization: Any
    epoch: D.TopologyEpoch
    measures: np.ndarray


def square_mesh(divisions: int, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-triangle mesh of [0, 1]² and its interface vertices on x = 1."""
    coordinates = np.linspace(0.0, 1.0, divisions + 1)
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    vertices = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    index = np.arange(vertices.shape[0]).reshape(x.shape)
    lower = np.stack((index[:-1, :-1], index[1:, :-1], index[1:, 1:]), axis=-1)
    upper = np.stack((index[:-1, :-1], index[1:, 1:], index[:-1, 1:]), axis=-1)
    triangles = np.concatenate((lower.reshape(-1, 3), upper.reshape(-1, 3)))
    return vertices, triangles.astype(np.int32), index[-1, :]


def solid_owner(
    vertices: np.ndarray, triangles: np.ndarray, interface: np.ndarray, index: int, /
) -> SolidOwner:
    mesh = D.CellMesh.from_triangles(jnp.asarray(vertices), jnp.asarray(triangles))
    temperature = D.FiniteElementFieldSpec("T", D.lagrange_element("triangle", 1))
    discretization = D.FiniteElementPlan(mesh, temperature).prepare()
    ones = jnp.ones((vertices.shape[0],), dtype=jnp.float64)
    return SolidOwner(
        vertices,
        triangles,
        interface,
        discretization,
        D.TopologyEpoch(index, mesh.geometry_id, mesh.topology_id, "solid-serial"),
        # Integrals of the P1 basis functions: the heat-content measure.
        np.asarray(discretization.mass(ones)),
    )


def uniform_solid(divisions: int, index: int, /) -> SolidOwner:
    return solid_owner(*square_mesh(divisions), index)


def barycentric_stencil(
    vertices: np.ndarray, triangles: np.ndarray, points: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Containing source triangle and barycentric weights of every target point."""
    corners = vertices[triangles]
    edges = np.stack((corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), -1)
    offsets = points[:, None, :] - corners[None, :, 0, :]
    local = np.linalg.solve(edges[None], offsets[..., None])[..., 0]
    weights = np.concatenate((1.0 - local.sum(-1, keepdims=True), local), axis=-1)
    inside = np.all(weights >= -1e-12, axis=-1)
    if not np.all(np.any(inside, axis=1)):
        raise ValueError("Target vertices leave the source triangulation.")
    cell = np.argmax(inside, axis=1)
    rows = np.arange(points.shape[0])
    return triangles[cell], np.clip(weights[rows, cell], 0.0, 1.0)


def refine_solid_transfer(coarse: SolidOwner, fine: SolidOwner, /) -> Any:
    """Native FE vertex interpolation, certified linear and conservative."""
    rows, weights = barycentric_stencil(coarse.vertices, coarse.triangles, fine.vertices)
    return vertex_interpolation_transfer(
        rows.astype(np.int32),
        weights,
        np.ones(rows.shape, dtype=np.bool_),
        source_size=coarse.vertices.shape[0],
        source_topology_id=coarse.epoch.topology_id,
        target_topology_id=fine.epoch.topology_id,
        preserves_linear=True,
        conservative=True,
        source_coordinates=coarse.vertices,
        target_coordinates=fine.vertices,
        source_measures=coarse.measures,
        target_measures=fine.measures,
    )


def solid_probe(solid: SolidOwner, /) -> Any:
    """Native prepared point query of the solid temperature."""
    reconstruction = prepare_finite_element_field_reconstruction(
        solid.discretization, "T", location_policy=SimplicialLocationPolicy(64, 16, 1)
    )
    return reconstruction.prepare_query(np.asarray((SOLID_PROBE,)))


# Fluid owner: the two-point FV cell grid, its FV geometry, and topology epoch.


class FluidGrid(NamedTuple):
    owners: Array
    neighbors: Array
    unit_conductances: Array
    interface_cells: Array
    interface_unit_conductance: float
    cell_area: float
    cell_count: int


class FluidOwner(NamedTuple):
    cells: tuple[int, int]
    grid: FluidGrid
    discretization: Any
    epoch: D.TopologyEpoch
    face_edges: np.ndarray


def fluid_owner(nx: int, ny: int, index: int, /) -> FluidOwner:
    hx, hy = 1.0 / nx, 1.0 / ny
    cell = np.arange(nx * ny).reshape(nx, ny)
    owners = np.concatenate((cell[:-1, :].reshape(-1), cell[:, :-1].reshape(-1)))
    neighbors = np.concatenate((cell[1:, :].reshape(-1), cell[:, 1:].reshape(-1)))
    conductances = np.concatenate(
        (np.full((nx - 1) * ny, hy / hx), np.full(nx * (ny - 1), hx / hy))
    )
    grid = FluidGrid(
        jnp.asarray(owners, dtype=jnp.int32),
        jnp.asarray(neighbors, dtype=jnp.int32),
        jnp.asarray(conductances, dtype=jnp.float64),
        jnp.asarray(cell[0, :], dtype=jnp.int32),
        # Interface faces see the cell center at half a cell width.
        hy / (0.5 * hx),
        hx * hy,
        nx * ny,
    )
    x = np.linspace(1.0, 2.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    stride = ny + 1
    quadrilaterals = np.asarray(
        [
            (
                i * stride + j,
                (i + 1) * stride + j,
                (i + 1) * stride + j + 1,
                i * stride + j + 1,
            )
            for i in range(nx)
            for j in range(ny)
        ],
        dtype=np.int32,
    )
    discretization = D.UnstructuredFiniteVolumePlan(
        np.stack((xx, yy), axis=-1).reshape((-1, 2)),
        quadrilaterals=quadrilaterals,
        field_name="T",
    ).prepare()
    epoch = D.TopologyEpoch(
        index,
        discretization.geometry_id,
        discretization.topology_id,
        "fluid-serial",
    )
    return FluidOwner((nx, ny), grid, discretization, epoch, y)


def nested_fluid_remap(coarse: FluidOwner, fine: FluidOwner, /) -> Any:
    """First-order remap on the exact common refinement of nested cell grids."""
    (nx, ny), (fx, fy) = coarse.cells, fine.cells
    rx, ry = fx // nx, fy // ny
    target = np.arange(fx * fy).reshape(fx, fy)
    parents = (target // fy // rx) * ny + (target % fy) // ry
    return D.UnstructuredConservativeRemapPlan(
        coarse.discretization,
        fine.discretization,
        np.arange(fx * fy + 1),
        parents.reshape(-1),
        np.asarray(fine.discretization.cell_volumes),
        method="first-order",
        provenance="nested-fluid-refinement",
    )


def interface_face_lineage(coarse: FluidOwner, fine: FluidOwner, /) -> Any:
    """Interface faces of the fine grid refined from their coarse parent face."""
    ratio = fine.cells[1] // coarse.cells[1]
    children = np.arange(fine.cells[1], dtype=np.int64)
    return phx.meshing.EntityLineage(
        1,
        f"fluid-interface-faces/{coarse.epoch.epoch_id}",
        f"fluid-interface-faces/{fine.epoch.epoch_id}",
        children // ratio,
        children,
        np.full(children.shape, int(phx.meshing.EntityLineageKind.REFINED_FROM)),
    )


def fluid_probe(fluid: FluidOwner, /) -> Array:
    """The cell whose average the fluid probe observes."""
    nx, ny = fluid.cells
    i = int((FLUID_PROBE[0] - 1.0) * nx)
    j = int(FLUID_PROBE[1] * ny)
    return jnp.asarray((i * ny + j,), dtype=jnp.int32)


# Interface owner: exact common refinement of the two interface partitions.


class Interface(NamedTuple):
    averages: np.ndarray
    temperature: D.FieldTransfer
    heat: D.FieldTransfer
    route_id: str


def face_averages(nodes: np.ndarray, edges: np.ndarray, /) -> np.ndarray:
    """Exact face averages of every P1 interface hat on the common refinement."""
    breaks = np.union1d(nodes, edges)
    matrix = np.zeros((edges.size - 1, nodes.size))
    identity = np.eye(nodes.size)
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        face = np.searchsorted(edges, 0.5 * (left + right)) - 1
        for node in range(nodes.size):
            values = np.interp((left, right), nodes, identity[node])
            matrix[face, node] += 0.5 * (right - left) * (values[0] + values[1])
    return matrix / np.diff(edges)[:, None]


def field(
    name: str, representation: D.FieldRepresentation, count: int, /
) -> D.DiscreteFieldSpace:
    layout = D.EntityDofLayout(f"{name}/entities", count, count)
    space = phx.linalg.ArraySpace((count,), dtype=jnp.float64, space_id=f"{name}/dofs")
    return D.DiscreteFieldSpace(
        name, f"{name}/support", layout, space, representation=representation
    )


def solid_fields(solid: SolidOwner, /) -> tuple[D.DiscreteFieldSpace, ...]:
    count = solid.interface.size
    tag = f"e{solid.epoch.index}"
    return (
        field(f"solid-interface-trace/{tag}", "basis_coefficient", count),
        field(f"solid-interface-loads/{tag}", "functional", count),
    )


def fluid_fields(fluid: FluidOwner, /) -> tuple[D.DiscreteFieldSpace, ...]:
    count = fluid.cells[1]
    tag = f"e{fluid.epoch.index}"
    return (
        field(f"fluid-interface-temperature/{tag}", "cell_average", count),
        field(f"fluid-interface-face-heat/{tag}", "flux_moment", count),
    )


def matrix_transfer(
    source: D.DiscreteFieldSpace,
    target: D.DiscreteFieldSpace,
    matrix: np.ndarray,
    properties: D.TransferProperties,
    name: str,
    /,
) -> D.FieldTransfer:
    values = jnp.asarray(matrix)
    return D.FieldTransfer(
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


def interface(solid: SolidOwner, fluid: FluidOwner, /) -> Interface:
    trace, loads = solid_fields(solid)
    temperature, face_heat = fluid_fields(fluid)
    averages = face_averages(solid.vertices[solid.interface, 1], fluid.face_edges)
    route = f"{solid.epoch.epoch_id}|{fluid.epoch.epoch_id}"
    return Interface(
        averages,
        matrix_transfer(
            trace,
            temperature,
            averages,
            D.TransferProperties(constant_preserving=True, exact_on=("constants",)),
            f"solid-trace-face-average/{route}",
        ),
        matrix_transfer(
            face_heat,
            loads,
            averages.T,
            D.TransferProperties(conservative=True),
            f"fluid-face-heat-basis-loads/{route}",
        ),
        route,
    )


# Native participants: backward Euler FE solid and SSPRK(3,3) FV fluid.


class SolidBackwardEuler(phx.solver.AbstractFixedStepMethod):
    """(ρc M + Δt k K) Tⁿ⁺¹ = ρc M Tⁿ + Sᵀ ℓ with interface nodal loads ℓ."""

    mass: Any
    factorization: phx.linalg.PreparedLinearSolve
    interface_nodes: Array
    prepared_step: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass: Any,
        factorization: phx.linalg.PreparedLinearSolve,
        interface_nodes: Array,
        step: float,
        epoch_id: str,
        /,
    ) -> None:
        self.mass = mass
        self.factorization = factorization
        self.interface_nodes = interface_nodes
        self.prepared_step = step
        self.method_id = f"solid-backward-euler:{epoch_id}:{step!r}"

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
        right_hand_side = SOLID_HEAT_CAPACITY * self.mass(state)
        right_hand_side = right_hand_side.at[self.interface_nodes].add(args)
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


def solid_factorization(solid: SolidOwner, step: float, /) -> Any:
    """One prepared LU of ρc M + Δt k K: the solid's symbolic/numeric preconditioner."""
    form = phx.equations.FiniteElementForm(
        "solid-backward-euler",
        "T",
        (
            phx.equations.MassAction("T", SOLID_HEAT_CAPACITY),
            phx.equations.DiffusionAction("T", step * SOLID_CONDUCTIVITY),
        ),
    )
    system, _ = phx.equations.compile_finite_element_problem(
        form, solid.discretization
    ).linear_system()
    return phx.linalg.prepare(
        system,
        phx.linalg.LinearSolvePolicy(
            phx.linalg.DenseLU(),
            materialization=phx.linalg.MaterializationPolicy(max_entries=16384),
        ),
    )


def waveform_plan(substeps: int, name: str, /) -> Any:
    nodes = tuple(np.linspace(0.0, 1.0, substeps + 1))
    return cpl.CouplingWaveformPlan(substeps + 1, 1, nodes, plan_id=f"{name}-{substeps}")


def solid_participant(
    solid: SolidOwner, factorization: Any, window_size: float, /
) -> cpl.FixedStepCouplingParticipant:
    trace, loads = solid_fields(solid)
    method = SolidBackwardEuler(
        # The P1 mass is assembled once on the host and reused by every step.
        solid.discretization.mass,
        factorization,
        jnp.asarray(solid.interface, dtype=jnp.int32),
        window_size / SOLID_SUBSTEPS,
        solid.epoch.epoch_id,
    )
    count = solid.interface.size
    measurement = cpl.CouplingMeasurement(
        phx.linalg.FunctionLinearOperator(
            lambda values: jnp.sum(values, keepdims=True),
            source=loads.vector_space,
            target=phx.linalg.ArraySpace((1,), dtype=jnp.float64),
            transpose_action=lambda amount: jnp.broadcast_to(amount, (count,)),
            operator_id=f"solid-interface-load-total/e{solid.epoch.index}",
        ),
        phx.units.ONE,
        representation="functional",
        support_id=loads.support_id,
        provenance_id="solid-p1-basis-loads",
        normalization="counting",
    )

    def bind(
        window: Any, views: tuple[Any, ...], model_state: None, key: None, args: None
    ) -> cpl.MethodWindowBinding:
        del window, key, args
        return cpl.MethodWindowBinding(views[0], model_state)

    return cpl.FixedStepCouplingParticipant(
        method,
        bind,
        lambda native, args: (native[method.interface_nodes],),
        subsystem_id="solid-fe",
        substeps=SOLID_SUBSTEPS,
        input_ports=(
            cpl.CouplingPort(
                "solid-fe/interface-loads",
                "input",
                loads.vector_space,
                field_space=loads,
                quantity=HEAT,
                measurement=measurement,
                temporal_kind="interval_integral",
                reference_scale=1.0,
            ),
        ),
        output_ports=(
            cpl.CouplingPort(
                "solid-fe/interface-temperature",
                "output",
                trace.vector_space,
                field_space=trace,
                waveform_plan=waveform_plan(SOLID_SUBSTEPS, "solid-steps"),
                reference_scale=1.0,
            ),
        ),
        discretization_bundle_id=solid.epoch.epoch_id,
    )


def fluid_rate(grid: FluidGrid, /) -> Callable[[Array, Array, Any], Array]:
    """Cell temperatures and accumulated face heat of the two-point FV scheme."""

    def rate(time: Array, state: Array, args: Any) -> Array:
        start, end, interface_start, interface_end, conductance = args
        temperature = state[: grid.cell_count]
        fraction = (time - start) / (end - start)
        interface_value = interface_start + fraction * (interface_end - interface_start)
        flux = (
            conductance
            * grid.unit_conductances
            * (temperature[grid.owners] - temperature[grid.neighbors])
        )
        # Heat leaving each interface face toward the solid.
        face_heat = (
            conductance
            * grid.interface_unit_conductance
            * (temperature[grid.interface_cells] - interface_value)
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
    fluid: FluidOwner, randomness: cpl.MethodParticipantRandomness, /
) -> cpl.FixedStepCouplingParticipant:
    temperature, face_heat = fluid_fields(fluid)
    grid = fluid.grid

    def bind(
        window: Any,
        views: tuple[Any, ...],
        model_state: tuple[Array, Array],
        key: Array | None,
        args: None,
    ) -> cpl.MethodWindowBinding:
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

    return cpl.FixedStepCouplingParticipant(
        phx.solver.SSPRK33FixedStepMethod(fluid_rate(grid)),
        bind,
        lambda native, args: (),
        subsystem_id="fluid-fv",
        substeps=FLUID_SUBSTEPS,
        input_ports=(
            cpl.CouplingPort(
                "fluid-fv/interface-temperature",
                "input",
                temperature.vector_space,
                field_space=temperature,
                waveform_plan=waveform_plan(FLUID_SUBSTEPS, "fluid-steps"),
                reference_scale=1.0,
            ),
        ),
        output_ports=(
            cpl.CouplingPort(
                "fluid-fv/face-heat",
                "output",
                face_heat.vector_space,
                field_space=face_heat,
                quantity=HEAT,
                measurement=cpl.CouplingMeasurement.extensive(
                    face_heat.vector_space,
                    face_heat.support_id,
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
        discretization_bundle_id=fluid.epoch.epoch_id,
    )


def solid_initial(solid: SolidOwner, /) -> Array:
    x, y = solid.vertices[:, 0], solid.vertices[:, 1]
    return jnp.asarray(1.0 + 0.5 * np.cos(np.pi * y) * x)


def fluid_initial(fluid: FluidOwner, /) -> Array:
    nx, ny = fluid.cells
    cells = np.tile(0.25 * np.arange(ny) / ny, nx)
    return jnp.asarray(np.concatenate((cells, np.zeros(ny))))


# The coupled composition: owners, participants, one prepared coupling epoch.


class CoupledModel(NamedTuple):
    solid: SolidOwner
    fluid: FluidOwner
    interface: Interface
    factorization: Any
    participants: tuple[Any, Any]
    solid_probe: Any
    fluid_probe: Array
    epoch: cpl.PreparedCouplingEpoch
    window_size: float
    randomness: cpl.MethodParticipantRandomness
    fluid_model_independent: bool


def coupling_policy() -> cpl.ImplicitCouplingPolicy:
    return cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-11,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=1e-14,
            maximum_steps=60,
        ),
        (
            cpl.CouplingTolerance("fluid-fv/interface-temperature", absolute=1e-10),
            cpl.CouplingTolerance("solid-fe/interface-loads", absolute=1e-10),
        ),
        fixed_point_sweep=cpl.CouplingSweep(
            "gauss-seidel", subsystem_order=("fluid-fv", "solid-fe")
        ),
    )


def prepare_epoch(
    solid: SolidOwner,
    fluid: FluidOwner,
    route: Interface,
    participants: tuple[Any, Any],
    states: dict[str, Any],
    start: float,
    window_size: float,
    /,
) -> cpl.PreparedCouplingEpoch:
    """Lower and prepare the coupled graph from accepted participant checkpoints."""
    exchanges = (
        cpl.CouplingExchange(
            "interface-temperature",
            "solid-fe/interface-temperature",
            "fluid-fv/interface-temperature",
            transfer=route.temperature,
            requirement=cpl.CouplingTransferRequirement(constant_preserving=True),
            temporal=cpl.CouplingTemporalConversion(
                "interpolate", transfer=cpl.BarycentricCouplingTemporalTransfer(1)
            ),
        ),
        cpl.CouplingExchange(
            "interface-heat",
            "fluid-fv/face-heat",
            "solid-fe/interface-loads",
            transfer=route.heat,
            requirement=cpl.CouplingTransferRequirement(conservative=True),
            temporal=cpl.CouplingTemporalConversion("window-integral"),
        ),
    )
    declaration = cpl.PartitionedCouplingDeclaration(
        participants, exchanges, coupling_policy()
    )
    problem = cpl.lower_partitioned_coupling(
        declaration, states, t0=start, t1=FINAL_TIME, window_size=window_size
    )
    return cpl.PreparedCouplingEpoch(
        problem.prepare(),
        (solid.epoch.epoch_id, fluid.epoch.epoch_id),
        (f"solid-steps-{SOLID_SUBSTEPS}", f"fluid-steps-{FLUID_SUBSTEPS}"),
        participant_epoch_codes=(solid.epoch.index, fluid.epoch.index),
        waveform_required_samples=(SOLID_SUBSTEPS + 1, FLUID_SUBSTEPS + 1),
        topology_code=solid.epoch.index + fluid.epoch.index,
    )


def build(
    window_size: float,
    /,
    *,
    randomness: cpl.MethodParticipantRandomness = "none",
    fluid_model_independent: bool = True,
) -> tuple[CoupledModel, cpl.CouplingState]:
    """Initial composition: 4×4 P1 solid, 3×3 FV fluid, epoch 0 at t = 0."""
    solid = uniform_solid(SOLID_DIVISIONS, 0)
    fluid = fluid_owner(*FLUID_CELLS, 0)
    route = interface(solid, fluid)
    factorization = solid_factorization(solid, window_size / SOLID_SUBSTEPS)
    participants = (
        solid_participant(solid, factorization, window_size),
        fluid_participant(fluid, randomness),
    )
    key = jax.random.key(7, impl=KEY_IMPL) if randomness == "carried-key" else None
    states = {
        "solid-fe": participants[0].initial_state(solid_initial(solid)),
        "fluid-fv": participants[1].initial_state(
            fluid_initial(fluid),
            model_state=(
                jnp.asarray(FLUID_CONDUCTANCE, dtype=jnp.float64),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            key=key,
        ),
    }
    epoch = prepare_epoch(solid, fluid, route, participants, states, 0.0, window_size)
    model = CoupledModel(
        solid,
        fluid,
        route,
        factorization,
        participants,
        solid_probe(solid),
        fluid_probe(fluid),
        epoch,
        window_size,
        randomness,
        fluid_model_independent,
    )
    return model, epoch.prepared_coupling.reference_state


advance_window = eqx.filter_jit(cpl.advance_coupling_window)


def advance(
    model: CoupledModel, state: cpl.CouplingState, windows: int, /
) -> tuple[cpl.CouplingState, Any]:
    """Accept `windows` implicit coupling windows; each is one accepted boundary."""
    result = None
    for _ in range(windows):
        result = advance_window(model.epoch.prepared_coupling, state, model.window_size)
        if not bool(result.successful):
            raise RuntimeError("A coupling window was not accepted.")
        state = result.accepted_state
    return state, result


def participant(state: cpl.CouplingState, subsystem_id: str, /) -> Any:
    return state.participant_states[state.subsystem_ids.index(subsystem_id)]


def observe(model: CoupledModel, state: cpl.CouplingState, /) -> tuple[float, float]:
    """Solid probe (native FE point query) and fluid probe (cell average)."""
    solid = participant(state, "solid-fe").native
    fluid = participant(state, "fluid-fv").native
    return (
        float(np.asarray(model.solid_probe.apply(solid))[0]),
        float(np.asarray(fluid[model.fluid_probe])[0]),
    )


def energies(model: CoupledModel, state: cpl.CouplingState, /) -> tuple[float, float]:
    """Heat content of the solid (ρc ∫ T_h) and of the fluid cells."""
    solid = np.asarray(participant(state, "solid-fe").native)
    fluid = np.asarray(participant(state, "fluid-fv").native)[
        : model.fluid.grid.cell_count
    ]
    return (
        float(SOLID_HEAT_CAPACITY * model.solid.measures @ solid),
        float(FLUID_HEAT_CAPACITY * model.fluid.grid.cell_area * np.sum(fluid)),
    )


# The composition: owner artifacts and the canonical coupling entries.


def owner_entries(model: CoupledModel, /) -> dict[str, lc.CompositionEntry]:
    """Discretizations, preconditioner, observations, and the interface route."""
    solid, fluid = model.solid, model.fluid
    solid_discretization = lc.CompositionEntry(
        solid,
        entry_id="solid-fe/discretization",
        role="discretization",
        owner_id="solid-fe",
        structure_id=solid.epoch.epoch_id,
        revision_id=solid.epoch.epoch_id,
        semantics_id="solid-fe:p1-temperature",
    )
    fluid_discretization = lc.CompositionEntry(
        fluid,
        entry_id="fluid-fv/discretization",
        role="discretization",
        owner_id="fluid-fv",
        structure_id=fluid.epoch.epoch_id,
        revision_id=fluid.epoch.epoch_id,
        semantics_id="fluid-fv:cell-temperature",
    )
    on_solid = (solid_discretization.binding("structure"),)
    on_fluid = (fluid_discretization.binding("structure"),)
    step = model.window_size / SOLID_SUBSTEPS
    entries = (
        solid_discretization,
        fluid_discretization,
        lc.CompositionEntry(
            model.factorization,
            entry_id="solid-fe/factorization",
            role="preconditioner",
            owner_id="solid-fe",
            structure_id=f"{solid.epoch.epoch_id}:dense-lu",
            revision_id=f"{solid.epoch.epoch_id}:dense-lu:{step!r}",
            semantics_id="solid-fe:backward-euler-operator",
            dependencies=on_solid,
        ),
        lc.CompositionEntry(
            model.solid_probe,
            entry_id="solid-fe/probe",
            role="observation",
            owner_id="solid-fe",
            structure_id=model.solid_probe.query_id,
            revision_id=model.solid_probe.query_id,
            semantics_id="solid-fe:temperature-probe",
            dependencies=on_solid,
        ),
        lc.CompositionEntry(
            model.fluid_probe,
            entry_id="fluid-fv/probe",
            role="observation",
            owner_id="fluid-fv",
            structure_id=f"{fluid.epoch.epoch_id}:probe-cell",
            revision_id=f"{fluid.epoch.epoch_id}:probe-cell",
            semantics_id="fluid-fv:temperature-probe",
            dependencies=on_fluid,
        ),
        lc.CompositionEntry(
            model.interface,
            entry_id="interface/route",
            role="interface-route",
            owner_id="interface",
            structure_id=model.interface.route_id,
            revision_id=model.interface.route_id,
            semantics_id="interface:solid-fluid-x=1",
            dependencies=on_solid + on_fluid,
        ),
    )
    return {entry.entry_id: entry for entry in entries}


def coupling_entries(
    model: CoupledModel,
    state: cpl.CouplingState,
    owners: dict[str, lc.CompositionEntry],
    /,
) -> dict[str, lc.CompositionEntry]:
    entries = cpl.coupling_composition_entries(
        model.epoch,
        state,
        native_dependencies={
            "solid-fe": (owners["solid-fe/discretization"].binding("structure"),),
            "fluid-fv": (owners["fluid-fv/discretization"].binding("structure"),),
        },
        # The fluid's conductance and step counter do not depend on its grid.
        discretization_independent=("fluid-fv",) if model.fluid_model_independent else (),
        epoch_dependencies=(
            owners["interface/route"].binding("structure"),
            owners["solid-fe/factorization"].binding("structure"),
        ),
    )
    return {entry.entry_id: entry for entry in entries}


def compose(model: CoupledModel, state: cpl.CouplingState, /) -> lc.Composition:
    """The accepted composition at the boundary `state` sits on."""
    owners = owner_entries(model)
    window = int(np.asarray(state.window_index))
    return lc.Composition(
        (*owners.values(), *coupling_entries(model, state, owners).values()),
        boundary_id=f"accepted-window-{window}",
    )


def target_entries(model: CoupledModel, /) -> dict[str, lc.CompositionEntry]:
    """Entries of a freshly staged target model at its prepared reference state."""
    owners = owner_entries(model)
    reference = model.epoch.prepared_coupling.reference_state
    return {**owners, **coupling_entries(model, reference, owners)}


def stage_solid_refinement(
    model: CoupledModel, composition: lc.Composition, /, *, fine: SolidOwner | None = None
) -> tuple[CoupledModel, lc.CompositionRebind]:
    """Host staging of a refined solid; any owner refusal raises and publishes nothing."""
    _, state = cpl.coupling_state_from_composition(composition)
    coarse = model.solid
    fine = (
        uniform_solid(2 * SOLID_DIVISIONS, coarse.epoch.index + 1)
        if fine is None
        else fine
    )
    transition = refine_solid_transfer(coarse, fine).epoch_transition(
        coarse.discretization.field_spaces[0],
        fine.discretization.field_spaces[0],
        coarse.epoch,
        fine.epoch,
        coarse.measures,
        fine.measures,
        # Uniform refinement of the same affine square: an exact restriction.
        geometry=D.TransferGeometryBinding(
            coarse.epoch.geometry_id,
            fine.epoch.geometry_id,
            "exact-restriction",
            source_topology_id=coarse.epoch.topology_id,
            target_topology_id=fine.epoch.topology_id,
            coverage_defect=0.0,
        ),
    )
    checkpoint = participant(state, "solid-fe")
    factorization = solid_factorization(fine, model.window_size / SOLID_SUBSTEPS)
    participants = (
        solid_participant(fine, factorization, model.window_size),
        model.participants[1],
    )
    route = interface(fine, model.fluid)
    states = {
        "solid-fe": cpl.MethodParticipantState(
            transition.apply(checkpoint.native).values,
            checkpoint.model_state,
            checkpoint.key_data,
            checkpoint.accepted_windows,
            checkpoint.native_steps,
        ),
        "fluid-fv": participant(state, "fluid-fv"),
    }
    start = float(np.asarray(state.time))
    epoch = prepare_epoch(
        fine, model.fluid, route, participants, states, start, model.window_size
    )
    target_model = model._replace(
        solid=fine,
        interface=route,
        factorization=factorization,
        participants=participants,
        solid_probe=solid_probe(fine),
        epoch=epoch,
    )
    target = target_entries(target_model)
    retained = (
        *(item for item in composition.entry_ids if item.startswith("fluid-fv/")),
        "solid-fe/windows",
        "exchange/interface-temperature",
        "coupling/budget",
        "coupling/clock",
    )
    reprepared = (
        "solid-fe/discretization",
        "solid-fe/factorization",
        "solid-fe/probe",
        "interface/route",
        "coupling/epoch",
    )
    rebind = lc.CompositionRebind(
        composition,
        retain=retained,
        reprepare=tuple(target[item] for item in reprepared),
        transports=(
            transition.composition_transport(
                composition.entry("solid-fe/native"), target["solid-fe/native"]
            ),
            cpl.coupling_exchange_transport(
                composition, epoch, tuple(target.values()), "interface-heat"
            ),
        ),
    )
    return target_model, rebind


def fluid_native_image(
    coarse: FluidOwner, fine: FluidOwner, transition: Any, lineage: Any, native: Array, /
) -> tuple[Array, Any]:
    """Fluid owner rule: cell remap, face heat split to child faces by length."""
    count = coarse.grid.cell_count
    cells = transition.apply(native[:count])
    ratio = coarse.cells[1] / fine.cells[1]
    parents = jnp.asarray(np.asarray(lineage.source_global_ids), dtype=jnp.int32)
    children = jnp.asarray(np.asarray(lineage.target_global_ids), dtype=jnp.int32)
    faces = (
        jnp.zeros((fine.cells[1],), dtype=native.dtype)
        .at[children]
        .set(native[count:][parents] * ratio)
    )
    return jnp.concatenate((cells.values, faces)), cells


def fluid_native_transport(
    coarse: FluidOwner,
    fine: FluidOwner,
    transition: Any,
    lineage: Any,
    source: lc.CompositionEntry,
    target: lc.CompositionEntry,
    /,
) -> lc.CompositionTransport:
    """Physical-remap evidence: [cell heat, accumulated face heat] before and after."""
    native = jnp.asarray(source.value)
    image, cells = fluid_native_image(coarse, fine, transition, lineage, native)
    staged = jnp.asarray(target.value)
    eps = jnp.finfo(staged.dtype).eps
    scale = jnp.maximum(jnp.max(jnp.abs(image)), 1.0)
    count = coarse.grid.cell_count
    source_content = jnp.stack((cells.source_content, jnp.sum(native[count:])))
    target_content = jnp.stack(
        (cells.target_content, jnp.sum(image[fine.grid.cell_count :]))
    )
    return lc.CompositionTransport(
        "physical-remap",
        (source.entry_id,),
        (target,),
        source_structure_ids=(coarse.epoch.epoch_id,),
        route_id=f"{transition.transition_id}|{lineage.lineage_id}",
        successful=cells.successful
        & (jnp.max(jnp.abs(staged - image)) <= 100 * eps * scale),
        source_content=source_content,
        target_content=target_content,
        content_tolerance=100 * eps * jnp.maximum(jnp.abs(source_content), 1.0),
    )


def stage_fluid_refinement(
    model: CoupledModel, composition: lc.Composition, /
) -> tuple[CoupledModel, lc.CompositionRebind]:
    """Host staging of a refined fluid; any owner refusal raises and publishes nothing."""
    _, state = cpl.coupling_state_from_composition(composition)
    coarse = model.fluid
    fine = fluid_owner(2 * coarse.cells[0], 2 * coarse.cells[1], coarse.epoch.index + 1)
    transition = nested_fluid_remap(coarse, fine).epoch_transition(
        coarse.discretization.cell_space,
        fine.discretization.cell_space,
        coarse.epoch,
        fine.epoch,
    )
    lineage = interface_face_lineage(coarse, fine)
    checkpoint = participant(state, "fluid-fv")
    image, _ = fluid_native_image(coarse, fine, transition, lineage, checkpoint.native)
    participants = (model.participants[0], fluid_participant(fine, model.randomness))
    route = interface(model.solid, fine)
    states = {
        "solid-fe": participant(state, "solid-fe"),
        "fluid-fv": cpl.MethodParticipantState(
            image,
            checkpoint.model_state,
            checkpoint.key_data,
            checkpoint.accepted_windows,
            checkpoint.native_steps,
        ),
    }
    start = float(np.asarray(state.time))
    epoch = prepare_epoch(
        model.solid, fine, route, participants, states, start, model.window_size
    )
    target_model = model._replace(
        fluid=fine,
        interface=route,
        participants=participants,
        fluid_probe=fluid_probe(fine),
        epoch=epoch,
    )
    target = target_entries(target_model)
    # The model state and carried key cross only if the fluid owner declared them
    # independent of its grid; otherwise they are stale and the rebind refuses.
    retained = (
        *(item for item in composition.entry_ids if item.startswith("solid-fe/")),
        *(
            item
            for item in ("fluid-fv/model-state", "fluid-fv/rng", "fluid-fv/windows")
            if item in composition.entry_ids
        ),
        "exchange/interface-heat",
        "coupling/budget",
        "coupling/clock",
    )
    reprepared = (
        "fluid-fv/discretization",
        "fluid-fv/probe",
        "interface/route",
        "coupling/epoch",
    )
    rebind = lc.CompositionRebind(
        composition,
        retain=retained,
        reprepare=tuple(target[item] for item in reprepared),
        transports=(
            fluid_native_transport(
                coarse,
                fine,
                transition,
                lineage,
                composition.entry("fluid-fv/native"),
                target["fluid-fv/native"],
            ),
            cpl.coupling_exchange_transport(
                composition, epoch, tuple(target.values()), "interface-temperature"
            ),
        ),
    )
    return target_model, rebind


def rebind(
    model: CoupledModel,
    state: cpl.CouplingState,
    side: Side,
    /,
    *,
    accepted: bool,
) -> tuple[CoupledModel, cpl.CouplingState, lc.CompositionRebindReceipt]:
    """Stage one side's refinement and publish it at one accepted boundary."""
    composition = compose(model, state)
    match side:
        case "solid":
            target_model, staged = stage_solid_refinement(model, composition)
        case "fluid":
            target_model, staged = stage_fluid_refinement(model, composition)
        case _:
            assert_never(side)
    receipt = lc.commit_composition_rebind(staged, accepted_boundary=accepted)
    if not receipt.published:
        return model, state, receipt
    epoch, published = cpl.coupling_state_from_composition(receipt.composition)
    return target_model._replace(epoch=epoch), published, receipt


# Independent semi-discrete reference (host SciPy, hand-assembled P1 matrices).


def p1_matrices(
    vertices: np.ndarray, triangles: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
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


def coupled_propagator(
    divisions: int, cells: tuple[int, int], time: float, /
) -> np.ndarray:
    """exp(t C⁻¹G) of the coupled semi-discrete system [solid nodes; fluid cells]."""
    vertices, triangles, nodes = square_mesh(divisions)
    mass, stiffness = p1_matrices(vertices, triangles)
    nx, ny = cells
    hx, hy = 1.0 / nx, 1.0 / ny
    cell = np.arange(nx * ny).reshape(nx, ny)
    laplacian = np.zeros((nx * ny, nx * ny))
    for owners, neighbors, conductance in (
        (cell[:-1, :], cell[1:, :], hy / hx),
        (cell[:, :-1], cell[:, 1:], hx / hy),
    ):
        np.add.at(laplacian, (owners, owners), conductance)
        np.add.at(laplacian, (neighbors, neighbors), conductance)
        np.add.at(laplacian, (owners, neighbors), -conductance)
        np.add.at(laplacian, (neighbors, owners), -conductance)
    solid_trace = np.eye(mass.shape[0])[nodes]
    fluid_trace = np.eye(nx * ny)[cell[0, :]]
    face_average = face_averages(vertices[nodes, 1], np.linspace(0.0, 1.0, ny + 1))
    face_average = face_average @ solid_trace
    face = FLUID_CONDUCTANCE * hy / (0.5 * hx)
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
        SOLID_HEAT_CAPACITY * mass, FLUID_HEAT_CAPACITY * hx * hy * np.eye(nx * ny)
    )
    size = generator.shape[0]
    rates = np.zeros((size, size))
    rates[...] = np.linalg.solve(capacity, generator)
    return scipy.linalg.expm(time * rates)


def red_prolongation(values: np.ndarray, divisions: int, /) -> np.ndarray:
    """Nested P1 values on the red-refined grid: midpoint means of coarse edges."""
    coarse = values.reshape(divisions + 1, divisions + 1)
    index = np.arange(2 * divisions + 1)
    low, high = index // 2, (index + 1) // 2
    return (
        0.5 * (coarse[low[:, None], low[None, :]] + coarse[high[:, None], high[None, :]])
    ).reshape(-1)


def reference_temperatures(side: Side | None, /) -> np.ndarray:
    """exp(t A) on the source topology, exact prolongation, exp(t A) on the target."""
    vertices, _, _ = square_mesh(SOLID_DIVISIONS)
    nx, ny = FLUID_CELLS
    solid = 1.0 + 0.5 * np.cos(np.pi * vertices[:, 1]) * vertices[:, 0]
    fluid = np.tile(0.25 * np.arange(ny) / ny, nx)
    source = coupled_propagator(SOLID_DIVISIONS, FLUID_CELLS, REBIND_TIME)
    boundary = source @ np.concatenate((solid, fluid))
    solid, fluid = boundary[: vertices.shape[0]], boundary[vertices.shape[0] :]
    divisions, cells = SOLID_DIVISIONS, FLUID_CELLS
    match side:
        case None:
            pass
        case "solid":
            solid, divisions = red_prolongation(solid, divisions), 2 * divisions
        case "fluid":
            fine = np.arange(4 * nx * ny).reshape(2 * nx, 2 * ny)
            fluid = fluid.reshape(nx, ny)[fine // (2 * ny) // 2, (fine % (2 * ny)) // 2]
            fluid, cells = fluid.reshape(-1), (2 * nx, 2 * ny)
        case _:
            assert_never(side)
    target = coupled_propagator(divisions, cells, FINAL_TIME - REBIND_TIME)
    return target @ np.concatenate((solid, fluid))


# Runs and evidence.


class RebindRun(NamedTuple):
    before: CoupledModel
    after: CoupledModel
    boundary: cpl.CouplingState
    published: cpl.CouplingState
    final: cpl.CouplingState
    receipt: lc.CompositionRebindReceipt | None


def run(window_size: float, side: Side | None, /) -> RebindRun:
    """Accept windows to t = 0.2, rebind one side (or none), continue to t = 0.4."""
    model, state = build(window_size)
    boundary, result = advance(model, state, round(REBIND_TIME / window_size))
    after, published, receipt = model, boundary, None
    if side is not None:
        after, published, receipt = rebind(
            model, boundary, side, accepted=bool(result.successful)
        )
    windows = round((FINAL_TIME - REBIND_TIME) / window_size)
    final, _ = advance(after, published, windows)
    return RebindRun(model, after, boundary, published, final, receipt)


def temperatures(model: CoupledModel, state: cpl.CouplingState, /) -> np.ndarray:
    """Solid nodal temperatures followed by fluid cell temperatures."""
    cells = model.fluid.grid.cell_count
    solid = np.asarray(participant(state, "solid-fe").native)
    return np.concatenate(
        (solid, np.asarray(participant(state, "fluid-fv").native)[:cells])
    )


def convergence() -> None:
    print(
        "continued convergence to the rebound semi-discrete reference at t =", FINAL_TIME
    )
    print(f"  {'refined side':14s} {'window':>8s} {'max error':>11s} {'rate':>6s}")
    for side in ("solid", "fluid"):
        reference = reference_temperatures(side)
        errors = []
        for window_size in WINDOW_SIZES:
            outcome = run(window_size, side)
            if outcome.receipt is None or not outcome.receipt.published:
                raise RuntimeError(f"The {side} rebind was not published.")
            computed = temperatures(outcome.after, outcome.final)
            errors.append(float(np.max(np.abs(computed - reference))))
            rate = "" if len(errors) == 1 else f"{np.log2(errors[-2] / errors[-1]):6.3f}"
            print(f"  {side:14s} {window_size:8.4f} {errors[-1]:11.3e} {rate:>6s}")
        rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
        if np.any(np.abs(rates - 1.0) > 0.2):
            raise RuntimeError(f"The {side}-refined run is not first order.")


def rebind_evidence(side: Side, /) -> None:
    outcome = run(0.05, side)
    receipt = outcome.receipt
    if receipt is None:
        raise RuntimeError("Missing rebind receipt.")
    unchanged = "fluid-fv" if side == "solid" else "solid-fe"
    kept = participant(outcome.boundary, unchanged)
    retained = participant(outcome.published, unchanged)
    before = sum(energies(outcome.before, outcome.boundary))
    after = sum(energies(outcome.after, outcome.published))
    initial_model, initial = build(0.05)
    start = sum(energies(initial_model, initial))
    end = sum(energies(outcome.after, outcome.final))
    transport = receipt.transports[0]
    evidence = {
        "published": receipt.published,
        "retained": ", ".join(receipt.retained),
        "reprepared": ", ".join(receipt.reprepared),
        "remapped": ", ".join(receipt.remapped),
        "content before": np.asarray(transport.source_content).tolist(),
        "content after": np.asarray(transport.target_content).tolist(),
        "energy change at rebind": (after - before) / abs(before),
        "total energy change 0 -> 0.4": (end - start) / abs(start),
        f"{unchanged} bitwise retained": kept.native is retained.native
        and bool(eqx.tree_equal(kept, retained)),
        "budgets retained": np.array_equal(
            np.asarray(outcome.boundary.cumulative_exchange_budget),
            np.asarray(outcome.published.cumulative_exchange_budget),
        ),
        "probes before": observe(outcome.before, outcome.boundary),
        "probes after": observe(outcome.after, outcome.published),
    }
    print(f"{side} refinement at the accepted boundary t = {REBIND_TIME}:")
    for name, value in evidence.items():
        print(f"  {name:32s} {value}")
    probes = np.asarray(evidence["probes before"]) - np.asarray(evidence["probes after"])
    if (
        not receipt.published
        or abs(evidence["energy change at rebind"]) > 1e-13
        or abs(evidence["total energy change 0 -> 0.4"]) > 1e-12
        or not evidence[f"{unchanged} bitwise retained"]
        or not evidence["budgets retained"]
        or np.max(np.abs(probes)) > 1e-13
    ):
        raise RuntimeError(f"The {side} rebind evidence failed.")


def non_nested_solid() -> SolidOwner:
    """A remeshed solid whose interior vertex moved: no conservative P1 route."""
    vertices, triangles, nodes = square_mesh(2 * SOLID_DIVISIONS)
    vertices = vertices.copy()
    vertices[4 * (2 * SOLID_DIVISIONS + 1) + 4] += (0.03, 0.02)
    return solid_owner(vertices, triangles, nodes, 1)


def refusals() -> None:
    window = 0.05
    model, state = build(window)
    boundary, _ = advance(model, state, round(REBIND_TIME / window))
    composition = compose(model, boundary)
    target_model, staged = stage_solid_refinement(model, composition)
    outcomes: dict[str, str] = {}
    try:
        lc.CompositionRebind(
            composition,
            retain=(*staged.retained, "solid-fe/probe"),
            reprepare=tuple(
                staged.candidate.entry(item)
                for item in staged.reprepared
                if item != "solid-fe/probe"
            ),
            transports=staged.transports,
        )
    except ValueError as error:
        outcomes["stale observation retained"] = str(error)
    dependent, dependent_state = build(window, fluid_model_independent=False)
    dependent_boundary, _ = advance(
        dependent, dependent_state, round(REBIND_TIME / window)
    )
    try:
        stage_fluid_refinement(dependent, compose(dependent, dependent_boundary))
    except ValueError as error:
        outcomes["unknown fluid model state"] = str(error)
    try:
        stage_solid_refinement(model, composition, fine=non_nested_solid())
    except ValueError as error:
        outcomes["failed target preparation"] = str(error)
    rejected = lc.commit_composition_rebind(staged, accepted_boundary=False)
    del target_model
    windows = round((FINAL_TIME - REBIND_TIME) / window)
    continued, _ = advance(model, boundary, windows)
    untouched, _ = advance(*build(window), round(FINAL_TIME / window))
    dependent_continued, _ = advance(dependent, dependent_boundary, windows)
    print("refused rebinds leave every old owner usable:")
    for name, message in outcomes.items():
        print(f"  {name:28s} {message[:96]}")
    evidence = {
        "rejected boundary published": rejected.published,
        "rejected keeps composition": rejected.composition is composition,
        "old owners continue bitwise": bool(eqx.tree_equal(continued, untouched)),
        "dependent owners continue": bool(
            dependent_continued.window_index == continued.window_index
        ),
    }
    for name, value in evidence.items():
        print(f"  {name:28s} {value}")
    if (
        len(outcomes) != 3
        or evidence["rejected boundary published"]
        or not evidence["rejected keeps composition"]
        or not evidence["old owners continue bitwise"]
        or not evidence["dependent owners continue"]
    ):
        raise RuntimeError("Refused rebinds did not keep the old composition usable.")


if __name__ == "__main__":
    convergence()
    rebind_evidence("solid")
    rebind_evidence("fluid")
    refusals()
