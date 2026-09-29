#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transient heat conduction across a finite-element and a virtual-element region.

Shared model of the coupled control and data-assimilation examples. The plate
``[0, 2] x [0, 1]`` is split at ``x = 1``: P1 triangles on the left (the
"conductor", ``rho c = 1``) and conforming degree-1 virtual elements on
quadrilaterals on the right (the "insulator", ``rho c = 2``), with matching
interface vertices, unit conductivity, and

``rho c u_t - Laplace(u) = s`` (``s = 1`` in the conductor only).

The left wall is held at ``u = 0``, top and bottom are insulated, and the right
wall ``x = 2`` receives the inward heat flux ``g = (g_lower, g_upper)`` on its
lower and upper halves. ``g`` is the P6 parameter binding ``"wall-heat-flux"``;
three temperature sensors are the P6 observation bindings ``"conductor-sensor"``
(finite element, exact point value) and ``"insulator-sensors"`` (virtual
element, labeled H1 projection). One ``ScalarTransmissionLaw`` with
``MatchingElimination`` couples the regions, so the semi-discrete system is an
ODE admitted as an index-one transient.

``host_semidiscrete`` is the independent reference of the consumers: it
assembles ``C``, ``K``, ``b``, ``B``, and the sensor rows ``H`` in NumPy from the
two meshes (P1 consistent mass and stiffness on the triangles; the degree-1
virtual-element projections with dofi-dofi stabilization on the squares;
barycentric and projected point evaluation). Only the coordinate chart of the
native state (which owner node each native coordinate is) is read from the
prepared transient, and it is verified against the mesh geometry.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain
from phydrax.solver import coupling as cpl


INTERFACE_X = 1.0
SOURCE = 1.0
CAPACITY = {"conductor": 1.0, "insulator": 2.0}
CONTROL = "wall-heat-flux"
SENSOR_BINDINGS = ("conductor-sensor", "insulator-sensors")
CONDUCTOR_SENSOR = np.asarray([[0.55, 0.45, 0.0]])
INSULATOR_SENSORS = np.asarray([[1.45, 0.3, 0.0], [1.6, 0.7, 0.0]])

WATT = phx.units.derived_unit("W", ((phx.units.JOULE, 1), (phx.units.SECOND, -1)))
HEAT_FLUX_UNIT = phx.units.derived_unit("W/m^2", ((WATT, 1), (phx.units.METER, -2)))
CONTRACT = phx.SpatialCoordinateContract(phx.units.METER)
TEMPERATURE = phx.measurement.QuantitySpec(
    "coupled-heat", "temperature", "temperature", phx.units.KELVIN, "temperature"
)
POINT = phx.measurement.SamplingSemantics(phx.measurement.SpatialSamplingKind.POINT)
WALL_HEAT_FLUX_PORT = phx.ValuePort(
    "right-wall-heat-flux",
    event_shape=(2,),
    component_ids=("g_lower", "g_upper"),
    representation="vector",
    dimensions=(HEAT_FLUX_UNIT.dimension, HEAT_FLUX_UNIT.dimension),
)
SENSOR_SUPPORTS = {
    "conductor-sensor": phx.measurement.PointSampleSupport(
        CONDUCTOR_SENSOR, ("c",), CONTRACT
    ),
    "insulator-sensors": phx.measurement.PointSampleSupport(
        INSULATOR_SENSORS, ("i-lower", "i-upper"), CONTRACT
    ),
}


def _grid_points(x0: float, x1: float, cells: int, /) -> np.ndarray:
    xs = np.linspace(x0, x1, cells + 1)
    ys = np.linspace(0.0, 1.0, cells + 1)
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)


def _quads(cells: int, /) -> list[tuple[int, int, int, int]]:
    result = []
    for row in range(cells):
        for column in range(cells):
            a = row * (cells + 1) + column
            result.append((a, a + 1, a + cells + 2, a + cells + 1))
    return result


def _facets_on(
    space: phx.discretization.FiniteElementDiscretization
    | phx.discretization.VirtualElementDiscretization,
    x: float,
    /,
) -> IntegrationDomain:
    """The owner's exterior facets lying on the vertical line ``x``."""
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    facets = np.all(np.isclose(np.asarray(probe.sites)[..., 0], x), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[facets]] = True
    return space.integration_domain("exterior_facet", EntitySelection(edges, mask))


def _conductor_source(points: Array, args: object) -> Array:
    del args
    return jnp.full(points.shape[:-1], SOURCE)


def _wall_flux(points: Array, args: object) -> Array:
    """Inward heat flux ``g_lower`` below ``y = 1/2`` and ``g_upper`` above."""
    if not isinstance(args, Mapping):
        raise TypeError("The insulator reads the bound heat flux from a mapping.")
    flux = jnp.asarray(args[CONTROL])
    return jnp.where(points[..., 1] < 0.5, flux[0], flux[1])


def _conductor(cells: int, /) -> tuple[cpl.VariationalComponent, IntegrationDomain]:
    triangles = [t for a, b, c, d in _quads(cells) for t in ((a, b, c), (a, c, d))]
    mesh = phx.discretization.CellMesh(
        _grid_points(0.0, INTERFACE_X, cells),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    nodes = np.asarray(space.dof_maps[0].dof_coordinates)
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "conductor-heat",
            "u",
            (
                phx.equations.DiffusionAction("u"),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(
                        _conductor_source, coefficient_id="conductor-source"
                    ),
                ),
            ),
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=np.isclose(nodes[:, 0], 0.0)
        ),
        dirichlet_values=0.0,
    )
    component = cpl.VariationalComponent("conductor", problem, field="u")
    return component, _facets_on(space, INTERFACE_X)


def _insulator(cells: int, /) -> tuple[cpl.VariationalComponent, IntegrationDomain]:
    mesh = phx.discretization.CellMesh.from_polygons(
        jnp.asarray(_grid_points(INTERFACE_X, 2.0, cells)),
        tuple(np.asarray(cell, dtype=np.int32) for cell in _quads(cells)),
    )
    space = phx.discretization.VirtualElementPlan(
        mesh,
        phx.discretization.VirtualElementFieldSpec(
            "u", phx.discretization.conforming_h1_virtual_element(1)
        ),
    ).prepare()
    problem = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm(
            "insulator-heat",
            "u",
            (
                phx.equations.DiffusionAction("u"),
                phx.equations.BoundaryLoadAction(
                    "u",
                    phx.equations.coefficient(_wall_flux, coefficient_id=CONTROL),
                    action_id="right-wall-heat-flux",
                    domain=_facets_on(space, 2.0),
                ),
            ),
        ),
        space,
    )
    component = cpl.VariationalComponent("insulator", problem, field="u")
    return component, _facets_on(space, INTERFACE_X)


def _interface(
    conductor: cpl.VariationalComponent, insulator: cpl.VariationalComponent, /
) -> tuple[cpl.InterfaceBinding, phx.domain.SubdomainCover]:
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(plate, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    witness = pairing.component.sample(phx.domain.PointSampling(8), key=jax.random.key(1))
    binding = cpl.InterfaceBinding(
        "cut",
        cpl.InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        tuple(
            cpl.InterfaceEndpoint(
                component.name,
                cpl.PairedSupportAttachment(cover, pairing.pairing_id, patch, witness),
                fields={"value": component.field_space_id("u")},
            )
            for component, patch in (
                (conductor, pairing.left_patch_id),
                (insulator, pairing.right_patch_id),
            )
        ),
    )
    return binding, cover


def wall_heat_flux_binding() -> cpl.ParameterBinding:
    """The controlled (or forcing) right-wall heat flux as a P6 parameter binding."""
    return cpl.ParameterBinding(
        CONTROL,
        WALL_HEAT_FLUX_PORT,
        targets=(cpl.RuntimeInput("insulator", CONTROL),),
        role="control",
        derivative=phx.DerivativeSurface.SOLVER_ARGUMENT,
    )


def sensor_bindings() -> tuple[cpl.FieldPointObservation, ...]:
    """Point temperature sensors in both regions as P6 observation bindings."""
    return tuple(
        cpl.FieldPointObservation(
            binding_id,
            component,
            "u",
            quantity=TEMPERATURE,
            support=SENSOR_SUPPORTS[binding_id],
            sampling=POINT,
            field_unit=phx.units.KELVIN,
        )
        for binding_id, component in zip(
            SENSOR_BINDINGS, ("conductor", "insulator"), strict=True
        )
    )


def coupled_heat_problem(cells: int, /) -> cpl.PreparedCoupledProblem:
    """Prepared spatial FE-VEM conduction problem with its bindings."""
    conductor, conductor_interface = _conductor(cells)
    insulator, insulator_interface = _insulator(cells)
    binding, cover = _interface(conductor, insulator)
    law = cpl.ScalarTransmissionLaw(
        "cut-heat",
        binding,
        (
            cpl.TransmissionSide("conductor", "conductor", "u", conductor_interface),
            cpl.TransmissionSide("insulator", "insulator", "u", insulator_interface),
        ),
        cpl.MatchingElimination(eliminated="insulator"),
    )
    plan = cpl.CoupledProblemPlan(
        "coupled-heat",
        components=(conductor, insulator),
        bindings=(binding,),
        laws=(law,),
        parameters=(wall_heat_flux_binding(),),
        observations=sensor_bindings(),
    )
    return cpl.prepare_coupled_problem(
        plan,
        interface_owners=(cover,),
        parameters={CONTROL: jnp.zeros((2,))},
    )


def coupled_heat_transient(cells: int, /) -> cpl.PreparedCoupledTransient:
    """The transient lowering: each owner's capacity, index-one admission."""
    prepared = coupled_heat_problem(cells)

    def arguments(time: Array, parameters: object) -> Mapping[str, object]:
        del time
        if not isinstance(parameters, Mapping):
            raise TypeError("Transient parameters are the bound parameter values.")
        return prepared.bind_arguments(parameters=parameters).arguments

    return cpl.prepare_coupled_transient(
        prepared,
        fields=tuple(
            cpl.TransientField(name, "u", capacity=value)
            for name, value in CAPACITY.items()
        ),
        arguments=arguments,
        arguments_id="wall-heat-flux-bound",
        parameters={CONTROL: jnp.zeros((2,))},
    )


def stage_termination() -> phx.nonlinear.NonlinearTermination:
    """Residual-certified implicit stages of the affine heat rows."""
    return phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-11,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=8,
    )


def implicit_euler_policy() -> phx.solver.DAESolvePolicy:
    """Fixed-step BDF1 (implicit Euler), the native one-step transition.

    Stage and tangent solves are matrix-free GMRES converged to roundoff, so
    the solution-map derivatives the linearization consumes are exact to
    solver precision.
    """
    stage = phx.nonlinear.NewtonKrylov(
        linear_policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.GMRES(restart=64),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1e-13, absolute=1e-15, max_steps=256
            ),
        )
    )
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(1),
        nonlinear_method=stage,
        initialization_method=stage,
        nonlinear_termination=stage_termination(),
        initialization_termination=stage_termination(),
    )


@dataclass(frozen=True)
class HostSemidiscrete:
    """Host ``C z' + K z = b + B g`` and sensor rows ``y = H z`` in native coordinates."""

    capacity: np.ndarray
    stiffness: np.ndarray
    source: np.ndarray
    flux: np.ndarray
    sensors: np.ndarray


def _coordinate_chart(
    transient: cpl.PreparedCoupledTransient, cells: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """0/1 maps from native coordinates to the owners' nodal coefficients.

    Probed by unit native states through ``PreparedCoupledTransient.field``
    and verified against the mesh geometry: the Dirichlet wall ``x = 0`` is
    dropped, every other node carries exactly one native coordinate, and all
    nodes of one coordinate coincide (the matching interface nodes of both
    owners share one).
    """
    size = transient.state_space.size
    parameters = {CONTROL: jnp.zeros((2,))}
    charts = tuple(
        np.stack(
            [
                np.asarray(
                    transient.field(
                        name,
                        "u",
                        0.0,
                        transient.state_view(jnp.asarray(unit)),
                        parameters,
                    )
                )
                for unit in np.eye(size)
            ],
            axis=1,
        )
        for name in ("conductor", "insulator")
    )
    stacked = np.concatenate(charts)
    selection = np.rint(stacked)
    if not np.allclose(stacked, selection, rtol=0.0, atol=1e-12) or not np.all(
        (selection == 0.0) | (selection == 1.0)
    ):
        raise RuntimeError("The native coordinates are not a nodal selection.")
    nodes = np.concatenate(
        (_grid_points(0.0, INTERFACE_X, cells), _grid_points(INTERFACE_X, 2.0, cells))
    )
    counts = selection.sum(axis=1)
    dirichlet = np.zeros(nodes.shape[0], dtype=np.bool_)
    dirichlet[: nodes.shape[0] // 2] = np.isclose(nodes[: nodes.shape[0] // 2, 0], 0.0)
    if not np.array_equal(counts, np.where(dirichlet, 0.0, 1.0)):
        raise RuntimeError("A free node is not exactly one native coordinate.")
    for column in selection.T:
        located = nodes[column == 1.0]
        if located.shape[0] == 0 or not np.allclose(located, located[0], atol=1e-14):
            raise RuntimeError("A native coordinate ties nodes at different points.")
    return selection[: nodes.shape[0] // 2], selection[nodes.shape[0] // 2 :]


def _triangle_system(cells: int, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P1 consistent mass, stiffness, and unit-source load of the conductor."""
    nodes = _grid_points(0.0, INTERFACE_X, cells)
    count = nodes.shape[0]
    mass = np.zeros((count, count))
    stiffness = np.zeros((count, count))
    load = np.zeros(count)
    reference_gradients = np.asarray([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    for a, b, c, d in _quads(cells):
        for triangle in ((a, b, c), (a, c, d)):
            vertices = nodes[list(triangle)]
            jacobian = np.stack((vertices[1] - vertices[0], vertices[2] - vertices[0]), 1)
            area = 0.5 * abs(np.linalg.det(jacobian))
            gradients = np.linalg.solve(jacobian.T, reference_gradients)
            block = np.ix_(triangle, triangle)
            mass[block] += area / 12.0 * (np.ones((3, 3)) + np.eye(3))
            stiffness[block] += area * gradients.T @ gradients
            load[list(triangle)] += SOURCE * area / 3.0
    return mass, stiffness, load


def _square_projection(
    vertices: np.ndarray, point: ArrayLike, /
) -> tuple[np.ndarray, float, np.ndarray]:
    """Degree-1 VEM projection of the vertex basis on an axis-aligned square.

    ``Pi phi_i(x) = 1/4 + s_i . (x - x_E) / (2 h)`` with ``s_i`` the sign of
    vertex ``i`` about the centroid ``x_E``: the gradient is the boundary
    integral ``|E|^-1 int phi_i n ds`` and the constant fixes the vertex mean.
    Returns the signs, the side ``h``, and ``Pi phi_i(point)``.
    """
    centroid = vertices.mean(axis=0)
    side = float(vertices[:, 0].max() - vertices[:, 0].min())
    signs = np.sign(vertices - centroid)
    offset = np.asarray(point, dtype=np.float64) - centroid
    return signs, side, 0.25 + signs @ offset / (2.0 * side)


def _square_system(cells: int, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Degree-1 VEM mass, stiffness, and wall-flux load of the insulator.

    The consistent parts integrate the projections exactly; the dofi-dofi
    stabilization ``tau (I - Pi)^T (I - Pi)`` scales with
    ``tau = trace(consistent) / rank`` (``rank = dim P1 = 3`` for the mass,
    ``dim P1 - 1 = 2`` for the stiffness), the owner's default policy.
    """
    nodes = _grid_points(INTERFACE_X, 2.0, cells)
    count = nodes.shape[0]
    mass = np.zeros((count, count))
    stiffness = np.zeros((count, count))
    for quad in _quads(cells):
        vertices = nodes[list(quad)]
        signs, side, _ = _square_projection(vertices, vertices[0])
        gram = signs @ signs.T
        projector = 0.25 + gram / 4.0
        kernel = (np.eye(4) - projector).T @ (np.eye(4) - projector)
        local_mass = side**2 / 16.0 + side**2 * gram / 48.0
        local_stiffness = gram / 4.0
        block = np.ix_(quad, quad)
        mass[block] += local_mass + np.trace(local_mass) / 3.0 * kernel
        stiffness[block] += local_stiffness + np.trace(local_stiffness) / 2.0 * kernel
    load = np.zeros((count, 2))
    wall = np.flatnonzero(np.isclose(nodes[:, 0], 2.0))
    for lower, upper in zip(wall[:-1], wall[1:], strict=True):
        length = nodes[upper, 1] - nodes[lower, 1]
        column = 0 if 0.5 * (nodes[lower, 1] + nodes[upper, 1]) < 0.5 else 1
        load[[lower, upper], column] += 0.5 * length
    return mass, stiffness, load


def _sensor_rows(cells: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Nodal weights of the conductor (P1) and insulator (projected) sensors."""
    conductor_nodes = _grid_points(0.0, INTERFACE_X, cells)
    conductor = np.zeros((CONDUCTOR_SENSOR.shape[0], conductor_nodes.shape[0]))
    for row, point in enumerate(CONDUCTOR_SENSOR[:, :2]):
        for a, b, c, d in _quads(cells):
            for triangle in ((a, b, c), (a, c, d)):
                vertices = conductor_nodes[list(triangle)]
                weights = np.linalg.solve(
                    np.vstack((vertices.T, np.ones(3))), np.append(point, 1.0)
                )
                if np.all(weights >= 0.0):
                    conductor[row] = 0.0
                    conductor[row, list(triangle)] = weights
    insulator_nodes = _grid_points(INTERFACE_X, 2.0, cells)
    insulator = np.zeros((INSULATOR_SENSORS.shape[0], insulator_nodes.shape[0]))
    for row, point in enumerate(INSULATOR_SENSORS[:, :2]):
        for quad in _quads(cells):
            vertices = insulator_nodes[list(quad)]
            if np.all(point >= vertices.min(axis=0)) and np.all(
                point <= vertices.max(axis=0)
            ):
                insulator[row] = 0.0
                insulator[row, list(quad)] = _square_projection(vertices, point)[2]
    return conductor, insulator


def host_semidiscrete(
    transient: cpl.PreparedCoupledTransient, cells: int, /
) -> HostSemidiscrete:
    """Host assembly of the semi-discrete system and sensors on ``cells`` per side.

    Each owner's nodal operators are assembled from its mesh, weighted by its
    ``rho c`` (``CAPACITY``), and restricted to the native coordinates by the
    verified chart ``P``: ``C = sum P^T (rho c M) P``, ``K = sum P^T A P``,
    ``b = P_c^T f``, ``B = P_i^T F``, ``H = blockdiag(W_c P_c, W_i P_i)``.
    """
    conductor, insulator = _coordinate_chart(transient, cells)
    conductor_mass, conductor_stiffness, load = _triangle_system(cells)
    insulator_mass, insulator_stiffness, wall_load = _square_system(cells)
    conductor_sensor, insulator_sensors = _sensor_rows(cells)
    return HostSemidiscrete(
        capacity=CAPACITY["conductor"] * conductor.T @ conductor_mass @ conductor
        + CAPACITY["insulator"] * insulator.T @ insulator_mass @ insulator,
        stiffness=conductor.T @ conductor_stiffness @ conductor
        + insulator.T @ insulator_stiffness @ insulator,
        source=conductor.T @ load,
        flux=insulator.T @ wall_load,
        sensors=np.concatenate(
            (conductor_sensor @ conductor, insulator_sensors @ insulator)
        ),
    )
