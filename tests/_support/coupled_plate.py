#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Heat conduction across a finite-element and a virtual-element region.

The plate ``[0, 2] x [0, 1]`` is split at ``x = 1`` into P1 triangles (left)
and degree-1 virtual elements on a perturbed brick mesh (right) whose interface
vertices do not coincide with the triangles'. ``-div(kappa grad u) = s`` holds
in both regions with conductivities ``kappa_left`` and ``kappa_right`` (the
triangles' conductivity may instead be a learned field), ``u = 0`` on
``x = 0``, a prescribed boundary heat flux ``g = kappa du/dn`` on ``x = 2``,
and insulated top and bottom sides. A mortar ``ScalarTransmissionLaw`` couples
the regions.

The exact solution for constant conductivities depends on ``x`` only and is
evaluated on the host (``exact_temperature``); it is independent of both
discretizations and serves as the observation reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain
from phydrax.solver import coupling as cpl


INTERFACE_X = 1.0
SOURCE = 2.0

# Units of the 2-D (per unit depth) conduction problem.
WATT = phx.units.derived_unit("W", ((phx.units.JOULE, 1), (phx.units.SECOND, -1)))
CONDUCTIVITY_UNIT = phx.units.derived_unit(
    "W/(m K)", ((WATT, 1), (phx.units.METER, -1), (phx.units.KELVIN, -1))
)
HEAT_FLUX_UNIT = phx.units.derived_unit("W/m^2", ((WATT, 1), (phx.units.METER, -2)))
LINE_HEAT_UNIT = phx.units.derived_unit("W/m", ((WATT, 1), (phx.units.METER, -1)))

type Coefficient = Callable[[Array, object], Array]


def exact_temperature(
    points: np.ndarray, kappa_left: float, kappa_right: float, flux: float, /
) -> np.ndarray:
    """Host reference ``u(x)`` of the piecewise-constant-conductivity plate."""
    x = np.asarray(points, dtype=np.float64)[..., 0]
    left = (-0.5 * SOURCE * x**2 + (2.0 * SOURCE + flux) * x) / kappa_left
    offset = (1.5 * SOURCE + flux) / kappa_left
    right = (
        -0.5 * SOURCE * (x - 1.0) ** 2 + (SOURCE + flux) * (x - 1.0)
    ) / kappa_right + offset
    return np.where(x <= INTERFACE_X, left, right)


def exact_left_wall_heat(flux: float, /) -> float:
    """Conormal flux content ``int kappa du/dn ds`` through ``x = 0`` per unit depth.

    With the outward normal ``n = -e_x`` this is the heat entering the plate
    through the left wall, the quantity ``FieldFluxObservation`` reports: every
    watt generated (``2 s``) or supplied (``g``) leaves there, so it is
    ``-(2 s + g)``.
    """
    return -(2.0 * SOURCE + flux)


def _selected_facets(
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


def _triangle_mesh(resolution: int, /) -> phx.discretization.CellMesh:
    axis = np.linspace(0.0, 1.0, resolution + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    triangles: list[tuple[int, int, int]] = []
    for row in range(resolution):
        for column in range(resolution):
            corner = row * (resolution + 1) + column
            upper = corner + resolution + 1
            triangles += [(corner, corner + 1, upper + 1), (corner, upper + 1, upper)]
    return phx.discretization.CellMesh(
        jnp.asarray(points),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", jnp.asarray(np.asarray(triangles, np.int32))
            ),
        ),
    )


def _brick_polygons(bricks: int, /) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Distorted running-bond hexagons and quadrilaterals on ``[1, 2] x [0, 1]``."""
    columns = 2 * bricks + 1
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 1.0, columns), np.linspace(0.0, 1.0, bricks + 1), indexing="xy"
    )
    line = np.arange(bricks + 1)[:, None]
    midpoint = (np.arange(columns) % 2 == 1)[None, :]
    interior = (line > 0) & (line < bricks)
    grid_y = grid_y + np.where(midpoint & interior, 0.15 * (-1.0) ** line / bricks, 0.0)
    mapped_y = grid_y + 0.06 * np.sin(2.0 * np.pi * grid_y) * (1.0 - 0.5 * grid_x)
    mapped_x = grid_x + 0.05 * np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
    points = np.stack((INTERFACE_X + mapped_x, mapped_y), axis=-1).reshape(-1, 2)
    cells: list[np.ndarray] = []
    for row in range(bricks):
        bounds = (
            [0, *range(1, columns - 1, 2), columns - 1]
            if row % 2
            else list(range(0, columns, 2))
        )
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            lower = [row * columns + column for column in range(start, stop + 1)]
            upper = [
                (row + 1) * columns + column for column in range(stop, start - 1, -1)
            ]
            cells.append(np.asarray(lower + upper, dtype=np.int32))
    return points, tuple(cells)


def runtime_value(arguments: object, name: str, /) -> Array:
    """One named runtime input from an owner's user arguments."""
    if not isinstance(arguments, Mapping):
        raise TypeError("Owner user arguments must be a mapping.")
    return jnp.asarray(arguments[name])


def uniform_triangle_conductivity(points: Array, context: object) -> Array:
    """``kappa_left``: FE coefficients read ``context.user_args``."""
    if not isinstance(context, phx.equations.FiniteElementExecutionContext):
        raise TypeError("FE coefficients receive the execution context.")
    return runtime_value(context.user_args, "conductivity") * jnp.ones(points.shape[:-1])


def _source(points: Array, args: object) -> Array:
    del args
    return SOURCE * jnp.ones(points.shape[:-1])


@dataclass(frozen=True)
class CoupledPlate:
    """Prepared-once pieces of the plate: components, law, binding, and domains."""

    triangles: cpl.VariationalComponent
    polygons: cpl.VariationalComponent
    law: cpl.ScalarTransmissionLaw
    binding: cpl.InterfaceBinding
    cover: phx.domain.SubdomainCover
    left_wall: IntegrationDomain
    right_wall: IntegrationDomain
    triangle_nodes: np.ndarray
    polygon_nodes: np.ndarray


def build_plate(
    level: int, /, *, triangle_conductivity: Coefficient = uniform_triangle_conductivity
) -> CoupledPlate:
    """Components of one refinement level (triangles ``4 * 2**level`` per side)."""
    fe_space = phx.discretization.FiniteElementPlan(
        _triangle_mesh(4 * 2**level),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    triangle_nodes = np.asarray(fe_space.dof_maps[0].dof_coordinates)
    fe = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "left-conduction",
            "u",
            (
                phx.equations.DiffusionAction(
                    "u",
                    phx.equations.coefficient(
                        triangle_conductivity, coefficient_id="conductivity-left"
                    ),
                ),
                phx.equations.SourceAction(
                    "u", phx.equations.coefficient(_source, coefficient_id="source")
                ),
            ),
        ),
        fe_space,
        constraint=phx.discretization.dirichlet_constraint(
            fe_space, "u", boundary_mask=np.isclose(triangle_nodes[:, 0], 0.0)
        ),
        dirichlet_values=0.0,
    )
    points, cells = _brick_polygons(3 * 2**level)
    vem_space = phx.discretization.VirtualElementPlan(
        phx.discretization.CellMesh.from_polygons(jnp.asarray(points), cells),
        phx.discretization.VirtualElementFieldSpec(
            "u", phx.discretization.conforming_h1_virtual_element(1)
        ),
    ).prepare()
    polygon_nodes = np.asarray(vem_space.dof_map.default_dof_points)
    right_wall = _selected_facets(vem_space, 2.0)

    def right_conductivity(points: Array, args: object) -> Array:
        return runtime_value(args, "conductivity") * jnp.ones(points.shape[:-1])

    def boundary_flux(points: Array, args: object) -> Array:
        return runtime_value(args, "heat-flux") * jnp.ones(points.shape[:-1])

    vem = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm(
            "right-conduction",
            "u",
            (
                phx.equations.DiffusionAction(
                    "u",
                    phx.equations.coefficient(
                        right_conductivity, coefficient_id="conductivity-right"
                    ),
                ),
                phx.equations.SourceAction(
                    "u", phx.equations.coefficient(_source, coefficient_id="source")
                ),
                phx.equations.BoundaryLoadAction(
                    "u",
                    phx.equations.coefficient(boundary_flux, coefficient_id="heat-flux"),
                    action_id="right-wall-heat-flux",
                    domain=right_wall,
                ),
            ),
        ),
        vem_space,
    )
    triangles = cpl.VariationalComponent("triangles", fe, field="u")
    polygons = cpl.VariationalComponent("polygons", vem, field="u")
    binding, cover = _interface(triangles, polygons)
    law = cpl.ScalarTransmissionLaw(
        "transmission",
        binding,
        (
            cpl.TransmissionSide(
                "triangles", "triangles", "u", _selected_facets(fe_space, 1.0)
            ),
            cpl.TransmissionSide(
                "polygons", "polygons", "u", _selected_facets(vem_space, 1.0)
            ),
        ),
        cpl.MortarImposition(cpl.MortarMultiplier("side-trace", side="polygons")),
    )
    return CoupledPlate(
        triangles=triangles,
        polygons=polygons,
        law=law,
        binding=binding,
        cover=cover,
        left_wall=_selected_facets(fe_space, 0.0),
        right_wall=right_wall,
        triangle_nodes=triangle_nodes,
        polygon_nodes=polygon_nodes,
    )


def _interface(
    triangles: cpl.VariationalComponent, polygons: cpl.VariationalComponent, /
) -> tuple[cpl.InterfaceBinding, phx.domain.SubdomainCover]:
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(plate, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    witness = pairing.component.sample(phx.domain.PointSampling(8), key=jr.key(1))
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
                (triangles, pairing.left_patch_id),
                (polygons, pairing.right_patch_id),
            )
        ),
    )
    return binding, cover


def conductivity_port(name: str, /) -> phx.ValuePort:
    """Scientific identity of one region's thermal conductivity."""
    return phx.ValuePort(
        f"thermal-conductivity-{name}",
        event_shape=(),
        component_ids=("kappa",),
        representation="scalar",
        dimensions=(CONDUCTIVITY_UNIT.dimension,),
    )


HEAT_FLUX_PORT = phx.ValuePort(
    "right-wall-heat-flux",
    event_shape=(),
    component_ids=("g",),
    representation="scalar",
    dimensions=(HEAT_FLUX_UNIT.dimension,),
)


def parameter_bindings() -> tuple[cpl.ParameterBinding, ...]:
    """Conductivities (physical parameters) and the right-wall heat flux (solver argument)."""
    left = cpl.ParameterBinding(
        "conductivity-left",
        conductivity_port("left"),
        targets=(cpl.RuntimeInput("triangles", "conductivity"),),
        role="coefficient",
        derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
    )
    right = cpl.ParameterBinding(
        "conductivity-right",
        conductivity_port("right"),
        targets=(cpl.RuntimeInput("polygons", "conductivity"),),
        role="coefficient",
        derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
    )
    flux = cpl.ParameterBinding(
        "heat-flux",
        HEAT_FLUX_PORT,
        targets=(cpl.RuntimeInput("polygons", "heat-flux"),),
        role="boundary",
        derivative=phx.DerivativeSurface.SOLVER_ARGUMENT,
    )
    return (left, right, flux)


def dense_policy(
    mode: phx.linalg.DifferentiationMode = "mathematical", /
) -> phx.linalg.LinearSolvePolicy:
    """Dense LU whose implicit derivative solves reuse the primal factors."""
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        differentiation=phx.linalg.DifferentiationPolicy(mode),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(route="primal-factors"),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=16_000_000, max_bytes=256 * 1024 * 1024
        ),
    )
