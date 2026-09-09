#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-topology, rotating planar magnetostatic machine geometry.

SI units throughout. This profile is magnetostatic at prescribed mechanical
angles: linear isotropic magnetic materials, impressed axial winding currents,
and permanent remanence. It does not model wave propagation, nonlinear B-H,
eddy/hysteresis losses, thermal feedback, motion-induced voltage, or end effects.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._strict import StrictModule
from phydrax.discretization import (
    CellMesh,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)


VACUUM_PERMEABILITY = 4.0e-7 * np.pi


class LinearMagneticRegion(StrictModule):
    """One isotropic region; remanence is in its material coordinate frame.

    ``winding_turn_density[k]`` is signed turns per square metre along +z
    for circuit k. Negative density is the return side of that winding.
    ``rotating`` rotates the region's remanence with the rotor, not its currents;
    this profile supports stationary windings only.
    """

    name: str = eqx.field(static=True)
    relative_permeability: float = eqx.field(static=True)
    remanence: Array
    winding_turn_density: Array
    rotating: bool = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        relative_permeability: float = 1.0,
        *,
        remanence: ArrayLike | Sequence[float] = (0.0, 0.0),
        winding_turn_density: ArrayLike | Sequence[float] = (),
        rotating: bool = False,
    ):
        mu = float(relative_permeability)
        br = np.asarray(remanence, dtype=float)
        winding = np.asarray(winding_turn_density, dtype=float)
        if not name or not isfinite(mu) or mu <= 0.0:
            raise ValueError("Regions require a name and positive finite permeability.")
        if br.shape != (2,) or not np.all(np.isfinite(br)):
            raise ValueError("Remanence must be a finite planar vector.")
        if winding.ndim != 1 or not np.all(np.isfinite(winding)):
            raise ValueError("Winding turn densities must be a finite vector.")
        if rotating and np.any(winding != 0.0):
            raise ValueError("This profile requires stationary windings.")
        self.name = str(name)
        self.relative_permeability = mu
        self.remanence = jnp.asarray(br)
        self.winding_turn_density = jnp.asarray(winding)
        self.rotating = bool(rotating)


class PlanarMachine(StrictModule):
    """A prepared H1-P1 Az model at one prescribed rotor angle.

    Coordinates at a nearby angle are obtained by rotating each node through
    ``angle_delta * rotation_weights``. Radius changes first apply the supplied
    radial velocity. Only source-free vacuum air may deform non-rigidly. The
    connectivity remains fixed throughout the local angle window; other angles
    require an explicitly prepared geometry with the same polar topology.

    Every exterior vertex has constant prescribed Az (magnetic insulation),
    fixing the scalar gauge. A balanced winding is required for gauge-independent
    flux linkage. Dense native Cholesky is intended for modest planar profiles,
    not an unqualified industrial-scale sparse machine solver.
    """

    discretization: FiniteElementDiscretization
    cell_regions: Array
    regions: tuple[LinearMagneticRegion, ...]
    free_nodes: Array
    boundary_nodes: Array
    rotation_weights: Array
    radial_velocity: Array
    airgap_cells: Array
    contour_edges: Array
    contour_cells: Array
    reference_angle: float = eqx.field(static=True)
    reference_radius: float = eqx.field(static=True)
    radius_bounds: tuple[float, float] = eqx.field(static=True)
    angle_window: float = eqx.field(static=True)
    axial_length: float = eqx.field(static=True)
    winding_count: int = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        cell_regions: ArrayLike,
        regions: Sequence[LinearMagneticRegion],
        *,
        rotation_weights: ArrayLike,
        radial_velocity: ArrayLike,
        airgap_cells: ArrayLike,
        reference_radius: float,
        radius_bounds: tuple[float, float],
        reference_angle: float = 0.0,
        angle_window: float = 0.05,
        axial_length: float = 0.1,
        contour_edges: ArrayLike | Sequence[Sequence[int]] = (),
        contour_cells: ArrayLike | Sequence[Sequence[int]] = (),
    ):
        if (
            not isinstance(mesh, CellMesh)
            or mesh.ambient_dimension != 2
            or len(mesh.blocks) != 1
            or mesh.blocks[0].cell_kind != "triangle"
        ):
            raise ValueError("A planar machine requires one 2D triangle mesh block.")
        points = np.asarray(mesh.coordinates)
        cells = np.asarray(mesh.blocks[0].vertices, dtype=np.int32)
        vertices = points[cells]
        first, second = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        if np.any(~np.isfinite(determinant) | (determinant <= 0.0)):
            raise ValueError(
                "Machine triangles must be nondegenerate and counterclockwise."
            )
        ids = np.asarray(cell_regions)
        region_values = tuple(regions)
        if not region_values or any(
            not isinstance(r, LinearMagneticRegion) for r in region_values
        ):
            raise ValueError("Machine regions must be nonempty linear magnetic regions.")
        if len({r.name for r in region_values}) != len(region_values):
            raise ValueError("Machine region names must be unique.")
        if (
            ids.shape != (len(cells),)
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any((ids < 0) | (ids >= len(region_values)))
        ):
            raise ValueError("Every triangle needs one valid integer region index.")
        counts = {r.winding_turn_density.size for r in region_values}
        if len(counts) != 1:
            raise ValueError("Every region must declare the same winding count.")
        weights = np.asarray(rotation_weights, dtype=float)
        radial = np.asarray(radial_velocity, dtype=float)
        air = np.asarray(airgap_cells)
        if weights.shape != (len(points),) or np.any(
            ~np.isfinite(weights) | (weights < 0) | (weights > 1)
        ):
            raise ValueError(
                "Rotation weights must have one finite value in [0, 1] per node."
            )
        if radial.shape != points.shape or not np.all(np.isfinite(radial)):
            raise ValueError("Radial velocity must have the mesh coordinate shape.")
        if air.shape != (len(cells),) or air.dtype != np.bool_ or not np.any(air):
            raise ValueError("A nonempty Boolean airgap cell mask is required.")
        mu = np.asarray([r.relative_permeability for r in region_values])[ids]
        br = np.stack([np.asarray(r.remanence) for r in region_values])[ids]
        winding = np.stack([np.asarray(r.winding_turn_density) for r in region_values])[
            ids
        ]
        rotor = np.asarray([r.rotating for r in region_values])[ids]
        if (
            np.any(mu[air] != 1.0)
            or np.any(br[air] != 0.0)
            or np.any(winding[air] != 0.0)
        ):
            raise ValueError("The torque airgap must be source-free vacuum air.")
        if np.any(weights[cells[rotor]] != 1.0):
            raise ValueError("Every rotating material cell must rotate rigidly.")
        stationary = ~air & ~rotor
        if np.any(weights[cells[stationary]] != 0.0) or np.any(
            radial[cells[stationary]] != 0.0
        ):
            raise ValueError("Stationary materials and windings must remain fixed.")
        if np.any((np.ptp(weights[cells], axis=1) != 0.0) & ~air):
            raise ValueError(
                "Only vacuum airgap cells may undergo differential rotation."
            )
        lengths = (
            float(reference_radius),
            *map(float, radius_bounds),
            float(angle_window),
            float(axial_length),
        )
        radius, lower, upper, window, length = lengths
        if not all(isfinite(v) for v in (*lengths, float(reference_angle))):
            raise ValueError("Machine geometric parameters must be finite.")
        if (
            not 0 < lower <= radius <= upper
            or lower == upper
            or window <= 0
            or length <= 0
        ):
            raise ValueError(
                "Machine radius bounds, angle window, and axial length must be positive."
            )
        winding_integral = np.sum(0.5 * determinant[:, None] * winding, axis=0)
        winding_scale = np.sum(0.5 * determinant[:, None] * np.abs(winding), axis=0)
        if np.any(np.abs(winding_integral) > 2e-6 * np.maximum(winding_scale, 1e-30)):
            raise ValueError("Each winding must have balanced +z and -z signed turns.")
        # Reject isolated/disconnected components: a single exterior gauge must
        # not conceal extra nullspaces or an unrelated overlapping component.
        adjacency = [set() for _ in points]
        for a, b, c in cells:
            adjacency[a].update((int(b), int(c)))
            adjacency[b].update((int(a), int(c)))
            adjacency[c].update((int(a), int(b)))
        reached, pending = {0}, [0]
        while pending:
            for neighbour in adjacency[pending.pop()]:
                if neighbour not in reached:
                    reached.add(neighbour)
                    pending.append(neighbour)
        if len(reached) != len(points):
            raise ValueError("Machine mesh must be connected without unused vertices.")
        prepared = FiniteElementPlan(
            mesh, FiniteElementFieldSpec("Az", lagrange_element("triangle", 1))
        ).prepare()
        boundary = np.asarray(prepared.dof_maps[0].boundary_dof_mask, dtype=bool)
        if not np.any(boundary) or np.all(boundary):
            raise ValueError(
                "Machine mesh needs exterior gauge and interior field degrees of freedom."
            )
        if np.any(weights[boundary] != 0.0) or np.any(radial[boundary] != 0.0):
            raise ValueError("The exterior magnetic boundary must remain stationary.")
        raw_edges = np.asarray(contour_edges)
        raw_neighbours = np.asarray(contour_cells)
        if raw_edges.size and (
            raw_edges.ndim != 2
            or raw_edges.shape[1] != 2
            or not np.issubdtype(raw_edges.dtype, np.integer)
        ):
            raise ValueError("Air contour edges must be integer vertex pairs.")
        if raw_neighbours.size and (
            raw_neighbours.ndim != 2
            or raw_neighbours.shape[1] != 2
            or not np.issubdtype(raw_neighbours.dtype, np.integer)
        ):
            raise ValueError("Air contour neighbours must be integer cell pairs.")
        edges = raw_edges.astype(np.int32, copy=False).reshape((-1, 2))
        neighbours = raw_neighbours.astype(np.int32, copy=False).reshape((-1, 2))
        if edges.shape != neighbours.shape:
            raise ValueError("Each air contour edge requires its two adjacent cells.")
        if edges.size:
            if len(edges) < 3 or len(np.unique(edges[:, 0])) != len(edges):
                raise ValueError("The air stress contour must be one simple polygon.")
            if np.any((edges < 0) | (edges >= len(points))) or np.any(
                (neighbours < 0) | (neighbours >= len(cells))
            ):
                raise ValueError("Air contour indices are out of range.")
            if not np.all(air[neighbours]) or np.any(
                neighbours[:, 0] == neighbours[:, 1]
            ):
                raise ValueError(
                    "Both distinct sides of a stress contour must be vacuum air."
                )
            if not np.array_equal(edges[:, 1], np.roll(edges[:, 0], -1)):
                raise ValueError(
                    "The air stress contour must form an ordered closed loop."
                )
            contour_points = points[edges[:, 0]]
            next_points = np.roll(contour_points, -1, axis=0)
            signed_area = np.sum(
                contour_points[:, 0] * next_points[:, 1]
                - contour_points[:, 1] * next_points[:, 0]
            )
            if not signed_area > 0.0:
                raise ValueError("The air stress contour must be counterclockwise.")
            edge_adjacency: dict[tuple[int, int], list[int]] = {}
            for cell_index, triangle in enumerate(cells):
                for first, second in (
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0]),
                ):
                    first_ = int(first)
                    second_ = int(second)
                    key = (min(first_, second_), max(first_, second_))
                    edge_adjacency.setdefault(key, []).append(cell_index)
            for edge, adjacent in zip(edges, neighbours, strict=True):
                actual = edge_adjacency.get(tuple(sorted(map(int, edge))), ())
                if len(actual) != 2 or set(actual) != set(map(int, adjacent)):
                    raise ValueError(
                        "Stress contour neighbours must be the edge's two actual cells."
                    )
        self.discretization = prepared
        self.cell_regions = jnp.asarray(ids, dtype=jnp.int32)
        self.regions = region_values
        self.free_nodes = jnp.asarray(np.flatnonzero(~boundary), dtype=jnp.int32)
        self.boundary_nodes = jnp.asarray(np.flatnonzero(boundary), dtype=jnp.int32)
        self.rotation_weights = jnp.asarray(weights)
        self.radial_velocity = jnp.asarray(radial)
        self.airgap_cells = jnp.asarray(air)
        self.contour_edges = jnp.asarray(edges)
        self.contour_cells = jnp.asarray(neighbours)
        self.reference_angle = float(reference_angle)
        self.reference_radius = radius
        self.radius_bounds = (lower, upper)
        self.angle_window = window
        self.axial_length = length
        self.winding_count = counts.pop()

    def coordinates(self, rotor_radius: ArrayLike, angle_delta: ArrayLike = 0.0) -> Array:
        """Realize a bounded radius and local rotation, rejecting inverted cells."""
        radius = jnp.asarray(rotor_radius)
        delta = jnp.asarray(angle_delta)
        if radius.shape != () or delta.shape != ():
            raise ValueError("Rotor radius and local angle must be scalars.")
        radius = eqx.error_if(
            radius,
            ~jnp.isfinite(radius)
            | (radius < self.radius_bounds[0])
            | (radius > self.radius_bounds[1]),
            "Rotor radius is outside the topology-preserving bounds.",
        )
        delta = eqx.error_if(
            delta,
            ~jnp.isfinite(delta) | (jnp.abs(delta) > self.angle_window),
            "Angle leaves this topology's local window; prepare the requested angle instead.",
        )
        base = (
            self.discretization.mesh.coordinates
            + (radius - self.reference_radius) * self.radial_velocity
        )
        angle = delta * self.rotation_weights
        cosine, sine = jnp.cos(angle), jnp.sin(angle)
        points = jnp.stack(
            (
                cosine * base[:, 0] - sine * base[:, 1],
                sine * base[:, 0] + cosine * base[:, 1],
            ),
            axis=-1,
        )
        vertices = points[self.discretization.mesh.blocks[0].vertices]
        first, second = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        determinant = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        return eqx.error_if(
            points,
            jnp.any(~jnp.isfinite(determinant) | (determinant <= 0.0)),
            "Rotor-angle geometry inverted or collapsed a machine triangle.",
        )


def polar_machine(
    angle: float = 0.0,
    *,
    sectors: int = 32,
    rotor_layers: int = 2,
    airgap_layers: int = 3,
    winding_layers: int = 2,
    stator_layers: int = 2,
    rotor_radius: float = 0.03,
    rotor_radius_bounds: tuple[float, float] = (0.024, 0.033),
    stator_inner_radius: float = 0.04,
    winding_outer_radius: float = 0.05,
    outer_radius: float = 0.065,
    axial_length: float = 0.1,
    rotor_relative_permeability: float = 20.0,
    stator_relative_permeability: float = 100.0,
    remanence: float = 0.8,
    turns: float = 100.0,
    salient: bool = False,
) -> PlanarMachine:
    """Build a two-pole rotating disk/reluctance-rotor FEM benchmark.

    Two distributed sinusoidal stationary winding axes (cos(phi), sin(phi))
    represent balanced closed windings. The rotor is uniformly magnetized +x in
    its own frame; ``salient=True`` replaces the quadrature-axis rotor sectors
    with air, giving a genuine permeability-driven reluctance torque.

    At arbitrary angle, cyclic rotor material indexing reduces mesh twist while
    preserving identical triangle connectivity at every sample. The airgap
    coordinates remain inside a certified local deformation window. The mesh is
    polygonal; refining sectors and radial layers resolves its geometry.
    """
    n, nr, ng, nw, ns = map(
        int, (sectors, rotor_layers, airgap_layers, winding_layers, stator_layers)
    )
    if (
        (n, nr, ng, nw, ns)
        != (sectors, rotor_layers, airgap_layers, winding_layers, stator_layers)
        or n < 8
        or n % 4
        or nr < 1
        or ng < 2
        or nw < 1
        or ns < 1
    ):
        raise ValueError(
            "Use sectors divisible by four (at least eight), positive material "
            "layers, and at least two airgap layers."
        )
    r, rs, rw, ro = map(
        float,
        (rotor_radius, stator_inner_radius, winding_outer_radius, outer_radius),
    )
    low, high = map(float, rotor_radius_bounds)
    angle_value, turns_value, remanence_value = map(float, (angle, turns, remanence))
    values = (r, rs, rw, ro, low, high, angle_value, turns_value, remanence_value)
    if not all(isfinite(value) for value in values) or not (
        0 < low <= r <= high < rs < rw < ro
    ):
        raise ValueError(
            "Rotor bounds must preserve a strictly positive airgap and ordered "
            "machine radii."
        )
    if turns_value <= 0 or remanence_value < 0:
        raise ValueError("Turns must be positive and remanence nonnegative.")
    pitch = 2 * np.pi / n
    # Adjacent air rings differ by at most 0.7*pitch/ng, including the local
    # angle window. These two sine-area bounds cover both triangle diagonals,
    # either twist sign, and the thinnest ring at the largest allowed radius.
    half_pitch = 0.5 * pitch
    required_ratio = max(
        1.0 / np.cos(half_pitch),
        np.cos(half_pitch) / np.cos(half_pitch + 0.7 * pitch / ng),
    )
    minimum_ring_width = (rs - high) / ng
    if rs / (rs - minimum_ring_width) <= required_ratio:
        raise ValueError(
            "Airgap layers are too thin for this sector count and rotor radius "
            "bound; refine angularly or widen the gap."
        )
    # Canonicalizing to one sector pitch keeps connectivity independent of the
    # prescribed angle. ``shift`` instead rotates the body-fixed salient region
    # assignment; the continuous PM direction remains ``reference_angle``.
    revolution_angle = float(np.remainder(angle_value, 2 * np.pi))
    whole_sector_shift = int(np.floor(revolution_angle / pitch + 0.5))
    shift = whole_sector_shift % n
    reduced = revolution_angle - whole_sector_shift * pitch
    phi = np.arange(n) * pitch
    radii = [r * (i + 1) / nr for i in range(nr)]
    derivatives = [(i + 1) / nr for i in range(nr)]
    weights = [1.0] * nr
    ring_angles = [phi + reduced] * nr
    for layer in range(1, ng + 1):
        fraction = layer / ng
        radii.append(r + fraction * (rs - r))
        derivatives.append(1.0 - fraction)
        weights.append(1.0 - fraction)
        ring_angles.append(phi + reduced * (1.0 - fraction))
    for inner, outer, count in ((rs, rw, nw), (rw, ro, ns)):
        for layer in range(1, count + 1):
            radii.append(inner + (outer - inner) * layer / count)
            derivatives.append(0.0)
            weights.append(0.0)
            ring_angles.append(phi)
    points, radial, rotation = [[0.0, 0.0]], [[0.0, 0.0]], [1.0]
    for radius, derivative, weight, angles in zip(
        radii, derivatives, weights, ring_angles, strict=True
    ):
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        points.extend(radius * directions)
        radial.extend(derivative * directions)
        rotation.extend([weight] * n)
    zero = (0.0, 0.0)
    regions = [
        LinearMagneticRegion(
            "rotor",
            rotor_relative_permeability,
            remanence=(remanence_value, 0.0),
            winding_turn_density=zero,
            rotating=True,
        ),
        LinearMagneticRegion("rotor-air", winding_turn_density=zero, rotating=True),
        LinearMagneticRegion("airgap", winding_turn_density=zero),
        LinearMagneticRegion(
            "stator", stator_relative_permeability, winding_turn_density=zero
        ),
    ]
    # Each winding's positive half carries exactly `turns` integrated turns.
    mid_phi = phi + 0.5 * pitch
    sector_area = 0.5 * np.sin(pitch) * (rw * rw - rs * rs)
    densities = np.stack((np.cos(mid_phi), np.sin(mid_phi)), axis=-1)
    densities *= turns_value / (sector_area * np.sum(np.maximum(densities, 0.0), axis=0))
    regions.extend(
        LinearMagneticRegion(f"winding-{i}", winding_turn_density=density)
        for i, density in enumerate(densities)
    )
    cells, ids, air = [], [], []

    def rotor_region(sector):
        body_sector = (sector - shift) % n
        return 1 if salient and abs(np.cos(mid_phi[body_sector])) < np.sqrt(0.5) else 0

    for i in range(n):
        cells.append((0, 1 + i, 1 + (i + 1) % n))
        ids.append(rotor_region(i))
        air.append(False)
    for ring in range(len(radii) - 1):
        for i in range(n):
            a, b = 1 + ring * n + i, 1 + ring * n + (i + 1) % n
            c = 1 + (ring + 1) * n + i
            d = 1 + (ring + 1) * n + (i + 1) % n
            cells.extend(((a, c, d), (a, d, b)))
            is_air = nr - 1 <= ring < nr + ng - 1
            is_winding = nr + ng - 1 <= ring < nr + ng + nw - 1
            if ring < nr - 1:
                region = rotor_region(i)
            elif is_air:
                region = 2
            elif is_winding:
                region = 4 + i
            else:
                region = 3
            ids.extend((region, region))
            air.extend((is_air, is_air))
    cells = np.asarray(cells, dtype=np.int32)
    # A middle-airgap polygon has vacuum elements on both sides. The Maxwell
    # contour check averages the two P1 traces rather than sharing energy AD.
    contour_ring = nr - 1 + max(1, ng // 2)
    edges = np.asarray(
        [
            (
                1 + contour_ring * n + i,
                1 + contour_ring * n + (i + 1) % n,
            )
            for i in range(n)
        ],
        dtype=np.int32,
    )
    adjacency = {}
    for cell_index, triangle in enumerate(cells):
        for a, b in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            adjacency.setdefault(tuple(sorted((int(a), int(b)))), []).append(cell_index)
    contour_cells = np.asarray(
        [adjacency[tuple(sorted(edge))] for edge in edges], dtype=np.int32
    )
    return PlanarMachine(
        CellMesh.from_triangles(np.asarray(points), cells),
        np.asarray(ids, dtype=np.int32),
        regions,
        rotation_weights=np.asarray(rotation),
        radial_velocity=np.asarray(radial),
        airgap_cells=np.asarray(air),
        reference_radius=r,
        radius_bounds=(low, high),
        reference_angle=angle_value,
        angle_window=0.2 * pitch,
        axial_length=axial_length,
        contour_edges=edges,
        contour_cells=contour_cells,
    )
