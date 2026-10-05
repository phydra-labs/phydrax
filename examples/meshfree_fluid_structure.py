#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One monolithic quasi-static fluid--structure step between two meshfree regions.

A compressible Newtonian fluid in slow (Stokes) flow fills ``[0, 1] x [0, 1]``
and an elastic solid ``[1, 2] x [0, 1]``; they share the interface ``x = 1``.
The fluid stress ``2 mu eps(v) + lambda div(v) I`` has the same isotropic block
operator as linear elasticity. One implicit step of length ``dt`` is solved
monolithically in the velocity of the fluid ``v`` and the displacement rate of
the solid ``w = (d_new - d_old) / dt``, whose stress is ``C : eps(dt w)``. The
interface laws are the kinematic condition ``v = w`` and the dynamic condition
``sigma_f n_f + sigma_s n_s = 0`` for every Cartesian component; a
``VectorTransmissionLaw`` imposes both through one mortar per component whose
multiplier is the fluid's outward traction.

Both regions are collocated point clouds (``PointBlockSystemPlan``) whose
interface is authorized by polygon charts. Explicit traction ghost layers keep
the PDE at interface cloud points and impose traction on ghost rows; strain
uses the solved cloud and ghost values in that same derivative family.
The step reports the integrated
interface force, the power delivered by the fluid and received by the solid as
dual pairings of the traction covector with each side's trace (their sum is the
power lost by the interface), and the solid's elastic power ``int eps(w) : C
eps(dt w)`` as an independent total balance of the work received.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization import (
    FacetTraceRule,
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    prepare_point_cloud_field_reconstruction,
    PreparedPointCloudDiscretization,
)
from phydrax.discretization.meshfree import (
    isotropic_elasticity_coefficients,
    LocalStencilPolicy,
    PointBlockSystemPlan,
    PointBoundaryCharts,
    PointGhostLayerPlan,
)
from phydrax.solver.coupling import (
    CoupledProblemPlan,
    CoupledSolution,
    InterfaceBinding,
    InterfaceEndpoint,
    InterfaceSource,
    MeshfreeBoundaryTrace,
    MeshfreeTraceComponent,
    MortarImposition,
    MortarMultiplier,
    PairedSupportAttachment,
    prepare_coupled_problem,
    PreparedCoupledProblem,
    solve_coupled_problem,
    VectorInterfaceResultants,
    VectorTransmissionCertificate,
    VectorTransmissionLaw,
    VectorTransmissionSide,
)


jax.config.update("jax_enable_x64", True)

INTERFACE_X = 1.0
VISCOSITY = 1.0
BULK_VISCOSITY = 0.5
SHEAR_MODULUS = 40.0
LAME_LAMBDA = 60.0
STEP = 0.01
INFLOW = 1.0
FIELDS = ("x", "y")


def inflow(y: np.ndarray, /) -> np.ndarray:
    return 4.0 * INFLOW * y * (1.0 - y)


def region(
    x0: float, resolution: int, side: str, /
) -> tuple[PreparedPointCloudDiscretization, PointBoundaryCharts, np.ndarray, np.ndarray]:
    """Grid cloud on ``[x0, x0 + 1] x [0, 1]`` and charts of its ``x = 1`` side."""
    h = 1.0 / resolution
    axis = np.linspace(0.0, 1.0, resolution + 1)
    x, y = np.meshgrid(x0 + axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    weight = np.where((axis > 0) & (axis < 1), h, 0.5 * h)
    volumes = np.outer(weight, weight).ravel()
    low, high = np.isclose(points[:, 0], x0), np.isclose(points[:, 0], x0 + 1.0)
    bottom, top = np.isclose(points[:, 1], 0.0), np.isclose(points[:, 1], 1.0)
    normals = np.zeros_like(points)
    normals[low], normals[high] = (-1.0, 0.0), (1.0, 0.0)
    normals[bottom], normals[top] = (0.0, -1.0), (0.0, 1.0)
    cloud = PointCloudPlan(
        points,
        volumes,
        boundary_mask=low | high | bottom | top,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
        neighbors=30,
    ).prepare()
    if side == "left":
        # Counter-clockwise: the interface edges run upward along x = 1.
        rising = np.stack((np.full(resolution, INTERFACE_X), axis[1:]), axis=1)
        vertices = np.concatenate(
            ([[0.0, 0.0], [INTERFACE_X, 0.0]], rising, [[0.0, 1.0]])
        )
        charts = tuple(range(1, 1 + resolution))
    else:
        falling = np.stack((np.full(resolution, INTERFACE_X), axis[::-1][:-1]), axis=1)
        vertices = np.concatenate(([[INTERFACE_X, 0.0], [2.0, 0.0], [2.0, 1.0]], falling))
        charts = tuple(range(3, 3 + resolution))
    atlas = phx.geometry.Polygon([tuple(v) for v in vertices]).compile().boundary_atlas
    authority = PointBoundaryCharts(
        cloud, atlas, FacetTraceRule("gauss-lobatto-legendre", points=2), charts=charts
    )
    interface = np.isclose(points[:, 0], INTERFACE_X) & ~bottom & ~top
    return cloud, authority, normals, np.flatnonzero(interface)


def block_component(
    name: str,
    x0: float,
    resolution: int,
    side: str,
    coefficients: Array,
    dirichlet_values: np.ndarray,
    /,
    *,
    source: np.ndarray | None = None,
) -> MeshfreeTraceComponent:
    """One region with Dirichlet ``dirichlet_values`` off the interface and body force ``source``."""
    cloud, charts, normals, interface = region(x0, resolution, side)
    count = cloud.state_shape[0]
    boundary = np.flatnonzero(np.asarray(cloud.plan.boundary_mask))
    dirichlet = np.setdiff1d(boundary, interface)
    measure = np.asarray(charts.measure)[interface]
    conditions: list[PointBoundaryCondition] = []
    for component, field in enumerate(FIELDS):
        conditions += [
            PointBoundaryCondition(
                "dirichlet",
                dirichlet,
                dirichlet_values[dirichlet, component],
                label=f"wall-{field}",
                component=component,
            ),
            PointBoundaryCondition(
                "neumann",
                interface,
                0.0,
                label=f"interface-{field}",
                component=component,
                normals=normals[interface],
                measure=measure,
            ),
        ]
    plan = PointBoundaryPlan(conditions, row_count=count, components=2)
    ghosts = PointGhostLayerPlan(plan).prepare(cloud)
    system = PointBlockSystemPlan(cloud, plan, components=FIELDS, ghosts=ghosts).prepare(
        coefficients
    )
    reconstruction = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=phx.geometry.Rectangle((x0 + 0.5, 0.5), (1.0, 1.0)).compile(),
        radius=3.5 / resolution,
        capacity=36,
    )
    return MeshfreeTraceComponent(
        system,
        reconstruction,
        tuple(
            MeshfreeBoundaryTrace(field, f"interface-{field}", charts) for field in FIELDS
        ),
        name=name,
        source=np.zeros((count, 2)) if source is None else source,
    )


def fluid(resolution: int, /) -> MeshfreeTraceComponent:
    points = region(0.0, resolution, "left")[0].points
    values = np.zeros((points.shape[0], 2))
    at_inlet = np.isclose(np.asarray(points)[:, 0], 0.0)
    values[at_inlet, 0] = inflow(np.asarray(points)[at_inlet, 1])
    viscous = isotropic_elasticity_coefficients(BULK_VISCOSITY, VISCOSITY, 2)
    return block_component("fluid", 0.0, resolution, "left", viscous, values)


def solid(resolution: int, /) -> MeshfreeTraceComponent:
    count = (resolution + 1) ** 2
    elastic = STEP * isotropic_elasticity_coefficients(LAME_LAMBDA, SHEAR_MODULUS, 2)
    return block_component(
        "solid", INTERFACE_X, resolution, "right", elastic, np.zeros((count, 2))
    )


def interface_binding(
    fluid_: MeshfreeTraceComponent, solid_: MeshfreeTraceComponent, /
) -> tuple[InterfaceBinding, phx.domain.SubdomainCover]:
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(plate, "x", (2, 1), cover_id="fsi")
    pairing = cover.pairings[0]
    witness = pairing.component.sample(phx.domain.PointSampling(8), key=jr.key(3))
    endpoints = tuple(
        InterfaceEndpoint(
            component.name,
            PairedSupportAttachment(cover, pairing.pairing_id, patch, witness),
            fields={field: component.field_space_id(field) for field in FIELDS},
        )
        for component, patch in (
            (fluid_, pairing.left_patch_id),
            (solid_, pairing.right_patch_id),
        )
    )
    binding = InterfaceBinding(
        "wetted",
        InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        endpoints,
    )
    return binding, cover


def prepare_step(
    fluid_resolution: int, solid_resolution: int, /
) -> PreparedCoupledProblem:
    return coupled_step(fluid(fluid_resolution), solid(solid_resolution))


def coupled_step(
    fluid_: MeshfreeTraceComponent, solid_: MeshfreeTraceComponent, /
) -> PreparedCoupledProblem:
    """The monolithic step of two prepared regions through the wetted interface."""
    binding, cover = interface_binding(fluid_, solid_)
    law = VectorTransmissionLaw(
        "wetted-interface",
        binding,
        (
            VectorTransmissionSide(
                "fluid", "fluid", FIELDS, fluid_.boundary_domain("x", "interface-x")
            ),
            VectorTransmissionSide(
                "solid", "solid", FIELDS, solid_.boundary_domain("x", "interface-x")
            ),
        ),
        MortarImposition(MortarMultiplier("side-trace", side="solid")),
    )
    plan = CoupledProblemPlan(
        "meshfree-fsi-step",
        components=(fluid_, solid_),
        bindings=(binding,),
        laws=(law,),
    )
    return prepare_coupled_problem(plan, interface_owners=(cover,))


def resultants(
    prepared: PreparedCoupledProblem, solution: CoupledSolution, /
) -> VectorInterfaceResultants:
    law = next(law for law in prepared.laws if law.law_id == "wetted-interface")
    certificate = law.certificate
    if not isinstance(certificate, VectorTransmissionCertificate):
        raise TypeError("The interface law was not lowered as a vector transmission.")
    return certificate.resultants(
        dict(solution.fields), solution.law_state("wetted-interface")
    )


def elastic_power(
    prepared: PreparedCoupledProblem, solution: CoupledSolution, /
) -> float:
    """``sum_i V_i eps(w)_i : C eps(dt w)_i`` from the solid's own derivative rows."""
    component = next(c for c in prepared.components if c.name == "solid")
    if not isinstance(component, MeshfreeTraceComponent):
        raise TypeError("The solid must be a meshfree trace component.")
    cloud = component.owner
    ghosts = component.ghost_layer
    if ghosts is None:
        raise ValueError("The solid must retain its prepared traction ghost layer.")
    rate = jnp.stack(
        [
            jnp.concatenate(
                (
                    solution.field("solid", field),
                    solution.field("solid", f"{field}-ghost"),
                )
            )
            for field in FIELDS
        ],
        axis=1,
    )
    gradient = jnp.stack(
        [
            ghosts.family.apply(rate, (1, 0))[: ghosts.cloud_count],
            ghosts.family.apply(rate, (0, 1))[: ghosts.cloud_count],
        ],
        axis=-1,
    )
    strain = 0.5 * (gradient + jnp.swapaxes(gradient, -1, -2))
    trace = jnp.trace(strain, axis1=-2, axis2=-1)
    stress = STEP * (
        2.0 * SHEAR_MODULUS * strain + LAME_LAMBDA * trace[:, None, None] * jnp.eye(2)
    )
    return float(
        jnp.sum(cloud.quadrature_weights * jnp.sum(strain * stress, axis=(-2, -1)))
    )


def run(
    levels: tuple[tuple[int, int], ...] = ((8, 6), (16, 12)), /
) -> list[dict[str, float]]:
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=64_000_000, max_bytes=1024 * 1024 * 1024
        ),
    )
    records: list[dict[str, float]] = []
    for fluid_resolution, solid_resolution in levels:
        start = time.perf_counter()
        prepared = prepare_step(fluid_resolution, solid_resolution)
        solution = solve_coupled_problem(prepared, policy=policy)
        interface = solution.interface("wetted-interface")
        if not bool(solution.accepted):
            raise RuntimeError(
                "FSI step refused: "
                f"{dict(zip(interface.names, np.asarray(interface.values)))}"
            )
        result = resultants(prepared, solution)
        stored = elastic_power(prepared, solution)
        record = {
            "force_x": float(result.force[0]),
            "force_y": float(result.force[1]),
            "fluid_power": float(result.minus_power),
            "solid_power": float(result.plus_power),
            "elastic_power": stored,
        }
        records.append(record)
        print(
            f"fluid n={fluid_resolution}, solid n={solid_resolution} "
            f"({time.perf_counter() - start:.1f} s): force ({record['force_x']:.5f}, "
            f"{record['force_y']:.5f}), power delivered {-record['fluid_power']:.6e}, "
            f"received {record['solid_power']:.6e}, elastic {stored:.6e}"
        )
        for name, gated, value, scale in zip(
            interface.names,
            interface.gated,
            np.asarray(interface.values),
            np.asarray(interface.scales),
            strict=True,
        ):
            print(f"  {name:<18} {value:.3e} (scale {scale:.3e}, gated {gated})")
    return records


if __name__ == "__main__":
    run()
