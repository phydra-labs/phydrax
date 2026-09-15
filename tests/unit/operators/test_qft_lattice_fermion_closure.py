#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization._cell_complex import polygonal_cell_complex
from phydrax.discretization._lattice_boundary import (
    CheckerboardEntityLayout,
    LatticeBoundaryPhasePlan,
)
from phydrax.discretization._oriented_path import prepare_cell_boundary_paths
from phydrax.discretization._topology import TensorTopology
from phydrax.graph._gauge_transport import GaugeCovariantShiftPlan, GaugeStaplePlan
from phydrax.graph._matrix_gauge import gauge_transform_links, MatrixGaugeLinkSpace
from phydrax.linalg._policies import FailurePolicy
from phydrax.linalg._rational_functions import RationalFunctionPolicy
from phydrax.linalg._shifted import ShiftedSolvePolicy
from phydrax.metrix._complex_matrix_manifold import SpecialUnitaryGroup
from phydrax.metrix._gauge_representation import FundamentalGaugeRepresentation
from phydrax.operators.path_integral._lattice_fermion import (
    canonical_euclidean_gamma_matrices,
    CloverWilsonDiracOperator,
    FreeWilsonDiracOperator,
    OverlapDiracOperator,
    plan_even_odd_schur,
    prepare_even_odd_schur,
    RationalSignApproximation,
    WilsonDiracOperator,
)
from phydrax.operators.path_integral._smearing import stout_smear, StoutSmearingPlan


def _tensor_boundary(shape=(2, 2), phases=(1.0, 1.0), maximum_displacement=3):
    topology = TensorTopology(
        tuple(f"x{axis}" for axis in range(len(shape))),
        shape,
        periodic=(True,) * len(shape),
    )
    return LatticeBoundaryPhasePlan(
        topology,
        jnp.asarray(phases),
        maximum_displacement=maximum_displacement,
    )


def _periodic_square_gauge():
    boundary = _tensor_boundary()
    cell_topology = polygonal_cell_complex(
        None,
        jnp.asarray([[0, 1, 3, 2]], dtype=jnp.int32),
        4,
    )
    group = SpecialUnitaryGroup(2)
    link_space = MatrixGaugeLinkSpace(cell_topology, group)
    representation = FundamentalGaugeRepresentation(group)
    coordinates = np.stack(
        np.unravel_index(np.arange(4), boundary.topology.axis_sizes), axis=-1
    )
    forward_sites = np.empty((4, 2), dtype=np.int32)
    forward_edges = np.empty((4, 2), dtype=np.int32)
    forward_orientations = np.empty((4, 2), dtype=np.int32)
    tails = np.asarray(link_space.tail_vertices)
    heads = np.asarray(link_space.head_vertices)
    for site in range(4):
        for axis, size in enumerate(boundary.topology.axis_sizes):
            neighbor_coordinate = coordinates[site].copy()
            neighbor_coordinate[axis] = (neighbor_coordinate[axis] + 1) % size
            neighbor = int(
                np.ravel_multi_index(neighbor_coordinate, boundary.topology.axis_sizes)
            )
            forward_sites[site, axis] = neighbor
            direct = np.flatnonzero((tails == site) & (heads == neighbor))
            reverse = np.flatnonzero((tails == neighbor) & (heads == site))
            if direct.size:
                forward_edges[site, axis] = int(direct[0])
                forward_orientations[site, axis] = 1
            else:
                forward_edges[site, axis] = int(reverse[0])
                forward_orientations[site, axis] = -1
    transport = GaugeCovariantShiftPlan(
        link_space,
        representation,
        forward_sites,
        forward_edges,
        forward_orientations,
        boundary,
    )
    return boundary, cell_topology, link_space, representation, transport


def _dense_action(operator):
    basis = jnp.eye(operator.source.size, dtype=operator.source.dtype)
    columns = jax.vmap(
        lambda coordinates: operator.target.flatten(
            operator.mv(operator.source.unflatten(coordinates))
        )
    )(basis)
    return jnp.swapaxes(columns, -1, -2)


def test_free_wilson_dispersion_matches_lattice_symbol():
    boundary = _tensor_boundary(shape=(4, 4))
    operator = FreeWilsonDiracOperator(
        boundary,
        mass=0.3,
        color_components=1,
        dtype=jnp.complex64,
    )
    gamma, _ = canonical_euclidean_gamma_matrices(2)
    momentum = jnp.asarray((jnp.pi / 2.0, jnp.pi))
    coordinates = jnp.stack(
        jnp.meshgrid(jnp.arange(4), jnp.arange(4), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    plane_wave = jnp.exp(1j * (coordinates @ momentum))
    spinor = jnp.asarray((0.7 + 0.2j, -0.1 + 0.5j))
    field = (plane_wave[:, None, None] * spinor[None, :, None]).astype(
        operator.source.dtype
    )
    symbol = (0.3 + jnp.sum(1.0 - jnp.cos(momentum))) * jnp.eye(2) + 1j * jnp.sum(
        gamma * jnp.sin(momentum)[:, None, None], axis=0
    )
    expected = (plane_wave[:, None, None] * (symbol @ spinor)[:, None]).astype(
        operator.target.dtype
    )

    assert jnp.allclose(operator.mv(field), expected, atol=2e-6)


def test_wilson_covariance_and_gamma5_hermiticity():
    boundary, _, link_space, representation, transport = _periodic_square_gauge()
    links = link_space.identity().astype(jnp.complex64)
    operator = WilsonDiracOperator(
        boundary,
        representation,
        transport,
        links,
        mass=0.2,
        dtype=jnp.complex64,
    )
    field = (
        jnp.arange(operator.source.size, dtype=jnp.float32).reshape(operator.source.shape)
        + 1j
    ).astype(jnp.complex64)
    algebra_coordinates = 0.07 * jnp.arange(
        4 * link_space.group.algebra_shape[0], dtype=jnp.float32
    ).reshape((4, -1))
    vertices = jax.vmap(lambda value: link_space.group.exp(link_space.group.hat(value)))(
        algebra_coordinates
    )
    transformed_links = gauge_transform_links(link_space, links, vertices)
    transformed_field = representation.apply(vertices, field)
    transformed_operator = operator.with_links(transformed_links)

    assert jnp.allclose(
        transformed_operator.mv(transformed_field),
        representation.apply(vertices, operator.mv(field)),
        atol=3e-5,
    )
    assert jnp.allclose(
        operator.adjoint_mv(field), operator.gamma5_conjugate_mv(field), atol=3e-5
    )


def test_even_odd_schur_reconstructs_full_solution():
    boundary = _tensor_boundary(shape=(2, 2))
    operator = FreeWilsonDiracOperator(
        boundary,
        mass=0.6,
        color_components=1,
        dtype=jnp.complex64,
    )
    layout = CheckerboardEntityLayout(boundary.topology)
    plan = plan_even_odd_schur(operator, layout)
    prepared = prepare_even_odd_schur(operator, plan)
    solution = (
        jnp.arange(operator.source.size, dtype=jnp.float32).reshape(operator.source.shape)
        / 7.0
        + 0.3j
    ).astype(jnp.complex64)
    rhs = operator.mv(solution)
    reduced_solution = layout.gather(solution, plan.reduced_parity)

    assert jnp.allclose(
        prepared.schur.mv(reduced_solution), prepared.reduce_rhs(rhs), atol=2e-5
    )
    assert jnp.allclose(prepared.reconstruct(reduced_solution, rhs), solution, atol=2e-5)


def test_clover_inverse_and_adjoint_match_small_dense_reference():
    boundary, _, link_space, representation, transport = _periodic_square_gauge()
    wilson = WilsonDiracOperator(
        boundary,
        representation,
        transport,
        link_space.identity().astype(jnp.complex64),
        mass=0.5,
        dtype=jnp.complex64,
    )
    block_size = wilson.spin_components * wilson.color_components
    clover_matrices = 0.08 * jnp.broadcast_to(
        jnp.eye(block_size, dtype=jnp.complex64),
        (wilson.site_count, block_size, block_size),
    )
    clover = CloverWilsonDiracOperator(
        wilson,
        clover_matrices.reshape(
            (
                wilson.site_count,
                wilson.spin_components,
                wilson.color_components,
                wilson.spin_components,
                wilson.color_components,
            )
        ),
    )
    dense = _dense_action(clover)
    vector = (jnp.arange(clover.source.size, dtype=jnp.float32) / 13.0 + 0.2j).astype(
        jnp.complex64
    )
    field = clover.source.unflatten(vector)

    assert jnp.all(clover.inverse_evidence.finite)
    assert jnp.allclose(
        clover.diagonal_mv(clover.diagonal_inverse_mv(field)),
        field,
        atol=2e-5,
    )
    assert jnp.allclose(
        clover.source.flatten(clover.adjoint_mv(field)),
        jnp.conj(dense.T) @ vector,
        atol=3e-5,
    )


def test_stout_smearing_preserves_group_and_is_differentiable():
    _, cell_topology, link_space, _, _ = _periodic_square_gauge()
    boundaries = prepare_cell_boundary_paths(cell_topology)
    staples = GaugeStaplePlan(link_space, boundaries)
    plan = StoutSmearingPlan(staples, rho=0.11, iterations=2)
    algebra_coordinates = 0.03 * jnp.arange(
        link_space.num_edges * link_space.group.algebra_shape[0], dtype=jnp.float32
    ).reshape((link_space.num_edges, -1))
    links = jax.vmap(lambda value: link_space.group.exp(link_space.group.hat(value)))(
        algebra_coordinates
    )
    result = stout_smear(plan, links)
    derivative = jax.grad(
        lambda scale: jnp.real(
            jnp.sum(
                stout_smear(
                    plan,
                    jax.vmap(
                        lambda value: link_space.group.exp(
                            link_space.group.hat(scale * value)
                        )
                    )(algebra_coordinates),
                ).links
            )
        )
    )(jnp.asarray(1.0))

    assert result.evidence.successful
    assert link_space.contains(result.links)
    assert jnp.isfinite(derivative)


def test_overlap_rational_action_matches_small_dense_equivalence():
    boundary = _tensor_boundary(shape=(2, 2))
    kernel = FreeWilsonDiracOperator(
        boundary,
        mass=-0.7,
        color_components=1,
        dtype=jnp.complex64,
    )
    approximation = RationalSignApproximation(
        jnp.asarray((0.05, 0.4, 2.0)),
        jnp.asarray((0.18, 0.42, 0.9)),
        constant=0.04,
        spectral_lower_bound=0.01,
        spectral_upper_bound=25.0,
        maximum_error=0.2,
    )
    policy = RationalFunctionPolicy(
        shifted=ShiftedSolvePolicy(
            "lanczos",
            max_dimension=kernel.source.size,
            relative_tolerance=2e-5,
            absolute_tolerance=2e-6,
        ),
        failure=FailurePolicy("status"),
    )
    overlap = OverlapDiracOperator(
        kernel,
        approximation,
        fermion_mass=0.1,
        rational_policy=policy,
    )
    vector = (jnp.arange(kernel.source.size, dtype=jnp.float32) / 9.0 + 0.15j).astype(
        jnp.complex64
    )
    field = kernel.source.unflatten(vector)
    result = overlap.action_with_evidence(field)
    hermitian = _dense_action(overlap.hermitian_kernel)
    squared = hermitian @ hermitian
    identity = jnp.eye(kernel.source.size, dtype=kernel.source.dtype)
    inverse_root = approximation.constant * identity
    for shift, weight in zip(approximation.shifts, approximation.weights, strict=True):
        inverse_root = inverse_root + weight * jnp.linalg.inv(squared + shift * identity)
    sign = hermitian @ inverse_root
    gamma5 = jnp.kron(jnp.eye(boundary.site_count), kernel.gamma5)
    expected_matrix = 0.55 * identity + 0.45 * gamma5 @ sign

    assert result.evidence.successful
    assert jnp.allclose(
        kernel.target.flatten(result.value),
        expected_matrix @ vector,
        atol=2e-3,
        rtol=2e-3,
    )
