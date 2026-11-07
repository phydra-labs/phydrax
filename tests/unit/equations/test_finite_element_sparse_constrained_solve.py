#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sparse elimination of strong Dirichlet constraints and its native solve route."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._cell_mesh import CellBlock, CellMesh


la = phx.linalg
_SPD = la.OperatorProperties(
    self_adjoint=True,
    positive_definite=True,
    evidence={"positive_definite": "asserted"},
)


def _triangles(count: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, count + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    index = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    lower = index[:-1, :-1].ravel()
    cells = np.concatenate(
        (
            np.stack((lower, lower + count + 1, lower + count + 2), axis=-1),
            np.stack((lower, lower + count + 2, lower + 1), axis=-1),
        )
    )
    return points, cells.astype(np.int32)


def _mesh(points: np.ndarray, cells: np.ndarray, kind: str) -> Any:
    if kind == "triangle":
        return CellMesh.from_triangles(jnp.asarray(points), jnp.asarray(cells))
    return CellMesh(
        points,
        (CellBlock("cells", kind, cells, global_ids=tuple(range(cells.shape[0]))),),
    )


def _annulus(parameters: np.ndarray) -> np.ndarray:
    radius = 1.0 + parameters[..., 0]
    angle = 0.5 * np.pi * parameters[..., 1]
    return np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=-1)


def _curved_p2_space(kind: str, component_shape: tuple[int, ...] = ()) -> Any:
    """P2 field on a quarter annulus whose P2 coordinate map interpolates the chart."""
    points, cells = _triangles(3)
    nodes = phx.discretization.FiniteElementDofMap(
        _mesh(points, cells, kind), (phx.discretization.lagrange_element(kind, 2),)
    )
    mesh = _mesh(_annulus(points), cells, kind)
    geometry = phx.discretization.CellGeometrySpec(
        {mesh.blocks[0].name: phx.discretization.lagrange_element(kind, 2)},
        {mesh.blocks[0].name: np.asarray(nodes.cell_dofs[0])},
        _annulus(np.asarray(nodes.dof_coordinates)),
    )
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            phx.discretization.lagrange_element(kind, 2),
            component_shape=component_shape,
        ),
        coordinate_spec=geometry,
    ).prepare()


def _dirichlet_laplace(space: Any, values: Any, *, properties: Any = _SPD) -> Any:
    return phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "dirichlet-laplace",
            "u",
            (phx.equations.DiffusionAction("u", 1.0),),
            properties=properties,
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=values,
    )


_NATIVE_CHOLESKY = la.LinearSolvePolicy(
    la.PCG(),
    tolerance=la.TolerancePolicy(relative=1.0e-14, absolute=0.0, max_steps=10),
    preconditioning=la.PreconditioningPolicy(
        la.SparseFactorizationPreconditionerBuilder(
            la.SparseFactorizationPolicy("cholesky", ordering="nested-dissection")
        )
    ),
)


def _factorization(prepared: Any) -> Any:
    """The native sparse factor retained by a prepared preconditioned solve."""
    state = prepared.preconditioning_state
    assert state is not None
    action = state.action
    assert isinstance(action, la.SparseFactorizationPreconditioner)
    return action.factorization


def test_blocked_stiffness_assembly_keeps_cell_order_and_tail() -> None:
    points, cells = _triangles(12)  # 288 cells: one full block and one tail.
    space = phx.discretization.FiniteElementPlan(
        CellMesh.from_triangles(jnp.asarray(points), jnp.asarray(cells)),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 2)
        ),
    ).prepare()
    geometry = space.block_geometries[0][0]
    gradients = np.asarray(geometry.physical_gradients)
    local = np.einsum(
        "cq,cqid,cqjd->cij",
        np.asarray(geometry.physical_weights),
        gradients,
        gradients,
    )
    transforms = np.asarray(space.dof_maps[0].cell_transforms[0])
    expected = np.einsum("cai,cab,cbj->cij", transforms, local, transforms)
    first = np.asarray(space.stiffness.coefficients)
    second = np.asarray(space.stiffness.coefficients)
    np.testing.assert_allclose(first, expected.ravel(), rtol=2.0e-13, atol=2.0e-14)
    np.testing.assert_array_equal(first, second)


def test_curved_p2_dirichlet_laplace_reproduces_x_through_the_sparse_route() -> None:
    space = _curved_p2_space("triangle")
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    problem = _dirichlet_laplace(space, lambda value: value[..., 0])
    system, rhs = problem.linear_system(storage="sparse")
    action_system, action_rhs = problem.linear_system()

    operator = system.operator
    free = int(problem.state_space.size)
    assert isinstance(operator, phx.sparse.SparseCoordinateOperator)
    assert operator.source.compatible(problem.state_space)
    assert operator.properties.certifies("positive_definite")
    # The eliminated block is the action operator exactly; Dirichlet columns
    # live in the right-hand side through the lift.
    identity = np.eye(free)
    sparse_columns = np.stack([np.asarray(operator.mv(column)) for column in identity])
    action_columns = np.stack(
        [np.asarray(action_system.operator.mv(column)) for column in identity]
    )
    np.testing.assert_allclose(sparse_columns, action_columns, rtol=0.0, atol=1.0e-12)
    np.testing.assert_array_equal(np.asarray(rhs), np.asarray(action_rhs))
    csr = problem.to_scipy_csr()
    assert csr.shape == (free, free)
    assert csr.nnz == operator.sparse_storage().nnz < free * free

    prepared = la.prepare(system, _NATIVE_CHOLESKY)
    factorization = _factorization(prepared)
    assert int(factorization.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert float(factorization.diagnostics.minimum_pivot) > 0.0
    result = la.solve(prepared, rhs)
    assert bool(result.successful)
    assert float(result.diagnostics.relative_residual) < 1.0e-14
    assert int(result.diagnostics.iterations) <= 1
    np.testing.assert_allclose(
        np.asarray(problem.expand(result.value)), points[:, 0], rtol=0.0, atol=1.0e-13
    )
    condition = la.estimate_condition_number(operator, max_dimension=free)
    assert 1.0 < float(condition.lower_bound) <= float(condition.value) < 1.0e4

    # One factorization serves every lift: u = y and u = x - 2y share it.
    context = phx.equations.FiniteElementExecutionContext
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    exact = (points[:, 1], points[:, 0] - 2.0 * points[:, 1])
    contexts = tuple(
        context(space.default_runtime, lift=constraint.lift(jnp.asarray(value)))
        for value in exact
    )
    columns = jnp.stack(
        tuple(problem.right_hand_side(item) for item in contexts), axis=-1
    )
    many = la.solve_many(prepared, columns)
    assert bool(jnp.all(many.successful))
    for index, (item, value) in enumerate(zip(contexts, exact, strict=True)):
        np.testing.assert_allclose(
            np.asarray(problem.expand(many.value[:, index], item)),
            value,
            rtol=0.0,
            atol=1.0e-13,
        )


def _distorted_hex_pyramid_tet_mesh(count: int) -> CellMesh:
    """Conforming hex/pyramid/tet unit-cube mesh with non-affine hexes and pyramids.

    Even cubes are hexahedra; odd cubes split into six pyramids on their faces
    with apex at the cube center, except the faces on ``x = 0``, which split
    into two tetrahedra each. Interior grid vertices and cube centers move by a
    fixed pseudo-random offset, so hexahedra and pyramid bases are non-planar.
    """
    spacing = 1.0 / count
    keys: dict[tuple[int, ...], int] = {}
    points: list[np.ndarray] = []
    rng = np.random.default_rng(7)

    def vertex(doubled: tuple[int, int, int]) -> int:
        if doubled not in keys:
            keys[doubled] = len(points)
            point = np.asarray(doubled, dtype=np.float64) * 0.5 * spacing
            if all(0 < value < 2 * count for value in doubled):
                point = point + rng.uniform(-0.15, 0.15, 3) * spacing
            points.append(point)
        return keys[doubled]

    corners = (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    )  # fmt: skip
    hexes, pyramids, tetrahedra = [], [], []
    for i in range(count):
        for j in range(count):
            for k in range(count):
                a, b, c, d, e, f, g, h = (
                    vertex((2 * (i + di), 2 * (j + dj), 2 * (k + dk)))
                    for di, dj, dk in corners
                )
                if (i + j + k) % 2 == 0:
                    hexes.append((a, b, c, d, e, f, g, h))
                    continue
                center = vertex((2 * i + 1, 2 * j + 1, 2 * k + 1))
                faces = (
                    (a, b, c, d), (e, h, g, f), (a, e, f, b),
                    (b, f, g, c), (d, c, g, h), (a, d, h, e),
                )  # fmt: skip
                for index, (q0, q1, q2, q3) in enumerate(faces):
                    if index == 5 and i == 0:
                        tetrahedra.extend(((q0, q1, q2, center), (q0, q2, q3, center)))
                    else:
                        pyramids.append((q0, q1, q2, q3, center))
    blocks = []
    offset = 0
    for name, kind, cells in (
        ("hex", "hexahedron", hexes),
        ("pyramid", "pyramid", pyramids),
        ("tet", "tetrahedron", tetrahedra),
    ):
        blocks.append(
            CellBlock(
                name,
                kind,
                np.asarray(cells, dtype=np.int32),
                global_ids=tuple(range(offset, offset + len(cells))),
            )
        )
        offset += len(cells)
    return CellMesh(np.asarray(points), tuple(blocks))


def test_mixed_kind_sparse_system_matches_action_and_reproduces_x() -> None:
    """Mixed hex/pyramid/tet blocks: one discrete operator for both storages.

    Each block integrates with its own prepared reference rule in the sparse
    structural assembly and in the matrix-free action alike, so the eliminated
    sparse block equals the action exactly and both pass the affine patch test.
    """
    mesh = _distorted_hex_pyramid_tet_mesh(3)
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            {
                block.name: phx.discretization.lagrange_element(block.cell_kind, 1)
                for block in mesh.blocks
            },
        ),
    ).prepare()
    problem = _dirichlet_laplace(space, lambda value: value[..., 0])
    system, rhs = problem.linear_system(storage="sparse")
    action_system, action_rhs = problem.linear_system()
    free = int(problem.state_space.size)
    assert free > 0

    identity = np.eye(free)
    sparse_columns = np.stack(
        [np.asarray(system.operator.mv(column)) for column in identity]
    )
    action_columns = np.stack(
        [np.asarray(action_system.operator.mv(column)) for column in identity]
    )
    np.testing.assert_allclose(sparse_columns, action_columns, rtol=0.0, atol=1.0e-14)
    np.testing.assert_array_equal(np.asarray(rhs), np.asarray(action_rhs))

    # Affine patch test on both storages: u = x restricted to the free DOFs.
    constraint = phx.discretization.dirichlet_constraint(space, "u")
    selected = constraint.constraint_map.coordinate_selection()
    assert selected is not None
    exact = np.asarray(space.dof_maps[0].dof_coordinates)[:, 0]
    for operator in (system.operator, action_system.operator):
        np.testing.assert_allclose(
            np.asarray(operator.mv(jnp.asarray(exact[selected]))),
            np.asarray(rhs),
            rtol=0.0,
            atol=1.0e-14,
        )
    result = la.solve(la.prepare(system, _NATIVE_CHOLESKY), rhs)
    assert bool(result.successful)
    np.testing.assert_allclose(
        np.asarray(problem.expand(result.value)), exact, rtol=0.0, atol=1.0e-13
    )


def test_sparse_route_without_definiteness_claims_uses_native_lu() -> None:
    space = _curved_p2_space("simplex:2")
    problem = _dirichlet_laplace(
        space, lambda value: value[..., 0], properties=la.OperatorProperties()
    )
    system, rhs = problem.linear_system(storage="sparse")
    assert not system.operator.properties.self_adjoint
    policy = la.LinearSolvePolicy(
        la.GMRES(),
        tolerance=la.TolerancePolicy(relative=1.0e-14, absolute=0.0, max_steps=10),
        preconditioning=la.PreconditioningPolicy(
            la.SparseFactorizationPreconditionerBuilder(
                la.SparseFactorizationPolicy("lu", ordering="nested-dissection")
            )
        ),
    )
    result = la.solve(system, rhs, policy=policy)
    assert bool(result.successful)
    np.testing.assert_allclose(
        np.asarray(problem.expand(result.value)),
        np.asarray(space.dof_maps[0].dof_coordinates)[:, 0],
        rtol=0.0,
        atol=1.0e-13,
    )


def test_sparse_route_refuses_without_structural_sparse_storage() -> None:
    space = _curved_p2_space("triangle")
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "matrix-free", "u", (phx.equations.DiffusionAction("u", 1.0),)
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "u"),
        dirichlet_values=0.0,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free"
        ),
    )
    with pytest.raises(ValueError, match="structural sparse actions"):
        problem.linear_system(storage="sparse")
    with pytest.raises(ValueError, match="storage must be"):
        problem.linear_system(storage="dense")  # ty: ignore[invalid-argument-type]


def test_constraint_map_elimination_refuses_non_selection_prolongation() -> None:
    full = la.ArraySpace((3,))
    reduced = la.ArraySpace((2,))
    prolongation = phx.sparse.SparseCoordinateOperator(
        phx.sparse.EdgeRelation(
            np.asarray([0, 1, 1], dtype=np.int32),
            np.asarray([0, 1, 2], dtype=np.int32),
            source_size=2,
            target_size=3,
        ),
        jnp.ones((3,)),
        source=reduced,
        target=full,
    )
    hanging = la.ConstraintMap(full, reduced, prolongation)
    assert hanging.coordinate_selection() is None
    with pytest.raises(la.LinearCapabilityError, match="coordinate-selection"):
        hanging.reduce_sparse_operator(la.IdentityLinearOperator(full))


def test_large_dirichlet_laplace_stays_sparse() -> None:
    """A 50k-DOF P2 Dirichlet problem solves on O(nnz) storage, never densified.

    The large case uses the preconditioned-Krylov branch of the sparse route
    (IC(0) on the eliminated block); the complete native factor shares the same
    eliminated operator and is exercised on the curved case above.
    """
    mesh = _mesh(*_triangles(112), "triangle")
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 2)
        ),
    ).prepare()
    assert space.dof_maps[0].global_dof_count >= 50_000
    problem = _dirichlet_laplace(space, lambda value: value[..., 0] + 2.0 * value[..., 1])
    system, rhs = problem.linear_system(storage="sparse")
    size = system.operator.source.size
    nnz = system.operator.sparse_storage().nnz
    assert size >= 49_000
    assert nnz < 20 * size
    policy = la.LinearSolvePolicy(
        la.PCG(),
        tolerance=la.TolerancePolicy(relative=1.0e-12, absolute=0.0, max_steps=2000),
        preconditioning=la.PreconditioningPolicy(
            la.IncompleteCholeskyPreconditionerBuilder(ordering="reverse-cuthill-mckee")
        ),
    )
    prepared = la.prepare(system, policy)
    factorization = _factorization(prepared)
    assert int(factorization.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert factorization.plan.factor_nnz <= nnz
    result = la.solve(prepared, rhs)
    assert bool(result.successful)
    assert float(result.diagnostics.relative_residual) < 1.0e-12
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    np.testing.assert_allclose(
        np.asarray(problem.expand(result.value)),
        dofs[:, 0] + 2.0 * dofs[:, 1],
        rtol=0.0,
        atol=1.0e-8,
    )


def _restricted_curved_p2_spaces(
    kind: str,
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.discretization.FiniteElementDiscretization,
]:
    from phydrax.discretization._cell_geometry import (
        PerCellRestrictedCellGeometryElement,
    )
    from phydrax.discretization._coordinate_enclosure import axes

    shared = _curved_p2_space(kind)
    block = shared.mesh.blocks[0]
    cells = tuple(int(value) for value in np.asarray(block.global_ids).tolist())
    p2 = phx.discretization.lagrange_element(kind, 2)
    restricted = PerCellRestrictedCellGeometryElement(
        p2, kind, cells, cells, (axes(2),) * len(cells)
    )
    geometry = phx.discretization.CellGeometrySpec(
        {block.name: restricted},
        {block.name: np.asarray(shared.dof_maps[0].cell_dofs[0])},
        np.asarray(shared.dof_maps[0].dof_coordinates),
    )
    space = phx.discretization.FiniteElementPlan(
        shared.mesh,
        phx.discretization.FiniteElementFieldSpec("u", p2),
        coordinate_spec=geometry,
    ).prepare()
    return shared, space


@pytest.mark.parametrize("kind", ("triangle", "simplex:2"))
def test_per_cell_exact_charts_prepare_and_solve(kind: str) -> None:
    """Adapted curved results carry per-cell exact charts; FE prepares and solves."""
    shared, space = _restricted_curved_p2_spaces(kind)
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    np.testing.assert_allclose(
        points, np.asarray(shared.dof_maps[0].dof_coordinates), rtol=0.0, atol=1e-14
    )
    problem = _dirichlet_laplace(space, lambda value: value[..., 0])
    system, rhs = problem.linear_system(storage="sparse")
    result = la.solve(system, rhs, policy=_NATIVE_CHOLESKY)
    assert bool(result.successful)
    np.testing.assert_allclose(
        np.asarray(problem.expand(result.value)), points[:, 0], rtol=0.0, atol=1e-13
    )
    action_system, action_rhs = problem.linear_system()
    np.testing.assert_allclose(
        np.asarray(action_system.operator.mv(result.value)),
        np.asarray(action_rhs),
        rtol=0.0,
        atol=1e-12,
    )


def test_restricted_triangle_boundary_geometry_is_differentiable() -> None:
    # Prepared facet actions currently admit named triangle/tensor families;
    # generic simplex:2 volume support is exercised separately above.
    shared, space = _restricted_curved_p2_spaces("triangle")
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    # Boundary geometry and its coordinate derivative also consume prepared
    # per-cell charts, without converting traced entity routes to NumPy.
    boundary_form = phx.equations.FiniteElementForm(
        "restricted-boundary-load", "u", (phx.equations.BoundaryLoadAction("u", 1.0),)
    )
    boundary = phx.equations.compile_finite_element_problem(boundary_form, space)
    shared_boundary = phx.equations.compile_finite_element_problem(boundary_form, shared)
    state = jnp.zeros(points.shape[0])
    np.testing.assert_allclose(
        boundary.full_residual(state),
        shared_boundary.full_residual(state),
        rtol=2.0e-13,
        atol=2.0e-14,
    )

    def load_at_coordinates(
        problem: phx.equations.CompiledFiniteElementProblem,
        coordinates: jax.Array,
    ) -> jax.Array:
        runtime = eqx.tree_at(
            lambda value: value.coordinates,
            problem.discretization.default_runtime,
            coordinates,
        )
        return problem.full_residual(
            state, phx.equations.FiniteElementExecutionContext(runtime)
        )

    coordinates = space.default_runtime.coordinates
    direction = coordinates * 0.1
    restricted_tangent = jax.jvp(
        lambda value: load_at_coordinates(boundary, value),
        (coordinates,),
        (direction,),
    )[1]
    shared_tangent = jax.jvp(
        lambda value: load_at_coordinates(shared_boundary, value),
        (coordinates,),
        (direction,),
    )[1]
    np.testing.assert_allclose(
        restricted_tangent, shared_tangent, rtol=2.0e-13, atol=2.0e-14
    )


def _elasticity_problem(space: Any, *actions: Any, **compile_options: Any) -> Any:
    elasticity = phx.equations.fem.linear_elasticity_form("u", 1.3, 0.7)
    return phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "elasticity",
            "u",
            elasticity.actions + actions,
            properties=elasticity.declared_properties,
        ),
        space,
        **compile_options,
    )


def _mixed_vector_space(count: int) -> Any:
    mesh = _distorted_hex_pyramid_tet_mesh(count)
    return phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u",
            {
                block.name: phx.discretization.lagrange_element(block.cell_kind, 1)
                for block in mesh.blocks
            },
            component_shape=(3,),
        ),
    ).prepare()


def _dense(operator: Any) -> np.ndarray:
    sparse = la.assemble_sparse(operator)
    assert isinstance(sparse, phx.sparse.SparseCoordinateOperator)
    relation = sparse.relation
    assert isinstance(relation, phx.sparse.EdgeRelation)
    valid = np.asarray(relation.valid, dtype=np.bool_)
    dense = np.zeros((relation.target_size, relation.source_size))
    np.add.at(
        dense,
        (
            np.asarray(relation.target_indices)[valid],
            np.asarray(relation.source_indices)[valid],
        ),
        np.asarray(sparse.coefficients)[valid],
    )
    return dense


def test_coupled_cell_assembly_uses_node_major_component_order() -> None:
    """A coupled ``k (x) I`` block scatters exactly like the per-component path."""
    space = _curved_p2_space("triangle", (2,))
    geometry = space.block_geometries[0][0]
    gradients = np.asarray(geometry.physical_gradients)
    local = np.einsum(
        "cq,cqid,cqjd->cij", np.asarray(geometry.physical_weights), gradients, gradients
    )
    cells, width = local.shape[:2]
    coupled = np.einsum("cij,ab->ciajb", local, np.eye(2)).reshape(
        cells, 2 * width, 2 * width
    )
    shared = space.assemble_cell_operator("u", (local,), operator_id="per-component")
    blocks = space.assemble_cell_operator(
        "u", (coupled,), operator_id="coupled", coupled_components=True
    )
    np.testing.assert_array_equal(_dense(blocks), _dense(shared))
    with pytest.raises(ValueError, match="Cell operator tensor shape"):
        space.assemble_cell_operator(
            "u", (local,), operator_id="coupled", coupled_components=True
        )


@pytest.mark.parametrize("case", ("curved-p2", "curved-p2-with-mass", "mixed-3d"))
def test_linear_elasticity_sparse_operator_is_the_residual_jacobian(case: str) -> None:
    """Structural elasticity assembles the opaque residual's exact Jacobian."""
    if case == "mixed-3d":
        problem = _elasticity_problem(_mixed_vector_space(2))
    else:
        mass = (phx.equations.MassAction("u", 0.4),) if case.endswith("mass") else ()
        problem = _elasticity_problem(_curved_p2_space("triangle", (2,)), *mass)
    zero = problem.state_space.zeros()
    jacobian = np.asarray(
        jax.jacfwd(lambda flat: problem.residual(flat.reshape(zero.shape)).reshape(-1))(
            zero.reshape(-1)
        )
    )
    # Direct CSR conversion refuses any lowering that would materialize.
    csr = problem.to_scipy_csr()
    size = zero.size
    assert csr.shape == (size, size)
    assert csr.nnz < size * size // 2
    scale = float(np.max(np.abs(jacobian)))
    np.testing.assert_allclose(csr.toarray(), jacobian, rtol=0.0, atol=1.0e-13 * scale)


def test_linear_elasticity_patch_test_through_sparse_cholesky() -> None:
    """An affine displacement on the boundary is reproduced at interior nodes."""
    space = _mixed_vector_space(3)
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    gradient = np.asarray(((1.0, 0.2, -0.3), (0.1, -0.5, 0.25), (0.4, 0.0, 0.7)))
    affine = dofs @ gradient.T + np.asarray((0.1, -0.2, 0.05))
    boundary = np.asarray(space.dof_maps[0].boundary_dof_mask, dtype=np.bool_)
    assert np.count_nonzero(~boundary) > 0
    problem = _elasticity_problem(
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=boundary
        ),
        dirichlet_values=np.where(boundary[:, None], affine, 0.0),
    )
    system, rhs = problem.linear_system(storage="sparse")
    operator = system.operator
    assert isinstance(operator, phx.sparse.SparseCoordinateOperator)
    assert operator.properties.certifies("self_adjoint")
    factor = la.factorize_sparse(
        operator, la.SparseFactorizationPolicy("cholesky", ordering="nested-dissection")
    )
    assert int(factor.status) == int(la.SparseFactorizationStatus.SUCCESS)
    assert int(factor.diagnostics.rejected_pivots) == 0
    assert float(factor.diagnostics.minimum_pivot) > 0.0
    solved = factor.solve(rhs)
    assert bool(solved.success)
    np.testing.assert_allclose(
        np.asarray(problem.expand(solved.value)), affine, rtol=0.0, atol=1.0e-13
    )


def test_clamped_linear_elasticity_is_spd_after_sparse_elimination() -> None:
    """Dirichlet elimination of the rigid modes leaves an SPD sparse block."""
    space = _mixed_vector_space(2)
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    full = _elasticity_problem(space).to_scipy_csr().toarray()
    singular = la.verify_dense_properties(
        full,
        policy=la.DensePropertyVerificationPolicy(require_positive_semidefinite=True),
    )
    assert bool(singular.successful)
    # The six rigid-body modes are the whole kernel of the free operator.
    assert int(singular.numerical_rank) == full.shape[0] - 6
    clamp = dofs[:, 0] < 1.0e-12
    problem = _elasticity_problem(
        space,
        phx.equations.SourceAction("u", jnp.asarray((0.0, 0.0, -1.0))),
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=clamp
        ),
        dirichlet_values=0.0,
    )
    system, rhs = problem.linear_system(storage="sparse")
    operator = system.operator
    assert isinstance(operator, phx.sparse.SparseCoordinateOperator)
    matrix = problem.to_scipy_csr().toarray()
    evidence = la.verify_dense_properties(
        matrix,
        policy=la.DensePropertyVerificationPolicy(require_positive_definite=True),
    )
    assert bool(evidence.successful)
    assert evidence.properties.certifies("positive_definite")
    verified = phx.sparse.SparseCoordinateOperator(
        operator.relation,
        operator.coefficients,
        source=operator.source,
        target=operator.target,
        properties=evidence.properties,
    )
    prepared = la.prepare(la.LinearSystem(verified), _NATIVE_CHOLESKY)
    factorization = _factorization(prepared)
    assert int(factorization.status) == int(la.SparseFactorizationStatus.SUCCESS)
    result = la.solve(prepared, rhs)
    assert bool(result.successful)
    value = np.asarray(result.value)
    np.testing.assert_allclose(
        value,
        np.linalg.solve(matrix, np.asarray(rhs)),
        rtol=0.0,
        atol=1.0e-12 * float(np.max(np.abs(value))),
    )
