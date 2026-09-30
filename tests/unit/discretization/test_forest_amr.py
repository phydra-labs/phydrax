#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _grid(shape: Any, *, periodic: Any = None, upper: Any = None) -> Any:
    periodic_ = (False,) * len(shape) if periodic is None else periodic
    upper_ = tuple(float(value) for value in shape) if upper is None else upper
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(cells, periodic=flag)
            for cells, flag in zip(shape, periodic_, strict=True)
        ),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.asarray([[0.0] * len(shape), list(upper_)]))


def _plan(
    shape: Any,
    *,
    maximum_level: Any,
    periodic: Any = None,
    balance: Any = "face",
    maps: Any = None,
) -> Any:
    return phx.discretization.ForestPlan(
        _grid(shape, periodic=periodic),
        maximum_level=maximum_level,
        balance=phx.discretization.AMRBalanceStencil(balance),
        maximum_leaf_capacity=1 << 14,
        maps=maps,
    )


def _marks(topology: Any, values: Any) -> Any:
    marks = np.zeros((topology.signature.leaf_capacity,), dtype=np.int8)
    marks[: topology.leaf_count] = values
    return marks


def _refine_slots(compiler: Any, topology: Any, slots: Any) -> Any:
    values = np.zeros((topology.leaf_count,), dtype=np.int8)
    values[list(slots)] = 1
    return compiler.adapt(topology, _marks(topology, values))


def _finest_boxes(topology: Any) -> Any:
    plan = topology.plan
    levels = topology.leaf_levels()
    shift = plan.maximum_level - levels
    lower = topology.leaf_coordinates() << shift[:, None]
    upper = lower + (1 << shift)[:, None]
    extent = np.asarray(plan.root_shape) << plan.maximum_level
    return levels, lower, upper, extent


def _contact_dimension(topology: Any, first: Any, second: Any) -> Any:
    """Largest contact dimension of two leaves over periodic images (-1: apart)."""
    levels, lower, upper, extent = _finest_boxes(topology)
    best = -1
    shifts = [
        (0, extent[axis], -extent[axis]) if topology.plan.periodic_axes[axis] else (0,)
        for axis in range(topology.plan.dimension)
    ]
    for shift in itertools.product(*shifts):
        overlap = np.minimum(upper[first], upper[second] + shift) - np.maximum(
            lower[first], lower[second] + shift
        )
        if np.all(overlap >= 0):
            best = max(best, int(np.count_nonzero(overlap > 0)))
    return best


def test_forest_amr_scenario_1() -> None:
    for shape, periodic, stencil in [
        ((2, 2), (False, True), "face"),
        ((2, 2), (False, True), "corner"),
        ((1, 1, 2), (True, False, False), "edge"),
    ]:
        plan = _plan(
            shape,
            maximum_level=5 if len(shape) == 2 else 4,
            periodic=periodic,
            balance=stencil,
        )
        compiler = phx.discretization.ForestTopologyCompiler(plan)
        topology = compiler.initialize(0).topology
        target = np.asarray(plan.root_shape) << plan.maximum_level
        point = (target // 2)[None, :]
        closures = 0
        for _ in range(plan.maximum_level):
            # ty: ignore[invalid-argument-type]
            slot = topology.locate_cells([plan.maximum_level], point)
            result = _refine_slots(compiler, topology, slot)
            assert result.status.successful
            closures += result.evidence.balance_refinements
            topology = result.topology
        levels = topology.leaf_levels()
        codimension = {
            "face": 1,
            "edge": min(2, plan.dimension),
            "corner": plan.dimension,
        }[stencil]
        assert closures > 0
        assert levels.max() == plan.maximum_level
        for first, second in itertools.combinations(range(topology.leaf_count), 2):
            if abs(int(levels[first]) - int(levels[second])) > 1:
                assert _contact_dimension(topology, first, second) < (
                    plan.dimension - codimension
                )
    plan = _plan((2, 1), maximum_level=3, periodic=(True, False))
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    topology = compiler.initialize(1).topology
    topology = _refine_slots(compiler, topology, (0, 5)).topology
    workset = topology.workset
    valid = np.asarray(workset.face_valid)
    faces = set(
        zip(
            np.asarray(workset.face_axes)[valid].tolist(),
            np.asarray(workset.face_minus)[valid].tolist(),
            np.asarray(workset.face_plus)[valid].tolist(),
            strict=True,
        )
    )
    levels, lower, upper, extent = _finest_boxes(topology)
    expected = set()
    for axis in range(plan.dimension):
        for minus, plus in itertools.product(range(topology.leaf_count), repeat=2):
            shifts = (0, extent[axis]) if plan.periodic_axes[axis] else (0,)
            touching = any(
                upper[minus, axis] == lower[plus, axis] + shift for shift in shifts
            )
            tangential = all(
                min(upper[minus, other], upper[plus, other])
                > max(lower[minus, other], lower[plus, other])
                for other in range(plan.dimension)
                if other != axis
            )
            if touching and tangential:
                expected.add((axis, minus, plus))
    assert faces == expected
    assert len(faces) == topology.face_count
    hanging = np.asarray(workset.face_coarse_fine)[valid]
    minus_levels = levels[np.asarray(workset.face_minus)[valid]]
    plus_levels = levels[np.asarray(workset.face_plus)[valid]]
    np.testing.assert_array_equal(hanging, minus_levels != plus_levels)
    assert np.all(
        np.asarray(workset.face_levels)[valid] == np.maximum(minus_levels, plus_levels)
    )
    plan = _plan((2, 2), maximum_level=4, periodic=(True, True))
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    source = compiler.initialize(2).topology
    rng = np.random.default_rng(3)
    # Refine two leaves of the first root and coarsen every family of the last two.
    values = np.where(np.arange(source.leaf_count) >= 32, -1, 0)
    values[[5, 10]] = 1
    result = compiler.adapt(source, _marks(source, values))
    assert result.status.code == "success"
    assert result.evidence.coarsened_families > 0
    target = result.topology
    transition = phx.discretization.ForestFieldTransition(source, target)
    assert transition.restricted_leaves > 0 and transition.prolonged_leaves > 0
    capacity = source.signature.leaf_capacity
    field = jnp.asarray(rng.normal(size=(capacity, 2, 3)))
    transferred = transition.routes.apply(field)
    assert bool(transferred.successful)
    np.testing.assert_allclose(transferred.conservation_residual, 0.0, atol=1e-13)
    constant = transition.routes.apply(jnp.full((capacity,), 2.5))
    np.testing.assert_allclose(constant.values[: target.leaf_count], 2.5, rtol=1e-15)
    np.testing.assert_array_equal(constant.values[target.leaf_count :], 0.0)
    source_values = (
        jnp.asarray(rng.normal(size=(capacity,))) * transition.routes.source_measures
    )
    target_values = jnp.asarray(rng.normal(size=(target.signature.leaf_capacity,)))
    pushed = transition.routes.apply(source_values).values
    np.testing.assert_allclose(
        jnp.sum(transition.routes.target_measures * pushed * target_values),
        jnp.sum(
            transition.routes.source_measures
            * source_values
            * transition.routes.hilbert_adjoint(target_values)
        ),
        rtol=1e-12,
    )


def test_forest_amr_scenario_2() -> None:
    plan = _plan((2, 1), maximum_level=3)
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    source = compiler.initialize(1).topology
    refined = _refine_slots(compiler, source, (1, 2)).topology
    assert refined.epoch.index == source.epoch.index + 1
    children = np.isin(
        np.arange(refined.leaf_count), np.flatnonzero(refined.leaf_levels() == 2)
    )
    coarsened = compiler.adapt(refined, _marks(refined, -children.astype(np.int8)))
    restored = coarsened.topology
    assert coarsened.evidence.coarsened_families == 2
    assert restored.topology_id == source.topology_id
    assert restored.epoch.index == source.epoch.index + 2
    np.testing.assert_array_equal(
        restored.workset.leaf_path_ids, source.workset.leaf_path_ids
    )
    values = jnp.asarray(
        np.random.default_rng(0).normal(size=(source.signature.leaf_capacity,))
    )
    values = jnp.where(source.workset.leaf_valid, values, 0.0)
    up = phx.discretization.ForestFieldTransition(source, refined).routes.apply(values)
    down = phx.discretization.ForestFieldTransition(refined, restored).routes.apply(
        up.values
    )
    np.testing.assert_allclose(down.values, values, rtol=0.0, atol=1e-15)
    plan = phx.discretization.ForestPlan(
        _grid((2, 2)),
        maximum_level=3,
        minimum_leaf_capacity=4,
        maximum_leaf_capacity=8,
    )
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    source = compiler.initialize(0).topology
    result = _refine_slots(compiler, source, (0, 1, 2))
    assert result.status.code == "capacity_exceeded"
    assert not result.status.successful
    assert result.topology.topology_id == source.topology_id
    assert result.evidence.target_leaf_count == 13
    with pytest.raises(ValueError, match=r"\{-1, 0, 1\}"):
        compiler.adapt(source, _marks(source, [2, 0, 0, 0]))
    plan = _plan((2, 1), maximum_level=3, periodic=(False, True))
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    coarse = _refine_slots(compiler, compiler.initialize(1).topology, (0, 3)).topology
    fine = _refine_slots(compiler, coarse, (0, 1, 5)).topology
    coarse_complex = phx.discretization.ForestCochainComplex(coarse)
    fine_complex = phx.discretization.ForestCochainComplex(fine)
    # A cylinder (periodic y, bounded x) has Euler characteristic zero.
    for complex_ in (coarse_complex, fine_complex):
        counts = complex_.entity_counts
        assert counts[0] - counts[1] + counts[2] == 0
    family = phx.discretization.ForestCochainTransfer(coarse_complex, fine_complex)
    rng = np.random.default_rng(7)
    for degree in range(plan.dimension + 1):
        transfer = family.transfer(degree)
        cochain = jnp.asarray(
            rng.normal(size=(coarse_complex.spaces[degree].size,))
            * np.asarray(coarse_complex.entity_valid[degree])
        )
        prolonged = transfer.prolongation.mv(cochain)
        np.testing.assert_allclose(
            transfer.restriction.mv(prolonged), cochain, atol=1e-13
        )
        if degree < plan.dimension:
            np.testing.assert_allclose(
                fine_complex.coboundary(degree).mv(prolonged),
                family.transfer(degree + 1).prolongation.mv(
                    coarse_complex.coboundary(degree).mv(cochain)
                ),
                atol=1e-13,
            )
        else:
            np.testing.assert_allclose(jnp.sum(prolonged), jnp.sum(cochain), atol=1e-13)
    with pytest.raises(ValueError, match="must refine"):
        phx.discretization.ForestCochainTransfer(fine_complex, coarse_complex)


@pytest.mark.parametrize(
    ("periodic", "betti"),
    [((False, False), (1, 0, 0)), ((False, True), (1, 1, 0))],
    ids=["disk", "cylinder"],
)
def test_forest_hilbert_cohomology_excludes_capacity_padding(
    periodic: tuple[bool, bool],
    betti: tuple[int, int, int],
) -> None:
    plan = _plan((2, 1), maximum_level=3, periodic=periodic)
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    topology = _refine_slots(compiler, compiler.initialize(1).topology, (0, 3)).topology
    forest = phx.discretization.ForestCochainComplex(topology)
    hodges = tuple(
        phx.discretization.DiagonalHodge(jnp.linspace(1.0, 2.0, space.size))
        for space in forest.spaces
    )
    complex_ = forest.hilbert_complex(hodges)
    matrices = tuple(
        np.asarray(
            jax.vmap(operator.mv)(jnp.eye(operator.source.size, dtype=jnp.float64)).T
        )
        for operator in complex_.differentials
    )
    ranks = tuple(np.linalg.matrix_rank(matrix, tol=1.0e-10) for matrix in matrices)
    computed = (
        complex_.space(0).size - ranks[0],
        complex_.space(1).size - ranks[0] - ranks[1],
        complex_.space(2).size - ranks[1],
    )
    assert computed == betti
    np.testing.assert_allclose(matrices[1] @ matrices[0], 0.0, atol=1.0e-13)


def test_forest_hilbert_adjoint_uses_restricted_metric() -> None:
    compiler = phx.discretization.ForestTopologyCompiler(_plan((2, 1), maximum_level=2))
    topology = _refine_slots(compiler, compiler.initialize(0).topology, (0,)).topology
    forest = phx.discretization.ForestCochainComplex(topology)
    hodges = tuple(
        phx.discretization.DiagonalHodge(jnp.linspace(0.75, 2.5, space.size))
        for space in forest.spaces
    )
    complex_ = forest.hilbert_complex(hodges)
    u = jnp.linspace(-0.7, 0.4, complex_.space(0).size)
    v = jnp.linspace(0.2, 1.1, complex_.space(1).size)
    d = complex_.differential(0)
    delta = phx.linalg.codifferential(complex_, 1)
    np.testing.assert_allclose(
        complex_.space(1).inner(d.mv(u), v),
        complex_.space(0).inner(u, delta.mv(v)),
        atol=1.0e-12,
    )


def test_forest_sparse_hodge_inverse_excludes_inactive_couplings() -> None:
    compiler = phx.discretization.ForestTopologyCompiler(_plan((2, 1), maximum_level=2))
    forest = phx.discretization.ForestCochainComplex(compiler.initialize(0).topology)
    matrices = tuple(
        2.0 * np.eye(space.size, dtype=np.float64) + 0.125 for space in forest.spaces
    )
    hodges = []
    for matrix in matrices:
        rows, columns = np.triu_indices(matrix.shape[0])
        hodges.append(
            phx.discretization.SparseHodge(
                rows,
                columns,
                matrix[rows, columns],
                matrix.shape[0],
            )
        )
    complex_ = forest.hilbert_complex(tuple(hodges))
    count = complex_.space(0).size
    assert forest.spaces[0].size > count
    rhs = jnp.linspace(0.25, 1.25, count)
    result = complex_.space(0).inverse_riesz(rhs)
    np.testing.assert_allclose(
        result,
        np.linalg.solve(matrices[0][:count, :count], np.asarray(rhs)),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    full_rhs = np.zeros((forest.spaces[0].size,), dtype=np.float64)
    full_rhs[:count] = np.asarray(rhs)
    wrong = np.linalg.solve(matrices[0], full_rhs)[:count]
    assert np.max(np.abs(wrong - np.asarray(result))) > 1.0e-4


def test_subcycled_reflux_restores_exact_conservation() -> None:
    plan = _plan((2, 2), maximum_level=3, periodic=(True, True))
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    topology = _refine_slots(compiler, compiler.initialize(1).topology, (0, 1)).topology
    workset = topology.workset
    geometry = phx.discretization.forest_leaf_geometry(topology)
    velocity = jnp.asarray([1.0, 0.5])
    minus = jnp.where(workset.face_valid, workset.face_minus, 0)
    plus = jnp.where(workset.face_valid, workset.face_plus, 0)

    def flux(state: Any) -> Any:
        speed = velocity[jnp.where(workset.face_valid, workset.face_axes, 0)]
        upwind = jnp.where(speed > 0.0, state[minus], state[plus])
        return jnp.where(workset.face_valid, speed * upwind * geometry.face_measures, 0.0)

    def rate(face_flux: Any) -> Any:
        change = jnp.zeros_like(geometry.volumes).at[minus].add(-face_flux)
        return change.at[plus].add(face_flux) / geometry.volumes

    def content(state: Any) -> Any:
        return jnp.sum(jnp.where(workset.leaf_valid, state * geometry.volumes, 0.0))

    fine = workset.leaf_valid & (workset.leaf_levels == topology.leaf_levels().max())
    rng = np.random.default_rng(5)
    state = jnp.where(workset.leaf_valid, jnp.asarray(rng.uniform(size=fine.shape)), 0.0)
    step = 0.02
    coarse_flux = flux(state)
    midpoint = jnp.where(fine, state + 0.5 * step * rate(coarse_flux), state)
    fine_flux = flux(midpoint)
    advanced = jnp.where(
        fine,
        midpoint + 0.5 * step * rate(fine_flux),
        state + step * rate(coarse_flux),
    )
    routes = phx.discretization.ForestRefluxRoutes(topology)
    register = routes.register(
        step * coarse_flux,
        0.5 * step * (coarse_flux + fine_flux),
        accumulated_time=step,
    )
    corrected = routes.apply(register, advanced, geometry.volumes)
    assert abs(float(content(advanced) - content(state))) > 1e-8
    np.testing.assert_allclose(content(corrected), content(state), rtol=0.0, atol=1e-15)


def test_mapped_roots_transfer_conserves_chart_volumes() -> None:
    def shear(points: Any, time: Any, args: Any) -> Any:
        del time, args
        values = jnp.asarray(points)
        return jnp.stack(
            (2.0 * values[..., 0], 3.0 * values[..., 1] + 0.5 * values[..., 0]), axis=-1
        )

    maps = phx.discretization.PatchCoordinateMapSet(shear, "shear")
    plan = _plan((1, 1), maximum_level=2, maps=maps)
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    source = compiler.initialize(1).topology
    target = _refine_slots(compiler, source, (0,)).topology
    geometry = phx.discretization.forest_leaf_geometry(target)
    assert bool(geometry.valid)
    expected = np.where(target.leaf_levels() == 2, 6.0 / 16.0, 6.0 / 4.0)
    np.testing.assert_allclose(
        geometry.volumes[: target.leaf_count], expected, rtol=1e-12
    )
    transition = phx.discretization.ForestFieldTransition(source, target)
    assert not transition.properties.constant_preserving
    transferred = transition.routes.apply(
        jnp.arange(source.signature.leaf_capacity, dtype=jnp.float64)
    )
    assert bool(transferred.successful)


def test_vertex_interpolation_reproduces_linear_fields_across_hanging_vertices() -> None:
    plan = _plan((2, 1), maximum_level=3)
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    source = _refine_slots(compiler, compiler.initialize(1).topology, (0, 3)).topology
    result = compiler.adapt(
        source,
        _marks(
            source,
            np.where(np.arange(source.leaf_count) < 4, -1, 0)
            + (np.arange(source.leaf_count) == 7),
        ),
    )
    target = result.topology
    source_layout = phx.discretization.ForestVertexLayout(source)
    target_layout = phx.discretization.ForestVertexLayout(target)
    assert bool(jnp.any(source_layout.hanging))
    transfer = phx.discretization.forest_vertex_interpolation(
        source_layout, target_layout
    )
    assert transfer.preserves_constants and transfer.preserves_linear

    def linear(points: Any) -> Any:
        return 0.5 + 2.0 * points[:, 0] - 3.0 * points[:, 1]

    source_points = source_layout.reference_coordinates()
    target_points = target_layout.reference_coordinates()
    np.testing.assert_allclose(
        transfer.apply(jnp.asarray(linear(source_points))),
        linear(target_points),
        atol=1e-13,
    )
    rng = np.random.default_rng(11)
    left = jnp.asarray(rng.normal(size=(source_layout.vertex_count,)))
    right = jnp.asarray(rng.normal(size=(target_layout.vertex_count,)))
    np.testing.assert_allclose(
        jnp.vdot(transfer.apply(left), right),
        jnp.vdot(left, transfer.pullback(right)),
        rtol=1e-12,
    )


def test_cut_embedding_assigns_components_to_containing_leaves() -> None:
    plan = phx.discretization.ForestPlan(
        _grid((2, 2), upper=(1.0, 1.0)),
        maximum_level=2,
        maximum_leaf_capacity=256,
    )
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    topology = _refine_slots(compiler, compiler.initialize(1).topology, (0, 5)).topology
    body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: points[:, 0] - 0.4, "forest-plane", 3
    )
    cut = phx.discretization.prepare_forest_cut_complex(
        topology,
        phx.discretization.EmbeddedLevelSetBodySet((body,)),
        phx.discretization.BlockAMRResourcePlan(
            maximum_components_per_cell=2,
            maximum_apertures_per_face=8,
            maximum_embedded_faces_per_cell=16,
        ),
    )
    complex_ = cut.complex
    active = np.asarray(complex_.component_active)
    slots = np.asarray(cut.component_leaf_slots)
    np.testing.assert_allclose(
        np.asarray(complex_.component_areas)[active].sum(), 0.6, atol=1e-12
    )
    assert np.all(slots[~active] == -1)
    lower, upper = topology.reference_bounds()
    centers = np.asarray(complex_.component_centers)[active]
    assert np.all(centers >= lower[slots[active]] - 1e-12)
    assert np.all(centers <= upper[slots[active]] + 1e-12)
    fluid = 0.5 * (lower + upper)[: topology.leaf_count, 0] > 0.4
    assert set(slots[active].tolist()) >= set(np.flatnonzero(fluid).tolist())
