#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Signature-grouped lane execution of spatial coupled problems.

A chain of unit strips ``[k, k + 1] x [0, 1]`` carries P1 Laplace owners of the
harmonic field ``u = sin(x / 2) exp(y / 2)`` with Dirichlet data on the outer
boundary and mortar transmission laws on the cuts ``x = k``. Interior strips
are translated copies of one executable, and so are the two mirrored end
strips; neither pair shares the other's Dirichlet rows. The per-component
execution of the same plan is the reference for every lane-execution
contract.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain


cpl = phx.solver.coupling

_CELLS = 3
_STRIPS = 5
_INTERIOR = ("s001", "s002", "s003")
_ENDS = ("s000", "s004")


def _harmonic(points: Array) -> Array:
    return jnp.sin(0.5 * points[..., 0]) * jnp.exp(0.5 * points[..., 1])


def _no_source(points: Array, args: object) -> Array:
    del args
    return jnp.zeros(points.shape[:-1], dtype=points.dtype)


def _unit_source(points: Array, args: object) -> Array:
    del args
    return jnp.ones(points.shape[:-1], dtype=points.dtype)


type _Source = Callable[[Array, object], Array]


def _strip_mesh(x0: float, /) -> phx.discretization.CellMesh:
    xs = np.linspace(x0, x0 + 1.0, _CELLS + 1)
    ys = np.linspace(0.0, 1.0, _CELLS + 1)
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
    triangles = []
    for j in range(_CELLS):
        for i in range(_CELLS):
            a = j * (_CELLS + 1) + i
            triangles += [(a, a + 1, a + _CELLS + 2), (a, a + _CELLS + 2, a + _CELLS + 1)]
    return phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class _Strip:
    component: cpl.VariationalComponent
    cuts: dict[str, IntegrationDomain]
    points: np.ndarray


def _strip(index: int, /, *, diffusivity: float, source: _Source) -> _Strip:
    """P1 owner of strip ``index``; interface rows of its cuts stay free."""
    d = phx.discretization
    space = d.FiniteElementPlan(
        _strip_mesh(float(index)),
        d.FiniteElementFieldSpec("u", d.lagrange_element("triangle", 1)),
    ).prepare()
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    sites = np.asarray(probe.sites)[..., 0]
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    dirichlet = np.array(space.dof_maps[0].boundary_dof_mask, dtype=np.bool_)
    edges = space.mesh.topology.entity_sets[1]
    cuts: dict[str, IntegrationDomain] = {}
    for side, x in (("left", float(index)), ("right", float(index + 1))):
        if (side == "left" and index == 0) or (side == "right" and index == _STRIPS - 1):
            continue
        mask = np.zeros((edges.count,), dtype=np.bool_)
        mask[
            np.asarray(exterior.entity_indices)[np.all(np.isclose(sites, x), axis=1)]
        ] = True
        cuts[side] = space.integration_domain(
            "exterior_facet", EntitySelection(edges, mask)
        )
        dirichlet &= ~(
            np.isclose(points[:, 0], x)
            & (points[:, 1] > 1.0e-12)
            & (points[:, 1] < 1.0 - 1.0e-12)
        )
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "laplace",
            "u",
            (
                phx.equations.DiffusionAction("u", diffusivity),
                phx.equations.SourceAction(
                    "u", phx.equations.coefficient(source, coefficient_id="f")
                ),
            ),
        ),
        space,
        constraint=d.dirichlet_constraint(space, "u", boundary_mask=dirichlet),
        dirichlet_values=_harmonic,
    )
    return _Strip(
        cpl.VariationalComponent(f"s{index:03d}", problem, field="u"), cuts, points
    )


@dataclass(frozen=True, slots=True)
class _Chain:
    cover: phx.domain.SubdomainCover
    components: tuple[cpl.VariationalComponent, ...]
    bindings: tuple[cpl.InterfaceBinding, ...]
    laws: tuple[cpl.ScalarTransmissionLaw, ...]
    points: tuple[np.ndarray, ...]

    def prepare(
        self,
        execution: cpl.CoupledExecutionPolicy | None,
        /,
        arguments: Mapping[str, object] | None = None,
    ) -> cpl.PreparedCoupledProblem:
        plan = cpl.CoupledProblemPlan(
            "strip-chain",
            components=self.components,
            bindings=self.bindings,
            laws=self.laws,
            resources=cpl.CoupledResourcePolicy(execution=execution),
        )
        return cpl.prepare_coupled_problem(
            plan, interface_owners=(self.cover,), arguments=arguments
        )


def _chain(
    *,
    diffusivities: dict[int, float] | None = None,
    sources: dict[int, _Source] | None = None,
) -> _Chain:
    scales = {} if diffusivities is None else diffusivities
    loads = {} if sources is None else sources
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([float(_STRIPS), 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(
        plate, "x", (_STRIPS, 1), cover_id="strips"
    )
    strips = [
        _strip(
            index,
            diffusivity=scales.get(index, 1.0),
            source=loads.get(index, _no_source),
        )
        for index in range(_STRIPS)
    ]
    bindings: list[cpl.InterfaceBinding] = []
    laws: list[cpl.ScalarTransmissionLaw] = []
    for index, pairing in enumerate(cover.pairings):
        witness = pairing.component.sample(
            phx.domain.PointSampling(8), key=jax.random.key(1)
        )
        minus, plus = strips[index], strips[index + 1]
        binding = cpl.InterfaceBinding(
            f"cut{index:03d}",
            cpl.InterfaceSource.paired_support(cover, pairing.pairing_id),
            "two-sided",
            tuple(
                cpl.InterfaceEndpoint(
                    role,
                    cpl.PairedSupportAttachment(
                        cover, pairing.pairing_id, patch, witness
                    ),
                    fields={"value": strip.component.field_space_id("u")},
                )
                for role, patch, strip in (
                    ("minus", pairing.left_patch_id, minus),
                    ("plus", pairing.right_patch_id, plus),
                )
            ),
        )
        bindings.append(binding)
        laws.append(
            cpl.ScalarTransmissionLaw(
                f"law{index:03d}",
                binding,
                (
                    cpl.TransmissionSide(
                        "minus", minus.component.name, "u", minus.cuts["right"]
                    ),
                    cpl.TransmissionSide(
                        "plus", plus.component.name, "u", plus.cuts["left"]
                    ),
                ),
                cpl.MortarImposition(cpl.MortarMultiplier("side-trace", side="plus")),
            )
        )
    return _Chain(
        cover,
        tuple(strip.component for strip in strips),
        tuple(bindings),
        tuple(laws),
        tuple(strip.points for strip in strips),
    )


def _policy() -> phx.linalg.LinearSolvePolicy:
    return phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())


@pytest.fixture(scope="module")
def chain() -> _Chain:
    return _chain()


@pytest.fixture(scope="module")
def reference(chain: _Chain) -> cpl.PreparedCoupledProblem:
    return chain.prepare(None)


@pytest.fixture(scope="module")
def lanes(chain: _Chain) -> cpl.PreparedCoupledProblem:
    return chain.prepare(cpl.CoupledExecutionPolicy(lane_capacity=2))


def _state(
    prepared: cpl.PreparedCoupledProblem, seed: int, /
) -> tuple[tuple[Array, ...], ...]:
    rng = np.random.default_rng(seed)
    return prepared.state_space.unflatten(
        jnp.asarray(rng.normal(size=(prepared.state_space.size,)))
    )


def test_homogeneous_strips_form_bounded_component_worksets(
    lanes: cpl.PreparedCoupledProblem,
) -> None:
    worksets = lanes.worksets
    assert worksets is not None
    ends, interior = worksets.components
    assert (ends.members, interior.members) == (_ENDS, _INTERIOR)
    # Three lanes in buckets of two: two buckets, one padded lane.
    assert (interior.estimate.bucket_count, interior.estimate.padded_lanes) == (2, 1)
    assert interior.estimate.working_set_bytes == 2 * (
        interior.estimate.lane_data_bytes // 4
        + interior.estimate.lane_input_bytes
        + interior.estimate.lane_output_bytes
        + interior.estimate.traced_intermediate_bytes
    )
    assert worksets.interfaces
    assert all(len(interface.members) >= 2 for interface in worksets.interfaces)


def test_lane_residual_and_operators_match_per_component_reference(
    reference: cpl.PreparedCoupledProblem, lanes: cpl.PreparedCoupledProblem
) -> None:
    state = _state(reference, 3)
    rows = _state(reference, 4)
    expected = reference.row_space.flatten(reference.residual(state))
    np.testing.assert_allclose(
        lanes.row_space.flatten(lanes.residual(state)), expected, rtol=1e-12, atol=1e-12
    )
    system, rhs = reference.linear_system()
    lane_system, lane_rhs = lanes.linear_system()
    space = reference.state_space
    for apply in ("mv", "transpose_mv"):
        np.testing.assert_allclose(
            space.flatten(getattr(lane_system.operator, apply)(rows)),
            space.flatten(getattr(system.operator, apply)(rows)),
            rtol=1e-12,
            atol=1e-12,
        )
    np.testing.assert_allclose(space.flatten(lane_rhs), space.flatten(rhs), atol=1e-12)
    np.testing.assert_allclose(
        reference.row_space.flatten(lanes.weak_operator().mv(state)),
        reference.row_space.flatten(reference.weak_operator().mv(state)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_lane_solve_is_certified_like_the_reference(
    chain: _Chain,
    reference: cpl.PreparedCoupledProblem,
    lanes: cpl.PreparedCoupledProblem,
) -> None:
    expected = cpl.solve_coupled_problem(reference, policy=_policy())
    solution = cpl.solve_coupled_problem(lanes, policy=_policy())
    assert bool(expected.accepted) and bool(solution.accepted)
    np.testing.assert_allclose(
        lanes.state_space.flatten(solution.state),
        reference.state_space.flatten(expected.state),
        rtol=1e-10,
        atol=1e-12,
    )
    for certificate, reference_certificate in zip(
        solution.components, expected.components, strict=True
    ):
        assert certificate.component == reference_certificate.component
        np.testing.assert_allclose(
            certificate.residual_norm, reference_certificate.residual_norm, atol=1e-10
        )
    # The coupled field is the harmonic field up to the P1 discretization error.
    field = solution.field("s002", "u")
    points = chain.points[2]
    exact = np.sin(0.5 * points[:, 0]) * np.exp(0.5 * points[:, 1])
    assert float(np.max(np.abs(np.asarray(field) - exact))) < 5.0e-3


def _grouped_residual_matches_reference(
    chain: _Chain,
    groups: tuple[tuple[str, ...], ...],
    /,
    arguments: Mapping[str, object] | None = None,
) -> None:
    prepared = chain.prepare(cpl.CoupledExecutionPolicy(lane_capacity=4), arguments)
    worksets = prepared.worksets
    assert worksets is not None
    assert tuple(group.members for group in worksets.components) == groups
    reference = chain.prepare(None, arguments)
    state = _state(reference, 5)
    np.testing.assert_allclose(
        prepared.row_space.flatten(prepared.residual(state, arguments)),
        reference.row_space.flatten(reference.residual(state, arguments)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_equal_programs_with_different_coefficients_share_lanes_with_own_data() -> None:
    # The diffusivity is owner array data: one executable, per-lane values.
    _grouped_residual_matches_reference(
        _chain(diffusivities={2: 2.0}), (_ENDS, _INTERIOR)
    )


def test_signature_separates_equal_shapes_with_different_programs() -> None:
    # A different source function changes the traced program, not any shape.
    _grouped_residual_matches_reference(
        _chain(sources={2: _unit_source}), (_ENDS, ("s001", "s003"))
    )


def _user_arguments(args: object, /) -> Mapping[str, object]:
    """The component's runtime arguments inside a finite-element source call."""
    assert isinstance(args, phx.equations.FiniteElementExecutionContext)
    user = args.user_args
    assert isinstance(user, Mapping)
    return user


def _scaled_source(points: Array, args: object) -> Array:
    divisor = _user_arguments(args)["divisor"]
    assert isinstance(divisor, int | float)
    return jnp.ones(points.shape[:-1], dtype=points.dtype) / divisor


def test_equal_static_arguments_of_different_types_never_share_lanes() -> None:
    # 2 == 2.0 in Python, but an int and a float are different static arguments.
    chain = _chain(sources=dict.fromkeys(range(_STRIPS), _scaled_source))
    arguments: dict[str, object] = {
        component.name: {"divisor": 2} for component in chain.components
    }
    arguments["s003"] = {"divisor": 2.0}
    _grouped_residual_matches_reference(chain, (_ENDS, ("s001", "s002")), arguments)
    prepared = chain.prepare(cpl.CoupledExecutionPolicy(lane_capacity=4), arguments)
    with pytest.raises(ValueError, match="prepared lane signature"):
        prepared.residual(
            prepared.state_space.zeros(), {**arguments, "s002": {"divisor": 2.0}}
        )


def _rule_source(slope: float, /) -> _Source:
    """Source ``sin(a)`` of the runtime amplitude ``a`` whose custom JVP rule is
    ``slope * cos(a)``: every slope traces the same primal program."""

    @jax.custom_jvp
    def profile(amplitude: Array) -> Array:
        return jnp.sin(amplitude)

    @profile.defjvp
    def _profile_jvp(
        primals: tuple[Array], tangents: tuple[Array]
    ) -> tuple[Array, Array]:
        (amplitude,), (tangent,) = primals, tangents
        return jnp.sin(amplitude), slope * jnp.cos(amplitude) * tangent

    def source(points: Array, args: object) -> Array:
        amplitude = _user_arguments(args)["amplitude"]
        assert isinstance(amplitude, Array)
        return profile(amplitude) * jnp.ones(points.shape[:-1], dtype=points.dtype)

    return source


def test_members_with_different_custom_derivative_rules_never_share_lanes() -> None:
    # Only strip 2's derivative rule differs; a shared lane would differentiate
    # it with the representative's rule (half its derivative).
    chain = _chain(
        sources={
            index: _rule_source(2.0 if index == 2 else 1.0) for index in range(_STRIPS)
        }
    )
    arguments = {
        component.name: {"amplitude": jnp.asarray(0.4)} for component in chain.components
    }
    lanes = chain.prepare(cpl.CoupledExecutionPolicy(lane_capacity=4), arguments)
    reference = chain.prepare(None, arguments)
    worksets = lanes.worksets
    assert worksets is not None
    assert tuple(group.members for group in worksets.components) == (
        _ENDS,
        ("s001", "s003"),
    )
    state = _state(reference, 6)

    def total(
        prepared: cpl.PreparedCoupledProblem, amplitudes: dict[str, object]
    ) -> Array:
        return jnp.sum(prepared.row_space.flatten(prepared.residual(state, amplitudes)))

    expected = jax.grad(lambda values: total(reference, values))(arguments)
    gradient = jax.grad(lambda values: total(lanes, values))(arguments)
    np.testing.assert_allclose(
        np.asarray(jax.tree.leaves(gradient)),
        np.asarray(jax.tree.leaves(expected)),
        rtol=1e-12,
        atol=1e-12,
    )
    # The strips' finite-element programs keep custom rules beyond first
    # order, so a lane-grouped second derivative is refused, not evaluated
    # with the representative's rules.
    derivatives = worksets.components[1].derivatives
    assert not derivatives.higher_order
    with pytest.raises(ValueError, match="not certified"):
        jax.hessian(
            lambda amplitude: total(
                lanes, {**arguments, "s003": {"amplitude": amplitude}}
            )
        )(jnp.asarray(0.4))


def _host_source(scale: float, /) -> _Source:
    """Source ``scale`` evaluated on the host through ``pure_callback``: every
    scale traces the same program."""

    def host(values: np.ndarray) -> np.ndarray:
        return np.full_like(values, scale)

    def source(points: Array, args: object) -> Array:
        del args
        values = points[..., 0]
        return jax.pure_callback(
            host,
            jax.ShapeDtypeStruct(values.shape, values.dtype),
            values,
            vmap_method="sequential",
        )

    return source


def test_members_with_different_host_callbacks_never_share_lanes() -> None:
    # Strip 2's host function triples its source; the others are equal
    # closures (same code, equal captured scale) and share lanes.
    _grouped_residual_matches_reference(
        _chain(
            sources={
                index: _host_source(3.0 if index == 2 else 1.0)
                for index in range(_STRIPS)
            }
        ),
        (_ENDS, ("s001", "s003")),
    )


def test_working_set_above_the_declared_bound_is_refused(chain: _Chain) -> None:
    with pytest.raises(ValueError, match="working-set bound"):
        chain.prepare(
            cpl.CoupledExecutionPolicy(lane_capacity=2, max_working_set_bytes=1)
        )


@pytest.mark.parametrize(
    ("capacity", "error"),
    [(0, ValueError), (65, ValueError), (True, TypeError)],
    ids=["empty", "above-bucket-range", "boolean"],
)
def test_lane_capacity_outside_the_bucket_range_is_refused(
    capacity: int, error: type[Exception]
) -> None:
    with pytest.raises(error):
        cpl.CoupledExecutionPolicy(lane_capacity=capacity)


def test_caller_state_and_prepared_lanes_stay_reusable(
    lanes: cpl.PreparedCoupledProblem,
) -> None:
    state = _state(lanes, 7)
    copies = jax.tree.map(np.array, state)
    system, _ = lanes.linear_system()
    first = (lanes.residual(state), system.operator.mv(state))
    second = (lanes.residual(state), system.operator.mv(state))
    for value, copy in zip(jax.tree.leaves(state), jax.tree.leaves(copies), strict=True):
        assert not value.is_deleted()
        np.testing.assert_array_equal(np.asarray(value), copy)
    for once, again in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
        np.testing.assert_array_equal(np.asarray(once), np.asarray(again))
    worksets = lanes.worksets
    assert worksets is not None
    assert all(
        not lane.is_deleted() for group in worksets.components for lane in group.lanes
    )


def test_runtime_arguments_outside_the_prepared_signature_are_refused(
    lanes: cpl.PreparedCoupledProblem,
) -> None:
    arguments = {name: {"scale": jnp.ones((2,))} for name in _INTERIOR}
    with pytest.raises(ValueError, match="prepared lane signature"):
        lanes.residual(lanes.state_space.zeros(), arguments)


def test_lanes_on_an_execution_group_match_single_device_reference(
    chain: _Chain, reference: cpl.PreparedCoupledProblem
) -> None:
    devices = jax.device_count()
    if devices < 2:
        pytest.skip(
            "Run with XLA_FLAGS=--xla_force_host_platform_device_count=4 to exercise "
            "lanes placed on an execution group."
        )
    from phydrax._execution_runtime import ExecutionRuntime

    group = ExecutionRuntime.current().root_group.spec
    prepared = chain.prepare(
        cpl.CoupledExecutionPolicy(lane_capacity=devices, execution_group=group)
    )
    state = _state(reference, 11)
    np.testing.assert_allclose(
        prepared.row_space.flatten(prepared.residual(state)),
        reference.row_space.flatten(reference.residual(state)),
        rtol=1e-12,
        atol=1e-12,
    )
    solution = cpl.solve_coupled_problem(prepared, policy=_policy())
    expected = cpl.solve_coupled_problem(reference, policy=_policy())
    assert bool(solution.accepted)
    np.testing.assert_allclose(
        prepared.state_space.flatten(solution.state),
        reference.state_space.flatten(expected.state),
        rtol=1e-10,
        atol=1e-12,
    )
