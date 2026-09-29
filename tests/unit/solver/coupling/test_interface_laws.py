#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Numerical-flux, integral-port, and field-transfer coupling laws.

Every reference is a closed-form field evaluated on the host; its sources are
exact automatic derivatives of that field, never the discrete system. Flux
laws use two fields that jump across ``x = 1`` exactly as the declared
interface physics requires; the port law uses a one-dimensional field closed
by a resistor and a voltage source; the transfer law uses two co-located
continua that exchange ``alpha (u_source - u_target)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import assert_never, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.system_modeling import (
    AcausalSystem,
    ConnectionSet,
    Connector,
    ConnectorType,
    ConnectorVariable,
)
from tests.unit.solver.coupling._cases import (
    build_region,
    dense_policy,
    interface_binding,
    ManufacturedField,
    Method,
    nodal_error,
    observed_rate,
    plate_cover,
    Region,
    RegionSpec,
)


cpl = phx.solver.coupling

type PointField = Callable[[Array], Array]
type Profile = Callable[[Array], Array]


# --- Manufactured fields ------------------------------------------------------------------


def _negative_laplacian(value: PointField, /) -> PointField:
    """Exact ``-Laplace(value)`` by automatic differentiation of the closed form."""
    hessian = jax.hessian(value)

    def source(points: Array) -> Array:
        flat = points.reshape((-1, 2))
        laplacian = jax.vmap(lambda point: jnp.trace(hessian(point)))(flat)
        return -laplacian.reshape(points.shape[:-1])

    return source


def _flux_moment(value: PointField, /) -> float:
    """``int_0^1 du/dx(1, y) y (1 - y) dy`` by a 16-point Gauss rule on the host."""
    nodes, weights = np.polynomial.legendre.leggauss(16)
    heights = 0.5 * (nodes + 1.0)
    points = jnp.stack((jnp.ones(16), jnp.asarray(heights)), axis=-1)
    derivative = jax.vmap(jax.grad(value))(points)[:, 0]
    return float(
        np.sum(0.5 * weights * np.asarray(derivative) * heights * (1.0 - heights))
    )


def _manufactured(
    field_id: str, value: PointField, source: PointField, /
) -> ManufacturedField:
    return ManufacturedField(field_id, value, source, _flux_moment(value), None)


def _base(points: Array) -> Array:
    x, y = points[..., 0], points[..., 1]
    return 1.5 + 0.25 * jnp.exp(x) * jnp.sin(y) + 0.1 * x**2 * y


def _jump(heights: Array) -> Array:
    return 0.2 + 0.1 * jnp.cos(jnp.pi * heights)


@dataclass(frozen=True, slots=True)
class JumpPhysics:
    """Heat leaving the minus side, ``Q(u_minus, u_plus)``, of one interface law."""

    physics_id: str
    leaving: Callable[[Array, Array], Array]


def _conductance_physics(conductance: float, /) -> JumpPhysics:
    return JumpPhysics(
        f"conductance-{conductance}", lambda minus, plus: conductance * (minus - plus)
    )


def _radiation_physics(emissivities: tuple[float, float], sigma: float, /) -> JumpPhysics:
    exchange = sigma / (1.0 / emissivities[0] + 1.0 / emissivities[1] - 1.0)
    return JumpPhysics("radiation", lambda minus, plus: exchange * (minus**4 - plus**4))


def jump_fields(physics: JumpPhysics, /) -> tuple[ManufacturedField, ManufacturedField]:
    """Minus/plus fields whose jump and flux across ``x = 1`` obey ``physics``.

    ``u_minus = g + (x - 1) (-Q(y) - dg/dx(1, y))`` has conormal flux
    ``du_minus/dx(1, y) = -Q(y)`` and trace ``g(1, y)``; ``u_plus = u_minus -
    Delta(y)`` carries the same flux (continuity) and the declared jump, where
    ``Q(y) = physics(g(1, y), g(1, y) - Delta(y))`` is the heat that leaves the
    minus side.
    """
    slope = jax.vmap(jax.grad(_base))

    def leaving(heights: Array) -> Array:
        trace = _base(jnp.stack((jnp.ones_like(heights), heights), axis=-1))
        return physics.leaving(trace, trace - _jump(heights))

    def minus(points: Array) -> Array:
        heights = points[..., 1]
        interface = jnp.stack((jnp.ones_like(heights), heights), axis=-1)
        gradient = slope(interface.reshape((-1, 2)))[:, 0].reshape(heights.shape)
        return _base(points) + (points[..., 0] - 1.0) * (-leaving(heights) - gradient)

    def plus(points: Array) -> Array:
        return minus(points) - _jump(points[..., 1])

    return (
        _manufactured(f"minus-{physics.physics_id}", minus, _negative_laplacian(minus)),
        _manufactured(f"plus-{physics.physics_id}", plus, _negative_laplacian(plus)),
    )


def _jump_l2() -> float:
    """``||Delta||_{L2(0, 1)}`` of the declared jump, by a 16-point Gauss rule."""
    nodes, weights = np.polynomial.legendre.leggauss(16)
    heights = 0.5 * (nodes + 1.0)
    return float(
        np.sqrt(np.sum(0.5 * weights * np.asarray(_jump(jnp.asarray(heights))) ** 2))
    )


# --- Conservative numerical flux ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FluxProblem:
    left: Region
    right: Region
    prepared: cpl.PreparedCoupledProblem


def flux_problem(
    physics: JumpPhysics,
    flux: cpl.AbstractInterfaceFlux,
    plus_method: Method,
    cells: tuple[int, int],
    /,
) -> FluxProblem:
    minus_field, plus_field = jump_fields(physics)
    left = build_region(RegionSpec("left", "fe", 0.0, 1.0, cells[0], 1), minus_field)
    right = build_region(
        RegionSpec("right", plus_method, 1.0, 2.0, cells[1], 1), plus_field
    )
    cover = plate_cover()
    binding = interface_binding(
        cover, left.component.field_space_id("u"), right.component.field_space_id("u")
    )
    law = cpl.ConservativeFluxLaw(
        "contact",
        binding,
        (
            cpl.TransmissionSide("left", "left", "u", left.interface),
            cpl.TransmissionSide("right", "right", "u", right.interface),
        ),
        flux,
    )
    plan = cpl.CoupledProblemPlan(
        "imperfect-contact",
        components=(left.component, right.component),
        bindings=(binding,),
        laws=(law,),
    )
    return FluxProblem(
        left, right, cpl.prepare_coupled_problem(plan, interface_owners=(cover,))
    )


@pytest.mark.parametrize("plus_method", ["fe", "vem"], ids=["fe-fe", "fe-vem"])
def test_conductance_flux_converges_to_the_contact_jump(plus_method: Method) -> None:
    physics = _conductance_physics(2.0)
    minus_field, plus_field = jump_fields(physics)
    sizes, errors = [], []
    solution = None
    for cells in ((4, 6), (8, 12)):
        problem = flux_problem(physics, cpl.InterfaceConductance(2.0), plus_method, cells)
        assert problem.prepared.execution == "linear"
        solution = cpl.solve_coupled_problem(problem.prepared, policy=dense_policy())
        assert bool(solution.native_successful) and bool(solution.accepted)
        sizes.append(1.0 / cells[0])
        errors.append(
            max(
                nodal_error(problem.left, solution.field("left", "u"), minus_field),
                nodal_error(problem.right, solution.field("right", "u"), plus_field),
            )
        )
    # P1 traces and a conductance-coupled interface converge at O(h^2).
    assert observed_rate(np.asarray(sizes), np.asarray(errors)) > 1.8
    assert solution is not None
    report = solution.interface("contact")
    # The one shared density leaves the minus side exactly as it enters the plus side.
    assert float(report.value("flux-conservation")) <= 1.0e-12 * float(
        report.scales[report.names.index("flux-conservation")]
    )
    # The solved traces carry the physically implied jump u_minus - u_plus = Q / h.
    assert float(report.value("trace-jump-l2")) == pytest.approx(_jump_l2(), rel=2.0e-2)


def test_gap_radiation_flux_solves_through_newton() -> None:
    emissivities, sigma = (0.8, 0.6), 1.0
    physics = _radiation_physics(emissivities, sigma)
    minus_field, plus_field = jump_fields(physics)
    flux = cpl.GapRadiation(emissivities, stefan_boltzmann=sigma)
    sizes, errors = [], []
    for cells in ((4, 6), (8, 12)):
        problem = flux_problem(physics, flux, "fe", cells)
        assert problem.prepared.execution == "nonlinear"
        solution = cpl.solve_coupled_problem(problem.prepared, policy=dense_policy())
        assert solution.nonlinear is not None and solution.linear is None
        assert bool(solution.native_successful) and bool(solution.accepted)
        sizes.append(1.0 / cells[0])
        errors.append(
            max(
                nodal_error(problem.left, solution.field("left", "u"), minus_field),
                nodal_error(problem.right, solution.field("right", "u"), plus_field),
            )
        )
    assert observed_rate(np.asarray(sizes), np.asarray(errors)) > 1.8


class _SquaredJump(cpl.AbstractInterfaceFlux):
    """A nonlinear density that misdeclares itself affine."""

    @property
    def affine(self) -> bool:
        return True

    @property
    def trace_degree(self) -> int:
        return 2

    def evaluate(
        self, minus: Array, plus: Array, points: Array, normals: Array, args: object, /
    ) -> Array:
        del points, normals, args
        return (plus - minus) ** 2


def test_flux_law_refuses_a_misdeclared_affine_flux() -> None:
    with pytest.raises(ValueError, match="declared affine"):
        flux_problem(_conductance_physics(1.0), _SquaredJump(), "fe", (2, 3))


# --- Integral port ------------------------------------------------------------------------------

# -u'' = 1 on [0, 1] x [0, 1] with u(0, y) = 0, insulated y = 0 and y = 1 is
# replaced by exact Dirichlet data there; the port x = 1 is closed by a resistor
# R in series with a source E: V = E - R I. With u = beta x - x^2 / 2, V = u(1)
# and I = u'(1) (the flow entering the field) give beta = 2, V = 3/2, I = 1.
RESISTANCE, SOURCE_VOLTAGE = 0.5, 2.0
PORT_POTENTIAL, PORT_FLOW = 1.5, 1.0


def _port_value(points: Array) -> Array:
    x = points[..., 0]
    return 2.0 * x - 0.5 * x**2


def _unit_source(points: Array) -> Array:
    return jnp.ones(points.shape[:-1], dtype=points.dtype)


PORT_FIELD = ManufacturedField("port", _port_value, _unit_source, 1.0 / 6.0, 2)

PIN = ConnectorType.create(
    "electrical-pin",
    (ConnectorVariable("v", "across", "V"), ConnectorVariable("i", "through", "A")),
)
_CONNECTORS = ("electrode", "resistor-p", "resistor-n", "source-p", "source-n", "ground")


def _network(pin: ConnectorType = PIN, /) -> AcausalSystem:
    return AcausalSystem.create(
        [Connector(name, pin) for name in _CONNECTORS],
        [
            ConnectionSet.create(("electrode", "resistor-p")),
            ConnectionSet.create(("resistor-n", "source-p")),
            ConnectionSet.create(("source-n", "ground")),
        ],
    )


def _constitutive() -> tuple[np.ndarray, np.ndarray]:
    """Resistor, voltage source, and ground rows over the connector variables."""
    keys = [(name, variable) for name in _CONNECTORS for variable in ("v", "i")]
    rows: list[tuple[dict[tuple[str, str], float], float]] = [
        (
            {
                ("resistor-p", "v"): 1.0,
                ("resistor-n", "v"): -1.0,
                ("resistor-p", "i"): -RESISTANCE,
            },
            0.0,
        ),
        ({("resistor-p", "i"): 1.0, ("resistor-n", "i"): 1.0}, 0.0),
        ({("source-p", "v"): 1.0, ("source-n", "v"): -1.0}, SOURCE_VOLTAGE),
        ({("source-p", "i"): 1.0, ("source-n", "i"): 1.0}, 0.0),
        ({("ground", "v"): 1.0}, 0.0),
    ]
    matrix = np.zeros((len(rows), len(keys)), dtype=np.float64)
    for index, (coefficients, _) in enumerate(rows):
        for key, value in coefficients.items():
            matrix[index, keys.index(key)] = value
    return matrix, np.asarray([value for _, value in rows], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PortDeclaration:
    """Lumped network and units of one port declaration."""

    system: AcausalSystem
    equations: np.ndarray
    rhs: np.ndarray
    potential_unit: str = "V"


def _resistor_port() -> PortDeclaration:
    equations, rhs = _constitutive()
    return PortDeclaration(_network(), equations, rhs)


def _port_law(region: Region, declaration: PortDeclaration, /) -> cpl.IntegralPortLaw:
    return cpl.IntegralPortLaw(
        "electrode-port",
        cpl.PortSide("electrode", region.spec.name, "u", region.interface),
        declaration.system,
        declaration.equations,
        declaration.rhs,
        potential_unit=declaration.potential_unit,
        flux_unit="A",
    )


@pytest.mark.parametrize("method", ["fe", "vem"])
def test_integral_port_closes_a_field_with_a_resistor(method: Method) -> None:
    region = build_region(RegionSpec("slab", method, 0.0, 1.0, 3, 2), PORT_FIELD)
    law = _port_law(region, _resistor_port())
    plan = cpl.CoupledProblemPlan(
        "electrode", components=(region.component,), bindings=(), laws=(law,)
    )
    prepared = cpl.prepare_coupled_problem(plan)
    evidence = prepared.chart.law("electrode-port").evidence
    assert isinstance(evidence, cpl.IntegralPortEvidence)
    # The network, owned by system modeling, presents V + R I = E at the port.
    np.testing.assert_allclose(
        evidence.port_relation, (1.0, RESISTANCE, SOURCE_VOLTAGE), rtol=1e-12
    )
    assert evidence.port_measure == pytest.approx(1.0, rel=1e-14)
    solution = cpl.solve_coupled_problem(prepared, policy=dense_policy())
    assert bool(solution.native_successful) and bool(solution.accepted)
    (variables,) = solution.law_state("electrode-port")
    # Quadratic fields are exact in P2 finite and k = 2 virtual elements.
    assert nodal_error(region, solution.field("slab", "u"), PORT_FIELD) < 1e-10
    keys = law.variable_keys
    potential = float(variables[keys.index(("electrode", "v"))])
    flow = float(variables[keys.index(("electrode", "i"))])
    assert potential == pytest.approx(PORT_POTENTIAL, abs=1e-10)
    assert flow == pytest.approx(PORT_FLOW, abs=1e-10)
    assert float(variables[keys.index(("resistor-p", "i"))]) == pytest.approx(-PORT_FLOW)
    report = solution.interface("electrode-port")
    assert float(report.value("port-potential-deviation")) < 1e-10


def _two_across() -> AcausalSystem:
    bad = ConnectorType.create(
        "double-potential",
        (ConnectorVariable("v", "across", "V"), ConnectorVariable("w", "across", "V")),
    )
    return AcausalSystem.create(
        [Connector("electrode", bad), Connector("mirror", bad)],
        [ConnectionSet.create(("electrode", "mirror"))],
    )


def _underdetermined() -> PortDeclaration:
    equations, rhs = _constitutive()
    return PortDeclaration(_network(), equations[:4], rhs[:4])


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            PortDeclaration(_two_across(), np.zeros((1, 4)), np.zeros((1,))),
            "exactly one across and one through",
        ),
        (PortDeclaration(_network(), *_constitutive(), "K"), "has unit 'V'"),
        (_underdetermined(), "supplies 10 equations"),
    ],
    ids=["undeclared-kinds", "unit-mismatch", "underdetermined-network"],
)
def test_integral_port_refuses_undeclared_connector_semantics(
    declaration: PortDeclaration, message: str
) -> None:
    region = build_region(RegionSpec("slab", "fe", 0.0, 1.0, 2, 1), PORT_FIELD)
    with pytest.raises(ValueError, match=message):
        _port_law(region, declaration)


# --- Volume field transfer -----------------------------------------------------------------------

EXCHANGE = 4.0


def _host_value(points: Array) -> Array:
    x, y = points[..., 0], points[..., 1]
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) + x


def _guest_value(points: Array) -> Array:
    x, y = points[..., 0], points[..., 1]
    return x * y + 0.5 * jnp.cos(x)


def _exchange_fields() -> tuple[ManufacturedField, ManufacturedField]:
    """Sources of ``-Laplace(u_s) + a (u_s - u_t) = f_s``, ``-Laplace(u_t) - a (u_s - u_t) = f_t``."""
    host_laplacian = _negative_laplacian(_host_value)
    guest_laplacian = _negative_laplacian(_guest_value)

    def host_source(points: Array) -> Array:
        rate = EXCHANGE * (_host_value(points) - _guest_value(points))
        return host_laplacian(points) + rate

    def guest_source(points: Array) -> Array:
        rate = EXCHANGE * (_host_value(points) - _guest_value(points))
        return guest_laplacian(points) - rate

    return (
        _manufactured("host", _host_value, host_source),
        _manufactured("guest", _guest_value, guest_source),
    )


type TransferRoute = Literal["field-query", "l2-projection"]


def _bound_transfer(
    host: phx.discretization.FiniteElementDiscretization,
    guest: phx.discretization.FiniteElementDiscretization,
    primal: Callable[[Array], Array],
    pullback: Callable[[Array], Array],
    properties: phx.discretization.TransferProperties,
    /,
) -> phx.discretization.FieldTransfer:
    """A native relation bound to the two components' exact field spaces."""
    source, target = host.field_spaces[0], guest.field_spaces[0]
    return phx.discretization.FieldTransfer(
        source,
        target,
        phx.linalg.FunctionLinearOperator(
            primal,
            source=source.vector_space,
            target=target.vector_space,
            transpose_action=pullback,
        ),
        dual_pullback_operator=phx.linalg.FunctionLinearOperator(
            pullback,
            source=target.vector_space,
            target=source.vector_space,
            transpose_action=primal,
        ),
        properties=properties,
    )


def _field_transfer(
    source: Region, target: Region, route: TransferRoute, /
) -> phx.discretization.FieldTransfer:
    host = source.problem.discretization
    guest = target.problem.discretization
    assert isinstance(host, phx.discretization.FiniteElementDiscretization)
    assert isinstance(guest, phx.discretization.FiniteElementDiscretization)
    match route:
        case "field-query":
            # The prepared FE field query at the guest's Lagrange nodes is the
            # nodal interpolant; it reproduces constants by partition of unity.
            query = phx.discretization.fem.prepare_finite_element_field_reconstruction(
                host, "u"
            ).prepare_query(np.asarray(guest.dof_maps[0].dof_coordinates))
            return _bound_transfer(
                host,
                guest,
                query.apply,
                query.transpose,
                phx.discretization.TransferProperties(constant_preserving=True),
            )
        case "l2-projection":
            from phydrax.geometry import (
                CommonRefinementPolicy,
                prepare_common_refinement,
            )

            refinement = prepare_common_refinement(
                host.mesh,
                guest.mesh,
                policy=CommonRefinementPolicy(overlap_simplices=True),
            )
            projection = phx.discretization.prepare_l2_projection_transfer(
                host,
                phx.discretization.prepare_l2_projection_target(guest, field_name="u"),
                refinement,
                field_name="u",
            )
            return _bound_transfer(
                host,
                guest,
                projection.apply,
                projection.pullback,
                phx.discretization.TransferProperties(
                    constant_preserving=projection.preserves_constants,
                    conservative=projection.conservative,
                ),
            )
        case _:
            assert_never(route)


def _transfer_problem(
    cells: tuple[int, int], route: TransferRoute, /
) -> tuple[Region, Region, cpl.FieldTransferLaw, cpl.PreparedCoupledProblem]:
    host_field, guest_field = _exchange_fields()
    host = build_region(
        RegionSpec("host", "fe", 0.0, 1.0, cells[0], 1, dirichlet="whole-boundary"),
        host_field,
    )
    guest = build_region(
        RegionSpec("guest", "fe", 0.0, 1.0, cells[1], 1, dirichlet="whole-boundary"),
        guest_field,
    )
    guest_space = guest.problem.discretization
    assert isinstance(guest_space, phx.discretization.FiniteElementDiscretization)
    law = cpl.FieldTransferLaw(
        "exchange",
        cpl.ContributionEndpoint("host", "u"),
        cpl.ContributionEndpoint("guest", "u"),
        _field_transfer(host, guest, route),
        guest_space.mass,
        exchange=EXCHANGE,
    )
    plan = cpl.CoupledProblemPlan(
        "two-continua",
        components=(host.component, guest.component),
        bindings=(),
        laws=(law,),
    )
    return host, guest, law, cpl.prepare_coupled_problem(plan)


@pytest.mark.parametrize(
    "route",
    [
        "field-query",
        pytest.param(
            "l2-projection",
            marks=pytest.mark.skipif(
                not meshcore_available(), reason="Common refinement requires meshcore."
            ),
        ),
    ],
)
def test_field_transfer_exchange_converges_between_nonmatching_meshes(
    route: TransferRoute,
) -> None:
    host_field, guest_field = _exchange_fields()
    sizes, errors = [], []
    for cells in ((5, 4), (10, 8)):
        host, guest, _, prepared = _transfer_problem(cells, route)
        evidence = prepared.chart.law("exchange").evidence
        assert isinstance(evidence, cpl.FieldTransferEvidence)
        assert evidence.constant_preserving and evidence.constant_defect < 1e-10
        assert evidence.dual_pairing_residual < 1e-10
        # The guest mass integrates the unit constant over the unit square.
        assert evidence.target_measure == pytest.approx(1.0, rel=1e-12)
        solution = cpl.solve_coupled_problem(prepared, policy=dense_policy())
        assert bool(solution.native_successful) and bool(solution.accepted)
        report = solution.interface("exchange")
        assert float(report.value("exchange-dissipation")) >= 0.0
        sizes.append(1.0 / cells[0])
        errors.append(
            max(
                nodal_error(host, solution.field("host", "u"), host_field),
                nodal_error(guest, solution.field("guest", "u"), guest_field),
            )
        )
    # P1 continua with an O(h^2) transfer converge at O(h^2).
    assert observed_rate(np.asarray(sizes), np.asarray(errors)) > 1.8


def test_field_transfer_refuses_unbound_or_nonconservative_transfers() -> None:
    host, guest, law, _ = _transfer_problem((3, 2), "field-query")
    swapped = cpl.FieldTransferLaw(
        "exchange",
        cpl.ContributionEndpoint("guest", "u"),
        cpl.ContributionEndpoint("host", "u"),
        law.transfer,
        law.measure,
        exchange=EXCHANGE,
    )
    plan = cpl.CoupledProblemPlan(
        "swapped",
        components=(host.component, guest.component),
        bindings=(),
        laws=(swapped,),
    )
    with pytest.raises(ValueError, match="maps field space"):
        cpl.prepare_coupled_problem(plan)
    transfer = law.transfer
    unclaimed = phx.discretization.FieldTransfer(
        transfer.source,
        transfer.target,
        transfer.primal_operator,
        dual_pullback_operator=transfer.dual_pullback_operator,
    )
    with pytest.raises(ValueError, match="constant preservation"):
        cpl.FieldTransferLaw(
            "exchange",
            cpl.ContributionEndpoint("host", "u"),
            cpl.ContributionEndpoint("guest", "u"),
            unclaimed,
            law.measure,
            exchange=EXCHANGE,
        )
