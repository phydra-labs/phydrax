#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical inventory functionals of coupling ports and their conservation."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx


cpl = phx.solver.coupling
_METER = phx.units.METER
_PER_LENGTH = phx.units.derived_unit("J/m", ((phx.units.JOULE, 1), (phx.units.METER, -1)))
_ENTHALPY = phx.units.derived_unit("J", ((phx.units.JOULE, 1),))
# P1 nodes and finite-volume cell edges on [0, 1]; the grids do not match.
_NODES = np.asarray((0.0, 0.1, 0.35, 0.6, 1.0))
_EDGES = np.asarray((0.0, 0.25, 0.5, 0.8, 1.0))


def _field(
    name: str,
    representation: phx.discretization.FieldRepresentation,
    count: int,
    /,
    *,
    components: tuple[int, ...] = (),
) -> phx.discretization.DiscreteFieldSpace:
    layout = phx.discretization.EntityDofLayout(
        f"{name}/entities", count, count, component_shape=components
    )
    space = phx.linalg.ArraySpace(
        (layout.size,), dtype=jnp.float64, space_id=f"{name}/coordinates"
    )
    return phx.discretization.DiscreteFieldSpace(
        name, f"{name}-support", layout, space, representation=representation
    )


def _p1_basis_integrals(nodes: np.ndarray, /) -> np.ndarray:
    """Exact integrals of the piecewise-linear hat functions."""
    widths = np.diff(nodes)
    weights = np.zeros_like(nodes)
    weights[:-1] += 0.5 * widths
    weights[1:] += 0.5 * widths
    return weights


def _hat_cell_averages(nodes: np.ndarray, edges: np.ndarray, /) -> np.ndarray:
    """Exact cell averages of every hat function on a nonmatching cell grid."""
    breaks = np.union1d(nodes, edges)
    matrix = np.zeros((edges.size - 1, nodes.size))
    for left, right in zip(breaks[:-1], breaks[1:], strict=True):
        cell = np.searchsorted(edges, 0.5 * (left + right)) - 1
        for node in range(nodes.size):
            unit = np.eye(nodes.size)[node]
            values = np.interp((left, right), nodes, unit)
            matrix[cell, node] += 0.5 * (right - left) * (values[0] + values[1])
    return matrix / np.diff(edges)[:, None]


def _density_port(
    port_id: str,
    direction: phx.solver.coupling.CouplingDirection,
    field: phx.discretization.DiscreteFieldSpace,
    weights: np.ndarray,
    /,
) -> phx.solver.coupling.CouplingPort:
    measure = phx.discretization.DiscreteMeasure(
        f"{field.name}-measure",
        field.support_id,
        f"{field.name}/entities",
        weights,
    )
    return cpl.CouplingPort(
        port_id,
        direction,
        field.vector_space,
        field_space=field,
        quantity=cpl.CouplingQuantity("enthalpy_per_length", _PER_LENGTH),
        measurement=cpl.CouplingMeasurement.from_measure(
            measure, field.vector_space, _METER
        ),
        temporal_kind="interval_integral",
        reference_scale=1.0,
    )


def _matrix_free_transfer(
    source: phx.discretization.DiscreteFieldSpace,
    target: phx.discretization.DiscreteFieldSpace,
    matrix: np.ndarray,
    name: str,
    /,
) -> phx.discretization.FieldTransfer:
    values = jnp.asarray(matrix)

    def primal(vector: Array) -> Array:
        return values @ vector

    def pullback(covector: Array) -> Array:
        return values.T @ covector

    return phx.discretization.FieldTransfer(
        source,
        target,
        # No transpose action is supplied: the certificate must transpose the
        # applied action itself rather than trust a declared matrix.
        phx.linalg.FunctionLinearOperator(
            primal,
            source=source.vector_space,
            target=target.vector_space,
            operator_id=f"{name}/primal",
        ),
        dual_pullback_operator=phx.linalg.FunctionLinearOperator(
            pullback,
            source=target.vector_space,
            target=source.vector_space,
            operator_id=f"{name}/pullback",
        ),
        properties=phx.discretization.TransferProperties(conservative=True),
    )


def _capabilities() -> phx.solver.coupling.CouplingSubsystemCapabilities:
    return cpl.CouplingSubsystemCapabilities(
        jit=True, differentiable=True, deterministic_replay=True, fixed_topology=True
    )


def _fe_to_fv_window(
    matrix: np.ndarray, profile: Callable[[np.ndarray], np.ndarray], /
) -> phx.solver.coupling.CouplingWindowResult:
    fe = _field("p1-nodal", "basis_coefficient", _NODES.size)
    fv = _field("fv-cells", "cell_average", _EDGES.size - 1)
    source = _density_port("fe-heat", "output", fe, _p1_basis_integrals(_NODES))
    target = _density_port("fv-heat", "input", fv, np.diff(_EDGES))
    amount = jnp.asarray(profile(_NODES))

    def produce(
        window: phx.solver.coupling.CouplingWindow,
        state: Array,
        inputs: tuple[Array, ...],
        args: None,
    ) -> phx.solver.coupling.CouplingSubsystemResult:
        del inputs, args
        return cpl.CouplingSubsystemResult(
            state, (amount * window.size,), successful=True, status=0
        )

    def consume(
        window: phx.solver.coupling.CouplingWindow,
        state: Array,
        inputs: tuple[Array, ...],
        args: None,
    ) -> phx.solver.coupling.CouplingSubsystemResult:
        del window, args
        return cpl.CouplingSubsystemResult(
            state + inputs[0], (), successful=True, status=0
        )

    graph = cpl.CouplingGraph(
        (
            cpl.CallableCouplingSubsystem(
                produce,
                subsystem_id="fe",
                output_ports=(source,),
                capabilities=_capabilities(),
            ),
            cpl.CallableCouplingSubsystem(
                consume,
                subsystem_id="fv",
                input_ports=(target,),
                capabilities=_capabilities(),
            ),
        ),
        (
            cpl.CouplingExchange(
                "heat",
                "fe-heat",
                "fv-heat",
                transfer=_matrix_free_transfer(fe, fv, matrix, "fe-to-fv"),
                requirement=cpl.CouplingTransferRequirement(conservative=True),
                temporal=cpl.CouplingTemporalConversion("window-integral"),
            ),
        ),
    )
    prepared = cpl.prepare_coupling(
        graph,
        (jnp.zeros(1), jnp.zeros(_EDGES.size - 1)),
        (jnp.zeros(_EDGES.size - 1),),
        policy=cpl.ExplicitCouplingPolicy(
            cpl.CouplingSweep("gauss-seidel", subsystem_order=("fe", "fv"))
        ),
    )
    return cpl.advance_coupling_window(prepared, prepared.reference_state, 2.0)


def test_nodal_basis_density_inventory_is_the_exact_field_integral() -> None:
    field = _field("p1-nodal", "basis_coefficient", _NODES.size)
    port = _density_port("nodal", "output", field, _p1_basis_integrals(_NODES))
    measurement = port.measurement
    assert measurement is not None
    assert measurement.representation == "density"

    inventory = measurement.inventory(jnp.asarray(2.0 + 3.0 * _NODES))

    # The P1 interpolant of 2 + 3x is exact, and its integral over [0, 1] is 3.5.
    np.testing.assert_allclose(inventory, [3.5], rtol=1e-14)


def test_flux_moment_inventory_sums_each_component_exactly_once() -> None:
    field = _field("face-moments", "flux_moment", 3, components=(2,))
    measurement = cpl.CouplingMeasurement.extensive(
        field.vector_space,
        field.support_id,
        provenance_id="face-integrated-flux",
        component_ids=("x", "y"),
    )
    port = cpl.CouplingPort(
        "faces",
        "output",
        field.vector_space,
        field_space=field,
        measurement=measurement,
        frame="cartesian-xy",
        reference_scale=1.0,
    )
    moments = np.asarray(((1.5, -2.0), (0.25, 4.0), (-0.75, 1.0)))

    inventory = measurement.inventory(jnp.asarray(moments.reshape(-1)))

    assert port.measurement is measurement
    np.testing.assert_allclose(inventory, moments.sum(axis=0), rtol=1e-15)
    with pytest.raises(ValueError, match="component frame"):
        cpl.CouplingPort(
            "scalar-faces",
            "output",
            field.vector_space,
            field_space=field,
            measurement=measurement,
            reference_scale=1.0,
        )


def test_density_and_extensive_storage_refuse_double_weighting() -> None:
    integrals = _field("cell-integrals", "cell_integral", 4)
    weights = jnp.asarray((0.25, 0.25, 0.3, 0.2))
    measure = phx.discretization.DiscreteMeasure(
        "cells", integrals.support_id, "cell-integrals/entities", weights
    )
    density = cpl.CouplingMeasurement.from_measure(
        measure, integrals.vector_space, _METER
    )
    with pytest.raises(ValueError, match="weighting the coordinates again"):
        cpl.CouplingPort(
            "integrals",
            "output",
            integrals.vector_space,
            field_space=integrals,
            measurement=density,
            reference_scale=1.0,
        )
    reweighted = phx.linalg.FunctionLinearOperator(
        lambda value: jnp.sum(weights * value, keepdims=True),
        source=integrals.vector_space,
        target=phx.linalg.ArraySpace((1,), dtype=jnp.float64),
        operator_id="cell-integral-reweighted",
    )
    with pytest.raises(ValueError, match="double count"):
        cpl.CouplingMeasurement(
            reweighted,
            phx.units.ONE,
            representation="extensive",
            support_id=integrals.support_id,
            provenance_id="reweighted-integrals",
            normalization="counting",
        )
    averages = _field("cell-averages", "cell_average", 4)
    extensive = cpl.CouplingMeasurement.extensive(
        averages.vector_space, averages.support_id, provenance_id="cell-sum"
    )
    with pytest.raises(ValueError, match="weighting the coordinates again"):
        cpl.CouplingPort(
            "averages",
            "output",
            averages.vector_space,
            field_space=averages,
            measurement=extensive,
            reference_scale=1.0,
        )


def test_nonmatching_conservative_projection_balances_the_true_functional() -> None:
    # Cell averages of the P1 interpolant satisfy sum_c |c| (P u)_c = int u exactly.
    matrix = _hat_cell_averages(_NODES, _EDGES)

    result = _fe_to_fv_window(matrix, lambda x: 1.0 + 4.0 * x)

    assert bool(result.successful)
    # Independent reference: 2 * int_0^1 (1 + 4x) dx = 6 J.
    np.testing.assert_allclose(result.accepted_exchange_budget, [[-6.0, 6.0]], rtol=1e-13)
    np.testing.assert_allclose(
        jnp.sum(jnp.diff(_EDGES) * result.accepted_state.participant_states[1]),
        6.0,
        rtol=1e-13,
    )


def test_false_conservation_claim_is_refused_by_the_dual_certificate() -> None:
    # Pointwise nodal interpolation at cell centers preserves constants but not
    # the integral on this nonmatching grid.
    centers = 0.5 * (_EDGES[1:] + _EDGES[:-1])
    matrix = np.stack(
        tuple(np.interp(centers, _NODES, row) for row in np.eye(_NODES.size)), axis=1
    )

    with pytest.raises(ValueError, match="L_target P = L_source"):
        _fe_to_fv_window(matrix, lambda x: 1.0 + 4.0 * x)


def test_probability_functional_cannot_measure_a_whole_window_amount() -> None:
    field = _field("mean", "cell_average", 2)
    mean = phx.linalg.FunctionLinearOperator(
        lambda value: jnp.mean(value, keepdims=True),
        source=field.vector_space,
        target=phx.linalg.ArraySpace((1,), dtype=jnp.float64),
        operator_id="cell-mean",
    )
    measurement = cpl.CouplingMeasurement(
        mean,
        phx.units.ONE,
        representation="functional",
        support_id=field.support_id,
        provenance_id="cell-mean",
        normalization="probability",
    )

    with pytest.raises(ValueError, match="averages"):
        cpl.CouplingPort(
            "mean",
            "output",
            field.vector_space,
            field_space=field,
            quantity=cpl.CouplingQuantity("enthalpy", _ENTHALPY),
            measurement=measurement,
            temporal_kind="interval_integral",
            reference_scale=1.0,
        )
