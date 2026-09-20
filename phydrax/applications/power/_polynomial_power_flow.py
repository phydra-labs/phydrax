#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Exact fixed-mode rectangular power-flow lowering and qualified root replay."""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...algebraic._isolated import (
    IsolatedPolynomialRootProblem,
    plan_isolated_roots,
    PolynomialRootResult,
    prepare_isolated_roots,
    PreparedIsolatedRootSolve,
    refresh_isolated_roots,
    solve_prepared_isolated_roots,
)
from ...algebraic._system import PolynomialScaling, SparsePolynomialSystem
from ...backends.homotopy_continuation import (
    HomotopyContinuationPolicy,
    HomotopyContinuationProvider,
)
from ...sparse import EdgeRelation
from ._network import CompiledNetwork


_SUPPORTED_MODES = frozenset(("pq", "pv", "reference", "q_min", "q_max"))


class FixedModePowerFlowPolynomial(StrictModule):
    """One exact real polynomial lowering of a declared rectangular fixed mode."""

    compiled: CompiledNetwork
    injections: Array
    system: SparsePolynomialSystem
    modes: tuple[str, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    polynomial_id: str = eqx.field(static=True)

    @property
    def bus_count(self) -> int:
        return len(self.compiled.network.buses)

    def coordinates(self, voltage: ArrayLike, /) -> Array:
        value = jnp.asarray(voltage)
        if value.shape != (self.bus_count,):
            raise ValueError("Voltage must contain one complex scalar per bus.")
        return jnp.concatenate((value.real, value.imag))

    def voltage(self, coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinates)
        if value.shape != (2 * self.bus_count,):
            raise ValueError("Rectangular coordinates must contain 2 * bus_count values.")
        return value[: self.bus_count] + 1j * value[self.bus_count :]

    def native_residual(self, voltage: ArrayLike, /) -> Array:
        """Replay the original fixed-mode physical equations, independent of lowering."""

        value = jnp.asarray(voltage)
        if value.shape != (self.bus_count,):
            raise ValueError("Voltage must contain one complex scalar per bus.")
        mismatch = self.compiled.bus_power(value) - self.injections
        reference = jnp.asarray([mode == "reference" for mode in self.modes])
        pv = jnp.asarray([mode == "pv" for mode in self.modes])
        target = self.compiled.initial_voltage
        real = jnp.where(reference, value.real - target.real, mismatch.real)
        imag = jnp.where(
            reference,
            value.imag - target.imag,
            jnp.where(
                pv,
                jnp.abs(value) ** 2 - self.compiled.voltage_setpoints**2,
                mismatch.imag,
            ),
        )
        return jnp.concatenate((real, imag))


class PreparedFixedModePowerFlowRoots(StrictModule):
    polynomial: FixedModePowerFlowPolynomial
    isolated: PreparedIsolatedRootSolve
    physical_residual_tolerance: float = eqx.field(static=True)
    mode_tolerance: float = eqx.field(static=True)
    operational_tolerance: float = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)


class PowerFlowRootCandidateEvidence(StrictModule):
    root_index: int = eqx.field(static=True)
    cluster_path_count: int = eqx.field(static=True)
    voltage: Array
    polynomial_residual_norm: float = eqx.field(static=True)
    original_residual_norm: float = eqx.field(static=True)
    imaginary_coordinate_norm: float = eqx.field(static=True)
    mode_complementarity_violation: float = eqx.field(static=True)
    generator_limit_violation: float = eqx.field(static=True)
    voltage_limit_violation: float = eqx.field(static=True)
    branch_limit_violation: float = eqx.field(static=True)
    near_real: bool = eqx.field(static=True)
    original_residual_accepted: bool = eqx.field(static=True)
    mode_complementarity_accepted: bool = eqx.field(static=True)
    generator_limits_accepted: bool = eqx.field(static=True)
    voltage_limits_accepted: bool = eqx.field(static=True)
    branch_limits_accepted: bool = eqx.field(static=True)
    physically_accepted: bool = eqx.field(static=True)


class FixedModePowerFlowRootSet(StrictModule):
    algebraic: PolynomialRootResult
    voltage: Array
    candidate_mask: Array
    near_real_mask: Array
    original_residual_mask: Array
    mode_complementarity_mask: Array
    generator_limit_mask: Array
    voltage_limit_mask: Array
    branch_limit_mask: Array
    operational_mask: Array
    physical_mask: Array
    candidates: tuple[PowerFlowRootCandidateEvidence, ...]
    polynomial_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)
    claim: str = eqx.field(
        static=True,
        default="fixed-mode-physical-replay-of-provider-candidates-not-all-power-flow-roots",
    )

    @property
    def physical_root_count(self) -> Array:
        return jnp.sum(self.physical_mask, dtype=jnp.int32)


def _modes(
    compiled: CompiledNetwork, modes: tuple[str, ...] | None, /
) -> tuple[str, ...]:
    selected = (
        compiled.control_modes if modes is None else tuple(str(mode) for mode in modes)
    )
    count = len(compiled.network.buses)
    if len(selected) != count or any(mode not in _SUPPORTED_MODES for mode in selected):
        raise ValueError("One supported fixed mode is required for every bus.")
    references = {index for index, mode in enumerate(selected) if mode == "reference"}
    if references != set(compiled.references):
        raise ValueError("Fixed-mode references must match compiled island references.")
    if any(
        mode in ("q_min", "q_max") and not compiled.generators_at_bus[index]
        for index, mode in enumerate(selected)
    ):
        raise ValueError("Reactive-bound modes require an in-service generator.")
    return selected


def _fixed_injections(
    compiled: CompiledNetwork,
    modes: tuple[str, ...],
    injections: ArrayLike | None,
    /,
) -> Array:
    count = len(compiled.network.buses)
    if injections is not None:
        values = jnp.asarray(injections)
        if values.shape != (count,):
            raise ValueError("Injections must contain one complex scalar per bus.")
        if not np.all(np.isfinite(np.asarray(values))):
            raise ValueError("Polynomial power-flow injections must be finite.")
        return values
    values = np.asarray(compiled.specified_power, dtype=np.complex128).copy()
    generators = compiled.network.generators
    for bus, mode in enumerate(modes):
        if mode not in ("q_min", "q_max"):
            continue
        indices = compiled.generators_at_bus[bus]
        bound = sum(
            generators[index].q_min if mode == "q_min" else generators[index].q_max
            for index in indices
        )
        if not math.isfinite(bound):
            raise ValueError(f"Mode {mode!r} requires a finite aggregate reactive bound.")
        values[bus] = values[bus].real + 1j * (
            bound - float(compiled.load_power[bus].imag)
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Polynomial power-flow injections must be finite.")
    return jnp.asarray(values)


def _add_term(equation, exponent: tuple[int, ...], coefficient: float) -> None:
    equation[exponent] = equation.get(exponent, 0.0) + float(coefficient)


def _admittance_rows(
    compiled: CompiledNetwork, /
) -> tuple[tuple[tuple[int, complex], ...], ...]:
    count = len(compiled.network.buses)
    rows: list[dict[int, complex]] = [dict() for _ in range(count)]
    relation = compiled.admittance.relation
    if not isinstance(relation, EdgeRelation):
        raise TypeError("Compiled power admittance must use canonical edge storage.")
    for source, target, coefficient, valid in zip(
        np.asarray(relation.source_indices),
        np.asarray(relation.target_indices),
        np.asarray(compiled.admittance.coefficients),
        np.asarray(relation.valid),
        strict=True,
    ):
        if bool(valid):
            row = rows[int(target)]
            index = int(source)
            row[index] = row.get(index, 0j) + complex(coefficient)
    return tuple(tuple(sorted(row.items())) for row in rows)


def _power_terms(
    equations: list[dict[tuple[int, ...], float]],
    admittance_rows: tuple[tuple[tuple[int, complex], ...], ...],
    bus: int,
    /,
) -> None:
    count = len(admittance_rows)
    p_equation, q_equation = equations[bus], equations[count + bus]
    for other, admittance in admittance_rows[bus]:
        conductance = float(admittance.real)
        susceptance = float(admittance.imag)
        ei_ej = [0] * (2 * count)
        ei_ej[bus] += 1
        ei_ej[other] += 1
        fi_fj = [0] * (2 * count)
        fi_fj[count + bus] += 1
        fi_fj[count + other] += 1
        fi_ej = [0] * (2 * count)
        fi_ej[count + bus] += 1
        fi_ej[other] += 1
        ei_fj = [0] * (2 * count)
        ei_fj[bus] += 1
        ei_fj[count + other] += 1
        _add_term(p_equation, tuple(ei_ej), conductance)
        _add_term(p_equation, tuple(fi_fj), conductance)
        _add_term(p_equation, tuple(fi_ej), susceptance)
        _add_term(p_equation, tuple(ei_fj), -susceptance)
        _add_term(q_equation, tuple(fi_ej), conductance)
        _add_term(q_equation, tuple(ei_fj), -conductance)
        _add_term(q_equation, tuple(ei_ej), -susceptance)
        _add_term(q_equation, tuple(fi_fj), -susceptance)


def compile_fixed_mode_power_flow_polynomial(
    compiled: CompiledNetwork,
    injections: ArrayLike | None = None,
    *,
    modes: tuple[str, ...] | None = None,
) -> FixedModePowerFlowPolynomial:
    """Lower one declared mode to exact sparse real rectangular quadratic equations."""

    if not isinstance(compiled, CompiledNetwork):
        raise TypeError("compiled must be a CompiledNetwork.")
    selected = _modes(compiled, modes)
    specified = _fixed_injections(compiled, selected, injections)
    count = len(compiled.network.buses)
    width = 2 * count
    equations: list[dict[tuple[int, ...], float]] = [dict() for _ in range(width)]
    admittance_rows = _admittance_rows(compiled)
    for bus in range(count):
        mode = selected[bus]
        if mode == "reference":
            real_exponent = [0] * width
            real_exponent[bus] = 1
            imaginary_exponent = [0] * width
            imaginary_exponent[count + bus] = 1
            _add_term(equations[bus], tuple(real_exponent), 1.0)
            _add_term(
                equations[bus], (0,) * width, -float(compiled.initial_voltage[bus].real)
            )
            _add_term(equations[count + bus], tuple(imaginary_exponent), 1.0)
            _add_term(
                equations[count + bus],
                (0,) * width,
                -float(compiled.initial_voltage[bus].imag),
            )
            continue
        _power_terms(equations, admittance_rows, bus)
        _add_term(equations[bus], (0,) * width, -float(specified[bus].real))
        if mode == "pv":
            equations[count + bus].clear()
            real_squared = [0] * width
            real_squared[bus] = 2
            imaginary_squared = [0] * width
            imaginary_squared[count + bus] = 2
            _add_term(equations[count + bus], tuple(real_squared), 1.0)
            _add_term(equations[count + bus], tuple(imaginary_squared), 1.0)
            _add_term(
                equations[count + bus],
                (0,) * width,
                -float(compiled.voltage_setpoints[bus] ** 2),
            )
        else:
            _add_term(
                equations[count + bus],
                (0,) * width,
                -float(specified[bus].imag),
            )
    equation_indices = []
    exponents = []
    coefficients = []
    for equation_index, equation in enumerate(equations):
        for exponent in sorted(equation):
            equation_indices.append(equation_index)
            exponents.append(exponent)
            coefficients.append(equation[exponent])
    buses = tuple(bus.id for bus in compiled.network.buses)
    variable_labels = tuple(f"bus:{bus}:e" for bus in buses) + tuple(
        f"bus:{bus}:f" for bus in buses
    )
    equation_labels = tuple(f"bus:{bus}:real" for bus in buses) + tuple(
        f"bus:{bus}:imag-or-voltage" for bus in buses
    )
    system = SparsePolynomialSystem.from_coo(
        variable_labels,
        equation_labels,
        equation_indices,
        exponents,
        jnp.asarray(coefficients),
    )
    structure_id = canonical_fingerprint(
        {
            "kind": "fixed-mode-power-flow-polynomial-structure",
            "support": system.support.support_id,
            "modes": list(selected),
            "buses": list(buses),
            "branches": [branch.id for branch in compiled.network.branches],
        }
    )
    polynomial_id = canonical_fingerprint(
        {
            "kind": "fixed-mode-power-flow-polynomial",
            "structure": structure_id,
            "system": system.system_id,
            "injections": [
                [float(value.real), float(value.imag)] for value in np.asarray(specified)
            ],
        }
    )
    return FixedModePowerFlowPolynomial(
        compiled,
        specified,
        system,
        selected,
        structure_id,
        polynomial_id,
    )


def _positive_tolerance(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def prepare_fixed_mode_power_flow_roots(
    polynomial: FixedModePowerFlowPolynomial,
    provider: HomotopyContinuationProvider,
    /,
    *,
    policy: HomotopyContinuationPolicy | None = None,
    scaling: PolynomialScaling | None = None,
    polynomial_residual_tolerance: float = 1e-8,
    cluster_tolerance: float = 1e-7,
    near_real_absolute_tolerance: float = 1e-8,
    near_real_relative_tolerance: float = 1e-8,
    physical_residual_tolerance: float = 1e-7,
    mode_tolerance: float = 1e-7,
    operational_tolerance: float = 1e-7,
) -> PreparedFixedModePowerFlowRoots:
    if not isinstance(polynomial, FixedModePowerFlowPolynomial):
        raise TypeError("polynomial must be FixedModePowerFlowPolynomial.")
    problem = IsolatedPolynomialRootProblem(polynomial.system, scaling)
    plan = plan_isolated_roots(
        problem,
        provider,
        policy=policy,
        residual_tolerance=polynomial_residual_tolerance,
        cluster_tolerance=cluster_tolerance,
        near_real_absolute_tolerance=near_real_absolute_tolerance,
        near_real_relative_tolerance=near_real_relative_tolerance,
    )
    isolated = prepare_isolated_roots(plan)
    physical = _positive_tolerance(
        physical_residual_tolerance, "physical_residual_tolerance"
    )
    mode = _positive_tolerance(mode_tolerance, "mode_tolerance")
    operational = _positive_tolerance(operational_tolerance, "operational_tolerance")
    identifier = canonical_fingerprint(
        {
            "kind": "prepared-fixed-mode-power-flow-roots",
            "polynomial": polynomial.polynomial_id,
            "isolated": isolated.preparation_id,
            "physical_residual_tolerance": physical,
            "mode_tolerance": mode,
            "operational_tolerance": operational,
        }
    )
    return PreparedFixedModePowerFlowRoots(
        polynomial,
        isolated,
        physical,
        mode,
        operational,
        identifier,
    )


def refresh_fixed_mode_power_flow_roots(
    prepared: PreparedFixedModePowerFlowRoots,
    polynomial_or_injections: FixedModePowerFlowPolynomial | ArrayLike,
    /,
) -> PreparedFixedModePowerFlowRoots:
    if not isinstance(prepared, PreparedFixedModePowerFlowRoots):
        raise TypeError("prepared must be PreparedFixedModePowerFlowRoots.")
    polynomial = (
        polynomial_or_injections
        if isinstance(polynomial_or_injections, FixedModePowerFlowPolynomial)
        else compile_fixed_mode_power_flow_polynomial(
            prepared.polynomial.compiled,
            polynomial_or_injections,
            modes=prepared.polynomial.modes,
        )
    )
    if polynomial.structure_id != prepared.polynomial.structure_id:
        raise ValueError(
            "Cannot refresh power-flow roots across a structural/mode mismatch."
        )
    old_problem = prepared.isolated.plan.problem
    problem = IsolatedPolynomialRootProblem(polynomial.system, old_problem.scaling)
    isolated = refresh_isolated_roots(prepared.isolated, problem)
    identifier = canonical_fingerprint(
        {
            "kind": "prepared-fixed-mode-power-flow-roots",
            "polynomial": polynomial.polynomial_id,
            "isolated": isolated.preparation_id,
            "physical_residual_tolerance": prepared.physical_residual_tolerance,
            "mode_tolerance": prepared.mode_tolerance,
            "operational_tolerance": prepared.operational_tolerance,
        }
    )
    return PreparedFixedModePowerFlowRoots(
        polynomial,
        isolated,
        prepared.physical_residual_tolerance,
        prepared.mode_tolerance,
        prepared.operational_tolerance,
        identifier,
    )


def _maximum_violation(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(
        max(
            np.max(lower - value, initial=0.0),
            np.max(value - upper, initial=0.0),
            0.0,
        )
    )


def _physical_evidence(
    prepared: PreparedFixedModePowerFlowRoots,
    algebraic: PolynomialRootResult,
    root_index: int,
    /,
) -> PowerFlowRootCandidateEvidence:
    polynomial = prepared.polynomial
    compiled = polynomial.compiled
    root = np.asarray(algebraic.roots[root_index])
    coordinates = root.real
    voltage = np.asarray(polynomial.voltage(coordinates))
    polynomial_residual = float(
        np.max(np.abs(np.asarray(polynomial.system.evaluate(root))), initial=0.0)
    )
    original_residual = float(
        np.max(np.abs(np.asarray(polynomial.native_residual(voltage))), initial=0.0)
    )
    imaginary_norm = float(np.max(np.abs(root.imag), initial=0.0))
    near_real = bool(algebraic.near_real_mask[root_index])
    buses = compiled.network.buses
    generators = compiled.network.generators
    bus_power = np.asarray(compiled.bus_power(voltage))
    required_generation = bus_power + np.asarray(compiled.load_power)
    magnitudes = np.abs(voltage)
    mode_violation = 0.0
    generator_violation = 0.0
    for bus, mode in enumerate(polynomial.modes):
        indices = compiled.generators_at_bus[bus]
        if indices:
            q_min = sum(generators[index].q_min for index in indices)
            q_max = sum(generators[index].q_max for index in indices)
            if mode == "reference":
                p_min = sum(generators[index].p_min for index in indices)
                p_max = sum(generators[index].p_max for index in indices)
                generator_violation = max(
                    generator_violation,
                    p_min - required_generation[bus].real,
                    required_generation[bus].real - p_max,
                )
            else:
                for index in indices:
                    generator = generators[index]
                    generator_violation = max(
                        generator_violation,
                        generator.p_min - generator.p,
                        generator.p - generator.p_max,
                    )
                    if mode == "pq":
                        generator_violation = max(
                            generator_violation,
                            generator.q_min - generator.q,
                            generator.q - generator.q_max,
                        )
        else:
            q_min, q_max = -math.inf, math.inf
        if mode == "pv":
            mode_violation = max(
                mode_violation,
                q_min - required_generation[bus].imag,
                required_generation[bus].imag - q_max,
            )
        elif mode == "q_min":
            declared = float(
                polynomial.injections[bus].imag + compiled.load_power[bus].imag
            )
            mode_violation = max(
                mode_violation,
                abs(declared - q_min),
                float(compiled.voltage_setpoints[bus]) - magnitudes[bus],
            )
        elif mode == "q_max":
            declared = float(
                polynomial.injections[bus].imag + compiled.load_power[bus].imag
            )
            mode_violation = max(
                mode_violation,
                abs(declared - q_max),
                magnitudes[bus] - float(compiled.voltage_setpoints[bus]),
            )
        elif mode == "reference" and indices:
            generator_violation = max(
                generator_violation,
                q_min - required_generation[bus].imag,
                required_generation[bus].imag - q_max,
            )
    mode_violation = max(float(mode_violation), 0.0)
    generator_violation = max(float(generator_violation), 0.0)
    voltage_violation = _maximum_violation(
        magnitudes,
        np.asarray([bus.v_min for bus in buses]),
        np.asarray([bus.v_max for bus in buses]),
    )
    branch_from, branch_to = compiled.branch_powers(voltage)
    branch_violation = float(
        np.max(
            np.maximum(np.abs(np.asarray(branch_from)), np.abs(np.asarray(branch_to)))
            - np.asarray([branch.rate for branch in compiled.network.branches]),
            initial=0.0,
        )
    )
    branch_violation = max(branch_violation, 0.0)
    original_accepted = original_residual <= prepared.physical_residual_tolerance
    mode_accepted = mode_violation <= prepared.mode_tolerance
    generator_accepted = generator_violation <= prepared.operational_tolerance
    voltage_accepted = voltage_violation <= prepared.operational_tolerance
    branch_accepted = branch_violation <= prepared.operational_tolerance
    physical = bool(
        near_real
        and original_accepted
        and mode_accepted
        and generator_accepted
        and voltage_accepted
        and branch_accepted
    )
    return PowerFlowRootCandidateEvidence(
        root_index,
        int(algebraic.cluster_path_counts[root_index]),
        jnp.asarray(voltage),
        polynomial_residual,
        original_residual,
        imaginary_norm,
        mode_violation,
        generator_violation,
        voltage_violation,
        branch_violation,
        near_real,
        original_accepted,
        mode_accepted,
        generator_accepted,
        voltage_accepted,
        branch_accepted,
        physical,
    )


def enumerate_prepared_fixed_mode_power_flow_roots(
    prepared: PreparedFixedModePowerFlowRoots, /
) -> FixedModePowerFlowRootSet:
    """Enumerate without mode switching and replay every clustered candidate physically."""

    if not isinstance(prepared, PreparedFixedModePowerFlowRoots):
        raise TypeError("prepared must be PreparedFixedModePowerFlowRoots.")
    algebraic = solve_prepared_isolated_roots(prepared.isolated)
    capacity = prepared.isolated.plan.policy.path_capacity
    bus_count = prepared.polynomial.bus_count
    voltage = np.full((capacity, bus_count), complex(np.nan, np.nan))
    candidate_mask = np.asarray(algebraic.root_mask, dtype=np.bool_)
    near_real_mask = np.zeros((capacity,), dtype=np.bool_)
    residual_mask = np.zeros((capacity,), dtype=np.bool_)
    mode_mask = np.zeros((capacity,), dtype=np.bool_)
    generator_mask = np.zeros((capacity,), dtype=np.bool_)
    voltage_mask = np.zeros((capacity,), dtype=np.bool_)
    branch_mask = np.zeros((capacity,), dtype=np.bool_)
    physical_mask = np.zeros((capacity,), dtype=np.bool_)
    candidates = []
    for root_index in np.flatnonzero(candidate_mask):
        evidence = _physical_evidence(prepared, algebraic, int(root_index))
        candidates.append(evidence)
        voltage[root_index] = np.asarray(evidence.voltage)
        near_real_mask[root_index] = evidence.near_real
        residual_mask[root_index] = evidence.original_residual_accepted
        mode_mask[root_index] = evidence.mode_complementarity_accepted
        generator_mask[root_index] = evidence.generator_limits_accepted
        voltage_mask[root_index] = evidence.voltage_limits_accepted
        branch_mask[root_index] = evidence.branch_limits_accepted
        physical_mask[root_index] = evidence.physically_accepted
    operational_mask = generator_mask & voltage_mask & branch_mask
    identifier = canonical_fingerprint(
        {
            "kind": "fixed-mode-power-flow-root-set",
            "polynomial": prepared.polynomial.polynomial_id,
            "algebraic": algebraic.result_id,
            "physical_indices": [int(index) for index in np.flatnonzero(physical_mask)],
            "candidate_evidence": [
                {
                    "root_index": evidence.root_index,
                    "cluster_path_count": evidence.cluster_path_count,
                    "polynomial_residual_norm": evidence.polynomial_residual_norm,
                    "original_residual_norm": evidence.original_residual_norm,
                    "imaginary_coordinate_norm": evidence.imaginary_coordinate_norm,
                    "mode_complementarity_violation": evidence.mode_complementarity_violation,
                    "generator_limit_violation": evidence.generator_limit_violation,
                    "voltage_limit_violation": evidence.voltage_limit_violation,
                    "branch_limit_violation": evidence.branch_limit_violation,
                    "physically_accepted": evidence.physically_accepted,
                }
                for evidence in candidates
            ],
        }
    )
    return FixedModePowerFlowRootSet(
        algebraic,
        jnp.asarray(voltage),
        jnp.asarray(candidate_mask),
        jnp.asarray(near_real_mask),
        jnp.asarray(residual_mask),
        jnp.asarray(mode_mask),
        jnp.asarray(generator_mask),
        jnp.asarray(voltage_mask),
        jnp.asarray(branch_mask),
        jnp.asarray(operational_mask),
        jnp.asarray(physical_mask),
        tuple(candidates),
        prepared.polynomial.polynomial_id,
        identifier,
    )


def enumerate_fixed_mode_power_flow_roots(
    compiled: CompiledNetwork,
    provider: HomotopyContinuationProvider,
    /,
    *,
    injections: ArrayLike | None = None,
    modes: tuple[str, ...] | None = None,
    policy: HomotopyContinuationPolicy | None = None,
    scaling: PolynomialScaling | None = None,
    polynomial_residual_tolerance: float = 1e-8,
    cluster_tolerance: float = 1e-7,
    near_real_absolute_tolerance: float = 1e-8,
    near_real_relative_tolerance: float = 1e-8,
    physical_residual_tolerance: float = 1e-7,
    mode_tolerance: float = 1e-7,
    operational_tolerance: float = 1e-7,
) -> FixedModePowerFlowRootSet:
    polynomial = compile_fixed_mode_power_flow_polynomial(
        compiled, injections, modes=modes
    )
    prepared = prepare_fixed_mode_power_flow_roots(
        polynomial,
        provider,
        policy=policy,
        scaling=scaling,
        polynomial_residual_tolerance=polynomial_residual_tolerance,
        cluster_tolerance=cluster_tolerance,
        near_real_absolute_tolerance=near_real_absolute_tolerance,
        near_real_relative_tolerance=near_real_relative_tolerance,
        physical_residual_tolerance=physical_residual_tolerance,
        mode_tolerance=mode_tolerance,
        operational_tolerance=operational_tolerance,
    )
    return enumerate_prepared_fixed_mode_power_flow_roots(prepared)


__all__ = [
    "FixedModePowerFlowPolynomial",
    "PreparedFixedModePowerFlowRoots",
    "PowerFlowRootCandidateEvidence",
    "FixedModePowerFlowRootSet",
    "compile_fixed_mode_power_flow_polynomial",
    "prepare_fixed_mode_power_flow_roots",
    "refresh_fixed_mode_power_flow_roots",
    "enumerate_prepared_fixed_mode_power_flow_roots",
    "enumerate_fixed_mode_power_flow_roots",
]
