#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Lossless DC LP/QP and rectangular AC structured NLP with original feasibility."""

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ... import optim
from ..._strict import StrictModule
from ...ein import contract
from ...linalg import ArraySpace
from ...sparse import compile_sparse_hessian, compile_sparse_jacobian, SparsePattern
from ._network import CompiledNetwork, PowerNetwork, PowerStudy
from ._power_flow import _compiled, _limit_violation, PowerFlowResult


def _mv(matrix, value):
    return contract("ij,j->i", matrix, value)


def _dc_matrices(compiled):
    network = compiled.network
    n, m = len(network.buses), len(network.branches)
    incidence = jnp.zeros((m, n))
    incidence = incidence.at[jnp.arange(m), compiled.from_indices].set(1.0)
    incidence = incidence.at[jnp.arange(m), compiled.to_indices].set(-1.0)
    for branch in network.branches:
        if branch.in_service and branch.x == 0:
            raise ValueError(
                "DC approximation requires nonzero reactance on every active branch."
            )
    coefficient = jnp.asarray(
        [1 / (b.x * b.tap) if b.in_service else 0.0 for b in network.branches]
    )
    branch_matrix = coefficient[:, None] * incidence
    offset = -coefficient * jnp.asarray([b.phase for b in network.branches])
    nodal_matrix = contract("ki,kj->ij", incidence, branch_matrix)
    nodal_offset = _mv(incidence.T, offset)
    # At unit voltage, passive conductance consumes g pu even in the DC approximation.
    demand = compiled.load_power.real + compiled.shunt_admittance.real
    return branch_matrix, offset, nodal_matrix, nodal_offset, demand


class DCFlowResult(StrictModule):
    angle: Array
    bus_power: Array
    branch_from: Array
    branch_to: Array
    reference_power: Array
    balance_residual: Array
    residual_norm: Array
    converged: Array
    native_result: Any
    approximation: str = eqx.field(static=True, default="lossless-unit-voltage-dc")


def solve_dc_power_flow(
    network: PowerNetwork | CompiledNetwork,
    *,
    study: PowerStudy | None = None,
    policy=None,
) -> DCFlowResult:
    """Solve DC balance as one native equality-constrained feasibility LP.

    reference_power is net generation-minus-load needed at each reference, zero
    elsewhere. This is not an AC feasibility certificate and does not enforce Q.
    """
    compiled = _compiled(network, study)
    n, nr = len(compiled.network.buses), len(compiled.references)
    branch, offset, nodal, nodal_offset, _ = _dc_matrices(compiled)
    refs = jnp.asarray(compiled.references, dtype=jnp.int32)
    selector = jnp.zeros((n, nr)).at[refs, jnp.arange(nr)].set(1.0)
    specified = compiled.specified_power.real
    rhs = specified - compiled.shunt_admittance.real - nodal_offset
    rhs = rhs.at[refs].set(-compiled.shunt_admittance.real[refs] - nodal_offset[refs])
    equality = jnp.concatenate((nodal, -selector), axis=1)
    angles = compiled.initial_angles[refs]
    reference_rows = jnp.concatenate((selector.T, jnp.zeros((nr, nr))), axis=1)
    program = optim.LinearProgram(
        jnp.zeros(n + nr),
        equality_matrix=jnp.concatenate((equality, reference_rows)),
        equality_rhs=jnp.concatenate((rhs, angles)),
        problem_id="balanced-dc-power-flow",
    )
    result = optim.solve_linear_program(program, policy=policy)
    theta = result.primal[:n]
    flow = _mv(branch, theta) + offset
    reference_power = _mv(selector, result.primal[n:])
    bus_power = _mv(nodal, theta) + nodal_offset + compiled.shunt_admittance.real
    expected = specified.at[refs].set(reference_power[refs])
    balance = bus_power - expected
    residual = jnp.max(jnp.abs(balance), initial=0.0)
    return DCFlowResult(
        theta,
        bus_power,
        flow,
        -flow,
        reference_power,
        balance,
        residual,
        result.valid & (residual <= 1e-6),
        result,
    )


class DCOPFCompilation(StrictModule):
    network: CompiledNetwork
    program: optim.LinearProgram | optim.QuadraticProgram
    branch_matrix: Array
    branch_offset: Array
    nodal_matrix: Array
    nodal_offset: Array
    demand: Array
    generation_map: Array


class DCOPFResult(StrictModule):
    angle: Array
    generator_power: Array
    branch_from: Array
    branch_to: Array
    balance_residual: Array
    objective: Array
    original_feasibility: Array
    converged: Array
    native_result: Any
    approximation: str = eqx.field(static=True, default="lossless-unit-voltage-dc")


def _require_dispatchable_islands(compiled):
    for island in compiled.islands:
        if not any(compiled.generators_at_bus[i] for i in island):
            raise ValueError(
                "OPF requires an explicit in-service generator in every island; no ideal external slack is added."
            )


def compile_dc_opf(
    network: PowerNetwork | CompiledNetwork, *, study: PowerStudy | None = None
) -> DCOPFCompilation:
    """Compile linear costs to LP or convex quadratic costs to native QP."""
    compiled = _compiled(network, study)
    _require_dispatchable_islands(compiled)
    buses, gens = compiled.network.buses, compiled.network.generators
    n, ng = len(buses), len(gens)
    if any(g.in_service and g.cost[0] < 0 for g in gens):
        raise ValueError(
            "DC OPF requires convex generator cost (quadratic coefficient >= 0)."
        )
    branch, offset, nodal, nodal_offset, demand = _dc_matrices(compiled)
    generation = (
        jnp.zeros((n, ng)).at[compiled.generator_indices, jnp.arange(ng)].set(1.0)
    )
    equality = jnp.concatenate((nodal, -generation), axis=1)
    refs = jnp.asarray(compiled.references, dtype=jnp.int32)
    reference_rows = (
        jnp.zeros((len(refs), n + ng)).at[jnp.arange(len(refs)), refs].set(1.0)
    )
    equality = jnp.concatenate((equality, reference_rows))
    rhs = jnp.concatenate((-demand - nodal_offset, compiled.initial_angles[refs]))
    limited = jnp.asarray(
        [
            i
            for i, b in enumerate(compiled.network.branches)
            if b.in_service and isfinite(b.rate)
        ],
        dtype=jnp.int32,
    )
    flow_rows = jnp.concatenate((branch[limited], jnp.zeros((len(limited), ng))), axis=1)
    rates = jnp.asarray([b.rate for b in compiled.network.branches])[limited]
    inequality = jnp.concatenate((flow_rows, -flow_rows))
    inequality_rhs = jnp.concatenate((rates - offset[limited], rates + offset[limited]))
    lower = jnp.asarray(
        [-float("inf")] * n + [g.p_min if g.in_service else 0.0 for g in gens]
    )
    upper = jnp.asarray(
        [float("inf")] * n + [g.p_max if g.in_service else 0.0 for g in gens]
    )
    linear = jnp.asarray([0.0] * n + [g.cost[1] if g.in_service else 0.0 for g in gens])
    options = dict(
        equality_matrix=equality,
        equality_rhs=rhs,
        inequality_matrix=inequality,
        inequality_rhs=inequality_rhs,
        bounds=optim.Bounds(lower, upper),
        problem_id="balanced-dc-opf",
    )
    if any(g.in_service and g.cost[0] != 0 for g in gens):
        quadratic = jnp.diag(
            jnp.asarray(
                [0.0] * n + [2 * g.cost[0] if g.in_service else 0.0 for g in gens]
            )
        )
        program = optim.QuadraticProgram(quadratic, linear, **options)
    else:
        program = optim.LinearProgram(linear, **options)
    return DCOPFCompilation(
        compiled, program, branch, offset, nodal, nodal_offset, demand, generation
    )


def solve_dc_opf(
    network: PowerNetwork | CompiledNetwork | DCOPFCompilation,
    *,
    study: PowerStudy | None = None,
    policy=None,
) -> DCOPFResult:
    if isinstance(network, DCOPFCompilation) and study is not None:
        raise ValueError("study is already bound in DCOPFCompilation.")
    compiled = (
        network
        if isinstance(network, DCOPFCompilation)
        else compile_dc_opf(network, study=study)
    )
    program = compiled.program
    result = (
        optim.solve_linear_program(program, policy=policy)
        if isinstance(program, optim.LinearProgram)
        else optim.solve_quadratic_program(program, policy=policy)
    )
    buses, gens = compiled.network.network.buses, compiled.network.network.generators
    n = len(buses)
    angle, generated = result.primal[:n], result.primal[n:]
    flow = _mv(compiled.branch_matrix, angle) + compiled.branch_offset
    balance = (
        _mv(compiled.nodal_matrix, angle)
        + compiled.nodal_offset
        + compiled.demand
        - _mv(compiled.generation_map, generated)
    )
    limits = jnp.asarray([b.rate for b in compiled.network.network.branches])
    violation = jnp.maximum(
        jnp.max(jnp.abs(balance), initial=0.0),
        jnp.max(jnp.abs(flow) - limits, initial=0.0),
    )
    violation = jnp.maximum(
        violation,
        _limit_violation(
            generated,
            jnp.asarray([g.p_min if g.in_service else 0.0 for g in gens]),
            jnp.asarray([g.p_max if g.in_service else 0.0 for g in gens]),
        ),
    )
    refs = jnp.asarray(compiled.network.references, dtype=jnp.int32)
    ref_error = jnp.max(
        jnp.abs(angle[refs] - compiled.network.initial_angles[refs]), initial=0.0
    )
    violation = jnp.maximum(violation, ref_error)
    objective = jnp.sum(
        jnp.asarray([g.cost[0] if g.in_service else 0.0 for g in gens]) * generated**2
        + jnp.asarray([g.cost[1] if g.in_service else 0.0 for g in gens]) * generated
        + jnp.asarray([g.cost[2] if g.in_service else 0.0 for g in gens])
    )
    return DCOPFResult(
        angle,
        generated,
        flow,
        -flow,
        balance,
        objective,
        violation,
        result.valid & (violation <= 1e-6),
        result,
    )


class ACOPFCompilation(StrictModule):
    network: CompiledNetwork
    program: optim.StructuredNonlinearProgram
    initial_coordinates: Array
    nonreferences: Array
    free_p: Array
    free_q: Array
    fixed_p: Array
    fixed_q: Array
    reference_rotation: Array
    voltage_size: int = eqx.field(static=True)

    def unpack(self, value):
        n = len(self.network.network.buses)
        refs = jnp.asarray(self.network.references, dtype=jnp.int32)
        voltage = value[:n].astype(self.reference_rotation.dtype)
        voltage = voltage.at[self.nonreferences].add(1j * value[n : self.voltage_size])
        voltage = voltage.at[refs].set(value[refs] * self.reference_rotation[refs])
        p = self.fixed_p.at[self.free_p].set(
            value[self.voltage_size : self.voltage_size + len(self.free_p)]
        )
        q = self.fixed_q.at[self.free_q].set(
            value[self.voltage_size + len(self.free_p) :]
        )
        return voltage, p + 1j * q


class ACOPFResult(StrictModule):
    voltage: Array
    generator_power: Array
    branch_from: Array
    branch_to: Array
    branch_loss: Array
    shunt_power: Array
    bus_balance: Array
    total_balance: Array
    objective: Array
    original_feasibility: Array
    converged: Array
    native_result: optim.StructuredNonlinearResult
    optimality_scope: str = eqx.field(static=True, default="local-nonlinear-optimum-only")


def compile_ac_opf(
    network: PowerNetwork | CompiledNetwork,
    *,
    study: PowerStudy | None = None,
    operating_point: PowerFlowResult | None = None,
) -> ACOPFCompilation:
    """Compile exact sparse rectangular equations and both terminal |S|² bounds.

    Reference angles are eliminated (reference magnitudes remain decision variables).
    Offline and fixed generator coordinates are eliminated; limits are never softened.
    Sparsity is declared from electrical topology, not inferred from a lucky sample.
    """
    compiled = _compiled(network, study)
    _require_dispatchable_islands(compiled)
    buses, gens = compiled.network.buses, compiled.network.generators
    n, ng = len(buses), len(gens)
    nonrefs = tuple(i for i in range(n) if i not in compiled.references)
    nonreference = jnp.asarray(nonrefs, dtype=jnp.int32)
    refs = jnp.asarray(compiled.references, dtype=jnp.int32)
    free_p_ids = tuple(
        i for i, g in enumerate(gens) if g.in_service and g.p_min != g.p_max
    )
    free_q_ids = tuple(
        i for i, g in enumerate(gens) if g.in_service and g.q_min != g.q_max
    )
    free_p, free_q = (
        jnp.asarray(free_p_ids, dtype=jnp.int32),
        jnp.asarray(free_q_ids, dtype=jnp.int32),
    )
    fixed_p = jnp.asarray(
        [g.p_min if g.in_service and g.p_min == g.p_max else 0.0 for g in gens]
    )
    fixed_q = jnp.asarray(
        [g.q_min if g.in_service and g.q_min == g.q_max else 0.0 for g in gens]
    )
    rotation = jnp.exp(1j * compiled.initial_angles)
    nv = n + len(nonrefs)
    size = nv + len(free_p_ids) + len(free_q_ids)
    initial_v = (
        compiled.initial_voltage if operating_point is None else operating_point.voltage
    )
    initial_g = (
        jnp.asarray([complex(g.p, g.q) if g.in_service else 0j for g in gens])
        if operating_point is None
        else operating_point.generator_power
    )
    if initial_v.shape != (n,) or initial_g.shape != (ng,):
        raise ValueError("Operating point shapes do not match the compiled network.")
    first = initial_v.real.at[refs].set(jnp.abs(initial_v[refs]))
    initial = jnp.concatenate(
        (
            first,
            initial_v.imag[nonreference],
            initial_g.real[free_p],
            initial_g.imag[free_q],
        )
    )
    voltage_lower = [-b.v_max for b in buses] + [-buses[i].v_max for i in nonrefs]
    voltage_upper = [b.v_max for b in buses] + [buses[i].v_max for i in nonrefs]
    for i in compiled.references:
        voltage_lower[i] = buses[i].v_min
    lower = jnp.asarray(
        voltage_lower
        + [gens[i].p_min for i in free_p_ids]
        + [gens[i].q_min for i in free_q_ids]
    )
    upper = jnp.asarray(
        voltage_upper
        + [gens[i].p_max for i in free_p_ids]
        + [gens[i].q_max for i in free_q_ids]
    )
    limited_ids = tuple(
        i
        for i, b in enumerate(compiled.network.branches)
        if b.in_service and isfinite(b.rate)
    )
    limited = jnp.asarray(limited_ids, dtype=jnp.int32)
    costs = jnp.asarray([g.cost if g.in_service else (0.0, 0.0, 0.0) for g in gens])

    def unpack(value):
        voltage = value[:n].astype(rotation.dtype)
        voltage = voltage.at[nonreference].add(1j * value[n:nv])
        voltage = voltage.at[refs].set(value[refs] * rotation[refs])
        p = fixed_p.at[free_p].set(value[nv : nv + len(free_p_ids)])
        q = fixed_q.at[free_q].set(value[nv + len(free_p_ids) :])
        return voltage, p + 1j * q

    def objective(value, args):
        _, generation = unpack(value)
        return jnp.sum(
            costs[:, 0] * generation.real**2 + costs[:, 1] * generation.real + costs[:, 2]
        )

    def constraints(value, args):
        voltage, generation = unpack(value)
        injection = (
            jnp.zeros(n, dtype=voltage.dtype)
            .at[compiled.generator_indices]
            .add(generation)
            - compiled.load_power
        )
        mismatch = voltage * jnp.conj(compiled.bus_currents(voltage)) - injection
        sf, st = compiled.branch_powers(voltage)
        return jnp.concatenate(
            (
                mismatch.real,
                mismatch.imag,
                jnp.abs(voltage[nonreference]) ** 2,
                jnp.abs(sf[limited]) ** 2,
                jnp.abs(st[limited]) ** 2,
            )
        )

    constraint_lower = jnp.asarray(
        [0.0] * (2 * n)
        + [buses[i].v_min ** 2 for i in nonrefs]
        + [-float("inf")] * (2 * len(limited_ids))
    )
    rates = [compiled.network.branches[i].rate ** 2 for i in limited_ids]
    constraint_upper = jnp.asarray(
        [0.0] * (2 * n) + [buses[i].v_max ** 2 for i in nonrefs] + rates + rates
    )
    sources = tuple(
        [f"bus:{b.id}:P" for b in buses]
        + [f"bus:{b.id}:Q" for b in buses]
        + [f"bus:{buses[i].id}:voltage-squared" for i in nonrefs]
        + [
            f"branch:{compiled.network.branches[i].id}:from-MVA-squared"
            for i in limited_ids
        ]
        + [
            f"branch:{compiled.network.branches[i].id}:to-MVA-squared"
            for i in limited_ids
        ]
    )
    voltage_columns = [{i} for i in range(n)]
    for k, i in enumerate(nonrefs):
        voltage_columns[i].add(n + k)
    neighbours = [{i} for i in range(n)]
    for branch_spec, f, t in zip(
        compiled.network.branches,
        np.asarray(compiled.from_indices),
        np.asarray(compiled.to_indices),
        strict=True,
    ):
        if branch_spec.in_service:
            neighbours[f].add(int(t))
            neighbours[t].add(int(f))
    rows = [set() for _ in sources]
    hessian_entries = {(i, i) for i in range(size)}
    for i in range(n):
        columns = set().union(*(voltage_columns[j] for j in neighbours[i]))
        rows[i].update(columns)
        rows[n + i].update(columns)
        # Power is bilinear only between a bus and its adjacent buses.
        for j in neighbours[i]:
            pair = voltage_columns[i] | voltage_columns[j]
            hessian_entries.update((a, b) for a in pair for b in pair)
    for k, g in enumerate(free_p_ids):
        rows[int(compiled.generator_indices[g])].add(nv + k)
    for k, g in enumerate(free_q_ids):
        rows[n + int(compiled.generator_indices[g])].add(nv + len(free_p_ids) + k)
    for k, i in enumerate(nonrefs):
        rows[2 * n + k].update(voltage_columns[i])
    for k, i in enumerate(limited_ids):
        columns = (
            voltage_columns[int(compiled.from_indices[i])]
            | voltage_columns[int(compiled.to_indices[i])]
        )
        rows[2 * n + len(nonrefs) + k].update(columns)
        rows[2 * n + len(nonrefs) + len(limited_ids) + k].update(columns)
        hessian_entries.update((a, b) for a in columns for b in columns)
    entries = tuple((i, j) for i, columns in enumerate(rows) for j in sorted(columns))
    pattern = SparsePattern.from_coo(
        [i for i, j in entries], [j for i, j in entries], (len(sources), size)
    )
    hentries = sorted(hessian_entries)
    hpattern = SparsePattern.from_coo(
        [i for i, j in hentries], [j for i, j in hentries], (size, size), symmetric=True
    )
    source = ArraySpace((size,), dtype=initial.dtype)
    target = ArraySpace((len(sources),), dtype=initial.dtype)
    jacobian = compile_sparse_jacobian(
        constraints,
        initial,
        source=source,
        target=target,
        structure=pattern,
        compiler="native",
        plan_id="balanced-ac-opf-jacobian",
    )

    def lagrangian(value, packed):
        return packed[1] * objective(value, packed[0]) + jnp.sum(
            packed[2] * constraints(value, packed[0])
        )

    hessian = compile_sparse_hessian(
        lagrangian,
        initial,
        space=source,
        structure=hpattern,
        sample_args=(None, jnp.asarray(1.0), jnp.zeros(len(sources))),
        compiler="native",
        plan_id="balanced-ac-opf-hessian",
    )
    program = optim.StructuredNonlinearProgram(
        objective,
        constraints,
        jacobian,
        variable_lower=lower,
        variable_upper=upper,
        constraint_lower=constraint_lower,
        constraint_upper=constraint_upper,
        constraint_sources=sources,
        hessian_plan=hessian,
        program_id="balanced-ac-opf",
        structure_id=f"balanced-ac-opf:{pattern.pattern_id}:{hpattern.pattern_id}",
    )
    return ACOPFCompilation(
        compiled,
        program,
        initial,
        nonreference,
        free_p,
        free_q,
        fixed_p,
        fixed_q,
        rotation,
        nv,
    )


def solve_ac_opf(
    network: PowerNetwork | CompiledNetwork | ACOPFCompilation,
    *,
    study: PowerStudy | None = None,
    operating_point: PowerFlowResult | None = None,
    method=None,
    termination=None,
    feasibility_tolerance: float = 1e-6,
) -> ACOPFResult:
    """Solve a native structured NLP and independently audit original AC equations.

    Convergence certifies the returned local candidate only, never global optimality.
    No load shedding, relaxed limits or feasibility-restoring penalty is introduced.
    """
    if not isfinite(feasibility_tolerance) or feasibility_tolerance <= 0:
        raise ValueError("feasibility_tolerance must be finite and positive.")
    if isinstance(network, ACOPFCompilation) and study is not None:
        raise ValueError("study is already bound in ACOPFCompilation.")
    compilation = (
        network
        if isinstance(network, ACOPFCompilation)
        else compile_ac_opf(network, study=study, operating_point=operating_point)
    )
    if isinstance(network, ACOPFCompilation) and operating_point is not None:
        raise ValueError("operating_point is already bound in an ACOPFCompilation.")
    method_ = (
        optim.PrimalDualInteriorPoint(mode="sparse-augmented")
        if method is None
        else method
    )
    native = optim.solve_structured_nonlinear(
        compilation.program,
        compilation.initial_coordinates,
        method=method_,
        termination=termination,
    )
    voltage, generated = compilation.unpack(native.optimization.parameters)
    compiled = compilation.network
    buses, gens = compiled.network.buses, compiled.network.generators
    sf, st = compiled.branch_powers(voltage)
    injection = (
        jnp.zeros(len(buses), dtype=voltage.dtype)
        .at[compiled.generator_indices]
        .add(generated)
        - compiled.load_power
    )
    mismatch = injection - voltage * jnp.conj(compiled.bus_currents(voltage))
    shunt = jnp.abs(voltage) ** 2 * jnp.conj(compiled.shunt_admittance)
    total_balance = jnp.sum(injection) - jnp.sum(sf + st) - jnp.sum(shunt)
    feasibility = jnp.max(jnp.abs(mismatch), initial=0.0)
    feasibility = jnp.maximum(
        feasibility,
        _limit_violation(
            jnp.abs(voltage),
            jnp.asarray([b.v_min for b in buses]),
            jnp.asarray([b.v_max for b in buses]),
        ),
    )
    feasibility = jnp.maximum(
        feasibility,
        _limit_violation(
            generated.real,
            jnp.asarray([g.p_min if g.in_service else 0 for g in gens]),
            jnp.asarray([g.p_max if g.in_service else 0 for g in gens]),
        ),
    )
    feasibility = jnp.maximum(
        feasibility,
        _limit_violation(
            generated.imag,
            jnp.asarray([g.q_min if g.in_service else 0 for g in gens]),
            jnp.asarray([g.q_max if g.in_service else 0 for g in gens]),
        ),
    )
    feasibility = jnp.maximum(
        feasibility,
        jnp.max(
            jnp.maximum(jnp.abs(sf), jnp.abs(st))
            - jnp.asarray([b.rate for b in compiled.network.branches]),
            initial=0.0,
        ),
    )
    refs = jnp.asarray(compiled.references, dtype=jnp.int32)
    reference_value = voltage[refs] * jnp.conj(compilation.reference_rotation[refs])
    feasibility = jnp.maximum(
        feasibility, jnp.max(jnp.abs(reference_value.imag), initial=0.0)
    )
    finite = (
        jnp.all(jnp.isfinite(voltage))
        & jnp.all(jnp.isfinite(generated))
        & jnp.isfinite(feasibility)
    )
    success = (
        native.optimization.successful & finite & (feasibility <= feasibility_tolerance)
    )
    objective = compilation.program.objective(native.optimization.parameters, None)
    return ACOPFResult(
        voltage,
        generated,
        sf,
        st,
        sf + st,
        shunt,
        mismatch,
        total_balance,
        objective,
        feasibility,
        success,
        native,
    )


__all__ = [
    "DCFlowResult",
    "solve_dc_power_flow",
    "DCOPFCompilation",
    "DCOPFResult",
    "compile_dc_opf",
    "solve_dc_opf",
    "ACOPFCompilation",
    "ACOPFResult",
    "compile_ac_opf",
    "solve_ac_opf",
]
