# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Basis-adjoint quantum charge/Poisson coupling with bounded Gummel updates.

The quantum operator and Poisson source use the same cell projector. The
partitioned outer iteration is eager so energy grids, isolated bound poles
and selected modes can be reprepared explicitly; it is not differentiated
through adaptive decisions. Each potential proposal is a native finite
PicardUpdate with a reusable native tridiagonal Poisson inverse action.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .... import linalg as la
from ...._strict import StrictModule
from ....nonlinear import (
    apply_prepared_nonlinear_update,
    NonlinearSystemProblem,
    PicardUpdate,
    prepare_nonlinear_update,
    refresh_nonlinear_update,
)
from ._basis import _array, EffectiveMass1D, SchrodingerResult, solve_schrodinger
from ._coherent import CoherentDevice, CoherentResult, integrate_coherent


class QuantumPoisson1D(StrictModule):
    """Two Dirichlet electrodes and N bulk cells; all input/output is SI.

    face_permittivity has N+1 values in F/m. fixed_charge_density is C/m3
    (dopants, classical *other* populations, and explicitly assigned sources).
    The same quantum electron population must not also be included there.
    Electrodes are ghost nodes one grid spacing outside the interior centers.
    """

    cell_volumes: Array
    face_capacitances: Array
    fixed_charge_density: Array
    electrode_voltages: Array
    operator: la.TridiagonalLinearOperator
    prepared: la.PreparedLinearSolve
    scale: Array

    def __init__(
        self, basis, face_permittivity, fixed_charge_density, electrode_voltages
    ):
        if not isinstance(basis, EffectiveMass1D):
            raise TypeError(
                "Quantum Poisson currently admits the EffectiveMass1D cell basis only."
            )
        n = basis.x.size
        eps = _array(
            np.broadcast_to(face_permittivity, (n + 1,)),
            "face permittivities",
            positive=True,
        )
        fixed = _array(
            np.broadcast_to(fixed_charge_density, (n,)), "fixed charge density"
        )
        voltages = _array(electrode_voltages, "electrode potentials")
        if voltages.shape != (2,):
            raise ValueError(
                "Exactly two electrostatic Dirichlet electrode voltages are required."
            )
        capacitance = eps * basis.area / basis.spacing
        scale = jnp.max(capacitance)
        op = la.TridiagonalLinearOperator(
            -capacitance[1:-1] / scale,
            (capacitance[:-1] + capacitance[1:]) / scale,
            -capacitance[1:-1] / scale,
            operator_id="quantum-poisson-cell-chain",
        )
        self.cell_volumes = basis.base_hamiltonian.cell_volumes
        self.face_capacitances, self.fixed_charge_density = capacitance, fixed
        self.electrode_voltages, self.scale, self.operator = voltages, scale, op
        budget = basis.base_hamiltonian.resources.workspace_bytes
        policy = la.LinearSolvePolicy(
            materialization=la.MaterializationPolicy(max_entries=1, max_bytes=budget),
            resources=la.SolveResourcePolicy(
                factorization_bytes=budget, workspace_bytes=budget
            ),
        )
        self.prepared = la.prepare(la.LinearSystem(op), policy)

    def right_hand_side(self, quantum_charge_density):
        quantum = jnp.asarray(quantum_charge_density)
        if quantum.shape != self.cell_volumes.shape:
            raise ValueError("Quantum charge must have one density per cell.")
        charge = (self.fixed_charge_density + quantum) * self.cell_volumes
        return (
            charge.at[0]
            .add(self.face_capacitances[0] * self.electrode_voltages[0])
            .at[-1]
            .add(self.face_capacitances[-1] * self.electrode_voltages[1])
            / self.scale
        )

    def residual(self, potential, quantum_charge_density):
        """Integrated physical Poisson residual, C per cell."""
        potential_ = jnp.asarray(potential)
        if potential_.shape != self.cell_volumes.shape:
            raise ValueError("Potential must have one value per Poisson cell.")
        return self.scale * (
            self.operator.mv(potential_) - self.right_hand_side(quantum_charge_density)
        )

    def solve(self, quantum_charge_density):
        return la.solve(self.prepared, self.right_hand_side(quantum_charge_density))

    def terminal_charge(self, potential):
        return self.face_capacitances[jnp.asarray([0, -1])] * (
            self.electrode_voltages - jnp.asarray(potential)[jnp.asarray([0, -1])]
        )


class QuantumPoissonEvidence(StrictModule):
    potential_defect: Array
    poisson_residual: Array
    charge_balance_error: Array
    poisson_residual_tolerance: Array
    charge_balance_tolerance: Array
    quantum_valid: Array
    updates_valid: Array
    converged: Array
    iterations: int = eqx.field(static=True)
    empirical_qualified: bool = eqx.field(static=True, default=False)


class QuantumPoissonResult(StrictModule):
    potential: Array
    quantum: SchrodingerResult | CoherentResult
    terminal_charges: Array
    evidence: QuantumPoissonEvidence
    nonlinear_updates: tuple
    successful: Array


def _self_consistent(
    poisson, evaluate, initial_potential, *, potential_tolerance, maximum_steps, damping
):
    if (
        isinstance(maximum_steps, bool)
        or not isinstance(maximum_steps, (int, np.integer))
        or maximum_steps < 1
        or not np.isfinite(potential_tolerance)
        or potential_tolerance <= 0
        or not np.isfinite(damping)
        or not 0 < damping <= 1
    ):
        raise ValueError(
            "Self-consistency requires a positive integer step bound, finite "
            "positive voltage tolerance, and damping in (0, 1]."
        )
    psi = _array(initial_potential, "initial electrostatic potential")
    if psi.shape != poisson.cell_volumes.shape:
        raise ValueError("Initial potential must have one value per quantum cell.")
    quantum = evaluate(psi)
    problem = NonlinearSystemProblem(
        lambda state, rhs: poisson.operator.mv(state) - rhs,
        problem_id="quantum-poisson-frozen-charge-update",
    )

    def inverse_action(residual):
        result = la.solve(poisson.prepared, residual)
        return eqx.error_if(
            result.value,
            ~jnp.all(result.successful),
            "Quantum Poisson inverse action failed.",
        )

    update = PicardUpdate(inverse_action, damping=damping)
    rhs = poisson.right_hand_side(quantum.charge_density)
    prepared = prepare_nonlinear_update(problem, psi, update, args=rhs)
    updates = []
    update_valid = jnp.asarray(True)
    for iteration in range(maximum_steps + 1):
        residual = poisson.residual(psi, quantum.charge_density)
        correction = la.solve(poisson.prepared, residual / poisson.scale)
        defect = jnp.max(jnp.abs(correction.value))
        update_valid = update_valid & jnp.all(correction.successful)
        if bool((defect <= potential_tolerance) & quantum.successful & update_valid):
            break
        if iteration == maximum_steps or not bool(quantum.successful & update_valid):
            break
        rhs = poisson.right_hand_side(quantum.charge_density)
        prepared = refresh_nonlinear_update(prepared, problem, psi, args=rhs)
        proposal, prepared = apply_prepared_nonlinear_update(prepared, psi, args=rhs)
        update_valid = update_valid & proposal.applied
        updates.append(proposal)
        if not bool(proposal.applied):
            break
        psi = proposal.state
        quantum = evaluate(psi)
    terminal = poisson.terminal_charge(psi)
    bulk = jnp.sum(
        (poisson.fixed_charge_density + quantum.charge_density) * poisson.cell_volumes
    )
    charge_error = jnp.abs(jnp.sum(terminal) + bulk)
    rounding = (
        32 * jnp.finfo(psi.dtype).eps * (jnp.abs(bulk) + jnp.sum(jnp.abs(terminal)))
    )
    charge_tolerance = (
        potential_tolerance
        * (poisson.face_capacitances[0] + poisson.face_capacitances[-1])
        + rounding
    )
    residual_tolerance = (
        2
        * potential_tolerance
        * jnp.max(poisson.face_capacitances[:-1] + poisson.face_capacitances[1:])
        + rounding
    )
    converged = (
        quantum.successful
        & update_valid
        & (defect <= potential_tolerance)
        & jnp.all(jnp.isfinite(residual))
        & (charge_error <= charge_tolerance)
        & (jnp.max(jnp.abs(residual)) <= residual_tolerance)
    )
    evidence = QuantumPoissonEvidence(
        potential_defect=defect,
        poisson_residual=jnp.max(jnp.abs(residual)),
        charge_balance_error=charge_error,
        poisson_residual_tolerance=residual_tolerance,
        charge_balance_tolerance=charge_tolerance,
        quantum_valid=quantum.successful,
        updates_valid=update_valid,
        converged=converged,
        iterations=len(updates),
    )
    return QuantumPoissonResult(
        potential=psi,
        quantum=quantum,
        terminal_charges=terminal,
        evidence=evidence,
        nonlinear_updates=tuple(updates),
        successful=converged,
    )


def solve_schrodinger_poisson(
    basis,
    poisson,
    chemical_potential,
    temperature,
    *,
    count,
    initial_potential=None,
    transverse=None,
    omitted_particle_tolerance=1e-8,
    potential_tolerance=1e-7,
    maximum_steps=100,
    damping=0.2,
):
    """Grand-canonical self-consistent confinement, not fixed particle number."""
    if not isinstance(basis, EffectiveMass1D):
        raise TypeError("Schrodinger-Poisson requires an EffectiveMass1D basis.")
    if not np.array_equal(
        np.asarray(basis.base_hamiltonian.cell_volumes), np.asarray(poisson.cell_volumes)
    ):
        raise ValueError("Quantum and Poisson charge measures must be identical.")
    initial = jnp.zeros_like(basis.x) if initial_potential is None else initial_potential

    def evaluate(psi):
        return solve_schrodinger(
            basis.hamiltonian(psi),
            chemical_potential,
            temperature,
            count=count,
            transverse=transverse,
            omitted_particle_tolerance=omitted_particle_tolerance,
        )

    return _self_consistent(
        poisson,
        evaluate,
        initial,
        potential_tolerance=potential_tolerance,
        maximum_steps=maximum_steps,
        damping=damping,
    )


def solve_coherent_poisson(
    device,
    poisson,
    *,
    initial_potential=None,
    bound_occupation=None,
    potential_tolerance=1e-7,
    maximum_steps=100,
    damping=0.2,
    integration_options=None,
):
    """Coherent NEGF-Poisson with fixed reservoir energies and explicit gates.

    device.hamiltonian is the zero-potential material/kinetic operator. Contact
    lead onsite and mu are explicit absolute energies at the applied bias;
    electrode voltages do not secretly shift reservoir band alignments.
    Charge/spectral/current and Poisson gates must all hold at the same state.
    """
    if not isinstance(device, CoherentDevice):
        raise TypeError("Coherent Poisson requires a CoherentDevice.")
    if not np.array_equal(
        np.asarray(device.hamiltonian.cell_volumes), np.asarray(poisson.cell_volumes)
    ):
        raise ValueError("Quantum and Poisson charge measures must be identical.")
    options = {} if integration_options is None else dict(integration_options)
    initial = (
        jnp.zeros_like(poisson.cell_volumes)
        if initial_potential is None
        else initial_potential
    )

    def evaluate(psi):
        return integrate_coherent(
            device.with_potential(psi), bound_occupation=bound_occupation, **options
        )

    return _self_consistent(
        poisson,
        evaluate,
        initial,
        potential_tolerance=potential_tolerance,
        maximum_steps=maximum_steps,
        damping=damping,
    )
