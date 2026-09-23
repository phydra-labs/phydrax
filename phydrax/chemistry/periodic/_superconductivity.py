#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._reciprocal import (
    PreparedReciprocalConnectivity,
    ReciprocalMeshPlan,
)
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve
from ...nonlinear import (
    FixedPointIteration,
    FixedPointProblem,
    NonlinearResult,
    NonlinearTermination,
)
from ...operators.periodic import (
    PeriodicBandManifold,
    PeriodicChernPlan,
    PeriodicChernRefinementEvidence,
    PeriodicChernResult,
    PeriodicCrossKConnection,
    PeriodicOverlapBundle,
    PeriodicSpectrumResult,
    PreparedPeriodicOrbitalPencil,
)
from ...operators.periodic._family import PreparedPeriodicTranslationFamily
from ...operators.quantum._fermionic_fock import FermionModeOrder
from ...operators.quantum._superconductivity import (
    BdGSpectrumResult,
    evaluate_bdg_spectrum,
    evaluate_bdg_thermal_kernel,
    evaluate_pairing_observables,
    FermionicBdGPlan,
    FermionicPairingPlan,
    NambuConvention,
    PairingObservableResult,
    prepare_fermionic_bdg,
    PreparedFermionicBdG,
)
from ...units import UnitDefinition


SuperconductingEnsemble = Literal["fixed-chemical-potential", "fixed-filling"]


class PairingChannelPlan(StrictModule):
    """Finite antisymmetric pairing basis Delta(k)=sum_a eta_a Phi_a(k)."""

    mode_order: FermionModeOrder = eqx.field(static=True)
    mesh: ReciprocalMeshPlan
    minus_k_indices: Array
    pairing_families: tuple[PreparedPeriodicTranslationFamily, ...]
    form_factors: Array
    coupling_matrix: Array
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    family_ids: tuple[str, ...] = eqx.field(static=True)
    phase_anchor: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode_order: FermionModeOrder,
        mesh: ReciprocalMeshPlan,
        minus_k_indices: ArrayLike,
        pairing_families: tuple[PreparedPeriodicTranslationFamily, ...],
        coupling_matrix: ArrayLike,
        channel_labels: tuple[str, ...],
        /,
        *,
        phase_anchor: int = 0,
        tolerance: float = 1.0e-10,
    ):
        if not isinstance(mode_order, FermionModeOrder) or not isinstance(
            mesh, ReciprocalMeshPlan
        ):
            raise TypeError(
                "Pairing channels require FermionModeOrder and ReciprocalMeshPlan."
            )
        minus = np.asarray(minus_k_indices)
        resolved_families = tuple(pairing_families)
        coupling = np.asarray(coupling_matrix, dtype=np.complex128)
        labels = tuple(str(value).strip() for value in channel_labels)
        channels = len(labels)
        modes = mode_order.mode_count
        expected = (
            channels,
            mesh.fractional_points.shape[0],
            modes,
            modes,
        )
        if (
            channels < 1
            or len(set(labels)) != channels
            or any(not value for value in labels)
        ):
            raise ValueError("Pairing channel labels must be unique and non-empty.")
        if len(resolved_families) != channels or any(
            not isinstance(family, PreparedPeriodicTranslationFamily)
            for family in resolved_families
        ):
            raise TypeError("Every pairing channel requires a prepared periodic family.")
        if any(
            family.plan.rank != mesh.rank
            or family.output_size != modes
            or family.input_size != modes
            for family in resolved_families
        ):
            raise ValueError(
                "Pairing families must align with mesh rank and fermion modes."
            )
        if (
            len({family.plan.convention.convention_id for family in resolved_families})
            != 1
        ):
            raise ValueError("All pairing channels must use one Fourier convention.")
        factors = np.stack(
            tuple(
                np.asarray(family.evaluate(mesh.fractional_points))
                for family in resolved_families
            ),
            axis=0,
        )
        families = tuple(family.prepared_id for family in resolved_families)
        if minus.shape != (expected[1],) or not np.issubdtype(minus.dtype, np.integer):
            raise TypeError("minus_k_indices must be an integer vector on the mesh.")
        minus = minus.astype(np.int32, copy=False)
        if (
            np.any(minus < 0)
            or np.any(minus >= expected[1])
            or not np.array_equal(minus[minus], np.arange(expected[1]))
        ):
            raise ValueError("The superconducting k→-k map must be an involution.")
        if factors.shape != expected or np.any(~np.isfinite(factors)):
            raise ValueError(
                f"Pairing form factors must be finite with shape {expected}."
            )
        if coupling.shape != (channels, channels) or np.any(~np.isfinite(coupling)):
            raise ValueError("Pairing coupling matrix must be finite and channel-square.")
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Pairing channel tolerance must be finite and positive.")
        paired_points = np.asarray(mesh.fractional_points)[minus] + np.asarray(
            mesh.fractional_points
        )
        if (
            np.max(np.abs(paired_points - np.rint(paired_points)), initial=0.0)
            > tolerance_
        ):
            raise ValueError("minus_k_indices do not map k to -k modulo the lattice.")
        factor_scale = max(float(np.max(np.abs(factors), initial=0.0)), 1.0)
        antisymmetry = factors + np.swapaxes(factors[:, minus], -1, -2)
        if np.max(np.abs(antisymmetry), initial=0.0) > tolerance_ * factor_scale:
            raise ValueError("Every channel must satisfy Phi_a(k)=-Phi_a^T(-k).")
        coupling_scale = max(float(np.max(np.abs(coupling), initial=0.0)), 1.0)
        if (
            np.max(np.abs(coupling - np.conj(coupling.T)), initial=0.0)
            > tolerance_ * coupling_scale
        ):
            raise ValueError("Pairing coupling matrix must be Hermitian.")
        eigenvalues = np.linalg.eigvalsh(coupling)
        if float(np.min(eigenvalues)) <= tolerance_ * coupling_scale:
            raise ValueError(
                "Attractive-channel coupling matrix must be positive definite."
            )
        anchor = int(phase_anchor)
        if not 0 <= anchor < channels:
            raise ValueError("phase_anchor is outside the channel range.")
        self.mode_order = mode_order
        self.mesh = mesh
        self.minus_k_indices = jnp.asarray(minus)
        self.pairing_families = resolved_families
        self.form_factors = jnp.asarray(factors)
        self.coupling_matrix = jnp.asarray(coupling)
        self.channel_labels = labels
        self.family_ids = families
        self.phase_anchor = anchor
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pairing-channel-plan",
                "mode_order": mode_order.order_id,
                "mesh": mesh.mesh_id,
                "labels": labels,
                "families": families,
                "phase_anchor": anchor,
                "arrays": array_tree_fingerprint(
                    {"minus_k": minus, "form_factors": factors, "coupling": coupling}
                ),
            }
        )

    @property
    def channel_count(self) -> int:
        return len(self.channel_labels)


class SuperconductingMeanFieldPlan(StrictModule):
    """Gauge-fixed finite-channel BCS closure on one orthonormal periodic pencil."""

    pencil: PreparedPeriodicOrbitalPencil
    channels: PairingChannelPlan
    ensemble: SuperconductingEnsemble = eqx.field(static=True)
    temperature_energy: float = eqx.field(static=True)
    chemical_potential: float | None = eqx.field(static=True)
    target_filling: float | None = eqx.field(static=True)
    number_update_gain: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    termination: NonlinearTermination
    minimum_gap: float = eqx.field(static=True)
    maximum_mode_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pencil: PreparedPeriodicOrbitalPencil,
        channels: PairingChannelPlan,
        /,
        *,
        ensemble: SuperconductingEnsemble,
        temperature_energy: float,
        chemical_potential: float | None = None,
        target_filling: float | None = None,
        number_update_gain: float = 0.25,
        damping: float = 0.5,
        termination: NonlinearTermination | None = None,
        minimum_gap: float = 1.0e-8,
        maximum_mode_count: int,
    ):
        if not isinstance(pencil, PreparedPeriodicOrbitalPencil) or not isinstance(
            channels, PairingChannelPlan
        ):
            raise TypeError(
                "Mean-field plan requires prepared pencil and pairing channels."
            )
        channels.mesh.require_cell(pencil.plan.basis.cell)
        if pencil.plan.basis.orbital_count != channels.mode_order.mode_count:
            raise ValueError(
                "Normal pencil and pairing channels use different mode counts."
            )
        if pencil.plan.basis.labels != channels.mode_order.labels:
            raise ValueError("Normal orbital labels must exactly match FermionModeOrder.")
        if (
            channels.pairing_families[0].plan.convention.convention_id
            != pencil.hamiltonian.plan.convention.convention_id
        ):
            raise ValueError(
                "Normal and pairing families use different Fourier conventions."
            )
        if ensemble not in ("fixed-chemical-potential", "fixed-filling"):
            raise ValueError("Unknown superconducting ensemble.")
        if ensemble == "fixed-chemical-potential":
            if chemical_potential is None or target_filling is not None:
                raise ValueError("Fixed-mu closure requires only chemical_potential.")
        elif target_filling is None or chemical_potential is not None:
            raise ValueError("Fixed-filling closure requires only target_filling.")
        thermal = float(temperature_energy)
        gain = float(number_update_gain)
        damping_ = float(damping)
        gap = float(minimum_gap)
        if any(
            not isfinite(value) or value <= 0.0
            for value in (thermal, gain, damping_, gap)
        ):
            raise ValueError(
                "Mean-field temperature, gains, damping, and gap must be positive."
            )
        if damping_ > 1.0:
            raise ValueError("Mean-field damping must not exceed one.")
        if (
            target_filling is not None
            and not 0.0 <= float(target_filling) <= channels.mode_order.mode_count
        ):
            raise ValueError("target_filling is outside the particle range per cell.")
        maximum = int(maximum_mode_count)
        if maximum < channels.mode_order.mode_count:
            raise ValueError(
                "Mean-field mode count exceeds the explicit resource policy."
            )
        termination_ = NonlinearTermination() if termination is None else termination
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        self.pencil = pencil
        self.channels = channels
        self.ensemble = ensemble
        self.temperature_energy = thermal
        self.chemical_potential = (
            None if chemical_potential is None else float(chemical_potential)
        )
        self.target_filling = None if target_filling is None else float(target_filling)
        self.number_update_gain = gain
        self.damping = damping_
        self.termination = termination_
        self.minimum_gap = gap
        self.maximum_mode_count = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "superconducting-mean-field-plan",
                "pencil": pencil.prepared_id,
                "channels": channels.plan_id,
                "ensemble": ensemble,
                "temperature_energy": thermal,
                "chemical_potential": self.chemical_potential,
                "target_filling": self.target_filling,
                "number_update_gain": gain,
                "damping": damping_,
                "minimum_gap": gap,
                "maximum_mode_count": maximum,
            }
        )


class SuperconductingMeanFieldResult(StrictModule):
    channel_amplitudes: Array
    chemical_potential: Array
    gap_residual: Array
    number_residual: Array
    phase_anchor_residual: Array
    nonlinear_result: NonlinearResult
    bdg: PreparedFermionicBdG
    spectrum: BdGSpectrumResult
    observables: PairingObservableResult
    successful: Array
    branch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _decode_state(
    plan: SuperconductingMeanFieldPlan, state: Array, /
) -> tuple[Array, Array]:
    count = plan.channels.channel_count
    amplitudes = state[:count] + 1.0j * state[count : 2 * count]
    chemical = (
        jnp.asarray(plan.chemical_potential, dtype=state.dtype)
        if plan.ensemble == "fixed-chemical-potential"
        else state[-1]
    )
    return amplitudes, chemical


def solve_superconducting_mean_field(
    plan: SuperconductingMeanFieldPlan,
    initial_channel_amplitudes: ArrayLike,
    /,
    *,
    initial_chemical_potential: float | None = None,
) -> SuperconductingMeanFieldResult:
    if not isinstance(plan, SuperconductingMeanFieldPlan):
        raise TypeError("plan must be SuperconductingMeanFieldPlan.")
    initial = np.asarray(initial_channel_amplitudes, dtype=np.complex128)
    count = plan.channels.channel_count
    if initial.shape != (count,) or np.any(~np.isfinite(initial)):
        raise ValueError("Initial channel amplitudes must be one finite complex vector.")
    anchor = plan.channels.phase_anchor
    if (
        abs(initial[anchor].imag) > plan.channels.tolerance
        or initial[anchor].real <= plan.channels.tolerance
    ):
        raise ValueError("Initial phase anchor must be real and strictly positive.")
    if plan.ensemble == "fixed-filling":
        if initial_chemical_potential is None or not isfinite(
            float(initial_chemical_potential)
        ):
            raise ValueError(
                "Fixed-filling closure requires finite initial chemical potential."
            )
    elif initial_chemical_potential is not None:
        raise ValueError("Fixed-mu closure does not accept initial_chemical_potential.")
    pencil = plan.pencil.evaluate(plan.channels.mesh.fractional_points)
    if not bool(pencil.successful):
        raise ValueError("Superconducting closure requires a valid periodic pencil.")
    identity = np.eye(plan.channels.mode_order.mode_count)
    overlap_defect = np.max(
        np.abs(np.asarray(pencil.overlaps) - identity[None, :, :]), initial=0.0
    )
    if overlap_defect > plan.channels.tolerance:
        raise ValueError(
            "Initial BdG closure supports only an orthonormal normal pencil."
        )
    normal = pencil.hamiltonians
    weights = plan.channels.mesh.weights
    factors = plan.channels.form_factors
    minus = plan.channels.minus_k_indices

    def mapping(state, args):
        del args
        amplitudes, chemical = _decode_state(plan, state)
        pairing = ein.contract("a,akij->kij", amplitudes, factors)
        _, _, _, density = evaluate_bdg_thermal_kernel(
            normal, pairing, minus, chemical, plan.temperature_energy
        )
        modes = plan.channels.mode_order.mode_count
        anomalous = density[:, :modes, modes:]
        projections = 0.5 * ein.contract(
            "akij,kij,k->a", jnp.conj(factors), anomalous, weights
        )
        updated = plan.channels.coupling_matrix @ projections
        anchor_value = updated[anchor]
        safe_magnitude = jnp.maximum(jnp.abs(anchor_value), plan.channels.tolerance)
        updated = updated * jnp.conj(anchor_value) / safe_magnitude
        updated = updated.at[anchor].set(jnp.real(updated[anchor]))
        encoded = jnp.concatenate((jnp.real(updated), jnp.imag(updated)))
        if plan.ensemble == "fixed-filling":
            normal_density = density[:, :modes, :modes]
            number = jnp.real(
                jnp.sum(weights * jnp.trace(normal_density, axis1=-2, axis2=-1))
            )
            chemical_updated = chemical + plan.number_update_gain * (
                jnp.asarray(plan.target_filling) - number
            )
            encoded = jnp.concatenate((encoded, chemical_updated[None]))
        return encoded

    encoded_initial = np.concatenate((initial.real, initial.imag))
    if plan.ensemble == "fixed-filling":
        encoded_initial = np.concatenate(
            (encoded_initial, np.asarray((float(initial_chemical_potential),)))
        )
    problem = FixedPointProblem(mapping, problem_id=f"superconducting-gap:{plan.plan_id}")
    nonlinear = FixedPointIteration(damping=plan.damping).solve(
        problem,
        jnp.asarray(encoded_initial),
        termination=plan.termination,
    )
    amplitudes, chemical = _decode_state(plan, nonlinear.state)
    pairing = ein.contract("a,akij->kij", amplitudes, factors)
    inverse_coupling = solve(
        LinearSystem(DenseLinearOperator(plan.channels.coupling_matrix)),
        amplitudes,
        policy=LinearSolvePolicy(DenseLU()),
    )
    if not bool(inverse_coupling.successful):
        raise RuntimeError("Pairing double-counting coupling solve failed.")
    double_counting = jnp.real(jnp.vdot(amplitudes, inverse_coupling.value))
    pairing_plan = FermionicPairingPlan(
        plan.channels.mode_order,
        plan.channels.minus_k_indices,
        plan.channels.mesh.weights,
        mesh_id=plan.channels.mesh.mesh_id,
        energy_unit=plan.pencil.plan.energy_unit.symbol,
        tolerance=plan.channels.tolerance,
    )
    bdg_plan = FermionicBdGPlan(
        NambuConvention(plan.channels.mode_order),
        pairing_plan,
        maximum_mode_count=plan.maximum_mode_count,
    )
    bdg = prepare_fermionic_bdg(
        bdg_plan,
        normal,
        pairing,
        chemical_potential=chemical,
        double_counting_constant=double_counting,
        normal_family_id=plan.pencil.hamiltonian.prepared_id,
        pairing_family_id=canonical_fingerprint(
            {
                "kind": "finite-pairing-channel-composition",
                "families": plan.channels.family_ids,
            }
        ),
    )
    spectrum = evaluate_bdg_spectrum(bdg)
    observables = evaluate_pairing_observables(
        bdg, temperature_energy=plan.temperature_energy
    )
    mapped = mapping(nonlinear.state, None)
    residual = mapped - nonlinear.state
    gap_residual = jnp.sqrt(jnp.sum(residual[: 2 * count] ** 2))
    number_residual = (
        jnp.asarray(0.0)
        if plan.ensemble == "fixed-chemical-potential"
        else jnp.abs(residual[-1]) / plan.number_update_gain
    )
    anchor_residual = jnp.abs(jnp.imag(amplitudes[anchor])) + jnp.maximum(
        -jnp.real(amplitudes[anchor]), 0.0
    )
    successful = (
        nonlinear.successful
        & spectrum.successful
        & (spectrum.minimum_direct_gap >= plan.minimum_gap)
        & (
            gap_residual
            <= plan.termination.residual_threshold(
                nonlinear.diagnostics.initial_residual_norm
            )
        )
        & (anchor_residual <= plan.channels.tolerance)
        & (jnp.real(amplitudes[anchor]) > plan.channels.tolerance)
    )
    if not bool(successful):
        raise ValueError(
            "Superconducting branch failed convergence, gauge anchor, or gap evidence."
        )
    branch_id = canonical_fingerprint(
        {
            "kind": "superconducting-mean-field-branch",
            "plan": plan.plan_id,
            "amplitudes": array_tree_fingerprint(np.asarray(amplitudes)),
            "chemical_potential": float(chemical),
        }
    )
    return SuperconductingMeanFieldResult(
        amplitudes,
        chemical,
        gap_residual,
        number_residual,
        anchor_residual,
        nonlinear,
        bdg,
        spectrum,
        observables,
        successful,
        branch_id,
        plan.plan_id,
    )


class BdGChernPlan(StrictModule):
    """Class-D composition through the canonical periodic Chern owner."""

    connectivity: PreparedReciprocalConnectivity
    nambu_connection_matrices: Array
    basis_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    minimum_gap: float = eqx.field(static=True)
    energy_unit: UnitDefinition
    refinement: PeriodicChernRefinementEvidence | None
    require_refinement: bool = eqx.field(static=True)
    quantization_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        connectivity: PreparedReciprocalConnectivity,
        nambu_connection_matrices: ArrayLike,
        /,
        *,
        basis_id: str,
        source_id: str,
        minimum_gap: float,
        energy_unit: UnitDefinition,
        refinement: PeriodicChernRefinementEvidence | None = None,
        require_refinement: bool = True,
        quantization_tolerance: float = 1.0e-6,
    ):
        if not isinstance(connectivity, PreparedReciprocalConnectivity):
            raise TypeError("connectivity must be PreparedReciprocalConnectivity.")
        matrices = np.asarray(nambu_connection_matrices, dtype=np.complex128)
        if (
            matrices.ndim != 3
            or matrices.shape[0] != connectivity.edge_count
            or matrices.shape[1] != matrices.shape[2]
            or matrices.shape[1] % 2
            or np.any(~np.isfinite(matrices))
        ):
            raise ValueError("Nambu cross-k connection matrices are invalid.")
        basis = str(basis_id).strip()
        source = str(source_id).strip()
        gap = float(minimum_gap)
        tolerance = float(quantization_tolerance)
        if not basis or not source or not isfinite(gap) or gap <= 0.0:
            raise ValueError("BdG Chern basis/source IDs and minimum gap are required.")
        if not isinstance(energy_unit, UnitDefinition):
            raise TypeError("energy_unit must be UnitDefinition.")
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("quantization_tolerance must be finite and non-negative.")
        if refinement is not None and not isinstance(
            refinement, PeriodicChernRefinementEvidence
        ):
            raise TypeError("refinement must be PeriodicChernRefinementEvidence or None.")
        if require_refinement and (refinement is None or not refinement.successful):
            raise ValueError(
                "Class-D Chern requires successful mesh refinement evidence."
            )
        self.connectivity = connectivity
        self.nambu_connection_matrices = jnp.asarray(matrices)
        self.basis_id = basis
        self.source_id = source
        self.energy_unit = energy_unit
        self.minimum_gap = gap
        self.refinement = refinement
        self.require_refinement = bool(require_refinement)
        self.quantization_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bdg-class-d-chern-plan",
                "connectivity": connectivity.prepared_id,
                "basis": basis,
                "source": source,
                "energy_unit": energy_unit.unit_id,
                "minimum_gap": gap,
                "refinement": None if refinement is None else refinement.evidence_id,
                "connection": array_tree_fingerprint(matrices),
            }
        )


class BdGTopologyResult(StrictModule):
    chern: PeriodicChernResult
    minimum_zero_energy_gap: Array
    negative_energy_rank: int = eqx.field(static=True)
    successful: Array
    plan_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)


def evaluate_bdg_chern(
    plan: BdGChernPlan,
    mean_field: SuperconductingMeanFieldResult,
    /,
) -> BdGTopologyResult:
    if not isinstance(plan, BdGChernPlan) or not isinstance(
        mean_field, SuperconductingMeanFieldResult
    ):
        raise TypeError("Class-D Chern requires typed plan and mean-field result.")
    spectrum = mean_field.spectrum
    mesh = plan.connectivity.plan.mesh
    if spectrum.mesh_id != mesh.mesh_id:
        raise ValueError("BdG spectrum and Chern connectivity use different meshes.")
    if (
        not bool(mean_field.successful)
        or float(spectrum.minimum_direct_gap) < plan.minimum_gap
    ):
        raise ValueError("Class-D Chern requires a converged fully gapped BdG branch.")
    if plan.basis_id != mean_field.bdg.plan.convention.convention_id:
        raise ValueError("BdG Chern basis ID does not match the Nambu convention.")
    modes = mean_field.bdg.plan.convention.mode_order.mode_count
    if plan.nambu_connection_matrices.shape[1] != 2 * modes:
        raise ValueError("Nambu connection dimension does not match the BdG basis.")
    residuals = jnp.sqrt(
        jnp.sum(
            jnp.abs(
                spectrum.hamiltonians @ spectrum.eigenvectors
                - spectrum.eigenvectors * spectrum.eigenvalues[:, None, :]
            )
            ** 2,
            axis=1,
        )
    )
    gram = jnp.conj(jnp.swapaxes(spectrum.eigenvectors, -1, -2)) @ spectrum.eigenvectors
    metric_residual = jnp.max(
        jnp.abs(gram - jnp.eye(2 * modes, dtype=gram.dtype)[None, :, :]), axis=(-2, -1)
    )
    periodic_spectrum = PeriodicSpectrumResult(
        mesh.fractional_points,
        mesh.weights,
        jnp.zeros_like(mesh.weights),
        spectrum.eigenvalues,
        spectrum.eigenvectors,
        residuals,
        metric_residual,
        jnp.ones_like(mesh.weights),
        jnp.ones_like(mesh.weights),
        spectrum.successful,
        plan.energy_unit,
        cell_id=mesh.cell_id,
        basis_id=plan.basis_id,
        support_id=mesh.mesh_id,
        pencil_id=mean_field.bdg.prepared_id,
    )
    manifold = PeriodicBandManifold(
        periodic_spectrum,
        np.arange(modes),
        gap_tolerance=plan.minimum_gap,
    )
    connection = PeriodicCrossKConnection(
        plan.connectivity,
        plan.nambu_connection_matrices,
        plan.basis_id,
        plan.source_id,
    )
    bundle = PeriodicOverlapBundle(manifold, connection)
    chern = PeriodicChernPlan(
        bundle,
        refinement=plan.refinement,
        require_refinement=plan.require_refinement,
        quantization_tolerance=plan.quantization_tolerance,
    ).evaluate()
    successful = mean_field.successful & chern.successful
    if not bool(successful):
        raise ValueError(
            "Class-D Chern failed link, quantization, or refinement evidence."
        )
    return BdGTopologyResult(
        chern,
        spectrum.minimum_direct_gap,
        modes,
        successful,
        plan.plan_id,
        mean_field.branch_id,
    )


__all__ = [
    "BdGChernPlan",
    "BdGTopologyResult",
    "PairingChannelPlan",
    "SuperconductingMeanFieldPlan",
    "SuperconductingMeanFieldResult",
    "evaluate_bdg_chern",
    "solve_superconducting_mean_field",
]
