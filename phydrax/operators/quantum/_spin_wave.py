#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._reciprocal import ReciprocalMeshPlan
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import (
    DenseEigh,
    DenseSchurQZ,
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    general_eigensolve,
    GeneralEigenproblem,
    GeneralEigenSelection,
    GeneralEigenSolvePolicy,
)
from ...sparse import EdgeRelation
from ..periodic._family import (
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    prepare_periodic_translation_family,
    PreparedPeriodicTranslationFamily,
)


class SpinWaveReferenceState(StrictModule):
    """Caller-supplied collinear ordered state in primitive-cell site order."""

    directions: Array
    spin_magnitudes: Array
    collinear_signs: Array
    common_axis: Array
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        directions: ArrayLike,
        spin_magnitudes: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
    ):
        values = np.asarray(directions, dtype=np.float64)
        magnitudes = np.asarray(spin_magnitudes, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 3
            or magnitudes.shape != values.shape[:1]
        ):
            raise ValueError(
                "Reference directions and spin magnitudes must have shapes (site,3) and (site,)."
            )
        if (
            np.any(~np.isfinite(values))
            or np.any(~np.isfinite(magnitudes))
            or np.any(magnitudes <= 0.0)
        ):
            raise ValueError("Reference spins must be finite with positive magnitudes.")
        norms = np.linalg.norm(values, axis=-1)
        if np.any(np.abs(norms - 1.0) > tolerance):
            raise ValueError("Spin-wave reference directions must be unit vectors.")
        axis = values[0]
        projections = values @ axis
        signs = np.where(projections >= 0.0, 1, -1)
        if (
            np.max(
                np.linalg.norm(values - signs[:, None] * axis[None, :], axis=-1),
                initial=0.0,
            )
            > tolerance
        ):
            raise ValueError("Released LSWT accepts only a collinear reference state.")
        self.directions = jnp.asarray(values)
        self.spin_magnitudes = jnp.asarray(magnitudes)
        self.collinear_signs = jnp.asarray(signs, dtype=jnp.int8)
        self.common_axis = jnp.asarray(axis)
        self.reference_id = canonical_fingerprint(
            {
                "kind": "spin-wave-reference-state",
                "arrays": array_tree_fingerprint(
                    {"directions": values, "spin_magnitudes": magnitudes}
                ),
            }
        )

    @property
    def site_count(self) -> int:
        return self.spin_magnitudes.size


class LinearSpinWavePlan(StrictModule):
    """Stable collinear bosonic Krein problem over canonical periodic families."""

    reference: SpinWaveReferenceState
    mesh: ReciprocalMeshPlan
    minus_q_indices: Array
    declared_goldstone_counts: Array
    mesh_id: str = eqx.field(static=True)
    energy_unit: str = eqx.field(static=True)
    torque_tolerance: float = eqx.field(static=True)
    stability_tolerance: float = eqx.field(static=True)
    krein_tolerance: float = eqx.field(static=True)
    maximum_site_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference: SpinWaveReferenceState,
        mesh: ReciprocalMeshPlan,
        minus_q_indices: ArrayLike,
        declared_goldstone_counts: ArrayLike,
        /,
        *,
        energy_unit: str,
        maximum_site_count: int,
        torque_tolerance: float = 1.0e-9,
        stability_tolerance: float = 1.0e-9,
        krein_tolerance: float = 1.0e-9,
    ):
        if not isinstance(reference, SpinWaveReferenceState):
            raise TypeError("reference must be SpinWaveReferenceState.")
        if not isinstance(mesh, ReciprocalMeshPlan):
            raise TypeError("mesh must be ReciprocalMeshPlan.")
        minus = np.asarray(minus_q_indices)
        goldstone = np.asarray(declared_goldstone_counts)
        if (
            minus.ndim != 1
            or minus.size != mesh.fractional_points.shape[0]
            or not np.issubdtype(minus.dtype, np.integer)
        ):
            raise TypeError(
                "minus_q_indices must be an integer vector on the reciprocal mesh."
            )
        if goldstone.shape != minus.shape or not np.issubdtype(
            goldstone.dtype, np.integer
        ):
            raise TypeError(
                "declared_goldstone_counts must be an integer vector on q points."
            )
        minus = minus.astype(np.int32, copy=False)
        goldstone = goldstone.astype(np.int32, copy=False)
        if (
            np.any(minus < 0)
            or np.any(minus >= minus.size)
            or not np.array_equal(minus[minus], np.arange(minus.size))
        ):
            raise ValueError("The q→-q map must be an involution.")
        if np.any(goldstone < 0) or np.any(goldstone > reference.site_count):
            raise ValueError("Declared Goldstone counts are outside the branch range.")
        maximum = int(maximum_site_count)
        if maximum < reference.site_count:
            raise ValueError("LSWT site count exceeds the explicit resource policy.")
        tolerances = tuple(
            float(value)
            for value in (torque_tolerance, stability_tolerance, krein_tolerance)
        )
        if any(not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("LSWT tolerances must be finite and positive.")
        paired_points = np.asarray(mesh.fractional_points)[minus] + np.asarray(
            mesh.fractional_points
        )
        if (
            np.max(np.abs(paired_points - np.rint(paired_points)), initial=0.0)
            > tolerances[1]
        ):
            raise ValueError("minus_q_indices do not map q to -q modulo the lattice.")
        unit = str(energy_unit).strip()
        if not unit:
            raise ValueError("LSWT energy_unit must be explicit.")
        self.reference = reference
        self.mesh = mesh
        self.minus_q_indices = jnp.asarray(minus)
        self.declared_goldstone_counts = jnp.asarray(goldstone)
        self.mesh_id = mesh.mesh_id
        self.energy_unit = unit
        self.torque_tolerance, self.stability_tolerance, self.krein_tolerance = tolerances
        self.maximum_site_count = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "linear-spin-wave-plan",
                "reference": reference.reference_id,
                "mesh": mesh.mesh_id,
                "energy_unit": unit,
                "minus_q": minus.tolist(),
                "goldstone": goldstone.tolist(),
                "maximum_site_count": maximum,
                "tolerances": tolerances,
            }
        )

    @property
    def q_count(self) -> int:
        return self.minus_q_indices.size


class PreparedLinearSpinWave(StrictModule):
    plan: LinearSpinWavePlan
    normal_family: PreparedPeriodicTranslationFamily
    pairing_family: PreparedPeriodicTranslationFamily
    normal_blocks: Array
    pairing_blocks: Array
    reference_torques: Array
    prepared_id: str = eqx.field(static=True)


class LinearSpinWaveResult(StrictModule):
    frequencies: Array
    positive_modes: Array
    negative_modes: Array
    krein_norms: Array
    dynamical_matrices: Array
    quadratic_hamiltonians: Array
    reference_torque_residual: Array
    quadratic_hermiticity_residual: Array
    paraunitarity_residual: Array
    frequency_pairing_residual: Array
    goldstone_residual: Array
    minimum_stability_eigenvalue: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)


class CollinearSpinWaveLowering(StrictModule):
    """Quadratic Holstein–Primakoff families and stationarity evidence."""

    normal_family: PreparedPeriodicTranslationFamily
    pairing_family: PreparedPeriodicTranslationFamily
    reference_torques: Array
    classical_reference_energy: Array
    lowering_id: str = eqx.field(static=True)


def spin_exchange_tensor(
    exchange: ArrayLike,
    symmetric_exchange: ArrayLike,
    dmi: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> Array:
    """Return J I + Gamma - [D]_x for ``-S_i^T tensor S_j``."""

    scalar = np.asarray(exchange)
    gamma = np.asarray(symmetric_exchange)
    dmi_ = np.asarray(dmi)
    tolerance_ = float(tolerance)
    if (
        scalar.ndim != 1
        or gamma.shape != scalar.shape + (3, 3)
        or dmi_.shape != scalar.shape + (3,)
    ):
        raise ValueError("Exchange, Gamma, and DMI arrays have incompatible bond shapes.")
    if (
        not isfinite(tolerance_)
        or tolerance_ <= 0.0
        or np.any(~np.isfinite(scalar))
        or np.any(~np.isfinite(gamma))
        or np.any(~np.isfinite(dmi_))
        or not np.isrealobj(scalar)
        or not np.isrealobj(gamma)
        or not np.isrealobj(dmi_)
    ):
        raise ValueError("Exchange, Gamma, and DMI values must be finite and real.")
    scale = max(float(np.max(np.abs(gamma), initial=0.0)), 1.0)
    if (
        np.max(np.abs(gamma - np.swapaxes(gamma, -1, -2)), initial=0.0)
        > tolerance_ * scale
        or np.max(np.abs(np.trace(gamma, axis1=-2, axis2=-1)), initial=0.0)
        > tolerance_ * scale
    ):
        raise ValueError("Gamma exchange tensors must be symmetric and traceless.")
    cross = np.zeros(gamma.shape, dtype=np.result_type(gamma, dmi_))
    cross[:, 0, 1] = -dmi_[:, 2]
    cross[:, 0, 2] = dmi_[:, 1]
    cross[:, 1, 0] = dmi_[:, 2]
    cross[:, 1, 2] = -dmi_[:, 0]
    cross[:, 2, 0] = -dmi_[:, 1]
    cross[:, 2, 1] = dmi_[:, 0]
    identity = np.eye(3, dtype=cross.dtype)
    return jnp.asarray(scalar[:, None, None] * identity + gamma - cross)


def _family_from_scalar_routes(
    coefficients: dict[tuple[int, int, tuple[int, ...]], complex],
    *,
    site_count: int,
    rank: int,
    hermitian: bool,
    maximum_dense_entries: int,
) -> PreparedPeriodicTranslationFamily:
    keys = tuple(sorted(coefficients))
    source = np.asarray([key[1] for key in keys], dtype=np.int32)
    target = np.asarray([key[0] for key in keys], dtype=np.int32)
    translations = np.asarray([key[2] for key in keys], dtype=np.int32)
    lookup = {key: index for index, key in enumerate(keys)}
    reverse = np.asarray(
        [
            lookup[(source[index], target[index], tuple(-translations[index]))]
            for index in range(len(keys))
        ],
        dtype=np.int32,
    )
    relation = EdgeRelation(
        source,
        target,
        source_size=site_count,
        target_size=site_count,
    )
    family_plan = PeriodicTranslationFamilyPlan(
        relation,
        translations,
        reverse,
        hermitian=hermitian,
        maximum_dense_entries=maximum_dense_entries,
    )
    values = np.asarray([coefficients[key] for key in keys], dtype=np.complex128).reshape(
        (-1, 1, 1)
    )
    family_state = PeriodicTranslationFamilyState(family_plan, values)
    return prepare_periodic_translation_family(family_plan, family_state)


def lower_collinear_spin_wave_bonds(
    plan: LinearSpinWavePlan,
    transverse_frames: ArrayLike,
    bond_senders: ArrayLike,
    bond_receivers: ArrayLike,
    translations: ArrayLike,
    exchange_tensors: ArrayLike,
    /,
    *,
    anisotropy: ArrayLike | None = None,
    anisotropy_axes: ArrayLike | None = None,
    zeeman_energies: ArrayLike | None = None,
    maximum_dense_entries: int = 4_000_000,
) -> CollinearSpinWaveLowering:
    """Lower a once-oriented stationary collinear model to canonical families.

    ``exchange_tensors[b]`` enters ``-S_i^T J_b S_j`` and may be constructed
    from exchange/Gamma/DMI by :func:`spin_exchange_tensor`.
    ``zeeman_energies`` is the vector h in the onsite term ``-h_i·S_i``.
    """

    if not isinstance(plan, LinearSpinWavePlan):
        raise TypeError("plan must be LinearSpinWavePlan.")
    sites = plan.reference.site_count
    rank = plan.mesh.rank
    frames = np.asarray(transverse_frames, dtype=np.float64)
    senders = np.asarray(bond_senders)
    receivers = np.asarray(bond_receivers)
    translations_ = np.asarray(translations)
    tensors = np.asarray(exchange_tensors, dtype=np.complex128)
    bonds = senders.size
    if (
        frames.shape != (sites, 3, 2)
        or senders.ndim != 1
        or receivers.shape != (bonds,)
        or translations_.shape != (bonds, rank)
        or tensors.shape != (bonds, 3, 3)
        or not np.issubdtype(senders.dtype, np.integer)
        or not np.issubdtype(receivers.dtype, np.integer)
        or not np.issubdtype(translations_.dtype, np.integer)
    ):
        raise ValueError(
            "Collinear HP frames, bonds, translations, or tensors have invalid shapes."
        )
    if (
        np.any(~np.isfinite(frames))
        or np.any(~np.isfinite(tensors))
        or np.any(senders < 0)
        or np.any(senders >= sites)
        or np.any(receivers < 0)
        or np.any(receivers >= sites)
    ):
        raise ValueError(
            "Collinear HP inputs must be finite with in-range bond endpoints."
        )
    if np.max(np.abs(np.imag(tensors)), initial=0.0) > plan.stability_tolerance:
        raise ValueError("Released collinear LSWT exchange tensors must be real.")
    tensors = np.real(tensors)
    x_axes = frames[:, :, 0]
    y_axes = frames[:, :, 1]
    z_axes = np.asarray(plan.reference.directions)
    frame_defect = max(
        float(np.max(np.abs(np.sum(x_axes * x_axes, axis=-1) - 1.0), initial=0.0)),
        float(np.max(np.abs(np.sum(y_axes * y_axes, axis=-1) - 1.0), initial=0.0)),
        float(np.max(np.abs(np.sum(x_axes * y_axes, axis=-1)), initial=0.0)),
        float(np.max(np.abs(np.sum(x_axes * z_axes, axis=-1)), initial=0.0)),
        float(np.max(np.abs(np.sum(y_axes * z_axes, axis=-1)), initial=0.0)),
        float(np.max(np.abs(np.cross(x_axes, y_axes) - z_axes), initial=0.0)),
    )
    if frame_defect > plan.stability_tolerance:
        raise ValueError(
            "Transverse frames must be right-handed orthonormal frames of the reference."
        )
    oriented = set()
    for sender, receiver, translation in zip(
        senders, receivers, translations_, strict=True
    ):
        key = (int(sender), int(receiver), tuple(translation))
        reverse_key = (
            int(receiver),
            int(sender),
            tuple(-int(value) for value in translation),
        )
        if key == reverse_key or key in oriented or reverse_key in oriented:
            raise ValueError(
                "Every LSWT physical bond must appear once in one orientation."
            )
        oriented.add(key)
    k = (
        np.zeros((sites,), dtype=np.float64)
        if anisotropy is None
        else np.asarray(anisotropy, dtype=np.float64)
    )
    axes = (
        np.broadcast_to([0.0, 0.0, 1.0], (sites, 3)).copy()
        if anisotropy_axes is None
        else np.asarray(anisotropy_axes, dtype=np.float64)
    )
    fields = (
        np.zeros((sites, 3), dtype=np.float64)
        if zeeman_energies is None
        else np.asarray(zeeman_energies, dtype=np.float64)
    )
    if (
        k.shape != (sites,)
        or axes.shape != (sites, 3)
        or fields.shape != (sites, 3)
        or np.any(~np.isfinite(k))
        or np.any(~np.isfinite(axes))
        or np.any(~np.isfinite(fields))
    ):
        raise ValueError("LSWT onsite anisotropy and Zeeman arrays are invalid.")
    active_axes = np.abs(k) > 0.0
    if np.any(
        active_axes
        & (np.abs(np.linalg.norm(axes, axis=-1) - 1.0) > plan.stability_tolerance)
    ):
        raise ValueError("Nonzero LSWT anisotropy requires a unit axis.")
    spins = np.asarray(plan.reference.spin_magnitudes)
    annihilation = x_axes - 1.0j * y_axes
    creation = x_axes + 1.0j * y_axes
    effective_fields = fields.copy()
    reference_energy = -float(np.sum(spins * np.sum(fields * z_axes, axis=-1)))
    zero_translation = (0,) * rank
    normal: dict[tuple[int, int, tuple[int, ...]], complex] = {
        (site, site, zero_translation): 0.0j for site in range(sites)
    }
    pairing: dict[tuple[int, int, tuple[int, ...]], complex] = {
        (site, site, zero_translation): 0.0j for site in range(sites)
    }

    def add(
        target: dict[tuple[int, int, tuple[int, ...]], complex],
        key: tuple[int, int, tuple[int, ...]],
        value: complex,
    ) -> None:
        target[key] = target.get(key, 0.0j) + value

    for bond in range(bonds):
        left = int(senders[bond])
        right = int(receivers[bond])
        translation = tuple(translations_[bond])
        reverse_translation = tuple(-value for value in translation)
        tensor = tensors[bond]
        left_spin = spins[left]
        right_spin = spins[right]
        longitudinal = np.real(z_axes[left] @ tensor @ z_axes[right])
        if abs(np.imag(z_axes[left] @ tensor @ z_axes[right])) > plan.stability_tolerance:
            raise ValueError("LSWT longitudinal exchange energy must be real.")
        normal[(left, left, zero_translation)] += right_spin * longitudinal
        normal[(right, right, zero_translation)] += left_spin * longitudinal
        hopping = (
            -0.5
            * np.sqrt(left_spin * right_spin)
            * (creation[left] @ tensor @ annihilation[right])
        )
        add(normal, (left, right, translation), hopping)
        add(normal, (right, left, reverse_translation), np.conj(hopping))
        pair = (
            -0.5
            * np.sqrt(left_spin * right_spin)
            * (creation[left] @ tensor @ creation[right])
        )
        add(pairing, (left, right, translation), pair)
        add(pairing, (right, left, reverse_translation), pair)
        effective_fields[left] += np.real(tensor @ (right_spin * z_axes[right]))
        effective_fields[right] += np.real(tensor.T @ (left_spin * z_axes[left]))
        reference_energy -= float(
            left_spin * right_spin * np.real(z_axes[left] @ tensor @ z_axes[right])
        )
    for site in range(sites):
        projection = float(z_axes[site] @ axes[site])
        transverse = annihilation[site] @ axes[site]
        normal[(site, site, zero_translation)] += (
            2.0 * k[site] * spins[site] * projection**2
            - k[site] * spins[site] * abs(transverse) ** 2
            + fields[site] @ z_axes[site]
        )
        pairing[(site, site, zero_translation)] += (
            -k[site] * spins[site] * (creation[site] @ axes[site]) ** 2
        )
        effective_fields[site] += 2.0 * k[site] * spins[site] * projection * axes[site]
        reference_energy -= k[site] * spins[site] ** 2 * projection**2
    torques = np.cross(z_axes, effective_fields)
    normal_family = _family_from_scalar_routes(
        normal,
        site_count=sites,
        rank=rank,
        hermitian=True,
        maximum_dense_entries=maximum_dense_entries,
    )
    pairing_family = _family_from_scalar_routes(
        pairing,
        site_count=sites,
        rank=rank,
        hermitian=False,
        maximum_dense_entries=maximum_dense_entries,
    )
    lowering_id = canonical_fingerprint(
        {
            "kind": "collinear-spin-wave-holstein-primakoff-lowering",
            "plan": plan.plan_id,
            "normal": normal_family.prepared_id,
            "pairing": pairing_family.prepared_id,
            "frames": array_tree_fingerprint(frames),
            "bonds": array_tree_fingerprint(
                {
                    "senders": senders,
                    "receivers": receivers,
                    "translations": translations_,
                    "exchange_tensors": tensors,
                }
            ),
        }
    )
    return CollinearSpinWaveLowering(
        normal_family,
        pairing_family,
        jnp.asarray(torques),
        jnp.asarray(reference_energy),
        lowering_id,
    )


def prepare_linear_spin_wave(
    plan: LinearSpinWavePlan,
    normal_family: PreparedPeriodicTranslationFamily,
    pairing_family: PreparedPeriodicTranslationFamily,
    reference_torques: ArrayLike,
    /,
) -> PreparedLinearSpinWave:
    """Bind HP quadratic periodic-family evaluations without a second phase loop."""

    if not isinstance(plan, LinearSpinWavePlan):
        raise TypeError("plan must be LinearSpinWavePlan.")
    if not isinstance(normal_family, PreparedPeriodicTranslationFamily) or not isinstance(
        pairing_family, PreparedPeriodicTranslationFamily
    ):
        raise TypeError(
            "LSWT normal and pairing inputs must be prepared periodic families."
        )
    if (
        normal_family.plan.rank != plan.mesh.rank
        or pairing_family.plan.rank != plan.mesh.rank
        or normal_family.output_size != plan.reference.site_count
        or normal_family.input_size != plan.reference.site_count
        or pairing_family.output_size != plan.reference.site_count
        or pairing_family.input_size != plan.reference.site_count
        or not normal_family.plan.hermitian
        or normal_family.plan.convention.convention_id
        != pairing_family.plan.convention.convention_id
    ):
        raise ValueError(
            "LSWT periodic families do not match the reciprocal mesh and site basis."
        )
    normal = np.asarray(
        normal_family.evaluate(plan.mesh.fractional_points), dtype=np.complex128
    )
    pairing = np.asarray(
        pairing_family.evaluate(plan.mesh.fractional_points), dtype=np.complex128
    )
    torque = np.asarray(reference_torques, dtype=np.float64)
    expected = (plan.q_count, plan.reference.site_count, plan.reference.site_count)
    if normal.shape != expected or pairing.shape != expected:
        raise ValueError(f"LSWT normal and pairing blocks must have shape {expected}.")
    if torque.shape != (plan.reference.site_count, 3):
        raise ValueError("reference_torques must have shape (site, 3).")
    if (
        np.any(~np.isfinite(normal))
        or np.any(~np.isfinite(pairing))
        or np.any(~np.isfinite(torque))
    ):
        raise ValueError("LSWT inputs must be finite.")
    scale = max(float(np.max(np.abs(normal), initial=0.0)), 1.0)
    if (
        np.max(np.abs(normal - np.conj(np.swapaxes(normal, -1, -2))), initial=0.0)
        > plan.stability_tolerance * scale
    ):
        raise ValueError("Every LSWT normal block must be Hermitian.")
    minus = np.asarray(plan.minus_q_indices)
    pair_defect = pairing - np.swapaxes(pairing[minus], -1, -2)
    pair_scale = max(float(np.max(np.abs(pairing), initial=0.0)), 1.0)
    if np.max(np.abs(pair_defect), initial=0.0) > plan.stability_tolerance * pair_scale:
        raise ValueError("Bosonic pairing must satisfy B(q)=B^T(-q).")
    torque_residual = float(np.max(np.linalg.norm(torque, axis=-1), initial=0.0))
    if torque_residual > plan.torque_tolerance:
        raise ValueError("LSWT reference is not torque-stationary.")
    normal_id = normal_family.prepared_id
    pairing_id = pairing_family.prepared_id
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-linear-spin-wave",
            "plan": plan.plan_id,
            "normal_family": normal_id,
            "pairing_family": pairing_id,
            "arrays": array_tree_fingerprint(
                {"normal": normal, "pairing": pairing, "torque": torque}
            ),
        }
    )
    return PreparedLinearSpinWave(
        plan,
        normal_family,
        pairing_family,
        jnp.asarray(normal),
        jnp.asarray(pairing),
        jnp.asarray(torque),
        prepared_id,
    )


def _solve_krein(matrix: Array, /) -> tuple[Array, Array]:
    solved = general_eigensolve(
        GeneralEigenproblem(DenseLinearOperator(matrix)),
        policy=GeneralEigenSolvePolicy(
            DenseSchurQZ(), selection=GeneralEigenSelection.all()
        ),
    )
    if not bool(solved.successful):
        raise RuntimeError("LSWT bosonic Krein eigensolve failed.")
    return solved.eigenvalues, solved.right_eigenvector_coordinates


def evaluate_linear_spin_wave(
    prepared: PreparedLinearSpinWave, /
) -> LinearSpinWaveResult:
    if not isinstance(prepared, PreparedLinearSpinWave):
        raise TypeError("prepared must be PreparedLinearSpinWave.")
    plan = prepared.plan
    sites = plan.reference.site_count
    minus = plan.minus_q_indices
    upper = jnp.concatenate((prepared.normal_blocks, prepared.pairing_blocks), axis=-1)
    lower = jnp.concatenate(
        (
            jnp.conj(jnp.swapaxes(prepared.pairing_blocks, -1, -2)),
            jnp.swapaxes(prepared.normal_blocks[minus], -1, -2),
        ),
        axis=-1,
    )
    quadratic = jnp.concatenate((upper, lower), axis=-2)
    eta = jnp.concatenate((jnp.ones((sites,)), -jnp.ones((sites,))))
    dynamical = eta[None, :, None] * quadratic
    raw = tuple(_solve_krein(matrix) for matrix in dynamical)
    values = np.stack(tuple(np.asarray(value) for value, _ in raw))
    vectors = np.stack(tuple(np.asarray(vector) for _, vector in raw))
    positive_vectors = []
    positive_values = []
    positive_norms = []
    pairing_residuals = []
    goldstone_residuals = []
    for q in range(plan.q_count):
        eigenvalues = values[q]
        eigenvectors = vectors[q]
        norms = np.real(
            np.sum(
                np.conj(eigenvectors) * np.asarray(eta)[:, None] * eigenvectors, axis=0
            )
        )
        scale = max(float(np.max(np.abs(eigenvalues), initial=0.0)), 1.0)
        if (
            np.max(np.abs(eigenvalues.imag), initial=0.0)
            > plan.stability_tolerance * scale
        ):
            raise ValueError("LSWT dynamical matrix has complex-frequency instability.")
        candidates = np.flatnonzero(
            (norms > plan.krein_tolerance)
            & (eigenvalues.real >= -plan.stability_tolerance * scale)
        )
        if candidates.size != sites:
            raise ValueError(
                "LSWT does not have exactly one positive-Krein branch per site."
            )
        selected = candidates[np.argsort(eigenvalues.real[candidates])]
        frequency = eigenvalues.real[selected]
        selected_norm = norms[selected]
        mode = eigenvectors[:, selected] / np.sqrt(selected_norm)[None, :]
        zero_count = int(
            np.count_nonzero(np.abs(frequency) <= plan.stability_tolerance * scale)
        )
        if zero_count != int(np.asarray(plan.declared_goldstone_counts)[q]):
            raise ValueError("LSWT zero modes do not match the declared Goldstone count.")
        positive_values.append(frequency)
        positive_vectors.append(mode)
        positive_norms.append(
            np.real(
                np.sum(
                    np.conj(mode) * np.asarray(eta)[:, None] * mode,
                    axis=0,
                )
            )
        )
        ordered = np.sort(eigenvalues.real)
        pairing_residuals.append(
            float(np.max(np.abs(ordered + ordered[::-1]), initial=0.0))
        )
        goldstone_residuals.append(
            float(np.max(np.abs(frequency[:zero_count]), initial=0.0))
            if zero_count
            else 0.0
        )
    frequencies = jnp.asarray(np.stack(positive_values))
    modes = jnp.asarray(np.stack(positive_vectors))
    krein_norms = jnp.asarray(np.stack(positive_norms))
    tau_x = jnp.concatenate(
        (
            jnp.concatenate((jnp.zeros((sites, sites)), jnp.eye(sites)), axis=1),
            jnp.concatenate((jnp.eye(sites), jnp.zeros((sites, sites))), axis=1),
        ),
        axis=0,
    )
    negative_modes = ein.contract("ab,kbc->kac", tau_x, jnp.conj(modes[minus]))
    full_modes = jnp.concatenate((modes, negative_modes), axis=-1)
    target_metric = jnp.diag(eta).astype(full_modes.dtype)
    metric = ein.contract("kic,i,kij->kcj", jnp.conj(full_modes), eta, full_modes)
    paraunitary = jnp.max(jnp.abs(metric - target_metric), initial=0.0)
    hermiticity = jnp.max(
        jnp.abs(quadratic - jnp.conj(jnp.swapaxes(quadratic, -1, -2))), initial=0.0
    )
    torque = jnp.max(
        jnp.sqrt(jnp.sum(prepared.reference_torques**2, axis=-1)), initial=0.0
    )
    stability_solve = eigensolve(
        Eigenproblem(
            DenseLinearOperator(
                quadratic,
                properties=OperatorProperties(
                    self_adjoint=True,
                    evidence={"self_adjoint": "construction"},
                ),
            )
        ),
        policy=EigenSolvePolicy(
            DenseEigh(),
            count=2 * sites,
            which="smallest-algebraic",
        ),
    )
    minimum_stability = jnp.min(stability_solve.eigenvalues)
    scale = jnp.maximum(jnp.max(jnp.abs(quadratic)), 1.0)
    successful = (
        jnp.all(stability_solve.successful)
        & (torque <= plan.torque_tolerance)
        & (hermiticity <= plan.stability_tolerance * scale)
        & (paraunitary <= 10.0 * plan.krein_tolerance)
        & (minimum_stability >= -plan.stability_tolerance * scale)
    )
    if not bool(successful):
        raise ValueError("LSWT branch failed stationarity, stability, or Krein evidence.")
    return LinearSpinWaveResult(
        frequencies,
        modes,
        negative_modes,
        krein_norms,
        dynamical,
        quadratic,
        torque,
        hermiticity,
        paraunitary,
        jnp.max(jnp.asarray(pairing_residuals), initial=0.0),
        jnp.max(jnp.asarray(goldstone_residuals), initial=0.0),
        minimum_stability,
        successful,
        prepared.prepared_id,
        plan.mesh_id,
    )


def spin_wave_structure_factor(
    result: LinearSpinWaveResult,
    transverse_vertices: ArrayLike,
    /,
) -> Array:
    """Return nonnegative one-magnon weights for caller-normalized vertices."""

    if not isinstance(result, LinearSpinWaveResult):
        raise TypeError("result must be LinearSpinWaveResult.")
    vertices = jnp.asarray(transverse_vertices)
    if vertices.shape != result.positive_modes.shape[:1] + (
        result.positive_modes.shape[1],
    ):
        raise ValueError("transverse_vertices must have shape (q, 2*site).")
    amplitudes = ein.contract("ki,kib->kb", jnp.conj(vertices), result.positive_modes)
    return jnp.real(amplitudes * jnp.conj(amplitudes))


__all__ = [
    "CollinearSpinWaveLowering",
    "LinearSpinWavePlan",
    "LinearSpinWaveResult",
    "PreparedLinearSpinWave",
    "SpinWaveReferenceState",
    "evaluate_linear_spin_wave",
    "lower_collinear_spin_wave_bonds",
    "prepare_linear_spin_wave",
    "spin_wave_structure_factor",
    "spin_exchange_tensor",
]
