#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Ricci jets, discrete Hodge representatives, moduli atlases, and topology."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...algebraic import SparsePolynomialSystem
from ...algebraic._exact_integer import exact_integer_rank
from ...linalg import FactorizationPolicy, inverse as invert_matrix


ProjectiveVarietyKind: TypeAlias = Literal[
    "hypersurface", "complete-intersection", "toric-complete-intersection"
]


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class KahlerMetricJet(StrictModule):
    """Sampled Hermitian metric and first/mixed complex derivatives."""

    metric: Array
    holomorphic_derivative: Array
    antiholomorphic_derivative: Array
    mixed_derivative: Array
    sample_ids: tuple[str, ...] = eqx.field(static=True)
    jet_id: str = eqx.field(static=True)

    def __init__(
        self,
        metric: ArrayLike,
        holomorphic_derivative: ArrayLike,
        antiholomorphic_derivative: ArrayLike,
        mixed_derivative: ArrayLike,
        sample_ids: Sequence[str],
        /,
    ):
        metric_ = np.asarray(metric, dtype=np.complex128)
        holomorphic = np.asarray(holomorphic_derivative, dtype=np.complex128)
        antiholomorphic = np.asarray(antiholomorphic_derivative, dtype=np.complex128)
        mixed = np.asarray(mixed_derivative, dtype=np.complex128)
        identifiers = tuple(_identifier(value, "sample ID") for value in sample_ids)
        if metric_.ndim != 3 or metric_.shape[-1] != metric_.shape[-2]:
            raise ValueError("metric must have shape (samples, dimension, dimension).")
        samples, dimension, _ = metric_.shape
        if holomorphic.shape != (samples, dimension, dimension, dimension):
            raise ValueError("holomorphic_derivative has the wrong shape.")
        if antiholomorphic.shape != holomorphic.shape:
            raise ValueError("antiholomorphic_derivative has the wrong shape.")
        if mixed.shape != (samples, dimension, dimension, dimension, dimension):
            raise ValueError("mixed_derivative has the wrong shape.")
        if len(identifiers) != samples or len(set(identifiers)) != samples:
            raise ValueError("One unique sample ID is required per metric jet.")
        if not all(
            np.all(np.isfinite(value))
            for value in (metric_, holomorphic, antiholomorphic, mixed)
        ):
            raise ValueError("Kähler metric jets must be finite.")
        content = {
            "kind": "kahler-metric-jet",
            "metric": array_tree_fingerprint(metric_),
            "holomorphic_derivative": array_tree_fingerprint(holomorphic),
            "antiholomorphic_derivative": array_tree_fingerprint(antiholomorphic),
            "mixed_derivative": array_tree_fingerprint(mixed),
            "sample_ids": identifiers,
        }
        self.metric = jnp.asarray(metric_)
        self.holomorphic_derivative = jnp.asarray(holomorphic)
        self.antiholomorphic_derivative = jnp.asarray(antiholomorphic)
        self.mixed_derivative = jnp.asarray(mixed)
        self.sample_ids = identifiers
        self.jet_id = canonical_fingerprint(content)


class CalabiYauRicciEvidence(StrictModule):
    ricci_tensors: Array
    scalar_curvatures: Array
    metric_hermiticity_residuals: Array
    minimum_metric_eigenvalues: Array
    ricci_norms: Array
    finite: Array
    positive: Array
    accepted: Array
    jet_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def evaluate_calabi_yau_ricci(
    jet: KahlerMetricJet,
    /,
    *,
    maximum_ricci_norm: float,
    positivity_floor: float = 0.0,
) -> CalabiYauRicciEvidence:
    """Compute Rₐb̄ = -∂ₐ∂b̄ log det g from native metric jets."""

    if not isinstance(jet, KahlerMetricJet):
        raise TypeError("jet must be KahlerMetricJet.")
    maximum = float(maximum_ricci_norm)
    floor = float(positivity_floor)
    if not math.isfinite(maximum) or maximum < 0.0 or not math.isfinite(floor):
        raise ValueError("Ricci and positivity tolerances must be finite.")
    metric = jet.metric
    hermitian = 0.5 * (metric + jnp.conj(jnp.swapaxes(metric, -1, -2)))
    inverse_result = invert_matrix(hermitian, FactorizationPolicy("lu"))
    inverse = inverse_result.value
    dimension = metric.shape[-1]
    ricci_samples = []
    for sample in range(metric.shape[0]):
        entries = []
        for holomorphic in range(dimension):
            row = []
            first = jet.holomorphic_derivative[sample, holomorphic]
            for antiholomorphic in range(dimension):
                second = jet.antiholomorphic_derivative[sample, antiholomorphic]
                mixed = jet.mixed_derivative[
                    sample,
                    holomorphic,
                    antiholomorphic,
                ]
                value = -jnp.trace(
                    inverse[sample] @ mixed
                    - inverse[sample] @ second @ inverse[sample] @ first
                )
                row.append(value)
            entries.append(jnp.stack(tuple(row)))
        ricci_samples.append(jnp.stack(tuple(entries)))
    ricci = jnp.stack(tuple(ricci_samples))
    scalars = jnp.real(ein.contract("sij,sji->s", inverse, ricci))
    hermiticity = jnp.max(
        jnp.abs(metric - jnp.conj(jnp.swapaxes(metric, -1, -2))),
        axis=(-2, -1),
    )
    eigenvalues = jnp.linalg.eigvalsh(hermitian)
    minimum = jnp.min(eigenvalues, axis=-1)
    norms = jnp.linalg.norm(ricci.reshape((ricci.shape[0], -1)), axis=-1)
    finite = (
        jnp.all(inverse_result.successful)
        & jnp.all(jnp.isfinite(ricci))
        & jnp.all(jnp.isfinite(scalars))
        & jnp.all(jnp.isfinite(norms))
    )
    positive = jnp.all(minimum > floor)
    accepted = finite & positive & jnp.all(norms <= maximum)
    evidence_id = canonical_fingerprint(
        {
            "kind": "calabi-yau-ricci-evidence",
            "jet": jet.jet_id,
            "maximum_ricci_norm": maximum,
            "positivity_floor": floor,
        }
    )
    return CalabiYauRicciEvidence(
        ricci_tensors=ricci,
        scalar_curvatures=scalars,
        metric_hermiticity_residuals=hermiticity,
        minimum_metric_eigenvalues=minimum,
        ricci_norms=norms,
        finite=finite,
        positive=positive,
        accepted=accepted,
        jet_id=jet.jet_id,
        evidence_id=evidence_id,
    )


class HarmonicKodairaSpencerPlan(StrictModule):
    """Finite Hodge projection of algebraic deformation representatives."""

    algebraic_representatives: Array
    closure_operator: Array
    coclosure_operator: Array
    inner_product: Array
    modulus_labels: tuple[str, ...] = eqx.field(static=True)
    harmonic_tolerance: float = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebraic_representatives: ArrayLike,
        closure_operator: ArrayLike,
        coclosure_operator: ArrayLike,
        inner_product: ArrayLike,
        modulus_labels: Sequence[str],
        /,
        *,
        harmonic_tolerance: float = 1e-10,
        rank_tolerance: float = 1e-10,
    ):
        algebraic = np.asarray(algebraic_representatives, dtype=np.complex128)
        closure = np.asarray(closure_operator, dtype=np.complex128)
        coclosure = np.asarray(coclosure_operator, dtype=np.complex128)
        metric = np.asarray(inner_product, dtype=np.complex128)
        labels = tuple(_identifier(value, "modulus label") for value in modulus_labels)
        if algebraic.ndim != 2 or not algebraic.size:
            raise ValueError("algebraic_representatives must be a non-empty matrix.")
        dimension, modulus_count = algebraic.shape
        if closure.ndim != 2 or closure.shape[1] != dimension:
            raise ValueError("closure_operator has the wrong domain.")
        if coclosure.ndim != 2 or coclosure.shape[1] != dimension:
            raise ValueError("coclosure_operator has the wrong domain.")
        if metric.shape != (dimension, dimension):
            raise ValueError("inner_product has the wrong shape.")
        if len(labels) != modulus_count or len(set(labels)) != modulus_count:
            raise ValueError("One unique label is required per algebraic representative.")
        harmonic = float(harmonic_tolerance)
        rank = float(rank_tolerance)
        if harmonic <= 0.0 or rank <= 0.0:
            raise ValueError("Hodge tolerances must be positive.")
        hermitian = 0.5 * (metric + metric.conj().T)
        if np.linalg.eigvalsh(hermitian)[0] <= rank:
            raise ValueError("Hodge inner product must be positive definite.")
        content = {
            "kind": "harmonic-kodaira-spencer-plan",
            "algebraic": array_tree_fingerprint(algebraic),
            "closure": array_tree_fingerprint(closure),
            "coclosure": array_tree_fingerprint(coclosure),
            "inner_product": array_tree_fingerprint(hermitian),
            "modulus_labels": labels,
            "harmonic_tolerance": harmonic,
            "rank_tolerance": rank,
        }
        self.algebraic_representatives = jnp.asarray(algebraic)
        self.closure_operator = jnp.asarray(closure)
        self.coclosure_operator = jnp.asarray(coclosure)
        self.inner_product = jnp.asarray(hermitian)
        self.modulus_labels = labels
        self.harmonic_tolerance = harmonic
        self.rank_tolerance = rank
        self.plan_id = canonical_fingerprint(content)


class HarmonicKodairaSpencerEvidence(StrictModule):
    closure_residuals: Array
    coclosure_residuals: Array
    gram_matrix: Array
    orthonormality_residual: Array
    harmonic_dimension: int = eqx.field(static=True)
    representative_rank: int = eqx.field(static=True)
    accepted: Array
    evidence_id: str = eqx.field(static=True)


class PreparedHarmonicKodairaSpencer(StrictModule):
    plan: HarmonicKodairaSpencerPlan
    harmonic_basis: Array
    representatives: Array
    coefficients: Array
    evidence: HarmonicKodairaSpencerEvidence
    prepared_id: str = eqx.field(static=True)


def prepare_harmonic_kodaira_spencer(
    plan: HarmonicKodairaSpencerPlan,
    /,
) -> PreparedHarmonicKodairaSpencer:
    """Project algebraic representatives into the discrete Hodge kernel."""

    if not isinstance(plan, HarmonicKodairaSpencerPlan):
        raise TypeError("plan must be HarmonicKodairaSpencerPlan.")
    closure = np.asarray(plan.closure_operator)
    coclosure = np.asarray(plan.coclosure_operator)
    metric = np.asarray(plan.inner_product)
    laplacian = closure.conj().T @ closure + coclosure.conj().T @ coclosure
    factor = np.linalg.cholesky(metric)
    inverse_factor = np.linalg.inv(factor)
    transformed = inverse_factor @ laplacian @ inverse_factor.conj().T
    transformed = 0.5 * (transformed + transformed.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    mask = eigenvalues <= plan.harmonic_tolerance
    if not np.any(mask):
        raise ValueError("Discrete Hodge operator has no admitted harmonic subspace.")
    harmonic_basis = inverse_factor.conj().T @ eigenvectors[:, mask]
    harmonic_gram = harmonic_basis.conj().T @ metric @ harmonic_basis
    harmonic_basis = (
        harmonic_basis @ np.linalg.inv(np.linalg.cholesky(harmonic_gram)).conj().T
    )
    algebraic = np.asarray(plan.algebraic_representatives)
    coefficients = harmonic_basis.conj().T @ metric @ algebraic
    representatives = harmonic_basis @ coefficients
    gram = representatives.conj().T @ metric @ representatives
    closure_residuals = np.linalg.norm(closure @ representatives, axis=0)
    coclosure_residuals = np.linalg.norm(coclosure @ representatives, axis=0)
    representative_rank = int(
        np.linalg.matrix_rank(coefficients, tol=plan.rank_tolerance)
    )
    orthonormality = float(
        np.linalg.norm(
            harmonic_basis.conj().T @ metric @ harmonic_basis
            - np.eye(harmonic_basis.shape[1])
        )
    )
    accepted = bool(
        representative_rank == algebraic.shape[1]
        and max(closure_residuals, default=0.0) <= plan.harmonic_tolerance
        and max(coclosure_residuals, default=0.0) <= plan.harmonic_tolerance
        and orthonormality <= 10.0 * plan.harmonic_tolerance
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "harmonic-kodaira-spencer-evidence",
            "plan": plan.plan_id,
            "harmonic_dimension": harmonic_basis.shape[1],
            "representative_rank": representative_rank,
            "closure_residuals": array_tree_fingerprint(closure_residuals),
            "coclosure_residuals": array_tree_fingerprint(coclosure_residuals),
            "gram": array_tree_fingerprint(gram),
        }
    )
    evidence = HarmonicKodairaSpencerEvidence(
        closure_residuals=jnp.asarray(closure_residuals),
        coclosure_residuals=jnp.asarray(coclosure_residuals),
        gram_matrix=jnp.asarray(gram),
        orthonormality_residual=jnp.asarray(orthonormality),
        harmonic_dimension=harmonic_basis.shape[1],
        representative_rank=representative_rank,
        accepted=jnp.asarray(accepted),
        evidence_id=evidence_id,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-harmonic-kodaira-spencer",
            "plan": plan.plan_id,
            "harmonic_basis": array_tree_fingerprint(harmonic_basis),
            "representatives": array_tree_fingerprint(representatives),
            "coefficients": array_tree_fingerprint(coefficients),
            "evidence": evidence_id,
        }
    )
    return PreparedHarmonicKodairaSpencer(
        plan=plan,
        harmonic_basis=jnp.asarray(harmonic_basis),
        representatives=jnp.asarray(representatives),
        coefficients=jnp.asarray(coefficients),
        evidence=evidence,
        prepared_id=prepared_id,
    )


class HarmonicModuliObservables(StrictModule):
    weil_petersson_metric: Array
    yukawa_tensor: Array
    metric_hermiticity_residual: Array
    minimum_metric_eigenvalue: Array
    yukawa_symmetry_residual: Array
    accepted: Array
    representative_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def evaluate_harmonic_moduli_observables(
    prepared: PreparedHarmonicKodairaSpencer,
    cup_product_tensor: ArrayLike,
    /,
    *,
    holomorphic_volume: complex,
    tolerance: float = 1e-9,
) -> HarmonicModuliObservables:
    """Compute WP and Yukawa tensors from qualified harmonic representatives."""

    if not isinstance(prepared, PreparedHarmonicKodairaSpencer):
        raise TypeError("prepared must be PreparedHarmonicKodairaSpencer.")
    cup = jnp.asarray(cup_product_tensor, dtype=prepared.representatives.dtype)
    dimension = prepared.representatives.shape[0]
    if cup.shape != (dimension, dimension, dimension):
        raise ValueError("cup_product_tensor must act on the representative space.")
    volume = complex(holomorphic_volume)
    if not np.isfinite(volume) or abs(volume) == 0.0:
        raise ValueError("holomorphic_volume must be finite and nonzero.")
    representatives = prepared.representatives
    metric = (
        jnp.conj(representatives.T)
        @ prepared.plan.inner_product
        @ representatives
        / abs(volume) ** 2
    )
    yukawa = (
        ein.contract(
            "abc,ai,bj,ck->ijk",
            cup,
            representatives,
            representatives,
            representatives,
        )
        / volume
    )
    hermiticity = jnp.max(jnp.abs(metric - jnp.conj(metric.T)))
    minimum = jnp.min(jnp.linalg.eigvalsh(0.5 * (metric + jnp.conj(metric.T))))
    permutations = (
        yukawa,
        jnp.transpose(yukawa, (1, 0, 2)),
        jnp.transpose(yukawa, (2, 1, 0)),
        jnp.transpose(yukawa, (0, 2, 1)),
    )
    symmetry = max(jnp.max(jnp.abs(value - yukawa)) for value in permutations)
    accepted = (
        prepared.evidence.accepted
        & (hermiticity <= tolerance)
        & (minimum > 0.0)
        & (symmetry <= tolerance)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "harmonic-moduli-observables",
            "representatives": prepared.prepared_id,
            "cup_product": array_tree_fingerprint(cup),
            "holomorphic_volume": (volume.real, volume.imag),
            "tolerance": float(tolerance),
        }
    )
    return HarmonicModuliObservables(
        weil_petersson_metric=metric,
        yukawa_tensor=yukawa,
        metric_hermiticity_residual=hermiticity,
        minimum_metric_eigenvalue=minimum,
        yukawa_symmetry_residual=symmetry,
        accepted=accepted,
        representative_id=prepared.prepared_id,
        evidence_id=evidence_id,
    )


@dataclass(frozen=True, slots=True)
class ComplexModuliPatch:
    """One affine coefficient patch with explicit polytope admission."""

    label: str
    global_to_local: np.ndarray
    offset: np.ndarray
    domain_normals: np.ndarray
    domain_offsets: np.ndarray
    patch_id: str

    def __init__(
        self,
        label: str,
        global_to_local: ArrayLike,
        offset: ArrayLike,
        domain_normals: ArrayLike,
        domain_offsets: ArrayLike,
        /,
    ):
        label_ = _identifier(label, "moduli patch label")
        mapping = np.asarray(global_to_local, dtype=np.complex128)
        offset_ = np.asarray(offset, dtype=np.complex128)
        normals = np.asarray(domain_normals, dtype=np.complex128)
        bounds = np.asarray(domain_offsets, dtype=np.float64)
        if mapping.ndim != 2 or mapping.shape[0] != mapping.shape[1]:
            raise ValueError("global_to_local must be square and invertible.")
        dimension = mapping.shape[0]
        if (
            offset_.shape != (dimension,)
            or normals.ndim != 2
            or normals.shape[1] != dimension
        ):
            raise ValueError("Moduli patch offsets or normals have the wrong shape.")
        if bounds.shape != (normals.shape[0],) or abs(np.linalg.det(mapping)) == 0.0:
            raise ValueError("Moduli patch domain or map is singular.")
        content = {
            "kind": "complex-moduli-patch",
            "label": label_,
            "global_to_local": array_tree_fingerprint(mapping),
            "offset": array_tree_fingerprint(offset_),
            "domain_normals": array_tree_fingerprint(normals),
            "domain_offsets": array_tree_fingerprint(bounds),
        }
        object.__setattr__(self, "label", label_)
        object.__setattr__(self, "global_to_local", mapping)
        object.__setattr__(self, "offset", offset_)
        object.__setattr__(self, "domain_normals", normals)
        object.__setattr__(self, "domain_offsets", bounds)
        object.__setattr__(self, "patch_id", canonical_fingerprint(content))

    def local(self, global_coordinates: ArrayLike, /) -> np.ndarray:
        value = np.asarray(global_coordinates, dtype=np.complex128)
        return self.global_to_local @ value + self.offset

    def global_coordinates(self, local_coordinates: ArrayLike, /) -> np.ndarray:
        value = np.asarray(local_coordinates, dtype=np.complex128)
        return np.linalg.solve(self.global_to_local, value - self.offset)

    def margin(self, local_coordinates: ArrayLike, /) -> float:
        value = np.asarray(local_coordinates, dtype=np.complex128)
        residual = self.domain_offsets - np.real(self.domain_normals @ value)
        return float(np.min(residual)) if residual.size else math.inf


@dataclass(frozen=True, slots=True)
class ComplexModuliAtlasEvidence:
    patch_labels: tuple[str, ...]
    transition_residual: float
    path_patch_labels: tuple[str, ...]
    monodromy: np.ndarray
    monodromy_residual: float
    accepted: bool
    evidence_id: str


def assess_complex_moduli_atlas(
    patches: Sequence[ComplexModuliPatch],
    path: ArrayLike,
    /,
    *,
    tolerance: float = 1e-10,
) -> ComplexModuliAtlasEvidence:
    """Follow a global path through a finite atlas and audit loop monodromy."""

    patches_ = tuple(patches)
    points = np.asarray(path, dtype=np.complex128)
    if not patches_ or any(
        not isinstance(value, ComplexModuliPatch) for value in patches_
    ):
        raise TypeError("patches must contain ComplexModuliPatch values.")
    dimension = patches_[0].global_to_local.shape[0]
    if any(value.global_to_local.shape != (dimension, dimension) for value in patches_):
        raise ValueError("All moduli patches must share one global dimension.")
    if points.ndim != 2 or points.shape[1] != dimension or points.shape[0] < 2:
        raise ValueError("Moduli path must contain at least two global points.")
    selected: list[ComplexModuliPatch] = []
    for point in points:
        candidates = tuple(
            (patch.margin(patch.local(point)), patch) for patch in patches_
        )
        margin, owner = max(candidates, key=lambda item: item[0])
        if margin < -tolerance:
            raise ValueError("A moduli path point is outside every declared patch.")
        selected.append(owner)
    transition_residual = 0.0
    monodromy = np.eye(dimension, dtype=np.complex128)
    for source, target, point in zip(
        selected[:-1], selected[1:], points[1:], strict=True
    ):
        source_local = source.local(point)
        global_reconstructed = source.global_coordinates(source_local)
        target_local = target.local(global_reconstructed)
        transition_residual = max(
            transition_residual,
            float(np.linalg.norm(target_local - target.local(point))),
        )
        transition = target.global_to_local @ np.linalg.inv(source.global_to_local)
        monodromy = transition @ monodromy
    closed = np.linalg.norm(points[-1] - points[0]) <= tolerance
    monodromy_residual = (
        float(np.linalg.norm(monodromy - np.eye(dimension))) if closed else math.nan
    )
    accepted = transition_residual <= tolerance and (
        not closed or monodromy_residual <= tolerance
    )
    labels = tuple(value.label for value in selected)
    evidence_id = canonical_fingerprint(
        {
            "kind": "complex-moduli-atlas-evidence",
            "patches": [value.patch_id for value in patches_],
            "path": array_tree_fingerprint(points),
            "selected": labels,
            "tolerance": float(tolerance),
        }
    )
    return ComplexModuliAtlasEvidence(
        tuple(value.label for value in patches_),
        transition_residual,
        labels,
        monodromy,
        monodromy_residual,
        accepted,
        evidence_id,
    )


class PeriodTransportPlan(StrictModule):
    """Piecewise Gauss–Manin transport with pairing-preservation evidence."""

    connection_samples: Array
    step_sizes: Array
    initial_periods: Array
    pairing: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        connection_samples: ArrayLike,
        step_sizes: ArrayLike,
        initial_periods: ArrayLike,
        pairing: ArrayLike,
        /,
    ):
        connection = np.asarray(connection_samples, dtype=np.complex128)
        steps = np.asarray(step_sizes, dtype=np.float64)
        initial = np.asarray(initial_periods, dtype=np.complex128)
        pairing_ = np.asarray(pairing, dtype=np.complex128)
        if connection.ndim != 3 or connection.shape[-1] != connection.shape[-2]:
            raise ValueError(
                "connection_samples must have shape (steps, periods, periods)."
            )
        count, dimension, _ = connection.shape
        if steps.shape != (count,) or np.any(steps <= 0.0):
            raise ValueError("One positive step size is required per connection sample.")
        if initial.shape != (dimension,) or pairing_.shape != (dimension, dimension):
            raise ValueError("Period initial data or pairing has the wrong shape.")
        content = {
            "kind": "period-transport-plan",
            "connection": array_tree_fingerprint(connection),
            "step_sizes": array_tree_fingerprint(steps),
            "initial_periods": array_tree_fingerprint(initial),
            "pairing": array_tree_fingerprint(pairing_),
        }
        self.connection_samples = jnp.asarray(connection)
        self.step_sizes = jnp.asarray(steps)
        self.initial_periods = jnp.asarray(initial)
        self.pairing = jnp.asarray(pairing_)
        self.plan_id = canonical_fingerprint(content)


class PeriodMonodromyEvidence(StrictModule):
    period_history: Array
    monodromy: Array
    pairing_residual: Array
    finite: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def transport_calabi_yau_periods(
    plan: PeriodTransportPlan,
    /,
    *,
    pairing_tolerance: float = 1e-9,
) -> PeriodMonodromyEvidence:
    if not isinstance(plan, PeriodTransportPlan):
        raise TypeError("plan must be PeriodTransportPlan.")
    periods = plan.initial_periods
    monodromy = jnp.eye(periods.shape[0], dtype=periods.dtype)
    history = [periods]
    for connection, step in zip(plan.connection_samples, plan.step_sizes, strict=True):

        def advance(value: Array) -> Array:
            first = connection @ value
            second = connection @ (value + 0.5 * step * first)
            third = connection @ (value + 0.5 * step * second)
            fourth = connection @ (value + step * third)
            return value + step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0

        periods = advance(periods)
        monodromy = jax.vmap(advance, in_axes=1, out_axes=1)(monodromy)
        history.append(periods)
    pairing_residual = jnp.linalg.norm(
        jnp.conj(monodromy.T) @ plan.pairing @ monodromy - plan.pairing
    )
    finite = jnp.all(jnp.isfinite(monodromy)) & jnp.all(jnp.isfinite(periods))
    accepted = finite & (pairing_residual <= pairing_tolerance)
    evidence_id = canonical_fingerprint(
        {
            "kind": "period-monodromy-evidence",
            "plan": plan.plan_id,
            "pairing_tolerance": float(pairing_tolerance),
        }
    )
    return PeriodMonodromyEvidence(
        period_history=jnp.stack(tuple(history)),
        monodromy=monodromy,
        pairing_residual=pairing_residual,
        finite=finite,
        accepted=accepted,
        plan_id=plan.plan_id,
        evidence_id=evidence_id,
    )


class KahlerModuliPlan(StrictModule):
    divisor_labels: tuple[str, ...] = eqx.field(static=True)
    intersection_tensor: Array
    cone_normals: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        divisor_labels: Sequence[str],
        intersection_tensor: ArrayLike,
        cone_normals: ArrayLike,
        /,
    ):
        labels = tuple(_identifier(value, "divisor label") for value in divisor_labels)
        intersections = np.asarray(intersection_tensor, dtype=np.float64)
        cone = np.asarray(cone_normals, dtype=np.float64)
        dimension = len(labels)
        if not labels or len(set(labels)) != dimension:
            raise ValueError("Divisor labels must be unique and non-empty.")
        if intersections.shape != (dimension, dimension, dimension):
            raise ValueError("intersection_tensor has the wrong shape.")
        if cone.ndim != 2 or cone.shape[1] != dimension or not cone.shape[0]:
            raise ValueError("cone_normals must define a non-empty polyhedral cone.")
        symmetry = max(
            np.max(np.abs(intersections - np.transpose(intersections, permutation)))
            for permutation in ((1, 0, 2), (2, 1, 0), (0, 2, 1))
        )
        if symmetry > 1e-12:
            raise ValueError("Calabi–Yau intersection tensors must be symmetric.")
        content = {
            "kind": "kahler-moduli-plan",
            "divisor_labels": labels,
            "intersection_tensor": array_tree_fingerprint(intersections),
            "cone_normals": array_tree_fingerprint(cone),
        }
        self.divisor_labels = labels
        self.intersection_tensor = jnp.asarray(intersections)
        self.cone_normals = jnp.asarray(cone)
        self.plan_id = canonical_fingerprint(content)


class KahlerModuliEvidence(StrictModule):
    total_volume: Array
    divisor_volumes: Array
    metric: Array
    cone_margins: Array
    minimum_metric_eigenvalue: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def evaluate_kahler_moduli(
    plan: KahlerModuliPlan,
    parameters: ArrayLike,
    /,
) -> KahlerModuliEvidence:
    if not isinstance(plan, KahlerModuliPlan):
        raise TypeError("plan must be KahlerModuliPlan.")
    values = jnp.asarray(parameters, dtype=plan.intersection_tensor.dtype)
    if values.shape != (len(plan.divisor_labels),):
        raise ValueError("Kähler parameters have the wrong shape.")

    def volume(coordinates: Array) -> Array:
        return (
            ein.contract(
                "abc,a,b,c->",
                plan.intersection_tensor,
                coordinates,
                coordinates,
                coordinates,
            )
            / 6.0
        )

    total = volume(values)
    divisors = 0.5 * ein.contract(
        "abc,b,c->a",
        plan.intersection_tensor,
        values,
        values,
    )
    metric = jax.hessian(lambda point: -jnp.log(volume(point)))(values)
    metric = 0.5 * (metric + metric.T)
    margins = plan.cone_normals @ values
    minimum = jnp.min(jnp.linalg.eigvalsh(metric))
    accepted = (
        jnp.all(jnp.isfinite(metric))
        & (total > 0.0)
        & jnp.all(margins > 0.0)
        & (minimum > 0.0)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "kahler-moduli-evidence",
            "plan": plan.plan_id,
            "parameters": array_tree_fingerprint(np.asarray(values)),
        }
    )
    return KahlerModuliEvidence(
        total_volume=total,
        divisor_volumes=divisors,
        metric=metric,
        cone_margins=margins,
        minimum_metric_eigenvalue=minimum,
        accepted=accepted,
        plan_id=plan.plan_id,
        evidence_id=evidence_id,
    )


@dataclass(frozen=True, slots=True)
class CharacteristicClassCertificate:
    refinement_values: tuple[complex, ...]
    closedness_residuals: tuple[float, ...]
    exact_integer: int
    convergence_error: float
    imaginary_residual: float
    closed: bool
    integral: bool
    certificate_id: str


def certify_characteristic_class(
    refinement_values: Sequence[complex],
    closedness_residuals: Sequence[float],
    exact_integer: int,
    /,
    *,
    convergence_tolerance: float,
    closedness_tolerance: float,
    integrality_tolerance: float,
) -> CharacteristicClassCertificate:
    values = tuple(complex(value) for value in refinement_values)
    residuals = tuple(float(value) for value in closedness_residuals)
    if len(values) < 2 or len(values) != len(residuals):
        raise ValueError("Characteristic certification requires aligned refinements.")
    if not all(np.isfinite(value) for value in (*values, *residuals)):
        raise ValueError("Characteristic certification values must be finite.")
    convergence = abs(values[-1] - values[-2])
    imaginary = abs(values[-1].imag)
    closed = max(residuals) <= float(closedness_tolerance)
    integral = (
        convergence <= float(convergence_tolerance)
        and imaginary <= float(integrality_tolerance)
        and abs(values[-1].real - int(exact_integer)) <= float(integrality_tolerance)
    )
    content = {
        "kind": "characteristic-class-certificate",
        "refinement_values": [(value.real, value.imag) for value in values],
        "closedness_residuals": residuals,
        "exact_integer": int(exact_integer),
        "convergence_tolerance": float(convergence_tolerance),
        "closedness_tolerance": float(closedness_tolerance),
        "integrality_tolerance": float(integrality_tolerance),
    }
    return CharacteristicClassCertificate(
        values,
        residuals,
        int(exact_integer),
        float(convergence),
        float(imaginary),
        closed,
        integral,
        canonical_fingerprint(content),
    )


@dataclass(frozen=True, slots=True)
class CalabiYauTopologyCertificate:
    betti_numbers: tuple[int, ...]
    euler_characteristic: int
    boundary_residual: int
    intersection_symmetry_residual: float
    exact: bool
    certificate_id: str


def certify_calabi_yau_topology(
    boundary_operators: Sequence[ArrayLike],
    intersection_form: ArrayLike,
    /,
) -> CalabiYauTopologyCertificate:
    boundaries = tuple(np.asarray(value, dtype=np.int64) for value in boundary_operators)
    intersection = np.asarray(intersection_form, dtype=np.int64)
    if not boundaries or any(value.ndim != 2 for value in boundaries):
        raise ValueError("Topology certification requires finite boundary matrices.")
    residual = 0
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        if lower.shape[1] != upper.shape[0]:
            raise ValueError("Adjacent boundary operators have incompatible shapes.")
        residual = max(residual, int(np.max(np.abs(lower @ upper), initial=0)))
    chain_dimensions = [boundaries[0].shape[0]]
    chain_dimensions.extend(value.shape[1] for value in boundaries)
    ranks = [exact_integer_rank(value) for value in boundaries]
    betti = []
    for degree, dimension in enumerate(chain_dimensions):
        outgoing_rank = ranks[degree - 1] if degree > 0 else 0
        incoming_rank = ranks[degree] if degree < len(ranks) else 0
        betti.append(dimension - outgoing_rank - incoming_rank)
    euler = sum((-1) ** degree * value for degree, value in enumerate(betti))
    if intersection.ndim != 2 or intersection.shape[0] != intersection.shape[1]:
        raise ValueError("intersection_form must be square.")
    symmetry = float(np.max(np.abs(intersection - intersection.T), initial=0))
    exact = residual == 0 and symmetry == 0.0 and all(value >= 0 for value in betti)
    content = {
        "kind": "calabi-yau-topology-certificate",
        "boundaries": [array_tree_fingerprint(value) for value in boundaries],
        "intersection_form": array_tree_fingerprint(intersection),
        "betti_numbers": betti,
        "euler_characteristic": euler,
    }
    return CalabiYauTopologyCertificate(
        tuple(betti),
        euler,
        residual,
        symmetry,
        exact,
        canonical_fingerprint(content),
    )


class ProjectiveVarietyPlan(StrictModule):
    """Canonical sparse hypersurface, complete-intersection, or toric CY family."""

    kind: ProjectiveVarietyKind = eqx.field(static=True)
    system: SparsePolynomialSystem
    ambient_weights: tuple[int, ...] = eqx.field(static=True)
    equation_degrees: tuple[int, ...] = eqx.field(static=True)
    toric_charge_matrix: Array | None
    maximum_monomials: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ProjectiveVarietyKind,
        system: SparsePolynomialSystem,
        ambient_weights: Sequence[int],
        equation_degrees: Sequence[int],
        /,
        *,
        toric_charge_matrix: ArrayLike | None = None,
        maximum_monomials: int = 100_000,
    ):
        if kind not in (
            "hypersurface",
            "complete-intersection",
            "toric-complete-intersection",
        ):
            raise ValueError("Unknown projective variety kind.")
        if not isinstance(system, SparsePolynomialSystem):
            raise TypeError("system must be SparsePolynomialSystem.")
        weights = tuple(ambient_weights)
        degrees = tuple(equation_degrees)
        if not weights or any(value <= 0 for value in weights):
            raise ValueError("Ambient weights must be positive.")
        if system.support.variable_count != len(weights):
            raise ValueError("Polynomial variables do not match the ambient weights.")
        if (
            not degrees
            or any(value <= 0 for value in degrees)
            or system.support.equation_count != len(degrees)
        ):
            raise ValueError("One positive degree is required per polynomial equation.")
        exponents = np.asarray(system.support.exponents, dtype=np.int32)
        equations = np.asarray(system.support.equation_indices, dtype=np.int32)
        weighted_degrees = exponents @ np.asarray(weights, dtype=np.int32)
        expected_degrees = np.asarray(degrees, dtype=np.int32)[equations]
        if np.any(weighted_degrees != expected_degrees):
            raise ValueError("A projective monomial violates weighted homogeneity.")
        maximum = int(maximum_monomials)
        if system.support.term_count > maximum:
            raise ValueError("Projective variety exceeds maximum_monomials.")
        if sum(degrees) != sum(weights):
            raise ValueError(
                "The declared complete intersection fails the CY degree condition."
            )
        charge = (
            None
            if toric_charge_matrix is None
            else np.asarray(toric_charge_matrix, dtype=np.int32)
        )
        if kind == "toric-complete-intersection":
            if charge is None or charge.ndim != 2 or charge.shape[1] != len(weights):
                raise ValueError("Toric varieties require a compatible charge matrix.")
            if np.any(exponents @ charge.T != 0):
                raise ValueError(
                    "Toric monomials violate the declared charge invariance."
                )
        elif charge is not None:
            raise ValueError("Only toric varieties may declare a toric charge matrix.")
        content = {
            "kind": "projective-variety-plan",
            "variety_kind": kind,
            "system": system.system_id,
            "ambient_weights": weights,
            "equation_degrees": degrees,
            "toric_charge_matrix": None
            if charge is None
            else array_tree_fingerprint(charge),
            "maximum_monomials": maximum,
        }
        self.kind = kind
        self.system = system
        self.ambient_weights = weights
        self.equation_degrees = degrees
        self.toric_charge_matrix = None if charge is None else jnp.asarray(charge)
        self.maximum_monomials = maximum
        self.plan_id = canonical_fingerprint(content)

    def evaluate(self, homogeneous_points: ArrayLike, /) -> Array:
        points = jnp.asarray(homogeneous_points)
        if points.shape[-1:] != (len(self.ambient_weights),):
            raise ValueError("Homogeneous points have the wrong ambient dimension.")
        return self.system.evaluate(points)

    def jacobian(self, homogeneous_point: ArrayLike, /) -> Array:
        point = jnp.asarray(homogeneous_point)
        if point.shape != (len(self.ambient_weights),):
            raise ValueError("A projective Jacobian requires one homogeneous point.")
        return self.system.jacobian(point)


class ProjectiveVarietyEvidence(StrictModule):
    equation_residuals: Array
    jacobian_singular_values: Array
    minimum_transversality_margin: Array
    exact_cy_degree_condition: Array
    accepted: Array
    plan_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def assess_projective_variety(
    plan: ProjectiveVarietyPlan,
    points: ArrayLike,
    /,
    *,
    equation_tolerance: float = 1e-9,
    transversality_tolerance: float = 1e-9,
) -> ProjectiveVarietyEvidence:
    if not isinstance(plan, ProjectiveVarietyPlan):
        raise TypeError("plan must be ProjectiveVarietyPlan.")
    values = jnp.asarray(points)
    if values.ndim != 2 or values.shape[1] != len(plan.ambient_weights):
        raise ValueError("Projective evidence points have the wrong shape.")
    equations = plan.evaluate(values)
    jacobians = jax.vmap(plan.jacobian)(values)
    singular_values = jnp.linalg.svd(jacobians, compute_uv=False)
    margin = jnp.min(singular_values[..., -1])
    residuals = jnp.max(jnp.abs(equations), axis=-1)
    exact = jnp.asarray(sum(plan.equation_degrees) == sum(plan.ambient_weights))
    accepted = (
        jnp.all(residuals <= equation_tolerance)
        & (margin > transversality_tolerance)
        & exact
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "projective-variety-evidence",
            "plan": plan.plan_id,
            "points": array_tree_fingerprint(np.asarray(values)),
            "equation_tolerance": float(equation_tolerance),
            "transversality_tolerance": float(transversality_tolerance),
        }
    )
    return ProjectiveVarietyEvidence(
        equation_residuals=residuals,
        jacobian_singular_values=singular_values,
        minimum_transversality_margin=margin,
        exact_cy_degree_condition=exact,
        accepted=accepted,
        plan_id=plan.plan_id,
        evidence_id=evidence_id,
    )


__all__ = [
    "CalabiYauRicciEvidence",
    "CalabiYauTopologyCertificate",
    "CharacteristicClassCertificate",
    "ComplexModuliAtlasEvidence",
    "ComplexModuliPatch",
    "HarmonicKodairaSpencerEvidence",
    "HarmonicKodairaSpencerPlan",
    "HarmonicModuliObservables",
    "KahlerMetricJet",
    "KahlerModuliEvidence",
    "KahlerModuliPlan",
    "PeriodMonodromyEvidence",
    "PeriodTransportPlan",
    "PreparedHarmonicKodairaSpencer",
    "ProjectiveVarietyEvidence",
    "ProjectiveVarietyKind",
    "ProjectiveVarietyPlan",
    "assess_complex_moduli_atlas",
    "assess_projective_variety",
    "certify_calabi_yau_topology",
    "certify_characteristic_class",
    "evaluate_calabi_yau_ricci",
    "evaluate_harmonic_moduli_observables",
    "evaluate_kahler_moduli",
    "prepare_harmonic_kodaira_spencer",
    "transport_calabi_yau_periods",
]
