#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite canonical/fugacity references from imaginary chemical potential."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ... import ein
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CanonicalFugacityPlan(StrictModule, NonTrainableState):
    """Static Fourier support and resource limits for a finite charge sector."""

    minimum_charge: int = eqx.field(static=True)
    maximum_charge: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    maximum_nodes: int = eqx.field(static=True)
    maximum_fourier_entries: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        minimum_charge: int,
        maximum_charge: int,
        /,
        *,
        node_count: int,
        temperature: float,
        maximum_nodes: int = 4096,
        maximum_fourier_entries: int = 4_194_304,
    ):
        minimum = int(minimum_charge)
        maximum = int(maximum_charge)
        nodes = int(node_count)
        temperature_ = float(temperature)
        node_limit = int(maximum_nodes)
        entry_limit = int(maximum_fourier_entries)
        sectors = maximum - minimum + 1
        if maximum < minimum:
            raise ValueError("maximum_charge must not be smaller than minimum_charge.")
        if nodes < sectors:
            raise ValueError("node_count must resolve every requested canonical sector.")
        if not np.isfinite(temperature_) or temperature_ <= 0.0:
            raise ValueError("temperature must be finite and positive.")
        if node_limit <= 0 or nodes > node_limit:
            raise ValueError("Imaginary-chemical-potential grid exceeds maximum_nodes.")
        if entry_limit <= 0 or nodes * sectors > entry_limit:
            raise ValueError("Canonical Fourier matrix exceeds maximum_fourier_entries.")
        self.minimum_charge = minimum
        self.maximum_charge = maximum
        self.node_count = nodes
        self.temperature = temperature_
        self.maximum_nodes = node_limit
        self.maximum_fourier_entries = entry_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "canonical-fugacity-plan",
                "minimum_charge": minimum,
                "maximum_charge": maximum,
                "node_count": nodes,
                "temperature": temperature_,
                "maximum_nodes": node_limit,
                "maximum_fourier_entries": entry_limit,
            }
        )


class PreparedCanonicalFugacity(StrictModule, NonTrainableState):
    """Prepared imaginary-potential grid and exact finite Fourier operators."""

    charges: Array
    angles: Array
    imaginary_chemical_potentials: Array
    inverse_fourier: Array
    forward_fourier: Array
    plan: CanonicalFugacityPlan
    prepared_id: str = eqx.field(static=True)


class ImaginaryChemicalPotentialReference(StrictModule):
    """Paired positive/negative imaginary-potential values and symmetry evidence."""

    positive_values: Array
    negative_values: Array
    periodic_origin: Array
    periodic_endpoint: Array
    conjugation_residual: Array
    periodicity_residual: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)


class CanonicalFugacityResult(StrictModule):
    """Canonical sectors with exact node reconstruction and residual evidence."""

    canonical_sectors: Array
    reconstructed_grand_partition: Array
    input_grand_partition: Array
    maximum_reconstruction_residual: Array
    conjugation_residual: Array
    periodicity_residual: Array
    valid: Array
    prepared_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


class FugacityEvaluation(StrictModule):
    chemical_potential: Array
    fugacity: Array
    grand_partition: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


def prepare_canonical_fugacity(
    plan: CanonicalFugacityPlan, /
) -> PreparedCanonicalFugacity:
    """Materialize a bounded discrete Fourier pair for the requested charges."""
    if not isinstance(plan, CanonicalFugacityPlan):
        raise TypeError("plan must be CanonicalFugacityPlan.")
    charges = jnp.arange(plan.minimum_charge, plan.maximum_charge + 1)
    angles = 2.0 * jnp.pi * jnp.arange(plan.node_count) / plan.node_count
    imaginary_mu = 1j * plan.temperature * angles
    inverse = jnp.exp(-1j * charges[:, None] * angles[None, :]) / plan.node_count
    forward = jnp.exp(1j * angles[:, None] * charges[None, :])
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-canonical-fugacity",
            "plan": plan.plan_id,
            "charges": array_tree_fingerprint(np.asarray(charges)),
            "angles": array_tree_fingerprint(np.asarray(angles)),
        }
    )
    return PreparedCanonicalFugacity(
        charges=charges,
        angles=angles,
        imaginary_chemical_potentials=imaginary_mu,
        inverse_fourier=inverse,
        forward_fourier=forward,
        plan=plan,
        prepared_id=prepared_id,
    )


def evaluate_imaginary_chemical_potential_reference(
    prepared: PreparedCanonicalFugacity,
    grand_partition: Callable[[Array], Array],
    /,
    *,
    grand_partition_id: str,
) -> ImaginaryChemicalPotentialReference:
    """Evaluate periodicity and charge-conjugation controls on the prepared grid."""
    if not isinstance(prepared, PreparedCanonicalFugacity):
        raise TypeError("prepared must be PreparedCanonicalFugacity.")
    if not callable(grand_partition):
        raise TypeError("grand_partition must be callable.")
    identifier = str(grand_partition_id)
    if not identifier:
        raise ValueError("grand_partition_id must be non-empty.")

    def evaluate(mu):
        value = jnp.asarray(grand_partition(mu))
        if value.shape != ():
            raise ValueError("grand_partition must return one scalar.")
        return value

    positive = jax.vmap(evaluate)(prepared.imaginary_chemical_potentials)
    negative = jax.vmap(evaluate)(-prepared.imaginary_chemical_potentials)
    origin = evaluate(jnp.asarray(0.0j))
    endpoint = evaluate(jnp.asarray(2.0j * jnp.pi * prepared.plan.temperature))
    conjugation = jnp.max(jnp.abs(negative - jnp.conj(positive)))
    periodicity = jnp.abs(endpoint - origin)
    finite = (
        jnp.all(jnp.isfinite(jnp.real(positive)))
        & jnp.all(jnp.isfinite(jnp.imag(positive)))
        & jnp.all(jnp.isfinite(jnp.real(negative)))
        & jnp.all(jnp.isfinite(jnp.imag(negative)))
        & jnp.isfinite(periodicity)
        & jnp.isfinite(conjugation)
    )
    reference_id = canonical_fingerprint(
        {
            "kind": "imaginary-chemical-potential-reference",
            "prepared": prepared.prepared_id,
            "grand_partition": identifier,
        }
    )
    return ImaginaryChemicalPotentialReference(
        positive_values=positive,
        negative_values=negative,
        periodic_origin=origin,
        periodic_endpoint=endpoint,
        conjugation_residual=conjugation,
        periodicity_residual=periodicity,
        finite=finite,
        prepared_id=prepared.prepared_id,
        reference_id=reference_id,
    )


def canonical_fugacity_transform(
    prepared: PreparedCanonicalFugacity,
    reference: ImaginaryChemicalPotentialReference,
    /,
    *,
    residual_tolerance: float = 1e-8,
) -> CanonicalFugacityResult:
    """Recover canonical sectors and refuse overclaimed unresolved transforms."""
    if not isinstance(prepared, PreparedCanonicalFugacity):
        raise TypeError("prepared must be PreparedCanonicalFugacity.")
    if not isinstance(reference, ImaginaryChemicalPotentialReference):
        raise TypeError("reference must be ImaginaryChemicalPotentialReference.")
    if reference.prepared_id != prepared.prepared_id:
        raise ValueError("Imaginary-potential reference belongs to another preparation.")
    tolerance = float(residual_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("residual_tolerance must be finite and non-negative.")
    canonical = ein.contract(
        "qn,n->q", prepared.inverse_fourier, reference.positive_values
    )
    reconstructed = ein.contract("nq,q->n", prepared.forward_fourier, canonical)
    residual = jnp.max(jnp.abs(reconstructed - reference.positive_values))
    valid = (
        reference.finite
        & jnp.all(jnp.isfinite(jnp.real(canonical)))
        & jnp.all(jnp.isfinite(jnp.imag(canonical)))
        & jnp.isfinite(residual)
        & (residual <= tolerance)
    )
    nan = jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=canonical.dtype)
    return CanonicalFugacityResult(
        canonical_sectors=jnp.where(valid, canonical, nan),
        reconstructed_grand_partition=reconstructed,
        input_grand_partition=reference.positive_values,
        maximum_reconstruction_residual=residual,
        conjugation_residual=reference.conjugation_residual,
        periodicity_residual=reference.periodicity_residual,
        valid=valid,
        prepared_id=prepared.prepared_id,
        claim="finite-charge-discrete-fourier-reference",
    )


def evaluate_fugacity_expansion(
    prepared: PreparedCanonicalFugacity,
    canonical_sectors: Array,
    chemical_potential: Array,
    /,
) -> FugacityEvaluation:
    """Evaluate the finite canonical expansion at a real or complex chemical potential."""
    if not isinstance(prepared, PreparedCanonicalFugacity):
        raise TypeError("prepared must be PreparedCanonicalFugacity.")
    sectors = jnp.asarray(canonical_sectors)
    expected = (prepared.charges.shape[0],)
    if sectors.shape != expected:
        raise ValueError(f"canonical_sectors must have shape {expected}.")
    mu = jnp.asarray(chemical_potential)
    if mu.shape != ():
        raise ValueError("chemical_potential must be scalar.")
    fugacity = jnp.exp(mu / prepared.plan.temperature)
    powers = fugacity**prepared.charges
    value = ein.contract("q,q->", sectors, powers)
    finite = (
        jnp.all(jnp.isfinite(jnp.real(sectors)))
        & jnp.all(jnp.isfinite(jnp.imag(sectors)))
        & jnp.isfinite(jnp.real(fugacity))
        & jnp.isfinite(jnp.imag(fugacity))
        & jnp.isfinite(jnp.real(value))
        & jnp.isfinite(jnp.imag(value))
    )
    return FugacityEvaluation(
        chemical_potential=mu,
        fugacity=fugacity,
        grand_partition=jnp.where(finite, value, jnp.nan + 1j * jnp.nan),
        finite=finite,
        prepared_id=prepared.prepared_id,
    )


__all__ = [
    "CanonicalFugacityPlan",
    "CanonicalFugacityResult",
    "FugacityEvaluation",
    "ImaginaryChemicalPotentialReference",
    "PreparedCanonicalFugacity",
    "canonical_fugacity_transform",
    "evaluate_fugacity_expansion",
    "evaluate_imaginary_chemical_potential_reference",
    "prepare_canonical_fugacity",
]
