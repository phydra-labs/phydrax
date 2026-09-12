#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...operators.quantum import HilbertRegisterLayout
from ...solver import FixedGridLocalHamiltonian, LocalHamiltonian, LocalHamiltonianTerm
from ...tensor_network import (
    add_mpo,
    build_local_term_mpo,
    build_prefix_quadratic_mpo,
    FiniteLocalTerm,
    FiniteMPOBuildEvidence,
    FixedStructureMPOCoefficients,
    MatrixProductOperator,
    PrefixQuadraticMPOEvidence,
    product_mpo,
)


_IDENTITY = jnp.eye(2, dtype=jnp.complex128)
_PAULI_X = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
_PAULI_Y = jnp.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=jnp.complex128)
_PAULI_Z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)


class SchwingerChainModel(StrictModule, NonTrainableState):
    """Open staggered spin-chain Schwinger model with exact Gauss elimination."""

    layout: HilbertRegisterLayout
    external_flux: Array
    lattice_spacing: Array
    mass: Array
    gauge_coupling: Array
    left_boundary_flux: Array
    valid: Array
    site_count: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    boundary: str = eqx.field(static=True)
    encoding: str = eqx.field(static=True)

    def __init__(
        self,
        site_count: int,
        /,
        *,
        lattice_spacing: float,
        mass: float,
        gauge_coupling: float,
        left_boundary_flux: float = 0.0,
        external_flux: ArrayLike | None = None,
    ):
        sites = int(site_count)
        spacing = float(lattice_spacing)
        mass_ = float(mass)
        coupling = float(gauge_coupling)
        boundary_flux = float(left_boundary_flux)
        if sites < 2:
            raise ValueError("Schwinger chains require at least two sites.")
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("lattice_spacing must be finite and positive.")
        if not all(np.isfinite(value) for value in (mass_, coupling, boundary_flux)):
            raise ValueError(
                "Schwinger mass, coupling, and boundary flux must be finite."
            )
        if coupling < 0.0:
            raise ValueError("gauge_coupling must be non-negative.")
        background = (
            np.zeros((sites - 1,), dtype=float)
            if external_flux is None
            else np.asarray(external_flux, dtype=float)
        )
        if background.shape != (sites - 1,) or np.any(~np.isfinite(background)):
            raise ValueError("external_flux must be finite with shape (site_count - 1,).")
        layout = HilbertRegisterLayout(
            tuple(f"site:{site}" for site in range(sites)),
            (2,) * sites,
        )
        model_id = canonical_fingerprint(
            {
                "kind": "open-schwinger-chain",
                "sites": sites,
                "lattice_spacing": spacing,
                "mass": mass_,
                "gauge_coupling": coupling,
                "left_boundary_flux": boundary_flux,
                "external_flux": array_tree_fingerprint(background),
                "charge": "half-z-plus-staggered-identity",
                "hopping": "xx-plus-yy-over-four-a",
            }
        )
        self.layout = layout
        self.external_flux = jnp.asarray(background)
        self.lattice_spacing = jnp.asarray(spacing)
        self.mass = jnp.asarray(mass_)
        self.gauge_coupling = jnp.asarray(coupling)
        self.left_boundary_flux = jnp.asarray(boundary_flux)
        self.valid = jnp.asarray(True)
        self.site_count = sites
        self.model_id = model_id
        self.boundary = "open"
        self.encoding = "staggered-qubit-z-charge"

    @property
    def electric_scale(self) -> Array:
        return 0.5 * self.lattice_spacing * self.gauge_coupling**2

    @property
    def hopping_scale(self) -> Array:
        return 0.25 / self.lattice_spacing

    def charge_operators(self, /) -> tuple[Array, ...]:
        return tuple(
            0.5 * (_PAULI_Z + ((-1.0) ** site) * _IDENTITY)
            for site in range(self.site_count)
        )


class SchwingerObservableSet(StrictModule):
    """Basis-configuration charge, flux, and condensate observables."""

    charges: Array
    electric_flux: Array
    condensate: Array
    total_charge: Array
    model_id: str = eqx.field(static=True)


class SchwingerMPOResult(StrictModule):
    """Full Hamiltonian MPO with separate local and prefix-square evidence."""

    operator: MatrixProductOperator
    local_evidence: FiniteMPOBuildEvidence
    electric_evidence: PrefixQuadraticMPOEvidence
    model_id: str = eqx.field(static=True)


class SchwingerBackgroundSchedule(StrictModule):
    """Fixed-grid MPO coefficients for an externally prescribed link flux."""

    time_grid: Array
    external_flux: Array
    coefficients: FixedStructureMPOCoefficients
    valid: Array
    model_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)


def schwinger_charge_values(
    model: SchwingerChainModel,
    occupations: ArrayLike,
    /,
) -> Array:
    """Return staggered charges for computational-basis occupations."""
    if not isinstance(model, SchwingerChainModel):
        raise TypeError("model must be SchwingerChainModel.")
    values = jnp.asarray(occupations, dtype=jnp.int32)
    if values.shape[-1:] != (model.site_count,):
        raise ValueError("occupations must have trailing Schwinger site axis.")
    staggered = (-1.0) ** jnp.arange(model.site_count)
    z_eigenvalues = 1.0 - 2.0 * values
    return 0.5 * (z_eigenvalues + staggered)


def reconstruct_schwinger_flux(
    model: SchwingerChainModel,
    occupations: ArrayLike,
    /,
) -> Array:
    """Reconstruct links from the left flux and exact open-chain Gauss law."""
    charges = schwinger_charge_values(model, occupations)
    return (
        model.left_boundary_flux
        + model.external_flux
        + jnp.cumsum(charges[..., :-1], axis=-1)
    )


def schwinger_gauss_residual(
    model: SchwingerChainModel,
    occupations: ArrayLike,
    electric_flux: ArrayLike,
    /,
) -> Array:
    """Return the maximum residual against exact eliminated-link reconstruction."""
    expected = reconstruct_schwinger_flux(model, occupations)
    observed = jnp.asarray(electric_flux)
    if observed.shape != expected.shape:
        raise ValueError("electric_flux shape must match reconstructed Schwinger links.")
    return jnp.max(jnp.abs(observed - expected))


def schwinger_observables(
    model: SchwingerChainModel,
    occupations: ArrayLike,
    /,
) -> SchwingerObservableSet:
    charges = schwinger_charge_values(model, occupations)
    flux = reconstruct_schwinger_flux(model, occupations)
    values = jnp.asarray(occupations)
    staggered = (-1.0) ** jnp.arange(model.site_count)
    condensate = jnp.mean(staggered * (values - 0.5), axis=-1)
    return SchwingerObservableSet(
        charges=charges,
        electric_flux=flux,
        condensate=condensate,
        total_charge=jnp.sum(charges, axis=-1),
        model_id=model.model_id,
    )


def _electric_coefficients(model: SchwingerChainModel, /):
    links = model.site_count - 1
    offsets = model.left_boundary_flux + model.external_flux
    suffix_count = jnp.arange(links, 0, -1, dtype=offsets.dtype)
    quadratic = model.electric_scale * jnp.concatenate(
        (suffix_count, jnp.zeros((1,), dtype=offsets.dtype))
    )
    linear = (
        2.0
        * model.electric_scale
        * jnp.concatenate(
            (
                jnp.flip(jnp.cumsum(jnp.flip(offsets))),
                jnp.zeros((1,), dtype=offsets.dtype),
            )
        )
    )
    constant = model.electric_scale * jnp.sum(offsets**2)
    return quadratic, linear, constant


def schwinger_local_hamiltonian(
    model: SchwingerChainModel,
    /,
    *,
    maximum_terms: int = 4096,
) -> LocalHamiltonian:
    """Build an exact small-chain local-term reference with quadratic term count."""
    if not isinstance(model, SchwingerChainModel):
        raise TypeError("model must be SchwingerChainModel.")
    charges = model.charge_operators()
    expected_terms = (
        2 * (model.site_count - 1)
        + model.site_count
        + model.site_count
        + model.site_count * (model.site_count - 1) // 2
        + 1
    )
    if int(maximum_terms) <= 0 or expected_terms > int(maximum_terms):
        raise ValueError("Schwinger local expansion exceeds maximum_terms.")
    terms: list[LocalHamiltonianTerm] = []
    for bond in range(model.site_count - 1):
        wires = (model.layout.wire_ids[bond], model.layout.wire_ids[bond + 1])
        terms.append(
            LocalHamiltonianTerm.from_product(
                (model.hopping_scale * _PAULI_X, _PAULI_X),
                wires,
                term_id=f"{model.model_id}:hopping-x:{bond}",
            )
        )
        terms.append(
            LocalHamiltonianTerm.from_product(
                (model.hopping_scale * _PAULI_Y, _PAULI_Y),
                wires,
                term_id=f"{model.model_id}:hopping-y:{bond}",
            )
        )
    for site, wire in enumerate(model.layout.wire_ids):
        terms.append(
            LocalHamiltonianTerm.from_product(
                (0.5 * model.mass * ((-1.0) ** site) * _PAULI_Z,),
                (wire,),
                term_id=f"{model.model_id}:mass:{site}",
            )
        )
    quadratic, linear, constant = _electric_coefficients(model)
    for site, (charge, wire) in enumerate(
        zip(charges, model.layout.wire_ids, strict=True)
    ):
        local = quadratic[site] * (charge @ charge) + linear[site] * charge
        terms.append(
            LocalHamiltonianTerm.from_product(
                (local,),
                (wire,),
                term_id=f"{model.model_id}:electric-local:{site}",
            )
        )
    for left in range(model.site_count):
        for right in range(left + 1, model.site_count):
            coefficient = 2.0 * quadratic[right]
            terms.append(
                LocalHamiltonianTerm.from_product(
                    (coefficient * charges[left], charges[right]),
                    (model.layout.wire_ids[left], model.layout.wire_ids[right]),
                    term_id=f"{model.model_id}:electric-pair:{left}:{right}",
                )
            )
    terms.append(
        LocalHamiltonianTerm.from_product(
            (constant * _IDENTITY,),
            (model.layout.wire_ids[0],),
            term_id=f"{model.model_id}:electric-constant",
        )
    )
    return LocalHamiltonian(
        model.layout,
        terms,
        hamiltonian_id=f"{model.model_id}:local-hamiltonian",
    )


def schwinger_mpo(model: SchwingerChainModel, /) -> SchwingerMPOResult:
    """Build the full open-chain Hamiltonian with a compact electric MPO."""
    if not isinstance(model, SchwingerChainModel):
        raise TypeError("model must be SchwingerChainModel.")
    terms: list[FiniteLocalTerm] = []
    for bond in range(model.site_count - 1):
        terms.append(
            FiniteLocalTerm(
                bond,
                (_PAULI_X, _PAULI_X),
                coefficient=model.hopping_scale,
            )
        )
        terms.append(
            FiniteLocalTerm(
                bond,
                (_PAULI_Y, _PAULI_Y),
                coefficient=model.hopping_scale,
            )
        )
    for site in range(model.site_count):
        terms.append(
            FiniteLocalTerm(
                site,
                (_PAULI_Z,),
                coefficient=0.5 * model.mass * ((-1.0) ** site),
            )
        )
    local = build_local_term_mpo((2,) * model.site_count, terms)
    offsets = jnp.concatenate(
        (
            model.left_boundary_flux + model.external_flux,
            jnp.zeros((1,), dtype=model.external_flux.dtype),
        )
    )
    weights = jnp.concatenate(
        (
            jnp.full(
                (model.site_count - 1,),
                model.electric_scale,
                dtype=model.external_flux.dtype,
            ),
            jnp.zeros((1,), dtype=model.external_flux.dtype),
        )
    )
    electric = build_prefix_quadratic_mpo(
        model.charge_operators(),
        offsets,
        prefix_weights=weights,
    )
    return SchwingerMPOResult(
        operator=add_mpo(local.operator, electric.operator),
        local_evidence=local.evidence,
        electric_evidence=electric.evidence,
        model_id=model.model_id,
    )


def schwinger_background_schedule(
    model: SchwingerChainModel,
    time_grid: ArrayLike,
    external_flux: ArrayLike,
    /,
) -> SchwingerBackgroundSchedule:
    """Lower time-grid background fluxes to existing fixed-structure MPOs."""
    if not isinstance(model, SchwingerChainModel):
        raise TypeError("model must be SchwingerChainModel.")
    times = jnp.asarray(time_grid)
    background = jnp.asarray(external_flux)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("time_grid must contain at least two knots.")
    if background.shape != (times.shape[0], model.site_count - 1):
        raise ValueError("external_flux must have shape (time_knots, site_count - 1).")
    if jnp.iscomplexobj(times) or jnp.iscomplexobj(background):
        raise TypeError("Schwinger schedule times and fluxes must be real-valued.")
    finite = jnp.all(jnp.isfinite(times)) & jnp.all(jnp.isfinite(background))
    increasing = jnp.all(jnp.diff(times) > 0.0)
    base = schwinger_mpo(model).operator
    charges = model.charge_operators()
    charge_mpos = tuple(
        build_local_term_mpo(
            (2,) * model.site_count,
            (FiniteLocalTerm(site, (charge,)),),
        ).operator
        for site, charge in enumerate(charges)
    )
    identity = product_mpo(
        jnp.stack((_IDENTITY,) * model.site_count),
        precision=base.precision,
    )
    delta = background - model.external_flux[None, :]
    suffix_delta = jnp.flip(jnp.cumsum(jnp.flip(delta, axis=1), axis=1), axis=1)
    charge_coefficients = (
        2.0
        * model.electric_scale
        * jnp.concatenate(
            (
                suffix_delta,
                jnp.zeros((times.shape[0], 1), dtype=delta.dtype),
            ),
            axis=1,
        )
    )
    base_offsets = model.left_boundary_flux + model.external_flux
    constant_coefficients = model.electric_scale * jnp.sum(
        (base_offsets[None, :] + delta) ** 2 - base_offsets[None, :] ** 2,
        axis=1,
    )
    coefficients = jnp.concatenate(
        (
            jnp.ones((times.shape[0], 1), dtype=delta.dtype),
            charge_coefficients,
            constant_coefficients[:, None],
        ),
        axis=1,
    )
    fixed = FixedStructureMPOCoefficients(
        (base,) + charge_mpos + (identity,),
        coefficients,
    )
    schedule_id = canonical_fingerprint(
        {
            "kind": "schwinger-background-mpo-schedule",
            "model": model.model_id,
            "time_shape": list(times.shape),
            "external_flux": array_tree_fingerprint(background),
            "structure": fixed.structure_id,
        }
    )
    return SchwingerBackgroundSchedule(
        time_grid=times,
        external_flux=background,
        coefficients=fixed,
        valid=finite & increasing,
        model_id=model.model_id,
        schedule_id=schedule_id,
    )


def schwinger_local_background_schedule(
    model: SchwingerChainModel,
    time_grid: ArrayLike,
    interval_external_flux: ArrayLike,
    /,
    *,
    maximum_terms: int = 4096,
) -> FixedGridLocalHamiltonian:
    """Build a piecewise-constant local-term schedule for small chains."""
    if not isinstance(model, SchwingerChainModel):
        raise TypeError("model must be SchwingerChainModel.")
    times = jnp.asarray(time_grid)
    background = jnp.asarray(interval_external_flux)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("time_grid must contain at least two knots.")
    intervals = int(times.shape[0] - 1)
    if background.shape != (intervals, model.site_count - 1):
        raise ValueError(
            "interval_external_flux must have shape (interval_count, site_count - 1)."
        )
    charges = model.charge_operators()
    term_count = (
        2 * (model.site_count - 1)
        + model.site_count
        + model.site_count
        + model.site_count * (model.site_count - 1) // 2
        + model.site_count
        + 1
    )
    if int(maximum_terms) <= 0 or term_count > int(maximum_terms):
        raise ValueError("Schwinger schedule basis exceeds maximum_terms.")
    terms: list[LocalHamiltonianTerm] = []
    coefficient_columns: list[Array] = []
    for bond in range(model.site_count - 1):
        wires = (model.layout.wire_ids[bond], model.layout.wire_ids[bond + 1])
        for label, matrix in (("x", _PAULI_X), ("y", _PAULI_Y)):
            terms.append(
                LocalHamiltonianTerm.from_product(
                    (matrix, matrix),
                    wires,
                    term_id=f"{model.model_id}:schedule-hopping-{label}:{bond}",
                )
            )
            coefficient_columns.append(jnp.full((intervals,), model.hopping_scale))
    for site, wire in enumerate(model.layout.wire_ids):
        terms.append(
            LocalHamiltonianTerm.from_product(
                (_PAULI_Z,),
                (wire,),
                term_id=f"{model.model_id}:schedule-mass:{site}",
            )
        )
        coefficient_columns.append(
            jnp.full((intervals,), 0.5 * model.mass * ((-1.0) ** site))
        )
    suffix_count = jnp.arange(
        model.site_count - 1,
        -1,
        -1,
        dtype=model.external_flux.dtype,
    )
    quadratic = model.electric_scale * suffix_count
    for site, (charge, wire) in enumerate(
        zip(charges, model.layout.wire_ids, strict=True)
    ):
        terms.append(
            LocalHamiltonianTerm.from_product(
                (charge @ charge,),
                (wire,),
                term_id=f"{model.model_id}:schedule-electric-square:{site}",
            )
        )
        coefficient_columns.append(jnp.full((intervals,), quadratic[site]))
    for left in range(model.site_count):
        for right in range(left + 1, model.site_count):
            terms.append(
                LocalHamiltonianTerm.from_product(
                    (charges[left], charges[right]),
                    (model.layout.wire_ids[left], model.layout.wire_ids[right]),
                    term_id=f"{model.model_id}:schedule-electric-pair:{left}:{right}",
                )
            )
            coefficient_columns.append(jnp.full((intervals,), 2.0 * quadratic[right]))
    offsets = model.left_boundary_flux + background
    suffix_offsets = jnp.flip(
        jnp.cumsum(jnp.flip(offsets, axis=1), axis=1),
        axis=1,
    )
    linear = (
        2.0
        * model.electric_scale
        * jnp.concatenate(
            (
                suffix_offsets,
                jnp.zeros((intervals, 1), dtype=background.dtype),
            ),
            axis=1,
        )
    )
    for site, (charge, wire) in enumerate(
        zip(charges, model.layout.wire_ids, strict=True)
    ):
        terms.append(
            LocalHamiltonianTerm.from_product(
                (charge,),
                (wire,),
                term_id=f"{model.model_id}:schedule-electric-linear:{site}",
            )
        )
        coefficient_columns.append(linear[:, site])
    terms.append(
        LocalHamiltonianTerm.from_product(
            (_IDENTITY,),
            (model.layout.wire_ids[0],),
            term_id=f"{model.model_id}:schedule-electric-constant",
        )
    )
    coefficient_columns.append(model.electric_scale * jnp.sum(offsets**2, axis=1))
    hamiltonian = LocalHamiltonian(
        model.layout,
        terms,
        hamiltonian_id=f"{model.model_id}:schedule-basis",
    )
    coefficients = jnp.stack(coefficient_columns, axis=1)
    return FixedGridLocalHamiltonian(
        hamiltonian,
        times,
        coefficients,
        schedule_id=canonical_fingerprint(
            {
                "kind": "schwinger-local-background-schedule",
                "model": model.model_id,
                "time_shape": list(times.shape),
                "external_flux": array_tree_fingerprint(background),
            }
        ),
    )


__all__ = [
    "SchwingerBackgroundSchedule",
    "SchwingerChainModel",
    "SchwingerMPOResult",
    "SchwingerObservableSet",
    "reconstruct_schwinger_flux",
    "schwinger_background_schedule",
    "schwinger_charge_values",
    "schwinger_gauss_residual",
    "schwinger_local_hamiltonian",
    "schwinger_local_background_schedule",
    "schwinger_mpo",
    "schwinger_observables",
]
