#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic bond-charge site-moment model: a real native producer of typed fields.

``BondChargeMomentModel`` is a small trainable native atomistic model whose
atom-level outputs are the exact moments of an explicitly defined model charge
density. It exists to exercise the typed predicted-moment consumers
(``MultipoleMomentFields`` with ``MultipoleElectrostaticPotential`` and
``InducedDipolePolarizationPotential``) end to end; it makes no chemistry,
accuracy or transferability claim.

Defining law (system Cartesian frame and units, every moment about nucleus
``i``; ``r_ij`` is the vector from atom ``i`` to the neighbor image ``j`` of the
directed graph, ``s_i`` the species class of atom ``i``):

* envelope ``f(d) = (1 - d^2 / r_c^2)^3`` for ``d < r_c`` and zero beyond (C2);
* charge transfer ``q_i = sum_j (chi_{s_j} - chi_{s_i}) f(d_ij)``; antisymmetric
  pair terms make ``sum_i q_i = 0`` exactly, finite or periodic;
* site density ``rho_i(r) = (q_i - B_i) delta(r) + sum_j b_ij delta(r - t r_ij)
  + c_{s_i} [g_{w_{s_i}}(r) - delta(r)]`` with bond charges
  ``b_ij = beta_{s_i} f(d_ij)``, ``B_i = sum_j b_ij``, a fixed bond fraction ``t``
  and a charge-``c`` unit-normalised isotropic Gaussian cloud ``g_w`` of standard
  deviation ``w`` compensated at the nucleus (``c < 0`` is an electron cloud);
* its moments: charge ``q_i``; dipole ``mu_i = t sum_j b_ij r_ij``; primitive
  raw (traceful) second moment ``Theta_i = int rho_i r r^T = t^2 sum_j b_ij
  r_ij r_ij^T + c_{s_i} w_{s_i}^2 I`` (the ``1/2 Theta : grad grad`` convention);
* physical isotropic polarizability ``alpha_i = exp(log_alpha_{s_i} - gamma
  sum_j f(d_ij))`` in charge^2 length^2 / energy;
* scalar energy ``E = sum_i eta_{s_i} q_i^2 / 2``: a local hardness self energy
  whose atom decomposition is complete.

Fields are published only when requested, in the ``"cartesian-primitive"``
basis whose name is the dipole and second-moment port ``normalization_id``
(the basis witness). ``explicit_site_moments`` evaluates the same law on the
host from explicitly enumerated point charges and a Gauss-Hermite quadrature
of each cloud; it shares no code with the JAX evaluation.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Sequence
from typing import Any, TypedDict

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax._differentiation import (
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.atomistic import (
    AbstractAtomisticPotential,
    AtomisticBatch,
    AtomisticConditionLevel,
    AtomisticDenseImageSearch,
    AtomisticFieldReference,
    AtomisticGraphExecutionPlan,
    AtomisticGraphInputs,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
    AtomisticObservableValue,
    AtomisticPotentialCapabilities,
    AtomisticPotentialProgram,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    CartesianMultipoles,
    FiniteMultipoleRoute,
    GaussianMultipoleWidths,
    InducedDipolePolarizationPotential,
    LearnedGraphPotentialTerm,
    MultipoleElectrostaticPotential,
    MultipoleElectrostaticsDefinition,
    MultipoleElectrostaticsEvidence,
    MultipoleElectrostaticsPlan,
    MultipoleMomentFields,
    MultipoleSiteKind,
    PermanentMultipoleSiteData,
    PolarizationPlan,
    PolarizationSiteKind,
    PolarizationStabilityPolicy,
    PolarizationTermEvidence,
)
from phydrax.atomistic._hessian import atomistic_edge_distance_locus
from phydrax.discretization import PeriodicCell
from phydrax.units import derived_unit, UnitDefinition
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)


BASIS = "cartesian-primitive"
"""The declared moment basis; also every moment port's ``normalization_id``."""
PRODUCER = "site-moments"
"""Program binding key of the producer term in every composition here."""
SPECIES = (1, 6, 8)
"""Admitted atomic numbers, in the order of every per-species parameter row."""
SITE_KINDS: tuple[MultipoleSiteKind, ...] = ("point", "gaussian")
"""Point and unit-normalised Gaussian sites (one ``Literal`` for both consumers)."""
_UNIQUE_ROWS = (0, 0, 0, 1, 1, 2)
_UNIQUE_COLUMNS = (0, 1, 2, 1, 2, 2)
_FIELDS = (
    AtomisticObservable.CHARGES,
    AtomisticObservable.DIPOLE,
    AtomisticObservable.QUADRUPOLE,
    AtomisticObservable.POLARIZABILITY,
)
_METHOD_ID = "bond-charge-site-moments-negative-position-gradient"


class BondChargeMomentConfiguration(StrictModule, NonTrainableState):
    """Static law constants: cutoff, admitted species and bond fraction ``t``."""

    cutoff: float = eqx.field(static=True)
    species: tuple[int, ...] = eqx.field(static=True)
    bond_fraction: float = eqx.field(static=True)

    def __init__(
        self, *, cutoff: float, species: Sequence[int], bond_fraction: float
    ) -> None:
        cutoff_ = float(cutoff)
        fraction = float(bond_fraction)
        table = tuple(int(value) for value in species)
        if not (math.isfinite(cutoff_) and cutoff_ > 0.0):
            raise ValueError("cutoff must be finite and positive.")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("bond_fraction must lie in (0, 1].")
        if not table or len(set(table)) != len(table) or min(table) < 1:
            raise ValueError("species must be distinct positive atomic numbers.")
        self.cutoff = cutoff_
        self.species = table
        self.bond_fraction = fraction


class BondChargeMomentModel(AbstractAtomisticPotential):
    """Trainable analytic site-moment model (module docstring states the law).

    Parameter rows follow ``configuration.species``: ``electronegativity``
    (charge), ``bond_charge`` (charge), ``cloud_charge`` (charge),
    ``cloud_width`` (length), ``hardness`` (energy / charge^2),
    ``log_polarizability`` (log of charge^2 length^2 / energy) and the scalar
    dimensionless ``screening``. Every one is a discoverable PARAMETER leaf.
    """

    electronegativity: Array
    bond_charge: Array
    cloud_charge: Array
    cloud_width: Array
    hardness: Array
    log_polarizability: Array
    screening: Array
    configuration: BondChargeMomentConfiguration
    scale: AtomisticScaleContract
    precision: AtomisticPrecisionPolicy
    charge_unit: UnitDefinition
    architecture_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        units: AtomisticUnitSystem,
        /,
        *,
        electronegativity: Any,
        bond_charge: Any,
        cloud_charge: Any,
        cloud_width: Any,
        hardness: Any,
        log_polarizability: Any,
        screening: float,
        cutoff: float = 3.0,
        bond_fraction: float = 0.35,
        species: Sequence[int] = SPECIES,
    ) -> None:
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be an AtomisticUnitSystem.")
        configuration = BondChargeMomentConfiguration(
            cutoff=cutoff, species=species, bond_fraction=bond_fraction
        )
        count = len(configuration.species)
        rows = {
            "electronegativity": electronegativity,
            "bond_charge": bond_charge,
            "cloud_charge": cloud_charge,
            "cloud_width": cloud_width,
            "hardness": hardness,
            "log_polarizability": log_polarizability,
        }
        for name, value in rows.items():
            host = np.asarray(value, dtype=np.float64)
            if host.shape != (count,) or not np.all(np.isfinite(host)):
                raise ValueError(f"{name} must be one finite value per species.")
            if name in ("cloud_width", "hardness") and np.any(host <= 0.0):
                raise ValueError(f"{name} must be positive.")
            setattr(self, name, jnp.asarray(host))
        if not math.isfinite(float(screening)):
            raise ValueError("screening must be finite.")
        self.screening = jnp.asarray(float(screening), dtype=jnp.float64)
        self.configuration = configuration
        self.scale = units.scale
        self.precision = AtomisticPrecisionPolicy()
        self.charge_unit = units.charge_unit
        self.architecture_id = canonical_fingerprint(
            {
                "kind": "bond-charge-site-moment-model",
                "law": "pair-transfer-bond-charges-gaussian-cloud-c2-envelope",
                "cutoff": float(configuration.cutoff).hex(),
                "species": list(configuration.species),
                "bond_fraction": float(configuration.bond_fraction).hex(),
                "scale": units.scale.scale_id,
                "charge_unit": units.charge_unit.unit_id,
                "basis": BASIS,
            }
        )
        self.method_id = _METHOD_ID

    # ------------------------------------------------------------ declarations

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        # Graph image routes carry the cell, so strain enters every moment.
        return AtomisticPotentialCapabilities(
            orthorhombic_periodic=True, triclinic_periodic=True, cell_derivative=True
        )

    @property
    def derivative_contract(self) -> DerivativeContract:
        """Polynomial-exponential in displacements; C2 at the cutoff join."""
        return DerivativeContract(
            (
                SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),
                SurfaceDerivative(
                    DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH
                ),
            ),
            route=DerivativeRoute.DIRECT,
            regularity=DerivativeRegularity.piecewise_smooth(
                continuity=2,
                conditions=(atomistic_edge_distance_locus(self.configuration.cutoff),),
            ),
        )

    def unit(self, observable: AtomisticObservable, /) -> UnitDefinition:
        """Declared physical output unit of one published field."""
        charge = self.charge_unit
        length = self.scale.length_unit
        energy = self.scale.energy_unit
        match observable:
            case AtomisticObservable.CHARGES:
                return charge
            case AtomisticObservable.DIPOLE:
                return derived_unit(
                    f"{charge.symbol}*{length.symbol}", ((charge, 1), (length, 1))
                )
            case AtomisticObservable.QUADRUPOLE:
                return derived_unit(
                    f"{charge.symbol}*{length.symbol}^2", ((charge, 1), (length, 2))
                )
            case AtomisticObservable.POLARIZABILITY:
                return derived_unit(
                    f"{charge.symbol}^2*{length.symbol}^2/{energy.symbol}",
                    ((charge, 2), (length, 2), (energy, -1)),
                )
        raise ValueError(f"The model publishes no {observable.value!r} field.")

    def port(self, observable: AtomisticObservable, /) -> phx.ValuePort:
        """Exact scientific port of one published atom-level field."""
        dimension = self.unit(observable).dimension
        match observable:
            case AtomisticObservable.CHARGES:
                return phx.ValuePort(
                    "bond-charge-model.atom-charge",
                    event_shape=(),
                    component_ids=("q",),
                    representation="atom-partial-charge",
                    dimensions=(dimension,),
                    normalization_id="identity",
                )
            case AtomisticObservable.DIPOLE:
                return phx.ValuePort(
                    "bond-charge-model.atom-dipole",
                    event_shape=(3,),
                    component_ids=("x", "y", "z"),
                    representation="atom-site-dipole",
                    dimensions=(dimension,) * 3,
                    frame_id="system-cartesian",
                    normalization_id=BASIS,
                )
            case AtomisticObservable.QUADRUPOLE:
                return phx.ValuePort(
                    "bond-charge-model.atom-raw-second-moment",
                    event_shape=(6,),
                    component_ids=("xx", "xy", "xz", "yy", "yz", "zz"),
                    representation="atom-primitive-raw-second-moment",
                    dimensions=(dimension,) * 6,
                    frame_id="system-cartesian",
                    normalization_id=BASIS,
                )
            case AtomisticObservable.POLARIZABILITY:
                return phx.ValuePort(
                    "bond-charge-model.atom-polarizability",
                    event_shape=(),
                    component_ids=("alpha",),
                    representation="atom-isotropic-polarizability",
                    dimensions=(dimension,),
                    normalization_id="isotropic-scalar",
                )
        raise ValueError(f"The model publishes no {observable.value!r} field.")

    def reference(
        self,
        observable: AtomisticObservable,
        /,
        *,
        producer: str = PRODUCER,
        unit: UnitDefinition | None = None,
    ) -> AtomisticFieldReference:
        """Atom-level consumer reference of one published port (declared unit default)."""
        return AtomisticFieldReference(
            producer,
            observable,
            self.port(observable),
            level=AtomisticConditionLevel.ATOM,
            unit=self.unit(observable) if unit is None else unit,
        )

    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        if batch.scale.scale_id != self.scale.scale_id:
            raise ValueError("Model and structure must share one exact scale contract.")

    # --------------------------------------------------------------- evaluation

    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        inputs.require_compatible(self.scale, self.capabilities.species_kind)
        selected: tuple[AtomisticObservable, ...] = ()
        if request is not None:
            request.require_supported(
                (AtomisticObservable.ENERGY, AtomisticObservable.ATOM_ENERGY, *_FIELDS),
                "BondChargeMomentModel",
            )
            selected = tuple(value for value in _FIELDS if value in request.observables)
        ir = inputs.graph.graph
        if ir.receivers is None or ir.senders is None or ir.edge_mask is None:
            raise ValueError("The moment model needs the bound graph's explicit edges.")
        dtype = jnp.dtype(self.precision.compute_dtype)
        cases, atoms = inputs.case_count, inputs.atom_capacity
        count = cases * atoms
        mask = inputs.atom_mask
        numbers = inputs.species_ids
        table = jnp.asarray(self.configuration.species, dtype=numbers.dtype)
        matches = numbers[:, None] == table[None, :]
        known = jnp.any(matches, axis=1) | ~mask
        kind = jnp.argmax(matches, axis=1)
        case_valid = inputs.successful & jnp.all(known.reshape((cases, atoms)), axis=1)

        receivers, senders = ir.receivers, ir.senders
        # displacement = x_receiver - x_neighbor_image, so the bond from the
        # receiving atom i to its neighbor image j is its negative.
        bond = -jnp.asarray(ir.edges["displacement"], dtype=dtype)
        ratio = jnp.sum(bond * bond, axis=-1) / (self.configuration.cutoff**2)
        active = ir.edge_mask & (ratio < 1.0) & mask[receivers] & mask[senders]
        envelope = jnp.where(active, (1.0 - jnp.where(active, ratio, 0.0)) ** 3, 0.0)
        own, other = kind[receivers], kind[senders]

        def gather(values: Array) -> Array:
            shape = (count, *values.shape[1:])
            return jnp.zeros(shape, dtype=dtype).at[receivers].add(values)

        charges = gather(
            (self.electronegativity[other] - self.electronegativity[own]) * envelope
        )
        bond_charges = self.bond_charge[own] * envelope
        fraction = self.configuration.bond_fraction
        dipoles = fraction * gather(bond_charges[:, None] * bond)
        bond_moments = gather(
            bond_charges[:, None, None] * bond[:, :, None] * bond[:, None, :]
        )
        cloud = (self.cloud_charge * self.cloud_width**2)[kind]
        second = fraction**2 * bond_moments + cloud[:, None, None] * jnp.eye(
            3, dtype=dtype
        )
        polarizability = jnp.exp(
            self.log_polarizability[kind] - self.screening * gather(envelope)
        )
        atom_energy = jnp.where(mask, 0.5 * self.hardness[kind] * charges**2, 0.0)
        values = {
            AtomisticObservable.CHARGES: charges,
            AtomisticObservable.DIPOLE: dipoles,
            AtomisticObservable.QUADRUPOLE: second[:, _UNIQUE_ROWS, _UNIQUE_COLUMNS],
            AtomisticObservable.POLARIZABILITY: polarizability,
        }
        output = jnp.dtype(self.precision.output_dtype)
        fields = tuple(
            AtomisticObservableValue(
                observable,
                self.port(observable),
                jnp.where(
                    mask.reshape((count,) + (1,) * (values[observable].ndim - 1)),
                    values[observable],
                    0.0,
                ).astype(output),
                case_valid,
                level=AtomisticConditionLevel.ATOM,
                case_count=cases,
                atom_capacity=atoms,
                unit=self.unit(observable),
            )
            for observable in selected
        )
        atom_energy = atom_energy.reshape((cases, atoms))
        return AtomisticModelOutputs(
            jnp.where(case_valid, atom_energy.sum(axis=1), jnp.nan).astype(output),
            atom_energy.astype(output),
            case_valid,
            additional=fields,
        )


def moment_model(
    units: AtomisticUnitSystem, /, **overrides: Any
) -> BondChargeMomentModel:
    """The fixture parameterization (H, C, O rows); ``overrides`` replace rows.

    Values are chosen for a well-conditioned test law, not fitted to any
    chemistry. Polarizability volumes ``(0.15, 0.35, 0.30)`` length^3 convert to
    physical polarizabilities by ``alpha = alpha_v / k`` with the system
    Coulomb constant ``k``.
    """
    volumes = np.asarray([0.15, 0.35, 0.30])
    parameters: dict[str, Any] = {
        "electronegativity": (0.10, 0.18, 0.42),
        "bond_charge": (-0.04, -0.07, -0.10),
        "cloud_charge": (-1.0, -4.0, -6.0),
        "cloud_width": (0.45, 0.55, 0.50),
        "hardness": (6.0, 5.0, 8.0),
        "log_polarizability": tuple(np.log(volumes / units.coulomb_constant)),
        "screening": 0.15,
    }
    parameters.update(overrides)
    return BondChargeMomentModel(units, **parameters)


def moment_fields(
    model: BondChargeMomentModel,
    /,
    *,
    producer: str = PRODUCER,
    dipole_unit: UnitDefinition | None = None,
    second_moment_unit: UnitDefinition | None = None,
) -> MultipoleMomentFields:
    """Every published moment degree as typed ``cartesian-primitive`` fields."""
    return MultipoleMomentFields(
        BASIS,
        charges=model.reference(AtomisticObservable.CHARGES, producer=producer),
        dipoles=model.reference(
            AtomisticObservable.DIPOLE, producer=producer, unit=dipole_unit
        ),
        second_moments=model.reference(
            AtomisticObservable.QUADRUPOLE, producer=producer, unit=second_moment_unit
        ),
    )


# ------------------------------------------------------- independent host law


def _neighbor_images(
    positions: np.ndarray, vectors: np.ndarray | None, cutoff: float, /
) -> list[list[tuple[int, np.ndarray]]]:
    """Every ``(neighbor, bond vector)`` image of each atom inside ``cutoff``."""
    count = positions.shape[0]
    if vectors is None:
        shifts = [np.zeros(3)]
    else:
        volume = abs(float(np.linalg.det(vectors)))
        reach = []
        for axis in range(3):
            first, second = (index for index in range(3) if index != axis)
            height = volume / float(
                np.linalg.norm(np.cross(vectors[first], vectors[second]))
            )
            reach.append(math.ceil(cutoff / height) + 1)
        shifts = [
            np.asarray(image, dtype=np.float64) @ vectors
            for image in itertools.product(*(range(-n, n + 1) for n in reach))
        ]
    bonds: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(count)]
    for i in range(count):
        for j in range(count):
            for shift in shifts:
                vector = positions[j] + shift - positions[i]
                distance = float(np.linalg.norm(vector))
                if distance == 0.0 or distance >= cutoff:
                    continue
                bonds[i].append((j, vector))
    return bonds


class ExplicitSiteMoments(TypedDict):
    """Host defining-law record: float64 site arrays and the producer energy."""

    charges: np.ndarray
    dipoles: np.ndarray
    second_moments: np.ndarray
    polarizabilities: np.ndarray
    energy: float


def explicit_site_moments(
    model: BondChargeMomentModel,
    positions: Any,
    atomic_numbers: Sequence[int],
    /,
    *,
    cell_vectors: Any = None,
    quadrature_order: int = 6,
) -> ExplicitSiteMoments:
    """Host evaluation of the defining law from explicitly enumerated charges.

    Each atom's density is assembled as a list of point charges (nuclear
    remainder and bond charges) whose moments are summed directly, and its
    Gaussian cloud second moment is integrated by a tensor Gauss-Hermite rule;
    the neighbor images are enumerated over every lattice shift that can reach
    the cutoff. Returns float64 ``charges (N,)``, ``dipoles (N, 3)``,
    ``second_moments (N, 3, 3)`` (primitive raw), ``polarizabilities (N,)`` and
    the model ``energy``. No JAX or native kernel is used.
    """
    x = np.asarray(positions, dtype=np.float64)
    vectors = None if cell_vectors is None else np.asarray(cell_vectors, np.float64)
    species = model.configuration.species
    kind = [species.index(int(number)) for number in atomic_numbers]
    chi = np.asarray(model.electronegativity, dtype=np.float64)
    beta = np.asarray(model.bond_charge, dtype=np.float64)
    cloud = np.asarray(model.cloud_charge, dtype=np.float64)
    width = np.asarray(model.cloud_width, dtype=np.float64)
    eta = np.asarray(model.hardness, dtype=np.float64)
    log_alpha = np.asarray(model.log_polarizability, dtype=np.float64)
    gamma = float(np.asarray(model.screening))
    cutoff = model.configuration.cutoff
    fraction = model.configuration.bond_fraction
    nodes, weights = np.polynomial.hermite_e.hermegauss(quadrature_order)
    weights = weights / math.sqrt(2.0 * math.pi)
    count = x.shape[0]
    charges = np.zeros(count)
    dipoles = np.zeros((count, 3))
    second_moments = np.zeros((count, 3, 3))
    polarizabilities = np.zeros(count)
    for i, bonds in enumerate(_neighbor_images(x, vectors, cutoff)):
        s = kind[i]
        transfer = 0.0
        coordination = 0.0
        points: list[tuple[float, np.ndarray]] = []
        for j, vector in bonds:
            f = (1.0 - float(vector @ vector) / cutoff**2) ** 3
            transfer += (chi[kind[j]] - chi[s]) * f
            coordination += f
            points.append((beta[s] * f, fraction * vector))
        nucleus = transfer - sum(value for value, _ in points)
        points.append((nucleus, np.zeros(3)))
        charge = sum(value for value, _ in points)
        dipole = sum(value * where for value, where in points)
        second = sum(value * np.outer(where, where) for value, where in points)
        # The compensated cloud c [g_w - delta] carries no charge or dipole.
        for a, b, c in itertools.product(range(quadrature_order), repeat=3):
            where = width[s] * np.asarray([nodes[a], nodes[b], nodes[c]])
            second = second + cloud[s] * weights[a] * weights[b] * weights[c] * np.outer(
                where, where
            )
        charges[i] = charge
        dipoles[i] = dipole
        second_moments[i] = second
        polarizabilities[i] = math.exp(log_alpha[s] - gamma * coordination)
    energy = 0.5 * float(np.sum(eta[kind] * charges**2))
    return ExplicitSiteMoments(
        charges=charges,
        dipoles=dipoles,
        second_moments=second_moments,
        polarizabilities=polarizabilities,
        energy=energy,
    )


def unique_second_moments(matrices: np.ndarray, /) -> np.ndarray:
    """Pack ``(N, 3, 3)`` raw second moments as the published six components."""
    return np.asarray(matrices)[:, _UNIQUE_ROWS, _UNIQUE_COLUMNS]


# ------------------------------------------------------------- compositions

UNITS = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
CLUSTER_NUMBERS = (8, 6, 1, 1, 8, 1)
"""A six-atom finite cluster: no molecule is implied."""
CLUSTER = np.asarray(
    [
        [0.00, 0.00, 0.00],
        [1.45, 0.10, -0.05],
        [2.05, 1.40, 0.20],
        [2.00, -0.85, 0.95],
        [-0.60, 1.35, 0.55],
        [-0.85, -0.70, -1.10],
    ]
)
"""Pair distances lie in [1.45, 2.68] or [3.42, 3.81]: every pair is well away
from the 3.0 cutoff join, so finite differences stay inside one smooth piece."""
SITE_WIDTHS = {1: 0.35, 6: 0.45, 8: 0.40}
"""Consumer Gaussian site standard deviations (length) per atomic number."""
PERIODIC_CELL = np.asarray([[6.2, 0.0, 0.0], [0.5, 6.1, 0.0], [0.3, -0.2, 6.3]])
PERIODIC_OFFSET = np.asarray([2.0, 2.0, 2.2])
"""``CLUSTER + PERIODIC_OFFSET`` lies inside ``PERIODIC_CELL``; its nearest
nonzero image pair is 3.57 apart, so periodic and finite bonds coincide."""
_SEARCH = AtomisticDenseImageSearch(maximum_dense_routes=8192)


def cluster_system(
    *,
    numbers: Sequence[int] = CLUSTER_NUMBERS,
    cell: np.ndarray | None = None,
    units: AtomisticUnitSystem = UNITS,
) -> Any:
    """Prepared element system of the given atoms (finite unless ``cell``)."""
    count = len(numbers)
    return AtomisticSystemPlan(
        np.arange(count, dtype=np.int64),
        np.asarray(numbers, dtype=np.int64),
        np.ones(count, dtype=np.float64),
        units,
        cell=None if cell is None else PeriodicCell(np.asarray(cell)),
    ).prepare()


def site_widths(numbers: Sequence[int] = CLUSTER_NUMBERS) -> GaussianMultipoleWidths:
    """Consumer Gaussian width classes, one per admitted species."""
    return GaussianMultipoleWidths(
        np.asarray([SITE_WIDTHS[number] for number in SPECIES]),
        np.asarray([SPECIES.index(int(number)) for number in numbers]),
    )


def multipole_consumer(
    model: BondChargeMomentModel,
    system: Any,
    site: MultipoleSiteKind,
    /,
    *,
    route: Any = None,
    fields: MultipoleMomentFields | None = None,
    numbers: Sequence[int] = CLUSTER_NUMBERS,
) -> MultipoleElectrostaticPotential:
    """Degree-two point or Gaussian multipole term reading the predicted fields."""
    plan = MultipoleElectrostaticsPlan(
        MultipoleElectrostaticsDefinition(
            site, 2, coulomb_constant=system.plan.units.coulomb_constant
        ),
        FiniteMultipoleRoute() if route is None else route,
    )
    return MultipoleElectrostaticPotential(
        plan,
        moment_fields=moment_fields(model) if fields is None else fields,
        widths=site_widths(numbers) if site == "gaussian" else None,
    )


def polarization_plan(system: Any, site: PolarizationSiteKind, /) -> PolarizationPlan:
    """Bare (``damping="none"``) finite stationary PCG with a dense certificate."""
    return PolarizationPlan(
        coulomb_constant=system.plan.units.coulomb_constant,
        site_kind=site,
        damping="none",
        route=FiniteMultipoleRoute(),
        tolerance=1.0e-12,
        maximum_iterations=200,
        stability=PolarizationStabilityPolicy("dense-spectrum"),
    )


def polarization_consumer(
    model: BondChargeMomentModel,
    system: Any,
    site: PolarizationSiteKind,
    /,
    *,
    fields: MultipoleMomentFields | None = None,
    polarizability: AtomisticFieldReference | None = None,
    numbers: Sequence[int] = CLUSTER_NUMBERS,
) -> InducedDipolePolarizationPotential:
    """Induced-dipole term whose permanent moments and alphas are all predicted.

    The positional ``PermanentMultipoleSiteData`` is the term's explicit-data
    slot: with typed fields bound it fixes only the site capacity (its damping
    is unused under ``damping="none"``); every moment and polarizability the
    solve reads comes from the producer at evaluation time.
    """
    count = system.capacity
    explicit_slot = PermanentMultipoleSiteData(
        np.zeros(count),
        np.zeros((count, 3)),
        np.zeros((count, 3, 3)),
        np.zeros(count),
        np.ones(count),
    )
    return InducedDipolePolarizationPotential(
        polarization_plan(system, site),
        explicit_slot,
        widths=site_widths(numbers) if site == "gaussian" else None,
        moment_fields=moment_fields(model) if fields is None else fields,
        polarizability_field=(
            model.reference(AtomisticObservable.POLARIZABILITY)
            if polarizability is None
            else polarizability
        ),
    )


def predicted_program(
    model: BondChargeMomentModel, system: Any, consumers: Sequence[Any], /
) -> Any:
    """Prepared program: the producer term first, then its typed consumers."""
    periodic = system.cell is not None
    count = system.capacity
    return prepare_program(
        AtomisticPotentialProgram(
            [
                LearnedGraphPotentialTerm(model, name=PRODUCER, allow_periodic=periodic),
                *consumers,
            ]
        ),
        system,
        graph_execution=AtomisticGraphExecutionPlan(
            128 if periodic else 8, backend="particle"
        ),
        search=_SEARCH,
        skin=0.3,
        search_capacities=(search_capacity(1024, particles_per_cell=count, images=512),),
        role_capacities=(role_capacity(1024, 128),),
        frame=(
            None
            if periodic
            else PeriodicCell(
                20.0 * np.eye(3),
                origin=np.full((3,), -10.0),
                periodic_axes=(False,) * 3,
            )
        ),
    )


def second_moment_matrices(unique: Any, /) -> np.ndarray:
    """Symmetric ``(N, 3, 3)`` raw second moments of the published six components."""
    packed = np.asarray(unique, dtype=np.float64)
    matrices = np.zeros((packed.shape[0], 3, 3))
    matrices[:, _UNIQUE_ROWS, _UNIQUE_COLUMNS] = packed
    matrices[:, _UNIQUE_COLUMNS, _UNIQUE_ROWS] = packed
    return matrices


type PredictedConsumer = (
    MultipoleElectrostaticPotential | InducedDipolePolarizationPotential
)
"""A typed-field consumer built by ``multipole_consumer``/``polarization_consumer``."""


def bound_moment_fields(consumer: PredictedConsumer, /) -> MultipoleMomentFields:
    """The consumer's predicted moment fields; explicit-moment terms refuse."""
    fields = consumer.moment_fields
    if fields is None:
        raise ValueError("The consumer reads explicit moments, not predicted fields.")
    return fields


def bound_moment_references(
    consumer: PredictedConsumer, /
) -> tuple[AtomisticFieldReference, AtomisticFieldReference, AtomisticFieldReference]:
    """Charge, dipole and second-moment references, each required present."""
    fields = bound_moment_fields(consumer)
    charges, dipoles, second = fields.charges, fields.dipoles, fields.second_moments
    if charges is None or dipoles is None or second is None:
        raise ValueError("The consumer does not read every predicted moment degree.")
    return charges, dipoles, second


def bound_polarizability(
    consumer: InducedDipolePolarizationPotential, /
) -> AtomisticFieldReference:
    """The polarization consumer's predicted physical polarizability reference."""
    reference = consumer.polarizability_field
    if reference is None:
        raise ValueError("The polarization consumer reads explicit polarizabilities.")
    return reference


def held_consumer(consumer: PredictedConsumer, evaluation: Any, /) -> PredictedConsumer:
    """The same consumer with the evaluated predictions frozen as explicit data.

    Its forces omit every ``d(moment)/dx`` and ``d(alpha)/dx`` contribution:
    the connected forces minus (producer alone + held consumer) isolate the
    predicted-field chain-rule share.
    """
    q, mu, unique = (
        np.asarray(evaluation.field(reference).values, dtype=np.float64)
        for reference in bound_moment_references(consumer)
    )
    theta = second_moment_matrices(unique)
    if isinstance(consumer, MultipoleElectrostaticPotential):
        return MultipoleElectrostaticPotential(
            consumer.plan,
            moments=CartesianMultipoles(
                jnp.asarray(q), jnp.asarray(mu), jnp.asarray(theta), maximum_degree=2
            ),
            widths=consumer.widths,
        )
    alpha = np.asarray(
        evaluation.field(bound_polarizability(consumer)).values, dtype=np.float64
    )
    return InducedDipolePolarizationPotential(
        consumer.plan,
        PermanentMultipoleSiteData(q, mu, theta, alpha, np.ones(q.shape[0])),
        widths=consumer.widths,
    )


def consumer_evidence(
    evaluation: Any, row: int, /
) -> MultipoleElectrostaticsEvidence | PolarizationTermEvidence:
    """The single multipole or polarization evidence record of one program row."""
    (record,) = (
        value
        for value in evaluation.term_evidence[row]
        if isinstance(value, (MultipoleElectrostaticsEvidence, PolarizationTermEvidence))
    )
    return record


def smoke() -> dict[str, Any]:
    """Runtime smoke: every finite point/Gaussian composition, with evidence.

    ``python -m tests._support.atomistic_moment_model`` prints this record:
    energies, success flags, the moment and polarizability sources the
    consumers recorded, and the largest predicted-field chain-rule force share
    (connected forces minus producer-alone plus held-prediction consumer forces).
    """
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    model = moment_model(UNITS)
    system = cluster_system()
    positions = np.asarray(CLUSTER)
    record: dict[str, Any] = {"architecture_id": model.architecture_id}
    alone = predicted_program(model, system, ())
    producer = alone.evaluate(positions, bundle_state(alone, positions))
    for site in SITE_KINDS:
        consumers: tuple[tuple[str, PredictedConsumer], ...] = (
            ("multipole", multipole_consumer(model, system, site)),
            ("polarization", polarization_consumer(model, system, site)),
        )
        for name, consumer in consumers:
            program = predicted_program(model, system, (consumer,))
            evaluation = program.evaluate(positions, bundle_state(program, positions))
            held = prepare_program(
                AtomisticPotentialProgram([held_consumer(consumer, evaluation)]), system
            )
            fixed = held.evaluate(positions, bundle_state(held, positions))
            response = np.asarray(evaluation.forces) - (
                np.asarray(producer.forces) + np.asarray(fixed.forces)
            )
            evidence = consumer_evidence(evaluation, 1)
            record[f"{site}-{name}"] = {
                "successful": bool(evaluation.successful),
                "held_successful": bool(fixed.successful),
                "energy": float(evaluation.energy),
                "term_energies": np.asarray(evaluation.term_energies).tolist(),
                "moment_source": evidence.moment_source,
                "polarizability_source": (
                    evidence.polarizability_source
                    if isinstance(evidence, PolarizationTermEvidence)
                    else None
                ),
                "fields_id": bound_moment_fields(consumer).fields_id,
                "maximum_chain_force": float(np.max(np.abs(response))),
            }
    return record


if __name__ == "__main__":
    print(json.dumps(smoke(), indent=2, sort_keys=True))
