#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Asymmetric-unit state, bounded symmetry expansion and expansion provenance.

An `AsymmetricUnitPlan` binds one exact setting, its invariant metric chart and
the molecules of the asymmetric unit (ASU).  Each ASU molecule occupies one
admitted `SitePosition`: the general position, or a special position whose
stabilizer acts on the molecule through a declared atom permutation that is an
admitted molecular automorphism.  Expansion uses the left-coset
representatives of each site stabilizer, so a molecule on a special position
contributes ``multiplicity`` copies and the rational ``Z'`` contribution
``1 / stabilizer_order`` without duplicate copies.  `AsymmetricUnitLayout` is
the plan's numerical expansion view without molecular witnesses; compiled
consumers hold it while host owners keep the plan.

Coordinates follow `SettingMetricChart`: row cell ``H`` (lower triangular from
the native Cholesky factor), fractional column coordinates ``s`` and Cartesian
``r = H^T s``.  ASU molecules are rigid conformers placed by a fractional
center and a scalar-first unit quaternion acting in that Cartesian frame.
Copy centers are wrapped into ``[0, 1)^3`` with recorded integer shifts while
every atom stays on the whole-molecule image branch of its copy, so a molecule
crossing a cell face is never split.  Skew cells are handled exactly: atom
offsets enter fractional coordinates through a native small solve with
``H^T`` and are transformed by the integer rotation of the operation.

Handedness is explicit.  A copy produced by an improper operation is the
mirror image of its ASU molecule; for a stereo-chiral component that copy is
the enantiomer, which the chirality policy must admit.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import quaternion_rotation_matrix
from ...ein import contract
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...typing import (
    Bool,
    Dim,
    Float64,
    HostBool,
    HostFloat64,
    HostInt64,
    Int32,
    parse,
)
from ._hall import (
    add_vectors,
    ExactAffineOperation,
    multiply_rotations,
    reduce_translation,
    rotate_vector,
    rotation_determinant,
)
from ._input import ConformerPool, MolecularComponent
from ._lattice import SettingCellResult, SettingMetricChart
from ._setting import OperationDim, SpaceGroupSetting
from ._wyckoff import SitePosition, StabilizerElement


ChiralityPolicy: TypeAlias = Literal["enantiopure", "racemic"]
"""``enantiopure`` refuses any enantiomer copy of a stereo-chiral component;
``racemic`` requires each stereo-chiral component to have both handednesses."""

_SMALL3 = SmallLinearSolvePlan(3)


class AsymmetricSiteDim(Dim, minimum=1):
    """Molecules of the asymmetric unit."""


class CopyDim(Dim, minimum=1):
    """Molecule copies of the expanded cell."""


class ExpandedAtomDim(Dim, minimum=1):
    """Atoms of the expanded cell."""


class SiteActionDim(Dim, minimum=0):
    """Non-identity stabilizer elements acting on special-site molecules."""


class ConformerRowDim(Dim, minimum=1):
    """Stacked conformers of every ASU molecule."""


class SiteAtomDim(Dim, minimum=1):
    """Padded atom slots of one ASU molecule."""


def _identity(count: int, /) -> tuple[int, ...]:
    return tuple(range(count))


def _permutation(values: Sequence[int], count: int, /) -> tuple[int, ...]:
    permutation = tuple(values)
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in permutation
    ):
        raise TypeError("site permutations must contain integers.")
    if sorted(permutation) != list(range(count)):
        raise ValueError(
            f"site permutation {permutation} is not a permutation of {count} atoms."
        )
    return permutation


@final
class AsymmetricUnitSite(StrictModule, NonTrainableState):
    """One ASU molecule: its conformer pool, site and admitted stabilizer action.

    ``site_permutations[e]`` is the atom permutation ``p`` realized by
    stabilizer element ``site.stabilizer[e]``: that element maps atom ``i`` of
    the molecule onto atom ``p[i]``.  The general position needs no
    permutations; a special position requires one per stabilizer element.
    """

    pool: ConformerPool
    site: SitePosition
    site_permutations: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        pool: ConformerPool,
        site: SitePosition,
        /,
        *,
        site_permutations: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if not isinstance(pool, ConformerPool):
            raise TypeError("pool must be a ConformerPool.")
        if not isinstance(site, SitePosition):
            raise TypeError("site must be a SitePosition.")
        count = pool.component.atom_count
        if site_permutations is None:
            if site.special:
                raise ValueError(
                    "A special-position molecule needs one admitted atom permutation per "
                    "stabilizer element."
                )
            permutations: tuple[tuple[int, ...], ...] = (_identity(count),)
        else:
            permutations = tuple(
                _permutation(values, count) for values in site_permutations
            )
        if len(permutations) != site.stabilizer_order:
            raise ValueError("site_permutations must align with the site stabilizer.")
        if permutations[0] != _identity(count):
            raise ValueError("The identity stabilizer element must act as the identity.")
        self.pool = pool
        self.site = site
        self.site_permutations = permutations


def _element_operation(
    operations: tuple[ExactAffineOperation, ...], element: StabilizerElement, /
) -> ExactAffineOperation:
    index, shift = element
    rotation, translation = operations[index]
    return rotation, (
        translation[0] + shift[0],
        translation[1] + shift[1],
        translation[2] + shift[2],
    )


def _element_product(
    operations: tuple[ExactAffineOperation, ...],
    position: dict[ExactAffineOperation, int],
    left: StabilizerElement,
    right: StabilizerElement,
    /,
) -> StabilizerElement:
    """Return the exact affine product ``left o right`` as ``(k, n)``."""
    left_rotation, left_translation = _element_operation(operations, left)
    right_rotation, right_translation = _element_operation(operations, right)
    rotation = multiply_rotations(left_rotation, right_rotation)
    translation = add_vectors(
        rotate_vector(left_rotation, right_translation), left_translation
    )
    reduced = reduce_translation(translation)
    shift = (
        translation[0] - reduced[0],
        translation[1] - reduced[1],
        translation[2] - reduced[2],
    )
    return position[(rotation, reduced)], (
        shift[0].numerator,
        shift[1].numerator,
        shift[2].numerator,
    )


def _admit_site_action(
    site: AsymmetricUnitSite,
    operations: tuple[ExactAffineOperation, ...],
    position: dict[ExactAffineOperation, int],
    /,
) -> None:
    """Verify the declared permutations form an admitted molecular group action."""
    component = site.pool.component
    stabilizer = site.site.stabilizer
    lookup = {element: index for index, element in enumerate(stabilizer)}
    automorphisms = frozenset(component.automorphisms)
    inverting = (
        frozenset()
        if component.mirror_automorphism is None
        else frozenset(
            tuple(
                automorphism[component.mirror_automorphism[i]]
                for i in range(len(automorphism))
            )
            for automorphism in component.automorphisms
        )
    )
    has_tetrahedral = bool(component.tetrahedral_stereo)
    for element, permutation in zip(stabilizer, site.site_permutations, strict=True):
        proper = rotation_determinant(operations[element[0]][0]) > 0
        if proper:
            admitted = permutation in automorphisms
        elif component.stereo_chiral:
            raise ValueError(
                f"Component {component.name!r} is stereo-chiral; an improper stabilizer "
                "element would map the molecule onto its own enantiomer."
            )
        elif has_tetrahedral:
            admitted = permutation in inverting
        else:
            admitted = permutation in automorphisms
        if not admitted:
            raise ValueError(
                f"Site permutation {permutation} of stabilizer element {element} is not an "
                f"admitted {'automorphism' if proper else 'mirror automorphism'} of "
                f"component {component.name!r}"
                + (
                    ""
                    if component.automorphisms_complete
                    else " (automorphism list incomplete)"
                )
                + "."
            )
    for left, left_permutation in zip(stabilizer, site.site_permutations, strict=True):
        for right, right_permutation in zip(
            stabilizer, site.site_permutations, strict=True
        ):
            product = _element_product(operations, position, left, right)
            index = lookup.get(product)
            if index is None:
                raise RuntimeError("A site stabilizer is not closed under composition.")
            composed = tuple(
                left_permutation[right_permutation[i]]
                for i in range(len(right_permutation))
            )
            if site.site_permutations[index] != composed:
                raise ValueError(
                    "site_permutations are not a group action: the product of stabilizer "
                    f"elements {left} and {right} acts as {site.site_permutations[index]}, "
                    f"not {composed}."
                )


@final
class AsymmetricUnitPlan(StrictModule, NonTrainableState):
    """Static ASU layout, admitted chirality and expansion provenance.

    Expanded copies are site-major and coset-minor; expanded atoms are
    copy-major.  Provenance arrays name, for every copy, its ASU site and
    operation and, for every atom, its copy, ASU site, local atom, component
    and operation.  ``copy_handedness`` is the determinant of the operation;
    ``copy_enantiomer`` marks mirror copies of stereo-chiral components.
    ``component_z_prime`` is the rational number of ASU molecules per component.
    """

    __strict_contract__ = True

    setting: SpaceGroupSetting
    chart: SettingMetricChart
    sites: tuple[AsymmetricUnitSite, ...]
    components: tuple[MolecularComponent, ...]
    rotations: HostInt64[OperationDim, Literal[3], Literal[3]]
    translations: HostFloat64[OperationDim, Literal[3]]
    site_component: HostInt64[AsymmetricSiteDim]
    site_atom_count: HostInt64[AsymmetricSiteDim]
    site_conformer_offset: HostInt64[AsymmetricSiteDim]
    site_conformer_count: HostInt64[AsymmetricSiteDim]
    conformer_table: HostFloat64[ConformerRowDim, SiteAtomDim, Literal[3]]
    site_constraints: HostFloat64[AsymmetricSiteDim, Literal[3], Literal[3]]
    site_offsets: HostFloat64[AsymmetricSiteDim, Literal[3]]
    site_constraint_mask: HostBool[AsymmetricSiteDim, Literal[3]]
    copy_site: HostInt64[CopyDim]
    copy_operation: HostInt64[CopyDim]
    copy_handedness: HostInt64[CopyDim]
    copy_enantiomer: HostBool[CopyDim]
    atom_copy: HostInt64[ExpandedAtomDim]
    atom_site: HostInt64[ExpandedAtomDim]
    atom_local: HostInt64[ExpandedAtomDim]
    atom_component: HostInt64[ExpandedAtomDim]
    atom_operation: HostInt64[ExpandedAtomDim]
    action_site: HostInt64[SiteActionDim]
    action_operation: HostInt64[SiteActionDim]
    action_permutation: HostInt64[SiteActionDim, SiteAtomDim]
    action_mask: HostBool[SiteActionDim, SiteAtomDim]
    chirality: ChiralityPolicy = eqx.field(static=True)
    component_z_prime: tuple[tuple[str, Fraction], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        setting: SpaceGroupSetting,
        sites: Sequence[AsymmetricUnitSite],
        /,
        *,
        chirality: ChiralityPolicy,
    ) -> None:
        if not isinstance(setting, SpaceGroupSetting):
            raise TypeError("setting must be a SpaceGroupSetting.")
        sites_ = tuple(sites)
        if not sites_:
            raise ValueError("An asymmetric unit needs at least one molecule.")
        if any(not isinstance(site, AsymmetricUnitSite) for site in sites_):
            raise TypeError("sites must be AsymmetricUnitSite values.")
        chirality_ = parse(chirality, ChiralityPolicy, "chirality")
        operations = setting.exact_operations()
        position = {operation: index for index, operation in enumerate(operations)}
        for site in sites_:
            if site.site.setting_id != setting.setting_id:
                raise ValueError("Every ASU site must belong to the plan setting.")
            if site.site.special:
                _admit_site_action(site, operations, position)
        components: list[MolecularComponent] = []
        component_index: dict[str, int] = {}
        for site in sites_:
            component = site.pool.component
            if component.component_id not in component_index:
                component_index[component.component_id] = len(components)
                components.append(component)
        determinants = [rotation_determinant(rotation) for rotation, _ in operations]
        copy_site, copy_operation = [], []
        for index, site in enumerate(sites_):
            for operation in site.site.coset_representatives:
                copy_site.append(index)
                copy_operation.append(operation)
        handedness = [determinants[operation] for operation in copy_operation]
        enantiomer = [
            handedness[c] < 0 and sites_[copy_site[c]].pool.component.stereo_chiral
            for c in range(len(copy_site))
        ]
        for component in components:
            if not component.stereo_chiral:
                continue
            signs = {
                handedness[c]
                for c in range(len(copy_site))
                if sites_[copy_site[c]].pool.component.component_id
                == component.component_id
            }
            if chirality_ == "enantiopure" and -1 in signs:
                raise ValueError(
                    f"Stereo-chiral component {component.name!r} would be copied by an "
                    "improper operation into its enantiomer under an enantiopure policy."
                )
            if chirality_ == "racemic" and -1 not in signs:
                raise ValueError(
                    f"A racemic policy needs an improper copy of stereo-chiral component "
                    f"{component.name!r}; the setting supplies none at its sites."
                )
        atom_counts = [site.pool.component.atom_count for site in sites_]
        width = max(atom_counts)
        conformer_rows: list[np.ndarray] = []
        offsets, counts = [], []
        for site in sites_:
            offsets.append(len(conformer_rows))
            counts.append(len(site.pool.conformers))
            for conformer in site.pool.conformers:
                values = np.asarray(jax.device_get(conformer.positions), dtype=np.float64)
                row = np.zeros((width, 3), dtype=np.float64)
                row[: values.shape[0]] = values
                conformer_rows.append(row)
        atom_copy, atom_site, atom_local, atom_component, atom_operation = (
            [],
            [],
            [],
            [],
            [],
        )
        for c, (index, operation) in enumerate(
            zip(copy_site, copy_operation, strict=True)
        ):
            component = sites_[index].pool.component
            for local in range(component.atom_count):
                atom_copy.append(c)
                atom_site.append(index)
                atom_local.append(local)
                atom_component.append(component_index[component.component_id])
                atom_operation.append(operation)
        constraints = np.zeros((len(sites_), 3, 3), dtype=np.float64)
        site_offsets = np.zeros((len(sites_), 3), dtype=np.float64)
        mask = np.zeros((len(sites_), 3), dtype=np.bool_)
        for index, site in enumerate(sites_):
            for row, (constraint, offset) in enumerate(
                zip(site.site.constraints, site.site.offsets, strict=True)
            ):
                constraints[index, row] = constraint
                site_offsets[index, row] = float(offset)
                mask[index, row] = True
        action_site, action_operation, action_permutation, action_mask = [], [], [], []
        for index, site in enumerate(sites_):
            for element, permutation in zip(
                site.site.stabilizer[1:], site.site_permutations[1:], strict=True
            ):
                action_site.append(index)
                action_operation.append(element[0])
                padded = np.zeros((width,), dtype=np.int64)
                padded[: len(permutation)] = permutation
                valid = np.zeros((width,), dtype=np.bool_)
                valid[: len(permutation)] = True
                action_permutation.append(padded)
                action_mask.append(valid)
        z_prime: dict[str, Fraction] = {}
        for site in sites_:
            key = site.pool.component.component_id
            z_prime[key] = z_prime.get(key, Fraction(0)) + Fraction(
                1, site.site.stabilizer_order
            )
        self.setting = setting
        self.chart = SettingMetricChart(setting)
        self.sites = sites_
        self.components = tuple(components)
        self.rotations = setting.rotation_array()
        self.translations = np.asarray(
            setting.translation_numerator_array(), dtype=np.float64
        ) / float(setting.translation_denominator)
        self.site_component = np.asarray(
            [component_index[site.pool.component.component_id] for site in sites_],
            dtype=np.int64,
        )
        self.site_atom_count = np.asarray(atom_counts, dtype=np.int64)
        self.site_conformer_offset = np.asarray(offsets, dtype=np.int64)
        self.site_conformer_count = np.asarray(counts, dtype=np.int64)
        self.conformer_table = np.stack(conformer_rows)
        self.site_constraints = constraints
        self.site_offsets = site_offsets
        self.site_constraint_mask = mask
        self.copy_site = np.asarray(copy_site, dtype=np.int64)
        self.copy_operation = np.asarray(copy_operation, dtype=np.int64)
        self.copy_handedness = np.asarray(handedness, dtype=np.int64)
        self.copy_enantiomer = np.asarray(enantiomer, dtype=np.bool_)
        self.atom_copy = np.asarray(atom_copy, dtype=np.int64)
        self.atom_site = np.asarray(atom_site, dtype=np.int64)
        self.atom_local = np.asarray(atom_local, dtype=np.int64)
        self.atom_component = np.asarray(atom_component, dtype=np.int64)
        self.atom_operation = np.asarray(atom_operation, dtype=np.int64)
        self.action_site = np.asarray(action_site, dtype=np.int64).reshape((-1,))
        self.action_operation = np.asarray(action_operation, dtype=np.int64).reshape(
            (-1,)
        )
        self.action_permutation = (
            np.stack(action_permutation)
            if action_permutation
            else np.zeros((0, width), dtype=np.int64)
        )
        self.action_mask = (
            np.stack(action_mask) if action_mask else np.zeros((0, width), dtype=np.bool_)
        )
        self.chirality = chirality_
        self.component_z_prime = tuple(z_prime.items())
        self.plan_id = canonical_fingerprint(
            {
                "kind": "crystal-asymmetric-unit-plan",
                "setting_id": setting.setting_id,
                "chirality": chirality_,
                "sites": [
                    {
                        "pool_id": site.pool.pool_id,
                        "site_key": site.site.site_key,
                        "permutations": [list(p) for p in site.site_permutations],
                    }
                    for site in sites_
                ],
            }
        )

    @property
    def site_count(self) -> int:
        return len(self.sites)

    @property
    def copy_count(self) -> int:
        return int(self.copy_site.shape[0])

    @property
    def atom_count(self) -> int:
        return int(self.atom_copy.shape[0])

    @property
    def operation_count(self) -> int:
        return self.setting.operation_count

    def molecules_per_cell(self, component_id: str, /) -> Fraction:
        """Return ``Z`` of a component: its ASU ``Z'`` times the operation count."""
        for key, value in self.component_z_prime:
            if key == component_id:
                return value * self.setting.operation_count
        raise KeyError(component_id)


@final
class AsymmetricUnitLayout(StrictModule, NonTrainableState):
    """Prepared numerical expansion layout of one `AsymmetricUnitPlan`.

    Holds exactly what `expand_asymmetric_unit` and the compiled symmetry
    charts read: the metric chart, the operation arrays, the stacked conformer
    table, the copy, atom, site-action and site-constraint indices, the element
    of every expanded atom and the plan's ``plan_id``.  The plan's host arrays
    are converted once to canonical JAX arrays (``int32`` indices, ``float64``
    tables, ``bool`` masks).  It carries none of the plan's molecular witnesses
    (sites, pools, components, setting): compiled consumers hold the prepared
    layout while host owners keep the plan.
    """

    __strict_contract__ = True

    chart: SettingMetricChart
    rotations: Int32[OperationDim, Literal[3], Literal[3]]
    translations: Float64[OperationDim, Literal[3]]
    site_conformer_offset: Int32[AsymmetricSiteDim]
    site_conformer_count: Int32[AsymmetricSiteDim]
    conformer_table: Float64[ConformerRowDim, SiteAtomDim, Literal[3]]
    site_constraints: Float64[AsymmetricSiteDim, Literal[3], Literal[3]]
    site_offsets: Float64[AsymmetricSiteDim, Literal[3]]
    site_constraint_mask: Bool[AsymmetricSiteDim, Literal[3]]
    copy_site: Int32[CopyDim]
    copy_operation: Int32[CopyDim]
    atom_copy: Int32[ExpandedAtomDim]
    atom_site: Int32[ExpandedAtomDim]
    atom_local: Int32[ExpandedAtomDim]
    atom_operation: Int32[ExpandedAtomDim]
    atomic_numbers: Int32[ExpandedAtomDim]
    action_site: Int32[SiteActionDim]
    action_operation: Int32[SiteActionDim]
    action_permutation: Int32[SiteActionDim, SiteAtomDim]
    action_mask: Bool[SiteActionDim, SiteAtomDim]
    plan_id: str = eqx.field(static=True)

    def __init__(self, plan: AsymmetricUnitPlan, /) -> None:
        if not isinstance(plan, AsymmetricUnitPlan):
            raise TypeError("plan must be an AsymmetricUnitPlan.")

        def indices(values: np.ndarray, /) -> Array:
            return jnp.asarray(values, dtype=jnp.int32)

        def reals(values: np.ndarray, /) -> Array:
            return jnp.asarray(values, dtype=jnp.float64)

        self.chart = plan.chart
        self.rotations = indices(plan.rotations)
        self.translations = reals(plan.translations)
        self.site_conformer_offset = indices(plan.site_conformer_offset)
        self.site_conformer_count = indices(plan.site_conformer_count)
        self.conformer_table = reals(plan.conformer_table)
        self.site_constraints = reals(plan.site_constraints)
        self.site_offsets = reals(plan.site_offsets)
        self.site_constraint_mask = jnp.asarray(plan.site_constraint_mask)
        self.copy_site = indices(plan.copy_site)
        self.copy_operation = indices(plan.copy_operation)
        self.atom_copy = indices(plan.atom_copy)
        self.atom_site = indices(plan.atom_site)
        self.atom_local = indices(plan.atom_local)
        self.atom_operation = indices(plan.atom_operation)
        # (component, local atom) element table from the static components,
        # gathered by the expanded-atom provenance indices.
        width = max(component.atom_count for component in plan.components)
        elements = np.zeros((len(plan.components), width), dtype=np.int32)
        for row, component in enumerate(plan.components):
            elements[row, : component.atom_count] = component.atomic_numbers
        self.atomic_numbers = jnp.asarray(elements)[
            indices(plan.atom_component), self.atom_local
        ]
        self.action_site = indices(plan.action_site)
        self.action_operation = indices(plan.action_operation)
        self.action_permutation = indices(plan.action_permutation)
        self.action_mask = jnp.asarray(plan.action_mask)
        self.plan_id = plan.plan_id

    @property
    def site_count(self) -> int:
        return int(self.site_conformer_count.shape[0])

    @property
    def copy_count(self) -> int:
        return int(self.copy_site.shape[0])

    @property
    def atom_count(self) -> int:
        return int(self.atom_copy.shape[0])

    @property
    def operation_count(self) -> int:
        return int(self.rotations.shape[0])


@final
class AsymmetricUnitState(StrictModule):
    """Dynamic ASU coordinates.

    ``centers`` are fractional ``(sites, 3)``; ``orientations`` are scalar-first
    unit quaternions ``(sites, 4)`` rotating the conformer local frame into the
    chart's Cartesian frame; ``cell_coefficients`` are `SettingMetricChart`
    coordinates in squared length units; ``conformer_indices`` select one
    conformer of each site's pool.
    """

    centers: Array
    orientations: Array
    cell_coefficients: Array
    conformer_indices: Array

    def __init__(
        self,
        centers: ArrayLike,
        orientations: ArrayLike,
        cell_coefficients: ArrayLike,
        conformer_indices: ArrayLike,
    ) -> None:
        centers_ = jnp.asarray(centers)
        orientations_ = jnp.asarray(orientations)
        coefficients = jnp.asarray(cell_coefficients)
        indices = jnp.asarray(conformer_indices)
        if centers_.ndim != 2 or centers_.shape[1] != 3:
            raise ValueError("centers must have shape (sites, 3).")
        if orientations_.shape != (centers_.shape[0], 4):
            raise ValueError("orientations must have shape (sites, 4).")
        if coefficients.ndim != 1:
            raise ValueError("cell_coefficients must be a vector.")
        if indices.shape != (centers_.shape[0],):
            raise ValueError("conformer_indices must have shape (sites,).")
        for name, value in (
            ("centers", centers_),
            ("orientations", orientations_),
            ("cell_coefficients", coefficients),
        ):
            if not jnp.issubdtype(value.dtype, jnp.floating):
                raise TypeError(f"{name} must be real floating values.")
        if centers_.dtype != orientations_.dtype or centers_.dtype != coefficients.dtype:
            raise TypeError("ASU coordinates must share one floating dtype.")
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError("conformer_indices must be integers.")
        self.centers = centers_
        self.orientations = orientations_
        self.cell_coefficients = coefficients
        self.conformer_indices = indices.astype(jnp.int32)


@final
class ExpandedCrystal(StrictModule):
    """Expanded cell of one ASU state with geometry and site evidence.

    ``positions`` are Cartesian ``(atoms, 3)`` and ``fractional`` their
    unwrapped fractional coordinates on each copy's whole-molecule branch.
    ``copy_centers`` are wrapped fractional copy centers with
    ``copy_center_shifts`` the integer shifts that wrapped them, and
    ``copy_center_positions`` their Cartesian positions.  ``site_offsets`` are
    the Cartesian ASU atom offsets ``Q y`` (padded per site).
    ``orientation_residual`` is, per site, the largest
    ``|C_h Q y_i - Q y_{p_h(i)}|`` over non-identity stabilizer elements
    (zero for the general position) and ``center_residual`` the largest
    wrapped site-constraint residual of the ASU center in fractional units.
    ``offset_solve_successful`` reports the native small solve with ``H^T``.
    Provenance arrays live on the plan named by ``plan_id``.
    """

    positions: Array
    fractional: Array
    copy_centers: Array
    copy_center_shifts: Array
    copy_center_positions: Array
    site_offsets: Array
    atom_conformers: Array
    cell: SettingCellResult
    orientation_residual: Array
    center_residual: Array
    offset_solve_successful: Array
    plan_id: str = eqx.field(static=True)


def expand_asymmetric_unit(
    plan: AsymmetricUnitPlan | AsymmetricUnitLayout, state: AsymmetricUnitState, /
) -> ExpandedCrystal:
    """Expand an ASU state into the full cell (jit-compatible).

    A host `AsymmetricUnitPlan` is the cold form: its `AsymmetricUnitLayout`
    is prepared on every call (host inspection, tests).  Compiled consumers
    prepare the layout once and pass it; only the layout is read.
    """
    if isinstance(plan, AsymmetricUnitPlan):
        layout = AsymmetricUnitLayout(plan)
    elif isinstance(plan, AsymmetricUnitLayout):
        layout = plan
    else:
        raise TypeError("plan must be an AsymmetricUnitPlan or AsymmetricUnitLayout.")
    if not isinstance(state, AsymmetricUnitState):
        raise TypeError("state must be an AsymmetricUnitState.")
    if state.centers.shape[0] != layout.site_count:
        raise ValueError(f"state must describe {layout.site_count} ASU molecules.")
    if state.cell_coefficients.shape != (layout.chart.dimension,):
        raise ValueError(
            f"cell_coefficients must have shape ({layout.chart.dimension},)."
        )
    dtype = state.centers.dtype
    counts = jnp.asarray(layout.site_conformer_count, dtype=jnp.int32)
    indices = eqx.error_if(
        state.conformer_indices,
        jnp.any((state.conformer_indices < 0) | (state.conformer_indices >= counts)),
        "conformer_indices exceed a site's conformer pool.",
    )
    cell = layout.chart.cell(state.cell_coefficients)
    vectors = cell.vectors
    orientation = quaternion_rotation_matrix(state.orientations)
    rows = jnp.asarray(layout.site_conformer_offset, dtype=jnp.int32) + indices
    local = jnp.asarray(layout.conformer_table, dtype=dtype)[rows]
    offsets = contract("mij,maj->mai", orientation, local, backend="jax")
    solved = solve_small_linear(
        _SMALL3, jnp.swapaxes(vectors, -1, -2), offsets.reshape((-1, 3)).T
    )
    fractional_offsets = solved.value.T.reshape(offsets.shape)
    rotations = jnp.asarray(layout.rotations, dtype=dtype)
    translations = jnp.asarray(layout.translations, dtype=dtype)
    copy_site = jnp.asarray(layout.copy_site, dtype=jnp.int32)
    copy_operation = jnp.asarray(layout.copy_operation, dtype=jnp.int32)
    raw = (
        contract(
            "cij,cj->ci",
            rotations[copy_operation],
            state.centers[copy_site],
            backend="jax",
        )
        + translations[copy_operation]
    )
    shifts = -jax.lax.stop_gradient(jnp.floor(raw))
    wrapped = raw + shifts
    atom_copy = jnp.asarray(layout.atom_copy, dtype=jnp.int32)
    atom_site = jnp.asarray(layout.atom_site, dtype=jnp.int32)
    atom_local = jnp.asarray(layout.atom_local, dtype=jnp.int32)
    atom_operation = jnp.asarray(layout.atom_operation, dtype=jnp.int32)
    fractional = wrapped[atom_copy] + contract(
        "aij,aj->ai",
        rotations[atom_operation],
        fractional_offsets[atom_site, atom_local],
        backend="jax",
    )
    positions = contract("ai,ij->aj", fractional, vectors, backend="jax")
    copy_positions = contract("ci,ij->cj", wrapped, vectors, backend="jax")
    cartesian = cell.cartesian_rotations
    actions = int(layout.action_site.shape[0])
    if actions:
        action_site = jnp.asarray(layout.action_site, dtype=jnp.int32)
        mapped = contract(
            "pij,paj->pai",
            cartesian[jnp.asarray(layout.action_operation, dtype=jnp.int32)],
            offsets[action_site],
            backend="jax",
        )
        target = offsets[
            action_site[:, None], jnp.asarray(layout.action_permutation, dtype=jnp.int32)
        ]
        distance = jnp.linalg.norm(mapped - target, axis=-1)
        worst = jnp.max(
            jnp.where(
                jnp.asarray(layout.action_mask), distance, jnp.zeros((), dtype=dtype)
            ),
            axis=-1,
        )
        orientation_residual = jnp.maximum(
            jax.ops.segment_max(worst, action_site, num_segments=layout.site_count),
            jnp.zeros((), dtype=dtype),
        )
    else:
        orientation_residual = jnp.zeros((layout.site_count,), dtype=dtype)
    raw_constraint = contract(
        "mij,mj->mi",
        jnp.asarray(layout.site_constraints, dtype=dtype),
        state.centers,
        backend="jax",
    ) - jnp.asarray(layout.site_offsets, dtype=dtype)
    wrapped_constraint = raw_constraint - jnp.round(raw_constraint)
    center_residual = jnp.max(
        jnp.where(
            jnp.asarray(layout.site_constraint_mask),
            jnp.abs(wrapped_constraint),
            jnp.zeros((), dtype=dtype),
        ),
        axis=-1,
    )
    return ExpandedCrystal(
        positions,
        fractional,
        wrapped,
        shifts.astype(jnp.int32),
        copy_positions,
        offsets,
        indices[atom_site],
        cell,
        orientation_residual,
        center_residual,
        jnp.all(solved.successful),
        layout.plan_id,
    )


__all__ = [
    "AsymmetricUnitLayout",
    "AsymmetricUnitPlan",
    "AsymmetricUnitSite",
    "AsymmetricUnitState",
    "ChiralityPolicy",
    "CopyDim",
    "expand_asymmetric_unit",
    "ExpandedAtomDim",
    "ExpandedCrystal",
]
