#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Persisted molecular-crystal packing recipes.

A `CrystalPackingRecipe` binds one prepared `PreparedCrystalPacker` (its
`CrystalPackingPlan` and lane count) to the generation stream configuration it
runs under: campaign, pending capacity and steps per transition.  Its
``recipe_id`` binds the complete plan identity (the scientific packing law
``law_id`` with the contact resources and atom bound), the lane count and the
unfiltered stream identity (stream, sampler, numeric-revision and executable
identities of `PreparedAtomisticGenerationStream.identity`), so a display-only
change keeps ``recipe_id`` and every scientific identity.

`write_crystal_packing_recipe` stores the complete definition as one bounded,
pickle-free array archive: molecular components (atoms, charges, bonds and
orders, stereo, spin, valence, automorphism bounds and the admitted masses
with their provenance), conformer pools (the admitted constructor coordinates
as float64 arrays, thresholds, source, provenance and declared weights),
formula counts, the contact policy and its admitted radii or the declared
contact matrix, the volume policy and its admitted data, every setting's
reference Hall entry with its exact transformation, declared group weights and
prior provenance, ``Z'``, chirality, atom bound, cell, relief and contact
resource policies, the lane count and stream configuration.  The manifest
records every owning identity and a content address over the canonical recipe
and the array collection digest.  Declared weights, admitted conformer
coordinates and admitted data tables are restoration witnesses: the
scientific identities bind the laws they define (normalized weights, local
geometry, values and table identities), while exact constructor replay,
byte-exact re-encoding and ``content_id`` bind the witnesses themselves.

`read_crystal_packing_recipe` admits the archive under explicit limits,
verifies the content address, rebuilds every value through its owning
constructor, verifies each recorded identity and finally re-encodes the rebuilt
recipe, refusing unless it reproduces the stored recipe and arrays exactly.
Unknown, missing or mistyped fields, unreferenced or altered arrays and
altered inputs refuse; nothing is imported dynamically or unpickled.

Replay is exact in the numerical profile that wrote the recipe (backend,
precision and JAX configuration): conformer frames are recomputed by the
conformer constructor, so a recipe written under another profile can refuse
on its conformer identities rather than restore different coordinates.  The
generation continuation itself stays in `PreparedAtomisticGenerationStream`
checkpoints; ``recipe.stream()`` rebuilds the stream they authenticate against.
Proposal filters are executable stream policies supplied explicitly to
``stream``; the recorded stream identity is the unfiltered stream's.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import final

import equinox as eqx
import jax
import numpy as np

from ..._array_archive import (
    array_collection_digest,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import parse
from .._generation import AbstractProposalFilter, AtomisticGenerationStreamPlan
from .._neighborhood_bundle import (
    _image_capacity_dict,
    _image_capacity_from,
    _recipe_list,
    _recipe_record,
    _role_capacity_dict,
    _role_capacity_from,
    _search_dict,
    _search_from,
)
from ._asu import ChiralityPolicy
from ._input import (
    ConformerPool,
    ConformerSource,
    ContactPolicy,
    ContactRule,
    ElementQuantity,
    ElementTable,
    MolecularComponent,
    MolecularConformer,
    MolecularCrystalInput,
    UndeclaredStereoPolicy,
    ValencePolicy,
    VolumePolicy,
)
from ._packing import (
    CrystalCellPolicy,
    CrystalContactResources,
    CrystalPackingPlan,
    CrystalReliefPolicy,
    CrystalSpaceGroupPolicy,
    PreparedCrystalPacker,
)
from ._setting import SettingTransformation, SpaceGroupSetting


_RECIPE_KIND = "crystal-packing-recipe"
_ARCHIVE_KIND = "crystal-packing-recipe-archive"
_ARCHIVE_FIELDS = frozenset({"kind", "recipe", "array_digest", "content_id", "arrays"})
_CONTACT_ARRAY = "contact-distances"


# -- recipe ---------------------------------------------------------------------------------------


@final
class CrystalPackingRecipe(StrictModule, NonTrainableState):
    """A prepared packer with the generation stream configuration it runs under.

    ``pending_capacity`` and ``steps_per_transition`` default as in
    `PreparedCrystalPacker.stream`.  ``stream_identity`` is the identity of
    the unfiltered prepared stream that generation checkpoints authenticate.
    """

    packer: PreparedCrystalPacker
    campaign_id: str = eqx.field(static=True)
    pending_capacity: int = eqx.field(static=True)
    steps_per_transition: int = eqx.field(static=True)
    stream_identity: tuple[tuple[str, str], ...] = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)

    def __init__(
        self,
        packer: PreparedCrystalPacker,
        /,
        *,
        campaign_id: str,
        pending_capacity: int | None = None,
        steps_per_transition: int | None = None,
    ) -> None:
        if not isinstance(packer, PreparedCrystalPacker):
            raise TypeError("packer must be a PreparedCrystalPacker.")
        stream = packer.stream(
            campaign_id,
            pending_capacity=pending_capacity,
            steps_per_transition=steps_per_transition,
        )
        identity = tuple(sorted(stream.prepare().identity().items()))
        self.packer = packer
        self.campaign_id = stream.campaign_id
        self.pending_capacity = stream.pending_capacity
        self.steps_per_transition = stream.steps_per_transition
        self.stream_identity = identity
        self.recipe_id = canonical_fingerprint(
            {
                "kind": _RECIPE_KIND,
                "plan": packer.plan.plan_id,
                "lanes": packer.lanes,
                "stream": dict(identity),
            }
        )

    @property
    def plan(self) -> CrystalPackingPlan:
        return self.packer.plan

    @property
    def lanes(self) -> int:
        return self.packer.lanes

    def stream(
        self, *, filters: Sequence[AbstractProposalFilter] = ()
    ) -> AtomisticGenerationStreamPlan:
        """The generation stream of this recipe (with explicit proposal filters)."""
        return self.packer.stream(
            self.campaign_id,
            pending_capacity=self.pending_capacity,
            steps_per_transition=self.steps_per_transition,
            filters=filters,
        )


@dataclass(frozen=True, slots=True)
class CrystalPackingRecipeReceipt:
    """Identity of one written or admitted recipe archive.

    ``content_id`` addresses the complete stored recipe (including display
    names and provenance statements) and ``array_digest`` its arrays;
    ``law_id`` is the scientific packing law, ``plan_id`` the complete plan
    (law, contact resources and atom bound) and ``recipe_id`` the plan with
    its lane count and stream configuration.
    """

    path: Path
    recipe_id: str
    law_id: str
    plan_id: str
    content_id: str
    array_digest: str


# -- strict values --------------------------------------------------------------------------------


def _text(value: object, label: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"The crystal recipe {label} must be a non-empty string.")
    return value


def _integer(value: object, label: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"The crystal recipe {label} must be an integer.")
    return value


def _real(value: object, label: str, /) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"The crystal recipe {label} must be a real number.")
    return float(value)


def _flag(value: object, label: str, /) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"The crystal recipe {label} must be a boolean.")
    return value


def _integers(value: object, label: str, /) -> list[int]:
    return [_integer(item, label) for item in _recipe_list(value)]


def _reals(value: object, label: str, /) -> list[float]:
    return [_real(item, label) for item in _recipe_list(value)]


def _rows(value: object, label: str, /) -> list[list[int]]:
    return [_integers(item, label) for item in _recipe_list(value)]


def _fraction(value: object, /) -> Fraction:
    text = _text(value, "setting transformation entry")
    try:
        number = Fraction(text)
    except ValueError as error:
        raise ValueError("A setting transformation entry is not a fraction.") from error
    if str(number) != text:
        raise ValueError("A setting transformation entry is not canonical.")
    return number


def _verified(recorded: object, actual: str, label: str, /) -> None:
    if _text(recorded, f"{label} identity") != actual:
        raise ValueError(f"The rebuilt {label} does not reproduce its recorded identity.")


class _Arrays:
    """Arrays of one recipe and the names its decoder consumed."""

    def __init__(self, values: Mapping[str, np.ndarray], /) -> None:
        self._values = values
        self._used: set[str] = set()

    def take(self, name: object, shape: tuple[int, ...], /) -> np.ndarray:
        key = _text(name, "array name")
        value = self._values.get(key)
        if value is None:
            raise ValueError(f"The crystal recipe array {key!r} is missing.")
        if key in self._used:
            raise ValueError(f"The crystal recipe array {key!r} is referenced twice.")
        if value.dtype != np.float64 or value.shape != shape:
            raise ValueError(f"The crystal recipe array {key!r} has the wrong layout.")
        self._used.add(key)
        return value

    def require_consumed(self) -> None:
        if set(self._values) != self._used:
            raise ValueError("The crystal recipe archive holds unreferenced arrays.")


def _host_float64(value: object, label: str, /) -> np.ndarray:
    host = np.asarray(jax.device_get(value))
    if host.dtype != np.float64:
        raise ValueError(
            f"The {label} is not float64; recipes persist float64 inputs only."
        )
    return np.ascontiguousarray(host)


# -- encoding -------------------------------------------------------------------------------------


def _table_dict(table: ElementTable, /) -> dict[str, object]:
    return {
        "quantity": table.quantity,
        "entries": [
            [number, value]
            for number, value in zip(table.atomic_numbers, table.values, strict=True)
        ],
        "provenance": table.provenance,
        "table_id": table.table_id,
    }


def _component_dict(component: MolecularComponent, /) -> dict[str, object]:
    source = component.mass_source()
    return {
        "name": component.name,
        "atomic_numbers": list(component.atomic_numbers),
        "formal_charges": list(component.formal_charges),
        "bonds": [list(bond) for bond in component.bonds],
        "bond_orders": list(component.bond_orders),
        "tetrahedral_stereo": [list(entry) for entry in component.tetrahedral_stereo],
        "double_bond_stereo": [list(entry) for entry in component.double_bond_stereo],
        "undeclared_stereo": (
            "unspecified"
            if component.unspecified_tetrahedral or component.unspecified_double_bonds
            else "refuse"
        ),
        "spin_multiplicity": component.spin_multiplicity,
        "masses": (
            {"kind": "table", "table": _table_dict(source)}
            if isinstance(source, ElementTable)
            else {"kind": "per-atom", "values": list(source)}
        ),
        "valence": component.valence,
        "max_automorphisms": component.max_automorphisms,
        "max_search_nodes": component.max_search_nodes,
        "component_id": component.component_id,
    }


def _pool_dict(
    pool: ConformerPool, index: int, arrays: dict[str, np.ndarray], /
) -> dict[str, object]:
    name = f"pool-{index:04d}/admitted-positions"
    arrays[name] = np.stack(
        [
            _host_float64(item.admitted_positions, "admitted conformer geometry")
            for item in pool.conformers
        ]
    )
    return {
        "component": _component_dict(pool.component),
        "admitted_positions": name,
        "conformers": [
            {
                "provenance": item.provenance,
                "source": item.source,
                "minimum_distance": item.minimum_distance,
                "stereo_volume": item.stereo_volume,
                "double_bond_cosine": item.double_bond_cosine,
                "conformer_id": item.conformer_id,
            }
            for item in pool.conformers
        ],
        "declared_weights": list(pool.declared_weights),
        "pool_id": pool.pool_id,
    }


def _input_dict(
    molecular: MolecularCrystalInput, arrays: dict[str, np.ndarray], /
) -> dict[str, object]:
    policy = molecular.contact_policy
    if policy is None:
        arrays[_CONTACT_ARRAY] = _host_float64(
            molecular.contact_distances, "declared contact matrix"
        )
        contact: dict[str, object] = {"kind": "matrix", "array": _CONTACT_ARRAY}
    else:
        contact = {
            "kind": "policy",
            "radii": _table_dict(policy.radii),
            "rule": policy.rule,
            "policy_id": policy.policy_id,
        }
    volume = molecular.volume_policy
    return {
        "pools": [
            _pool_dict(pool, index, arrays) for index, pool in enumerate(molecular.pools)
        ],
        "formula_counts": list(molecular.formula_counts),
        "contact": contact,
        "volume": {
            "formula_unit_volume": volume.formula_unit_volume,
            "atomic_volumes": None
            if volume.atomic_volumes is None
            else _table_dict(volume.atomic_volumes),
            "policy_id": volume.policy_id,
        },
        "input_id": molecular.input_id,
    }


def _setting_dict(setting: SpaceGroupSetting, /) -> dict[str, object]:
    transformation = setting.transformation
    return {
        "reference_hall_number": setting.reference_hall_number,
        "matrix": [str(value) for value in transformation.matrix],
        "origin_shift": [str(value) for value in transformation.origin_shift],
        "transformation_id": transformation.transformation_id,
        "operation_count": setting.operation_count,
        "setting_id": setting.setting_id,
    }


def _cell_dict(cell: CrystalCellPolicy, /) -> dict[str, object]:
    return {
        "volume_range": list(cell.volume_range),
        "scaled": cell.scaled,
        "oversample": cell.oversample,
        "maximum_axis_ratio": cell.maximum_axis_ratio,
        "angle_range": list(cell.angle_range),
        "minimum_height": cell.minimum_height,
        "minimum_height_floor": cell.minimum_height_floor,
        "maximum_condition": cell.maximum_condition,
        "shape_draws": cell.shape_draws,
        "policy_id": cell.policy_id,
    }


def _relief_dict(relief: CrystalReliefPolicy, /) -> dict[str, object]:
    return {
        "maximum_steps": relief.maximum_steps,
        "check_interval": relief.check_interval,
        "overlap_tolerance": relief.overlap_tolerance,
        "volume_tolerance": relief.volume_tolerance,
        "maximum_step": relief.maximum_step,
        "step_scale": relief.step_scale,
        "cell_step_scale": relief.cell_step_scale,
        "maximum_cell_strain": relief.maximum_cell_strain,
        "volume_compression": relief.volume_compression,
        "maximum_rotation": relief.maximum_rotation,
        "policy_id": relief.policy_id,
    }


def _resources_dict(resources: CrystalContactResources, /) -> dict[str, object]:
    return {
        "search": _search_dict(resources.search),
        "search_capacities": [
            _image_capacity_dict(item) for item in resources.search_capacities
        ],
        "role_capacities": [
            _role_capacity_dict(item) for item in resources.role_capacities
        ],
        "skin": resources.skin,
        "maximum_image_count": resources.maximum_image_count,
        "resources_id": resources.resources_id,
    }


def _encoded(
    recipe: CrystalPackingRecipe, /
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Canonical JSON recipe (lists only) and its float64 arrays."""
    plan = recipe.plan
    groups = plan.groups
    arrays: dict[str, np.ndarray] = {}
    body: dict[str, object] = {
        "kind": _RECIPE_KIND,
        "plan": {
            "input": _input_dict(plan.molecular_input, arrays),
            "groups": {
                "settings": [_setting_dict(item) for item in groups.settings],
                "declared_weights": list(groups.declared_weights),
                "provenance": groups.provenance,
                "policy_id": groups.policy_id,
            },
            "z_prime": plan.z_prime,
            "chirality": plan.chirality,
            "maximum_atoms": plan.maximum_atoms,
            "cell": _cell_dict(plan.cell),
            "relief": _relief_dict(plan.relief),
            "resources": _resources_dict(plan.resources),
            "law_id": plan.law_id,
            "plan_id": plan.plan_id,
        },
        "generation": {
            "campaign_id": recipe.campaign_id,
            "lanes": recipe.lanes,
            "pending_capacity": recipe.pending_capacity,
            "steps_per_transition": recipe.steps_per_transition,
            "stream_identity": dict(recipe.stream_identity),
        },
        "recipe_id": recipe.recipe_id,
    }
    return body, arrays


def _content_id(body: Mapping[str, object], array_digest: str, /) -> str:
    return canonical_fingerprint(
        {"kind": _ARCHIVE_KIND, "recipe": body, "array_digest": array_digest}
    )


# -- decoding -------------------------------------------------------------------------------------


_TABLE_FIELDS = frozenset({"quantity", "entries", "provenance", "table_id"})


def _table_from(value: object, label: str, /) -> ElementTable:
    record = _recipe_record(value, _TABLE_FIELDS, label)
    entries: dict[int, float] = {}
    for item in _recipe_list(record["entries"]):
        pair = _recipe_list(item)
        if len(pair) != 2:
            raise ValueError(f"The {label} entries are not (number, value) pairs.")
        number = _integer(pair[0], f"{label} atomic number")
        if number in entries:
            raise ValueError(f"The {label} declares atomic number {number} twice.")
        entries[number] = _real(pair[1], f"{label} value")
    table = ElementTable(
        parse(record["quantity"], ElementQuantity, f"{label} quantity"),
        entries,
        provenance=_text(record["provenance"], f"{label} provenance"),
    )
    _verified(record["table_id"], table.table_id, label)
    return table


_COMPONENT_FIELDS = frozenset(
    {
        "name",
        "atomic_numbers",
        "formal_charges",
        "bonds",
        "bond_orders",
        "tetrahedral_stereo",
        "double_bond_stereo",
        "undeclared_stereo",
        "spin_multiplicity",
        "masses",
        "valence",
        "max_automorphisms",
        "max_search_nodes",
        "component_id",
    }
)


def _component_from(value: object, /) -> MolecularComponent:
    record = _recipe_record(value, _COMPONENT_FIELDS, "molecular component")
    masses_record = record["masses"]
    if not isinstance(masses_record, Mapping):
        raise ValueError("The molecular component masses must be a record.")
    masses: ElementTable | list[float]
    match masses_record.get("kind"):
        case "table":
            table_record = _recipe_record(masses_record, {"kind", "table"}, "mass source")
            masses = _table_from(table_record["table"], "mass table")
        case "per-atom":
            per_atom = _recipe_record(masses_record, {"kind", "values"}, "mass source")
            masses = _reals(per_atom["values"], "per-atom mass")
        case _:
            raise ValueError("The molecular component mass source kind is unknown.")
    component = MolecularComponent(
        _integers(record["atomic_numbers"], "atomic number"),
        _rows(record["bonds"], "bond"),
        bond_orders=_integers(record["bond_orders"], "bond order"),
        masses=masses,
        spin_multiplicity=_integer(record["spin_multiplicity"], "spin multiplicity"),
        formal_charges=_integers(record["formal_charges"], "formal charge"),
        tetrahedral_stereo=_rows(record["tetrahedral_stereo"], "tetrahedral stereo"),
        double_bond_stereo=_rows(record["double_bond_stereo"], "double-bond stereo"),
        undeclared_stereo=parse(
            record["undeclared_stereo"], UndeclaredStereoPolicy, "undeclared stereo"
        ),
        valence=parse(record["valence"], ValencePolicy, "valence policy"),
        name=_text(record["name"], "component name"),
        max_automorphisms=_integer(record["max_automorphisms"], "automorphism bound"),
        max_search_nodes=_integer(record["max_search_nodes"], "search-node bound"),
    )
    _verified(record["component_id"], component.component_id, "molecular component")
    return component


_POOL_FIELDS = frozenset(
    {"component", "admitted_positions", "conformers", "declared_weights", "pool_id"}
)
_CONFORMER_FIELDS = frozenset(
    {
        "provenance",
        "source",
        "minimum_distance",
        "stereo_volume",
        "double_bond_cosine",
        "conformer_id",
    }
)


def _pool_from(value: object, arrays: _Arrays, /) -> ConformerPool:
    record = _recipe_record(value, _POOL_FIELDS, "conformer pool")
    component = _component_from(record["component"])
    entries = [
        _recipe_record(item, _CONFORMER_FIELDS, "conformer")
        for item in _recipe_list(record["conformers"])
    ]
    geometry = arrays.take(
        record["admitted_positions"], (len(entries), component.atom_count, 3)
    )
    conformers = []
    for entry, positions in zip(entries, geometry, strict=True):
        conformer = MolecularConformer(
            component,
            positions,
            provenance=_text(entry["provenance"], "conformer provenance"),
            source=parse(entry["source"], ConformerSource, "conformer source"),
            minimum_distance=_real(entry["minimum_distance"], "minimum distance"),
            stereo_volume=_real(entry["stereo_volume"], "stereo volume"),
            double_bond_cosine=_real(entry["double_bond_cosine"], "double-bond cosine"),
        )
        _verified(entry["conformer_id"], conformer.conformer_id, "conformer")
        conformers.append(conformer)
    pool = ConformerPool(
        component,
        conformers,
        weights=_reals(record["declared_weights"], "conformer weight"),
    )
    _verified(record["pool_id"], pool.pool_id, "conformer pool")
    return pool


_INPUT_FIELDS = frozenset({"pools", "formula_counts", "contact", "volume", "input_id"})


def _input_from(value: object, arrays: _Arrays, /) -> MolecularCrystalInput:
    record = _recipe_record(value, _INPUT_FIELDS, "molecular input")
    pools = [_pool_from(item, arrays) for item in _recipe_list(record["pools"])]
    counts = _integers(record["formula_counts"], "formula count")
    contact_record = record["contact"]
    if not isinstance(contact_record, Mapping):
        raise ValueError("The molecular input contact must be a record.")
    contact: ContactPolicy | np.ndarray
    match contact_record.get("kind"):
        case "policy":
            fields = _recipe_record(
                contact_record, {"kind", "radii", "rule", "policy_id"}, "contact policy"
            )
            policy = ContactPolicy(
                _table_from(fields["radii"], "contact radius table"),
                rule=parse(fields["rule"], ContactRule, "contact rule"),
            )
            _verified(fields["policy_id"], policy.policy_id, "contact policy")
            contact = policy
        case "matrix":
            fields = _recipe_record(contact_record, {"kind", "array"}, "contact matrix")
            atoms = sum(
                pool.component.atom_count * count
                for pool, count in zip(pools, counts, strict=True)
            )
            contact = arrays.take(fields["array"], (atoms, atoms))
        case _:
            raise ValueError("The molecular input contact kind is unknown.")
    volume_record = _recipe_record(
        record["volume"],
        {"formula_unit_volume", "atomic_volumes", "policy_id"},
        "volume policy",
    )
    explicit = volume_record["formula_unit_volume"]
    tabulated = volume_record["atomic_volumes"]
    volume = VolumePolicy(
        formula_unit_volume=None
        if explicit is None
        else _real(explicit, "formula-unit volume"),
        atomic_volumes=None
        if tabulated is None
        else _table_from(tabulated, "atomic volume table"),
    )
    _verified(volume_record["policy_id"], volume.policy_id, "volume policy")
    molecular = MolecularCrystalInput(
        pools, contact=contact, volume=volume, formula_counts=counts
    )
    _verified(record["input_id"], molecular.input_id, "molecular input")
    return molecular


_SETTING_FIELDS = frozenset(
    {
        "reference_hall_number",
        "matrix",
        "origin_shift",
        "transformation_id",
        "operation_count",
        "setting_id",
    }
)


def _setting_from(value: object, /) -> SpaceGroupSetting:
    record = _recipe_record(value, _SETTING_FIELDS, "space-group setting")
    matrix = [_fraction(item) for item in _recipe_list(record["matrix"])]
    shift = [_fraction(item) for item in _recipe_list(record["origin_shift"])]
    if len(matrix) != 9 or len(shift) != 3:
        raise ValueError("A setting transformation has a 3x3 matrix and 3 shifts.")
    transformation = SettingTransformation([matrix[0:3], matrix[3:6], matrix[6:9]], shift)
    _verified(
        record["transformation_id"],
        transformation.transformation_id,
        "setting transformation",
    )
    setting = SpaceGroupSetting(
        _integer(record["reference_hall_number"], "reference Hall number"),
        transformation=transformation,
        maximum_operation_count=_integer(record["operation_count"], "operation count"),
    )
    _verified(record["setting_id"], setting.setting_id, "space-group setting")
    return setting


def _groups_from(value: object, /) -> CrystalSpaceGroupPolicy:
    record = _recipe_record(
        value,
        {"settings", "declared_weights", "provenance", "policy_id"},
        "space-group policy",
    )
    groups = CrystalSpaceGroupPolicy(
        [_setting_from(item) for item in _recipe_list(record["settings"])],
        weights=_reals(record["declared_weights"], "setting weight"),
        provenance=_text(record["provenance"], "space-group prior provenance"),
    )
    _verified(record["policy_id"], groups.policy_id, "space-group policy")
    return groups


def _pair(value: object, label: str, /) -> tuple[float, float]:
    values = _reals(value, label)
    if len(values) != 2:
        raise ValueError(f"The crystal recipe {label} must hold two numbers.")
    return values[0], values[1]


_CELL_FIELDS = frozenset(
    {
        "volume_range",
        "scaled",
        "oversample",
        "maximum_axis_ratio",
        "angle_range",
        "minimum_height",
        "minimum_height_floor",
        "maximum_condition",
        "shape_draws",
        "policy_id",
    }
)


def _cell_from(value: object, /) -> CrystalCellPolicy:
    record = _recipe_record(value, _CELL_FIELDS, "cell policy")
    cell = CrystalCellPolicy(
        volume_range=_pair(record["volume_range"], "volume range"),
        scaled=_flag(record["scaled"], "volume scaling"),
        oversample=_real(record["oversample"], "oversample"),
        maximum_axis_ratio=_real(record["maximum_axis_ratio"], "axis ratio"),
        angle_range=_pair(record["angle_range"], "angle range"),
        minimum_height=_real(record["minimum_height"], "minimum height"),
        minimum_height_floor=_real(record["minimum_height_floor"], "height floor"),
        maximum_condition=_real(record["maximum_condition"], "maximum condition"),
        shape_draws=_integer(record["shape_draws"], "shape draws"),
    )
    _verified(record["policy_id"], cell.policy_id, "cell policy")
    return cell


_RELIEF_FIELDS = frozenset(
    {
        "maximum_steps",
        "check_interval",
        "overlap_tolerance",
        "volume_tolerance",
        "maximum_step",
        "step_scale",
        "cell_step_scale",
        "maximum_cell_strain",
        "volume_compression",
        "maximum_rotation",
        "policy_id",
    }
)


def _relief_from(value: object, /) -> CrystalReliefPolicy:
    record = _recipe_record(value, _RELIEF_FIELDS, "relief policy")
    relief = CrystalReliefPolicy(
        maximum_steps=_integer(record["maximum_steps"], "maximum steps"),
        check_interval=_integer(record["check_interval"], "check interval"),
        overlap_tolerance=_real(record["overlap_tolerance"], "overlap tolerance"),
        volume_tolerance=_real(record["volume_tolerance"], "volume tolerance"),
        maximum_step=_real(record["maximum_step"], "maximum step"),
        step_scale=_real(record["step_scale"], "step scale"),
        cell_step_scale=_real(record["cell_step_scale"], "cell step scale"),
        maximum_cell_strain=_real(record["maximum_cell_strain"], "cell strain"),
        volume_compression=_real(record["volume_compression"], "volume compression"),
        maximum_rotation=_real(record["maximum_rotation"], "maximum rotation"),
    )
    _verified(record["policy_id"], relief.policy_id, "relief policy")
    return relief


_RESOURCE_FIELDS = frozenset(
    {
        "search",
        "search_capacities",
        "role_capacities",
        "skin",
        "maximum_image_count",
        "resources_id",
    }
)


def _resources_from(value: object, /) -> CrystalContactResources:
    record = _recipe_record(value, _RESOURCE_FIELDS, "contact resources")
    resources = CrystalContactResources(
        _search_from(record["search"]),
        search_capacities=[
            _image_capacity_from(item)
            for item in _recipe_list(record["search_capacities"])
        ],
        role_capacities=[
            _role_capacity_from(item) for item in _recipe_list(record["role_capacities"])
        ],
        skin=_real(record["skin"], "skin"),
        maximum_image_count=_integer(record["maximum_image_count"], "image bound"),
    )
    _verified(record["resources_id"], resources.resources_id, "contact resources")
    return resources


_PLAN_FIELDS = frozenset(
    {
        "input",
        "groups",
        "z_prime",
        "chirality",
        "maximum_atoms",
        "cell",
        "relief",
        "resources",
        "law_id",
        "plan_id",
    }
)


def _plan_from(value: object, arrays: _Arrays, /) -> CrystalPackingPlan:
    record = _recipe_record(value, _PLAN_FIELDS, "packing plan")
    plan = CrystalPackingPlan(
        _input_from(record["input"], arrays),
        _groups_from(record["groups"]),
        z_prime=_integer(record["z_prime"], "Z'"),
        chirality=parse(record["chirality"], ChiralityPolicy, "chirality policy"),
        resources=_resources_from(record["resources"]),
        cell=_cell_from(record["cell"]),
        relief=_relief_from(record["relief"]),
        maximum_atoms=_integer(record["maximum_atoms"], "atom bound"),
    )
    _verified(record["law_id"], plan.law_id, "packing law")
    _verified(record["plan_id"], plan.plan_id, "packing plan")
    return plan


_GENERATION_FIELDS = frozenset(
    {
        "campaign_id",
        "lanes",
        "pending_capacity",
        "steps_per_transition",
        "stream_identity",
    }
)


def _recipe_from(
    body: object, arrays: Mapping[str, np.ndarray], /
) -> CrystalPackingRecipe:
    record = _recipe_record(body, {"kind", "plan", "generation", "recipe_id"}, "recipe")
    if record["kind"] != _RECIPE_KIND:
        raise ValueError("The archive does not hold a crystal packing recipe.")
    consumed = _Arrays(arrays)
    plan = _plan_from(record["plan"], consumed)
    consumed.require_consumed()
    generation = _recipe_record(
        record["generation"], _GENERATION_FIELDS, "generation configuration"
    )
    identity = _recipe_record(
        generation["stream_identity"],
        {"stream", "samplers", "revisions", "executables"},
        "stream identity",
    )
    recipe = CrystalPackingRecipe(
        plan.prepare(lanes=_integer(generation["lanes"], "lane count")),
        campaign_id=_text(generation["campaign_id"], "campaign"),
        pending_capacity=_integer(generation["pending_capacity"], "pending capacity"),
        steps_per_transition=_integer(
            generation["steps_per_transition"], "steps per transition"
        ),
    )
    if {key: _text(item, "stream identity") for key, item in identity.items()} != dict(
        recipe.stream_identity
    ):
        raise ValueError(
            "The rebuilt packer does not reproduce the recorded sampler and stream "
            "identities."
        )
    _verified(record["recipe_id"], recipe.recipe_id, "packing recipe")
    return recipe


# -- public archive API ---------------------------------------------------------------------------


def write_crystal_packing_recipe(
    path: str | os.PathLike[str],
    recipe: CrystalPackingRecipe,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> CrystalPackingRecipeReceipt:
    """Atomically publish ``recipe`` as one bounded pickle-free array archive."""
    if not isinstance(recipe, CrystalPackingRecipe):
        raise TypeError("recipe must be a CrystalPackingRecipe.")
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be ArrayArchiveLimits.")
    body, arrays = _encoded(recipe)
    digest = array_collection_digest(arrays)
    content = _content_id(body, digest)
    destination = write_array_archive(
        path,
        manifest={
            "kind": _ARCHIVE_KIND,
            "recipe": body,
            "array_digest": digest,
            "content_id": content,
        },
        arrays=arrays,
        limits=limits,
    )
    return CrystalPackingRecipeReceipt(
        destination,
        recipe.recipe_id,
        recipe.plan.law_id,
        recipe.plan.plan_id,
        content,
        digest,
    )


def read_crystal_packing_recipe(
    path: str | os.PathLike[str],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[CrystalPackingRecipe, CrystalPackingRecipeReceipt]:
    """Admit, rebuild and verify one recipe archive; any deviation refuses."""
    if not isinstance(limits, ArrayArchiveLimits):
        raise TypeError("limits must be ArrayArchiveLimits.")
    manifest, arrays = read_array_archive(path, limits=limits)
    if set(manifest) != _ARCHIVE_FIELDS or manifest["kind"] != _ARCHIVE_KIND:
        raise ValueError("The archive is not a crystal packing recipe archive.")
    digest = array_collection_digest(arrays)
    if manifest["array_digest"] != digest:
        raise ValueError("The crystal recipe array collection digest is invalid.")
    body = manifest["recipe"]
    if not isinstance(body, Mapping):
        raise ValueError("The crystal recipe body must be a record.")
    content = _content_id(body, digest)
    if manifest["content_id"] != content:
        raise ValueError("The crystal recipe content address is invalid.")
    recipe = _recipe_from(body, arrays)
    rebuilt, rebuilt_arrays = _encoded(recipe)
    if rebuilt != body or array_collection_digest(rebuilt_arrays) != digest:
        raise ValueError(
            "The rebuilt crystal recipe does not reproduce the stored recipe exactly."
        )
    return recipe, CrystalPackingRecipeReceipt(
        Path(os.fspath(path)),
        recipe.recipe_id,
        recipe.plan.law_id,
        recipe.plan.plan_id,
        content,
        digest,
    )


__all__ = [
    "CrystalPackingRecipe",
    "CrystalPackingRecipeReceipt",
    "read_crystal_packing_recipe",
    "write_crystal_packing_recipe",
]
