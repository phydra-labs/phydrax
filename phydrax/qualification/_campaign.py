#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


CampaignRoleName = Literal[
    "calibration",
    "model_selection",
    "interval_calibration",
    "locked_evaluation",
    "prospective",
]
_ROLE_NAMES: tuple[CampaignRoleName, ...] = (
    "calibration",
    "model_selection",
    "interval_calibration",
    "locked_evaluation",
    "prospective",
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _identifiers(
    values: Sequence[str],
    name: str,
    /,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of identifiers.")
    normalized = tuple(_identifier(value, name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique identifiers.")
    return tuple(sorted(normalized))


@dataclass(frozen=True, slots=True)
class ScientificCase:
    """One immutable scientific case with explicit grouping and ancestry."""

    case_id: str
    independent_unit_id: str
    construct_id: str
    condition_id: str
    preparation_id: str
    batch_id: str
    source_manifest_ids: tuple[str, ...]
    parent_case_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(
            self,
            "independent_unit_id",
            _identifier(self.independent_unit_id, "independent_unit_id"),
        )
        object.__setattr__(
            self, "construct_id", _identifier(self.construct_id, "construct_id")
        )
        object.__setattr__(
            self, "condition_id", _identifier(self.condition_id, "condition_id")
        )
        object.__setattr__(
            self, "preparation_id", _identifier(self.preparation_id, "preparation_id")
        )
        object.__setattr__(self, "batch_id", _identifier(self.batch_id, "batch_id"))
        object.__setattr__(
            self,
            "source_manifest_ids",
            _identifiers(self.source_manifest_ids, "source_manifest_ids"),
        )
        parents = _identifiers(self.parent_case_ids, "parent_case_ids", allow_empty=True)
        if self.case_id in parents:
            raise ValueError("A scientific case cannot be its own parent.")
        object.__setattr__(self, "parent_case_ids", parents)

    def to_record(self) -> dict[str, object]:
        """Return the deterministic JSON-ready case record."""
        return {
            "case_id": self.case_id,
            "independent_unit_id": self.independent_unit_id,
            "construct_id": self.construct_id,
            "condition_id": self.condition_id,
            "preparation_id": self.preparation_id,
            "batch_id": self.batch_id,
            "source_manifest_ids": list(self.source_manifest_ids),
            "parent_case_ids": list(self.parent_case_ids),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ScientificCase:
        """Reconstruct a scientific case from its serialized record."""
        if not isinstance(record, Mapping):
            raise TypeError("Scientific-case record must be a mapping.")
        source_manifest_ids = record["source_manifest_ids"]
        parent_case_ids = record["parent_case_ids"]
        if not isinstance(source_manifest_ids, Sequence) or isinstance(
            source_manifest_ids, str
        ):
            raise TypeError("Serialized source_manifest_ids must be a sequence.")
        if not isinstance(parent_case_ids, Sequence) or isinstance(parent_case_ids, str):
            raise TypeError("Serialized parent_case_ids must be a sequence.")
        return cls(
            str(record["case_id"]),
            str(record["independent_unit_id"]),
            str(record["construct_id"]),
            str(record["condition_id"]),
            str(record["preparation_id"]),
            str(record["batch_id"]),
            tuple(str(value) for value in source_manifest_ids),
            tuple(str(value) for value in parent_case_ids),
        )


@dataclass(frozen=True, slots=True)
class CampaignRole:
    """The exact case membership of one fixed campaign role."""

    name: CampaignRoleName
    case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        name = _identifier(self.name, "campaign role name")
        if name not in _ROLE_NAMES:
            raise ValueError(
                "Campaign role must be calibration, model_selection, "
                "interval_calibration, locked_evaluation, or prospective."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "case_ids",
            _identifiers(self.case_ids, "campaign role case_ids", allow_empty=True),
        )

    def to_record(self) -> dict[str, object]:
        """Return the deterministic JSON-ready role record."""
        return {"name": self.name, "case_ids": list(self.case_ids)}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> CampaignRole:
        """Reconstruct a campaign role from its serialized record."""
        if not isinstance(record, Mapping):
            raise TypeError("Campaign-role record must be a mapping.")
        case_ids = record["case_ids"]
        if not isinstance(case_ids, Sequence) or isinstance(case_ids, str):
            raise TypeError("Serialized campaign role case_ids must be a sequence.")
        return cls(
            str(record["name"]),
            tuple(str(value) for value in case_ids),
        )


class ScientificCampaign(StrictModule, NonTrainableState):
    """Content-addressed, leakage-controlled scientific campaign membership."""

    cases: tuple[ScientificCase, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    independent_unit_ids: tuple[str, ...] = eqx.field(static=True)
    roles: tuple[CampaignRole, ...] = eqx.field(static=True)
    preprocessing_source_ids: tuple[str, ...] = eqx.field(static=True)
    criteria_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        cases: Sequence[ScientificCase],
        roles: Sequence[CampaignRole],
        /,
        *,
        preprocessing_source_ids: Sequence[str] = (),
        criteria_ids: Sequence[str] = (),
    ):
        if not isinstance(cases, Sequence) or isinstance(cases, str) or not cases:
            raise TypeError("cases must be a non-empty sequence of ScientificCase.")
        if any(not isinstance(case, ScientificCase) for case in cases):
            raise TypeError("cases must contain only ScientificCase values.")
        if not isinstance(roles, Sequence) or isinstance(roles, str) or not roles:
            raise TypeError("roles must be a non-empty sequence of CampaignRole.")
        if any(not isinstance(role, CampaignRole) for role in roles):
            raise TypeError("roles must contain only CampaignRole values.")

        cases_ = tuple(sorted(cases, key=lambda case: case.case_id))
        case_ids = tuple(case.case_id for case in cases_)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Scientific case IDs must be unique within a campaign.")

        provided_roles = tuple(sorted(roles, key=lambda role: role.name))
        role_names = tuple(role.name for role in provided_roles)
        if len(set(role_names)) != len(role_names):
            raise ValueError("Campaign role names must be unique.")
        role_by_name = {role.name: role for role in provided_roles}
        roles_ = tuple(
            role_by_name[name] if name in role_by_name else CampaignRole(name, ())
            for name in _ROLE_NAMES
        )
        membership = tuple(case_id for role in roles_ for case_id in role.case_ids)
        unknown = set(membership) - set(case_ids)
        if unknown:
            raise ValueError(
                "Campaign roles reference unknown case IDs: " + ", ".join(sorted(unknown))
            )
        if len(set(membership)) != len(membership):
            raise ValueError("A scientific case must not appear in more than one role.")
        missing = set(case_ids) - set(membership)
        if missing:
            raise ValueError(
                "Every scientific case must appear in exactly one role; missing: "
                + ", ".join(sorted(missing))
            )

        role_cases = {role.name: role.case_ids for role in roles_}
        if not role_cases.get("calibration"):
            raise ValueError("A qualification campaign requires calibration cases.")
        if not role_cases.get("locked_evaluation"):
            raise ValueError("A qualification campaign requires locked_evaluation cases.")
        role_by_case = {
            case_id: role.name for role in roles_ for case_id in role.case_ids
        }

        roles_by_coordinate: dict[str, dict[str, set[str]]] = {
            "independent_unit_id": {},
            "preparation_id": {},
            "batch_id": {},
        }
        for case in cases_:
            role = role_by_case[case.case_id]
            coordinates = (
                ("independent_unit_id", case.independent_unit_id),
                ("preparation_id", case.preparation_id),
                ("batch_id", case.batch_id),
            )
            for coordinate_name, coordinate_id in coordinates:
                roles_by_coordinate[coordinate_name].setdefault(coordinate_id, set()).add(
                    role
                )
        for coordinate_name, assigned_roles in roles_by_coordinate.items():
            crossing_ids = sorted(
                coordinate_id
                for coordinate_id, assigned in assigned_roles.items()
                if len(assigned) > 1
            )
            if crossing_ids:
                raise ValueError(
                    f"Cases sharing a {coordinate_name} cannot cross campaign roles: "
                    + ", ".join(crossing_ids)
                )

        cases_by_id = {case.case_id: case for case in cases_}
        for case in cases_:
            unknown_parents = set(case.parent_case_ids) - set(case_ids)
            if unknown_parents:
                raise ValueError(
                    f"Scientific case {case.case_id!r} has parents outside the "
                    "campaign: " + ", ".join(sorted(unknown_parents))
                )
            ancestors: set[str] = set()
            frontier = list(case.parent_case_ids)
            while frontier:
                parent_id = frontier.pop()
                if parent_id == case.case_id:
                    raise ValueError("Scientific case ancestry must be acyclic.")
                if parent_id in ancestors:
                    continue
                ancestors.add(parent_id)
                frontier.extend(cases_by_id[parent_id].parent_case_ids)
            crossing_ancestors = sorted(
                ancestor_id
                for ancestor_id in ancestors
                if role_by_case[ancestor_id] != role_by_case[case.case_id]
            )
            if crossing_ancestors:
                raise ValueError(
                    "The transitive closure of parent_case_ids cannot cross "
                    f"campaign roles for {case.case_id!r}: "
                    + ", ".join(crossing_ancestors)
                )

        preprocessing = _identifiers(
            preprocessing_source_ids,
            "preprocessing_source_ids",
            allow_empty=True,
        )
        unknown_preprocessing = set(preprocessing) - set(case_ids)
        if unknown_preprocessing:
            raise ValueError(
                "Preprocessing source IDs must reference campaign cases: "
                + ", ".join(sorted(unknown_preprocessing))
            )
        permitted_preprocessing = set(role_cases.get("calibration", ())) | set(
            role_cases.get("model_selection", ())
        )
        forbidden_preprocessing = set(preprocessing) - permitted_preprocessing
        if forbidden_preprocessing:
            raise ValueError(
                "Preprocessing sources must be calibration or model_selection cases: "
                + ", ".join(sorted(forbidden_preprocessing))
            )

        criteria = _identifiers(criteria_ids, "criteria_ids", allow_empty=True)
        self.cases = cases_
        self.case_ids = case_ids
        self.independent_unit_ids = tuple(
            sorted(roles_by_coordinate["independent_unit_id"])
        )
        self.roles = roles_
        self.preprocessing_source_ids = preprocessing
        self.criteria_ids = criteria
        self.campaign_id = canonical_fingerprint(self._content_record())

    def _content_record(self) -> dict[str, object]:
        return {
            "kind": "scientific-campaign",
            "cases": [case.to_record() for case in self.cases],
            "roles": [role.to_record() for role in self.roles],
            "preprocessing_source_ids": list(self.preprocessing_source_ids),
            "criteria_ids": list(self.criteria_ids),
        }

    def to_record(self) -> dict[str, object]:
        """Return a deterministic JSON-ready record with its content address."""
        return {**self._content_record(), "campaign_id": self.campaign_id}

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> ScientificCampaign:
        """Reconstruct and content-verify a serialized scientific campaign."""
        if not isinstance(record, Mapping):
            raise TypeError("Scientific-campaign record must be a mapping.")
        cases = record["cases"]
        roles = record["roles"]
        preprocessing = record["preprocessing_source_ids"]
        criteria = record["criteria_ids"]
        sequence_fields = {
            "cases": cases,
            "roles": roles,
            "preprocessing_source_ids": preprocessing,
            "criteria_ids": criteria,
        }
        if any(
            not isinstance(values, Sequence) or isinstance(values, str)
            for values in sequence_fields.values()
        ):
            raise TypeError("Serialized campaign collections must be sequences.")
        value = cls(
            tuple(ScientificCase.from_record(case) for case in cases),
            tuple(CampaignRole.from_record(role) for role in roles),
            preprocessing_source_ids=tuple(str(item) for item in preprocessing),
            criteria_ids=tuple(str(item) for item in criteria),
        )
        recorded_id = record.get("campaign_id")
        if recorded_id is not None and str(recorded_id) != value.campaign_id:
            raise ValueError(
                "Serialized scientific campaign has an invalid content address."
            )
        return value


__all__ = ["CampaignRole", "CampaignRoleName", "ScientificCampaign", "ScientificCase"]
