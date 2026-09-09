#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from datetime import date
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any

import equinox as eqx

from ..._fingerprint import canonical_fingerprint, canonical_json
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_FPML_NAMESPACE = "http://www.fpml.org/FpML-5/confirmation"
_MAX_FPML_BYTES = 1_048_576
_MAX_FPML_DEPTH = 16
_MAX_FPML_NODES = 128
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_HEX_DIGITS = frozenset("0123456789abcdef")
_ALLOWED_TAGS = frozenset(
    {
        "dataDocument",
        "trade",
        "tradeHeader",
        "partyTradeIdentifier",
        "tradeId",
        "tradeDate",
        "fxSingleLeg",
        "exchangedCurrency1",
        "exchangedCurrency2",
        "payerPartyReference",
        "receiverPartyReference",
        "paymentAmount",
        "currency",
        "amount",
        "valueDate",
    }
)
_ALLOWED_ATTRIBUTES = frozenset({"href"})


def _local_name(value: str, /) -> str:
    return value.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str, /) -> tuple[ET.Element, ...]:
    return tuple(child for child in element if _local_name(child.tag) == name)


def _child(element: ET.Element, name: str, path: str, /) -> ET.Element:
    matches = _children(element, name)
    if len(matches) != 1:
        raise ValueError(f"missing-or-duplicate:{path}/{name}")
    return matches[0]


def _exact_children(
    element: ET.Element,
    names: tuple[str, ...],
    path: str,
    /,
) -> None:
    observed = tuple(_local_name(child.tag) for child in element)
    if observed != names:
        raise ValueError(f"unsupported-structure:{path}")


def _text(element: ET.Element, path: str, /) -> str:
    if tuple(element):
        raise ValueError(f"nested-text:{path}")
    value = "" if element.text is None else element.text.strip()
    if not value:
        raise ValueError(f"missing-text:{path}")
    return value


def _reference(element: ET.Element, path: str, /) -> str:
    href = element.attrib.get("href", "")
    if not href or href != href.strip():
        raise ValueError(f"missing-reference:{path}")
    return href


def _iso_date(value: str, path: str, /) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid-date:{path}") from error
    return parsed.isoformat()


def _amount(value: str, path: str, /) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"invalid-amount:{path}") from error
    if not number.is_finite() or number <= 0:
        raise ValueError(f"invalid-amount:{path}")
    return format(number, "f")


def _currency(value: str, path: str, /) -> str:
    if _CURRENCY.fullmatch(value) is None:
        raise ValueError(f"invalid-currency:{path}")
    return value


def _payment(element: ET.Element, path: str, /) -> dict[str, str]:
    _exact_children(
        element,
        ("payerPartyReference", "receiverPartyReference", "paymentAmount"),
        path,
    )
    payment = _child(element, "paymentAmount", path)
    _exact_children(payment, ("currency", "amount"), f"{path}/paymentAmount")
    return {
        "payer_party_id": _reference(
            _child(element, "payerPartyReference", path),
            f"{path}/payerPartyReference",
        ),
        "receiver_party_id": _reference(
            _child(element, "receiverPartyReference", path),
            f"{path}/receiverPartyReference",
        ),
        "currency": _currency(
            _text(
                _child(payment, "currency", f"{path}/paymentAmount"),
                "currency",
            ),
            f"{path}/paymentAmount/currency",
        ),
        "amount": _amount(
            _text(
                _child(payment, "amount", f"{path}/paymentAmount"),
                "amount",
            ),
            f"{path}/paymentAmount/amount",
        ),
    }


def _unknown_terms(root: ET.Element, /) -> tuple[str, ...]:
    terms: set[str] = set()
    pending = [(root, "", 1)]
    observed_nodes = 0
    while pending:
        element, path, depth = pending.pop()
        observed_nodes += 1
        if observed_nodes > _MAX_FPML_NODES:
            terms.add("resource-limit:maximum-nodes")
            break
        name = _local_name(element.tag)
        current = f"{path}/{name}" if path else name
        if depth > _MAX_FPML_DEPTH:
            terms.add("resource-limit:maximum-depth")
            continue
        if not element.tag.startswith(f"{{{_FPML_NAMESPACE}}}"):
            terms.add(f"unsupported-namespace:{current}")
        if name not in _ALLOWED_TAGS:
            terms.add(f"unsupported-element:{current}")
        for attribute in element.attrib:
            attribute_name = _local_name(attribute)
            if "}" in attribute or attribute_name not in _ALLOWED_ATTRIBUTES:
                terms.add(f"unsupported-attribute:{current}/@{attribute_name}")
        for child in reversed(tuple(element)):
            pending.append((child, current, depth + 1))
    return tuple(sorted(terms))


def _validate_payment_record(record: Mapping[str, Any], name: str, /) -> None:
    expected = {
        "payer_party_id",
        "receiver_party_id",
        "currency",
        "amount",
    }
    if set(record) != expected:
        raise ValueError(f"{name} fields are not the explicit supported subset.")
    _record_text(record, "payer_party_id")
    _record_text(record, "receiver_party_id")
    _currency(_record_text(record, "currency"), f"{name}/currency")
    _amount(_record_text(record, "amount"), f"{name}/amount")


def _validate_contract_record(record: Mapping[str, Any], /) -> None:
    expected = {
        "contract_type",
        "trade_id",
        "trade_date",
        "exchanged_currency_1",
        "exchanged_currency_2",
        "value_date",
    }
    if set(record) != expected or record["contract_type"] != "fx-forward":
        raise ValueError("FX-forward fields are not the explicit supported subset.")
    _record_text(record, "trade_id")
    _iso_date(_record_text(record, "trade_date"), "trade_date")
    _iso_date(_record_text(record, "value_date"), "value_date")
    for name in ("exchanged_currency_1", "exchanged_currency_2"):
        payment = record[name]
        if not isinstance(payment, Mapping):
            raise TypeError(f"{name} must be a mapping.")
        _validate_payment_record(payment, name)


class FpMLImportResult(StrictModule, NonTrainableState):
    """Accepted narrow FpML contract record or a structured fail-closed refusal."""

    accepted: bool = eqx.field(static=True)
    contract_type: str | None = eqx.field(static=True)
    contract_json: str | None = eqx.field(static=True)
    unsupported_terms: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        accepted: bool,
        contract_type: str | None,
        contract_record: Mapping[str, Any] | None,
        unsupported_terms: tuple[str, ...],
        source_id: str,
    ):
        if not isinstance(accepted, bool):
            raise TypeError("accepted must be boolean.")
        if (
            not isinstance(source_id, str)
            or len(source_id) != 64
            or any(character not in _HEX_DIGITS for character in source_id)
        ):
            raise ValueError("source_id must be a lowercase SHA-256 digest.")
        if not isinstance(unsupported_terms, tuple) or any(
            not isinstance(item, str) or not item for item in unsupported_terms
        ):
            raise TypeError("unsupported_terms must be a tuple of non-empty strings.")
        terms = tuple(sorted(set(unsupported_terms)))
        if accepted:
            if contract_type != "fx-forward" or not isinstance(contract_record, Mapping):
                raise ValueError(
                    "Accepted FpML results require one supported contract record."
                )
            if terms:
                raise ValueError("Accepted FpML results cannot contain refusals.")
            _validate_contract_record(contract_record)
            contract_json = canonical_json(dict(contract_record))
        else:
            if contract_type is not None or contract_record is not None or not terms:
                raise ValueError(
                    "Refused FpML results require terms and no contract record."
                )
            contract_json = None
        content = {
            "kind": "fpml-import-result",
            "accepted": accepted,
            "contract_type": contract_type,
            "contract": None if contract_json is None else json.loads(contract_json),
            "unsupported_terms": list(terms),
            "source_id": source_id,
        }
        self.accepted = accepted
        self.contract_type = contract_type
        self.contract_json = contract_json
        self.unsupported_terms = terms
        self.source_id = source_id
        self.result_id = canonical_fingerprint(content)

    def contract_record(self) -> Mapping[str, object]:
        """Return the accepted contract or raise with every refusal term."""
        if not self.accepted or self.contract_json is None:
            raise ValueError(
                "Unsupported FpML contract: " + "; ".join(self.unsupported_terms)
            )
        value = json.loads(self.contract_json, parse_constant=_reject_nonfinite)
        if not isinstance(value, dict):
            raise RuntimeError("FpML contract record invariant was violated.")
        return MappingProxyType(value)

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "fpml-import-result",
            "accepted": self.accepted,
            "contract_type": self.contract_type,
            "contract": (
                None if self.contract_json is None else json.loads(self.contract_json)
            ),
            "unsupported_terms": list(self.unsupported_terms),
            "source_id": self.source_id,
            "result_id": self.result_id,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object], /) -> FpMLImportResult:
        if not isinstance(record, Mapping):
            raise TypeError("FpML import result record must be a mapping.")
        expected = {
            "kind",
            "accepted",
            "contract_type",
            "contract",
            "unsupported_terms",
            "source_id",
            "result_id",
        }
        if set(record) != expected or record["kind"] != "fpml-import-result":
            raise ValueError("FpML import result fields are not canonical.")
        terms = record["unsupported_terms"]
        if not isinstance(terms, list) or any(
            not isinstance(item, str) for item in terms
        ):
            raise TypeError("Serialized unsupported_terms must be a string list.")
        contract = record["contract"]
        if contract is not None and not isinstance(contract, Mapping):
            raise TypeError("Serialized contract must be a mapping or null.")
        contract_type = record["contract_type"]
        if contract_type is not None and not isinstance(contract_type, str):
            raise TypeError("Serialized contract_type must be a string or null.")
        value = cls(
            accepted=record["accepted"],
            contract_type=contract_type,
            contract_record=contract,
            unsupported_terms=tuple(terms),
            source_id=record["source_id"],
        )
        if record["result_id"] != value.result_id:
            raise ValueError("Serialized FpML result identity is invalid.")
        return value


def _reject_nonfinite(value: str, /) -> object:
    raise ValueError(f"Non-finite JSON constant {value!r} is not supported.")


def _refusal(source_id: str, *terms: str) -> FpMLImportResult:
    return FpMLImportResult(
        accepted=False,
        contract_type=None,
        contract_record=None,
        unsupported_terms=tuple(terms),
        source_id=source_id,
    )


def import_fpml_contract(
    payload: str | bytes,
    /,
    *,
    maximum_bytes: int = _MAX_FPML_BYTES,
) -> FpMLImportResult:
    """Import the explicit single-leg FX-forward subset; refuse everything else."""
    if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int):
        raise TypeError("maximum_bytes must be an integer.")
    if maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be positive.")
    if not isinstance(payload, (str, bytes)):
        raise TypeError("FpML payload must be text or bytes.")
    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    source_id = hashlib.sha256(encoded).hexdigest()
    if len(encoded) > maximum_bytes:
        return _refusal(source_id, "resource-limit:maximum-bytes")
    if isinstance(payload, str):
        text = payload
    else:
        try:
            text = encoded.decode("utf-8")
        except UnicodeDecodeError:
            return _refusal(source_id, "malformed-utf8")
    upper = text.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        return _refusal(source_id, "unsupported-xml-declaration:doctype-or-entity")
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return _refusal(source_id, "malformed-xml")
    if _local_name(root.tag) != "dataDocument":
        return _refusal(source_id, f"unsupported-root:{_local_name(root.tag)}")
    unknown = _unknown_terms(root)
    if unknown:
        return _refusal(source_id, *unknown)
    try:
        _exact_children(root, ("trade",), "dataDocument")
        trade = _child(root, "trade", "dataDocument")
        _exact_children(trade, ("tradeHeader", "fxSingleLeg"), "trade")
        header = _child(trade, "tradeHeader", "trade")
        product = _child(trade, "fxSingleLeg", "trade")
        _exact_children(
            header,
            ("partyTradeIdentifier", "tradeDate"),
            "trade/tradeHeader",
        )
        identifier = _child(
            header,
            "partyTradeIdentifier",
            "trade/tradeHeader",
        )
        _exact_children(
            identifier,
            ("tradeId",),
            "trade/tradeHeader/partyTradeIdentifier",
        )
        _exact_children(
            product,
            ("exchangedCurrency1", "exchangedCurrency2", "valueDate"),
            "trade/fxSingleLeg",
        )
        trade_id = _child(
            identifier,
            "tradeId",
            "trade/tradeHeader/partyTradeIdentifier",
        )
        record = {
            "contract_type": "fx-forward",
            "trade_id": _text(trade_id, "trade/tradeHeader/tradeId"),
            "trade_date": _iso_date(
                _text(
                    _child(header, "tradeDate", "trade/tradeHeader"),
                    "trade/tradeHeader/tradeDate",
                ),
                "trade/tradeHeader/tradeDate",
            ),
            "exchanged_currency_1": _payment(
                _child(product, "exchangedCurrency1", "trade/fxSingleLeg"),
                "trade/fxSingleLeg/exchangedCurrency1",
            ),
            "exchanged_currency_2": _payment(
                _child(product, "exchangedCurrency2", "trade/fxSingleLeg"),
                "trade/fxSingleLeg/exchangedCurrency2",
            ),
            "value_date": _iso_date(
                _text(
                    _child(product, "valueDate", "trade/fxSingleLeg"),
                    "trade/fxSingleLeg/valueDate",
                ),
                "trade/fxSingleLeg/valueDate",
            ),
        }
    except ValueError as error:
        return _refusal(source_id, str(error))
    return FpMLImportResult(
        accepted=True,
        contract_type="fx-forward",
        contract_record=record,
        unsupported_terms=(),
        source_id=source_id,
    )


def _record_text(record: Mapping[str, Any], name: str, /) -> str:
    value = record[name]
    if not isinstance(value, str) or not value:
        raise ValueError(f"FpML export field {name!r} must be a non-empty string.")
    return value


def _subelement(parent: ET.Element, name: str, text: str | None = None) -> ET.Element:
    child = ET.SubElement(parent, f"{{{_FPML_NAMESPACE}}}{name}")
    if text is not None:
        child.text = text
    return child


def _export_payment(parent: ET.Element, name: str, record: Mapping[str, Any]) -> None:
    _validate_payment_record(record, name)
    element = _subelement(parent, name)
    _subelement(element, "payerPartyReference").set(
        "href", _record_text(record, "payer_party_id")
    )
    _subelement(element, "receiverPartyReference").set(
        "href", _record_text(record, "receiver_party_id")
    )
    payment = _subelement(element, "paymentAmount")
    _subelement(
        payment,
        "currency",
        _currency(_record_text(record, "currency"), f"{name}/currency"),
    )
    _subelement(
        payment,
        "amount",
        _amount(_record_text(record, "amount"), f"{name}/amount"),
    )


def export_fpml_contract(contract: FpMLImportResult | Mapping[str, Any], /) -> str:
    """Export only the supported single-leg FX-forward record, deterministically."""
    if isinstance(contract, FpMLImportResult):
        record = contract.contract_record()
    elif isinstance(contract, Mapping):
        record = contract
    else:
        raise TypeError("contract must be an FpMLImportResult or mapping.")
    if record.get("contract_type") != "fx-forward":
        raise ValueError("FpML export supports only contract_type='fx-forward'.")
    expected = {
        "contract_type",
        "trade_id",
        "trade_date",
        "exchanged_currency_1",
        "exchanged_currency_2",
        "value_date",
    }
    if set(record) != expected:
        raise ValueError(
            "FX-forward export fields are not the explicit supported subset."
        )
    first = record["exchanged_currency_1"]
    second = record["exchanged_currency_2"]
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise TypeError("Exchanged-currency records must be mappings.")
    root = ET.Element(f"{{{_FPML_NAMESPACE}}}dataDocument")
    trade = _subelement(root, "trade")
    header = _subelement(trade, "tradeHeader")
    identifier = _subelement(header, "partyTradeIdentifier")
    _subelement(identifier, "tradeId", _record_text(record, "trade_id"))
    _subelement(
        header,
        "tradeDate",
        _iso_date(_record_text(record, "trade_date"), "trade_date"),
    )
    product = _subelement(trade, "fxSingleLeg")
    _export_payment(product, "exchangedCurrency1", first)
    _export_payment(product, "exchangedCurrency2", second)
    _subelement(
        product,
        "valueDate",
        _iso_date(_record_text(record, "value_date"), "value_date"),
    )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", short_empty_elements=True) + "\n"


__all__ = ["FpMLImportResult", "export_fpml_contract", "import_fpml_contract"]
