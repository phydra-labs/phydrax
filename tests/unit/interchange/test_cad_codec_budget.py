"""Explicit codec ledgers survive admission, proof preparation and publication."""

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Never

import pytest

from phydrax._external_resource import bounded_resource_from_bytes, ResourceLimits
from phydrax._resource_ticket import BatchTicket
from phydrax.geometry.brep import BRepModel, BRepTessellationPolicy
from phydrax.interchange import CadImportPolicy, CadInterchangeError, read_cad
from tests._support.cad_models import native_solid, placement_budget


_FORMATS = ("step", "iges", "brep_text")


def _codec(format_name: str) -> ModuleType:
    module_name = "cad_brep_text" if format_name == "brep_text" else format_name
    return import_module(f"phydrax.interchange._{module_name}")


def _policy(model: BRepModel, maximum_bytes: int = 1 << 24) -> CadImportPolicy:
    return CadImportPolicy(
        model.coordinate_contract,
        ResourceLimits(maximum_bytes, 64, 200_000, 4_000_000, 0),
        tessellation=BRepTessellationPolicy(realize=False),
    )


@pytest.mark.parametrize("format_name", _FORMATS)
def test_codec_preserves_explicit_budget(
    format_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _codec(format_name)
    model = native_solid("box")
    path = tmp_path / ("box.brep" if format_name == "brep_text" else f"box.{format_name}")
    unit = {"source_length_unit": model.coordinate_contract.length_unit} if format_name == "brep_text" else {}
    with placement_budget() as budget:
        seen = []

        def observe(owner: Any, name: str) -> None:
            original = getattr(owner, name)

            def call(*args: Any, **kwargs: Any) -> Any:
                assert kwargs.get("coordinate_budget") is budget
                seen.append(name)
                return original(*args, **kwargs)

            monkeypatch.setattr(owner, name, call)

        observe(codec, "prepare_cad_export_geometry")
        if hasattr(codec, "certified_cad_export_tolerances"):
            observe(codec, "certified_cad_export_tolerances")
        observe(codec.CadStage, "publish")
        if hasattr(codec, "build_cad_import_tolerance_evidence"):
            observe(codec, "build_cad_import_tolerance_evidence")
        getattr(codec, f"write_{format_name}")(model, path, coordinate_budget=budget)
        assert "prepare_cad_export_geometry" in seen
        policy = _policy(model)
        data = path.read_bytes()
        resource = bounded_resource_from_bytes(data, limits=policy.limits)
        for result in (
            getattr(codec, f"decode_{format_name}_resource")(
                resource, policy, coordinate_budget=budget, **unit
            ),
            getattr(codec, f"decode_{format_name}_bytes")(
                data, policy, coordinate_budget=budget, **unit
            ),
            getattr(codec, f"read_{format_name}")(
                path, policy, trusted_root=tmp_path, coordinate_budget=budget, **unit
            ),
            read_cad(path, policy, trusted_root=tmp_path, coordinate_budget=budget, **unit),
        ):
            assert result.model.geometry is not None
        assert seen.count("publish") == 4
        if format_name != "iges":
            assert seen.count("build_cad_import_tolerance_evidence") == 4
            assert "certified_cad_export_tolerances" in seen


@pytest.mark.parametrize("format_name", _FORMATS)
def test_codec_explicit_budget_keeps_source_byte_limit(format_name: str) -> None:
    codec = _codec(format_name)
    model = native_solid("box")
    unit = {"source_length_unit": model.coordinate_contract.length_unit} if format_name == "brep_text" else {}
    with placement_budget() as budget, pytest.raises(CadInterchangeError):
        getattr(codec, f"decode_{format_name}_bytes")(
            b"oversized", _policy(model, maximum_bytes=1), coordinate_budget=budget, **unit
        )


@pytest.mark.parametrize("format_name", _FORMATS)
@pytest.mark.parametrize("boundary", ("admission", "publication"))
def test_codec_deadline_refuses_before_publication(
    format_name: str, boundary: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _codec(format_name)
    model = native_solid("box")
    path = tmp_path / "refused-output"
    with placement_budget() as budget:
        original_checkpoint = type(budget.ticket).checkpoint
        expired = boundary == "admission"

        def checkpoint(ticket: BatchTicket) -> None:
            if ticket is budget.ticket and expired:
                raise TimeoutError("codec deadline")
            return original_checkpoint(ticket)

        monkeypatch.setattr(type(budget.ticket), "checkpoint", checkpoint)
        writer_class = codec._IgesWriter if format_name == "iges" else codec._Writer
        original_encode = writer_class.encode

        def encode(writer: Any) -> Any:
            nonlocal expired
            data = original_encode(writer)
            expired = True
            return data

        monkeypatch.setattr(writer_class, "encode", encode)
        with pytest.raises(TimeoutError, match="codec deadline"):
            getattr(codec, f"write_{format_name}")(model, path, coordinate_budget=budget)
        assert not path.exists()
        expired = False


@pytest.mark.parametrize("format_name", _FORMATS)
def test_codec_decode_deadline_refuses_at_admission(
    format_name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _codec(format_name)
    model = native_solid("box")
    unit = {"source_length_unit": model.coordinate_contract.length_unit} if format_name == "brep_text" else {}
    with placement_budget() as budget:
        original_checkpoint = type(budget.ticket).checkpoint

        def checkpoint(ticket: BatchTicket) -> None:
            if ticket is budget.ticket:
                raise TimeoutError("codec deadline")
            return original_checkpoint(ticket)

        with monkeypatch.context() as patch:
            patch.setattr(type(budget.ticket), "checkpoint", checkpoint)
            with pytest.raises(TimeoutError, match="codec deadline"):
                getattr(codec, f"decode_{format_name}_bytes")(
                    b"", _policy(model), coordinate_budget=budget, **unit
                )


@pytest.mark.parametrize("format_name", _FORMATS)
def test_codec_stage_budget_exhaustion_remains_resource_refusal(
    format_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError

    codec = _codec(format_name)
    model = native_solid("box")
    path = tmp_path / "source"
    getattr(codec, f"write_{format_name}")(model, path)
    unit = {"source_length_unit": model.coordinate_contract.length_unit} if format_name == "brep_text" else {}
    with placement_budget() as budget:
        exhaustion = CoordinateEnclosureResourceError("coefficient_work", 0, 1, 0)

        def publish(stage: Any, **kwargs: Any) -> Never:
            assert kwargs["coordinate_budget"] is budget
            raise exhaustion

        monkeypatch.setattr(codec.CadStage, "publish", publish)
        with pytest.raises(CadInterchangeError) as caught:
            getattr(codec, f"read_{format_name}")(
                path, _policy(model), trusted_root=tmp_path,
                coordinate_budget=budget, **unit
            )
        assert caught.value.refusal.reason == "limit"
        assert caught.value.refusal.entity == "coordinate-budget"
        assert caught.value.__cause__ is exhaustion
        assert caught.value.resource_manifest is not None


def test_native_archive_revalidates_with_the_shared_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = import_module("phydrax.interchange._cad_archive")
    model_owner = import_module("phydrax.geometry.brep._model")
    model = native_solid("box")
    path = tmp_path / "model.phx"
    with placement_budget() as budget:
        original = model_owner._validate_exact_geometry
        seen = []

        def validate(*args: Any, **kwargs: Any) -> Any:
            assert kwargs["budget"] is budget
            seen.append(True)
            return original(*args, **kwargs)

        monkeypatch.setattr(model_owner, "_validate_exact_geometry", validate)
        archive.save_brep_archive(model, path, coordinate_budget=budget)
        restored = archive.load_brep_archive(path, coordinate_budget=budget)
        assert seen
        assert restored.model_id == model.model_id


def test_native_archive_reuses_request_certified_trim_loops(tmp_path: Path) -> None:
    archive = import_module("phydrax.interchange._cad_archive")
    model = native_solid("box")
    path = tmp_path / "loops.phx"
    archive.save_brep_archive(model, path)
    with placement_budget() as budget:
        first = archive.load_brep_archive(path, coordinate_budget=budget)
        second = archive.load_brep_archive(path, coordinate_budget=budget)
        assert second.model_id == first.model_id == model.model_id
        for before, after in zip(first.trim_domains, second.trim_domains, strict=True):
            assert before is not None and after is not None
            assert after.outer is before.outer


@pytest.mark.parametrize("boundary", ("admission", "publication"))
def test_native_archive_deadline_refuses_before_publication(
    boundary: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = import_module("phydrax.interchange._cad_archive")
    model = native_solid("box")
    path = tmp_path / "refused.phx"
    with placement_budget() as budget:
        checkpoint = type(budget.ticket).checkpoint
        factor = archive._factor_definitions
        expired = boundary == "admission"

        def check(ticket: BatchTicket) -> None:
            if ticket is budget.ticket and expired:
                raise TimeoutError("archive deadline")
            return checkpoint(ticket)

        def prepare(*args: Any, **kwargs: Any) -> Any:
            nonlocal expired
            result = factor(*args, **kwargs)
            expired = True
            return result

        monkeypatch.setattr(type(budget.ticket), "checkpoint", check)
        monkeypatch.setattr(archive, "_factor_definitions", prepare)
        with pytest.raises(TimeoutError, match="archive deadline"):
            archive.save_brep_archive(model, path, coordinate_budget=budget)
        assert not path.exists()
        expired = False
