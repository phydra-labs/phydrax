#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import importlib.util
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from phydrax.interchange import ResourceReadError
from phydrax.interchange.fmi import FMICoSimulationSession, inspect_fmu


_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def compiled_fmu(tmp_path_factory):
    if importlib.util.find_spec("fmpy") is None:
        pytest.skip("requires optional FMPy and a native C compiler")
    if sys.platform not in ("darwin", "linux"):
        pytest.skip("qualification compiler invocation supports POSIX macOS/Linux")
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("requires a native C compiler for the original real FMU specimen")
    import fmpy

    root = tmp_path_factory.mktemp("real-energy-fmu").resolve()
    extension = ".dylib" if sys.platform == "darwin" else ".so"
    library = root / ("energy_accumulator" + extension)
    subprocess.run(
        [
            compiler,
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            "-fPIC",
            "-O2",
            str(_DATA / "energy_accumulator.c"),
            "-o",
            str(library),
        ],
        check=True,
        timeout=30,
        capture_output=True,
    )
    archive = root / "energy.fmu"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.write(_DATA / "energy_accumulator.xml", "modelDescription.xml")
        output.write(library, f"binaries/{fmpy.platform}/{library.name}")
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def test_real_fmu_integration_event_and_actual_state_restore(compiled_fmu):
    path, digest = compiled_fmu
    with FMICoSimulationSession(
        path.name,
        sha256=digest,
        trusted_root=path.parent,
        license_id="LicenseRef-PHYDRA-Proprietary",
        start_values={"u": 3.0, "gain": 2.0, "label": "qualification"},
    ) as session:
        assert session.get_values(("label",)) == {"label": "qualification"}
        first = session.advance(0.25)
        assert first.reached_time == 0.25 and not first.early_return
        assert session.get_values(("x",))["x"] == pytest.approx(1.5)
        state = session.save_state()
        event = session.advance(1)
        assert event.status == "discard" and event.early_return and not event.terminated
        assert event.reached_time == 0.5
        assert session.get_values(("x", "event_done")) == {"x": -3.0, "event_done": True}
        session.restore_state(state)
        assert session.time == 0.25
        assert session.get_values(("x", "event_done")) == {"x": 1.5, "event_done": False}
        session.advance(0.4)
        assert session.get_values(("x",))["x"] == pytest.approx(2.4)
        session.free_state(state)
        session.set_values({"u": 1.0})
        session.advance(0.5)
        session.advance(0.75)
        assert session.get_values(("x",))["x"] == pytest.approx(-2.1)
    assert session.closed
    assert session.artifact.status == "complete"
    with pytest.raises(RuntimeError, match="closed"):
        session.get_values(("x",))


def test_real_fmu_termination_and_parameter_lifecycle(compiled_fmu):
    path, digest = compiled_fmu
    with FMICoSimulationSession(
        path.name,
        sha256=digest,
        trusted_root=path.parent,
        license_id="LicenseRef-PHYDRA-Proprietary",
        start_values={"u": 1.0, "stop_at_event": True},
    ) as session:
        with pytest.raises(ValueError, match="not settable at this lifecycle phase"):
            session.set_values({"gain": 2.0})
        with pytest.raises(ValueError, match="requires a finite number"):
            session.set_values({"u": "2"})
        result = session.advance(1)
        assert result.terminated and result.reached_time == 0.5
        with pytest.raises(ValueError, match="active session and a finite later time"):
            session.advance(2)


def test_archive_pin_and_path_extraction_policy(tmp_path):
    root = tmp_path.resolve()
    path = root / "bad.fmu"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "modelDescription.xml", (_DATA / "energy_accumulator.xml").read_bytes()
        )
        archive.writestr("../outside", b"untrusted")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(
        ResourceReadError, match="Archive member path is not confined"
    ) as traversal:
        inspect_fmu(path.name, sha256=digest, trusted_root=root)
    assert traversal.value.reason == "policy"
    with pytest.raises(
        ResourceReadError, match="Archive member path is not confined"
    ) as session_traversal:
        FMICoSimulationSession(
            path.name,
            sha256=digest,
            trusted_root=root,
            license_id="LicenseRef-Test-Only",
        )
    assert session_traversal.value.reason == "policy"
    assert not (root.parent / "outside").exists()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "modelDescription.xml", (_DATA / "energy_accumulator.xml").read_bytes()
        )
        link = zipfile.ZipInfo("resources/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, b"/etc/hosts")
    with pytest.raises(
        ResourceReadError, match="Archive symbolic links are disabled"
    ) as symbolic_link:
        inspect_fmu(
            path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            trusted_root=root,
        )
    assert symbolic_link.value.reason == "policy"
    with pytest.raises(ValueError, match="SHA-256 does not match its required pin"):
        inspect_fmu(path.name, sha256="0" * 64, trusted_root=root)


def test_xml_entities_and_archive_expansion_fail_before_runtime_import(tmp_path):
    path = tmp_path.resolve() / "bad.fmu"
    valid_xml = (_DATA / "energy_accumulator.xml").read_bytes()
    declaration_end = valid_xml.index(b"\n") + 1
    xml = (
        valid_xml[:declaration_end]
        + (
            b"<!DOCTYPE fmiModelDescription [<!ENTITY external "
            b'SYSTEM "file:///etc/hosts">]>\n'
        )
        + valid_xml[declaration_end:].replace(
            b'modelName="PhydraxEnergyAccumulator"',
            b'modelName="&external;"',
            1,
        )
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("modelDescription.xml", xml)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(
        ValueError, match="document types and entity declarations are forbidden"
    ):
        inspect_fmu(path.name, sha256=digest, trusted_root=path.parent)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "modelDescription.xml", (_DATA / "energy_accumulator.xml").read_bytes()
        )
        archive.writestr("resources/large", b"0" * 65536)
    with pytest.raises(ResourceReadError, match="byte limit") as expansion:
        inspect_fmu(
            path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            trusted_root=path.parent,
            max_unpacked_bytes=4096,
        )
    assert expansion.value.reason == "limit"
