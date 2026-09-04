#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.geometry import RigidFrame
from phydrax.interchange import AdapterError, AdapterStatus
from phydrax.interchange.opticstudio import _adapter as adapter
from phydrax.optics.geometric import SequentialOpticsPlan


class _FakeSurface:
    def __init__(self, name: str):
        self.name = name
        self.MaterialCell = object()
        self.Comment = ""
        self.Radius = 0.0
        self.Conic = 0.0
        self.Thickness = 0.0
        self.SemiDiameter = None


class _FakeLensDataEditor:
    def __init__(self):
        self.surfaces = [_FakeSurface("object"), _FakeSurface("image")]

    def GetSurfaceAt(self, index: int):
        return self.surfaces[index]

    def InsertNewSurfaceAt(self, index: int):
        surface = _FakeSurface(f"surface-{index}")
        self.surfaces.insert(index, surface)
        return surface


class _FakeSystem:
    def __init__(self):
        self.new_calls = 0
        self.sequential_calls = 0
        self.SystemData = SimpleNamespace(Units=SimpleNamespace(LensUnits="initial-unit"))
        self.LDE = _FakeLensDataEditor()

    def new(self, *, saveifneeded: bool):
        assert saveifneeded is False
        self.new_calls += 1
        self.LDE = _FakeLensDataEditor()

    def make_sequential(self):
        self.sequential_calls += 1


class _FakeResult:
    def __init__(self, payload: object):
        self.payload = payload

    def to_json(self):
        return json.dumps(self.payload, sort_keys=False, separators=(", ", ": "))


class _FakeConnection:
    def __init__(self, system: _FakeSystem, *, connect_failure: bool = False):
        self.system = system
        self.connect_failure = connect_failure
        self.connect_modes = []
        self.disconnect_calls = 0

    def connect(self, *, mode: str):
        self.connect_modes.append(mode)
        if self.connect_failure:
            raise RuntimeError("mock connection failure")
        return self.system

    def disconnect(self):
        self.disconnect_calls += 1


class _FakeZOSPy(ModuleType):
    def __init__(self, *, connect_failure: bool = False, analysis_failure: bool = False):
        super().__init__("zospy")
        self.__version__ = "9.7.mock"
        self.system = _FakeSystem()
        self.connection = _FakeConnection(self.system, connect_failure=connect_failure)
        self.constructor_keywords = []
        self.material_calls = []
        self.analysis_calls = []
        self.analysis_failure = analysis_failure
        self.constants = SimpleNamespace(
            SystemData=SimpleNamespace(ZemaxSystemUnits=SimpleNamespace(Meters="metres"))
        )
        self.solvers = SimpleNamespace(material_model=self._material_model)
        self.analyses = SimpleNamespace(
            reports=SimpleNamespace(
                CardinalPoints=self._analysis("cardinal-points"),
                SurfaceData=self._analysis("surface-data"),
                SystemData=self._analysis("system-data"),
            )
        )

        module = self

        class ZOS:
            def __new__(cls, **keywords):
                module.constructor_keywords.append(dict(keywords))
                return module.connection

        self.ZOS = ZOS

    def _material_model(self, cell: object, *, refractive_index: float):
        self.material_calls.append((cell, refractive_index))

    def _analysis(self, name: str):
        module = self

        class Analysis:
            def __init__(self, **settings):
                self.settings = settings

            def run(self, system: object, *, oncomplete: str):
                assert system is module.system
                module.analysis_calls.append((name, dict(self.settings), oncomplete))
                if module.analysis_failure:
                    raise RuntimeError("mock analysis failure")
                return _FakeResult(
                    {
                        "settings": dict(reversed(tuple(self.settings.items()))),
                        "name": name,
                        "data": {"b": 2, "a": 1},
                    }
                )

        return Analysis


def _install_fake_zospy(
    monkeypatch, *, connect_failure: bool = False, analysis_failure: bool = False
):
    module = _FakeZOSPy(
        connect_failure=connect_failure, analysis_failure=analysis_failure
    )
    monkeypatch.setitem(sys.modules, "zospy", module)
    return module


def _plan(
    surface_kinds=("plane", "sphere", "conic"),
    interactions=("transmit", "transmit", "transmit"),
):
    surface_count = len(surface_kinds)
    frames = tuple(
        RigidFrame(np.eye(3), np.asarray((0.0, 0.0, value)))
        for value in (0.0, 10.0, 15.0)[:surface_count]
    )
    curvatures = np.asarray((0.0, 0.02, -0.04)[:surface_count])
    conics = np.asarray((0.0, 0.0, -1.0)[:surface_count])
    coefficients = np.zeros((surface_count, 1))
    coefficient_active = np.zeros((surface_count, 1), dtype=bool)
    apertures = np.full((surface_count,), 5.0)
    aperture_active = np.ones((surface_count,), dtype=bool)
    indices = np.asarray((1.0, 1.5, 1.5, 1.0)[: surface_count + 1])
    return SequentialOpticsPlan(
        frames,
        surface_kinds,
        interactions,
        curvatures,
        conics,
        coefficients,
        coefficient_active,
        apertures,
        aperture_active,
        np.full((surface_count,), 100.0),
        indices,
    )


def _even_asphere_plan():
    return SequentialOpticsPlan(
        (RigidFrame.identity(3),),
        ("even-asphere",),
        ("transmit",),
        np.asarray((0.02,)),
        np.asarray((-1.0,)),
        np.asarray(((1.0e-6,),)),
        np.asarray(((True,),)),
        np.asarray((5.0,)),
        np.asarray((True,)),
        np.asarray((100.0,)),
        np.asarray((1.0, 1.5)),
    )


def test_module_use_is_lazy_until_an_availability_or_session_operation(monkeypatch):
    calls = []

    def unexpected_import(name: str):
        calls.append(name)
        raise AssertionError("optional dependency import was not requested")

    monkeypatch.setattr(adapter.importlib, "import_module", unexpected_import)
    backend = adapter.OpticStudioBackend()
    request = adapter.OpticStudioAnalysisRequest("system_data")
    assert backend.backend_id
    assert adapter.OPTICSTUDIO_CAPABILITIES.host_only
    assert adapter.OPTICSTUDIO_CAPABILITIES.requires_explicit_release
    assert request.analysis == "system-data"
    assert calls == []


def test_availability_is_nonthrowing_when_optional_dependency_is_unavailable(
    monkeypatch,
):
    def unavailable(name: str):
        assert name == "zospy"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(adapter.importlib, "import_module", unavailable)
    availability = adapter.opticstudio_availability()
    assert availability.available is False
    assert availability.reason == "optional dependency 'zospy' could not be imported"
    assert availability.versions == ()
    assert availability.capabilities is adapter.OPTICSTUDIO_CAPABILITIES


def test_session_disconnects_once_after_success_and_failure(monkeypatch):
    zospy = _install_fake_zospy(monkeypatch)
    backend = adapter.OpticStudioBackend(opticstudio_directory=" C:/OpticStudio ")
    with backend.open_session() as session:
        assert session.closed is False
        assert session.producer_version == "9.7.mock"
        assert {"system", "connection", "zospy"}.isdisjoint(dir(session))
    assert session.closed is True
    session.close()
    assert zospy.connection.connect_modes == ["standalone"]
    assert zospy.connection.disconnect_calls == 1
    assert zospy.constructor_keywords == [{"opticstudio_directory": "C:/OpticStudio"}]

    second = _install_fake_zospy(monkeypatch, analysis_failure=True)
    with pytest.raises(RuntimeError, match="mock analysis failure"):
        with backend.open_session() as failed_session:
            adapter.run_opticstudio_analysis(
                failed_session, adapter.OpticStudioAnalysisRequest("system-data")
            )
    assert second.connection.disconnect_calls == 1


def test_failed_connection_is_also_disconnected(monkeypatch):
    zospy = _install_fake_zospy(monkeypatch, connect_failure=True)
    with pytest.raises(RuntimeError, match="mock connection failure"):
        adapter.OpticStudioBackend().open_session()
    assert zospy.connection.disconnect_calls == 1


def test_supported_sequential_export_is_lossless_and_si_normalized(monkeypatch):
    zospy = _install_fake_zospy(monkeypatch)
    with adapter.OpticStudioBackend().open_session() as session:
        report = adapter.export_sequential_to_opticstudio(
            _plan(), session, length_unit_in_metres=1.0e-3
        )
        result = adapter.run_opticstudio_analysis(
            session, adapter.OpticStudioAnalysisRequest("system-data")
        )
    assert report.status is AdapterStatus.LOSSLESS
    assert report.valid
    assert result.artifact.parent_artifact_ids == (report.target_id,)
    assert zospy.system.new_calls == 1
    assert zospy.system.sequential_calls == 1
    assert zospy.system.SystemData.Units.LensUnits == "metres"
    surfaces = zospy.system.LDE.surfaces[1:-1]
    assert [surface.Radius for surface in surfaces] == pytest.approx((0.0, 0.05, -0.025))
    assert [surface.Thickness for surface in surfaces] == pytest.approx(
        (0.01, 0.005, 0.0)
    )
    assert [surface.SemiDiameter for surface in surfaces] == pytest.approx(
        (0.005, 0.005, 0.005)
    )
    assert [value for _, value in zospy.material_calls] == pytest.approx(
        (1.0, 1.5, 1.5, 1.0)
    )


def test_unsupported_features_are_reported_before_vendor_mutation(monkeypatch):
    zospy = _install_fake_zospy(monkeypatch)
    with adapter.OpticStudioBackend().open_session() as session:
        with pytest.raises(AdapterError) as error:
            adapter.export_sequential_to_opticstudio(
                _even_asphere_plan(), session, length_unit_in_metres=1.0e-3
            )
    assert error.value.status is AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
    assert error.value.features == ("surfaces[0].surface_kind=even-asphere",)
    assert error.value.report.valid is False
    assert tuple(loss.path for loss in error.value.report.losses) == (
        "surfaces[0].surface_kind",
    )
    assert zospy.system.new_calls == 0
    assert zospy.material_calls == []
    assert zospy.connection.disconnect_calls == 1


def test_requests_and_detached_results_have_deterministic_normalization(monkeypatch):
    zospy = _install_fake_zospy(monkeypatch)
    first_request = adapter.OpticStudioAnalysisRequest(
        "CARDINAL_POINTS",
        settings={"wavelength": np.int64(1), "surface_2": "Image", "surface_1": 1},
    )
    second_request = adapter.OpticStudioAnalysisRequest(
        "cardinal-points",
        settings={"surface_1": 1, "surface_2": "Image", "wavelength": 1},
    )
    assert first_request.settings == second_request.settings
    assert first_request.request_id == second_request.request_id
    with adapter.OpticStudioBackend().open_session() as session:
        first = adapter.run_opticstudio_analysis(session, first_request)
        second = adapter.run_opticstudio_analysis(session, second_request)
    assert first.payload_json == (
        '{"data":{"a":1,"b":2},"name":"cardinal-points",'
        '"settings":{"surface_1":1,"surface_2":"Image","wavelength":1}}'
    )
    assert first.payload_json == second.payload_json
    assert first.result_id == second.result_id
    assert first.report.status is AdapterStatus.LOSSLESS
    assert first.artifact.content_digest == second.artifact.content_digest
    assert zospy.analysis_calls == [
        (
            "cardinal-points",
            {"surface_1": 1, "surface_2": "Image", "wavelength": 1},
            "Close",
        ),
        (
            "cardinal-points",
            {"surface_1": 1, "surface_2": "Image", "wavelength": 1},
            "Close",
        ),
    ]


def test_traced_values_are_rejected_at_host_boundary():
    @jax.jit
    def request_from_traced_surface(surface):
        return adapter.OpticStudioAnalysisRequest(
            "surface-data", settings={"surface": surface}
        )

    with pytest.raises(TypeError, match="traced value.*host-only"):
        request_from_traced_surface(jnp.asarray(1))


def test_export_rejects_a_traced_plan_before_session_access(monkeypatch):
    zospy = _install_fake_zospy(monkeypatch)
    plan = _plan()
    with adapter.OpticStudioBackend().open_session() as session:

        @jax.jit
        def export_with_curvatures(curvatures):
            traced_plan = eqx.tree_at(lambda value: value.curvatures, plan, curvatures)
            return adapter.export_sequential_to_opticstudio(
                traced_plan, session, length_unit_in_metres=1.0e-3
            )

        with pytest.raises(TypeError, match="traced value.*host-only"):
            export_with_curvatures(plan.curvatures)
    assert zospy.system.new_calls == 0
