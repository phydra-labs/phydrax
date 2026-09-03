#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib import import_module

import phydrax
from phydrax import applications, equations, lifecycle
from phydrax.applications import cardiovascular


_SUBPACKAGE_MODULES = {
    "anatomy": (
        "_coordinates",
        "_high_order",
        "_image_boundary",
        "_microstructure",
        "_purkinje_attachment",
        "_roles",
        "_surfaces",
        "_transfers",
    ),
    "circulation": (
        "_closed_loop",
        "_components",
        "_coronary",
        "_devices",
        "_ledger",
        "_network",
        "_oxygen",
        "_periodic",
        "_valves",
        "_vascular_1d",
    ),
    "electrophysiology": (
        "_activation",
        "_aliev_panfilov",
        "_atrial_models",
        "_bidomain",
        "_conduction_network",
        "_eikonal",
        "_integration",
        "_membrane_scaling",
        "_monodomain",
        "_nodal_models",
        "_pacing",
        "_purkinje_models",
        "_reaction",
        "_reaction_ir",
        "_regional_assignment",
        "_ventricular_models",
    ),
    "hemodynamics": (
        "_ale",
        "_domain",
        "_fixed_wall_lbm",
        "_immersed_fsi",
        "_leaflets",
        "_ports",
        "_rheology",
    ),
    "mechanics": (
        "_active_strain",
        "_active_stress",
        "_chambers",
        "_contraction",
        "_electromechanics",
        "_growth",
        "_guccione",
        "_holzapfel_ogden",
        "_materials",
        "_sarcomere",
        "_supports",
        "_unloading",
    ),
    "observations": (
        "_cine",
        "_electrograms",
        "_lge",
        "_metadata",
        "_pressure_volume",
        "_registration",
        "_sampling",
        "_strain",
    ),
    "personalization": (
        "_cohorts",
        "_design",
        "_inverse",
        "_likelihood",
        "_parameters",
        "_random_fields",
        "_reanalysis",
        "_surrogates",
        "_validation",
    ),
}


def test_cardiovascular_root_owns_only_cross_domain_contracts_and_subpackages():
    assert applications.cardiovascular is cardiovascular
    expected = list(_SUBPACKAGE_MODULES)
    for module_name in ("_case", "_commercial", "_execution", "_quantities"):
        module = import_module(f"{cardiovascular.__name__}.{module_name}")
        expected.extend(module.__all__)
        for name in module.__all__:
            assert getattr(cardiovascular, name) is getattr(module, name)

    assert cardiovascular.__all__ == expected
    assert len(cardiovascular.__all__) == len(set(cardiovascular.__all__))
    assert "HarmonicCoordinatePlan" not in cardiovascular.__all__
    assert "PressureVolumeLoopPlan" not in cardiovascular.__all__


def test_each_domain_facade_exports_each_owned_symbol_once():
    for subpackage_name, module_names in _SUBPACKAGE_MODULES.items():
        facade = getattr(cardiovascular, subpackage_name)
        expected: list[str] = []
        for module_name in module_names:
            module = import_module(f"{facade.__name__}.{module_name}")
            expected.extend(module.__all__)
            for name in module.__all__:
                assert getattr(facade, name) is getattr(module, name)
        assert facade.__all__ == expected
        assert len(facade.__all__) == len(set(facade.__all__))


def test_shared_substrates_keep_their_generic_public_owners():
    assert phydrax.ArrayArchiveLimits.__module__ == "phydrax._array_archive"
    assert equations.TensorDiffusionAction.__module__ == "phydrax.equations._variational"
    assert lifecycle.SupportBundleAuthorization.__module__ == "phydrax.lifecycle._archive"
    assert "TensorDiffusionAction" not in cardiovascular.electrophysiology.__all__
    assert "ArrayArchiveLimits" not in cardiovascular.__all__
    assert "SupportBundleAuthorization" not in cardiovascular.__all__
