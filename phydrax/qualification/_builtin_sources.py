#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Built-in source atlas for the omniphysics program."""

from __future__ import annotations

from ._closure_taxonomy import SourceReuseClass
from ._source_audits import _SOURCE_AUDITS
from ._source_reference import SourceAbsorptionLedger, SourceReference, SourceReview


_SOURCE_SPECS = (
    (
        "adamantine",
        "https://github.com/adamantine-sim/adamantine",
        "Apache-2.0-WITH-LLVM-exception",
        "permissive",
        ("additive-manufacturing", "moving-heat-source", "material-state"),
    ),
    (
        "exa-ca",
        "https://github.com/LLNL/ExaCA",
        "MIT",
        "permissive",
        ("grain-growth", "thermal-history-transfer"),
    ),
    (
        "bernaise",
        "https://github.com/gautelinga/BERNAISE",
        "MIT",
        "permissive",
        ("electrohydrodynamics", "phase-field", "surface-charge"),
    ),
    (
        "py-stokes",
        "https://github.com/rajeshrinet/pystokes",
        "MIT",
        "permissive",
        ("phoresis", "stokesian-dynamics"),
    ),
    (
        "sfepy",
        "https://github.com/sfepy/sfepy",
        "BSD-3-Clause",
        "permissive",
        ("piezoelectricity", "finite-element"),
    ),
    (
        "ross",
        "https://github.com/petrobras/ross",
        "Apache-2.0",
        "permissive",
        ("rotordynamics", "bearings", "seals"),
    ),
    (
        "pylife",
        "https://github.com/boschresearch/pylife",
        "Apache-2.0",
        "permissive",
        ("fatigue", "load-collectives"),
    ),
    (
        "pybamm",
        "https://github.com/pybamm-team/PyBaMM",
        "BSD-3-Clause",
        "permissive",
        ("electrochemistry", "porous-electrode"),
    ),
    (
        "cantera",
        "https://github.com/Cantera/cantera",
        "BSD-3-Clause",
        "permissive",
        ("surface-chemistry", "thermochemistry"),
    ),
    (
        "openfoam",
        "https://github.com/OpenFOAM/OpenFOAM-dev",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("multiphase-flow", "combustion", "industrial-cfd"),
    ),
    (
        "additive-foam",
        "https://github.com/ORNL/AdditiveFOAM",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("additive-manufacturing", "heat-source-calibration"),
    ),
    (
        "laserbeam-foam",
        "https://github.com/laserbeamfoam/LaserbeamFoam",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("melt-pool", "laser-ray-tracing"),
    ),
    (
        "openfast",
        "https://github.com/OpenFAST/openfast",
        "Apache-2.0",
        "permissive",
        ("wind-energy", "aero-hydro-servo-elastic"),
    ),
    (
        "opengeosys",
        "https://github.com/ufz/ogs",
        "BSD-3-Clause",
        "permissive",
        ("thmc", "geotechnical", "subsurface"),
    ),
    (
        "project-chrono",
        "https://github.com/projectchrono/chrono",
        "BSD-3-Clause",
        "permissive",
        ("multibody", "contact", "vehicles"),
    ),
    (
        "mfix",
        "https://mfix.netl.doe.gov/gitlab/exa/docs",
        "source-available",
        "source-available",
        ("multiphase-reactors", "tfm-dem-pic"),
    ),
    (
        "palace",
        "https://github.com/awslabs/palace",
        "Apache-2.0",
        "permissive",
        ("electromagnetics", "ports", "frequency-domain"),
    ),
    (
        "warpx",
        "https://github.com/BLAST-WarpX/warpx",
        "BSD-3-Clause",
        "permissive",
        ("plasma", "particle-in-cell"),
    ),
    (
        "simvascular",
        "https://github.com/SimVascular/SimVascular",
        "BSD-3-Clause",
        "permissive",
        ("medical-devices", "vascular-flow"),
    ),
    (
        "jsbsim",
        "https://github.com/JSBSim-Team/jsbsim",
        "LGPL-2.1-or-later",
        "weak-copyleft",
        ("flight-dynamics", "aircraft-systems"),
    ),
    (
        "opm-flow",
        "https://github.com/OPM/opm-simulators",
        "GPL-3.0-or-later",
        "strong-copyleft",
        ("reservoir", "wells", "schedules"),
    ),
)


def _source_reference(spec) -> SourceReference:
    source_id, _, licence, reuse, concepts = spec
    (
        repository_url,
        revision,
        archive_digest,
        licence_digest,
        readme_record,
        licence_record,
    ) = _SOURCE_AUDITS[source_id]
    reuse_class = SourceReuseClass(reuse)
    return SourceReference.create(
        source_id,
        repository_url,
        revision,
        licence,
        reuse_class,
        concepts=concepts,
        archive_digest=archive_digest,
        licence_digest=licence_digest,
        relevant_documents=(readme_record, licence_record),
        code_inspected=False,
        behavior_inspected=False,
        copying_permitted=reuse_class
        in (SourceReuseClass.PERMISSIVE, SourceReuseClass.PUBLIC_DOMAIN),
        provider_only=reuse_class
        in (
            SourceReuseClass.STRONG_COPYLEFT,
            SourceReuseClass.SOURCE_AVAILABLE,
        ),
        technical_reviewer="phydrax-source-metadata-audit-2026-09-20",
        legal_reviewer="automated-licence-classification-2026-09-20",
        review_status=SourceReview.TECHNICALLY_REVIEWED,
    )


def builtin_source_absorption_ledger() -> SourceAbsorptionLedger:
    return SourceAbsorptionLedger.create(
        tuple(_source_reference(spec) for spec in _SOURCE_SPECS)
    )


__all__ = ["builtin_source_absorption_ledger"]
