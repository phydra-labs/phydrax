#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Build/run an external material-point oracle using the unmodified pinned class.

This tool does not import Phydrax or reproduce the constitutive equations. Its
caller includes acquired flexodeal.cc with main renamed and exposes protected
source branch functions for direct boundary/derivative observations. The output
is numerical cross-implementation evidence, never biological validation. The
source remains subject to the LGPL notice in tests/fixtures/flexodeal_0698e3d.
A fresh destination is mandatory; the content-addressed run record is published
only after source/runtime verification and successful executable output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / ".tmp/skeletal-production-sources"
ASSETS = ROOT / "tests/fixtures/flexodeal_0698e3d"
IMAGE_SHA256 = "1599deffe696840921d196b34cf180554eaee8873aaf9d39613fb9fe4f047471"
PARAMETER_NAMES = (
    "maximum_fiber_stress_Pa",
    "bulk_modulus_Pa",
    "maximum_base_stress_Pa",
    "base_scale",
    "maximum_strain_rate_per_s",
    "fat_bulk_modulus_Pa",
    "fat_scale",
    "fat_c1_Pa",
    "fat_fraction",
    "base_c1",
    "base_c2",
    "base_c3",
)

CALLER = r"""
#include <fstream>
#include <iomanip>
#define main flexodeal_reference_program_main
#include "flexodeal.cc"
#undef main

class MaterialOracle : public Flexodeal::Muscle_Tissues_Three_Field<3> {
public:
  using Flexodeal::Muscle_Tissues_Three_Field<3>::Muscle_Tissues_Three_Field;
  using Flexodeal::Muscle_Tissues_Three_Field<3>::get_length_stress;
  using Flexodeal::Muscle_Tissues_Three_Field<3>::get_dlength_stress_dstretch;
  using Flexodeal::Muscle_Tissues_Three_Field<3>::get_strain_rate_stress;
  using Flexodeal::Muscle_Tissues_Three_Field<3>::get_dstrain_rate_stress_dstrain_rate;
  using Flexodeal::Muscle_Tissues_Three_Field<3>::get_passive_stress;
  using Flexodeal::Muscle_Tissues_Three_Field<3>::get_dpassive_stress_dstretch;
  using Flexodeal::Muscle_Tissues_Three_Field<3>::get_aponeurosis_stress;
  using Flexodeal::Muscle_Tissues_Three_Field<3>::get_daponeurosis_stress_dstretch;
};

template <typename T> void matrix(const T &value) {
  std::cout << '[';
  for (unsigned int i=0; i<3; ++i) {
    if (i) std::cout << ',';
    std::cout << '[';
    for (unsigned int j=0; j<3; ++j) {
      if (j) std::cout << ',';
      std::cout << value[i][j];
    }
    std::cout << ']';
  }
  std::cout << ']';
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1]);
  if (!input) return 3;
  std::cout << std::setprecision(17);
  unsigned int id, tissue, dynamic;
  while (input >> id >> tissue >> dynamic) {
    double p[12], pressure, dilation, activation, dt;
    dealii::Tensor<2,3> F, previous;
    dealii::Tensor<1,3> direction;
    for (auto &v : p) input >> v;
    for (unsigned int i=0; i<3; ++i)
      for (unsigned int j=0; j<3; ++j) input >> F[i][j];
    for (unsigned int i=0; i<3; ++i)
      for (unsigned int j=0; j<3; ++j) input >> previous[i][j];
    input >> pressure >> dilation >> activation >> dt;
    for (unsigned int i=0; i<3; ++i) input >> direction[i];
    if (!input || (tissue != 1 && tissue != 2)) return 4;
    MaterialOracle material(dynamic ? "dynamic" : "quasi-static", tissue,
      p[0], p[1], p[5], p[4], direction[0], direction[1], direction[2],
      p[2], p[3], p[9], p[10], p[11], p[6], p[7], p[8]);
    const auto grad_velocity = (F-previous)/dt;
    material.update_material_data(F, pressure, dilation, activation, grad_velocity, dt);
    const auto tau = material.get_tau();
    const dealii::Tensor<2,3> full_tau(tau);
    const auto first_piola = full_tau * dealii::transpose(dealii::invert(F));
    std::cout << "{\"id\":" << id << ",\"tissue_id\":" << tissue
      << ",\"dynamic\":" << (dynamic ? "true" : "false") << ",\"parameter_values\":[";
    for (unsigned int i=0; i<12; ++i) {
      if (i) std::cout << ',';
      std::cout << p[i];
    }
    std::cout << "],\"F\":"; matrix(F);
    std::cout << ",\"F_previous\":"; matrix(previous);
    std::cout << ",\"reference_grad_velocity\":"; matrix(grad_velocity);
    std::cout << ",\"reference_direction\":[" << direction[0] << ','
      << direction[1] << ',' << direction[2] << ']';
    std::cout << ",\"pressure_Pa\":" << pressure << ",\"dilation\":" << dilation
      << ",\"activation\":" << activation << ",\"dt_s\":" << dt;
    std::cout << ",\"first_piola_Pa\":"; matrix(first_piola);
    std::cout << ",\"kirchhoff_Pa\":"; matrix(tau);
    std::cout << ",\"kirchhoff_volumetric_Pa\":"; matrix(material.get_tau_vol());
    std::cout << ",\"source_unweighted_active_Pa\":"; matrix(material.get_tau_iso_muscle_active());
    std::cout << ",\"source_unweighted_passive_fiber_Pa\":"; matrix(material.get_tau_iso_muscle_passive());
    std::cout << ",\"source_unweighted_base_Pa\":"; matrix(material.get_tau_iso_muscle_basematerial());
    std::cout << ",\"volume_ratio\":" << material.get_det_F()
      << ",\"fiber_stretch\":" << material.get_stretch()
      << ",\"isochoric_fiber_stretch\":" << material.get_stretch_bar()
      << ",\"normalized_strain_rate\":" << material.get_strain_rate_bar()
      << ",\"full_normalized_strain_rate\":" << material.get_strain_rate()
      << ",\"dPsi_vol_dJ_Pa\":" << material.get_dPsi_vol_dJ()
      << ",\"d2Psi_vol_dJ2_Pa\":" << material.get_d2Psi_vol_dJ2()
      << ",\"active_force_length\":" << material.get_length_stress()
      << ",\"active_force_length_derivative\":" << material.get_dlength_stress_dstretch()
      << ",\"active_force_velocity\":" << material.get_strain_rate_stress()
      << ",\"active_force_velocity_derivative\":" << material.get_dstrain_rate_stress_dstrain_rate()
      << ",\"passive_muscle_stress\":" << material.get_passive_stress()
      << ",\"passive_muscle_stress_derivative\":" << material.get_dpassive_stress_dstretch()
      << ",\"passive_aponeurosis_stress\":" << material.get_aponeurosis_stress()
      << ",\"passive_aponeurosis_stress_derivative\":" << material.get_daponeurosis_stress_dstretch()
      << "}\n";
  }
  return input.eof() ? 0 : 5;
}
"""

CMAKE = """cmake_minimum_required(VERSION 3.13.4)
find_package(deal.II 9.5.1 EXACT REQUIRED HINTS ${DEAL_II_DIR})
deal_ii_initialize_cached_variables()
project(almonacid_material_oracle)
add_executable(almonacid_material_oracle caller.cc)
target_include_directories(almonacid_material_oracle PRIVATE "@SOURCE@")
deal_ii_setup_target(almonacid_material_oracle)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
"""


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parameter_values(tissue: int, fraction: float) -> list[float]:
    # Unchanged repository parameters.prm, with explicitly identified fat probes.
    if tissue == 1:
        return [
            2e5,
            1e7,
            2e5,
            1.0,
            5.0,
            1e7,
            1.0,
            1.3e5,
            fraction,
            0.1990559575103343,
            0.3662334826469149,
            0.0,
        ]
    return [
        2e5,
        1e7,
        2e5,
        1.0,
        5.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.6896264975,
        -3.4551410975,
        484.92055395,
    ]


def samples() -> list[dict]:
    result = []
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def append(label, tissue, dynamic, stretch, rate=0.0, fraction=0.0):
        # F=I and |a0|=stretch provide exactly represented branch inputs: the
        # source explicitly accepts non-unit a0. Offsets include neighboring ULPs.
        dt = 0.01
        spatial_rate = 5.0 * rate / stretch
        previous = [
            [1.0 - dt * spatial_rate, 0.0, 0.0],
            [0.0, 1.0 + 0.5 * dt * spatial_rate, 0.0],
            [0.0, 0.0, 1.0 + 0.5 * dt * spatial_rate],
        ]
        result.append(
            dict(
                label=label,
                tissue_id=tissue,
                dynamic=dynamic,
                parameter_values=parameter_values(tissue, fraction),
                F=identity,
                F_previous=previous,
                pressure_Pa=1700.0,
                dilation=1.013,
                activation=0.63,
                dt_s=dt,
                reference_direction=[stretch, 0.0, 0.0],
            )
        )

    for tissue, knots in (
        (1, (0.4, 1.0, 1.25, 1.5, 1.65, 1.75)),
        (2, (1.0, 1.01, 1.02, 1.15)),
    ):
        for knot in knots:
            for stretch in (
                knot - 1e-7,
                math.nextafter(knot, -math.inf),
                knot,
                math.nextafter(knot, math.inf),
                knot + 1e-7,
            ):
                append("stretch-boundary", tissue, False, stretch)
        for stretch in (0.3, 0.8, 1.005, 1.015, 1.08, 1.35, 1.58, 1.9):
            append("stretch-interior", tissue, False, stretch)
    for knot in (-1.2, -0.25, 0.0, 0.05, 0.75):
        for rate in (knot - 1e-7, knot, knot + 1e-7):
            append("rate-boundary", 1, True, 1.1, rate)
    for rate in (-1.5, -0.6, -0.1, 0.025, 0.3, 1.0):
        append("rate-interior", 1, True, 1.1, rate)
    for tissue, fraction in ((1, 0.0), (1, 0.37), (1, 1.0), (2, 0.0)):
        for dynamic in (False, True):
            append("general-deformation", tissue, dynamic, 1.0, fraction=fraction)
            result[-1].update(
                F=[[1.17, 0.12, -0.03], [0.04, 0.92, 0.08], [0.01, 0.0, 1.04]],
                F_previous=[[1.16, 0.10, -0.01], [0.03, 0.93, 0.07], [0.0, 0.01, 1.03]],
                reference_direction=[
                    math.cos(0.267035375555),
                    0.0,
                    math.sin(0.267035375555),
                ],
            )
    return [dict(id=index, **row) for index, row in enumerate(result)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCES / "flexodeal-0698e3d")
    parser.add_argument(
        "--reference-build",
        type=Path,
        default=SOURCES / "flexodeal-build-qualified-runtime",
    )
    parser.add_argument(
        "--runtime-image",
        type=Path,
        default=SOURCES / "dealii-9.5.1-sonoma-arm64-clang17.dmg",
    )
    parser.add_argument(
        "--deal-ii",
        type=Path,
        default=Path(
            "/Applications/deal.II.app/Contents/Resources/spack/opt/dealii-9.5.1-kc32"
        ),
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.destination.resolve()
    spec = json.loads((ASSETS / "manifest.json").read_text())
    source_hash = digest(source / "flexodeal.cc")
    if source_hash != spec["upstream_files"]["flexodeal.cc"]["sha256"]:
        raise ValueError("Pinned source digest mismatch")
    if digest(args.runtime_image) != IMAGE_SHA256:
        raise ValueError("Official runtime image digest mismatch")
    build_record = json.loads((args.reference_build / "build-record.json").read_text())
    for name, artifact in build_record["artifacts"].items():
        if digest(args.reference_build / name) != artifact["sha256"]:
            raise ValueError(f"Recorded reference-build artifact mismatch: {name}")
    if not args.deal_ii.is_dir() or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Installed deal.II and finite positive timeout are required")
    application = SOURCES / "dealii-mounted/deal.II.app"
    installed_application = Path("/Applications/deal.II.app")
    if installed_application.resolve() != application.resolve():
        raise ValueError(
            "The authorized deal.II link must resolve into the pinned read-only mount"
        )
    expected_deal_ii = application / "Contents/Resources/spack/opt/dealii-9.5.1-kc32"
    if args.deal_ii.resolve() != expected_deal_ii.resolve():
        raise ValueError("Oracle must use deal.II from the pinned official runtime mount")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "caller.cc").write_text(CALLER)
    (destination / "CMakeLists.txt").write_text(
        CMAKE.replace("@SOURCE@", source.as_posix())
    )
    (destination / "source-manifest.json").write_bytes(
        (ASSETS / "manifest.json").read_bytes()
    )
    (destination / "source-NOTICE.txt").write_bytes((ASSETS / "NOTICE.txt").read_bytes())
    (destination / "source-LICENSE").write_bytes((ASSETS / "LICENSE").read_bytes())
    (destination / "runtime-evidence.txt").write_bytes(
        (args.reference_build / "runtime-evidence.txt").read_bytes()
    )
    (destination / "reference-build-record.json").write_bytes(
        (args.reference_build / "build-record.json").read_bytes()
    )
    environment = os.environ.copy()
    linker_directory = Path(
        "/Applications/deal.II.app/Contents/Resources/brew/opt/llvm/bin"
    )
    environment.update(MPICH_CXX="/usr/bin/clang++", MPICH_CC="/usr/bin/clang")
    environment["PATH"] = str(linker_directory) + os.pathsep + environment["PATH"]
    runtime_files = (
        Path("/usr/bin/clang++"),
        linker_directory / "ld64.lld",
        args.deal_ii / "lib/libdeal_II.9.5.1.dylib",
        installed_application
        / "Contents/Resources/spack/opt/mpich-4.1.2-aqfc/bin/mpic++",
    )
    runtime_hashes = {str(path.resolve()): digest(path) for path in runtime_files}
    commands = []

    def run(command, log):
        commands.append(command)
        with (destination / log).open("xb") as stream:
            subprocess.run(
                command,
                cwd=destination,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=args.timeout,
            )

    run(["/usr/bin/clang++", "--version"], "compiler-version.txt")
    run([str(linker_directory / "ld64.lld"), "--version"], "linker-version.txt")
    if (
        "Apple clang version 17.0.0"
        not in (destination / "compiler-version.txt").read_text()
    ):
        raise ValueError("Oracle requires the observed Appleclang17 compiler")
    if "LLD 17.0.2" not in (destination / "linker-version.txt").read_text():
        raise ValueError("Oracle requires the observed LLD17.0.2 linker")
    run(
        [
            "cmake",
            "-S",
            str(destination),
            "-B",
            str(destination / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            f"-DDEAL_II_DIR={args.deal_ii.resolve()}",
        ],
        "configure.log",
    )
    run(
        ["cmake", "--build", str(destination / "build"), "--parallel", "2"], "compile.log"
    )
    rows = samples()
    (destination / "inputs.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    with (destination / "inputs.txt").open("x") as stream:
        for row in rows:
            values = [
                row["id"],
                row["tissue_id"],
                int(row["dynamic"]),
                *row["parameter_values"],
                *(v for line in row["F"] for v in line),
                *(v for line in row["F_previous"] for v in line),
                row["pressure_Pa"],
                row["dilation"],
                row["activation"],
                row["dt_s"],
                *row["reference_direction"],
            ]
            stream.write(" ".join(str(value) for value in values) + "\n")
    executable = destination / "build/almonacid_material_oracle"
    run([str(executable), str(destination / "inputs.txt")], "samples.jsonl")
    observed = [
        json.loads(line)
        for line in (destination / "samples.jsonl").read_text().splitlines()
    ]
    if [row["id"] for row in observed] != list(range(len(rows))):
        raise ValueError("Oracle output is incomplete or out of order")
    if digest(source / "flexodeal.cc") != source_hash:
        raise ValueError("Upstream source changed during oracle generation")
    run(["/usr/bin/otool", "-L", str(executable)], "linked-libraries.txt")
    if any(digest(Path(path)) != checksum for path, checksum in runtime_hashes.items()):
        raise ValueError("Compiler or reference runtime changed during oracle generation")
    artifact_names = (
        "caller.cc",
        "CMakeLists.txt",
        "source-manifest.json",
        "source-NOTICE.txt",
        "source-LICENSE",
        "runtime-evidence.txt",
        "reference-build-record.json",
        "compiler-version.txt",
        "linker-version.txt",
        "configure.log",
        "compile.log",
        "inputs.txt",
        "inputs.json",
        "samples.jsonl",
        "linked-libraries.txt",
        "build/almonacid_material_oracle",
        "build/CMakeCache.txt",
        "build/compile_commands.json",
    )
    record = dict(
        source_id=spec["source_id"],
        source_sha256=source_hash,
        source_directory=str(source),
        runtime_image_sha256=IMAGE_SHA256,
        generator_sha256=digest(Path(__file__)),
        platform=platform.platform(),
        python=sys.version,
        commands=commands,
        parameter_names=PARAMETER_NAMES,
        runtime_files_sha256=runtime_hashes,
        environment={key: environment[key] for key in ("MPICH_CXX", "MPICH_CC", "PATH")},
        sample_count=len(observed),
        caveats=[
            "External numerical material oracle, not biological validation.",
            "Repository defaults differ from the publication parameter table.",
            "Source component getters are unweighted; total includes the fat mixture.",
            "Rate boundary inputs undergo upstream floating-point kinematics; compare the recorded actual rate.",
            "Source hand-coded branch derivatives are observations, not native AD overrides.",
        ],
        artifacts={
            name: dict(
                sha256=digest(destination / name),
                bytes=(destination / name).stat().st_size,
            )
            for name in artifact_names
        },
    )
    record["content_id"] = hashlib.sha256(
        json.dumps(
            record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    with (destination / "run-record.json").open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "run_record": str(destination / "run-record.json"),
                "content_id": record["content_id"],
                "sample_count": len(observed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
