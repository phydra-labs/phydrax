# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Real cuEST DF-RHF qualification; run with ``python -m benchmarks.cuest_molecular``.

The PySCF oracle uses the identical explicit spherical auxiliary basis and metric
cutoff. Exact four-center HF is reported separately, never as a matched DF oracle.
No CPU fallback is selected when the cuEST provider cannot be admitted.

Geometry admission/binding, one-electron evaluation, and the first J/K call are
timed separately. The RHF loop is native-integral host execution, not a compiled
full-SCF program; first operation/shape calls can include JAX compilation.

Shared runtime/source, package, native-binary, device, and compiler evidence is
retained. A later raw-primal-FFI study measures real lowering/compilation/warm
costs in a separate memory window; it does not make the host SCF loop one JIT
program. Use ``benchmarks.cuest_molecular_campaign`` for fresh-process orbital
and auxiliary capacity series; one H2 point supports no broader speedup claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal

import jax
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    logical_array_bytes,
)
from phydrax._cuda_native import cuda_native_resources
from phydrax._execution_device import device_platform
from phydrax._execution_sampling import sample_device_memory, sample_host_memory


_EVIDENCE_FIELDS = (
    "auxiliary_count",
    "changed_geometry_energy_hartree",
    "command",
    "compiled_kernel_external_gpu_memory",
    "compiled_kernels",
    "dependencies",
    "device_inventory",
    "energy_hartree",
    "environment",
    "error",
    "execution_flags",
    "external_gpu_memory",
    "forces_hartree_per_bohr",
    "identity",
    "independent_reference",
    "integral_definition_id",
    "method",
    "native_component",
    "orbital_count",
    "policies",
    "preparation_memory",
    "provider",
    "requested_capacity",
    "scientific_checks",
    "scientific_tolerances",
    "selected_devices",
    "status",
    "timing_evidence",
    "timings_seconds",
)


type ShellDefinition = tuple[int, tuple[tuple[float, float], ...]]

# Same tolerances as the matched-reference integration scenarios. These gates
# are declared before measurement and are evaluated only outside timed regions.
_OE_MATRIX_RTOL = 2.0e-10
_OE_MATRIX_ATOL = 2.0e-11
_DF_ENERGY_ATOL = 2.0e-9
_DF_FORCE_RTOL = 4.0e-5
_DF_FORCE_ATOL = 8.0e-7
_DF_JK_RTOL = 3.0e-8
_DF_JK_ATOL = 2.0e-9


def _pyscf_basis(shells: tuple[ShellDefinition, ...]) -> list[list[int | list[float]]]:
    """Adapt typed shell definitions to PySCF's heterogeneous external records."""
    return [
        [angular, *([exponent, coefficient] for exponent, coefficient in primitives)]
        for angular, primitives in shells
    ]


@dataclass(frozen=True)
class CuestCase:
    structure: phx.atomistic.AtomicStructure
    system: phx.atomistic.AtomisticSystemPlan
    basis: phx.operators.quantum.gaussian.PreparedGaussianBasis
    auxiliary: phx.operators.quantum.gaussian.PreparedGaussianBasis
    source: phx.operators.quantum.gaussian.CuestIntegralPlan
    integrals: phx.operators.quantum.gaussian.MolecularIntegralPlan
    hf: phx.chemistry.MolecularHartreeFockPlan
    positions: np.ndarray
    orbital_shells: tuple[ShellDefinition, ...]
    auxiliary_shells: tuple[ShellDefinition, ...]
    pyscf_to_native: np.ndarray


def build_case(
    *,
    angular: int = 1,
    molecules: int = 1,
    bond_angstrom: float = 0.82,
    exponent_scale: float = 1.0,
    auxiliary_exponent_scale: float = 1.0,
    fitting_cutoff: float = 1.0e-12,
    relative_conditioning: bool = False,
    fitting_algorithm: Literal["qr", "matrix-power"] = "matrix-power",
    variable_workspace_bytes: int = 64 * 1024**2,
    maximum_workspace_bytes: int = 512 * 1024**2,
    maximum_cached_geometries: int = 1,
) -> CuestCase:
    """Explicit, identically normalized native/PySCF shells, without library lookup."""
    gaussian = phx.operators.quantum.gaussian
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    coordinates = np.asarray(
        [
            [5.0 * molecule, 0.13 * molecule, sign * bond_angstrom / 2]
            for molecule in range(molecules)
            for sign in (-1, 1)
        ],
        dtype=np.float64,
    )
    count = len(coordinates)
    ids = np.arange(11, 11 + count, dtype=np.int64)
    structure = phx.atomistic.AtomicStructure(
        np.ones(count, dtype=np.int32),
        coordinates,
        np.full(count, 1.008),
        units.scale,
        particle_ids=ids,
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=np.repeat(np.arange(molecules), 2)
    )
    orbital_shells: list[ShellDefinition] = [
        (
            0,
            (
                (3.42525091 * exponent_scale, 0.15432897),
                (0.62391373 * exponent_scale, 0.53532814),
                (0.16885540 * exponent_scale, 0.44463454),
            ),
        ),
        (0, ((0.075 * exponent_scale, 1.0),)),
    ]
    orbital_shells.extend(
        (ell, (((0.6 if ell == 1 else 0.5) * exponent_scale, 1.0),))
        for ell in range(1, angular + 1)
    )
    auxiliary_shells = tuple(
        (ell, ((exponent * auxiliary_exponent_scale, 1.0),))
        for ell in range(2 * angular + 1)
        for exponent in (0.12, 0.4, 1.3, 4.0, 12.0)
    )

    def prepare(
        shells: Sequence[ShellDefinition], role: str
    ) -> phx.operators.quantum.gaussian.PreparedGaussianBasis:
        return gaussian.GaussianBasisPlan(
            tuple(
                gaussian.GaussianShellPlan(
                    int(particle),
                    ell,
                    np.asarray([row[0] for row in primitives], dtype=np.float64),
                    np.asarray([row[1] for row in primitives], dtype=np.float64),
                    representation=gaussian.GaussianShellRepresentation.REAL_SPHERICAL,
                    role=role,
                )
                for particle in ids
                for ell, primitives in shells
            ),
            maximum_basis_functions=max(512, count * 200),
            source_id=f"cuest-qualification-explicit-{role}",
            role=role,
        ).prepare(system)

    basis, auxiliary = (
        prepare(orbital_shells, "orbital"),
        prepare(auxiliary_shells, "auxiliary"),
    )
    source = gaussian.CuestIntegralPlan(
        auxiliary,
        fitting_cutoff=fitting_cutoff,
        relative_conditioning=relative_conditioning,
        fitting_algorithm=fitting_algorithm,
        screening_threshold=1.0e-14,
        variable_workspace_bytes=variable_workspace_bytes,
        maximum_workspace_bytes=maximum_workspace_bytes,
        maximum_cached_geometries=maximum_cached_geometries,
    )
    integrals = gaussian.MolecularIntegralPlan(source)
    hf = phx.chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        phx.chemistry.MolecularElectronicSectorPlan(0, 1),
        phx.chemistry.ElectronicReferenceKind.RESTRICTED,
        convergence=phx.chemistry.SCFConvergencePlan(
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
            commutator_tolerance=1.0e-10,
            maximum_iterations=200,
        ),
        integrals=integrals,
    )
    permutation = []
    offset = 0
    for _ in ids:
        for ell, _ in orbital_shells:
            local = (
                [0]
                if ell == 0
                else [2, 0, 1]
                if ell == 1
                else [ell]
                + [index for m in range(1, ell + 1) for index in (ell + m, ell - m)]
            )
            permutation.extend(offset + index for index in local)
            offset += len(local)
    positions = coordinates * float(
        phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
    )
    return CuestCase(
        structure,
        system,
        basis,
        auxiliary,
        source,
        integrals,
        hf,
        positions,
        tuple(orbital_shells),
        auxiliary_shells,
        np.asarray(permutation),
    )


def external_reference(
    case: CuestCase, positions: np.ndarray | None = None, *, exact: bool = False
) -> Any:
    """PySCF DF-RHF with independently constructed metric-pseudoinverse factors.

    PySCF's default Cholesky admission differs from an explicit metric eigenvalue
    cutoff, so build its packed DF factors ourselves. Auxiliary AO permutations
    cancel in B M^-1 B^T; the orbital permutation is applied by callers.
    """
    from pyscf import df, gto, scf
    from pyscf.lib import pack_tril

    coordinates = case.positions if positions is None else positions
    molecule = gto.M(
        atom=[("H", tuple(point)) for point in coordinates],
        basis={"H": _pyscf_basis(case.orbital_shells)},
        unit="Bohr",
        cart=False,
        verbose=0,
    )
    reference = scf.RHF(molecule)
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.max_cycle = 200
    if not exact:
        auxiliary = df.addons.make_auxmol(
            molecule, {"H": _pyscf_basis(case.auxiliary_shells)}
        )
        metric = auxiliary.intor("int2c2e", hermi=1)
        eigenvalues, eigenvectors = np.linalg.eigh(metric)
        cutoff = case.source.fitting_cutoff
        if case.source.relative_conditioning:
            cutoff *= eigenvalues[-1]
        retained = eigenvalues > cutoff
        if not np.any(retained):
            raise RuntimeError(
                "Independent DF reference has no retained auxiliary metric modes."
            )
        three_center = df.incore.aux_e2(molecule, auxiliary, intor="int3c2e", aosym="s1")
        factors = np.einsum(
            "ijp,pr->rij",
            three_center,
            eigenvectors[:, retained] / np.sqrt(eigenvalues[retained]),
            optimize=True,
        )
        reference = reference.density_fit(
            auxbasis={"H": _pyscf_basis(case.auxiliary_shells)}
        )
        reference.with_df._cderi = pack_tril(factors)
    return reference


class ExternalGpuMemory:
    """Sample this process's externally observed GPU memory, not an allocator estimate."""

    def __init__(self) -> None:
        self.samples: list[int] = []
        self.error: str | None = None
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> ExternalGpuMemory:
        if shutil.which("nvidia-smi") is None:
            self.error = "nvidia-smi unavailable"
            return self

        def sample() -> None:
            while not self.stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=5,
                    )
                    memory = 0
                    for line in result.stdout.splitlines():
                        pid, mib = [field.strip() for field in line.split(",")]
                        if int(pid) == os.getpid():
                            memory += int(mib) * 1024**2
                    self.samples.append(memory)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    self.error = str(error)
                    return
                self.stop.wait(0.05)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()

    def record(self) -> dict[str, Any]:
        return {
            "source": "nvidia-smi compute-apps current PID; 50 ms sampling",
            "samples": len(self.samples),
            "peak_bytes": max(self.samples) if self.samples else None,
            "unavailable_reason": self.error,
        }


def timed(function: Callable[[], Any]) -> tuple[Any, float]:
    start = perf_counter()
    value = function()
    jax.block_until_ready(value)
    return value, perf_counter() - start


def matched_df_checks(
    *,
    energy: float,
    reference_energy: float,
    forces: np.ndarray,
    reference_forces: np.ndarray,
    coulomb: np.ndarray,
    reference_coulomb: np.ndarray,
    exchange: np.ndarray,
    reference_exchange: np.ndarray,
    reference_converged: bool,
    native_jk_successful: bool,
) -> dict[str, bool]:
    """Gate the real benchmark using the matching DF Hamiltonian, not exact HF."""
    return {
        "reference_converged": reference_converged,
        "native_jk_successful": native_jk_successful,
        "energies_finite": bool(np.isfinite(energy) and np.isfinite(reference_energy)),
        "forces_finite": bool(
            np.all(np.isfinite(forces)) and np.all(np.isfinite(reference_forces))
        ),
        "coulomb_finite": bool(
            np.all(np.isfinite(coulomb)) and np.all(np.isfinite(reference_coulomb))
        ),
        "exchange_finite": bool(
            np.all(np.isfinite(exchange)) and np.all(np.isfinite(reference_exchange))
        ),
        "matched_df_energy": bool(
            np.isclose(energy, reference_energy, rtol=0.0, atol=_DF_ENERGY_ATOL)
        ),
        "matched_df_forces": bool(
            forces.shape == reference_forces.shape
            and np.allclose(
                forces, reference_forces, rtol=_DF_FORCE_RTOL, atol=_DF_FORCE_ATOL
            )
        ),
        "matched_df_coulomb": bool(
            coulomb.shape == reference_coulomb.shape
            and np.allclose(
                coulomb, reference_coulomb, rtol=_DF_JK_RTOL, atol=_DF_JK_ATOL
            )
        ),
        "matched_df_exchange": bool(
            exchange.shape == reference_exchange.shape
            and np.allclose(
                exchange, reference_exchange, rtol=_DF_JK_RTOL, atol=_DF_JK_ATOL
            )
        ),
    }


def _package_versions() -> dict[str, str | None]:
    packages = (
        "phydrax",
        "phydrax-cuda13",
        "phydrax-cuest",
        "jax",
        "jaxlib",
        "jax-cuda13-plugin",
        "jax-cuda13-pjrt",
        "numpy",
        "scipy",
        "pyscf",
        "nvidia-cuda-runtime",
        "nvidia-cublas",
        "nvidia-cusolver",
        "nvidia-cusparse",
        "nvidia-cuda-nvrtc",
        "nvidia-nvjitlink",
    )
    result: dict[str, str | None] = {}
    for package in sorted(packages):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _binary_identity(path: str, /) -> dict[str, str | int]:
    location = Path(path)
    with location.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(location), "sha256": digest, "bytes": location.stat().st_size}


def _native_provenance() -> dict[str, Any]:
    components = [
        value for value in cuda_native_resources().components if value.name == "cuest"
    ]
    if len(components) != 1:
        raise RuntimeError(
            "cuEST benchmark requires exactly one admitted native component identity."
        )
    component = components[0]
    vendor = [
        path for path in component.preloaded if Path(path).name.startswith("libcuest.so")
    ]
    if len(vendor) != 1:
        raise RuntimeError(
            "cuEST benchmark cannot identify its admitted vendor runtime uniquely."
        )
    distribution = importlib.metadata.distribution("phydrax-cuest")
    manifests = [
        file
        for file in distribution.files or ()
        if str(file) == "phydrax_cuest/build.json"
    ]
    if len(manifests) != 1:
        raise RuntimeError(
            "Installed phydrax-cuest RECORD lacks its unique vendor build manifest."
        )
    manifest_path = distribution.locate_file(manifests[0])
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError("cuEST vendor build manifest must be a JSON object.")
    return {
        "component": component.name,
        "version": component.version,
        "cuda_major": component.major,
        "build_hash": component.build_hash,
        "companion_binary": _binary_identity(component.path),
        "vendor_binary": _binary_identity(vendor[0]),
        "admitted_dependency_paths": list(component.preloaded),
        "vendor_build_manifest": manifest,
        "vendor_manifest_path": str(manifest_path),
    }


def _device_inventory() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    return {
        "status": "observed",
        "source": "nvidia-smi read-only inventory; not all devices are necessarily used",
        "columns": "index,uuid,pci.bus_id,name,driver_version",
        "csv": result.stdout,
    }


def _selected_devices(
    bound: phx.operators.quantum.gaussian.CuestGeometry, /
) -> list[dict[str, str | int]]:
    return [
        {
            "id": device.id,
            "process_index": device.process_index,
            "platform": device_platform(device),
            "jax_platform": device.platform,
            "kind": device.device_kind,
            "runtime_platform_version": device.client.platform_version,
        }
        for device in sorted(
            bound.key.devices(), key=lambda value: (value.process_index, value.id)
        )
    ]


def _preparation_memory(
    prepared: phx.operators.quantum.gaussian.PreparedCuestIntegrals,
    bound: phx.operators.quantum.gaussian.CuestGeometry,
    /,
) -> dict[str, Any]:
    devices = tuple(
        sorted(bound.key.devices(), key=lambda value: (value.process_index, value.id))
    )
    return {
        "scope": "instantaneous preparation snapshot; not a total-process or vendor-memory bound",
        "prepared_recipe_logical_array_bytes": logical_array_bytes(prepared),
        "binding_logical_array_bytes": logical_array_bytes(bound),
        "binding_certificate_bytes": bound.key.nbytes,
        "host": sample_host_memory().to_payload(),
        "device_allocator": [
            sample.to_payload() for sample in sample_device_memory(devices)
        ],
        "native_owned_workspace_limit_bytes": prepared.plan.maximum_workspace_bytes,
        "native_variable_scratch_limit_bytes": prepared.plan.variable_workspace_bytes,
        "native_budget_scope": "queried host/device native workspaces and tracked wrapper allocations, including leased generations; excludes JAX arrays, driver/JIT and vendor-private allocations",
    }


def _finite_metric(value: float, /) -> float | None:
    """Retain failed records as finite JSON; failed scientific gates explain nulls."""
    return value if np.isfinite(value) else None


def _benchmark_record(args: argparse.Namespace, /) -> dict[str, Any]:
    """Capture input, software, and device provenance before phase clocks."""
    return {
        "provider": "cuEST",
        "method": "nonperiodic all-electron real-FP64 integer DF-RHF",
        "status": "not_admitted",
        "timings_seconds": {},
        "reference_definition": "same explicit spherical auxiliary basis and metric cutoff; exact HF is not the DF oracle",
        "scientific_tolerances": {
            "one_electron_matrices": {
                "absolute": _OE_MATRIX_ATOL,
                "relative": _OE_MATRIX_RTOL,
                "units": "dimensionless overlap; kinetic/attraction in hartree",
            },
            "energy_hartree": {"absolute": _DF_ENERGY_ATOL, "relative": 0.0},
            "forces_hartree_per_bohr": {
                "absolute": _DF_FORCE_ATOL,
                "relative": _DF_FORCE_RTOL,
            },
            "coulomb_exchange_hartree": {
                "absolute": _DF_JK_ATOL,
                "relative": _DF_JK_RTOL,
            },
        },
        "requested_capacity": {
            "angular": args.angular,
            "molecules": args.molecules,
            "repeats": args.repeats,
        },
        "command": [sys.executable, *sys.argv],
        "environment": capture_environment().to_dict(),
        "dependencies": _package_versions(),
        "device_inventory": _device_inventory(),
        "execution_flags": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_ENABLE_COMPILATION_CACHE",
                "JAX_ENABLE_X64",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "NVIDIA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
            )
        },
        "identity": capture_benchmark_identity(
            Path(__file__).resolve().parents[1], Path(__file__), _EVIDENCE_FIELDS
        ).to_dict(),
    }


def _benchmark_case(args: argparse.Namespace, record: dict[str, Any], /) -> CuestCase:
    """Prepare the selected capacity and retain its actual integral policies."""
    phases = record["timings_seconds"]
    start = perf_counter()
    case = build_case(angular=args.angular, molecules=args.molecules)
    phases["basis_and_plan_setup"] = perf_counter() - start
    record["integral_definition_id"] = case.integrals.definition_id(case.basis)
    record["orbital_count"] = case.basis.basis_function_count
    record["auxiliary_count"] = case.auxiliary.basis_function_count
    record["policies"] = {
        "fitting_algorithm": case.source.fitting_algorithm,
        "fitting_cutoff": case.source.fitting_cutoff,
        "relative_conditioning": case.source.relative_conditioning,
        "screening_threshold": case.source.screening_threshold,
        "variable_workspace_bytes": case.source.variable_workspace_bytes,
        "maximum_workspace_bytes": case.source.maximum_workspace_bytes,
        "maximum_cached_geometries": case.source.maximum_cached_geometries,
    }
    return case


def _host_numerics(
    case: CuestCase, record: dict[str, Any], repeats: int, /
) -> tuple[
    phx.operators.quantum.gaussian.CuestGeometry,
    phx.chemistry.RestrictedMeanFieldState,
    phx.chemistry.MolecularGradientResult,
]:
    """Execute the selected RHF host workflow with its original phase clocks."""
    phases = record["timings_seconds"]
    prepared, phases["basis_layout_preparation"] = timed(
        lambda: case.source.prepare(case.basis)
    )
    bound, phases["component_admission_and_geometry_binding"] = timed(
        lambda: prepared.bind_geometry(case.positions, np.ones(len(case.positions)))
    )
    one, phases["one_electron"] = timed(bound.one_electron)
    if not bool(one.successful):
        raise RuntimeError("cuEST one-electron evaluation unsuccessful")
    record["status"] = "admitted"
    record["native_component"] = _native_provenance()
    record["selected_devices"] = _selected_devices(bound)
    record["preparation_memory"] = _preparation_memory(prepared, bound)
    dimension = case.basis.basis_function_count
    vector = np.linspace(0.2, 0.8, dimension)
    density = np.outer(vector, vector)

    def changed_density(index: int) -> np.ndarray:
        return density * (0.3 + 0.1 * index) + np.eye(dimension) * (0.01 * (index + 1))

    first_jk, phases["first_jk_setup_and_compilation"] = timed(
        lambda: bound.jk(changed_density(0))
    )
    if not bool(first_jk.successful):
        raise RuntimeError("cuEST first J/K evaluation unsuccessful")
    changed = []
    for index in range(repeats):
        actual = changed_density(index)
        jk, seconds = timed(lambda: bound.jk(actual))
        if not bool(jk.successful):
            raise RuntimeError("cuEST changed-density J/K evaluation unsuccessful")
        changed.append(seconds)
    phases["changed_density_jk"] = changed
    record["timing_evidence"] = {
        "execution": "native cuEST integrals in a host RHF loop; no full-SCF JIT",
        "performance_scope": "one orbital/auxiliary capacity point; no scaling or broader speedup extrapolation",
        "phases_may_include_first_shape_compilation": [
            "component_admission_and_geometry_binding",
            "one_electron",
            "first_jk_setup_and_compilation",
            "scf",
            "analytic_forces",
        ],
        "changed_density_jk": {
            "samples": repeats,
            "first_shape_compilation_included": False,
            "setup": "same bound geometry and same-shape J/K executed before samples",
            "density_sequence": "D_i = outer(v,v) * (0.3+0.1*i) + I * 0.01*(i+1), v = linspace(0.2,0.8,n)",
        },
        "scf": "first host SCF; includes first-use density-factor shapes and orchestration",
        "analytic_forces": "first analytic gradient call; includes operation setup/compilation",
        "changed_geometry_scf": "host SCF including fresh geometry binding and native setup; reuses compiled shapes where possible",
    }
    state, phases["scf"] = timed(lambda: case.hf.solve_atomic_units(case.positions))
    if not bool(state.evidence.successful):
        raise RuntimeError("cuEST native SCF did not converge")
    gradient, phases["analytic_forces"] = timed(
        lambda: case.hf.analytic_gradient_atomic_units(case.positions, state)
    )
    if not bool(gradient.successful):
        raise RuntimeError("cuEST analytic gradient unsuccessful")
    moved = case.positions.copy()
    moved[-1, 2] += 0.08
    moved_state, phases["changed_geometry_scf"] = timed(
        lambda: case.hf.solve_atomic_units(moved)
    )
    if not bool(moved_state.evidence.successful):
        raise RuntimeError("cuEST changed-geometry SCF did not converge")
    record.update(
        energy_hartree=float(state.total_energy),
        forces_hartree_per_bohr=np.asarray(gradient.forces).tolist(),
        changed_geometry_energy_hartree=float(moved_state.total_energy),
    )
    return bound, state, gradient


def _reference_admission(case: CuestCase, record: dict[str, Any], /) -> tuple[bool, Any]:
    """Distinguish a missing required oracle from an installed oracle failure."""
    try:
        reference = external_reference(case)
    except ModuleNotFoundError as error:
        if error.name != "pyscf":
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            return False, None
        record["independent_reference"] = {
            "status": "dependency_absent",
            "dependency": "pyscf",
        }
        record["status"] = "unqualified"
        record["error"] = "Required independent matched DF oracle is unavailable."
        return False, None
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        return False, None
    return True, reference


def _qualify_reference(
    case: CuestCase,
    state: phx.chemistry.RestrictedMeanFieldState,
    gradient: phx.chemistry.MolecularGradientResult,
    reference: Any,
    record: dict[str, Any],
    /,
) -> bool:
    """Measure and admit the independent matched-DF reference in phase order."""
    phases = record["timings_seconds"]
    df_energy, phases["pyscf_matched_df_scf"] = timed(reference.kernel)
    if not reference.converged or not np.isfinite(float(df_energy)):
        raise RuntimeError(
            "Independent matched PySCF DF-RHF did not converge to a finite energy"
        )
    exact = external_reference(case, exact=True)
    exact_energy, phases["pyscf_exact_scf"] = timed(exact.kernel)
    if not exact.converged or not np.isfinite(float(exact_energy)):
        raise RuntimeError(
            "Independent exact PySCF RHF did not converge to a finite energy"
        )
    df_gradient, phases["pyscf_matched_df_analytic_forces"] = timed(
        lambda: reference.nuc_grad_method().kernel()
    )
    permutation = case.pyscf_to_native
    reference_density = reference.make_rdm1()[np.ix_(permutation, permutation)]
    jk = case.integrals.jk(case.basis, case.positions, reference_density, None)
    ref_j, ref_k = reference.get_jk(dm=reference.make_rdm1())
    native_forces = np.asarray(gradient.forces, dtype=np.float64)
    ref_forces = -np.asarray(df_gradient, dtype=np.float64)
    native_j, native_k = (
        np.asarray(jk.coulomb, dtype=np.float64),
        np.asarray(jk.exchange, dtype=np.float64),
    )
    ref_j = np.asarray(ref_j, dtype=np.float64)[np.ix_(permutation, permutation)]
    ref_k = np.asarray(ref_k, dtype=np.float64)[np.ix_(permutation, permutation)]
    checks = matched_df_checks(
        energy=float(state.total_energy),
        reference_energy=float(df_energy),
        forces=native_forces,
        reference_forces=ref_forces,
        coulomb=native_j,
        reference_coulomb=ref_j,
        exchange=native_k,
        reference_exchange=ref_k,
        reference_converged=bool(reference.converged),
        native_jk_successful=bool(jk.successful),
    )
    record["scientific_checks"] = checks
    record["independent_reference"] = {
        "status": "qualified" if all(checks.values()) else "failed",
        "matched_df_converged": bool(reference.converged),
        "matched_df_energy_hartree": _finite_metric(float(df_energy)),
        "df_energy_error_hartree": _finite_metric(
            float(state.total_energy) - float(df_energy)
        ),
        "df_force_max_abs_error": _finite_metric(
            float(np.max(np.abs(native_forces - ref_forces)))
        ),
        "df_j_max_abs_error": _finite_metric(float(np.max(np.abs(native_j - ref_j)))),
        "df_k_max_abs_error": _finite_metric(float(np.max(np.abs(native_k - ref_k)))),
        "exact_converged": bool(exact.converged),
        "exact_energy_hartree_not_matched_oracle": _finite_metric(float(exact_energy)),
        "df_minus_exact_hartree": _finite_metric(float(df_energy) - float(exact_energy)),
    }
    if not all(checks.values()):
        record["status"] = "failed"
        record["error"] = "Matched DF scientific qualification failed: " + ", ".join(
            name for name, passed in checks.items() if not passed
        )
        return False
    return True


def _compiled_kernel_measurements(
    bound: phx.operators.quantum.gaussian.CuestGeometry,
    state: phx.chemistry.MeanFieldState,
    gradient: phx.chemistry.MolecularGradientResult,
    record: dict[str, Any],
    repeats: int,
    /,
) -> None:
    """Keep real raw-FFI compiler work in its separate memory observation window."""
    from benchmarks._cuest_kernels import measure_cuest_kernels

    if not isinstance(state, phx.chemistry.RestrictedMeanFieldState):
        raise TypeError(
            "cuEST compiler qualification requires the actual restricted SCF state."
        )
    with ExternalGpuMemory() as compiler_memory:
        record["compiled_kernels"] = measure_cuest_kernels(
            bound, state, gradient, repeats=repeats
        )
    record["compiled_kernel_external_gpu_memory"] = compiler_memory.record()
    record["compiled_kernel_external_gpu_memory"]["scope"] = (
        "later raw-primal-FFI lowering/compile/first/warm study; not the host-workflow peak"
    )


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    record = _benchmark_record(args)
    case = _benchmark_case(args, record)
    with ExternalGpuMemory() as memory:
        try:
            bound, state, gradient = _host_numerics(case, record, args.repeats)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            if record["status"] == "admitted":
                record["status"] = "failed"
    record["external_gpu_memory"] = memory.record()
    record["external_gpu_memory"]["scope"] = (
        "host workflow through changed-geometry SCF; excludes later independent oracle and raw-FFI compiler study"
    )
    if record["status"] != "admitted":
        return record
    reference_available, reference = _reference_admission(case, record)
    if not reference_available:
        return record
    try:
        if not _qualify_reference(case, state, gradient, reference, record):
            return record
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    try:
        _compiled_kernel_measurements(bound, state, gradient, record, args.repeats)
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        return record
    record["status"] = "completed"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--molecules", type=int, default=1)
    parser.add_argument("--angular", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.molecules < 1 or args.repeats < 1:
        parser.error("molecules and repeats must be positive")
    try:
        result = benchmark(args)
    except Exception as error:
        result = {
            "provider": "cuEST",
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
