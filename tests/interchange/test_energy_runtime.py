#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import jax
import pytest

from phydrax.interchange.energy_runtime import (
    EnergyRuntimeError,
    pin_energy_executable,
    PinnedExecutable,
    run_energy_command,
    run_opendss,
)


@pytest.fixture
def python_executable():
    return pin_energy_executable(
        sys.executable, version=sys.version.split()[0], license_id="PSF-2.0"
    )


def test_command_detaches_computed_artifact_and_cleans_work_directory(python_executable):
    # A real external numerical calculation, not an engine mock/echo.
    program = (
        "import json,pathlib; p=pathlib.Path; "
        "values=json.loads(p('load.json').read_text()); "
        "p('energy.json').write_text(json.dumps(sum(v*h for v,h in values))); "
        "print(p.cwd())"
    )
    result = run_energy_command(
        python_executable,
        ("-c", program),
        inputs={"load.json": b"[[100,0.25],[200,0.5]]"},
        outputs=("energy.json",),
    )
    assert json.loads(result.output("energy.json")) == 125
    assert not Path(result.stdout.decode().strip()).exists()
    assert (
        result.outputs[0].artifact.content_digest
        == hashlib.sha256(result.output("energy.json")).hexdigest()
    )
    assert result.artifact.status == "complete"


def test_timeout_retains_partial_diagnostics_and_cleans_directory(python_executable):
    with pytest.raises(EnergyRuntimeError) as caught:
        run_energy_command(
            python_executable,
            (
                "-c",
                "import pathlib,time; print(pathlib.Path.cwd(),flush=True); time.sleep(60)",
            ),
            inputs={},
            timeout=1,
        )
    result = caught.value.result
    assert result.timed_out
    assert result.returncode != 0
    assert result.artifact.status == "failed"
    assert not Path(result.stdout.decode().strip()).exists()


def test_nonzero_exit_and_missing_output_do_not_report_success(python_executable):
    with pytest.raises(EnergyRuntimeError) as caught:
        run_energy_command(
            python_executable,
            ("-c", "import sys; sys.stderr.write('invalid physical model'); sys.exit(7)"),
            inputs={},
        )
    assert caught.value.result.returncode == 7
    assert caught.value.result.stderr == b"invalid physical model"
    with pytest.raises(EnergyRuntimeError) as caught:
        run_energy_command(
            python_executable,
            ("-c", "sum(range(10))"),
            inputs={},
            outputs=("required.csv",),
        )
    assert caught.value.result.artifact.status == "failed"


def test_untrusted_paths_symlinks_and_pin_mismatch_fail_closed(python_executable):
    with pytest.raises(ValueError):
        run_energy_command(
            python_executable, ("-c", "sum(range(10))"), inputs={"../escape": b"x"}
        )
    with pytest.raises(EnergyRuntimeError) as caught:
        run_energy_command(
            python_executable,
            ("-c", "import os; os.symlink('/etc/hosts','result')"),
            inputs={},
            outputs=("result",),
        )
    assert caught.value.result.outputs == ()
    wrong_pin = PinnedExecutable(
        python_executable.path,
        "0" * 64,
        python_executable.version,
        python_executable.license_id,
    )
    with pytest.raises(EnergyRuntimeError) as caught:
        run_energy_command(wrong_pin, ("-c", "sum(range(10))"), inputs={})
    assert caught.value.result.returncode is None


def test_collected_output_bound_is_enforced(python_executable):
    with pytest.raises(EnergyRuntimeError) as caught:
        run_energy_command(
            python_executable,
            ("-c", "import pathlib; pathlib.Path('result').write_bytes(b'x'*4096)"),
            inputs={},
            outputs=("result",),
            max_output_bytes=1024,
        )
    assert caught.value.result.artifact.status == "failed"
    assert caught.value.result.outputs == ()


def test_host_boundary_rejects_even_argument_free_jit(python_executable):
    @jax.jit
    def transformed():
        run_energy_command(python_executable, ("-c", "sum(range(10))"), inputs={})
        return 1

    with pytest.raises(TypeError):
        transformed()


@pytest.mark.skipif(
    importlib.util.find_spec("opendssdirect") is None,
    reason="requires the optional real OpenDSSDirect engine",
)
def test_real_opendss_solution_keeps_multiphase_units_and_power_balance():
    result = run_opendss(
        (
            "Clear",
            "New Circuit.qualification basekv=12.47 pu=1 phases=3 bus1=source",
            "New Line.feed bus1=source bus2=load phases=3 r1=0.1 x1=0.2 r0=0.3 x0=0.4 length=1 units=km",
            "New Load.demand bus1=load phases=3 conn=wye kv=12.47 kw=100 kvar=25",
            "Set voltagebases=[12.47]",
            "CalcVoltageBases",
            "Solve",
        ),
        license_id="LicenseRef-Caller-Qualification",
    )
    assert result.converged
    assert (
        len(result.node_names)
        == len(result.node_voltages)
        == len(result.node_voltages_pu)
    )
    assert set(result.bus_names) == {"source", "load"}
    assert result.total_power[0] < -100
    assert result.losses[0] > 0
    assert -result.total_power[0] == pytest.approx(
        100 + result.losses[0] / 1000, rel=1e-5
    )
    assert result.artifact.status == "complete"
