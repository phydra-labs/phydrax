#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned Genesis 1.3 version 4 oracle for the time-dependent averaged FEL.

Genesis 4 (GPL-3.0) is used only as a caller-pinned external executable; no
source is copied. :func:`run_genesis4` translates the supported subset of a
:class:`FELTimeDependentPlan` and uniform :class:`FELBeamSlices` into a Genesis
input deck and lattice, runs it through the shared pinned-command runtime, and
reads the time-resolved power from the published ``.out.h5`` artifact.

Supported subset (everything else is refused): SI scale; the
``"angular-spectrum"`` model on a square grid centered on the axis; fundamental
harmonic only; an open window without head padding; modules with drifts but no
thin quadrupoles, phase shifters, or smooth focusing; no wakes, prebunching,
or pulse-envelope seed (Genesis cannot express an arbitrary sampled envelope,
so a Gaussian pulse is declared with :class:`Genesis4GaussianSeed`); every
slice identical; slice spacing an integer multiple ``sample`` of the
wavelength. :class:`FELSpaceCharge` maps onto Genesis ``&efield``: the
radial solve (``harmonics`` → ``nz``, ``azimuthal_modes`` → ``nphi``,
``radial_cells`` → ``ngrid``, ``radial_extent`` → ``rmax``) and, for a
bunch-scale plan, Genesis' own uniform-disk long-range field
(``longrange``); Genesis has no transverse space charge, so
``transverse="applied"`` is refused. Genesis' bunch coordinate ``s`` grows
toward the head, so slot ``j`` of the ``"positive-late"`` window is Genesis
slice ``S − 1 − j``.
"""

from __future__ import annotations

import math
from pathlib import Path

import equinox as eqx
import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array

from ...._external_runtime import (
    PinnedExecutable,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from ...._physical import ElectromagneticScaleContract
from ...._strict import StrictModule
from ...._validation import finite_real_scalar, positive_finite_float, positive_integer
from ._slices import FELBeamSlices
from ._time_dependent import FELTimeDependentPlan


_OUTPUT = "fel.out.h5"


class Genesis4GaussianSeed(StrictModule):
    """Gaussian seed pulse ``P(ζ) = P₀ exp(−(ζ − ζ_c)²/(2σ²))`` for Genesis.

    ``center_position`` is in the slice coordinate of the time-dependent plan;
    ``waist`` is the ``1/e`` field radius ``w₀`` focused ``focus_position``
    downstream of the entrance.
    """

    peak_power: float = eqx.field(static=True)
    center_position: float = eqx.field(static=True)
    rms_length: float = eqx.field(static=True)
    waist: float = eqx.field(static=True)
    focus_position: float = eqx.field(static=True)

    def __init__(
        self,
        peak_power: float,
        /,
        *,
        center_position: float,
        rms_length: float,
        waist: float,
        focus_position: float = 0.0,
    ) -> None:
        self.peak_power = positive_finite_float(peak_power, "peak_power")
        self.center_position = finite_real_scalar(center_position, "center_position")
        self.rms_length = positive_finite_float(rms_length, "rms_length")
        self.waist = positive_finite_float(waist, "waist")
        self.focus_position = finite_real_scalar(focus_position, "focus_position")


class Genesis4Result(StrictModule):
    """Genesis time-resolved power on the plan's window slots.

    ``power[z, j]`` (W) is ordered like ``FELTimeDependentResult.power`` (head
    first) at the Genesis output positions ``positions``;
    ``pulse_energy[z]`` is ``Σ_j P Δζ / c``. ``provider_version`` and
    ``output_sha256`` identify the pinned run.
    """

    positions: Array
    power: Array
    pulse_energy: Array
    provider_version: str = eqx.field(static=True)
    output_sha256: str = eqx.field(static=True)


def _number(value: float, /) -> str:
    return repr(float(value))


def _uniform_value(values: np.ndarray, name: str, /) -> np.ndarray:
    if not np.all(values == values[:1]):
        raise ValueError(f"The Genesis oracle needs identical slices ({name}).")
    return values[0]


def _lattice_text(plan: FELTimeDependentPlan, /) -> tuple[str, str]:
    lattice = plan.core.lattice
    helical = lattice.polarization == "helical"
    focusing = "kx=0.5, ky=0.5" if helical else "kx=0.0, ky=1.0"
    lines: list[str] = []
    names: list[str] = []
    for index, segment in enumerate(lattice.segments):
        if (
            segment.quadrupole_integrated_gradient != 0.0
            or segment.phase_shift != 0.0
            or segment.smooth_focusing_gradient != 0.0
        ):
            raise ValueError(
                "The Genesis oracle refuses thin quadrupoles, phase shifters, and "
                "smooth focusing."
            )
        device = segment.device
        strength = lattice.deflections[index]
        rms = strength if helical else strength / math.sqrt(2.0)
        lines.append(
            f"U{index}: UNDULATOR = {{lambdau={_number(device.period)}, "
            f"nwig={device.period_count}, aw={_number(rms)}, "
            f"helical={'true' if helical else 'false'}, {focusing}}};"
        )
        names.append(f"U{index}")
        if segment.drift_length > 0.0:
            lines.append(f"D{index}: DRIFT = {{l={_number(segment.drift_length)}}};")
            names.append(f"D{index}")
    lines.append(f"FEL: LINE = {{{', '.join(names)}}};")
    return "\n".join(lines) + "\n", "FEL"


def _grid(plan: FELTimeDependentPlan, /) -> tuple[int, float]:
    space = plan.core.field_space
    if plan.core.transverse != "angular-spectrum" or space is None:
        raise ValueError("The Genesis oracle needs the angular-spectrum grid model.")
    rows, columns = space.shape
    coordinates = np.asarray(space.transverse_coordinates)
    extent = float(np.max(np.abs(coordinates)))
    if rows != columns or not np.allclose(
        np.max(coordinates, axis=(0, 1)), -np.min(coordinates, axis=(0, 1))
    ):
        raise ValueError("The Genesis oracle needs a square grid centered on the axis.")
    return rows, extent


def genesis4_input(
    plan: FELTimeDependentPlan,
    slices: FELBeamSlices,
    /,
    *,
    seed: Genesis4GaussianSeed | None,
    random_seed: int = 1,
) -> dict[str, bytes]:
    """Translate the supported subset into Genesis ``fel.in`` and ``fel.lat``."""
    if not isinstance(plan, FELTimeDependentPlan):
        raise TypeError("plan must be an FELTimeDependentPlan.")
    if not isinstance(slices, FELBeamSlices):
        raise TypeError("slices must be FELBeamSlices.")
    core = plan.core
    if core.lattice.scale.scale_id != ElectromagneticScaleContract.si().scale_id:
        raise ValueError("The Genesis oracle needs the SI scale.")
    if core.harmonics != (1,):
        raise ValueError("The Genesis oracle models the fundamental only.")
    if plan.boundary != "open" or plan.head_padding != 0:
        raise ValueError("The Genesis oracle needs an open window without padding.")
    if (
        core.wake is not None
        or core.seed is not None
        or plan.prebunching is not None
        or plan.pulse_seed is not None
    ):
        raise ValueError(
            "The Genesis oracle refuses wakes, prebunching, and plan seeds; declare "
            "a Genesis4GaussianSeed instead."
        )
    space_charge = plan.space_charge
    if space_charge is not None and space_charge.transverse == "applied":
        raise ValueError("Genesis has no transverse space charge.")
    spacing = slices.position_spacing
    if spacing is None:
        raise ValueError("The Genesis oracle needs uniformly spaced slices.")
    sample = spacing / core.wavelength
    if abs(sample - round(sample)) > 1.0e-9 or round(sample) < 1:
        raise ValueError("Slice spacing must be an integer multiple of the wavelength.")
    random = positive_integer(random_seed, "random_seed")
    current = _uniform_value(np.asarray(slices.currents), "currents")
    gamma = _uniform_value(np.asarray(slices.lorentz_factors), "lorentz_factors")
    spread = _uniform_value(np.asarray(slices.relative_energy_spreads), "spreads")
    emittance = _uniform_value(np.asarray(slices.normalized_emittances), "emittances")
    beta = _uniform_value(np.asarray(slices.beta_functions), "beta_functions")
    alpha = _uniform_value(np.asarray(slices.alpha_functions), "alpha_functions")
    ngrid, dgrid = _grid(plan)
    lattice_text, beamline = _lattice_text(plan)
    loading = core.loading
    count = slices.slice_count
    blocks = [
        "&setup",
        "rootname = fel",
        "lattice = fel.lat",
        f"beamline = {beamline}",
        f"lambda0 = {_number(core.wavelength)}",
        f"gamma0 = {_number(float(gamma))}",
        f"delz = {_number(core.lattice.step_length)}",
        f"seed = {random}",
        f"npart = {loading.particle_count}",
        f"nbins = {loading.particles_per_beamlet}",
        f"shotnoise = {'true' if loading.shot_noise == 'fawley' else 'false'}",
        "exclude_field_dump = true",
        "exclude_fft_output = true",
        "&end",
        "",
        "&time",
        "s0 = 0",
        f"slen = {_number(count * spacing)}",
        f"sample = {round(sample)}",
        "&end",
        "",
    ]
    if seed is not None:
        if not isinstance(seed, Genesis4GaussianSeed):
            raise TypeError("seed must be a Genesis4GaussianSeed or None.")
        head = float(np.asarray(slices.positions)[0])
        center = (count - 1) * spacing - (seed.center_position - head)
        blocks += [
            "&profile_gauss",
            "label = seedpower",
            f"c0 = {_number(seed.peak_power)}",
            f"s0 = {_number(center)}",
            f"sig = {_number(seed.rms_length)}",
            "&end",
            "",
        ]
    blocks += [
        "&field",
        f"power = {'@seedpower' if seed is not None else '0'}",
        f"waist_size = {_number(seed.waist if seed is not None else dgrid)}",
        f"waist_pos = {_number(seed.focus_position if seed is not None else 0.0)}",
        f"dgrid = {_number(dgrid)}",
        f"ngrid = {ngrid}",
        "&end",
        "",
        "&beam",
        f"current = {_number(float(current))}",
        f"gamma = {_number(float(gamma))}",
        f"delgam = {_number(float(spread * gamma))}",
        f"ex = {_number(float(emittance[0]))}",
        f"ey = {_number(float(emittance[1]))}",
        f"betax = {_number(float(beta[0]))}",
        f"betay = {_number(float(beta[1]))}",
        f"alphax = {_number(float(alpha[0]))}",
        f"alphay = {_number(float(alpha[1]))}",
        "&end",
        "",
    ]
    if space_charge is not None:
        blocks += [
            "&efield",
            f"longrange = {'true' if space_charge.bunch is not None else 'false'}",
            f"rmax = {_number(space_charge.radial_extent)}",
            f"ngrid = {space_charge.radial_cells}",
            f"nz = {space_charge.harmonics}",
            f"nphi = {space_charge.azimuthal_modes}",
            "&end",
            "",
        ]
    blocks += [
        "&track",
        "&end",
        "",
    ]
    return {
        "fel.in": "\n".join(blocks).encode(),
        "fel.lat": lattice_text.encode(),
    }


def run_genesis4(
    executable: PinnedExecutable,
    plan: FELTimeDependentPlan,
    slices: FELBeamSlices,
    destination: str | Path,
    /,
    *,
    seed: Genesis4GaussianSeed | None = None,
    random_seed: int = 1,
    timeout: float = 1800.0,
    maximum_output_bytes: int = 1 << 30,
) -> Genesis4Result:
    """Run pinned Genesis 4 on the translated case and read its power history."""
    if not isinstance(executable, PinnedExecutable):
        raise TypeError("executable must be a PinnedExecutable.")
    inputs = genesis4_input(plan, slices, seed=seed, random_seed=random_seed)
    artifacts = PinnedFileOutputs(
        str(destination),
        (PinnedFileRequest(_OUTPUT, maximum_output_bytes),),
        maximum_output_bytes,
    )
    run = run_pinned_command(
        executable,
        ("fel.in",),
        inputs=inputs,
        timeout=timeout,
        artifacts=artifacts,
    )
    artifact = run.file_artifact(_OUTPUT)
    with h5py.File(artifact.location, "r") as output:
        positions = np.asarray(output["Lattice/zplot"], dtype=np.float64)
        power = np.asarray(output["Field/power"], dtype=np.float64)
    if power.ndim != 2 or power.shape != (positions.shape[0], slices.slice_count):
        raise ValueError("Genesis output does not match the translated window.")
    spacing = slices.position_spacing
    if spacing is None:
        raise ValueError("The Genesis oracle needs uniformly spaced slices.")
    ordered = power[:, ::-1]
    light = float(plan.core.lattice.scale.speed_of_light)
    return Genesis4Result(
        positions=jnp.asarray(positions),
        power=jnp.asarray(ordered),
        pulse_energy=jnp.asarray(np.sum(ordered, axis=1) * spacing / light),
        provider_version=executable.version,
        output_sha256=artifact.sha256,
    )


__all__ = [
    "Genesis4GaussianSeed",
    "Genesis4Result",
    "genesis4_input",
    "run_genesis4",
]
